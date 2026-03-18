"""
RabbitMQ Connection
===================
Async connection singleton using aio-pika.

Exchange layout
---------------
nexusrag.parse   direct   routing_key="parse"
nexusrag.embed   direct   routing_key="embed"
nexusrag.caption direct   routing_key="caption"
nexusrag.kg      direct   routing_key=<workspace_id>   ← per-workspace serialisation

All queues are durable so messages survive broker restarts.

Retry strategy
--------------
Messages that fail processing are retried up to MAX_RETRIES times
with exponential backoff (via x-death headers).  After MAX_RETRIES
the message is routed to a dead-letter queue for manual inspection.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Awaitable, Callable

import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractRobustConnection

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Exchange / queue names ──────────────────────────────────────────────────
EXCHANGE_PARSE   = "nexusrag.parse"
EXCHANGE_EMBED   = "nexusrag.embed"
EXCHANGE_CAPTION = "nexusrag.caption"
EXCHANGE_KG      = "nexusrag.kg"

QUEUE_PARSE   = "nexusrag.parse"
QUEUE_EMBED   = "nexusrag.embed"
QUEUE_CAPTION = "nexusrag.caption"
# KG queues are named nexusrag.kg.<workspace_id> and created on-demand

# ── Dead-letter exchange for failed messages ────────────────────────────────
DLX_EXCHANGE = "nexusrag.dlx"
DLQ_QUEUE    = "nexusrag.dead-letter"

# ── Retry settings ──────────────────────────────────────────────────────────
MAX_RETRIES   = 3           # Total attempts = MAX_RETRIES + 1 (first try + retries)
RETRY_DELAYS  = [5, 15, 60] # Seconds between retries (exponential-ish backoff)


# ── Singleton connection ────────────────────────────────────────────────────
_connection: AbstractRobustConnection | None = None
_lock = asyncio.Lock()


async def get_connection() -> AbstractRobustConnection:
    """Return (or create) the shared robust connection to RabbitMQ."""
    global _connection
    async with _lock:
        if _connection is None or _connection.is_closed:
            # Add jitter to reconnect interval to prevent thundering herd
            jitter = random.uniform(0, 3)
            reconnect_interval = 5 + jitter
            logger.info(f"Connecting to RabbitMQ: {settings.RABBITMQ_URL}")
            _connection = await aio_pika.connect_robust(
                settings.RABBITMQ_URL,
                reconnect_interval=reconnect_interval,
            )
            logger.info("RabbitMQ connected")
    return _connection


async def close_connection() -> None:
    global _connection
    if _connection and not _connection.is_closed:
        await _connection.close()
        _connection = None


# ── DLX / DLQ setup ──────────────────────────────────────────────────────────
async def _ensure_dlx(channel: aio_pika.Channel) -> None:
    """Declare the dead-letter exchange and queue (idempotent)."""
    dlx = await channel.declare_exchange(
        DLX_EXCHANGE, ExchangeType.FANOUT, durable=True
    )
    dlq = await channel.declare_queue(DLQ_QUEUE, durable=True)
    await dlq.bind(dlx)


# ── Publisher helpers ───────────────────────────────────────────────────────
async def publish(exchange_name: str, routing_key: str, payload: dict) -> None:
    """Publish a JSON message to *exchange_name* with *routing_key*."""
    conn = await get_connection()
    async with conn.channel() as channel:
        exchange = await channel.declare_exchange(
            exchange_name, ExchangeType.DIRECT, durable=True
        )
        body = json.dumps(payload).encode()
        await exchange.publish(
            Message(body, delivery_mode=DeliveryMode.PERSISTENT),
            routing_key=routing_key,
        )
        logger.debug(f"Published to {exchange_name}/{routing_key}: {payload}")


async def _republish_with_delay(
    channel: aio_pika.Channel,
    exchange_name: str,
    routing_key: str,
    body: bytes,
    retry_count: int,
) -> None:
    """Republish a message with updated retry header after a delay."""
    delay = RETRY_DELAYS[min(retry_count, len(RETRY_DELAYS) - 1)]
    # Add jitter ±30% to prevent synchronized retries
    delay = delay * (0.7 + random.random() * 0.6)

    logger.info(
        f"Scheduling retry {retry_count + 1}/{MAX_RETRIES} "
        f"for {exchange_name}/{routing_key} in {delay:.1f}s"
    )
    await asyncio.sleep(delay)

    exchange = await channel.declare_exchange(
        exchange_name, ExchangeType.DIRECT, durable=True
    )
    headers = {"x-retry-count": retry_count + 1}
    await exchange.publish(
        Message(
            body,
            delivery_mode=DeliveryMode.PERSISTENT,
            headers=headers,
        ),
        routing_key=routing_key,
    )
    logger.info(
        f"Retried message to {exchange_name}/{routing_key} "
        f"(attempt {retry_count + 1}/{MAX_RETRIES})"
    )


# ── Consumer helper ─────────────────────────────────────────────────────────
MessageHandler = Callable[[dict], Awaitable[None]]


async def consume(
    exchange_name: str,
    queue_name: str,
    routing_key: str,
    handler: MessageHandler,
    prefetch_count: int = 1,
) -> None:
    """
    Start consuming messages from *queue_name*.

    Messages are ack'd after *handler* returns successfully.
    On failure, messages are retried up to MAX_RETRIES times with
    exponential backoff.  After exhausting retries the message is
    sent to the dead-letter queue for manual inspection.

    The coroutine runs indefinitely — use asyncio.create_task() to run
    it in the background.
    """
    conn = await get_connection()
    channel = await conn.channel()
    await channel.set_qos(prefetch_count=prefetch_count)

    # Ensure DLX exists
    await _ensure_dlx(channel)

    exchange = await channel.declare_exchange(
        exchange_name, ExchangeType.DIRECT, durable=True
    )

    # Try to declare queue with DLX arguments.
    # If the queue already exists without DLX (from an older version),
    # RabbitMQ returns PRECONDITION_FAILED which closes the channel.
    # In that case, open a new channel and declare without DLX args.
    dlx_ok = True
    try:
        queue = await channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": DLX_EXCHANGE,
            },
        )
    except Exception:
        dlx_ok = False
        logger.warning(
            f"Queue {queue_name} exists with different arguments. "
            f"Using existing queue (DLX will not apply until queue is recreated). "
            f"To fix: delete the queue in RabbitMQ management UI and restart."
        )
        # The channel was likely closed by the error — open a fresh one
        channel = await conn.channel()
        await channel.set_qos(prefetch_count=prefetch_count)
        exchange = await channel.declare_exchange(
            exchange_name, ExchangeType.DIRECT, durable=True
        )
        queue = await channel.declare_queue(queue_name, durable=True)

    await queue.bind(exchange, routing_key=routing_key)

    logger.info(f"Consuming {exchange_name}/{routing_key} → {queue_name}")

    async with queue.iterator() as messages:
        async for message in messages:
            async with message.process(requeue=False):
                # Read retry count from headers
                headers = message.headers or {}
                retry_count = int(headers.get("x-retry-count", 0))

                try:
                    payload = json.loads(message.body)
                    await handler(payload)
                except Exception as e:
                    if retry_count < MAX_RETRIES:
                        logger.warning(
                            f"Handler error on {queue_name} (attempt {retry_count + 1}/{MAX_RETRIES + 1}): {e}"
                        )
                        # Schedule retry in background (don't block consumer)
                        asyncio.create_task(
                            _republish_with_delay(
                                channel, exchange_name, routing_key,
                                message.body, retry_count,
                            )
                        )
                        # Don't re-raise — message is ack'd, retry is queued
                    else:
                        logger.error(
                            f"Handler error on {queue_name} after {MAX_RETRIES + 1} attempts: {e}",
                            exc_info=True,
                        )
                        # Let the message go to DLX (nack without requeue)
                        raise


async def consume_kg(workspace_id: int, handler: MessageHandler) -> None:
    """
    Consume KG messages for a specific workspace.
    prefetch_count=1 ensures sequential processing within the workspace.
    """
    queue_name = f"nexusrag.kg.{workspace_id}"
    routing_key = str(workspace_id)
    await consume(
        EXCHANGE_KG, queue_name, routing_key,
        handler, prefetch_count=1,
    )

"""
RabbitMQ Connection
===================
Async connection singleton using aio-pika.

Exchange layout
---------------
hrag.parse   direct   routing_key="parse"
hrag.embed   direct   routing_key="embed"
hrag.caption direct   routing_key="caption"
hrag.kg      direct   routing_key=<workspace_id>   ← per-workspace serialisation

All queues are durable so messages survive broker restarts.

Retry strategy (RabbitMQ-native)
---------------------------------
Messages that fail processing are published to a *delay queue* with a
per-message TTL.  When the TTL expires the message is dead-lettered back
to its original exchange / routing-key for re-delivery.

This approach is crash-safe: the retry message is persisted on the broker
instead of being held in worker memory via asyncio.sleep().

After MAX_RETRIES the message is routed to a dead-letter queue for manual
inspection.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Awaitable, Callable

import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractRobustConnection

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Exchange / queue names ──────────────────────────────────────────────────
EXCHANGE_PARSE   = "hrag.parse"
EXCHANGE_EMBED   = "hrag.embed"
EXCHANGE_CAPTION = "hrag.caption"
EXCHANGE_KG      = "hrag.kg"

QUEUE_PARSE   = "hrag.parse"
QUEUE_EMBED   = "hrag.embed"
QUEUE_CAPTION = "hrag.caption"
# KG queues are named hrag.kg.<workspace_id> and created on-demand
QUEUE_KG_PREFIX = "hrag.kg"

# ── Dead-letter exchange for failed messages ────────────────────────────────
DLX_EXCHANGE = "hrag.dlx"
DLQ_QUEUE    = "hrag.dead-letter"

# ── Retry settings ──────────────────────────────────────────────────────────
MAX_RETRIES   = 3              # Total attempts = MAX_RETRIES + 1 (first try + retries)
RETRY_DELAYS  = [5, 15, 60]   # Seconds — mapped to per-message TTL

# Retry exchange: delay queues dead-letter back to this exchange which
# fans out to the original exchange via x-dead-letter-exchange on each
# delay queue.  We use a single HEADERS exchange so that we can route
# each message back to its original exchange/routing_key using headers.
RETRY_EXCHANGE = "hrag.retry"
_RETRY_QUEUE_NAMES = [
    f"hrag.retry.{d}s" for d in RETRY_DELAYS
]


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


# ── Retry infrastructure ───────────────────────────────────────────────────
async def _ensure_retry_queues(channel: aio_pika.Channel) -> None:
    """
    Create the retry exchange and per-delay queues (idempotent).

    Each delay queue has a TTL matching its delay.  When a message's TTL
    expires it is dead-lettered to the *retry exchange* (DIRECT) which
    re-routes it back to the original exchange using the
    ``x-original-routing-key`` header stored on the message.

    Flow:
        handler fail → publish to hrag.retry.Xs (with TTL = X s)
            → TTL expires → DLX to hrag.retry (DIRECT)
            → route back to original exchange/routing_key
    """
    # Retry exchange — messages land here after TTL expires
    await channel.declare_exchange(
        RETRY_EXCHANGE, ExchangeType.DIRECT, durable=True,
    )

    for delay_sec, q_name in zip(RETRY_DELAYS, _RETRY_QUEUE_NAMES):
        try:
            queue = await channel.declare_queue(
                q_name,
                durable=True,
                arguments={
                    "x-message-ttl": delay_sec * 1000,          # ms
                    "x-dead-letter-exchange": RETRY_EXCHANGE,   # after TTL → retry exchange
                },
            )
            # Bind so publisher can publish directly by queue name
            # (no routing key needed — we publish directly to the queue)
        except Exception:
            # Queue already exists with different args — use as-is
            logger.debug(f"Retry queue {q_name} already exists (using existing)")


async def _publish_to_retry_queue(
    channel: aio_pika.Channel,
    exchange_name: str,
    routing_key: str,
    body: bytes,
    retry_count: int,
) -> None:
    """
    Publish a failed message to the appropriate delay queue.

    The message carries headers that let the retry exchange route it
    back to the correct original exchange and routing key when TTL expires.
    """
    delay_idx = min(retry_count, len(RETRY_DELAYS) - 1)
    delay_sec = RETRY_DELAYS[delay_idx]
    q_name = _RETRY_QUEUE_NAMES[delay_idx]

    # Add jitter ±30% as per-message TTL override
    jittered_ms = int(delay_sec * 1000 * (0.7 + random.random() * 0.6))

    headers = {
        "x-retry-count": retry_count + 1,
        "x-original-exchange": exchange_name,
        "x-original-routing-key": routing_key,
    }

    logger.info(
        f"Scheduling retry {retry_count + 1}/{MAX_RETRIES} "
        f"for {exchange_name}/{routing_key} via {q_name} (TTL={jittered_ms}ms)"
    )

    # Publish directly to the default exchange with routing_key = queue name
    # This puts the message into the delay queue.
    await channel.default_exchange.publish(
        Message(
            body,
            delivery_mode=DeliveryMode.PERSISTENT,
            headers=headers,
            expiration=jittered_ms,   # per-message TTL override (ms)
        ),
        routing_key=q_name,
    )


# ── Retry consumer —————————————————————————————————————————————————————————
async def _start_retry_consumer(channel: aio_pika.Channel) -> None:
    """
    Consume messages from the retry exchange after they expire from
    delay queues and re-publish them to their original exchange/routing_key.
    """
    retry_exchange = await channel.declare_exchange(
        RETRY_EXCHANGE, ExchangeType.DIRECT, durable=True,
    )

    # We need a queue bound to the retry exchange to catch expired messages.
    # Since delay queues DLX to RETRY_EXCHANGE with routing_key = original
    # queue name, we create a single catch-all queue.
    retry_requeue_name = "hrag.retry.requeue"
    try:
        retry_queue = await channel.declare_queue(
            retry_requeue_name, durable=True,
        )
    except Exception:
        retry_queue = await channel.declare_queue(
            retry_requeue_name, durable=True,
        )

    # Bind to all known delay queue names as routing keys
    # (when DLX fires, routing key = original queue name from the delay queue)
    for q_name in _RETRY_QUEUE_NAMES:
        await retry_queue.bind(retry_exchange, routing_key=q_name)

    async with retry_queue.iterator() as messages:
        async for message in messages:
            async with message.process():
                headers = message.headers or {}
                orig_exchange = headers.get("x-original-exchange", "")
                orig_routing_key = headers.get("x-original-routing-key", "")

                if not orig_exchange:
                    logger.warning(
                        "[retry] Message missing x-original-exchange header — "
                        "sending to DLQ"
                    )
                    continue

                # Re-publish to original exchange
                exchange = await channel.declare_exchange(
                    orig_exchange, ExchangeType.DIRECT, durable=True,
                )
                await exchange.publish(
                    Message(
                        message.body,
                        delivery_mode=DeliveryMode.PERSISTENT,
                        headers={k: v for k, v in headers.items()},
                    ),
                    routing_key=orig_routing_key,
                )
                logger.info(
                    f"[retry] Re-published to {orig_exchange}/{orig_routing_key} "
                    f"(attempt {headers.get('x-retry-count', '?')})"
                )


# ── Publisher helpers ───────────────────────────────────────────────────────
async def publish(exchange_name: str, routing_key: str, payload: dict) -> None:
    """Publish a JSON message to *exchange_name* with *routing_key*."""
    conn = await get_connection()
    async with conn.channel() as channel:
        exchange = await channel.declare_exchange(
            exchange_name, ExchangeType.DIRECT, durable=True
        )
        # Use default=str to handle UUID serialization
        body = json.dumps(payload, default=str).encode()
        await exchange.publish(
            Message(body, delivery_mode=DeliveryMode.PERSISTENT),
            routing_key=routing_key,
        )
        logger.debug(f"Published to {exchange_name}/{routing_key}: {payload}")


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
    On failure, messages are retried up to MAX_RETRIES times using
    RabbitMQ-native delay queues (crash-safe).  After exhausting retries
    the message is sent to the dead-letter queue for manual inspection.

    The coroutine runs indefinitely — use asyncio.create_task() to run
    it in the background.
    """
    from app.workers.metrics import worker_metrics

    conn = await get_connection()
    channel = await conn.channel()
    await channel.set_qos(prefetch_count=prefetch_count)

    # Ensure DLX and retry queues exist
    await _ensure_dlx(channel)
    await _ensure_retry_queues(channel)

    exchange = await channel.declare_exchange(
        exchange_name, ExchangeType.DIRECT, durable=True
    )

    # Try to declare queue with DLX arguments.
    # If the queue already exists without DLX (from an older version),
    # RabbitMQ returns PRECONDITION_FAILED which closes the channel.
    # In that case, open a new channel and declare without DLX args.
    try:
        queue = await channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": DLX_EXCHANGE,
            },
        )
    except Exception:
        logger.warning(
            f"Queue {queue_name} exists with different arguments. "
            f"Using existing queue (DLX will not apply until queue is recreated). "
            f"To fix: delete the queue in RabbitMQ management UI and restart."
        )
        # The channel was likely closed by the error — open a fresh one
        channel = await conn.channel()
        await channel.set_qos(prefetch_count=prefetch_count)
        await _ensure_retry_queues(channel)
        exchange = await channel.declare_exchange(
            exchange_name, ExchangeType.DIRECT, durable=True
        )
        queue = await channel.declare_queue(queue_name, durable=True)

    await queue.bind(exchange, routing_key=routing_key)

    # Start the retry re-publisher in background
    asyncio.create_task(_start_retry_consumer(channel))

    logger.info(f"Consuming {exchange_name}/{routing_key} → {queue_name}")

    async with queue.iterator() as messages:
        async for message in messages:
            async with message.process(requeue=False):
                # Read retry count from headers
                headers = message.headers or {}
                retry_count = int(headers.get("x-retry-count", 0))

                start_time = time.monotonic()
                try:
                    payload = json.loads(message.body)
                    await handler(payload)
                    # Track success
                    elapsed = time.monotonic() - start_time
                    worker_metrics.record_success(queue_name, elapsed)
                except Exception as e:
                    elapsed = time.monotonic() - start_time
                    worker_metrics.record_failure(queue_name, elapsed)

                    if retry_count < MAX_RETRIES:
                        logger.warning(
                            f"Handler error on {queue_name} "
                            f"(attempt {retry_count + 1}/{MAX_RETRIES + 1}): {e}"
                        )
                        # Publish to delay queue (crash-safe — message persisted on broker)
                        try:
                            await _publish_to_retry_queue(
                                channel, exchange_name, routing_key,
                                message.body, retry_count,
                            )
                        except Exception as retry_err:
                            logger.error(
                                f"Failed to publish retry for {queue_name}: {retry_err}"
                            )
                        # Don't re-raise — message is ack'd, retry is queued on broker
                    else:
                        logger.error(
                            f"Handler error on {queue_name} after "
                            f"{MAX_RETRIES + 1} attempts: {e}",
                            exc_info=True,
                        )
                        # Let the message go to DLX (nack without requeue)
                        raise


async def consume_kg(workspace_id: int, handler: MessageHandler) -> None:
    """
    Consume KG messages for a specific workspace.
    prefetch_count=1 ensures sequential processing within the workspace.
    """
    queue_name = f"hrag.kg.{workspace_id}"
    routing_key = str(workspace_id)
    await consume(
        EXCHANGE_KG, queue_name, routing_key,
        handler, prefetch_count=settings.WORKER_PREFETCH_KG,
    )

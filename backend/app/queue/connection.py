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
"""
from __future__ import annotations

import asyncio
import json
import logging
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


# ── Singleton connection ────────────────────────────────────────────────────
_connection: AbstractRobustConnection | None = None
_lock = asyncio.Lock()


async def get_connection() -> AbstractRobustConnection:
    """Return (or create) the shared robust connection to RabbitMQ."""
    global _connection
    async with _lock:
        if _connection is None or _connection.is_closed:
            logger.info(f"Connecting to RabbitMQ: {settings.RABBITMQ_URL}")
            _connection = await aio_pika.connect_robust(
                settings.RABBITMQ_URL,
                reconnect_interval=5,
            )
            logger.info("RabbitMQ connected")
    return _connection


async def close_connection() -> None:
    global _connection
    if _connection and not _connection.is_closed:
        await _connection.close()
        _connection = None


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
    On unhandled exception the message is nack'd without requeue
    (goes to dead-letter or is dropped) to avoid poison-pill loops.
    The coroutine runs indefinitely — use asyncio.create_task() to run
    it in the background.
    """
    conn = await get_connection()
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
                try:
                    payload = json.loads(message.body)
                    await handler(payload)
                except Exception as e:
                    logger.error(
                        f"Handler error on {queue_name}: {e}", exc_info=True
                    )
                    # message.process(requeue=False) nack's automatically on exception
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

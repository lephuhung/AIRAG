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
import uuid
from typing import Awaitable, Callable

import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractRobustConnection

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Exchange / queue names ──────────────────────────────────────────────────
EXCHANGE_PARSE = "hrag.parse"
EXCHANGE_EMBED = "hrag.embed"
EXCHANGE_CAPTION = "hrag.caption"
EXCHANGE_KG = "hrag.kg"

QUEUE_PARSE = "hrag.parse"
QUEUE_EMBED = "hrag.embed"
QUEUE_CAPTION = "hrag.caption"
# KG queues are named hrag.kg.<workspace_id> and created on-demand
QUEUE_KG_PREFIX = "hrag.kg"

# ── Dead-letter exchange for failed messages ────────────────────────────────
DLX_EXCHANGE = "hrag.dlx"
DLQ_QUEUE = "hrag.dead-letter"

# ── Retry settings ──────────────────────────────────────────────────────────
MAX_RETRIES = 3  # Total attempts = MAX_RETRIES + 1 (first try + retries)
RETRY_DELAYS = [5, 15, 60]  # Seconds — mapped to per-message TTL

# Retry exchange: delay queues dead-letter back to this exchange which
# fans out to the original exchange via x-dead-letter-exchange on each
# delay queue.  We use a single HEADERS exchange so that we can route
# each message back to its original exchange/routing_key using headers.
RETRY_EXCHANGE = "hrag.retry"
_RETRY_QUEUE_NAMES = [f"hrag.retry.{d}s" for d in RETRY_DELAYS]


# ── Singleton connection ────────────────────────────────────────────────────
_connection: AbstractRobustConnection | None = None
_lock = asyncio.Lock()
_retry_consumer_started: bool = False


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
                heartbeat=120,
            )
            logger.info("RabbitMQ connected")
            # Update health server state so /ready returns true
            # Both rabbitmq and ready must be True for /ready to return 200
            try:
                from app.workers.health_server import update_worker_state
                update_worker_state(rabbitmq=True, ready=True)
            except Exception:
                pass  # non-fatal if health server not yet initialized
    return _connection


async def close_connection() -> None:
    global _connection
    if _connection and not _connection.is_closed:
        await _connection.close()
        _connection = None
        try:
            from app.workers.health_server import update_worker_state
            update_worker_state(rabbitmq=False, ready=False)
        except Exception:
            pass  # non-fatal


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
    Create per-delay queues (idempotent).

    Each delay queue has a TTL matching its delay. When a message's TTL
    expires it is dead-lettered directly back to the original KG exchange
    (EXCHANGE_KG) using the original routing key preserved in the message
    headers. This bypasses the need for a separate retry consumer.

    Flow:
        handler fail → publish to hrag.retry.Xs (with TTL = X s, x-original-routing-key header)
            → TTL expires → DLX to hrag.kg (EXCHANGE_KG) with original routing key
            → routes back to hrag.kg.{workspace_id} queue automatically
    """
    # Declare the KG exchange (used as DLX so messages return to right queue on retry)
    await channel.declare_exchange(
        EXCHANGE_KG,
        ExchangeType.DIRECT,
        durable=True,
    )
    # Keep RETRY_EXCHANGE for the retry requeue consumer path (MAX_RETRIES exhausted)
    await channel.declare_exchange(
        RETRY_EXCHANGE,
        ExchangeType.DIRECT,
        durable=True,
    )

    for delay_sec, q_name in zip(RETRY_DELAYS, _RETRY_QUEUE_NAMES):
        try:
            queue = await channel.declare_queue(
                q_name,
                durable=True,
                arguments={
                    "x-message-ttl": delay_sec * 1000,  # ms
                    # DLX directly to EXCHANGE_KG so expired message routes back
                    # to hrag.kg.{workspace_id} using the original routing key
                    "x-dead-letter-exchange": EXCHANGE_KG,
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
            expiration=jittered_ms,  # per-message TTL override (ms)
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
        RETRY_EXCHANGE,
        ExchangeType.DIRECT,
        durable=True,
    )

    # We need a queue bound to the retry exchange to catch expired messages.
    # Since delay queues DLX to RETRY_EXCHANGE with routing_key = original
    # queue name, we create a single catch-all queue.
    retry_requeue_name = "hrag.retry.requeue"
    try:
        retry_queue = await channel.declare_queue(
            retry_requeue_name,
            durable=True,
        )
    except Exception:
        retry_queue = await channel.declare_queue(
            retry_requeue_name,
            durable=True,
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
                    orig_exchange,
                    ExchangeType.DIRECT,
                    durable=True,
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


def _get_timeout_for_queue(queue_name: str) -> int:
    """Determine handler timeout based on queue type."""
    if "parse" in queue_name:
        return settings.WORKER_PARSE_TIMEOUT
    if "embed" in queue_name:
        return settings.WORKER_EMBED_TIMEOUT
    if "caption" in queue_name:
        return settings.WORKER_CAPTION_TIMEOUT
    if "kg" in queue_name:
        return settings.WORKER_KG_TIMEOUT
    return 60  # default


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

    handler_timeout = _get_timeout_for_queue(queue_name)

    conn = await get_connection()
    channel = await conn.channel()
    await channel.set_qos(prefetch_count=prefetch_count)

    # Ensure DLX and retry queues exist
    await _ensure_dlx(channel)
    await _ensure_retry_queues(channel)

    exchange = await channel.declare_exchange(
        exchange_name, ExchangeType.DIRECT, durable=True
    )

    queue_declaration_failed = False

    try:
        queue = await channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": DLX_EXCHANGE,
                "x-max-length": 10000,
                "x-overflow": "reject-publish",
            },
        )
    except Exception:
        logger.warning(
            f"Queue {queue_name} exists with different arguments. "
            f"Using existing queue (DLX will not apply until queue is recreated). "
            f"To fix: delete the queue in RabbitMQ management UI and restart."
        )
        queue_declaration_failed = True

    if queue_declaration_failed:
        channel = await conn.channel()
        await channel.set_qos(prefetch_count=prefetch_count)
        await _ensure_dlx(channel)
        await _ensure_retry_queues(channel)
        exchange = await channel.declare_exchange(
            exchange_name, ExchangeType.DIRECT, durable=True
        )
        queue = await channel.declare_queue(queue_name, durable=True)

    await queue.bind(exchange, routing_key=routing_key)

    # Start retry consumer only once globally (not per queue)
    global _retry_consumer_started
    if not _retry_consumer_started:
        _retry_consumer_started = True
        retry_channel = await conn.channel()
        asyncio.create_task(_start_retry_consumer(retry_channel))

    logger.info(f"Consuming {exchange_name}/{routing_key} → {queue_name}")

    restart_count = 0
    while True:
        try:
            logger.info(
                f"queue_iterator starting",
                extra={"queue_name": queue_name},
            )
            async with queue.iterator() as messages:
                async for message in messages:
                    async with message.process(requeue=False):
                        headers = message.headers or {}
                        retry_count = int(headers.get("x-retry-count", 0))
                        document_id = "unknown"
                        minio_key = "unknown"

                        start_time = time.monotonic()
                        try:
                            payload = json.loads(message.body)
                            document_id = str(payload.get("document_id", "unknown"))
                            minio_key = payload.get("minio_key", "unknown")

                            async with asyncio.timeout(handler_timeout):
                                await handler(payload)

                            elapsed = time.monotonic() - start_time
                            worker_metrics.record_success(queue_name, elapsed)
                        except asyncio.TimeoutError:
                            elapsed = time.monotonic() - start_time
                            worker_metrics.record_failure(queue_name, elapsed)
                            logger.warning(
                                f"{queue_name} handler timed out after {handler_timeout}s",
                                extra={
                                    "queue_name": queue_name,
                                    "document_id": document_id,
                                    "minio_key": minio_key,
                                    "retry_count": retry_count,
                                    "timeout": handler_timeout,
                                },
                            )
                            # Timeout: rollback document status to PENDING and requeue for retry
                            # Only do this for parse queue (other queues don't have status to rollback)
                            if "parse" in queue_name and document_id != "unknown":
                                try:
                                    from app.core.database import async_session_maker
                                    from sqlalchemy import select
                                    from app.models.document import Document, DocumentStatus

                                    async with async_session_maker() as rollback_db:
                                        result = await rollback_db.execute(
                                            select(Document).where(
                                                Document.id == uuid.UUID(document_id)
                                            )
                                        )
                                        doc = result.scalar_one_or_none()
                                        if doc and doc.status not in (
                                            DocumentStatus.PENDING,
                                            DocumentStatus.INDEXED,
                                            DocumentStatus.FAILED,
                                        ):
                                            doc.status = DocumentStatus.PENDING
                                            doc.error_message = f"timeout_retry: handler timed out after {elapsed:.1f}s"
                                            await rollback_db.commit()
                                            logger.info(
                                                f"[timeout_rollback] doc={document_id} "
                                                f"status reverted to PENDING"
                                            )
                                except Exception as rollback_err:
                                    logger.warning(
                                        f"[timeout_rollback] failed for doc={document_id}: {rollback_err}"
                                    )
                            # For KG worker: reset kg_done=False so retry can process again
                            if "kg" in queue_name and document_id != "unknown":
                                try:
                                    from app.core.database import async_session_maker
                                    from sqlalchemy import select
                                    from app.models.document import Document, DocumentStatus

                                    async with async_session_maker() as rollback_db:
                                        result = await rollback_db.execute(
                                            select(Document).where(
                                                Document.id == uuid.UUID(document_id)
                                            )
                                        )
                                        doc = result.scalar_one_or_none()
                                        if doc:
                                            doc.kg_done = False
                                            doc.status = DocumentStatus.BUILDING_KG
                                            doc.error_message = f"timeout_retry: KG handler timed out after {elapsed:.1f}s — will retry"
                                            await rollback_db.commit()
                                            logger.info(
                                                f"[timeout_rollback] doc={document_id} "
                                                f"kg_done reset to False for retry"
                                            )
                                except Exception as rollback_err:
                                    logger.warning(
                                        f"[timeout_rollback] KG failed for doc={document_id}: {rollback_err}"
                                    )
                                try:
                                    from app.core.database import async_session_maker
                                    from sqlalchemy import select
                                    from app.models.document import Document, DocumentStatus

                                    async with async_session_maker() as rollback_db:
                                        result = await rollback_db.execute(
                                            select(Document).where(
                                                Document.id == uuid.UUID(document_id)
                                            )
                                        )
                                        doc = result.scalar_one_or_none()
                                        if doc and doc.status not in (
                                            DocumentStatus.PENDING,
                                            DocumentStatus.INDEXED,
                                            DocumentStatus.FAILED,
                                        ):
                                            doc.status = DocumentStatus.PENDING
                                            doc.error_message = f"timeout_retry: handler timed out after {elapsed:.1f}s"
                                            await rollback_db.commit()
                                            logger.info(
                                                f"[timeout_rollback] doc={document_id} "
                                                f"status reverted to PENDING"
                                            )
                                except Exception as rollback_err:
                                    logger.warning(
                                        f"[timeout_rollback] failed for doc={document_id}: {rollback_err}"
                                    )
                            # Requeue the message for retry (with existing retry count, no new retry)
                            if retry_count < MAX_RETRIES:
                                try:
                                    await _publish_to_retry_queue(
                                        channel,
                                        exchange_name,
                                        routing_key,
                                        message.body,
                                        retry_count,
                                    )
                                except Exception as retry_err:
                                    logger.error(
                                        f"Failed to publish retry for {queue_name}: {retry_err}",
                                        extra={
                                            "queue_name": queue_name,
                                            "document_id": document_id,
                                            "minio_key": minio_key,
                                            "retry_count": retry_count,
                                            "error": str(retry_err),
                                        },
                                    )
                            # Ack the message; wrap in try-except because the channel may be
                            # in a broken state after CancelledError from asyncio.timeout.
                            try:
                                await message.ack()
                            except Exception as ack_err:
                                logger.warning(f"[{queue_name}] message.ack() failed: {ack_err} — ignoring")
                        except Exception as e:
                            elapsed = time.monotonic() - start_time
                            worker_metrics.record_failure(queue_name, elapsed)
                            if retry_count < MAX_RETRIES:
                                logger.warning(
                                    f"Handler error on {queue_name} "
                                    f"(attempt {retry_count + 1}/{MAX_RETRIES + 1}): {e}",
                                    extra={
                                        "queue_name": queue_name,
                                        "document_id": document_id,
                                        "minio_key": minio_key,
                                        "retry_count": retry_count,
                                        "error": str(e),
                                        "error_type": type(e).__name__,
                                    },
                                )
                                try:
                                    await _publish_to_retry_queue(
                                        channel,
                                        exchange_name,
                                        routing_key,
                                        message.body,
                                        retry_count,
                                    )
                                except Exception as retry_err:
                                    logger.error(
                                        f"Failed to publish retry for {queue_name}: {retry_err}",
                                        extra={
                                            "queue_name": queue_name,
                                            "document_id": document_id,
                                            "minio_key": minio_key,
                                            "retry_count": retry_count,
                                            "error": str(retry_err),
                                        },
                                    )
                                await message.ack()
                            else:
                                # MAX_RETRIES exhausted — send to DLQ, not discard
                                logger.error(
                                    f"[{queue_name}] Max retries {MAX_RETRIES} reached — "
                                    f"sending to DLQ",
                                    extra={
                                        "queue_name": queue_name,
                                        "document_id": document_id,
                                        "minio_key": minio_key,
                                        "retry_count": retry_count,
                                        "error": str(e),
                                        "error_type": type(e).__name__,
                                    },
                                )
                                try:
                                    await message.reject(requeue=False)  # trigger x-dead-letter-exchange → DLQ
                                    logger.info(f"[{queue_name}] Message sent to DLQ")
                                except Exception as dlq_err:
                                    logger.error(f"[{queue_name}] Failed to send to DLQ: {dlq_err}")
                                    await message.ack()  # fallback: ack to prevent infinite loop
            logger.warning(
                f"queue_iterator exhausted unexpectedly — reconnecting",
                extra={"queue_name": queue_name},
            )
        except Exception as it_error:
            restart_count += 1
            logger.error(
                f"queue_iterator error",
                extra={
                    "queue_name": queue_name,
                    "error": str(it_error),
                    "error_type": type(it_error).__name__,
                    "restart_count": restart_count,
                },
            )
            if restart_count >= 10:
                logger.critical(
                    f"queue_iterator max restarts reached, abandoning",
                    extra={"queue_name": queue_name, "restart_count": restart_count},
                )
                break
            await asyncio.sleep(5)


async def consume_kg(workspace_id: int, handler: MessageHandler) -> None:
    """
    Consume KG messages for a specific workspace.
    prefetch_count=1 ensures sequential processing within the workspace.
    """
    queue_name = f"hrag.kg.{workspace_id}"
    routing_key = str(workspace_id)
    await consume(
        EXCHANGE_KG,
        queue_name,
        routing_key,
        handler,
        prefetch_count=settings.WORKER_PREFETCH_KG,
    )

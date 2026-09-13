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
EXCHANGE_MEMORY = "hrag.memory"

QUEUE_PARSE = "hrag.parse"
QUEUE_EMBED = "hrag.embed"
QUEUE_CAPTION = "hrag.caption"
QUEUE_MEMORY = "hrag.memory"
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


# ── Revision-stage reporting (P1 Task 3) ─────────────────────────────────
# Each pipeline queue owns exactly one revision stage. The retry path below
# reports stage transitions for the FAILED message's revision only, so a
# retry/redelivery can never touch another generation's rows:
#
# - before a retryable message is requeued: running → pending for that
#   message revision/stage (``mark_stage_retry_pending`` preserves
#   ``attempt_count``; the next worker claim (``claim_stage_running``)
#   bumps it);
# - when retries are exhausted: that stage → failed (terminal) AND that
#   revision is terminalized (``mark_failed``) in the same transaction.
#
# Terminal stages (completed / skipped / failed) and unknown revisions are
# left untouched — the note helpers are best-effort and never raise.
_STAGE_BY_QUEUE_SUBSTRING: tuple[tuple[str, str], ...] = (
    ("parse", "parse"),
    ("embed", "embed"),
    ("caption", "caption"),
    ("kg", "kg"),
)


def stage_for_queue(queue_name: str, exchange_name: str = "") -> str | None:
    """Map a pipeline queue/exchange to its revision-owned stage.

    Returns ``None`` for non-pipeline queues (e.g. memory), which own no
    revision stage and are left untouched by the retry path.
    """
    haystack = f"{exchange_name} {queue_name}".lower()
    for substring, stage in _STAGE_BY_QUEUE_SUBSTRING:
        if substring in haystack:
            return stage
    return None


async def _note_stage(
    revision_id,
    stage: str,
    *,
    kind: str,
    failure_class: str | None = None,
    db=None,
) -> bool:
    """Apply one queue-owned stage edge; never raises.

    ``kind="retry"`` moves running → pending before a retry requeue;
    ``kind="exhausted"`` atomically moves pending/running → failed AND
    terminalizes that revision (``mark_failed``) in the same DB
    transaction, so an exhausted stage can never strand a live revision
    with a pending row and no redelivery. Returns True when the note
    applied without infrastructure error (including idempotent
    convergence on an already-failed stage), False when the row was left
    untouched (terminal non-failed state, unknown revision/stage) or the
    infrastructure failed. When ``db`` is
    None a private session is opened and committed; otherwise the
    caller's transaction owns the write (flushed, never committed).
    """
    from app.services.agents.v2.persistence.document_revisions import (
        DocumentRevisionsRepository,
        InvalidStageTransition,
        RevisionNotFailable,
        UnknownRevisionStage,
    )

    async def _apply(session) -> None:
        repo = DocumentRevisionsRepository(session)
        if kind == "retry":
            await repo.mark_stage_retry_pending(revision_id, stage)
            return
        await repo.mark_stage_failed(
            revision_id, stage, failure_class=failure_class
        )
        try:
            await repo.mark_failed(
                revision_id,
                stage=stage,
                error_class=failure_class or "exhausted",
            )
        except RevisionNotFailable as exc:
            # Revision already terminal (failed/published/abandoned):
            # the stage edge above still stands; only the revision
            # transition is a no-op.
            logger.info(
                f"[stage] exhausted note left rev={revision_id} "
                f"revision untouched: {exc}"
            )

    try:
        if db is None:
            from app.core.database import async_session_maker

            try:
                async with async_session_maker() as session:
                    await _apply(session)
                    await session.commit()
            except (InvalidStageTransition, UnknownRevisionStage) as e:
                logger.info(
                    f"[stage] {kind} note left rev={revision_id} "
                    f"stage={stage} untouched: {e}"
                )
                return False
        else:
            await _apply(db)
        return True
    except (InvalidStageTransition, UnknownRevisionStage) as e:
        logger.info(
            f"[stage] {kind} note left rev={revision_id} "
            f"stage={stage} untouched: {e}"
        )
        return False
    except Exception as e:  # noqa: BLE001 - retry path must never break
        logger.warning(
            f"[stage] could not note {kind} for rev={revision_id} "
            f"stage={stage}: {e}"
        )
        return False


async def note_stage_retry_pending(revision_id, stage: str, *, db=None) -> bool:
    """Best-effort running → pending before a retry requeue (never raises)."""
    return await _note_stage(revision_id, stage, kind="retry", db=db)


async def note_stage_exhausted(
    revision_id, stage: str, *, failure_class: str | None = None, db=None
) -> bool:
    """Best-effort pending/running → failed on exhausted retries (never raises)."""
    return await _note_stage(
        revision_id, stage, kind="exhausted", failure_class=failure_class, db=db
    )


async def _note_failure_stage(
    *,
    queue_name: str,
    exchange_name: str,
    revision_id,
    retry_count: int,
    failure_class: str,
    force_exhausted: bool = False,
    db=None,
) -> str | None:
    """Map one failed delivery to its stage and apply the queue-owned note.

    The stage comes ONLY from ``stage_for_queue`` — never a hardcoded
    literal — so a wrong-stage mutation is caught by the handler-level
    tests. ``retry_count < MAX_RETRIES`` notes running → pending before
    the retry requeue; exhaustion (or ``force_exhausted`` when the retry
    publish itself failed and no redelivery will arrive) fails the stage
    and terminalizes the revision atomically. Non-pipeline queues and
    stageless messages resolve to ``None`` and are left untouched.
    Never raises. Returns the resolved stage (or ``None``).
    """
    if revision_id is None:
        return None
    stage = stage_for_queue(queue_name, exchange_name)
    if stage is None:
        return None
    if not force_exhausted and retry_count < MAX_RETRIES:
        await note_stage_retry_pending(revision_id, stage, db=db)
    else:
        await note_stage_exhausted(
            revision_id, stage, failure_class=failure_class, db=db
        )
    return stage


class AbandonedStageRecoveryError(RuntimeError):
    """Pre-handler stage recovery hit an infrastructure failure.

    Raised BEFORE the pipeline handler runs so the broker does NOT ack
    the delivery as a success: aio-pika's ``ProcessContext`` rejects on
    raise, which routes to the DLQ instead of silently stranding the
    stage in ``running`` with an acked (lost) message.
    """


def _is_claim_recovery_redelivery(message, retry_count: int) -> bool:
    """True when this delivery may carry an abandoned ``running`` claim.

    Ownership-loss signals only: the broker redelivered flag (the owner
    died before ack) or the bounded retry header/count (a previous
    attempt's ``running → pending`` note never landed, or the attempt
    died without reaching a queue failure branch). Plain duplicates —
    neither signal — return False so the worker claim still no-ops them
    cheaply without touching the row.
    """
    try:
        redelivered = bool(getattr(message, "redelivered", False))
    except Exception:  # noqa: BLE001 - a quirky message must not break consume
        redelivered = False
    if redelivered:
        return True
    if retry_count >= 1:
        return True
    try:
        headers = message.headers or {}
    except Exception:  # noqa: BLE001
        return False
    try:
        x_death = headers.get("x-death")
    except Exception:  # noqa: BLE001
        return False
    return isinstance(x_death, (list, tuple)) and len(x_death) > 0


async def _recover_abandoned_claim(
    *, message, queue_name: str, exchange_name: str, revision_id, retry_count: int
) -> str | None:
    """Recover this delivery's exact stage ``running → pending`` if signaled.

    Idempotent: only the message revision's own queue-mapped stage is
    touched; terminal (completed/skipped/failed) and unknown rows are a
    logged no-op (``"noop"``) — the worker's atomic claim still decides
    the delivery. Returns ``"recovered"`` when the edge applied,
    ``None`` when no recovery was warranted (no revision, no pipeline
    stage, no ownership-loss signal). Raises
    :class:`AbandonedStageRecoveryError` on infrastructure failure so the
    caller never runs the handler into a no-op ack.
    """
    from app.services.agents.v2.persistence.document_revisions import (
        DocumentRevisionsRepository,
        InvalidStageTransition,
        UnknownRevisionStage,
    )

    if revision_id is None:
        return None
    stage = stage_for_queue(queue_name, exchange_name)
    if stage is None:
        return None
    if not _is_claim_recovery_redelivery(message, retry_count):
        return None
    from app.core.database import async_session_maker

    try:
        async with async_session_maker() as session:
            repo = DocumentRevisionsRepository(session)
            try:
                await repo.mark_stage_retry_pending(revision_id, stage)
            except (InvalidStageTransition, UnknownRevisionStage) as e:
                logger.info(
                    f"[stage] pre-handler recovery left rev={revision_id} "
                    f"stage={stage} untouched: {e}"
                )
                return "noop"
            await session.commit()
    except Exception as e:  # infra failure must raise, never strand
        raise AbandonedStageRecoveryError(
            f"pre-handler recovery failed for rev={revision_id} "
            f"stage={stage}: {e}"
        ) from e
    logger.info(
        f"[stage] pre-handler recovery rev={revision_id} "
        f"stage={stage} running→pending"
    )
    return "recovered"


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
    global _connection, _pub_channel
    # Drop the cached publisher channel — it dies with the connection.
    _pub_channel = None
    _pub_exchanges.clear()
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
                    "x-dead-letter-exchange": RETRY_EXCHANGE,
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
# Publishing used to open a fresh channel (2 broker round-trips) per message —
# cache one long-lived channel + declared exchanges instead.
_pub_channel: aio_pika.abc.AbstractChannel | None = None
_pub_exchanges: dict[str, aio_pika.abc.AbstractExchange] = {}
_pub_lock = asyncio.Lock()


async def _get_publish_exchange(exchange_name: str) -> aio_pika.abc.AbstractExchange:
    """Return a declared exchange on the shared publisher channel."""
    global _pub_channel
    async with _pub_lock:
        if _pub_channel is None or _pub_channel.is_closed:
            conn = await get_connection()
            _pub_channel = await conn.channel()
            _pub_exchanges.clear()
        exchange = _pub_exchanges.get(exchange_name)
        if exchange is None:
            exchange = await _pub_channel.declare_exchange(
                exchange_name, ExchangeType.DIRECT, durable=True
            )
            _pub_exchanges[exchange_name] = exchange
        return exchange


async def _reset_publish_channel() -> None:
    global _pub_channel
    async with _pub_lock:
        if _pub_channel is not None and not _pub_channel.is_closed:
            try:
                await _pub_channel.close()
            except Exception:
                pass
        _pub_channel = None
        _pub_exchanges.clear()


async def publish(exchange_name: str, routing_key: str, payload: dict) -> None:
    """Publish a JSON message to *exchange_name* with *routing_key*."""
    # Use default=str to handle UUID serialization
    body = json.dumps(payload, default=str).encode()
    message = Message(body, delivery_mode=DeliveryMode.PERSISTENT)
    try:
        exchange = await _get_publish_exchange(exchange_name)
        await exchange.publish(message, routing_key=routing_key)
    except Exception as e:
        # Stale/broken cached channel — reset and retry once on a fresh one.
        logger.warning(f"publish to {exchange_name}/{routing_key} failed ({e}) — retrying on fresh channel")
        await _reset_publish_channel()
        exchange = await _get_publish_exchange(exchange_name)
        await exchange.publish(message, routing_key=routing_key)
    logger.debug(f"Published to {exchange_name}/{routing_key}: {payload}")


async def ensure_kg_queue(workspace_id) -> None:
    """
    Declare + bind the per-workspace KG queue BEFORE publishing to it.

    The queue is normally declared by the KG worker's consumer, but its
    workspace poller only discovers new workspaces every ~30s. A document
    uploaded to a brand-new workspace would publish its KGMessage before the
    queue existed — RabbitMQ silently drops unroutable messages, leaving the
    document stuck in BUILDING_KG forever. Declaring here (idempotent, same
    arguments as consume()) makes the message buffer until the worker binds.
    """
    queue_name = f"{QUEUE_KG_PREFIX}.{workspace_id}"
    conn = await get_connection()
    async with conn.channel() as channel:
        exchange = await channel.declare_exchange(
            EXCHANGE_KG, ExchangeType.DIRECT, durable=True
        )
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
            await queue.bind(exchange, routing_key=str(workspace_id))
        except Exception as e:
            # Queue exists with different args — it is already bound by the
            # consumer, so publishing is safe; just log and move on.
            logger.debug(f"ensure_kg_queue({queue_name}): {e}")


# ── Worker control-plane ────────────────────────────────────────────────────
# Broadcast commands (pause / resume / restart / set_prefetch) to worker
# processes. Every worker REPLICA gets its own exclusive auto-delete queue
# bound to its worker_type AND "all", so one publish reaches all replicas.
# This is how the admin UI manages Docker-containerised workers without
# touching the Docker socket: "restart" makes the worker drain + exit and the
# container's restart policy (always) brings it back up.
CONTROL_EXCHANGE = "hrag.control"

CONTROL_COMMANDS = {"pause", "resume", "restart", "set_prefetch"}


async def publish_control(target: str, command: str, extra: dict | None = None) -> None:
    """
    Publish a control command to workers.

    target:  a worker type ("parse"/"embed"/"caption"/"kg"/"memory") or "all".
    command: one of CONTROL_COMMANDS.
    extra:   optional command arguments (e.g. {"prefetch": 4}).
    """
    if command not in CONTROL_COMMANDS:
        raise ValueError(f"Unknown control command: {command}")
    payload = {"command": command, "target": target, **(extra or {})}
    conn = await get_connection()
    async with conn.channel() as channel:
        exchange = await channel.declare_exchange(
            CONTROL_EXCHANGE, ExchangeType.DIRECT, durable=True
        )
        # Non-persistent: control commands are ephemeral by design — a worker
        # that is down will come back with default state anyway.
        await exchange.publish(
            Message(json.dumps(payload).encode()),
            routing_key=target,
        )
    logger.info(f"[control] published {command} → {target}")


async def consume_control(
    worker_type: str, handler: Callable[[dict], Awaitable[None]]
) -> None:
    """
    Consume control commands for this worker replica.

    Each call creates an EXCLUSIVE auto-delete queue (one per process) bound
    to routing keys <worker_type> and "all" — broadcast semantics. Runs
    forever; callers wrap it in a task and restart on failure.
    """
    conn = await get_connection()
    channel = await conn.channel()
    exchange = await channel.declare_exchange(
        CONTROL_EXCHANGE, ExchangeType.DIRECT, durable=True
    )
    queue = await channel.declare_queue("", exclusive=True, auto_delete=True)
    await queue.bind(exchange, routing_key=worker_type)
    await queue.bind(exchange, routing_key="all")
    logger.info(f"[control] listening for commands (worker_type={worker_type})")

    async with queue.iterator() as messages:
        async for message in messages:
            async with message.process():
                try:
                    payload = json.loads(message.body)
                except Exception:
                    logger.warning("[control] unparseable control message — ignored")
                    continue
                try:
                    await handler(payload)
                except Exception as e:
                    # Control handler errors must never kill the listener.
                    logger.error(f"[control] handler failed for {payload}: {e}", exc_info=True)


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
    if "memory" in queue_name:
        return settings.WORKER_MEMORY_TIMEOUT
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
    try:
        await _consume_on_channel(
            conn, channel, exchange_name, queue_name, routing_key,
            handler, prefetch_count, handler_timeout,
        )
    finally:
        # The consume loop only exits on cancellation (pause/shutdown) or
        # abandon — close the channel so repeated pause/resume cycles from
        # the control-plane don't leak channels on the shared connection.
        try:
            if not channel.is_closed:
                await channel.close()
        except Exception:
            pass


async def _consume_on_channel(
    conn,
    channel,
    exchange_name: str,
    queue_name: str,
    routing_key: str,
    handler: MessageHandler,
    prefetch_count: int,
    handler_timeout: int,
) -> None:
    from app.workers.metrics import worker_metrics

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
                    # ignore_processed=True: we ack/reject manually in the
                    # failure branches below. Without it, aio_pika's context
                    # manager ack()s AGAIN on clean exit (it does not check
                    # `processed` when ignore_processed is False) → raises
                    # MessageProcessError, which kills the iterator and, after
                    # 10 such failures, abandons the queue entirely.
                    async with message.process(requeue=False, ignore_processed=True):
                        headers = message.headers or {}
                        retry_count = int(headers.get("x-retry-count", 0))
                        document_id = "unknown"
                        minio_key = "unknown"
                        revision_id = None

                        start_time = time.monotonic()
                        try:
                            payload = json.loads(message.body)
                            document_id = str(payload.get("document_id", "unknown"))
                            minio_key = payload.get("minio_key", "unknown")
                            raw_revision_id = payload.get("revision_id")
                            if raw_revision_id:
                                try:
                                    revision_id = uuid.UUID(str(raw_revision_id))
                                except (TypeError, ValueError):
                                    revision_id = None

                            # Abandoned-claim recovery (P1 Task 3 I-R1): a
                            # durable running row outlives its owner when the
                            # broker redelivers after a crash (redelivered)
                            # or when the retry note failed before the retry
                            # publish (retry header). Recover ONLY this
                            # message revision's exact stage running→pending
                            # so the worker's atomic claim below can execute
                            # it. Terminal stages converge to a no-op; plain
                            # duplicates (no signal) skip recovery entirely.
                            # Infrastructure failure raises (see below) so the
                            # handler never no-ops into a stranding ack.
                            await _recover_abandoned_claim(
                                message=message,
                                queue_name=queue_name,
                                exchange_name=exchange_name,
                                revision_id=revision_id,
                                retry_count=retry_count,
                            )

                            async with asyncio.timeout(handler_timeout):
                                await handler(payload)

                            elapsed = time.monotonic() - start_time
                            worker_metrics.record_success(queue_name, elapsed)
                        except AbandonedStageRecoveryError:
                            # Never handle as a worker failure (no ack, no
                            # retry-note, no DLQ-via-exhaustion): propagate so
                            # the broker rejects instead of acking.
                            raise
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
                            will_retry = retry_count < MAX_RETRIES
                            # Timeout: rollback document status to PENDING and requeue for retry
                            # Only do this for parse queue (other queues don't have status to rollback).
                            # When retries are exhausted the message is dropped, so leave a
                            # terminal FAILED state instead of a PENDING that never runs.
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
                                            if will_retry:
                                                doc.status = DocumentStatus.PENDING
                                                doc.error_message = f"timeout_retry: handler timed out after {elapsed:.1f}s"
                                            else:
                                                doc.status = DocumentStatus.FAILED
                                                doc.error_message = (
                                                    f"timeout: handler timed out after {elapsed:.1f}s — retries exhausted"
                                                )
                                            await rollback_db.commit()
                                            logger.info(
                                                f"[timeout_rollback] doc={document_id} "
                                                f"status set to {doc.status}"
                                            )
                                except Exception as rollback_err:
                                    logger.warning(
                                        f"[timeout_rollback] failed for doc={document_id}: {rollback_err}"
                                    )
                            # For KG worker: reset kg_done=False so retry can process again.
                            # When retries are exhausted, mark kg_done=True (KG skipped) so the
                            # document is not stuck in BUILDING_KG forever.
                            if "kg" in queue_name and document_id != "unknown":
                                try:
                                    from app.core.database import async_session_maker
                                    from sqlalchemy import select
                                    from app.models.document import Document, DocumentStatus
                                    from app.workers.utils import check_and_finalize

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
                                            if will_retry:
                                                doc.kg_done = False
                                                doc.status = DocumentStatus.BUILDING_KG
                                                doc.error_message = f"timeout_retry: KG handler timed out after {elapsed:.1f}s — will retry"
                                            else:
                                                doc.kg_done = True
                                                doc.error_message = (
                                                    "kg_timeout: retries exhausted — KG skipped"
                                                )
                                            await rollback_db.commit()
                                            logger.info(
                                                f"[timeout_rollback] doc={document_id} "
                                                f"kg_done={'reset for retry' if will_retry else 'True (KG skipped)'}"
                                            )
                                            if not will_retry:
                                                await check_and_finalize(
                                                    doc,
                                                    rollback_db,
                                                    revision_id=revision_id,
                                                )
                                except Exception as rollback_err:
                                    logger.warning(
                                        f"[timeout_rollback] KG failed for doc={document_id}: {rollback_err}"
                                    )
                            # Revision-owned stage retry (P1 Task 3): before
                            # requeueing, move ONLY this message's
                            # revision/stage running → pending (the attempt is
                            # preserved; the next worker running mark bumps
                            # it). Exhaustion terminalizes that stage.
                            # Terminal stages and unknown revisions are left
                            # untouched by the note helpers.
                            await _note_failure_stage(
                                queue_name=queue_name,
                                exchange_name=exchange_name,
                                revision_id=revision_id,
                                retry_count=retry_count,
                                failure_class="TimeoutError",
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
                                    # Retry-publish failure honesty: the retry
                                    # edge above already moved running →
                                    # pending, but no redelivery will arrive
                                    # (the message is acked below). Fail the
                                    # stage and terminalize the revision so
                                    # the row never strands pending.
                                    await _note_failure_stage(
                                        queue_name=queue_name,
                                        exchange_name=exchange_name,
                                        revision_id=revision_id,
                                        retry_count=retry_count,
                                        failure_class="TimeoutError",
                                        force_exhausted=True,
                                    )
                            # Ack the message; wrap in try-except because the channel may be
                            # in a broken state after CancelledError from asyncio.timeout.
                            try:
                                await message.ack()
                            except Exception as ack_err:
                                logger.warning(f"[{queue_name}] message.ack() failed: {ack_err} — ignoring")
                        except Exception as e:
                            import os
                            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")
                            elapsed = time.monotonic() - start_time
                            worker_metrics.record_failure(queue_name, elapsed)
                            is_cuda = "No CUDA GPUs are available" in str(e) or "cuda" in str(e).lower()
                            log_fn = logger.error if is_cuda else logger.warning
                            log_fn(
                                f"Handler error on {queue_name} "
                                f"(attempt {retry_count + 1}/{MAX_RETRIES + 1}): {e}",
                                extra={
                                    "queue_name": queue_name,
                                    "document_id": document_id,
                                    "minio_key": minio_key,
                                    "retry_count": retry_count,
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                    "is_cuda_error": is_cuda,
                                    "cuda_visible_devices": cuda_visible,
                                },
                            )
                            # Revision-owned stage retry (P1 Task 3): report
                            # running → pending for ONLY this message's
                            # revision/stage BEFORE the retry requeue below.
                            # Exhaustion takes the failed edge instead (no
                            # requeue follows, so no retry edge is reported).
                            await _note_failure_stage(
                                queue_name=queue_name,
                                exchange_name=exchange_name,
                                revision_id=revision_id,
                                retry_count=retry_count,
                                failure_class=type(e).__name__,
                            )
                            if retry_count < MAX_RETRIES:
                                log_fn(
                                    f"[{queue_name}] Scheduling retry {retry_count + 1}/{MAX_RETRIES + 1} "
                                    f"for {queue_name}/{queue_name}",
                                    extra={
                                        "queue_name": queue_name,
                                        "document_id": document_id,
                                        "minio_key": minio_key,
                                        "retry_count": retry_count,
                                        "error": str(e),
                                        "error_type": type(e).__name__,
                                        "is_cuda_error": is_cuda,
                                        "cuda_visible_devices": cuda_visible,
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
                                    # Retry-publish failure honesty: the retry
                                    # edge above already moved running →
                                    # pending, but no redelivery will arrive
                                    # (the message is acked below). Fail the
                                    # stage and terminalize the revision so
                                    # the row never strands pending.
                                    await _note_failure_stage(
                                        queue_name=queue_name,
                                        exchange_name=exchange_name,
                                        revision_id=revision_id,
                                        retry_count=retry_count,
                                        failure_class=type(e).__name__,
                                        force_exhausted=True,
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
                                        "is_cuda_error": is_cuda,
                                        "cuda_visible_devices": cuda_visible,
                                    },
                                )
                                # Retries exhausted (P1 Task 3): fail ONLY
                                # this message's revision/stage and terminalize
                                # its revision atomically (no requeue follows).
                                await _note_failure_stage(
                                    queue_name=queue_name,
                                    exchange_name=exchange_name,
                                    revision_id=revision_id,
                                    retry_count=retry_count,
                                    failure_class=type(e).__name__,
                                )
                                try:
                                    await message.reject(requeue=False)  # trigger x-dead-letter-exchange → DLQ
                                    logger.info(f"[{queue_name}] Message sent to DLQ")
                                except Exception as dlq_err:
                                    logger.error(f"[{queue_name}] Failed to send to DLQ: {dlq_err}")
                                    await message.ack()  # fallback: ack to prevent infinite loop

                    # A message was handled without the iterator blowing up —
                    # reset the consecutive-failure counter so the abandon guard
                    # below only trips on 10 *consecutive* iterator failures.
                    restart_count = 0
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

"""
Worker Management API
======================
Endpoints for monitoring RabbitMQ queues, pipeline status, retry operations,
health checks, worker process management, and dead-letter queue inspection.

Proxies RabbitMQ Management HTTP API + queries DB for pipeline status.
Worker processes are managed as background asyncio subprocesses.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.deps import get_db, require_superadmin
from app.models.document import Document, DocumentStatus
from app.models.user import User
from app.services.rabbitmq_management import get_rabbitmq_management
from app.queue.connection import (
    publish, EXCHANGE_PARSE, DLQ_QUEUE,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workers", tags=["workers"])

# Queue name prefix for filtering hrag queues only
_QUEUE_PREFIX = "hrag."

# ══════════════════════════════════════════════════════════════════════════════
# In-process worker management
# ══════════════════════════════════════════════════════════════════════════════
# Each managed worker is an asyncio subprocess (runs `python -m app.workers.runner`).
# The API server keeps track of them so users can start/stop/restart from the UI.

_VALID_WORKER_TYPES = {"parse", "embed", "caption", "kg"}


class _ManagedWorker:
    """Track a single worker subprocess."""

    def __init__(self, worker_type: str, process: asyncio.subprocess.Process):
        self.worker_type = worker_type
        self.process = process
        self.started_at = time.time()
        self.restart_count = 0

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def is_alive(self) -> bool:
        return self.process.returncode is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_type": self.worker_type,
            "pid": self.pid,
            "alive": self.is_alive,
            "started_at": self.started_at,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "restart_count": self.restart_count,
            "return_code": self.process.returncode,
        }


# worker_type → list of ManagedWorker (supports multiple instances per type)
_workers: dict[str, list[_ManagedWorker]] = {}
_workers_lock = asyncio.Lock()


async def _spawn_worker(worker_type: str) -> _ManagedWorker:
    """Spawn a single worker subprocess."""
    env = os.environ.copy()
    env["WORKER_TYPE"] = worker_type

    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "app.workers.runner",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    )

    worker = _ManagedWorker(worker_type, process)
    logger.info(f"Spawned worker-{worker_type} (PID={process.pid})")

    # Start log streaming in background
    asyncio.create_task(_stream_worker_logs(worker))

    return worker


async def _stream_worker_logs(worker: _ManagedWorker) -> None:
    """Stream subprocess stdout/stderr to main logger."""
    prefix = f"[worker-{worker.worker_type}:{worker.pid}]"
    try:
        while True:
            line = await worker.process.stdout.readline()
            if not line:
                break
            logger.info(f"{prefix} {line.decode().rstrip()}")
    except Exception:
        pass
    logger.info(f"{prefix} Process exited (code={worker.process.returncode})")


def _extract_queue_info(q: dict[str, Any]) -> dict[str, Any]:
    """Extract relevant fields from a RabbitMQ queue object."""
    msg_stats = q.get("message_stats", {})
    args = q.get("arguments", {})
    return {
        "name": q.get("name", ""),
        "messages_ready": q.get("messages_ready", 0),
        "messages_unacked": q.get("messages_unacknowledged", 0),
        "consumers": q.get("consumers", 0),
        "message_rate_in": (
            msg_stats.get("publish_details", {}).get("rate", 0)
            if msg_stats else 0
        ),
        "message_rate_out": (
            msg_stats.get("deliver_get_details", {}).get("rate", 0)
            if msg_stats else 0
        ),
        "has_dlx": bool(args.get("x-dead-letter-exchange")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Health Check
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/health")
async def workers_health(db: AsyncSession = Depends(get_db), user: User = Depends(require_superadmin)):
    """
    Comprehensive health check for the worker system.
    Returns status of RabbitMQ, each queue, active consumers, managed workers,
    and pipeline status summary.
    """
    health: dict[str, Any] = {
        "status": "healthy",
        "checks": {},
    }

    # ── RabbitMQ connectivity ─────────────────────────────────────────────
    try:
        mgmt = get_rabbitmq_management()
        node_health = await mgmt.get_node_health()
        health["checks"]["rabbitmq"] = {
            "status": "healthy",
            "version": node_health.get("rabbitmq_version"),
            "cluster": node_health.get("cluster_name"),
            "queue_totals": node_health.get("queue_totals", {}),
        }
    except Exception as exc:
        health["status"] = "degraded"
        health["checks"]["rabbitmq"] = {
            "status": "unhealthy",
            "error": str(exc),
        }

    # ── Queue health ──────────────────────────────────────────────────────
    queue_health: dict[str, Any] = {}
    try:
        mgmt = get_rabbitmq_management()
        raw_queues = await mgmt.list_queues()
        for q in raw_queues:
            name = q.get("name", "")
            if not name.startswith(_QUEUE_PREFIX):
                continue
            info = _extract_queue_info(q)
            worker_type = name.replace(_QUEUE_PREFIX, "").split(".")[0]

            status = "healthy"
            warnings = []
            if info["consumers"] == 0 and name != DLQ_QUEUE:
                status = "warning"
                warnings.append("no consumers — messages will queue up")
            if info["messages_ready"] > 100:
                status = "warning"
                warnings.append(f"backlog: {info['messages_ready']} messages pending")

            queue_health[name] = {
                "status": status,
                "consumers": info["consumers"],
                "messages_ready": info["messages_ready"],
                "messages_unacked": info["messages_unacked"],
                "has_dlx": info["has_dlx"],
                "warnings": warnings,
            }

            if status == "warning" and health["status"] == "healthy":
                health["status"] = "degraded"

    except Exception:
        pass  # RabbitMQ already flagged above

    health["checks"]["queues"] = queue_health

    # ── Dead-letter queue ─────────────────────────────────────────────────
    dlq_count = 0
    for q_name, q_info in queue_health.items():
        if q_name == DLQ_QUEUE:
            dlq_count = q_info.get("messages_ready", 0)
    health["checks"]["dead_letter_queue"] = {
        "status": "warning" if dlq_count > 0 else "healthy",
        "messages": dlq_count,
    }
    if dlq_count > 0 and health["status"] == "healthy":
        health["status"] = "degraded"

    # ── Managed workers (in-process) ──────────────────────────────────────
    managed = {}
    async with _workers_lock:
        for wtype, workers in _workers.items():
            alive = [w for w in workers if w.is_alive]
            managed[wtype] = {
                "running": len(alive),
                "total_spawned": len(workers),
                "pids": [w.pid for w in alive],
            }
    health["checks"]["managed_workers"] = managed

    # ── Pipeline (stuck documents) ────────────────────────────────────────
    _stuck_statuses = [
        DocumentStatus.PARSING,
        DocumentStatus.OCRING,
        DocumentStatus.CHUNKING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.BUILDING_KG,
    ]
    result = await db.execute(
        select(func.count(Document.id)).where(
            Document.status.in_(_stuck_statuses)
        )
    )
    in_progress = result.scalar() or 0

    result2 = await db.execute(
        select(func.count(Document.id)).where(
            Document.status == DocumentStatus.FAILED
        )
    )
    failed = result2.scalar() or 0

    health["checks"]["pipeline"] = {
        "status": "warning" if failed > 0 else "healthy",
        "documents_in_progress": in_progress,
        "documents_failed": failed,
    }

    return health


# ══════════════════════════════════════════════════════════════════════════════
# Worker Process Management (start / stop / restart / list)
# ══════════════════════════════════════════════════════════════════════════════


class WorkerStartRequest(BaseModel):
    worker_type: str
    count: int = 1


@router.get("/managed")
async def list_managed_workers(user: User = Depends(require_superadmin)):
    """List all managed worker processes started from this API server."""
    result: dict[str, Any] = {}
    async with _workers_lock:
        for wtype, workers in _workers.items():
            # Clean up dead workers from the list
            result[wtype] = [w.to_dict() for w in workers]
    return {"workers": result}


@router.post("/start")
async def start_worker(req: WorkerStartRequest, user: User = Depends(require_superadmin)):
    """
    Start one or more worker processes of the given type.
    Workers run as subprocesses managed by this API server.
    """
    wtype = req.worker_type.lower()
    if wtype not in _VALID_WORKER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid worker_type: {wtype}. Choose from: {sorted(_VALID_WORKER_TYPES)}",
        )
    if req.count < 1 or req.count > 8:
        raise HTTPException(status_code=400, detail="count must be 1-8")

    started = []
    async with _workers_lock:
        if wtype not in _workers:
            _workers[wtype] = []

        for _ in range(req.count):
            worker = await _spawn_worker(wtype)
            _workers[wtype].append(worker)
            started.append({"pid": worker.pid, "worker_type": wtype})

    return {"status": "ok", "started": started}


@router.post("/stop/{worker_type}")
async def stop_workers(worker_type: str, pid: int | None = None, user: User = Depends(require_superadmin)):
    """
    Stop managed workers of the given type.
    If pid is specified, only that worker is stopped.
    Otherwise, all workers of the type are stopped.
    """
    wtype = worker_type.lower()
    if wtype not in _VALID_WORKER_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid worker_type: {wtype}")

    stopped = []
    async with _workers_lock:
        workers = _workers.get(wtype, [])
        for w in workers:
            if not w.is_alive:
                continue
            if pid is not None and w.pid != pid:
                continue
            try:
                w.process.send_signal(signal.SIGTERM)
                # Give it 5s to shutdown gracefully
                try:
                    await asyncio.wait_for(w.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    w.process.kill()
                stopped.append({"pid": w.pid, "worker_type": wtype})
            except Exception as exc:
                logger.warning(f"Failed to stop worker PID={w.pid}: {exc}")

        # Remove dead workers from list
        _workers[wtype] = [w for w in workers if w.is_alive]

    if not stopped:
        raise HTTPException(
            status_code=404,
            detail=f"No running workers of type '{wtype}'" + (f" with PID={pid}" if pid else ""),
        )

    return {"status": "ok", "stopped": stopped}


@router.post("/restart/{worker_type}")
async def restart_workers(worker_type: str, user: User = Depends(require_superadmin)):
    """
    Restart all managed workers of the given type.
    Stops all existing ones and starts the same count of new ones.
    """
    wtype = worker_type.lower()
    if wtype not in _VALID_WORKER_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid worker_type: {wtype}")

    async with _workers_lock:
        old_workers = _workers.get(wtype, [])
        alive_count = len([w for w in old_workers if w.is_alive])

        # Stop all existing
        for w in old_workers:
            if w.is_alive:
                try:
                    w.process.send_signal(signal.SIGTERM)
                    try:
                        await asyncio.wait_for(w.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        w.process.kill()
                except Exception:
                    pass

        # Start new ones (at least 1)
        count = max(alive_count, 1)
        _workers[wtype] = []
        started = []
        for _ in range(count):
            worker = await _spawn_worker(wtype)
            _workers[wtype].append(worker)
            started.append({"pid": worker.pid, "worker_type": wtype})

    return {
        "status": "ok",
        "stopped_count": alive_count,
        "started": started,
    }


@router.delete("/managed/{worker_type}")
async def remove_dead_workers(worker_type: str, user: User = Depends(require_superadmin)):
    """Remove dead/exited worker entries from the managed list."""
    wtype = worker_type.lower()
    async with _workers_lock:
        before = len(_workers.get(wtype, []))
        _workers[wtype] = [w for w in _workers.get(wtype, []) if w.is_alive]
        after = len(_workers[wtype])

    return {"status": "ok", "removed": before - after, "remaining": after}


# ══════════════════════════════════════════════════════════════════════════════
# Dead-Letter Queue Management
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/dead-letter")
async def get_dead_letter_messages(count: int = 20, user: User = Depends(require_superadmin)):
    """
    Peek at messages in the dead-letter queue.
    Messages are NOT consumed — they remain in the queue.
    """
    try:
        mgmt = get_rabbitmq_management()
        messages = await mgmt.get_messages(DLQ_QUEUE, count=min(count, 100))
        return {
            "queue": DLQ_QUEUE,
            "count": len(messages),
            "messages": [
                {
                    "payload": m.get("payload", ""),
                    "headers": m.get("properties", {}).get("headers", {}),
                    "exchange": m.get("exchange", ""),
                    "routing_key": m.get("routing_key", ""),
                    "redelivered": m.get("redelivered", False),
                }
                for m in messages
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to read DLQ: {exc}")


@router.post("/dead-letter/purge")
async def purge_dead_letter(user: User = Depends(require_superadmin)):
    """Clear all messages from the dead-letter queue."""
    try:
        mgmt = get_rabbitmq_management()
        await mgmt.purge_queue(DLQ_QUEUE)
        return {"status": "ok", "queue": DLQ_QUEUE}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to purge DLQ: {exc}")


@router.post("/dead-letter/retry")
async def retry_dead_letter_messages(count: int = 100, user: User = Depends(require_superadmin)):
    """
    Move messages from dead-letter queue back to their original queues.
    Reads messages from DLQ and republishes them to the original exchange.
    """
    try:
        import json
        mgmt = get_rabbitmq_management()
        # Get messages with ack (consume them)
        messages = await mgmt.get_messages(
            DLQ_QUEUE, count=min(count, 100), ack_mode="ack_requeue_false"
        )

        retried = 0
        for m in messages:
            try:
                # Extract original exchange from x-death headers or routing key
                headers = m.get("properties", {}).get("headers", {})
                routing_key = m.get("routing_key", "")
                payload_str = m.get("payload", "{}")

                # Determine original exchange from first x-death entry
                x_death = headers.get("x-death", [])
                if x_death and isinstance(x_death, list):
                    original_exchange = x_death[0].get("exchange", "")
                    original_routing_key = x_death[0].get("routing-keys", [routing_key])[0]
                else:
                    # Fallback: infer from routing key
                    original_exchange = f"hrag.{routing_key}" if routing_key else ""
                    original_routing_key = routing_key

                if original_exchange:
                    payload = json.loads(payload_str)
                    # Reset retry count
                    await publish(original_exchange, original_routing_key, payload)
                    retried += 1
            except Exception as exc:
                logger.warning(f"Failed to retry DLQ message: {exc}")

        return {"status": "ok", "retried": retried, "total_in_dlq": len(messages)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to retry DLQ: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Queue Management
# ══════════════════════════════════════════════════════════════════════════════


@router.delete("/queues/{queue_name}")
async def delete_queue(queue_name: str, user: User = Depends(require_superadmin)):
    """Delete a queue entirely. Use for migration (e.g., recreating with DLX args)."""
    if not queue_name.startswith(_QUEUE_PREFIX):
        raise HTTPException(status_code=400, detail="Can only delete hrag.* queues")
    try:
        mgmt = get_rabbitmq_management()
        await mgmt.delete_queue(queue_name)
        return {"status": "ok", "queue": queue_name, "action": "deleted"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to delete queue: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Existing endpoints (overview, queues list, purge, retry, pipeline)
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/overview")
async def get_overview(db: AsyncSession = Depends(get_db), user: User = Depends(require_superadmin)):
    """
    Combined RabbitMQ stats + DB document pipeline counts.
    Gracefully handles RabbitMQ being unreachable.
    """
    rabbitmq_connected = True
    queues_data: list[dict[str, Any]] = []
    active_workers: dict[str, int] = {}

    try:
        mgmt = get_rabbitmq_management()
        raw_queues = await mgmt.list_queues()

        for q in raw_queues:
            name = q.get("name", "")
            if not name.startswith(_QUEUE_PREFIX):
                continue
            info = _extract_queue_info(q)
            queues_data.append(info)

            # Skip DLQ — it has no worker consumers, don't count it
            if name == DLQ_QUEUE:
                continue

            # Extract worker type from queue name (e.g. "hrag.parse" → "parse")
            worker_type = name.replace(_QUEUE_PREFIX, "").split(".")[0]
            active_workers[worker_type] = (
                active_workers.get(worker_type, 0) + info["consumers"]
            )
    except Exception as exc:
        logger.warning(f"RabbitMQ Management unreachable: {exc}")
        rabbitmq_connected = False

    # Pipeline summary from DB
    result = await db.execute(
        select(Document.status, func.count(Document.id)).group_by(Document.status)
    )
    status_counts = {row[0]: row[1] for row in result.all()}

    pipeline_summary = {
        "pending": status_counts.get(DocumentStatus.PENDING, 0),
        "parsing": status_counts.get(DocumentStatus.PARSING, 0),
        "ocring": status_counts.get(DocumentStatus.OCRING, 0),
        "chunking": status_counts.get(DocumentStatus.CHUNKING, 0),
        "embedding": status_counts.get(DocumentStatus.EMBEDDING, 0),
        "building_kg": status_counts.get(DocumentStatus.BUILDING_KG, 0),
        "indexed": status_counts.get(DocumentStatus.INDEXED, 0),
        "failed": status_counts.get(DocumentStatus.FAILED, 0),
    }

    # Include managed workers info
    managed_workers: dict[str, int] = {}
    async with _workers_lock:
        for wtype, workers in _workers.items():
            managed_workers[wtype] = len([w for w in workers if w.is_alive])

    return {
        "queues": queues_data,
        "pipeline_summary": pipeline_summary,
        "active_workers": active_workers,
        "managed_workers": managed_workers,
        "rabbitmq_connected": rabbitmq_connected,
    }


@router.get("/queues")
async def list_queues(user: User = Depends(require_superadmin)):
    """Returns all hrag.* queues with full metrics."""
    try:
        mgmt = get_rabbitmq_management()
        raw_queues = await mgmt.list_queues()
        return [
            _extract_queue_info(q)
            for q in raw_queues
            if q.get("name", "").startswith(_QUEUE_PREFIX)
        ]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RabbitMQ unreachable: {exc}")


@router.post("/queues/{queue_name}/purge")
async def purge_queue(queue_name: str, user: User = Depends(require_superadmin)):
    """Clear all pending messages from a specific queue."""
    if not queue_name.startswith(_QUEUE_PREFIX):
        raise HTTPException(status_code=400, detail="Can only purge hrag.* queues")
    try:
        mgmt = get_rabbitmq_management()
        await mgmt.purge_queue(queue_name)
        return {"status": "ok", "queue": queue_name}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to purge queue: {exc}")


@router.post("/retry-failed")
async def retry_all_failed(
    workspace_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Reset all FAILED documents to PENDING and republish ParseMessage."""
    query = select(Document).where(Document.status == DocumentStatus.FAILED)
    if workspace_id is not None:
        query = query.where(Document.workspace_id == workspace_id)

    result = await db.execute(query)
    failed_docs = result.scalars().all()

    count = 0
    for doc in failed_docs:
        doc.status = DocumentStatus.PENDING
        doc.error_message = None
        doc.embed_done = False
        doc.captions_done = False
        doc.kg_done = False
        await db.commit()

        # Republish parse task
        await publish(EXCHANGE_PARSE, "parse", {
            "document_id": doc.id,
            "workspace_id": doc.workspace_id,
            "minio_key": doc.filename,
        })
        count += 1

    return {"status": "ok", "retried_count": count}


@router.post("/retry-failed/{document_id}")
async def retry_single_failed(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Reset a single FAILED document to PENDING and republish ParseMessage."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != DocumentStatus.FAILED:
        raise HTTPException(status_code=400, detail="Document is not in FAILED status")

    doc.status = DocumentStatus.PENDING
    doc.error_message = None
    doc.embed_done = False
    doc.captions_done = False
    doc.kg_done = False
    await db.commit()

    await publish(EXCHANGE_PARSE, "parse", {
        "document_id": doc.id,
        "workspace_id": doc.workspace_id,
        "minio_key": doc.filename,
    })

    return {"status": "ok", "document_id": document_id}


@router.get("/pipeline")
async def get_pipeline(
    workspace_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """
    Returns in-progress + recently failed documents with detailed status.
    Does not return all indexed docs to keep payload small.
    """
    _active_statuses = [
        DocumentStatus.PENDING,
        DocumentStatus.PARSING,
        DocumentStatus.OCRING,
        DocumentStatus.CHUNKING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.BUILDING_KG,
        DocumentStatus.FAILED,
    ]

    query = (
        select(Document)
        .where(Document.status.in_(_active_statuses))
        .order_by(Document.updated_at.desc())
        .limit(100)
    )
    if workspace_id is not None:
        query = query.where(Document.workspace_id == workspace_id)

    result = await db.execute(query)
    docs = result.scalars().all()

    return {
        "documents": [
            {
                "id": d.id,
                "filename": d.original_filename or d.filename,
                "workspace_id": d.workspace_id,
                "status": d.status.value if hasattr(d.status, "value") else d.status,
                "embed_done": d.embed_done,
                "captions_done": d.captions_done,
                "kg_done": d.kg_done,
                "processing_time_ms": d.processing_time_ms,
                "error_message": d.error_message,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None,
            }
            for d in docs
        ]
    }

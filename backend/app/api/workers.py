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
import httpx
import time
import uuid
from datetime import timezone
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
    publish,
    publish_control,
    EXCHANGE_PARSE,
    DLQ_QUEUE,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workers", tags=["workers"])

# Queue name prefix for filtering hrag queues only
_QUEUE_PREFIX = "hrag."

# ══════════════════════════════════════════════════════════════════════════════
# Worker control-plane (Docker-containerised workers)
# ══════════════════════════════════════════════════════════════════════════════
# Workers run as Docker containers (compose services worker-parse/embed/…).
# The API cannot start/stop containers directly (no Docker socket); instead it
# publishes commands to the hrag.control exchange (app/queue/connection.py):
#   pause   → worker stops consuming (container stays up, queue buffers)
#   resume  → worker consumes again
#   restart → worker drains + exits; the container restart policy (always)
#             brings it back up = a real restart from the UI
# Replicas are discovered dynamically via Docker DNS on the compose service
# name (one A record per replica) — no hardcoded container names.

_VALID_WORKER_TYPES = {"parse", "embed", "caption", "kg", "memory"}

# worker type → compose service name
_WORKER_SERVICES: dict[str, str] = {
    "parse":   "worker-parse",
    "embed":   "worker-embed",
    "caption": "worker-caption",
    "kg":      "worker-kg",
}
_WORKER_HEALTH_PORT = 8081


async def _resolve_worker_instances(service: str) -> list[str]:
    """Resolve a compose service name to per-replica IPs (best-effort)."""
    import socket

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(
            service, _WORKER_HEALTH_PORT, type=socket.SOCK_STREAM
        )
    except socket.gaierror:
        return []
    return sorted({info[4][0] for info in infos})


async def _instance_display_name(ip: str) -> str:
    """Reverse-resolve a replica IP to its container name (best-effort)."""
    loop = asyncio.get_running_loop()
    try:
        name, _ = await loop.getnameinfo((ip, _WORKER_HEALTH_PORT), 0)
        return name.split(".")[0]
    except Exception:
        return ip


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
            msg_stats.get("publish_details", {}).get("rate", 0) if msg_stats else 0
        ),
        "message_rate_out": (
            msg_stats.get("deliver_get_details", {}).get("rate", 0) if msg_stats else 0
        ),
        "has_dlx": bool(args.get("x-dead-letter-exchange")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Health Check Helpers
# ══════════════════════════════════════════════════════════════════════════════


async def _check_openai_health(
    url: str | None, model: str | None, api_key: str | None = None
) -> dict[str, Any]:
    """Check health of an OpenAI-compatible API endpoint."""
    if not url:
        return {"status": "unhealthy", "error": "URL not configured"}

    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=3.0) as client:
            # We check the /models endpoint to confirm the server is up and knows about the model
            # Note: We strip /v1 if present to reach /v1/models correctly if base_url includes it
            base = url.rstrip("/")
            models_url = f"{base}/models"

            resp = await client.get(models_url, headers=headers)
            if resp.status_code != 200:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {resp.status_code}",
                    "url": url,
                }

            data = resp.json()
            models = [m.get("id") for m in data.get("data", [])]

            if model and model not in models:
                # If model is not in list, it might still be ok if it's a dynamic endpoint,
                # but usually it's a sign of a mismatch. We'll mark as warning.
                return {
                    "status": "warning",
                    "error": f"Model '{model}' not found in active list",
                    "available_models": models[:5],
                    "url": url,
                }

            return {"status": "healthy", "model": model, "url": url}
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc), "url": url}


# ══════════════════════════════════════════════════════════════════════════════
# Health Check
# ══════════════════════════════════════════════════════════════════════════════

# Worker health server ports (per container, via docker network)
# Access via docker service name: http://worker-{type}:8081/health
# For replicas, list all container names - health check requires ALL to be healthy
# Note: Docker DNS aliases for replicas are NOT auto-generated. Use actual container names
# which follow the pattern: {project}-{service}-{replica_number}
async def _check_worker_containers() -> dict[str, Any]:
    """Check health of worker container replicas via their health servers.

    Replicas are discovered through Docker DNS (compose service name → one A
    record per replica), so scaling with `docker compose up --scale` is picked
    up automatically. A paused worker still counts as a live container.
    """
    import httpx

    results: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for worker_type, service in _WORKER_SERVICES.items():
            ips = await _resolve_worker_instances(service)
            worker_results = []
            for ip in ips:
                container = await _instance_display_name(ip)
                try:
                    resp = await client.get(
                        f"http://{ip}:{_WORKER_HEALTH_PORT}/health"
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        paused = bool(data.get("paused", False))
                        worker_results.append({
                            "container": container,
                            "status": "paused" if paused else "healthy",
                            "worker_type": data.get("worker_type", worker_type),
                            "paused": paused,
                            "ready": bool(data.get("ready", False)),
                            "uptime_seconds": data.get("uptime_seconds", 0),
                        })
                    else:
                        worker_results.append({
                            "container": container,
                            "status": "unreachable",
                            "code": resp.status_code,
                        })
                except Exception as exc:
                    worker_results.append({
                        "container": container,
                        "status": "offline",
                        "error": str(exc)[:100],
                    })
            results[worker_type] = {
                "instances": worker_results,
                "healthy_count": len([
                    r for r in worker_results
                    if r.get("status") in ("healthy", "paused")
                ]),
                "total": len(worker_results),
            }
    return results


@router.get("/health")
async def workers_health(
    db: AsyncSession = Depends(get_db), user: User = Depends(require_superadmin)
):
    """
    Comprehensive health check for the worker system.
    Returns status of RabbitMQ, each queue, active consumers, managed workers,
    worker containers (via health servers), and pipeline status summary.
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

    # ── Worker containers ──────────────────────────────────────────────
    try:
        container_health = await _check_worker_containers()
        container_status = "healthy"
        for wtype, info in container_health.items():
            if info["healthy_count"] == 0:
                container_status = "unhealthy"
            elif info["healthy_count"] < info["total"]:
                container_status = "degraded"
        health["checks"]["worker_containers"] = {
            "status": container_status,
            "workers": container_health,
        }
        if container_status == "unhealthy":
            health["status"] = "degraded"
    except Exception as exc:
        health["checks"]["worker_containers"] = {
            "status": "unknown",
            "error": str(exc)[:200],
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

    # ── Pipeline (stuck documents) ────────────────────────────────────────
    _stuck_statuses = [
        DocumentStatus.PARSING,
        DocumentStatus.OCRING,
        DocumentStatus.CHUNKING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.BUILDING_KG,
    ]
    result = await db.execute(
        select(func.count(Document.id)).where(Document.status.in_(_stuck_statuses))
    )
    in_progress = result.scalar() or 0

    result2 = await db.execute(
        select(func.count(Document.id)).where(Document.status == DocumentStatus.FAILED)
    )
    failed = result2.scalar() or 0

    health["checks"]["pipeline"] = {
        "status": "warning" if failed > 0 else "healthy",
        "documents_in_progress": in_progress,
        "documents_failed": failed,
    }

    # ── AI Services Health ────────────────────────────────────────────────
    llm_tasks = [
        _check_openai_health(
            os.getenv("HUNYUAN_OCR_API_URL"), os.getenv("HUNYUAN_OCR_MODEL")
        ),
        _check_openai_health(
            os.getenv("MEMORY_AGENT_BASE_URL") or os.getenv("LEGAL_KG_LLM_BASE_URL"),
            os.getenv("MEMORY_AGENT_MODEL") or os.getenv("LEGAL_KG_LLM_MODEL"),
        ),
        _check_openai_health(
            os.getenv("OPENAI_COMPATIBLE_BASE_URL"),
            os.getenv("OPENAI_COMPATIBLE_MODEL"),
            os.getenv("OPENAI_COMPATIBLE_API_KEY"),
        ),
    ]
    llm_results = await asyncio.gather(*llm_tasks)

    health["checks"]["llm_services"] = {
        "ocr": llm_results[0],
        "memory": llm_results[1],
        "main_llm": llm_results[2],
    }

    return health


# ══════════════════════════════════════════════════════════════════════════════
# Worker Control (pause / resume / restart via the hrag.control exchange)
# ══════════════════════════════════════════════════════════════════════════════


class WorkerStartRequest(BaseModel):
    worker_type: str
    count: int = 1  # deprecated — replica count is fixed by docker compose


def _validate_worker_type(worker_type: str) -> str:
    wtype = worker_type.lower()
    if wtype not in _VALID_WORKER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid worker_type: {wtype}. Choose from: {sorted(_VALID_WORKER_TYPES)}",
        )
    return wtype


@router.get("/managed")
async def list_managed_workers(user: User = Depends(require_superadmin)):
    """Live status of worker container replicas (via their health servers)."""
    container_health = await _check_worker_containers()
    return {
        "workers": {
            wtype: info["instances"] for wtype, info in container_health.items()
        }
    }


@router.post("/start")
async def start_worker(
    req: WorkerStartRequest, user: User = Depends(require_superadmin)
):
    """
    RESUME consumption for all replicas of the given worker type.

    Workers run as Docker containers — the API cannot spawn new processes.
    `count` is accepted for backward compatibility but ignored; scale replicas
    with `docker compose up -d --scale worker-<type>=N`.
    """
    wtype = _validate_worker_type(req.worker_type)
    await publish_control(wtype, "resume")
    return {
        "status": "ok",
        "command": "resume",
        "worker_type": wtype,
        "note": "count is ignored — replica count is managed by docker compose",
    }


@router.post("/stop/{worker_type}")
async def stop_workers(worker_type: str, user: User = Depends(require_superadmin)):
    """
    PAUSE consumption for all replicas of the given worker type.

    Containers stay up (Docker healthcheck keeps passing); pending messages
    buffer durably in RabbitMQ until resume.
    """
    wtype = _validate_worker_type(worker_type)
    await publish_control(wtype, "pause")
    return {"status": "ok", "command": "pause", "worker_type": wtype}


@router.post("/restart/{worker_type}")
async def restart_workers(worker_type: str, user: User = Depends(require_superadmin)):
    """
    RESTART all replicas of the given worker type.

    Each worker drains its in-flight message and exits; the container restart
    policy (`always`) brings it back up with fresh state.
    """
    wtype = _validate_worker_type(worker_type)
    await publish_control(wtype, "restart")
    return {"status": "ok", "command": "restart", "worker_type": wtype}


@router.post("/restart-all")
async def restart_all_workers(user: User = Depends(require_superadmin)):
    """Restart ALL worker types (broadcast on the control exchange)."""
    await publish_control("all", "restart")
    return {"status": "ok", "command": "restart", "worker_type": "all"}


class PrefetchRequest(BaseModel):
    prefetch: int


@router.post("/prefetch/{worker_type}")
async def set_worker_prefetch(
    worker_type: str, req: PrefetchRequest, user: User = Depends(require_superadmin)
):
    """
    Tune message prefetch (concurrent handlers) at runtime.

    In-memory only — reverts to the env value on the next container restart.
    """
    wtype = _validate_worker_type(worker_type)
    if not 1 <= req.prefetch <= 64:
        raise HTTPException(status_code=400, detail="prefetch must be 1-64")
    await publish_control(wtype, "set_prefetch", {"prefetch": req.prefetch})
    return {
        "status": "ok",
        "command": "set_prefetch",
        "worker_type": wtype,
        "prefetch": req.prefetch,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dead-Letter Queue Management
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/dead-letter")
async def get_dead_letter_messages(
    count: int = 20, user: User = Depends(require_superadmin)
):
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
async def retry_dead_letter_messages(
    count: int = 100, user: User = Depends(require_superadmin)
):
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
                    original_routing_key = x_death[0].get(
                        "routing-keys", [routing_key]
                    )[0]
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
async def get_overview(
    db: AsyncSession = Depends(get_db), user: User = Depends(require_superadmin)
):
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

    # Legacy in-process managed workers were removed — workers run as Docker
    # containers now. Key kept (always empty) for frontend compatibility.
    managed_workers: dict[str, int] = {}

    # Include container worker counts (workers running as Docker containers)
    # For KG, active_workers shows consumers per workspace queue, not container count
    # KG uses per-workspace queues so 1 container consuming from multiple queues
    try:
        container_health = await _check_worker_containers()
        container_workers: dict[str, int] = {
            wtype: info["healthy_count"] for wtype, info in container_health.items()
        }
    except Exception:
        container_workers = {}

    return {
        "queues": queues_data,
        "pipeline_summary": pipeline_summary,
        "active_workers": active_workers,
        "managed_workers": managed_workers,
        "container_workers": container_workers,
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
        await publish(
            EXCHANGE_PARSE,
            "parse",
            {
                "document_id": doc.id,
                "workspace_id": doc.workspace_id,
                "minio_key": doc.upload_s3_key or doc.filename,
                "original_filename": doc.original_filename or doc.filename,
                "is_chat_upload": doc.is_chat_upload,
            },
        )
        count += 1

    return {"status": "ok", "retried_count": count}


@router.post("/retry-failed/{document_id}")
async def retry_single_failed(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Reset a single FAILED document to PENDING and republish ParseMessage."""
    result = await db.execute(select(Document).where(Document.id == document_id))
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

    await publish(
        EXCHANGE_PARSE,
        "parse",
        {
            "document_id": doc.id,
            "workspace_id": doc.workspace_id,
            "minio_key": doc.upload_s3_key or doc.filename,
            "original_filename": doc.original_filename or doc.filename,
            "is_chat_upload": doc.is_chat_upload,
        },
    )

    return {"status": "ok", "document_id": document_id}


@router.post("/pipeline/{document_id}/cancel")
async def cancel_pipeline_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """
    Cancel an in-progress document by marking it as FAILED and cleaning up
    all associated pipeline data (ChromaDB vectors, KG nodes, MinIO objects).
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    active_statuses = [
        DocumentStatus.PENDING,
        DocumentStatus.PARSING,
        DocumentStatus.OCRING,
        DocumentStatus.CHUNKING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.BUILDING_KG,
    ]
    if doc.status not in active_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel document in '{doc.status}' status",
        )

    doc.status = DocumentStatus.FAILED
    doc.error_message = "Cancelled by user"
    doc.embed_done = False
    doc.captions_done = False
    doc.kg_done = False
    await db.commit()

    try:
        from app.services.retrieval.rag_service import get_rag_service

        rag_service = get_rag_service(db, doc.workspace_id)
        await rag_service.delete_document(document_id)
    except Exception as exc:
        logger.warning(f"Cleanup failed for cancelled doc {document_id}: {exc}")

    logger.info(f"Cancelled pipeline document {document_id}")
    return {"status": "ok", "document_id": document_id}


@router.post("/retry-stuck")
async def retry_stuck_documents(
    workspace_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """
    Reset documents stuck in intermediate states (ocring, parsing, chunking,
    embedding, building_kg) back to PENDING and republish ParseMessage so
    workers can reprocess them. Does NOT touch PENDING, INDEXED, or FAILED docs.
    """
    _stuck_statuses = [
        DocumentStatus.PARSING,
        DocumentStatus.OCRING,
        DocumentStatus.CHUNKING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.BUILDING_KG,
    ]

    query = select(Document).where(Document.status.in_(_stuck_statuses))
    if workspace_id is not None:
        query = query.where(Document.workspace_id == workspace_id)

    result = await db.execute(query)
    stuck_docs = result.scalars().all()

    retried = 0
    for doc in stuck_docs:
        doc.status = DocumentStatus.PENDING
        doc.error_message = f"stuck_retry: was {doc.status.value}, reset at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        doc.embed_done = False
        doc.captions_done = False
        doc.kg_done = False
        await db.commit()

        await publish(
            EXCHANGE_PARSE,
            "parse",
            {
                "document_id": doc.id,
                "workspace_id": doc.workspace_id,
                "minio_key": doc.upload_s3_key or doc.filename,
                "original_filename": doc.original_filename or doc.filename,
                "is_chat_upload": doc.is_chat_upload,
            },
        )
        retried += 1

    return {
        "status": "ok",
        "retried_count": retried,
        "documents": [
            {
                "id": str(doc.id),
                "filename": doc.original_filename or doc.filename,
                "previous_status": doc.status.value if hasattr(doc.status, "value") else doc.status,
            }
            for doc in stuck_docs
        ],
    }


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
                "updated_at": (
                    d.updated_at.replace(tzinfo=timezone.utc).isoformat()
                    if d.updated_at
                    else None
                ),
            }
            for d in docs
        ]
    }


# ══════════════════════════════════════════════════════════════════════════════
# GPU / VRAM monitoring (NVML)
# ══════════════════════════════════════════════════════════════════════════════
# Board-level memory/utilization/temperature from NVML is global (covers vLLM
# engines + workers on the same GPU). The per-process breakdown requires the
# backend container to run with `pid: "host"` (docker-compose.services.yml) —
# in an isolated PID namespace NVML only returns the backend's own process.
# Without pid:host the processes list is simply incomplete; everything else
# still works.

_nvml_state = {"initialized": False, "failed": False}


def _gpu_process_label(pid: int) -> str:
    """Best-effort friendly label for a GPU process via /proc (host PIDs)."""
    # With pid:host our own PID matches what NVML reports — and the uvicorn
    # --reload app child has an unhelpful `python -c "from multiprocessing…"`
    # cmdline, so identify ourselves first.
    if pid == os.getpid():
        return "backend (uvicorn)"

    def _argv(p: int) -> list[str]:
        try:
            with open(f"/proc/{p}/cmdline", "rb") as f:
                return [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
        except OSError:
            return []

    argv = _argv(pid)
    if not argv:
        return f"pid {pid}"
    joined = " ".join(argv)

    # vLLM engine-core processes are renamed ("VLLM::EngineCore") — the model
    # name lives on the parent `vllm serve <model>` command line.
    if "vllm" in joined.lower():
        vllm_argv = argv
        if "serve" not in vllm_argv:
            try:
                with open(f"/proc/{pid}/status") as f:
                    ppid = next(
                        int(line.split()[1]) for line in f if line.startswith("PPid:")
                    )
                parent = _argv(ppid)
                if any("vllm" in a.lower() for a in parent):
                    vllm_argv = parent
            except (OSError, StopIteration, ValueError):
                pass
        model = None
        if "serve" in vllm_argv:
            i = vllm_argv.index("serve")
            if i + 1 < len(vllm_argv) and not vllm_argv[i + 1].startswith("-"):
                model = vllm_argv[i + 1]
        return f"vLLM · {model.rsplit('/', 1)[-1]}" if model else "vLLM"

    if "app.workers.runner" in joined:
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                for kv in f.read().split(b"\0"):
                    if kv.startswith(b"WORKER_TYPE="):
                        return "worker-" + kv.split(b"=", 1)[1].decode(errors="replace")
        except OSError:
            pass
        return "worker"

    if "uvicorn" in joined and "app.main:app" in joined:
        return "backend (uvicorn)"

    exe = os.path.basename(argv[0])
    if exe.startswith("python") and len(argv) > 1:
        if argv[1] == "-m" and len(argv) > 2:
            return argv[2]
        if not argv[1].startswith("-"):
            return os.path.basename(argv[1])
    return exe


def _read_gpu_stats() -> dict[str, Any]:
    """Read per-GPU VRAM/utilization via NVML. Blocking — run in a thread."""
    try:
        import pynvml
    except ImportError:
        return {"available": False, "error": "pynvml not installed", "gpus": []}

    if _nvml_state["failed"]:
        return {"available": False, "error": "NVML unavailable", "gpus": []}
    if not _nvml_state["initialized"]:
        try:
            pynvml.nvmlInit()
            _nvml_state["initialized"] = True
        except Exception as e:  # no NVIDIA driver / not a GPU host
            _nvml_state["failed"] = True
            logger.warning("NVML init failed, GPU stats disabled: %s", e)
            return {"available": False, "error": str(e), "gpus": []}

    gpus: list[dict[str, Any]] = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except pynvml.NVMLError:
                util = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                temp = None
            used_mb = mem.used // (1024 * 1024)
            total_mb = mem.total // (1024 * 1024)
            procs: list[dict[str, Any]] = []
            try:
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    used = getattr(p, "usedGpuMemory", None)
                    procs.append(
                        {
                            "pid": p.pid,
                            "label": _gpu_process_label(p.pid),
                            "memory_mb": (used // (1024 * 1024)) if used else 0,
                        }
                    )
                procs.sort(key=lambda x: x["memory_mb"], reverse=True)
            except pynvml.NVMLError:
                pass
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "memory_used_mb": used_mb,
                    "memory_total_mb": total_mb,
                    "memory_pct": round(used_mb / total_mb * 100, 1) if total_mb else 0.0,
                    "utilization_pct": util,
                    "temperature_c": temp,
                    "processes": procs,
                }
            )
    except Exception as e:
        logger.warning("NVML read failed: %s", e)
        return {"available": False, "error": str(e), "gpus": gpus}
    return {"available": True, "gpus": gpus}


@router.get("/gpu")
async def get_gpu_stats(user: User = Depends(require_superadmin)):
    """Real-time GPU VRAM / utilization / temperature for the Workers page."""
    return await asyncio.to_thread(_read_gpu_stats)

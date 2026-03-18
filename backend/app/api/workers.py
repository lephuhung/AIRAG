"""
Worker Management API
======================
Endpoints for monitoring RabbitMQ queues, pipeline status, and retry operations.
Proxies RabbitMQ Management HTTP API + queries DB for pipeline status.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update

from app.core.deps import get_db
from app.models.document import Document, DocumentStatus
from app.services.rabbitmq_management import get_rabbitmq_management
from app.queue.connection import publish, EXCHANGE_PARSE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workers", tags=["workers"])

# Queue name prefix for filtering nexusrag queues only
_QUEUE_PREFIX = "nexusrag."


def _extract_queue_info(q: dict[str, Any]) -> dict[str, Any]:
    """Extract relevant fields from a RabbitMQ queue object."""
    msg_stats = q.get("message_stats", {})
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
    }


# ------------------------------------------------------------------
# GET /workers/overview — Dashboard data (single call)
# ------------------------------------------------------------------
@router.get("/overview")
async def get_overview(db: AsyncSession = Depends(get_db)):
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

            # Extract worker type from queue name (e.g. "nexusrag.parse" → "parse")
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

    return {
        "queues": queues_data,
        "pipeline_summary": pipeline_summary,
        "active_workers": active_workers,
        "rabbitmq_connected": rabbitmq_connected,
    }


# ------------------------------------------------------------------
# GET /workers/queues — Detailed queue list
# ------------------------------------------------------------------
@router.get("/queues")
async def list_queues():
    """Returns all nexusrag.* queues with full metrics."""
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


# ------------------------------------------------------------------
# POST /workers/queues/{queue_name}/purge — Purge a queue
# ------------------------------------------------------------------
@router.post("/queues/{queue_name}/purge")
async def purge_queue(queue_name: str):
    """Clear all pending messages from a specific queue."""
    if not queue_name.startswith(_QUEUE_PREFIX):
        raise HTTPException(status_code=400, detail="Can only purge nexusrag.* queues")
    try:
        mgmt = get_rabbitmq_management()
        await mgmt.purge_queue(queue_name)
        return {"status": "ok", "queue": queue_name}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to purge queue: {exc}")


# ------------------------------------------------------------------
# POST /workers/retry-failed — Retry all failed documents
# ------------------------------------------------------------------
@router.post("/retry-failed")
async def retry_all_failed(
    workspace_id: int | None = None,
    db: AsyncSession = Depends(get_db),
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


# ------------------------------------------------------------------
# POST /workers/retry-failed/{document_id} — Retry single failed doc
# ------------------------------------------------------------------
@router.post("/retry-failed/{document_id}")
async def retry_single_failed(
    document_id: int,
    db: AsyncSession = Depends(get_db),
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


# ------------------------------------------------------------------
# GET /workers/pipeline — Pipeline detail (per-document breakdown)
# ------------------------------------------------------------------
@router.get("/pipeline")
async def get_pipeline(
    workspace_id: int | None = None,
    db: AsyncSession = Depends(get_db),
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

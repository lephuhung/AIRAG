"""
MinIO Events Webhook
====================
Receives S3 event notifications from MinIO when a file is PUT into the
nexusrag-uploads bucket and publishes a ParseMessage to RabbitMQ.

MinIO must be configured with:
  MINIO_NOTIFY_WEBHOOK_ENABLE_NEXUSRAG=on
  MINIO_NOTIFY_WEBHOOK_ENDPOINT_NEXUSRAG=http://backend:8080/api/v1/minio/events

And the bucket event must be registered:
  mc event add local/nexusrag-uploads arn:minio:sqs::NEXUSRAG:webhook --event put
"""
from __future__ import annotations

import logging
import re
import urllib.parse

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_db
from app.models.document import Document, DocumentStatus
from app.queue.publisher import publish_parse_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/minio", tags=["minio-events"])

# Key format: kb_{workspace_id}/doc_{document_id}.{ext}
_KEY_RE = re.compile(r"^kb_(\d+)/doc_(\d+)\.\w+$")


@router.post("/events")
async def handle_minio_event(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Receive S3 event notification from MinIO.

    For each ObjectCreated event in nexusrag-uploads, look up the matching
    Document record and publish a ParseMessage to RabbitMQ (idempotent —
    duplicate events for already-processing documents are silently ignored).
    """
    try:
        payload = await request.json()
    except Exception as e:
        logger.warning(f"[minio_events] Failed to parse request body: {e}")
        return {"status": "ok"}

    records = payload.get("Records", [])
    if not records:
        return {"status": "ok"}

    for record in records:
        event_name = record.get("eventName", "")
        if not event_name.startswith("s3:ObjectCreated"):
            continue

        bucket = record.get("s3", {}).get("bucket", {}).get("name", "")
        key_raw = record.get("s3", {}).get("object", {}).get("key", "")
        key = urllib.parse.unquote(key_raw)

        if bucket != settings.MINIO_BUCKET_UPLOADS:
            logger.debug(
                f"[minio_events] ignoring event for bucket '{bucket}' "
                f"(expected '{settings.MINIO_BUCKET_UPLOADS}')"
            )
            continue

        match = _KEY_RE.match(key)
        if not match:
            logger.warning(
                f"[minio_events] key '{key}' does not match expected pattern — skipping"
            )
            continue

        workspace_id = int(match.group(1))
        document_id = int(match.group(2))

        result = await db.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()

        if document is None:
            logger.warning(
                f"[minio_events] doc={document_id} not found in DB — skipping"
            )
            continue

        if document.status != DocumentStatus.PENDING:
            logger.debug(
                f"[minio_events] doc={document_id} status={document.status.value} "
                f"— skipping (not PENDING, idempotent)"
            )
            continue

        try:
            await publish_parse_task(
                document_id=document_id,
                workspace_id=workspace_id,
                minio_key=key,
                original_filename=document.original_filename,
            )
            logger.info(
                f"[minio_events] doc={document_id} workspace={workspace_id} "
                f"queued for parsing via webhook"
            )
        except Exception as e:
            logger.error(
                f"[minio_events] Failed to publish parse task for doc {document_id}: {e}"
            )

    return {"status": "ok"}

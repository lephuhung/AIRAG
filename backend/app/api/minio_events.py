"""
MinIO Events Webhook
====================
Receives S3 event notifications from MinIO when a file is PUT into the
hrag-uploads bucket and records the object's arrival metadata.

**The webhook is metadata-only.** It issues no body read, allocates no
revision, and publishes no parse task: for the presigned flow ``/confirm``
is the authoritative producer, and the direct-upload API publishes after
its own MinIO write. This handler exists to stage
``source_arrivals.arrival_identity`` so an arrival can be matched to the
``/confirm`` callback for the same storage event.

MinIO must be configured with:
  MINIO_NOTIFY_WEBHOOK_ENABLE_HRAG=on
  MINIO_NOTIFY_WEBHOOK_ENDPOINT_HRAG=http://backend:8080/api/v1/minio/events

And the bucket event must be registered:
  mc event add local/hrag-uploads arn:minio:sqs::HRAG:webhook --event put
"""
from __future__ import annotations

import logging
import re
import uuid

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_db
from app.queue.publisher import record_source_arrival
from app.services.agents.v2.persistence.source_identity import (
    InvalidSourceObjectKey,
    MissingObjectVersion,
    object_key_from_s3_event,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/minio", tags=["minio-events"])

# Key shapes: kb_{workspace_id}/doc_{document_id}.{ext} and the chat-upload
# form kb_{workspace_id}/chat_file_{document_id}.{ext}.
_KEY_RE = re.compile(
    r"^kb_([0-9a-f-]+)/(?:doc|chat_file)_([0-9a-f-]+)\.\w+$"
)


@router.post("/events")
async def handle_minio_event(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Record each ``ObjectCreated`` arrival; never allocate or publish.

    Duplicate deliveries collapse on the ``source_arrivals.arrival_identity``
    unique key, so redelivery is idempotent.
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
        if bucket != settings.MINIO_BUCKET_UPLOADS:
            logger.debug(
                f"[minio_events] ignoring event for bucket '{bucket}' "
                f"(expected '{settings.MINIO_BUCKET_UPLOADS}')"
            )
            continue

        obj = record.get("s3", {}).get("object", {})
        key_raw = obj.get("key", "")
        # S3/MinIO event keys are form-encoded; decode exactly once, then
        # normalize. Storage keys are NOT encoded, so this is the only
        # trigger that form-decodes.
        try:
            key = object_key_from_s3_event(key_raw)
        except InvalidSourceObjectKey as e:
            logger.warning(
                f"[minio_events] invalid object key {key_raw!r}: {e} — skipping"
            )
            continue

        match = _KEY_RE.match(key)
        if not match:
            logger.warning(
                f"[minio_events] key '{key}' does not match expected pattern — skipping"
            )
            continue

        workspace_id = uuid.UUID(match.group(1))
        document_id = uuid.UUID(match.group(2))

        version_id = obj.get("versionId") or None
        etag = obj.get("eTag") or None
        size_raw = obj.get("size")
        try:
            size_bytes = int(size_raw) if size_raw is not None else None
        except (TypeError, ValueError):
            size_bytes = None

        try:
            arrival_identity = await record_source_arrival(
                db,
                bucket=bucket,
                object_key=key,
                version_id=version_id,
                etag=etag,
                size_bytes=size_bytes,
            )
            await db.commit()
        except MissingObjectVersion as e:
            logger.warning(
                f"[minio_events] doc={document_id} key={key} has no version "
                f"selector ({e}) — arrival not recorded"
            )
            continue
        except Exception as e:
            await db.rollback()
            logger.error(
                f"[minio_events] failed to record arrival for doc={document_id} "
                f"key={key}: {e}"
            )
            continue

        logger.info(
            f"[minio_events] recorded arrival ws={workspace_id} doc={document_id} "
            f"key={key} identity={arrival_identity}"
        )

    return {"status": "ok"}

"""
Queue Publisher
===============
High-level helpers called by the FastAPI upload endpoint.
"""

from __future__ import annotations

import uuid

from app.queue import connection as mq
from app.queue.messages import ParseMessage


def _is_chat_upload_key(minio_key: str, workspace_id: uuid.UUID) -> bool:
    """Detect chat upload from MinIO key pattern.

    Chat files from format_check.py use the prefix:
        kb_{workspace_id}/chat_file_{uuid}.* (e.g., .docx, .md)
    Regular uploads use:
        kb_{workspace_id}/{document_id}/filename
    """
    prefix = f"kb_{workspace_id}/chat_file_"
    return minio_key.startswith(prefix)


async def publish_parse_task(
    document_id: uuid.UUID,
    workspace_id: uuid.UUID,
    minio_key: str,
    original_filename: str,
) -> None:
    """Publish a ParseMessage to hrag.parse queue."""
    is_chat_upload = _is_chat_upload_key(minio_key, workspace_id)
    await mq.publish(
        mq.EXCHANGE_PARSE,
        "parse",
        ParseMessage(
            document_id=document_id,
            workspace_id=workspace_id,
            minio_key=minio_key,
            original_filename=original_filename,
            is_chat_upload=is_chat_upload,
        ).model_dump(mode="json"),
    )

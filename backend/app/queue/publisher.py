"""
Queue Publisher
===============
High-level helpers called by the FastAPI upload endpoint.
"""
from __future__ import annotations

from app.queue import connection as mq
from app.queue.messages import ParseMessage


async def publish_parse_task(
    document_id: int,
    workspace_id: int,
    minio_key: str,
    original_filename: str,
) -> None:
    """Publish a ParseMessage to hrag.parse queue."""
    await mq.publish(
        mq.EXCHANGE_PARSE,
        "parse",
        ParseMessage(
            document_id=document_id,
            workspace_id=workspace_id,
            minio_key=minio_key,
            original_filename=original_filename,
        ).model_dump(),
    )

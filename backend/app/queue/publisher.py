"""
Queue Publisher
===============
High-level helpers called by the FastAPI upload endpoint.
"""

from __future__ import annotations

import uuid

from app.queue import connection as mq
from app.queue.messages import MemorySaveMessage, ParseMessage


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


async def publish_memory_save_task(
    user_id: uuid.UUID,
    user_message: str,
    assistant_message: str = "",
    session_id: str | None = None,
) -> None:
    """Publish a MemorySaveMessage to the durable hrag.memory queue.

    Replaces the old fire-and-forget ``asyncio.create_task`` save: the memory
    worker does the LLM fact-extraction + Graphiti write, and RabbitMQ retries
    transient failures (5s/15s/60s) before dead-lettering — so a fact is not
    lost if the worker, Neo4j, or the LLM is briefly unavailable.
    """
    await mq.publish(
        mq.EXCHANGE_MEMORY,
        "memory",
        MemorySaveMessage(
            user_id=user_id,
            user_message=user_message,
            assistant_message=assistant_message,
            session_id=session_id,
        ).model_dump(mode="json"),
    )

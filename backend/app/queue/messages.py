"""
Queue Message Schemas
=====================
Pydantic models for every message type passed through RabbitMQ.
All document-pipeline messages carry document_id + workspace_id as
primary keys AND the required ``revision_id`` that owns the work.

``revision_id`` is required with no default: a message that does not
name a revision cannot be executed, because revision state (not the
mutable ``Document`` row) is authoritative for processing. Child
messages published by a worker preserve the same ``revision_id`` and
``build_profile`` so every stage of one ingest event executes against
exactly one revision.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel


class ParseMessage(BaseModel):
    """Dispatched by the API after a file is uploaded."""

    document_id: uuid.UUID
    workspace_id: uuid.UUID
    revision_id: uuid.UUID
    build_profile: str
    minio_key: str  # key in hrag-uploads bucket
    original_filename: str
    is_chat_upload: bool = False  # True → skip embed/caption/kg workers


class EmbedMessage(BaseModel):
    """Dispatched by parse_worker once structural parsing is done."""

    document_id: uuid.UUID
    workspace_id: uuid.UUID
    revision_id: uuid.UUID
    build_profile: str


class CaptionMessage(BaseModel):
    """Dispatched by parse_worker for image + table captioning."""

    document_id: uuid.UUID
    workspace_id: uuid.UUID
    revision_id: uuid.UUID
    build_profile: str


class KGMessage(BaseModel):
    """
    Dispatched by parse_worker for Knowledge-Graph ingest.
    routing_key = str(workspace_id) so that a single KG worker
    processes all documents for the same workspace sequentially —
    preventing concurrent writes to the same LightRAG graph files.
    """

    document_id: uuid.UUID
    workspace_id: uuid.UUID
    revision_id: uuid.UUID
    build_profile: str
    # Prefer markdown_s3_key: kg_worker downloads the markdown from MinIO so the
    # broker message stays small (and retry copies stay cheap). The inline
    # `markdown` field remains only for backward compatibility with in-flight /
    # DLQ messages published before markdown_s3_key existed.
    markdown: str = ""
    markdown_s3_key: str | None = None


class MemorySaveMessage(BaseModel):
    """
    Dispatched by the chat endpoints after a turn completes, to persist the
    user's message as a Graphiti personal-memory episode out-of-band.

    Unlike the document pipeline messages this carries no document/workspace —
    it is keyed on the user. The expensive LLM fact-extraction + Neo4j write are
    done in the memory worker (handle_memory), so a transient failure is retried
    durably by RabbitMQ instead of being lost in a fire-and-forget task.
    """

    user_id: uuid.UUID
    user_message: str
    assistant_message: str = ""
    session_id: str | None = None

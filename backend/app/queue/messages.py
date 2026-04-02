"""
Queue Message Schemas
=====================
Pydantic models for every message type passed through RabbitMQ.
All messages carry document_id + workspace_id as primary keys.
"""
from __future__ import annotations

import uuid

from pydantic import BaseModel


class ParseMessage(BaseModel):
    """Dispatched by the API after a file is uploaded."""
    document_id: uuid.UUID
    workspace_id: uuid.UUID
    minio_key: str          # key in hrag-uploads bucket
    original_filename: str


class EmbedMessage(BaseModel):
    """Dispatched by parse_worker once structural parsing is done."""
    document_id: uuid.UUID
    workspace_id: uuid.UUID


class CaptionMessage(BaseModel):
    """Dispatched by parse_worker for image + table captioning."""
    document_id: uuid.UUID
    workspace_id: uuid.UUID


class KGMessage(BaseModel):
    """
    Dispatched by parse_worker for Knowledge-Graph ingest.
    routing_key = str(workspace_id) so that a single KG worker
    processes all documents for the same workspace sequentially —
    preventing concurrent writes to the same LightRAG graph files.
    """
    document_id: uuid.UUID
    workspace_id: uuid.UUID
    markdown: str           # full markdown from parse phase

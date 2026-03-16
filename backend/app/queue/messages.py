"""
Queue Message Schemas
=====================
Pydantic models for every message type passed through RabbitMQ.
All messages carry document_id + workspace_id as primary keys.
"""
from __future__ import annotations

from pydantic import BaseModel


class ParseMessage(BaseModel):
    """Dispatched by the API after a file is uploaded."""
    document_id: int
    workspace_id: int
    minio_key: str          # key in nexusrag-uploads bucket
    original_filename: str


class EmbedMessage(BaseModel):
    """Dispatched by parse_worker once structural parsing is done."""
    document_id: int
    workspace_id: int


class CaptionMessage(BaseModel):
    """Dispatched by parse_worker for image + table captioning."""
    document_id: int
    workspace_id: int


class KGMessage(BaseModel):
    """
    Dispatched by parse_worker for Knowledge-Graph ingest.
    routing_key = str(workspace_id) so that a single KG worker
    processes all documents for the same workspace sequentially —
    preventing concurrent writes to the same LightRAG graph files.
    """
    document_id: int
    workspace_id: int
    markdown: str           # full markdown from parse phase

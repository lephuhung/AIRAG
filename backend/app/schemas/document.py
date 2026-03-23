from pydantic import BaseModel
from datetime import datetime
from typing import Any
from app.models.document import DocumentStatus


class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    file_type: str
    file_size: int


class DocumentCreate(DocumentBase):
    workspace_id: int


class DocumentTypeInfo(BaseModel):
    id: int
    slug: str
    name: str
    model_config = {"from_attributes": True}


class DocumentResponse(DocumentBase):
    id: int
    workspace_id: int
    status: DocumentStatus
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    # HRAG fields
    page_count: int = 0
    image_count: int = 0
    table_count: int = 0
    parser_version: str | None = None
    processing_time_ms: int = 0
    # Digital signature metadata (None if not a native PDF or no signatures found)
    digital_signatures: list[dict[str, Any]] | None = None
    # Document type classification
    document_type_id: int | None = None
    document_type: DocumentTypeInfo | None = None
    # Sub-task completion flags (set independently by each worker after CHUNKING)
    embed_done: bool = False
    captions_done: bool = False
    kg_done: bool = False

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    status: DocumentStatus
    message: str

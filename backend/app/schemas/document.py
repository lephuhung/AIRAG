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
    # Official document reference number
    document_number: str | None = None
    # Document title/subject extracted from header
    document_title: str | None = None
    # Manual signer name override
    signer_name: str | None = None
    # Sub-task completion flags (set independently by each worker after CHUNKING)
    kg_done: bool = False
    location: str | None = None
    issuing_agency: str | None = None
    parent_agency: str | None = None
    published_date: str | None = None

    model_config = {"from_attributes": True}


class DocumentUpdate(BaseModel):
    document_number: str | None = None
    document_title: str | None = None
    signer_name: str | None = None
    published_date: str | None = None
    issuing_agency: str | None = None
    location: str | None = None
    parent_agency: str | None = None


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    status: DocumentStatus
    message: str

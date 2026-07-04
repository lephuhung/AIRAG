"""
RAG-related Pydantic schemas for request/response validation.
"""

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query endpoint."""

    question: str = Field(
        ..., min_length=1, max_length=1000, description="The question to query"
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of chunks to retrieve"
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None, description="Filter to specific document IDs"
    )
    mode: str = Field(
        default="hybrid",
        description="Search mode: hybrid (default), vector_only, naive, local, global",
    )


class CitationResponse(BaseModel):
    """A source citation."""

    source_file: str
    document_id: uuid.UUID
    page_no: int = 0
    heading_path: list[str] = []
    formatted: str = ""

    @field_validator("heading_path", mode="before")
    @classmethod
    def validate_heading_path(cls, v):
        """Coerce string heading_path to list."""
        if isinstance(v, str):
            if not v:
                return []
            if ">" in v:
                return [p.strip() for p in v.split(">")]
            return [v]
        return v or []


class RetrievedChunkResponse(BaseModel):
    """Response schema for a single retrieved chunk."""

    content: str
    chunk_id: str
    score: float
    metadata: dict
    citation: CitationResponse | None = None

    model_config = {"from_attributes": True}


class DocumentImageResponse(BaseModel):
    """Response schema for a document image."""

    image_id: str
    document_id: uuid.UUID
    page_no: int
    caption: str = ""
    width: int = 0
    height: int = 0
    url: str = ""


class DocumentBrief(BaseModel):
    """Minimal document metadata for chat message attachments."""

    id: uuid.UUID
    filename: str
    original_filename: str
    file_type: str
    status: str
    document_number: str | None = None

    model_config = {"from_attributes": True}


class RAGQueryResponse(BaseModel):
    """Response schema for RAG query."""

    query: str
    chunks: list[RetrievedChunkResponse]
    context: str
    total_chunks: int
    knowledge_graph_summary: str = ""
    citations: list[CitationResponse] = []
    image_refs: list[DocumentImageResponse] = []


class DocumentProcessRequest(BaseModel):
    """Request schema for document processing."""

    document_id: uuid.UUID


class DocumentProcessResponse(BaseModel):
    """Response schema for document processing."""

    document_id: uuid.UUID
    status: str
    chunk_count: int
    message: str


class BatchProcessRequest(BaseModel):
    """Request schema for batch document processing."""

    document_ids: list[uuid.UUID] = Field(
        ..., min_length=1, description="List of document IDs to process"
    )


class ProjectRAGStatsResponse(BaseModel):
    """Response schema for workspace RAG statistics."""

    workspace_id: uuid.UUID
    total_documents: int
    indexed_documents: int
    total_chunks: int
    image_count: int = 0
    hrag_documents: int = 0


# ---------------------------------------------------------------------------
# Knowledge Graph schemas
# ---------------------------------------------------------------------------


class KGEntityResponse(BaseModel):
    """A knowledge graph entity (node)."""

    name: str
    entity_type: str = "Unknown"
    description: str = ""
    degree: int = 0  # number of relationships


class KGRelationshipResponse(BaseModel):
    """A knowledge graph relationship (edge)."""

    source: str
    target: str
    description: str = ""
    keywords: str = ""
    weight: float = 1.0


class KGGraphNodeResponse(BaseModel):
    """Node in the graph visualization payload."""

    id: str
    label: str
    entity_type: str = "Unknown"
    degree: int = 0


class KGGraphEdgeResponse(BaseModel):
    """Edge in the graph visualization payload."""

    source: str
    target: str
    label: str = ""
    weight: float = 1.0


class KGGraphResponse(BaseModel):
    """Full graph export for frontend visualization."""

    nodes: list[KGGraphNodeResponse] = []
    edges: list[KGGraphEdgeResponse] = []
    is_truncated: bool = False


class KGAnalyticsResponse(BaseModel):
    """Knowledge Graph analytics summary."""

    entity_count: int = 0
    relationship_count: int = 0
    entity_types: dict[str, int] = {}  # type → count
    top_entities: list[KGEntityResponse] = []  # top N by degree
    avg_degree: float = 0.0


class DocumentBreakdownItem(BaseModel):
    """Per-document breakdown for analytics."""

    document_id: uuid.UUID
    filename: str
    chunk_count: int = 0
    image_count: int = 0
    page_count: int = 0
    file_size: int = 0
    status: str = "pending"


class ProjectAnalyticsResponse(BaseModel):
    """Extended project analytics."""

    stats: ProjectRAGStatsResponse
    kg_analytics: KGAnalyticsResponse | None = None
    document_breakdown: list[DocumentBreakdownItem] = []


# ---------------------------------------------------------------------------
# Chat schemas
# ---------------------------------------------------------------------------


class ChatMessageSchema(BaseModel):
    """A single chat message in conversation history."""

    role: str = Field(..., description="user or assistant")
    content: str


class ChatRequest(BaseModel):
    """Request for the chat endpoint."""

    message: str = Field(..., min_length=1, max_length=5000)
    history: list[ChatMessageSchema] = []
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation context. If provided, exchange summaries will be used instead of history[]",
    )
    document_ids: list[uuid.UUID] | None = None
    enable_thinking: bool = False
    force_search: bool = (
        False  # Pre-search before LLM call; injects sources as context directly
    )


class ChatSourceChunk(BaseModel):
    """A source chunk referenced in the chat answer."""

    index: str  # 4-char alphanumeric ID, e.g. "id12" (was: int)
    chunk_id: str

    @field_validator("index", mode="before")
    @classmethod
    def coerce_index_to_str(cls, v):
        return str(v) if not isinstance(v, str) else v

    @field_validator("heading_path", mode="before")
    @classmethod
    def coerce_heading_path(cls, v):
        if isinstance(v, str):
            if not v: return []
            return v.split(" > ")
        return v or []

    @field_validator("document_id", mode="before")
    @classmethod
    def coerce_document_id(cls, v):
        if isinstance(v, str):
            try:
                return uuid.UUID(v)
            except ValueError:
                return v
        return v

    @field_validator("chunk_id", mode="before")
    @classmethod
    def coerce_chunk_id(cls, v, info):
        # Handle legacy format with chunk_index or no chunk_id
        if v is not None and v != "":
            return v
        # Fallback: try to construct from chunk_index if present in data
        data = info.data if hasattr(info, 'data') else {}
        chunk_idx = data.get("chunk_index")
        if chunk_idx is not None:
            return f"chunk_{chunk_idx}"
        doc_id = data.get("document_id", "")
        return f"chunk_{doc_id}_{chunk_idx}" if chunk_idx else "unknown"

    content: str = ""
    chunk_id: str = ""
    document_id: uuid.UUID
    page_no: int = 0
    heading_path: list[str] = []
    score: float = 0.0
    source_type: str = "vector"  # "vector" | "kg"
    source_file: str | None = None
    # Citation cấp điều khoản: hiển thị "Điều 17 — 85/2016/NĐ-CP" thay vì tên file
    document_number: str | None = None
    article_label: str | None = None
    # Hiệu lực pháp lý của văn bản nguồn (badge cảnh báo ở SourcesPanel):
    # "effective" | "superseded" | "partially_amended" | "unknown" | None
    validity_status: str | None = None
    superseded_by: str | None = None


class ChatImageRef(BaseModel):
    """An image referenced in the chat answer."""

    @field_validator("document_id", mode="before")
    @classmethod
    def coerce_document_id(cls, v):
        if isinstance(v, str):
            try:
                return uuid.UUID(v)
            except ValueError:
                return v
        return v

    ref_id: str | None = None  # 4-char alphanumeric ID, e.g. "p4f2"
    image_id: str
    document_id: uuid.UUID
    page_no: int = 0
    caption: str = ""
    url: str = ""
    width: int = 0
    height: int = 0


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""

    answer: str
    sources: list[ChatSourceChunk] = []
    related_entities: list[str] = []
    kg_summary: str | None = None
    image_refs: list[ChatImageRef] = []
    thinking: str | None = None
    potential_abbreviations: list[str] | None = None


class PersistedChatMessage(BaseModel):
    """A persisted chat message from the database."""

    @field_validator("document_ids", mode="before")
    @classmethod
    def coerce_doc_ids(cls, v):
        if v is None: return None
        if isinstance(v, str):
            # Try to parse as JSON if it's a string
            import json
            try:
                v = json.loads(v)
            except:
                return None
        if isinstance(v, list):
            out = []
            for item in v:
                if isinstance(item, str):
                    try:
                        out.append(uuid.UUID(item))
                    except ValueError:
                        continue
                elif isinstance(item, uuid.UUID):
                    out.append(item)
            return out
        return None

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, v):
        """Handle legacy source formats from older messages."""
        if v is None: return None
        if not isinstance(v, list):
            return None
        normalized = []
        for item in v:
            if not isinstance(item, dict):
                continue
            # Normalize legacy format to ChatSourceChunk format
            normalized_item = {
                "index": str(item.get("index", item.get("chunk_index", "???"))),
                "chunk_id": item.get("chunk_id") or f"chunk_{item.get('chunk_index', 'unknown')}",
                "content": item.get("content", ""),
                "document_id": item.get("document_id", ""),
                "page_no": item.get("page_no", 0),
                "heading_path": item.get("heading_path", []),
                "score": item.get("score", 0.0),
                "source_type": item.get("source_type", "vector"),
                "source_file": item.get("source_file") or item.get("source"),
            }
            normalized.append(normalized_item)
        return normalized if normalized else None

    id: uuid.UUID
    message_id: str
    role: str
    content: str
    document_ids: list[uuid.UUID] | None = None
    attached_docs: list[DocumentBrief] | None = None
    sources: list[ChatSourceChunk] | None = None
    related_entities: list[str] | None = None
    image_refs: list[ChatImageRef] | None = None
    thinking: str | None = None
    agent_steps: list | None = None
    potential_abbreviations: list[str] | None = None
    people_data: list[dict] | None = None
    created_at: str  # ISO format

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    """Response for GET chat history."""

    session_id: str
    messages: list[PersistedChatMessage]
    total: int


class SessionDocumentsResponse(BaseModel):
    """Response for GET session documents (for @mention autocomplete)."""

    documents: list[DocumentBrief]


class RateSourceRequest(BaseModel):
    """Request to rate a source citation."""

    message_id: str = Field(..., description="The message_id containing the source")
    source_index: str = Field(..., description="Source citation ID, e.g. 'id12'")
    rating: Literal["relevant", "partial", "not_relevant"] = Field(
        ..., description="Source rating"
    )


class RateSourceResponse(BaseModel):
    """Response after rating a source."""

    success: bool
    message_id: str
    ratings: dict[str, str]


class LLMCapabilitiesResponse(BaseModel):
    """Response for LLM capabilities check."""

    provider: str
    model: str
    supports_thinking: bool
    supports_vision: bool
    thinking_default: bool = True


# ---------------------------------------------------------------------------
# Debug / QA schemas
# ---------------------------------------------------------------------------


class DebugRetrievedSource(BaseModel):
    """A retrieved source for debug inspection."""

    index: str  # 4-char alphanumeric ID (was: int)
    document_id: uuid.UUID

    @field_validator("index", mode="before")
    @classmethod
    def coerce_index_to_str(cls, v):
        return str(v) if not isinstance(v, str) else v

    page_no: int
    heading_path: list[str] = []
    source_file: str = ""
    content_preview: str = ""  # first 500 chars
    score: float = 0.0
    source_type: str = "vector"


class DebugChatResponse(BaseModel):
    """Full debug response — retrieval + LLM answer for quality inspection."""

    # Query
    question: str
    workspace_id: uuid.UUID

    # Retrieval
    retrieved_sources: list[DebugRetrievedSource] = []
    kg_summary: str = ""
    total_sources: int = 0

    # LLM
    system_prompt: str = ""
    answer: str = ""
    thinking: str | None = None

    # Images
    image_count: int = 0

    # Meta
    provider: str = ""
    model: str = ""

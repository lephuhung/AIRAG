"""
RAG API endpoints for document querying and retrieval.
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.deps import get_db, get_current_active_user
from app.core.deps import verify_workspace_access as _verify_workspace_access
from app.core.exceptions import NotFoundError
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentImage, DocumentStatus
from app.models.user import User
import logging

from app.schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedChunkResponse,
    CitationResponse,
    DocumentImageResponse,
    DocumentProcessRequest,
    DocumentProcessResponse,
    BatchProcessRequest,
    ProjectRAGStatsResponse,
    KGEntityResponse,
    KGRelationshipResponse,
    KGGraphResponse,
    KGGraphNodeResponse,
    KGGraphEdgeResponse,
    KGAnalyticsResponse,
    DocumentBreakdownItem,
    ProjectAnalyticsResponse,
    ChatRequest,
    ChatResponse,
    ChatSourceChunk,
    ChatImageRef,
    PersistedChatMessage,
    ChatHistoryResponse,
    LLMCapabilitiesResponse,
    DebugRetrievedSource,
    DebugChatResponse,
    RateSourceRequest,
)

logger = logging.getLogger(__name__)
import string, random
from app.services.rag_service import get_rag_service

# In-progress statuses — documents currently in the pipeline
_IN_PROGRESS = (
    DocumentStatus.PARSING,
    DocumentStatus.OCRING,
    DocumentStatus.CHUNKING,
    DocumentStatus.EMBEDDING,
    DocumentStatus.BUILDING_KG,
)

# ---------------------------------------------------------------------------
# Citation ID generation — 4-char alphanumeric IDs matching PageIndex format
# ---------------------------------------------------------------------------
_CITATION_ID_CHARS = string.ascii_lowercase + string.digits


def _generate_citation_id(existing: set[str]) -> str:
    """Generate a unique 4-char alphanumeric citation ID.

    Always contains at least one letter so it cannot be confused with
    old-style numeric indices (e.g. "1", "23").
    """
    while True:
        cid = "".join(random.choices(_CITATION_ID_CHARS, k=4))
        if any(c.isalpha() for c in cid) and cid not in existing:
            return cid

router = APIRouter(prefix="/rag", tags=["rag"])

UPLOAD_DIR = "uploads"

# Prompt constants — see chat_prompt.py for full documentation
from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT


async def verify_workspace_access(
    workspace_id: int,
    db: AsyncSession,
    user: User | None = None,
) -> KnowledgeBase:
    """Verify knowledge base exists and user has access."""
    return await _verify_workspace_access(workspace_id, user, db)


# Convenience: create a standard user dep for all RAG endpoints
_user_dep = Depends(get_current_active_user)


@router.post("/query/{workspace_id}", response_model=RAGQueryResponse)
async def query_documents(
    workspace_id: int,
    request: RAGQueryRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Query indexed documents using semantic search (+ optional KG)."""
    await verify_workspace_access(workspace_id, db, user)

    rag_service = get_rag_service(db, workspace_id)

    # Try deep query if available
    from app.services.nexus_rag_service import NexusRAGService
    if isinstance(rag_service, NexusRAGService) and request.mode != "vector_only":
        result = await rag_service.query_deep(
            question=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
            mode=request.mode,
        )

        chunks_response = []
        for i, chunk in enumerate(result.chunks):
            citation = result.citations[i] if i < len(result.citations) else None
            citation_resp = None
            if citation:
                citation_resp = CitationResponse(
                    source_file=citation.source_file,
                    document_id=citation.document_id,
                    page_no=citation.page_no,
                    heading_path=citation.heading_path,
                    formatted=citation.format(),
                )
            chunks_response.append(RetrievedChunkResponse(
                content=chunk.content,
                chunk_id=f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}",
                score=0.0,
                metadata={
                    "source": chunk.source_file,
                    "page_no": chunk.page_no,
                    "heading_path": " > ".join(chunk.heading_path),
                },
                citation=citation_resp,
            ))

        image_refs = [
            DocumentImageResponse(
                image_id=img.image_id,
                document_id=img.document_id,
                page_no=img.page_no,
                caption=img.caption,
                width=img.width,
                height=img.height,
                url=f"/static/doc-images/kb_{workspace_id}/images/{img.image_id}.png",
            )
            for img in result.image_refs
        ]

        citations = [
            CitationResponse(
                source_file=c.source_file,
                document_id=c.document_id,
                page_no=c.page_no,
                heading_path=c.heading_path,
                formatted=c.format(),
            )
            for c in result.citations
        ]

        return RAGQueryResponse(
            query=result.query,
            chunks=chunks_response,
            context=result.context,
            total_chunks=len(result.chunks),
            knowledge_graph_summary=result.knowledge_graph_summary,
            citations=citations,
            image_refs=image_refs,
        )

    # Fallback: legacy sync query
    result = rag_service.query(
        question=request.question,
        top_k=request.top_k,
        document_ids=request.document_ids
    )

    return RAGQueryResponse(
        query=result.query,
        chunks=[
            RetrievedChunkResponse(
                content=chunk.content,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
                metadata=chunk.metadata
            )
            for chunk in result.chunks
        ],
        context=result.context,
        total_chunks=len(result.chunks)
    )


@router.post("/process/{document_id}", response_model=DocumentProcessResponse)
async def process_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Trigger document processing (parsing + indexing) as a background task."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status in _IN_PROGRESS:
        # Allow re-trigger if embed never ran (e.g. message was dropped by RabbitMQ).
        # A truly in-progress document will have embed_done or captions_done progressing.
        truly_in_progress = document.status in (DocumentStatus.PARSING, DocumentStatus.OCRING) or document.embed_done
        if truly_in_progress:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document is already being processed"
            )
        # embed_done=False while CHUNKING/EMBEDDING → worker message was lost, allow retry
        # Fall through to reset + requeue below

    if document.status == DocumentStatus.INDEXED:
        return DocumentProcessResponse(
            document_id=document_id,
            status=document.status.value,
            chunk_count=document.chunk_count,
            message="Document is already indexed"
        )

    from pathlib import Path
    file_path = Path(UPLOAD_DIR) / document.filename

    if not file_path.exists() and not document.upload_s3_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found — no local file and no MinIO key"
        )

    # Reset sub-task flags and queue via RabbitMQ (same flow as upload)
    document.status = DocumentStatus.PENDING
    document.error_message = None
    document.embed_done = False
    document.captions_done = False
    document.kg_done = False
    await db.commit()

    if document.upload_s3_key:
        # New flow: re-publish parse task so workers pick it up from MinIO
        try:
            from app.queue.publisher import publish_parse_task
            await publish_parse_task(
                document_id=document_id,
                workspace_id=document.workspace_id,
                minio_key=document.upload_s3_key,
                original_filename=document.original_filename,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to queue document for processing: {str(e)}"
            )
    else:
        # Legacy fallback: file still on disk, process inline
        from app.api.documents import process_document_background
        import asyncio
        asyncio.get_event_loop().create_task(
            process_document_background(document_id, str(file_path), document.workspace_id)
        )

    return DocumentProcessResponse(
        document_id=document_id,
        status="pending",
        chunk_count=0,
        message="Document queued for processing."
    )


@router.post("/process-batch")
async def process_batch(
    request: BatchProcessRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """
    Process multiple documents sequentially in the background.
    Publishes each document to the parse queue (RabbitMQ).
    """
    accepted_ids = []
    skipped_ids = []

    for doc_id in request.document_ids:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc is None:
            skipped_ids.append(doc_id)
            continue

        # Skip documents already in the pipeline
        if doc.status in _IN_PROGRESS:
            skipped_ids.append(doc_id)
            continue

        if not doc.upload_s3_key:
            skipped_ids.append(doc_id)
            continue

        doc.status = DocumentStatus.PENDING
        doc.error_message = None
        doc.embed_done = False
        doc.captions_done = False
        doc.kg_done = False
        accepted_ids.append((doc_id, doc.upload_s3_key, doc.workspace_id, doc.original_filename))

    await db.commit()

    if accepted_ids:
        from app.queue.publisher import publish_parse_task
        for doc_id, minio_key, workspace_id, original_filename in accepted_ids:
            try:
                await publish_parse_task(
                    document_id=doc_id,
                    workspace_id=workspace_id,
                    minio_key=minio_key,
                    original_filename=original_filename,
                )
            except Exception as e:
                logger.error(f"[process_batch] Failed to queue doc {doc_id}: {e}")

    return {
        "message": f"Processing {len(accepted_ids)} document(s)",
        "accepted": [a[0] for a in accepted_ids],
        "skipped": skipped_ids,
    }


@router.post("/reindex/{document_id}", response_model=DocumentProcessResponse)
async def reindex_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Re-process an existing document through the NexusRAG pipeline."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status in _IN_PROGRESS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is currently being processed"
        )

    if not document.upload_s3_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original file not found in MinIO — cannot reindex"
        )

    rag_service = get_rag_service(db, document.workspace_id)

    # Delete existing data first
    try:
        await rag_service.delete_document(document_id)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to delete old data for reindex: {e}")

    # Delete old markdown object from MinIO
    if document.markdown_s3_key:
        try:
            from app.services.storage_service import get_storage_service
            await get_storage_service().delete_markdown(document.markdown_s3_key)
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to delete MinIO object for reindex of doc {document_id}: {e}"
            )

    # Reset document metadata
    document.status = DocumentStatus.PENDING
    document.chunk_count = 0
    document.markdown_s3_key = None
    document.image_count = 0
    document.table_count = 0
    document.embed_done = False
    document.captions_done = False
    document.kg_done = False
    document.parser_version = None
    document.error_message = None
    await db.commit()

    # Re-publish to parse queue — worker will download from MinIO
    try:
        from app.queue.publisher import publish_parse_task
        await publish_parse_task(
            document_id=document_id,
            workspace_id=document.workspace_id,
            minio_key=document.upload_s3_key,
            original_filename=document.original_filename,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue document for reindex: {str(e)}"
        )

    return DocumentProcessResponse(
        document_id=document_id,
        status="pending",
        chunk_count=0,
        message="Document queued for re-processing"
    )


@router.post("/reindex-workspace/{workspace_id}")
async def reindex_workspace(
    workspace_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """
    Reindex ALL documents in a workspace.
    Deletes the old vector collection (handles embedding dimension changes)
    and re-processes every document through the NexusRAG pipeline.
    Runs in background — returns immediately with document count.
    """
    await verify_workspace_access(workspace_id, db, user)

    # Find all documents in this workspace
    result = await db.execute(
        select(Document).where(
            Document.workspace_id == workspace_id,
            Document.status.notin_(list(_IN_PROGRESS)),
        )
    )
    documents = list(result.scalars().all())

    if not documents:
        return {"message": "No documents to reindex", "document_count": 0}

    # Delete old vector collection (required when embedding dimensions change)
    try:
        from app.services.vector_store import get_vector_store
        vs = get_vector_store(workspace_id)
        vs.delete_collection()
        logger.info(f"Deleted old vector collection for workspace {workspace_id}")
    except Exception as e:
        logger.warning(f"Failed to delete old collection: {e}")

    async def _reindex_all(doc_ids: list[int], ws_id: int):
        """Background task: reindex each document sequentially via RabbitMQ."""
        from app.core.database import AsyncSessionLocal
        from app.queue.publisher import publish_parse_task as _publish
        async with AsyncSessionLocal() as session:
            rag_service = get_rag_service(session, ws_id)
            for did in doc_ids:
                try:
                    res = await session.execute(
                        select(Document).where(Document.id == did)
                    )
                    doc = res.scalar_one_or_none()
                    if not doc:
                        continue

                    if not doc.upload_s3_key:
                        logger.warning(f"Skipping doc {did}: no upload_s3_key in MinIO")
                        continue

                    # Delete old chunk data for this document
                    try:
                        await rag_service.delete_document(did)
                    except Exception:
                        pass

                    # Reset metadata
                    doc.status = DocumentStatus.PENDING
                    doc.chunk_count = 0
                    doc.image_count = 0
                    doc.embed_done = False
                    doc.captions_done = False
                    doc.kg_done = False
                    doc.error_message = None
                    await session.commit()

                    # Publish to parse queue — worker downloads from MinIO
                    await _publish(
                        document_id=did,
                        workspace_id=ws_id,
                        minio_key=doc.upload_s3_key,
                        original_filename=doc.original_filename,
                    )
                    logger.info(f"Reindex queued for document {did} in workspace {ws_id}")
                except Exception as e:
                    logger.error(f"Failed to queue reindex for document {did}: {e}")

    doc_ids = [d.id for d in documents]
    background_tasks.add_task(_reindex_all, doc_ids, workspace_id)

    return {
        "message": f"Reindexing {len(doc_ids)} documents in background",
        "document_count": len(doc_ids),
        "document_ids": doc_ids,
    }


@router.get("/stats/{workspace_id}", response_model=ProjectRAGStatsResponse)
async def get_workspace_rag_stats(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get RAG statistics for a knowledge base."""
    await verify_workspace_access(workspace_id, db, user)

    total_result = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == workspace_id)
    )
    total_documents = total_result.scalar() or 0

    indexed_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == workspace_id,
            Document.status.in_([DocumentStatus.INDEXED, DocumentStatus.BUILDING_KG])
        )
    )
    indexed_documents = indexed_result.scalar() or 0

    # Count NexusRAG documents (parser_version = 'docling')
    nexusrag_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == workspace_id,
            Document.parser_version == "docling"
        )
    )
    nexusrag_documents = nexusrag_result.scalar() or 0

    # Count total images
    image_result = await db.execute(
        select(func.count(DocumentImage.id))
        .join(Document, DocumentImage.document_id == Document.id)
        .where(Document.workspace_id == workspace_id)
    )
    image_count = image_result.scalar() or 0

    rag_service = get_rag_service(db, workspace_id)
    try:
        total_chunks = rag_service.get_chunk_count()
    except Exception:
        total_chunks = 0

    return ProjectRAGStatsResponse(
        workspace_id=workspace_id,
        total_documents=total_documents,
        indexed_documents=indexed_documents,
        total_chunks=total_chunks,
        image_count=image_count,
        nexusrag_documents=nexusrag_documents,
    )


@router.get("/chunks/{document_id}")
async def get_document_chunks(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get all chunks for a specific document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status not in (DocumentStatus.INDEXED, DocumentStatus.BUILDING_KG):
        return {
            "document_id": document_id,
            "status": document.status.value,
            "chunks": [],
            "message": "Document is not yet indexed"
        }

    rag_service = get_rag_service(db, document.workspace_id)

    chunk_ids = [f"doc_{document_id}_chunk_{i}" for i in range(document.chunk_count)]

    try:
        results = rag_service.vector_store.get_by_ids(chunk_ids)

        chunks = []
        for i in range(len(results.get("ids", []))):
            chunks.append({
                "chunk_id": results["ids"][i],
                "content": results["documents"][i] if results.get("documents") else None,
                "metadata": results["metadatas"][i] if results.get("metadatas") else {}
            })

        return {
            "document_id": document_id,
            "status": document.status.value,
            "chunk_count": document.chunk_count,
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Knowledge Graph exploration endpoints (Phase 9)
# ---------------------------------------------------------------------------

# Module-level cache: workspace_id → KnowledgeGraphService
# Avoids re-initializing LightRAG (and reloading the embedding model) on every request.
_kg_service_cache: dict[int, "KnowledgeGraphService"] = {}


async def _get_kg_service(workspace_id: int):
    """Get KnowledgeGraphService for a knowledge base — cached per workspace."""
    from app.services.knowledge_graph_service import KnowledgeGraphService
    if workspace_id not in _kg_service_cache:
        _kg_service_cache[workspace_id] = KnowledgeGraphService(workspace_id)
    return _kg_service_cache[workspace_id]


@router.get("/entities/{workspace_id}", response_model=list[KGEntityResponse])
async def get_kg_entities(
    workspace_id: int,
    search: str | None = None,
    entity_type: str | None = None,
    limit: int = 200,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """List entities in the workspace's knowledge graph."""
    await verify_workspace_access(workspace_id, db, user)
    kg = await _get_kg_service(workspace_id)
    try:
        entities = await kg.get_entities(
            search=search, entity_type=entity_type, limit=limit, offset=offset
        )
        return [KGEntityResponse(**e) for e in entities]
    except Exception as e:
        logger.error(f"Failed to get KG entities for workspace {workspace_id}: {e}")
        return []


@router.get("/relationships/{workspace_id}", response_model=list[KGRelationshipResponse])
async def get_kg_relationships(
    workspace_id: int,
    entity: str | None = None,
    limit: int = 500,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """List relationships in the workspace's knowledge graph."""
    await verify_workspace_access(workspace_id, db, user)
    kg = await _get_kg_service(workspace_id)
    try:
        rels = await kg.get_relationships(entity_name=entity, limit=limit)
        return [KGRelationshipResponse(**r) for r in rels]
    except Exception as e:
        logger.error(f"Failed to get KG relationships for workspace {workspace_id}: {e}")
        return []


@router.get("/graph/{workspace_id}", response_model=KGGraphResponse)
async def get_kg_graph(
    workspace_id: int,
    center: str | None = None,
    max_depth: int = 3,
    max_nodes: int = 150,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Export knowledge graph data for frontend visualization."""
    await verify_workspace_access(workspace_id, db, user)
    kg = await _get_kg_service(workspace_id)
    try:
        data = await kg.get_graph_data(
            center_entity=center, max_depth=max_depth, max_nodes=max_nodes
        )
        return KGGraphResponse(
            nodes=[KGGraphNodeResponse(**n) for n in data["nodes"]],
            edges=[KGGraphEdgeResponse(**e) for e in data["edges"]],
            is_truncated=data.get("is_truncated", False),
        )
    except Exception as e:
        logger.error(f"Failed to export KG graph for workspace {workspace_id}: {e}")
        return KGGraphResponse()


@router.get("/analytics/{workspace_id}", response_model=ProjectAnalyticsResponse)
async def get_workspace_analytics(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get extended analytics for a knowledge base (stats + KG + per-doc breakdown)."""
    await verify_workspace_access(workspace_id, db, user)

    # Base stats
    total_result = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == workspace_id)
    )
    total_documents = total_result.scalar() or 0

    indexed_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == workspace_id,
            Document.status.in_([DocumentStatus.INDEXED, DocumentStatus.BUILDING_KG]),
        )
    )
    indexed_documents = indexed_result.scalar() or 0

    nexusrag_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == workspace_id,
            Document.parser_version == "docling",
        )
    )
    nexusrag_documents = nexusrag_result.scalar() or 0

    image_result = await db.execute(
        select(func.count(DocumentImage.id))
        .join(Document, DocumentImage.document_id == Document.id)
        .where(Document.workspace_id == workspace_id)
    )
    image_count = image_result.scalar() or 0

    rag_service = get_rag_service(db, workspace_id)
    try:
        total_chunks = rag_service.get_chunk_count()
    except Exception:
        total_chunks = 0

    stats = ProjectRAGStatsResponse(
        workspace_id=workspace_id,
        total_documents=total_documents,
        indexed_documents=indexed_documents,
        total_chunks=total_chunks,
        image_count=image_count,
        nexusrag_documents=nexusrag_documents,
    )

    # KG analytics (optional — only if NexusRAG active)
    kg_analytics = None
    if nexusrag_documents > 0:
        try:
            kg = await _get_kg_service(workspace_id)
            analytics_data = await kg.get_analytics()
            kg_analytics = KGAnalyticsResponse(
                entity_count=analytics_data["entity_count"],
                relationship_count=analytics_data["relationship_count"],
                entity_types=analytics_data["entity_types"],
                top_entities=[KGEntityResponse(**e) for e in analytics_data["top_entities"]],
                avg_degree=analytics_data["avg_degree"],
            )
        except Exception as e:
            logger.warning(f"Failed to get KG analytics for workspace {workspace_id}: {e}")

    # Per-document breakdown
    doc_result = await db.execute(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
    )
    documents = doc_result.scalars().all()
    breakdown = [
        DocumentBreakdownItem(
            document_id=d.id,
            filename=d.original_filename,
            chunk_count=d.chunk_count,
            image_count=d.image_count or 0,
            page_count=d.page_count or 0,
            file_size=d.file_size,
            status=d.status.value if hasattr(d.status, "value") else str(d.status),
        )
        for d in documents
    ]

    return ProjectAnalyticsResponse(
        stats=stats,
        kg_analytics=kg_analytics,
        document_breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# LLM Capabilities endpoint
# ---------------------------------------------------------------------------

@router.get("/capabilities", response_model=LLMCapabilitiesResponse)
async def get_llm_capabilities(
    user: User = Depends(get_current_active_user),
):
    """Check LLM provider capabilities (thinking, vision)."""
    from app.services.llm import get_llm_provider
    from app.core.config import settings

    provider = get_llm_provider()
    provider_name = settings.LLM_PROVIDER.lower()

    # Per-provider thinking default:
    # Gemini: thinking ON by default (fast, cloud-based)
    # Ollama: thinking OFF by default (slow on local hardware), configurable via OLLAMA_ENABLE_THINKING
    if provider_name == "ollama":
        thinking_default = settings.OLLAMA_ENABLE_THINKING
    else:
        thinking_default = provider.supports_thinking()

    return LLMCapabilitiesResponse(
        provider=settings.LLM_PROVIDER,
        model=settings.OLLAMA_MODEL if provider_name == "ollama" else settings.LLM_MODEL_FAST,
        supports_thinking=provider.supports_thinking(),
        supports_vision=provider.supports_vision(),
        thinking_default=thinking_default,
    )


# ---------------------------------------------------------------------------
# Debug endpoint — inspect retrieval + LLM answer quality
# ---------------------------------------------------------------------------

@router.post("/debug-chat/{workspace_id}", response_model=DebugChatResponse)
async def debug_chat(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """
    Debug version of chat — returns retrieval details + system prompt + answer
    so you can inspect what the LLM received vs what it answered.
    """
    kb = await verify_workspace_access(workspace_id, db, user)

    rag_service = get_rag_service(db, workspace_id)

    # -- 1. Retrieve --
    chunks = []
    citations = []
    kg_summary = ""

    from app.services.nexus_rag_service import NexusRAGService
    if isinstance(rag_service, NexusRAGService):
        result = await rag_service.query_deep(
            question=request.message,
            top_k=8,
            document_ids=request.document_ids,
            mode="hybrid",
            include_images=False,
        )
        chunks = result.chunks
        citations = result.citations
        kg_summary = result.knowledge_graph_summary

    # -- 2. Build sources + context (same logic as chat endpoint) --
    debug_used_ids: set[str] = set()
    debug_sources: list[DebugRetrievedSource] = []
    context_parts = []
    for i, chunk in enumerate(chunks):
        citation = citations[i] if i < len(citations) else None
        cid = _generate_citation_id(debug_used_ids)
        debug_used_ids.add(cid)
        debug_sources.append(DebugRetrievedSource(
            index=cid,
            document_id=chunk.document_id,
            page_no=chunk.page_no,
            heading_path=chunk.heading_path,
            source_file=citation.source_file if citation else "",
            content_preview=chunk.content[:500],
            score=0.0,
            source_type="vector",
        ))
        meta_parts = []
        if citation:
            meta_parts.append(citation.source_file)
            if citation.page_no:
                meta_parts.append(f"page {citation.page_no}")
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""
        if heading:
            meta_parts.append(heading)
        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
        context_parts.append(f"Source [{cid}]{meta_line}:\n{chunk.content}")

    # NOTE: KG summary NOT added as citable source (can contain hallucinated data)
    context = "\n\n---\n\n".join(context_parts)

    # -- 3. Build prompt (same architecture as chat endpoint) --
    # SHORT system prompt + sources/rules in USER MESSAGE
    sys_prompt = (kb.system_prompt or DEFAULT_SYSTEM_PROMPT) + HARD_SYSTEM_PROMPT

    # Build user message: CONTEXT → RULES → QUESTION
    user_parts: list[str] = []
    user_parts.append("I have retrieved the following document sources for you.\n")
    user_parts.append("=== DOCUMENT SOURCES ===")
    user_parts.append(context)
    user_parts.append("=== END SOURCES ===\n")

    user_parts.append(
        "IMPORTANT INSTRUCTIONS:\n"
        "- CRITICAL: Read EVERY source carefully before answering. The answer often "
        "requires combining data from MULTIPLE sources. Do NOT skip any source.\n"
        "- TABLE DATA: Sources contain table data as 'Key, Year = Value' pairs. "
        "You MUST extract the actual values. "
        "Example: 'ROE, 2023 = 12,8%. ROE, 2024 = 15,6%' means ROE was 12.8% in 2023 "
        "and 15.6% in 2024. Report these numbers in your answer.\n"
        "- Use the DOCUMENT SOURCES above to answer. Do NOT add outside knowledge.\n"
        "- You MAY compare, synthesize, and reason across multiple sources.\n"
        "- Cite every fact using the source IDs shown in brackets, e.g. [a3x9][b2m7] — one ID per bracket.\n"
        "- For images: [IMG-p4f2][IMG-q7r3] — use the IDs shown in the image list.\n"
        "- NEVER say 'không có thông tin' or 'no information' for data that IS present "
        "in any source. If a source contains 'Key = Value', report that value.\n"
        "- Only say information is unavailable when you have checked ALL sources "
        "and none contains the answer.\n"
        "- If no source is relevant at all, say: "
        "\"Tài liệu không chứa thông tin này.\" without any citations.\n"
        "- Answer in the same language as my question.\n"
    )

    # Conversation context recap (if history exists)
    if request.history:
        last_exchange = request.history[-2:]
        recap_parts = []
        for msg in last_exchange:
            prefix = "User" if msg.role == "user" else "Assistant"
            recap_parts.append(f"{prefix}: {msg.content[:300]}")
        user_parts.append(
            "CONVERSATION CONTEXT (previous exchange):\n"
            + "\n".join(recap_parts) + "\n"
        )

    user_parts.append(f"My question: {request.message}")
    user_content = "\n".join(user_parts)

    # -- 4. Call LLM --
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage, LLMResult

    provider = get_llm_provider()

    messages: list[LLMMessage] = []
    for msg in request.history[-10:]:
        role = "user" if msg.role == "user" else "assistant"
        messages.append(LLMMessage(role=role, content=msg.content))
    messages.append(LLMMessage(role="user", content=user_content))

    answer = ""
    thinking_text: str | None = None
    try:
        llm_result = await provider.acomplete(
            messages,
            system_prompt=sys_prompt,
            temperature=0.1,
            max_tokens=4096,
            think=request.enable_thinking,
        )
        if isinstance(llm_result, LLMResult):
            answer = llm_result.content
            thinking_text = llm_result.thinking or None
        else:
            answer = llm_result
        # Strip Gemini token artifacts (e.g. <unused778>:)
        import re
        answer = re.sub(r'<unused\d+>:?\s*', '', answer).strip()
    except Exception as e:
        answer = f"LLM error: {e}"

    from app.core.config import settings as _s
    return DebugChatResponse(
        question=request.message,
        workspace_id=workspace_id,
        retrieved_sources=debug_sources,
        kg_summary=kg_summary,
        total_sources=len(debug_sources),
        system_prompt=f"[SYSTEM]: {sys_prompt}\n\n[USER MESSAGE]:\n{user_content}",
        answer=answer,
        thinking=thinking_text,
        image_count=0,
        provider=_s.LLM_PROVIDER,
        model=_s.OLLAMA_MODEL if _s.LLM_PROVIDER == "ollama" else _s.LLM_MODEL_FAST,
    )

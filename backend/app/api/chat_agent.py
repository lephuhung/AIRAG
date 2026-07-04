"""
Shared RAG / SSE helpers for HRAG
=================================

Originally the legacy semi-agentic chat loop lived here; that path has been
removed in favour of the LangGraph supervisor. This module now only hosts the
helpers still shared across the codebase:

  - ``_execute_search_documents`` / ``_fetch_direct_document_content`` —
    document retrieval used by the LangGraph RAG tools
    (``app/services/agent/tools.py``, ``app/services/agents/react_tools.py``).
  - ``_generate_citation_id`` — citation id allocation.
  - ``_get_accessible_workspaces`` — workspace scoping (web + Telegram).
  - ``format_sse_event`` / ``sse_with_heartbeat`` — SSE formatting helpers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import string
import uuid
from typing import AsyncGenerator, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.tenant import TenantUser
from app.models.knowledge_base import KnowledgeBase
from app.models.document import DocumentImage
from app.schemas.rag import (
    ChatSourceChunk,
    ChatImageRef,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_VISION_IMAGES = 3
SSE_HEARTBEAT_INTERVAL = 15  # seconds

_CITATION_ID_CHARS = string.ascii_lowercase + string.digits


def _generate_citation_id(existing: set[str]) -> str:
    """Generate a unique 4-char alphanumeric citation ID."""
    while True:
        cid = "".join(random.choices(_CITATION_ID_CHARS, k=4))
        if any(c.isalpha() for c in cid) and cid not in existing:
            return cid


# ---------------------------------------------------------------------------
# SSE Helpers (ported from PageIndex backend/app/api/v1/chat.py)
# ---------------------------------------------------------------------------


def format_sse_event(event: str, data: dict) -> str:
    """Format data as an SSE event string."""
    def json_serial(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)

    json_data = json.dumps(data, default=json_serial, ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


async def sse_with_heartbeat(
    source: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Wrap an SSE generator with periodic heartbeat comments.

    SSE spec allows lines starting with ':' as comments — browsers/clients
    silently ignore them but they keep the TCP connection alive, preventing
    timeouts when the upstream LLM takes a long time to respond.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _pump():
        try:
            async for event in source:
                await queue.put(event)
        except Exception:
            pass
        finally:
            await queue.put(None)  # sentinel

    task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=SSE_HEARTBEAT_INTERVAL
                )
                if event is None:
                    break
                yield event
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Tool executor — retrieval via HRAG
# ---------------------------------------------------------------------------


async def _get_accessible_workspaces(db: AsyncSession, user: User) -> list[uuid.UUID]:
    """Get all knowledge base IDs the user has access to."""
    if user.is_superadmin:
        result = await db.execute(select(KnowledgeBase.id))
        return list(result.scalars().all())

    # Get user's tenants
    tenant_result = await db.execute(
        select(TenantUser.tenant_id).where(TenantUser.user_id == user.id)
    )
    user_tenant_ids = list(tenant_result.scalars().all())

    from sqlalchemy import or_

    query = select(KnowledgeBase.id).where(
        or_(
            KnowledgeBase.visibility == "public",
            KnowledgeBase.owner_id == user.id,
            KnowledgeBase.tenant_id.in_(user_tenant_ids) if user_tenant_ids else False,
        )
    )
    result = await db.execute(query)
    return list(result.scalars().all())


MAX_DIRECT_DOC_CHARS = 48000  # ~12k tokens per doc


async def _fetch_direct_document_content(
    document_ids: list[uuid.UUID],
    db: AsyncSession,
) -> list[tuple]:
    """Fetch markdown content directly from MinIO for specific document UUIDs.

    This is used for chat-uploaded documents that are parsed but not indexed
    in ChromaDB (no embed pipeline). Returns list of (content, doc) tuples.
    """
    from sqlalchemy import select
    from app.models.document import Document
    from app.services.storage_service import get_storage_service

    if not document_ids:
        return []

    storage = get_storage_service()
    # Normalize document_ids — may be str or UUID (passed through langgraph state)
    normalized_ids: list[uuid.UUID] = []
    for d in document_ids:
        if isinstance(d, str):
            try:
                normalized_ids.append(uuid.UUID(d))
            except (ValueError, TypeError):
                continue
        elif isinstance(d, uuid.UUID):
            normalized_ids.append(d)
    result = await db.execute(select(Document).where(Document.id.in_(normalized_ids)))
    docs = result.scalars().all()
    # Bug fix: key doc_map by str(uuid) so lookups work whether caller has UUID or str
    doc_map: dict[str, Document] = {str(doc.id): doc for doc in docs}

    contents = []
    for doc_id in document_ids:
        lookup_key = str(doc_id) if isinstance(doc_id, (str, uuid.UUID)) else doc_id
        doc = doc_map.get(lookup_key)
        if not doc:
            logger.warning(f"[direct_doc] Document {doc_id} not found")
            continue
        # ROOT-CAUSE GATE: this full-document dump exists ONLY for chat-uploaded
        # files that are parsed but NOT embedded in ChromaDB (no chunks yet). For a
        # doc that is already indexed (chunk_count > 0), dumping the whole markdown
        # as a single score=1.0 "chunk_0" collapses citation granularity: one id
        # ends up covering dozens of articles, so the model cites an internal
        # article number (e.g. [14]) instead of a real source id, and the 48KB
        # mega-source crowds out precise chunks. Indexed docs are already reachable
        # via the scoped chunked query_deep search that runs alongside this — so
        # skip the dump for them and let proper per-chunk citations win.
        if (getattr(doc, "chunk_count", 0) or 0) > 0:
            logger.info(
                f"[direct_doc] Skip full-doc dump for indexed doc {doc_id} "
                f"(chunk_count={doc.chunk_count}); using chunked vector search"
            )
            continue
        if not doc.markdown_s3_key:
            logger.warning(f"[direct_doc] Document {doc_id} has no markdown_s3_key")
            continue
        try:
            md = await storage.download_markdown(doc.markdown_s3_key)
            truncated = md[:MAX_DIRECT_DOC_CHARS]
            if len(md) > MAX_DIRECT_DOC_CHARS:
                truncated += f"\n\n[... nội dung đã được cắt bớt ({len(md)} → {MAX_DIRECT_DOC_CHARS} ký tự) ...]"
            contents.append((truncated, doc))
            logger.info(f"[direct_doc] Loaded {len(truncated)} chars from doc {doc_id}")
        except Exception as e:
            logger.warning(
                f"[direct_doc] Failed to download markdown for doc {doc_id}: {e}"
            )

    return contents


async def _execute_search_documents(
    workspace_ids: list[uuid.UUID],
    query: str,
    top_k: int,
    db: AsyncSession,
    existing_ids: set[str],
    document_ids: Optional[list[uuid.UUID]] = None,
    search_mode: str = "hybrid",  # Phase 1: "vector" | "kg" | "hybrid"
    scoped_to_documents: bool = False,  # Phase 3: skip broad search when doc_ids known
) -> tuple[str, list[ChatSourceChunk], list[ChatImageRef], list[dict], list[str]]:
    """Execute document search across multiple workspaces and return best chunks.

    When document_ids is provided, fetches content directly from MinIO (for
    chat-uploaded documents that bypassed the ChromaDB embed pipeline).

    Args:
        search_mode: Controls retrieval strategy.
            "vector" — skip KG search, vector only (fast)
            "kg"     — skip vector search, KG only
            "hybrid" — run both (default)
        scoped_to_documents: Phase 3. When True and document_ids is non-empty,
            skips the broad per-workspace search loop and ONLY queries the vector
            store with the document_ids filter. Use when caller already knows which
            documents to search (e.g. after resolve_doc, or file upload context).

    Returns:
        (context_text, sources, image_refs, image_parts_for_vision, kg_summaries)
    """
    from app.services.rag_service import get_rag_service
    from app.services.hrag_service import HRAGService
    from pathlib import Path as _P

    # Translate search_mode to HRAGService.query_deep mode param
    _mode_map = {"vector": "vector", "kg": "kg", "hybrid": "hybrid"}
    hrag_mode = _mode_map.get(search_mode, "hybrid")
    if search_mode != "hybrid":
        logger.info(f"[RAG] search_mode={search_mode!r} → query_deep mode={hrag_mode!r}")

    all_chunks = []
    all_kg_summaries = []

    # Get workspace titles for better labeling
    from app.models.knowledge_base import KnowledgeBase

    ws_result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id.in_(workspace_ids))
    )
    ws_map = {ws.id: ws.name for ws in ws_result.scalars().all()}

    # ── Direct document fetch (for chat-uploaded files with markdown in MinIO) ──
    if document_ids:
        direct_contents = await _fetch_direct_document_content(document_ids, db)
        for idx, (content, doc) in enumerate(direct_contents):
            from types import SimpleNamespace

            chunk = SimpleNamespace(
                content=content,
                document_id=doc.id,
                chunk_index=0,
                page_no=0,
                heading_path=[],
                source_file=doc.original_filename or "document",
                image_refs=[],
                score=1.0,  # High score to prioritize over vector results
            )
            all_chunks.append((1.0, chunk, None, doc.workspace_id))
            logger.info(
                f"[RAG] Direct doc content loaded: doc={doc.id}, chars={len(content)}"
            )

    # Phase 3: UUID-scoped search — skip broad workspace loop, use document filter only
    if scoped_to_documents and document_ids:
        logger.info(
            f"[RAG] UUID-scoped: skipping broad workspace search, "
            f"querying {len(document_ids)} doc(s) directly via HRAG filter"
        )
        # Bug fix: previously iterated ALL workspaces and broke on the first
        # iteration even when 0 chunks were returned — that meant a doc in
        # workspace[1] was never reachable if workspace[0] returned 0 chunks.
        # Now: only search the workspaces that actually contain the documents.
        # Bug fix 2: document_ids may contain str (not UUID) when passed through
        # langgraph state — normalize to UUIDs before comparison.
        from app.models.document import Document
        normalized_doc_ids: list[uuid.UUID] = []
        for d in document_ids:
            if isinstance(d, str):
                try:
                    normalized_doc_ids.append(uuid.UUID(d))
                except (ValueError, TypeError):
                    continue
            elif isinstance(d, uuid.UUID):
                normalized_doc_ids.append(d)
        if len(normalized_doc_ids) != len(document_ids):
            logger.warning(
                f"[RAG] UUID-scoped: {len(document_ids) - len(normalized_doc_ids)} "
                f"of {len(document_ids)} doc id(s) were not valid UUIDs"
            )
        doc_ws_result = await db.execute(
            select(Document.workspace_id, Document.id).where(
                Document.id.in_(normalized_doc_ids)
            )
        )
        doc_ws_map: dict[uuid.UUID, set[uuid.UUID]] = {}
        for ws_id_val, doc_id_val in doc_ws_result:
            doc_ws_map.setdefault(ws_id_val, set()).add(doc_id_val)
        scoped_workspaces = [ws for ws in workspace_ids if ws in doc_ws_map]
        if not scoped_workspaces:
            logger.warning(
                f"[RAG] UUID-scoped: none of {len(normalized_doc_ids)} doc(s) belong to "
                f"user's {len(workspace_ids)} accessible workspace(s) — "
                f"doc→workspace mapping: { {str(k)[:8]: len(v) for k, v in doc_ws_map.items()} }"
            )
        else:
            logger.info(
                f"[RAG] UUID-scoped: narrowed {len(workspace_ids)} → "
                f"{len(scoped_workspaces)} workspace(s) containing the doc(s)"
            )
            for workspace_id in scoped_workspaces:
                rag_service = get_rag_service(db, workspace_id)
                if not isinstance(rag_service, HRAGService):
                    continue
                # Filter document_ids to only those in this workspace
                ws_doc_ids = [
                    d for d in normalized_doc_ids
                    if d in doc_ws_map.get(workspace_id, set())
                ]
                if not ws_doc_ids:
                    continue
                try:
                    result = await rag_service.query_deep(
                        question=query,
                        top_k=min(top_k, 10),
                        document_ids=ws_doc_ids,  # Hard filter: only these docs in this ws
                        mode=hrag_mode,
                        include_images=False,
                    )
                    for i, chunk in enumerate(result.chunks):
                        citation = result.citations[i] if i < len(result.citations) else None
                        score = getattr(chunk, "score", 0.0)
                        all_chunks.append((score, chunk, citation, workspace_id))
                    if result.knowledge_graph_summary:
                        all_kg_summaries.append(
                            f"### Knowledge Graph Insights\n{result.knowledge_graph_summary}"
                        )
                    logger.info(
                        f"[RAG] UUID-scoped query on workspace {workspace_id}: "
                        f"{len(result.chunks)} chunks"
                    )
                except Exception as e:
                    logger.warning(
                        f"[RAG] UUID-scoped search failed on workspace {workspace_id}: {e}"
                    )
        # Skip the broad workspace loop below
    else:
        # Fan out the per-workspace searches CONCURRENTLY. Each query_deep hits
        # ChromaDB + reranker (~seconds); running the workspaces sequentially made
        # total latency scale with workspace count. IMPORTANT: HRAGService.query_deep
        # uses its db AsyncSession, and an AsyncSession is NOT safe to share across
        # concurrent tasks — so each task gets its OWN session from AsyncSessionLocal.
        from app.core.database import AsyncSessionLocal
        import asyncio as _asyncio

        async def _search_one_ws(workspace_id):
            """Query ONE workspace on its own db session. Returns (chunks, kg)."""
            logger.info(
                f"[RAG] External Search: query='{query}' on workspace {workspace_id}"
            )
            packed: list = []          # (score, chunk, citation, workspace_id)
            kg_summary: str | None = None
            async with AsyncSessionLocal() as ws_db:
                rag_service = get_rag_service(ws_db, workspace_id)
                chunks = []
                citations = []
                if isinstance(rag_service, HRAGService):
                    try:
                        result = await rag_service.query_deep(
                            question=query,
                            top_k=min(top_k, 10),
                            document_ids=document_ids,
                            mode=hrag_mode,
                            include_images=False,
                        )
                        chunks = result.chunks
                        citations = result.citations
                        if result.knowledge_graph_summary:
                            kg_summary = (
                                f"### Knowledge Graph Insights\n{result.knowledge_graph_summary}"
                            )
                    except Exception as e:
                        logger.warning(f"Search failed for workspace {workspace_id}: {e}")
                else:
                    from types import SimpleNamespace

                    try:
                        legacy = rag_service.query(question=query, top_k=min(top_k, 10))
                        for i, c in enumerate(legacy.chunks):
                            chunks.append(
                                SimpleNamespace(
                                    content=c.content,
                                    document_id=int(c.metadata.get("document_id", 0)),
                                    chunk_index=i,
                                    page_no=int(c.metadata.get("page_no", 0)),
                                    heading_path=str(c.metadata.get("heading_path", "")).split(
                                        " > "
                                    )
                                    if c.metadata.get("heading_path")
                                    else [],
                                    source_file=str(c.metadata.get("source", "")),
                                    image_refs=[],
                                    score=c.score,
                                )
                            )
                    except Exception as e:
                        logger.warning(
                            f"Legacy search failed for workspace {workspace_id}: {e}"
                        )

            # Pack chunks with their citation for sorting
            for i, chunk in enumerate(chunks):
                citation = citations[i] if i < len(citations) else None
                score = getattr(chunk, "score", 0.0)
                packed.append((score, chunk, citation, workspace_id))
            return packed, kg_summary

        ws_results = await _asyncio.gather(
            *[_search_one_ws(ws) for ws in workspace_ids]
        )
        for packed, kg_summary in ws_results:
            all_chunks.extend(packed)
            if kg_summary:
                all_kg_summaries.append(kg_summary)

    # Sort all aggregated chunks by score descending
    all_chunks.sort(key=lambda x: x[0], reverse=True)
    logger.info(
        f"[RAG] Found total {len(all_chunks)} potential chunks across {len(workspace_ids)} workspaces"
    )

    # Take top_k
    best_chunks = all_chunks[:top_k]

    # Build sources
    sources: list[ChatSourceChunk] = []
    context_parts: list[str] = []
    chunk_image_ids: list[str] = []
    seen_image_ids: set[str] = set()
    source_pages = set()

    # Fetch document metadata for context
    doc_ids_in_chunks = {chunk.document_id for _, chunk, _, _ in best_chunks}
    doc_meta_map = {}
    if doc_ids_in_chunks:
        from app.models.document import Document
        # Handle string vs UUID ids (some systems return string doc ids)
        try:
            valid_uuids = [uuid.UUID(str(did)) for did in doc_ids_in_chunks]
        except ValueError:
            valid_uuids = list(doc_ids_in_chunks)
        doc_result = await db.execute(select(Document).where(Document.id.in_(valid_uuids)))
        for doc in doc_result.scalars().all():
            doc_meta_map[str(doc.id)] = {
                "published_date": doc.published_date,
                "document_number": doc.document_number,
                "document_title": doc.document_title,
                "validity_status": doc.validity_status,
                "superseded_by": doc.superseded_by_number,
            }

    from app.services.heading_path import extract_article_nos

    for score, chunk, citation, workspace_id in best_chunks:
        cid = _generate_citation_id(existing_ids)
        existing_ids.add(cid)
        _meta = doc_meta_map.get(str(chunk.document_id)) or {}
        _art_nos = extract_article_nos(chunk.heading_path)
        sources.append(
            ChatSourceChunk(
                index=cid,
                chunk_id=f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}",
                content=chunk.content,
                document_id=chunk.document_id,
                page_no=chunk.page_no,
                heading_path=chunk.heading_path,
                score=score,
                source_type="vector",
                source_file=citation.source_file if citation else getattr(chunk, "source_file", ""),
                document_number=_meta.get("document_number"),
                article_label=", ".join(f"Điều {n}" for n in _art_nos[:3]) or None,
                validity_status=_meta.get("validity_status"),
                superseded_by=_meta.get("superseded_by"),
            )
        )
        logger.info(
            f"[RAG] Selected Chunk [{cid}] (KB {workspace_id}) score={score:.3f}: {chunk.content[:60]}..."
        )

        # Collect images to fetch
        for iid in getattr(chunk, "image_refs", []) or []:
            if iid and iid not in seen_image_ids:
                seen_image_ids.add(iid)
                chunk_image_ids.append(iid)

        if getattr(chunk, "page_no", 0) > 0:
            source_pages.add(
                (getattr(chunk, "document_id", 0), getattr(chunk, "page_no", 0))
            )

        meta_parts = []
        if citation:
            meta_parts.append(citation.source_file)
            if citation.page_no:
                meta_parts.append(f"page {citation.page_no}")
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""
        if heading:
            meta_parts.append(heading)

        doc_meta = doc_meta_map.get(str(chunk.document_id))
        validity_warning = ""
        if doc_meta:
            if doc_meta["document_number"]:
                meta_parts.append(f"Số hiệu: {doc_meta['document_number']}")
            if doc_meta["published_date"]:
                meta_parts.append(f"Ngày ban hành: {doc_meta['published_date']}")
            if doc_meta.get("validity_status") == "superseded":
                by = doc_meta.get("superseded_by")
                validity_warning = (
                    "\n⚠️ VĂN BẢN NÀY ĐÃ HẾT HIỆU LỰC"
                    + (f" — đã được thay thế bởi {by}." if by else ".")
                )
            elif doc_meta.get("validity_status") == "partially_amended":
                validity_warning = (
                    "\n⚠️ Văn bản này đã được sửa đổi/bãi bỏ một phần bởi văn bản khác."
                )

        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
        context_parts.append(
            f"Nguồn [{cid}]{meta_line}:{validity_warning}\n{chunk.content}"
        )

    context = ""
    if all_kg_summaries:
        context += "## Knowledge Graph Entities & Relationships\n"
        context += "\n\n".join(all_kg_summaries)
        context += "\n\n---\n\n"

    context += "## Document Chunks\n"
    context += "\n\n---\n\n".join(context_parts)

    resolved_images: list[DocumentImage] = []
    if chunk_image_ids:
        img_result = await db.execute(
            select(DocumentImage).where(DocumentImage.image_id.in_(chunk_image_ids))
        )
        resolved_images = list(img_result.scalars().all())

    if not resolved_images and source_pages:
        from sqlalchemy import or_, and_

        page_filters = [
            and_(
                DocumentImage.document_id == doc_id,
                DocumentImage.page_no == page_no,
            )
            for doc_id, page_no in source_pages
        ]
        img_result = await db.execute(select(DocumentImage).where(or_(*page_filters)))
        resolved_images = list(img_result.scalars().all())
        seen = set()
        deduped = []
        for img in resolved_images:
            if img.image_id not in seen:
                seen.add(img.image_id)
                deduped.append(img)
        resolved_images = deduped

    chat_image_refs: list[ChatImageRef] = []
    image_context_parts: list[str] = []
    image_parts: list[dict] = []

    for img in resolved_images[:MAX_VISION_IMAGES]:
        img_ref_id = _generate_citation_id(existing_ids)
        existing_ids.add(img_ref_id)
        # Figure out which workspace this image belongs to in order to construct the correct URL
        # For simplicity we query the document's workspace_id
        workspace_id = (
            img.document.workspace_id
            if hasattr(img, "document") and img.document
            else workspace_ids[0]
            if workspace_ids
            else 0
        )
        img_url = f"/static/doc-images/kb_{workspace_id}/images/{img.image_id}.png"
        chat_image_refs.append(
            ChatImageRef(
                ref_id=img_ref_id,
                image_id=img.image_id,
                document_id=img.document_id,
                page_no=img.page_no,
                caption=img.caption or "",
                url=img_url,
                width=img.width,
                height=img.height,
            )
        )
        cap = f'"{img.caption}"' if img.caption else "no caption"
        image_context_parts.append(f"- [IMG-{img_ref_id}] Page {img.page_no}: {cap}")

        img_path = _P(img.file_path)
        if img_path.exists():
            try:
                img_bytes = img_path.read_bytes()
                mime = img.mime_type or "image/png"
                image_parts.append(
                    {
                        "inline_data": {"mime_type": mime, "data": img_bytes},
                        "page_no": img.page_no,
                        "caption": img.caption or "",
                        "img_ref_id": img_ref_id,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to read image {img.image_id}: {e}")

    if image_context_parts:
        context += "\n\nDocument Images:\n" + "\n".join(image_context_parts)

    # ── Drop orphan sources (defense-in-depth) ──────────────────────────────
    # A chunk whose document_id no longer exists in Postgres (document deleted /
    # re-uploaded, but its old vectors lingered in ChromaDB) renders as a source
    # the frontend can't open ("Nguồn <index>", click does nothing). Purging
    # ChromaDB removes these at rest; this filter guarantees none ever reach the
    # client even if a fresh orphan slips in. KG sources (nil document_id) are
    # kept — they are rendered as KG chips, not document links.
    if sources:
        from app.models.document import Document as _Doc
        _nil = uuid.UUID(int=0)

        def _coerce_uuid(v):
            if isinstance(v, uuid.UUID):
                return v
            if isinstance(v, str):
                try:
                    return uuid.UUID(v)
                except (ValueError, TypeError):
                    return None
            return None

        _cand: set[uuid.UUID] = set()
        for s in sources:
            did = _coerce_uuid(getattr(s, "document_id", None))
            if did is not None and did != _nil:
                _cand.add(did)
        if _cand:
            _rows = await db.execute(select(_Doc.id).where(_Doc.id.in_(_cand)))
            _alive = {r[0] for r in _rows.all()}
            _dropped = _cand - _alive
            if _dropped:
                before = len(sources)
                sources = [
                    s for s in sources
                    if getattr(s, "source_type", None) == "kg"
                    or _coerce_uuid(getattr(s, "document_id", None)) in _alive
                ]
                logger.warning(
                    f"[RAG] Dropped {before - len(sources)} orphan source(s) "
                    f"(document_id not in DB): {[str(d)[:8] for d in _dropped]}"
                )

    return context, sources, chat_image_refs, image_parts, all_kg_summaries



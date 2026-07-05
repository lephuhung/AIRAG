"""
Agent Tools
===========

Tool definitions for the NexusRAG LangGraph agent.

Tools are plain async functions — they are called directly by the
``tool_executor`` node (NOT via LangChain ToolNode) so they can
accept the full AgentState context injected at call time.

Available tools:
    search_documents   — hybrid vector+KG+BM25 search (wraps existing HRAG pipeline)
    list_documents     — list all documents in the workspace(s)
    summarize_document — get a summary of a specific document
    query_knowledge_graph — query LightRAG knowledge graph for entity relationships
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.services.agent.state import AgentState

logger = logging.getLogger(__name__)


def _format_doc_title(doc_title: str | None, original_filename: str | None) -> str:
    """
    Format document title for display.

    If doc_title is set and non-empty → use it.
    Otherwise → format original_filename into readable title:
      - Strip file extension (.pdf, .docx, etc.)
      - Replace underscores/hyphens with spaces
      - Title-case each word
      - Handle common patterns like "luat117_2025" → "Luật 117/2025"
    """
    import re
    import os

    if doc_title and doc_title.strip():
        return doc_title.strip()

    if not original_filename:
        return "Văn bản không tên"

    # Strip extension
    name = os.path.splitext(original_filename)[0]

    # Handle patterns like "luat117_2025", "nd13_2023", "tt15_2024"
    num_match = re.search(r'(luat|nd|tt|nq|qd|pl|bl)[_\s]*(\d+)', name, re.IGNORECASE)
    year_match = re.search(r'(20\d{2})', name)

    if num_match and year_match:
        doc_type = num_match.group(1).upper()
        num = num_match.group(2)
        year = year_match.group(1)
        type_map = {"LUAT": "Luật", "ND": "Nghị định", "TT": "Thông tư",
                    "NQ": "Nghị quyết", "QD": "Quyết định", "PL": "Pháp lệnh", "BL": "Bộ luật"}
        type_str = type_map.get(doc_type, doc_type)
        return f"{type_str} {num}/{year}"

    # Fallback: replace separators with spaces, title case
    name = re.sub(r'[_\-]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    if len(name) > 2:
        name = name.title()

    return name if name else "Văn bản không tên"


# ---------------------------------------------------------------------------
# Helpers (sync, run in thread pool to avoid blocking event loop)
# ---------------------------------------------------------------------------

def _write_file_sync(path: str, data: bytes) -> None:
    """Synchronous file write — must run in asyncio.to_thread()"""
    with open(path, "wb") as f:
        f.write(data)


# ---------------------------------------------------------------------------
# Tool registry — maps tool name → callable
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, str] = {
    "search_documents": "search the knowledge base for relevant document sections",
    "list_documents": "list all documents available in the current workspace",
    "summarize_document": "get a comprehensive summary of a specific document",
    "get_documents_content": "fetch raw markdown content from multiple documents by their UUIDs for detailed reading/summarization/editing",
    "query_knowledge_graph": "query the knowledge graph for entity relationships",
    "search_documents_number": "search for documents by their official document number (văn bản số)",
    "search_abbreviation": "search for the meaning of an abbreviation or acronym",
    # MongoDB people search tools
    "search_people_by_cccd": "search for a person by their CCCD (Căn cước công dân) national ID number",
    "search_people_by_name": "search for persons by their full name or partial name",
    "search_people_by_bhxh": "search for a person by their BHXH (Bảo hiểm xã hội) number",
    "search_people_by_phone": "search for persons by their phone number",
    "search_people_advanced": "search for persons using combinations of exact/partial information like Name + Date of birth + Address/Hometown",
    # Document reference resolution
    "resolve_document_reference": "resolve a document reference (e.g. 'Luật An ninh mạng 2025') to find the correct document UUID by matching document type, title keywords, year, and document number",
}


# ---------------------------------------------------------------------------
# Tool 1: search_documents
# Wraps the existing _execute_search_documents from chat_agent.py
# ---------------------------------------------------------------------------


async def search_documents(
    query: str,
    top_k: int,
    workspace_ids: list[uuid.UUID],
    existing_citation_ids: set,
    db: "AsyncSession",
    document_ids: Optional[list[uuid.UUID]] = None,
    search_mode: Optional[str] = None,  # Phase 1: "vector" | "kg" | "hybrid"; None → HRAG_DEFAULT_QUERY_MODE
    scoped_to_documents: bool = False,  # Phase 3: restrict search to document_ids only
) -> dict:
    """
    Hybrid search across workspaces via HRAG pipeline.

    Args:
        search_mode: Controls retrieval strategy.
            "vector" — vector search only (fast, good for extraction/summarize)
            "kg"     — knowledge graph only (good for relationship queries)
            "hybrid" — vector + KG (default, most thorough)
        scoped_to_documents: Phase 3. When True and document_ids is non-empty,
            skips broad workspace search and ONLY queries the specified documents.
            Improves precision when context is already narrowed to specific docs
            (e.g. after resolve_doc or when user uploads a specific file).

    Returns:
        dict with keys: context_text, sources, images, image_parts, kg_summaries
    """
    from app.api.chat_agent import _execute_search_documents, _generate_citation_id
    from app.core.config import settings

    if search_mode is None:
        search_mode = settings.HRAG_DEFAULT_QUERY_MODE

    (
        context_text,
        sources,
        image_refs,
        image_parts,
        kg_summaries,
    ) = await _execute_search_documents(
        workspace_ids=workspace_ids,
        query=query,
        top_k=top_k,
        db=db,
        existing_ids=existing_citation_ids,
        document_ids=document_ids,
        search_mode=search_mode,
        scoped_to_documents=scoped_to_documents,  # Phase 3
    )
    return {
        "context_text": context_text,
        "sources": sources,
        "images": image_refs,
        "image_parts": image_parts,
        "kg_summaries": kg_summaries,
    }


# ---------------------------------------------------------------------------
# Tool 2: list_documents
# ---------------------------------------------------------------------------


async def list_documents(
    workspace_ids: list[int],
    db: "AsyncSession",
) -> dict:
    """
    Return a list of all indexed documents in the given workspace(s).

    Returns:
        dict with keys: text (formatted list), document_count
    """
    from sqlalchemy import select
    from app.models.document import Document, DocumentStatus
    from app.models.knowledge_base import KnowledgeBase

    try:
        # Get workspace names
        ws_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(workspace_ids))
        )
        ws_map = {ws.id: ws.name for ws in ws_result.scalars().all()}

        # Get indexed documents
        doc_result = await db.execute(
            select(Document)
            .where(
                Document.workspace_id.in_(workspace_ids),
                Document.status == DocumentStatus.INDEXED,
            )
            .order_by(Document.workspace_id, Document.created_at.desc())
        )
        docs = doc_result.scalars().all()

        if not docs:
            return {
                "text": "Không có tài liệu nào đã được lập chỉ mục trong workspace này.",
                "document_count": 0,
            }

        # Group by workspace
        by_ws: dict[int, list] = {}
        for doc in docs:
            by_ws.setdefault(doc.workspace_id, []).append(doc)

        lines = []
        doc_counter = 1
        for ws_id, ws_docs in by_ws.items():
            ws_name = ws_map.get(ws_id, f"KB {ws_id}")
            lines.append(f"\n### Workspace: {ws_name}")
            for i, doc in enumerate(ws_docs, 1):
                page_info = f", {doc.page_count} trang" if doc.page_count else ""
                chunk_info = f", {doc.chunk_count} đoạn" if doc.chunk_count else ""
                doc_title = _format_doc_title(doc.document_title, doc.original_filename)
                lines.append(
                    f"{i}. **{doc_title}** (DocRef: doc{doc_counter:02d})"
                    f"{page_info}{chunk_info}"
                )
                doc_counter += 1

        text = f"Tổng cộng **{len(docs)} tài liệu** đã được lập chỉ mục:\n"
        text += "\n".join(lines)

        return {"text": text, "document_count": len(docs)}

    except Exception as e:
        logger.error(f"[tool:list_documents] Failed: {e}")
        return {
            "text": "Không thể lấy danh sách tài liệu. Vui lòng thử lại.",
            "document_count": 0,
        }


# ---------------------------------------------------------------------------
# Tool 3: summarize_document
# ---------------------------------------------------------------------------


async def summarize_document(
    document_id: int,
    db: "AsyncSession",
) -> dict:
    """
    Generate a comprehensive summary of a document by reading its parsed markdown.

    Returns:
        dict with keys: text (summary), document_name, document_id
    """
    from sqlalchemy import select
    from app.models.document import Document, DocumentStatus
    from app.services.storage_service import get_storage_service
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage

    try:
        # Fetch document
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()

        if not doc:
            return {
                "text": f"Không tìm thấy tài liệu với ID {document_id}.",
                "document_name": "",
                "document_id": document_id,
            }

        # Allow chat-upload files (PARSE_DONE or beyond) even if not yet INDEXED,
        # as long as markdown content exists in MinIO
        if doc.status not in (
            DocumentStatus.INDEXED,
            DocumentStatus.CHUNKING,
            DocumentStatus.EMBEDDING,
            DocumentStatus.PARSING,
        ):
            return {
                "text": f"Tài liệu '{doc.original_filename}' chưa được xử lý xong.",
                "document_name": doc.original_filename,
                "document_id": document_id,
            }

        # Load markdown from MinIO
        if not doc.markdown_s3_key:
            return {
                "text": f"Tài liệu '{doc.original_filename}' không có nội dung đã phân tích.",
                "document_name": doc.original_filename,
                "document_id": document_id,
            }

        storage = get_storage_service()
        try:
            markdown_text = await storage.download_markdown(doc.markdown_s3_key)
        except Exception as e:
            return {
                "text": f"Lỗi tải markdown từ S3: {e}",
                "document_name": doc.original_filename,
                "document_id": document_id,
            }

        # Truncate to avoid exceeding context window (~16k chars ≈ 4k tokens)
        MAX_CHARS = 16000
        truncated = markdown_text[:MAX_CHARS]
        if len(markdown_text) > MAX_CHARS:
            truncated += "\n\n[... nội dung đã được cắt bớt ...]"

        # Call main LLM for summarization
        llm = get_llm_provider()
        summary_prompt = (
            f"Hãy tóm tắt toàn diện tài liệu sau bằng tiếng Việt. "
            f"Bao gồm: mục đích chính, các điểm quan trọng, số liệu, và kết luận.\n\n"
            f"Tài liệu: {doc.original_filename}\n\n"
            f"Nội dung:\n{truncated}"
        )

        summary = await llm.acomplete(
            messages=[LLMMessage(role="user", content=summary_prompt)],
            temperature=0.1,
            max_tokens=1024,
        )
        summary_text = (
            summary
            if isinstance(summary, str)
            else getattr(summary, "content", str(summary))
        )

        return {
            "text": summary_text,
            "document_name": doc.original_filename,
            "document_id": document_id,
        }

    except Exception as e:
        logger.error(f"[tool:summarize_document] Failed for doc {document_id}: {e}")
        return {
            "text": "Không thể tóm tắt tài liệu. Vui lòng thử lại.",
            "document_name": "",
            "document_id": document_id,
        }


# ---------------------------------------------------------------------------
# Tool 3b: get_documents_content
# ---------------------------------------------------------------------------


async def get_documents_content(
    document_ids: list[uuid.UUID],
    db: "AsyncSession",
) -> dict:
    """
    Fetch raw markdown content from multiple documents by their UUIDs.

    This is useful when the user references documents by name (@docname)
    or by attached document IDs, and the agent needs the full content
    for tasks like summarization, grammar checking, or editing suggestions.

    Args:
        document_ids: List of document UUIDs to fetch
        db: Database session

    Returns:
        dict with keys:
            - documents: list of {id, filename, content, error}
            - total_count: int
            - errors: list of error messages if any documents failed
    """
    from sqlalchemy import select
    from app.models.document import Document, DocumentStatus
    from app.services.storage_service import get_storage_service

    MAX_CHARS_PER_DOC = 48000  # ~12k tokens, leave room for prompt

    results = []
    errors = []

    if not document_ids:
        return {
            "documents": [],
            "total_count": 0,
            "errors": [],
        }

    try:
        # Fetch all documents in one query
        result = await db.execute(select(Document).where(Document.id.in_(document_ids)))
        docs = result.scalars().all()

        # Create a map for quick lookup (keys are UUID objects)
        doc_map = {doc.id: doc for doc in docs}

        storage = get_storage_service()

        for doc_id in document_ids:
            # Convert string UUIDs to UUID objects for lookup (doc_map keys are UUIDs)
            doc_id_uuid = uuid.UUID(doc_id) if isinstance(doc_id, str) else doc_id
            doc = doc_map.get(doc_id_uuid)

            if not doc:
                errors.append(f"Không tìm thấy tài liệu {doc_id}")
                results.append(
                    {
                        "id": str(doc_id),
                        "filename": "Unknown",
                        "content": None,
                        "error": f"Không tìm thấy tài liệu",
                    }
                )
                continue

            # Allow chat-upload files (PARSE_DONE or beyond) even if not yet INDEXED,
            # as long as markdown content exists in MinIO
            if doc.status not in (
                DocumentStatus.INDEXED,
                DocumentStatus.CHUNKING,
                DocumentStatus.EMBEDDING,
                DocumentStatus.PARSING,
            ):
                errors.append(
                    f"Tài liệu '{doc.original_filename}' chưa được xử lý xong"
                )
                results.append(
                    {
                        "id": str(doc_id),
                        "filename": doc.original_filename,
                        "content": None,
                        "error": f"Tài liệu chưa được xử lý xong (status: {doc.status.value if hasattr(doc.status, 'value') else doc.status}). Vui lòng đợi một chút và thử lại.",
                    }
                )
                continue

            if not doc.markdown_s3_key:
                errors.append(f"Tài liệu '{doc.original_filename}' không có nội dung")
                results.append(
                    {
                        "id": str(doc_id),
                        "filename": doc.original_filename,
                        "content": None,
                        "error": "Không có nội dung markdown",
                    }
                )
                continue

            try:
                markdown_text = await storage.download_markdown(doc.markdown_s3_key)

                # Truncate if too long
                truncated = markdown_text[:MAX_CHARS_PER_DOC]
                if len(markdown_text) > MAX_CHARS_PER_DOC:
                    truncated += f"\n\n[... nội dung đã được cắt bớt ({len(markdown_text)} → {MAX_CHARS_PER_DOC} ký tự) ...]"

                results.append(
                    {
                        "id": str(doc_id),
                        "filename": doc.original_filename,
                        "content": truncated,
                        "error": None,
                    }
                )

            except Exception as e:
                errors.append(f"Lỗi tải '{doc.original_filename}': {e}")
                results.append(
                    {
                        "id": str(doc_id),
                        "filename": doc.original_filename,
                        "content": None,
                        "error": str(e),
                    }
                )

        return {
            "documents": results,
            "total_count": len(results),
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"[tool:get_documents_content] Failed: {e}")
        return {
            "documents": [],
            "total_count": 0,
            "errors": [f"Lỗi hệ thống: {e}"],
        }


# ---------------------------------------------------------------------------
# Tool 3c: get_document_format
# ---------------------------------------------------------------------------


async def get_document_format(
    document_ids: list[uuid.UUID],
    db: "AsyncSession",
) -> dict:
    """
    Extract and return format metadata for Word (.docx) documents.

    Downloads the original docx file from MinIO and extracts formatting
    information including margins, fonts, line spacing, etc.
    Used when user asks to check document formatting.

    Args:
        document_ids: List of document UUIDs to extract format from
        db: Database session

    Returns:
        dict with keys:
            - documents: list of {id, filename, file_type, format_data, error}
            - total_count: int
            - errors: list of error messages if any documents failed
    """
    import tempfile
    import os

    from sqlalchemy import select
    from app.models.document import Document, DocumentStatus
    from app.services.storage_service import get_storage_service
    from app.services.agents.docx_formatter_tools import extract_docx_format, extract_docx_format_sync

    results = []
    errors = []

    if not document_ids:
        return {
            "documents": [],
            "total_count": 0,
            "errors": [],
        }

    try:
        # Fetch all documents in one query
        result = await db.execute(select(Document).where(Document.id.in_(document_ids)))
        docs = result.scalars().all()
        doc_map = {doc.id: doc for doc in docs}

        storage = get_storage_service()

        for doc_id in document_ids:
            doc = doc_map.get(doc_id)

            if not doc:
                errors.append(f"Không tìm thấy tài liệu {doc_id}")
                results.append({
                    "id": str(doc_id),
                    "filename": "Unknown",
                    "file_type": None,
                    "format_data": None,
                    "error": "Không tìm thấy tài liệu",
                })
                continue

            # Only process Word documents
            file_type = doc.file_type.lower() if doc.file_type else ""
            if file_type not in ("docx", "word", ".docx"):
                results.append({
                    "id": str(doc_id),
                    "filename": doc.original_filename,
                    "file_type": file_type,
                    "format_data": None,
                    "error": f"Không phải file Word (.docx). Loại file: {file_type or 'unknown'}",
                })
                continue

            if not doc.upload_s3_key:
                results.append({
                    "id": str(doc_id),
                    "filename": doc.original_filename,
                    "file_type": file_type,
                    "format_data": None,
                    "error": "Không tìm thấy file gốc trong MinIO",
                })
                continue

            try:
                # Download docx from MinIO to temp file
                tmp_dir = tempfile.gettempdir()
                tmp_path = os.path.join(tmp_dir, f"format_check_{doc_id}.docx")

                file_data = await storage.download_file(doc.upload_s3_key)
                # Use asyncio.to_thread to avoid blocking the event loop
                import asyncio
                await asyncio.to_thread(_write_file_sync, tmp_path, file_data)

                # Extract format metadata in thread pool (CPU-bound docx parsing)
                format_data = await asyncio.to_thread(extract_docx_format_sync, tmp_path)

                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

                if format_data.get("error"):
                    results.append({
                        "id": str(doc_id),
                        "filename": doc.original_filename,
                        "file_type": file_type,
                        "format_data": None,
                        "error": format_data["error"],
                    })
                else:
                    results.append({
                        "id": str(doc_id),
                        "filename": doc.original_filename,
                        "file_type": file_type,
                        "format_data": format_data,
                        "error": None,
                    })

            except Exception as e:
                logger.error(f"[tool:get_document_format] Failed for doc {doc_id}: {e}")
                errors.append(f"Lỗi xử lý '{doc.original_filename}': {e}")
                results.append({
                    "id": str(doc_id),
                    "filename": doc.original_filename,
                    "file_type": file_type,
                    "format_data": None,
                    "error": str(e),
                })

        return {
            "documents": results,
            "total_count": len(results),
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"[tool:get_document_format] Failed: {e}")
        return {
            "documents": [],
            "total_count": 0,
            "errors": [f"Lỗi hệ thống: {e}"],
        }


# ---------------------------------------------------------------------------
# Tool 4: query_knowledge_graph
# ---------------------------------------------------------------------------


async def query_knowledge_graph(
    entity: str,
    workspace_ids: list[int],
    db: "AsyncSession",
) -> dict:
    """
    Query the LightRAG knowledge graph for entity relationships.

    Returns:
        dict with keys: text (formatted KG results)
    """
    from app.services.kg.knowledge_graph_service import get_kg_service

    results = []
    for ws_id in workspace_ids:
        try:
            kg_service = get_kg_service(workspace_id=ws_id)
            # Use naive mode for entity lookup (faster than hybrid)
            # Bug fix: LegalKGService.query() takes `question=`, not `query=`.
            kg_result = await kg_service.query(
                question=entity,
                mode="naive",
            )
            if kg_result and kg_result.strip():
                results.append(f"**Workspace {ws_id}:**\n{kg_result}")
        except Exception as e:
            logger.warning(
                f"[tool:query_knowledge_graph] KG query failed for ws {ws_id}: {e}"
            )

    if not results:
        return {
            "text": f"Không tìm thấy thông tin về '{entity}' trong knowledge graph."
        }

    return {"text": "\n\n".join(results)}


# ---------------------------------------------------------------------------
# Tool 4b: get_document_relations — quan hệ văn bản ↔ văn bản + hiệu lực
# ---------------------------------------------------------------------------

_REL_LABELS = {
    "CAN_CU": "căn cứ",
    "VIEN_DAN": "viện dẫn",
    "SUA_DOI": "sửa đổi, bổ sung",
    "THAY_THE": "thay thế",
    "BAI_BO": "bãi bỏ",
    "REFERENCES": "tham chiếu",
}
_EVENT_LABELS = {
    "thay_the": "thay thế",
    "bai_bo": "bãi bỏ",
    "het_hieu_luc": "làm hết hiệu lực",
}
_VALIDITY_LABELS = {
    "effective": "còn hiệu lực",
    "superseded": "ĐÃ HẾT HIỆU LỰC",
    "partially_amended": "đã được sửa đổi/bãi bỏ một phần",
    "unknown": "chưa xác định hiệu lực",
}


async def get_document_relations(
    document: str,
    workspace_ids: list,
    db: "AsyncSession",
    relation: Optional[str] = None,
) -> dict:
    """
    Tra quan hệ VĂN BẢN ↔ VĂN BẢN (căn cứ / viện dẫn / sửa đổi / thay thế /
    bãi bỏ) và trạng thái hiệu lực, gộp từ HAI nguồn:

    - bảng ``documents`` (validity_status/superseded_by/validity_events —
      trích regex từ điều khoản thi hành, phủ cả kho hiện có);
    - Neo4j LegalKG (edges CAN_CU/VIEN_DAN/SUA_DOI/THAY_THE/BAI_BO).

    Returns: dict với key ``text`` (kết quả đã format tiếng Việt).
    """
    import re as _re

    from sqlalchemy import or_, select

    from app.models.document import Document
    from app.services.kg.knowledge_graph_service import get_kg_service

    ref = (document or "").strip()
    if not ref:
        return {"text": "Lỗi: thiếu tên/số hiệu văn bản."}

    # Số hiệu dạng "85/2016(/NĐ-CP)" nếu có trong ref — match DB chặt hơn title
    num_match = _re.search(r"\d{1,4}/\d{4}(?:/[\w\-]+)?", ref)
    number_pat = f"%{num_match.group(0)}%" if num_match else None

    lines: list[str] = []
    seen: set[tuple] = set()

    # ── Nguồn 1: bảng documents ─────────────────────────────────────────────
    matched_docs = []
    try:
        conds = [Document.document_title.ilike(f"%{ref}%")]
        if number_pat:
            conds.append(Document.document_number.ilike(number_pat))
        else:
            conds.append(Document.document_number.ilike(f"%{ref}%"))
        matched_docs = (await db.execute(
            select(Document).where(
                Document.workspace_id.in_(workspace_ids), or_(*conds)
            ).limit(5)
        )).scalars().all()

        for doc in matched_docs:
            status = _VALIDITY_LABELS.get(doc.validity_status or "unknown")
            head = f"• {doc.document_number or doc.document_title}: {status}"
            if doc.validity_status == "superseded" and doc.superseded_by_number:
                head += f" — được thay thế bởi {doc.superseded_by_number}"
            if doc.effective_date:
                head += f" (có hiệu lực từ {doc.effective_date})"
            lines.append(head)
            # Sự kiện full (đổi hiệu lực văn bản khác) luôn hiện; partial
            # (sửa cụm từ/khoản lẻ) cap lại để không nhấn chìm tín hiệu chính.
            events = sorted(
                doc.validity_events or [],
                key=lambda ev: ev.get("scope") != "full",
            )
            shown_partial = 0
            for ev in events:
                is_partial = ev.get("scope") == "partial"
                if is_partial and shown_partial >= 8:
                    continue
                key = (doc.document_number, ev.get("kind"), ev.get("target_number"))
                if key in seen:
                    continue
                seen.add(key)
                shown_partial += is_partial
                kind = _EVENT_LABELS.get(ev.get("kind"), ev.get("kind"))
                scope = " MỘT PHẦN" if is_partial else " toàn bộ"
                lines.append(
                    f"  - {doc.document_number} {kind}{scope}: {ev.get('target_number')}"
                )
            hidden = sum(1 for ev in events if ev.get("scope") == "partial") - shown_partial
            if hidden > 0:
                lines.append(f"  - … và {hidden} tuyên bố sửa đổi một phần khác")

        # Văn bản KHÁC trong kho từng tuyên bố nhắm vào ref (chiều ngược)
        if number_pat:
            others = (await db.execute(
                select(Document).where(
                    Document.workspace_id.in_(workspace_ids),
                    Document.validity_events.isnot(None),
                )
            )).scalars().all()
            matched_ids = {d.id for d in matched_docs}
            bare = num_match.group(0).lower()
            for doc in others:
                if doc.id in matched_ids:
                    continue
                for ev in doc.validity_events or []:
                    target = (ev.get("target_number") or "").lower()
                    if bare not in target:
                        continue
                    kind = _EVENT_LABELS.get(ev.get("kind"), ev.get("kind"))
                    scope = " MỘT PHẦN" if ev.get("scope") == "partial" else " toàn bộ"
                    key = (doc.document_number, ev.get("kind"), ev.get("target_number"))
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(
                        f"• {doc.document_number} {kind}{scope}: {ev.get('target_number')}"
                        f" (trích: \"{(ev.get('quote') or '')[:120]}...\")"
                    )
    except Exception as e:
        logger.warning(f"[tool:get_document_relations] DB lookup failed: {e}")

    # ── Nguồn 2: Neo4j LegalKG ──────────────────────────────────────────────
    rel_filter = None
    if relation:
        rel_key = relation.strip().upper()
        if rel_key in _REL_LABELS:
            rel_filter = [rel_key]
    for ws_id in workspace_ids:
        try:
            kg_service = get_kg_service(workspace_id=ws_id)
            if not hasattr(kg_service, "get_document_relations"):
                continue  # lightrag mode không có traversal văn bản
            kg_rels = await kg_service.get_document_relations(
                ref, relation_types=rel_filter
            )
            for r in kg_rels:
                key = (r["source"], r["relation"], r["target"])
                if key in seen:
                    continue
                seen.add(key)
                label = _REL_LABELS.get(r["relation"], r["relation"])
                desc = f" ({r['description'][:100]})" if r.get("description") else ""
                lines.append(f"• {r['source']} —[{label}]→ {r['target']}{desc}")
        except Exception as e:
            logger.warning(
                f"[tool:get_document_relations] KG lookup failed ws={ws_id}: {e}"
            )

    if not lines:
        return {
            "text": (
                f"Không tìm thấy quan hệ văn bản nào cho '{ref}' trong kho. "
                f"Kho chỉ biết quan hệ giữa các văn bản ĐÃ được upload hoặc "
                f"được nhắc tới trong điều khoản thi hành của chúng."
            )
        }
    return {"text": f"Quan hệ văn bản cho '{ref}':\n" + "\n".join(lines)}


# ---------------------------------------------------------------------------
# Tool 5: search_documents_number
# ---------------------------------------------------------------------------


async def search_documents_number(
    query: str,
    workspace_ids: list[int],
    db: "AsyncSession",
) -> dict:
    """
    Search for documents by their official document number (văn bản số).

    Returns:
        dict with keys: text (formatted results), documents (list of doc info)
    """
    from sqlalchemy import select, or_
    from app.models.document import Document, DocumentStatus
    import re

    try:
        # Create a fuzzy pattern by replacing spaces, punctuation with %
        # e.g., "60/QĐ-UBND" -> "60%QĐ%UBND", to match both "60/QĐ-UBND" and "60_QÐ_UBND.pdf"
        fuzzy_query = re.sub(r"[\s/\-_.,]+", "%", query.strip())
        fuzzy_pattern = f"%{fuzzy_query}%"

        result = await db.execute(
            select(Document)
            .where(
                Document.workspace_id.in_(workspace_ids),
                Document.status == DocumentStatus.INDEXED,
                or_(
                    Document.document_number.ilike(fuzzy_pattern),
                    Document.original_filename.ilike(fuzzy_pattern),
                    Document.markdown_s3_key.ilike(fuzzy_pattern),
                    Document.upload_s3_key.ilike(fuzzy_pattern),
                ),
            )
            .order_by(Document.created_at.desc())
            .limit(20)
        )
        docs = result.scalars().all()

        if not docs:
            return {
                "text": f"Không tìm thấy tài liệu nào có số văn bản '{query}'.",
                "documents": [],
            }

        lines = [f"Tìm thấy **{len(docs)} tài liệu** có số văn bản liên quan:"]
        from app.schemas.rag import ChatSourceChunk
        import uuid
        import random, string

        doc_list = []
        for i, doc in enumerate(docs, 1):
            chars = string.ascii_lowercase + string.digits
            while True:
                cid = "".join(random.choices(chars, k=4))
                if any(c.isalpha() for c in cid):
                    break
            
            chunk_obj = ChatSourceChunk(
                index=cid,
                chunk_id=f"doc_{doc.id}_meta",
                content=f"Tài liệu: {_format_doc_title(doc.document_title, doc.original_filename)}\nSố văn bản: {doc.document_number or 'N/A'}",
                document_id=doc.id,
                page_no=0,
                heading_path=[],
                score=1.0, # Exact match by number
                source_type="metadata",
                source_file=doc.original_filename
            )
            doc_list.append(chunk_obj)

            doc_title = _format_doc_title(doc.document_title, doc.original_filename)
            lines.append(
                f"{i}. **{doc_title}**\n"
                f"   Số văn bản: {doc.document_number or 'N/A'}\n"
                f"   Mã trích dẫn (Citation ID): [{cid}]"
            )

        return {
            "text": "\n".join(lines),
            "documents": doc_list,
        }

    except Exception as e:
        logger.error(f"[tool:search_documents_number] Failed: {e}")
        return {
            "text": "Không thể tìm kiếm theo số văn bản. Vui lòng thử lại.",
            "documents": [],
        }


# ---------------------------------------------------------------------------
# Tool 6: search_abbreviation
# ---------------------------------------------------------------------------


async def search_abbreviation(
    abbreviation: str,
    workspace_ids: list[int],
    db: "AsyncSession",
) -> dict:
    """
    Search for the meaning of an abbreviation or acronym.

    Returns:
        dict with keys: text (meaning or ask for clarification), abbreviation, found
    """
    from sqlalchemy import select
    from app.models.abbreviation import Abbreviation

    logger.info(f"[tool:search_abbreviation] searching abbreviation={abbreviation!r}, workspace_ids={workspace_ids}, db={type(db).__name__}")
    try:
        result = await db.execute(
            select(Abbreviation)
            .where(
                Abbreviation.short_form.ilike(f"%{abbreviation}%"),
                Abbreviation.is_active == True,
            )
            .limit(10)
        )
        abbreviations = result.scalars().all()
        logger.info(f"[tool:search_abbreviation] found {len(abbreviations)} results for '{abbreviation}'")

        if not abbreviations:
            return {
                "text": f"Không tìm thấy nghĩa của '{abbreviation}'. "
                f"Bạn có thể cho biết '{abbreviation}' là viết tắt của gì không?",
                "abbreviation": abbreviation,
                "found": False,
            }

        if len(abbreviations) == 1:
            ab = abbreviations[0]
            return {
                "text": f"**{ab.short_form}** = {ab.full_form}\n"
                f"{f'Mô tả: {ab.description}' if ab.description else ''}",
                "abbreviation": ab.short_form,
                "full_form": ab.full_form,
                "description": ab.description,
                "found": True,
            }

        lines = [f"Tìm thấy **{len(abbreviations)} kết quả** cho '{abbreviation}':"]
        for i, ab in enumerate(abbreviations, 1):
            lines.append(f"{i}. **{ab.short_form}** = {ab.full_form}")

        return {
            "text": "\n".join(lines),
            "abbreviation": abbreviation,
            "found": True,
            "results": [
                {
                    "short_form": ab.short_form,
                    "full_form": ab.full_form,
                    "description": ab.description,
                }
                for ab in abbreviations
            ],
        }

    except Exception as e:
        logger.error(f"[tool:search_abbreviation] Failed: {e}")
        return {
            "text": f"Không thể tìm kiếm nghĩa của '{abbreviation}'. Vui lòng thử lại.",
            "abbreviation": abbreviation,
            "found": False,
        }


# ---------------------------------------------------------------------------
# MongoDB People Search Tools
# ---------------------------------------------------------------------------


async def search_people_by_cccd(cccd: str):
    """
    Search for a person by CCCD (Căn cước công dân) number.
    Exact match on the cccd field. Yields partial results.
    """
    from app.services.people.mongo_people_service import BUSY_MESSAGE, search_by_cccd as _svc

    try:
        async for res in _svc(cccd):
            yield res
    except Exception as e:
        logger.error(f"[tool:search_people_by_cccd] Failed: {e}")
        yield {"found": False, "error": "unavailable", "persons": [], "display": BUSY_MESSAGE}


async def search_people_by_name(name: str, limit: int = 10):
    """
    Search for persons by name (ho_ten). Yields partial results.
    Case-insensitive partial regex match.

    Returns:
        dict with keys: found, count, persons, display
    """
    from app.services.people.mongo_people_service import BUSY_MESSAGE, search_by_name as _svc

    try:
        async for res in _svc(name, limit=limit):
            yield res
    except Exception as e:
        logger.error(f"[tool:search_people_by_name] Failed: {e}")
        yield {
            "found": False,
            "error": "unavailable",
            "count": 0,
            "persons": [],
            "display": BUSY_MESSAGE,
        }


async def search_people_by_bhxh(so_bhxh: str):
    """
    Search for a person by BHXH (Bảo hiểm xã hội) number.
    Exact or loose regex match. Yields partial results.

    Returns:
        dict with keys: found, person, display
    """
    from app.services.people.mongo_people_service import BUSY_MESSAGE, search_by_bhxh as _svc

    try:
        async for res in _svc(so_bhxh):
            yield res
    except Exception as e:
        logger.error(f"[tool:search_people_by_bhxh] Failed: {e}")
        yield {"found": False, "error": "unavailable", "persons": [], "display": BUSY_MESSAGE}


async def search_people_by_phone(phone: str, limit: int = 10):
    """
    Search for persons by phone number (so_dien_thoai).
    Exact, ends-with, or contains match. Yields partial results.

    Returns:
        dict with keys: found, count, persons, display
    """
    from app.services.people.mongo_people_service import BUSY_MESSAGE, search_by_phone as _svc

    try:
        async for res in _svc(phone, limit=limit):
            yield res
    except Exception as e:
        logger.error(f"[tool:search_people_by_phone] Failed: {e}")
        yield {
            "found": False,
            "error": "unavailable",
            "count": 0,
            "persons": [],
            "display": BUSY_MESSAGE,
        }

async def search_people_advanced(criteria: dict, limit: int = 10):
    """
    Search for persons by multiple criteria (Name + DoB + Address + etc).
    Yields partial results.
    """
    from app.services.people.mongo_people_service import BUSY_MESSAGE, search_by_advanced as _svc

    try:
        async for res in _svc(criteria, limit=limit):
            yield res
    except Exception as e:
        logger.error(f"[tool:search_people_advanced] Failed: {e}")
        yield {
            "found": False,
            "error": "unavailable",
            "count": 0,
            "persons": [],
            "display": BUSY_MESSAGE,
        }


# ---------------------------------------------------------------------------
# Document Reference Resolution Tool
# ---------------------------------------------------------------------------

import re
from sqlalchemy import select, and_, or_
from app.models.document import Document, DocumentStatus
from app.models.document_type import DocumentType


async def _suggest_low_score_match(reference: str, top_candidates: list[dict]) -> str | None:
    """Ask the main LLM to phrase a 'did you mean …?' hint for low-score matches.

    Mirrors the legacy low-score suggestion behaviour of resolve_document_reference,
    now that extraction/query/scoring live in the shared resolver core.
    """
    if not top_candidates:
        return None
    try:
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMMessage

        candidates_desc = "\n".join([
            f"- {i+1}. **{c.get('document_title') or c.get('filename')}** "
            f"(Số: {c.get('document_number') or 'N/A'}, Ngày: {c.get('published_date') or 'N/A'})"
            for i, c in enumerate(top_candidates)
        ])
        suggestion_prompt = f"""Bạn là trợ lý tìm kiếm văn bản pháp luật Việt Nam.

Người dùng hỏi: "{reference}"

Các văn bản gần đúng tìm thấy (điểm thấp):
{candidates_desc}

Nhiệm vụ: Viết 1-2 câu gợi ý ngắn bằng tiếng Việt, hỏi người dùng có phải đang tìm một trong các văn bản trên không.
Nếu các văn bản không liên quan, gợi ý người dùng cung cấp thêm thông tin.
Chỉ trả lời bằng tiếng Việt, ngắn gọn (dưới 100 từ).
"""
        llm = get_llm_provider()
        resp = await llm.acomplete(
            messages=[LLMMessage(role="user", content=suggestion_prompt)],
            temperature=0.3,
            max_tokens=256,
        )
        suggestion = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
        logger.info(f"[resolve_document_reference] LLM suggestion for low-score: {suggestion[:100]}")
        return suggestion
    except Exception as e:
        logger.warning(f"[resolve_document_reference] LLM suggestion failed: {e}")
        return None


async def resolve_document_reference(
    reference: str,
    workspace_ids: list[int],
    db: "AsyncSession",
    topic: str | None = None,
) -> dict:
    """
    Resolve a document reference (e.g., "Luật An ninh mạng 2025") to candidate documents.

    Uses LLM to extract: document_number, document_title, document_type_slug, year.
    Then queries database with these structured fields.

    Args:
        topic: the FULL user question (e.g. "Nghị định 85 có bao nhiêu cấp độ hệ
               thống thông tin"). Its subject content drives the vector fallback
               and disambiguates same-number documents. When omitted, falls back
               to `reference`.

    Returns:
        dict with keys:
            - candidates: list of matched documents with scores
            - total: count of candidates
            - ambiguous: true if multiple candidates found
            - message: human-readable summary
            - section_reference: Điều/Chương/Khoản reference if present
            - suggestion: LLM 'did you mean' hint for low-score matches
    """
    logger.info(f"[resolve_document_reference] called with reference={reference!r}, topic={topic!r}, workspace_ids={workspace_ids}")
    try:
        # Delegate extraction/query/scoring to the shared resolver core so the
        # ReAct path and the LangGraph supervisor's resolve_doc_agent stay in sync.
        from app.services.agent.doc_resolver import resolve_candidates

        res = await resolve_candidates(reference, workspace_ids, db, topic=topic, use_llm_fallback=True)
        ranked = res["candidates"]
        section_reference = res.get("section_reference")

        # Map core candidates (score 0..1) → legacy candidate dicts (score 0..100)
        candidates: list[dict] = []
        for c in ranked[:5]:
            candidates.append({
                "document_id": c["document_id"],
                "filename": c.get("title"),
                "document_title": c.get("title"),
                "document_number": c.get("document_number", ""),
                "published_date": c.get("published_date", ""),
                "score": int(round(c.get("score", 0.0) * 100)),
                "strategies": c.get("strategies", []),
                "workspace_id": None,
            })

        # ── No match → similar-doc hint or generic not-found ─────────────────
        if not candidates:
            similar = res.get("similar", [])
            if similar:
                lines = [f"Không tìm thấy văn bản chính xác cho '{reference}'. Có thể bạn đang tìm:"]
                for i, d in enumerate(similar[:5], 1):
                    title = d.get("title", "")
                    num = d.get("document_number", "")
                    lines.append(f"{i}. **{title}**" + (f" (Số: {num})" if num else ""))
                message = "\n".join(lines)
            else:
                message = (
                    f"Không tìm thấy văn bản nào phù hợp với '{reference}'. "
                    f"Vui lòng cung cấp thêm thông tin (số văn bản, năm ban hành, "
                    f"hoặc tên đầy đủ) hoặc yêu cầu liệt kê các văn bản hiện có."
                )
            return {
                "candidates": [],
                "total": 0,
                "ambiguous": False,
                "message": message,
                "section_reference": section_reference,
                "suggestion": None,
            }

        top = candidates[0]
        top_score = top["score"]  # 0..100
        # Ambiguous ONLY when the TOP match is a strong DB hit (exact number/title)
        # AND a second candidate is near-tied. Vector-fallback guesses ("strategy":
        # "vector") are NOT a basis for stopping to ask the user — the ReAct agent
        # should auto-scope the best guess and answer (matches pre-refactor
        # behaviour, where tools.py resolve had no vector stage at all). Keying off
        # the strategy (not the score) is robust to the reranker's score scale.
        top_from_db = "db_query" in (top.get("strategies") or [])
        ambiguous = (
            len(candidates) > 1
            and top_from_db
            and (candidates[1]["score"] / max(top_score, 1)) >= 0.75
        )

        # ── Low-score LLM suggestion (ported legacy behaviour) ───────────────
        suggestion = None
        if top_score < 50:
            suggestion = await _suggest_low_score_match(reference, candidates[:3])

        # ── Human-readable message ───────────────────────────────────────────
        # Decisive when NOT ambiguous so the agent searches within the scoped
        # document instead of enumerating look-alikes and asking the user.
        if not ambiguous:
            # NOTE: Do NOT put the document UUID in this message. The LLM reads
            # it as a tool observation and, when told to cite sources, fabricates
            # a citation from the first 8 hex of the UUID (e.g. [75a75810]) that
            # the frontend can't resolve and leaks as raw text. Scoping is handled
            # programmatically via ctx.document_ids / data.resolved_document_ids.
            message = (
                f"Đã xác định văn bản: **{top['document_title'] or top['filename']}**. "
                f"Hãy tìm nội dung được hỏi TRONG văn bản này, "
                f"không cần hỏi lại người dùng."
            )
        else:
            msg_parts = [f"Tìm thấy **{len(candidates)} văn bản** có thể phù hợp:"]
            for i, c in enumerate(candidates, 1):
                title = c["document_title"] or c["filename"]
                # UUID intentionally omitted — see note above. The LLM picks by
                # index/title; document scoping is resolved in code, not by the LLM.
                msg_parts.append(
                    f"{i}. **{title}** (score: {c['score']})"
                )
            msg_parts.append("\n⚠️ Có nhiều văn bản tương tự. Hãy chọn phù hợp nhất hoặc hỏi người dùng.")
            message = "\n".join(msg_parts)

        return {
            "candidates": candidates,
            "total": len(candidates),
            "ambiguous": ambiguous,
            "message": message,
            "section_reference": section_reference,
            "suggestion": suggestion,
        }

    except Exception as e:
        logger.error(f"[tool:resolve_document_reference] Failed: {e}", exc_info=True)
        return {
            "candidates": [],
            "total": 0,
            "ambiguous": False,
            "message": f"Lỗi tìm kiếm văn bản: {e}",
            "section_reference": None,
            "suggestion": None,
        }

async def search_document_section(
    section_reference: str,
    workspace_ids: list[str],
    document_ids: list[str] | None = None
) -> dict:
    """
    Tìm kiếm và lấy nội dung chính xác của một phần/mục/chương/điều
    dựa vào metadata 'heading_path'.
    """
    from app.services.embedding.vector_store import get_vector_store
    from app.services.embedding.embedder import get_embedding_service
    from app.api.chat_agent import _generate_citation_id
    from app.schemas.rag import ChatSourceChunk
    
    all_chunks = []
    
    # Metadata filter: if we have document_ids, fetch all their chunks first.
    # Python-side filtering is more robust than ChromaDB's limited metadata operators.
    if document_ids:
        if len(document_ids) == 1:
            where_filter = {"document_id": str(document_ids[0])}
        else:
            where_filter = {"document_id": {"$in": [str(d) for d in document_ids]}}
    else:
        # Fallback to broad search if no specific document is resolved
        where_filter = None

    for ws_id in workspace_ids:
        try:
            vstore = get_vector_store(ws_id)
            
            # 1. Try structural lookup via metadata
            res = vstore.get_by_metadata(where=where_filter) if where_filter else {"documents": [], "metadatas": []}
            
            if res.get("documents") and res.get("metadatas"):
                # Filter in Python for substring match in heading_path
                ref_norm = section_reference.lower().strip().rstrip(".")

                # Precise matching using regex to avoid partial matches (e.g., "Điều 3" vs "Điều 30")
                import re
                ref_pattern = rf"(?i)\b{re.escape(ref_norm)}(?!\d)\b" # Match whole word, no digit immediately after

                # Tham chiếu có SỐ ĐIỀU ("Điều 17", "Khoản 2 Điều 8") → match
                # chính xác trên metadata article_nos ("17|18"); chunk chưa có
                # article_nos (chưa backfill) rơi về regex heading_path.
                art_ref = re.search(r"(?i)\bđiều\s+(\d+[a-zA-Z]?)\b", ref_norm)
                want_art = art_ref.group(1).lower() if art_ref else None

                for doc, meta in zip(res["documents"], res["metadatas"]):
                    path = meta.get("heading_path", "")
                    art_nos = meta.get("article_nos") or ""
                    match = False

                    if want_art and art_nos:
                        match = want_art in art_nos.lower().split("|")
                    elif isinstance(path, str) and path:
                        # Split by separator and check components
                        components = [c.strip() for c in path.split(">")]
                        if any(re.search(ref_pattern, c) for c in components):
                            match = True
                    elif isinstance(path, list):
                        if any(re.search(ref_pattern, str(c)) for c in path):
                            match = True

                    if match:
                        all_chunks.append({"content": doc, "metadata": meta})
            
            # 2. If metadata search found nothing, fallback to semantic search restricted to the document
            if not all_chunks and document_ids:
                logger.info(f"[tool:search_document_section] Metadata search failed for '{section_reference}', trying semantic fallback")
                embedder = get_embedding_service()
                query_emb = embedder.embed_query(section_reference)
                
                # Broaden the filter for semantic search
                sem_where = where_filter
                sem_res = vstore.query(query_embedding=query_emb, n_results=10, where=sem_where)
                
                if sem_res.get("documents"):
                    for doc, meta in zip(sem_res["documents"], sem_res["metadatas"]):
                        all_chunks.append({"content": doc, "metadata": meta})
                        
        except Exception as e:
            logger.error(f"[tool:search_document_section] Error fetching from workspace {ws_id}: {e}")
            
    if not all_chunks:
        return {
            "text": f"Không tìm thấy dữ liệu nào khớp với `{section_reference}` trong (các) văn bản được yêu cầu.",
            "sources": []
        }
            
    # Sắp xếp các chunk theo thứ tự xuất hiện trong tài liệu (page_no, chunk_index)
    all_chunks.sort(key=lambda c: (
        str(c["metadata"].get("document_id", "")),
        int(c["metadata"].get("page_no", 0)),
        int(c["metadata"].get("chunk_index", 0))
    ))
            
    # Deduplicate chunks to prevent overlapping text
    unique_chunks = []
    seen_content = set()
    for c in all_chunks:
        content_hash = hash(c["content"].strip())
        if content_hash not in seen_content:
            unique_chunks.append(c)
            seen_content.add(content_hash)
            
    # Nối text lại để Answer Generator tóm tắt
    combined_text = "\n\n".join([c["content"] for c in unique_chunks])
    
    # Convert to ChatSourceChunk objects for consistent RAG pipeline
    sources = []
    existing_ids = set()
    
    for c in unique_chunks:
        meta = c["metadata"]
        doc_id_raw = meta.get("document_id")
        
        # Coerce document_id to UUID
        try:
            if isinstance(doc_id_raw, str):
                doc_id = uuid.UUID(doc_id_raw)
            else:
                doc_id = doc_id_raw if isinstance(doc_id_raw, uuid.UUID) else uuid.uuid4()
        except Exception:
            doc_id = uuid.uuid4()

        # Handle heading_path
        h_path = meta.get("heading_path", [])
        if isinstance(h_path, str):
            h_path = [p.strip() for p in h_path.split(">")] if ">" in h_path else [h_path]
        elif not h_path:
            h_path = ["Tài liệu"]

        chunk_obj = ChatSourceChunk(
            index=_generate_citation_id(existing_ids),
            chunk_id=str(meta.get("id") or meta.get("chunk_id") or uuid.uuid4()),
            content=c["content"],
            document_id=doc_id,
            page_no=int(meta.get("page_no", 0)),
            heading_path=h_path,
            score=float(c.get("score", 0.0)),
            source_type="vector",
            source_file=str(meta.get("filename") or meta.get("original_filename") or "Tài liệu")
        )
        sources.append(chunk_obj)
        existing_ids.add(chunk_obj.index)

    return {
        "text": combined_text,
        "sources": sources
    }

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
) -> dict:
    """
    Hybrid search across workspaces via HRAG pipeline.

    Returns:
        dict with keys: context_text, sources, images, image_parts, kg_summaries
    """
    from app.api.chat_agent import _execute_search_documents

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
    )
    return {
        "context_text": context_text,
        "sources": [s.model_dump() for s in sources],
        "images": [i.model_dump() for i in image_refs],
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
        for ws_id, ws_docs in by_ws.items():
            ws_name = ws_map.get(ws_id, f"KB {ws_id}")
            lines.append(f"\n### Workspace: {ws_name}")
            for i, doc in enumerate(ws_docs, 1):
                page_info = f", {doc.page_count} trang" if doc.page_count else ""
                chunk_info = f", {doc.chunk_count} đoạn" if doc.chunk_count else ""
                lines.append(
                    f"{i}. **{doc.original_filename}** (ID: {doc.id})"
                    f"{page_info}{chunk_info}"
                )

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
    from app.services.knowledge_graph_service import get_kg_service

    results = []
    for ws_id in workspace_ids:
        try:
            kg_service = get_kg_service(workspace_id=ws_id)
            # Use naive mode for entity lookup (faster than hybrid)
            kg_result = await kg_service.query(
                query=entity,
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
        doc_list = []
        for i, doc in enumerate(docs, 1):
            doc_info = {
                "id": doc.id,
                "filename": doc.original_filename,
                "document_number": doc.document_number,
            }
            doc_list.append(doc_info)
            lines.append(
                f"{i}. **{doc.original_filename}**\n"
                f"   Số văn bản: {doc.document_number or 'N/A'}\n"
                f"   ID: {doc.id}"
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
    from app.services.mongo_people_service import search_by_cccd as _svc

    try:
        async for res in _svc(cccd):
            yield res
    except Exception as e:
        logger.error(f"[tool:search_people_by_cccd] Failed: {e}")
        yield {"found": False, "persons": [], "display": f"Lỗi tìm kiếm CCCD: {e}"}


async def search_people_by_name(name: str, limit: int = 10):
    """
    Search for persons by name (ho_ten). Yields partial results.
    Case-insensitive partial regex match.

    Returns:
        dict with keys: found, count, persons, display
    """
    from app.services.mongo_people_service import search_by_name as _svc

    try:
        async for res in _svc(name, limit=limit):
            yield res
    except Exception as e:
        logger.error(f"[tool:search_people_by_name] Failed: {e}")
        yield {
            "found": False,
            "count": 0,
            "persons": [],
            "display": f"Lỗi tìm kiếm tên: {e}",
        }


async def search_people_by_bhxh(so_bhxh: str):
    """
    Search for a person by BHXH (Bảo hiểm xã hội) number.
    Exact or loose regex match. Yields partial results.

    Returns:
        dict with keys: found, person, display
    """
    from app.services.mongo_people_service import search_by_bhxh as _svc

    try:
        async for res in _svc(so_bhxh):
            yield res
    except Exception as e:
        logger.error(f"[tool:search_people_by_bhxh] Failed: {e}")
        yield {"found": False, "person": None, "display": f"Lỗi tìm kiếm BHXH: {e}"}


async def search_people_by_phone(phone: str, limit: int = 10):
    """
    Search for persons by phone number (so_dien_thoai).
    Exact, ends-with, or contains match. Yields partial results.

    Returns:
        dict with keys: found, count, persons, display
    """
    from app.services.mongo_people_service import search_by_phone as _svc

    try:
        async for res in _svc(phone, limit=limit):
            yield res
    except Exception as e:
        logger.error(f"[tool:search_people_by_phone] Failed: {e}")
        yield {
            "found": False,
            "count": 0,
            "persons": [],
            "display": f"Lỗi tìm kiếm SĐT: {e}",
        }

async def search_people_advanced(criteria: dict, limit: int = 10):
    """
    Search for persons by multiple criteria (Name + DoB + Address + etc).
    Yields partial results.
    """
    from app.services.mongo_people_service import search_by_advanced as _svc

    try:
        async for res in _svc(criteria, limit=limit):
            yield res
    except Exception as e:
        logger.error(f"[tool:search_people_advanced] Failed: {e}")
        yield {
            "found": False,
            "count": 0,
            "persons": [],
            "display": f"Lỗi tìm kiếm phức tạp: {e}",
        }


# ---------------------------------------------------------------------------
# Document Reference Resolution Tool
# ---------------------------------------------------------------------------

import re
from sqlalchemy import select, and_, or_
from app.models.document import Document, DocumentStatus
from app.models.document_type import DocumentType


# Vietnamese document type keywords → slug mapping
_DOC_TYPE_KEYWORDS: dict[str, str] = {
    "luật": "luat",
    "nghị định": "nghi_dinh",
    "thông tư": "thong_tu",
    "quyết định": "quyet_dinh",
    "nghị quyết": "nghi_quyet",
    "pháp lệnh": "phap_lenh",
    "công văn": "cong_van",
    "chỉ thị": "chi_thi",
    "báo cáo": "bao_cao",
    "tờ trình": "to_trinh",
    "biên bản": "bien_ban",
    "hợp đồng": "hop_dong",
    "kế hoạch": "ke_hoach",
    "hướng dẫn": "huong_dan",
    "đơn": "don_tu",
    "thông báo": "thong_bao",
}


def _parse_document_reference(reference: str) -> dict:
    """
    Parse a document reference string into structured components.
    E.g. "Luật An ninh mạng 2025" → {doc_type_slug: "luat", year: "2025", title_keywords: ["An", "ninh", "mạng"]}
    E.g. "Tóm tắt điều 27 Luật An ninh mạng 2018" → {doc_type_slug: "luat", year: "2018", title_keywords: ["An", "ninh", "mạng"]}
    """
    text = reference.strip()

    # 0. Remove common action phrases at the start (NOT part of document name)
    import re
    action_patterns = [
        r'^tóm\s*tắt\s*(điều\s*)?\d*\s*',   # "tóm tắt điều 27"
        r'^tra\s*cứu\s+',                      # "tra cứu"
        r'^tìm\s+',                           # "tìm"
        r'^xem\s+',                           # "xem"
        r'^liệt\s*kê\s*',                    # "liệt kê"
        r'^tổng\s*hợp\s*',                   # "tổng hợp"
        r'^nội\s*dung\s+',                    # "nội dung"
    ]
    for pattern in action_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # 1. Detect document type
    doc_type_slug = None
    remaining = text
    for keyword, slug in _DOC_TYPE_KEYWORDS.items():
        if keyword.lower() in text.lower():
            doc_type_slug = slug
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            remaining = pattern.sub("", remaining, count=1).strip()
            break

    # 2. Extract year (4-digit number in range 1900-2099)
    year = None
    year_match = re.search(r'\b(19|20)\d{2}\b', remaining)
    if year_match:
        year = year_match.group()
        remaining = remaining.replace(year, "", 1).strip()

    # 3. Remove ordinal patterns like "điều 27", standalone numbers
    remaining = re.sub(r'\bđiều\s*\d+\b', "", remaining, flags=re.IGNORECASE)
    remaining = re.sub(r'\b\d+\b', "", remaining)
    remaining = re.sub(r'\s+', " ", remaining).strip()

    # 4. Extract title keywords (non-empty words/phrases, skip short ones)
    title_keywords = [kw.strip() for kw in remaining.split() if kw.strip() and len(kw.strip()) > 1]

    return {
        "doc_type_slug": doc_type_slug,
        "year": year,
        "title_keywords": title_keywords,
        "original": reference,
    }


def _score_document(doc: Document, parsed: dict) -> tuple[int, str]:
    """
    Score how well a document matches the parsed reference.
    Returns (score, match_details).
    """
    score = 0
    details = []

    # Match: document type
    if parsed["doc_type_slug"] and doc.document_type:
        if doc.document_type.slug == parsed["doc_type_slug"]:
            score += 40
            details.append("type: ✓")
        elif doc.document_type.slug == parsed["doc_type_slug"]:
            score += 20
            details.append("type: ~")

    # Match: title keywords (all must appear in document_title)
    if parsed["title_keywords"] and doc.document_title:
        title_lower = doc.document_title.lower()
        matched_kws = 0
        for kw in parsed["title_keywords"]:
            if kw.lower() in title_lower:
                matched_kws += 1
        if matched_kws > 0:
            kw_ratio = matched_kws / len(parsed["title_keywords"])
            score += int(30 * kw_ratio)
            details.append(f"title: {matched_kws}/{len(parsed['title_keywords'])}")

    # Match: year in published_date (compare 4-digit year only)
    if parsed["year"] and doc.published_date:
        import re
        date_years = re.findall(r'(?:19|20)\d{2}', doc.published_date)
        if parsed["year"] in date_years:
            score += 20
            details.append("year: ✓")

    # Bonus: document_number contains any keyword
    if parsed["title_keywords"] and doc.document_number:
        doc_num_lower = doc.document_number.lower()
        if any(kw.lower() in doc_num_lower for kw in parsed["title_keywords"]):
            score += 10
            details.append("doc_num: ✓")

    # Build year_match display for logger
    import re
    date_years = re.findall(r'(?:19|20)\d{2}', doc.published_date) if doc.published_date else []
    year_match_log = parsed["year"] in date_years if parsed["year"] else False

    logger.info(f"[_score_document] doc_type={doc.document_type.slug if doc.document_type else None}, parsed_type={parsed['doc_type_slug']}, title_match={matched_kws if parsed['title_keywords'] and doc.document_title else 'N/A'}, year_match={year_match_log} (doc_date_years={date_years}, query_year={parsed['year']}) -> final_score={score}")
    return score, ", ".join(details) if details else "no match"


async def resolve_document_reference(
    reference: str,
    workspace_ids: list[int],
    db: "AsyncSession",
) -> dict:
    """
    Resolve a document reference (e.g., "Luật An ninh mạng 2025") to candidate documents.

    Uses LLM to extract: document_number, document_title, document_type_slug, year.
    Then queries database with these structured fields.

    Returns:
        dict with keys:
            - candidates: list of matched documents with scores
            - total: count of candidates
            - ambiguous: true if multiple candidates found
            - message: human-readable summary
            - llm_parsed: the LLM extraction result (for debugging)
    """
    logger.info(f"[resolve_document_reference] called with reference={reference!r}, workspace_ids={workspace_ids}")
    try:
        # ── Step 1: LLM extraction ──────────────────────────────────────────────
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMMessage

        llm = get_llm_provider()
        extraction_prompt = (f"""
Bạn là hệ thống chuẩn hóa truy vấn văn bản pháp luật Việt Nam để tìm kiếm database.

Nhiệm vụ:
Đọc câu hỏi người dùng và trích xuất metadata phục vụ search.

NGUYÊN TẮC QUAN TRỌNG:

1. Người dùng hỏi "Luật" thường nhớ tên:
- Luật An ninh mạng
- Luật BHXH 2025

=> ưu tiên document_title

2. Người dùng hỏi "Nghị định", "Thông tư", "Quyết định" thường nhớ số:
- Nghị định 361
- Nghị định 53/2022/NĐ-CP
- Thông tư 23

=> ưu tiên document_number

3. Nếu có cả số + tên thì lấy cả hai.

4. Nếu có Điều/Khoản/Chương thì đưa vào section_reference.

5. Nếu hỏi văn bản hướng dẫn / về một luật khác:
- nghị định về Luật An ninh mạng
- thông tư hướng dẫn BHYT

=> related_to = tên luật/chủ đề

Loại văn bản map slug:
Luật -> luat
Bộ luật -> bo_luat
Nghị định -> nghi_dinh
Thông tư -> thong_tu
Quyết định -> quyet_dinh
Nghị quyết -> nghi_quyet
Chỉ thị -> chi_thi

Trả JSON:

{{
  "document_type": "",
  "document_type_slug": "",
  "document_number": "",
  "document_title": "",
  "year": "",
  "section_reference": "",
  "related_to": "",
  "search_priority": []
}}

search_priority rules:
- Luật => ["title","year"]
- Nghị định/Thông tư => ["number","year","title"]
- Nếu chỉ hỏi chủ đề => ["related_to"]

Câu hỏi: "{reference}"

JSON:
""")
        try:
            llm_response = await llm.acomplete(
                messages=[LLMMessage(role="user", content=extraction_prompt)],
                temperature=0.1,
                max_tokens=256,
            )
            llm_text = llm_response if isinstance(llm_response, str) else getattr(llm_response, "content", str(llm_response))
            logger.info(f"[tool:resolve_document_reference] LLM extraction: {llm_text!r}")

            # Parse JSON from LLM response
            import json, re
            # Try to extract JSON from response (handle potential code blocks)
            json_match = re.search(r'\{[^{}]*\}', llm_text, re.DOTALL)
            if json_match:
                llm_parsed = json.loads(json_match.group())
            else:
                llm_parsed = json.loads(llm_text.strip())

            doc_number = llm_parsed.get("document_number") or None
            doc_title = llm_parsed.get("document_title") or None
            doc_type_slug = llm_parsed.get("document_type_slug") or None
            year = llm_parsed.get("year") or None
            section_reference = llm_parsed.get("section_reference") or None
            ref_doc_title = llm_parsed.get("referenced_document_title") or None
            logger.info(f"[tool:resolve_document_reference] LLM extracted: doc_number={doc_number!r}, doc_title={doc_title!r}, doc_type_slug={doc_type_slug!r}, year={year!r}, section_reference={section_reference!r}, ref_doc_title={ref_doc_title!r}")

            # Guard: clear doc_number if it looks like a title (contains common title words)
            # Vietnamese document numbers like "361/2025/NĐ-CP" or "NĐ-CP" are VALID and must be preserved
            title_keywords = ["về", "theo", "ban hành", "quy định", "hướng dẫn", "của", "nghị định", "thông tư", "luật", "quyết định"]
            if doc_number and any(kw in doc_number.lower() for kw in title_keywords):
                logger.warning(f"[tool:resolve_document_reference] doc_number={doc_number!r} looks like a title - clearing")
                doc_number = None

        except Exception as e:
            logger.warning(f"[tool:resolve_document_reference] LLM extraction failed: {e}, falling back to regex parser")
            parsed = _parse_document_reference(reference)
            doc_number = None
            doc_title = None
            ref_doc_title = None
            doc_type_slug = parsed["doc_type_slug"]
            year = parsed["year"]
            llm_parsed = {"fallback": True, "error": str(e)}

        # ── Step 2: Build query from LLM-extracted fields ───────────────────────
        query = select(Document).where(
            Document.workspace_id.in_(workspace_ids),
            Document.status == DocumentStatus.INDEXED,
        )

        # Filter by document_type if extracted
        if doc_type_slug:
            query = query.join(
                DocumentType,
                Document.document_type_id == DocumentType.id,
            ).where(DocumentType.slug == doc_type_slug)

        # Filter by document_number (exact or partial match)
        if doc_number:
            # Normalize: strip leading zeros and common prefixes for flexible matching
            normalized_num = doc_number.strip()
            query = query.where(
                or_(
                    Document.document_number == normalized_num,
                    Document.document_number.ilike(f"%{normalized_num}%"),
                    Document.document_number.ilike(f"%/{normalized_num}%"),
                )
            )

        # Filter by document_title keywords (all must match)
        if doc_title:
            title_keywords = [kw.strip() for kw in doc_title.split() if kw.strip()]
            title_conditions = [
                Document.document_title.ilike(f"%{kw}%")
                for kw in title_keywords
            ]
            if title_conditions:
                query = query.where(and_(*title_conditions))

        # Filter by referenced_document_title (extra context, e.g. "Nghị định 53 về Luật X" → search X)
        if ref_doc_title:
            ref_keywords = [kw.strip() for kw in ref_doc_title.split() if kw.strip()]
            ref_conditions = [
                Document.document_title.ilike(f"%{kw}%")
                for kw in ref_keywords
            ]
            if ref_conditions:
                from sqlalchemy import or_ as or_cond
                query = query.where(or_cond(*ref_conditions))

        # Filter by year in published_date
        # Only filter by year if NOT filtering by doc_number (year in "361/2025" is decree ID, not publish year)
        if year and not doc_number:
            query = query.where(Document.published_date.ilike(f"%{year}%"))

        # Execute query
        logger.info(f"[resolve_document_reference] executing query: doc_number={doc_number!r}, doc_title={doc_title!r}, doc_type_slug={doc_type_slug!r}, year={year!r}, ref_doc_title={ref_doc_title!r}")
        result = await db.execute(query.order_by(Document.created_at.desc()).limit(50))
        docs = result.scalars().all()
        logger.info(f"[resolve_document_reference] query returned {len(docs)} documents")

        # Score each document (LLM fields used directly, no regex parsing needed)
        scored_docs = []
        for doc in docs:
            doc_title_val = doc.document_title or ""
            doc_num_val = doc.document_number or ""
            doc_date_val = doc.published_date or ""
            doc_type_val = doc.document_type.slug if doc.document_type else "none"
            logger.info(f"[resolve_document_reference] scoring doc: id={doc.id}, title={doc_title_val!r}, doc_num={doc_num_val!r}, date={doc_date_val!r}, type={doc_type_val}")
            score = 0
            details = []

            # Document type match
            if doc_type_slug and doc.document_type:
                if doc.document_type.slug == doc_type_slug:
                    score += 40
                    details.append("type: ✓")

            # Document number exact/partial match
            if doc_number and doc_num_val:
                if doc_num_val.strip() == doc_number.strip():
                    score += 40
                    details.append("doc_num: exact ✓")
                elif doc_number.strip() in doc_num_val.strip():
                    score += 25
                    details.append("doc_num: partial ✓")

            # Document title match (keyword overlap)
            if doc_title:
                title_lower = doc_title.lower()
                title_val_lower = doc_title_val.lower()
                matched = sum(1 for kw in title_lower.split() if kw in title_val_lower)
                total = len(title_lower.split())
                if matched > 0:
                    score += int(30 * matched / total)
                    details.append(f"title: {matched}/{total}")

            # Year match in published_date
            if year and doc_date_val:
                if year in doc_date_val:
                    score += 20
                    details.append("year: ✓")

            # Referenced document title match (extra context bonus)
            if ref_doc_title:
                ref_lower = ref_doc_title.lower()
                title_val_lower = doc_title_val.lower()
                ref_matched = sum(1 for kw in ref_lower.split() if kw in title_val_lower)
                ref_total = len(ref_lower.split())
                if ref_matched > 0:
                    score += int(20 * ref_matched / ref_total)
                    details.append(f"ref_title: {ref_matched}/{ref_total}")

            logger.info(f"[resolve_document_reference]   -> score={score}, details={', '.join(details) if details else 'no match'}")
            if score > 0:
                doc_type_name = doc.document_type.name if doc.document_type else "Unknown"
                scored_docs.append({
                    "document_id": str(doc.id),
                    "filename": doc.original_filename,
                    "document_title": doc.document_title,
                    "document_number": doc.document_number,
                    "doc_type": doc_type_name,
                    "published_date": doc.published_date,
                    "score": score,
                    "match_details": ", ".join(details) if details else "no match",
                    "workspace_id": str(doc.workspace_id),
                })

        # Sort by score descending
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # ── Step 3: Low-score fallback with LLM suggestion ──────────────────────
        # If top candidates have low scores (< 50), use LLM to generate suggestions
        low_score_threshold = 50
        suggestion = None
        if scored_docs and scored_docs[0]["score"] < low_score_threshold:
            top_candidates = scored_docs[:3]  # Top 3 for suggestion
            try:
                from app.services.llm import get_llm_provider
                from app.services.llm.types import LLMMessage

                llm = get_llm_provider()
                candidates_desc = "\n".join([
                    f"- {i+1}. **{c['document_title'] or c['filename']}** (Số: {c['document_number'] or 'N/A'}, "
                    f"Loại: {c['doc_type']}, Ngày: {c['published_date'] or 'N/A'})"
                    for i, c in enumerate(top_candidates)
                ])

                suggestion_prompt = f"""Bạn là trợ lý tìm kiếm văn bản pháp luật Việt Nam.

Người dùng hỏi: "{reference}"
LLM đã trích xuất: Số văn bản="{doc_number or 'N/A'}", Loại="{doc_type_slug or 'N/A'}", Năm="{year or 'N/A'}"

Các văn bản gần đúng tìm thấy (điểm thấp):
{candidates_desc}

Nhiệm vụ: Viết 1-2 câu gợi ý ngắn bằng tiếng Việt, hỏi người dùng có phải đang tìm một trong các văn bản trên không.
Nếu các văn bản không liên quan, gợi ý người dùng cung cấp thêm thông tin.

Ví dụ:
- "Có phải bạn đang tìm **Nghị định 361/2025/NĐ-CP** ban hành ngày 09/01/2026 về 'Quy định về vị trí việc làm công chức' không?"
- "Không tìm thấy văn bản chính xác. Bạn có đang tìm một trong các văn bản trên không?"

Chỉ trả lời bằng tiếng Việt, ngắn gọn (dưới 100 từ).
"""
                suggestion_response = await llm.acomplete(
                    messages=[LLMMessage(role="user", content=suggestion_prompt)],
                    temperature=0.3,
                    max_tokens=256,
                )
                suggestion = suggestion_response if isinstance(suggestion_response, str) else getattr(suggestion_response, "content", str(suggestion_response))
                logger.info(f"[resolve_document_reference] LLM suggestion for low-score: {suggestion[:100]}")
            except Exception as e:
                logger.warning(f"[resolve_document_reference] LLM suggestion failed: {e}")

        # Build response
        if not scored_docs:
            # No match found - return suggestion to list documents
            return {
                "candidates": [],
                "total": 0,
                "ambiguous": False,
                "message": (
                    f"Không tìm thấy văn bản nào phù hợp với '{reference}'. "
                    f"Vui lòng cung cấp thêm thông tin (số văn bản, năm ban hành, "
                    f"hoặc tên đầy đủ) hoặc yêu cầu liệt kê các văn bản hiện có."
                ),
                "llm_parsed": {"doc_number": doc_number, "doc_title": doc_title, "doc_type_slug": doc_type_slug, "year": year},
                "section_reference": section_reference,
                "suggestion": suggestion,
            }

        # Take top 5 candidates
        candidates = scored_docs[:5]
        top_score = candidates[0]["score"]

        # If multiple candidates have same top score → ambiguous
        ambiguous = len([c for c in candidates if c["score"] == top_score]) > 1

        if len(candidates) == 1:
            message = (
                f"Tìm thấy văn bản: **{candidates[0]['document_title'] or candidates[0]['filename']}** "
                f"(ID: {candidates[0]['document_id']})"
            )
        else:
            msg_parts = [f"Tìm thấy **{len(candidates)} văn bản** phù hợp:"]
            for i, c in enumerate(candidates, 1):
                title = c["document_title"] or c["filename"]
                msg_parts.append(
                    f"{i}. **{title}** (ID: {c['document_id']}, "
                    f"type: {c['doc_type']}, score: {c['score']})"
                )
            if ambiguous:
                msg_parts.append("\n⚠️ Có nhiều văn bản tương tự. Agent nên chọn phù hợp nhất hoặc hỏi user.")
            message = "\n".join(msg_parts)

        return {
            "candidates": candidates,
            "total": len(candidates),
            "ambiguous": ambiguous,
            "message": message,
            "llm_parsed": {"doc_number": doc_number, "doc_title": doc_title, "doc_type_slug": doc_type_slug, "year": year},
            "section_reference": section_reference,
            "suggestion": suggestion,  # LLM suggestion for low-score matches
        }

    except Exception as e:
        logger.error(f"[tool:resolve_document_reference] Failed: {e}")
        return {
            "candidates": [],
            "total": 0,
            "ambiguous": False,
            "message": f"Lỗi tìm kiếm văn bản: {e}",
            "llm_parsed": None,
            "section_reference": None,
            "suggestion": None,
        }


async def search_document_section(
    section_reference: str,
    workspace_ids: list[int],
    document_ids: list[int] | None = None
) -> dict:
    """
    Tìm kiếm và lấy nội dung chính xác của một phần/mục/chương/điều
    dựa vào metadata 'heading_path'.
    """
    from app.services.vector_store import get_vector_store
    
    all_chunks = []
    
    # Sử dụng $contains để lọc các chunk mà có heading_path chứa section_reference.
    # Thường heading_path lưu dạng chuỗi JSON hoặc list flattened như "Chương 3 > Điều 27"
    # Nên dùng $contains với substring match.
    where_filter = {"heading_path": {"$contains": section_reference}}
    
    # Nâng cao: Nếu có document_ids, kết hợp thêm điều kiện $and
    if document_ids:
        if len(document_ids) == 1:
            where_filter = {
                "$and": [
                    {"document_id": str(document_ids[0])},
                    {"heading_path": {"$contains": section_reference}}
                ]
            }
        else:
            where_filter = {
                "$and": [
                    {"document_id": {"$in": [str(d) for d in document_ids]}},
                    {"heading_path": {"$contains": section_reference}}
                ]
            }

    for ws_id in workspace_ids:
        try:
            # Need to get standard UUID format or string
            vstore = get_vector_store(ws_id)
            res = vstore.get_by_metadata(where=where_filter)
            if res.get("documents") and res.get("metadatas"):
                for doc, meta in zip(res["documents"], res["metadatas"]):
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
        c["metadata"].get("document_id", ""),
        int(c["metadata"].get("page_no", 0)),
        int(c["metadata"].get("chunk_index", 0))
    ))
            
    # Nối text lại để Answer Generator tóm tắt
    combined_text = "\n\n".join([c["content"] for c in all_chunks])
    
    # Có thể bị lặp metadata nếu nhiều chunk cùng 1 trang, ta lọc distinct theo id hoặc ref
    sources = []
    seen = set()
    for c in all_chunks:
        meta = c["metadata"]
        key = f"{meta.get('document_id')}_{meta.get('page_no')}"
        if key not in seen:
            sources.append(meta)
            seen.add(key)
            
    return {
        "text": combined_text,
        "sources": sources
    }

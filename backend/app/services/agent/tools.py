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
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.services.agent.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool registry — maps tool name → callable
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, str] = {
    "search_documents": "search the knowledge base for relevant document sections",
    "list_documents": "list all documents available in the current workspace",
    "summarize_document": "get a comprehensive summary of a specific document",
    "query_knowledge_graph": "query the knowledge graph for entity relationships",
    "search_documents_number": "search for documents by their official document number (văn bản số)",
    "search_abbreviation": "search for the meaning of an abbreviation or acronym",
    # MongoDB people search tools
    "search_people_by_cccd": "search for a person by their CCCD (Căn cước công dân) national ID number",
    "search_people_by_name": "search for persons by their full name or partial name",
    "search_people_by_bhxh": "search for a person by their BHXH (Bảo hiểm xã hội) number",
    "search_people_by_phone": "search for persons by their phone number",
}


# ---------------------------------------------------------------------------
# Tool 1: search_documents
# Wraps the existing _execute_search_documents from chat_agent.py
# ---------------------------------------------------------------------------


async def search_documents(
    query: str,
    top_k: int,
    workspace_ids: list[int],
    existing_citation_ids: set,
    db: "AsyncSession",
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

        if doc.status != DocumentStatus.INDEXED:
            return {
                "text": f"Tài liệu '{doc.original_filename}' chưa được lập chỉ mục.",
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


async def search_people_by_cccd(cccd: str) -> dict:
    """
    Search for a person by CCCD (Căn cước công dân) number.
    Exact match on the cccd field.

    Returns:
        dict with keys: found, person, display
    """
    from app.services.mongo_people_service import search_by_cccd as _svc
    try:
        return await _svc(cccd)
    except Exception as e:
        logger.error(f"[tool:search_people_by_cccd] Failed: {e}")
        return {"found": False, "person": None, "display": f"Lỗi tìm kiếm CCCD: {e}"}


async def search_people_by_name(name: str, limit: int = 10) -> dict:
    """
    Search for persons by name (ho_ten).
    Case-insensitive partial regex match.

    Returns:
        dict with keys: found, count, persons, display
    """
    from app.services.mongo_people_service import search_by_name as _svc
    try:
        return await _svc(name, limit=limit)
    except Exception as e:
        logger.error(f"[tool:search_people_by_name] Failed: {e}")
        return {
            "found": False,
            "count": 0,
            "persons": [],
            "display": f"Lỗi tìm kiếm tên: {e}",
        }


async def search_people_by_bhxh(so_bhxh: str) -> dict:
    """
    Search for a person by BHXH (Bảo hiểm xã hội) number.
    Exact or loose regex match.

    Returns:
        dict with keys: found, person, display
    """
    from app.services.mongo_people_service import search_by_bhxh as _svc
    try:
        return await _svc(so_bhxh)
    except Exception as e:
        logger.error(f"[tool:search_people_by_bhxh] Failed: {e}")
        return {"found": False, "person": None, "display": f"Lỗi tìm kiếm BHXH: {e}"}


async def search_people_by_phone(phone: str, limit: int = 10) -> dict:
    """
    Search for persons by phone number (so_dien_thoai).
    Exact, ends-with, or contains match.

    Returns:
        dict with keys: found, count, persons, display
    """
    from app.services.mongo_people_service import search_by_phone as _svc
    try:
        return await _svc(phone, limit=limit)
    except Exception as e:
        logger.error(f"[tool:search_people_by_phone] Failed: {e}")
        return {
            "found": False,
            "count": 0,
            "persons": [],
            "display": f"Lỗi tìm kiếm SĐT: {e}",
        }

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

    context_text, sources, image_refs, image_parts, kg_summaries = (
        await _execute_search_documents(
            workspace_ids=workspace_ids,
            query=query,
            top_k=top_k,
            db=db,
            existing_ids=existing_citation_ids,
        )
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
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
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
        markdown_bytes = await storage.download_file(doc.markdown_s3_key)
        markdown_text = markdown_bytes.decode("utf-8", errors="replace")

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
        summary_text = summary if isinstance(summary, str) else getattr(summary, "content", str(summary))

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
    from app.services.knowledge_graph_service import KnowledgeGraphService

    results = []
    for ws_id in workspace_ids:
        try:
            kg_service = KnowledgeGraphService(workspace_id=ws_id)
            # Use naive mode for entity lookup (faster than hybrid)
            kg_result = await kg_service.query(
                query=entity,
                mode="naive",
            )
            if kg_result and kg_result.strip():
                results.append(f"**Workspace {ws_id}:**\n{kg_result}")
        except Exception as e:
            logger.warning(f"[tool:query_knowledge_graph] KG query failed for ws {ws_id}: {e}")

    if not results:
        return {
            "text": f"Không tìm thấy thông tin về '{entity}' trong knowledge graph."
        }

    return {"text": "\n\n".join(results)}

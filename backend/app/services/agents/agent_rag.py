"""
Agent RAG
========

RAG functionality for document search, knowledge graph queries, and document operations.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


class AgentRagState(BaseModel):
    messages: list = []
    intent: str | None = None
    rewritten_query: str = ""
    workspace_ids: list = []
    document_ids: list | None = None
    sources: list = []
    images: list = []
    image_parts: list = []
    kg_summaries: list = []
    abbreviation_results: list = []
    final_answer: str | None = None


def route_by_intent(
    state: AgentRagState,
) -> Literal[
    "search_documents",
    "list_documents",
    "summarize_document",
    "kg_query",
    "search_doc_num",
    "search_abbr",
]:
    """Route to the appropriate tool based on intent from supervisor."""
    intent = state.intent

    intent_to_node = {
        "search": "search_documents",
        "list_docs": "list_documents",
        "summarize": "summarize_document",
        "kg_query": "kg_query",
        "search_doc_num": "search_doc_num",
        "search_abbr": "search_abbr",
    }

    return intent_to_node.get(intent, "search_documents")


TOOL_REGISTRY = {
    "search_documents": "search the knowledge base for relevant document sections",
    "list_documents": "list all documents available in the current workspace",
    "summarize_document": "get a comprehensive summary of a specific document",
    "query_knowledge_graph": "query the knowledge graph for entity relationships",
    "search_documents_number": "search for documents by their official document number",
    "search_abbreviation": "search for the meaning of an abbreviation or acronym",
}


async def search_documents_node(state: AgentRagState) -> AgentRagState:
    """Execute document search using HRAG pipeline."""
    from app.api.chat_agent import _execute_search_documents

    # Use rewritten_query preferentially, then fall back to last message
    query = state.rewritten_query
    if not query:
        messages = state.messages
        if not messages:
            state.final_answer = "No query provided."
            return state
        last_message = messages[-1]
        query = (
            last_message.content if hasattr(last_message, "content") else str(last_message)
        )

    workspace_ids = state.workspace_ids

    try:
        from app.core.deps import get_db

        async for db in get_db():
            if not workspace_ids:
                # Fallback: load all workspaces
                from sqlalchemy import select
                from app.models.workspace import Workspace
                result = await db.execute(select(Workspace))
                workspaces = result.scalars().all()
                workspace_ids = [ws.id for ws in workspaces]

            if not workspace_ids:
                state.final_answer = "No workspaces found."
                return state

            (
                context_text,
                sources,
                image_refs,
                image_parts,
                kg_summaries,
            ) = await _execute_search_documents(
                workspace_ids=workspace_ids,
                query=query,
                top_k=8,
                db=db,
                existing_ids=set(),
            )

            state.sources = [s.model_dump() for s in sources]
            state.images = [i.model_dump() for i in image_refs]
            state.image_parts = image_parts
            state.kg_summaries = kg_summaries

            if not context_text:
                state.final_answer = f"Không tìm thấy kết quả cho: {query}"
            else:
                state.final_answer = context_text
            break
    except Exception as e:
        state.final_answer = f"Lỗi tìm kiếm: {str(e)}"

    return state


async def list_documents_node(state: AgentRagState) -> AgentRagState:
    """List all documents in workspace."""
    from sqlalchemy import select
    from app.models.document import Document, DocumentStatus
    from app.models.workspace import Workspace
    from app.core.deps import get_db

    try:
        async for db in get_db():
            workspace_ids = state.workspace_ids
            if not workspace_ids:
                ws_result = await db.execute(select(Workspace))
                workspaces = ws_result.scalars().all()
                workspace_ids = [ws.id for ws in workspaces]

            if not workspace_ids:
                state.final_answer = "No workspaces found."
                return state

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
                state.final_answer = "Không có tài liệu nào đã được lập chỉ mục."
                return state

            lines = [f"Tổng cộng **{len(docs)} tài liệu** đã được lập chỉ mục:"]
            for i, doc in enumerate(docs[:20], 1):
                lines.append(f"{i}. **{doc.original_filename}** (ID: {doc.id})")

            if len(docs) > 20:
                lines.append(f"... và {len(docs) - 20} tài liệu khác")

            state.final_answer = "\n".join(lines)
            break
    except Exception as e:
        state.final_answer = f"Lỗi: {str(e)}"

    return state


async def summarize_document_node(state: AgentRagState) -> AgentRagState:
    """Summarize a specific document."""
    from app.core.deps import get_db

    messages = state.messages
    if not messages:
        state.final_answer = "No document specified."
        return state

    last_message = messages[-1]
    query = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    try:
        import re

        doc_id_match = re.search(
            r"(?:doc|document|tài liệu)[^\d]*(\d+)", query, re.IGNORECASE
        )
        if doc_id_match:
            document_id = int(doc_id_match.group(1))
        else:
            state.final_answer = (
                "Vui lòng chỉ định ID tài liệu (ví dụ: 'tóm tắt tài liệu 5')."
            )
            return state

        from sqlalchemy import select
        from app.models.document import Document, DocumentStatus
        from app.services.storage_service import get_storage_service
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMMessage

        async for db in get_db():
            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            doc = result.scalar_one_or_none()

            if not doc:
                state.final_answer = f"Không tìm thấy tài liệu với ID {document_id}."
                return state

            if doc.status != DocumentStatus.INDEXED:
                state.final_answer = (
                    f"Tài liệu '{doc.original_filename}' chưa được lập chỉ mục."
                )
                return state

            if not doc.markdown_s3_key:
                state.final_answer = (
                    f"Tài liệu '{doc.original_filename}' không có nội dung."
                )
                return state

            storage = get_storage_service()
            markdown_bytes = await storage.download_file(doc.markdown_s3_key)
            markdown_text = markdown_bytes.decode("utf-8", errors="replace")

            MAX_CHARS = 16000
            truncated = markdown_text[:MAX_CHARS]
            if len(markdown_text) > MAX_CHARS:
                truncated += "\n\n[... nội dung đã được cắt bớt ...]"

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

            state.final_answer = summary_text
            break
    except Exception as e:
        state.final_answer = f"Lỗi tóm tắt: {str(e)}"

    return state


async def kg_query_node(state: AgentRagState) -> AgentRagState:
    """Query knowledge graph."""
    from app.services.knowledge_graph_service import KnowledgeGraphService
    from app.core.deps import get_db
    from sqlalchemy import select
    from app.models.workspace import Workspace

    messages = state.messages
    if not messages:
        state.final_answer = "No entity specified."
        return state

    last_message = messages[-1]
    query = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    try:
        async for db in get_db():
            workspace_ids = state.workspace_ids
            if not workspace_ids:
                result = await db.execute(select(Workspace))
                workspaces = result.scalars().all()
                workspace_ids = [ws.id for ws in workspaces]

            results = []
            for ws_id in workspace_ids[:3]:
                try:
                    kg_service = KnowledgeGraphService(workspace_id=ws_id)
                    kg_result = await kg_service.query(query=query, mode="naive")
                    if kg_result and kg_result.strip():
                        results.append(f"**Workspace {ws_id}:**\n{kg_result}")
                except Exception:
                    pass

            if results:
                state.final_answer = "\n\n".join(results)
            else:
                state.final_answer = (
                    f"Không tìm thấy thông tin về '{query}' trong knowledge graph."
                )
            break
    except Exception as e:
        state.final_answer = f"Lỗi: {str(e)}"

    return state


async def search_doc_num_node(state: AgentRagState) -> AgentRagState:
    """Search by document number."""
    from sqlalchemy import select, or_
    from app.models.document import Document, DocumentStatus
    from app.models.workspace import Workspace
    from app.core.deps import get_db

    messages = state.messages
    if not messages:
        state.final_answer = "No document number specified."
        return state

    last_message = messages[-1]
    query = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    import re

    doc_num_match = re.search(r"\d+", query)
    search_term = doc_num_match.group(0) if doc_num_match else query

    try:
        async for db in get_db():
            workspace_ids = state.workspace_ids
            if not workspace_ids:
                result = await db.execute(select(Workspace))
                workspaces = result.scalars().all()
                workspace_ids = [ws.id for ws in workspaces]

            doc_result = await db.execute(
                select(Document)
                .where(
                    Document.workspace_id.in_(workspace_ids),
                    Document.status == DocumentStatus.INDEXED,
                    or_(
                        Document.document_number.ilike(f"%{search_term}%"),
                        Document.original_filename.ilike(f"%{search_term}%"),
                    ),
                )
                .limit(10)
            )
            docs = doc_result.scalars().all()

            if not docs:
                state.final_answer = (
                    f"Không tìm thấy tài liệu nào có số văn bản '{search_term}'."
                )
            else:
                lines = [f"Tìm thấy **{len(docs)} tài liệu**:"]
                for i, doc in enumerate(docs, 1):
                    lines.append(
                        f"{i}. **{doc.original_filename}**\n"
                        f"   Số văn bản: {doc.document_number or 'N/A'}\n"
                        f"   ID: {doc.id}"
                    )
                state.final_answer = "\n".join(lines)
            break
    except Exception as e:
        state.final_answer = f"Lỗi: {str(e)}"

    return state


async def search_abbr_node(state: AgentRagState) -> AgentRagState:
    """Search abbreviation — stores structured results in abbreviation_results."""
    from sqlalchemy import select
    from app.models.abbreviation import Abbreviation
    from app.core.deps import get_db

    messages = state.messages
    if not messages:
        state.final_answer = "No abbreviation specified."
        return state

    last_message = messages[-1]
    query = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    import re

    abbr_match = re.search(r"(?:abbr|viết tắt của)[:\s]+(\w+)", query, re.IGNORECASE)
    search_term = abbr_match.group(1) if abbr_match else query.strip()

    try:
        async for db in get_db():
            result = await db.execute(
                select(Abbreviation)
                .where(
                    Abbreviation.short_form.ilike(f"%{search_term}%"),
                    Abbreviation.is_active == True,
                )
                .limit(10)
            )
            abbreviations = result.scalars().all()

            if not abbreviations:
                state.final_answer = f"Không tìm thấy nghĩa của '{search_term}'."
                state.abbreviation_results = []
            elif len(abbreviations) == 1:
                ab = abbreviations[0]
                state.final_answer = f"**{ab.short_form}** = {ab.full_form}"
                state.abbreviation_results = [
                    {
                        "short_form": ab.short_form,
                        "full_form": ab.full_form,
                        "description": getattr(ab, "description", None),
                    }
                ]
            else:
                lines = [f"Tìm thấy **{len(abbreviations)} kết quả**:"]
                abbr_list = []
                for ab in abbreviations:
                    lines.append(f"- **{ab.short_form}** = {ab.full_form}")
                    abbr_list.append(
                        {
                            "short_form": ab.short_form,
                            "full_form": ab.full_form,
                            "description": getattr(ab, "description", None),
                        }
                    )
                state.final_answer = "\n".join(lines)
                state.abbreviation_results = abbr_list
            break
    except Exception as e:
        state.final_answer = f"Lỗi: {str(e)}"

    return state


def create_agent_rag():
    """Create and compile the RAG agent graph."""
    from langgraph.graph import START

    graph = StateGraph(AgentRagState)

    graph.add_node("search_documents", search_documents_node)
    graph.add_node("list_documents", list_documents_node)
    graph.add_node("summarize_document", summarize_document_node)
    graph.add_node("kg_query", kg_query_node)
    graph.add_node("search_doc_num", search_doc_num_node)
    graph.add_node("search_abbr", search_abbr_node)

    graph.add_conditional_edges(
        START,
        route_by_intent,
        {
            "search_documents": "search_documents",
            "list_documents": "list_documents",
            "summarize_document": "summarize_document",
            "kg_query": "kg_query",
            "search_doc_num": "search_doc_num",
            "search_abbr": "search_abbr",
        },
    )

    graph.add_edge("search_documents", END)
    graph.add_edge("list_documents", END)
    graph.add_edge("summarize_document", END)
    graph.add_edge("kg_query", END)
    graph.add_edge("search_doc_num", END)
    graph.add_edge("search_abbr", END)

    return graph.compile()

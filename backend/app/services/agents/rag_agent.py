"""
RAG Agent
=========

Single-file agent handling all document-related operations:
- search_documents       → hybrid vector+KG+BM25 search
- list_documents         → list indexed docs in workspace
- summarize_document     → fetch doc content (LLM summarize happens in answer_generator)
- kg_query               → knowledge graph entity lookup
- search_doc_num         → search by official document number
- search_abbr            → abbreviation lookup
- mongo_search_*         → MongoDB people search by CCCD/name/BHW/phone

Tool dispatch via registry pattern - adding new tools only requires
updating the registry, not routing code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.services.agents.models import SupervisorState

logger = logging.getLogger(__name__)

# =============================================================================
# Result Mappers (must be defined before registry)
# =============================================================================

def _map_search_result(result: dict) -> dict:
    """Map search_documents result to SupervisorState."""
    return {
        "sources": result.get("sources", []),
        "images": result.get("images", []),
        "kg_summaries": result.get("kg_summaries", []),
    }


def _map_list_result(result: dict) -> dict:
    """Map list_documents result to SupervisorState."""
    return {
        "kg_summaries": [result.get("text", "")],
    }


def _map_summarize_result(result: dict) -> dict:
    """Map summarize_document result to SupervisorState."""
    return {
        "kg_summaries": [result.get("text", "")],
        "text_input": result.get("text", ""),
    }


def _map_kg_result(result: dict) -> dict:
    """Map kg_query result to SupervisorState."""
    return {
        "kg_summaries": [result.get("text", "")],
    }


def _map_doc_num_result(result: dict) -> dict:
    """Map search_documents_number result to SupervisorState."""
    return {
        "kg_summaries": [result.get("text", "")],
        "sources": result.get("documents", []),
    }


def _map_abbr_result(result: dict) -> dict:
    """Map search_abbreviation result to SupervisorState."""
    results = result.get("results", [])
    # When only 1 result, search_abbreviation returns no "results" key
    # but includes full_form + description directly
    if not results and result.get("found"):
        short = result.get("abbreviation", "")
        full = result.get("full_form", "")
        desc = result.get("description", "")
        if short and full:
            results = [{"short_form": short, "full_form": full, "description": desc}]

    # If abbreviation was found and expanded, loop back to supervisor
    # to re-classify with the full form (e.g., "an ninh mạng" instead of "ANM")
    should_loop = bool(results)

    return {
        "abbreviation_results": results,
        "kg_summaries": [result.get("text", "")],
        "should_loop_back": should_loop,
        # Set expanded_query so supervisor re-classifies with full form
        "expanded_query": results[0].get("full_form", "") if results else "",
        "rewritten_query": results[0].get("full_form", "") if results else "",
    }


def _map_mongo_result(result: dict) -> dict:
    """Map MongoDB people search result to SupervisorState."""
    return {
        "mongo_results": result.get("persons", []),
        "kg_summaries": [result.get("display", "")],
    }


# =============================================================================
# Tool Functions
# =============================================================================

async def _tool_search(state: SupervisorState) -> dict:
    """Hybrid document search."""
    from app.services.agent.tools import search_documents
    from app.services.agent.streaming import get_current_db

    workspace_ids = state.get("workspace_ids", [])
    rewritten_query = state.get("rewritten_query", "")
    document_ids = state.get("document_ids")

    return await search_documents(
        query=rewritten_query,
        top_k=8,
        workspace_ids=workspace_ids,
        existing_citation_ids=set(),
        db=get_current_db(),
        document_ids=document_ids,
    )


async def _tool_list_docs(state: SupervisorState) -> dict:
    """List all indexed documents in workspace."""
    from app.services.agent.tools import list_documents
    from app.services.agent.streaming import get_current_db

    db = get_current_db()
    return await list_documents(
        workspace_ids=state.get("workspace_ids", []),
        db=db,
    )


async def _tool_summarize(state: SupervisorState) -> dict:
    """Fetch document content for summarization."""
    from app.services.agent.tools import summarize_document
    from app.services.agent.streaming import get_current_db

    document_ids = state.get("document_ids") or []
    if not document_ids:
        return {"text": "Vui lòng đính kèm file hoặc chỉ định ID tài liệu."}

    return await summarize_document(
        document_id=document_ids[0],
        db=get_current_db(),
    )


async def _tool_kg_query(state: SupervisorState) -> dict:
    """Query knowledge graph for entity relationships."""
    from app.services.agent.tools import query_knowledge_graph
    from app.services.agent.streaming import get_current_db

    rewritten_query = state.get("rewritten_query", "")
    return await query_knowledge_graph(
        entity=rewritten_query,
        workspace_ids=state.get("workspace_ids", []),
        db=get_current_db(),
    )


async def _tool_search_doc_num(state: SupervisorState) -> dict:
    """Search documents by official document number."""
    from app.services.agent.tools import search_documents_number
    from app.services.agent.streaming import get_current_db

    return await search_documents_number(
        query=state.get("rewritten_query", ""),
        workspace_ids=state.get("workspace_ids", []),
        db=get_current_db(),
    )


async def _tool_search_abbr(state: SupervisorState) -> dict:
    """Search abbreviation meaning."""
    from app.services.agent.tools import search_abbreviation
    from app.services.agent.streaming import get_current_db

    raw_query = state.get("rewritten_query", "")
    workspace_ids = state.get("workspace_ids", [])
    db = get_current_db()

    # Extract abbreviation from query patterns:
    # - "khái niệm ANM là gì" → "ANM"
    # - "'ANM'" or '"ANM"' → "ANM"
    # - "ANM nghĩa là gì" → "ANM"
    import re
    abbreviation_match = re.search(r'\b([A-Z]{2,})\b', raw_query)  # uppercase acronyms
    if not abbreviation_match:
        abbreviation_match = re.search(r'[\'"]([^\'"]+)[\'"]', raw_query)  # quoted
    abbreviation = abbreviation_match.group(1) if abbreviation_match else raw_query

    logger.info(f"[_tool_search_abbr] abbreviation={abbreviation!r}, workspace_ids={workspace_ids}, db={type(db).__name__}")

    return await search_abbreviation(
        abbreviation=abbreviation,
        workspace_ids=workspace_ids,
        db=db,
    )


# =============================================================================
# Tool Registry
# =============================================================================

RAG_TOOL_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "search": (_tool_search, _map_search_result),
    "list_docs": (_tool_list_docs, _map_list_result),
    "summarize": (_tool_summarize, _map_summarize_result),
    "kg_query": (_tool_kg_query, _map_kg_result),
    "search_doc_num": (_tool_search_doc_num, _map_doc_num_result),
    "search_abbr": (_tool_search_abbr, _map_abbr_result),
}


# =============================================================================
# RAG Agent Node
# =============================================================================

async def rag_agent_node(state: SupervisorState) -> dict:
    """
    Execute RAG operation based on intent.

    Flow:
    1. Look up tool in registry
    2. Call tool function
    3. Map result to SupervisorState
    4. Emit sources/images events for SSE streaming
    5. Return partial state update
    """
    from app.services.agents.models import AgentType
    from app.services.agent.streaming import push_event

    intent = state.get("intent", "search")
    logger.info(f"[rag_agent] intent={intent!r}")

    # Emit status
    await push_event(state, "status", {"step": "searching", "detail": "Đang tìm kiếm..."})

    if intent not in RAG_TOOL_REGISTRY:
        logger.warning(f"[rag_agent] No tool for intent {intent!r}")
        return {"next_agent": AgentType.RAG}

    tool_fn, mapper = RAG_TOOL_REGISTRY[intent]

    try:
        result = await tool_fn(state)
        updates = mapper(result)

        # Emit sources and images for SSE streaming
        sources = updates.get("sources", [])
        images = updates.get("images", [])
        if sources:
            await push_event(state, "sources", sources)
        if images:
            await push_event(state, "images", images)

        # Add iteration count
        updates["iterations"] = state.get("iterations", 0) + 1

        logger.info(
            f"[rag_agent] completed: sources={len(sources)}, "
            f"kg_summaries={len(updates.get('kg_summaries', []))}"
        )

        return updates

    except Exception as e:
        logger.error(f"[rag_agent] tool {intent} failed: {e}", exc_info=True)
        return {
            "kg_summaries": [f"Lỗi tìm kiếm: {str(e)}"],
            "iterations": state.get("iterations", 0) + 1,
        }
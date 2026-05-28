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

from langfuse import get_client

if TYPE_CHECKING:
    from app.services.agents.models import SupervisorState

logger = logging.getLogger(__name__)


# =============================================================================
# Langfuse client (lazy initialization for span creation)
# =============================================================================

def _get_langfuse_client():
    """Get or create Langfuse client for manual span instrumentation."""
    try:
        return get_client()
    except Exception as e:
        logger.warning(f"[langfuse] Failed to get client: {e}")
        return None


class _NullContext:
    """Null context manager for when Langfuse is unavailable."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, **kwargs):
        pass


def _null_context():
    return _NullContext()


async def _with_langfuse_span(name: str, input_data: dict, coro):
    """
    Execute an async coroutine within a Langfuse observation (SDK v4).

    Usage:
        result = await _with_langfuse_span(
            "search_documents",
            {"query": q, "workspace_ids": [...]},
            search_documents(...),
        )
    """
    langfuse = _get_langfuse_client()
    if not langfuse:
        return await coro

    try:
        obs = langfuse.start_observation(
            name=name,
            input=input_data,
            level="DEFAULT",
        )
        result = await coro
        obs.update(output={"result": result})
        obs.end()
        return result
    except Exception as e:
        logger.warning(f"[langfuse] Span failed for {name}: {e}")
        return await coro

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

    # Use the combined expanded_query from result if available (for multi-abbr case),
    # otherwise fall back to first result's full_form
    expanded_query = result.get("expanded_query", "")
    if not expanded_query and results:
        expanded_query = results[0].get("full_form", "")

    return {
        "abbreviation_results": results,
        "kg_summaries": [result.get("text", "")],
        "should_loop_back": should_loop,
        "expanded_query": expanded_query,
        "rewritten_query": expanded_query,
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
    """Hybrid document search — search_mode is set by supervisor based on intent.

    Phase 3 (UUID-Scoped Search): When document_ids is already known
    (from resolve_doc, file upload, or prior summarize), sets scoped_to_documents=True
    to restrict search exclusively to those documents rather than searching the
    entire workspace. This improves precision and avoids diluted results.
    """
    from app.services.agent.tools import search_documents
    from app.services.agent.streaming import get_current_db

    workspace_ids = state.get("workspace_ids", [])
    rewritten_query = state.get("rewritten_query", "")
    document_ids = state.get("document_ids")
    # Phase 1: use supervisor-determined search_mode (vector | kg | hybrid)
    search_mode = state.get("search_mode", "hybrid")

    # Phase 3: UUID-scoped search — when document_ids are already known,
    # restrict search to only those documents (skip wide workspace search)
    scoped = bool(document_ids)
    if scoped:
        logger.info(
            f"[_tool_search] UUID-scoped: restricting to {len(document_ids)} document(s): "
            f"{[str(d) for d in document_ids[:3]]}"
        )

    existing_ids = state.get("existing_citation_ids", {})
    if _get_langfuse_client():
        return await _with_langfuse_span(
            "search_documents",
            {
                "query": rewritten_query,
                "workspace_ids": [str(ws) for ws in workspace_ids],
                "document_ids": [str(d) for d in document_ids] if document_ids else None,
                "search_mode": search_mode,
            },
            _search_impl(workspace_ids, rewritten_query, document_ids, search_mode, scoped, existing_ids),
        )
    else:
        return await _search_impl(workspace_ids, rewritten_query, document_ids, search_mode, scoped, existing_ids)


async def _search_impl(workspace_ids, rewritten_query, document_ids, search_mode, scoped, existing_ids) -> dict:
    """Implementation of search_documents tool (no Langfuse)."""
    from app.services.agent.tools import search_documents
    from app.services.agent.streaming import get_current_db

    # Convert dict keys to set for search_documents
    ids_set = set(existing_ids.keys()) if isinstance(existing_ids, dict) else set(existing_ids)

    if scoped:
        logger.info(
            f"[_tool_search] UUID-scoped: restricting to {len(document_ids)} document(s): "
            f"{[str(d) for d in document_ids[:3]]}"
        )
    return await search_documents(
        query=rewritten_query,
        top_k=8,
        workspace_ids=workspace_ids,
        existing_citation_ids=ids_set,
        db=get_current_db(),
        document_ids=document_ids,
        search_mode=search_mode,
        scoped_to_documents=scoped,
    )

async def _tool_list_docs(state: SupervisorState) -> dict:
    """List all indexed documents in workspace."""
    langfuse = _get_langfuse_client()
    if langfuse:
        return await _with_langfuse_span(
            "list_documents",
            {"workspace_ids": [str(ws) for ws in state.get("workspace_ids", [])]},
            _list_docs_impl(state),
        )
    else:
        return await _list_docs_impl(state)


async def _list_docs_impl(state: SupervisorState) -> dict:
    """Implementation of list_documents tool (no Langfuse)."""
    from app.services.agent.tools import list_documents
    from app.services.agent.streaming import get_current_db

    return await list_documents(
        workspace_ids=state.get("workspace_ids", []),
        db=get_current_db(),
    )


async def _tool_summarize(state: SupervisorState) -> dict:
    """Fetch document content for summarization."""
    from app.services.agent.tools import summarize_document
    from app.services.agent.streaming import get_current_db

    document_ids = state.get("document_ids") or []
    if not document_ids:
        return {"text": "Vui lòng đính kèm file hoặc chỉ định ID tài liệu."}

    langfuse = _get_langfuse_client()
    if langfuse:
        return await _with_langfuse_span(
            "summarize_document",
            {"document_ids": [str(d) for d in document_ids]},
            _summarize_impl(document_ids),
        )
    else:
        return await _summarize_impl(document_ids)


async def _summarize_impl(document_ids: list) -> dict:
    """Implementation of summarize_document tool (no Langfuse)."""
    from app.services.agent.tools import summarize_document
    from app.services.agent.streaming import get_current_db

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
    langfuse = _get_langfuse_client()
    if langfuse:
        return await _with_langfuse_span(
            "query_knowledge_graph",
            {"entity": rewritten_query, "workspace_ids": [str(ws) for ws in state.get("workspace_ids", [])]},
            _kg_impl(state),
        )
    else:
        return await _kg_impl(state)


async def _kg_impl(state: SupervisorState) -> dict:
    """Implementation of query_knowledge_graph tool (no Langfuse)."""
    from app.services.agent.tools import query_knowledge_graph
    from app.services.agent.streaming import get_current_db

    return await query_knowledge_graph(
        entity=state.get("rewritten_query", ""),
        workspace_ids=state.get("workspace_ids", []),
        db=get_current_db(),
    )


async def _tool_search_doc_num(state: SupervisorState) -> dict:
    """Search documents by official document number."""
    from app.services.agent.tools import search_documents_number
    from app.services.agent.streaming import get_current_db

    langfuse = _get_langfuse_client()
    if langfuse:
        return await _with_langfuse_span(
            "search_documents_number",
            {"query": state.get("rewritten_query", ""), "workspace_ids": [str(ws) for ws in state.get("workspace_ids", [])]},
            _search_doc_num_impl(state),
        )
    else:
        return await _search_doc_num_impl(state)


async def _search_doc_num_impl(state: SupervisorState) -> dict:
    """Implementation of search_documents_number tool (no Langfuse)."""
    from app.services.agent.tools import search_documents_number
    from app.services.agent.streaming import get_current_db

    return await search_documents_number(
        query=state.get("rewritten_query", ""),
        workspace_ids=state.get("workspace_ids", []),
        db=get_current_db(),
    )


async def _tool_search_abbr(state: SupervisorState) -> dict:
    """Search abbreviation meaning - handles multiple abbreviations in one query."""
    from sqlalchemy import select
    from app.services.agent.streaming import get_current_db
    from app.models.abbreviation import Abbreviation

    raw_query = state.get("rewritten_query", "")
    db = get_current_db()

    langfuse = _get_langfuse_client()
    if langfuse:
        return await _with_langfuse_span(
            "search_abbreviation",
            {"query": raw_query},
            _execute_search_abbr(raw_query, db),
        )
    else:
        return await _execute_search_abbr(raw_query, db)


async def _execute_search_abbr(raw_query: str, db) -> dict:
    """Execute abbreviation search logic (extracted for reuse)."""
    import re
    from sqlalchemy import select
    from app.models.abbreviation import Abbreviation

    # Find all uppercase sequences (2+ chars) that are abbreviations
    all_abbr_matches = re.findall(r'\b([A-Z]{2,})\b', raw_query)
    # Also check for quoted abbreviations
    quoted_matches = re.findall(r'[\'"]([^\'"]+)[\'"]', raw_query)
    all_abbreviations = all_abbr_matches + [m for m in quoted_matches if len(m) >= 2]

    logger.info(f"[_tool_search_abbr] found abbreviations={all_abbreviations!r}, raw_query={raw_query!r}")

    # If no abbreviations found, search with the original query
    if not all_abbreviations:
        abbreviation = raw_query
        logger.info(f"[_tool_search_abbr] no abbr found, searching raw_query={abbreviation!r}")
        result = await db.execute(
            select(Abbreviation)
            .where(
                Abbreviation.short_form.ilike(f"%{abbreviation}%"),
                Abbreviation.is_active == True,
            )
            .limit(10)
        )
        abbreviations = result.scalars().all()
        return _build_abbr_response(abbreviations, abbreviation)

    # Search for ALL abbreviations found
    if len(all_abbreviations) == 1:
        # Single abbreviation - use existing logic for backward compatibility
        abbreviation = all_abbreviations[0]
        result = await db.execute(
            select(Abbreviation)
            .where(
                Abbreviation.short_form.ilike(f"%{abbreviation}%"),
                Abbreviation.is_active == True,
            )
            .limit(10)
        )
        abbreviations = result.scalars().all()
        return _build_abbr_response(abbreviations, abbreviation)

    # Multiple abbreviations - search for each and combine
    all_results = []
    expanded_parts = []
    not_found = []

    for abbreviation in all_abbreviations:
        result = await db.execute(
            select(Abbreviation)
            .where(
                Abbreviation.short_form.ilike(f"%{abbreviation}%"),
                Abbreviation.is_active == True,
            )
            .limit(10)
        )
        abbreviations = result.scalars().all()
        if abbreviations:
            for ab in abbreviations:
                all_results.append({
                    "short_form": ab.short_form,
                    "full_form": ab.full_form,
                    "description": ab.description,
                })
                if ab.full_form not in expanded_parts:
                    expanded_parts.append(ab.full_form)
        else:
            not_found.append(abbreviation)

    # Build expanded query with ALL expanded forms
    expanded_query = " ".join(expanded_parts) if expanded_parts else ""

    # Build text response
    lines = [f"Tìm thấy **{len(all_results)} kết quả** cho các từ viết tắt:"]
    for i, ab in enumerate(all_results, 1):
        lines.append(f"{i}. **{ab['short_form']}** = {ab['full_form']}")
        if ab.get('description'):
            lines.append(f"   Mô tả: {ab['description']}")

    if not_found:
        lines.append(f"\nKhông tìm thấy: {', '.join(not_found)}")

    text = "\n".join(lines)
    should_loop = bool(all_results)

    logger.info(f"[_tool_search_abbr] multi-abbr: found={len(all_results)}, expanded_query={expanded_query!r}")

    return {
        "text": text,
        "results": all_results,
        "found": should_loop,
        "expanded_query": expanded_query,
        "abbreviation": "|".join(all_abbreviations),
    }


def _build_abbr_response(abbreviations, original_abbr):
    """Helper to build abbreviation response."""
    if not abbreviations:
        return {
            "text": f"Không tìm thấy nghĩa của '{original_abbr}'. "
                    f"Bạn có thể cho biết '{original_abbr}' là viết tắt của gì không?",
            "abbreviation": original_abbr,
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

    lines = [f"Tìm thấy **{len(abbreviations)} kết quả** cho '{original_abbr}':"]
    for i, ab in enumerate(abbreviations, 1):
        lines.append(f"{i}. **{ab.short_form}** = {ab.full_form}")

    return {
        "text": "\n".join(lines),
        "abbreviation": original_abbr,
        "found": True,
        "results": [
            {"short_form": ab.short_form, "full_form": ab.full_form, "description": ab.description}
            for ab in abbreviations
        ],
    }


# =============================================================================
# Tool: resolve_doc
# =============================================================================


async def _tool_resolve_doc(state) -> dict:
    """Resolve document reference (e.g., 'Luật An ninh mạng 2025') to UUID."""
    from app.services.agent.tools import resolve_document_reference
    from app.services.agent.streaming import get_current_db

    reference = state.get("rewritten_query", "") or state.get("raw_query", "")
    logger.info(f"[_tool_resolve_doc] reference={reference!r}, workspace_ids={state.get('workspace_ids', [])}")

    if not reference:
        return {
            "candidates": [],
            "total": 0,
            "ambiguous": False,
            "message": "Không có thông tin tìm kiếm văn bản.",
        }

    langfuse = _get_langfuse_client()
    if langfuse:
        return await _with_langfuse_span(
            "resolve_document_reference",
            {"reference": reference, "workspace_ids": [str(ws) for ws in state.get("workspace_ids", [])]},
            _resolve_doc_impl(state, reference),
        )
    else:
        return await _resolve_doc_impl(state, reference)


async def _resolve_doc_impl(state, reference: str) -> dict:
    """Implementation of resolve_document_reference tool (no Langfuse)."""
    from app.services.agent.tools import resolve_document_reference
    from app.services.agent.streaming import get_current_db

    result = await resolve_document_reference(
        reference=reference,
        workspace_ids=state.get("workspace_ids", []),
        db=get_current_db(),
    )
    logger.info(f"[_tool_resolve_doc] result: {result.get('total')} candidates, message={result.get('message', '')[:100]}")
    return result


def _map_resolve_doc_result(result: dict) -> dict:
    """Map resolve_document_reference result to SupervisorState."""
    candidates = result.get("candidates", [])
    resolved_ids = [c["document_id"] for c in candidates]
    section_reference = result.get("section_reference")
    logger.info(f"[_map_resolve_doc_result] resolved_ids={resolved_ids}, section_reference={section_reference}, ambiguous={result.get('ambiguous')}")

    if resolved_ids:
        # Document(s) found — route to search_section (or summarize if no section_ref)
        intent_to_use = "search_section" if section_reference else "summarize"

        logger.info(f"[_map_resolve_doc_result] → intent={intent_to_use}, section_reference={section_reference}, document_ids={resolved_ids}")

        return {
            "kg_summaries": [result.get("message", "")],
            "document_ids": resolved_ids,
            "resolve_doc_ambiguous": result.get("ambiguous", False),
            "should_loop_back": False,
            "intent": intent_to_use,
            "section_reference": section_reference,
            # Flag to tell route_from_rag that search_section tool hasn't run yet
            "_pending_search_section": True,
        }
    else:
        # No match — go to answer_generator with "not found" message
        return {
            "kg_summaries": [result.get("message", "")],
            "resolved_document_ids": [],
            "document_ids": [],
            "resolve_doc_ambiguous": False,
            "should_loop_back": False,
        }

# =============================================================================
# Helper: extract section content from raw markdown
# =============================================================================


def _extract_section_from_markdown(markdown_text: str, section_ref: str) -> str:
    """
    Extract a specific section (Điều X, Chương Y, etc.) from raw markdown.

    Uses heuristic markers:
    - For "Điều X": finds "## Điều X" or "## Điều X." and extracts until next heading
    - For "Chương X": finds "## Chương X" and extracts until next Chương
    """
    import re

    if not markdown_text or not section_ref:
        return ""

    section_ref = section_ref.strip()

    # Normalize section reference for matching
    # "Điều 5" → try to match "## Điều 5" (with optional period after number)
    section_pattern = section_ref.replace(".", r"\.?")

    # Try multiple patterns to find the section header
    patterns = [
        rf"(?m)^##\s*{re.escape(section_pattern)}[.\s]",  # ## Điều 5
        rf"(?m)^##\s*{re.escape(section_pattern)}",       # ## Điều 5 (no period)
        rf"(?m)^#\s*{re.escape(section_pattern)}[.\s]",   # # Điều 5
        rf"(?m)^###\s*{re.escape(section_pattern)}[.\s]", # ### Điều 5
    ]

    start_pos = -1
    for pattern in patterns:
        match = re.search(pattern, markdown_text, re.IGNORECASE)
        if match:
            start_pos = match.start()
            break

    if start_pos == -1:
        # Try looser match: look for section_ref followed by newline
        loose_match = re.search(
            rf"(?m)(?:^|\n)\s*{re.escape(section_ref)}[.\s]*\n",
            markdown_text,
            re.IGNORECASE
        )
        if loose_match:
            start_pos = loose_match.end()

    if start_pos == -1:
        return ""

    # Find the end of this section
    # We want to continue until the next heading of the SAME or HIGHER level.
    # If section_ref is "Điều 3", we should stop at "Điều 4", "Chương II", etc.
    # A simple but effective heuristic: look for headings that start with 
    # common Vietnamese legal structural words.
    
    # Extract only the type part from section_ref (e.g., "Điều" from "Điều 3")
    type_match = re.match(r"^([^\d\s]+)", section_ref)
    section_type = type_match.group(1) if type_match else ""
    
    # Look for next heading starting with ##
    # We'll search one by one to find a "real" boundary
    remaining_text = markdown_text[start_pos + 1:]
    heading_matches = list(re.finditer(r"(?m)^#+\s+(.+)$", remaining_text))
    
    end_pos = len(markdown_text)
    
    for h_match in heading_matches:
        h_text = h_match.group(1).strip()
        # If the next heading is a "big" structural unit or the next of the same type
        # (e.g., next "Điều" if we are in "Điều X", or any "Chương", "Phần", "Mục")
        is_boundary = False
        
        # Stop at higher level units
        if any(h_text.startswith(kw) for kw in ["Chương", "Phần", "Mục", "LỜI NÓI ĐẦU"]):
            is_boundary = True
        # Stop at next unit of same type (if type was extracted)
        elif section_type and h_text.startswith(section_type):
            is_boundary = True
        # If no type, stop at any ## heading that doesn't look like a sub-item (number only)
        elif not section_type and not re.match(r"^\d+\.", h_text):
            is_boundary = True
            
        if is_boundary:
            end_pos = start_pos + 1 + h_match.start()
            break

    section_text = markdown_text[start_pos:end_pos].strip()

    # Limit to a reasonable size (32k chars — enough for a long article)
    MAX_CHARS = 32000
    if len(section_text) > MAX_CHARS:
        section_text = section_text[:MAX_CHARS] + (
            f"\n\n[... nội dung đã cắt bớt (quá dài) ...]"
        )

    return section_text


# =============================================================================
# Tool: search_section
# =============================================================================

async def _tool_search_section(state) -> dict:
    """Search specific section/chapter in a document.

    Uses heading_path metadata search first, falls back to content-based
    section extraction from the raw markdown (more reliable for docs without
    proper heading_path metadata).
    """
    from app.services.agent.tools import search_document_section
    from app.services.agent.streaming import get_current_db
    from sqlalchemy import select
    from app.models.document import Document, DocumentStatus
    from app.services.storage_service import get_storage_service

    section_reference = state.get("section_reference", "")
    if not section_reference:
        return {"text": "Không tìm thấy thông tin chương/điều cần tra cứu.", "sources": []}

    workspace_ids = state.get("workspace_ids", [])
    document_ids = state.get("document_ids")

    langfuse = _get_langfuse_client()
    if langfuse:
        return await _with_langfuse_span(
            "search_document_section",
            {"section_reference": section_reference, "workspace_ids": [str(ws) for ws in workspace_ids], "document_ids": [str(d) for d in document_ids] if document_ids else None},
            _execute_search_section(section_reference, workspace_ids, document_ids),
        )
    else:
        return await _execute_search_section(section_reference, workspace_ids, document_ids)


async def _execute_search_section(section_reference: str, workspace_ids, document_ids) -> dict:
    """Execute section search logic (extracted for reuse)."""
    from app.services.agent.tools import search_document_section
    from app.services.agent.streaming import get_current_db
    from sqlalchemy import select
    from app.models.document import Document
    from app.services.storage_service import get_storage_service

    # Try heading_path metadata search first
    result = await search_document_section(
        section_reference=section_reference,
        workspace_ids=workspace_ids,
        document_ids=document_ids,
    )

    if result.get("sources"):
        logger.info(
            f"[search_section] heading_path search found {len(result['sources'])} sources "
            f"for '{section_reference}'"
        )
        return result

    # heading_path search returned 0 results — fall back to content-based extraction
    logger.info(
        f"[search_section] heading_path search returned 0 results for "
        f"'{section_reference}', falling back to content extraction"
    )

    if not document_ids:
        return {
            "text": f"Không tìm thấy tài liệu nào phù hợp với '{section_reference}'.",
            "sources": []
        }

    try:
        db = get_current_db()
        doc_result = await db.execute(
            select(Document).where(Document.id == document_ids[0])
        )
        doc = doc_result.scalar_one_or_none()

        if not doc or not doc.markdown_s3_key:
            return {
                "text": f"Không tìm thấy nội dung cho '{section_reference}'.",
                "sources": []
            }

        storage = get_storage_service()
        markdown_text = await storage.download_markdown(doc.markdown_s3_key)

        # Extract section from content using markers like "Điều X", "Chương Y", etc.
        extracted_text = _extract_section_from_markdown(markdown_text, section_reference)

        if extracted_text:
            # Generate a short index ID for citation (e.g., "ss1")
            import uuid as _uuid
            short_id = str(_uuid.uuid4().hex[:4])
            from app.schemas.rag import ChatSourceChunk
            
            chunk_obj = ChatSourceChunk(
                index=short_id,
                chunk_id=f"doc_{doc.id}_section_{section_reference.replace(' ', '_')}",
                content=extracted_text,  # Full text for LLM
                document_id=doc.id,
                page_no=0,
                heading_path=[section_reference],
                source_type="extraction",
                source_file=doc.original_filename
            )
            
            result = {
                "text": extracted_text,
                "sources": [chunk_obj],
                "content_extracted": True,
            }
            logger.info(
                f"[search_section] content extraction got {len(extracted_text):,} chars "
                f"for '{section_reference}'"
            )
            return result
        else:
            # No section found in content — return a helpful message
            return {
                "text": f"Không tìm thấy nội dung điều/khoản '{section_reference}' trong tài liệu.",
                "sources": []
            }

    except Exception as e:
        logger.warning(
            f"[search_section] content extraction failed: {e}"
        )
        return {
            "text": f"Không thể truy xuất nội dung cho '{section_reference}'.",
            "sources": []
        }

def _map_search_section(result: dict) -> dict:
    """Map search_section result to SupervisorState."""
    return {
        "kg_summaries": [result.get("text", "")],
        "sources": result.get("sources", []),
        # Change intent so route_from_rag goes to answer_generator (not back to supervisor)
        "intent": "summarize",
        # Clear section_reference so route_from_rag knows tool has executed
        "section_reference": "",
    }

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
    # resolve_doc removed — handled by dedicated resolve_doc_agent (Phase 2)
    "search_section": (_tool_search_section, _map_search_section),
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
    logger.info(f"[LANGGRAPH_NODE] Entering rag_agent_node, intent={intent!r}, rewritten_query={state.get('rewritten_query', '')[:100]!r}, document_ids={state.get('document_ids')}")

    # Emit status
    status_map = {
        "search": "Đang tìm kiếm tài liệu...",
        "list_docs": "Đang lấy danh sách...",
        "summarize": "Đang tóm tắt...",
        "kg_query": "Đang truy vấn đồ thị...",
        "search_doc_num": "Đang tra cứu số văn bản...",
        "search_abbr": "Đang tra cứu viết tắt...",
        # resolve_doc removed — handled by resolve_doc_agent (Phase 2)
        "search_section": "Đang trích xuất nội dung phần/chương/điều...",
    }

    await push_event(state, "status", {"step": "searching", "detail": status_map.get(intent, "Đang xử lý...")})

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
            f"[LANGGRAPH_DECISION] rag_agent_node completed: sources={len(sources)}, "
            f"kg_summaries={len(updates.get('kg_summaries', []))}"
        )

        return updates

    except Exception as e:
        logger.error(f"[rag_agent] tool {intent} failed: {e}", exc_info=True)
        return {
            "kg_summaries": [f"Lỗi tìm kiếm: {str(e)}"],
            "iterations": state.get("iterations", 0) + 1,
        }
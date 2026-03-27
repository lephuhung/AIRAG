"""
Agent Nodes
===========

LangGraph node functions for the NexusRAG chat agent.

Each node receives the full AgentState and returns a partial state update dict.
Nodes are pure async functions — LangGraph merges the returned dict into state.

Nodes:
    memory_recall       — load user memories from pgvector
    intent_classifier   — Qwen3-4B: classify intent + rewrite query
    agent_rag_executor  — invoke agent_rag subgraph (search/list/summarize/kg/abbr)
    answer_generator    — main LLM: generate answer with sources in context
    direct_answer       — main LLM: answer greetings/chitchat directly
    write_executor      — invoke agent_write subgraph (summarize/edit/grammar)

SSE streaming:
    All nodes call push_event(state, ev_type, ev_data) to push events into the
    shared asyncio.Queue injected by stream_agent_to_sse. answer_generator and
    direct_answer use provider.astream() to push tokens one-by-one in real-time.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agent.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent classifier system prompt — for Qwen3-4B
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """\
You are an intent classification assistant for a Vietnamese document Q&A system.
Your job is to classify the user's message and rewrite their query for better retrieval.

Respond ONLY with valid JSON. No explanation, no markdown, no extra text.

Intent categories:
- "greeting"  : greetings, thanks, farewells, simple chitchat → no document search needed
- "personal"  : questions about the user themselves (where they work, their name, their role, their preferences, anything about "tôi/I/me/my") → answer from personal memory, no document search
- "search"    : questions about document content, data, facts, analysis → needs search
- "list_docs" : user wants to know what documents/files are available
- "summarize" : user wants a summary of a specific document
- "kg_query"  : user asks about entity relationships, organizational charts, knowledge graph
- "search_doc_num" : user asks about document numbers (văn bản số), reference numbers, official document IDs
- "search_abbr" : user asks about SHORT abbreviations, acronyms, or their meanings (e.g., "BMNN", "TTGT"). Do NOT use for general concepts or multiple words (e.g., "an ninh mạng" is "search").
- "write_summarize"     : user provides a TEXT PASSAGE and wants it summarized or key points extracted
- "write_suggest_edits" : user provides a TEXT PASSAGE and wants editing/improvement suggestions
- "write_grammar_check" : user provides a TEXT PASSAGE and wants grammar/style checking

Output format:
{"intent": "<category>", "rewritten_query": "<improved Vietnamese/English search query>", "needs_tool": true|false, "write_action": "<action or empty>", "text_input": "<extracted text or empty>"}

Rules:
- For "greeting": set rewritten_query to "" and needs_tool to false
- For "personal": set rewritten_query to the user's question verbatim and needs_tool to false
- For "search_abbr" (abbreviation queries): ONLY if the target is a short abbreviation (usually uppercase, 2-6 chars). Otherwise default to "search".
- For write intents: extract the text to process into "text_input", set write_action to the specific action, set needs_tool to false
- For "write_summarize": write_action = "summarize" (or "extract_key_points" if user asks for key points)
- For "write_suggest_edits": write_action = "suggest_edits"
- For "write_grammar_check": write_action = "grammar_check"
- For all other intents: rewrite the query to be specific and detailed for retrieval
- If the message contains a document ID, preserve it in the output
- Default to "search" when uncertain

Examples:
User: "xin chào"  → {"intent": "greeting", "rewritten_query": "", "needs_tool": false, "write_action": "", "text_input": ""}
User: "tôi đang công tác ở đâu?" → {"intent": "personal", "rewritten_query": "tôi đang công tác ở đâu?", "needs_tool": false, "write_action": "", "text_input": ""}
User: "doanh thu 2024 là bao nhiêu?" → {"intent": "search", "rewritten_query": "doanh thu thuần tổng doanh thu năm 2024 theo quý", "needs_tool": true, "write_action": "", "text_input": ""}
User: "an ninh mạng là gì?" → {"intent": "search", "rewritten_query": "định nghĩa an ninh mạng khái niệm", "needs_tool": true, "write_action": "", "text_input": ""}
User: "có tài liệu gì trong hệ thống?" → {"intent": "list_docs", "rewritten_query": "danh sách tài liệu", "needs_tool": true, "write_action": "", "text_input": ""}
User: "tóm tắt tài liệu ID 5" → {"intent": "summarize", "rewritten_query": "tóm tắt tài liệu 5", "needs_tool": true, "write_action": "", "text_input": ""}
User: "BMNN là gì?" → {"intent": "search_abbr", "rewritten_query": "BMNN", "needs_tool": true, "write_action": "", "text_input": ""}
User: "tóm tắt đoạn văn sau: [đoạn văn dài]" → {"intent": "write_summarize", "rewritten_query": "", "needs_tool": false, "write_action": "summarize", "text_input": "[đoạn văn dài]"}
User: "kiểm tra ngữ pháp: Hôm nay tôi đi học." → {"intent": "write_grammar_check", "rewritten_query": "", "needs_tool": false, "write_action": "grammar_check", "text_input": "Hôm nay tôi đi học."}
User: "đề xuất chỉnh sửa văn bản này: [nội dung]" → {"intent": "write_suggest_edits", "rewritten_query": "", "needs_tool": false, "write_action": "suggest_edits", "text_input": "[nội dung]"}
"""

_VALID_INTENTS = {
    "greeting",
    "personal",
    "search",
    "list_docs",
    "summarize",
    "kg_query",
    "search_doc_num",
    "search_abbr",
    "write_summarize",
    "write_suggest_edits",
    "write_grammar_check",
}


def _parse_classifier_output(raw: str) -> dict:
    """Parse Qwen3-4B classifier JSON output with safe fallback."""
    raw = raw.strip()

    # Strip markdown code fences if present
    if "```json" in raw:
        raw = raw.split("```json")[-1].split("```")[0].strip()
    elif "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1].strip()

    try:
        data = json.loads(raw)
        intent = data.get("intent", "search")
        if intent not in _VALID_INTENTS:
            logger.warning(
                f"[classifier] Unknown intent '{intent}', defaulting to 'search'"
            )
            intent = "search"
        return {
            "intent": intent,
            "rewritten_query": data.get("rewritten_query", ""),
            "needs_tool": data.get("needs_tool", True),
            "write_action": data.get("write_action", ""),
            "text_input": data.get("text_input", ""),
        }
    except json.JSONDecodeError:
        logger.warning(
            f"[classifier] Failed to parse JSON: {raw[:100]!r}, defaulting to search"
        )
        return {"intent": "search", "rewritten_query": "", "needs_tool": True, "write_action": "", "text_input": ""}


# ---------------------------------------------------------------------------
# Node: memory_recall
# ---------------------------------------------------------------------------


def _get_msg_role(msg) -> str:
    """Extract role from a LangChain message or plain dict."""
    if isinstance(msg, dict):
        return msg.get("role", "")
    # LangChain messages use .type ("human"/"ai"/"system"), not .role
    msg_type = getattr(msg, "type", None)
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"
    # Fallback: some messages may have .role
    return getattr(msg, "role", "") or ""


def _extract_last_user_message(state: "AgentState") -> str:
    """Extract the most recent user message text from state messages."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if _get_msg_role(msg) == "user":
            content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else ""
            )
            return content or ""
    return ""


async def memory_recall(state: "AgentState") -> dict:
    """
    Load relevant user memories from Graphiti (temporal knowledge graph) and
    inject into state as a formatted string.

    Graphiti stores conversation episodes in Neo4j and extracts temporal facts
    (entities + relationships) from them. Search is hybrid: semantic + BM25 +
    graph traversal — significantly richer than flat pgvector similarity search.

    Non-blocking — if Graphiti is unavailable, the pipeline continues normally
    with an empty memory context.
    """
    from app.services.agent.streaming import push_event

    await push_event(
        state,
        "status",
        {"step": "analyzing", "detail": "Đang tải bộ nhớ người dùng..."},
    )

    user_id = state.get("user_id")
    if not user_id:
        return {"user_memory_context": ""}

    # ── Input: use expanded_query from abbr_expander if available, else extract original ──
    expanded_query = state.get("rewritten_query", "")
    user_message = ""

    if expanded_query:
        user_message = expanded_query
        logger.debug(f"[memory_recall] Using expanded query: {user_message!r}")
    else:
        user_message = _extract_last_user_message(state)

    if not user_message:
        return {"user_memory_context": ""}

    try:
        from app.services.graphiti_client import search_user_memory

        memory_context = await search_user_memory(user_id, user_message, top_k=5)
        if memory_context:
            logger.info(
                f"[memory_recall] Graphiti injected {len(memory_context)} chars for user {user_id}"
            )
            return {"user_memory_context": memory_context}
    except Exception as e:
        logger.warning(f"[memory_recall] Graphiti search failed: {e}")

    return {"user_memory_context": ""}


# ---------------------------------------------------------------------------
# Node: intent_classifier
# ---------------------------------------------------------------------------


async def intent_classifier(state: "AgentState") -> dict:
    """
    Use Qwen3-4B (via memory agent endpoint) to classify intent and rewrite query.

    Extracts the last user message, calls the classifier, and returns:
        intent, rewritten_query updates to state.
    """
    from app.services.agent.streaming import push_event

    await push_event(
        state, "status", {"step": "analyzing", "detail": "Đang phân tích câu hỏi..."}
    )

    # ── Input: use expanded_query from abbr_expander if available, else extract original ──
    expanded_query = state.get("rewritten_query", "")
    user_message = ""

    if expanded_query:
        user_message = expanded_query
        logger.debug(f"[intent_classifier] Using expanded query: {user_message!r}")
    else:
        messages = state.get("messages", [])
        if messages and isinstance(messages, list):
            for msg in reversed(messages):
                if _get_msg_role(msg) == "user":
                    content = getattr(msg, "content", None) or (
                        msg.get("content") if isinstance(msg, dict) else ""
                    )
                    user_message = content or ""
                    break

    if not user_message:
        return {"intent": "search", "rewritten_query": ""}

    try:
        from app.services.llm import get_memory_agent
        from app.services.llm.types import LLMMessage as _LLMMsg

        classifier = get_memory_agent()
        response_text = ""

        async for chunk in classifier.astream(
            [_LLMMsg(role="user", content=user_message)],
            system_prompt=_CLASSIFIER_SYSTEM,
            temperature=0.0,
            max_tokens=128,
        ):
            if hasattr(chunk, "text") and chunk.text:
                response_text += str(chunk.text)

        result = _parse_classifier_output(response_text)

        # Detailed logging for debugging intent classification
        logger.info(f"[intent_classifier] RAW_LLM_RESPONSE: {response_text!r}")
        logger.info(
            f"[intent_classifier] PARSED_RESULT: intent={result['intent']!r} "
            f"rewritten={result['rewritten_query']!r} "
            f"needs_tool={result.get('needs_tool', True)!r}"
        )

        # Emit a meaningful status based on intent
        intent_labels = {
            "greeting": "Tin nhắn thông thường",
            "search": "Tìm kiếm tài liệu",
            "list_docs": "Liệt kê tài liệu",
            "summarize": "Tóm tắt tài liệu",
            "kg_query": "Truy vấn đồ thị tri thức",
            "search_abbr": "Tra cứu viết tắt",
            "search_doc_num": "Tra cứu số văn bản",
            "write_summarize": "Tóm tắt văn bản",
            "write_suggest_edits": "Đề xuất chỉnh sửa",
            "write_grammar_check": "Kiểm tra ngữ pháp",
        }
        intent_label = intent_labels.get(result["intent"], "Tìm kiếm")
        await push_event(
            state,
            "status",
            {
                "step": "searching",
                "detail": f"Phân loại: {intent_label}",
            },
        )

        return {
            "intent": result["intent"],
            "rewritten_query": result["rewritten_query"] or user_message,
            "write_action": result.get("write_action", ""),
            "text_input": result.get("text_input", ""),
        }

    except Exception as e:
        logger.error(
            f"[intent_classifier] Classifier failed: {e} — defaulting to 'search'"
        )
        return {"intent": "search", "rewritten_query": user_message}


# ---------------------------------------------------------------------------
# Node: abbr_expander  (global — runs after intent_classifier, before routing)
# ---------------------------------------------------------------------------


async def abbr_expander(state: "AgentState") -> dict:
    """
    Initial abbreviation check — runs START → memory_recall → intent_classifier.
    Identifies candidates for expansion and records potential ones not in DB.
    """
    from app.services.agent.streaming import push_event

    user_message = _extract_last_user_message(state)
    if not user_message:
        return {}

    # Tìm tất cả token viết hoa liên tiếp có khả năng là viết tắt (2+ ký tự)
    import re
    abbr_candidates = re.findall(r"\b[A-ZĐẮẰẶẤẦẨẪẬẮẶẪẨẦ]{2,}\b", user_message)
    if not abbr_candidates:
        return {}

    try:
        from app.services.agent.streaming import get_current_db
        from sqlalchemy import select
        from app.models.abbreviation import Abbreviation

        db = get_current_db()
        if db is None:
            return {}

        expanded_message = user_message
        found_any = False
        potential_abbreviations = []

        for candidate in abbr_candidates:
            result = await db.execute(
                select(Abbreviation)
                .where(
                    Abbreviation.short_form.ilike(candidate),
                    Abbreviation.is_active == True,
                )
                .limit(1)
            )
            abbr = result.scalar_one_or_none()
            if abbr and abbr.full_form:
                expanded_message = expanded_message.replace(candidate, abbr.full_form)
                found_any = True
                logger.info(
                    f"[abbr_expander] Expanded {candidate!r} → {abbr.full_form!r}"
                )
            else:
                # Không thấy trong DB → record as potential
                potential_abbreviations.append(candidate)
                logger.debug(f"[abbr_expander] Potential missing abbr: {candidate!r}")

        updates = {}
        if found_any:
            await push_event(
                state,
                "status",
                {"step": "searching", "detail": f"Mở rộng viết tắt: {expanded_message}"},
            )
            updates["rewritten_query"] = expanded_message
        
        if potential_abbreviations:
            updates["potential_abbreviations"] = potential_abbreviations
            await push_event(state, "potential_abbreviations", potential_abbreviations)

        return updates

    except Exception as e:
        logger.warning(f"[abbr_expander] Failed to expand abbreviations: {e}")

    return {}


# ---------------------------------------------------------------------------
# Node: tool_executor
# ---------------------------------------------------------------------------


async def tool_executor(state: "AgentState") -> dict:
    """
    Dispatch to the appropriate tool based on classified intent.

    Updates: sources, images, image_parts, kg_summaries, tool_called, iterations
    Pushes: status, sources, images events into the SSE queue.
    """
    from app.services.agent import tools as _tools
    from app.services.agent.streaming import push_event, get_current_db

    intent = state.get("intent", "search")
    query = state.get("rewritten_query") or ""
    workspace_ids = state.get("workspace_ids", [])
    existing_ids = state.get("existing_citation_ids", set())
    # Đọc db từ ContextVar — bypass LangGraph TypedDict key filtering
    db = get_current_db()
    iterations = state.get("iterations", 0) + 1

    logger.info(f"[tool_executor] intent={intent!r} query={query!r}")

    # Emit status event indicating what we're doing
    tool_status_map = {
        "search": "Đang tìm kiếm tài liệu liên quan...",
        "list_docs": "Đang lấy danh sách tài liệu...",
        "summarize": "Đang tóm tắt tài liệu...",
        "kg_query": "Đang truy vấn đồ thị tri thức...",
    }
    await push_event(
        state,
        "status",
        {
            "step": "searching",
            "detail": tool_status_map.get(intent, "Đang xử lý yêu cầu..."),
        },
    )

    result_update: dict = {
        "tool_called": True,
        "iterations": iterations,
        "sources": [],
        "images": [],
        "image_parts": [],
        "kg_summaries": [],
    }

    try:
        if intent == "search":
            from app.core.config import settings

            top_k = settings.HRAG_RERANKER_TOP_K

            tool_result = await _tools.search_documents(
                query=query,
                top_k=top_k,
                workspace_ids=workspace_ids,
                existing_citation_ids=existing_ids,
                db=db,
            )
            result_update["sources"] = tool_result["sources"]
            result_update["images"] = tool_result["images"]
            result_update["image_parts"] = tool_result["image_parts"]
            result_update["kg_summaries"] = tool_result["kg_summaries"]

        elif intent == "list_docs":
            tool_result = await _tools.list_documents(
                workspace_ids=workspace_ids,
                db=db,
            )
            # Store list result as a synthetic "source" for context
            result_update["kg_summaries"] = [tool_result["text"]]

        elif intent == "summarize":
            # Extract document ID from query if present, else use first doc
            import re

            doc_id_match = re.search(
                r"\b(?:id\s*[:=]?\s*)?(\d+)\b", query, re.IGNORECASE
            )
            doc_id = int(doc_id_match.group(1)) if doc_id_match else 0

            if doc_id:
                tool_result = await _tools.summarize_document(
                    document_id=doc_id,
                    db=db,
                )
                result_update["kg_summaries"] = [tool_result["text"]]
            else:
                # Fallback to search if no doc ID found
                logger.warning(
                    "[tool_executor] summarize intent but no doc_id found — falling back to search"
                )
                from app.core.config import settings

                tool_result = await _tools.search_documents(
                    query=query,
                    top_k=settings.HRAG_RERANKER_TOP_K,
                    workspace_ids=workspace_ids,
                    existing_citation_ids=existing_ids,
                    db=db,
                )
                result_update["sources"] = tool_result["sources"]
                result_update["images"] = tool_result["images"]
                result_update["image_parts"] = tool_result["image_parts"]
                result_update["kg_summaries"] = tool_result["kg_summaries"]

        elif intent == "kg_query":
            tool_result = await _tools.query_knowledge_graph(
                entity=query,
                workspace_ids=workspace_ids,
                db=db,
            )
            result_update["kg_summaries"] = [tool_result["text"]]

        elif intent == "search_doc_num":
            tool_result = await _tools.search_documents_number(
                query=query,
                workspace_ids=workspace_ids,
                db=db,
            )
            result_update["doc_numbers"] = tool_result.get("results", [])
            result_update["tool_status"] = tool_result.get("status", "completed")

        elif intent == "search_abbr":
            logger.info(
                f"[tool_executor] SEARCH_ABBR: querying abbreviation for query={query!r}"
            )
            tool_result = await _tools.search_abbreviation(
                abbreviation=query,
                workspace_ids=workspace_ids,
                db=db,
            )
            logger.info(f"[tool_executor] SEARCH_ABBR: tool_result={tool_result!r}")
            # DEBUG: Log what we're storing
            if tool_result.get("results"):
                logger.info(
                    f"[tool_executor] SEARCH_ABBR: storing results={tool_result.get('results')!r}"
                )
            elif tool_result.get("found") and tool_result.get("abbreviation"):
                logger.info(
                    f"[tool_executor] SEARCH_ABBR: storing single result abbreviation={tool_result.get('abbreviation')!r}, full_form={tool_result.get('full_form')!r}"
                )
            else:
                logger.info(f"[tool_executor] SEARCH_ABBR: no results found")
            if tool_result.get("results"):
                result_update["abbreviation_results"] = tool_result.get("results", [])
            elif tool_result.get("found") and tool_result.get("abbreviation"):
                result_update["abbreviation_results"] = [
                    {
                        "short_form": tool_result.get("abbreviation"),
                        "full_form": tool_result.get("full_form"),
                        "description": tool_result.get("description"),
                    }
                ]
            else:
                result_update["abbreviation_results"] = []
            result_update["needs_clarification"] = tool_result.get(
                "needs_clarification", False
            )
            result_update["tool_status"] = tool_result.get("status", "completed")

            # Check if we should expand query for routing instead of direct answer
            abbreviation_results = tool_result.get("results") or (
                [tool_result] if tool_result.get("found") else []
            )
            if abbreviation_results:
                first_result = abbreviation_results[0]
                full_form = first_result.get("full_form", "")
                if full_form:
                    # Check if query contains additional context beyond the abbreviation
                    # Simple heuristic: query length significantly longer than abbrev
                    abbrev_len = len(first_result.get("abbreviation", ""))
                    query_len = len(query)
                    has_context = query_len > abbrev_len + 10
                    if has_context:
                        # Replace abbreviation with full form in the original query
                        abbrev = first_result.get("abbreviation", "")
                        expanded = query.replace(abbrev, full_form)
                        result_update["expanded_query"] = expanded
                        logger.info(
                            f"[tool_executor] SEARCH_ABBR: Expanding query for routing. "
                            f"Original: {query!r}, Expanded: {expanded!r}"
                        )

        else:
            logger.warning(
                f"[tool_executor] Unknown intent {intent!r}, defaulting to search"
            )
            from app.core.config import settings

            # Use expanded_query if available (from abbreviation + context detection)
            search_query = state.get("expanded_query") or query
            tool_result = await _tools.search_documents(
                query=search_query,
                top_k=settings.HRAG_RERANKER_TOP_K,
                workspace_ids=workspace_ids,
                existing_citation_ids=existing_ids,
                db=db,
            )
            if search_query != query:
                logger.info(
                    f"[tool_executor] Using expanded_query for search: {search_query!r}"
                )
            result_update["sources"] = tool_result["sources"]
            result_update["images"] = tool_result["images"]
            result_update["image_parts"] = tool_result["image_parts"]
            result_update["kg_summaries"] = tool_result["kg_summaries"]

    except Exception as e:
        logger.error(f"[tool_executor] Tool execution failed: {e}")

    # ── Push sources and images events into the SSE queue ───────────────────
    sources = result_update.get("sources", [])
    images = result_update.get("images", [])

    if sources:
        logger.info(f"[tool_executor] Pushing {len(sources)} sources to SSE queue")
        await push_event(state, "sources", sources)
        await push_event(
            state,
            "status",
            {
                "step": "retrieved",
                "detail": f"Tìm thấy {len(sources)} nguồn tài liệu liên quan",
            },
        )

    if images:
        logger.info(f"[tool_executor] Pushing {len(images)} images to SSE queue")
        await push_event(state, "images", images)

    return result_update


# ---------------------------------------------------------------------------
# Node: answer_generator
# ---------------------------------------------------------------------------


async def answer_generator(state: "AgentState") -> dict:
    """
    Main LLM node — generates final answer using retrieved context.

    Reads sources/images/kg_summaries from state, builds context string,
    streams tokens via provider.astream() and pushes each token to the SSE queue.
    Returns final_answer for state persistence.
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage as _LLMMsg
    from app.services.agent.streaming import push_event

    provider = get_llm_provider()

    await push_event(
        state, "status", {"step": "generating", "detail": "Đang tạo câu trả lời..."}
    )

    # Build context from accumulated retrieval results
    sources = state.get("sources", [])
    kg_summaries = state.get("kg_summaries", [])
    abbreviation_results = state.get("abbreviation_results", [])
    intent = state.get("intent", "")
    rewritten_query = state.get("rewritten_query", "")
    system_prompt = state.get("system_prompt", "")
    user_memory = state.get("user_memory_context", "")
    messages = state.get("messages", [])
    enable_thinking = state.get("enable_thinking", False)
    potential_abbreviations = state.get("potential_abbreviations", [])

    # DEBUG: Log state at start of answer_generator
    logger.info(
        f"[answer_generator] START - intent={intent!r}, rewritten_query={rewritten_query!r}"
    )
    logger.info(
        f"[answer_generator] sources={len(sources)} items, kg_summaries={len(kg_summaries)} items"
    )
    logger.info(f"[answer_generator] abbreviation_results={abbreviation_results!r}")

    # Inject memory into system prompt if available
    effective_system = system_prompt
    if user_memory and "No relevant memories" not in user_memory:
        effective_system = (
            f"USER MEMORY:\n{user_memory}\n\n"
            "Use this info when relevant. Do NOT include the header 'USER MEMORY' in your response.\n"
            "IMPORTANT: Do NOT add any citation markers like [id1], [mem1], [1] etc. when using memory facts.\n\n"
        ) + effective_system

    # Build context string
    context_parts = []
    if kg_summaries:
        context_parts.append(
            "## Knowledge Graph / Tool Results\n" + "\n\n".join(kg_summaries)
        )

    # Add abbreviation search results to context
    abbreviation_results = state.get("abbreviation_results", [])
    if abbreviation_results:
        ab_parts = ["## Abbreviation Results\n"]
        for ab in abbreviation_results:
            short_form = ab.get("short_form", "")
            full_form = ab.get("full_form", "")
            description = ab.get("description", "")
            ab_parts.append(f"- **{short_form}** = {full_form}")
            if description:
                ab_parts.append(f"  Mô tả: {description}")
        context_parts.append("\n".join(ab_parts))
        logger.info(
            f"[answer_generator] Added {len(abbreviation_results)} abbreviation results to context"
        )

    if sources:
        chunk_parts = []
        for src in sources:
            cid = src.get("index", "??")
            content = src.get("content", "")
            source_file = src.get("source_file", "")
            page_no = src.get("page_no", 0)
            heading_path = src.get("heading_path", [])
            meta_parts = []
            if source_file:
                meta_parts.append(source_file)
            if page_no:
                meta_parts.append(f"page {page_no}")
            if heading_path:
                meta_parts.append(" > ".join(heading_path))
            meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
            chunk_parts.append(f"Source [{cid}]{meta_line}:\n{content}")

        context_parts.append("## Document Chunks\n" + "\n\n---\n\n".join(chunk_parts))

    # Build llm messages — convert from state messages
    llm_messages: list[_LLMMsg] = []
    for msg in (messages or [])[-10:]:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = _get_msg_role(msg) or "user"
            content = getattr(msg, "content", "")
        llm_messages.append(_LLMMsg(role=role, content=content))

    # Inject context as the last "user" turn supplement
    if context_parts:
        context_text = "\n\n".join(context_parts)
        query_msg = f"Question: {rewritten_query}" if rewritten_query else ""
        inject = (
            "\n\n=== RETRIEVED CONTEXT ===\n"
            + (f"{query_msg}\n\n" if query_msg else "")
            + context_text
            + "\n=== END CONTEXT ===\n\n"
            "INSTRUCTIONS:\n"
            "- Answer using ONLY the retrieved sources above.\n"
            "- Cite sources using their unique IDs after each sentence, e.g. [id1][id2].\n"
            "- If the sources do not contain enough information to answer fully, "
            "provide as much detail as possible and clearly state what is missing.\n"
            "- Avoid saying 'information not found' if you can provide a partial answer.\n"
            "- TABLE DATA: 'Key, Year = Value' pairs are table cells.\n"
        )
        # Append to last user message
        if llm_messages and llm_messages[-1].role == "user":
            llm_messages[-1] = _LLMMsg(
                role="user",
                content=llm_messages[-1].content + inject,
            )
        else:
            llm_messages.append(_LLMMsg(role="user", content=inject))

    from app.core.config import settings

    # Stream tokens in real-time via astream()
    answer_parts: list[str] = []
    thinking_parts: list[str] = []

    try:
        async for chunk in provider.astream(
            messages=llm_messages,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            system_prompt=effective_system,
            think=enable_thinking,
        ):
            # Handle thinking tokens (extended thinking mode)
            if chunk.type == "thinking" and chunk.text:
                thinking_parts.append(chunk.text)
                await push_event(state, "thinking", {"text": chunk.text})

            # Handle regular answer tokens
            elif chunk.type == "text" and chunk.text:
                answer_parts.append(chunk.text)
                await push_event(state, "token", chunk.text)

    except Exception as e:
        logger.error(f"[answer_generator] LLM streaming failed: {e}", exc_info=True)
        # Fallback: try non-streaming
        try:
            result = await provider.acomplete(
                messages=llm_messages,
                temperature=0.1,
                max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                system_prompt=effective_system,
                think=enable_thinking,
            )
            fallback_answer = (
                result
                if isinstance(result, str)
                else getattr(result, "content", str(result))
            )
            answer_parts.append(fallback_answer)
            await push_event(state, "token", fallback_answer)
        except Exception as e2:
            logger.error(f"[answer_generator] Fallback also failed: {e2}")
            error_msg = "Xin lỗi, tôi gặp lỗi khi tạo câu trả lời. Vui lòng thử lại."
            answer_parts.append(error_msg)
            await push_event(state, "token", error_msg)

    final_answer = "".join(answer_parts)

    # Nếu không tìm thấy tài liệu và có từ viết tắt tiềm năng -> gợi ý thêm
    is_not_found = "không tìm thấy tài liệu phù hợp câu hỏi" in [s.lower() for s in kg_summaries]
    if is_not_found and potential_abbreviations:
        suggestion = "\n\nBạn có muốn thêm giải thích cho các từ viết tắt này không?"
        final_answer += suggestion
        await push_event(state, "token", suggestion)
        await push_event(state, "potential_abbreviations", potential_abbreviations)
        logger.info(f"[answer_generator] Pushed potential_abbreviations: {potential_abbreviations}")

    return {"final_answer": final_answer}


# ---------------------------------------------------------------------------
# Node: direct_answer
# ---------------------------------------------------------------------------


async def direct_answer(state: "AgentState") -> dict:
    """
    Answer greetings / chitchat directly without document retrieval.
    Uses the main LLM provider with memory context if available.
    Streams tokens in real-time via push_event.
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage as _LLMMsg
    from app.core.config import settings
    from app.services.agent.streaming import push_event

    provider = get_llm_provider()
    messages = state.get("messages", [])
    system_prompt = state.get("system_prompt", "")
    user_memory = state.get("user_memory_context", "")
    enable_thinking = state.get("enable_thinking", False)

    await push_event(
        state, "status", {"step": "generating", "detail": "Đang trả lời..."}
    )

    intent = state.get("intent", "greeting")
    effective_system = system_prompt
    if user_memory and "No relevant memories" not in user_memory:
        if intent == "personal":
            effective_system = (
                f"USER MEMORY:\n{user_memory}\n\n"
                "Answer directly about the user. Do NOT include the header 'USER MEMORY' in your response.\n"
                "IMPORTANT: Do NOT add any citation markers like [id1], [mem1], [1] etc. when using memory facts.\n\n"
            ) + effective_system
        else:
            effective_system = (
                f"USER MEMORY:\n{user_memory}\n\n"
                "Use this info when relevant. Do NOT include the header 'USER MEMORY' in your response.\n"
                "IMPORTANT: Do NOT add any citation markers like [id1], [mem1], [1] etc. when using memory facts.\n\n"
            ) + effective_system

    llm_messages: list[_LLMMsg] = []
    for msg in (messages or [])[-6:]:
        if isinstance(msg, dict):
            role, content = msg.get("role", "user"), msg.get("content", "")
        else:
            role, content = getattr(msg, "role", "user"), getattr(msg, "content", "")
        llm_messages.append(_LLMMsg(role=role, content=content))

    answer_parts: list[str] = []

    try:
        async for chunk in provider.astream(
            messages=llm_messages,
            temperature=0.5,
            max_tokens=512,
            system_prompt=effective_system,
            think=enable_thinking,
        ):
            if chunk.text:
                answer_parts.append(chunk.text)
                await push_event(state, "token", chunk.text)

    except Exception as e:
        logger.error(f"[direct_answer] LLM streaming failed: {e}", exc_info=True)
        # Fallback: non-streaming
        try:
            result = await provider.acomplete(
                messages=llm_messages,
                temperature=0.5,
                max_tokens=512,
                system_prompt=effective_system,
                think=enable_thinking,
            )
            fallback = (
                result
                if isinstance(result, str)
                else getattr(result, "content", str(result))
            )
            answer_parts.append(fallback)
            await push_event(state, "token", fallback)
        except Exception as e2:
            logger.error(f"[direct_answer] Fallback also failed: {e2}")
            greeting = "Xin chào! Tôi có thể giúp gì cho bạn?"
            answer_parts.append(greeting)
            await push_event(state, "token", greeting)

    return {"final_answer": "".join(answer_parts)}


# ---------------------------------------------------------------------------
# Node: write_executor  (true subgraph invocation)
# ---------------------------------------------------------------------------

# Cached compiled write subgraph — built once on first use
_write_subgraph = None


def _get_write_subgraph():
    """Lazy singleton for the compiled agent_write subgraph."""
    global _write_subgraph
    if _write_subgraph is None:
        from app.services.agents.agent_write import create_agent_write
        _write_subgraph = create_agent_write()
        logger.info("[write_executor] agent_write subgraph compiled and cached")
    return _write_subgraph


def _transform_input(state: "AgentState") -> dict:
    """
    Map AgentState → AgentWriteState.

    AgentWriteState keys: messages, user_id, workspace_ids,
                          text_input, write_action, result, error

    Ưu tiên text_input:
    1. text_input từ intent_classifier (write_summarize / write_suggest_edits / write_grammar_check)
    2. kg_summaries[0] nếu intent = summarize (RAG đã fetch raw document content)
    3. Fallback: last user message
    """
    intent = state.get("intent", "")
    write_action = state.get("write_action", "")
    text_input = state.get("text_input", "")

    # Khi intent = "summarize": RAG subgraph đã fetch raw doc content vào kg_summaries
    # → dùng kg_summaries[0] làm text_input cho write agent
    if not text_input and intent == "summarize":
        kg_summaries = state.get("kg_summaries", [])
        if kg_summaries:
            text_input = kg_summaries[0]
            logger.info(
                f"[_transform_input] Using kg_summaries[0] as text_input for summarize "
                f"(len={len(text_input)})"
            )

    # Fallback: extract text from last user message if classifier didn't isolate it
    if not text_input:
        text_input = _extract_last_user_message(state)

    # Fallback: derive action from intent when classifier left write_action blank
    if not write_action:
        write_action = {
            "write_summarize": "summarize",
            "write_suggest_edits": "suggest_edits",
            "write_grammar_check": "grammar_check",
            "summarize": "summarize",  # RAG-triggered summarize intent
        }.get(intent, "summarize")

    return {
        "messages": [],            # write subgraph doesn't need chat history
        "user_id": state.get("user_id"),
        "workspace_ids": state.get("workspace_ids", []),
        "text_input": text_input,
        "write_action": write_action,
        "result": "",
        "error": None,
    }


def _transform_output(write_result: dict) -> dict:
    """
    Map AgentWriteState output → AgentState partial update.

    Picks the final answer from result (or error fallback) and
    returns only the keys that belong to AgentState.
    """
    result_text = write_result.get("result", "")
    error = write_result.get("error")
    if error and not result_text:
        result_text = f"Lỗi xử lý văn bản: {error}"
    return {"final_answer": result_text}


async def write_executor(state: "AgentState") -> dict:
    """
    True subgraph node: invokes the compiled agent_write LangGraph as a child graph.

    Flow:
        AgentState
          ↓  _transform_input()
        AgentWriteState  ──▶  agent_write subgraph
          (route_write_action → summarize/suggest_edits/grammar → answer node)
          ↓  _transform_output()
        AgentState partial update  { final_answer: str }

    Handles intents: write_summarize, write_suggest_edits, write_grammar_check.
    Streams the result as tokens into SSE after subgraph completes.
    """
    from app.services.agent.streaming import push_event

    intent = state.get("intent", "")
    write_action = state.get("write_action", "") or {
        "write_summarize": "summarize",
        "write_suggest_edits": "suggest_edits",
        "write_grammar_check": "grammar_check",
    }.get(intent, "summarize")

    logger.info(
        f"[write_executor] intent={intent!r} write_action={write_action!r} "
        f"text_input={str(state.get('text_input', ''))[:80]!r}"
    )

    await push_event(
        state,
        "status",
        {"step": "processing", "detail": "Đang xử lý văn bản..."},
    )

    # ── Transform: AgentState → AgentWriteState ──────────────────────────
    write_input = _transform_input(state)

    # ── Invoke child graph ────────────────────────────────────────────────
    try:
        subgraph = _get_write_subgraph()
        write_output = await subgraph.ainvoke(write_input)
        logger.info(
            f"[write_executor] subgraph completed, result_len={len(write_output.get('result', ''))}"
        )
    except Exception as e:
        logger.error(f"[write_executor] subgraph invocation failed: {e}", exc_info=True)
        write_output = {"result": "", "error": str(e)}

    # ── Transform: AgentWriteState → AgentState partial update ───────────
    partial = _transform_output(write_output)
    result_text = partial.get("final_answer", "")

    # ── Stream result as tokens (~80 chars per chunk for smooth UX) ───────
    if result_text:
        chunk_size = 80
        for i in range(0, len(result_text), chunk_size):
            await push_event(state, "token", result_text[i : i + chunk_size])

    return partial


# ---------------------------------------------------------------------------
# Node: agent_rag_executor  (true subgraph invocation)
# ---------------------------------------------------------------------------

# Cached compiled rag subgraph — built once on first use
_rag_subgraph = None


def _get_rag_subgraph():
    """Lazy singleton for the compiled agent_rag subgraph."""
    global _rag_subgraph
    if _rag_subgraph is None:
        from app.services.agents.agent_rag import create_agent_rag
        _rag_subgraph = create_agent_rag()
        logger.info("[agent_rag_executor] agent_rag subgraph compiled and cached")
    return _rag_subgraph


def _transform_rag_input(state: "AgentState") -> dict:
    """
    Map AgentState → AgentRagState dict for the RAG subgraph.

    AgentRagState fields: messages, intent, rewritten_query, workspace_ids,
                          document_ids, sources, images, image_parts,
                          kg_summaries, abbreviation_results, final_answer
    """
    return {
        "messages": state.get("messages", []),
        "intent": state.get("intent", "search"),
        "rewritten_query": state.get("rewritten_query", ""),
        "workspace_ids": state.get("workspace_ids", []),
        "document_ids": state.get("document_ids"),
        "sources": [],
        "images": [],
        "image_parts": [],
        "kg_summaries": [],
        "abbreviation_results": [],
        "final_answer": None,
    }


def _transform_rag_output(rag_result: dict, state: "AgentState") -> dict:
    """
    Map AgentRagState output → AgentState partial update.

    Extracts: sources, images, image_parts, kg_summaries,
              abbreviation_results, tool_called, iterations.
    Does NOT set final_answer — that's left to answer_generator.
    """
    sources = rag_result.get("sources", []) or []
    images = rag_result.get("images", []) or []
    image_parts = rag_result.get("image_parts", []) or []
    kg_summaries = rag_result.get("kg_summaries", []) or []
    abbreviation_results = rag_result.get("abbreviation_results", []) or []

    # Inject the final_answer from RAG node into kg_summaries so answer_generator
    # can use it as context (for list_docs, summarize, kg_query, search_doc_num)
    # For search_documents, the context is already in sources, but final_answer
    # might contain extra formatting or KG summaries that are useful.
    final_answer_from_rag = rag_result.get("final_answer") or ""
    if final_answer_from_rag:
        # If it's a search intent, we prepend it to kg_summaries as a "Formatted Context Hint"
        # If it's a non-search intent, it's the primary answer content.
        if sources and isinstance(sources, list):
            # Check if final_answer_from_rag is just a concatenation of sources
            # to avoid extreme redundancy. If it's short or seems processed, keep it.
            if len(final_answer_from_rag) > 100:
                kg_summaries = [f"### PRE-FORMATTED RAG CONTEXT:\n{final_answer_from_rag}"] + list(kg_summaries)
        else:
            kg_summaries = [final_answer_from_rag] + list(kg_summaries)

    return {
        "sources": sources,
        "images": images,
        "image_parts": image_parts,
        "kg_summaries": kg_summaries,
        "abbreviation_results": abbreviation_results,
        "tool_called": True,
        "iterations": state.get("iterations", 0) + 1,  # Increment properly
    }


async def agent_rag_executor(state: "AgentState") -> dict:
    """
    True subgraph node: invokes the compiled agent_rag LangGraph as a child graph.

    Flow:
        AgentState
          ↓  _transform_rag_input()
        AgentRagState  ──▶  agent_rag subgraph
          (routes by intent → search_documents | list_documents | ... )
          ↓  _transform_rag_output()
        AgentState partial update  → answer_generator

    Handles all RAG intents: search, list_docs, summarize, kg_query,
    search_doc_num, search_abbr.
    """
    from app.services.agent.streaming import push_event

    intent = state.get("intent", "search")
    query = state.get("rewritten_query", "")
    workspace_ids = state.get("workspace_ids", [])
    existing_ids = state.get("existing_citation_ids", set())

    logger.info(
        f"[agent_rag_executor] intent={intent!r} query={query!r} "
        f"workspaces={workspace_ids}"
    )

    # Emit status event
    tool_status_map = {
        "search": "Đang tìm kiếm tài liệu liên quan...",
        "list_docs": "Đang lấy danh sách tài liệu...",
        "summarize": "Đang tóm tắt tài liệu...",
        "kg_query": "Đang truy vấn đồ thị tri thức...",
        "search_doc_num": "Đang tra cứu số văn bản...",
        "search_abbr": "Đang tra cứu viết tắt...",
    }
    await push_event(
        state,
        "status",
        {
            "step": "searching",
            "detail": tool_status_map.get(intent, "Đang xử lý yêu cầu..."),
        },
    )

    # ── Transform: AgentState → AgentRagState ────────────────────────────
    rag_input = _transform_rag_input(state)

    # ── Invoke child graph ────────────────────────────────────────────────
    try:
        subgraph = _get_rag_subgraph()
        rag_output = await subgraph.ainvoke(rag_input)
        logger.info(
            f"[agent_rag_executor] subgraph completed, "
            f"sources={len(rag_output.get('sources', []))}, "
            f"final_answer_len={len(str(rag_output.get('final_answer', '')))}"
        )
    except Exception as e:
        logger.error(f"[agent_rag_executor] subgraph invocation failed: {e}", exc_info=True)
        rag_output = {
            "sources": [], "images": [], "image_parts": [],
            "kg_summaries": [], "abbreviation_results": [], "final_answer": None,
        }

    # ── Transform: AgentRagState → AgentState partial update ─────────────
    partial = _transform_rag_output(rag_output, state)

    # ── Push sources and images events into the SSE queue ────────────────
    sources = partial.get("sources", [])
    images = partial.get("images", [])

    if sources:
        logger.info(f"[agent_rag_executor] Pushing {len(sources)} sources to SSE")
        await push_event(state, "sources", sources)
        await push_event(
            state,
            "status",
            {
                "step": "retrieved",
                "detail": f"Tìm thấy {len(sources)} nguồn tài liệu liên quan",
            },
        )

    if images:
        logger.info(f"[agent_rag_executor] Pushing {len(images)} images to SSE")
        await push_event(state, "images", images)

    return partial


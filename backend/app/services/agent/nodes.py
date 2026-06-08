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
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def strip_thinking_tags(text: str) -> str:
    """
    Remove <think>...</think> or <thought>...</thought> tags and their content
    from a string. Acts as a safety net if the provider fails to strip them.
    Case-insensitive.
    """
    if not text:
        return ""
    # Strip both <think> and <thought> tags and their contents
    # Case-insensitive, dotall for multiline thinking.
    # Use \s* at start and end to catch associated newlines.
    cleaned = re.sub(
        r"\s*<(think|thought)>[\s\S]*?<\/\1>\s*", "\n", text, flags=re.IGNORECASE
    )
    # Also catch unclosed tags at the end of the string
    cleaned = re.sub(r"\s*<(think|thought)>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


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
- "resolve_doc" : user references a specific document by name, type, or year without providing the document ID (e.g., "Luật An ninh mạng 2025", "Nghị định 60/2019", "Thông tư 23"). The system needs to resolve the reference to a document UUID before searching.
- "write_summarize"     : user provides a TEXT PASSAGE and wants it summarized or key points extracted
- "write_suggest_edits" : user provides a TEXT PASSAGE and wants editing/improvement suggestions
- "write_grammar_check" : user provides a TEXT PASSAGE and wants grammar/style checking
- "write_format_check"  : user wants to CHECK or EVALUATE the FORMATTING of an attached Word document (margins, fonts, line spacing, etc.) — user mentions "định dạng", "kiểm tra định dạng", "format", "căn lề", "cỡ chữ", "trình bày"
- "mongo_search_cccd"  : user asks to look up a person by their CCCD (Căn cước công dân) number. The query contains a national ID number (12 digits).
- "mongo_search_name"  : user asks to find/search for a person by their name.
- "mongo_search_bhxh"  : user asks to look up a person by their BHXH (Bảo hiểm xã hội) number.
- "mongo_search_phone"  : user asks to find a person by their phone number.

Output format:
{"intent": "<category>", "rewritten_query": "<improved Vietnamese/English search query>", "needs_tool": true|false, "write_action": "<action or empty>", "text_input": "<extracted text or empty>"}

Rules:
- For "greeting": set rewritten_query to "" and needs_tool to false
- For "personal": set rewritten_query to the user's question verbatim and needs_tool to false
- For "search_abbr" (abbreviation queries): ONLY if the target is a short abbreviation (usually uppercase, 2-6 chars). Otherwise default to "search".
- For "search_doc_num": set rewritten_query to ONLY the exact document number or ID (e.g., "172/GM-UBND"), without any extra words.
- For write intents: extract the text to process into "text_input", set write_action to the specific action, set needs_tool to false
- For "write_summarize": write_action = "summarize" (or "extract_key_points" if user asks for key points)
- For "write_suggest_edits": write_action = "suggest_edits"
- For "write_grammar_check": write_action = "grammar_check"
- For "write_format_check": write_action = "format_check", set needs_tool to false
- For "mongo_search_cccd": rewritten_query = the CCCD number itself (digits only, 9-12 digits)
- For "mongo_search_name": rewritten_query = the person's name or partial name
- For "mongo_search_bhxh": rewritten_query = the BHXH number
- For "mongo_search_phone": rewritten_query = the phone number
- For "resolve_doc": set rewritten_query to the full document reference as-is (e.g., "Luật An ninh mạng 2025", "Nghị định 60/2019"). Preserve the full reference including type keywords, names, and years.
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
User: "tóm tắt @quyche2024.pdf" → {"intent": "summarize", "rewritten_query": "tóm tắt @quyche2024.pdf", "needs_tool": true, "write_action": "", "text_input": ""}
User: "summarize @report.pdf" → {"intent": "summarize", "rewritten_query": "summarize @report.pdf", "needs_tool": true, "write_action": "", "text_input": ""}
User: "tìm văn bản số 60/QĐ-UBND giúp tôi" → {"intent": "search_doc_num", "rewritten_query": "60/QĐ-UBND", "needs_tool": true, "write_action": "", "text_input": ""}
User: "BMNN là gì?" → {"intent": "search_abbr", "rewritten_query": "BMNN", "needs_tool": true, "write_action": "", "text_input": ""}
User: "tìm người có CCCD 079203012345" → {"intent": "mongo_search_cccd", "rewritten_query": "079203012345", "needs_tool": true, "write_action": "", "text_input": ""}
User: "tra cứu CCCD 079203012345" → {"intent": "mongo_search_cccd", "rewritten_query": "079203012345", "needs_tool": true, "write_action": "", "text_input": ""}
User: "tìm ông Nguyễn Văn A" → {"intent": "mongo_search_name", "rewritten_query": "Nguyễn Văn A", "needs_tool": true, "write_action": "", "text_input": ""}
User: "ai có mã BHXH 1234567890" → {"intent": "mongo_search_bhxh", "rewritten_query": "1234567890", "needs_tool": true, "write_action": "", "text_input": ""}
User: "số điện thoại 0909123456" → {"intent": "mongo_search_phone", "rewritten_query": "0909123456", "needs_tool": true, "write_action": "", "text_input": ""}
User: "tìm người qua số BHXH 001234567890" → {"intent": "mongo_search_bhxh", "rewritten_query": "001234567890", "needs_tool": true, "write_action": "", "text_input": ""}
User: "tóm tắt đoạn văn sau: [đoạn văn dài]" → {"intent": "write_summarize", "rewritten_query": "", "needs_tool": false, "write_action": "summarize", "text_input": "[đoạn văn dài]"}
User: "kiểm tra ngữ pháp: Hôm nay tôi đi học." → {"intent": "write_grammar_check", "rewritten_query": "", "needs_tool": false, "write_action": "grammar_check", "text_input": "Hôm nay tôi đi học."}
User: "đề xuất chỉnh sửa văn bản này: [nội dung]" → {"intent": "write_suggest_edits", "rewritten_query": "", "needs_tool": false, "write_action": "suggest_edits", "text_input": "[nội dung]"}
User: "kiểm tra định dạng file đính kèm" → {"intent": "write_format_check", "rewritten_query": "", "needs_tool": false, "write_action": "format_check", "text_input": ""}
User: "đánh giá thể thức văn bản Word này" → {"intent": "write_format_check", "rewritten_query": "", "needs_tool": false, "write_action": "format_check", "text_input": ""}
User: "check format of attached document" → {"intent": "write_format_check", "rewritten_query": "", "needs_tool": false, "write_action": "format_check", "text_input": ""}
User: "Tóm tắt điều 27 Luật An ninh mạng 2025" → {"intent": "resolve_doc", "rewritten_query": "Luật An ninh mạng 2025", "needs_tool": true, "write_action": "", "text_input": ""}
User: "Nghị định 60/2019 là gì?" → {"intent": "resolve_doc", "rewritten_query": "Nghị định 60/2019", "needs_tool": true, "write_action": "", "text_input": ""}
User: "tìm Thông tư 23/2021/TT-BYT" → {"intent": "resolve_doc", "rewritten_query": "Thông tư 23/2021/TT-BYT", "needs_tool": true, "write_action": "", "text_input": ""}
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
    "resolve_doc",   # resolve document reference to UUID by type/title/year
    "write_summarize",
    "write_suggest_edits",
    "write_grammar_check",
    "write_format_check",
    # mongo people search intents
    "mongo_search_cccd",
    "mongo_search_name",
    "mongo_search_bhxh",
    "mongo_search_phone",
}

# Intent classification cache — LRU with TTL 60s and maxsize 500
# Skips the memory-agent LLM call for repeated/folded queries within a session.
import hashlib as _hashlib
import time as _time
from collections import OrderedDict

_INTENT_CACHE: OrderedDict[str, tuple[dict, float]] = OrderedDict()
_CACHE_MAXSIZE = 500
_CACHE_TTL = 60.0  # seconds


def _get_cache_key(message: str) -> str:
    """Fast cache key: SHA256 truncated to 32 chars."""
    return _hashlib.sha256(message.encode()).hexdigest()[:32]


def _get_cached_intent(message: str) -> dict | None:
    """LRU lookup with TTL check. Moves accessed item to end."""
    cache_key = _get_cache_key(message)
    if cache_key in _INTENT_CACHE:
        result, cached_at = _INTENT_CACHE[cache_key]
        if _time.time() - cached_at < _CACHE_TTL:
            _INTENT_CACHE.move_to_end(cache_key)
            return result
        del _INTENT_CACHE[cache_key]
    return None


def _set_cached_intent(message: str, result: dict) -> None:
    """LRU insert with maxsize eviction (evicts oldest from front)."""
    cache_key = _get_cache_key(message)
    if cache_key in _INTENT_CACHE:
        _INTENT_CACHE.move_to_end(cache_key)
    elif len(_INTENT_CACHE) >= _CACHE_MAXSIZE:
        _INTENT_CACHE.popitem(last=False)
    _INTENT_CACHE[cache_key] = (result, _time.time())


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
        return {
            "intent": "search",
            "rewritten_query": "",
            "needs_tool": True,
            "write_action": "",
            "text_input": "",
        }


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

    # ── Intent classification cache (LRU + TTL) ────────────────────────────
    cached = _get_cached_intent(user_message)
    if cached is not None:
        logger.info(f"[intent_classifier] Cache hit for key={_get_cache_key(user_message)[:8]}…")
        return {
            "intent": cached["intent"],
            "rewritten_query": cached.get("rewritten_query") or user_message,
            "original_query": user_message,
            "write_action": cached.get("write_action", ""),
            "text_input": cached.get("text_input", ""),
        }

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
            "resolve_doc": "Tìm văn bản theo tên",
            "write_summarize": "Tóm tắt văn bản",
            "write_suggest_edits": "Đề xuất chỉnh sửa",
            "write_grammar_check": "Kiểm tra ngữ pháp",
            "write_format_check": "Kiểm tra định dạng",
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

        # Cache result (LRU + TTL, keyed on exact user message)
        _set_cached_intent(user_message, result)

        return {
            "intent": result["intent"],
            "rewritten_query": result["rewritten_query"] or user_message,
            "original_query": user_message,  # Store actual user message for validation
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
    # Loại trừ các token dính với dấu / hoặc - (như trong số hiệu văn bản 172/GM-UBND)
    import re

    abbr_candidates = re.findall(
        r"(?<!/)(?<!-)\b[A-ZĐẮẰẶẤẦẨẪẬẮẶẪẨẦ]{2,}\b(?!/)(?!-)", user_message
    )
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
                {
                    "step": "searching",
                    "detail": f"Mở rộng viết tắt: {expanded_message}",
                },
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
    images = state.get("images", [])
    new_sources = []
    new_images = []

    # Phase 3.6: Intent-aware thinking trigger
    # Rules (override supervisor's blanket decision with finer-grained logic):
    # - Complex synthesis intents (kg_query, summarize, search_section) → always think
    # - Simple extraction intents (search, search_doc_num, list_docs, resolve_doc):
    #     * Single / few-doc query (len(document_ids) <= 2) → no thinking
    #       Rationale: extractive answer from a single document, no synthesis needed.
    #       Skipping thinking removes the 5–15s "no progress" gap that broke the
    #       streaming UX.
    #     * Multi-doc synthesis (len(document_ids) > 2) → thinking helps
    #       The LLM needs to compare/contrast across 3+ documents, structure themes.
    # - Abbreviation / greeting / direct → no thinking needed
    _ALWAYS_THINK_INTENTS = {"kg_query", "summarize", "search_section"}
    _SIMPLE_INTENTS = {"search", "search_doc_num", "list_docs", "resolve_doc"}
    _NO_THINK_INTENTS = {"search_abbr", "greeting", "personal"}
    source_count = len(sources) + len(kg_summaries)
    doc_ids_count = len(state.get("document_ids") or [])

    if intent in _ALWAYS_THINK_INTENTS:
        enable_thinking = True
        logger.info(f"[answer_generator] Thinking FORCED for complex intent: {intent!r}")
    elif intent in _NO_THINK_INTENTS:
        enable_thinking = False
        logger.info(f"[answer_generator] Thinking DISABLED for {intent!r} (no-think intent)")
    elif intent in _SIMPLE_INTENTS:
        if doc_ids_count > 2:
            enable_thinking = True
            logger.info(
                f"[answer_generator] Thinking ENABLED for {intent!r} "
                f"(multi-doc synthesis: {doc_ids_count} docs)"
            )
        else:
            enable_thinking = False
            logger.info(
                f"[answer_generator] Thinking DISABLED for {intent!r} "
                f"(single/few-doc query: {doc_ids_count} docs, {source_count} sources)"
            )
    else:
        # Unknown/write intents: keep supervisor decision
        logger.info(f"[answer_generator] Using supervisor thinking decision: {enable_thinking} for {intent!r}")

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
            f"{user_memory}\n\n"
            "IMPORTANT: Do NOT copy these facts directly. When using a memory fact, "
            "paraphrase it in your own words and cite it as [MEM-1], [MEM-2], etc. "
            "For example: 'The user works at Công an tỉnh Hà Tĩnh [MEM-1]' instead of copying the fact verbatim. "
            "Only include relevant memories.\n\n"
        ) + effective_system

    # Build context string — always inject instructions (even when no sources)
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

    # ── resolve_doc → summarize: fetch document content if not yet retrieved ──
    doc_ids_for_summarize = state.get("document_ids") or []
    if intent == "summarize" and doc_ids_for_summarize and not sources:
        logger.info(f"[answer_generator] resolve_doc result pending summarize: fetching content for {len(doc_ids_for_summarize)} docs")
        from app.services.agent import tools as _tools
        from app.services.agent.streaming import get_current_db

        db = get_current_db()
        all_texts = []
        
        from app.schemas.rag import ChatSourceChunk
        import random, string
        existing_ids = {s.index if hasattr(s, "index") else s.get("index") for s in sources if (hasattr(s, "index") and s.index) or (isinstance(s, dict) and s.get("index"))}
        def _get_next_cid() -> str:
            chars = string.ascii_lowercase + string.digits
            while True:
                cid = "".join(random.choices(chars, k=4))
                if any(c.isalpha() for c in cid) and cid not in existing_ids:
                    existing_ids.add(cid)
                    return cid

        for i, doc_id in enumerate(doc_ids_for_summarize):
            try:
                # Use get_documents_content instead of summarize_document
                # to get RAW markdown content (not LLM summary)
                doc_uuid_str = str(doc_id)
                result = await _tools.get_documents_content(
                    document_ids=[doc_uuid_str],
                    db=db,
                )
                docs = result.get("documents", [])
                if docs and docs[0].get("content"):
                    # Use raw markdown content instead of summary
                    raw_content = docs[0]["content"]
                    doc_filename = docs[0].get("filename", "Unknown")
                    # Truncate if too long (keep first 50k chars to fit context)
                    MAX_CHARS = 50000
                    if len(raw_content) > MAX_CHARS:
                        raw_content = raw_content[:MAX_CHARS] + "\n\n[... nội dung đã được cắt bớt ...]"
                    
                    cid = _get_next_cid()
                    meta_line = f" ({doc_filename})"
                    all_texts.append(f"Source [{cid}]{meta_line}:\n{raw_content}")
                    
                    sources.append(ChatSourceChunk(
                        index=cid,
                        chunk_id=f"doc_{doc_id}_full",
                        content=raw_content[:500], # Preview for UI
                        document_id=doc_id,
                        page_no=0,
                        heading_path=[],
                        score=1.0,
                        source_type="vector",
                        source_file=doc_filename,
                    ))
                    logger.info(f"[answer_generator] fetched raw doc_id={doc_id} as [{cid}]: {len(raw_content)} chars")
                else:
                    # Fallback to summarize if no raw content
                    error = docs[0].get("error", "Unknown error") if docs else "No document returned"
                    logger.warning(f"[answer_generator] No raw content for {doc_id}, error: {error}")
                    summ_result = await _tools.summarize_document(document_id=doc_id, db=db)
                    if summ_result.get("text"):
                        cid = _get_next_cid()
                        doc_filename = summ_result.get("document_name", "Unknown")
                        meta_line = f" ({doc_filename})"
                        all_texts.append(f"Source [{cid}]{meta_line}:\n{summ_result['text']}")
                        new_sources.append(ChatSourceChunk(
                            index=cid,
                            chunk_id=f"doc_{doc_id}_summary",
                            content=summ_result["text"][:500],
                            document_id=doc_id,
                            page_no=0,
                            heading_path=[],
                            score=1.0,
                            source_type="vector",
                            source_file=doc_filename,
                        ))
                        logger.info(f"[answer_generator] fallback summarize for {doc_id} as [{cid}]: {len(summ_result['text'])} chars")
            except Exception as e:
                logger.warning(f"[answer_generator] get_documents_content failed for {doc_id}: {e}")
                try:
                    summ_result = await _tools.summarize_document(document_id=doc_id, db=db)
                    if summ_result.get("text"):
                        cid = _get_next_cid()
                        doc_filename = summ_result.get("document_name", "Unknown")
                        meta_line = f" ({doc_filename})"
                        all_texts.append(f"Source [{cid}]{meta_line}:\n{summ_result['text']}")
                        new_sources.append(ChatSourceChunk(
                            index=cid,
                            chunk_id=f"doc_{doc_id}_summary",
                            content=summ_result["text"][:500],
                            document_id=doc_id,
                            page_no=0,
                            heading_path=[],
                            score=1.0,
                            source_type="vector",
                            source_file=doc_filename,
                        ))
                        logger.info(f"[answer_generator] fallback summarize for {doc_id} as [{cid}]: {len(summ_result['text'])} chars")
                except Exception as e2:
                    logger.warning(f"[answer_generator] summarize_document also failed for {doc_id}: {e2}")

        if all_texts:
            context_parts.append("## Document Content (Raw Markdown)\n" + "\n\n---\n\n".join(all_texts))
            logger.info(f"[answer_generator] Added {len(all_texts)} document texts to context")
        else:
            # Fallback: use the resolve_doc result message
            logger.warning("[answer_generator] No document content fetched, using resolve_doc message")
            
        # PUSH to UI so frontend knows about these dynamically added sources!
        if sources:
            from app.services.agent.streaming import push_event
            await push_event(state, "sources", sources)

    if sources:
        chunk_parts = []
        for src in sources:
            if isinstance(src, dict):
                cid = src.get("index", "??")
                content = src.get("content", "")
                source_file = src.get("source_file", "")
                page_no = src.get("page_no", 0)
                heading_path = src.get("heading_path", [])
            else:
                cid = getattr(src, "index", "??")
                content = getattr(src, "content", "")
                source_file = getattr(src, "source_file", "")
                page_no = getattr(src, "page_no", 0)
                heading_path = getattr(src, "heading_path", [])

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
    # IMPORTANT: Truncate old assistant messages to prevent topic contamination.
    # User messages are kept intact (needed for context: "tài liệu này", "văn bản này").
    # Old assistant responses are truncated because they can contain long answers
    # about a DIFFERENT topic (e.g. "an ninh mạng") that biases the LLM
    # when answering a new question (e.g. "bí mật nhà nước").
    _HISTORY_ASSISTANT_TRUNCATE = 150  # chars — enough to indicate topic, not flood tokens
    recent_messages = (messages or [])[-10:]
    llm_messages: list[_LLMMsg] = []
    for i, msg in enumerate(recent_messages):
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = _get_msg_role(msg) or "user"
            content = getattr(msg, "content", "")

        # Truncate old assistant messages (all except the very last message)
        is_last_message = (i == len(recent_messages) - 1)
        if not is_last_message and role == "assistant" and len(content) > _HISTORY_ASSISTANT_TRUNCATE:
            content = content[:_HISTORY_ASSISTANT_TRUNCATE] + "… [lược bỏ]"

        llm_messages.append(_LLMMsg(role=role, content=content))

    # Always inject instructions so LLM never fabricates when context is empty
    # Use modular instructions based on intent — saves 30-60% tokens
    # Phase 3.6: pass enable_thinking to inject _THINKING_DIRECTIVE when active
    from app.prompts.agents.answer_instructions import get_instructions_for_intent

    context_text = "\n\n".join(context_parts) if context_parts else "(no retrieved context)"
    query_msg = f"Question: {rewritten_query}" if rewritten_query else ""
    intent_instructions = get_instructions_for_intent(intent, enable_thinking=enable_thinking)
    inject = (
        "\n\n=== RETRIEVED CONTEXT ===\n"
        + (f"{query_msg}\n\n" if query_msg else "")
        + context_text
        + "\n=== END CONTEXT ===\n\n"
        + intent_instructions
    )
    # Always append instructions (no `if context_parts:` guard)
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

    final_answer = strip_thinking_tags("".join(answer_parts))

    # Nếu không tìm thấy tài liệu và có từ viết tắt tiềm năng -> gợi ý thêm
    is_not_found = "không tìm thấy tài liệu phù hợp câu hỏi" in [
        s.lower() for s in kg_summaries
    ]
    if is_not_found and potential_abbreviations:
        suggestion = "\n\nBạn có muốn thêm giải thích cho các từ viết tắt này không?"
        final_answer += suggestion
        await push_event(state, "token", suggestion)
        await push_event(state, "potential_abbreviations", potential_abbreviations)
        logger.info(
            f"[answer_generator] Pushed potential_abbreviations: {potential_abbreviations}"
        )

    return {
        "final_answer": final_answer,
        "sources": new_sources,
        "images": new_images,
        "potential_abbreviations": potential_abbreviations,
    }


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

    await push_event(
        state, "status", {"step": "generating", "detail": "Đang trả lời..."}
    )

    intent = state.get("intent", "greeting")
    
    # For direct answers (greeting/personal), the massive RAG system prompt 
    # (which forces the LLM to complain if there are no sources) is counter-productive.
    # We build a focused system prompt that retains language rules but drops RAG rules.
    effective_system = (
        "You are a helpful AI assistant. "
        "You MUST answer in the SAME language as the user's question.\n"
        "- If the user asks in Vietnamese → answer entirely in Vietnamese.\n"
        "- If the user asks in English → answer entirely in English.\n"
    )
    
    if user_memory and "No relevant memories" not in user_memory:
        if intent == "personal":
            effective_system += (
                f"{user_memory}\n\n"
                "Answer the user's question using ONLY the memory above. "
                "Paraphrase each fact in your own words and cite as [MEM-1], [MEM-2], etc. "
                "Do NOT copy facts verbatim. Do NOT ask the user to upload documents.\n"
            )
        else:
            effective_system += (
                f"{user_memory}\n\n"
                "IMPORTANT: Paraphrase memory facts in your own words and cite as [MEM-1], [MEM-2], etc. Do NOT copy verbatim.\n"
            )
    else:
        if intent == "personal":
            effective_system += (
                "\nCRITICAL INSTRUCTION:\n"
                "The user is asking a personal question about themselves, but you do NOT have any memory of this.\n"
                "Politely inform the user that you do not know this information because they haven't shared it with you yet."
            )

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
            think=False,  # Disable thinking for greetings — fast, direct response
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
                think=False,  # Disable thinking for greetings — fast, direct response
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

    return {"final_answer": strip_thinking_tags("".join(answer_parts))}


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

    # Khi intent = "summarize" HOẶC write intents với attached docs:
    # RAG subgraph đã fetch raw doc content vào kg_summaries
    # → dùng kg_summaries[0] làm text_input cho write agent
    write_intents_with_kg = {"summarize", "write_summarize", "write_suggest_edits", "write_grammar_check"}
    if not text_input and intent in write_intents_with_kg:
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
            "write_format_check": "format_check",
            "summarize": "summarize",  # RAG-triggered summarize intent
        }.get(intent, "summarize")

    result = {
        "messages": [],  # write subgraph doesn't need chat history
        "user_id": state.get("user_id"),
        "workspace_ids": state.get("workspace_ids", []),
        "text_input": text_input,
        "write_action": write_action,
        "result": "",
        "error": None,
    }

    # Pass format_data for format_check action
    if write_action == "format_check":
        result["format_data"] = state.get("format_data")
        result["file_name"] = state.get("file_name", "tài liệu")

    return result


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
          (route_write_action → summarize/suggest_edits/grammar/format → answer node)
          ↓  _transform_output()
        AgentState partial update  { final_answer: str }

    Handles intents: write_summarize, write_suggest_edits, write_grammar_check, write_format_check.
    Streams the result as tokens into SSE after subgraph completes.
    """
    from app.services.agent.streaming import push_event

    intent = state.get("intent", "")
    write_action = state.get("write_action", "") or {
        "write_summarize": "summarize",
        "write_suggest_edits": "suggest_edits",
        "write_grammar_check": "grammar_check",
        "write_format_check": "format_check",
    }.get(intent, "summarize")

    doc_ids = state.get("document_ids") or []
    text_input = state.get("text_input", "")

    logger.info(
        f"[write_executor] intent={intent!r} write_action={write_action!r} "
        f"text_input={str(text_input)[:80]!r} doc_ids={doc_ids!r}"
    )

    await push_event(
        state,
        "status",
        {"step": "processing", "detail": "Đang xử lý văn bản..."},
    )

    # ── Guard: check for missing required data ───────────────────────────
    write_action_labels = {
        "summarize": "tóm tắt",
        "suggest_edits": "đề xuất chỉnh sửa",
        "grammar_check": "kiểm tra ngữ pháp",
        "format_check": "kiểm tra định dạng",
    }
    action_label = write_action_labels.get(write_action, write_action)

    if write_action == "format_check":
        # Format check REQUIRES a docx file attachment
        if not doc_ids:
            msg = (
                "Bạn chưa đính kèm file nào để kiểm tra định dạng.\n\n"
                "Vui lòng đính kèm một file Word (.docx) rồi hỏi lại, ví dụ:\n"
                "• \"Kiểm tra định dạng file đính kèm\"\n"
                "• \"Đánh giá thể thức văn bản này\""
            )
            for i in range(0, len(msg), 80):
                await push_event(state, "token", msg[i:i+80])
            return {"final_answer": msg}

        # Fetch format metadata
        logger.info(
            f"[write_executor] format_check: fetching format metadata for {len(doc_ids)} docs"
        )
        try:
            from app.services.agent import tools as _tools
            from app.services.agent.streaming import get_current_db

            db = get_current_db()
            tool_result = await _tools.get_document_format(
                document_ids=doc_ids,
                db=db,
            )
            docs_with_format = tool_result.get("documents", [])
            if docs_with_format:
                first_doc = docs_with_format[0]
                if first_doc.get("format_data"):
                    state["format_data"] = first_doc["format_data"]
                    state["file_name"] = first_doc.get("filename", "tài liệu")
                    logger.info(
                        f"[write_executor] format_check: extracted format for {first_doc.get('filename')}"
                    )
                elif first_doc.get("error"):
                    err = first_doc.get("error", "")
                    if "không phải file Word" in err.lower():
                        msg = (
                            f"File \"{first_doc.get('filename', 'của bạn')}\" không phải định dạng Word (.docx).\n\n"
                            "Hiện tại tôi chỉ hỗ trợ kiểm tra định dạng cho file Word (.docx).\n"
                            "Vui lòng đính kèm một file Word và thử lại."
                        )
                    else:
                        msg = (
                            f"Không thể đọc file \"{first_doc.get('filename', 'của bạn')}\": {err}\n\n"
                            "Vui lòng thử đính kèm file khác hoặc kiểm tra lại file gốc."
                        )
                    for i in range(0, len(msg), 80):
                        await push_event(state, "token", msg[i:i+80])
                    return {"final_answer": msg}
        except Exception as e:
            logger.warning(f"[write_executor] Failed to fetch format metadata: {e}")

    elif not doc_ids and not text_input:
        # Text-based write actions require either attached docs OR inline text
        msg = (
            f"Bạn chưa cung cấp văn bản để {action_label}.\n\n"
            "Vui lòng làm một trong các cách sau:\n"
            "• Dán nội dung văn bản trực tiếp vào tin nhắn\n"
            "• Đính kèm file tài liệu và hỏi lại (ví dụ: \"Tóm tắt file đính kèm\")"
        )
        for i in range(0, len(msg), 80):
            await push_event(state, "token", msg[i:i+80])
        return {"final_answer": msg}

    # ── Fetch referenced doc content via @docname mentions ─────────────────
    # When user references @docname but intent_classifier couldn't extract text
    # (write_summarize / write_suggest_edits / write_grammar_check with attached docs)
    elif doc_ids and not text_input:
        logger.info(
            f"[write_executor] Fetching content for {len(doc_ids)} referenced docs"
        )
        try:
            from app.services.agent import tools as _tools
            from app.services.agent.streaming import get_current_db

            db = get_current_db()
            tool_result = await _tools.get_documents_content(
                document_ids=doc_ids,
                db=db,
            )
            doc_texts = []
            fetch_errors = []
            for doc in tool_result.get("documents", []):
                if doc.get("content"):
                    doc_texts.append(
                        f"# {doc.get('filename', 'Document')}\n\n{doc.get('content')}"
                    )
                elif doc.get("error"):
                    fetch_errors.append(f"• {doc.get('filename', 'Document')}: {doc.get('error')}")
            if doc_texts:
                combined = "\n\n---\n\n".join(doc_texts)
                state["kg_summaries"] = [combined]
                logger.info(
                    f"[write_executor] Fetched {len(doc_texts)} docs, "
                    f"total content {len(combined)} chars"
                )
            if fetch_errors and not doc_texts:
                # All docs failed to fetch
                err_msg = (
                    "Không thể đọc được các file đính kèm:\n"
                    + "\n".join(fetch_errors)
                    + "\n\nVui lòng kiểm tra lại file hoặc thử đính kèm file khác."
                )
                for i in range(0, len(err_msg), 80):
                    await push_event(state, "token", err_msg[i:i+80])
                return {"final_answer": err_msg}
        except Exception as e:
            logger.warning(f"[write_executor] Failed to fetch doc content: {e}")

    # ── Transform: AgentState → AgentWriteState ──────────────────────────
    write_input = _transform_input(state)

    # ── Inject format_data for format_check ───────────────────────────────
    if write_action == "format_check":
        write_input["format_data"] = state.get("format_data")
        write_input["file_name"] = state.get("file_name", "tài liệu")

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
    # Note: tokens are already streamed in real-time via push_event() inside
    # the tool nodes (summarize_text_node, suggest_edits_node, etc.)
    partial = _transform_output(write_output)

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
        "mongo_results": [],
        "final_answer": None,
    }


def _transform_rag_output(rag_result: dict, state: "AgentState") -> dict:
    """
    Map AgentRagState output → AgentState partial update.

    Extracts: sources, images, image_parts, kg_summaries,
              abbreviation_results, tool_called, iterations.
    Sets final_answer from rag_result so answer_generator can use it directly for mongo intents.
    """
    sources = rag_result.get("sources", []) or []
    images = rag_result.get("images", []) or []
    image_parts = rag_result.get("image_parts", []) or []
    kg_summaries = rag_result.get("kg_summaries", []) or []
    abbreviation_results = rag_result.get("abbreviation_results", []) or []
    mongo_results = rag_result.get("mongo_results", []) or []

    # Inject the final_answer from RAG node into kg_summaries so answer_generator
    # can use it as context (for list_docs, summarize, kg_query, search_doc_num, mongo)
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
                kg_summaries = [
                    f"### PRE-FORMATTED RAG CONTEXT:\n{final_answer_from_rag}"
                ] + list(kg_summaries)
        else:
            kg_summaries = [final_answer_from_rag] + list(kg_summaries)

    return {
        "sources": sources,
        "images": images,
        "image_parts": image_parts,
        "kg_summaries": kg_summaries,
        "abbreviation_results": abbreviation_results,
        "mongo_results": mongo_results,
        "final_answer": final_answer_from_rag,  # Set for mongo intents so answer_generator can use it
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
        "mongo_search_cccd": "Đang tra cứu CCCD trong cơ sở dữ liệu...",
        "mongo_search_name": "Đang tìm kiếm người theo tên...",
        "mongo_search_bhxh": "Đang tra cứu BHXH trong cơ sở dữ liệu...",
        "mongo_search_phone": "Đang tìm kiếm người theo số điện thoại...",
    }
    await push_event(
        state,
        "status",
        {
            "step": "searching",
            "detail": tool_status_map.get(intent, "Đang xử lý yêu cầu..."),
        },
    )

    # ── FABRICATED QUERY GUARD ─────────────────────────────────────────────
    # For mongo searches, validate that rewritten_query actually appears in the
    # ORIGINAL user message. If LLM fabricated a phone/CCCD/BHXH number during
    # its reasoning, reject it here BEFORE wasting a subgraph call.
    original_query = state.get("original_query", "")
    mongo_intents = (
        "mongo_search_cccd",
        "mongo_search_bhxh",
        "mongo_search_phone",
        "mongo_search_name",
    )
    if intent in mongo_intents and query and original_query:
        # Check if the query value actually exists verbatim in the original question
        if query.strip() not in original_query.strip():
            logger.warning(
                f"[agent_rag_executor] FABRICATED query detected: {query!r} "
                f"not in original: {original_query!r} — skipping subgraph"
            )
            await push_event(
                state,
                "status",
                {
                    "step": "searching",
                    "detail": "Phát hiện truy vấn không hợp lệ — bỏ qua",
                },
            )
            return {
                "sources": [],
                "images": [],
                "image_parts": [],
                "kg_summaries": [],
                "abbreviation_results": [],
                "mongo_results": [],
                "tool_called": True,
                "iterations": state.get("iterations", 0) + 1,
            }

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
        logger.error(
            f"[agent_rag_executor] subgraph invocation failed: {e}", exc_info=True
        )
        rag_output = {
            "sources": [],
            "images": [],
            "image_parts": [],
            "kg_summaries": [],
            "abbreviation_results": [],
            "final_answer": None,
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

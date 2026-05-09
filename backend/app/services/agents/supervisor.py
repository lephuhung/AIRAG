"""
Supervisor Agent
================

LLM-based supervisor that decides which agent to invoke next.
Single LLM call does BOTH intent classification + routing decision.

Graph Flow:
    START → supervisor → [rag | write | direct] → supervisor (loop)
                                                        ↓
                                                       END

Replaces the dead code version that was never integrated.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END

from app.services.agents.models import (
    SupervisorState,
    AgentType,
)

if TYPE_CHECKING:
    from app.services.llm.types import LLMMessage

logger = logging.getLogger(__name__)

# =============================================================================
# Supervisor Prompt
# =============================================================================

from app.prompts.agents.supervisor_prompt import _SUPERVISOR_PROMPT

# Intent abbreviation → canonical name (safety net for LLM shortcutting intent names)
_INTENT_NORMALIZE: dict[str, str] = {
    # Common LLM shortcuts for people intents
    "search_phone": "mongo_search_phone",
    "search_cccd": "mongo_search_cccd",
    "search_name": "mongo_search_name",
    "search_bhxh": "mongo_search_bhxh",
    "phone_search": "mongo_search_phone",
    "cccd_search": "mongo_search_cccd",
    "name_search": "mongo_search_name",
    "bhxh_search": "mongo_search_bhxh",
    "search_advanced": "mongo_search_advanced",
    # some LLMs drop the prefix entirely
    "find_phone": "mongo_search_phone",
    "find_person": "mongo_search_name",
    "advanced_search": "mongo_search_advanced",
}

# Intent → Agent routing table used as fallback inside _parse_supervisor_response
_INTENT_TO_AGENT_FALLBACK: dict[str, str] = {
    "greeting": "direct",
    "personal": "direct",
    "search": "rag",
    "list_docs": "rag",
    "summarize": "rag",
    "kg_query": "rag",
    "search_doc_num": "rag",
    "search_abbr": "rag",
    "resolve_doc": "resolve_doc",  # Phase 2: dedicated agent
    "search_section": "rag",
    "write_summarize": "write",
    "write_suggest_edits": "write",
    "write_grammar_check": "write",
    "write_format_check": "write",
    "mongo_search_cccd": "people",
    "mongo_search_name": "people",
    "mongo_search_bhxh": "people",
    "mongo_search_phone": "people",
    "mongo_search_advanced": "people",
}

# ---------------------------------------------------------------------------
# Keyword-based safety net: Vietnamese document queries that MUST go to RAG.
# Small classifier models (Qwen3-4B) frequently misclassify these as "greeting"
# or "direct".  When ANY of these patterns match the user message AND the LLM
# chose "direct", we override to "rag" + intent="search".
# ---------------------------------------------------------------------------
import re as _re

_MUST_RAG_PATTERNS: list[_re.Pattern] = [
    # "X là gì" — definition questions about topics in the knowledge base
    _re.compile(r"(?:là\s+gì|nghĩa\s+là\s+gì)", _re.IGNORECASE),
    # Responsibility / obligation questions
    _re.compile(r"trách\s+nhiệm", _re.IGNORECASE),
    # Regulation / policy questions
    _re.compile(r"quy\s+định", _re.IGNORECASE),
    # Legal concepts
    _re.compile(r"(?:luật|nghị\s+định|thông\s+tư|điều\s+\d|chương\s+\d)", _re.IGNORECASE),
    # "khái niệm" — concept/definition
    _re.compile(r"khái\s+niệm", _re.IGNORECASE),
    # "điều kiện" — conditions
    _re.compile(r"điều\s+kiện", _re.IGNORECASE),
    # "nguyên tắc" — principles
    _re.compile(r"nguyên\s+tắc", _re.IGNORECASE),
    # "chủ quản" — manager/custodian (legal role, not a person search)
    _re.compile(r"chủ\s+quản", _re.IGNORECASE),
    # "hệ thống thông tin" — information system
    _re.compile(r"hệ\s+thống\s+thông\s+tin", _re.IGNORECASE),
    # "bảo vệ / bảo mật / an ninh / an toàn" — security topics
    _re.compile(r"(?:bảo\s+vệ|bảo\s+mật|an\s+ninh|an\s+toàn)", _re.IGNORECASE),
    # "tiêu hủy / lưu trữ / bảo quản" — archival / destruction
    _re.compile(r"(?:tiêu\s+hủy|lưu\s+trữ|bảo\s+quản)", _re.IGNORECASE),
    # "xử lý / xử phạt" — processing / penalties
    _re.compile(r"(?:xử\s+lý|xử\s+phạt)", _re.IGNORECASE),
    # "chế độ / chính sách" — policy / regime
    _re.compile(r"(?:chế\s+độ|chính\s+sách)", _re.IGNORECASE),
    # "thẩm quyền" — authority / jurisdiction
    _re.compile(r"thẩm\s+quyền", _re.IGNORECASE),
    # "tóm tắt" — summarize (but not write_summarize which has inline text)
    _re.compile(r"tóm\s+tắt", _re.IGNORECASE),
    # "so sánh" — compare
    _re.compile(r"so\s+sánh", _re.IGNORECASE),
    # "nội dung" — content of document
    _re.compile(r"nội\s+dung", _re.IGNORECASE),
    # Document number patterns
    _re.compile(r"văn\s+bản\s+số", _re.IGNORECASE),
    _re.compile(r"\d+/\d+/(?:NĐ|TT|QĐ|NQ)", _re.IGNORECASE),
]

# Strict greeting patterns — only override to direct if the ENTIRE message
# matches one of these (after stripping whitespace/punctuation).  Anything
# longer or containing topic keywords should NOT be treated as a greeting.
_GREETING_ONLY_PATTERNS: list[_re.Pattern] = [
    _re.compile(r"^(?:xin\s+)?chào[\s!.?]*$", _re.IGNORECASE),
    _re.compile(r"^(?:hi|hello|hey)[\s!.?]*$", _re.IGNORECASE),
    _re.compile(r"^cảm\s+ơn[\s!.?]*$", _re.IGNORECASE),
    _re.compile(r"^(?:tạm\s+biệt|bye)[\s!.?]*$", _re.IGNORECASE),
]


def _should_force_rag(message: str) -> bool:
    """Return True if message contains keywords that MUST be handled by RAG."""
    return any(p.search(message) for p in _MUST_RAG_PATTERNS)


def _is_pure_greeting(message: str) -> bool:
    """Return True only if message is a bare greeting with no topic content."""
    return any(p.match(message.strip()) for p in _GREETING_ONLY_PATTERNS)


# ---------------------------------------------------------------------------
# Phase 3: Personal reference detection — for smart memory_recall routing
# Detects queries that contain personal pronouns/references which require
# Graphiti memory to answer correctly (e.g. "đơn vị tôi", "cơ quan tôi").
# This pattern fires REGARDLESS of intent (even intent=search can need memory
# when the question references the user's own identity/workplace/org).
# ---------------------------------------------------------------------------
_PERSONAL_REF_PATTERN = _re.compile(
    r"\b("
    r"tôi|của\s+tôi|cho\s+tôi"
    r"|đơn\s+vị\s+tôi|cơ\s+quan\s+tôi|nơi\s+tôi|chỗ\s+tôi"
    r"|chúng\s+tôi|của\s+chúng\s+tôi"
    r"|công\s+tác\s+của\s+tôi|làm\s+việc\s+của\s+tôi"
    r")\b",
    _re.IGNORECASE | _re.UNICODE,
)


def _query_needs_memory(intent: str, query: str) -> bool:
    """Determine whether the memory_recall → query_enricher pipeline should run.

    Returns True when:
    - intent is 'personal' (user asking about themselves)
    - OR the query contains personal reference keywords (even for RAG intents,
      e.g. "Đơn vị tôi có cần tuân thủ quy định này không?" → intent=search
      but still needs memory to resolve "đơn vị tôi")
    """
    if intent == "personal":
        return True
    return bool(_PERSONAL_REF_PATTERN.search(query))

# =============================================================================
# Vietnamese stop words — these are NEVER abbreviations
_VI_STOP_WORDS: frozenset[str] = frozenset({
    "là", "và", "của", "có", "cho", "này", "đó", "với",
    "các", "được", "theo", "trong", "về", "từ", "đến",
    "khi", "nào", "như", "hay", "hoặc", "nếu", "thì",
    "sẽ", "đã", "đang", "tôi", "bạn", "anh", "chị",
    "gì", "nào", "sao", "thế", "nên", "nhưng", "mà",
    "ra", "vào", "lên", "xuống", "qua", "lại",
    "mọt", "hai", "ba", "rất", "cũng", "vẫn", "chỉ",
    "không", "phải", "biết", "thân", "hỏi",
    # common 2-letter words that look like abbreviations
    "bộ", "bạ", "mì", "tả", "tấ",
})

# Vietnamese vowels for heuristic detection
_VI_VOWELS: frozenset[str] = frozenset(
    "aeiouy"
    "àáảãạăắằẳẵặâấầẩẫậ"
    "èéẻẽẹêếềểễệ"
    "ìíỉĩị"
    "òóỏõọôốồổỗộơớờởỡợ"
    "ùúủũụưứừửữự"
    "ỳýỷỹỵ"
)


def _is_likely_abbreviation(word: str) -> bool:
    """Heuristic: trừ word có khả năng là viết tắt nếu:
    1. Toàn uppercase (BMNN, TTGT) → chắc chắn
    2. Lowercase nhưng không có ngườí âm âm (bmnn, ttgt) → có thể
    3. Không thuộc stop words tiếng Việt
    """
    if len(word) < 2:
        return False

    # All-uppercase: definitely abbreviation
    if word.isupper() and word.isalpha():
        return True

    # Stop words: never abbreviation
    if word.lower() in _VI_STOP_WORDS:
        return False

    # Pure-lowercase heuristic: low vowel ratio = likely abbreviation
    if word.islower():
        vowel_count = sum(1 for c in word if c in _VI_VOWELS)
        vowel_ratio = vowel_count / len(word)
        # < 20% vowels and 2-6 chars → likely abbreviation (bmnn=0%, ttgt=0%)
        if len(word) <= 6 and vowel_ratio < 0.20:
            return True

    return False


async def _expand_abbreviations_in_message(
    message: str,
) -> tuple[str, bool, list[str], dict[str, list]]:
    """
    Expand abbreviations found in message using smart heuristic detection.

    Returns:
        (expanded_message, was_modified, potential_abbreviations, multi_meaning_map)
        - expanded_message: message with single-meaning abbreviations expanded
        - was_modified: True if any expansion happened
        - potential_abbreviations: words that look like abbreviations but not in DB
        - multi_meaning_map: {abbr: [{full_form, description}, ...]} for disambiguation
    """
    import re

    # Extract candidate tokens using smart heuristic instead of greedy regex
    all_tokens = re.findall(r'\b([\w]{2,})\b', message)
    candidate_abbrs = [t for t in all_tokens if _is_likely_abbreviation(t)]

    if not candidate_abbrs:
        return message, False, [], {}

    # Remove duplicates while preserving order
    unique_abbrs = list(dict.fromkeys(candidate_abbrs))
    logger.debug(f"[abbr_expand] Candidate abbreviations: {unique_abbrs}")

    try:
        from app.services.agent.streaming import get_current_db
        from sqlalchemy import select, func
        from app.models.abbreviation import Abbreviation
        from collections import defaultdict

        db = get_current_db()
        if db is None:
            return message, False, [], {}

        # Single batch query for all candidates
        result = await db.execute(
            select(Abbreviation)
            .where(
                func.lower(Abbreviation.short_form).in_([a.lower() for a in unique_abbrs]),
                Abbreviation.is_active == True,
            )
        )
        all_abbrs_db = result.scalars().all()

        # Group by lowercase short_form
        abbr_map: dict[str, list] = defaultdict(list)
        for abbr_obj in all_abbrs_db:
            abbr_map[abbr_obj.short_form.lower()].append(abbr_obj)

        expanded_message = message
        any_expanded = False
        potential_abbreviations: list[str] = []
        multi_meaning_map: dict[str, list] = {}  # abbr → [{full_form, description}, ...]

        for abbr in unique_abbrs:
            all_matches = abbr_map.get(abbr.lower(), [])

            if len(all_matches) == 0:
                # Not in DB — flag as potential unknown abbreviation
                logger.debug(f"[abbr_expand] No DB entry for candidate: {abbr!r}")
                potential_abbreviations.append(abbr)

            elif len(all_matches) == 1:
                # Single meaning — safe to auto-expand
                abbr_obj = all_matches[0]
                pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
                new_message = pattern.sub(abbr_obj.full_form, expanded_message)
                if new_message != expanded_message:
                    logger.info(f"[abbr_expand] Auto-expand: {abbr!r} → {abbr_obj.full_form!r}")
                    expanded_message = new_message
                    any_expanded = True

            else:
                # Multiple meanings — store for LLM-based disambiguation
                multi_meaning_map[abbr] = [
                    {"full_form": m.full_form, "description": m.description or ""}
                    for m in all_matches
                ]
                short_forms = [f"{m.short_form}={m.full_form}" for m in all_matches]
                logger.info(
                    f"[abbr_expand] Multi-meaning for {abbr!r}: {' | '.join(short_forms)}"
                    f" — will disambiguate via LLM"
                )

        return expanded_message, any_expanded, potential_abbreviations, multi_meaning_map

    except Exception as e:
        logger.warning(f"[abbr_expand] Failed: {e}")
        return message, False, [], {}


async def _disambiguate_multi_meaning_abbrs(
    multi_meaning_map: dict[str, list],
    user_message: str,
) -> tuple[str, dict[str, str], list[str]]:
    """LLM-based disambiguation for abbreviations with multiple meanings.

    Uses memory agent (Qwen3-4B) to infer which meaning fits the context.

    Returns:
        (expanded_text_addition, chosen_map, low_confidence_abbrs)
        - expanded_text_addition: extra text to append to query with chosen expansions
        - chosen_map: {abbr: chosen_full_form} for high-confidence choices
        - low_confidence_abbrs: list of abbrs where LLM couldn't decide (need user clarification)
    """
    if not multi_meaning_map:
        return "", {}, []

    from app.services.llm import get_memory_agent
    from app.services.llm.types import LLMMessage as _LLMMsg

    chosen_map: dict[str, str] = {}
    low_confidence_abbrs: list[str] = []

    for abbr, meanings in multi_meaning_map.items():
        meanings_text = "\n".join(
            f"  {i+1}. {m['full_form']}" + (f" — {m['description']}" if m.get('description') else "")
            for i, m in enumerate(meanings)
        )
        prompt = (
            f'Từ viết tắt "{abbr}" có các nghĩa sau:\n{meanings_text}\n\n'
            f'Câu hỏi của user: "{user_message}"\n\n'
            f'Dựa vào ngữ cảnh câu hỏi, chọn nghĩa phù hợp nhất.\n'
            f'Nếu ngữ cảnh không đủ rõ để chọn, trả về confidence: "low".\n\n'
            f'Output JSON: {{"chosen": "<full_form>", "confidence": "high" or "low", "reasoning": "<1 sentence>"}}'
        )
        try:
            agent = get_memory_agent()
            resp_text = ""
            async for chunk in agent.astream(
                [_LLMMsg(role="user", content=prompt)],
                system_prompt="You are a Vietnamese abbreviation disambiguation assistant. Output valid JSON only.",
                temperature=0.0,
                max_tokens=120,
            ):
                if hasattr(chunk, "text") and chunk.text:
                    resp_text += chunk.text

            import json as _json
            # Strip markdown fences if present
            clean = resp_text.strip()
            if "```" in clean:
                clean = clean.split("```")[-2].strip() if clean.count("```") >= 2 else clean.replace("```json", "").replace("```", "").strip()
            result = _json.loads(clean)
            confidence = result.get("confidence", "low")
            chosen = result.get("chosen", "")
            reasoning = result.get("reasoning", "")

            if confidence == "high" and chosen:
                chosen_map[abbr] = chosen
                logger.info(
                    f"[disambig] {abbr!r} → {chosen!r} (high confidence): {reasoning}"
                )
            else:
                low_confidence_abbrs.append(abbr)
                logger.info(
                    f"[disambig] {abbr!r}: low confidence, will ask user. reasoning={reasoning!r}"
                )
        except Exception as e:
            logger.warning(f"[disambig] Failed for {abbr!r}: {e}")
            low_confidence_abbrs.append(abbr)

    # Build expansion text for chosen abbrs
    expansion_parts = [f"{abbr}={full}" for abbr, full in chosen_map.items()]
    expansion_text = "; ".join(expansion_parts) if expansion_parts else ""

    return expansion_text, chosen_map, low_confidence_abbrs




def _extract_user_message(state: SupervisorState) -> str:
    """Extract the last user message from state.messages.

    On abbreviation loop-back, prefer expanded_query so supervisor
    re-classifies using the full form (e.g., "an ninh mạng") instead of
    the original abbreviation (e.g., "ANM").
    """
    # expanded_query is set by rag_agent when abbreviation is found
    expanded = state.get("expanded_query", "")
    if expanded:
        return expanded

    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "")
        else:
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            if role == "human":
                role = "user"
        if role == "user":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            return content or ""
    return ""


def _parse_supervisor_response(raw: str) -> dict:
    """Parse LLM JSON response with fallbacks.

    Extracts task_plan (Phase 4) and computes pending_intent from it.
    After parsing, enforces intent→agent agreement: if the intent maps to
    a different agent than what the LLM chose, the intent-based mapping wins.
    """
    raw = raw.strip()

    # Strip thinking tags (Qwen3.x with thinking enabled)
    import re as _re_parse
    raw = _re_parse.sub(r'<think>.*?</think>', '', raw, flags=_re_parse.DOTALL).strip()

    # Strip markdown code fences
    if "```json" in raw:
        raw = raw.split("```json")[-1].split("```")[0].strip()
    elif "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1].strip()

    try:
        data = json.loads(raw)
        next_agent = data.get("next_agent", "finish")
        intent = data.get("intent", "search")
        task_plan = data.get("task_plan") or []

        # Normalize intent name (LLM sometimes uses shorthand like "search_phone")
        intent = _INTENT_NORMALIZE.get(intent, intent)

        # Validate next_agent
        valid_agents = {
            AgentType.RAG, AgentType.WRITE, AgentType.PEOPLE,
            AgentType.DIRECT, AgentType.FINISH,
        }
        if next_agent not in valid_agents:
            corrected = _INTENT_TO_AGENT_FALLBACK.get(next_agent)
            if corrected:
                logger.info(
                    f"[supervisor] Corrected next_agent '{next_agent}' → '{corrected}' "
                    f"(LLM returned intent name instead of agent name)"
                )
                next_agent = corrected
            else:
                logger.warning(
                    f"[supervisor] Invalid next_agent '{next_agent}', defaulting to 'finish'"
                )
                next_agent = AgentType.FINISH

        # ── Deterministic intent→agent override ──────────────────────────
        expected_agent = _INTENT_TO_AGENT_FALLBACK.get(intent)
        if (
            expected_agent
            and next_agent != AgentType.FINISH
            and expected_agent != next_agent
        ):
            logger.warning(
                f"[supervisor] Intent→agent mismatch: intent={intent!r} expects "
                f"{expected_agent!r} but LLM chose {next_agent!r} — overriding to {expected_agent!r}"
            )
            next_agent = expected_agent

        # ── Phase 4: Extract pending_intent from task_plan ───────────────
        # task_plan = ["resolve_doc", "search_section"]
        #   → intent = "resolve_doc" (first step)
        #   → pending_intent = "search_section" (final goal after prerequisite)
        pending_intent = None
        if task_plan and len(task_plan) > 1:
            # Validate: first step should match intent
            if task_plan[0] != intent:
                logger.info(
                    f"[supervisor] task_plan[0]={task_plan[0]!r} != intent={intent!r}"
                    f" — trusting task_plan[0]"
                )
                intent = task_plan[0]
                # Re-compute next_agent for corrected intent
                corrected = _INTENT_TO_AGENT_FALLBACK.get(intent)
                if corrected:
                    next_agent = corrected
            # Last step in plan = final goal
            pending_intent = task_plan[-1]
            logger.info(
                f"[supervisor] task_plan={task_plan}, "
                f"current_step={intent!r}, pending_intent={pending_intent!r}"
            )

        return {
            "next_agent": next_agent,
            "intent": intent,
            "task_plan": task_plan,
            "pending_intent": pending_intent,
            "reasoning": data.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        logger.warning(f"[supervisor] Failed to parse JSON: {raw[:200]!r}")
        return {
            "next_agent": AgentType.FINISH, "intent": "search",
            "task_plan": [], "pending_intent": None,
            "reasoning": "Parse failed, defaulting to finish",
        }


# =============================================================================
# Graph Nodes
# =============================================================================

async def supervisor_node(state: SupervisorState) -> dict:
    """
    Classify intent + decide next agent in ONE LLM call.

    This is the core supervisor logic:
    1. Extract user message
    2. Expand abbreviations in message
    3. Call LLM with supervisor prompt
    4. Parse response and update state
    """
    # get_llm_provider not needed here — supervisor uses OllamaLLMProvider directly

    user_message = _extract_user_message(state)
    if not user_message:
        return {
            "next_agent": AgentType.DIRECT,
            "intent": "greeting",
            "iterations": state.get("iterations", 0) + 1,
        }

    # Guard: max iterations
    from app.core.config import settings
    max_iter = getattr(settings, "NEXUSRAG_LG_MAX_ITERATIONS", 5)
    iterations = state.get("iterations", 0)

    # Special case: search_section loop back from rag — don't count against iterations
    # This happens when route_from_rag routes back to supervisor for search_section
    pending_section = state.get("section_reference")
    is_search_section_pending = (
        state.get("intent") == "search_section" and pending_section
    )
    if is_search_section_pending and iterations >= 1:
        # Still going — this is the supervisor re-entering from rag's route_from_rag
        # Just route directly to rag without incrementing iteration counter
        logger.info(f"[supervisor] search_section loop — re-routing to rag (iter={iterations})")
        return {
            "next_agent": AgentType.RAG,
            "intent": "search_section",
            "original_query": user_message,
            "rewritten_query": user_message,
            "iterations": iterations,  # Don't increment
            "should_loop_back": False,
            "section_reference": pending_section,
            "document_ids": state.get("document_ids") or [],
        }

    if iterations >= max_iter:
        logger.warning(f"[supervisor] Max iterations ({max_iter}) reached, forcing finish")
        return {"next_agent": AgentType.FINISH, "intent": state.get("intent", "search")}

    try:
        from app.services.llm.types import LLMMessage as _LLMMsg

        # On loop-back from abbreviation expansion, use expanded_query for re-classification
        expanded = state.get("expanded_query", "")
        query_for_classifier = expanded if expanded else user_message
        was_modified = False
        expanded_message = expanded

        # Expand abbreviations in message before classification
        # New: uses smart heuristic detection + returns multi_meaning_map for disambiguation
        potential_abbreviations = []
        multi_meaning_map: dict = {}
        clarification_needed = False
        clarification_message = ""
        if not expanded:
            expanded_message, was_modified, potential_abbreviations, multi_meaning_map = (
                await _expand_abbreviations_in_message(user_message)
            )
            if was_modified:
                logger.info(f"[supervisor] Abbreviations expanded: {user_message!r} -> {expanded_message!r}")
                query_for_classifier = expanded_message

            # Handle multi-meaning abbreviations: LLM tries to disambiguate from context
            if multi_meaning_map:
                _, chosen_map, low_confidence_abbrs = await _disambiguate_multi_meaning_abbrs(
                    multi_meaning_map, user_message
                )
                # Apply high-confidence choices to query
                if chosen_map:
                    import re as _re_abbr
                    for abbr, full_form in chosen_map.items():
                        pattern = _re_abbr.compile(r'\b' + _re_abbr.escape(abbr) + r'\b', _re_abbr.IGNORECASE)
                        new_q = pattern.sub(full_form, query_for_classifier)
                        if new_q != query_for_classifier:
                            logger.info(f"[supervisor] Disambiguated: {abbr!r} → {full_form!r}")
                            query_for_classifier = new_q
                            expanded_message = query_for_classifier
                            was_modified = True
                # Prepare clarification for low-confidence ones
                if low_confidence_abbrs:
                    clarification_needed = True
                    clarify_parts = []
                    for abbr in low_confidence_abbrs:
                        meanings = multi_meaning_map.get(abbr, [])
                        opts = "\n".join(f"  {i+1}. {m['full_form']}" for i, m in enumerate(meanings))
                        clarify_parts.append(f"**{abbr.upper()}** có thể là:\n{opts}")
                    clarification_message = (
                        "Vui lòng cho biết ý nghĩa của từ viết tắt bên dưới để tôi có thể trả lời chính xác hơn:\n\n"
                        + "\n\n".join(clarify_parts)
                        + "\n\nBạn muốn hỏi về nghĩa nào?"
                    )

            if potential_abbreviations:
                from app.services.agent.streaming import push_event
                await push_event(state, "potential_abbreviations", potential_abbreviations)


        # Phase 4: Use Qwen3.6-35B for plan-aware classification
        # We check the OLLAMA_HOST to decide whether to use native Ollama
        # or OpenAI-compatible provider (e.g. vLLM serving the 35B model).
        from app.core.config import settings
        if "/v1" in settings.OLLAMA_HOST:
            from app.services.llm.openai_compatible import OpenAICompatibleLLMProvider
            classifier = OpenAICompatibleLLMProvider(
                base_url=settings.OLLAMA_HOST,
                model=settings.OLLAMA_MODEL,
                api_key="none"
            )
        else:
            from app.services.llm.ollama import OllamaLLMProvider
            classifier = OllamaLLMProvider(
                host=settings.OLLAMA_HOST,
                model=settings.OLLAMA_MODEL,
            )
        response_text = ""

        async for chunk in classifier.astream(
            [_LLMMsg(role="user", content=query_for_classifier)],
            system_prompt=_SUPERVISOR_PROMPT.format(max_iterations=max_iter),
            temperature=0.0,
            max_tokens=512,
            think=False,  # Disable thinking to reduce latency for classification
        ):
            if hasattr(chunk, "text") and chunk.text:
                response_text += str(chunk.text)

        decision = _parse_supervisor_response(response_text)

        # ── Keyword safety net 1: direct/greeting → rag ─────────────────────
        # If the LLM classified as greeting/direct but the message contains
        # document-related keywords, override to rag + search.
        if (
            decision["next_agent"] == AgentType.DIRECT
            and decision["intent"] in ("greeting", "personal")
            and _should_force_rag(query_for_classifier)
            and not _is_pure_greeting(query_for_classifier)
        ):
            logger.warning(
                f"[supervisor] Keyword safety net: message contains document keywords "
                f"but LLM chose direct/{decision['intent']} — overriding to rag/search"
            )
            decision["next_agent"] = AgentType.RAG
            decision["intent"] = "search"
            decision["reasoning"] = f"(overridden by keyword safety net) {decision.get('reasoning', '')}"

        # ── Phase 4: Plan-aware prerequisite check ──────────────────────────
        # If the LLM didn't generate a proper task_plan but the query references
        # a named document and the intent requires document UUID, inject
        # resolve_doc as prerequisite. This is a fallback — with Qwen3.6-35B
        # the task_plan should usually be correct.
        _REQUIRES_DOC_INTENTS = {"summarize", "search_section"}
        task_plan = decision.get("task_plan") or []
        if (
            decision["intent"] in _REQUIRES_DOC_INTENTS
            and not state.get("document_ids")
            and (not task_plan or task_plan[0] != "resolve_doc")
        ):
            _NAMED_DOC_PATTERN = _re.compile(
                r"(?:lu\u1eadt|ngh\u1ecb\s+\u0111\u1ecbnh|th\u00f4ng\s+t\u01b0|quy\u1ebft\s+\u0111\u1ecbnh|ngh\u1ecb\s+quy\u1ebft|ph\u00e1p\s+l\u1ec7nh|b\u1ed9\s+lu\u1eadt)"
                r"\s+\S",
                _re.IGNORECASE | _re.UNICODE,
            )
            if _NAMED_DOC_PATTERN.search(query_for_classifier):
                original_intent = decision["intent"]
                decision["pending_intent"] = original_intent
                decision["intent"] = "resolve_doc"
                decision["next_agent"] = AgentType.RESOLVE_DOC
                decision["task_plan"] = ["resolve_doc", original_intent]
                logger.info(
                    f"[supervisor] Prerequisite check: injected resolve_doc before "
                    f"{original_intent!r}, task_plan={decision['task_plan']}"
                )

        logger.info(
            f"[LANGGRAPH_ROUTE] user_message={user_message!r} -> "
            f"next_agent={decision['next_agent']!r}, intent={decision['intent']!r}, "
            f"task_plan={decision.get('task_plan', [])!r}, "
            f"pending_intent={decision.get('pending_intent')!r}, "
            f"reasoning={decision.get('reasoning', '')!r}"
        )

        # Build return dict
        # Check if we're looping back with accumulated results - don't reset should_loop_back
        # This happens when rag_agent resolved documents and we need to go to answer_generator
        existing_should_loop = state.get("should_loop_back", False)
        had_results = bool(state.get("kg_summaries") or state.get("sources"))

        # If we have accumulated results from a previous rag loop, go to answer_generator
        if existing_should_loop and had_results:
            logger.info(f"[supervisor] Looping back with {len(state.get('kg_summaries', []))} kg_summaries, going to answer_generator")
            # Preserve document_ids from rag_agent result - needed for summarize in answer_generator
            preserved_doc_ids = state.get("document_ids") or state.get("resolved_document_ids") or []
            return {
                "next_agent": AgentType.ANSWER_GENERATOR,
                "intent": "summarize",  # Force summarize to trigger content fetch in answer_generator
                "original_query": user_message,
                "rewritten_query": state.get("rewritten_query", user_message),
                "iterations": iterations + 1,
                "should_loop_back": False,
                "document_ids": preserved_doc_ids,
                "potential_abbreviations": potential_abbreviations,
            }

        result = {
            "next_agent": decision["next_agent"],
            "intent": decision["intent"],
            "original_query": user_message,
            "rewritten_query": user_message,  # needed by tool functions
            "iterations": iterations + 1,
            # Reset loop flag on each supervisor entry
            "should_loop_back": False,
            "potential_abbreviations": potential_abbreviations,
            "clarification_needed": clarification_needed,
            "clarification_message": clarification_message,
            # Phase 4: Plan-aware fields
            "task_plan": decision.get("task_plan") or [],
            "pending_intent": decision.get("pending_intent"),
        }

        # Phase 3: Smart memory routing — determine if memory_recall is needed
        intent = decision["intent"]
        needs_memory = _query_needs_memory(intent, query_for_classifier)
        result["needs_memory"] = needs_memory
        if needs_memory:
            logger.info(
                f"[supervisor] needs_memory=True for intent={intent!r}, "
                f"query={query_for_classifier[:60]!r}"
            )
        else:
            logger.debug(f"[supervisor] needs_memory=False for intent={intent!r}")

        # Emit clarification event for low-confidence disambiguations
        if clarification_needed and clarification_message:
            from app.services.agent.streaming import push_event
            await push_event(state, "clarification", {"message": clarification_message})
            logger.info(f"[supervisor] Clarification needed for: {list(multi_meaning_map.keys())}")

        # Smart thinking decision: RAG tasks don't need thinking (just retrieval),
        # but complex synthesis with many sources or personal tasks benefit from it
        thinking_intents = {"greeting", "personal", "write_suggest_edits", "write_grammar_check"}
        rag_intents = {"search", "list_docs", "summarize", "kg_query", "search_doc_num", "resolve_doc", "search_section", "search_abbr"}
        intent = decision["intent"]

        # Check if user has personal memory from Graphiti - needs thinking to incorporate
        user_memory_context = state.get("user_memory_context", "")
        has_memory = bool(user_memory_context and "No relevant memories" not in user_memory_context)

        if intent in thinking_intents:
            result["enable_thinking"] = True
        elif has_memory:
            # Personal memory found - needs thinking to incorporate facts correctly
            result["enable_thinking"] = True
            logger.info(f"[supervisor] Thinking enabled for {intent} with personal memory")
        elif intent in rag_intents:
            # RAG tasks: check source count - many results benefit from thinking to synthesize
            source_count = len(state.get("sources", [])) + len(state.get("kg_summaries", []))
            if source_count >= 5:
                result["enable_thinking"] = True  # Complex synthesis needed
                logger.info(f"[supervisor] Thinking enabled for {intent} with {source_count} sources")
            else:
                result["enable_thinking"] = False  # Simple retrieval
        else:
            result["enable_thinking"] = False

        # Determine search_mode based on intent (Phase 1: Smart RAG Routing)
        # kg_query → kg only; search/summarize/search_doc_num → vector only; else → hybrid
        _KG_INTENTS = {"kg_query"}
        _VECTOR_INTENTS = {"search", "summarize", "search_doc_num", "list_docs", "search_section"}
        if intent in _KG_INTENTS:
            search_mode = "kg"
        elif intent in _VECTOR_INTENTS:
            search_mode = "vector"
        else:
            search_mode = "hybrid"  # resolve_doc, search_abbr, unknown
        result["search_mode"] = search_mode
        logger.info(f"[supervisor] search_mode={search_mode!r} for intent={intent!r}")

        # If abbreviations were expanded, include expanded_query for downstream nodes
        if was_modified:
            result["expanded_query"] = expanded_message
            result["rewritten_query"] = expanded_message

        return result

    except Exception as e:
        logger.error(f"[supervisor] LLM call failed: {e}")
        # Fail-safe: default to RAG (document search) instead of direct.
        # Direct gives empty answers; RAG at least attempts document retrieval.
        return {
            "next_agent": AgentType.RAG,
            "intent": state.get("intent", "search"),
            "iterations": iterations + 1,
        }


async def direct_answer_node(state: SupervisorState) -> dict:
    """
    Answer greetings/personal questions directly without document retrieval.

    Emits token events via push_event for SSE streaming compatibility.
    """
    logger.info(f"[LANGGRAPH_NODE] Entering direct_answer_node, intent={state.get('intent')!r}")
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage
    from app.services.agent.streaming import push_event

    provider = get_llm_provider()
    messages = state.get("messages", [])
    system_prompt = state.get("system_prompt", "")
    user_memory = state.get("user_memory_context", "")

    # Emit status
    await push_event(state, "status", {"step": "generating", "detail": "Đang trả lời..."})

    # Build LLM messages
    llm_messages: list[LLMMessage] = []
    for msg in (messages or [])[-6:]:
        if isinstance(msg, dict):
            role, content = msg.get("role", "user"), msg.get("content", "")
        else:
            role, content = getattr(msg, "role", "user"), getattr(msg, "content", "")
            
        # Truncate content if it's an assistant message to save context window
        if role == "assistant" and len(content) > 500:
            content = content[:500] + "...\n[Nội dung đã được rút gọn]"
            
        llm_messages.append(LLMMessage(role=role, content=content))

    # Inject memory if available
    effective_system = system_prompt
    if user_memory and "No relevant memories" not in user_memory:
        effective_system = (
            f"{user_memory}\n\n"
            "IMPORTANT: Do NOT copy these facts directly. When using a memory fact, "
            "paraphrase it in your own words and cite it as [MEM-1], [MEM-2], etc. "
            "For example: 'The user works at Công an tỉnh Hà Tĩnh [MEM-1]' instead of copying the fact verbatim. "
            "Only include relevant memories.\n\n"
        ) + effective_system

    answer_parts = []
    try:
        async for chunk in provider.astream(
            messages=llm_messages,
            temperature=0.5,
            max_tokens=2048,
            system_prompt=effective_system,
            think=False,
        ):
            if chunk.text:
                answer_parts.append(chunk.text)
                # Emit token for SSE streaming
                await push_event(state, "token", chunk.text)
    except Exception as e:
        logger.error(f"[direct_answer] LLM streaming failed: {e}")
        answer_parts = ["Xin chào! Tôi có thể giúp gì cho bạn?"]
        await push_event(state, "token", answer_parts[0])

    # Import strip_thinking_tags from nodes
    from app.services.agent.nodes import strip_thinking_tags
    final_answer = strip_thinking_tags("".join(answer_parts))

    return {"final_answer": final_answer, "next_agent": AgentType.FINISH}


# =============================================================================
# Answer Generator Node — RAG-only
# =============================================================================

async def answer_generator_node(state: SupervisorState) -> dict:
    """
    Generate final answer from accumulated RAG sources/context.
    Wraps the answer_generator from nodes.py for supervisor graph.

    This node is now RAG-only — write/direct agents bypass it entirely,
    and mongo has its own formatter node. This eliminates the "God Node"
    pattern where one node handled all intent types.

    Uses merge pattern over DEFAULT_STATE to avoid manual field-by-field copy.
    """
    from app.services.agent.nodes import answer_generator as _orig_ag
    from app.services.agent.state import DEFAULT_STATE

    intent = state.get("intent", "")
    logger.info(f"[LANGGRAPH_NODE] Entering answer_generator_node, intent={intent!r}")

    # Merge SupervisorState over DEFAULT_STATE — keys present in state win.
    agent_state = {**DEFAULT_STATE}
    for k, v in state.items():
        if v is not None or k not in DEFAULT_STATE:
            agent_state[k] = v

    # Override control fields
    agent_state["tool_called"] = True
    agent_state["existing_citation_ids"] = {}
    agent_state["citation_map"] = {}

    result = await _orig_ag(agent_state)

    return {
        "final_answer": result.get("final_answer", ""),
        "next_agent": AgentType.FINISH,
    }


# =============================================================================
# Mongo Formatter Node — People search only
# =============================================================================

async def mongo_formatter_node(state: SupervisorState) -> dict:
    """
    Format MongoDB people search results via a lightweight LLM call.

    Uses a small, focused prompt (~1KB) instead of the full RAG system prompt
    (~13KB). This saves ~80% tokens compared to routing through answer_generator.
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage as _LLMMsg
    from app.services.agent.streaming import push_event
    from app.services.agent.nodes import strip_thinking_tags

    logger.info(f"[LANGGRAPH_NODE] Entering mongo_formatter_node")

    existing_final = state.get("final_answer", "")
    if not existing_final:
        return {
            "final_answer": "Không tìm thấy dữ liệu.",
            "next_agent": AgentType.FINISH,
        }

    await push_event(state, "status", {"step": "generating", "detail": "Đang trình bày kết quả..."})

    # Focused prompt — only mongo formatting rules
    format_system = (
        "Bạn là một trợ lý truy vấn cơ sở dữ liệu.\n"
        "Nhiệm vụ: Đọc dữ liệu hồ sơ người dân bên dưới và trình bày lại "
        "NGẮN GỌN, SẠCH SẼ, DỄ ĐỌC bằng TIẾNG VIỆT.\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. Mỗi hồ sơ = 1 KHỐI riêng biệt, bắt đầu bởi 1., 2., 3., ...\n"
        "2. Mỗi khối cách nhau bằng MỘT DÒNG TRỐNG.\n"
        "3. Dòng đầu là tiêu đề (in đậm).\n"
        "4. Mỗi thông tin nằm trên 1 dòng riêng, có dấu ';'.\n"
        "5. KHÔNG viết nhiều thông tin trên cùng một dòng.\n"
        "6. KHÔNG dùng ký hiệu [xxx] hay ObjectId trong câu trả lời.\n"
        "7. Bỏ qua field rỗng/null.\n"
        "8. Nếu không có kết quả, nói rõ 'Không tìm thấy'.\n"
    )
    format_user = (
        f"Dữ liệu truy vấn:\n{existing_final}\n\n"
        "Hãy trình bày lại dữ liệu dễ đọc cho người dùng."
    )

    provider = get_llm_provider()
    mongo_messages = [
        _LLMMsg(role="system", content=format_system),
        _LLMMsg(role="user", content=format_user),
    ]

    try:
        answer_parts: list[str] = []
        async for chunk in provider.astream(
            messages=mongo_messages, temperature=0.1, max_tokens=4096,
        ):
            if chunk.type == "text" and chunk.text:
                await push_event(state, "token", chunk.text)
                answer_parts.append(chunk.text)
        final = strip_thinking_tags("".join(answer_parts))
        return {"final_answer": final, "next_agent": AgentType.FINISH}
    except Exception as e:
        logger.error(f"[mongo_formatter_node] LLM format failed: {e} — using raw")
        await push_event(state, "token", existing_final)
        return {"final_answer": existing_final, "next_agent": AgentType.FINISH}


# =============================================================================
# Routing Functions
# =============================================================================

def route_from_supervisor(state: SupervisorState) -> str:
    """Route to appropriate agent based on supervisor's decision.

    Phase 3: Uses needs_memory flag (set by supervisor_node via _query_needs_memory)
    to decide whether to go through memory_recall → query_enricher first, or
    bypass directly to the target agent.

    Handles the case where supervisor returns next_agent=None on loop back,
    meaning we're done and should END.
    """
    next_agent = state.get("next_agent")
    intent = state.get("intent", "")
    needs_memory = state.get("needs_memory", False)

    if next_agent is None or next_agent == AgentType.FINISH:
        return END

    # Phase 2: resolve_doc routes directly to its dedicated agent (no memory needed)
    if next_agent == AgentType.RESOLVE_DOC or (
        next_agent == AgentType.RAG and intent == "resolve_doc"
    ):
        return "resolve_doc_agent"

    # Phase 3: Smart memory routing
    # needs_memory=True  → memory_recall → query_enricher → target agent
    # needs_memory=False → direct to target agent (bypass Graphiti entirely)
    if needs_memory:
        logger.info(
            f"[route_from_supervisor] needs_memory=True (intent={intent!r}) "
            f"→ memory_recall"
        )
        return "memory_recall"

    # Direct bypass: map AgentType → node name
    _DIRECT_MAP: dict[str, str] = {
        AgentType.RAG: "rag",
        AgentType.WRITE: "write",
        AgentType.DIRECT: "direct",
        AgentType.PEOPLE: "people",
        AgentType.ANSWER_GENERATOR: "answer_generator",
    }
    target = _DIRECT_MAP.get(next_agent)
    if target:
        logger.info(
            f"[route_from_supervisor] needs_memory=False (intent={intent!r}) "
            f"→ direct to {target!r}"
        )
        return target

    # Fallback: go through memory_recall (safe default)
    return "memory_recall"


def route_from_rag(state: SupervisorState) -> str:
    """Route after rag agent completes.

    If abbreviation was found and expanded, loop back to supervisor
    to re-classify with the full form. Otherwise go to answer_generator.

    For search_section intent, we check section_reference:
    - If section_reference is still in state → search_section tool NOT yet executed
      (supervisor set it but tool hasn't run yet) → route back to supervisor
      (supervisor will re-route to rag with same intent, and rag will execute tool)
    - If section_reference is None/empty → tool already ran, kg_summaries has content
      → go to answer_generator
    """
    pending_section = state.get("section_reference")
    intent = state.get("intent")
    loop = state.get("should_loop_back")
    logger.info(f"[route_from_rag] intent={intent!r}, section_reference={pending_section!r}, should_loop_back={loop}")
    if loop:
        return "supervisor"
    if intent == "search_section":
        if pending_section:
            # section_reference still present → tool not yet executed
            # Loop back to supervisor so it re-routes to rag
            return "supervisor"
        return "answer_generator"
    return "answer_generator"


def route_from_resolve_doc(state: SupervisorState) -> str:
    """Route after resolve_doc_agent completes.

    Phase 4 (Plan-Aware): Uses pending_intent from task_plan to determine
    what to do after document resolution.

    Priority:
    1. next_agent=FINISH → END (ambiguous/not-found, already streamed answer)
    2. intent='search_section' and section_reference → rag (search_section tool)
    3. pending_intent='search_section' and section_reference → rag
    4. pending_intent='summarize' → answer_generator
    5. pending_intent='search' → rag (general search within resolved doc)
    6. Default → answer_generator
    """
    next_agent = state.get("next_agent")
    intent = state.get("intent", "")
    section_ref = state.get("section_reference", "")
    pending_intent = state.get("pending_intent")

    if next_agent == AgentType.FINISH:
        logger.info("[route_from_resolve_doc] next_agent=FINISH → END (not found / ambiguous)")
        return END

    # resolve_doc_agent already set intent=search_section when section_ref found
    if intent == "search_section" and section_ref:
        logger.info(f"[route_from_resolve_doc] has section_ref={section_ref!r} → rag")
        return "rag"

    # Phase 4: Check pending_intent from task_plan
    if pending_intent:
        logger.info(
            f"[route_from_resolve_doc] pending_intent={pending_intent!r}, "
            f"section_ref={section_ref!r}"
        )
        if pending_intent == "search_section" and section_ref:
            return "rag"
        elif pending_intent == "search":
            return "rag"
        elif pending_intent == "summarize":
            return "answer_generator"

    logger.info("[route_from_resolve_doc] document resolved → answer_generator")
    return "answer_generator"


# shouldContinue_from_supervisor removed — was dead code (never called).


# =============================================================================
# Graph Builder
# =============================================================================

def create_supervisor_graph():
    """
    Build and compile the supervisor-based multi-agent graph.

    Phase 3 Flow (Smart Memory):
        START → supervisor
                  ├── [needs_memory=True] → memory_recall → query_enricher
                  │     ├── [personal] → direct
                  │     ├── [search/summarize/...] → rag → answer_generator
                  │     ├── [write] → write → END
                  │     └── [people] → people → mongo_formatter → END
                  │
                  └── [needs_memory=False] → target agent (direct bypass)
                        ├── rag → answer_generator → END
                        ├── write → END
                        ├── direct → END
                        └── people → mongo_formatter → END

    resolve_doc always bypasses memory (no personal context needed).
    """
    graph = StateGraph(SupervisorState)

    # Nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("memory_recall", _memory_recall_wrapper)
    graph.add_node("query_enricher", _query_enricher_wrapper)   # Phase 3: NEW
    graph.add_node("rag", _rag_agent_wrapper)
    graph.add_node("resolve_doc_agent", _resolve_doc_agent_wrapper)  # Phase 2
    graph.add_node("write", _write_agent_wrapper)
    graph.add_node("people", _people_agent_wrapper)
    graph.add_node("direct", direct_answer_node)
    graph.add_node("answer_generator", answer_generator_node)  # RAG-only
    graph.add_node("mongo_formatter", mongo_formatter_node)    # People-only

    # Edges
    graph.add_edge(START, "supervisor")

    # Phase 3: Conditional edges from supervisor based on needs_memory flag
    # - needs_memory=True  → "memory_recall" (→ query_enricher → target)
    # - needs_memory=False → target node directly (bypass Graphiti)
    # - resolve_doc always → "resolve_doc_agent"
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            # Memory path (personal reference detected)
            "memory_recall": "memory_recall",
            # Direct bypass paths (no personal context needed)
            "rag": "rag",
            "write": "write",
            "direct": "direct",
            "people": "people",
            # Special cases
            "answer_generator": "answer_generator",  # loop-back with accumulated results
            "resolve_doc_agent": "resolve_doc_agent",  # Phase 2: dedicated agent
            END: END,
        },
    )

    # Phase 3: memory_recall → query_enricher (always: enricher is a no-op when not needed)
    graph.add_edge("memory_recall", "query_enricher")

    # Phase 3: query_enricher → target agent based on next_agent / intent
    def route_from_enricher(state: SupervisorState) -> str:
        """Route after query enrichment to the correct agent.

        personal intent → direct (answer from memory, no RAG needed)
        all other intents → their designated agent
        """
        next_agent = state.get("next_agent", AgentType.FINISH)
        intent = state.get("intent", "")

        if intent == "personal":
            return "direct"

        _MAP = {
            AgentType.RAG: "rag",
            AgentType.WRITE: "write",
            AgentType.DIRECT: "direct",
            AgentType.PEOPLE: "people",
        }
        target = _MAP.get(next_agent, "direct")
        logger.info(
            f"[route_from_enricher] intent={intent!r}, next_agent={next_agent!r} → {target!r}"
        )
        return target

    graph.add_conditional_edges(
        "query_enricher",
        route_from_enricher,
        {
            "rag": "rag",
            "write": "write",
            "direct": "direct",
            "people": "people",
        },
    )

    # Intent-specific output pipelines:
    # - RAG → answer_generator (needs LLM to compose answer from sources)
    # - resolve_doc_agent → answer_generator / rag (section search) / END (ambiguous)
    # - Write → END (write_agent already produces final_answer via streaming)
    # - People → mongo_formatter (lightweight LLM call, ~1KB prompt)
    # - Direct → END (direct_answer_node already produces final_answer via streaming)
    graph.add_conditional_edges("rag", route_from_rag, {"supervisor": "supervisor", "answer_generator": "answer_generator"})
    graph.add_conditional_edges(
        "resolve_doc_agent",
        route_from_resolve_doc,
        {
            "answer_generator": "answer_generator",
            "rag": "rag",
            END: END,
        },
    )
    graph.add_edge("write", END)
    graph.add_edge("people", "mongo_formatter")
    graph.add_edge("direct", END)

    # Terminal nodes
    graph.add_edge("answer_generator", END)
    graph.add_edge("mongo_formatter", END)

    return graph.compile()


# =============================================================================
# Singleton - cached compiled graph (same pattern as agent/graph.py)
# =============================================================================

import threading as _threading

_supervisor_graph = None
_supervisor_graph_lock = _threading.Lock()


def get_supervisor_graph():
    """Return cached compiled supervisor graph.

    Thread-safe double-checked locking: avoids redundant graph builds
    when multiple coroutines start simultaneously.
    """
    global _supervisor_graph
    if _supervisor_graph is None:
        with _supervisor_graph_lock:
            if _supervisor_graph is None:
                _supervisor_graph = create_supervisor_graph()
    return _supervisor_graph


def reset_supervisor_graph():
    """Force the singleton to rebuild on next call (e.g. after hot-reload)."""
    global _supervisor_graph
    with _supervisor_graph_lock:
        _supervisor_graph = None


# =============================================================================
# Agent Wrappers (imported from new agent files)
# =============================================================================

async def _memory_recall_wrapper(state: SupervisorState) -> dict:
    """
    Load user memories from Graphiti into SupervisorState.user_memory_context.
    Called before direct/write/rag so every agent has access to personal memory.
    """
    import uuid as _uuid

    user_id = state.get("user_id")
    if not user_id:
        return {}

    user_message = state.get("rewritten_query") or state.get("original_query", "")
    if not user_message:
        return {}

    try:
        from app.services.graphiti_client import search_user_memory

        # user_id in state may be int or UUID depending on how it was passed
        uid = user_id
        if isinstance(uid, int):
            uid = _uuid.UUID(int=uid)
        elif isinstance(uid, str):
            uid = _uuid.UUID(uid)

        memory = await search_user_memory(uid, user_message, top_k=5)
        if memory:
            logger.info(f"[memory_recall_wrapper] Graphiti injected {len(memory)} chars")
            return {"user_memory_context": memory}
        else:
            logger.info(f"[memory_recall_wrapper] Graphiti found no relevant memory for user_id={uid}")
    except Exception as e:
        logger.warning(f"[memory_recall_wrapper] failed: {e}")

    return {}


async def _query_enricher_wrapper(state: SupervisorState) -> dict:
    """Phase 3: Rewrite query by replacing personal references with concrete info from memory.

    Runs after memory_recall. If memory context contains facts about the user
    (e.g. "Bạn công tác tại Công an tỉnh Hà Tĩnh"), rewrites the query like:
      "Đơn vị tôi có cần tuân thủ...?" → "Công an tỉnh Hà Tĩnh có cần tuân thủ...?"

    This ensures RAG search uses concrete terms rather than personal pronouns,
    yielding more accurate retrieval results for personal+RAG hybrid queries.
    For intent=personal (no RAG needed), this is a no-op and direct_answer_node
    uses memory from user_memory_context directly.
    """
    memory = state.get("user_memory_context", "")
    query = state.get("rewritten_query", "") or state.get("original_query", "")
    intent = state.get("intent", "")

    # personal intent → direct_answer_node handles memory directly, no rewrite needed
    if intent == "personal":
        return {}

    # Nothing to enrich if no memory or no personal reference in query
    if not memory or not query or "No relevant memories" in memory:
        return {}
    if not _PERSONAL_REF_PATTERN.search(query):
        return {}

    # Extract fact lines from memory context (format: "[Memory]\n- fact\n- fact")
    facts = [
        line.lstrip("- ").strip()
        for line in memory.split("\n")
        if line.strip().startswith("-")
    ]
    if not facts:
        return {}

    try:
        from app.services.llm import get_memory_agent
        from app.services.llm.types import LLMMessage as _LLMMsg
        import re as _re_enrich

        agent = get_memory_agent()
        facts_text = "\n".join(f"- {f}" for f in facts)
        prompt = (
            f"FACTS about the user:\n{facts_text}\n\n"
            f"Rewrite the query below, replacing personal pronouns and references "
            f"(tôi, của tôi, đơn vị tôi, cơ quan tôi, chúng tôi, etc.) with the "
            f"CONCRETE information from the FACTS above.\n"
            f"If no replacement is possible or needed, output the query unchanged.\n"
            f"Output ONLY the rewritten query, nothing else.\n\n"
            f"Query: {query}"
        )

        response = ""
        async for chunk in agent.astream(
            [_LLMMsg(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=128,
        ):
            if hasattr(chunk, "text") and chunk.text:
                response += chunk.text

        # Strip <think>...</think> tags that Qwen3 may emit
        response = _re_enrich.sub(
            r"<think>.*?</think>", "", response, flags=_re_enrich.DOTALL
        ).strip()

        enriched = response.strip()
        if enriched and enriched != query:
            logger.info(
                f"[query_enricher] Enriched: {query!r} → {enriched!r}"
            )
            return {"rewritten_query": enriched}
        else:
            logger.debug(f"[query_enricher] No enrichment applied for: {query!r}")

    except Exception as e:
        logger.warning(f"[query_enricher] Failed: {e}")

    return {}


async def _rag_agent_wrapper(state: SupervisorState) -> dict:
    """Wrapper that imports and calls rag_agent_node."""
    from app.services.agents.rag_agent import rag_agent_node

    return await rag_agent_node(state)


async def _resolve_doc_agent_wrapper(state: SupervisorState) -> dict:
    """Phase 2: Wrapper that imports and calls resolve_doc_agent_node."""
    from app.services.agents.resolve_doc_agent import resolve_doc_agent_node

    return await resolve_doc_agent_node(state)


async def _write_agent_wrapper(state: SupervisorState) -> dict:
    """Wrapper that imports and calls write_agent_node."""
    from app.services.agents.write_agent import write_agent_node

    return await write_agent_node(state)


async def _people_agent_wrapper(state: SupervisorState) -> dict:
    """Wrapper that imports and calls people_agent_node."""
    from app.services.agents.people_agent import people_agent_node

    return await people_agent_node(state)


# =============================================================================
# Backward Compatibility - Re-export for existing imports
# =============================================================================

# Keep old SupervisorState name for compatibility during transition
SupervisorStateModel = SupervisorState
create_supervisor = create_supervisor_graph

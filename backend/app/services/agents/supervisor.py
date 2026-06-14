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
import re
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END

from app.services.agents.models import (
    SupervisorState,
    AgentType,
    Intent,
)
def _get_langfuse_client():
    from app.services.agent.langfuse_tracing import _get_langfuse_client as get_client
    return get_client()


def _react_on() -> bool:
    """Whether the RAG group is served by the ReAct executor (tool-aware planning)."""
    from app.core.config import settings
    return bool(getattr(settings, "NEXUSRAG_LG_RAG_REACT", False))

if TYPE_CHECKING:
    from app.services.llm.types import LLMMessage

logger = logging.getLogger(__name__)

# =============================================================================
# Supervisor Prompt
# =============================================================================

from app.prompts.agents.supervisor_prompt import _SUPERVISOR_PROMPT

# Intent abbreviation → canonical name (safety net for LLM shortcutting intent names)
# P0-6: Expanded to cover more observed LLM shorthand patterns.
_INTENT_NORMALIZE: dict[str, str] = {
    # People: search_X ↔ X_search variants
    "search_phone": "mongo_search_phone",
    "search_cccd": "mongo_search_cccd",
    "search_name": "mongo_search_name",
    "search_bhxh": "mongo_search_bhxh",
    "phone_search": "mongo_search_phone",
    "cccd_search": "mongo_search_cccd",
    "name_search": "mongo_search_name",
    "bhxh_search": "mongo_search_bhxh",
    "search_advanced": "mongo_search_advanced",
    "advanced_search": "mongo_search_advanced",
    # People: LLM drops the "mongo_" prefix entirely
    "find_phone": "mongo_search_phone",
    "find_person": "mongo_search_name",
    "search_person": "mongo_search_name",
    "find_by_name": "mongo_search_name",
    "find_by_phone": "mongo_search_phone",
    "find_by_cccd": "mongo_search_cccd",
    # RAG: doc resolution variants
    "query_doc": "resolve_doc",
    "find_doc": "resolve_doc",
    "lookup_doc": "resolve_doc",
    "resolve_document": "resolve_doc",
    # RAG: section variants
    "lookup_section": "search_section",
    "get_article": "search_section",
    "find_article": "search_section",
    "search_article": "search_section",
    "get_section": "search_section",
    # RAG: summarize variants
    "summarize_doc": "summarize",
    "doc_summary": "summarize",
}

# Intent → Agent routing table used as fallback inside _parse_supervisor_response.
# Uses Intent.* and AgentType.* constants (P0-5) so renames stay in sync.
_INTENT_TO_AGENT_FALLBACK: dict[str, str] = {
    Intent.GREETING:           AgentType.DIRECT,
    Intent.PERSONAL:           AgentType.DIRECT,
    Intent.SEARCH:             AgentType.RAG,
    Intent.LIST_DOCS:          AgentType.RAG,
    Intent.SUMMARIZE:          AgentType.RAG,
    Intent.KG_QUERY:           AgentType.RAG,
    Intent.SEARCH_DOC_NUM:     AgentType.RAG,
    Intent.SEARCH_ABBR:        AgentType.RAG,
    Intent.SEARCH_SECTION:     AgentType.RAG,
    Intent.RESOLVE_DOC:        AgentType.RESOLVE_DOC,  # Phase 2: dedicated agent
    Intent.WRITE_SUMMARIZE:    AgentType.WRITE,
    Intent.WRITE_SUGGEST_EDITS: AgentType.WRITE,
    Intent.WRITE_GRAMMAR_CHECK: AgentType.WRITE,
    Intent.WRITE_FORMAT_CHECK:  AgentType.WRITE,
    Intent.MONGO_SEARCH_CCCD:   AgentType.PEOPLE,
    Intent.MONGO_SEARCH_NAME:   AgentType.PEOPLE,
    Intent.MONGO_SEARCH_BHXH:   AgentType.PEOPLE,
    Intent.MONGO_SEARCH_PHONE:  AgentType.PEOPLE,
    Intent.MONGO_SEARCH_ADVANCED: AgentType.PEOPLE,
}


# =============================================================================
# Module-level compiled regex patterns (P1-7: avoid recompile per call)
# =============================================================================

# Detects personal pronouns like "tôi", "đơn vị tôi", "của tôi", "chỗ tôi"…
# Used as safety net for needs_memory — fires when LLM misses personal context.
_PERSONAL_REF_PATTERN: re.Pattern[str] = re.compile(
    r"(?<!\w)(tôi|của\s+tôi|cho\s+tôi|đơn\s+vị\s+tôi|cơ\s+quan\s+tôi|nơi\s+tôi|chỗ\s+tôi|chúng\s+tôi|của\s+chúng\s+tôi|công\s+tác\s+của\s+tôi|làm\s+việc\s+của\s+tôi|tôi\s+tên|tên\s+(của\s+)?tôi|tôi\s+là\s+ai|tôi\s+làm\s+việc|tôi\s+công\s+tác|tôi\s+đang\s\sở)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

# Detects comparison/capability questions: "X của tôi có thể Y không?"
# Used by query_analyzer_node to set needs_comparison=True.
_COMPARISON_PATTERN: re.Pattern[str] = re.compile(
    r"(?:"
    r"có\s+(?:thể|đủ|đáp\s+ứng|phù\s+hợp|dùng\s+để|sử\s+dụng\s+để|thực\s+hiện)"
    r"|đủ\s+(?:điều\s+kiện|tiêu\s+chuẩn|yêu\s+cầu)"
    r"|sử\s+dụng\s+được"
    r"|dùng\s+để"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Detects references to a NAMED legal document (used by _REQUIRES_DOC_INTENTS
# prerequisite check). Covers both full names ("Luật An ninh mạng") and
# abbreviated forms ("NĐ 13", "TT 15", "QĐ 53").
# See also: resolve_doc_agent._DOC_TYPE_KEYWORDS — should be unified in P1-6.
_NAMED_DOC_PATTERN: re.Pattern[str] = re.compile(
    r"(?:"
    r"luật|nghị\s+định|thông\s+tư|quyết\s+định|nghị\s+quyết|pháp\s+lệnh|bộ\s+luật"
    r"|nđ|tt|qđ|nq|cp|ttlt"
    r")\s+\S",
    re.IGNORECASE | re.UNICODE,
)


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
    OPTIMIZED: Batches ALL abbreviations into a single LLM call instead of
    calling the LLM once per abbreviation (saves ~1s × N sequential calls).

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
    from app.prompts.agents.disambiguation_prompt import (
        build_batch_disambiguation_prompt,
    )

    chosen_map: dict[str, str] = {}
    low_confidence_abbrs: list[str] = []

    # Build a single prompt for ALL abbreviations at once
    sys_prompt, user_prompt = build_batch_disambiguation_prompt(
        multi_meaning_map, user_message
    )

    try:
        agent = get_memory_agent()
        resp_text = ""
        async for chunk in agent.astream(
            [_LLMMsg(role="user", content=user_prompt)],
            system_prompt=sys_prompt,
            temperature=0.0,
            max_tokens=256,
        ):
            if hasattr(chunk, "text") and chunk.text:
                resp_text += chunk.text

        import json as _json
        # Strip thinking tags and markdown fences
        clean = resp_text.strip()
        import re as _re_disamb
        clean = _re_disamb.sub(r'<think>.*?</think>', '', clean, flags=_re_disamb.DOTALL).strip()
        if "```" in clean:
            clean = clean.split("```")[-2].strip() if clean.count("```") >= 2 else clean.replace("```json", "").replace("```", "").strip()

        result = _json.loads(clean)
        # Result format: {"results": [{"abbr": ..., "chosen": ..., "confidence": ..., "reasoning": ...}, ...]}
        results_list = result.get("results", [])

        for item in results_list:
            abbr = item.get("abbr", "")
            confidence = item.get("confidence", "low")
            chosen = item.get("chosen", "")
            reasoning = item.get("reasoning", "")

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
        logger.warning(f"[disambig] Batch disambiguation failed: {e}")
        # Fallback: mark all as low confidence
        low_confidence_abbrs.extend(multi_meaning_map.keys())

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
        needs_memory = data.get("needs_memory", False)
        is_legal_query = data.get("is_legal_query", False)
        # mentions_specific_doc removed (P0-3 — was dead code; never used in routing).
        # Langfuse can derive "named doc referenced" from task_plan.includes("resolve_doc").

        # Normalize intent name (LLM sometimes uses shorthand like "search_phone")
        intent = _INTENT_NORMALIZE.get(intent, intent)

        # Validate next_agent
        valid_agents = {
            AgentType.RAG, AgentType.WRITE, AgentType.PEOPLE,
            AgentType.DIRECT, AgentType.FINISH, AgentType.RESOLVE_DOC,
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
            "needs_memory": needs_memory,
            "is_legal_query": is_legal_query,
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


# Heuristic keywords that hint at multi-step / complex queries
_COMPLEX_QUERY_KEYWORDS: frozenset[str] = frozenset({
    "so sánh", "khác nhau", "giống nhau", "phân biệt",
    "liệt kê", "tất cả", "từng",
    "và", "với", "cùng",  # conjunctions linking multiple entities
})

# Patterns that suggest a comparison or multi-doc query
_MULTI_DOC_PATTERN: re.Pattern[str] = re.compile(
    r"(?:"
    r"so\s+sánh|khác\s+(?:nhau|gì|biệt)|giống\s+nhau"
    r"|(?:luật|nghị\s+định|thông\s+tư|nđ|tt|qđ).*?(?:và|với|so\s+với).*?(?:luật|nghị\s+định|thông\s+tư|nđ|tt|qđ)"
    r")",
    re.IGNORECASE | re.UNICODE,
)


async def query_analyzer_node(state: SupervisorState) -> dict:
    """Analyze user query to extract structured metadata BEFORE supervisor routing.

    Runs as the FIRST node in the graph.  Output informs supervisor_node:
    - sub_queries  : decomposed questions for multi-step execution
    - extracted_params : doc names, sections, IDs for precise tool invocation
    - query_complexity : determines if single-agent or multi-agent needed

    OPTIMIZATION: Uses lightweight heuristic to fast-path simple queries
    (skips LLM call, saves ~1-2s). Only invokes LLM for queries that
    heuristically look complex (comparisons, multi-doc references).
    """
    user_message = _extract_user_message(state)
    logger.info(f"[LANGGRAPH_NODE] Entering query_analyzer_node with query: {user_message!r}")
    if not user_message or len(user_message.strip()) < 10:
        # Very short messages (greetings) don't need analysis
        return {"query_complexity": "simple", "sub_queries": None, "extracted_params": None}

    # Skip analysis on loop-back (already analyzed on first pass)
    if state.get("iterations", 0) >= 1:
        return {}

    # ── Fast-path heuristic: skip LLM for simple queries ─────────────────
    # Only call the LLM analyzer if the query looks like it could be
    # multi-step or comparative. This saves ~1-2s for ~80% of queries.
    msg_lower = user_message.lower()
    looks_complex = _MULTI_DOC_PATTERN.search(msg_lower) is not None

    # ── Comparison detection: "X của tôi có thể Y không?" ─────────────
    # If query has comparison pattern AND personal reference, set needs_comparison=True.
    # This ensures answer_generator will COMPAREd user context vs. doc requirements.
    has_comparison = _COMPARISON_PATTERN.search(msg_lower) is not None
    has_personal = _PERSONAL_REF_PATTERN.search(user_message) is not None
    needs_comparison = has_comparison and has_personal

    if needs_comparison:
        logger.info(
            f"[query_analyzer] Comparison query detected: "
            f"has_comparison={has_comparison}, has_personal={has_personal}"
        )

    if not looks_complex and not needs_comparison:
        logger.info("[query_analyzer] Fast-path: simple query (no complex heuristic match), skipping LLM")
        return {"query_complexity": "simple", "sub_queries": None, "extracted_params": None, "needs_comparison": False}

    logger.info("[query_analyzer] Complex heuristic matched — invoking LLM for detailed analysis")

    try:
        from app.services.llm.types import LLMMessage as _LLMMsg
        from app.prompts.agents.query_analyzer_prompt import _QUERY_ANALYZER_PROMPT
        from app.services.llm import get_memory_agent

        # Use the memory agent (Qwen-memory 4B / Qwen3-4B) for complex structured extraction
        analyzer_llm = get_memory_agent()

        response_text = ""
        async for chunk in analyzer_llm.astream(
            [_LLMMsg(role="user", content=user_message)],
            system_prompt=_QUERY_ANALYZER_PROMPT,
            temperature=0.0,
            max_tokens=512,
            think=False,
        ):
            if hasattr(chunk, "text") and chunk.text:
                response_text += str(chunk.text)

        # Parse JSON response — same cleanup as supervisor
        raw = response_text.strip()
        import re as _re_qa
        raw = _re_qa.sub(r'<think>.*?</think>', '', raw, flags=_re_qa.DOTALL).strip()
        if "```json" in raw:
            raw = raw.split("```json")[-1].split("```")[0].strip()
        elif "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1].strip()

        parsed = json.loads(raw)
        complexity = parsed.get("complexity", "simple")
        sub_queries = parsed.get("sub_queries") or []
        extracted_params = parsed.get("extracted_params") or {}

        logger.info(
            f"[query_analyzer] complexity={complexity!r}, "
            f"sub_queries={len(sub_queries)}, "
            f"doc_refs={extracted_params.get('document_refs', [])!r}, "
            f"sections={extracted_params.get('sections', [])!r}, "
            f"comparison={extracted_params.get('comparison_mode', False)}"
        )

        return {
            "query_complexity": complexity,
            "sub_queries": sub_queries if len(sub_queries) > 1 else None,
            "extracted_params": extracted_params if extracted_params else None,
            "current_step_index": 0,
            "retry_count": 0,
            "needs_comparison": needs_comparison,
        }

    except json.JSONDecodeError as e:
        logger.warning(f"[query_analyzer] JSON parse failed: {e}, raw={response_text[:200]!r}")
        return {"query_complexity": "simple", "sub_queries": None, "extracted_params": None, "needs_comparison": False}
    except Exception as e:
        logger.warning(f"[query_analyzer] Failed: {e}")
        return {"query_complexity": "simple", "sub_queries": None, "extracted_params": None, "needs_comparison": False}


async def supervisor_node(state: SupervisorState) -> dict:
    """
    Classify intent + decide next agent in ONE LLM call.

    This is the core supervisor logic:
    1. Extract user message
    2. Expand abbreviations in message
    3. Use query_analyzer output (if available) for enhanced context
    4. Call LLM with supervisor prompt
    5. Parse response and update state
    6. For multi-step queries, build proper task_plan from sub_queries
    """
    # get_llm_provider not needed here — supervisor uses OllamaLLMProvider directly

    user_message = _extract_user_message(state)
    iterations = state.get("iterations", 0)
    logger.info(
        f"[LANGGRAPH_NODE] Entering supervisor_node, iteration={iterations}, "
        f"current_intent={state.get('intent')!r}, pending_intent={state.get('pending_intent')!r}"
    )
    if not user_message:
        return {
            "next_agent": AgentType.DIRECT,
            "intent": "greeting",
            "iterations": iterations + 1,
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

        # Phase 5: Build analyzer context string for supervisor prompt
        _analyzer_context = ""
        _qa_complexity = state.get("query_complexity", "simple")
        _qa_sub_queries = state.get("sub_queries") or []
        _qa_params = state.get("extracted_params") or {}
        if _qa_complexity != "simple" and (_qa_sub_queries or _qa_params):
            _ctx_parts = [
                "\n═══════════════════════════════════════════════════════",
                "PRE-ANALYZED QUERY CONTEXT (from query_analyzer)",
                "═══════════════════════════════════════════════════════\n",
                f"Complexity: {_qa_complexity}",
            ]
            if _qa_sub_queries:
                _ctx_parts.append(f"Sub-queries ({len(_qa_sub_queries)}):")
                for i, sq in enumerate(_qa_sub_queries):
                    _ctx_parts.append(f"  {i+1}. [{sq.get('intent_hint', '?')}] {sq.get('query', '')}")
            if _qa_params.get("document_refs"):
                _ctx_parts.append(f"Document references: {_qa_params['document_refs']}")
            if _qa_params.get("sections"):
                _ctx_parts.append(f"Sections: {_qa_params['sections']}")
            if _qa_params.get("comparison_mode"):
                _ctx_parts.append("Mode: COMPARISON (user wants to compare items)")
            _ctx_parts.append("")
            _ctx_parts.append("Use this pre-analyzed context to inform your routing decision.")
            _ctx_parts.append("For multi-step queries, the system will handle step progression automatically.")
            _ctx_parts.append("")
            _analyzer_context = "\n".join(_ctx_parts)

        async for chunk in classifier.astream(
            [_LLMMsg(role="user", content=query_for_classifier)],
            system_prompt=_SUPERVISOR_PROMPT.format(
                max_iterations=max_iter,
                analyzer_context=_analyzer_context,
            ),
            temperature=0.0,
            max_tokens=256,  # JSON output ~100-150 tokens; reduced from 512 to save latency
            think=False,  # Disable thinking to reduce latency for classification
        ):
            if hasattr(chunk, "text") and chunk.text:
                response_text += str(chunk.text)

        decision = _parse_supervisor_response(response_text)

        # ── Keyword safety net 0: needs_memory fallback ─────────────────────
        # If the query explicitly contains personal pronouns, forcefully trigger
        # memory_recall even if the LLM classification missed it (e.g. typos).
        # Pattern is module-level (P1-7) — no per-call compile.
        if not decision.get("needs_memory") and _PERSONAL_REF_PATTERN.search(query_for_classifier):
            logger.info("[supervisor] Keyword safety net: forcing needs_memory=True because query contains personal keywords")
            decision["needs_memory"] = True

        # ── Keyword safety net 1: direct/greeting → rag ─────────────────────
        # If the LLM classified as greeting/direct but the message contains
        # document-related keywords, override to rag + search.
        if (
            decision["next_agent"] == AgentType.DIRECT
            and decision["intent"] in ("greeting", "personal")
            and decision.get("is_legal_query", False)
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
            # Pattern is module-level (P1-7) \u2014 no per-call compile.
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
            else:
                # V2.1: "Ask user when uncertain" — if intent requires a doc but
                # regex didn't match a named doc AND no document_ids in state,
                # ask the user to specify which document they want.
                from app.services.agents.clarification import (
                    ask_user_clarification,
                    should_ask_for_doc_reference,
                )
                if should_ask_for_doc_reference(query_for_classifier, has_named_doc_match=False):
                    logger.info(
                        f"[supervisor] V2.1 fallback: asking user for doc reference "
                        f"(intent={decision['intent']!r}, no named doc matched)"
                    )
                    await ask_user_clarification(
                        state,
                        question=(
                            "Bạn muốn tra cứu văn bản nào? "
                            "Vui lòng cung cấp tên đầy đủ hoặc số hiệu văn bản "
                            "(ví dụ: 'Nghị định 13/2023/NĐ-CP' hoặc 'Luật An ninh mạng 2018')."
                        ),
                        context={"type": "missing_doc_reference", "intent": decision["intent"]},
                    )

        # ── Keyword safety net 2: attached file → rag ──────────────────────
        # If the user attached a document this turn but the LLM routed to direct
        # chat, the file would be silently ignored. Force the rag (react_executor)
        # branch so the uploaded file is actually read/searched/checked.
        if state.get("document_ids") and decision["next_agent"] == AgentType.DIRECT:
            logger.warning(
                f"[supervisor] document_ids present but next_agent=direct — "
                f"overriding to rag so the attached file is used"
            )
            decision["next_agent"] = AgentType.RAG
            if decision.get("intent") in ("greeting", "personal", "direct", "chitchat"):
                decision["intent"] = "search"

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
            # Comparison flag from query_analyzer (for answer_generator comparison mode)
            # This is passed through state from query_analyzer_node
            "needs_comparison": state.get("needs_comparison", False),
        }

        # ── Phase 5: Query Analyzer enhanced routing ────────────────────
        # If query_analyzer detected a complex query, override the plan
        complexity = state.get("query_complexity", "simple")
        sub_queries = state.get("sub_queries") or []
        extracted_params = state.get("extracted_params") or {}
        current_step = state.get("current_step_index", 0)

        if complexity != "simple" and sub_queries and current_step < len(sub_queries):
            # Multi-step execution: use sub_queries to build a proper plan
            current_sq = sub_queries[current_step]
            sq_intent = current_sq.get("intent_hint", "search")
            sq_query = current_sq.get("query", user_message)

            # Normalize intent hint
            sq_intent = _INTENT_NORMALIZE.get(sq_intent, sq_intent)

            # Determine agent from intent
            sq_agent = _INTENT_TO_AGENT_FALLBACK.get(sq_intent, "rag")

            logger.info(
                f"[supervisor] Phase 5: multi-step routing — "
                f"complexity={complexity!r}, step {current_step+1}/{len(sub_queries)}, "
                f"intent={sq_intent!r}, agent={sq_agent!r}, query={sq_query[:80]!r}"
            )

            result["intent"] = sq_intent
            result["next_agent"] = sq_agent
            result["rewritten_query"] = sq_query
            result["current_step_index"] = current_step

            # Build task_plan from remaining sub_queries
            remaining_intents = [sq.get("intent_hint", "search") for sq in sub_queries[current_step:]]
            result["task_plan"] = remaining_intents
            if len(remaining_intents) > 1:
                result["pending_intent"] = remaining_intents[-1]

        # Pre-fill section_reference from extracted_params if available
        if extracted_params.get("sections") and not state.get("section_reference"):
            sections = extracted_params["sections"]
            if current_step < len(sub_queries or []):
                # Match section to current sub_query if possible
                current_sq_text = (sub_queries[current_step].get("query", "") if sub_queries else "")
                for sec in sections:
                    if sec.lower() in current_sq_text.lower():
                        result["section_reference"] = sec
                        break
                else:
                    # Default: use first unprocessed section
                    result["section_reference"] = sections[0] if sections else None
            elif sections:
                result["section_reference"] = sections[0]

        # Phase 3: Smart memory routing — determine if memory_recall is needed
        intent = decision["intent"]
        needs_memory = decision.get("needs_memory", False)
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

        # Complexity-aware thinking decision.
        # Thinking is ONLY enabled when:
        #   1. query_complexity != "simple" (multi-step, comparison queries), OR
        #   2. intent = "kg_query" (always needs reasoning for graph relationships)
        # This replaces the old logic that checked source_count >= 5
        # (which was dead code — sources are always empty at supervisor time).
        intent = decision["intent"]
        complexity = state.get("query_complexity", "simple")
        is_complex = complexity != "simple"

        if intent == "kg_query":
            result["enable_thinking"] = True
            logger.info(f"[supervisor] Thinking ENABLED for kg_query (always-think intent)")
        elif is_complex:
            result["enable_thinking"] = True
            logger.info(f"[supervisor] Thinking ENABLED for {intent!r} (complexity={complexity!r})")
        else:
            result["enable_thinking"] = False
            logger.info(f"[supervisor] Thinking DISABLED for {intent!r} (simple query)")

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

        logger.info(
            f"[LANGGRAPH_DECISION] supervisor_node decision: next_agent={result.get('next_agent')!r}, "
            f"intent={result.get('intent')!r}, rewritten_query={result.get('rewritten_query', '')[:100]!r}, "
            f"needs_memory={result.get('needs_memory')}, search_mode={result.get('search_mode')!r}"
        )

        # ── Langfuse: emit routing decision span ──────────────────────────────
        langfuse = _get_langfuse_client()
        if langfuse:
            try:
                decision_intent = decision.get("intent", intent)
                keyword_override = (
                    decision.get("reasoning", "").startswith("(overridden by keyword safety net)")
                    if decision.get("reasoning") else False
                )
                obs = langfuse.start_observation(
                    name="supervisor_routing_decision",
                    input={
                        "user_message": user_message,
                        "expanded_query": expanded_message,
                        "intent": decision_intent,
                        "next_agent": str(decision.get("next_agent")),
                        "needs_memory": bool(result.get("needs_memory")),
                        "search_mode": str(result.get("search_mode")),
                        "task_plan": result.get("task_plan") or [],
                        "pending_intent": str(result.get("pending_intent") or ""),
                        "query_complexity": str(state.get("query_complexity", "simple")),
                    },
                    level="DEBUG",
                )
                obs.update(
                    output={
                        "next_agent": str(result.get("next_agent")),
                        "intent": str(result.get("intent")),
                        "needs_memory": bool(result.get("needs_memory")),
                        "keyword_override": keyword_override,
                        "search_mode": str(result.get("search_mode")),
                        "task_plan": result.get("task_plan") or [],
                    }
                )
                obs.end()
            except Exception as e:
                logger.warning(f"[langfuse] supervisor_routing_decision span failed: {e}")

        return result

    except Exception as e:
        logger.error(f"[supervisor] LLM call failed: {e}")
        # Fail-safe: default to RAG (document search) instead of direct.
        # Direct gives empty answers; RAG at least attempts document retrieval.
        logger.warning(
            f"[LANGGRAPH_DECISION] supervisor_node exception fallback: next_agent=RAG, "
            f"intent={state.get('intent', 'search')!r}"
        )
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
# Result Evaluator Node — Phase 5
# =============================================================================


async def result_evaluator_node(state: SupervisorState) -> dict:
    """Evaluate agent results and decide: go to answer, retry, or next step.

    Phase 5 node that runs AFTER rag/resolve_doc agents.

    Decision logic:
    1. If more sub_queries to execute → advance step, route back to supervisor
    2. If results empty AND retry_count < 2 → retry with fallback strategy
    3. Otherwise → route to answer_generator
    """
    from app.services.agent.streaming import push_event

    sources = state.get("sources", [])
    kg_summaries = state.get("kg_summaries", [])
    intent = state.get("intent", "")
    retry_count = state.get("retry_count", 0)
    sub_queries = state.get("sub_queries") or []
    current_step = state.get("current_step_index", 0)
    complexity = state.get("query_complexity", "simple")

    has_results = bool(sources)
    # Check kg_summaries for SUBSTANTIVE content (>50 chars, not just metadata)
    # Short entries like "Đã xác định văn bản: **title**" are metadata from resolve_doc,
    # not actual search results. With operator.add reducer, these accumulate and would
    # cause false positives without this filter.
    _MIN_SUBSTANTIVE_LEN = 50
    substantive_kg = [
        s for s in kg_summaries
        if isinstance(s, str) and len(s.strip()) >= _MIN_SUBSTANTIVE_LEN
        and not s.strip().startswith("Đã xác định văn bản:")
    ]
    if not has_results and substantive_kg:
        has_results = True

    logger.info(
        f"[LANGGRAPH_NODE] Entering result_evaluator_node, intent={intent!r}, has_results={has_results}, "
        f"sources={len(sources)}, kg_summaries={len(kg_summaries)} (substantive={len(substantive_kg)}), "
        f"complexity={complexity!r}, step={current_step}/{len(sub_queries)}, "
        f"retry_count={retry_count}"
    )

    # ── Check 1: More sub_queries to execute? ─────────────────────────────
    if complexity != "simple" and sub_queries and (current_step + 1) < len(sub_queries):
        # Accumulate current step results
        step_result = {
            "step_intent": intent,
            "step_index": current_step,
            "sources_count": len(sources),
            "kg_summaries": kg_summaries[:3],  # Keep summaries compact
        }

        next_step = current_step + 1
        next_sq = sub_queries[next_step]
        next_intent = next_sq.get("intent_hint", "search")
        next_query = next_sq.get("query", "")

        logger.info(
            f"[result_evaluator] Advancing to step {next_step+1}/{len(sub_queries)}: "
            f"intent={next_intent!r}, query={next_query[:80]!r}"
        )

        await push_event(state, "status", {
            "step": "searching",
            "detail": f"Bước {next_step+1}/{len(sub_queries)}: {next_query[:60]}...",
        })

        logger.info(
            f"[LANGGRAPH_DECISION] result_evaluator_node decision: next_agent='supervisor_loop' (advance to step {next_step+1})"
        )
        return {
            "current_step_index": next_step,
            "accumulated_results": [step_result],
            "retry_count": 0,  # Reset retry for new step
            # Route back to supervisor for next step
            "next_agent": "supervisor_loop",
        }

    # ── Check 2: Results insufficient? Retry with fallback ────────────────
    if not has_results and retry_count < 2:
        _FALLBACK_MAP = {
            "search": "kg_query",
            "kg_query": "search",
            "search_section": "search",  # Section not found → try general search
        }
        fallback_intent = _FALLBACK_MAP.get(intent)
        if fallback_intent:
            logger.info(
                f"[result_evaluator] No results for {intent!r}, "
                f"retrying with {fallback_intent!r} (retry {retry_count+1}/2)"
            )
            await push_event(state, "status", {
                "step": "searching",
                "detail": f"Thử phương pháp tìm kiếm khác...",
            })
            logger.info(
                f"[LANGGRAPH_DECISION] result_evaluator_node decision: next_agent={_INTENT_TO_AGENT_FALLBACK.get(fallback_intent, 'rag')!r} (retry with intent {fallback_intent!r})"
            )
            return {
                "intent": fallback_intent,
                "next_agent": _INTENT_TO_AGENT_FALLBACK.get(fallback_intent, "rag"),
                "retry_count": retry_count + 1,
                "retry_strategy": f"fallback_{fallback_intent}",
                "search_mode": "kg" if fallback_intent == "kg_query" else "vector",
            }

    # ── Default: sufficient or max retries → answer_generator ─────────────
    logger.info(
        f"[LANGGRAPH_DECISION] result_evaluator_node decision: next_agent=ANSWER_GENERATOR (sufficient results or max retries)"
    )
    return {
        "next_agent": AgentType.ANSWER_GENERATOR,
    }


def route_from_evaluator(state: SupervisorState) -> str:
    """Route after result evaluation.

    - ANSWER_GENERATOR → answer (sufficient results or max retries)
    - supervisor_loop  → supervisor (more sub_queries to execute)
    - rag / other      → retry with different strategy
    """
    langfuse = _get_langfuse_client()
    next_agent = state.get("next_agent")

    if next_agent == AgentType.ANSWER_GENERATOR:
        target = "answer_generator"
    elif next_agent == "supervisor_loop":
        target = "supervisor"
    else:
        _RETRY_MAP = {
            AgentType.RAG: "rag",
            AgentType.PEOPLE: "people",
            AgentType.WRITE: "write",
        }
        target = _RETRY_MAP.get(next_agent, "rag")

    if langfuse:
        try:
            obs = langfuse.start_observation(
                name="route_from_evaluator",
                input={
                    "next_agent": str(next_agent),
                    "intent": str(state.get("intent", "")),
                    "query_complexity": str(state.get("query_complexity", "simple")),
                    "current_step": int(state.get("current_step_index", 0)),
                },
                level="DEFAULT",
            )
            obs.update(
                output={
                    "target_node": target,
                    "next_agent": str(next_agent),
                    "is_retry": next_agent not in (AgentType.ANSWER_GENERATOR, "supervisor_loop"),
                }
            )
            obs.end()
        except Exception as e:
            logger.warning(f"[langfuse] route_from_evaluator span failed: {e}")

    if next_agent == AgentType.ANSWER_GENERATOR:
        logger.info("[LANGGRAPH_ROUTE] result_evaluator -> answer_generator")
        return "answer_generator"

    if next_agent == "supervisor_loop":
        logger.info("[LANGGRAPH_ROUTE] result_evaluator -> supervisor")
        return "supervisor"

    logger.info(f"[LANGGRAPH_ROUTE] result_evaluator -> {target} (retry)")
    return target


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
# ReAct Executor Node — RAG group (tool-aware planning)
# =============================================================================

async def react_executor_node(state: SupervisorState) -> dict:
    """Single tool-calling ReAct loop for the RAG group.

    The LLM is given the RAG tool schemas and decides which tools to call (in
    parallel where independent), iterating until it can answer. Replaces the
    static ``rag → result_evaluator → answer_generator`` chain when
    ``NEXUSRAG_LG_RAG_REACT`` is enabled. resolve_doc / section / kg / memory
    are all just tools here — no intent taxonomy, no prerequisite plan.
    """
    import asyncio
    from app.core.config import settings
    from app.services.agent.streaming import push_event
    from app.services.agent.nodes import strip_thinking_tags
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage as _LLMMsg
    from app.services.agents.react_tools import ToolContext, RAG_TOOL_SCHEMAS, dispatch_tool
    from app.prompts.agents.react_prompt import build_react_system_prompt

    user_message = _extract_user_message(state)
    query = state.get("rewritten_query") or state.get("original_query") or user_message
    max_steps = getattr(settings, "NEXUSRAG_REACT_MAX_TOOL_STEPS", 6)
    top_k = getattr(settings, "NEXUSRAG_REACT_TOP_K", 8)
    logger.info(f"[LANGGRAPH_NODE] Entering react_executor_node, query={query!r}, max_steps={max_steps}")

    ctx = ToolContext(
        workspace_ids=state.get("workspace_ids", []) or [],
        document_ids=state.get("document_ids"),
        # Files attached this turn — kept separate so resolve_document_reference can
        # re-scope document_ids without losing track of the user's uploaded file.
        uploaded_document_ids=state.get("document_ids"),
        user_id=state.get("user_id"),
        session_id=state.get("session_id"),
        existing_citation_ids={},
        top_k=top_k,
        state=state,
    )

    llm = get_llm_provider()
    system_prompt = build_react_system_prompt(state.get("user_memory_context", "") or "")
    msgs: list = [_LLMMsg(role="user", content=query)]
    sources: list = []
    images: list = []

    # How many sources/images already pushed to the client — avoids re-sending.
    _emitted = {"sources": 0, "images": 0}

    async def _emit_artifacts() -> None:
        """Flush newly-collected sources/images to the client.

        Called BEFORE a turn that may stream the answer so the client has the
        citation data by the time answer tokens arrive.
        """
        if len(sources) > _emitted["sources"]:
            await push_event(state, "sources", sources)
            _emitted["sources"] = len(sources)
        if len(images) > _emitted["images"]:
            await push_event(state, "images", images)
            _emitted["images"] = len(images)

    async def _run_turn(use_tools: bool):
        """Run one LLM turn, streaming answer text to the client as it arrives.

        Speculative streaming (mirrors the legacy agent): we emit text tokens
        live, but only while no tool call has appeared in this turn. If the turn
        ends up requesting tools, that streamed text was a thinking-aloud
        preamble — we send ``token_rollback`` so the client discards it before
        the tools run. When the turn produces no tool calls, the streamed text
        IS the final answer (no re-generation, no fake chunking).
        """
        calls, text = [], ""
        streamed = False
        async for c in llm.astream(
            msgs,
            system_prompt=system_prompt,
            tools=RAG_TOOL_SCHEMAS if use_tools else None,
            tool_choice="auto" if use_tools else None,
            temperature=0.1,
            max_tokens=4096,
            think=False,
        ):
            if c.type == "function_call" and c.function_call:
                calls.append(c.function_call)
            elif c.type == "text" and c.text:
                text += c.text
                if not calls:  # speculative — stop emitting once a tool call shows up
                    await push_event(state, "token", c.text)
                    streamed = True
        if calls and streamed:
            await push_event(state, "token_rollback", {})
        return calls, text

    async def _finish(answer_text: str) -> dict:
        # Answer text was already streamed live by _run_turn — here we only
        # finalize (clean thinking tags, emit a fallback if nothing came out).
        await _emit_artifacts()
        final = strip_thinking_tags(answer_text or "").strip()
        if not final:
            final = "Xin lỗi, tôi chưa tạo được câu trả lời từ kho văn bản."
            await push_event(state, "token", final)
        logger.info(
            f"[react_executor] done: {len(sources)} sources, answer={len(final)} chars"
        )
        return {
            "final_answer": final,
            "sources": sources,
            "images": images,
            "document_ids": ctx.document_ids,
            "next_agent": AgentType.FINISH,
        }

    try:
        for step in range(max_steps):
            await _emit_artifacts()
            calls, text = await _run_turn(use_tools=True)
            if not calls:
                return await _finish(text)

            # Assistant turn that requested tools — synthesise stable ids so the
            # tool-result messages reference them on the next request.
            tool_calls_payload = []
            for i, fc in enumerate(calls):
                cid = f"call_{step}_{i}"
                fc["_id"] = cid
                tool_calls_payload.append({
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                    },
                })
            msgs.append(_LLMMsg(role="assistant", content=text or "", tool_calls=tool_calls_payload))

            names = ", ".join(fc.get("name", "?") for fc in calls)
            await push_event(state, "status", {"step": "searching", "detail": f"Đang tra cứu: {names}"})
            logger.info(f"[react_executor] step {step + 1}/{max_steps}: {len(calls)} call(s): {names}")

            # Parallel tool execution — independent calls in the same turn run together.
            results = await asyncio.gather(
                *[dispatch_tool(fc.get("name", ""), fc.get("args", {}), ctx) for fc in calls]
            )

            stop = False
            for fc, r in zip(calls, results):
                sources.extend(r.get("sources", []) or [])
                images.extend(r.get("images", []) or [])
                msgs.append(_LLMMsg(role="tool", tool_call_id=fc["_id"], content=r.get("summary", "")))
                if (r.get("data") or {}).get("stop"):
                    stop = True

            if stop:  # ask_user already emitted clarification → wait for the user
                logger.info("[react_executor] ask_user → stopping loop for clarification")
                return {
                    "final_answer": "",
                    "next_agent": AgentType.FINISH,
                    "clarification_needed": True,
                    "sources": sources,
                    "images": images,
                }

        # Loop guard hit → force a final synthesis without tools.
        logger.warning(f"[react_executor] max steps ({max_steps}) reached — forcing synthesis")
        msgs.append(_LLMMsg(role="user", content=(
            "Hãy tổng hợp câu trả lời cuối cùng từ thông tin đã thu thập ở trên. "
            "Nếu chưa đủ căn cứ, nói rõ phần nào còn thiếu."
        )))
        await _emit_artifacts()
        _, text = await _run_turn(use_tools=False)
        return await _finish(text)

    except Exception as e:
        logger.error(f"[react_executor] loop failed: {e}", exc_info=True)
        return await _finish("")


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
    langfuse = _get_langfuse_client()
    next_agent = state.get("next_agent")
    intent = state.get("intent", "")
    needs_memory = state.get("needs_memory", False)

    if next_agent is None or next_agent == AgentType.FINISH:
        target = "END"
    elif next_agent == AgentType.RESOLVE_DOC or (
        next_agent == AgentType.RAG and intent == "resolve_doc"
    ):
        target = "resolve_doc_agent"
    elif needs_memory:
        target = "memory_recall"
    else:
        _DIRECT_MAP: dict[str, str] = {
            AgentType.RAG: "rag",
            AgentType.WRITE: "write",
            AgentType.DIRECT: "direct",
            AgentType.PEOPLE: "people",
            AgentType.ANSWER_GENERATOR: "answer_generator",
        }
        target = _DIRECT_MAP.get(next_agent, "memory_recall")

    if langfuse:
        try:
            obs = langfuse.start_observation(
                name="route_from_supervisor",
                input={
                    "next_agent": str(next_agent),
                    "intent": str(intent),
                    "needs_memory": bool(needs_memory),
                    "query_complexity": str(state.get("query_complexity", "simple")),
                    "task_plan": state.get("task_plan") or [],
                    "pending_intent": str(state.get("pending_intent") or ""),
                },
                level="DEFAULT",
            )
            obs.update(output={"target_node": target, "keyword_override": bool(state.get("_keyword_override"))})
            obs.end()
        except Exception as e:
            logger.warning(f"[langfuse] route_from_supervisor span failed: {e}")

    if next_agent is None or next_agent == AgentType.FINISH:
        logger.info("[LANGGRAPH_ROUTE] supervisor -> END")
        return END

    if next_agent == AgentType.RESOLVE_DOC or (
        next_agent == AgentType.RAG and intent == "resolve_doc"
    ):
        if _react_on():
            logger.info(f"[LANGGRAPH_ROUTE] supervisor -> react_executor (resolve via tools, intent={intent!r})")
            return "react_executor"
        logger.info(f"[LANGGRAPH_ROUTE] supervisor -> resolve_doc_agent (intent={intent!r})")
        return "resolve_doc_agent"

    if needs_memory:
        logger.info(
            f"[LANGGRAPH_ROUTE] supervisor -> memory_recall (needs_memory=True, intent={intent!r})"
        )
        return "memory_recall"

    _DIRECT_MAP: dict[str, str] = {
        AgentType.RAG: "rag",
        AgentType.WRITE: "write",
        AgentType.DIRECT: "direct",
        AgentType.PEOPLE: "people",
        AgentType.ANSWER_GENERATOR: "answer_generator",
    }
    target = _DIRECT_MAP.get(next_agent)
    if target == "rag" and _react_on():
        target = "react_executor"
    if target:
        logger.info(
            f"[LANGGRAPH_ROUTE] supervisor -> {target} (needs_memory=False, intent={intent!r})"
        )
        return target

    logger.info(f"[LANGGRAPH_ROUTE] supervisor -> memory_recall (fallback, intent={intent!r})")
    return "memory_recall"


def route_from_rag(state: SupervisorState) -> str:
    """Route after rag agent completes.

    Phase 5: Routes through result_evaluator for quality checking and
    multi-step execution. Abbreviation loop-back and search_section
    pending cases still go directly to supervisor.

    Priority:
    1. should_loop_back (abbreviation) → supervisor
    2. search_section with pending section_reference → supervisor
    3. Everything else → result_evaluator (quality check + multi-step)
    """
    langfuse = _get_langfuse_client()
    pending_section = state.get("section_reference")
    intent = state.get("intent")
    loop = state.get("should_loop_back")

    if loop:
        target = "supervisor"
    elif intent == "search_section" and pending_section:
        target = "supervisor"
    else:
        target = "result_evaluator"

    if langfuse:
        try:
            obs = langfuse.start_observation(
                name="route_from_rag",
                input={
                    "intent": str(intent),
                    "section_reference": str(pending_section or ""),
                    "should_loop_back": bool(loop),
                    "has_sources": bool(state.get("sources")),
                    "has_kg_summaries": bool(state.get("kg_summaries")),
                },
                level="DEFAULT",
            )
            obs.update(output={"target_node": target})
            obs.end()
        except Exception as e:
            logger.warning(f"[langfuse] route_from_rag span failed: {e}")

    logger.info(
        f"[route_from_rag] intent={intent!r}, section_reference={pending_section!r}, "
        f"should_loop_back={loop}, complexity={state.get('query_complexity', 'simple')!r}"
    )

    if loop:
        logger.info("[LANGGRAPH_ROUTE] rag -> supervisor (abbreviation loop-back)")
        return "supervisor"

    if intent == "search_section" and pending_section:
        logger.info("[LANGGRAPH_ROUTE] rag -> supervisor (pending search_section)")
        return "supervisor"

    logger.info("[LANGGRAPH_ROUTE] rag -> result_evaluator")
    return "result_evaluator"


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
    langfuse = _get_langfuse_client()
    next_agent = state.get("next_agent")
    intent = state.get("intent", "")
    section_ref = state.get("section_reference", "")
    pending_intent = state.get("pending_intent")

    if next_agent == AgentType.FINISH:
        target = "END"
    elif intent == "search_section" and section_ref:
        target = "rag"
    elif pending_intent == "search_section" and section_ref:
        target = "rag"
    elif pending_intent == "search":
        target = "rag"
    elif pending_intent == "summarize":
        target = "answer_generator"
    else:
        target = "answer_generator"

    if langfuse:
        try:
            obs = langfuse.start_observation(
                name="route_from_resolve_doc",
                input={
                    "next_agent": str(next_agent),
                    "intent": str(intent),
                    "section_reference": str(section_ref or ""),
                    "pending_intent": str(pending_intent or ""),
                    "document_ids": [str(d) for d in (state.get("document_ids") or [])],
                },
                level="DEFAULT",
            )
            obs.update(output={"target_node": target})
            obs.end()
        except Exception as e:
            logger.warning(f"[langfuse] route_from_resolve_doc span failed: {e}")

    # If we have a pending_intent from the supervisor's task_plan, continue with it
    # even if resolve_doc couldn't fully resolve (ambiguous case)
    if pending_intent:
        logger.info(
            f"[route_from_resolve_doc] pending_intent={pending_intent!r}, "
            f"section_ref={section_ref!r}"
        )
        if pending_intent == "search_section" and section_ref:
            logger.info("[LANGGRAPH_ROUTE] resolve_doc_agent -> rag (pending search_section)")
            return "rag"
        elif pending_intent == "search":
            logger.info("[LANGGRAPH_ROUTE] resolve_doc_agent -> rag (pending search)")
            return "rag"
        elif pending_intent == "summarize":
            logger.info("[LANGGRAPH_ROUTE] resolve_doc_agent -> answer_generator (pending summarize)")
            return "answer_generator"

    if next_agent == AgentType.FINISH:
        logger.info("[LANGGRAPH_ROUTE] resolve_doc_agent -> END (not found / ambiguous)")
        return END

    if intent == "search_section" and section_ref:
        logger.info(f"[LANGGRAPH_ROUTE] resolve_doc_agent -> rag (has section_ref={section_ref!r})")
        return "rag"

    logger.info("[LANGGRAPH_ROUTE] resolve_doc_agent -> answer_generator (resolved)")
    return "answer_generator"


# shouldContinue_from_supervisor removed — was dead code (never called).


# =============================================================================
# Graph Builder
# =============================================================================

def create_supervisor_graph():
    """
    Build and compile the supervisor-based multi-agent graph.

    Phase 5 Flow (Query Analyzer + Result Evaluator + Multi-Step):
        START → query_analyzer → supervisor
                  ├── [needs_memory=True] → memory_recall → query_enricher
                  │     ├── [personal] → direct
                  │     ├── [search/...] → rag → result_evaluator → answer_generator
                  │     ├── [write] → write → END
                  │     └── [people] → people → mongo_formatter → END
                  │
                  └── [needs_memory=False] → target agent (direct bypass)
                        ├── rag → result_evaluator → answer_generator → END
                        ├── write → END
                        ├── direct → END
                        └── people → mongo_formatter → END

    result_evaluator checks quality and handles:
    - Multi-step sub_queries: advance to next step → supervisor
    - Empty results: retry with fallback strategy → rag
    - Sufficient results: → answer_generator

    resolve_doc always bypasses memory (no personal context needed).
    """
    graph = StateGraph(SupervisorState)

    # Nodes
    graph.add_node("query_analyzer", query_analyzer_node)       # Phase 5: NEW
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("result_evaluator", result_evaluator_node)   # Phase 5: NEW
    graph.add_node("memory_recall", _memory_recall_wrapper)
    graph.add_node("query_enricher", _query_enricher_wrapper)   # Phase 3
    graph.add_node("rag", _rag_agent_wrapper)
    graph.add_node("resolve_doc_agent", _resolve_doc_agent_wrapper)  # Phase 2
    graph.add_node("write", _write_agent_wrapper)
    graph.add_node("people", _people_agent_wrapper)
    graph.add_node("direct", direct_answer_node)
    graph.add_node("answer_generator", answer_generator_node)  # RAG-only
    graph.add_node("mongo_formatter", mongo_formatter_node)    # People-only
    graph.add_node("react_executor", react_executor_node)      # RAG group (flag: NEXUSRAG_LG_RAG_REACT)

    # Edges — Phase 5: START → query_analyzer → supervisor
    graph.add_edge(START, "query_analyzer")
    graph.add_edge("query_analyzer", "supervisor")

    # Conditional edges from supervisor based on needs_memory flag
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
            "react_executor": "react_executor",  # RAG group via tool-aware ReAct loop
            END: END,
        },
    )

    # memory_recall → query_enricher (always: enricher is a no-op when not needed)
    graph.add_edge("memory_recall", "query_enricher")

    # query_enricher → target agent based on next_agent / intent
    def route_from_enricher(state: SupervisorState) -> str:
        """Route after query enrichment to the correct agent.

        personal intent → direct (answer from memory, no RAG needed)
        all other intents → their designated agent
        """
        langfuse = _get_langfuse_client()
        next_agent = state.get("next_agent", AgentType.FINISH)
        intent = state.get("intent", "")

        if intent == "personal":
            target = "direct"
        else:
            _MAP = {
                AgentType.RAG: "rag",
                AgentType.WRITE: "write",
                AgentType.DIRECT: "direct",
                AgentType.PEOPLE: "people",
            }
            target = _MAP.get(next_agent, "direct")

        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="route_from_enricher",
                    input={
                        "next_agent": str(next_agent),
                        "intent": str(intent),
                        "has_memory_context": bool(state.get("user_memory_context")),
                    },
                    level="DEFAULT",
                )
                obs.update(output={"target_node": target})
                obs.end()
            except Exception as e:
                logger.warning(f"[langfuse] route_from_enricher span failed: {e}")

        if intent == "personal":
            logger.info("[LANGGRAPH_ROUTE] query_enricher -> direct (personal intent)")
            return "direct"

        _MAP = {
            AgentType.RAG: "rag",
            AgentType.WRITE: "write",
            AgentType.DIRECT: "direct",
            AgentType.PEOPLE: "people",
        }
        target = _MAP.get(next_agent, "direct")
        if target == "rag" and _react_on():
            target = "react_executor"
        logger.info(
            f"[LANGGRAPH_ROUTE] query_enricher -> {target} (intent={intent!r}, next_agent={next_agent!r})"
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
            "react_executor": "react_executor",
        },
    )

    # Intent-specific output pipelines:
    # - RAG → result_evaluator (Phase 5: quality check + multi-step)
    # - resolve_doc_agent → result_evaluator / rag (section) / END (ambiguous)
    # - Write → END (write_agent already produces final_answer via streaming)
    # - People → mongo_formatter (lightweight LLM call, ~1KB prompt)
    # - Direct → END (direct_answer_node already produces final_answer via streaming)

    # Phase 5: rag → result_evaluator (instead of direct to answer_generator)
    graph.add_conditional_edges(
        "rag",
        route_from_rag,
        {
            "supervisor": "supervisor",
            "result_evaluator": "result_evaluator",
        },
    )

    # resolve_doc_agent → result_evaluator (for multi-step) / rag / END
    graph.add_conditional_edges(
        "resolve_doc_agent",
        route_from_resolve_doc,
        {
            "answer_generator": "answer_generator",
            "rag": "rag",
            END: END,
        },
    )

    # Phase 5: result_evaluator → answer_generator / rag (retry) / supervisor (next step)
    graph.add_conditional_edges(
        "result_evaluator",
        route_from_evaluator,
        {
            "answer_generator": "answer_generator",
            "rag": "rag",
            "supervisor": "supervisor",
        },
    )

    graph.add_edge("write", END)
    graph.add_edge("people", "mongo_formatter")
    graph.add_edge("direct", END)

    # Terminal nodes
    graph.add_edge("answer_generator", END)
    graph.add_edge("mongo_formatter", END)
    graph.add_edge("react_executor", END)  # RAG group (flag) — executor is terminal

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
    langfuse = _get_langfuse_client()
    logger.info("[LANGGRAPH_NODE] Entering memory_recall_wrapper")
    import uuid as _uuid

    user_id = state.get("user_id")
    if not user_id:
        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="memory_recall",
                    input={"user_id": None},
                    level="DEFAULT",
                )
                obs.update(output={"outcome": "no_user_id"})
                obs.end()
            except Exception as e:
                logger.warning(f"[langfuse] memory_recall span failed: {e}")
        return {}

    user_message = state.get("rewritten_query") or state.get("original_query", "")
    if not user_message:
        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="memory_recall",
                    input={"user_id": str(user_id)},
                    level="DEFAULT",
                )
                obs.update(output={"outcome": "no_message"})
                obs.end()
            except Exception as e:
                logger.warning(f"[langfuse] memory_recall span failed: {e}")
        return {}

    try:
        from app.services.graphiti_client import search_user_memory

        uid = user_id
        if isinstance(uid, int):
            uid = _uuid.UUID(int=uid)
        elif isinstance(uid, str):
            uid = _uuid.UUID(uid)

        memory = await search_user_memory(uid, user_message, top_k=5)
        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="memory_recall",
                    input={"user_id": str(uid), "query": user_message},
                    level="DEFAULT",
                )
                obs.update(
                    output={
                        "outcome": "found" if memory and "No relevant memories" not in memory else "no_memory",
                        "memory_chars": len(memory) if memory else 0,
                    }
                )
                obs.end()
            except Exception as e:
                logger.warning(f"[langfuse] memory_recall span failed: {e}")

        if memory and "No relevant memories" not in memory:
            logger.info(f"[memory_recall_wrapper] Graphiti injected {len(memory)} chars")
            return {"user_memory_context": memory}
        else:
            logger.info(f"[memory_recall_wrapper] Graphiti found no relevant memory for user_id={uid}")
    except Exception as e:
        logger.warning(f"[memory_recall_wrapper] failed: {e}")
        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="memory_recall",
                    input={"user_id": str(user_id), "query": user_message},
                    level="DEFAULT",
                )
                obs.update(output={"outcome": "error", "error": str(e)})
                obs.end()
            except Exception:
                pass

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
    logger.info("[LANGGRAPH_NODE] Entering query_enricher_wrapper")
    langfuse = _get_langfuse_client()
    memory = state.get("user_memory_context", "")
    query = state.get("rewritten_query", "") or state.get("original_query", "")
    intent = state.get("intent", "")

    # personal intent → direct_answer_node handles memory directly, no rewrite needed
    if intent == "personal":
        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="query_enricher",
                    input={"query": query, "intent": intent},
                    level="DEFAULT",
                )
                obs.update(output={"outcome": "skipped", "reason": "personal_intent"})
                obs.end()
            except Exception as e:
                logger.warning(f"[langfuse] query_enricher span failed: {e}")
        return {}

    # Nothing to enrich if no memory or no personal reference in query
    if not memory or not query or "No relevant memories" in memory:
        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="query_enricher",
                    input={"query": query, "intent": intent},
                    level="DEFAULT",
                )
                obs.update(output={"outcome": "skipped", "reason": "no_memory"})
                obs.end()
            except Exception as e:
                logger.warning(f"[langfuse] query_enricher span failed: {e}")
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
            if langfuse:
                try:
                    obs = langfuse.start_observation(
                        name="query_enricher",
                        input={"query": query, "intent": intent},
                        level="DEFAULT",
                    )
                    obs.update(output={"outcome": "enriched", "enriched_query": enriched})
                    obs.end()
                except Exception as e:
                    logger.warning(f"[langfuse] query_enricher span failed: {e}")
            return {"rewritten_query": enriched}
        else:
            logger.debug(f"[query_enricher] No enrichment applied for: {query!r}")
            if langfuse:
                try:
                    obs = langfuse.start_observation(
                        name="query_enricher",
                        input={"query": query, "intent": intent},
                        level="DEFAULT",
                    )
                    obs.update(output={"outcome": "no_change"})
                    obs.end()
                except Exception as e:
                    logger.warning(f"[langfuse] query_enricher span failed: {e}")

    except Exception as e:
        logger.warning(f"[query_enricher] Failed: {e}")
        if langfuse:
            try:
                obs = langfuse.start_observation(
                    name="query_enricher",
                    input={"query": query, "intent": intent},
                    level="DEFAULT",
                )
                obs.update(output={"outcome": "error", "error": str(e)})
                obs.end()
            except Exception:
                pass

    return {}


async def _rag_agent_wrapper(state: SupervisorState) -> dict:
    """Wrapper that imports and calls rag_agent_node."""
    logger.info("[LANGGRAPH_NODE] Entering rag_agent_wrapper")
    from app.services.agents.rag_agent import rag_agent_node

    return await rag_agent_node(state)


async def _resolve_doc_agent_wrapper(state: SupervisorState) -> dict:
    """Phase 2: Wrapper that imports and calls resolve_doc_agent_node."""
    logger.info("[LANGGRAPH_NODE] Entering resolve_doc_agent_wrapper")
    from app.services.agents.resolve_doc_agent import resolve_doc_agent_node

    return await resolve_doc_agent_node(state)


async def _write_agent_wrapper(state: SupervisorState) -> dict:
    """Wrapper that imports and calls write_agent_node."""
    logger.info("[LANGGRAPH_NODE] Entering write_agent_wrapper")
    from app.services.agents.write_agent import write_agent_node

    return await write_agent_node(state)


async def _people_agent_wrapper(state: SupervisorState) -> dict:
    """Wrapper that imports and calls people_agent_node."""
    logger.info("[LANGGRAPH_NODE] Entering people_agent_wrapper")
    from app.services.agents.people_agent import people_agent_node

    return await people_agent_node(state)


# =============================================================================
# Backward Compatibility - Re-export for existing imports
# =============================================================================

# Keep old SupervisorState name for compatibility during transition
SupervisorStateModel = SupervisorState
create_supervisor = create_supervisor_graph

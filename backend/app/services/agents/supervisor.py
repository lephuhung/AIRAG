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
    r"(?<!\w)(tôi|của\s+tôi|cho\s+tôi|đơn\s+vị\s+tôi|cơ\s+quan\s+tôi|nơi\s+tôi|chỗ\s+tôi|chúng\s+tôi|của\s+chúng\s+tôi|công\s+tác\s+của\s+tôi|làm\s+việc\s+của\s+tôi|tôi\s+tên|tên\s+(của\s+)?tôi|tôi\s+là\s+ai|tôi\s+làm\s+việc|tôi\s+công\s+tác|tôi\s+đang\s+ở)(?!\w)",
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

    Uses memory agent (gemma-4-E4B) to infer which meaning fits the context.
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
        # Same robust fence-stripping as _parse_supervisor_response / query_analyzer_node
        if "```json" in clean:
            clean = clean.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean:
            parts = clean.split("```")
            if len(parts) >= 3:
                clean = parts[1].strip()

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


# ---------------------------------------------------------------------------
# Follow-up query condensing (history-aware rewrite)
# ---------------------------------------------------------------------------

_CONDENSE_SYSTEM_PROMPT = """Bạn là trợ lý viết lại câu hỏi cho hệ thống tìm kiếm tài liệu.

Nhiệm vụ: dựa vào LỊCH SỬ HỘI THOẠI, viết lại CÂU HỎI HIỆN TẠI thành một câu hỏi \
ĐỘC LẬP, TỰ CHỨA ĐẦY ĐỦ NGỮ CẢNH (đối tượng, chủ đề, tên/loại văn bản... mà người \
dùng đang ngầm nhắc tới ở các lượt trước).

QUY TẮC:
- Giữ nguyên Ý ĐỊNH của câu hỏi hiện tại, chỉ bổ sung ngữ cảnh còn thiếu từ lịch sử.
- Nếu câu hỏi hiện tại ĐÃ tự chứa đầy đủ ngữ cảnh (hoặc đổi sang chủ đề mới), \
trả lại NGUYÊN VĂN câu hỏi đó.
- KHÔNG trả lời câu hỏi. KHÔNG giải thích. CHỈ trả về đúng MỘT câu hỏi đã viết lại."""


def _get_prior_history(state: SupervisorState, max_turns: int = 6) -> list[tuple[str, str]]:
    """Return (role, content) pairs for the conversation BEFORE the current message.

    The current (last) user message is excluded. Roles are normalized to
    'user'/'assistant'. Used to contextualize follow-up questions.
    """
    messages = state.get("messages", []) or []
    pairs: list[tuple[str, str]] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = "user" if msg.get("role", "user") in ("human", "user") else "assistant"
            content = msg.get("content", "")
        else:
            mtype = getattr(msg, "type", "") or getattr(msg, "role", "")
            role = "user" if mtype in ("human", "user") else "assistant"
            content = getattr(msg, "content", "")
        if content:
            pairs.append((role, content))
    # Drop the last user turn (the current message) so only PRIOR context remains
    if pairs and pairs[-1][0] == "user":
        pairs = pairs[:-1]
    return pairs[-max_turns:]


# A message that already names a specific document (doc-type + number, or an
# explicit document number) is self-contained w.r.t. its subject. Follow-up
# condensation must NOT run on it — otherwise a small model merges a DIFFERENT
# document from the prior turn into it (e.g. asking "Nghị định 53 ..." right after
# "Nghị định 85 ..." leaked ND 85's subject into the ND 53 query → wrong answer).
_DOC_REF_PATTERNS = [
    re.compile(r'\b\d+\s*/\s*\d{2,4}\s*/\s*[\w\-]+', re.IGNORECASE),   # 53/2022/NĐ-CP
    re.compile(r'\bsố\s+\d+', re.IGNORECASE),                          # số 361
    re.compile(
        r'(luật|bộ\s*luật|nghị\s*định|nghị\s*quyết|thông\s*tư(?:\s*liên\s*tịch)?|'
        r'quyết\s*định|pháp\s*lệnh|chỉ\s*thị)\b[^\n]{0,25}?\d',
        re.IGNORECASE,
    ),  # "Nghị định 53", "Thông tư 15", "Luật ... 2018"
]


def _has_explicit_doc_reference(message: str) -> bool:
    """True if the message itself names a specific document (type+number or an
    explicit document number) — i.e. it does NOT need prior-turn context to know
    which document is meant."""
    return any(p.search(message or "") for p in _DOC_REF_PATTERNS)


# Anaphora / continuation cues that signal a question LEANS on prior turns to be
# understood ("nó", "này", "vừa rồi", "còn ... thì sao", "văn bản trên"…).
_FOLLOWUP_CUES = re.compile(
    r"(?<!\w)(nó|này|đó|đấy|ấy|kia|nêu\s+trên|ở\s+trên|trên\s+đây|theo\s+đó|"
    r"vừa\s+(rồi|nêu|xong|đề\s+cập)|vậy(\s+còn)?|thế(\s+còn)?|còn\s+lại|tương\s+tự|"
    r"trường\s+hợp\s+(này|đó)|văn\s+bản\s+(này|đó|trên|ấy)|điều\s+(này|đó)|"
    r"quy\s+định\s+(này|đó)|cái\s+(này|đó|kia))(?!\w)",
    re.IGNORECASE | re.UNICODE,
)


def _needs_context(message: str) -> bool:
    """Heuristic: does this question DEPEND on prior turns to be understood?

    Returns True only for genuinely dependent follow-ups — those carrying
    anaphora / continuation cues, or very short fragments that lack their own
    subject. A self-contained question (its own clear topic, no anaphora) returns
    False so the follow-up condenser LEAVES IT ALONE instead of injecting the
    previous turn's document/subject into it — which is what made a standalone
    question like "Dịch vụ giám sát an ninh mạng là gì" get rewritten to
    "... theo Luật An ninh mạng 2018" and then resolve to the wrong document.
    """
    m = (message or "").strip()
    if not m:
        return False
    if _FOLLOWUP_CUES.search(m):
        return True
    # Very short fragments ("thời hạn bao lâu?") tend to be elliptical — they rely
    # on an implicit subject established earlier. Longer questions are assumed to
    # carry their own subject and are left untouched.
    word_count = len([w for w in re.split(r"\s+", m) if w])
    return word_count <= 4


# Tag the frontend embeds in a message when a workspace document is @mentioned
# (see ChatPanel.tsx). Prior turns keep it in their stored text even after the
# user drops the @mention on a later turn — that is what makes carry-forward work.
_DOC_ID_TAG_RE = re.compile(r"<document_id=([^>]+)>", re.IGNORECASE)


# Transformation / extraction requests that operate on an *implicit* source
# document ("cho tôi bảng biểu …", "liệt kê …", "tóm tắt …", "thời hạn …"). They
# carry no subject of their own, so when a document was just attached they almost
# always mean "do this to THAT document".
_DOC_TRANSFORM_CUES = re.compile(
    r"(?<!\w)("
    r"bảng\s*(biểu)?|lập\s*bảng|tạo\s*bảng|kẻ\s*bảng|"
    r"liệt\s*kê|danh\s*sách|danh\s*mục|"
    r"tóm\s*(tắt|lược|gọn)|rút\s*gọn|"
    r"trích\s*(dẫn|xuất)?|"
    r"thống\s*kê|"
    r"so\s*sánh|đối\s*chiếu|"
    r"sơ\s*đồ|biểu\s*đồ|lưu\s*đồ|"
    r"mục\s*lục|bố\s*cục|dàn\s*ý|cấu\s*trúc|"
    r"các\s*bước|quy\s*trình|trình\s*tự|"
    r"thời\s*hạn|mốc\s*thời\s*gian|deadline|"
    r"nội\s*dung|ý\s*chính|điểm\s*chính|ý\s*nghĩa|"
    r"viết\s*lại|giải\s*thích|"
    r"gồm\s*(những|các)?\s*gì|có\s*(những|các)?\s*gì"
    r")(?!\w)",
    re.IGNORECASE | re.UNICODE,
)


# Phrases that explicitly broaden the search back to the whole corpus — when the
# user says these, do NOT inherit the attached document; let open RAG run.
_CORPUS_BROADENING_CUES = re.compile(
    r"(tất\s*cả\s*(các\s*)?(văn\s*bản|tài\s*liệu)|"
    r"(văn\s*bản|tài\s*liệu)\s*(nào|khác)|"
    r"toàn\s*bộ\s*(văn\s*bản|tài\s*liệu)|"
    r"trong\s*kho|"
    r"tìm\s*(kiếm\s*)?(các\s*)?(văn\s*bản|tài\s*liệu))",
    re.IGNORECASE | re.UNICODE,
)


def _recent_attached_doc_ids(state: SupervisorState, max_turns: int = 6) -> list[str]:
    """Most-recently attached document_id(s) found in prior USER turns.

    Scans prior history (newest first) for the ``<document_id=…>`` tags the
    frontend embeds when a file is @mentioned, and returns the ids from the most
    recent turn that carried any. Empty list when nothing was ever attached.
    """
    prior = _get_prior_history(state, max_turns=max_turns)
    for role, content in reversed(prior):
        if role != "user":
            continue
        ids = [s.strip() for s in _DOC_ID_TAG_RE.findall(content or "") if s.strip()]
        if ids:
            return ids
    return []


async def _recent_session_doc_ids(session_id) -> list[str]:
    """Most-recently attached document_id(s) for this session, read from DB.

    Reads ``chat_messages.document_ids`` (newest user turn first), which is
    populated for BOTH @mentioned workspace docs AND directly-uploaded chat files
    — unlike the in-message ``<document_id=…>`` tag, which only the @mention path
    emits (so the tag-scan in ``_recent_attached_doc_ids`` misses uploaded files).
    Also survives the history window (state["messages"] is capped at 10 turns).
    Best-effort: returns [] when there is no DB, no session, or on any error.
    """
    if not session_id:
        return []
    try:
        from app.services.agent.streaming import get_current_db
        from app.models.chat_message import ChatMessage
        from sqlalchemy import select
        import uuid as _uuid

        db = get_current_db()
        if db is None:
            return []
        try:
            sid = _uuid.UUID(str(session_id))
        except (ValueError, TypeError):
            sid = session_id
        rows = await db.execute(
            select(ChatMessage.document_ids)
            .where(ChatMessage.session_id == sid, ChatMessage.role == "user")
            .order_by(ChatMessage.created_at.desc())
            .limit(8)
        )
        for (doc_ids_json,) in rows.all():
            ids = [str(d).strip() for d in (doc_ids_json or []) if str(d).strip()]
            if ids:
                return ids
    except Exception as e:
        logger.debug(f"[supervisor] _recent_session_doc_ids failed: {e}")
    return []


def _is_doc_followup_request(message: str) -> bool:
    """Smart heuristic for the sticky-document carry-forward.

    True when the current question reads as a CONTINUATION / transformation of an
    already-attached document rather than a brand-new, self-contained topic. Used
    only to decide whether to inherit the most-recently attached document_id when
    the request itself carries none (the user dropped the @mention on a follow-up).
    Conservative on purpose: a question that names a *specific* document, or that
    explicitly broadens to the whole corpus, is left to the normal open flow.
    """
    m = (message or "").strip()
    if not m:
        return False
    if _has_explicit_doc_reference(m):  # names a (likely different) specific doc
        return False
    if _CORPUS_BROADENING_CUES.search(m):  # user is widening the search on purpose
        return False
    # Conservative safety net. The frontend is now the AUTHORITY on attached-file
    # scope: while a quoted file is active it ships its document_id on EVERY turn
    # (persistent chips), and clearing the chip means the user wants open search.
    # So an empty document_ids reaching here usually means "no active scope on
    # purpose" — we must NOT re-impose a stale doc. We only nudge back onto the
    # last attached doc for clearly dependent follow-ups (anaphora / short ellipsis
    # / an explicit transformation over an implicit source), which covers the case
    # where the frontend scope was genuinely lost (hard reload, history replay).
    return _needs_context(m) or bool(_DOC_TRANSFORM_CUES.search(m))


async def _condense_followup_query(message: str, prior: list[tuple[str, str]]) -> str:
    """Rewrite a follow-up question into a self-contained query using prior turns.

    Uses the small memory agent (gemma-4-E4B). A bare follow-up such as "thời hạn trả
    kết quả là bao lâu?" loses the subject established earlier (e.g. a specific tax
    document), so RAG search drifts to unrelated documents. Returns the original
    message unchanged when there is no prior history or on any failure.
    """
    if not prior:
        return message
    try:
        from app.services.llm import get_memory_agent
        from app.services.llm.types import LLMMessage as _LLMMsg
        import re as _re_condense

        lines: list[str] = []
        for role, content in prior:
            speaker = "Người dùng" if role == "user" else "Trợ lý"
            lines.append(f"{speaker}: {str(content)[:800]}")
        transcript = "\n".join(lines)
        if not transcript:
            return message

        agent = get_memory_agent()
        user_content = (
            f"LỊCH SỬ HỘI THOẠI:\n{transcript}\n\n"
            f"CÂU HỎI HIỆN TẠI: {message}\n\n"
            "Câu hỏi đã viết lại (độc lập, tự chứa ngữ cảnh):"
        )
        result = await agent.acomplete(
            [_LLMMsg(role="user", content=user_content)],
            system_prompt=_CONDENSE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=256,
            think=False,
        )
        rewritten = result if isinstance(result, str) else getattr(result, "content", "")
        rewritten = _re_condense.sub(r"<think>.*?</think>", "", rewritten or "", flags=_re_condense.DOTALL)
        rewritten = rewritten.strip().strip('"').strip()
        return rewritten or message
    except Exception as e:
        logger.warning(f"[supervisor] Follow-up condense failed, using original: {e}")
        return message


def _parse_supervisor_response(raw: str) -> dict:
    """Parse LLM JSON response with fallbacks.

    Extracts task_plan (Phase 4) and computes pending_intent from it.
    After parsing, enforces intent→agent agreement: if the intent maps to
    a different agent than what the LLM chose, the intent-based mapping wins.
    """
    raw = raw.strip()

    # Strip thinking tags (thinking-capable models with reasoning enabled)
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

# Detects 2+ section references joined by a conjunction: "Điều 5 và Điều 7",
# "Chương II với Chương III", "Điều 3 đến Điều 5". A single nested reference
# ("Chương II Khoản 3") has no conjunction between two section heads → won't
# match, so it stays fast-path. Used by query_analyzer_node to invoke the LLM
# for multi_section decomposition (the fast-path heuristic otherwise misses it).
_MULTI_SECTION_PATTERN: re.Pattern[str] = re.compile(
    r"(?:điều|chương|khoản|mục|điểm)\s+\S+.{0,40}?"
    r"\b(?:và|với|so\s+với|đến|,)\b"
    r".{0,20}?(?:điều|chương|khoản|mục|điểm)\s+\S+",
    re.IGNORECASE | re.UNICODE,
)

# A leaked leading plan line ("KẾ HOẠCH: <ý> [đã có] | ...") — the ReAct tool-turn
# protocol (react_prompt rule 8) requires this line WITH tool calls; it must never
# open the user-facing answer. Definition lives next to the prompt that mandates
# it so prod and the prompt-eval suite strip/measure the SAME pattern.
from app.prompts.agents.react_prompt import PLAN_LINE_RE as _PLAN_LINE_RE

# Detects a person identifier (CCCD/BHXH: 9-12 digits, or phone: 0 + 9-10 digits).
# On its own this is a pure single-agent people lookup; combined with a SECOND
# task (a named document or a "và" conjunction) it signals a cross_agent query
# that the LLM analyzer must decompose. Used by query_analyzer_node.
_PERSON_ID_PATTERN: re.Pattern[str] = re.compile(
    r"(?<!\d)(?:0\d{9,10}|\d{9,12})(?!\d)",
    re.UNICODE,
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
    # ── What makes a query "complex" enough to invoke the LLM analyzer ──
    # (fast-path skips the LLM for everything else). Three shapes, each one a
    # decomposition the fast-path would otherwise silently collapse to "simple":
    #   1. multi_doc     : "so sánh…", "Luật X và Luật Y"
    #   2. multi_section : "Điều 5 và Điều 7" (2+ sections joined)
    #   3. cross_agent   : a person id (CCCD/BHXH/phone) + a 2nd task (named doc / "và")
    looks_multi_doc = _MULTI_DOC_PATTERN.search(msg_lower) is not None
    looks_multi_section = _MULTI_SECTION_PATTERN.search(msg_lower) is not None
    has_person_id = _PERSON_ID_PATTERN.search(user_message) is not None
    has_named_doc = _NAMED_DOC_PATTERN.search(msg_lower) is not None
    looks_cross_agent = has_person_id and (
        has_named_doc or re.search(r"\bvà\b", msg_lower) is not None
    )
    looks_complex = looks_multi_doc or looks_multi_section or looks_cross_agent

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

    if not looks_complex:
        # needs_comparison alone does NOT warrant the LLM analyzer: the regex above
        # already set the flag, and observed runs spent ~6s only to come back with
        # complexity='simple' / 1 sub_query (unused). Only multi_doc / multi_section /
        # cross_agent shapes need actual LLM decomposition.
        if needs_comparison:
            logger.info("[query_analyzer] Fast-path: comparison flag set by regex, skipping LLM")
        else:
            logger.info("[query_analyzer] Fast-path: simple query (no complex heuristic match), skipping LLM")
        return {"query_complexity": "simple", "sub_queries": None, "extracted_params": None, "needs_comparison": needs_comparison}

    logger.info(
        "[query_analyzer] Complex heuristic matched — invoking LLM "
        f"(multi_doc={looks_multi_doc}, multi_section={looks_multi_section}, "
        f"cross_agent={looks_cross_agent}, needs_comparison={needs_comparison})"
    )

    try:
        from app.services.llm.types import LLMMessage as _LLMMsg
        from app.prompts.agents.query_analyzer_prompt import _QUERY_ANALYZER_PROMPT
        from app.services.llm import get_memory_agent

        # Use the memory agent (gemma-4-E4B, served as `qwen-memory`) for complex structured extraction
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

    # ── Sticky attached-document scope (first pass only) ──────────────────
    # A doc-Q&A session attaches a file via the `<document_id=…>` tag, but that
    # tag is only present on turns where the user keeps the @mention. On a bare
    # follow-up ("cho tôi bảng biểu về thời hạn các công việc") the tag is gone,
    # so request.document_ids is empty, RAG drifts to the whole workspace and the
    # agent ends up asking "which document?". When the follow-up clearly continues
    # on / transforms the attached doc, inherit the most-recently attached
    # document_id(s) from prior turns so scoping behaves as if it were re-mentioned.
    # Mutating state in place lets every document_ids read in this node (routing,
    # doc-resolution, the final return) see the inherited scope consistently.
    if iterations == 0 and not state.get("document_ids"):
        # Source priority, strongest signal first. Only USER-attached docs count —
        # we deliberately do NOT infer scope from the previous answer's citations,
        # because the frontend owns scope (persistent quote chips) and inferring a
        # doc the user never attached would silently override that explicit model.
        #  1. user-attached docs persisted on chat_messages.document_ids (covers
        #     @mentioned docs AND uploaded chat files; survives the history window)
        #  2. <document_id=…> tags scanned from in-state history text (fallback
        #     when DB is unavailable / the turn isn't persisted yet)
        _sid = state.get("session_id")
        _carried = (
            await _recent_session_doc_ids(_sid)
            or _recent_attached_doc_ids(state)
        )
        if _carried and _is_doc_followup_request(user_message):
            state["document_ids"] = _carried
            logger.info(
                f"[supervisor] Sticky doc scope: inherited document_ids={_carried} "
                f"from prior turn for follow-up {user_message!r}"
            )

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

        # ── Follow-up contextualization (first pass only) ─────────────────
        # A bare follow-up ("thời hạn trả kết quả là bao lâu?") drops the subject
        # established earlier (e.g. a specific tax document), so RAG search drifts
        # to unrelated documents. Rewrite it into a self-contained query using the
        # prior turns. Runs once (iterations==0) and never on abbreviation loop-back.
        # was_modified=True makes it flow into expanded_query/rewritten_query below,
        # so downstream RAG search + classification both use the contextualized query.
        _condense_eligible = (
            iterations == 0
            and not expanded
            and not _has_explicit_doc_reference(query_for_classifier)
            and _needs_context(query_for_classifier)
        )
        if iterations == 0 and not expanded and not _condense_eligible:
            logger.info(
                f"[supervisor] Skip follow-up condensation — query is self-contained "
                f"(explicit doc ref or no follow-up cue): {query_for_classifier!r}"
            )
        if _condense_eligible:
            _prior = _get_prior_history(state)
            if _prior:
                _contextualized = await _condense_followup_query(query_for_classifier, _prior)
                if _contextualized and _contextualized.strip() != query_for_classifier.strip():
                    logger.info(
                        f"[supervisor] Follow-up contextualized: "
                        f"{query_for_classifier!r} -> {_contextualized!r}"
                    )
                    query_for_classifier = _contextualized.strip()
                    expanded_message = query_for_classifier
                    was_modified = True

        # Phase 4: Use Qwen3.6-35B for plan-aware classification
        # We check the OLLAMA_HOST to decide whether to use native Ollama
        # or OpenAI-compatible provider (e.g. vLLM serving the 35B model).
        from app.core.config import settings
        from app.services.agent.langfuse_tracing import trace_llm
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
        # Langfuse: trace the intent-classification LLM call as a generation
        classifier = trace_llm(classifier, label="supervisor_classifier")
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

        # ── Dataset trace: record the supervisor routing decision ─────────────
        try:
            from app.services.agent.trace_collector import get_collector

            _coll = get_collector()
            if _coll is not None:
                _coll.add_routing(
                    node="supervisor",
                    intent=str(result.get("intent")) if result.get("intent") else None,
                    next_agent=str(result.get("next_agent")) if result.get("next_agent") else None,
                    needs_memory=bool(result.get("needs_memory")),
                    search_mode=str(result.get("search_mode")) if result.get("search_mode") else None,
                    task_plan=result.get("task_plan") or [],
                    pending_intent=str(result.get("pending_intent")) if result.get("pending_intent") else None,
                    query_complexity=str(state.get("query_complexity", "simple")),
                    reasoning=decision.get("reasoning") if isinstance(decision, dict) else None,
                    user_message=user_message,
                    expanded_query=expanded_message,
                )
        except Exception:
            pass

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
    from app.core.config import settings
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage
    from app.services.agent.streaming import push_event

    provider = get_llm_provider()
    messages = state.get("messages", [])
    system_prompt = state.get("system_prompt", "")
    user_memory = state.get("user_memory_context", "")

    # Deterministic "saved to memory" notice — extract personal facts from the
    # user's message in parallel (the memory worker persists the same facts every
    # turn). A pure question ("tôi tên gì?") yields [] → no notice.
    import asyncio as _aio_dm
    from app.services.graphiti_client import extract_personal_facts as _extract_facts_dm
    _dm_user_msg = _extract_user_message(state)
    _dm_fact_task = (
        _aio_dm.create_task(_extract_facts_dm(_dm_user_msg)) if _dm_user_msg else None
    )

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

    # Buffer the full answer, THEN sanitize + replay. Buffering lets us strip
    # leaked source labels ("Source: Memory", "[Memory]") and fabricated
    # citations before any token reaches the client, and keeps [MEM-1] markers
    # whole (never split across token events → no flickering memory icon).
    answer_parts = []
    try:
        async for chunk in provider.astream(
            messages=llm_messages,
            temperature=0.5,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            system_prompt=effective_system,
            think=False,
        ):
            if chunk.text:
                answer_parts.append(chunk.text)
    except Exception as e:
        logger.error(f"[direct_answer] LLM streaming failed: {e}")
        answer_parts = ["Xin chào! Tôi có thể giúp gì cho bạn?"]

    from app.services.agent.nodes import (
        strip_thinking_tags,
        sanitize_citations,
        _stream_final_answer,
        build_memory_saved_notice,
    )
    final_answer = strip_thinking_tags("".join(answer_parts))
    # No RAG sources on this path → empty valid set; sanitize keeps
    # [MEM-N]/[IMG-…] and strips leaked labels + fabricated markers.
    final_answer = sanitize_citations(final_answer, set())
    if _dm_fact_task is not None:
        try:
            _dm_facts = await _dm_fact_task
        except Exception:
            _dm_facts = []
        final_answer = final_answer + build_memory_saved_notice(_dm_facts)
    await _stream_final_answer(state, final_answer)

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
        # result_evaluator_node only ever sets next_agent=RAG on the retry path
        # (every fallback intent maps to RAG), so retries always go to "rag".
        target = "rag"

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

async def _format_mongo_block(state: SupervisorState) -> str:
    """Render the person-record block (Block 1) via a lightweight LLM call.

    Streams tokens as it goes and returns the formatted text. Falls back to the
    raw MongoDB display if the formatting call fails.
    """
    from app.core.config import settings
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage as _LLMMsg
    from app.services.agent.streaming import push_event
    from app.services.agent.nodes import strip_thinking_tags

    existing_final = state.get("final_answer", "") or ""

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
            messages=mongo_messages, temperature=0.1, max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        ):
            if chunk.type == "text" and chunk.text:
                await push_event(state, "token", chunk.text)
                answer_parts.append(chunk.text)
        return strip_thinking_tags("".join(answer_parts))
    except Exception as e:
        logger.error(f"[mongo_formatter_node] mongo block format failed: {e} — using raw")
        await push_event(state, "token", existing_final)
        return existing_final


async def _format_doc_block(state: SupervisorState) -> str:
    """Render the related-documents block (Block 2) over the companion RAG search.

    Reuses the full answer_generator (citations, source formatting, streaming)
    but on an ISOLATED sub-state so it only sees the document sources — not the
    MongoDB person record (which lives in kg_summaries/final_answer). Returns ""
    when there are no document sources to render.
    """
    from app.services.agent.nodes import answer_generator as _orig_ag
    from app.services.agent.state import DEFAULT_STATE

    sources = state.get("sources") or []
    if not sources:
        return ""

    # Build an isolated state: document sources + their KG, NOT the mongo block.
    agent_state = {**DEFAULT_STATE}
    for k, v in state.items():
        if v is not None or k not in DEFAULT_STATE:
            agent_state[k] = v
    agent_state["sources"] = sources
    agent_state["kg_summaries"] = state.get("people_doc_kg", []) or []
    agent_state["mongo_results"] = []
    agent_state["intent"] = "search"
    agent_state["final_answer"] = None
    agent_state["tool_called"] = True
    agent_state["existing_citation_ids"] = {}
    agent_state["citation_map"] = {}

    result = await _orig_ag(agent_state)
    return (result.get("final_answer") or "").strip()


async def mongo_formatter_node(state: SupervisorState) -> dict:
    """
    Two-block formatter for the people path.

    Block 1 — person record(s) from MongoDB, via a lightweight ~1KB prompt.
    Block 2 — related documents found by the companion RAG search
              (people_doc_search_node), rendered with citations.

    Every person lookup (CCCD / phone / name / …) runs the document search too,
    because the same identifier or name commonly appears in indexed documents.
    The two blocks are kept visually separate per product decision.
    """
    from app.services.agent.streaming import push_event

    logger.info(f"[LANGGRAPH_NODE] Entering mongo_formatter_node")

    existing_final = state.get("final_answer", "")
    has_docs = bool(state.get("sources"))

    if not existing_final and not has_docs:
        return {
            "final_answer": "Không tìm thấy dữ liệu.",
            "next_agent": AgentType.FINISH,
        }

    await push_event(state, "status", {"step": "generating", "detail": "Đang trình bày kết quả..."})

    # ── Block 1: person record(s) ────────────────────────────────────────────
    if existing_final:
        block1 = await _format_mongo_block(state)
    else:
        block1 = "Không tìm thấy hồ sơ người dân phù hợp."
        await push_event(state, "token", block1)

    # ── Block 2: related documents (companion RAG search) ────────────────────
    block2 = ""
    if has_docs:
        header = "\n\n---\n\n### 📄 Tài liệu liên quan\n\n"
        await push_event(state, "token", header)
        try:
            block2_body = await _format_doc_block(state)
        except Exception as e:
            logger.error(f"[mongo_formatter_node] doc block format failed: {e}")
            block2_body = ""
        if block2_body:
            block2 = header + block2_body
        else:
            note = "_Không tìm thấy tài liệu liên quan._"
            await push_event(state, "token", note)
            block2 = header + note

    final = (block1 + block2).strip()
    return {"final_answer": final, "next_agent": AgentType.FINISH}


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
    # Deterministic "saved to memory" notice: extract durable personal facts from
    # the user's message IN PARALLEL with the tool loop (~0 added wall-clock). The
    # memory worker persists the same facts every turn (add_conversation_episode),
    # so we surface the notice from this extraction instead of relying on the model
    # calling save_memory (which it does inconsistently). Read-only — no double save.
    from app.services.graphiti_client import extract_personal_facts as _extract_facts
    _auto_fact_task = (
        asyncio.create_task(_extract_facts(user_message)) if user_message else None
    )
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
    # Inject the query_analyzer plan (sub_queries) as an explicit checklist so
    # the loop knows every sub-question it must cover; also feeds the judge.
    _plan = state.get("sub_queries") or None
    system_prompt = build_react_system_prompt(
        state.get("user_memory_context", "") or "",
        plan=_plan,
        extracted_params=state.get("extracted_params"),
    )
    # Reasoning-token / judge knobs (see config.py NEXUSRAG_REACT_*).
    think_tools = bool(getattr(settings, "NEXUSRAG_REACT_THINK_TOOL_TURNS", True))
    judge_on = bool(getattr(settings, "NEXUSRAG_REACT_JUDGE", True))
    max_reflections = int(getattr(settings, "NEXUSRAG_REACT_MAX_REFLECTIONS", 2))
    # Plan lines reused by the judge to check completeness.
    _plan_lines = [
        (sq.get("query") or "").strip()
        for sq in (_plan or [])
        if isinstance(sq, dict) and (sq.get("query") or "").strip()
    ]
    if _plan_lines:
        logger.info(
            f"[react_executor] plan injected ({len(_plan_lines)} steps): {_plan_lines}"
        )
    logger.info(
        f"[react_executor] think_tools={think_tools} judge={judge_on} "
        f"max_reflections={max_reflections}"
    )
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

    async def _run_turn(use_tools: bool, think: bool = False):
        """Run one LLM turn, BUFFERING its tool calls and answer text.

        We deliberately do NOT stream text to the client speculatively. In
        native tool-calling mode the model sometimes writes prose BEFORE emitting
        a tool call, and the function-call chunks only arrive at the END of the
        stream — so there is no reliable way to tell, mid-stream, whether the
        text being produced is the final answer or a throw-away preamble.
        Streaming it live and then retracting it with ``token_rollback`` is
        exactly what made the answer flash on screen and vanish on tool steps.
        Instead we buffer here and let ``_finish`` replay the finalised answer
        progressively (see ``_stream_out``) — nothing is ever taken back, so
        there is no flicker.

        Emits a one-off "composing" status the moment answer text starts arriving
        (tool-only turns produce little/no text), so the user isn't left staring
        at a stale "searching" status while the answer is generated.
        """
        calls, text = [], ""
        announced = False
        async for c in llm.astream(
            msgs,
            system_prompt=system_prompt,
            tools=RAG_TOOL_SCHEMAS if use_tools else None,
            tool_choice="auto" if use_tools else None,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            think=think,
        ):
            if c.type == "function_call" and c.function_call:
                calls.append(c.function_call)
            elif c.type == "thinking" and c.text:
                # Surface the model's reasoning (why it picks a tool / how it
                # synthesises) to the client, same channel as answer_generator.
                await push_event(state, "thinking", c.text)
            elif c.type == "text" and c.text:
                text += c.text
                # Threshold: tool turns now legitimately open with a short
                # `KẾ HOẠCH:` line (react_prompt rule 8) BEFORE their function
                # calls arrive — don't flip the status to "composing" for it.
                # Real answers blow past 200 chars almost immediately.
                if not announced and not calls and len(text) > 200:
                    announced = True
                    await push_event(state, "status", {"step": "composing", "detail": "Đang soạn câu trả lời..."})
        return calls, text

    async def _run_streaming_synthesis() -> str:
        """Run the FINAL no-tools synthesis turn, streaming sanitised text LIVE.

        Only safe on the last turn (step/reflection budget spent): nothing
        downstream can retract streamed text — the judge, if it still runs,
        only APPENDS a caveat, never rewrites. This cuts the buffered-draft →
        judge → replay tail (~6-10s on observed traces) out of the perceived
        latency: the user starts reading while the judge is still scoring.

        Each chunk goes through _CitationSafeStreamer (a ``[marker]`` is never
        split across token events) and then sanitize_citations (invented
        markers are dropped BEFORE they render). Returns the accumulated text
        exactly as the client saw it.
        """
        from app.services.agent.nodes import _CitationSafeStreamer, sanitize_citations

        _valid = {str(getattr(s, "index", "")) for s in sources}
        _valid.discard("")
        streamer = _CitationSafeStreamer()
        streamed = ""
        announced = False

        async def _emit(piece: str) -> None:
            nonlocal streamed
            if not piece:
                return
            clean = sanitize_citations(piece, _valid)
            if clean:
                await push_event(state, "token", clean)
                streamed += clean

        # Head holdback: buffer the FIRST line so a leaked leading `KẾ HOẠCH:`
        # line (tool-turn protocol, react_prompt rule 8) is stripped BEFORE
        # anything reaches the client — streamed text can never be retracted.
        head = ""
        head_done = False

        async def _feed_out(piece: str, final_flush: bool = False) -> None:
            nonlocal head, head_done
            if head_done:
                await _emit(piece)
                return
            head += piece or ""
            _is_planish = head.lstrip().upper().startswith("KẾ HOẠCH")
            if "\n" in head or final_flush or (len(head) > 160 and not _is_planish):
                head_done = True
                await _emit(_PLAN_LINE_RE.sub("", head))
                head = ""

        async for c in llm.astream(
            msgs,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            think=False,
        ):
            if c.type == "thinking" and c.text:
                await push_event(state, "thinking", c.text)
            elif c.type == "text" and c.text:
                if not announced:
                    announced = True
                    await push_event(state, "status", {"step": "composing", "detail": "Đang soạn câu trả lời..."})
                await _feed_out(streamer.feed(c.text))
        await _feed_out(streamer.flush(), final_flush=True)
        return streamed.strip()

    async def _stream_out(text: str) -> None:
        """Replay a finalised answer to the client as progressive ``token`` events.

        No speculative streaming happened upstream, so there is nothing to roll
        back — ``text`` is already cleaned + sanitised. We pace the replay
        (~0.3s total, bounded event count) so it renders with a typing feel
        instead of landing in one block, without any ``token_rollback`` flicker.
        """
        if not text:
            return
        import asyncio as _aio
        parts = re.findall(r"\S+\s*", text) or [text]
        max_emits = 100
        group = max(1, (len(parts) + max_emits - 1) // max_emits)
        n_emits = (len(parts) + group - 1) // group
        delay = min(0.02, 0.3 / max(1, n_emits))
        buf = ""
        for i, p in enumerate(parts):
            buf += p
            if (i + 1) % group == 0:
                await push_event(state, "token", buf)
                buf = ""
                await _aio.sleep(delay)
        if buf:
            await push_event(state, "token", buf)

    # Last judge verdict, surfaced on the returned state for tracing (see below).
    _last_verdict: dict = {}
    # Tool rounds actually executed — used to skip the judge on trivial queries
    # (no analyzer plan + a single search round): the ~5s judge pass adds little
    # there, and sanitize_citations already drops invented citation markers.
    tool_rounds = 0
    # Personal facts persisted via save_memory this turn — appended as a notice
    # at the end of the answer so the user knows their info was remembered.
    saved_facts: list[str] = []

    # Transparency caveat appended when the answer is finalised on a 'revise'
    # verdict (judge unsatisfied but budget spent) — shared by the buffered
    # (_finish) and live-streamed (forced synthesis) paths.
    _REVISE_CAVEAT = (
        "\n\n> ⚠️ *Lưu ý: câu trả lời có thể chưa bao quát đầy đủ và một số "
        "trích dẫn số hiệu Điều/Khoản cần được đối chiếu lại với văn bản gốc "
        "trước khi áp dụng.*"
    )

    async def _memory_notice() -> str:
        """Notice about remembered personal info. Prefer facts the model explicitly
        saved via save_memory; otherwise fall back to the deterministic parallel
        extraction (the worker persists these regardless) so the notice appears
        even when the model forgot to call the tool."""
        from app.services.agent.nodes import build_memory_saved_notice
        _notice_facts = list(saved_facts)
        if _auto_fact_task is not None:
            try:
                _auto = await _auto_fact_task
            except Exception:
                _auto = []
            if not _notice_facts:
                _notice_facts = _auto
        return build_memory_saved_notice(_notice_facts)

    async def _finish(answer_text: str) -> dict:
        # Sources/images first so the client has citation data before the answer.
        await _emit_artifacts()
        from app.services.agent.nodes import sanitize_citations
        final = strip_thinking_tags(answer_text or "").strip()
        # Drop a leaked leading plan line (tool-turn protocol, react_prompt rule 8).
        final = _PLAN_LINE_RE.sub("", final).strip()
        # Drop citation markers the LLM invented (e.g. [75a75810] derived from a
        # document UUID) that map to no real source — else they leak as raw text.
        _valid_cids = {str(getattr(s, "index", "")) for s in sources}
        _valid_cids.discard("")
        final = sanitize_citations(final, _valid_cids)
        if not final:
            final = "Xin lỗi, tôi chưa tạo được câu trả lời từ kho văn bản."
        # Transparency caveat: when the answer is finalised on a 'revise' verdict
        # (the judge was NOT satisfied but the reflection/step budget is spent),
        # warn the user instead of presenting possibly-ungrounded article/section
        # numbers as authoritative. Only fires on 'revise' — a 'pass' (or a judge
        # that failed and fail-opened to pass) never adds this.
        if (_last_verdict or {}).get("verdict") == "revise":
            final = final + _REVISE_CAVEAT
        final = final + await _memory_notice()
        # Replay the finalised, sanitised answer (no speculative tokens were sent,
        # so this is the ONLY thing the client ever renders → zero flicker).
        await _stream_out(final)
        logger.info(
            f"[react_executor] done: {len(sources)} sources, answer={len(final)} chars, "
            f"judge={_last_verdict.get('verdict')!r} score={_last_verdict.get('score')}"
        )
        return {
            "final_answer": final,
            "sources": sources,
            "images": images,
            "document_ids": ctx.document_ids,
            "next_agent": AgentType.FINISH,
            "judge_verdict": _last_verdict.get("verdict"),
            "judge_score": _last_verdict.get("score"),
            "judge_feedback": _last_verdict.get("feedback"),
        }

    def _sources_digest(limit: int = 12) -> str:
        """Compact rendering of collected sources for the judge to check grounding."""
        lines = []
        for s in sources[:limit]:
            idx = getattr(s, "index", "") or "?"
            loc = getattr(s, "source_file", None) or " > ".join(getattr(s, "heading_path", []) or [])
            body = (getattr(s, "content", "") or "").strip().replace("\n", " ")
            lines.append(f"[{idx}] {loc}: {body[:300]}")
        return "\n".join(lines)

    async def _judge(draft: str) -> dict:
        """LLM-as-judge: score the DRAFT against sources + plan. Returns verdict dict.

        Fail-open: any error / unparseable output → treat as ``pass`` so the judge
        can never block a usable answer (it only ever ADDS quality, never breaks).
        """
        from app.prompts.agents.judge_prompt import (
            JUDGE_SYSTEM_PROMPT,
            build_judge_user_prompt,
        )

        user_prompt = build_judge_user_prompt(
            question=user_message or query,
            draft=draft or "",
            sources_digest=_sources_digest(),
            plan_lines=_plan_lines,
        )

        async def _call() -> dict:
            raw = ""
            async for c in llm.astream(
                [_LLMMsg(role="user", content=user_prompt)],
                system_prompt=JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
                # 768 (was 512): observed truncated-JSON failures ("Unterminated
                # string") when the judge wrote long `missing` lists — the prompt
                # now also caps list length, this is the safety margin.
                max_tokens=768,
                think=False,
            ):
                if c.type == "text" and c.text:
                    raw += c.text
            cleaned = strip_thinking_tags(raw or "").strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start:end + 1]
            data = json.loads(cleaned)
            verdict = str(data.get("verdict", "pass")).lower()
            if verdict not in ("pass", "revise"):
                verdict = "pass"
            return {
                "verdict": verdict,
                "score": data.get("score"),
                "missing": data.get("missing") or [],
                "feedback": str(data.get("feedback") or "").strip(),
            }

        # Best-effort Langfuse span, done inline so a tracing hiccup can NEVER
        # break (or double-await) the judge call itself.
        _obs = None
        try:
            _lf = _get_langfuse_client()
            if _lf:
                _obs = _lf.start_observation(
                    name="react_judge",
                    input={"draft_len": len(draft or ""), "sources": len(sources)},
                    level="DEFAULT",
                )
        except Exception:
            _obs = None
        try:
            v = await _call()
            if _obs is not None:
                try:
                    _obs.update(output=v)
                    _obs.end()
                except Exception:
                    pass
            logger.info(
                f"[react_executor] judge verdict={v.get('verdict')!r} "
                f"score={v.get('score')} missing={v.get('missing')}"
            )
            return v
        except Exception as e:
            logger.warning(f"[react_executor] judge failed ({e}) — passing draft through")
            return {"verdict": "pass", "score": None, "missing": [], "feedback": ""}

    async def _judge_and_maybe_finish(draft: str, reflections: int) -> dict | None:
        """Run the judge on a draft. Return a finish-dict when done, else None.

        None means "keep looping" — the caller has already appended the judge's
        critique to ``msgs`` and should continue gathering. When the judge is
        disabled, or passes, or the reflection budget is spent, we finalise.
        """
        nonlocal _last_verdict
        if not judge_on:
            return await _finish(draft)
        if not _plan_lines and tool_rounds <= 1:
            logger.info(
                f"[react_executor] skipping judge (simple query: no plan, "
                f"{tool_rounds} tool round(s))"
            )
            return await _finish(draft)
        verdict = await _judge(draft)
        _last_verdict = verdict
        if verdict["verdict"] == "pass" or reflections >= max_reflections:
            return await _finish(draft)
        # revise + budget remaining → feed critique back and loop for more tools.
        missing = "; ".join(str(m) for m in verdict.get("missing") or [])
        critique = (
            "Bản nháp trên CHƯA đạt. "
            + (f"Còn thiếu: {missing}. " if missing else "")
            + (verdict.get("feedback") or "Hãy tra cứu thêm để bổ sung căn cứ còn thiếu.")
            + " Hãy GỌI THÊM CÔNG CỤ để thu thập thông tin còn thiếu rồi trả lời lại; "
            "chưa được kết luận nếu chưa đủ căn cứ."
        )
        msgs.append(_LLMMsg(role="user", content=critique))
        await push_event(state, "status", {"step": "searching", "detail": "Đang rà soát và bổ sung..."})
        return None

    try:
        seen_calls: set[str] = set()   # (name|args) signatures already executed
        no_progress = 0                # consecutive tool-turns adding no new sources
        reflections = 0                # times the judge sent the draft back for more
        nudged = False                 # sufficiency check injected after first data?
        last_draft: str | None = None  # most recent revised-away draft (for reuse)
        draft_source_count = -1        # sources count when last_draft was written
        for step in range(max_steps):
            await _emit_artifacts()
            calls, text = await _run_turn(use_tools=True, think=think_tools)
            if not calls:
                # Model produced a DRAFT (stopped calling tools). Gate it through
                # the judge before streaming: pass → finish; revise (with budget)
                # → critique is appended to msgs and we loop for more tools.
                done = await _judge_and_maybe_finish(text, reflections)
                if done is not None:
                    return done
                # Judge asked for a revision. Keep this draft + the source count
                # behind it: if the loop later ends WITHOUT gathering anything
                # new, we reuse it instead of regenerating the same answer.
                last_draft = text
                draft_source_count = len(sources)
                reflections += 1
                continue

            # Anti-loop guard: if EVERY tool call this turn repeats an earlier
            # call verbatim, the model is spinning (e.g. re-running the same
            # search_documents) — stop and synthesise from what we already have.
            sigs = [
                f"{fc.get('name', '')}|"
                f"{json.dumps(fc.get('args', {}), sort_keys=True, ensure_ascii=False)}"
                for fc in calls
            ]
            if sigs and all(s in seen_calls for s in sigs):
                logger.warning(
                    f"[react_executor] step {step + 1}: all {len(calls)} call(s) repeat "
                    f"earlier ones — breaking to synthesis"
                )
                break
            seen_calls.update(sigs)

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
            # Don't leak internal tool/function names to the UI — keep the
            # user-facing status generic. Raw names still go to the log.
            await push_event(state, "status", {"step": "searching", "detail": "Đang tra cứu thông tin..."})
            logger.info(
                f"[react_executor] step {step + 1}/{max_steps}: {len(calls)} call(s): {names} "
                f"args={[fc.get('args', {}) for fc in calls]}"
            )

            # Parallel tool execution — independent calls in the same turn run together.
            sources_before = len(sources)
            results = await asyncio.gather(
                *[dispatch_tool(fc.get("name", ""), fc.get("args", {}), ctx) for fc in calls]
            )
            tool_rounds += 1

            stop = False
            for fc, r in zip(calls, results):
                sources.extend(r.get("sources", []) or [])
                images.extend(r.get("images", []) or [])
                msgs.append(_LLMMsg(role="tool", tool_call_id=fc["_id"], content=r.get("summary", "")))
                _data = r.get("data") or {}
                if _data.get("stop"):
                    stop = True
                # Remember facts actually persisted so we can tell the user at the
                # end of the answer that we saved their info.
                if fc.get("name") == "save_memory" and _data.get("saved"):
                    _f = (fc.get("args") or {}).get("fact")
                    if _f:
                        saved_facts.append(str(_f).strip())

            if stop:  # ask_user already emitted clarification → wait for the user
                logger.info("[react_executor] ask_user → stopping loop for clarification")
                return {
                    "final_answer": "",
                    "next_agent": AgentType.FINISH,
                    "clarification_needed": True,
                    "sources": sources,
                    "images": images,
                }

            # Sufficiency gate: the FIRST time a tool round actually returns data,
            # make the model STOP and take stock via an EXPLICIT plan-status
            # checklist (not hidden thinking — reasoning tokens are off on tool
            # turns and aren't fed back across turns anyway; a written plan line
            # persists in msgs, so later turns see what is already covered).
            # This is the dynamic early-exit — easy queries answer in 2-3 steps;
            # only genuinely missing pieces trigger more tools. Measured before
            # this rework: the free-form "ĐÁNH GIÁ đủ chưa?" nudge was ignored in
            # 11/18 runs, mostly by chasing decree numbers that appeared inside
            # retrieved chunks ("Căn cứ Nghị định ...") — hence the explicit ban.
            if not nudged and sources_before == 0 and len(sources) > 0:
                nudged = True
                from app.prompts.agents.react_prompt import SUFFICIENCY_NUDGE_PROMPT
                msgs.append(_LLMMsg(role="user", content=SUFFICIENCY_NUDGE_PROMPT))
                logger.info(
                    f"[react_executor] step {step + 1}: sufficiency check injected "
                    f"({len(sources)} sources)"
                )

            # No-progress guard: once we already have sources, two consecutive
            # tool-turns that surface nothing new mean further searching is
            # unproductive — synthesise from what we have. (resolve_doc and other
            # source-less scoping tools don't trip this since sources_before==0.)
            if sources_before > 0 and len(sources) == sources_before:
                no_progress += 1
                if no_progress >= 2:
                    logger.warning(
                        f"[react_executor] step {step + 1}: {no_progress} no-progress "
                        f"turns — breaking to synthesis ({len(sources)} sources)"
                    )
                    break
            else:
                no_progress = 0

        # Reuse-draft shortcut: if we already have a judged draft and NO new
        # sources were gathered since it was written, regenerating would produce
        # the same answer — return the draft and skip a full synthesis + judge
        # pass (saves ~10s on queries where the reflection found nothing new).
        if last_draft is not None and len(sources) == draft_source_count:
            logger.info(
                f"[react_executor] reusing last draft (no new sources since draft; "
                f"{len(sources)} sources) — skipping redundant synthesis"
            )
            return await _finish(last_draft)

        # Loop exhausted or an anti-loop guard fired → force a final synthesis.
        logger.warning(
            f"[react_executor] forcing synthesis ({len(sources)} sources collected)"
        )
        # Strict attached-file scope: if the user has a quoted/attached file active
        # this turn and did NOT explicitly ask to broaden to the whole corpus, the
        # answer must stay INSIDE that file — when the content isn't there, say so
        # plainly instead of wandering the workspace or guessing from general
        # knowledge. (The whole-corpus path stays available via the user explicitly
        # broadening, which routes here without an attached scope.)
        _attached_scope = bool(ctx.uploaded_document_ids or state.get("document_ids"))
        _broadened = bool(_CORPUS_BROADENING_CUES.search(user_message or ""))
        if _attached_scope and not _broadened:
            synthesis_instr = (
                "Hãy trả lời CHỈ dựa trên nội dung của FILE ĐÍNH KÈM đã thu thập ở trên. "
                "Nếu thông tin người dùng hỏi KHÔNG có trong file đính kèm, hãy trả lời "
                "rõ ràng: \"Nội dung bạn hỏi không có trong file đính kèm.\" — TUYỆT ĐỐI "
                "không suy đoán, không lấy thông tin ngoài file và không mở rộng tìm kiếm "
                "ra toàn kho văn bản."
            )
        else:
            synthesis_instr = (
                "Hãy tổng hợp câu trả lời cuối cùng từ thông tin đã thu thập ở trên. "
                "Nếu chưa đủ căn cứ, nói rõ phần nào còn thiếu."
            )
        # Anti-fabrication guard: this is the LAST turn (budget spent), so the
        # model won't get another judge round to catch invented citations. Force
        # it to stay grounded — the recurring failure is inventing Điều/Khoản
        # numbers not present in the sources.
        synthesis_instr += (
            " QUAN TRỌNG: CHỈ dùng nội dung CÓ trong nguồn đã thu thập; TUYỆT ĐỐI "
            "không bịa hay suy diễn số hiệu Điều/Khoản/văn bản không xuất hiện trong "
            "nguồn. Nếu không chắc mã điều khoản chính xác, hãy diễn đạt chung "
            "(vd \"theo quy định về hành vi bị cấm\") và nêu rõ cần đối chiếu văn bản "
            "gốc, thay vì ghi một mã cụ thể có thể sai."
        )
        # If the last reflection judge flagged concrete gaps, hand them to the
        # synthesis turn so it can address them rather than repeat the mistake.
        _prev_missing = "; ".join(str(m) for m in (_last_verdict or {}).get("missing") or [])
        if (_last_verdict or {}).get("verdict") == "revise" and _prev_missing:
            synthesis_instr = (
                f"Lần đánh giá trước phát hiện vấn đề: {_prev_missing}. "
                "Hãy ưu tiên khắc phục các vấn đề đó khi trả lời. " + synthesis_instr
            )
        msgs.append(_LLMMsg(role="user", content=synthesis_instr))
        await _emit_artifacts()
        # think=False on the FINAL synthesis: after a long tool loop the context
        # is large, and reasoning here would eat the token budget and leave no
        # room for the answer itself (observed: empty answer → canned apology).
        # Tool-decision turns still reason; the answer turn just answers.
        #
        # LIVE streaming: the budget is spent, so a judge 'revise' here can only
        # ever APPEND a caveat (never loop again / retract text) — that makes
        # speculative streaming safe on THIS turn only. The judge then runs on
        # the finished text while the user is already reading it.
        final = await _run_streaming_synthesis()
        if not final:
            # Nothing was streamed — fall back to the buffered path so the
            # canned-apology fallback (and its replay) still applies.
            return await _finish("")
        # One judge pass on the forced synthesis (same skip rule as the draft
        # path: trivial single-round queries don't pay the ~5s judge).
        if judge_on and (_plan_lines or tool_rounds > 1):
            _last_verdict = await _judge(final)
        tail = ""
        if (_last_verdict or {}).get("verdict") == "revise":
            tail += _REVISE_CAVEAT
        tail += await _memory_notice()
        if tail:
            await push_event(state, "token", tail)
            final += tail
        logger.info(
            f"[react_executor] done (live-streamed synthesis): {len(sources)} sources, "
            f"answer={len(final)} chars, judge={_last_verdict.get('verdict')!r} "
            f"score={_last_verdict.get('score')}"
        )
        return {
            "final_answer": final,
            "sources": sources,
            "images": images,
            "document_ids": ctx.document_ids,
            "next_agent": AgentType.FINISH,
            "judge_verdict": _last_verdict.get("verdict"),
            "judge_score": _last_verdict.get("score"),
            "judge_feedback": _last_verdict.get("feedback"),
        }

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
    graph.add_node("people_doc_search", _people_doc_search_wrapper)  # People + RAG
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
    # People path: always run the companion document (RAG) search before
    # formatting, so person lookups also surface documents that mention the
    # identifier/name.  people → people_doc_search → mongo_formatter (two blocks)
    graph.add_edge("people", "people_doc_search")
    graph.add_edge("people_doc_search", "mongo_formatter")
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


    # Extract fact lines from memory context (format: "<header>\n- fact\n- fact")
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

        # Strip <think>...</think> tags the model may emit
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


async def _people_doc_search_wrapper(state: SupervisorState) -> dict:
    """Wrapper that imports and calls people_doc_search_node."""
    logger.info("[LANGGRAPH_NODE] Entering people_doc_search_wrapper")
    from app.services.agents.people_agent import people_doc_search_node

    return await people_doc_search_node(state)


# =============================================================================
# Backward Compatibility - Re-export for existing imports
# =============================================================================

# Keep old SupervisorState name for compatibility during transition
SupervisorStateModel = SupervisorState
create_supervisor = create_supervisor_graph

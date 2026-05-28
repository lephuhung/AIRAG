"""
Answer Generator Instructions — Intent-Specific Modules
========================================================

Replaces the monolithic INSTRUCTIONS string in answer_generator with
modular, intent-specific instruction blocks.

Usage:
    from app.prompts.agents.answer_instructions import get_instructions_for_intent
    instructions = get_instructions_for_intent(intent)

Token savings:
    - RAG intents: ~30% fewer instruction tokens (no mongo/phone rules)
    - Mongo intents: ~40% fewer (no citation/table rules)
    - Abbreviation: ~60% fewer (minimal instructions needed)
    - list_docs: ~50% fewer (simple list output)
"""

from __future__ import annotations

# =============================================================================
# BASE — Always included regardless of intent
# =============================================================================

_BASE_INSTRUCTIONS = (
    "- CRITICAL: Answer the CURRENT question based ONLY on the RETRIEVED CONTEXT above. "
    "Previous conversation messages are provided only for reference continuity (e.g. 'tài liệu này'). "
    "Do NOT reuse, blend, or repeat information from previous assistant answers. "
    "Each question must be answered independently from the retrieved sources.\n"
    "- Answer based ONLY on the retrieved sources above. "
    "If the retrieved context is empty or says 'no results', say so — do NOT fill in details from your own knowledge.\n"
    "- If the sources do not contain enough information to answer fully, "
    "be honest about it. Provide what you can, clearly note what is missing, "
    "and suggest what the user might do next.\n"
    "- If NO sources are relevant or available (context shows 'no results' or is empty), "
    "respond that you cannot find relevant information and ASK the user to provide more details "
    "or clarify their question. Do NOT list or introduce documents in the system. Do NOT fabricate citation IDs.\n"
    "- Keep your tone friendly and helpful, not robotic or overly formal.\n"
    "- End with a brief 1-2 line suggestion for what to explore next, "
    "if appropriate (start with 'Gợi ý:' or 'Suggestion:').\n"
)

# =============================================================================
# RAG — Citation & source-based answering (search, summarize, kg_query, etc.)
# =============================================================================

_RAG_INSTRUCTIONS = (
    "- Cite sources using their unique IDs in brackets, e.g. [a3z9] or [b2m7]. "
    "ONLY cite sources that actually appear in the RETRIEVED CONTEXT above. "
    "Do NOT invent or hallucinate citation IDs.\n"
    "- Memory facts: paraphrase in your own words and cite as [MEM-{id}]. "
    "Do NOT copy facts verbatim.\n"
    "- TABLE DATA: 'Key, Year = Value' pairs are table cells.\n"
    "- LEGAL DOCUMENTS: If sources contain conflicting rules from documents of different years, "
    "ALWAYS prioritize the newest document (latest year/date). You MUST structure your answer to "
    "state the current rule from the newest document first, and then explicitly mention how it "
    "updated or changed from the older document (e.g. 'Theo văn bản mới nhất [A] thì..., "
    "thay đổi so với quy định cũ tại [B] là...').\n"
)

# =============================================================================
# MONGO — People database search (strict anti-hallucination rules)
# =============================================================================

_MONGO_INSTRUCTIONS = (
    "- You have NO access to external databases, phone records, or personal information "
    "about any individual except what appears in the 'RETRIEVED CONTEXT' section above.\n"
    "- DATABASE RECORDS: If the context includes 'Cơ Sở Dữ Liệu Người Dân', "
    "ONLY report the information that appears EXPLICITLY in those records. "
    "Do NOT infer, guess, or fabricate related phone numbers, names, IDs, "
    "or any other personal information not present in the records. "
    "If a record does not contain a field (e.g., no address, no birthdate), "
    "simply state that the information is not available — do not fill in with assumptions.\n"
    "- PHONE NUMBER SEARCH STRICT RULE:\n"
    "  You have NO knowledge of any specific Vietnamese individual's phone number, "
    "name, CCCD, or BHXH beyond what appears EXPLICITLY in the retrieved database records above.\n"
    "  When a phone search returns NO records:\n"
    "    ✅ CORRECT: 'Không tìm thấy người nào có số điện thoại này trong cơ sở dữ liệu.'\n"
    "    ❌ WRONG: Mentioning ANY other phone number (e.g., 0949755968, 0339755968) "
    "or ANY person's name (e.g., Huỳnh Minh Khải) — even if you think you 'recognize' it. "
    "You do NOT have real-time access to Vietnamese phone records. "
    "Any name or number NOT in the retrieved context is a hallucination.\n"
    "  When a phone search returns records:\n"
    "    ✅ CORRECT: Report ONLY the fields that appear verbatim in the records. "
    "If a phone number is not in the records, do not mention it — even if you believe you know who it belongs to.\n"
    "  FIREWALL RULE: The moment you write a sentence containing a phone number or name "
    "that does NOT appear in the 'Cơ Sở Dữ Liệu Người Dân' section above, "
    "you are hallucinating. Stop immediately and revise.\n"
    "- SPARSE RECORDS (e.g. UID/Facebook records with only phone + ID, no name): "
    "If a record has no person's name attached, do NOT mention it as a result. "
    "Skip it entirely. Only include records where a person's name is present.\n"
    "- PARENT/GUARDIAN PHONE NUMBERS: If the only phone number in a record "
    "belongs to a parent or guardian (e.g., mother's phone in vaccination records), "
    "do NOT report it as the person's own phone number. "
    "You may mention it briefly as 'phone of parent/guardian' only if directly relevant.\n"
)

# =============================================================================
# ABBREVIATION — Minimal instructions for abbreviation lookups
# =============================================================================

_ABBR_INSTRUCTIONS = (
    "- Present the abbreviation meaning clearly and concisely.\n"
    "- If multiple meanings exist, list them all.\n"
    "- If the abbreviation was not found, ask the user to clarify.\n"
)

# =============================================================================
# LIST_DOCS — Simple listing output
# =============================================================================

_LIST_DOCS_INSTRUCTIONS = (
    "- Present the document list clearly and in an organized manner.\n"
    "- Include document names, counts, and any available metadata.\n"
)


# =============================================================================
# THINKING DIRECTIVE — Only injected when extended thinking is enabled
# Guides LLM to perform structured reasoning instead of restating sources
# =============================================================================

_THINKING_DIRECTIVE = (
    "\n## Yêu cầu cho phần suy nghĩ (Thinking)\n"
    "Phần suy nghĩ của bạn phải cực kỳ NGẮN GỌN (tối đa 3-5 câu). "
    "TUYỆT ĐỐI KHÔNG tóm tắt, trích dẫn hay lặp lại nội dung chi tiết. "
    "Hãy NHANH CHÓNG kiểm tra mốc thời gian của các văn bản trong source: "
    "nếu có sự khác biệt (ví dụ: văn bản cũ quy định 3, văn bản mới quy định 2), "
    "hãy note lại ID của văn bản mới nhất để ưu tiên trả lời trước. "
    "Vạch ra dàn ý nhanh (ví dụ: 'Dùng [a1] (mới nhất) để trả lời chính, dùng [b2] (cũ) để đối chiếu thay đổi').\n"
)


# =============================================================================
# Intent → Instructions mapping
# =============================================================================

# RAG intents that need citation rules
_RAG_INTENTS = {
    "search", "summarize", "kg_query", "search_doc_num",
    "resolve_doc", "search_section",
}

# MongoDB people search intents
_MONGO_INTENTS = {
    "mongo_search_cccd", "mongo_search_name",
    "mongo_search_bhxh", "mongo_search_phone",
    "mongo_search_advanced",
}


def get_instructions_for_intent(intent: str, enable_thinking: bool = False) -> str:
    """
    Return the appropriate INSTRUCTIONS string for the given intent.

    Composes BASE + intent-specific modules to minimize token usage
    while preserving all safety rules relevant to the intent.
    When enable_thinking=True, appends _THINKING_DIRECTIVE to guide structured reasoning.

    Args:
        intent: The classified intent string (e.g. "search", "mongo_search_cccd")
        enable_thinking: Whether extended thinking is active (injects reasoning guide)

    Returns:
        Composed INSTRUCTIONS string ready to inject into the prompt.
    """
    parts = ["INSTRUCTIONS:\n", _BASE_INSTRUCTIONS]

    if intent in _RAG_INTENTS:
        parts.append(_RAG_INSTRUCTIONS)
    elif intent in _MONGO_INTENTS:
        parts.append(_MONGO_INSTRUCTIONS)
    elif intent == "search_abbr":
        parts.append(_ABBR_INSTRUCTIONS)
    elif intent == "list_docs":
        parts.append(_LIST_DOCS_INSTRUCTIONS)
    else:
        # Fallback: include RAG instructions for unknown intents
        parts.append(_RAG_INSTRUCTIONS)

    # Phase 3.6: Inject thinking directive only when extended thinking is enabled
    # This guides the LLM to perform structured analysis instead of restating sources
    if enable_thinking:
        parts.append(_THINKING_DIRECTIVE)

    return "".join(parts)

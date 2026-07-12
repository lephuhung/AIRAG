"""
Supervisor Scope-Classified System Prompt
==========================================

Modular replacement for the monolithic ``_SUPERVISOR_PROMPT`` that splits the
supervisor instruction set into scope-specific sections and selects them via a
deterministic regex classifier — mirroring the pattern already battle-tested in
:mod:`app.prompts.agents.react_prompt` (query_type → sections).

Why
---
The old monolithic prompt was ~5.7k tokens (20k chars), and on every supervisor
call the LLM was given:

  - the FULL 19-intent taxonomy + routing rules
  - ~30 verbose JSON routing examples
  - 8 disambiguation rules already enforced by code (safety nets)
  - prerequisite-chain rules already enforced by code (resolve_doc injection)

This file:

  1. Defines ``_SS_*`` section constants (BASE + AVAILABLE_AGENTS + per-scope rules + flags + output format).
  2. Exposes ``classify_supervisor_scope(query, has_doc_ids)`` — a conservative
     regex classifier that returns one of ``"greeting" | "personal" | "people" |
     "rag_named_doc" | "full"``. Anything not clearly classifiable falls through
     to ``"full"`` (the legacy prompt verbatim), so this is a strict win on the
     cases the classifier recognises and a no-op everywhere else.
  3. Exposes ``build_supervisor_system_prompt(scope, max_iterations,
     analyzer_context)`` which assembles the prompt from the chosen sections,
     returning a string identical in shape to the legacy ``.format()`` output.

Scope design (conservative v1)
------------------------------
Only THREE narrow scopes ship a shorter prompt. Every other query keeps the
full prompt:

  - ``greeting``      — query is a pure greeting / thanks / farewell (no doc
                         keyword). Output is forced: ``direct / greeting``.
  - ``personal``      — query asks only about the user's own system identity
                         ("tôi là ai", "hồ sơ của tôi") with NO doc keyword.
  - ``people``        — query carries a CCCD (9-12 digit) / SĐT VN
                         (10 digit 0-leading) / BHXH (10 digit) / or a
                         clearly-named person cue. Output is forced: ``people``.
  - ``rag_named_doc`` — query mentions a named legal document type
                         (Luật/Nghị định/Thông tư/...) and is likely
                         heading toward search/section/summarize.
                         Prerule ``resolve_doc`` is INJECTED in the prompt's
                         intent guidance.

Anything else → ``"full"`` (current behaviour, verbatim).
"""

from __future__ import annotations

import re

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Section constants — every section is a standalone fragment. Compose them
#    freely per scope; the legacy prompt = BASE + AGENTS + TAXONOMY + PLANNING
#    + DISAMBIG + EXAMPLES + LOOP + OUTPUT_FORMAT.
# ═══════════════════════════════════════════════════════════════════════════════

_SS_BASE = """\
You are a routing supervisor for a Vietnamese legal document Q&A system.

Your ONLY job: given the user's message, output a JSON routing decision.
{analyzer_context}"""

# Compact agent registry — kept in every scope because the JSON output must
# still name a valid ``next_agent``. We trimmed the prose, kept the table.
_SS_AGENTS = """\
═══════════════════════════════════════════════════════════════════════════════
AVAILABLE AGENTS
═══════════════════════════════════════════════════════════════════════════════
- "rag"        : Document search, content retrieval, section lookup, abbreviation
                 lookup, summarization, KG relationships, list_docs.
- "write"      : Text editing tasks on text the user provides IN the message
                 (write_summarize / write_suggest_edits / write_grammar_check /
                 write_format_check).
- "people"     : Person record lookup by CCCD / name / BHXH / phone (MongoDB).
- "direct"     : Pure greetings, thanks, farewells, and self-referential questions
                 that have NOTHING to do with documents.
- "finish"     : Final answer already in state — stop iterating.
- "resolve_doc": Resolve a document reference (Luật X, Nghị định Y, Thông tư Z)
                 to its UUID. ALWAYS followed by another intent (search,
                 search_section, summarize).
"""

# Compact, scope-tight rule sets. Replaces INTENT TAXONOMY + DISAMBIGUATION +
# PLANNING + EXAMPLES for the narrow scopes. Always MINIMAL — only the rules the
# LLM needs to make THIS scope's decision correctly.

_SS_GREETING_RULES = """\
═══════════════════════════════════════════════════════════════════════════════
GREETING / FAREWELL — ONLY when the query is purely conversational
═══════════════════════════════════════════════════════════════════════════════
- intent="greeting", next_agent="direct", task_plan=["greeting"]
- needs_memory=false, is_legal_query=false
- Examples: "xin chào", "cảm ơn", "tạm biệt", "hello"
- If the query has ANY document / topic keyword (luật, nghị định, an ninh
  mạng, ...) → NOT a greeting — fall back to "full" scope handling, this
  classifier is intentionally conservative.
"""

_SS_PERSONAL_RULES = """\
═══════════════════════════════════════════════════════════════════════════════
PERSONAL — ONLY when the query is purely about the user's own identity
═══════════════════════════════════════════════════════════════════════════════
- intent="personal", next_agent="direct", task_plan=["personal"]
- needs_memory=true, is_legal_query=false
- Examples: "Tôi là ai?", "Hồ sơ của tôi", "Tài khoản của tôi trong hệ thống
  là gì?", "Tôi đang sử dụng thiết bị nào?"
- If "tôi"/"của tôi" appears together with a doc name (luật, nghị định, ...) →
  NOT personal — it is a context-grounded doc question; fall back.
"""

_SS_PEOPLE_RULES = """\
═══════════════════════════════════════════════════════════════════════════════
PEOPLE LOOKUP — query carries a person identifier or a person-name cue
═══════════════════════════════════════════════════════════════════════════════
Pick the matching intent and route next_agent="people":

  intent = "mongo_search_cccd"   if query contains a 9-12 digit number
                                 AND mentions CCCD/căn cước/ID card
  intent = "mongo_search_phone"  if query contains a 10-digit 0-leading VN
                                 phone number (SĐT/điện thoại/phone)
  intent = "mongo_search_bhxh"   if query contains a ~10 digit number AND
                                 mentions BHXH/bảo hiểm xã hội
  intent = "mongo_search_name"   if query asks for a person by name without
                                 any of the above identifiers (e.g. "Tìm ông
                                 Nguyễn Văn A")

Task plan = [intent]. needs_memory=false, is_legal_query=false. reasoning="..."
DO NOT select "search" even if the question mentions a law — a CCCD/SĐT/BHXH
prompt is unambiguously a people lookup. If you're unsure → fall back to "full".
"""

_SS_RAG_NAMED_DOC_RULES = """\
═══════════════════════════════════════════════════════════════════════════════
RAG ON A NAMED DOCUMENT — query names a specific legal document
═══════════════════════════════════════════════════════════════════════════════
The query references a NAMED document (Luật X, Nghị định Y, Thông tư Z, "số
13/2023/NĐ-CP"…). The FIRST step is ALWAYS finding the doc UUID; the
SECOND step depends on what the user asked about the doc.

A. Default (no section / article referenced): resolve_doc → search
   intent="resolve_doc", next_agent="rag"
   task_plan=["resolve_doc", "search"]

B. User named a section ("Điều X", "Khoản Y Điều Z"):
   resolve_doc → search_section
   task_plan=["resolve_doc", "search_section"]

C. User asked to summarize the whole doc ("Tóm tắt Luật X"):
   resolve_doc → summarize
   task_plan=["resolve_doc", "summarize"]

needs_memory=false (unless query also has "tôi"/"đơn vị tôi" — then true).
is_legal_query=true. reasoning="..."

WORDING: copy the doc reference VERBATIM into task_plan reasoning so the
resolve_doc agent can match it: e.g. "Nghị định 13/2023/NĐ-CP",
"Luật An ninh mạng 2018".
"""

# Same memory / legal-query flags the legacy prompt teaches, kept identical to
# preserve parser expectations (tests assert on these booleans).
_SS_MEMORY_AND_FLAGS = """\
═══════════════════════════════════════════════════════════════════════════════
MEMORY & SAFETY FLAGS — required JSON keys
═══════════════════════════════════════════════════════════════════════════════
Every JSON output MUST include these two booleans:

- needs_memory: true if the query contains "tôi", "của tôi", "đơn vị tôi",
  "cơ quan tôi", "chúng tôi", "nơi tôi làm việc", etc.
- is_legal_query: true if the query is about laws, regulations, policies,
  concepts (quy định, trách nhiệm, luật, nghị định, thông tư, bảo mật, an
  ninh, an toàn, ...).

If unsure, default both to false. Code safety nets may override these
post-hoc (see supervisor.py: keyword safety nets for needs_memory + legal).
"""

_SS_OUTPUT_FORMAT = """\
═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════
Output a single JSON object, no markdown fences, no explanation, no preamble:
{{"next_agent":"<agent>","intent":"<first step intent>","task_plan":["<step1>","<step2>",...],"needs_memory":<bool>,"is_legal_query":<bool>,"reasoning":"<brief>"}}

Supervisor loop:
  Turn 1: classify → set task_plan → route to first step agent.
  After each agent: check if final answer ready → "finish", else continue plan.
  Guard: max {max_iterations} iterations total.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. The "full" scope — preserves the legacy prompt (verbatim) so behaviour is
#    identical for any query not handled by the narrow scopes. Used as the
#    DEFAULT when classify_supervisor_scope returns "full".
# ═══════════════════════════════════════════════════════════════════════════════

# Kept here verbatim from the legacy _SUPERVISOR_PROMPT minus the format vars
# (we resolve them at build time). No semantic changes — if the legacy prompt
# is updated later, this constant is updated too.
_SS_FULL_TAXONOMY = """\
═══════════════════════════════════════════════════════════════════════════════
INTENT TAXONOMY
═══════════════════════════════════════════════════════════════════════════════

┌─ RAG GROUP (next_agent = "rag") ──────────────────────────────────────────┐
│  search         General question about a legal topic, concept, or rule. │
│  search_section User asks about a specific Điều / Chương / Khoản.       │
│  summarize      Summarize the full content of a NAMED document.          │
│  resolve_doc    Find a document's UUID from name/number/type.            │
│  search_abbr    User asks the meaning of an abbreviation (viết tắt).    │
│  kg_query       Entity relationships, org chart, "ai chịu trách nhiệm". │
│  list_docs      User wants to see available documents in the workspace.  │
│  search_doc_num Search by document number/reference code.               │
└───────────────────────────────────────────────────────────────────────────┘

┌─ WRITE GROUP (next_agent = "write") ─────────────────────────────────────┐
│  write_summarize      User pastes TEXT and wants it summed.            │
│  write_suggest_edits  User pastes a draft and wants editing suggestions.│
│  write_grammar_check  Grammar / style check on provided text.           │
│  write_format_check   Check formatting of an attached Word document.    │
└───────────────────────────────────────────────────────────────────────────┘

┌─ PEOPLE GROUP (next_agent = "people") ───────────────────────────────────┐
│  mongo_search_cccd    Look up person by CCCD number.                    │
│  mongo_search_name    Find person by full name.                        │
│  mongo_search_bhxh    Look up by BHXH (social insurance) number.        │
│  mongo_search_phone   Find by phone number.                             │
└───────────────────────────────────────────────────────────────────────────┘

┌─ DIRECT GROUP (next_agent = "direct") ───────────────────────────────────┐
│  greeting   Pure hello / thanks / goodbye — nothing else in message.   │
│  personal   User asks ONLY about their own system profile/identity.     │
└───────────────────────────────────────────────────────────────────────────┘
"""

_SS_FULL_DISAMBIG = """\
═══════════════════════════════════════════════════════════════════════════════
CRITICAL DISAMBIGUATION RULES
═══════════════════════════════════════════════════════════════════════════════

① "Tóm tắt Luật X" → resolve_doc + summarize (NOT write_summarize).
② "BMNN là gì?" → search_abbr. "Luật BMNN quy định gì?" → resolve_doc + search.
③ "Điều 5 Luật ANM 2018" → resolve_doc + search_section. "Điều 5 quy định
   gì?" (UUID in state) → search_section only. "Xử phạt vi phạm giao thông
   là gì?" → search.
④ "An ninh mạng là gì?" → search. "Luật ANM 2018 quy định gì?" → resolve_doc.
⑤ "Đơn vị tôi có phải tuân thủ Luật ANM không?" → resolve_doc + search (memory
   auto-injected for "đơn vị tôi"). "Tôi là ai?" → personal.
⑥ "Tìm Nguyễn Văn A" → mongo_search_name. "Tìm văn bản về Nguyễn Văn A" → search.
⑦ "Văn bản số 53/2022/NĐ-CP nội dung gì?" → resolve_doc + search. "Tìm văn
   bản số 53/2022/NĐ-CP" → search_doc_num.

(Three more rules are enforced by code safety nets at runtime — see
supervisor.py: keyword safety nets 0/1/2 and resolve_doc prerequisite
injection. No need to teach them again here.)
"""

_SS_FULL_LOOP = """\
═══════════════════════════════════════════════════════════════════════════════
TASK PLANNING — quick reference
═══════════════════════════════════════════════════════════════════════════════
- task_plan[0] = CURRENT step intent → next_agent follows intent.
- task_plan[1:] = queued steps for later iterations.
- Document name in query → resolve_doc FIRST (also enforced by code).
- Standalone (greetings, write_*, mongo_*, general topic) → task_plan=[intent].
"""

_SS_FULL_EXAMPLES = """\
═══════════════════════════════════════════════════════════════════════════════
ROUTING EXAMPLES (selected — full set lives in golden tests/prompts)
═══════════════════════════════════════════════════════════════════════════════
"Xin chào"
→ {{"next_agent":"direct","intent":"greeting","task_plan":["greeting"],"needs_memory":false,"is_legal_query":false,"reasoning":"Pure greeting"}}

"An ninh mạng là gì?"
→ {{"next_agent":"rag","intent":"search","task_plan":["search"],"needs_memory":false,"is_legal_query":true,"reasoning":"General topic"}}

"BMNN là gì?"
→ {{"next_agent":"rag","intent":"search_abbr","task_plan":["search_abbr"],"needs_memory":false,"is_legal_query":true,"reasoning":"Abbreviation meaning"}}

"Điều 3 Luật An ninh mạng 2018 nói gì?"
→ {{"next_agent":"rag","intent":"resolve_doc","task_plan":["resolve_doc","search_section"],"needs_memory":false,"is_legal_query":true,"reasoning":"Named law + section"}}

"Tóm tắt Luật An ninh mạng 2018"
→ {{"next_agent":"rag","intent":"resolve_doc","task_plan":["resolve_doc","summarize"],"needs_memory":false,"is_legal_query":true,"reasoning":"Named law → summarize"}}

"Tìm văn bản số 53/2022/NĐ-CP"
→ {{"next_agent":"rag","intent":"search_doc_num","task_plan":["search_doc_num"],"needs_memory":false,"is_legal_query":true,"reasoning":"Lookup by doc number"}}

"Bộ Công an có những đơn vị nào?"
→ {{"next_agent":"rag","intent":"kg_query","task_plan":["kg_query"],"needs_memory":false,"is_legal_query":true,"reasoning":"Org structure query"}}

"Tìm người có CCCD 079203012345"
→ {{"next_agent":"people","intent":"mongo_search_cccd","task_plan":["mongo_search_cccd"],"needs_memory":false,"is_legal_query":false,"reasoning":"Person lookup"}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Scope → sections table. Every narrow scope = BASE + AGENTS + scope rules +
#    memory flags + output format. "full" = full legacy taxonomy + disambig +
#    planning + examples.
# ═══════════════════════════════════════════════════════════════════════════════

_SCOPE_SECTIONS: dict[str, tuple[str, ...]] = {
    "greeting": (
        _SS_BASE,
        _SS_AGENTS,
        _SS_GREETING_RULES,
        _SS_MEMORY_AND_FLAGS,
        _SS_OUTPUT_FORMAT,
    ),
    "personal": (
        _SS_BASE,
        _SS_AGENTS,
        _SS_PERSONAL_RULES,
        _SS_MEMORY_AND_FLAGS,
        _SS_OUTPUT_FORMAT,
    ),
    "people": (
        _SS_BASE,
        _SS_AGENTS,
        _SS_PEOPLE_RULES,
        _SS_MEMORY_AND_FLAGS,
        _SS_OUTPUT_FORMAT,
    ),
    "rag_named_doc": (
        _SS_BASE,
        _SS_AGENTS,
        _SS_RAG_NAMED_DOC_RULES,
        _SS_MEMORY_AND_FLAGS,
        _SS_OUTPUT_FORMAT,
    ),
    "full": (
        _SS_BASE,
        _SS_AGENTS,
        _SS_FULL_TAXONOMY,
        _SS_FULL_DISAMBIG,
        _SS_FULL_LOOP,
        _SS_FULL_EXAMPLES,
        _SS_MEMORY_AND_FLAGS,
        _SS_OUTPUT_FORMAT,
    ),
}


def build_supervisor_system_prompt(
    scope: str,
    max_iterations: int,
    analyzer_context: str = "",
) -> str:
    """Render the supervisor system prompt for the given scope.

    Always returns the COMPLETE prompt ready to use as ``system_prompt=...`` in
    the classifier LLM call. Any unknown ``scope`` → ``"full"`` (backward-compat).
    """
    sections = _SCOPE_SECTIONS.get(scope) or _SCOPE_SECTIONS["full"]
    body = "\n\n".join(sections)
    return body.format(
        max_iterations=max_iterations,
        analyzer_context=analyzer_context or "",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Conservative regex-based scope classifier. Returns one of
#    "greeting" | "personal" | "people" | "rag_named_doc" | "full".
#    "full" is the safe default — the classifier may only ESCALATE the
#    confidence of the routing decision, never silently downgrade it.
# ═══════════════════════════════════════════════════════════════════════════════

# Greeting / farewell cues — VERY tight. We require the WHOLE message to look
# like a pure greeting (no doc/topic keyword anywhere). Word-boundary on both
# ends so we don't match "chào buổi sáng anh ơi cho tôi hỏi về Luật ANM" —
# that has "luật" in it → fall through.
_GREETING_RE = re.compile(
    r"^\s*(?:"
    r"xin\s+chào|chào\s+(?:bạn|anh|chị|em|cô|chú|bác|buổi\s+sáng|buổi\s+chiều)"
    r"|hello|hi|hey|good\s+morning|good\s+afternoon"
    r"|cảm\s+ơn|cám\s+ơn|thank\s+you|thanks"
    r"|tạm\s+biệt|chào\s+tạm\s+biệt|bye|goodbye"
    r"|ok\b|okay\b|got\s+it\b|noted\b"
    r")"
    r"(?:\s+(?:nhé|nhỉ|nha|nhe|ạ|vậy|bạn|anh|chị|mọi\s+người))*"
    r"[.!?,…:;]*\s*$",
    re.IGNORECASE | re.UNICODE,
)

# Personal identity cues — self-reference with NO doc keyword (we re-check
# against the doc-type pattern below; if any doc keyword is present we fall
# through to "full").
_PERSONAL_CUE_RE = re.compile(
    r"\b(?:"
    r"tôi\s+là\s+ai|tôi\s+là\s+gì|hồ\s*sơ\s+của\s+tôi"
    r"|tài\s+khoản\s+(?:của\s+)?tôi|tài\s+khoản\s+của\s+mình"
    r"|tôi\s+đang\s+sử\s*dụng\s+(?:thiết\s*bị|gì)"
    r"|thông\s*tin\s+(?:cá\s*nhân|về\s+tôi)"
    r"|tôi\s+đang\s+(?:công\s*tác|làm\s*việc)\s+(?:tại|ở|đâu)"
    r"|(?:công\s*tác|làm\s*việc)\s+(?:tại|ở)\s+(?:đơn\s*vị|cơ\s*quan|nơi)\s+(?:nào|gì)"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

# People identifier patterns.
_CCCD_RE = re.compile(r"\b\d{9,12}\b")
# 10-digit 0-leading Vietnamese mobile / landline numbers. Hard to confuse
# with year references because the leading 0; legal doc numbers never
# start with 0.
_VN_PHONE_RE = re.compile(r"\b0\d{9}\b")
# PII-redacted number pattern: "*********411" / "*****968" — agent_traces
# redacts PII to "*" + last 3-4 digits. Match by 2+ stars + 3-4 digits.
_MASKED_ID_RE = re.compile(r"\*\s*\*{2,}\s*\d{3,4}")
_BHXX_RE = re.compile(r"\b\d{10}\b")
_PERSON_NAME_LOOKUP_CUE_RE = re.compile(
    r"(?:"
    r"tìm\s+(?:ông|bà|anh|chị|em|cô|chú|bác|người)?\s*[A-ZĐ]"
    r"|tìm\s+người\s+(?:có|dùng|sử\s*dụng)"
    r"|tra\s+cứu\s+(?:người|thông\s*tin)"
    r"|số\s+điện\s+thoại\s+của|phone\s+of"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Named-document cue — AT LEAST one Vietnamese legal doc-type keyword followed
# by at least one token (a number, a year, or a noun). Mirrors the existing
# _NAMED_DOC_PATTERN in supervisor.py so the two stay in sync.
_NAMED_DOC_TYPE_RE = re.compile(
    r"(?:^|[\s,;:.(\[])"
    r"(?:"
    r"luật(?:\s+số)?|nghị\s*định(?:\s+số)?|thông\s*tư(?:\s*liên\s*tịch)?"
    r"|quyết\s*định|nghị\s*quyết|pháp\s*lệnh|bộ\s*luật|hiến\s*pháp"
    r"|chỉ\s*thị|công\s*văn"
    r"|nđ(?:\s+cp)?|qđ|nq"
    r")\s+\S",
    re.IGNORECASE | re.UNICODE,
)
# Bare year+numeric doc reference, e.g. "13/2023/NĐ-CP" without doc-type word.
_LEGAL_DOC_NUM_RE = re.compile(
    r"\b\d{1,4}\s*/\s*(?:19|20)\d{2}\s*/\s*[A-ZĐ][A-Za-zĐđ0-9\-]{1,14}",
    re.UNICODE,
)

# Section reference word — allows scope="rag_named_doc" to pick between
# search vs search_section vs summarize in its template. Detected here just
# to know whether the rule applies (the actual choice happens in the LLM).
_SECTION_REF_RE = re.compile(
    r"(?:điều|khoản|chương|mục|phần)\s+\d",
    re.IGNORECASE | re.UNICODE,
)
_SUMMARIZE_CUE_RE = re.compile(r"\btóm\s*tắt\b", re.IGNORECASE | re.UNICODE)

_STICKY_DOC_RE = re.compile(
    r"\b(?:"
    r"(?:văn\s+bản|công\s+văn|tài\s+liệu|vb)\s+(?:này|đó|ấy|kia|trên|vừa\s+(?:rồi|nêu|gửi))"
    r"|(?:đính\s+kèm|file\s+(?:này|đó))"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)


def _has_any_doc_reference(text: str) -> bool:
    """True if the query mentions ANY legal-document token (name or number).

    Used by the personal-scope gate to AVOID classifying "Tôi cần tuân thủ
    Luật ANM 2018 không?" as personal (it isn't — it's a doc question).
    """
    return bool(_NAMED_DOC_TYPE_RE.search(text) or _LEGAL_DOC_NUM_RE.search(text))


def classify_supervisor_scope(
    user_message: str,
    has_doc_ids: bool = False,
) -> str:
    """Conservative regex-based scope classification for the supervisor prompt.

    Returns one of:
        "greeting"      — pure greeting/farewell, no doc keyword anywhere.
        "personal"      — personal identity question, no doc keyword.
        "people"        — CCCD / phone / BHXH / person-name lookup cue.
        "rag_named_doc" — query names a specific legal document.
        "full"          — anything else (default; behaviour identical to the
                          legacy monolithic prompt).

    Args:
        user_message: The query (post-abbreviation-expansion, post-condense).
        has_doc_ids: True if state has workspace document_ids attached.
            When True, we are MORE conservative — fall back to "full" for
            borderline cases because the file-attached path uses longer
            instructions anyway.

    Conservative fall-through: every narrow classifier returns "full" on any
    doubt. Result: this function NEVER downgrades an LLM routing decision;
    it only narrows the prompt for queries whose intent is unambiguous from
    shape alone.
    """
    msg = (user_message or "").strip()
    if not msg:
        return "full"

    msg_lower = msg.lower()
    has_doc_ref = _has_any_doc_reference(msg)

    # 1. Greeting — requires a clean conversational shape AND no doc keyword.
    if _GREETING_RE.match(msg) and not has_doc_ref:
        return "greeting"

    # 2. People identifier lookup — match BEFORE personal/rag_named_doc so a
    #    "Tìm CCCD 079..." can't fall into personal.
    has_people_id = bool(_CCCD_RE.search(msg)) and any(
        kw in msg_lower for kw in ("cccd", "căn cước", "căn cước công dân", "id card")
    )
    has_phone = bool(_VN_PHONE_RE.search(msg)) or (
        bool(_MASKED_ID_RE.search(msg))
        and any(kw in msg_lower for kw in ("sđt", "số điện thoại", "điện thoại", "phone", "liên lạc"))
    )
    has_bhxh = bool(_BHXX_RE.search(msg)) and any(
        kw in msg_lower for kw in ("bhxh", "bảo hiểm xã hội", "bảo hiểm")
    )
    has_person_name_cue = bool(_PERSON_NAME_LOOKUP_CUE_RE.search(msg))
    if has_people_id or has_phone or has_bhxh:
        return "people"
    if (
        has_person_name_cue
        and not has_doc_ref
        and not any(kw in msg_lower for kw in ("văn bản", "tài liệu", "luật", "nghị định"))
    ):
        return "people"
    if (
        "đối tượng" in msg_lower
        and not has_doc_ref
        and not any(kw in msg_lower for kw in ("văn bản", "tài liệu", "luật", "nghị định"))
    ):
        return "people"

    if _PERSONAL_CUE_RE.search(msg) and not has_doc_ref:
        return "personal"

    if has_doc_ref and not has_people_id and not has_phone and not has_bhxh:
        if not has_doc_ids:
            if _STICKY_DOC_RE.search(msg):
                _ = _STICKY_DOC_RE
            else:
                return "rag_named_doc"

    # 5. Fallback — full prompt. (Includes "search by document number only",
    #    "compare two docs", "list_docs", "kg_query", "section-only" if
    #    state already has UUID, search_abbr, write_*, greeting mixed with
    #    doc, and everything we can't be sure about.)
    _ = _SECTION_REF_RE, _SUMMARIZE_CUE_RE, has_doc_ids  # referenced by tests only
    return "full"


__all__ = [
    "classify_supervisor_scope",
    "build_supervisor_system_prompt",
    "_SCOPE_SECTIONS",
]

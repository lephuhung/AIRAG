"""Task 1 — frozen v1 route-intent parity corpus (Phase 4A gate).

Characterization snapshot of CURRENT v1 semantic behavior, captured BEFORE
any v2 routing change. Each case records:

- ``v1_intent`` — the expected v1 **semantic intent** (taxonomy intent such
  as ``greeting`` / ``search`` / ``mongo_search_phone`` / ``kg_query`` /
  ``summarize``). This is deliberately NOT ``next_agent``: legacy routing
  fields (``next_agent`` / ``pending_intent`` / task plans) are never
  migrated into v2 contracts and therefore never appear in this corpus.
- ``v1_scope`` / ``v1_intent_source`` — provenance: ``deterministic`` cases
  are decided by ``classify_supervisor_scope`` +
  ``deterministic_decision_for_scope`` without any model call (asserted
  exactly); ``model-full-taxonomy`` cases fall through to the full v1
  prompt (``scope == "full"``, asserted) and the recorded intent documents
  the full-taxonomy model expectation (NOT asserted against a live model —
  characterization only, no network in tests).
- ``v2_*`` — the SEPARATELY recorded Phase 4A target: expected v2
  ``QueryAnalysis`` (``work_type`` + ``domains``) and execution topology
  (``route`` + frozen ``reason_code``). ``v2_reason_code = None`` means the
  fast route is expected but the exact frozen reason code is assigned by
  Task 4 (targetless ``document.retrieve`` fast path does not exist yet).
  Shape-validity of these targets is asserted; equality with CURRENT v2
  output is NOT (Tasks 3/4 close the gap and must pass every case here).

Verified against current code on 2026-09-14 (see task-1 report for the
probe outputs):
``classify_supervisor_scope`` -> greeting / people(phone, cccd, name) /
rag_named_doc / full exactly as recorded below; current v2
``analyze_query`` still misroutes the greeting-prefixed factual query to
``explain``/``memory``, phone/``là ai`` people queries to
``lookup``/``knowledge_graph``, and the bare general comparison to
``compare`` — the divergences Tasks 3/4 must fix.
"""
from __future__ import annotations

REQUIRED_IDS = frozenset(
    {
        "greeting-pure",
        "greeting-prefix-factual",
        "people-phone",
        "people-cccd",
        "people-name-is-ai",
        "people-name-tim-ong",
        "rag-general-thai-san",
        "rag-general-compare",
        "summarize-named-doc",
        "kg-org-structure",
    }
)

#: Frozen corpus. Key contract: no ``next_agent`` / ``pending_intent`` /
#: ``task_plan`` keys (enforced by test_fixtures_record_semantic_intent_only).
INTENT_CASES: tuple[dict, ...] = (
    {
        "id": "greeting-pure",
        "query": "xin chào",
        "v1_intent": "greeting",
        "v1_scope": "greeting",
        "v1_intent_source": "deterministic",
        "v1_needs_memory": False,
        "v1_is_legal_query": False,
        "v2_work_type": "direct",
        "v2_domains": ("memory",),
        "v2_route": "direct",
        "v2_reason_code": "direct_greeting",
        "note": "Pure greeting; v1 deterministic short-circuit, no model call.",
    },
    {
        "id": "greeting-prefix-factual",
        "query": "chào anh, hỏi về chế độ thai sản?",
        "v1_intent": "search",
        "v1_scope": "full",
        "v1_intent_source": "model-full-taxonomy",
        "v1_needs_memory": False,
        "v1_is_legal_query": True,
        "v2_work_type": "retrieve",
        "v2_domains": ("document",),
        "v2_route": "fast_domain",
        "v2_reason_code": None,
        "note": "Greeting prefix + factual remainder is NOT a greeting; "
        "factual retrieve fast path (targetless).",
    },
    {
        "id": "people-phone",
        "query": "0901234567 là ai?",
        "v1_intent": "mongo_search_phone",
        "v1_scope": "people",
        "v1_intent_source": "deterministic",
        "v1_needs_memory": False,
        "v1_is_legal_query": False,
        "v2_work_type": "lookup",
        "v2_domains": ("people",),
        "v2_route": "fast_domain",
        "v2_reason_code": "simple_people_lookup",
        "note": "10-digit 0-leading VN phone; deterministic people lookup.",
    },
    {
        "id": "people-cccd",
        "query": "079012345678 CCCD này của ai?",
        "v1_intent": "mongo_search_cccd",
        "v1_scope": "people",
        "v1_intent_source": "deterministic",
        "v1_needs_memory": False,
        "v1_is_legal_query": False,
        "v2_work_type": "lookup",
        "v2_domains": ("people",),
        "v2_route": "fast_domain",
        "v2_reason_code": "simple_people_lookup",
        "note": "12-digit number + CCCD keyword; deterministic people lookup.",
    },
    {
        "id": "people-name-is-ai",
        "query": "Nguyễn Văn A là ai?",
        "v1_intent": "mongo_search_name",
        "v1_scope": "full",
        "v1_intent_source": "model-full-taxonomy",
        "v1_needs_memory": False,
        "v1_is_legal_query": False,
        "v2_work_type": "lookup",
        "v2_domains": ("people",),
        "v2_route": "fast_domain",
        "v2_reason_code": "simple_people_lookup",
        "note": "Named-person 'là ai?' is a people lookup (full-taxonomy "
        "rule: person name without identifiers -> mongo_search_name), not KG.",
    },
    {
        "id": "people-name-tim-ong",
        "query": "Tìm ông Nguyễn Văn A",
        "v1_intent": "mongo_search_name",
        "v1_scope": "people",
        "v1_intent_source": "deterministic",
        "v1_needs_memory": False,
        "v1_is_legal_query": False,
        "v2_work_type": "lookup",
        "v2_domains": ("people",),
        "v2_route": "fast_domain",
        "v2_reason_code": "simple_people_lookup",
        "note": "Explicit person-name cue; deterministic people lookup.",
    },
    {
        "id": "rag-general-thai-san",
        "query": "chế độ thai sản được quy định thế nào?",
        "v1_intent": "search",
        "v1_scope": "full",
        "v1_intent_source": "model-full-taxonomy",
        "v1_needs_memory": False,
        "v1_is_legal_query": True,
        "v2_work_type": "retrieve",
        "v2_domains": ("document",),
        "v2_route": "fast_domain",
        "v2_reason_code": None,
        "note": "General document retrieve FAST: bounded targetless "
        "document retrieval, no explicit binding required.",
    },
    {
        "id": "rag-general-compare",
        "query": "sự khác biệt giữa nghỉ phép và nghỉ ốm?",
        "v1_intent": "search",
        "v1_scope": "full",
        "v1_intent_source": "model-full-taxonomy",
        "v1_needs_memory": False,
        "v1_is_legal_query": True,
        "v2_work_type": "retrieve",
        "v2_domains": ("document",),
        "v2_route": "fast_domain",
        "v2_reason_code": None,
        "note": "General factual retrieve; comparison topology ONLY with "
        "explicit multi-target research (two resolved docs).",
    },
    {
        "id": "summarize-named-doc",
        "query": "Tóm tắt Nghị định A",
        "v1_intent": "summarize",
        "v1_prerequisite": "resolve_doc",
        "v1_scope": "rag_named_doc",
        "v1_intent_source": "model-full-taxonomy",
        "v1_needs_memory": False,
        "v1_is_legal_query": True,
        "v2_work_type": "summarize",
        "v2_domains": ("document",),
        "v2_route": "fast_domain",
        "v2_reason_code": "exact_document_metadata",
        "note": "One-doc bounded summarize once one target is resolved; "
        "resolve_doc is a semantic prerequisite, not complexity.",
    },
    {
        "id": "kg-org-structure",
        "query": "Bộ Công an có những đơn vị nào?",
        "v1_intent": "kg_query",
        "v1_scope": "full",
        "v1_intent_source": "model-full-taxonomy",
        "v1_needs_memory": False,
        "v1_is_legal_query": True,
        "v2_work_type": "lookup",
        "v2_domains": ("knowledge_graph",),
        "v2_route": "fast_domain",
        "v2_reason_code": "simple_kg_lookup",
        "note": "Simple KG org-structure query; knowledge_graph lookup.",
    },
)

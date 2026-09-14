"""Task 1 — v1 route-intent parity corpus test (Phase 4A characterization gate).

Proves the fixtures in ``intent_cases.py`` represent CURRENT v1 behavior
before v2 routing changes (Tasks 3/4):

- deterministic v1 cases: ``classify_supervisor_scope`` plus
  ``deterministic_decision_for_scope`` must reproduce the recorded v1
  semantic intent exactly;
- non-deterministic cases: ``classify_supervisor_scope`` must return the
  recorded scope (``full`` or ``rag_named_doc``) with no deterministic
  decision, i.e. current deterministic code makes NO claim and the recorded
  intent documents the model/derived expectation for that scope;
- every recorded v2 target must construct valid frozen ``QueryAnalysis``
  (via the canonical ``validate_query_analysis``) and every recorded
  ``v2_route`` must be a frozen ``Route`` member; cases whose frozen
  ``reason_code`` does not exist yet carry an explicit
  ``v2_reason_pending_contract_extension`` marker plus a named
  ``v2_reason_proposal`` for the Task 4 ruling (shape check only — current
  v2 output is NOT asserted here; Tasks 3/4 close the gap);
- fixtures record semantic intent only: no ``next_agent`` /
  ``pending_intent`` / task-plan leakage into the corpus.
"""
from __future__ import annotations

from typing import get_args

from app.prompts.agents.supervisor_scope import (
    classify_supervisor_scope,
    deterministic_decision_for_scope,
)
from app.services.agents.v2.contracts.routing import (
    QueryAnalysis,
    Route,
    RouteDecision,
)
from app.services.agents.v2.contracts.validation import validate_query_analysis

from tests.agents.v2.golden.intent_cases import INTENT_CASES, REQUIRED_IDS

#: Allowed ``v1_intent_source`` vocabulary. ``derived-from-v1-prerequisite-rewrite``
#: = terminal semantic intent derived from a v1 resolve_doc prerequisite chain
#: (scoped prompt + ``supervisor.py`` prerequisite injection), not model-emitted.
INTENT_SOURCES = frozenset(
    {"deterministic", "model-full-taxonomy", "derived-from-v1-prerequisite-rewrite"}
)

#: Brief-pinned query strings (M1): catches query drift that an id-only gate
#: would miss. Byte-verified against task-1-brief.md:16-27 on 2026-09-14.
EXPECTED_QUERIES = {
    "greeting-pure": "xin chào",
    "greeting-prefix-factual": "chào anh, hỏi về chế độ thai sản?",
    "people-phone": "0901234567 là ai?",
    "people-cccd": "079012345678 CCCD này của ai?",
    "people-name-is-ai": "Nguyễn Văn A là ai?",
    "people-name-tim-ong": "Tìm ông Nguyễn Văn A",
    "rag-general-thai-san": "chế độ thai sản được quy định thế nào?",
    "rag-general-compare": "sự khác biệt giữa nghỉ phép và nghỉ ốm?",
    "summarize-named-doc": "Tóm tắt Nghị định A",
    "kg-org-structure": "Bộ Công an có những đơn vị nào?",
}

_DETERMINISTIC = [c for c in INTENT_CASES if c["v1_intent_source"] == "deterministic"]
_NON_DETERMINISTIC = [c for c in INTENT_CASES if c["v1_intent_source"] != "deterministic"]


def test_corpus_covers_required_cases() -> None:
    ids = {case["id"] for case in INTENT_CASES}
    assert REQUIRED_IDS <= ids, f"missing required cases: {REQUIRED_IDS - ids}"
    by_id = {case["id"]: case["query"] for case in INTENT_CASES}
    for case_id, query in EXPECTED_QUERIES.items():
        assert by_id.get(case_id) == query, f"{case_id}: query drifted"


def test_provenance_vocabulary_and_bucket_membership() -> None:
    for case in INTENT_CASES:
        assert case["v1_intent_source"] in INTENT_SOURCES, (
            f"{case['id']}: unknown provenance {case['v1_intent_source']!r}"
        )
    assert len(_DETERMINISTIC) + len(_NON_DETERMINISTIC) == len(INTENT_CASES), (
        "every case must sit in exactly one behavior bucket"
    )


def test_deterministic_v1_intents_reproduce_current_behavior() -> None:
    assert _DETERMINISTIC, "corpus must contain deterministic v1 cases"
    for case in _DETERMINISTIC:
        scope = classify_supervisor_scope(case["query"])
        assert scope == case["v1_scope"], f"{case['id']}: scope {scope!r}"
        decision = deterministic_decision_for_scope(scope, case["query"])
        assert decision is not None, f"{case['id']}: expected deterministic decision"
        assert decision["intent"] == case["v1_intent"], f"{case['id']}: intent"
        assert decision["needs_memory"] is case["v1_needs_memory"], f"{case['id']}"
        assert decision["is_legal_query"] is case["v1_is_legal_query"], f"{case['id']}"


def test_non_deterministic_cases_match_recorded_scope() -> None:
    assert _NON_DETERMINISTIC, "corpus must contain non-deterministic v1 cases"
    for case in _NON_DETERMINISTIC:
        scope = classify_supervisor_scope(case["query"])
        assert scope == case["v1_scope"], (
            f"{case['id']}: expected scope {case['v1_scope']!r}, got {scope!r}"
        )
        assert deterministic_decision_for_scope(scope, case["query"]) is None, (
            f"{case['id']}: deterministic code must make no claim"
        )


def test_v2_targets_are_valid_frozen_contracts() -> None:
    for case in INTENT_CASES:
        assert case["v2_route"] in get_args(Route), (
            f"{case['id']}: v2_route {case['v2_route']!r} not a frozen Route"
        )
        analysis = QueryAnalysis(
            work_type=case["v2_work_type"],
            domains=tuple(case["v2_domains"]),
        )
        validate_query_analysis(analysis)
        has_code = case.get("v2_reason_code") is not None
        has_marker = bool(case.get("v2_reason_pending_contract_extension"))
        assert has_code != has_marker, (
            f"{case['id']}: need exactly one of v2_reason_code / "
            "v2_reason_pending_contract_extension"
        )
        if has_marker:
            assert case.get("v2_reason_proposal"), (
                f"{case['id']}: pending extension needs a named proposal"
            )
            assert case["v2_route"] == "fast_domain", (
                f"{case['id']}: pending extension only valid for fast_domain targets"
            )
        else:
            RouteDecision(route=case["v2_route"], reason_code=case["v2_reason_code"])


def test_fixtures_record_semantic_intent_only() -> None:
    banned = {"next_agent", "pending_intent", "task_plan"}
    for case in INTENT_CASES:
        assert not (banned & set(case)), f"{case['id']}: legacy routing key leaked"

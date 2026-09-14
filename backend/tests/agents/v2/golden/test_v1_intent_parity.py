"""Task 1 — v1 route-intent parity corpus test (Phase 4A characterization gate).

Proves the fixtures in ``intent_cases.py`` represent CURRENT v1 behavior
before v2 routing changes (Tasks 3/4):

- deterministic v1 cases: ``classify_supervisor_scope`` plus
  ``deterministic_decision_for_scope`` must reproduce the recorded v1
  semantic intent exactly;
- model-dependent cases: ``classify_supervisor_scope`` must return the recorded
  scope (``full`` or ``rag_named_doc``) with no deterministic decision, i.e.
  current deterministic code makes NO claim and the recorded intent
  documents the full-taxonomy model expectation;
- every recorded v2 target must construct valid frozen ``QueryAnalysis`` /
  ``RouteDecision`` contracts (shape check only — current v2 output is NOT
  asserted here; Tasks 3/4 close the gap);
- fixtures record semantic intent only: no ``next_agent`` /
  ``pending_intent`` / task-plan leakage into the corpus.
"""
from __future__ import annotations

from app.prompts.agents.supervisor_scope import (
    classify_supervisor_scope,
    deterministic_decision_for_scope,
)
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision

from tests.agents.v2.golden.intent_cases import INTENT_CASES, REQUIRED_IDS

_DETERMINISTIC = [c for c in INTENT_CASES if c["v1_intent_source"] == "deterministic"]
_MODEL = [c for c in INTENT_CASES if c["v1_intent_source"] == "model-full-taxonomy"]


def test_corpus_covers_required_cases() -> None:
    ids = {case["id"] for case in INTENT_CASES}
    assert REQUIRED_IDS <= ids, f"missing required cases: {REQUIRED_IDS - ids}"


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


def test_model_cases_fall_through_to_full_scope() -> None:
    assert _MODEL, "corpus must contain model-dependent v1 cases"
    for case in _MODEL:
        scope = classify_supervisor_scope(case["query"])
        assert scope == case["v1_scope"], (
            f"{case['id']}: expected scope {case['v1_scope']!r}, got {scope!r}"
        )
        assert deterministic_decision_for_scope(scope, case["query"]) is None, (
            f"{case['id']}: deterministic code must make no claim"
        )


def test_greeting_prefix_with_factual_remainder_is_not_greeting() -> None:
    query = "chào anh, hỏi về chế độ thai sản?"
    assert classify_supervisor_scope(query) != "greeting"


def test_v2_targets_are_valid_frozen_contracts() -> None:
    for case in INTENT_CASES:
        analysis = QueryAnalysis(
            work_type=case["v2_work_type"],
            domains=tuple(case["v2_domains"]),
        )
        assert analysis.work_type == case["v2_work_type"]
        if case["v2_reason_code"] is not None:
            decision = RouteDecision(
                route=case["v2_route"], reason_code=case["v2_reason_code"]
            )
            assert decision.route == case["v2_route"]


def test_fixtures_record_semantic_intent_only() -> None:
    banned = {"next_agent", "pending_intent", "task_plan"}
    for case in INTENT_CASES:
        assert not (banned & set(case)), f"{case['id']}: legacy routing key leaked"

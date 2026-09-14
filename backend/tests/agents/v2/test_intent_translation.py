"""Task 3 — translate v1 intent taxonomy into v2 QueryAnalysis (RED first).

Typed v1 intent is advisory semantic input only: ``analyze_query`` accepts an
optional ``IntentDecision`` and maps it onto frozen ``QueryAnalysis``; the
deterministic ``decide_route()`` policy keeps route authority. Generic
regex/keyword classifiers (``là ai``, ``khác biệt``, ``tổng hợp``,
``đánh giá``) must not govern once typed intent is available, and no
confidence field may be added to frozen ``QueryAnalysis``.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from langgraph.runtime import Runtime

from app.services.agents.v2.contracts.binding import DocumentBindingSet
from app.services.agents.v2.contracts.capability import CapabilityRuntimeContext
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.routing import QueryAnalysis
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.nodes.routing import analyze_query, decide_route, route_node
from app.services.agents.v2.semantic.intent import IntentDecision

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")

FULL_CAPABILITIES = frozenset(
    {
        "people.lookup",
        "document.search",
        "document.read",
        "section.read",
        "knowledge_graph.query",
        "memory.lookup",
    }
)


def make_semantic(query: str) -> SemanticContext:
    return SemanticContext(
        contextualized_query=query,
        normalized_query=query,
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def make_intent(intent: str, *, source: str = "model") -> IntentDecision:
    return IntentDecision(
        intent=intent,
        source=source,  # type: ignore[arg-type]
        confidence=None,
        needs_memory=False,
        is_legal_query=False,
    )


def make_runtime(*, intent_classifier: Any = None) -> Runtime:
    context = GraphRuntimeContext(
        capability_runtime=CapabilityRuntimeContext(
            request_id="req-1",
            run_id="run-1",
            user_id=USER_ID,
            workspace_ids=(WORKSPACE_ID,),
            can_read_people=True,
            allowed_capabilities=FULL_CAPABILITIES,
            deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        services=RuntimeServices(intent_classifier=intent_classifier),
    )
    return Runtime(context=context)


def make_state(query: str) -> SupervisorV2State:
    return SupervisorV2State(
        contract_version="2.0",
        request=RequestContext(
            contract_version="2.0",
            request_id="req-1",
            thread_id="thread-1",
            original_query=query,
            known_documents=(),
        ),
        conversation=ConversationContext(
            summary="",
            active_entities=(),
            last_focus=None,
            recent_turns=(),
        ),
        semantic=make_semantic(query),
        bindings=DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        query_analysis=None,
        route_decision=None,
        execution=ExecutionState(plan=None, task_results=(), evidence_evaluation=None),
        clarification=None,
        final_response=None,
    )


# ---------------------------------------------------------------------------
# Required mapping table (brief, Task 3)
# ---------------------------------------------------------------------------

MAPPING_CASES = [
    ("mongo_search_cccd", "lookup", ("people",)),
    ("mongo_search_phone", "lookup", ("people",)),
    ("mongo_search_bhxh", "lookup", ("people",)),
    ("mongo_search_name", "lookup", ("people",)),
    ("mongo_search_advanced", "lookup", ("people",)),
    ("search", "retrieve", ("document",)),
    ("search_doc_num", "retrieve", ("document",)),
    ("search_section", "retrieve", ("document", "section")),
    ("summarize", "summarize", ("document",)),
    ("kg_query", "lookup", ("knowledge_graph",)),
    ("greeting", "direct", ("memory",)),
]


@pytest.mark.parametrize(
    "intent,work_type,domains", MAPPING_CASES, ids=[case[0] for case in MAPPING_CASES]
)
def test_typed_intent_maps_to_query_analysis(
    intent: str, work_type: str, domains: tuple[str, ...]
) -> None:
    analysis = analyze_query(make_semantic("bất kỳ truy vấn nào"), intent=make_intent(intent))
    assert analysis.work_type == work_type
    assert analysis.domains == domains


def test_resolve_doc_is_prerequisite_not_complexity() -> None:
    analysis = analyze_query(
        make_semantic("Tóm tắt Nghị định A"), intent=make_intent("resolve_doc")
    )
    assert analysis.work_type == "retrieve"
    assert analysis.domains == ("document",)


def test_typed_people_name_beats_la_ai_keyword() -> None:
    # "là ai" is a KG keyword in the legacy regexes; typed people intent wins.
    analysis = analyze_query(
        make_semantic("Nguyễn Văn A là ai?"),
        intent=make_intent("mongo_search_name"),
    )
    assert (analysis.work_type, analysis.domains) == ("lookup", ("people",))


def test_typed_search_beats_khac_biet_compare_keyword() -> None:
    analysis = analyze_query(
        make_semantic("sự khác biệt giữa nghỉ phép và nghỉ ốm?"),
        intent=make_intent("search"),
    )
    assert (analysis.work_type, analysis.domains) == ("retrieve", ("document",))


def test_typed_search_beats_tong_hop_summary_keyword() -> None:
    analysis = analyze_query(
        make_semantic("tổng hợp quy định về nghỉ phép năm?"),
        intent=make_intent("search"),
    )
    assert (analysis.work_type, analysis.domains) == ("retrieve", ("document",))


def test_typed_search_beats_danh_gia_evaluate_keyword() -> None:
    analysis = analyze_query(
        make_semantic("đánh giá chung về chế độ thai sản?"),
        intent=make_intent("search"),
    )
    assert (analysis.work_type, analysis.domains) == ("retrieve", ("document",))


def test_typed_greeting_prefix_factual_is_retrieve() -> None:
    # Legacy text path reads the "chào" prefix as conversational; typed
    # search intent (Task-1 case greeting-prefix-factual) is retrieve.
    analysis = analyze_query(
        make_semantic("chào anh, hỏi về chế độ thai sản?"),
        intent=make_intent("search"),
    )
    assert (analysis.work_type, analysis.domains) == ("retrieve", ("document",))


def test_no_confidence_on_frozen_query_analysis() -> None:
    assert "confidence" not in QueryAnalysis.model_fields
    analysis = analyze_query(
        make_semantic("chế độ thai sản được quy định thế nào?"),
        intent=make_intent("search"),
    )
    assert "confidence" not in analysis.model_dump()


def test_legacy_path_unchanged_without_intent() -> None:
    # No typed intent: the deterministic regex behavior is preserved exactly.
    legacy = analyze_query(make_semantic("chào anh, hỏi về chế độ thai sản?"))
    assert (legacy.work_type, legacy.domains) == ("explain", ("memory",))
    legacy_people = analyze_query(make_semantic("Nguyễn Văn A là ai?"))
    assert "knowledge_graph" in legacy_people.domains


def test_unknown_intent_falls_back_to_legacy() -> None:
    legacy = analyze_query(make_semantic("chế độ thai sản được quy định thế nào?"))
    fallback = analyze_query(
        make_semantic("chế độ thai sản được quy định thế nào?"),
        intent="not_a_real_intent",  # type: ignore[arg-type]
    )
    assert (fallback.work_type, fallback.domains) == (
        legacy.work_type,
        legacy.domains,
    )


def test_intent_accepts_plain_taxonomy_string() -> None:
    analysis = analyze_query(
        make_semantic("Tóm tắt Nghị định A"), intent="summarize"  # type: ignore[arg-type]
    )
    assert (analysis.work_type, analysis.domains) == ("summarize", ("document",))


# ---------------------------------------------------------------------------
# Route authority stays with deterministic decide_route()
# ---------------------------------------------------------------------------


class FakeIntentClassifier:
    """Minimal stand-in exposing the cached() seam route_node consumes."""

    def __init__(self, decision: IntentDecision | None) -> None:
        self._decision = decision

    def cached(self, query: str, *, has_doc_ids: bool = False) -> IntentDecision | None:
        _ = (query, has_doc_ids)
        return self._decision


@pytest.mark.asyncio
async def test_route_node_consumes_cached_typed_intent() -> None:
    query = "Nguyễn Văn A là ai?"
    state = make_state(query)
    runtime = make_runtime(
        intent_classifier=FakeIntentClassifier(make_intent("mongo_search_name"))
    )
    update = await route_node(state, runtime)
    assert (update["query_analysis"].work_type, update["query_analysis"].domains) == (
        "lookup",
        ("people",),
    )
    # Deterministic policy still owns the route: people lookup fast path.
    assert update["route_decision"].route == "fast_domain"
    assert update["route_decision"].reason_code == "simple_people_lookup"


@pytest.mark.asyncio
async def test_route_node_without_classifier_keeps_legacy_behavior() -> None:
    query = "Nguyễn Văn A là ai?"
    state = make_state(query)
    update = await route_node(state, make_runtime(intent_classifier=None))
    legacy = analyze_query(make_semantic(query))
    assert update["query_analysis"].work_type == legacy.work_type
    assert update["query_analysis"].domains == legacy.domains


@pytest.mark.asyncio
async def test_route_node_typed_retrieve_still_needs_decide_route() -> None:
    # Typed retrieve/document with no binding must NOT become a fast path by
    # itself: decide_route() owns that call (Task 4 owns the targetless gate).
    query = "chế độ thai sản được quy định thế nào?"
    state = make_state(query)
    runtime = make_runtime(
        intent_classifier=FakeIntentClassifier(make_intent("search"))
    )
    update = await route_node(state, runtime)
    expected = decide_route(
        update["query_analysis"],
        state["semantic"],
        state["bindings"],
        allowed_capabilities=FULL_CAPABILITIES,
        request=state["request"],
    )
    assert update["route_decision"] == expected

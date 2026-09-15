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
from app.services.agents.v2.contracts.conversation import EntityReference
from app.services.agents.v2.contracts.semantic import (
    DocumentReference,
    SectionReference,
    SemanticContext,
)
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
        "document.retrieve",
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
    # resolve_doc is a prerequisite marker without terminal semantics: it
    # falls back to the legacy reference/text path, which never treats
    # resolution cost as execution complexity.
    analysis = analyze_query(
        make_semantic("Tóm tắt Nghị định A"), intent=make_intent("resolve_doc")
    )
    assert analysis.work_type == "summarize"
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
# Fix round 1/5 (I1) — typed path must preserve typed ref-derived semantics.
# Only generic lexical patterns are demoted; person/document/section refs
# and reference-count arity (compare/multi_goal) survive typed intent.
# ---------------------------------------------------------------------------

DOC_ID_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_ID_B = UUID("22222222-2222-2222-2222-222222222222")


def make_resolved_doc_ref(ref_id: str, document_id: UUID) -> DocumentReference:
    return DocumentReference(
        ref_id=ref_id,
        original_span="tài liệu kiểm thử",
        normalized_reference="tài liệu kiểm thử",
        requested_role="target",
        revision_requirement=None,
        resolution_status="resolved",
        resolved_document_id=document_id,
    )


def make_section_ref(ref_id: str = "s1") -> SectionReference:
    return SectionReference(ref_id=ref_id, label="Điều 5", structure_node_id=None)


def make_person_ref(ref_id: str = "p1") -> EntityReference:
    return EntityReference(ref_id=ref_id, kind="person", label="Nguyễn Văn A")


def make_semantic_with_refs(
    query: str,
    *,
    document_refs: tuple[DocumentReference, ...] = (),
    person_refs: tuple[EntityReference, ...] = (),
    section_refs: tuple[SectionReference, ...] = (),
) -> SemanticContext:
    return SemanticContext(
        contextualized_query=query,
        normalized_query=query,
        abbreviations=(),
        coreferences=(),
        document_refs=document_refs,
        person_refs=person_refs,
        section_refs=section_refs,
        blocking_ambiguities=(),
    )


def test_typed_search_two_resolved_docs_is_compare() -> None:
    # I1.1: two resolved targets + typed search must reach the compare
    # skill (comparison topology), not a targetless retrieve.
    semantic = make_semantic_with_refs(
        "so sánh hai văn bản",
        document_refs=(
            make_resolved_doc_ref("r1", DOC_ID_A),
            make_resolved_doc_ref("r2", DOC_ID_B),
        ),
    )
    analysis = analyze_query(semantic, intent=make_intent("search"))
    assert (analysis.work_type, analysis.domains) == ("compare", ("document",))


def test_typed_search_three_resolved_docs_is_multi_goal() -> None:
    semantic = make_semantic_with_refs(
        "tra cứu ba văn bản",
        document_refs=(
            make_resolved_doc_ref("r1", DOC_ID_A),
            make_resolved_doc_ref("r2", DOC_ID_B),
            make_resolved_doc_ref(
                "r3", UUID("33333333-3333-3333-3333-333333333333")
            ),
        ),
    )
    analysis = analyze_query(semantic, intent=make_intent("search"))
    assert (analysis.work_type, analysis.domains) == ("multi_goal", ("document",))


def test_typed_search_keeps_section_refs() -> None:
    # I1.2: spec §9 / Task 7 gate needs QueryAnalysis(retrieve,
    # document+section) to stay achievable with the classifier wired.
    semantic = make_semantic_with_refs(
        "Điều 5 nói gì?",
        document_refs=(make_resolved_doc_ref("r1", DOC_ID_A),),
        section_refs=(make_section_ref(),),
    )
    analysis = analyze_query(semantic, intent=make_intent("search"))
    assert (analysis.work_type, analysis.domains) == (
        "retrieve",
        ("document", "section"),
    )


def test_typed_search_keeps_person_refs_as_cross_domain() -> None:
    # Typed ref-derived people + document families stay cross-domain,
    # exactly as the legacy path derives from the same refs.
    semantic = make_semantic_with_refs(
        "tra cứu văn bản",
        document_refs=(make_resolved_doc_ref("r1", DOC_ID_A),),
        person_refs=(make_person_ref(),),
    )
    typed = analyze_query(semantic, intent=make_intent("search"))
    legacy = analyze_query(semantic)
    assert (typed.work_type, typed.domains) == ("cross_domain", ("document", "people"))
    assert (legacy.work_type, legacy.domains) == (typed.work_type, typed.domains)


def test_resolve_doc_falls_back_to_legacy() -> None:
    # I1.3: resolve_doc carries no terminal semantics (the v1 task plan is
    # dropped at the Task-2 boundary), so analysis falls back to the
    # reference/text path instead of a fixed retrieve/document.
    for query in (
        "Tóm tắt Nghị định A",
        "Điều 5 Luật An ninh mạng quy định gì?",
        "chế độ thai sản được quy định thế nào?",
    ):
        semantic = make_semantic(query)
        typed = analyze_query(semantic, intent=make_intent("resolve_doc"))
        legacy = analyze_query(semantic)
        assert (typed.work_type, typed.domains) == (legacy.work_type, legacy.domains)


def test_corpus_summarize_named_doc_live_resolve_doc() -> None:
    # I1.3 driven from the frozen Task-1 corpus: v1 emits resolve_doc as the
    # first step for "Tóm tắt Nghị định A" (corpus v1_prerequisite), and the
    # terminal v2 target stays summarize/document.
    from tests.agents.v2.golden.intent_cases import INTENT_CASES

    case = next(c for c in INTENT_CASES if c["id"] == "summarize-named-doc")
    assert case["v1_prerequisite"] == "resolve_doc"
    analysis = analyze_query(
        make_semantic(case["query"]), intent=make_intent("resolve_doc")
    )
    assert (analysis.work_type, analysis.domains) == (
        case["v2_work_type"],
        tuple(case["v2_domains"]),
    )


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


# ---------------------------------------------------------------------------
# Final fix wave — I1 (evaluate reachable with classifier wired), I4
# (typed ``personal`` falls back to the legacy fast path), Mod5 (typed
# path keeps the write boundary).
# ---------------------------------------------------------------------------


def _raising_provider_factory():
    raise AssertionError("deterministic scope must not invoke the model")


class _FakeSearchChunk:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeSearchProvider:
    """A model that WOULD misclassify the compliance query to ``search``."""

    def __init__(self) -> None:
        self.called = False

    async def astream(self, messages, **kwargs):
        self.called = True
        yield _FakeSearchChunk(
            '{"intent":"search","needs_memory":false,"is_legal_query":true}'
        )


def test_classify_evaluate_deterministic_scope() -> None:
    from app.services.agents.v2.semantic.intent import classify_evaluate

    for query in (
        "Đánh giá tuân thủ quy định nội bộ",
        "Đánh giá tính pháp lý của hợp đồng mới",
        "Tôi có tuân thủ quy định nội bộ không?",
        "Kiểm tra mức độ tuân thủ an toàn thông tin",
    ):
        decision = classify_evaluate(query)
        assert decision is not None, query
        assert decision.intent == "evaluate"
        assert decision.source == "deterministic"
        assert decision.is_legal_query is True

    # Bare ``đánh giá`` is a generic keyword under the Task-3 ruling, not a
    # compliance signal: it must stay reachable by the model path.
    assert classify_evaluate("đánh giá chung về chế độ thai sản?") is None
    assert classify_evaluate("chế độ thai sản được quy định thế nào?") is None


def test_classify_evaluate_informational_topic_not_assessment() -> None:
    """Final re-review I5: ordinary informational compliance-topic queries
    (topic mention, no assessment request) must NOT classify as evaluate."""
    from app.services.agents.v2.semantic.intent import classify_evaluate

    for query in (
        "quy định về tuân thủ thuế là gì?",
        "chế tài xử phạt khi không tuân thủ quy định?",
        "hồ sơ tuân thủ gồm những giấy tờ gì?",
        "tuân thủ pháp luật có ý nghĩa gì?",
        "compliance là gì?",
    ):
        assert classify_evaluate(query) is None, query


def test_classify_evaluate_short_circuits_model() -> None:
    from app.services.agents.v2.semantic.intent import IntentClassifier

    classifier = IntentClassifier(provider_factory=_raising_provider_factory)

    async def _run() -> IntentDecision:
        return await classifier.classify("Đánh giá tuân thủ quy định nội bộ")

    import asyncio

    decision = asyncio.run(_run())
    assert decision.intent == "evaluate"
    assert decision.source == "deterministic"


@pytest.mark.asyncio
async def test_route_node_compliance_with_classifier_wired() -> None:
    # Final review I1: with the production classifier wired and a model that
    # would emit the generic ``search`` intent, a compliance query must still
    # reach the governed evaluate/compliance topology — never the targetless
    # document retrieval the typed ``search`` misclassification produced
    # before this fix.
    from app.services.agents.v2.semantic.intent import IntentClassifier

    query = "Đánh giá tuân thủ quy định nội bộ"
    state = make_state(query)
    provider = _FakeSearchProvider()
    runtime = make_runtime(
        intent_classifier=IntentClassifier(
            provider_factory=lambda: provider
        )
    )
    update = await route_node(state, runtime)
    assert update["query_analysis"].work_type == "evaluate"
    assert update["query_analysis"].domains == ("document",)
    assert update["route_decision"].route == "complex_research"
    assert update["route_decision"].reason_code == "compliance_evaluation"
    # The deterministic evaluate scope short-circuits before the model.
    assert provider.called is False


@pytest.mark.parametrize(
    "query",
    [
        "quy định về tuân thủ thuế là gì?",
        "chế tài xử phạt khi không tuân thủ quy định?",
        "hồ sơ tuân thủ gồm những giấy tờ gì?",
    ],
)
@pytest.mark.asyncio
async def test_route_node_informational_compliance_topic_stays_targetless(
    query: str,
) -> None:
    """Final re-review I5: a compliance-topic question with no assessment
    request stays the general-RAG targetless fast path; the model is still
    consulted (advisory) but the deterministic router remains authoritative."""
    from app.services.agents.v2.semantic.intent import IntentClassifier

    state = make_state(query)
    provider = _FakeSearchProvider()
    runtime = make_runtime(
        intent_classifier=IntentClassifier(provider_factory=lambda: provider)
    )
    update = await route_node(state, runtime)
    assert update["query_analysis"].work_type == "retrieve"
    assert update["query_analysis"].domains == ("document",)
    assert update["route_decision"].route == "fast_domain"
    assert update["route_decision"].reason_code == "targetless_document_retrieval"
    # Model advisory boundary preserved: informational queries consult the
    # model (no deterministic short-circuit), but route authority stays
    # deterministic.
    assert provider.called is True


def test_typed_personal_falls_back_to_legacy_fast_path() -> None:
    # Final review I4: typed ``personal`` had no ``direct`` branch, so it
    # fell through to complex_research while the legacy path served it fast.
    # Until a frozen-reason ruling records typed ``direct``, ``personal``
    # must fall back to the legacy deterministic path.
    for query in ("Tôi là ai?", "Tôi đang công tác tại đâu?"):
        typed = analyze_query(make_semantic(query), intent=make_intent("personal"))
        legacy = analyze_query(make_semantic(query))
        assert (typed.work_type, typed.domains) == (
            legacy.work_type,
            legacy.domains,
        )


@pytest.mark.asyncio
async def test_route_node_typed_personal_served_fast() -> None:
    query = "Tôi là ai?"
    state = make_state(query)
    runtime = make_runtime(
        intent_classifier=FakeIntentClassifier(make_intent("personal"))
    )
    update = await route_node(state, runtime)
    assert update["route_decision"].route == "fast_domain"
    assert update["route_decision"].reason_code == "simple_kg_lookup"


def test_typed_search_preserves_write_boundary() -> None:
    # Final review Mod5: a write request misclassified to typed ``search``
    # must still surface the write domain (defence-in-depth), never a
    # document-read fast path.
    analysis = analyze_query(
        make_semantic("Viết báo cáo tổng kết năm"), intent=make_intent("search")
    )
    assert "write" in analysis.domains
    decision = decide_route(
        analysis,
        make_semantic("Viết báo cáo tổng kết năm"),
        DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        allowed_capabilities=FULL_CAPABILITIES,
    )
    assert decision.route == "complex_research"
    assert decision.reason_code == "simple_write_operation"

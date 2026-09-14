"""Task 4 — general factual RAG targetless fast path (TDD, failing first).

Proves the Phase 4A gate for reference-free factual retrieval:

- ``search -> retrieve/document`` with no explicit document binding is a
  bounded targetless ``document.retrieve`` fast path (never requires exactly
  one binding);
- ``build_fast_plan`` maps the targetless reason to a single-task
  ``document.retrieve`` plan with empty ``target_ids`` (workspace scope);
- route decisions check the ACTUAL available capability catalog, not only
  ``allowed_capabilities``: allowed-but-unavailable falls back to a
  deterministic ``complex_research``/``runtime_dependency`` decision and never
  routes into an execution path that cannot exist;
- a model classification call itself never makes a query complex
  (deterministic- and model-source intents yield identical topology);
- Phase 4A gate trio via typed intent: people fast path, greeting-prefixed
  factual fast path, bare general factual fast path.
"""
from __future__ import annotations

from uuid import UUID

import pytest

from app.services.agents.v2.contracts.binding import DocumentBindingSet
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.validation import validate_fast_plan
from app.services.agents.v2.nodes.routing import analyze_query, decide_route
from app.services.agents.v2.semantic.intent import IntentDecision

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")

#: Catalog in which the retrieval capability is genuinely deployable.
RETRIEVE_AVAILABLE = frozenset({"document.retrieve"})
#: Request permission grants retrieval, but the deployment cannot serve it.
RETRIEVE_ALLOWED_ONLY = frozenset({"document.retrieve"})
RETRIEVE_UNAVAILABLE: frozenset[str] = frozenset()


def bare_factual_semantic(
    query: str = "chế độ thai sản được quy định thế nào?",
) -> SemanticContext:
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


def empty_bindings() -> DocumentBindingSet:
    return DocumentBindingSet(bindings=(), revision_requirement_refs=())


def search_intent(source: str = "deterministic") -> IntentDecision:
    return IntentDecision(
        intent="search",
        source=source,  # type: ignore[arg-type]
        confidence=None,
        needs_memory=False,
        is_legal_query=True,
    )


# ---------------------------------------------------------------------------
# Targetless topology: no binding required
# ---------------------------------------------------------------------------


def test_reference_free_factual_is_targetless_fast_path() -> None:
    semantic = bare_factual_semantic()
    analysis = analyze_query(semantic, intent=search_intent())
    assert (analysis.work_type, analysis.domains) == ("retrieve", ("document",))
    decision = decide_route(
        analysis,
        semantic,
        empty_bindings(),
        allowed_capabilities=RETRIEVE_ALLOWED_ONLY,
        available_capabilities=RETRIEVE_AVAILABLE,
    )
    assert decision.route == "fast_domain"
    assert decision.reason_code == "targetless_document_retrieval"


def test_targetless_fast_plan_is_single_document_retrieve_task() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    semantic = bare_factual_semantic()
    analysis = analyze_query(semantic, intent=search_intent())
    decision = decide_route(
        analysis,
        semantic,
        empty_bindings(),
        allowed_capabilities=RETRIEVE_ALLOWED_ONLY,
        available_capabilities=RETRIEVE_AVAILABLE,
    )
    plan = build_fast_plan(semantic, empty_bindings(), analysis, decision)
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.capability == "document.retrieve"
    assert task.input.kind == "document.retrieve"
    assert task.input.target_ids == ()
    assert plan.target_units == ()
    assert task.depends_on == ()
    validate_fast_plan(plan, empty_bindings())


def test_greeting_prefix_factual_is_targetless_fast_path() -> None:
    """Gate case: greeting prefix + factual remainder is factual, not direct."""
    semantic = bare_factual_semantic("chào anh, hỏi về chế độ thai sản?")
    analysis = analyze_query(semantic, intent=search_intent())
    decision = decide_route(
        analysis,
        semantic,
        empty_bindings(),
        allowed_capabilities=RETRIEVE_ALLOWED_ONLY,
        available_capabilities=RETRIEVE_AVAILABLE,
    )
    assert (decision.route, decision.reason_code) == (
        "fast_domain",
        "targetless_document_retrieval",
    )


def test_general_compare_is_targetless_fast_path() -> None:
    """Final review Mod1: rag-general-compare is a factual retrieve, not a
    comparison topology — the bare ``khác biệt`` keyword stays demoted under
    typed ``search``, so the frozen corpus targetless reason must hold here
    too (the only Phase 4A gate case that previously lacked a route pin)."""
    semantic = bare_factual_semantic(
        "sự khác biệt giữa nghỉ phép và nghỉ ốm?"
    )
    analysis = analyze_query(semantic, intent=search_intent())
    assert (analysis.work_type, analysis.domains) == ("retrieve", ("document",))
    decision = decide_route(
        analysis,
        semantic,
        empty_bindings(),
        allowed_capabilities=RETRIEVE_ALLOWED_ONLY,
        available_capabilities=RETRIEVE_AVAILABLE,
    )
    assert (decision.route, decision.reason_code) == (
        "fast_domain",
        "targetless_document_retrieval",
    )


def test_corpus_targetless_cases_route_via_decide_route() -> None:
    """Final review Mod1: every frozen-corpus targetless case now carries the
    approved frozen reason and reaches it through the real ``decide_route``."""
    from tests.agents.v2.golden.intent_cases import INTENT_CASES

    targetless_ids = {
        "greeting-prefix-factual",
        "rag-general-thai-san",
        "rag-general-compare",
    }
    for case in INTENT_CASES:
        if case["id"] not in targetless_ids:
            continue
        assert case["v2_reason_code"] == "targetless_document_retrieval"
        semantic = bare_factual_semantic(case["query"])
        analysis = analyze_query(semantic, intent=search_intent())
        decision = decide_route(
            analysis,
            semantic,
            empty_bindings(),
            allowed_capabilities=RETRIEVE_ALLOWED_ONLY,
            available_capabilities=RETRIEVE_AVAILABLE,
        )
        assert (decision.route, decision.reason_code) == (
            "fast_domain",
            "targetless_document_retrieval",
        )


def test_people_gate_case_stays_people_fast_path() -> None:
    """Gate case: Nguyễn Văn A là ai? -> people fast path (unchanged)."""
    from app.services.agents.v2.contracts.conversation import EntityReference

    semantic = SemanticContext(
        contextualized_query="Nguyễn Văn A là ai?",
        normalized_query="Nguyễn Văn A là ai?",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(
            EntityReference(ref_id="p1", kind="person", label="Nguyễn Văn A"),
        ),
        section_refs=(),
        blocking_ambiguities=(),
    )
    analysis = analyze_query(
        semantic,
        intent=IntentDecision(
            intent="mongo_search_name",
            source="model",
            confidence=0.9,
            needs_memory=False,
            is_legal_query=False,
        ),
    )
    decision = decide_route(
        analysis,
        semantic,
        empty_bindings(),
        allowed_capabilities=frozenset({"people.lookup"}),
        available_capabilities=frozenset({"people.lookup"}),
    )
    assert (decision.route, decision.reason_code) == (
        "fast_domain",
        "simple_people_lookup",
    )


# ---------------------------------------------------------------------------
# Actual catalog availability, not only allowed flags
# ---------------------------------------------------------------------------


def test_allowed_but_unavailable_retrieval_falls_back_deterministically() -> None:
    """Must not route into an execution path that cannot exist."""
    semantic = bare_factual_semantic()
    analysis = analyze_query(semantic, intent=search_intent())
    decision = decide_route(
        analysis,
        semantic,
        empty_bindings(),
        allowed_capabilities=RETRIEVE_ALLOWED_ONLY,
        available_capabilities=RETRIEVE_UNAVAILABLE,
    )
    assert (decision.route, decision.reason_code) == (
        "complex_research",
        "runtime_dependency",
    )


def test_unavailable_capability_never_builds_a_fast_plan() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    semantic = bare_factual_semantic()
    analysis = analyze_query(semantic, intent=search_intent())
    decision = decide_route(
        analysis,
        semantic,
        empty_bindings(),
        allowed_capabilities=RETRIEVE_ALLOWED_ONLY,
        available_capabilities=RETRIEVE_UNAVAILABLE,
    )
    with pytest.raises(ValueError, match="fast_domain"):
        build_fast_plan(semantic, empty_bindings(), analysis, decision)


# ---------------------------------------------------------------------------
# Model classification never makes a query complex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["deterministic", "model"])
def test_intent_source_never_changes_topology(source: str) -> None:
    semantic = bare_factual_semantic()
    analysis = analyze_query(semantic, intent=search_intent(source))
    decision = decide_route(
        analysis,
        semantic,
        empty_bindings(),
        allowed_capabilities=RETRIEVE_ALLOWED_ONLY,
        available_capabilities=RETRIEVE_AVAILABLE,
    )
    assert analysis.work_type == "retrieve"
    assert (decision.route, decision.reason_code) == (
        "fast_domain",
        "targetless_document_retrieval",
    )


def test_targetless_plan_dispatches_through_shared_scheduler() -> None:
    """Targetless plan executes via TaskScheduler + registry (no direct exec)."""
    import asyncio
    from datetime import UTC, datetime

    from app.services.agents.v2.capabilities import (
        CapabilityRegistration,
        build_capability_registry,
    )
    from app.services.agents.v2.contracts.base import CONTRACT_VERSION
    from app.services.agents.v2.contracts.capability import (
        CapabilityDescriptor,
        CapabilityRuntimeContext,
        DocumentRetrieveOutput,
    )
    from app.services.agents.v2.contracts.evidence import EvidenceUseRef
    from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
    from app.services.agents.v2.contracts.state import (
        GraphRuntimeContext,
        RuntimeServices,
    )
    from app.services.agents.v2.execution.scheduler import TaskScheduler
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    class StubRetrieveCapability:
        descriptor = CapabilityDescriptor(
            name="document.retrieve",
            domain="document",
            operation_type="search",
            supports_parallel=False,
        )

        def __init__(self, result: AgentResult) -> None:
            self._result = result
            self.calls: list[tuple[AgentRequest, CapabilityRuntimeContext]] = []

        async def execute(
            self, request: AgentRequest, runtime: CapabilityRuntimeContext
        ) -> AgentResult:
            assert isinstance(request, AgentRequest)
            assert isinstance(runtime, CapabilityRuntimeContext)
            self.calls.append((request, runtime))
            return self._result

    semantic = bare_factual_semantic()
    analysis = analyze_query(semantic, intent=search_intent())
    decision = decide_route(
        analysis,
        semantic,
        empty_bindings(),
        allowed_capabilities=RETRIEVE_ALLOWED_ONLY,
        available_capabilities=RETRIEVE_AVAILABLE,
    )
    plan = build_fast_plan(semantic, empty_bindings(), analysis, decision)

    task_id = plan.tasks[0].task_id
    stub = StubRetrieveCapability(
        AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=task_id,
            status="success",
            data=DocumentRetrieveOutput(
                kind="document.retrieve", retrieved_unit_count=3
            ),
            evidence_uses=(EvidenceUseRef(use_id=UUID(int=7)),),
            coverage_observations=(),
            error=None,
        )
    )
    runtime = CapabilityRuntimeContext(
        request_id="req-1",
        run_id="run-1",
        user_id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
        workspace_ids=(UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),),
        can_read_people=False,
        allowed_capabilities=RETRIEVE_ALLOWED_ONLY,
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    registry = build_capability_registry(
        [CapabilityRegistration(capability=stub)], runtime
    )

    class _FakeSession:
        async def commit(self) -> None:
            return None

    class _FakeLeases:
        session = _FakeSession()

        async def acquire_or_refresh(self, *args: object, **kwargs: object) -> None:
            return None

    context = GraphRuntimeContext(
        capability_runtime=runtime,
        services=RuntimeServices(retention_leases=_FakeLeases()),
    )
    report = asyncio.run(
        TaskScheduler(registry).execute(
            plan=plan, runtime=context, prior_results=(), bindings=empty_bindings()
        )
    )
    assert len(report.results) == 1
    assert report.results[0].status == "success"
    (request, _) = stub.calls[0]
    assert isinstance(request, AgentRequest)
    assert request.task_id == task_id

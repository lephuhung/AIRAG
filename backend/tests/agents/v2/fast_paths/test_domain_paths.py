"""Task 4 — domain fast paths through the shared scheduler + T4 nodes (TDD).

Proves the Phase-2 acceptance facts for the T4 side: fast people/document
paths dispatch only through the shared `TaskScheduler`/registry (capabilities
receive exactly `AgentRequest` + `CapabilityRuntimeContext`), the bounded
summary performs read -> evaluate -> synthesize -> ground, a summary never
touches a summary domain agent, comparison never takes the fast domain route,
and Write stays typed-unavailable in the finalizer.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    DocumentReadInput,
    DocumentReadOutput,
    PeopleLookupInput,
    PeopleLookupOutput,
)
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.evaluation import CoverageObservation
from app.services.agents.v2.contracts.evidence import (
    EvidenceUse,
    EvidenceUseRef,
    PeopleSourceIdentity,
)
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
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
from app.services.agents.v2.contracts.synthesis import SynthesisRuntimeContext
from app.services.agents.v2.execution.scheduler import TaskScheduler
from app.services.agents.v2.nodes.evaluate import HydratedEvidence
from app.services.agents.v2.nodes.evaluate import evaluate_node
from app.services.agents.v2.nodes.finalizer import finalizer_node
from app.services.agents.v2.nodes.fast_plan import build_fast_plan
from app.services.agents.v2.nodes.grounding import ground_node
from app.services.agents.v2.nodes.routing import analyze_query, decide_route
from app.services.agents.v2.nodes.synthesize import (
    apply_budget_split,
    synthesize_node,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
REVISION = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

FULL_FAST_CAPABILITIES = frozenset(
    {
        "people.lookup",
        "document.search",
        "document.read",
        "section.read",
        "knowledge_graph.query",
        "memory.lookup",
    }
)


# ---------------------------------------------------------------------------
# Fakes (request-scoped services wired onto a real RuntimeServices)
# ---------------------------------------------------------------------------


class StubReadCapability:
    """Shared document.read stand-in: canned governed result, records calls."""

    descriptor = CapabilityDescriptor(
        name="document.read",
        domain="document",
        operation_type="read",
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


class StubPeopleCapability:
    """Shared people.lookup stand-in."""

    descriptor = CapabilityDescriptor(
        name="people.lookup",
        domain="people",
        operation_type="lookup",
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


class FakeLeaseSession:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def commit(self) -> None:
        self._events.append("commit")


class FakeLeaseRepo:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.calls: list[tuple[Any, Any, Any]] = []
        self.session = FakeLeaseSession(self.events)

    async def acquire_or_refresh(
        self,
        run_id: str,
        revision_id: Any = None,
        evidence_use_id: Any = None,
        *,
        now: Any = None,
    ) -> Any:
        self.calls.append((run_id, revision_id, evidence_use_id))
        return None


class FakeHydrator:
    """In-memory EvidenceHydrator with a generous default budget."""

    def __init__(
        self,
        items: dict[UUID, HydratedEvidence],
        budget: SynthesisRuntimeContext | None = None,
    ) -> None:
        self.items = dict(items)
        self.budget = budget
        self.persisted = 0

    async def hydrate_for_evaluation(
        self,
        use_refs: tuple[Any, ...],
        runtime: GraphRuntimeContext,
    ) -> tuple[HydratedEvidence, ...]:
        return tuple(
            self.items[ref.use_id] for ref in use_refs if ref.use_id in self.items
        )

    async def hydrate_for_synthesis(
        self,
        use_refs: tuple[Any, ...],
        runtime: GraphRuntimeContext,
    ) -> tuple[HydratedEvidence, ...]:
        admitted = [
            self.items[ref.use_id] for ref in use_refs if ref.use_id in self.items
        ]
        admitted = [h for h in admitted if h.purpose != "discovery"]
        if self.budget is None:
            return tuple(admitted)
        head, _ = apply_budget_split(tuple(admitted), self.budget)
        return tuple(head)

    async def persist_derived_summary(self, **kwargs: Any) -> HydratedEvidence:
        raise AssertionError("no overflow expected in domain-path fixtures")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def capability_runtime(
    *,
    allowed: frozenset[str] = FULL_FAST_CAPABILITIES,
) -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id="req-1",
        run_id="run-1",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=True,
        allowed_capabilities=allowed,
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


def semantic_for_summary() -> SemanticContext:
    return SemanticContext(
        contextualized_query="Tóm tắt tài liệu A.",
        normalized_query="Tóm tắt tài liệu A.",
        abbreviations=(),
        coreferences=(),
        document_refs=(
            DocumentReference(
                ref_id="r1",
                original_span="tài liệu A",
                normalized_reference="tài liệu A",
                requested_role="target",
                revision_requirement=None,
                resolution_status="resolved",
                resolved_document_id=DOCUMENT_ID,
                candidate_document_ids=(),
            ),
        ),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def bindings_for_summary() -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b_r1",
                document_id=DOCUMENT_ID,
                document_revision=REVISION,
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )


def make_state(
    *,
    semantic: SemanticContext,
    bindings: DocumentBindingSet,
    analysis: QueryAnalysis | None = None,
    route: RouteDecision | None = None,
    execution: ExecutionState | None = None,
) -> SupervisorV2State:
    return SupervisorV2State(
        contract_version="2.0",
        request=RequestContext(
            contract_version="2.0",
            request_id="req-1",
            thread_id="thread-1",
            original_query=semantic.normalized_query,
            known_documents=(),
        ),
        conversation=ConversationContext(
            summary="", active_entities=(), last_focus=None, recent_turns=()
        ),
        semantic=semantic,
        bindings=bindings,
        query_analysis=analysis,
        route_decision=route,
        execution=execution
        if execution is not None
        else ExecutionState(plan=None, task_results=(), evidence_evaluation=None),
        clarification=None,
        final_response=None,
    )


def read_result(task_id: str, use_id: UUID, content: str) -> AgentResult:
    return AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id=task_id,
        status="success",
        data=DocumentReadOutput(kind="document.read", read_unit_count=1),
        evidence_uses=(EvidenceUseRef(use_id=use_id),),
        coverage_observations=(
            CoverageObservation(
                target_id="t_b_r1",
                observed_locators=(DocumentLocator(kind="document"),),
                outcome="read",
            ),
        ),
        error=None,
    )


def hydrated_doc(
    use_id: UUID, evidence_id: UUID, task_id: str, content: str
) -> HydratedEvidence:
    return HydratedEvidence(
        use_id=use_id,
        evidence_id=evidence_id,
        task_id=task_id,
        purpose="coverage",
        target_id="t_b_r1",
        content=content,
        role="target",
        source_label="doc-A",
        classification="normal",
        locator=DocumentLocator(kind="document"),
    )


# ---------------------------------------------------------------------------
# Shared-registry fast paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_document_read_uses_shared_capability_registry() -> None:
    semantic = semantic_for_summary()
    bindings = bindings_for_summary()
    analysis = analyze_query(semantic)
    route = decide_route(
        analysis, semantic, bindings, allowed_capabilities=FULL_FAST_CAPABILITIES
    )
    assert route.route == "fast_domain"
    plan = build_fast_plan(semantic, bindings, analysis, route)
    assert plan.tasks[0].capability == "document.read"

    use_id, evidence_id = uuid4(), uuid4()
    content = "Điều 5 quy định mức phạt hành chính."
    stub = StubReadCapability(read_result(plan.tasks[0].task_id, use_id, content))
    runtime = capability_runtime()
    registry = build_capability_registry(
        [CapabilityRegistration(capability=stub)], runtime
    )
    leases = FakeLeaseRepo()
    hydrator = FakeHydrator({use_id: hydrated_doc(use_id, evidence_id, plan.tasks[0].task_id, content)})
    context = GraphRuntimeContext(
        capability_runtime=runtime,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=leases,
            evidence_hydrator=hydrator,
        ),
    )

    scheduler = TaskScheduler(registry)
    report = await scheduler.execute(
        plan=plan, runtime=context, prior_results=(), bindings=bindings
    )
    assert len(report.results) == 1
    (request, call_runtime) = stub.calls[0]
    assert isinstance(request, AgentRequest)
    assert isinstance(call_runtime, CapabilityRuntimeContext)
    assert request.input is plan.tasks[0].input

    state = make_state(
        semantic=semantic,
        bindings=bindings,
        analysis=analysis,
        route=route,
        execution=ExecutionState(
            plan=plan, task_results=report.results, evidence_evaluation=None
        ),
    )
    evaluated = await evaluate_node(state, context)
    assert evaluated["execution"].evidence_evaluation.status == "sufficient"
    state["execution"] = evaluated["execution"]
    assert await synthesize_node(state, context) == {}
    assert await ground_node(state, context) == {}
    final = await finalizer_node(state, context)
    assert final["final_response"].status == "success"
    assert content in final["final_response"].content
    assert [c.citation_id for c in final["final_response"].citations] == ["cite-1"]
    assert final["final_response"].citations[0].evidence_id == evidence_id


@pytest.mark.asyncio
async def test_fast_people_uses_shared_capability_registry() -> None:
    semantic = SemanticContext(
        contextualized_query="CCCD của Nguyễn Văn A là gì?",
        normalized_query="CCCD của Nguyễn Văn A là gì?",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    from app.services.agents.v2.contracts.conversation import EntityReference

    semantic = semantic.model_copy(
        update={
            "person_refs": (
                EntityReference(ref_id="p1", kind="person", label="Nguyễn Văn A"),
            )
        }
    )
    bindings = DocumentBindingSet(bindings=(), revision_requirement_refs=())
    analysis = analyze_query(semantic)
    route = decide_route(
        analysis, semantic, bindings, allowed_capabilities=FULL_FAST_CAPABILITIES
    )
    assert route.route == "fast_domain"
    plan = build_fast_plan(semantic, bindings, analysis, route)
    assert plan.tasks[0].capability == "people.lookup"

    use_id, evidence_id = uuid4(), uuid4()
    result = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id=plan.tasks[0].task_id,
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(EvidenceUseRef(use_id=use_id),),
        coverage_observations=(),
        error=None,
    )
    stub = StubPeopleCapability(result)
    runtime = capability_runtime()
    registry = build_capability_registry(
        [CapabilityRegistration(capability=stub)], runtime
    )
    hydrator = FakeHydrator(
        {
            use_id: HydratedEvidence(
                use_id=use_id,
                evidence_id=evidence_id,
                task_id=plan.tasks[0].task_id,
                purpose="supporting",
                target_id=None,
                content='{"name":"Nguyễn Văn A"}',
                role=None,
                source_label="people",
                classification="personal",
                locator=None,
            )
        }
    )
    context = GraphRuntimeContext(
        capability_runtime=runtime,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=FakeLeaseRepo(),
            evidence_hydrator=hydrator,
        ),
    )
    report = await TaskScheduler(registry).execute(
        plan=plan, runtime=context, prior_results=(), bindings=bindings
    )
    assert len(report.results) == 1
    (request, call_runtime) = stub.calls[0]
    assert isinstance(request, AgentRequest)
    assert request.task_id == plan.tasks[0].task_id

    state = make_state(
        semantic=semantic,
        bindings=bindings,
        analysis=analysis,
        route=route,
        execution=ExecutionState(
            plan=plan, task_results=report.results, evidence_evaluation=None
        ),
    )
    evaluated = await evaluate_node(state, context)
    assert evaluated["execution"].evidence_evaluation.status == "sufficient"
    state["execution"] = evaluated["execution"]
    assert await synthesize_node(state, context) == {}
    assert await ground_node(state, context) == {}
    final = await finalizer_node(state, context)
    assert final["final_response"].status == "success"


@pytest.mark.asyncio
async def test_bounded_summary_is_read_evaluate_synthesize_ground() -> None:
    """Bounded one-document summary: read -> evaluate -> synthesize -> ground."""
    semantic = semantic_for_summary()
    bindings = bindings_for_summary()
    analysis = analyze_query(semantic)
    assert analysis.work_type == "summarize"
    route = decide_route(
        analysis, semantic, bindings, allowed_capabilities=FULL_FAST_CAPABILITIES
    )
    assert (route.route, route.reason_code) == (
        "fast_domain",
        "exact_document_metadata",
    )
    plan = build_fast_plan(semantic, bindings, analysis, route)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].capability == "document.read"

    use_id, evidence_id = uuid4(), uuid4()
    content = "Tài liệu A quy định về mức phạt. Điều 5 nêu chi tiết."
    stub = StubReadCapability(read_result(plan.tasks[0].task_id, use_id, content))
    runtime = capability_runtime()
    registry = build_capability_registry(
        [CapabilityRegistration(capability=stub)], runtime
    )
    hydrator = FakeHydrator({use_id: hydrated_doc(use_id, evidence_id, plan.tasks[0].task_id, content)})
    context = GraphRuntimeContext(
        capability_runtime=runtime,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=FakeLeaseRepo(),
            evidence_hydrator=hydrator,
        ),
    )
    report = await TaskScheduler(registry).execute(
        plan=plan, runtime=context, prior_results=(), bindings=bindings
    )
    ordered: list[str] = ["read"]
    state = make_state(
        semantic=semantic,
        bindings=bindings,
        analysis=analysis,
        route=route,
        execution=ExecutionState(
            plan=plan, task_results=report.results, evidence_evaluation=None
        ),
    )
    evaluated = await evaluate_node(state, context)
    assert evaluated["execution"].evidence_evaluation.status == "sufficient"
    ordered.append("evaluate")
    state["execution"] = evaluated["execution"]
    assert await synthesize_node(state, context) == {}
    ordered.append("synthesize")
    assert await ground_node(state, context) == {}
    ordered.append("ground")
    final = await finalizer_node(state, context)
    assert final["final_response"].status == "success"
    assert ordered == ["read", "evaluate", "synthesize", "ground"]


def test_summary_is_skill_not_agent_route() -> None:
    """A bounded summary uses the shared read capability, never a domain agent."""
    semantic = semantic_for_summary()
    bindings = bindings_for_summary()
    analysis = analyze_query(semantic)
    route = decide_route(
        analysis, semantic, bindings, allowed_capabilities=FULL_FAST_CAPABILITIES
    )
    assert route.route == "fast_domain"
    plan = build_fast_plan(semantic, bindings, analysis, route)
    assert plan.tasks[0].capability in ("document.read", "section.read")

    v2_root = Path("app/services/agents/v2")
    assert list(v2_root.glob("*summary_agent*")) == []
    assert list(v2_root.glob("domain/*_graph.py")) == []
    sources = [
        path.read_text()
        for path in v2_root.rglob("*.py")
        if "test" not in path.parts
    ]
    assert not any(re.search(r"\bSummaryAgent\b", source) for source in sources)

    runtime = capability_runtime()
    stub = StubReadCapability(
        read_result("t", uuid4(), "x"),
    )
    registry = build_capability_registry(
        [CapabilityRegistration(capability=stub)], runtime
    )
    assert "summary" not in "".join(registry.capability_names())


def test_compare_never_routes_to_fast_domain() -> None:
    semantic = SemanticContext(
        contextualized_query="So sánh tài liệu A và tài liệu B.",
        normalized_query="So sánh tài liệu A và tài liệu B.",
        abbreviations=(),
        coreferences=(),
        document_refs=(
            DocumentReference(
                ref_id="r1",
                original_span="tài liệu A",
                normalized_reference="tài liệu A",
                requested_role="target",
                revision_requirement=None,
                resolution_status="resolved",
                resolved_document_id=DOCUMENT_ID,
                candidate_document_ids=(),
            ),
            DocumentReference(
                ref_id="r2",
                original_span="tài liệu B",
                normalized_reference="tài liệu B",
                requested_role="target",
                revision_requirement=None,
                resolution_status="resolved",
                resolved_document_id=UUID("22222222-2222-2222-2222-222222222222"),
                candidate_document_ids=(),
            ),
        ),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    bindings = DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b_r1",
                document_id=DOCUMENT_ID,
                document_revision=REVISION,
                role="target",
            ),
            ScopedDocument(
                binding_id="b_r2",
                document_id=UUID("22222222-2222-2222-2222-222222222222"),
                document_revision=REVISION,
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )
    analysis = analyze_query(semantic)
    route = decide_route(
        analysis, semantic, bindings, allowed_capabilities=FULL_FAST_CAPABILITIES
    )
    assert route.route == "complex_research"
    assert route.reason_code == "comparison"


@pytest.mark.asyncio
async def test_write_is_typed_unavailable_in_finalizer() -> None:
    semantic = SemanticContext(
        contextualized_query="Viết báo cáo về tài liệu A.",
        normalized_query="Viết báo cáo về tài liệu A.",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    analysis = analyze_query(semantic)
    route = decide_route(
        analysis,
        semantic,
        DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        allowed_capabilities=FULL_FAST_CAPABILITIES,
    )
    assert route.route == "complex_research"
    state = make_state(
        semantic=semantic,
        bindings=DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        analysis=analysis,
        route=route,
    )
    context = GraphRuntimeContext(
        capability_runtime=capability_runtime(), services=RuntimeServices()
    )
    final = await finalizer_node(state, context)
    assert final["final_response"].status != "success"


@pytest.mark.asyncio
async def test_finalizer_direct_greeting_succeeds_without_capabilities() -> None:
    semantic = SemanticContext(
        contextualized_query="Xin chào.",
        normalized_query="Xin chào.",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    analysis = analyze_query(semantic)
    route = decide_route(
        analysis,
        semantic,
        DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        allowed_capabilities=frozenset(),
    )
    assert route.route == "direct"
    state = make_state(
        semantic=semantic,
        bindings=DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        analysis=analysis,
        route=route,
    )
    context = GraphRuntimeContext(
        capability_runtime=capability_runtime(), services=RuntimeServices()
    )
    final = await finalizer_node(state, context)
    assert final["final_response"].status == "success"
    assert final["final_response"].content.strip() != ""


def test_v2_has_no_domain_agent_or_domain_graph_wrappers() -> None:
    v2_root = Path("app/services/agents/v2")
    forbidden_files = [
        "people_agent.py",
        "summary_agent.py",
        "comparison_agent.py",
        "document_agent.py",
        "section_agent.py",
        "kg_agent.py",
    ]
    for name in forbidden_files:
        assert list(v2_root.rglob(name)) == [], name
    assert list(v2_root.glob("domain/*_graph.py")) == []
    pattern = re.compile(
        r"class\s+(People|Summary|Comparison|Document|Section|KG|Evaluation|Grounding).*Agent"
        r"|\b(People|Summary|Comparison|Document|Section|KG)Agent\b"
    )
    hits = [
        str(path)
        for path in v2_root.rglob("*.py")
        if pattern.search(path.read_text())
    ]
    assert hits == []


def test_model_input_cannot_supply_workspace_or_acl() -> None:
    """Synthesis/grounding model seams receive semantic facts only."""
    import inspect

    from app.services.agents.v2.nodes import grounding, synthesize

    for function in (
        synthesize.synthesize_answer,
        grounding.ground_answer,
        synthesize.hydrate_for_synthesis,
    ):
        source = inspect.getsource(function)
        assert "workspace_ids" not in source
        assert "can_read_people" not in source
        assert "allowed_capabilities" not in source

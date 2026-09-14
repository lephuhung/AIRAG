"""Task 10 — governed Adaptive Planner (Phase 5 backend).

Proves the planner boundary: ``ResearchPlanningInput`` stays the authoritative
runtime envelope, only a minimized/redacted projection reaches the model, the
planner returns a proposal (never dispatches tools), deterministic skill
policies win when available, and every model proposal is validated
(scope/capability/budget/DAG) before lease/checkpoint/scheduler. A model
failure or an ungovernable proposal yields the typed unavailable boundary
with zero dispatch — never a fabricated plan.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    DocumentReadInput,
    DocumentReadOutput,
)
from app.services.agents.v2.contracts.evaluation import CoverageObservation
from app.services.agents.v2.contracts.evidence import (
    DocumentSourceIdentity,
    EvidenceUseRef,
)
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.planning import (
    DiscoveryPolicy,
    ResearchBudgetView,
    ResearchPlanningInput,
)
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.state import (
    GraphRuntimeContext,
    RuntimeServices,
)
from app.services.agents.v2.planning import (
    AdaptivePlanner,
    PlannerError,
    build_planner_model_input,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")
REV_A = UUID("a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1")
REV_B = UUID("b2b2b2b2-b2b2-b2b2-b2b2-b2b2b2b2b2b2")
PLANNER_QUERY = "So sánh nghĩa vụ A và nghĩa vụ B"


class FakeChunk:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakePlannerProvider:
    """Model double: serves one canned JSON proposal, records calls."""

    def __init__(self, payload: object, *, raise_on_call: Exception | None = None) -> None:
        self._payload = payload
        self.raise_on_call = raise_on_call
        self.calls: list[tuple] = []

    async def astream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((messages, kwargs))
        if self.raise_on_call is not None:
            raise self.raise_on_call
        yield FakeChunk(json.dumps(self._payload))


class FakeReadCapability:
    """Atomic stub: one bounded read per planned target."""

    def __init__(self) -> None:
        self.descriptor = CapabilityDescriptor(
            name="document.read",  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=True,
        )
        self.calls: list[tuple[AgentRequest, object]] = []
        self.uses: dict[UUID, tuple[str, str]] = {}

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        assert isinstance(request.input, DocumentReadInput)
        target_ids = tuple(request.input.target_ids)
        assert len(target_ids) == 1
        use_id = uuid4()
        self.uses[use_id] = (request.task_id, target_ids[0])
        return AgentResult(
            contract_version="2.0",
            task_id=request.task_id,
            status="success",
            data=DocumentReadOutput(kind="document.read", read_unit_count=1),
            evidence_uses=(EvidenceUseRef(use_id=use_id),),
            coverage_observations=(
                CoverageObservation(
                    target_id=target_ids[0],
                    observed_locators=(DocumentLocator(kind="document"),),
                    outcome="read",
                ),
            ),
            error=None,
        )


class FakeLeases:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, object, object]] = []
        self.session = self._Session()

    class _Session:
        def __init__(self) -> None:
            self.commits = 0

        async def commit(self) -> None:
            self.commits += 1

    async def acquire_or_refresh(
        self, run_id: str, revision_id: object = None, evidence_use_id: object = None
    ) -> object:
        self.acquired.append((run_id, revision_id, evidence_use_id))
        return {"run_id": run_id, "revision": revision_id, "use": evidence_use_id}


class FakeHydrator:
    def __init__(self, capability: FakeReadCapability) -> None:
        self._capability = capability

    async def hydrate_for_evaluation(
        self, use_refs, *, runtime, plan, bindings
    ):  # type: ignore[no-untyped-def]
        from app.services.agents.v2.nodes.evaluate import HydratedEvidence

        binding_by_target = {
            unit.target_id: next(
                binding
                for binding in bindings.bindings
                if binding.binding_id == unit.binding_id
            )
            for unit in plan.target_units
        }
        admitted = []
        for ref in use_refs:
            owner = self._capability.uses.get(ref.use_id)
            if owner is None:
                continue
            task_id, target_id = owner
            binding = binding_by_target[target_id]
            locator = next(
                unit.requested_locator
                for unit in plan.target_units
                if unit.target_id == target_id
            )
            admitted.append(
                HydratedEvidence(
                    use_id=ref.use_id,
                    evidence_id=uuid4(),
                    task_id=task_id,
                    purpose="coverage",
                    target_id=target_id,
                    content=f"content of {target_id}",
                    role=binding.role,
                    source_label=target_id,
                    source_identity=DocumentSourceIdentity(
                        kind="document",
                        document_id=binding.document_id,
                        document_revision=binding.document_revision,
                        locator=locator,
                    ),
                    classification="normal",
                    locator=locator,
                    document_revision=binding.document_revision,
                )
            )
        return tuple(admitted)

    async def hydrate_for_synthesis(
        self, use_refs, *, runtime, plan, bindings, budget
    ):  # type: ignore[no-untyped-def]
        return await self.hydrate_for_evaluation(
            use_refs, runtime=runtime, plan=plan, bindings=bindings
        )


def _semantic(query: str = PLANNER_QUERY) -> SemanticContext:
    return SemanticContext(
        contextualized_query=query,
        normalized_query=query.lower(),
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def _bindings() -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b1",
                document_id=DOC_A,
                document_revision=str(REV_A),
                role="target",
            ),
            ScopedDocument(
                binding_id="b2",
                document_id=DOC_B,
                document_revision=str(REV_B),
                role="reference",
            ),
        ),
        revision_requirement_refs=(),
    )


def _analysis(work_type: str = "multi_goal") -> QueryAnalysis:
    return QueryAnalysis(
        work_type=work_type,  # type: ignore[arg-type]
        domains=("document",),  # type: ignore[arg-type]
    )


def _planning_input(
    work_type: str = "multi_goal",
    capability_names: frozenset[str] = frozenset({"document.read", "section.read"}),
    *,
    discovery: bool = False,
    max_tasks: int = 8,
) -> ResearchPlanningInput:
    return ResearchPlanningInput(
        semantic=_semantic(),
        bindings=_bindings(),
        query_analysis=_analysis(work_type),
        capability_catalog=tuple(
            CapabilityDescriptor(
                name=name,  # type: ignore[arg-type]
                domain="document",  # type: ignore[arg-type]
                operation_type="read",
                supports_parallel=True,
            )
            for name in sorted(capability_names)
        ),
        discovery_policy=DiscoveryPolicy(
            allow_reference_discovery=discovery,
            allow_supporting_discovery=discovery,
            max_discovered_documents=4 if discovery else 0,
        ),
        budget=ResearchBudgetView(
            max_tasks_remaining=max_tasks,
            max_replans_remaining=0,
            max_parallel_branches=2,
        ),
    )


def _capability_runtime(
    run_id: str,
    allowed: frozenset[str],
) -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id=f"req-{run_id}",
        run_id=run_id,
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=allowed,
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


def _harness(
    run_id: str,
    *,
    planner: AdaptivePlanner | None,
    allowed: frozenset[str] | None = None,
) -> tuple[FakeReadCapability, FakeLeases, GraphRuntimeContext]:
    capability = FakeReadCapability()
    leases = FakeLeases()
    granted = allowed if allowed is not None else frozenset({"document.read"})
    runtime = _capability_runtime(run_id, granted)
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], runtime
    )
    from app.services.agent.runtime_selector import PlanBindingResolver

    context = GraphRuntimeContext(
        capability_runtime=runtime,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=leases,
            evidence_hydrator=FakeHydrator(capability),
            pinned_target_resolver=PlanBindingResolver(),
            adaptive_planner=planner,
        ),
    )
    return capability, leases, context


def _child_input(work_type: str = "multi_goal") -> dict:
    from app.services.agents.v2.complex_research_graph import ComplexResearchState

    return ComplexResearchState(
        contract_version="2.0",
        semantic=_semantic(),
        bindings=_bindings(),
        query_analysis=_analysis(work_type),
        route_decision=RouteDecision(
            route="complex_research", reason_code="multi_goal"
        ),
        plan=None,
        task_results=(),
        evaluation=None,
        replans_remaining=0,
    )


def _runtime_for(context: GraphRuntimeContext):  # type: ignore[no-untyped-def]
    from langgraph.runtime import Runtime

    return Runtime(context=context)


def _two_read_proposal() -> dict:
    return {
        "tasks": [
            {
                "capability": "document.read",
                "task_objective": "Read nghia vu A",
                "targets": ["b1"],
                "depends_on": [],
            },
            {
                "capability": "document.read",
                "task_objective": "Read nghia vu B",
                "targets": ["b2"],
                "depends_on": [0],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Projection: minimized/redacted model input
# ---------------------------------------------------------------------------


def test_planner_model_input_minimizes_and_redacts_internals() -> None:
    planning_input = _planning_input()
    model_input = build_planner_model_input(planning_input)
    payload = model_input.model_dump_json()
    # No trusted identity or internal identifiers reach the model.
    assert str(DOC_A) not in payload
    assert str(DOC_B) not in payload
    assert str(REV_A) not in payload
    assert str(REV_B) not in payload
    assert "person_identifier" not in payload
    assert "use_id" not in payload
    assert "evidence" not in payload.lower()
    # The planner still sees what it needs: query, work type, governed refs.
    assert model_input.query == PLANNER_QUERY
    assert model_input.work_type == "multi_goal"
    binding_ids = {binding.binding_id for binding in model_input.bindings}
    assert binding_ids == {"b1", "b2"}
    catalog_names = {entry.name for entry in model_input.capability_catalog}
    assert catalog_names == {"document.read", "section.read"}
    assert model_input.budget.max_tasks_remaining == 8


# ---------------------------------------------------------------------------
# Deterministic policies win when available (fallback order)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_skill_preferred_over_model() -> None:
    provider = FakePlannerProvider(_two_read_proposal())
    planner = AdaptivePlanner(provider_factory=lambda: provider)
    proposal = await planner.propose_initial(
        _planning_input(work_type="compare"),
        _harness("run-planner-det", planner=planner)[2],
    )
    plan = proposal.plan
    assert [task.capability for task in plan.tasks] == ["document.read", "document.read"]
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Model proposal -> governed server-side construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_proposal_builds_validated_plan_without_dispatch() -> None:
    from app.services.agents.v2.contracts.validation import validate_task_plan

    provider = FakePlannerProvider(_two_read_proposal())
    planner = AdaptivePlanner(provider_factory=lambda: provider)
    capability, _, context = _harness("run-planner-model", planner=planner)
    proposal = await planner.propose_initial(_planning_input(), context)
    plan = proposal.plan
    assert proposal.reduce_spec is None
    assert plan.plan_id == "adaptive-multi_goal"
    assert plan.goal == PLANNER_QUERY
    assert [task.task_id for task in plan.tasks] == ["T1", "T2"]
    assert plan.tasks[1].depends_on == ("T1",)
    assert {unit.binding_id for unit in plan.target_units} == {"b1", "b2"}
    assert {task.input.target_ids[0] for task in plan.tasks} == {"t1", "t2"}  # type: ignore[union-attr]
    validate_task_plan(plan, _bindings())
    assert provider.calls != []
    # Proposal-only: no capability was dispatched.
    assert capability.calls == []


@pytest.mark.asyncio
async def test_model_proposal_rejects_minted_identity() -> None:
    provider = FakePlannerProvider(
        {"tasks": [{"capability": "document.read", "task_objective": "Read X",
                     "targets": ["b_unknown"], "depends_on": []}]}
    )
    planner = AdaptivePlanner(provider_factory=lambda: provider)
    _, _, context = _harness("run-planner-identity", planner=planner)
    with pytest.raises((PlannerError, ValueError)):
        await planner.propose_initial(_planning_input(), context)


@pytest.mark.asyncio
async def test_model_proposal_rejects_scope_widening() -> None:
    # Capability outside the current catalog/allowed set.
    provider = FakePlannerProvider(
        {"tasks": [{"capability": "people.lookup", "task_objective": "Find A",
                     "targets": [], "depends_on": []}]}
    )
    planner = AdaptivePlanner(provider_factory=lambda: provider)
    _, _, context = _harness("run-planner-scope", planner=planner)
    with pytest.raises((PlannerError, ValueError)):
        await planner.propose_initial(_planning_input(), context)


@pytest.mark.asyncio
async def test_model_proposal_rejects_discovery_when_policy_closed() -> None:
    provider = FakePlannerProvider(
        {"tasks": [{"capability": "document.search", "task_objective": "Discover",
                     "targets": [], "depends_on": []}]}
    )
    planner = AdaptivePlanner(provider_factory=lambda: provider)
    _, _, context = _harness(
        "run-planner-discovery",
        planner=planner,
        allowed=frozenset({"document.read", "document.search"}),
    )
    with pytest.raises((PlannerError, ValueError)):
        await planner.propose_initial(
            _planning_input(
                capability_names=frozenset({"document.read", "document.search"}),
            ),
            context,
        )


@pytest.mark.asyncio
async def test_model_proposal_rejects_cycles_and_over_budget() -> None:
    provider = FakePlannerProvider(
        {"tasks": [
            {"capability": "document.read", "task_objective": "A",
             "targets": ["b1"], "depends_on": [1]},
            {"capability": "document.read", "task_objective": "B",
             "targets": ["b2"], "depends_on": [0]},
        ]}
    )
    planner = AdaptivePlanner(provider_factory=lambda: provider)
    _, _, context = _harness("run-planner-cycle", planner=planner)
    with pytest.raises((PlannerError, ValueError)):
        await planner.propose_initial(_planning_input(), context)

    over_budget = FakePlannerProvider(_two_read_proposal())
    over_planner = AdaptivePlanner(provider_factory=lambda: over_budget)
    _, _, over_context = _harness("run-planner-budget", planner=over_planner)
    with pytest.raises((PlannerError, ValueError)):
        await over_planner.propose_initial(
            _planning_input(max_tasks=1), over_context
        )


# ---------------------------------------------------------------------------
# Failure path: typed unavailable, zero dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_failure_falls_back_to_typed_unavailable() -> None:
    from app.services.agents.v2.complex_research_graph import (
        COMPLEX_RESEARCH_UNAVAILABLE,
        decide_node,
        validate_checkpoint_node,
    )

    provider = FakePlannerProvider(
        _two_read_proposal(), raise_on_call=RuntimeError("planner down")
    )
    planner = AdaptivePlanner(provider_factory=lambda: provider)
    capability, _, context = _harness("run-planner-down", planner=planner)
    output = await validate_checkpoint_node(
        _child_input(), _runtime_for(context)
    )
    assert output.get("plan") is None
    decided = await decide_node(_child_input(), _runtime_for(context))
    assert decided["unavailable"].code == COMPLEX_RESEARCH_UNAVAILABLE
    assert capability.calls == []


@pytest.mark.asyncio
async def test_unwired_planner_keeps_legacy_unavailable() -> None:
    from app.services.agents.v2.complex_research_graph import (
        COMPLEX_RESEARCH_UNAVAILABLE,
        decide_node,
        validate_checkpoint_node,
    )

    capability, _, context = _harness("run-planner-unwired", planner=None)
    output = await validate_checkpoint_node(
        _child_input(), _runtime_for(context)
    )
    assert output.get("plan") is None
    decided = await decide_node(_child_input(), _runtime_for(context))
    assert decided["unavailable"].code == COMPLEX_RESEARCH_UNAVAILABLE
    assert capability.calls == []


# ---------------------------------------------------------------------------
# End to end: model plan checkpoints then dispatches through the scheduler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_model_planned_reads_dispatch_once() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    provider = FakePlannerProvider(_two_read_proposal())
    planner = AdaptivePlanner(provider_factory=lambda: provider)
    capability, _, context = _harness("run-planner-e2e", planner=planner)
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(), context=context
    )
    assert output["plan"] is not None
    assert output["plan"].plan_id == "adaptive-multi_goal"
    assert [task.task_id for task in output["plan"].tasks] == ["T1", "T2"]
    assert [call[0].task_id for call in capability.calls] == ["T1", "T2"]
    assert output["evaluation"].status == "sufficient"


def test_planner_service_is_runtime_only() -> None:
    from app.services.agents.v2.contracts.base import ContractModel

    assert RuntimeServices().adaptive_planner is None
    assert not (isinstance(AdaptivePlanner(), ContractModel))
    assert isinstance(RuntimeServices(adaptive_planner=AdaptivePlanner()).adaptive_planner, AdaptivePlanner)

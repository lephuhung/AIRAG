"""Task 12 — governed complex work-type expansion (Phase 5 backend).

Serial TDD slices in brief order: ``multi_goal`` (slice 1), generic
``cross_domain`` (slice 2), ``evaluate``/compliance (slice 3). Every slice
proves the same boundary: deterministic skill policies propose a bounded
parallel-read DAG over already-bound documents (never minting identity,
never dispatching), the governed entry still validates/leases/checkpoints
before the shared scheduler dispatches, and below-intake inputs stay on the
model path or the typed unavailable boundary. No ReAct loop is introduced.
"""
from __future__ import annotations

from uuid import UUID

import pytest
from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.planning import (
    DiscoveryPolicy,
    ResearchBudgetView,
    ResearchPlanningInput,
)
from app.services.agents.v2.contracts.routing import QueryAnalysis
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.capability import CapabilityDescriptor

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")
DOC_C = UUID("33333333-3333-3333-3333-333333333333")
REV_A = UUID("a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1")
REV_B = UUID("b2b2b2b2-b2b2-b2b2-b2b2-b2b2b2b2b2b2")
REV_C = UUID("c3c3c3c3-c3c3-c3c3-c3c3-c3c3c3c3c3c3")

EXPANSION_QUERY = "Tổng hợp nghĩa vụ A, B và C để đối chiếu"


def _semantic(query: str = EXPANSION_QUERY) -> SemanticContext:
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


def _three_bindings() -> DocumentBindingSet:
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
            ScopedDocument(
                binding_id="b3",
                document_id=DOC_C,
                document_revision=str(REV_C),
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )


def _two_bindings() -> DocumentBindingSet:
    bindings = _three_bindings()
    return DocumentBindingSet(
        bindings=tuple(b for b in bindings.bindings if b.binding_id != "b3"),
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
    bindings: DocumentBindingSet | None = None,
    max_tasks: int = 8,
) -> ResearchPlanningInput:
    return ResearchPlanningInput(
        semantic=_semantic(),
        bindings=bindings or _three_bindings(),
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
            allow_reference_discovery=False,
            allow_supporting_discovery=False,
            max_discovered_documents=0,
        ),
        budget=ResearchBudgetView(
            max_tasks_remaining=max_tasks,
            max_replans_remaining=0,
            max_parallel_branches=2,
        ),
    )


# ---------------------------------------------------------------------------
# Slice 1 — multi_goal deterministic skill
# ---------------------------------------------------------------------------


def test_multi_goal_skill_claims_only_multi_goal() -> None:
    from app.services.agents.v2.skills.multi_goal import policy as multi_goal_policy

    assert multi_goal_policy.supports_work_type("multi_goal") is True
    assert multi_goal_policy.supports_work_type("compare") is False
    assert multi_goal_policy.supports_work_type("evaluate") is False
    assert multi_goal_policy.supports_work_type("cross_domain") is False


def test_multi_goal_three_bound_documents_plan_parallel_reads() -> None:
    from app.services.agents.v2.contracts.validation import validate_task_plan
    from app.services.agents.v2.skills.multi_goal import policy as multi_goal_policy

    plan = multi_goal_policy.build_multi_goal_plan(_planning_input())
    assert [task.task_id for task in plan.tasks] == ["T1", "T2", "T3"]
    assert [task.capability for task in plan.tasks] == [
        "document.read",
        "document.read",
        "document.read",
    ]
    assert all(task.depends_on == () for task in plan.tasks)
    assert {unit.binding_id for unit in plan.target_units} == {"b1", "b2", "b3"}
    assert {task.input.target_ids[0] for task in plan.tasks} == {"t1", "t2", "t3"}  # type: ignore[union-attr]
    validate_task_plan(plan, _three_bindings())


def test_multi_goal_two_bindings_stays_below_intake() -> None:
    """Routing arity (compare=2, multi_goal=3+) is mirrored: a two-binding
    multi_goal input refuses deterministically so the governed model path
    (or the typed unavailable boundary when unwired) still owns it."""
    from app.services.agents.v2.contracts.validation import ContractValidationError
    from app.services.agents.v2.skills.multi_goal import policy as multi_goal_policy

    with pytest.raises(ContractValidationError, match="multi_goal requires"):
        multi_goal_policy.build_multi_goal_plan(
            _planning_input(bindings=_two_bindings())
        )


def test_multi_goal_wrong_work_type_refuses() -> None:
    from app.services.agents.v2.contracts.validation import ContractValidationError
    from app.services.agents.v2.skills.multi_goal import policy as multi_goal_policy

    with pytest.raises(ContractValidationError):
        multi_goal_policy.build_multi_goal_plan(_planning_input(work_type="compare"))


def test_multi_goal_missing_read_capability_refuses_with_zero_dispatch() -> None:
    from app.services.agents.v2.contracts.validation import ContractValidationError
    from app.services.agents.v2.skills.multi_goal import policy as multi_goal_policy

    with pytest.raises(ContractValidationError, match="undispatchable"):
        multi_goal_policy.build_multi_goal_plan(
            _planning_input(capability_names=frozenset({"section.read"}))
        )


def test_multi_goal_over_task_budget_refuses() -> None:
    from app.services.agents.v2.contracts.validation import ContractValidationError
    from app.services.agents.v2.skills.multi_goal import policy as multi_goal_policy

    with pytest.raises(ContractValidationError, match="budget"):
        multi_goal_policy.build_multi_goal_plan(
            _planning_input(max_tasks=2)
        )


def test_build_initial_proposal_selects_multi_goal_skill() -> None:
    from app.services.agents.v2.complex_research_graph import build_initial_proposal

    proposal = build_initial_proposal(_planning_input())
    assert proposal.plan.tasks[0].capability == "document.read"
    assert len(proposal.plan.tasks) == 3
    assert proposal.reduce_spec is None


# ---------------------------------------------------------------------------
# Slice 2 — generic cross_domain deterministic skill
# ---------------------------------------------------------------------------


def _person_semantic(query: str = EXPANSION_QUERY):  # type: ignore[no-untyped-def]
    from app.services.agents.v2.contracts.conversation import EntityReference

    return SemanticContext(
        contextualized_query=query,
        normalized_query=query.lower(),
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(
            EntityReference(ref_id="p1", kind="person", label="A"),
        ),
        section_refs=(),
        blocking_ambiguities=(),
    )


def _cross_domain_input(
    capability_names: frozenset[str] = frozenset({"document.read", "section.read"}),
    *,
    semantic: SemanticContext | None = None,
    bindings: DocumentBindingSet | None = None,
) -> ResearchPlanningInput:
    return ResearchPlanningInput(
        semantic=semantic or _semantic(),
        bindings=bindings or _two_bindings(),
        query_analysis=_analysis("cross_domain"),
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
            allow_reference_discovery=False,
            allow_supporting_discovery=False,
            max_discovered_documents=0,
        ),
        budget=ResearchBudgetView(
            max_tasks_remaining=8,
            max_replans_remaining=0,
            max_parallel_branches=2,
        ),
    )


def _empty_bindings() -> DocumentBindingSet:
    return DocumentBindingSet(bindings=(), revision_requirement_refs=())


def test_cross_domain_skill_claims_only_cross_domain() -> None:
    from app.services.agents.v2.skills.cross_domain import policy as xd_policy

    assert xd_policy.supports_work_type("cross_domain") is True
    assert xd_policy.supports_work_type("multi_goal") is False
    assert xd_policy.supports_work_type("evaluate") is False
    assert xd_policy.supports_work_type("compare") is False


def test_cross_domain_with_person_matches_people_first() -> None:
    """The named-person branch stays byte-identical to the pilot's governed
    first step: one targetless ``people.lookup``, no fabricated person."""
    from app.services.agents.v2.skills.cross_domain import policy as xd_policy

    plan = xd_policy.build_cross_domain_plan(
        _cross_domain_input(
            capability_names=frozenset({"people.lookup", "document.read"}),
            semantic=_person_semantic(),
            bindings=_empty_bindings(),
        )
    )
    assert plan.plan_id == "people-first-p1"
    assert [task.task_id for task in plan.tasks] == ["T1"]
    assert plan.tasks[0].capability == "people.lookup"
    assert plan.tasks[0].depends_on == ()


def test_cross_domain_without_person_plans_bound_reads() -> None:
    from app.services.agents.v2.contracts.validation import validate_task_plan
    from app.services.agents.v2.skills.cross_domain import policy as xd_policy

    plan = xd_policy.build_cross_domain_plan(_cross_domain_input())
    assert [task.task_id for task in plan.tasks] == ["T1", "T2"]
    assert [task.capability for task in plan.tasks] == [
        "document.read",
        "document.read",
    ]
    assert all(task.depends_on == () for task in plan.tasks)
    assert {unit.binding_id for unit in plan.target_units} == {"b1", "b2"}
    validate_task_plan(plan, _two_bindings())


def test_cross_domain_open_input_refuses() -> None:
    """No person and no bound documents: nothing governable to propose —
    the governed model path (or the typed unavailable boundary when unwired)
    still owns the input."""
    from app.services.agents.v2.contracts.validation import ContractValidationError
    from app.services.agents.v2.skills.cross_domain import policy as xd_policy

    with pytest.raises(ContractValidationError, match="cross_domain needs"):
        xd_policy.build_cross_domain_plan(
            _cross_domain_input(bindings=_empty_bindings())
        )


def test_cross_domain_wrong_work_type_refuses() -> None:
    from app.services.agents.v2.contracts.validation import ContractValidationError
    from app.services.agents.v2.skills.cross_domain import policy as xd_policy

    with pytest.raises(ContractValidationError):
        xd_policy.build_cross_domain_plan(_planning_input(work_type="compare"))


def test_cross_domain_missing_people_capability_refuses() -> None:
    from app.services.agents.v2.contracts.validation import ContractValidationError
    from app.services.agents.v2.skills.cross_domain import policy as xd_policy

    with pytest.raises(ContractValidationError, match="people.lookup"):
        xd_policy.build_cross_domain_plan(
            _cross_domain_input(semantic=_person_semantic())
        )


def test_build_initial_proposal_selects_cross_domain_skill() -> None:
    from app.services.agents.v2.complex_research_graph import build_initial_proposal

    proposal = build_initial_proposal(_cross_domain_input())
    assert [task.capability for task in proposal.plan.tasks] == [
        "document.read",
        "document.read",
    ]
    assert proposal.reduce_spec is None


# ---------------------------------------------------------------------------
# Slice 3 — evaluate / compliance deterministic skill + v2 serving
# ---------------------------------------------------------------------------


def _evaluate_input(
    capability_names: frozenset[str] = frozenset({"document.read", "section.read"}),
    *,
    bindings: DocumentBindingSet | None = None,
) -> ResearchPlanningInput:
    return ResearchPlanningInput(
        semantic=_semantic(),
        bindings=bindings or _two_bindings(),
        query_analysis=_analysis("evaluate"),
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
            allow_reference_discovery=False,
            allow_supporting_discovery=False,
            max_discovered_documents=0,
        ),
        budget=ResearchBudgetView(
            max_tasks_remaining=8,
            max_replans_remaining=0,
            max_parallel_branches=2,
        ),
    )


def _single_target_bindings() -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b1",
                document_id=DOC_A,
                document_revision=str(REV_A),
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )


def test_evaluate_skill_claims_only_evaluate() -> None:
    from app.services.agents.v2.skills.evaluate import policy as eval_policy

    assert eval_policy.supports_work_type("evaluate") is True
    assert eval_policy.supports_work_type("compare") is False
    assert eval_policy.supports_work_type("multi_goal") is False
    assert eval_policy.supports_work_type("cross_domain") is False


def test_evaluate_bound_documents_plan_parallel_reads() -> None:
    from app.services.agents.v2.contracts.validation import validate_task_plan
    from app.services.agents.v2.skills.evaluate import policy as eval_policy

    plan = eval_policy.build_evaluate_plan(_evaluate_input())
    assert [task.task_id for task in plan.tasks] == ["T1", "T2"]
    assert [task.capability for task in plan.tasks] == [
        "document.read",
        "document.read",
    ]
    assert all(task.depends_on == () for task in plan.tasks)
    assert {unit.binding_id for unit in plan.target_units} == {"b1", "b2"}
    validate_task_plan(plan, _two_bindings())


def test_evaluate_single_bound_document_plans() -> None:
    """Compliance review of even one bound document is governed v2 work."""
    from app.services.agents.v2.skills.evaluate import policy as eval_policy

    plan = eval_policy.build_evaluate_plan(
        _evaluate_input(bindings=_single_target_bindings())
    )
    assert [task.task_id for task in plan.tasks] == ["T1"]
    assert plan.tasks[0].capability == "document.read"


def test_evaluate_open_input_refuses() -> None:
    """No bound documents: nothing governable to assess — the governed model
    path (when wired with open discovery) or the typed unavailable boundary
    still owns the input."""
    from app.services.agents.v2.contracts.validation import ContractValidationError
    from app.services.agents.v2.skills.evaluate import policy as eval_policy

    with pytest.raises(ContractValidationError, match="evaluate needs"):
        eval_policy.build_evaluate_plan(
            _evaluate_input(bindings=_empty_bindings())
        )


def test_evaluate_wrong_work_type_refuses() -> None:
    from app.services.agents.v2.contracts.validation import ContractValidationError
    from app.services.agents.v2.skills.evaluate import policy as eval_policy

    with pytest.raises(ContractValidationError):
        eval_policy.build_evaluate_plan(_planning_input(work_type="compare"))


def test_evaluate_missing_read_capability_refuses() -> None:
    from app.services.agents.v2.contracts.validation import ContractValidationError
    from app.services.agents.v2.skills.evaluate import policy as eval_policy

    with pytest.raises(ContractValidationError, match="undispatchable"):
        eval_policy.build_evaluate_plan(
            _evaluate_input(capability_names=frozenset({"section.read"}))
        )


def test_build_initial_proposal_selects_evaluate_skill() -> None:
    from app.services.agents.v2.complex_research_graph import build_initial_proposal

    proposal = build_initial_proposal(_evaluate_input())
    assert [task.capability for task in proposal.plan.tasks] == [
        "document.read",
        "document.read",
    ]
    assert proposal.reduce_spec is None


def test_governed_evaluate_does_not_require_v1_fallback() -> None:
    """Task 12: ``evaluate``/``compliance_evaluation`` is served by the v2
    governed complex DAG (bounded reads + evidence evaluation + grounded
    synthesis), so the post-router fallback guard must not fire for it.
    Write-domain and unknown outcomes still fall back."""
    from app.services.agent.rollout_control import requires_v1_fallback
    from app.services.agents.v2.contracts.routing import (
        QueryAnalysis,
        RouteDecision,
    )

    analysis = QueryAnalysis(work_type="evaluate", domains=("document",))
    decision = RouteDecision(
        route="complex_research", reason_code="compliance_evaluation"
    )
    assert requires_v1_fallback(analysis, decision) is False


# ---------------------------------------------------------------------------
# End to end: deterministic expansion plans dispatch through the scheduler
# ---------------------------------------------------------------------------


class _FakeReadCapability:
    """Atomic stub: one bounded read per planned target."""

    def __init__(self) -> None:
        from app.services.agents.v2.contracts.capability import (
            CapabilityDescriptor,
            DocumentReadInput,
        )

        self._input_cls = DocumentReadInput
        self.descriptor = CapabilityDescriptor(
            name="document.read",  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=True,
        )
        self.calls: list = []
        self.uses: dict = {}

    async def execute(self, request, runtime):  # type: ignore[no-untyped-def]
        from uuid import uuid4

        from app.services.agents.v2.contracts.capability import DocumentReadOutput
        from app.services.agents.v2.contracts.evaluation import CoverageObservation
        from app.services.agents.v2.contracts.evidence import EvidenceUseRef
        from app.services.agents.v2.contracts.execution import AgentResult
        from app.services.agents.v2.contracts.locators import DocumentLocator

        self.calls.append((request, runtime))
        assert isinstance(request.input, self._input_cls)
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


class _FakeLeases:
    def __init__(self) -> None:
        self.acquired: list = []
        self.session = self._Session()

    class _Session:
        def __init__(self) -> None:
            self.commits = 0

        async def commit(self) -> None:
            self.commits += 1

    async def acquire_or_refresh(
        self, run_id: str, revision_id=None, evidence_use_id=None  # type: ignore[no-untyped-def]
    ) -> object:
        self.acquired.append((run_id, revision_id, evidence_use_id))
        return {"run_id": run_id, "revision": revision_id, "use": evidence_use_id}


class _FakeHydrator:
    def __init__(self, capability: _FakeReadCapability) -> None:
        self._capability = capability

    async def hydrate_for_evaluation(
        self, use_refs, **kwargs  # type: ignore[no-untyped-def]
    ):  # type: ignore[no-untyped-def]
        from uuid import uuid4

        from app.services.agents.v2.contracts.evidence import DocumentSourceIdentity
        from app.services.agents.v2.nodes.evaluate import HydratedEvidence

        plan = kwargs["plan"]
        bindings = kwargs["bindings"]
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
        self, use_refs, **kwargs  # type: ignore[no-untyped-def]
    ):  # type: ignore[no-untyped-def]
        return await self.hydrate_for_evaluation(use_refs, **kwargs)


def _e2e_harness(run_id: str):  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from app.services.agent.runtime_selector import PlanBindingResolver
    from app.services.agents.v2.capabilities import (
        CapabilityRegistration,
        build_capability_registry,
    )
    from app.services.agents.v2.contracts.capability import CapabilityRuntimeContext
    from app.services.agents.v2.contracts.state import (
        GraphRuntimeContext,
        RuntimeServices,
    )

    capability = _FakeReadCapability()
    leases = _FakeLeases()
    runtime = CapabilityRuntimeContext(
        request_id=f"req-{run_id}",
        run_id=run_id,
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=frozenset({"document.read"}),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], runtime
    )
    context = GraphRuntimeContext(
        capability_runtime=runtime,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=leases,
            evidence_hydrator=_FakeHydrator(capability),
            pinned_target_resolver=PlanBindingResolver(),
            adaptive_planner=None,
        ),
    )
    return capability, context


def _e2e_child(work_type: str, reason: str, bindings: DocumentBindingSet):  # type: ignore[no-untyped-def]
    from app.services.agents.v2.complex_research_graph import ComplexResearchState
    from app.services.agents.v2.contracts.routing import RouteDecision

    return ComplexResearchState(
        contract_version="2.0",
        semantic=_semantic(),
        bindings=bindings,
        query_analysis=_analysis(work_type),
        route_decision=RouteDecision(
            route="complex_research", reason_code=reason  # type: ignore[arg-type]
        ),
        plan=None,
        task_results=(),
        evaluation=None,
        replans_remaining=0,
    )


@pytest.mark.asyncio
async def test_end_to_end_multi_goal_reads_dispatch_once() -> None:
    """Three bound documents: deterministic plan checkpoints, all three reads
    dispatch exactly once through the shared scheduler, no model involved."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    capability, context = _e2e_harness("run-expand-mg-e2e")
    output = await build_complex_research_subgraph().ainvoke(
        _e2e_child("multi_goal", "multi_goal", _three_bindings()),
        context=context,
    )
    assert output["plan"] is not None
    assert output["plan"].plan_id == "multi-goal-b1-b2-b3"
    assert [call[0].task_id for call in capability.calls] == ["T1", "T2", "T3"]
    assert output["evaluation"].status == "sufficient"


@pytest.mark.asyncio
async def test_end_to_end_evaluate_reads_dispatch_without_v1_fallback() -> None:
    """Compliance work: the governed plan executes in v2 — the scheduler's
    pre-dispatch fallback guard (which fired for every ``evaluate`` route
    before Task 12) lets it through, both reads dispatch, and the
    evaluation verdict is sufficient."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    capability, context = _e2e_harness("run-expand-ev-e2e")
    output = await build_complex_research_subgraph().ainvoke(
        _e2e_child("evaluate", "compliance_evaluation", _two_bindings()),
        context=context,
    )
    assert output["plan"] is not None
    assert output["plan"].plan_id == "evaluate-b1-b2"
    assert [call[0].task_id for call in capability.calls] == ["T1", "T2"]
    assert output["evaluation"].status == "sufficient"

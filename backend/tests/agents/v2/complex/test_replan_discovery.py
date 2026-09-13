"""Bounded append-only replan + discovery + research loop (Phase 3, Task 5).

Proves the canonical loop ``plan -> validate_checkpoint -> execute ->
evaluate -> decide -> (replan -> validate_checkpoint)...`` with every replan
checkpointed before the next execute, completed tasks never rerun, budgets
enforced, discovery UUID/pin/promotion semantics, the framework-neutral
summarize skill (no summary agent), and the subagent advisory boundary.
"""
from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.contracts.binding import (
    BindingPromotionRequest,
    DocumentBindingSet,
    DocumentDiscoveryCandidate,
    ScopedDocument,
)
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    DocumentReadInput,
    DocumentReadOutput,
    DocumentSearchInput,
    DocumentSearchOutput,
    KnowledgeGraphInput,
    PeopleLookupInput,
    PeopleLookupOutput,
    SectionReadInput,
)
from app.services.agents.v2.contracts.evaluation import (
    Contradiction,
    Coverage,
    CoverageObservation,
    EvidenceEvaluation,
)
from app.services.agents.v2.contracts.evidence import DocumentSourceIdentity, EvidenceUseRef
from app.services.agents.v2.contracts.execution import (
    AgentRequest,
    AgentResult,
    TaskExecutionSummary,
)
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
    DiscoveryPolicy,
    InitialTaskOrigin,
    ReplanTaskOrigin,
    ResearchBudgetView,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.routing import QueryAnalysis
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.state import (
    GraphRuntimeContext,
    RuntimeServices,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_replan,
)
from app.services.agents.v2.dependencies.people_document import (
    PeopleDocumentMaterialization,
)
from app.services.agents.v2.tools.gateway import (
    AgentToolGateway,
    CapabilityInvocationProposal,
    UnplannedCapabilityDispatch,
    require_planned_dispatch,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")
REV_A = UUID("a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1")
REV_B = UUID("b2b2b2b2-b2b2-b2b2-b2b2-b2b2b2b2b2b2")
COMPARE_QUERY = "So sánh Chương II A và Chương III B"


class FakeReadCapability:
    """Atomic stub: one bounded read per planned target."""

    def __init__(
        self,
        name: str = "document.read",
        *,
        missing_targets: frozenset[str] = frozenset(),
        fail_first_for: frozenset[str] = frozenset(),
        truncate_first_for: frozenset[str] = frozenset(),
    ) -> None:
        domain = "section" if name == "section.read" else "document"
        self.descriptor = CapabilityDescriptor(
            name=name,  # type: ignore[arg-type]
            domain=domain,  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=True,
        )
        self.calls: list[tuple[AgentRequest, object]] = []
        self.missing_targets = missing_targets
        self.fail_first_for = fail_first_for
        self.truncate_first_for = truncate_first_for
        self._failed_once: set[str] = set()
        self._truncated_once: set[str] = set()
        self.uses: dict[UUID, tuple[str, str]] = {}

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        target_id = request.input.target_ids[0]  # type: ignore[union-attr]
        if target_id in self.fail_first_for and target_id not in self._failed_once:
            self._failed_once.add(target_id)
            return AgentResult(
                contract_version="2.0",
                task_id=request.task_id,
                status="success",
                data=DocumentReadOutput(kind="document.read", read_unit_count=0),
                evidence_uses=(),
                coverage_observations=(
                    CoverageObservation(
                        target_id=target_id,
                        observed_locators=(),
                        outcome="missing",
                    ),
                ),
                error=None,
            )
        if (
            target_id in self.truncate_first_for
            and target_id not in self._truncated_once
        ):
            self._truncated_once.add(target_id)
            use_id = uuid4()
            self.uses[use_id] = (request.task_id, target_id)
            return AgentResult(
                contract_version="2.0",
                task_id=request.task_id,
                status="success",
                data=DocumentReadOutput(kind="document.read", read_unit_count=1),
                evidence_uses=(EvidenceUseRef(use_id=use_id),),
                coverage_observations=(
                    CoverageObservation(
                        target_id=target_id,
                        observed_locators=(),
                        outcome="truncated",
                    ),
                ),
                error=None,
            )
        if target_id in self.missing_targets:
            return AgentResult(
                contract_version="2.0",
                task_id=request.task_id,
                status="success",
                data=DocumentReadOutput(kind="document.read", read_unit_count=0),
                evidence_uses=(),
                coverage_observations=(
                    CoverageObservation(
                        target_id=target_id,
                        observed_locators=(),
                        outcome="missing",
                    ),
                ),
                error=None,
            )
        use_id = uuid4()
        self.uses[use_id] = (request.task_id, target_id)
        return AgentResult(
            contract_version="2.0",
            task_id=request.task_id,
            status="success",
            data=DocumentReadOutput(kind="document.read", read_unit_count=1),
            evidence_uses=(EvidenceUseRef(use_id=use_id),),
            coverage_observations=(
                CoverageObservation(
                    target_id=target_id,
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
    def __init__(self, capabilities: tuple[FakeReadCapability, ...]) -> None:
        self._capabilities = capabilities

    def _use_owners(self) -> dict[UUID, tuple[str, str]]:
        owners: dict[UUID, tuple[str, str]] = {}
        for capability in self._capabilities:
            owners.update(capability.uses)
        return owners

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
        locator_by_target = {
            unit.target_id: unit.requested_locator for unit in plan.target_units
        }
        use_owners = self._use_owners()
        admitted = []
        for ref in use_refs:
            owner = use_owners.get(ref.use_id)
            if owner is None:
                continue
            task_id, target_id = owner
            binding = binding_by_target[target_id]
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
                        locator=locator_by_target[target_id],
                    ),
                    classification="normal",
                    locator=locator_by_target[target_id],
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


def _semantic() -> SemanticContext:
    return SemanticContext(
        contextualized_query=COMPARE_QUERY,
        normalized_query=COMPARE_QUERY.lower(),
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


def _analysis(work_type: str = "compare") -> QueryAnalysis:
    return QueryAnalysis(work_type=work_type, domains=("document",))  # type: ignore[arg-type]


def _read_task(
    task_id: str = "T1",
    target: str = "t1",
    capability: str = "document.read",
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability=capability,
        task_objective=f"Read {target}",
        input=DocumentReadInput(kind="document.read", target_ids=(target,)),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def _two_target_plan() -> TaskPlan:
    return TaskPlan(
        contract_version="2.0",
        plan_id="p1",
        goal=COMPARE_QUERY,
        target_units=(
            TargetUnit(
                target_id="t1",
                binding_id="b1",
                requested_locator=DocumentLocator(kind="document"),
                completion_criteria=(CoverageCriterion(kind="coverage"),),
            ),
            TargetUnit(
                target_id="t2",
                binding_id="b2",
                requested_locator=DocumentLocator(kind="document"),
                completion_criteria=(CoverageCriterion(kind="coverage"),),
            ),
        ),
        tasks=(_read_task("T1", "t1"), _read_task("T2", "t2")),
    )


def _harness(
    run_id: str = "run-replan-1",
    missing: frozenset[str] = frozenset(),
    allowed: frozenset[str] | None = None,
    fail_first_for: frozenset[str] = frozenset(),
    truncate_first_for: frozenset[str] = frozenset(),
) -> tuple[tuple[FakeReadCapability, ...], FakeLeases, GraphRuntimeContext]:
    capabilities = (
        FakeReadCapability(
            "document.read",
            missing_targets=missing,
            fail_first_for=fail_first_for,
            truncate_first_for=truncate_first_for,
        ),
        FakeReadCapability(
            "section.read",
            missing_targets=missing,
            fail_first_for=fail_first_for,
            truncate_first_for=truncate_first_for,
        ),
    )
    leases = FakeLeases()
    runtime = CapabilityRuntimeContext(
        request_id="req-replan-1",
        run_id=run_id,
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=(
            allowed
            if allowed is not None
            else frozenset({"document.read", "section.read"})
        ),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability) for capability in capabilities],
        runtime,
    )
    return (
        capabilities,
        leases,
        GraphRuntimeContext(
            capability_runtime=runtime,
            services=RuntimeServices(
                capability_registry=registry,
                retention_leases=leases,
                evidence_hydrator=FakeHydrator(capabilities),
            ),
        ),
    )


def _policy(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "allow_reference_discovery": False,
        "allow_supporting_discovery": False,
        "max_discovered_documents": 0,
    }
    values.update(overrides)
    return DiscoveryPolicy(**values)  # type: ignore[arg-type]


def _budget(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "max_tasks_remaining": 8,
        "max_replans_remaining": 2,
        "max_parallel_branches": 2,
    }
    values.update(overrides)
    return ResearchBudgetView(**values)  # type: ignore[arg-type]


@pytest.fixture
def replan_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the bounded replan loop: deployment settings allow 2 replans."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(
            V2_MAX_TASKS=8, V2_MAX_PARALLEL_BRANCHES=2, V2_MAX_REPLANS=2
        ),
    )


def _child_input(**overrides):  # type: ignore[no-untyped-def]
    from app.services.agents.v2.complex_research_graph import ComplexResearchState

    values: dict = {
        "contract_version": "2.0",
        "semantic": _semantic(),
        "bindings": _bindings(),
        "query_analysis": _analysis(),
        "plan": None,
        "task_results": (),
        "evaluation": None,
        "replans_remaining": 0,
    }
    values.update(overrides)
    return ComplexResearchState(**values)  # type: ignore[arg-type]


def _appended(
    current: TaskPlan,
    task_id: str = "T3",
    capability: str = "document.read",
    target: str = "t2",
    reason: str = "coverage gap for target t2",
) -> TaskPlan:
    return current.model_copy(
        update={
            "tasks": current.tasks
            + (
                TaskSpec(
                    task_id=task_id,
                    capability=capability,
                    task_objective=f"Re-read {target}",
                    input=DocumentReadInput(
                        kind="document.read", target_ids=(target,)
                    ),
                    depends_on=(),
                    origin=ReplanTaskOrigin(
                        kind="replan",
                        reason=reason,
                        task_ids=("T2",),
                        evidence_use_ids=(),
                    ),
                ),
            )
        }
    )


# ---------------------------------------------------------------------------
# Replan validation: thin runtime wrapper over the frozen validator (R31)
# ---------------------------------------------------------------------------


def test_replan_can_add_tasks_but_cannot_widen_authorization() -> None:
    from app.services.agents.v2.replanning import ReplanRejected, validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()
    from app.services.agents.v2.complex_research_graph import (
        build_task_execution_summaries,
    )

    outcomes = build_task_execution_summaries(())

    accepted = validate_runtime_replan(
        current, _appended(current), outcomes, _policy(), _budget(), context
    )
    assert [task.task_id for task in accepted.tasks] == ["T1", "T2", "T3"]

    # Widening authorization: rewriting a completed task's capability fails.
    widened = current.model_copy(
        update={
            "tasks": (
                current.tasks[0].model_copy(update={"capability": "people.lookup"}),
                current.tasks[1],
            )
            + current.tasks[2:]
        }
    )
    # (no new task appended either: prefix itself is rewritten)
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(
            current, widened, outcomes, _policy(), _budget(), context
        )

    # A capability outside the CURRENT runtime catalog cannot be appended,
    # even though the frozen shape rules would otherwise accept it.
    kg_append = current.model_copy(
        update={
            "tasks": current.tasks
            + (
                TaskSpec(
                    task_id="T3",
                    capability="knowledge_graph.query",
                    task_objective="KG lookup",
                    input=KnowledgeGraphInput(
                        kind="knowledge_graph.query", query="A"
                    ),
                    depends_on=(),
                    origin=ReplanTaskOrigin(
                        kind="replan",
                        reason="extra",
                        task_ids=(),
                        evidence_use_ids=(),
                    ),
                ),
            )
        }
    )
    with pytest.raises(ReplanRejected):
        validate_runtime_replan(
            current, kg_append, outcomes, _policy(), _budget(), context
        )


def test_replan_task_budget_exhausted() -> None:
    from app.services.agents.v2.replanning import validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(
            current,
            _appended(current),
            (),
            _policy(),
            _budget(max_tasks_remaining=0),
            context,
        )


def test_replan_budget_zero_replans_remaining() -> None:
    from app.services.agents.v2.replanning import validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(
            current,
            _appended(current),
            (),
            _policy(),
            _budget(max_replans_remaining=0),
            context,
        )


def test_replan_parallel_branch_bound() -> None:
    from app.services.agents.v2.replanning import ReplanRejected, validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()
    wide = current
    for index, target in (("T3", "t1"), ("T4", "t2"), ("T5", "t1")):
        wide = wide.model_copy(
            update={
                "tasks": wide.tasks
                + (
                    TaskSpec(
                        task_id=index,
                        capability="document.read",
                        task_objective=f"Re-read {target}",
                        input=DocumentReadInput(
                            kind="document.read", target_ids=(target,)
                        ),
                        depends_on=(),
                        origin=ReplanTaskOrigin(
                            kind="replan",
                            reason="fan out",
                            task_ids=(),
                            evidence_use_ids=(),
                        ),
                    ),
                )
            }
        )
    with pytest.raises(ReplanRejected):
        validate_runtime_replan(
            current, wide, (), _policy(), _budget(max_parallel_branches=2), context
        )


def test_completed_task_immutability() -> None:
    from app.services.agents.v2.replanning import validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()

    rewritten_objective = current.model_copy(
        update={
            "tasks": (
                current.tasks[0].model_copy(
                    update={"task_objective": "changed after completion"}
                ),
                current.tasks[1],
                _appended(current).tasks[2],
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(
            current, rewritten_objective, (), _policy(), _budget(), context
        )

    dropped = _appended(current).model_copy(
        update={"tasks": (_appended(current).tasks[1], _appended(current).tasks[2])}
    )
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(current, dropped, (), _policy(), _budget(), context)

    retargeted = _appended(current).model_copy(
        update={"target_units": _appended(current).target_units[:1]}
    )
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(current, retargeted, (), _policy(), _budget(), context)

    renamed = _appended(current).model_copy(
        update={"plan_id": "other", "goal": "other goal"}
    )
    with pytest.raises(ContractValidationError):
        validate_runtime_replan(current, renamed, (), _policy(), _budget(), context)


@pytest.mark.asyncio
async def test_replan_origin_carries_evidence_use_trigger_lineage(replan_settings: None) -> None:
    """Evidence-triggered replan records evidence_use_ids (spec §26)."""
    from app.services.agents.v2.complex_research_graph import (
        build_replan_proposal,
        complex_evaluate_node,
        complex_execute_node,
        validate_checkpoint_node,
    )

    _, _, context = _harness(
        run_id="run-lineage-1", missing=frozenset({"t2"})
    )

    # First pass only: drive validate/execute/evaluate manually through nodes.
    child = _child_input(replans_remaining=2)
    child = {**child, **await validate_checkpoint_node(child, context)}
    child = {**child, **await complex_execute_node(child, context)}
    child = {**child, **await complex_evaluate_node(child, context)}
    assert child["evaluation"].status == "insufficient"

    proposal = build_replan_proposal(child, context)
    assert proposal is not None
    new_tasks = proposal.tasks[len(child["plan"].tasks):]
    assert len(new_tasks) >= 1
    prior_use_ids = {
        ref.use_id for result in child["task_results"] for ref in result.evidence_uses
    }
    assert prior_use_ids, "the successful T1 read produced an admitted use"
    for task in new_tasks:
        assert isinstance(task.origin, ReplanTaskOrigin)
        assert set(task.origin.evidence_use_ids) <= prior_use_ids
        assert set(task.origin.task_ids) <= {
            task.task_id for task in child["plan"].tasks
        }
    assert prior_use_ids & {
        use_id
        for task in new_tasks
        for use_id in task.origin.evidence_use_ids
    }, "the replan preserves the triggering-use context"


def test_no_evidence_not_found_replan_stays_distinguishable() -> None:
    """T1=not_found with no uses stays a replan input, never a rerun ( §26)."""
    from app.services.agents.v2.complex_research_graph import (
        build_task_execution_summaries,
    )
    from app.services.agents.v2.replanning import validate_runtime_replan

    _, _, context = _harness()
    current = _two_target_plan()
    not_found_result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="not_found",
        data=PeopleLookupOutput(kind="people.lookup", matched=False),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    outcomes = build_task_execution_summaries((not_found_result,))
    assert outcomes[0].status == "not_found"
    assert outcomes[0].error_code is None
    # The replan appends only new tasks; T1 is never rerun and no use exists.
    accepted = validate_runtime_replan(
        current, _appended(current), outcomes, _policy(), _budget(), context
    )
    assert [task.task_id for task in accepted.tasks] == ["T1", "T2", "T3"]


@pytest.mark.asyncio
async def test_timeout_stays_distinguishable_from_not_found() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_task_execution_summaries,
    )
    from app.services.agents.v2.dependencies.people_document import (
        materialize_person_dependency,
    )

    timeout_result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="error",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error={"code": "TIMEOUT", "message": "upstream slow", "retryable": True},  # type: ignore[arg-type]
    )
    outcomes = build_task_execution_summaries((timeout_result,))
    assert outcomes[0].status == "error"
    assert outcomes[0].error_code == "TIMEOUT"

    _, _, context = _harness(run_id="run-timeout-1")
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=timeout_result,
        runtime=context,
        plan=TaskPlan(
            contract_version="2.0",
            plan_id="pp",
            goal="q",
            target_units=(),
            tasks=(
                TaskSpec(
                    task_id="T1",
                    capability="people.lookup",
                    task_objective="find A",
                    input=PeopleLookupInput(kind="people.lookup", query="A"),
                    depends_on=(),
                    origin=InitialTaskOrigin(kind="initial"),
                ),
            ),
        ),
        bindings=_bindings(),
        query="nghi dinh nao neu CCCD cua A",
    )
    assert outcome.kind == "failed"
    assert outcome.error_code == "TIMEOUT"
    assert outcome.kind != "not_found"


# ---------------------------------------------------------------------------
# Canonical loop: checkpointed replan, completed tasks never rerun (R32)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completed_task_is_not_rerun(replan_settings: None) -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    # T2's first read is truncated (it owns admitted evidence, so the task
    # gate passes) while t2 coverage stays incomplete: the replan re-reads
    # t2 once and must never redispatch the completed T1/T2 tasks.
    (document_capability, _), _, context = _harness(
        run_id="run-norerun-1", truncate_first_for=frozenset({"t2"})
    )
    child = _child_input(replans_remaining=2)
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    assert output["evaluation"].status == "sufficient"
    assert len(output["plan"].tasks) == 3
    served = [call[0].task_id for call in document_capability.calls]
    assert served.count("T1") == 1
    assert served.count("T2") == 1
    assert served.count("T3") == 1


@pytest.mark.asyncio
async def test_replan_loop_terminates_on_terminal_or_typed_failure() -> None:
    from app.services.agents.v2.complex_research_graph import (
        _decide_branch,
        build_complex_research_subgraph,
    )

    # No budget left: decide finalizes even when insufficient.
    _, _, context = _harness(run_id="run-term-1", missing=frozenset({"t2"}))
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(replans_remaining=0), context=context
    )
    assert output["evaluation"].status == "insufficient"
    assert len(output["plan"].tasks) == 2
    assert _decide_branch(output) == "finalize"

    # Contradictory evaluations never replan: the evaluator owns conflicts.
    contradictory = _child_input(
        plan=_two_target_plan(),
        evaluation=EvidenceEvaluation(
            status="contradictory",
            coverage=Coverage(items=()),
            missing=(),
            contradictions=(
                Contradiction(
                    contradiction_id="c1",
                    claim_a="A",
                    claim_b="not A",
                    evidence_use_ids=(uuid4(),),
                ),
            ),
        ),
        replans_remaining=2,
    )
    assert _decide_branch(contradictory) == "finalize"

    # Unsupported work stays the typed unavailable boundary.
    unsupported = _child_input(query_analysis=_analysis("evaluate"))
    assert _decide_branch(unsupported) == "finalize"


def test_canonical_loop_edges_checkpoint_replan_before_execute() -> None:
    from app.services.agents.v2.complex_research_graph import (
        _build_complex_research_graph,
    )

    graph = _build_complex_research_graph().compile()
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("plan", "validate_checkpoint") in edges
    assert ("validate_checkpoint", "execute") in edges
    assert ("execute", "materialize") in edges
    assert ("evaluate", "decide") in edges
    assert ("replan", "validate_checkpoint") in edges
    assert ("finalize", "__end__") in edges


@pytest.mark.asyncio
async def test_tool_gateway_checkpoints_replan_before_dispatch() -> None:
    """Propose checkpoint dispatch: the new task runs only via the accepted plan."""
    from app.services.agents.v2.execution import execute_ready_tasks

    registry_names = ("document.read", "section.read")
    capabilities = tuple(FakeReadCapability(name) for name in registry_names)
    leases = FakeLeases()
    base = CapabilityRuntimeContext(
        request_id="req-gw-1",
        run_id="run-gw-1",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=frozenset(registry_names),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability) for capability in capabilities],
        base,
    )
    runtime = GraphRuntimeContext(
        capability_runtime=base,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=leases,
            evidence_hydrator=FakeHydrator(capabilities),
        ),
    )
    gateway = AgentToolGateway()
    current = _two_target_plan()
    outcome = await gateway.propose(
        CapabilityInvocationProposal(
            capability="document.read",
            objective="Re-read t2",
            input=DocumentReadInput(kind="document.read", target_ids=("t2",)),
            depends_on=("T2",),
        ),
        current,
        runtime,
    )
    assert outcome.accepted is True
    # The gateway never dispatches: proposing alone executes nothing.
    assert all(capability.calls == [] for capability in capabilities)
    assert not hasattr(gateway, "checkpoint")

    # The CURRENT checkpoint runs only T1/T2...
    first = await execute_ready_tasks(
        plan=current, results=(), registry=registry, runtime=runtime,
        bindings=_bindings(),
    )
    assert {result.task_id for result in first.results} == {"T1", "T2"}

    # ...and the checkpointed replan dispatches only the NEW task after it.
    second = await execute_ready_tasks(
        plan=outcome.plan,
        results=first.results,
        registry=registry,
        runtime=runtime,
        bindings=_bindings(),
    )
    assert {result.task_id for result in second.results} == {"T1", "T2", "T3"}
    assert require_planned_dispatch(outcome.plan, "T3", runtime).task_id == "T3"


@pytest.mark.asyncio
async def test_gateway_rejects_when_planned_task_leaves_current_catalog() -> None:
    capabilities = (FakeReadCapability("document.read"),)
    full_runtime = CapabilityRuntimeContext(
        request_id="req-gw-2",
        run_id="run-gw-2",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=frozenset({"document.read", "section.read"}),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    full_registry = build_capability_registry(
        [CapabilityRegistration(capability=capability) for capability in capabilities]
        + [
            CapabilityRegistration(
                capability=FakeReadCapability("section.read"),
            )
        ],
        full_runtime,
    )
    gateway = AgentToolGateway()
    current = _two_target_plan()
    outcome = await gateway.propose(
        CapabilityInvocationProposal(
            capability="section.read",
            objective="Re-read t1 by section",
            input=SectionReadInput(kind="section.read", target_ids=("t1",)),
        ),
        current,
        GraphRuntimeContext(
            capability_runtime=full_runtime,
            services=RuntimeServices(capability_registry=full_registry),
        ),
    )
    assert outcome.accepted is True

    narrowed_runtime = CapabilityRuntimeContext(
        request_id="req-gw-2",
        run_id="run-gw-2",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=frozenset({"document.read"}),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    narrowed_registry = build_capability_registry(
        [CapabilityRegistration(capability=capabilities[0])],
        narrowed_runtime,
    )
    narrowed = GraphRuntimeContext(
        capability_runtime=narrowed_runtime,
        services=RuntimeServices(capability_registry=narrowed_registry),
    )
    # The previously accepted section task left the current catalog: a new
    # proposal against the narrowed runtime cannot widen it back.
    retry = await gateway.propose(
        CapabilityInvocationProposal(
            capability="section.read",
            objective="Re-read t1 by section",
            input=SectionReadInput(kind="section.read", target_ids=("t1",)),
        ),
        current,
        narrowed,
    )
    assert retry.accepted is False
    assert retry.plan == current


# ---------------------------------------------------------------------------
# Discovery semantics: UUID identity, pin/promotion, no autonomous targets (R33)
# ---------------------------------------------------------------------------


def test_discovery_candidate_uuid_uniqueness() -> None:
    from app.services.agents.v2.discovery import mint_candidate

    first = mint_candidate(DOC_A, str(REV_A))
    second = mint_candidate(DOC_A, str(REV_A))
    assert isinstance(first.candidate_id, UUID)
    assert first.candidate_id != second.candidate_id
    assert isinstance(first, DocumentDiscoveryCandidate)


def test_parallel_discovery_tasks_cannot_collide() -> None:
    from app.services.agents.v2.discovery import mint_candidate
    from app.services.agents.v2.tools.discovery_candidates import (
        DiscoveryCandidateRegistry,
    )

    candidates = tuple(mint_candidate(DOC_A, str(REV_A)) for _ in range(8))
    ids = [candidate.candidate_id for candidate in candidates]
    assert len(set(ids)) == len(ids)
    results = tuple(
        AgentResult(
            contract_version="2.0",
            task_id=f"T{index}",
            status="success",
            data=DocumentSearchOutput(
                kind="document.search", candidates=(candidate,)
            ),
            evidence_uses=(),
            coverage_observations=(),
            error=None,
        )
        for index, candidate in enumerate(candidates)
    )
    registry = DiscoveryCandidateRegistry.from_results(results)
    assert len(registry) == len(candidates)


def test_no_autonomous_target_creation() -> None:
    from app.services.agents.v2.discovery import (
        DiscoveryDisabled,
        request_addition,
    )
    from app.services.agents.v2.tools.discovery_candidates import (
        DiscoveryCandidateRegistry,
    )

    result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=DocumentSearchOutput(
            kind="document.search",
            candidates=(
                DocumentDiscoveryCandidate(
                    candidate_id=uuid4(),
                    document_id=DOC_A,
                    document_revision=str(REV_A),
                ),
            ),
        ),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    registry = DiscoveryCandidateRegistry.from_results([result])
    candidate_id = registry.candidates()[0].candidate_id
    # The planner cannot create a new user target from discovery.
    with pytest.raises((DiscoveryDisabled, ValueError, ContractValidationError)):
        request_addition(
            registry, candidate_id, "target", policy=_policy()  # type: ignore[arg-type]
        )
    # Additions create supporting/discovered bindings only.
    addition = request_addition(
        registry,
        candidate_id,
        "discovered",
        policy=_policy(allow_reference_discovery=True, max_discovered_documents=1),
    )
    assert addition.requested_role == "discovered"
    supporting = request_addition(
        registry,
        candidate_id,
        "supporting",
        policy=_policy(allow_supporting_discovery=True, max_discovered_documents=1),
    )
    assert supporting.requested_role == "supporting"
    # The frozen replan validator rejects autonomous target-unit additions.
    current = _two_target_plan()
    proposed = current.model_copy(
        update={
            "target_units": current.target_units
            + (
                TargetUnit(
                    target_id="t3",
                    binding_id="b1",
                    requested_locator=DocumentLocator(kind="document"),
                    completion_criteria=(CoverageCriterion(kind="coverage"),),
                ),
            ),
            "tasks": current.tasks + (_appended(current).tasks[2],),
        }
    )
    with pytest.raises(ContractValidationError):
        validate_replan(current, proposed, (), _policy(), _budget())


def test_policy_disabled_discovery() -> None:
    from app.services.agents.v2.discovery import DiscoveryDisabled, request_addition
    from app.services.agents.v2.tools.discovery_candidates import (
        DiscoveryCandidateRegistry,
    )

    candidate_id = uuid4()
    registry = DiscoveryCandidateRegistry.from_results(
        [
            AgentResult(
                contract_version="2.0",
                task_id="T1",
                status="success",
                data=DocumentSearchOutput(
                    kind="document.search",
                    candidates=(
                        DocumentDiscoveryCandidate(
                            candidate_id=candidate_id,
                            document_id=DOC_A,
                            document_revision=str(REV_A),
                        ),
                    ),
                ),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )
        ]
    )
    with pytest.raises(DiscoveryDisabled):
        request_addition(registry, candidate_id, "discovered", policy=_policy())
    # A document.search replan is rejected while discovery is disabled.
    current = _two_target_plan()
    search_append = current.model_copy(
        update={
            "tasks": current.tasks
            + (
                TaskSpec(
                    task_id="T3",
                    capability="document.search",
                    task_objective="discover more",
                    input=DocumentSearchInput(
                        kind="document.search", query="Dieu 5"
                    ),
                    depends_on=(),
                    origin=ReplanTaskOrigin(
                        kind="replan",
                        reason="discover",
                        task_ids=(),
                        evidence_use_ids=(),
                    ),
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_replan(current, search_append, (), _policy(), _budget())


def test_discovery_acl_denial() -> None:
    from app.services.agents.v2.discovery import (
        DiscoveryDenied,
        require_candidate_visible,
    )

    candidate = DocumentDiscoveryCandidate(
        candidate_id=uuid4(),
        document_id=DOC_A,
        document_revision=str(REV_A),
    )
    assert require_candidate_visible(candidate, frozenset({DOC_A})) is candidate
    with pytest.raises(DiscoveryDenied):
        require_candidate_visible(candidate, frozenset({DOC_B}))


def test_exact_revision_pin_and_promotion() -> None:
    from app.services.agents.v2.discovery import resolve_pin, validate_promotion
    from app.services.agents.v2.discovery import PromotionRequiresApproval
    from app.services.agents.v2.tools.discovery_candidates import (
        DiscoveryCandidateRegistry,
    )

    candidate_id = uuid4()
    registry = DiscoveryCandidateRegistry.from_results(
        [
            AgentResult(
                contract_version="2.0",
                task_id="T1",
                status="success",
                data=DocumentSearchOutput(
                    kind="document.search",
                    candidates=(
                        DocumentDiscoveryCandidate(
                            candidate_id=candidate_id,
                            document_id=DOC_A,
                            document_revision=str(REV_A),
                        ),
                    ),
                ),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )
        ]
    )
    document_id, revision = resolve_pin(registry, candidate_id)
    assert document_id == DOC_A
    assert revision == str(REV_A)

    request = BindingPromotionRequest(
        source_binding_id="b1", requested_role="reference"
    )
    # Promotion requires an explicit validated policy/user action.
    with pytest.raises(PromotionRequiresApproval):
        validate_promotion(_bindings(), request, approved=False, reason="compare needs it")
    with pytest.raises(PromotionRequiresApproval):
        validate_promotion(_bindings(), request, approved=True, reason="  ")
    approved = validate_promotion(
        _bindings(), request, approved=True, reason="user confirmed reference"
    )
    assert approved == request
    with pytest.raises((PromotionRequiresApproval, ContractValidationError)):
        validate_promotion(
            _bindings(),
            BindingPromotionRequest(
                source_binding_id="missing", requested_role="reference"
            ),
            approved=True,
            reason="user confirmed",
        )


def test_current_latest_rebinding() -> None:
    from app.services.agents.v2.discovery import pin_for_rebinding

    # An ordinary pin stays pinned when a newer revision appears.
    assert (
        pin_for_rebinding(
            pinned_revision=str(REV_A),
            latest_revision="rev-new",
            is_current_required=False,
            user_authorized_refresh=True,
        )
        == str(REV_A)
    )
    # Explicit current/latest semantics rebind only with user authorization.
    assert (
        pin_for_rebinding(
            pinned_revision=str(REV_A),
            latest_revision="rev-new",
            is_current_required=True,
            user_authorized_refresh=False,
        )
        == str(REV_A)
    )
    assert (
        pin_for_rebinding(
            pinned_revision=str(REV_A),
            latest_revision="rev-new",
            is_current_required=True,
            user_authorized_refresh=True,
        )
        == "rev-new"
    )


# ---------------------------------------------------------------------------
# Model-facing projections carry no People scalar (R34)
# ---------------------------------------------------------------------------


def test_replan_input_carries_no_people_scalar() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_model_observations,
        build_model_replan_input,
    )
    from app.services.agents.v2.dependencies.people_document import (
        append_materialized_dependent,
    )

    base = TaskPlan(
        contract_version="2.0",
        plan_id="pp",
        goal="nghi dinh nao neu CCCD cua A",
        target_units=(),
        tasks=(
            TaskSpec(
                task_id="T1",
                capability="people.lookup",
                task_objective="find A",
                input=PeopleLookupInput(kind="people.lookup", query="A"),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )
    use_id = uuid4()
    materialized = append_materialized_dependent(
        current=base,
        outcomes=(
            # Success outcome for the people task (typed, no evidence content).
            TaskExecutionSummary(task_id="T1", status="success"),
        ),
        outcome=PeopleDocumentMaterialization(
            kind="materialized",
            scalar="012345678901",
            input=DocumentSearchInput(
                kind="document.search",
                query="nghi dinh",
                person_identifier="012345678901",
            ),
            error_code=None,
            reason="governed scalar materialized server-side",
            evidence_use_ids=(use_id,),
        ),
        query="nghi dinh",
        next_task_id="T2",
        policy=_policy(
            allow_supporting_discovery=True, max_discovered_documents=1
        ),
        budget=_budget(max_tasks_remaining=7, max_replans_remaining=1),
    )
    assert materialized.tasks[1].input.person_identifier == "012345678901"  # type: ignore[union-attr]

    _, _, context = _harness(run_id="run-redact-1")
    child = _child_input(
        plan=materialized,
        task_results=(),
        evaluation=None,
        replans_remaining=1,
        people_scalar_available={"T1": True},
    )
    model_input = build_model_replan_input(child, context)
    payload = model_input.model_dump_json()
    assert "012345678901" not in payload
    assert model_input.current_plan is not None
    for task in model_input.current_plan.tasks:
        assert getattr(task.input, "person_identifier", None) is None

    people_result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(EvidenceUseRef(use_id=use_id),),
        coverage_observations=(),
        error=None,
    )
    observations = build_model_observations(
        _child_input(
            plan=materialized,
            task_results=(people_result,),
            people_scalar_available={"T1": True},
        )
    )
    assert len(observations) == 1
    assert observations[0].projection.dependency_scalar_available is True  # type: ignore[union-attr]
    assert "012345678901" not in observations[0].model_dump_json()


# ---------------------------------------------------------------------------
# Summarize is a skill, not an agent route (R35)
# ---------------------------------------------------------------------------


def test_summary_is_skill_not_agent_route() -> None:
    from app.services.agents.v2.skills.summarize import policy as summarize_policy

    assert callable(summarize_policy.build_summarize_plan)
    assert summarize_policy.supports_work_type("summarize") is True
    assert summarize_policy.supports_work_type("compare") is False
    assert summarize_policy.supports_work_type("evaluate") is False
    v2_root = Path(__file__).resolve().parents[5] / "app" / "services" / "agents" / "v2"
    assert list(v2_root.glob("*summary_agent*")) == []
    import app.services.agents.v2.complex_research_graph as subgraph_module

    source = inspect.getsource(subgraph_module)
    assert "SummaryAgent" not in source
    assert "summary_agent" not in source


def test_bounded_summarize_is_read_evaluate_synthesize_ground() -> None:
    from app.services.agents.v2.complex_research_graph import build_planning_input
    from app.services.agents.v2.skills.summarize import policy as summarize_policy

    _, _, context = _harness(run_id="run-summarize-1")
    single_bindings = DocumentBindingSet(
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
    child = _child_input(
        query_analysis=_analysis("summarize"), bindings=single_bindings
    )
    plan = summarize_policy.build_summarize_plan(build_planning_input(child, context))
    assert len(plan.target_units) == 1
    assert len(plan.tasks) == 1
    assert plan.tasks[0].capability in ("document.read", "section.read")
    assert isinstance(plan.tasks[0].origin, InitialTaskOrigin)
    import app.services.agents.v2.skills.summarize.policy as policy_module

    source = inspect.getsource(policy_module)
    assert "capability.execute(" not in source
    assert "TaskScheduler" not in source
    assert "FinalResponse" not in source
    assert "EvidenceUse" not in source


# ---------------------------------------------------------------------------
# Subagents are advisory only: no plan/task/execution authority (§9)
# ---------------------------------------------------------------------------


def test_subagent_cannot_append_authoritative_tasks() -> None:
    """A fabricated subagent task has no authority until validated+checkpointed."""
    _, _, context = _harness()
    current = _two_target_plan()
    # A subagent invents a task outside the governed boundary: the
    # pre-dispatch guard rejects it because no validated checkpoint owns it.
    with pytest.raises(UnplannedCapabilityDispatch):
        require_planned_dispatch(current, "T-subagent-1", context)
    # Advisory text is not a TaskSpec: it cannot enter the plan or scheduler.
    assert not hasattr(current, "append_subagent_task")
    import app.services.agents.v2.complex_research_graph as subgraph_module

    source = inspect.getsource(subgraph_module)
    assert "subagent" not in source.lower()


def test_subagent_cannot_receive_raw_people_record_or_runtime_secrets() -> None:
    """Advisory consumers see typed observations only: no raw record, no secrets."""
    from app.services.agents.v2.tools.adapters import AgentToolAdapter
    from app.services.agents.v2.tools.observations import ObservationProjector

    _, _, context = _harness(run_id="run-subagent-1")
    raw_scalar = "987654321098"
    people_result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(EvidenceUseRef(use_id=uuid4()),),
        coverage_observations=(),
        error=None,
    )
    observation = ObservationProjector.project(
        people_result, dependency_scalar_available=True
    )
    payload = observation.model_dump_json()
    assert raw_scalar not in payload
    for secret in (
        "workspace",
        "user_id",
        "deadline",
        "secret",
        "token",
        "acl",
        "service",
        "storage_key",
    ):
        assert secret not in payload.lower()

    adapter = AgentToolAdapter(context.services.capability_registry)
    runtime_only = {
        "request_id",
        "run_id",
        "user_id",
        "workspace_ids",
        "can_read_people",
        "allowed_capabilities",
        "deadline_at",
    }
    for name in adapter.visible_tool_names():
        assert not (set(adapter.input_fields(name)) & runtime_only), name

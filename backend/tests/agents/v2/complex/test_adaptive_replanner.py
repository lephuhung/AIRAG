"""Task 11 — evaluator-driven bounded append-only replan (Phase 5 backend).

Proves the governed replan boundary: the existing evaluator contracts
(``EvidenceEvaluation`` / ``MissingRequirement`` / ``Contradiction`` /
``TaskExecutionSummary``) are reused (imported, never redefined), only a
minimized gap projection reaches the model, deterministic gap coverage wins
when it can construct (zero model calls), the model path serves only
advisable gaps the deterministic policy cannot build, every replan stays
append-only (completed tasks immutable) through the single governed append,
and bounds (max tasks/replans/budgets/catalog/hard scope) hold with zero
dispatch on any failure.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.contracts.binding import (
    DocumentBindingSet,
    ScopedDocument,
)
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    DocumentReadInput,
    DocumentReadOutput,
    DocumentSearchInput,
)
from app.services.agents.v2.contracts.evaluation import (
    Contradiction,
    Coverage,
    CoverageItem,
    CoverageObservation,
    EvidenceEvaluation,
    MissingRequirement,
)
from app.services.agents.v2.contracts.evidence import (
    DocumentSourceIdentity,
    EvidenceUseRef,
)
from app.services.agents.v2.contracts.execution import (
    AgentRequest,
    AgentResult,
    TaskExecutionSummary,
)
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.state import (
    GraphRuntimeContext,
    RuntimeServices,
)
from app.services.agents.v2.planning.replan_projection import (
    build_replanner_model_input,
)
from app.services.agents.v2.planning.replanner import (
    AdaptiveReplanner,
    ReplannerError,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")
REV_A = UUID("a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1")
REV_B = UUID("b2b2b2b2-b2b2-b2b2-b2b2-b2b2b2b2b2b2")
REPLAN_QUERY = "So sánh Chương II A và Chương III B"


@pytest.fixture
def replan_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the bounded replan loop: deployment settings allow 2 replans."""
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(
            V2_MAX_TASKS=8, V2_MAX_PARALLEL_BRANCHES=2, V2_MAX_REPLANS=2
        ),
    )


class FakeChunk:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeReplanProvider:
    """Model double: serves one canned JSON proposal, records calls."""

    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.calls: list[tuple] = []

    async def astream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((messages, kwargs))
        yield FakeChunk(json.dumps(self._payload))


class FakeReadCapability:
    """Atomic stub: t2 reads missing (unless constructed without gaps)."""

    def __init__(
        self,
        name: str = "document.read",
        *,
        missing_targets: frozenset[str] = frozenset({"t2"}),
    ) -> None:
        domain = "section" if name == "section.read" else "document"
        self.descriptor = CapabilityDescriptor(
            name=name,  # type: ignore[arg-type]
            domain=domain,  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=True,
        )
        self.calls: list[tuple[AgentRequest, object]] = []
        self.uses: dict[UUID, tuple[str, str]] = {}
        self.missing_targets = missing_targets

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        target_id = request.input.target_ids[0]  # type: ignore[union-attr]
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


class _CatalogStub:
    """Zero-dispatch catalog filler so the registry view carries the name."""

    def __init__(self, capability_name: str) -> None:
        self.descriptor = CapabilityDescriptor(
            name=capability_name,  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="search" if capability_name == "document.search" else "read",  # type: ignore[arg-type]
            supports_parallel=True,
        )
        self.calls: list = []

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        raise AssertionError("catalog filler must never dispatch")


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


def _semantic() -> SemanticContext:
    return SemanticContext(
        contextualized_query=REPLAN_QUERY,
        normalized_query=REPLAN_QUERY.lower(),
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


def _analysis() -> QueryAnalysis:
    return QueryAnalysis(work_type="compare", domains=("document",))  # type: ignore[arg-type]


def _two_target_plan() -> TaskPlan:
    def _read_task(task_id: str, target: str) -> TaskSpec:
        return TaskSpec(
            task_id=task_id,
            capability="document.read",
            task_objective=f"Read {target}",
            input=DocumentReadInput(kind="document.read", target_ids=(target,)),
            depends_on=(),
            origin=InitialTaskOrigin(kind="initial"),
        )

    return TaskPlan(
        contract_version="2.0",
        plan_id="p1",
        goal=REPLAN_QUERY,
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


def _coverage_gap_evaluation() -> EvidenceEvaluation:
    return EvidenceEvaluation(
        status="insufficient",
        coverage=Coverage(
            items=(
                CoverageItem(
                    target_id="t1",
                    observed_locators=(DocumentLocator(kind="document"),),
                    status="read_complete",
                ),
                CoverageItem(
                    target_id="t2",
                    observed_locators=(),
                    status="truncated",
                ),
            )
        ),
        missing=(
            MissingRequirement(
                target_id="t2",
                criterion_kind="coverage",
                semantic_criterion_id=None,
                description="target t2 has no admitted read coverage",
            ),
        ),
        contradictions=(),
    )


def _harness(
    run_id: str,
    *,
    replanner: AdaptiveReplanner | None,
    capability_names: frozenset[str] = frozenset({"document.read"}),
    allowed: frozenset[str] | None = None,
    missing_targets: frozenset[str] = frozenset({"t2"}),
) -> tuple[FakeReadCapability, FakeLeases, GraphRuntimeContext]:
    """Registry carries exactly ``capability_names`` (deterministic coverage
    needs a read cap in the catalog; the retrieve-only tests exclude it)."""
    from app.services.agent.runtime_selector import PlanBindingResolver

    reads: list[FakeReadCapability] = []
    fillers: list[_CatalogStub] = []
    for name in sorted(capability_names):
        if name in ("document.read", "section.read"):
            reads.append(FakeReadCapability(name, missing_targets=missing_targets))
        else:
            fillers.append(_CatalogStub(name))
    capability = reads[0] if reads else FakeReadCapability(missing_targets=missing_targets)
    leases = FakeLeases()
    granted = allowed if allowed is not None else capability_names
    runtime = CapabilityRuntimeContext(
        request_id=f"req-{run_id}",
        run_id=run_id,
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=granted,
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    registry = build_capability_registry(
        [CapabilityRegistration(capability=entry) for entry in (*reads, *fillers)],
        runtime,
    )
    context = GraphRuntimeContext(
        capability_runtime=runtime,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=leases,
            evidence_hydrator=FakeHydrator(capability),
            pinned_target_resolver=PlanBindingResolver(),
            adaptive_replanner=replanner,
        ),
    )
    return capability, leases, context


def _gap_state(capability: FakeReadCapability | None = None) -> dict:
    """Checkpointed state with an advisable coverage gap on t2 (R32 shape).

    When the executing capability is supplied, the preset T1 admitted use
    is registered with it so a later graph ``evaluate`` hydrates the same
    use instead of treating T1 as uncovered (mid-flight resume fidelity).
    """
    from app.services.agents.v2.complex_research_graph import ComplexResearchState

    t1_use = uuid4()
    t2_use = uuid4()
    if capability is not None:
        capability.uses[t1_use] = ("T1", "t1")
        capability.uses[t2_use] = ("T2", "t2")
    return ComplexResearchState(
        contract_version="2.0",
        semantic=_semantic(),
        bindings=_bindings(),
        query_analysis=_analysis(),
        route_decision=RouteDecision(
            route="complex_research", reason_code="comparison"
        ),
        plan=_two_target_plan(),
        task_results=(
            AgentResult(
                contract_version="2.0",
                task_id="T1",
                status="success",
                data=DocumentReadOutput(kind="document.read", read_unit_count=1),
                evidence_uses=(EvidenceUseRef(use_id=t1_use),),
                coverage_observations=(
                    CoverageObservation(
                        target_id="t1",
                        observed_locators=(DocumentLocator(kind="document"),),
                        outcome="read",
                    ),
                ),
                error=None,
            ),
            AgentResult(
                contract_version="2.0",
                task_id="T2",
                status="success",
                data=DocumentReadOutput(kind="document.read", read_unit_count=1),
                evidence_uses=(EvidenceUseRef(use_id=t2_use),),
                coverage_observations=(
                    CoverageObservation(
                        target_id="t2",
                        observed_locators=(),
                        outcome="truncated",
                    ),
                ),
                error=None,
            ),
        ),
        evaluation=_coverage_gap_evaluation(),
        replans_remaining=2,
    )


def _replanner_for(payload: object) -> tuple[AdaptiveReplanner, FakeReplanProvider]:
    provider = FakeReplanProvider(payload)
    return AdaptiveReplanner(provider_factory=lambda: provider), provider


# ---------------------------------------------------------------------------
# Projection: minimized evaluator gaps only
# ---------------------------------------------------------------------------


def test_replanner_model_input_projects_minimized_gaps_only(
    replan_settings: None,
) -> None:
    _, _, context = _harness("run-replan-proj", replanner=None)
    model_input = build_replanner_model_input(_gap_state(), context)
    payload = model_input.model_dump_json()
    # No trusted identity, internal evidence UUIDs, scalars, or raw evidence.
    assert str(DOC_A) not in payload
    assert str(DOC_B) not in payload
    assert str(REV_A) not in payload
    assert str(REV_B) not in payload
    assert "use_id" not in payload
    assert "person_identifier" not in payload
    assert "evidence" not in payload.lower()
    # The replanner still sees what it needs: gaps, outcomes, bounds.
    assert model_input.query == REPLAN_QUERY
    assert [gap.target_id for gap in model_input.gaps] == ["t2"]
    assert model_input.gaps[0].criterion_kind == "coverage"
    assert [outcome.task_id for outcome in model_input.outcomes] == ["T1", "T2"]
    assert model_input.outcomes[1].status == "success"
    assert model_input.budget.max_replans_remaining == 2
    assert "document.read" in {entry.name for entry in model_input.capability_catalog}


def test_replanner_projection_never_leaks_scalar_or_runtime_secret() -> None:
    """R51/R53 analogue: KNOWN scalar + KNOWN secret stay out of the projection."""
    import json

    sentinel_scalar = "987654321098"
    sentinel_secret = "sentinel-workspace-" + WORKSPACE_ID.hex[:8]
    _, _, context = _harness("run-replan-leak", replanner=None)
    context = GraphRuntimeContext(
        capability_runtime=context.capability_runtime,
        services=RuntimeServices(
            capability_registry=context.services.capability_registry,
            retention_leases=context.services.retention_leases,
            evidence_hydrator=context.services.evidence_hydrator,
            authorization={"runtime_secret": sentinel_secret},
        ),
    )
    assert sentinel_secret in json.dumps(context.services.authorization)
    plan = _two_target_plan()
    assert sentinel_scalar not in plan.model_dump_json()
    scalar_plan = plan.model_copy(
        update={
            "tasks": (
                plan.tasks[0],
                plan.tasks[1].model_copy(
                    update={
                        "capability": "document.search",
                        "input": DocumentSearchInput(
                            kind="document.search",
                            query="nghi dinh",
                            person_identifier=sentinel_scalar,
                        ),
                    }
                ),
            )
        }
    )
    assert sentinel_scalar in scalar_plan.model_dump_json()
    state = dict(_gap_state())
    state["plan"] = scalar_plan
    payload = build_replanner_model_input(state, context).model_dump_json()  # type: ignore[arg-type]
    assert sentinel_scalar not in payload
    assert sentinel_secret not in payload


# ---------------------------------------------------------------------------
# Deterministic gap coverage wins when it can construct (zero model calls)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_replan_preferred_over_model(
    replan_settings: None,
) -> None:
    replanner, provider = _replanner_for({"tasks": []})
    _, _, context = _harness("run-replan-det", replanner=replanner)
    proposal = await replanner.propose_replan(_gap_state(), context)
    assert proposal is not None
    assert [task.task_id for task in proposal.new_tasks] == ["T3"]
    assert proposal.new_tasks[0].capability == "document.read"
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Model path serves advisable gaps the deterministic policy cannot build
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_fallback_retrieve_serves_gap_without_read_catalog(
    replan_settings: None,
) -> None:
    """Read caps left the runtime catalog but retrieve is authorized: the
    model proposes a bounded retrieve re-read of the gap target."""
    replanner, provider = _replanner_for(
        {
            "tasks": [
                {
                    "capability": "document.retrieve",
                    "task_objective": "Re-read t2 via retrieve",
                    "targets": ["t2"],
                    "depends_on_steps": [],
                    "depends_on_completed": ["T2"],
                }
            ]
        }
    )
    _, _, context = _harness(
        "run-replan-ret",
        replanner=replanner,
        capability_names=frozenset({"document.retrieve"}),
        allowed=frozenset({"document.retrieve"}),
    )
    proposal = await replanner.propose_replan(_gap_state(), context)
    assert proposal is not None
    assert len(provider.calls) == 1
    (new_task,) = proposal.new_tasks
    assert new_task.task_id == "T3"
    assert new_task.capability == "document.retrieve"
    assert new_task.input.target_ids == ("t2",)  # type: ignore[union-attr]
    assert new_task.origin.kind == "replan"
    assert "T2" in new_task.origin.task_ids  # type: ignore[union-attr]
    # Both admitted uses survive as trigger lineage (deterministic parity).
    assert len(new_task.origin.evidence_use_ids) == 2  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_model_proposal_rejects_minted_target(
    replan_settings: None,
) -> None:
    replanner, provider = _replanner_for(
        {
            "tasks": [
                {
                    "capability": "document.retrieve",
                    "task_objective": "Read unknown",
                    "targets": ["t_unknown"],
                    "depends_on_steps": [],
                    "depends_on_completed": [],
                }
            ]
        }
    )
    _, _, context = _harness(
        "run-replan-mint",
        replanner=replanner,
        capability_names=frozenset({"document.retrieve"}),
        allowed=frozenset({"document.retrieve"}),
    )
    assert await replanner.propose_replan(_gap_state(), context) is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_model_proposal_rejects_scope_widening(
    replan_settings: None,
) -> None:
    replanner, provider = _replanner_for(
        {
            "tasks": [
                {
                    "capability": "people.lookup",
                    "task_objective": "Find A",
                    "targets": [],
                    "depends_on_steps": [],
                    "depends_on_completed": [],
                }
            ]
        }
    )
    _, _, context = _harness(
        "run-replan-scope",
        replanner=replanner,
        capability_names=frozenset({"document.retrieve", "people.lookup"}),
        allowed=frozenset({"document.retrieve", "people.lookup"}),
    )
    assert await replanner.propose_replan(_gap_state(), context) is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_model_proposal_over_task_budget_is_refused(
    replan_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(
            V2_MAX_TASKS=3, V2_MAX_PARALLEL_BRANCHES=2, V2_MAX_REPLANS=2
        ),
    )
    replanner, provider = _replanner_for(
        {
            "tasks": [
                {
                    "capability": "document.retrieve",
                    "task_objective": f"Re-read {index}",
                    "targets": ["t2"],
                    "depends_on_steps": [],
                    "depends_on_completed": [],
                }
                for index in range(5)
            ]
        }
    )
    _, _, context = _harness(
        "run-replan-budget",
        replanner=replanner,
        capability_names=frozenset({"document.retrieve"}),
        allowed=frozenset({"document.retrieve"}),
    )
    assert await replanner.propose_replan(_gap_state(), context) is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_model_garbage_finalizes_with_zero_dispatch(
    replan_settings: None,
) -> None:
    replanner, provider = _replanner_for({"tasks": "not-a-list"})
    capability, _, context = _harness(
        "run-replan-garbage",
        replanner=replanner,
        capability_names=frozenset({"document.retrieve"}),
        allowed=frozenset({"document.retrieve"}),
    )
    assert await replanner.propose_replan(_gap_state(), context) is None
    assert len(provider.calls) == 1
    assert capability.calls == []


@pytest.mark.asyncio
async def test_contradictory_evaluation_never_replans(
    replan_settings: None,
) -> None:
    replanner, provider = _replanner_for(
        {
            "tasks": [
                {
                    "capability": "document.retrieve",
                    "task_objective": "Re-read",
                    "targets": ["t2"],
                    "depends_on_steps": [],
                    "depends_on_completed": [],
                }
            ]
        }
    )
    _, _, context = _harness(
        "run-replan-contra",
        replanner=replanner,
        capability_names=frozenset({"document.retrieve"}),
        allowed=frozenset({"document.retrieve"}),
    )
    state = dict(_gap_state())
    state["evaluation"] = EvidenceEvaluation(
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
    )
    assert await replanner.propose_replan(state, context) is None  # type: ignore[arg-type]
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Graph boundary: unwired keeps legacy behavior; wired stays governed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unwired_replanner_keeps_legacy_deterministic_replan(
    replan_settings: None,
) -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_governed_replan_proposal,
        build_replan_proposal,
    )

    _, _, context = _harness("run-replan-unwired", replanner=None)
    assert context.services.adaptive_replanner is None
    legacy = build_replan_proposal(_gap_state(), context)
    governed = await build_governed_replan_proposal(_gap_state(), context)
    assert legacy is not None and governed is not None
    assert [task.task_id for task in governed.new_tasks] == [
        task.task_id for task in legacy.new_tasks
    ]


@pytest.mark.asyncio
async def test_governed_replan_end_to_end_appends_once(
    replan_settings: None,
) -> None:
    """Wired replanner on the real graph: deterministic T3 still appends
    exactly once and dispatches once; the model is never consulted."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    replanner, provider = _replanner_for({"tasks": []})
    capability, _, context = _harness(
        "run-replan-e2e", replanner=replanner, missing_targets=frozenset()
    )
    child = _gap_state(capability)
    child["replans_remaining"] = 2
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    assert [task.task_id for task in output["plan"].tasks] == ["T1", "T2", "T3"]
    assert [call[0].task_id for call in capability.calls] == ["T3"]
    assert provider.calls == []


def test_replanner_service_is_runtime_only() -> None:
    from app.services.agents.v2.contracts.base import ContractModel

    assert RuntimeServices().adaptive_replanner is None
    assert not (isinstance(AdaptiveReplanner(), ContractModel))
    assert isinstance(
        RuntimeServices(adaptive_replanner=AdaptiveReplanner()).adaptive_replanner,
        AdaptiveReplanner,
    )


def test_replanner_reuses_frozen_evaluator_contracts() -> None:
    """Task 11 reuses EvidenceEvaluation / MissingRequirement / Contradiction /
    TaskExecutionSummary: the frozen types imported, never redefined here."""
    import app.services.agents.v2.contracts.evaluation as evaluation_module
    import app.services.agents.v2.contracts.execution as execution_module
    import app.services.agents.v2.planning.replan_projection as projection_module
    import app.services.agents.v2.planning.replanner as replanner_module

    assert replanner_module.EvidenceEvaluation is evaluation_module.EvidenceEvaluation
    assert replanner_module.MissingRequirement is evaluation_module.MissingRequirement
    assert replanner_module.Contradiction is evaluation_module.Contradiction
    assert replanner_module.TaskExecutionSummary is execution_module.TaskExecutionSummary
    assert projection_module.MissingRequirement is evaluation_module.MissingRequirement
    assert TaskExecutionSummary(task_id="T1", status="success").task_id == "T1"
    assert ReplannerError("no governable replan") is not None

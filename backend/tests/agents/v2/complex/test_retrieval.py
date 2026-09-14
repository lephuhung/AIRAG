"""P0 Task 4 — deterministic retrieve planning through the shared scheduler.

Proves the retrieve policy proposes a validated, checkpointed
``document.retrieve`` plan (unscoped targetless or hard-scoped explicit
targets), the existing ``validate -> lease -> checkpoint -> TaskScheduler``
path dispatches exactly one task, and a missing catalog entry produces the
typed unavailable boundary with zero dispatch. Compare/summarize/cross-domain
behavior is unchanged.
"""
from __future__ import annotations

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
    DocumentRetrieveInput,
    DocumentRetrieveOutput,
)
from app.services.agents.v2.contracts.evidence import DocumentSourceIdentity, EvidenceUseRef
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
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
from app.services.agents.v2.contracts.validation import ContractValidationError
from app.services.agents.v2.nodes.evaluate import HydratedEvidence

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")
REV_A = UUID("a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1")
REV_B = UUID("b2b2b2b2-b2b2-b2b2-b2b2-b2b2b2b2b2b2")
RETRIEVE_QUERY = "Điều 5 của A nói gì?"


class FakeRetrieveCapability:
    """Atomic stub serving ``document.retrieve`` with count-only output.

    Mirrors the Task 2 capability contract: scoped calls persist one
    target-bound coverage use per requested target; unscoped calls persist
    one targetless supporting use; output carries only the count.
    """

    def __init__(self) -> None:
        self.descriptor = CapabilityDescriptor(
            name="document.retrieve",  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="search",
            supports_parallel=True,
        )
        self.calls: list[tuple[AgentRequest, object]] = []
        self.uses: dict[UUID, str | None] = {}

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        assert isinstance(request.input, DocumentRetrieveInput)
        target_ids = tuple(request.input.target_ids)
        if target_ids:
            uses = []
            for target_id in target_ids:
                use_id = uuid4()
                self.uses[use_id] = target_id
                uses.append(EvidenceUseRef(use_id=use_id))
        else:
            use_id = uuid4()
            self.uses[use_id] = None
            uses = [EvidenceUseRef(use_id=use_id)]
        return AgentResult(
            contract_version="2.0",
            task_id=request.task_id,
            status="success",
            data=DocumentRetrieveOutput(
                kind="document.retrieve", retrieved_unit_count=len(uses)
            ),
            evidence_uses=tuple(uses),
            coverage_observations=(),
            error=None,
        )


class FakeLeases:
    """Retention-lease double: records acquisitions, never releases."""

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

    async def release_run(self, run_id: str, reason: str = "terminal") -> int:
        return 0


class FakeHydrator:
    """Governed hydration double admitting the fake's retrieval uses.

    Scoped coverage uses hydrate with the pinned revision and the plan's own
    requested locator; unscoped supporting uses hydrate targetless.
    """

    def __init__(self, capability: FakeRetrieveCapability) -> None:
        self._capability = capability

    async def hydrate_for_evaluation(
        self, use_refs, *, runtime, plan, bindings
    ) -> tuple[HydratedEvidence, ...]:
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
        admitted: list[HydratedEvidence] = []
        for ref in use_refs:
            if ref.use_id not in self._capability.uses:
                continue
            target_id = self._capability.uses[ref.use_id]
            if target_id is None:
                admitted.append(
                    HydratedEvidence(
                        use_id=ref.use_id,
                        evidence_id=uuid4(),
                        task_id="T1",
                        purpose="supporting",
                        target_id=None,
                        content="unscoped chunk",
                        role=None,
                        source_label="workspace",
                        source_identity=DocumentSourceIdentity(
                            kind="document",
                            document_id=DOC_A,
                            document_revision=str(REV_A),
                            locator=DocumentLocator(kind="document"),
                        ),
                        classification="normal",
                        locator=DocumentLocator(kind="document"),
                        document_revision=str(REV_A),
                    )
                )
                continue
            binding = binding_by_target[target_id]
            locator = locator_by_target[target_id]
            admitted.append(
                HydratedEvidence(
                    use_id=ref.use_id,
                    evidence_id=uuid4(),
                    task_id="T1",
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
    ) -> tuple[HydratedEvidence, ...]:
        return await self.hydrate_for_evaluation(
            use_refs, runtime=runtime, plan=plan, bindings=bindings
        )

    async def persist_derived_summary(self, **kwargs) -> HydratedEvidence:
        raise AssertionError("retrieve pilot never persists derived summaries")


def _semantic(query: str = RETRIEVE_QUERY) -> SemanticContext:
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


def _bindings(*docs: tuple[str, UUID, UUID, str]) -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=tuple(
            ScopedDocument(
                binding_id=binding_id,
                document_id=document_id,
                document_revision=str(revision),
                role=role,  # type: ignore[arg-type]
            )
            for binding_id, document_id, revision, role in docs
        ),
        revision_requirement_refs=(),
    )


def _unscoped_bindings() -> DocumentBindingSet:
    return _bindings()


def _scoped_bindings() -> DocumentBindingSet:
    return _bindings(("b_r1", DOC_A, REV_A, "target"))


def _analysis(work_type: str = "retrieve") -> QueryAnalysis:
    return QueryAnalysis(work_type=work_type, domains=("document",))  # type: ignore[arg-type]


def _planning_input(
    bindings: DocumentBindingSet,
    capability_names: frozenset[str] = frozenset({"document.retrieve"}),
    semantic: SemanticContext | None = None,
) -> ResearchPlanningInput:
    from app.services.agents.v2.contracts.capability import CapabilityDescriptor

    return ResearchPlanningInput(
        semantic=semantic or _semantic(),
        bindings=bindings,
        query_analysis=_analysis(),
        capability_catalog=tuple(
            CapabilityDescriptor(
                name=name,  # type: ignore[arg-type]
                domain="document",  # type: ignore[arg-type]
                operation_type="search",
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
            max_tasks_remaining=8, max_replans_remaining=0, max_parallel_branches=2
        ),
    )


def _capability_runtime(
    run_id: str = "run-retrieve-1",
    allowed: frozenset[str] | None = None,
) -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id="req-retrieve-1",
        run_id=run_id,
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=(
            allowed if allowed is not None else frozenset({"document.retrieve"})
        ),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


def _harness(
    run_id: str = "run-retrieve-1",
    *,
    bindings: DocumentBindingSet | None = None,
    allowed: frozenset[str] | None = None,
) -> tuple[FakeRetrieveCapability, FakeLeases, GraphRuntimeContext]:
    from app.services.agent.runtime_selector import PlanBindingResolver

    capability = FakeRetrieveCapability()
    leases = FakeLeases()
    runtime = _capability_runtime(run_id, allowed)
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], runtime
    )
    context = GraphRuntimeContext(
        capability_runtime=runtime,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=leases,
            evidence_hydrator=FakeHydrator(capability),
            # Production ingress wires the request-scoped resolver the
            # shared scheduler feeds before dispatch; targeted retrieve
            # harnesses must wire one too (F3 raises on unwired targeted
            # plans instead of masquerading as a denial).
            pinned_target_resolver=PlanBindingResolver(),
        ),
    )
    return capability, leases, context


def _child_input(
    bindings: DocumentBindingSet,
    query: str = RETRIEVE_QUERY,
) -> dict:
    from app.services.agents.v2.complex_research_graph import ComplexResearchState

    return ComplexResearchState(
        contract_version="2.0",
        semantic=_semantic(query),
        bindings=bindings,
        query_analysis=_analysis(),
        route_decision=RouteDecision(
            route="complex_research", reason_code="multi_document_research"
        ),
        plan=None,
        task_results=(),
        evaluation=None,
        replans_remaining=0,
    )


# ---------------------------------------------------------------------------
# Policy: deterministic retrieve plans
# ---------------------------------------------------------------------------


def test_retrieve_is_skill_not_agent_route() -> None:
    from app.services.agents.v2.skills.retrieve import policy as retrieve_policy

    assert callable(retrieve_policy.build_retrieve_plan)
    assert retrieve_policy.supports_work_type("retrieve") is True
    assert retrieve_policy.supports_work_type("compare") is False


def test_unscoped_retrieve_plans_single_targetless_task() -> None:
    from app.services.agents.v2.skills.retrieve import policy as retrieve_policy

    plan = retrieve_policy.build_retrieve_plan(_planning_input(_unscoped_bindings()))
    assert plan.target_units == ()
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.task_id == "T1"
    assert task.capability == "document.retrieve"
    assert isinstance(task.input, DocumentRetrieveInput)
    assert task.input.target_ids == ()
    assert task.input.query == RETRIEVE_QUERY
    assert task.input.top_k == 8
    assert task.depends_on == ()


def test_scoped_retrieve_creates_target_units_and_task_targets() -> None:
    from app.services.agents.v2.skills.retrieve import policy as retrieve_policy

    plan = retrieve_policy.build_retrieve_plan(_planning_input(_scoped_bindings()))
    assert len(plan.target_units) == 1
    unit = plan.target_units[0]
    assert unit.target_id == "t1"
    assert unit.binding_id == "b_r1"
    assert isinstance(unit.requested_locator, DocumentLocator)
    criterion = next(
        c for c in unit.completion_criteria if isinstance(c, CoverageCriterion)
    )
    assert criterion.minimum_status == "read_partial"
    task = plan.tasks[0]
    assert task.capability == "document.retrieve"
    assert task.input.target_ids == ("t1",)


def test_scoped_retrieve_dedupes_same_document_identity() -> None:
    """A query ref and an explicit ref to one document yield one target unit."""
    from app.services.agents.v2.skills.retrieve import policy as retrieve_policy

    bindings = _bindings(
        ("b_r1", DOC_A, REV_A, "target"),
        ("b_api_explicit:9", DOC_A, REV_A, "target"),
        ("b_r2", DOC_B, REV_B, "target"),
    )
    plan = retrieve_policy.build_retrieve_plan(_planning_input(bindings))
    assert len(plan.target_units) == 2
    bound_documents = {
        next(
            b.document_id
            for b in bindings.bindings
            if b.binding_id == unit.binding_id
        )
        for unit in plan.target_units
    }
    assert bound_documents == {DOC_A, DOC_B}
    task = plan.tasks[0]
    assert tuple(sorted(task.input.target_ids)) == tuple(
        sorted(unit.target_id for unit in plan.target_units)
    )


def test_scoped_retrieve_ignores_non_target_roles() -> None:
    from app.services.agents.v2.skills.retrieve import policy as retrieve_policy

    bindings = _bindings(
        ("b_r1", DOC_A, REV_A, "target"),
        ("b_ref", DOC_B, REV_B, "reference"),
    )
    plan = retrieve_policy.build_retrieve_plan(_planning_input(bindings))
    assert [unit.binding_id for unit in plan.target_units] == ["b_r1"]
    assert plan.tasks[0].input.target_ids == ("t1",)


def test_retrieve_without_catalog_entry_fails_closed() -> None:
    from app.services.agents.v2.skills.retrieve import policy as retrieve_policy

    with pytest.raises(ContractValidationError):
        retrieve_policy.build_retrieve_plan(
            _planning_input(_unscoped_bindings(), frozenset({"document.read"}))
        )


def test_build_initial_proposal_selects_retrieve() -> None:
    from app.services.agents.v2.complex_research_graph import build_initial_proposal

    proposal = build_initial_proposal(_planning_input(_unscoped_bindings()))
    assert proposal.plan.tasks[0].capability == "document.retrieve"
    assert proposal.reduce_spec is None


def test_build_initial_proposal_preserves_other_work_types() -> None:
    from app.services.agents.v2.complex_research_graph import build_initial_proposal
    from app.services.agents.v2.contracts.validation import ContractValidationError

    # Task 12 covers evaluate deterministically; uncovered work (explain)
    # still has no complex skill policy and stays the typed unavailable
    # boundary.
    with pytest.raises(ContractValidationError):
        build_initial_proposal(_planning_input(_unscoped_bindings()).model_copy(
            update={"query_analysis": _analysis("explain")}
        ))


@pytest.mark.asyncio
async def test_missing_catalog_produces_typed_unavailable_with_zero_dispatch() -> None:
    """No ``document.retrieve`` in the catalog: no plan, no dispatch."""
    from app.services.agents.v2.complex_research_graph import validate_checkpoint_node

    capability, _, context = _harness(
        run_id="run-retrieve-nocatalog", allowed=frozenset({"document.read"})
    )
    output = await validate_checkpoint_node(
        _child_input(_unscoped_bindings()), _runtime_for(context)
    )
    assert output.get("plan") is None
    assert capability.calls == []


@pytest.mark.asyncio
async def test_missing_catalog_decide_returns_typed_unavailable() -> None:
    from app.services.agents.v2.complex_research_graph import (
        COMPLEX_RESEARCH_UNAVAILABLE,
        decide_node,
    )

    _, _, context = _harness(run_id="run-retrieve-nocatalog-2")
    state = _child_input(_unscoped_bindings())
    output = await decide_node(state, _runtime_for(context))
    assert output["unavailable"].code == COMPLEX_RESEARCH_UNAVAILABLE


def _runtime_for(context: GraphRuntimeContext):  # type: ignore[no-untyped-def]
    from langgraph.runtime import Runtime

    return Runtime(context=context)


@pytest.mark.asyncio
async def test_unscoped_retrieve_checkpoints_then_dispatches_once() -> None:
    """Real validate -> lease -> checkpoint -> scheduler path, one dispatch."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    capability, _, context = _harness(run_id="run-retrieve-unscoped")
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(_unscoped_bindings()), context=context
    )
    assert output["plan"] is not None
    assert len(output["plan"].tasks) == 1
    assert output["plan"].tasks[0].capability == "document.retrieve"
    assert len(output["task_results"]) == 1
    assert len(capability.calls) == 1
    assert output["evaluation"].status == "sufficient"


@pytest.mark.asyncio
async def test_scoped_retrieve_graph_is_sufficient_on_pinned_revision() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    capability, _, context = _harness(run_id="run-retrieve-scoped")
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(_scoped_bindings()), context=context
    )
    assert output["plan"] is not None
    assert [unit.target_id for unit in output["plan"].target_units] == ["t1"]
    assert len(capability.calls) == 1
    assert output["evaluation"].status == "sufficient"
    assert output["evaluation"].coverage.items[0].status == "read_partial"

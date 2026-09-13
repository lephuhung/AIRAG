"""Multi-document comparison pilot through one adaptive complex planner (Phase 3, Task 3).

The complex-research boundary is a checkpointed LangGraph subgraph
(``build_complex_research_subgraph``) attached as the supervisor's
``complex_boundary`` node. It proposes an initial two-target ``TaskPlan``
through the framework-neutral compare skill policy, validates + leases +
checkpoints it, executes ONLY through the shared ``TaskScheduler``, and
evaluates through the shared ``evaluate_evidence``. Synthesis/grounding stay
outside the subgraph. Discovery/replan are rejected here (T5 adds them).
"""
from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

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
)
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.evaluation import (
    Contradiction,
    CoverageObservation,
)
from app.services.agents.v2.contracts.evidence import EvidenceUseRef
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.planning import CoverageCriterion
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.synthesis import AnswerDraft, AnswerClaim
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_answer_draft,
    validate_task_plan,
)
from app.services.agents.v2.nodes.evaluate import HydratedEvidence, evaluate_evidence

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")
REV_A = UUID("a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1")
REV_B = UUID("b2b2b2b2-b2b2-b2b2-b2b2-b2b2b2b2b2b2")
COMPARE_QUERY = "So sánh Chương II A và Chương III B"


class FakeReadCapability:
    """Atomic stub: shared document.read implementation for fast + complex."""

    def __init__(self, *, missing_targets: frozenset[str] = frozenset()) -> None:
        self.descriptor = CapabilityDescriptor(
            name="document.read",
            domain="document",
            operation_type="read",
            supports_parallel=True,
        )
        self.calls: list[tuple[AgentRequest, object]] = []
        self.missing_targets = missing_targets
        self.uses: dict[UUID, str] = {}

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        assert isinstance(request.input, DocumentReadInput)
        target_id = request.input.target_ids[0]
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
        self.uses[use_id] = target_id
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


class FakeSectionCapability:
    """Atomic stub: shared section.read implementation."""

    def __init__(self) -> None:
        self.descriptor = CapabilityDescriptor(
            name="section.read",
            domain="section",
            operation_type="read",
            supports_parallel=True,
        )
        self.calls: list[tuple[AgentRequest, object]] = []

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        raise AssertionError("compare pilot reads whole documents, not sections")


class FakeLeases:
    """Retention-lease double: records acquisitions, never releases."""

    def __init__(self) -> None:
        self.acquired: list[tuple[str, object, object]] = []
        self.released: list[str] = []
        self.session = self._Session(self)

    class _Session:
        def __init__(self, outer: "FakeLeases") -> None:
            self._outer = outer
            self.commits = 0

        async def commit(self) -> None:
            self.commits += 1

    async def acquire_or_refresh(
        self, run_id: str, revision_id: object = None, evidence_use_id: object = None
    ) -> object:
        self.acquired.append((run_id, revision_id, evidence_use_id))
        return {"run_id": run_id, "revision": revision_id, "use": evidence_use_id}

    async def release_run(self, run_id: str, reason: str = "terminal") -> int:
        self.released.append(run_id)
        return 0


class FakeHydrator:
    """Governed hydration double: admits current-run uses with pinned revisions."""

    def __init__(self, capability: FakeReadCapability) -> None:
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
        use_target = dict(self._capability.uses)
        admitted: list[HydratedEvidence] = []
        for ref in use_refs:
            target_id = use_target.get(ref.use_id)
            if target_id is None:
                continue
            binding = binding_by_target[target_id]
            task_id = next(
                task.task_id
                for task in plan.tasks
                if target_id in getattr(task.input, "target_ids", ())
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
                    classification="normal",
                    locator=DocumentLocator(kind="document"),
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
        raise AssertionError("compare pilot never persists derived summaries")


def _semantic(query: str = COMPARE_QUERY) -> SemanticContext:
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


def _analysis(work_type: str = "compare") -> QueryAnalysis:
    return QueryAnalysis(work_type=work_type, domains=("document",))  # type: ignore[arg-type]


def _capability_runtime(
    run_id: str = "run-compare-1",
    allowed: frozenset[str] | None = None,
) -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id="req-compare-1",
        run_id=run_id,
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=(
            allowed if allowed is not None else frozenset({"document.read", "section.read"})
        ),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


def _harness(
    run_id: str = "run-compare-1", missing: frozenset[str] = frozenset()
) -> tuple[FakeReadCapability, FakeLeases, GraphRuntimeContext]:
    capability = FakeReadCapability(missing_targets=missing)
    leases = FakeLeases()
    runtime = _capability_runtime(run_id)
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability), CapabilityRegistration(capability=FakeSectionCapability())],
        runtime,
    )
    context = GraphRuntimeContext(
        capability_runtime=runtime,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=leases,
            evidence_hydrator=FakeHydrator(capability),
        ),
    )
    return capability, leases, context


def _child_input() -> dict:
    from app.services.agents.v2.complex_research_graph import ComplexResearchState

    return ComplexResearchState(
        contract_version="2.0",
        semantic=_semantic(),
        bindings=_bindings(),
        query_analysis=_analysis(),
        plan=None,
        task_results=(),
        evaluation=None,
        replans_remaining=0,
    )


def _parent_state() -> SupervisorV2State:
    return SupervisorV2State(
        contract_version=CONTRACT_VERSION,
        request=RequestContext(
            contract_version=CONTRACT_VERSION,
            request_id="req-compare-1",
            thread_id="thread-compare-1",
            original_query=COMPARE_QUERY,
            known_documents=(),
        ),
        conversation=ConversationContext(
            summary="",
            active_entities=(),
            last_focus=None,
            recent_turns=(),
        ),
        semantic=_semantic(),
        bindings=_bindings(),
        query_analysis=_analysis(),
        route_decision=RouteDecision(route="complex_research", reason_code="comparison"),
        execution=ExecutionState(plan=None, task_results=(), evidence_evaluation=None),
        clarification=None,
        final_response=None,
    )


# ---------------------------------------------------------------------------
# Ownership: compare is a skill over shared capabilities, not an agent route
# ---------------------------------------------------------------------------


def test_compare_is_skill_not_agent_route() -> None:
    from app.services.agents.v2.skills.compare import policy as compare_policy

    assert callable(compare_policy.build_compare_plan)
    assert compare_policy.supports_work_type("compare") is True
    v2_root = Path(__file__).resolve().parents[5] / "app" / "services" / "agents" / "v2"
    banned = list(v2_root.glob("*comparison_agent*")) + list(v2_root.glob("*compare_agent*"))
    assert banned == []
    import app.services.agents.v2.complex_research_graph as subgraph_module

    source = inspect.getsource(subgraph_module)
    assert "ComparisonAgent" not in source
    assert "comparison_agent" not in source


def test_fast_and_complex_share_same_capability_instance_or_factory() -> None:
    capability, _, context = _harness()
    registry = context.services.capability_registry
    assert registry.get("document.read") is capability
    assert registry.get("section.read") is not capability
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    scheduler = TaskScheduler(registry)
    assert scheduler._registry.get("document.read") is capability  # type: ignore[attr-defined]


def test_complex_agent_uses_request_scoped_tool_catalog() -> None:
    from app.services.agents.v2.tools.adapters import (
        AgentToolAdapter,
        build_agent_tool_catalog,
    )

    _, _, context = _harness()
    registry = context.services.capability_registry
    catalog = build_agent_tool_catalog(registry)
    assert {entry.name for entry in catalog} == {"document.read", "section.read"}
    adapter = AgentToolAdapter(registry)
    assert adapter.visible_tool_names() == frozenset({"document.read", "section.read"})
    assert adapter.is_visible("people.lookup") is False

    narrowed = _capability_runtime(allowed=frozenset({"document.read"}))
    narrowed_registry = build_capability_registry(
        [CapabilityRegistration(capability=FakeReadCapability())], narrowed
    )
    assert {entry.name for entry in build_agent_tool_catalog(narrowed_registry)} == {
        "document.read"
    }


def test_complex_agent_cannot_execute_capability_without_scheduler() -> None:
    import app.services.agents.v2.complex_research_graph as subgraph_module
    import app.services.agents.v2.skills.compare.policy as policy_module

    for module in (subgraph_module, policy_module):
        source = inspect.getsource(module)
        assert "capability.execute(" not in source
        assert "registry.get(" not in source
        assert "TaskScheduler(" not in source
        assert "class TaskScheduler" not in source
    policy_source = inspect.getsource(policy_module)
    policy_imports = [
        line.strip()
        for line in policy_source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    assert not any("scheduler" in line.lower() for line in policy_imports)
    assert "TaskScheduler(" not in policy_source
    assert "shared_scheduler_for" in inspect.getsource(subgraph_module)


# ---------------------------------------------------------------------------
# Checkpointing: the subgraph inherits the supervisor saver
# ---------------------------------------------------------------------------


def test_complex_subgraph_does_not_open_its_own_checkpointer() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    subgraph = build_complex_research_subgraph()
    assert getattr(subgraph, "checkpointer", None) is None
    source = Path(
        "app/services/agents/v2/complex_research_graph.py"
    ).read_text()
    for forbidden in (
        "checkpointer=",
        "InMemorySaver",
        "PostgresSaver",
        "AsyncPostgresSaver",
    ):
        assert forbidden not in source, forbidden


def test_complex_subgraph_uses_shared_task_scheduler() -> None:
    import app.services.agents.v2.complex_research_graph as subgraph_module
    from app.services.agents.v2.execution import scheduler as scheduler_module

    source = inspect.getsource(subgraph_module)
    assert "TaskScheduler" in source
    assert "class TaskScheduler" not in source
    assert "shared_scheduler_for" in source
    assert hasattr(scheduler_module, "shared_scheduler_for")
    assert "execute_ready_tasks" in inspect.getsource(scheduler_module)


@pytest.mark.asyncio
async def test_complex_subgraph_is_checkpointed_under_supervisor_saver() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
        make_complex_boundary_node,
    )

    _, _, context = _harness()
    subgraph = build_complex_research_subgraph()
    parent = StateGraph(SupervisorV2State)
    parent.add_node("complex_boundary", make_complex_boundary_node(subgraph))
    parent.set_entry_point("complex_boundary")
    parent.set_finish_point("complex_boundary")
    saver = InMemorySaver()
    compiled = parent.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "thread-compare-1"}}

    await compiled.ainvoke(_parent_state(), config=config, context=context)

    snapshot = await compiled.aget_state(config)
    execution = snapshot.values["execution"]
    plan = execution.plan if hasattr(execution, "plan") else execution["plan"]
    tasks = plan.tasks if hasattr(plan, "tasks") else plan["tasks"]
    assert plan is not None
    assert len(tasks) == 2
    results = execution.task_results if hasattr(execution, "task_results") else execution["task_results"]
    assert len(results) == 2
    evaluation = (
        execution.evidence_evaluation
        if hasattr(execution, "evidence_evaluation")
        else execution["evidence_evaluation"]
    )
    assert evaluation is not None
    status = evaluation.status if hasattr(evaluation, "status") else evaluation["status"]
    assert status == "sufficient"


@pytest.mark.asyncio
async def test_complex_subgraph_resumes_from_interrupt() -> None:
    from langgraph.types import Command, interrupt

    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
        make_complex_boundary_node,
    )

    capability, leases, context = _harness()
    original_execute = capability.execute

    interrupted = {"raised": False}

    async def flaky_execute(request: AgentRequest, runtime: object) -> AgentResult:
        if not interrupted["raised"]:
            interrupted["raised"] = True
            interrupt("paused before first read")
        return await original_execute(request, runtime)

    capability.execute = flaky_execute  # type: ignore[method-assign]
    subgraph = build_complex_research_subgraph()
    parent = StateGraph(SupervisorV2State)
    parent.add_node("complex_boundary", make_complex_boundary_node(subgraph))
    parent.set_entry_point("complex_boundary")
    parent.set_finish_point("complex_boundary")
    compiled = parent.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "thread-compare-interrupt"}}

    suspended = await compiled.ainvoke(_parent_state(), config=config, context=context)
    assert interrupted["raised"] is True
    assert "__interrupt__" in suspended
    leases_after_interrupt = list(leases.acquired)
    assert leases_after_interrupt, "validate pinned the plan before the interrupt"
    assert leases.released == [], "interrupt keeps leases active (never releases)"

    resumed = await compiled.ainvoke(
        Command(resume="continue"), config=config, context=context
    )
    execution = resumed["execution"]
    assert len(execution.task_results) == 2
    assert execution.evidence_evaluation is not None
    assert execution.evidence_evaluation.status == "sufficient"
    assert len(leases.acquired) >= len(leases_after_interrupt), (
        "resume refreshes the retained leases"
    )
    assert leases.released == []


@pytest.mark.asyncio
async def test_complex_subgraph_resumes_under_supervisor_checkpointer() -> None:
    from app.services.agents.supervisor_v2 import create_supervisor_v2_graph

    compiled = create_supervisor_v2_graph(InMemorySaver())
    graph = compiled.get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("complex_boundary", "synthesize") in edges
    assert ("complex_boundary", "finalizer") in edges
    assert ("complex_boundary", "ground") not in edges


# ---------------------------------------------------------------------------
# Boundary adapter: explicit parent <-> child mapping, ephemeral planning input
# ---------------------------------------------------------------------------


def test_complex_boundary_maps_parent_to_child_state() -> None:
    from app.services.agents.v2.complex_research_graph import (
        MAX_REPLANS,
        build_complex_research_state,
    )

    child = build_complex_research_state(_parent_state())
    assert child["contract_version"] == "2.0"
    assert child["semantic"] == _semantic()
    assert child["bindings"] == _bindings()
    assert child["query_analysis"] == _analysis()
    assert child["plan"] is None
    assert child["task_results"] == ()
    assert child["evaluation"] is None
    assert child["replans_remaining"] == MAX_REPLANS
    assert "planning_input" not in child
    assert "ResearchPlanningInput" not in json.dumps(
        {key: str(value)[:64] for key, value in child.items()}
    )


def test_complex_boundary_maps_child_result_back_to_execution_state() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_state,
        merge_complex_result_into_supervisor,
    )
    from app.services.agents.v2.skills.compare.policy import build_compare_plan
    from app.services.agents.v2.complex_research_graph import build_planning_input

    _, _, context = _harness()
    child = build_complex_research_state(_parent_state())
    planning_input = build_planning_input(child, context)
    plan = build_compare_plan(planning_input)
    merged = merge_complex_result_into_supervisor(
        _parent_state(), {**child, "plan": plan, "task_results": ()}
    )
    assert set(merged.keys()) == {"execution"}
    execution = merged["execution"]
    assert isinstance(execution, ExecutionState)
    assert execution.plan == plan
    assert execution.task_results == ()
    assert execution.evidence_evaluation is None
    assert "evaluation" not in ExecutionState.model_fields


@pytest.mark.asyncio
async def test_complex_subgraph_does_not_require_root_state_schema() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    _, _, context = _harness()
    subgraph = build_complex_research_subgraph()
    output = await subgraph.ainvoke(_child_input(), context=context)
    assert output["plan"] is not None
    assert len(output["task_results"]) == 2
    assert output["evaluation"].status == "sufficient"


def test_planning_input_is_ephemeral_never_checkpointed() -> None:
    from app.services.agents.v2.complex_research_graph import build_planning_input

    _, _, context = _harness()
    planning_input = build_planning_input(_child_input(), context)
    assert planning_input.query_analysis == _analysis()
    assert len(planning_input.capability_catalog) == 2
    assert planning_input.current_plan is None

    import app.services.agents.v2.complex_research_graph as subgraph_module

    source = inspect.getsource(subgraph_module)
    assert '"planning_input"' not in source
    assert "planning_input=" not in source.replace("build_planning_input", "")


@pytest.mark.asyncio
async def test_planning_input_never_reaches_checkpoint_bytes() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
        make_complex_boundary_node,
    )

    _, _, context = _harness()
    parent = StateGraph(SupervisorV2State)
    parent.add_node(
        "complex_boundary",
        make_complex_boundary_node(build_complex_research_subgraph()),
    )
    parent.set_entry_point("complex_boundary")
    parent.set_finish_point("complex_boundary")
    compiled = parent.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "thread-compare-ephemeral"}}
    await compiled.ainvoke(_parent_state(), config=config, context=context)
    snapshot = await compiled.aget_state(config)
    payload = json.dumps(snapshot.values, default=str)
    assert "capability_catalog" not in payload
    assert "ResearchPlanningInput" not in payload
    assert "task_outcomes" not in payload


# ---------------------------------------------------------------------------
# Scope: legal/compliance excluded, unsupported types fail closed
# ---------------------------------------------------------------------------


def test_phase3_scope_excludes_legal_and_compliance_skills() -> None:
    skills_dir = Path(__file__).resolve().parents[5] / "app" / "services" / "agents" / "v2" / "skills"
    assert not (skills_dir / "legal_analysis").exists()
    assert not (skills_dir / "compliance").exists()
    with pytest.raises((ImportError, ModuleNotFoundError)):
        __import__("app.services.agents.v2.skills.legal_analysis.policy")
    with pytest.raises((ImportError, ModuleNotFoundError)):
        __import__("app.services.agents.v2.skills.compliance.policy")
    from app.services.agents.v2.skills.compare import policy as compare_policy

    assert compare_policy.supports_work_type("evaluate") is False


@pytest.mark.asyncio
async def test_unsupported_work_type_returns_typed_unavailable() -> None:
    from app.services.agents.v2.complex_research_graph import (
        COMPLEX_RESEARCH_UNAVAILABLE,
        build_complex_research_subgraph,
    )

    _, _, context = _harness()
    child = _child_input()
    child["query_analysis"] = _analysis("evaluate")
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    assert output["plan"] is None, "unsupported work never fabricates a plan"
    assert output["task_results"] == ()
    assert output["unavailable"] is not None
    assert output["unavailable"].code == COMPLEX_RESEARCH_UNAVAILABLE == (
        "COMPLEX_RESEARCH_UNAVAILABLE"
    )


# ---------------------------------------------------------------------------
# Supervisor routing after the complex subgraph (R4): both branches
# ---------------------------------------------------------------------------


def _branch_state(status: str | None):
    from app.services.agents.v2.contracts.evaluation import (
        Coverage,
        EvidenceEvaluation,
    )

    evaluation = (
        None
        if status is None
        else EvidenceEvaluation(
            status=status,  # type: ignore[arg-type]
            coverage=Coverage(items=()),
            missing=(),
            contradictions=(),
        )
    )
    state = _parent_state()
    state["execution"] = ExecutionState(
        plan=None, task_results=(), evidence_evaluation=evaluation
    )
    return state


def test_complex_boundary_routes_sufficient_to_synthesize() -> None:
    from app.services.agents.supervisor_v2 import _complex_branch

    assert _complex_branch(_branch_state("sufficient")) == "synthesize"


def test_complex_boundary_routes_other_to_finalizer() -> None:
    from app.services.agents.supervisor_v2 import _complex_branch

    for status in ("insufficient", "contradictory", "needs_input", None):
        assert _complex_branch(_branch_state(status)) == "finalizer", status  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Comparison scenarios: two-target bounded comparison over shared capabilities
# ---------------------------------------------------------------------------


def test_compare_plan_has_exact_two_target_ranges() -> None:
    from app.services.agents.v2.complex_research_graph import build_planning_input
    from app.services.agents.v2.skills.compare.policy import build_compare_plan

    _, _, context = _harness()
    plan = build_compare_plan(build_planning_input(_child_input(), context))
    validate_task_plan(plan, _bindings())
    assert len(plan.target_units) == 2
    assert {unit.target_id for unit in plan.target_units} == {"t1", "t2"}
    assert plan.target_units[0].binding_id == "b1"
    assert plan.target_units[1].binding_id == "b2"
    assert all(
        isinstance(unit.requested_locator, DocumentLocator)
        for unit in plan.target_units
    )
    assert len(plan.tasks) == 2
    assert [task.task_id for task in plan.tasks] == ["T1", "T2"]


def test_compare_plan_uses_parallel_safe_reads() -> None:
    from app.services.agents.v2.complex_research_graph import build_planning_input
    from app.services.agents.v2.skills.compare.policy import build_compare_plan

    _, _, context = _harness()
    plan = build_compare_plan(build_planning_input(_child_input(), context))
    assert all(task.depends_on == () for task in plan.tasks)
    assert {task.capability for task in plan.tasks} == {"document.read"}
    catalog = {entry.name: entry for entry in context.services.capability_registry.catalog()}
    assert catalog["document.read"].supports_parallel is True


def test_compare_plan_assigns_target_and_reference_roles() -> None:
    from app.services.agents.v2.complex_research_graph import build_planning_input
    from app.services.agents.v2.skills.compare.policy import build_compare_plan

    _, _, context = _harness()
    plan = build_compare_plan(build_planning_input(_child_input(), context))
    roles = {
        next(
            binding.role
            for binding in _bindings().bindings
            if binding.binding_id == unit.binding_id
        )
        for unit in plan.target_units
    }
    assert roles == {"target", "reference"}


def test_compare_plan_rejects_non_target_reference_roles() -> None:
    from app.services.agents.v2.complex_research_graph import build_planning_input
    from app.services.agents.v2.skills.compare.policy import build_compare_plan

    _, _, context = _harness()
    child = _child_input()
    child["bindings"] = DocumentBindingSet(
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
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )
    with pytest.raises(ContractValidationError):
        build_compare_plan(build_planning_input(child, context))


@pytest.mark.asyncio
async def test_compare_complete_coverage_is_sufficient() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    capability, _, context = _harness()
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(), context=context
    )
    assert len(capability.calls) == 2
    for request, runtime in capability.calls:
        assert not isinstance(runtime, dict)
        assert hasattr(runtime, "user_id")
    evaluation = output["evaluation"]
    assert evaluation.status == "sufficient"
    assert evaluation.missing == ()
    assert {
        (item.target_id, item.status) for item in evaluation.coverage.items
    } == {("t1", "read_complete"), ("t2", "read_complete")}


@pytest.mark.asyncio
async def test_compare_contradictory_evidence_is_reported() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    class TwoSidedJudge:
        async def assess_criterion(self, *, criterion, evidence) -> bool:
            return True

        async def detect_contradictions(self, *, evidence):
            uses = [item.use_id for item in evidence]
            assert len(uses) >= 2
            return (
                Contradiction(
                    contradiction_id="c1",
                    claim_a="A says X",
                    claim_b="B says not-X",
                    evidence_use_ids=tuple(uses[:2]),
                ),
            )

    _, _, context = _harness()
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(), context=context
    )
    use_refs = [
        ref for result in output["task_results"] for ref in result.evidence_uses
    ]
    assert len(use_refs) == 2
    evaluation = await evaluate_evidence(
        plan=output["plan"],
        bindings=_bindings(),
        results=output["task_results"],
        semantic=_semantic(),
        runtime=context,
        semantic_judge=TwoSidedJudge(),  # type: ignore[arg-type]
    )
    assert evaluation.status == "contradictory"
    assert len(evaluation.contradictions) == 1


@pytest.mark.asyncio
async def test_compare_insufficient_one_sided_coverage() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    _, _, context = _harness(missing=frozenset({"t2"}))
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(), context=context
    )
    evaluation = output["evaluation"]
    assert evaluation.status == "insufficient"
    assert any(
        requirement.target_id == "t2" for requirement in evaluation.missing
    )


@pytest.mark.asyncio
async def test_compare_claims_are_grounded_and_use_bound() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    _, _, context = _harness()
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(), context=context
    )
    assert output["evaluation"].status == "sufficient"
    admitted = frozenset(
        ref.use_id for result in output["task_results"] for ref in result.evidence_uses
    )
    assert len(admitted) == 2
    draft = AnswerDraft(
        content="A and B differ.",
        claims=(
            AnswerClaim(
                claim_id="claim-1",
                text="A and B differ.",
                evidence_use_ids=tuple(admitted),
            ),
        ),
    )
    validate_answer_draft(draft, admitted)
    with pytest.raises(ContractValidationError):
        validate_answer_draft(
            AnswerDraft(
                content="Discovery claim.",
                claims=(
                    AnswerClaim(
                        claim_id="claim-2",
                        text="Discovery claim.",
                        evidence_use_ids=(uuid4(),),
                    ),
                ),
            ),
            admitted,
        )

"""Multi-document comparison pilot through one adaptive complex planner (Phase 3, Task 3).

The complex-research boundary is a checkpointed LangGraph subgraph
(``build_complex_research_subgraph``) attached as the supervisor's
``complex_boundary`` node. It proposes an initial two-target ``TaskPlan``
through the framework-neutral compare skill policy, validates + leases +
checkpoints it, executes ONLY through the shared ``TaskScheduler``, and
evaluates through the shared ``evaluate_evidence``. Synthesis/grounding stay
outside the subgraph. Discovery/replan are rejected here (T5 adds them).

Ownership sequence (R17): the ``plan`` node computes the proposal into
ephemeral run-scoped scratch — never into checkpointed state — and the
``validate_checkpoint`` node validates, leases, and only then persists it, so
no checkpoint ever carries an unvalidated or unleased plan.
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
    SectionReadInput,
    SectionReadOutput,
)
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.evaluation import (
    Contradiction,
    CoverageObservation,
)
from app.services.agents.v2.contracts.evidence import DocumentSourceIdentity, EvidenceUseRef
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
from app.services.agents.v2.contracts.locators import (
    ContentLocator,
    DocumentLocator,
    SectionLocator,
)
from app.services.agents.v2.contracts.planning import CoverageCriterion
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import SectionReference, SemanticContext
from app.services.agents.v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.synthesis import (
    AnswerDraft,
    AnswerClaim,
    SynthesisEvidence,
)
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
    """Atomic stub: shared read implementation for fast + complex.

    Serves ``document.read`` or ``section.read`` (by registration name) with
    the exact requested coordinate per target, so coverage proves the planned
    locator rather than a hardcoded one.
    """

    def __init__(
        self,
        name: str = "document.read",
        *,
        missing_targets: frozenset[str] = frozenset(),
        locator_for: dict[str, ContentLocator] | None = None,
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
        self._locator_for = locator_for or {}
        self.uses: dict[UUID, str] = {}

    def _locator(self, target_id: str) -> ContentLocator:
        return self._locator_for.get(target_id, DocumentLocator(kind="document"))

    def _output(self, read_unit_count: int) -> object:
        if self.descriptor.name == "section.read":
            return SectionReadOutput(
                kind="section.read", read_unit_count=read_unit_count
            )
        return DocumentReadOutput(
            kind="document.read", read_unit_count=read_unit_count
        )

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        assert isinstance(request.input, (DocumentReadInput, SectionReadInput))
        target_id = request.input.target_ids[0]
        locator = self._locator(target_id)
        if target_id in self.missing_targets:
            return AgentResult(
                contract_version="2.0",
                task_id=request.task_id,
                status="success",
                data=self._output(0),  # type: ignore[arg-type]
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
            data=self._output(1),  # type: ignore[arg-type]
            evidence_uses=(EvidenceUseRef(use_id=use_id),),
            coverage_observations=(
                CoverageObservation(
                    target_id=target_id,
                    observed_locators=(locator,),
                    outcome="read",
                ),
            ),
            error=None,
        )


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
    """Governed hydration double: admits current-run uses with pinned revisions.

    The admitted locator is the plan's own requested coordinate per target, so
    only reads of the exact planned range complete coverage.
    """

    def __init__(self, capabilities: tuple[FakeReadCapability, ...]) -> None:
        self._capabilities = capabilities

    def _use_targets(self) -> dict[UUID, str]:
        targets: dict[UUID, str] = {}
        for capability in self._capabilities:
            targets.update(capability.uses)
        return targets

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
        use_targets = self._use_targets()
        admitted: list[HydratedEvidence] = []
        for ref in use_refs:
            target_id = use_targets.get(ref.use_id)
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


def _section_semantic() -> SemanticContext:
    """Chapter/section query: each side names an exact section coordinate."""
    return SemanticContext(
        contextualized_query=COMPARE_QUERY,
        normalized_query=COMPARE_QUERY.lower(),
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(
            SectionReference(ref_id="r1", label="Chương II", structure_node_id="chap-II"),
            SectionReference(ref_id="r2", label="Chương III", structure_node_id="chap-III"),
        ),
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


def _section_bindings() -> DocumentBindingSet:
    """Convention-shaped pins (`b_<ref_id>`) so section refs join their side."""
    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b_r1",
                document_id=DOC_A,
                document_revision=str(REV_A),
                role="target",
            ),
            ScopedDocument(
                binding_id="b_r2",
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
    run_id: str = "run-compare-1",
    missing: frozenset[str] = frozenset(),
    allowed: frozenset[str] | None = None,
    locator_for: dict[str, ContentLocator] | None = None,
) -> tuple[tuple[FakeReadCapability, ...], FakeLeases, GraphRuntimeContext]:
    document_capability = FakeReadCapability(
        "document.read", missing_targets=missing, locator_for=locator_for
    )
    section_capability = FakeReadCapability(
        "section.read", missing_targets=missing, locator_for=locator_for
    )
    capabilities = (document_capability, section_capability)
    leases = FakeLeases()
    runtime = _capability_runtime(run_id, allowed)
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability) for capability in capabilities],
        runtime,
    )
    from app.services.agent.runtime_selector import PlanBindingResolver

    context = GraphRuntimeContext(
        capability_runtime=runtime,
        services=RuntimeServices(
            capability_registry=registry,
            retention_leases=leases,
            evidence_hydrator=FakeHydrator(capabilities),
            # Production ingress wires the request-scoped resolver the
            # shared scheduler feeds before dispatch (round 2 N1:
            # targeted read plans raise without one).
            pinned_target_resolver=PlanBindingResolver(),
        ),
    )
    return capabilities, leases, context


def _child_input() -> dict:
    from app.services.agents.v2.complex_research_graph import ComplexResearchState

    return ComplexResearchState(
        contract_version="2.0",
        semantic=_semantic(),
        bindings=_bindings(),
        query_analysis=_analysis(),
        # Task 7B fix (R73): the execute node derives the post-router
        # v1-fallback guard from the resolved routing — child inputs carry
        # the same supported verdict production threads from the router.
        route_decision=RouteDecision(
            route="complex_research", reason_code="comparison"
        ),
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


@pytest.mark.asyncio
async def test_fast_and_complex_share_same_capability_instance_or_factory() -> None:
    """Fast and complex paths dispatch through the SAME registry instance.

    Drives the real seams: the deterministic fast-plan builder, the shared
    scheduler constructor, and the full complex subgraph — all against one
    request-scoped registry — then asserts one capability instance served all
    three tasks.
    """
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )
    from app.services.agents.v2.execution.scheduler import shared_scheduler_for
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    (document_capability, _), _, context = _harness()
    registry = context.services.capability_registry

    fast_semantic = SemanticContext(
        contextualized_query="Xem A",
        normalized_query="xem a",
        abbreviations=(),
        coreferences=(),
        document_refs=(
            _resolved_doc_ref("r1", DOC_A),
        ),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    fast_bindings = DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b_r1",
                document_id=DOC_A,
                document_revision=str(REV_A),
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )
    fast_plan = build_fast_plan(
        fast_semantic,
        fast_bindings,
        _analysis("retrieve"),
        RouteDecision(route="fast_domain", reason_code="exact_document_metadata"),
    )
    fast_report = await shared_scheduler_for(context).execute(
        plan=fast_plan, runtime=context, bindings=fast_bindings
    )
    assert len(fast_report.results) == 1

    output = await build_complex_research_subgraph().ainvoke(
        _child_input(), context=context
    )
    assert output["evaluation"].status == "sufficient"

    assert registry.get("document.read") is document_capability
    served = [call[0].task_id for call in document_capability.calls]
    assert fast_report.results[0].task_id in served
    assert {"T1", "T2"} <= set(served)


def _resolved_doc_ref(ref_id: str, document_id: UUID):  # type: ignore[no-untyped-def]
    from app.services.agents.v2.contracts.semantic import DocumentReference

    return DocumentReference(
        ref_id=ref_id,
        original_span=f"tài liệu {ref_id}",
        normalized_reference=f"tài liệu {ref_id}",
        requested_role="target",
        revision_requirement=None,
        resolution_status="resolved",
        resolved_document_id=document_id,
        candidate_document_ids=(),
    )


def test_complex_agent_uses_request_scoped_tool_catalog() -> None:
    """The planner's catalog IS the narrowed runtime catalog (real seam)."""
    from app.services.agents.v2.complex_research_graph import build_planning_input

    _, _, context = _harness()
    registry = context.services.capability_registry
    planning_input = build_planning_input(_child_input(), context)
    assert planning_input.capability_catalog == registry.catalog()
    assert {entry.name for entry in planning_input.capability_catalog} == {
        "document.read",
        "section.read",
    }

    narrowed_allowed = frozenset({"document.read"})
    (narrowed_doc, _), _, narrowed_context = _harness(
        run_id="run-compare-narrowed", allowed=narrowed_allowed
    )
    narrowed_registry = narrowed_context.services.capability_registry
    assert narrowed_registry.get("document.read") is narrowed_doc
    narrowed_input = build_planning_input(_child_input(), narrowed_context)
    assert narrowed_input.capability_catalog == narrowed_registry.catalog()
    assert {entry.name for entry in narrowed_input.capability_catalog} == {
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
# Checkpointing: the subgraph inherits the supervisor saver (R17)
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
async def test_plan_node_writes_no_state() -> None:
    """R21 single-owner: `plan` is a marker; the proposal lives in validate."""
    from app.services.agents.v2.complex_research_graph import (
        plan_node,
        validate_checkpoint_node,
    )

    _, leases, context = _harness(run_id="run-plan-marker")
    assert await plan_node(_child_input(), context) == {}
    decided = await validate_checkpoint_node(_child_input(), context)
    assert [task.task_id for task in decided["plan"].tasks] == ["T1", "T2"]
    assert leases.acquired, "validate leases the pins before checkpointing the plan"
    assert leases.session.commits >= 1


@pytest.mark.asyncio
async def test_unsupported_request_never_consumes_prior_proposal() -> None:
    """R21 regression (reviewer reproduction, exact): same run_id, no leak.

    compare `plan_node` → unsupported `plan_node` (SAME run_id/bindings) →
    `validate_checkpoint_node` for the unsupported request must yield no plan
    and zero leases, and `decide` must return typed unavailable.
    """
    from app.services.agents.v2.complex_research_graph import (
        decide_node,
        plan_node,
        validate_checkpoint_node,
    )

    _, leases, context = _harness(run_id="run-r21-same")
    await plan_node(_child_input(), context)
    unsupported = _child_input()
    # Task 12 covers evaluate deterministically; the unsupported fixture
    # uses explain (reachable at the complex boundary via
    # multi_document_research, unlike fast-path lookup).
    unsupported["query_analysis"] = _analysis("explain")
    await plan_node(unsupported, context)
    update = await validate_checkpoint_node(unsupported, context)
    assert "plan" not in update
    assert leases.acquired == []
    assert leases.session.commits == 0
    decision = await decide_node(unsupported, context)
    assert decision["unavailable"].code == "COMPLEX_RESEARCH_UNAVAILABLE"


@pytest.mark.asyncio
async def test_concurrent_sequences_share_no_proposal_state() -> None:
    """R21 isolation: interleaved and concurrent runs cannot exchange plans."""
    import asyncio

    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
        validate_checkpoint_node,
    )

    _, leases, context = _harness(run_id="run-r21-shared")
    supported = _child_input()
    unsupported = _child_input()
    # Task 12 covers evaluate deterministically; the unsupported fixture
    # uses explain (reachable at the complex boundary, no skill policy).
    unsupported["query_analysis"] = _analysis("explain")
    first = await validate_checkpoint_node(supported, context)
    second = await validate_checkpoint_node(unsupported, context)
    third = await validate_checkpoint_node(supported, context)
    assert first["plan"] is not None and third["plan"] is not None
    assert "plan" not in second
    assert first["plan"].goal == COMPARE_QUERY

    subgraph = build_complex_research_subgraph()
    compare_out, unsupported_out = await asyncio.gather(
        subgraph.ainvoke(_child_input(), context=context),
        subgraph.ainvoke(unsupported, context=context),
    )
    assert compare_out["evaluation"].status == "sufficient"
    assert unsupported_out["plan"] is None
    assert unsupported_out["unavailable"].code == "COMPLEX_RESEARCH_UNAVAILABLE"
    assert unsupported_out["task_results"] == ()


@pytest.mark.asyncio
async def test_no_checkpoint_carries_plan_before_lease_commit() -> None:
    """R17 saver audit: no inner checkpoint holds an unvalidated/unleased plan.

    Reuses the reviewer's method: run the boundary under a parent saver, list
    the subgraph's inner checkpoints, and assert every checkpoint predating
    the first lease acquisition carries no plan — the first checkpoint with a
    plan must come from `validate_checkpoint`, after the lease commit.
    """
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
        make_complex_boundary_node,
    )

    _, leases, context = _harness(run_id="run-audit-1")
    parent = StateGraph(SupervisorV2State)
    parent.add_node(
        "complex_boundary",
        make_complex_boundary_node(build_complex_research_subgraph()),
    )
    parent.set_entry_point("complex_boundary")
    parent.set_finish_point("complex_boundary")
    saver = InMemorySaver()
    compiled = parent.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "thread-audit-1"}}
    await compiled.ainvoke(_parent_state(), config=config, context=context)

    inner: list[tuple[str, dict]] = []
    async for tup in saver.alist(config):
        ns = tup.config["configurable"].get("checkpoint_ns", "")
        if not ns:
            continue
        inner.append((ns, dict(tup.checkpoint["channel_values"])))
    assert inner, "the subgraph wrote inner checkpoints under the parent saver"
    # `alist` yields newest-first; audit oldest-first.
    inner.reverse()
    first_plan_at = next(
        (index for index, (_, values) in enumerate(inner) if values.get("plan") is not None),
        None,
    )
    assert first_plan_at is not None, "validate_checkpoint persisted the plan"
    assert first_plan_at > 0, "the plan-node checkpoint must carry no plan"
    for _, values in inner[:first_plan_at]:
        assert values.get("plan") is None
    assert leases.acquired, "leases were committed before the first plan checkpoint"
    assert leases.session.commits >= 1


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
    """Interrupt keeps leases; resume skips planning and strictly refreshes."""
    from langgraph.types import Command, interrupt

    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
        make_complex_boundary_node,
    )
    import app.services.agents.v2.skills.compare.policy as policy_module

    (document_capability, _), leases, context = _harness(run_id="run-resume-1")
    original_execute = document_capability.execute
    original_plan = policy_module.build_compare_plan
    planner_calls = {"count": 0}

    def counting_plan(planning_input):  # type: ignore[no-untyped-def]
        planner_calls["count"] += 1
        return original_plan(planning_input)

    interrupted = {"raised": False}

    async def flaky_execute(request: AgentRequest, runtime: object) -> AgentResult:
        if not interrupted["raised"]:
            interrupted["raised"] = True
            interrupt("paused before first read")
        return await original_execute(request, runtime)

    document_capability.execute = flaky_execute  # type: ignore[method-assign]
    policy_module.build_compare_plan = counting_plan  # type: ignore[method-assign]
    try:
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
        assert planner_calls["count"] == 1
        leases_after_interrupt = list(leases.acquired)
        assert leases_after_interrupt, "validate pinned the plan before the interrupt"
        assert leases.released == [], "interrupt keeps leases active (never releases)"
        commits_after_interrupt = leases.session.commits

        resumed = await compiled.ainvoke(
            Command(resume="continue"), config=config, context=context
        )
        execution = resumed["execution"]
        assert len(execution.task_results) == 2
        assert execution.evidence_evaluation is not None
        assert execution.evidence_evaluation.status == "sufficient"
        assert planner_calls["count"] == 1, "resume must not recompute the plan"
        assert len(leases.acquired) > len(leases_after_interrupt), (
            "resume strictly refreshes the retained leases"
        )
        # R22: the SAME revision-only pins from before the interrupt were
        # re-acquired after resume (pair identity, not just a larger count).
        pins_pre = {(revision, use) for (_, revision, use) in leases_after_interrupt}
        pins_post = {
            (revision, use) for (_, revision, use) in leases.acquired[len(leases_after_interrupt):]
        }
        assert pins_pre and pins_pre <= pins_post
        assert leases.session.commits > commits_after_interrupt
        assert leases.released == []
        assert {run for run, _, _ in leases.acquired} == {"run-resume-1"}
    finally:
        policy_module.build_compare_plan = original_plan  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_resume_refreshes_pre_interrupt_lease_pairs() -> None:
    """R22: interrupt AFTER both reads exist; resume re-acquires SAME pairs.

    The interrupt fires inside `evaluate`, so both evidence uses (and their
    revision leases) already exist in checkpointed results. The resumed
    `evaluate` refreshes the identical `(revision_id, evidence_use_id)`
    tuples (pair identity with fresh acquisitions) — not merely a larger
    total from new-use leasing.
    """
    from langgraph.types import Command, interrupt

    import app.services.agents.v2.complex_research_graph as cx_module
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
        make_complex_boundary_node,
    )

    (document_capability, _), leases, context = _harness(run_id="run-r22-pairs")
    original_evaluate = cx_module.evaluate_evidence
    struck = {"interrupted": False}

    async def flaky_evaluate(**kwargs):  # type: ignore[no-untyped-def]
        if not struck["interrupted"]:
            struck["interrupted"] = True
            interrupt("paused after reads")
        return await original_evaluate(**kwargs)

    cx_module.evaluate_evidence = flaky_evaluate  # type: ignore[method-assign]
    try:
        subgraph = build_complex_research_subgraph()
        parent = StateGraph(SupervisorV2State)
        parent.add_node("complex_boundary", make_complex_boundary_node(subgraph))
        parent.set_entry_point("complex_boundary")
        parent.set_finish_point("complex_boundary")
        compiled = parent.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "thread-r22-pairs"}}

        suspended = await compiled.ainvoke(_parent_state(), config=config, context=context)
        assert struck["interrupted"] is True
        assert "__interrupt__" in suspended
        pre = list(leases.acquired)
        use_ids = set(document_capability.uses.keys())
        assert len(use_ids) == 2, "both reads dispatched and leased pre-interrupt"
        pins_pre = {(revision, use) for (_, revision, use) in pre}
        assert (REV_A, None) in pins_pre and (REV_B, None) in pins_pre
        assert use_ids <= {use for (_, _, use) in pre if use is not None}, (
            "both dispatched uses were leased before the interrupt"
        )

        resumed = await compiled.ainvoke(
            Command(resume="continue"), config=config, context=context
        )
        execution = resumed["execution"]
        assert execution.evidence_evaluation.status == "sufficient"
        post = leases.acquired[len(pre):]
        assert post, "resume re-acquired leases (not a silent no-op)"
        pins_post = {(revision, use) for (_, revision, use) in post}
        assert pins_pre <= pins_post, (
            "every pre-interrupt pair was re-acquired after resume"
        )
        for pair in pins_pre:
            total_before = sum(1 for entry in pre if (entry[1], entry[2]) == pair)
            total_after = sum(1 for entry in leases.acquired if (entry[1], entry[2]) == pair)
            assert total_after > total_before, f"pair {pair} was refreshed, not just kept"
        assert leases.released == []
        assert {run for run, _, _ in leases.acquired} == {"run-r22-pairs"}
    finally:
        cx_module.evaluate_evidence = original_evaluate  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_execute_without_plan_touches_no_leases() -> None:
    """R22 explicit no-op: plan-less execute returns empty and leases nothing."""
    from app.services.agents.v2.complex_research_graph import complex_execute_node

    _, leases, context = _harness(run_id="run-noplan")
    child = _child_input()
    child["query_analysis"] = _analysis("evaluate")
    assert await complex_execute_node(child, context) == {}
    assert leases.acquired == []
    assert leases.session.commits == 0


@pytest.mark.asyncio
async def test_complex_subgraph_resumes_under_supervisor_checkpointer() -> None:
    """R19: the REAL supervisor graph interrupts at the boundary and resumes.

    Compiles `create_supervisor_v2_graph` against a saver, drives a compare
    turn to a mid-execution interrupt, resumes with `Command(resume=...)`,
    and asserts the terminal grounded success — plus the static R4 wiring.
    """
    from langgraph.types import Command, interrupt

    from app.services.agents.supervisor_v2 import create_supervisor_v2_graph
    from app.services.agents.v2.contracts.semantic import (
        DocumentReference,
        SemanticDraft,
    )
    from app.services.agents.v2.nodes.evaluate import AnswerDraftChannel

    compiled = create_supervisor_v2_graph(InMemorySaver())
    graph = compiled.get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("complex_boundary", "synthesize") in edges
    assert ("complex_boundary", "finalizer") in edges
    assert ("complex_boundary", "ground") not in edges

    draft = SemanticDraft(
        provisional_contextualized_query=COMPARE_QUERY,
        abbreviations=(),
        coreferences=(),
        document_refs=(
            DocumentReference(
                ref_id="r1",
                original_span="Chương II A",
                normalized_reference="chuong ii a",
                requested_role="target",
                revision_requirement=None,
                resolution_status="resolved",
                resolved_document_id=DOC_A,
                candidate_document_ids=(),
            ),
            DocumentReference(
                ref_id="r2",
                original_span="Chương III B",
                normalized_reference="chuong iii b",
                requested_role="reference",
                revision_requirement=None,
                resolution_status="resolved",
                resolved_document_id=DOC_B,
                candidate_document_ids=(),
            ),
        ),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=(),
    )
    pins = DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b_r1",
                document_id=DOC_A,
                document_revision=str(REV_A),
                role="target",
            ),
            ScopedDocument(
                binding_id="b_r2",
                document_id=DOC_B,
                document_revision=str(REV_B),
                role="reference",
            ),
        ),
        revision_requirement_refs=(),
    )

    (document_capability, _section_capability), leases, context = _harness(
        run_id="run-e2e-1", allowed=frozenset({"document.read"})
    )
    original_execute = document_capability.execute
    interrupted = {"raised": False}

    async def flaky_execute(request: AgentRequest, runtime: object) -> AgentResult:
        if not interrupted["raised"]:
            interrupted["raised"] = True
            interrupt("paused before first read")
        return await original_execute(request, runtime)

    document_capability.execute = flaky_execute  # type: ignore[method-assign]

    class _Adapter:
        def __init__(self, draft: SemanticDraft) -> None:
            self._draft = draft

        async def build_draft(self, request: object, conversation: object) -> SemanticDraft:
            return self._draft

    class _Resolver:
        def __init__(self, binding_set: DocumentBindingSet) -> None:
            self._binding_set = binding_set

        async def resolve(self, document_refs: object, capability_runtime: object) -> DocumentBindingSet:
            return self._binding_set

    context.services.semantic_adapter = _Adapter(draft)
    context.services.binding_resolver = _Resolver(pins)
    context.services.answer_draft_channel = AnswerDraftChannel()

    from app.services.agents.supervisor_v2 import build_initial_v2_state

    initial = build_initial_v2_state(
        request=RequestContext(
            contract_version=CONTRACT_VERSION,
            request_id="req-e2e-1",
            thread_id="thread-e2e-1",
            original_query=COMPARE_QUERY,
            known_documents=(),
        )
    )
    e2e_config = {"configurable": {"thread_id": "thread-e2e-1"}}
    suspended = await compiled.ainvoke(initial, e2e_config, context=context)
    assert interrupted["raised"] is True
    assert "__interrupt__" in suspended

    terminal = await compiled.ainvoke(
        Command(resume="continue"), e2e_config, context=context
    )
    final = terminal["final_response"]
    status = final.status if hasattr(final, "status") else final["status"]
    assert status == "success"
    content = final.content if hasattr(final, "content") else final["content"]
    assert content
    citations = final.citations if hasattr(final, "citations") else final["citations"]
    assert len(citations) == 2
    assert leases.released == []


# ---------------------------------------------------------------------------
# Boundary adapter: explicit parent <-> child mapping, ephemeral planning input
# ---------------------------------------------------------------------------


def test_complex_boundary_maps_parent_to_child_state(monkeypatch) -> None:
    from types import SimpleNamespace

    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_state,
    )

    # R36: production entry uses the SAME settings-driven limits the budget
    # view enforces — no hardcoded constant anywhere.
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(
            V2_MAX_TASKS=8, V2_MAX_PARALLEL_BRANCHES=2, V2_MAX_REPLANS=3
        ),
    )
    child = build_complex_research_state(_parent_state())
    assert child["contract_version"] == "2.0"
    assert child["semantic"] == _semantic()
    assert child["bindings"] == _bindings()
    assert child["query_analysis"] == _analysis()
    assert child["plan"] is None
    assert child["task_results"] == ()
    assert child["evaluation"] is None
    assert child["replans_remaining"] == 3
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


def test_research_budget_derives_from_settings_and_execution_state(monkeypatch) -> None:
    """R20: limits come from implementation settings, consumed by live state."""
    from types import SimpleNamespace

    from app.services.agents.v2.complex_research_graph import (
        V2ResearchLimits,
        build_planning_input,
        build_research_budget_view,
    )
    from app.services.agents.v2.skills.compare.policy import build_compare_plan

    _, _, context = _harness()
    child = _child_input()
    assert V2ResearchLimits.from_settings(SimpleNamespace()) == V2ResearchLimits(
        max_tasks=8, max_parallel_branches=2, max_replans=0
    )
    assert build_research_budget_view(child, context).max_tasks_remaining == 8

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: SimpleNamespace(V2_MAX_TASKS=3, V2_MAX_PARALLEL_BRANCHES=1),
    )
    limited = build_research_budget_view(child, context)
    assert limited.max_tasks_remaining == 3
    assert limited.max_parallel_branches == 1
    # Consumed tasks shrink the remainder: a 2-task plan leaves 1.
    plan = build_compare_plan(build_planning_input(child, context))
    used = build_research_budget_view({**child, "plan": plan}, context)
    assert used.max_tasks_remaining == 1


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
    # Task 12 covers evaluate deterministically; the unsupported fixture
    # uses explain (reachable at the complex boundary, no skill policy).
    child["query_analysis"] = _analysis("explain")
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
    """Whole-document fallback: exact full ranges, one read per side."""
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
    assert [task.capability for task in plan.tasks] == [
        "document.read",
        "document.read",
    ]


def test_compare_plan_emits_exact_section_coordinates() -> None:
    """R18: named chapters become exact section coordinates (not whole docs)."""
    from app.services.agents.v2.complex_research_graph import build_planning_input
    from app.services.agents.v2.skills.compare.policy import build_compare_plan

    _, _, context = _harness()
    child = _child_input()
    child["semantic"] = _section_semantic()
    child["bindings"] = _section_bindings()
    plan = build_compare_plan(build_planning_input(child, context))
    validate_task_plan(plan, _section_bindings())
    locators = {unit.target_id: unit.requested_locator for unit in plan.target_units}
    assert locators["t1"] == SectionLocator(kind="section", structure_node_id="chap-II")
    assert locators["t2"] == SectionLocator(kind="section", structure_node_id="chap-III")
    assert {task.capability for task in plan.tasks} == {"section.read"}
    assert plan.tasks[0].input.target_ids == ("t1",)
    assert plan.tasks[1].input.target_ids == ("t2",)


def test_compare_plan_mixes_section_and_document_sides() -> None:
    """Only the side with a named section reads by section; else whole doc."""
    from app.services.agents.v2.complex_research_graph import build_planning_input
    from app.services.agents.v2.skills.compare.policy import build_compare_plan

    _, _, context = _harness()
    child = _child_input()
    semantic = _section_semantic()
    child["semantic"] = semantic.model_copy(
        update={"section_refs": semantic.section_refs[:1]}
    )
    child["bindings"] = _section_bindings()
    plan = build_compare_plan(build_planning_input(child, context))
    validate_task_plan(plan, _section_bindings())
    locators = {unit.target_id: unit.requested_locator for unit in plan.target_units}
    assert locators["t1"] == SectionLocator(kind="section", structure_node_id="chap-II")
    assert isinstance(locators["t2"], DocumentLocator)
    assert [task.capability for task in plan.tasks] == [
        "section.read",
        "document.read",
    ]


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

    (document_capability, _), _, context = _harness()
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(), context=context
    )
    assert len(document_capability.calls) == 2
    for request, runtime in document_capability.calls:
        assert not isinstance(runtime, dict)
        assert hasattr(runtime, "user_id")
    evaluation = output["evaluation"]
    assert evaluation.status == "sufficient"
    assert evaluation.missing == ()
    assert {
        (item.target_id, item.status) for item in evaluation.coverage.items
    } == {("t1", "read_complete"), ("t2", "read_complete")}


@pytest.mark.asyncio
async def test_compare_section_run_covers_exact_ranges() -> None:
    """R18 integration: section reads of the exact coordinates cover fully."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )

    locator_for = {
        "t1": SectionLocator(kind="section", structure_node_id="chap-II"),
        "t2": SectionLocator(kind="section", structure_node_id="chap-III"),
    }
    (_, section_capability), _, context = _harness(
        run_id="run-compare-sections", locator_for=locator_for
    )
    child = _child_input()
    child["semantic"] = _section_semantic()
    child["bindings"] = _section_bindings()
    output = await build_complex_research_subgraph().ainvoke(child, context=context)
    assert len(section_capability.calls) == 2
    evaluation = output["evaluation"]
    assert evaluation.status == "sufficient"
    assert {
        (item.target_id, item.status) for item in evaluation.coverage.items
    } == {("t1", "read_complete"), ("t2", "read_complete")}


@pytest.mark.asyncio
async def test_compare_contradictory_evidence_is_reported() -> None:
    """Contradictions come from the shared evaluator and block synthesis (R4)."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )
    from app.services.agents.supervisor_v2 import _complex_branch

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
    merged = _parent_state()
    merged["execution"] = ExecutionState(
        plan=output["plan"],
        task_results=output["task_results"],
        evidence_evaluation=evaluation,
    )
    assert _complex_branch(merged) == "finalizer"


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
    """Real synthesis/grounding seam on real subgraph evidence (R20)."""
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_subgraph,
    )
    from app.services.agents.v2.nodes.grounding import ground_answer
    from app.services.agents.v2.nodes.synthesize import build_extractive_draft

    _, _, context = _harness()
    output = await build_complex_research_subgraph().ainvoke(
        _child_input(), context=context
    )
    assert output["evaluation"].status == "sufficient"
    admitted = frozenset(
        ref.use_id for result in output["task_results"] for ref in result.evidence_uses
    )
    assert len(admitted) == 2
    hydrator = context.services.evidence_hydrator
    hydrated = await hydrator.hydrate_for_evaluation(
        tuple(ref for result in output["task_results"] for ref in result.evidence_uses),
        runtime=context,
        plan=output["plan"],
        bindings=_bindings(),
    )
    assert len(hydrated) == 2
    synthesis_evidence = tuple(
        SynthesisEvidence(
            use_id=item.use_id,
            content=item.content,
            role=item.role,
            target_id=item.target_id,
            source_label=item.source_label,
        )
        for item in hydrated
    )
    draft = build_extractive_draft(synthesis_evidence)
    validate_answer_draft(draft, admitted)
    grounded = await ground_answer(draft=draft, evidence=hydrated)
    assert len(grounded.citations) == 2
    assert {citation.evidence_id for citation in grounded.citations} == {
        item.evidence_id for item in hydrated
    }
    validate_answer_draft(grounded.draft, admitted)
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

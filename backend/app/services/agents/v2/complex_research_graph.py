"""One adaptive complex-research boundary as a checkpointed subgraph (Phase 3, Task 3).

``ComplexResearchGraph`` is the ONLY adaptive planning/replanning boundary. It
is compiled WITHOUT its own checkpointer and attached as the supervisor's
``complex_boundary`` node, so the supervisor saver checkpoints and namespaces
every accepted plan/execute/evaluate step (a shadow run's isolated saver is
inherited the same way). Ownership is a single chain::

    Supervisor -> complex_boundary adapter -> ComplexResearch subgraph
        -> validate/lease/checkpoint -> shared TaskScheduler
        -> CapabilityRegistry -> Capability.execute

The subgraph never resolves or executes a capability itself: the ``execute``
node runs the shared ``TaskScheduler`` (built through the R5
``shared_scheduler_for`` seam); the ``evaluate`` node calls the shared
``evaluate_evidence(...)`` function. Synthesis/grounding stay OUTSIDE the
subgraph (supervisor nodes). No domain agents/subgraphs, no second scheduler,
no subgraph-owned checkpointer, no ``plan_checkpoint`` service, no
``EvidenceEvaluator`` service.

``ResearchPlanningInput`` is ephemeral: rebuilt from checkpointed state plus
the request-scoped registry on every planner call, never stored in
``ComplexResearchState`` or ``SupervisorV2State``.

Pilot scope is initial-plan-only plus the compare skill: discovery and replan
are rejected here (T5 adds them). An unsupported work type (``evaluate``,
``cross_domain``, ``multi_goal``, ...) returns the typed
``COMPLEX_RESEARCH_UNAVAILABLE`` marker from the ``decide`` node — never a
fabricated plan.
"""
from __future__ import annotations

import inspect
import json
import warnings
from collections.abc import Mapping
from typing import Any, Literal, TypedDict
from uuid import UUID

from pydantic import ValidationError

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from .contracts.base import ContractModel
from .contracts.binding import DocumentBindingSet
from .contracts.evaluation import EvidenceEvaluation
from .contracts.execution import AgentResult, TaskExecutionSummary
from .contracts.evidence import EvidenceUseRef
from .contracts.planning import (
    DiscoveryPolicy,
    ResearchBudgetView,
    ResearchPlanningInput,
    TaskPlan,
)
from .contracts.routing import QueryAnalysis
from .contracts.semantic import SemanticContext
from .contracts.state import GraphRuntimeContext, SupervisorV2State
from .contracts.validation import ContractValidationError, validate_task_plan
from .execution.scheduler import refresh_pairs_for_checkpoint, shared_scheduler_for
from .nodes.context import node_context
from .nodes.evaluate import evaluate_evidence
from .nodes.execute import execution_update
from .skills.compare import policy as compare_policy

__all__ = [
    "COMPLEX_RESEARCH_UNAVAILABLE",
    "MAX_REPLANS",
    "ComplexResearchError",
    "ComplexResearchState",
    "ComplexResearchUnavailable",
    "build_complex_research_state",
    "build_complex_research_subgraph",
    "build_discovery_policy",
    "build_planning_input",
    "build_research_budget_view",
    "build_task_execution_summaries",
    "collect_evidence_use_refs",
    "complex_evaluate_node",
    "complex_execute_node",
    "decide_node",
    "finalize_node",
    "make_complex_boundary_node",
    "merge_complex_result_into_supervisor",
    "normalize_complex_state",
    "plan_node",
    "require_query_analysis",
    "validate_checkpoint_node",
]

#: Typed-unavailable code for out-of-pilot work (evaluate/compliance, ...).
COMPLEX_RESEARCH_UNAVAILABLE = "COMPLEX_RESEARCH_UNAVAILABLE"

#: Initial-plan-only pilot: no replan budget is carried or consumed (T5 adds it).
MAX_REPLANS = 0

#: Ephemeral planner sizing (implementation config, not a runtime field).
MAX_TASKS = 8
MAX_PARALLEL_BRANCHES = 2


class ComplexResearchError(ValueError):
    """The complex boundary cannot proceed; nothing may be fabricated."""


class ComplexResearchUnavailable(ContractModel):
    """Typed unavailable marker for out-of-pilot work (not a plan)."""

    code: Literal["COMPLEX_RESEARCH_UNAVAILABLE"]
    reason: str


class ComplexResearchState(TypedDict, total=False):
    """Checkpointed subgraph state. It never stores ResearchPlanningInput."""

    contract_version: str
    semantic: SemanticContext
    bindings: DocumentBindingSet
    query_analysis: QueryAnalysis | None
    plan: TaskPlan | None
    task_results: tuple[AgentResult, ...]
    evaluation: EvidenceEvaluation | None
    replans_remaining: int
    unavailable: ComplexResearchUnavailable | None


def require_query_analysis(state: ComplexResearchState) -> QueryAnalysis:
    """Return the non-null query analysis before any complex planning."""
    analysis = state.get("query_analysis")
    if analysis is None:
        raise ComplexResearchError(
            "complex planning requires a non-null query_analysis; "
            "refusing to plan without one"
        )
    return analysis


def build_task_execution_summaries(
    results: tuple[AgentResult, ...],
) -> tuple[TaskExecutionSummary, ...]:
    """Ephemeral minimal projection of checkpointed results (never persisted).

    Keeps ``not_found`` distinguishable from denial/timeout/infrastructure
    failure for future replanning, mirroring the frozen outcome validation:
    an ``error`` always carries its code, a ``denied`` carries only
    ``PERMISSION_DENIED``, anything else carries no code.
    """
    summaries: list[TaskExecutionSummary] = []
    for result in results:
        if result.status == "error":
            code = result.error.code if result.error is not None else "INTERNAL_ERROR"
            summaries.append(
                TaskExecutionSummary(task_id=result.task_id, status=result.status, error_code=code)  # type: ignore[arg-type]
            )
        elif result.status == "denied":
            code = (
                result.error.code
                if result.error is not None and result.error.code == "PERMISSION_DENIED"
                else None
            )
            summaries.append(
                TaskExecutionSummary(task_id=result.task_id, status=result.status, error_code=code)  # type: ignore[arg-type]
            )
        else:
            summaries.append(
                TaskExecutionSummary(task_id=result.task_id, status=result.status)
            )
    return tuple(summaries)


def collect_evidence_use_refs(
    results: tuple[AgentResult, ...],
) -> tuple[EvidenceUseRef, ...]:
    """Ephemeral deduplicated use refs behind checkpointed results."""
    seen: set[UUID] = set()
    refs: list[EvidenceUseRef] = []
    for result in results:
        for ref in result.evidence_uses:
            if ref.use_id not in seen:
                seen.add(ref.use_id)
                refs.append(ref)
    return tuple(refs)


def build_discovery_policy(runtime: GraphRuntimeContext) -> DiscoveryPolicy:
    """Runtime-only policy: discovery is rejected for this pilot (T5 adds it)."""
    _ = runtime
    return DiscoveryPolicy(
        allow_reference_discovery=False,
        allow_supporting_discovery=False,
        max_discovered_documents=0,
    )


def build_research_budget_view(
    state: ComplexResearchState, runtime: GraphRuntimeContext
) -> ResearchBudgetView:
    """Ephemeral budget derived from the CURRENT execution state.

    Never persisted; rebuilt on each planner/replanner call. ``runtime`` is
    accepted for the T5 signature (deadline/parallelism source) and unused
    while the pilot sizes from implementation constants.
    """
    _ = runtime
    plan = state.get("plan")
    used_tasks = len(plan.tasks) if plan is not None else 0
    return ResearchBudgetView(
        max_tasks_remaining=max(0, MAX_TASKS - used_tasks),
        max_replans_remaining=max(0, min(state.get("replans_remaining", 0), MAX_REPLANS)),
        max_parallel_branches=MAX_PARALLEL_BRANCHES,
    )


def _capability_catalog(runtime: GraphRuntimeContext) -> tuple[Any, ...]:
    """Request-scoped planner catalog: the intersected registry view, or fail."""
    registry = runtime.services.capability_registry
    if registry is None:
        raise ComplexResearchError(
            "complex planning requires the request-scoped capability_registry; "
            "refusing to plan without the current runtime catalog"
        )
    return tuple(registry.catalog())


def build_planning_input(
    state: ComplexResearchState, runtime: GraphRuntimeContext
) -> ResearchPlanningInput:
    """Ephemeral projection, rebuilt on every planner/replanner call.

    Uses only frozen ``GraphRuntimeContext`` fields plus implementation
    policy/budget helpers. Never returned into checkpointed state.
    """
    results = tuple(state.get("task_results", ()))
    return ResearchPlanningInput(
        semantic=state["semantic"],
        bindings=state["bindings"],
        query_analysis=require_query_analysis(state),
        capability_catalog=_capability_catalog(runtime),
        discovery_policy=build_discovery_policy(runtime),
        budget=build_research_budget_view(state, runtime),
        current_plan=state.get("plan"),
        task_outcomes=build_task_execution_summaries(results),
        prior_evidence_uses=collect_evidence_use_refs(results),
        prior_evaluation=state.get("evaluation"),
    )


def _coerce_slot(value: Any, model: Any, *, slot: str) -> Any:
    """Return a fully live contract, re-validating checkpoint-serde mappings.

    Mirrors the supervisor's ``_coerce_slot``: live models are NEVER trusted
    as-is (checkpoint serde revives the envelope while leaving nested values
    as ``list``/``dict``), so every value round-trips through JSON
    validation to rebuild the full tree before any node walks it. Anything
    that cannot be re-validated fails closed.
    """
    if value is None:
        return None
    if isinstance(value, model):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return model.model_validate_json(value.model_dump_json())
        except (ValidationError, TypeError, ValueError) as exc:
            raise ComplexResearchError(
                f"complex checkpoint slot {slot!r} does not re-validate "
                f"as {model.__name__}: {exc}"
            ) from exc
    if isinstance(value, Mapping):
        payload = dict(value)
        try:
            candidate = model.model_validate(payload)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return model.model_validate_json(candidate.model_dump_json())
        except ValidationError:
            pass
        try:
            return model.model_validate_json(json.dumps(payload, default=str))
        except (ValidationError, TypeError, ValueError) as exc:
            raise ComplexResearchError(
                f"complex checkpoint slot {slot!r} is not a valid "
                f"{model.__name__}: {exc}"
            ) from exc
    raise ComplexResearchError(
        f"complex checkpoint slot {slot!r} is not a {model.__name__} "
        f"(got {type(value).__name__})"
    )


def normalize_complex_state(state: ComplexResearchState) -> ComplexResearchState:
    """Coerce a resumed checkpoint mapping back into usable contracts.

    The subgraph inherits the supervisor saver, so a resume re-enters
    mid-graph with serde-degraded nested contracts (mappings/lists where
    live models stood). This mirrors the supervisor's
    ``normalize_checkpoint_state``: re-validate every slot and return a NEW
    state; the stored checkpoint is never mutated. Full cross-reference
    validation stays with ``validate_checkpoint_node``.
    """
    raw_results = state.get("task_results", ())
    try:
        items = tuple(raw_results or ())
    except TypeError as exc:
        raise ComplexResearchError(
            "complex checkpoint task_results is not a sequence"
        ) from exc
    results = tuple(
        _coerce_slot(item, AgentResult, slot="task_results[]") for item in items
    )
    return ComplexResearchState(
        contract_version=str(state.get("contract_version", "2.0")),
        semantic=_coerce_slot(state.get("semantic"), SemanticContext, slot="semantic"),
        bindings=_coerce_slot(state.get("bindings"), DocumentBindingSet, slot="bindings"),
        query_analysis=_coerce_slot(
            state.get("query_analysis"), QueryAnalysis, slot="query_analysis"
        ),
        plan=_coerce_slot(state.get("plan"), TaskPlan, slot="plan"),
        task_results=results,
        evaluation=_coerce_slot(state.get("evaluation"), EvidenceEvaluation, slot="evaluation"),
        replans_remaining=int(state.get("replans_remaining", 0)),
        unavailable=_coerce_slot(
            state.get("unavailable"), ComplexResearchUnavailable, slot="unavailable"
        ),
    )


async def plan_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Propose the initial plan through the compare skill (no execution).

    Unsupported work types propose nothing: ``plan`` stays ``None`` so the
    ``decide`` node can return the typed unavailable boundary.
    """
    context = node_context(runtime)
    state = normalize_complex_state(state)
    analysis = require_query_analysis(state)
    if not compare_policy.supports_work_type(analysis.work_type):
        return {"plan": None}
    planning_input = build_planning_input(state, context)
    try:
        plan = compare_policy.build_compare_plan(planning_input)
    except ContractValidationError as exc:
        raise ComplexResearchError(
            f"compare planning failed closed: {exc}"
        ) from exc
    return {"plan": plan}


async def _commit_lease_session(runtime: GraphRuntimeContext) -> None:
    """Commit the dedicated lease unit of work before the checkpointable update."""
    repo = runtime.services.retention_leases
    session = getattr(repo, "session", None) if repo is not None else None
    commit = getattr(session, "commit", None) if session is not None else None
    if commit is None:
        raise ComplexResearchError(
            "retention-lease repository exposes no commitable session; "
            "the pinned leases cannot be committed before the checkpoint"
        )
    result = commit()
    if inspect.isawaitable(result):
        await result


async def _lease_pinned_state(
    *,
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    results: tuple[AgentResult, ...],
    runtime: GraphRuntimeContext,
) -> None:
    """Lease every revision/use the validated plan pins (before checkpoint).

    Plan target revisions are pinned as revision-only leases; every
    checkpointed use re-acquires the SAME ``(run, revision, use)`` rows the
    scheduler leased at dispatch (shared ``refresh_pairs_for_checkpoint``
    recipe), so an interrupt keeps them active and a resume refreshes them.
    The subgraph never releases leases.
    """
    run_id = runtime.capability_runtime.run_id
    pairs: list[tuple[UUID | None, UUID | None]] = list(
        refresh_pairs_for_checkpoint(plan, bindings, results)
    )
    binding_by_id = {binding.binding_id: binding for binding in bindings.bindings}
    for unit in plan.target_units:
        binding = binding_by_id.get(unit.binding_id)
        if binding is None:
            raise ComplexResearchError(
                f"target {unit.target_id} binds unknown binding {unit.binding_id}; "
                "refusing to checkpoint an unresolvable pin"
            )
        try:
            revision = UUID(str(binding.document_revision))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ComplexResearchError(
                f"binding {binding.binding_id} pins revision "
                f"{binding.document_revision!r}, not a revision id; refusing "
                "to checkpoint an unleasable pin"
            ) from exc
        pairs.append((revision, None))
    seen: set[tuple[Any, Any]] = set()
    ordered: list[tuple[Any, Any]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            ordered.append(pair)
    if not ordered:
        return
    repo = runtime.services.retention_leases
    if repo is None:
        raise ComplexResearchError(
            "the validated plan pins revisions/uses but no retention-lease "
            "service is wired; refusing to checkpoint unleased pins"
        )
    for revision_id, use_id in ordered:
        result = repo.acquire_or_refresh(run_id, revision_id, use_id)
        if inspect.isawaitable(result):
            await result
    await _commit_lease_session(runtime)


async def validate_checkpoint_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Validate the proposed plan and lease its pins; the saver checkpoints.

    Initial-plan-only: ``validate_task_plan`` is authoritative (T5 adds the
    append-only ``validate_replan`` path). There is no ``plan_checkpoint``
    service — the returned plan enters ``ComplexResearchState`` and the
    supervisor saver performs the checkpoint. A ``None`` plan is the
    unsupported-work path: nothing to validate or pin.
    """
    context = node_context(runtime)
    state = normalize_complex_state(state)
    plan = state.get("plan")
    if plan is None:
        return {}
    bindings = state["bindings"]
    validate_task_plan(plan, bindings)
    await _lease_pinned_state(
        plan=plan,
        bindings=bindings,
        results=tuple(state.get("task_results", ())),
        runtime=context,
    )
    return {"plan": plan}


async def complex_execute_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Execute the checkpointed plan through the shared TaskScheduler only.

    The subgraph never resolves or executes a capability itself: the single
    ``TaskScheduler`` dispatches every ready task and leases every new
    ``EvidenceUse`` before the results are returned (lease before checkpoint).
    """
    context = node_context(runtime)
    state = normalize_complex_state(state)
    plan = state.get("plan")
    if plan is None:
        return {}
    scheduler = shared_scheduler_for(context)
    report = await scheduler.execute(
        plan=plan,
        runtime=context,
        prior_results=tuple(state.get("task_results", ())),
        bindings=state.get("bindings"),
    )
    return {"task_results": report.results}


async def complex_evaluate_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Evaluate through the shared ``evaluate_evidence`` function (no service)."""
    context = node_context(runtime)
    state = normalize_complex_state(state)
    plan = state.get("plan")
    if plan is None:
        return {}
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=state["bindings"],
        results=tuple(state.get("task_results", ())),
        semantic=state["semantic"],
        runtime=context,
    )
    return {"evaluation": evaluation}


def _work_type_of(state: ComplexResearchState) -> str:
    """Best-effort work type for the unavailable reason (never raises)."""
    analysis = state.get("query_analysis")
    if analysis is None:
        return "unknown"
    if isinstance(analysis, dict):
        return str(analysis.get("work_type", "unknown"))
    return str(getattr(analysis, "work_type", "unknown"))


async def decide_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Recommend the terminal step: initial-plan-only, no replan/discovery.

    A missing plan means the work type is out of pilot scope: return the typed
    ``COMPLEX_RESEARCH_UNAVAILABLE`` marker. Otherwise the evaluation stands
    as-is; the supervisor routes ``sufficient`` to synthesis and everything
    else to the finalizer (R4). Evaluator owns sufficiency/contradictions.
    """
    node_context(runtime)
    if state.get("plan") is None:
        work_type = _work_type_of(state)
        return {
            "unavailable": ComplexResearchUnavailable(
                code=COMPLEX_RESEARCH_UNAVAILABLE,
                reason=(
                    f"work type {work_type!r} is out of scope for the "
                    "Phase-3 comparison pilot (initial-plan-only compare)"
                ),
            )
        }
    return {}


async def finalize_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Terminal subgraph step: state already carries plan/results/evaluation."""
    node_context(runtime)
    _ = state
    return {}


def _decide_branch(state: ComplexResearchState) -> str:
    """Decide routing seam: initial-plan-only always finalizes (T5 adds replan)."""
    _ = state
    return "finalize"


def _add_complex_edges(graph: StateGraph) -> None:
    graph.set_entry_point("plan")
    graph.add_edge("plan", "validate_checkpoint")
    graph.add_edge("validate_checkpoint", "execute")
    graph.add_edge("execute", "evaluate")
    graph.add_edge("evaluate", "decide")
    graph.add_conditional_edges("decide", _decide_branch, {"finalize": "finalize"})
    graph.add_edge("finalize", END)


def build_complex_research_subgraph() -> Any:
    """Build the adaptive planning boundary as a checkpointed subgraph.

    Compiled WITHOUT a checkpointer so it inherits the supervisor's saver
    when attached as the ``complex_boundary`` node (a shadow run's isolated
    saver is inherited the same way). It never opens its own production or
    shadow saver, which keeps shadow execution isolated.
    """
    graph = StateGraph(ComplexResearchState, context_schema=GraphRuntimeContext)
    graph.add_node("plan", plan_node)
    graph.add_node("validate_checkpoint", validate_checkpoint_node)
    graph.add_node("execute", complex_execute_node)
    graph.add_node("evaluate", complex_evaluate_node)
    graph.add_node("decide", decide_node)
    graph.add_node("finalize", finalize_node)
    _add_complex_edges(graph)
    return graph.compile()


def build_complex_research_state(state: SupervisorV2State) -> ComplexResearchState:
    """Explicit parent -> child mapping (TypedDict mapping access on the parent).

    Only checkpointed supervisor fields move; runtime-derived planning input
    is built ephemerally inside the subgraph. Nested frozen Pydantic values
    keep attribute access.
    """
    execution = state["execution"]
    return ComplexResearchState(
        contract_version=state["contract_version"],
        semantic=state["semantic"],
        bindings=state["bindings"],
        query_analysis=state["query_analysis"],
        plan=execution.plan,
        task_results=execution.task_results,
        evaluation=execution.evidence_evaluation,
        replans_remaining=MAX_REPLANS,
    )


def merge_complex_result_into_supervisor(
    state: SupervisorV2State, child: ComplexResearchState
) -> dict:
    """Merge only the execution result back (frozen ``evidence_evaluation`` name)."""
    _ = state
    return execution_update(
        state,
        plan=child.get("plan"),
        task_results=child.get("task_results", ()),
        evidence_evaluation=child.get("evaluation"),
    )


def make_complex_boundary_node(complex_subgraph: Any) -> Any:
    """Wrap the compiled subgraph as the supervisor's ``complex_boundary`` node."""

    async def complex_boundary_node(
        state: SupervisorV2State,
        runtime: Any,
        config: Any = None,
    ) -> dict:
        context = node_context(runtime)
        child_input = build_complex_research_state(state)
        child_output = await complex_subgraph.ainvoke(
            child_input, config=config, context=context
        )
        return merge_complex_result_into_supervisor(state, child_output)

    return complex_boundary_node

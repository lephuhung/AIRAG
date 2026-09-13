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
``ComplexResearchState`` or ``SupervisorV2State``. The unvalidated plan
proposal is equally ephemeral (R17/R21, single owner): ``plan_node`` is a
deterministic entry marker that writes NO state, and ``validate_checkpoint_node``
computes the proposal, validates it, acquires the leases, and only then returns
it into state — there is NO cross-node proposal store at all, so no checkpoint
ever carries an unvalidated or unleased plan and no proposal can leak between
requests. The binding ownership sequence holds end to end::

    proposal -> validate -> lease -> checkpoint -> scheduler

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
from dataclasses import dataclass
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
    "V2ResearchLimits",
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

#: Fallback planner sizing when implementation settings carry no v2 limits.
#: ``V2ResearchLimits.from_settings()`` is the live source; these match the
#: gateway defaults so behavior is identical with or without settings.
MAX_TASKS = 8
MAX_PARALLEL_BRANCHES = 2


@dataclass(frozen=True)
class V2ResearchLimits:
    """Implementation planner sizing (config, NOT a frozen contract).

    Read from implementation settings via :meth:`from_settings`, combined with
    the CURRENT execution state by ``build_research_budget_view``. It is never
    persisted and never a ``GraphRuntimeContext`` field: the planner consumes
    it, the scheduler still enforces.
    """

    max_tasks: int = MAX_TASKS
    max_parallel_branches: int = MAX_PARALLEL_BRANCHES
    max_replans: int = MAX_REPLANS

    @classmethod
    def from_settings(cls, settings: Any = None) -> "V2ResearchLimits":
        """Derive limits from implementation settings (lazy import, no coupling)."""
        if settings is None:
            from app.core.config import get_settings

            settings = get_settings()
        return cls(
            max_tasks=int(getattr(settings, "V2_MAX_TASKS", cls.max_tasks)),
            max_parallel_branches=int(
                getattr(settings, "V2_MAX_PARALLEL_BRANCHES", cls.max_parallel_branches)
            ),
            max_replans=int(getattr(settings, "V2_MAX_REPLANS", cls.max_replans)),
        )


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
    """Ephemeral budget from implementation settings + CURRENT execution state.

    Never persisted; rebuilt on each planner/replanner call. Settings supply
    the deployment limits; the live plan and ``replans_remaining`` supply
    what is already consumed.
    """
    limits = V2ResearchLimits.from_settings()
    plan = state.get("plan")
    used_tasks = len(plan.tasks) if plan is not None else 0
    return ResearchBudgetView(
        max_tasks_remaining=max(0, limits.max_tasks - used_tasks),
        max_replans_remaining=max(0, min(state.get("replans_remaining", 0), limits.max_replans)),
        max_parallel_branches=limits.max_parallel_branches,
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
    """Deterministic entry marker (R21 single-owner: writes NO state).

    The planner lives entirely inside ``validate_checkpoint_node``; this node
    only fails closed when complex planning has no analysis yet, and returns
    ``{}`` so the checkpoint written after it carries NO plan. Unsupported
    work types pass through untouched: ``plan`` stays ``None`` so ``decide``
    can return the typed unavailable boundary.
    """
    node_context(runtime)
    state = normalize_complex_state(state)
    require_query_analysis(state)
    return {}


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


def _pinned_pairs(
    plan: TaskPlan,
    bindings: DocumentBindingSet | None,
    results: tuple[AgentResult, ...],
) -> list[tuple[Any, Any]]:
    """Ordered unique ``(revision_id, use_id)`` pairs the plan pins (R22).

    The union of the scheduler's shared use-pair recipe (revision-anchored
    when the owning task resolves to pinned revisions, evidence-only
    otherwise) and the plan's own revision-only pins — the SAME rows
    ``validate_checkpoint_node`` leases before the plan checkpoint, so a
    resume re-acquires identical pairs instead of minting new ones.
    """
    pairs: list[tuple[Any, Any]] = list(
        refresh_pairs_for_checkpoint(plan, bindings, results)
    )
    if bindings is not None:
        binding_by_id = {binding.binding_id: binding for binding in bindings.bindings}
        for unit in plan.target_units:
            binding = binding_by_id.get(unit.binding_id)
            if binding is None:
                raise ComplexResearchError(
                    f"target {unit.target_id} binds unknown binding "
                    f"{unit.binding_id}; refusing to checkpoint an unresolvable pin"
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
    return ordered


async def _acquire_pairs(
    pairs: list[tuple[Any, Any]],
    *,
    runtime: GraphRuntimeContext,
    missing_message: str,
) -> None:
    """Acquire every pair for this run and commit before any checkpoint."""
    if not pairs:
        return
    repo = runtime.services.retention_leases
    if repo is None:
        raise ComplexResearchError(missing_message)
    run_id = runtime.capability_runtime.run_id
    for revision_id, use_id in pairs:
        result = repo.acquire_or_refresh(run_id, revision_id, use_id)
        if inspect.isawaitable(result):
            await result
    await _commit_lease_session(runtime)


async def _lease_pinned_state(
    *,
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    results: tuple[AgentResult, ...],
    runtime: GraphRuntimeContext,
) -> None:
    """Lease every revision/use the validated plan pins (before checkpoint).

    The subgraph never releases leases: an interrupt keeps them active and a
    resume re-acquires the same rows via ``_refresh_existing_pairs``.
    """
    await _acquire_pairs(
        _pinned_pairs(plan, bindings, results),
        runtime=runtime,
        missing_message=(
            "the validated plan pins revisions/uses but no retention-lease "
            "service is wired; refusing to checkpoint unleased pins"
        ),
    )


async def _refresh_existing_pairs(
    *,
    plan: TaskPlan,
    bindings: DocumentBindingSet | None,
    results: tuple[AgentResult, ...],
    runtime: GraphRuntimeContext,
) -> None:
    """Refresh the plan's pinned pairs on (re-)entry to ``execute`` (R22).

    Re-acquires the SAME ``(run, revision, use)`` rows ``validate`` leased —
    including the revision-only plan pins, which need no prior results — so a
    resume provably refreshes pre-interrupt leases instead of merely minting
    new-use leases. New uses are still leased by the scheduler itself before
    its results return.
    """
    await _acquire_pairs(
        _pinned_pairs(plan, bindings, results),
        runtime=runtime,
        missing_message=(
            "checkpointed pins exist but no retention-lease service is "
            "wired; refusing to run with unrefreshable pins"
        ),
    )


async def validate_checkpoint_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Own the proposal end to end: compute, validate, lease, persist (R21).

    Single-owner node (R17/R21): the planner invocation lives HERE, so there
    is no cross-node proposal store that could go stale or leak between
    requests — concurrent invocations sharing a run id compute from their own
    checkpointed state. Initial-plan-only: ``validate_task_plan`` is
    authoritative (T5 adds the append-only ``validate_replan`` path). There
    is no ``plan_checkpoint`` service — the returned plan enters
    ``ComplexResearchState`` and the supervisor saver performs the
    checkpoint, so the checkpoint that first persists ``plan`` is written by
    the SAME node that already acquired its leases. An unplannable input is
    the unsupported-work path: nothing is validated, leased, or persisted.
    """
    context = node_context(runtime)
    state = normalize_complex_state(state)
    try:
        proposal = compare_policy.build_compare_plan(
            build_planning_input(state, context)
        )
    except ContractValidationError:
        return {}
    bindings = state["bindings"]
    validate_task_plan(proposal, bindings)
    await _lease_pinned_state(
        plan=proposal,
        bindings=bindings,
        results=tuple(state.get("task_results", ())),
        runtime=context,
    )
    return {"plan": proposal}


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
    prior_results = tuple(state.get("task_results", ()))
    await _refresh_existing_pairs(
        plan=plan,
        bindings=state.get("bindings"),
        results=prior_results,
        runtime=context,
    )
    scheduler = shared_scheduler_for(context)
    report = await scheduler.execute(
        plan=plan,
        runtime=context,
        prior_results=prior_results,
        bindings=state.get("bindings"),
    )
    return {"task_results": report.results}


async def complex_evaluate_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Evaluate through the shared ``evaluate_evidence`` function (no service).

    Refreshes the pinned pairs on entry first, so a resume landing here
    re-acquires the exact ``(run, revision, use)`` rows behind the
    checkpointed results (R22) before the evaluation checkpoint is written.
    """
    context = node_context(runtime)
    state = normalize_complex_state(state)
    plan = state.get("plan")
    if plan is None:
        return {}
    results = tuple(state.get("task_results", ()))
    await _refresh_existing_pairs(
        plan=plan,
        bindings=state.get("bindings"),
        results=results,
        runtime=context,
    )
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=state["bindings"],
        results=results,
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
        # A resume re-enters the subgraph mid-graph, so slots no node
        # rewrote (e.g. the validated plan) arrive serde-degraded; coerce
        # the raw child output exactly like a checkpoint load before the
        # supervisor wrapper re-validates the merged aggregate.
        child_output = normalize_complex_state(child_output)
        return merge_complex_result_into_supervisor(state, child_output)

    return complex_boundary_node

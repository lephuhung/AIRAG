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

Pilot scope is skill-selected initial planning (compare, large/iterative
summarize with a deterministic map/reduce proposal) plus the bounded
append-only replan loop (Task 5, fix round 1): ``decide`` routes an advisable
coverage gap — or a no-evidence targetless failure awaiting its fallback
(R37) — to the ``replan`` node, which proposes from validated
observations/evaluation gaps (never raw evidence), and
``validate_checkpoint`` validates the append-only replan, leases, and
checkpoints it before the next ``execute``. The ``settle`` node hands
discovery candidates to the Binding Resolver (policy-gated, R39). An
unsupported work type (``evaluate``, ``cross_domain``, ``multi_goal``, ...)
returns the typed ``COMPLEX_RESEARCH_UNAVAILABLE`` marker from the ``decide``
node — never a fabricated plan.
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
from .contracts.binding import DocumentBindingSet, ScopedDocument
from .contracts.capability import (
    DocumentReadInput,
    DocumentSearchInput,
    PeopleLookupInput,
    SectionReadInput,
)
from .contracts.evaluation import EvidenceEvaluation
from .contracts.execution import AgentResult, TaskExecutionSummary
from .contracts.evidence import EvidenceUseRef
from .contracts.locators import SectionLocator
from .contracts.planning import (
    DiscoveryPolicy,
    InitialTaskOrigin,
    ReplanTaskOrigin,
    ResearchBudgetView,
    ResearchPlanningInput,
    TaskPlan,
    TaskSpec,
)
from .contracts.routing import QueryAnalysis, RouteDecision
from .contracts.semantic import SemanticContext
from .contracts.state import GraphRuntimeContext, SupervisorV2State
from .contracts.synthesis import SynthesisInput
from .contracts.validation import ContractValidationError, validate_task_plan
from .dependencies.people_document import (
    MaterializationError,
    append_materialized_dependent,
    materialize_person_dependency,
    redact_scalar_for_model,
)
from .execution.scheduler import (
    refresh_active_run_async as _refresh_active_run_async,
    refresh_pairs_for_checkpoint,
    shared_scheduler_for,
)
from .discovery import DiscoveryDenied, DiscoveryDisabled, request_addition
from .nodes.context import node_context
from .nodes.evaluate import _EVIDENCE_SUPPLYING_CAPABILITIES, evaluate_evidence
from .planning import PlannerError
from .nodes.execute import execution_update
from .nodes.synthesize import DEFAULT_SYNTHESIS_BUDGET, synthesize_answer
from .replanning import (
    ReplanRejected,
    append_replan_tasks,
)
from .skills.compare import policy as compare_policy
from .skills.retrieve import policy as retrieve_policy
from .skills.summarize import policy as summarize_policy
from .skills.summarize.policy import ReduceSpec
from .tools.discovery_candidates import (
    CandidateNotFound,
    DiscoveryCandidateRegistry,
    InvalidCandidateRole,
)
from .tools.observations import AgentToolObservation, ObservationProjector

__all__ = [
    "COMPLEX_RESEARCH_UNAVAILABLE",
    "_build_complex_research_graph",
    "ComplexResearchError",
    "ComplexResearchState",
    "ComplexResearchUnavailable",
    "V2ResearchLimits",
    "build_complex_research_state",
    "build_complex_research_subgraph",
    "build_governed_initial_proposal",
    "build_discovery_policy",
    "build_initial_proposal",
    "build_model_observations",
    "build_model_replan_input",
    "build_planning_input",
    "build_replan_proposal",
    "build_research_budget_view",
    "build_task_execution_summaries",
    "collect_evidence_use_refs",
    "complex_evaluate_node",
    "complex_execute_node",
    "decide_node",
    "finalize_node",
    "make_complex_boundary_node",
    "discovery_settle_node",
    "merge_complex_result_into_supervisor",
    "summarize_reduce_node",
    "normalize_complex_state",
    "people_document_materialize_node",
    "plan_node",
    "replan_advisable",
    "replan_node",
    "require_query_analysis",
    "validate_checkpoint_node",
]

#: Typed-unavailable code for out-of-pilot work (evaluate/compliance, ...).
COMPLEX_RESEARCH_UNAVAILABLE = "COMPLEX_RESEARCH_UNAVAILABLE"

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
    #: Closed by default (R36): the single source of truth for the replan
    #: budget is ``V2_MAX_REPLANS`` in implementation settings, read by
    #: :meth:`from_settings`. No hardcoded constant exists anywhere else.
    max_replans: int = 0

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
    #: The already-resolved frozen router outcome, threaded from the
    #: supervisor alongside ``query_analysis`` (R73). The execute node
    #: derives the post-router v1-fallback guard from these two slots —
    #: never from a fresh classification.
    route_decision: RouteDecision | None
    plan: TaskPlan | None
    task_results: tuple[AgentResult, ...]
    evaluation: EvidenceEvaluation | None
    replans_remaining: int
    unavailable: ComplexResearchUnavailable | None
    #: True only on the pass that appended a materialized T2 (R26 routing).
    #: The ``materialize -> execute`` edge is taken exactly then, so the
    #: scheduler dispatches the checkpointed T2; every other pass routes to
    #: ``evaluate``. Appends are bounded by the validate_replan task budget,
    #: so the loop always terminates.
    materialized_new_task: bool
    #: Checkpointed materialization decisions per people task (R28): True
    #: iff the scalar was extractable under current governance right now.
    #: Future model-facing observation builders MUST read this map (via the
    #: projector's explicit ``dependency_scalar_available`` parameter) rather
    #: than inferring availability from status.
    people_scalar_available: dict[str, bool]
    #: Explicit deterministic REDUCE specification for summarize map/reduce
    #: workflows (R43, implementation-only — not a frozen contract). Set only
    #: by the summarize initial proposal; ``None`` everywhere else. The
    #: reduce node consumes it; replans clear it so no stale spec survives
    #: a changed task set.
    reduce_spec: Any
    #: Discovery candidates left unsettled because the policy cap was
    #: reached (R44): candidate-ID strings in first-seen order. Recorded,
    #: never added; a later pass with budget skips already-bound candidates
    #: and settles the rest.
    discovery_deferred: tuple[str, ...]


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
    """Runtime-only policy: discovery stays rejected unless settings allow it.

    Deployment settings may authorize reference/supporting expansion
    (``V2_ALLOW_REFERENCE_DISCOVERY`` / ``V2_ALLOW_SUPPORTING_DISCOVERY`` /
    ``V2_MAX_DISCOVERED_DOCUMENTS``); the default keeps the pilot closed.
    Policy restricts but never grants permission: the request-scoped registry
    and current ACL still narrow every dispatch.
    """
    _ = runtime
    try:
        from app.core.config import get_settings

        settings = get_settings()
    except Exception:
        settings = None
    return DiscoveryPolicy(
        allow_reference_discovery=bool(
            getattr(settings, "V2_ALLOW_REFERENCE_DISCOVERY", False)
        ),
        allow_supporting_discovery=bool(
            getattr(settings, "V2_ALLOW_SUPPORTING_DISCOVERY", False)
        ),
        max_discovered_documents=int(
            getattr(settings, "V2_MAX_DISCOVERED_DOCUMENTS", 0)
        ),
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
        route_decision=_coerce_slot(
            state.get("route_decision"), RouteDecision, slot="route_decision"
        ),
        plan=_coerce_slot(state.get("plan"), TaskPlan, slot="plan"),
        task_results=results,
        evaluation=_coerce_slot(state.get("evaluation"), EvidenceEvaluation, slot="evaluation"),
        replans_remaining=int(state.get("replans_remaining", 0)),
        unavailable=_coerce_slot(
            state.get("unavailable"), ComplexResearchUnavailable, slot="unavailable"
        ),
        materialized_new_task=_coerce_checkpoint_flag(
            state.get("materialized_new_task", False)
        ),
        people_scalar_available=_coerce_availability_map(
            state.get("people_scalar_available", {})
        ),
        reduce_spec=_coerce_reduce_spec(state.get("reduce_spec")),
        discovery_deferred=_coerce_deferred_list(
            state.get("discovery_deferred", ())
        ),
    )


def _coerce_checkpoint_flag(value: Any) -> bool:
    """Coerce the checkpointed materialize-routing flag (fail closed)."""
    if isinstance(value, bool):
        return value
    raise ComplexResearchError(
        "complex checkpoint slot 'materialized_new_task' is not a bool "
        f"(got {type(value).__name__})"
    )


def _coerce_reduce_spec(value: Any) -> ReduceSpec | None:
    """Coerce the checkpointed reduce spec (fail closed, R43)."""
    if value is None:
        return None
    if isinstance(value, ReduceSpec):
        if value.mode != "extractive" or not value.map_task_ids:
            raise ComplexResearchError(
                "complex checkpoint slot 'reduce_spec' is not a valid "
                "extractive reduce specification"
            )
        return value
    if isinstance(value, Mapping):
        try:
            spec = ReduceSpec(
                map_task_ids=tuple(value.get("map_task_ids", ()) or ()),
                mode=value.get("mode", "extractive"),
            )
        except (TypeError, ValueError) as exc:
            raise ComplexResearchError(
                f"complex checkpoint slot 'reduce_spec' is not a valid "
                f"reduce specification: {exc}"
            ) from exc
        return _coerce_reduce_spec(spec)
    raise ComplexResearchError(
        "complex checkpoint slot 'reduce_spec' is not a reduce "
        f"specification (got {type(value).__name__})"
    )


def _coerce_deferred_list(value: Any) -> tuple[str, ...]:
    """Coerce the checkpointed deferred-candidate record (fail closed, R44)."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) and item for item in value
    ):
        return tuple(value)
    raise ComplexResearchError(
        "complex checkpoint slot 'discovery_deferred' is not a tuple of "
        f"candidate-id strings (got {type(value).__name__})"
    )


def _coerce_availability_map(value: Any) -> dict[str, bool]:
    """Coerce the checkpointed scalar-availability map (fail closed)."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        coerced = {str(key): item for key, item in value.items()}
        if all(isinstance(item, bool) for item in coerced.values()):
            return coerced
    raise ComplexResearchError(
        "complex checkpoint slot 'people_scalar_available' is not a "
        f"str->bool mapping (got {type(value).__name__})"
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


def build_model_replan_input(
    state: ComplexResearchState, runtime: GraphRuntimeContext
) -> ResearchPlanningInput:
    """Model-facing replanning projection: redacted plan, never raw evidence (R34).

    Ephemeral like :func:`build_planning_input` and never checkpointed. The
    checkpointed ``current_plan`` keeps its governed scalar (frozen contract
    fidelity); the projection the planner model sees carries
    ``person_identifier=None`` via ``redact_scalar_for_model`` (R25
    carry-over). Every model-facing planning/replanning projection MUST flow
    through here.
    """
    base = build_planning_input(state, runtime)
    if base.current_plan is None:
        return base
    return base.model_copy(
        update={"current_plan": redact_scalar_for_model(base.current_plan)}
    )


def build_model_observations(
    state: ComplexResearchState,
) -> tuple[AgentToolObservation, ...]:
    """Validated per-task observations the replanner may see (R34).

    Projects each checkpointed ``AgentResult`` through the typed
    ``ObservationProjector`` with the checkpointed per-task availability
    decision — never raw evidence, never the governed scalar. An unknown
    result kind fails closed instead of leaking a generic carrier.
    """
    availability = state.get("people_scalar_available", {}) or {}
    observations: list[AgentToolObservation] = []
    for result in tuple(state.get("task_results", ())):
        flag = availability.get(result.task_id) if isinstance(availability, Mapping) else None
        observations.append(
            ObservationProjector.project(
                result,
                dependency_scalar_available=bool(flag) if flag is not None else None,
            )
        )
    return tuple(observations)


def _evaluation_slot(value: Any, name: str, default: Any = None) -> Any:
    """Read one evaluation slot from a live model or its serde mapping."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True)
class InitialProposal:
    """Deterministic initial proposal: executable plan + optional reduce spec.

    ``reduce_spec`` is set only for summarize map/reduce workflows (R43);
    every other pilot carries ``None``.
    """

    plan: TaskPlan
    reduce_spec: ReduceSpec | None


@dataclass(frozen=True)
class ReplanProposal:
    """Deterministic append-only replan proposal (R52): new tasks only.

    ``build_replan_proposal`` returns this proposal -- never an authoritative
    plan and never an already-appended plan. The governed
    ``validate_checkpoint_node`` performs the single authoritative append +
    validation (``append_replan_tasks``) before leasing/checkpointing.
    """

    new_tasks: tuple[TaskSpec, ...]
    outcomes: tuple[TaskExecutionSummary, ...]
    policy: DiscoveryPolicy
    budget: ResearchBudgetView


def _people_first_plan(planning_input: ResearchPlanningInput) -> TaskPlan:
    """Deterministic first step of the cross-domain People→Document pilot.

    A single targetless ``people.lookup`` (T1) when the finalized semantics
    names a person; the deterministic materializer appends the governed
    dependent on success and the recovery replan owns ``not_found``/``TIMEOUT``.
    Fails closed without person references or without the capability in the
    current catalog — the planner never fabricates a person to look up.
    """
    references = sorted(
        planning_input.semantic.person_refs, key=lambda reference: reference.ref_id
    )
    if not references:
        raise ContractValidationError(
            "cross_domain pilot needs a person reference for the governed "
            "first lookup; refusing to fabricate one"
        )
    catalog = {entry.name for entry in planning_input.capability_catalog}
    if "people.lookup" not in catalog:
        raise ContractValidationError(
            "cross_domain pilot needs 'people.lookup' in the request-scoped "
            "capability catalog; refusing to plan an undispatchable lookup"
        )
    first = references[0]
    task = TaskSpec(
        task_id="T1",
        capability="people.lookup",
        task_objective=f"Resolve person {first.label} ({first.ref_id})",
        input=PeopleLookupInput(
            kind="people.lookup",
            query=planning_input.semantic.contextualized_query,
        ),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id=f"people-first-{first.ref_id}",
        goal=planning_input.semantic.contextualized_query,
        target_units=(),
        tasks=(task,),
    )
    validate_task_plan(plan, planning_input.bindings)
    return plan


async def build_governed_initial_proposal(
    planning_input: ResearchPlanningInput, runtime: GraphRuntimeContext
) -> InitialProposal:
    """Deterministic skill first, governed Adaptive Planner otherwise (Task 10).

    Without a wired ``adaptive_planner`` service this is exactly
    :func:`build_initial_proposal` (legacy behavior: work no skill covers
    raises :class:`ContractValidationError`). With one, the planner owns
    the deterministic-first ordering — a covering skill's refusal is final
    and the model path runs only for uncovered work types (v1-owned
    ``evaluate`` excluded until Task 12) — and the model path stays
    proposal-only: the caller still validates, leases, and checkpoints
    before the shared scheduler dispatches anything.
    """
    planner = getattr(runtime.services, "adaptive_planner", None)
    if planner is None:
        return build_initial_proposal(planning_input)
    propose = getattr(planner, "propose_initial", None)
    if propose is None:
        raise ComplexResearchError(
            "adaptive_planner exposes no propose_initial; refusing to plan "
            "through a miswired planner service"
        )
    return await propose(planning_input, runtime)


def build_initial_proposal(planning_input: ResearchPlanningInput) -> InitialProposal:
    """Select the skill/policy from the query work type (R38), then propose.

    ``compare`` (complex) routes to the compare skill; large/iterative
    ``summarize`` (complex) routes to the summarize skill with its
    deterministic map/reduce workflow; ``retrieve`` routes to the retrieve
    skill with its single deterministic ``document.retrieve`` task;
    ``cross_domain`` with a named person
    routes to the deterministic people-first lookup (the People→Document
    pilot's governed first step). Bounded summarize never reaches here (the
    deterministic router keeps single-document summaries on the fast path).
    Any other work type raises :class:`ContractValidationError` so the
    caller returns the typed unavailable boundary, never a plan.
    """
    work_type = planning_input.query_analysis.work_type
    if work_type == compare_policy.COMPARE_WORK_TYPE:
        return InitialProposal(
            plan=compare_policy.build_compare_plan(planning_input),
            reduce_spec=None,
        )
    if work_type == summarize_policy.SUMMARIZE_WORK_TYPE:
        workflow = summarize_policy.build_summarize_workflow(planning_input)
        return InitialProposal(plan=workflow.plan, reduce_spec=workflow.reduce)
    if work_type == retrieve_policy.RETRIEVE_WORK_TYPE:
        return InitialProposal(
            plan=retrieve_policy.build_retrieve_plan(planning_input),
            reduce_spec=None,
        )
    if work_type == "cross_domain":
        return InitialProposal(
            plan=_people_first_plan(planning_input), reduce_spec=None
        )
    raise ContractValidationError(
        f"work type {work_type!r} has no complex skill policy; out of scope"
    )


#: Outcome statuses that admit a no-evidence recovery replan (R37): the
#: completed task stays completed (never rerun) and the fallback carries no
#: fabricated scalar. Denial and needs_input are terminal and never recover.
_RECOVERABLE_OUTCOME_STATUSES = frozenset({"not_found", "error"})


def _recovery_candidates(plan: Any, results: tuple[Any, ...]) -> tuple[Any, ...]:
    """Targetless evidence-supplying tasks whose failure left no target gap.

    A ``people.lookup`` (or other targetless evidence supplier) returning
    ``not_found`` or ``error`` with no admitted uses cannot create a
    target-based ``MissingRequirement``, yet the run must stay actionable
    (spec §26): these tasks are recovery candidates unless a dependent task
    was already appended for them. Tolerant of checkpoint-serde mappings so
    the decide seam stays total on resumed state.
    """
    tasks = _evaluation_slot(plan, "tasks", ()) or ()
    result_by_task = {
        _evaluation_slot(result, "task_id"): result for result in results
    }
    candidates: list[Any] = []
    for task in tasks:
        capability = _evaluation_slot(task, "capability")
        if capability not in _EVIDENCE_SUPPLYING_CAPABILITIES:
            continue
        if capability == "document.search":
            continue
        task_input = _evaluation_slot(task, "input")
        if _evaluation_slot(task_input, "target_ids", ()):
            continue
        task_id = _evaluation_slot(task, "task_id")
        result = result_by_task.get(task_id)
        if result is None:
            continue
        if _evaluation_slot(result, "status") not in _RECOVERABLE_OUTCOME_STATUSES:
            continue
        if tuple(_evaluation_slot(result, "evidence_uses", ()) or ()):
            continue
        if any(
            task_id in (_evaluation_slot(other, "depends_on", ()) or ())
            for other in tasks
            if _evaluation_slot(other, "task_id") != task_id
        ):
            continue
        candidates.append(task)
    return tuple(candidates)


def replan_advisable(state: ComplexResearchState) -> bool:
    """True only for a bounded coverage-gap or no-evidence recovery replan.

    Advisable means: a checkpointed plan exists, the latest evaluation is
    ``insufficient``, replan budget remains, and either coverage-kind gaps
    exist (automatic re-read; semantic gaps and contradictions belong to
    skill strategy and the evaluator) or a targetless evidence-supplying
    task failed without admitted uses and awaits its fallback (R37, spec
    §26). Everything else — sufficient, contradictory, needs_input,
    missing evaluation, exhausted budget — finalizes. Total function: never
    raises on shape drift.
    """
    try:
        if state.get("plan") is None:
            return False
        if state.get("unavailable") is not None:
            return False
        evaluation = state.get("evaluation")
        if evaluation is None:
            return False
        if _evaluation_slot(evaluation, "status") != "insufficient":
            return False
        if int(state.get("replans_remaining", 0)) < 1:
            return False
        missing = _evaluation_slot(evaluation, "missing", ()) or ()
        for requirement in missing:
            if _evaluation_slot(requirement, "criterion_kind") != "coverage":
                return False
        if missing:
            return True
        plan = state.get("plan")
        results = tuple(state.get("task_results", ()))
        return bool(_recovery_candidates(plan, results))
    except (TypeError, ValueError, AttributeError):
        return False


def _recovery_search_task(
    failed: TaskSpec,
    result: AgentResult,
    *,
    next_task_id: str,
    query: str,
    evidence_use_ids: tuple[Any, ...],
) -> TaskSpec:
    """Fallback discovery search after a no-evidence targetless failure (R37).

    The completed task is never rerun and no downstream input is fabricated:
    the search carries the checkpointed plan goal as its query and
    ``person_identifier=None`` — the governed scalar is never guessed. The
    reason keeps ``not_found`` distinct from ``TIMEOUT``/other errors.
    Depends on the failed task so ordering lineage survives; policy
    validation (discovery must be authorized) happens in the wrapper.
    """
    status = result.status
    if status == "not_found":
        reason = (
            f"task {failed.task_id} ({failed.capability}) returned not_found "
            "with no admitted evidence; fallback discovery search without "
            "a materialized scalar"
        )
    else:
        code = result.error.code if result.error is not None else "INTERNAL_ERROR"
        reason = (
            f"task {failed.task_id} ({failed.capability}) failed with "
            f"{code} and no admitted evidence; fallback discovery search "
            "without a materialized scalar"
        )
    return TaskSpec(
        task_id=next_task_id,
        capability="document.search",
        task_objective=(
            f"Broad discovery search after {failed.task_id} "
            f"returned {status} with no admitted evidence"
        ),
        input=DocumentSearchInput(
            kind="document.search",
            query=query,
            person_identifier=None,
        ),
        depends_on=(failed.task_id,),
        origin=ReplanTaskOrigin(
            kind="replan",
            reason=reason,
            task_ids=(failed.task_id,),
            evidence_use_ids=evidence_use_ids,
        ),
    )


def build_replan_proposal(
    state: ComplexResearchState, runtime: GraphRuntimeContext
) -> ReplanProposal | None:
    """Propose the bounded append-only replan from validated gaps (R32/R37).

    Consumes the redacted model-facing projection (R34) and the validated
    observations — never raw evidence. One dependency-free re-read per
    missing coverage target (section reads for section coordinates, document
    reads otherwise), plus one policy-gated fallback discovery search per
    no-evidence targetless failure (R37: ``not_found``/``TIMEOUT`` stay
    distinct, the completed task is never rerun, no scalar is fabricated).
    Every new task carries ``ReplanTaskOrigin`` trigger lineage (attempted
    task IDs + prior evidence-use IDs). Returns a :class:`ReplanProposal`
    (new tasks + validation inputs), never an authoritative appended plan;
    the governed ``validate_checkpoint_node`` owns the append + validation.
    Returns ``None`` when no bounded replan is advisable; the caller then
    spends the budget so ``decide`` finalizes instead of looping.
    """
    if not replan_advisable(state):
        return None
    plan = state["plan"]
    assert plan is not None
    evaluation = state["evaluation"]
    assert evaluation is not None
    results = tuple(state.get("task_results", ()))
    outcomes = build_task_execution_summaries(results)
    prior_use_ids = tuple(ref.use_id for ref in collect_evidence_use_refs(results))
    _ = build_model_observations(state)
    model_input = build_model_replan_input(state, runtime)
    catalog = {entry.name for entry in model_input.capability_catalog}
    unit_by_target = {unit.target_id: unit for unit in plan.target_units}
    attempted_by_target: dict[str, list[str]] = {}
    for task in plan.tasks:
        for target_id in getattr(task.input, "target_ids", None) or ():
            attempted_by_target.setdefault(target_id, []).append(task.task_id)
    outcome_ids = {outcome.task_id for outcome in outcomes}
    taken = {task.task_id for task in plan.tasks}
    index = len(plan.tasks) + 1
    new_tasks: list[TaskSpec] = []
    for gap in _evaluation_slot(evaluation, "missing", ()) or ():
        target_id = _evaluation_slot(gap, "target_id")
        unit = unit_by_target.get(target_id)
        if unit is None:
            return None
        if isinstance(unit.requested_locator, SectionLocator):
            capability = "section.read"
            task_input: Any = SectionReadInput(
                kind="section.read", target_ids=(target_id,)
            )
        else:
            capability = "document.read"
            task_input = DocumentReadInput(
                kind="document.read", target_ids=(target_id,)
            )
        if capability not in catalog:
            return None
        while f"T{index}" in taken:
            index += 1
        task_id = f"T{index}"
        index += 1
        taken.add(task_id)
        trigger_tasks = tuple(
            task_id_
            for task_id_ in attempted_by_target.get(target_id, ())
            if task_id_ in outcome_ids
        )
        new_tasks.append(
            TaskSpec(
                task_id=task_id,
                capability=capability,
                task_objective=(
                    f"Re-read {target_id} for missing coverage requirement"
                ),
                input=task_input,
                depends_on=(),
                origin=ReplanTaskOrigin(
                    kind="replan",
                    reason=(
                        f"coverage gap for target {target_id}: "
                        f"{_evaluation_slot(gap, 'description')}"
                    ),
                    task_ids=trigger_tasks,
                    evidence_use_ids=prior_use_ids,
                ),
            )
        )
    result_by_task = {result.task_id: result for result in results}
    for failed in _recovery_candidates(plan, results):
        if "document.search" not in catalog:
            return None
        while f"T{index}" in taken:
            index += 1
        task_id = f"T{index}"
        index += 1
        taken.add(task_id)
        recovery_result = result_by_task.get(failed.task_id)
        if recovery_result is None:
            return None
        new_tasks.append(
            _recovery_search_task(
                failed,
                recovery_result,
                next_task_id=task_id,
                query=plan.goal,
                evidence_use_ids=prior_use_ids,
            )
        )
    if not new_tasks:
        return None
    # R52: return a PROPOSAL, not an authoritative plan. The governed
    # validate_checkpoint_node performs the single authoritative append +
    # validation (append_replan_tasks) before lease/checkpoint.
    return ReplanProposal(
        new_tasks=tuple(new_tasks),
        outcomes=outcomes,
        policy=model_input.discovery_policy,
        budget=model_input.budget,
    )


async def replan_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Deterministic replan entry marker (R21 single-owner: writes NO state).

    The replan proposal lives entirely inside ``validate_checkpoint_node``;
    this node only fails closed when entered without a checkpointed plan or
    without an advisable gap, and returns ``{}`` so no checkpoint written
    after it carries an unvalidated plan.
    """
    node_context(runtime)
    state = normalize_complex_state(state)
    if state.get("plan") is None:
        raise ComplexResearchError(
            "replan requires a checkpointed plan; refusing to replan without one"
        )
    if not replan_advisable(state):
        raise ComplexResearchError(
            "replan entered without an advisable coverage gap; refusing to "
            "fabricate a replan"
        )
    return {}


async def validate_checkpoint_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Own the proposal end to end: compute, validate, lease, persist (R21).

    Single-owner node (R17/R21): the planner invocation lives HERE, so there
    is no cross-node proposal store that could go stale or leak between
    requests — concurrent invocations sharing a run id compute from their own
    checkpointed state. Initial entry (no checkpointed plan) validates the
    compare-skill proposal with ``validate_task_plan`` (via the governed
    entry, which consults the wired ``adaptive_planner`` only when no
    deterministic skill covers the work type); replan entry (a
    checkpointed plan exists) validates the gap-driven append with the
    runtime replan wrapper (frozen ``validate_replan`` + current catalog).
    There is no ``plan_checkpoint`` service — the returned plan enters
    ``ComplexResearchState`` and the supervisor saver performs the
    checkpoint, so the checkpoint that first persists the plan/replan is
    written by the SAME node that already acquired its leases. An
    unplannable input is the unsupported-work path (initial) or a spent
    budget (replan, so ``decide`` finalizes): nothing unvalidated is leased
    or persisted.
    """
    context = node_context(runtime)
    state = normalize_complex_state(state)
    if state.get("plan") is None:
        try:
            initial = await build_governed_initial_proposal(
                build_planning_input(state, context), context
            )
        except (ContractValidationError, PlannerError):
            return {}
        bindings = state["bindings"]
        validate_task_plan(initial.plan, bindings)
        await _lease_pinned_state(
            plan=initial.plan,
            bindings=bindings,
            results=tuple(state.get("task_results", ())),
            runtime=context,
        )
        update: dict[str, Any] = {
            "plan": initial.plan,
            "materialized_new_task": False,
        }
        if initial.reduce_spec is not None:
            update["reduce_spec"] = initial.reduce_spec
        return update
    current = state["plan"]
    assert current is not None
    proposal = build_replan_proposal(state, context)
    if proposal is None:
        # Deterministically unplannable: spend the budget so decide finalizes
        # instead of routing back here forever.
        return {
            "replans_remaining": 0,
            "materialized_new_task": False,
            "reduce_spec": None,
        }
    # R52: the single authoritative append + validation happen HERE, inside
    # the governed entry point, on the proposal returned by
    # build_replan_proposal (which never appends or validates itself).
    try:
        accepted = append_replan_tasks(
            current,
            proposal.new_tasks,
            proposal.outcomes,
            proposal.policy,
            proposal.budget,
            context,
        )
    except (ContractValidationError, ReplanRejected):
        # Deterministically invalid: spend the budget so decide finalizes.
        return {
            "replans_remaining": 0,
            "materialized_new_task": False,
            "reduce_spec": None,
        }
    await _lease_pinned_state(
        plan=accepted,
        bindings=state["bindings"],
        results=tuple(state.get("task_results", ())),
        runtime=context,
    )
    remaining = int(state.get("replans_remaining", 0))
    return {
        "plan": accepted,
        "replans_remaining": max(0, remaining - 1),
        "materialized_new_task": False,
        # The task set changed: no stale reduce spec may survive.
        "reduce_spec": None,
    }


def _production_v1_fallback_guard(state: ComplexResearchState) -> Any:
    """Build the production post-router fallback guard (R73).

    Derived from the ALREADY-resolved frozen ``QueryAnalysis`` and
    ``RouteDecision`` in graph state — the sole guard the production
    ComplexResearch scheduler call passes. Fires (returns True) when the
    resolved route is write, evaluate/legal/compliance, or otherwise
    unsupported, so the request falls back to v1 BEFORE any capability
    execution or user-visible output. ``None`` outcomes fail closed
    (guard fires) via ``requires_v1_fallback``.
    """
    analysis = state.get("query_analysis")
    decision = state.get("route_decision")

    def _guard() -> bool:
        from app.services.agent.rollout_control import requires_v1_fallback

        return bool(requires_v1_fallback(analysis, decision))

    return _guard


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
    try:
        run_id = str(context.capability_runtime.run_id or "")
    except Exception:
        run_id = ""
    if run_id:
        # I2: refresh the distributed active registration for the run
        # lifetime (long/resumed runs outlive the fixed TTL otherwise).
        await _refresh_active_run_async(run_id)
    scheduler = shared_scheduler_for(context)
    report = await scheduler.execute(
        plan=plan,
        runtime=context,
        prior_results=prior_results,
        bindings=state.get("bindings"),
        v1_fallback_guard=_production_v1_fallback_guard(state),
    )
    return {"task_results": report.results}


async def people_document_materialize_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Deterministic People→Document materialization (R23, Phase 3 Task 4).

    Runs BETWEEN the T1 execute result and the append/validate/checkpoint of
    T2::

        T1 people.lookup result (checkpointed by the execute node)
        -> governed hydration under current ACL/expiry (no connector call)
        -> exact approved scalar -> concrete DocumentSearchInput
        -> append T2 TaskSpec(input=<concrete input>)
        -> validate_replan -> retention-lease commit -> checkpoint T2
        -> TaskScheduler dispatches T2 unchanged (next execute pass)

    This is a deterministic node, not a planner replan: it consumes no
    planner replan budget and proposes no open discovery. The narrow
    dependency-scoped policy authorizes exactly this governed dependent. Any
    non-materializable outcome (``not_found``/denied/timeout/error, expired
    or unauthorized use, missing/conflicting scalar, foreign-owned evidence)
    appends nothing and records ``False`` in ``people_scalar_available``:
    nothing is fabricated. Dormant when the plan has no unanswered successful
    ``people.lookup``.

    Routing (R26): the returned ``materialized_new_task`` is True only on the
    pass that appended T2, so the graph routes back to ``execute`` exactly
    once per append and the scheduler dispatches the checkpointed T2
    unchanged; all other passes route to ``evaluate``.
    """
    context = node_context(runtime)
    state = normalize_complex_state(state)
    plan = state.get("plan")
    bindings = state.get("bindings")
    availability: dict[str, bool] = dict(state.get("people_scalar_available", {}))
    if plan is None or bindings is None:
        # Explicit reset: a bare {} would leave a stale True routing flag
        # in state and loop materialize -> execute forever.
        return {"materialized_new_task": False}
    results = tuple(state.get("task_results", ()))
    result_by_task = {result.task_id: result for result in results}
    examined = False
    for task in plan.tasks:
        if task.capability != "people.lookup":
            continue
        people_result = result_by_task.get(task.task_id)
        if people_result is None:
            continue
        if any(
            dependent.capability == "document.search"
            and task.task_id in dependent.depends_on
            for dependent in plan.tasks
        ):
            continue
        examined = True
        outcome = await materialize_person_dependency(
            people_task_id=task.task_id,
            people_result=people_result,
            runtime=context,
            plan=plan,
            bindings=bindings,
            query=plan.goal,
        )
        availability[task.task_id] = outcome.kind == "materialized"
        if outcome.kind != "materialized":
            continue
        taken = {existing.task_id for existing in plan.tasks}
        index = len(plan.tasks) + 1
        while f"T{index}" in taken:
            index += 1
        limits = V2ResearchLimits.from_settings()
        # R52: append_materialized_dependent returns the concrete T2 PROPOSAL
        # (never an authoritative plan); the single authoritative append +
        # validation (append_replan_tasks) happen HERE, inside this governed
        # node, before lease/checkpoint.
        outcomes = build_task_execution_summaries(results)
        policy = DiscoveryPolicy(
            allow_reference_discovery=False,
            allow_supporting_discovery=True,
            max_discovered_documents=1,
        )
        budget = ResearchBudgetView(
            max_tasks_remaining=max(0, limits.max_tasks - len(plan.tasks)),
            max_replans_remaining=1,
            max_parallel_branches=limits.max_parallel_branches,
        )
        try:
            dependent = append_materialized_dependent(
                current=plan,
                outcome=outcome,
                query=plan.goal,
                next_task_id=f"T{index}",
            )
            accepted = append_replan_tasks(
                plan, (dependent,), outcomes, policy, budget, context
            )
        except (MaterializationError, ContractValidationError, ReplanRejected):
            continue
        await _lease_pinned_state(
            plan=accepted,
            bindings=bindings,
            results=results,
            runtime=context,
        )
        return {
            "plan": accepted,
            "materialized_new_task": True,
            "people_scalar_available": availability,
        }
    if not examined:
        return {
            "materialized_new_task": False,
            "people_scalar_available": availability,
        }
    return {"materialized_new_task": False, "people_scalar_available": availability}


async def discovery_settle_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Settle discovery candidates through the Binding Resolver (R39).

    Runs after ``materialize`` and before ``evaluate``. Rebuilds the
    ephemeral candidate index from checkpointed ``document.search`` results,
    validates each unsettled candidate through ``discovery.request_addition``
    (policy first: disabled discovery settles nothing), then hands the
    frozen ``BindingAdditionRequest`` plus the server-side candidate to the
    injected ``runtime.services.binding_resolver`` via its
    ``add_discovered_binding`` seam. The resolver — never the tools layer —
    revalidates current ACL/scope and creates/pins the exact discovered
    revision; an ACL denial (``DiscoveryDenied``) settles to nothing. New
    pins are validated (exact candidate identity, discovered/supporting
    role) and retention-leased before the checkpoint. Dormant (``{}``)
    when there is no plan/bindings, no search candidates, or no resolver
    wired: candidates simply stay unsettled and nothing is created.
    """
    context = node_context(runtime)
    state = normalize_complex_state(state)
    plan = state.get("plan")
    bindings = state.get("bindings")
    if plan is None or bindings is None:
        return {}
    search_ids = {
        task.task_id for task in plan.tasks if task.capability == "document.search"
    }
    if not search_ids:
        return {}
    registry = DiscoveryCandidateRegistry.from_results(
        result for result in tuple(state.get("task_results", ()))
        if result.task_id in search_ids
    )
    if len(registry) == 0:
        return {}
    resolver = context.services.binding_resolver
    if resolver is None:
        return {}
    policy = build_discovery_policy(context)
    existing = {
        (binding.document_id, binding.document_revision)
        for binding in bindings.bindings
    }
    # R44: the cap counts discovery-created bindings already pinned plus
    # pending additions in this pass. Only role "discovered" is counted:
    # "supporting" bindings may be user-bound, which must never consume
    # the discovery budget.
    remaining = policy.max_discovered_documents - sum(
        1 for binding in bindings.bindings if binding.role == "discovered"
    )
    settled: list[ScopedDocument] = []
    deferred: list[str] = []
    for candidate in registry.candidates():
        if (candidate.document_id, candidate.document_revision) in existing:
            continue
        if remaining <= 0:
            # Recorded, never added: a later pass still sees the candidate.
            deferred.append(str(candidate.candidate_id))
            continue
        if policy.allow_reference_discovery:
            role = "discovered"
        elif policy.allow_supporting_discovery:
            role = "supporting"
        else:
            continue
        try:
            request = request_addition(
                registry, candidate.candidate_id, role, policy=policy
            )
        except (DiscoveryDisabled, CandidateNotFound, InvalidCandidateRole):
            continue
        add = getattr(resolver, "add_discovered_binding", None)
        if add is None:
            raise ComplexResearchError(
                "binding resolver exposes no add_discovered_binding seam; "
                "refusing to settle discovery without an owning resolver"
            )
        try:
            created = add(request, candidate, context.capability_runtime)
            if inspect.isawaitable(created):
                created = await created
        except DiscoveryDenied:
            continue
        if not isinstance(created, ScopedDocument):
            raise ComplexResearchError(
                "binding resolver did not return a ScopedDocument for a "
                "discovery candidate; refusing an unowned pin"
            )
        if (
            created.document_id != candidate.document_id
            or created.document_revision != candidate.document_revision
        ):
            raise ComplexResearchError(
                "binding resolver did not pin the exact discovered revision; "
                "refusing a moved pin"
            )
        if created.role not in ("discovered", "supporting"):
            raise ComplexResearchError(
                "discovery settled a non-discovery role; the planner cannot "
                "create user targets"
            )
        settled.append(created)
        existing.add((candidate.document_id, candidate.document_revision))
        remaining -= 1
    if not settled and not deferred:
        return {}
    update: dict[str, Any] = {"discovery_deferred": tuple(deferred)}
    if settled:
        updated = bindings.model_copy(
            update={"bindings": bindings.bindings + tuple(settled)}
        )
        update["bindings"] = updated
    pairs: list[tuple[Any, Any]] = []
    for binding in settled:
        try:
            revision = UUID(str(binding.document_revision))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ComplexResearchError(
                f"settled binding {binding.binding_id} pins revision "
                f"{binding.document_revision!r}, not a revision id; refusing "
                "to checkpoint an unleasable pin"
            ) from exc
        pairs.append((revision, None))
    await _acquire_pairs(
        pairs,
        runtime=context,
        missing_message=(
            "discovery settled bindings but no retention-lease service is "
            "wired; refusing to checkpoint unleased pins"
        ),
    )
    return update


async def summarize_reduce_node(
    state: ComplexResearchState,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Drive the deterministic REDUCE through the EXISTING synthesis path (R47).

    Runs between ``evaluate`` and ``decide``. Active only for a summarize
    map plan whose checkpointed reduce spec survived with a ``sufficient``
    evaluation. It calls the existing ``synthesize_answer`` boundary with
    the map-task uses IN SPEC ORDER under the existing synthesis budget —
    so hydration, budget/overflow governance, drafting (extractive
    default), claim validation, and fresh-use leasing all run inside the
    framework's own synthesis path — then stores the validated draft in
    the existing ``answer_draft_channel`` handoff for the ground node. The
    agent never owns the reduce; no new capability and no new contract are
    involved. Dormant (``{}``) for every other plan and for non-sufficient
    evaluations. Mandatory: a missing channel or synthesis seam fails
    closed with a typed error instead of silently skipping the reduce.
    """
    context = node_context(runtime)
    state = normalize_complex_state(state)
    plan = state.get("plan")
    evaluation = state.get("evaluation")
    spec = state.get("reduce_spec")
    if plan is None or evaluation is None or spec is None:
        return {}
    if _evaluation_slot(evaluation, "status") != "sufficient":
        return {}
    if not str(getattr(plan, "plan_id", "")).startswith("summarize-"):
        return {}
    if not isinstance(spec, ReduceSpec):
        return {}
    if spec.mode != "extractive":
        raise ComplexResearchError(
            f"unknown reduce mode {spec.mode!r}; refusing to guess a reduction"
        )
    channel = getattr(context.services, "answer_draft_channel", None)
    if channel is None:
        raise ComplexResearchError(
            "summarize reduce needs the answer_draft_channel handoff; "
            "refusing to silently skip the mandatory reduce"
        )
    store = getattr(channel, "store_draft", None)
    if store is None:
        raise ComplexResearchError(
            "answer_draft_channel exposes no store_draft; refusing to "
            "reduce without the framework handoff"
        )
    results = tuple(state.get("task_results", ()))
    result_by_task = {result.task_id: result for result in results}
    ordered_refs = []
    for map_task_id in spec.map_task_ids:
        result = result_by_task.get(map_task_id)
        if result is None:
            raise ComplexResearchError(
                f"reduce spec names map task {map_task_id!r} with no "
                "checkpointed result; refusing a partial reduction"
            )
        ordered_refs.extend(result.evidence_uses)
    if not ordered_refs:
        raise ComplexResearchError(
            "reduce spec names map tasks with no evidence uses; refusing "
            "an empty reduction"
        )
    # The existing synthesis boundary owns hydration (budget/overflow),
    # drafting, validation, and leasing from here on.
    synthesis = await synthesize_answer(
        synthesis_input=SynthesisInput(
            semantic=state["semantic"],
            evaluation=evaluation,
            evidence_uses=tuple(ordered_refs),
        ),
        runtime=context,
        plan=plan,
        bindings=state["bindings"],
        budget=DEFAULT_SYNTHESIS_BUDGET,
    )
    store(
        context.capability_runtime.run_id,
        draft=synthesis.draft,
        evidence=synthesis.evidence,
    )
    return {}


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
    """Recommend the terminal step: replan only when advisable (R32/R37).

    A missing plan means the work type is out of pilot scope: return the typed
    ``COMPLEX_RESEARCH_UNAVAILABLE`` marker. Otherwise the evaluation stands
    as-is; the supervisor routes ``sufficient`` to synthesis and everything
    else to the finalizer (R4). Evaluator owns sufficiency/contradictions;
    the ``replan`` edge (owned by ``_decide_branch``) fires only when
    ``replan_advisable`` holds (coverage gaps or no-evidence recovery).
    Discovery stays policy-gated.
    """
    node_context(runtime)
    if state.get("plan") is None:
        work_type = _work_type_of(state)
        return {
            "unavailable": ComplexResearchUnavailable(
                code=COMPLEX_RESEARCH_UNAVAILABLE,
                reason=(
                    f"work type {work_type!r} has no complex skill policy "
                    "(compare and large/iterative summarize are supported)"
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


def _materialize_branch(state: ComplexResearchState) -> str:
    """R26 routing seam: re-execute exactly when T2 was just appended.

    ``materialized_new_task`` is True only on the pass that appended the
    dependent, so the scheduler dispatches the checkpointed T2 unchanged and
    the loop always terminates (appends are budget-bounded; a pass with no
    append routes to ``settle`` even when tasks remain undispatched, e.g.
    after a deadline truncation).
    """
    if bool(state.get("materialized_new_task", False)):
        return "execute"
    return "settle"


def _decide_branch(state: ComplexResearchState) -> str:
    """Decide routing seam (R32): replan only for advisable coverage gaps."""
    if replan_advisable(state):
        return "replan"
    return "finalize"


def _add_complex_edges(graph: StateGraph) -> None:
    graph.set_entry_point("plan")
    graph.add_edge("plan", "validate_checkpoint")
    graph.add_edge("validate_checkpoint", "execute")
    graph.add_edge("execute", "materialize")
    graph.add_conditional_edges(
        "materialize", _materialize_branch, {"execute": "execute", "settle": "settle"}
    )
    graph.add_edge("settle", "evaluate")
    graph.add_edge("evaluate", "reduce")
    graph.add_edge("reduce", "decide")
    graph.add_conditional_edges(
        "decide", _decide_branch, {"replan": "replan", "finalize": "finalize"}
    )
    graph.add_edge("replan", "validate_checkpoint")
    graph.add_edge("finalize", END)


def _build_complex_research_graph() -> StateGraph:
    """Assemble the subgraph nodes and edges without compiling.

    The compiled subgraph is produced by
    :func:`build_complex_research_subgraph` (no saver: it inherits the
    supervisor's). Tests compile this builder with an explicit saver to
    audit the subgraph's own checkpoints. This module never names a saver
    implementation and never passes one at compile time.
    """
    graph = StateGraph(ComplexResearchState, context_schema=GraphRuntimeContext)
    graph.add_node("plan", plan_node)
    graph.add_node("validate_checkpoint", validate_checkpoint_node)
    graph.add_node("execute", complex_execute_node)
    graph.add_node("materialize", people_document_materialize_node)
    graph.add_node("settle", discovery_settle_node)
    graph.add_node("evaluate", complex_evaluate_node)
    graph.add_node("reduce", summarize_reduce_node)
    graph.add_node("replan", replan_node)
    graph.add_node("decide", decide_node)
    graph.add_node("finalize", finalize_node)
    _add_complex_edges(graph)
    return graph


def build_complex_research_subgraph() -> Any:
    """Build the adaptive planning boundary as a checkpointed subgraph.

    Compiled WITHOUT a checkpointer so it inherits the supervisor's saver
    when attached as the ``complex_boundary`` node (a shadow run's isolated
    saver is inherited the same way). It never opens its own production or
    shadow saver, which keeps shadow execution isolated.
    """
    return _build_complex_research_graph().compile()


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
        route_decision=state["route_decision"],
        plan=execution.plan,
        task_results=execution.task_results,
        evaluation=execution.evidence_evaluation,
        # R36: production entry uses the SAME settings-driven limits the
        # budget view enforces — never a hardcoded constant.
        replans_remaining=V2ResearchLimits.from_settings().max_replans,
        reduce_spec=None,
        discovery_deferred=(),
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

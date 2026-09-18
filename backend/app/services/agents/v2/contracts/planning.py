"""Task planning, completion criteria, and research planning input (spec §9, §13, §16).

``TaskPlan`` is the canonical factual execution owner: ``target_id → TargetUnit``,
``task_id → TaskSpec``, and ``binding_id → ScopedDocument``. Completion criteria
belong only to logical ``TargetUnit``s, so they never repeat the target ID.
``DiscoveryPolicy`` and ``ResearchBudgetView`` are planner inputs; the budget view
is ephemeral and never persisted.
"""
from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from .base import ContractModel, ContractVersion
from .binding import DocumentBindingSet
from .capability import CapabilityDescriptor, CapabilityInput, DocumentSearchInput
from .evaluation import EvidenceEvaluation
from .evidence import EvidenceUseRef
from .execution import TaskExecutionSummary
from .intent import IntentAnalysis
from .locators import ContentLocator
from .routing import QueryAnalysis
from .semantic import SemanticContext
from .validation_support import ContractValidationError
from ..discovery_bootstrap.contracts import (
    DiscoveryCheckpoint,
    ResearchTargetSelection,
)


class CoverageCriterion(ContractModel):
    """Spec §13.1: deterministic coverage criterion for a target unit."""

    kind: Literal["coverage"]
    minimum_status: Literal["read_partial", "read_complete"] = "read_complete"
    allow_partial_reason: str | None = None


class SemanticCriterion(ContractModel):
    """Spec §13.1: trusted-policy-only target-level judgment objective."""

    kind: Literal["semantic"]
    criterion_id: str
    description: str


CompletionCriterion = Annotated[
    CoverageCriterion | SemanticCriterion,
    Field(discriminator="kind"),
]


class TargetUnit(ContractModel):
    """Spec §9: the logical document/read requirement of one factual route.

    ``binding_id`` resolves document identity, pinned revision, and role from the
    binding set, so document details are never copied here.
    """

    target_id: str
    binding_id: str
    requested_locator: ContentLocator
    completion_criteria: tuple[CompletionCriterion, ...]


class InitialTaskOrigin(ContractModel):
    """Spec §13.2: task emitted by the initial fast or complex plan."""

    kind: Literal["initial"]


class ReplanTaskOrigin(ContractModel):
    """Spec §13.2: task appended by a bounded replan.

    ``evidence_use_ids`` preserves the triggering run/task/target/purpose context;
    bare evidence IDs remain reserved for content lineage.
    """

    kind: Literal["replan"]
    reason: str
    task_ids: tuple[str, ...]
    evidence_use_ids: tuple[UUID, ...]


class DiscoveryProbeTaskOrigin(ContractModel):
    """Discovery spec §13: one accepted bootstrap search probe owned by the plan."""

    kind: Literal["discovery_probe"]
    probe_id: str
    slot_id: str
    round: Literal[1, 2]


class DiscoveryExpansionTaskOrigin(ContractModel):
    """Discovery spec §13: factual task appended by governed plan expansion."""

    kind: Literal["discovery_expansion"]
    source_search_task_ids: tuple[str, ...]
    target_slot_ids: tuple[str, ...]
    selected_aggregate_ids: tuple[UUID, ...]


TaskOrigin = Annotated[
    InitialTaskOrigin
    | ReplanTaskOrigin
    | DiscoveryProbeTaskOrigin
    | DiscoveryExpansionTaskOrigin,
    Field(discriminator="kind"),
]


class TaskSpec(ContractModel):
    """Spec §13.2: one operation definition owned by the plan."""

    task_id: str
    capability: str
    task_objective: str
    input: CapabilityInput
    depends_on: tuple[str, ...] = ()
    origin: TaskOrigin


class TaskPlan(ContractModel):
    """Spec §3/§13.2: the checkpointed authoritative factual execution owner."""

    contract_version: ContractVersion
    plan_id: str
    goal: str
    target_units: tuple[TargetUnit, ...]
    tasks: tuple[TaskSpec, ...]


class DiscoveryPolicy(ContractModel):
    """Spec §16: authorized expansion semantics for research/discovery."""

    allow_reference_discovery: bool
    allow_supporting_discovery: bool
    max_discovered_documents: int


class ResearchBudgetView(ContractModel):
    """Spec §16: ephemeral runtime-derived planner budget; never persisted."""

    max_tasks_remaining: int
    max_replans_remaining: int
    max_parallel_branches: int


class DiscoveryBudgetView(ContractModel):
    """Discovery spec §11.3: separate bootstrap probe budget; never persisted.

    Discovery probes and factual research tasks share one plan lineage but
    keep separate counters: this view is the probe/round side.
    """

    probes_remaining: int
    rounds_remaining: int
    top_k: int


class ResearchPlanningInput(ContractModel):
    """Spec §16: the planner/replanner request.

    Initial planning receives no current plan, outcomes, evidence uses, or
    evaluation; replanning receives the append-only plan, one summary per
    attempted task, current validated use refs, and the latest evaluation.
    """

    semantic: SemanticContext
    bindings: DocumentBindingSet
    query_analysis: QueryAnalysis
    capability_catalog: tuple[CapabilityDescriptor, ...]
    discovery_policy: DiscoveryPolicy
    budget: ResearchBudgetView
    current_plan: TaskPlan | None = None
    task_outcomes: tuple[TaskExecutionSummary, ...] = ()
    prior_evidence_uses: tuple[EvidenceUseRef, ...] = ()
    prior_evaluation: EvidenceEvaluation | None = None
    target_selection: ResearchTargetSelection | None = None
    discovery_checkpoint: DiscoveryCheckpoint | None = None
    #: Checkpointed whole-request intent analysis (multi-intent spec §33.3):
    #: threaded parent → child so the planner sees every detected intent and
    #: dependency without re-classifying. ``None`` on the flag-off path and on
    #: legacy checkpoints; planning behavior without it is unchanged.
    intent_analysis: IntentAnalysis | None = None


# Spec §3: models embedding discriminated-union aliases rebuild explicitly.
TargetUnit.model_rebuild()
TaskSpec.model_rebuild()
TaskPlan.model_rebuild()


def _normalize_probe_query(query: str) -> str:
    """Whitespace/casefold normalization for probe-query equality.

    Duplicated (not imported) from ``discovery_bootstrap.validation`` so the
    data/planning layer never reaches back into the validation layer; both
    copies must stay byte-identical in behavior (spec §18 dependency
    direction).
    """

    return " ".join(query.split()).casefold()


def _require_provable_probe_origin(
    task: TaskSpec,
    origin: DiscoveryProbeTaskOrigin,
    discovery_checkpoint: DiscoveryCheckpoint | None,
    owned_probe_ids: set[str],
) -> None:
    """Prove a probe-origin task is exactly one accepted checkpoint probe.

    Every requirement checkable at the data layer is re-proven here: an
    origin label alone never grants free budget capacity, so any malformed
    or spoofed ``DiscoveryProbeTaskOrigin`` fails closed instead of counting
    as exempt (or as ordinary factual work).
    """

    if discovery_checkpoint is None:
        raise ContractValidationError(
            f"task {task.task_id} carries a discovery probe origin but no "
            "DiscoveryCheckpoint exists"
        )
    if task.capability != "document.search" or not isinstance(
        task.input, DocumentSearchInput
    ):
        raise ContractValidationError(
            f"discovery probe task {task.task_id} must be document.search "
            "with DocumentSearchInput"
        )
    if task.input.person_identifier is not None:
        raise ContractValidationError(
            f"discovery probe task {task.task_id} must not carry a "
            "materialized person scalar"
        )
    matches = [
        probe
        for probe in discovery_checkpoint.accepted_probes
        if probe.probe_id == origin.probe_id
    ]
    if len(matches) != 1:
        raise ContractValidationError(
            f"discovery probe task {task.task_id} must match exactly one "
            f"accepted probe ({len(matches)} matched {origin.probe_id!r})"
        )
    probe = matches[0]
    if origin.slot_id != probe.slot_id or origin.round != probe.round:
        raise ContractValidationError(
            f"discovery probe task {task.task_id} origin does not match "
            f"probe {probe.probe_id} slot/round"
        )
    if _normalize_probe_query(task.input.query) != _normalize_probe_query(
        probe.query
    ):
        raise ContractValidationError(
            f"discovery probe task {task.task_id} query does not match "
            f"probe {probe.probe_id}"
        )
    if probe.probe_id in owned_probe_ids:
        raise ContractValidationError(
            f"probe {probe.probe_id} is owned by multiple plan tasks"
        )
    owned_probe_ids.add(probe.probe_id)


def build_research_budget_view(
    plan: TaskPlan,
    *,
    discovery_checkpoint: DiscoveryCheckpoint | None,
    max_tasks: int,
    max_replans: int,
    max_parallel_branches: int,
    total_task_limit: int,
) -> ResearchBudgetView:
    """Discovery spec §11.3: pure research budget for an existing plan.

    Only factual tasks consume ``max_tasks``; a
    ``DiscoveryProbeTaskOrigin`` task is budget-exempt ONLY after the exact
    probe-origin proof succeeds — every malformed or spoofed origin raises
    ``ContractValidationError`` before any budget is constructed.
    ``max_tasks_remaining`` is bounded by BOTH the remaining factual budget
    (``max_tasks - factual_task_count``) and the remaining total plan
    capacity (``total_task_limit`` counts every task, probes included), each
    clamped at zero.
    """

    factual_task_count = 0
    owned_probe_ids: set[str] = set()
    for task in plan.tasks:
        origin = task.origin
        if isinstance(origin, DiscoveryProbeTaskOrigin):
            _require_provable_probe_origin(
                task, origin, discovery_checkpoint, owned_probe_ids
            )
            continue
        factual_task_count += 1
    factual_remaining = max_tasks - factual_task_count
    total_remaining = total_task_limit - len(plan.tasks)
    return ResearchBudgetView(
        max_tasks_remaining=max(0, min(factual_remaining, total_remaining)),
        max_replans_remaining=max(0, max_replans),
        max_parallel_branches=max_parallel_branches,
    )

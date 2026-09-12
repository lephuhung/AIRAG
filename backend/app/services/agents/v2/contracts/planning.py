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
from .capability import CapabilityDescriptor, CapabilityInput
from .evaluation import EvidenceEvaluation
from .evidence import EvidenceUseRef
from .execution import TaskExecutionSummary
from .locators import ContentLocator
from .routing import QueryAnalysis
from .semantic import SemanticContext


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


TaskOrigin = Annotated[InitialTaskOrigin | ReplanTaskOrigin, Field(discriminator="kind")]


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


# Spec §3: models embedding discriminated-union aliases rebuild explicitly.
TargetUnit.model_rebuild()
TaskSpec.model_rebuild()
TaskPlan.model_rebuild()

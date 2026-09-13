"""Summarize skill policy: bounded single-target summary (Phase 3, Task 5).

Framework-neutral source of truth, mirroring ``skills/compare/policy.py``: given
an ephemeral ``ResearchPlanningInput`` the policy proposes the initial
``TaskPlan`` for a ``summarize`` request — one bounded read of one bound
document, no dependencies, initial origin. It never executes a capability,
never checkpoints, and never reaches past validation: the returned plan still
passes through ``validate_task_plan`` in the subgraph ``validate_checkpoint``
node, the supervisor saver, and the shared scheduler.

Routing (amendment §6): the bounded case stays on this single read (fast when
the router chooses it); the large/iterative case uses the complex planner with
a deterministic map/reduce workflow proposed as governed work. The evaluator
owns coverage; synthesis/grounding stay outside agent authority. There is NO
``summary_agent``.

Pilot scope (initial-plan-only):

- exactly one bound document with role ``target`` — any other arity or role
  fails closed;
- deterministic map/reduce proposal: every ``SemanticContext.section_refs``
  entry carrying a ``structure_node_id`` names one map chunk of the single
  target document (in ``ref_id`` order), each becoming a ``TargetUnit`` with
  the exact ``SectionLocator`` coordinate plus one ``section.read`` map task;
  the reduce step is governed synthesis downstream, never a task here. With
  no named sections the plan falls back to a single whole-document
  ``DocumentLocator`` (``document.read``) — the bounded shape the fast path
  also serves;
- map fan-out is bounded by ``ResearchBudgetView.max_tasks_remaining``: more
  named sections than task budget fails closed instead of silently dropping
  content;
- every used read capability must be present in the request-scoped
  capability catalog;
- discovery/replan are rejected here (the replan loop owns appends): only
  ``summarize`` is supported, every other work type fails closed so the caller
  can return the typed ``COMPLEX_RESEARCH_UNAVAILABLE`` boundary instead of a
  fabricated plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...contracts.binding import ScopedDocument
from ...contracts.capability import DocumentReadInput, SectionReadInput
from ...contracts.locators import DocumentLocator, SectionLocator
from ...contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    ResearchPlanningInput,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from ...contracts.semantic import SemanticContext
from ...contracts.validation import ContractValidationError, validate_task_plan

__all__ = [
    "SUMMARIZE_WORK_TYPE",
    "MAX_SUMMARY_TARGETS",
    "ReduceMode",
    "ReduceSpec",
    "SummarizeWorkflow",
    "build_summarize_plan",
    "build_summarize_workflow",
    "supports_work_type",
]

#: The only work type this pilot plans. Everything else fails closed.
SUMMARIZE_WORK_TYPE = "summarize"

#: Exact single-target bounded summary: one target side, one bounded read.
MAX_SUMMARY_TARGETS = 1

_SUMMARY_TARGET_ID = "t1"

#: The only deterministic reduce mode (R43): ordered extractive reduction
#: over the admitted map evidence, executed by the framework's existing
#: ``build_extractive_draft`` — never by the agent, never a new capability.
ReduceMode = Literal["extractive"]


@dataclass(frozen=True)
class ReduceSpec:
    """Explicit deterministic REDUCE specification (implementation-only).

    Not a frozen contract: it is the policy's deterministic workflow
    descriptor consumed by the subgraph's reduce node. ``map_task_ids`` is
    the exact ordered map-task lineage the reduction must cover; ``mode``
    selects the existing framework reduction (extractive only).
    """

    map_task_ids: tuple[str, ...]
    mode: ReduceMode = "extractive"


@dataclass(frozen=True)
class SummarizeWorkflow:
    """Deterministic map/reduce workflow descriptor (implementation-only).

    ``plan`` carries the ordered MAP read tasks (deterministic targets and
    order); ``reduce`` carries the explicit reduce stage the framework's
    reduce node drives through the existing evaluate → synthesis → grounding
    boundaries. The agent never owns the reduce.
    """

    plan: TaskPlan
    reduce: ReduceSpec


def supports_work_type(work_type: str) -> bool:
    """True only for the ``summarize`` pilot; ``compare``/``evaluate`` are out."""
    return work_type == SUMMARIZE_WORK_TYPE


def _summary_target(bindings: object) -> ScopedDocument:
    """Return the single bound ``target`` document in deterministic order.

    Fails closed unless the run binds exactly one document with role
    ``target``: the planner cannot autonomously add targets or reinterpret
    roles.
    """
    ordered = sorted(bindings.bindings, key=lambda binding: binding.binding_id)  # type: ignore[union-attr]
    if len(ordered) != MAX_SUMMARY_TARGETS:
        raise ContractValidationError(
            f"summarize requires exactly {MAX_SUMMARY_TARGETS} bound document, "
            f"got {len(ordered)}; refusing to fabricate summary targets"
        )
    target = ordered[0]
    if target.role != "target":
        raise ContractValidationError(
            f"summarize requires a 'target' document (got role {target.role!r}); "
            "refusing to reinterpret roles"
        )
    return target


def _map_chunks(semantic: SemanticContext) -> tuple[str, ...]:
    """Structure-node IDs naming map chunks of the single target, in order.

    With exactly one bound document there is no side ambiguity: every
    section reference carrying a coordinate names one map chunk.
    Deterministic ``ref_id`` order wins.
    """
    matches = sorted(
        (
            reference
            for reference in semantic.section_refs
            if reference.structure_node_id is not None
        ),
        key=lambda reference: reference.ref_id,
    )
    return tuple(
        reference.structure_node_id
        for reference in matches
        if reference.structure_node_id is not None
    )


def _map_task(
    task_id: str, target_id: str, binding: ScopedDocument, structure_node_id: str
) -> TaskSpec:
    """One map read: the exact section coordinate, nothing else."""
    return TaskSpec(
        task_id=task_id,
        capability="section.read",
        task_objective=(
            f"Read section {structure_node_id} for summary "
            f"(document {binding.document_id})"
        ),
        input=SectionReadInput(kind="section.read", target_ids=(target_id,)),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def _whole_document_task(binding: ScopedDocument) -> TaskSpec:
    """The bounded fallback: one whole-document read."""
    return TaskSpec(
        task_id="T1",
        capability="document.read",
        task_objective=(
            "Read the summary target "
            f"(document {binding.document_id})"
        ),
        input=DocumentReadInput(kind="document.read", target_ids=(_SUMMARY_TARGET_ID,)),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def _require_read_capabilities(
    planning_input: ResearchPlanningInput, tasks: tuple[TaskSpec, ...]
) -> None:
    """Fail closed when the runtime catalog cannot serve the planned reads."""
    names = {entry.name for entry in planning_input.capability_catalog}
    for task in tasks:
        if task.capability not in names:
            raise ContractValidationError(
                f"summarize requires {task.capability!r} in the request-scoped "
                "capability catalog; refusing to plan an undispatchable read"
            )


def build_summarize_workflow(planning_input: ResearchPlanningInput) -> SummarizeWorkflow:
    """Propose the deterministic map/reduce workflow (R43).

    Returns the ordered MAP plan plus the explicit REDUCE specification the
    framework's reduce node drives. Raises :class:`ContractValidationError`
    for any out-of-pilot input (unsupported work type, wrong arity/role,
    missing read capability, or a plan that fails frozen validation) — the
    caller turns that into the typed unavailable boundary, never a partial
    plan.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        raise ContractValidationError(
            "summarize skill cannot plan work type "
            f"{planning_input.query_analysis.work_type!r}; out of pilot scope"
        )
    target = _summary_target(planning_input.bindings)
    chunks = _map_chunks(planning_input.semantic)
    if chunks:
        # Deterministic map fan-out: one target + one section read per
        # named chunk. Bounded by the task budget: silently dropping a
        # named chunk would corrupt the summary, so excess fails closed.
        if len(chunks) > planning_input.budget.max_tasks_remaining:
            raise ContractValidationError(
                f"summarize names {len(chunks)} section(s) but the task "
                f"budget allows {planning_input.budget.max_tasks_remaining}; "
                "refusing a partial map"
            )
        units: list[TargetUnit] = []
        tasks: list[TaskSpec] = []
        for position, structure_node_id in enumerate(chunks, start=1):
            target_id = f"s{position}"
            units.append(
                TargetUnit(
                    target_id=target_id,
                    binding_id=target.binding_id,
                    requested_locator=SectionLocator(
                        kind="section", structure_node_id=structure_node_id
                    ),
                    completion_criteria=(CoverageCriterion(kind="coverage"),),
                )
            )
            tasks.append(
                _map_task(f"T{position}", target_id, target, structure_node_id)
            )
        plan_id = f"summarize-{target.binding_id}-map{len(chunks)}"
    else:
        units = [
            TargetUnit(
                target_id=_SUMMARY_TARGET_ID,
                binding_id=target.binding_id,
                requested_locator=DocumentLocator(kind="document"),
                completion_criteria=(CoverageCriterion(kind="coverage"),),
            )
        ]
        tasks = [_whole_document_task(target)]
        plan_id = f"summarize-{target.binding_id}"
    _require_read_capabilities(planning_input, tuple(tasks))
    plan = TaskPlan(
        contract_version="2.0",
        plan_id=plan_id,
        goal=planning_input.semantic.contextualized_query,
        target_units=tuple(units),
        tasks=tuple(tasks),
    )
    validate_task_plan(plan, planning_input.bindings)
    return SummarizeWorkflow(
        plan=plan,
        reduce=ReduceSpec(
            map_task_ids=tuple(task.task_id for task in tasks),
            mode="extractive",
        ),
    )


def build_summarize_plan(planning_input: ResearchPlanningInput) -> TaskPlan:
    """Propose the initial one-target bounded summary plan (map stage only).

    Thin wrapper over :func:`build_summarize_workflow` for callers that only
    need the executable plan; the reduce stage stays with the workflow.
    """
    return build_summarize_workflow(planning_input).plan

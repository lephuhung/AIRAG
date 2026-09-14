"""Evaluate/compliance skill policy: governed assessment evidence (Task 12).

Framework-neutral source of truth: given an ephemeral ``ResearchPlanningInput``
the policy proposes the initial ``TaskPlan`` for an ``evaluate`` request — one
bounded read per distinct bound document, no dependencies (parallel safe
reads), initial origin. The compliance judgment itself is never a capability:
it comes from the existing governed pipeline over the gathered evidence (the
shared ``evaluate_evidence`` verdict, bounded append-only replan, grounded
synthesis). This skill only ensures the assessment starts from
revision-pinned reads of already-bound documents.

It never executes a capability, never checkpoints, and never reaches past
validation: the returned plan still passes through ``validate_task_plan`` in
the subgraph ``validate_checkpoint`` node, the supervisor saver, and the
shared scheduler.

Pilot scope (initial-plan-only):

- ``evaluate`` work type only; every other work type fails closed;
- at least one distinct settled-role binding; an unbound compliance question
  fails closed so the governed model path (when wired with open discovery)
  or the typed unavailable boundary still owns it — never a fabricated
  assessment target;
- at most ``MAX_EVALUATE_TARGETS`` bounded reads and no more than the
  remaining task budget;
- each side's ``requested_locator`` derives from the query semantics exactly
  like the compare skill (section coordinate when named, whole-document
  fallback) on the shared ``bounded_reads`` mechanics;
- every used read capability must be present in the request-scoped capability
  catalog;
- discovery/replan are rejected here: only bounded reads over already-bound
  documents are proposed.
"""
from __future__ import annotations

from ...contracts.planning import (
    ResearchPlanningInput,
    TaskPlan,
)
from ...contracts.validation import ContractValidationError, validate_task_plan
from ..bounded_reads import (
    coverage_unit,
    locator_for,
    plannable_bindings,
    read_task,
    require_read_capabilities,
)

__all__ = [
    "EVALUATE_WORK_TYPE",
    "MAX_EVALUATE_TARGETS",
    "build_evaluate_plan",
    "covers_input",
    "supports_work_type",
]

#: The only work type this pilot plans. Everything else fails closed.
EVALUATE_WORK_TYPE = "evaluate"

#: Hard ceiling: a single initial proposal stays a bounded fan-out of reads,
#: never an open-ended crawl.
MAX_EVALUATE_TARGETS = 8


def supports_work_type(work_type: str) -> bool:
    """True only for the ``evaluate`` pilot; ``compare`` and the rest are out."""
    return work_type == EVALUATE_WORK_TYPE


def covers_input(planning_input: ResearchPlanningInput) -> bool:
    """True when the deterministic skill owns this input (Task 12 planner seam).

    Ownership is work-type plus intake: at least one settled-role binding
    within the hard ceiling. A covered input whose build still refuses
    (missing catalog capability, over task budget) is a deliberate final
    refusal — the model path must not second-guess it. Unbound compliance
    questions stay uncovered so the governed model path still owns them.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        return False
    count = len(plannable_bindings(planning_input.bindings))
    return 1 <= count <= MAX_EVALUATE_TARGETS


def build_evaluate_plan(planning_input: ResearchPlanningInput) -> TaskPlan:
    """Propose the initial bounded compliance-assessment read plan.

    Raises :class:`ContractValidationError` for any out-of-pilot input
    (unsupported work type, no bound documents, more bound documents than
    the hard ceiling or the remaining task budget allows, missing read
    capability, or a plan that fails frozen validation) — the caller turns
    that into the governed model path (when a planner is wired and the input
    is open) or the typed unavailable boundary, never a partial plan and
    never a dispatch.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        raise ContractValidationError(
            "evaluate skill cannot plan work type "
            f"{planning_input.query_analysis.work_type!r}; out of pilot scope"
        )
    sides = plannable_bindings(planning_input.bindings)
    if not sides:
        raise ContractValidationError(
            "evaluate needs at least one bound document to assess, got none; "
            "refusing to fabricate a compliance target"
        )
    if len(sides) > MAX_EVALUATE_TARGETS:
        raise ContractValidationError(
            f"evaluate plans at most {MAX_EVALUATE_TARGETS} bounded reads, "
            f"got {len(sides)} bound documents; refusing an open-ended fan-out"
        )
    if len(sides) > planning_input.budget.max_tasks_remaining:
        raise ContractValidationError(
            f"evaluate needs {len(sides)} read task(s) but the task budget "
            f"allows {planning_input.budget.max_tasks_remaining}; refusing a "
            "plan that cannot fit the remaining budget"
        )
    semantic = planning_input.semantic
    locators = [locator_for(side, semantic) for side in sides]
    units = tuple(
        coverage_unit(f"t{index + 1}", side, locator)
        for index, (side, locator) in enumerate(zip(sides, locators))
    )
    tasks = tuple(
        read_task(
            f"T{index + 1}",
            f"t{index + 1}",
            side,
            locator,
            objective_noun="compliance",
        )
        for index, (side, locator) in enumerate(zip(sides, locators))
    )
    require_read_capabilities(
        {entry.name for entry in planning_input.capability_catalog},
        tasks,
        owner="evaluate",
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="evaluate-" + "-".join(side.binding_id for side in sides),
        goal=planning_input.semantic.contextualized_query,
        target_units=units,
        tasks=tasks,
    )
    validate_task_plan(plan, planning_input.bindings)
    return plan

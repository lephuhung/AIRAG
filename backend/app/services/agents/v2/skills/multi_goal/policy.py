"""Multi-goal skill policy: bounded parallel reads for multi-target work (Task 12).

Framework-neutral source of truth, mirroring ``skills/compare/policy.py``:
given an ephemeral ``ResearchPlanningInput`` the policy proposes the initial
``TaskPlan`` for a ``multi_goal`` request — one bounded read per distinct
bound document, no dependencies (parallel safe reads), initial origin. It
never executes a capability, never checkpoints, and never reaches past
validation: the returned plan still passes through ``validate_task_plan`` in
the subgraph ``validate_checkpoint`` node, the supervisor saver, and the
shared scheduler.

Pilot scope (initial-plan-only):

- ``multi_goal`` work type only; every other work type fails closed;
- at least ``MIN_MULTI_GOAL_TARGETS`` distinct settled-role bindings
  (mirroring the routing arity where two sides are ``compare`` and three or
  more are ``multi_goal``); fewer bound documents fail closed so the governed
  model path (or the typed unavailable boundary when unwired) still owns the
  input — never a fabricated target;
- at most ``MAX_MULTI_GOAL_TARGETS`` bounded reads and no more than the
  remaining task budget; larger inputs fail closed instead of leasing a plan
  that cannot fit;
- each side's ``requested_locator`` derives from the query semantics exactly
  like the compare skill (section coordinate when named, whole-document
  fallback); the shared ``bounded_reads`` mechanics keep the three Task-12
  skills on one convention;
- every used read capability must be present in the request-scoped capability
  catalog;
- discovery/replan are rejected here: only bounded reads over already-bound
  documents are proposed, so the caller returns the typed
  ``COMPLEX_RESEARCH_UNAVAILABLE`` boundary instead of a fabricated plan.
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
    "MAX_MULTI_GOAL_TARGETS",
    "MIN_MULTI_GOAL_TARGETS",
    "MULTI_GOAL_WORK_TYPE",
    "build_multi_goal_plan",
    "covers_input",
    "supports_work_type",
]

#: The only work type this pilot plans. Everything else fails closed.
MULTI_GOAL_WORK_TYPE = "multi_goal"

#: Routing arity mirror: two bound sides are ``compare``; multi-goal starts
#: at three distinct bound documents.
MIN_MULTI_GOAL_TARGETS = 3

#: Hard ceiling even when the remaining budget is larger: a single initial
#: proposal stays a bounded fan-out of reads, never an open-ended crawl.
MAX_MULTI_GOAL_TARGETS = 8


def supports_work_type(work_type: str) -> bool:
    """True only for the ``multi_goal`` pilot; ``compare`` and the rest are out."""
    return work_type == MULTI_GOAL_WORK_TYPE


def covers_input(planning_input: ResearchPlanningInput) -> bool:
    """True when the deterministic skill owns this input (Task 12 planner seam).

    Ownership is work-type plus arity only: a covered input whose build still
    refuses (missing catalog capability, over task budget) is a deliberate
    final refusal — the model path must not second-guess it. Below-intake
    inputs (fewer bound documents than the routing arity) stay uncovered so
    the governed model path still owns them.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        return False
    count = len(plannable_bindings(planning_input.bindings))
    return MIN_MULTI_GOAL_TARGETS <= count <= MAX_MULTI_GOAL_TARGETS


def build_multi_goal_plan(planning_input: ResearchPlanningInput) -> TaskPlan:
    """Propose the initial bounded multi-goal read plan.

    Raises :class:`ContractValidationError` for any out-of-pilot input
    (unsupported work type, fewer than three distinct bound documents, more
    bound documents than the hard ceiling or the remaining task budget
    allows, missing read capability, or a plan that fails frozen validation)
    — the caller turns that into the governed model path (when a planner is
    wired and the input is below intake) or the typed unavailable boundary,
    never a partial plan and never a dispatch.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        raise ContractValidationError(
            "multi_goal skill cannot plan work type "
            f"{planning_input.query_analysis.work_type!r}; out of pilot scope"
        )
    sides = plannable_bindings(planning_input.bindings)
    if len(sides) < MIN_MULTI_GOAL_TARGETS:
        raise ContractValidationError(
            f"multi_goal requires at least {MIN_MULTI_GOAL_TARGETS} distinct "
            f"bound documents, got {len(sides)}; refusing to fabricate goals"
        )
    if len(sides) > MAX_MULTI_GOAL_TARGETS:
        raise ContractValidationError(
            f"multi_goal plans at most {MAX_MULTI_GOAL_TARGETS} bounded reads, "
            f"got {len(sides)} bound documents; refusing an open-ended fan-out"
        )
    if len(sides) > planning_input.budget.max_tasks_remaining:
        raise ContractValidationError(
            f"multi_goal needs {len(sides)} read task(s) but the task budget "
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
            objective_noun="multi-goal",
        )
        for index, (side, locator) in enumerate(zip(sides, locators))
    )
    require_read_capabilities(
        {entry.name for entry in planning_input.capability_catalog},
        tasks,
        owner="multi_goal",
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="multi-goal-" + "-".join(side.binding_id for side in sides),
        goal=planning_input.semantic.contextualized_query,
        target_units=units,
        tasks=tasks,
    )
    validate_task_plan(plan, planning_input.bindings)
    return plan

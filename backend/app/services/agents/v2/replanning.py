"""Bounded append-only replan validation, runtime side (Phase 3, Task 5).

Thin deterministic layer that WRAPS the frozen
:func:`contracts.validation.validate_replan` (R31): the frozen validator owns
every append-only/budget/discovery rule and is never redefined or forked here.
This module adds only the runtime-only checks the frozen layer cannot express:

- capability membership in the CURRENT request-scoped catalog (registry view
  intersected with current ``allowed_capabilities``/``can_read_people``);
- cancellation/deadline-safe behavior (a cancelled run or a passed deadline
  never validates a replan);
- entry fan-out width against ``max_parallel_branches`` (the planner chooses
  fan-out; the scheduler still enforces).

It raises typed errors and never mutates: the caller node checkpoints the
returned plan before the next execute step.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from .contracts.planning import (
    DiscoveryPolicy,
    ResearchBudgetView,
    TaskPlan,
    TaskSpec,
    TaskExecutionSummary,
)
from .contracts.state import GraphRuntimeContext
from .contracts.validation import ContractValidationError, validate_replan

__all__ = [
    "ReplanRejected",
    "append_replan_tasks",
    "check_replan_dispatchable",
    "entry_fanout_width",
    "validate_runtime_replan",
]


class ReplanRejected(ValueError):
    """A replan proposal failed a runtime-only check (catalog/deadline/width)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def check_replan_dispatchable(runtime: GraphRuntimeContext) -> None:
    """Fail closed when the run can no longer dispatch work.

    A pending cancellation propagates (never converts into a plan), and a
    passed ``deadline_at`` rejects the replan instead of validating work that
    the scheduler could never dispatch.
    """
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is not None and task.cancelling() > 0:
        raise asyncio.CancelledError()
    deadline = runtime.capability_runtime.deadline_at
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    if _now() >= deadline:
        raise ReplanRejected(
            "the run deadline has passed; refusing to validate a replan "
            "the scheduler could never dispatch"
        )


def _catalog_names(runtime: GraphRuntimeContext) -> frozenset[str] | None:
    """Current request-scoped catalog names, or ``None`` when unwired."""
    services = runtime.services
    registry = getattr(services, "capability_registry", None) if services else None
    if registry is None:
        return None
    try:
        names = registry.capability_names()
    except AttributeError:
        try:
            names = frozenset(entry.name for entry in registry.catalog())
        except (AttributeError, TypeError):
            return None
    try:
        return frozenset(names)
    except TypeError:
        return None


def _require_capabilities_within_current_runtime_catalog(
    new_tasks: tuple[TaskSpec, ...], runtime: GraphRuntimeContext
) -> None:
    """Every NEW task must stay servable by the CURRENT runtime catalog.

    The frozen validator owns shape/append-only/budget/discovery; this is the
    runtime intersection (registry view ∩ current permissions) that only the
    live request can answer. Completed tasks are immutable history — the
    scheduler never redispatches them, so only appended tasks are checked.
    A new task whose capability left the catalog — or was never permitted —
    rejects the whole replan instead of checkpointing undispatchable work.
    """
    trusted = runtime.capability_runtime
    names = _catalog_names(runtime)
    for task in new_tasks:
        if task.capability not in trusted.allowed_capabilities:
            raise ReplanRejected(
                f"task {task.task_id} capability {task.capability!r} is not "
                "in the current allowed capabilities; refusing a replan that "
                "widens authorization"
            )
        if task.capability == "people.lookup" and not trusted.can_read_people:
            raise ReplanRejected(
                f"task {task.task_id} capability 'people.lookup' is not "
                "permitted for this request"
            )
        if names is not None and task.capability not in names:
            raise ReplanRejected(
                f"task {task.task_id} capability {task.capability!r} is not "
                "in the current request-scoped catalog; refusing an "
                "undispatchable replan"
            )


def append_replan_tasks(
    current: TaskPlan, new_tasks: tuple[TaskSpec, ...]
) -> TaskPlan:
    """The single authoritative append-only construction site (R50).

    Every governed owner that appends tasks to a checkpointed plan constructs
    the append-only successor HERE, then validates it through
    :func:`validate_runtime_replan` before it may be leased or checkpointed.
    No other function in the v2 tree builds an appended plan, so rogue append
    forms -- temporary variables, unpacking, list construction, or a helper
    that returns an already-appended plan -- cannot exist elsewhere (the AST
    guard in ``test_replan_discovery.py`` enforces this structurally).
    """
    return current.model_copy(update={"tasks": current.tasks + tuple(new_tasks)})


def entry_fanout_width(new_tasks: tuple[TaskSpec, ...]) -> int:
    """New tasks that can start together: those depending on no other new task."""
    new_ids = {task.task_id for task in new_tasks}
    return sum(
        1 for task in new_tasks if not (set(task.depends_on) & new_ids)
    )


def validate_runtime_replan(
    current: TaskPlan,
    proposed: TaskPlan,
    outcomes: tuple[TaskExecutionSummary, ...],
    policy: DiscoveryPolicy,
    budget: ResearchBudgetView,
    runtime: GraphRuntimeContext,
) -> TaskPlan:
    """Validate an append-only replan against frozen rules + current runtime.

    Order: cancellation/deadline first (no validation past a dead run), then
    the frozen ``validate_replan`` (append-only, plan_id/goal stability,
    completed-task prefix, origin/budget/discovery rules), then the
    runtime-catalog membership and fan-out width the frozen layer cannot see.
    Returns the accepted plan unchanged; raises :class:`ReplanRejected` for
    runtime failures and :class:`ContractValidationError` for frozen-rule
    failures. Never mutates either plan.
    """
    check_replan_dispatchable(runtime)
    accepted = validate_replan(current, proposed, outcomes, policy, budget)
    new_tasks = accepted.tasks[len(current.tasks):]
    _require_capabilities_within_current_runtime_catalog(new_tasks, runtime)
    width = entry_fanout_width(new_tasks)
    if width > budget.max_parallel_branches:
        raise ReplanRejected(
            f"replan entry fan-out width {width} exceeds "
            f"max_parallel_branches {budget.max_parallel_branches}"
        )
    return accepted

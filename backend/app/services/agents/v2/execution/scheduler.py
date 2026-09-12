"""The one and only capability-dispatch path for v2 (Phase 2, Task 3).

``TaskScheduler`` is shared by the Phase-2 fast paths and Phase-3 complex
research: no other module resolves a capability from the registry or calls
``capability.execute``. One dispatch is one guarded step::

    resolve ready TaskSpec
    -> verify it exists in the authoritative checkpointed plan
    -> registry.get(TaskSpec.capability) (typed denial on drift, never silent)
    -> build AgentRequest (TaskSpec.input travels exactly as checkpointed)
    -> capability.execute(request, runtime.capability_runtime) under the deadline
    -> frozen result validation (task id, status/error/data/uses/coverage)
    -> acquire_or_refresh a retention lease for every newly created use
    -> append the immutable result

Deadline/cancellation (T2-M3 ruling) is owned here because no capability port
accepts a deadline: the gate stops before dispatch when the run is cancelled
or ``capability_runtime.deadline_at`` has passed, and the in-flight call runs
under ``asyncio.wait_for`` with the remaining budget. Incomplete work is never
turned into success — cancellation and timeout propagate, and no synthetic
result is fabricated.

Leases: every newly created EvidenceUse is leased before results are
returned so checkpointed uses sit under an active lease (T3 round 1, C1).
The single owner of lease SQL stays
``persistence.retention_leases.RevisionRetentionLeaseRepository``:
``acquire_or_refresh(run_id, revision_id, evidence_use_id)`` with a nullable
``revision_id``. The scheduler resolves each executed task's target
revisions through the checkpointed plan plus the checkpointed bindings
(threaded explicitly from ``execute_node`` — never supervisor/graph state)
and leases every (new use × task revision) pair; a use that resolves to no
revision (targetless People/KG/memory uses) is leased evidence-only
(``revision_id=None`` bound to the use id), which the GC predicate matches
through its ``evidence_use_id`` branch. Nothing is silently skipped: a
missing lease service fails closed whenever a fresh use needs it. The lease
session is committed once, after all acquisitions and before returning,
mirroring the ``binding_node`` safe ordering.
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from ..capabilities import (
    CapabilityDenied,
    CapabilityNotRegistered,
    CapabilityRegistry,
    CapabilityUnavailable,
    denied_result,
    error_result,
)
from ..contracts.binding import DocumentBindingSet
from ..contracts.execution import AgentRequest, AgentResult
from ..contracts.planning import TaskPlan, TaskSpec
from ..contracts.state import GraphRuntimeContext
from ..contracts.validation import ContractValidationError, validate_agent_result

__all__ = [
    "DispatchReport",
    "SchedulerError",
    "TaskScheduler",
    "execute_ready_tasks",
]


class SchedulerError(ValueError):
    """Dispatch-boundary failure: the task must not be checkpointed."""


@dataclass(frozen=True)
class DispatchReport:
    """The typed outcome of one scheduler run (T3 round 1, M3).

    ``results`` are the prior plus newly appended immutable results.
    ``truncated`` is True when the deadline stopped dispatch while tasks
    remained undispatched, so T4/T6 can distinguish a truncated dispatch
    from a complete one. Incomplete work is never reported as complete:
    truncation carries no synthetic results, and cancellation/timeout raise
    instead of returning a report at all.
    """

    results: tuple[AgentResult, ...]
    truncated: bool = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _deadline_at(runtime: GraphRuntimeContext) -> datetime:
    deadline = runtime.capability_runtime.deadline_at
    if deadline.tzinfo is None:
        return deadline.replace(tzinfo=timezone.utc)
    return deadline


def _raise_if_cancelled() -> None:
    """Propagate a pending cancellation before dispatch (never converts)."""
    task = asyncio.current_task()
    if task is not None and task.cancelling() > 0:
        raise asyncio.CancelledError()


def _dispatch_allowed(runtime: GraphRuntimeContext) -> bool:
    return _now() < _deadline_at(runtime)


def _seconds_until_deadline(runtime: GraphRuntimeContext) -> float:
    return (_deadline_at(runtime) - _now()).total_seconds()


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _commit_lease_session(repo: Any) -> None:
    """Commit the dedicated lease unit of work (mirrors ``binding_node``)."""
    session = getattr(repo, "session", None)
    commit = getattr(session, "commit", None)
    if commit is None:
        raise SchedulerError(
            "retention-lease repository exposes no commitable session; "
            "the use lease cannot be committed before the checkpointable update"
        )
    await _maybe_await(commit())


def _task_target_revisions(
    task: TaskSpec,
    plan: TaskPlan,
    bindings: DocumentBindingSet | None,
) -> tuple[UUID, ...]:
    """Ordered unique pinned revisions for the task's input targets."""
    target_ids: tuple[str, ...] = tuple(
        getattr(task.input, "target_ids", None) or ()
    )
    if not target_ids:
        return ()
    if bindings is None:
        raise SchedulerError(
            f"task {task.task_id} reads targets but no checkpointed bindings "
            "were supplied; refusing to checkpoint unleashed evidence uses"
        )
    binding_by_id = {binding.binding_id: binding for binding in bindings.bindings}
    target_by_id = {unit.target_id: unit for unit in plan.target_units}
    revisions: list[UUID] = []
    for target_id in target_ids:
        unit = target_by_id.get(target_id)
        if unit is None:
            raise ContractValidationError(
                f"task {task.task_id} references unknown target {target_id}"
            )
        binding = binding_by_id.get(unit.binding_id)
        if binding is None:
            raise SchedulerError(
                f"target {target_id} binds unknown binding {unit.binding_id}; "
                "refusing to checkpoint unleashed evidence uses"
            )
        try:
            revision = UUID(binding.document_revision)
        except ValueError as exc:
            raise SchedulerError(
                f"binding {binding.binding_id} pins revision "
                f"{binding.document_revision!r}, not a revision id; refusing "
                "to checkpoint unleashed evidence uses"
            ) from exc
        if revision not in revisions:
            revisions.append(revision)
    return tuple(revisions)


async def _lease_new_uses(
    *,
    task: TaskSpec,
    plan: TaskPlan,
    bindings: DocumentBindingSet | None,
    result: AgentResult,
    runtime: GraphRuntimeContext,
    known_use_ids: set[UUID],
) -> bool:
    """Lease every newly created use of one result. Returns True if acquired.

    Revision-anchored when the use resolves to a pinned document revision;
    evidence-only (``revision_id=None``) otherwise — never silently skipped.
    """
    fresh = [
        ref
        for ref in result.evidence_uses
        if ref.use_id not in known_use_ids
    ]
    if not fresh:
        return False
    revisions = _task_target_revisions(task, plan, bindings)
    for ref in fresh:
        known_use_ids.add(ref.use_id)
    repo = runtime.services.retention_leases
    if repo is None:
        raise SchedulerError(
            f"task {task.task_id} created evidence uses but no "
            "retention-lease service is wired; refusing to checkpoint "
            "unleashed uses"
        )
    run_id = runtime.capability_runtime.run_id
    if revisions:
        for ref in fresh:
            for revision in revisions:
                await _maybe_await(
                    repo.acquire_or_refresh(run_id, revision, ref.use_id)
                )
    else:
        for ref in fresh:
            await _maybe_await(
                repo.acquire_or_refresh(run_id, None, ref.use_id)
            )
    return True


async def _dispatch_one(
    task: TaskSpec,
    *,
    registry: CapabilityRegistry | None,
    runtime: GraphRuntimeContext,
) -> AgentResult:
    """Resolve and execute one checkpoint-owned task (deadline already gated)."""
    if registry is None:
        raise SchedulerError(
            f"task {task.task_id} requires capability {task.capability!r} "
            "but no capability registry is wired on runtime.services"
        )
    try:
        capability = registry.get(task.capability)
    except CapabilityDenied as exc:
        return denied_result(
            task.task_id, code="PERMISSION_DENIED", message=str(exc)
        )
    except CapabilityUnavailable as exc:
        return error_result(
            task.task_id, code="DEPENDENCY_UNAVAILABLE", message=str(exc)
        )
    except CapabilityNotRegistered as exc:
        return error_result(
            task.task_id, code="CONTRACT_MISMATCH", message=str(exc)
        )
    request = AgentRequest(
        contract_version="2.0",
        task_id=task.task_id,
        objective=task.task_objective,
        input=task.input,
    )
    return await asyncio.wait_for(
        capability.execute(request, runtime.capability_runtime),
        timeout=_seconds_until_deadline(runtime),
    )


async def execute_ready_tasks(
    *,
    plan: TaskPlan,
    results: tuple[AgentResult, ...] = (),
    registry: CapabilityRegistry | None,
    runtime: GraphRuntimeContext,
    bindings: DocumentBindingSet | None = None,
) -> DispatchReport:
    """Execute every ready plan task in plan order; report all results.

    ``plan`` is the authoritative checkpointed plan: prior results that do not
    resolve to its tasks fail closed, and only its tasks are ever dispatched.
    ``bindings`` are the checkpointed bindings used solely to resolve lease
    revisions (never supervisor/graph state). Ready means all ``depends_on``
    tasks already have results; dispatch is sequential in plan order so
    dependencies complete before dependents. Tasks that can never
    become ready (unknown dependency or cycle — the frozen validator should
    have rejected the plan at checkpoint time) raise ``SchedulerError``
    instead of silently returning a partial set.
    """
    task_by_id = {task.task_id: task for task in plan.tasks}
    for result in results:
        if result.task_id not in task_by_id:
            raise ContractValidationError(
                f"task result references unknown task {result.task_id}; "
                "results must resolve to the checkpointed plan"
            )
        validate_agent_result(result, plan)
    completed = list(results)
    done = {result.task_id for result in completed}
    known_use_ids = {
        ref.use_id for result in completed for ref in result.evidence_uses
    }
    leased_any = False
    truncated = False
    while True:
        ready = next(
            (
                task
                for task in plan.tasks
                if task.task_id not in done
                and all(dependency in done for dependency in task.depends_on)
            ),
            None,
        )
        if ready is None:
            if any(task.task_id not in done for task in plan.tasks):
                remaining = sorted(
                    task.task_id
                    for task in plan.tasks
                    if task.task_id not in done
                )
                raise SchedulerError(
                    f"tasks {remaining} can never become ready (unknown "
                    "dependency or cycle); refusing to return a silent partial"
                )
            break
        _raise_if_cancelled()
        if not _dispatch_allowed(runtime):
            truncated = True
            break
        result = await _dispatch_one(ready, registry=registry, runtime=runtime)
        if result.task_id != ready.task_id:
            raise ContractValidationError(
                f"capability {ready.capability!r} returned a result for task "
                f"{result.task_id!r}, expected {ready.task_id!r}"
            )
        validate_agent_result(result, plan)
        if await _lease_new_uses(
            task=ready,
            plan=plan,
            bindings=bindings,
            result=result,
            runtime=runtime,
            known_use_ids=known_use_ids,
        ):
            leased_any = True
        completed.append(result)
        done.add(result.task_id)
    if leased_any:
        repo = runtime.services.retention_leases
        if repo is None:  # pragma: no cover - guarded in _lease_new_uses
            raise SchedulerError(
                "leases were acquired but the retention-lease service is gone"
            )
        await _commit_lease_session(repo)
    return DispatchReport(results=tuple(completed), truncated=truncated)


class TaskScheduler:
    """The one and only capability-dispatch path for v2."""

    def __init__(self, registry: CapabilityRegistry | None) -> None:
        self._registry = registry

    async def execute(
        self,
        plan: TaskPlan,
        runtime: GraphRuntimeContext,
        prior_results: tuple[AgentResult, ...] = (),
        bindings: DocumentBindingSet | None = None,
    ) -> DispatchReport:
        """Execute the plan's ready tasks; report prior plus new results.

        ``plan`` must be the checkpointed plan (``execute_node`` enforces
        ownership via ``require_checkpointed_plan``); ``bindings`` are the
        checkpointed bindings for lease resolution. Cancellation and deadline
        stop dispatch without fabricating success; a deadline stop is
        recorded on the returned ``DispatchReport.truncated`` flag.
        """
        return await execute_ready_tasks(
            plan=plan,
            results=prior_results,
            registry=self._registry,
            runtime=runtime,
            bindings=bindings,
        )

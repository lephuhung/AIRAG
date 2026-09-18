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
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

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
    "V1FallbackRequired",
    "assert_scheduler_input_passthrough",
    "execute_ready_tasks",
    "ActiveRunHeartbeat",
    "is_run_active",
    "is_run_cancel_requested",
    "is_run_cancel_requested_async",
    "refresh_active_run",
    "refresh_active_run_async",
    "refresh_pairs_for_checkpoint",
    "register_active_run",
    "start_active_run_heartbeat",
    "register_active_run_async",
    "request_run_cancellation",
    "request_run_cancellation_async",
    "shared_scheduler_for",
    "unregister_active_run",
    "unregister_active_run_async",
]


class SchedulerError(ValueError):
    """Dispatch-boundary failure: the task must not be checkpointed."""


class V1FallbackRequired(SchedulerError):
    """A v2 candidate must fall back to v1 BEFORE any capability execution.

    Raised by the Task 7B pre-dispatch guard when the resolved
    QueryAnalysis/Router outcome is write or otherwise unsupported (see
    ``app.services.agent.rollout_control.requires_v1_fallback``; Task 12
    serves evaluate/compliance through the governed v2 DAG). The v2
    outer runner lets this propagate (never converts it to an error event)
    so the entrypoint serves v1 with zero v2 user-visible output.
    """


# ---------------------------------------------------------------------------
# Active-run registry + distributed cancellation (Task 7B).
#
# Every scheduler dispatch registers its run and honors cancellation BEFORE
# dispatching: the local asyncio cancel path (``_raise_if_cancelled``) plus
# a distributed cancel flag. With REDIS_ENABLED the flag is a cross-process
# Redis key (``v2run:cancel:<run_id>``); otherwise an in-process set backs
# it so single-process deploys and unit tests behave identically. All Redis
# access is best-effort and never breaks dispatch when Redis is unreachable.
# ---------------------------------------------------------------------------

_ACTIVE_RUN_TTL_SECONDS = 300

_local_active_runs: set[str] = set()
_local_cancel_requests: set[str] = set()


def _active_key(run_id: str) -> str:
    return f"v2run:active:{run_id}"


def _cancel_key(run_id: str) -> str:
    return f"v2run:cancel:{run_id}"


def _redis_client_if_enabled() -> Any | None:
    """Return the shared async Redis client when enabled, else ``None``."""
    try:
        from app.core.redis_client import get_redis, is_redis_enabled

        if is_redis_enabled():
            return get_redis()
    except Exception:
        logger.warning("canary redis unavailable", exc_info=True)
    return None


async def register_active_run_async(run_id: str) -> None:
    """Register ``run_id`` as active, awaiting the Redis write (R75).

    The write is awaited so a cross-process dispatch checking Redis
    immediately after CANNOT observe a missing key — this is the ordering
    guarantee the fire-and-forget sync wrapper cannot give. Never raises.
    """
    if not run_id:
        return
    _local_active_runs.add(str(run_id))
    client = _redis_client_if_enabled()
    if client is None:
        return
    try:
        await client.set(_active_key(run_id), "1", ex=_ACTIVE_RUN_TTL_SECONDS)
    except Exception:
        logger.warning("canary active-run register failed", exc_info=True)


async def unregister_active_run_async(run_id: str) -> None:
    """Drop ``run_id`` from the registry, awaiting Redis (terminal cleanup)."""
    if not run_id:
        return
    _local_active_runs.discard(str(run_id))
    _local_cancel_requests.discard(str(run_id))
    client = _redis_client_if_enabled()
    if client is None:
        return
    try:
        await client.delete(_active_key(run_id), _cancel_key(run_id))
    except Exception:
        logger.warning("canary active-run unregister failed", exc_info=True)


async def request_run_cancellation_async(run_id: str) -> None:
    """Flag ``run_id`` for distributed cancellation, awaiting Redis (R75)."""
    if not run_id:
        return
    _local_cancel_requests.add(str(run_id))
    client = _redis_client_if_enabled()
    if client is None:
        return
    try:
        await client.set(_cancel_key(run_id), "1", ex=_ACTIVE_RUN_TTL_SECONDS)
    except Exception:
        logger.warning("canary cancel request failed", exc_info=True)


_HEARTBEAT_INTERVAL_SECONDS = 60.0


class ActiveRunHeartbeat:
    """Run-lifetime keeper for the distributed active-run key (fix round 2).

    Registration at run start plus per-dispatch refresh still leaves a gap:
    a single capability dispatch (or a quiet resumed stretch) longer than
    the fixed TTL lets the key vanish before terminal cleanup. The
    heartbeat re-``SET``s the key every ``interval_seconds`` for the whole
    run; the owner stops it at the terminal boundary (idempotent).
    """

    def __init__(self, run_id: str, task: "asyncio.Task[None]") -> None:
        self._run_id = str(run_id)
        self._task = task

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def active(self) -> bool:
        task = self._task
        return task is not None and not task.done()

    async def stop(self) -> None:
        """Stop refreshing (idempotent, never raises)."""
        task, self._task = self._task, None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — stop is best effort
            pass


async def _heartbeat_loop(run_id: str, interval_seconds: float) -> None:
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            await refresh_active_run_async(run_id)
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 — heartbeat never breaks the run
        logger.warning("canary active-run heartbeat failed", exc_info=True)


def start_active_run_heartbeat(
    run_id: str, interval_seconds: float = _HEARTBEAT_INTERVAL_SECONDS
) -> ActiveRunHeartbeat | None:
    """Start refreshing ``run_id``'s active key for the run lifetime.

    Returns the heartbeat handle (the caller stops it at the terminal
    boundary) or ``None`` when there is no run or no running loop.
    Never raises.
    """
    if not run_id:
        return None
    try:
        interval = float(interval_seconds)
    except (TypeError, ValueError):
        return None
    if interval <= 0:
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    try:
        task = loop.create_task(_heartbeat_loop(str(run_id), interval))
    except Exception:
        logger.warning("canary active-run heartbeat start failed", exc_info=True)
        return None
    return ActiveRunHeartbeat(str(run_id), task)


def refresh_active_run(run_id: str) -> None:
    """Refresh ``run_id``'s active TTL (sync; best-effort, never raises)."""
    register_active_run(run_id)


async def refresh_active_run_async(run_id: str) -> None:
    """Refresh ``run_id``'s active TTL for long/resumed runs (awaited)."""
    await register_active_run_async(run_id)


def register_active_run(run_id: str) -> None:
    """Register ``run_id`` as an active v2 run (idempotent, never raises)."""
    if not run_id:
        return
    _local_active_runs.add(str(run_id))
    try:
        from app.core.redis_client import get_redis, is_redis_enabled

        if is_redis_enabled():
            client = get_redis()
            result = client.set(
                _active_key(run_id), "1", ex=_ACTIVE_RUN_TTL_SECONDS
            )
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                task = asyncio.get_running_loop().create_task(result)  # type: ignore[arg-type]
                task.add_done_callback(lambda _t: _t.exception() if not _t.cancelled() else None)
    except Exception:
        logger.warning("canary active-run register failed", exc_info=True)


def unregister_active_run(run_id: str) -> None:
    """Drop ``run_id`` from the active set (terminal boundary; never raises)."""
    if not run_id:
        return
    _local_active_runs.discard(str(run_id))
    _local_cancel_requests.discard(str(run_id))
    try:
        from app.core.redis_client import get_redis, is_redis_enabled

        if is_redis_enabled():
            client = get_redis()
            result = client.delete(_active_key(run_id), _cancel_key(run_id))
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                task = asyncio.get_running_loop().create_task(result)  # type: ignore[arg-type]
                task.add_done_callback(lambda _t: _t.exception() if not _t.cancelled() else None)
    except Exception:
        logger.warning("canary active-run unregister failed", exc_info=True)


def request_run_cancellation(run_id: str) -> None:
    """Flag ``run_id`` for distributed cancellation (never raises)."""
    if not run_id:
        return
    _local_cancel_requests.add(str(run_id))
    try:
        from app.core.redis_client import get_redis, is_redis_enabled

        if is_redis_enabled():
            client = get_redis()
            result = client.set(
                _cancel_key(run_id), "1", ex=_ACTIVE_RUN_TTL_SECONDS
            )
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                task = asyncio.get_running_loop().create_task(result)  # type: ignore[arg-type]
                task.add_done_callback(lambda _t: _t.exception() if not _t.cancelled() else None)
    except Exception:
        logger.warning("canary cancel request failed", exc_info=True)


def is_run_cancel_requested(run_id: str) -> bool:
    """True when cancellation was requested for ``run_id`` (sync part)."""
    if not run_id:
        return False
    if str(run_id) in _local_cancel_requests:
        return True
    try:
        from app.core.redis_client import get_redis, is_redis_enabled

        if is_redis_enabled():
            client = get_redis()
            exists = client.exists(_cancel_key(run_id))
            if not (asyncio.iscoroutine(exists) or asyncio.isfuture(exists)):
                return bool(exists)
    except Exception:
        logger.warning("canary cancel check failed", exc_info=True)
    return False


async def is_run_cancel_requested_async(run_id: str) -> bool:
    """Async cancel check: local set first, then Redis when enabled."""
    if not run_id:
        return False
    if str(run_id) in _local_cancel_requests:
        return True
    try:
        from app.core.redis_client import get_redis, is_redis_enabled

        if is_redis_enabled():
            return bool(await get_redis().exists(_cancel_key(run_id)))
    except Exception:
        logger.warning("canary async cancel check failed", exc_info=True)
    return False


def is_run_active(run_id: str) -> bool:
    """True when ``run_id`` is currently registered (sync local view)."""
    return bool(run_id) and str(run_id) in _local_active_runs


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


def _use_id_of(use: Any) -> UUID | None:
    """Coerce one evidence-use identity (live model or serde mapping)."""
    raw = use.get("use_id") if isinstance(use, dict) else getattr(use, "use_id", None)
    if raw is None:
        return None
    if isinstance(raw, UUID):
        return raw
    try:
        return UUID(str(raw))
    except (ValueError, AttributeError, TypeError):
        return None


def _slot(value: Any, name: str) -> Any:
    """Read one slot from a live model or its checkpoint-serde mapping."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _refresh_task_revisions(task: Any, plan: Any, bindings: Any) -> tuple[UUID, ...]:
    """Pinned revisions for one task, tolerant of checkpoint-serde mappings.

    Same target → binding → revision-UUID recipe as :func:`_task_target_revisions`
    (the dispatch owner), but duck-typed so real ``get_state`` values — whose
    nested contracts revive as ``list``/``dict`` — resolve identically to live
    models. Unresolvable targets yield no revisions (the caller falls back to
    evidence-only pairs); malformed revision pins raise ``SchedulerError`` just
    like dispatch, so a corrupt pin can never look refreshed.
    """
    task_input = _slot(task, "input") or {}
    target_ids = tuple(_slot(task_input, "target_ids") or ())
    if not target_ids:
        return ()
    raw_bindings = _slot(bindings, "bindings")
    if raw_bindings is None:
        raise SchedulerError(
            "checkpoint supplies no bindings for targeted tasks; "
            "refusing to refresh unleashed uses"
        )
    binding_by_id = {
        _slot(binding, "binding_id"): binding for binding in (raw_bindings or ())
    }
    raw_units = _slot(plan, "target_units") or ()
    target_by_id = {_slot(unit, "target_id"): unit for unit in raw_units}
    revisions: list[UUID] = []
    for target_id in target_ids:
        unit = target_by_id.get(target_id)
        if unit is None:
            return ()
        binding = binding_by_id.get(_slot(unit, "binding_id"))
        if binding is None:
            return ()
        try:
            revision = UUID(str(_slot(binding, "document_revision")))
        except (ValueError, AttributeError, TypeError) as exc:
            raise SchedulerError(
                "checkpoint binding pins a non-UUID revision; refusing to "
                "refresh a corrupt pin"
            ) from exc
        if revision not in revisions:
            revisions.append(revision)
    return tuple(revisions)


def refresh_pairs_for_checkpoint(
    plan: Any,
    bindings: Any,
    results: Any,
) -> tuple[tuple[UUID | None, UUID], ...]:
    """``(revision_id, use_id)`` lease pairs for a checkpointed execution.

    Shared recipe (T8-I2) mirroring :func:`_lease_new_uses` anchoring: every
    checkpointed use re-acquires the SAME ``(run, revision, use)`` rows the
    scheduler leased at dispatch — revision-anchored when the owning task
    resolves to pinned revisions, evidence-only (``revision_id=None``)
    otherwise. Deduplicated, order-stable. Tasks whose targets no longer
    resolve (stale checkpoint) contribute no pairs; uses without a coercible
    identity are skipped. Total function: never raises on shape drift.
    """
    try:
        items = tuple(results or ())
    except TypeError:
        return ()
    tasks: dict[Any, Any] = {}
    try:
        plan_tasks = (
            plan.get("tasks", ())
            if isinstance(plan, dict)
            else getattr(plan, "tasks", None) or ()
        )
        for task in plan_tasks:
            task_id = (
                task.get("task_id")
                if isinstance(task, dict)
                else getattr(task, "task_id", None)
            )
            if task_id is not None:
                tasks.setdefault(task_id, task)
    except Exception:
        return ()
    pairs: list[tuple[UUID | None, UUID]] = []
    seen: set[tuple[Any, Any]] = set()
    for item in items:
        try:
            if isinstance(item, dict):
                item_task_id = item.get("task_id")
                uses = item.get("evidence_uses") or ()
            else:
                item_task_id = getattr(item, "task_id", None)
                uses = getattr(item, "evidence_uses", None) or ()
            use_ids = []
            for use in uses:
                uid = _use_id_of(use)
                if uid is not None:
                    use_ids.append(uid)
            if not use_ids:
                continue
            revisions: tuple[UUID, ...] = ()
            task = tasks.get(item_task_id)
            if task is not None:
                try:
                    revisions = _refresh_task_revisions(task, plan, bindings)
                except Exception:
                    revisions = ()
            anchors = revisions or (None,)
            for uid in use_ids:
                for revision in anchors:
                    key = (revision, uid)
                    if key not in seen:
                        seen.add(key)
                        pairs.append((revision, uid))
        except Exception:
            continue
    return tuple(pairs)


def assert_scheduler_input_passthrough(task: TaskSpec, request: AgentRequest) -> None:
    """Executable People→Document invariant: the scheduler never rewrites input.

    The deterministic dependency materializer builds the concrete
    ``TaskSpec.input`` BEFORE the dependent task is appended/checkpointed, so
    by dispatch time the input is final. The scheduler executes it exactly as
    checkpointed: no mutation, no lazy materialization, no second scalar
    source. Any drift between the checkpointed task and the dispatch request
    fails closed instead of dispatching a rewritten input.
    """
    if request.task_id != task.task_id:
        raise SchedulerError(
            f"dispatch request targets task {request.task_id!r}, expected "
            f"checkpointed task {task.task_id!r}; refusing a rewritten dispatch"
        )
    if request.input != task.input:
        raise SchedulerError(
            f"dispatch request input drifted from checkpointed task "
            f"{task.task_id!r}; the scheduler never rewrites a task input"
        )
    if request.objective != task.task_objective:
        raise SchedulerError(
            f"dispatch request objective drifted from checkpointed task "
            f"{task.task_id!r}; the scheduler never rewrites a task input"
        )


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
    # People→Document invariant: the concrete input was materialized BEFORE
    # this task was checkpointed, so it travels to the capability verbatim.
    assert_scheduler_input_passthrough(task, request)
    # Per-dispatch wall time: capability + task id + outcome only, never
    # request/result payloads (v2 content-suppression tracing policy).
    from app.services.agent.timing_recorder import get_recorder

    recorder = get_recorder()
    started = time.monotonic()
    try:
        if recorder is not None:
            async with recorder.span(
                "dispatch", task.capability, meta={"task_id": task.task_id}
            ):
                result = await asyncio.wait_for(
                    capability.execute(request, runtime.capability_runtime),
                    timeout=_seconds_until_deadline(runtime),
                )
        else:
            result = await asyncio.wait_for(
                capability.execute(request, runtime.capability_runtime),
                timeout=_seconds_until_deadline(runtime),
            )
    except BaseException as exc:
        logger.info(
            "[v2dispatch] %s task=%s failed after %dms (%s)",
            task.capability,
            task.task_id,
            int((time.monotonic() - started) * 1000),
            type(exc).__name__,
        )
        raise
    logger.info(
        "[v2dispatch] %s task=%s status=%s in %dms",
        task.capability,
        task.task_id,
        getattr(result.status, "value", result.status),
        int((time.monotonic() - started) * 1000),
    )
    return result


async def _run_pre_dispatch_guards(
    runtime: GraphRuntimeContext,
    v1_fallback_guard: Any | None = None,
) -> None:
    """Run the Task 7B guards BEFORE each scheduler dispatch (in order).

    1. Local asyncio cancellation propagates (never converted).
    2. The run registers as active via the AWAITED Redis write, then
       distributed cancellation is honored — a requested cancel raises
       ``CancelledError`` before any capability executes. Awaiting the
       registration is the R75 ordering guarantee: a cross-process
       dispatch checking Redis immediately after cannot observe a
       missing key.
    3. The optional ``v1_fallback_guard`` (sync or async zero-arg callable
       supplied by the caller that owns the resolved QueryAnalysis/Router
       outcome) fires ``V1FallbackRequired`` — also before any capability
       executes. ``None`` (the default) preserves the exact prior behavior.
    """
    _raise_if_cancelled()
    run_id = ""
    try:
        run_id = str(runtime.capability_runtime.run_id or "")
    except Exception:
        run_id = ""
    if run_id:
        await register_active_run_async(run_id)
        if await is_run_cancel_requested_async(run_id):
            raise asyncio.CancelledError()
    if v1_fallback_guard is not None:
        fired = v1_fallback_guard()
        if inspect.isawaitable(fired):
            fired = await fired
        if fired:
            raise V1FallbackRequired(
                "v2 candidate resolved to a v1-only route (write or "
                "unsupported); falling back "
                "to v1 before any capability execution"
            )


#: Capabilities that resolve scoped targets through the constructor-injected
#: ``PinnedTargetResolver``. ``document.read`` / ``section.read`` inputs require
#: non-empty ``target_ids`` by contract; ``document.retrieve`` supports both
#: scoped (non-empty) and unscoped (empty) requests. Only a task in this set
#: whose input carries non-empty ``target_ids`` counts as targeted — every
#: other capability (people/KG/memory lookups, search, write) never requires
#: a resolver, so the shadow rig and targetless plans stay supported.
_PINNED_TARGET_CAPABILITIES = frozenset(
    {"document.retrieve", "document.read", "section.read"}
)


def _plan_has_targeted_document_tasks(plan: Any) -> bool:
    """True when the plan dispatches a scoped resolver-consumer task.

    Narrow by construction: only a ``document.retrieve`` / ``document.read``
    / ``section.read`` task whose input carries non-empty ``target_ids``
    counts. Targetless plans (empty ``target_ids``, other capabilities, or
    no tasks) return False, so the shadow rig and unscoped retrieval never
    require a resolver. Duck-typed via ``_slot`` so checkpoint-serde
    mappings read identically to live models.
    """
    tasks = _slot(plan, "tasks") or ()
    for task in tasks:
        if _slot(task, "capability") not in _PINNED_TARGET_CAPABILITIES:
            continue
        task_input = _slot(task, "input") or {}
        if tuple(_slot(task_input, "target_ids") or ()):
            return True
    return False


async def _feed_pinned_targets(
    plan: TaskPlan,
    bindings: DocumentBindingSet | None,
    runtime: GraphRuntimeContext,
) -> None:
    """Feed the request-scoped pinned-target resolver before dispatch.

    P0 factual-retrieval live-gate fix: fresh (non-resume) turns reach the
    shared scheduler with a fresh, empty resolver, so every scoped target
    reads as "unknown" and retrieval fails closed before the provider is
    called. The scheduler — the one and only dispatch path — installs the
    authoritative checkpointed plan + bindings on the runtime-only
    ``pinned_target_resolver`` service (the exact instance the document
    capabilities resolve through) immediately before the first dispatch.
    The feed replaces the whole mapping, so replans/resumes never inherit
    stale targets. The resume-path pre-feed stays as an idempotent
    compatibility refresh; this dispatch-time feed is the authoritative
    owner (not scheduler input materialization: ``TaskSpec.input`` is
    never mutated).

    The feed is async-aware: an ``async def feed`` result is awaited, so an
    installing or exploding async feed can neither silently no-op nor
    masquerade as a denial. A wiring fault on a TARGETED plan (no
    resolver, no callable ``feed``, missing plan/bindings, or a ``feed``
    exception — sync or async) raises typed ``SchedulerError`` with a
    warning — it must never masquerade as a capability
    ``denied``/``SCOPE_VIOLATION`` authorization decision. A genuine target
    mismatch AFTER a successful feed still fails closed in the capability
    (``denied``/``SCOPE_VIOLATION``). Targetless plans with no resolver
    remain supported (no-op).
    """
    targeted = _plan_has_targeted_document_tasks(plan)
    try:
        resolver = getattr(
            getattr(runtime, "services", None), "pinned_target_resolver", None
        )
    except Exception:
        resolver = None
    if resolver is None:
        if targeted:
            logger.warning(
                "pinned-target feed has no resolver for a targeted plan; "
                "raising instead of reporting a wiring fault as a denial"
            )
            raise SchedulerError(
                "targeted document tasks (document.retrieve/document.read/"
                "section.read with target_ids) require a wired "
                "pinned-target resolver; refusing to report a wiring fault "
                "as an authorization denial"
            )
        return
    feed = getattr(resolver, "feed", None)
    if not callable(feed):
        if targeted:
            logger.warning(
                "pinned-target resolver has no callable feed for a targeted "
                "plan; raising instead of reporting a wiring fault as a denial"
            )
            raise SchedulerError(
                "pinned-target resolver exposes no callable feed; refusing "
                "to report a wiring fault as an authorization denial"
            )
        return
    if plan is None or bindings is None:
        if targeted:
            logger.warning(
                "pinned-target feed is missing checkpointed plan/bindings "
                "for a targeted plan; raising instead of denying"
            )
            raise SchedulerError(
                "targeted document tasks require checkpointed plan "
                "+ bindings for the pinned-target feed; refusing to report "
                "a wiring fault as an authorization denial"
            )
        return
    try:
        fed = feed(plan, bindings)
        if inspect.isawaitable(fed):
            await fed
    except SchedulerError:
        raise
    except Exception as exc:
        logger.warning(
            "pinned-target feed failed; raising instead of reporting a "
            "wiring fault as a denial",
            exc_info=True,
        )
        if targeted:
            raise SchedulerError(
                "pinned-target feed failed; refusing to report a wiring "
                "fault as an authorization denial"
            ) from exc


async def execute_ready_tasks(
    *,
    plan: TaskPlan,
    results: tuple[AgentResult, ...] = (),
    registry: CapabilityRegistry | None,
    runtime: GraphRuntimeContext,
    bindings: DocumentBindingSet | None = None,
    v1_fallback_guard: Any | None = None,
    total_task_limit: int | None = None,
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

    ``total_task_limit`` is the optional absolute plan-entry ceiling
    (discovery spec §11.3): when supplied, a plan already carrying more
    tasks than the limit is rejected BEFORE any result validation or
    dispatch. ``None`` (the default) preserves the exact prior behavior —
    ``validate_task_plan`` stays cap-free so fast/legacy plans are
    unaffected.
    """
    if total_task_limit is not None and len(plan.tasks) > total_task_limit:
        raise SchedulerError("plan exceeds total task limit")
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
    # Authoritative dispatch-time feed: the request-scoped pinned-target
    # resolver resolves the checkpointed plan's targets for every dispatch
    # below (fresh turns included); the resume-path pre-feed stays as an
    # idempotent compatibility refresh.
    await _feed_pinned_targets(plan, bindings, runtime)
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
        await _run_pre_dispatch_guards(runtime, v1_fallback_guard)
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
        v1_fallback_guard: Any | None = None,
        total_task_limit: int | None = None,
    ) -> DispatchReport:
        """Execute the plan's ready tasks; report prior plus new results.

        ``plan`` must be the checkpointed plan (``execute_node`` enforces
        ownership via ``require_checkpointed_plan``); ``bindings`` are the
        checkpointed bindings for lease resolution. Cancellation and deadline
        stop dispatch without fabricating success; a deadline stop is
        recorded on the returned ``DispatchReport.truncated`` flag.
        ``v1_fallback_guard`` is the Task 7B pre-dispatch hook (see
        ``execute_ready_tasks``); ``total_task_limit`` is the optional
        absolute plan-entry ceiling threaded to ``execute_ready_tasks``.
        ``None`` defaults preserve prior behavior.
        """
        return await execute_ready_tasks(
            plan=plan,
            results=prior_results,
            registry=self._registry,
            runtime=runtime,
            bindings=bindings,
            v1_fallback_guard=v1_fallback_guard,
            total_task_limit=total_task_limit,
        )


def shared_scheduler_for(runtime: GraphRuntimeContext) -> TaskScheduler:
    """Return the shared scheduler bound to the request-scoped registry.

    R5 seam for Phase 3: the complex subgraph's ``execute`` node dispatches
    ONLY through this constructor. No second scheduler is defined or
    duplicated here — this is the same class Phase 2's ``execute_node``
    builds inline — and no capability is resolved outside it.
    """
    return TaskScheduler(runtime.services.capability_registry)

"""Request-scoped execution package (Phase 2, Task 3).

Owns the shared capability-dispatch path (:mod:`scheduler`) and re-exports
the single :class:`RuntimeServices` defined in ``contracts/state.py`` as the
canonical import path. This package never defines a second service bag: plan
persistence is LangGraph state plus the supervisor checkpointer (no service),
and evidence evaluation is the shared ``evaluate_evidence(...)`` function in
``nodes/evaluate.py`` (no evaluator service).
"""
from __future__ import annotations

from ..contracts.state import RuntimeServices
from .scheduler import (
    DispatchReport,
    SchedulerError,
    TaskScheduler,
    V1FallbackRequired,
    execute_ready_tasks,
    is_run_active,
    is_run_cancel_requested,
    is_run_cancel_requested_async,
    refresh_active_run,
    refresh_active_run_async,
    register_active_run,
    register_active_run_async,
    request_run_cancellation,
    request_run_cancellation_async,
    shared_scheduler_for,
    unregister_active_run,
    unregister_active_run_async,
)

__all__ = [
    "DispatchReport",
    "RuntimeServices",
    "SchedulerError",
    "TaskScheduler",
    "V1FallbackRequired",
    "execute_ready_tasks",
    "is_run_active",
    "is_run_cancel_requested",
    "is_run_cancel_requested_async",
    "refresh_active_run",
    "refresh_active_run_async",
    "register_active_run",
    "register_active_run_async",
    "request_run_cancellation",
    "request_run_cancellation_async",
    "shared_scheduler_for",
    "unregister_active_run",
    "unregister_active_run_async",
]

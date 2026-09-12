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
from .scheduler import SchedulerError, TaskScheduler, execute_ready_tasks

__all__ = [
    "RuntimeServices",
    "SchedulerError",
    "TaskScheduler",
    "execute_ready_tasks",
]

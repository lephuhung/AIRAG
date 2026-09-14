"""Governed Adaptive Planner (Phase 5, Task 10).

Proposal-only planning behind the v2 boundaries. ``ResearchPlanningInput``
stays the authoritative runtime envelope; :func:`build_planner_model_input`
projects it to a minimized/redacted model input (binding IDs and roles only —
never document/revision UUIDs, evidence UUIDs, or governed scalars), and
:func:`AdaptivePlanner.propose_initial` turns a model step list into a
server-constructed :class:`TaskPlan` that still passes the frozen validators
before lease/checkpoint/scheduler.

The planner never dispatches tools, never mints document identity, and never
widens hard scope. Deterministic skill policies win whenever they cover the
work type; the model path runs only for work the skills do not cover. A model
failure — or any ungovernable proposal — raises :class:`PlannerError` so the
caller keeps the typed unavailable boundary with zero dispatch.
"""
from __future__ import annotations

from .planner import AdaptivePlanner, PlannerError
from .projection import PlannerModelInput, build_planner_model_input

__all__ = [
    "AdaptivePlanner",
    "PlannerError",
    "PlannerModelInput",
    "build_planner_model_input",
]

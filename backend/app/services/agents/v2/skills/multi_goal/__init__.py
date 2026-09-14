"""Multi-goal skill package: bounded parallel reads over bound documents.

The framework-neutral policy lives in :mod:`policy`; that module is the source
of truth a native skill file would mirror. No agent route lives here.
"""

from .policy import (
    MAX_MULTI_GOAL_TARGETS,
    MIN_MULTI_GOAL_TARGETS,
    MULTI_GOAL_WORK_TYPE,
    build_multi_goal_plan,
    covers_input,
    supports_work_type,
)

__all__ = [
    "MAX_MULTI_GOAL_TARGETS",
    "MIN_MULTI_GOAL_TARGETS",
    "MULTI_GOAL_WORK_TYPE",
    "build_multi_goal_plan",
    "covers_input",
    "supports_work_type",
]

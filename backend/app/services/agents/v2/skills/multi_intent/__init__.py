"""Multi-intent skill package: parallel evidence tasks from detected intents.

The framework-neutral policy lives in :mod:`policy`; that module is the source
of truth a native skill file would mirror. No agent route lives here.
"""

from .policy import (
    MULTI_INTENT_WORK_TYPE,
    build_multi_intent_plan,
    covers_input,
    supports_work_type,
)

__all__ = [
    "MULTI_INTENT_WORK_TYPE",
    "build_multi_intent_plan",
    "covers_input",
    "supports_work_type",
]

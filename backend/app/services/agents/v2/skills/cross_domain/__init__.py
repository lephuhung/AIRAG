"""Cross-domain skill package: governed cross-family execution know-how.

The framework-neutral policy lives in :mod:`policy`; that module is the source
of truth a native skill file would mirror. No agent route lives here.
"""

from .policy import (
    CROSS_DOMAIN_WORK_TYPE,
    MAX_CROSS_DOMAIN_TARGETS,
    build_cross_domain_plan,
    covers_input,
    supports_work_type,
)

__all__ = [
    "CROSS_DOMAIN_WORK_TYPE",
    "MAX_CROSS_DOMAIN_TARGETS",
    "build_cross_domain_plan",
    "covers_input",
    "supports_work_type",
]

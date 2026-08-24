"""
Worker Config Watch
====================
Keeps each worker process's LLM runtime-config snapshot fresh without any
push infrastructure (no Redis pub/sub, no RabbitMQ control exchange).

Design (docs/plan-llm-runtime-config.md §6):
  * Every worker message handler calls ``ensure_fresh_config()`` FIRST, before
    resolving any LLM provider. The check is a single lightweight DB read of
    the ``_config_version`` row — done inside ``refresh_snapshot()``.
  * Fail-open contract: a failed config check must NEVER kill the worker or
    drop a message. Every exception is swallowed and logged as a warning,
    throttled so a broken DB connection doesn't spam the logs on every message.
  * When the version changed, the snapshot is rebuilt in-process and the very
    next provider resolution (via the sync factories reading the snapshot)
    picks up the new model/URL/key. A job already running keeps its old
    provider — config changes take effect from the next message onward.

Not wired into: parse_worker (uses no LLM within this feature's scope).
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# Warn about repeated config-check failures at most once per this many seconds.
_WARN_INTERVAL_SECONDS = 60.0

_last_warn_ts: float = 0.0


async def ensure_fresh_config() -> None:
    """Refresh the RuntimeConfigService snapshot if the DB version changed.

    Call at the TOP of every worker message handler, before any provider is
    resolved. Never raises — a config-check failure is logged (throttled) and
    ignored so message processing continues with the last-known-good snapshot.
    """
    global _last_warn_ts

    try:
        from app.services.runtime_config import refresh_snapshot

        await refresh_snapshot()
    except Exception as exc:
        # Fail-open: keep serving with the current snapshot (.env defaults if
        # nothing was ever loaded). Throttle the log so a sustained outage
        # produces one warning per minute instead of one per message.
        now = time.monotonic()
        if now - _last_warn_ts >= _WARN_INTERVAL_SECONDS:
            _last_warn_ts = now
            logger.warning(
                f"[config_watch] LLM config check failed "
                f"(keeping last-known-good snapshot): {exc}"
            )

"""
Redis client — async singleton for shared cross-process state.

Enables the backend to run more than one worker process / replica by moving
state that used to live in a single process out to Redis:
  - stream cancel   : pub/sub so a Stop hitting ANY process reaches the process
                      actually running the agent run (Phase 1, this change)
  - GPU semaphore   : distributed concurrency limit (Phase 2, later)
  - retrieval cache : shared TTL cache across processes (Phase 3, later)

Everything is gated by ``settings.REDIS_ENABLED``. When it is false (the
default) the callers fall back to their existing in-process behaviour and this
module never opens a connection — the deploy runs exactly as before.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Optional

from app.core.config import settings

if TYPE_CHECKING:  # avoid importing redis at module load when it's unused
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Unique per process. Lets pub/sub consumers recognise self-published messages
# and tags any per-instance keys we may add later.
INSTANCE_ID: str = uuid.uuid4().hex

_client: "Optional[Redis]" = None


def is_redis_enabled() -> bool:
    """True when shared cross-process state should be routed through Redis."""
    return settings.REDIS_ENABLED


def get_redis() -> "Redis":
    """Return the singleton async Redis client (lazy init).

    Raises RuntimeError when REDIS_ENABLED is false — callers MUST gate on
    ``is_redis_enabled()`` before using this.
    """
    global _client
    if not settings.REDIS_ENABLED:
        raise RuntimeError("Redis is disabled (REDIS_ENABLED=false)")
    if _client is None:
        import redis.asyncio as aioredis

        _client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,  # pub/sub + cache callers get str, not bytes
            socket_connect_timeout=3,
            socket_timeout=5,
            health_check_interval=30,
            retry_on_timeout=True,
        )
        logger.info(
            "[redis] Client created for %s (instance=%s)",
            settings.REDIS_URL,
            INSTANCE_ID,
        )
    return _client


async def ping_redis() -> bool:
    """Best-effort connectivity probe. Never raises."""
    if not settings.REDIS_ENABLED:
        return False
    try:
        return bool(await get_redis().ping())
    except Exception as e:  # noqa: BLE001 — startup probe, must not be fatal
        logger.warning("[redis] ping failed: %s", e)
        return False


async def close_redis() -> None:
    """Close the singleton client (call on app shutdown)."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:  # noqa: BLE001 — shutdown, swallow
            pass
        _client = None
        logger.info("[redis] Client closed")

"""
Distributed GPU-search concurrency limiter — two tiers.

  Tier 1 (always): a per-process ``asyncio.Semaphore`` so a single backend
    process never self-oversubscribes the GPU.
  Tier 2 (only when ``REDIS_ENABLED``): a cluster-wide permit shared by ALL
    backend worker processes / replicas that sit on the same GPU, so running
    ``uvicorn --workers N`` (or N replicas) can't multiply the fan-out's
    activation-memory peak N-fold (OOM incident 2026-07-04).

The Tier-2 permit is a self-healing sliding-window semaphore: each holder ZADDs
a unique token scored by acquire-time into one ZSET; every acquire first prunes
tokens older than ``HRAG_SEARCH_GPU_PERMIT_TTL`` so a holder that crashed without
releasing can never permanently shrink capacity. When Redis is disabled (or has
a hiccup) this transparently degrades to the local semaphore only — exactly the
single-process behaviour that shipped before.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from typing import Optional

from app.core.config import settings
from app.core.redis_client import INSTANCE_ID, get_redis, is_redis_enabled

logger = logging.getLogger(__name__)

# Per-process soft cap (Tier 1). Sized from settings at import time, matching
# the old module-level asyncio.Semaphore in chat_agent.py.
_local_semaphore = asyncio.Semaphore(max(1, settings.HRAG_SEARCH_GPU_CONCURRENCY))

_ZSET_KEY = "gpu:search:permits"

# Atomic acquire: prune expired holders, then take a slot iff there's room.
# KEYS[1]=zset  ARGV: 1=limit 2=ttl_ms 3=token 4=now_ms  -> 1 acquired / 0 full.
_ACQUIRE_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local token = ARGV[3]
local now = tonumber(ARGV[4])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - ttl)
if redis.call('ZCARD', key) < limit then
  redis.call('ZADD', key, now, token)
  redis.call('PEXPIRE', key, ttl)
  return 1
end
return 0
"""


@contextlib.asynccontextmanager
async def gpu_search_slot():
    """Acquire a GPU-search slot: local soft cap + (optional) global hard cap."""
    async with _local_semaphore:
        token = await _acquire_global()
        try:
            yield
        finally:
            if token is not None:
                await _release_global(token)


async def _acquire_global() -> Optional[str]:
    """Acquire a cluster-wide permit. Returns a token, or None (local-only)."""
    if not is_redis_enabled() or settings.HRAG_SEARCH_GPU_GLOBAL_CONCURRENCY <= 0:
        return None

    token = f"{INSTANCE_ID}:{uuid.uuid4().hex}"
    limit = settings.HRAG_SEARCH_GPU_GLOBAL_CONCURRENCY
    ttl_ms = int(settings.HRAG_SEARCH_GPU_PERMIT_TTL * 1000)
    deadline = time.monotonic() + settings.HRAG_SEARCH_GPU_WAIT_TIMEOUT
    delay = 0.025
    while True:
        try:
            now_ms = int(time.time() * 1000)
            ok = await get_redis().eval(
                _ACQUIRE_LUA, 1, _ZSET_KEY, limit, ttl_ms, token, now_ms
            )
        except Exception as e:  # noqa: BLE001 — Redis must never wedge search
            logger.warning(
                "[gpu-sem] global acquire failed, proceeding local-only: %s", e
            )
            return None
        if int(ok) == 1:
            return token
        if time.monotonic() >= deadline:
            logger.warning(
                "[gpu-sem] wait timeout (%.1fs) — proceeding WITHOUT a global permit "
                "(per-process cap still applies)",
                settings.HRAG_SEARCH_GPU_WAIT_TIMEOUT,
            )
            return None
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 0.2)


async def _release_global(token: str) -> None:
    try:
        await get_redis().zrem(_ZSET_KEY, token)
    except Exception as e:  # noqa: BLE001 — token self-expires via TTL anyway
        logger.warning("[gpu-sem] release failed (token expires via TTL): %s", e)

"""
Memory Backend Dispatcher
=========================

Routes memory read/write to the backend(s) selected by
``NEXUSRAG_MEMORY_BACKEND``:

    graphiti   (default) → Graphiti temporal KG (Neo4j) — existing behavior
    openviking           → OpenViking context DB (viking://, dedicated server)
    both                 → write to both, read merged (OpenViking first,
                           Graphiti appended)
    none                 → memory disabled (no-op)

Every caller that used to import from ``graphiti_client`` directly should go
through this module instead so the backend switch is a config flag, not a code
edit:

    from app.services.memory.memory_backend import (
        initialize_memory,
        search_user_memory,
        add_conversation_episode,
        save_user_fact,
        save_user_fact_background,
    )

The functions keep the exact same signatures as their Graphiti counterparts, so
the supervisor nodes, agent tools and workers are unchanged apart from the import.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_BACKENDS = ("graphiti", "openviking", "both", "none")


def memory_backend() -> str:
    """Normalized backend selector (defaults to graphiti)."""
    val = (settings.NEXUSRAG_MEMORY_BACKEND or "graphiti").strip().lower()
    return val if val in _BACKENDS else "graphiti"


def is_openviking_enabled() -> bool:
    """OpenViking active when selected AND configured (URL set)."""
    sel = memory_backend()
    if sel not in ("openviking", "both"):
        return False
    from app.services.memory.openviking_client import is_enabled as _ov_enabled

    return _ov_enabled()


async def initialize_memory() -> None:
    """Idempotent startup init for every selected backend. Never raises."""
    for backend, fn in (
        ("graphiti", "_init_graphiti"),
        ("openviking", "_init_openviking"),
    ):
        if memory_backend() not in ("both", backend):
            continue
        try:
            if backend == "graphiti":
                from app.services.memory.graphiti_client import initialize_graphiti

                await initialize_graphiti()
            else:
                from app.services.memory.openviking_client import initialize_openviking

                await initialize_openviking()
        except Exception as exc:
            logger.warning(f"[memory_backend] init {backend} failed (non-fatal): {exc}")


async def search_user_memory(
    user_id: uuid.UUID | int | str,
    query: str,
    top_k: int = 5,
) -> str:
    """Recall relevant user memory across the selected backend(s).

    - graphiti  → Graphiti string (existing behavior)
    - openviking → OpenViking string
    - both      → OpenViking first, then append Graphiti facts (dedup-free; the
                  two backends return different shapes of memory, so a union is
                  useful, bounded by the same budget each side enforces)
    - none      → ""
    """
    sel = memory_backend()
    if sel == "none" or not query.strip():
        return ""

    if sel in ("openviking", "both"):
        try:
            from app.services.memory.openviking_client import search_user_memory as _ov_search

            ov = await _ov_search(user_id, query, top_k=top_k)
            if ov:
                if sel == "openviking":
                    return ov
                # both: OpenViking result first
                try:
                    from app.services.memory.graphiti_client import search_user_memory as _g_search

                    g = await _g_search(user_id, query, top_k=top_k)
                except Exception as exc:
                    logger.warning(f"[memory_backend] graphiti recall failed: {exc}")
                    g = ""
                return (ov + "\n" + g).strip() if g else ov
            # openviking empty → fall through to graphiti when 'both'
            if sel == "openviking":
                return ""
        except Exception as exc:
            logger.warning(f"[memory_backend] openviking recall failed: {exc}")
            if sel == "openviking":
                return ""

    # graphiti (default) or 'both' fallback
    try:
        from app.services.memory.graphiti_client import search_user_memory as _g_search

        return await _g_search(user_id, query, top_k=top_k)
    except Exception as exc:
        logger.warning(f"[memory_backend] graphiti recall failed: {exc}")
        return ""


async def add_conversation_episode(
    user_id: uuid.UUID | int | str,
    user_message: str,
    assistant_message: str | None = None,
    session_id: str | None = None,
) -> None:
    """Persist the turn to every selected backend.

    - graphiti   → Graphiti episode (user turn only, LLM fact-extraction)
    - openviking → OpenViking session commit (async memory extraction server-side)
    - both       → both writes; a failure in one backend does NOT fail the other
    - none       → no-op
    """
    if not user_message or not user_message.strip():
        return

    sel = memory_backend()
    if sel == "none":
        return

    errors: list[Exception] = []

    if sel in ("graphiti", "both"):
        try:
            from app.services.memory.graphiti_client import add_conversation_episode as _g_add

            await _g_add(
                user_id=user_id,
                user_message=user_message,
                assistant_message=assistant_message or "",
                session_id=session_id,
            )
        except Exception as exc:
            errors.append(exc)
            logger.warning(f"[memory_backend] graphiti episode failed: {exc}")

    if sel in ("openviking", "both"):
        try:
            from app.services.memory.openviking_client import (
                add_conversation_episode as _ov_add,
            )

            await _ov_add(
                user_id=user_id,
                user_message=user_message,
                assistant_message=assistant_message or "",
                session_id=session_id,
            )
        except Exception as exc:
            errors.append(exc)
            logger.warning(f"[memory_backend] openviking episode failed: {exc}")

    if errors and sel != "both":
        # Re-raise on single-backend mode so the memory worker's durable retry
        # / DLQ still works; in 'both' mode we already logged and swallowed.
        raise errors[0]


async def save_user_fact(user_id: uuid.UUID | int | str, fact: str) -> None:
    sel = memory_backend()
    if sel == "none":
        return
    if sel in ("openviking", "both"):
        try:
            from app.services.memory.openviking_client import save_user_fact as _ov_save

            await _ov_save(user_id, fact)
        except Exception as exc:
            logger.warning(f"[memory_backend] openviking save_user_fact failed: {exc}")
            if sel == "openviking":
                raise
    if sel in ("graphiti", "both"):
        try:
            from app.services.memory.graphiti_client import save_user_fact as _g_save

            await _g_save(user_id, fact)
        except Exception as exc:
            logger.warning(f"[memory_backend] graphiti save_user_fact failed: {exc}")
            if sel == "graphiti":
                raise


def save_user_fact_background(user_id: uuid.UUID | int | str, fact: str) -> None:
    sel = memory_backend()
    if sel == "none":
        return
    # Fire-and-forget per backend; the Graphiti variant already exists and the
    # OpenViking variant schedules its own task, so just call both dispatchers.
    try:
        from app.services.memory.graphiti_client import (
            save_user_fact_background as _g_bg,
        )

        if sel in ("graphiti", "both"):
            _g_bg(user_id, fact)
    except Exception as exc:
        logger.warning(f"[memory_backend] graphiti background save failed: {exc}")
    try:
        from app.services.memory.openviking_client import (
            save_user_fact_background as _ov_bg,
        )

        if sel in ("openviking", "both"):
            _ov_bg(user_id, fact)
    except Exception as exc:
        logger.warning(f"[memory_backend] openviking background save failed: {exc}")
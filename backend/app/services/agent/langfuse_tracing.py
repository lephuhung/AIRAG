"""
Langfuse Tracing Utilities
===========================

Shared helpers for manual Langfuse instrumentation across agent files.
Avoids code duplication and ensures consistent span naming/metadata.
"""

from __future__ import annotations

import logging
from typing import Optional

from langfuse import get_client

logger = logging.getLogger(__name__)


def _get_langfuse_client():
    """Get or create Langfuse client for manual span instrumentation."""
    try:
        return get_client()
    except Exception as e:
        logger.warning(f"[langfuse] Failed to get client: {e}")
        return None


class _NullContext:
    """Null context manager for when Langfuse is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, **kwargs):
        pass

    def end(self, **kwargs):
        pass


async def with_langfuse_span(name: str, input_data: dict, coro):
    """
    Execute an async coroutine within a Langfuse observation (SDK v4).

    Usage:
        result = await with_langfuse_span(
            "search_documents",
            {"query": q, "workspace_ids": [...]},
            search_documents(...),
        )

    Returns:
        The result of coro, whether or not Langfuse is available.
    """
    langfuse = _get_langfuse_client()
    if not langfuse:
        return await coro

    try:
        obs = langfuse.start_observation(
            name=name,
            input=input_data,
            level="DEFAULT",
        )
        result = await coro
        obs.update(output={"result": result})
        obs.end()
        return result
    except Exception as e:
        logger.warning(f"[langfuse] Span failed for {name}: {e}")
        return await coro


def langfuse_span_sync(name: str, input_data: dict, coro):
    """
    Synchronous wrapper for Langfuse spans (for non-async coroutines).
    """
    langfuse = _get_langfuse_client()
    if not langfuse:
        return coro

    try:
        obs = langfuse.start_observation(
            name=name,
            input=input_data,
            level="DEFAULT",
        )
        result = coro
        obs.update(output={"result": result})
        obs.end()
        return result
    except Exception as e:
        logger.warning(f"[langfuse] Span sync failed for {name}: {e}")
        return coro
"""
Parse Timing
============
Lightweight per-stage wall-clock instrumentation for the document parse
pipeline (Docling / OCR / legacy).  The goal is a single queryable answer
to "which stage ate the time" per document — no external tracing stack.

Usage::

    timer = ParseTimer()
    with timer.stage("docling_convert_ms"):
        conv_result = converter.convert(path)
    timer.merge(parsed.timing)
    summary = timer.summary()   # {"docling_convert_ms": 1234, ...}

``stage()`` is a plain synchronous context manager — it works around both
sync and ``await`` blocks (the ``with`` only brackets the wall-clock span;
it does not need to be async itself).  The timer is not thread-safe by
design: each parse runs its stages sequentially (even inside
``asyncio.to_thread``), so a single dict suffices.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


class ParseTimer:
    """Accumulate named stage durations (ms) for one parse run."""

    def __init__(self) -> None:
        self._timings: dict[str, int] = {}
        self._meta: dict[str, Any] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Record wall-clock ms spent inside the block under ``name``.

        Re-entering the same name accumulates (e.g. a stage that runs in
        two places).  Names should end in ``_ms`` for readability.
        """
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._timings[name] = self._timings.get(name, 0) + int(
                (time.monotonic() - t0) * 1000
            )

    def set(self, key: str, value: Any) -> None:
        """Attach a non-timing fact (page count, backend, file size…)."""
        self._meta[key] = value

    def merge(self, other: dict[str, Any] | None) -> None:
        """Merge a sub-pipeline's timing dict (e.g. ``ParsedDocument.timing``)."""
        if not other:
            return
        for k, v in other.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self._timings[k] = self._timings.get(k, 0) + int(v)
            else:
                self._meta[k] = v

    def summary(self) -> dict[str, Any]:
        """Flat ``{stage_ms: int, meta: value}`` dict — JSONB-serialisable."""
        return {**self._timings, **self._meta}

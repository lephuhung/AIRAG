"""Per-turn timing spans for the v2 agent graph (Network-tab-style data).

One :class:`TimingRecorder` is installed on a ContextVar at the start of a
v2 turn (``stream_v2_turn_events``) so the graph task and every downstream
tap (capability dispatch, retrieval stages) inherit it — same propagation
contract as :mod:`trace_collector`. Spans form a tree via ``parent_span``
labels: the ``turn`` root owns ``node:*`` spans, ``node:execute`` owns
``dispatch:*`` spans, a dispatch owns ``stage:*`` spans.

Content-suppression policy (same as ``synthesis_tracing``): only span
names, durations, statuses and numeric/count meta are recorded — never
query text, chunk content, prompts, or answers.

Persistence is best-effort in its own session (same contract as
``AgentTraceService``): a flush failure never breaks the chat response.
Gated by ``settings.NEXUSRAG_V2_TIMING`` (default True).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_recorder_ctx: ContextVar[Optional["TimingRecorder"]] = ContextVar(
    "_timing_recorder", default=None
)


def timing_enabled() -> bool:
    try:
        from app.core.config import settings

        return bool(getattr(settings, "NEXUSRAG_V2_TIMING", True))
    except Exception:
        return False


def get_recorder() -> Optional["TimingRecorder"]:
    """Return the recorder for the current turn, or None when off."""
    return _recorder_ctx.get()


def set_recorder(recorder: "TimingRecorder | None"):
    """Install a recorder on the current context. Returns a reset token."""
    return _recorder_ctx.set(recorder)


def reset_recorder(token) -> None:
    try:
        _recorder_ctx.reset(token)
    except Exception:  # pragma: no cover - defensive
        pass


class TimingRecorder:
    """Accumulates nested timing spans for one v2 turn.

    ``span()`` is the async-context-manager form for code paths that own
    both start and end (dispatch, retrieval stages). ``record_span()`` is
    the event form for spans whose start/end arrive as separate stream
    chunks (graph task events in ``_v2_run_graph``); it derives wall-clock
    ``started_at`` from the recorder's monotonic anchor so callers only
    deal in ``time.monotonic()`` readings.
    """

    def __init__(self, *, run_id: str, thread_id: str):
        self.run_id = run_id
        self.thread_id = thread_id
        self.spans: list[dict] = []
        self._stack: list[str] = []
        self._t0 = time.monotonic()
        self._started_at = datetime.now(timezone.utc)

    # ── internal ─────────────────────────────────────────────────────────
    def _parent(self) -> str | None:
        return self._stack[-1] if self._stack else None

    def _wall_at(self, monotonic_ts: float) -> datetime:
        return self._started_at + timedelta(seconds=monotonic_ts - self._t0)

    def _append(
        self,
        *,
        kind: str,
        name: str,
        started_monotonic: float,
        ended_monotonic: float,
        status: str,
        meta: dict | None,
        parent: str | None,
    ) -> None:
        self.spans.append(
            {
                "span_id": uuid.uuid4(),
                "run_id": self.run_id,
                "thread_id": self.thread_id,
                "parent_span": parent,
                "kind": kind,
                "name": name,
                "started_at": self._wall_at(started_monotonic),
                "duration_ms": max(
                    0, int((ended_monotonic - started_monotonic) * 1000)
                ),
                "status": status,
                "meta": meta or None,
            }
        )

    # ── span recorders ───────────────────────────────────────────────────
    @asynccontextmanager
    async def span(self, kind: str, name: str, meta: dict | None = None):
        """Time an async block as one span; exceptions mark it and re-raise."""
        label = f"{kind}:{name}"
        parent = self._parent()
        self._stack.append(label)
        started = time.monotonic()
        status = "ok"
        try:
            yield
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except BaseException:
            status = "error"
            raise
        finally:
            self._stack.pop()
            self._append(
                kind=kind,
                name=name,
                started_monotonic=started,
                ended_monotonic=time.monotonic(),
                status=status,
                meta=meta,
                parent=parent,
            )

    # ── external parent stack (event-driven spans) ─────────────────────────
    def push_label(self, label: str) -> None:
        """Push a span label so nested ``span()`` calls parent under it.

        Used for event-driven spans (graph task start/end chunks) where the
        span boundary is not a lexical ``async with`` block: the task-start
        chunk pushes ``node:<name>`` so a capability dispatch inside the
        node records ``parent_span='node:<name>'``; the task-end chunk pops.
        """
        self._stack.append(label)

    def pop_label(self, label: str) -> None:
        """Pop ``label`` if it is the innermost; otherwise remove it in place."""
        if self._stack and self._stack[-1] == label:
            self._stack.pop()
        elif label in self._stack:
            self._stack.remove(label)

    def record_span(
        self,
        kind: str,
        name: str,
        *,
        started_monotonic: float,
        ended_monotonic: float,
        status: str = "ok",
        meta: dict | None = None,
        parent: str | None = None,
    ) -> None:
        """Record a span whose start/end were observed as separate events."""
        self._append(
            kind=kind,
            name=name,
            started_monotonic=started_monotonic,
            ended_monotonic=ended_monotonic,
            status=status,
            meta=meta,
            parent=parent if parent is not None else self._parent(),
        )

    def close_turn(self, *, status: str = "ok") -> None:
        """Append the root ``turn`` span covering the whole run."""
        self._append(
            kind="turn",
            name="turn",
            started_monotonic=self._t0,
            ended_monotonic=time.monotonic(),
            status=status,
            meta=None,
            parent=None,
        )

    # ── persistence ──────────────────────────────────────────────────────
    async def flush(self, session_factory: Any | None = None) -> int:
        """Insert all recorded spans in one transaction; returns row count.

        Best-effort: any failure is logged and swallowed — timing data must
        never break the chat response.
        """
        if not self.spans:
            return 0
        if session_factory is None:
            try:
                from app.core.database import async_session_maker

                session_factory = async_session_maker
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[timing] no session factory: %s", exc)
                return 0
        rows = [
            {
                "span_id": str(s["span_id"]),
                "run_id": s["run_id"],
                "thread_id": s["thread_id"],
                "parent_span": s["parent_span"],
                "kind": s["kind"],
                "name": s["name"],
                "started_at": s["started_at"],
                "duration_ms": s["duration_ms"],
                "status": s["status"],
                "meta": json.dumps(s["meta"]) if s["meta"] is not None else None,
            }
            for s in self.spans
        ]
        try:
            from sqlalchemy import text

            async with session_factory() as session:
                await session.execute(
                    text(
                        "INSERT INTO agent_timing_spans ("
                        "span_id, run_id, thread_id, parent_span, kind, name, "
                        "started_at, duration_ms, status, meta"
                        ") VALUES ("
                        "CAST(:span_id AS UUID), :run_id, :thread_id, "
                        ":parent_span, :kind, :name, :started_at, "
                        ":duration_ms, :status, CAST(:meta AS JSONB))"
                    ),
                    rows,
                )
                await session.commit()
            return len(rows)
        except Exception as exc:
            logger.warning("[timing] span flush failed (non-fatal): %s", exc)
            return 0

"""
TraceCollector — per-run accumulator for the agent distillation dataset.
=========================================================================

Mirrors the ContextVar pattern already used for the SSE event queue
(``app/services/agent/streaming.py``): a single ``TraceCollector`` is set on a
ContextVar at the start of a run (in ``stream_agent_events``) and read from
anywhere downstream — supervisor routing, the TracedLLMProvider wrapper, and the
tool dispatchers — *without* threading it through LangGraph state (which strips
unknown keys). ``asyncio.create_task`` copies the context, so the background
graph task inherits the collector.

Every tap is best-effort and a no-op when:
  * ``settings.NEXUSRAG_TRACE_DATASET`` is False, or
  * no collector is set on the current context (e.g. a KG worker calling the
    shared LLM provider outside any chat run).

System prompts are intentionally NOT stored verbatim (they are static text in
``app/prompts/``): each LLM call keeps only ``prompt_ref`` = {sha256[:12], chars}
so the dataset stays lean. The dynamic messages (user turn, history, tool
results) and the completion ARE stored — that is the distillation signal.

PII redaction is applied later, centrally, in ``AgentTraceService.record``.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cap on any single stored text blob so a pathological tool result / context
# dump can't bloat a row. Generous — distillation wants the full prompt.
_MAX_TEXT = 60_000

_collector_ctx: ContextVar[Optional["TraceCollector"]] = ContextVar(
    "_trace_collector", default=None
)


def trace_enabled() -> bool:
    try:
        from app.core.config import settings

        return bool(getattr(settings, "NEXUSRAG_TRACE_DATASET", True))
    except Exception:
        return False


def get_collector() -> Optional["TraceCollector"]:
    """Return the collector for the current run, or None when tracing is off."""
    return _collector_ctx.get()


def set_collector(collector: "TraceCollector | None"):
    """Install a collector on the current context. Returns a reset token."""
    return _collector_ctx.set(collector)


def reset_collector(token) -> None:
    try:
        _collector_ctx.reset(token)
    except Exception:  # pragma: no cover - defensive
        pass


def _clip(value: Any) -> Any:
    """Truncate over-long strings; recurse shallowly into list/dict values."""
    if isinstance(value, str):
        return value if len(value) <= _MAX_TEXT else value[:_MAX_TEXT] + "…[truncated]"
    if isinstance(value, list):
        return [_clip(v) for v in value]
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    return value


def _prompt_ref(system_prompt: Optional[str]) -> Optional[dict]:
    if not system_prompt:
        return None
    h = hashlib.sha256(system_prompt.encode("utf-8", "ignore")).hexdigest()[:12]
    return {"sha256": h, "chars": len(system_prompt)}


class TraceCollector:
    """Accumulates one ordered trace for a single agent run."""

    def __init__(self, *, channel: str = "web", backend: str = "langgraph"):
        self.trace_id = uuid.uuid4()
        self.channel = channel
        self.backend = backend

        self.steps: list[dict] = []
        self._seq = 0
        self._t0 = time.monotonic()

        # Roll-ups surfaced as indexed columns / summary
        self.original_query: str = ""
        self.final_answer: str = ""
        self.intent: Optional[str] = None
        self.next_agent: Optional[str] = None
        self.query_complexity: Optional[str] = None
        self.success: bool = False
        self.error: Optional[str] = None
        self.meta: dict = {}
        self.token_usage: dict[str, int] = {}
        self.latency_ms: Optional[int] = None

    # ── internal ─────────────────────────────────────────────────────────────
    def _ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def _add(self, step_type: str, data: dict) -> None:
        self._seq += 1
        self.steps.append(
            {"seq": self._seq, "t_ms": self._ms(), "type": step_type, "data": _clip(data)}
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start_run(
        self,
        *,
        original_query: str,
        workspace_ids: list | None = None,
        document_ids: list | None = None,
        user_id: Any = None,
        session_id: Any = None,
        history_len: int = 0,
    ) -> None:
        self.original_query = original_query or ""
        self.meta = {
            "workspace_ids": [str(w) for w in (workspace_ids or [])],
            "document_ids": [str(d) for d in (document_ids or [])],
            "user_id": str(user_id) if user_id else None,
            "session_id": str(session_id) if session_id else None,
            "history_len": history_len,
        }

    def finish(
        self,
        *,
        final_answer: str | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        if final_answer is not None:
            self.final_answer = final_answer
        self.success = success
        if error:
            self.error = error
        self.latency_ms = self._ms()

    # ── step recorders (all best-effort) ──────────────────────────────────────
    def add_routing(self, *, node: str, **fields: Any) -> None:
        """Record a routing decision (supervisor classification or an edge router)."""
        try:
            # Keep the most informative roll-ups for the indexed columns.
            if fields.get("intent"):
                self.intent = fields["intent"]
            if fields.get("next_agent"):
                self.next_agent = fields["next_agent"]
            if fields.get("query_complexity"):
                self.query_complexity = fields["query_complexity"]
            self._add("routing", {"node": node, **fields})
        except Exception as e:  # pragma: no cover - never break the run
            logger.debug(f"[trace] add_routing failed: {e}")

    def add_llm_call(
        self,
        *,
        label: str,
        model: str | None,
        params: dict | None,
        messages: list | None,
        output: Any,
        usage: dict | None = None,
        system_prompt: str | None = None,
        error: str | None = None,
    ) -> None:
        try:
            data: dict = {
                "label": label,
                "model": model,
                "params": params or {},
                "prompt_ref": _prompt_ref(system_prompt),
                "messages": messages or [],
                "output": output,
            }
            if usage:
                data["usage"] = usage
                for k, v in usage.items():
                    if isinstance(v, (int, float)):
                        self.token_usage[k] = self.token_usage.get(k, 0) + int(v)
            if error:
                data["error"] = error
            self._add("llm_call", data)
        except Exception as e:  # pragma: no cover
            logger.debug(f"[trace] add_llm_call failed: {e}")

    def add_tool_call(
        self,
        *,
        name: str,
        args: dict | None,
        result_summary: str | None = None,
        sources_count: int = 0,
        images_count: int = 0,
        data: dict | None = None,
        latency_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        try:
            entry: dict = {
                "name": name,
                "args": args or {},
                "result_summary": result_summary or "",
                "sources_count": sources_count,
                "images_count": images_count,
            }
            if data:
                entry["data"] = data
            if latency_ms is not None:
                entry["latency_ms"] = latency_ms
            if error:
                entry["error"] = error
            self._add("tool_call", entry)
        except Exception as e:  # pragma: no cover
            logger.debug(f"[trace] add_tool_call failed: {e}")

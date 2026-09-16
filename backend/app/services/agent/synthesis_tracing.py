"""Content-suppressed LLM tracing for grounded answer synthesis (spec §7.2).

The synthesis prompt carries hydrated evidence plaintext and the completion
carries the claim proposal, so the generic ``TracedLLMProvider`` — which
serializes full messages, system prompt, and output into Langfuse and the
dataset ``TraceCollector`` — is forbidden on this path.

``ContentSuppressedLLMProvider`` emits the same observation shapes but with
content replaced by allowlisted operational metadata only: role, model,
message/character counts, token estimates, usage totals, latency, outcome /
failure code, repair flag, and cancellation. It never records query text,
evidence text, prompts, raw output, answer text, reasoning, or internal IDs.

Per-call metadata travels through a ContextVar (``synthesis_trace_metadata``)
because provider signatures accept no extra kwargs and the wrapper must stay a
drop-in ``LLMProvider``. All Langfuse/collector interaction is best-effort:
an exporter failure never changes the synthesis result.
"""
from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Optional

from app.services.agent.langfuse_tracing import (
    _get_langfuse_client,
    _model_name,
    _model_params,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ContentSuppressedLLMProvider",
    "record_synthesis_outcome",
    "reset_synthesis_trace_metadata",
    "synthesis_trace_metadata",
    "trace_llm_suppressed",
]

#: Allowlisted metadata keys (spec §7.2). Anything else is dropped.
_METADATA_ALLOWLIST = frozenset(
    {
        "role",
        "provider",
        "model",
        "evidence_count",
        "input_chars",
        "input_tokens_estimate",
        "claim_count",
        "latency_ms",
        "outcome",
        "failure_code",
        "repair",
        "cancelled",
    }
)

_metadata_ctx: ContextVar[dict] = ContextVar("_synthesis_trace_meta", default={})


def _clean_metadata(meta: dict | None) -> dict:
    """Keep only allowlisted keys; values must stay primitive/content-free."""
    out: dict = {}
    for key, value in (meta or {}).items():
        if key not in _METADATA_ALLOWLIST:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def synthesis_trace_metadata(**fields):
    """Install allowlisted per-call metadata for the next provider call.

    Returns the ContextVar token; pair with ``reset_synthesis_trace_metadata``.
    """
    return _metadata_ctx.set(_clean_metadata(fields))


def reset_synthesis_trace_metadata(token) -> None:
    try:
        _metadata_ctx.reset(token)
    except Exception:  # pragma: no cover - defensive
        pass


def _record_suppressed(inner, label, messages, system_prompt, temperature,
                       max_tokens, think, meta, latency_ms, usage,
                       outcome, error_type=None) -> None:
    """Dataset-collector record with all content fields suppressed."""
    try:
        from app.services.agent.trace_collector import get_collector

        coll = get_collector()
        if coll is None:
            return
        coll.add_llm_call(
            label=label,
            model=_model_name(inner),
            params=_model_params(temperature, max_tokens, think),
            messages=[],
            output={
                "suppressed": True,
                "outcome": outcome,
                "latency_ms": latency_ms,
                "message_count": len(messages or []),
                "input_chars": sum(
                    len(getattr(m, "content", "") or "")
                    for m in (messages or [])
                )
                + (len(system_prompt) if system_prompt else 0),
                **meta,
            },
            usage=usage,
            system_prompt=None,
            error=error_type,
        )
    except Exception:  # pragma: no cover - never break the LLM call
        pass


def record_synthesis_outcome(
    *,
    outcome: str,
    claim_count: int | None = None,
    failure_code: str | None = None,
    repair: bool = False,
    latency_ms: int | None = None,
    cancelled: bool = False,
) -> None:
    """Record the post-parse synthesis outcome into the dataset collector.

    Metadata-only: claim counts and closed codes, never claim text or raw
    output. Best-effort; a collector failure never affects the result.
    """
    try:
        from app.services.agent.trace_collector import get_collector

        coll = get_collector()
        if coll is None:
            return
        coll.add_llm_call(
            label="synthesis.outcome",
            model=None,
            params=None,
            messages=[],
            output=_clean_metadata(
                {
                    "outcome": outcome,
                    "claim_count": claim_count,
                    "failure_code": failure_code,
                    "repair": repair,
                    "latency_ms": latency_ms,
                    "cancelled": cancelled,
                }
            ),
            usage=None,
            system_prompt=None,
            error=None,
        )
    except Exception:  # pragma: no cover - never break the run
        pass


class ContentSuppressedLLMProvider:
    """``LLMProvider`` wrapper that traces metadata only — never content.

    Langfuse generation input/output and the dataset ``llm_call`` record carry
    counts, usage, latency, and outcome; message bodies, system prompts,
    completions, and thinking are never serialized. Unknown attributes
    delegate to the inner provider.
    """

    def __init__(self, inner, label: str = "synthesis_llm") -> None:
        self._inner = inner
        self._label = label

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _start_gen(self, meta, temperature, max_tokens, think):
        try:
            lf = _get_langfuse_client()
            if lf is None:
                return None
            return lf.start_observation(
                name=f"llm.{self._label}",
                as_type="generation",
                model=_model_name(self._inner),
                input={"suppressed": True, **meta},
                model_parameters=_model_params(temperature, max_tokens, think),
            )
        except Exception:  # pragma: no cover - exporter must not break calls
            return None

    @staticmethod
    def _safe_update(gen, **kwargs) -> None:
        if gen is None:
            return
        try:
            gen.update(**kwargs)
        except Exception:  # pragma: no cover
            pass

    @staticmethod
    def _safe_end(gen) -> None:
        if gen is None:
            return
        try:
            gen.end()
        except Exception:  # pragma: no cover
            pass

    @staticmethod
    def _read_usage(inner) -> Optional[dict]:
        usage = getattr(inner, "_last_usage", None)
        return usage if isinstance(usage, dict) and usage else None

    def complete(self, messages, *, temperature=0.0, max_tokens=4096,
                 system_prompt=None, think=False, **kwargs):
        meta = _clean_metadata(_metadata_ctx.get())
        started = time.monotonic()
        gen = self._start_gen(meta, temperature, max_tokens, think)
        try:
            result = self._inner.complete(
                messages, temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, think=think, **kwargs)
            usage = self._read_usage(self._inner)
            latency_ms = int((time.monotonic() - started) * 1000)
            self._safe_update(
                gen,
                output={"suppressed": True, "outcome": "ok", **meta},
                usage_details=usage,
            )
            _record_suppressed(
                self._inner, self._label, messages, system_prompt,
                temperature, max_tokens, think, meta, latency_ms, usage, "ok")
            return result
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            error_type = type(exc).__name__
            self._safe_update(gen, level="ERROR", status_message=error_type)
            _record_suppressed(
                self._inner, self._label, messages, system_prompt,
                temperature, max_tokens, think, meta, latency_ms, None,
                "error", error_type=error_type)
            raise
        finally:
            self._safe_end(gen)

    async def acomplete(self, messages, *, temperature=0.0, max_tokens=4096,
                        system_prompt=None, think=False, **kwargs):
        meta = _clean_metadata(_metadata_ctx.get())
        started = time.monotonic()
        gen = self._start_gen(meta, temperature, max_tokens, think)
        try:
            result = await self._inner.acomplete(
                messages, temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, think=think, **kwargs)
            usage = self._read_usage(self._inner)
            latency_ms = int((time.monotonic() - started) * 1000)
            self._safe_update(
                gen,
                output={"suppressed": True, "outcome": "ok", **meta},
                usage_details=usage,
            )
            _record_suppressed(
                self._inner, self._label, messages, system_prompt,
                temperature, max_tokens, think, meta, latency_ms, usage, "ok")
            return result
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            error_type = type(exc).__name__
            self._safe_update(gen, level="ERROR", status_message=error_type)
            _record_suppressed(
                self._inner, self._label, messages, system_prompt,
                temperature, max_tokens, think, meta, latency_ms, None,
                "error", error_type=error_type)
            raise
        finally:
            self._safe_end(gen)

    async def astream(self, messages, *, temperature=0.0, max_tokens=4096,
                      system_prompt=None, think=False, **kwargs):
        """Streamed variant of ``acomplete`` — chunks pass through verbatim.

        Same content suppression: only operational metadata is traced or
        recorded; chunk text never enters Langfuse or the collector.
        """
        meta = _clean_metadata(_metadata_ctx.get())
        started = time.monotonic()
        gen = self._start_gen(meta, temperature, max_tokens, think)
        try:
            async for chunk in self._inner.astream(
                messages, temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, think=think, **kwargs):
                yield chunk
            usage = self._read_usage(self._inner)
            latency_ms = int((time.monotonic() - started) * 1000)
            self._safe_update(
                gen,
                output={"suppressed": True, "outcome": "ok", **meta},
                usage_details=usage,
            )
            _record_suppressed(
                self._inner, self._label, messages, system_prompt,
                temperature, max_tokens, think, meta, latency_ms, usage, "ok")
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            error_type = type(exc).__name__
            self._safe_update(gen, level="ERROR", status_message=error_type)
            _record_suppressed(
                self._inner, self._label, messages, system_prompt,
                temperature, max_tokens, think, meta, latency_ms, None,
                "error", error_type=error_type)
            raise
        finally:
            self._safe_end(gen)


def trace_llm_suppressed(inner, label: str = "synthesis_llm"):
    """Wrap an ``LLMProvider`` with content-suppressed synthesis tracing.

    No-op (returns ``inner`` unchanged) when ``LANGFUSE_TRACE_LLM`` is
    disabled, mirroring ``trace_llm``. An already content-traced provider is
    never re-wrapped — suppression must not be weakened by stacking the
    generic full-content tracer on top.
    """
    try:
        from app.core.config import settings

        if not getattr(settings, "LANGFUSE_TRACE_LLM", True):
            return inner
    except Exception:
        pass
    if isinstance(inner, ContentSuppressedLLMProvider):
        return inner
    return ContentSuppressedLLMProvider(inner, label=label)

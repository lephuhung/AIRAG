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


# ===========================================================================
# LLM provider tracing wrapper
# ===========================================================================
#
# All LLM calls in the LangGraph supervisor go through custom provider classes
# (app/services/llm/) that wrap raw SDKs (google.genai / ollama / openai+httpx),
# NOT LangChain ChatModels. The Langfuse LangChain CallbackHandler therefore
# cannot see any prompt, completion, model, latency or token usage.
#
# TracedLLMProvider wraps any LLMProvider and emits a Langfuse "generation"
# observation for every complete()/acomplete()/astream() call. When invoked
# inside a LangGraph node (during a traced graph.ainvoke), the generation nests
# under the current node span automatically; outside a trace it becomes its own
# observation. Falls back to the inner provider verbatim if Langfuse is down.


def _model_name(inner) -> Optional[str]:
    for attr in ("_model", "model", "_model_name", "model_name"):
        val = getattr(inner, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


def _serialize_messages(messages, system_prompt: Optional[str]) -> list[dict]:
    """Serialize LLMMessage list (+ system prompt) into JSON-friendly dicts.

    Logs full text content; images/raw provider content are reduced to a
    compact placeholder so the payload stays serializable.
    """
    out: list[dict] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    for m in messages or []:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
            images = m.get("images")
            tool_calls = m.get("tool_calls")
        else:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
            images = getattr(m, "images", None)
            tool_calls = getattr(m, "tool_calls", None)
        entry: dict = {"role": role, "content": content}
        if images:
            entry["images"] = f"[{len(images)} image(s)]"
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.append(entry)
    return out


def _model_params(temperature, max_tokens, think) -> dict:
    params = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if think is not None:
        params["think"] = think
    return params


class TracedLLMProvider:
    """Transparent wrapper that traces an inner LLMProvider's calls in Langfuse.

    Unknown attributes/methods delegate to the inner provider, so this is a
    drop-in replacement anywhere an LLMProvider is expected.
    """

    def __init__(self, inner, label: str = "llm"):
        self._inner = inner
        self._label = label

    # Delegate everything not explicitly overridden (supports_vision, etc.)
    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _start_gen(self, lf, messages, system_prompt, temperature, max_tokens, think):
        return lf.start_observation(
            name=f"llm.{self._label}",
            as_type="generation",
            model=_model_name(self._inner),
            input=_serialize_messages(messages, system_prompt),
            model_parameters=_model_params(temperature, max_tokens, think),
        )

    @staticmethod
    def _read_usage(inner) -> Optional[dict]:
        usage = getattr(inner, "_last_usage", None)
        return usage if isinstance(usage, dict) and usage else None

    @staticmethod
    def _result_output(result):
        # complete()/acomplete() return str or LLMResult
        content = getattr(result, "content", None)
        if content is not None:
            thinking = getattr(result, "thinking", "") or ""
            out = {"content": content}
            if thinking:
                out["thinking"] = thinking
            return out
        return result

    def complete(self, messages, *, temperature=0.0, max_tokens=4096,
                 system_prompt=None, think=False, **kwargs):
        lf = _get_langfuse_client()
        if not lf:
            return self._inner.complete(
                messages, temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, think=think, **kwargs)
        gen = self._start_gen(lf, messages, system_prompt, temperature, max_tokens, think)
        try:
            result = self._inner.complete(
                messages, temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, think=think, **kwargs)
            gen.update(output=self._result_output(result),
                       usage_details=self._read_usage(self._inner))
            return result
        except Exception as e:
            gen.update(level="ERROR", status_message=str(e))
            raise
        finally:
            gen.end()

    async def acomplete(self, messages, *, temperature=0.0, max_tokens=4096,
                        system_prompt=None, think=False, **kwargs):
        lf = _get_langfuse_client()
        if not lf:
            return await self._inner.acomplete(
                messages, temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, think=think, **kwargs)
        gen = self._start_gen(lf, messages, system_prompt, temperature, max_tokens, think)
        try:
            result = await self._inner.acomplete(
                messages, temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, think=think, **kwargs)
            gen.update(output=self._result_output(result),
                       usage_details=self._read_usage(self._inner))
            return result
        except Exception as e:
            gen.update(level="ERROR", status_message=str(e))
            raise
        finally:
            gen.end()

    async def astream(self, messages, *, temperature=0.0, max_tokens=4096,
                      system_prompt=None, think=False, **kwargs):
        lf = _get_langfuse_client()
        if not lf:
            async for chunk in self._inner.astream(
                messages, temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, think=think, **kwargs):
                yield chunk
            return
        gen = self._start_gen(lf, messages, system_prompt, temperature, max_tokens, think)
        text_parts: list[str] = []
        think_parts: list[str] = []
        tool_calls: list = []
        try:
            async for chunk in self._inner.astream(
                messages, temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, think=think, **kwargs):
                ctype = getattr(chunk, "type", None)
                if ctype == "text":
                    text_parts.append(getattr(chunk, "text", "") or "")
                elif ctype == "thinking":
                    think_parts.append(getattr(chunk, "text", "") or "")
                elif ctype == "function_call":
                    fc = getattr(chunk, "function_call", None)
                    if fc:
                        tool_calls.append(fc)
                yield chunk
            output: dict = {"content": "".join(text_parts)}
            if think_parts:
                output["thinking"] = "".join(think_parts)
            if tool_calls:
                output["tool_calls"] = tool_calls
            gen.update(output=output, usage_details=self._read_usage(self._inner))
        except Exception as e:
            gen.update(level="ERROR", status_message=str(e))
            raise
        finally:
            gen.end()


def trace_llm(inner, label: str = "llm"):
    """Wrap an LLMProvider with Langfuse generation tracing.

    No-op (returns inner unchanged) when LANGFUSE_TRACE_LLM is disabled or the
    provider is already wrapped.
    """
    try:
        from app.core.config import settings
        if not getattr(settings, "LANGFUSE_TRACE_LLM", True):
            return inner
    except Exception:
        pass
    if isinstance(inner, TracedLLMProvider):
        return inner
    return TracedLLMProvider(inner, label=label)
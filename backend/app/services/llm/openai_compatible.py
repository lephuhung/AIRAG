"""
OpenAI-Compatible LLM Provider
================================
Supports any endpoint implementing the OpenAI Chat Completions API:
- vLLM  (e.g. http://10.10.0.240:8000/v1)
- LM Studio, llama.cpp server, LiteLLM, etc.

Set in .env:
    LLM_PROVIDER=openai_compatible
    OPENAI_COMPATIBLE_BASE_URL=http://10.10.0.240:8000/v1
    OPENAI_COMPATIBLE_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
    OPENAI_COMPATIBLE_API_KEY=none          # bắt buộc có giá trị, nhưng có thể là bất kỳ string nào
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import AsyncGenerator, Optional

import httpx
import numpy as np

from app.services.llm.base import EmbeddingProvider, LLMProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk

logger = logging.getLogger(__name__)

# Strip <think>...</think> blocks
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


# Connection pool limits (shared singleton, thread-safe)
_HTTPX_LIMITS = None
_HTTPX_LOCK = threading.Lock()


def _get_httpx_limits():
    """Get or create shared httpx connection pool limits."""
    global _HTTPX_LIMITS
    if _HTTPX_LIMITS is None:
        with _HTTPX_LOCK:
            if _HTTPX_LIMITS is None:
                import httpx
                _HTTPX_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=120.0)
    return _HTTPX_LIMITS


def _to_openai_messages(
    messages: list[LLMMessage],
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Convert LLMMessage list to OpenAI-format message dicts."""
    result: list[dict] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    for msg in messages:
        # Tool result turn (function-calling round-trip).
        if msg.role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id or "",
                "content": msg.content or "",
            })
        elif msg.tool_calls:
            # Assistant turn that requested tool calls. content may be empty.
            entry: dict = {"role": msg.role, "tool_calls": msg.tool_calls}
            if msg.content:
                entry["content"] = msg.content
            result.append(entry)
        elif msg.images:
            # Multi-modal: build content list
            content: list[dict] = [{"type": "text", "text": msg.content}]
            for img in msg.images:
                import base64
                b64 = base64.b64encode(img.data).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            result.append({"role": msg.role, "content": content})
        else:
            result.append({"role": msg.role, "content": msg.content})
    return result


class OpenAICompatibleLLMProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible HTTP endpoint (vLLM, LM Studio, etc.)."""

    def __init__(
        self,
        base_url: str = "http://10.10.0.240:8000/v1",
        model: str = "default",
        api_key: str = "none",
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._sync_client_instance: Optional[object] = None
        self._async_client_instance: Optional[object] = None
        # Token usage from the most recent call — read by the Langfuse tracing
        # wrapper (TracedLLMProvider) to populate generation usage_details.
        self._last_usage: Optional[dict] = None

    @staticmethod
    def _usage_dict(usage) -> Optional[dict]:
        """Map an OpenAI usage object to Langfuse usage_details (input/output/total)."""
        if usage is None:
            return None
        try:
            raw = {
                "input": getattr(usage, "prompt_tokens", None),
                "output": getattr(usage, "completion_tokens", None),
                "total": getattr(usage, "total_tokens", None),
            }
            cleaned = {k: v for k, v in raw.items() if isinstance(v, int)}
            return cleaned or None
        except Exception:
            return None

    def _sync_client(self):
        if self._sync_client_instance is None:
            from openai import OpenAI
            limits = _get_httpx_limits()
            self._sync_client_instance = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=httpx.Client(limits=limits),
            )
        return self._sync_client_instance

    def _async_client(self):
        if self._async_client_instance is None:
            from openai import AsyncOpenAI
            limits = _get_httpx_limits()
            self._async_client_instance = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=httpx.AsyncClient(limits=limits),
            )
        return self._async_client_instance

    @staticmethod
    def _strip_think(text: str) -> str:
        if "<think>" in text:
            text = _THINK_RE.sub("", text).strip()
        return text

    @staticmethod
    def _parse_xml_tool_call(xml_str: str) -> dict | None:
        """Fallback parser for Qwen-style XML tool calls: <function=name><parameter=key>val</parameter></function>"""
        func_match = re.search(r"<function=([^>]+)>(.*?)</function>", xml_str, re.DOTALL)
        if func_match:
            func_name = func_match.group(1).strip()
            params_str = func_match.group(2)
            args = {}
            for param_match in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", params_str, re.DOTALL):
                args[param_match.group(1).strip()] = param_match.group(2).strip()
            return {"name": func_name, "args": args}
        return None

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        oai_msgs = _to_openai_messages(messages, system_prompt)
        self._last_usage = None
        try:
            client = self._sync_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=oai_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": think}},
            )
            self._last_usage = self._usage_dict(getattr(response, "usage", None))
            content = response.choices[0].message.content or ""
            content = self._strip_think(content)
            return content
        except Exception as e:
            logger.error(f"OpenAI-compatible complete() failed: {e}", exc_info=True)
            return ""

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        oai_msgs = _to_openai_messages(messages, system_prompt)
        self._last_usage = None
        try:
            client = self._async_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=oai_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": think}},
            )
            self._last_usage = self._usage_dict(getattr(response, "usage", None))
            content = response.choices[0].message.content or ""
            content = self._strip_think(content)
            return content
        except Exception as e:
            logger.error(f"OpenAI-compatible acomplete() failed: {e}", exc_info=True)
            return ""

    async def astream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
        tools: list | None = None,
        tool_choice: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        oai_msgs = _to_openai_messages(messages, system_prompt)
        self._last_usage = None
        # Qwen reasoning models DEGENERATE into verbatim repetition when thinking
        # is driven at a near-greedy temperature — a single turn can emit 40k+
        # chars of looping reasoning and blow the whole latency/token budget
        # (observed on Qwen3.6 with temperature=0.1). Qwen's guidance for
        # thinking mode is temp≈0.6 / top_p≈0.95; adding a presence_penalty
        # collapses the repetition (measured: 8.7k→0.9k reasoning chars, 22s→2s).
        # Apply these ONLY when thinking is on, so non-thinking calls keep the
        # caller's deterministic sampling.
        if think:
            temperature = max(temperature, 0.6)
        kwargs: dict = dict(
            model=self._model,
            messages=oai_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            # Ask the server to emit a final usage-only chunk (vLLM/OpenAI support
            # this); captured below for Langfuse token accounting.
            stream_options={"include_usage": True},
        )
        if think:
            kwargs["top_p"] = 0.95
            kwargs["presence_penalty"] = 1.0
        if tools:
            kwargs["tools"] = tools
            # Allow the model to emit several independent tool calls in one turn
            # (executor runs them concurrently). vLLM honours this for
            # tool-calling-capable models; harmless if the server ignores it.
            kwargs["parallel_tool_calls"] = True
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        # Qwen3.5 vLLM: use chat_template_kwargs to control thinking
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": think}}

        try:
            client = self._async_client()
            # State machine for <tool_call>...</tool_call> detection
            tool_buffer = ""
            in_tool_call = False
            think_buffer = ""
            in_think = False

            # Buffer for text that comes before a tool call in native OpenAI mode
            pre_tool_text = ""

            # Native (OpenAI function-calling) tool calls arrive FRAGMENTED across
            # deltas: the name is sent once, the arguments stream in pieces. They
            # MUST be accumulated by index and parsed once at the end — parsing
            # each fragment independently corrupts multi-token args (e.g. long
            # Vietnamese queries). Keyed by tool-call index → {"name", "args"}.
            native_tool_calls: dict[int, dict] = {}

            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                # Final usage-only chunk has empty choices but carries token counts.
                if getattr(chunk, "usage", None):
                    self._last_usage = self._usage_dict(chunk.usage)
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Handle native tool_calls (OpenAI function calling).
                # Accumulate fragments by index here; the assembled calls are
                # emitted after the stream completes (see flush below).
                if delta.tool_calls:
                    if pre_tool_text.strip():
                        yield StreamChunk(type="thinking", text=pre_tool_text)
                        pre_tool_text = ""
                    for tc in delta.tool_calls:
                        idx = tc.index if getattr(tc, "index", None) is not None else 0
                        slot = native_tool_calls.setdefault(idx, {"name": "", "args": ""})
                        if tc.function:
                            if tc.function.name:
                                slot["name"] = tc.function.name
                            if tc.function.arguments:
                                slot["args"] += tc.function.arguments
                    continue

                content = delta.content or ""
                # vLLM streams reasoning via delta.reasoning (Qwen3.5 thinking mode).
                # Some vLLM reasoning-parsers expose it as delta.reasoning_content
                # instead — accept either so thinking is never silently dropped.
                reasoning = (
                    getattr(delta, "reasoning", None)
                    or getattr(delta, "reasoning_content", None)
                    or ""
                )

                # Yield reasoning chunks first (before content in same delta)
                if reasoning:
                    yield StreamChunk(type="thinking", text=reasoning)

                if not content:
                    continue

                # Buffer content in case it's followed by a tool_call in the same turn
                pre_tool_text += content
                
                # We yield it normally, but if a tool_call follows, we've already yielded it.
                # Actually, the best way in native mode is to yield AFTER checking tool_calls
                # but chunks come sequentially. 

                # Handle <think>...</think> inline tags (e.g. QwQ, DeepSeek-R1)
                if in_think:
                    if "</think>" in content:
                        before_end, after_end = content.split("</think>", 1)
                        think_buffer += before_end
                        yield StreamChunk(type="thinking", text=think_buffer)
                        think_buffer = ""
                        in_think = False
                        # Terminator often followed by \n\n, strip it
                        content = after_end.lstrip("\n ")
                    else:
                        think_buffer += content
                        yield StreamChunk(type="thinking", text=content)
                        continue

                if "<think>" in content:
                    before, rest = content.split("<think>", 1)
                    if before:
                        yield StreamChunk(type="text", text=before)
                    if "</think>" in rest:
                        think_part, after = rest.split("</think>", 1)
                        yield StreamChunk(type="thinking", text=think_part)
                        content = after.lstrip("\n ")
                    else:
                        think_buffer = rest
                        in_think = True
                        yield StreamChunk(type="thinking", text=rest)
                        continue
                elif "</think>" in content:
                    # Model starts thinking without <think> tag but we found the end marker
                    before, after = content.split("</think>", 1)
                    if before:
                        yield StreamChunk(type="thinking", text=before)
                    content = after.lstrip("\n ")


                # Handle <tool_call>...</tool_call> XML tags (Qwen-style)
                if in_tool_call:
                    tool_buffer += content
                    if "</tool_call>" in tool_buffer or "</function>" in tool_buffer:
                        # 1) Try <tool_call> pattern
                        match = re.search(r"<tool_call>(.*?)</tool_call>", tool_buffer, re.DOTALL)
                        if match:
                            raw_str = match.group(1).strip()
                            try:
                                tool_data = json.loads(raw_str)
                                yield StreamChunk(
                                    type="function_call",
                                    function_call={
                                        "name": tool_data.get("name", ""),
                                        "args": tool_data.get("arguments", {}),
                                    },
                                )
                            except json.JSONDecodeError:
                                fallback = self._parse_xml_tool_call(raw_str)
                                if fallback:
                                    yield StreamChunk(type="function_call", function_call=fallback)
                                else:
                                    yield StreamChunk(type="text", text=tool_buffer)
                            after = tool_buffer.split("</tool_call>", 1)[1]
                            tool_buffer = ""
                            in_tool_call = False
                            if after.strip():
                                yield StreamChunk(type="text", text=after)
                            continue

                        # 2) Try raw <function=...>...</function> pattern if no <tool_call> wraps it
                        f_match = re.search(r"(<function=[^>]+>.*?</function>)", tool_buffer, re.DOTALL)
                        if f_match:
                            raw_str = f_match.group(1).strip()
                            fallback = self._parse_xml_tool_call(raw_str)
                            if fallback:
                                yield StreamChunk(type="function_call", function_call=fallback)
                            else:
                                yield StreamChunk(type="text", text=raw_str)
                            after = tool_buffer.split("</function>", 1)[1]
                            tool_buffer = ""
                            in_tool_call = False
                            # Only yield after if it doesn't contain a stray </tool_call>
                            after = after.replace("</tool_call>", "").strip()
                            if after:
                                yield StreamChunk(type="text", text=after)

                elif "<tool_call>" in content or "<function=" in content:
                    trigger = "<tool_call>" if "<tool_call>" in content else "<function="
                    before, rest = content.split(trigger, 1)
                    if before.strip() and before.strip() != "\n":
                        # Crucial FIX: If a tool call is about to happen, any text BEFORE it 
                        # should be treated as thinking/reasoning, NOT as the final answer.
                        # This prevents "hallucinated" answers from appearing before tool results.
                        yield StreamChunk(type="thinking", text=before)
                    in_tool_call = True
                    tool_buffer = trigger + rest
                    
                    if "</tool_call>" in tool_buffer or "</function>" in tool_buffer:
                        # Re-run same logic if it completes instantly in one chunk
                        match = re.search(r"<tool_call>(.*?)</tool_call>", tool_buffer, re.DOTALL)
                        if match:
                            raw_str = match.group(1).strip()
                            try:
                                tool_data = json.loads(raw_str)
                                yield StreamChunk(
                                    type="function_call",
                                    function_call={
                                        "name": tool_data.get("name", ""),
                                        "args": tool_data.get("arguments", {}),
                                    },
                                )
                            except json.JSONDecodeError:
                                fallback = self._parse_xml_tool_call(raw_str)
                                if fallback:
                                    yield StreamChunk(type="function_call", function_call=fallback)
                                else:
                                    yield StreamChunk(type="text", text=tool_buffer)
                            after = tool_buffer.split("</tool_call>", 1)[1]
                            tool_buffer = ""
                            in_tool_call = False
                            if after.strip():
                                yield StreamChunk(type="text", text=after)
                            continue
                        
                        f_match = re.search(r"(<function=[^>]+>.*?</function>)", tool_buffer, re.DOTALL)
                        if f_match:
                            raw_str = f_match.group(1).strip()
                            fallback = self._parse_xml_tool_call(raw_str)
                            if fallback:
                                yield StreamChunk(type="function_call", function_call=fallback)
                            else:
                                yield StreamChunk(type="text", text=raw_str)
                            after = tool_buffer.split("</function>", 1)[1]
                            tool_buffer = ""
                            in_tool_call = False
                            after = after.replace("</tool_call>", "").strip()
                            if after:
                                yield StreamChunk(type="text", text=after)
                else:
                    if "</tool_call>" in content:
                        content = content.replace("</tool_call>", "").strip()
                    if content:
                        yield StreamChunk(type="text", text=content)

            # Flush accumulated native tool calls — assemble fragmented args,
            # then parse once. Ordered by index to preserve the model's intended
            # call order (matters for parallel tool calls).
            for idx in sorted(native_tool_calls):
                slot = native_tool_calls[idx]
                if not slot["name"]:
                    continue
                try:
                    args = json.loads(slot["args"] or "{}")
                except json.JSONDecodeError:
                    logger.warning(
                        f"[openai_compatible] native tool_call args parse failed "
                        f"for {slot['name']!r}: {slot['args'][:200]!r}"
                    )
                    args = {}
                yield StreamChunk(
                    type="function_call",
                    function_call={"name": slot["name"], "args": args},
                )

            if in_tool_call and tool_buffer:
                yield StreamChunk(type="text", text=tool_buffer)

        except Exception as e:
            logger.error(f"OpenAI-compatible astream() failed: {e}", exc_info=True)
            yield StreamChunk(type="text", text="")

    def supports_vision(self) -> bool:
        # Most modern multimodal models on vLLM/compatible servers support vision
        return True

    def supports_thinking(self) -> bool:
        # Qwen3 and other thinking-capable models support extended reasoning
        return True


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embedding provider via OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        base_url: str = "http://10.10.0.240:8000/v1",
        model: str = "BAAI/bge-m3",
        api_key: str = "none",
        dimension: int = 1024,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._dimension = dimension
        self._sync_client: Optional[object] = None
        self._async_client: Optional[object] = None

    def _get_sync_client(self):
        if self._sync_client is None:
            from openai import OpenAI
            limits = _get_httpx_limits()
            self._sync_client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=httpx.Client(limits=limits),
            )
        return self._sync_client

    def _get_async_client(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            limits = _get_httpx_limits()
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=httpx.AsyncClient(limits=limits),
            )
        return self._async_client

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        client = self._get_sync_client()
        clean = [t.strip() or "[empty]" for t in texts]
        try:
            response = client.embeddings.create(model=self._model, input=clean)
            vecs = [d.embedding for d in response.data]
            arr = np.array(vecs, dtype=np.float32)
            if np.any(np.isnan(arr)):
                arr = np.nan_to_num(arr, nan=0.0)
            return arr
        except Exception as e:
            logger.error(f"OpenAI-compatible embed_sync failed: {e}")
            return np.zeros((len(texts), self._dimension), dtype=np.float32)

    async def embed(self, texts: list[str]) -> np.ndarray:
        client = self._get_async_client()
        clean = [t.strip() or "[empty]" for t in texts]
        try:
            response = await client.embeddings.create(model=self._model, input=clean)
            vecs = [d.embedding for d in response.data]
            arr = np.array(vecs, dtype=np.float32)
            if np.any(np.isnan(arr)):
                arr = np.nan_to_num(arr, nan=0.0)
            return arr
        except Exception as e:
            logger.error(f"OpenAI-compatible async embed failed: {e}")
            return np.zeros((len(texts), self._dimension), dtype=np.float32)

    def get_dimension(self) -> int:
        return self._dimension

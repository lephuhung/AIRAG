"""
OpenAI-Compatible LLM Provider
================================
Supports any endpoint implementing the OpenAI Chat Completions API:
- vLLM  (e.g. http://127.0.0.1:8000/v1)
- LM Studio, llama.cpp server, LiteLLM, etc.

Set in .env:
    LLM_PROVIDER=openai_compatible
    OPENAI_COMPATIBLE_BASE_URL=http://127.0.0.1:8000/v1
    OPENAI_COMPATIBLE_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
    OPENAI_COMPATIBLE_API_KEY=none          # bắt buộc có giá trị, nhưng có thể là bất kỳ string nào
"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncGenerator, Optional

import numpy as np

from app.services.llm.base import EmbeddingProvider, LLMProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk

logger = logging.getLogger(__name__)

# Strip <think>...</think> blocks
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _to_openai_messages(
    messages: list[LLMMessage],
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Convert LLMMessage list to OpenAI-format message dicts."""
    result: list[dict] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    for msg in messages:
        if msg.images:
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
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "default",
        api_key: str = "none",
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key

    def _sync_client(self):
        from openai import OpenAI
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _async_client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

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
        try:
            client = self._sync_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=oai_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
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
        try:
            client = self._async_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=oai_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
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
    ) -> AsyncGenerator[StreamChunk, None]:
        oai_msgs = _to_openai_messages(messages, system_prompt)
        kwargs: dict = dict(
            model=self._model,
            messages=oai_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools

        try:
            client = self._async_client()
            # State machine for <tool_call>...</tool_call> detection
            tool_buffer = ""
            in_tool_call = False
            think_buffer = ""
            in_think = False

            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Handle native tool_calls (OpenAI function calling)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function:
                            try:
                                args = json.loads(tc.function.arguments or "{}")
                            except json.JSONDecodeError:
                                args = {}
                            yield StreamChunk(
                                type="function_call",
                                function_call={
                                    "name": tc.function.name or "",
                                    "args": args,
                                },
                            )
                    continue

                content = delta.content or ""
                if not content:
                    continue

                # Handle <think>...</think> inline tags (e.g. QwQ, DeepSeek-R1)
                if in_think:
                    if "</think>" in content:
                        before_end, after_end = content.split("</think>", 1)
                        think_buffer += before_end
                        yield StreamChunk(type="thinking", text=think_buffer)
                        think_buffer = ""
                        in_think = False
                        content = after_end
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
                        content = after
                    else:
                        think_buffer = rest
                        in_think = True
                        yield StreamChunk(type="thinking", text=rest)
                        continue

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
                        yield StreamChunk(type="text", text=before)
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

            if in_tool_call and tool_buffer:
                yield StreamChunk(type="text", text=tool_buffer)

        except Exception as e:
            logger.error(f"OpenAI-compatible astream() failed: {e}", exc_info=True)
            yield StreamChunk(type="text", text="")

    def supports_vision(self) -> bool:
        # Most modern multimodal models on vLLM/compatible servers support vision
        return True

    def supports_thinking(self) -> bool:
        return False


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embedding provider via OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "BAAI/bge-m3",
        api_key: str = "none",
        dimension: int = 1024,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._dimension = dimension

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
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
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
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

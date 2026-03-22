"""
Local vLLM Provider
===================
Loads a vLLM engine in-process for low-latency, local inference.
Used when MEMORY_AGENT_LOCAL=true.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Optional

from app.services.llm.base import LLMProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk

logger = logging.getLogger(__name__)

class LocalVLLMProvider(LLMProvider):
    """vLLM engine running in-process."""

    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float = 0.1,
        max_model_len: int | None = None,
        cuda_device: str = "auto",
    ):
        self._model_name = model
        self._gpu_mem = gpu_memory_utilization
        self._max_len = max_model_len
        self._cuda_device = cuda_device
        
        self._llm = None
        self._processor = None
        self._sampling_params = None
        self._lock = asyncio.Lock()

    def _get_engine(self):
        """Lazy-load the vLLM engine."""
        if self._llm is not None:
            return self._llm, self._processor, self._sampling_params

        try:
            import os
            # Set CUDA_VISIBLE_DEVICES if specified
            if self._cuda_device not in ("", "auto"):
                os.environ["CUDA_VISIBLE_DEVICES"] = self._cuda_device
                logger.info(f"[vLLM/local] Forced CUDA_VISIBLE_DEVICES={self._cuda_device}")

            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor
        except ImportError:
            raise RuntimeError("Local vLLM requires 'vllm' and 'transformers' packages.")

        logger.info(f"[vLLM/local] Loading model {self._model_name} (gpu_mem={self._gpu_mem})")
        
        llm_kwargs = {
            "model": self._model_name,
            "trust_remote_code": True,
            "gpu_memory_utilization": self._gpu_mem,
            "tensor_parallel_size": 1,
            "enforce_eager": True,
        }
        if self._max_len:
            llm_kwargs["max_model_len"] = self._max_len

        self._llm = LLM(**llm_kwargs)
        self._processor = AutoProcessor.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        self._sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
        
        logger.info(f"[vLLM/local] Model {self._model_name} loaded successfully")
        return self._llm, self._processor, self._sampling_params

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        """Synchronous generation (running in thread pool via acomplete)."""
        llm, processor, _ = self._get_engine()
        
        from vllm import SamplingParams
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

        # Build chat input
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        
        for m in messages:
            msgs.append({"role": m.role, "content": m.content})
            
        prompt = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        
        outputs = llm.generate([prompt], params)
        text = outputs[0].outputs[0].text.strip()
        
        if think:
            return LLMResult(content=text, thinking="")
        return text

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        """Run complete() in a thread to avoid blocking the event loop."""
        async with self._lock:
            return await asyncio.to_thread(
                self.complete,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                think=think
            )

    async def astream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        think: bool = False,
        tools: list | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming fallback for local vLLM (returns full text as one chunk)."""
        result = await self.acomplete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            think=think
        )
        text = result.content if isinstance(result, LLMResult) else result
        yield StreamChunk(type="text", text=text)

    def supports_vision(self) -> bool:
        return False

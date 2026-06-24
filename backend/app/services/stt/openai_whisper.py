"""
OpenAI-compatible STT Provider
==============================
Optional provider that posts the audio to an OpenAI-compatible
``POST /v1/audio/transcriptions`` endpoint (OpenAI itself, or a self-hosted
vLLM serving Whisper). Uses the ``openai`` client already in requirements,
mirroring how ``LLM_PROVIDER=openai_compatible`` works elsewhere.
"""
from __future__ import annotations

import logging

from app.services.stt.base import STTProvider, STTResult

logger = logging.getLogger(__name__)


class OpenAIWhisperSTTProvider(STTProvider):
    """STT provider backed by an OpenAI-compatible transcriptions endpoint."""

    def __init__(
        self,
        base_url: str = "",
        model: str = "whisper-1",
        api_key: str = "",
    ):
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self._base_url or None,
                # vLLM ignores the key but the client requires a non-empty value.
                api_key=self._api_key or "sk-no-key",
            )
        return self._client

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str = "",
        filename: str = "audio.webm",
        language: str | None = None,
    ) -> STTResult:
        client = self._get_client()
        kwargs: dict = {"model": self._model, "file": (filename, audio, content_type or None)}
        if language:
            kwargs["language"] = language
        resp = await client.audio.transcriptions.create(**kwargs)
        return STTResult(
            text=(getattr(resp, "text", "") or "").strip(),
            language=language or "",
        )

"""
STT Provider Package
====================
Factory for the speech-to-text provider selected via ``STT_PROVIDER``.

Usage::

    from app.services.stt import get_stt_provider

    stt = get_stt_provider()                 # uses STT_PROVIDER from .env
    result = await stt.transcribe(audio_bytes, content_type="audio/webm")
"""
from __future__ import annotations

from functools import lru_cache

from app.services.stt.base import STTProvider, STTResult


@lru_cache
def get_stt_provider() -> STTProvider:
    """Create (and cache) the STT provider configured via ``STT_PROVIDER``."""
    from app.core.config import settings

    provider = settings.STT_PROVIDER.lower()

    if provider in ("faster_whisper", "faster-whisper", "fasterwhisper"):
        from app.services.stt.faster_whisper import FasterWhisperSTTProvider

        return FasterWhisperSTTProvider(
            model=settings.STT_FW_MODEL,
            device=settings.STT_FW_DEVICE,
            compute_type=settings.STT_FW_COMPUTE_TYPE,
            download_root=settings.STT_FW_MODEL_DIR,
        )

    if provider in ("openai", "openai_compatible"):
        from app.services.stt.openai_whisper import OpenAIWhisperSTTProvider

        return OpenAIWhisperSTTProvider(
            base_url=settings.STT_OPENAI_BASE_URL,
            model=settings.STT_OPENAI_MODEL,
            api_key=settings.STT_OPENAI_API_KEY,
        )

    raise ValueError(
        f"Unknown STT_PROVIDER: {provider!r}. Supported: faster_whisper, openai"
    )


__all__ = ["get_stt_provider", "STTProvider", "STTResult"]

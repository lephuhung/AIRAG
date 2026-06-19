"""
TTS Provider Package
=====================
Factory for the text-to-speech provider selected via ``TTS_PROVIDER``.

Usage::

    from app.services.tts import get_tts_provider

    tts = get_tts_provider()          # uses TTS_PROVIDER from .env
    result = await tts.synthesize("Xin chào", voice="female, young adult")
"""
from __future__ import annotations

from functools import lru_cache

from app.services.tts.base import TTSProvider, TTSResult, TTSVoice


@lru_cache
def get_tts_provider() -> TTSProvider:
    """Create (and cache) the TTS provider configured via ``TTS_PROVIDER``."""
    from app.core.config import settings

    provider = settings.TTS_PROVIDER.lower()

    if provider == "omnivoice":
        from app.services.tts.omnivoice import OmniVoiceTTSProvider

        return OmniVoiceTTSProvider(
            base_url=settings.TTS_OMNIVOICE_BASE_URL,
            model=settings.TTS_OMNIVOICE_MODEL,
            api_key=settings.TTS_OMNIVOICE_API_KEY,
        )

    raise ValueError(f"Unknown TTS_PROVIDER: {provider!r}. Supported: omnivoice")


__all__ = ["get_tts_provider", "TTSProvider", "TTSResult", "TTSVoice"]

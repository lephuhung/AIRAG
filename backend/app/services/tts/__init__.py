"""
TTS Provider Package
=====================
Factory for the text-to-speech provider.

Resolution order (V2 runtime-config, plan §12.4):
  1. DB override for role ``tts`` (WebUI connection) → OmniVoice-compatible
     endpoint with base_url/model/api_key from the effective config
     (OmniVoice speaks the OpenAI audio-speech dialect, so any compatible
     connection works through the same provider class).
  2. Otherwise legacy ``TTS_PROVIDER`` behaviour from .env, unchanged.

The provider instance is cached per runtime-config snapshot version: an admin
override on the WebUI takes effect on the next resolution without restart.

Usage::

    from app.services.tts import get_tts_provider

    tts = get_tts_provider()
    result = await tts.synthesize("Xin chào", voice="female, young adult")
"""
from __future__ import annotations

from app.services.tts.base import TTSProvider, TTSResult, TTSVoice

# (snapshot_version, provider) — rebuilt only when the runtime config changes.
_cached: tuple[int, TTSProvider] | None = None


def _build_tts_provider() -> TTSProvider:
    """Resolve the active TTS provider from runtime config / .env defaults."""
    from app.core.config import settings
    from app.services.runtime_config import get_effective_sync

    cfg = get_effective_sync("tts")

    if cfg.source == "db":
        if cfg.provider in ("openai_compatible", "omnivoice"):
            from app.services.tts.omnivoice import OmniVoiceTTSProvider

            return OmniVoiceTTSProvider(
                base_url=cfg.base_url,
                model=cfg.model or settings.TTS_OMNIVOICE_MODEL,
                api_key=cfg.api_key,
            )
        raise ValueError(
            f"Unsupported TTS connection provider: {cfg.provider!r}. "
            f"Supported: openai_compatible (OmniVoice-compatible endpoints)"
        )

    # ── Legacy .env behaviour (unchanged) ──────────────────────────────────
    provider = settings.TTS_PROVIDER.lower()

    if provider == "omnivoice":
        from app.services.tts.omnivoice import OmniVoiceTTSProvider

        return OmniVoiceTTSProvider(
            base_url=settings.TTS_OMNIVOICE_BASE_URL,
            model=settings.TTS_OMNIVOICE_MODEL,
            api_key=settings.TTS_OMNIVOICE_API_KEY,
        )

    raise ValueError(f"Unknown TTS_PROVIDER: {provider!r}. Supported: omnivoice")


def get_tts_provider() -> TTSProvider:
    """Get the active TTS provider (cached per runtime-config snapshot version)."""
    global _cached
    from app.services.runtime_config import snapshot_version

    ver = snapshot_version()
    if _cached is not None and _cached[0] == ver:
        return _cached[1]
    provider = _build_tts_provider()
    _cached = (ver, provider)
    return provider


__all__ = ["get_tts_provider", "TTSProvider", "TTSResult", "TTSVoice"]

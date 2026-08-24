"""
STT Provider Package
====================
Factory for the speech-to-text provider.

Resolution order (V2 runtime-config, plan §12.4):
  1. DB override for role ``stt`` (WebUI connection) when
     ``provider == "openai_compatible"`` → OpenAI-compatible transcriptions
     endpoint with base_url/model/api_key from the effective config.
  2. Otherwise legacy ``STT_PROVIDER`` behaviour from .env, unchanged:
     faster_whisper (local) | openai / openai_compatible.

The provider instance is cached per runtime-config snapshot version: an admin
override on the WebUI takes effect on the next resolution without restart.

Usage::

    from app.services.stt import get_stt_provider

    stt = get_stt_provider()
    result = await stt.transcribe(audio_bytes, content_type="audio/webm")
"""
from __future__ import annotations

from app.services.stt.base import STTProvider, STTResult

# (snapshot_version, provider) — rebuilt only when the runtime config changes.
_cached: tuple[int, STTProvider] | None = None


def _build_stt_provider() -> STTProvider:
    """Resolve the active STT provider from runtime config / .env defaults."""
    from app.core.config import settings
    from app.services.runtime_config import get_effective_sync

    cfg = get_effective_sync("stt")

    if cfg.source == "db":
        if cfg.provider == "openai_compatible":
            from app.services.stt.openai_whisper import OpenAIWhisperSTTProvider

            return OpenAIWhisperSTTProvider(
                base_url=cfg.base_url,
                model=cfg.model or settings.STT_OPENAI_MODEL,
                api_key=cfg.api_key,
            )
        if cfg.provider in ("faster_whisper", "faster-whisper", "fasterwhisper", "local"):
            # Local engine overridden via WebUI (model swap) — device params
            # still come from .env (they are hardware-bound).
            from app.services.stt.faster_whisper import FasterWhisperSTTProvider

            return FasterWhisperSTTProvider(
                model=cfg.model or settings.STT_FW_MODEL,
                device=settings.STT_FW_DEVICE,
                compute_type=settings.STT_FW_COMPUTE_TYPE,
                download_root=settings.STT_FW_MODEL_DIR,
            )
        raise ValueError(
            f"Unsupported STT connection provider: {cfg.provider!r}. "
            f"Supported: openai_compatible, faster_whisper"
        )

    # ── Legacy .env behaviour (unchanged) ──────────────────────────────────
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


def get_stt_provider() -> STTProvider:
    """Get the active STT provider (cached per runtime-config snapshot version)."""
    global _cached
    from app.services.runtime_config import snapshot_version

    ver = snapshot_version()
    if _cached is not None and _cached[0] == ver:
        return _cached[1]
    provider = _build_stt_provider()
    _cached = (ver, provider)
    return provider


__all__ = ["get_stt_provider", "STTProvider", "STTResult"]

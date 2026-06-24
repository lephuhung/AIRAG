"""
STT Provider Base Classes
=========================
Abstract interface for speech-to-text transcription. Mirrors the structure of
``app/services/tts/base.py`` (the inverse feature) so the engine can be swapped
(faster-whisper for local/offline today, an OpenAI-compatible server later)
without touching the API or frontend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class STTResult:
    """Transcribed text plus light metadata."""

    text: str
    language: str = ""  # detected (or forced) language code, e.g. "vi"
    duration: float = 0.0  # audio length in seconds (0 if unknown)


class STTProvider(ABC):
    """Abstract interface for speech-to-text transcription."""

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str = "",
        filename: str = "audio.webm",
        language: str | None = None,
    ) -> STTResult:
        """Transcribe ``audio`` bytes into text.

        Args:
            audio: Raw audio container bytes (webm/opus, wav, mp3, m4a, ogg ...).
            content_type: MIME type reported by the client (best-effort hint).
            filename: Original filename / extension hint for the provider.
            language: Force a language code (e.g. ``"vi"``); ``None`` → auto-detect.
        """
        ...

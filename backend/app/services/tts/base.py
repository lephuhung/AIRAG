"""
TTS Provider Base Classes
=========================
Abstract interface for text-to-speech synthesis. Mirrors the structure of
``app/services/llm/base.py`` so the engine can be swapped (OmniVoice today,
a Vietnamese-native engine later) without touching the API or frontend.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TTSVoice:
    """A selectable voice exposed to the UI.

    ``id`` is the provider-specific value passed back to ``synthesize(voice=...)``.
    For OmniVoice it is a design-prompt string (e.g. ``"female, young adult, moderate pitch"``).
    """

    id: str
    label: str
    gender: str = ""  # "male" | "female" | ""
    language: str = ""  # informational; OmniVoice is English/Chinese focused


@dataclass
class TTSResult:
    """Synthesized audio plus its container format."""

    audio: bytes
    media_type: str = "audio/wav"


class TTSProvider(ABC):
    """Abstract interface for text-to-speech synthesis."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "",
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> TTSResult:
        """Synthesize ``text`` into audio bytes.

        Args:
            text: Plain text to read aloud (caller strips markdown/citations).
            voice: Provider-specific voice id (see :meth:`list_voices`). Empty → default.
            speed: Speaking-rate multiplier (1.0 = normal).
            pitch: Pitch multiplier (1.0 = neutral); providers map this however fits.
        """
        ...

    @abstractmethod
    def list_voices(self) -> list[TTSVoice]:
        """Return the curated voices this provider offers to the UI."""
        ...

    async def alist_voices(self) -> list[TTSVoice]:
        """Async variant; default runs :meth:`list_voices` in a thread."""
        return await asyncio.to_thread(self.list_voices)

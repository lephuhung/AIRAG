"""
OmniVoice TTS Provider
======================
Talks to an `omnivoice-server` instance (https://github.com/maemreyo/omnivoice-server),
an OpenAI-compatible TTS server exposing ``POST /v1/audio/speech`` → WAV.

Voice model: OmniVoice is driven by an ``instructions`` design-prompt string built
from English-style attributes (gender / age / pitch / accent). We expose a curated
set of those prompts as :class:`TTSVoice` ids. ``speed`` maps to the native ``speed``
field; ``pitch`` is folded into the instruction prompt (OmniVoice has no numeric pitch).

NOTE: OmniVoice does not advertise Vietnamese support — quality on Vietnamese text is
unverified. If it is poor, add a sibling provider here; the API/UI stay unchanged.
"""
from __future__ import annotations

import logging

import httpx

from app.services.llm.openai_compatible import _get_httpx_limits
from app.services.tts.base import TTSProvider, TTSResult, TTSVoice

logger = logging.getLogger(__name__)


# Curated voices surfaced in the UI. ``id`` is an OmniVoice design prompt.
_VOICES: list[TTSVoice] = [
    TTSVoice(id="", label="Mặc định (server)", gender="", language="en"),
    TTSVoice(id="female, young adult, moderate pitch", label="Nữ trẻ", gender="female", language="en"),
    TTSVoice(id="female, middle-aged, moderate pitch", label="Nữ trung niên", gender="female", language="en"),
    TTSVoice(id="male, young adult, moderate pitch", label="Nam trẻ", gender="male", language="en"),
    TTSVoice(id="male, middle-aged, low pitch", label="Nam trầm", gender="male", language="en"),
]

# Map a numeric pitch multiplier to an OmniVoice pitch keyword.
def _pitch_keyword(pitch: float) -> str:
    if pitch <= 0.85:
        return "low pitch"
    if pitch >= 1.15:
        return "high pitch"
    return "moderate pitch"


class OmniVoiceTTSProvider(TTSProvider):
    """TTS provider backed by an omnivoice-server HTTP endpoint."""

    def __init__(
        self,
        base_url: str = "http://omnivoice:8880/v1",
        model: str = "omnivoice",
        api_key: str = "",
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                limits=_get_httpx_limits(),
                headers=headers,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    def list_voices(self) -> list[TTSVoice]:
        return list(_VOICES)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "",
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> TTSResult:
        body: dict = {
            "model": self._model,
            "input": text,
            "response_format": "wav",
        }
        if speed and speed != 1.0:
            body["speed"] = round(float(speed), 2)

        # Build the design-prompt. ``voice`` already encodes gender/age/pitch; when a
        # non-default pitch is requested and the voice prompt omits one, fold it in.
        instructions = voice.strip()
        if instructions and pitch and pitch != 1.0 and "pitch" not in instructions:
            instructions = f"{instructions}, {_pitch_keyword(pitch)}"
        if instructions:
            body["instructions"] = instructions

        client = self._get_client()
        resp = await client.post("/audio/speech", json=body)
        resp.raise_for_status()
        media_type = resp.headers.get("content-type", "audio/wav").split(";")[0].strip()
        return TTSResult(audio=resp.content, media_type=media_type or "audio/wav")

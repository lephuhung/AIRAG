"""
TTS API — read assistant answers aloud.

``GET  /tts/voices``     → curated voice list for the UI.
``POST /tts/synthesize`` → WAV audio for the given text.

Auth via ``get_principal`` (JWT or API key), mirroring the chat endpoints.
Per-user voice/speed preferences live in ``users.settings["tts"]`` and act as
the fallback when the request omits them.
"""
from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import Response

from app.core.config import settings
from app.core.deps import get_principal
from app.core.exceptions import BadRequestError
from app.models.user import User
from app.schemas.tts import (
    TTSSynthesizeRequest,
    TTSVoiceResponse,
    TTSVoicesResponse,
)
from app.services.tts import get_tts_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])


def _user_tts_prefs(user: User) -> dict:
    prefs = getattr(user, "settings", None) or {}
    tts = prefs.get("tts") if isinstance(prefs, dict) else None
    return tts if isinstance(tts, dict) else {}


@router.get("/voices", response_model=TTSVoicesResponse)
async def list_voices(user: User = Depends(get_principal)):
    """List voices offered by the active TTS provider."""
    voices: list[TTSVoiceResponse] = []
    if settings.TTS_ENABLED:
        try:
            provider = get_tts_provider()
            voices = [
                TTSVoiceResponse(id=v.id, label=v.label, gender=v.gender, language=v.language)
                for v in await provider.alist_voices()
            ]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("TTS list_voices failed: %s", exc)
    return TTSVoicesResponse(
        enabled=settings.TTS_ENABLED,
        provider=settings.TTS_PROVIDER,
        voices=voices,
    )


@router.post("/synthesize")
async def synthesize(
    body: TTSSynthesizeRequest,
    user: User = Depends(get_principal),
):
    """Synthesize speech for ``body.text`` and return the audio bytes."""
    if not settings.TTS_ENABLED:
        return Response(content=b"", status_code=503)

    text = body.text.strip()
    if not text:
        raise BadRequestError("text must not be empty")
    if len(text) > settings.TTS_MAX_CHARS:
        text = text[: settings.TTS_MAX_CHARS]

    prefs = _user_tts_prefs(user)
    voice = body.voice if body.voice is not None else prefs.get("voice", settings.TTS_DEFAULT_VOICE)
    speed = body.speed if body.speed is not None else float(prefs.get("speed", settings.TTS_DEFAULT_SPEED))
    pitch = body.pitch if body.pitch is not None else float(prefs.get("pitch", 1.0))

    provider = get_tts_provider()
    try:
        result = await provider.synthesize(text, voice=voice or "", speed=speed, pitch=pitch)
    except httpx.HTTPError as exc:
        logger.warning("TTS synthesize failed: %s", exc)
        return Response(content=b"", status_code=503)

    return Response(
        content=result.audio,
        media_type=result.media_type,
        headers={"Cache-Control": "no-store"},
    )

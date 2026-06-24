"""
STT API — transcribe voice recordings to text for the chat input.

``POST /stt/transcribe`` → text for an uploaded audio blob (browser MediaRecorder).
``GET  /stt/status``     → {enabled, provider} so the UI can show/hide the mic.

Auth via ``get_principal`` (JWT or API key), mirroring the chat / TTS endpoints.
The recording is transcribed in-process and discarded — it is NOT stored in MinIO
or the documents pipeline.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import settings
from app.core.deps import get_principal
from app.core.exceptions import BadRequestError
from app.models.user import User
from app.schemas.stt import STTStatusResponse, STTTranscribeResponse
from app.services.stt import get_stt_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt", tags=["stt"])

# Accepted audio MIME prefixes / extensions from the browser recorder + uploads.
_ALLOWED_CONTENT_TYPES = (
    "audio/webm",
    "audio/ogg",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/flac",
)


@router.get("/status", response_model=STTStatusResponse)
async def stt_status(user: User = Depends(get_principal)):
    """Report whether STT is enabled and which provider is active."""
    return STTStatusResponse(enabled=settings.STT_ENABLED, provider=settings.STT_PROVIDER)


@router.post("/transcribe", response_model=STTTranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    user: User = Depends(get_principal),
):
    """Transcribe an uploaded audio recording into text."""
    if not settings.STT_ENABLED:
        raise BadRequestError("Speech-to-text is disabled")

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and not content_type.startswith("audio/"):
        raise BadRequestError(f"Unsupported content type: {content_type!r}")

    audio = await file.read()
    if not audio:
        raise BadRequestError("Empty audio upload")

    max_bytes = settings.STT_MAX_UPLOAD_MB * 1024 * 1024
    if len(audio) > max_bytes:
        raise BadRequestError(
            f"Audio too large ({len(audio) // (1024 * 1024)}MB > {settings.STT_MAX_UPLOAD_MB}MB)"
        )

    provider = get_stt_provider()
    try:
        result = await provider.transcribe(
            audio,
            content_type=content_type,
            filename=file.filename or "audio.webm",
            language=settings.STT_LANGUAGE or None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("STT transcribe failed: %s", exc)
        raise BadRequestError("Transcription failed") from exc

    return STTTranscribeResponse(
        text=result.text,
        language=result.language,
        duration=result.duration,
    )

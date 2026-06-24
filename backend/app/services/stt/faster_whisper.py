"""
faster-whisper STT Provider
============================
Local / offline speech-to-text using `faster-whisper` (CTranslate2 Whisper).
This is the default engine — same spirit as ``handy`` (github.com/cjpais/handy),
which runs Whisper entirely on-device.

The model is loaded once, lazily, on first use and cached at module level.
``faster-whisper`` decodes the audio container itself via PyAV, so webm/opus
blobs from the browser ``MediaRecorder`` work directly — no external ffmpeg call.
Inference (CTranslate2) is blocking, so it runs in a worker thread.
"""
from __future__ import annotations

import asyncio
import io
import logging

from app.services.stt.base import STTProvider, STTResult

logger = logging.getLogger(__name__)

# Loaded once on first transcription; keyed by (model, device, compute_type).
_model = None
_model_key: tuple | None = None


def _load_model(model: str, device: str, compute_type: str, download_root: str):
    """Load (and cache) a faster-whisper model. Blocking — call via to_thread."""
    global _model, _model_key
    key = (model, device, compute_type, download_root)
    if _model is not None and _model_key == key:
        return _model

    from faster_whisper import WhisperModel  # imported lazily — heavy dep

    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s",
        model, device, compute_type,
    )
    _model = WhisperModel(
        model,
        device=device,
        compute_type=compute_type,
        download_root=download_root or None,
    )
    _model_key = key
    return _model


class FasterWhisperSTTProvider(STTProvider):
    """STT provider backed by a local faster-whisper model."""

    def warmup(self) -> None:
        """Load the model now so the first user request isn't a cold start.

        Blocking (downloads + initializes the model, can take ~70s for
        large-v3) — call via ``asyncio.to_thread`` from startup.
        """
        _load_model(
            self._model, self._device, self._compute_type, self._download_root
        )

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",
        compute_type: str = "default",
        download_root: str = "",
    ):
        self._model = model
        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root

    def _transcribe_sync(self, audio: bytes, language: str | None) -> STTResult:
        model = _load_model(
            self._model, self._device, self._compute_type, self._download_root
        )
        segments, info = model.transcribe(
            io.BytesIO(audio),
            language=language,
            vad_filter=True,  # drop silence/noise like handy's VAD
        )
        # ``segments`` is a generator — materialize it to force transcription.
        text = "".join(seg.text for seg in segments).strip()
        return STTResult(
            text=text,
            language=getattr(info, "language", "") or "",
            duration=float(getattr(info, "duration", 0.0) or 0.0),
        )

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str = "",
        filename: str = "audio.webm",
        language: str | None = None,
    ) -> STTResult:
        return await asyncio.to_thread(self._transcribe_sync, audio, language)

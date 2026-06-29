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
import re

from app.services.stt.base import STTProvider, STTResult

logger = logging.getLogger(__name__)

# Loaded once on first transcription; keyed by (model, device, compute_type).
_model = None
_model_key: tuple | None = None

# Per-segment confidence gate. Whisper (esp. large-v3) hallucinates fabricated
# text on silence/noise/very short clips — segments that are really "no speech"
# still come back with plausible words. Drop a segment when the model is both
# fairly sure it's silence (high no_speech_prob) AND not confident in the words
# (low avg_logprob). Tuned conservatively so real, quiet speech survives.
_NO_SPEECH_PROB_MAX = 0.6
_AVG_LOGPROB_MIN = -1.0

# Known Whisper hallucination phrases (from YouTube subtitles in the training
# data). When a clip has no real speech, the decoder loves to emit these. If the
# WHOLE transcript reduces to one of these, treat it as empty rather than text.
# Compared after lower-casing + collapsing whitespace + stripping punctuation.
_HALLUCINATION_PHRASES = frozenset(
    {
        # Vietnamese
        "hãy subscribe cho kênh để ủng hộ mình nhé",
        "hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn",
        "ghiền mì gõ",
        "cảm ơn các bạn đã theo dõi",
        "cảm ơn các bạn đã lắng nghe",
        "hẹn gặp lại các bạn trong video tiếp theo",
        "cảm ơn các bạn đã xem video",
        "đừng quên like và subscribe",
        # English
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "subscribe to my channel",
        "thank you",
    }
)


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace — for blocklist match."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_hallucination(text: str) -> bool:
    """True if the full transcript is (only) a known Whisper hallucination."""
    norm = _normalize(text)
    if not norm:
        return False
    return norm in _HALLUCINATION_PHRASES


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
            # Anti-hallucination decode settings:
            temperature=0.0,  # greedy — no sampling-driven invention
            condition_on_previous_text=False,  # stop runaway repetition loops
            no_speech_threshold=_NO_SPEECH_PROB_MAX,
            log_prob_threshold=_AVG_LOGPROB_MIN,
            compression_ratio_threshold=2.4,  # high ratio => repetitive garbage
        )
        # ``segments`` is a generator — materialize it to force transcription.
        # Drop segments the model flags as confident-silence-but-low-confidence-words.
        kept = []
        for seg in segments:
            no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            avg_lp = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
            if no_speech > _NO_SPEECH_PROB_MAX and avg_lp < _AVG_LOGPROB_MIN:
                logger.debug(
                    "STT: dropped likely-hallucinated segment (no_speech=%.2f "
                    "avg_logprob=%.2f): %r",
                    no_speech, avg_lp, seg.text,
                )
                continue
            kept.append(seg.text)

        text = "".join(kept).strip()
        # Final guard: whole transcript is a known YouTube-subtitle hallucination.
        if _is_hallucination(text):
            logger.info("STT: discarded known hallucination phrase: %r", text)
            text = ""
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

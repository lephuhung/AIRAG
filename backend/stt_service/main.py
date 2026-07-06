"""
hrag-stt — faster-whisper STT microservice (CPU)
================================================
Hosts the Whisper model (faster-whisper / CTranslate2, CPU int8) and serves it
over an OpenAI-compatible `POST /v1/audio/transcriptions` endpoint. The backend
then sets STT_PROVIDER=openai + STT_OPENAI_BASE_URL=http://stt:8091/v1 and drops
its in-process Whisper copy → backend workers stay pure-CPU and scale freely.

Runs the SAME FasterWhisperSTTProvider the backend used to run in-process — this
process must boot with STT_PROVIDER=faster_whisper (it IS the model host), NOT
openai, otherwise it would call itself.

STT stays on CPU deliberately: CTranslate2 4.8 needs CUDA 12 (libcublas.so.12)
but the image ships CUDA 13, so GPU inference can't load (see docs/vllm.md /
docker-compose.services.yml backend STT comment). So this is a CPU sidecar with
NO GPU reservation — it never competes for the 48GB GPU.

The OpenAI client the backend uses (OpenAIWhisperSTTProvider) posts multipart
{file, model, language?} and reads `resp.text`, so this endpoint mirrors that.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile

from app.core.config import settings
from app.services.stt import get_stt_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt_service")

if settings.STT_PROVIDER.lower() in ("openai", "openai_compatible"):
    raise RuntimeError(
        "hrag-stt must run with STT_PROVIDER=faster_whisper "
        f"(got {settings.STT_PROVIDER!r}) — it IS the model host, not a client."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    provider = get_stt_provider()
    logger.info("[stt] warming Whisper %s (device=%s, compute=%s) …",
                settings.STT_FW_MODEL, settings.STT_FW_DEVICE, settings.STT_FW_COMPUTE_TYPE)
    try:
        # warmup() is blocking (CTranslate2 model load) — run off the event loop.
        if hasattr(provider, "warmup"):
            await asyncio.to_thread(provider.warmup)
        logger.info("[stt] ready (%s)", settings.STT_FW_MODEL)
    except Exception as e:  # noqa: BLE001
        logger.warning("[stt] warmup failed (non-fatal, will load on first call): %s", e)
    yield


app = FastAPI(title="hrag-stt", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": settings.STT_PROVIDER,
        "model": settings.STT_FW_MODEL,
        "device": settings.STT_FW_DEVICE,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):
    audio = await file.read()
    provider = get_stt_provider()
    result = await provider.transcribe(
        audio,
        content_type=(file.content_type or ""),
        filename=file.filename or "audio.webm",
        language=language or None,
    )
    if response_format == "text":
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(result.text)
    # OpenAI transcription JSON shape is {"text": ...}; extra fields are ignored
    # by the client (it only reads .text) but useful for debugging.
    return {"text": result.text, "language": result.language, "duration": result.duration}

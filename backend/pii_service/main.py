"""
PII Extraction Microservice
===========================

Serves the Meddies/meddies-pii-v2 model (350M encoder, multilingual
token-classification) over HTTP so the backend people-agent can offload PII
detection (SĐT / CCCD / BHXH / tên người / …) instead of regex heuristics.

IMPORTANT — this model is NOT a causal LM:
  * It is a fine-tuned ENCODER (LiquidAI LFM2.5 Encoder 350M + LoRA r128 +
    37-tag BIOES classification head).
  * It must be loaded through the custom ``MeddiesPiiExtractor`` class shipped
    inside the model repo (modeling_meddies_pii.py), then called as
    ``model.extract(text) -> [Span(label, start, end, text)]``.
  * Loading it with AutoModelForCausalLM / pipeline("text-generation") fails
    with "Unrecognized model … Should have a `model_type` key" — which is why
    the first version of this service crash-looped 273 times.

Endpoints:
    GET  /health  -> {status, model, device, weights}
    POST /extract -> {text} -> {entities: dict[str, list[str]], model}

Environment:
    PII_MODEL_NAME  (default Meddies/meddies-pii-v2)
    PII_DEVICE      cpu | cuda (default cpu — 350M encoder fits CPU fine)
    PII_WEIGHTS     adapter | merged (default adapter = benchmarked path)
    PII_DTYPE       bfloat16 | float32 (auto: cuda->bfloat16, cpu->float32)
    PII_CONCURRENCY max parallel /extract calls (default 1)
    PII_CPU_THREADS torch intra-op threads for CPU inference (default 0 = auto)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pii_service")

# ── Configuration ───────────────────────────────────────────────────────────

_PII_MODEL_NAME = os.getenv("PII_MODEL_NAME", "Meddies/meddies-pii-v2")
_PII_DEVICE = os.getenv("PII_DEVICE", "cpu")
# "adapter" = benchmarked release path: downloads base weights, applies the LoRA
# adapter live (needs PEFT). "merged" = self-contained model.safetensors (no
# base download), but the loader module still imports PEFT at the top level, so
# peft must be installed either way.
_PII_WEIGHTS = os.getenv("PII_WEIGHTS", "adapter")
_PII_DTYPE = os.getenv("PII_DTYPE", "")
_PII_CONCURRENCY = max(1, int(os.getenv("PII_CONCURRENCY", "1")))
_PII_CPU_THREADS = int(os.getenv("PII_CPU_THREADS", "0"))

try:
    import torch  # noqa: E402
except Exception:  # pragma: no cover
    torch = None

if _PII_CPU_THREADS > 0 and torch is not None:
    torch.set_num_threads(_PII_CPU_THREADS)

# Dtype defaults: bf16 on GPU (evaluated path), fp32 on CPU (safer/faster).
if not _PII_DTYPE:
    _PII_DTYPE = "bfloat16" if _PII_DEVICE.startswith("cuda") else "float32"


# ── Model globals (loaded once in lifespan) ─────────────────────────────────

_extractor = None  # MeddiesPiiExtractor
_extract_sem = asyncio.Semaphore(_PII_CONCURRENCY)


def _load_extractor() -> None:
    """snapshot_download the model repo, import its custom loader, build it."""
    global _extractor

    from huggingface_hub import snapshot_download

    logger.info("[pii-service] downloading %s …", _PII_MODEL_NAME)
    path = snapshot_download(_PII_MODEL_NAME)
    if path not in sys.path:
        sys.path.insert(0, path)

    # Custom loader from the model repo (not part of transformers core).
    from modeling_meddies_pii import MeddiesPiiExtractor  # noqa: E402

    logger.info(
        "[pii-service] building MeddiesPiiExtractor (weights=%s device=%s dtype=%s) …",
        _PII_WEIGHTS, _PII_DEVICE, _PII_DTYPE,
    )
    _extractor = MeddiesPiiExtractor.from_pretrained(
        path,
        device=_PII_DEVICE,
        dtype=_PII_DTYPE,
        weights=_PII_WEIGHTS,
    )
    logger.info("[pii-service] model loaded from %s", path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_extractor()
        # Warm-up so the first /extract call is not slow.
        _ = _extract_sync("Bệnh nhân Nguyễn Văn A, SĐT 0912 345 678, CCCD 001203005123")
        logger.info(
            "[pii-service] ready (model=%s, device=%s, weights=%s)",
            _PII_MODEL_NAME, _PII_DEVICE, _PII_WEIGHTS,
        )
    except Exception as e:
        logger.error("[pii-service] failed to load model: %s", e, exc_info=True)
        raise RuntimeError(f"Cannot load PII model {_PII_MODEL_NAME}: {e}") from e

    yield

    logger.info("[pii-service] shutting down …")


app = FastAPI(title="pii-extraction-service", lifespan=lifespan)


# ── Request / Response schemas ──────────────────────────────────────────────

class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    entities: dict[str, list[str]]
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    weights: str


# ── Extraction logic (sync — runs in a thread pool) ─────────────────────────

def _extract_sync(text: str) -> dict[str, list[str]]:
    """Run one extraction synchronously (call via asyncio.to_thread).

    Meddies spans carry (label, start, end, text). We collapse them into
    {label: [extracted values]} preserving first-occurrence order, dropping
    empty duplicates.
    """
    if _extractor is None:
        raise RuntimeError("PII model not loaded")

    spans = _extractor.extract(text)

    entities: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for span in sorted(spans, key=lambda s: s.start):
        label = span.label
        value = span.text.strip()
        if not value or (label, value) in seen:
            continue
        seen.add((label, value))
        entities.setdefault(label, []).append(value)

    logger.debug("[pii-service] extract(%r) -> %s", text, entities)
    return entities


# ── HTTP endpoints ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok" if _extractor is not None else "not_ready",
        "model": _PII_MODEL_NAME,
        "device": _PII_DEVICE,
        "weights": _PII_WEIGHTS,
    }


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    async with _extract_sem:
        entities = await asyncio.to_thread(_extract_sync, req.text)
    return {"entities": entities, "model": _PII_MODEL_NAME}
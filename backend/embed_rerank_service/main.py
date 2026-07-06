"""
hrag-embed-rerank — GPU embed + rerank microservice
===================================================
Hosts the embedder (SentenceTransformer) + reranker (CrossEncoder) on the GPU
and serves them over HTTP so the backend/workers can drop their in-process GPU
copies and scale on CPU (WEB_CONCURRENCY>1). See docs/scaling.md.

This process runs the existing EmbeddingService / RerankerService in *in-process*
mode — it must boot with HRAG_EMBED_RERANK_URL unset/empty, otherwise it would try
to call itself.

Endpoints:
    GET  /health   → {status, model, dimension, reranker_model}
    POST /embed    → {texts:[...]}  -> {model, dimension, embeddings:[[...]]}
    POST /rerank   → {query, documents:[...], top_k?, min_score?}
                     -> {model, results:[{index, score, text}]}

GPU safety: a single shared model now fields every worker's requests, and the
rerank fan-out is the known VRAM-spike path (memory `search-cuda-oom-silent-fail`).
So GPU work is funnelled through an asyncio.Semaphore (EMBED_RERANK_CONCURRENCY,
default 1) as a hard, in-process ceiling — defense-in-depth on top of the
backend's cluster-wide Redis GPU cap.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.core.config import settings
from app.services.embedding.embedder import get_embedding_service
from app.services.retrieval.reranker import get_reranker_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embed_rerank_service")

# Hard ceiling on concurrent GPU batches (embed + rerank share the GPU).
_GPU_CONCURRENCY = max(1, int(os.getenv("EMBED_RERANK_CONCURRENCY", "1")))
_gpu_sema = asyncio.Semaphore(_GPU_CONCURRENCY)

if settings.HRAG_EMBED_RERANK_URL:
    # Guard against a misconfig where the service points at itself.
    raise RuntimeError(
        "hrag-embed-rerank must run with HRAG_EMBED_RERANK_URL empty "
        f"(got {settings.HRAG_EMBED_RERANK_URL!r}) — it IS the model host."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    emb = get_embedding_service()
    rr = get_reranker_service()
    logger.info("[embed-rerank] loading + warming models …")
    _ = emb.model
    emb.warmup()
    _ = rr.model
    rr.warmup()
    logger.info(
        "[embed-rerank] ready (embedder=%s dim=%s reranker=%s, gpu_concurrency=%d)",
        emb.model_name, emb.dimension, rr.model_name, _GPU_CONCURRENCY,
    )
    yield


app = FastAPI(title="hrag-embed-rerank", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str]


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int | None = None
    min_score: float | None = None


@app.get("/health")
async def health():
    emb = get_embedding_service()
    rr = get_reranker_service()
    return {
        "status": "ok",
        "model": emb.model_name,
        "dimension": emb.dimension,
        "reranker_model": rr.model_name,
    }


@app.post("/embed")
async def embed(req: EmbedRequest):
    emb = get_embedding_service()
    async with _gpu_sema:
        vectors = await asyncio.to_thread(emb.embed_texts, req.texts)
    return {"model": emb.model_name, "dimension": emb.dimension, "embeddings": vectors}


@app.post("/rerank")
async def rerank(req: RerankRequest):
    rr = get_reranker_service()
    async with _gpu_sema:
        results = await asyncio.to_thread(
            rr.rerank, req.query, req.documents, req.top_k, req.min_score
        )
    return {
        "model": rr.model_name,
        "results": [{"index": r.index, "score": r.score, "text": r.text} for r in results],
    }

"""
Model Pre-loader
=================
Eagerly initializes heavy ML model singletons at startup so that
the first user request does not pay the cold-start penalty.

Usage (API server — inside lifespan):
    from app.services.models.loader import preload_models
    preload_models()

Usage (Worker — before consuming):
    from app.services.models.loader import preload_worker_models
    preload_worker_models("parse")       # loads Docling models
    preload_worker_models("embed")       # loads embedding model
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_restart_only_models() -> tuple[Optional[str], Optional[str]]:
    """WebUI overrides for embedding/rerank roles (plan §12.4).

    RESTART-ONLY semantics: read ONCE here, before any model is constructed.
    Returns ``(embedding_model, reranker_model)`` — None entries mean "use the
    .env default". A big warning is logged because changing the embedding
    model invalidates existing vector dimensions (full reindex required).
    """
    embedding_model: Optional[str] = None
    reranker_model: Optional[str] = None
    try:
        from app.services.runtime_config import get_effective_sync

        emb_cfg = get_effective_sync("embedding")
        rr_cfg = get_effective_sync("rerank")
        if emb_cfg.source == "db" and emb_cfg.model:
            embedding_model = emb_cfg.model
            logger.warning(
                "[preload] EMBEDDING model overridden via WebUI → %r "
                "(restart semantics applied now; changing dimensions vs the "
                "indexed corpus REQUIRES a full reindex)",
                embedding_model,
            )
        if rr_cfg.source == "db" and rr_cfg.model:
            reranker_model = rr_cfg.model
            logger.warning(
                "[preload] RERANKER model overridden via WebUI → %r "
                "(restart semantics applied now)",
                reranker_model,
            )
        if emb_cfg.source == "db" and emb_cfg.base_url:
            logger.info(
                "[preload] embedding connection base_url=%s noted (local "
                "SentenceTransformer loading ignores it; remote embed-rerank "
                "mode still uses HRAG_EMBED_RERANK_URL)",
                emb_cfg.base_url,
            )
    except Exception as exc:
        logger.warning(
            f"[preload] runtime-config model overrides unavailable ({exc}) — using .env"
        )
    return embedding_model, reranker_model


def preload_models() -> None:
    """Eagerly load Embedding + Reranker models used by the API server."""
    t0 = time.time()
    from app.core.config import settings

    # WebUI embedding/rerank overrides — restart-only, resolved before any
    # model construction (plan §12.4).
    emb_override, rr_override = _resolve_restart_only_models()

    # In remote embed/rerank mode the models live in the hrag-embed-rerank
    # service — this process must NOT load them (it holds no GPU state).
    if settings.HRAG_EMBED_RERANK_URL:
        logger.info("[preload] remote embed/rerank mode — skipping local retrieval models")
    else:
        logger.info("[preload] Loading retrieval models …")

        # 1. Embedding model (sentence-transformers)
        from app.services.embedding.embedder import get_embedding_service

        emb = get_embedding_service(model_name=emb_override)
        _ = emb.model  # triggers lazy load
        logger.info(f"[preload] Embedding model ready ({emb.model_name})")
        # Warmup: encode dummy texts to initialize CUDA kernels
        emb.warmup()

        # 2. Reranker model (cross-encoder)
        from app.services.retrieval.reranker import get_reranker_service

        rr = get_reranker_service(model_name=rr_override)
        _ = rr.model  # triggers lazy load
        logger.info(f"[preload] Reranker model ready ({rr.model_name})")
        # Warmup: score dummy pairs to initialize CUDA kernels
        rr.warmup()

        elapsed = time.time() - t0
        logger.info(f"[preload] Retrieval models loaded + warmed up in {elapsed:.1f}s")

    # 3. Memory Agent (Qwen via vLLM API — used by chat_agent for memory extraction)

    if settings.MEMORY_AGENT_LOCAL:
        try:
            t1 = time.time()
            logger.info("[preload] Loading Memory Agent model …")
            from app.services.llm import get_memory_agent

            agent = get_memory_agent()
            # Trigger the lazy vLLM engine load
            from app.services.llm.vllm_local import LocalVLLMProvider

            if isinstance(agent, LocalVLLMProvider):
                agent._get_engine()
            logger.info(
                f"[preload] Memory Agent ready ({settings.MEMORY_AGENT_MODEL}) in {time.time() - t1:.1f}s"
            )
        except Exception as e:
            logger.warning(f"[preload] Memory Agent pre-load failed (non-fatal): {e}")


def preload_worker_models(worker_type: str) -> None:
    """Eagerly load models specific to a worker type.

    Args:
        worker_type: One of "parse", "embed", "caption", "kg".
    """
    t0 = time.time()
    logger.info(f"[preload] Loading models for worker={worker_type} …")

    if worker_type == "parse":
        # Docling pipeline and (optionally) the local OCR model
        _preload_docling()
        _preload_ocr()

    elif worker_type == "embed":
        from app.core.config import settings

        if settings.HRAG_EMBED_RERANK_URL:
            logger.info("[preload] remote embed/rerank mode — skipping local embedder")
        else:
            # Embedding model (same as retrieval) — WebUI override applies at
            # worker start too (restart-only semantics, plan §12.4).
            from app.services.embedding.embedder import get_embedding_service

            w_emb, _w_rr = _resolve_restart_only_models()
            emb = get_embedding_service(model_name=w_emb)
            _ = emb.model
            logger.info(f"[preload] Embedding model ready ({emb.model_name})")

    elif worker_type == "caption":
        # Caption worker uses LLM providers — no heavy local model to preload
        pass

    elif worker_type == "kg":
        # KG worker uses LLM provider + LightRAG — initialization is per-workspace
        pass

    elapsed = time.time() - t0
    logger.info(f"[preload] Worker={worker_type} models loaded in {elapsed:.1f}s")


def _preload_docling() -> None:
    """Pre-initialize the Docling document converter so first parse is fast."""
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            EasyOcrOptions,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption

        from app.core.config import settings

        ocr_options = EasyOcrOptions(lang=["vi", "en"], force_full_page_ocr=True)
        pipeline_options = PdfPipelineOptions(
            do_ocr=settings.HRAG_ENABLE_OCR,
            ocr_options=ocr_options,
            force_backend_text=True,  # Use PyPdfiumDocumentBackend for better text extraction
        )

        # This triggers the download + load of Docling's internal models
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            }
        )
        logger.info("[preload] Docling converter initialized")
    except Exception as e:
        logger.warning(f"[preload] Docling pre-load failed (non-fatal): {e}")


def _preload_ocr() -> None:
    """Pre-initialize the local HunyuanOCR vLLM engine so first parse is fast."""
    from app.core.config import settings

    if not settings.HRAG_OCR_LOCAL:
        logger.info("[preload] OCR is remote (HRAG_OCR_LOCAL=false) — skipping")
        return

    try:
        t0 = time.time()
        logger.info("[preload] Loading local OCR model (HunyuanOCR) …")
        from app.services.parsing.ocr_service import HunyuanOCRService

        svc = HunyuanOCRService()
        # Trigger the lazy vLLM engine load
        svc._get_local_llm()
        elapsed = time.time() - t0
        logger.info(
            f"[preload] OCR model ready ({settings.HUNYUAN_OCR_MODEL}) in {elapsed:.1f}s"
        )
    except Exception as e:
        logger.warning(f"[preload] OCR pre-load failed (non-fatal): {e}")

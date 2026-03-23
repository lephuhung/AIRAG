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

logger = logging.getLogger(__name__)


def preload_models() -> None:
    """Eagerly load Embedding + Reranker models used by the API server."""
    t0 = time.time()
    logger.info("[preload] Loading retrieval models …")

    # 1. Embedding model (sentence-transformers)
    from app.services.embedder import get_embedding_service
    emb = get_embedding_service()
    _ = emb.model  # triggers lazy load
    logger.info(f"[preload] Embedding model ready ({emb.model_name})")

    # 2. Reranker model (cross-encoder)
    from app.services.reranker import get_reranker_service
    rr = get_reranker_service()
    _ = rr.model  # triggers lazy load
    logger.info(f"[preload] Reranker model ready ({rr.model_name})")

    elapsed = time.time() - t0
    logger.info(f"[preload] Retrieval models loaded in {elapsed:.1f}s")

    # 3. Memory Agent (Qwen via vLLM API — used by chat_agent for memory extraction)
    from app.core.config import settings
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
            logger.info(f"[preload] Memory Agent ready ({settings.MEMORY_AGENT_MODEL}) in {time.time() - t1:.1f}s")
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
        # Embedding model (same as retrieval)
        from app.services.embedder import get_embedding_service
        emb = get_embedding_service()
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

        ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        pipeline_options = PdfPipelineOptions(
            do_ocr=settings.HRAG_ENABLE_OCR,
            ocr_options=ocr_options,
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
        from app.services.ocr_service import HunyuanOCRService
        svc = HunyuanOCRService()
        # Trigger the lazy vLLM engine load
        svc._get_local_llm()
        elapsed = time.time() - t0
        logger.info(f"[preload] OCR model ready ({settings.HUNYUAN_OCR_MODEL}) in {elapsed:.1f}s")
    except Exception as e:
        logger.warning(f"[preload] OCR pre-load failed (non-fatal): {e}")


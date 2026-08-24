"""
Reranker Service
================
Cross-encoder reranker for improving retrieval precision.

Default model: BAAI/bge-reranker-v2-m3 (multilingual, 100+ languages).
Configurable via HRAG_RERANKER_MODEL in settings.

Usage:
    reranker = get_reranker_service()
    ranked = reranker.rerank("user question", ["chunk1", "chunk2", ...], top_k=5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)


# Shared blocking HTTP client for remote rerank mode (lazy singleton).
_http_client = None


def _get_http_client():
    global _http_client
    if _http_client is None:
        import httpx

        _http_client = httpx.Client(timeout=settings.HRAG_EMBED_RERANK_TIMEOUT)
    return _http_client


@dataclass
class RerankResult:
    """A single reranked item with its original index and relevance score."""
    index: int          # Original position in the input list
    score: float        # Cross-encoder relevance score (higher = more relevant)
    text: str           # The chunk text


class RerankerService:
    """
    Cross-encoder reranker service.
    Scores (query, document) pairs jointly through a transformer,
    producing far more accurate relevance scores than bi-encoder cosine similarity.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.HRAG_RERANKER_MODEL
        self._model = None
        # Remote mode: call the hrag-embed-rerank service instead of loading the
        # cross-encoder locally (so backend workers hold no GPU state).
        self._remote = (settings.HRAG_EMBED_RERANK_URL or "").rstrip("/") or None

    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._remote:
            raise RuntimeError(
                "RerankerService.model accessed in remote mode "
                "(HRAG_EMBED_RERANK_URL set) — no local model is loaded."
            )
        if self._model is None:
            from sentence_transformers import CrossEncoder
            device = settings.HRAG_RERANKER_DEVICE
            st_device = None if device == "auto" else device
            logger.info(f"Loading reranker model: {self.model_name} (device={device})")
            self._model = CrossEncoder(self.model_name, device=st_device)
            logger.info(f"Reranker model loaded: {self.model_name}")
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[RerankResult]:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The user's search query
            documents: List of document texts to rerank
            top_k: Maximum number of results to return (None = all)
            min_score: Minimum relevance score threshold (None = no filtering)

        Returns:
            List of RerankResult sorted by score (descending),
            filtered by top_k and min_score.
        """
        if not documents:
            return []

        if self._remote:
            # Service applies the same sort + min_score + top_k filtering and
            # returns results already ordered, so just map them back 1:1.
            resp = _get_http_client().post(
                f"{self._remote}/rerank",
                json={
                    "query": query,
                    "documents": list(documents),
                    "top_k": top_k,
                    "min_score": min_score,
                },
            )
            resp.raise_for_status()
            return [
                RerankResult(index=r["index"], score=r["score"], text=r["text"])
                for r in resp.json()["results"]
            ]

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, doc) for doc in documents]

        # Score all pairs in a single batch
        scores = self.model.predict(pairs, batch_size=settings.HRAG_RERANKER_BATCH_SIZE).tolist()

        # Build results with original indices
        results = [
            RerankResult(index=i, score=s, text=doc)
            for i, (s, doc) in enumerate(zip(scores, documents))
        ]

        # Sort by score descending (most relevant first)
        results.sort(key=lambda r: r.score, reverse=True)

        # Apply min_score filter
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]

        return results

    def warmup(self) -> None:
        """
        Pre-warm the cross-encoder by scoring dummy (query, doc) pairs to
        initialize CUDA kernels and verify the model produces valid scores.
        """
        dummy_pairs = [
            ("Vietnamese law query", "legal document content here"),
            ("administrative procedure", "procedure steps for government"),
            ("policy regulation", "regulation text with articles"),
        ]
        logger.info(f"[reranker] Warmup: scoring {len(dummy_pairs)} pairs")
        self.model.predict(dummy_pairs)
        logger.info(f"[reranker] Warmup complete for {self.model_name}")


# Singleton instance
_default_service: Optional[RerankerService] = None


def get_reranker_service(model_name: Optional[str] = None) -> RerankerService:
    """Get or create the default reranker service.

    ``model_name`` (optional) lets the pre-loader inject a WebUI override at
    process start (restart-only semantics, plan §12.4).
    """
    global _default_service
    if _default_service is None:
        _default_service = RerankerService(model_name=model_name)
    return _default_service

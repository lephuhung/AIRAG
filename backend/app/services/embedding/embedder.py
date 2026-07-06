"""
Embedding Service
=================
Generates vector embeddings using sentence-transformers.

Model is set by HRAG_EMBEDDING_MODEL (deployed: mainguyen9/vietlegal-harrier-0.6b,
1024-dim; code default falls back to BAAI/bge-m3 if unset).
"""
from __future__ import annotations

import logging
from typing import Sequence, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


# Shared HTTP client for remote embed/rerank mode (lazy singleton). A blocking
# client is fine: every call site is sync and already wrapped in asyncio.to_thread.
_http_client = None


def _get_http_client():
    global _http_client
    if _http_client is None:
        import httpx

        _http_client = httpx.Client(timeout=settings.HRAG_EMBED_RERANK_TIMEOUT)
    return _http_client


class EmbeddingService:
    """
    Service for generating text embeddings.
    Uses sentence-transformers for local embedding generation.
    """

    # Dimension lookup for common models (used before model is loaded)
    _KNOWN_DIMS = {
        "BAAI/bge-m3": 1024,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "intfloat/multilingual-e5-large-instruct": 1024,
    }

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.HRAG_EMBEDDING_MODEL
        self._model = None
        # Remote mode: call the hrag-embed-rerank service over HTTP instead of
        # loading the model locally. Toggle inside the class (not just the
        # singleton) because some callers construct EmbeddingService() directly.
        self._remote = (settings.HRAG_EMBED_RERANK_URL or "").rstrip("/") or None
        self._remote_dim: Optional[int] = None

    def _post_embed(self, texts: list[str]) -> list[list[float]]:
        """POST texts to the remote /embed endpoint (remote mode only)."""
        resp = _get_http_client().post(
            f"{self._remote}/embed", json={"texts": texts}
        )
        resp.raise_for_status()
        data = resp.json()
        if self._remote_dim is None:
            self._remote_dim = data.get("dimension")
        return data["embeddings"]

    @property
    def model(self):
        """Lazy load the model onto the configured device."""
        if self._remote:
            raise RuntimeError(
                "EmbeddingService.model accessed in remote mode "
                "(HRAG_EMBED_RERANK_URL set) — no local model is loaded. "
                "This process must not hold GPU embedder state."
            )
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = settings.HRAG_EMBEDDING_DEVICE  # "auto" | "cpu" | "cuda"
            # SentenceTransformer accepts "cpu", "cuda", "cuda:0", etc.
            # Pass None for "auto" so sentence-transformers picks the best device.
            st_device = None if device == "auto" else device
            logger.info(f"Loading embedding model: {self.model_name} (device={device})")
            self._model = SentenceTransformer(self.model_name, device=st_device)
            logger.info(
                f"Embedding model loaded: {self.model_name} "
                f"(dim={self._model.get_embedding_dimension()})"
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension size."""
        if self._remote:
            if self._remote_dim is None:
                # Lazily fetch from the service health endpoint (before any embed
                # call), fall back to the static lookup if unreachable.
                try:
                    resp = _get_http_client().get(f"{self._remote}/health")
                    resp.raise_for_status()
                    self._remote_dim = resp.json().get("dimension")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[embedder] remote /health dim fetch failed: {e}")
            return self._remote_dim or self._KNOWN_DIMS.get(self.model_name, 1024)
        if self._model is not None:
            return self._model.get_embedding_dimension()
        return self._KNOWN_DIMS.get(self.model_name, 1024)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        if self._remote:
            return self._post_embed([text])[0]
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not texts:
            return []
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")
        if self._remote:
            # Same empty-text filtering as local so behavior (and the ChromaDB
            # length-mismatch warning below) is byte-identical across the boundary.
            result = self._post_embed(valid_texts)
        else:
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=settings.HRAG_EMBEDDING_BATCH_SIZE,
            )
            result = embeddings.tolist()
        if len(result) != len(texts):
            logger.warning(
                f"[embedder] Filtered {len(texts) - len(result)} empty/whitespace texts "
                f"({len(texts)} input → {len(result)} embeddings). "
                f"This WILL cause ChromaDB length mismatch in embed_worker."
            )
        return result

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        return self.embed_text(query)

    def warmup(self, texts: list[str] | None = None) -> None:
        """
        Pre-warm the model by encoding dummy/real texts to initialize CUDA kernels
        and verify the model produces valid embeddings.

        Args:
            texts: Optional list of texts to encode. If None, uses 5 generic
                   dummy strings. Providing real texts (e.g., common queries)
                   gives better warmup quality.
        """
        if self._remote:
            logger.info("[embedder] remote mode — warmup handled by the service")
            return
        warmup_texts = texts or [
            "Vietnamese administrative document query",
            "Legal regulation search",
            "Policy keyword lookup",
            "Document classification request",
            "Entity extraction query",
        ]
        logger.info(f"[embedder] Warmup: encoding {len(warmup_texts)} texts")
        self.embed_texts(warmup_texts)
        logger.info(f"[embedder] Warmup complete for {self.model_name}")


# Default service instance (singleton)
_default_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the default embedding service."""
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service


def embed_text(text: str) -> list[float]:
    """Convenience function to embed a single text."""
    return get_embedding_service().embed_text(text)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Convenience function to embed multiple texts."""
    return get_embedding_service().embed_texts(texts)

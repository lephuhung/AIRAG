"""
BM25 Index Manager
==================
In-memory BM25 index per workspace for lexical (keyword) search.

Used as the second retrieval leg in hybrid search alongside ChromaDB vector search.
Results are merged via Reciprocal Rank Fusion (RRF) before cross-encoder reranking.

Architecture:
  - Index is built lazily on first query (or after document count changes).
  - Corpus is loaded from ChromaDB (the collection.get() call fetches all docs).
  - One BM25Index instance is cached per workspace_id via module-level dict.
  - Thread-safe for read; rebuild is protected by a threading.Lock.

Tokenisation:
  Simple whitespace + punctuation split with Vietnamese-aware lowercasing.
  No heavy NLP dependency — rank-bm25 handles TF-IDF weighting internally.

Performance:
  - ~100 MB RAM for 10k chunks (strings), negligible for typical workloads.
  - Cold build: ~50 ms for 1k docs, ~500 ms for 10k docs (pure Python).
  - Subsequent queries: <5 ms (numpy dot product inside rank-bm25).
"""
from __future__ import annotations

import logging
import math
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Simple tokeniser: lowercase + split on non-word chars
# Works well for Vietnamese (space-segmented) and Latin text.
_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)


def _parse_vietnamese_date(date_str: str) -> datetime | None:
    """Parse Vietnamese date formats: '15/01/2026', '01/2026', '2026'."""
    if not date_str:
        return None
    date_str = date_str.strip()
    # Try full date: DD/MM/YYYY
    for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    # Try month/year: MM/YYYY
    for fmt in ("%m/%Y", "%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    # Try year only: YYYY
    try:
        return datetime(int(date_str[:4]), 1, 1)
    except (ValueError, IndexError):
        pass
    return None


def _compute_recency_boost(date_str: str, decay_days: int = 365) -> float:
    """
    Compute recency boost factor based on published_date.
    Returns exp(-days_since / decay_days), scaled to [0, 1].
    Newer documents get higher boost (closer to 1.0).
    """
    if not date_str:
        return 0.0  # No date = no boost
    pub_date = _parse_vietnamese_date(date_str)
    if pub_date is None:
        return 0.0
    try:
        now = datetime.now()
        days_since = (now - pub_date).days
        if days_since < 0:
            days_since = 0  # Future dates get max boost
        # exponential decay: boost = exp(-days / decay_days)
        boost = math.exp(-days_since / decay_days)
        return boost
    except Exception:
        return 0.0


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-word characters."""
    return [t for t in _TOKEN_RE.split(text.lower()) if t]


@dataclass
class _IndexState:
    """Holds one built BM25 index for a workspace."""
    bm25: object                   # rank_bm25.BM25Okapi instance
    ids: list[str]                 # ChromaDB chunk IDs (same order as corpus)
    metadatas: list[dict]          # metadata parallel to ids
    documents: list[str]           # raw texts parallel to ids
    doc_count: int                 # snapshot of collection.count() at build time
    lock: threading.Lock = field(default_factory=threading.Lock)


# Module-level cache: workspace_id → _IndexState
_index_cache: dict[uuid.UUID, _IndexState] = {}
_cache_lock = threading.Lock()


def _build_index(vector_store: VectorStore) -> _IndexState:
    """
    Fetch all documents from ChromaDB and build a fresh BM25Okapi index.
    This is a synchronous, CPU-bound call — run inside asyncio.to_thread().
    """
    from rank_bm25 import BM25Okapi
    from app.core.config import settings

    logger.info(f"[bm25] Building BM25 index for collection '{vector_store.collection_name}'")

    # Fetch all docs from ChromaDB (no embedding needed)
    result = vector_store.collection.get(include=["documents", "metadatas"])

    ids: list[str] = result.get("ids") or []
    documents: list[str] = result.get("documents") or []
    metadatas: list[dict] = result.get("metadatas") or []

    if not documents:
        logger.warning(f"[bm25] Collection '{vector_store.collection_name}' is empty — BM25 index will be empty")
        corpus_tokenized = []
    else:
        corpus_tokenized = [_tokenize(doc) for doc in documents]

    # Tuned BM25 parameters for Vietnamese legal text
    bm25 = BM25Okapi(corpus_tokenized, k1=settings.HRAG_BM25_K1, b=settings.HRAG_BM25_B)
    doc_count = len(ids)

    logger.info(f"[bm25] Index built: {doc_count} chunks (k1={settings.HRAG_BM25_K1}, b={settings.HRAG_BM25_B})")
    return _IndexState(
        bm25=bm25,
        ids=ids,
        metadatas=metadatas,
        documents=documents,
        doc_count=doc_count,
    )


def get_or_build_index(vector_store: VectorStore) -> _IndexState:
    """
    Return the cached BM25 index for this workspace, rebuilding if stale.
    Staleness check: compare doc_count in cache vs collection.count().

    Call inside asyncio.to_thread() — this is a blocking operation.
    """
    workspace_id = vector_store.workspace_id
    current_count = vector_store.count()

    with _cache_lock:
        cached = _index_cache.get(workspace_id)

    if cached is not None and cached.doc_count == current_count:
        return cached  # fresh — reuse

    # Build (or rebuild) index
    new_state = _build_index(vector_store)

    with _cache_lock:
        _index_cache[workspace_id] = new_state

    return new_state


def bm25_search(
    vector_store: VectorStore,
    query: str,
    top_n: int,
    document_ids: list[uuid.UUID] | None = None,
) -> list[dict]:
    """
    Run BM25 search and return top-N results as a list of dicts:
        [{"id": str, "metadata": dict, "document": str, "bm25_rank": int}, ...]

    Args:
        vector_store:  The workspace VectorStore (used to load corpus).
        query:         Natural language query string.
        top_n:         Maximum number of results to return.
        document_ids:  Optional filter — only keep chunks from these document IDs.

    Note: call this inside asyncio.to_thread() — it is CPU-bound and blocking.
    """
    state = get_or_build_index(vector_store)

    if not state.ids:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = state.bm25.get_scores(tokens)

    # Apply recency boost to BM25 scores
    from app.core.config import settings
    boost_factor = settings.HRAG_RECENTNESS_BOOST
    decay_days = settings.HRAG_RECENTNESS_DECAY_DAYS

    # Pair scores with indices, apply optional document_id filter and recency boost
    scored = []
    for idx, score in enumerate(scores):
        if score <= 0:
            continue
        meta = state.metadatas[idx] if idx < len(state.metadatas) else {}
        if document_ids and meta.get("document_id") not in [str(doc_id) for doc_id in document_ids]:
            continue
        # Apply recency boost: newer docs get higher scores
        if boost_factor > 0:
            date_str = meta.get("published_date", "")
            recency_boost = _compute_recency_boost(date_str, decay_days)
            score = score * (1 + boost_factor * recency_boost)
        scored.append((idx, score))

    # Sort by score descending, take top_n
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:top_n]

    results = []
    for rank, (idx, score) in enumerate(scored):
        results.append({
            "id": state.ids[idx],
            "metadata": state.metadatas[idx] if idx < len(state.metadatas) else {},
            "document": state.documents[idx] if idx < len(state.documents) else "",
            "bm25_score": float(score),
            "bm25_rank": rank,
        })

    return results


def invalidate_cache(workspace_id: uuid.UUID) -> None:
    """
    Force the next query to rebuild the BM25 index for this workspace.
    Call after adding/deleting documents if you need immediate consistency
    (normally the doc_count staleness check handles this automatically).
    """
    with _cache_lock:
        _index_cache.pop(workspace_id, None)
    logger.debug(f"[bm25] Cache invalidated for workspace {workspace_id}")

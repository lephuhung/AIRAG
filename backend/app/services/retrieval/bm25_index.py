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
import pickle
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.embedding.vector_store import VectorStore

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


# Vietnamese word segmenter (pyvi), lazily resolved once.
#   None      → not yet probed
#   False     → probed, unavailable (fall back to whitespace tokenisation)
#   callable  → pyvi ViTokenizer.tokenize
_vi_segmenter: object = None


def _get_vi_segmenter():
    """Lazily import pyvi's segmenter; cache the result (or False on failure)."""
    global _vi_segmenter
    if _vi_segmenter is None:
        try:
            from pyvi import ViTokenizer

            _vi_segmenter = ViTokenizer.tokenize
        except Exception:  # pragma: no cover - optional dependency
            logger.warning(
                "[bm25] HRAG_BM25_WORD_SEGMENT=true but pyvi is not available — "
                "falling back to whitespace tokenisation"
            )
            _vi_segmenter = False
    return _vi_segmenter


def _tokenize(text: str) -> list[str]:
    """
    Tokenise text for BM25.

    Default: lowercase + split on non-word characters (whitespace-aware, works
    for both Vietnamese space-segmented text and Latin script).

    When HRAG_BM25_WORD_SEGMENT is enabled, run pyvi word segmentation first so
    multi-syllable Vietnamese words are kept as single tokens (pyvi joins them
    with '_', e.g. "quyết_định"). MUST be applied identically at index-build and
    query time — both go through this one function, so they stay consistent.
    """
    from app.core.config import settings

    if settings.HRAG_BM25_WORD_SEGMENT:
        segmenter = _get_vi_segmenter()
        if segmenter:
            try:
                text = segmenter(text)
            except Exception:
                pass  # fall through to plain tokenisation on any pyvi error
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

# Bump when the on-disk format or tokenisation logic changes, to invalidate
# stale pickles written by an older build.
_PERSIST_VERSION = 1


def _persist_path(workspace_id: uuid.UUID) -> Path:
    from app.core.config import settings

    return Path(settings.BASE_DIR) / "data" / "bm25" / f"{workspace_id}.pkl"


def _save_index(workspace_id: uuid.UUID, state: _IndexState) -> None:
    """Pickle the index to disk (best-effort; never raises)."""
    from app.core.config import settings

    if not settings.HRAG_BM25_PERSIST:
        return
    path = _persist_path(workspace_id)
    payload = {
        "version": _PERSIST_VERSION,
        "word_segment": settings.HRAG_BM25_WORD_SEGMENT,
        "doc_count": state.doc_count,
        "bm25": state.bm25,
        "ids": state.ids,
        "metadatas": state.metadatas,
        "documents": state.documents,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)  # atomic on POSIX
        logger.info(f"[bm25] Persisted index for workspace {workspace_id} ({state.doc_count} chunks)")
    except Exception as e:  # pragma: no cover - disk issues are non-fatal
        logger.warning(f"[bm25] Failed to persist index for {workspace_id}: {e}")


def _load_index(workspace_id: uuid.UUID, expected_count: int) -> _IndexState | None:
    """
    Load a persisted index from disk if it matches the current corpus size,
    persist version, and tokenisation setting. Returns None on any mismatch/error.
    """
    from app.core.config import settings

    if not settings.HRAG_BM25_PERSIST:
        return None
    path = _persist_path(workspace_id)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as e:  # pragma: no cover
        logger.warning(f"[bm25] Failed to read persisted index for {workspace_id}: {e}")
        return None

    if (
        payload.get("version") != _PERSIST_VERSION
        or payload.get("word_segment") != settings.HRAG_BM25_WORD_SEGMENT
        or payload.get("doc_count") != expected_count
    ):
        # Stale (corpus changed, version bumped, or tokeniser toggled) → rebuild.
        return None

    logger.info(f"[bm25] Loaded persisted index for workspace {workspace_id} ({expected_count} chunks)")
    return _IndexState(
        bm25=payload["bm25"],
        ids=payload["ids"],
        metadatas=payload["metadatas"],
        documents=payload["documents"],
        doc_count=payload["doc_count"],
    )


def _build_index(vector_store: VectorStore) -> _IndexState:
    """
    Fetch all documents from ChromaDB and build a fresh BM25Okapi index.
    This is a synchronous, CPU-bound call — run inside asyncio.to_thread().
    """
    from rank_bm25 import BM25Okapi
    from app.core.config import settings

    logger.info(f"[bm25] Building BM25 index for collection '{vector_store.collection_name}'")

    # Fetch all docs from ChromaDB (no embedding needed). Go through _run so a
    # stale collection handle (collection recreated out-of-process) self-heals
    # instead of failing the whole BM25 build.
    result = vector_store._run(lambda col: col.get(include=["documents", "metadatas"]))

    ids: list[str] = result.get("ids") or []
    documents: list[str] = result.get("documents") or []
    metadatas: list[dict] = result.get("metadatas") or []

    if not documents:
        # Empty corpus: do NOT construct BM25Okapi — rank_bm25 divides by the
        # corpus size (avgdl = num_doc / corpus_size) and raises ZeroDivisionError
        # on an empty list. Return an empty state instead; the search path's
        # `if not state.ids` guard short-circuits before touching `bm25`.
        logger.warning(f"[bm25] Collection '{vector_store.collection_name}' is empty — BM25 index will be empty")
        return _IndexState(bm25=None, ids=[], metadatas=[], documents=[], doc_count=0)

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

    # Try a persisted index from disk before rebuilding from ChromaDB.
    loaded = _load_index(workspace_id, current_count)
    if loaded is not None:
        with _cache_lock:
            _index_cache[workspace_id] = loaded
        return loaded

    # Build (or rebuild) index, then persist for future cold starts.
    new_state = _build_index(vector_store)
    _save_index(workspace_id, new_state)

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

    # NOTE: recency boost is intentionally NOT applied here. RRF merging uses the
    # BM25 *rank* (not the raw score), and the post-merge DeepRetriever._apply_recency_boost
    # already applies recency once to all chunks. Boosting here too would double-count
    # recency for BM25-only hits. Keep raw BM25 scores so the rank reflects pure lexical
    # relevance.

    # Pair scores with indices, apply optional document_id filter
    scored = []
    for idx, score in enumerate(scores):
        if score <= 0:
            continue
        meta = state.metadatas[idx] if idx < len(state.metadatas) else {}
        if document_ids and meta.get("document_id") not in [str(doc_id) for doc_id in document_ids]:
            continue
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
    # Also drop the persisted copy so the next query rebuilds from scratch.
    try:
        _persist_path(workspace_id).unlink(missing_ok=True)
    except Exception:
        pass
    logger.debug(f"[bm25] Cache invalidated for workspace {workspace_id}")

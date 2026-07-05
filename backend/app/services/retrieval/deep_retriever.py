"""
Deep Retriever
===============

Hybrid retrieval combining Knowledge Graph (LightRAG) + Vector Search (ChromaDB)
+ BM25 Lexical Search + Cross-encoder Reranking (bge-reranker-v2-m3).

Pipeline:
  1. KG query  (parallel) → entity/relationship summary
  2. Vector search → over-fetch top-N candidates (HRAG_VECTOR_PREFETCH)
  3. BM25 search  (parallel with vector, optional) → top-N lexical candidates
  4. Merge vector + BM25 via Reciprocal Rank Fusion (RRF)
  5. Cross-encoder rerank → precision filter to top-K (HRAG_RERANKER_TOP_K)
  6. Merge with citations + optional image references

Retrieval caching:
  - Results are cached in-memory for 5 minutes (TTL) keyed by
    (workspace_id + query_hash + top_k + mode + document_ids).
  - Cache is invalidated automatically when BM25 index is rebuilt
    (BM25 index already tracks doc_count staleness).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document, DocumentImage, DocumentTable
from app.services.embedding.embedder import EmbeddingService
from app.services.embedding.vector_store import VectorStore
from app.services.kg.knowledge_graph_service import KnowledgeGraphService
from app.services.retrieval.reranker import RerankerService, get_reranker_service
from app.services.models.parsed_document import (
    Citation,
    DeepRetrievalResult,
    EnrichedChunk,
    ExtractedImage,
    ExtractedTable,
)

logger = logging.getLogger(__name__)

# ── Retrieval result cache ────────────────────────────────────────────────────
# In-process fallback, used when REDIS_ENABLED=false (unchanged behaviour). When
# Redis is on, the cache is SHARED across all backend worker processes/replicas,
# so a repeat query reuses ANY process's prior work (see _get/_set below).
_RETRIEVAL_CACHE: dict[str, tuple[DeepRetrievalResult, float]] = {}
_CACHE_LOCK = threading.Lock()
_RETRIEVAL_CACHE_TTL = 300.0  # 5 minutes

_REDIS_CACHE_PREFIX = "rag:cache:"        # rag:cache:{cache_key}      -> JSON result
_REDIS_WS_INDEX_PREFIX = "rag:cache:ws:"  # rag:cache:ws:{workspace}   -> SET of cache_keys


def _retrieval_cache_key(
    workspace_id: uuid.UUID,
    question: str,
    top_k: int,
    mode: str,
    document_ids: Optional[list[uuid.UUID]],
) -> str:
    """Build a cache key from retrieval parameters."""
    doc_ids_str = "|".join(sorted(str(d) for d in (document_ids or [])))
    raw = f"{workspace_id}:{question}:{top_k}:{mode}:{doc_ids_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:40]


# ── (de)serialization for the shared (Redis) cache ────────────────────────────
# DeepRetrievalResult is a plain nested-dataclass tree, so asdict()+json is a
# safe, version-tolerant wire format (no pickle). bbox tuples round-trip via list.
def _encode_result(result: DeepRetrievalResult) -> str:
    import dataclasses
    import json

    return json.dumps(dataclasses.asdict(result))


def _decode_result(raw: str) -> DeepRetrievalResult:
    import json

    d = json.loads(raw)
    return DeepRetrievalResult(
        chunks=[EnrichedChunk(**c) for c in d.get("chunks", [])],
        citations=[Citation(**c) for c in d.get("citations", [])],
        context=d.get("context", ""),
        query=d.get("query", ""),
        mode=d.get("mode", "hybrid"),
        knowledge_graph_summary=d.get("knowledge_graph_summary", ""),
        image_refs=[
            ExtractedImage(**{**i, "bbox": tuple(i["bbox"]) if i.get("bbox") else None})
            for i in d.get("image_refs", [])
        ],
        table_refs=[ExtractedTable(**t) for t in d.get("table_refs", [])],
    )


async def _get_cached_result(cache_key: str) -> Optional[DeepRetrievalResult]:
    """Return cached result if fresh, else None (shared via Redis when enabled)."""
    from app.core.redis_client import get_redis, is_redis_enabled

    if is_redis_enabled():
        try:
            raw = await get_redis().get(_REDIS_CACHE_PREFIX + cache_key)
        except Exception as e:  # noqa: BLE001 — cache miss on any Redis hiccup
            logger.warning("[retrieval_cache] redis get failed: %s", e)
            return None
        if not raw:
            return None
        try:
            result = _decode_result(raw)
        except Exception as e:  # noqa: BLE001 — corrupt/old entry: treat as miss
            logger.warning("[retrieval_cache] decode failed (ignoring entry): %s", e)
            return None
        logger.info(f"[retrieval_cache] HIT (redis) key={cache_key[:8]}…")
        return result

    with _CACHE_LOCK:
        cached = _RETRIEVAL_CACHE.get(cache_key)
    if cached is None:
        return None
    result, cached_at = cached
    if time.time() - cached_at > _RETRIEVAL_CACHE_TTL:
        with _CACHE_LOCK:
            _RETRIEVAL_CACHE.pop(cache_key, None)
        return None
    logger.info(f"[retrieval_cache] HIT key={cache_key[:8]}…")
    return result


async def _set_cached_result(
    cache_key: str, result: DeepRetrievalResult, workspace_id: uuid.UUID
) -> None:
    """Store result with a 5-minute TTL (shared via Redis when enabled)."""
    from app.core.redis_client import get_redis, is_redis_enabled

    if is_redis_enabled():
        try:
            payload = _encode_result(result)
        except Exception as e:  # noqa: BLE001 — never fail a query over caching
            logger.warning("[retrieval_cache] encode failed (skip caching): %s", e)
            return
        ttl = int(_RETRIEVAL_CACHE_TTL)
        idx = _REDIS_WS_INDEX_PREFIX + str(workspace_id)
        try:
            pipe = get_redis().pipeline()
            pipe.set(_REDIS_CACHE_PREFIX + cache_key, payload, ex=ttl)
            pipe.sadd(idx, cache_key)  # per-workspace index → real invalidation
            pipe.expire(idx, ttl + 60)
            await pipe.execute()
        except Exception as e:  # noqa: BLE001
            logger.warning("[retrieval_cache] redis set failed: %s", e)
        else:
            logger.info(
                f"[retrieval_cache] SET (redis) key={cache_key[:8]}… ({len(result.chunks)} chunks)"
            )
        return

    with _CACHE_LOCK:
        _RETRIEVAL_CACHE[cache_key] = (result, time.time())
    logger.info(f"[retrieval_cache] SET key={cache_key[:8]}… ({len(result.chunks)} chunks)")


async def invalidate_retrieval_cache(workspace_id: uuid.UUID) -> None:
    """Remove all cache entries for a workspace (call after document changes).

    The Redis path uses a per-workspace index SET so it actually clears entries.
    (The in-process path matches by str(workspace_id) prefix — but keys are
    sha256 hashes, so it never matched; kept verbatim for the disabled path.)
    """
    from app.core.redis_client import get_redis, is_redis_enabled

    if is_redis_enabled():
        try:
            r = get_redis()
            idx = _REDIS_WS_INDEX_PREFIX + str(workspace_id)
            keys = await r.smembers(idx)
            if keys:
                await r.delete(*[_REDIS_CACHE_PREFIX + k for k in keys])
            await r.delete(idx)
            logger.debug(
                f"[retrieval_cache] Invalidated {len(keys)} redis entries for workspace {workspace_id}"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[retrieval_cache] redis invalidate failed: %s", e)
        return

    with _CACHE_LOCK:
        keys_to_remove = [k for k in _RETRIEVAL_CACHE if k.startswith(str(workspace_id))]
        for k in keys_to_remove:
            _RETRIEVAL_CACHE.pop(k, None)
    logger.debug(f"[retrieval_cache] Invalidated {len(keys_to_remove)} entries for workspace {workspace_id}")


class DeepRetriever:
    """
    Hybrid retriever: KG traversal + vector similarity + cross-encoder reranking.
    """

    def __init__(
        self,
        workspace_id: uuid.UUID,
        kg_service: Optional[KnowledgeGraphService],
        vector_store: VectorStore,
        embedder: EmbeddingService,
        db: Optional[AsyncSession] = None,
        reranker: Optional[RerankerService] = None,
    ):
        self.workspace_id = workspace_id
        self.kg_service = kg_service
        self.vector_store = vector_store
        self.embedder = embedder
        self.db = db
        self.reranker = reranker or get_reranker_service()

    async def query(
        self,
        question: str,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        document_ids: Optional[list[uuid.UUID]] = None,
        include_images: bool = True,
    ) -> DeepRetrievalResult:
        """
        Execute hybrid retrieval with reranking.

        Flow:
          1. [parallel] KG query + Vector over-fetch (HRAG_VECTOR_PREFETCH)
                        + BM25 lexical search (when HRAG_ENABLE_BM25=true)
          2. Merge vector + BM25 via Reciprocal Rank Fusion (RRF)
          3. Cross-encoder rerank merged results → final top_k
          4. Optionally find related images from chunk pages
          5. Assemble structured context for LLM

        Args:
            question: Natural language query
            mode: "hybrid" (default), "naive", "local", "global", "vector_only"
            top_k: Number of final chunks to return (after reranking)
            document_ids: Optional filter to specific documents
            include_images: Whether to find related images

        Returns:
            DeepRetrievalResult with chunks, citations, context, and optional images
        """
        # Fall back to the .env-configured defaults when the caller omits them.
        if mode is None:
            mode = settings.HRAG_DEFAULT_QUERY_MODE
        if top_k is None:
            top_k = settings.HRAG_RERANKER_TOP_K

        # ── Retrieval result cache ───────────────────────────────────────────
        cache_key = _retrieval_cache_key(self.workspace_id, question, top_k, mode, document_ids)
        cached = await _get_cached_result(cache_key)
        if cached is not None:
            return cached

        # Run KG and vector search in parallel
        kg_task = None
        if self.kg_service and mode != "vector_only":
            kg_task = asyncio.create_task(
                self._kg_query(question, mode)
            )

        # Over-fetch from vector DB for reranking
        prefetch_k = max(settings.HRAG_VECTOR_PREFETCH, top_k * 3)
        vector_task = asyncio.create_task(
            asyncio.to_thread(
                self._vector_query, question, prefetch_k, document_ids
            )
        )

        # BM25 lexical search (parallel with vector search)
        bm25_task = None
        if settings.HRAG_ENABLE_BM25:
            bm25_task = asyncio.create_task(
                asyncio.to_thread(
                    self._bm25_query, question, settings.HRAG_BM25_PREFETCH, document_ids
                )
            )

        # Await results
        kg_summary = ""
        if kg_task:
            try:
                kg_summary = await kg_task
            except Exception as e:
                logger.warning(f"KG query failed, continuing with vector only: {e}")

        raw_chunks, raw_citations = await vector_task

        bm25_results: list[dict] = []
        if bm25_task:
            try:
                bm25_results = await bm25_task
            except Exception as e:
                logger.warning(f"[deep_retriever] BM25 search failed (non-fatal): {e}")

        # Merge vector + BM25 via RRF, then rerank
        if bm25_results:
            raw_chunks, raw_citations = self._rrf_merge(
                raw_chunks, raw_citations, bm25_results
            )

        # Post-RRF recency boost: favor newer documents when relevance is similar
        # This boosts both vector and BM25 results based on published_date
        if settings.HRAG_RECENTNESS_BOOST > 0:
            raw_chunks, raw_citations = self._apply_recency_boost(
                raw_chunks, raw_citations
            )

        # Hiệu lực pháp lý: tra DB (không phải metadata Chroma — trạng thái
        # đổi khi văn bản thay thế được upload sau). Chunk của văn bản
        # superseded bị demote, TRỪ khi caller scope đích danh document_ids
        # (user chủ động hỏi văn bản cũ thì vẫn trả, kèm cảnh báo ở context).
        validity_map = await self._fetch_validity_map(raw_chunks)
        demote = (
            not document_ids
            and settings.HRAG_SUPERSEDED_DEMOTE < 1.0
            and any(status == "superseded" for status, _ in validity_map.values())
        )

        # Rerank: cross-encoder scoring for precision. Khi demote, lấy dư gấp
        # đôi để chunk văn bản còn hiệu lực ngoài top_k có cơ hội thế chỗ.
        chunks, citations = await asyncio.to_thread(
            self._rerank_chunks, question, raw_chunks, raw_citations,
            top_k * 2 if demote else top_k,
        )
        for chunk in chunks:
            status, by = validity_map.get(str(chunk.document_id), ("", ""))
            chunk.validity_status = status
            chunk.superseded_by = by
        if demote:
            chunks, citations = self._demote_superseded(chunks, citations, top_k)

        # Find related images and tables
        image_refs = []
        table_refs = []
        if include_images and self.db and chunks:
            page_nos = {(str(c.document_id), c.page_no) for c in chunks if c.page_no > 0}
            if page_nos:
                # Sequential on purpose: both share self.db, and one AsyncSession
                # must never run two statements concurrently (raises
                # InvalidRequestError "concurrent operations are not permitted").
                image_refs = await self._find_related_images(page_nos)
                table_refs = await self._find_related_tables(page_nos)

        # Assemble context
        context = self._assemble_context(chunks, citations, kg_summary, image_refs, table_refs)

        result = DeepRetrievalResult(
            chunks=chunks,
            citations=citations,
            context=context,
            query=question,
            mode=mode,
            knowledge_graph_summary=kg_summary,
            image_refs=image_refs,
            table_refs=table_refs,
        )
        # Cache the result for 5 minutes
        await _set_cached_result(cache_key, result, self.workspace_id)
        return result

    async def _kg_query(self, question: str, mode: str) -> str:
        """Get raw KG context (entities + relationships) relevant to the question.

        Uses factual graph data instead of LLM-generated narrative to avoid
        hallucination from LightRAG's aquery().
        """
        if not self.kg_service:
            return ""
        try:
            return await asyncio.wait_for(
                self.kg_service.get_relevant_context(question),
                timeout=settings.HRAG_KG_QUERY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("KG raw context retrieval timed out")
            return ""
        except Exception as e:
            logger.warning(f"KG raw context failed: {e}")
            return ""

    def _bm25_query(
        self,
        question: str,
        top_n: int,
        document_ids: Optional[list[uuid.UUID]],
    ) -> list[dict]:
        """
        BM25 lexical search (synchronous, CPU-bound — run in thread).
        Returns list of dicts with id, metadata, document, bm25_rank.
        """
        from app.services.retrieval.bm25_index import bm25_search
        return bm25_search(
            vector_store=self.vector_store,
            query=question,
            top_n=top_n,
            document_ids=document_ids,
        )

    def _rrf_merge(
        self,
        vector_chunks: list[EnrichedChunk],
        vector_citations: list[Citation],
        bm25_results: list[dict],
        k: int | None = None,
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """
        Reciprocal Rank Fusion: merge vector search and BM25 results.

        RRF score = 1/(k + rank_vector) + 1/(k + rank_bm25)
        where rank is 1-indexed; items only in one list get only that term.

        Returns deduplicated list sorted by RRF score descending, preserving
        EnrichedChunk / Citation objects from the vector results when available
        (BM25-only hits are added from bm25_results metadata).
        """
        rrf_k = k or settings.HRAG_RRF_K

        # Map chunk_id → (EnrichedChunk, Citation, vector_rank 1-indexed)
        vector_map: dict[str, tuple[EnrichedChunk, Citation, int]] = {}
        for rank, (chunk, citation) in enumerate(zip(vector_chunks, vector_citations), start=1):
            chunk_id = f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}"
            vector_map[chunk_id] = (chunk, citation, rank)

        # Map chunk_id → bm25_rank 1-indexed
        bm25_map: dict[str, tuple[dict, int]] = {}
        for rank, result in enumerate(bm25_results, start=1):
            bm25_map[result["id"]] = (result, rank)

        # Collect all unique IDs
        all_ids = set(vector_map.keys()) | set(bm25_map.keys())

        scored: list[tuple[float, str]] = []
        for cid in all_ids:
            rrf_score = 0.0
            if cid in vector_map:
                rrf_score += 1.0 / (rrf_k + vector_map[cid][2])
            if cid in bm25_map:
                rrf_score += 1.0 / (rrf_k + bm25_map[cid][1])
            scored.append((rrf_score, cid))

        scored.sort(key=lambda x: x[0], reverse=True)

        merged_chunks: list[EnrichedChunk] = []
        merged_citations: list[Citation] = []

        for _, cid in scored:
            if cid in vector_map:
                chunk, citation, _ = vector_map[cid]
                merged_chunks.append(chunk)
                merged_citations.append(citation)
            else:
                # BM25-only hit: reconstruct EnrichedChunk from metadata
                bm25_hit, _ = bm25_map[cid]
                meta = bm25_hit.get("metadata", {})
                heading_path = []
                heading_str = meta.get("heading_path", "")
                if heading_str and isinstance(heading_str, str):
                    heading_path = heading_str.split(" > ")
                image_refs = [x for x in (meta.get("image_ids") or "").split("|") if x]
                table_refs = [x for x in (meta.get("table_ids") or "").split("|") if x]
                chunk = EnrichedChunk(
                    content=bm25_hit.get("document", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    source_file=meta.get("source", ""),
                    document_id=meta.get("document_id", 0),
                    page_no=meta.get("page_no", 0),
                    heading_path=heading_path,
                    image_refs=image_refs,
                    table_refs=table_refs,
                    has_table=meta.get("has_table", False),
                    has_code=meta.get("has_code", False),
                )
                merged_chunks.append(chunk)
                merged_citations.append(Citation(
                    source_file=meta.get("source", "Unknown"),
                    document_id=meta.get("document_id", 0),
                    page_no=meta.get("page_no", 0),
                    heading_path=heading_path,
                ))

        bm25_only = len(all_ids) - len(vector_map)
        if bm25_only > 0:
            logger.debug(
                f"[deep_retriever] RRF merged {len(vector_chunks)} vector + "
                f"{len(bm25_results)} BM25 → {len(merged_chunks)} unique "
                f"({bm25_only} BM25-only hits promoted)"
            )
        else:
            logger.debug(
                f"[deep_retriever] RRF merged {len(vector_chunks)} vector + "
                f"{len(bm25_results)} BM25 → {len(merged_chunks)} unique"
            )

        return merged_chunks, merged_citations

    def _apply_recency_boost(
        self,
        chunks: list[EnrichedChunk],
        citations: list[Citation],
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """
        Apply recency boost to chunks based on published_date metadata.
        Boost factor = HRAG_RECENTNESS_BOOST * exp(-days_since / decay_days)
        Newer documents get higher boost.
        """
        import math
        from datetime import datetime

        def parse_vietnamese_date(date_str: str):
            """Parse Vietnamese date formats: '15/01/2026', '01/2026', '2026'."""
            if not date_str:
                return None
            date_str = date_str.strip()
            for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    pass
            for fmt in ("%m/%Y", "%m-%Y"):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    pass
            try:
                return datetime(int(date_str[:4]), 1, 1)
            except (ValueError, IndexError):
                return None

        def compute_boost(date_str: str) -> float:
            if not date_str:
                return 0.0
            pub_date = parse_vietnamese_date(date_str)
            if pub_date is None:
                return 0.0
            try:
                now = datetime.now()
                days_since = (now - pub_date).days
                if days_since < 0:
                    days_since = 0
                boost = math.exp(-days_since / settings.HRAG_RECENTNESS_DECAY_DAYS)
                return boost
            except Exception:
                return 0.0

        boosted_chunks = []
        boosted_citations = []
        boost_factor = settings.HRAG_RECENTNESS_BOOST

        # Batch-fetch published_date for ALL chunks in a single ChromaDB call
        # instead of one .get() per chunk (was N round-trips per query).
        chunk_ids = [
            f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}" for chunk in chunks
        ]
        date_by_id: dict[str, str] = {}
        if chunk_ids:
            try:
                results = self.vector_store.collection.get(
                    ids=chunk_ids, include=["metadatas"]
                )
                got_ids = results.get("ids") or []
                got_metas = results.get("metadatas") or []
                for cid, meta in zip(got_ids, got_metas):
                    date_by_id[cid] = (meta or {}).get("published_date", "") or ""
            except Exception:
                date_by_id = {}

        for chunk_id, chunk, citation in zip(chunk_ids, chunks, citations):
            date_str = date_by_id.get(chunk_id, "")
            recency_boost = compute_boost(date_str)
            # Initialize score if not set, then apply boost
            base_score = chunk.score if chunk.score else 0.5  # default score if not set
            chunk.score = base_score * (1 + boost_factor * recency_boost)
            boosted_chunks.append(chunk)
            boosted_citations.append(citation)

        return boosted_chunks, boosted_citations

    def _vector_query(
        self,
        question: str,
        top_k: int,
        document_ids: Optional[list[uuid.UUID]],
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """Synchronous vector search via ChromaDB (over-fetch stage)."""
        query_embedding = self.embedder.embed_query(question)

        where = None
        if document_ids:
            where = {"document_id": {"$in": [str(doc_id) for doc_id in document_ids]}}

        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        chunks = []
        citations = []

        for i, doc_text in enumerate(results.get("documents", [])):
            meta = results["metadatas"][i] if results.get("metadatas") else {}

            heading_path = []
            heading_str = meta.get("heading_path", "")
            if heading_str:
                heading_path = heading_str.split(" > ") if isinstance(heading_str, str) else []

            image_refs = []
            image_ids_str = meta.get("image_ids", "")
            if image_ids_str and isinstance(image_ids_str, str):
                image_refs = [iid for iid in image_ids_str.split("|") if iid]

            table_refs = []
            table_ids_str = meta.get("table_ids", "")
            if table_ids_str and isinstance(table_ids_str, str):
                table_refs = [tid for tid in table_ids_str.split("|") if tid]

            chunk = EnrichedChunk(
                content=doc_text,
                chunk_index=meta.get("chunk_index", i),
                source_file=meta.get("source", ""),
                document_id=meta.get("document_id", 0),
                page_no=meta.get("page_no", 0),
                heading_path=heading_path,
                image_refs=image_refs,
                table_refs=table_refs,
                has_table=meta.get("has_table", False),
                has_code=meta.get("has_code", False),
            )
            chunks.append(chunk)

            citations.append(Citation(
                source_file=meta.get("source", "Unknown"),
                document_id=meta.get("document_id", 0),
                page_no=meta.get("page_no", 0),
                heading_path=heading_path,
            ))

        return chunks, citations

    def _rerank_chunks(
        self,
        question: str,
        chunks: list[EnrichedChunk],
        citations: list[Citation],
        top_k: int,
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """
        Cross-encoder reranking: score each (query, chunk) pair jointly,
        then filter by relevance threshold and return top_k.
        """
        if not chunks:
            return [], []

        # Extract texts for reranking
        doc_texts = [c.content for c in chunks]

        reranked = self.reranker.rerank(
            query=question,
            documents=doc_texts,
            top_k=top_k,
            min_score=settings.HRAG_MIN_RELEVANCE_SCORE,
        )

        if not reranked:
            # Fallback: if reranker filtered everything, keep top 3 by original order
            logger.warning(
                f"Reranker filtered all {len(chunks)} chunks below threshold "
                f"{settings.HRAG_MIN_RELEVANCE_SCORE}, falling back to top 3"
            )
            fallback_chunks = chunks[:min(3, len(chunks))]
            for c in fallback_chunks:
                c.score = 0.001 # Small non-zero score for fallback visibility
            return fallback_chunks, citations[:min(3, len(citations))]

        # Map reranked results back to original chunks/citations and set scores
        reranked_chunks = []
        reranked_citations = []
        for r in reranked:
            chunk = chunks[r.index]
            chunk.score = r.score
            reranked_chunks.append(chunk)
            reranked_citations.append(citations[r.index])

        logger.info(
            f"Reranked {len(chunks)} → {len(reranked)} chunks "
            f"(scores: {reranked[0].score:.3f} → {reranked[-1].score:.3f})"
        )

        return reranked_chunks, reranked_citations

    async def _fetch_validity_map(
        self, chunks: list[EnrichedChunk]
    ) -> dict[str, tuple[str, str]]:
        """document_id → (validity_status, superseded_by_number) cho các chunk.

        Best-effort: không có db session hoặc query lỗi → map rỗng (mọi thứ
        chạy tiếp như trước khi có validity).
        """
        if not self.db or not chunks:
            return {}
        doc_ids = []
        for c in chunks:
            try:
                doc_ids.append(uuid.UUID(str(c.document_id)))
            except (ValueError, AttributeError):
                continue
        if not doc_ids:
            return {}
        try:
            result = await self.db.execute(
                select(
                    Document.id,
                    Document.validity_status,
                    Document.superseded_by_number,
                ).where(Document.id.in_(set(doc_ids)))
            )
            return {
                str(row[0]): (row[1] or "unknown", row[2] or "")
                for row in result.all()
            }
        except Exception as e:
            logger.warning(f"[deep_retriever] validity lookup failed (non-fatal): {e}")
            return {}

    @staticmethod
    def _demote_superseded(
        chunks: list[EnrichedChunk],
        citations: list[Citation],
        top_k: int,
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """Nhân điểm chunk của văn bản superseded rồi sắp lại, cắt về top_k."""
        for chunk in chunks:
            if chunk.validity_status == "superseded":
                chunk.score *= settings.HRAG_SUPERSEDED_DEMOTE
        paired = sorted(
            zip(chunks, citations), key=lambda p: p[0].score, reverse=True
        )[:top_k]
        return [p[0] for p in paired], [p[1] for p in paired]

    async def _find_related_images(
        self,
        page_refs: set[tuple[uuid.UUID, int]],  # (document_id, page_no)
    ) -> list[ExtractedImage]:
        """Find images on the exact same pages as retrieved chunks."""
        if not self.db:
            return []

        images = []
        for doc_id, page_no in page_refs:
            result = await self.db.execute(
                select(DocumentImage).where(
                    DocumentImage.document_id == doc_id,
                    DocumentImage.page_no == page_no,
                )
            )
            for img in result.scalars().all():
                images.append(ExtractedImage(
                    image_id=img.image_id,
                    document_id=img.document_id,
                    page_no=img.page_no,
                    file_path=img.file_path,
                    caption=img.caption,
                    width=img.width,
                    height=img.height,
                    mime_type=img.mime_type,
                ))

        # Deduplicate by image_id
        seen = set()
        unique = []
        for img in images:
            if img.image_id not in seen:
                seen.add(img.image_id)
                unique.append(img)

        return unique

    async def _find_related_tables(
        self,
        page_refs: set[tuple[uuid.UUID, int]],
    ) -> list[ExtractedTable]:
        """Find tables on the exact same pages as retrieved chunks."""
        if not self.db:
            return []

        tables = []
        for doc_id, page_no in page_refs:
            result = await self.db.execute(
                select(DocumentTable).where(
                    DocumentTable.document_id == doc_id,
                    DocumentTable.page_no == page_no,
                )
            )
            for tbl in result.scalars().all():
                tables.append(ExtractedTable(
                    table_id=tbl.table_id,
                    document_id=tbl.document_id,
                    page_no=tbl.page_no,
                    content_markdown=tbl.content_markdown,
                    caption=tbl.caption,
                    num_rows=tbl.num_rows,
                    num_cols=tbl.num_cols,
                ))

        # Deduplicate by table_id
        seen = set()
        unique = []
        for tbl in tables:
            if tbl.table_id not in seen:
                seen.add(tbl.table_id)
                unique.append(tbl)

        return unique

    @staticmethod
    def _assemble_context(
        chunks: list[EnrichedChunk],
        citations: list[Citation],
        kg_summary: str,
        image_refs: list[ExtractedImage],
        table_refs: list[ExtractedTable] | None = None,
    ) -> str:
        """Assemble a structured context string for the LLM."""
        parts = []

        # KG insights
        if kg_summary:
            parts.append("## Knowledge Graph Insights")
            parts.append(kg_summary)
            parts.append("")

        # Retrieved chunks with citations
        if chunks:
            parts.append("## Retrieved Document Sections")
            for i, (chunk, citation) in enumerate(zip(chunks, citations)):
                parts.append(f"### [{i + 1}] {citation.format()}")
                if chunk.validity_status == "superseded":
                    by = (
                        f" — đã được thay thế bởi {chunk.superseded_by}"
                        if chunk.superseded_by else ""
                    )
                    parts.append(f"⚠️ VĂN BẢN NÀY ĐÃ HẾT HIỆU LỰC{by}.")
                elif chunk.validity_status == "partially_amended":
                    parts.append(
                        "⚠️ Văn bản này đã được sửa đổi/bãi bỏ một phần bởi văn bản khác."
                    )
                parts.append(chunk.content)
                parts.append("")

        # Available images
        if image_refs:
            parts.append("## Available Document Images")
            for img in image_refs:
                caption_str = f': "{img.caption}"' if img.caption else ""
                parts.append(
                    f"- Image p.{img.page_no}{caption_str} (id: {img.image_id})"
                )
            parts.append("")

        # Available tables
        if table_refs:
            parts.append("## Available Document Tables")
            for tbl in table_refs:
                caption_str = f': "{tbl.caption}"' if tbl.caption else ""
                parts.append(
                    f"- Table p.{tbl.page_no} ({tbl.num_rows}x{tbl.num_cols}){caption_str}"
                )
            parts.append("")

        if not parts:
            return "No relevant documents found for this query."

        return "\n".join(parts)

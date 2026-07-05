# Embedding & retrieval — reference

Unlike the answer/OCR LLMs, the embedder and reranker are **in-process GPU
models** (SentenceTransformers), loaded inside `hrag-backend` and the embed
worker — **not** vLLM engines. They share the same GPU as the vLLM engines, so
concurrency is bounded on purpose.

## Models (runtime, 2026-07)

| Role | Model | Where | Config |
|------|-------|-------|--------|
| Embedder | `mainguyen9/vietlegal-harrier-0.6b` | in-process, `device=cuda` | `HRAG_EMBEDDING_MODEL`, `HRAG_EMBEDDING_DEVICE` |
| Reranker (cross-encoder) | `BAAI/bge-reranker-v2-m3` | in-process | `HRAG_RERANKER_MODEL`, `HRAG_RERANKER_TOP_K=8` |
| Vector store | ChromaDB, **per-workspace collections** | `hrag-chromadb` | `CHROMA_HOST=chromadb` `:8000` (host `:8002`) |

`EmbeddingService` (`app/services/embedder.py`) lazy-loads the model onto the
device; `Reranker` (`app/services/reranker.py`) wraps a `CrossEncoder`. Both are
singletons — first call pays the load cost.

## Write path (embed worker)

`app/workers/embed_worker.py`, per document:
1. Load `raw_chunks_json` from the DB (written by parse).
2. **Contextual enrichment** (`HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS=true`): the
   memory agent generates a 1-sentence context per chunk from the doc markdown
   and **prepends** it before embedding (Anthropic contextual-retrieval). The
   context sentence + OCR-stripped chunk are kept in `raw_chunks_json` so the
   caption re-embed doesn't lose it.
3. Embed → upsert to the workspace's ChromaDB collection.
4. Clear `raw_chunks_json` (plain path) to free DB space; caption path keeps it
   until finalize.

## Read path (DeepRetriever.query)

`app/services/deep_retriever.py` — hybrid retrieval:
1. Vector search (Chroma) **+** BM25 (`HRAG_ENABLE_BM25=true`, prefetch
   `HRAG_BM25_PREFETCH`) run concurrently.
2. Fuse → cross-encoder rerank → precision filter to `HRAG_RERANKER_TOP_K` (8).
   `mode`: `hybrid` (default) | `naive` | `local` | `global` | `vector_only`.

## GPU concurrency guard (do not remove)

`app/api/chat_agent.py`: `HRAG_SEARCH_GPU_CONCURRENCY` (default **2**) →
`asyncio.Semaphore` around GPU search, plus `torch.cuda.empty_cache()` at the end
of search. This exists because a multi-workspace fan-out (5 workspaces × rerank
`to_thread`) hit 5× peak VRAM and silently OOM'd the reranker (see memory
`search-cuda-oom-silent-fail`). **Harness implication:** do not blast many
concurrent `debug-chat`/search calls — you will re-trigger the OOM the semaphore
is protecting against. The A/B harness runs cases **serially** for this reason.

## Reindex / maintenance endpoints (JWT)

| Endpoint | Use |
|----------|-----|
| `POST /rag/reindex/{document_id}` | re-embed one document |
| `POST /rag/reindex-workspace/{workspace_id}` | re-embed a whole workspace |
| `GET /rag/stats/{workspace_id}` | chunk/vector counts |
| `GET /rag/chunks/{document_id}` | inspect stored chunks |

Scripts: `python -m scripts.purge_orphan_vectors` (Chroma vectors with no DB row —
the "phantom source" bug), `python -m scripts.backfill_heading_path` (metadata-only
re-derive of heading paths for section-by-Điều retrieval).

## Gotchas

- **Collection is per-workspace** — cross-workspace search fans out and multiplies
  GPU load (the reason for the semaphore above).
- Changing `HRAG_EMBEDDING_MODEL` changes vector dimensionality → **existing
  collections become incompatible**; a reindex is required, not just a restart.
- Embedder/reranker load lazily → the first query after a `hrag-backend` restart
  is slow (model load), not a retrieval regression.

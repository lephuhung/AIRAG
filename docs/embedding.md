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
the "phantom source" bug), `python -m scripts.backfill_subdivision_nos`
(metadata-only backfill of Khoản/Điểm subdivision fields — see below; it is the
reference pattern for repairing chunk metadata in place without re-embedding).

## Legal subdivision metadata (Khoản/Điểm)

Legal chunks carry four extra Chroma metadata keys, written by the embed worker
(pipe-separated strings; internal/artifact types keep typed lists):

| Key | Example | Meaning |
|-----|---------|---------|
| `khoan_nos` | `"1|2"` | Khoản numbers covering this chunk |
| `diem_labels` | `"a|b"` | Điểm letters covering this chunk |
| `subdivision_refs` | `"khoan:1\|khoan:1/diem:a\|khoan:2"` | Atomic parent-aware refs — the only proof a điểm belongs to a khoản |
| `subdivision_schema_version` | `1` | `0`/absent = pre-backfill chunk |

A continuation chunk inherits the Khoản/Điểm interval open at its first
character (derived sequence-aware by
`app/services/parsing/heading_path.py::derive_subdivision_metadata`, never by
scanning chunks independently).

`search_document_section` resolves composite references against these fields:
`khoản 2 Điều 8` requires `2` in `khoan_nos`; `điểm a khoản 2 Điều 8` requires
the single atomic token `khoan:2/diem:a` in `subdivision_refs` (independent
index intersections are never used to infer parentage). If every Điều
candidate is schema version `0` the lookup falls back to Điều-level chunks;
once any candidate declares version `1`, an empty subdivision match is
authoritative (not-found, never semantic top-k).

Backfill for pre-existing vectors (dry-run by default, `--apply` writes;
requires at least one scope selector):

```bash
python -m scripts.backfill_subdivision_nos --workspace <WS_ID>            # dry-run
python -m scripts.backfill_subdivision_nos --collection kb_<ws> --apply
python -m scripts.backfill_subdivision_nos --document-id <ID> --apply --batch-size 200
```

It groups chunks by `(document_id, revision_id|legacy)`, sorts by `ordinal`
(falling back to `chunk_index`), merges the four keys into the full existing
metadata dict, and updates via `collection.update` — a malformed group is left
untouched atomically, and reruns are idempotent. It prints counts only, never
chunk text.

Observability: the parser emits one counts-only line per legal document,

```text
[legal-structure] document_id=... khoan_candidates=... khoan_accepted=...
diem_candidates=... diem_accepted=... ambiguous_rejected=...
```

Read it as an acceptance/rejection signal for the deterministic detector, not
a ground-truth miss ratio.

## Gotchas

- **Collection is per-workspace** — cross-workspace search fans out and multiplies
  GPU load (the reason for the semaphore above).
- Changing `HRAG_EMBEDDING_MODEL` changes vector dimensionality → **existing
  collections become incompatible**; a reindex is required, not just a restart.
- Embedder/reranker load lazily → the first query after a `hrag-backend` restart
  is slow (model load), not a retrieval regression.

## LangGraph v2 (shared capabilities)

The v2 fast and complex paths share the same capability implementations
(document search, section, KG) over this retrieval read path — no separate
retrieval stack. Factual v2 queries additionally execute revision-aware
retrieval through the `document.retrieve` capability: the request-scoped
service loads the exact `DocumentRevisionBuild` manifest for the pinned (or
workspace-resolved current) revision and queries **only that manifest's
recorded `embedding_namespace`** with hard document/revision filters — it
never falls back to a current-config namespace when the manifest is absent
or incompatible (fail closed, typed `dependency_error`). `document_ids` are
a hard scope after API ACL filtering. Probes: [`harness.md`](harness.md)
(P0 factual-retrieval live gate). Ownership model: [`CLAUDE.md`](../CLAUDE.md)
(canonical). Like every probe here, retrieval exercises the running stack
as-is — **never restart vLLM engines** ([`vllm.md`](vllm.md)).

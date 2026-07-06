# AIRAG — Hybrid RAG for Vietnamese Legal & Administrative Documents

### Vector search + Knowledge Graph + agentic chat with citations

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React_19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Upload documents → ask questions → get answers grounded in cited sources.**

AIRAG (a.k.a. Hybrid RAG / NexusRAG) is a knowledge base that combines dense vector
search, lexical BM25, temporal + legal knowledge graphs, and an LLM chat agent — tuned
for Vietnamese administrative/legal documents (Nghị quyết, Quyết định, Công văn, …) with
automatic citations, a document-validity layer, OCR for scanned PDFs, and voice I/O.

[Highlights](#highlights) · [Quick start](#quick-start) · [Architecture](#architecture) · [Scaling out](#scaling-out) · [Configuration](#configuration) · [Ports](#service-ports)

---

## Highlights

- **Hybrid retrieval** — dense (bi-encoder) + BM25 lexical, fused with Reciprocal Rank
  Fusion, then re-scored by a cross-encoder reranker; optional Knowledge-Graph context.
- **Legal-aware** — structure-aware chunking (Phần / Chương / Mục / Điều), a
  document-**validity layer** (effective / amended / superseded) that demotes or warns on
  outdated documents, and a **LegalKG** entity graph on Neo4j.
- **Agentic chat** — a single **LangGraph supervisor** classifies intent and routes in one
  LLM call, with an opt-in tool-calling **ReAct executor** for RAG queries. Answers stream
  over SSE with inline 4-char citations (e.g. `[a3z1]`).
- **Deep parsing + OCR** — Docling for born-digital PDF/DOCX/PPTX; a self-hosted vLLM OCR
  engine reconstructs scanned pages (with Vietnamese-diacritic handling).
- **Voice** — local Whisper speech-to-text for the chat mic, and TTS for spoken answers.
- **Reach the agent from anywhere** — REST + SSE, generic **API keys**, and a **Telegram**
  bot (self-service account linking, DB-managed bot config).
- **Multi-tenant** — per-workspace vector collections, KG namespace, chat history and
  optional per-document-type system prompts; JWT + tenant model; automatic **audit logging**.
- **Horizontally scalable** — an optional Redis layer makes cross-process Stop, a
  cluster-wide GPU concurrency cap, and a shared retrieval cache work across multiple
  backend workers/replicas (see [Scaling out](#scaling-out)).
- **Observability** — Langfuse LLM/LangGraph tracing + Loki/Grafana log aggregation.

---

## Quick start

Everything runs via Docker Compose. There are **three** compose files:

| File | Contents |
|---|---|
| `docker-compose.services.yml` | Full app stack — infra (PostgreSQL, ChromaDB, RabbitMQ, MinIO, Neo4j, Redis) **plus** backend, frontend, the parse/embed/caption/kg/memory workers, and the model-offload services (`embed-rerank` GPU, `stt` CPU), behind Traefik. |
| `docker-compose.vllm.yml` | The two self-hosted vLLM engines (OCR + memory/intent model). Split out because they own the GPU. |
| `docker-compose.langfuse.yml` | Optional self-hosted Langfuse tracing stack. |

```bash
cp .env.example .env
# Edit .env — set GOOGLE_AI_API_KEY (default LLM provider is Gemini),
# or switch LLM_PROVIDER=ollama | openai_compatible.

# 1) (optional) GPU inference engines — OCR + intent/memory model
docker compose -f docker-compose.vllm.yml up -d

# 2) the whole application + infrastructure
docker compose -f docker-compose.services.yml up -d
```

The stack is served through **Traefik** on `:80` (host `service.hatinh.local` → backend,
frontend on its own route). Backend health/docs are on `:8080` (`/health`, `/docs`).
Source dirs (`./backend`, frontend) are bind-mounted for live reload in the default
`development` target.

> The old top-level `run_dev.sh` / `run_bk.sh` / `run_fe.sh` / `run_workers.sh` /
> `setup.sh` scripts were removed — start everything through Docker Compose.

### Frontend-only local dev

```bash
cd frontend
pnpm install
pnpm dev      # Vite dev server on :5174 (proxies /api → backend:8080)
pnpm build    # production bundle → dist/
```

---

## Architecture

### Document processing pipeline

`parse_worker` fans out to **three independent RabbitMQ queues that run in parallel**
(not sequentially); the document is finalized once all three complete:

```
Upload ──► parse_worker ─┬─► embed_worker    → ChromaDB (vectors)
   (MinIO,  Docling/OCR)  ├─► caption_worker  → MinIO captions, then re-embeds
                          └─► kg_worker       → LegalKG (Neo4j) / LightRAG
```

Worker type is chosen per process via `WORKER_TYPE` (`parse` | `embed` | `caption` |
`kg` | `memory`). Uploads are de-duplicated by `content_hash` (sha256). Scanned PDFs are
detected and routed through the vLLM OCR engine.

### Retrieval

```
question
  ├─ dense vector over-fetch (ChromaDB)   ┐
  ├─ BM25 lexical search                  ├─ run in parallel
  └─ Knowledge-Graph lookup               ┘
        └─► RRF merge (vector + BM25)
              └─► cross-encoder rerank → top-k
                    └─► validity layer: demote / warn on superseded documents
                          └─► assemble cited context for the LLM
```

### Chat agent (LangGraph supervisor)

A single supervisor graph classifies intent **and** routes in one LLM call, with optional
memory recall (Graphiti), multi-step decomposition, and a result evaluator that may
re-route:

```
START → query_analyzer → supervisor ─┬─ rag → answer_generator
                                      ├─ resolve_doc → …
                                      ├─ write  → docx/answer
                                      ├─ people → mongo_formatter
                                      └─ direct → answer
                              → result_evaluator → END
```

Setting `NEXUSRAG_LG_RAG_REACT=true` routes RAG-group queries to a tool-calling **ReAct
executor** instead of the fixed intent→tool chain: the model is given the RAG tool schemas
(search / section / kg / resolve_doc / memory) and decides which to call.

### Storage

| Service | Purpose |
|---|---|
| **PostgreSQL** (pgvector) | Metadata, chunks, chat history, users, workspaces, tenants, audit logs, LightRAG KV/vector store |
| **ChromaDB** | Dense vector embeddings — one collection per workspace |
| **MinIO** | Raw uploads, parsed markdown, image captions (S3-compatible) |
| **Neo4j** | LegalKG entity graph + Graphiti temporal memory |
| **Redis** | Optional shared cross-process state for multi-worker (Stop, GPU cap, cache) |
| **LightRAG** | File-based graph (`data/lightrag/kb_{id}/`) when `HRAG_KG_MODE=lightrag` |

---

## Scaling out

By default the backend runs as a **single process** (`WEB_CONCURRENCY=1`), which keeps some
state in-process. A **Redis layer** (opt-in, `REDIS_ENABLED=true`) makes the backend safe to
run as multiple worker processes / replicas:

| Concern | Single process (default) | With Redis |
|---|---|---|
| Stop button (cancel a run) | in-process registry | pub/sub — reaches whichever worker owns the run |
| GPU-search concurrency cap | per-process semaphore | cluster-wide, self-healing permit (protects shared VRAM) |
| Retrieval cache | per-process dict | shared cache with per-workspace invalidation |
| DB session per request | ✅ request-private (fixed) | ✅ |

To scale out (on a machine with GPU headroom): build the **production** image
(`APP_ENV=production`), set `REDIS_ENABLED=true` and `WEB_CONCURRENCY=N`, run the inline
migrations once with a single worker then set `AUTO_CREATE_TABLES=false`. A lifespan guard
warns if these prerequisites are missing.

> ℹ️ Retrieval models no longer live in the backend process: the embedder + reranker run in a
> dedicated **`embed-rerank`** GPU service (`HRAG_EMBED_RERANK_URL`, on by default in compose)
> and Whisper STT in a **`stt`** CPU sidecar (`STT_PROVIDER=openai`). So `WEB_CONCURRENCY>1` is
> **no longer GPU-bound** — the backend scales on CPU/RAM. (Ingestion workers still embed
> locally for now.)

📖 **Full turn-up runbook (post hardware upgrade):** [`docs/scaling.md`](docs/scaling.md) —
step-by-step checklist, the GPU-VRAM constraint, per-phase knobs, and verification commands.

---

## Configuration

All settings come from `.env` (copy `.env.example`). Config keys use two prefixes:
`HRAG_*` (retrieval/pipeline) and `NEXUSRAG_*` (agent). Selected defaults:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` \| `ollama` \| `openai_compatible` |
| `GOOGLE_AI_API_KEY` | — | Required for Gemini |
| `HRAG_ENABLED` | `true` | Use the hybrid retrieval service (vs. legacy) |
| `HRAG_KG_MODE` | `legal` | `legal` (LegalKG/Neo4j) or `lightrag` |
| `HRAG_ENABLE_KG` | `true` | Toggle KG extraction |
| `HRAG_ENABLE_BM25` | `true` | Hybrid BM25 alongside vector search |
| `HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS` | `false` | LLM-generated context prepended before embedding |
| `HRAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | Bi-encoder (1024-dim); override per deployment |
| `HRAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `HRAG_EMBED_RERANK_URL` | `` (empty) | Set → embed/rerank run in the `embed-rerank` GPU service (empty = in-process); compose defaults it to `http://embed-rerank:8090` |
| `NEXUSRAG_LG_RAG_REACT` | `false` | Route RAG queries to the tool-calling ReAct executor |
| `REDIS_ENABLED` | `false` | Enable the shared cross-process state layer |
| `WEB_CONCURRENCY` | `1` | Backend worker processes (see [Scaling out](#scaling-out)) |
| `AUTO_CREATE_TABLES` | `true` | Run inline schema migrations at startup |

See `.env.example` for the full set of options.

---

## Service ports

Published by `docker-compose.services.yml` (host → container):

| Service | Port(s) | Notes |
|---|---|---|
| Traefik | `80`, `8089` | App entrypoint + dashboard |
| Backend | `8080` | FastAPI — `/health`, `/docs` |
| Frontend | via Traefik | React 19 + Vite SPA |
| PostgreSQL | `5433` | pgvector |
| ChromaDB | `8002` | Vector store |
| Redis | `6380` | Scale-out state (opt-in) |
| RabbitMQ | `5672`, `15672` | Broker + management UI (guest/guest) |
| MinIO | `9000`, `9001` | S3 API + console |
| Neo4j | `7474`, `7687` | Browser + Bolt |
| embed-rerank | `8090` | GPU embed + rerank microservice (scale-out offload) |
| stt | `8091` | Whisper speech-to-text (CPU-only sidecar) |
| Grafana / Loki | `3000` / `3100` | Log dashboards + aggregation |
| vLLM OCR / memory | `8001` / `8088` | From `docker-compose.vllm.yml` |

---

## Tech stack

**Backend** — FastAPI (async, SSE) · SQLAlchemy 2.0 + asyncpg · LangGraph · Docling ·
ChromaDB · Neo4j / LightRAG · sentence-transformers (bge-m3 / bge-reranker-v2-m3) ·
faster-whisper (STT) · Redis · RabbitMQ (aio-pika) · MinIO (aioboto3).

**Frontend** — React 19 + TypeScript 5.9 · Vite 7 · TailwindCSS 4 · Zustand · React Query
v5 · React Router v7 · react-markdown + KaTeX.

**LLM providers** — Gemini (`gemini-2.5-flash`), Ollama, or any OpenAI-compatible endpoint
(vLLM). The intent classifier / memory agent runs on a self-hosted vLLM model.

---

<div align="center">

MIT License &copy; 2026

</div>

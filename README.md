# NexusRAG

### Hybrid Knowledge Base with Agentic Chat, Citations & Knowledge Graph

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React_19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Upload documents. Ask questions. Get cited answers.**

NexusRAG combines vector search, knowledge graph extraction, and LLM-powered chat into one seamless RAG pipeline — with automatic citations, agentic routing, and support for Gemini, Ollama, or any OpenAI-compatible provider.

[Features](#features) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Tech Stack](#tech-stack) · [Configuration](#configuration)

---

## Features

### Deep Document Parsing

NexusRAG uses [Docling](https://github.com/docling-project/docling) for structural document understanding:

- **Structural preservation** — Heading hierarchy (`H1 > H2 > H3`), page boundaries, paragraph grouping
- **Multi-format** — PDF, DOCX, PPTX, HTML, TXT with consistent markdown output
- **Hybrid chunking** — Semantic + structural boundaries (respects headings, tables; never splits mid-sentence)
- **Page-aware metadata** — Every chunk carries page number, heading path, and image/table references

### Hybrid Retrieval Pipeline

| Stage | Technology | Details |
|---|---|---|
| **Embedding** | BAAI/bge-m3 | 1024-dim multilingual bi-encoder |
| **Vector Search** | ChromaDB | Cosine similarity, over-fetch top-20 |
| **Knowledge Graph** | LightRAG / LegalKG | Entity/relationship extraction for Vietnamese admin docs |
| **Reranking** | BAAI/bge-reranker-v2-m3 | Cross-encoder joint scoring |
| **Generation** | Gemini / Ollama / OpenAI-compatible | Streaming chat with function calling |

**Retrieval flow:**
1. Vector over-fetch (top-20) + KG entity lookup run in parallel
2. Cross-encoder reranking — all candidates scored jointly with the query
3. Keep top-8 above relevance threshold, with fallback to top-3
4. Media discovery — find images/tables on matched pages

### Citation System

Every answer is grounded in source documents with **4-character citation IDs** (e.g., `[a3z1]`):

- Inline clickable badges embedded in answer text
- Source cards showing filename, page number, heading path, and relevance score
- Cross-navigation — click a citation to jump to the exact section in the document viewer
- Image references cited separately as `[IMG-p4f2]`

### Knowledge Graph

Interactive force-directed graph built from extracted entities and relationships:

- **Entity types** — Person, Organization, Product, Location, Event, Regulation, etc.
- **Pan & zoom** — Mouse drag, scroll wheel, keyboard reset
- **Node interaction** — Click to select, hover to highlight edges, drag to reposition
- **Query modes** — Naive, Local (multi-hop), Global, Hybrid

NexusRAG also includes **LegalKG** — a domain-specific KG extraction service tuned for Vietnamese administrative documents (Nghị quyết, Quyết định, Công văn, etc.).

### Agentic Chat

A LangGraph-based chat agent with real-time SSE streaming:

- **Agent steps** — Visual timeline: Analyzing → Retrieving → Generating → Done
- **11 intent types** — `search`, `list_docs`, `summarize`, `kg_query`, `search_doc_num`, `search_abbr`, `write_summarize`, `write_suggest_edits`, `write_grammar_check`, `greeting`, `personal`
- **Extended thinking** — Configurable reasoning depth (minimal → high)
- **Chat history** — Persistent per workspace with message ratings

### Multi-Provider LLM

Switch between cloud and local models with a single env var:

| Provider | Models | Notes |
|---|---|---|
| **Gemini** | `gemini-2.5-flash`, `gemini-3.1-flash-lite` | Cloud — auto thinking budget |
| **Ollama** | `qwen3.5:9b`, `gemma3:12b`, etc. | Local — native protocol |
| **OpenAI-compatible** | vLLM, LM Studio | Self-hosted OpenAI-compatible endpoints |

### Workspace Isolation

Each workspace has its own:
- ChromaDB collection
- LightRAG KG directory
- Chat history
- Custom system prompt (optional override per document type)

Multi-tenancy enforced via PostgreSQL `tenant` model and JWT claims.

---

## Quick Start

### Option A: Docker (Full Stack)

```bash
cp .env.example .env
# Edit .env — set GOOGLE_AI_API_KEY (or configure Ollama)
docker compose up -d
```

First build takes ~5–10 minutes (~2.5GB ML models). Access at http://localhost:5174

### Option B: Local Development

```bash
./setup.sh                    # One-time setup: venv, pip deps, infra services, frontend deps
./run_dev.sh                  # Starts backend + frontend + workers combined
```

Or manually in three terminals:

```bash
# Terminal 1 — Backend (port 8080)
./run_bk.sh

# Terminal 2 — Frontend (port 5174)
./run_fe.sh

# Terminal 3 — Workers (parse, embed, caption, kg)
./run_workers.sh
```

### Frontend Only

```bash
cd frontend
pnpm install
pnpm dev          # Dev server on port 5174
pnpm build        # Production bundle → dist/
```

---

## Architecture

### Document Processing Pipeline

```
Upload → parse_worker → embed_worker → caption_worker → kg_worker
             ↓                ↓               ↓               ↓
           MinIO          ChromaDB       MinIO (captions)  LightRAG / LegalKG
```

Workers are selected at runtime via the `WORKER_TYPE` env var.

### Chat Agent (LangGraph)

```
START
  → abbr_expander        (expand abbreviations before routing)
  → memory_recall        (load user memories from Graphiti)
  → intent_classifier    (classify + rewrite query via Qwen3-4B)
  → [direct_answer]      (greeting / personal intent)
  → [write_executor]     (write_summarize / suggest_edits / grammar_check)
  → [agent_rag_executor] (search / list_docs / summarize / kg_query / search_doc_num / search_abbr)
      → [write_executor] (intent=summarize: RAG fetches doc, Write summarizes)
      → [answer_generator]
  → END
```

Controlled by `NEXUSRAG_AGENT_BACKEND` (`legacy` | `langgraph`).

### Storage

| Service | Purpose |
|---|---|
| **PostgreSQL** | Document metadata, chunks, chat history, workspaces, users, tenants |
| **ChromaDB** | Vector embeddings (HTTP client, containerized) |
| **MinIO** | Raw uploads, parsed markdown, captions (S3-compatible) |
| **LightRAG** | File-based KG (NetworkX + NanoVectorDB) |
| **Neo4j** | Optional KG backend for LegalKGService |

---

## Tech Stack

### Backend

| Technology | Purpose |
|---|---|
| **FastAPI** | Async web framework with SSE streaming |
| **SQLAlchemy 2.0** | Async ORM with PostgreSQL (asyncpg) |
| **ChromaDB** | Vector store — per-workspace collections |
| **LightRAG / LegalKG** | Knowledge graph extraction |
| **Docling** | Document parsing (PDF, DOCX, PPTX) |
| **BAAI/bge-m3** | Embeddings (1024-dim) |
| **BAAI/bge-reranker-v2-m3** | Cross-encoder reranking |
| **LangGraph** | Agent state machine |

### Frontend

| Technology | Purpose |
|---|---|
| **React 19** + **TypeScript** | UI framework |
| **Vite** | Dev server and bundler |
| **TailwindCSS** | Styling with dark/light theme |
| **Zustand** | Client state management |
| **React Query v5** | Server state (fetching, caching, mutations) |
| **Framer Motion** | Animations |
| **React Router v7** | Routing (multi-page SPA) |
| **react-markdown** + **KaTeX** | Markdown + LaTeX rendering |

### Infrastructure

| Technology | Purpose |
|---|---|
| **PostgreSQL 15** (pgvector) | Metadata + vector storage |
| **ChromaDB** | Vector embeddings |
| **RabbitMQ** | Async job queue |
| **MinIO** | Object storage |
| **nginx** | Reverse proxy + SPA serving |
| **Docker Compose** | Full-stack deployment |

---

## Configuration

All configuration via `.env` (copy from `.env.example`). Two prefix families coexist:

- `NEXUSRAG_*` — core pipeline settings
- `HRAG_*` — additional features (contextual embeddings, BM25)

### Key Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `ollama`, or `openai_compatible` |
| `GOOGLE_AI_API_KEY` | — | Required for Gemini |
| `NEXUSRAG_AGENT_BACKEND` | `langgraph` | `legacy` or `langgraph` |
| `NEXUSRAG_ENABLE_KG` | `true` | Toggle knowledge graph extraction |
| `NEXUSRAG_ENABLE_IMAGE_CAPTIONING` | `true` | LLM captioning for extracted images |
| `NEXUSRAG_VECTOR_PREFETCH` | `20` | Candidates before reranking |
| `NEXUSRAG_RERANKER_TOP_K` | `8` | Final results after reranking |
| `HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS` | `true` | Contextual embedding enrichment |
| `HRAG_ENABLE_BM25` | `false` | Enable BM25 alongside vector search |

See `.env.example` for all ~208 configuration options.

---

## API

All endpoints prefixed with `/api/v1`. Interactive docs at http://localhost:8080/docs

### Core Endpoints

| Method | Path | Description |
|---|---|---|
| `GET/POST` | `/workspaces` | List / create workspaces |
| `PUT/DELETE` | `/workspaces/{id}` | Update / delete workspace |
| `POST` | `/documents/upload/{workspace_id}` | Upload document |
| `GET` | `/documents/{id}/markdown` | Get parsed content |
| `DELETE` | `/documents/{id}` | Delete document |
| `POST` | `/rag/query/{workspace_id}` | Hybrid search |
| `POST` | `/rag/chat/{workspace_id}/stream` | Streaming chat (SSE) |
| `GET` | `/rag/chat/{workspace_id}/history` | Chat history |
| `GET` | `/rag/graph/{workspace_id}` | Knowledge graph data |
| `GET` | `/rag/analytics/{workspace_id}` | Workspace analytics |

---

## Document Types

NexusRAG classifies Vietnamese administrative documents into 29 types:

Nghị quyết (cá biệt), Quyết định (cá biệt), Chỉ thị, Quy chế, Quy định, Thông báo, Hướng dẫn, Chương trình, Kế hoạch, Phương án, Đề án, Dự án, Báo cáo, Biên bản, Tờ trình, Hợp đồng, Công văn, Công điện, Bản ghi nhớ, Bản thỏa thuận, Giấy ủy quyền, Giấy mời, Giấy giới thiệu, Giấy nghỉ phép, Phiếu gửi, Phiếu chuyển, Phiếu báo, Thư công.

Each type can have a custom system prompt and KG extraction prompt per workspace.

---

<div align="center">

MIT License &copy; 2026

</div>

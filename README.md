# NexusRAG

Knowledge Base management system with hybrid semantic search, knowledge graph, cross-encoder reranking, and LLM-powered agentic chat with citations.

## Features

- **Document Processing** — PDF, DOCX, PPTX, HTML, TXT, MD parsing via Docling with structural extraction
- **Hybrid Retrieval** — ChromaDB vector search + LightRAG knowledge graph + cross-encoder reranking (BAAI/bge-reranker-v2-m3)
- **Multilingual Embeddings** — BAAI/bge-m3 (1024-dim, 100+ languages)
- **Agentic Streaming Chat** — Gemini-powered with function calling (`search_documents` tool), SSE streaming, and extended thinking visualization
- **Citation System** — 4-char alphanumeric citation IDs (e.g. `[a3z1]`) with document/page/heading tracking
- **Image Extraction** — Automatic image extraction from documents with LLM-generated captions (Gemini Vision)
- **Table Extraction** — Structured table parsing to markdown with captions
- **Knowledge Graph** — LightRAG entity/relationship extraction with configurable entity types and graph visualization
- **Multi-model Support** — Gemini (primary) + Ollama (local fallback)
- **Extended Thinking** — Gemini 2.5 thinking budget + Gemini 3.x thinking levels (minimal/low/medium/high)
- **Workspace System** — Per-workspace document isolation, custom system prompts, chat history persistence
- **Analytics Dashboard** — Document stats, chunk metrics, KG entity/relationship counts

## Architecture

```
Frontend (React 19 + Vite 7 + TailwindCSS 4 + Zustand)
  ↕ REST API + SSE Streaming
Backend (FastAPI + SQLAlchemy 2.0 async)
  ↕
PostgreSQL 15 (metadata + chat history)
ChromaDB (vector embeddings)
LightRAG (knowledge graph — file-based, no extra services)
  ↕
LLM: Gemini API (chat + KG extraction + image captioning)
     Ollama (local fallback)
Embeddings: BAAI/bge-m3 (sentence-transformers)
Reranker: BAAI/bge-reranker-v2-m3 (cross-encoder)
```

## System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB+ |
| Disk | 5 GB (models + deps) | 10 GB+ |
| Python | 3.10+ | 3.11+ |
| Node.js | 18+ | 22 LTS |
| Docker | 20+ (optional for local dev) | Latest |

## Quick Start

### Option A: Docker (Full Stack — 1 command)

```bash
cd NexusRAG
cp .env.example .env
# Edit .env — at minimum set GOOGLE_AI_API_KEY
docker compose up -d
```

This starts all 4 services: **PostgreSQL**, **ChromaDB**, **Backend**, **Frontend**.

First build takes ~5-10 minutes (downloads ML models ~2.5GB).

Open http://localhost:5174

### Option B: Local Development (setup script)

```bash
cd NexusRAG
./setup.sh
```

The script will:
1. Check prerequisites (Python 3.10+, Node 18+, pnpm, Docker)
2. Create Python venv and install dependencies
3. Create `.env` from `.env.example`
4. Start PostgreSQL + ChromaDB via Docker
5. Optionally pre-download ML models (~2.5GB)
6. Install frontend dependencies

Then start the servers:

```bash
# Terminal 1 — Backend (port 8080)
./run_bk.sh

# Terminal 2 — Frontend (port 5174)
./run_fe.sh
```

Open http://localhost:5174

### Option C: Manual Setup

```bash
# 1. Start services
docker compose -f docker-compose.services.yml up -d

# 2. Configure environment
cp .env.example .env
# Edit .env — at minimum set GOOGLE_AI_API_KEY

# 3. Backend
python3 -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload --port 8080

# 4. Frontend (new terminal)
cd frontend && pnpm install && pnpm dev
```

Open http://localhost:5174

## Environment Variables

### Required

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...@localhost:5433/nexusrag` | PostgreSQL connection string |
| `GOOGLE_AI_API_KEY` | — | Google AI API key (required for Gemini provider) |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8002` | ChromaDB port |

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | LLM provider: `gemini` or `ollama` |
| `LLM_MODEL_FAST` | `gemini-2.5-flash` | Gemini model for chat and KG extraction |
| `LLM_THINKING_LEVEL` | `medium` | Thinking level for Gemini 3.x+: `minimal`/`low`/`medium`/`high` |
| `LLM_MAX_OUTPUT_TOKENS` | `8192` | Max output tokens (includes thinking tokens) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3:12b` | Ollama model name |

### RAG Pipeline

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXUSRAG_ENABLED` | `true` | Enable NexusRAG pipeline |
| `NEXUSRAG_ENABLE_KG` | `true` | Enable knowledge graph extraction |
| `NEXUSRAG_ENABLE_IMAGE_EXTRACTION` | `true` | Extract images from documents |
| `NEXUSRAG_ENABLE_IMAGE_CAPTIONING` | `true` | Generate image captions via LLM |
| `NEXUSRAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence-transformers embedding model (1024-dim) |
| `NEXUSRAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker model |
| `NEXUSRAG_CHUNK_MAX_TOKENS` | `512` | Max chunk size in characters |
| `NEXUSRAG_VECTOR_PREFETCH` | `20` | Number of candidates before reranking |
| `NEXUSRAG_RERANKER_TOP_K` | `8` | Top results after reranking |
| `NEXUSRAG_MIN_RELEVANCE_SCORE` | `0.15` | Minimum relevance threshold |
| `NEXUSRAG_DEFAULT_QUERY_MODE` | `hybrid` | Default retrieval mode |
| `NEXUSRAG_KG_LANGUAGE` | `Vietnamese` | KG extraction language |

### KG Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `KG_EMBEDDING_PROVIDER` | `gemini` | Embedding provider for KG |
| `KG_EMBEDDING_MODEL` | `gemini-embedding-001` | KG embedding model |
| `KG_EMBEDDING_DIMENSION` | `3072` | KG embedding dimension |

## API Endpoints

All endpoints are prefixed with `/api/v1`. Full API docs at http://localhost:8000/docs

### Workspaces

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/workspaces` | List all workspaces |
| `POST` | `/workspaces` | Create workspace |
| `GET` | `/workspaces/summary` | Compact list for dropdowns |
| `GET` | `/workspaces/{id}` | Get workspace details |
| `PUT` | `/workspaces/{id}` | Update name/description/system_prompt |
| `DELETE` | `/workspaces/{id}` | Delete workspace + cleanup |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/documents/workspace/{workspace_id}` | List documents in workspace |
| `POST` | `/documents/upload/{workspace_id}` | Upload file (PDF/DOCX/PPTX/HTML/TXT/MD) |
| `GET` | `/documents/{id}` | Get document metadata |
| `GET` | `/documents/{id}/markdown` | Get parsed markdown content |
| `GET` | `/documents/{id}/images` | List extracted images |
| `DELETE` | `/documents/{id}` | Delete document |

### RAG — Search & Processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rag/query/{workspace_id}` | Hybrid semantic search (vector + KG + reranking) |
| `POST` | `/rag/process/{document_id}` | Process document (parse → chunk → embed → index) |
| `POST` | `/rag/process-batch` | Batch process multiple documents |
| `POST` | `/rag/reindex/{document_id}` | Re-process a single document |
| `POST` | `/rag/reindex-workspace/{workspace_id}` | Reindex all documents in workspace |

### RAG — Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rag/chat/{workspace_id}` | One-shot chat with citations |
| `POST` | `/rag/chat/{workspace_id}/stream` | Agentic streaming chat (SSE) with function calling |
| `GET` | `/rag/chat/{workspace_id}/history` | Get chat history |
| `DELETE` | `/rag/chat/{workspace_id}/history` | Clear chat history |
| `POST` | `/rag/chat/{workspace_id}/rate` | Rate a response (thumbs up/down) |
| `POST` | `/rag/debug-chat/{workspace_id}` | Debug endpoint with full retrieval trace |

### RAG — Analytics & Knowledge Graph

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/rag/stats/{workspace_id}` | Document counts, chunk stats |
| `GET` | `/rag/analytics/{workspace_id}` | Full analytics with per-document breakdown |
| `GET` | `/rag/graph/{workspace_id}` | Knowledge graph nodes and edges |
| `GET` | `/rag/entities/{workspace_id}` | Extracted entities |
| `GET` | `/rag/relationships/{workspace_id}` | Extracted relationships |

## Tech Stack

### Backend
- **FastAPI** — Async web framework
- **SQLAlchemy 2.0** — Async ORM with PostgreSQL (asyncpg)
- **ChromaDB** — Vector store for semantic search
- **LightRAG** — Knowledge graph construction and querying
- **Docling** — High-fidelity document parsing (PDF/DOCX/PPTX/HTML)
- **sentence-transformers** — BAAI/bge-m3 embeddings + BAAI/bge-reranker-v2-m3 reranking
- **google-genai** — Gemini API (chat, thinking, vision, function calling)
- **ollama** — Local LLM fallback

### Frontend
- **React 19** + **TypeScript 5.9**
- **Vite 7** — Dev server and bundler
- **TailwindCSS 4** — Utility-first styling
- **Zustand 5** — Lightweight state management
- **React Query 5** — Async data fetching and caching
- **Framer Motion 12** — Animations and transitions
- **react-markdown** + **remark-gfm** + **KaTeX** — Rich markdown rendering with math support
- **react-syntax-highlighter** — Code block syntax highlighting
- **lucide-react** — Icons

### Infrastructure
- **PostgreSQL 15** — Document metadata, chat history, workspace config
- **ChromaDB** — Vector embeddings storage
- **LightRAG** — File-based knowledge graph (NetworkX + NanoVectorDB, no extra services)

## RAG Pipeline

```
                    ┌─────────────────────────────┐
                    │     Document Upload          │
                    │  (PDF/DOCX/PPTX/HTML/TXT)   │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │     Docling Parser           │
                    │  → Markdown + Images + Tables│
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
   ┌──────────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │  Text Chunking  │  │   Image     │  │   Table     │
   │  (512 chars,    │  │ Extraction  │  │ Extraction  │
   │   50 overlap)   │  │ + Captions  │  │ → Markdown  │
   └───────┬─────────┘  └─────────────┘  └─────────────┘
           │
     ┌─────┼─────────────────┐
     │                       │
┌────▼─────────┐   ┌────────▼────────┐
│  ChromaDB    │   │   LightRAG      │
│  bge-m3      │   │   KG Extraction │
│  (1024-dim)  │   │   (Entities +   │
│              │   │    Relations)    │
└──────────────┘   └─────────────────┘


           Query Flow
           ─────────
    ┌─────────────────────┐
    │   User Question      │
    └──────────┬──────────┘
               │
     ┌─────────┼──────────────┐
     │ (parallel)             │
┌────▼────────┐    ┌──────────▼────────┐
│ Vector      │    │  KG Query         │
│ Search      │    │  (local/global/   │
│ (prefetch   │    │   hybrid)         │
│  top-20)    │    └───────────────────┘
└──────┬──────┘
       │
┌──────▼──────────────┐
│ Cross-encoder       │
│ Reranking           │
│ (bge-reranker-v2-m3)│
│ → top-K results     │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│ LLM Generation      │
│ (Gemini + thinking) │
│ → Answer with       │
│   [citation IDs]    │
└─────────────────────┘
```

## Project Structure

```
NexusRAG/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── core/
│   │   │   ├── config.py            # Settings & env vars
│   │   │   ├── database.py          # SQLAlchemy async setup
│   │   │   ├── deps.py              # Dependency injection
│   │   │   └── exceptions.py        # Error handlers
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   │   ├── knowledge_base.py    # Workspace model
│   │   │   ├── document.py          # Document + Image + Table
│   │   │   └── chat_message.py      # Chat history
│   │   ├── schemas/                 # Pydantic DTOs
│   │   ├── api/                     # REST endpoints
│   │   │   ├── router.py            # Route aggregator
│   │   │   ├── workspaces.py        # Workspace CRUD
│   │   │   ├── documents.py         # Document upload/management
│   │   │   ├── rag.py               # Search, process, analytics
│   │   │   ├── chat_agent.py        # Agentic streaming chat
│   │   │   └── chat_prompt.py       # System prompt templates
│   │   └── services/                # Business logic
│   │       ├── deep_rag_service.py        # Pipeline orchestrator
│   │       ├── deep_document_parser.py    # Docling parsing
│   │       ├── deep_retriever.py          # Hybrid retrieval
│   │       ├── knowledge_graph_service.py # LightRAG wrapper
│   │       ├── vector_store.py            # ChromaDB operations
│   │       ├── embedder.py                # Embedding service
│   │       ├── reranker.py                # Cross-encoder reranking
│   │       ├── chunker.py                 # Text chunking
│   │       └── llm/                       # LLM provider abstraction
│   │           ├── base.py                # Abstract interface
│   │           ├── gemini.py              # Gemini implementation
│   │           ├── ollama.py              # Ollama implementation
│   │           └── types.py               # LLMMessage, StreamChunk
│   ├── data/                        # Runtime data
│   │   ├── docling/                 # Extracted images per workspace
│   │   └── lightrag/                # KG storage per workspace
│   ├── scripts/
│   │   └── download_models.py       # Pre-download ML models
│   ├── uploads/                     # Uploaded files
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # Root router
│   │   ├── pages/
│   │   │   ├── KnowledgeBasesPage.tsx
│   │   │   └── WorkspacePage.tsx
│   │   ├── components/
│   │   │   ├── rag/                 # Chat, search, upload, KG, analytics
│   │   │   ├── layout/             # AppShell, Sidebar, TopBar
│   │   │   └── ui/                 # UI primitives
│   │   ├── hooks/                   # useRAGChatStream, useChatHistory
│   │   ├── stores/                  # Zustand stores
│   │   ├── lib/                     # API client, utils
│   │   └── types/
│   ├── .nvmrc                       # Node version spec
│   └── package.json
├── docker-compose.yml               # Full stack (prod)
├── docker-compose.services.yml      # Dev services only (PostgreSQL + ChromaDB)
├── Dockerfile.backend               # Backend Docker image
├── Dockerfile.frontend              # Frontend Docker image (nginx)
├── nginx.conf                       # Nginx config for frontend
├── setup.sh                         # Local dev setup script
├── .env.example                     # Environment variables template
└── README.md
```

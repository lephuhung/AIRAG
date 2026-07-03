<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **AIRAG** (6661 symbols, 14593 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/AIRAG/context` | Codebase overview, check index freshness |
| `gitnexus://repo/AIRAG/clusters` | All functional areas |
| `gitnexus://repo/AIRAG/processes` | All execution flows |
| `gitnexus://repo/AIRAG/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

---

# AIRAG / HRAG — Agent Guidance

## Project Name

The repo is named **AIRAG** on disk; the product is **HRAG** (Hybrid RAG). Both names appear throughout the codebase and docs.

## Dev Commands

### Infrastructure (required before backend)
```bash
docker compose -f docker-compose.services.yml up -d
# Starts: PostgreSQL:5433, ChromaDB:8002, RabbitMQ:5672+15672, MinIO:9000+9001, Neo4j:7474+7687, vLLM-OCR:8001, vLLM-memory:8088
```

### Backend
```bash
./run_dev.sh        # Starts uvicorn on :8080 (backend must already be activated in .venv)
# Prerequisites: infra services above must be healthy (script prompts to continue if not)
```

### Frontend
```bash
cd frontend
pnpm install        # First time only
pnpm dev            # Vite dev server on :5174
pnpm build          # TypeScript compile + Vite bundle → dist/
pnpm lint           # ESLint
```

### No formal test suite — use http://localhost:8080/docs for manual API testing
Eval scripts exist in `backend/scripts/` (`eval_rag.py`, `eval_ragas_synthetic.py`).

## Architecture

### Two Parallel Agent Systems

The backend has **three** agent entry points (not two):

| File | Type | Entrypoint | Config |
|------|------|------------|--------|
| `app/api/chat_agent.py` | Legacy SSE loop — manual LLM tool-calling | `agent_chat_stream()` | `NEXUSRAG_AGENT_BACKEND=legacy` (default) |
| `app/api/chat_agent_lg.py` | LangGraph supervisor graph | `stream_agent_to_sse(graph, state)` via `get_supervisor_graph()` | `NEXUSRAG_AGENT_BACKEND=langgraph` |
| `app/services/agent/` (dir) | **Thin compatibility shim** — no own graph | Re-exports `get_supervisor_graph` as `build_agent_graph` | Used by LangGraph endpoint |

**Supervisor graph** (`app/services/agents/supervisor.py`) is the actual LangGraph implementation:

```
START → supervisor_node (classify + route in ONE LLM call)
          ↓
       memory_recall (Graphiti personal context)
          ↓
    [rag | write | people | direct]  ← routed by supervisor's next_agent
          ↓                    ↓                    ↓
   _rag_agent_wrapper   _write_agent_wrapper  _people_agent_wrapper
   (search, list, kg,    (summarize, edit,     (MongoDB people
    abbr, doc_num)       grammar, format)      search by
                                             CCCD/name/BHXH/phone)
          ↓                    ↓                    ↓
       answer_generator_node (main LLM generates final answer)
          ↓
         END
```

**Confusion trap**: `app/services/agent/__init__.py` imports `get_supervisor_graph` from `app/services/agents/` and aliases it as `build_agent_graph`. There is NO separate `graph.py` — the only graph builder is `create_supervisor_graph()` in `supervisor.py`. The `app/services/agent/` dir is a backward-compat shim, not a separate architecture.

**`NEXUSRAG_AGENT_BACKEND=langgraph`** routes to `chat_agent_lg.py` → uses `get_supervisor_graph()` (supervisor architecture). There is no second config flag for supervisor vs. original LangGraph — they are the same thing.

### Document Processing Pipeline (Async Workers)
```
Upload → parse_worker → embed_worker → caption_worker → kg_worker
          MinIO       ChromaDB      MinIO (captions)  LightRAG / LegalKG
```
Worker type selected at runtime via `WORKER_TYPE` env var when running `python -m app.workers.runner`.

### Storage
- **PostgreSQL** — metadata, chat history, users, workspaces
- **ChromaDB** — vector embeddings (per-workspace collections)
- **MinIO** — raw uploads and parsed markdown
- **LightRAG** — file-based KG (NetworkX + NanoVectorDB) in `backend/data/lightrag/kb_{id}/`
- **Neo4j** — Graphiti temporal memory + optional LegalKG backend

### DB Migrations
No Alembic. Schema migrations run inline in `app/main.py` lifespan using raw SQL (`ALTER TABLE IF EXISTS`).

### Two LLM Providers Used Simultaneously
- **Intent classifier**: always uses memory agent (Qwen3-4B via `MEMORY_AGENT_BASE_URL`)
- **Main LLM / answer generator**: uses `LLM_PROVIDER` (`gemini` | `ollama` | `openai_compatible`)
- Both are called from the same request flow.

### Frontend Stack
React 19 + TypeScript + Vite + TailwindCSS 4 + Zustand + React Query v5 + Framer Motion. SSE streaming via `useRAGChatStream` hook.

## Key Config Variables

| Variable                                     | Default  | Notes                                     |            |                     |
| ----------------------------------------------| ----------| -------------------------------------------| ------------| ---------------------|
| `NEXUSRAG_AGENT_BACKEND`                     | `legacy` | `langgraph` enables services/agent/ graph |            |                     |
| `LLM_PROVIDER`                               | `gemini` | `gemini` \                                | `ollama` \ | `openai_compatible` |
| `NEXUSRAG_LG_MAX_ITERATIONS`                 | `6`      | Loop guard for LangGraph agents           |            |                     |
| `NEXUSRAG_ENABLE_KG`                         | `true`   | Toggle KG extraction                      |            |                     |
| `HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS`          | `false`  | LLM-generated context before embedding    |            |                     |
| `HRAG_ENABLE_BM25`                           | `true`   | Hybrid BM25 alongside vector search       |            |                     |
| `HRAG_KG_MODE`                               | `legal`  | `legal` (LegalKGService) or `lightrag`    |            |                     |
| `NEXUSRAG_LG_USE_MEMORY_AGENT_AS_CLASSIFIER` | `true`   | Use Qwen3-4B for intent classification    |            |                     |

## Repo Conventions

- **`from __future__ import annotations`** used throughout — do not remove
- **`app/services/llm/`** — LLM provider abstraction with `BaseLLMProvider`
- **`app/services/agent/`** and **`app/services/agents/`** — two separate agent dirs; do not confuse them
- **SSE events** emitted via `push_event()` from `services/agent/streaming.py` using ContextVar
- **16 intent types** defined in `services/agents/models.py` (Intent class)
- **MongoDB** used for people search only (separate from PostgreSQL main DB)
- **Abbreviation expansion** happens before intent classification in both agent systems

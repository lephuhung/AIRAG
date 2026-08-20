# OpenViking Memory Backend

Agent long-term memory via [OpenViking](https://github.com/volcengine/OpenViking)
— an open-source context database that stores memories as a tiered virtual
filesystem (`viking://`) and runs its own async LLM extraction pipeline on
session commit.

OpenViking is an **optional** second memory backend alongside Graphiti. It is
wired into the same LangGraph pipeline nodes (read: `memory_recall` +
`query_enricher` + ReAct memory tools; write: `memory_worker` + chat endpoints)
through a small dispatcher so switching backends is a config flag, not a code
change.

## When to use it

- You want richer per-user memory: OpenViking extracts preferences, events,
  identities and experiences from full conversations (both sides of the
  dialogue), tiered as L0/L1/L2 with on-demand loading.
- You want memory you can inspect/debug: `ov ls viking://user/…/memories` shows
  files like a filesystem, with retrieval trajectories.
- You want to share a memory server across multiple agent stacks.

Graphiti remains the default (temporal KG with time-aware fact invalidation);
OpenViking complements it — `NEXUSRAG_MEMORY_BACKEND=both` writes to both and
merges recall.

## How it works

```
user turn ─┬─ LangGraph supervisor (NEXUSRAG_AGENT_BACKEND=langgraph)
           │
           ├─ recall:  memory_recall node → memory_backend.search_user_memory()
           │           └─ OpenViking: semantic search over
           │              viking://user/nexusrag_{user_id}/memories
           │              (vector search w/ intent analysis; identity queries
           │              fall back to reading the memory directory directly)
           │
           └─ capture: chat endpoint → hrag.memory queue → memory_worker
                       → memory_backend.add_conversation_episode()
                       └─ OpenViking: create session nexusrag_{session_id},
                          add user+assistant messages, commit()
                          → server-side async: archive + LLM summary + memory
                            extraction into the user's memories namespace
```

Each AIRAG user maps to an OpenViking user ``nexusrag_{uuid}`` under a shared
account ``nexusrag_`` (the ``nexusrag_`` prefix keeps multiple stacks sharing one
server collision-free). The OpenViking server runs in **trusted auth mode**
(``server.auth_mode: "trusted"``): AIRAG is the trusted server-side party and
asserts each user per-request via ``X-OpenViking-Account`` + ``X-OpenViking-User``
headers, authenticated with the root API key (``OPENVIKING_API_KEY``). This is
required — a bare ROOT key is rejected from tenant-scoped data APIs. The SDK
generally supplies these headers automatically (one cached client per AIRAG user).

## Enabling

```bash
# 1. Copy the new vars into .env (see .env.example):
#    NEXUSRAG_MEMORY_BACKEND=openviking      # or both
#    OPENVIKING_URL=http://openviking:1933   # in-compose URL (host: http://localhost:1933)
#    OPENVIKING_API_KEY=...                  # matches root_api_key in the server config

# 2. Start the OpenViking server (defined in docker-compose.services.yml):
docker compose -f docker-compose.services.yml up -d openviking

# 3. Restart the backend + memory worker so the new env applies:
docker compose -f docker-compose.services.yml restart hrag-backend worker-memory
```

The compose definition injects the full `ov.conf` via
`OPENVIKING_CONF_CONTENT` and reuses AIRAG's own model services:

| OpenViking needs | Wired to |
|------------------|----------|
| Embedding (dense) | `embed-rerank` service → `POST /v1/embeddings` (OpenAI-compatible; same SentenceTransformer model as RAG, 1024-dim) |
| VLM / LLM (summaries + memory extraction) | memory-agent LLM proxy (`host.docker.internal:20128`, model `airag-memory`) |

No new GPU model is required and the backend request path holds nothing extra;
the OpenViking server does its own (background, throttled) vectorization and
LLM calls.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXUSRAG_MEMORY_BACKEND` | `graphiti` | `graphiti` \| `openviking` \| `both` \| `none` |
| `OPENVIKING_URL` | _(empty)_ | Server URL; empty = OpenViking disabled. Compose: `http://openviking:1933` |
| `OPENVIKING_API_KEY` | _(empty)_ | Root/API key (`root_api_key` in the server ov.conf) — must match |
| `OPENVIKING_ACCOUNT` | `nexusrag` | Shared OpenViking account all AIRAG users live under (trusted mode) |
| `OPENVIKING_TIMEOUT` | `30.0` | HTTP client timeout (s) |
| `OPENVIKING_TOP_K` | `5` | Top-k memory contexts recalled per search |
| `OPENVIKING_SCORE_THRESHOLD` | _(None)_ | Optional minimum relevance score for recall |
| `OPENVIKING_VLM_BASE_URL` / `_MODEL` / `_API_KEY` | memory-agent proxy / `airag-memory` | LLM used by the OpenViking server for extraction (via ov.conf) |
| `OPENVIKING_EMBED_BASE_URL` / `_MODEL` / `_API_KEY` | `http://embed-rerank:8090/v1` / `vietlegal-harrier` | Embedding provider for the OpenViking server (via ov.conf) |

## Backend behavior

- Read: `search_user_memory()` searches the user's memories namespace; identity
  queries ("tôi tên là gì") fall back to a full directory read when the vector
  search misses. Output lands in `SupervisorState.user_memory_context`, the same
  channel Graphiti uses — `query_enricher` and the agents consume it unchanged.
- Write: every turn is committed to the user's session; the server asynchronously
  extracts memories. The agent's "ghi nhớ" tool writes an explicit note file to
  `viking://user/nexusrag_{uuid}/memories/agent_notes/`.
- `both` mode: recall returns OpenViking context first, Graphiti facts appended;
  writes go to both backends independently — a failure in one does
  not fail the other (single-backend mode still raises so the worker retries/DLQs).

## Troubleshooting

- OpenViking disabled: `OPENVIKING_URL` empty or `NEXUSRAG_MEMORY_BACKEND=none` —
  everything behaves exactly as before (Graphiti only).
- Server down: backend logs `[openviking] … failed` warnings; recall falls back
  to Graphiti (`both`) or returns empty (`openviking`); writes are retried by
  the memory worker queue.
- Check the server: `docker logs hrag-openviking`, or from the host:
  `curl http://localhost:1933/health`.
- Browse memories: `ov ls viking://user/nexusrag_<uuid>/memories -R` against the
  published port (root key in `OPENVIKING_API_KEY`).
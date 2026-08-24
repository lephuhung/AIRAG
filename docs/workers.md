# Workers — pipeline & control reference

## Topology

One image, five worker roles selected per-container by `WORKER_TYPE`
(`app/workers/runner.py`). `parse` fans out to **three parallel** RabbitMQ
queues — they are NOT sequential:

```
Upload → parse_worker ─┬─→ embed_worker    (ChromaDB vectors)
        (MinIO,         ├─→ caption_worker  (image captions → re-embed enriched chunks)
         Docling/OCR)   └─→ kg_worker       (LegalKG / LightRAG)
memory_worker (Graphiti personal memory) runs independently of the upload path.
```

Finalization is checked once all three (embed/caption/kg) complete.

| `WORKER_TYPE` | Container | Exchange / queue (`app/queue/connection.py`) | Prefetch env |
|---------------|-----------|----------------------------------------------|--------------|
| `parse` | hrag-worker-parse | `hrag.parse` | `WORKER_PREFETCH_PARSE` |
| `embed` | hrag-worker-embed | `hrag.embed` | `WORKER_PREFETCH_EMBED` |
| `caption` | hrag-worker-caption | `hrag.caption` | `WORKER_PREFETCH_CAPTION` |
| `kg` | hrag-worker-kg | `hrag.kg.{workspace_id}` (per-workspace, ensured on demand) | — |
| `memory` | hrag-worker-memory | `hrag.memory` | `WORKER_PREFETCH_MEMORY` |

Each worker also serves liveness/readiness on **:8081** — `GET /health`,
`GET /ready`.

### Runtime LLM config sync (`config_watch.py`)
LLM models are assignable per task from the admin WebUI (see README "Runtime
LLM config" and `docs/plan-llm-runtime-config.md`). Workers pick up changes
WITHOUT restart: at the start of every message handler, caption/kg/memory/embed
workers call `ensure_fresh_config()` which reads `_config_version` from the
`system_settings` table and refreshes the in-process provider snapshot when it
changed. Cost is one cheap SELECT per message; the check is fail-open (errors
are logged and ignored — a config hiccup never kills a worker or drops a
message). A job already in flight finishes with the previous model; the next
message uses the new one. Embedding/rerank model overrides are the exception —
they apply only on worker/backend restart (GPU preload + vector dimension).

## Control plane — pause/resume WITHOUT docker.sock

Workers subscribe to the `hrag.control` fanout exchange
(`CONTROL_COMMANDS = {pause, resume, restart, set_prefetch}`). The admin UI
drives them via the backend `/workers/*` API (no Docker socket needed):

| Endpoint (prefix `/workers`, JWT) | Action |
|-----------------------------------|--------|
| `GET /workers/overview`, `GET /workers/pipeline`, `GET /workers/queues` | inspect state / queue depths |
| `POST /workers/stop/{type}` · `POST /workers/start` · `POST /workers/restart/{type}` · `POST /workers/restart-all` | lifecycle via control-plane |
| `POST /workers/prefetch/{type}` | live `set_prefetch` |
| `GET /workers/dead-letter` · `POST /workers/dead-letter/{purge,retry}` | DLQ management |
| `POST /workers/retry-failed[/{document_id}]` · `POST /workers/retry-stuck` | requeue |
| `POST /workers/pipeline/{document_id}/cancel` | cancel one doc's pipeline |
| `POST /workers/queues/{name}/purge` · `DELETE /workers/queues/{name}` | queue ops |

**Harness note:** prefer these API calls / `hrag.control` over `docker restart`
for workers — pausing a worker is graceful and does not cold-start anything.
Restarting workers is fine (they hold no GPU model); restarting **vLLM** is not
(see `docs/vllm.md`).

## Reliability traits (don't "fix" these)

- `caption_worker` deliberately avoids `SELECT … FOR UPDATE`; it relies on a
  `captions_done` idempotency flag + an asyncio semaphore to bound concurrent
  caption calls.
- `kg_worker` uses **per-workspace** queues (`hrag.kg.{ws}`), ensured lazily
  (`ensure_kg_queue`); a workspace-poller spawns a consumer per active workspace.
- Retry/backoff uses delay queues `hrag.retry.{N}s` with DLX back to the origin
  exchange — see memory `worker-pipeline-hardening`.
- `raw_chunks_json` lifecycle: parse writes it; embed reads → clears on plain
  path, or keeps chunks + contextual sentence for caption re-embed; finalize
  clears. See `docs/embedding.md`.

## Quick checks

```bash
docker exec hrag-backend curl -s http://localhost:8080/api/v1/workers/overview \
  -H "Authorization: Bearer $TOKEN" | jq
# worker health (from host, if 8081 published) or via `docker exec`:
docker exec hrag-worker-embed curl -s http://localhost:8081/health
```

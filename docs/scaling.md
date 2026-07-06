# Scaling out — multi-worker runbook (post hardware upgrade)

> **When to read this:** you've upgraded the hardware (more GPU VRAM / a second
> GPU / more CPU) and want the backend to serve more concurrent chat + search
> load by running **>1 backend worker process**. Until then the stack runs
> single-process and none of this is active — every flag below defaults to the
> single-process value, so the current deploy behaves exactly as it does today.

## The one-paragraph model

The backend was made horizontally scalable behind an **opt-in Redis layer**
(Phases 0–5). Redis is only needed to coordinate the *small* pieces of state
that must be shared across processes: the **Stop button** (cancel a stream
started on another worker), the **GPU concurrency cap** (so N workers sharing one
GPU don't multiply the VRAM peak N-fold), and the **retrieval cache** (so a query
answered by one worker is cheap on another). Everything else — the SSE relay, the
detached generation run, BM25 — stays **per-process** by design and needs no
coordination. Flip two flags (`REDIS_ENABLED=true`, `WEB_CONCURRENCY=N`), give
the GPU headroom, and the backend fans out.

## The binding constraint: GPU VRAM — now offloaded to a shared service

Historically each backend worker loaded its **own** copy of the retrieval models
(embedder ~4.8GB + cross-encoder reranker) into GPU VRAM, so
`VRAM(N workers) ≈ N × (embedder + reranker) + vLLM engines + slack`. On the
single 48GB GPU the vLLM engines already claim most of it, leaving no room for a
second copy — which capped `WEB_CONCURRENCY` at `1`.

**This is now solved (Phase 1): the embed + rerank models are extracted into a
standalone HTTP microservice.** The compose service `embed-rerank` (container
`hrag-embed-rerank`, port `8090`, GPU 0) runs `uvicorn embed_rerank_service.main:app`
and hosts the SentenceTransformer embedder (`HRAG_EMBEDDING_MODEL`) plus the
CrossEncoder reranker (`HRAG_RERANKER_MODEL`). It exposes `GET /health`,
`POST /embed` (`{texts}` → `{model, dimension, embeddings}`) and `POST /rerank`
(`{query, documents, top_k?, min_score?}` → `{model, results:[{index,score,text}]}`).
Code lives in `backend/embed_rerank_service/`.

The toggle is **`HRAG_EMBED_RERANK_URL`**:

- **Set** (backend default in compose:
  `HRAG_EMBED_RERANK_URL=${HRAG_EMBED_RERANK_URL-http://embed-rerank:8090}`) →
  `EmbeddingService` + `RerankerService` become thin HTTP clients that call the
  service. The backend then holds **no GPU state** on the chat/search hot path, so
  `WEB_CONCURRENCY` is **no longer GPU-bound** — it scales on CPU/RAM alone. Set the
  var empty to disable and fall back to in-process models. `HRAG_EMBED_RERANK_TIMEOUT`
  (default `30.0s`) bounds those HTTP calls.
- The **service itself** runs with `HRAG_EMBED_RERANK_URL` empty (it *is* the model
  host). Its GPU fan-out is bounded by `EMBED_RERANK_CONCURRENCY` (default `1`, set
  on the service container in compose) — a hard ceiling on concurrent GPU batches so
  the rerank fan-out VRAM spike stays flat regardless of how many backend workers
  call in at once.

> **Workers still embed locally (for now).** Phase 1 routes only the **backend**
> chat/search hot path to the service. The `embed` / `caption` / `memory` workers
> continue to load the embedder in-process, so plan GPU budget for those copies.

The alternative — more VRAM / a second GPU to pin retrieval models onto a different
device than the vLLM engines — still works, but is no longer required to go
multi-worker on the backend.

### Whisper STT — CPU sidecar

Speech-to-text was the *other* heavy model the backend loaded in-process. It is now
extracted into the compose service `stt` (container `hrag-stt`, port `8091`,
**CPU-only — no GPU reservation**), which reuses `airag-backend:latest` and runs
`uvicorn stt_service.main:app` to host faster-whisper (CTranslate2, `large-v3`, CPU
`int8`) behind an OpenAI-compatible `POST /v1/audio/transcriptions` (+ `GET /health`).
Code lives in `backend/stt_service/`.

The backend points at it **by default** in compose: `STT_PROVIDER=openai` +
`STT_OPENAI_BASE_URL=http://stt:8091/v1` — the existing `OpenAIWhisperSTTProvider`
client calls the sidecar instead of loading Whisper locally. The sidecar itself runs
`STT_PROVIDER=faster_whisper` (it *is* the model host). STT stays on CPU
deliberately: CTranslate2 needs CUDA 12 but the image ships CUDA 13. No new config
keys — it reuses the existing `STT_*` settings. To revert to in-process transcription
set `STT_PROVIDER=faster_whisper` and clear `STT_OPENAI_BASE_URL`.

With both `embed-rerank` (GPU) and `stt` (CPU sidecar) extracted, the backend request
path holds **no GPU model and no heavy CPU model** — so `WEB_CONCURRENCY` scales on
CPU/RAM alone rather than being pinned to a single process by model VRAM.

CPU/RAM are secondary: BM25 is in-RAM **per process** (each worker rebuilds its
own), so more workers = more RAM for BM25 indexes, but that's rarely the limit.

## Turn-up checklist

Do these in order. Steps 1–4 are one-time; 5 is the flip.

**1. Bring up Redis** (already defined in `docker-compose.services.yml` as the
`redis` service, host port `6380`, `--maxmemory 256mb --maxmemory-policy
allkeys-lru`, no persistence — it holds only ephemeral coordination state).

```bash
docker compose -f docker-compose.services.yml up -d redis
docker exec hrag-redis redis-cli ping   # → PONG
```

**2. Bring up `embed-rerank` and confirm it's healthy.** The backend routes its
embed/rerank to this service **by default** (`HRAG_EMBED_RERANK_URL` is set to
`http://embed-rerank:8090` in compose) and `depends_on` it with
`condition: service_healthy` — so a multi-worker backend won't boot until the
service is up. Without it (var empty) every backend worker loads the models itself
and you're GPU-bound again.

```bash
docker compose -f docker-compose.services.yml up -d embed-rerank
curl -sf http://localhost:8090/health   # → ok
```

**2b. Bring up `stt`** (CPU-only, no GPU gate). The backend routes voice
transcription to it by default (`STT_PROVIDER=openai`,
`STT_OPENAI_BASE_URL=http://stt:8091/v1`) and `depends_on` it with
`condition: service_started`. Whisper on the backend is now optional; if the sidecar
is down, revert to in-process (`STT_PROVIDER=faster_whisper`).

```bash
docker compose -f docker-compose.services.yml up -d stt
curl -sf http://localhost:8091/health   # → ok
```

**3. Run the DB migrations once, single-process.** The schema migrations run
inline in the `app/main.py` lifespan (`CREATE TABLE IF NOT EXISTS` / `ALTER …`).
If N workers boot with `AUTO_CREATE_TABLES=true` they race on the same DDL. So:
boot **once** with 1 worker to apply migrations, then set
`AUTO_CREATE_TABLES=false` for the multi-worker run.

**4. Build the production image.** The dev image runs `uvicorn --reload` (single
process, source bind-mounted). The production `CMD` honours `WEB_CONCURRENCY`:

```dockerfile
CMD ["sh","-c","exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers ${WEB_CONCURRENCY:-1}"]
```

Build with `APP_ENV=production` (or whatever selects the prod stage) so `--reload`
is off — `--reload` and `--workers >1` are mutually exclusive.

**5. Flip the flags** (env / compose):

```bash
REDIS_ENABLED=true
WEB_CONCURRENCY=4          # ← N, sized to GPU headroom (see above)
AUTO_CREATE_TABLES=false   # migrations already applied in step 2
```

On boot `app/main.py` **warns loudly** if you got this wrong — it checks:
`WEB_CONCURRENCY>1` with `REDIS_ENABLED=false` (Stop / GPU cap / cache would be
per-process and wrong), with `AUTO_CREATE_TABLES=true` (migration race), and — if
`HRAG_EMBED_RERANK_URL` is empty — reminds you each worker loads its own retrieval
models (N× VRAM). With the `embed-rerank` service routed the backend holds no GPU
state, so that last warning no longer applies. Watch the logs on first multi-worker
boot.

## What each phase gives you (and its knobs)

| Phase | Shared state | Mechanism | Flag / knob |
|-------|--------------|-----------|-------------|
| **0** | Redis client | Async singleton (`app/core/redis_client.py`), lazy connect, best-effort ping, per-process `INSTANCE_ID` | `REDIS_ENABLED`, `REDIS_URL` |
| **1** | **Stop button** across workers | Cancel pub/sub on channel `stream:cancel`; local fast-path first, else publish so the owning worker cancels its task | `REDIS_ENABLED` |
| **2** | **GPU search cap** cluster-wide | Two-tier: per-process `asyncio.Semaphore` (Tier 1, always on) + Redis ZSET sliding-window w/ Lua, TTL self-heal (Tier 2) | `HRAG_SEARCH_GPU_CONCURRENCY` (per-proc), `HRAG_SEARCH_GPU_GLOBAL_CONCURRENCY` (cluster; `0`=off), `HRAG_SEARCH_GPU_PERMIT_TTL`, `HRAG_SEARCH_GPU_WAIT_TIMEOUT` |
| **3** | **Retrieval cache** shared | Key+TTL (`rag:cache:`) + per-workspace index SET for invalidation; JSON-serialised `DeepRetrievalResult` (dataclass, not pickle) | `REDIS_ENABLED` (falls back to in-process LRU when off) |
| **4a** | Request-safe sessions | `HRAGService` built fresh per request; only the Neo4j driver (LegalKG) stays cached | — (correctness fix, always on) |
| **5** | Multi-worker boot | `WEB_CONCURRENCY` → `uvicorn --workers`; boot-time guard warnings | `WEB_CONCURRENCY` |

**Fail-open is deliberate.** Every Redis path degrades to local behaviour if
Redis is slow or down — a search never wedges waiting for a global permit
(`WAIT_TIMEOUT` then proceeds local-only), cancel still works locally, cache
misses just recompute. A Redis outage degrades *coordination*, it does not take
the API down.

## Sizing the GPU cap

`HRAG_SEARCH_GPU_GLOBAL_CONCURRENCY` is the **cluster-wide** ceiling on
simultaneous GPU-heavy searches (the fan-out + rerank is the VRAM spike, per
memory `search-cuda-oom-silent-fail`). Set it to how many concurrent rerank
passes the GPU can hold **at once**, independent of how many worker processes
exist. `HRAG_SEARCH_GPU_CONCURRENCY` is the per-process soft cap (keep it ≤
global). Rule of thumb: `GLOBAL` ≈ the value that was safe single-process — it
does **not** scale up with worker count, because they share one GPU.

## Verifying it actually coordinates

```bash
# permits currently held (should stay ≤ GLOBAL under load)
docker exec hrag-redis redis-cli zcard gpu:search:permits
# cancel channel has subscribers (one per worker)
docker exec hrag-redis redis-cli pubsub numsub stream:cancel
# cache keys populating
docker exec hrag-redis redis-cli --scan --pattern 'rag:cache:*' | head
# Stop button cross-worker: start a stream, hit Stop from a different tab —
# the /stream/cancel endpoint publishes if the run isn't local. Confirm the
# generation stops (not just the SSE) in the owning worker's logs.
```

## What is NOT yet horizontally scalable (future work)

Raising `WEB_CONCURRENCY` scales the **chat/search API**. It does **not** address:

> **Model-extraction roadmap.** (1) ✅ `embed-rerank` — retrieval models off the
> backend hot path (GPU service). (2) ✅ `stt` — Whisper off the backend (CPU
> sidecar). (3) ⏳ guard the latent in-process GPU flags below in multi-worker mode.
> (4) ⏳ optionally route the LightRAG KG embedding + the ingestion workers through
> `embed-rerank` too.

- **Ingestion workers still embed in-process** — `embed` / `caption` / `memory`
  workers load their own embedder copy; only the backend chat/search hot path is
  routed to the `embed-rerank` service (Phase 1). Routing the workers through it too
  is future work.
- **Latent in-process GPU flags** — `HRAG_OCR_LOCAL=true` and
  `MEMORY_AGENT_LOCAL=true` load an OCR / memory model *inside* each backend worker,
  re-introducing an N× VRAM multiplier that defeats the `embed-rerank` offload. Keep
  them `false` (remote) in multi-worker mode; a boot-time guard for them is future
  work.
- **KG worker per-workspace queues** (`hrag.kg.{ws}`) — the worker fleet is
  scaled via `WORKER_TYPE` replicas + prefetch, not by `WEB_CONCURRENCY`; the
  per-workspace consumer model isn't horizontally sharded across KG workers yet.
- **BM25 per-process RAM** — each worker holds its own index; acceptable, but
  watch RAM as N grows.

## Related

- `docs/vllm.md` — the GPU-VRAM budget these workers share (read first).
- `docs/workers.md` — the *other* scaling axis (`WORKER_TYPE` replicas, prefetch).
- Memory `redis-scale-out` — phase-by-phase implementation notes.
- Memory `search-cuda-oom-silent-fail` — why the GPU search cap exists.
- Memory `vllm-memory GPU util` — the VRAM headroom math.

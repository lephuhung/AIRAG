# vLLM engines — operational reference

> **Harness rule #1: never restart, stop, or `compose down` the vLLM engines.**
> They are GPU-resident, take minutes to cold-start, and are shared by OCR +
> the classifier/memory agent + KG. Restarting them stalls the whole pipeline
> and can OOM on reload. To apply a code/config change, restart **only**
> `hrag-backend` (source is bind-mounted). `.claude/settings.json` hard-denies
> `docker restart|stop hrag-vllm-*` and the whole `docker-compose.vllm.yml`.

## Three LLM endpoints (two local engines + one remote)

The stack talks to **three** OpenAI-compatible servers. Only two are local
containers (in `docker-compose.vllm.yml`, on the shared external network
`airag_hrag_network`); the main answer model is a **remote** box.

| Role | Container / host | URL (from backend) | Model | Restartable? |
|------|------------------|--------------------|-------|--------------|
| **Main answer LLM** | remote `10.10.0.240:8000` | `OPENAI_COMPATIBLE_BASE_URL` / `OLLAMA_HOST` = `http://10.10.0.240:8000/v1` | `Qwen/Qwen3.6-35B-A3B-FP8` | ❌ not ours |
| **Classifier / memory agent / Graphiti / LegalKG** | `hrag-vllm-memory` :8088 | `MEMORY_AGENT_BASE_URL` = `http://vllm-memory:8088/v1` | `qwen-memory` (gemma-4-E4B-it) | ❌ do not |
| **OCR (parse path)** | `hrag-vllm-ocr` :8001 | `HUNYUAN_OCR_API_URL` = `http://vllm-ocr:8001/v1` | `unlimited-ocr` | ❌ do not |

Runtime config (2026-07): `LLM_PROVIDER=openai_compatible`,
`NEXUSRAG_AGENT_BACKEND=langgraph`, `NEXUSRAG_LG_RAG_REACT=true`.

## Engine tuning (why they're fragile)

Both local engines share one GPU with a tight VRAM budget (see memory
`vllm-memory GPU util`). Tuned values baked into `docker-compose.vllm.yml`:

- **`hrag-vllm-ocr`** — `--gpu-memory-utilization 0.28 --max-num-seqs 16
  --max-model-len 12288`, CUDA graphs on (~1.8s/page at 16-way).
- **`hrag-vllm-memory`** — `--gpu-memory-utilization 0.42 --max-num-seqs 32
  --max-model-len 8192`, CUDA graphs on (~63 tok/s). Pushing util to 0.50 OOMs
  the pipeline (forgets the backend/worker/omnivoice slack) — do not "optimize".

## Health checks — how the harness probes vLLM WITHOUT restarting

Read-only. Never restart to "fix" a slow response; check liveness first:

```bash
# memory/classifier engine
curl -s http://localhost:8088/v1/models | jq '.data[].id'
# OCR engine
curl -s http://localhost:8001/v1/models | jq '.data[].id'
# main answer model (remote)
curl -s http://10.10.0.240:8000/v1/models | jq '.data[].id'
# from inside the backend container (service DNS names):
docker exec hrag-backend curl -s http://vllm-memory:8088/v1/models
```

If an engine is down, that is an infra event — surface it, do not auto-restart.
A human restarts with `docker compose -f docker-compose.vllm.yml up -d <svc>`.

## Gotchas

- The main model is **remote and unmanaged** — latency/availability spikes there
  look like agent slowness but are not a local-stack problem.
- `MEMORY_AGENT_BASE_URL` powers the intent classifier, query analyzer, condense
  judge, contextual-embedding enrichment AND Graphiti — one engine, many callers.
  A/B flags that change classifier/analyzer behaviour still hit the same engine;
  no restart needed, only `hrag-backend`.
- vLLM base URLs use **service DNS** (`vllm-memory`, `vllm-ocr`) inside the
  compose network, but **`localhost:8088/8001`** from the host.

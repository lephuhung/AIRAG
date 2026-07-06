# AIRAG Harness

One page for every way we exercise, evaluate and A/B the system. Backend tests and
evals run **inside the `hrag-backend` container** (WORKDIR `/app/backend`), where the
app deps and the reachable providers (vLLM, Chroma, Postgres) already live. The
`Makefile` at the repo root is the single entrypoint — `make help` lists targets.

## Layers

| Layer | What it checks | Needs | Entrypoint |
|-------|----------------|-------|------------|
| **Claude Code harness** | Agent permissions / safety rails | — | `.claude/settings.json` |
| **CI** (offline) | Syntax smoke + dataset validity + FE lint/build | GitHub only | `.github/workflows/ci.yml` |
| **Unit tests** | Legal chunker, validity extractor | container | `make test` |
| **Retrieval golden set** | Recall@k, section-by-Điều, validity layer | Chroma + real corpus | `make test-recall test-section test-validity` |
| **Prompt-eval suite** | Router/analyzer/judge/write prompts (LLM-judge) | live LLM | `make eval-prompts` |
| **RAG evals** | End-to-end answer quality over HTTP | live agent | `make eval-rag`, `make eval-ragas` |
| **Agent A/B** | Two configs on one query set, diffed | live agent | `make ab`, `make ab-compare` |

## Reference docs (read these before touching the stack)

| Doc | Covers | Key rule |
|-----|--------|----------|
| [vllm.md](vllm.md) | 3 LLM endpoints (2 local engines + remote main) | **never restart vLLM** |
| [auth.md](auth.md) | JWT vs API key, login, minting a token | debug-chat needs a **JWT** |
| [workers.md](workers.md) | 5 worker roles, queues, control plane | pause via API, not docker |
| [embedding.md](embedding.md) | in-process embedder/reranker, hybrid retrieval | search is **serial** (GPU semaphore) |
| [scaling.md](scaling.md) | multi-worker / Redis scale-out runbook | `WEB_CONCURRENCY>1` needs `REDIS_ENABLED` + GPU headroom |

## Keeping these docs in sync (REQUIRED)

Architecture changes MUST update the doc that describes them, **in the same
change** — see the "Keeping architecture docs in sync" rule in `CLAUDE.md` for the
full doc→owner map and rules. In short: touch a service/worker/agent-graph/config
flag/port/storage/pipeline/scale-out, and you also touch `README.md`, `CLAUDE.md`,
`AGENTS.md`, and the relevant `docs/*.md` in this table (plus `.env.example` for a
new flag/default). The pre-commit hook (`.claude/settings.json`) reminds you; a
stale doc is treated as a bug.

## Endpoints to test (all under `http://localhost:8080/api/v1`, JWT unless noted)

```
POST /auth/login                         → get access_token  (auth.md)
GET  /auth/me                            → verify token
GET  /workspaces                         → list workspace UUIDs (pick one for A/B)
GET  /rag/stats/{workspace_id}           → chunk/vector counts
POST /rag/debug-chat/{workspace_id}      → full agent answer + retrieved_sources (the A/B probe)
POST /rag/query/{workspace_id}           → retrieval only (no answer)
POST /rag/chat/agent-lg/{ws}/stream      → SSE agent stream (also accepts X-API-Key)
GET  /workers/overview  /workers/pipeline → pipeline & queue state (workers.md)
GET  /config/status                      → provider/config snapshot
GET  /health  /docs                      → liveness + OpenAPI (no auth)
```

Smoke one call:
```bash
API=http://localhost:8080/api/v1
curl -s $API/rag/stats/$WS -H "Authorization: Bearer $TOKEN" | jq
```

## First run

```bash
make up            # start the stack (services compose)
make dev-deps      # install pytest + PyYAML into hrag-backend
make check         # offline parity: unit tests + FE lint
```

## Retrieval & prompt evals

```bash
make test-recall                 # soft-gated vs backend/tests/retrieval/baseline_recall.json
make eval-prompts                # PROMPT_EVAL=1 pytest tests/prompts → JSON report
make compare-prompts A=old.json B=new.json
```

Reports land in `backend/tests/{prompts,retrieval}/reports/` (git-ignored). The
prompt suite is skipped unless `PROMPT_EVAL=1` (handled by the `make` target) so a
bare `pytest` never talks to the models.

**Anti-fabrication guards** (2026-07-05) have a two-layer eval:
- `tests/services/test_grounding_guard.py` — LAYER 1, deterministic (no LLM,
  runs under a bare `pytest`): `_extract_doc_numbers` / `_ungrounded_doc_numbers`
  flag legal doc numbers cited in an answer but absent from the sources (hard
  retract), plus `_ungrounded_article_numbers` for `Điều N` inventions (soft
  caveat only — collision-prone, so recall-limited by design).
- `tests/prompts/test_sufficiency_gate.py` — LAYER 2, live (`PROMPT_EVAL=1`):
  `_judge_sources_sufficient` (memory agent) must call a partial-grounding case
  (penalty question, classification-only sources) INSUFFICIENT while still
  passing questions the sources genuinely cover. Rate-based (N runs, majority).
- `tests/services/test_resolve_confidence_gate.py` — deterministic: the resolve
  vector-fallback gate (`_is_low_confidence_vector_match`) demotes a vector-only
  nearest-neighbor below the medium threshold (30 == 0.30) to not-found, so a
  named document absent from the workspace stops resolving to an unrelated file.

## Agent A/B harness

Compares two backend **arms** on the same query set. An arm = an env configuration;
flip the flag, `docker restart hrag-backend`, run the arm. The report snapshots the
live `NEXUSRAG_*` env so it is self-describing. Reuses the golden-retrieval YAML
schema (`query` + optional `expect_document` / `expect_article` / `negative`).

**Auth + workspace.** `debug-chat` is JWT-gated and `workspace_id` is a **UUID**
(not the numeric id `eval_rag.py` still uses — that script predates auth and is
stale). Export credentials once, and pass the target workspace UUID:

```bash
export AB_USER=admin@hrag.local AB_PASSWORD=...   # or: export AB_TOKEN=<jwt>
WS=3e10d875-...                                   # a workspace UUID with documents

# arm A — current default
make ab ARM=base  QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WS

# flip the flag, then reload backend
#   e.g. set NEXUSRAG_LG_RAG_REACT=true in .env
make restart-backend

# arm B — ReAct executor
make ab ARM=react QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WS

# diff — prints metric deltas + per-case regressions (A hit → B miss) + env delta
make ab-compare A=backend/tests/prompts/reports/ab_base_<ts>.json \
                B=backend/tests/prompts/reports/ab_react_<ts>.json
```

Metrics per arm (all comparable, no judge needed): `latency_ms` (mean/p50/p95),
`source_count_mean`, `article_hit_rate`, `doc_hit_rate`, `refuse_rate_positive`
(lower is better in-corpus), `refuse_rate_negative` (higher is better out-of-corpus).
`ab-compare` exits non-zero if any in-corpus case regressed — usable as a gate.

## Notes / gotchas

- **CI is offline by design.** Backend runtime deps (torch, docling) are too heavy
  and the retrieval/prompt/RAG evals need the live stack — those run via `make`
  against a running backend, not in GitHub Actions.
- **`.gitignore` is selective.** `backend/scripts/*` and `backend/tests/{retrieval,services}`
  are force-tracked, but `pytest.ini`, `tests/conftest.py`, `tests/prompts/*` and
  `requirements-dev.txt` are **not** tracked. CI therefore does not run the prompt
  suite; keep those files in the container image / bind mount. `ab_eval.py` needs
  `git add -f backend/scripts/ab_eval.py` to be tracked.
- **`.claude/settings.json` is local** (the whole `.claude/` dir is git-ignored) —
  it configures this checkout's Claude Code permissions + a pre-commit reminder to
  run GitNexus `detect_changes()`; it is not shared via git.

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

Compares two backend **arms** (`v1` | `v2`) on the same query set. An arm is
selected **server-side only**: the driver (`backend/scripts/ab_eval.py`)
creates a chat session, then posts the arm in the body of the authenticated
superadmin evaluation endpoint (`POST /api/v1/admin/agent/evaluate` →
`{"version": arm}`). Client graph-version headers are never sent — the
driver raises instead of transmitting one. Reuses the golden-retrieval YAML
schema (`query` + optional `expect_document` / `expect_article` / `negative`).

**Auth + workspace.** Export credentials once, and pass the target workspace UUID:

```bash
export AB_USER=admin@hrag.local AB_PASSWORD=...   # or: export AB_TOKEN=<jwt>
WS=3e10d875-...                                   # a workspace UUID with documents

# arm A — v1 default
make ab ARM=v1 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WS

# arm B — v2 (server-side selection; no client header, no backend restart)
make ab ARM=v2 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WS

# optional pinned paths: OUTPUT=<report.json> on `ab`, OUTPUT=<diff.json> on `ab-compare`
make ab ARM=v1 QUERIES=... WORKSPACE=$WS OUTPUT=reports/ab_v1.json

# diff — prints metric deltas + per-case regressions; refuses on evaluator mismatch
make ab-compare A=backend/tests/prompts/reports/ab_v1_<ts>.json \
                B=backend/tests/prompts/reports/ab_v2_<ts>.json
```

### Session-SSE preflight (Phase-3 Task 1)

The golden preflight used by the rollout gates. The driver reads the named SSE
events (`status` / `thinking` / `sources` / `images` / `token` / `complete` /
`error`) to exactly one terminal event per turn and records
`latency_ms` / citations / `status` per case, redacting auth tokens and raw
message text before persisting. `backend/scripts/replay_v2.py` re-judges
recorded transcripts through the same evaluator.

**Shared evaluator version.** Both arms are judged by the SAME preflight
evaluator (`EVALUATOR_VERSION` in `backend/scripts/ab_eval.py`, imported — never
redefined — by `replay_v2.py`). Every report persists `evaluator_version`, and
`ab-compare` REFUSES (exit 2) when the two reports' versions differ instead of
diffing across evaluators.

**Offline unit suite** (no live stack, no LLM) — the harness-executable
deliverable:

```bash
.superpowers/sdd/2026-09-11-langgraph-v2-phase3-rollout/harness.sh \
  'python -m pytest tests/scripts/test_v2_ab_replay.py -q'
```

from `/app/backend`. The `make ab ...` form above is documented for the Compose
stack (needs the live backend + providers + `AB_TOKEN`); it is not run offline.

Metrics per arm (all comparable, no judge needed): `latency_ms` (mean/p50/p95),
`source_count_mean`, `article_hit_rate`, `doc_hit_rate`, `refuse_rate_positive`
(lower is better in-corpus), `refuse_rate_negative` (higher is better out-of-corpus).
`ab-compare` exits non-zero if any in-corpus case regressed — usable as a gate.

## Phase-3 golden preflight (live, operational)

The rollout gate preflight is the `make ab` form above with the golden
retrieval set, one report per arm, then a diff:

```bash
WS=<workspace-uuid-with-documents>
make ab ARM=v1 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WS OUTPUT=backend/tests/reports/v1-preflight.json
make ab ARM=v2 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WS OUTPUT=backend/tests/reports/v2-preflight.json
make ab-compare A=backend/tests/reports/v1-preflight.json B=backend/tests/reports/v2-preflight.json
```

Needs the live Compose stack + providers + `AB_TOKEN` (or `AB_USER` +
`AB_PASSWORD`); the v2 arm additionally needs the canary enabled for the
target workspace (or an authenticated superadmin `--token` drive through
`POST /api/v1/admin/agent/evaluate`). Not runnable offline — record the
reports as gate evidence when run.

## P0 factual-retrieval live gate (v2 `document.retrieve`)

Two authenticated factual probes prove reference-free and hard-scoped v2
queries execute revision-aware retrieval instead of terminating with zero
capability calls. Both need a JWT (`auth.md`: `POST /auth/login`, or the
test-only in-container mint) and a workspace with indexed documents; the v2
arm needs the canary for that workspace (or a superadmin evaluate drive).
**Never restart vLLM or any engine for these probes** (`vllm.md`).

```bash
API=http://localhost:8080/api/v1
TOKEN=$(curl -s -X POST $API/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"'"$AB_USER"'","password":"'"$AB_PASSWORD"'"}' | jq -r .access_token)
WS=<workspace-uuid-with-documents>
DOC=<document-uuid-in-$WS>   # from GET /workspaces + /rag/stats/$WS

# The v2 arm is server-selected: enable the canary for $WS (or drive the
# probes through the authenticated superadmin evaluate endpoint). No
# request field forces v2 — do not send a version field (ChatRequest has
# none; extras are dropped and the turn would silently serve v1).

# 1) unscoped factual query (reference-free; retrieval over workspace scope)
 curl -s -N $API/rag/chat/agent-lg/$WS/stream \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message":"<factual question the corpus covers>"}'

# 2) hard-scoped factual query (document_ids are a hard scope, not candidates)
 curl -s -N $API/rag/chat/agent-lg/$WS/stream \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message":"<factual question>","document_ids":["'"$DOC"'"]}'
```

Required evidence per probe: non-zero `document.retrieve` task /
capability call and `EvidenceUse`; at least one citation on success; every
scoped citation's document ID is a subset of the requested `document_ids`;
no ~100 ms zero-dispatch terminal (a factual complex terminal with zero
capability calls is recorded as `factual_zero_dispatch` and counted as an
error for rollout gates — `app/services/agent/rollout_metrics.py`,
`scripts/collect_v2_rollout_report.py`; typed `denied`/unsupported outcomes
stay typed and are not errors). If credentials or the live stack are
unavailable, record the exact blocker/commands — never fake success.

## Operational hand-off (live steps not runnable from a worktree session)

These require the live Compose stack (`hrag-backend` bind-mounts the main
repo, so a worktree session cannot execute them) and, for canary, real
traffic. Run them from a checkout that owns the stack:

```bash
# Run from the repository root; the migration + gate scripts take one initial
# `cd backend` so later lines never resolve `backend/backend`.
cd backend
# 0) one-time: apply the v2 schema migration against the live DB, then verify
python -m app.services.agents.v2.persistence.migrate apply --dsn <prod-dsn>
python -m app.services.agents.v2.persistence.migrate check --dsn <prod-dsn>
# 1) golden preflight (see above), then ab-compare — must be green
# 2) enable shadow first (no production effect), watch one window:
#    NEXUSRAG_AGENT_V2_SHADOW_ENABLED=true NEXUSRAG_AGENT_V2_SHADOW_PERCENT=5
# 3) staged canary via the superadmin API (each stage needs real
#    agent_rollout_metrics traffic over the gate window):
#    PUT /api/v1/admin/agent/rollout {enabled, canary_percent, workspaces}
#    shadow 5% -> internal-workspace canary -> 5% -> 25% -> 50% -> 100%
#    (100% = v2-eligible traffic only; Write/evaluate stay on v1)
# 4) offline gate over collected metrics:
python scripts/collect_v2_rollout_report.py --dsn <prod-dsn> --out /tmp/v2-rollout.json
python scripts/check_v2_rollout_gate.py --report /tmp/v2-rollout.json
# 5) emergency brake (any stage): PUT /api/v1/admin/agent/rollout {kill_switch: true}
#    -> new requests go to v1, control revision increments, active v2 runs
#    are cancelled without success.
```

`make` targets that need the live stack or the frontend toolchain and cannot
run in the offline harness (`make test-recall`, `make test-section`,
`make test-validity`, `make fe-lint`, `make fe-build`) are likewise
executed here at promotion time; record their output with the gate evidence.

## Offline validation (runnable here)

```bash
H=.superpowers/sdd/2026-09-11-langgraph-v2-phase3-rollout/harness.sh
$H 'python -m pytest tests/agents/v2 tests/api tests/migrations/v2 tests/workers -q --ignore=tests/agents/v2/orchestrator_compat'
# tests/agents/v2/orchestrator_compat hardcodes a localhost:5433 DSN and is
# excluded from the harness run; it is validated from the host bench venv
# instead (cwd backend/):
#   python -m pytest tests/agents/v2/orchestrator_compat -q
# Known outcome (Task 8 evidence): the ONLY failure is the E2 probe
# (hardcoded localhost:5433, unreachable from the bridge-network harness).
```

Static guards (must all pass — empty output — while v1 stays the
default/rollback path):

```bash
! find backend/app/services/agents/v2 -type f \
  \( -name 'people_agent.py' -o -name 'summary_agent.py' -o -name 'comparison_agent.py' \
     -o -name 'document_agent.py' -o -name 'section_agent.py' -o -name 'kg_agent.py' \
     -o -path '*/domain/*_graph.py' \) | grep .
! rg -n 'capability\.execute\(' backend/app/services/agents/v2/tools
! rg -n 'TaskScheduler|scheduler\.execute\(|checkpointer' backend/app/services/agents/v2/tools
! rg -n 'safe_metadata|Mapping\[str' backend/app/services/agents/v2/tools
```

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

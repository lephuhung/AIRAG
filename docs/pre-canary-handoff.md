# Pre-canary handoff — LangGraph v2 (Task 3)

Ownership: this runbook is the executable checklist for promoting the
`feat/langgraph-v2` stack from a worktree session to a live canary. Stable
entrypoints stay in `docs/harness.md` (this file links there instead of
duplicating them). Status of every item below was verified from the Task-3
worktree line (`fe241c8` plus the fix-round commit; exact HEAD in
`task-3-report.md`) unless marked **[NOT RUN — operational]**.

## 0. Verdict: checkpoint dependency

`ModuleNotFoundError: No module named 'langgraph.checkpoint.postgres'` on
worktree hosts is **an uninstalled declared requirement, not declaration
drift**: `backend/requirements.txt` pins `langgraph-checkpoint-postgres==2.0.25`
and `psycopg[binary]==3.2.3`; `backend/requirements-v2-benchmark.txt` agrees
on the postgres pin. Guard: `backend/tests/agents/v2/persistence/test_checkpoint_readiness.py`
(2 passed = pins declared; 2 skipped = extra absent here with the install
command). Do not churn pins; do not add import fallbacks.

Read-only container evidence (fix round I1, no service action — `hrag-backend`
was already `Up (healthy)`; only `docker exec` imports and `pip show` ran,
no writes, no restarts):

```
docker exec hrag-backend python -c "from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver"
→ IMPORT OK <class 'langgraph.checkpoint.postgres.aio.AsyncPostgresSaver'>
docker exec hrag-backend python -c "…assert callable(AsyncPostgresSaver.from_conn_string) and callable(AsyncPostgresSaver.setup)"
→ SURFACE OK from_conn_string+setup
docker exec hrag-backend pip show langgraph langgraph-checkpoint langgraph-checkpoint-postgres psycopg
→ langgraph 1.1.10, langgraph-checkpoint 2.1.2,
  langgraph-checkpoint-postgres 2.0.25, psycopg(-binary) 3.2.3
```

So the project image is dependency-complete for the exact declared pins.
Accurate boundary: image readiness ≠ ability to run worktree suites there —
the container bind-mounts the deploy snapshot
`/home/AIRAG/.deploy/langgraph-v2-f766bda/backend → /app/backend` (not this
worktree, not the main repo), and its database is live, so worktree
persistence suites and any live-DB writes stay operator steps (§2).

## 1. Install / verify (dependency-complete environment)

Run inside the backend container image or the bench venv (NOT the bare
worktree host):

```bash
pip install -r backend/requirements.txt
pip check  # must report no broken requirements (postgres pin vs langgraph major)
python - <<'EOF'
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
assert callable(AsyncPostgresSaver.from_conn_string) and callable(AsyncPostgresSaver.setup)
from app.services.agents.v2.persistence.checkpoint import (
    create_v2_checkpointer, check_v2_checkpointer, setup_v2_checkpointer, CHECKPOINT_TABLES)
print("checkpoint import probe OK:", sorted(CHECKPOINT_TABLES))
EOF
cd backend && python -m pytest tests/agents/v2/persistence/test_checkpoint_readiness.py -q
# Expected: 4 passed, 0 skipped.
```

## 2. Checkpoint setup probe (live DB, operator step)

```bash
cd backend
python -m app.services.agents.v2.persistence.migrate apply --dsn <prod-dsn>
python -m app.services.agents.v2.persistence.migrate check --dsn <prod-dsn>
python -m pytest tests/agents/v2/persistence -q
python -m pytest tests/agents/v2/orchestrator_compat -q  # host bench venv per harness.md
```

## 3. Persistence / orchestrator suites (worktree-runnable subset)

```bash
cd backend
python -m pytest tests/agents/v2 -q -p no:cacheprovider --ignore=tests/agents/v2/persistence \
  --deselect "tests/agents/v2/orchestrator_compat/test_compatibility_probe.py::test_probe_current_interpreter_frozen_api[postgresql://postgres:postgres@localhost:5433/hrag_test]"
python -m pytest tests/api -q -p no:cacheprovider -k "chat or history or session or ingress or conversation"
```

## 4. Frontend / static verification

```bash
cd frontend && npm run build && npx vitest run && npx tsc -p e2e/tsconfig.e2e.json --noEmit
# From the repository root (NOT / — there is no frontend/ or playwright install there):
cd frontend && npx playwright test --config e2e/playwright.config.ts   # needs `npx playwright install`
```

Static guards (must all pass — empty output) — see `docs/harness.md`
"Static guards" for the exact four commands (no domain agents/subgraphs;
tools/ never dispatches capabilities, schedulers, checkpointers, or
metadata mappings).

## 5. Live SSE / browser canary scenarios (operational)

Drive through the superadmin evaluate endpoint or the session chat with the
canary enabled for one internal workspace
(`PUT /api/v1/admin/agent/rollout {enabled, canary_percent, workspaces}`).
Per scenario assert the user-visible outcome from
`docs/public-sse-event-inventory.md`:

1. Factual retrieval turn → `token*` → `complete{status:"success"}`; citations rendered; history reload shows persisted `citations`.
2. Ambiguous follow-up (`văn bản này` over ≥2 prior docs) → `clarification_required` with server-issued options → resume posts `{clarification_id, selected_option_id}` only → `clarification_resolved` → answer.
3. Ordinal recovery (`file thứ hai`) → single pinned revision (binder only).
4. Cancel mid-stream → `POST …/stream/cancel` → quiet terminal, clean next turn (mirrors `frontend/e2e/chat-stream.e2e.ts` once browsers exist).
5. Out-of-scope history identity + anaphora → generic typed error, no UUID/title in any user-facing text.

## 6. Expected telemetry / checkpoint / resume signals

- `agent_rollout_metrics` rows per terminal turn (v1 + v2 arms; `security_unobservable` sentinel rows counted invalid, never dropped).
- Terminal `complete` checkpoints carry the pinned `DocumentBindingSet` (immutable `document_revision` UUIDs) + retention leases acquired pre-checkpoint; terminal lease release post-checkpoint by the outer runner.
- Suspended clarification turns checkpoint the persisted `ClarificationRequest`; resume passes `resume_clarification` output through verbatim with graph-owned navigation.
- Offline gate: `scripts/collect_v2_rollout_report.py` → `scripts/check_v2_rollout_gate.py` must pass over real gate-window traffic per stage.

## 6A. Grounded-LLM synthesis signals (spec §20)

The v2 `synthesize` node is now the bounded grounded-LLM subgraph
(`agents/v2/synthesis/`; canonical ownership in `CLAUDE.md` → "V2 grounded
answer synthesis"). Per-stage checks on top of §5–§6:

- **Synthesis failure.** Invalid model output or failed support/citation
  validation after the single repair fails closed with `synthesis_failed`:
  zero tokens, exactly one typed `error`, and the safe Vietnamese message
  persisted as nonblank assistant content (reload shows it).
- **Privacy tracing.** Synthesis calls run under `synthesis_llm`
  (`ContentSuppressedLLMProvider`): Langfuse generations and the dataset
  trace collector carry allowlisted operational metadata only — never
  query/evidence/prompt/answer text. Verify a traced turn shows metadata,
  not content.
- **Target coverage.** A multi-target (compare) question must cite evidence
  from every required target; a required target starved to zero under budget
  fails closed (`selection_missing_target`), never a silent one-sided answer.
- **Checkpoint compat + latency.** Pre-change v2 checkpoints normalize to
  `synthesis=None`; current checkpoints missing the key fail closed. The
  deliberate `prepared` + `attempt_reserved` checkpoint barriers before the
  first provider call add bounded latency — accepted for restart safety.
- **Summarize bypass guard.** A summarize map/reduce turn makes zero
  provider calls inside `summarize_reduce_node`; the final summary flows
  through the same outer synthesis state machine (one owner, ≤2 calls total).
- **Citation ordering.** On success: `status(generating)` → one `citation`
  frame → first `token`; `complete` repeats the identical citation identity
  set; markers are `[a3z9]`-style, 1–3 per claim, no references list.
- **People/KG exclusions.** People answers still render the card path with
  zero synthesis-model calls; KG-only claims terminate typed unavailable —
  never a fake document citation.
- **Rollback checks.** Kill switch and v1 default are unchanged; a synthesis
  regression at any stage → `kill_switch: true` (§7) — no deploy needed.

## 7. Abort / rollback criteria

- Any stage: `PUT /api/v1/admin/agent/rollout {kill_switch: true}` → all new requests to v1, control revision increments, active v2 runs cancelled without success.
- Abort the promotion (stay on v1) on: gate failure over the window; unobservable-verdict rate above threshold; any UUID/title/PII in user-facing clarification/error text; any binding pinned outside the requesting workspace scope; `evaluate`/Write traffic reaching v2 execution (must fall back to v1).
- Rollout order: `shadow 5% → internal workspace canary → 5% → 25% → 50% → 100%` (of v2-eligible traffic only).

## 8. Explicitly NOT run from the worktree session

- `pip install` / any dependency or system-package installation.
- Docker/Compose start/restart, Postgres start, backend/vLLM (re)start.
- Live migration `apply`, live golden preflight, A/B harness, live canary, metrics collection/gating.
- Browser E2E execution (`chromium_headless_shell-1217` absent; `npx playwright install` needs a download).
- `tests/agents/v2/persistence/test_checkpoint.py` live-DB tests (need the postgres extra + disposable DB).
- `make` live-stack targets (`test-recall`, `test-section`, `test-validity`, `fe-lint`, `fe-build`) — run at promotion time per `docs/harness.md`.

# Task 7 Report — Lazy v1/v2 Runtime Selector + Admin Evaluation Surface + Entrypoint Wiring

## Status: DONE

- Base: `3a7477b`; branch `feat/langgraph-v2`; worktree `/home/AIRAG/.worktrees/langgraph-v2`.
- Commit: `feat: add v1 v2 runtime selector` (ids in result header).
- Files (brief list + 2 force-added test files + this report):
  - Created `backend/app/services/agent/runtime_selector.py` (~700 lines: version
    normalization, `V2NotReadyError`, `require_v2_schema_ready`,
    `resolve_agent_graph` lazy resolver, `configured_agent_version`,
    `resolve_request_version` admin-only override, `resolve_runtime_scope`
    intersection, `persist_raw_user_message`, v2 ingress services
    `ChatMessagesService` / `V2AuthorizationService` /
    `GovernorEvidenceBuilder` / `PlanBindingResolver`, `make_v1_preprocess_closure`,
    `make_abbreviation_lookup`, `V2Ingress` + `build_v2_ingress` async CM,
    `run_v2_turn_sse` interim runner, `clarification_reply_is_fresh_turn`,
    `terminal_state_is_error`, `undispatched_tasks` recipe).
  - Created `backend/app/api/agent_admin.py` (`GET /admin/agent/status`,
    `POST /admin/agent/evaluate` + `run_admin_evaluation`; router-level
    `require_superadmin` dependency).
  - Modified `backend/app/api/router.py` (+2, registers `agent_admin_router`).
  - Modified `backend/app/core/config.py` (+9: `NEXUSRAG_AGENT_GRAPH_VERSION`
    default `"v1"` + fail-fast membership check in the `Settings` validator).
  - Modified `.env.example` (+8, documented the new variable).
  - Modified `backend/app/api/chat_agent_lg.py` (raw persist before expansion,
    scope ∩, resolver dispatch, v1 path byte-identical, v2 via
    `_stream_v2_standalone`).
  - Modified `backend/app/api/chat_session.py` (raw persist via helper, scope ∩,
    resolver dispatch, v2 ingress with evidence commit/rollback + session close).
  - Modified `backend/app/services/integrations/telegram_service.py` (active
    workspace now intersected with accessible — revoked actives fail closed;
    raw persist via helper; resolver dispatch; v2 via `_collect_v2_telegram_events`).
  - Created `backend/tests/api/test_agent_runtime_selector.py` (21 tests) and
    `backend/tests/api/test_agent_v2_ingress.py` (22 tests), both `git add -f`
    (`backend/tests/` is gitignored).
- The dirty `docs/.../plans/...md` hunk is the orchestrator's concurrent
  amendment — NOT staged or touched here (same as T6 rounds).

## TDD evidence

1. Wrote both test files first; harness run failed at collection with
   `ModuleNotFoundError: app.services.agent.runtime_selector` (red phase).
2. Implemented selector → admin → entrypoints; iterated green. Deliberate
   test-or-contract fixes during green (no weakened assertions):
   - Subprocess laziness probe used a host-side cwd (`FileNotFoundError` in
     the container) → `_backend_root()` fallback to live cwd.
   - `PreprocessingResult`/`FinalResponse` fakes used wrong fields
     (extra-forbid) → corrected to real shapes (`original_query`,
     `preprocessing_status`, `preprocessor_trace`; dropped
     `suggested_followups`).
   - Session raw-persist pin rewritten to the helper call form
     (`raw_text=request.message`).
   - Laziness probe initially forbade transitive supervisor *imports*, but
     the pre-existing `app.services.agent.__init__` re-exports the v1 getter,
     so any submodule import loads the module. Re-pinned to the brief's
     actual contract — no graph CONSTRUCTION at import (both singletons stay
     `None`) — plus the AST pin that all supervisor imports in the selector
     are function-local.
3. Final: new files **43 passed**; regressions **797 passed, 1 failed**
   (pre-existing E5 `localhost:5433` probe, files untouched).

## Commands run (via `harness.sh`, from repo root)

- `harness.sh 'python -m pytest tests/api/test_agent_runtime_selector.py tests/api/test_agent_v2_ingress.py -q'` → **43 passed**.
- `harness.sh 'python -m pytest tests/api tests/agents/v2 -q'` → **797 passed, 1 failed** (E5 probe, pre-existing on base).
- `harness.sh 'python -c "import app.api.router, ...; ..."'` → `imports-ok`.
- `git diff --stat`, caller-grep before editing existing symbols (GitNexus
  absent per E4), `git diff --cached --check` clean before commit.

## Controller-ruling compliance

- **Default v1, fail-fast config**: `NEXUSRAG_AGENT_GRAPH_VERSION` defaults
  `v1`; `normalize_agent_version` + `Settings` validator reject anything but
  v1|v2 (case-insensitive).
- **Lazy factories ONLY**: AST test pins all supervisor imports function-local;
  subprocess test pins both singletons unbuilt after selector import.
  `resolve_agent_graph("v1")` imports `get_supervisor_graph` lazily;
  `"v2"` awaits `require_v2_schema_ready()` (Phase-1 `check_v2_schema` via
  `make_engine(DATABASE_URL)` on a worker thread + read-only
  `check_v2_checkpointer`) BEFORE touching the v2 getter; unready →
  `V2NotReadyError`, never silent v1 fallback; missing lifespan singleton →
  `V2NotReadyError` (translated from `SupervisorV2Error`).
- **Ingress**: raw text persisted before expansion/preprocessing on all three
  conversational entrypoints (standalone reordered — it previously persisted
  the *expanded* text); scope = authenticated ∩ requested everywhere
  (Telegram's trusted-active gap closed: revoked actives now fail closed);
  no header parameter exists on any selection function (signature-pinned);
  admin override is the only per-request override, 401 anonymous / 403
  non-admin / 400 bad value.
- **T6 hand-off**: per-request `RuntimeServices` built via T6's
  `build_runtime_services`/`build_graph_runtime_context` (no second
  definition — single-definition test pins `v2/contracts/state.py`);
  registry via `build_v2_capability_registry` with `V1ServiceBundle` holding
  real adapters (`DeterministicSemanticAdapter` + v1 preprocess closure,
  `V1BindingResolver` over `resolve_document_bindings`, governor-backed
  `EvidenceBuilder`, `GovernorEvidenceHydrator`, chat/authorization services);
  `PlanBindingResolver` is the shipped 15-line pure resolver (unfed → `None`
  → read capabilities fail closed); lease repo owns a DEDICATED session
  (separate `async_session_maker()` sessions for leases vs evidence; close
  order pinned); fresh `AnswerDraftChannel` per turn (truncation stays
  runtime-only + `undispatched_tasks` recipe re-exported); `V1ServiceUnavailable`
  on sync-lookup-inside-loop; `ClarificationUnsatisfiable` →
  `clarification_reply_is_fresh_turn` contract; `terminal_state_is_error`
  marks non-success placeholder terminals as terminal-errored, never validated.
- **Entrypoints**: all three call `resolve_agent_graph` (source-pinned, no
  `get_supervisor_graph(` / `get_supervisor_v2_graph(` calls remain); v1
  streaming bodies byte-identical.

## Residual risks / T8 prerequisites

- `PlanBindingResolver` is constructed UNFED at ingress (no plan exists yet);
  first-turn `document.read`/`section.read` dispatches resolve to `None` and
  fail closed as `CapabilityUnavailable`. T8 must feed it from the latest
  checkpointed plan/bindings before each resume-invoke; wiring the feed into
  the in-run execute path (scheduler/`execute_node`) is flagged as follow-up
  work outside T7's file list.
- No pinned-revision content reader is wired (`V1DocumentContentReader` /
  `V1SectionContentReader` seams raise until revision-exact readers exist);
  evidence writes fail closed without `EVIDENCE_ENCRYPTION_KEYS` configured
  (both by design — typed `dependency_error`, never guesses).
- `run_v2_turn_sse` is the interim T7 turn runner (single `ainvoke` +
  terminal mapping); T8 owns the production streaming adapter and terminal
  lease release (`release_run`), which this task never performs.
- `GovernorEvidenceBuilder` people path re-minimizes through public
  `persist_people_evidence` with the minimized field set (deterministic,
  converges); structural `validate_evidence_use` only — plan/target
  resolution stays with hydration.
- Pre-existing E5 probe failure (`localhost:5433`) is environmental,
  unrelated, unchanged.

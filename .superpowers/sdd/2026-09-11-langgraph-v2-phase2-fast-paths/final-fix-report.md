# Final-review fix wave report — LangGraph v2 Phase 2 (`cae946e` + 1 commit)

Single-commit fix wave applying the actionable findings of
`task-final-review.md` (reviewer verdict: spec PASS, quality PASS WITH
FINDINGS — 0 Critical, 1 Important F1, 1 Phase-3-Important F3, 8 Minor).

TDD was followed: the 5 new tests below were added first and all failed in
the exact predicted manner (F1 `DocumentAdapterError` before DB access,
F4 clarify → `502`, F4 terminal → leases never released, F2/F8 source
pins), then the fixes turned them green.

## Implemented

### F1 (Important) — production binding role policy
- **Fix chosen:** role policy in the wiring (not a semantic-adapter
  change). `build_v2_ingress` now constructs
  `V1BindingResolver(session_factory=session_factory, default_role="target")`
  (`backend/app/services/agent/runtime_selector.py`). The semantic adapter
  still emits `requested_role=None` for every user reference (contract
  untouched); the resolver supplies the `"target"` default so a role-less
  resolved reference pins instead of raising `DocumentAdapterError`
  before any DB access.
- **Test:** `test_production_binding_resolver_pins_role_less_reference`
  (`backend/tests/api/test_agent_v2_ingress.py`) pins a role-less
  `resolved` reference through the PRODUCTION resolver wiring — the
  resolver instance comes from `build_v2_ingress` services, not a
  hand-built fake — with `document_views` monkeypatched at the DB
  boundary. Asserts binding id `b_r1`, the revision identity, and role
  `"target"`.

### F2 (Minor) — dead scope read on `/agent-lg/stream`
- **Fix chosen:** REMOVED (not implemented). `ChatRequest`
  (`backend/app/schemas/rag.py:233`) has no `workspace_ids` field, so both
  reads were dead code: the `getattr(request, "workspace_ids", None)`
  intersection inside `langgraph_chat_stream` and the
  `if hasattr(request, "workspace_ids") …` override in the
  `chat_stream_langgraph` endpoint (`backend/app/api/chat_agent_lg.py`).
  The authenticated scope (`_get_accessible_workspaces_lg`) remains the
  sole source of truth; the generator now passes `requested_ids=None`
  explicitly with a comment. No ACL behavior changes (the old code always
  resolved to the authenticated scope anyway).
- **Test:** `test_standalone_stream_has_no_dead_scope_override` pins the
  absence of both dead reads.

### F4 (Minor) — admin evaluation surface bypasses the reviewed adapter
- **Fix:** `_run_v2_eval` (`backend/app/api/agent_admin.py`) no longer
  `ainvoke`s the graph directly. It now collects
  `stream_v2_turn_events(graph, runtime_context, thread_id,
  initial_state, plan_resolver)` — the reviewed adapter with
  `_v2_suspend_request` detection, the one-terminal rule, and terminal
  lease release — then commits the evidence unit of work (rolls back when
  the stream itself raises, mirroring `chat_agent_lg`). Consequences:
  a top-level suspend (RETURNED `__interrupt__`, never raised) surfaces
  as the clarify `complete` instead of `502`, and terminal turns release
  the run's leases instead of lingering to TTL.
- **Tests:** `test_admin_v2_eval_surfaces_clarify_without_502`
  (interrupt-return + `clarify_wait` checkpoint → single `complete` with
  the question, leases untouched, evidence committed) and
  `test_admin_v2_eval_releases_leases_on_terminal` (success → single
  `complete`, `release_run(run_id, "terminal")` recorded).

### F8 (Minor) — dead `run_v2_turn_sse`
- **Fix:** removed `run_v2_turn_sse` from
  `backend/app/services/agent/runtime_selector.py` (it was the file's
  last symbol; no other code referenced it — verified by grep, only
  stale `.pyc` caches, comments, and tests mention the name). No
  unreferenced helpers remained with it (its imports were function-local;
  `AsyncIterator` is still used by `build_v2_ingress`).
- **Test:** `test_dead_v2_turn_runner_removed` asserts the symbol is gone.

## Verification
- `harness.sh 'python -m pytest tests/agents/v2 tests/api -q
  --ignore=tests/agents/v2/orchestrator_compat'` → **805 passed**
  (final-review Gate A baseline was 787 on a narrower file set; the
  delta is the wider `tests/api` dir plus the 5 new tests — zero
  failures).
- Frontend rollback suite via direct vitest invocation
  (`frontend/node_modules/.bin/vitest run
  src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx`)
  → **7 passed**. (The brief's literal `pnpm test -- ChatPanel.rollback`
  arg-forwarding is broken in this environment — `test: '--': unary
  operator expected` from the pnpm shim — so the equivalent direct
  vitest run was used; no frontend code was touched.)
- Pre-commit scope check: `git diff --stat` shows exactly the 4 intended
  files; the stray `pnpm-lock.yaml` my failed `pnpm test` probe created
  was deleted. GitNexus `impact`/`detect-changes` could not be run —
  `.gitnexus/` does not exist in this worktree and no GitNexus tool is
  available to this session (consistent with the final review's E4/E7
  ruling that GitNexus is absent); the manual equivalent (full-diff
  review + repo-wide grep for dangling references) found no out-of-scope
  changes.

## RECORD ONLY (not implemented, per brief)

- **F3 (Important for Phase 3, not reachable in Phase 2):** ingress mints
  a fresh `run_id` per request while resume/hydration are run-scoped;
  Phase-3 complex-research resume will deny pre-interrupt uses as
  `cross_run` and mint shadow lease rows. Phase-3 entry requirement:
  reuse the checkpointed run id across resume or re-create uses under the
  new run before hydration.
- **F5:** v1 entrypoint deltas — raw user text persisted in
  `chat_agent_lg` (was abbreviation-expanded text; display-only, aligns
  with the raw-authority rule); Telegram now intersects the linked
  workspace with accessible workspaces (narrowing, fail-closed). For the
  ledger.
- **F6** (over-broad lease anchoring: scheduler leases every fresh use
  against the cross product of use × every pinned revision of the task),
  **F7** (finalizer re-synthesizes when a draft exists but grounding
  failed — deterministic, but two synthesize+ground passes),
  **F9** (routing consults permission, not service availability →
  `DEPENDENCY_UNAVAILABLE` instead of catalog-level denial),
  **F10** (checkpoint-state hygiene: stale pins accumulate in
  `state["bindings"]`; `runtime_dependency` reason code is checkpointed
  business state) — Phase-3/ledger items.
- Residual risks carried forward unchanged from the final review: R1
  (durability window at dispatch — framework-level, recovery re-plans
  deterministically), no live end-to-end composition harness, D3
  LLM-disambiguation determinism under load.

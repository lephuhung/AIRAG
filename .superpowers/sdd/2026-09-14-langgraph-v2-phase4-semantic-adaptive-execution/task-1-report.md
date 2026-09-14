# Task 1 Report — Freeze a route-intent parity corpus from V1

- BASE: `8df0d3d` (recorded before edits via `git rev-parse HEAD`)
- HEAD: `4576153` (`test(v2): capture v1 route intent parity`)
- Review package: `.superpowers/sdd/2026-09-14-langgraph-v2-phase4-semantic-adaptive-execution/review-8df0d3d..19f41ff.diff`
  (manual `git diff BASE..HEAD`; `scripts/review-package` does not exist in this worktree — same as Task 0.
  `19f41ff` tree ≡ `4576153` tree minus the diff file itself, so the package is representative.)

## Changed files (new only, force-added: `backend/tests/` is gitignored, existing tests are force-tracked)

- `backend/tests/agents/v2/golden/intent_cases.py` — frozen 10-case corpus: `v1_intent` (semantic intent, never
  `next_agent`), `v1_scope` + `v1_intent_source` provenance, flags, and separately recorded Phase 4A v2 targets
  (`QueryAnalysis` work_type/domains + route/reason; `v2_reason_code=None` where Task 4 assigns the frozen code).
- `backend/tests/agents/v2/golden/test_v1_intent_parity.py` — 6 characterization tests (absolute import per repo
  convention; no `__init__.py` added since sibling test dirs are namespace packages).

## TDD: RED → GREEN

- RED: `python3 -m pytest tests/agents/v2/golden/test_v1_intent_parity.py -q` → collection `ImportError`
  (fixtures module did not exist). A second RED iteration: relative import failed (no parent package) → switched to
  absolute `tests.agents.v2.golden...` import; model-case assertion generalized from `scope == "full"` to the recorded
  scope (`rag_named_doc` for the summarize case).
- GREEN: same command → `6 passed in 0.35s`.

## Test results

- Focused: `tests/agents/v2/golden/test_v1_intent_parity.py` → 6 passed.
- Regression: `tests/agents/v2/contracts` + `tests/agents/v2/fast_paths` → 270 passed, 1 skipped.
- Wider: `tests/agents/v2 --ignore=orchestrator_compat --ignore=persistence` → 749 passed, 3 skipped.
- `tests/agents/v2/persistence/test_checkpoint.py` does NOT collect: pre-existing blocker
  `ModuleNotFoundError: No module named 'langgraph.checkpoint.postgres'` (known baseline gap, unrelated — new files
  cannot affect that import; orchestrator_compat excluded per `docs/harness.md` default).

## Current-v1 probe (grounding, run before writing fixtures)

`classify_supervisor_scope` → greeting / people(phone, cccd, name) / rag_named_doc / full exactly as recorded;
`deterministic_decision_for_scope` resolves only the 4 deterministic cases. Current v2 `analyze_query` still
misroutes greeting-prefixed factual → `explain`/`memory`, `090… là ai?`/`Nguyễn Văn A là ai?` → `lookup`/`kg`,
bare general comparison → `compare` — the divergences Tasks 3/4 must fix (targets recorded, not asserted).

## GitNexus impact / detect_changes

- No existing function/class/method edited (2 new test-only files) → no upstream-impact run required; supervisor
  progress update sent before edits (risk LOW, no callers/processes affected).
- `node .gitnexus/run.cjs detect-changes --scope all -r …` → "No changes detected" (no existing symbols/flows
  touched; new files postdate the index). Verified via `git status`: only the 2 staged new files.

## Commit

- `test(v2): capture v1 route intent parity` (exact subject; force-add needed due to `backend/tests/` gitignore).

## Self-review

- Corpus covers all 10 required brief cases verbatim (incl. `Tóm tắt Nghị định A` placeholder form); `REQUIRED_IDS`
  gate enforces coverage.
- Invariants held: semantic/model output advisory only (fixtures assert nothing about live models, no network);
  v2 owns policy (v2 targets are shape-checked frozen contracts, current output not asserted); no secrets/PII
  (phone/CCCD digits are already-public fixture patterns from v1 prompts); v1 untouched/default.
- One provider factory / no resolver clones / no direct capability execution: N/A — test-only change, zero
  production code.
- `v1_prerequisite: resolve_doc` on the summarize case is intent metadata, not a task plan; `next_agent` /
  `pending_intent` / `task_plan` keys banned by test.

## Blockers / concerns

- **PLAN-OWNER RULING REQUIRED before Tasks 3/4 (from review round 1, I1):** the frozen `RouteReason`
  (`backend/app/services/agents/v2/contracts/routing.py:26-48`) and `_FAST_CAPABILITY_FOR_REASON`
  (`backend/app/services/agents/v2/nodes/fast_plan.py:57-62`) have NO code for the recorded targetless
  `document.retrieve` fast path. The corpus now marks the 3 affected cases
  (`greeting-prefix-factual`, `rag-general-thai-san`, `rag-general-compare`) with
  `v2_reason_pending_contract_extension: True` + named proposal `targetless_document_retrieval` (test-enforced:
  every case has exactly one of code/marker). Either extend the frozen `RouteReason` vocabulary (Task 4 scope
  decision) or retarget those cases before Tasks 3/4 assert this corpus — `task-4-brief.md` currently never
  mentions the extension.

- Pre-existing: `langgraph-checkpoint-postgres` missing in this image (persistence tests uncollectable); pytest
  itself IS available (9.1.1), so the brief's fallback did not trigger.
- `detect-changes` index appears stale for brand-new files; manual `git status`/`git diff` used as the backstop.
- `people-name-is-ai` (`Nguyễn Văn A là ai?` → `mongo_search_name`) is a model-side expectation (scope `full`);
  if the future intent adapter's model disagrees, this fixture is the tiebreaker to revisit deliberately.

## Review round 1 fixes (FIX_BASE `4576153`)

- I1: `v2_route` now asserted for all 10 cases (`in get_args(Route)`); the 3 targetless-retrieve cases carry
  `v2_reason_pending_contract_extension: True` + `v2_reason_proposal: "targetless_document_retrieval"`; new test
  requires exactly one of code/marker per case; docstring aligned; `RouteReason` extension raised as plan-owner
  blocker above.
- M1: `EXPECTED_QUERIES` id→query pin in the test file catches query drift (mutation-verified).
- M2: `INTENT_SOURCES` vocabulary test + exactly-one-bucket membership (mutation-verified).
- M3: summarize case relabeled `derived-from-v1-prerequisite-rewrite` (verified: `supervisor.py:1539-1549`
  rewrites emitted intent to `resolve_doc`); test renamed `test_non_deterministic_cases_match_recorded_scope`.
- M4: `validate_query_analysis` (canonical gate) used; tautological asserts removed; duplicate weak greeting test
  removed (covered by scope test + query pin).
- M5: HEAD filled in above; stale committed `review-8df0d3d..19f41ff.diff` untracked (deleted) per reviewer.
- M6 deferred: no `__init__.py` added — namespace-package absolute imports work under the documented harness
  invocation (`cd backend && python -m pytest tests/...`); bare-`pytest` failure is pre-existing (also affects
  `contracts/test_contract_base.py`), not a regression.

RED/GREEN: added marker/vocab/pin tests first → RED (`test_v2_targets_are_valid_frozen_contracts` failed on the
3 unmarked cases, `assert False != False`); after fixture markers → GREEN `6 passed in 0.40s`. In-memory mutation
probes (bad `v2_route`, dropped marker, query drift, unknown provenance) all caught with case-id messages.
Regression: contracts+fast_paths `270 passed, 1 skipped`; wider v2 (minus orchestrator_compat/persistence)
`749 passed, 3 skipped`; persistence still uncollectable (pre-existing missing `langgraph-checkpoint-postgres`).

- FIX HEAD: `857ab83` (amended to fold staged test/report edits into the single fix commit; subject exact
  per brief).
- New review package (untracked, `.superpowers/` is gitignored):
  `.superpowers/sdd/2026-09-14-langgraph-v2-phase4-semantic-adaptive-execution/review-fix-4576153.diff`
  (manual `git diff 4576153..857ab83`; regenerated after the final amend so it is byte-exact).

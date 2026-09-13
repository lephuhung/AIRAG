# LangGraph v2 Phase-3 — Handoff (remaining tasks)

> **Status:** Phase 3 in progress. Tasks 0–5 complete and committed. Task 6 (shadow) mid-fix-loop.
> Tasks 7A, 7B, 8 not started.
> **Branch:** `feat/langgraph-v2` · **Head at handoff:** `7c3c493` · **Worktree:** `/home/AIRAG/.worktrees/langgraph-v2`
> **Plan:** `docs/superpowers/plans/2026-09-11-langgraph-v2-phase3-rollout.md`
> **Spec (binding authority):** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`
> **Normative amendment:** `docs/superpowers/plans/2026-09-11-langgraph-v2-agent-tool-node-amendment.md`

## How to resume (subagent-driven-development)

- SDD workspace (git-ignored scratch, but present on disk):
  `.superpowers/sdd/2026-09-11-langgraph-v2-phase3-rollout/`
  - `progress.md` — the full ledger: model assignment, environment rulings E1–E5, pre-flight scan
    rulings R1–R9, and every task's review/fix-round history **including parked findings**.
  - `task-N-brief.md` / `task-N-report.md` / `task-N-review.md` / `review-*.diff` per task.
  - `harness.sh` — **run every plan command through this**, not `docker exec hrag-backend`
    (that container mounts the MAIN repo, not this worktree). Usage:
    `./.superpowers/sdd/2026-09-11-langgraph-v2-phase3-rollout/harness.sh 'python -m pytest tests/... -q'`
- Agent/model assignment (user-directed):
  - implementer: `langgraph-v2-coder` → `opencode-go/muse-spark-1.3-contributor` (thinking high).
  - reviewer: `langgraph-v2-reviewer-sol` → `openai-codex/gpt-5.6-sol` (thinking medium);
    fallback `langgraph-v2-reviewer` → `deepseek/deepseek-flash`.
- Generate a task brief: `<superpowers>/skills/subagent-driven-development/scripts/task-brief <plan.md> <N>`.
  Review package: `…/scripts/review-package <plan.md> <BASE> <HEAD>`.
- Environment notes: `.gitnexus/` is absent in this worktree (skip `detect-changes`; use grep + narrow
  `git add`). `backend/scripts/` is gitignored → `git add -f` new scripts. `tests/agents/v2/orchestrator_compat`
  hardcodes `localhost:5433` and is excluded in the harness (E2).

## Completed (do not redo)

| Task | Commit range | Result |
|---|---|---|
| 0 preflight | — | Phase-2 gate 766 passed; no old layout |
| 1 A/B harness | `85ea6f0`→`07a94ad` | spec PASS |
| 2 tool gateway | `bc7c7f6`→`17b3c87` | spec PASS |
| 3 compare + complex subgraph | `e54f009`→`f484728` | spec PASS |
| 4 People→Document | `e5e84be`→`bc69db3` | spec PASS |
| 5 replan/discovery/summarize | `dd2e529`→`8e956f3` | complete with **2 parked test-only findings** |

### Parked / deferred from earlier tasks (for final whole-branch triage)
- **T5 parked (test-only):** the AST append-ownership guard is defeatable by a comprehension + indirect
  `dict(tasks=...)` update form; and the rogue-plan behavioral assertion is non-causal. Production invariant
  (single governed `replanning.append_replan_tasks` + frozen `validate_replan` + checkpoint-before-dispatch)
  is correct. See `progress.md`.
- **Deferred rollout gaps:**
  - scalar-backed `document.search` still returns `DEPENDENCY_UNAVAILABLE` when `person_identifier` is present
    (`capabilities/document.py`) → People→Document cannot yet produce candidates end-to-end.
  - T5 must consume `redact_scalar_for_model()` + the checkpointed per-task availability decision for every
    model-facing planning/replanning projection (policy owns the model planner).
  - T2 query for People→Document is derived from the plan goal until T5's planner-query wiring.

---

## Remaining task 1 — Task 6: side-effect-free shadow execution (mid-fix-loop)

Commits: `c07b16b` → `61e41fc` → `7c3c493` (head). Reviewer sol still returns **Spec ❌**.

### Open findings at `7c3c493`
- **B1/R62** — no spy on the real `ChatSession.title` mutation (`chat_session.py:1154-1155`); required spies can
  silently degrade to `env:` skips so the proof passes without them; chat/title tables not row-counted.
- **B2/R63** — the real chat-session relay (`relay.put_nowait`) and any webhook emitter are not spied; the
  efficacy test proves only a subset.
- **H1/R64** — the real-hook factual E2E test selects `DocumentRevision.status == 'active'` but the
  authoritative states are `published` (`models/document_revision.py`) → the test skips vacuously;
  `_latest_active_revision_id` repeats the wrong predicate; People-lookup errors are swallowed into a generic
  zero-task outcome instead of a typed gap.
- **M1/R65** — `_stop_shadow_task` does an unbounded `await task` after `wait_for`, so a cancellation-resistant
  shadow can block primary cleanup indefinitely (probe exit 124).
- **NEW authorization bug** — the shadow People lookup runs even when `can_read_people=False`
  (`shadow_runtime.py:381-425,851-861,874-883`). **Must fix.**

### Ruling R66 to apply on resume (controller)
1. `_should_resolve_people`-style gate: the semantic adapter MUST NOT call the read-only People source unless
   `can_read_people` is true; People-source failures must surface as a TYPED dependency-gap outcome (metrics +
   report), never a generic zero-task error.
2. Replace the `active` predicate with the authoritative `published` revision state (both the test fixture and
   `_latest_active_revision_id`).
3. Spy on the real `ChatSession.title` assignment path, the real chat relay (`relay.put_nowait`), and a real
   webhook emitter; a required spy that cannot be installed must FAIL the test (no `env:` skip for required
   targets); include chat/title tables in the row-count equality set.
4. `_stop_shadow_task` MUST be bounded end-to-end (bounded cancel-await, re-cancel, and force `stopped=False`
   reachable) and the caller must fail the shadow closed on `stopped=False` before primary cleanup; test against
   the real hook with a persistently cancellation-resistant shadow.
Then run the T6 gate and the full complex gate, and re-review with `langgraph-v2-reviewer-sol`.

---

## Remaining task 2 — Task 7A: rollout-control schema migration (migration-only release)

Plan section "Task 7A". Files: modify `backend/app/services/agents/v2/persistence/migrate.py`; test
`backend/tests/migrations/v2/test_rollout_control_migration.py`.

**Critical controller ruling R7 from the pre-flight scan — the plan's literal "1→2" is wrong.**
Schema version **2 is already taken** (T3 evidence-only lease upgrade). Task 7A must migrate to **version 3**:
- `V2_SCHEMA_VERSION = 3`; add a version-aware expected-table set (v1 tables ∪ rollout tables);
- `apply_v2_schema` performs **stepwise** upgrades 1→2 (`_LEASE_EVIDENCE_ONLY_ALTER`) then 2→3 (create
  `agent_rollout_control` + append-only `agent_rollout_metrics` + seed one disabled control row); fresh create
  lands at 3;
- `check_v2_schema` must not report the new tables as `extra`;
- keep the advisory lock and the legacy-unchanged guard;
- extend (never weaken) `tests/migrations/v2/test_migration_control.py` and
  `test_lease_evidence_only_upgrade.py`; the new test starts at version 2.
- Migration is applied and verified **before** Task 7B is deployed.
Command (harness): `harness.sh 'python -m app.services.agents.v2.persistence.migrate --apply'` then `--check`.

## Remaining task 3 — Task 7B: canary controls, kill switch, metrics, rollback gates

Plan section "Task 7B". New: `models/agent_rollout_control.py`, `models/agent_rollout_metric.py`,
`services/agent/rollout_control.py`, `services/agent/rollout_metrics.py`,
`scripts/collect_v2_rollout_report.py`, `scripts/check_v2_rollout_gate.py`; modify `models/v2_registry.py`,
`models/__init__.py`, `execution/scheduler.py`, `core/config.py`, `.env.example`,
`services/agent/runtime_selector.py`, `services/agent/streaming.py`, `api/chat_session.py`,
`api/chat_agent_lg.py`, `services/integrations/telegram_service.py`, `api/agent_admin.py`. Tests:
`tests/api/test_agent_canary_selection.py`, `tests/agents/v2/test_rollout_metrics.py`.

Key rulings from the pre-flight scan:
- **R8:** selection is server-owned; the runtime selector must NOT inspect semantic domains. Only an
  endpoint/request type deterministically known to be Write routes to v1 before bucketing. Everything else is
  bucketed; a v2 candidate whose Router resolves to `write`/`evaluate`/legal/compliance/unsupported falls back
  to v1 before any capability execution or user-visible output. `CANARY_PERCENT=100` = 100% of **v2-eligible**
  traffic, never a global v1 replacement.
- Deterministic bucket from authenticated workspace + persisted request ID + `NEXUSRAG_AGENT_V2_BUCKET_SALT`;
  admin-only override retained; ordinary headers ignored.
- Config flags: `NEXUSRAG_AGENT_V2_ENABLED=false`, `..._SHADOW_PERCENT=0`, `..._CANARY_PERCENT=0`,
  `..._CANARY_WORKSPACES=`, `..._BUCKET_SALT=<secret>`; DB control authoritative within environment ceilings.
- Metrics: arm, hashed request/workspace IDs, timing, terminal status, citation count, cancellation state, and
  authoritative security counters (`checkpoint_secret`, `ungrounded_factual_success`, `acl_leak`,
  `duplicate_production_write`); missing/default security fields are INVALID, never "safe". Do NOT compare
  `grounded_quality` across arms (Quality is gated in Task 1's golden preflight).
- Gate: ≥200 completed samples/arm, ≥24 continuous hours, zero security violations, v2 error-rate regression
  ≤1pp, v2 p95 regression ≤15%, cancellation failure ≤0.1%. Golden/preflight report schemas are rejected by the
  live checker.
- Add the three eligibility tests: `test_rollout_100_percent_still_routes_write_to_v1`,
  `test_rollout_100_percent_still_routes_evaluate_to_v1`,
  `test_supported_compare_uses_v2_at_100_percent_eligible_rollout`.
- New scripts under `backend/scripts/` → `git add -f`.

## Remaining task 4 — Task 8: documentation + staged rollout

Plan section "Task 8". Modify `README.md`, `CLAUDE.md`, `docs/harness.md`, `docs/scaling.md`, `docs/workers.md`,
`docs/embedding.md`, `docs/auth.md`, `backend/docs/langgraph_architecture.md`,
`backend/app/services/agent/langgraph_diagram.md`, `backend/docs/route_permissions.md`.

- Document the final ownership model (Agent vs Node vs Capability vs Skill), the TaskPlan/scheduler invariant,
  sensitive observation projection, People→Document materialization, shadow isolation, kill switch, and v1
  removal criteria. Do not duplicate the frozen architecture into unrelated docs.
- **R9:** T8's live steps (`make ab`, staged canary over real traffic) cannot run in this worktree without the
  live stack; deliver the docs + runbook + the offline parts of the validation suite, and record the live steps
  as an operational hand-off with exact commands.
- Offline validation: `tests/agents/v2`, `tests/api`, `tests/migrations/v2`, `tests/workers`, then
  `make test-recall`, `make test-section`, `make test-validity`, `make fe-lint`, `make fe-build` (live-stack
  parts are operational).

## Phase-3 final acceptance gate (after T8)

- Complex-Research gate suite: `tests/agents/v2/complex/test_tool_gateway.py`, `test_comparison.py`,
  `test_people_document.py`, `test_replan_discovery.py` — with the named proofs listed in the plan.
- Static guards (must stay green):
  ```bash
  ! find backend/app/services/agents/v2 -type f \( -name 'people_agent.py' -o -name 'summary_agent.py' \
     -o -name 'comparison_agent.py' -o -name 'document_agent.py' -o -name 'section_agent.py' \
     -o -name 'kg_agent.py' -o -path '*/domain/*_graph.py' \) | grep .
  ! rg -n 'capability\.execute\(' backend/app/services/agents/v2/tools
  ! rg -n 'TaskScheduler|scheduler\.execute\(|checkpointer' backend/app/services/agents/v2/tools
  ! rg -n 'safe_metadata|Mapping\[str' backend/app/services/agents/v2/tools
  ```
- v1 remains the default/rollback path until persisted rollout control promotes v2.

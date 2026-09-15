# LangGraph v2 Pre-Canary Hardening Implementation Plan

> **Spec authority:** `docs/superpowers/specs/2026-09-14-langgraph-v2-semantic-adaptive-execution-design.md`
>
> **Continuation of:** `docs/superpowers/plans/2026-09-14-langgraph-v2-phase4-semantic-adaptive-execution.md`

**Goal:** Close the approved pre-canary gaps without widening v1 authority or running a live canary: narrow M11, make Phase 4C conversation coreference consumable end-to-end through an ACL-rechecked conversation-resource bridge, add browser/public-event coverage, restore frontend type/build health, and verify PostgreSQL checkpoint dependency readiness.

**Global constraints**

- Work only in `/home/AIRAG/.worktrees/langgraph-v2` on `feat/langgraph-v2`.
- Do not push, merge, deploy, restart Docker services, vLLM, or live infrastructure.
- TDD is mandatory: add a focused failing test, run it and record the expected failure, then implement the minimum change and rerun.
- Run GitNexus upstream impact before modifying every function/class/method. HIGH/CRITICAL results must be reported before edit. Run `detect_changes --scope compare --base-ref main` before every commit/final completion.
- V1 remains advisory for intent/document candidates. V2 remains sole authority for ACL, identity binding, revision pin, routing, scheduling, evidence, and checkpointing.
- History may surface only server-issued document identities as `KnownDocumentResource(source="conversation")`; it must never directly create an authorized binding. Current-turn resolver/binder must re-check ACL and pin revisions.
- Preserve public API compatibility and fail-open history loading. Person references remain default-deny without `can_read_people`.
- Coding model: `opencode-go/muse-spark-1.3-contributor`. Review model: `opencode-go/deepseek-v4.1-flash`.

---

## Task 1 — M11 and the Phase 4C conversation-resource bridge

**Expected production seams:**

- `backend/app/services/agents/v2/semantic/intent.py`
- `backend/app/services/agent/runtime_selector.py`
- `backend/app/services/agents/supervisor_v2.py`
- smallest necessary semantic/conversation adapter or contract helper only

**Expected tests:**

- `backend/tests/agents/v2/test_intent_translation.py`
- `backend/tests/agents/v2/test_discourse_coreference.py`
- `backend/tests/agents/v2/test_discourse_fix_round1.py` or a focused new continuation test
- relevant ingress/API regression tests

### 1A. Pin M11

Add a table-driven RED test proving ordinary informational/action queries that merely contain a generic verb and `pháp lý` remain non-`evaluate`, including representative Vietnamese and English forms. Keep positive assessment requests reachable. Implement the smallest cue-boundary change in `classify_evaluate`; do not add model authority or broaden lexical routing.

Acceptance:

- Generic verbs such as read/find/show/explain/summarize combined with a legal-topic phrase do not classify as `evaluate` unless an explicit assessment/compliance action is present.
- Existing explicit legal/compliance assessment cases still classify as `evaluate`.
- Existing informational compliance targetless path remains green.

### 1B. Build a server-issued conversation candidate bridge

Use identities persisted by the server on prior chat messages (`document_ids` and/or public citation metadata), never UUIDs parsed from message text. Preserve deterministic chronological order, deduplicate by document UUID while retaining the most recent useful order, and bound the projection to the already bounded history window.

The loader/ingress must return both:

1. the existing label/text-only `ConversationContext`; and
2. current-request `KnownDocumentResource` entries with `source="conversation"` merged with current-turn known resources.

Use a small typed loader result or helper rather than placing document UUIDs inside `ConversationContext`. Keep old call sites/API behavior compatible where practical. Malformed historical JSON/UUID values are ignored and history failures still produce an empty discourse/resource result.

Conversation resources are candidates only. The semantic adapter may create deterministic conversation-backed document references only when the current query contains a supported anaphora (`văn bản này`, `tài liệu này`, `file thứ hai`, etc.). Those references must flow through the existing request-scoped document identity/binding boundary so ACL is rechecked for the current workspace/user and the existing binder remains the only revision-pin authority. No direct binding from history is allowed.

Required RED/GREEN behavior:

- A prior assistant citation or message `document_ids` plus `văn bản này` produces one conversation candidate and a resolvable semantic reference only after current ACL resolution.
- Two ordered prior documents plus `file thứ hai` selects the second candidate deterministically.
- Duplicate historical identities do not change ordinal ordering or create duplicate refs.
- An out-of-scope/deleted historical document never binds and no UUID/title leaks through clarification/error text.
- No prior identity means the existing silent zero-local-referent behavior remains.
- `can_read_people=false` behavior is unchanged.
- Standalone/non-UUID thread IDs and DB failures remain fail-open.
- Existing `attachment`, `ui_selection`, and `api_explicit` semantics remain unchanged.
- Session chat, standalone/workspace chat, Telegram, and admin-evaluation ingress callers retain compatible behavior.

Run focused tests, then the full non-persistence v2 suite and relevant API tests. Commit only Task 1 files and write a task report under this plan's SDD workspace with impact, RED, GREEN, regression, and residual-risk evidence.

---

## Task 2 — Frontend type correctness and browser/public-event inventory

**Expected seams:**

- `frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx`
- `frontend/src/pages/AdminLLMConfigPage.tsx`
- `frontend/vite.config.ts`
- existing public chat hook/components and focused tests only when a real behavior gap is found
- a small browser harness/config under `frontend/e2e/` or equivalent
- an event inventory document under `docs/`

First run `npm run build` and pin every current TypeScript error. Fix errors minimally without suppressing strictness globally or changing runtime semantics accidentally. In particular, fix unused test variables with meaningful assertions, call `sendMessage` with its real typed signature, remove genuinely unused UI symbols/helpers, align LLM test request typing with the actual API contract, and make Vitest/Vite config typing explicit rather than weakening compiler options.

Add a public-event inventory mapping each supported SSE event to backend DTO, parser/hook transition, visible UI outcome, reload behavior, and automated test. Inventory must include token, rollback, sources/citations, images, people data, abbreviations, clarification, resume, complete status, cancellation, unknown events, and errors.

Add browser-level coverage using the already installed `playwright` library or the smallest justified test-runner dependency. Mock network at the browser boundary; do not require a live backend. Cover at minimum:

- streamed token followed by complete;
- clarification rendering and server-issued resume selection;
- citation rendering after history reload;
- cancellation/abort with no stale retractable artifacts;
- public error rendering;
- unknown-event tolerance.

Do not add production-only test hooks. If browser binaries are unavailable, the committed browser spec/harness must still type/lint successfully and the exact environment blocker must be recorded; do not claim browser execution.

Verification:

- focused Vitest tests;
- all frontend Vitest tests;
- `npm run build` exits 0;
- browser E2E command, or precise executable-environment blocker.

Commit only Task 2 files and write a task report.

---

## Task 3 — PostgreSQL checkpoint readiness and live-canary handoff

**Expected seams:**

- `backend/requirements.txt` / benchmark requirements only if pins are incorrect
- smallest relevant checkpoint/readiness tests or health/status seam
- `docs/harness.md` or a focused pre-canary runbook

Start with a RED readiness test that imports the exact production PostgreSQL checkpointer symbols used by the app and validates the configured package/pin relationship. Do not duplicate checkpoint implementations or add a fallback that silently changes persistence semantics.

Determine whether the host failure (`No module named langgraph.checkpoint.postgres`) is code/config drift or merely an uninstalled declared requirement. If declarations are already correct, do not churn requirement pins: add the minimal executable readiness guard/documented installation command and run tests in a dependency-complete project environment if one is available without restarting services. If no such environment exists, record the exact blocker and leave code truthful.

Run persistence/orchestrator suites when import readiness is satisfied. Do not start/restart Postgres, Docker services, backend, vLLM, or a live canary. Instead produce a pre-canary handoff containing:

- dependency installation/verification command;
- checkpoint setup/import probe;
- persistence/orchestrator commands;
- live SSE/browser canary scenarios;
- expected telemetry/checkpoint/resume signals;
- abort and rollback criteria;
- explicit commands that were not executed in this worktree.

Finally run fresh available backend/API/frontend/static architecture verification and GitNexus `detect_changes --scope compare --base-ref main`. Commit Task 3 documentation/readiness changes and write a final report with verified facts separated from operationally blocked items.

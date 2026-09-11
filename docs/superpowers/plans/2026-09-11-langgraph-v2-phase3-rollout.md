# LangGraph v2 Phase 3 Complex Pilots and Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complex research in bounded pilots, build session-SSE replay/A-B/shadow infrastructure, and roll out v2 through deterministic canaries with immediate rollback.

**Architecture:** The Phase-0 winner implements the approved ComplexResearchGraph contract. Pilots add comparison, then People→Document, then bounded replan/discovery; rollout tooling drives the same session SSE path as users and suppresses all shadow side effects.

**Tech Stack:** Python 3.11, selected orchestrator, FastAPI session SSE, Redis, PostgreSQL, pytest, benchmark JSON, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- Phase 2 full gate must pass; v1 remains default until canary gate.
- No complex feature may bypass TaskPlan validation, runtime ACL, evidence governance, evaluation, or grounding.
- Shadow v2 cannot persist chat messages, memory, title, audit mutations, evidence/use rows, checkpoints, or outbound events.
- Rollout bucket selection is deterministic and server-owned.
- Before editing existing symbols run exact impact; before every commit run compare-scope detect-changes and stage narrow paths.

---

### Task 1: Build Session-SSE Replay and A/B Harness First

**Files:**
- Modify: `backend/scripts/ab_eval.py`
- Create: `backend/scripts/replay_v2.py`
- Create: `backend/tests/scripts/test_v2_ab_replay.py`
- Modify: `Makefile`
- Modify: `docs/harness.md`

**Interfaces:**
- Produces: session-SSE driver, normalized v1/v2 report, offline replay input/output.

- [ ] **Step 1: Impact-check harness symbols**

```bash
impact({target: "scripts.ab_eval.cmd_run", direction: "upstream"})
impact({target: "scripts.ab_eval._call", direction: "upstream"})
```

Record direct callers/processes for the exact existing `ab_eval.py` symbols before replacing its HTTP driver.

- [ ] **Step 2: Write failing harness tests**

Assert the Phase-2 authenticated admin evaluation endpoint already exists before this task. The driver authenticates as admin, creates a chat session, POSTs `/api/admin/agent-evaluation/run` with `arm`, `session_id`, and `message`, parses named SSE events, waits for `complete|error`, records graph version/status/citations/latency/token count, and redacts auth/message PII. Assert it never uses debug-chat, never sends `X-Agent-Graph-Version`, and receives 403 for non-admin credentials.

```bash
cd backend && pytest tests/scripts/test_v2_ab_replay.py -q
```

Expected: FAIL before the session-SSE driver exists.

- [ ] **Step 3: Implement concrete session driver**

```python
@dataclass(frozen=True)
class SessionRun:
    graph_version: str
    terminal_event: str
    content: str
    citation_count: int
    latency_ms: float

async def run_session_sse(client: httpx.AsyncClient, base_url: str, token: str, query: str, graph_version: str) -> SessionRun:
    session = await client.post(f"{base_url}/api/chat-sessions", headers={"Authorization": f"Bearer {token}"}, json={"title": "v2-eval"})
    session.raise_for_status()
    session_id = session.json()["id"]
    headers = {"Authorization": f"Bearer {token}"}
    started = time.perf_counter()
    events = await read_sse(
        client,
        f"{base_url}/api/admin/agent-evaluation/run",
        headers,
        {"arm": graph_version, "session_id": session_id, "message": query},
    )
    terminal = require_single_terminal(events)
    return SessionRun(graph_version, terminal.name, terminal.content, terminal.citation_count, (time.perf_counter() - started) * 1000)
```

Define `read_sse` and `require_single_terminal` in the same script.

- [ ] **Step 4: Add replay/A-B commands and run tests**

```bash
cd backend && pytest tests/scripts/test_v2_ab_replay.py -q
make ab ARM=v1 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE
make ab ARM=v2 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE
```

Expected: tests pass; live commands produce comparable JSON when credentials/services are supplied.

- [ ] **Step 5: Commit before complex implementation**

```bash
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/scripts/ab_eval.py backend/scripts/replay_v2.py backend/tests/scripts/test_v2_ab_replay.py Makefile docs/harness.md
git commit -m "test: add session SSE v2 evaluation harness"
```

---

### Task 2: Implement Multi-Document Comparison Pilot

**Files:**
- Create: `backend/app/services/agents/v2/complex_research_graph.py`
- Create: `backend/tests/agents/v2/complex/test_comparison.py`
- Modify: `backend/app/services/agents/supervisor_v2.py`

**Interfaces:**
- Produces: validated two-target bounded comparison; no discovery/replan.

- [ ] **Step 1: Impact-check composition**

```bash
impact({target: "app.services.agents.supervisor_v2.create_supervisor_v2_graph", direction: "upstream"})
```

- [ ] **Step 2: Write failing comparison tests**

Cover two exact document/range targets, parallel safe reads, target/reference role preservation, complete coverage, contradictory sources, insufficient one-sided coverage, grounded use-bound claims, and no discovery/replan.

```bash
cd backend && pytest tests/agents/v2/complex/test_comparison.py -q
```

Expected: FAIL before ComplexResearchGraph exists.

- [ ] **Step 3: Implement selected orchestrator adapter**

```python
class ComplexResearchGraph:
    def __init__(self, planner: ResearchPlanner, scheduler: TaskScheduler):
        self._planner = planner
        self._scheduler = scheduler

    async def run(self, planning_input: ResearchPlanningInput, runtime: GraphRuntimeContext) -> ComplexResearchResult:
        plan = await self._planner.create_plan(planning_input)
        validate_task_plan(plan, planning_input.bindings)
        results = await self._scheduler.execute(plan, runtime)
        return ComplexResearchResult(plan=plan, task_results=results)
```

Define `ResearchPlanner`, `TaskScheduler`, and `ComplexResearchResult` in this module using approved contracts. Validator rejects discovery/replan for this pilot.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/agents/v2/complex/test_comparison.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/complex_research_graph.py backend/app/services/agents/supervisor_v2.py backend/tests/agents/v2/complex/test_comparison.py
git commit -m "feat: add v2 comparison pilot"
```

---

### Task 3: Implement People-to-Document Dependency Pilot

**Files:**
- Create: `backend/tests/agents/v2/complex/test_people_document.py`
- Modify: `backend/app/services/agents/v2/complex_research_graph.py`

**Interfaces:**
- Produces: dependency-aware People→Document execution with minimal replanner context.

- [ ] **Step 1: Write failing dependency tests**

Test People success supplies validated downstream input; `not_found` creates no EvidenceUse but remains in TaskExecutionSummary; TIMEOUT is distinct; denied does not become not_found; completed tasks never rerun; dependency input cannot be fabricated; People persistence obeys minimization/encryption/TTL/audit.

```bash
cd backend && pytest tests/agents/v2/complex/test_people_document.py -q
```

Expected: FAIL before dependency execution is enabled.

- [ ] **Step 2: Implement dependency-ready execution**

The scheduler dispatches only tasks whose `depends_on` succeeded with required typed output. Build `ResearchPlanningInput.current_plan`, `task_outcomes`, `prior_evidence_uses`, and `prior_evaluation` from checkpoint/store projections. Runtime validator rejects unknown dependencies and output references.

```bash
cd backend && pytest tests/agents/v2/complex/test_people_document.py -q
```

Expected: dependency and no-evidence outcome tests pass.

- [ ] **Step 3: Run and commit**

```bash
cd backend && pytest tests/agents/v2/complex/test_people_document.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/complex_research_graph.py backend/tests/agents/v2/complex/test_people_document.py
git commit -m "feat: add people document dependency pilot"
```

---

### Task 4: Add Bounded Replan and Discovery

**Files:**
- Create: `backend/app/services/agents/v2/replanning.py`
- Create: `backend/app/services/agents/v2/discovery.py`
- Create: `backend/tests/agents/v2/complex/test_replan_discovery.py`

**Interfaces:**
- Produces: append-only `validate_replan`, UUID discovery candidates, validated addition/promotion.

- [ ] **Step 1: Write failing replan/discovery tests**

Cover max replans/tasks/branches, completed task immutability, EvidenceUse trigger lineage, no-evidence not_found, timeout, UUID collision resistance across two discovery tasks, no autonomous target creation, policy-disabled discovery, ACL denial, candidate promotion exact pin, and current/latest rebinding.

```bash
cd backend && pytest tests/agents/v2/complex/test_replan_discovery.py -q
```

Expected: FAIL before replan/discovery modules exist.

- [ ] **Step 2: Implement deterministic validation**

```python
def validate_replan(current: TaskPlan, proposed: TaskPlan, outcomes: tuple[TaskExecutionSummary, ...], policy: DiscoveryPolicy, budget: ResearchBudgetView) -> TaskPlan:
    _require_existing_prefix(current, proposed)
    _require_completed_tasks_unchanged(current, proposed, outcomes)
    _require_new_ids(current, proposed)
    _require_budget(proposed, budget)
    _require_discovery_policy(proposed, policy)
    return proposed
```

Define all five helpers in `replanning.py`. Discovery candidate IDs use `uuid4`; BindingAdditionRequest can create supporting/discovered bindings only, while promotion requires explicit validated user/policy action.

- [ ] **Step 3: Run and commit**

```bash
cd backend && pytest tests/agents/v2/complex/test_replan_discovery.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/replanning.py backend/app/services/agents/v2/discovery.py backend/tests/agents/v2/complex/test_replan_discovery.py
git commit -m "feat: add bounded v2 replan and discovery"
node .gitnexus/run.cjs analyze
```

---

## Phase 3 Complex §26 Gate

Run only after Tasks 2–4. Named tests must include: `test_comparison_requires_both_target_ranges`, `test_people_not_found_without_evidence_is_not_timeout`, `test_people_timeout_cannot_fabricate_dependency_input`, `test_completed_task_is_not_rerun`, `test_replan_origin_uses_evidence_use_ids`, `test_discovery_candidate_uuid_unique_across_tasks`, `test_discovery_cannot_create_target`, and `test_replan_is_bounded_append_only`.

```bash
docker exec hrag-backend pytest tests/agents/v2/complex/test_comparison.py tests/agents/v2/complex/test_people_document.py tests/agents/v2/complex/test_replan_discovery.py -q
```

Expected: all named complex tests pass; these scenarios are not claimed by Phase 1 or Phase 2.

---

### Task 5: Implement Side-Effect-Free Shadow Execution

**Files:**
- Create: `backend/app/services/agent/shadow_runtime.py`
- Create: `backend/scripts/shadow_v2.py`
- Create: `backend/tests/agents/v2/test_shadow_runtime.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `backend/app/api/chat_session.py`

**Interfaces:**
- Produces: `ShadowRuntimeServices`, sampled v2 shadow execution, metrics-only comparison.

- [ ] **Step 1: Impact-check session execution**

```bash
impact({target: "app.api.chat_session.chat_stream_session", direction: "upstream"})
```

- [ ] **Step 2: Write side-effect suppression tests**

After adding validated shadow settings (`NEXUSRAG_AGENT_V2_SHADOW_PERCENT=0`, bucket salt, max concurrency, timeout) before constructing the test app, assert primary v1 alone persists production assistant/chat/title/memory/audit mutation/events. Shadow v2 receives read-only source snapshots but writable isolated ephemeral conversation/evidence/use/checkpoint/audit stores so normal graph writes succeed without touching production. Outbound SSE/webhook/title/memory writers are rejecting sinks. Assert v2 schema compatibility is checked before shadow selection, no shadow SSE is emitted, cancellation follows primary, and report contains only hashed request ID, route/status/latency/citation counts/error class.

```bash
cd backend && pytest tests/agents/v2/test_shadow_runtime.py -q
```

Expected: FAIL before shadow service boundary exists.

- [ ] **Step 3: Implement shadow service boundary**

```python
@dataclass(frozen=True)
class ShadowDecision:
    enabled: bool
    bucket: int


def shadow_decision(request_id: str, percentage: int, salt: str) -> ShadowDecision:
    digest = hashlib.sha256(f"{salt}:{request_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10000
    return ShadowDecision(bucket < percentage * 100, bucket)
```

`ShadowRuntimeServices` supplies read-only document/source adapters and writable request-scoped in-memory implementations for conversation snapshots, evidence records/uses, checkpoints, and audits; reads observe prior writes within that shadow run. Only external side effects—production DB/object/vector mutation, SSE/webhook emission, title updates, and memory writes—raise `ShadowSideEffectError`. Before starting v2, call the same `require_v2_schema_ready()` used by selection. The session endpoint starts shadow only after primary request persistence and discards shadow output except redacted metrics.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/agents/v2/test_shadow_runtime.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agent/shadow_runtime.py backend/scripts/shadow_v2.py backend/tests/agents/v2/test_shadow_runtime.py backend/app/core/config.py .env.example backend/app/api/chat_session.py
git commit -m "feat: add side effect free v2 shadowing"
```

---

### Task 6: Add Deterministic Canary Controls and Rollback Gates

**Files:**
- Create: `backend/app/models/agent_rollout_control.py`
- Create: `backend/app/models/agent_rollout_metric.py`
- Modify: `backend/app/models/v2_registry.py`
- Modify: `backend/app/models/__init__.py`
- Create: `backend/app/services/agent/rollout_control.py`
- Create: `backend/app/services/agent/rollout_metrics.py`
- Modify: `backend/app/services/agents/v2/persistence/migrate.py`
- Modify: `backend/app/services/agents/v2/execution/scheduler.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `backend/app/services/agent/runtime_selector.py`
- Modify: `backend/app/services/agent/streaming.py`
- Modify: `backend/app/api/chat_session.py`
- Modify: `backend/app/api/chat_agent_lg.py`
- Modify: `backend/app/services/integrations/telegram_service.py`
- Modify: `backend/app/api/agent_admin.py`
- Create: `backend/tests/api/test_agent_canary_selection.py`
- Create: `backend/tests/agents/v2/test_rollout_metrics.py`
- Create: `backend/tests/fixtures/rollout/v1-pass.json`
- Create: `backend/tests/fixtures/rollout/v2-pass.json`
- Create: `backend/tests/fixtures/rollout/v2-security-fail.json`
- Create: `backend/tests/fixtures/rollout/v2-latency-fail.json`
- Create: `backend/tests/fixtures/rollout/v2-error-rate-fail.json`
- Create: `backend/tests/fixtures/rollout/v2-cancellation-fail.json`
- Create: `backend/tests/fixtures/rollout/v2-quality-fail.json`
- Create: `backend/tests/fixtures/rollout/v2-window-fail.json`
- Create: `backend/scripts/collect_v2_rollout_report.py`
- Create: `backend/scripts/check_v2_rollout_gate.py`

**Interfaces:**
- Produces: DB-backed per-request rollout decision, dynamic kill switch, deterministic bucketing, Redis active-run cancellation, terminal metric collection, versioned per-arm reports, and exact report gate.

- [ ] **Step 1: Impact-check selector/settings**

```bash
impact({target: "app.core.config.Settings", direction: "upstream"})
impact({target: "app.services.agent.runtime_selector.resolve_agent_graph", direction: "upstream"})
impact({target: "app.api.agent_admin.run_agent_evaluation", direction: "upstream"})
impact({target: "app.api.chat_session.chat_stream_session", direction: "upstream"})
impact({target: "app.api.chat_session.cancel_stream_session", direction: "upstream"})
impact({target: "app.api.chat_agent_lg.langgraph_chat_stream", direction: "upstream"})
impact({target: "app.services.integrations.telegram_service._handle_question", direction: "upstream"})
impact({target: "app.services.agent.streaming.stream_agent_events", direction: "upstream"})
```

- [ ] **Step 2: Write deterministic selection tests**

Create config defaults before constructing tests. Test DB `enabled=false` forces v1 even when environment allows v2; control row is read on every request (no process-local stale cache); explicit evaluation override requires admin; workspace allowlist precedes percentage; stable bucketing; percentage 0/100; ordinary headers ignored; v2 schema incompatibility forces v1/error before graph creation. Test session, direct SSE, Telegram, shadow, and admin-evaluation v2 starts all call `ActiveRunRegistry.register(run_id, cancellation_token)` before graph invocation and unregister in `finally`; Redis disable messages set the same token in every worker; scheduler checks it immediately before every capability dispatch; cancellation rolls back retractable output and cannot emit success. Metric tests insert terminal observations and generate deterministic arm reports matching the fixture schema.

```bash
cd backend && pytest tests/api/test_agent_canary_selection.py -q
```

Expected: FAIL before canary controls exist.

- [ ] **Step 3: Add exact controls**

```text
NEXUSRAG_AGENT_V2_ENABLED=false
NEXUSRAG_AGENT_V2_SHADOW_PERCENT=0
NEXUSRAG_AGENT_V2_CANARY_PERCENT=0
NEXUSRAG_AGENT_V2_CANARY_WORKSPACES=
NEXUSRAG_AGENT_V2_BUCKET_SALT=<secret-from-runtime>
```

Environment values are bootstrap ceilings only. Add a singleton `agent_rollout_control` database row containing `enabled`, shadow/canary percentages, workspace allowlist, revision, and updated timestamp. `RolloutControlRepository.get_current()` is awaited on every request; no process-local cache decides an arm. Selection uses authenticated workspace ID plus persisted request ID, checks DB enabled, schema readiness, workspace allowlist, then `sha256(salt:workspace_id:request_id) % 10000 < percent * 100`. Ordinary client headers are always ignored; the admin evaluation endpoint is the only override. Increment `V2_SCHEMA_VERSION` from 1 to 2. Migration 1→2 creates `agent_rollout_control` plus append-only `agent_rollout_metrics`, then seeds exactly one disabled control row under the existing advisory lock; version 0, gaps, and versions newer than 2 remain incompatible. The same readiness check used by selector/shadow/canary requires version 2 after this task.

- [ ] **Step 4: Implement the concrete metric collector, report schema, and rollback wiring**

`RolloutMetricsCollector.record_terminal()` writes one row keyed by run ID with arm, hashed request/workspace IDs, started/completed timestamps, terminal status, latency_ms, grounded-quality score, citation count, cancellation requested/succeeded flags, and four boolean security violations (`acl_leak`, `duplicate_production_write`, `checkpoint_secret`, `ungrounded_factual_success`). It stores no query/answer/auth token. `collect_v2_rollout_report.py --arm v1|v2 --since ... --until ... --output ...` aggregates this table into versioned JSON containing arm, window bounds/hours, completed sample count, error rate, p50/p95, mean grounded quality, cancellation count/failure rate, and each security-violation count. Unit tests compare exact output with committed v1/v2 pass fixtures and security, latency, error-rate, cancellation, quality, and sample/window failure fixtures.

`ActiveRunRegistry` uses Redis sets `agent:v2:active:{worker_id}` plus pub/sub `agent:v2:cancel`; every v2 ingress registers its run/token before invoking streaming/graph code and unregisters in `finally`. The process listener maps each run ID to its local token. The disable endpoint commits `enabled=false` and incremented revision, enumerates active sets, and publishes each run ID; `stream_agent_events` converts token cancellation to rollback/cancelled terminal output, and `TaskScheduler` checks immediately before each dispatch. New requests select v1; cancelled runs never report success.

```bash
cd backend
python scripts/collect_v2_rollout_report.py --arm v1 --since "$SINCE" --until "$UNTIL" --output tests/reports/v1-canary.json
python scripts/collect_v2_rollout_report.py --arm v2 --since "$SINCE" --until "$UNTIL" --output tests/reports/v2-canary.json
python scripts/check_v2_rollout_gate.py --v1 tests/reports/v1-canary.json --v2 tests/reports/v2-canary.json --min-samples 200 --min-window-hours 24 --require-zero-security-violations
```

The checker fails on any security count, missing/wrong arm or schema version, window mismatch, fewer than 200 completed requests per arm, under 24 continuous hours, v2 error-rate regression over 1 percentage point, p95 regression over 15%, cancellation failure over 0.1%, or grounded-quality loss over 2 percentage points. Fixture tests prove pass plus each failure exit code.

- [ ] **Step 5: Run and commit**

```bash
cd backend && pytest tests/api/test_agent_canary_selection.py tests/agents/v2/test_rollout_metrics.py -q
cd backend && python scripts/check_v2_rollout_gate.py --v1 tests/fixtures/rollout/v1-pass.json --v2 tests/fixtures/rollout/v2-pass.json --min-samples 200 --min-window-hours 24 --require-zero-security-violations
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/models/agent_rollout_control.py backend/app/models/agent_rollout_metric.py backend/app/models/v2_registry.py backend/app/models/__init__.py backend/app/services/agent/rollout_control.py backend/app/services/agent/rollout_metrics.py backend/app/services/agents/v2/persistence/migrate.py backend/app/services/agents/v2/execution/scheduler.py backend/app/core/config.py .env.example backend/app/services/agent/runtime_selector.py backend/app/services/agent/streaming.py backend/app/api/chat_session.py backend/app/api/chat_agent_lg.py backend/app/services/integrations/telegram_service.py backend/app/api/agent_admin.py backend/tests/api/test_agent_canary_selection.py backend/tests/agents/v2/test_rollout_metrics.py backend/tests/fixtures/rollout backend/scripts/collect_v2_rollout_report.py backend/scripts/check_v2_rollout_gate.py
git commit -m "feat: add deterministic v2 canary controls"
```

---

### Task 7: Complete Documentation and Rollout

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/harness.md`
- Modify: `docs/scaling.md`
- Modify: `docs/workers.md`
- Modify: `docs/embedding.md`
- Modify: `docs/auth.md`
- Modify: `backend/docs/langgraph_architecture.md`
- Modify: `backend/app/services/agent/langgraph_diagram.md`
- Modify: `backend/docs/route_permissions.md`

**Interfaces:**
- Produces: canonical operational/runbook documentation; no duplicate architecture in `AGENTS.md`.

- [ ] **Step 1: Update exact operational content**

Document revision allocate/build/verify/publish, reindex and deletion/tombstone/executable GC schedule, evidence encryption/retention/audit, checkpoint setup/check/restore/current ACL, auth and admin eval override, session-SSE A/B, writable isolated shadow stores, DB-backed per-request canary/kill-switch controls, active-run cancellation, ≥200-per-arm/24-hour/zero-security promotion thresholds, and v1 removal criteria. Correct stale legacy-loop and “no formal test suite” statements. Do not copy architecture into `AGENTS.md`.

```bash
grep -RInE 'legacy loop|no formal test suite' README.md CLAUDE.md docs backend/docs || true
grep -RIn 'NEXUSRAG_AGENT_V2_' .env.example CLAUDE.md docs/auth.md docs/harness.md
```

Expected: first command finds no active stale guidance after edits; second finds documented controls.

- [ ] **Step 2: Execute staged rollout gates**

```bash
make ab ARM=v1 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE OUTPUT=backend/tests/reports/v1-canary.json
make ab ARM=v2 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE OUTPUT=backend/tests/reports/v2-canary.json
python backend/scripts/check_v2_rollout_gate.py --v1 backend/tests/reports/v1-canary.json --v2 backend/tests/reports/v2-canary.json --min-samples 200 --min-window-hours 24 --require-zero-security-violations
```

Then run shadow 5%, internal workspace canary, 5%, 25%, 50%, and 100%. Each stage must collect at least 200 completed requests per arm across a continuous 24-hour window with zero security violations before promotion. Any hard gate failure calls the admin disable endpoint, verifies the DB control revision changed, verifies new requests select v1, and verifies active v2 runs receive cancellation and cannot complete successfully.

- [ ] **Step 3: Run final validation**

```bash
docker exec hrag-backend pytest tests/agents/v2 tests/api tests/migrations/v2 tests/workers -q
make test-recall
make test-section
make test-validity
make fe-lint
make fe-build
node .gitnexus/run.cjs analyze
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git diff --check
```

Expected: all pass and reports meet gates.

- [ ] **Step 4: Commit docs only**

```bash
git add README.md CLAUDE.md docs/harness.md docs/scaling.md docs/workers.md docs/embedding.md docs/auth.md backend/docs/langgraph_architecture.md backend/app/services/agent/langgraph_diagram.md backend/docs/route_permissions.md
git diff --cached --check
git commit -m "docs: publish LangGraph v2 rollout runbook"
```

V1 removal is a separate future plan after 100% v2 stability gates; this plan does not delete it.

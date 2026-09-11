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

### Task 0: Verify Phase-3 Paths, Symbols, and Rollout Preconditions

**Files:**
- Read: all Modify paths and selected orchestrator imports below
- Test: shell preflight only

**Interfaces:**
- Produces: repository-drift manifest and verified Phase-2 baseline.

- [ ] **Step 1: Verify paths and create conflicts**

```bash
set -e
for path in backend/scripts/ab_eval.py Makefile docs/harness.md backend/app/services/agents/supervisor_v2.py backend/app/services/agents/v2/execution/scheduler.py backend/app/services/agent/runtime_selector.py backend/app/api/chat_session.py; do test -e "$path"; done
for path in backend/app/services/agents/v2/complex_research_graph.py backend/app/services/agents/v2/dependencies/people_document.py backend/app/services/agent/shadow_runtime.py backend/app/services/agents/v2/persistence/shadow_checkpoint.py; do test ! -e "$path"; done
python - <<'PY'
import re, pathlib
plan = pathlib.Path('docs/superpowers/plans/2026-09-11-langgraph-v2-phase3-rollout.md').read_text()
modify = [p.split(':')[0] for p in re.findall(r'^- Modify: `([^`]+)`', plan, re.M)]
create = [p.split(':')[0] for p in re.findall(r'^- Create: `([^`]+)`', plan, re.M)]
missing = [p for p in modify if not pathlib.Path(p).exists()]
conflict = [p for p in create if pathlib.Path(p).exists()]
assert not missing and not conflict, {'missing': missing, 'conflict': conflict}
print(f'phase3 paths ok: {len(set(modify))} modify, {len(set(create))} create')
PY
rg -n 'create_supervisor_v2_graph|resolve_agent_graph|run_agent_evaluation' backend/app
```

Expected: all Modify paths/symbols exist and Create paths do not conflict. `TaskScheduler` is intentionally not asserted here because Task 2 introduces it in `v2/execution/scheduler.py`. Use discovered qualified names for impact commands.

- [ ] **Step 2: Re-run Phase-2 and dependency API gates**

```bash
docker exec hrag-backend pytest tests/agents/v2 tests/api/test_agent_runtime_selector.py tests/api/test_agent_v2_streaming.py -q
docker exec hrag-backend python - <<'PY'
from inspect import signature
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
import langgraph.graph.state as _lg_state
assert 'context_schema' in signature(StateGraph).parameters
assert hasattr(_lg_state, 'CompiledStateGraph')
PY
```

Expected: PASS before complex/rollout edits.

---

### Task 1: Build Golden Session-SSE A/B Preflight Harness

**Files:**
- Modify: `backend/scripts/ab_eval.py`
- Create: `backend/scripts/replay_v2.py`
- Create: `backend/tests/scripts/test_v2_ab_replay.py`
- Modify: `Makefile`
- Modify: `docs/harness.md`

**Interfaces:**
- Produces: session-SSE driver, normalized golden preflight report, and offline replay input/output; it does not claim live canary duration/sample evidence.

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

Define `read_sse` and `require_single_terminal` in the same script. Define `score_quality(runs, evaluator_version)` used identically for both arms, and record `evaluator_version` in every preflight report; a preflight quality comparison is valid only when both arms carry the same `evaluator_version`.

- [ ] **Step 4: Add replay/A-B commands and run tests**

```bash
cd backend && pytest tests/scripts/test_v2_ab_replay.py -q
make ab ARM=v1 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE
make ab ARM=v2 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE
```

Expected: tests pass; commands produce comparable functional JSON plus a quality comparison from the **shared evaluator** — the same evaluator implementation and version run over both arms on the golden dataset. This preflight has no 24-hour or live sample-count threshold and cannot promote a canary; it is the only place quality is compared, precisely because it is the only place both arms share one evaluator.

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
- Modify: `backend/app/services/agents/v2/execution/scheduler.py`
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

Define `ResearchPlanner` and `ComplexResearchResult` in `complex_research_graph.py`, and define `class TaskScheduler` in `backend/app/services/agents/v2/execution/scheduler.py` (import it here — this is the single shared scheduler that Phase 3 Tasks 3 and 6 instrument for dependency materialization and cancellation). `TaskScheduler` is a thin class that owns results accumulation and dependency materialization and delegates dispatch to the Phase-2 `execute_ready_tasks(plan, results, registry, runtime)` function; its `execute(plan, runtime) -> tuple[AgentResult, ...]` method is the shared entrypoint. `ComplexResearchGraph` receives the scheduler by injection and never defines a second scheduler. Validator rejects discovery/replan for this pilot.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/agents/v2/complex/test_comparison.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/complex_research_graph.py backend/app/services/agents/v2/execution/scheduler.py backend/app/services/agents/supervisor_v2.py backend/tests/agents/v2/complex/test_comparison.py
git commit -m "feat: add v2 comparison pilot"
```

---

### Task 3: Implement People-to-Document Dependency Pilot

**Files:**
- Create: `backend/app/services/agents/v2/dependencies/__init__.py`
- Create: `backend/app/services/agents/v2/dependencies/people_document.py`
- Create: `backend/tests/agents/v2/complex/test_people_document.py`
- Modify: `backend/app/services/agents/v2/complex_research_graph.py`
- Modify: `backend/app/services/agents/v2/execution/scheduler.py`

**Interfaces:**
- Produces: `PeopleDocumentDependencyAdapter.materialize(...) -> DocumentSearchInput` and dependency-aware execution without generic output references.

- [ ] **Step 1: Write failing dependency tests**

Test People success hydrates only the governed admitted People EvidenceUse and materializes only the exact identifier required by DocumentSearchInput. Assert unrelated/sensitive People fields never enter planner input, TaskPlan, AgentResult.data, checkpoint, or downstream input. `not_found` creates no use and dispatches no T2; denied/error/TIMEOUT dispatch no T2 and remain distinct summaries; completed tasks never rerun; missing/expired/unauthorized evidence cannot be materialized; no input can be fabricated.

```bash
cd backend && pytest tests/agents/v2/complex/test_people_document.py -q
```

Expected: FAIL before dependency execution is enabled.

- [ ] **Step 2: Implement dependency-ready execution**

The scheduler treats `depends_on` as ordering only. For a recognized People→Document edge it calls `PeopleDocumentDependencyAdapter`, which receives T1 result/use refs plus current runtime, performs governed hydration with ACL/expiry/minimization checks, extracts the named allowed scalar through a fixed mapping, and constructs a concrete validated DocumentSearchInput immediately before T2 dispatch. It never exposes raw People records to the planner/checkpoint. Non-success or absent admitted use returns a typed blocked dependency and no T2 dispatch. Runtime rejects unknown dependency materializers; no generic TaskOutputRef is added.

```bash
cd backend && pytest tests/agents/v2/complex/test_people_document.py -q
```

Expected: dependency and no-evidence outcome tests pass.

- [ ] **Step 3: Run and commit**

```bash
cd backend && pytest tests/agents/v2/complex/test_people_document.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/dependencies/__init__.py backend/app/services/agents/v2/dependencies/people_document.py backend/app/services/agents/v2/complex_research_graph.py backend/app/services/agents/v2/execution/scheduler.py backend/tests/agents/v2/complex/test_people_document.py
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
- Create: `backend/app/services/agents/v2/persistence/shadow_checkpoint.py`
- Create: `backend/scripts/shadow_v2.py`
- Create: `backend/tests/agents/v2/test_shadow_runtime.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `backend/app/api/chat_session.py`

**Interfaces:**
- Produces: `create_shadow_supervisor_v2_graph()`, isolated saver/stores, `ShadowRuntimeServices`, sampled metrics-only execution.

- [ ] **Step 1: Impact-check session execution**

```bash
impact({target: "app.api.chat_session.chat_stream_session", direction: "upstream"})
```

- [ ] **Step 2: Write side-effect suppression tests**

Assert shadow never calls `get_supervisor_v2_graph()` and instead calls `create_supervisor_v2_graph(checkpointer=isolated_saver)` once per isolated bundle. After a shadow run, query production checkpointer/evidence/use/audit/chat/memory/title tables and assert zero new rows; assert zero outbound SSE/webhook events. Shadow gets read-only source snapshots plus writable request-scoped isolated conversation/evidence/use/audit stores and an InMemorySaver (or dedicated temporary PostgreSQL schema in integration tests), so graph writes/read-after-write work without production access. Cancellation follows primary; report contains only hashed request ID, route/status/latency/citation count/error class.

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

`create_shadow_supervisor_v2_graph()` creates a fresh isolated saver using the exact Phase-0-supported InMemorySaver API, then calls `create_supervisor_v2_graph(checkpointer=isolated_saver)`; it never reuses the production compiled singleton whose AsyncPostgresSaver is compile-time bound. `ShadowRuntimeServices` supplies read-only source adapters and writable request-scoped in-memory conversation/evidence/use/audit stores. Reject production DB/object/vector mutation, outbound SSE/webhook, title, and memory writes. Start shadow only after primary request persistence and discard output except redacted metrics.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/agents/v2/test_shadow_runtime.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agent/shadow_runtime.py backend/app/services/agents/v2/persistence/shadow_checkpoint.py backend/scripts/shadow_v2.py backend/tests/agents/v2/test_shadow_runtime.py backend/app/core/config.py .env.example backend/app/api/chat_session.py
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
- Create: `backend/tests/fixtures/rollout/v2-evaluator-version-fail.json`
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

Create config defaults before constructing tests. Test DB `enabled=false` forces v1 even when environment allows v2; control row is read on every request (no process-local stale cache); explicit evaluation override requires admin; workspace allowlist precedes percentage; stable bucketing; percentage 0/100; ordinary headers ignored; v2 schema incompatibility forces v1/error before graph creation. Test session, direct SSE, Telegram, shadow, and admin-evaluation v2 starts all call `ActiveRunRegistry.register(run_id, cancellation_token)` before graph invocation and unregister in `finally`; Redis disable messages set the same token in every worker; scheduler checks it immediately before every capability dispatch; cancellation rolls back retractable output and cannot emit success. Metric tests insert terminal observations and generate deterministic arm reports matching the fixture schema. Add `test_canary_metric_producers_fire`: injecting an ACL leak, a duplicate production write, a leaked checkpoint secret field, and an ungrounded factual success each makes the corresponding counter ≥ 1, proving producers are authoritative rather than defaulted. The live checker rejects golden/preflight report schemas even when their counts are large; only reports aggregated from `agent_rollout_metrics` with continuous window metadata are eligible. Live reports carry no quality metric: quality is compared only in the golden preflight by the shared evaluator, and the checker rejects any live report that contains a quality or evaluator field (including a preflight report passed by mistake).

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

Environment values are bootstrap ceilings only. Add a singleton `agent_rollout_control` database row containing `enabled`, shadow/canary percentages, workspace allowlist, revision, and updated timestamp. `RolloutControlRepository.get_current()` is awaited on every request; no process-local cache decides an arm. Selection uses authenticated workspace ID plus persisted request ID, checks DB enabled, schema readiness, workspace allowlist, then `sha256(salt:workspace_id:request_id) % 10000 < percent * 100`. Ordinary client headers are always ignored; the admin evaluation endpoint is the only override. Increment `V2_SCHEMA_VERSION` from 1 to 2. Migration 1→2 creates `agent_rollout_control` plus append-only `agent_rollout_metrics`, then seeds exactly one disabled control row under the existing advisory lock; version 0, gaps, and versions newer than 2 remain incompatible. The same readiness check used by selector/shadow/canary requires version 2 after this task. Apply the same two-release discipline as Phase 1: release A deploys migration-capable code only (no selector/graph path requires schema 2), runs `migrate --apply` and verifies version 2; release B then deploys the selector/shadow/canary code that requires schema 2. Never deploy code requiring schema 2 before the 1→2 migration is applied and verified.

- [ ] **Step 4: Implement the concrete metric collector, report schema, and rollback wiring**

`RolloutMetricsCollector.record_terminal()` writes one row keyed by run ID with arm, hashed request/workspace IDs, started/completed timestamps, terminal status, latency_ms, citation count, cancellation requested/succeeded flags, and four security-violation counts (`acl_leak`, `duplicate_production_write`, `checkpoint_secret`, `ungrounded_factual_success`). It stores no query/answer/auth token. Every metric has an authoritative producer and is written explicitly at terminal emission; all counters and scalar fields are non-null, and a missing/default value is a validation error, never treated as secure. There is deliberately **no** `grounded_quality` field: within v2 a successful terminal already requires every material factual claim to be grounded, so a grounded/total ratio is a tautology near 1.0, and v1 has no comparable counterpart — comparing it across arms would be meaningless. The v2 grounding invariant is instead enforced as the `ungrounded_factual_success` correctness counter (must be 0), and user-visible quality is compared only in the golden A/B preflight with the shared evaluator:

```text
checkpoint_secret          -> deterministic checkpoint serialization scanner (rejects ACL/identity/deadline/secret fields in serialized state)
ungrounded_factual_success -> finalizer grounding invariant (success requires every material factual claim bound to an admitted EvidenceUse); counted as a correctness violation, never used as a quality score
acl_leak                   -> current-ACL hydration/grounding violation detector
duplicate_production_write -> idempotency/audit mutation detector (ON CONFLICT no-op vs. second durable side effect)
```

`collect_v2_rollout_report.py --arm v1|v2 --since ... --until ... --output ...` aggregates this table into versioned JSON containing arm, window bounds/hours, completed sample count, error rate, p50/p95, cancellation count/failure rate, and each security-violation count — no quality field. Unit tests compare exact output with committed v1/v2 pass fixtures and security, latency, error-rate, cancellation, and sample/window failure fixtures; `v2-evaluator-version-fail.json` proves a report carrying a quality/evaluator field is rejected as a non-live schema.

`ActiveRunRegistry` uses Redis sets `agent:v2:active:{worker_id}` plus pub/sub `agent:v2:cancel`; every v2 ingress registers its run/token before invoking streaming/graph code and unregisters in `finally`. The process listener maps each run ID to its local token. The disable endpoint commits `enabled=false` and incremented revision, enumerates active sets, and publishes each run ID; `stream_agent_events` converts token cancellation to rollback/cancelled terminal output, and `TaskScheduler` checks immediately before each dispatch. New requests select v1; cancelled runs never report success.

```bash
cd backend
python scripts/collect_v2_rollout_report.py --arm v1 --since "$SINCE" --until "$UNTIL" --output tests/reports/v1-canary.json
python scripts/collect_v2_rollout_report.py --arm v2 --since "$SINCE" --until "$UNTIL" --output tests/reports/v2-canary.json
python scripts/check_v2_rollout_gate.py --v1 tests/reports/v1-canary.json --v2 tests/reports/v2-canary.json --min-samples 200 --min-window-hours 24 --require-zero-security-violations
```

The checker fails on any security count, missing/wrong arm or schema version, window mismatch, fewer than 200 completed requests per arm, under 24 continuous hours, v2 error-rate regression over 1 percentage point, p95 regression over 15%, or cancellation failure over 0.1%. Quality is out of scope for this checker by construction: a quality regression can only be detected by the shared-evaluator golden preflight, and any attempt to pass a preflight report to the live checker is rejected by schema. Fixture tests prove pass plus each failure exit code.

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

- [ ] **Step 2: Execute golden preflight, then the separate live rollout gate**

Golden preflight compares functionality/quality only, both computed by the same shared evaluator for both arms:

```bash
make ab ARM=v1 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE OUTPUT=backend/tests/reports/v1-preflight.json
make ab ARM=v2 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE OUTPUT=backend/tests/reports/v2-preflight.json
make ab-compare A=backend/tests/reports/v1-preflight.json B=backend/tests/reports/v2-preflight.json
```

After preflight passes, enable shadow 5%, then internal workspace canary and 5/25/50/100% stages. For each stage, wait for actual `agent_rollout_metrics` traffic spanning a continuous 24-hour window, then collect database-backed reports and apply live thresholds:

```bash
python backend/scripts/collect_v2_rollout_report.py --arm v1 --since "$SINCE" --until "$UNTIL" --output backend/tests/reports/v1-live-canary.json
python backend/scripts/collect_v2_rollout_report.py --arm v2 --since "$SINCE" --until "$UNTIL" --output backend/tests/reports/v2-live-canary.json
python backend/scripts/check_v2_rollout_gate.py --v1 backend/tests/reports/v1-live-canary.json --v2 backend/tests/reports/v2-live-canary.json --min-samples 200 --min-window-hours 24 --require-zero-security-violations
```

Batch A/B JSON is never accepted by the live checker, and the live gate never compares quality. Any failure calls the admin disable endpoint, verifies control revision changed, new requests select v1, and active v2 runs are cancelled without success.

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

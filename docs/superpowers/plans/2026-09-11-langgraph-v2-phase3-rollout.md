# LangGraph v2 Phase 3 Complex Research and Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one adaptive complex-research planning boundary over the shared Phase-2 capabilities, then validate comparison, People→Document, bounded replan/discovery, shadow execution, and controlled rollout without introducing domain agents or direct model-to-capability execution.

**Architecture:** The Phase-0 winner implements `ComplexResearchGraph`, but the framework is constrained by the frozen execution model: the complex agent proposes an initial `TaskPlan` or append-only replan; deterministic validators accept/reject it; authoritative plan state is checkpointed; the shared scheduler alone dispatches capabilities; evaluator/synthesis/grounding remain outside agent authority. Agent-facing tool adapters are proposal gateways plus safe observation projectors over the same Phase-2 capabilities.

**Tech Stack:** Python 3.11, selected orchestrator, LangGraph, Pydantic v2, FastAPI session SSE, Redis, PostgreSQL, pytest, benchmark JSON, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

**Normative ownership amendment:** `docs/superpowers/plans/2026-09-11-langgraph-v2-agent-tool-node-amendment.md`

## Global Constraints

- Phase 2 full gate must pass; v1 remains default until rollout gates promote v2.
- The complex-research agent is the only adaptive planner/replanner boundary.
- “Tool selection” means proposing `TaskSpec.capability`; it never means calling `capability.execute()` directly.
- Every factual task must be validated and checkpointed before scheduler dispatch.
- Fast and complex paths use the same `CapabilityRegistry` and capability implementations.
- Agent/model observations are explicit projections; raw sensitive capability outputs are not planner-visible by default.
- People→Document remains deterministic dependency materialization over governed evidence, never agent handoff or raw People observation.
- Evaluator owns sufficiency/contradiction authority; grounding owns factual success/citations.
- Subagents, if selected by the orchestrator, are advisory/context-isolation helpers only and have no capability execution, TaskPlan, EvidenceUse, sufficiency, or FinalResponse authority.
- Shadow v2 cannot write production checkpoints/evidence/audit/chat/memory/title or outbound events.
- Rollout bucket selection is deterministic and server-owned.
- Before editing existing symbols run exact impact; before every commit run compare-scope detect-changes and stage narrow paths.

Canonical Phase-3 additions:

```text
backend/app/services/agents/v2/
├── tools/
│   ├── adapters.py
│   ├── gateway.py
│   └── observations.py
├── dependencies/
│   └── people_document.py
├── skills/
│   ├── summarize.*
│   ├── compare.*
│   ├── legal_analysis.*
│   └── compliance.*
├── replanning.py
├── discovery.py
└── complex_research_graph.py
```

---

### Task 0: Verify Phase-3 Preconditions and Architecture Guards

**Files:**
- Read: Phase-2 implementation and amendment
- Test: shell preflight only

**Interfaces:**
- Produces: verified node/capability Phase-2 baseline and no conflicting domain-agent layout.

- [ ] **Step 1: Verify Phase-2 files and no old layout**

```bash
set -e
for path in \
  backend/app/services/agents/supervisor_v2.py \
  backend/app/services/agents/v2/nodes/execute.py \
  backend/app/services/agents/v2/nodes/evaluate.py \
  backend/app/services/agents/v2/execution/scheduler.py \
  backend/app/services/agents/v2/capabilities/people.py \
  backend/app/services/agents/v2/capabilities/document.py \
  backend/app/services/agent/runtime_selector.py; do
  test -e "$path"
done

test ! -d backend/app/services/agents/v2/domain
! find backend/app/services/agents/v2 -type f \
  \( -name 'people_agent.py' -o -name 'summary_agent.py' -o -name 'comparison_agent.py' \
     -o -name 'document_agent.py' -o -name 'section_agent.py' -o -name 'kg_agent.py' \) | grep .
```

- [ ] **Step 2: Re-run Phase-2 gate**

```bash
docker exec hrag-backend pytest tests/agents/v2 tests/api/test_agent_runtime_selector.py tests/api/test_agent_v2_streaming.py -q
```

Expected: PASS before complex changes.

---

### Task 1: Build Golden Session-SSE A/B Preflight Harness

**Files:**
- Modify: `backend/scripts/ab_eval.py`
- Create: `backend/scripts/replay_v2.py`
- Test: `backend/tests/scripts/test_v2_ab_replay.py`
- Modify: `Makefile`
- Modify: `docs/harness.md`

**Interfaces:**
- Produces: authenticated session-SSE driver and golden functional/quality comparison; does not claim live canary duration/sample evidence.

- [ ] **Step 1: Write failing harness tests**

Assert admin-only evaluation endpoint exists, driver creates a session, sends server-side arm selection, reads named SSE events to one terminal, records latency/citations/status, redacts auth/message PII, never sends client graph-version headers, and returns 403 for non-admin.

- [ ] **Step 2: Implement shared-evaluator preflight**

Both v1 and v2 outputs are evaluated by the same preflight evaluator version. Persist `evaluator_version` into reports and reject comparison if versions differ.

```bash
cd backend && pytest tests/scripts/test_v2_ab_replay.py -q
make ab ARM=v1 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE
make ab ARM=v2 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE
```

- [ ] **Step 3: Commit**

```bash
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/scripts/ab_eval.py backend/scripts/replay_v2.py backend/tests/scripts/test_v2_ab_replay.py Makefile docs/harness.md
git commit -m "test: add session SSE v2 evaluation harness"
```

---

### Task 2: Implement Governed Agent Tool Gateway and Safe Observations

**Files:**
- Create: `backend/app/services/agents/v2/tools/__init__.py`
- Create: `backend/app/services/agents/v2/tools/gateway.py`
- Create: `backend/app/services/agents/v2/tools/adapters.py`
- Create: `backend/app/services/agents/v2/tools/observations.py`
- Test: `backend/tests/agents/v2/complex/test_tool_gateway.py`

**Interfaces:**
- Produces: `CapabilityInvocationProposal`, `AgentToolGateway`, request-scoped framework adapters, `AgentToolObservation`, and sensitive observation projectors.

- [ ] **Step 1: Write failing governance tests**

Named tests:

```text
test_agent_tool_call_creates_validated_task_before_dispatch
test_tool_adapter_cannot_call_capability_directly
test_unplanned_capability_dispatch_is_rejected
test_unknown_or_unauthorized_tool_is_rejected_at_execution
test_planner_never_receives_runtime_secrets
test_people_observation_does_not_expose_raw_record
test_capability_output_is_not_model_observation_by_default
```

- [ ] **Step 2: Implement proposal gateway**

```python
@dataclass(frozen=True)
class CapabilityInvocationProposal:
    capability: str
    objective: str
    input: CapabilityInput
    depends_on: tuple[str, ...] = ()

@dataclass(frozen=True)
class AgentToolObservation:
    task_id: str
    status: AgentStatus
    evidence_use_ids: tuple[UUID, ...]
    coverage: tuple[CoverageObservation, ...]
    result_kind: str
    safe_metadata: Mapping[str, str]
```

`AgentToolGateway.invoke(...)` must perform:

```text
proposal
-> create proposed TaskSpec/append-only plan
-> validate_task_plan or validate_replan
-> persist/checkpoint authoritative plan
-> shared TaskScheduler dispatch
-> validate AgentResult/EvidenceUse
-> ObservationProjector
-> AgentToolObservation
```

It never calls a capability directly.

- [ ] **Step 3: Implement request-scoped framework adapters**

```text
base capability registry
∩ current runtime permissions
∩ feature flags
∩ service availability
= agent-visible tool catalog
```

The framework-facing schema contains only allowed `CapabilityInput`; runtime authority is injected server-side. People/memory/sensitive projectors return status/use IDs/safe metadata only.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/agents/v2/complex/test_tool_gateway.py -q
! rg -n 'capability\.execute\(' backend/app/services/agents/v2/tools
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/tools backend/tests/agents/v2/complex/test_tool_gateway.py
git commit -m "feat: add governed complex agent tool gateway"
```

---

### Task 3: Implement Multi-Document Comparison Pilot Through One Complex Planner

**Files:**
- Create: `backend/app/services/agents/v2/skills/compare.*`
- Create: `backend/app/services/agents/v2/complex_research_graph.py`
- Modify: `backend/app/services/agents/v2/execution/scheduler.py`
- Modify: `backend/app/services/agents/supervisor_v2.py`
- Test: `backend/tests/agents/v2/complex/test_comparison.py`

**Interfaces:**
- Produces: one complex planning boundary, compare skill/policy, validated two-target bounded comparison; no replan/discovery yet.

- [ ] **Step 1: Write failing comparison/ownership tests**

Cover exact two-target ranges, parallel safe reads, target/reference roles, complete coverage, contradictory evidence, insufficient one-sided coverage, grounded use-bound claims, and:

```text
test_compare_is_skill_not_agent_route
test_fast_and_complex_share_same_capability_instance_or_factory
test_complex_agent_uses_request_scoped_tool_catalog
test_complex_agent_cannot_execute_capability_without_scheduler
```

- [ ] **Step 2: Implement initial-plan-only complex research**

```python
class ComplexResearchGraph:
    def __init__(
        self,
        planner: ResearchPlanner,
        scheduler: TaskScheduler,
        evaluator: EvidenceEvaluator,
    ):
        self._planner = planner
        self._scheduler = scheduler
        self._evaluator = evaluator

    async def run(
        self,
        planning_input: ResearchPlanningInput,
        runtime: GraphRuntimeContext,
    ) -> ComplexResearchResult:
        proposed = await self._planner.create_plan(planning_input)
        plan = validate_task_plan(proposed, planning_input.bindings)
        await runtime.services.plan_checkpoint.persist(plan)
        results = await self._scheduler.execute(plan, runtime)
        evaluation = await self._evaluator.evaluate(plan, results, runtime)
        return ComplexResearchResult(plan=plan, task_results=results, evaluation=evaluation)
```

The planner may choose capabilities only by placing them into `TaskSpec`. No framework-native tool call may bypass the gateway/scheduler invariant. For this pilot, reject discovery/replan.

- [ ] **Step 3: Replace only Phase-2 `complex_boundary` implementation**

`supervisor_v2` composition remains outer orchestration; complex path calls `ComplexResearchGraph`, then shared synthesis/grounding/finalizer nodes.

- [ ] **Step 4: Test and commit**

```bash
cd backend && pytest tests/agents/v2/complex/test_tool_gateway.py tests/agents/v2/complex/test_comparison.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/skills backend/app/services/agents/v2/complex_research_graph.py backend/app/services/agents/v2/execution/scheduler.py backend/app/services/agents/supervisor_v2.py backend/tests/agents/v2/complex/test_comparison.py
git commit -m "feat: add governed v2 comparison research"
```

---

### Task 4: Implement People→Document Deterministic Dependency Materialization

**Files:**
- Create: `backend/app/services/agents/v2/dependencies/__init__.py`
- Create: `backend/app/services/agents/v2/dependencies/people_document.py`
- Modify: `backend/app/services/agents/v2/execution/scheduler.py`
- Modify: `backend/app/services/agents/v2/tools/observations.py`
- Test: `backend/tests/agents/v2/complex/test_people_document.py`

**Interfaces:**
- Produces: `PeopleDocumentDependencyAdapter.materialize(...) -> DocumentSearchInput`; People planner observation remains minimized.

- [ ] **Step 1: Write failing dependency/security tests**

Require:

```text
People success -> governed EvidenceUse -> exact scalar -> DocumentSearchInput
raw People row never enters planner/checkpoint
CCCD/DOB/unrelated fields absent from AgentToolObservation
not_found -> no T2
PERMISSION_DENIED/TIMEOUT/error -> no T2 and remain distinct
expired/unauthorized use -> no materialization
no fabricated downstream input
```

- [ ] **Step 2: Implement materializer**

`depends_on` expresses ordering only. The scheduler recognizes a registered People→Document materializer; it receives T1 result/use refs plus current runtime, hydrates only the admitted People evidence under current ACL/expiry/minimization policy, extracts one fixed allowed scalar, validates concrete `DocumentSearchInput`, and only then dispatches T2.

Planner/tool observation returns status/use IDs and dependency availability metadata, never the sensitive scalar itself.

- [ ] **Step 3: Test and commit**

```bash
cd backend && pytest tests/agents/v2/complex/test_people_document.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/dependencies backend/app/services/agents/v2/execution/scheduler.py backend/app/services/agents/v2/tools/observations.py backend/tests/agents/v2/complex/test_people_document.py
git commit -m "feat: add governed people document dependency"
```

---

### Task 5: Add Bounded Append-Only Replan, Discovery, and Research Loop

**Files:**
- Create: `backend/app/services/agents/v2/replanning.py`
- Create: `backend/app/services/agents/v2/discovery.py`
- Modify: `backend/app/services/agents/v2/complex_research_graph.py`
- Modify: `backend/app/services/agents/v2/tools/gateway.py`
- Test: `backend/tests/agents/v2/complex/test_replan_discovery.py`

**Interfaces:**
- Produces: append-only `validate_replan`, UUID discovery candidates, bounded PLAN→VALIDATE→CHECKPOINT→EXECUTE→EVALUATE→REPLAN loop.

- [ ] **Step 1: Write failing replan/discovery tests**

Cover task/replan/parallel budgets, completed task immutability, `EvidenceUse` trigger lineage, no-evidence not_found, TIMEOUT, discovery UUID uniqueness, no autonomous target creation, policy-disabled discovery, ACL denial, exact revision pin/promotion, current/latest rebinding, and:

```text
test_replan_can_add_tasks_but_cannot_widen_authorization
test_completed_task_is_not_rerun
test_tool_gateway_checkpoints_replan_before_dispatch
test_subagent_cannot_append_authoritative_tasks
```

- [ ] **Step 2: Implement deterministic replan validation**

```python
def validate_replan(
    current: TaskPlan,
    proposed: TaskPlan,
    outcomes: tuple[TaskExecutionSummary, ...],
    policy: DiscoveryPolicy,
    budget: ResearchBudgetView,
) -> TaskPlan:
    _require_existing_prefix(current, proposed)
    _require_completed_tasks_unchanged(current, proposed, outcomes)
    _require_new_ids(current, proposed)
    _require_budget(proposed, budget)
    _require_discovery_policy(proposed, policy)
    _require_capabilities_within_current_runtime_catalog(proposed)
    return proposed
```

- [ ] **Step 3: Implement canonical research loop**

```text
planner proposes initial plan
-> validate
-> checkpoint
-> scheduler
-> evaluator
-> sufficient/terminal? stop
-> build ResearchPlanningInput(current_plan, task_outcomes, prior_evidence_uses, prior_evaluation)
-> planner proposes append-only plan
-> validate_replan
-> checkpoint
-> scheduler
-> evaluator
-> bounded repeat
```

Planner sees validated observations/evaluation gaps, not raw evidence by default. Governed synthesis later hydrates evidence content separately.

- [ ] **Step 4: Implement discovery semantics**

Discovery candidate IDs use UUID; additions create supporting/discovered bindings only; promotion requires explicit validated policy/user action; planner cannot create new user targets autonomously.

- [ ] **Step 5: Test and commit**

```bash
cd backend && pytest tests/agents/v2/complex/test_replan_discovery.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/replanning.py backend/app/services/agents/v2/discovery.py backend/app/services/agents/v2/complex_research_graph.py backend/app/services/agents/v2/tools/gateway.py backend/tests/agents/v2/complex/test_replan_discovery.py
git commit -m "feat: add bounded v2 research replanning"
```

## Complex Research Gate

```bash
docker exec hrag-backend pytest \
  tests/agents/v2/complex/test_tool_gateway.py \
  tests/agents/v2/complex/test_comparison.py \
  tests/agents/v2/complex/test_people_document.py \
  tests/agents/v2/complex/test_replan_discovery.py -q
```

Named proofs must include:

```text
test_complex_agent_uses_request_scoped_tool_catalog
test_fast_and_complex_share_same_capability_instance_or_factory
test_agent_tool_call_creates_validated_task_before_dispatch
test_tool_adapter_cannot_call_capability_directly
test_compare_is_skill_not_agent_route
test_people_observation_does_not_expose_raw_record
test_people_document_dependency_is_not_agent_handoff
test_replan_can_add_tasks_but_cannot_widen_authorization
test_subagent_cannot_receive_raw_people_record_or_runtime_secrets
```

---

### Task 6: Implement Side-Effect-Free Shadow Execution

**Files:**
- Create: `backend/app/services/agent/shadow_runtime.py`
- Create: `backend/app/services/agents/v2/persistence/shadow_checkpoint.py`
- Create: `backend/scripts/shadow_v2.py`
- Test: `backend/tests/agents/v2/test_shadow_runtime.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `backend/app/api/chat_session.py`

**Interfaces:**
- Produces: separately compiled shadow graph, isolated saver/writable stores, read-only source adapters, metrics-only output.

- [ ] **Step 1: Write side-effect tests**

Assert shadow never calls production `get_supervisor_v2_graph()` and instead compiles with `InMemorySaver` or isolated temporary PostgreSQL schema. After a run, production checkpoint/evidence/use/audit/chat/memory/title rows are unchanged; no outbound SSE/webhook events are sent.

- [ ] **Step 2: Implement isolated shadow bundle**

```text
fresh isolated saver
+ isolated conversation/evidence/use/audit stores
+ read-only source adapters
-> create_supervisor_v2_graph(checkpointer=isolated_saver)
```

Cancellation follows primary; output is discarded except redacted metrics.

- [ ] **Step 3: Test and commit**

```bash
cd backend && pytest tests/agents/v2/test_shadow_runtime.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agent/shadow_runtime.py backend/app/services/agents/v2/persistence/shadow_checkpoint.py backend/scripts/shadow_v2.py backend/tests/agents/v2/test_shadow_runtime.py backend/app/core/config.py .env.example backend/app/api/chat_session.py
git commit -m "feat: add side effect free v2 shadowing"
```

---

### Task 7A: Apply Rollout-Control Schema Migration (Migration-Only Release)

**Files:**
- Modify: `backend/app/services/agents/v2/persistence/migrate.py`
- Test: `backend/tests/migrations/v2/test_rollout_control_migration.py`

**Interfaces:**
- Produces: advisory-locked idempotent schema 1→2 migration creating `agent_rollout_control` and append-only `agent_rollout_metrics`, seeding one disabled control row.

- [ ] **Step 1: Write migration test**

Start at version 1; assert migration creates tables/control row, is idempotent, uses advisory lock, imports no ORM metadata, and rejects unsupported gaps/newer versions.

- [ ] **Step 2: Implement raw-SQL migration and deploy it before consumers**

Release A contains migration-capable code only. Apply and verify schema 2 before Task 7B is deployed.

```bash
docker exec hrag-backend python -m app.services.agents.v2.persistence.migrate --apply
docker exec hrag-backend python -m app.services.agents.v2.persistence.migrate --check
```

- [ ] **Step 3: Commit migration-only release**

```bash
cd backend && pytest tests/migrations/v2/test_rollout_control_migration.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/persistence/migrate.py backend/tests/migrations/v2/test_rollout_control_migration.py
git commit -m "feat: migrate rollout control schema to version 2"
```

---

### Task 7B: Add Canary Controls, Kill Switch, Metrics, and Rollback Gates

**Files:**
- Create: `backend/app/models/agent_rollout_control.py`
- Create: `backend/app/models/agent_rollout_metric.py`
- Modify: `backend/app/models/v2_registry.py`
- Modify: `backend/app/models/__init__.py`
- Create: `backend/app/services/agent/rollout_control.py`
- Create: `backend/app/services/agent/rollout_metrics.py`
- Modify: `backend/app/services/agents/v2/execution/scheduler.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `backend/app/services/agent/runtime_selector.py`
- Modify: `backend/app/services/agent/streaming.py`
- Modify: `backend/app/api/chat_session.py`
- Modify: `backend/app/api/chat_agent_lg.py`
- Modify: `backend/app/services/integrations/telegram_service.py`
- Modify: `backend/app/api/agent_admin.py`
- Test: `backend/tests/api/test_agent_canary_selection.py`
- Test: `backend/tests/agents/v2/test_rollout_metrics.py`
- Create: `backend/scripts/collect_v2_rollout_report.py`
- Create: `backend/scripts/check_v2_rollout_gate.py`

**Interfaces:**
- Produces: DB-backed rollout decision, deterministic buckets, Redis active-run cancellation, authoritative security counters, live report/gate.

- [ ] **Step 1: Write selection/cancellation/metric tests**

Test control row read per request, workspace allowlist before percentage, deterministic bucket, ordinary headers ignored, admin-only override, active run registration/unregistration, distributed cancellation before each scheduler dispatch, and authoritative producers for:

```text
checkpoint_secret
ungrounded_factual_success
acl_leak
duplicate_production_write
```

Missing/default security fields are invalid, never interpreted as safe.

- [ ] **Step 2: Implement controls**

```text
NEXUSRAG_AGENT_V2_ENABLED=false
NEXUSRAG_AGENT_V2_SHADOW_PERCENT=0
NEXUSRAG_AGENT_V2_CANARY_PERCENT=0
NEXUSRAG_AGENT_V2_CANARY_WORKSPACES=
NEXUSRAG_AGENT_V2_BUCKET_SALT=<runtime-secret>
```

DB control is authoritative within environment ceilings. Selection is server-owned and uses authenticated workspace + persisted request ID.

- [ ] **Step 3: Implement live metrics without pseudo-quality metric**

Live metrics contain arm, hashed request/workspace IDs, timing, terminal status, citation count, cancellation state, and security counters. Do **not** compare `grounded_quality` across v1/v2: v2 grounded-success completeness is an invariant, not an arm-neutral quality measure. User-visible quality is gated in Task 1 golden preflight by the shared evaluator.

- [ ] **Step 4: Implement live report/gate**

Requirements:

```text
>= 200 completed samples per arm
>= 24 continuous hours
zero security violations
v2 error-rate regression <= 1 percentage point
v2 p95 regression <= 15%
cancellation failure <= 0.1%
```

Golden/preflight report schemas are rejected by live checker.

- [ ] **Step 5: Test and commit**

```bash
cd backend && pytest tests/api/test_agent_canary_selection.py tests/agents/v2/test_rollout_metrics.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/models/agent_rollout_control.py backend/app/models/agent_rollout_metric.py backend/app/models/v2_registry.py backend/app/models/__init__.py backend/app/services/agent/rollout_control.py backend/app/services/agent/rollout_metrics.py backend/app/services/agents/v2/execution/scheduler.py backend/app/core/config.py .env.example backend/app/services/agent/runtime_selector.py backend/app/services/agent/streaming.py backend/app/api/chat_session.py backend/app/api/chat_agent_lg.py backend/app/services/integrations/telegram_service.py backend/app/api/agent_admin.py backend/tests/api/test_agent_canary_selection.py backend/tests/agents/v2/test_rollout_metrics.py backend/scripts/collect_v2_rollout_report.py backend/scripts/check_v2_rollout_gate.py
git commit -m "feat: add deterministic v2 canary controls"
```

---

### Task 8: Complete Documentation and Staged Rollout

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
- Produces: operational documentation and controlled rollout runbook.

- [ ] **Step 1: Document final ownership model**

Document Agent vs Node vs Capability vs Skill, TaskPlan/scheduler execution invariant, sensitive observation projection, People→Document materialization, shadow isolation, kill switch, and v1 removal criteria. Do not duplicate the full frozen architecture into unrelated docs.

- [ ] **Step 2: Run golden preflight**

```bash
make ab ARM=v1 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE OUTPUT=backend/tests/reports/v1-preflight.json
make ab ARM=v2 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml WORKSPACE=$WORKSPACE OUTPUT=backend/tests/reports/v2-preflight.json
make ab-compare A=backend/tests/reports/v1-preflight.json B=backend/tests/reports/v2-preflight.json
```

- [ ] **Step 3: Run shadow then staged canary**

```text
shadow 5%
-> internal workspace canary
-> 5%
-> 25%
-> 50%
-> 100%
```

Each canary stage requires real `agent_rollout_metrics` traffic over the live gate window. Failure disables v2, increments control revision, routes new requests to v1, and cancels active v2 runs without success.

- [ ] **Step 4: Final validation**

```bash
docker exec hrag-backend pytest tests/agents/v2 tests/api tests/migrations/v2 tests/workers -q
make test-recall
make test-section
make test-validity
make fe-lint
make fe-build
```

## Phase-3 Final Acceptance Gate

Before promotion prove all of:

```text
one adaptive complex-research planning boundary only
no domain-agent/domain-graph wrappers
fast and complex share capability implementations
agent tool call -> validated checkpointed TaskSpec -> scheduler -> capability
agent-facing adapters cannot call capability directly
People raw record never reaches planner/checkpoint
People->Document is deterministic materialization
replan append-only and authorization cannot widen
subagent has no execution/persistence/sufficiency authority
shadow has zero production writes
live canary uses DB-backed metrics only
```

Static guards:

```bash
set -e
! find backend/app/services/agents/v2 -type f \
  \( -name 'people_agent.py' -o -name 'summary_agent.py' -o -name 'comparison_agent.py' \
     -o -name 'document_agent.py' -o -name 'section_agent.py' -o -name 'kg_agent.py' \
     -o -path '*/domain/*_graph.py' \) | grep .

! rg -n 'capability\.execute\(' backend/app/services/agents/v2/tools
```

Expected: all gates pass while v1 remains the rollback/default path until persisted rollout control promotes v2.

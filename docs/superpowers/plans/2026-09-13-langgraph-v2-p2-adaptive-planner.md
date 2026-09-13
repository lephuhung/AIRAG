# LangGraph v2 P2 Governed Adaptive Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make complex multi-step factual requests use an LLM planner while retaining deterministic P0 retrieval fallback and all execution governance.

**Architecture:** Inject a request-scoped planner service, expose only typed/redacted planning inputs, and invoke it inside the proposal-owning validation node. Validate, lease, and checkpoint every accepted initial/replan proposal before the shared scheduler executes it.

**Tech Stack:** Python 3.11, LangGraph, Pydantic v2, OpenAI-compatible vLLM client, Langfuse, pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-langgraph-v2-factual-retrieval-reindex-design.md`

## Global Constraints

- P0 and P1 gates must pass first.
- Planner is runtime-only and never checkpointed.
- Planner sees no ACL, raw evidence, personal scalar, token, service, namespace, or revision identity.
- Planner only proposes; evaluator/scheduler/grounding remain authoritative.
- Invalid, cyclic, unauthorized, over-budget, or scope-widening proposals are rejected.
- Replans are append-only and bounded; deterministic retrieval is the timeout/disabled fallback.
- Planner feature flag defaults off and must be documented in `.env.example`.

---

### Task 1: Define typed planner boundary and sanitized projection

**Files:**
- Create: `backend/app/services/agents/v2/planning/__init__.py`
- Create: `backend/app/services/agents/v2/planning/adapter.py`
- Modify: `backend/app/services/agents/v2/contracts/state.py`
- Modify: `backend/tests/agents/v2/complex/test_adaptive_planner.py`

**Interfaces:**
- Produces: `PlannerProposal`, `ComplexPlanner` protocol, `build_planner_projection(...)`; `RuntimeServices.planner`.

- [ ] **Step 1: Write RED tests**

Assert the projection includes semantic goal, narrowed catalog, budget, typed observations, outcomes and gaps; assert known People scalar, runtime secret, document revision, namespace, clients, and raw evidence are absent.

- [ ] **Step 2: Run RED**

Run: `cd backend && PYTHONPATH=. pytest tests/agents/v2/complex/test_adaptive_planner.py -q`

- [ ] **Step 3: Implement typed boundary**

```python
@dataclass(frozen=True)
class PlannerProposal:
    tasks: tuple[TaskSpec, ...]

class ComplexPlanner(Protocol):
    async def propose_initial(self, planning_input: ResearchPlanningInput, observations: tuple[AgentToolObservation, ...]) -> PlannerProposal: ...
    async def propose_replan(self, planning_input: ResearchPlanningInput, observations: tuple[AgentToolObservation, ...]) -> PlannerProposal: ...
```

Use existing redaction/projector helpers; add only runtime-only `planner: Any = None` to `RuntimeServices`.

- [ ] **Step 4: Run GREEN and commit**

```bash
cd backend && PYTHONPATH=. pytest tests/agents/v2/complex/test_adaptive_planner.py tests/agents/v2/contracts -q
git add backend/app/services/agents/v2/planning backend/app/services/agents/v2/contracts/state.py backend/tests/agents/v2/complex/test_adaptive_planner.py
git commit -m "feat(v2): define governed planner boundary"
```

### Task 2: Implement OpenAI-compatible planner adapter

**Files:**
- Modify: `backend/app/services/agents/v2/planning/adapter.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `backend/tests/agents/v2/complex/test_adaptive_planner.py`

**Interfaces:**
- Produces: `OpenAIComplexPlanner`; settings `V2_ADAPTIVE_PLANNER_ENABLED=false`, `V2_PLANNER_MAX_TOKENS`, `V2_PLANNER_TIMEOUT_SEC`.

- [ ] **Step 1: Write RED adapter tests**

Assert valid structured output maps to `TaskSpec` proposals; malformed JSON, unknown capability, timeout, extra fields, and prompt-injected runtime fields return typed rejection/fallback signals; no call occurs when disabled.

- [ ] **Step 2: Implement minimal structured call**

Use the existing OpenAI-compatible client/configuration and Langfuse conventions. Require strict JSON schema derived from allowed typed proposal fields. The prompt contains only `build_planner_projection` output and instructs that capability selection is advisory.

- [ ] **Step 3: Run GREEN and commit**

```bash
cd backend && PYTHONPATH=. pytest tests/agents/v2/complex/test_adaptive_planner.py -q
git add backend/app/services/agents/v2/planning/adapter.py backend/app/core/config.py .env.example backend/tests/agents/v2/complex/test_adaptive_planner.py
git commit -m "feat(v2): add structured adaptive planner"
```

### Task 3: Integrate initial planning and append-only replanning

**Files:**
- Modify: `backend/app/services/agents/v2/complex_research_graph.py`
- Modify: `backend/app/services/agent/runtime_selector.py`
- Modify: `backend/tests/agents/v2/complex/test_adaptive_planner.py`
- Modify: `backend/tests/agents/v2/complex/test_replan_discovery.py`

**Interfaces:**
- Consumes: `RuntimeServices.planner` and `PlannerProposal`.
- Produces: governed planner path inside `validate_checkpoint_node`.

- [ ] **Step 1: Write RED real-graph tests**

Prove planner call occurs for a multi-step query; proposal is absent from intermediate checkpoints; accepted plan is validated/leased/checkpointed before dispatch; two dependent tasks execute; bad proposal is rejected; timeout uses deterministic retrieve plan; replan limit terminates.

- [ ] **Step 2: Integrate in the single-owner node**

Initial: if enabled and request is complex, call planner inside `validate_checkpoint_node`, construct proposed `TaskPlan`, validate against current bindings/catalog/budget, lease, then return it. Replan: pass only redacted typed observations and call `append_replan_tasks`; never replace completed tasks or widen explicit hard scope.

- [ ] **Step 3: Run GREEN and commit**

```bash
cd backend && PYTHONPATH=. pytest tests/agents/v2/complex -q
git add backend/app/services/agents/v2/complex_research_graph.py backend/app/services/agent/runtime_selector.py backend/tests/agents/v2/complex
git commit -m "feat(v2): execute governed adaptive plans"
```

### Task 4: Planner metrics, shadow parity, documentation and live gate

**Files:**
- Modify: `backend/app/services/agent/rollout_metrics.py`
- Modify: `backend/app/services/agent/shadow_runtime.py`
- Modify: `backend/tests/agents/v2/test_rollout_metrics.py`
- Modify: `backend/tests/agents/v2/test_shadow_runtime.py`
- Modify: `docs/harness.md`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Produces: planner call/latency/fallback metrics and shadow-safe planner execution.

- [ ] **Step 1: Write RED tests**

Assert planner metrics contain counts/latency/fallback but no prompt/evidence/secrets; shadow planner uses isolated saver/stores and emits no events; cancellation/deadline stop planner work.

- [ ] **Step 2: Implement and run GREEN**

Run: `cd backend && PYTHONPATH=. pytest tests/agents/v2/test_rollout_metrics.py tests/agents/v2/test_shadow_runtime.py tests/agents/v2/complex -q`

- [ ] **Step 3: Run full static/regression gate and commit**

```bash
! rg 'capability\.execute\(' backend/app/services/agents/v2/tools
! rg 'TaskScheduler|scheduler\.execute\(|checkpointer' backend/app/services/agents/v2/tools
cd backend && PYTHONPATH=. pytest tests/agents/v2 tests/api tests/migrations/v2 tests/workers -q
git diff --check
git add backend/app/services/agent backend/tests/agents/v2 docs/harness.md CLAUDE.md README.md
git commit -m "test(v2): gate adaptive multi-step planning"
```

- [ ] **Step 4: Live multi-step acceptance**

Enable planner only in the internal test workspace. Run a golden question requiring at least two causally related tasks. Required evidence: one planner LLM call, validated checkpointed plan, at least two task results, non-zero retrieval EvidenceUses, grounded citations, no scope widening, and no vLLM restart.

# LangGraph v2 Phase 2 Fast Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the independent v2 supervisor, deterministic fast paths, shared typed capabilities, evaluator/synthesis/grounding, clarification, selector, and streaming compatibility while v1 remains production default.

**Architecture:** Context, binding, routing, execution, evaluation, synthesis, grounding, clarification, and finalization are LangGraph nodes with predetermined ownership. Every factual fast route creates and checkpoints a deterministic one-task `TaskPlan`; the execute node delegates only to the shared scheduler, which resolves the request-scoped `CapabilityRegistry` and invokes atomic capabilities with `CapabilityRuntimeContext`. There are no People/Document/Section/KG domain agents or domain subgraphs.

**Tech Stack:** Python 3.11, LangGraph Phase-0 winner, Pydantic v2, FastAPI, PostgreSQL checkpointer, pytest, SSE, React/Vitest.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

**Normative ownership amendment:** `docs/superpowers/plans/2026-09-11-langgraph-v2-agent-tool-node-amendment.md`

## Global Constraints

- Phase 1 gate must pass; v1 remains default.
- The frozen spec owns business contracts; the agent/tool/node amendment owns implementation taxonomy, package layout, and tool execution semantics.
- Create `supervisor_v2.py` before selector wiring.
- Only direct non-factual conversation bypasses factual evidence/grounding.
- Write (pasted-text grammar/proofread/rewrite) is outside this v2 rollout; v1 remains its owner.
- Every factual execution owns a validated checkpointed `TaskPlan` before capability dispatch.
- Only the scheduler may execute a capability.
- Capabilities receive `AgentRequest` + `CapabilityRuntimeContext`, never arbitrary `SupervisorV2State`/`GraphRuntimeContext`.
- Fast and complex paths must use the same capability implementations.
- Only the outer adapter streams user-facing prose.
- Current runtime ACL replaces historical context on every resume.
- Before each existing-symbol edit run named GitNexus impact; before each commit run compare-scope detect-changes and narrow staging.

Canonical Phase-2 package layout:

```text
backend/app/services/agents/v2/
├── nodes/
│   ├── context.py
│   ├── binding.py
│   ├── routing.py
│   ├── fast_plan.py
│   ├── execute.py
│   ├── evaluate.py
│   ├── synthesize.py
│   ├── grounding.py
│   ├── clarification.py
│   └── finalizer.py
├── capabilities/
│   ├── people.py
│   ├── document.py
│   ├── section.py
│   ├── knowledge_graph.py
│   ├── abbreviation.py
│   └── memory.py
├── adapters/
│   ├── __init__.py
│   ├── semantic.py
│   ├── conversation.py
│   ├── document.py
│   └── deep_research.py
├── execution/
│   ├── __init__.py
│   └── scheduler.py
└── events.py
```

Do not create `context_graph.py`, `binding_graph.py`, `routing_graph.py`, `planning.py`, `evaluation.py`, `grounding.py`, `clarification.py`, or `v2/domain/*_graph.py`.

---

### Task 0: Verify Phase-2 Paths, Amendment Guards, Symbols, and Selected APIs

**Files:**
- Read: all Modify paths and imports named below
- Read: `docs/superpowers/plans/2026-09-11-langgraph-v2-agent-tool-node-amendment.md`
- Test: shell preflight only

**Interfaces:**
- Produces: phase manifest of exact symbols/imports and proves no old domain-graph layout exists.

- [ ] **Step 1: Verify repository paths and canonical module layout**

```bash
set -e
for path in backend/app/main.py backend/app/api/chat_session.py backend/app/api/chat_agent_lg.py backend/app/services/integrations/telegram_service.py backend/app/services/agent/streaming.py backend/app/core/config.py; do test -e "$path"; done

test ! -e backend/app/services/agents/v2/execution.py
for old in context_graph.py binding_graph.py routing_graph.py planning.py evaluation.py grounding.py clarification.py; do
  test ! -e "backend/app/services/agents/v2/$old"
done
test ! -d backend/app/services/agents/v2/domain
! rg -q 'run_agent_evaluation' backend/app
```

Expected: no conflicting old v2 implementation layout exists.

- [ ] **Step 2: Verify amendment static guards before implementation**

```bash
! find backend/app/services/agents/v2 -type f \
  \( -name 'people_agent.py' -o -name 'summary_agent.py' -o -name 'comparison_agent.py' \
     -o -name 'document_agent.py' -o -name 'section_agent.py' -o -name 'kg_agent.py' \
     -o -path '*/domain/*_graph.py' \) 2>/dev/null | grep .
```

Expected: exits 0.

- [ ] **Step 3: Verify selected dependency API**

```bash
cd backend && python - <<'PY'
from inspect import signature
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
import langgraph.graph.state as _lg_state
assert 'context_schema' in signature(StateGraph).parameters
assert hasattr(_lg_state, 'CompiledStateGraph')
print(AsyncPostgresSaver, InMemorySaver, interrupt, Command)
PY
```

Expected: Phase-0 selected APIs still exist.

- [ ] **Step 4: Verify Phase-1 readiness**

```bash
docker exec hrag-backend python -m app.services.agents.v2.persistence.migrate --check
docker exec hrag-backend python -m app.services.agents.v2.persistence.checkpoint --check
```

Expected: both exit 0.

---

### Task 1: Implement Context, Binding, Semantic Finalization, and Routing Nodes

**Files:**
- Create: `backend/app/services/agents/v2/nodes/__init__.py`
- Create: `backend/app/services/agents/v2/nodes/context.py`
- Create: `backend/app/services/agents/v2/nodes/binding.py`
- Create: `backend/app/services/agents/v2/nodes/routing.py`
- Test: `backend/tests/agents/v2/test_context_binding_routes.py`

**Interfaces:**
- Produces: `context_node`, `binding_node`, `semantic_finalizer_node`, `route_node`, plus pure helpers `build_semantic_draft`, `resolve_bindings`, `finalize_semantic`, `analyze_query`, `decide_route`.

- [ ] **Step 1: Write failing lifecycle/routing tests**

Test greeting/direct, People/fast, exact Section/fast, KG/fast, comparison/complex, cross-domain dependency/complex, required ambiguous document/clarify, Write typed unavailable, abbreviations, coreference follow-up, irrelevant attachment exclusion, ordinary/current/pinned revision behavior, prompt-injection content remaining data, and no domain-agent routing names.

```bash
cd backend && pytest tests/agents/v2/test_context_binding_routes.py -q
```

Expected: FAIL.

- [ ] **Step 2: Implement semantic lifecycle helpers**

```python
async def build_semantic_draft(
    request: RequestContext,
    conversation: ConversationContext,
    runtime: GraphRuntimeContext,
) -> SemanticDraft:
    return await runtime.services.semantic_adapter.build_draft(request, conversation)

async def resolve_bindings(
    draft: SemanticDraft,
    runtime: GraphRuntimeContext,
) -> DocumentBindingSet:
    return await runtime.services.binding_resolver.resolve(
        draft.document_refs,
        runtime.capability_runtime,
    )

async def finalize_semantic(
    draft: SemanticDraft,
    bindings: DocumentBindingSet,
) -> SemanticContext:
    return SemanticContext(
        contextualized_query=draft.provisional_contextualized_query,
        normalized_query=_normalize_validated(draft),
        abbreviations=draft.abbreviations,
        coreferences=draft.coreferences,
        document_refs=_apply_resolution(draft.document_refs, bindings),
        person_refs=draft.person_refs,
        section_refs=draft.section_refs,
        blocking_ambiguities=finalize_blocking_ambiguities(
            draft.preliminary_ambiguities,
            bindings,
        ),
    )
```

Define helpers in these node modules; none may widen runtime scope or copy binding IDs into semantic contracts.

- [ ] **Step 3: Implement deterministic-first routing**

```text
direct non-factual       -> direct
required ambiguity       -> clarify
one bounded operation    -> fast_domain
bounded one-doc summary  -> fast_domain
two-target comparison    -> complex_research
cross-domain dependency  -> complex_research
multi-goal/iterative RAG -> complex_research
Write                     -> typed unavailable
```

Routing may inspect request-scoped capability availability but cannot persist runtime permissions into business state.

- [ ] **Step 4: Implement node wrappers and test**

Each node reads only required state fields and returns a reducer-compatible update; domain business logic stays outside the node.

```bash
cd backend && pytest tests/agents/v2/test_context_binding_routes.py -q
```

- [ ] **Step 5: Commit**

```bash
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/nodes backend/tests/agents/v2/test_context_binding_routes.py
git commit -m "feat: add v2 context binding and routing nodes"
```

---

### Task 2: Implement Shared Atomic Capabilities and Request-Scoped Registry

**Files:**
- Modify: `backend/app/services/agents/v2/capabilities/__init__.py`
- Create: `backend/app/services/agents/v2/capabilities/people.py`
- Create: `backend/app/services/agents/v2/capabilities/document.py`
- Create: `backend/app/services/agents/v2/capabilities/section.py`
- Create: `backend/app/services/agents/v2/capabilities/knowledge_graph.py`
- Create: `backend/app/services/agents/v2/capabilities/abbreviation.py`
- Create: `backend/app/services/agents/v2/capabilities/memory.py`
- Modify: `backend/app/services/agents/v2/adapters/__init__.py`
- Test: `backend/tests/agents/v2/fast_paths/test_capabilities.py`
- Test: `backend/tests/agents/v2/test_adapters_and_registry.py`

**Interfaces:**
- Produces: atomic `Capability` implementations and a request-scoped registry filtered by current runtime permissions/feature/service availability.

- [ ] **Step 1: Write failing capability-boundary tests**

Named tests:

```text
test_capability_accepts_agent_request_and_capability_runtime_only
test_capability_cannot_read_supervisor_root_state
test_model_input_cannot_supply_workspace_or_acl
test_people_capability_enforces_current_people_permission
test_document_capability_requires_pinned_authorized_revision
test_section_read_emits_read_coverage_not_search_coverage
test_registry_intersects_runtime_permission_and_service_availability
test_v2_has_no_domain_agent_or_domain_graph_wrappers
```

```bash
cd backend && pytest tests/agents/v2/fast_paths/test_capabilities.py tests/agents/v2/test_adapters_and_registry.py -q
```

Expected: FAIL.

- [ ] **Step 2: Implement the single capability protocol**

Import and re-export the Phase-1 `Capability` protocol from `capabilities/__init__.py`; do not define a second one. `CapabilityDescriptor`, `CapabilityInput`, `CapabilityOutput`, and `CapabilityRuntimeContext` are imported from the frozen `contracts/capability.py`; no capability module may redefine or field-extend them.

```python
class Capability(Protocol):
    descriptor: CapabilityDescriptor

    async def execute(
        self,
        request: AgentRequest,
        runtime: CapabilityRuntimeContext,
    ) -> AgentResult: ...
```

Capabilities adapt existing v1 services/tools but return typed minimized v2 outputs/evidence uses. They never receive `GraphRuntimeContext` and never read supervisor state.

- [ ] **Step 3: Implement registry construction**

```text
base descriptors
∩ allowed_capabilities
∩ can_read_people/current ACL
∩ feature flags
∩ service health
= request-scoped CapabilityRegistry
```

Unknown/unavailable capability names fail closed.

- [ ] **Step 4: Run tests and commit**

```bash
cd backend && pytest tests/agents/v2/fast_paths/test_capabilities.py tests/agents/v2/test_adapters_and_registry.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/capabilities backend/app/services/agents/v2/adapters backend/tests/agents/v2/fast_paths/test_capabilities.py backend/tests/agents/v2/test_adapters_and_registry.py
git commit -m "feat: add shared v2 capabilities"
```

---

### Task 3: Build Deterministic Fast Plan Node and Shared Scheduler

**Files:**
- Create: `backend/app/services/agents/v2/nodes/fast_plan.py`
- Create: `backend/app/services/agents/v2/nodes/execute.py`
- Create: `backend/app/services/agents/v2/execution/__init__.py`
- Create: `backend/app/services/agents/v2/execution/scheduler.py`
- Test: `backend/tests/agents/v2/fast_paths/test_fast_plan.py`
- Test: `backend/tests/agents/v2/fast_paths/test_scheduler.py`

**Interfaces:**
- Produces: `build_fast_plan(...) -> TaskPlan`, `fast_plan_node`, `execute_ready_tasks(...)`, and `execute_node`.

- [ ] **Step 1: Write failing ownership/checkpoint tests**

Require People/KG plans to have one TaskSpec and zero TargetUnits; Section/Document plans have one TaskSpec and required TargetUnits; direct greeting has no plan; bounded summary uses document/section read rather than a summary agent; compare never routes fast; no planner model is called; every result/use/coverage resolves to checkpointed task/target IDs.

Add:

```text
test_task_must_be_checkpointed_before_capability_dispatch
test_fast_people_uses_shared_capability_registry
test_fast_document_read_uses_shared_capability_registry
```

- [ ] **Step 2: Implement deterministic fast plan**

```python
def build_fast_plan(
    semantic: SemanticContext,
    bindings: DocumentBindingSet,
    analysis: QueryAnalysis,
    route: RouteDecision,
) -> TaskPlan:
    if route.route != "fast_domain":
        raise ValueError("fast plan requires fast_domain route")
    task_id = _stable_task_id(semantic.normalized_query, route.reason_code, 0)
    targets = _build_required_targets(semantic, bindings, route)
    task = TaskSpec(
        task_id=task_id,
        capability=_capability_for(route, analysis),
        task_objective=semantic.normalized_query,
        input=_capability_input(task_id, targets, semantic),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id=_stable_plan_id(task_id),
        goal=semantic.normalized_query,
        target_units=targets,
        tasks=(task,),
    )
    validate_task_plan(plan, bindings)
    return plan
```

- [ ] **Step 3: Implement the only capability dispatch path**

`execute_ready_tasks` must:

```text
resolve ready TaskSpec
-> verify TaskSpec exists in authoritative checkpointed plan
-> registry.resolve(TaskSpec.capability)
-> build AgentRequest
-> call capability.execute(request, runtime.capability_runtime)
-> validate AgentResult.task_id/status/error/output/evidence uses
-> append immutable result
```

It stops before dispatch on cancellation/deadline. EvidenceUse insertion occurs only after TaskPlan checkpoint ownership is established.

- [ ] **Step 4: Implement execute node**

```python
async def execute_node(
    state: SupervisorV2State,
    runtime: GraphRuntimeContext,
) -> dict:
    plan = require_checkpointed_plan(state)
    results = await execute_ready_tasks(
        plan=plan,
        results=state.execution.task_results,
        registry=runtime.services.capability_registry,
        runtime=runtime,
    )
    return execution_update(results)
```

- [ ] **Step 5: Test and commit**

```bash
cd backend && pytest tests/agents/v2/fast_paths/test_fast_plan.py tests/agents/v2/fast_paths/test_scheduler.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/nodes/fast_plan.py backend/app/services/agents/v2/nodes/execute.py backend/app/services/agents/v2/execution backend/tests/agents/v2/fast_paths/test_fast_plan.py backend/tests/agents/v2/fast_paths/test_scheduler.py
git commit -m "feat: add deterministic v2 fast planning and scheduler"
```

---

### Task 4: Implement Evaluator, Synthesis, Grounding, and Finalizer Nodes

**Files:**
- Create: `backend/app/services/agents/v2/nodes/evaluate.py`
- Create: `backend/app/services/agents/v2/nodes/synthesize.py`
- Create: `backend/app/services/agents/v2/nodes/grounding.py`
- Create: `backend/app/services/agents/v2/nodes/finalizer.py`
- Test: `backend/tests/agents/v2/test_evaluation_grounding.py`
- Test: `backend/tests/agents/v2/fast_paths/test_domain_paths.py`

**Interfaces:**
- Produces: `evaluate_evidence`, `evaluate_node`, `hydrate_for_synthesis`, `synthesize_node`, `ground_answer`, `ground_node`, `finalizer_node`.

- [ ] **Step 1: Write failing evidence/grounding tests**

Cover search-not-read-coverage; wrong/partial section; revision mismatch; targetless supporting use; discovery use excluded from synthesis; source tombstone; expired use; People minimization; same record/two uses; derived faithfulness; all four evaluation statuses and precedence; unmapped factual assertion revise once then insufficient; deterministic citation; synthesis budget overflow persisted as validated derived evidence; synthesis-only reuse revalidates current uses; bounded summary performs read -> evaluate -> synthesize -> ground and `test_summary_is_skill_not_agent_route` (a bounded one-document summary routes through the shared document/section read capability plus synthesis, never a summary domain agent).

- [ ] **Step 2: Implement evaluator authority**

```python
async def evaluate_evidence(... ) -> EvidenceEvaluation:
    hydrated = await runtime.services.evidence_hydrator.hydrate_for_evaluation(...)
    coverage = build_coverage(plan, bindings, results, hydrated)
    missing = find_missing_requirements(plan, coverage)
    contradictions = await analyze_contradictions_bounded(hydrated)
    if semantic.blocking_ambiguities or has_needs_input_result(results):
        status = "needs_input"
    elif has_blocking_conflict(contradictions):
        status = "contradictory"
    elif missing:
        status = "insufficient"
    else:
        status = "sufficient"
    return EvidenceEvaluation(...)
```

Deterministic rules own coverage, revision/locator compatibility, ACL/expiry/source validity, criterion presence, and status precedence. A bounded model may assist semantic criteria/contradiction interpretation only over governed evidence; it cannot call tools or self-certify factual success.

- [ ] **Step 3: Implement synthesis and grounding**

Factual/domain synthesis requires `EvidenceEvaluation.status == "sufficient"`. Hydration resolves admitted `EvidenceUseRef` to ephemeral `SynthesisEvidence(use_id, content, role, target_id, source_label)` under current ACL/expiry/revision checks. Grounding requires every material factual assertion to map to admitted `AnswerClaim.evidence_use_ids`; revise once, otherwise emit insufficient.

- [ ] **Step 4: Implement finalizer**

Only direct non-factual or grounded factual paths may emit success. Write remains typed unavailable in v2.

- [ ] **Step 5: Test and commit**

```bash
cd backend && pytest tests/agents/v2/test_evaluation_grounding.py tests/agents/v2/fast_paths/test_domain_paths.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/nodes/evaluate.py backend/app/services/agents/v2/nodes/synthesize.py backend/app/services/agents/v2/nodes/grounding.py backend/app/services/agents/v2/nodes/finalizer.py backend/tests/agents/v2/test_evaluation_grounding.py backend/tests/agents/v2/fast_paths/test_domain_paths.py
git commit -m "feat: add v2 evaluation synthesis and grounding nodes"
```

---

### Task 5: Implement Clarification Interrupt and Resume Node

**Files:**
- Create: `backend/app/services/agents/v2/nodes/clarification.py`
- Test: `backend/tests/agents/v2/test_clarification_resume.py`

**Interfaces:**
- Produces: `build_clarification`, `clarify_node`, `interrupt_for_clarification`, `resume_clarification`.

- [ ] **Step 1: Write failing interrupt/resume tests**

Cover stable candidate UUID/order, question-specific ref IDs, expiry, invalid selection, selected candidate no longer authorized, current ACL replacement, raw reply loaded by `ChatMessage.id`, checkpoint without runtime secrets, and resolved flow restarting at Binding Resolver.

- [ ] **Step 2: Implement interrupt/resume**

```python
async def interrupt_for_clarification(request: ClarificationRequest) -> dict[str, object]:
    return interrupt({"clarification": request.model_dump(mode="json")})

async def resume_clarification(
    message_id: UUID,
    request: ClarificationRequest,
    runtime: GraphRuntimeContext,
) -> Command:
    message = await runtime.services.chat_messages.get_user_message(message_id)
    resolution = parse_clarification_resolution(message.content, request)
    if datetime.now(timezone.utc) >= request.expires_at:
        raise ClarificationExpired(request.clarification_id)
    candidate = _selected_candidate(resolution, request)
    await runtime.services.authorization.require_document(
        candidate.document_id,
        runtime.capability_runtime,
    )
    return Command(resume=resolution.model_dump(mode="json"), goto="binding")
```

- [ ] **Step 3: Test and commit**

```bash
cd backend && pytest tests/agents/v2/test_clarification_resume.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/nodes/clarification.py backend/tests/agents/v2/test_clarification_resume.py
git commit -m "feat: add v2 clarification node"
```

---

### Task 6: Compose `supervisor_v2` from Nodes Before Selector Wiring

**Files:**
- Create: `backend/app/services/agents/supervisor_v2.py`
- Create: `backend/app/services/agents/v2/events.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/agents/v2/test_supervisor_v2.py`
- Test: `backend/tests/agents/v2/test_supervisor_v2_lifespan.py`

**Interfaces:**
- Produces: `create_supervisor_v2_graph(checkpointer)`, lifecycle singleton helpers, and Phase-3 `complex_boundary` replacement seam.

- [ ] **Step 1: Write composition/lifespan tests**

Assert graph imports node functions from `v2/nodes`, contains context/binding/semantic_finalizer/route/direct/clarify/fast_plan/execute/evaluate/synthesize/ground/finalizer/complex_boundary, contains no domain-agent/subgraph modules, checkpoints fast TaskPlan before execute, and complex routes return typed unavailable until Phase 3.

- [ ] **Step 2: Compose graph**

```python
def create_supervisor_v2_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    graph = StateGraph(SupervisorV2State, context_schema=GraphRuntimeContext)
    graph.add_node("context", context_node)
    graph.add_node("binding", binding_node)
    graph.add_node("semantic_finalizer", semantic_finalizer_node)
    graph.add_node("route", route_node)
    graph.add_node("direct", direct_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("fast_plan", fast_plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("ground", ground_node)
    graph.add_node("finalizer", finalizer_node)
    graph.add_node("complex_boundary", complex_unavailable_node)
    graph.set_entry_point("context")
    _add_supervisor_edges(graph)
    return graph.compile(checkpointer=checkpointer)
```

Lifespan owns one opened AsyncPostgresSaver context; web startup never calls saver setup/migration.

- [ ] **Step 3: Test and commit**

```bash
cd backend && pytest tests/agents/v2/test_supervisor_v2.py tests/agents/v2/test_supervisor_v2_lifespan.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/supervisor_v2.py backend/app/services/agents/v2/events.py backend/app/main.py backend/tests/agents/v2/test_supervisor_v2.py backend/tests/agents/v2/test_supervisor_v2_lifespan.py
git commit -m "feat: compose node based supervisor v2"
```

---

### Task 7: Add Lazy Runtime Selector and Wire All Entrypoints

**Files:**
- Create: `backend/app/services/agent/runtime_selector.py`
- Create: `backend/app/api/agent_admin.py`
- Modify: `backend/app/api/router.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `backend/app/api/chat_session.py`
- Modify: `backend/app/api/chat_agent_lg.py`
- Modify: `backend/app/services/integrations/telegram_service.py`
- Test: `backend/tests/api/test_agent_runtime_selector.py`
- Test: `backend/tests/api/test_agent_v2_ingress.py`

**Interfaces:**
- Produces: lazy `resolve_agent_graph(version)`, schema readiness gate, authenticated admin-only evaluation selection.

- [ ] **Step 1: Write selector/ingress tests**

Assert default v1, invalid config fail-fast, lazy factories, no graph construction at import, standalone/session/Telegram all use resolver, v2 schema/checkpoint compatibility precedes graph selection, raw user text persists before normalization, current runtime scope is authenticated scope intersect requested scope, ordinary graph-version headers are ignored, admin evaluation override is the only per-request arm override.

- [ ] **Step 2: Implement lazy resolver**

```python
async def resolve_agent_graph(version: AgentGraphVersion):
    if version == "v1":
        from app.services.agents.supervisor import get_supervisor_graph
        return get_supervisor_graph()
    await require_v2_schema_ready()
    from app.services.agents.supervisor_v2 import get_supervisor_v2_graph
    return get_supervisor_v2_graph()
```

- [ ] **Step 3: Test and commit**

```bash
cd backend && pytest tests/api/test_agent_runtime_selector.py tests/api/test_agent_v2_ingress.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agent/runtime_selector.py backend/app/api/agent_admin.py backend/app/api/router.py backend/app/core/config.py .env.example backend/app/api/chat_session.py backend/app/api/chat_agent_lg.py backend/app/services/integrations/telegram_service.py backend/tests/api/test_agent_runtime_selector.py backend/tests/api/test_agent_v2_ingress.py
git commit -m "feat: add v1 v2 runtime selector"
```

---

### Task 8: Preserve SSE, Cancellation, and Resume Compatibility

**Files:**
- Modify: `backend/app/services/agent/streaming.py`
- Test: `backend/tests/api/test_agent_v2_streaming.py`
- Modify: `frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx`
- Modify: `frontend/src/hooks/useRAGChatStream.ts` only if a test proves incompatibility

**Interfaces:**
- Produces: existing SSE contract, one terminal event, cancellation propagation into scheduler/capability tokens, stable checkpoint thread resume.

- [ ] **Step 1: Write compatibility tests**

Assert status/thinking/sources/images/token/token_rollback/potential_abbreviations/people_data/error/complete event compatibility; only outer final prose streams; one terminal event; rollback clears accumulators; disconnect/cancel prevents success; stable thread ID resumes; current ACL is reinjected.

- [ ] **Step 2: Implement event adapter behind existing formatter**

Do not change frontend hook unless the contract test proves mismatch. Cancellation must prevent all later scheduler dispatches and factual success.

- [ ] **Step 3: Test and commit**

```bash
cd backend && pytest tests/api/test_agent_v2_streaming.py tests/agents/v2/test_clarification_resume.py -q
cd frontend && pnpm test -- ChatPanel.rollback
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agent/streaming.py backend/tests/api/test_agent_v2_streaming.py frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx frontend/src/hooks/useRAGChatStream.ts
git diff --cached --check
git commit -m "feat: preserve v2 stream and resume compatibility"
```

## Phase-2 Acceptance Gate

Required proofs include the frozen-contract tests plus:

```text
test_v2_has_no_domain_agent_or_domain_graph_wrappers
test_fast_people_uses_shared_capability_registry
test_fast_document_read_uses_shared_capability_registry
test_capability_receives_capability_runtime_context_only
test_capability_cannot_read_supervisor_root_state
test_model_input_cannot_supply_workspace_or_acl
test_bounded_summary_is_read_evaluate_synthesize_ground
test_compare_never_routes_to_fast_domain
test_task_must_be_checkpointed_before_capability_dispatch
```

Static guards:

```bash
set -e
! find backend/app/services/agents/v2 -type f \
  \( -name 'people_agent.py' -o -name 'summary_agent.py' -o -name 'comparison_agent.py' \
     -o -name 'document_agent.py' -o -name 'section_agent.py' -o -name 'kg_agent.py' \
     -o -path '*/domain/*_graph.py' \) | grep .

! rg -n \
  'class\s+(People|Summary|Comparison|Document|Section|KG|Evaluation|Grounding).*Agent|\b(People|Summary|Comparison|Document|Section|KG)Agent\b' \
  backend/app/services/agents/v2
```

Whole Phase-2 validation:

```bash
docker exec hrag-backend pytest tests/agents/v2 tests/api/test_agent_runtime_selector.py tests/api/test_agent_v2_ingress.py tests/api/test_agent_v2_streaming.py -q
docker exec hrag-backend pytest tests/agents tests/migrations tests/services -q
make test-recall
make test-section
make test-validity
make fe-lint
make fe-build
node .gitnexus/run.cjs analyze
```

Expected: all pass with v1 still production default and complex research still unavailable until Phase 3.

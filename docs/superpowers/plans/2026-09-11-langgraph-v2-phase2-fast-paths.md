# LangGraph v2 Phase 2 Fast Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the independent v2 supervisor and validated direct, clarification, and deterministic fast paths while preserving all external API/SSE behavior and keeping v1 default.

**Architecture:** Context runs Draft→Binding Resolver→Semantic Finalizer before deterministic analysis/routing. Every factual fast route creates one checkpointed TaskPlan, shared execution/evaluation/evidence/grounding handles the task, and only then is a lazy external selector wired into all three runtime entrypoints.

**Tech Stack:** Python 3.11, LangGraph Phase-0 winner, Pydantic v2, FastAPI, PostgreSQL checkpointer, pytest, SSE, React/Vitest.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- Phase 1 gate must pass; v1 remains default.
- Create `supervisor_v2.py` before selector wiring.
- Only direct non-factual conversation bypasses factual evidence/grounding. Write (pasted-text grammar/proofread/rewrite) is explicitly **outside the LangGraph v2 rollout scope**: v1 remains its owner until a separate approved implementation plan defines non-evidence success semantics. v2 keeps a typed unavailable boundary, and no phase in this suite enables Write.
- Only the outer adapter streams final prose.
- Current runtime ACL replaces historical context on every resume.
- Before each existing-symbol edit run named GitNexus impact; before each commit run compare-scope detect-changes and narrow staging.

---

### Task 0: Verify Phase-2 Paths, Symbols, and Selected APIs

**Files:**
- Read: all Modify paths and imports named below
- Test: shell preflight only

**Interfaces:**
- Produces: phase manifest of exact symbols/imports; no runtime code.

- [ ] **Step 1: Verify repository paths and module layout**

```bash
set -e
for path in backend/app/main.py backend/app/api/chat_session.py backend/app/api/chat_agent_lg.py backend/app/services/integrations/telegram_service.py backend/app/services/agent/streaming.py backend/app/core/config.py; do test -e "$path"; done
test ! -e backend/app/services/agents/v2/execution.py
test ! -d backend/app/services/agents/v2/execution
! rg -q 'run_agent_evaluation' backend/app
python - <<'PY'
import re, pathlib
plan = pathlib.Path('docs/superpowers/plans/2026-09-11-langgraph-v2-phase2-fast-paths.md').read_text()
modify = [p.split(':')[0] for p in re.findall(r'^- Modify: `([^`]+)`', plan, re.M)]
create = [p.split(':')[0] for p in re.findall(r'^- Create: `([^`]+)`', plan, re.M)]
missing = [p for p in modify if not pathlib.Path(p).exists()]
conflict = [p for p in create if pathlib.Path(p).exists()]
assert not missing and not conflict, {'missing': missing, 'conflict': conflict}
print(f'phase2 paths ok: {len(set(modify))} modify, {len(set(create))} create')
PY
rg -n 'get_supervisor_graph|stream_agent_events|chat_stream_session|langgraph_chat_stream' backend/app
```

Expected: Modify paths exist; `execution/` package path has no conflicting `execution.py`; `run_agent_evaluation` is still undefined until Task 6; record exact symbols for impacts.

- [ ] **Step 2: Verify selected dependency API**

```bash
cd backend && python - <<'PY'
from inspect import signature
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import interrupt, Command
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
import langgraph.graph.state as _lg_state
assert 'context_schema' in signature(StateGraph).parameters
assert hasattr(_lg_state, 'CompiledStateGraph')
print(AsyncPostgresSaver)
PY
```

Expected: Phase-0 API proof still passes; stop on dependency drift.

- [ ] **Step 3: Verify Phase-1 readiness commands**

```bash
docker exec hrag-backend python -m app.services.agents.v2.persistence.migrate --check
docker exec hrag-backend python -m app.services.agents.v2.persistence.checkpoint --check
```

Expected: both exit 0 before Phase 2 edits.

---

### Task 1: Implement Context, Binding, Finalization, and Routing Services

**Files:**
- Create: `backend/app/services/agents/v2/context_graph.py`
- Create: `backend/app/services/agents/v2/binding_graph.py`
- Create: `backend/app/services/agents/v2/routing_graph.py`
- Create: `backend/tests/agents/v2/test_context_binding_routes.py`

**Interfaces:**
- Produces: `build_semantic_draft`, `resolve_bindings`, `finalize_semantic`, `analyze_query`, `decide_route`.

- [ ] **Step 1: Write failing lifecycle/routing tests**

Test greeting/direct, People/fast, exact Section/fast, KG/fast, comparison/complex, required ambiguous document/clarify, grammar Write returning the declared out-of-scope typed unavailable response, abbreviations, coreference follow-up, irrelevant attachment exclusion, ordinary/current/pinned revision behavior, and prompt-injection content remaining data.

```bash
cd backend && pytest tests/agents/v2/test_context_binding_routes.py -q
```

Expected: FAIL.

- [ ] **Step 2: Implement the exact semantic lifecycle**

```python
async def build_semantic_draft(request: RequestContext, conversation: ConversationContext, runtime: GraphRuntimeContext) -> SemanticDraft:
    return await runtime.services.semantic_adapter.build_draft(request, conversation)

async def resolve_bindings(draft: SemanticDraft, runtime: GraphRuntimeContext) -> DocumentBindingSet:
    return await runtime.services.binding_resolver.resolve(draft.document_refs, runtime.capability_runtime)

async def finalize_semantic(draft: SemanticDraft, bindings: DocumentBindingSet) -> SemanticContext:
    return SemanticContext(
        contextualized_query=draft.provisional_contextualized_query,
        normalized_query=_normalize_validated(draft),
        abbreviations=draft.abbreviations,
        coreferences=draft.coreferences,
        document_refs=_apply_resolution(draft.document_refs, bindings),
        person_refs=draft.person_refs,
        section_refs=draft.section_refs,
        blocking_ambiguities=finalize_blocking_ambiguities(draft.preliminary_ambiguities, bindings),
    )
```

Define `_normalize_validated`, `_apply_resolution`, and `finalize_blocking_ambiguities` in `context_graph.py`; neither can widen scope or write binding IDs into SemanticContext. Tests prove an ambiguity resolved by binding is removed, while a remaining required ambiguity stays blocking and routes to clarify; preliminary ambiguities are never copied blindly.

- [ ] **Step 3: Implement deterministic-first analyzer/router**

Router emits direct only for non-factual conversation, clarify for required ambiguity, fast for one bounded domain operation, and complex for comparison/dependency/multi-goal/research. Capability availability is checked against the request-scoped registry; semantic complexity is not persisted.

```bash
cd backend && pytest tests/agents/v2/test_context_binding_routes.py -q
```

Expected: all lifecycle and routing cases pass.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/agents/v2/test_context_binding_routes.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/context_graph.py backend/app/services/agents/v2/binding_graph.py backend/app/services/agents/v2/routing_graph.py backend/tests/agents/v2/test_context_binding_routes.py
git commit -m "feat: add v2 context binding and routing"
```

---

### Task 2: Build Deterministic Fast Plans and Shared Scheduler

**Files:**
- Create: `backend/app/services/agents/v2/planning.py`
- Create: `backend/app/services/agents/v2/execution/__init__.py`
- Create: `backend/app/services/agents/v2/execution/scheduler.py`
- Create: `backend/tests/agents/v2/fast_paths/test_fast_plan.py`
- Create: `backend/tests/agents/v2/fast_paths/test_scheduler.py`

**Interfaces:**
- Produces: `build_fast_plan(semantic: SemanticContext, bindings: DocumentBindingSet, analysis: QueryAnalysis, route: RouteDecision) -> TaskPlan` and `execute_ready_tasks(plan: TaskPlan, results: tuple[AgentResult, ...], registry: CapabilityRegistry, runtime: GraphRuntimeContext) -> tuple[AgentResult, ...]`.

- [ ] **Step 1: Write failing ownership tests**

Require People/KG plans to have one TaskSpec and zero TargetUnits; Section/Document plans have one TaskSpec and one or more TargetUnits; direct greeting has no plan; no planner model is called. Resume must resolve each result/use/coverage task/target through the checkpointed plan.

```bash
cd backend && pytest tests/agents/v2/fast_paths/test_fast_plan.py tests/agents/v2/fast_paths/test_scheduler.py -q
```

Expected: FAIL before planning and scheduler modules exist.

- [ ] **Step 2: Implement deterministic IDs and plan**

```python
def build_fast_plan(semantic: SemanticContext, bindings: DocumentBindingSet, analysis: QueryAnalysis, route: RouteDecision) -> TaskPlan:
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
    plan = TaskPlan(contract_version="2.0", plan_id=_stable_plan_id(task_id), goal=semantic.normalized_query, target_units=targets, tasks=(task,))
    validate_task_plan(plan, bindings)
    return plan
```

Define all underscored helpers in the module and test stable uniqueness.

- [ ] **Step 3: Implement scheduler association/cancellation**

`execute_ready_tasks` resolves TaskSpec capability, constructs AgentRequest without copied capability name, invokes with current runtime, validates AgentResult.task_id, stops dispatch after cancellation/deadline, and appends immutable results. It writes EvidenceUse only after owning TaskPlan checkpoint succeeds.

```bash
cd backend && pytest tests/agents/v2/fast_paths/test_scheduler.py -q
```

Expected: scheduler association and cancellation tests pass.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/agents/v2/fast_paths/test_fast_plan.py tests/agents/v2/fast_paths/test_scheduler.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/planning.py backend/app/services/agents/v2/execution/__init__.py backend/app/services/agents/v2/execution/scheduler.py backend/tests/agents/v2/fast_paths
git commit -m "feat: add deterministic v2 fast planning"
```

---

### Task 3: Implement Evidence Evaluation, Hydration, Grounding, and Domain Fast Graphs

**Files:**
- Create: `backend/app/services/agents/v2/evaluation.py`
- Create: `backend/app/services/agents/v2/grounding.py`
- Create: `backend/app/services/agents/v2/domain/people_graph.py`
- Create: `backend/app/services/agents/v2/domain/document_graph.py`
- Create: `backend/app/services/agents/v2/domain/section_graph.py`
- Create: `backend/app/services/agents/v2/domain/knowledge_graph.py`
- Create: `backend/tests/agents/v2/fast_paths/test_domain_paths.py`
- Create: `backend/tests/agents/v2/test_evaluation_grounding.py`

**Interfaces:**
- Produces: `evaluate_evidence`, `hydrate_for_synthesis`, `ground_answer`, `render_final_response`.

- [ ] **Step 1: Write failing evidence/grounding tests**

Name tests for: search success not read coverage; wrong/partial section; revision mismatch; targetless supporting use; discovery use cannot synthesize; source tombstone; expired use; People minimization; same record/two uses; derived faithfulness; all four statuses and precedence; unmapped factual assertion revise then insufficient; deterministic citation; `test_synthesis_budget_overflow_persists_grounded_derived_evidence`; and `test_synthesis_only_existing_evidence_revalidates_current_uses_without_research`. Keep pasted-text Write disabled in v2 until a separate plan owns it; this suite does not invent that exception.

```bash
cd backend && pytest tests/agents/v2/fast_paths/test_domain_paths.py tests/agents/v2/test_evaluation_grounding.py -q
```

Expected: FAIL before evaluation, grounding, and domain modules exist.

- [ ] **Step 2: Implement deterministic evaluation**

```python
async def evaluate_evidence(semantic: SemanticContext, plan: TaskPlan, bindings: DocumentBindingSet, results: tuple[AgentResult, ...], use_refs: tuple[EvidenceUseRef, ...], runtime: GraphRuntimeContext) -> EvidenceEvaluation:
    hydrated = await runtime.services.evidence_hydrator.hydrate_for_evaluation(use_refs, plan, bindings, runtime.capability_runtime)
    coverage = build_coverage(plan, bindings, results, hydrated)
    missing = find_missing_requirements(plan, coverage)
    contradictions = analyze_contradictions(hydrated)
    if semantic.blocking_ambiguities or has_needs_input_result(results):
        status = "needs_input"
    elif has_blocking_conflict(contradictions):
        status = "contradictory"
    elif missing:
        status = "insufficient"
    else:
        status = "sufficient"
    return EvidenceEvaluation(status=status, coverage=coverage, missing=missing, contradictions=contradictions)
```

Define each helper in the same module. Governed hydration resolves EvidenceUse→EvidenceRecord and checks current ACL, retention, source availability, revision, locator, and derived validation before contradiction analysis; EvidenceUse identity alone is never analyzed as content. Deterministic precedence is `needs_input` → `contradictory` → `insufficient` → `sufficient`. Search never emits read coverage.

- [ ] **Step 3: Implement authorized hydration and use-bound claims**

Hydrator accepts current TaskPlan, bindings, runtime, and synthesis budget separately; resolves EvidenceUse→EvidenceRecord; rechecks ACL/expiry/revision/source/locator; excludes discovery uses; validates derived faithfulness; returns ephemeral `SynthesisEvidence(use_id, content, role, target_id, source_label)`. Grounding requires every material factual assertion to map to AnswerClaim evidence_use_ids admitted by this call.

```bash
cd backend && pytest tests/agents/v2/test_evaluation_grounding.py -q -k 'hydrate or grounding or citation'
```

Expected: hydration and grounding tests pass.

- [ ] **Step 4: Implement independent domain graphs**

Each enabled domain graph accepts typed input and returns typed output without reading root state. People, Document, Section, and KG are enabled in this phase. The router returns a typed unavailable response for Write, which is declared out of scope for this rollout with v1 as owner; do not implement a plan-only business-contract exception. Source-backed Write, if ever enabled by a separate approved plan, follows normal evaluation/grounding.

```bash
cd backend && pytest tests/agents/v2/fast_paths/test_domain_paths.py -q
```

Expected: all five domain paths pass.

- [ ] **Step 5: Run and commit**

```bash
cd backend && pytest tests/agents/v2/fast_paths/test_domain_paths.py tests/agents/v2/test_evaluation_grounding.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/evaluation.py backend/app/services/agents/v2/grounding.py backend/app/services/agents/v2/domain backend/tests/agents/v2/fast_paths/test_domain_paths.py backend/tests/agents/v2/test_evaluation_grounding.py
git commit -m "feat: add v2 fast domain evaluation and grounding"
```

---

### Task 4: Implement Clarification Interrupt and Resume

**Files:**
- Create: `backend/app/services/agents/v2/clarification.py`
- Create: `backend/tests/agents/v2/test_clarification_resume.py`

**Interfaces:**
- Produces: `build_clarification`, `interrupt_for_clarification`, `resume_clarification`.

- [ ] **Step 1: Write failing interrupt/resume tests**

Cover stable candidate UUID/order, question-specific ref IDs, expiry, invalid selection, selected candidate no longer authorized, current ACL replacement, raw reply loaded by `ChatMessage.id` from chat DB, `ClarificationResolution` mapped into the interrupted checkpoint command, checkpoint without runtime secrets, and resolved flow restarting at Binding Resolver.

```bash
cd backend && pytest tests/agents/v2/test_clarification_resume.py -q
```

Expected: FAIL before clarification module exists.

- [ ] **Step 2: Implement interrupt and validation**

```python
async def interrupt_for_clarification(request: ClarificationRequest) -> dict[str, object]:
    return interrupt({"clarification": request.model_dump(mode="json")})

async def resume_clarification(message_id: UUID, request: ClarificationRequest, runtime: GraphRuntimeContext) -> Command:
    message = await runtime.services.chat_messages.get_user_message(message_id)
    resolution = parse_clarification_resolution(message.content, request)
    if datetime.now(timezone.utc) >= request.expires_at:
        raise ClarificationExpired(request.clarification_id)
    candidate = _selected_candidate(resolution, request)
    await runtime.services.authorization.require_document(candidate.document_id, runtime.capability_runtime)
    return Command(resume=resolution.model_dump(mode="json"), goto="binding")
```

Define `ClarificationExpired` and `_selected_candidate`; resume returns to binding resolution, never directly trusts old bindings.

- [ ] **Step 3: Run and commit**

```bash
cd backend && pytest tests/agents/v2/test_clarification_resume.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/clarification.py backend/tests/agents/v2/test_clarification_resume.py
git commit -m "feat: add v2 clarification resume"
```

---

### Task 5: Compose supervisor_v2 Before Creating the Selector

**Files:**
- Create: `backend/app/services/agents/supervisor_v2.py`
- Create: `backend/app/services/agents/v2/events.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/agents/v2/test_supervisor_v2.py`
- Create: `backend/tests/agents/v2/test_supervisor_v2_lifespan.py`

**Interfaces:**
- Produces: `create_supervisor_v2_graph(checkpointer)`, `initialize_supervisor_v2_graph(checkpointer)`, `get_supervisor_v2_graph()`, and `close_supervisor_v2_graph()`; `app.main.lifespan` owns exactly one opened AsyncPostgresSaver context for the process lifetime.

- [ ] **Step 1: Impact-check the existing lifespan owner**

```bash
impact({target: "app.main.lifespan", direction: "upstream"})
```

- [ ] **Step 2: Write composition and lifespan tests**

Assert module imports without v1; node names include context, binding, semantic_finalizer, route, direct, clarify, fast_plan, execute, evaluate, ground, finalizer, and complex_boundary; every route has a terminal path; business functions are imported rather than defined in `supervisor_v2.py`; checkpointer is passed to compile; factual fast state checkpoints plan before execution; and complex routes return a typed `COMPLEX_RESEARCH_UNAVAILABLE` response until Phase 3. Lifespan tests monkeypatch `create_v2_checkpointer` with an async context manager and prove enter/open once → compatibility check → `initialize_supervisor_v2_graph(saver)` → serve → `close_supervisor_v2_graph()` → context exit/connection close once. `saver.setup()` is never called by web startup because the deployment command owns checkpointer DDL. Startup failure closes the saver and publishes no singleton; `get_supervisor_v2_graph()` before initialization raises a typed readiness error.

```bash
cd backend && pytest tests/agents/v2/test_supervisor_v2.py -q
```

Expected: FAIL before supervisor_v2 exists.

- [ ] **Step 3: Compose nodes and bind the saver to application lifespan**

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
    graph.add_node("ground", ground_node)
    graph.add_node("finalizer", finalizer_node)
    graph.add_node("complex_boundary", complex_unavailable_node)
    graph.set_entry_point("context")
    _add_supervisor_edges(graph)
    return graph.compile(checkpointer=checkpointer)
```

`app.main.lifespan` owns `async with create_v2_checkpointer(settings.CHECKPOINT_DATABASE_URL) as saver:` for the entire serving lifetime. After `check_v2_checkpoint_schema(saver)` passes, it calls `initialize_supervisor_v2_graph(saver)`; shutdown first rejects new v2 resolution, awaits active graph calls, calls `close_supervisor_v2_graph()`, then exits the saver context. `get_supervisor_v2_graph()` returns only the initialized compiled singleton and never opens a saver itself. Define `_add_supervisor_edges` in this module. `direct` and grounded paths converge on `finalizer`. `complex_boundary` is a real selected edge whose node emits a typed non-success `COMPLEX_RESEARCH_UNAVAILABLE` FinalResponse until Phase 3 replaces only that node implementation; the route is never missing and never throws an unhandled disabled exception.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/agents/v2/test_supervisor_v2.py tests/agents/v2/test_supervisor_v2_lifespan.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/supervisor_v2.py backend/app/services/agents/v2/events.py backend/app/main.py backend/tests/agents/v2/test_supervisor_v2.py backend/tests/agents/v2/test_supervisor_v2_lifespan.py
git commit -m "feat: compose independent supervisor v2"
node .gitnexus/run.cjs analyze
```

---

### Task 6: Add Lazy Runtime Selector and Wire All Entrypoints

**Files:**
- Create: `backend/app/services/agent/runtime_selector.py`
- Create: `backend/app/api/agent_admin.py`
- Modify: `backend/app/api/router.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `backend/app/api/chat_session.py`
- Modify: `backend/app/api/chat_agent_lg.py`
- Modify: `backend/app/services/integrations/telegram_service.py`
- Create: `backend/tests/api/test_agent_runtime_selector.py`
- Create: `backend/tests/api/test_agent_v2_ingress.py`

**Interfaces:**
- Produces: lazy async `resolve_agent_graph(version)`, `require_v2_schema_ready()`, version-safe ingress adapters, and authenticated admin-only evaluation selection.

- [ ] **Step 1: Impact-check all production seams**

```bash
impact({target: "app.core.config.Settings", direction: "upstream"})
impact({target: "app.api.chat_session.chat_stream_session", direction: "upstream"})
impact({target: "app.api.chat_agent_lg.langgraph_chat_stream", direction: "upstream"})
impact({target: "app.services.integrations.telegram_service._handle_question", direction: "upstream"})
```

Stop for HIGH/CRITICAL risk pending review.

- [ ] **Step 2: Write selector tests**

Assert default v1, invalid config fail-fast, factories imported lazily, no graph constructed at module import, standalone/session/Telegram use `resolve_agent_graph`, v2 schema/checkpoint compatibility is checked before any configured selection, v2 raw text persists before normalization, and request scope is authenticated scope intersect requested scope. Assert ordinary client `X-Agent-Graph-Version` is ignored. Assert only an authenticated admin principal may call the evaluation endpoint that explicitly selects v1/v2; non-admin receives 403. Import `from app.api.agent_admin import run_agent_evaluation` and assert the admin route is bound to that handler so the Phase-3 impact target is stable.

```bash
cd backend && pytest tests/api/test_agent_runtime_selector.py tests/api/test_agent_v2_ingress.py -q
```

Expected: FAIL before resolver and wiring exist.

- [ ] **Step 3: Implement lazy factories**

```python
AgentGraphVersion = Literal["v1", "v2"]

async def resolve_agent_graph(version: AgentGraphVersion):
    if version == "v1":
        from app.services.agents.supervisor import get_supervisor_graph
        return get_supervisor_graph()
    await require_v2_schema_ready()
    from app.services.agents.supervisor_v2 import get_supervisor_v2_graph
    return await get_supervisor_v2_graph()
```

Settings validates `NEXUSRAG_AGENT_GRAPH_VERSION` and defaults to `v1`. `resolve_agent_graph("v2")` first awaits `require_v2_schema_ready()` for both v2 model tables and AsyncPostgresSaver tables. Entrypoints await the resolver only after authentication/raw message persistence and build current runtime context separately. They never read a client graph-version header. `POST /api/admin/agent-evaluation/run` is served by `async def run_agent_evaluation(...)` in `backend/app/api/agent_admin.py` (this exact name is the Phase-3 impact target). It uses the existing admin authorization dependency, accepts a validated server-side arm, performs the same schema check for v2, and is the sole per-request arm override used by the Phase-3 A/B driver.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/api/test_agent_runtime_selector.py tests/api/test_agent_v2_ingress.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agent/runtime_selector.py backend/app/api/agent_admin.py backend/app/api/router.py backend/app/core/config.py .env.example backend/app/api/chat_session.py backend/app/api/chat_agent_lg.py backend/app/services/integrations/telegram_service.py backend/tests/api/test_agent_runtime_selector.py backend/tests/api/test_agent_v2_ingress.py
git commit -m "feat: add v1 v2 runtime selector"
```

---

### Task 7: Preserve SSE, Cancellation, and Resume Compatibility

**Files:**
- Create: `backend/tests/api/test_agent_v2_streaming.py`
- Modify: `backend/app/services/agent/streaming.py`
- Modify: `frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx`
- Modify: `frontend/src/hooks/useRAGChatStream.ts` only if the test proves incompatibility

**Interfaces:**
- Produces: same named SSE contract with one terminal event and cancellation propagation.

- [ ] **Step 1: Impact-check streaming symbol**

```bash
impact({target: "app.services.agent.streaming.stream_agent_events", direction: "upstream"})
impact({target: "app.services.agent.streaming.stream_agent_to_sse", direction: "upstream"})
```

- [ ] **Step 2: Write compatibility tests**

Assert status/thinking/sources/images/token/token_rollback/potential_abbreviations/people_data/error/complete names and payload keys, only outer answer tokens, one terminal event, rollback clears accumulators, disconnect/cancel prevents success, stable thread_id resumes checkpoint, and current ACL is reinjected.

```bash
cd backend && pytest tests/api/test_agent_v2_streaming.py -q
```

Expected: FAIL before v2 event adaptation exists.

- [ ] **Step 3: Add a v2 event adapter behind existing formatting**

Do not change frontend hook unless a failing contract test proves a mismatch. Propagate stop-button cancellation to graph task and capability cancellation tokens; cancellation emits typed error/terminal behavior and no later capability dispatch.

```bash
cd backend && pytest tests/api/test_agent_v2_streaming.py -q
cd frontend && pnpm test -- ChatPanel.rollback
```

Expected: backend and frontend compatibility tests pass.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/api/test_agent_v2_streaming.py tests/agents/v2/test_clarification_resume.py -q
cd frontend && pnpm test -- ChatPanel.rollback
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/tests/api/test_agent_v2_streaming.py backend/app/services/agent/streaming.py frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx frontend/src/hooks/useRAGChatStream.ts
git diff --cached --check
git commit -m "feat: preserve v2 stream and resume compatibility"
```

## Named §26 Acceptance Gate

Create or retain named tests for Phase-2 scenarios only: strict root/nested versioning; factual fast checkpoint ownership; binding/target/task/evidence/use integrity; sensitive-output exclusion; no identity duplication; typed capability unions; criterion identity; search-not-coverage; ordinary/current/pinned revisions; relation integrity; revision mismatch; cross-run reuse; purpose/target rules; People minimization; synthesis sufficient gate; same-record/two-uses; unmapped assertion; derived faithfulness; deterministic citation; direct greeting; runtime exclusion; raw query ownership; unresolved projections; prompt injection; incompatible checkpoint; `test_synthesis_budget_overflow_persists_grounded_derived_evidence`; and `test_synthesis_only_existing_evidence_revalidates_current_uses_without_research`. People→Document and replan acceptance remain exclusively in Phase 3.

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

Expected: all pass with `NEXUSRAG_AGENT_GRAPH_VERSION=v1` still default.

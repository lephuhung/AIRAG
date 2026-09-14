# LangGraph v2 Phase 4 — V1 Intelligence, V2 Contracts & Adaptive Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the proven semantic/intent/document-resolution intelligence of `LLM-Optimize`, translate it into frozen v2 contracts, achieve fast-path parity, synchronize the public frontend transport contract, then add governed adaptive planning only for genuinely complex execution topologies.

**Architecture:** V1 remains the semantic intelligence source for simple-query understanding, document identity, intent, contextualization, and fast-path semantics. A request-scoped adapter/cache translates those outputs into `SemanticDraft`/`SemanticContext`/`QueryAnalysis`; v2 remains authoritative for ACL, immutable revision binding, deterministic route policy, planning validation, scheduling, evidence, checkpointing, and grounding. A separate public transport adapter maps internal v2 results to versioned UI-safe DTO/SSE events so the frontend never depends directly on checkpoint/runtime contracts.

**Tech Stack:** Python 3, LangGraph, Pydantic v2, SQLAlchemy async, existing AIRAG LLM providers/runtime config, Qdrant/vector search, pytest, React/TypeScript, SSE chat transport, frontend component/hook tests.

**Spec:** `docs/superpowers/specs/2026-09-14-langgraph-v2-semantic-adaptive-execution-design.md`

## Global Constraints

- Reuse the existing v1 `app.services.agent.doc_resolver.resolve_candidates()` behavior; do not clone a second resolver implementation.
- Preserve v1 fast-path semantic quality where already proven.
- V1 legacy `SupervisorState`, `next_agent`, `pending_intent`, mutable `document_ids`, and agent-to-agent routing do not become v2 contracts.
- Preserve frozen v2 `SemanticContext`, `QueryAnalysis`, `RouteDecision`, `TaskPlan`, `TaskSpec`, `AgentResult`, ACL, binding, scheduler, evidence, and checkpoint boundaries.
- `RouteDecision` is deterministic v2 policy. Semantic models never directly choose route/capability/tool.
- Document resolver candidates are server-side identity candidates; v2 binding remains the sole immutable revision-pin authority.
- Whether an LLM was needed to understand a query is not a complexity signal.
- Frontend never consumes raw `SupervisorV2State`, runtime services, checkpoint payloads, hidden model reasoning, planner prompts, or internal evidence-store identifiers.
- Public chat/SSE DTOs are an explicit compatibility boundary and must support v1/v2 canary operation.
- Planner work starts only after v1-v2 fast-path parity **and core frontend transport compatibility** pass.
- Reuse existing LLM runtime role configuration; do not add separate router/planner provider stacks.
- One scheduler, one supervisor checkpointer, one capability registry.

---

## Task 0 — Add explicit semantic-router and planner LLM roles

**Priority:** P0 prerequisite

**Files:**
- Modify: `backend/app/services/runtime_config.py`
- Modify: `backend/app/services/llm/__init__.py`
- Modify: `backend/app/api/llm_config.py` only if inherited-role metadata requires it
- Modify: `frontend/src/types/llmConfig.ts`
- Modify: `frontend/src/pages/AdminLLMConfigPage.tsx`
- Modify: `frontend/src/lib/translations/vi.json`
- Modify: `frontend/src/lib/translations/en.json`
- Test: runtime-config/provider/API/UI tests

**Interfaces:**
- Produces: `get_semantic_router_provider()` and `get_planner_provider()`.
- Preserves: `get_thinking_provider()` behavior.

- [ ] Add `semantic_router` and `planner` to runtime roles.
- [ ] Implement fallback to the **effective** `thinking` role, including DB override inheritance.
- [ ] Keep explicit per-role assignment higher priority than inherited thinking.
- [ ] Refactor current reasoning-provider construction into one shared role-aware factory with distinct cache/tracing labels.
- [ ] Add the two roles to Admin UI types/metadata/translations.
- [ ] Ensure GET config reports truthful inherited effective connection/model semantics.
- [ ] Audit control-plane LLM callsites and record intentional role ownership; abbreviation/contextualization/document semantic calls must not accidentally use the main answer model.
- [ ] Run runtime-config/provider/API tests.
- [ ] Commit: `feat(v2): add semantic router and planner llm roles`.

**Acceptance:** changing the `thinking` DB assignment updates unassigned semantic-router/planner effective configs after refresh; explicit assignments remain independent.

---

## Task 1 — Build the v1 fast-path golden parity corpus before changing behavior

**Priority:** P0 gate foundation

**Files:**
- Create: `backend/tests/agents/v2/golden/test_v1_intelligence_parity.py`
- Create: `backend/tests/agents/v2/golden/fast_path_cases.py` or equivalent fixture data
- Read: `backend/app/prompts/agents/supervisor_scope.py`
- Read: `backend/app/services/agents/supervisor.py`
- Read: `backend/app/services/agent/doc_resolver.py`
- Read: `backend/app/services/agents/resolve_doc_agent.py`

**Interfaces:**
- Produces: versioned expected semantic/intent/document-resolution cases used by Tasks 2–6.

- [ ] Add pure greeting and greeting-prefix factual cases.
- [ ] Add People cases: phone, CCCD, BHXH, name, `Nguyễn Văn A là ai?`.
- [ ] Add general factual RAG cases with no explicit document.
- [ ] Add exact title and approximate title cases.
- [ ] Add full official number cases.
- [ ] Add bare/short number cases such as `luật số 24`, `thông tư 15`.
- [ ] Add number + issuing-agency cases such as `Thông tư 15 của Bộ Công an`.
- [ ] Add year + topic and topic-only remembered-document cases.
- [ ] Add exact section and nested `Điểm/Khoản/Điều` cases.
- [ ] Add one-document summarize and simple KG cases.
- [ ] Add ambiguous candidates and low-confidence target cases.
- [ ] Add follow-up/coreference sequences.
- [ ] Capture expected v1 intent/resolution behavior and expected v2 final topology separately.
- [ ] Run the corpus against v1 helpers to prove the fixture reflects current `LLM-Optimize` behavior.
- [ ] Commit: `test(v2): capture v1 fast path intelligence parity`.

**Gate:** later tasks may improve behavior, but must not silently regress cases already passing in v1.

---

## Task 2 — Add request-scoped IntelligenceService/cache boundary

**Priority:** P0 prerequisite for resolver/model wiring

**Files:**
- Modify: `backend/app/services/agents/v2/contracts/state.py` (`RuntimeServices` only)
- Create: `backend/app/services/agents/v2/semantic/intelligence.py`
- Modify: `backend/app/services/agents/supervisor_v2.py`
- Modify: runtime construction owner as needed
- Test: new runtime-only intelligence tests

**Interfaces:**
- Produces conceptually:

```python
class IntentDecision: ...
class DocumentIdentityResolution: ...
class IntelligenceService:
    async def classify_intent(...): ...
    async def resolve_document(...): ...
```

- Runtime-only; no new `SupervisorV2State` slot.

- [ ] Add one request-scoped runtime service that caches intent/document/semantic inference for the current turn.
- [ ] Key cache by current request/query plus relevant conversation semantic context.
- [ ] Prove cache cannot cross user/workspace/turn/shadow-live boundaries.
- [ ] Prove the service is not serialized into checkpoints.
- [ ] Ensure repeated `build_semantic_draft()` calls in binding and semantic-finalizer paths reuse cached expensive work.
- [ ] Add cancellation/deadline handling.
- [ ] Commit: `feat(v2): add request scoped semantic intelligence service`.

**Acceptance:** one turn may rebuild drafts multiple times, but the same document-resolution/semantic model inference executes at most once per cache key.

---

## Task 3 — Adapt the proven v1 document resolver into v2 document identity

**Priority:** P0

**Files:**
- Create: `backend/app/services/agents/v2/semantic/document_identity.py`
- Reuse: `backend/app/services/agent/doc_resolver.py`
- Modify: `backend/app/services/agents/v2/adapters/semantic.py`
- Modify: `backend/app/services/agents/supervisor_v2.py` runtime wiring
- Test: focused document identity + binding tests

**Interfaces:**
- Consumes: `resolve_candidates(reference, workspace_ids, db, topic=full_query, ...)`.
- Produces: v2 `DocumentReference` states: `resolved`, `ambiguous`, `not_found`; never a legacy state update.

- [ ] Write failing tests for exact number, short number, number+agency, approximate title, topic-only fallback, ambiguous candidates, low confidence, and explicit-number identity dominance over topic rerank.
- [ ] Wrap `resolve_candidates()` rather than copying `_extract_by_regex`, `_query_db`, vector, fuzzy, or rerank implementations.
- [ ] Pass the full contextualized question as `topic` so the existing vector/rerank behavior is preserved.
- [ ] Translate clear winners to `DocumentReference(resolved_document_id=...)`.
- [ ] Translate close candidates to `candidate_document_ids` + `resolution_status="ambiguous"`.
- [ ] Do not force-bind low-confidence candidates.
- [ ] Keep current runtime workspace/authorization scope authoritative; model text can never mint trusted workspace IDs.
- [ ] Preserve the section hint returned by the resolver for later `SectionReference` construction.
- [ ] Feed the resulting v2 reference into the existing binding resolver; do not pin revisions in the identity adapter.
- [ ] Run golden document cases plus binding regression tests.
- [ ] Commit: `feat(v2): adapt v1 document resolver to v2 identities`.

**Acceptance:** resolving a document may internally use DB → LLM → vector → rerank, yet a one-document/one-section user request remains eligible for fast path.

---

## Task 4 — Translate v1 intent intelligence into V2 semantic facts and QueryAnalysis

**Priority:** P0

**Files:**
- Create: `backend/app/services/agents/v2/semantic/intent.py`
- Reuse/read: `backend/app/prompts/agents/supervisor_scope.py`
- Reuse/read: v1 supervisor classification prompt/taxonomy in `backend/app/services/agents/supervisor.py`
- Modify: `backend/app/services/agents/v2/adapters/semantic.py`
- Modify: `backend/app/services/agents/v2/nodes/routing.py`
- Test: intent translation + routing tests

**Interfaces:**
- Consumes: v1 narrow deterministic scope behavior and semantic-router model for uncertain/full cases.
- Produces: runtime-only `IntentDecision`, translated to existing `QueryAnalysis`.

- [ ] Keep deterministic v1 short-circuit behavior for unambiguous greeting/personal/People cases.
- [ ] Reuse the v1 intent taxonomy for uncertain/full cases; do not import `next_agent` as a v2 decision.
- [ ] Map simple v1 intents into v2 semantics:
  - `mongo_search_*` → `lookup/people`
  - `search` → `retrieve/document`
  - `search_doc_num` → `retrieve/document`
  - `search_section` → `retrieve/document+section`
  - one-doc `summarize` → `summarize/document`
  - `kg_query` → `lookup/knowledge_graph`
  - greeting/personal → direct/memory semantics
- [ ] Treat `resolve_doc` as a semantic prerequisite to identity resolution, not an execution-complexity signal.
- [ ] Remove generic keyword/regex patterns such as `là ai`, `khác biệt`, `tổng hợp`, `đánh giá` from being authoritative work-type decisions where v1 semantic intent is available.
- [ ] Keep regex/patterns only as conservative safety nets or narrow deterministic scopes.
- [ ] Do not add confidence to frozen `QueryAnalysis`.
- [ ] Run People/KG/general-RAG/greeting-prefix golden cases.
- [ ] Commit: `feat(v2): translate v1 intent intelligence into v2 analysis`.

**Acceptance:** `Nguyễn Văn A là ai?` is not classified as KG solely due to `là ai`; `sự khác biệt giữa nghỉ phép và nghỉ ốm` does not become a complex document comparison unless semantic intent/targets require it.

---

## Task 5 — Complete SemanticDraft translation: section, person, coreference, conversation focus

**Priority:** P0

**Files:**
- Modify: `backend/app/services/agents/v2/adapters/semantic.py`
- Modify: `backend/app/services/agents/v2/adapters/conversation.py`
- Modify: `backend/app/services/agents/v2/nodes/context.py` where focus persistence belongs
- Create focused semantic/coreference tests

**Interfaces:**
- Consumes: cached intelligence results from Tasks 2–4.
- Produces: complete v2 `SemanticDraft` / finalized `SemanticContext`.

- [ ] Remove unconditional `coreferences=()`, `person_refs=()`, `section_refs=()` information loss.
- [ ] Build `SectionReference` from trusted resolver/semantic output, including nested `Điểm/Khoản/Điều` locators where supported.
- [ ] Populate People/person refs from validated People intent/identifiers.
- [ ] Preserve typed active document/person/section entities in conversation context.
- [ ] Populate `last_focus` from validated completed turns.
- [ ] Resolve unambiguous `văn bản này`, `nghị định trên`, `điều này`, `ông ấy`, `file thứ hai` against current authorized conversation entities.
- [ ] Emit clarification for true ambiguity rather than guessing.
- [ ] Ensure history cannot reauthorize a document outside current runtime scope.
- [ ] Run multi-turn golden sequences.
- [ ] Commit: `feat(v2): complete semantic and discourse translation`.

---

## Task 6 — Add deterministic Simple Eligibility Gate and achieve v1 fast-path parity

**Priority:** P0 — hard gate before Planner

**Files:**
- Modify: `backend/app/services/agents/v2/nodes/routing.py`
- Optionally create: `backend/app/services/agents/v2/semantic/execution_shape.py`
- Modify: `backend/tests/agents/v2/fast_paths/test_domain_paths.py`
- Extend: golden parity tests

**Interfaces:**
- Consumes: finalized `SemanticContext`, translated `QueryAnalysis`, bindings, capability availability.
- Produces: deterministic v2 `RouteDecision`.

- [ ] Define runtime execution shape/simple-eligibility from logical evidence operations, not internal helper-call count.
- [ ] Fast eligibility includes known bounded workflows: People lookup, general factual retrieve, one document, one section, search-by-number after resolution, one-doc bounded summarize, simple KG.
- [ ] Document identity resolution complexity must not force `complex_research`.
- [ ] A semantic-router LLM call must not itself force `complex_research`.
- [ ] Route to Planner/complex only for query-specific DAG/dependency needs: multi-target compare, evaluate/compliance, cross-domain dependency, multi-goal, iterative evidence completion.
- [ ] Keep `decide_route()` free of LLM calls.
- [ ] Run complete Task-1 parity corpus comparing v1 semantic result and expected v2 topology.
- [ ] Record semantic-router call counts and assert zero where deterministic v1 narrow scopes suffice.
- [ ] Commit: `feat(v2): restore fast path parity with simple eligibility gate`.

**Hard acceptance gate:** v2 simple-query behavior must meet or exceed `LLM-Optimize` on the versioned corpus before Planner begins.

---

## Task 6A — Synchronize V2 public transport contract with the frontend

**Priority:** P0 — frontend compatibility gate before Planner/canary

**Why:** the current frontend is centered on legacy chat types (`ChatMessage`, `ChatSourceChunk`, `AgentStep`) and `useRAGChatStream` parses legacy-style SSE `status/token/source/...` events. It does not yet model v2 clarification/resume, richer terminal states, or complex execution progress. The solution is **not** to expose `SupervisorV2State`; create an explicit public chat transport boundary.

**Files — Backend:**
- Create or modify the reviewed v2 chat/SSE transport adapter that owns `stream_v2_turn_events` / `stream_v2_turn_to_sse`.
- Create: `backend/app/services/agents/v2/transport/contracts.py` or equivalent public DTO owner.
- Modify persistence/serialization owner for assistant chat metadata only where public metadata must survive reload.
- Add backend transport-contract tests.

**Files — Frontend:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/hooks/useRAGChatStream.ts`
- Modify chat panel/message/timeline components that render clarification/progress/citations.
- Modify: `frontend/src/hooks/useChatHistory.ts` and history hydration adapters if persisted metadata changes.
- Add frontend hook/component tests for SSE parsing and reload behavior.

**Interfaces:**
- Consumes internal v2: `ClarificationRequest`, `RouteDecision`, safe progress summaries from `TaskPlan`/`AgentResult`, governed citations, `FinalResponse`.
- Produces a UI-safe versioned/discriminated public event union. Conceptual minimum:

```ts
type ChatTransportEvent =
  | { version: "2"; type: "status"; phase: ChatPhase; detail?: string }
  | { version: "2"; type: "clarification_required"; request: PublicClarification }
  | { version: "2"; type: "clarification_resolved"; requestId: string }
  | { version: "2"; type: "source"; source: PublicCitationSource }
  | { version: "2"; type: "token"; text: string }
  | { version: "2"; type: "complete"; response: PublicFinalResponse }
  | { version: "2"; type: "error"; message: string; retryable: boolean }
  | { version: "2"; type: "cancelled" };
```

Exact naming may follow existing endpoint conventions, but it must be discriminated and validated rather than an untyped open payload.

- [ ] Inventory every SSE event currently emitted by v1 and v2 and every frontend branch that consumes it.
- [ ] Define the public transport DTO/event contract separately from checkpoint/business contracts.
- [ ] Preserve v1-compatible events or add a normalization adapter so canary users receive one frontend shape regardless of serving arm.
- [ ] Add `clarifying`, `planning`, `executing`, `evaluating`, `generating`, `error/cancelled` UI phases as needed; do not expose hidden reasoning.
- [ ] Represent clarification as structured data: `clarification_id/request_id`, prompt, allowed choices, safe labels/metadata, expiry if applicable.
- [ ] Add frontend action for selecting a clarification option and resuming the same suspended v2 thread/run through the reviewed backend resume endpoint.
- [ ] Do not let the frontend manufacture document UUIDs, workspace scope, binding IDs, or arbitrary clarification values; submit only server-issued option identifiers/allowed values.
- [ ] Normalize v1/v2 citations/sources to one TypeScript presentation type.
- [ ] Persist only public message metadata needed for reload; prove a page refresh reconstructs clarification/completed message state without reading checkpoints.
- [ ] Update `useRAGChatStream` to tolerate/ignore unknown forward-compatible events instead of crashing the stream.
- [ ] Add hook tests for fragmented SSE frames, duplicate/idempotent events, unknown events, cancel/error, clarification interrupt/resume, and normal completion.
- [ ] Add browser/component tests for:
  - simple RAG + citations;
  - exact-section fast path;
  - People lookup;
  - ambiguous document → choices → resume → answer;
  - low-confidence/not-found document state;
  - refresh after completion;
  - refresh while clarification is outstanding where product behavior supports resume.
- [ ] Add backend/frontend contract fixture tests so event-field drift fails CI.
- [ ] Commit: `feat(v2): synchronize chat transport contract with frontend`.

**Security boundary:** public transport may expose user-safe labels, candidate option identifiers, citation metadata, and progress summaries. It must not expose ACL internals, raw checkpoint state, planner prompts, hidden CoT, internal evidence-use UUIDs unless they are explicitly approved public citation handles, or runtime service/config secrets.

**Hard acceptance gate:** fast-path v2 is not canary-ready until actual web UI behavior passes the same simple/clarification golden scenarios as backend Task 6.

---

## Task 7 — Integrate the governed Adaptive Planner for true complex queries

**Priority:** P1 after Tasks 6 and 6A gates

**Supersedes/absorbs:** `docs/superpowers/plans/2026-09-13-langgraph-v2-p2-adaptive-planner.md`

**Files:**
- Create: `backend/app/services/agents/v2/planning/__init__.py`
- Create: `backend/app/services/agents/v2/planning/adapter.py`
- Modify: `backend/app/services/agents/v2/contracts/state.py` (`RuntimeServices.planner` only)
- Modify: `backend/app/services/agents/v2/complex_research_graph.py`
- Consume: `get_planner_provider()`
- Test: `backend/tests/agents/v2/complex/test_adaptive_planner.py`

**Interfaces:**
- Consumes: existing `ResearchPlanningInput`, finalized v2 semantics/bindings/capability catalogue.
- Produces: proposal-only typed planner output, converted to validated `TaskPlan`.

- [ ] Build a minimized/redacted model projection from existing planning inputs.
- [ ] Exclude ACL internals, raw evidence, secrets, runtime clients, and unnecessary internal evidence UUIDs from model input.
- [ ] Implement strict structured initial-plan proposal.
- [ ] Validate proposal through existing task-plan/scope/capability/budget checks.
- [ ] Acquire leases/checkpoint only after validation and before scheduler dispatch.
- [ ] Fall back to existing deterministic policy when planner is disabled/times out/returns malformed or rejected output and a safe deterministic policy exists.
- [ ] Never let Planner mint trusted document identity or widen hard scope.
- [ ] Emit only public summarized planning/execution progress through Task 6A transport; never stream the raw internal DAG/checkpoint to the browser.
- [ ] Commit: `feat(v2): add governed adaptive planner`.

---

## Task 8 — Project evaluator gaps into bounded append-only replan

**Priority:** P1

**Files:**
- Modify: `backend/app/services/agents/v2/complex_research_graph.py`
- Modify: `backend/app/services/agents/v2/replanning.py`
- Modify: `backend/app/services/agents/v2/nodes/evaluate.py` only if projection helper belongs there
- Test: replan/evaluation suites

**Interfaces:**
- Reuse: existing `EvidenceEvaluation`, `MissingRequirement`, `Contradiction`, `TaskExecutionSummary`.
- Do not create a parallel persisted planner-gap contract.

- [ ] Project only minimized typed evaluator facts to the planner.
- [ ] Support bounded gaps such as missing target coverage, insufficient evidence, failed dependency, supporting/reference discovery, and contradictions requiring one bounded additional read.
- [ ] Keep replans append-only; completed tasks cannot be replaced.
- [ ] Enforce max tasks, max replans, budgets, capability availability, and hard scope.
- [ ] Expose only coarse public `evaluating/replanning` progress through the frontend transport; no raw evaluator gaps are required in UI unless separately designed.
- [ ] Keep default rollout conservative until termination/quality tests pass.
- [ ] Commit: `feat(v2): add evaluator driven bounded replanning`.

---

## Task 9 — Expand unsupported complex work types only after planner/replan quality passes

**Priority:** P1/P2

**Order:**

1. `multi_goal` composed from supported atomic capabilities.
2. generic `cross_domain` beyond current People → Document vertical slice.
3. `evaluate` / compliance with trusted typed criteria and grounded evidence.

- [ ] Add real graph tests with causal dependencies for each work type.
- [ ] Add transport/UI E2E coverage for any new user-visible progress/clarification state introduced by each work type.
- [ ] Keep complex execution as validated DAGs; do not introduce an unconstrained ReAct loop.
- [ ] Commit each independently reviewable work type separately.

---

## Task 10 — Observability, shadow parity, frontend compatibility, and rollout

**Priority:** continuous + final gate

**Files:**
- Modify existing rollout metrics/shadow runtime files identified during implementation
- Extend golden replay tooling if useful
- Extend frontend contract/E2E test fixtures from Task 6A

- [ ] Add semantic-intelligence metrics: deterministic bypass, model call count, document resolver stages used, ambiguity/low-confidence rate, latency.
- [ ] Add v1-v2 parity counters on shadow replay.
- [ ] Add Planner metrics: calls, latency, proposal rejection reason, fallback, task count, DAG depth, replan count, final evidence sufficiency.
- [ ] Add public transport schema/version/error metrics without logging sensitive payloads.
- [ ] Keep planner-proposal shadow distinct from full read-only execution shadow.
- [ ] Extend shadow support for document read/retrieve parity before claiming full planner execution quality.
- [ ] Run backend contract + frontend SSE/UI compatibility suites before every canary promotion.
- [ ] Canary through existing v1/v2 selector with v1 preserved as rollback.
- [ ] Promote only after fast-path parity, frontend compatibility, contract safety, planner quality, and grounded-answer gates pass.

---

## Final Acceptance Criteria

Phase 4 is complete only when all of the following are true:

- v1 document-number/approximate-title/topic-based resolution behavior is available behind v2 adapters;
- v1 simple-intent strengths are preserved or improved;
- `Điều 5 Luật An ninh mạng quy định gì?` resolves document + section and routes to fast bounded execution without Planner;
- approximate references such as `Điều 5 luật an ninh`, `luật số 24`, or `Thông tư 15 của Bộ Công an` can resolve through the proven multi-stage resolver without making the user query complex;
- People/general RAG/section/summarize/KG fast paths meet or exceed `LLM-Optimize` golden behavior;
- v1 legacy state/routing fields do not leak into v2 contracts;
- frontend does not consume raw `SupervisorV2State`/checkpoint/runtime contracts;
- v1 and v2 chat arms normalize into a stable public frontend transport shape during canary;
- clarification is structured, selectable, resumable, and survives supported history/reload flows;
- citations/sources and terminal status render consistently for both fast and complex v2 turns;
- actual web E2E passes simple RAG, exact section, People, ambiguity/clarification, not-found/low-confidence, cancel/error, and reload scenarios;
- current v2 ACL, revision binding, scheduler, checkpoint, evidence, and grounding invariants remain intact;
- true multi-step queries alone enter adaptive planning;
- semantic-router/planner model assignment is independently configurable through existing Admin LLM Runtime Config;
- shadow/canary gates pass with v1 still available as rollback.

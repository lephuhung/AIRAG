# LangGraph v2 — V1 Intelligence to V2 Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the two proven v1 intelligence capabilities explicitly requested for v2 — **route/intent classification** and **document identity/number resolution** — behind v2 typed contracts, restore simple fast-path behavior, then complete discourse/frontend compatibility and only afterwards expand adaptive complex execution.

**Architecture:** Do not interpret “reuse the v1 brain” as “port the v1 supervisor.” The immediate migration surface is narrow: v1 decides semantic intent/work type and resolves remembered document identity; an adapter translates those results into v2 `SemanticContext`/`DocumentReference`/`QueryAnalysis`. V2 alone owns `RouteDecision`, ACL, revision binding, capability availability, planning, scheduling, evidence, checkpoints, and grounding.

**Tech Stack:** Python 3, LangGraph, Pydantic v2, SQLAlchemy async, existing AIRAG LLM providers/runtime config, Qdrant/vector search, pytest, React/TypeScript, SSE chat transport.

**Spec:** `docs/superpowers/specs/2026-09-14-langgraph-v2-semantic-adaptive-execution-design.md`

---

## 0. Scope boundary — read this before implementing any task

### What “move the v1 brain to v2” means in this plan

**Immediate migration — required:**

1. **V1 route/intent intelligence**
   - greeting/direct vs factual;
   - People lookup;
   - general RAG search;
   - search by document number;
   - search section;
   - one-document summarize;
   - KG lookup;
   - resolve-document as a semantic prerequisite.

2. **V1 document identity/number intelligence**
   - official/full number;
   - short/bare number;
   - number + issuing agency;
   - approximate/partial title;
   - topic/year-based recollection;
   - existing DB → LLM → vector → rerank/fuzzy behavior from `resolve_candidates()`.

**Not part of the V1-intent migration itself:**

- legacy `next_agent` / `pending_intent` / `SupervisorState`;
- v1 task plans;
- v1 agent-to-agent routing;
- ACL/binding/checkpoint behavior;
- full conversation/coreference migration;
- Planner/replan;
- frontend transport.

Those items have their own later phases below. They may be rollout dependencies, but they are **not acceptance criteria for proving that V1 intent was transferred correctly**.

### Canonical ownership

```text
V1 intelligence
  intent/work type
  document identity candidates
          │
          ▼
V1 -> V2 adapters
          │
          ▼
V2 contracts
  DocumentReference
  SemanticContext
  QueryAnalysis
          │
          ▼
V2 deterministic policy
  RouteDecision
          │
    ┌─────┴─────┐
    ▼           ▼
 Fast Path   Complex boundary
```

**Invariant:** V1/model output says **what the query means**. V2 says **what the system may execute**.

---

# Phase 4A — Route/Intent Intelligence Migration

This phase proves that v2 understands simple queries at least as well as `LLM-Optimize`. It does **not** require document fuzzy resolution, coreference, frontend clarification, or Planner to be considered complete.

## Task 0 — Add explicit semantic-router and planner LLM roles

**Priority:** P0 shared prerequisite

**Files:**
- Modify: `backend/app/services/runtime_config.py`
- Modify: `backend/app/services/llm/__init__.py`
- Modify: `backend/app/api/llm_config.py` only if effective inheritance metadata requires it
- Modify: `frontend/src/types/llmConfig.ts`
- Modify: `frontend/src/pages/AdminLLMConfigPage.tsx`
- Modify i18n files
- Test runtime-role/provider/API/UI behavior

**Required behavior:**

- [ ] Add `semantic_router` and `planner` roles.
- [ ] Unassigned roles inherit the **effective** `thinking` connection/model including DB overrides.
- [ ] Explicit assignment overrides inheritance.
- [ ] Use one provider factory; do not hardcode a separate OpenAI-only stack.
- [ ] Keep `get_thinking_provider()` backward compatible.
- [ ] Audit semantic/control-plane LLM callsites so role ownership is deliberate.
- [ ] Commit: `feat(v2): add semantic router and planner llm roles`.

---

## Task 1 — Freeze a route-intent parity corpus from V1

**Priority:** P0

**Files:**
- Create: `backend/tests/agents/v2/golden/intent_cases.py`
- Create: `backend/tests/agents/v2/golden/test_v1_intent_parity.py`
- Read/reuse: `backend/app/prompts/agents/supervisor_scope.py`
- Read/reuse: v1 supervisor classification prompt/taxonomy

**Cases required:**

```text
xin chào                                  -> greeting/direct
chào anh, hỏi về chế độ thai sản?         -> factual search, NOT greeting
0901234567 là ai?                          -> people lookup
079012345678 CCCD này của ai?             -> people lookup
Nguyễn Văn A là ai?                        -> people lookup
Tìm ông Nguyễn Văn A                      -> people lookup
chế độ thai sản được quy định thế nào?    -> general document retrieve FAST
sự khác biệt giữa nghỉ phép và nghỉ ốm?   -> general factual retrieve unless explicit multi-target research
Tóm tắt Nghị định A                       -> one-doc bounded summarize when one target is resolved
<simple KG query>                          -> knowledge_graph lookup
```

- [ ] Record expected **v1 semantic intent**, not `next_agent`.
- [ ] Record separately the expected **v2 QueryAnalysis + topology**.
- [ ] Prove fixtures represent current v1 behavior before changing v2.
- [ ] Commit: `test(v2): capture v1 route intent parity`.

---

## Task 2 — Add runtime-only typed `IntentDecision` adapter/cache

**Priority:** P0

**Files:**
- Modify: `backend/app/services/agents/v2/contracts/state.py` (`RuntimeServices` only)
- Create: `backend/app/services/agents/v2/semantic/intent.py`
- Modify runtime construction/wiring owner
- Add unit tests

**Internal interface:**

```python
class IntentDecision:
    intent: str
    source: Literal["deterministic", "model"]
    confidence: float | None
    needs_memory: bool = False
    is_legal_query: bool = False
```

- [ ] Runtime-only; never add `IntentDecision` to checkpoint state.
- [ ] Keep v1 deterministic narrow scopes for obvious greeting/People cases.
- [ ] For uncertain/full cases, reuse v1 taxonomy/prompt behavior through `semantic_router` role.
- [ ] Cache the result per request/turn so repeated semantic draft builds do not repeat classification.
- [ ] Do not return/import `next_agent`, `pending_intent`, or v1 task plans.
- [ ] Commit: `feat(v2): add typed v1 intent adapter`.

---

## Task 3 — Translate V1 intent taxonomy into V2 `QueryAnalysis`

**Priority:** P0

**Files:**
- Modify: `backend/app/services/agents/v2/nodes/routing.py`
- Modify semantic adapter only where it consumes `IntentDecision`
- Add mapping/routing tests

**Required mapping:**

```text
mongo_search_* -> work_type=lookup,    domains=(people,)
search         -> work_type=retrieve,  domains=(document,)
search_doc_num -> work_type=retrieve,  domains=(document,)
search_section -> work_type=retrieve,  domains=(document, section)
summarize      -> work_type=summarize, domains=(document,)
kg_query       -> work_type=lookup,    domains=(knowledge_graph,)
greeting       -> work_type=direct
resolve_doc    -> semantic prerequisite, NOT complexity
```

- [ ] V1 intent is a semantic input only; V2 still calls deterministic `decide_route()`.
- [ ] Remove generic regex/keywords such as `là ai`, `khác biệt`, `tổng hợp`, `đánh giá` from being authoritative when typed intent is available.
- [ ] Do not add confidence to frozen `QueryAnalysis`.
- [ ] Pass all Task-1 cases.
- [ ] Commit: `feat(v2): translate v1 intent into query analysis`.

### Task 3 erratum (final re-review M8): v2-only narrow `evaluate` deterministic scope

The Task-3 ruling above ("Remove generic regex/keywords such as … `đánh giá`
from being authoritative when typed intent is available") has one explicit,
v2-only exception. A **compliance/legal assessment request** is recognized
deterministically before any model call because the v1 taxonomy has no
`evaluate` intent — without it, a compliance question would be
model-classified to `search` and silently degrade to the targetless
document-retrieval fast path (final review I1).

Cue rule (deterministic, v2-only, never added to the shared v1 taxonomy
prompt):

- an explicit assessment head (`đánh giá`/`kiểm tra`/`rà soát`/`thẩm định`/
  `đối chiếu`/`xác định`/`review`/`assess`/`evaluate`) **combined with** a
  compliance/legal domain cue (`tuân thủ`/`compliance`/`tính pháp lý`/
  `pháp lý`); or
- an explicit degree/level phrase (`mức độ tuân thủ`/`mức độ rủi ro`); or
- the yes/no form (`…tuân thủ … không?`).

A bare compliance-topic noun (`quy định về tuân thủ thuế là gì?`, `hồ sơ
 tuân thủ gồm gì?`, `chế tài khi không tuân thủ`) is **not** an assessment
request and keeps `retrieve/document → fast_domain/targetless_document_retrieval`.
The bare `đánh giá` keyword remains demoted.

Why deterministic: route authority stays with `decide_route`; the scope only
supplies an advisory typed `IntentDecision(intent="evaluate")`, exactly like
the greeting/personal/people narrow scopes.

Cost if wrong: over-broad cues would hijack ordinary informational
compliance-topic queries off the general-RAG fast path into the
compliance-assessment topology (planner model call, or typed-unavailable
when no binding/planner); over-narrow cues would let a compliance question
fall back to targetless retrieval (the final-review I1 regression).

---

## Task 4 — Restore simple execution topology for general factual RAG

**Priority:** P0 — Phase 4A gate

**Files:**
- Modify: `backend/app/services/agents/v2/nodes/routing.py`
- Modify: `backend/app/services/agents/v2/nodes/fast_plan.py`
- Modify capability-catalog availability checks
- Extend fast-path tests

**Required behavior:**

- [ ] `search -> retrieve/document` without an explicit document binding is a bounded **targetless document retrieval** fast path where policy allows workspace search.
- [ ] Add/repair `document.retrieve` fast-plan mapping for reference-free factual RAG.
- [ ] Do not require exactly one binding for general document retrieval.
- [ ] Route decisions must check **actual available capability catalog**, not only `allowed_capabilities` requested/granted flags.
- [ ] Missing runtime capability gets a deterministic fallback/unavailable decision; do not route into an execution path that cannot exist.
- [ ] A model classification call itself never makes a query complex.
- [ ] Commit: `feat(v2): restore general rag fast path`.

### Phase 4A acceptance gate

Phase 4A is complete when all Task-1 route-intent cases produce the correct `QueryAnalysis` and bounded topology. In particular:

```text
Nguyễn Văn A là ai?                       -> people fast path
chào anh, hỏi về chế độ thai sản?         -> factual retrieve fast path
chế độ thai sản được quy định thế nào?    -> targetless document.retrieve fast path
```

**Do not block this gate on:** fuzzy document resolution, conversation focus, frontend clarification UI, or Planner.

---

# Phase 4B — Document Identity / Number Migration

This phase implements the second v1 capability explicitly requested: robust resolution when users remember a document number/title only approximately.

## Task 5 — Freeze V1 document-resolution parity cases

**Priority:** P0

**Files:**
- Create: `backend/tests/agents/v2/golden/document_identity_cases.py`
- Create: `backend/tests/agents/v2/golden/test_v1_document_identity_parity.py`
- Reuse: `backend/app/services/agent/doc_resolver.py`

**Cases:**

- [ ] full official number;
- [ ] short/bare number (`luật số 24`, `thông tư 15`);
- [ ] number + issuing agency (`Thông tư 15 của Bộ Công an`);
- [ ] exact title;
- [ ] approximate/partial title;
- [ ] year + topic;
- [ ] topic-only remembered document;
- [ ] explicit number identity must dominate topic rerank;
- [ ] ambiguous candidates;
- [ ] low-confidence/no candidate.

- [ ] Record candidate IDs/scores/status expected from v1 behavior where stable enough for a fixture.
- [ ] Commit: `test(v2): capture v1 document identity parity`.

---

## Task 6 — Adapt `resolve_candidates()` into V2 `DocumentReference`

**Priority:** P0

**Files:**
- Create: `backend/app/services/agents/v2/semantic/document_identity.py`
- Reuse directly: `backend/app/services/agent/doc_resolver.py`
- Modify: `backend/app/services/agents/v2/adapters/semantic.py`
- Modify runtime service/cache wiring
- Add binding integration tests

**Required behavior:**

- [ ] Wrap `resolve_candidates()`; do not clone its regex/SQL/LLM/vector/fuzzy/rerank code.
- [ ] Pass the full contextualized user question as `topic`.
- [ ] Translate a clear winner to `DocumentReference(resolution_status="resolved", resolved_document_id=...)`.
- [ ] Translate close candidates to `resolution_status="ambiguous"` + candidate IDs.
- [ ] Keep not-found as not-found; do not fabricate an identity.
- [ ] Do not force-bind low-confidence candidates.
- [ ] Resolver output remains inside current authorized workspace scope.
- [ ] Existing v2 binding resolver remains the only revision-pin authority.
- [ ] Cache expensive resolver/model/vector work per request.
- [ ] Commit: `feat(v2): adapt v1 document resolver to v2 contract`.

---

## Task 7 — Bridge the minimum section signal needed for simple section fast path

**Priority:** P0 for the canonical section query; this is **not** the full discourse/coreference task.

**Files:**
- Modify semantic adapter/context finalization as needed
- Add section fast-path tests

**Required behavior:**

- [ ] Preserve resolver/preprocessor `section_reference` instead of dropping it.
- [ ] Translate a simple authoritative `Điều/Chương/Khoản` locator into the existing v2 section contract shape required by fast routing.
- [ ] Preserve candidate document IDs across preprocessing persistence/translation.
- [ ] Do not implement full multi-turn coreference/focus in this task.
- [ ] If `section.read` requires a revision-specific `structure_node_id`, either resolve that locator deterministically here/at capability boundary or use a bounded document-retrieve fallback; do not pretend the capability is available when it will fail closed.
- [ ] Commit: `feat(v2): bridge section locator for fast retrieval`.

### Phase 4B acceptance gate

The canonical case must be:

```text
"Điều 5 Luật An ninh mạng quy định gì?"
    -> V1-compatible document identity resolution
    -> V2 DocumentReference + section locator
    -> V2 immutable binding
    -> QueryAnalysis(retrieve, document+section)
    -> fast bounded section/document retrieval
    -> NO Planner
```

Resolver complexity (`DB -> LLM -> vector -> rerank`) is not execution complexity.

---

# Phase 4C — Semantic Discourse Completeness

This is a separate follow-up capability phase. It is **not** part of proving that V1 route intent or document-number resolution was transferred correctly.

## Task 8 — Complete person/coreference/conversation focus

**Priority:** P1 before multi-turn production parity

**Files:**
- Modify: `backend/app/services/agents/v2/adapters/semantic.py`
- Modify: `backend/app/services/agents/v2/adapters/conversation.py`
- Modify context/history ingress owner
- Add multi-turn tests

**Required behavior:**

- [ ] Populate `person_refs` where a stable typed identity exists.
- [ ] Populate typed `active_entities` instead of flattening everything to `concept`.
- [ ] Populate `last_focus` from validated outcomes.
- [ ] Ensure production v2 ingress actually supplies relevant conversation history/context.
- [ ] Resolve unambiguous `văn bản này`, `nghị định trên`, `điều này`, `ông ấy`, `file thứ hai`.
- [ ] True ambiguity becomes clarification.
- [ ] History never reauthorizes out-of-scope resources.
- [ ] Commit: `feat(v2): complete discourse and coreference context`.

---

# Phase 4D — Public Transport / Frontend Synchronization

This is a **production-canary gate**, not a prerequisite for proving Phase 4A route-intent parity and not a prerequisite for writing backend Planner code.

## Task 9 — Define a versioned public chat/SSE contract

**Priority:** P0 before canary

**Files — Backend:**
- Create/modify v2 public transport DTO owner
- Modify v1/v2 SSE normalization adapter
- Modify public chat persistence only for metadata that must survive reload
- Add transport-contract tests

**Files — Frontend:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/hooks/useRAGChatStream.ts`
- Modify chat UI/history hydration components
- Add hook/component/E2E tests

**Boundary:** frontend never consumes raw `SupervisorV2State`, checkpoint state, `TaskPlan`, runtime services, hidden model reasoning, or internal evidence UUIDs.

**Minimum public event union:**

```text
status
clarification_required
clarification_resolved
source/citation
token
complete
error
cancelled
```

**Required behavior:**

- [ ] Inventory every v1/v2 emitted event and every frontend consumer branch.
- [ ] Normalize v1 and v2 into one frontend presentation contract during canary.
- [ ] Add structured clarification options + server-issued option IDs + resume metadata.
- [ ] Frontend may submit only server-issued clarification values/options; never fabricate document UUID/workspace/binding identity.
- [ ] Normalize citations into one stable public type and persist enough metadata to rebuild completed messages after refresh.
- [ ] Add UI phases such as `clarifying`, `planning`, `executing`, `evaluating`, `generating` where useful, without exposing CoT.
- [ ] Unknown forward-compatible event types must not crash the stream.
- [ ] Test fragmented SSE frames and actual `event:` + `data:` framing, not only data-only frames.
- [ ] Browser/component E2E: simple RAG+citation, exact section, People, clarification/resume, not-found, cancel/error, reload.
- [ ] Commit: `feat(v2): synchronize public chat contract with frontend`.

---

# Phase 5 — Adaptive Complex Execution

Planner is a distinct complex-execution phase. It consumes the already-correct semantic contracts from Phase 4A/4B/4C. Frontend Task 9 must pass before canarying Planner, but Planner backend implementation may proceed in parallel after Phase 4A/4B gates.

## Task 10 — Integrate governed Adaptive Planner

**Priority:** P1 after Phase 4A + 4B backend gates

**Files:**
- Create planning adapter/package
- Modify `RuntimeServices` with runtime-only planner service
- Modify `complex_research_graph.py`
- Consume `get_planner_provider()`
- Add complex planner tests

**Required behavior:**

- [ ] Use existing `ResearchPlanningInput` as authoritative runtime planning envelope.
- [ ] Project only minimized/redacted model input.
- [ ] Planner returns a proposal, never dispatches tools.
- [ ] Validate scope/capability/budget/DAG before lease/checkpoint/scheduler.
- [ ] Do not expose raw internal evidence UUIDs unnecessarily to the model.
- [ ] Fallback to safe deterministic policies when available.
- [ ] Planner cannot mint document identity or widen hard scope.
- [ ] Commit: `feat(v2): add governed adaptive planner`.

---

## Task 11 — Evaluator-driven bounded append-only replan

**Priority:** P1

- [ ] Reuse existing `EvidenceEvaluation`, `MissingRequirement`, `Contradiction`, `TaskExecutionSummary`.
- [ ] Project minimized evaluator gaps only.
- [ ] Replan is append-only; completed tasks are immutable.
- [ ] Enforce max tasks/replans/budgets/capability catalog/hard scope.
- [ ] Commit: `feat(v2): add bounded adaptive replanning`.

---

## Task 12 — Expand complex work types

**Priority:** P1/P2

Order:

1. `multi_goal`;
2. generic `cross_domain`;
3. `evaluate` / compliance.

Keep execution as validated DAGs; do not introduce an unconstrained ReAct loop.

---

# Cross-cutting rollout blockers — independent of V1 intent migration

The following findings are valid production-safety blockers, but they must not be confused with route-intent migration acceptance:

- ACL filtering for standalone/fallback and defense-in-depth document reads;
- current-ACL filtering when hydrating chat history/sources/people data;
- tracing redaction/approved sink policy;
- correct `langgraph.checkpoint.postgres` dependency/readiness;
- capability service availability probes/catalog correctness;
- public frontend transport compatibility before canary.

These items block **production rollout/canary**, not the semantic proof that Phase 4A successfully translated V1 intent into v2 contracts.

---

# Dependency graph

```text
Task 0 LLM roles
   |
   +--------------------+
   |                    |
   v                    v
Phase 4A             Phase 4B
Intent migration     Document identity migration
T1 -> T2 -> T3 -> T4 T5 -> T6 -> T7
   |                    |
   +---------+----------+
             |
             v
      backend fast-path parity
             |
       +-----+-------------------+
       |                         |
       v                         v
Phase 4C                    Phase 5 backend
Discourse/coreference       Planner/replan
T8                          T10 -> T11 -> T12
       |
       +-------------+
                     |
                     v
              Phase 4D frontend
              Public transport T9
                     |
                     v
          production canary / rollout
```

Frontend work is a canary dependency, not an artificial blocker on Planner implementation.

---

# Final acceptance criteria

## Phase 4A — Intent migration

- V2 no longer uses regex keywords as the primary classifier when v1 intent intelligence is available.
- `Nguyễn Văn A là ai?` -> People fast path.
- greeting prefix + factual remainder -> factual fast path.
- general legal factual query without named document -> targetless `document.retrieve` fast path.
- V1 `next_agent`/`pending_intent` do not leak into v2 contracts.

## Phase 4B — Document identity migration

- V1 full/short document number and approximate-title behavior is available behind v2 `DocumentReference`.
- `Điều 5 Luật An ninh mạng quy định gì?` is a fast bounded query, not complex.
- low-confidence/ambiguous identity never becomes a fabricated binding.

## Phase 4C — Discourse

- typed focus/coreference works for supported multi-turn flows under current ACL.

## Phase 4D — Frontend

- v1/v2 normalize to one public chat contract.
- citations, clarification/resume, terminal status, and reload are UI-safe and test-covered.

## Phase 5 — Complex

- only genuinely multi-step/dependent work enters Planner.
- Planner proposals remain governed by V2 validation/scheduler/evidence/checkpoint boundaries.

## Production canary

- ACL/history/tracing/dependency/capability-readiness blockers are cleared.
- Backend parity and frontend transport E2E pass.
- V1 remains rollback until the agreed observation window is clean.

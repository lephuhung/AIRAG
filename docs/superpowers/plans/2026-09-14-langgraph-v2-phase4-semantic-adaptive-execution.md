# LangGraph v2 Phase 4 — Semantic Intelligence & Adaptive Execution Plan

> **Execution method:** implement task-by-task with tests first. Do not weaken frozen authorization, binding, scheduler, evidence, or checkpoint boundaries.

**Goal:** make LangGraph v2 reliably understand, route, plan, execute, evaluate, and answer real simple and multi-step queries by completing semantics, adding deterministic-first LLM fallback routing, and integrating a governed adaptive planner using the existing runtime LLM-role configuration system.

**Spec:** `docs/superpowers/specs/2026-09-14-langgraph-v2-semantic-adaptive-execution-design.md`

## Global constraints

- Reuse the existing LLM connection/role runtime config. Do not add separate planner/router provider URL/model/key settings.
- Preserve current frozen v2 business contracts unless a concrete incompatibility is proven.
- `QueryAnalysis` remains unchanged; confidence/model diagnostics are ephemeral telemetry only.
- Model outputs never contain/own authorization, workspace scope, capability grants, immutable revision identity, raw evidence, or executable side effects.
- `RouteDecision` remains deterministic policy.
- Planner proposals must pass existing validation, leases, checkpointing, and shared scheduler before execution.
- One scheduler, one supervisor checkpointer, one capability registry.
- Existing deterministic compare/summarize/retrieve policies remain safe fallbacks.
- v1 remains the rollback arm until Phase-4 rollout gates pass.
- Do not port legacy semantic regex/LLM behavior blindly; legacy code is evidence and reusable implementation only after Phase-4 tests prove the behavior.

---

## Task 0 — Add semantic-router and planner LLM roles + audit control-plane LLM ownership

**Priority:** P0 prerequisite

**Files:**
- Modify: `backend/app/services/runtime_config.py`
- Modify: `backend/app/services/llm/__init__.py`
- Modify: `backend/app/api/llm_config.py` only if inherited-role response metadata requires it
- Modify: `frontend/src/types/llmConfig.ts`
- Modify: `frontend/src/pages/AdminLLMConfigPage.tsx`
- Modify: `frontend/src/lib/translations/vi.json`
- Modify: `frontend/src/lib/translations/en.json`
- Audit: existing internal/control-plane LLM callsites, including semantic preprocessing and abbreviation disambiguation
- Add/modify tests for runtime LLM-role configuration and API/UI typing.

**Required behavior:**

- Add roles `semantic_router` and `planner`.
- If either has no explicit DB role assignment, inherit the **effective** `thinking` connection/model, including DB overrides.
- Explicit role assignment wins over inheritance.
- Implement inheritance in the effective runtime resolution path; do not only alias `_build_from_settings()` because that would miss a DB-configured `thinking` role.
- Add one shared reasoning-role provider factory and wrappers such as:

```python
get_semantic_router_provider()
get_planner_provider()
```

- Keep `get_thinking_provider()` backward compatible.
- Langfuse labels are distinct: `thinking_llm`, `semantic_router_llm`, `planner_llm`.
- Existing `/admin/llm-config/{role}` assignment endpoint is reused unchanged where possible.
- Admin UI exposes the two new task roles using existing connection/model selectors.
- GET config/UI must display truthful inherited connection/model semantics; do not show a misleading `@env` source when the effective fallback comes from a DB-configured `thinking` role. Prefer explicit `inherited_from`/effective assignment metadata over UI inference.
- No schema migration: reuse `system_settings` role keys.
- Audit existing control-plane calls that bypass task-role ownership. In particular, legacy abbreviation disambiguation currently calls the main provider. Assign such calls deliberately (`memory_agent`, `semantic_router`, or retained `main`) based on function ownership; do not let this remain accidental.

**Tests:**

- default semantic-router config equals effective thinking config;
- default planner config equals effective thinking config;
- changing thinking DB assignment changes inherited roles after snapshot refresh;
- explicit semantic-router/planner assignment does not change when thinking changes;
- provider caches are isolated by role/config version;
- API accepts both new roles and rejects unknown roles;
- inherited role response reports truthful effective source/connection;
- no API key is exposed in plaintext;
- frontend `LlmRole` contains both new roles.

---

## Task 1 — Complete the semantic draft instead of dropping refs

**Priority:** P0

**Files:**
- Modify: `backend/app/services/agents/v2/adapters/semantic.py`
- Read/reuse selectively: `backend/app/services/agents/semantic_preprocessor.py`
- Modify/create: `backend/app/services/agents/v2/semantic/` internal helpers if needed
- Modify: `backend/tests/agents/v2/test_adapters_and_registry.py`
- Modify: `backend/tests/agents/v2/test_context_binding_routes.py`
- Add focused semantic extraction tests.

**Required behavior:**

Build validated deterministic extraction for:

- `section_refs` (Chương/Mục/Điều/Khoản/Điểm and equivalent English cues);
- `person_refs` from high-confidence person identifiers/cues;
- coreference candidates against current `ConversationContext`;
- normalized phone/CCCD/BHXH forms with overlap handling;
- document references already supplied by preprocessing.

Remove the current unconditional information loss:

```python
coreferences=()
person_refs=()
section_refs=()
```

Do not guess ambiguous references. Emit `BlockingAmbiguity` when required.

**Legacy preprocessor review constraints:**

- existing section regex is not sufficient as the v2 extractor: it only covers `Chương|Điều|Mục` and currently captures only a single non-space token as the document part;
- reconcile the legacy `RefExtraction.parse_basis` annotation (`regex_section_phrase`) with the implementation/ranker/lookup value (`regex_section`) before reusing that path;
- preserve the useful deterministic document-reference arbitration/metadata lookup only where tests prove compatibility with v2 reference/binding rules.

**Acceptance examples:**

- `Điều 5 Luật An ninh mạng nói gì?` creates a section ref plus a complete document ref, not merely document token `Luật`.
- `Khoản 2 Điều 5 Luật An ninh mạng` preserves the nested locator semantics needed by `SectionLocator`.
- `079012345678 CCCD này của ai?` creates a person ref/domain signal.
- raw ambiguous numeric identifier does not silently become document retrieval.
- known-document API scope remains transport scope and cannot be minted by text/model output.

---

## Task 2 — Populate typed conversation focus for multi-turn coreference

**Priority:** P0

**Files:**
- Modify: `backend/app/services/agents/v2/adapters/conversation.py`
- Modify the conversation snapshot/update owner identified by impact analysis
- Modify: `backend/app/services/agents/v2/nodes/context.py` only if focus propagation belongs at node boundary
- Add tests for conversation snapshots and multi-turn context.

**Required behavior:**

- preserve authoritative document/person/section entities as typed `ActiveEntity` values instead of flattening all summary entities to `concept` when a typed identity is known;
- populate `last_focus` from validated completed turns/semantic outcomes;
- resolve exact `văn bản này`, `nghị định trên`, `điều này`, `ông ấy`, `file thứ hai` when one authorized referent exists;
- ambiguity across multiple plausible active entities becomes typed clarification, not arbitrary selection;
- current runtime ACL is authoritative on resume/follow-up; history never reauthorizes a document.

**Golden sequence:**

```text
Turn 1: Nghị định A quy định gì?
Turn 2: Điều 5 của văn bản này thì sao?
Turn 3: So sánh điều đó với Điều 7 Nghị định B.
```

Each follow-up must preserve the correct stable reference/binding identity without requiring the user to repeat A.

---

## Task 3 — Add one request-scoped ephemeral semantic-inference/cache seam

**Priority:** P0 prerequisite for any semantic LLM use

**Files:**
- Modify: `backend/app/services/agents/v2/contracts/state.py` (`RuntimeServices` only; no checkpointed state field)
- Create: `backend/app/services/agents/v2/semantic/inference.py`
- Modify: `backend/app/services/agent/runtime_selector.py`
- Modify: `backend/app/services/agents/supervisor_v2.py` if construction is owned there
- Add tests proving no inference object enters checkpoints.

**Why this must precede model-backed semantic extraction:**

The current graph calls `build_semantic_draft()` in `binding_node` and rebuilds it again in `semantic_finalizer_node`. If an LLM-backed adapter is introduced without a request-scoped cache, the same turn can issue duplicate semantic LLM calls before routing.

**Interfaces:**

Define an internal strict result/protocol conceptually equivalent to:

```python
class SemanticRoutingInference(...):
    work_type: WorkType
    domains: tuple[Domain, ...]
    dependency_hints: tuple[...]
    confidence: float
    requires_clarification: bool
```

Exact internal schema may also carry validated semantic extraction candidates needed by Tasks 1/2.

**Required behavior:**

- runtime-only; never stored in `SupervisorV2State`;
- cache key is tied to the current request/semantic input and cannot leak between turns or shadow/live runs;
- one semantic model inference per turn can be reused by binding-time draft construction, semantic finalization, and routing;
- deterministic-only turns do not create a model inference entry;
- inference projection contains only safe semantic/contextual text and typed known references;
- no ACL/service objects/raw evidence/secrets;
- malformed/unknown fields fail closed and fall back to deterministic semantics where safe;
- cancellation/deadline bounds model work.

---

## Task 4 — Implement the semantic-router LLM adapter

**Priority:** P0

**Files:**
- Create: `backend/app/services/agents/v2/semantic/router.py`
- Create: `backend/app/prompts/agents/v2_semantic_router.py` or equivalent prompt owner
- Consume: `get_semantic_router_provider()` from Task 0
- Consume: the request-scoped inference/cache seam from Task 3
- Add tests for structured output and injection resistance.

**Required behavior:**

- only invoked when deterministic semantic analysis is uncertain or semantic completion requires it;
- strict structured output; temperature appropriate for classification;
- translates to existing `QueryAnalysis` rather than changing that frozen contract;
- model cannot select route/capability/tool;
- confidence stays ephemeral/telemetry-only;
- pure greeting, obvious people identifier, exact resolved bounded section/doc operations bypass the model.

**Regression matrix:**

- `Nguyễn Văn A là ai?` -> People when person semantics are present, not KG solely because of `là ai`.
- `chào anh, cho tôi hỏi về chế độ thai sản?` -> factual, not direct greeting.
- `sự khác biệt giữa nghỉ phép và nghỉ ốm là gì?` -> ordinary factual retrieval unless true multi-target comparison semantics exist.
- `tổng hợp giúp tôi chính sách nghỉ ốm` does not become multi-document complex merely from `tổng hợp`.

---

## Task 5 — Refactor route_node into deterministic-first hybrid routing

**Priority:** P0

**Files:**
- Modify: `backend/app/services/agents/v2/nodes/routing.py`
- Modify: `backend/tests/agents/v2/test_context_binding_routes.py`
- Modify: `backend/tests/agents/v2/fast_paths/test_domain_paths.py`
- Add hybrid-router focused tests.

**Required flow:**

```text
validated SemanticContext
    -> deterministic analysis + certainty/policy test
    -> if certain: existing QueryAnalysis
    -> if uncertain: reuse cached semantic inference OR semantic-router call
    -> validate QueryAnalysis
    -> deterministic decide_route(...)
```

Do not put LLM access inside `decide_route()`.

Capability availability and hard API scope remain deterministic inputs to route policy.

---

## Task 6 — Build the Phase-4 golden semantic/routing E2E harness

**Priority:** P0 gate before planner

**Files:**
- Create: `backend/tests/agents/v2/golden/` fixtures/tests or an equivalent existing harness location
- Optionally extend: `backend/scripts/replay_v2.py`
- Add a versioned golden query dataset under tests.

**Coverage:**

At minimum:

- direct greeting;
- reference-free factual RAG;
- exact document;
- exact section and nested clause/point locator;
- People phone/CCCD/BHXH/name;
- KG;
- greeting-prefix factual;
- coreference follow-up;
- one/multi-document summary;
- comparison;
- People->Document;
- cross-domain;
- multi-goal;
- evaluate/compliance;
- ambiguity/clarify.

For each case record/assert:

```text
semantic -> QueryAnalysis -> RouteDecision -> expected plan topology
```

Semantic-router call count must be zero for deterministic obvious cases. Binding-time + finalizer-time draft rebuild must not cause duplicate model calls.

---

## Task 7 — Absorb and implement the governed adaptive planner

**Priority:** P1 after semantic/routing gate

**Supersedes/absorbs:** `docs/superpowers/plans/2026-09-13-langgraph-v2-p2-adaptive-planner.md`

**Files:**
- Create: `backend/app/services/agents/v2/planning/__init__.py`
- Create: `backend/app/services/agents/v2/planning/adapter.py`
- Modify: `backend/app/services/agents/v2/contracts/state.py` (`RuntimeServices.planner` runtime-only)
- Modify: `backend/app/services/agents/v2/complex_research_graph.py`
- Modify: `backend/app/services/agent/runtime_selector.py`
- Consume: `get_planner_provider()` from Task 0
- Add: `backend/tests/agents/v2/complex/test_adaptive_planner.py`

**Change from old P2 plan:**

Do **not** create a planner-owned OpenAI client/config stack. `ComplexPlanner` uses the existing `LLMProvider` selected by `planner` role configuration.

**Required behavior:**

- reuse existing `ResearchPlanningInput` as the authoritative ephemeral planning envelope; add a separate safe model projection rather than widening the frozen contract;
- safe minimized planner projection;
- strict proposal schema;
- no raw evidence/ACL/secrets/runtime objects;
- initial proposal and append-only replan proposals;
- validation/lease/checkpoint before scheduler;
- invalid/timeout/malformed proposal falls back to deterministic policy when available;
- explicit hard scope cannot widen;
- proposal never appears in checkpoint before validation.

---

## Task 8 — Project existing evaluator gaps to the planner and enable bounded adaptive replan

**Priority:** P1

**Files:**
- Modify: `backend/app/services/agents/v2/nodes/evaluate.py` only where a safe projection helper belongs
- Modify: `backend/app/services/agents/v2/complex_research_graph.py`
- Modify: `backend/app/services/agents/v2/replanning.py`
- Modify: `backend/tests/agents/v2/complex/test_replan_discovery.py`
- Modify: `backend/tests/agents/v2/test_evaluation_grounding.py`

**Important:** do not invent a parallel persisted “planner gap” contract. `ResearchPlanningInput` already carries `prior_evaluation`, and `EvidenceEvaluation` already owns `MissingRequirement` and `Contradiction`. Build a minimized model-facing projection from those existing contracts plus `TaskExecutionSummary`.

**Required behavior:**

Expose only typed/minimized facts such as:

- missing target coverage;
- insufficient semantic criterion;
- not-found/error dependency outcome from task summary;
- supporting/reference discovery needed under policy;
- contradiction identifiers/use refs requiring a bounded additional read.

Replan is append-only and bounded by existing budget/config. No completed-task replacement.

Change default rollout behavior only after tests/golden gates demonstrate termination and quality; do not simply raise `V2_MAX_REPLANS` globally at implementation start.

---

## Task 9 — Expand complex work types in controlled order

**Priority:** P1/P2

Implement only after Tasks 0-8 pass.

### 9A. `multi_goal`

Compose multiple already-supported objectives into one validated DAG.

### 9B. `cross_domain`

Generalize dependencies beyond current People->Document vertical slice while keeping typed dependency materialization.

### 9C. `evaluate` / compliance

Introduce trusted typed semantic completion criteria and evidence-grounded evaluation. Model-generated criteria are proposals subject to validation and cannot select authorization/scope.

**Tests:** real graph cases with causal dependencies, not mocked route-only assertions.

---

## Task 10 — Observability, shadow, and rollout gates

**Priority:** P0/P1 throughout; final gate

**Files:**
- Modify: `backend/app/services/agent/rollout_metrics.py`
- Modify: `backend/app/services/agent/shadow_runtime.py`
- Modify: `backend/tests/agents/v2/test_rollout_metrics.py`
- Modify: `backend/tests/agents/v2/test_shadow_runtime.py`
- Modify: `backend/scripts/collect_v2_rollout_report.py`
- Modify: `backend/scripts/check_v2_rollout_gate.py`
- Update: `docs/harness.md`, `README.md`, `CLAUDE.md` as appropriate.

**Semantic-router metrics:**

- call/bypass count;
- p50/p95 latency;
- parse/fallback rate;
- work-type/domain confusion against golden labels;
- deterministic/model disagreement in shadow mode;
- semantic inference cache hit/reuse count (to detect duplicate calls).

**Planner metrics:**

- call/latency;
- proposal rejection reason;
- fallback rate;
- tasks/DAG depth;
- replan count;
- budget exhaustion;
- final sufficiency and grounded-citation rate.

No prompts, raw evidence, personal scalar, API keys, or chain-of-thought in rollout metrics.

**Rollout sequence:**

1. unit/contract regression;
2. golden replay;
3. semantic-router shadow;
4. planner shadow;
5. internal workspace enablement;
6. existing canary controller;
7. promotion only after agreed quality/safety thresholds.

---

## Implementation dependency order

```text
Task 0  LLM roles + role-ownership audit
   |
   +--> Task 3 inference/cache seam
   |       |
   |       +--> Task 1 semantic extraction/completion
   |               |
   |               +--> Task 2 conversation/coreference
   |               |
   |               +--> Task 4 semantic-router adapter
   |                        |
   |                        +--> Task 5 hybrid routing
   |                                  |
   |                                  +--> Task 6 golden gate
   |                                             |
   +---------------------------------------------+--> Task 7 adaptive planner
                                                      |
                                                      +--> Task 8 adaptive replan
                                                               |
                                                               +--> Task 9 complex expansion

Task 10 observability/shadow spans the phase and is mandatory for rollout.
```

Tasks 1 and 3 are intentionally coupled during implementation: deterministic extraction can start independently, but **no model-backed semantic completion may be wired into `build_semantic_draft()` before Task 3 prevents duplicate binding/finalizer calls**.

## Definition of done

Phase 4 is complete only when:

- semantic refs/coreferences used by routing are populated rather than silently dropped;
- obvious queries route deterministically without unnecessary LLM calls;
- one turn does not duplicate semantic inference across binding/finalization/routing;
- uncertain routing uses the configured `semantic_router` role and produces validated existing `QueryAnalysis`;
- complex supported queries use the configured `planner` role, with deterministic fallback and all existing governance intact;
- multi-turn references work in the golden harness under current ACL;
- planner/replan loops terminate within budget;
- final answers are grounded and cited;
- semantic-router/planner can be assigned models independently through existing Admin LLM configuration;
- control-plane LLM callsites have explicit role ownership rather than accidental use of the main answer model;
- shadow/canary quality gates pass with v1 still available as rollback.

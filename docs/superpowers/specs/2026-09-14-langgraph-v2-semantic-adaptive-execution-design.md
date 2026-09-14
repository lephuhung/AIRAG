# LangGraph v2 Phase 4 — Semantic Intelligence & Adaptive Execution Design

**Date:** 2026-09-14

## 1. Goal

Move LangGraph v2 from contract-first/runtime-pilot maturity to reliable end-to-end execution for real user queries by adding semantic completeness, deterministic-first LLM fallback routing, governed adaptive planning/replanning, and query-quality gates without weakening the frozen authorization, binding, scheduler, evidence, checkpoint, or grounding boundaries.

This phase does **not** rebuild the v2 runtime. The existing contract/runtime machinery remains authoritative:

```text
proposal -> validate -> lease -> checkpoint -> TaskScheduler
          -> CapabilityRegistry -> Capability.execute
          -> Evidence -> Evaluator -> Grounding/Synthesis
```

The new intelligence is advisory and typed. Models never own ACL, bindings, tool execution, checkpoint writes, or scope widening.

## 2. Existing baseline to preserve

Keep the following architecture stable unless a concrete Phase-4 regression proves a change is required:

- `SemanticContext`, `QueryAnalysis`, `RouteDecision`, `TaskPlan`, `TaskSpec`, `AgentResult` frozen contract shapes.
- request-scoped authorization and capability registry.
- immutable document revision bindings and binding provenance.
- shared `TaskScheduler` as the only capability dispatcher.
- evidence governance, hydration, evaluation, grounding, and retention leases.
- supervisor-owned checkpointer; no subgraph-owned second saver.
- current fast paths for deterministic bounded operations.
- current complex-research subgraph and deterministic compare/summarize/retrieve policies as fallback/reference behavior.
- v1 as rollback arm during rollout.

## 3. Key current gaps

### 3.1 Semantic information is dropped before routing

`draft_from_preprocessing(...)` currently preserves document refs and abbreviations but initializes:

```python
coreferences=()
person_refs=()
section_refs=()
```

This makes exact-section routing, person-aware routing, and multi-turn references such as `văn bản này`, `nghị định trên`, or `ông ấy` unreliable.

### 3.2 Query analysis is deterministic-only

`analyze_query()` currently relies on refs, supervisor-scope classification, and keyword/regex sets. This is correct for high-confidence cases but unsafe as a universal classifier. The frozen design already allows uncertain cases to use a small model.

### 3.3 The complex runtime exists, but planning is still mainly policy-selected

`ComplexResearchGraph` already executes validated plans through the shared scheduler and supports deterministic compare/summarize/retrieve plus bounded dependency/replan machinery. The missing capability is an adaptive planner that can propose typed multi-step plans for supported complex work while remaining subordinate to validation/governance.

### 3.4 Conversation state is structurally capable but under-populated

`ConversationContext` already has `active_entities`, `last_focus`, and recent turns, but the legacy adapter currently maps summary `key_entities` to generic `concept` entities and leaves `last_focus=None`. Phase 4 must populate useful typed discourse references before coreference resolution can be dependable.

## 4. LLM role architecture

Phase 4 MUST reuse the existing runtime LLM configuration system. It MUST NOT create a second model-connection stack for LangGraph v2.

Existing architecture:

```text
llm_conn.<conn_id>  -> provider + base_url + encrypted api key + extras
llm_role.<role>     -> conn_id + model
```

Add two logical task roles:

```text
semantic_router
planner
```

### 4.1 Inheritance

When the new role has no explicit assignment:

```text
semantic_router -> inherit effective `thinking`
planner         -> inherit effective `thinking`
```

`effective thinking` means the currently resolved thinking role, including a DB override when one exists, not merely `.env` defaults.

An explicit assignment to `semantic_router` or `planner` wins over inheritance.

No new `V2_*_BASE_URL`, `V2_*_MODEL`, or `V2_*_API_KEY` settings are introduced. Feature enablement, timeout, token, and budget settings may remain implementation settings, but provider/connection/model selection belongs to runtime role configuration.

### 4.2 Provider factories

Refactor/reuse the current provider factory so callers can obtain independently traced providers:

```text
thinking_llm
semantic_router_llm
planner_llm
```

`get_thinking_provider()` remains backward compatible. New wrappers may be `get_semantic_router_provider()` and `get_planner_provider()`, implemented through one shared reasoning-role factory.

### 4.3 Admin API/UI

The existing `/admin/llm-config/{role}` endpoint and connection catalogue remain authoritative. Add the two roles to backend `ROLES`, frontend `LlmRole`/`LLM_ROLES`, role metadata, and i18n descriptions.

The API/UI must not misleadingly display `@env` when a new role is inheriting a DB-configured `thinking` assignment. Expose inheritance metadata or resolve assignment presentation so the effective connection/model is truthful.

No database migration is required: `system_settings` already stores open-ended role keys.

## 5. Semantic completion layer

### 5.1 Deterministic-first extraction

Before calling a model, extract high-confidence values deterministically:

- normalized phone forms (`0...`, `+84`, spaces/dashes removed);
- CCCD/BHXH candidates with collision-aware classification;
- explicit document references;
- article/section/chapter/clause references;
- obvious person cues/names where current preprocessing can validate them;
- exact discourse references that map unambiguously to one active entity.

Do not add regexes merely to cover every linguistic case. Rules stop when confidence becomes low.

### 5.2 Small-model semantic inference

For unresolved/ambiguous semantic content, a small-model call may produce a strict internal result containing only semantic facts needed to finish `SemanticDraft` and/or classify the query.

This model output is **ephemeral and uncheckpointed**. It may contain confidence/diagnostic values, but those values do not modify frozen `QueryAnalysis`.

The semantic model cannot emit workspace IDs, authorization, capability grants, binding IDs, document revision identities, raw tool instructions, or executable tasks.

### 5.3 Reuse one semantic inference per turn

The frozen design says routing may reuse validated small-model output when Context already called a model. Therefore Phase 4 must avoid the topology:

```text
Context LLM call -> Router LLM call for the same semantics
```

Prefer:

```text
Context deterministic extraction
    -> optional semantic inference
    -> validate ephemeral inference
    -> finalize SemanticContext
    -> router reuses inference facts when needed
```

Use a request-scoped runtime-only service/channel in `RuntimeServices` or equivalent ephemeral ownership seam. Do not add model output to `SupervisorV2State` or checkpoints.

If no context-model call was necessary but deterministic routing remains uncertain, the router may invoke the semantic-router provider once.

## 6. Hybrid semantic router

### 6.1 Deterministic high-confidence cases

Keep model-free routing for cases such as:

- pure greeting/conversation;
- normalized exact People identifier lookup;
- exact bounded section/document operations whose references are already resolved;
- explicit deterministic hard-scope behavior;
- cases with blocking ambiguity that already require clarification.

### 6.2 Uncertain cases

Introduce an internal classifier result, separate from frozen business contracts. Conceptual shape:

```python
class SemanticRoutingInference:
    work_type: WorkType
    domains: tuple[Domain, ...]
    dependency_hints: tuple[...]
    confidence: float
    requires_clarification: bool
```

Exact implementation shape is internal and may evolve. After validation it is translated into the existing `QueryAnalysis`.

`confidence` is telemetry/policy input only; it is not added to `QueryAnalysis`.

### 6.3 Policy remains deterministic

The model does not return `RouteDecision`. `decide_route(...)` or its successor remains deterministic policy over validated semantic facts, current bindings, request hard scope, and runtime capability availability.

The LLM therefore answers **what the query means**, not **what the system is allowed to execute**.

### 6.4 Known routing defects to eliminate

Regression coverage must include:

- People query with `là ai` must not be stolen by KG semantics when a person identifier/name is present.
- greeting prefix plus factual remainder must not become direct conversation.
- conceptual `khác biệt`, `tổng hợp`, `đánh giá` must not automatically imply a multi-document operation without sufficient semantic evidence.
- raw numeric identifiers must not fall silently into document retrieval.
- one query spanning multiple domains must distinguish true dependency from incidental lexical overlap.

## 7. Conversation/coreference resolution

Populate and consume `ConversationContext` as the discourse contract intended:

- retain typed active document/person/section entities when they become authoritative;
- derive `last_focus` from validated turn outcomes rather than leaving it permanently `None`;
- resolve phrases such as `văn bản này`, `nghị định trên`, `file thứ hai`, `điều này`, `ông ấy` to stable known reference IDs when unambiguous;
- preserve blocking ambiguity when multiple prior entities are plausible;
- never revive unauthorized document identity from history; current runtime scope remains authoritative.

A follow-up turn must not need the user to repeat a previously resolved document/person merely because the surface name is absent in the new query.

## 8. Governed adaptive planner

Absorb the existing `2026-09-13-langgraph-v2-p2-adaptive-planner.md` work into this phase rather than implementing a parallel planner stack.

### 8.1 Planner boundary

Planner input is request-scoped, typed, minimized, and redacted. It may see:

- finalized semantic goal/query analysis;
- current validated bindings/target catalogue in safe projected form;
- currently available capability catalogue;
- budget remaining;
- typed task outcomes;
- evaluator gaps;
- redacted model-facing observations.

It must not see ACL internals, service clients, API keys, namespaces, raw governed evidence, unrestricted personal scalar data, or mutable runtime objects.

### 8.2 Planner output

Planner returns proposals only. It does not dispatch.

```text
PlannerProposal
    -> construct proposed TaskPlan/append tasks
    -> validate_task_plan
    -> scope/capability/budget checks
    -> retention lease
    -> checkpoint
    -> shared TaskScheduler
```

Invalid, cyclic, unauthorized, over-budget, unknown-capability, malformed, or scope-widening proposals are rejected.

### 8.3 Deterministic fallback

Timeout, disabled planner, malformed structured output, or rejected proposal falls back to existing deterministic supported policies where a safe policy exists. Unsupported work remains typed-unavailable or clarification; never fabricate a plan.

### 8.4 Replanning

Replanning is append-only, bounded, and evaluator-driven. Typical typed gaps include:

- missing target coverage;
- insufficient evidence;
- dependency unavailable/not found;
- unresolved supporting/reference discovery;
- conflicting evidence requiring a bounded additional read.

The planner cannot replace completed tasks or widen explicit hard scope.

## 9. Expand complex work types after planner integration

Only after semantic/router/planner gates pass, add governed policy support for currently unsupported complex work types in dependency order:

1. `multi_goal` composed from already-supported atomic capabilities;
2. `cross_domain` dependencies beyond the existing People->Document vertical slice;
3. `evaluate` / compliance using typed semantic completion criteria and grounded evidence.

Do not implement one giant unconstrained ReAct loop. Complex work remains a validated DAG executed by the existing scheduler.

## 10. Observability

Measure the new control-plane LLMs separately:

Semantic router:
- call count / bypass count;
- latency and token usage;
- deterministic-vs-model disagreement on golden/shadow data;
- fallback/parse failure rate;
- confusion matrix by work type/domain.

Planner:
- call count and latency;
- proposal validation rejection rate and reason;
- deterministic fallback rate;
- task count / DAG depth / replan count;
- budget exhaustion;
- final sufficiency and grounded-answer rate.

Do not log raw sensitive evidence or secrets in these metrics.

## 11. Golden E2E acceptance set

Phase completion is based on user-query quality, not only contract/unit tests. Maintain a golden matrix covering at least:

- greeting/direct;
- general factual reference-free RAG;
- exact document and exact section;
- People phone/CCCD/BHXH/name lookup;
- KG lookup;
- greeting-prefix factual query;
- follow-up/coreference across turns;
- one- and multi-document summary;
- section/document comparison;
- People->Document dependency;
- cross-domain dependency;
- multi-goal request;
- compliance/evaluate;
- ambiguous query requiring clarification.

For each case assert as applicable:

```text
semantic -> route -> plan -> task results -> evidence -> evaluation -> grounded answer
```

Also record latency, number of semantic-router calls, planner calls, replans, retrieved evidence count, and citations.

## 12. Rollout

1. Unit/contract regression.
2. Golden replay with deterministic baseline.
3. Semantic-router shadow telemetry; no user-visible behavior change.
4. Planner shadow telemetry; no tool side effects beyond existing isolated shadow guarantees.
5. Internal workspace feature enablement.
6. Canary using existing v1/v2 rollout control.
7. Promote only after quality and safety gates pass; preserve v1 rollback until the agreed observation window is clean.

## 13. Non-goals

- Replacing the shared scheduler.
- Adding a second checkpointer.
- Letting a model decide authorization/scope.
- Letting a model execute arbitrary tools.
- Replacing all deterministic routing with LLM classification.
- Persisting raw model reasoning/chain-of-thought.
- Redesigning frozen v2 business contracts merely to carry confidence/telemetry.
- Creating separate planner/router endpoint configuration outside the existing LLM role system.

# LangGraph v2 Phase 4 — V1 Intelligence, V2 Contracts & Adaptive Execution Design

**Date:** 2026-09-14

## 1. Goal

Phase 4 must preserve the parts of `LLM-Optimize` that already understand real user queries well, while moving their outputs behind the stricter v2 contracts and execution boundaries.

The guiding rule is:

> **V1 answers “what does the user mean, and what are they referring to?”**  
> **V2 answers “what may the system execute, how is it validated, and what evidence is sufficient?”**

Phase 4 therefore does **not** rewrite v1 semantic intelligence with a new regex-first classifier. It reuses/refactors proven v1 behavior for:

- intent understanding;
- named/partial/approximate document references;
- document-number handling;
- document identity resolution;
- abbreviation/contextualization behavior;
- People/simple RAG/section/summarize/KG fast-path semantics;

and translates that intelligence into the existing v2 contract/runtime model.

The existing v2 execution chain remains authoritative:

```text
semantic meaning / resolved identities
        -> V2 contracts
        -> deterministic RouteDecision
        -> validated plan
        -> lease/checkpoint
        -> TaskScheduler
        -> CapabilityRegistry
        -> Evidence
        -> Evaluator
        -> Grounding/Synthesis
```

Models never own ACL, trusted workspace scope, immutable revision binding, capability authorization, checkpoint writes, or arbitrary tool execution.

## 2. Architecture boundary

```text
┌──────────────────────────────────────────────────────────┐
│ INTELLIGENCE LAYER — preserve/refactor proven V1 logic  │
│                                                          │
│ - follow-up contextualization                            │
│ - abbreviation understanding                             │
│ - People/simple RAG/section/summarize/KG intent          │
│ - document mention understanding                         │
│ - document number / agency / year hints                  │
│ - DB -> LLM -> vector -> fuzzy/rerank document resolver │
└───────────────────────────┬──────────────────────────────┘
                            │ typed adapter boundary
                            ▼
┌──────────────────────────────────────────────────────────┐
│ CONTRACT LAYER — V2                                      │
│                                                          │
│ SemanticDraft / SemanticContext                          │
│ DocumentReference / SectionReference                     │
│ QueryAnalysis / RouteDecision                            │
│ DocumentBindingSet                                       │
│ TaskPlan / TaskSpec / AgentResult                        │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│ EXECUTION/GOVERNANCE LAYER — V2                          │
│                                                          │
│ Fast Path / ComplexResearchGraph                         │
│ Planner / TaskScheduler / CapabilityRegistry             │
│ ACL / immutable revision pinning                         │
│ Checkpoint/resume / Evidence / Evaluator / Grounding     │
└───────────────────────────┬──────────────────────────────┘
                            │ public transport adapter
                            ▼
┌──────────────────────────────────────────────────────────┐
│ PRESENTATION/TRANSPORT LAYER                              │
│                                                          │
│ versioned chat DTO + SSE events                          │
│ frontend state / clarification / sources / progress      │
└──────────────────────────────────────────────────────────┘
```

V1 legacy state/control fields such as `next_agent`, `pending_intent`, mutable `document_ids`, and agent-to-agent routing are not imported into v2 contracts.

## 3. Existing v1 intelligence to preserve

### 3.1 Document resolution

`app.services.agent.doc_resolver.resolve_candidates()` is the proven identity-resolution core and already exists on `feat/langgraph-v2`. Phase 4 reuses it through a v2 adapter rather than copying/reimplementing it.

Its behavior to preserve includes:

```text
reference + full question topic
    -> regex/metadata hints
    -> DB exact/partial candidate search
    -> LLM metadata extraction when DB is dry
    -> DB search again
    -> vector fallback using the full question
    -> number-identity filter
    -> topic rerank
    -> fuzzy similar-title fallback
```

This supports users who remember only:

- a full document number;
- a short number such as `15`;
- a number plus agency such as `Thông tư 15 của Bộ Công an`;
- a partial/approximate title;
- a year plus topic;
- only the subject matter of the document.

A difficult identity resolution does **not** make the user request complex. For example, resolving `Điều 5 luật an ninh mạng` may internally require DB/LLM/vector work, but once it resolves to one document + one section, the execution topology is still a bounded fast-path read.

### 3.2 Intent classification

The v1 behavior in `supervisor_scope.py` and the v1 supervisor prompt/taxonomy is retained as the semantic basis for simple intents:

- greeting/direct;
- personal/contextual;
- People lookup;
- general RAG search;
- named-document resolution;
- search by document number;
- section read/search;
- one-document summarize;
- KG lookup;
- list/search-abbreviation where supported.

Deterministic narrow scopes continue to bypass an LLM where v1 already treats them as unambiguous. Uncertain/full cases may use the configured semantic-routing model, but its result is translated into v2 semantic facts rather than accepted as a v2 route.

### 3.3 Follow-up and abbreviation intelligence

Existing v1 follow-up contextualization and abbreviation behavior are implementation evidence to reuse selectively. Phase 4 must preserve the user-visible behavior while moving ownership of model selection to explicit LLM task roles and keeping outputs within v2 semantic contracts.

## 4. V2 boundaries that remain authoritative

The following remain v2-owned and must not be delegated back to v1:

- trusted `RequestContext` scope;
- authorization/workspace ACL;
- `DocumentReference.resolved_document_id` acceptance rules;
- immutable document revision pinning;
- retention leases;
- deterministic `RouteDecision`;
- capability availability;
- plan validation;
- scheduler dispatch;
- evidence governance/evaluation;
- checkpoint/resume;
- final grounding and citation policy.

The document resolver may propose/return server-side candidate document IDs, but the existing v2 binding resolver remains the only authority that pins an authorized current/historical revision.

## 5. Runtime-only intelligence adapters

Phase 4 introduces request-scoped runtime adapters, not new checkpoint contracts.

Conceptually:

```python
class IntentDecision:
    intent: str
    task_hints: tuple[str, ...]
    needs_memory: bool
    is_legal_query: bool
    source: Literal["deterministic", "model"]
    confidence: float | None

class DocumentIdentityResolution:
    reference: str
    resolved_document_id: UUID | None
    candidate_document_ids: tuple[UUID, ...]
    section_reference: str | None
    status: Literal["resolved", "ambiguous", "not_found"]
    score: float | None
```

These shapes are internal/runtime-only. They are translated into frozen v2 contracts and never checkpointed as raw model/resolver output.

## 6. Request-scoped inference/cache seam

The current v2 graph rebuilds `SemanticDraft` in both binding and semantic-finalization paths. Therefore any model-backed intent/document inference must be cached request-locally so a single turn does not run the same LLM/vector resolution multiple times.

Use `RuntimeServices` or an equivalent request-scoped runtime owner for:

- intent classification result;
- document identity resolution result;
- semantic completion result.

Cache keys must include the current request/query and relevant conversation semantic context. Cache data cannot cross turns, users, workspaces, or live/shadow runs.

No inference object is serialized into `SupervisorV2State`.

## 7. Translation into V2 semantic contracts

The intelligence adapter enriches `SemanticDraft` with:

- resolved/ambiguous/not-found `DocumentReference` values;
- `SectionReference` values;
- typed People/person references where available;
- abbreviation resolutions;
- coreference resolutions;
- blocking ambiguities only when user choice is genuinely required.

Example:

```text
User: "Điều 5 Luật An ninh mạng quy định gì?"

V1 intelligence:
  intent = search_section
  document mention = "Luật An ninh mạng"
  document_id = X
  section = "Điều 5"

Adapter -> V2:
  DocumentReference(resolved_document_id=X, status="resolved")
  SectionReference(label="Điều 5")

Binding -> immutable revision pin

QueryAnalysis:
  work_type="retrieve"
  domains=("document", "section")

RouteDecision:
  route="fast_domain"
```

No Planner call is required.

## 8. Simple eligibility and routing policy

Whether an LLM was needed to understand the query is **not** a complexity signal.

Complexity is determined from the logical evidence/execution topology after semantic resolution.

A query is eligible for a fast/bounded workflow when the system can select a known atomic capability or fixed bounded workflow without synthesizing a query-specific DAG.

Examples that should stay fast where capabilities are available:

- People lookup;
- general factual RAG retrieval;
- one resolved document retrieval;
- one resolved section read;
- search by document number after identity resolution;
- one-document bounded summarize;
- simple KG lookup.

Planner is required only for true query-specific multi-step topology such as:

- multi-target comparison;
- evaluate/compliance across evidence sets;
- People -> Document dependency;
- multi-goal requests;
- generic cross-domain dependency;
- iterative evidence completion where a fixed bounded workflow is insufficient.

The semantic/intelligence model never directly returns the final `RouteDecision`.

## 9. LLM role architecture

Reuse the existing `llm_conn.*` + `llm_role.*` runtime configuration system.

Add:

```text
semantic_router
planner
```

Defaults:

```text
semantic_router -> inherit effective thinking
planner         -> inherit effective thinking
```

Explicit admin assignment overrides inheritance.

Do not add separate planner/router base-URL/model/API-key stacks.

Existing internal LLM calls used by semantic/document/abbreviation/follow-up logic must receive deliberate role ownership; accidental use of the main answer model must be removed where appropriate.

## 10. Fast-path parity gate

Before adaptive planner work begins, v2 must meet or exceed `LLM-Optimize` behavior on simple-query semantics and document resolution.

Golden parity coverage must include:

- pure greeting;
- People by phone/CCCD/BHXH/name;
- general factual RAG;
- exact legal document title;
- approximate/partial document title;
- full official number;
- short/bare number;
- number + agency;
- year + topic;
- topic-only document recollection;
- exact section;
- nested Khoản/Điều/Điểm references;
- one-document summarize;
- KG lookup;
- greeting prefix + factual query;
- ambiguous document candidate requiring clarification;
- low-confidence target that must not be force-bound;
- multi-turn follow-up/coreference.

For every golden case compare at least:

```text
semantic interpretation
resolved document identity/candidates
intent
V2 QueryAnalysis
V2 RouteDecision
expected capability/workflow
```

Planner implementation does not begin until this gate passes.

## 11. Frontend/public transport contract synchronization

The frontend must not depend directly on `SupervisorV2State`, checkpoint schemas, runtime-only semantic inference, or raw internal `TaskPlan` objects. Internal v2 contracts and the public chat transport contract have different stability/security responsibilities.

Introduce a versioned public DTO/SSE boundary that maps internal v2 outcomes into UI-safe events and persisted chat metadata.

Conceptually:

```text
Internal v2 contracts
    SemanticContext
    ClarificationRequest
    RouteDecision
    TaskPlan / AgentResult
    Evidence / FinalResponse
          │
          ▼
Public Chat Transport Adapter
          │
          ├─ status/progress events
          ├─ clarification_required
          ├─ clarification_resolved
          ├─ sources/citations
          ├─ final_response
          └─ typed error/cancel terminal
          │
          ▼
TypeScript transport contracts
          │
          ▼
useRAGChatStream / ChatPanel / history hydration
```

### 11.1 Public transport requirements

- Version the public event/envelope contract independently from checkpoint schema where practical.
- Preserve currently supported v1 events during canary/dual-runtime operation.
- Do not expose raw ACL state, hidden model reasoning, internal evidence-store identifiers, service objects, planner prompts, or unrestricted checkpoint contents.
- Clarification must be first-class structured data, not only a tokenized prose message.
- Candidate document choices must use safe public option IDs/titles/metadata needed by the UI; frontend must not fabricate document identities.
- Final response/citations must have one stable frontend representation regardless of whether v1 or v2 served the turn.
- Unknown event types must fail safely/ignore compatibly rather than crash the active stream.

### 11.2 Frontend state requirements

The frontend TypeScript model must support at least:

- `clarifying` status and structured clarification choices;
- semantic/document-resolution progress without exposing internal reasoning;
- planning/executing/evaluating phases for complex turns;
- stable source/citation payloads;
- terminal `success | clarify | error | cancelled` states;
- persisted/reloaded messages carrying the same public metadata needed to reconstruct the UI after refresh.

Existing `ChatStreamStatus`/`AgentStepType` may be extended or replaced by a discriminated public event union; avoid growing an untyped `Record<string, unknown>` event switch.

### 11.3 Compatibility gate

Fast-path v2 canary is not considered frontend-ready until the same golden flows work through the actual web stream and history reload:

- simple RAG answer with citations;
- exact section answer;
- People lookup;
- ambiguous document -> clarification choices -> resume -> answer;
- not-found/low-confidence document behavior;
- cancel/error terminal;
- refresh/reload after a completed or clarified turn.

Planner rollout additionally requires frontend handling for planning/task progress and resumed/replanned runs, but the UI consumes only public progress summaries, not the internal DAG contract.

## 12. Governed adaptive planner

After fast-path parity and core frontend transport compatibility pass, complex work uses the Phase-4 planner role.

Planner input is typed, minimized, request-scoped, and redacted. Planner output is proposal-only:

```text
PlannerProposal
    -> validate
    -> scope/capability/budget checks
    -> retention lease where needed
    -> checkpoint
    -> TaskScheduler
```

The planner cannot widen explicit hard scope, mint trusted document IDs, replace completed tasks, bypass capability authorization, or dispatch tools directly.

Existing deterministic compare/summarize/retrieve policies remain safe fallback/reference implementations.

## 13. Evaluator-driven bounded replan

Reuse existing `EvidenceEvaluation`, `MissingRequirement`, `Contradiction`, and task summaries. Do not create a parallel persisted gap contract merely for the planner.

Project only minimized facts into model-facing replan input. Replans are append-only, bounded by configured budgets, and validated before scheduling.

## 14. Complex-work expansion order

Only after fast-path parity and planner integration:

1. `multi_goal` composed from supported atomic capabilities;
2. generic `cross_domain` beyond existing People -> Document vertical slices;
3. `evaluate` / compliance with typed criteria and grounded evidence.

Do not replace this with an unconstrained ReAct loop.

## 15. Shadow and rollout

Rollout stages:

1. unit/contract tests;
2. v1-v2 fast-path golden parity;
3. backend -> frontend public transport contract tests;
4. web E2E fast-path + clarification/history compatibility;
5. semantic/document-intelligence shadow comparison;
6. planner-proposal shadow;
7. full read-only document execution shadow where available;
8. internal workspace enablement;
9. canary through existing v1/v2 rollout controls;
10. promotion only after quality/safety/UI compatibility gates pass.

V1 remains the rollback arm until the observation window is clean.

## 16. Non-goals

- Rewriting proven v1 document-resolution behavior as a new regex-only implementation.
- Importing v1 mutable SupervisorState/agent routing into v2.
- Letting v1/model output decide ACL or trusted workspace scope.
- Letting a semantic model directly choose v2 capabilities/routes.
- Exposing raw v2 checkpoint/state contracts directly to the frontend.
- Exposing hidden model reasoning, raw planner prompts, or internal evidence IDs to the UI.
- Replacing `TaskScheduler`, capability registry, evidence, or checkpointer.
- Adding a second planner/router provider configuration stack.
- Persisting model reasoning/chain-of-thought.
- Changing frozen v2 contracts merely to carry confidence/diagnostics.

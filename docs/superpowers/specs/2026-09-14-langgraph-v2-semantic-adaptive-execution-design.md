# LangGraph v2 — V1 Intelligence to V2 Contracts Design

**Date:** 2026-09-14

## 1. Goal

The migration must preserve the parts of `LLM-Optimize` that already work well without importing the legacy supervisor control model into v2.

The phrase **“reuse the v1 brain”** has two immediate meanings only:

1. reuse **v1 route/intent understanding**;
2. reuse **v1 document identity/number resolution**.

Everything else is a separate capability phase.

The ownership rule is:

> **V1-derived intelligence says what the query means and which document the user probably refers to.**  
> **V2 decides what is authorized, how it is bound, which topology executes, and whether evidence is sufficient.**

V1 legacy fields such as `next_agent`, `pending_intent`, mutable `document_ids`, `SupervisorState`, and v1 task plans are not migrated.

---

## 2. Canonical architecture

```text
User query
   |
   +-----------------------------+
   |                             |
   v                             v
V1 intent intelligence      V1 document resolver
(intent/work type)          (identity candidates)
   |                             |
   +-------------+---------------+
                 |
                 v
         V1 -> V2 adapters
                 |
                 v
          V2 typed contracts
     DocumentReference / SemanticContext
              QueryAnalysis
                 |
                 v
       deterministic RouteDecision
                 |
        +--------+--------+
        |                 |
        v                 v
     Fast Path       Complex boundary
        |                 |
        +--------+--------+
                 v
     V2 scheduler/evidence/grounding
                 |
                 v
       Public transport adapter
                 |
                 v
              Frontend
```

The frontend never receives raw checkpoint or runtime state.

---

# Phase 4A — Route/Intent Intelligence Migration

## 3. Scope

Phase 4A migrates only the semantic route/intent intelligence from v1.

The relevant v1 semantics include:

- greeting/direct;
- People lookup;
- general factual RAG search;
- search by document number;
- section search/read intent;
- one-document summarize;
- KG lookup;
- `resolve_doc` as a semantic prerequisite.

V1 intent does **not** become a v2 `RouteDecision`.

Conceptually:

```python
class IntentDecision:
    intent: str
    source: Literal["deterministic", "model"]
    confidence: float | None
```

Then:

```text
IntentDecision
    -> translate
    -> QueryAnalysis
    -> deterministic V2 route policy
```

## 4. Intent mapping

The minimum mapping is:

```text
mongo_search_* -> lookup / people
search         -> retrieve / document
search_doc_num -> retrieve / document
search_section -> retrieve / document + section
summarize      -> summarize / document
kg_query       -> lookup / knowledge_graph
greeting       -> direct
resolve_doc    -> semantic prerequisite, not complexity
```

Generic lexical patterns such as `là ai`, `khác biệt`, `tổng hợp`, or `đánh giá` must not remain the primary classifier once typed v1 intent is available.

### 4.1 Erratum (final re-review M8): v2-only narrow `evaluate` deterministic scope

The Task-3 ruling above has one explicit, v2-only exception. A
**compliance/legal assessment request** is recognized deterministically
before any model call because the v1 taxonomy has no `evaluate` intent —
without it, a compliance question would be model-classified to `search` and
silently degrade to the targetless document-retrieval fast path (final
review I1).

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

## 5. General factual fast path

A reference-free factual query such as:

```text
"chế độ thai sản được quy định thế nào?"
```

must be able to produce:

```text
QueryAnalysis(retrieve, document)
    -> targetless workspace document retrieval
    -> fast bounded execution
```

An explicit document binding is not required for every factual RAG query.

Route policy must consider the **actual available capability catalog**, not only requested/allowed capability names.

## 6. Phase 4A acceptance

At minimum:

```text
Nguyễn Văn A là ai?                       -> People fast path
chào anh, hỏi về chế độ thai sản?         -> factual retrieve fast path
chế độ thai sản được quy định thế nào?    -> targetless document retrieval fast path
```

Phase 4A does not wait for fuzzy document identity, conversation coreference, frontend clarification UI, or Planner.

---

# Phase 4B — Document Identity / Number Migration

## 7. Scope

This phase migrates the second proven v1 capability explicitly requested: robust document resolution.

Reuse `app.services.agent.doc_resolver.resolve_candidates()` through an adapter; do not create a competing implementation.

Preserve its multi-stage behavior:

```text
reference + full question topic
    -> regex/metadata hints
    -> DB exact/partial search
    -> LLM metadata extraction if DB misses
    -> DB retry
    -> vector fallback using full question
    -> number identity filter
    -> topic rerank
    -> fuzzy similar-title fallback
```

Supported real-user recollection includes:

- full official number;
- short/bare number;
- number + agency;
- approximate title;
- partial title;
- year + topic;
- topic-only recollection.

## 8. Adapter boundary

Resolver output becomes v2 document identity facts:

```text
resolved   -> DocumentReference(resolved_document_id=...)
ambiguous  -> DocumentReference(candidate_document_ids=...)
not_found  -> unresolved/not-found reference
```

The resolver does not pin revisions. Existing v2 binding remains the only immutable revision authority.

Low-confidence identity must not silently become a binding.

## 9. Canonical section query

The required behavior is:

```text
"Điều 5 Luật An ninh mạng quy định gì?"
    -> resolve document identity
    -> preserve simple section locator
    -> v2 immutable binding
    -> QueryAnalysis(retrieve, document+section)
    -> bounded fast retrieval
    -> NO Planner
```

The internal cost of document resolution is not a complexity signal.

Phase 4B only requires the minimum authoritative section locator needed for this bounded case. Full multi-turn section/coreference semantics belong to Phase 4C.

---

# Phase 4C — Semantic Discourse Completeness

## 10. Scope

This is a separate capability phase, not part of proving Phase 4A or 4B migration success.

It includes:

- typed `person_refs`;
- typed conversation `active_entities`;
- `last_focus`;
- production history/context ingress;
- unambiguous coreference such as `văn bản này`, `điều này`, `ông ấy`, `file thứ hai`;
- clarification for true ambiguity.

History never reauthorizes a resource outside current runtime ACL.

---

# Phase 4D — Public Transport / Frontend Synchronization

## 11. Transport boundary

The frontend must not consume:

- `SupervisorV2State`;
- checkpoint payloads;
- runtime services;
- raw `TaskPlan`;
- hidden model reasoning;
- internal evidence-store identifiers.

Introduce a versioned public chat/SSE boundary that normalizes v1 and v2 into one UI-safe contract.

Minimum public events:

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

Clarification is structured, not only prose. The frontend submits only server-issued option IDs/allowed values; it cannot mint document/workspace/binding identity.

## 12. Frontend requirements

The web client must correctly handle:

- simple RAG + citations;
- exact section;
- People lookup;
- clarification -> select -> resume -> answer;
- not-found/low-confidence;
- cancel/error terminal;
- reload after completed/clarified turns.

Unknown future event types must not crash the active stream.

Frontend compatibility blocks production canary, but it does **not** block proving Phase 4A intent migration or writing backend Planner code.

---

# Phase 5 — Adaptive Complex Execution

## 13. Planner boundary

Planner starts only after the semantic contracts used by complex work are reliable. It is not part of V1 intent migration.

Planner input is minimized/redacted and based on existing v2 planning contracts. Planner output is proposal-only:

```text
Planner proposal
    -> validate
    -> scope/capability/budget checks
    -> lease/checkpoint
    -> TaskScheduler
```

Planner cannot:

- widen hard scope;
- mint trusted document identity;
- execute tools directly;
- replace completed tasks;
- bypass capability authorization.

## 14. Replanning

Replanning reuses existing evaluator contracts (`EvidenceEvaluation`, `MissingRequirement`, `Contradiction`, task summaries). It is append-only and bounded.

## 15. Complex expansion order

1. `multi_goal`;
2. generic `cross_domain`;
3. `evaluate` / compliance.

No unconstrained ReAct loop.

---

# 16. Cross-cutting production rollout blockers

These are independent of whether V1 intent migration is semantically correct:

- shared ACL filtering for standalone/fallback document IDs;
- defense-in-depth workspace filtering in document tools;
- current-ACL filtering of history/sources/people data;
- tracing redaction/approved sink policy;
- correct `langgraph.checkpoint.postgres` dependency/readiness;
- actual capability service catalog/readiness;
- frontend public transport compatibility.

They block production canary, not Phase 4A semantic acceptance.

---

# 17. LLM role architecture

Reuse existing runtime LLM configuration.

Add:

```text
semantic_router
planner
```

Unassigned roles inherit the **effective** `thinking` role including DB overrides. Explicit role assignment wins.

No second provider/model/API-key configuration stack is introduced.

Control-plane LLM callsites must have deliberate ownership; do not accidentally use the main answer model.

---

# 18. Dependency and rollout model

```text
Phase 4A Intent
      +
Phase 4B Document Identity
      |
      v
Backend fast-path parity
      |
      +--------------------+
      |                    |
      v                    v
Phase 4C Discourse     Phase 5 Planner backend
      |                    |
      +---------+----------+
                |
                v
       Phase 4D Frontend/Public DTO
                |
                v
        Production canary/rollout
```

V1 remains the default/rollback arm until production gates pass.

---

# 19. Non-goals

- Porting the entire v1 supervisor into v2.
- Treating V1 `next_agent` as V2 route authority.
- Replacing the proven document resolver with regex-only logic.
- Exposing internal v2 state directly to frontend.
- Letting semantic/planner models decide ACL or trusted scope.
- Replacing shared scheduler/evidence/checkpointer.
- Persisting hidden model reasoning/chain-of-thought.

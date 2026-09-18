# Multi-Intent Routing — Implementation Plan

**Date:** 2026-09-18
**Status:** Initial plan — implementation pending
**Goal:** Land the multi-intent semantic contract, classifier, router gate, and planner input required by the multi-intent routing spec, behind a flag that is off by default, without changing flag-off routing behavior.

**Spec:** `docs/multi-intent-routing-spec.md` (including §33 supersession amendment, commit `1147849`).

## 1. Phase boundary

This plan covers the full routing change because the pieces are not independently shippable: a dormant `IntentAnalysis` contract delivers nothing, and a classifier without the router gate reproduces the bug. The safe boundary is the **feature flag**, not a contract-only phase.

Required flag-off behavior (`V2_MULTI_INTENT_ROUTING_ENABLED=false`, the default):

- `route_node` keeps the current `IntentDecision` advisory path byte-for-byte (`classify_deterministic` → `classify_evaluate` → single-intent model);
- `decide_route` sees no `intent_analysis` and behaves exactly as today;
- `QueryAnalysis`/`RouteDecision` wire shape is unchanged — the new `WorkType`/`RouteReason` literal members are never emitted;
- `intent_analysis` state slot exists but stays `None` on every flag-off turn;
- v1 graph, rollout selection, shadow runs, fast paths, and discovery-dormant behavior are untouched.

Flag-on behavior: LLM semantic classification first, deterministic identifier extraction never feeds intent, `intent_count > 1` forces `complex_research`, classifier failure forces `complex_research` (never legacy fast path).

## 2. Execution preconditions

Use a clean dedicated worktree. Before Task 1:

```bash
git status --short
git diff --check
```

Do not start if any file named by this plan already has unrelated modifications. Never use whole-file `git add` on a pre-dirty file. This plan does not require task-by-task commits; commit only after the complete diff has been reviewed and only when explicitly requested.

Run all Python tests from `backend/`:

```bash
cd backend
python -m pytest ...
```

Before editing any symbol, run the repository-required GitNexus upstream impact analysis for that symbol and stop for user confirmation on HIGH or CRITICAL risk. Before any eventual commit, run `detect_changes(scope="compare", base_ref="main")`.

Known HIGH/CRITICAL-impact symbols this plan touches (warn before editing):

- `decide_route`, `analyze_query`, `route_node` — every supervisor turn flows through them;
- `validate_supervisor_state`, `migrate_checkpoint_payload` — checkpoint authority;
- `SupervisorV2State` — every checkpoint payload;
- `build_v2_ingress` — runtime wiring.

`classify_supervisor_scope`, `people_intent_from_query`, `deterministic_decision_for_scope` are **shared with v1** — never edited by this plan; v2 only stops consuming them in the flag-on intent path.

## 3. Frozen decisions

### 3.1 Intent contract

New data-only module `contracts/intent.py` (imports `ContractModel` only):

```python
class DetectedIntent(ContractModel):
    intent_id: str                      # server-assigned "i1","i2",... by position
    name: str                           # semantic intent name; registry key
    confidence: float | None = None     # metadata only, never routing authority
    depends_on: tuple[int, ...] = ()    # indexes into intents, strictly earlier
    description: str | None = None


class IntentAnalysis(ContractModel):
    primary_intent: str | None
    intents: tuple[DetectedIntent, ...]
    is_multi_intent: bool
    requires_complex_execution: bool     # model advisory; router recomputes
    semantic_summary: str | None = None
    source: Literal["model", "deterministic"]
```

`IntentAnalysis` is a **checkpointed** `ContractModel` — route authority depends on `intent_count`, so resume must replay the identical analysis. This replaces `IntentDecision` (runtime-only, single-intent) on the flag-on path; `IntentDecision` stays for flag-off.

Spec deviation recorded here (spec update required): `DetectedIntent.depends_on` does not exist in spec §5.1 — it is needed to distinguish the parallel case (Test 8) from the sequential dependency case (Test 7) deterministically. Index-based (not name-based) to stay unambiguous when two intents share a name.

Contract validation (`validate_intent_analysis` in `contracts/validation.py`): non-blank unique `intent_id`s (position-derived `i{n}`), `name` non-blank, `confidence` in `[0,1]` when present, `depends_on` indexes in range and strictly earlier (topological), `is_multi_intent == (len(intents) > 1)` exact consistency, `primary_intent` blank-or-member-or-None. Registry membership is deliberately NOT a contract rule — an unknown name is a routing outcome (`semantic_uncertainty`), not a malformed payload.

### 3.2 Intent registry

New pure-data module `semantic/intent_registry.py` (no imports from contracts/routing — consumed by both):

```python
INTENT_REGISTRY: dict[str, IntentSpec] = {
    "people_lookup":       {"execution": "atomic",  "fast_path": "simple_people_lookup",
                            "work_type": "lookup",    "domains": ("people",),
                            "capability": "people.lookup"},
    "people_search":       {"execution": "atomic",  "fast_path": "simple_people_lookup",
                            "work_type": "lookup",    "domains": ("people",),
                            "capability": "people.lookup"},
    "document_lookup":     {"execution": "atomic",  "fast_path": "targetless_document_retrieval",
                            "work_type": "retrieve",  "domains": ("document",),
                            "capability": "document.retrieve"},
    "document_search":     {"execution": "atomic",  "fast_path": "targetless_document_retrieval",
                            "work_type": "retrieve",  "domains": ("document",),
                            "capability": "document.retrieve"},
    "section_lookup":      {"execution": "atomic",  "fast_path": "exact_section_retrieval",
                            "work_type": "retrieve",  "domains": ("document", "section"),
                            "capability": "section.read"},
    "kg_lookup":           {"execution": "atomic",  "fast_path": "simple_kg_lookup",
                            "work_type": "lookup",    "domains": ("knowledge_graph",),
                            "capability": "knowledge_graph.query"},
    "memory_lookup":       {"execution": "atomic",  "fast_path": None,
                            "work_type": "explain",   "domains": ("memory",),
                            "capability": "memory.lookup"},
    "direct_answer":       {"execution": "atomic",  "fast_path": "direct_conversation",
                            "work_type": "direct",    "domains": ("memory",),
                            "capability": None},
    "greeting":            {"execution": "atomic",  "fast_path": "direct_greeting",
                            "work_type": "direct",    "domains": ("memory",),
                            "capability": None},
    "personal":            {"execution": "atomic",  "fast_path": "direct_conversation",
                            "work_type": "explain",   "domains": ("memory",),
                            "capability": "memory.lookup"},
    "list_documents":      {"execution": "atomic",  "fast_path": "targetless_document_retrieval",
                            "work_type": "retrieve",  "domains": ("document",),
                            "capability": "document.retrieve"},
    "summarize":           {"execution": "complex", "work_type": "summarize",
                            "domains": ("document",), "capability": None},
    "compare_documents":   {"execution": "complex", "work_type": "compare",
                            "domains": ("document",), "capability": None},
    "evaluate_compliance": {"execution": "complex", "work_type": "evaluate",
                            "domains": ("document",), "capability": None},
    "cross_domain_research": {"execution": "complex", "work_type": "cross_domain",
                            "domains": (),            "capability": None},
    "write":               {"execution": "complex", "work_type": "retrieve",
                            "domains": ("write",),    "capability": None},
}


def needs_complex_execution(analysis: IntentAnalysis) -> bool: ...
```

`needs_complex_execution` implements spec §8.1 exactly: empty intents → True; `len > 1` → True; registry miss → True; `execution != "atomic"` → True.

`fast_path` is a **candidate** reason only. The existing downstream gates (bound count, section coordinate, conversational guard) still apply after the intent gate — `document_lookup` with no bound target must not claim `document.read`. Single-atomic-intent flows into the same `analyze_query` → `decide_route` machinery via the registry's `work_type`/`domains` mapping, replacing `_INTENT_ANALYSIS` on the flag-on path.

`document_search` maps to `document.retrieve`, not `document.search` (spec §33.4): `document.search` emits discovery candidates requiring `settle` + `V2_ALLOW_*_DISCOVERY` (default-off); `document.retrieve` emits evidence directly.

### 3.3 Contract literal additions

`contracts/routing.py`:

- `WorkType` gains `"multi_intent"`;
- `RouteReason` gains `"multi_intent"` and `"semantic_uncertainty"`.

Additive literal members never appear in old checkpoints — backward compatible. `rollout_control._V2_KNOWN_*` sets derive from `get_args` on the frozen literals, so they pick the new members automatically; the defensive fallback frozensets in the `except` branch are updated to match.

### 3.4 Checkpoint schema revision 3

`contracts/state.py`:

```python
CHECKPOINT_SCHEMA_REVISION: Final[Literal[3]] = 3
CheckpointSchemaRevision = Literal[1, 2, 3]

_REVISION_THREE_ONLY_KEYS = frozenset({"intent_analysis"})
```

`migrate_checkpoint_payload` chains: revision-1 payloads take the existing revision-2 migration, then `intent_analysis=None` is added; revision-2 payloads get `intent_analysis=None`. A migrated `None` means "legacy path" — consistent because `route_decision`/`query_analysis` were already checkpointed when the old run routed; the slot is only produced and consumed inside `route_node`, which never re-runs on resume past the route edge. `SupervisorV2State.intent_analysis: IntentAnalysis | None` joins `_SLOT_MODELS`, `_NULLABLE_SLOTS`, `build_initial_v2_state`, and fresh-turn clearing. `validate_supervisor_state` validates the slot contract only (no cross-check against route decision — a `semantic_uncertainty` route legitimately pairs with `intent_analysis=None`).

### 3.5 Classifier service

New module `semantic/multi_intent.py`:

```python
class MultiIntentClassifier:
    """Request-scoped multi-intent classifier (flag-on path)."""

    def __init__(self, *, provider_factory: Callable[[], Any] | None = None) -> None: ...
    async def classify(
        self, query: str, *, has_doc_ids: bool = False
    ) -> IntentAnalysis: ...
```

- Per-turn cache + single-flight, same pattern as `IntentClassifier`.
- Input is the **finalized** `semantic.contextualized_query` (post coreference resolution) — satisfies spec §21 without reordering graph nodes.
- Greeting/personal short-circuit is kept (spec §33.1): the deterministic scope check runs first but ONLY `greeting`/`personal` scopes produce a short-circuit `IntentAnalysis(source="deterministic")`; `people`/`rag_named_doc`/`full` scopes all fall through to the model.
- Identifier scopes (`phone`/`cccd`/`bhxh`/name) and `classify_evaluate` are never consulted on the flag-on path.
- Structured output (spec §17): provider call via `acomplete` with the `IntentAnalysis` JSON schema embedded in the prompt and strict `IntentAnalysis.model_validate` of the complete response — **no** `_extract_json_object` prose salvage, no substring extraction. Deviation recorded: provider-native `with_structured_output` does not exist in `LLMProvider`; strict schema validation satisfies "no free-form regex parsing" and native support is deferred to later plans.
- `temperature=0.0`, `think=False`, small `max_tokens`.
- Every failure mode returns `None` to the caller (never a coerced `search`): provider exception, timeout, invalid JSON, schema mismatch, empty intents — the router owns the `semantic_uncertainty` outcome. Caller logs one warning line with no query content beyond the existing classifier logging policy.

`RuntimeServices` gains `multi_intent_classifier: Any = None` (typed `Any`, same framework-free reason as `intent_classifier`). Ingress wires one per turn only when the flag is enabled; flag-off turns leave it `None`.

### 3.6 Router gate

`route_node` flag-on flow:

```python
intent_analysis = await _intent_analysis_for_route(
    context.services.multi_intent_classifier, state
)
if intent_analysis is None:
    analysis = QueryAnalysis(work_type="multi_intent", domains=..., dependency_hints=())
    decision = RouteDecision(route="complex_research", reason_code="semantic_uncertainty")
else:
    analysis = analyze_query(state["semantic"], intent_analysis=intent_analysis)
    decision = decide_route(analysis, ..., intent_analysis=intent_analysis)
return {"query_analysis": analysis, "route_decision": decision,
        "intent_analysis": intent_analysis}
```

`analyze_query` gains an `intent_analysis` parameter (mutually exclusive with `intent`): multi-intent → `work_type="multi_intent"`, domains = union of registry domains ∪ `_ref_domains(semantic)`; single intent → registry `work_type`/`domains` ∪ ref domains, same ref-arity promotion as today (`_typed_work_type`).

`decide_route` gains an `intent_analysis` parameter; the intent gate sits **after** write/ambiguity/unresolved-reference checks (clarify still wins) and **before** every fast branch:

```python
if intent_analysis is not None and needs_complex_execution(intent_analysis):
    if len(intent_analysis.intents) > 1:
        return RouteDecision(route="complex_research", reason_code="multi_intent")
    return RouteDecision(route="complex_research", reason_code="semantic_uncertainty")
```

Wait — single complex intents must keep their specific reasons (`compare` → `comparison`, `evaluate` → `compliance_evaluation`, write → `simple_write_operation`). So the gate is: `len > 1` → `multi_intent`; `len == 0` or registry miss → `semantic_uncertainty`; single known complex intent → fall through to the existing work-type branches (which produce the specific reasons); single atomic → falls through to existing fast branches. This preserves every existing reason code.

Flag-off: `intent_analysis` parameter is `None` everywhere; `decide_route`/`analyze_query` are byte-identical in behavior.

`requires_v1_fallback` needs no code change beyond the fallback frozensets (§3.3) — `multi_intent`/`semantic_uncertainty` are valid v2 outcomes served by the governed complex DAG.

### 3.7 Planner input and multi-intent skill

`ResearchPlanningInput` gains `intent_analysis: IntentAnalysis | None = None`. `ComplexResearchState`, `build_complex_research_state`, `normalize_complex_state` thread it (parent → child only; the child never mutates it, so no merge-back).

`PlannerModelInput` gains `primary_intent: str | None` and `intents: tuple[PlannerIntentRef, ...]` where `PlannerIntentRef = {name, depends_on}` — confidence stays out of the model projection (metadata only, minimization rule). `build_planner_model_input` populates from `intent_analysis`; `None` → empty tuple.

New deterministic skill `skills/multi_intent/policy.py` covering `work_type == "multi_intent"`:

- **Eligible:** every intent has `depends_on == ()` AND at least one intent maps to a governed `capability` in the registry AND task count ≤ `budget.max_tasks_remaining`.
- **Build:** one targetless task per evidence-bearing intent — `people_lookup`/`people_search` → `people.lookup`, `document_search`/`document_lookup`/`list_documents` → `document.retrieve` (empty `target_ids`), `kg_lookup` → `knowledge_graph.query`, `memory_lookup`/`personal` → `memory.lookup`. Complex intents (`evaluate_compliance`, `compare_documents`, `summarize`, `cross_domain_research`) emit **no task** — the pipeline `evaluate`/`synthesize` nodes own them. All emitted tasks have `depends_on=()` (parallel topology), `origin=InitialTaskOrigin`.
- **Refuse:** any `depends_on` non-empty (sequential semantics belong to the model planner / people materializer), zero evidence-bearing intents, missing catalog capability, over budget. Covering-skill refusal is final per existing planner ordering.

Sequential dependent case (spec Test 7): the classifier marks `document_search.depends_on=[people_lookup]` → skill refuses → governed model planner proposes `T1 people.lookup` alone; the existing `people_document_materialize_node` appends the governed `document.search` on success. The planner prompt gains one line: intents with `depends_on` must produce only the upstream task — person-dependent document search is materializer-owned and must not be proposed.

`build_initial_proposal` adds the `multi_intent` branch before the generic fallthrough; its `ContractValidationError` does NOT fall back to `build_unscoped_retrieve_plan` (a multi-intent request silently degraded to one retrieval would re-create the original bug — fail closed to the model path or unavailable boundary).

### 3.8 Configuration

```python
V2_MULTI_INTENT_ROUTING_ENABLED: bool = Field(default=False)
```

`.env.example` entry next to the other v2 flags, documented as dormant-until-rollout. No other new settings: classifier provider reuse the existing `semantic_router` role.

## 4. File map

| File | Responsibility |
|---|---|
| `contracts/intent.py` (new) | `DetectedIntent`, `IntentAnalysis` checkpointed models. |
| `contracts/routing.py` | `WorkType`/`RouteReason` literal additions only. |
| `contracts/state.py` | Revision 3, `intent_analysis` slot, `multi_intent_classifier` runtime slot. |
| `contracts/validation.py` | `validate_intent_analysis`, migration chain, root-slot wiring. |
| `semantic/intent_registry.py` (new) | `INTENT_REGISTRY`, `IntentSpec`, `needs_complex_execution`. |
| `semantic/multi_intent.py` (new) | `MultiIntentClassifier` service. |
| `nodes/routing.py` | `intent_analysis` parameters, intent gate, `_intent_analysis_for_route`, flag branch. |
| `contracts/planning.py` | `ResearchPlanningInput.intent_analysis`. |
| `planning/projection.py` | `PlannerIntentRef`, intents in model input. |
| `planning/planner.py` | Prompt line for dependent-intent rule. |
| `skills/multi_intent/policy.py` (new) | Deterministic parallel evidence plan. |
| `complex_research_graph.py` | `multi_intent` dispatch branch, child-state threading. |
| `agent/rollout_control.py` | Defensive fallback frozensets only. |
| `agent/runtime_selector.py` | Flag-gated classifier wiring in `build_v2_ingress`. |
| `core/config.py`, `.env.example` | `V2_MULTI_INTENT_ROUTING_ENABLED`. |
| `CLAUDE.md` | Flag, revision 3, multi-intent seam documentation. |

## 5. Task order

### Task 1 — Intent contracts, registry, and literal additions

**Files**

- Create `backend/app/services/agents/v2/contracts/intent.py`.
- Create `backend/app/services/agents/v2/semantic/intent_registry.py`.
- Modify `backend/app/services/agents/v2/contracts/routing.py`.
- Modify `backend/app/services/agents/v2/contracts/validation.py`.
- Add `backend/tests/agents/v2/contracts/test_intent_analysis.py`.

**Changes**

1. Implement `DetectedIntent`/`IntentAnalysis` per §3.1 and `validate_intent_analysis` with every listed rule, including `is_multi_intent` consistency and topological `depends_on`.
2. Implement `INTENT_REGISTRY` + `needs_complex_execution` per §3.2 — spec §8.1 semantics exactly.
3. Add `"multi_intent"` to `WorkType`; `"multi_intent"`/`"semantic_uncertainty"` to `RouteReason`. Update the `rollout_control` fallback frozensets to match.
4. No consumer wiring yet — everything is importable and validatable only.

**Required tests**

- valid single/multi/dependent analyses pass; `is_multi_intent` mismatch rejected; duplicate/blank `intent_id` rejected; `depends_on` self/later/out-of-range rejected; confidence out of `[0,1]` rejected; `primary_intent` not among intents rejected;
- `needs_complex_execution` truth table: empty → True, two → True, unknown name → True, complex execution → True, single atomic → False;
- unknown intent name passes contract validation (registry membership is a routing concern);
- strict JSON round-trip of `IntentAnalysis` through a checkpoint-shaped dict.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts/test_intent_analysis.py -q
```

### Task 2 — Checkpoint revision 3 and `intent_analysis` root slot

**Files**

- Modify `backend/app/services/agents/v2/contracts/state.py`.
- Modify `backend/app/services/agents/v2/contracts/validation.py`.
- Modify `backend/app/services/agents/supervisor_v2.py` (slot registration, initial state, fresh-turn hygiene).
- Add `backend/tests/agents/v2/contracts/test_checkpoint_revision.py` cases (extend the existing file).

**Changes**

1. `CHECKPOINT_SCHEMA_REVISION = 3`; `CheckpointSchemaRevision = Literal[1,2,3]`; `_REVISION_THREE_ONLY_KEYS`.
2. Migration: rev 1 → existing rev-2 migration → add `intent_analysis=None`; rev 2 → add `intent_analysis=None`; rev 3 missing the key → reject; partial rev-3 shape on older discriminator → reject; unknown revision → reject.
3. `SupervisorV2State.intent_analysis: IntentAnalysis | None`; register in `_SLOT_MODELS`/`_NULLABLE_SLOTS`; `build_initial_v2_state` emits `None`; fresh-turn clearing includes it; `validate_supervisor_state` runs `validate_intent_analysis` when non-`None`.

**Required tests**

- rev-1 and rev-2 exact shapes migrate to 3 with all slots `None`;
- rev-2 payload already carrying `intent_analysis` → partial-shape reject;
- rev-3 payload missing `intent_analysis` key → reject;
- slot coercion rebuilds `IntentAnalysis` after JSON round-trip;
- fresh-turn hygiene clears a populated slot.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts/test_checkpoint_revision.py \
  tests/agents/v2/test_supervisor_v2.py -q
```

### Task 3 — Multi-intent classifier service

**Files**

- Create `backend/app/services/agents/v2/semantic/multi_intent.py`.
- Modify `backend/app/services/agents/v2/contracts/state.py` (`RuntimeServices.multi_intent_classifier`).
- Modify `backend/app/services/agent/runtime_selector.py` (flag-gated wiring in `build_v2_ingress`).
- Modify `backend/app/core/config.py`, `.env.example`.
- Add `backend/tests/agents/v2/test_multi_intent_classifier.py`.

**Changes**

1. `MultiIntentClassifier` per §3.5: greeting/personal short-circuit (`source="deterministic"`), model path with strict `IntentAnalysis.model_validate`, `None` on every failure, per-turn cache + single-flight.
2. Classifier prompt per spec §16: full-request analysis, multiple intents allowed, no first-intent stop, full dependent-request expansion, identifier presence must not decide the whole intent. Output schema = `IntentAnalysis` minus `source`/`intent_id` — the server assigns `intent_id`s by position and `source` by path, so the model cannot mint either.
3. `V2_MULTI_INTENT_ROUTING_ENABLED` setting + `.env.example`.
4. Ingress wires `multi_intent_classifier` only when the flag is on; flag-off leaves `None` and never constructs the service.

**Required tests**

- compound phone+legal query returns ≥2 intents including a document intent (stubbed provider);
- pure phone query returns single `people_lookup`;
- invalid JSON / schema mismatch / provider exception → `None`, no exception escapes;
- greeting short-circuits without a provider call; a phone query DOES reach the provider (regression: the exact bug);
- single-flight and per-turn cache identical to `IntentClassifier` semantics;
- flag-off ingress leaves the slot `None`.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/test_multi_intent_classifier.py \
  tests/agents/v2/test_intent_adapter.py -q
```

### Task 4 — Router intent gate and fallback inversion

**Files**

- Modify `backend/app/services/agents/v2/nodes/routing.py`.
- Modify `backend/app/services/agent/rollout_control.py` (fallback frozensets, §3.3).
- Add `backend/tests/agents/v2/test_multi_intent_routing.py`.
- Modify `backend/tests/agents/v2/test_context_binding_routes.py` (flag-on parity cases only — existing flag-off cases unchanged).

**Changes**

1. `_intent_analysis_for_route` (flag-on only): checkpoint slot first (resume determinism — a resumed turn never re-classifies), then `services.multi_intent_classifier`; `None` classifier or `None` result → forced `semantic_uncertainty` complex outcome, **never** the legacy `_intent_for_route` path.
2. `analyze_query(..., intent_analysis=...)` per §3.6.
3. `decide_route(..., intent_analysis=...)`: gate after write/ambiguity/unresolved, before fast branches; `len>1` → `multi_intent`; `len==0`/registry-miss → `semantic_uncertainty`; otherwise fall through.
4. `route_node` returns `intent_analysis` in its update (checkpoints the analysis beside the decision).

**Required tests**

- spec §28 Tests 1–6 at route level: pure phone → `fast_domain/simple_people_lookup`; people-only compound ("số điện thoại X và địa chỉ") stays single-intent fast when the classifier says one intent; phone+doc → `complex_research/multi_intent`; phone+compliance → `multi_intent`; single compare → `complex_research/comparison`; document lookup → fast;
- classifier `None` → `complex_research/semantic_uncertainty` — and specifically a phone-bearing query with a failed classifier must NOT reach `simple_people_lookup`;
- unknown single intent name → `semantic_uncertainty`;
- `intents` containing an unknown name among valid ones → `multi_intent` (not silent drop);
- resumed state with checkpointed `intent_analysis` never calls the classifier;
- flag-off: identical inputs produce byte-identical decisions to current code (regression corpus unchanged).

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/test_multi_intent_routing.py \
  tests/agents/v2/test_context_binding_routes.py \
  tests/agents/v2/fast_paths -q
```

### Task 5 — Planner input, projection, and `multi_intent` skill

**Files**

- Modify `backend/app/services/agents/v2/contracts/planning.py`.
- Modify `backend/app/services/agents/v2/complex_research_graph.py`.
- Modify `backend/app/services/agents/v2/planning/projection.py`.
- Modify `backend/app/services/agents/v2/planning/planner.py`.
- Create `backend/app/services/agents/v2/skills/multi_intent/__init__.py` and `policy.py`.
- Add `backend/tests/agents/v2/complex/test_multi_intent_plan.py`.

**Changes**

1. `ResearchPlanningInput.intent_analysis`; thread through `ComplexResearchState`/`build_complex_research_state`/`normalize_complex_state`/`build_planning_input`.
2. `PlannerModelInput.primary_intent` + `intents` (name + `depends_on` only).
3. `skills/multi_intent/policy.py` per §3.7; register the `multi_intent` branch in `build_initial_proposal` and `_skill_covers_work_type`; refusal does NOT use the unscoped-retrieve fallback.
4. Planner prompt: dependent intents produce only the upstream task; person-dependent document search is materializer-owned.

**Required tests**

- spec Test 8 shape: `[people_lookup, document_search, evaluate_compliance]` → plan = `T1 people.lookup` + `T2 document.retrieve`, both `depends_on=()`, `InitialTaskOrigin`; evaluation then depends structurally on both (no T3 task exists);
- spec Test 7 shape: `document_search.depends_on=[0]` → deterministic skill refuses → model planner path emits `T1 people.lookup` only, and `people_document_materialize_node` appends `T2 document.search` on the real subgraph (reuse the `test_people_document.py` harness);
- plan with a forged `document.search` carrying `person_identifier` still rejected (materializer ownership unchanged);
- over-budget and missing-catalog refusals are final (no unscoped-retrieve fallback for `multi_intent`);
- `intent_analysis=None` planning input validates and plans exactly as before (flag-off parity);
- planner model projection contains no confidence values and no raw identifiers beyond existing `person_refs` labels.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/complex/test_multi_intent_plan.py \
  tests/agents/v2/complex/test_people_document.py \
  tests/agents/v2/complex/test_adaptive_planner.py -q
```

### Task 6 — End-to-end supervisor integration and canonical docs

**Files**

- Modify `backend/tests/agents/v2/test_supervisor_v2.py` or add `test_multi_intent_e2e.py`.
- Modify `CLAUDE.md`.

**Changes**

1. End-to-end flag-on supervisor runs (in-memory saver, stubbed classifier + capabilities): the §28 Test 4 query flows `context → binding → semantic_finalizer → route → complex_boundary` with a checkpointed `IntentAnalysis`, dispatches both evidence tasks through the shared `TaskScheduler`, and reaches synthesis — proving checkpoint order (plan validation precedes dispatch).
2. Resume test: checkpoint after `route`, resume, assert classifier is not re-invoked and the checkpointed analysis drives the identical decision.
3. `CLAUDE.md`: revision 3 + `intent_analysis` slot, the flag and its default, the LLM-first semantic rule with the deterministic-router authority split, pointer to the spec §33 supersession. Do not claim parallel dispatch, provider-native structured output, or discovery-candidate multi-intent — those are later plans.

**Final verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts \
  tests/agents/v2/complex \
  tests/agents/v2/fast_paths \
  tests/agents/v2/test_multi_intent_classifier.py \
  tests/agents/v2/test_multi_intent_routing.py \
  tests/agents/v2/test_intent_adapter.py \
  tests/agents/v2/test_supervisor_v2.py -q
cd .. && git diff --check
git status --short
```

Review the complete diff and confirm:

- flag-off routing produces byte-identical `QueryAnalysis`/`RouteDecision` on the existing test corpus;
- no regex/identifier check feeds any flag-on intent decision;
- every flag-on failure mode lands on `complex_research`, none on a fast path;
- `classify_supervisor_scope`/`deterministic_decision_for_scope`/`people_intent_from_query` unchanged (v1-shared);
- no capability dispatch added outside the shared `TaskScheduler`;
- no query text, phone numbers, or personal identifiers added to logs/traces;
- only intended files changed; `detect_changes(scope="compare", base_ref="main")` reviewed.

## 6. Exit criteria

- [ ] Spec §29 acceptance criteria all pass under the flag (see Task 4 test list for the route-level subset; Tasks 5–6 cover plan/topology criteria).
- [ ] The original production defect query routes `complex_research` and produces people + document evidence tasks.
- [ ] Pure phone lookup still routes `simple_people_lookup` → `people.lookup` with no behavior change.
- [ ] Every classifier failure mode produces `semantic_uncertainty`/complex — zero fast-path fallbacks.
- [ ] Revision-1 and revision-2 checkpoints migrate cleanly; resumed runs never re-classify.
- [ ] `V2_MULTI_INTENT_ROUTING_ENABLED` defaults `false`; flag-off corpus is byte-identical.
- [ ] `CLAUDE.md`/`.env.example` describe only landed behavior.

## 7. Later plans

| Plan | Produces |
|---|---|
| Concurrent dispatch | Scheduler batch-executes independent ready tasks honoring `supports_parallel` + `max_parallel_branches`; spec "parallel" is currently topology-only per §33.4. |
| Provider-native structured output | `LLMProvider.acomplete_structured` / per-provider JSON-schema enforcement replacing prompt+strict-validate. |
| Discovery-candidate multi-intent | `document.search` as an evidence intent once `V2_ALLOW_*_DISCOVERY` opens; settle→read integration with `IntentAnalysis` slots. |
| Per-intent replan gaps | Adaptive replanner consumes per-intent outcomes to target missing evidence instead of whole-query gaps. |
| Classifier telemetry | Intent distribution/confidence dashboards feeding model tuning and the flag rollout decision. |

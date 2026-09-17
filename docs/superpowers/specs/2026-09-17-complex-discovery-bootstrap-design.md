# Complex Research Discovery Bootstrap Design

**Date:** 2026-09-17  
**Status:** Approved design, revision 4 — three independent review rounds, no blocker remaining
**Scope:** LangGraph v2 complex-research path (`summarize` and `compare` first)

## 1. Problem

The v2 complex-research planner currently requires document bindings before it can produce useful plans:

- `summarize` requires exactly one bound `target` document;
- `compare` requires exactly two bound documents, one `target` and one `reference`;
- `document.search` runs only after a plan exists;
- generic discovery can add only `supporting` or `discovered` bindings;
- explicit or quoted documents may already be immutable `ScopedDocument` bindings before the graph knows whether they are research targets or merely conversational context.

Consequently, a complex request without usable `document_ids` or resolved references can fail before the system searches for the documents needed to plan it. A quoted document can also be incorrectly treated as the answer target when the user is asking about another document.

The worktree also contains a targetless `retrieve-unscoped` fallback for a covering skill that refuses its input. That fallback provides grounded workspace evidence, but it does not select authoritative document roles for summarize/compare. Bootstrap discovery must compose with, not erase, that fallback.

## 2. Goals

1. Discover missing summarize/compare documents inside the authenticated workspace before final research planning.
2. Separate immutable binding provenance from turn-specific research target selection.
3. Auto-select only high-precision candidates; otherwise suspend and ask the user.
4. Preserve the execution invariant:

   ```text
   validated + checkpointed TaskPlan → shared TaskScheduler → capability
   ```

5. Preserve ACL revalidation, exact-revision pinning, retention leases, restart safety, cancellation, and model-facing privacy.
6. Allow at most five probes, two rounds, top-five results per probe, and a 15-second discovery sub-deadline.
7. Keep v1 and unrelated v2 paths unchanged.
8. Avoid making search a mandatory dependency when explicit bindings already satisfy every required research slot.
9. Reuse existing candidate, capability, scheduler, binding-resolver, and clarification machinery rather than introducing parallel systems.

## 3. Non-goals

- The planner will not call a database, vector index, capability, or legacy agent directly.
- Discovery will not be a separate autonomous agent or scheduler.
- The feature will not replace v1 or globally enable v2.
- The model will not mint document IDs, revisions, binding IDs, candidate IDs, target roles, or authorization.
- Low-confidence semantic matches will not be silently promoted.
- Workspace scope will never be widened.
- Write and unsupported routes remain on existing fallback paths.
- Optional verification search for already-satisfied slots is excluded from the initial implementation.
- Thematic multi-document summary is a later rollout phase after named summary and compare pass their gates.

## 4. Routing and Topology

### 4.1 Discovery admission and routing precedence

The route node receives the server-owned feature snapshot as a pure policy input. `decide_route` gains a keyword-only `discovery_bootstrap_enabled: bool = False` parameter (the existing callers stay byte-compatible), and `route_node` passes it from the settings snapshot.

The only route-time admission that is both decidable and necessary is the **unresolved or ambiguous document-identity reference** for `summarize`/`compare`. Today `decide_route` sends every open required document reference to semantic clarification (`unresolved_required_binding`). The flag-gated branch must sit immediately before that check, after the `write` and `blocking_ambiguities` branches:

```text
decide_route order (unchanged unless noted):
  1. "write" in domains                  → complex_research (simple_write_operation)
  2. semantic.blocking_ambiguities       → clarify (essential_ambiguity)
  3. NEW: flag on AND work_type in {summarize, compare}
          AND _has_open_required_reference(semantic)
                                        → complex_research (discovery_bootstrap)
  4. _has_open_required_reference        → clarify (unresolved_required_binding)   [flag off only]
  5. ...existing branches unchanged...
```

The predicate is `_has_open_required_reference`, which is true for every non-`resolved` status including `not_found` and `error` — not only for genuinely ambiguous references. That broader set is intentional here: a reference that could not be resolved locally is exactly the case bootstrap search exists to satisfy.

Everything after that point is untouched, and the `write` branch keeps its precedence.

Semantic-binding-conflict, thematic-expansion, and ambiguous-compare-side admission are **not** decidable at route time and are **not** route-time reasons:

The route node does not have the conflict signal; the `bound_count == 1` fast-summary gate fires before any conflict check; and `analyze_query` can already label “quoted A + named B” as `compare` from reference arity alone. Route-time admission therefore stays restricted to the unresolved/ambiguous document-identity case, and the complex boundary is authoritative for everything else through `derive_target_slots`, `reconcile_explicit_bindings`, and `determine_missing_slots`.

The route node may still checkpoint an observability-only summary:

```python
class DiscoveryNeed(ContractModel):
    required: bool
    reason: Literal["unresolved_document_slot"]
```

`discovery_need` is never a routing input after the route node; on disagreement the child's `determine_missing_slots` wins. Validation requires `required == (reason is not None)`, and the whole slot is `None` with the flag off.

Inside the complex boundary, discovery is required when any of these conditions holds:

1. a required target slot is unresolved;
2. an explicit binding conflicts with query semantics, such as quoted A while the query names B;
3. a thematic summary requires multiple document targets;
4. a compare side is missing or ambiguous.

Discovery is skipped when all required slots are confidently satisfied by existing bindings. Therefore:

- “tóm tắt tài liệu này” with one clearly referenced attachment can retain the existing fast path;
- “tóm tắt Nghị định 13” with quoted A reaches complex discovery even though B is not yet bound;
- compare remains a complex route, but skips its search phase when both sides are already resolved;
- a search outage does not break a request whose required slots are already satisfied.

With the feature flag off, the `decide_route` ordering, every reason code, and every route result are byte-compatible.

### 4.2 Complex topology

```text
context → binding → semantic_finalizer → route
                                      ↓
                              derive_target_slots
                                      ↓
                         reconcile_explicit_bindings
                                      ↓
                            determine_missing_slots
                            ┌─────────┴──────────┐
                            │                    │
                     no discovery          discovery required
                            │                    │
                  finalize_target_selection     ↓
                            │          discovery_propose
                            │                    ↓
                            │      discovery_validate_checkpoint
                            │                    ↓
                            │       shared TaskScheduler search
                            │                    ↓
                            │       aggregate_candidates_by_slot
                            │                    ↓
                            │          deterministic_select
                            │          ┌─────────┼──────────┐
                            │          │         │          │
                            │       round 2   clarify    confident
                            │          │         │          │
                            │          │    child returns    │
                            │          │    clarification    │
                            │          │         ↓           │
                            │          │  parent checkpoints │
                            │          │  request + manifest │
                            │          │         ↓           │
                            │          │  clarify_wait       │
                            │          │  interrupt/resume   │
                            │          └─────────┴──────→ selection_settle
                            │                               ↓
                            └────────────────────→ finalize_target_selection
                                                            ↓
                                    fresh research plan OR governed expansion
                                                            ↓
                                      validate / lease / checkpoint
                                                            ↓
                                         shared TaskScheduler research
                                                            ↓
                                    generic settle for non-bootstrap searches
                                                            ↓
                                      evaluate → reduce → decide
                                                            ↓
                                           synthesize / finalize
```

If discovery was skipped, the skill builds a fresh research plan from `ResearchTargetSelection`. If discovery ran, the first checkpointed plan contains governed search tasks and `expand_discovery_plan` appends target units and factual tasks into the same plan lineage.

## 5. Composition with Existing Targetless Fallback

The targetless retrieval fallback and bootstrap have distinct ownership:

- **Flag off:** `retrieve-unscoped` remains unchanged for a covering skill refusal, including summarize, compare, evaluate, multi-goal, and cross-domain.
- **Flag on, discovery required:** bootstrap replaces the fallback only for summarize/compare. Bootstrap failure clarifies or returns typed unavailable; it does not fall through to targetless retrieval and bypass document-role selection.
- **Flag on, slots already satisfied:** existing direct research planning remains. A subsequent skill refusal follows the existing fallback contract.
- **Other work types:** evaluate, multi-goal, and cross-domain retain the current targetless fallback.
- **Retrieve:** scoped and unscoped retrieve policy remains unchanged.

Targetless retrieval is valid for grounded workspace evidence. It is not a substitute for authoritative role selection when summarize/compare has missing or conflicting slots.

The seam is `build_initial_proposal` in `v2/complex_research_graph.py`, which today catches `ContractValidationError` from the summarize/compare skill and returns `build_unscoped_retrieve_plan(...)` — exactly the bypass this section forbids. With the flag on and a standing bootstrap admission or `ResearchTargetSelection`, that unscoped fallback must be suppressed for `summarize`/`compare` and converted into a typed unavailable/clarification outcome instead. That requires threading the flag and the admission/selection into `ResearchPlanningInput` (the same plumbing described in §7), since `build_initial_proposal(planning_input)` has neither today. With the flag off, or when no selection exists, the current fallback is unchanged.

## 6. Checkpoint Schema Compatibility

`ContractVersion` remains `"2.0"`. It is shared by evidence, snapshots, binding audit, transport, and other envelopes, so bumping it solely for one new checkpoint slot would require an unnecessarily broad data migration.

Add a root checkpoint discriminator instead:

```python
checkpoint_schema_revision: Literal[1, 2]
```

Revision 2 adds four required nullable root slots:

```python
discovery_need: DiscoveryNeed | None
discovery: DiscoveryCheckpoint | None
document_selection_clarification: DocumentSelectionClarification | None
research_target_selection: ResearchTargetSelection | None
```

Rules:

- revision 1 is the existing root shape without these slots;
- a legacy root missing `checkpoint_schema_revision` is interpreted only by the explicit migration path as revision 1;
- revision 1 normalizes to revision 2 with all four new slots set to `None` and preserves the existing synthesis compatibility rule;
- revision 2 requires `checkpoint_schema_revision` and all four keys; `discovery_need` and `discovery` may be `None` whenever the flag is off or discovery has not started, `document_selection_clarification` may be `None` whenever no document selection is pending, and `research_target_selection` may be `None` until finalized;
- `clarification` and `document_selection_clarification` are mutually exclusive, and the validator enforces it;
- an unknown revision or a revision-2 payload missing a required key fails closed;
- the context node clears all four new slots at the start of a new turn;
- after normalization, every new checkpoint write uses revision 2.

Enumerated seams (mirroring the sibling synthesis spec §17.1):

- `SupervisorV2State` is `total=True`, so the four keys must be added to the TypedDict;
- `_CHECKPOINT_REQUIRED_KEYS` in `contracts/validation.py` gains the four keys for revision 2;
- the revision-1 → revision-2 migration runs in `normalize_checkpoint_state` **before** `validate_checkpoint_payload`;
- `_SLOT_MODELS` in `supervisor_v2.py` gains `document_selection_clarification`, `discovery_need`, `discovery`, and `research_target_selection`, each mapped to a concrete model (never an `Annotated` union alias);
- `_NULLABLE_SLOTS` gains all four;
- `build_initial_v2_state` initializes all four to `None` and is the sole **ingress** writer of `checkpoint_schema_revision=2` for a fresh turn (the migration materializes the key for legacy payloads, and `shadow_runtime.build_shadow_bundle` either reuses `build_initial_v2_state` or is read as revision 1);
- `validate_supervisor_state` validates whichever is non-`None`;
- `validate_supervisor_state` deliberately ignores unknown root keys, so unknown-revision and missing-key fail-closed is owned by the payload gate **after** migration, not by the aggregate validator.

Centralize root migrations in one table/function rather than adding another unrelated one-off conditional beside the current synthesis migration. Tests must distinguish a valid revision-1 checkpoint from a corrupt revision-2 checkpoint.

## 7. Binding Provenance vs Research Selection

`ScopedDocument.role` remains immutable provenance and authorization state. Planner policies must no longer infer the current research assignment by scanning every binding role.

Add:

```python
class TargetSlot(ContractModel):
    slot_id: str
    intended_role: Literal["target", "reference"]
    subject_hint: str
    required: bool
    min_selections: int
    max_selections: int
    explicit_binding_ids: tuple[str, ...]
    source: Literal[
        "semantic_reference",
        "explicit_resource",
        "research_need",
    ]


class SlotBindingSelection(ContractModel):
    slot_id: str
    binding_ids: tuple[str, ...]


class ResearchTargetSelection(ContractModel):
    slot_bindings: tuple[SlotBindingSelection, ...]
    context_binding_ids: tuple[str, ...]
```

Validation requires:

- unique slot and binding IDs;
- every selected binding exists in `DocumentBindingSet`;
- each slot obeys `min_selections <= count <= max_selections`;
- a required slot meets its minimum;
- context bindings do not create coverage and are not passed to factual read tasks unless selected into a target/reference slot.

Examples:

| Request | Research selection |
|---|---|
| Quote A + “tóm tắt tài liệu này” | A fills the primary target slot. |
| Quote A + “tóm tắt Nghị định 13” | B/Nghị định 13 fills the target slot; A remains a context binding and creates no `TargetUnit`. |
| Quote A + “so sánh tài liệu này với Nghị định 13” | A fills side 1; discovered B fills side 2. |
| Thematic summary | One slot selects three documents by default and at most five. |

`SupervisorV2State.research_target_selection` is the durable owner. The complex boundary maps it explicitly into and out of child state, and evaluation/synthesis validate and consume the same checkpointed value. This ownership also applies when discovery is skipped; selection cannot be inferred from `DiscoveryCheckpoint`.

The summarize and compare skills consume `ResearchTargetSelection`, not all binding roles. Add a selection-aware plan validator:

```text
validate_task_plan(plan, bindings, target_selection=None)
```

It requires every `TargetUnit.binding_id` to be authorized by a selected logical slot and verifies the unit's logical target/reference use against that slot. A selected existing binding may retain an immutable `supporting`, `discovered`, `target`, or `reference` provenance role. Calls without `target_selection` retain the current target/reference binding-role restriction.

Threading the selection only into the skill planner is not sufficient, because the plan is validated at three separate points on the bootstrap path. All three must pass the checkpointed selection:

- `validate_checkpoint_node`'s own `validate_task_plan(initial.plan, bindings)` call in `v2/complex_research_graph.py` — this is the **first** check on the bootstrap path and sits outside any `try/except`, so it fails as a boundary error before the aggregate validator ever runs;
- `_validate_execution_state(execution, bindings, target_selection)`, reached from `validate_supervisor_state`, which `_wrap_node` runs on the merged `complex_boundary` update and on every later node entry;
- the planner entry point `v2/planning/planner.py` and `validate_research_planning_input`.

Plumbing: `ResearchPlanningInput` gains `target_selection: ResearchTargetSelection | None`, populated by `build_planning_input(state, runtime)` from the checkpointed root slot and passed to `build_governed_initial_proposal` and the summarize/compare skill policies, which validate their own plans. Validators therefore do not read state themselves; their callers pass the selection, and every call site defaults to `None` when it is absent, so flag-off and non-bootstrap states keep the strict role check.

Call sites that stay selection-blind and therefore strict are: `validate_fast_plan`, `validate_replan`/replan append, and every other skill's planning path. `validate_supervisor_state` must additionally validate any non-`None` `research_target_selection` against the current `DocumentBindingSet`. Evaluation and synthesis receive the same selection so primary and reference evidence cannot be misattributed.

### 7.1 Slot cardinality

- named summary primary: `min=1`, `max=1`;
- each compare side: `min=1`, `max=1`;
- thematic summary targets: `min=3`, `max=5`;
- optional readable references for named summary use a separate `reference` slot with `min=0`, `max=4`.

Multi-selection slots count **distinct documents**, not distinct revisions: two revisions of the same document can never satisfy two selections of one thematic slot. The first aggregate for a document identity wins the slot position, and the other revision remains lineage.

A document selected as readable reference receives a `TargetUnit`, a partial-read criterion, and a factual task. A generic `supporting` binding with no target unit is metadata only and must never be claimed as answer context.

## 8. Discovery Contracts

Create `backend/app/services/agents/v2/discovery/` with the following contracts.

### 8.1 Search probe

```python
class SearchProbe(ContractModel):
    probe_id: str
    slot_id: str
    query: str
    round: Literal[1, 2]
    origin: Literal["semantic", "planner_fallback"]
```

A validator reconstructs accepted probes server-side and enforces non-blank normalized queries, known slots, uniqueness, capability availability, round/probe budgets, and no normalized-equivalent repeats.

### 8.2 Candidate match and aggregation

```python
MatchKind = Literal[
    "exact_document_number",
    "exact_normalized_title",
    "lexical_title",
    "semantic",
]


class ProbeCandidateMatch(ContractModel):
    probe_id: str
    task_id: str
    candidate_id: UUID
    document_id: UUID
    document_revision: str
    rank: int
    confidence: float | None
    match_kind: MatchKind
    calibration_version: str | None


class SlotCandidateAggregate(ContractModel):
    aggregate_id: UUID
    slot_id: str
    document_id: UUID
    document_revision: str
    source_candidate_ids: tuple[UUID, ...]
    source_probe_ids: tuple[str, ...]
    source_task_ids: tuple[str, ...]
    best_match_kind: MatchKind
    aggregate_confidence: float | None
    best_rank: int
```

Aggregation groups by the authoritative aggregate key `(slot_id, document_id, document_revision)` — one aggregate per document identity per slot — and preserves every probe/candidate/task lineage edge.

A second, finer key is used only for verification: `(task_id, document_id, document_revision)` identifies a candidate match produced by one search task. The expansion validator accepts a selection when any member of the aggregate's `source_task_ids` matches that key.

That separation matters for durability: candidate IDs are minted with `uuid4()` (`v2/discovery.py`, `V1DocumentSearchService`), which is deliberate and stays unchanged. A candidate UUID is therefore **non-authoritative lineage only**. Matches and aggregates are rebuilt on every entry from checkpointed `AgentResult`s plus the live registry. A repeated search attempt that mints new candidate UUIDs still converges to the same per-slot aggregates.

Deterministic ordering is:

```text
exact_document_number
> exact_normalized_title
> lexical_title
> semantic
```

`aggregate_id` is a server-owned deterministic UUID derived from the checkpoint namespace, slot ID, document ID, and exact revision — the same tuple as the aggregate key, so it is unique per aggregate. It is derived from checkpointed facts, never minted, so it is stable across a repeated attempt even though candidate UUIDs are not. It is the selectable identity; source candidate IDs remain non-authoritative lineage only.

For every non-exact aggregate, including lexical-only, semantic-only, and mixed lexical/semantic matches:

```text
aggregate_confidence = max(match-kind-calibrated confidence across source matches)
```

Lexical and semantic calibrators may use different feature mappings but must produce the same probability-like confidence domain and record their mapping version. A mixed aggregate keeps the strongest match kind for ordering and the maximum calibrated confidence for its gate. Using the maximum avoids inflating confidence merely because the planner emitted more probes. Ties break by best rank, then first accepted probe order, then stable document identity.

The confidence margin is defined **at the selection boundary**, not at the top two:

```text
single-select slot (min_selections == 1):
    margin = confidence(#1) - confidence(#2)
single-select slot with only one surviving aggregate:
    margin = confidence(#1) - 0.0
multi-select slot (min_selections > 1):
    margin = confidence(last selected) - confidence(first excluded)
```

For a multi-select slot the last selected aggregate is the `min_selections`-th candidate when the slot is filled to its minimum, or the `max_selections`-th once it is saturated. A thematic slot whose top five all clear the confidence gate therefore never clarifies merely because #1 and #2 are close. When a multi-select slot has no excluded aggregate, the margin gate passes.

Exact matches ignore confidence; multiple distinct exact matches remain ambiguous rather than being resolved by score.

### 8.3 Selection

```python
class DiscoverySelection(ContractModel):
    slot_id: str
    selected_aggregate_ids: tuple[UUID, ...]
    authority: Literal[
        "exact_match_policy",
        "confidence_policy",
        "user_choice",
    ]
```

Selection count applies to aggregate IDs and must satisfy slot cardinality. Thematic summary can select multiple aggregates for one slot. Selection settlement resolves each aggregate to its authoritative document identity and retains all source candidate/probe/task IDs as provenance without counting them as separate selections.

Because `aggregate_id` is derived from `(slot_id, document_id, revision)`, settlement never needs a candidate UUID to resolve a selection, and the same selection resolves to the same identity after a repeated search attempt.

### 8.4 Discovery checkpoint

```python
class DiscoveryCheckpoint(ContractModel):
    target_slots: tuple[TargetSlot, ...]
    accepted_probes: tuple[SearchProbe, ...]
    candidate_matches: tuple[ProbeCandidateMatch, ...]
    slot_aggregates: tuple[SlotCandidateAggregate, ...]
    selections: tuple[DiscoverySelection, ...]
    rounds_consumed: int
    probes_consumed: int
    status: Literal[
        "searching",
        "ranking",
        "clarification",
        "selected",
        "unavailable",
    ]
    clarification_manifest: DocumentSelectionManifest | None
```

`SupervisorV2State.discovery` is a required revision-2 key with canonical idle value `None`. The complex boundary maps it explicitly into and out of child state together with the independently owned `research_target_selection`. `ComplexResearchState` is extended with `discovery`, `research_target_selection`, and `pending_clarification` as nullable members, and `build_complex_research_state` / `normalize_complex_state` / `merge_complex_result_into_supervisor` are explicit change points (see §15.1).

## 9. Document Identity Search and Confidence Calibration

Bootstrap reuses `document.search`, but the backing identity-search service must expose enough authoritative metadata for selection.

### 9.1 Identity-search pipeline

For each probe:

```text
normalize query
  → parse document number
  → exact workspace-scoped metadata lookup by normalized number
  → exact normalized-title lookup
  → bounded lexical title lookup
  → semantic document-level vector probe
  → deduplicate authoritative identity
  → one bounded batch rerank
  → calibration
```

Exact metadata lookups are server-owned and ACL/workspace scoped. The semantic probe extends the current lightweight vector-only discovery helper, which already obtains distances but currently discards them and performs no rerank.

The revised adapter may preserve the best chunk text per document only ephemerally for one rerank call. Candidate text is never checkpointed, logged, traced, placed in a model-facing observation, or returned in clarification.

### 9.2 Match kinds

- a validated exact normalized document number can auto-select without semantic confidence;
- a unique validated exact normalized title can auto-select without semantic confidence;
- lexical-title and semantic matches require confidence and margin gates;
- conflicting exact matches clarify.

### 9.3 Calibrator

Reranker and similarity scores are not probabilities. Add a model-versioned `DiscoveryConfidenceCalibrator`.

The initial calibrator is an offline-fitted monotonic isotonic mapping over labeled `(query, candidate, relevant)` examples. Evaluation is stratified by query shape and workspace, but the mapping is global per retrieval/reranker model version; per-workspace calibration is excluded initially because sparse workspace labels would be unstable.

The calibration artifact records:

- retrieval model hash;
- reranker model hash;
- labeled dataset hash;
- mapping version and knots;
- fit metrics and evaluation timestamp.

A missing or hash-mismatched artifact disables lexical/semantic auto-selection. Exact number/title matches remain eligible; other candidates clarify. Canary enablement is blocked until the artifact passes the labeled-data gates.

Initial calibrated thresholds:

```text
confidence >= 0.82
selection-boundary margin (§8.2) >= 0.12
```

Thresholds apply only to calibrated output. Changing model hashes, artifact, threshold, or margin requires rerunning the rollout gate.

## 10. Discovery Admission and Probe Generation

Probe generation is hybrid:

1. deterministic probes come from resolved references, normalized numbers/titles, compare sides, and thematic subjects;
2. if a required slot lacks a usable probe, the governed planner may propose additional query text only for existing slots;
3. the validator rebuilds accepted probes server-side.

Budgets:

- at most five probes total;
- at most two rounds;
- top five candidates per probe;
- probes in a round may run in parallel when the capability descriptor permits;
- round two requires a recorded unresolved slot/gap from round one;
- normalized-equivalent probes cannot repeat.

If reconciliation satisfies every required slot, no probe is emitted and no search dependency is introduced.

## 11. Scheduler-Compatible Deadline and Budgets

### 11.1 Derived deadline

Discovery starts with:

```text
discovery_deadline = min(
    request_deadline,
    discovery_started_at + V2_DISCOVERY_DEADLINE_SECONDS,
)
```

Build a derived immutable `CapabilityRuntimeContext` with the same request/run/user/workspace/capability permissions and only the narrowed `deadline_at`. Dispatch still goes through the shared `TaskScheduler`; discovery nodes never wrap capability calls in a direct `asyncio.wait_for` and never create another scheduler.

Outer cancellation and kill-switch behavior remain authoritative.

### 11.2 Restart and dispatch semantics

The existing scheduler checkpoints `AgentResult` only after `capability.execute()` returns (`TaskScheduler.execute_ready_tasks` re-dispatches every plan task that has no checkpointed result, and `complex_execute_node` returns `{"task_results": report.results}`). The supported guarantee is therefore explicit **at-least-once read execution**, not zero dispatches in the post-capability/pre-checkpoint crash window.

What the design actually guarantees:

- a search task whose result is already checkpointed is never re-executed on resume;
- a crash after the read call but before its result checkpoint may repeat that read;
- discovery capabilities must remain side-effect free;
- candidate UUIDs stay `uuid4()` (no determinism is claimed or required);
- every authoritative identity is re-derivable from checkpointed facts: aggregates partition by `(slot_id, document_id, document_revision)`, candidate matches key on `(task_id, document_id, document_revision)`, `aggregate_id` is derived from the aggregate key, candidate registry is rebuilt per entry, selection settlement and binding resolution deduplicate on `(document_id, revision)`, and plan expansion deduplicates on the aggregate key;
- no duplicate **checkpointed** tasks, results, bindings, or expansion units is committed for the same authoritative key;
- no lease survives for an un-checkpointed use: leases are DB rows keyed by `(run_id, revision_id, evidence_use_id)`, so a repeated read's fresh use ID yields a separate row that expires by TTL rather than a duplicate for one use;

It is explicitly **not** guaranteed that a repeated attempt leaves no orphan rows. A re-dispatched read mints fresh evidence-use and acquisition IDs (`capabilities/document.py` `acquisition_id = uuid4()`, and the people capability does the same), so the un-checkpointed attempt's leases and acquisition rows are orphaned and expire by their own TTL. "No duplicate leases" therefore means no second lease for one evidence use, not zero rows.

Durable pre-dispatch receipts or capability result caching would be needed for exactly-once dispatch and are outside this feature. The rollout and tests must not claim that stronger guarantee.

### 11.3 Separate budgets

Add:

```python
class DiscoveryBudgetView(ContractModel):
    probes_remaining: int
    rounds_remaining: int
    top_k: int


class ResearchBudgetView(ContractModel):
    max_tasks_remaining: int
    max_replans_remaining: int
    max_parallel_branches: int
```

Discovery tasks and factual research tasks remain in one plan lineage but have separate counters. `build_research_budget_view` counts only factual tasks, identified by task origin; `DiscoveryProbeTaskOrigin` tasks do not consume research capacity.

Add an absolute safety cap:

```text
V2_TOTAL_MAX_TASKS=13
```

with default discovery maximum 5 plus research maximum 8. The cap is enforced at the seams that can admit tasks: the expansion validator, the replan budget, and the scheduler's plan-entry check, which gains the total limit as an explicit parameter. `validate_task_plan` itself stays cap-free so fast/legacy plans are unaffected.

## 12. Existing Discovery Composition and Binding Ownership

Bootstrap reuses:

- `DocumentSearchCapability`;
- `DocumentDiscoveryCandidate`;
- persisted `AgentResult`;
- `DiscoveryCandidateRegistry`;
- the existing binding resolver;
- ACL checks and exact-revision pinning.

It does not create a second candidate namespace.

Settle ownership is disjoint:

- existing `discovery_settle_node` processes only non-bootstrap search task IDs, including R37/recovery, and can add only `discovered` or `supporting` bindings;
- new `selection_settle_node` processes only bootstrap task IDs recorded in `DiscoveryCheckpoint` and can create intent-selected target/reference bindings;
- generic settle excludes bootstrap task IDs;
- selection settle rejects candidates not owned by bootstrap outcomes;
- both paths share candidate registry and resolver helpers and deduplicate `(document_id, revision)`;
- `discovery_deferred` and `V2_MAX_DISCOVERED_DOCUMENTS` remain generic-settle state/policy;
- bootstrap unresolved candidates remain in `DiscoveryCheckpoint` and use bootstrap slot/probe budgets.

Add `BindingSelectionRequest`; existing `BindingAdditionRequest` and `BindingPromotionRequest` remain unchanged and cannot create a target.

Selection provenance records:

- selected binding ID;
- source search task IDs;
- selected aggregate IDs;
- all source candidate IDs and probe IDs behind those aggregates (non-authoritative lineage);
- target slot ID;
- authority: exact policy, confidence policy, or user choice.

If the document is already bound, the resolver reuses its exact pinned identity and records turn-specific selection without minting a duplicate binding. If a new role-bearing binding is required, the resolver creates it only through `BindingSelectionRequest`, ACL revalidation, exact pinning, provenance audit, and lease commit.

## 13. Task Origins and Plan Expansion

Extend the `TaskOrigin` union in `v2/contracts/planning.py` with:

```python
class DiscoveryProbeTaskOrigin(ContractModel):
    kind: Literal["discovery_probe"]
    probe_id: str
    slot_id: str
    round: Literal[1, 2]


class DiscoveryExpansionTaskOrigin(ContractModel):
    kind: Literal["discovery_expansion"]
    source_search_task_ids: tuple[str, ...]
    target_slot_ids: tuple[str, ...]
    selected_aggregate_ids: tuple[UUID, ...]
```

`_validate_task_origin` currently assumes every non-`InitialTaskOrigin` carries `reason`, `task_ids`, and `evidence_use_ids`, so a bootstrap plan raises `AttributeError` — which is not `ContractValidationError` and therefore escapes `validate_checkpoint_node`'s `except` clause as a typed error. The validator must dispatch by origin kind: `ReplanTaskOrigin` keeps its existing checks, `DiscoveryProbeTaskOrigin` validates probe/slot/round lineage, and `DiscoveryExpansionTaskOrigin` validates that its search task IDs and target slot IDs exist and that `selected_aggregate_ids` resolve within the checkpoint. Both new members join the `TaskOrigin` discriminated union.

These origins make discovery/research budget accounting and audit lineage explicit:

```text
query → slot → probe → search task → candidate aggregate → selection
→ binding → TargetUnit → factual task → evidence
```

### 13.1 Expansion validator

Add a separate frozen validator:

```text
validate_plan_expansion(
    current,
    proposed,
    discovery_checkpoint,
    target_selection,
    selected_bindings,
    discovery_outcomes,
    research_budget,
    total_task_limit,
) -> TaskPlan
```

It is not `validate_replan`; the existing rule that replans cannot change target units remains unchanged.

The expansion validator enforces:

1. plan ID, goal, contract version, and every existing task are unchanged;
2. existing tasks and target units are exact prefixes;
3. at least one target unit and factual task are appended;
4. every new unit maps to a checkpoint-selected slot and existing selected binding;
5. slot cardinality and required minimums hold;
6. new target/task IDs are unique and cannot shadow bootstrap IDs;
7. new tasks reference only selected/appended units and current catalog capabilities;
8. every new task carries `DiscoveryExpansionTaskOrigin` with valid bootstrap lineage;
9. discovery outcomes cover every attempted bootstrap task;
10. selected aggregates belong to those outcomes: at least one member of `source_task_ids` matches the outcome's `(task_id, document_id, document_revision)` key;
11. discovery, research, and total task budgets all hold;
12. the complete plan passes selection-aware `validate_task_plan` against current bindings and the checkpointed `ResearchTargetSelection`.

`expand_discovery_plan` is the single authoritative constructor for this transition. It appends units/tasks, invokes `validate_plan_expansion`, acquires leases, and returns nothing checkpointable before validation and lease commit.

### 13.2 Structural ownership guard

Update `test_subagent_cannot_append_authoritative_tasks` deliberately:

- `append_replan_tasks` remains the only replan append surface;
- `expand_discovery_plan` is the only discovery-expansion append surface;
- its only governed caller/plan-return owner is `research_expand_node`;
- the AST scanner covers target-unit appends as well as task appends;
- explicit constructor/caller allowlists remain; wildcard or directory allowlists are forbidden.

## 14. Skill Semantics

### 14.1 Named summary

- one primary target slot, exactly one selection;
- optional readable references use a separate slot with at most four selections;
- each readable reference receives a target unit with partial-read coverage;
- evidence/synthesis receives `ResearchTargetSelection` and cannot render reference content as if it belonged to the primary target.

### 14.2 Thematic summary

- one multi-selection target slot;
- default minimum three, maximum five;
- one map/read lineage per selected target;
- evaluator requires coverage per target;
- rollout is deferred until named summary and compare gates pass.

### 14.3 Comparison

Comparison is defined by exactly two logical side slots and target units, not by two unique document bindings.

Cross-document compare:

```text
side 1 → binding A / locator A
side 2 → binding B / locator B
```

Same-document section compare:

```text
one binding A
side 1 TargetUnit → binding A / Chapter I locator
side 2 TargetUnit → binding A / Chapter III locator
```

The compare skill no longer requires one target-role binding plus one reference-role binding as its cardinality source. It requires exactly two selected side units, allows a shared binding only when finalized semantics provide distinct section locators, and rejects an accidental same-document collision.

## 15. Clarification and Resume

Do not add a second interrupt protocol and do not rebind the persisted clarification slot to a `typing.Annotated` union: the checkpoint coercion seams (`_SLOT_MODELS`, `_coerce_slot`, `_optional_model`, `_as_model`) are model-typed and call `isinstance`/`model_validate`/`__name__`, which an alias cannot satisfy. Keep concrete classes and use **two mutually exclusive nullable root slots**:

- the existing `ClarificationRequest` keeps its name and frozen fields and additively gains `kind: Literal["semantic"] = "semantic"`;
- a new sibling slot `document_selection_clarification: DocumentSelectionClarification | None`;
- a validator rule that at most one of them is non-`None`.

```python
class DocumentSelectionChoice(ContractModel):
    choice_token: str
    title: str
    document_number: str | None


class DocumentSelectionSlot(ContractModel):
    slot_id: str
    slot_label: str
    min_selections: int
    max_selections: int
    choices: tuple[DocumentSelectionChoice, ...]


class DocumentSelectionClarification(ContractModel):
    kind: Literal["document_selection"]
    contract_version: ContractVersion
    clarification_id: str
    question: str
    slots: tuple[DocumentSelectionSlot, ...]
    expires_at: datetime


class DocumentSelectionManifestEntry(ContractModel):
    choice_token_digest: str
    slot_id: str
    aggregate_id: UUID
    document_id: UUID
    document_revision: str


class DocumentSelectionManifest(ContractModel):
    clarification_id: str
    entries: tuple[DocumentSelectionManifestEntry, ...]
    expires_at: datetime
    status: Literal["pending", "consumed"]
```

The manifest is internal checkpoint state inside `DiscoveryCheckpoint`. Choice tokens are random opaque values; only an HMAC digest is persisted. Entries must refer to aggregates in the same checkpoint and exactly reproduce their document identity/revision. The manifest and public request share clarification ID and expiry.

Revision-1 checkpoint migration injects `kind="semantic"` into legacy clarification request mappings. Because the two request shapes are stored in separate slots, the seams that must dispatch are:

- `_SLOT_MODELS` / `_NULLABLE_SLOTS` / `_coerce_slot` — add the new slot as a concrete model; the existing `clarification` entry is unchanged;
- `validate_supervisor_state` / `validate_clarification_request` — validate whichever slot is set and enforce mutual exclusion;
- `_retire_stale_clarification` — must skip a live document-selection request instead of validating it as semantic;
- `nodes/clarification.py` `_live_persisted_request` / `_request_from_mapping` / `clarify_node` / `interrupt_for_clarification` — `clarify_node` falls back to `build_clarification(state["semantic"])` whenever `_live_persisted_request` (which reads only the `clarification` slot) returns `None`; with a live `document_selection_clarification` that fallback would mint a semantic request and violate the mutual-exclusion rule, so the persist stage must re-emit the sibling slot unchanged and never regenerate from `semantic`;
- `supervisor_v2.py` `clarify_wait_node` and `_resolution_from_mapping` — dispatch resume by the pending slot and by `payload["kind"]`;
- `v2/events.py` `clarification_public_metadata` — currently reads `candidates` and silently emits `options: []` for an unknown shape, which would make the persisted public resume block choice-less and degrade every structured document-selection reply into a fresh turn; it must project the per-slot choices;
- `streaming.py` `_coerce_clarification_request`, `_v2_suspend_request`, `_emit_suspend_turn`, `load_v2_pending_clarification`, `resolve_v2_resume_command`, `prepare_v2_resume_command` — including the suspend turn's user-visible content, which currently reads `pending.question`, hence the `question` field above;
- the runner ingress schema `ClarificationSelection` and its entry points — `api/chat_session.py` currently carries a single `selected_option_id`, while `api/chat_agent_lg.py` and `services/integrations/telegram_service.py` call `resolve_v2_resume_command` with no selection arguments at all; all three gain an additive per-slot `slot_selections` payload for document selection while the semantic path is unchanged;
- `nodes/clarification.py` `resume_clarification` — added as a document-selection resume constructor;
- `v2/transport.py` `build_clarification_required` / `validate_clarification_selection` — gain an additive document-selection frame carrying per-slot choices (at most three), `slot_label`, and `min/max_selections`, with no UUID, revision, score, digest, or internal ID.

The resolution payload is runner-supplied JSON, not checkpointed state, so it is dispatched by its `kind` key without a `TypeAdapter`: the existing shape becomes the `"semantic"` member (`kind` defaulted for compatibility), and document selection uses:

```python
class DocumentSlotResolution(ContractModel):
    slot_id: str
    choice_tokens: tuple[str, ...]


class DocumentSelectionResolution(ContractModel):
    kind: Literal["document_selection"]
    contract_version: ContractVersion
    clarification_id: str
    selections: tuple[DocumentSlotResolution, ...]
    declined: bool
```

`_resolution_from_mapping` selects the concrete class by `payload["kind"]` (defaulting to the semantic shape when absent), so existing semantic resumes are unchanged.

`declined=True` requires no selections. Otherwise each unresolved required slot appears exactly once and obeys manifest cardinality.

### 15.1 Parent/child ownership, resume authority, and terminal shapes

The complex child does not call `interrupt()` directly. On ambiguity it terminates that invocation with `DiscoveryCheckpoint.status="clarification"`, the pending manifest, and a `DocumentSelectionClarification` in `ComplexResearchState.pending_clarification`. Three existing seams are explicit change points:

- `build_complex_research_state` maps root `discovery`, `research_target_selection`, and `document_selection_clarification` into the child input on every entry;
- `normalize_complex_state` is extended with the three new child slots so a resumed checkpoint keeps them instead of dropping unknown keys;
- `merge_complex_result_into_supervisor` returns the child's discovery checkpoint, pending document-selection request, and finalized selection in addition to `execution_update(...)`.

A resumed child continues from its own checkpoint under the parent saver, so the root values written by `clarify_wait` are the **authoritative** input on re-entry: `build_complex_research_state` re-injects the root `discovery` (with its consumed manifest and recorded selections) and `research_target_selection`, overriding the child's stale copy. A new deterministic `discovery_entry_branch` selects the child's first node. The plan-present case is checked **first** and keeps today's semantics: a checkpoint that already has a `plan` goes to the existing `validate_checkpoint_node` plan-present arm and bypasses discovery entirely. Only when no plan exists does the branch consider discovery: `status="clarification"` plus recorded selections goes to `selection_settle`; satisfied slots go to `finalize_target_selection`; otherwise `discovery_propose`.

`complex_boundary` atomically maps the discovery checkpoint, the pending request, and the root `document_selection_clarification` slot in one node update, and never returns a `Command` (navigation stays static). `_complex_branch` gains a `"clarify"` return when that update carries a pending document-selection request, and its edge map gains `"clarify": "clarify"` so the existing `clarify_persist` → `clarify_wait` pair checkpoints the request and manifest before interrupting. `_complex_branch` continues to return `"synthesize"`/`"finalizer"` otherwise. A document-selection suspend deliberately keeps `route_decision.route` at its complex value rather than `"clarify"`, so the existing `route == "clarify" requires clarification` invariant is untouched.

On resume, `clarify_wait` dispatches by the pending slot:

- the semantic slot keeps the current validation/application path and returns `Command(goto="binding")`;
- the document-selection slot validates clarification ID, expiry, token digests, slot ownership, uniqueness, cardinality, current ACL/workspace visibility, and exact revision availability against the root discovery manifest, marks it consumed, records aggregate selections, clears the pending request, and returns `Command(goto="complex_boundary")`, which re-enters through `discovery_entry_branch` directly into `selection_settle`;
- a declined or failed selection clears the pending request, sets the root discovery checkpoint to `status="unavailable"`, clears the route, and terminates with a typed response plus the safe request for a name, number, or reformulated query. It never falls through to targetless retrieval.

`selection_settle` pins the selected identities, commits leases/provenance, and only then checkpoints bindings and `ResearchTargetSelection`. It immediately precedes `finalize_target_selection`, which is also the no-discovery path, so both admission outcomes converge before research planning.

The public projection exposes at most three safe choices per unresolved slot and never exposes document UUID, revision, workspace, ACL, score, confidence, evidence ID, use ID, internal candidate ID, aggregate ID, or manifest digest.

## 16. Failure Behavior

| Condition | Required behavior |
|---|---|
| Required slots already satisfied | Skip discovery; search outage cannot fail the request. |
| No candidate after two rounds | Clarify for a name/number; never choose arbitrarily. |
| Search timeout/dependency failure with missing slots | Typed unavailable or clarification; no targetless bypass. |
| Candidate appears in several probes | Aggregate once per `(slot_id, document_id, revision)` and preserve all probe/candidate/task lineage. |
| Search result already checkpointed | Never re-execute on resume. |
| Crash between read and result checkpoint | The side-effect-free read may repeat; authoritative state converges by key; orphan leases/uses expire by TTL. |
| Multiple distinct exact matches | Clarify. |
| Semantic confidence or margin below threshold | Clarify; margin is the selection-boundary margin from §8.2, so a saturated multi-select slot does not clarify on a top-two tie. |
| Calibration artifact missing/hash-mismatched | Exact matches may auto-select; lexical/semantic matches clarify; canary readiness fails. |
| Accidental same document for two compare sides | Reject; allow only explicit same-document distinct-locator compare. |
| ACL changes before resume | Hide sensitive reason; refresh choices or return typed denial. |
| Selected revision no longer pinnable | Reject; never move silently to a newer revision. |
| User declines every choice | Clear the pending request, mark discovery `unavailable`, terminate typed with a request for a name/number; never bypass to targetless retrieval. |
| Budget exhausted | Clarify or typed unavailable; no additional search. |
| Cancellation/kill switch | Terminal cancellation; planner and synthesis do not continue. |
| Invalid expansion | Fail closed before research dispatch. |

## 17. Configuration

Declare and validate:

```text
V2_DISCOVERY_BOOTSTRAP_ENABLED=false
V2_DISCOVERY_MAX_PROBES=5
V2_DISCOVERY_MAX_ROUNDS=2
V2_DISCOVERY_TOP_K=5
V2_DISCOVERY_DEADLINE_SECONDS=15
V2_DISCOVERY_SUMMARY_TARGETS=3
V2_DISCOVERY_MAX_SUMMARY_TARGETS=5
V2_DISCOVERY_CONFIDENCE_THRESHOLD=0.82
V2_DISCOVERY_MARGIN_THRESHOLD=0.12
V2_DISCOVERY_CALIBRATION_ARTIFACT=/app/config/discovery-calibration.json
V2_MAX_TASKS=8
V2_TOTAL_MAX_TASKS=13
```

The research task cap reuses the existing declared `V2_MAX_TASKS` (currently the live setting consumed by `V2ResearchLimits.from_settings()`); `V2_RESEARCH_MAX_TASKS` is not introduced, to avoid a second name for one limit. `V2_TOTAL_MAX_TASKS` is new and defaults to the discovery plus research maxima.

Validation enforces positive limits, summary minimum no greater than maximum, thresholds in `[0,1]`, total task cap no smaller than discovery plus research defaults, and runtime discovery deadline no later than the outer deadline.

Enabling semantic auto-selection requires a readable calibration artifact whose model hashes match the effective providers.

Settings already consumed by `V2ResearchLimits.from_settings()` and generic discovery policy must also remain declared and validated:

```text
V2_MAX_TASKS
V2_MAX_PARALLEL_BRANCHES
V2_MAX_REPLANS
V2_ALLOW_REFERENCE_DISCOVERY
V2_ALLOW_SUPPORTING_DISCOVERY
V2_MAX_DISCOVERED_DOCUMENTS
```

Environment values ignored by Pydantic are not configuration support.

## 18. Module Boundaries

Create:

```text
backend/app/services/agents/v2/discovery/
  __init__.py
  contracts.py
  policy.py
  calibration.py
  projection.py
  nodes.py
  plan_expansion.py
```

Integrate with:

- `v2/contracts/base.py` and `state.py` for checkpoint schema revision and the four new root slots;
- `v2/contracts/clarification.py` for the concrete `DocumentSelectionClarification`/`DocumentSelectionResolution` pair and the manifest;
- `v2/contracts/planning.py` for the two new `TaskOrigin` members and budgets;
- `v2/contracts/validation.py` for `_CHECKPOINT_REQUIRED_KEYS`/revision dispatch, `_validate_task_origin` kind dispatch, `_validate_execution_state`/`validate_research_planning_input` selection threading, and the expansion validator;
- `supervisor_v2.py` for `_SLOT_MODELS`/`_NULLABLE_SLOTS`/`_coerce_slot`, the migration in `normalize_checkpoint_state`, `build_initial_v2_state`, `_retire_stale_clarification`, `_resolution_from_mapping`, `_complex_branch`, and the `clarify` edge map;
- `v2/nodes/routing.py` for the flag-gated `decide_route` admission branch;
- `v2/nodes/clarification.py` for `_live_persisted_request`/`_request_from_mapping`/`clarify_node` and the document-selection resume constructor;
- `v2/complex_research_graph.py` for topology, `discovery_entry_branch`, the three state-mapping seams, `ResearchPlanningInput.target_selection`, `build_planning_input`, settle separation, and suppression of the `build_initial_proposal` unscoped fallback;
- `v2/capabilities/document.py` and the v1-backed adapter for identity matches/ranking;
- `v2/skills/summarize/` and `compare/` for selection-driven plans;
- `supervisor_v2.py` and `runtime_selector.py` for wiring, and `services/agent/shadow_runtime.py` for shadow ingress;
- `agent/streaming.py` and `v2/events.py` for progress and the document-selection suspend/resume frame (`clarification_public_metadata` included);
- `v2/transport.py` for the additive document-selection public frame and selection validation;
- `schemas/rag.py` and the runner entry points (`api/chat_session.py`, `api/chat_agent_lg.py`, `services/integrations/telegram_service.py`) for the additive per-slot selection payload;
- `core/config.py` and `.env.example` for controls.

Keep scoring, calibration, selection, role reconciliation, and expansion outside the already large `complex_research_graph.py` except for graph composition and state mapping.

## 19. Observability and Privacy

Record metadata only:

- slots resolved from explicit bindings versus discovery;
- probes/rounds and candidates per probe;
- match-kind counts;
- auto-selection, clarification, and “none” rates;
- aggregate confidence bands, never raw query/candidate text;
- discovery duration and timeout;
- selection authority;
- offline target-slot and selected-target correctness.

Do not log query text, titles, document numbers, IDs, revisions, prompts, candidate text, user selections, evidence, or ACL facts in timing/rollout metadata. Content-suppressed tracing applies to discovery planner and reranker calls.

## 20. Implementation Phases

### Phase 1 — Contract groundwork

Implement checkpoint schema revision/migration with the enumerated seams, the four root slots, target slots/cardinality, research target selection and its validator/planner threading, candidate match/aggregation on the two keys, selections, task-origin members and kind dispatch, budgets, the two concrete clarification slots, and the expansion validator. Graph behavior remains disabled.

### Phase 2 — Identity-search and calibration

Add exact-number, exact-title, lexical-title, semantic document search, bounded rerank, model-versioned calibration artifact loading, and labeled golden data.

### Phase 3 — Named summary and compare bootstrap

Add slot derivation/reconciliation, conditional discovery admission, probe validation/execution, aggregation, deterministic selection, and same-document comparison semantics.

### Phase 4 — Binding and plan expansion

Add selection settle, ACL/pin/lease/provenance, `expand_discovery_plan`, AST ownership guard updates, and crash/resume tests.

### Phase 5 — Clarification/resume

Add the document-selection suspend/resume round trip: public per-slot frame, opaque choice manifest, multi-slot resolution, declined-selection terminal shape, expiry/ACL/revision validation, and same-checkpoint resume.

### Phase 6 — Thematic summary

Enable three-to-five target thematic map/reduce only after prior gates pass.

## 21. Testing

### 21.1 Contract and unit tests

- revision-1 checkpoint migration, all revision-2 required nullable slots, new-turn clearing, revision-2 missing-key rejection, and `clarification`/`document_selection_clarification` mutual exclusion;
- route precedence proving unresolved summarize/compare document identity reaches complex discovery instead of semantic clarification, with the flag off preserving the clarify result;
- immutable binding role versus checkpointed research selection, including `_validate_execution_state` and `validate_research_planning_input` threading the selection while fast/replan callers stay strict;
- `_validate_task_origin` kind dispatch for both new origins (no `AttributeError` escaping as a typed error);
- slot cardinality for named, compare, optional references, and thematic summary, including that two revisions of one document cannot fill two selections of one multi-select slot;
- conditional discovery admission and flag-off compatibility;
- exact/lexical/semantic identity search;
- deterministic derived aggregate IDs over the aggregate key, one aggregate per document identity per slot, multi-probe deduplication, lexical/semantic/mixed confidence aggregation, tie-breaking, and the single-select versus multi-select margin boundary;
- calibrator artifact/model-hash validation;
- separate discovery/research/total budgets;
- derived discovery deadline;
- `validate_plan_expansion` acceptance and every mutation/rebinding rejection;
- AST ownership guard coverage for task and target-unit appends;
- concrete clarification slots, manifest integrity, parent/child checkpoint handoff through the three state-mapping seams, resume authority from root slots, `_complex_branch` clarify return, `discovery_entry_branch` resume target, and public redaction;
- a `streaming`/`transport` suspend→resume round trip for a document-selection request;
- at-least-once crash-window convergence without duplicate checkpointed authoritative state.

### 21.2 Required scenarios

1. Named summary without `document_ids`: exact number → target → read → summarize.
2. Quoted A, summarize named B: A remains context; B is the only primary target.
3. Compare explicit A with discovered B.
4. Compare two discovered documents with independent slots.
5. Compare two sections of one document: one binding, two locators/target units.
6. Explicit complete A/B with search backend down: discovery skipped and research continues.
7. Thematic summary selects three to five targets and evaluates each target separately, without clarifying when all five clear the confidence gate.
8. Ambiguous selection interrupts with at most three safe choices per slot; the second pass of `complex_boundary` is checkpointed with the request before suspension, and resume re-enters at `selection_settle` without dispatching another probe.
9. Forged, expired, foreign-workspace, ACL-revoked, or revision-stale choice is rejected.
10. Candidate repeated across probes aggregates once with complete lineage.
11. Resume after a checkpointed search result does not re-execute it; a forced post-call/pre-checkpoint crash may repeat the side-effect-free read but converges to deduplicated candidate, binding, lease, and plan state with no revision drift.
12. R37 generic discovery remains supporting/discovered-only and cannot consume bootstrap candidates.
13. Flag off preserves current fast paths and `retrieve-unscoped` fallback.
14. Flag on replaces fallback only when summarize/compare has missing/conflicting slots.
15. Evaluate/multi-goal/cross-domain retain current targetless fallback.
16. Flag-off routing is byte-compatible: an open required document reference still routes to semantic clarification, and the four new slots stay `None` with no discovery state written.

### 21.3 Security assertions

- model-minted IDs, roles, slots, and choice tokens are rejected;
- unauthorized candidates do not appear in choices or distinguishable errors;
- public/model projections contain no raw UUID, revision, ACL, raw score, evidence ID, use ID, secret, or candidate text;
- all dispatch remains scheduler-owned.

## 22. Rollout

1. Fit and version calibration artifact; keep feature flag off.
2. Run unit/integration/golden tests and shadow traffic.
3. Enable named summary/compare for internal workspace allowlist.
4. Roll out 5% → 25% → 50% → 100% of v2-eligible traffic.
5. Enable thematic summary in a separate gated stage.

Promotion gates:

- zero workspace/ACL leaks;
- zero duplicate checkpointed tasks, results, bindings, leases, or expansion units across crash/resume;
- documented at-least-once read dispatch passes forced crash-window convergence tests;
- target-slot resolution accuracy at least 95%;
- selected-target accuracy at least 95%;
- semantic auto-selection false-positive rate at most 1%;
- calibration hashes match effective models;
- discovery p95 at most 15 seconds when discovery is required;
- explicit-complete requests introduce no discovery dependency;
- clarification/typed-unavailable rates reviewed against baseline;
- no grounding/citation regressions.

Auto-selection optimizes precision over recall. Uncertainty clarifies rather than selecting the wrong legal/administrative document.

The feature flag and v2 kill switch provide rollback. Disabling bootstrap restores existing v2 behavior and never changes v1.

## 23. Definition of Done

The feature is complete when:

1. checkpoint schema migration distinguishes valid legacy state from corrupt current state;
2. planner skills use `ResearchTargetSelection` rather than scanning all immutable binding roles;
3. named summary and compare succeed without supplied IDs when confident authorized candidates exist;
4. explicit complete requests skip discovery;
5. quoted context does not override a different query-named target;
6. thematic slot cardinality and multi-target coverage are enforced before its rollout;
7. multi-probe aggregation is deterministic and auditable;
8. semantic auto-selection uses a versioned calibrated mapping, not raw score;
9. bootstrap and generic discovery share candidate identity but have disjoint role authority;
10. `validate_plan_expansion` and AST guards prove one governed expansion owner;
11. discovery and research dispatches are validated, checkpointed, leased, and scheduler-owned;
12. clarification uses the existing interrupt protocol with two mutually exclusive concrete request slots and resumes the same checkpoint;
13. same-document comparison uses one binding and two target units;
14. hard probe, round, research, total-task, deadline, cancellation, and kill-switch budgets hold;
15. checkpointed work is not re-executed, uncheckpointed read dispatch is safely repeatable, and duplicate authoritative state cannot be committed;
16. security, durability, precision, latency, grounding, and rollout gates pass.

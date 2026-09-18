# Discovery Bootstrap Phase 1 — Contract Groundwork Implementation Plan

**Date:** 2026-09-17
**Status:** Revised plan, revision 2 — implementation pending
**Goal:** Land the frozen contracts, exact checkpoint migration, selection-aware validation, governed expansion validation, and hard budget/config seams required by discovery bootstrap while keeping graph admission and execution disabled.

**Spec:** `docs/superpowers/specs/2026-09-17-complex-discovery-bootstrap-design.md` revision 5.

## 1. Phase boundary

Phase 1 is additive contract groundwork. It must not add discovery graph nodes, change routing, dispatch a bootstrap search, or enable the feature flag.

Required dormant behavior:

- `V2_DISCOVERY_BOOTSTRAP_ENABLED` defaults to `false`;
- existing calls with `target_selection=None` and `discovery_checkpoint=None` keep the current strict target/reference role checks;
- current fast, generic-replan, retrieve-unscoped, v1, rollout, and shadow behavior remains unchanged;
- new origin types are rejected unless the caller supplies the matching discovery context;
- new selection-driven summarize/compare branches are callable in tests but unreachable from production routing until later phases.

## 2. Execution preconditions

Use a clean dedicated worktree. Before Task 1:

```bash
git status --short
git diff --check
```

Do not start if any file named by this plan already has unrelated modifications. In particular, never use whole-file `git add` on a pre-dirty file. This plan does not require task-by-task commits; commit only after the complete Phase-1 diff has been reviewed and only when explicitly requested.

Run all Python tests from `backend/`:

```bash
cd backend
python -m pytest ...
```

Before editing any symbol, run the repository-required GitNexus upstream impact analysis for that symbol and stop for user confirmation on HIGH or CRITICAL risk. Before any eventual commit, run `detect_changes(scope="compare", base_ref="main")`.

## 3. Frozen decisions

### 3.1 Dependency direction

Create a cycle-free helper module:

```text
contracts/validation_support.py
```

It owns and exports:

```python
class ContractValidationError(ValueError): ...
class IncompatibleCheckpointError(ContractValidationError): ...

def _fail(message: str) -> NoReturn: ...
def _require_non_blank(value: object, field: str) -> None: ...
def _require_unique(values: Iterable[object], field: str) -> None: ...
def _require_contract_version(declared: object, boundary: str) -> None: ...
```

The existing `backend/app/services/agents/v2/discovery.py` generic candidate/binding-handoff module remains byte-for-byte unchanged; bootstrap code uses the separate `discovery_bootstrap/` package to avoid module/package shadowing. `contracts/validation.py` imports and re-exports the two exception classes for compatibility. `discovery_bootstrap/contracts.py` is data-only and never imports `contracts/validation.py`, `contracts/state.py`, or `contracts/planning.py`. Discovery validators import only discovery models plus `validation_support`. Because that dependency is one-way, `contracts/state.py` imports the concrete discovery models directly; it must not type new slots as `Any`.

### 3.2 Durable target ownership

`ResearchTargetSelection` owns its frozen `target_slots`. A final selection is therefore valid when `discovery=None`, which is required when explicit bindings satisfy every slot and search is skipped.

```python
class TargetSlot(ContractModel):
    slot_id: str
    intended_role: Literal["target", "reference"]
    subject_hint: str
    requested_locator: ContentLocator
    required: bool
    min_selections: int
    max_selections: int
    explicit_binding_ids: tuple[str, ...]
    source: Literal["semantic_reference", "explicit_resource", "research_need"]


class SelectedBindingRef(ContractModel):
    target_id: str
    binding_id: str
    selected_aggregate_id: UUID | None
    authority: Literal[
        "explicit_binding",
        "exact_match_policy",
        "confidence_policy",
        "user_choice",
    ]


class SlotBindingSelection(ContractModel):
    slot_id: str
    selections: tuple[SelectedBindingRef, ...]


class ResearchTargetSelection(ContractModel):
    work_type: Literal["summarize", "compare"]
    target_slots: tuple[TargetSlot, ...]
    slot_bindings: tuple[SlotBindingSelection, ...]
    context_binding_ids: tuple[str, ...]
```

`SelectedBindingRef.target_id` is allocated server-side during final selection and is globally unique within the selection. Selection-aware skills reuse it verbatim as `TargetUnit.target_id`; the validator maps the target unit back to its selected binding and owning slot. `TargetUnit` itself is unchanged, preserving the exact flag-off wire/checkpoint shape.

### 3.3 Discovery-scoped aggregate identity

`DiscoveryCheckpoint` owns a server-derived `discovery_id`. Aggregate identity is:

```python
def aggregate_id_for(
    discovery_id: UUID,
    slot_id: str,
    document_id: UUID,
    document_revision: str,
) -> UUID:
    return uuid5(discovery_id, f"{slot_id}:{document_id}:{document_revision}")
```

There is no process-global `PUBLIC_NAMESPACE`. The same candidate set is stable across resume in one discovery lineage and separated across runs.

### 3.4 Context-required discovery origins

The canonical validator signature is:

```python
def validate_task_plan(
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    *,
    target_selection: ResearchTargetSelection | None = None,
    discovery_checkpoint: DiscoveryCheckpoint | None = None,
) -> None: ...
```

Rules:

- a `DiscoveryProbeTaskOrigin` is valid only on `document.search` with `DocumentSearchInput` and only when it exactly matches an accepted probe in `discovery_checkpoint`;
- a `DiscoveryExpansionTaskOrigin` requires both discovery and target-selection context;
- without discovery context, both discovery origin types fail closed;
- an origin label never exempts a factual task from budget by itself.

### 3.5 Checkpoint migration

The root discriminator remains:

```python
CHECKPOINT_SCHEMA_REVISION: Final[Literal[2]] = 2
CheckpointSchemaRevision = Literal[1, 2]
```

Legacy classification is exact:

- missing discriminator + all four revision-2-only slots absent → revision 1;
- explicit `checkpoint_schema_revision=1` + all four revision-2-only slots absent → revision 1;
- missing/`1` discriminator + any revision-2-only slot present → corrupt partial shape, reject;
- revision 2 missing any required slot → reject;
- unknown revision → reject.

## 4. File map

| File | Responsibility |
|---|---|
| `contracts/validation_support.py` | Cycle-free validation errors/helpers. |
| `contracts/clarification.py` | Semantic kind plus document-selection request, manifest, and resolution models. |
| `contracts/planning.py` | Discovery origins, budget views, `ResearchPlanningInput.target_selection`/`discovery_checkpoint`; `TargetUnit` stays unchanged. |
| `contracts/capability.py` | Checkpoint-safe identity-match metadata on `DocumentSearchOutput`. |
| `contracts/state.py` | Revision discriminator and four typed nullable root slots. |
| `contracts/validation.py` | Root migration/gates, context-aware plan/origin validation, aggregate-state threading. |
| `discovery_bootstrap/contracts.py` | Data-only discovery, slot, selection, candidate, and checkpoint models. |
| `discovery_bootstrap/validation.py` | Slot/selection/checkpoint validators, aggregation, deterministic ordering/margin. |
| `discovery_bootstrap/plan_expansion.py` | Pure governed append constructor and expansion validator. |
| `supervisor_v2.py` | Slot registration, migration call, fresh-state initialization and hygiene. |
| `complex_research_graph.py` | Dormant child-state/planning-input plumbing only. |
| `planning/planner.py` | Selection-aware adaptive plan validation. |
| `skills/summarize/policy.py` | Dormant selection-driven target construction. |
| `skills/compare/policy.py` | Dormant selection-driven side construction, including same-document locators. |
| `execution/scheduler.py`, `replanning.py` | Explicit optional total-task entry guard; unchanged when omitted. |
| `core/config.py`, `.env.example` | Declared hard-bounded settings, flag off. |
| `CLAUDE.md` | Canonical checkpoint/selection ownership and config documentation. |

## 5. Task order

### Task 1 — Cycle-free validation support and clarification contracts

**Files**

- Create `backend/app/services/agents/v2/contracts/validation_support.py`.
- Modify `backend/app/services/agents/v2/contracts/validation.py`.
- Modify `backend/app/services/agents/v2/contracts/clarification.py`.
- Add `backend/tests/agents/v2/contracts/test_document_selection_clarification.py`.

**Changes**

1. Move the two exception classes and shared primitive helpers to `validation_support.py`; import/re-export them from `validation.py` so existing imports remain valid.
2. Add to the existing semantic request/resolution with defaults for legacy payloads:

```python
class ClarificationRequest(ContractModel):
    kind: Literal["semantic"] = "semantic"
    # existing fields unchanged


class ClarificationResolution(ContractModel):
    kind: Literal["semantic"] = "semantic"
    # existing fields unchanged
```

3. Add concrete document-selection models and `MAX_PUBLIC_CHOICES = 3`:

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

4. Validators enforce non-blank IDs/question/title/token, global token uniqueness across all public slots, unique slot IDs, bounded choices, aware expiry, valid cardinality, and declined XOR selections. The raw single-use tokens are checkpointed only in the public request slot for stable replay; the internal manifest stores only HMAC digests. Tests must not claim that raw tokens are absent from the entire checkpoint.
5. Add regression coverage proving existing imports of both exception classes from `contracts.validation` remain valid.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts/test_document_selection_clarification.py \
  tests/agents/v2/contracts/test_locators_routing_clarification.py -q
```

### Task 2 — Data-only discovery and durable selection contracts

**Files**

- Create `backend/app/services/agents/v2/discovery_bootstrap/__init__.py`.
- Create `backend/app/services/agents/v2/discovery_bootstrap/contracts.py`.
- Create `backend/app/services/agents/v2/discovery_bootstrap/validation.py` with slot/selection validation; Task 3 extends it.
- Modify `backend/app/services/agents/v2/contracts/planning.py` to add the two discovery origin data models to `TaskOrigin`; validation remains Task 4.
- Modify `backend/app/services/agents/v2/contracts/capability.py` to add checkpoint-safe identity-match output metadata.
- Modify `backend/tests/agents/v2/contracts/factories.py`.
- Add `backend/tests/agents/v2/contracts/test_discovery_selection.py`.
- Add `backend/tests/agents/v2/contracts/test_validation_imports.py`.

**Models**

Implement the frozen decisions in §3.2 plus:

```python
class DiscoveryNeed(ContractModel):
    required: bool
    reason: Literal["unresolved_document_slot"] | None


class SearchProbe(ContractModel):
    probe_id: str
    slot_id: str
    query: str
    round: Literal[1, 2]
    origin: Literal["semantic", "planner_fallback"]


MatchKind = Literal[
    "exact_document_number",
    "exact_normalized_title",
    "lexical_title",
    "semantic",
]


class DocumentIdentityMatch(ContractModel):
    candidate_id: UUID
    document_id: UUID
    document_revision: str
    rank: int
    confidence: float | None
    match_kind: MatchKind
    calibration_version: str | None


class ProbeCandidateMatch(ContractModel):
    probe_id: str
    slot_id: str
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


class DiscoverySelection(ContractModel):
    slot_id: str
    selected_aggregate_ids: tuple[UUID, ...]
    authority: Literal["exact_match_policy", "confidence_policy", "user_choice"]


class DiscoveryCheckpoint(ContractModel):
    discovery_id: UUID
    target_slots: tuple[TargetSlot, ...]
    accepted_probes: tuple[SearchProbe, ...]
    candidate_matches: tuple[ProbeCandidateMatch, ...]
    slot_aggregates: tuple[SlotCandidateAggregate, ...]
    selections: tuple[DiscoverySelection, ...]
    rounds_consumed: int
    probes_consumed: int
    status: Literal["searching", "ranking", "clarification", "selected", "unavailable"]
    clarification_manifest: DocumentSelectionManifest | None
```

`MatchKind` and `DocumentIdentityMatch` live in `contracts/capability.py`; discovery contracts import them. Add to the existing output without changing generic discovery behavior:

```python
class DocumentSearchOutput(ContractModel):
    kind: Literal["document.search"]
    candidates: tuple[DocumentDiscoveryCandidate, ...]
    identity_matches: tuple[DocumentIdentityMatch, ...] = ()
```

Generic search keeps `identity_matches=()`. A later bootstrap adapter must populate one metadata record per candidate; no candidate text enters this contract.

Add the data-only origin models to `contracts/planning.py` now so Task 3 can validate accepted-probe/task ownership:

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

Both join the discriminated `TaskOrigin` union in this task. Their authority checks are deliberately deferred to Task 4, but no checkpoint validator may treat an unvalidated origin label as authority.

`discovery_bootstrap/contracts.py` may import `ContractModel`, locator types, clarification models, and the data-only identity-match types from `contracts/capability.py`. It must not import `contracts.validation` or `discovery_bootstrap.validation`.

Implement in `discovery_bootstrap/validation.py`:

```python
def validate_target_slots(slots: tuple[TargetSlot, ...]) -> None: ...


def validate_research_target_selection(
    selection: ResearchTargetSelection,
    bindings: DocumentBindingSet,
    discovery_checkpoint: DiscoveryCheckpoint | None = None,
    query_analysis: QueryAnalysis | None = None,
) -> None: ...
```

This first version validates `DiscoveryNeed.required == (reason is not None)`, slot/cardinality/binding/context/distinct-document rules, `work_type` shape, same-document compare constraints, exact `selection.target_slots == discovery_checkpoint.target_slots` when discovery exists, and explicit-versus-discovery provenance. Task 3 adds full aggregate/checkpoint reconstruction without changing the signature.

**Selection tests**

- selection valid with `discovery=None` and explicit binding authority;
- unknown slot/binding rejected;
- duplicate context binding rejected;
- selected/context overlap rejected;
- discovered selection without aggregate ID rejected;
- explicit selection with aggregate ID rejected;
- two revisions of one document cannot satisfy a multi-select slot;
- one binding may occupy two compare slots only with distinct locators;
- duplicate/blank selected `target_id` is rejected, and target-unit mapping is covered in Task 4;
- `TaskSpec` strictly parses both new discriminated origin members, without yet granting them execution authority;
- `DocumentSearchOutput.identity_matches` round-trips strictly, while the existing generic constructor defaults to an empty tuple;
- clean subprocess imports succeed in both orders: `contracts.validation → discovery_bootstrap.contracts` and `discovery_bootstrap.contracts → contracts.validation`, followed by `supervisor_v2`.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts/test_discovery_selection.py \
  tests/agents/v2/contracts/test_validation_imports.py -q
```

### Task 3 — Discovery validation, aggregation, and margin

**Files**

- Modify `backend/app/services/agents/v2/discovery_bootstrap/validation.py`.
- Add `backend/tests/agents/v2/contracts/test_discovery_aggregation.py`.
- Add `backend/tests/agents/v2/contracts/test_discovery_checkpoint.py`.

**Interfaces**

```python
def aggregate_id_for(
    discovery_id: UUID,
    slot_id: str,
    document_id: UUID,
    document_revision: str,
) -> UUID: ...


def aggregate_matches(
    discovery_id: UUID,
    matches: tuple[ProbeCandidateMatch, ...],
    accepted_probes: tuple[SearchProbe, ...],
    slots: tuple[TargetSlot, ...],
) -> tuple[SlotCandidateAggregate, ...]: ...


def selection_margin(
    ordered: tuple[SlotCandidateAggregate, ...],
    selected_ids: tuple[UUID, ...],
    slot: TargetSlot,
    authority: str,
) -> float: ...


def validate_discovery_checkpoint(
    checkpoint: DiscoveryCheckpoint,
    *,
    plan: TaskPlan | None,
    task_results: tuple[AgentResult, ...] = (),
    clarification: DocumentSelectionClarification | None = None,
    max_probes: int = 5,
    max_rounds: int = 2,
    top_k: int = 5,
) -> None: ...


def validate_research_target_selection(
    selection: ResearchTargetSelection,
    bindings: DocumentBindingSet,
    discovery_checkpoint: DiscoveryCheckpoint | None = None,
    query_analysis: QueryAnalysis | None = None,
) -> None: ...
```

**Required validation**

- non-blank/unique slots, probes, tasks, candidates, aggregates, and selections;
- normalized-equivalent probe rejection and exact counters;
- positive rank bounded by `top_k`;
- confidence finite and in `[0,1]`;
- exact matches have `confidence=None` and `calibration_version=None`;
- lexical/semantic matches require both confidence and calibration version;
- every accepted probe maps to exactly one probe-origin task in the supplied plan, and every `ProbeCandidateMatch` is exactly reconstructed from that task's checkpointed `AgentResult.data.identity_matches` member and corresponding candidate identity;
- aggregates are exactly reconstructed from candidate matches, including complete source lineage and order;
- non-exact order is match-kind, confidence descending, best rank, accepted-probe order, document identity;
- exact order omits confidence; multiple distinct exact candidates do not auto-select;
- selected IDs are unique and belong to their slot;
- policy selection is an exact ranking prefix; user choice may select another offered aggregate;
- multi-select margin uses the actual last selected candidate, including `max_selections` when saturated;
- status-specific invariants hold; when clarification is pending, manifest and public request have the same ID/expiry/slots, digests are non-blank and unique, and entries exactly match checkpoint aggregates. HMAC recomputation is deliberately left to the runtime resume boundary in Phase 5.

**Required tests**

Include registry-loss rebuild from checkpointed `identity_matches`, cross-run aggregate-ID separation, NaN/infinite confidence rejection, forged probe-match/aggregate reconstruction rejection, unknown probe/task rejection, duplicate selection slot rejection, exact ambiguity, single candidate margin, selected-non-top policy rejection, and five selected from six candidate boundary.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts/test_discovery_aggregation.py \
  tests/agents/v2/contracts/test_discovery_checkpoint.py \
  tests/agents/v2/contracts/test_discovery_selection.py -q
```

### Task 4 — Context-aware discovery-origin plan validation

**Files**

- Modify `backend/app/services/agents/v2/contracts/validation.py`.
- Modify `backend/tests/agents/v2/contracts/factories.py`.
- Add `backend/tests/agents/v2/contracts/test_discovery_task_origins.py`.
- Add `backend/tests/agents/v2/contracts/test_selection_aware_validation.py`.

**Origin validation**

Use the two origin models already frozen into `TaskOrigin` by Task 2. Change `_validate_plan_structure`/`_validate_task_origin` to receive the optional discovery/selection context from the signature in §3.4; do not redefine or widen the models here.

**Probe-origin checks**

- capability and input kind are both `document.search`;
- input has no target IDs or materialized People scalar;
- exactly one accepted probe matches `probe_id`;
- origin slot/round and normalized input query equal the probe;
- one plan task owns one probe and one probe is not owned by multiple tasks;
- discovery context is mandatory.

**Expansion-origin checks**

- discovery and target selection are mandatory;
- source task IDs are unique bootstrap search tasks;
- selected aggregates are in checkpoint selections;
- target slot IDs are unique known selected slots;
- each input target ID resolves through `SelectedBindingRef.target_id`, and `target_slot_ids` equals the owning slots of those selected refs.

**Strict compatibility**

`validate_fast_plan` and all non-bootstrap callers pass no discovery context; without `target_selection`, the existing `TargetUnit` shape and historical role validation remain unchanged. `validate_replan` gains optional `target_selection`/`discovery_checkpoint` parameters with strict defaults: non-bootstrap outcomes remain unchanged, while a later replan may use the contexts only to revalidate an already bootstrap-expanded current plan. New replan tasks still require `ReplanTaskOrigin`, cannot mutate target units, and cannot masquerade as discovery tasks.

**Required regression test**

A `document.read` task carrying `DiscoveryProbeTaskOrigin` must fail and must count as factual work; the old proposed test that allowed this shape must not exist.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts/test_discovery_task_origins.py \
  tests/agents/v2/contracts/test_selection_aware_validation.py \
  tests/agents/v2/contracts/test_task_summary_checkpoint.py -q
```

### Task 5 — Exact checkpoint revision and root-slot wiring

**Files**

- Modify `backend/app/services/agents/v2/contracts/state.py`.
- Modify `backend/app/services/agents/v2/contracts/validation.py`.
- Modify `backend/app/services/agents/supervisor_v2.py`.
- Modify `backend/app/services/agents/v2/nodes/context.py` only if fresh-turn clearing is owned there rather than the supervisor wrapper.
- Add `backend/tests/agents/v2/contracts/test_checkpoint_revision.py`.

**Typed state**

Add the discriminator and four required nullable keys to `SupervisorV2State`. Import the concrete discovery models directly from the data-only module; do not use `Any` or runtime-unresolvable forward references.

**Migration**

```python
_REVISION_TWO_ONLY_KEYS = frozenset({
    "discovery_need",
    "discovery",
    "document_selection_clarification",
    "research_target_selection",
})


def migrate_checkpoint_payload(payload: Mapping[str, object]) -> dict[str, object]:
    values = dict(payload)
    revision = values.get("checkpoint_schema_revision")
    has_new_shape = any(key in values for key in _REVISION_TWO_ONLY_KEYS)

    if revision in (None, 1):
        if has_new_shape:
            raise IncompatibleCheckpointError(
                "legacy checkpoint carries a partial revision-2 shape"
            )
        values.setdefault("synthesis", None)
        values["checkpoint_schema_revision"] = CHECKPOINT_SCHEMA_REVISION
        values.update({key: None for key in _REVISION_TWO_ONLY_KEYS})
        return values
    if revision != CHECKPOINT_SCHEMA_REVISION:
        raise IncompatibleCheckpointError(
            f"unknown checkpoint_schema_revision {revision!r}"
        )
    return values
```

After migration, the existing required-key gate rejects any missing revision-2 key. Add all four models to `_SLOT_MODELS` and `_NULLABLE_SLOTS`; initialize them in `build_initial_v2_state`; clear them on a genuinely fresh turn. `validate_supervisor_state` passes the root execution plan/results and sibling `document_selection_clarification` into `validate_discovery_checkpoint`, then validates `research_target_selection` against the same discovery, bindings, and query analysis. Add `kind="semantic"` during mapping normalization only where needed for old clarification JSON.

**Required tests**

- missing discriminator and no new keys migrates;
- explicit revision 1 and no new keys migrates;
- missing discriminator plus one/all new keys rejects;
- revision 1 plus one new key rejects;
- revision 2 missing each required key rejects parametrically;
- unknown revision rejects;
- fresh state writes revision 2 and all four keys;
- new-turn hygiene clears all four;
- slot coercion returns concrete models after JSON round-trip.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts/test_checkpoint_revision.py \
  tests/agents/v2/test_supervisor_v2.py -q
```

### Task 6 — Selection-aware planner, skills, and boundary plumbing

**Files**

- Modify `backend/app/services/agents/v2/contracts/planning.py`.
- Modify `backend/app/services/agents/v2/contracts/validation.py`.
- Modify `backend/app/services/agents/v2/complex_research_graph.py`.
- Modify `backend/app/services/agents/v2/planning/planner.py`.
- Modify `backend/app/services/agents/v2/skills/summarize/policy.py`.
- Modify `backend/app/services/agents/v2/skills/compare/policy.py`.
- Add/modify focused tests under `backend/tests/agents/v2/contracts/` and `complex/`.

**Planning input**

```python
class ResearchPlanningInput(ContractModel):
    # existing fields unchanged
    target_selection: ResearchTargetSelection | None = None
    discovery_checkpoint: DiscoveryCheckpoint | None = None
```

`build_planning_input` populates it from the child checkpoint. Add `discovery` and `research_target_selection` to `ComplexResearchState`, `build_complex_research_state`, `normalize_complex_state`, and `merge_complex_result_into_supervisor`.

Every relevant validator receives:

```python
target_selection=state.get("research_target_selection")
discovery_checkpoint=state.get("discovery")
```

Do not derive target slots from `discovery`; use `target_selection.target_slots`.

**Summarize policy**

When selection exists:

- identify the primary target slot by frozen slot identity/role, not by scanning all binding roles;
- construct targets only from `SlotBindingSelection.selections`;
- create each `TargetUnit` with the selected ref's exact `target_id` and binding, plus the owning slot's locator;
- leave context bindings unused;
- validate with `target_selection` and optional discovery context.

When selection is `None`, preserve the current strict single-target implementation byte-for-byte.

**Compare policy**

When selection exists:

- require exactly two logical side slots;
- construct one target unit per side from the selected binding and slot locator;
- allow one binding on both sides only when locators are distinct;
- reject accidental same-document/same-locator collisions;
- reuse each side's selected `target_id` and validate with the same contexts.

When selection is `None`, preserve current strict role/arity behavior.

**Required tests**

- explicit complete selection with `discovery=None` plans successfully;
- quote A stays context while selected B is the sole summary target;
- selected supporting/discovered binding is accepted only through selection;
- context binding cannot authorize a target unit;
- same-document two-section compare passes;
- same-document same-locator compare fails;
- planner, skill, validate-checkpoint, and aggregate-state validators all use the same selection.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts/test_selection_aware_validation.py \
  tests/agents/v2/complex/test_adaptive_planner.py \
  tests/agents/v2/complex/test_complex_expansion.py -q
```

### Task 7 — Typed governed expansion and ownership guard

**Files**

- Create `backend/app/services/agents/v2/discovery_bootstrap/plan_expansion.py`.
- Modify `backend/tests/agents/v2/complex/test_complex_expansion.py`.
- Add `backend/tests/agents/v2/complex/test_discovery_plan_expansion.py`.

**Interfaces**

```python
def factual_tasks(
    plan: TaskPlan,
    *,
    discovery_checkpoint: DiscoveryCheckpoint | None,
) -> tuple[TaskSpec, ...]: ...


def validate_plan_expansion(
    current: TaskPlan,
    proposed: TaskPlan,
    checkpoint: DiscoveryCheckpoint,
    selection: ResearchTargetSelection,
    bindings: DocumentBindingSet,
    outcomes: tuple[AgentResult, ...],
    *,
    capability_names: frozenset[str],
    max_research_tasks: int,
    total_task_limit: int,
) -> None: ...


def expand_discovery_plan(
    current: TaskPlan,
    checkpoint: DiscoveryCheckpoint,
    selection: ResearchTargetSelection,
    bindings: DocumentBindingSet,
    outcomes: tuple[AgentResult, ...],
    appended_units: tuple[TargetUnit, ...],
    appended_tasks: tuple[TaskSpec, ...],
    *,
    capability_names: frozenset[str],
    max_research_tasks: int,
    total_task_limit: int,
) -> TaskPlan: ...
```

`Sequence[Any]`, `getattr`-based outcome inspection, and synthetic dynamic outcome objects are forbidden.

Implement every invariant in spec §13.1, especially:

- exact old prefixes;
- all attempted search tasks have exactly one typed result;
- only `checkpoint.selections` authorize expansion;
- aggregate source task/result identity is exact;
- `SelectedBindingRef.selected_aggregate_id` matches slot and exact binding document/revision;
- each new unit's target ID, binding, and locator exactly match one selected ref and its owning slot;
- expansion-origin task slots, aggregates, and source search tasks correspond to the target units it reads;
- discovery, research, and total budgets hold;
- final `validate_task_plan` receives both contexts.

`expand_discovery_plan` is pure and does not lease. Phase 4's sole caller, `research_expand_node`, must call `_lease_pinned_state` and commit before returning the plan into checkpoint state.

Extend the AST guard so only `append_replan_tasks` and `expand_discovery_plan` append task/target tuples. Keep explicit function allowlists; no wildcard or directory allowlist.

**Negative tests**

Reject unselected aggregate, missing/duplicate/failed source outcome where success is required, non-bootstrap source task, foreign slot, aggregate↔binding document mismatch, revision mismatch, wrong locator, mismatched origin slots, forged capability, and over-budget plan.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/complex/test_discovery_plan_expansion.py \
  tests/agents/v2/complex/test_complex_expansion.py -q
```

### Task 8 — Hard budgets, scheduler/replan seam, and configuration

**Files**

- Modify `backend/app/services/agents/v2/contracts/planning.py`.
- Modify `backend/app/services/agents/v2/complex_research_graph.py`.
- Modify `backend/app/services/agents/v2/execution/scheduler.py`.
- Modify `backend/app/services/agents/v2/replanning.py` as needed for total remaining capacity.
- Modify `backend/app/core/config.py`.
- Modify `.env.example`.
- Add `backend/tests/agents/v2/contracts/test_discovery_budgets.py`.
- Add focused scheduler/replan tests.

**Budget views**

```python
class DiscoveryBudgetView(ContractModel):
    probes_remaining: int
    rounds_remaining: int
    top_k: int


def build_research_budget_view(
    plan: TaskPlan,
    *,
    discovery_checkpoint: DiscoveryCheckpoint | None,
    max_tasks: int,
    max_replans: int,
    max_parallel_branches: int,
    total_task_limit: int,
) -> ResearchBudgetView: ...
```

Count a task as a free probe only after the Task-4 exact probe-origin validation succeeds. All malformed/spoofed origins fail before budget construction.

Add an optional explicit plan-entry guard:

```python
async def execute_ready_tasks(..., total_task_limit: int | None = None) -> DispatchReport:
    if total_task_limit is not None and len(plan.tasks) > total_task_limit:
        raise SchedulerError("plan exceeds total task limit")
    ...
```

`TaskScheduler.execute` threads the optional parameter. Existing callers omit it and preserve behavior; later bootstrap callers must pass it. Replan remaining capacity is bounded by both factual and total remaining capacity.

**Settings**

```python
V2_DISCOVERY_BOOTSTRAP_ENABLED: bool = Field(default=False)
V2_DISCOVERY_MAX_PROBES: int = Field(default=5, ge=1, le=5)
V2_DISCOVERY_MAX_ROUNDS: int = Field(default=2, ge=1, le=2)
V2_DISCOVERY_TOP_K: int = Field(default=5, ge=1, le=5)
V2_DISCOVERY_DEADLINE_SECONDS: int = Field(default=15, ge=1, le=15)
V2_DISCOVERY_SUMMARY_TARGETS: int = Field(default=3, ge=3, le=5)
V2_DISCOVERY_MAX_SUMMARY_TARGETS: int = Field(default=5, ge=3, le=5)
V2_DISCOVERY_CONFIDENCE_THRESHOLD: float = Field(default=0.82, ge=0.0, le=1.0)
V2_DISCOVERY_MARGIN_THRESHOLD: float = Field(default=0.12, ge=0.0, le=1.0)
V2_DISCOVERY_CALIBRATION_ARTIFACT: str = Field(
    default="/app/config/discovery-calibration.json"
)
V2_TOTAL_MAX_TASKS: int = Field(default=13, gt=0)
```

Validate existing `V2_MAX_TASKS > 0`, `V2_MAX_PARALLEL_BRANCHES > 0`, `V2_MAX_REPLANS >= 0`, and `V2_MAX_DISCOVERED_DOCUMENTS >= 0`; validate summary min ≤ max and total cap ≥ configured probe max + `V2_MAX_TASKS`. Add matching `.env.example` entries. Artifact readability/model-hash enforcement remains Phase 2 because the feature is disabled and the providers/calibrator do not exist yet.

**Required tests**

- every hard ceiling rejects a larger value;
- free valid probe does not consume research capacity;
- read task spoofing a probe origin is rejected, never made free;
- scheduler rejects over-total plan when explicit limit is passed;
- scheduler behavior is unchanged when limit is omitted;
- replan cannot exceed total remaining capacity;
- defaults are exactly `5/2/5/15/3/5/0.82/0.12/13` and flag false.

**Verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts/test_discovery_budgets.py \
  tests/agents/v2/execution -q
```

### Task 9 — Canonical documentation and final regression gate

**Files**

- Modify `CLAUDE.md`.
- Confirm `.env.example` from Task 8.

Document only landed Phase-1 facts:

- checkpoint schema revision 2 and exact revision-1 migration;
- four nullable discovery root slots;
- `ResearchTargetSelection` owns target slots independently of discovery;
- discovery origins require discovery validation context;
- new settings exist but `V2_DISCOVERY_BOOTSTRAP_ENABLED=false` and no route/node behavior is enabled.

Do not claim identity search, automatic selection, clarification resume, or graph execution until their later phases land.

**Final verification**

```bash
cd backend && python -m pytest \
  tests/agents/v2/contracts \
  tests/agents/v2/complex \
  tests/agents/v2/execution \
  tests/agents/v2/test_supervisor_v2.py -q
cd .. && git diff --check
git status --short
```

Review the complete diff and confirm:

- no route result, graph node, capability, public transport, or runner behavior was added;
- flag-off validator outcomes remain unchanged for existing plans;
- no discovery model imports `contracts.validation`;
- no discovery origin passes without discovery context;
- explicit selection validates with `discovery=None`;
- no raw query/title/document identifiers were added to logs/traces;
- only intended files changed.

## 6. Phase-1 exit criteria

- [ ] All Task 1–9 focused tests and the final regression gate pass.
- [ ] Import-order tests prove no validation/discovery cycle.
- [ ] Revision-1 exact shapes migrate; partial/discriminator-less revision-2 shapes fail closed.
- [ ] `ResearchTargetSelection` validates independently when discovery was skipped.
- [ ] A discovery-probe origin on `document.read` fails and cannot bypass budgets.
- [ ] Checkpointed identity-match output rebuilds probe matches without a live registry; aggregates are discovery-scoped, reconstructed, and ordered deterministically.
- [ ] Expansion rejects unselected or identity-mismatched aggregate/binding lineage.
- [ ] Hard probe/round/top-k/deadline ceilings and total plan-entry limit are tested.
- [ ] `CLAUDE.md` and `.env.example` describe only landed dormant contracts.
- [ ] `V2_DISCOVERY_BOOTSTRAP_ENABLED` remains `false`; production graph behavior is unchanged.

## 7. Later plans

| Plan | Produces |
|---|---|
| Phase 2 — identity search and calibration | Exact-number/title/lexical/semantic pipeline, bounded rerank, calibration artifact loader and model-hash gates. |
| Phase 3 — named summary and compare bootstrap | Slot derivation/reconciliation, conditional route admission, probe execution, deterministic selection, same-document comparison. |
| Phase 4 — binding settle and expansion wiring | `BindingSelectionRequest`, ACL/pin/lease/provenance, `selection_settle_node`, `research_expand_node`. |
| Phase 5 — clarification suspend/resume | Per-slot public frame, digest manifest, expiry/current-ACL/current-revision validation, same-checkpoint resume. |
| Phase 6 — thematic summary | Three-to-five-target map/reduce after prior gates pass. |

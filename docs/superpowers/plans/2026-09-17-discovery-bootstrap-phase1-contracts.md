# Discovery Bootstrap Phase 1 — Contract Groundwork Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the frozen v2 contracts, checkpoint schema revision, validators, and config that the discovery bootstrap needs, with graph behavior still disabled.

**Architecture:** Phase 1 is additive contract work inside `app/services/agents/v2/`. Four new root checkpoint slots (`discovery_need`, `discovery`, `document_selection_clarification`, `research_target_selection`) join `SupervisorV2State` behind a `checkpoint_schema_revision` discriminator, legacy revision-1 checkpoints migrate forward, new discovery contracts live in a new `v2/discovery/` package, and plan validation becomes selection-aware without weakening existing strict callers.

**Tech Stack:** Python 3.12, Pydantic v2 (`ContractModel`: strict, frozen, `extra="forbid"`), LangGraph `TypedDict` state, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-complex-discovery-bootstrap-design.md`

## Global Constraints

- `ContractVersion` stays exactly `"2.0"`; do not add `"2.1"`.
- Every persisted business contract derives from `ContractModel` (`backend/app/services/agents/v2/contracts/base.py`): `extra="forbid"`, `frozen=True`, `strict=True`.
- Never bind a checkpoint slot model to a `typing.Annotated` union alias — `_SLOT_MODELS` is `dict[str, type]` and `_coerce_slot` calls `isinstance`, `model_validate_json`, and `model.__name__`.
- The flag-off path must stay byte-compatible: no new route result, no changed validator outcome when `research_target_selection is None`.
- `validate_task_plan(plan, bindings)` keeps its current strict role check when called without a selection.
- No behavior is enabled in Phase 1: `V2_DISCOVERY_BOOTSTRAP_ENABLED` stays `false` and no graph node is added.
- Commit only the files each task names; the worktree contains unrelated dirty changes that must not be staged.
- Run tests from `backend/`: `cd backend && python -m pytest ...`.

## Reused Code Facts (verified in this worktree)

These already exist; the plan builds on them instead of inventing replacements:

| Name | Location |
|---|---|
| `ContractModel` | `app/services/agents/v2/contracts/base.py` |
| `_fail`, `_require_non_blank`, `_require_unique`, `_require_contract_version`, `_optional_model`, `_as_model` | `app/services/agents/v2/contracts/validation.py:121-232,1281` |
| `validate_supervisor_state`, `_validate_execution_state`, `validate_task_plan` | `app/services/agents/v2/contracts/validation.py` |
| `_SLOT_MODELS`, `_NULLABLE_SLOTS`, `_coerce_slot`, `normalize_checkpoint_state`, `build_initial_v2_state` | `app/services/agents/supervisor_v2.py:225-330,602` |
| `_lease_pinned_state` | `app/services/agents/v2/complex_research_graph.py:647` |
| `validate_checkpoint_node` (with `validate_task_plan(initial.plan, bindings)`) | `app/services/agents/v2/complex_research_graph.py:~1216` |
| `Settings(BaseSettings)` with an existing `model_validator` | `app/core/config.py:55,694` |
| Test factories `request_context`, `semantic_context`, `scoped_document`, `binding_set`, `target_unit`, `document_reference` | `backend/tests/agents/v2/contracts/factories.py` |
| `DocumentReadInput(kind, target_ids)`, `DocumentLocator(kind="document")`, `CoverageCriterion(kind="coverage", ...)`, `TargetUnit(target_id, binding_id, requested_locator, completion_criteria)` | `contracts/capability.py:97`, `contracts/locators.py:17`, `contracts/planning.py:27,49` |

## File Structure

| File | Responsibility |
|---|---|
| `contracts/clarification.py` | Document-selection request/slot/choice/manifest/resolution contracts. |
| `contracts/state.py` | Four new root slots + `checkpoint_schema_revision`. |
| `contracts/planning.py` | Two new `TaskOrigin` members, `DiscoveryBudgetView`, factual-task budget view. |
| `contracts/validation.py` | Required keys, migration, new validators, origin dispatch, selection threading. |
| `v2/discovery/__init__.py`, `v2/discovery/contracts.py` | Admission, slot, probe, aggregation, selection, checkpoint contracts + policies. |
| `v2/discovery/plan_expansion.py` | `validate_plan_expansion` and `expand_discovery_plan`. |
| `supervisor_v2.py` | Slot registration, migration call, fresh-state initialization. |
| `complex_research_graph.py`, `planning/planner.py` | Selection plumbing into planning input and plan validation. |
| `core/config.py`, `.env.example` | New declared, validated settings. |
| `backend/tests/agents/v2/contracts/*`, `backend/tests/agents/v2/complex/*` | Tests. |

Task order is dependency order: clarification contracts → discovery contracts → origins → checkpoint wiring → selection validation → expansion → budgets. Do not reorder; each task's imports must already exist.

---

### Task 1: Document-selection clarification contracts

**Files:**
- Modify: `backend/app/services/agents/v2/contracts/clarification.py`
- Modify: `backend/app/services/agents/v2/contracts/validation.py`
- Test: `backend/tests/agents/v2/contracts/test_document_selection_clarification.py`

**Interfaces:**
- Consumes: `ContractModel`, `ContractVersion`, `_fail`, `_require_non_blank`, `_require_unique`, `_require_contract_version`.
- Produces:
  - `MAX_PUBLIC_CHOICES = 3`
  - `DocumentSelectionChoice(choice_token, title, document_number)`
  - `DocumentSelectionSlot(slot_id, slot_label, min_selections, max_selections, choices)`
  - `DocumentSelectionClarification(kind, contract_version, clarification_id, question, slots, expires_at)`
  - `DocumentSelectionManifestEntry(choice_token_digest, slot_id, aggregate_id, document_id, document_revision)`
  - `DocumentSelectionManifest(clarification_id, entries, expires_at, status)`
  - `DocumentSlotResolution(slot_id, choice_tokens)`
  - `DocumentSelectionResolution(kind, contract_version, clarification_id, selections, declined)`
  - `validate_document_selection_request(request) -> None`
  - `validate_document_selection_resolution(resolution) -> None`
  - `validate_clarification_slots_exclusive(state: Mapping[str, object]) -> None`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/agents/v2/contracts/test_document_selection_clarification.py
"""Concrete document-selection request/resolution and slot exclusivity."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.agents.v2.contracts.clarification import (
    DocumentSelectionChoice,
    DocumentSelectionClarification,
    DocumentSelectionResolution,
    DocumentSelectionSlot,
    DocumentSlotResolution,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_clarification_slots_exclusive,
    validate_document_selection_request,
    validate_document_selection_resolution,
)


def _slot(choices: tuple[DocumentSelectionChoice, ...] | None = None) -> DocumentSelectionSlot:
    if choices is None:
        choices = (
            DocumentSelectionChoice(
                choice_token="tok-1", title="Nghị định 13", document_number="13/2023"
            ),
        )
    return DocumentSelectionSlot(
        slot_id="primary",
        slot_label="Tài liệu chính",
        min_selections=1,
        max_selections=1,
        choices=choices,
    )


def _request(slot: DocumentSelectionSlot | None = None) -> DocumentSelectionClarification:
    return DocumentSelectionClarification(
        kind="document_selection",
        contract_version="2.0",
        clarification_id="clar-1",
        question="Bạn muốn tóm tắt tài liệu nào?",
        slots=(slot or _slot(),),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )


def test_valid_request_passes() -> None:
    validate_document_selection_request(_request())


def test_more_than_three_choices_is_rejected() -> None:
    choices = tuple(
        DocumentSelectionChoice(choice_token=f"t{i}", title=f"T{i}", document_number=None)
        for i in range(4)
    )

    with pytest.raises(ContractValidationError):
        validate_document_selection_request(_request(_slot(choices)))


def test_duplicate_choice_token_is_rejected() -> None:
    choices = (
        DocumentSelectionChoice(choice_token="dup", title="A", document_number=None),
        DocumentSelectionChoice(choice_token="dup", title="B", document_number=None),
    )

    with pytest.raises(ContractValidationError):
        validate_document_selection_request(_request(_slot(choices)))


def test_blank_question_is_rejected() -> None:
    request = _request().model_copy(update={"question": "  "})

    with pytest.raises(ContractValidationError):
        validate_document_selection_request(request)


def test_valid_resolution_passes() -> None:
    validate_document_selection_resolution(
        DocumentSelectionResolution(
            kind="document_selection",
            contract_version="2.0",
            clarification_id="clar-1",
            selections=(DocumentSlotResolution(slot_id="primary", choice_tokens=("tok-1",)),),
            declined=False,
        )
    )


def test_declined_resolution_must_not_select() -> None:
    with pytest.raises(ContractValidationError):
        validate_document_selection_resolution(
            DocumentSelectionResolution(
                kind="document_selection",
                contract_version="2.0",
                clarification_id="clar-1",
                selections=(DocumentSlotResolution(slot_id="primary", choice_tokens=("tok-1",)),),
                declined=True,
            )
        )


def test_semantic_and_document_selection_slots_are_mutually_exclusive() -> None:
    with pytest.raises(ContractValidationError):
        validate_clarification_slots_exclusive(
            {"clarification": object(), "document_selection_clarification": object()}
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_document_selection_clarification.py -v`
Expected: FAIL — `ImportError: cannot import name 'DocumentSelectionChoice'`.

- [ ] **Step 3: Add the contracts**

Append to `backend/app/services/agents/v2/contracts/clarification.py` (`datetime`, `Literal`, `UUID`, `ContractModel`, `ContractVersion` are already imported):

```python
MAX_PUBLIC_CHOICES = 3
"""Spec §15: at most this many safe choices are ever shown per slot."""


class DocumentSelectionChoice(ContractModel):
    """Spec §15: one opaque, user-visible document choice."""

    choice_token: str
    title: str
    document_number: str | None


class DocumentSelectionSlot(ContractModel):
    """Spec §15: one unresolved slot's choices and cardinality."""

    slot_id: str
    slot_label: str
    min_selections: int
    max_selections: int
    choices: tuple[DocumentSelectionChoice, ...]


class DocumentSelectionClarification(ContractModel):
    """Spec §15: the persisted document-selection request (sibling slot)."""

    kind: Literal["document_selection"]
    contract_version: ContractVersion
    clarification_id: str
    question: str
    slots: tuple[DocumentSelectionSlot, ...]
    expires_at: datetime


class DocumentSelectionManifestEntry(ContractModel):
    """Spec §15: internal token digest -> aggregate identity mapping."""

    choice_token_digest: str
    slot_id: str
    aggregate_id: UUID
    document_id: UUID
    document_revision: str


class DocumentSelectionManifest(ContractModel):
    """Spec §15: checkpointed, digest-only manifest."""

    clarification_id: str
    entries: tuple[DocumentSelectionManifestEntry, ...]
    expires_at: datetime
    status: Literal["pending", "consumed"]


class DocumentSlotResolution(ContractModel):
    """Spec §15: one slot's chosen tokens."""

    slot_id: str
    choice_tokens: tuple[str, ...]


class DocumentSelectionResolution(ContractModel):
    """Spec §15: the runner-supplied document-selection answer."""

    kind: Literal["document_selection"]
    contract_version: ContractVersion
    clarification_id: str
    selections: tuple[DocumentSlotResolution, ...]
    declined: bool
```

- [ ] **Step 4: Add the validators**

In `contracts/validation.py`, extend the clarification import and add these functions next to `validate_clarification_request`:

```python
def validate_document_selection_request(request: DocumentSelectionClarification) -> None:
    """Spec §15: bounded public choices, unique slots/tokens, sane cardinality."""

    _require_contract_version(request.contract_version, "DocumentSelectionClarification")
    _require_non_blank(
        request.clarification_id, "DocumentSelectionClarification.clarification_id"
    )
    _require_non_blank(request.question, "DocumentSelectionClarification.question")
    if not request.slots:
        _fail("DocumentSelectionClarification must declare at least one slot")
    _require_unique((slot.slot_id for slot in request.slots), "DocumentSelectionSlot.slot_id")
    for slot in request.slots:
        _require_non_blank(slot.slot_id, "DocumentSelectionSlot.slot_id")
        _require_non_blank(slot.slot_label, "DocumentSelectionSlot.slot_label")
        if slot.min_selections < 1 or slot.max_selections < slot.min_selections:
            _fail(f"DocumentSelectionSlot {slot.slot_id} has invalid cardinality")
        if not slot.choices:
            _fail(f"DocumentSelectionSlot {slot.slot_id} offers no choices")
        if len(slot.choices) > MAX_PUBLIC_CHOICES:
            _fail(
                f"DocumentSelectionSlot {slot.slot_id} offers more than "
                f"{MAX_PUBLIC_CHOICES} public choices"
            )
        _require_unique(
            (choice.choice_token for choice in slot.choices),
            f"DocumentSelectionSlot {slot.slot_id} choice_token",
        )
        for choice in slot.choices:
            _require_non_blank(choice.choice_token, "DocumentSelectionChoice.choice_token")
            _require_non_blank(choice.title, "DocumentSelectionChoice.title")


def validate_document_selection_resolution(
    resolution: DocumentSelectionResolution,
) -> None:
    """Spec §15: declined XOR per-slot selections, unique slots, non-empty tokens."""

    _require_contract_version(resolution.contract_version, "DocumentSelectionResolution")
    _require_unique(
        (entry.slot_id for entry in resolution.selections),
        "DocumentSlotResolution.slot_id",
    )
    if resolution.declined:
        if resolution.selections:
            _fail("declined document-selection resolution must not select any slot")
        return
    if not resolution.selections:
        _fail("document-selection resolution must select at least one slot")
    for entry in resolution.selections:
        _require_non_blank(entry.slot_id, "DocumentSlotResolution.slot_id")
        if not entry.choice_tokens:
            _fail(f"DocumentSlotResolution {entry.slot_id} selects no tokens")
        _require_unique(entry.choice_tokens, f"DocumentSlotResolution {entry.slot_id}")


def validate_clarification_slots_exclusive(state: Mapping[str, object]) -> None:
    """Spec §6/§15: at most one clarification slot may be live."""

    if state.get("clarification") is not None and (
        state.get("document_selection_clarification") is not None
    ):
        _fail("clarification and document_selection_clarification cannot both be set")
```

- [ ] **Step 5: Run the tests**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_document_selection_clarification.py tests/agents/v2/contracts/test_locators_routing_clarification.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/agents/v2/contracts/clarification.py \
        backend/app/services/agents/v2/contracts/validation.py \
        backend/tests/agents/v2/contracts/test_document_selection_clarification.py
git commit -m "feat(v2): add concrete document-selection clarification contracts"
```

---

### Task 2: Discovery admission, slot, and research-selection contracts

**Files:**
- Create: `backend/app/services/agents/v2/discovery/__init__.py`
- Create: `backend/app/services/agents/v2/discovery/contracts.py`
- Test: `backend/tests/agents/v2/contracts/test_discovery_selection.py`
- Modify: `backend/tests/agents/v2/contracts/factories.py`

**Interfaces:**
- Consumes: `ContractModel`, `_fail`, `_require_non_blank`, `_require_unique`, `DocumentBindingSet`.
- Produces:
  - `DiscoveryNeed(required, reason)`
  - `TargetSlot(slot_id, intended_role, subject_hint, required, min_selections, max_selections, explicit_binding_ids, source)`
  - `SlotBindingSelection(slot_id, binding_ids)`
  - `ResearchTargetSelection(slot_bindings, context_binding_ids)`
  - `validate_target_slots(slots) -> None`
  - `validate_research_target_selection(selection, bindings, slots) -> None`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/agents/v2/contracts/test_discovery_selection.py
"""Admission shape, slot cardinality, and research target selection."""
from __future__ import annotations

import pytest

from app.services.agents.v2.contracts.validation import ContractValidationError
from app.services.agents.v2.discovery.contracts import (
    DiscoveryNeed,
    ResearchTargetSelection,
    SlotBindingSelection,
    TargetSlot,
    validate_research_target_selection,
    validate_target_slots,
)
from tests.agents.v2.contracts.factories import binding_set, scoped_document

PRIMARY = TargetSlot(
    slot_id="primary",
    intended_role="target",
    subject_hint="Nghị định 13",
    required=True,
    min_selections=1,
    max_selections=1,
    explicit_binding_ids=(),
    source="research_need",
)


def _bindings(*binding_ids: str):
    return binding_set(
        bindings=tuple(scoped_document(binding_id=binding_id) for binding_id in binding_ids)
    )


def test_discovery_need_reason_must_match_required() -> None:
    assert DiscoveryNeed(required=False, reason=None).required is False

    with pytest.raises(ValueError):
        DiscoveryNeed(required=True, reason=None)


def test_valid_selection_passes() -> None:
    selection = ResearchTargetSelection(
        slot_bindings=(SlotBindingSelection(slot_id="primary", binding_ids=("b1",)),),
        context_binding_ids=("b2",),
    )

    validate_target_slots((PRIMARY,))
    validate_research_target_selection(selection, _bindings("b1", "b2"), (PRIMARY,))


def test_over_selection_is_rejected() -> None:
    selection = ResearchTargetSelection(
        slot_bindings=(SlotBindingSelection(slot_id="primary", binding_ids=("b1", "b2")),),
        context_binding_ids=(),
    )

    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, _bindings("b1", "b2"), (PRIMARY,))


def test_required_slot_without_selection_is_rejected() -> None:
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(
            ResearchTargetSelection(slot_bindings=(), context_binding_ids=()),
            _bindings("b1"),
            (PRIMARY,),
        )


def test_unknown_binding_is_rejected() -> None:
    selection = ResearchTargetSelection(
        slot_bindings=(SlotBindingSelection(slot_id="primary", binding_ids=("missing",)),),
        context_binding_ids=(),
    )

    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, _bindings("b1"), (PRIMARY,))


def test_binding_cannot_be_both_selected_and_context() -> None:
    selection = ResearchTargetSelection(
        slot_bindings=(SlotBindingSelection(slot_id="primary", binding_ids=("b1",)),),
        context_binding_ids=("b1",),
    )

    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, _bindings("b1"), (PRIMARY,))


def test_required_slot_must_require_at_least_one_selection() -> None:
    bad = PRIMARY.model_copy(update={"min_selections": 0})

    with pytest.raises(ContractValidationError):
        validate_target_slots((bad,))
```

No factory change is needed: the test uses the existing `binding_set`/`scoped_document`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_discovery_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: app.services.agents.v2.discovery`.

- [ ] **Step 3: Create the package and contracts**

`backend/app/services/agents/v2/discovery/__init__.py`:

```python
"""Spec §8: governed discovery-bootstrap contracts and policies."""
```

`backend/app/services/agents/v2/discovery/contracts.py`:

```python
"""Spec §7-§8: discovery admission, slots, aggregation, and selection."""
from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid5

from pydantic import model_validator

from ..contracts.base import ContractModel
from ..contracts.binding import DocumentBindingSet
from ..contracts.clarification import DocumentSelectionManifest
from ..contracts.validation import (
    ContractValidationError,
    _fail,
    _require_non_blank,
    _require_unique,
)


class DiscoveryNeed(ContractModel):
    """Spec §4.1: route-time, observability-only admission summary."""

    required: bool
    reason: Literal["unresolved_document_slot"] | None

    @model_validator(mode="after")
    def _check_reason(self) -> "DiscoveryNeed":
        if self.required != (self.reason is not None):
            raise ValueError("DiscoveryNeed.reason must be set exactly when required")
        return self


class TargetSlot(ContractModel):
    """Spec §7: one logical research slot the skill must fill."""

    slot_id: str
    intended_role: Literal["target", "reference"]
    subject_hint: str
    required: bool
    min_selections: int
    max_selections: int
    explicit_binding_ids: tuple[str, ...]
    source: Literal["semantic_reference", "explicit_resource", "research_need"]


class SlotBindingSelection(ContractModel):
    """Spec §7: the binding IDs chosen for one logical slot."""

    slot_id: str
    binding_ids: tuple[str, ...]


class ResearchTargetSelection(ContractModel):
    """Spec §7: the durable turn-specific research assignment."""

    slot_bindings: tuple[SlotBindingSelection, ...]
    context_binding_ids: tuple[str, ...]


def validate_target_slots(slots: tuple[TargetSlot, ...]) -> None:
    """Spec §7.1: unique IDs, sane cardinality, required slots fillable."""

    _require_unique((slot.slot_id for slot in slots), "TargetSlot.slot_id")
    for slot in slots:
        _require_non_blank(slot.slot_id, "TargetSlot.slot_id")
        _require_non_blank(slot.subject_hint, "TargetSlot.subject_hint")
        if slot.min_selections < 0 or slot.max_selections < slot.min_selections:
            raise ContractValidationError(
                f"TargetSlot {slot.slot_id} has invalid cardinality "
                f"{slot.min_selections}..{slot.max_selections}"
            )
        if slot.required and slot.min_selections < 1:
            raise ContractValidationError(
                f"required TargetSlot {slot.slot_id} must require at least one selection"
            )
        _require_unique(
            slot.explicit_binding_ids, f"TargetSlot {slot.slot_id}.explicit_binding_ids"
        )


def validate_research_target_selection(
    selection: ResearchTargetSelection,
    bindings: DocumentBindingSet,
    slots: tuple[TargetSlot, ...],
) -> None:
    """Spec §7: every chosen binding exists and every slot obeys cardinality."""

    validate_target_slots(slots)
    known = {binding.binding_id for binding in bindings.bindings}
    slot_by_id = {slot.slot_id: slot for slot in slots}
    _require_unique(
        (entry.slot_id for entry in selection.slot_bindings), "slot_bindings.slot_id"
    )
    selected: set[str] = set()
    for entry in selection.slot_bindings:
        slot = slot_by_id.get(entry.slot_id)
        if slot is None:
            raise ContractValidationError(
                f"selection references unknown slot {entry.slot_id!r}"
            )
        _require_unique(entry.binding_ids, f"slot {entry.slot_id} binding_ids")
        for binding_id in entry.binding_ids:
            if binding_id not in known:
                raise ContractValidationError(
                    f"slot {entry.slot_id} references unknown binding {binding_id!r}"
                )
            selected.add(binding_id)
        count = len(entry.binding_ids)
        if not (slot.min_selections <= count <= slot.max_selections):
            raise ContractValidationError(
                f"slot {entry.slot_id} selected {count} bindings, "
                f"expected {slot.min_selections}..{slot.max_selections}"
            )
    for slot in slots:
        if not slot.required:
            continue
        if not any(
            entry.slot_id == slot.slot_id and entry.binding_ids
            for entry in selection.slot_bindings
        ):
            raise ContractValidationError(f"required slot {slot.slot_id} has no selection")
    for binding_id in selection.context_binding_ids:
        if binding_id not in known:
            raise ContractValidationError(f"context binding {binding_id!r} is not bound")
    overlap = selected.intersection(selection.context_binding_ids)
    if overlap:
        raise ContractValidationError(
            f"bindings cannot be both selected and context: {sorted(overlap)}"
        )
```

`_fail` and `DocumentSelectionManifest` are imported for Tasks 3 and 4; if a linter rejects the unused import at this task boundary, keep `DocumentSelectionManifest` (Task 3 uses it in `DiscoveryCheckpoint`) and drop `_fail` until Task 3 needs it.

- [ ] **Step 4: Run the tests**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_discovery_selection.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/agents/v2/discovery/__init__.py \
        backend/app/services/agents/v2/discovery/contracts.py \
        backend/tests/agents/v2/contracts/test_discovery_selection.py
git commit -m "feat(v2): add discovery admission, slot, and selection contracts"
```

---

### Task 3: Probe, aggregation, selection, and discovery checkpoint contracts

**Files:**
- Modify: `backend/app/services/agents/v2/discovery/contracts.py`
- Test: `backend/tests/agents/v2/contracts/test_discovery_aggregation.py`

**Interfaces:**
- Consumes: `TargetSlot`, `validate_target_slots`, `DocumentSelectionManifest`.
- Produces:
  - `PUBLIC_NAMESPACE: UUID`
  - `MatchKind`
  - `SearchProbe(probe_id, slot_id, query, round, origin)`
  - `ProbeCandidateMatch(probe_id, slot_id, task_id, candidate_id, document_id, document_revision, rank, confidence, match_kind, calibration_version)`
  - `SlotCandidateAggregate(aggregate_id, slot_id, document_id, document_revision, source_candidate_ids, source_probe_ids, source_task_ids, best_match_kind, aggregate_confidence, best_rank)`
  - `DiscoverySelection(slot_id, selected_aggregate_ids, authority)`
  - `DiscoveryCheckpoint(target_slots, accepted_probes, candidate_matches, slot_aggregates, selections, rounds_consumed, probes_consumed, status, clarification_manifest)`
  - `aggregate_id_for(slot_id, document_id, document_revision) -> UUID`
  - `aggregate_matches(matches, slots) -> tuple[SlotCandidateAggregate, ...]`
  - `selection_margin(aggregates, aggregate_ids_by_slot, slots, selected) -> float`
  - `validate_discovery_checkpoint(checkpoint, slots) -> None`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/agents/v2/contracts/test_discovery_aggregation.py
"""Aggregation keys, dedup, confidence aggregation, and margin boundary."""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.contracts.validation import ContractValidationError
from app.services.agents.v2.discovery.contracts import (
    DiscoveryCheckpoint,
    DiscoverySelection,
    ProbeCandidateMatch,
    TargetSlot,
    aggregate_id_for,
    aggregate_matches,
    selection_margin,
    validate_discovery_checkpoint,
)

DOC_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
DOC_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _slot(min_selections: int = 1, max_selections: int = 1) -> TargetSlot:
    return TargetSlot(
        slot_id="primary",
        intended_role="target",
        subject_hint="A",
        required=True,
        min_selections=min_selections,
        max_selections=max_selections,
        explicit_binding_ids=(),
        source="research_need",
    )


def _match(
    task_id: str, probe_id: str, doc: UUID, confidence: float, rank: int
) -> ProbeCandidateMatch:
    return ProbeCandidateMatch(
        probe_id=probe_id,
        slot_id="primary",
        task_id=task_id,
        candidate_id=uuid4(),
        document_id=doc,
        document_revision="rev-1",
        rank=rank,
        confidence=confidence,
        match_kind="semantic",
        calibration_version="cal-1",
    )


def test_same_document_from_two_tasks_and_probes_is_one_aggregate() -> None:
    matches = (_match("t1", "p1", DOC_A, 0.70, 1), _match("t2", "p2", DOC_A, 0.90, 2))

    aggregates = aggregate_matches(matches, (_slot(),))

    assert len(aggregates) == 1
    assert aggregates[0].aggregate_id == aggregate_id_for("primary", DOC_A, "rev-1")
    assert aggregates[0].source_probe_ids == ("p1", "p2")
    assert aggregates[0].source_task_ids == ("t1", "t2")
    assert aggregates[0].aggregate_confidence == pytest.approx(0.90)
    assert aggregates[0].best_rank == 1


def test_two_documents_produce_two_aggregates() -> None:
    matches = (_match("t1", "p1", DOC_A, 0.90, 1), _match("t1", "p1", DOC_B, 0.80, 2))

    aggregates = aggregate_matches(matches, (_slot(max_selections=2),))

    assert {aggregate.document_id for aggregate in aggregates} == {DOC_A, DOC_B}


def test_aggregate_id_is_stable_for_minted_candidate_ids() -> None:
    first = aggregate_matches((_match("t1", "p1", DOC_A, 0.9, 1),), (_slot(),))[0]
    second = aggregate_matches((_match("t1", "p1", DOC_A, 0.9, 1),), (_slot(),))[0]

    assert first.aggregate_id == second.aggregate_id


def test_multi_select_margin_ignores_top_two_tie() -> None:
    slot = _slot(min_selections=3, max_selections=5)
    matches = tuple(
        _match("t1", "p1", UUID(int=index + 1), 0.90 - index * 0.001, index + 1)
        for index in range(5)
    )
    aggregates = aggregate_matches(matches, (slot,))
    by_slot = {slot.slot_id: tuple(a.aggregate_id for a in aggregates)}
    selected = {slot.slot_id: tuple(a.aggregate_id for a in aggregates[:3])}

    margin = selection_margin(aggregates, by_slot, (slot,), selected)

    assert margin == pytest.approx(0.001)


def test_single_select_margin_uses_top_two() -> None:
    slot = _slot()
    matches = (_match("t1", "p1", DOC_A, 0.90, 1), _match("t1", "p1", DOC_B, 0.70, 2))
    aggregates = aggregate_matches(matches, (slot,))
    by_slot = {slot.slot_id: tuple(a.aggregate_id for a in aggregates)}

    margin = selection_margin(aggregates, by_slot, (slot,), {slot.slot_id: (aggregates[0].aggregate_id,)})

    assert margin == pytest.approx(0.20)


def test_checkpoint_rejects_selection_for_unknown_slot() -> None:
    checkpoint = DiscoveryCheckpoint(
        target_slots=(_slot(),),
        accepted_probes=(),
        candidate_matches=(),
        slot_aggregates=(),
        selections=(
            DiscoverySelection(
                slot_id="ghost",
                selected_aggregate_ids=(uuid4(),),
                authority="confidence_policy",
            ),
        ),
        rounds_consumed=0,
        probes_consumed=0,
        status="selected",
        clarification_manifest=None,
    )

    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, (_slot(),))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_discovery_aggregation.py -v`
Expected: FAIL — `ImportError: cannot import name 'ProbeCandidateMatch'`.

- [ ] **Step 3: Implement**

Append to `discovery/contracts.py`:

```python
PUBLIC_NAMESPACE = UUID("6f1d4f8e-6a2f-4a5b-9c3d-0f5a7b1c2d3e")
"""Spec §8.2: namespace for derived aggregate IDs (never minted randomly)."""

MatchKind = Literal[
    "exact_document_number",
    "exact_normalized_title",
    "lexical_title",
    "semantic",
]

_MATCH_KIND_ORDER: dict[str, int] = {
    "exact_document_number": 0,
    "exact_normalized_title": 1,
    "lexical_title": 2,
    "semantic": 3,
}

_EXACT_MATCH_KINDS = frozenset({"exact_document_number", "exact_normalized_title"})


class SearchProbe(ContractModel):
    """Spec §8.1: one validated, budgeted discovery probe."""

    probe_id: str
    slot_id: str
    query: str
    round: Literal[1, 2]
    origin: Literal["semantic", "planner_fallback"]


class ProbeCandidateMatch(ContractModel):
    """Spec §8.2: one search task's match for one authoritative document.

    ``slot_id`` is copied from the owning ``SearchProbe``; it is a fact about
    the probe, not a new authority.
    """

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
    """Spec §8.2: one document identity within one slot."""

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
    """Spec §8.3: the chosen aggregates for one slot."""

    slot_id: str
    selected_aggregate_ids: tuple[UUID, ...]
    authority: Literal["exact_match_policy", "confidence_policy", "user_choice"]


class DiscoveryCheckpoint(ContractModel):
    """Spec §8.4: the checkpointed discovery bootstrap state."""

    target_slots: tuple[TargetSlot, ...]
    accepted_probes: tuple[SearchProbe, ...]
    candidate_matches: tuple[ProbeCandidateMatch, ...]
    slot_aggregates: tuple[SlotCandidateAggregate, ...]
    selections: tuple[DiscoverySelection, ...]
    rounds_consumed: int
    probes_consumed: int
    status: Literal["searching", "ranking", "clarification", "selected", "unavailable"]
    clarification_manifest: DocumentSelectionManifest | None = None


def aggregate_id_for(slot_id: str, document_id: UUID, document_revision: str) -> UUID:
    """Spec §8.2: derive the selectable aggregate ID from checkpointed facts."""

    return uuid5(PUBLIC_NAMESPACE, f"{slot_id}:{document_id}:{document_revision}")


def aggregate_matches(
    matches: tuple[ProbeCandidateMatch, ...], slots: tuple[TargetSlot, ...]
) -> tuple[SlotCandidateAggregate, ...]:
    """Spec §8.2: one aggregate per ``(slot_id, document_id, revision)``."""

    slot_ids = {slot.slot_id for slot in slots}
    grouped: dict[tuple[str, UUID, str], list[ProbeCandidateMatch]] = {}
    for match in matches:
        if match.slot_id not in slot_ids:
            raise ContractValidationError(
                f"candidate match references unknown slot {match.slot_id!r}"
            )
        grouped.setdefault(
            (match.slot_id, match.document_id, match.document_revision), []
        ).append(match)
    aggregates: list[SlotCandidateAggregate] = []
    for (slot_id, document_id, revision), group in grouped.items():
        best = min(
            group, key=lambda item: (_MATCH_KIND_ORDER[item.match_kind], item.rank)
        )
        confidences = [
            item.confidence
            for item in group
            if item.confidence is not None and item.match_kind not in _EXACT_MATCH_KINDS
        ]
        aggregates.append(
            SlotCandidateAggregate(
                aggregate_id=aggregate_id_for(slot_id, document_id, revision),
                slot_id=slot_id,
                document_id=document_id,
                document_revision=revision,
                source_candidate_ids=tuple(
                    sorted({item.candidate_id for item in group}, key=str)
                ),
                source_probe_ids=tuple(dict.fromkeys(item.probe_id for item in group)),
                source_task_ids=tuple(dict.fromkeys(item.task_id for item in group)),
                best_match_kind=best.match_kind,
                aggregate_confidence=max(confidences) if confidences else None,
                best_rank=min(item.rank for item in group),
            )
        )
    aggregates.sort(
        key=lambda item: (
            _MATCH_KIND_ORDER[item.best_match_kind],
            item.best_rank,
            str(item.document_id),
        )
    )
    return tuple(aggregates)


def selection_margin(
    aggregates: tuple[SlotCandidateAggregate, ...],
    aggregate_ids_by_slot: dict[str, tuple[UUID, ...]],
    slots: tuple[TargetSlot, ...],
    selected: dict[str, tuple[UUID, ...]],
) -> float:
    """Spec §8.2: margin at the selection boundary, not between the top two."""

    margin = 1.0
    by_id = {aggregate.aggregate_id: aggregate for aggregate in aggregates}
    for slot in slots:
        ordered = [
            by_id[aggregate_id]
            for aggregate_id in aggregate_ids_by_slot.get(slot.slot_id, ())
            if aggregate_id in by_id
        ]
        chosen = set(selected.get(slot.slot_id, ()))
        if slot.min_selections == 1:
            scored = [item for item in ordered if item.aggregate_confidence is not None]
            if not scored:
                continue
            top = scored[0].aggregate_confidence or 0.0
            second = scored[1].aggregate_confidence if len(scored) > 1 else 0.0
            margin = min(margin, top - (second or 0.0))
            continue
        last_selected = [item for item in ordered if item.aggregate_id in chosen]
        first_excluded = [item for item in ordered if item.aggregate_id not in chosen]
        if not first_excluded or not last_selected:
            continue
        boundary = last_selected[min(slot.min_selections, len(last_selected)) - 1]
        if (
            boundary.aggregate_confidence is None
            or first_excluded[0].aggregate_confidence is None
        ):
            continue
        margin = min(
            margin,
            boundary.aggregate_confidence - first_excluded[0].aggregate_confidence,
        )
    return margin


def validate_discovery_checkpoint(
    checkpoint: DiscoveryCheckpoint, slots: tuple[TargetSlot, ...]
) -> None:
    """Spec §8.4: slot ownership, budgets, and selection integrity."""

    validate_target_slots(slots)
    slot_by_id = {slot.slot_id: slot for slot in slots}
    _require_unique((probe.probe_id for probe in checkpoint.accepted_probes), "probe_id")
    for probe in checkpoint.accepted_probes:
        if probe.slot_id not in slot_by_id:
            _fail(f"probe {probe.probe_id} references unknown slot {probe.slot_id!r}")
        _require_non_blank(probe.query, f"probe {probe.probe_id} query")
    _require_unique(
        (aggregate.aggregate_id for aggregate in checkpoint.slot_aggregates), "aggregate_id"
    )
    aggregate_by_id = {
        aggregate.aggregate_id: aggregate for aggregate in checkpoint.slot_aggregates
    }
    for aggregate in checkpoint.slot_aggregates:
        if aggregate.slot_id not in slot_by_id:
            _fail(
                f"aggregate {aggregate.aggregate_id} references unknown slot "
                f"{aggregate.slot_id!r}"
            )
        expected = aggregate_id_for(
            aggregate.slot_id, aggregate.document_id, aggregate.document_revision
        )
        if aggregate.aggregate_id != expected:
            _fail(f"aggregate {aggregate.aggregate_id} is not derived from its key")
    for selection in checkpoint.selections:
        slot = slot_by_id.get(selection.slot_id)
        if slot is None:
            _fail(f"selection references unknown slot {selection.slot_id!r}")
        _require_unique(
            selection.selected_aggregate_ids, f"slot {selection.slot_id} selection"
        )
        for aggregate_id in selection.selected_aggregate_ids:
            aggregate = aggregate_by_id.get(aggregate_id)
            if aggregate is None or aggregate.slot_id != selection.slot_id:
                _fail(
                    f"selection for {selection.slot_id} references foreign aggregate "
                    f"{aggregate_id}"
                )
        count = len(selection.selected_aggregate_ids)
        if not (slot.min_selections <= count <= slot.max_selections):
            _fail(
                f"slot {selection.slot_id} selected {count}, expected "
                f"{slot.min_selections}..{slot.max_selections}"
            )
    if checkpoint.probes_consumed < 0 or checkpoint.rounds_consumed < 0:
        _fail("discovery budgets cannot be negative")
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_discovery_aggregation.py tests/agents/v2/contracts/test_discovery_selection.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/agents/v2/discovery/contracts.py \
        backend/tests/agents/v2/contracts/test_discovery_aggregation.py
git commit -m "feat(v2): add discovery probe, aggregation, and checkpoint contracts"
```

---

### Task 4: Task origins with kind dispatch

**Files:**
- Modify: `backend/app/services/agents/v2/contracts/planning.py`
- Modify: `backend/app/services/agents/v2/contracts/validation.py`
- Modify: `backend/tests/agents/v2/contracts/factories.py`
- Test: `backend/tests/agents/v2/contracts/test_discovery_task_origins.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `DiscoveryProbeTaskOrigin(kind="discovery_probe", probe_id, slot_id, round)`
  - `DiscoveryExpansionTaskOrigin(kind="discovery_expansion", source_search_task_ids, target_slot_ids, selected_aggregate_ids)`
  - `_validate_task_origin` dispatching by kind.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/agents/v2/contracts/test_discovery_task_origins.py
"""New task origins validate by kind instead of raising AttributeError."""
from __future__ import annotations

import pytest

from app.services.agents.v2.contracts.planning import DiscoveryProbeTaskOrigin
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_task_plan,
)
from tests.agents.v2.contracts.factories import binding_set, plan_with_origin


def test_probe_origin_passes() -> None:
    plan = plan_with_origin(
        DiscoveryProbeTaskOrigin(
            kind="discovery_probe", probe_id="p1", slot_id="primary", round=1
        )
    )

    validate_task_plan(plan, binding_set())


def test_blank_probe_id_is_rejected() -> None:
    plan = plan_with_origin(
        DiscoveryProbeTaskOrigin(
            kind="discovery_probe", probe_id=" ", slot_id="primary", round=1
        )
    )

    with pytest.raises(ContractValidationError):
        validate_task_plan(plan, binding_set())


def test_expansion_origin_references_unknown_task() -> None:
    from uuid import uuid4

    from app.services.agents.v2.contracts.planning import (
        DiscoveryExpansionTaskOrigin,
    )

    plan = plan_with_origin(
        DiscoveryExpansionTaskOrigin(
            kind="discovery_expansion",
            source_search_task_ids=("ghost",),
            target_slot_ids=("primary",),
            selected_aggregate_ids=(uuid4(),),
        )
    )

    with pytest.raises(ContractValidationError):
        validate_task_plan(plan, binding_set())
```

Add to `factories.py` (uses the verified constructors):

```python
def plan_with_origin(origin, *, task_id: str = "task-1") -> TaskPlan:
    """One-target, one-task plan carrying ``origin``."""
    return TaskPlan(
        contract_version="2.0",
        plan_id="plan-1",
        goal="Mục tiêu",
        target_units=(target_unit(target_id="t1", binding_id="b1"),),
        tasks=(
            TaskSpec(
                task_id=task_id,
                capability="document.read",
                task_objective="Đọc tài liệu",
                input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
                depends_on=(),
                origin=origin,
            ),
        ),
    )
```

Add `TaskSpec` to the existing `contracts.planning` import block in `factories.py` if it is not already imported.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_discovery_task_origins.py -v`
Expected: FAIL — `AttributeError: 'DiscoveryProbeTaskOrigin' object has no attribute 'reason'`.

- [ ] **Step 3: Add the origins and the dispatch**

`contracts/planning.py` — after `ReplanTaskOrigin`:

```python
class DiscoveryProbeTaskOrigin(ContractModel):
    """Spec §13: a governed bootstrap ``document.search`` task."""

    kind: Literal["discovery_probe"]
    probe_id: str
    slot_id: str
    round: Literal[1, 2]


class DiscoveryExpansionTaskOrigin(ContractModel):
    """Spec §13: a factual task appended after discovery selection."""

    kind: Literal["discovery_expansion"]
    source_search_task_ids: tuple[str, ...]
    target_slot_ids: tuple[str, ...]
    selected_aggregate_ids: tuple[UUID, ...]
```

and widen the union:

```python
TaskOrigin = Annotated[
    InitialTaskOrigin
    | ReplanTaskOrigin
    | DiscoveryProbeTaskOrigin
    | DiscoveryExpansionTaskOrigin,
    Field(discriminator="kind"),
]
```

`contracts/validation.py` — replace `_validate_task_origin` (currently around line 619):

```python
def _validate_task_origin(task: TaskSpec, task_ids: set[str]) -> None:
    origin = task.origin
    if isinstance(origin, InitialTaskOrigin):
        return
    if isinstance(origin, DiscoveryProbeTaskOrigin):
        _require_non_blank(
            origin.probe_id, f"DiscoveryProbeTaskOrigin.probe_id for task {task.task_id}"
        )
        _require_non_blank(
            origin.slot_id, f"DiscoveryProbeTaskOrigin.slot_id for task {task.task_id}"
        )
        return
    if isinstance(origin, DiscoveryExpansionTaskOrigin):
        _require_unique(
            origin.source_search_task_ids,
            f"DiscoveryExpansionTaskOrigin.source_search_task_ids for task {task.task_id}",
        )
        _require_unique(
            origin.target_slot_ids,
            f"DiscoveryExpansionTaskOrigin.target_slot_ids for task {task.task_id}",
        )
        _require_unique(
            origin.selected_aggregate_ids,
            f"DiscoveryExpansionTaskOrigin.selected_aggregate_ids for task {task.task_id}",
        )
        if task.task_id in origin.source_search_task_ids:
            _fail(f"task {task.task_id} origin cannot reference its own task id")
        for reference in origin.source_search_task_ids:
            if reference not in task_ids:
                _fail(f"task {task.task_id} origin references unknown task {reference}")
        return
    _require_non_blank(origin.reason, f"ReplanTaskOrigin.reason for task {task.task_id}")
    _require_unique(origin.task_ids, "ReplanTaskOrigin.task_ids")
    _require_unique(origin.evidence_use_ids, "ReplanTaskOrigin.evidence_use_ids")
    if task.task_id in origin.task_ids:
        _fail(f"task {task.task_id} origin cannot reference its own task id")
    for reference in origin.task_ids:
        if reference not in task_ids:
            _fail(f"task {task.task_id} origin references unknown task {reference}")
```

Extend the planning import block in `validation.py` with `DiscoveryExpansionTaskOrigin` and `DiscoveryProbeTaskOrigin`.

- [ ] **Step 4: Run the tests**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_discovery_task_origins.py tests/agents/v2/contracts/test_task_summary_checkpoint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/agents/v2/contracts/planning.py \
        backend/app/services/agents/v2/contracts/validation.py \
        backend/tests/agents/v2/contracts/test_discovery_task_origins.py \
        backend/tests/agents/v2/contracts/factories.py
git commit -m "feat(v2): add discovery task origins with kind dispatch"
```

---

### Task 5: Checkpoint schema revision, migration, and slot registration

**Files:**
- Modify: `backend/app/services/agents/v2/contracts/state.py`
- Modify: `backend/app/services/agents/v2/contracts/validation.py`
- Modify: `backend/app/services/agents/supervisor_v2.py`
- Modify: `backend/tests/agents/v2/contracts/factories.py`
- Test: `backend/tests/agents/v2/contracts/test_checkpoint_revision.py`

**Interfaces:**
- Consumes: all four contract types from Tasks 1–3.
- Produces:
  - `CHECKPOINT_SCHEMA_REVISION: Final[Literal[2]] = 2` and `CheckpointSchemaRevision = Literal[1, 2]` in `contracts/state.py`
  - `migrate_checkpoint_payload(payload) -> dict[str, object]` in `contracts/validation.py`
  - four new `SupervisorV2State` keys registered in `_SLOT_MODELS`/`_NULLABLE_SLOTS`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/agents/v2/contracts/test_checkpoint_revision.py
"""Revision-1 checkpoints migrate; corrupt revision-2 checkpoints fail closed."""
from __future__ import annotations

import pytest

from app.services.agents.supervisor_v2 import normalize_checkpoint_state
from app.services.agents.v2.contracts.state import CHECKPOINT_SCHEMA_REVISION
from app.services.agents.v2.contracts.validation import (
    IncompatibleCheckpointError,
    migrate_checkpoint_payload,
)
from tests.agents.v2.contracts.factories import minimal_supervisor_payload

NEW_SLOTS = (
    "discovery_need",
    "discovery",
    "document_selection_clarification",
    "research_target_selection",
)


def test_revision_one_payload_gains_every_new_slot() -> None:
    payload = minimal_supervisor_payload()
    payload.pop("checkpoint_schema_revision", None)
    for key in NEW_SLOTS:
        payload.pop(key, None)

    migrated = migrate_checkpoint_payload(payload)

    assert migrated["checkpoint_schema_revision"] == CHECKPOINT_SCHEMA_REVISION
    for key in NEW_SLOTS:
        assert migrated[key] is None


def test_revision_two_payload_missing_a_slot_fails_closed() -> None:
    payload = minimal_supervisor_payload()
    payload.pop("discovery", None)

    with pytest.raises(IncompatibleCheckpointError):
        migrate_checkpoint_payload(payload)


def test_unknown_revision_fails_closed() -> None:
    payload = minimal_supervisor_payload()
    payload["checkpoint_schema_revision"] = 99

    with pytest.raises(IncompatibleCheckpointError):
        migrate_checkpoint_payload(payload)


def test_normalize_accepts_legacy_shape() -> None:
    legacy = minimal_supervisor_payload()
    legacy.pop("checkpoint_schema_revision", None)
    for key in NEW_SLOTS:
        legacy.pop(key, None)

    view = normalize_checkpoint_state(legacy)

    assert view["checkpoint_schema_revision"] == CHECKPOINT_SCHEMA_REVISION
    assert view["discovery"] is None
    assert view["research_target_selection"] is None


def test_fresh_state_is_revision_two() -> None:
    from app.services.agents.supervisor_v2 import build_initial_v2_state

    state = build_initial_v2_state(request=_request())

    assert state["checkpoint_schema_revision"] == CHECKPOINT_SCHEMA_REVISION
    assert state["discovery_need"] is None


def _request():
    from tests.agents.v2.contracts.factories import request_context

    return request_context()
```

Add to `factories.py`:

```python
def minimal_supervisor_payload() -> dict:
    """A minimal revision-2 root payload for migration tests."""
    from app.services.agents.supervisor_v2 import build_initial_v2_state

    return dict(build_initial_v2_state(request=request_context()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_checkpoint_revision.py -v`
Expected: FAIL — `ImportError: cannot import name 'migrate_checkpoint_payload'`.

- [ ] **Step 3: Add state keys and the migration**

`contracts/state.py` — add `Final` to the `typing` import, then:

```python
CHECKPOINT_SCHEMA_REVISION: Final[Literal[2]] = 2
"""Spec §6: the revision this build writes and validates."""

CheckpointSchemaRevision = Literal[1, 2]
```

Extend `SupervisorV2State` (keep `total=True`; append after `synthesis`):

```python
    checkpoint_schema_revision: CheckpointSchemaRevision
    discovery_need: DiscoveryNeed | None
    discovery: DiscoveryCheckpoint | None
    document_selection_clarification: DocumentSelectionClarification | None
    research_target_selection: ResearchTargetSelection | None
```

Import `DiscoveryCheckpoint`, `DiscoveryNeed`, `ResearchTargetSelection` from `..discovery.contracts` and `DocumentSelectionClarification` from `.clarification`. If that creates an import cycle (`discovery.contracts` imports `contracts.validation`, which imports `contracts.state`), type the four keys as `Any` in `state.py` and register the real models in `_SLOT_MODELS` — `state.py` must not import `discovery`. **Do that by default:** use `Any` here, and keep the concrete models in `_SLOT_MODELS`.

`contracts/validation.py` — extend `_CHECKPOINT_REQUIRED_KEYS` and add the migration:

```python
_CHECKPOINT_REQUIRED_KEYS = (
    "request",
    "conversation",
    "semantic",
    "bindings",
    "execution",
    "query_analysis",
    "route_decision",
    "clarification",
    "synthesis",
    "final_response",
    "checkpoint_schema_revision",
    "discovery_need",
    "discovery",
    "document_selection_clarification",
    "research_target_selection",
)

_REVISION_ONE_ROOT_ADDITIONS = (
    "discovery_need",
    "discovery",
    "document_selection_clarification",
    "research_target_selection",
)


def migrate_checkpoint_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Spec §6: materialize the supported legacy root shape.

    Revision 1 is the pre-discovery root shape: it lacks
    ``checkpoint_schema_revision`` and the four discovery slots. A revision-2
    payload missing a required key is corrupt and is never migrated; the
    caller's required-key gate rejects it.
    """

    migrated = dict(payload)
    revision = migrated.get("checkpoint_schema_revision")
    if revision is None:
        # The pre-synthesis migration also materializes ``synthesis``.
        migrated.setdefault("synthesis", None)
        migrated["checkpoint_schema_revision"] = 2
        for key in _REVISION_ONE_ROOT_ADDITIONS:
            migrated.setdefault(key, None)
    elif revision != 2:
        raise IncompatibleCheckpointError(
            f"checkpoint declares unknown checkpoint_schema_revision {revision!r}"
        )
    return migrated
```

and in `validate_checkpoint_payload`, migrate first and gate the revision:

```python
    payload = migrate_checkpoint_payload(payload)
    declared = payload.get("contract_version")
    ...
    if payload.get("checkpoint_schema_revision") != 2:
        raise IncompatibleCheckpointError(
            "checkpoint payload must declare checkpoint_schema_revision 2 after migration"
        )
```

- [ ] **Step 4: Register the four slots in `supervisor_v2.py`**

Import the four concrete contracts:

```python
from .v2.contracts.clarification import (
    ClarificationRequest,
    ClarificationResolution,
    DocumentSelectionClarification,
)
from .v2.discovery.contracts import (
    DiscoveryCheckpoint,
    DiscoveryNeed,
    ResearchTargetSelection,
)
from .v2.contracts.state import CHECKPOINT_SCHEMA_REVISION
```

Add to `_SLOT_MODELS`:

```python
    "discovery_need": DiscoveryNeed,
    "discovery": DiscoveryCheckpoint,
    "document_selection_clarification": DocumentSelectionClarification,
    "research_target_selection": ResearchTargetSelection,
```

Add the same four names to `_NULLABLE_SLOTS`.

Replace the inline legacy branch in `normalize_checkpoint_state`:

```python
    values = dict(state)
    if values.get("contract_version") == CONTRACT_VERSION and "synthesis" not in values:
        values["synthesis"] = None
```

with:

```python
    values = migrate_checkpoint_payload(state)
```

(import `migrate_checkpoint_payload` from `.v2.contracts.validation`).

Add the new keys to the `SupervisorV2State(...)` construction in `build_initial_v2_state`:

```python
        checkpoint_schema_revision=CHECKPOINT_SCHEMA_REVISION,
        discovery_need=None,
        discovery=None,
        document_selection_clarification=None,
        research_target_selection=None,
```

- [ ] **Step 5: Run the tests**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_checkpoint_revision.py -v`
Expected: PASS.

Run: `cd backend && python -m pytest tests/agents/v2/test_supervisor_v2.py tests/agents/v2/contracts -q`
Expected: PASS. Fix any regression before committing.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/agents/v2/contracts/state.py \
        backend/app/services/agents/v2/contracts/validation.py \
        backend/app/services/agents/supervisor_v2.py \
        backend/tests/agents/v2/contracts/test_checkpoint_revision.py \
        backend/tests/agents/v2/contracts/factories.py
git commit -m "feat(v2): add checkpoint schema revision and discovery root slots"
```

---

### Task 6: Selection-aware plan validation and plumbing

**Files:**
- Modify: `backend/app/services/agents/v2/contracts/validation.py`
- Modify: `backend/app/services/agents/v2/complex_research_graph.py`
- Modify: `backend/app/services/agents/v2/planning/planner.py`
- Test: `backend/tests/agents/v2/contracts/test_selection_aware_validation.py`

**Interfaces:**
- Consumes: `ResearchTargetSelection`, `TargetSlot`, `validate_research_target_selection` (Task 2); `DiscoveryCheckpoint` (Task 3).
- Produces:
  - `validate_task_plan(plan, bindings, *, target_selection=None, target_slots=())`
  - `ResearchPlanningInput.target_selection: ResearchTargetSelection | None`
  - `ComplexResearchState["research_target_selection"]` and `["discovery"]`, carried across the boundary.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/agents/v2/contracts/test_selection_aware_validation.py
"""A selected non-role binding is usable only through a selection."""
from __future__ import annotations

import pytest

from app.services.agents.v2.contracts.binding import DocumentBindingSet
from app.services.agents.v2.contracts.planning import InitialTaskOrigin
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_task_plan,
)
from app.services.agents.v2.discovery.contracts import (
    ResearchTargetSelection,
    SlotBindingSelection,
    TargetSlot,
)
from tests.agents.v2.contracts.factories import (
    binding_set,
    plan_with_origin,
    scoped_document,
)

SLOT = TargetSlot(
    slot_id="primary",
    intended_role="target",
    subject_hint="A",
    required=True,
    min_selections=1,
    max_selections=1,
    explicit_binding_ids=(),
    source="research_need",
)


def _supporting() -> DocumentBindingSet:
    return binding_set(bindings=(scoped_document(binding_id="b1", role="supporting"),))


def _plan():
    return plan_with_origin(InitialTaskOrigin(kind="initial"))


def test_supporting_binding_rejected_without_selection() -> None:
    with pytest.raises(ContractValidationError):
        validate_task_plan(_plan(), _supporting())


def test_supporting_binding_accepted_with_selection() -> None:
    selection = ResearchTargetSelection(
        slot_bindings=(SlotBindingSelection(slot_id="primary", binding_ids=("b1",)),),
        context_binding_ids=(),
    )

    validate_task_plan(
        _plan(), _supporting(), target_selection=selection, target_slots=(SLOT,)
    )


def test_context_binding_does_not_authorize_a_target_unit() -> None:
    selection = ResearchTargetSelection(slot_bindings=(), context_binding_ids=("b1",))

    with pytest.raises(ContractValidationError):
        validate_task_plan(
            _plan(), _supporting(), target_selection=selection, target_slots=(SLOT,)
        )


def test_role_binding_still_passes_without_selection() -> None:
    validate_task_plan(_plan(), binding_set())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_selection_aware_validation.py -v`
Expected: FAIL — `TypeError: validate_task_plan() got an unexpected keyword argument 'target_selection'`.

- [ ] **Step 3: Make `validate_task_plan` selection-aware**

Replace the signature and the target-unit loop in `validate_task_plan`:

```python
def validate_task_plan(
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    *,
    target_selection: ResearchTargetSelection | None = None,
    target_slots: tuple[TargetSlot, ...] = (),
) -> None:
    """Spec §13/§26: ID, DAG, binding, target, and criteria integrity for a plan.

    Spec §7: ``target_selection`` authorizes a logical target/reference use for
    a binding whose immutable provenance role is not itself target/reference.
    Without it the historical strict role check applies.
    """

    _validate_plan_structure(plan)
    binding_by_id = {binding.binding_id: binding for binding in bindings.bindings}
    authorized: dict[str, str] = {}
    if target_selection is not None:
        validate_research_target_selection(target_selection, bindings, target_slots)
        role_by_slot = {slot.slot_id: slot.intended_role for slot in target_slots}
        for entry in target_selection.slot_bindings:
            for binding_id in entry.binding_ids:
                authorized[binding_id] = role_by_slot[entry.slot_id]
    for unit in plan.target_units:
        binding = binding_by_id.get(unit.binding_id)
        if binding is None:
            _fail(
                f"target unit {unit.target_id} references unknown binding {unit.binding_id}"
            )
        if binding.role not in ("target", "reference") and unit.binding_id not in authorized:
            _fail(
                f"target unit {unit.target_id} cannot bind a {binding.role!r} document: "
                "target/reference roles are required"
            )
    referenced = {
        target_id for task in plan.tasks for target_id in _input_target_ids(task.input)
    }
    for unit in plan.target_units:
        if unit.target_id not in referenced:
            _fail(f"target unit {unit.target_id} is not referenced by any task input")
```

- [ ] **Step 4: Thread the selection at the bootstrap call sites**

1. `_validate_execution_state` (in the same file):

```python
def _validate_execution_state(
    execution: ExecutionState,
    bindings: DocumentBindingSet,
    target_selection: ResearchTargetSelection | None = None,
    target_slots: tuple[TargetSlot, ...] = (),
) -> None:
    ...
    validate_task_plan(
        execution.plan,
        bindings,
        target_selection=target_selection,
        target_slots=target_slots,
    )
```

2. In `validate_supervisor_state`, replace the final `_validate_execution_state(execution, bindings)` call with:

```python
    document_selection = _optional_model(
        state, "document_selection_clarification", DocumentSelectionClarification
    )
    if document_selection is not None:
        validate_document_selection_request(document_selection)
    validate_clarification_slots_exclusive(state)
    selection = _optional_model(state, "research_target_selection", ResearchTargetSelection)
    discovery = _optional_model(state, "discovery", DiscoveryCheckpoint)
    _validate_execution_state(
        execution,
        bindings,
        selection,
        discovery.target_slots if discovery is not None else (),
    )
```

3. `complex_research_graph.py` — in `validate_checkpoint_node`, replace:

```python
        bindings = state["bindings"]
        validate_task_plan(initial.plan, bindings)
```

with:

```python
        bindings = state["bindings"]
        selection = state.get("research_target_selection")
        discovery = state.get("discovery")
        validate_task_plan(
            initial.plan,
            bindings,
            target_selection=selection,
            target_slots=discovery.target_slots if discovery is not None else (),
        )
```

4. Add `target_selection` to `ResearchPlanningInput` and populate it in `build_planning_input(state, context)` from `state.get("research_target_selection")`; pass it through to `build_governed_initial_proposal` and to the summarize/compare skill policy calls that validate their own plans.

5. Add `discovery` and `research_target_selection` to `ComplexResearchState`, to `build_complex_research_state`, to `normalize_complex_state`, and to `merge_complex_result_into_supervisor` so the values cross the boundary. In `normalize_complex_state`, use `state.get("discovery")` / `state.get("research_target_selection")` (they are already contract instances or mappings; reuse `_coerce_slot` with the concrete models).

- [ ] **Step 5: Run the tests**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_selection_aware_validation.py -v`
Expected: PASS.

Run: `cd backend && python -m pytest tests/agents/v2/complex tests/agents/v2/contracts -q`
Expected: PASS. Any failure here is a real threading regression: fix before committing.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/agents/v2/contracts/validation.py \
        backend/app/services/agents/v2/complex_research_graph.py \
        backend/app/services/agents/v2/planning/planner.py \
        backend/tests/agents/v2/contracts/test_selection_aware_validation.py
git commit -m "feat(v2): make plan validation selection-aware"
```

---

### Task 7: Expansion validator and the governed append owner

**Files:**
- Create: `backend/app/services/agents/v2/discovery/plan_expansion.py`
- Modify: `backend/tests/agents/v2/complex/test_complex_expansion.py`
- Test: `backend/tests/agents/v2/complex/test_discovery_plan_expansion.py`
- Modify: `backend/tests/agents/v2/contracts/factories.py`

**Interfaces:**
- Consumes: `DiscoveryCheckpoint`, `TargetSlot`, `ResearchTargetSelection`, `validate_task_plan`.
- Produces:
  - `validate_plan_expansion(current, proposed, checkpoint, selection, bindings, outcomes, *, target_slots, max_research_tasks, total_task_limit) -> None`
  - `expand_discovery_plan(current, checkpoint, selection, bindings, outcomes, appended_units, appended_tasks, *, target_slots, max_research_tasks, total_task_limit) -> TaskPlan`
  - `factual_tasks(plan) -> tuple[TaskSpec, ...]`

`expand_discovery_plan` does **not** acquire leases. The governed caller (`research_expand_node`, Phase 4) calls the existing `_lease_pinned_state` before checkpointing, exactly like `validate_checkpoint_node` does today. This keeps one lease authority on the graph boundary.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/agents/v2/complex/test_discovery_plan_expansion.py
"""Expansion appends only checkpoint-authorized units and stays budgeted."""
from __future__ import annotations

from uuid import UUID

import pytest

from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
    DiscoveryExpansionTaskOrigin,
    TargetUnit,
    TaskSpec,
)
from app.services.agents.v2.contracts.capability import DocumentReadInput
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.validation import ContractValidationError
from app.services.agents.v2.discovery.contracts import (
    DiscoveryCheckpoint,
    ResearchTargetSelection,
    SearchProbe,
    SlotBindingSelection,
    TargetSlot,
)
from app.services.agents.v2.discovery.plan_expansion import validate_plan_expansion
from tests.agents.v2.contracts.factories import (
    DOCUMENT_ID,
    REVISION,
    binding_set,
    bootstrap_plan,
    scoped_document,
)

AGGREGATE_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")

SLOT = TargetSlot(
    slot_id="primary",
    intended_role="target",
    subject_hint="A",
    required=True,
    min_selections=1,
    max_selections=1,
    explicit_binding_ids=(),
    source="research_need",
)


def _checkpoint() -> DiscoveryCheckpoint:
    from app.services.agents.v2.discovery.contracts import (
        SlotCandidateAggregate,
        aggregate_id_for,
    )

    return DiscoveryCheckpoint(
        target_slots=(SLOT,),
        accepted_probes=(
            SearchProbe(
                probe_id="p1",
                slot_id="primary",
                query="Nghị định 13",
                round=1,
                origin="semantic",
            ),
        ),
        candidate_matches=(),
        slot_aggregates=(
            SlotCandidateAggregate(
                aggregate_id=aggregate_id_for("primary", DOCUMENT_ID, REVISION),
                slot_id="primary",
                document_id=DOCUMENT_ID,
                document_revision=REVISION,
                source_candidate_ids=(),
                source_probe_ids=("p1",),
                source_task_ids=("search-1",),
                best_match_kind="exact_document_number",
                aggregate_confidence=None,
                best_rank=1,
            ),
        ),
        selections=(),
        rounds_consumed=1,
        probes_consumed=1,
        status="selected",
        clarification_manifest=None,
    )


def _bindings():
    return binding_set(bindings=(scoped_document(binding_id="b1"),))


def _selection() -> ResearchTargetSelection:
    return ResearchTargetSelection(
        slot_bindings=(SlotBindingSelection(slot_id="primary", binding_ids=("b1",)),),
        context_binding_ids=(),
    )


def _outcome():
    return type(
        "Outcome",
        (),
        {"task_id": "search-1", "document_id": DOCUMENT_ID, "document_revision": REVISION},
    )()


def _appended_unit() -> TargetUnit:
    return TargetUnit(
        target_id="t1",
        binding_id="b1",
        requested_locator=DocumentLocator(kind="document"),
        completion_criteria=(CoverageCriterion(kind="coverage"),),
    )


def _appended_task(task_id: str = "read-1") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability="document.read",
        task_objective="Đọc tài liệu",
        input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
        depends_on=("search-1",),
        origin=DiscoveryExpansionTaskOrigin(
            kind="discovery_expansion",
            source_search_task_ids=("search-1",),
            target_slot_ids=("primary",),
            selected_aggregate_ids=(aggregate_id_for("primary", DOCUMENT_ID, REVISION),),
        ),
    )


def _proposed(current, *, units=(), tasks=()):
    return current.model_copy(
        update={
            "target_units": current.target_units + tuple(units),
            "tasks": current.tasks + tuple(tasks),
        }
    )


def test_no_appended_unit_is_rejected() -> None:
    bootstrap = bootstrap_plan()

    with pytest.raises(ContractValidationError):
        validate_plan_expansion(
            bootstrap,
            bootstrap,
            _checkpoint(),
            _selection(),
            _bindings(),
            (_outcome(),),
            target_slots=(SLOT,),
            max_research_tasks=8,
            total_task_limit=13,
        )


def test_removing_an_existing_task_is_rejected() -> None:
    bootstrap = bootstrap_plan()
    truncated = bootstrap.model_copy(update={"tasks": ()})

    with pytest.raises(ContractValidationError):
        validate_plan_expansion(
            bootstrap,
            truncated,
            _checkpoint(),
            _selection(),
            _bindings(),
            (_outcome(),),
            target_slots=(SLOT,),
            max_research_tasks=8,
            total_task_limit=13,
        )


def test_over_budget_is_rejected() -> None:
    bootstrap = bootstrap_plan()
    expanded = _proposed(bootstrap, units=(_appended_unit(),), tasks=(_appended_task(),))

    with pytest.raises(ContractValidationError):
        validate_plan_expansion(
            bootstrap,
            expanded,
            _checkpoint(),
            _selection(),
            _bindings(),
            (_outcome(),),
            target_slots=(SLOT,),
            max_research_tasks=0,
            total_task_limit=13,
        )


def test_valid_expansion_passes() -> None:
    bootstrap = bootstrap_plan()
    expanded = _proposed(bootstrap, units=(_appended_unit(),), tasks=(_appended_task(),))

    validate_plan_expansion(
        bootstrap,
        expanded,
        _checkpoint(),
        _selection(),
        _bindings(),
        (_outcome(),),
        target_slots=(SLOT,),
        max_research_tasks=8,
        total_task_limit=13,
    )


def test_unauthorized_unit_is_rejected() -> None:
    bootstrap = bootstrap_plan()
    unauthorized = _appended_unit().model_copy(update={"binding_id": "b2"})
    expanded = _proposed(bootstrap, units=(unauthorized,), tasks=(_appended_task(),))

    with pytest.raises(ContractValidationError):
        validate_plan_expansion(
            bootstrap,
            expanded,
            _checkpoint(),
            _selection(),
            _bindings(),
            (_outcome(),),
            target_slots=(SLOT,),
            max_research_tasks=8,
            total_task_limit=13,
        )
```

Add to `factories.py`:

```python
def search_task(task_id: str = "search-1", probe_id: str = "p1"):
    from app.services.agents.v2.contracts.capability import DocumentSearchInput
    from app.services.agents.v2.contracts.planning import (
        DiscoveryProbeTaskOrigin,
        TaskSpec,
    )

    return TaskSpec(
        task_id=task_id,
        capability="document.search",
        task_objective="Tìm tài liệu",
        input=DocumentSearchInput(kind="document.search", query="Nghị định 13"),
        depends_on=(),
        origin=DiscoveryProbeTaskOrigin(
            kind="discovery_probe", probe_id=probe_id, slot_id="primary", round=1
        ),
    )


def bootstrap_plan(*, task_id: str = "search-1") -> TaskPlan:
    """A discovery bootstrap plan: probe tasks only, no target units."""
    return TaskPlan(
        contract_version="2.0",
        plan_id="plan-1",
        goal="Tìm và tóm tắt",
        target_units=(),
        tasks=(search_task(task_id),),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/agents/v2/complex/test_discovery_plan_expansion.py -v`
Expected: FAIL — `ModuleNotFoundError: app.services.agents.v2.discovery.plan_expansion`.

- [ ] **Step 3: Implement the validator and constructor**

`backend/app/services/agents/v2/discovery/plan_expansion.py`:

```python
"""Spec §13.1: the single governed discovery-expansion transition."""
from __future__ import annotations

from typing import Any, Sequence

from ..contracts.binding import DocumentBindingSet
from ..contracts.planning import (
    DiscoveryExpansionTaskOrigin,
    DiscoveryProbeTaskOrigin,
    TaskPlan,
    TaskSpec,
)
from ..contracts.validation import (
    _fail,
    validate_task_plan,
)
from .contracts import (
    DiscoveryCheckpoint,
    ResearchTargetSelection,
    TargetSlot,
)


def factual_tasks(plan: TaskPlan) -> tuple[TaskSpec, ...]:
    """Spec §11.3: bootstrap probes never consume research budget."""

    return tuple(
        task for task in plan.tasks if not isinstance(task.origin, DiscoveryProbeTaskOrigin)
    )


def validate_plan_expansion(
    current: TaskPlan,
    proposed: TaskPlan,
    checkpoint: DiscoveryCheckpoint,
    selection: ResearchTargetSelection,
    bindings: DocumentBindingSet,
    outcomes: Sequence[Any],
    *,
    target_slots: tuple[TargetSlot, ...] = (),
    max_research_tasks: int,
    total_task_limit: int,
) -> None:
    """Spec §13.1: append-only, selection-authorized, budgeted expansion."""

    if (
        proposed.plan_id != current.plan_id
        or proposed.goal != current.goal
        or proposed.contract_version != current.contract_version
    ):
        _fail("plan expansion must preserve plan_id, goal, and contract_version")
    for index, task in enumerate(current.tasks):
        if index >= len(proposed.tasks) or proposed.tasks[index] != task:
            _fail("plan expansion must preserve every existing task as an exact prefix")
    for index, unit in enumerate(current.target_units):
        if index >= len(proposed.target_units) or proposed.target_units[index] != unit:
            _fail("plan expansion must preserve every existing target unit as an exact prefix")
    if len(proposed.target_units) <= len(current.target_units):
        _fail("plan expansion must append at least one target unit")
    new_tasks = proposed.tasks[len(current.tasks):]
    if not new_tasks:
        _fail("plan expansion must append at least one factual task")

    slot_by_id = {slot.slot_id: slot for slot in target_slots}
    authorized = {
        binding_id
        for entry in selection.slot_bindings
        for binding_id in entry.binding_ids
        if entry.slot_id in slot_by_id
    }
    for unit in proposed.target_units[len(current.target_units):]:
        if unit.binding_id not in authorized:
            _fail(
                f"appended target unit {unit.target_id} is not authorized by the selection"
            )

    bootstrap_task_ids = {
        task.task_id
        for task in current.tasks
        if isinstance(task.origin, DiscoveryProbeTaskOrigin)
    }
    outcome_keys = {
        (
            outcome.task_id,
            getattr(outcome, "document_id", None),
            getattr(outcome, "document_revision", None),
        )
        for outcome in outcomes
    }
    aggregate_by_id = {
        aggregate.aggregate_id: aggregate for aggregate in checkpoint.slot_aggregates
    }
    for task in new_tasks:
        origin = task.origin
        if not isinstance(origin, DiscoveryExpansionTaskOrigin):
            _fail(f"appended task {task.task_id} must carry DiscoveryExpansionTaskOrigin")
        for reference in origin.source_search_task_ids:
            if reference not in bootstrap_task_ids:
                _fail(
                    f"expanded task {task.task_id} references non-bootstrap task {reference}"
                )
        for aggregate_id in origin.selected_aggregate_ids:
            aggregate = aggregate_by_id.get(aggregate_id)
            if aggregate is None:
                _fail(
                    f"expanded task {task.task_id} references unknown aggregate {aggregate_id}"
                )
            if not any(
                (task_id, aggregate.document_id, aggregate.document_revision) in outcome_keys
                for task_id in aggregate.source_task_ids
            ):
                _fail(f"aggregate {aggregate_id} has no outcome for any of its source tasks")

    if len(factual_tasks(proposed)) > max_research_tasks:
        _fail("plan expansion exceeds the research task budget")
    if len(proposed.tasks) > total_task_limit:
        _fail("plan expansion exceeds the total task limit")
    validate_task_plan(
        proposed,
        bindings,
        target_selection=selection,
        target_slots=target_slots,
    )


def expand_discovery_plan(
    current: TaskPlan,
    checkpoint: DiscoveryCheckpoint,
    selection: ResearchTargetSelection,
    bindings: DocumentBindingSet,
    outcomes: Sequence[Any],
    appended_units: tuple[Any, ...],
    appended_tasks: tuple[TaskSpec, ...],
    *,
    target_slots: tuple[TargetSlot, ...],
    max_research_tasks: int,
    total_task_limit: int,
) -> TaskPlan:
    """Spec §13.1: the single governed constructor for this transition.

    Returns nothing checkpointable before validation passes. The caller
    (``research_expand_node``) leases the returned plan through
    ``_lease_pinned_state`` before writing it to the checkpoint.
    """

    proposed = current.model_copy(
        update={
            "target_units": current.target_units + tuple(appended_units),
            "tasks": current.tasks + tuple(appended_tasks),
        }
    )
    validate_plan_expansion(
        current,
        proposed,
        checkpoint,
        selection,
        bindings,
        outcomes,
        target_slots=target_slots,
        max_research_tasks=max_research_tasks,
        total_task_limit=total_task_limit,
    )
    return proposed
```

- [ ] **Step 4: Extend the AST ownership guard**

In `backend/tests/agents/v2/complex/test_complex_expansion.py`, make the allowlist explicit and cover target-unit appends:

```python
ALLOWED_PLAN_APPEND_OWNERS = frozenset({"append_replan_tasks", "expand_discovery_plan"})


def test_only_governed_owners_append_to_plans() -> None:
    found = scan_plan_append_owners(BACKEND_ROOT)  # existing scanner, extended
    assert found == ALLOWED_PLAN_APPEND_OWNERS, sorted(found)
```

Extend the scanner so it flags `plan.target_units + (...)` and `plan.tasks + (...)` outside the allowlist, and add a synthetic-violator case that writes such an expression to a temp module and asserts the scanner reports it. Keep the existing no-wildcard/no-directory rule.

- [ ] **Step 5: Run the tests**

Run: `cd backend && python -m pytest tests/agents/v2/complex/test_discovery_plan_expansion.py tests/agents/v2/complex/test_complex_expansion.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/agents/v2/discovery/plan_expansion.py \
        backend/tests/agents/v2/complex/test_discovery_plan_expansion.py \
        backend/tests/agents/v2/complex/test_complex_expansion.py \
        backend/tests/agents/v2/contracts/factories.py
git commit -m "feat(v2): add governed discovery plan expansion"
```

---

### Task 8: Discovery budgets and validated configuration

**Files:**
- Modify: `backend/app/services/agents/v2/contracts/planning.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `backend/tests/agents/v2/contracts/factories.py`
- Test: `backend/tests/agents/v2/contracts/test_discovery_budgets.py`

**Interfaces:**
- Consumes: `DiscoveryProbeTaskOrigin`, `TaskPlan`, `factual_tasks` (Task 7).
- Produces:
  - `DiscoveryBudgetView(probes_remaining, rounds_remaining, top_k)`
  - `build_research_budget_view(plan, *, max_tasks, max_replans, max_parallel_branches) -> ResearchBudgetView`
  - Settings `V2_DISCOVERY_*` and `V2_TOTAL_MAX_TASKS`; `V2_MAX_TASKS` is reused for research.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/agents/v2/contracts/test_discovery_budgets.py
"""Factual-task counting excludes bootstrap probes; settings validate."""
from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.agents.v2.contracts.planning import build_research_budget_view
from tests.agents.v2.contracts.factories import bootstrap_plan, scoped_document, binding_set, plan_with_origin
from app.services.agents.v2.contracts.planning import InitialTaskOrigin


def test_probe_tasks_do_not_consume_the_research_budget() -> None:
    plan = bootstrap_plan()

    view = build_research_budget_view(
        plan, max_tasks=8, max_replans=2, max_parallel_branches=3
    )

    assert view.max_tasks_remaining == 8


def test_factual_tasks_consume_the_research_budget() -> None:
    plan = plan_with_origin(InitialTaskOrigin(kind="initial"))

    view = build_research_budget_view(
        plan, max_tasks=8, max_replans=2, max_parallel_branches=3
    )

    assert view.max_tasks_remaining == 7


def test_total_cap_must_cover_discovery_plus_research() -> None:
    with pytest.raises(ValueError):
        Settings(V2_DISCOVERY_MAX_PROBES=5, V2_MAX_TASKS=8, V2_TOTAL_MAX_TASKS=10)


def test_thresholds_must_be_probabilities() -> None:
    with pytest.raises(ValueError):
        Settings(V2_DISCOVERY_CONFIDENCE_THRESHOLD=1.5)


def test_summary_min_cannot_exceed_max() -> None:
    with pytest.raises(ValueError):
        Settings(V2_DISCOVERY_SUMMARY_TARGETS=5, V2_DISCOVERY_MAX_SUMMARY_TARGETS=3)


def test_required_discovery_settings_have_defaults() -> None:
    settings = Settings()

    assert settings.V2_DISCOVERY_BOOTSTRAP_ENABLED is False
    assert settings.V2_DISCOVERY_MAX_PROBES == 5
    assert settings.V2_DISCOVERY_MAX_ROUNDS == 2
    assert settings.V2_DISCOVERY_TOP_K == 5
    assert settings.V2_DISCOVERY_DEADLINE_SECONDS == 15
    assert settings.V2_DISCOVERY_CONFIDENCE_THRESHOLD == pytest.approx(0.82)
    assert settings.V2_DISCOVERY_MARGIN_THRESHOLD == pytest.approx(0.12)
    assert settings.V2_TOTAL_MAX_TASKS == 13
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_discovery_budgets.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_research_budget_view'` / `ValidationError: Extra inputs are not permitted`.

- [ ] **Step 3: Add the budget view**

In `contracts/planning.py`:

```python
class DiscoveryBudgetView(ContractModel):
    """Spec §11.3: ephemeral discovery budget; never persisted."""

    probes_remaining: int
    rounds_remaining: int
    top_k: int


def build_research_budget_view(
    plan: TaskPlan,
    *,
    max_tasks: int,
    max_replans: int,
    max_parallel_branches: int,
) -> ResearchBudgetView:
    """Spec §11.3: count factual tasks only; bootstrap probes are free."""

    factual = sum(
        1 for task in plan.tasks if not isinstance(task.origin, DiscoveryProbeTaskOrigin)
    )
    return ResearchBudgetView(
        max_tasks_remaining=max(0, max_tasks - factual),
        max_replans_remaining=max_replans,
        max_parallel_branches=max_parallel_branches,
    )
```

`DiscoveryProbeTaskOrigin` is defined in this same module (Task 4), so no import is needed.

- [ ] **Step 4: Add and validate the settings**

In `backend/app/core/config.py`, next to the existing `V2_MAX_TASKS: int = Field(default=8)`:

```python
    V2_DISCOVERY_BOOTSTRAP_ENABLED: bool = Field(default=False)
    V2_DISCOVERY_MAX_PROBES: int = Field(default=5, gt=0)
    V2_DISCOVERY_MAX_ROUNDS: int = Field(default=2, gt=0)
    V2_DISCOVERY_TOP_K: int = Field(default=5, gt=0)
    V2_DISCOVERY_DEADLINE_SECONDS: int = Field(default=15, gt=0)
    V2_DISCOVERY_SUMMARY_TARGETS: int = Field(default=3, gt=0)
    V2_DISCOVERY_MAX_SUMMARY_TARGETS: int = Field(default=5, gt=0)
    V2_DISCOVERY_CONFIDENCE_THRESHOLD: float = Field(default=0.82, ge=0.0, le=1.0)
    V2_DISCOVERY_MARGIN_THRESHOLD: float = Field(default=0.12, ge=0.0, le=1.0)
    V2_DISCOVERY_CALIBRATION_ARTIFACT: str = Field(
        default="/app/config/discovery-calibration.json"
    )
    V2_TOTAL_MAX_TASKS: int = Field(default=13, gt=0)
```

Extend the existing `@model_validator(mode="after")` on `Settings` (present at `config.py:694`) with:

```python
        if self.V2_DISCOVERY_SUMMARY_TARGETS > self.V2_DISCOVERY_MAX_SUMMARY_TARGETS:
            raise ValueError(
                "V2_DISCOVERY_SUMMARY_TARGETS cannot exceed V2_DISCOVERY_MAX_SUMMARY_TARGETS"
            )
        if self.V2_TOTAL_MAX_TASKS < self.V2_DISCOVERY_MAX_PROBES + self.V2_MAX_TASKS:
            raise ValueError(
                "V2_TOTAL_MAX_TASKS must be at least V2_DISCOVERY_MAX_PROBES + V2_MAX_TASKS"
            )
        return self
```

If the existing validator already ends with `return self`, extend the same function instead of adding a second one.

`.env.example` — add the same keys with their defaults beside the existing `V2_MAX_TASKS=8` line, each with a short comment naming the spec section.

- [ ] **Step 5: Run the tests**

Run: `cd backend && python -m pytest tests/agents/v2/contracts/test_discovery_budgets.py -q`
Expected: PASS.

Run: `cd backend && python -c "from app.core.config import Settings; s=Settings(); print(s.V2_DISCOVERY_TOP_K, s.V2_TOTAL_MAX_TASKS)"`
Expected: `5 13`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/agents/v2/contracts/planning.py \
        backend/app/core/config.py .env.example \
        backend/tests/agents/v2/contracts/test_discovery_budgets.py \
        backend/tests/agents/v2/contracts/factories.py
git commit -m "feat(v2): add discovery budgets and validated settings"
```

---

## Phase 1 Exit Criteria

- [ ] `cd backend && python -m pytest tests/agents/v2/contracts tests/agents/v2/complex tests/agents/v2/test_supervisor_v2.py -q` passes.
- [ ] `git diff --check` is clean.
- [ ] No graph node, route result, or capability was added; `V2_DISCOVERY_BOOTSTRAP_ENABLED` defaults to `false`.
- [ ] A revision-1 checkpoint loads and runs; a revision-2 checkpoint missing a required key fails closed.
- [ ] Unrelated worktree changes are still uncommitted.

## Later Plans (not written yet)

Each gets its own plan once Phase 1 lands, because each produces independently testable software:

| Plan | Spec phases | Produces |
|---|---|---|
| Phase 2 — identity search and calibration | §9, §10 | Exact-number/title/lexical/semantic probe pipeline, bounded rerank, calibration artifact loader and gates. |
| Phase 3 — named summary and compare bootstrap | §4, §14 | Slot derivation, reconciliation, conditional admission, probe execution, deterministic selection, same-document compare. |
| Phase 4 — binding settle and expansion wiring | §12, §13 | `selection_settle_node`, `BindingSelectionRequest`, ACL/pin/lease/provenance, `research_expand_node`. |
| Phase 5 — clarification suspend and resume | §15 | Public per-slot frame, suspend/resume round trip through streaming/transport/runner ingress. |
| Phase 6 — thematic summary | §14.2 | Three-to-five target map/reduce after earlier gates pass. |

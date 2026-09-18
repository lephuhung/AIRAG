"""Discovery plan Task 2 — durable research target selection validation."""
from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from app.services.agents.v2.contracts.binding import DocumentDiscoveryCandidate
from app.services.agents.v2.contracts.capability import (
    DocumentIdentityMatch,
    DocumentSearchInput,
    DocumentSearchOutput,
)
from app.services.agents.v2.contracts.locators import SectionLocator
from app.services.agents.v2.contracts.planning import (
    DiscoveryExpansionTaskOrigin,
    DiscoveryProbeTaskOrigin,
    InitialTaskOrigin,
    TaskSpec,
)
from app.services.agents.v2.contracts.routing import QueryAnalysis
from app.services.agents.v2.contracts.validation import ContractValidationError
from app.services.agents.v2.discovery_bootstrap.contracts import (
    DiscoveryCheckpoint,
    DiscoveryNeed,
    DiscoverySelection,
    SlotBindingSelection,
)
from app.services.agents.v2.discovery_bootstrap.validation import (
    aggregate_id_for,
    validate_discovery_need,
    validate_research_target_selection,
)

from . import factories

DISCOVERY_ID = uuid4()


def _discovery_checkpoint(
    *,
    target_slots: tuple | None = None,
    aggregate_id=None,
    aggregate_document_id=None,
    aggregate_revision: str = factories.REVISION,
    selection_authority: str = "confidence_policy",
) -> DiscoveryCheckpoint:
    if target_slots is None:
        target_slots = (factories.target_slot(),)
    aggregates = ()
    selections = ()
    if aggregate_id is not None:
        from app.services.agents.v2.discovery_bootstrap.contracts import SlotCandidateAggregate

        aggregates = (
            SlotCandidateAggregate(
                aggregate_id=aggregate_id,
                slot_id="slot-1",
                document_id=aggregate_document_id,
                document_revision=aggregate_revision,
                source_candidate_ids=(uuid4(),),
                source_probe_ids=("probe-1",),
                source_task_ids=("T-search",),
                best_match_kind="semantic",
                aggregate_confidence=0.9,
                best_rank=1,
            ),
        )
        selections = (
            DiscoverySelection(
                slot_id="slot-1",
                selected_aggregate_ids=(aggregate_id,),
                authority=selection_authority,  # type: ignore[arg-type]
            ),
        )
    return DiscoveryCheckpoint(
        discovery_id=DISCOVERY_ID,
        target_slots=target_slots,
        accepted_probes=(),
        candidate_matches=(),
        slot_aggregates=aggregates,
        selections=selections,
        rounds_consumed=0,
        probes_consumed=0,
        status="selected",
        clarification_manifest=None,
    )


def test_explicit_selection_valid_without_discovery() -> None:
    validate_research_target_selection(
        factories.target_selection(), factories.binding_set()
    )


def test_discovery_need_invariant() -> None:
    validate_discovery_need(DiscoveryNeed(required=False, reason=None))
    validate_discovery_need(
        DiscoveryNeed(required=True, reason="unresolved_document_slot")
    )
    with pytest.raises(ValidationError):
        DiscoveryNeed(required=True, reason=None)
    with pytest.raises(ValidationError):
        DiscoveryNeed(required=False, reason="unresolved_document_slot")


def test_unknown_slot_rejected() -> None:
    selection = factories.target_selection(
        slot_bindings=(
            SlotBindingSelection(
                slot_id="ghost",
                selections=(factories.selected_binding_ref(),),
            ),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_unknown_binding_rejected() -> None:
    selection = factories.target_selection(
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    factories.selected_binding_ref(binding_id="ghost"),
                ),
            ),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_duplicate_context_binding_rejected() -> None:
    selection = factories.target_selection(context_binding_ids=("b2", "b2"))
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(),
            factories.scoped_document(
                binding_id="b2", document_id=factories.OTHER_DOCUMENT_ID
            ),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, bindings)


def test_selected_context_overlap_rejected() -> None:
    selection = factories.target_selection(context_binding_ids=("b1",))
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_discovery_backed_selection_requires_aggregate_id() -> None:
    checkpoint = _discovery_checkpoint()
    selection = factories.target_selection(
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    factories.selected_binding_ref(
                        authority="confidence_policy",
                        selected_aggregate_id=None,
                    ),
                ),
            ),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(
            selection, factories.binding_set(), checkpoint
        )


def test_discovery_backed_selection_requires_checkpoint() -> None:
    selection = factories.target_selection(
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    factories.selected_binding_ref(
                        authority="confidence_policy",
                        selected_aggregate_id=uuid4(),
                    ),
                ),
            ),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_explicit_selection_with_aggregate_id_rejected() -> None:
    selection = factories.target_selection(
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    factories.selected_binding_ref(
                        authority="explicit_binding",
                        selected_aggregate_id=uuid4(),
                    ),
                ),
            ),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_discovery_backed_selection_matches_checkpoint() -> None:
    aggregate_id = aggregate_id_for(
        DISCOVERY_ID, "slot-1", factories.DOCUMENT_ID, factories.REVISION
    )
    checkpoint = _discovery_checkpoint(
        aggregate_id=aggregate_id, aggregate_document_id=factories.DOCUMENT_ID
    )
    selection = factories.target_selection(
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    factories.selected_binding_ref(
                        authority="confidence_policy",
                        selected_aggregate_id=aggregate_id,
                    ),
                ),
            ),
        )
    )
    validate_research_target_selection(
        selection, factories.binding_set(), checkpoint
    )


def test_checkpoint_selected_aggregate_cannot_be_dropped() -> None:
    slot = factories.target_slot(
        explicit_binding_ids=("b1",), max_selections=2
    )
    aggregate_id = aggregate_id_for(
        DISCOVERY_ID,
        "slot-1",
        factories.OTHER_DOCUMENT_ID,
        factories.REVISION,
    )
    checkpoint = _discovery_checkpoint(
        target_slots=(slot,),
        aggregate_id=aggregate_id,
        aggregate_document_id=factories.OTHER_DOCUMENT_ID,
    )
    selection = factories.target_selection(
        target_slots=(slot,),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(factories.selected_binding_ref(binding_id="b1"),),
            ),
        ),
    )
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(),
            factories.scoped_document(
                binding_id="b2", document_id=factories.OTHER_DOCUMENT_ID
            ),
        )
    )
    with pytest.raises(
        ContractValidationError, match="resolve every checkpoint-selected aggregate"
    ):
        validate_research_target_selection(selection, bindings, checkpoint)


def test_discovery_backed_foreign_aggregate_rejected() -> None:
    aggregate_id = aggregate_id_for(
        DISCOVERY_ID, "slot-1", factories.DOCUMENT_ID, factories.REVISION
    )
    checkpoint = _discovery_checkpoint(
        aggregate_id=aggregate_id,
        aggregate_document_id=factories.OTHER_DOCUMENT_ID,
    )
    selection = factories.target_selection(
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    factories.selected_binding_ref(
                        authority="confidence_policy",
                        selected_aggregate_id=aggregate_id,
                    ),
                ),
            ),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(
            selection, factories.binding_set(), checkpoint
        )


def test_selection_slots_must_equal_checkpoint_slots() -> None:
    checkpoint = _discovery_checkpoint(
        target_slots=(factories.target_slot(slot_id="other"),)
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(
            factories.target_selection(), factories.binding_set(), checkpoint
        )


def test_work_type_mismatch_rejected() -> None:
    analysis = QueryAnalysis(work_type="compare", domains=("document",))
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(
            factories.target_selection(),
            factories.binding_set(),
            query_analysis=analysis,
        )


def test_two_revisions_cannot_fill_multi_select_slot() -> None:
    slot = factories.target_slot(
        min_selections=2, max_selections=2, explicit_binding_ids=("b1", "b2")
    )
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(binding_id="b1"),
            factories.scoped_document(
                binding_id="b2", document_revision=factories.OTHER_REVISION
            ),
        )
    )
    selection = factories.target_selection(
        target_slots=(slot,),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    factories.selected_binding_ref(target_id="t1", binding_id="b1"),
                    factories.selected_binding_ref(target_id="t2", binding_id="b2"),
                ),
            ),
        ),
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, bindings)


def test_same_document_twice_in_slot_rejected_even_within_cardinality() -> None:
    slot = factories.target_slot(
        min_selections=1, max_selections=2, explicit_binding_ids=("b1", "b2")
    )
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(binding_id="b1"),
            factories.scoped_document(
                binding_id="b2", document_revision=factories.OTHER_REVISION
            ),
        )
    )
    selection = factories.target_selection(
        target_slots=(slot,),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    factories.selected_binding_ref(target_id="t1", binding_id="b1"),
                    factories.selected_binding_ref(target_id="t2", binding_id="b2"),
                ),
            ),
        ),
    )
    with pytest.raises(ContractValidationError, match="more than once"):
        validate_research_target_selection(selection, bindings)


def test_same_document_across_compare_slots_rejected_with_distinct_bindings() -> None:
    slots = (
        factories.target_slot(slot_id="side-1", explicit_binding_ids=("b1",)),
        factories.target_slot(
            slot_id="side-2",
            intended_role="reference",
            explicit_binding_ids=("b2",),
        ),
    )
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(binding_id="b1"),
            factories.scoped_document(binding_id="b2"),
        )
    )
    selection = factories.target_selection(
        work_type="compare",
        target_slots=slots,
        slot_bindings=(
            SlotBindingSelection(
                slot_id="side-1",
                selections=(factories.selected_binding_ref(target_id="t1"),),
            ),
            SlotBindingSelection(
                slot_id="side-2",
                selections=(
                    factories.selected_binding_ref(target_id="t2", binding_id="b2"),
                ),
            ),
        ),
    )
    with pytest.raises(ContractValidationError, match="distinct slot locators"):
        validate_research_target_selection(selection, bindings)


def test_same_document_compare_distinct_locators_allow_explicit_bindings() -> None:
    slots = (
        factories.target_slot(
            slot_id="side-1",
            explicit_binding_ids=("b1",),
            requested_locator=SectionLocator(
                kind="section", structure_node_id="node-1"
            ),
        ),
        factories.target_slot(
            slot_id="side-2",
            intended_role="reference",
            explicit_binding_ids=("b2",),
            requested_locator=SectionLocator(
                kind="section", structure_node_id="node-2"
            ),
        ),
    )
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(binding_id="b1"),
            factories.scoped_document(binding_id="b2"),
        )
    )
    selection = factories.target_selection(
        work_type="compare",
        target_slots=slots,
        slot_bindings=(
            SlotBindingSelection(
                slot_id="side-1",
                selections=(factories.selected_binding_ref(target_id="t1"),),
            ),
            SlotBindingSelection(
                slot_id="side-2",
                selections=(
                    factories.selected_binding_ref(target_id="t2", binding_id="b2"),
                ),
            ),
        ),
    )
    validate_research_target_selection(selection, bindings)


def test_same_binding_two_compare_slots_distinct_locators() -> None:
    slots = (
        factories.target_slot(
            slot_id="side-1",
            requested_locator=SectionLocator(
                kind="section", structure_node_id="node-1"
            ),
        ),
        factories.target_slot(
            slot_id="side-2",
            intended_role="reference",
            requested_locator=SectionLocator(
                kind="section", structure_node_id="node-2"
            ),
        ),
    )
    selection = factories.target_selection(
        work_type="compare",
        target_slots=slots,
        slot_bindings=(
            SlotBindingSelection(
                slot_id="side-1",
                selections=(factories.selected_binding_ref(target_id="t1"),),
            ),
            SlotBindingSelection(
                slot_id="side-2",
                selections=(factories.selected_binding_ref(target_id="t2"),),
            ),
        ),
    )
    validate_research_target_selection(selection, factories.binding_set())


def test_same_binding_two_slots_same_locator_rejected() -> None:
    slots = (
        factories.target_slot(slot_id="side-1"),
        factories.target_slot(slot_id="side-2", intended_role="reference"),
    )
    selection = factories.target_selection(
        work_type="compare",
        target_slots=slots,
        slot_bindings=(
            SlotBindingSelection(
                slot_id="side-1",
                selections=(factories.selected_binding_ref(target_id="t1"),),
            ),
            SlotBindingSelection(
                slot_id="side-2",
                selections=(factories.selected_binding_ref(target_id="t2"),),
            ),
        ),
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_same_binding_two_summarize_slots_rejected() -> None:
    slots = (
        factories.target_slot(
            slot_id="s1",
            requested_locator=SectionLocator(
                kind="section", structure_node_id="n1"
            ),
        ),
        factories.target_slot(
            slot_id="s2",
            requested_locator=SectionLocator(
                kind="section", structure_node_id="n2"
            ),
        ),
    )
    selection = factories.target_selection(
        target_slots=slots,
        slot_bindings=(
            SlotBindingSelection(
                slot_id="s1",
                selections=(factories.selected_binding_ref(target_id="t1"),),
            ),
            SlotBindingSelection(
                slot_id="s2",
                selections=(factories.selected_binding_ref(target_id="t2"),),
            ),
        ),
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


@pytest.mark.parametrize("target_id", ["", "  "])
def test_blank_target_id_rejected(target_id: str) -> None:
    selection = factories.target_selection(
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(factories.selected_binding_ref(target_id=target_id),),
            ),
        )
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_duplicate_target_id_across_slots_rejected() -> None:
    slots = (
        factories.target_slot(slot_id="s1", required=False, min_selections=0),
        factories.target_slot(
            slot_id="s2",
            intended_role="reference",
            explicit_binding_ids=("b2",),
        ),
    )
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(),
            factories.scoped_document(
                binding_id="b2", document_id=factories.OTHER_DOCUMENT_ID
            ),
        )
    )
    selection = factories.target_selection(
        target_slots=slots,
        slot_bindings=(
            SlotBindingSelection(
                slot_id="s1",
                selections=(factories.selected_binding_ref(target_id="t1"),),
            ),
            SlotBindingSelection(
                slot_id="s2",
                selections=(
                    factories.selected_binding_ref(
                        target_id="t1", binding_id="b2"
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, bindings)


def test_required_slot_needs_a_selection() -> None:
    selection = factories.target_selection(slot_bindings=())
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_explicit_binding_must_be_listed_by_owning_slot() -> None:
    slot = factories.target_slot(explicit_binding_ids=())
    selection = factories.target_selection(target_slots=(slot,))
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_empty_target_slots_rejected() -> None:
    selection = factories.target_selection(target_slots=(), slot_bindings=())
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_blank_subject_hint_rejected() -> None:
    slot = factories.target_slot(subject_hint="  ")
    selection = factories.target_selection(target_slots=(slot,))
    with pytest.raises(ContractValidationError):
        validate_research_target_selection(selection, factories.binding_set())


def test_new_task_origins_parse_strictly() -> None:
    adapter = TypeAdapter(TaskSpec)
    probe = adapter.validate_python(
        {
            "task_id": "T1",
            "capability": "document.search",
            "task_objective": "probe",
            "input": {"kind": "document.search", "query": "nghị định 13"},
            "depends_on": (),
            "origin": {
                "kind": "discovery_probe",
                "probe_id": "p1",
                "slot_id": "slot-1",
                "round": 1,
            },
        }
    )
    assert isinstance(probe.origin, DiscoveryProbeTaskOrigin)
    expansion = adapter.validate_python(
        {
            "task_id": "T2",
            "capability": "document.read",
            "task_objective": "read",
            "input": {"kind": "document.read", "target_ids": ("t1",)},
            "depends_on": ("T1",),
            "origin": {
                "kind": "discovery_expansion",
                "source_search_task_ids": ("T1",),
                "target_slot_ids": ("slot-1",),
                "selected_aggregate_ids": (uuid4(),),
            },
        }
    )
    assert isinstance(expansion.origin, DiscoveryExpansionTaskOrigin)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "task_id": "T3",
                "capability": "document.search",
                "task_objective": "probe",
                "input": {"kind": "document.search", "query": "x"},
                "depends_on": (),
                "origin": {"kind": "discovery_probe", "probe_id": "p"},
            }
        )
    assert isinstance(
        factories.read_task().origin, InitialTaskOrigin
    )


def test_document_search_output_identity_matches_round_trip() -> None:
    candidate_id = uuid4()
    output = DocumentSearchOutput(
        kind="document.search",
        candidates=(
            DocumentDiscoveryCandidate(
                candidate_id=candidate_id,
                document_id=factories.DOCUMENT_ID,
                document_revision="rev-1",
            ),
        ),
        identity_matches=(
            DocumentIdentityMatch(
                candidate_id=candidate_id,
                document_id=factories.DOCUMENT_ID,
                document_revision="rev-1",
                rank=1,
                confidence=0.9,
                match_kind="semantic",
                calibration_version="cal-1",
            ),
        ),
    )
    payload = output.model_dump(mode="python")
    restored = DocumentSearchOutput.model_validate(payload)
    assert restored == output
    assert restored.identity_matches[0].match_kind == "semantic"


def test_generic_search_output_defaults_empty_identity_matches() -> None:
    output = DocumentSearchOutput(
        kind="document.search",
        candidates=(),
    )
    assert output.identity_matches == ()


def test_discovery_checkpoint_model_round_trip() -> None:
    checkpoint = _discovery_checkpoint()
    restored = DiscoveryCheckpoint.model_validate(
        checkpoint.model_dump(mode="python")
    )
    assert restored == checkpoint


def test_search_input_unchanged() -> None:
    search = DocumentSearchInput(kind="document.search", query="x")
    assert search.person_identifier is None

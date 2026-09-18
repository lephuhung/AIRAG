"""Discovery plan Task 3 — aggregate identity, reconstruction, ordering, margin."""
from __future__ import annotations

import math
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.contracts.validation import ContractValidationError
from app.services.agents.v2.discovery_bootstrap.contracts import (
    ProbeCandidateMatch,
    SearchProbe,
    SlotCandidateAggregate,
)
from app.services.agents.v2.discovery_bootstrap.validation import (
    aggregate_id_for,
    aggregate_matches,
    selection_margin,
)

from . import factories

DISCOVERY_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
OTHER_DISCOVERY_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _probe(probe_id: str = "p1", slot_id: str = "slot-1", round_: int = 1) -> SearchProbe:
    return SearchProbe(
        probe_id=probe_id,
        slot_id=slot_id,
        query="nghị định 13",
        round=round_,  # type: ignore[arg-type]
        origin="semantic",
    )


def _match(
    *,
    probe_id: str = "p1",
    slot_id: str = "slot-1",
    task_id: str = "T-search",
    candidate_id: UUID | None = None,
    document_id: UUID = factories.DOCUMENT_ID,
    document_revision: str = factories.REVISION,
    rank: int = 1,
    confidence: float | None = 0.9,
    match_kind: str = "semantic",
    calibration_version: str | None = "cal-1",
) -> ProbeCandidateMatch:
    return ProbeCandidateMatch(
        probe_id=probe_id,
        slot_id=slot_id,
        task_id=task_id,
        candidate_id=candidate_id or uuid4(),
        document_id=document_id,
        document_revision=document_revision,
        rank=rank,
        confidence=confidence,
        match_kind=match_kind,  # type: ignore[arg-type]
        calibration_version=calibration_version,
    )


def _aggregate(
    *,
    slot_id: str = "slot-1",
    document_id: UUID = factories.DOCUMENT_ID,
    document_revision: str = factories.REVISION,
    best_match_kind: str = "semantic",
    aggregate_confidence: float | None = 0.9,
    best_rank: int = 1,
    discovery_id: UUID = DISCOVERY_ID,
) -> SlotCandidateAggregate:
    return SlotCandidateAggregate(
        aggregate_id=aggregate_id_for(
            discovery_id, slot_id, document_id, document_revision
        ),
        slot_id=slot_id,
        document_id=document_id,
        document_revision=document_revision,
        source_candidate_ids=(uuid4(),),
        source_probe_ids=("p1",),
        source_task_ids=("T-search",),
        best_match_kind=best_match_kind,  # type: ignore[arg-type]
        aggregate_confidence=aggregate_confidence,
        best_rank=best_rank,
    )


def test_aggregate_id_deterministic_within_lineage() -> None:
    first = aggregate_id_for(
        DISCOVERY_ID, "slot-1", factories.DOCUMENT_ID, "rev-1"
    )
    second = aggregate_id_for(
        DISCOVERY_ID, "slot-1", factories.DOCUMENT_ID, "rev-1"
    )
    assert first == second


def test_aggregate_id_separated_across_runs() -> None:
    assert aggregate_id_for(
        DISCOVERY_ID, "slot-1", factories.DOCUMENT_ID, "rev-1"
    ) != aggregate_id_for(
        OTHER_DISCOVERY_ID, "slot-1", factories.DOCUMENT_ID, "rev-1"
    )


def test_aggregate_groups_one_identity_per_slot() -> None:
    matches = (
        _match(candidate_id=UUID("00000000-0000-0000-0000-000000000001"), rank=2),
        _match(
            probe_id="p2",
            candidate_id=UUID("00000000-0000-0000-0000-000000000002"),
            rank=1,
            confidence=0.8,
        ),
    )
    probes = (_probe(), _probe("p2"))
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate.document_id == factories.DOCUMENT_ID
    assert aggregate.best_rank == 1
    assert aggregate.aggregate_confidence == pytest.approx(0.9)
    assert set(aggregate.source_probe_ids) == {"p1", "p2"}
    assert len(aggregate.source_candidate_ids) == 2
    assert aggregate.source_task_ids == ("T-search",)
    assert aggregate.aggregate_id == aggregate_id_for(
        DISCOVERY_ID, "slot-1", factories.DOCUMENT_ID, factories.REVISION
    )


def test_revisions_aggregate_separately() -> None:
    matches = (
        _match(),
        _match(document_revision=factories.OTHER_REVISION, rank=2),
    )
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, (_probe(),), (factories.target_slot(),)
    )
    assert len(aggregates) == 2


def test_exact_match_beats_nonexact_in_order_and_drops_confidence() -> None:
    matches = (
        _match(rank=1, confidence=0.99, match_kind="semantic"),
        _match(
            document_id=factories.OTHER_DOCUMENT_ID,
            rank=3,
            confidence=None,
            match_kind="exact_document_number",
            calibration_version=None,
        ),
    )
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, (_probe(),), (factories.target_slot(),)
    )
    assert aggregates[0].document_id == factories.OTHER_DOCUMENT_ID
    assert aggregates[0].aggregate_confidence is None
    assert aggregates[1].document_id == factories.DOCUMENT_ID


def test_nonexact_orders_confidence_descending_before_rank() -> None:
    low_rank_high_conf = _match(
        document_id=factories.OTHER_DOCUMENT_ID, rank=4, confidence=0.95
    )
    high_rank_low_conf = _match(rank=1, confidence=0.7)
    aggregates = aggregate_matches(
        DISCOVERY_ID,
        (high_rank_low_conf, low_rank_high_conf),
        (_probe(),),
        (factories.target_slot(),),
    )
    assert aggregates[0].document_id == factories.OTHER_DOCUMENT_ID
    assert aggregates[1].document_id == factories.DOCUMENT_ID


def test_single_select_margin_top_minus_runner_up() -> None:
    slot = factories.target_slot()
    ordered = (
        _aggregate(aggregate_confidence=0.9),
        _aggregate(
            document_id=factories.OTHER_DOCUMENT_ID, aggregate_confidence=0.7
        ),
    )
    margin = selection_margin(
        ordered, (ordered[0].aggregate_id,), slot, "confidence_policy"
    )
    assert margin == pytest.approx(0.2)


def test_single_candidate_margin() -> None:
    slot = factories.target_slot()
    ordered = (_aggregate(aggregate_confidence=0.9),)
    margin = selection_margin(
        ordered, (ordered[0].aggregate_id,), slot, "confidence_policy"
    )
    assert margin == pytest.approx(0.9)


def test_non_top_policy_selection_rejected() -> None:
    slot = factories.target_slot()
    ordered = (
        _aggregate(aggregate_confidence=0.9),
        _aggregate(
            document_id=factories.OTHER_DOCUMENT_ID, aggregate_confidence=0.7
        ),
    )
    with pytest.raises(ContractValidationError):
        selection_margin(
            ordered, (ordered[1].aggregate_id,), slot, "confidence_policy"
        )
    margin = selection_margin(
        ordered, (ordered[1].aggregate_id,), slot, "user_choice"
    )
    assert margin == pytest.approx(0.7)


def test_multi_select_margin_uses_last_selected() -> None:
    slot = factories.target_slot(min_selections=3, max_selections=5)
    ordered = tuple(
        _aggregate(
            document_id=UUID(int=i + 1),
            aggregate_confidence=0.9 - i * 0.1,
        )
        for i in range(4)
    )
    selected = tuple(aggregate.aggregate_id for aggregate in ordered[:3])
    margin = selection_margin(ordered, selected, slot, "confidence_policy")
    assert margin == pytest.approx(0.7 - 0.6)


def test_saturated_five_of_six_margin() -> None:
    slot = factories.target_slot(min_selections=3, max_selections=5)
    confidences = (0.95, 0.9, 0.88, 0.86, 0.80, 0.79)
    ordered = tuple(
        _aggregate(document_id=UUID(int=i + 1), aggregate_confidence=c)
        for i, c in enumerate(confidences)
    )
    selected = tuple(aggregate.aggregate_id for aggregate in ordered[:5])
    margin = selection_margin(ordered, selected, slot, "confidence_policy")
    assert margin == pytest.approx(0.80 - 0.79)


def test_multi_select_without_excluded_passes() -> None:
    slot = factories.target_slot(min_selections=2, max_selections=5)
    ordered = (
        _aggregate(aggregate_confidence=0.9),
        _aggregate(
            document_id=factories.OTHER_DOCUMENT_ID, aggregate_confidence=0.85
        ),
    )
    selected = tuple(aggregate.aggregate_id for aggregate in ordered)
    assert selection_margin(ordered, selected, slot, "confidence_policy") == math.inf


def test_unknown_selected_id_rejected() -> None:
    slot = factories.target_slot()
    ordered = (_aggregate(),)
    with pytest.raises(ContractValidationError):
        selection_margin(ordered, (uuid4(),), slot, "user_choice")


def test_match_referencing_unknown_probe_rejected() -> None:
    with pytest.raises(ContractValidationError):
        aggregate_matches(
            DISCOVERY_ID,
            (_match(probe_id="ghost"),),
            (_probe(),),
            (factories.target_slot(),),
        )


def test_match_referencing_unknown_slot_rejected() -> None:
    with pytest.raises(ContractValidationError):
        aggregate_matches(
            DISCOVERY_ID,
            (_match(slot_id="ghost"),),
            (_probe(),),
            (factories.target_slot(),),
        )


def test_match_probe_slot_mismatch_rejected() -> None:
    slots = (
        factories.target_slot(),
        factories.target_slot(slot_id="s2", required=False, min_selections=0),
    )
    with pytest.raises(ContractValidationError):
        aggregate_matches(
            DISCOVERY_ID,
            (_match(slot_id="s2"),),
            (_probe(slot_id="slot-1"),),
            slots,
        )

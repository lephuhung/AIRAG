"""Discovery slot/selection/checkpoint validators (discovery spec §7, §8).

Imports only discovery models, the data-only capability/clarification types
they reference, plan/result contracts for reconstruction, and the cycle-free
``contracts/validation_support`` primitives — never ``contracts/validation``.
"""
from __future__ import annotations

import math
from uuid import UUID, uuid5

from ..contracts.binding import DocumentBindingSet, ScopedDocument
from ..contracts.capability import DocumentSearchInput, DocumentSearchOutput
from ..contracts.clarification import DocumentSelectionClarification
from ..contracts.execution import AgentResult
from ..contracts.planning import DiscoveryProbeTaskOrigin, TaskPlan
from ..contracts.routing import QueryAnalysis
from ..contracts.validation_support import (
    _fail,
    _require_non_blank,
    _require_unique,
)
from .contracts import (
    DiscoveryCheckpoint,
    DiscoveryNeed,
    ProbeCandidateMatch,
    ResearchTargetSelection,
    SearchProbe,
    SelectedBindingRef,
    SlotCandidateAggregate,
    TargetSlot,
)

_EXACT_MATCH_KINDS = frozenset({"exact_document_number", "exact_normalized_title"})
_MATCH_KIND_ORDER = {
    "exact_document_number": 0,
    "exact_normalized_title": 1,
    "lexical_title": 2,
    "semantic": 3,
}
_POLICY_AUTHORITIES = frozenset({"exact_match_policy", "confidence_policy"})


def _normalize_probe_query(query: str) -> str:
    return " ".join(query.split()).casefold()


def _require_finite_confidence(
    confidence: float | None,
    calibration_version: str | None,
    match_kind: str,
    field: str,
) -> None:
    if match_kind in _EXACT_MATCH_KINDS:
        if confidence is not None or calibration_version is not None:
            _fail(
                f"{field}: exact match kind {match_kind!r} requires confidence=None "
                "and calibration_version=None"
            )
        return
    if confidence is None:
        _fail(f"{field}: non-exact match kind {match_kind!r} requires a confidence")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        _fail(f"{field}: confidence must be finite and in [0, 1]")
    if calibration_version is None or not calibration_version.strip():
        _fail(f"{field}: non-exact match kind {match_kind!r} requires a calibration_version")


def validate_discovery_need(need: DiscoveryNeed) -> None:
    """Discovery spec §4.1: the observability flag mirrors its reason."""

    if need.required != (need.reason is not None):
        _fail("DiscoveryNeed.required must equal (reason is not None)")


def validate_target_slots(slots: tuple[TargetSlot, ...]) -> None:
    """Discovery spec §7: unique non-blank slot identities and sane cardinality."""

    if not slots:
        _fail("target_slots must contain at least one slot")
    _require_unique((slot.slot_id for slot in slots), "TargetSlot.slot_id")
    for slot in slots:
        _require_non_blank(slot.slot_id, "TargetSlot.slot_id")
        _require_non_blank(slot.subject_hint, "TargetSlot.subject_hint")
        _require_unique(slot.explicit_binding_ids, "TargetSlot.explicit_binding_ids")
        for binding_id in slot.explicit_binding_ids:
            _require_non_blank(binding_id, "TargetSlot.explicit_binding_ids")
        if slot.min_selections < 0 or slot.max_selections < 1:
            _fail(
                f"TargetSlot {slot.slot_id} has invalid cardinality "
                f"{slot.min_selections}/{slot.max_selections}"
            )
        if slot.min_selections > slot.max_selections:
            _fail(
                f"TargetSlot {slot.slot_id} min_selections exceeds max_selections"
            )
        if slot.required and slot.min_selections < 1:
            _fail(f"required TargetSlot {slot.slot_id} needs min_selections >= 1")


def _binding_by_id(bindings: DocumentBindingSet, binding_id: str) -> ScopedDocument:
    for binding in bindings.bindings:
        if binding.binding_id == binding_id:
            return binding
    _fail(f"selection references unknown binding {binding_id}")


def validate_research_target_selection(
    selection: ResearchTargetSelection,
    bindings: DocumentBindingSet,
    discovery_checkpoint: DiscoveryCheckpoint | None = None,
    query_analysis: QueryAnalysis | None = None,
) -> None:
    """Discovery spec §7: self-contained durable selection invariants."""

    validate_target_slots(selection.target_slots)
    role_counts = {"target": 0, "reference": 0}
    for slot in selection.target_slots:
        role_counts[slot.intended_role] += 1
    if selection.work_type == "summarize":
        if role_counts["target"] != 1:
            _fail("a summarize selection requires exactly one target slot")
    elif len(selection.target_slots) != 2 or role_counts != {"target": 1, "reference": 1}:
        _fail(
            "a compare selection requires exactly one target slot and one "
            "reference slot"
        )
    if query_analysis is not None and query_analysis.work_type != selection.work_type:
        _fail(
            f"ResearchTargetSelection.work_type {selection.work_type!r} does not match "
            f"QueryAnalysis.work_type {query_analysis.work_type!r}"
        )
    if (
        discovery_checkpoint is not None
        and selection.target_slots != discovery_checkpoint.target_slots
    ):
        _fail(
            "ResearchTargetSelection.target_slots must equal the checkpointed "
            "discovery target slots"
        )

    slot_by_id = {slot.slot_id: slot for slot in selection.target_slots}
    binding_by_id = {binding.binding_id: binding for binding in bindings.bindings}
    _require_unique(selection.context_binding_ids, "context_binding_ids")
    for binding_id in selection.context_binding_ids:
        if binding_id not in binding_by_id:
            _fail(f"context binding {binding_id} is not in the binding set")
    for slot in selection.target_slots:
        for binding_id in slot.explicit_binding_ids:
            if binding_id not in binding_by_id:
                _fail(
                    f"slot {slot.slot_id} lists explicit binding {binding_id} "
                    "absent from the binding set"
                )

    _require_unique(
        (slot_selection.slot_id for slot_selection in selection.slot_bindings),
        "SlotBindingSelection.slot_id",
    )
    selected_target_ids: list[str] = []
    selected_aggregates_by_slot: dict[str, set[UUID]] = {}
    document_refs: dict[UUID, list[tuple[str, SelectedBindingRef]]] = {}
    selected_binding_ids: set[str] = set()
    aggregates_by_id = (
        {aggregate.aggregate_id: aggregate for aggregate in discovery_checkpoint.slot_aggregates}
        if discovery_checkpoint is not None
        else {}
    )
    selections_by_slot = (
        {sel.slot_id: sel for sel in discovery_checkpoint.selections}
        if discovery_checkpoint is not None
        else {}
    )

    for slot_selection in selection.slot_bindings:
        slot = slot_by_id.get(slot_selection.slot_id)
        if slot is None:
            _fail(f"slot selection references unknown slot {slot_selection.slot_id}")
        _require_unique(
            (ref.binding_id for ref in slot_selection.selections),
            f"SlotBindingSelection {slot.slot_id} binding_id",
        )
        distinct_documents: set[UUID] = set()
        for ref in slot_selection.selections:
            _require_non_blank(ref.target_id, "SelectedBindingRef.target_id")
            selected_target_ids.append(ref.target_id)
            binding = binding_by_id.get(ref.binding_id)
            if binding is None:
                _fail(
                    f"slot {slot.slot_id} selects unknown binding {ref.binding_id}"
                )
            if binding.document_id in distinct_documents:
                _fail(
                    f"slot {slot.slot_id} selects document {binding.document_id} "
                    "more than once"
                )
            selected_binding_ids.add(ref.binding_id)
            document_refs.setdefault(binding.document_id, []).append(
                (slot.slot_id, ref)
            )
            distinct_documents.add(binding.document_id)
            if ref.authority == "explicit_binding":
                if ref.selected_aggregate_id is not None:
                    _fail(
                        f"explicit-binding selection {ref.target_id} must not carry "
                        "a selected_aggregate_id"
                    )
                if ref.binding_id not in slot.explicit_binding_ids:
                    _fail(
                        f"explicit-binding selection {ref.target_id} names binding "
                        f"{ref.binding_id} not listed by slot {slot.slot_id}"
                    )
                continue
            if ref.selected_aggregate_id is None:
                _fail(
                    f"discovery-backed selection {ref.target_id} requires a "
                    "selected_aggregate_id"
                )
            if discovery_checkpoint is None:
                _fail(
                    f"discovery-backed selection {ref.target_id} requires a "
                    "DiscoveryCheckpoint"
                )
            aggregate = aggregates_by_id.get(ref.selected_aggregate_id)
            if (
                aggregate is None
                or aggregate.slot_id != slot.slot_id
                or aggregate.document_id != binding.document_id
                or aggregate.document_revision != binding.document_revision
            ):
                _fail(
                    f"selected aggregate {ref.selected_aggregate_id} does not match "
                    f"slot {slot.slot_id} and binding {ref.binding_id} identity"
                )
            discovery_selection = selections_by_slot.get(slot.slot_id)
            if (
                discovery_selection is None
                or ref.selected_aggregate_id not in discovery_selection.selected_aggregate_ids
                or discovery_selection.authority != ref.authority
            ):
                _fail(
                    f"selected aggregate {ref.selected_aggregate_id} is not recorded "
                    f"in the checkpointed selections for slot {slot.slot_id} "
                    f"with authority {ref.authority!r}"
                )
            selected_aggregates_by_slot.setdefault(slot.slot_id, set()).add(
                ref.selected_aggregate_id
            )
        count = len(distinct_documents)
        if not slot.min_selections <= count <= slot.max_selections:
            _fail(
                f"slot {slot.slot_id} selects {count} distinct document(s) but "
                f"requires {slot.min_selections}..{slot.max_selections}"
            )

    _require_unique(selected_target_ids, "SelectedBindingRef.target_id")
    overlap = selected_binding_ids & set(selection.context_binding_ids)
    if overlap:
        _fail(f"binding(s) cannot be both selected and context: {sorted(overlap)}")

    for document_id, refs in document_refs.items():
        slot_ids = {slot_id for slot_id, _ in refs}
        if len(slot_ids) <= 1:
            continue
        locators = {slot_by_id[slot_id].requested_locator for slot_id in slot_ids}
        explicit_compare = all(
            ref.authority == "explicit_binding" for _, ref in refs
        )
        if (
            selection.work_type != "compare"
            or len(slot_ids) != 2
            or len(locators) != 2
            or not explicit_compare
        ):
            _fail(
                f"document {document_id} may fill two slots only for an "
                "explicit compare selection with distinct slot locators"
            )

    for slot in selection.target_slots:
        if slot.required and not any(
            slot_selection.slot_id == slot.slot_id
            for slot_selection in selection.slot_bindings
        ):
            _fail(f"required slot {slot.slot_id} has no slot binding selection")

    if discovery_checkpoint is not None:
        for discovery_selection in discovery_checkpoint.selections:
            if discovery_selection.slot_id not in slot_by_id:
                _fail(
                    f"checkpoint selection references unknown slot "
                    f"{discovery_selection.slot_id}"
                )
            represented = selected_aggregates_by_slot.get(
                discovery_selection.slot_id, set()
            )
            if represented != set(discovery_selection.selected_aggregate_ids):
                _fail(
                    f"selection for slot {discovery_selection.slot_id} does not "
                    "resolve every checkpoint-selected aggregate"
                )


def aggregate_id_for(
    discovery_id: UUID,
    slot_id: str,
    document_id: UUID,
    document_revision: str,
) -> UUID:
    """Discovery spec §3.3: deterministic discovery-scoped aggregate identity."""

    return uuid5(discovery_id, f"{slot_id}:{document_id}:{document_revision}")


def _aggregate_sort_key(
    aggregate: SlotCandidateAggregate,
    slot_order: dict[str, int],
    probe_order: dict[str, int],
) -> tuple[object, ...]:
    first_probe = min(
        (probe_order[probe_id] for probe_id in aggregate.source_probe_ids),
        default=len(probe_order),
    )
    confidence = (
        aggregate.aggregate_confidence
        if aggregate.best_match_kind not in _EXACT_MATCH_KINDS
        else 0.0
    )
    return (
        slot_order.get(aggregate.slot_id, len(slot_order)),
        _MATCH_KIND_ORDER[aggregate.best_match_kind],
        -(confidence or 0.0),
        aggregate.best_rank,
        first_probe,
        str(aggregate.document_id),
        aggregate.document_revision,
    )


def aggregate_matches(
    discovery_id: UUID,
    matches: tuple[ProbeCandidateMatch, ...],
    accepted_probes: tuple[SearchProbe, ...],
    slots: tuple[TargetSlot, ...],
) -> tuple[SlotCandidateAggregate, ...]:
    """Discovery spec §8.2: one aggregate per document identity per slot."""

    probe_order = {probe.probe_id: index for index, probe in enumerate(accepted_probes)}
    probe_by_id = {probe.probe_id: probe for probe in accepted_probes}
    slot_order = {slot.slot_id: index for index, slot in enumerate(slots)}
    groups: dict[tuple[str, UUID, str], list[ProbeCandidateMatch]] = {}
    for match in matches:
        probe = probe_by_id.get(match.probe_id)
        if probe is None:
            _fail(f"candidate match references unknown probe {match.probe_id}")
        if match.slot_id not in slot_order:
            _fail(f"candidate match references unknown slot {match.slot_id}")
        if probe.slot_id != match.slot_id:
            _fail(
                f"candidate match slot {match.slot_id} does not match probe "
                f"{match.probe_id} slot {probe.slot_id}"
            )
    for match in matches:
        groups.setdefault(
            (match.slot_id, match.document_id, match.document_revision), []
        ).append(match)

    aggregates: list[SlotCandidateAggregate] = []
    for (slot_id, document_id, document_revision), members in groups.items():
        members.sort(
            key=lambda match: (
                probe_order.get(match.probe_id, len(probe_order)),
                match.rank,
                str(match.candidate_id),
            )
        )
        best_match_kind = min(
            (match.match_kind for match in members),
            key=lambda kind: _MATCH_KIND_ORDER[kind],
        )
        if best_match_kind in _EXACT_MATCH_KINDS:
            aggregate_confidence = None
        else:
            aggregate_confidence = max(
                (
                    match.confidence
                    for match in members
                    if match.confidence is not None
                ),
                default=None,
            )
        aggregates.append(
            SlotCandidateAggregate(
                aggregate_id=aggregate_id_for(
                    discovery_id, slot_id, document_id, document_revision
                ),
                slot_id=slot_id,
                document_id=document_id,
                document_revision=document_revision,
                source_candidate_ids=tuple(
                    dict.fromkeys(match.candidate_id for match in members)
                ),
                source_probe_ids=tuple(
                    dict.fromkeys(match.probe_id for match in members)
                ),
                source_task_ids=tuple(
                    dict.fromkeys(match.task_id for match in members)
                ),
                best_match_kind=best_match_kind,
                aggregate_confidence=aggregate_confidence,
                best_rank=min(match.rank for match in members),
            )
        )
    aggregates.sort(
        key=lambda aggregate: _aggregate_sort_key(aggregate, slot_order, probe_order)
    )
    return tuple(aggregates)


def _aggregate_confidence(aggregate: SlotCandidateAggregate) -> float:
    if aggregate.best_match_kind in _EXACT_MATCH_KINDS:
        return 1.0
    return aggregate.aggregate_confidence or 0.0


def selection_margin(
    ordered: tuple[SlotCandidateAggregate, ...],
    selected_ids: tuple[UUID, ...],
    slot: TargetSlot,
    authority: str,
) -> float:
    """Discovery spec §8.2: confidence margin at the actual selection boundary."""

    slot_ordered = [
        aggregate for aggregate in ordered if aggregate.slot_id == slot.slot_id
    ]
    if not slot_ordered:
        _fail(f"slot {slot.slot_id} has no candidate aggregates")
    positions: list[int] = []
    for selected_id in selected_ids:
        for index, aggregate in enumerate(slot_ordered):
            if aggregate.aggregate_id == selected_id:
                positions.append(index)
                break
        else:
            _fail(
                f"selected aggregate {selected_id} is not offered in slot "
                f"{slot.slot_id}"
            )
    if slot.min_selections <= 1:
        if positions != [0] and authority != "user_choice":
            _fail(
                f"single-select slot {slot.slot_id} requires the top-ranked "
                "aggregate unless authority is user_choice"
            )
        boundary = positions[0] if authority == "user_choice" else 0
        runner_up = slot_ordered[boundary + 1] if boundary + 1 < len(slot_ordered) else None
        return _aggregate_confidence(slot_ordered[boundary]) - (
            _aggregate_confidence(runner_up) if runner_up is not None else 0.0
        )
    boundary = max(positions)
    excluded = next(
        (
            aggregate
            for aggregate in slot_ordered[boundary + 1 :]
            if aggregate.aggregate_id not in selected_ids
        ),
        None,
    )
    if excluded is None:
        return math.inf
    return _aggregate_confidence(slot_ordered[boundary]) - _aggregate_confidence(
        excluded
    )


def _validate_accepted_probes(
    checkpoint: DiscoveryCheckpoint,
    *,
    max_probes: int,
    max_rounds: int,
) -> None:
    slot_ids = {slot.slot_id for slot in checkpoint.target_slots}
    _require_unique(
        (probe.probe_id for probe in checkpoint.accepted_probes), "SearchProbe.probe_id"
    )
    normalized_queries: set[str] = set()
    for probe in checkpoint.accepted_probes:
        _require_non_blank(probe.probe_id, "SearchProbe.probe_id")
        _require_non_blank(probe.query, "SearchProbe.query")
        if probe.slot_id not in slot_ids:
            _fail(f"probe {probe.probe_id} references unknown slot {probe.slot_id}")
        if probe.round > max_rounds:
            _fail(f"probe {probe.probe_id} exceeds the round budget")
        normalized = _normalize_probe_query(probe.query)
        if normalized in normalized_queries:
            _fail(
                f"probe {probe.probe_id} repeats a normalized-equivalent query"
            )
        normalized_queries.add(normalized)
    if len(checkpoint.accepted_probes) > max_probes:
        _fail("accepted probes exceed the probe budget")
    if checkpoint.probes_consumed != len(checkpoint.accepted_probes):
        _fail("probes_consumed must equal the number of accepted probes")
    expected_rounds = max(
        (probe.round for probe in checkpoint.accepted_probes), default=0
    )
    if checkpoint.rounds_consumed != expected_rounds:
        _fail("rounds_consumed must equal the highest accepted probe round")


def _reconstruct_candidate_matches(
    checkpoint: DiscoveryCheckpoint,
    plan: TaskPlan | None,
    task_results: tuple[AgentResult, ...],
) -> tuple[ProbeCandidateMatch, ...]:
    results_by_task = {result.task_id: result for result in task_results}
    probe_task_ids: dict[str, str] = {}
    if plan is not None:
        for task in plan.tasks:
            origin = task.origin
            if not isinstance(origin, DiscoveryProbeTaskOrigin):
                continue
            if origin.probe_id in probe_task_ids:
                _fail(
                    f"probe {origin.probe_id} is owned by more than one plan task"
                )
            probe_task_ids[origin.probe_id] = task.task_id

    expected: list[ProbeCandidateMatch] = []
    probe_by_id = {probe.probe_id: probe for probe in checkpoint.accepted_probes}
    for probe in checkpoint.accepted_probes:
        task_id = probe_task_ids.get(probe.probe_id)
        if task_id is None:
            _fail(
                f"accepted probe {probe.probe_id} has no probe-origin task in the plan"
            )
        task = next(task for task in plan.tasks if task.task_id == task_id)
        origin = task.origin
        if (
            not isinstance(origin, DiscoveryProbeTaskOrigin)
            or origin.slot_id != probe.slot_id
            or origin.round != probe.round
            or task.capability != "document.search"
            or not isinstance(task.input, DocumentSearchInput)
            or _normalize_probe_query(task.input.query) != _normalize_probe_query(probe.query)
        ):
            _fail(
                f"probe-origin task {task_id} does not exactly match probe "
                f"{probe.probe_id}"
            )
        result = results_by_task.get(task_id)
        if result is None or not isinstance(result.data, DocumentSearchOutput):
            continue
        output = result.data
        _require_unique(
            (candidate.candidate_id for candidate in output.candidates),
            f"task {task_id} candidates.candidate_id",
        )
        _require_unique(
            (match.candidate_id for match in output.identity_matches),
            f"task {task_id} identity_matches.candidate_id",
        )
        candidates = {candidate.candidate_id: candidate for candidate in output.candidates}
        if set(candidates) != {
            match.candidate_id for match in output.identity_matches
        }:
            _fail(
                f"task {task_id} must carry exactly one identity match per "
                "returned candidate"
            )
        for match in output.identity_matches:
            candidate = candidates[match.candidate_id]
            if (
                candidate.document_id != match.document_id
                or candidate.document_revision != match.document_revision
            ):
                _fail(
                    f"identity match {match.candidate_id} in task {task_id} does "
                    "not correspond to a returned candidate"
                )
            expected.append(
                ProbeCandidateMatch(
                    probe_id=probe.probe_id,
                    slot_id=probe.slot_id,
                    task_id=task_id,
                    candidate_id=match.candidate_id,
                    document_id=match.document_id,
                    document_revision=match.document_revision,
                    rank=match.rank,
                    confidence=match.confidence,
                    match_kind=match.match_kind,
                    calibration_version=match.calibration_version,
                )
            )
    owned_probes = set(probe_task_ids)
    orphan = owned_probes - set(probe_by_id)
    if orphan:
        _fail(f"plan carries probe-origin tasks for unaccepted probes: {sorted(orphan)}")
    return tuple(expected)


def _validate_candidate_match(match: ProbeCandidateMatch, *, top_k: int) -> None:
    _require_non_blank(match.probe_id, "ProbeCandidateMatch.probe_id")
    _require_non_blank(match.task_id, "ProbeCandidateMatch.task_id")
    _require_non_blank(match.document_revision, "ProbeCandidateMatch.document_revision")
    if match.rank < 1 or match.rank > top_k:
        _fail(f"candidate match rank {match.rank} is outside 1..{top_k}")
    _require_finite_confidence(
        match.confidence,
        match.calibration_version,
        match.match_kind,
        "ProbeCandidateMatch",
    )


def _validate_selections(
    checkpoint: DiscoveryCheckpoint,
    ordered: tuple[SlotCandidateAggregate, ...],
) -> None:
    slot_by_id = {slot.slot_id: slot for slot in checkpoint.target_slots}
    _require_unique(
        (selection.slot_id for selection in checkpoint.selections),
        "DiscoverySelection.slot_id",
    )
    for selection in checkpoint.selections:
        slot = slot_by_id.get(selection.slot_id)
        if slot is None:
            _fail(f"selection references unknown slot {selection.slot_id}")
        _require_unique(
            selection.selected_aggregate_ids,
            f"DiscoverySelection {selection.slot_id} selected_aggregate_ids",
        )
        if not selection.selected_aggregate_ids:
            _fail(f"selection for slot {selection.slot_id} is empty")
        slot_aggregates = [
            aggregate for aggregate in ordered if aggregate.slot_id == slot.slot_id
        ]
        offered = {aggregate.aggregate_id for aggregate in slot_aggregates}
        for aggregate_id in selection.selected_aggregate_ids:
            if aggregate_id not in offered:
                _fail(
                    f"selection for slot {selection.slot_id} names aggregate "
                    f"{aggregate_id} not offered in that slot"
                )
        if not slot.min_selections <= len(selection.selected_aggregate_ids) <= slot.max_selections:
            _fail(
                f"selection for slot {selection.slot_id} violates slot cardinality"
            )
        exact_count = sum(
            1
            for aggregate in slot_aggregates
            if aggregate.best_match_kind in _EXACT_MATCH_KINDS
        )
        if (
            selection.authority == "exact_match_policy"
            and (
                exact_count > 1
                or any(
                    aggregate.best_match_kind not in _EXACT_MATCH_KINDS
                    for aggregate in slot_aggregates
                    if aggregate.aggregate_id in selection.selected_aggregate_ids
                )
            )
        ):
            _fail(
                f"exact-match selection for slot {selection.slot_id} requires a "
                "single exact candidate"
            )
        if selection.authority == "confidence_policy" and any(
            aggregate.best_match_kind in _EXACT_MATCH_KINDS
            for aggregate in slot_aggregates
            if aggregate.aggregate_id in selection.selected_aggregate_ids
        ):
            _fail(
                f"confidence-policy selection for slot {selection.slot_id} must "
                "not name exact matches"
            )
        if selection.authority in _POLICY_AUTHORITIES:
            prefix = tuple(
                aggregate.aggregate_id
                for aggregate in slot_aggregates[: len(selection.selected_aggregate_ids)]
            )
            if tuple(selection.selected_aggregate_ids) != prefix:
                _fail(
                    f"policy selection for slot {selection.slot_id} must be a "
                    "ranking prefix"
                )
        selection_margin(
            tuple(slot_aggregates),
            selection.selected_aggregate_ids,
            slot,
            selection.authority,
        )


def _validate_manifest(
    checkpoint: DiscoveryCheckpoint,
    clarification: DocumentSelectionClarification | None,
) -> None:
    manifest = checkpoint.clarification_manifest
    if manifest is None:
        if clarification is not None:
            _fail(
                "a sibling document-selection request requires a pending manifest"
            )
        return
    _require_non_blank(
        manifest.clarification_id, "DocumentSelectionManifest.clarification_id"
    )
    if manifest.expires_at.tzinfo is None or manifest.expires_at.utcoffset() is None:
        _fail("DocumentSelectionManifest.expires_at must be timezone-aware")
    _require_unique(
        (entry.choice_token_digest for entry in manifest.entries),
        "DocumentSelectionManifestEntry.choice_token_digest",
    )
    aggregates_by_id = {
        aggregate.aggregate_id: aggregate for aggregate in checkpoint.slot_aggregates
    }
    for entry in manifest.entries:
        _require_non_blank(
            entry.choice_token_digest,
            "DocumentSelectionManifestEntry.choice_token_digest",
        )
        _require_non_blank(
            entry.document_revision, "DocumentSelectionManifestEntry.document_revision"
        )
        aggregate = aggregates_by_id.get(entry.aggregate_id)
        if (
            aggregate is None
            or aggregate.slot_id != entry.slot_id
            or aggregate.document_id != entry.document_id
            or aggregate.document_revision != entry.document_revision
        ):
            _fail(
                f"manifest entry {entry.choice_token_digest!r} does not exactly "
                "match a checkpointed aggregate"
            )
    if manifest.status == "pending":
        if checkpoint.status != "clarification":
            _fail("a pending manifest requires status 'clarification'")
        if clarification is None:
            _fail(
                "a pending clarification manifest requires the sibling public request"
            )
        _validate_document_selection_request(clarification)
        if manifest.clarification_id != clarification.clarification_id:
            _fail("manifest and public request clarification_id differ")
        if manifest.expires_at != clarification.expires_at:
            _fail("manifest and public request expiry differ")
        request_slots = {slot.slot_id: slot for slot in clarification.slots}
        entry_counts: dict[str, int] = {}
        for entry in manifest.entries:
            entry_counts[entry.slot_id] = entry_counts.get(entry.slot_id, 0) + 1
        if set(entry_counts) != set(request_slots):
            _fail("manifest entries and public request must cover the same slots")
        for slot_id, count in entry_counts.items():
            if count != len(request_slots[slot_id].choices):
                _fail(
                    f"manifest entries for slot {slot_id} must equal its "
                    "public choice count"
                )
        return
    if manifest.status == "consumed":
        if clarification is not None:
            _fail("a consumed manifest requires no sibling public request")
        if checkpoint.status not in ("selected", "unavailable"):
            _fail(
                f"a consumed manifest cannot stand in status {checkpoint.status!r}"
            )
        return
    _fail(f"unknown manifest status {manifest.status!r}")


def _validate_document_selection_request(
    request: DocumentSelectionClarification,
) -> None:
    _require_non_blank(request.clarification_id, "DocumentSelectionClarification.clarification_id")
    _require_non_blank(request.question, "DocumentSelectionClarification.question")
    if request.expires_at.tzinfo is None or request.expires_at.utcoffset() is None:
        _fail("DocumentSelectionClarification.expires_at must be timezone-aware")
    if not request.slots:
        _fail("DocumentSelectionClarification.slots must not be empty")
    _require_unique(
        (slot.slot_id for slot in request.slots), "DocumentSelectionSlot.slot_id"
    )
    tokens: list[str] = []
    for slot in request.slots:
        _require_non_blank(slot.slot_id, "DocumentSelectionSlot.slot_id")
        _require_non_blank(slot.slot_label, "DocumentSelectionSlot.slot_label")
        if not slot.choices or len(slot.choices) > 3:
            _fail(
                f"selection slot {slot.slot_id} must offer 1..3 choices"
            )
        if not 1 <= slot.min_selections <= slot.max_selections <= len(slot.choices):
            _fail(
                f"selection slot {slot.slot_id} has invalid cardinality "
                f"{slot.min_selections}..{slot.max_selections} for "
                f"{len(slot.choices)} choices"
            )
        for choice in slot.choices:
            _require_non_blank(choice.choice_token, "DocumentSelectionChoice.choice_token")
            _require_non_blank(choice.title, "DocumentSelectionChoice.title")
            tokens.append(choice.choice_token)
    _require_unique(tokens, "DocumentSelectionChoice.choice_token")


def validate_discovery_checkpoint(
    checkpoint: DiscoveryCheckpoint,
    *,
    plan: TaskPlan | None,
    task_results: tuple[AgentResult, ...] = (),
    clarification: DocumentSelectionClarification | None = None,
    max_probes: int = 5,
    max_rounds: int = 2,
    top_k: int = 5,
) -> None:
    """Discovery spec §8.4: reconstruct and compare all derived state."""

    validate_target_slots(checkpoint.target_slots)
    _require_unique(
        (result.task_id for result in task_results), "AgentResult.task_id"
    )
    _validate_accepted_probes(
        checkpoint, max_probes=max_probes, max_rounds=max_rounds
    )
    if plan is None and (checkpoint.accepted_probes or checkpoint.candidate_matches):
        _fail("accepted probes or candidate matches require the checkpointed plan")
    expected_matches = _reconstruct_candidate_matches(
        checkpoint, plan, task_results
    )
    if checkpoint.candidate_matches != expected_matches:
        _fail(
            "candidate_matches cannot be reproduced from accepted probes and "
            "checkpointed task results"
        )
    for match in checkpoint.candidate_matches:
        _validate_candidate_match(match, top_k=top_k)
    expected_aggregates = aggregate_matches(
        checkpoint.discovery_id,
        checkpoint.candidate_matches,
        checkpoint.accepted_probes,
        checkpoint.target_slots,
    )
    if checkpoint.slot_aggregates != expected_aggregates:
        _fail("slot_aggregates cannot be reproduced from candidate matches")
    _validate_selections(checkpoint, expected_aggregates)

    if checkpoint.status == "clarification":
        if (
            checkpoint.clarification_manifest is None
            or checkpoint.clarification_manifest.status != "pending"
            or clarification is None
        ):
            _fail(
                "status 'clarification' requires a pending manifest and the "
                "sibling public request"
            )
    elif checkpoint.clarification_manifest is not None and (
        checkpoint.clarification_manifest.status != "consumed"
    ):
        _fail(
            f"status {checkpoint.status!r} cannot carry a "
            f"{checkpoint.clarification_manifest.status!r} manifest"
        )
    if checkpoint.status == "selected":
        required_slots = {
            slot.slot_id for slot in checkpoint.target_slots if slot.required
        }
        selected_slots = {selection.slot_id for selection in checkpoint.selections}
        if not checkpoint.selections or not required_slots <= selected_slots:
            _fail("status 'selected' requires selections for every required slot")
    if checkpoint.status in ("searching", "ranking", "unavailable") and (
        checkpoint.selections
    ):
        _fail(f"status {checkpoint.status!r} cannot carry selections")
    _validate_manifest(checkpoint, clarification)

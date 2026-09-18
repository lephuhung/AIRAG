"""Discovery plan Task 3 — checkpoint reconstruction and fail-closed validation."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.contracts.binding import DocumentDiscoveryCandidate
from app.services.agents.v2.contracts.capability import (
    DocumentIdentityMatch,
    DocumentSearchInput,
    DocumentSearchOutput,
)
from app.services.agents.v2.contracts.clarification import (
    DocumentSelectionChoice,
    DocumentSelectionClarification,
    DocumentSelectionManifest,
    DocumentSelectionManifestEntry,
    DocumentSelectionSlot,
)
from app.services.agents.v2.contracts.execution import AgentResult
from app.services.agents.v2.contracts.planning import (
    DiscoveryProbeTaskOrigin,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.validation import ContractValidationError
from app.services.agents.v2.discovery_bootstrap.contracts import (
    DiscoveryCheckpoint,
    DiscoverySelection,
    SearchProbe,
)
from app.services.agents.v2.discovery_bootstrap.validation import (
    aggregate_matches,
    validate_discovery_checkpoint,
)

from . import factories

DISCOVERY_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
EXPIRES = datetime(2026, 9, 18, tzinfo=UTC)


def _probe(probe_id: str = "p1", slot_id: str = "slot-1", query: str = "nđ13") -> SearchProbe:
    return SearchProbe(
        probe_id=probe_id,
        slot_id=slot_id,
        query=query,
        round=1,
        origin="semantic",
    )


def _probe_task(
    task_id: str = "T-search", probe_id: str = "p1", slot_id: str = "slot-1"
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability="document.search",
        task_objective="probe slot",
        input=DocumentSearchInput(kind="document.search", query="nđ13"),
        depends_on=(),
        origin=DiscoveryProbeTaskOrigin(
            kind="discovery_probe", probe_id=probe_id, slot_id=slot_id, round=1
        ),
    )


def _plan(*tasks: TaskSpec) -> TaskPlan:
    return TaskPlan(
        contract_version="2.0",
        plan_id="plan-1",
        goal="bootstrap",
        target_units=(),
        tasks=tasks,
    )


def _search_result(
    task_id: str = "T-search",
    *,
    candidate_id: UUID | None = None,
    document_id: UUID = factories.DOCUMENT_ID,
    revision: str = factories.REVISION,
    rank: int = 1,
    confidence: float | None = 0.9,
    match_kind: str = "semantic",
    calibration_version: str | None = "cal-1",
) -> AgentResult:
    candidate_id = candidate_id or uuid4()
    return AgentResult(
        contract_version="2.0",
        task_id=task_id,
        status="success",
        data=DocumentSearchOutput(
            kind="document.search",
            candidates=(
                DocumentDiscoveryCandidate(
                    candidate_id=candidate_id,
                    document_id=document_id,
                    document_revision=revision,
                ),
            ),
            identity_matches=(
                DocumentIdentityMatch(
                    candidate_id=candidate_id,
                    document_id=document_id,
                    document_revision=revision,
                    rank=rank,
                    confidence=confidence,
                    match_kind=match_kind,  # type: ignore[arg-type]
                    calibration_version=calibration_version,
                ),
            ),
        ),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )


def _expected_matches(checkpoint_probes, plan, results):
    """Rebuild the expected ProbeCandidateMatch tuple the validator derives."""
    results_by_task = {r.task_id: r for r in results}
    probe_tasks = {
        t.origin.probe_id: t
        for t in plan.tasks
        if isinstance(t.origin, DiscoveryProbeTaskOrigin)
    }
    out = []
    for probe in checkpoint_probes:
        task = probe_tasks[probe.probe_id]
        result = results_by_task[task.task_id]
        for m in result.data.identity_matches:
            from app.services.agents.v2.discovery_bootstrap.contracts import ProbeCandidateMatch

            out.append(
                ProbeCandidateMatch(
                    probe_id=probe.probe_id,
                    slot_id=probe.slot_id,
                    task_id=task.task_id,
                    candidate_id=m.candidate_id,
                    document_id=m.document_id,
                    document_revision=m.document_revision,
                    rank=m.rank,
                    confidence=m.confidence,
                    match_kind=m.match_kind,
                    calibration_version=m.calibration_version,
                )
            )
    return tuple(out)


def _checkpoint(
    *,
    probes=(),
    matches=(),
    aggregates=(),
    selections=(),
    status="selected",
    manifest=None,
    rounds=0,
    probes_consumed=None,
    target_slots=None,
) -> DiscoveryCheckpoint:
    if target_slots is None:
        target_slots = (factories.target_slot(),)
    return DiscoveryCheckpoint(
        discovery_id=DISCOVERY_ID,
        target_slots=target_slots,
        accepted_probes=probes,
        candidate_matches=matches,
        slot_aggregates=aggregates,
        selections=selections,
        rounds_consumed=rounds,
        probes_consumed=(
            len(probes) if probes_consumed is None else probes_consumed
        ),
        status=status,
        clarification_manifest=manifest,
    )


def _valid_selected_checkpoint():
    plan = _plan(_probe_task())
    probes = (_probe(),)
    results = (_search_result(),)
    matches = _expected_matches(probes, plan, results)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    selections = (
        DiscoverySelection(
            slot_id="slot-1",
            selected_aggregate_ids=(aggregates[0].aggregate_id,),
            authority="confidence_policy",
        ),
    )
    checkpoint = _checkpoint(
        probes=probes,
        matches=matches,
        aggregates=aggregates,
        selections=selections,
        rounds=1,
    )
    return checkpoint, plan, results


def test_checkpoint_rebuilds_after_registry_loss() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    validate_discovery_checkpoint(
        checkpoint, plan=plan, task_results=results
    )


def test_forged_candidate_match_rejected() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    forged = checkpoint.candidate_matches[0].model_copy(
        update={"confidence": 0.01}
    )
    bad = checkpoint.model_copy(
        update={"candidate_matches": (forged,), "slot_aggregates": ()}
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(bad, plan=plan, task_results=results)


def test_forged_aggregate_rejected() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    forged = checkpoint.slot_aggregates[0].model_copy(
        update={"aggregate_id": uuid4()}
    )
    bad = checkpoint.model_copy(
        update={
            "slot_aggregates": (forged,),
            "selections": (
                DiscoverySelection(
                    slot_id="slot-1",
                    selected_aggregate_ids=(forged.aggregate_id,),
                    authority="confidence_policy",
                ),
            ),
        }
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(bad, plan=plan, task_results=results)


def test_unknown_probe_task_rejected() -> None:
    checkpoint, _plan_obj, results = _valid_selected_checkpoint()
    other_plan = _plan(
        _probe_task(task_id="T-other", probe_id="p9", slot_id="slot-1")
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            checkpoint, plan=other_plan, task_results=results
        )


def test_plan_without_probe_task_rejected() -> None:
    checkpoint, _plan_obj, _results = _valid_selected_checkpoint()
    empty_plan = _plan()
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            checkpoint, plan=empty_plan, task_results=()
        )


def test_no_plan_with_probes_rejected() -> None:
    checkpoint, _, results = _valid_selected_checkpoint()
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            checkpoint, plan=None, task_results=results
        )


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.5])
def test_nonfinite_or_out_of_range_confidence_rejected(confidence: float) -> None:
    plan = _plan(_probe_task())
    probes = (_probe(),)
    results = (
        _search_result(confidence=confidence),
    )
    matches = _expected_matches(probes, plan, results)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    checkpoint = _checkpoint(
        probes=probes, matches=matches, aggregates=aggregates, rounds=1
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            checkpoint, plan=plan, task_results=results
        )


def test_exact_match_with_confidence_rejected() -> None:
    plan = _plan(_probe_task())
    probes = (_probe(),)
    results = (
        _search_result(
            confidence=0.9,
            match_kind="exact_document_number",
            calibration_version=None,
        ),
    )
    matches = _expected_matches(probes, plan, results)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    checkpoint = _checkpoint(
        probes=probes, matches=matches, aggregates=aggregates, rounds=1
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            checkpoint, plan=plan, task_results=results
        )


def test_nonexact_match_requires_calibration_version() -> None:
    plan = _plan(_probe_task())
    probes = (_probe(),)
    results = (
        _search_result(confidence=0.9, calibration_version=None),
    )
    matches = _expected_matches(probes, plan, results)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    checkpoint = _checkpoint(
        probes=probes, matches=matches, aggregates=aggregates, rounds=1
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            checkpoint, plan=plan, task_results=results
        )


def _search_result_two_matches(
    task_id: str = "T-search",
    *,
    first_kind: str = "exact_document_number",
    second_kind: str = "exact_normalized_title",
    first_confidence: float | None = None,
    second_confidence: float | None = None,
    calibration_version: str | None = None,
) -> AgentResult:
    first_candidate = uuid4()
    second_candidate = uuid4()
    return AgentResult(
        contract_version="2.0",
        task_id=task_id,
        status="success",
        data=DocumentSearchOutput(
            kind="document.search",
            candidates=(
                DocumentDiscoveryCandidate(
                    candidate_id=first_candidate,
                    document_id=factories.DOCUMENT_ID,
                    document_revision=factories.REVISION,
                ),
                DocumentDiscoveryCandidate(
                    candidate_id=second_candidate,
                    document_id=factories.OTHER_DOCUMENT_ID,
                    document_revision=factories.REVISION,
                ),
            ),
            identity_matches=(
                DocumentIdentityMatch(
                    candidate_id=first_candidate,
                    document_id=factories.DOCUMENT_ID,
                    document_revision=factories.REVISION,
                    rank=1,
                    confidence=first_confidence,
                    match_kind=first_kind,  # type: ignore[arg-type]
                    calibration_version=calibration_version,
                ),
                DocumentIdentityMatch(
                    candidate_id=second_candidate,
                    document_id=factories.OTHER_DOCUMENT_ID,
                    document_revision=factories.REVISION,
                    rank=2,
                    confidence=second_confidence,
                    match_kind=second_kind,  # type: ignore[arg-type]
                    calibration_version=calibration_version,
                ),
            ),
        ),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )


def test_exact_ambiguity_not_auto_selected() -> None:
    plan = _plan(_probe_task())
    probes = (_probe(),)
    # Two distinct exact identities for the same slot.
    results = (_search_result_two_matches(),)
    matches = _expected_matches(probes, plan, results)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    assert len(aggregates) == 2
    checkpoint = _checkpoint(
        probes=probes,
        matches=matches,
        aggregates=aggregates,
        selections=(
            DiscoverySelection(
                slot_id="slot-1",
                selected_aggregate_ids=(aggregates[0].aggregate_id,),
                authority="exact_match_policy",
            ),
        ),
        rounds=1,
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            checkpoint, plan=plan, task_results=results
        )


def test_duplicate_selection_slot_rejected() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    duplicate = checkpoint.selections[0]
    bad = checkpoint.model_copy(
        update={"selections": checkpoint.selections + (duplicate,)}
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(bad, plan=plan, task_results=results)


def test_policy_selection_must_be_ranking_prefix() -> None:
    plan = _plan(_probe_task())
    probes = (_probe(),)
    results = (
        _search_result_two_matches(
            first_kind="semantic",
            second_kind="semantic",
            first_confidence=0.9,
            second_confidence=0.7,
            calibration_version="cal-1",
        ),
    )
    matches = _expected_matches(probes, plan, results)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    # Select the runner-up under a policy authority: not a prefix.
    checkpoint = _checkpoint(
        probes=probes,
        matches=matches,
        aggregates=aggregates,
        selections=(
            DiscoverySelection(
                slot_id="slot-1",
                selected_aggregate_ids=(aggregates[1].aggregate_id,),
                authority="confidence_policy",
            ),
        ),
        rounds=1,
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            checkpoint, plan=plan, task_results=results
        )


def test_normalized_equivalent_probe_rejected() -> None:
    probes = (_probe(query="nđ13"), _probe("p2", query="  NĐ13 "))
    checkpoint = _checkpoint(
        probes=probes,
        status="searching",
        rounds=1,
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, plan=None)


def test_counters_must_be_exact() -> None:
    checkpoint = _checkpoint(status="searching", probes_consumed=1)
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, plan=None)


def test_status_selected_requires_required_slot_selection() -> None:
    checkpoint = _checkpoint(status="selected", selections=())
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, plan=None)


def test_manifest_request_consistency() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    aggregate = checkpoint.slot_aggregates[0]
    manifest = DocumentSelectionManifest(
        clarification_id="clar-1",
        entries=(
            DocumentSelectionManifestEntry(
                choice_token_digest="hmac:abc",
                slot_id="slot-1",
                aggregate_id=aggregate.aggregate_id,
                document_id=aggregate.document_id,
                document_revision=aggregate.document_revision,
            ),
        ),
        expires_at=EXPIRES,
        status="pending",
    )
    clarification = DocumentSelectionClarification(
        kind="document_selection",
        contract_version="2.0",
        clarification_id="clar-1",
        question="Chọn tài liệu?",
        slots=(
            DocumentSelectionSlot(
                slot_id="slot-1",
                slot_label="Mục tiêu",
                min_selections=1,
                max_selections=1,
                choices=(
                    DocumentSelectionChoice(
                        choice_token="tok-1",
                        title="Nghị định 13",
                        document_number=None,
                    ),
                ),
            ),
        ),
        expires_at=EXPIRES,
    )
    pending = checkpoint.model_copy(
        update={
            "status": "clarification",
            "selections": (),
            "clarification_manifest": manifest,
        }
    )
    validate_discovery_checkpoint(
        pending, plan=plan, task_results=results, clarification=clarification
    )


def test_manifest_entry_must_match_checkpoint_aggregate() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    manifest = DocumentSelectionManifest(
        clarification_id="clar-1",
        entries=(
            DocumentSelectionManifestEntry(
                choice_token_digest="hmac:abc",
                slot_id="slot-1",
                aggregate_id=uuid4(),
                document_id=factories.DOCUMENT_ID,
                document_revision=factories.REVISION,
            ),
        ),
        expires_at=EXPIRES,
        status="pending",
    )
    clarification = DocumentSelectionClarification(
        kind="document_selection",
        contract_version="2.0",
        clarification_id="clar-1",
        question="Chọn tài liệu?",
        slots=(
            DocumentSelectionSlot(
                slot_id="slot-1",
                slot_label="Mục tiêu",
                min_selections=1,
                max_selections=1,
                choices=(
                    DocumentSelectionChoice(
                        choice_token="tok-1",
                        title="Nghị định 13",
                        document_number=None,
                    ),
                ),
            ),
        ),
        expires_at=EXPIRES,
    )
    pending = checkpoint.model_copy(
        update={
            "status": "clarification",
            "selections": (),
            "clarification_manifest": manifest,
        }
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            pending, plan=plan, task_results=results, clarification=clarification
        )


def test_clarification_status_requires_manifest() -> None:
    checkpoint = _checkpoint(status="clarification")
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, plan=None)


def test_duplicate_task_result_id_rejected() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            checkpoint, plan=plan, task_results=results + results
        )


def test_probe_task_capability_mismatch_rejected() -> None:
    plan = _plan(
        _probe_task().model_copy(update={"capability": "document.retrieve"})
    )
    probes = (_probe(),)
    results = (_search_result(),)
    matches = _expected_matches(probes, plan, results)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    checkpoint = _checkpoint(
        probes=probes, matches=matches, aggregates=aggregates, rounds=1
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, plan=plan, task_results=results)


def test_probe_task_query_mismatch_rejected() -> None:
    task = _probe_task().model_copy(
        update={
            "input": DocumentSearchInput(kind="document.search", query="khác")
        }
    )
    plan = _plan(task)
    probes = (_probe(),)
    results = (_search_result(),)
    matches = _expected_matches(probes, plan, results)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    checkpoint = _checkpoint(
        probes=probes, matches=matches, aggregates=aggregates, rounds=1
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, plan=plan, task_results=results)


def test_result_identity_match_set_mismatch_rejected() -> None:
    result = _search_result()
    extra = DocumentDiscoveryCandidate(
        candidate_id=uuid4(),
        document_id=factories.OTHER_DOCUMENT_ID,
        document_revision=factories.REVISION,
    )
    result = result.model_copy(
        update={
            "data": result.data.model_copy(
                update={"candidates": result.data.candidates + (extra,)}
            )
        }
    )
    plan = _plan(_probe_task())
    probes = (_probe(),)
    results = (result,)
    matches = _expected_matches(probes, plan, results)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    checkpoint = _checkpoint(
        probes=probes, matches=matches, aggregates=aggregates, rounds=1
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, plan=plan, task_results=results)


def test_result_identity_match_document_mismatch_rejected() -> None:
    result = _search_result()
    match = result.data.identity_matches[0].model_copy(
        update={"document_id": factories.OTHER_DOCUMENT_ID}
    )
    result = result.model_copy(
        update={
            "data": result.data.model_copy(
                update={"identity_matches": (match,)}
            )
        }
    )
    plan = _plan(_probe_task())
    probes = (_probe(),)
    results = (result,)
    matches = _expected_matches(probes, plan, results)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, probes, (factories.target_slot(),)
    )
    checkpoint = _checkpoint(
        probes=probes, matches=matches, aggregates=aggregates, rounds=1
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, plan=plan, task_results=results)


def _manifest_for(
    aggregate,
    *,
    status: str,
    entries: tuple | None = None,
) -> DocumentSelectionManifest:
    if entries is None:
        entries = (
            DocumentSelectionManifestEntry(
                choice_token_digest="hmac:abc",
                slot_id=aggregate.slot_id,
                aggregate_id=aggregate.aggregate_id,
                document_id=aggregate.document_id,
                document_revision=aggregate.document_revision,
            ),
        )
    return DocumentSelectionManifest(
        clarification_id="clar-1",
        entries=entries,
        expires_at=EXPIRES,
        status=status,  # type: ignore[arg-type]
    )


def _clarification(*, choices: tuple | None = None) -> DocumentSelectionClarification:
    if choices is None:
        choices = (
            DocumentSelectionChoice(
                choice_token="tok-1", title="Nghị định 13", document_number=None
            ),
        )
    return DocumentSelectionClarification(
        kind="document_selection",
        contract_version="2.0",
        clarification_id="clar-1",
        question="Chọn tài liệu?",
        slots=(
            DocumentSelectionSlot(
                slot_id="slot-1",
                slot_label="Mục tiêu",
                min_selections=1,
                max_selections=1,
                choices=choices,
            ),
        ),
        expires_at=EXPIRES,
    )


def test_pending_manifest_requires_clarification_status() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    manifest = _manifest_for(checkpoint.slot_aggregates[0], status="pending")
    bad = checkpoint.model_copy(update={"clarification_manifest": manifest})
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            bad,
            plan=plan,
            task_results=results,
            clarification=_clarification(),
        )


def test_pending_manifest_entry_count_must_match_choices() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    manifest = _manifest_for(checkpoint.slot_aggregates[0], status="pending")
    clarification = _clarification(
        choices=(
            DocumentSelectionChoice(
                choice_token="tok-1", title="Nghị định 13", document_number=None
            ),
            DocumentSelectionChoice(
                choice_token="tok-2", title="Nghị định 15", document_number=None
            ),
        )
    )
    pending = checkpoint.model_copy(
        update={
            "status": "clarification",
            "selections": (),
            "clarification_manifest": manifest,
        }
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            pending, plan=plan, task_results=results, clarification=clarification
        )


def test_pending_manifest_requires_sibling_request() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    manifest = _manifest_for(checkpoint.slot_aggregates[0], status="pending")
    pending = checkpoint.model_copy(
        update={
            "status": "clarification",
            "selections": (),
            "clarification_manifest": manifest,
        }
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(pending, plan=plan, task_results=results)


def test_sibling_request_without_pending_manifest_rejected() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            checkpoint,
            plan=plan,
            task_results=results,
            clarification=_clarification(),
        )


def test_consumed_manifest_selected_without_request() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    manifest = _manifest_for(checkpoint.slot_aggregates[0], status="consumed")
    consumed = checkpoint.model_copy(update={"clarification_manifest": manifest})
    validate_discovery_checkpoint(consumed, plan=plan, task_results=results)


def test_consumed_manifest_unavailable_without_request() -> None:
    checkpoint = _checkpoint(
        status="unavailable",
        manifest=_manifest_for(
            factories.target_slot(), status="consumed", entries=()
        ),
    )
    validate_discovery_checkpoint(checkpoint, plan=None)


@pytest.mark.parametrize(
    "update",
    (
        {"clarification_id": " "},
        {"expires_at": datetime(2026, 9, 18)},  # noqa: DTZ001 - intentionally naive
    ),
)
def test_consumed_manifest_requires_structural_integrity(update: dict) -> None:
    checkpoint = _checkpoint(
        status="unavailable",
        manifest=_manifest_for(
            factories.target_slot(), status="consumed", entries=()
        ).model_copy(update=update),
    )
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, plan=None)


def test_consumed_manifest_rejects_sibling_request() -> None:
    checkpoint, plan, results = _valid_selected_checkpoint()
    manifest = _manifest_for(checkpoint.slot_aggregates[0], status="consumed")
    consumed = checkpoint.model_copy(update={"clarification_manifest": manifest})
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(
            consumed,
            plan=plan,
            task_results=results,
            clarification=_clarification(),
        )


def test_consumed_manifest_rejected_while_searching() -> None:
    manifest = _manifest_for(
        factories.target_slot(), status="consumed", entries=()
    )
    checkpoint = _checkpoint(status="searching", manifest=manifest)
    with pytest.raises(ContractValidationError):
        validate_discovery_checkpoint(checkpoint, plan=None)


def _pending_clarification_state() -> dict:
    from app.services.agents.supervisor_v2 import build_initial_v2_state
    from app.services.agents.v2.contracts.routing import RouteDecision
    from app.services.agents.v2.contracts.state import ExecutionState

    checkpoint, plan, results = _valid_selected_checkpoint()
    clarification = _clarification()
    manifest = _manifest_for(
        checkpoint.slot_aggregates[0], status="pending"
    )
    pending = checkpoint.model_copy(
        update={
            "status": "clarification",
            "selections": (),
            "clarification_manifest": manifest,
        }
    )
    state = dict(build_initial_v2_state(request=factories.request_context()))
    state["semantic"] = factories.semantic_context()
    state["execution"] = ExecutionState(
        plan=plan,
        task_results=results,
        evidence_evaluation=None,
    )
    state["route_decision"] = RouteDecision(
        route="clarify", reason_code="essential_ambiguity"
    )
    state["discovery"] = pending
    state["document_selection_clarification"] = clarification
    return state


def test_clarify_route_accepts_document_selection_request() -> None:
    from app.services.agents.v2.contracts.validation import validate_supervisor_state

    validate_supervisor_state(_pending_clarification_state())  # type: ignore[arg-type]


def test_document_selection_request_requires_discovery_checkpoint() -> None:
    from app.services.agents.v2.contracts.validation import validate_supervisor_state

    state = _pending_clarification_state()
    state["discovery"] = None
    with pytest.raises(
        ContractValidationError, match="requires a DiscoveryCheckpoint"
    ):
        validate_supervisor_state(state)  # type: ignore[arg-type]

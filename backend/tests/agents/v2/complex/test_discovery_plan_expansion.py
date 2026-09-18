"""Discovery plan Task 7 — typed governed expansion (spec §13.1/§13.2).

``expand_discovery_plan`` is the single authoritative append constructor for
the bootstrap transition; ``validate_plan_expansion`` is the frozen invariant
set. These tests exercise every §13.1 invariant with typed contracts only —
no ``Sequence[Any]`` outcomes, no getattr-based outcome inspection, no
synthetic dynamic outcome objects.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.contracts.binding import (
    DocumentBindingSet,
    DocumentDiscoveryCandidate,
)
from app.services.agents.v2.contracts.capability import (
    DocumentIdentityMatch,
    DocumentReadInput,
    DocumentRetrieveInput,
    DocumentSearchInput,
    DocumentSearchOutput,
)
from app.services.agents.v2.contracts.execution import AgentError, AgentResult
from app.services.agents.v2.contracts.locators import SectionLocator
from app.services.agents.v2.contracts.planning import (
    DiscoveryExpansionTaskOrigin,
    DiscoveryProbeTaskOrigin,
    InitialTaskOrigin,
    ReplanTaskOrigin,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_task_plan,
)
from app.services.agents.v2.discovery_bootstrap.contracts import (
    DiscoveryCheckpoint,
    DiscoverySelection,
    ProbeCandidateMatch,
    ResearchTargetSelection,
    SearchProbe,
    SelectedBindingRef,
    SlotBindingSelection,
    SlotCandidateAggregate,
    TargetSlot,
)
from app.services.agents.v2.discovery_bootstrap.plan_expansion import (
    expand_discovery_plan,
    factual_tasks,
    validate_plan_expansion,
)
from app.services.agents.v2.discovery_bootstrap.validation import aggregate_matches

from tests.agents.v2.contracts import factories

DISCOVERY_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
DOC_A = factories.DOCUMENT_ID
DOC_B = factories.OTHER_DOCUMENT_ID
REV = factories.REVISION
OTHER_REV = factories.OTHER_REVISION

CAPABILITIES = frozenset({"document.search", "document.read", "section.read"})
MAX_RESEARCH_TASKS = 8
TOTAL_TASK_LIMIT = 13


# ---------------------------------------------------------------------------
# Typed fixture builders
# ---------------------------------------------------------------------------


def _probe(
    probe_id: str = "p1",
    slot_id: str = "slot-1",
    query: str = "nđ13",
    round: int = 1,
) -> SearchProbe:
    return SearchProbe(
        probe_id=probe_id,
        slot_id=slot_id,
        query=query,
        round=round,  # type: ignore[arg-type]
        origin="semantic",
    )


def _probe_task(
    task_id: str = "T-search",
    probe_id: str = "p1",
    slot_id: str = "slot-1",
    query: str = "nđ13",
    round: int = 1,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability="document.search",
        task_objective="probe slot",
        input=DocumentSearchInput(kind="document.search", query=query),
        depends_on=(),
        origin=DiscoveryProbeTaskOrigin(
            kind="discovery_probe",
            probe_id=probe_id,
            slot_id=slot_id,
            round=round,  # type: ignore[arg-type]
        ),
    )


def _search_result(
    task_id: str = "T-search",
    *,
    status: str = "success",
    document_id: UUID = DOC_A,
    revision: str = REV,
    rank: int = 1,
    confidence: float | None = 0.9,
    match_kind: str = "semantic",
    calibration_version: str | None = "cal-1",
    candidate_id: UUID | None = None,
) -> AgentResult:
    """A typed checkpointed ``AgentResult`` — never a synthetic outcome."""
    candidate_id = candidate_id or uuid4()
    if status == "success":
        data: object = DocumentSearchOutput(
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
        )
        error = None
    elif status == "error":
        data = None
        error = AgentError(code="TIMEOUT", message="probe timed out", retryable=True)
    else:
        data = None
        error = None
    return AgentResult(
        contract_version="2.0",
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        data=data,  # type: ignore[arg-type]
        evidence_uses=(),
        coverage_observations=(),
        error=error,
    )


def _matches(
    probes: tuple[SearchProbe, ...],
    plan: TaskPlan,
    results: tuple[AgentResult, ...],
) -> tuple[ProbeCandidateMatch, ...]:
    """Rebuild the checkpointed ProbeCandidateMatch tuple the runtime derives."""
    results_by_task = {result.task_id: result for result in results}
    probe_tasks = {
        task.origin.probe_id: task
        for task in plan.tasks
        if isinstance(task.origin, DiscoveryProbeTaskOrigin)
    }
    out: list[ProbeCandidateMatch] = []
    for probe in probes:
        task = probe_tasks[probe.probe_id]
        result = results_by_task[task.task_id]
        assert isinstance(result.data, DocumentSearchOutput)
        for match in result.data.identity_matches:
            out.append(
                ProbeCandidateMatch(
                    probe_id=probe.probe_id,
                    slot_id=probe.slot_id,
                    task_id=task.task_id,
                    candidate_id=match.candidate_id,
                    document_id=match.document_id,
                    document_revision=match.document_revision,
                    rank=match.rank,
                    confidence=match.confidence,
                    match_kind=match.match_kind,
                    calibration_version=match.calibration_version,
                )
            )
    return tuple(out)


def _expansion_read(
    task_id: str,
    target_ids: tuple[str, ...],
    source_ids: tuple[str, ...],
    slot_ids: tuple[str, ...],
    aggregate_ids: tuple[UUID, ...],
    *,
    depends_on: tuple[str, ...] = ("T-search",),
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability="document.read",
        task_objective="read selected",
        input=DocumentReadInput(kind="document.read", target_ids=target_ids),
        depends_on=depends_on,
        origin=DiscoveryExpansionTaskOrigin(
            kind="discovery_expansion",
            source_search_task_ids=source_ids,
            target_slot_ids=slot_ids,
            selected_aggregate_ids=aggregate_ids,
        ),
    )


@dataclass
class _Case:
    current: TaskPlan
    checkpoint: DiscoveryCheckpoint
    selection: ResearchTargetSelection
    bindings: DocumentBindingSet
    outcomes: tuple[AgentResult, ...]
    appended_units: tuple[TargetUnit, ...]
    appended_tasks: tuple[TaskSpec, ...]
    aggregate: SlotCandidateAggregate
    slot: TargetSlot


def _summarize_case() -> _Case:
    """One probe, one aggregate, one selected binding, one read task."""
    slot = factories.target_slot(slot_id="slot-1", explicit_binding_ids=())
    bindings = factories.binding_set()
    probe = _probe()
    search_task = _probe_task()
    current = TaskPlan(
        contract_version="2.0",
        plan_id="plan-1",
        goal="bootstrap",
        target_units=(),
        tasks=(search_task,),
    )
    outcome = _search_result()
    matches = _matches((probe,), current, (outcome,))
    aggregates = aggregate_matches(DISCOVERY_ID, matches, (probe,), (slot,))
    aggregate = aggregates[0]
    checkpoint = DiscoveryCheckpoint(
        discovery_id=DISCOVERY_ID,
        target_slots=(slot,),
        accepted_probes=(probe,),
        candidate_matches=matches,
        slot_aggregates=aggregates,
        selections=(
            DiscoverySelection(
                slot_id="slot-1",
                selected_aggregate_ids=(aggregate.aggregate_id,),
                authority="confidence_policy",
            ),
        ),
        rounds_consumed=1,
        probes_consumed=1,
        status="selected",
        clarification_manifest=None,
    )
    selection = factories.target_selection(
        target_slots=(slot,),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    SelectedBindingRef(
                        target_id="t1",
                        binding_id="b1",
                        selected_aggregate_id=aggregate.aggregate_id,
                        authority="confidence_policy",
                    ),
                ),
            ),
        ),
    )
    unit = factories.target_unit(
        target_id="t1",
        binding_id="b1",
        requested_locator=slot.requested_locator,
    )
    read = _expansion_read(
        "T-read",
        ("t1",),
        ("T-search",),
        ("slot-1",),
        (aggregate.aggregate_id,),
    )
    return _Case(
        current=current,
        checkpoint=checkpoint,
        selection=selection,
        bindings=bindings,
        outcomes=(outcome,),
        appended_units=(unit,),
        appended_tasks=(read,),
        aggregate=aggregate,
        slot=slot,
    )


def _proposed(
    case: _Case,
    *,
    units: tuple[TargetUnit, ...] | None = None,
    tasks: tuple[TaskSpec, ...] | None = None,
) -> TaskPlan:
    return case.current.model_copy(
        update={
            "target_units": case.current.target_units
            + (case.appended_units if units is None else units),
            "tasks": case.current.tasks
            + (case.appended_tasks if tasks is None else tasks),
        }
    )


def _validate(
    case: _Case,
    proposed: TaskPlan | None = None,
    *,
    checkpoint: DiscoveryCheckpoint | None = None,
    selection: ResearchTargetSelection | None = None,
    bindings: DocumentBindingSet | None = None,
    outcomes: tuple[AgentResult, ...] | None = None,
    capability_names: frozenset[str] = CAPABILITIES,
    max_research_tasks: int = MAX_RESEARCH_TASKS,
    total_task_limit: int = TOTAL_TASK_LIMIT,
) -> None:
    validate_plan_expansion(
        case.current,
        proposed if proposed is not None else _proposed(case),
        checkpoint if checkpoint is not None else case.checkpoint,
        selection if selection is not None else case.selection,
        bindings if bindings is not None else case.bindings,
        outcomes if outcomes is not None else case.outcomes,
        capability_names=capability_names,
        max_research_tasks=max_research_tasks,
        total_task_limit=total_task_limit,
    )


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_expand_discovery_plan_returns_only_validated_plan() -> None:
    case = _summarize_case()
    expanded = expand_discovery_plan(
        case.current,
        case.checkpoint,
        case.selection,
        case.bindings,
        case.outcomes,
        case.appended_units,
        case.appended_tasks,
        capability_names=CAPABILITIES,
        max_research_tasks=MAX_RESEARCH_TASKS,
        total_task_limit=TOTAL_TASK_LIMIT,
    )
    assert expanded.plan_id == case.current.plan_id
    assert expanded.goal == case.current.goal
    assert expanded.target_units == case.appended_units
    assert expanded.tasks == case.current.tasks + case.appended_tasks
    # Purity: the checkpointed current plan is untouched.
    assert case.current.target_units == ()
    assert len(case.current.tasks) == 1
    # Invariant 13 holds for the returned plan itself.
    validate_task_plan(
        expanded,
        case.bindings,
        target_selection=case.selection,
        discovery_checkpoint=case.checkpoint,
    )


def test_validate_plan_expansion_accepts() -> None:
    _validate(_summarize_case())


def test_factual_tasks_exempts_only_checkpointed_probe() -> None:
    case = _summarize_case()
    proposed = _proposed(case)
    assert factual_tasks(proposed, discovery_checkpoint=case.checkpoint) == (
        case.appended_tasks[0],
    )
    with pytest.raises(ContractValidationError):
        factual_tasks(proposed, discovery_checkpoint=None)


def test_compare_two_slot_expansion() -> None:
    """Two logical side slots, two probes, two selected bindings (§14.3)."""
    slot_a = factories.target_slot(
        slot_id="slot-1", intended_role="target", explicit_binding_ids=()
    )
    slot_b = factories.target_slot(
        slot_id="slot-2", intended_role="reference", explicit_binding_ids=()
    )
    slots = (slot_a, slot_b)
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(binding_id="b1", document_id=DOC_A),
            factories.scoped_document(
                binding_id="b2", document_id=DOC_B, role="reference"
            ),
        )
    )
    probes = (
        _probe("p1", "slot-1", "nghị định A"),
        _probe("p2", "slot-2", "nghị định B"),
    )
    search_a = _probe_task("T-search-1", "p1", "slot-1", "nghị định A")
    search_b = _probe_task("T-search-2", "p2", "slot-2", "nghị định B")
    current = TaskPlan(
        contract_version="2.0",
        plan_id="plan-compare",
        goal="compare",
        target_units=(),
        tasks=(search_a, search_b),
    )
    outcome_a = _search_result("T-search-1", document_id=DOC_A)
    outcome_b = _search_result("T-search-2", document_id=DOC_B)
    outcomes = (outcome_a, outcome_b)
    matches = _matches(probes, current, outcomes)
    aggregates = aggregate_matches(DISCOVERY_ID, matches, probes, slots)
    by_slot = {aggregate.slot_id: aggregate for aggregate in aggregates}
    checkpoint = DiscoveryCheckpoint(
        discovery_id=DISCOVERY_ID,
        target_slots=slots,
        accepted_probes=probes,
        candidate_matches=matches,
        slot_aggregates=aggregates,
        selections=(
            DiscoverySelection(
                slot_id="slot-1",
                selected_aggregate_ids=(by_slot["slot-1"].aggregate_id,),
                authority="confidence_policy",
            ),
            DiscoverySelection(
                slot_id="slot-2",
                selected_aggregate_ids=(by_slot["slot-2"].aggregate_id,),
                authority="confidence_policy",
            ),
        ),
        rounds_consumed=1,
        probes_consumed=2,
        status="selected",
        clarification_manifest=None,
    )
    selection = factories.target_selection(
        work_type="compare",
        target_slots=slots,
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    SelectedBindingRef(
                        target_id="t1",
                        binding_id="b1",
                        selected_aggregate_id=by_slot["slot-1"].aggregate_id,
                        authority="confidence_policy",
                    ),
                ),
            ),
            SlotBindingSelection(
                slot_id="slot-2",
                selections=(
                    SelectedBindingRef(
                        target_id="t2",
                        binding_id="b2",
                        selected_aggregate_id=by_slot["slot-2"].aggregate_id,
                        authority="confidence_policy",
                    ),
                ),
            ),
        ),
    )
    units = (
        factories.target_unit(
            target_id="t1",
            binding_id="b1",
            requested_locator=slot_a.requested_locator,
        ),
        factories.target_unit(
            target_id="t2",
            binding_id="b2",
            requested_locator=slot_b.requested_locator,
        ),
    )
    tasks = (
        _expansion_read(
            "T-read-1",
            ("t1",),
            ("T-search-1",),
            ("slot-1",),
            (by_slot["slot-1"].aggregate_id,),
            depends_on=("T-search-1",),
        ),
        _expansion_read(
            "T-read-2",
            ("t2",),
            ("T-search-2",),
            ("slot-2",),
            (by_slot["slot-2"].aggregate_id,),
            depends_on=("T-search-2",),
        ),
    )
    case = _Case(
        current=current,
        checkpoint=checkpoint,
        selection=selection,
        bindings=bindings,
        outcomes=outcomes,
        appended_units=units,
        appended_tasks=tasks,
        aggregate=by_slot["slot-1"],
        slot=slot_a,
    )
    _validate(case)
    expanded = expand_discovery_plan(
        case.current,
        case.checkpoint,
        case.selection,
        case.bindings,
        case.outcomes,
        case.appended_units,
        case.appended_tasks,
        capability_names=CAPABILITIES,
        max_research_tasks=MAX_RESEARCH_TASKS,
        total_task_limit=TOTAL_TASK_LIMIT,
    )
    assert expanded.tasks == current.tasks + tasks


def test_failed_outcome_on_unselected_probe_is_preserved() -> None:
    """Non-success statuses are preserved: a probe that produced no selected
    candidate may keep its ``not_found`` outcome while another probe's
    selected aggregate still authorizes the expansion."""
    case = _summarize_case()
    second_probe = _probe("p2", "slot-1", "nghị định khác")
    second_task = _probe_task(
        "T-search-2", "p2", "slot-1", "nghị định khác"
    )
    current = case.current.model_copy(
        update={"tasks": case.current.tasks + (second_task,)}
    )
    outcomes = case.outcomes + (
        _search_result("T-search-2", status="not_found"),
    )
    probes = case.checkpoint.accepted_probes + (second_probe,)
    checkpoint = case.checkpoint.model_copy(
        update={
            "accepted_probes": probes,
            "probes_consumed": len(probes),
        }
    )
    _validate(replace(case, current=current, checkpoint=checkpoint), outcomes=outcomes)


# ---------------------------------------------------------------------------
# Invariants 1-2: identity, immutability, exact prefixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "update",
    [{"plan_id": "forged"}, {"goal": "forged goal"}],
)
def test_plan_identity_mutation_rejected(update: dict) -> None:
    case = _summarize_case()
    proposed = _proposed(case).model_copy(update=update)
    with pytest.raises(ContractValidationError):
        _validate(case, proposed)


def test_existing_task_mutated_rejected() -> None:
    case = _summarize_case()
    mutated = case.current.tasks[0].model_copy(
        update={"task_objective": "forged objective"}
    )
    proposed = _proposed(case).model_copy(
        update={"tasks": (mutated,) + case.appended_tasks}
    )
    with pytest.raises(ContractValidationError):
        _validate(case, proposed)


def test_existing_task_dropped_rejected() -> None:
    case = _summarize_case()
    proposed = _proposed(case).model_copy(
        update={"tasks": case.appended_tasks}
    )
    with pytest.raises(ContractValidationError):
        _validate(case, proposed)


def test_current_target_unit_rejected() -> None:
    """A bootstrap plan carries probes only; units exist only after expansion."""
    case = _summarize_case()
    current = case.current.model_copy(
        update={"target_units": case.appended_units[:1]}
    )
    with pytest.raises(
        ContractValidationError,
        match="must not carry target units",
    ):
        _validate(replace(case, current=current))


def test_no_appended_unit_rejected() -> None:
    case = _summarize_case()
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, units=()))


def test_no_appended_task_rejected() -> None:
    case = _summarize_case()
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, tasks=()))


# ---------------------------------------------------------------------------
# Invariants 4-6: unit → selected ref → aggregate → binding identity
# ---------------------------------------------------------------------------


def test_unit_target_id_without_selected_ref_rejected() -> None:
    case = _summarize_case()
    unit = case.appended_units[0].model_copy(update={"target_id": "t9"})
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, units=(unit,)))


def test_unit_binding_mismatch_rejected() -> None:
    case = _summarize_case()
    unit = case.appended_units[0].model_copy(update={"binding_id": "b2"})
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, units=(unit,)))


def test_unit_wrong_locator_rejected() -> None:
    case = _summarize_case()
    unit = case.appended_units[0].model_copy(
        update={
            "requested_locator": SectionLocator(
                kind="section", structure_node_id="n9"
            )
        }
    )
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, units=(unit,)))


def test_unselected_aggregate_rejected() -> None:
    """An aggregate offered in the slot but absent from checkpoint.selections
    cannot authorize expansion (invariant 11)."""
    case = _summarize_case()
    # A second aggregate for a different document exists in the slot but is
    # not checkpoint-selected; binding b2 matches its identity exactly.
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(),
            factories.scoped_document(
                binding_id="b2", document_id=DOC_B, role="discovered"
            ),
        )
    )
    other_match = ProbeCandidateMatch(
        probe_id="p1",
        slot_id="slot-1",
        task_id="T-search",
        candidate_id=uuid4(),
        document_id=DOC_B,
        document_revision=REV,
        rank=2,
        confidence=0.5,
        match_kind="semantic",
        calibration_version="cal-1",
    )
    matches = case.checkpoint.candidate_matches + (other_match,)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, case.checkpoint.accepted_probes, (case.slot,)
    )
    unselected = next(
        aggregate
        for aggregate in aggregates
        if aggregate.aggregate_id != case.aggregate.aggregate_id
    )
    checkpoint = case.checkpoint.model_copy(
        update={
            "candidate_matches": matches,
            "slot_aggregates": aggregates,
        }
    )
    selection = factories.target_selection(
        target_slots=(case.slot,),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    SelectedBindingRef(
                        target_id="t1",
                        binding_id="b2",
                        selected_aggregate_id=unselected.aggregate_id,
                        authority="confidence_policy",
                    ),
                ),
            ),
        ),
    )
    unit = factories.target_unit(
        target_id="t1",
        binding_id="b2",
        requested_locator=case.slot.requested_locator,
    )
    read = _expansion_read(
        "T-read",
        ("t1",),
        ("T-search",),
        ("slot-1",),
        (unselected.aggregate_id,),
    )
    # The outcome must contain the second candidate too so reconstruction
    # stays exact; the ref still names a non-selected aggregate.
    outcome = _search_result()
    second_data = outcome.data.model_copy(
        update={
            "candidates": outcome.data.candidates
            + (
                DocumentDiscoveryCandidate(
                    candidate_id=other_match.candidate_id,
                    document_id=DOC_B,
                    document_revision=REV,
                ),
            ),
            "identity_matches": outcome.data.identity_matches
            + (
                DocumentIdentityMatch(
                    candidate_id=other_match.candidate_id,
                    document_id=DOC_B,
                    document_revision=REV,
                    rank=2,
                    confidence=0.5,
                    match_kind="semantic",
                    calibration_version="cal-1",
                ),
            ),
        }
    )
    outcome = outcome.model_copy(update={"data": second_data})
    proposed = _proposed(case, units=(unit,), tasks=(read,))
    with pytest.raises(ContractValidationError):
        _validate(
            case,
            proposed,
            checkpoint=checkpoint,
            selection=selection,
            bindings=bindings,
            outcomes=(outcome,),
        )


def test_aggregate_binding_document_mismatch_rejected() -> None:
    case = _summarize_case()
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(document_id=DOC_B),
        )
    )
    with pytest.raises(ContractValidationError):
        _validate(case, bindings=bindings)


def test_aggregate_binding_revision_mismatch_rejected() -> None:
    case = _summarize_case()
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(document_revision="rev-9"),
        )
    )
    with pytest.raises(ContractValidationError):
        _validate(case, bindings=bindings)


def test_foreign_slot_rejected() -> None:
    """A selection whose slots are not the checkpointed slots cannot expand."""
    case = _summarize_case()
    foreign_slot = factories.target_slot(
        slot_id="slot-9", explicit_binding_ids=()
    )
    selection = factories.target_selection(
        target_slots=(foreign_slot,),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-9",
                selections=(
                    SelectedBindingRef(
                        target_id="t1",
                        binding_id="b1",
                        selected_aggregate_id=case.aggregate.aggregate_id,
                        authority="confidence_policy",
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ContractValidationError):
        _validate(case, selection=selection)


def test_checkpoint_not_selected_rejected() -> None:
    case = _summarize_case()
    for status in ("searching", "ranking", "unavailable"):
        checkpoint = case.checkpoint.model_copy(update={"status": status})
        with pytest.raises(ContractValidationError):
            _validate(case, checkpoint=checkpoint)


# ---------------------------------------------------------------------------
# Invariants 7-9: id uniqueness, catalog, expansion-origin correspondence
# ---------------------------------------------------------------------------


def test_appended_task_shadows_bootstrap_task_id_rejected() -> None:
    case = _summarize_case()
    forged = case.appended_tasks[0].model_copy(update={"task_id": "T-search"})
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, tasks=(forged,)))


def test_duplicate_appended_target_id_rejected() -> None:
    case = _summarize_case()
    duplicate = case.appended_units[0].model_copy(update={"binding_id": "b1"})
    with pytest.raises(ContractValidationError):
        _validate(
            case,
            _proposed(case, units=case.appended_units + (duplicate,)),
        )


def test_appended_task_references_unknown_target_rejected() -> None:
    case = _summarize_case()
    forged = _expansion_read(
        "T-read",
        ("t9",),
        ("T-search",),
        ("slot-1",),
        (case.aggregate.aggregate_id,),
    )
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, tasks=(forged,)))


def test_appended_targetless_task_rejected() -> None:
    """A targetless catalog task has no selected-unit lineage and cannot be
    appended under a discovery-expansion origin (invariant 8)."""
    case = _summarize_case()
    forged = TaskSpec(
        task_id="T-rogue",
        capability="document.retrieve",
        task_objective="workspace retrieval outside selected units",
        input=DocumentRetrieveInput(
            kind="document.retrieve", query="outside selection", target_ids=(), top_k=1
        ),
        depends_on=(),
        origin=DiscoveryExpansionTaskOrigin(
            kind="discovery_expansion",
            source_search_task_ids=(),
            target_slot_ids=(),
            selected_aggregate_ids=(),
        ),
    )
    with pytest.raises(ContractValidationError):
        _validate(
            case,
            _proposed(case, tasks=(forged,)),
            capability_names=CAPABILITIES | {"document.retrieve"},
        )


def test_appended_task_references_selected_but_not_appended_unit() -> None:
    """New tasks may read only appended units, not a selected ref whose unit
    was never appended (invariant 8)."""
    case = _summarize_case()
    # A second selected ref exists in the selection for another binding, but
    # no unit is appended for it.
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(),
            factories.scoped_document(binding_id="b2", document_id=DOC_B),
        )
    )
    other_match = ProbeCandidateMatch(
        probe_id="p1",
        slot_id="slot-1",
        task_id="T-search",
        candidate_id=uuid4(),
        document_id=DOC_B,
        document_revision=REV,
        rank=2,
        confidence=0.85,
        match_kind="semantic",
        calibration_version="cal-1",
    )
    matches = case.checkpoint.candidate_matches + (other_match,)
    aggregates = aggregate_matches(
        DISCOVERY_ID, matches, case.checkpoint.accepted_probes, (case.slot,)
    )
    other = next(
        aggregate
        for aggregate in aggregates
        if aggregate.aggregate_id != case.aggregate.aggregate_id
    )
    slot = case.slot.model_copy(update={"max_selections": 2})
    checkpoint = case.checkpoint.model_copy(
        update={
            "target_slots": (slot,),
            "candidate_matches": matches,
            "slot_aggregates": aggregates,
            "selections": (
                DiscoverySelection(
                    slot_id="slot-1",
                    selected_aggregate_ids=(
                        case.aggregate.aggregate_id,
                        other.aggregate_id,
                    ),
                    authority="confidence_policy",
                ),
            ),
        }
    )
    selection = factories.target_selection(
        target_slots=(slot,),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    SelectedBindingRef(
                        target_id="t1",
                        binding_id="b1",
                        selected_aggregate_id=case.aggregate.aggregate_id,
                        authority="confidence_policy",
                    ),
                    SelectedBindingRef(
                        target_id="t2",
                        binding_id="b2",
                        selected_aggregate_id=other.aggregate_id,
                        authority="confidence_policy",
                    ),
                ),
            ),
        ),
    )
    # The task reads t2, which is selected but has no appended unit.
    forged = _expansion_read(
        "T-read",
        ("t2",),
        ("T-search",),
        ("slot-1",),
        (other.aggregate_id,),
    )
    outcome = _search_result()
    second_data = outcome.data.model_copy(
        update={
            "candidates": outcome.data.candidates
            + (
                DocumentDiscoveryCandidate(
                    candidate_id=other_match.candidate_id,
                    document_id=DOC_B,
                    document_revision=REV,
                ),
            ),
            "identity_matches": outcome.data.identity_matches
            + (
                DocumentIdentityMatch(
                    candidate_id=other_match.candidate_id,
                    document_id=DOC_B,
                    document_revision=REV,
                    rank=2,
                    confidence=0.85,
                    match_kind="semantic",
                    calibration_version="cal-1",
                ),
            ),
        }
    )
    outcome = outcome.model_copy(update={"data": second_data})
    proposed = _proposed(case, tasks=(forged,))
    with pytest.raises(ContractValidationError):
        _validate(
            case,
            proposed,
            checkpoint=checkpoint,
            selection=selection,
            bindings=bindings,
            outcomes=(outcome,),
        )

    unread_unit = factories.target_unit(target_id="t2", binding_id="b2")
    with pytest.raises(ContractValidationError, match="have no factual task"):
        _validate(
            case,
            _proposed(
                case,
                units=case.appended_units + (unread_unit,),
            ),
            checkpoint=checkpoint,
            selection=selection,
            bindings=bindings,
            outcomes=(outcome,),
        )


def test_forged_capability_rejected() -> None:
    case = _summarize_case()
    with pytest.raises(ContractValidationError):
        _validate(case, capability_names=frozenset({"document.search"}))


@pytest.mark.parametrize(
    "origin",
    [
        InitialTaskOrigin(kind="initial"),
        ReplanTaskOrigin(
            kind="replan",
            reason="smuggled",
            task_ids=("T-search",),
            evidence_use_ids=(),
        ),
        DiscoveryProbeTaskOrigin(
            kind="discovery_probe",
            probe_id="p1",
            slot_id="slot-1",
            round=1,
        ),
    ],
)
def test_appended_task_without_expansion_origin_rejected(origin: object) -> None:
    case = _summarize_case()
    forged = case.appended_tasks[0].model_copy(update={"origin": origin})
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, tasks=(forged,)))


def test_origin_slot_mismatch_rejected() -> None:
    case = _summarize_case()
    forged = _expansion_read(
        "T-read",
        ("t1",),
        ("T-search",),
        ("slot-9",),
        (case.aggregate.aggregate_id,),
    )
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, tasks=(forged,)))


def test_origin_aggregate_mismatch_rejected() -> None:
    case = _summarize_case()
    forged = _expansion_read(
        "T-read",
        ("t1",),
        ("T-search",),
        ("slot-1",),
        (uuid4(),),
    )
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, tasks=(forged,)))


def test_non_bootstrap_source_task_rejected() -> None:
    """A non-probe task cannot be named as a discovery source task."""
    case = _summarize_case()
    other = TaskSpec(
        task_id="T-other",
        capability="document.search",
        task_objective="not a probe",
        input=DocumentSearchInput(kind="document.search", query="nđ13"),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )
    current = case.current.model_copy(
        update={"tasks": case.current.tasks + (other,)}
    )
    forged = _expansion_read(
        "T-read",
        ("t1",),
        ("T-search", "T-other"),
        ("slot-1",),
        (case.aggregate.aggregate_id,),
    )
    with pytest.raises(ContractValidationError):
        _validate(
            replace(case, current=current),
            _proposed(replace(case, current=current), tasks=(forged,)),
        )


def test_unknown_source_task_rejected() -> None:
    case = _summarize_case()
    forged = _expansion_read(
        "T-read",
        ("t1",),
        ("T-ghost",),
        ("slot-1",),
        (case.aggregate.aggregate_id,),
    )
    with pytest.raises(ContractValidationError):
        _validate(case, _proposed(case, tasks=(forged,)))


# ---------------------------------------------------------------------------
# Invariant 10: typed outcome coverage, uniqueness, success for selected
# ---------------------------------------------------------------------------


def test_missing_outcome_rejected() -> None:
    case = _summarize_case()
    with pytest.raises(ContractValidationError):
        _validate(case, outcomes=())


def test_duplicate_outcome_rejected() -> None:
    case = _summarize_case()
    with pytest.raises(ContractValidationError):
        _validate(case, outcomes=case.outcomes + case.outcomes)


def test_outcome_for_unknown_task_rejected() -> None:
    case = _summarize_case()
    forged = _search_result("T-ghost")
    with pytest.raises(ContractValidationError):
        _validate(case, outcomes=case.outcomes + (forged,))


def test_non_probe_current_task_rejected() -> None:
    """The checkpointed bootstrap plan itself must be probe-only (§13.1)."""
    case = _summarize_case()
    ordinary_search = TaskSpec(
        task_id="T-ordinary-search",
        capability="document.search",
        task_objective="ordinary search",
        input=DocumentSearchInput(kind="document.search", query="ordinary"),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )
    current = case.current.model_copy(
        update={"tasks": case.current.tasks + (ordinary_search,)}
    )
    with pytest.raises(
        ContractValidationError,
        match="non-probe task",
    ):
        _validate(replace(case, current=current))


def test_failed_source_outcome_rejected() -> None:
    """A failed outcome cannot back a selected aggregate member."""
    case = _summarize_case()
    with pytest.raises(ContractValidationError):
        _validate(case, outcomes=(_search_result(status="error"),))


def test_outcome_identity_mismatch_rejected() -> None:
    """The outcome's identity matches must reproduce each aggregate member on
    (task_id, document_id, document_revision)."""
    case = _summarize_case()
    wrong = _search_result(document_id=DOC_B)
    with pytest.raises(ContractValidationError):
        _validate(case, outcomes=(wrong,))


# ---------------------------------------------------------------------------
# Invariant 12: budgets
# ---------------------------------------------------------------------------


def test_research_budget_exceeded_rejected() -> None:
    case = _summarize_case()
    with pytest.raises(ContractValidationError):
        _validate(case, max_research_tasks=0)


def test_total_task_limit_exceeded_rejected() -> None:
    case = _summarize_case()
    with pytest.raises(ContractValidationError):
        _validate(case, total_task_limit=1)


def test_probe_tasks_do_not_consume_research_budget() -> None:
    """The probe stays free only while checkpoint-owned: a probe-only current
    plan plus one factual append fits ``max_research_tasks=1``."""
    case = _summarize_case()
    _validate(case, max_research_tasks=1, total_task_limit=13)


def test_spoofed_probe_origin_never_grants_free_capacity() -> None:
    """A ``document.read`` carrying a probe origin is excluded from the
    factual count but the plan still fails — the label alone is never
    authority."""
    case = _summarize_case()
    spoofed = TaskSpec(
        task_id="T-spoof",
        capability="document.read",
        task_objective="spoofed probe",
        input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
        depends_on=(),
        origin=DiscoveryProbeTaskOrigin(
            kind="discovery_probe",
            probe_id="p1",
            slot_id="slot-1",
            round=1,
        ),
    )
    current = case.current.model_copy(
        update={"tasks": case.current.tasks + (spoofed,)}
    )
    # Give the spoofed task a preserved non-success outcome so coverage is
    # complete; the plan must still fail — a probe label on a read task is
    # invalid and one accepted probe cannot be owned by two tasks.
    outcomes = case.outcomes + (_search_result("T-spoof", status="error"),)
    with pytest.raises(ContractValidationError):
        _validate(replace(case, current=current), outcomes=outcomes)

"""Discovery plan Task 4 — context-required discovery task origins."""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.contracts.capability import (
    DocumentReadInput,
    DocumentRetrieveInput,
    DocumentSearchInput,
)
from app.services.agents.v2.contracts.planning import (
    DiscoveryExpansionTaskOrigin,
    DiscoveryProbeTaskOrigin,
    InitialTaskOrigin,
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
    SearchProbe,
    SlotCandidateAggregate,
    SlotBindingSelection,
)
from app.services.agents.v2.discovery_bootstrap.validation import aggregate_id_for

from . import factories

DISCOVERY_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


def _probe() -> SearchProbe:
    return SearchProbe(
        probe_id="p1",
        slot_id="slot-1",
        query="nđ13",
        round=1,
        origin="semantic",
    )


def _probe_task(query: str = "nđ13", probe_id: str = "p1") -> TaskSpec:
    return TaskSpec(
        task_id="T-search",
        capability="document.search",
        task_objective="probe slot",
        input=DocumentSearchInput(kind="document.search", query=query),
        depends_on=(),
        origin=DiscoveryProbeTaskOrigin(
            kind="discovery_probe", probe_id=probe_id, slot_id="slot-1", round=1
        ),
    )


def _aggregate() -> SlotCandidateAggregate:
    return SlotCandidateAggregate(
        aggregate_id=aggregate_id_for(
            DISCOVERY_ID, "slot-1", factories.DOCUMENT_ID, factories.REVISION
        ),
        slot_id="slot-1",
        document_id=factories.DOCUMENT_ID,
        document_revision=factories.REVISION,
        source_candidate_ids=(uuid4(),),
        source_probe_ids=("p1",),
        source_task_ids=("T-search",),
        best_match_kind="semantic",
        aggregate_confidence=0.9,
        best_rank=1,
    )


def _checkpoint(aggregate: SlotCandidateAggregate | None = None) -> DiscoveryCheckpoint:
    if aggregate is None:
        aggregate = _aggregate()
    return DiscoveryCheckpoint(
        discovery_id=DISCOVERY_ID,
        target_slots=(factories.target_slot(),),
        accepted_probes=(_probe(),),
        candidate_matches=(),
        slot_aggregates=(aggregate,),
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


def _selection(aggregate: SlotCandidateAggregate) -> object:
    from app.services.agents.v2.discovery_bootstrap.contracts import (
        SelectedBindingRef,
    )

    return factories.target_selection(
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


def _plan(*tasks: TaskSpec, target_units=()) -> TaskPlan:
    return TaskPlan(
        contract_version="2.0",
        plan_id="plan-1",
        goal="bootstrap",
        target_units=target_units,
        tasks=tasks,
    )


def _expansion_plan(aggregate: SlotCandidateAggregate) -> TaskPlan:
    return _plan(
        _probe_task(),
        TaskSpec(
            task_id="T-read",
            capability="document.read",
            task_objective="read selected",
            input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
            depends_on=("T-search",),
            origin=DiscoveryExpansionTaskOrigin(
                kind="discovery_expansion",
                source_search_task_ids=("T-search",),
                target_slot_ids=("slot-1",),
                selected_aggregate_ids=(aggregate.aggregate_id,),
            ),
        ),
        target_units=(factories.target_unit(),),
    )


def test_probe_origin_valid_with_discovery_context() -> None:
    plan = _plan(_probe_task())
    validate_task_plan(
        plan, factories.binding_set(), discovery_checkpoint=_checkpoint()
    )


def test_probe_origin_requires_discovery_context() -> None:
    plan = _plan(_probe_task())
    with pytest.raises(ContractValidationError):
        validate_task_plan(plan, factories.binding_set())


def test_read_task_cannot_carry_probe_origin() -> None:
    task = TaskSpec(
        task_id="T-read",
        capability="document.read",
        task_objective="spoofed",
        input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
        depends_on=(),
        origin=DiscoveryProbeTaskOrigin(
            kind="discovery_probe", probe_id="p1", slot_id="slot-1", round=1
        ),
    )
    plan = _plan(task, target_units=(factories.target_unit(),))
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan, factories.binding_set(), discovery_checkpoint=_checkpoint()
        )


def test_probe_origin_unknown_probe_rejected() -> None:
    plan = _plan(_probe_task(probe_id="ghost"))
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan, factories.binding_set(), discovery_checkpoint=_checkpoint()
        )


def test_probe_origin_query_mismatch_rejected() -> None:
    plan = _plan(_probe_task(query="khác"))
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan, factories.binding_set(), discovery_checkpoint=_checkpoint()
        )


def test_probe_origin_slot_or_round_mismatch_rejected() -> None:
    task = _probe_task().model_copy(
        update={
            "origin": DiscoveryProbeTaskOrigin(
                kind="discovery_probe", probe_id="p1", slot_id="slot-1", round=2
            )
        }
    )
    plan = _plan(task)
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan, factories.binding_set(), discovery_checkpoint=_checkpoint()
        )


def test_one_probe_cannot_be_owned_by_two_tasks() -> None:
    second = _probe_task().model_copy(update={"task_id": "T-search-2"})
    plan = _plan(_probe_task(), second)
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan, factories.binding_set(), discovery_checkpoint=_checkpoint()
        )


def test_probe_task_cannot_carry_people_scalar() -> None:
    task = _probe_task().model_copy(
        update={
            "input": DocumentSearchInput(
                kind="document.search",
                query="nđ13",
                person_identifier="Nguyen Van A",
            )
        }
    )
    plan = _plan(task)
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan, factories.binding_set(), discovery_checkpoint=_checkpoint()
        )


def test_expansion_origin_valid_with_both_contexts() -> None:
    aggregate = _aggregate()
    plan = _expansion_plan(aggregate)
    validate_task_plan(
        plan,
        factories.binding_set(),
        target_selection=_selection(aggregate),
        discovery_checkpoint=_checkpoint(aggregate),
    )


def test_expansion_origin_requires_both_contexts() -> None:
    aggregate = _aggregate()
    plan = _expansion_plan(aggregate)
    with pytest.raises(ContractValidationError):
        validate_task_plan(plan, factories.binding_set())
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan,
            factories.binding_set(),
            discovery_checkpoint=_checkpoint(aggregate),
        )
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan,
            factories.binding_set(),
            target_selection=_selection(aggregate),
        )


def test_expansion_origin_unknown_source_task_rejected() -> None:
    aggregate = _aggregate()
    plan = _expansion_plan(aggregate).model_copy(
        update={
            "tasks": (
                _probe_task(),
                TaskSpec(
                    task_id="T-read",
                    capability="document.read",
                    task_objective="read selected",
                    input=DocumentReadInput(
                        kind="document.read", target_ids=("t1",)
                    ),
                    depends_on=(),
                    origin=DiscoveryExpansionTaskOrigin(
                        kind="discovery_expansion",
                        source_search_task_ids=("T-ghost",),
                        target_slot_ids=("slot-1",),
                        selected_aggregate_ids=(aggregate.aggregate_id,),
                    ),
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan,
            factories.binding_set(),
            target_selection=_selection(aggregate),
            discovery_checkpoint=_checkpoint(aggregate),
        )


def test_expansion_origin_non_probe_source_rejected() -> None:
    aggregate = _aggregate()
    other = TaskSpec(
        task_id="T-other",
        capability="document.search",
        task_objective="not a probe",
        input=DocumentSearchInput(kind="document.search", query="nđ13"),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )
    plan = _plan(
        _probe_task(),
        other,
        TaskSpec(
            task_id="T-read",
            capability="document.read",
            task_objective="read selected",
            input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
            depends_on=(),
            origin=DiscoveryExpansionTaskOrigin(
                kind="discovery_expansion",
                source_search_task_ids=("T-search", "T-other"),
                target_slot_ids=("slot-1",),
                selected_aggregate_ids=(aggregate.aggregate_id,),
            ),
        ),
        target_units=(factories.target_unit(),),
    )
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan,
            factories.binding_set(),
            target_selection=_selection(aggregate),
            discovery_checkpoint=_checkpoint(aggregate),
        )


def test_expansion_origin_lineage_must_match_selected_refs() -> None:
    aggregate = _aggregate()
    plan = _expansion_plan(aggregate).model_copy(
        update={
            "tasks": (
                _probe_task(),
                TaskSpec(
                    task_id="T-read",
                    capability="document.read",
                    task_objective="read selected",
                    input=DocumentReadInput(
                        kind="document.read", target_ids=("t1",)
                    ),
                    depends_on=(),
                    origin=DiscoveryExpansionTaskOrigin(
                        kind="discovery_expansion",
                        source_search_task_ids=("T-search",),
                        target_slot_ids=("slot-1",),
                        selected_aggregate_ids=(uuid4(),),
                    ),
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan,
            factories.binding_set(),
            target_selection=_selection(aggregate),
            discovery_checkpoint=_checkpoint(aggregate),
        )


def test_expansion_origin_self_reference_rejected() -> None:
    aggregate = _aggregate()
    plan = _plan(
        _probe_task(),
        TaskSpec(
            task_id="T-read",
            capability="document.read",
            task_objective="read selected",
            input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
            depends_on=(),
            origin=DiscoveryExpansionTaskOrigin(
                kind="discovery_expansion",
                source_search_task_ids=("T-search", "T-read"),
                target_slot_ids=("slot-1",),
                selected_aggregate_ids=(aggregate.aggregate_id,),
            ),
        ),
        target_units=(factories.target_unit(),),
    )
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan,
            factories.binding_set(),
            target_selection=_selection(aggregate),
            discovery_checkpoint=_checkpoint(aggregate),
        )


def test_targetless_expansion_task_rejected() -> None:
    aggregate = _aggregate()
    plan = _plan(
        _probe_task(),
        TaskSpec(
            task_id="T-rogue",
            capability="document.retrieve",
            task_objective="workspace retrieval outside selection",
            input=DocumentRetrieveInput(
                kind="document.retrieve",
                query="outside selection",
                target_ids=(),
                top_k=1,
            ),
            depends_on=(),
            origin=DiscoveryExpansionTaskOrigin(
                kind="discovery_expansion",
                source_search_task_ids=(),
                target_slot_ids=(),
                selected_aggregate_ids=(),
            ),
        ),
        target_units=(factories.target_unit(),),
    )
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan,
            factories.binding_set(),
            target_selection=_selection(aggregate),
            discovery_checkpoint=_checkpoint(aggregate),
        )


def test_explicit_only_selection_yields_empty_lineage() -> None:
    from app.services.agents.v2.discovery_bootstrap.contracts import (
        SelectedBindingRef,
    )

    selection = factories.target_selection(
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    SelectedBindingRef(
                        target_id="t1",
                        binding_id="b1",
                        selected_aggregate_id=None,
                        authority="explicit_binding",
                    ),
                ),
            ),
        ),
    )
    checkpoint = DiscoveryCheckpoint(
        discovery_id=DISCOVERY_ID,
        target_slots=(factories.target_slot(),),
        accepted_probes=(_probe(),),
        candidate_matches=(),
        slot_aggregates=(),
        selections=(),
        rounds_consumed=1,
        probes_consumed=1,
        status="selected",
        clarification_manifest=None,
    )
    plan = _plan(
        _probe_task(),
        TaskSpec(
            task_id="T-read",
            capability="document.read",
            task_objective="read selected",
            input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
            depends_on=("T-search",),
            origin=DiscoveryExpansionTaskOrigin(
                kind="discovery_expansion",
                source_search_task_ids=(),
                target_slot_ids=("slot-1",),
                selected_aggregate_ids=(),
            ),
        ),
        target_units=(factories.target_unit(),),
    )
    validate_task_plan(
        plan,
        factories.binding_set(),
        target_selection=selection,
        discovery_checkpoint=checkpoint,
    )

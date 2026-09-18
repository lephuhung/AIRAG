"""Discovery plan Task 4 — selection-aware plan and replan validation."""
from __future__ import annotations

import pytest

from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    DocumentReadInput,
)
from app.services.agents.v2.contracts.evaluation import (
    Coverage,
    EvidenceEvaluation,
)
from app.services.agents.v2.contracts.locators import SectionLocator
from app.services.agents.v2.contracts.planning import (
    DiscoveryPolicy,
    ReplanTaskOrigin,
    ResearchBudgetView,
    ResearchPlanningInput,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.routing import QueryAnalysis
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_replan,
    validate_research_planning_input,
    validate_task_plan,
)
from app.services.agents.v2.discovery_bootstrap.contracts import (
    SelectedBindingRef,
    SlotBindingSelection,
)

from . import factories


def _selected_plan(binding_id: str = "b1") -> TaskPlan:
    return TaskPlan(
        contract_version="2.0",
        plan_id="plan-1",
        goal="summarize",
        target_units=(factories.target_unit(binding_id=binding_id),),
        tasks=(factories.read_task(target_ids=("t1",)),),
    )


def _selection(**kwargs) -> object:
    return factories.target_selection(**kwargs)


def test_explicit_selection_without_discovery_plans() -> None:
    selection = _selection()
    validate_task_plan(
        _selected_plan(), factories.binding_set(), target_selection=selection
    )


def test_supporting_binding_accepted_only_through_selection() -> None:
    bindings = factories.binding_set(
        bindings=(factories.scoped_document(role="supporting"),)
    )
    with pytest.raises(ContractValidationError):
        validate_task_plan(_selected_plan(), bindings)
    validate_task_plan(
        _selected_plan(), bindings, target_selection=_selection()
    )


def test_context_binding_cannot_authorize_target_unit() -> None:
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(),
            factories.scoped_document(
                binding_id="b2",
                document_id=factories.OTHER_DOCUMENT_ID,
                role="supporting",
            ),
        )
    )
    selection = _selection(context_binding_ids=("b2",))
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            _selected_plan(binding_id="b2"),
            bindings,
            target_selection=selection,
        )


def test_target_unit_binding_mismatch_rejected() -> None:
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(),
            factories.scoped_document(
                binding_id="b2", document_id=factories.OTHER_DOCUMENT_ID
            ),
        )
    )
    selection = _selection(
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    factories.selected_binding_ref(binding_id="b2"),
                ),
            ),
        ),
        target_slots=(
            factories.target_slot(explicit_binding_ids=("b2",)),
        ),
    )
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            _selected_plan(), bindings, target_selection=selection
        )


def test_target_unit_locator_mismatch_rejected() -> None:
    selection = _selection()
    plan = _selected_plan().model_copy(
        update={
            "target_units": (
                factories.target_unit(
                    requested_locator=SectionLocator(
                        kind="section", structure_node_id="n1"
                    )
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan, factories.binding_set(), target_selection=selection
        )


def test_unknown_target_id_rejected_under_selection() -> None:
    selection = _selection()
    plan = _selected_plan().model_copy(
        update={
            "target_units": (
                factories.target_unit(target_id="t9"),
            ),
            "tasks": (factories.read_task(target_ids=("t9",)),),
        }
    )
    with pytest.raises(ContractValidationError):
        validate_task_plan(
            plan, factories.binding_set(), target_selection=selection
        )


def test_selected_ref_without_target_unit_rejected() -> None:
    plan = _selected_plan().model_copy(
        update={"target_units": (), "tasks": ()}
    )
    with pytest.raises(ContractValidationError, match="have no TargetUnit"):
        validate_task_plan(
            plan, factories.binding_set(), target_selection=_selection()
        )


def test_strict_default_without_selection_unchanged() -> None:
    plan = factories.read_plan()
    validate_task_plan(plan, factories.binding_set())


def test_replan_with_contexts_validates_expanded_plan() -> None:
    bindings = factories.binding_set()
    selection = _selection()
    current = _selected_plan()
    new_task = TaskSpec(
        task_id="T2",
        capability="document.read",
        task_objective="extra read",
        input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
        depends_on=("T1",),
        origin=ReplanTaskOrigin(
            kind="replan", reason="more", task_ids=("T1",), evidence_use_ids=()
        ),
    )
    proposed = current.model_copy(
        update={"tasks": current.tasks + (new_task,)}
    )
    accepted = validate_replan(
        current,
        proposed,
        (),
        DiscoveryPolicy(
            allow_reference_discovery=False,
            allow_supporting_discovery=False,
            max_discovered_documents=0,
        ),
        ResearchBudgetView(
            max_tasks_remaining=5,
            max_replans_remaining=1,
            max_parallel_branches=2,
        ),
        bindings=bindings,
        target_selection=selection,
    )
    assert accepted is proposed


def test_replan_contexts_require_bindings() -> None:
    current = _selected_plan()
    proposed = current.model_copy(
        update={
            "tasks": current.tasks
            + (
                TaskSpec(
                    task_id="T2",
                    capability="document.read",
                    task_objective="extra",
                    input=DocumentReadInput(
                        kind="document.read", target_ids=("t1",)
                    ),
                    depends_on=(),
                    origin=ReplanTaskOrigin(
                        kind="replan",
                        reason="more",
                        task_ids=(),
                        evidence_use_ids=(),
                    ),
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_replan(
            current,
            proposed,
            (),
            DiscoveryPolicy(
                allow_reference_discovery=False,
                allow_supporting_discovery=False,
                max_discovered_documents=0,
            ),
            ResearchBudgetView(
                max_tasks_remaining=5,
                max_replans_remaining=1,
                max_parallel_branches=2,
            ),
            target_selection=_selection(),
        )


def test_replan_cannot_masquerade_as_discovery_task() -> None:
    from app.services.agents.v2.contracts.planning import DiscoveryProbeTaskOrigin
    from app.services.agents.v2.contracts.capability import DocumentSearchInput

    current = _selected_plan()
    proposed = current.model_copy(
        update={
            "tasks": current.tasks
            + (
                TaskSpec(
                    task_id="T2",
                    capability="document.search",
                    task_objective="probe",
                    input=DocumentSearchInput(
                        kind="document.search", query="q"
                    ),
                    depends_on=(),
                    origin=DiscoveryProbeTaskOrigin(
                        kind="discovery_probe",
                        probe_id="p1",
                        slot_id="slot-1",
                        round=1,
                    ),
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError):
        validate_replan(
            current,
            proposed,
            (),
            DiscoveryPolicy(
                allow_reference_discovery=True,
                allow_supporting_discovery=True,
                max_discovered_documents=3,
            ),
            ResearchBudgetView(
                max_tasks_remaining=5,
                max_replans_remaining=1,
                max_parallel_branches=2,
            ),
            bindings=factories.binding_set(),
            target_selection=_selection(),
        )


def test_replan_over_discovery_expanded_plan() -> None:
    from uuid import UUID, uuid4

    from app.services.agents.v2.contracts.capability import DocumentSearchInput
    from app.services.agents.v2.contracts.planning import (
        DiscoveryExpansionTaskOrigin,
        DiscoveryProbeTaskOrigin,
    )
    from app.services.agents.v2.discovery_bootstrap.contracts import (
        DiscoveryCheckpoint,
        DiscoverySelection,
        SearchProbe,
        SlotCandidateAggregate,
    )
    from app.services.agents.v2.discovery_bootstrap.validation import (
        aggregate_id_for,
    )

    discovery_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    aggregate = SlotCandidateAggregate(
        aggregate_id=aggregate_id_for(
            discovery_id, "slot-1", factories.DOCUMENT_ID, factories.REVISION
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
    checkpoint = DiscoveryCheckpoint(
        discovery_id=discovery_id,
        target_slots=(factories.target_slot(),),
        accepted_probes=(
            SearchProbe(
                probe_id="p1",
                slot_id="slot-1",
                query="nđ13",
                round=1,
                origin="semantic",
            ),
        ),
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
    selection = _selection(
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
    current = TaskPlan(
        contract_version="2.0",
        plan_id="plan-1",
        goal="summarize",
        target_units=(factories.target_unit(),),
        tasks=(
            TaskSpec(
                task_id="T-search",
                capability="document.search",
                task_objective="probe slot",
                input=DocumentSearchInput(kind="document.search", query="nđ13"),
                depends_on=(),
                origin=DiscoveryProbeTaskOrigin(
                    kind="discovery_probe",
                    probe_id="p1",
                    slot_id="slot-1",
                    round=1,
                ),
            ),
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
        ),
    )
    appended = TaskSpec(
        task_id="T-extra",
        capability="document.read",
        task_objective="extra read",
        input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
        depends_on=("T-read",),
        origin=ReplanTaskOrigin(
            kind="replan", reason="more", task_ids=(), evidence_use_ids=()
        ),
    )
    proposed = current.model_copy(update={"tasks": current.tasks + (appended,)})
    policy = DiscoveryPolicy(
        allow_reference_discovery=False,
        allow_supporting_discovery=False,
        max_discovered_documents=0,
    )
    budget = ResearchBudgetView(
        max_tasks_remaining=5, max_replans_remaining=1, max_parallel_branches=2
    )
    accepted = validate_replan(
        current,
        proposed,
        (),
        policy,
        budget,
        bindings=factories.binding_set(),
        target_selection=selection,
        discovery_checkpoint=checkpoint,
    )
    assert accepted is proposed
    with pytest.raises(ContractValidationError):
        validate_replan(current, proposed, (), policy, budget)


def test_planning_input_threads_contexts() -> None:
    bindings = factories.binding_set()
    selection = _selection()
    planning_input = ResearchPlanningInput(
        semantic=factories.semantic_context(),
        bindings=bindings,
        query_analysis=QueryAnalysis(
            work_type="summarize", domains=("document",)
        ),
        capability_catalog=(
            CapabilityDescriptor(
                name="document.read",
                domain="document",
                operation_type="read",
                supports_parallel=True,
            ),
        ),
        discovery_policy=DiscoveryPolicy(
            allow_reference_discovery=False,
            allow_supporting_discovery=False,
            max_discovered_documents=0,
        ),
        budget=ResearchBudgetView(
            max_tasks_remaining=5,
            max_replans_remaining=1,
            max_parallel_branches=2,
        ),
        current_plan=_selected_plan(),
        prior_evaluation=EvidenceEvaluation(
            status="insufficient",
            coverage=Coverage(items=()),
            missing=(),
            contradictions=(),
        ),
        target_selection=selection,
    )
    validate_research_planning_input(planning_input)


def test_planning_input_selection_work_type_must_match_analysis() -> None:
    planning_input = ResearchPlanningInput(
        semantic=factories.semantic_context(),
        bindings=factories.binding_set(),
        query_analysis=QueryAnalysis(
            work_type="compare", domains=("document",)
        ),
        capability_catalog=(
            CapabilityDescriptor(
                name="document.read",
                domain="document",
                operation_type="read",
                supports_parallel=True,
            ),
        ),
        discovery_policy=DiscoveryPolicy(
            allow_reference_discovery=False,
            allow_supporting_discovery=False,
            max_discovered_documents=0,
        ),
        budget=ResearchBudgetView(
            max_tasks_remaining=5,
            max_replans_remaining=1,
            max_parallel_branches=2,
        ),
        target_selection=_selection(),
    )
    with pytest.raises(ContractValidationError):
        validate_research_planning_input(planning_input)

"""Discovery plan Task 6 — selection-aware policies and child-state plumbing.

Dormant groundwork only: summarize/compare skills plan from a checkpointed
``ResearchTargetSelection`` when one is present (legacy role-scan path is
byte-for-byte unchanged when it is absent), the complex child carries the
discovery/selection models end to end, and the governed entries pass both
contexts into the frozen validators. Model-facing projections never gain
selection or discovery data.
"""
from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
)
from app.services.agents.v2.contracts.locators import (
    DocumentLocator,
    SectionLocator,
)
from app.services.agents.v2.contracts.planning import (
    DiscoveryPolicy,
    ResearchBudgetView,
    ResearchPlanningInput,
)
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.validation import ContractValidationError
from app.services.agents.v2.discovery_bootstrap.contracts import (
    DiscoveryCheckpoint,
    ResearchTargetSelection,
    SelectedBindingRef,
    SlotBindingSelection,
)
from app.services.agents.v2.planning import build_planner_model_input
from app.services.agents.v2.planning.replan_projection import (
    build_replanner_model_input_from_envelope,
)

from tests.agents.v2.contracts import factories


USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
AGGREGATE_ID = UUID("99999999-9999-9999-9999-999999999999")
DISCOVERY_ID = UUID("88888888-8888-8888-8888-888888888888")


def _semantic(query: str = "Tóm tắt tài liệu đã chọn") -> SemanticContext:
    return SemanticContext(
        contextualized_query=query,
        normalized_query=query.lower(),
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def _analysis(work_type: str = "summarize") -> QueryAnalysis:
    return QueryAnalysis(
        work_type=work_type,  # type: ignore[arg-type]
        domains=("document",),  # type: ignore[arg-type]
    )


def _catalog(names: frozenset[str]) -> tuple[CapabilityDescriptor, ...]:
    return tuple(
        CapabilityDescriptor(
            name=name,  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=True,
        )
        for name in sorted(names)
    )


def _planning_input(
    work_type: str = "summarize",
    *,
    bindings: DocumentBindingSet | None = None,
    selection: ResearchTargetSelection | None = None,
    discovery: DiscoveryCheckpoint | None = None,
    capability_names: frozenset[str] = frozenset(
        {"document.read", "section.read", "document.retrieve"}
    ),
    max_tasks: int = 8,
) -> ResearchPlanningInput:
    return ResearchPlanningInput(
        semantic=_semantic(),
        bindings=bindings if bindings is not None else factories.binding_set(),
        query_analysis=_analysis(work_type),
        capability_catalog=_catalog(capability_names),
        discovery_policy=DiscoveryPolicy(
            allow_reference_discovery=False,
            allow_supporting_discovery=False,
            max_discovered_documents=0,
        ),
        budget=ResearchBudgetView(
            max_tasks_remaining=max_tasks,
            max_replans_remaining=0,
            max_parallel_branches=4,
        ),
        target_selection=selection,
        discovery_checkpoint=discovery,
    )


def _discovery(target_slots: tuple = ()) -> DiscoveryCheckpoint:
    return DiscoveryCheckpoint(
        discovery_id=DISCOVERY_ID,
        target_slots=target_slots,
        accepted_probes=(),
        candidate_matches=(),
        slot_aggregates=(),
        selections=(),
        rounds_consumed=0,
        probes_consumed=0,
        status="searching",
        clarification_manifest=None,
    )


# ---------------------------------------------------------------------------
# Summarize: selection branch
# ---------------------------------------------------------------------------


def test_summarize_explicit_selection_plans_without_discovery() -> None:
    from app.services.agents.v2.skills.summarize import policy as summarize_policy

    workflow = summarize_policy.build_summarize_workflow(
        _planning_input(selection=factories.target_selection())
    )
    plan = workflow.plan
    assert plan.plan_id == "summarize-selection-t1"
    assert [unit.target_id for unit in plan.target_units] == ["t1"]
    assert plan.target_units[0].binding_id == "b1"
    assert [task.task_id for task in plan.tasks] == ["T1"]
    assert plan.tasks[0].capability == "document.read"
    assert workflow.reduce.map_task_ids == ("T1",)
    assert workflow.reduce.mode == "extractive"


def test_summarize_context_binding_never_becomes_target() -> None:
    from app.services.agents.v2.skills.summarize import policy as summarize_policy

    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(role="reference"),
            factories.scoped_document(
                binding_id="b2",
                document_id=factories.OTHER_DOCUMENT_ID,
            ),
        )
    )
    selection = factories.target_selection(
        context_binding_ids=("b1",),
        target_slots=(factories.target_slot(explicit_binding_ids=("b2",)),),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(
                    factories.selected_binding_ref(
                        target_id="t7", binding_id="b2"
                    ),
                ),
            ),
        ),
    )
    workflow = summarize_policy.build_summarize_workflow(
        _planning_input(bindings=bindings, selection=selection)
    )
    assert [unit.binding_id for unit in workflow.plan.target_units] == ["b2"]
    assert workflow.plan.plan_id == "summarize-selection-t7"


def test_summarize_discovered_binding_accepted_only_via_selection() -> None:
    from app.services.agents.v2.skills.summarize import policy as summarize_policy

    bindings = factories.binding_set(
        bindings=(factories.scoped_document(role="discovered"),)
    )
    with pytest.raises(ContractValidationError):
        summarize_policy.build_summarize_workflow(
            _planning_input(bindings=bindings)
        )
    workflow = summarize_policy.build_summarize_workflow(
        _planning_input(
            bindings=bindings, selection=factories.target_selection()
        )
    )
    assert workflow.plan.target_units[0].binding_id == "b1"


def test_summarize_reference_slot_gets_partial_coverage() -> None:
    from app.services.agents.v2.skills.summarize import policy as summarize_policy

    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(),
            factories.scoped_document(
                binding_id="b2",
                document_id=factories.OTHER_DOCUMENT_ID,
                role="reference",
            ),
        )
    )
    selection = factories.target_selection(
        target_slots=(
            factories.target_slot(),
            factories.target_slot(
                slot_id="slot-2",
                intended_role="reference",
                explicit_binding_ids=("b2",),
            ),
        ),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(factories.selected_binding_ref(),),
            ),
            SlotBindingSelection(
                slot_id="slot-2",
                selections=(
                    factories.selected_binding_ref(
                        target_id="t2", binding_id="b2"
                    ),
                ),
            ),
        ),
    )
    workflow = summarize_policy.build_summarize_workflow(
        _planning_input(bindings=bindings, selection=selection)
    )
    plan = workflow.plan
    assert plan.plan_id == "summarize-selection-t1-t2"
    assert [task.task_id for task in plan.tasks] == ["T1", "T2"]
    target_criterion = plan.target_units[0].completion_criteria[0]
    reference_criterion = plan.target_units[1].completion_criteria[0]
    assert target_criterion.minimum_status == "read_complete"
    assert reference_criterion.minimum_status == "read_partial"
    assert reference_criterion.allow_partial_reason == "reference context"


def test_summarize_section_locator_selection_uses_section_read() -> None:
    from app.services.agents.v2.skills.summarize import policy as summarize_policy

    selection = factories.target_selection(
        target_slots=(
            factories.target_slot(
                requested_locator=SectionLocator(
                    kind="section", structure_node_id="n1"
                )
            ),
        ),
    )
    workflow = summarize_policy.build_summarize_workflow(
        _planning_input(selection=selection)
    )
    assert workflow.plan.tasks[0].capability == "section.read"


def test_summarize_selection_wrong_work_type_rejected() -> None:
    from app.services.agents.v2.skills.summarize import policy as summarize_policy

    selection = factories.target_selection(
        work_type="compare",
        target_slots=(
            factories.target_slot(),
            factories.target_slot(
                slot_id="slot-2",
                intended_role="reference",
                explicit_binding_ids=("b1",),
            ),
        ),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(factories.selected_binding_ref(),),
            ),
            SlotBindingSelection(
                slot_id="slot-2",
                selections=(
                    factories.selected_binding_ref(
                        target_id="t2", binding_id="b1"
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ContractValidationError):
        summarize_policy.build_summarize_workflow(
            _planning_input(selection=selection)
        )


# ---------------------------------------------------------------------------
# Compare: selection branch
# ---------------------------------------------------------------------------


def _compare_selection(
    *,
    target_locator=None,
    reference_locator=None,
    bindings_explicit: bool = True,
) -> ResearchTargetSelection:
    return factories.target_selection(
        work_type="compare",
        target_slots=(
            factories.target_slot(
                requested_locator=target_locator,
                explicit_binding_ids=("b1",),
            ),
            factories.target_slot(
                slot_id="slot-2",
                intended_role="reference",
                requested_locator=reference_locator,
                explicit_binding_ids=("b1",) if bindings_explicit else ("b2",),
            ),
        ),
        slot_bindings=(
            SlotBindingSelection(
                slot_id="slot-1",
                selections=(factories.selected_binding_ref(),),
            ),
            SlotBindingSelection(
                slot_id="slot-2",
                selections=(
                    factories.selected_binding_ref(
                        target_id="t2",
                        binding_id="b1" if bindings_explicit else "b2",
                    ),
                ),
            ),
        ),
    )


def test_compare_same_binding_two_section_locators() -> None:
    from app.services.agents.v2.skills.compare import policy as compare_policy

    selection = _compare_selection(
        target_locator=SectionLocator(kind="section", structure_node_id="n1"),
        reference_locator=SectionLocator(kind="section", structure_node_id="n2"),
    )
    plan = compare_policy.build_compare_plan(
        _planning_input("compare", selection=selection)
    )
    assert plan.plan_id == "compare-selection-t1-t2"
    assert [unit.target_id for unit in plan.target_units] == ["t1", "t2"]
    assert {unit.binding_id for unit in plan.target_units} == {"b1"}
    assert {task.capability for task in plan.tasks} == {"section.read"}
    assert [task.task_id for task in plan.tasks] == ["T1", "T2"]


def test_compare_same_binding_same_locator_fails() -> None:
    from app.services.agents.v2.skills.compare import policy as compare_policy

    locator = SectionLocator(kind="section", structure_node_id="n1")
    selection = _compare_selection(
        target_locator=locator, reference_locator=locator
    )
    with pytest.raises(ContractValidationError):
        compare_policy.build_compare_plan(
            _planning_input("compare", selection=selection)
        )


def test_compare_two_bindings_two_slots() -> None:
    from app.services.agents.v2.skills.compare import policy as compare_policy

    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(),
            factories.scoped_document(
                binding_id="b2",
                document_id=factories.OTHER_DOCUMENT_ID,
                role="reference",
            ),
        )
    )
    selection = _compare_selection(bindings_explicit=False)
    plan = compare_policy.build_compare_plan(
        _planning_input("compare", bindings=bindings, selection=selection)
    )
    assert [unit.binding_id for unit in plan.target_units] == ["b1", "b2"]
    assert {task.capability for task in plan.tasks} == {"document.read"}


def test_compare_selection_wrong_work_type_rejected() -> None:
    from app.services.agents.v2.skills.compare import policy as compare_policy

    with pytest.raises(ContractValidationError):
        compare_policy.build_compare_plan(
            _planning_input(
                "compare", selection=factories.target_selection()
            )
        )


# ---------------------------------------------------------------------------
# Fallback composition: selection suppresses the targetless fallback
# ---------------------------------------------------------------------------


def test_fallback_suppressed_for_summarize_with_selection() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_initial_proposal,
    )

    with pytest.raises(ContractValidationError):
        build_initial_proposal(
            _planning_input(
                "summarize",
                selection=factories.target_selection(work_type="compare"),
            )
        )


def test_fallback_suppressed_for_compare_with_selection() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_initial_proposal,
    )

    with pytest.raises(ContractValidationError):
        build_initial_proposal(
            _planning_input(
                "compare",
                selection=factories.target_selection(),
            )
        )


def test_fallback_preserved_for_summarize_without_selection() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_initial_proposal,
    )

    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(),
            factories.scoped_document(
                binding_id="b2",
                document_id=factories.OTHER_DOCUMENT_ID,
                role="reference",
            ),
        )
    )
    proposal = build_initial_proposal(
        _planning_input("summarize", bindings=bindings)
    )
    assert proposal.plan.plan_id == "retrieve-unscoped"
    assert proposal.reduce_spec is None


def test_fallback_preserved_for_other_work_types() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_initial_proposal,
    )

    proposal = build_initial_proposal(
        _planning_input(
            "evaluate",
            bindings=factories.binding_set(bindings=()),
        )
    )
    assert proposal.plan.plan_id == "retrieve-unscoped"


# ---------------------------------------------------------------------------
# Parent -> child -> normalize -> merge: both models round-trip
# ---------------------------------------------------------------------------


def _parent_state(
    *,
    discovery: DiscoveryCheckpoint | None = None,
    selection: ResearchTargetSelection | None = None,
) -> SupervisorV2State:
    state = SupervisorV2State(
        contract_version="2.0",
        request=RequestContext(
            contract_version="2.0",
            request_id="req-select-1",
            thread_id="thread-select-1",
            original_query="Tóm tắt tài liệu đã chọn",
            known_documents=(),
        ),
        conversation=factories.conversation_context(),
        semantic=_semantic(),
        bindings=factories.binding_set(),
        query_analysis=_analysis(),
        route_decision=RouteDecision(
            route="complex_research", reason_code="comparison"
        ),
        execution=ExecutionState(
            plan=None, task_results=(), evidence_evaluation=None
        ),
        clarification=None,
        final_response=None,
    )
    state["discovery"] = discovery
    state["research_target_selection"] = selection
    return state


def test_child_state_round_trips_selection_and_discovery() -> None:
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_state,
        merge_complex_result_into_supervisor,
        normalize_complex_state,
    )

    selection = factories.target_selection()
    discovery = _discovery(selection.target_slots)
    parent = _parent_state(discovery=discovery, selection=selection)
    child = build_complex_research_state(parent)
    assert child["discovery"] is discovery
    assert child["research_target_selection"] is selection
    serde = json.loads(
        json.dumps(child, default=lambda value: value.model_dump(mode="json"))
    )
    normalized = normalize_complex_state(serde)
    assert isinstance(normalized["discovery"], DiscoveryCheckpoint)
    assert normalized["discovery"] == discovery
    assert isinstance(
        normalized["research_target_selection"], ResearchTargetSelection
    )
    assert normalized["research_target_selection"] == selection
    merged = merge_complex_result_into_supervisor(parent, normalized)
    assert merged["discovery"] == discovery
    assert merged["research_target_selection"] == selection
    assert merged["execution"].plan is None


# ---------------------------------------------------------------------------
# Governed entries pass both contexts; projections stay privacy-safe
# ---------------------------------------------------------------------------


def _runtime_context(allowed: frozenset[str]) -> GraphRuntimeContext:
    from datetime import UTC, datetime

    runtime = CapabilityRuntimeContext(
        request_id="req-select-1",
        run_id="run-select-1",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=allowed,
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    return GraphRuntimeContext(
        capability_runtime=runtime, services=RuntimeServices()
    )


def test_planning_input_carries_selection_and_discovery() -> None:
    from app.services.agents.v2.capabilities import (
        CapabilityRegistration,
        build_capability_registry,
    )
    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_state,
        build_planning_input,
    )

    class _Capability:
        descriptor = CapabilityDescriptor(
            name="document.read",  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=True,
        )

        async def execute(self, request, runtime):  # type: ignore[no-untyped-def]
            raise AssertionError("never dispatched here")

    context = _runtime_context(frozenset({"document.read"}))
    context.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=_Capability())],
        context.capability_runtime,
    )
    selection = factories.target_selection()
    discovery = _discovery(selection.target_slots)
    child = build_complex_research_state(
        _parent_state(discovery=discovery, selection=selection)
    )
    planning_input = build_planning_input(child, context)
    assert planning_input.target_selection is selection
    assert planning_input.discovery_checkpoint is discovery


@pytest.mark.asyncio
async def test_validate_checkpoint_node_passes_both_contexts(
    monkeypatch,
) -> None:
    import app.services.agents.v2.complex_research_graph as graph_module
    from app.services.agents.v2.capabilities import (
        CapabilityRegistration,
        build_capability_registry,
    )
    from app.services.agents.v2.complex_research_graph import (
        ComplexResearchState,
        validate_checkpoint_node,
    )

    class _Capability:
        descriptor = CapabilityDescriptor(
            name="document.read",  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=True,
        )

        async def execute(self, request, runtime):  # type: ignore[no-untyped-def]
            raise AssertionError("never dispatched here")

    class _Leases:
        class _Session:
            async def commit(self) -> None:
                return None

        def __init__(self) -> None:
            self.session = self._Session()

        async def acquire_or_refresh(
            self, run_id, revision_id=None, evidence_use_id=None
        ):
            return None

    captured: list[dict] = []
    real_validate = graph_module.validate_task_plan

    def _spy(plan, bindings, **kwargs):
        captured.append(kwargs)
        return real_validate(plan, bindings, **kwargs)

    monkeypatch.setattr(graph_module, "validate_task_plan", _spy)
    selection = factories.target_selection()
    discovery = _discovery(selection.target_slots)
    context = _runtime_context(frozenset({"document.read"}))
    context.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=_Capability())],
        context.capability_runtime,
    )
    context.services.retention_leases = _Leases()
    bindings = factories.binding_set(
        bindings=(
            factories.scoped_document(document_revision=str(uuid4())),
        )
    )
    state = ComplexResearchState(
        contract_version="2.0",
        semantic=_semantic(),
        bindings=bindings,
        query_analysis=_analysis("summarize"),
        route_decision=RouteDecision(
            route="complex_research", reason_code="comparison"
        ),
        plan=None,
        task_results=(),
        evaluation=None,
        replans_remaining=0,
        discovery=discovery,
        research_target_selection=selection,
    )
    update = await validate_checkpoint_node(state, context)
    assert update.get("plan") is not None
    assert captured
    assert captured[0]["target_selection"] == selection
    assert captured[0]["discovery_checkpoint"] == discovery


@pytest.mark.asyncio
async def test_adaptive_planner_passes_contexts_to_final_validation(
    monkeypatch,
) -> None:
    import app.services.agents.v2.planning.planner as planner_module
    from app.services.agents.v2.capabilities import (
        CapabilityRegistration,
        build_capability_registry,
    )
    from app.services.agents.v2.planning import AdaptivePlanner

    captured: list[dict] = []

    def _spy(plan, bindings, **kwargs):
        captured.append(kwargs)
        return None

    monkeypatch.setattr(planner_module, "validate_task_plan", _spy)

    class _Chunk:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    class _Provider:
        async def astream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            yield _Chunk(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "capability": "document.read",
                                "task_objective": "read b1",
                                "targets": ["b1"],
                                "depends_on": [],
                            }
                        ]
                    }
                )
            )

    class _Capability:
        descriptor = CapabilityDescriptor(
            name="document.read",  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=True,
        )

        async def execute(self, request, runtime):  # type: ignore[no-untyped-def]
            raise AssertionError("never dispatched here")

    planner = AdaptivePlanner(provider_factory=lambda: _Provider())
    context = _runtime_context(frozenset({"document.read"}))
    context.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=_Capability())],
        context.capability_runtime,
    )
    discovery = _discovery(factories.target_selection().target_slots)
    await planner.propose_initial(
        _planning_input(
            "multi_goal",
            discovery=discovery,
            capability_names=frozenset({"document.read"}),
        ),
        context,
    )
    assert captured
    assert captured[0]["target_selection"] is None
    assert captured[0]["discovery_checkpoint"] == discovery


def test_projections_carry_no_selection_or_discovery_data() -> None:
    discovery = _discovery(factories.target_selection().target_slots)
    selection = factories.target_selection()
    bare = _planning_input()
    rich = _planning_input(selection=selection, discovery=discovery)
    planner_bare = build_planner_model_input(bare).model_dump_json()
    planner_rich = build_planner_model_input(rich).model_dump_json()
    assert planner_rich == planner_bare
    replanner_bare = build_replanner_model_input_from_envelope(
        bare
    ).model_dump_json()
    replanner_rich = build_replanner_model_input_from_envelope(
        rich
    ).model_dump_json()
    assert replanner_rich == replanner_bare
    for payload in (planner_rich, replanner_rich):
        assert str(AGGREGATE_ID) not in payload
        assert str(DISCOVERY_ID) not in payload
        assert str(factories.DOCUMENT_ID) not in payload
        assert "discovery_checkpoint" not in payload
        assert "target_selection" not in payload
        assert "research_target_selection" not in payload

"""Phase 2 Task 3 — deterministic fast-plan builder tests (TDD, failing first).

`build_fast_plan` is a pure function of frozen values: it raises unless the
route is `fast_domain`, mints stable ids, maps the router reason code to the
single shared capability, builds the required target units (none for
People/KG; pinned targets for document/section reads), and validates the plan
before returning. No planner model is ever called.
"""
from __future__ import annotations

import inspect
from uuid import UUID

import pytest

from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import (
    DocumentReference,
    SectionReference,
    SemanticContext,
)
from app.services.agents.v2.contracts.validation import validate_fast_plan

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def semantic(
    query: str = "Ai là Nguyễn Văn A?",
    *,
    document_refs: tuple = (),
    person_refs: tuple = (),
    section_refs: tuple = (),
) -> SemanticContext:
    return SemanticContext(
        contextualized_query=query,
        normalized_query=query,
        abbreviations=(),
        coreferences=(),
        document_refs=document_refs,
        person_refs=person_refs,
        section_refs=section_refs,
        blocking_ambiguities=(),
    )


def resolved_doc_ref(ref_id: str = "r1") -> DocumentReference:
    return DocumentReference(
        ref_id=ref_id,
        original_span="A",
        normalized_reference="A",
        requested_role="target",
        revision_requirement=None,
        resolution_status="resolved",
        resolved_document_id=DOCUMENT_ID,
        candidate_document_ids=(),
    )


def pin(binding_id: str = "b_r1") -> ScopedDocument:
    return ScopedDocument(
        binding_id=binding_id,
        document_id=DOCUMENT_ID,
        document_revision="11111111-1111-1111-1111-111111111111",
        role="target",
    )


def bindings(*pins: ScopedDocument) -> DocumentBindingSet:
    return DocumentBindingSet(bindings=tuple(pins), revision_requirement_refs=())


def people_analysis() -> QueryAnalysis:
    return QueryAnalysis(work_type="lookup", domains=("people",), dependency_hints=())


def people_route() -> RouteDecision:
    return RouteDecision(route="fast_domain", reason_code="simple_people_lookup")


# ---------------------------------------------------------------------------
# People / KG: one TaskSpec, zero TargetUnits
# ---------------------------------------------------------------------------


def test_fast_people_plan_has_one_task_and_zero_targets() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    plan = build_fast_plan(
        semantic(person_refs=({"ref_id": "p1", "kind": "person", "label": "A"},)),
        bindings(),
        people_analysis(),
        people_route(),
    )
    assert len(plan.tasks) == 1
    assert plan.target_units == ()
    task = plan.tasks[0]
    assert task.capability == "people.lookup"
    assert task.input.kind == "people.lookup"
    assert task.depends_on == ()
    validate_fast_plan(plan, bindings())


def test_fast_kg_plan_has_one_task_and_zero_targets() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    query = "A thuộc đơn vị nào?"
    plan = build_fast_plan(
        semantic(query),
        bindings(),
        QueryAnalysis(work_type="lookup", domains=("knowledge_graph",)),
        RouteDecision(route="fast_domain", reason_code="simple_kg_lookup"),
    )
    assert len(plan.tasks) == 1
    assert plan.target_units == ()
    assert plan.tasks[0].capability == "knowledge_graph.query"
    validate_fast_plan(plan, bindings())


# ---------------------------------------------------------------------------
# Document / section: one TaskSpec with the required TargetUnits
# ---------------------------------------------------------------------------


def test_fast_document_plan_has_required_targets() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    query = "Điều 5 của A nói gì?"
    plan = build_fast_plan(
        semantic(query, document_refs=(resolved_doc_ref(),)),
        bindings(pin()),
        QueryAnalysis(work_type="retrieve", domains=("document",)),
        RouteDecision(route="fast_domain", reason_code="exact_document_metadata"),
    )
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.capability == "document.read"
    assert task.input.kind == "document.read"
    assert len(plan.target_units) == 1
    unit = plan.target_units[0]
    assert unit.binding_id == "b_r1"
    assert task.input.target_ids == (unit.target_id,)
    validate_fast_plan(plan, bindings(pin()))


def test_fast_section_plan_has_required_targets() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    query = "Mục 2 của A nói gì?"
    plan = build_fast_plan(
        semantic(
            query,
            document_refs=(resolved_doc_ref(),),
            section_refs=(
                SectionReference(
                    ref_id="s1", label="Mục 2", structure_node_id="node-5"
                ),
            ),
        ),
        bindings(pin()),
        QueryAnalysis(work_type="retrieve", domains=("document", "section")),
        RouteDecision(route="fast_domain", reason_code="exact_section_retrieval"),
    )
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.capability == "section.read"
    assert task.input.kind == "section.read"
    assert len(plan.target_units) == 1
    assert task.input.target_ids == (plan.target_units[0].target_id,)
    validate_fast_plan(plan, bindings(pin()))


# ---------------------------------------------------------------------------
# Router-owned outcomes: direct / summary / compare
# ---------------------------------------------------------------------------


def test_direct_greeting_has_no_plan() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    with pytest.raises(ValueError, match="fast_domain"):
        build_fast_plan(
            semantic("Xin chào"),
            bindings(),
            QueryAnalysis(work_type="explain", domains=("memory",)),
            RouteDecision(route="direct", reason_code="direct_greeting"),
        )


def test_bounded_summary_uses_document_read_not_summary_agent() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    plan = build_fast_plan(
        semantic("Tóm tắt A", document_refs=(resolved_doc_ref(),)),
        bindings(pin()),
        QueryAnalysis(work_type="summarize", domains=("document",)),
        RouteDecision(route="fast_domain", reason_code="exact_document_metadata"),
    )
    assert plan.tasks[0].capability == "document.read"
    assert "summary" not in plan.tasks[0].capability
    validate_fast_plan(plan, bindings(pin()))


def test_compare_never_routes_fast() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan
    from app.services.agents.v2.nodes.routing import decide_route

    analysis = QueryAnalysis(work_type="compare", domains=("document",))
    decision = decide_route(
        analysis,
        semantic(
            "So sánh A và B",
            document_refs=(resolved_doc_ref("r1"), resolved_doc_ref("r2")),
        ),
        bindings(pin("b_r1"), pin("b_r2")),
        allowed_capabilities=frozenset({"document.read"}),
    )
    assert decision.route == "complex_research"
    with pytest.raises(ValueError, match="fast_domain"):
        build_fast_plan(
            semantic("So sánh A và B"),
            bindings(pin("b_r1"), pin("b_r2")),
            analysis,
            decision,
        )


def test_fast_plan_calls_no_planner_model() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    parameters = inspect.signature(build_fast_plan).parameters
    assert set(parameters) == {"semantic", "bindings", "analysis", "route"}
    first = build_fast_plan(
        semantic(person_refs=({"ref_id": "p1", "kind": "person", "label": "A"},)),
        bindings(),
        people_analysis(),
        people_route(),
    )
    second = build_fast_plan(
        semantic(person_refs=({"ref_id": "p1", "kind": "person", "label": "A"},)),
        bindings(),
        people_analysis(),
        people_route(),
    )
    assert first == second
    assert first.tasks[0].task_id == second.tasks[0].task_id
    assert first.plan_id == second.plan_id


def test_fast_plan_ids_are_stable_per_query_and_reason() -> None:
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    kwargs = dict(
        semantic=semantic(
            person_refs=({"ref_id": "p1", "kind": "person", "label": "A"},)
        ),
        bindings=bindings(),
        analysis=people_analysis(),
    )
    people = build_fast_plan(route=people_route(), **kwargs)
    other_reason = build_fast_plan(
        route=RouteDecision(route="fast_domain", reason_code="simple_kg_lookup"),
        **kwargs,
    )
    assert people.tasks[0].task_id != other_reason.tasks[0].task_id
    assert people.plan_id != other_reason.plan_id


# ---------------------------------------------------------------------------
# fast_plan_node: checkpoint the plan before any dispatch
# ---------------------------------------------------------------------------


def _node_runtime() -> object:
    from datetime import UTC, datetime

    from app.services.agents.v2.contracts.capability import (
        CapabilityRuntimeContext,
    )
    from app.services.agents.v2.contracts.state import GraphRuntimeContext
    from app.services.agents.v2.contracts.state import RuntimeServices

    return GraphRuntimeContext(
        capability_runtime=CapabilityRuntimeContext(
            request_id="req-1",
            run_id="run-1",
            user_id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
            workspace_ids=(UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),),
            can_read_people=True,
            allowed_capabilities=frozenset({"document.read"}),
            deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        services=RuntimeServices(),
    )


def _node_state(**overrides: object) -> dict:
    from app.services.agents.v2.contracts.conversation import ConversationContext
    from app.services.agents.v2.contracts.request import RequestContext
    from app.services.agents.v2.contracts.state import ExecutionState

    state: dict = {
        "contract_version": "2.0",
        "request": RequestContext(
            contract_version="2.0",
            request_id="req-1",
            thread_id="thread-1",
            original_query="Điều 5 của A nói gì?",
            known_documents=(),
        ),
        "conversation": ConversationContext(
            summary="", active_entities=(), last_focus=None, recent_turns=()
        ),
        "semantic": semantic("Điều 5 của A nói gì?", document_refs=(resolved_doc_ref(),)),
        "bindings": bindings(pin()),
        "query_analysis": QueryAnalysis(work_type="retrieve", domains=("document",)),
        "route_decision": RouteDecision(
            route="fast_domain", reason_code="exact_document_metadata"
        ),
        "execution": ExecutionState(
            plan=None, task_results=(), evidence_evaluation=None
        ),
        "clarification": None,
        "final_response": None,
    }
    state.update(overrides)
    return state  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_fast_plan_node_checkpoints_plan_before_dispatch() -> None:
    from app.services.agents.v2.contracts.state import ExecutionState
    from app.services.agents.v2.nodes.fast_plan import fast_plan_node

    update = await fast_plan_node(_node_state(), _node_runtime())  # type: ignore[arg-type]
    assert set(update) == {"execution"}
    execution = update["execution"]
    assert isinstance(execution, ExecutionState)
    assert execution.plan is not None
    assert len(execution.plan.tasks) == 1
    assert execution.plan.tasks[0].capability == "document.read"
    assert execution.task_results == ()
    assert execution.evidence_evaluation is None


@pytest.mark.asyncio
async def test_fast_plan_node_clears_stale_results_and_evaluation() -> None:
    """C2: a turn-2 fast plan resets results AND a non-None prior evaluation."""
    from app.services.agents.v2.contracts.evaluation import Coverage, EvidenceEvaluation
    from app.services.agents.v2.contracts.state import ExecutionState
    from app.services.agents.v2.nodes.fast_plan import fast_plan_node

    stale_evaluation = EvidenceEvaluation(
        status="insufficient",
        coverage=Coverage(items=()),
        missing=(),
        contradictions=(),
    )
    state = _node_state()
    state["execution"] = ExecutionState(
        plan=None,
        task_results=state["execution"].task_results,
        evidence_evaluation=stale_evaluation,
    )
    update = await fast_plan_node(state, _node_runtime())  # type: ignore[arg-type]
    execution = update["execution"]
    assert execution.plan is not None
    assert execution.task_results == ()
    assert execution.evidence_evaluation is None


@pytest.mark.asyncio
async def test_fast_plan_node_fails_closed_without_fast_route() -> None:
    from app.services.agents.v2.nodes.fast_plan import FastPlanError, fast_plan_node

    with pytest.raises(FastPlanError):
        await fast_plan_node(
            _node_state(
                route_decision=RouteDecision(
                    route="direct", reason_code="direct_greeting"
                )
            ),
            _node_runtime(),  # type: ignore[arg-type]
        )
    with pytest.raises(FastPlanError):
        await fast_plan_node(
            _node_state(query_analysis=None, route_decision=None),
            _node_runtime(),  # type: ignore[arg-type]
        )

"""Spec §13.3/§16/§25 — task outcomes, replan context, and checkpoint gating.

Covers the brief's Step 1 items "TaskExecutionSummary/replan context" and
"incompatible checkpoint rejection", plus the §26 facts: a People→Document
replan sees the current plan and ``T1=not_found`` even when T1 produced no
EvidenceUse, ``T1=error``/``error_code=TIMEOUT`` stays distinguishable, a failed
or absent People task cannot supply the materialized ``person_identifier``, and
incompatible pre-release fixtures/checkpoints are rejected rather than migrated.
"""
from __future__ import annotations

from typing import cast

import pytest

from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    DocumentReadInput,
    DocumentSearchInput,
    DocumentSearchOutput,
)
from app.services.agents.v2.contracts.evaluation import Coverage, CoverageObservation, EvidenceEvaluation
from app.services.agents.v2.contracts.execution import AgentResult, TaskExecutionSummary
from app.services.agents.v2.contracts.planning import (
    DiscoveryPolicy,
    ReplanTaskOrigin,
    ResearchBudgetView,
    ResearchPlanningInput,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.state import (
    CHECKPOINT_SCHEMA_REVISION,
    ExecutionState,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    IncompatibleCheckpointError,
    validate_agent_result,
    validate_checkpoint_payload,
    validate_replan,
    validate_research_planning_input,
    validate_supervisor_state,
    validate_task_outcomes,
)

from .factories import (
    binding_set,
    conversation_context,
    kg_task,
    people_plan,
    person_search_task,
    read_plan,
    request_context,
    semantic_context,
    target_unit,
)

POLICY = DiscoveryPolicy(
    allow_reference_discovery=True,
    allow_supporting_discovery=True,
    max_discovered_documents=3,
)
BUDGET = ResearchBudgetView(max_tasks_remaining=3, max_replans_remaining=1, max_parallel_branches=2)


def _evaluation(status: str) -> EvidenceEvaluation:
    return EvidenceEvaluation(
        status=status,  # type: ignore[arg-type]
        coverage=Coverage(items=()),
        missing=(),
        contradictions=(),
    )


def _replan_task(task_id: str = "T2") -> TaskSpec:
    return kg_task(task_id).model_copy(
        update={
            "origin": ReplanTaskOrigin(
                kind="replan", reason="T1 not_found", task_ids=("T1",), evidence_use_ids=()
            )
        }
    )


def _append(current: TaskPlan, *tasks: TaskSpec) -> TaskPlan:
    return current.model_copy(update={"tasks": current.tasks + tasks})


def _person_search_replan_task(*, depends_on: tuple[str, ...] = ("T1",)) -> TaskSpec:
    return person_search_task(depends_on=depends_on).model_copy(
        update={
            "origin": ReplanTaskOrigin(
                kind="replan",
                reason="People scalar available",
                task_ids=("T1",),
                evidence_use_ids=(),
            )
        }
    )


def _planning_input(*, current_plan: TaskPlan | None, with_outcomes: bool) -> ResearchPlanningInput:
    outcome = TaskExecutionSummary(task_id="T1", status="not_found")
    return ResearchPlanningInput(
        semantic=semantic_context(),
        bindings=binding_set(),
        query_analysis=QueryAnalysis(work_type="cross_domain", domains=("people", "document")),
        capability_catalog=(
            CapabilityDescriptor(
                name="document.read",
                domain="document",
                operation_type="read",
                supports_parallel=True,
            ),
        ),
        discovery_policy=POLICY,
        budget=BUDGET,
        current_plan=current_plan,
        task_outcomes=(outcome,) if with_outcomes else (),
        prior_evidence_uses=(),
        prior_evaluation=_evaluation("insufficient") if current_plan is not None else None,
    )


def _state(**overrides: object) -> SupervisorV2State:
    base: dict[str, object] = {
        "contract_version": "2.0",
        "request": request_context(),
        "conversation": conversation_context(),
        "semantic": semantic_context(),
        "bindings": binding_set(),
        "query_analysis": None,
        "route_decision": RouteDecision(route="fast_domain", reason_code="exact_section_retrieval"),
        "execution": ExecutionState(plan=read_plan(), task_results=(), evidence_evaluation=None),
        "clarification": None,
        "final_response": None,
        "synthesis": None,
        "checkpoint_schema_revision": CHECKPOINT_SCHEMA_REVISION,
        "discovery_need": None,
        "discovery": None,
        "document_selection_clarification": None,
        "research_target_selection": None,
        "intent_analysis": None,
    }
    base.update(overrides)
    return cast(SupervisorV2State, base)


def test_not_found_and_timeout_outcomes_are_distinguishable() -> None:
    plan = people_plan()
    not_found = TaskExecutionSummary(task_id="T1", status="not_found")
    timeout = TaskExecutionSummary(task_id="T1", status="error", error_code="TIMEOUT")
    assert not_found != timeout
    validate_task_outcomes((not_found,), plan)
    validate_task_outcomes((timeout,), plan)


def test_error_code_is_only_valid_for_failure_statuses() -> None:
    with pytest.raises(ContractValidationError, match="error_code"):
        validate_task_outcomes(
            (TaskExecutionSummary(task_id="T1", status="not_found", error_code="TIMEOUT"),),
            people_plan(),
        )
    with pytest.raises(ContractValidationError, match="error_code"):
        validate_task_outcomes(
            (TaskExecutionSummary(task_id="T1", status="error"),), people_plan()
        )
    validate_task_outcomes(
        (TaskExecutionSummary(task_id="T1", status="denied", error_code="PERMISSION_DENIED"),),
        people_plan(),
    )


def test_outcomes_resolve_to_checkpointed_tasks_and_are_unique() -> None:
    with pytest.raises(ContractValidationError, match="TX"):
        validate_task_outcomes(
            (TaskExecutionSummary(task_id="TX", status="success"),), people_plan()
        )
    duplicate = TaskExecutionSummary(task_id="T1", status="success")
    with pytest.raises(ContractValidationError, match="duplicate"):
        validate_task_outcomes((duplicate, duplicate), people_plan())


def test_initial_planning_receives_no_prior_execution_facts() -> None:
    validate_research_planning_input(_planning_input(current_plan=None, with_outcomes=False))
    with pytest.raises(ContractValidationError, match="initial"):
        validate_research_planning_input(_planning_input(current_plan=None, with_outcomes=True))


def test_replanning_requires_the_current_plan_and_latest_evaluation() -> None:
    replan = _planning_input(current_plan=people_plan(), with_outcomes=True)
    validate_research_planning_input(replan)
    with pytest.raises(ContractValidationError, match="evaluation"):
        validate_research_planning_input(
            replan.model_copy(update={"prior_evaluation": None})
        )


def test_replan_appends_without_mutating_completed_tasks() -> None:
    current = people_plan()
    proposed = _append(current, _replan_task())
    outcomes = (TaskExecutionSummary(task_id="T1", status="not_found"),)
    assert validate_replan(current, proposed, outcomes, POLICY, BUDGET) is proposed

    rerun = _append(current, _replan_task().model_copy(update={"task_id": "T1"}))
    with pytest.raises(ContractValidationError):
        validate_replan(current, rerun, outcomes, POLICY, BUDGET)


def test_replan_cannot_change_the_existing_prefix() -> None:
    current = people_plan()
    mutated = current.model_copy(
        update={
            "tasks": (
                current.tasks[0].model_copy(update={"task_objective": "something else"}),
                _replan_task(),
            )
        }
    )
    with pytest.raises(ContractValidationError, match="append-only"):
        validate_replan(
            current,
            mutated,
            (TaskExecutionSummary(task_id="T1", status="success"),),
            POLICY,
            BUDGET,
        )


def test_replan_cannot_add_target_units() -> None:
    current = read_plan()
    extra_target = target_unit(target_id="t2")
    extra_task = current.tasks[0].model_copy(
        update={
            "task_id": "T2",
            "input": DocumentReadInput(kind="document.read", target_ids=("t2",)),
            "origin": ReplanTaskOrigin(
                kind="replan", reason="coverage gap", task_ids=("T1",), evidence_use_ids=()
            ),
        }
    )
    proposed = current.model_copy(
        update={
            "target_units": current.target_units + (extra_target,),
            "tasks": current.tasks + (extra_task,),
        }
    )
    with pytest.raises(ContractValidationError, match="target"):
        validate_replan(
            current,
            proposed,
            (TaskExecutionSummary(task_id="T1", status="success"),),
            POLICY,
            BUDGET,
        )


def test_replan_requires_an_appended_replan_task() -> None:
    current = people_plan()
    with pytest.raises(ContractValidationError, match="append"):
        validate_replan(current, current, (), POLICY, BUDGET)

    initial_origin = people_plan().model_copy(
        update={"tasks": (people_plan().tasks[0].model_copy(update={"task_id": "T2"}),)}
    )
    with pytest.raises(ContractValidationError, match="ReplanTaskOrigin"):
        validate_replan(current, _append(current, initial_origin.tasks[0]), (), POLICY, BUDGET)


def test_replan_respects_replan_and_task_budget() -> None:
    current = people_plan()
    proposed = _append(current, _replan_task())
    with pytest.raises(ContractValidationError, match="replan budget"):
        validate_replan(
            current,
            proposed,
            (),
            POLICY,
            BUDGET.model_copy(update={"max_replans_remaining": 0}),
        )
    with pytest.raises(ContractValidationError, match="budget"):
        validate_replan(
            current,
            proposed,
            (),
            POLICY,
            BUDGET.model_copy(update={"max_tasks_remaining": 0}),
        )


def test_replan_respects_discovery_policy() -> None:
    current = read_plan()
    search_task = read_plan().tasks[0].model_copy(
        update={
            "task_id": "T2",
            "capability": "document.search",
            "depends_on": ("T1",),
            "origin": ReplanTaskOrigin(
                kind="replan", reason="discover references", task_ids=("T1",), evidence_use_ids=()
            ),
        }
    )
    search_task = search_task.model_copy(
        update={"input": DocumentSearchInput(kind="document.search", query="A")}
    )
    disabled = POLICY.model_copy(
        update={"allow_reference_discovery": False, "allow_supporting_discovery": False}
    )
    with pytest.raises(ContractValidationError, match="discovery"):
        validate_replan(current, _append(current, search_task), (), disabled, BUDGET)
    validate_replan(current, _append(current, search_task), (), POLICY, BUDGET)


def test_replan_cannot_materialize_a_person_identifier_from_a_failed_people_task() -> None:
    current = people_plan()
    proposed = _append(current, _person_search_replan_task())
    timeout = (TaskExecutionSummary(task_id="T1", status="error", error_code="TIMEOUT"),)
    with pytest.raises(ContractValidationError, match="person_identifier"):
        validate_replan(current, proposed, timeout, POLICY, BUDGET)

    succeeded = (TaskExecutionSummary(task_id="T1", status="success"),)
    assert validate_replan(current, proposed, succeeded, POLICY, BUDGET) is proposed


def test_replan_cannot_materialize_a_person_identifier_without_a_people_task() -> None:
    current = read_plan()
    proposed = _append(current, _person_search_replan_task(depends_on=()))
    with pytest.raises(ContractValidationError, match="person_identifier"):
        validate_replan(
            current,
            proposed,
            (TaskExecutionSummary(task_id="T1", status="success"),),
            POLICY,
            BUDGET,
        )


def test_checkpoint_payload_rejects_missing_or_foreign_versions() -> None:
    payload = {
        "contract_version": "2.0",
        "request": {"contract_version": "2.0"},
        "conversation": {},
        "semantic": {},
        "bindings": {},
        "execution": {"plan": None, "task_results": [], "evidence_evaluation": None},
        "query_analysis": None,
        "route_decision": None,
        "clarification": None,
        "synthesis": None,
        "final_response": None,
        "synthesis": None,
        "checkpoint_schema_revision": CHECKPOINT_SCHEMA_REVISION,
        "discovery_need": None,
        "discovery": None,
        "document_selection_clarification": None,
        "research_target_selection": None,
        "intent_analysis": None,
    }
    validate_checkpoint_payload(payload)

    with pytest.raises(IncompatibleCheckpointError, match="1.0"):
        validate_checkpoint_payload({**payload, "contract_version": "1.0"})
    without_version = {key: value for key, value in payload.items() if key != "contract_version"}
    with pytest.raises(IncompatibleCheckpointError, match="contract_version"):
        validate_checkpoint_payload(without_version)
    with pytest.raises(IncompatibleCheckpointError, match="plan"):
        validate_checkpoint_payload(
            {**payload, "execution": {"plan": {"contract_version": "2.1"}, "task_results": []}}
        )
    with pytest.raises(IncompatibleCheckpointError, match="task_results"):
        validate_checkpoint_payload(
            {
                **payload,
                "execution": {
                    "plan": None,
                    "task_results": [{"contract_version": "2.1"}],
                    "evidence_evaluation": None,
                },
            }
        )
    with pytest.raises(IncompatibleCheckpointError, match="request"):
        validate_checkpoint_payload(
            {**payload, "request": {"request_id": "req-1"}}
        )
    with pytest.raises(IncompatibleCheckpointError, match="missing"):
        validate_checkpoint_payload(
            {"contract_version": "2.0", "checkpoint_schema_revision": CHECKPOINT_SCHEMA_REVISION}
        )


def test_typed_checkpoint_state_validates_when_consistent() -> None:
    validate_supervisor_state(_state())


def test_checkpoint_state_requires_a_plan_for_results_and_evaluation() -> None:
    result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    state = _state(
        execution=ExecutionState(plan=None, task_results=(result,), evidence_evaluation=None)
    )
    with pytest.raises(ContractValidationError, match="TaskPlan"):
        validate_supervisor_state(state)

    empty_plan = ExecutionState(plan=None, task_results=(), evidence_evaluation=_evaluation("insufficient"))
    with pytest.raises(ContractValidationError, match="TaskPlan"):
        validate_supervisor_state(_state(execution=empty_plan))


def test_checkpoint_state_rejects_unresolved_cross_references() -> None:
    from app.services.agents.v2.contracts.semantic import CurrentRevisionRequirement

    stale_semantic = semantic_context()
    stale_semantic = stale_semantic.model_copy(
        update={
            "document_refs": (
                stale_semantic.document_refs[0].model_copy(
                    update={"revision_requirement": CurrentRevisionRequirement(kind="current")}
                ),
            )
        }
    )
    with pytest.raises(ContractValidationError, match="current"):
        validate_supervisor_state(_state(semantic=stale_semantic))


def test_direct_route_keeps_no_plan_and_clarify_route_keeps_a_request() -> None:
    direct = ExecutionState(plan=None, task_results=(), evidence_evaluation=None)
    validate_supervisor_state(
        _state(
            route_decision=RouteDecision(route="direct", reason_code="direct_greeting"),
            execution=direct,
        )
    )
    with pytest.raises(ContractValidationError, match="direct"):
        validate_supervisor_state(
            _state(route_decision=RouteDecision(route="direct", reason_code="direct_greeting"))
        )
    with pytest.raises(ContractValidationError, match="ClarificationRequest"):
        validate_supervisor_state(
            _state(route_decision=RouteDecision(route="clarify", reason_code="essential_ambiguity"))
        )


def test_agent_result_data_must_match_the_task_capability() -> None:
    result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=DocumentSearchOutput(kind="document.search", candidates=()),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    with pytest.raises(ContractValidationError, match="capability"):
        validate_agent_result(result, read_plan())


def test_only_read_capabilities_may_report_read_coverage() -> None:
    observed = CoverageObservation(
        target_id="t1", observed_locators=(DocumentLocator(kind="document"),), outcome="read"
    )
    people_result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=None,
        evidence_uses=(),
        coverage_observations=(observed,),
        error=None,
    )
    with pytest.raises(ContractValidationError, match="coverage"):
        validate_agent_result(people_result, people_plan())


def test_successful_read_must_report_coverage() -> None:
    result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    with pytest.raises(ContractValidationError, match="without coverage observations"):
        validate_agent_result(result, read_plan())

    covered = result.model_copy(
        update={
            "coverage_observations": (
                CoverageObservation(
                    target_id="t1",
                    observed_locators=(DocumentLocator(kind="document"),),
                    outcome="read",
                ),
            )
        }
    )
    validate_agent_result(covered, read_plan())


def test_status_and_agent_error_stay_consistent() -> None:
    from app.services.agents.v2.contracts.execution import AgentError

    denied_without_error = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="denied",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    with pytest.raises(ContractValidationError, match="AgentError"):
        validate_agent_result(denied_without_error, read_plan())

    not_found_with_error = denied_without_error.model_copy(
        update={
            "status": "not_found",
            "error": AgentError(code="INTERNAL_ERROR", message="boom", retryable=False),
        }
    )
    with pytest.raises(ContractValidationError, match="AgentError"):
        validate_agent_result(not_found_with_error, read_plan())

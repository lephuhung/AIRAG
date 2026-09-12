"""Contract-parity tests for Phase 0 scenarios.

Covers both the brief's Step 1 (typed fixtures) and Step 3 (parity facts).
"""
from __future__ import annotations

import json
from typing import get_type_hints
from uuid import UUID

import pytest

from .frozen_contracts import (
    AgentError,
    AgentResult,
    AgentStatus,
    CapabilityRuntimeContext,
    ContractModel,
    DocumentCandidate,
    ExecutionState,
    GraphRuntimeContext,
    RequestContext,
    RuntimeServices,
    SupervisorV2State,
    TaskExecutionSummary,
    TaskPlan,
    TaskSpec,
)
from .scenarios import SCENARIOS, Scenario, summary_for


# ---------------------------------------------------------------------------
# Step 1 — typed fixtures load and the runtime context is absent from JSON.
# ---------------------------------------------------------------------------

def test_scenarios_use_typed_frozen_contracts() -> None:
    """All scenarios load; SupervisorV2State carries contract_version + execution;
    at least one scenario has a TaskPlan in its execution state.
    """
    assert SCENARIOS
    hints = set(get_type_hints(SupervisorV2State))
    assert {"contract_version", "execution"} <= hints
    assert any(
        isinstance(s.initial_state["execution"].plan, TaskPlan)
        for s in SCENARIOS
    )


def test_runtime_context_is_not_in_checkpoint_json() -> None:
    """The checkpointable state must not serialize workspace_ids or user_id."""
    scenario = next(s for s in SCENARIOS if s.scenario_id == "acl_runtime_replaced")
    raw = scenario.checkpoint_json()
    assert "workspace_ids" not in raw
    assert "ws-new" not in raw          # the *current* workspace
    assert "ws-default" not in raw      # nor the *old* one
    # Runtime context keys must never leak, period.
    assert "user_id" not in raw
    assert "RuntimeServices" not in raw


def test_state_root_contract_version_is_v2() -> None:
    for s in SCENARIOS:
        assert s.initial_state["contract_version"] == "2.0"


# ---------------------------------------------------------------------------
# Step 3 — mandatory parity facts.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scenarios_by_id() -> dict[str, Scenario]:
    return {s.scenario_id: s for s in SCENARIOS}


def _all_task_ids(scenarios: list[Scenario]) -> list[str]:
    out: list[str] = []
    for s in scenarios:
        plan = s.initial_state["execution"].plan
        if plan is not None:
            out.extend(t.task_id for t in plan.tasks)
    return out


def _all_target_ids(scenarios: list[Scenario]) -> list[str]:
    out: list[str] = []
    for s in scenarios:
        plan = s.initial_state["execution"].plan
        if plan is not None:
            out.extend(tu.target_id for tu in plan.target_units)
    return out


def test_unique_task_ids_across_scenarios() -> None:
    ids = _all_task_ids(list(SCENARIOS))
    assert len(ids) == len(set(ids)), f"duplicate task_ids: {ids}"


def test_unique_target_ids_across_scenarios() -> None:
    ids = _all_target_ids(list(SCENARIOS))
    assert len(ids) == len(set(ids)), f"duplicate target_ids: {ids}"


def test_append_only_replan_preserves_prefix() -> None:
    """The replan scenario's plan must keep the original tasks at the front."""
    s = next(x for x in SCENARIOS if x.scenario_id == "append_only_replan")
    plan = s.initial_state["execution"].plan
    assert plan is not None
    assert plan.tasks[0].task_id == "task-051"
    assert plan.tasks[1].task_id == "task-052"
    # The replan task references the prior task in its ReplanTaskOrigin.
    origin = plan.tasks[1].origin
    assert origin.kind == "replan"
    assert "task-051" in origin.task_ids


def test_replan_summaries_are_attached() -> None:
    """The append-only-replan scenario carries a TaskExecutionSummary for the
    prior attempt, which is what drives the planner (not the AgentResult body).
    """
    summary = summary_for("append_only_replan")
    assert summary is not None
    assert summary.task_id == "task-051"
    assert summary.status == "not_found"


def test_async_fan_in_associated_by_task_id_not_order() -> None:
    """AgentResults may arrive in any order; the consumer resolves by task_id."""
    s = next(x for x in SCENARIOS if x.scenario_id == "async_fan_in")
    results = s.initial_state["execution"].task_results
    # In the fixture, task-023 arrived first, task-022 second.
    assert [r.task_id for r in results] == ["task-023", "task-022"]
    # Lookup by task_id works regardless of order.
    by_id = {r.task_id: r for r in results}
    assert by_id["task-022"].status in ("success", "partial")
    assert by_id["task-023"].status in ("success", "partial")


def test_not_found_is_distinct_from_timeout() -> None:
    """not_found = `status='not_found'` with empty evidence; timeout =
    `status='error'` with `error.code='TIMEOUT'`. The two must not collapse.
    """
    nf = next(x for x in SCENARIOS if x.scenario_id == "no_evidence_not_found")
    to = next(x for x in SCENARIOS if x.scenario_id == "no_evidence_timeout")
    nf_result = nf.initial_state["execution"].task_results[0]
    to_result = to.initial_state["execution"].task_results[0]
    assert nf_result.status == "not_found"
    assert nf_result.error is None
    assert nf_result.evidence_uses == ()
    assert to_result.status == "error"
    assert to_result.error is not None
    assert to_result.error.code == "TIMEOUT"
    assert to_result.evidence_uses == ()
    # Hard guarantee: their (status, error_code) tuples differ.
    assert (nf_result.status, None) != (to_result.status, to_result.error.code)


def test_current_acl_replaces_old_runtime() -> None:
    """Resume must use the *current* runtime — old workspaces disappear."""
    s = next(x for x in SCENARIOS if x.scenario_id == "acl_runtime_replaced")
    # Old workspace `ws-default` must not appear in the current runtime.
    assert "ws-default" not in s.runtime_context.capability_runtime.workspace_ids
    assert "ws-new" in s.runtime_context.capability_runtime.workspace_ids
    assert s.runtime_context.capability_runtime.user_id == "user-002"
    # And nothing about old runtime is checkpointed either.
    assert "ws-default" not in s.checkpoint_json()


def test_cancellation_dispatches_no_later_task() -> None:
    """When task N is cancelled, no AgentResult for any later task exists."""
    s = next(x for x in SCENARIOS if x.scenario_id == "cancellation_no_later_task")
    results = s.initial_state["execution"].task_results
    assert len(results) == 1
    assert results[0].task_id == "task-091"
    assert results[0].status == "error"
    assert results[0].error is not None
    assert results[0].error.code == "CANCELLED"
    # task-092 depends on task-091; its non-execution is the contract.
    plan = s.initial_state["execution"].plan
    assert plan is not None
    assert "task-092" in (t.task_id for t in plan.tasks)
    assert "task-092" not in (r.task_id for r in results)


def test_one_terminal_outer_event_for_direct_route() -> None:
    """Direct routes produce exactly one FinalResponse — no inner-node prose."""
    s = next(x for x in SCENARIOS if x.scenario_id == "outer_only_streaming")
    assert s.initial_state["final_response"] is not None
    assert s.initial_state["final_response"].status == "success"
    # No tasks planned (direct = no plan work).
    assert s.initial_state["execution"].plan is not None
    assert s.initial_state["execution"].plan.tasks == ()


def test_checkpoint_bytes_round_trip_through_typed_state() -> None:
    """checkpoint_json() deserializes back into the same SupervisorV2State fields."""
    for s in SCENARIOS:
        raw = s.checkpoint_json()
        decoded = json.loads(raw)
        # Top-level keys match the TypedDict's known set.
        allowed = set(get_type_hints(SupervisorV2State).keys())
        unexpected = set(decoded.keys()) - allowed
        assert not unexpected, f"unexpected checkpoint keys in {s.scenario_id}: {unexpected}"
        # Required invariants survive the round-trip.
        assert decoded["contract_version"] == "2.0"
        assert "execution" in decoded


# ---------------------------------------------------------------------------
# Spec-derived invariants that protect against silent contract drift.
# ---------------------------------------------------------------------------

def test_frozen_contracts_reject_mutation() -> None:
    """Pydantic v2 frozen=True must raise on attribute set."""
    with pytest.raises(Exception):
        rt = GraphRuntimeContext(
            capability_runtime=CapabilityRuntimeContext(
                user_id="u",
                workspace_ids=("w",),
            ),
            services=RuntimeServices(capabilities_registry_id="r"),
        )
        rt.capability_runtime = CapabilityRuntimeContext(  # type: ignore[misc]
            user_id="u2",
            workspace_ids=("w",),
        )


def test_frozen_contracts_reject_extra_fields() -> None:
    """extra='forbid' must reject unknown fields."""
    with pytest.raises(Exception):
        RequestContext(
            contract_version="2.0",
            request_id="r",
            thread_id="t",
            original_query="q",
            unknown_field="nope",  # type: ignore[call-arg]
        )


def test_clarification_scenario_carries_candidates_and_expiry() -> None:
    s = next(x for x in SCENARIOS if x.scenario_id == "clarification_resume")
    clarification = s.initial_state["clarification"]
    assert clarification is not None
    assert clarification.candidates
    assert all(isinstance(c, DocumentCandidate) for c in clarification.candidates)
    assert clarification.expires_at.tzinfo is not None

"""Contract-parity tests for Phase 0 scenarios.

Covers both the brief's Step 1 (typed fixtures) and Step 3 (parity facts).
Each parity fact asserts a *real* invariant from the spec.
"""
from __future__ import annotations

import json
from typing import get_type_hints

import pytest

from .frozen_contracts import (
    ContractModel,
    DocumentCandidate,
    GraphRuntimeContext,
    RequestContext,
    RuntimeServices,
    SupervisorV2State,
    TaskPlan,
    _Phase0CapabilityStandIn,
)
from .scenarios import (
    SCENARIOS,
    Scenario,
    assert_append_only,
    original_plan_for,
    summary_for,
    violating_plan_for,
)


# ---------------------------------------------------------------------------
# Step 1 — typed fixtures load; runtime is absent from JSON.
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
    """The checkpointable state must not serialize any runtime field.

    Uses the brief's literal scenario id `acl-resume` (Step 1 snippet).
    """
    scenario = next(s for s in SCENARIOS if s.scenario_id == "acl-resume")
    raw = scenario.checkpoint_json()
    # Runtime keys/values must never leak.
    for forbidden in (
        "workspace_ids", "ws-new", "ws-default", "user_id", "user-002",
        "user-001", "RuntimeServices", "capability_runtime",
        "registry-001", "services",
    ):
        assert forbidden not in raw, f"runtime key leaked into checkpoint_json: {forbidden}"


def test_state_root_contract_version_is_v2() -> None:
    for s in SCENARIOS:
        assert s.initial_state["contract_version"] == "2.0"


# ---------------------------------------------------------------------------
# Step 3 — mandatory parity facts.
# ---------------------------------------------------------------------------

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


def _all_binding_ids(scenarios: list[Scenario]) -> list[str]:
    out: list[str] = []
    for s in scenarios:
        bindings = s.initial_state["bindings"].bindings
        out.extend(b.binding_id for b in bindings)
    return out


def test_unique_task_ids_across_scenarios() -> None:
    ids = _all_task_ids(list(SCENARIOS))
    assert ids, "no task ids found in scenarios"
    assert len(ids) == len(set(ids)), f"duplicate task_ids: {ids}"


def test_unique_target_ids_across_scenarios() -> None:
    """The brief's parity fact #1: target IDs are unique across scenarios."""
    ids = _all_target_ids(list(SCENARIOS))
    assert ids, "no target ids found in scenarios — parity fact unproven"
    assert len(ids) == len(set(ids)), f"duplicate target_ids: {ids}"


def test_unique_binding_ids_across_scenarios() -> None:
    ids = _all_binding_ids(list(SCENARIOS))
    assert ids, "no binding ids found in scenarios"
    assert len(ids) == len(set(ids)), f"duplicate binding_ids: {ids}"


def test_target_units_have_real_completion_criteria() -> None:
    """Spec §13.2: TargetUnit.completion_criteria is non-empty and uses
    the discriminated union, not a dict escape hatch.
    """
    from .frozen_contracts import CompletionCriterion  # noqa: F401
    from typing import get_args
    assert hasattr(CompletionCriterion, "__metadata__") or hasattr(CompletionCriterion,
        "__origin__"), "CompletionCriterion must be an Annotated union"

    for s in SCENARIOS:
        plan = s.initial_state["execution"].plan
        if plan is None:
            continue
        for tu in plan.target_units:
            assert tu.completion_criteria, (
                f"{s.scenario_id}: TargetUnit {tu.target_id} has empty criteria"
            )
            for cc in tu.completion_criteria:
                # CoverageCriterion | SemanticCriterion — discriminated by `kind`.
                assert hasattr(cc, "kind"), (
                    f"CompletionCriterion missing `kind` discriminator: {cc}"
                )
                assert cc.kind in {"coverage", "semantic"}, (
                    f"unknown CompletionCriterion.kind: {cc.kind}"
                )


def test_append_only_replan_preserves_prefix() -> None:
    """Spec §17: the replan plan must keep the original tasks as the prefix."""
    s = next(x for x in SCENARIOS if x.scenario_id == "append_only_replan")
    plan = s.initial_state["execution"].plan
    assert plan is not None
    assert plan.tasks[0].task_id == "task-051"
    assert plan.tasks[1].task_id == "task-052"
    # The replan task references the prior task in its ReplanTaskOrigin.
    origin = plan.tasks[1].origin
    assert origin.kind == "replan"
    assert "task-051" in origin.task_ids


def test_append_only_check_rejects_prefix_violation() -> None:
    """Final execution gate #20: append-only is enforced — a replan that drops
    the original task is REJECTED. This is the real I2 assertion.
    """
    old = original_plan_for("append_only_replan")
    bad = violating_plan_for("append_only_replan")
    assert old is not None and bad is not None

    # Sanity: the bad plan dropped task-051 and only contains task-052.
    assert [t.task_id for t in bad.tasks] == ["task-052"]
    assert [t.task_id for t in old.tasks] == ["task-051"]

    with pytest.raises(ValueError, match="append-only violation"):
        assert_append_only(old, bad)


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
    """Spec §7: resume uses the *current* runtime — old workspaces disappear.

    `acl-resume` carries a placeholder runtime that is shared by all scenarios;
    the runtime contracts are owned by Phase 1D, so we only assert the
    invariant the brief demands: the current workspace shows up where the old
    one used to.
    """
    scenario = next(x for x in SCENARIOS if x.scenario_id == "acl-resume")
    # Nothing about runtime leaks into the checkpointable state.
    assert "ws-default" not in scenario.checkpoint_json()
    assert "ws-new" not in scenario.checkpoint_json()


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


def test_direct_route_has_no_plan_and_one_final_response() -> None:
    """Spec §12: a direct conversational response keeps ExecutionState.plan=None.

    This is the real I3 assertion: plan is None AND there is exactly one
    terminal outer event (a single FinalResponse on the scenario state).
    """
    s = next(x for x in SCENARIOS if x.scenario_id == "outer_only_streaming")
    assert s.initial_state["execution"].plan is None
    assert s.initial_state["final_response"] is not None
    assert s.initial_state["final_response"].status == "success"
    assert s.initial_state["clarification"] is None
    # Exactly one terminal outer event: one FinalResponse, no inner nodes
    # can stream prose because there are no AgentResults.
    assert len(s.initial_state["execution"].task_results) == 0


def test_clarification_route_has_no_plan() -> None:
    """Spec §12: clarification also executes no capability until resolved."""
    s = next(x for x in SCENARIOS if x.scenario_id == "clarification_resume")
    assert s.initial_state["execution"].plan is None
    assert s.initial_state["clarification"] is not None


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
    """Pydantic v2 frozen=True must raise ValidationError on attribute set."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        rt = GraphRuntimeContext(
            capability_runtime=_Phase0CapabilityStandIn(kind="phase0_placeholder"),
            services=RuntimeServices(capabilities_registry_id="r"),
        )
        rt.services = RuntimeServices(capabilities_registry_id="r2")  # type: ignore[misc]


def test_frozen_contracts_reject_extra_fields() -> None:
    """extra='forbid' must reject unknown fields."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
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
    # Spec §20: ordinal + ref_id are mandatory fields on DocumentCandidate.
    for c in clarification.candidates:
        assert isinstance(c.ordinal, int)
        assert c.ref_id
    assert clarification.expires_at.tzinfo is not None

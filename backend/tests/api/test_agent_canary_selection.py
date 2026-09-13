"""Task 7B — deterministic v2 canary selection tests (TDD, failing first).

Covers the server-owned selection contract (R8): control row read per
request, workspace allowlist before percentage, deterministic bucket over
authenticated workspace + persisted request ID + salt, ordinary headers
ignored, admin-only override, kill-switch authority, the deterministically-
known Write endpoint pre-exclusion, post-router fallback for
write/evaluate/compliance/unsupported routes, active-run registration /
distributed cancellation before each scheduler dispatch, and the single
TaskScheduler ownership chain.
"""
from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
OTHER_WORKSPACE_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
REQUEST_ID = "req-persisted-0001"
SALT = "test-bucket-salt"


class FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class FakeSession:
    """Minimal async session: returns the CURRENT control row per call."""

    def __init__(self):
        from app.models.agent_rollout_control import AgentRolloutControl

        self.row = AgentRolloutControl(
            id=1,
            enabled=True,
            shadow_percent=0,
            canary_percent=100,
            canary_workspaces=[],
            kill_switch=False,
            updated_by=None,
            version=1,
        )
        self.execute_calls = 0

    async def execute(self, *args, **kwargs):
        self.execute_calls += 1
        return FakeResult(self.row)


def make_env(**overrides):
    from app.services.agent.rollout_control import CanaryEnv

    base = {
        "enabled": True,
        "canary_percent": 100.0,
        "canary_workspaces": (),
        "bucket_salt": SALT,
    }
    base.update(overrides)
    return CanaryEnv(**base)


def make_control(session, **overrides):
    for key, value in overrides.items():
        setattr(session.row, key, value)
    return session.row


# ---------------------------------------------------------------------------
# Deterministic bucket
# ---------------------------------------------------------------------------


def test_bucket_is_deterministic_for_same_inputs():
    from app.services.agent.rollout_control import deterministic_bucket

    first = deterministic_bucket(
        workspace_id=str(WORKSPACE_ID), request_id=REQUEST_ID, salt=SALT
    )
    second = deterministic_bucket(
        workspace_id=str(WORKSPACE_ID), request_id=REQUEST_ID, salt=SALT
    )
    assert first == second
    assert 0.0 <= first < 100.0


def test_bucket_differs_across_request_ids():
    from app.services.agent.rollout_control import deterministic_bucket

    buckets = {
        deterministic_bucket(
            workspace_id=str(WORKSPACE_ID),
            request_id=f"req-{index:04d}",
            salt=SALT,
        )
        for index in range(25)
    }
    assert len(buckets) > 1


def test_bucket_matches_sha256_vector():
    import hashlib

    from app.services.agent.rollout_control import deterministic_bucket

    digest = hashlib.sha256(
        f"{WORKSPACE_ID}|{REQUEST_ID}|{SALT}".encode("utf-8")
    ).hexdigest()
    expected = (int(digest[:16], 16) % 10000) / 100.0
    assert (
        deterministic_bucket(
            workspace_id=str(WORKSPACE_ID), request_id=REQUEST_ID, salt=SALT
        )
        == expected
    )


# ---------------------------------------------------------------------------
# Selection: control row read per request, allowlist before percentage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_control_row_is_read_per_request_not_cached():
    from app.services.agent import rollout_control

    session = FakeSession()
    first = await rollout_control.read_control_row(session)
    assert session.execute_calls == 1
    assert first is not None and first.enabled is True
    # Operator retunes between requests: the next read must see it.
    session.row.canary_percent = 0
    second = await rollout_control.read_control_row(session)
    assert session.execute_calls == 2
    assert second is not None and second.canary_percent == 0


def test_allowlist_checked_before_percentage():
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(
        session, canary_percent=100, canary_workspaces=[str(OTHER_WORKSPACE_ID)]
    )
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(),
    )
    assert arm == "v1"


def test_allowlisted_workspace_at_100_percent_gets_v2():
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(
        session, canary_percent=100, canary_workspaces=[str(WORKSPACE_ID)]
    )
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(),
    )
    assert arm == "v2"


def test_zero_percent_is_always_v1():
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(session, canary_percent=0)
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(canary_percent=0.0),
    )
    assert arm == "v1"


def test_env_ceiling_caps_db_percent():
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(session, canary_percent=100)
    # Env ceiling 0 keeps everything on v1 even though the DB row says 100.
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(canary_percent=0.0),
    )
    assert arm == "v1"


def test_disabled_env_flag_forces_v1():
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(session, canary_percent=100)
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(enabled=False),
    )
    assert arm == "v1"


def test_disabled_control_row_forces_v1():
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(session, enabled=False, canary_percent=100)
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(),
    )
    assert arm == "v1"


def test_kill_switch_forces_v1_over_everything():
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(session, canary_percent=100, kill_switch=True)
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(),
        admin_override="v2",
        is_superadmin=True,
    )
    assert arm == "v1"


def test_missing_control_row_fails_closed_to_v1():
    from app.services.agent.rollout_control import select_canary_arm

    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=None,
        env=make_env(),
    )
    assert arm == "v1"


# ---------------------------------------------------------------------------
# Ordinary headers ignored; admin-only override retained
# ---------------------------------------------------------------------------


def test_selection_takes_no_header_input():
    from app.services.agent.rollout_control import select_canary_arm

    params = set(inspect.signature(select_canary_arm).parameters)
    assert not {name for name in params if "header" in name.lower()}


def test_admin_override_is_superadmin_only():
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(session, canary_percent=0)
    # Non-admin callers cannot lift themselves onto v2 via the override.
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(canary_percent=0.0),
        admin_override="v2",
        is_superadmin=False,
    )
    assert arm == "v1"
    # An authenticated superadmin override is honored when the kill
    # switch is off.
    session2 = FakeSession()
    make_control(session2, canary_percent=0, kill_switch=False)
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session2.row,
        env=make_env(enabled=True, canary_percent=100.0),
        admin_override="v2",
        is_superadmin=True,
    )
    assert arm == "v2"


def test_admin_override_rejects_invalid_values():
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    with pytest.raises(ValueError):
        select_canary_arm(
            workspace_id=str(WORKSPACE_ID),
            request_id=REQUEST_ID,
            control=session.row,
            env=make_env(),
            admin_override="v9",
            is_superadmin=True,
        )


# ---------------------------------------------------------------------------
# Rollout-eligibility tests (named, binding)
# ---------------------------------------------------------------------------


def test_rollout_100_percent_still_routes_write_to_v1():
    """A deterministically-known Write endpoint never buckets to v2."""
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(session, canary_percent=100)
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(),
        is_write_endpoint=True,
    )
    assert arm == "v1"


def test_rollout_100_percent_still_routes_evaluate_to_v1():
    """A v2 candidate resolving to evaluate/compliance falls back to v1."""
    from app.services.agent.rollout_control import requires_v1_fallback
    from app.services.agents.v2.contracts.routing import (
        QueryAnalysis,
        RouteDecision,
    )

    analysis = QueryAnalysis(work_type="evaluate", domains=("document",))
    decision = RouteDecision(
        route="complex_research", reason_code="compliance_evaluation"
    )
    assert requires_v1_fallback(analysis, decision) is True


def test_supported_compare_uses_v2_at_100_percent_eligible_rollout():
    """A supported compare route stays on the v2 candidate at full rollout."""
    from app.services.agent.rollout_control import (
        requires_v1_fallback,
        select_canary_arm,
    )
    from app.services.agents.v2.contracts.routing import (
        QueryAnalysis,
        RouteDecision,
    )

    session = FakeSession()
    make_control(session, canary_percent=100)
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(),
    )
    assert arm == "v2"
    analysis = QueryAnalysis(work_type="compare", domains=("document",))
    decision = RouteDecision(route="complex_research", reason_code="comparison")
    assert requires_v1_fallback(analysis, decision) is False


def test_write_domain_requires_v1_fallback():
    from app.services.agent.rollout_control import requires_v1_fallback
    from app.services.agents.v2.contracts.routing import (
        QueryAnalysis,
        RouteDecision,
    )

    analysis = QueryAnalysis(work_type="retrieve", domains=("document", "write"))
    decision = RouteDecision(
        route="complex_research", reason_code="simple_write_operation"
    )
    assert requires_v1_fallback(analysis, decision) is True


def test_unknown_route_fails_closed_to_v1():
    from app.services.agent.rollout_control import requires_v1_fallback

    assert requires_v1_fallback(None, None) is True
    assert requires_v1_fallback({"work_type": "nope"}, {"route": "nope"}) is True


# ---------------------------------------------------------------------------
# Active-run registration + distributed cancellation before dispatch
# ---------------------------------------------------------------------------


def _scheduler_plan():
    from app.services.agents.v2.contracts.capability import PeopleLookupInput
    from app.services.agents.v2.contracts.planning import (
        InitialTaskOrigin,
        TaskPlan,
        TaskSpec,
    )

    return TaskPlan(
        contract_version="2.0",
        plan_id="plan-canary",
        goal="Ai là Nguyễn Văn A?",
        target_units=(),
        tasks=(
            TaskSpec(
                task_id="T1",
                capability="people.lookup",
                task_objective="Ai là Nguyễn Văn A?",
                input=PeopleLookupInput(kind="people.lookup", query="Nguyễn Văn A"),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )


def _graph_runtime(registry):
    from app.services.agents.v2.contracts.capability import CapabilityRuntimeContext
    from app.services.agents.v2.contracts.state import (
        GraphRuntimeContext,
        RuntimeServices,
    )
    return GraphRuntimeContext(
        capability_runtime=CapabilityRuntimeContext(
            request_id="req-canary-1",
            run_id=f"run-canary-{uuid4().hex[:8]}",
            user_id=USER_ID,
            workspace_ids=(WORKSPACE_ID,),
            can_read_people=True,
            allowed_capabilities=frozenset({"people.lookup"}),
            deadline_at=datetime.now(UTC) + timedelta(minutes=5),
        ),
        services=RuntimeServices(
            capability_registry=registry, retention_leases=None
        ),
    )


def _people_registry(canned_result=None):
    import asyncio

    from app.services.agents.v2.capabilities import (
        CapabilityRegistration,
        build_capability_registry,
    )
    from app.services.agents.v2.contracts.base import CONTRACT_VERSION
    from app.services.agents.v2.contracts.capability import (
        CapabilityDescriptor,
        CapabilityRuntimeContext,
    )
    from app.services.agents.v2.contracts.evidence import EvidenceUseRef
    from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult

    class StubCapability:
        def __init__(self):
            self.descriptor = CapabilityDescriptor(
                name="people.lookup",  # type: ignore[arg-type]
                domain="people",  # type: ignore[arg-type]
                operation_type="lookup",
                supports_parallel=False,
            )
            self.calls: list = []

        async def execute(self, request, runtime):
            self.calls.append((request, runtime))
            if canned_result is not None:
                return canned_result
            from app.services.agents.v2.contracts.capability import (
                PeopleLookupOutput,
            )

            return AgentResult(
                contract_version=CONTRACT_VERSION,
                task_id=request.task_id,
                status="success",
                data=PeopleLookupOutput(kind="people.lookup", matched=True),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )

    stub = StubCapability()
    runtime = CapabilityRuntimeContext(
        request_id="req-canary-1",
        run_id="run-canary",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=True,
        allowed_capabilities=frozenset({"people.lookup"}),
        deadline_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    registry = build_capability_registry(
        [CapabilityRegistration(capability=stub)], runtime
    )
    return registry, stub


@pytest.mark.asyncio
async def test_active_run_registered_before_dispatch():
    from app.services.agents.v2.execution import scheduler as scheduler_module
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    registry, stub = _people_registry()
    runtime = _graph_runtime(registry)
    run_id = runtime.capability_runtime.run_id
    scheduler_module.unregister_active_run(run_id)
    report = await TaskScheduler(registry).execute(
        _scheduler_plan(), runtime, prior_results=(), bindings=None
    )
    assert len(report.results) == 1
    assert stub.calls != []
    assert scheduler_module.is_run_active(run_id) is True
    scheduler_module.unregister_active_run(run_id)


@pytest.mark.asyncio
async def test_distributed_cancellation_stops_dispatch_before_execute():
    import asyncio

    from app.services.agents.v2.execution import scheduler as scheduler_module
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    registry, stub = _people_registry()
    runtime = _graph_runtime(registry)
    run_id = runtime.capability_runtime.run_id
    scheduler_module.register_active_run(run_id)
    scheduler_module.request_run_cancellation(run_id)
    try:
        with pytest.raises(asyncio.CancelledError):
            await TaskScheduler(registry).execute(
                _scheduler_plan(), runtime, prior_results=(), bindings=None
            )
    finally:
        scheduler_module.unregister_active_run(run_id)
    assert stub.calls == []


@pytest.mark.asyncio
async def test_v1_fallback_guard_fires_before_capability_execution():
    from app.services.agents.v2.execution.scheduler import (
        TaskScheduler,
        V1FallbackRequired,
    )

    registry, stub = _people_registry()
    runtime = _graph_runtime(registry)
    with pytest.raises(V1FallbackRequired):
        await TaskScheduler(registry).execute(
            _scheduler_plan(),
            runtime,
            prior_results=(),
            bindings=None,
            v1_fallback_guard=lambda: True,
        )
    assert stub.calls == []


@pytest.mark.asyncio
async def test_single_scheduler_ownership_chain_preserved():
    from app.services.agents.v2.execution import scheduler as scheduler_module

    assert hasattr(scheduler_module, "shared_scheduler_for")
    assert hasattr(scheduler_module, "TaskScheduler")
    assert hasattr(scheduler_module, "execute_ready_tasks")
    # No second scheduler, no plain-Python execute loop, no gateway: the
    # only Scheduler-named classes are the dispatcher, its base error, and
    # the typed v1-fallback signal (a SchedulerError subclass).
    import inspect

    from app.services.agents.v2.execution.scheduler import V1FallbackRequired

    names = sorted(
        name
        for name, member in inspect.getmembers(scheduler_module, inspect.isclass)
        if "Scheduler" in name or "Fallback" in name
    )
    assert names == ["SchedulerError", "TaskScheduler", "V1FallbackRequired"]
    assert issubclass(V1FallbackRequired, scheduler_module.SchedulerError)

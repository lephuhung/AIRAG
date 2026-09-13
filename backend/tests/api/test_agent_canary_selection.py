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


# ---------------------------------------------------------------------------
# Task 7B fix round 1 (R73-R79): production-wired fallback, salt, ordering,
# lifecycle, telegram, real-path eligibility
# ---------------------------------------------------------------------------


def test_empty_bucket_salt_fails_closed_when_canary_enabled():
    """R77: canary enabled with an empty salt must fail closed to v1."""
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(session, canary_percent=100)
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(canary_percent=100.0, bucket_salt=""),
    )
    assert arm == "v1"


def test_empty_bucket_salt_ok_while_disabled():
    """R77: the empty default stays fine while canary is disabled (v1)."""
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(session, canary_percent=0, enabled=False)
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(enabled=False, canary_percent=0.0, bucket_salt=""),
    )
    assert arm == "v1"


def test_workspace_allowlist_env_and_db_both_ceilings():
    """Minor: a non-empty DB list must not replace the env ceiling (intersect)."""
    from app.services.agent.rollout_control import select_canary_arm

    session = FakeSession()
    make_control(session, canary_percent=100, canary_workspaces=[str(OTHER_WORKSPACE_ID)])
    arm = select_canary_arm(
        workspace_id=str(WORKSPACE_ID),
        request_id=REQUEST_ID,
        control=session.row,
        env=make_env(canary_workspaces=(str(WORKSPACE_ID),)),
    )
    assert arm == "v1"


def _complex_execute_state(work_type, domains, route, reason):
    from app.services.agents.v2.complex_research_graph import ComplexResearchState
    from app.services.agents.v2.contracts.binding import DocumentBindingSet
    from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
    from app.services.agents.v2.contracts.semantic import SemanticContext

    return ComplexResearchState(
        contract_version="2.0",
        semantic=SemanticContext(
            contextualized_query="q",
            normalized_query="q",
            abbreviations=(),
            coreferences=(),
            document_refs=(),
            person_refs=(),
            section_refs=(),
            blocking_ambiguities=(),
        ),
        bindings=DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        query_analysis=QueryAnalysis(work_type=work_type, domains=tuple(domains)),
        route_decision=RouteDecision(route=route, reason_code=reason),
        plan=_scheduler_plan(),
        task_results=(),
        evaluation=None,
        replans_remaining=1,
        unavailable=None,
        materialized_new_task=False,
        people_scalar_available={},
        reduce_spec=None,
        discovery_deferred=(),
    )


@pytest.mark.asyncio
async def test_production_complex_execute_falls_back_on_write_route():
    """R73/R79: the real complex execute node falls back on a write route.

    Goes through the production ``complex_execute_node`` with real router
    state (frozen QueryAnalysis/RouteDecision) and the sole scheduler call:
    zero capability calls and zero v2 output (typed V1FallbackRequired).
    """
    from app.services.agents.v2.complex_research_graph import complex_execute_node
    from app.services.agents.v2.execution.scheduler import V1FallbackRequired

    registry, stub = _people_registry()
    runtime = _graph_runtime(registry)
    state = _complex_execute_state(
        "retrieve", ("document", "write"), "complex_research", "simple_write_operation"
    )
    with pytest.raises(V1FallbackRequired):
        await complex_execute_node(state, runtime)
    assert stub.calls == []


@pytest.mark.asyncio
async def test_production_complex_execute_falls_back_on_evaluate_route():
    """R73/R79: evaluate/legal/compliance falls back via the real execute node."""
    from app.services.agents.v2.complex_research_graph import complex_execute_node
    from app.services.agents.v2.execution.scheduler import V1FallbackRequired

    registry, stub = _people_registry()
    runtime = _graph_runtime(registry)
    state = _complex_execute_state(
        "evaluate", ("document",), "complex_research", "compliance_evaluation"
    )
    with pytest.raises(V1FallbackRequired):
        await complex_execute_node(state, runtime)
    assert stub.calls == []


@pytest.mark.asyncio
async def test_production_complex_execute_runs_supported_compare():
    """R79: a supported compare route dispatches through the sole scheduler."""
    from app.services.agents.v2.complex_research_graph import complex_execute_node

    registry, stub = _people_registry()
    runtime = _graph_runtime(registry)
    state = _complex_execute_state(
        "compare", ("document",), "complex_research", "comparison"
    )
    result = await complex_execute_node(state, runtime)
    assert stub.calls != []
    assert len(tuple(result.get("task_results", ()))) == 1


@pytest.mark.asyncio
async def test_async_redis_registration_is_awaited_before_cancel_check():
    """R75: the Redis SET is awaited (ordered before dispatch), not fire-and-forget."""
    import app.core.redis_client as redis_client
    from app.services.agents.v2.execution import scheduler as scheduler_module

    events: list[str] = []
    state = {"active_set_done": False}

    class AsyncRedisDouble:
        async def set(self, *args, **kwargs):
            events.append("set")
            state["active_set_done"] = True
            return True

        async def exists(self, *args, **kwargs):
            events.append("exists")
            # The SET must have COMPLETED before the check runs; a
            # fire-and-forget task would observe False here.
            assert state["active_set_done"] is True
            return 0

        async def delete(self, *args, **kwargs):
            events.append("delete")
            return 1

    double = AsyncRedisDouble()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(redis_client, "is_redis_enabled", lambda: True)
    monkeypatch.setattr(redis_client, "get_redis", lambda: double)
    try:
        run_id = f"run-order-{uuid4().hex[:8]}"
        await scheduler_module.register_active_run_async(run_id)
        assert await scheduler_module.is_run_cancel_requested_async(run_id) is False
        assert events.index("set") < events.index("exists")
        await scheduler_module.request_run_cancellation_async(run_id)
        assert await scheduler_module.is_run_cancel_requested_async(run_id) is True
    finally:
        monkeypatch.undo()
        scheduler_module.unregister_active_run(run_id)


@pytest.mark.asyncio
async def test_pre_dispatch_guards_await_registration_before_dispatch():
    """R75: the dispatch boundary awaits Redis registration before executing."""
    import app.core.redis_client as redis_client
    from app.services.agents.v2.execution import scheduler as scheduler_module
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    completions: list[str] = []

    class AsyncRedisDouble:
        async def set(self, *args, **kwargs):
            completions.append("set-done")
            return True

        async def exists(self, *args, **kwargs):
            # Registration must be complete before the cancel check.
            assert "set-done" in completions
            completions.append("exists-done")
            return 0

        async def delete(self, *args, **kwargs):
            return 1

    double = AsyncRedisDouble()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(redis_client, "is_redis_enabled", lambda: True)
    monkeypatch.setattr(redis_client, "get_redis", lambda: double)
    try:
        registry, stub = _people_registry()
        runtime = _graph_runtime(registry)
        report = await TaskScheduler(registry).execute(
            _scheduler_plan(), runtime, prior_results=(), bindings=None
        )
        assert len(report.results) == 1
        assert completions.index("set-done") < completions.index("exists-done")
    finally:
        monkeypatch.undo()
        scheduler_module.unregister_active_run(
            runtime.capability_runtime.run_id
        )


@pytest.mark.asyncio
async def test_active_run_refresh_extends_registration():
    """Active-run TTL is refreshable for long/resumed runs (R75 lifecycle)."""
    import app.core.redis_client as redis_client
    from app.services.agents.v2.execution import scheduler as scheduler_module

    calls: list[tuple] = []

    class AsyncRedisDouble:
        async def set(self, *args, **kwargs):
            calls.append(("set", args, kwargs))
            return True

        async def exists(self, *args, **kwargs):
            return 0

        async def delete(self, *args, **kwargs):
            return 1

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(redis_client, "is_redis_enabled", lambda: True)
    monkeypatch.setattr(redis_client, "get_redis", lambda: AsyncRedisDouble())
    try:
        run_id = f"run-refresh-{uuid4().hex[:8]}"
        await scheduler_module.register_active_run_async(run_id)
        await scheduler_module.refresh_active_run_async(run_id)
        assert scheduler_module.is_run_active(run_id) is True
        set_calls = [call for call in calls if call[0] == "set"]
        assert len(set_calls) == 2
        await scheduler_module.unregister_active_run_async(run_id)
        assert scheduler_module.is_run_active(run_id) is False
    finally:
        monkeypatch.undo()


def test_operator_cancel_path_flags_run():
    """R75: the operator cancel entrypoint flags the run for cancellation."""
    import asyncio

    from app.api.agent_admin import cancel_agent_run
    from app.services.agents.v2.execution import scheduler as scheduler_module

    async def _run():
        run_id = f"run-op-{uuid4().hex[:8]}"
        scheduler_module.register_active_run(run_id)

        class FakeUser:
            pass

        await cancel_agent_run(run_id, user=FakeUser(), db=None)
        assert scheduler_module.is_run_cancel_requested(run_id) is True
        scheduler_module.unregister_active_run(run_id)

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())


def test_telegram_fallback_resolves_v1_graph():
    """R76: Telegram V1FallbackRequired serves the v1 graph (not v2)."""
    import asyncio

    from app.services.agents.v2.execution.scheduler import V1FallbackRequired
    from app.services.integrations import telegram_service as telegram_module

    resolved: list[str] = []

    async def fake_resolve(version):
        resolved.append(version)
        return f"graph-{version}"

    async def collect_raises(**kwargs):
        raise V1FallbackRequired("v1-only route")

    async def _run():
        graph, events = await telegram_module._collect_v2_or_fallback_to_v1(
            graph="graph-v2",
            version="v2",
            collect_fn=collect_raises,
            resolve_fn=fake_resolve,
            collect_kwargs={},
        )
        return graph, events

    graph, events = (
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())
    )
    assert graph == "graph-v1"
    assert events is None
    assert resolved == ["v1"]


# ---------------------------------------------------------------------------
# Task 7B fix round 2: real router/ingress fallback proof (Critical 1 + Important 12),
# awaited operator cancellation (Important 3), run-lifetime heartbeat (Important 4)
# ---------------------------------------------------------------------------


def _router_semantic(normalized_query):
    from app.services.agents.v2.contracts.semantic import SemanticContext

    return SemanticContext(
        contextualized_query=normalized_query,
        normalized_query=normalized_query,
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def _supervisor_state_with_plan(semantic, plan):
    from app.services.agents.v2.contracts.binding import DocumentBindingSet
    from app.services.agents.v2.contracts.conversation import ConversationContext
    from app.services.agents.v2.contracts.request import RequestContext
    from app.services.agents.v2.contracts.state import ExecutionState, SupervisorV2State

    return SupervisorV2State(
        contract_version="2.0",
        request=RequestContext(
            contract_version="2.0",
            request_id="req-router-1",
            thread_id="thread-router-1",
            original_query=semantic.normalized_query,
            known_documents=(),
        ),
        conversation=ConversationContext(
            summary="",
            active_entities=(),
            last_focus=None,
            recent_turns=(),
        ),
        semantic=semantic,
        bindings=DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        query_analysis=None,
        route_decision=None,
        execution=ExecutionState(
            plan=plan, task_results=(), evidence_evaluation=None
        ),
        clarification=None,
        final_response=None,
    )


@pytest.mark.asyncio
async def test_operator_cancel_awaits_distributed_write():
    """Important 3: the operator endpoint awaits the Redis write before
    returning (ordering proven through the endpoint, not the local set)."""
    import app.core.redis_client as redis_client
    from app.api.agent_admin import cancel_agent_run
    from app.services.agents.v2.execution import scheduler as scheduler_module

    store: dict = {}
    events: list[str] = []

    class DictBackedRedis:
        async def set(self, key, value, ex=None):
            events.append("set")
            store[key] = value
            return True

        async def exists(self, *keys):
            events.append("exists")
            # Ordering: the SET must have COMPLETED before any read observes.
            assert "set" in events
            return sum(1 for key in keys if key in store)

        async def delete(self, *keys):
            for key in keys:
                store.pop(key, None)
            return 1

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(redis_client, "is_redis_enabled", lambda: True)
    monkeypatch.setattr(redis_client, "get_redis", lambda: DictBackedRedis())
    try:
        run_id = f"run-op-await-{uuid4().hex[:8]}"

        class FakeUser:
            pass

        response = await cancel_agent_run(run_id, user=FakeUser(), db=None)
        assert response.cancel_requested is True
        assert "set" in events
        # Cross-process view: drop the process-local flag; the distributed
        # (Redis) flag alone must still read back as requested.
        scheduler_module._local_cancel_requests.discard(run_id)
        assert (
            await scheduler_module.is_run_cancel_requested_async(run_id) is True
        )
    finally:
        monkeypatch.undo()
        scheduler_module.unregister_active_run(run_id)


@pytest.mark.asyncio
async def test_active_run_heartbeat_refreshes_for_run_lifetime():
    """Important 4: the run-lifetime heartbeat keeps the active-run key alive
    across TTL-length dispatches and stops at the terminal boundary."""
    import asyncio

    import app.core.redis_client as redis_client
    from app.services.agents.v2.execution import scheduler as scheduler_module

    sets: list = []

    class CountingRedis:
        async def set(self, *args, **kwargs):
            sets.append(args)
            return True

        async def exists(self, *args, **kwargs):
            return 0

        async def delete(self, *args, **kwargs):
            return 1

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(redis_client, "is_redis_enabled", lambda: True)
    monkeypatch.setattr(redis_client, "get_redis", lambda: CountingRedis())
    try:
        run_id = f"run-heartbeat-{uuid4().hex[:8]}"
        heartbeat = scheduler_module.start_active_run_heartbeat(
            run_id, interval_seconds=0.02
        )
        assert heartbeat is not None
        await asyncio.sleep(0.09)
        assert len(sets) >= 2
        await heartbeat.stop()
        frozen = len(sets)
        await asyncio.sleep(0.05)
        assert len(sets) == frozen
        await heartbeat.stop()  # idempotent
    finally:
        monkeypatch.undo()
        scheduler_module.unregister_active_run(run_id)


# NOTE (fix round 3, R83): the hand-sequenced ``route_node`` -> state-builder
# -> ``complex_execute_node`` tests and the direct-raise stub-graph streaming
# test that lived here are REMOVED — R83 rules they do not satisfy the
# fallback/eligibility proof. The R83 proof suite below replaces them with
# strictly stronger coverage (same assertions plus real ingress entry,
# canary-100 selection as entrypoints apply it, and the legal case).


# ---------------------------------------------------------------------------
# Task 7B fix round 3 (R83): fallback/eligibility proof through the REAL
# serving ingress adapter with canary-100 selection as entrypoints apply it.
#
# The proof enters via ``stream_v2_turn_events`` driving a COMPILED graph
# whose route node is the production ``route_node`` (real
# analyze_query/decide_route) and whose execute node is the production
# ``complex_execute_node`` (sole scheduler + production guard, stub
# capability). Canary selection uses the production ``resolve_serving_arm``
# entrypoint path at CANARY_PERCENT=100. No test below hand-sequences
# route -> state-builder -> execute in the test body, and no graph
# directly raises fallback: every fallback fires from the production
# guard on a real router outcome.
# ---------------------------------------------------------------------------


def _patch_canary_100_settings():
    """Apply CANARY_PERCENT=100 ceilings the way the environment does."""
    import app.core.config as _config

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(_config.settings, "NEXUSRAG_AGENT_V2_ENABLED", True)
    monkeypatch.setattr(_config.settings, "NEXUSRAG_AGENT_V2_CANARY_PERCENT", 100.0)
    monkeypatch.setattr(_config.settings, "NEXUSRAG_AGENT_V2_CANARY_WORKSPACES", "")
    monkeypatch.setattr(_config.settings, "NEXUSRAG_AGENT_V2_BUCKET_SALT", SALT)
    return monkeypatch


def _canary_100_session():
    """Control row as the operator seeds it for a full canary window."""
    session = FakeSession()
    make_control(
        session,
        enabled=True,
        canary_percent=100,
        canary_workspaces=[],
        kill_switch=False,
    )
    return session


async def _resolve_arm_at_canary_100(*, request_id: str):
    """Production arm selection exactly as the chat entrypoints apply it."""
    from app.services.agent.rollout_control import resolve_serving_arm

    return await resolve_serving_arm(
        db=_canary_100_session(),
        workspace_ids=[WORKSPACE_ID],
        request_id=request_id,
        is_write_endpoint=False,
    )


def _canary_serving_graph(plan, registry, runtime_context):
    """Compiled serving-path graph: REAL router -> REAL execute node.

    The ``route`` node runs the production ``route_node`` (real
    ``analyze_query``/``decide_route`` over the query text); the
    ``execute`` node runs the production ingress mapping
    (``build_complex_research_state``) plus the production
    ``complex_execute_node`` (sole scheduler + production
    ``_production_v1_fallback_guard``). Capabilities are stubbed (a stub
    capability is explicitly acceptable); routing and guarding are not.
    """
    from typing import TypedDict

    from langgraph.graph import StateGraph
    from langgraph.runtime import Runtime

    from app.services.agents.v2.complex_research_graph import (
        build_complex_research_state,
        complex_execute_node,
    )
    from app.services.agents.v2.nodes.routing import route_node

    # NOTE: ``object`` (not ``Any``) — langgraph resolves the state-schema
    # annotations in the test-module namespace, where a function-local
    # ``Any`` import is invisible (NameError at graph build time).
    class _ServingState(TypedDict, total=False):
        query: str
        supervisor: object
        query_analysis: object
        route_decision: object
        task_results: object
        final_response: object

    async def _route(state: _ServingState) -> dict:
        supervisor = _supervisor_state_with_plan(
            _router_semantic(state["query"]), plan
        )
        update = await route_node(
            supervisor, Runtime(context=runtime_context)
        )
        merged = dict(supervisor)
        merged.update(update)
        return {
            "supervisor": merged,
            "query_analysis": update["query_analysis"],
            "route_decision": update["route_decision"],
        }

    async def _execute(state: _ServingState) -> dict:
        from app.services.agents.v2.contracts.response import FinalResponse

        child_state = build_complex_research_state(state["supervisor"])
        result = await complex_execute_node(child_state, runtime_context)
        return {
            "task_results": tuple(result.get("task_results", ())),
            "final_response": FinalResponse(
                contract_version="2.0",
                status="success",
                content="Kết quả so sánh hai phương án.",
                citations=(),
            ),
        }

    graph = StateGraph(_ServingState)
    graph.add_node("route", _route)
    graph.add_node("execute", _execute)
    graph.set_entry_point("route")
    graph.add_edge("route", "execute")
    graph.set_finish_point("execute")
    return graph.compile()


async def _drive_serving_adapter(query, *, request_id, thread_id):
    """Select the arm (production path, canary 100) then drive the REAL
    serving ingress adapter over the compiled serving-path graph.

    Returns ``(arm, stub, yielded)`` or raises ``V1FallbackRequired``
    (with ``yielded`` attached as ``exc.yielded``) for v1-only routes.
    """
    from app.services.agent.streaming import stream_v2_turn_events
    from app.services.agents.v2.execution.scheduler import V1FallbackRequired

    arm = await _resolve_arm_at_canary_100(request_id=request_id)
    registry, stub = _people_registry()
    runtime_context = _graph_runtime(registry)
    graph = _canary_serving_graph(_scheduler_plan(), registry, runtime_context)
    yielded: list = []
    try:
        async for event in stream_v2_turn_events(
            graph=graph,
            runtime_context=runtime_context,
            thread_id=thread_id,
            initial_state={"query": query},
        ):
            yielded.append(event)
    except V1FallbackRequired as exc:
        exc.yielded = yielded  # type: ignore[attr-defined]
        exc.capability_calls = list(stub.calls)  # type: ignore[attr-defined]
        raise
    return arm, stub, yielded


def _assert_zero_v2_output(yielded):
    assert all(
        event.get("event") not in ("token", "complete", "sources", "images")
        for event in yielded
    ), f"fallback leaked v2 user-visible output: {yielded!r}"


@pytest.mark.asyncio
async def test_ingress_write_falls_back_with_zero_output_at_canary_100():
    """R83: write -> fallback through the serving ingress at CANARY 100."""
    from app.services.agents.v2.execution.scheduler import V1FallbackRequired

    settings_patch = _patch_canary_100_settings()
    try:
        with pytest.raises(V1FallbackRequired) as excinfo:
            await _drive_serving_adapter(
                "Viết báo cáo tổng kết năm",
                request_id="req-ingress-write-1",
                thread_id="thread-ingress-write-1",
            )
    finally:
        settings_patch.undo()
    assert excinfo.value.capability_calls == []
    _assert_zero_v2_output(excinfo.value.yielded)


@pytest.mark.asyncio
async def test_ingress_evaluate_falls_back_with_zero_output_at_canary_100():
    """R83: evaluate -> fallback through the serving ingress at CANARY 100."""
    from app.services.agents.v2.execution.scheduler import V1FallbackRequired

    settings_patch = _patch_canary_100_settings()
    try:
        with pytest.raises(V1FallbackRequired) as excinfo:
            await _drive_serving_adapter(
                "Đánh giá tuân thủ quy định nội bộ",
                request_id="req-ingress-evaluate-1",
                thread_id="thread-ingress-evaluate-1",
            )
    finally:
        settings_patch.undo()
    assert excinfo.value.capability_calls == []
    _assert_zero_v2_output(excinfo.value.yielded)


@pytest.mark.asyncio
async def test_ingress_legal_falls_back_with_zero_output_at_canary_100():
    """R83: legal/compliance -> fallback through the serving ingress at
    CANARY 100 (a legal query the real router resolves to
    compliance_evaluation)."""
    from app.services.agents.v2.execution.scheduler import V1FallbackRequired

    settings_patch = _patch_canary_100_settings()
    try:
        with pytest.raises(V1FallbackRequired) as excinfo:
            await _drive_serving_adapter(
                "Đánh giá tính pháp lý của hợp đồng mới",
                request_id="req-ingress-legal-1",
                thread_id="thread-ingress-legal-1",
            )
    finally:
        settings_patch.undo()
    assert excinfo.value.capability_calls == []
    _assert_zero_v2_output(excinfo.value.yielded)


@pytest.mark.asyncio
async def test_ingress_supported_compare_dispatches_at_canary_100():
    """R83: a supported compare query is selected to v2 at CANARY 100 and
    really dispatches through the sole scheduler to a user-visible
    terminal (not fallback)."""
    settings_patch = _patch_canary_100_settings()
    try:
        arm, stub, yielded = await _drive_serving_adapter(
            "So sánh hiệu quả hai phương án",
            request_id="req-ingress-compare-1",
            thread_id="thread-ingress-compare-1",
        )
    finally:
        settings_patch.undo()
    assert arm == "v2"
    assert stub.calls != []
    assert any(event.get("event") == "complete" for event in yielded)


# ---------------------------------------------------------------------------
# Task 7B fix round 3 (R82): the clarification-suspend path must stop the
# run heartbeat (leases stay); a resume owns a fresh heartbeat.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suspend_stops_heartbeat_and_resume_starts_fresh():
    """R82: suspending for clarification stops the run heartbeat cleanly
    (no stale background task); the next turn starts and owns a fresh one."""
    import asyncio
    from datetime import datetime as _dt
    from datetime import timezone as _tz
    from uuid import uuid4 as _uuid4

    import app.core.redis_client as redis_client
    from app.services.agent.streaming import stream_v2_turn_events
    from app.services.agents.v2.execution import scheduler as scheduler_module

    sets: list = []

    class CountingRedis:
        async def set(self, *args, **kwargs):
            sets.append(args)
            return True

        async def exists(self, *args, **kwargs):
            return 0

        async def delete(self, *args, **kwargs):
            return 1

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(redis_client, "is_redis_enabled", lambda: True)
    monkeypatch.setattr(redis_client, "get_redis", lambda: CountingRedis())

    handles: list = []
    real_start = scheduler_module.start_active_run_heartbeat

    def _capture_start(run_id, interval_seconds=60.0):
        handle = real_start(run_id, interval_seconds=0.02)
        handles.append(handle)
        return handle

    monkeypatch.setattr(
        scheduler_module, "start_active_run_heartbeat", _capture_start
    )
    try:
        from app.services.agents.v2.contracts.base import CONTRACT_VERSION
        from app.services.agents.v2.contracts.clarification import (
            ClarificationRequest,
        )

        pending = ClarificationRequest(
            contract_version=CONTRACT_VERSION,
            clarification_id=f"clar-{_uuid4().hex[:8]}",
            reason="semantic_ambiguity",
            question="Bạn muốn nói đến tài liệu nào?",
            unresolved_ref_ids=(),
            candidates=(),
            expires_at=_dt.now(_tz.utc).replace(year=2030),
        )

        class _SuspendGraph:
            async def ainvoke(self, *args, **kwargs):
                return {
                    "__interrupt__": (True,),
                    "clarification": pending,
                }

        registry, _stub = _people_registry()
        runtime = _graph_runtime(registry)
        yielded = [
            event
            async for event in stream_v2_turn_events(
                graph=_SuspendGraph(),
                runtime_context=runtime,
                thread_id=f"thread-suspend-{_uuid4().hex[:8]}",
                initial_state={"query": "ho so"},
            )
        ]
        assert any(event.get("event") == "complete" for event in yielded)
        assert len(handles) == 1
        # The suspend path stopped its heartbeat: no longer active and no
        # further background refreshes.
        assert handles[0] is not None
        assert handles[0].active is False
        frozen = len(sets)
        await asyncio.sleep(0.07)
        assert len(sets) == frozen
        # A resume (next turn on the thread) owns a fresh heartbeat.
        yielded_resume = [
            event
            async for event in stream_v2_turn_events(
                graph=_SuspendGraph(),
                runtime_context=runtime,
                thread_id=f"thread-suspend-{_uuid4().hex[:8]}",
                initial_state={"query": "ho so"},
            )
        ]
        assert any(event.get("event") == "complete" for event in yielded_resume)
        assert len(handles) == 2
        assert handles[1] is not None
        assert handles[1] is not handles[0]
        await handles[1].stop()
        assert handles[1].active is False
    finally:
        monkeypatch.undo()

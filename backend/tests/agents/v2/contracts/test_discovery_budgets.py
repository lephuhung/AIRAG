"""Discovery plan Task 8 — hard budgets, scheduler/replan seams, config.

Covers the Phase-1 dormant groundwork: the pure
``contracts.planning.build_research_budget_view`` probe-exemption proof, the
optional ``total_task_limit`` seams on the scheduler and the replan runtime
wrapper, and the ``V2_DISCOVERY_*`` / ``V2_TOTAL_MAX_TASKS`` settings with
their hard ceilings. No production graph behavior is enabled by any of this.
"""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    DocumentReadInput,
    DocumentSearchInput,
    PeopleLookupInput,
    PeopleLookupOutput,
)
from app.services.agents.v2.contracts.execution import AgentResult
from app.services.agents.v2.contracts.planning import (
    DiscoveryBudgetView,
    DiscoveryPolicy,
    DiscoveryProbeTaskOrigin,
    InitialTaskOrigin,
    ReplanTaskOrigin,
    ResearchBudgetView,
    TaskPlan,
    TaskSpec,
    build_research_budget_view,
)
from app.services.agents.v2.contracts.state import (
    GraphRuntimeContext,
    RuntimeServices,
)
from app.services.agents.v2.contracts.validation import ContractValidationError
from app.services.agents.v2.discovery_bootstrap.contracts import (
    DiscoveryCheckpoint,
    SearchProbe,
)
from app.services.agents.v2.execution.scheduler import (
    SchedulerError,
    TaskScheduler,
    execute_ready_tasks,
)
from app.services.agents.v2.replanning import (
    ReplanRejected,
    append_replan_tasks,
    validate_runtime_replan,
)

from . import factories

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DISCOVERY_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


# ---------------------------------------------------------------------------
# Settings: declared defaults and hard ceilings (discovery spec §17)
# ---------------------------------------------------------------------------

_V2_ENV_NAMES = (
    "V2_MAX_TASKS",
    "V2_MAX_PARALLEL_BRANCHES",
    "V2_MAX_REPLANS",
    "V2_MAX_DISCOVERED_DOCUMENTS",
    "V2_DISCOVERY_BOOTSTRAP_ENABLED",
    "V2_DISCOVERY_MAX_PROBES",
    "V2_DISCOVERY_MAX_ROUNDS",
    "V2_DISCOVERY_TOP_K",
    "V2_DISCOVERY_DEADLINE_SECONDS",
    "V2_DISCOVERY_SUMMARY_TARGETS",
    "V2_DISCOVERY_MAX_SUMMARY_TARGETS",
    "V2_DISCOVERY_CONFIDENCE_THRESHOLD",
    "V2_DISCOVERY_MARGIN_THRESHOLD",
    "V2_TOTAL_MAX_TASKS",
)


@pytest.fixture
def clean_v2_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _V2_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_discovery_defaults_and_flag(clean_v2_env: None) -> None:
    from app.core.config import Settings

    settings = Settings(_env_file=None)
    assert settings.V2_DISCOVERY_BOOTSTRAP_ENABLED is False
    assert settings.V2_DISCOVERY_MAX_PROBES == 5
    assert settings.V2_DISCOVERY_MAX_ROUNDS == 2
    assert settings.V2_DISCOVERY_TOP_K == 5
    assert settings.V2_DISCOVERY_DEADLINE_SECONDS == 15
    assert settings.V2_DISCOVERY_SUMMARY_TARGETS == 3
    assert settings.V2_DISCOVERY_MAX_SUMMARY_TARGETS == 5
    assert settings.V2_DISCOVERY_CONFIDENCE_THRESHOLD == 0.82
    assert settings.V2_DISCOVERY_MARGIN_THRESHOLD == 0.12
    assert (
        settings.V2_DISCOVERY_CALIBRATION_ARTIFACT
        == "/app/config/discovery-calibration.json"
    )
    assert settings.V2_TOTAL_MAX_TASKS == 13


@pytest.mark.parametrize(
    "overrides",
    (
        # Above every hard ceiling.
        {"V2_DISCOVERY_MAX_PROBES": 6},
        {"V2_DISCOVERY_MAX_ROUNDS": 3},
        {"V2_DISCOVERY_TOP_K": 6},
        {"V2_DISCOVERY_DEADLINE_SECONDS": 16},
        {"V2_DISCOVERY_SUMMARY_TARGETS": 6},
        {"V2_DISCOVERY_MAX_SUMMARY_TARGETS": 6},
        {"V2_DISCOVERY_CONFIDENCE_THRESHOLD": 1.01},
        {"V2_DISCOVERY_MARGIN_THRESHOLD": 1.01},
        # Below every hard floor.
        {"V2_DISCOVERY_MAX_PROBES": 0},
        {"V2_DISCOVERY_MAX_ROUNDS": 0},
        {"V2_DISCOVERY_TOP_K": 0},
        {"V2_DISCOVERY_DEADLINE_SECONDS": 0},
        {"V2_DISCOVERY_SUMMARY_TARGETS": 2},
        {"V2_DISCOVERY_MAX_SUMMARY_TARGETS": 2},
        {"V2_DISCOVERY_CONFIDENCE_THRESHOLD": -0.1},
        {"V2_DISCOVERY_MARGIN_THRESHOLD": -0.1},
        {"V2_TOTAL_MAX_TASKS": 0},
        # Existing planner limits are validated too.
        {"V2_MAX_TASKS": 0},
        {"V2_MAX_TASKS": -1},
        {"V2_MAX_PARALLEL_BRANCHES": 0},
        {"V2_MAX_REPLANS": -1},
        {"V2_MAX_DISCOVERED_DOCUMENTS": -1},
    ),
)
def test_discovery_settings_reject_invalid_values(overrides: dict) -> None:
    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_summary_min_greater_than_max_rejected() -> None:
    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            V2_DISCOVERY_SUMMARY_TARGETS=5,
            V2_DISCOVERY_MAX_SUMMARY_TARGETS=3,
        )
    # The boundary (min == max) is legal.
    assert (
        Settings(
            _env_file=None,
            V2_DISCOVERY_SUMMARY_TARGETS=4,
            V2_DISCOVERY_MAX_SUMMARY_TARGETS=4,
        ).V2_DISCOVERY_SUMMARY_TARGETS
        == 4
    )


def test_total_cap_must_cover_probes_plus_research() -> None:
    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            V2_DISCOVERY_MAX_PROBES=5,
            V2_MAX_TASKS=8,
            V2_TOTAL_MAX_TASKS=12,
        )
    # Exactly probes + research max is the legal boundary.
    assert (
        Settings(
            _env_file=None,
            V2_DISCOVERY_MAX_PROBES=5,
            V2_MAX_TASKS=8,
            V2_TOTAL_MAX_TASKS=13,
        ).V2_TOTAL_MAX_TASKS
        == 13
    )


# ---------------------------------------------------------------------------
# Pure budget builder (discovery spec §11.3): probe exemption must be proven
# ---------------------------------------------------------------------------

_PROBE_QUERY = "nđ13"


def _probe(*, probe_id: str = "p1", slot_id: str = "slot-1", round: int = 1) -> SearchProbe:
    return SearchProbe(
        probe_id=probe_id,
        slot_id=slot_id,
        query=_PROBE_QUERY,
        round=round,  # type: ignore[arg-type]
        origin="semantic",
    )


def _checkpoint(
    probes: tuple[SearchProbe, ...] | None = None,
) -> DiscoveryCheckpoint:
    if probes is None:
        probes = (_probe(),)
    return DiscoveryCheckpoint(
        discovery_id=DISCOVERY_ID,
        target_slots=(factories.target_slot(),),
        accepted_probes=probes,
        candidate_matches=(),
        slot_aggregates=(),
        selections=(),
        rounds_consumed=max((probe.round for probe in probes), default=0),
        probes_consumed=len(probes),
        status="searching",
        clarification_manifest=None,
    )


def _probe_task(
    task_id: str = "T-search",
    *,
    query: str = _PROBE_QUERY,
    probe_id: str = "p1",
    slot_id: str = "slot-1",
    round: int = 1,
    person_identifier: str | None = None,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability="document.search",
        task_objective="probe slot",
        input=DocumentSearchInput(
            kind="document.search",
            query=query,
            person_identifier=person_identifier,
        ),
        depends_on=(),
        origin=DiscoveryProbeTaskOrigin(
            kind="discovery_probe",
            probe_id=probe_id,
            slot_id=slot_id,
            round=round,  # type: ignore[arg-type]
        ),
    )


def _read_task(task_id: str = "T1", target_ids: tuple[str, ...] = ("t1",)) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability="document.read",
        task_objective="read target",
        input=DocumentReadInput(kind="document.read", target_ids=target_ids),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def _spoofed_read_probe(task_id: str = "T-spoof") -> TaskSpec:
    """A document.read task masquerading as a discovery probe."""
    return TaskSpec(
        task_id=task_id,
        capability="document.read",
        task_objective="spoofed probe",
        input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
        depends_on=(),
        origin=DiscoveryProbeTaskOrigin(
            kind="discovery_probe", probe_id="p1", slot_id="slot-1", round=1
        ),
    )


def _plan(*tasks: TaskSpec) -> TaskPlan:
    return TaskPlan(
        contract_version="2.0",
        plan_id="plan-1",
        goal="bootstrap",
        target_units=(factories.target_unit(),),
        tasks=tasks,
    )


def _budget(
    plan: TaskPlan,
    *,
    discovery_checkpoint: DiscoveryCheckpoint | None = None,
    max_tasks: int = 8,
    max_replans: int = 0,
    max_parallel_branches: int = 2,
    total_task_limit: int = 13,
) -> ResearchBudgetView:
    return build_research_budget_view(
        plan,
        discovery_checkpoint=discovery_checkpoint,
        max_tasks=max_tasks,
        max_replans=max_replans,
        max_parallel_branches=max_parallel_branches,
        total_task_limit=total_task_limit,
    )


def test_discovery_budget_view_shape() -> None:
    view = DiscoveryBudgetView(probes_remaining=5, rounds_remaining=2, top_k=5)
    assert (view.probes_remaining, view.rounds_remaining, view.top_k) == (5, 2, 5)


def test_valid_probe_is_budget_exempt() -> None:
    plan = _plan(_probe_task(), _read_task())
    budget = _budget(plan, discovery_checkpoint=_checkpoint())
    # 1 factual task of 8, probe exempt: min(8 - 1, 13 - 2) == 7.
    assert budget.max_tasks_remaining == 7


def test_valid_probe_normalized_query_still_exempt() -> None:
    plan = _plan(_probe_task(query="  NĐ13   định "), _read_task())
    checkpoint = _checkpoint(
        probes=(
            SearchProbe(
                probe_id="p1",
                slot_id="slot-1",
                query="nđ13 định",
                round=1,
                origin="semantic",
            ),
        )
    )
    budget = _budget(plan, discovery_checkpoint=checkpoint)
    assert budget.max_tasks_remaining == 7


def test_all_tasks_factual_without_probe_origins() -> None:
    plan = _plan(_read_task("T1"), _read_task("T2"))
    budget = _budget(plan)
    assert budget.max_tasks_remaining == 6


def test_read_task_spoofing_probe_origin_rejected() -> None:
    plan = _plan(_spoofed_read_probe())
    with pytest.raises(ContractValidationError):
        _budget(plan, discovery_checkpoint=_checkpoint())


def test_probe_origin_without_checkpoint_rejected() -> None:
    plan = _plan(_probe_task())
    with pytest.raises(ContractValidationError):
        _budget(plan, discovery_checkpoint=None)


def test_probe_origin_unknown_probe_rejected() -> None:
    plan = _plan(_probe_task(probe_id="ghost"))
    with pytest.raises(ContractValidationError):
        _budget(plan, discovery_checkpoint=_checkpoint())


def test_probe_origin_slot_mismatch_rejected() -> None:
    plan = _plan(_probe_task(slot_id="slot-2"))
    with pytest.raises(ContractValidationError):
        _budget(plan, discovery_checkpoint=_checkpoint())


def test_probe_origin_round_mismatch_rejected() -> None:
    plan = _plan(_probe_task(round=2))
    with pytest.raises(ContractValidationError):
        _budget(plan, discovery_checkpoint=_checkpoint())


def test_probe_origin_query_mismatch_rejected() -> None:
    plan = _plan(_probe_task(query="khác"))
    with pytest.raises(ContractValidationError):
        _budget(plan, discovery_checkpoint=_checkpoint())


def test_probe_origin_person_scalar_rejected() -> None:
    plan = _plan(_probe_task(person_identifier="Nguyen Van A"))
    with pytest.raises(ContractValidationError):
        _budget(plan, discovery_checkpoint=_checkpoint())


def test_probe_origin_wrong_capability_rejected() -> None:
    task = TaskSpec(
        task_id="T-kg",
        capability="knowledge_graph.query",
        task_objective="spoofed",
        input=DocumentSearchInput(kind="document.search", query=_PROBE_QUERY),
        depends_on=(),
        origin=DiscoveryProbeTaskOrigin(
            kind="discovery_probe", probe_id="p1", slot_id="slot-1", round=1
        ),
    )
    plan = _plan(task)
    with pytest.raises(ContractValidationError):
        _budget(plan, discovery_checkpoint=_checkpoint())


def test_one_probe_cannot_be_owned_by_two_tasks() -> None:
    plan = _plan(
        _probe_task("T-search-1"),
        _probe_task("T-search-2"),
    )
    with pytest.raises(ContractValidationError):
        _budget(plan, discovery_checkpoint=_checkpoint())


def test_max_tasks_remaining_bounded_by_factual_budget() -> None:
    plan = _plan(_probe_task(), _read_task(), _read_task("T2", ("t1",)))
    # factual = 2, len(tasks) = 3: min(8 - 2, 13 - 3) == 6.
    assert _budget(plan, discovery_checkpoint=_checkpoint()).max_tasks_remaining == 6
    # factual = 2, max_tasks = 2: clamped at zero.
    assert (
        _budget(
            plan, discovery_checkpoint=_checkpoint(), max_tasks=2
        ).max_tasks_remaining
        == 0
    )


def test_max_tasks_remaining_bounded_by_total_cap() -> None:
    plan = _plan(_probe_task(), _read_task())
    # probes are exempt from the factual count but still consume the total:
    # min(8 - 1, 3 - 2) == 1.
    assert (
        _budget(
            plan, discovery_checkpoint=_checkpoint(), total_task_limit=3
        ).max_tasks_remaining
        == 1
    )
    # A plan already at the total cap has zero remaining capacity.
    assert (
        _budget(
            plan, discovery_checkpoint=_checkpoint(), total_task_limit=2
        ).max_tasks_remaining
        == 0
    )


def test_budget_passthrough_fields() -> None:
    plan = _plan(_read_task())
    budget = _budget(plan, max_replans=3, max_parallel_branches=4)
    assert budget.max_replans_remaining == 3
    assert budget.max_parallel_branches == 4
    negative = _budget(plan, max_replans=-2)
    assert negative.max_replans_remaining == 0


# ---------------------------------------------------------------------------
# Scheduler plan-entry seam (discovery spec §11.3)
# ---------------------------------------------------------------------------


class _StubCapability:
    """Minimal atomic capability: records calls, returns the canned result."""

    def __init__(self, name: str, domain: str, result: AgentResult) -> None:
        self.descriptor = CapabilityDescriptor(
            name=name,  # type: ignore[arg-type]
            domain=domain,  # type: ignore[arg-type]
            operation_type="lookup",
            supports_parallel=False,
        )
        self._result = result
        self.calls: list[object] = []

    async def execute(self, request, runtime) -> AgentResult:  # type: ignore[no-untyped-def]
        self.calls.append(request)
        return self._result


def _capability_runtime() -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id="req-1",
        run_id="run-1",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=True,
        allowed_capabilities=frozenset({"people.lookup"}),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


def _graph_runtime(registry: object) -> GraphRuntimeContext:
    return GraphRuntimeContext(
        capability_runtime=_capability_runtime(),
        services=RuntimeServices(capability_registry=registry),  # type: ignore[arg-type]
    )


def _people_result(task_id: str) -> AgentResult:
    return AgentResult(
        contract_version="2.0",
        task_id=task_id,
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )


@pytest.mark.asyncio
async def test_scheduler_rejects_plan_over_explicit_total_limit() -> None:
    plan = factories.people_plan()
    stub = _StubCapability("people.lookup", "people", _people_result("T1"))
    registry = build_capability_registry(
        [CapabilityRegistration(capability=stub)], _capability_runtime()
    )
    runtime = _graph_runtime(registry)
    with pytest.raises(SchedulerError, match="plan exceeds total task limit"):
        await execute_ready_tasks(
            plan=plan,
            registry=registry,
            runtime=runtime,
            total_task_limit=0,
        )
    assert stub.calls == []
    # The threaded TaskScheduler.execute enforces the same entry check.
    with pytest.raises(SchedulerError, match="plan exceeds total task limit"):
        await TaskScheduler(registry).execute(
            plan, runtime, total_task_limit=0
        )
    assert stub.calls == []


@pytest.mark.asyncio
async def test_scheduler_unchanged_when_limit_omitted() -> None:
    plan = factories.people_plan()
    stub = _StubCapability("people.lookup", "people", _people_result("T1"))
    registry = build_capability_registry(
        [CapabilityRegistration(capability=stub)], _capability_runtime()
    )
    report = await TaskScheduler(registry).execute(plan, _graph_runtime(registry))
    assert report.truncated is False
    assert [result.task_id for result in report.results] == ["T1"]
    assert len(stub.calls) == 1


# ---------------------------------------------------------------------------
# Replan runtime seam (discovery spec §11.3)
# ---------------------------------------------------------------------------


def _replan_runtime() -> GraphRuntimeContext:
    return GraphRuntimeContext(
        capability_runtime=_capability_runtime(),
        services=RuntimeServices(),
    )


def _policy() -> DiscoveryPolicy:
    return DiscoveryPolicy(
        allow_reference_discovery=False,
        allow_supporting_discovery=False,
        max_discovered_documents=0,
    )


def _replan_task(task_id: str = "T2") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability="people.lookup",
        task_objective="retry people lookup",
        input=PeopleLookupInput(kind="people.lookup", query="A"),
        depends_on=(),
        origin=ReplanTaskOrigin(
            kind="replan",
            reason="coverage gap",
            task_ids=("T1",),
            evidence_use_ids=(),
        ),
    )


def test_replan_rejects_proposed_plan_over_total_limit() -> None:
    current = factories.people_plan()
    proposed = current.model_copy(
        update={"tasks": current.tasks + (_replan_task(),)}
    )
    budget = ResearchBudgetView(
        max_tasks_remaining=8,
        max_replans_remaining=1,
        max_parallel_branches=2,
    )
    with pytest.raises(ReplanRejected):
        validate_runtime_replan(
            current,
            proposed,
            (),
            _policy(),
            budget,
            _replan_runtime(),
            total_task_limit=1,
        )
    # Inside the explicit limit, the identical proposal is still accepted.
    accepted = validate_runtime_replan(
        current,
        proposed,
        (),
        _policy(),
        budget,
        _replan_runtime(),
        total_task_limit=2,
    )
    assert [task.task_id for task in accepted.tasks] == ["T1", "T2"]


def test_append_replan_tasks_rejects_over_total_limit() -> None:
    current = factories.people_plan()
    budget = ResearchBudgetView(
        max_tasks_remaining=8,
        max_replans_remaining=1,
        max_parallel_branches=2,
    )
    with pytest.raises(ReplanRejected):
        append_replan_tasks(
            current,
            (_replan_task(),),
            (),
            _policy(),
            budget,
            _replan_runtime(),
            total_task_limit=1,
        )
    # Omitted limit preserves the prior behavior exactly.
    accepted = append_replan_tasks(
        current,
        (_replan_task(),),
        (),
        _policy(),
        budget,
        _replan_runtime(),
    )
    assert [task.task_id for task in accepted.tasks] == ["T1", "T2"]

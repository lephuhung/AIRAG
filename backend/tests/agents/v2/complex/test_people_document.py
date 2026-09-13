"""People -> Document deterministic dependency materialization (Phase 3, Task 4).

The governed People scalar is materialized server-side into a concrete
``DocumentSearchInput`` BEFORE the dependent task is appended/checkpointed:
never an agent handoff, never inside the scheduler. ``depends_on`` expresses
ordering only; the scheduler executes ``TaskSpec.input`` exactly as
checkpointed and never mutates or lazily materializes it. The planner sees
only T1 status, ``evidence_use_ids``, and ``dependency_scalar_available``.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    DocumentSearchInput,
    PeopleLookupInput,
    PeopleLookupOutput,
)
from app.services.agents.v2.contracts.evidence import EvidenceUseRef
from app.services.agents.v2.contracts.execution import (
    AgentError,
    AgentRequest,
    AgentResult,
)
from app.services.agents.v2.contracts.planning import (
    DiscoveryPolicy,
    InitialTaskOrigin,
    ResearchBudgetView,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.state import (
    GraphRuntimeContext,
    RuntimeServices,
)
from app.services.agents.v2.dependencies.people_document import (
    PERSON_IDENTIFIER_FIELD,
    PeopleDocumentMaterialization,
    append_materialized_dependent,
    build_dependent_search_task,
    extract_person_identifier,
    materialize_person_dependency,
)
from app.services.agents.v2.tools.observations import ObservationProjector

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
USE_ID = UUID("44444444-4444-4444-4444-444444444444")
SCALAR = "079000000001"
OTHER_SCALAR = "079000000002"

RAW_PEOPLE_ROW = {
    "record_id": "rec-1",
    "name": "Nguyen Van A",
    PERSON_IDENTIFIER_FIELD: SCALAR,
    "cccd": SCALAR,
    "dob": "1990-01-01",
    "address": "Hanoi",
    "phone": "0900000000",
    "email": "a@example.com",
}

FORBIDDEN_OBSERVATION_TOKENS = (
    "cccd",
    "national_id",
    "citizen_id",
    "dob",
    "birth",
    "address",
    "phone",
    "email",
    "personnel",
)


def _minimized_content(identifier: str | None = None) -> str:
    payload: dict[str, object] = {"name": "Nguyen Van A"}
    if identifier is not None:
        payload[PERSON_IDENTIFIER_FIELD] = identifier
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class FakeHydratedEvidence:
    def __init__(self, content: str, use_id: UUID = USE_ID) -> None:
        self.use_id = use_id
        self.content = content


class FakeHydrator:
    """Governed-hydration stand-in: returns exactly the admitted set."""

    def __init__(self, admitted: tuple[FakeHydratedEvidence, ...] = ()) -> None:
        self._admitted = admitted
        self.calls: list[tuple[tuple[EvidenceUseRef, ...], object, object]] = []

    async def hydrate_for_evaluation(
        self,
        use_refs: tuple[EvidenceUseRef, ...],
        *,
        runtime: object,
        plan: object,
        bindings: object,
    ) -> tuple[FakeHydratedEvidence, ...]:
        self.calls.append((use_refs, plan, bindings))
        return self._admitted


class FakeLeaseSession:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def commit(self) -> None:
        self._events.append("commit")


class FakeLeaseRepo:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[object, object, object]] = []
        self.session = FakeLeaseSession(events)

    async def acquire_or_refresh(
        self, run_id: object, revision_id: object, use_id: object
    ) -> None:
        self.events.append("acquire")
        self.calls.append((run_id, revision_id, use_id))


class StubSearchCapability:
    """Atomic document.search stub: records the exact checkpointed input."""

    def __init__(self, result: AgentResult) -> None:
        self.descriptor = CapabilityDescriptor(
            name="document.search",  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="search",
            supports_parallel=True,
        )
        self._result = result
        self.calls: list[tuple[AgentRequest, CapabilityRuntimeContext]] = []

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        self.calls.append((request, runtime))
        return self._result


def _runtime(
    *,
    hydrator: FakeHydrator | None = None,
    leases: FakeLeaseRepo | None = None,
    allowed: frozenset[str] | None = None,
    can_read_people: bool = True,
    registry: object = None,
) -> GraphRuntimeContext:
    return GraphRuntimeContext(
        capability_runtime=CapabilityRuntimeContext(
            request_id="req-1",
            run_id="run-1",
            user_id=USER_ID,
            workspace_ids=(WORKSPACE_ID,),
            can_read_people=can_read_people,
            allowed_capabilities=(
                allowed
                if allowed is not None
                else frozenset({"people.lookup", "document.search"})
            ),
            deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        services=RuntimeServices(
            capability_registry=registry,
            evidence_hydrator=hydrator,
            retention_leases=leases,
        ),
    )


def _people_plan() -> TaskPlan:
    return TaskPlan(
        contract_version=CONTRACT_VERSION,
        plan_id="p1",
        goal="A xuat hien trong nghi dinh nao",
        target_units=(),
        tasks=(
            TaskSpec(
                task_id="T1",
                capability="people.lookup",
                task_objective="Find person A",
                input=PeopleLookupInput(kind="people.lookup", query="Nguyen Van A"),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )


def _people_success(use_id: UUID = USE_ID) -> AgentResult:
    return AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T1",
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(EvidenceUseRef(use_id=use_id),),
        coverage_observations=(),
        error=None,
    )


def _bindings() -> DocumentBindingSet:
    return DocumentBindingSet(bindings=(), revision_requirement_refs=())


def _policy() -> DiscoveryPolicy:
    return DiscoveryPolicy(
        allow_reference_discovery=False,
        allow_supporting_discovery=True,
        max_discovered_documents=1,
    )


def _budget() -> ResearchBudgetView:
    return ResearchBudgetView(
        max_tasks_remaining=7, max_replans_remaining=1, max_parallel_branches=2
    )


# ---------------------------------------------------------------------------
# Scalar extraction: exact approved field only, fail closed
# ---------------------------------------------------------------------------


def test_extract_person_identifier_reads_only_the_approved_field() -> None:
    assert extract_person_identifier(_minimized_content(SCALAR)) == SCALAR
    # Raw-only aliases never satisfy the extractor: no fallback, no guessing.
    raw_only = json.dumps(
        {"name": "Nguyen Van A", "cccd": SCALAR},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert extract_person_identifier(raw_only) is None
    assert extract_person_identifier(_minimized_content(None)) is None
    assert extract_person_identifier("") is None
    assert extract_person_identifier("not-json") is None
    assert extract_person_identifier(json.dumps({PERSON_IDENTIFIER_FIELD: "  "})) is None


# ---------------------------------------------------------------------------
# Required Task 4 tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_people_document_materializes_before_task_append() -> None:
    plan = _people_plan()
    result = _people_success()
    runtime = _runtime(hydrator=FakeHydrator((FakeHydratedEvidence(_minimized_content(SCALAR)),)))
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=result,
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert isinstance(outcome, PeopleDocumentMaterialization)
    assert outcome.kind == "materialized"
    assert outcome.scalar == SCALAR
    assert outcome.input is not None
    assert outcome.input.person_identifier == SCALAR
    # T2 does not exist yet: materialization precedes the append.
    assert [task.task_id for task in plan.tasks] == ["T1"]
    task = build_dependent_search_task(
        outcome, people_task_id="T1", query="nghi dinh", next_task_id="T2"
    )
    assert task.task_id == "T2"
    assert task.capability == "document.search"
    assert task.depends_on == ("T1",)
    assert isinstance(task.input, DocumentSearchInput)
    assert task.input.person_identifier == SCALAR


@pytest.mark.asyncio
async def test_t2_checkpoint_contains_final_document_search_input() -> None:
    plan = _people_plan()
    result = _people_success()
    runtime = _runtime(hydrator=FakeHydrator((FakeHydratedEvidence(_minimized_content(SCALAR)),)))
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=result,
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert outcome.kind == "materialized"
    from app.services.agents.v2.contracts.execution import TaskExecutionSummary

    proposed = append_materialized_dependent(
        current=plan,
        outcomes=(TaskExecutionSummary(task_id="T1", status="success"),),
        outcome=outcome,
        query="nghi dinh",
        next_task_id="T2",
        policy=_policy(),
        budget=_budget(),
    )
    assert [task.task_id for task in proposed.tasks] == ["T1", "T2"]
    checkpointed = proposed.tasks[1].input
    assert isinstance(checkpointed, DocumentSearchInput)
    assert checkpointed.person_identifier == SCALAR
    assert checkpointed.query == "nghi dinh"
    # The checkpoint payload already carries the final concrete input.
    payload = json.loads(proposed.model_dump_json())
    t2 = next(task for task in payload["tasks"] if task["task_id"] == "T2")
    assert t2["input"]["person_identifier"] == SCALAR
    assert t2["input"]["query"] == "nghi dinh"
    assert t2["depends_on"] == ["T1"]


@pytest.mark.asyncio
async def test_scheduler_never_rewrites_task_input() -> None:
    from app.services.agents.v2.contracts.capability import DocumentSearchOutput
    from app.services.agents.v2.execution.scheduler import execute_ready_tasks

    plan = TaskPlan(
        contract_version=CONTRACT_VERSION,
        plan_id="p1",
        goal="A xuat hien trong nghi dinh nao",
        target_units=(),
        tasks=(
            TaskSpec(
                task_id="T1",
                capability="people.lookup",
                task_objective="Find person A",
                input=PeopleLookupInput(kind="people.lookup", query="Nguyen Van A"),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
            TaskSpec(
                task_id="T2",
                capability="document.search",
                task_objective="Search decrees",
                input=DocumentSearchInput(
                    kind="document.search", query="nghi dinh", person_identifier=SCALAR
                ),
                depends_on=("T1",),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )
    search_result = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T2",
        status="success",
        data=DocumentSearchOutput(kind="document.search", candidates=()),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    search = StubSearchCapability(search_result)
    runtime = _runtime(
        registry=build_capability_registry(
            [CapabilityRegistration(capability=search)], runtime=_runtime().capability_runtime
        )
    )
    report = await execute_ready_tasks(
        plan=plan,
        results=(_people_success(),),
        registry=runtime.services.capability_registry,
        runtime=runtime,
        bindings=_bindings(),
    )
    assert len(search.calls) == 1
    request, _ = search.calls[0]
    # The scheduler dispatched the checkpointed input verbatim: no rewrite,
    # no lazy materialization, no second scalar source.
    assert request.input == plan.tasks[1].input
    assert isinstance(request.input, DocumentSearchInput)
    assert request.input.person_identifier == SCALAR
    assert [result.task_id for result in report.results] == ["T1", "T2"]


@pytest.mark.asyncio
async def test_people_scalar_never_enters_planner_observation() -> None:
    observation = ObservationProjector.project(_people_success())
    assert observation.projection.kind == "people.lookup"
    assert observation.projection.matched is True
    assert observation.projection.dependency_scalar_available is True
    dumped = observation.model_dump_json().lower()
    assert SCALAR not in dumped
    for token in FORBIDDEN_OBSERVATION_TOKENS:
        assert token not in dumped
    # The raw People row never enters the planner observation either.
    for value in RAW_PEOPLE_ROW.values():
        if isinstance(value, str) and len(value) >= 4 and value != SCALAR:
            assert value.lower() not in dumped


@pytest.mark.asyncio
async def test_people_not_found_does_not_append_t2() -> None:
    plan = _people_plan()
    result = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T1",
        status="not_found",
        data=PeopleLookupOutput(kind="people.lookup", matched=False),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    runtime = _runtime(hydrator=FakeHydrator(()))
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=result,
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert outcome.kind == "not_found"
    assert outcome.input is None
    assert outcome.scalar is None
    with pytest.raises(Exception):
        append_materialized_dependent(
            current=plan,
            outcomes=(),
            outcome=outcome,
            query="nghi dinh",
            next_task_id="T2",
            policy=_policy(),
            budget=_budget(),
        )
    assert [task.task_id for task in plan.tasks] == ["T1"]


@pytest.mark.asyncio
async def test_people_timeout_does_not_fabricate_t2() -> None:
    plan = _people_plan()
    result = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T1",
        status="error",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=AgentError(code="TIMEOUT", message="people.lookup timed out", retryable=True),
    )
    runtime = _runtime(hydrator=FakeHydrator(()))
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=result,
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert outcome.kind == "failed"
    assert outcome.error_code == "TIMEOUT"
    assert outcome.input is None
    assert [task.task_id for task in plan.tasks] == ["T1"]


# ---------------------------------------------------------------------------
# Distinct typed outcomes: denial / error / expired / missing scalar
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_permission_denied_and_error_stay_distinct_from_not_found() -> None:
    plan = _people_plan()
    runtime = _runtime(hydrator=FakeHydrator(()))
    denied = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T1",
        status="denied",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=AgentError(
            code="PERMISSION_DENIED", message="people.lookup denied", retryable=False
        ),
    )
    denied_outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=denied,
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert denied_outcome.kind == "denied"
    assert denied_outcome.error_code == "PERMISSION_DENIED"
    assert denied_outcome.input is None

    failed = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T1",
        status="error",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=AgentError(
            code="INTERNAL_ERROR", message="people.lookup broke", retryable=False
        ),
    )
    failed_outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=failed,
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert failed_outcome.kind == "failed"
    assert failed_outcome.error_code == "INTERNAL_ERROR"
    assert {denied_outcome.kind, failed_outcome.kind} != {"not_found"}
    assert [task.task_id for task in plan.tasks] == ["T1"]


@pytest.mark.asyncio
async def test_expired_or_unauthorized_evidence_use_does_not_materialize() -> None:
    plan = _people_plan()
    # Success + matched, but the governed hydrator admits nothing: the use is
    # expired, tombstoned, or no longer authorized under current runtime ACL.
    runtime = _runtime(hydrator=FakeHydrator(()))
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=_people_success(),
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert outcome.kind == "unavailable"
    assert outcome.input is None
    assert outcome.scalar is None
    assert [task.task_id for task in plan.tasks] == ["T1"]


@pytest.mark.asyncio
async def test_matched_without_extractable_scalar_never_fabricates_input() -> None:
    plan = _people_plan()
    runtime = _runtime(
        hydrator=FakeHydrator((FakeHydratedEvidence(_minimized_content(None)),))
    )
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=_people_success(),
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert outcome.kind == "unavailable"
    assert outcome.input is None
    # Never a guessed or blank person_identifier downstream.
    assert outcome.scalar is None


@pytest.mark.asyncio
async def test_conflicting_scalars_across_uses_fail_closed() -> None:
    plan = _people_plan()
    use2 = uuid4()
    result = AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id="T1",
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(EvidenceUseRef(use_id=USE_ID), EvidenceUseRef(use_id=use2)),
        coverage_observations=(),
        error=None,
    )
    runtime = _runtime(
        hydrator=FakeHydrator(
            (
                FakeHydratedEvidence(_minimized_content(SCALAR)),
                FakeHydratedEvidence(_minimized_content(OTHER_SCALAR), use_id=use2),
            )
        )
    )
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=result,
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert outcome.kind == "unavailable"
    assert outcome.input is None


@pytest.mark.asyncio
async def test_raw_people_row_never_enters_checkpoint_plan() -> None:
    plan = _people_plan()
    runtime = _runtime(hydrator=FakeHydrator((FakeHydratedEvidence(_minimized_content(SCALAR)),)))
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=_people_success(),
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    from app.services.agents.v2.contracts.execution import TaskExecutionSummary

    proposed = append_materialized_dependent(
        current=plan,
        outcomes=(TaskExecutionSummary(task_id="T1", status="success"),),
        outcome=outcome,
        query="nghi dinh",
        next_task_id="T2",
        policy=_policy(),
        budget=_budget(),
    )
    dumped = proposed.model_dump_json().lower()
    for token in FORBIDDEN_OBSERVATION_TOKENS:
        if token == PERSON_IDENTIFIER_FIELD:
            continue
        assert token not in dumped
    assert "1990-01-01" not in dumped
    assert "0900000000" not in dumped

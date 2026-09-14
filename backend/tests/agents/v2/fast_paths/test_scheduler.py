"""Phase 2 Task 3 — shared scheduler + execute node tests (TDD, failing first).

`TaskScheduler` (in `execution/scheduler.py`, re-exported by
`execution/__init__.py`) is the one and only capability-dispatch path for v2:
ready TaskSpec -> checkpoint-ownership check -> shared-registry resolve ->
typed `AgentRequest` -> `capability.execute(request, capability_runtime)` ->
frozen validation -> retention lease per new use -> immutable append. Task
input travels exactly as checkpointed; cancellation/deadline stop dispatch and
never fabricate success.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    DocumentReadInput,
    PeopleLookupInput,
)
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.evidence import EvidenceUseRef
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
from app.services.agents.v2.contracts.evaluation import CoverageObservation
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.planning import (
    InitialTaskOrigin,
    TargetUnit,
    TaskPlan,
    TaskSpec,
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

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
REVISION = "11111111-1111-1111-1111-111111111111"


class StubCapability:
    """Minimal atomic capability: records its call, returns the canned result."""

    def __init__(self, name: str, domain: str, result: AgentResult | None = None) -> None:
        self.descriptor = CapabilityDescriptor(
            name=name,  # type: ignore[arg-type]
            domain=domain,  # type: ignore[arg-type]
            operation_type="lookup",
            supports_parallel=False,
        )
        self._result = result
        self.calls: list[tuple[AgentRequest, CapabilityRuntimeContext]] = []
        self.entered = asyncio.Event()

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        assert isinstance(request, AgentRequest)
        assert isinstance(runtime, CapabilityRuntimeContext)
        self.calls.append((request, runtime))
        self.entered.set()
        assert self._result is not None
        return self._result


class BlockingCapability(StubCapability):
    def __init__(self, name: str, domain: str) -> None:
        super().__init__(name, domain, result=None)
        self.release = asyncio.Event()

    async def execute(self, request: AgentRequest, runtime: CapabilityRuntimeContext) -> AgentResult:
        self.calls.append((request, runtime))
        self.entered.set()
        await self.release.wait()
        raise AssertionError("must be cancelled before release")


class SleepingCapability(StubCapability):
    async def execute(self, request: AgentRequest, runtime: CapabilityRuntimeContext) -> AgentResult:
        self.calls.append((request, runtime))
        self.entered.set()
        await asyncio.sleep(30)
        raise AssertionError("wait_for must time out first")


class FakeLeaseSession:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def commit(self) -> None:
        self._events.append("commit")


class FakeLeaseRepo:
    """Stand-in for RevisionRetentionLeaseRepository (T1 fake shape)."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[Any, Any, Any]] = []
        self.session = FakeLeaseSession(events)

    async def acquire_or_refresh(
        self,
        run_id: str,
        revision_id: Any = None,
        evidence_use_id: Any = None,
        *,
        now: Any = None,
    ) -> Any:
        self.calls.append((run_id, revision_id, evidence_use_id))
        self.events.append(f"acquire:{evidence_use_id}")
        return None


def capability_runtime(
    *,
    allowed: frozenset[str] = frozenset({"people.lookup", "document.read"}),
    deadline_at: datetime | None = None,
) -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id="req-1",
        run_id="run-1",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=True,
        allowed_capabilities=allowed,
        deadline_at=deadline_at or datetime(2030, 1, 1, tzinfo=UTC),
    )


def services(
    *,
    registry: Any = None,
    leases: Any = None,
) -> RuntimeServices:
    return RuntimeServices(capability_registry=registry, retention_leases=leases)


def graph_runtime(
    *,
    registry: Any = None,
    leases: Any = None,
    allowed: frozenset[str] = frozenset({"people.lookup", "document.read"}),
    deadline_at: datetime | None = None,
) -> GraphRuntimeContext:
    return GraphRuntimeContext(
        capability_runtime=capability_runtime(
            allowed=allowed, deadline_at=deadline_at
        ),
        services=services(registry=registry, leases=leases),
    )


def registry_for(*stubs: StubCapability, allowed: frozenset[str] | None = None):
    names = frozenset(stub.descriptor.name for stub in stubs)
    runtime = capability_runtime(allowed=allowed if allowed is not None else names)
    return build_capability_registry(
        [CapabilityRegistration(capability=stub) for stub in stubs], runtime
    )


def people_plan(task_id: str = "fast_simple_people_lookup_0") -> TaskPlan:
    return TaskPlan(
        contract_version="2.0",
        plan_id="plan-people",
        goal="Ai là Nguyễn Văn A?",
        target_units=(),
        tasks=(
            TaskSpec(
                task_id=task_id,
                capability="people.lookup",
                task_objective="Ai là Nguyễn Văn A?",
                input=PeopleLookupInput(kind="people.lookup", query="Nguyễn Văn A"),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )


def read_plan(task_id: str = "fast_exact_document_metadata_0") -> TaskPlan:
    return TaskPlan(
        contract_version="2.0",
        plan_id="plan-doc",
        goal="Điều 5 của A nói gì?",
        target_units=(
            TargetUnit(
                target_id="t_b_r1",
                binding_id="b_r1",
                requested_locator=DocumentLocator(kind="document"),
                completion_criteria=(),
            ),
        ),
        tasks=(
            TaskSpec(
                task_id=task_id,
                capability="document.read",
                task_objective="Điều 5 của A nói gì?",
                input=DocumentReadInput(kind="document.read", target_ids=("t_b_r1",)),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )


def binding_set() -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b_r1",
                document_id=DOCUMENT_ID,
                document_revision=REVISION,
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )


def people_result(task_id: str, use_id: UUID | None = None) -> AgentResult:
    from app.services.agents.v2.contracts.capability import PeopleLookupOutput

    return AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id=task_id,
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(EvidenceUseRef(use_id=use_id or uuid4()),),
        coverage_observations=(),
        error=None,
    )


def read_result(task_id: str, use_id: UUID | None = None) -> AgentResult:
    from app.services.agents.v2.contracts.capability import DocumentReadOutput

    return AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id=task_id,
        status="success",
        data=DocumentReadOutput(kind="document.read", read_unit_count=1),
        evidence_uses=(EvidenceUseRef(use_id=use_id or uuid4()),),
        coverage_observations=(
            CoverageObservation(
                target_id="t_b_r1",
                observed_locators=(DocumentLocator(kind="document"),),
                outcome="read",
            ),
        ),
        error=None,
    )


def make_state(
    *,
    plan: TaskPlan | None = None,
    task_results: tuple[AgentResult, ...] = (),
    semantic: SemanticContext | None = None,
) -> SupervisorV2State:
    return SupervisorV2State(
        contract_version="2.0",
        request=RequestContext(
            contract_version="2.0",
            request_id="req-1",
            thread_id="thread-1",
            original_query="Ai là Nguyễn Văn A?",
            known_documents=(),
        ),
        conversation=ConversationContext(
            summary="", active_entities=(), last_focus=None, recent_turns=()
        ),
        semantic=semantic
        or SemanticContext(
            contextualized_query="q",
            normalized_query="q",
            abbreviations=(),
            coreferences=(),
            document_refs=(),
            person_refs=(),
            section_refs=(),
            blocking_ambiguities=(),
        ),
        bindings=binding_set(),
        query_analysis=QueryAnalysis(work_type="lookup", domains=("people",)),
        route_decision=RouteDecision(
            route="fast_domain", reason_code="simple_people_lookup"
        ),
        execution=ExecutionState(
            plan=plan, task_results=task_results, evidence_evaluation=None
        ),
        clarification=None,
        final_response=None,
    )


# ---------------------------------------------------------------------------
# Ownership gate: no dispatch without a checkpointed plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_must_be_checkpointed_before_capability_dispatch() -> None:
    from app.services.agents.v2.nodes.execute import (
        MissingCheckpointedPlan,
        execute_node,
    )

    stub = StubCapability(
        "people.lookup", "people", people_result("fast_simple_people_lookup_0")
    )
    registry = registry_for(stub)
    runtime = graph_runtime(registry=registry, leases=FakeLeaseRepo([]))
    with pytest.raises(MissingCheckpointedPlan):
        await execute_node(make_state(plan=None), runtime)
    assert stub.calls == []


@pytest.mark.asyncio
async def test_scheduler_rejects_results_for_unknown_tasks() -> None:
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    stub = StubCapability(
        "people.lookup", "people", people_result("fast_simple_people_lookup_0")
    )
    scheduler = TaskScheduler(registry_for(stub))
    forged = people_result("forged-task")
    with pytest.raises(ContractValidationError):
        await scheduler.execute(
            people_plan(), graph_runtime(), prior_results=(forged,)
        )
    assert stub.calls == []


# ---------------------------------------------------------------------------
# Shared registry dispatch: people + document fast paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_people_uses_shared_capability_registry() -> None:
    from app.services.agents.v2.execution.scheduler import TaskScheduler
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    plan = build_fast_plan(
        SemanticContext(
            contextualized_query="Ai là Nguyễn Văn A?",
            normalized_query="Ai là Nguyễn Văn A?",
            abbreviations=(),
            coreferences=(),
            document_refs=(),
            person_refs=({"ref_id": "p1", "kind": "person", "label": "A"},),
            section_refs=(),
            blocking_ambiguities=(),
        ),
        DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        QueryAnalysis(work_type="lookup", domains=("people",)),
        RouteDecision(route="fast_domain", reason_code="simple_people_lookup"),
    )
    task_id = plan.tasks[0].task_id
    stub = StubCapability("people.lookup", "people", people_result(task_id))
    registry = registry_for(stub)
    scheduler = TaskScheduler(registry)
    events: list[str] = []
    report = await scheduler.execute(
        plan, graph_runtime(registry=registry, leases=FakeLeaseRepo(events))
    )
    assert report.truncated is False
    results = report.results
    assert len(results) == 1
    assert results[0].task_id == task_id
    assert results[0].status == "success"
    # Resolved through the shared registry instance, not a private copy.
    assert registry.get("people.lookup") is stub
    assert len(stub.calls) == 1
    request, runtime = stub.calls[0]
    assert request.task_id == task_id
    assert isinstance(runtime, CapabilityRuntimeContext)


@pytest.mark.asyncio
async def test_fast_document_read_uses_shared_capability_registry() -> None:
    from app.services.agents.v2.execution.scheduler import TaskScheduler
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan
    from app.services.agents.v2.nodes.routing import decide_route
    from app.services.agents.v2.contracts.semantic import DocumentReference

    semantic = SemanticContext(
        contextualized_query="Điều 5 của A nói gì?",
        normalized_query="Điều 5 của A nói gì?",
        abbreviations=(),
        coreferences=(),
        document_refs=(
            DocumentReference(
                ref_id="r1",
                original_span="A",
                normalized_reference="A",
                requested_role="target",
                revision_requirement=None,
                resolution_status="resolved",
                resolved_document_id=DOCUMENT_ID,
                candidate_document_ids=(),
            ),
        ),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    analysis = QueryAnalysis(work_type="retrieve", domains=("document",))
    route = decide_route(
        analysis, semantic, binding_set(), allowed_capabilities=frozenset({"document.read"})
    )
    assert route.route == "fast_domain"
    plan = build_fast_plan(semantic, binding_set(), analysis, route)
    task_id = plan.tasks[0].task_id
    stub = StubCapability("document.read", "document", read_result(task_id))
    registry = registry_for(stub)
    events: list[str] = []
    report = await TaskScheduler(registry).execute(
        plan,
        graph_runtime(registry=registry, leases=FakeLeaseRepo(events)),
        bindings=binding_set(),
    )
    assert report.truncated is False
    results = report.results
    assert len(results) == 1
    assert results[0].coverage_observations[0].target_id == "t_b_r1"
    assert registry.get("document.read") is stub


def test_task_scheduler_is_defined_once_and_shared() -> None:
    from app.services.agents.v2.execution import TaskScheduler as Reexported
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    assert Reexported is TaskScheduler
    root = Path(__file__).resolve().parents[4] / "app" / "services" / "agents" / "v2"
    definers = [
        path
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
        and "class TaskScheduler" in path.read_text()
    ]
    assert [path.name for path in definers] == ["scheduler.py"]
    resolvers = [
        path
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
        and "registry.get(" in path.read_text()
    ]
    assert [path.name for path in resolvers] == ["scheduler.py"]
    execute_node_src = (root / "nodes" / "execute.py").read_text()
    assert "TaskScheduler" in execute_node_src


def test_runtime_services_excludes_plan_checkpoint_and_evaluator_service() -> None:
    from app.services.agents.v2.contracts.base import ContractModel, RuntimeModel
    from app.services.agents.v2.contracts.state import RuntimeServices as Canonical
    from app.services.agents.v2.execution import RuntimeServices as Reexported

    assert Reexported is Canonical
    assert issubclass(RuntimeServices, RuntimeModel)
    assert not issubclass(RuntimeServices, ContractModel)
    assert set(RuntimeServices.model_fields) == {
        "retention_leases",
        "semantic_adapter",
        "binding_resolver",
        "capability_registry",
        "chat_messages",
        "authorization",
        "evidence_hydrator",
        "answer_draft_channel",
        "pinned_target_resolver",
    }
    assert "plan_checkpoint" not in RuntimeServices.model_fields
    assert not any("evaluator" in name for name in RuntimeServices.model_fields)
    assert RuntimeServices().capability_registry is None
    assert RuntimeServices().pinned_target_resolver is None


# ---------------------------------------------------------------------------
# Input fidelity: the checkpointed input is executed verbatim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_never_rewrites_task_input() -> None:
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    plan = read_plan()
    task = plan.tasks[0]
    stub = StubCapability("document.read", "document", read_result(task.task_id))
    registry = registry_for(stub)
    events: list[str] = []
    await TaskScheduler(registry).execute(
        plan,
        graph_runtime(registry=registry, leases=FakeLeaseRepo(events)),
        bindings=binding_set(),
    )
    assert len(stub.calls) == 1
    sent = stub.calls[0][0].input
    assert sent == task.input
    assert sent is task.input


@pytest.mark.asyncio
async def test_execute_leases_targetless_people_use_with_evidence_only_lease() -> None:
    """C1: a fresh targetless use is leased with revision_id None (never skipped)."""
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    plan = people_plan()
    use_id = uuid4()
    stub = StubCapability(
        "people.lookup", "people", people_result(plan.tasks[0].task_id, use_id)
    )
    registry = registry_for(stub)
    events: list[str] = []
    repo = FakeLeaseRepo(events)
    report = await TaskScheduler(registry).execute(
        plan,
        graph_runtime(registry=registry, leases=repo),
    )
    assert report.truncated is False
    assert len(report.results) == 1
    assert report.results[0].evidence_uses[0].use_id == use_id
    assert repo.calls == [("run-1", None, use_id)]
    assert f"acquire:{use_id}" in events
    assert "commit" in events
    assert events.index(f"acquire:{use_id}") < events.index("commit")


# ---------------------------------------------------------------------------
# Leases: every new use leased before results are returned; never re-leased
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_leases_new_evidence_uses_before_checkpoint() -> None:
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    plan = read_plan()
    use_id = uuid4()
    stub = StubCapability(
        "document.read", "document", read_result(plan.tasks[0].task_id, use_id)
    )
    registry = registry_for(stub)
    events: list[str] = []
    report = await TaskScheduler(registry).execute(
        plan,
        graph_runtime(registry=registry, leases=FakeLeaseRepo(events)),
        bindings=binding_set(),
    )
    assert report.truncated is False
    results = report.results
    assert len(results) == 1
    assert results[0].evidence_uses[0].use_id == use_id
    assert f"acquire:{use_id}" in events
    assert "commit" in events
    assert events.index(f"acquire:{use_id}") < events.index("commit")

    # A resumed execute with the same checkpointed results leases nothing new.
    rerun = await TaskScheduler(registry).execute(
        plan,
        graph_runtime(registry=registry, leases=FakeLeaseRepo(events)),
        prior_results=results,
        bindings=binding_set(),
    )
    assert rerun.results == results
    assert rerun.truncated is False
    assert stub.calls and len(stub.calls) == 1
    assert events.count(f"acquire:{use_id}") == 1


# ---------------------------------------------------------------------------
# Result integrity: mismatched task ids never become checkpointed facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_rejects_mismatched_task_result() -> None:
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    plan = read_plan()
    stub = StubCapability(
        "document.read", "document", read_result("some-other-task")
    )
    registry = registry_for(stub)
    events: list[str] = []
    with pytest.raises(ContractValidationError):
        await TaskScheduler(registry).execute(
            plan,
            graph_runtime(registry=registry, leases=FakeLeaseRepo(events)),
            bindings=binding_set(),
        )
    assert events == []


@pytest.mark.asyncio
async def test_every_result_use_and_coverage_resolves_to_checkpointed_ids() -> None:
    from app.services.agents.v2.contracts.validation import validate_agent_result
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    plan = read_plan()
    stub = StubCapability(
        "document.read", "document", read_result(plan.tasks[0].task_id)
    )
    registry = registry_for(stub)
    events: list[str] = []
    report = await TaskScheduler(registry).execute(
        plan,
        graph_runtime(registry=registry, leases=FakeLeaseRepo(events)),
        bindings=binding_set(),
    )
    assert report.truncated is False
    results = report.results
    assert len(results) == 1
    validate_agent_result(results[0], plan)
    assert results[0].task_id in {task.task_id for task in plan.tasks}
    target_ids = {unit.target_id for unit in plan.target_units}
    for observation in results[0].coverage_observations:
        assert observation.target_id in target_ids


# ---------------------------------------------------------------------------
# Deadline / cancellation own the dispatch boundary (T2-M3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_stops_before_dispatch_on_deadline() -> None:
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    plan = read_plan()
    stub = StubCapability(
        "document.read", "document", read_result(plan.tasks[0].task_id)
    )
    registry = registry_for(stub)
    events: list[str] = []
    report = await TaskScheduler(registry).execute(
        plan,
        graph_runtime(
            registry=registry,
            leases=FakeLeaseRepo(events),
            deadline_at=datetime.now(UTC) - timedelta(seconds=1),
        ),
        bindings=binding_set(),
    )
    assert report.results == ()
    assert report.truncated is True
    assert stub.calls == []
    assert events == []


@pytest.mark.asyncio
async def test_scheduler_fails_closed_when_tasks_never_ready() -> None:
    """M5: unknown dependencies/cycles are a typed failure, never a silent partial."""
    from app.services.agents.v2.execution.scheduler import SchedulerError, TaskScheduler

    ghost = TaskSpec(
        task_id="G1",
        capability="people.lookup",
        task_objective="ghost dependency",
        input=PeopleLookupInput(kind="people.lookup", query="A"),
        depends_on=("no-such-task",),
        origin=InitialTaskOrigin(kind="initial"),
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-ghost",
        goal="ghost",
        target_units=(),
        tasks=(ghost,),
    )
    stub = StubCapability("people.lookup", "people", people_result("G1"))
    with pytest.raises(SchedulerError):
        await TaskScheduler(registry_for(stub)).execute(
            plan, graph_runtime(registry=registry_for(stub))
        )
    assert stub.calls == []


@pytest.mark.asyncio
async def test_scheduler_cancellation_never_becomes_success() -> None:
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    plan = read_plan()
    stub = BlockingCapability("document.read", "document")
    registry = registry_for(stub)
    scheduler = TaskScheduler(registry)
    task = asyncio.create_task(
        scheduler.execute(
            plan, graph_runtime(registry=registry), bindings=binding_set()
        )
    )
    await stub.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stub.calls and len(stub.calls) == 1


@pytest.mark.asyncio
async def test_scheduler_timeout_never_becomes_success() -> None:
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    plan = read_plan()
    stub = SleepingCapability("document.read", "document")
    registry = registry_for(stub)
    events: list[str] = []
    with pytest.raises(TimeoutError):
        await TaskScheduler(registry).execute(
            plan,
            graph_runtime(
                registry=registry,
                leases=FakeLeaseRepo(events),
                deadline_at=datetime.now(UTC) + timedelta(milliseconds=50),
            ),
            bindings=binding_set(),
        )
    assert stub.calls and len(stub.calls) == 1
    assert events == []


# ---------------------------------------------------------------------------
# Registry denial is a typed outcome, never a silent success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_denial_returns_typed_denied_result() -> None:
    from app.services.agents.v2.contracts.validation import validate_agent_result
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    plan = read_plan()
    excluded_read = StubCapability("document.read", "document")
    other = StubCapability(
        "people.lookup", "people", people_result(plan.tasks[0].task_id)
    )
    registry = registry_for(
        excluded_read, other, allowed=frozenset({"people.lookup"})
    )
    report = await TaskScheduler(registry).execute(
        plan, graph_runtime(registry=registry), bindings=binding_set()
    )
    assert report.truncated is False
    results = report.results
    assert len(results) == 1
    assert results[0].task_id == plan.tasks[0].task_id
    assert results[0].status == "denied"
    assert results[0].error is not None
    assert results[0].error.code == "PERMISSION_DENIED"
    assert excluded_read.calls == []
    validate_agent_result(results[0], plan)


# ---------------------------------------------------------------------------
# execute_node: checkpointed plan in, frozen execution partial out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_node_returns_frozen_execution_partial() -> None:
    from app.services.agents.v2.nodes.execute import execute_node

    plan = people_plan()
    stub = StubCapability(
        "people.lookup", "people", people_result(plan.tasks[0].task_id)
    )
    registry = registry_for(stub)
    events: list[str] = []
    update = await execute_node(
        make_state(plan=plan), graph_runtime(registry=registry, leases=FakeLeaseRepo(events))
    )
    assert set(update) == {"execution"}
    execution = update["execution"]
    assert isinstance(execution, ExecutionState)
    assert execution.plan == plan
    assert len(execution.task_results) == 1
    assert execution.task_results[0].task_id == plan.tasks[0].task_id
    assert "evidence_evaluation" in ExecutionState.model_fields
    assert execution.evidence_evaluation is None


@pytest.mark.asyncio
async def test_execute_node_preserves_prior_results_without_redispatch() -> None:
    from app.services.agents.v2.nodes.execute import execute_node

    plan = read_plan()
    stub = StubCapability(
        "document.read", "document", read_result(plan.tasks[0].task_id)
    )
    registry = registry_for(stub)
    events: list[str] = []
    prior = (
        AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=plan.tasks[0].task_id,
            status="not_found",
            data=None,
            evidence_uses=(),
            coverage_observations=(
                CoverageObservation(
                    target_id="t_b_r1",
                    observed_locators=(),
                    outcome="missing",
                ),
            ),
            error=None,
        ),
    )
    update = await execute_node(
        make_state(plan=plan, task_results=prior),
        graph_runtime(registry=registry, leases=FakeLeaseRepo(events)),
    )
    assert update["execution"].task_results == prior
    assert stub.calls == []


# ---------------------------------------------------------------------------
# P0 live-gate fix: the scheduler feeds pinned targets before dispatch.
#
# Fresh (non-resume) turns dispatch through the shared scheduler with a
# fresh, empty request-scoped resolver. Until the scheduler feeds that
# resolver from the authoritative checkpointed plan + bindings, every
# scoped target is "unknown" and the retrieve capability fails closed
# (denied/SCOPE_VIOLATION) before the retrieval provider is ever called.
# ---------------------------------------------------------------------------

RETRIEVE_REVISION = "22222222-2222-2222-2222-222222222222"


class _FeedFakeEvidence:
    """Minimal evidence builder: mints one ref per persist, records calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def persist_use(
        self, *, source, content, provenance, task_id, purpose, target_id
    ) -> EvidenceUseRef:
        ref = EvidenceUseRef(use_id=uuid4())
        self.calls.append(
            {
                "source": source,
                "content": content,
                "task_id": task_id,
                "purpose": purpose,
                "target_id": target_id,
                "use_id": ref.use_id,
            }
        )
        return ref


class _FeedFakeRetrieval:
    """Preset revision-owned chunks; records the exact dispatch filters."""

    def __init__(self, chunks=()) -> None:
        self._chunks = tuple(chunks)
        self.calls: list[dict] = []

    async def retrieve(
        self, query: str, *, top_k: int, allowed_targets, workspace_ids
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "allowed_targets": allowed_targets,
                "workspace_ids": workspace_ids,
            }
        )
        return self._chunks


def _retrieve_stack(task_id: str = "T1", *, target_ids=("t1",), chunks=None):
    """Fresh-turn retrieve wiring: a real, initially EMPTY PlanBindingResolver.

    The SAME resolver instance is injected into the real retrieve
    capability AND (when the services bag supports it) exposed as the
    runtime-only ``pinned_target_resolver`` service — exactly the ingress
    wiring under repair. The scheduler must feed it before dispatch.
    """
    from app.services.agent.runtime_selector import PlanBindingResolver
    from app.services.agents.v2.capabilities.document import (
        DocumentRetrieveCapability,
        RevisionRetrievedChunk,
    )
    from app.services.agents.v2.contracts.capability import DocumentRetrieveInput
    from app.services.agents.v2.contracts.locators import (
        ChunkRangeLocator,
        SectionLocator,
    )

    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-retrieve",
        goal="factual query",
        target_units=(
            TargetUnit(
                target_id="t1",
                binding_id="b_t1",
                requested_locator=SectionLocator(
                    kind="section", structure_node_id="node-5"
                ),
                completion_criteria=(),
            ),
        ),
        tasks=(
            TaskSpec(
                task_id=task_id,
                capability="document.retrieve",
                task_objective="factual query",
                input=DocumentRetrieveInput(
                    kind="document.retrieve",
                    query="lan bmnn",
                    target_ids=tuple(target_ids),
                ),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )
    bindings = DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b_t1",
                document_id=DOCUMENT_ID,
                document_revision=RETRIEVE_REVISION,
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )
    if chunks is None:
        chunks = (
            RevisionRetrievedChunk(
                document_id=DOCUMENT_ID,
                document_revision=RETRIEVE_REVISION,
                locator=ChunkRangeLocator(
                    kind="chunk_range", start="c1", end="c1"
                ),
                content="secret chunk",
                score=0.9,
                target_id="t1",
            ),
        )
    resolver = PlanBindingResolver()  # fresh turn: empty until fed
    assert resolver.resolve("t1") is None
    service = _FeedFakeRetrieval(chunks)
    evidence = _FeedFakeEvidence()
    capability = DocumentRetrieveCapability(
        service=service, evidence=evidence, resolver=resolver
    )
    runtime = capability_runtime(allowed=frozenset({"document.retrieve"}))
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], runtime
    )
    events: list[str] = []
    ctx = GraphRuntimeContext(
        capability_runtime=runtime,
        services=services(registry=registry, leases=FakeLeaseRepo(events)),
    )
    if hasattr(ctx.services, "pinned_target_resolver"):
        ctx.services.pinned_target_resolver = resolver
    state = make_state(plan=plan, task_results=())
    state = SupervisorV2State(
        contract_version=state["contract_version"],
        request=state["request"],
        conversation=state["conversation"],
        semantic=state["semantic"],
        bindings=bindings,
        query_analysis=state["query_analysis"],
        route_decision=state["route_decision"],
        execution=ExecutionState(
            plan=plan, task_results=(), evidence_evaluation=None
        ),
        clarification=None,
        final_response=None,
    )
    return {
        "plan": plan,
        "bindings": bindings,
        "resolver": resolver,
        "service": service,
        "evidence": evidence,
        "registry": registry,
        "runtime": ctx,
        "events": events,
        "state": state,
    }


@pytest.mark.asyncio
async def test_fresh_turn_pinned_targets_fed_before_dispatch() -> None:
    """RED: a fresh hard-scoped retrieve turn must reach the provider.

    Before the fix the shared scheduler never feeds the request-scoped
    resolver, so T1 is denied/SCOPE_VIOLATION and the provider is never
    called. After the fix the same fresh turn succeeds with one governed
    EvidenceUse and citation-scoped output (no raw chunks on the result).
    """
    from app.services.agents.v2.nodes.execute import execute_node

    stack = _retrieve_stack()
    update = await execute_node(stack["state"], stack["runtime"])
    results = update["execution"].task_results
    assert len(results) == 1
    result = results[0]
    assert result.status == "success"
    assert result.data is not None
    assert result.data.kind == "document.retrieve"
    assert result.data.retrieved_unit_count == 1
    assert len(result.evidence_uses) == 1
    assert len(stack["service"].calls) == 1
    assert stack["evidence"].calls[0]["purpose"] == "coverage"
    assert stack["evidence"].calls[0]["target_id"] == "t1"
    # Raw chunk content lives only in the Evidence Store, never in output.
    assert "secret chunk" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_scheduler_feeds_matching_targets_but_denies_mismatch() -> None:
    """A pinned target resolves; an unknown target still fails closed."""
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    stack = _retrieve_stack(target_ids=("t9",))
    report = await TaskScheduler(stack["registry"]).execute(
        stack["plan"],
        stack["runtime"],
        bindings=stack["bindings"],
    )
    assert len(report.results) == 1
    result = report.results[0]
    assert result.status == "denied"
    assert result.error is not None
    assert result.error.code == "SCOPE_VIOLATION"
    assert stack["service"].calls == []


@pytest.mark.asyncio
async def test_scheduler_feed_replaces_stale_mappings_across_replans() -> None:
    """A replan must not inherit the previous plan's pinned targets."""
    from app.services.agents.v2.contracts.locators import SectionLocator
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    stack = _retrieve_stack()
    scheduler = TaskScheduler(stack["registry"])
    first = await scheduler.execute(
        stack["plan"], stack["runtime"], bindings=stack["bindings"]
    )
    assert first.results[0].status == "success"
    assert stack["resolver"].resolve("t1") is not None

    other_doc = UUID("33333333-3333-3333-3333-333333333333")
    other_rev = "44444444-4444-4444-4444-444444444444"
    plan_b = TaskPlan(
        contract_version="2.0",
        plan_id="plan-retrieve-b",
        goal="factual query b",
        target_units=(
            TargetUnit(
                target_id="t2",
                binding_id="b_t2",
                requested_locator=SectionLocator(
                    kind="section", structure_node_id="node-9"
                ),
                completion_criteria=(),
            ),
        ),
        tasks=(),
    )
    bindings_b = DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b_t2",
                document_id=other_doc,
                document_revision=other_rev,
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )
    second = await scheduler.execute(
        plan_b, stack["runtime"], bindings=bindings_b
    )
    assert second.results == ()
    assert stack["resolver"].resolve("t1") is None
    assert stack["resolver"].resolve("t2") is not None


@pytest.mark.asyncio
async def test_scheduler_missing_resolver_stays_fail_closed() -> None:
    """No pinned-target service: scoped dispatch denies, provider uncalled."""
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    stack = _retrieve_stack()
    if hasattr(stack["runtime"].services, "pinned_target_resolver"):
        stack["runtime"].services.pinned_target_resolver = None
    # The capability keeps its own unfed resolver: unknown targets deny.
    stack["resolver"]._documents.clear()
    stack["resolver"]._plan = None
    report = await TaskScheduler(stack["registry"]).execute(
        stack["plan"],
        stack["runtime"],
        bindings=stack["bindings"],
    )
    assert len(report.results) == 1
    assert report.results[0].status == "denied"
    assert report.results[0].error is not None
    assert report.results[0].error.code == "SCOPE_VIOLATION"
    assert stack["service"].calls == []


@pytest.mark.asyncio
async def test_scheduler_unscoped_retrieve_needs_no_pinned_targets() -> None:
    """Empty target_ids run over workspace scope with no resolver feed."""
    from app.services.agent.runtime_selector import PlanBindingResolver
    from app.services.agents.v2.capabilities.document import (
        DocumentRetrieveCapability,
        RevisionRetrievedChunk,
    )
    from app.services.agents.v2.contracts.capability import DocumentRetrieveInput
    from app.services.agents.v2.contracts.locators import ChunkRangeLocator
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-unscoped",
        goal="broad query",
        target_units=(),
        tasks=(
            TaskSpec(
                task_id="U1",
                capability="document.retrieve",
                task_objective="broad query",
                input=DocumentRetrieveInput(
                    kind="document.retrieve", query="broad", target_ids=()
                ),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )
    chunk = RevisionRetrievedChunk(
        document_id=DOCUMENT_ID,
        document_revision=RETRIEVE_REVISION,
        locator=ChunkRangeLocator(
            kind="chunk_range", start="c1", end="c1"
        ),
        content="workspace chunk",
        score=0.5,
        target_id=None,
    )
    resolver = PlanBindingResolver()
    service = _FeedFakeRetrieval((chunk,))
    evidence = _FeedFakeEvidence()
    capability = DocumentRetrieveCapability(
        service=service, evidence=evidence, resolver=resolver
    )
    runtime = capability_runtime(allowed=frozenset({"document.retrieve"}))
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], runtime
    )
    events: list[str] = []
    ctx = GraphRuntimeContext(
        capability_runtime=runtime,
        services=services(registry=registry, leases=FakeLeaseRepo(events)),
    )
    report = await TaskScheduler(registry).execute(plan, ctx, bindings=None)
    assert len(report.results) == 1
    result = report.results[0]
    assert result.status == "success"
    assert result.data is not None
    assert result.data.retrieved_unit_count == 1
    assert len(service.calls) == 1
    assert service.calls[0]["allowed_targets"] == ()
    assert evidence.calls[0]["purpose"] == "supporting"
    assert evidence.calls[0]["target_id"] is None

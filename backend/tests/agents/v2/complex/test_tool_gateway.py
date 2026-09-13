"""Governed agent tool gateway + safe observation projections (Phase 3, Task 2).

The agent/propose boundary is the ONLY adaptive planning boundary and it NEVER
executes a capability: tool adapters translate calls into
``CapabilityInvocationProposal``s, ``AgentToolGateway.propose`` returns a
validated append-only plan (or a typed rejection), and ``ObservationProjector``
exposes only typed per-capability projections. Execution stays with the shared
``TaskScheduler``; checkpointing stays with the LangGraph graph/saver.
"""
from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import TypedDict
from uuid import UUID

import pytest

from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.contracts.base import ContractModel
from app.services.agents.v2.contracts.binding import (
    DocumentBindingSet,
    DocumentDiscoveryCandidate,
    ScopedDocument,
)
from app.services.agents.v2.contracts.capability import (
    AbbreviationResolveOutput,
    CapabilityDescriptor,
    DocumentReadInput,
    DocumentSearchInput,
    DocumentSearchOutput,
    KnowledgeGraphInput,
    MemoryLookupOutput,
    PeopleLookupInput,
    PeopleLookupOutput,
    WriteOutput,
)
from app.services.agents.v2.contracts.evidence import EvidenceUseRef
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
from app.services.agents.v2.contracts.locators import DocumentLocator
from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.state import GraphRuntimeContext, RuntimeServices
from app.services.agents.v2.contracts.validation import validate_task_plan
from app.services.agents.v2.tools.adapters import (
    AgentToolAdapter,
    AgentToolCall,
    build_agent_tool_catalog,
)
from app.services.agents.v2.tools.discovery_candidates import (
    CandidateNotFound,
    DiscoveryCandidateRegistry,
    InvalidCandidateRole,
    candidate_addition_request,
)
from app.services.agents.v2.tools.gateway import (
    AgentToolGateway,
    CapabilityInvocationProposal,
    UnplannedCapabilityDispatch,
    require_planned_dispatch,
)
from app.services.agents.v2.tools.observations import (
    AgentToolObservation,
    DocumentSearchObservation,
    NoObservation,
    ObservationProjectionUnavailable,
    ObservationProjector,
    PeopleLookupObservation,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")
USE_ID = UUID("44444444-4444-4444-4444-444444444444")
CANDIDATE_ID_1 = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
CANDIDATE_ID_2 = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
REVISION_1 = "rev-1"
REVISION_2 = "rev-2"


class FakeCapability:
    """Atomic stub: never executed by the gateway in these tests."""

    def __init__(self, name: str, *, domain: str = "document") -> None:
        self.descriptor = CapabilityDescriptor(
            name=name,  # type: ignore[arg-type]
            domain=domain,  # type: ignore[arg-type]
            operation_type="read",
            supports_parallel=False,
        )
        self.calls: list[tuple[AgentRequest, object]] = []

    async def execute(self, request: AgentRequest, runtime: object) -> AgentResult:
        self.calls.append((request, runtime))
        raise AssertionError("tool gateway must never execute a capability")


def _registrations() -> list[CapabilityRegistration]:
    return [
        CapabilityRegistration(
            capability=FakeCapability("people.lookup", domain="people"),
        ),
        CapabilityRegistration(capability=FakeCapability("document.search")),
        CapabilityRegistration(capability=FakeCapability("document.read")),
        CapabilityRegistration(
            capability=FakeCapability("knowledge_graph.query", domain="knowledge_graph")
        ),
    ]


def _runtime(
    allowed: frozenset[str] | None = None,
    *,
    can_read_people: bool = True,
    registry: object = None,
) -> GraphRuntimeContext:
    allowed = (
        allowed
        if allowed is not None
        else frozenset(
            {"people.lookup", "document.search", "document.read", "knowledge_graph.query"}
        )
    )
    from app.services.agents.v2.contracts.capability import CapabilityRuntimeContext

    return GraphRuntimeContext(
        capability_runtime=CapabilityRuntimeContext(
            request_id="req-1",
            run_id="run-1",
            user_id=USER_ID,
            workspace_ids=(WORKSPACE_ID,),
            can_read_people=can_read_people,
            allowed_capabilities=allowed,
            deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        services=RuntimeServices(capability_registry=registry),
    )


def _bindings() -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id="b1",
                document_id=DOCUMENT_ID,
                document_revision=REVISION_1,
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )


def _read_task(task_id: str = "T1") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability="document.read",
        task_objective="Doc Dieu 5 cua A",
        input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def _read_plan() -> TaskPlan:
    return TaskPlan(
        contract_version="2.0",
        plan_id="p1",
        goal="Doc Dieu 5 cua A",
        target_units=(
            TargetUnit(
                target_id="t1",
                binding_id="b1",
                requested_locator=DocumentLocator(kind="document"),
                completion_criteria=(CoverageCriterion(kind="coverage"),),
            ),
        ),
        tasks=(_read_task(),),
    )


def _gateway(**kwargs: object) -> AgentToolGateway:
    return AgentToolGateway(**kwargs)  # type: ignore[arg-type]


def _people_result(task_id: str = "T1") -> AgentResult:
    return AgentResult(
        contract_version="2.0",
        task_id=task_id,
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(EvidenceUseRef(use_id=USE_ID),),
        coverage_observations=(),
        error=None,
    )


def _candidate(
    candidate_id: UUID = CANDIDATE_ID_1,
    document_id: UUID = DOCUMENT_ID,
    revision: str = REVISION_1,
) -> DocumentDiscoveryCandidate:
    return DocumentDiscoveryCandidate(
        candidate_id=candidate_id,
        document_id=document_id,
        document_revision=revision,
    )


def _search_result(
    task_id: str = "T2",
    candidates: tuple[DocumentDiscoveryCandidate, ...] | None = None,
) -> AgentResult:
    if candidates is None:
        candidates = (
            _candidate(CANDIDATE_ID_1, DOCUMENT_ID, REVISION_1),
            _candidate(CANDIDATE_ID_2, OTHER_DOCUMENT_ID, REVISION_2),
        )
    return AgentResult(
        contract_version="2.0",
        task_id=task_id,
        status="success",
        data=DocumentSearchOutput(kind="document.search", candidates=candidates),
        evidence_uses=(EvidenceUseRef(use_id=USE_ID),),
        coverage_observations=(),
        error=None,
    )


# ---------------------------------------------------------------------------
# Proposal boundary: validated task before dispatch, never execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_tool_call_creates_validated_task_before_dispatch() -> None:
    registry = build_capability_registry(
        _registrations(), _runtime().capability_runtime
    )
    adapter = AgentToolAdapter(registry)
    gateway = _gateway()
    runtime = _runtime(registry=registry)
    current = _read_plan()

    call = AgentToolCall(
        capability="knowledge_graph.query",
        objective="A thuoc don vi nao?",
        input=KnowledgeGraphInput(kind="knowledge_graph.query", query="A"),
        depends_on=("T1",),
    )
    outcome = await gateway.propose(adapter.to_proposal(call), current, runtime)

    assert outcome.accepted is True
    assert outcome.rejection is None
    # The validated append-only plan owns the new task BEFORE any dispatch.
    validate_task_plan(outcome.plan, _bindings())
    assert len(outcome.plan.tasks) == 2
    new_task = outcome.plan.tasks[-1]
    assert new_task.capability == "knowledge_graph.query"
    assert new_task.task_id != "T1"
    # The planned task resolves through the execution guard; nothing executed.
    resolved = require_planned_dispatch(outcome.plan, new_task.task_id, runtime)
    assert resolved.task_id == new_task.task_id
    for registration in _registrations():
        assert registration.capability.calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_tool_adapter_cannot_call_capability_directly() -> None:
    registry = build_capability_registry(
        _registrations(), _runtime().capability_runtime
    )
    adapter = AgentToolAdapter(registry)
    assert not hasattr(adapter, "execute")
    assert "execute" not in dir(adapter)
    proposal = adapter.to_proposal(
        AgentToolCall(
            capability="document.read",
            objective="Doc Dieu 5",
            input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
        )
    )
    assert isinstance(proposal, CapabilityInvocationProposal)
    assert proposal.capability == "document.read"
    # Translation is pure: no capability saw the call.
    for registration in _registrations():
        assert registration.capability.calls == []  # type: ignore[attr-defined]


def test_tool_gateway_is_proposal_and_observation_only() -> None:
    gateway = _gateway()
    assert callable(getattr(gateway, "propose"))
    for forbidden in ("execute", "dispatch", "checkpoint", "scheduler", "checkpointer"):
        assert not hasattr(gateway, forbidden), forbidden
    import app.services.agents.v2.tools.gateway as gateway_module

    source = inspect.getsource(gateway_module)
    for forbidden in (
        "capability.execute(",
        "TaskScheduler",
        "scheduler.execute(",
        "checkpointer",
    ):
        assert forbidden not in source, forbidden


def test_unplanned_capability_dispatch_is_rejected() -> None:
    runtime = _runtime()
    with pytest.raises(UnplannedCapabilityDispatch):
        require_planned_dispatch(_read_plan(), "T99", runtime)


def _t1_read_result() -> AgentResult:
    from app.services.agents.v2.contracts.capability import DocumentReadOutput
    from app.services.agents.v2.contracts.evaluation import CoverageObservation

    return AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=DocumentReadOutput(kind="document.read", read_unit_count=1),
        evidence_uses=(),
        coverage_observations=(
            CoverageObservation(
                target_id="t1",
                observed_locators=(DocumentLocator(kind="document"),),
                outcome="read",
            ),
        ),
        error=None,
    )


@pytest.mark.asyncio
async def test_unknown_or_unauthorized_tool_is_rejected_at_execution() -> None:
    from app.services.agents.v2.capabilities import CapabilityDenied
    from app.services.agents.v2.execution import execute_ready_tasks

    full_registry = build_capability_registry(
        _registrations(), _runtime().capability_runtime
    )

    # Unknown capability: the shared scheduler fails closed with a typed error.
    mystery_plan = _read_plan().model_copy(
        update={
            "tasks": (
                _read_task(),
                TaskSpec(
                    task_id="T2",
                    capability="mystery.tool",
                    task_objective="?",
                    input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
                    depends_on=(),
                    origin=InitialTaskOrigin(kind="initial"),
                ),
            )
        }
    )
    report = await execute_ready_tasks(
        plan=mystery_plan,
        results=(_t1_read_result(),),
        registry=full_registry,
        runtime=_runtime(registry=full_registry),
        bindings=_bindings(),
    )
    by_id = {result.task_id: result for result in report.results}
    assert by_id["T2"].status == "error"
    assert by_id["T2"].error is not None
    assert by_id["T2"].error.code == "CONTRACT_MISMATCH"

    # Unauthorized at execution: permitted at plan time, revoked before dispatch.
    gateway = _gateway()
    outcome = await gateway.propose(
        CapabilityInvocationProposal(
            capability="people.lookup",
            objective="Tra CCCD cua A",
            input=PeopleLookupInput(kind="people.lookup", query="A"),
        ),
        _read_plan(),
        _runtime(registry=full_registry),
    )
    assert outcome.accepted is True
    new_task_id = outcome.plan.tasks[-1].task_id

    # Same deployment, revoked runtime: people.lookup is registered but denied.
    revoked_runtime = _runtime(
        allowed=frozenset({"document.search", "document.read", "knowledge_graph.query"}),
        can_read_people=False,
    )
    revoked_registry = build_capability_registry(
        _registrations(), revoked_runtime.capability_runtime
    )
    revoked = _runtime(
        allowed=frozenset(
            {"document.search", "document.read", "knowledge_graph.query"}
        ),
        can_read_people=False,
        registry=revoked_registry,
    )
    # The pre-dispatch guard fails closed first...
    with pytest.raises(CapabilityDenied):
        require_planned_dispatch(outcome.plan, new_task_id, revoked)
    # ...and the shared scheduler owns the typed denial at dispatch time.
    denied_report = await execute_ready_tasks(
        plan=outcome.plan,
        results=(_t1_read_result(),),
        registry=revoked_registry,
        runtime=revoked,
        bindings=_bindings(),
    )
    denied = {result.task_id: result for result in denied_report.results}[new_task_id]
    assert denied.status == "denied"
    assert denied.error is not None
    assert denied.error.code == "PERMISSION_DENIED"


# ---------------------------------------------------------------------------
# Observation boundary: typed projections, no secrets, no raw records
# ---------------------------------------------------------------------------


def test_planner_never_receives_runtime_secrets() -> None:
    registry = build_capability_registry(
        _registrations(), _runtime().capability_runtime
    )
    adapter = AgentToolAdapter(registry)
    runtime_only = {
        "request_id",
        "run_id",
        "user_id",
        "workspace_ids",
        "can_read_people",
        "allowed_capabilities",
        "deadline_at",
    }
    for name in adapter.visible_tool_names():
        assert not (set(adapter.input_fields(name)) & runtime_only), name
    catalog_json = "[" + ",".join(d.model_dump_json() for d in adapter.describe()) + "]"
    for secret in ("workspace", "user_id", "deadline", "can_read_people", "allowed_"):
        assert secret not in catalog_json
    payload = ObservationProjector.project(_people_result()).model_dump_json()
    for secret in (
        "workspace",
        "user_id",
        "deadline",
        "secret",
        "token",
        "acl",
        "service",
        "storage_key",
    ):
        assert secret not in payload.lower()


def test_people_observation_does_not_expose_raw_record() -> None:
    observation = ObservationProjector.project(_people_result())
    assert isinstance(observation, AgentToolObservation)
    assert isinstance(observation.projection, PeopleLookupObservation)
    assert observation.projection.matched is True
    # R28 (Task 4 fix1): without a checkpointed materialization decision the
    # flag fails closed -- success + matched alone cannot prove extractability.
    assert observation.projection.dependency_scalar_available is False
    assert observation.evidence_use_ids == (USE_ID,)
    payload = observation.model_dump_json()
    assert "record_id" not in payload
    assert "0123456789" not in payload


def test_capability_output_is_not_model_observation_by_default() -> None:
    memory_result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=MemoryLookupOutput(kind="memory.lookup", matched_count=1),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    write_result = AgentResult(
        contract_version="2.0",
        task_id="T2",
        status="success",
        data=WriteOutput(kind="write", applied=True),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    with pytest.raises(ObservationProjectionUnavailable):
        ObservationProjector.project(memory_result)
    with pytest.raises(ObservationProjectionUnavailable):
        ObservationProjector.project(write_result)


def test_observation_projection_is_typed_no_mapping() -> None:
    observation = ObservationProjector.project(_search_result())
    assert isinstance(observation, ContractModel)
    assert isinstance(observation.projection, DocumentSearchObservation)
    import app.services.agents.v2.tools.observations as observations_module

    source = inspect.getsource(observations_module)
    assert "Mapping[str" not in source
    assert "safe_metadata" not in source
    for field_name, field in type(observation.projection).model_fields.items():
        assert "Mapping" not in str(field.annotation), field_name
        assert "dict" not in str(field.annotation).lower(), field_name


def test_unknown_result_kind_projection_fails_closed() -> None:
    abbreviation_result = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="success",
        data=AbbreviationResolveOutput(kind="abbreviation.resolve", resolutions=()),
        evidence_uses=(),
        coverage_observations=(),
        error=None,
    )
    with pytest.raises(ObservationProjectionUnavailable):
        ObservationProjector.project(abbreviation_result)
    denied = AgentResult(
        contract_version="2.0",
        task_id="T1",
        status="denied",
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error={"code": "PERMISSION_DENIED", "message": "no", "retryable": False},  # type: ignore[arg-type]
    )
    observation = ObservationProjector.project(denied)
    assert isinstance(observation.projection, NoObservation)


def test_document_search_observation_exposes_candidate_ids_only() -> None:
    observation = ObservationProjector.project(_search_result())
    projection = observation.projection
    assert isinstance(projection, DocumentSearchObservation)
    assert projection.candidate_count == 2
    assert projection.candidate_ids == (CANDIDATE_ID_1, CANDIDATE_ID_2)
    payload = observation.model_dump_json()
    assert REVISION_1 not in payload
    assert REVISION_2 not in payload
    assert "document_revision" not in payload


def test_planner_cannot_select_document_revision_directly() -> None:
    assert "document_revision" not in DocumentSearchObservation.model_fields
    assert "candidate_revision" not in DocumentSearchObservation.model_fields
    selectable = set(DocumentSearchInput.model_fields)
    assert not (selectable & {"candidate_id", "document_revision", "document_id"})


@pytest.mark.asyncio
async def test_planner_supplied_person_identifier_is_rejected() -> None:
    gateway = _gateway()
    outcome = await gateway.propose(
        CapabilityInvocationProposal(
            capability="document.search",
            objective="Tim nghi dinh bang CCCD",
            input=DocumentSearchInput(
                kind="document.search", query="CCCD", person_identifier="012345678901"
            ),
            depends_on=("T1",),
        ),
        _read_plan(),
        _runtime(),
    )
    assert outcome.accepted is False
    assert outcome.rejection is not None
    assert outcome.plan == _read_plan()


# ---------------------------------------------------------------------------
# Discovery candidates: ephemeral index over checkpointed AgentResult.data
# ---------------------------------------------------------------------------


def test_candidate_registry_reconstructs_from_checkpointed_agent_result() -> None:
    registry = DiscoveryCandidateRegistry.from_results([_search_result()])
    assert len(registry) == 2
    assert CANDIDATE_ID_1 in registry
    first = registry.get(CANDIDATE_ID_1)
    assert first.document_id == DOCUMENT_ID
    assert first.document_revision == REVISION_1
    assert registry.get(CANDIDATE_ID_2).document_revision == REVISION_2


def test_candidate_ids_survive_interrupt_and_process_restart() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph

    class CheckpointedResearch(TypedDict):
        results: tuple

    def run_search(state: CheckpointedResearch) -> dict:
        return {"results": (_search_result(), _people_result())}

    graph = StateGraph(CheckpointedResearch)
    graph.add_node("search", run_search)
    graph.set_entry_point("search")
    graph.set_finish_point("search")
    compiled = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "thread-restart-1"}}
    compiled.invoke({"results": ()}, config=config)

    # Interrupt boundary: read the REAL checkpointed AgentResult.data path.
    snapshot = compiled.get_state(config)
    checkpointed = snapshot.values["results"]
    assert len(checkpointed) == 2
    # Checkpoint storage degrades nested data to a plain mapping; the fresh
    # rebuild must prove itself against that degraded shape, not live models.
    from collections.abc import Mapping

    assert isinstance(checkpointed[0].data, Mapping)
    fresh = DiscoveryCandidateRegistry.from_results(checkpointed)
    assert [c.candidate_id for c in fresh.candidates()] == [
        CANDIDATE_ID_1,
        CANDIDATE_ID_2,
    ]
    assert fresh.get(CANDIDATE_ID_1) == _candidate(
        CANDIDATE_ID_1, DOCUMENT_ID, REVISION_1
    )
    assert fresh.get(CANDIDATE_ID_2).document_revision == REVISION_2

    # Resume shape: re-read the same thread after the interruption and rebuild
    # again from scratch; the index is stable with no persistent store.
    resumed = compiled.get_state(config)
    rebuilt = DiscoveryCandidateRegistry.from_results(resumed.values["results"])
    assert [c.candidate_id for c in rebuilt.candidates()] == [
        c.candidate_id for c in fresh.candidates()
    ]
    assert rebuilt.get(CANDIDATE_ID_1).document_revision == REVISION_1


def test_candidate_registry_does_not_require_persistence_table() -> None:
    import app.services.agents.v2.tools.discovery_candidates as registry_module

    source = inspect.getsource(registry_module)
    for forbidden in (
        "Table(",
        "Column",
        "migration",
        "create_table",
        "engine",
        "session",
        "checkpointer",
    ):
        assert forbidden not in source, forbidden
    registry = DiscoveryCandidateRegistry.from_results([_search_result()])
    for forbidden in ("save", "load", "persist", "commit", "engine", "session"):
        assert not hasattr(registry, forbidden), forbidden


def test_candidate_revision_remains_stable_if_newer_revision_publishes_before_resume() -> None:
    newer = _search_result(
        "T3",
        candidates=(_candidate(CANDIDATE_ID_1, DOCUMENT_ID, "rev-9"),),
    )
    registry = DiscoveryCandidateRegistry.from_results([_search_result(), newer])
    # The candidate identity pins the immutable discovered revision.
    assert registry.get(CANDIDATE_ID_1).document_revision == REVISION_1


def test_model_observation_never_exposes_candidate_revision() -> None:
    observation = ObservationProjector.project(_search_result())
    payload = observation.model_dump_json()
    assert str(CANDIDATE_ID_1) in payload
    assert REVISION_1 not in payload
    assert REVISION_2 not in payload
    assert "revision" not in payload.lower()


def test_binding_resolver_revalidates_candidate_acl_before_binding() -> None:
    from pathlib import Path

    from app.services.agents.v2.contracts.binding import BindingAdditionRequest

    registry = DiscoveryCandidateRegistry.from_results([_search_result()])

    # (i) The registry is an index ONLY: nothing on it returns a binding and
    # no authorization parameter exists anywhere on its surface.
    assert not hasattr(registry, "bind")
    for name in ("from_results", "get", "candidates"):
        for parameter in inspect.signature(getattr(registry, name)).parameters:
            assert "authoriz" not in parameter, (name, parameter)

    # (ii) No tools/ module references binding-construction contracts.
    # Resolve from this test file so the check is independent of cwd.
    tools_dir = (
        Path(__file__).resolve().parents[5] / "backend/app/services/agents/v2/tools"
    )
    assert tools_dir.is_dir()
    assert len(sorted(tools_dir.glob("*.py"))) >= 5
    for module_path in sorted(tools_dir.glob("*.py")):
        source = module_path.read_text()
        assert "ScopedDocument" not in source, module_path.name
        assert "DocumentBindingSet" not in source, module_path.name

    # (iii) The pure handoff rejects unknown candidates and invalid roles with
    # typed errors, and returns only a request carrying candidate UUID + role.
    with pytest.raises(CandidateNotFound):
        candidate_addition_request(registry, UUID(int=0), "discovered")
    with pytest.raises(InvalidCandidateRole):
        candidate_addition_request(registry, CANDIDATE_ID_1, "target")
    request = candidate_addition_request(registry, CANDIDATE_ID_1, "discovered")
    assert isinstance(request, BindingAdditionRequest)
    assert request.candidate_id == CANDIDATE_ID_1
    assert request.requested_role == "discovered"
    assert not isinstance(request, ScopedDocument)
    assert "document_revision" not in type(request).model_fields

    # (iv) The candidate revision stays server-side only: reachable via get(),
    # never through any model observation.
    assert registry.get(CANDIDATE_ID_1).document_revision == REVISION_1
    payload = ObservationProjector.project(_search_result()).model_dump_json()
    assert REVISION_1 not in payload
    assert "revision" not in payload.lower()


@pytest.mark.asyncio
async def test_malformed_proposal_input_returns_typed_rejection() -> None:
    from types import SimpleNamespace

    gateway = _gateway()
    current = _read_plan()
    # A duck-typed object with a matching kind is NOT the frozen CapabilityInput:
    # it must fail closed as a typed rejection, never escape as an exception.
    outcome = await gateway.propose(
        CapabilityInvocationProposal(  # type: ignore[arg-type]
            capability="document.read",
            objective="Doc Dieu 5",
            input=SimpleNamespace(kind="document.read", target_ids=("t1",)),
        ),
        current,
        _runtime(),
    )
    assert outcome.accepted is False
    assert outcome.rejection is not None
    assert outcome.rejection.code == "invalid_proposal"
    assert outcome.plan == current


@pytest.mark.asyncio
async def test_stale_registry_cannot_widen_current_runtime_authority() -> None:
    from app.services.agents.v2.capabilities import CapabilityDenied

    stale_full_registry = build_capability_registry(
        _registrations(), _runtime().capability_runtime
    )
    gateway = _gateway()

    # Stale full registry attached, but current runtime revoked People access.
    outcome = await gateway.propose(
        CapabilityInvocationProposal(
            capability="people.lookup",
            objective="Tra CCCD cua A",
            input=PeopleLookupInput(kind="people.lookup", query="A"),
        ),
        _read_plan(),
        _runtime(can_read_people=False, registry=stale_full_registry),
    )
    assert outcome.accepted is False
    assert outcome.rejection is not None
    assert outcome.rejection.code == "unauthorized_capability"
    assert outcome.plan == _read_plan()

    # Stale full registry attached, but capability absent from allowed set.
    narrowed = await gateway.propose(
        CapabilityInvocationProposal(
            capability="document.read",
            objective="Doc Dieu 5",
            input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
        ),
        _read_plan(),
        _runtime(
            allowed=frozenset({"people.lookup"}), registry=stale_full_registry
        ),
    )
    assert narrowed.accepted is False
    assert narrowed.rejection is not None
    assert narrowed.rejection.code == "unauthorized_capability"

    # The pre-dispatch guard narrows the same way.
    with pytest.raises(CapabilityDenied):
        require_planned_dispatch(
            _read_plan(),
            "T1",
            _runtime(allowed=frozenset({"people.lookup"}), registry=stale_full_registry),
        )


def test_agent_tool_catalog_is_permission_intersected() -> None:
    allowed_runtime = _runtime(
        allowed=frozenset({"document.read"}), can_read_people=False
    )
    registry = build_capability_registry(_registrations(), allowed_runtime.capability_runtime)
    catalog = build_agent_tool_catalog(registry)
    assert {d.name for d in catalog} == {"document.read"}
    adapter = AgentToolAdapter(registry)
    assert adapter.visible_tool_names() == frozenset({"document.read"})
    assert adapter.is_visible("people.lookup") is False


# ---------------------------------------------------------------------------
# P0 Task 2: document.retrieve projects a count-only typed observation
# ---------------------------------------------------------------------------


def _retrieve_result(task_id: str = "T1", count: int = 2) -> AgentResult:
    from app.services.agents.v2.contracts.capability import DocumentRetrieveOutput

    return AgentResult(
        contract_version="2.0",
        task_id=task_id,
        status="success",
        data=DocumentRetrieveOutput(
            kind="document.retrieve", retrieved_unit_count=count
        ),
        evidence_uses=(EvidenceUseRef(use_id=USE_ID),),
        coverage_observations=(),
        error=None,
    )


def test_document_retrieve_observation_is_count_only() -> None:
    from app.services.agents.v2.tools.observations import DocumentRetrieveObservation

    observation = ObservationProjector.project(_retrieve_result())
    assert isinstance(observation.projection, DocumentRetrieveObservation)
    assert observation.projection.retrieved_unit_count == 2
    assert observation.evidence_use_ids == (USE_ID,)
    assert observation.result_kind == "document.retrieve"
    # Count-only: no chunk content, locator, or revision carrier exists.
    payload = observation.model_dump_json()
    assert "secret chunk" not in payload
    assert "revision" not in payload.lower()


# ---------------------------------------------------------------------------
# P0 Task 5: document.retrieve model-facing schema through the real registry
# ---------------------------------------------------------------------------


def test_task5_document_retrieve_schema_projection_through_real_registry() -> None:
    """The real request-scoped registry projects a governed retrieve schema."""
    from datetime import datetime, timezone

    from app.services.agents import supervisor_v2
    from app.services.agents.v2.contracts.capability import (
        CapabilityRuntimeContext,
        DocumentRetrieveInput,
    )
    from app.services.agents.v2.tools.adapters import AgentToolAdapter

    runtime = CapabilityRuntimeContext(
        request_id="req-task5-gw",
        run_id="run-task5-gw",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=frozenset({"document.retrieve"}),
        deadline_at=datetime.now(timezone.utc),
    )

    class _FakeRetrieval:
        pass

    class _StubSession:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *args: object) -> None:
            return None

    registry = supervisor_v2.build_v2_capability_registry(
        runtime,
        bundle=supervisor_v2.V1ServiceBundle(
            session_factory=lambda: _StubSession(),
            document_retrieval=_FakeRetrieval(),
        ),
        evidence=object(),
        resolver=object(),
        available_services=frozenset({"v1-revision-retrieval"}),
    )
    adapter = AgentToolAdapter(registry)

    assert adapter.is_visible("document.retrieve") is True
    assert adapter.input_fields("document.retrieve") == tuple(
        DocumentRetrieveInput.model_fields
    )
    proposal = adapter.to_proposal(
        AgentToolCall(
            capability="document.retrieve",
            objective="answer the factual question",
            input=DocumentRetrieveInput(
                kind="document.retrieve", query="what changed", top_k=8
            ),
        )
    )
    assert proposal.capability == "document.retrieve"
    assert isinstance(proposal.input, DocumentRetrieveInput)
    # Runtime authority is never model-supplied: no scope/ACL field exists.
    for forbidden in ("workspace_ids", "document_ids", "namespace"):
        assert forbidden not in adapter.input_fields("document.retrieve")

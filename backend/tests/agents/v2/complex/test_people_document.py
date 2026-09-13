"""People -> Document deterministic dependency materialization (Phase 3, Task 4).

The governed People scalar is materialized server-side into a concrete
``DocumentSearchInput`` BEFORE the dependent task is appended/checkpointed:
never an agent handoff, never inside the scheduler. ``depends_on`` expresses
ordering only; the scheduler executes ``TaskSpec.input`` exactly as
checkpointed and never mutates or lazily materializes it.

Proof style (R27): the required ordering/checkpoint/no-T2 properties are proven
on the REAL compiled subgraph with a real saver
(``build_complex_research_subgraph(checkpointer=InMemorySaver)``); helper-level
unit tests remain as additional coverage only.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.services.agents.v2.capabilities import (
    CapabilityRegistration,
    build_capability_registry,
)
from app.services.agents.v2.complex_research_graph import (
    _build_complex_research_graph,
    people_document_materialize_node,
)


def _compile_subgraph(saver: InMemorySaver):  # real saver, test-owned (R27)
    return _build_complex_research_graph().compile(checkpointer=saver)
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    DocumentSearchInput,
    DocumentSearchOutput,
    PeopleLookupInput,
    PeopleLookupOutput,
)
from app.services.agents.v2.contracts.evidence import DocumentSourceIdentity, EvidenceUseRef, PeopleSourceIdentity
from app.services.agents.v2.contracts.execution import (
    AgentError,
    AgentRequest,
    AgentResult,
    TaskExecutionSummary,
)
from app.services.agents.v2.contracts.planning import (
    DiscoveryPolicy,
    InitialTaskOrigin,
    ResearchBudgetView,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.routing import QueryAnalysis
from app.services.agents.v2.contracts.semantic import SemanticContext
from app.services.agents.v2.contracts.state import (
    GraphRuntimeContext,
    RuntimeServices,
)
from app.services.agents.v2.dependencies.people_document import (
    PERSON_IDENTIFIER_FIELD,
    MaterializationError,
    PeopleDocumentMaterialization,
    append_materialized_dependent,
    build_dependent_search_task,
    extract_person_identifier,
    materialize_person_dependency,
    redact_scalar_for_model,
)
from app.services.agents.v2.nodes.evaluate import HydratedEvidence
from app.services.agents.v2.tools.observations import (
    ObservationProjector,
    PeopleLookupObservation,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
USE_ID = UUID("44444444-4444-4444-4444-444444444444")
EVIDENCE_ID = UUID("55555555-5555-5555-5555-555555555555")
SCALAR = "079000000001"
OTHER_SCALAR = "079000000002"
GOAL = "A xuat hien trong nghi dinh nao"

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


def _hydrated(
    content: str | None,
    *,
    use_id: UUID = USE_ID,
    task_id: str = "T1",
    purpose: str = "supporting",
    target_id: str | None = None,
    source_label: str | None = "people",
    source_identity: object | None = None,
) -> HydratedEvidence:
    return HydratedEvidence(
        use_id=use_id,
        evidence_id=EVIDENCE_ID,
        task_id=task_id,
        purpose=purpose,  # type: ignore[arg-type]
        target_id=target_id,
        content=content,  # type: ignore[arg-type]
        role=None,
        source_label=source_label,
        source_identity=(
            source_identity
            if source_identity is not None
            else PeopleSourceIdentity(kind="people", record_id="rec-1")
        ),
        classification="personal",
        locator=None,
        document_revision=None,
    )


class FakeHydrator:
    """Governed-hydration stand-in: admits exactly the mapped uses."""

    def __init__(self, admitted: tuple[HydratedEvidence, ...] = ()) -> None:
        self._by_use = {item.use_id: item for item in admitted}

    async def hydrate_for_evaluation(
        self,
        use_refs: tuple[EvidenceUseRef, ...],
        *,
        runtime: object,
        plan: object,
        bindings: object,
    ) -> tuple[HydratedEvidence, ...]:
        return tuple(
            self._by_use[ref.use_id] for ref in use_refs if ref.use_id in self._by_use
        )


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


class StubPeopleCapability:
    """Atomic people.lookup stub returning a governed success."""

    def __init__(self, use_id: UUID = USE_ID) -> None:
        self.descriptor = CapabilityDescriptor(
            name="people.lookup",  # type: ignore[arg-type]
            domain="people",  # type: ignore[arg-type]
            operation_type="lookup",
            supports_parallel=True,
        )
        self._use_id = use_id
        self.calls: list[tuple[AgentRequest, CapabilityRuntimeContext]] = []

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        self.calls.append((request, runtime))
        return AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=request.task_id,
            status="success",
            data=PeopleLookupOutput(kind="people.lookup", matched=True),
            evidence_uses=(EvidenceUseRef(use_id=self._use_id),),
            coverage_observations=(),
            error=None,
        )


class StubSearchCapability:
    """Atomic document.search stub: records the exact checkpointed input."""

    def __init__(self) -> None:
        self.descriptor = CapabilityDescriptor(
            name="document.search",  # type: ignore[arg-type]
            domain="document",  # type: ignore[arg-type]
            operation_type="search",
            supports_parallel=True,
        )
        self.calls: list[tuple[AgentRequest, CapabilityRuntimeContext]] = []

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        self.calls.append((request, runtime))
        return AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=request.task_id,
            status="success",
            data=DocumentSearchOutput(kind="document.search", candidates=()),
            evidence_uses=(),
            coverage_observations=(),
            error=None,
        )


def _runtime(
    *,
    hydrator: FakeHydrator | None = None,
    leases: FakeLeaseRepo | None = None,
    registry: object = None,
) -> GraphRuntimeContext:
    return GraphRuntimeContext(
        capability_runtime=CapabilityRuntimeContext(
            request_id="req-1",
            run_id="run-1",
            user_id=USER_ID,
            workspace_ids=(WORKSPACE_ID,),
            can_read_people=True,
            allowed_capabilities=frozenset({"people.lookup", "document.search"}),
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
        goal=GOAL,
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


def _semantic() -> SemanticContext:
    return SemanticContext(
        contextualized_query=GOAL,
        normalized_query=GOAL,
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def _analysis() -> QueryAnalysis:
    # A non-compare work type: the compare skill fails closed in
    # validate_checkpoint, so the input T1 plan passes through untouched.
    return QueryAnalysis(work_type="lookup", domains=("people",))


def _materialized_outcome() -> PeopleDocumentMaterialization:
    return PeopleDocumentMaterialization(
        kind="materialized",
        scalar=SCALAR,
        input=DocumentSearchInput(
            kind="document.search", query="nghi dinh", person_identifier=SCALAR
        ),
        error_code=None,
        reason="governed scalar materialized server-side",
        evidence_use_ids=(USE_ID,),
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
# R25: redacted model-facing projection vs checkpointed plan
# ---------------------------------------------------------------------------


def test_redact_scalar_for_model_keeps_checkpoint_but_hides_scalar() -> None:
    plan = _people_plan()
    task = build_dependent_search_task(
        _materialized_outcome(), people_task_id="T1", query="nghi dinh", next_task_id="T2"
    )
    checkpointed = plan.model_copy(update={"tasks": plan.tasks + (task,)})
    projected = redact_scalar_for_model(checkpointed)
    # The checkpointed plan keeps the governed scalar (frozen contract).
    assert checkpointed.tasks[1].input.person_identifier == SCALAR  # type: ignore[union-attr]
    # The model-facing projection carries none of it.
    redacted_input = projected.tasks[1].input
    assert isinstance(redacted_input, DocumentSearchInput)
    assert redacted_input.person_identifier is None
    assert redacted_input.query == "nghi dinh"
    assert projected.tasks[1].task_id == "T2"
    assert projected.tasks[1].depends_on == ("T1",)
    assert SCALAR not in projected.model_dump_json()
    # The checkpointed plan is untouched by the projection.
    assert checkpointed.tasks[1].input.person_identifier == SCALAR  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# R28.2: the typed schema is the only carrier (no raw-field escape hatch)
# ---------------------------------------------------------------------------


def test_people_projection_schema_carries_no_raw_field() -> None:
    assert set(PeopleLookupObservation.model_fields) == {
        "kind",
        "matched",
        "dependency_scalar_available",
    }


# ---------------------------------------------------------------------------
# R28.3: availability reflects the materialization decision, fail-closed default
# ---------------------------------------------------------------------------


def test_availability_defaults_to_unavailable_without_materialization_decision() -> None:
    observation = ObservationProjector.project(_people_success())
    assert observation.projection.kind == "people.lookup"
    assert observation.projection.matched is True
    # success + matched alone cannot prove extractability (R24).
    assert observation.projection.dependency_scalar_available is False


@pytest.mark.asyncio
async def test_availability_reflects_materialization_decision() -> None:
    plan = _people_plan()
    runtime = _runtime(hydrator=FakeHydrator((_hydrated(_minimized_content(SCALAR)),)))
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=_people_success(),
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert outcome.kind == "materialized"
    available = ObservationProjector.project(
        _people_success(), dependency_scalar_available=(outcome.kind == "materialized")
    )
    assert available.projection.dependency_scalar_available is True

    # Matched but not extractable: the planner is told unavailable.
    denied_runtime = _runtime(hydrator=FakeHydrator(()))
    denied_outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=_people_success(),
        runtime=denied_runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert denied_outcome.kind == "unavailable"
    unavailable = ObservationProjector.project(
        _people_success(),
        dependency_scalar_available=(denied_outcome.kind == "materialized"),
    )
    assert unavailable.projection.dependency_scalar_available is False


# ---------------------------------------------------------------------------
# R28.1: hydrated evidence must be People evidence owned by T1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_materializer_rejects_evidence_owned_by_another_task() -> None:
    plan = _people_plan()
    runtime = _runtime(
        hydrator=FakeHydrator((_hydrated(_minimized_content(SCALAR), task_id="OTHER"),))
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
    assert "T1" in outcome.reason


@pytest.mark.asyncio
async def test_materializer_rejects_non_people_evidence() -> None:
    from app.services.agents.v2.contracts.locators import DocumentLocator

    plan = _people_plan()
    doc_identity = DocumentSourceIdentity(
        kind="document",
        document_id=UUID("11111111-1111-1111-1111-111111111111"),
        document_revision="rev-1",
        locator=DocumentLocator(kind="document"),
    )
    for bad in (
        # REAL wrong-source proof (R29): the TYPED identity is a document
        # record carrying an extractable scalar -- and the label even claims
        # "people". The typed identity check still refuses it: labels are
        # not a security boundary.
        _hydrated(_minimized_content(SCALAR), source_identity=doc_identity),
        _hydrated(_minimized_content(SCALAR), purpose="coverage", target_id="t1"),
        _hydrated(content=None),
    ):
        runtime = _runtime(hydrator=FakeHydrator((bad,)))
        outcome = await materialize_person_dependency(
            people_task_id="T1",
            people_result=_people_success(),
            runtime=runtime,
            plan=plan,
            bindings=_bindings(),
            query="nghi dinh",
        )
        assert outcome.kind == "unavailable", bad
        assert outcome.input is None
        assert outcome.scalar is None, bad


def test_hydrated_evidence_without_identity_fails() -> None:
    """R30.1: the typed identity is REQUIRED -- omitting it is a TypeError."""
    with pytest.raises(TypeError):
        HydratedEvidence(  # type: ignore[call-arg]
            use_id=USE_ID,
            evidence_id=EVIDENCE_ID,
            task_id="T1",
            purpose="supporting",
            target_id=None,
            content=_minimized_content(SCALAR),
            role=None,
            source_label="people",
            classification="personal",
            locator=None,
            document_revision=None,
        )


@pytest.mark.asyncio
async def test_typed_wrong_source_appends_no_t2_on_real_node() -> None:
    """R30.2: typed non-People identity -> no scalar AND no T2 end-to-end."""
    from app.services.agents.v2.contracts.locators import DocumentLocator

    doc_identity = DocumentSourceIdentity(
        kind="document",
        document_id=UUID("11111111-1111-1111-1111-111111111111"),
        document_revision="rev-1",
        locator=DocumentLocator(kind="document"),
    )
    hydrator = FakeHydrator((_hydrated(_minimized_content(SCALAR), source_identity=doc_identity),))
    runtime = _runtime(hydrator=hydrator)
    state = {
        "contract_version": "2.0",
        "semantic": _semantic(),
        "bindings": _bindings(),
        "query_analysis": _analysis(),
        "plan": _people_plan(),
        "task_results": (_people_success(),),
        "replans_remaining": 0,
    }
    update = await people_document_materialize_node(state, runtime)  # type: ignore[arg-type]
    assert update.get("plan", None) is None
    assert update.get("materialized_new_task", False) is False
    assert update.get("people_scalar_available", {}) == {"T1": False}
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=_people_success(),
        runtime=runtime,
        plan=_people_plan(),
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert outcome.kind == "unavailable"
    assert outcome.scalar is None
    assert outcome.input is None


# ---------------------------------------------------------------------------
# R28.4: the append path can never overwrite the materialized scalar
# ---------------------------------------------------------------------------


def test_build_never_overwrites_materialized_scalar() -> None:
    task = build_dependent_search_task(
        _materialized_outcome(),
        people_task_id="T1",
        query="a completely different query",
        next_task_id="T2",
    )
    assert isinstance(task.input, DocumentSearchInput)
    assert task.input.query == "a completely different query"
    assert task.input.person_identifier == SCALAR
    with pytest.raises(MaterializationError):
        build_dependent_search_task(
            PeopleDocumentMaterialization(
                kind="unavailable",
                scalar=None,
                input=None,
                error_code=None,
                reason="no T2",
            ),
            people_task_id="T1",
            query="nghi dinh",
            next_task_id="T2",
        )


# ---------------------------------------------------------------------------
# Graph-level proofs on the REAL subgraph with a real saver (R26/R27)
# ---------------------------------------------------------------------------


def _graph_harness() -> tuple[
    StubPeopleCapability, StubSearchCapability, FakeHydrator, FakeLeaseRepo, list[str]
]:
    people = StubPeopleCapability()
    search = StubSearchCapability()
    hydrator = FakeHydrator((_hydrated(_minimized_content(SCALAR)),))
    events: list[str] = []
    leases = FakeLeaseRepo(events)
    return people, search, hydrator, leases, events


def _graph_runtime(
    people: StubPeopleCapability,
    search: StubSearchCapability,
    hydrator: FakeHydrator,
    leases: FakeLeaseRepo,
) -> GraphRuntimeContext:
    base = _runtime()
    registry = build_capability_registry(
        [CapabilityRegistration(capability=people), CapabilityRegistration(capability=search)],
        base.capability_runtime,
    )
    return _runtime(hydrator=hydrator, leases=leases, registry=registry)


def _graph_input() -> dict:
    return {
        "contract_version": "2.0",
        "semantic": _semantic(),
        "bindings": _bindings(),
        "query_analysis": _analysis(),
        "plan": _people_plan(),
        "task_results": (),
        "replans_remaining": 0,
    }


def _slot(values: object, name: str) -> object:
    if isinstance(values, dict):
        return values.get(name)
    return getattr(values, name)


def _tasks_of(plan: object) -> tuple:
    return tuple(_slot(plan, "tasks") or ())


def _results_of(values: object) -> tuple:
    results = _slot(values, "task_results") or ()
    return tuple(results)


@pytest.mark.asyncio
async def test_people_document_materializes_before_task_append() -> None:
    """R26/R27: T1 executes -> materialize appends T2 -> scheduler runs T2.

    Proves on the real compiled subgraph with a real saver that the T2
    checkpoint (concrete input) precedes the T2 result, and that the final
    checkpointed plan carries both tasks with both results present.
    """
    people, search, hydrator, leases, events = _graph_harness()
    runtime = _graph_runtime(people, search, hydrator, leases)
    saver = InMemorySaver()
    graph = _compile_subgraph(saver)
    config = {"configurable": {"thread_id": "thread-people-doc-1"}}

    await graph.ainvoke(_graph_input(), config=config, context=runtime)

    snapshot = await graph.aget_state(config)
    plan = snapshot.values["plan"]
    tasks = _tasks_of(plan)
    assert [t["task_id"] if isinstance(t, dict) else t.task_id for t in tasks] == [
        "T1",
        "T2",
    ]
    results = _results_of(snapshot.values)
    assert [r["task_id"] if isinstance(r, dict) else r.task_id for r in results] == [
        "T1",
        "T2",
    ]
    # The materialize node appended T2 and the scheduler dispatched it: the
    # checkpointed T2 input is exactly what the search capability received.
    t2 = tasks[1]
    t2_input = _slot(t2, "input")
    assert len(search.calls) == 1
    dispatched = search.calls[0][0].input
    assert dispatched == (
        t2_input
        if not isinstance(t2_input, dict)
        else DocumentSearchInput.model_validate(t2_input)
    )
    assert dispatched.person_identifier == SCALAR
    # Ordering proof from the real checkpoint history: the first checkpoint
    # whose plan contains T2 carries no T2 result yet -- the append preceded
    # the dispatch, never the reverse.
    history: list[dict] = []
    async for tup in saver.alist(config):
        history.append(dict(tup.checkpoint["channel_values"]))
    history.reverse()
    first_t2_at = next(
        (
            index
            for index, values in enumerate(history)
            if values.get("plan") is not None
            and "T2"
            in [
                t.get("task_id", "") if isinstance(t, dict) else t.task_id
                for t in _tasks_of(values["plan"])
            ]
        ),
        None,
    )
    assert first_t2_at is not None, "T2 was never checkpointed"
    first_results = _results_of(history[first_t2_at])
    assert [
        r.get("task_id", "") if isinstance(r, dict) else r.task_id for r in first_results
    ] == ["T1"]
    assert "commit" in events, "leases were committed before the T2 checkpoint"


@pytest.mark.asyncio
async def test_t2_checkpoint_contains_final_document_search_input() -> None:
    """R27: the ACTUAL LangGraph checkpoint carries the concrete T2 input."""
    people, search, hydrator, leases, _ = _graph_harness()
    runtime = _graph_runtime(people, search, hydrator, leases)
    saver = InMemorySaver()
    graph = _compile_subgraph(saver)
    config = {"configurable": {"thread_id": "thread-people-doc-2"}}

    await graph.ainvoke(_graph_input(), config=config, context=runtime)

    snapshot = await graph.aget_state(config)
    t2 = _tasks_of(snapshot.values["plan"])[1]
    t2_input = _slot(t2, "input")
    if isinstance(t2_input, dict):
        t2_input = DocumentSearchInput.model_validate(t2_input)
    assert isinstance(t2_input, DocumentSearchInput)
    assert t2_input.person_identifier == SCALAR
    assert t2_input.query == GOAL
    depends = _slot(t2, "depends_on")
    assert list(depends) == ["T1"]


@pytest.mark.asyncio
async def test_graph_never_appends_t2_without_governed_scalar() -> None:
    """R27: not_found/denied/timeout/unavailable on the real node -> no T2."""
    from app.services.agents.v2.contracts.capability import PeopleLookupOutput as _Out

    cases = {
        "not_found": AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id="T1",
            status="not_found",
            data=_Out(kind="people.lookup", matched=False),
            evidence_uses=(),
            coverage_observations=(),
            error=None,
        ),
        "denied": AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id="T1",
            status="denied",
            data=None,
            evidence_uses=(),
            coverage_observations=(),
            error=AgentError(
                code="PERMISSION_DENIED", message="denied", retryable=False
            ),
        ),
        "timeout": AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id="T1",
            status="error",
            data=None,
            evidence_uses=(),
            coverage_observations=(),
            error=AgentError(code="TIMEOUT", message="timed out", retryable=True),
        ),
        "unavailable": _people_success(),  # matched, but hydrator admits nothing
    }
    kinds = {"not_found": "not_found", "denied": "denied", "timeout": "failed"}
    for name, result in cases.items():
        hydrator = FakeHydrator(())
        runtime = _runtime(hydrator=hydrator)
        state = {
            "contract_version": "2.0",
            "semantic": _semantic(),
            "bindings": _bindings(),
            "query_analysis": _analysis(),
            "plan": _people_plan(),
            "task_results": (result,),
            "replans_remaining": 0,
        }
        update = await people_document_materialize_node(state, runtime)  # type: ignore[arg-type]
        assert update.get("plan", None) is None, name
        assert update.get("materialized_new_task", False) is False, name
        if name in kinds:
            outcome = await materialize_person_dependency(
                people_task_id="T1",
                people_result=result,
                runtime=runtime,
                plan=_people_plan(),
                bindings=_bindings(),
                query="nghi dinh",
            )
            assert outcome.kind == kinds[name], name
            assert outcome.input is None, name
    # The unavailable case is the distinct typed outcome, not not_found.
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=_people_success(),
        runtime=_runtime(hydrator=FakeHydrator(())),
        plan=_people_plan(),
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert outcome.kind == "unavailable"
    assert outcome.input is None


# ---------------------------------------------------------------------------
# Scheduler + observation properties (helper-level additional coverage)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_never_rewrites_task_input() -> None:
    from app.services.agents.v2.execution.scheduler import execute_ready_tasks

    plan = TaskPlan(
        contract_version=CONTRACT_VERSION,
        plan_id="p1",
        goal=GOAL,
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
    recorded: list[AgentRequest] = []

    class RecordingSearch(StubSearchCapability):
        async def execute(  # type: ignore[override]
            self, request: AgentRequest, runtime: CapabilityRuntimeContext
        ) -> AgentResult:
            recorded.append(request)
            return search_result

    recording = RecordingSearch()
    runtime = _runtime(
        registry=build_capability_registry(
            [CapabilityRegistration(capability=recording)],
            _runtime().capability_runtime,
        )
    )
    report = await execute_ready_tasks(
        plan=plan,
        results=(_people_success(),),
        registry=runtime.services.capability_registry,
        runtime=runtime,
        bindings=_bindings(),
    )
    assert len(recorded) == 1
    request = recorded[0]
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
    # No materialization decision known -> fail closed.
    assert observation.projection.dependency_scalar_available is False
    dumped = observation.model_dump_json().lower()
    assert SCALAR not in dumped
    for token in FORBIDDEN_OBSERVATION_TOKENS:
        assert token not in dumped


def test_synthesis_projection_exposes_label_only_never_typed_identity() -> None:
    """R29.5: the typed source identity must NOT reach the synthesis model."""
    from app.services.agents.v2.contracts.synthesis import SynthesisEvidence
    from app.services.agents.v2.nodes.synthesize import _project

    assert "source_identity" not in SynthesisEvidence.model_fields
    item = _hydrated(_minimized_content(SCALAR))
    assert isinstance(item.source_identity, PeopleSourceIdentity)
    projected = _project(item)
    assert projected.source_label == "people"
    assert not hasattr(projected, "source_identity")
    assert "source_identity" not in projected.model_dump_json()


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
            outcome=outcome,
            query="nghi dinh",
            next_task_id="T2",
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
        hydrator=FakeHydrator((_hydrated(_minimized_content(None)),))
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
                _hydrated(_minimized_content(SCALAR)),
                _hydrated(_minimized_content(OTHER_SCALAR), use_id=use2),
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
    runtime = _runtime(hydrator=FakeHydrator((_hydrated(_minimized_content(SCALAR)),)))
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=_people_success(),
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    # R50: append_materialized_dependent returns the concrete T2 PROPOSAL;
    # the single governed append builds the authoritative plan.
    from app.services.agents.v2.replanning import append_replan_tasks

    dependent = append_materialized_dependent(
        current=plan,
        outcome=outcome,
        query="nghi dinh",
        next_task_id="T2",
    )
    # R52: the single governed append + validation happen inside
    # append_replan_tasks; the concrete T2 is only a proposal here.
    proposed = append_replan_tasks(
        plan,
        (dependent,),
        (TaskExecutionSummary(task_id="T1", status="success"),),
        DiscoveryPolicy(
            allow_reference_discovery=False,
            allow_supporting_discovery=True,
            max_discovered_documents=1,
        ),
        ResearchBudgetView(
            max_tasks_remaining=8,
            max_replans_remaining=1,
            max_parallel_branches=2,
        ),
        runtime,
    )
    dumped = proposed.model_dump_json().lower()
    for token in FORBIDDEN_OBSERVATION_TOKENS:
        if token == PERSON_IDENTIFIER_FIELD:
            continue
        assert token not in dumped
    assert "1990-01-01" not in dumped
    assert "0900000000" not in dumped


@pytest.mark.asyncio
async def test_full_materialization_appends_validated_t2() -> None:
    plan = _people_plan()
    runtime = _runtime(hydrator=FakeHydrator((_hydrated(_minimized_content(SCALAR)),)))
    outcome = await materialize_person_dependency(
        people_task_id="T1",
        people_result=_people_success(),
        runtime=runtime,
        plan=plan,
        bindings=_bindings(),
        query="nghi dinh",
    )
    assert outcome.kind == "materialized"
    assert isinstance(outcome, PeopleDocumentMaterialization)
    assert outcome.scalar == SCALAR
    assert outcome.input is not None
    assert outcome.input.person_identifier == SCALAR
    assert [task.task_id for task in plan.tasks] == ["T1"]
    task = build_dependent_search_task(
        outcome, people_task_id="T1", query="nghi dinh", next_task_id="T2"
    )
    assert task.task_id == "T2"
    assert task.capability == "document.search"
    assert task.depends_on == ("T1",)
    assert isinstance(task.input, DocumentSearchInput)
    assert task.input.person_identifier == SCALAR

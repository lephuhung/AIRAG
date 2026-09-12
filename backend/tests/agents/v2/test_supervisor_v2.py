"""Task 6 — compose node-based supervisor v2 (composition tests).

Red-phase first: this module imports ``supervisor_v2`` and ``v2.events``,
which do not exist yet, so collection must fail until the implementation
lands. All graph runs use fakes + ``InMemorySaver`` (no database, no
network); the only production import under test is the frozen contract
layer plus the T1–T5 node functions being composed.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.services.agents.v2.capabilities import (
    CapabilityDescriptor,
    CapabilityRegistration,
    CapabilityRuntimeContext,
    build_capability_registry,
)
from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.capability import PeopleLookupOutput
from app.services.agents.v2.contracts.conversation import (
    ConversationContext,
    ConversationTurn,
    EntityReference,
)
from app.services.agents.v2.contracts.evidence import EvidenceUseRef
from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.semantic import DocumentReference, SemanticContext, SemanticDraft
from app.services.agents.v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.validation import validate_supervisor_state
from app.services.agents.v2.nodes.binding import binding_node
from app.services.agents.v2.nodes.clarification import build_clarification
from app.services.agents.v2.nodes.context import context_node, semantic_finalizer_node
from app.services.agents.v2.nodes.evaluate import HydratedEvidence, evaluate_node
from app.services.agents.v2.nodes.execute import MissingCheckpointedPlan, execute_node
from app.services.agents.v2.nodes.finalizer import finalizer_node
from app.services.agents.v2.nodes.routing import route_node
from app.services.agents.v2.nodes.evaluate import AnswerDraftChannel
from app.services.agents.v2.nodes.fast_plan import fast_plan_node
from app.services.agents.v2.nodes.grounding import ground_node
from app.services.agents.v2.nodes.synthesize import synthesize_node

from app.services.agents.supervisor_v2 import (
    SUPERVISOR_V2_NODES,
    create_supervisor_v2_graph,
    normalize_checkpoint_state,
)
from app.services.agents.v2.events import (
    SSE_COMPLETE,
    SSE_ERROR,
    final_response_payload,
    format_sse_event,
    terminal_event_for_response,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")
REVISION_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
OTHER_REVISION_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
USE_ID = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")
EVIDENCE_ID = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")

FULL_CAPABILITIES = frozenset(
    {
        "people.lookup",
        "document.search",
        "document.read",
        "section.read",
        "knowledge_graph.query",
        "memory.lookup",
        "abbreviation.resolve",
    }
)

EXPECTED_NODES = (
    "context",
    "binding",
    "semantic_finalizer",
    "route",
    "direct",
    "clarify",
    # Persist and suspend split: suspending in the persist node (or its
    # edge) discards the persist update, so the suspension checkpoint would
    # carry a clarify route with no request. Two super-steps keep D6.
    "clarify_wait",
    "fast_plan",
    "execute",
    "evaluate",
    "synthesize",
    "ground",
    "finalizer",
    "complex_boundary",
)


# ---------------------------------------------------------------------------
# Fakes (mirror the T1/T3 test style: real RuntimeServices, fake services)
# ---------------------------------------------------------------------------


class FakeSemanticAdapter:
    def __init__(self, draft: SemanticDraft) -> None:
        self._draft = draft
        self.calls: list[tuple[RequestContext, ConversationContext]] = []

    async def build_draft(
        self, request: RequestContext, conversation: ConversationContext
    ) -> SemanticDraft:
        self.calls.append((request, conversation))
        return self._draft


class FakeBindingResolver:
    def __init__(self, binding_set: DocumentBindingSet) -> None:
        self._binding_set = binding_set
        self.calls: list[tuple[Any, CapabilityRuntimeContext]] = []

    async def resolve(
        self, document_refs: Any, capability_runtime: CapabilityRuntimeContext
    ) -> DocumentBindingSet:
        self.calls.append((document_refs, capability_runtime))
        return self._binding_set


class FakeSession:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def commit(self) -> None:
        self._events.append("commit")


class FakeLeaseRepo:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.session = FakeSession(events)

    async def acquire_or_refresh(
        self, run_id: str, revision_id: Any, evidence_use_id: Any = None, **kwargs: Any
    ) -> SimpleNamespace:
        self.events.append(f"acquire:{revision_id}:{evidence_use_id}")
        return SimpleNamespace(run_id=run_id, revision_id=revision_id)


class FakeCapability:
    """Records exactly what the scheduler hands to a capability."""

    def __init__(
        self, *, name: str = "people.lookup", domain: str = "people", result: AgentResult
    ) -> None:
        self.descriptor = CapabilityDescriptor(
            name=name,
            domain=domain,  # type: ignore[arg-type]
            operation_type="lookup",
            supports_parallel=True,
        )
        self._result = result
        self.calls: list[tuple[Any, Any]] = []

    async def execute(self, request: AgentRequest, runtime: CapabilityRuntimeContext) -> AgentResult:
        # Acceptance gate: capabilities receive AgentRequest +
        # CapabilityRuntimeContext and never supervisor/graph state.
        assert isinstance(request, AgentRequest)
        assert isinstance(runtime, CapabilityRuntimeContext)
        assert not isinstance(runtime, GraphRuntimeContext)
        self.calls.append((request, runtime))
        assert self._result.task_id == request.task_id
        return self._result


class FakeHydrator:
    """Admits every presented use as a supporting, targetless people fact."""

    def __init__(self, content: str = "Nguyễn Văn A là cán bộ.") -> None:
        self._content = content

    async def hydrate_for_evaluation(
        self, use_refs: tuple[EvidenceUseRef, ...], *, runtime: Any, plan: Any, bindings: Any
    ) -> tuple[HydratedEvidence, ...]:
        task_id = plan.tasks[0].task_id if plan.tasks else "t0"
        return tuple(
            HydratedEvidence(
                use_id=ref.use_id,
                evidence_id=EVIDENCE_ID,
                task_id=task_id,
                purpose="supporting",
                target_id=None,
                content=self._content,
                role=None,
                source_label="people-record",
                classification="personal",
                locator=None,
            )
            for ref in use_refs
        )

    async def hydrate_for_synthesis(
        self, use_refs: tuple[EvidenceUseRef, ...], *, runtime: Any, plan: Any, bindings: Any, budget: Any
    ) -> tuple[HydratedEvidence, ...]:
        return await self.hydrate_for_evaluation(
            use_refs, runtime=runtime, plan=plan, bindings=bindings
        )


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def make_request(query: str = "Xin chào") -> RequestContext:
    return RequestContext(
        contract_version="2.0",
        request_id="req-1",
        thread_id="thread-1",
        original_query=query,
        known_documents=(),
    )


def make_conversation() -> ConversationContext:
    return ConversationContext(
        summary="", active_entities=(), last_focus=None, recent_turns=()
    )


def make_runtime_context(
    *,
    allowed: frozenset[str] = FULL_CAPABILITIES,
    semantic_adapter: Any = None,
    binding_resolver: Any = None,
    capability_registry: Any = None,
    evidence_hydrator: Any = None,
    retention_leases: Any = None,
    answer_draft_channel: Any = None,
    deadline: datetime | None = None,
) -> GraphRuntimeContext:
    return GraphRuntimeContext(
        capability_runtime=CapabilityRuntimeContext(
            request_id="req-1",
            run_id="run-1",
            user_id=USER_ID,
            workspace_ids=(WORKSPACE_ID,),
            can_read_people=True,
            allowed_capabilities=allowed,
            deadline_at=deadline or datetime(2030, 1, 1, tzinfo=UTC),
        ),
        services=RuntimeServices(
            retention_leases=retention_leases,
            semantic_adapter=semantic_adapter,
            binding_resolver=binding_resolver,
            capability_registry=capability_registry,
            evidence_hydrator=evidence_hydrator,
            answer_draft_channel=answer_draft_channel,
        ),
    )


def make_state(
    *,
    request: RequestContext | None = None,
    semantic: SemanticContext | None = None,
    bindings: DocumentBindingSet | None = None,
) -> SupervisorV2State:
    return SupervisorV2State(
        contract_version="2.0",
        request=request or make_request(),
        conversation=make_conversation(),
        semantic=semantic
        or SemanticContext(
            contextualized_query="",
            normalized_query="",
            abbreviations=(),
            coreferences=(),
            document_refs=(),
            person_refs=(),
            section_refs=(),
            blocking_ambiguities=(),
        ),
        bindings=bindings or DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        query_analysis=None,
        route_decision=None,
        execution=ExecutionState(plan=None, task_results=(), evidence_evaluation=None),
        clarification=None,
        final_response=None,
    )


def greeting_draft() -> SemanticDraft:
    return SemanticDraft(
        provisional_contextualized_query="xin chào",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=(),
    )


def people_draft() -> SemanticDraft:
    return SemanticDraft(
        provisional_contextualized_query="tìm nguyễn văn a",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(EntityReference(ref_id="p1", kind="person", label="Nguyễn Văn A"),),
        section_refs=(),
        preliminary_ambiguities=(),
    )


def unresolved_ref(ref_id: str = "r1") -> DocumentReference:
    return DocumentReference(
        ref_id=ref_id,
        original_span="Nghị định 12/2020",
        normalized_reference="nghị định 12/2020",
        requested_role="target",
        revision_requirement=None,
        resolution_status="unresolved",
        resolved_document_id=None,
        candidate_document_ids=(),
    )


def resolved_ref(
    ref_id: str = "r1", document_id: UUID = DOCUMENT_ID
) -> DocumentReference:
    return DocumentReference(
        ref_id=ref_id,
        original_span=f"tài liệu {ref_id}",
        normalized_reference=f"tài liệu {ref_id}",
        requested_role="target",
        revision_requirement=None,
        resolution_status="resolved",
        resolved_document_id=document_id,
        candidate_document_ids=(),
    )


def pin_for(ref_id: str, document_id: UUID, revision: UUID) -> ScopedDocument:
    return ScopedDocument(
        binding_id=f"b_{ref_id}",
        document_id=document_id,
        document_revision=str(revision),
        role="target",
    )


def people_success(task_id: str) -> AgentResult:
    return AgentResult(
        contract_version="2.0",
        task_id=task_id,
        status="success",
        data=PeopleLookupOutput(kind="people.lookup", matched=True),
        evidence_uses=(EvidenceUseRef(use_id=USE_ID),),
        coverage_observations=(),
        error=None,
    )


def compile_graph(runtime: GraphRuntimeContext, thread_id: str = "t1"):
    graph = create_supervisor_v2_graph(InMemorySaver())
    return graph, {"configurable": {"thread_id": thread_id}}


def N(values: Any) -> SupervisorV2State:
    """Normalize graph output/checkpoint values (models or raw mappings)."""
    return normalize_checkpoint_state(dict(values))


# ---------------------------------------------------------------------------
# Composition: node inventory and provenance
# ---------------------------------------------------------------------------


def test_graph_contains_all_fourteen_nodes() -> None:
    assert tuple(SUPERVISOR_V2_NODES) == EXPECTED_NODES
    graph = create_supervisor_v2_graph(InMemorySaver())
    node_ids = set(graph.get_graph().nodes)
    for name in EXPECTED_NODES:
        assert name in node_ids, f"missing graph node {name!r}"


def test_graph_nodes_come_from_v2_nodes() -> None:
    origins = {
        name: getattr(fn, "__v2_origin__", fn) for name, fn in SUPERVISOR_V2_NODES.items()
    }
    assert origins["context"] is context_node
    assert origins["binding"] is binding_node
    assert origins["semantic_finalizer"] is semantic_finalizer_node
    assert origins["route"] is route_node
    assert origins["fast_plan"] is fast_plan_node
    assert origins["execute"] is execute_node
    assert origins["evaluate"] is evaluate_node
    assert origins["synthesize"] is synthesize_node
    assert origins["ground"] is ground_node
    assert origins["finalizer"] is finalizer_node
    # Thin boundary functions owned by supervisor_v2 (no domain business).
    for name in ("direct", "clarify", "clarify_wait", "complex_boundary"):
        assert origins[name].__module__ == "app.services.agents.supervisor_v2"


def test_graph_has_no_domain_agent_or_subgraph() -> None:
    for name in SUPERVISOR_V2_NODES:
        assert "agent" not in name
        assert "subgraph" not in name
        assert "domain" not in name
    for name, fn in SUPERVISOR_V2_NODES.items():
        origin = getattr(fn, "__v2_origin__", fn)
        assert origin.__module__.startswith(
            ("app.services.agents.v2.nodes", "app.services.agents.supervisor_v2")
        ), f"node {name!r} comes from {origin.__module__}"
    v2_root = Path(__file__).resolve().parents[3] / "app" / "services" / "agents" / "v2"
    assert v2_root.is_dir()
    banned = [
        "people_agent.py",
        "summary_agent.py",
        "comparison_agent.py",
        "document_agent.py",
        "section_agent.py",
        "kg_agent.py",
    ]
    found = [p.name for p in v2_root.rglob("*.py") if p.name in banned]
    assert found == [], f"domain-agent modules present: {found}"
    domain_graphs = list(v2_root.glob("domain/*_graph.py"))
    assert domain_graphs == []
    # M7: symbol-level guard mirroring the Phase-2 acceptance gate — an
    # innocuously named wrapper module would pass the filename/name checks.
    agent_pattern = re.compile(
        r"class\s+(People|Summary|Comparison|Document|Section|KG|Evaluation|Grounding).*Agent"
        r"|\b(People|Summary|Comparison|Document|Section|KG)Agent\b"
    )
    hits = [
        str(path)
        for path in sorted(v2_root.rglob("*.py"))
        if agent_pattern.search(path.read_text())
    ]
    assert hits == [], f"domain-agent symbols present: {hits}"


# ---------------------------------------------------------------------------
# Direct route end to end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_greeting_end_to_end() -> None:
    adapter = FakeSemanticAdapter(greeting_draft())
    resolver = FakeBindingResolver(
        DocumentBindingSet(bindings=(), revision_requirement_refs=())
    )
    runtime = make_runtime_context(semantic_adapter=adapter, binding_resolver=resolver)
    graph, config = compile_graph(runtime)
    # D3: the draft is rebuilt per node (context/binding/finalizer turns).
    result = await graph.ainvoke(make_state(), config, context=runtime)
    assert adapter.calls, "semantic adapter was never consulted"
    first = adapter.calls[0]
    assert all(call == first for call in adapter.calls), "adapter is not deterministic"
    response = N(result)["final_response"]
    assert response.status == "success"
    assert response.content.strip()
    assert response.citations == ()
    # Direct non-factual success keeps plan=None even with a stale checkpoint.
    assert N(result)["execution"].plan is None
    assert N(result)["execution"].task_results == ()


# ---------------------------------------------------------------------------
# Clarify route: persist + interrupt + validated checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clarify_route_interrupts_with_persisted_request() -> None:
    draft = SemanticDraft(
        provisional_contextualized_query="mở nghị định 12/2020",
        abbreviations=(),
        coreferences=(),
        document_refs=(unresolved_ref(),),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=(),
    )
    runtime = make_runtime_context(
        semantic_adapter=FakeSemanticAdapter(draft),
        binding_resolver=FakeBindingResolver(
            DocumentBindingSet(bindings=(), revision_requirement_refs=())
        ),
    )
    graph, config = compile_graph(runtime, thread_id="clarify-1")
    state = make_state(request=make_request("Mở Nghị định 12/2020"))
    result = await graph.ainvoke(state, config, context=runtime)
    interrupts = result.get("__interrupt__", ())
    assert len(interrupts) == 1
    stored = graph.get_state(config)
    coerced = N(stored.values)
    clarification = coerced["clarification"]
    assert clarification is not None
    assert interrupts[0].value == {"clarification": clarification.model_dump(mode="json")}
    # D6: the clarify checkpoint carries a persisted request and validates.
    validate_supervisor_state(coerced)
    assert coerced["route_decision"].route == "clarify"


# ---------------------------------------------------------------------------
# Fast people path: validated plan checkpointed before any dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_people_checkpoints_plan_before_dispatch() -> None:
    events: list[str] = []
    lease_repo = FakeLeaseRepo(events)
    # Resolve the stable fast task id by running the planner once off-graph.
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan
    from app.services.agents.v2.nodes.context import finalize_semantic
    from app.services.agents.v2.nodes.routing import analyze_query, decide_route

    empty = DocumentBindingSet(bindings=(), revision_requirement_refs=())
    semantic = finalize_semantic(people_draft(), empty)
    analysis = analyze_query(semantic)
    decision = decide_route(
        analysis, semantic, empty,
        allowed_capabilities=FULL_CAPABILITIES,
    )
    assert decision.route == "fast_domain"
    plan = build_fast_plan(semantic, empty, analysis, decision)
    capability = FakeCapability(
        name="people.lookup", domain="people", result=people_success(plan.tasks[0].task_id)
    )
    runtime = make_runtime_context(
        semantic_adapter=FakeSemanticAdapter(people_draft()),
        binding_resolver=FakeBindingResolver(empty),
        evidence_hydrator=FakeHydrator(),
        retention_leases=lease_repo,
        answer_draft_channel=AnswerDraftChannel(),
    )
    registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], runtime.capability_runtime
    )
    runtime.services.capability_registry = registry

    graph, config = compile_graph(runtime, thread_id="fast-1")
    result = await graph.ainvoke(
        make_state(request=make_request("Tìm Nguyễn Văn A")), config, context=runtime
    )
    # Capability received AgentRequest + CapabilityRuntimeContext only.
    assert len(capability.calls) == 1
    request, call_runtime = capability.calls[0]
    assert isinstance(request, AgentRequest)
    assert isinstance(call_runtime, CapabilityRuntimeContext)
    assert call_runtime.run_id == "run-1"
    assert call_runtime.workspace_ids == (WORKSPACE_ID,)

    response = N(result)["final_response"]
    assert response.status == "success"
    assert response.citations, "factual success must carry citations"

    # The fast TaskPlan was checkpointed BEFORE any capability dispatched:
    # the oldest checkpoint holding a plan must hold zero task results.
    history = [entry async for entry in graph.aget_state_history(config)]
    history = [entry for entry in history if "execution" in (entry.values or {})]
    oldest_with_plan = None
    for entry in reversed(history):
        plan_slot = _plan_of(entry.values)
        if plan_slot is not None:
            oldest_with_plan = entry
            break
    assert oldest_with_plan is not None, "no checkpoint ever held the fast plan"
    assert _results_of(oldest_with_plan.values) == ()
    assert any(_results_of(entry.values) != () for entry in history)


def _plan_of(values: Any) -> Any:
    execution = values["execution"]
    plan = execution["plan"] if isinstance(execution, dict) else execution.plan
    if isinstance(plan, dict):
        return None if plan is None else plan
    return plan


def _results_of(values: Any) -> tuple:
    execution = values["execution"]
    results = (
        execution["task_results"] if isinstance(execution, dict) else execution.task_results
    )
    return tuple(results)


@pytest.mark.asyncio
async def test_execute_requires_checkpointed_plan() -> None:
    capability = FakeCapability(
        name="people.lookup",
        domain="people",
        result=people_success("never-dispatched"),
    )
    runtime = make_runtime_context()
    runtime.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], runtime.capability_runtime
    )
    with pytest.raises(MissingCheckpointedPlan):
        await execute_node(make_state(), Runtime(context=runtime))
    assert capability.calls == []


# ---------------------------------------------------------------------------
# Complex route: typed unavailable, no dispatch (Phase 3 owns the seam)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complex_route_returns_typed_unavailable() -> None:
    events: list[str] = []
    draft = SemanticDraft(
        provisional_contextualized_query="so sánh tài liệu một và tài liệu hai",
        abbreviations=(),
        coreferences=(),
        document_refs=(resolved_ref("r1", DOCUMENT_ID), resolved_ref("r2", OTHER_DOCUMENT_ID)),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=(),
    )
    pins = DocumentBindingSet(
        bindings=(
            pin_for("r1", DOCUMENT_ID, REVISION_ID),
            pin_for("r2", OTHER_DOCUMENT_ID, OTHER_REVISION_ID),
        ),
        revision_requirement_refs=(),
    )
    capability = FakeCapability(
        name="people.lookup",
        domain="people",
        result=people_success("never-dispatched"),
    )
    runtime = make_runtime_context(
        semantic_adapter=FakeSemanticAdapter(draft),
        binding_resolver=FakeBindingResolver(pins),
        retention_leases=FakeLeaseRepo(events),
    )
    runtime.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], runtime.capability_runtime
    )
    graph, config = compile_graph(runtime, thread_id="complex-1")
    result = await graph.ainvoke(
        make_state(request=make_request("So sánh hai tài liệu")), config, context=runtime
    )
    assert N(result)["route_decision"].route == "complex_research"
    assert N(result)["final_response"].status in ("denied", "error")
    assert N(result)["final_response"].status != "success"
    assert capability.calls == [], "complex routes must not dispatch Phase-2 capabilities"
    assert any(event.startswith("acquire:") for event in events)


# ---------------------------------------------------------------------------
# T5 D-1: checkpoint normalization across a real round-trip
# ---------------------------------------------------------------------------


def _json_round_trip(values: dict) -> dict:
    def default(value: Any) -> Any:
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, frozenset):
            return sorted(str(item) for item in value)
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        raise TypeError(f"not checkpoint-serializable: {type(value).__name__}")

    return json.loads(json.dumps(values, default=default))


@pytest.mark.asyncio
async def test_checkpoint_round_trip_normalizes_nested_contracts() -> None:
    events: list[str] = []
    lease_repo = FakeLeaseRepo(events)
    runtime = make_runtime_context(
        semantic_adapter=FakeSemanticAdapter(people_draft()),
        binding_resolver=FakeBindingResolver(
            DocumentBindingSet(bindings=(), revision_requirement_refs=())
        ),
        evidence_hydrator=FakeHydrator(),
        retention_leases=lease_repo,
        answer_draft_channel=AnswerDraftChannel(),
    )
    from app.services.agents.v2.nodes.context import finalize_semantic
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan
    from app.services.agents.v2.nodes.routing import analyze_query, decide_route

    empty = DocumentBindingSet(bindings=(), revision_requirement_refs=())
    semantic = finalize_semantic(people_draft(), empty)
    analysis = analyze_query(semantic)
    decision = decide_route(
        analysis, semantic, empty,
        allowed_capabilities=runtime.capability_runtime.allowed_capabilities,
    )
    plan = build_fast_plan(semantic, empty, analysis, decision)
    capability = FakeCapability(
        name="people.lookup", domain="people", result=people_success(plan.tasks[0].task_id)
    )
    runtime.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], runtime.capability_runtime
    )
    graph, config = compile_graph(runtime, thread_id="roundtrip-1")
    await graph.ainvoke(
        make_state(request=make_request("Tìm Nguyễn Văn A")), config, context=runtime
    )
    stored = graph.get_state(config)
    raw = _json_round_trip(dict(stored.values))
    # Real checkpoint serde delivers plain mappings/lists, not models.
    assert isinstance(raw["semantic"], dict)
    assert isinstance(raw["execution"], dict)
    coerced = normalize_checkpoint_state(raw)
    assert isinstance(coerced["semantic"], SemanticContext)
    assert isinstance(coerced["bindings"], DocumentBindingSet)
    assert isinstance(coerced["query_analysis"].domains, tuple)
    assert coerced["execution"].plan is not None
    assert isinstance(coerced["execution"].task_results[0], AgentResult)
    assert coerced["execution"].evidence_evaluation is not None
    validate_supervisor_state(coerced)


@pytest.mark.asyncio
async def test_second_turn_over_json_checkpoint_state() -> None:
    adapter = FakeSemanticAdapter(greeting_draft())
    runtime = make_runtime_context(
        semantic_adapter=adapter,
        binding_resolver=FakeBindingResolver(
            DocumentBindingSet(bindings=(), revision_requirement_refs=())
        ),
    )
    graph, config = compile_graph(runtime, thread_id="turn-2")
    first = N(
        await graph.ainvoke(make_state(), config, context=runtime)
    )
    assert first["final_response"].status == "success"
    stored = graph.get_state(config)
    resumed = _json_round_trip(dict(stored.values))
    resumed["request"] = make_request("Chào bạn").model_dump(mode="json")
    second = N(await graph.ainvoke(resumed, config, context=runtime))
    assert second["final_response"].status == "success"
    assert second["execution"].plan is None


# ---------------------------------------------------------------------------
# T3-N2: DispatchReport.truncated stays runtime-only (no ExecutionState slot)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_truncation_stays_runtime_only() -> None:
    from app.services.agents.v2.execution import TaskScheduler
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan
    from app.services.agents.v2.nodes.context import finalize_semantic
    from app.services.agents.v2.nodes.routing import analyze_query, decide_route

    empty = DocumentBindingSet(bindings=(), revision_requirement_refs=())
    semantic = finalize_semantic(people_draft(), empty)
    analysis = analyze_query(semantic)
    decision = decide_route(
        analysis, semantic, empty, allowed_capabilities=FULL_CAPABILITIES
    )
    plan = build_fast_plan(semantic, empty, analysis, decision)
    capability = FakeCapability(
        name="people.lookup", domain="people", result=people_success(plan.tasks[0].task_id)
    )
    expired = make_runtime_context(deadline=datetime(2000, 1, 1, tzinfo=UTC))
    expired.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], expired.capability_runtime
    )
    report = await TaskScheduler(expired.services.capability_registry).execute(
        plan=plan, runtime=expired, prior_results=(), bindings=empty
    )
    assert report.truncated is True
    assert report.results == ()
    assert capability.calls == []

    # execute_node persists results only; the truncation flag never enters
    # checkpoint state (frozen ExecutionState has no slot for it).
    live = make_runtime_context(retention_leases=FakeLeaseRepo([]))
    live.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)], live.capability_runtime
    )
    base = make_state()
    state = SupervisorV2State(
        **{
            **base,
            "execution": ExecutionState(
                plan=plan, task_results=(), evidence_evaluation=None
            ),
        }
    )
    update = await execute_node(state, Runtime(context=live))
    assert set(update) == {"execution"}
    assert "truncated" not in json.dumps(update["execution"].model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Frozen transport boundaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_construction_gates_and_dispatches() -> None:
    from app.services.agents.supervisor_v2 import (
        V1PeopleLookupService,
        V1ServiceBundle,
        build_v2_capability_registry,
    )
    from app.services.agents.v2.capabilities import CapabilityUnavailable
    from app.services.agents.v2.contracts.capability import (
        AbbreviationResolveInput,
        PeopleLookupInput,
    )

    class FakeEvidence:
        async def persist_use(self, **kwargs: Any) -> EvidenceUseRef:
            return EvidenceUseRef(use_id=uuid4())

    capability_runtime = make_runtime_context().capability_runtime
    # Nothing available: every capability is gated out (typed unavailable).
    gated = build_v2_capability_registry(
        capability_runtime,
        bundle=V1ServiceBundle(),
        available_services=frozenset(),
    )
    with pytest.raises(CapabilityUnavailable):
        gated.get("people.lookup")
    with pytest.raises(CapabilityUnavailable):
        gated.get("document.read")

    bundle = V1ServiceBundle(
        people_lookup=V1PeopleLookupService(
            lookup=lambda query, limit=10: {"name": "Nguyễn Văn A", "record_id": "p-1"}
        ),
        abbreviation_lookup=lambda token: "Nghị định" if token.lower() == "nđ" else None,
        evidence=FakeEvidence(),
    )
    registry = build_v2_capability_registry(
        capability_runtime,
        bundle=bundle,
        available_services=frozenset({"v1-people", "v1-abbreviation"}),
    )
    people = registry.get("people.lookup")
    result = await people.execute(
        AgentRequest(
            contract_version="2.0",
            task_id="t1",
            objective="o",
            input=PeopleLookupInput(kind="people.lookup", query="Tìm Nguyễn Văn A"),
        ),
        capability_runtime,
    )
    assert result.status == "success"
    assert result.data.matched is True
    abbreviation = registry.get("abbreviation.resolve")
    abbr_result = await abbreviation.execute(
        AgentRequest(
            contract_version="2.0",
            task_id="t2",
            objective="o",
            input=AbbreviationResolveInput(kind="abbreviation.resolve", tokens=("nđ",)),
        ),
        capability_runtime,
    )
    assert abbr_result.status == "success"
    # Reads stay gated without their persistence seams (no silent fallback).
    with pytest.raises(CapabilityUnavailable):
        registry.get("document.read")


@pytest.mark.asyncio
async def test_clarify_misuse_clears_clarify_route() -> None:
    # Review minor 3: a runner that clears the persisted request (the old
    # Proof-4 misuse) gets a terminal typed error on a D6-valid checkpoint
    # — route retired with the request — never a raise, never a stranding.
    from langgraph.types import Command

    shared: dict = {"tid": "resume-misuse"}
    graph, runtime, _ = await _suspend_ambiguous(shared)
    config = shared["config"]
    result = await graph.ainvoke(
        Command(
            resume={
                "contract_version": "2.0",
                "clarification_id": "x",
                "selected_candidate_id": None,
            },
            update={"clarification": None},
        ),
        config,
        context=runtime,
    )
    assert not result.get("__interrupt__", ())
    resumed = normalize_checkpoint_state(dict(result))
    assert resumed["clarification"] is None
    assert resumed["route_decision"] is None
    assert resumed["final_response"] is not None
    assert resumed["final_response"].status == "error"
    validate_supervisor_state(resumed)
    stored = graph.get_state(config)
    assert stored.next == ()


@pytest.mark.asyncio
async def test_context_node_error_converts_to_typed_error() -> None:
    # NEW-C1: a resolved ref with no pin (outer-goto stale-read shape)
    # converts at the owned boundary instead of escaping and poisoning.
    adapter_draft = SemanticDraft(
        provisional_contextualized_query="mở nghị định 12",
        abbreviations=(),
        coreferences=(),
        document_refs=(resolved_ref("r1", DOCUMENT_ID),),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=(),
    )
    runtime = make_runtime_context(
        semantic_adapter=FakeSemanticAdapter(adapter_draft),
        binding_resolver=FakeBindingResolver(
            DocumentBindingSet(bindings=(), revision_requirement_refs=())
        ),
    )
    graph, config = compile_graph(runtime, thread_id="ctx-err-1")
    result = await graph.ainvoke(
        make_state(request=make_request("Mở Nghị định 12")), config, context=runtime
    )
    resumed = normalize_checkpoint_state(dict(result))
    assert resumed["final_response"] is not None
    assert resumed["final_response"].status == "error"
    # Pre-finalization failure: semantics were never finalized (blank
    # placeholder), so the frozen aggregate check cannot pass by
    # construction — the guarantee here is typed-error terminal, no escape,
    # no poisoned thread.
    stored = graph.get_state(config)
    assert stored.next == ()
    # The thread is not poisoned: continuing ends cleanly on the same error.
    again = await graph.ainvoke(None, config, context=runtime)
    assert normalize_checkpoint_state(dict(again))["final_response"].status == "error"


@pytest.mark.asyncio
async def test_scheduler_error_converts_to_typed_error() -> None:
    # NEW-C1/NEW-I1: execute with no registry wired converts to a STICKY
    # typed error (never masked downstream) and dispatches nothing.
    runtime = make_runtime_context(
        semantic_adapter=FakeSemanticAdapter(people_draft()),
        binding_resolver=FakeBindingResolver(
            DocumentBindingSet(bindings=(), revision_requirement_refs=())
        ),
        evidence_hydrator=FakeHydrator(),
        retention_leases=FakeLeaseRepo([]),
        answer_draft_channel=AnswerDraftChannel(),
        capability_registry=None,
    )
    graph, config = compile_graph(runtime, thread_id="sched-err-1")
    result = await graph.ainvoke(
        make_state(request=make_request("Tìm Nguyễn Văn A")), config, context=runtime
    )
    resumed = normalize_checkpoint_state(dict(result))
    assert resumed["final_response"] is not None
    assert resumed["final_response"].status == "error"
    validate_supervisor_state(resumed)
    stored = graph.get_state(config)
    assert stored.next == ()
    # Zero dispatches: no checkpoint ever holds a task result.
    history = [entry async for entry in graph.aget_state_history(config)]
    assert history, "expected owned checkpoints"
    for entry in history:
        values = entry.values or {}
        execution = values.get("execution")
        results = (
            execution.get("task_results", ())
            if isinstance(execution, dict)
            else getattr(execution, "task_results", ())
        )
        assert tuple(results) == ()


@pytest.mark.asyncio
async def test_binding_failure_is_sticky_no_dispatch() -> None:
    # NEW-I1 (review P8/P9a): a fail-closed binding/lease failure ends the
    # turn in typed error with ZERO capability dispatches — the marker is
    # never overwritten downstream and the thread stays continuable.
    events: list[str] = []
    draft = SemanticDraft(
        provisional_contextualized_query="so sánh tài liệu một và tài liệu hai",
        abbreviations=(),
        coreferences=(),
        document_refs=(resolved_ref("r1", DOCUMENT_ID), resolved_ref("r2", OTHER_DOCUMENT_ID)),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=(),
    )

    class _FailingResolver:
        async def resolve(self, document_refs: Any, capability_runtime: Any) -> Any:
            from app.services.agents.v2.nodes.binding import BindingNodeError

            raise BindingNodeError("lease backend unreachable")

    capability = FakeCapability(
        name="people.lookup",
        domain="people",
        result=people_success("never-dispatched"),
    )
    runtime = make_runtime_context(
        semantic_adapter=FakeSemanticAdapter(draft),
        binding_resolver=_FailingResolver(),
        retention_leases=FakeLeaseRepo(events),
        answer_draft_channel=AnswerDraftChannel(),
    )
    runtime.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)],
        runtime.capability_runtime,
    )
    graph, config = compile_graph(runtime, thread_id="bind-fail-1")
    result = await graph.ainvoke(
        make_state(request=make_request("So sánh hai tài liệu")), config, context=runtime
    )
    resumed = normalize_checkpoint_state(dict(result))
    assert resumed["final_response"] is not None
    assert resumed["final_response"].status == "error"
    assert capability.calls == []
    stored = graph.get_state(config)
    assert stored.next == ()
    again = await graph.ainvoke(None, config, context=runtime)
    assert normalize_checkpoint_state(dict(again))["final_response"].status == "error"


@pytest.mark.asyncio
async def test_lease_backend_failure_is_sticky_no_dispatch() -> None:
    # NEW-I1 (review P8): resolver pins, but the lease write fails — the
    # pin must not be checkpointed and nothing may dispatch afterwards.
    events: list[str] = []
    draft = SemanticDraft(
        provisional_contextualized_query="so sánh tài liệu một và tài liệu hai",
        abbreviations=(),
        coreferences=(),
        document_refs=(resolved_ref("r1", DOCUMENT_ID), resolved_ref("r2", OTHER_DOCUMENT_ID)),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=(),
    )
    pins = DocumentBindingSet(
        bindings=(
            pin_for("r1", DOCUMENT_ID, REVISION_ID),
            pin_for("r2", OTHER_DOCUMENT_ID, OTHER_REVISION_ID),
        ),
        revision_requirement_refs=(),
    )

    class _FailingLeaseRepo(FakeLeaseRepo):
        async def acquire_or_refresh(
            self, run_id: str, revision_id: Any, evidence_use_id: Any = None, **kwargs: Any
        ) -> Any:
            self.events.append(f"acquire:{revision_id}:{evidence_use_id}")
            raise RuntimeError("lease backend down")

    capability = FakeCapability(
        name="people.lookup",
        domain="people",
        result=people_success("never-dispatched"),
    )
    runtime = make_runtime_context(
        semantic_adapter=FakeSemanticAdapter(draft),
        binding_resolver=FakeBindingResolver(pins),
        retention_leases=_FailingLeaseRepo(events),
        answer_draft_channel=AnswerDraftChannel(),
    )
    runtime.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)],
        runtime.capability_runtime,
    )
    graph, config = compile_graph(runtime, thread_id="lease-fail-1")
    result = await graph.ainvoke(
        make_state(request=make_request("So sánh hai tài liệu")), config, context=runtime
    )
    resumed = normalize_checkpoint_state(dict(result))
    assert resumed["final_response"] is not None
    assert resumed["final_response"].status == "error"
    assert capability.calls == []
    # The failed pin never reached checkpoint state.
    assert resumed["bindings"].bindings == ()
    stored = graph.get_state(config)
    assert stored.next == ()
    again = await graph.ainvoke(None, config, context=runtime)
    assert normalize_checkpoint_state(dict(again))["final_response"].status == "error"


def test_model_input_cannot_supply_workspace_or_acl() -> None:
    from app.services.agents.v2.contracts.capability import PeopleLookupInput

    with pytest.raises(ValidationError):
        AgentRequest(
            contract_version="2.0",
            task_id="t1",
            objective="o",
            input=PeopleLookupInput(kind="people.lookup", query="q"),
            workspace_ids=(WORKSPACE_ID,),  # type: ignore[call-arg]
        )
    assert "workspace_ids" not in AgentRequest.model_fields
    assert "allowed_capabilities" not in AgentRequest.model_fields


def test_invalid_aggregate_is_rejected() -> None:
    from app.services.agents.v2.contracts.routing import RouteDecision

    greeting = SemanticContext(
        contextualized_query="xin chào",
        normalized_query="xin chào",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    base = make_state(semantic=greeting)
    # A settled non-clarify aggregate validates.
    validate_supervisor_state(base)
    # Clarify route without a persisted request must never be written.
    broken_clarify = SupervisorV2State(
        **{
            **base,
            "route_decision": RouteDecision(
                route="clarify", reason_code="essential_ambiguity"
            ),
            "clarification": None,
        }
    )
    with pytest.raises(Exception):
        validate_supervisor_state(broken_clarify)


# ---------------------------------------------------------------------------
# v2/events.py: SSE helpers for the outer (T8) adapter
# ---------------------------------------------------------------------------


def test_format_sse_matches_v1_wire_format() -> None:
    assert format_sse_event("token", {"text": "x"}) == 'event: token\ndata: {"text": "x"}\n\n'
    assert format_sse_event(SSE_COMPLETE, {"answer": "ok"}).startswith("event: complete\n")


def test_terminal_event_mapping() -> None:
    from app.services.agents.v2.contracts.response import FinalResponse

    success = FinalResponse(
        contract_version=CONTRACT_VERSION, status="success", content="answer", citations=()
    )
    event, data = terminal_event_for_response(success)
    assert event == SSE_COMPLETE
    assert data["answer"] == "answer"
    payload = final_response_payload(success)
    assert payload["status"] == "success"

    denied = FinalResponse(
        contract_version=CONTRACT_VERSION, status="denied", content="no", citations=()
    )
    event, _ = terminal_event_for_response(denied)
    assert event == SSE_ERROR

    # M2: clarify is a terminal answer to its turn (the question), so it
    # streams as a complete event carrying the clarify status.
    clarify = FinalResponse(
        contract_version=CONTRACT_VERSION,
        status="clarify",
        content="which document?",
        citations=(),
    )
    event, data = terminal_event_for_response(clarify)
    assert event == SSE_COMPLETE
    assert data["status"] == "clarify"


def test_events_payload_is_additive_over_v1() -> None:
    # M1: wire-identical, payload-additive. Every v1 complete key survives;
    # status/citations are documented extras T8 must tolerate, not identity.
    from app.services.agents.v2.contracts.response import FinalResponse

    payload = final_response_payload(
        FinalResponse(
            contract_version=CONTRACT_VERSION,
            status="success",
            content="answer",
            citations=(),
        )
    )
    for key in (
        "answer",
        "sources",
        "images",
        "potential_abbreviations",
        "people_data",
    ):
        assert key in payload, f"v1 complete key missing: {key}"
    assert payload["answer"] == "answer"


def test_probe_excludes_unwired_gates() -> None:
    # M6: the probe opens only gates whose adapter default is complete
    # without per-request scope. Scoped/unwired backings stay closed until
    # T7 enables them explicitly via available_services.
    from app.services.agents.supervisor_v2 import probe_v1_services

    probed = probe_v1_services()
    assert "v1-abbreviation" not in probed
    assert "v1-document-content" not in probed
    assert "v1-section-content" not in probed
    assert "v1-knowledge-graph" not in probed
    assert "v1-memory" not in probed


@pytest.mark.asyncio
async def test_dedicated_retention_leases_owns_its_session() -> None:
    # I3/M4(a): the lease repo must own a dedicated unit of work — the
    # helper opens a session used for nothing else and closes it on exit.
    from app.services.agents.supervisor_v2 import dedicated_retention_leases

    closed: list[str] = []

    class FakeLeaseSession:
        async def close(self) -> None:
            closed.append("close")

    made: list[str] = []

    def factory() -> FakeLeaseSession:
        made.append("open")
        return FakeLeaseSession()

    async with dedicated_retention_leases(factory) as repo:  # type: ignore[arg-type]
        assert made == ["open"]
        assert repo.session is not None
        assert closed == []
    assert closed == ["close"]


def test_undispatched_tasks_recipe() -> None:
    # I1: the T8 terminal-boundary recipe for detecting a truncated
    # dispatch whose DispatchReport.truncated flag was never checkpointed.
    from app.services.agents.supervisor_v2 import undispatched_tasks
    from app.services.agents.v2.nodes.context import finalize_semantic
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan
    from app.services.agents.v2.nodes.routing import analyze_query, decide_route

    empty = DocumentBindingSet(bindings=(), revision_requirement_refs=())
    semantic = finalize_semantic(people_draft(), empty)
    analysis = analyze_query(semantic)
    decision = decide_route(
        analysis, semantic, empty, allowed_capabilities=FULL_CAPABILITIES
    )
    plan = build_fast_plan(semantic, empty, analysis, decision)
    (task_id,) = [task.task_id for task in plan.tasks]
    assert undispatched_tasks(plan, ()) == (task_id,)
    assert undispatched_tasks(plan, (people_success(task_id),)) == ()


def test_direct_node_clears_stale_plan() -> None:
    # M3: the commented behavior, now exercised — a stale factual plan
    # from a prior turn never rides alongside a direct success.
    import asyncio

    from app.services.agents import supervisor_v2 as sv2
    from app.services.agents.v2.contracts.routing import RouteDecision
    from app.services.agents.v2.nodes.context import finalize_semantic
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan
    from app.services.agents.v2.nodes.routing import analyze_query, decide_route

    empty = DocumentBindingSet(bindings=(), revision_requirement_refs=())
    semantic = finalize_semantic(people_draft(), empty)
    analysis = analyze_query(semantic)
    decision = decide_route(
        analysis, semantic, empty, allowed_capabilities=FULL_CAPABILITIES
    )
    stale_plan = build_fast_plan(semantic, empty, analysis, decision)
    greeting = SemanticContext(
        contextualized_query="xin chào",
        normalized_query="xin chào",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    state = SupervisorV2State(
        **{
            **make_state(semantic=greeting),
            "route_decision": RouteDecision(
                route="direct", reason_code="direct_greeting"
            ),
            "execution": ExecutionState(
                plan=stale_plan, task_results=(), evidence_evaluation=None
            ),
        }
    )
    update = asyncio.run(
        sv2.SUPERVISOR_V2_NODES["direct"](
            state, Runtime(context=make_runtime_context())
        )
    )
    assert update["execution"].plan is None
    validate_supervisor_state({**normalize_checkpoint_state(dict(state)), **update})


def test_stale_clarification_retires_before_direct() -> None:
    # C1 Proofs 2/3: a clarification orphaned by a turn change retires at
    # the owned boundary instead of crashing it. The suspended request
    # named r1; the new turn's projection has no r1 — direct proceeds with
    # clarification=None and a valid aggregate.
    import asyncio

    from app.services.agents import supervisor_v2 as sv2
    from app.services.agents.v2.contracts.routing import RouteDecision
    from app.services.agents.v2.nodes.clarification import build_clarification

    old_semantic = SemanticContext(
        contextualized_query="mở nghị định 12/2020",
        normalized_query="mở nghị định 12/2020",
        abbreviations=(),
        coreferences=(),
        document_refs=(unresolved_ref(),),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    stale_request = build_clarification(old_semantic)
    greeting = SemanticContext(
        contextualized_query="cảm ơn",
        normalized_query="cảm ơn",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    state = SupervisorV2State(
        **{
            **make_state(semantic=greeting),
            "route_decision": RouteDecision(
                route="direct", reason_code="direct_conversation"
            ),
            "clarification": stale_request,
        }
    )
    update = asyncio.run(
        sv2.SUPERVISOR_V2_NODES["direct"](
            state, Runtime(context=make_runtime_context())
        )
    )
    assert update["clarification"] is None
    assert update["execution"].plan is None
    validate_supervisor_state({**normalize_checkpoint_state(dict(state)), **update})


# ---------------------------------------------------------------------------
# C1: clarify selection-resume advances (T5 contract, graph level)
# ---------------------------------------------------------------------------


def _ambiguous_preprocessing(doc_a: UUID, doc_b: UUID):
    from app.services.agents.semantic_preprocessor import (
        DocumentCandidate as LegacyCandidate,
        DocumentRefEntry,
        PreprocessingResult,
    )

    return PreprocessingResult(
        original_query="Mở Nghị định 12",
        normalized_query="mở nghị định 12",
        abbreviations=[],
        document_refs=[
            DocumentRefEntry(
                ref_id="r1",
                original_span="Nghị định 12",
                span_offset=(3, 15),
                reference="nghị định 12",
                candidates=[
                    LegacyCandidate(
                        document_id=doc_a,
                        match_basis="exact_number",
                        confidence=0.9,
                    ),
                    LegacyCandidate(
                        document_id=doc_b,
                        match_basis="exact_number",
                        confidence=0.8,
                    ),
                ],
                resolution_status="ambiguous",
            )
        ],
        blocking_ambiguities=[],
        preprocessing_status="ok",
        preprocessor_trace=[],
    )


class _EchoBindingResolver:
    """Pins whatever resolved refs it is given (selection-aware stand-in)."""

    def __init__(self, revision: UUID) -> None:
        self._revision = revision
        self.calls: list[tuple[Any, CapabilityRuntimeContext]] = []

    async def resolve(
        self, document_refs: Any, capability_runtime: CapabilityRuntimeContext
    ) -> DocumentBindingSet:
        self.calls.append((tuple(document_refs), capability_runtime))
        pins = tuple(
            ScopedDocument(
                binding_id=f"b_{ref.ref_id}",
                document_id=ref.resolved_document_id,
                document_revision=str(self._revision),
                role="target",
            )
            for ref in document_refs
            if ref.resolution_status == "resolved"
            and ref.resolved_document_id is not None
        )
        return DocumentBindingSet(bindings=pins, revision_requirement_refs=())


class _FakeChatMessages:
    def __init__(self, contents: dict[UUID, str]) -> None:
        self.contents = dict(contents)

    async def get_user_message(self, message_id: UUID) -> SimpleNamespace:
        return SimpleNamespace(id=message_id, content=self.contents[message_id])


class _FakeAuthorization:
    def __init__(self, grants: dict[UUID, set[UUID]] | None = None) -> None:
        self.grants = grants
        self.calls: list[tuple[UUID, CapabilityRuntimeContext]] = []

    async def require_document(
        self, document_id: UUID, capability_runtime: CapabilityRuntimeContext
    ) -> None:
        self.calls.append((document_id, capability_runtime))
        if self.grants is not None:
            allowed = any(
                document_id in self.grants.get(workspace, set())
                for workspace in capability_runtime.workspace_ids
            )
            if not allowed:
                raise PermissionError(f"document {document_id} is not authorized")


def _doc_success(task_id: str, target_id: str, use_id: UUID) -> AgentResult:
    from app.services.agents.v2.contracts.capability import DocumentReadOutput
    from app.services.agents.v2.contracts.evaluation import CoverageObservation
    from app.services.agents.v2.contracts.locators import DocumentLocator

    return AgentResult(
        contract_version="2.0",
        task_id=task_id,
        status="success",
        data=DocumentReadOutput(kind="document.read", read_unit_count=1),
        evidence_uses=(EvidenceUseRef(use_id=use_id),),
        coverage_observations=(
            CoverageObservation(
                target_id=target_id,
                observed_locators=(DocumentLocator(kind="document"),),
                outcome="read",
            ),
        ),
        error=None,
    )


async def _suspend_ambiguous(shared: dict) -> tuple[Any, Any, UUID]:
    """Turn 1 with the REAL adapter: ambiguous r1 suspends with 2 options."""
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    doc_a = uuid4()
    doc_b = uuid4()
    revision = uuid4()
    adapter = DeterministicSemanticAdapter(
        preprocess=lambda query: _ambiguous_preprocessing(doc_a, doc_b)
    )
    runtime = make_runtime_context(
        semantic_adapter=adapter,
        binding_resolver=_EchoBindingResolver(revision),
        evidence_hydrator=FakeHydrator(),
        retention_leases=FakeLeaseRepo(shared.setdefault("events", [])),
        answer_draft_channel=AnswerDraftChannel(),
    )
    graph, config = compile_graph(runtime, thread_id=shared.setdefault("tid", "resume-1"))
    result = await graph.ainvoke(
        make_state(request=make_request("Mở Nghị định 12")), config, context=runtime
    )
    assert len(result.get("__interrupt__", ())) == 1
    stored = graph.get_state(config)
    suspended = normalize_checkpoint_state(dict(stored.values))
    assert suspended["clarification"] is not None
    assert len(suspended["clarification"].candidates) == 2
    validate_supervisor_state(suspended)
    shared.update(
        graph=graph, config=config, runtime=runtime,
        suspended=suspended, revision=revision,
    )
    return graph, runtime, suspended["clarification"].candidates[0].document_id


@pytest.mark.asyncio
async def test_clarify_selection_resume_advances_to_binding() -> None:
    from langgraph.types import Command

    from app.services.agents.v2.nodes.clarification import resume_clarification

    shared: dict = {}
    graph, runtime, _ = await _suspend_ambiguous(shared)
    suspended = shared["suspended"]
    revision = shared["revision"]
    config = shared["config"]
    selected_doc = suspended["clarification"].candidates[0].document_id

    history_before = [entry async for entry in graph.aget_state_history(config)]
    message_id = uuid4()
    runtime.services.chat_messages = _FakeChatMessages({message_id: "1"})
    runtime.services.authorization = _FakeAuthorization(
        grants={WORKSPACE_ID: {c.document_id for c in suspended["clarification"].candidates}}
    )
    # The resumed turn dispatches document.read through the shared registry.
    from app.services.agents.v2.nodes.context import finalize_semantic
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan
    from app.services.agents.v2.nodes.routing import analyze_query, decide_route

    resolved_ref = unresolved_ref().model_copy(
        update={
            "resolution_status": "resolved",
            "resolved_document_id": selected_doc,
            "candidate_document_ids": (),
        }
    )
    doc_semantic = SemanticContext(
        contextualized_query="mở nghị định 12",
        normalized_query="mở nghị định 12",
        abbreviations=(),
        coreferences=(),
        document_refs=(resolved_ref,),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )
    doc_bindings = DocumentBindingSet(
        bindings=(pin_for("r1", selected_doc, revision),),
        revision_requirement_refs=(),
    )
    analysis = analyze_query(doc_semantic)
    decision = decide_route(
        analysis, doc_semantic, doc_bindings,
        allowed_capabilities=FULL_CAPABILITIES,
    )
    assert decision.route == "fast_domain"
    plan = build_fast_plan(doc_semantic, doc_bindings, analysis, decision)
    (doc_task_id,) = [task.task_id for task in plan.tasks]
    capability = FakeCapability(
        name="document.read",
        domain="document",
        result=_doc_success(doc_task_id, "t_b_r1", USE_ID),
    )
    runtime.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=capability)],
        runtime.capability_runtime,
    )

    command = await resume_clarification(message_id, suspended["clarification"], runtime)
    # Verbatim T5 artifact (no outer goto — the graph owns navigation).
    assert not command.goto
    result = await graph.ainvoke(command, config, context=runtime)
    assert not result.get("__interrupt__", ()), "resume must not re-suspend"
    resumed = normalize_checkpoint_state(dict(result))
    # The route ADVANCES: no re-ask, request retired, selection recorded.
    assert resumed["route_decision"].route == "fast_domain"
    assert resumed["clarification"] is None
    selections = [
        known
        for known in resumed["request"].known_documents
        if known.source == "ui_selection"
    ]
    assert len(selections) == 1
    assert selections[0].resource_id == "r1"
    assert selections[0].document_id == selected_doc
    pins = {binding.binding_id: binding for binding in resumed["bindings"].bindings}
    assert pins["b_r1"].document_id == selected_doc
    assert len(capability.calls) == 1
    assert resumed["final_response"] is not None
    assert resumed["final_response"].status == "insufficient"
    validate_supervisor_state(resumed)
    # The thread is clean and terminal (re-review minor 3): no re-suspend
    # above, next == () here, and a continue re-ends on the same terminal.
    stored = graph.get_state(config)
    assert stored.next == ()
    again = await graph.ainvoke(None, config, context=runtime)
    assert normalize_checkpoint_state(dict(again))["final_response"].status == "insufficient"
    # EVERY post-resume owned checkpoint validates (no silent invalid write).
    history_after = [entry async for entry in graph.aget_state_history(config)]
    new_entries = history_after[: len(history_after) - len(history_before)]
    assert new_entries, "resume must checkpoint owned boundaries"
    for entry in new_entries:
        values = entry.values or {}
        if "execution" not in values:
            continue
        validate_supervisor_state(normalize_checkpoint_state(dict(values)))
    # The selection pin was leased before it could be checkpointed.
    assert any(str(revision) in event for event in shared["events"])


@pytest.mark.asyncio
async def test_clarify_forged_selection_converts_to_typed_error() -> None:
    from langgraph.types import Command

    shared: dict = {"tid": "resume-forged"}
    graph, runtime, _ = await _suspend_ambiguous(shared)
    config = shared["config"]
    runtime.services.authorization = _FakeAuthorization(grants={WORKSPACE_ID: set()})
    forged = {
        "contract_version": "2.0",
        "clarification_id": shared["suspended"]["clarification"].clarification_id,
        "selected_candidate_id": "not-an-offered-candidate",
    }
    # Plain resume (no outer goto exists anymore): the node's own
    # navigation drives the conversion to the finalizer for a terminal
    # typed error.
    result = await graph.ainvoke(Command(resume=forged), config, context=runtime)
    assert not result.get("__interrupt__", ())
    resumed = normalize_checkpoint_state(dict(result))
    assert resumed["clarification"] is None
    assert resumed["final_response"] is not None
    assert resumed["final_response"].status == "error"


@pytest.mark.asyncio
async def test_clarify_denied_selection_converts_to_typed_denied() -> None:
    from langgraph.types import Command

    from app.services.agents.v2.nodes.clarification import resume_clarification

    shared: dict = {"tid": "resume-denied"}
    graph, runtime, _ = await _suspend_ambiguous(shared)
    suspended = shared["suspended"]
    config = shared["config"]
    message_id = uuid4()
    runtime.services.chat_messages = _FakeChatMessages({message_id: "1"})
    # Mint the command while authorized, then revoke: the CURRENT
    # (resume-time) ACL — not the mint-time one — must decide.
    runtime.services.authorization = _FakeAuthorization(
        grants={WORKSPACE_ID: {c.document_id for c in suspended["clarification"].candidates}}
    )
    command = await resume_clarification(message_id, suspended["clarification"], runtime)
    runtime.services.authorization = _FakeAuthorization(grants={WORKSPACE_ID: set()})
    # Verbatim T5 artifact: the node's own navigation drives the conversion
    # to the finalizer for a terminal typed denial.
    result = await graph.ainvoke(command, config, context=runtime)
    resumed = normalize_checkpoint_state(dict(result))
    assert resumed["clarification"] is None
    assert resumed["final_response"] is not None
    assert resumed["final_response"].status == "denied"

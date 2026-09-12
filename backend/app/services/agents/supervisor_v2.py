"""Node-based supervisor v2: composition, adapters, and lifecycle (Phase 2, Task 6).

Composes the T1–T5 node functions into one ``StateGraph``. v1 stays the
production default; this module changes no v1 behavior and builds no graph
at import time (the lifespan owns the single opened saver, T7 owns
per-request ingress).

Global constraints honored here:

- Only nodes and atomic capabilities are created — no domain agents, no
  domain subgraphs (pinned by ``test_graph_has_no_domain_agent_or_subgraph``).
- Every factual execution owns a validated checkpointed ``TaskPlan`` before
  capability dispatch: ``fast_plan_node`` checkpoints the plan and
  ``execute_node`` fails closed without one; the only dispatcher is the
  shared ``TaskScheduler`` (owned by T3, reused untouched).
- Capabilities receive ``AgentRequest`` + ``CapabilityRuntimeContext`` and
  never supervisor/graph state.
- Frozen contract types are imported, never redefined. ``RuntimeServices``
  has exactly ONE definition (``contracts/state.py``); this module only
  constructs it via ``build_runtime_services`` for T7's per-request ingress.

Checkpoint-normalization (T5 D-1): real checkpoint round-trips deserialize
nested values as ``list``/``dict``. Every node entry passes through
``normalize_checkpoint_state`` first, which re-validates/coerces the nested
contracts (``semantic``, ``bindings``, ``query_analysis``,
``route_decision``, ``execution``, ``clarification``, ``request``,
``conversation``) before any node walks them. The frozen
``validate_supervisor_state`` is applied at the boundaries this task owns —
``direct_node``, the clarify persist+interrupt composition, and the initial
state envelope — so an invalid aggregate is never written by T6-owned code.

``DispatchReport.truncated`` (T3-N2) stays runtime-only ON PURPOSE: the
frozen ``ExecutionState`` has no slot for it, and persisting dispatch
metadata would need a contract change owned by the frozen Phase-1 schema.
``execute_node`` persists results only; truncation is visible to the live
runner via the scheduler report, never via checkpoint state.

Lease ordering: binding pins acquire/commit retention leases inside the
nodes (T1/T3-owned); terminal release (``release_run``) belongs to the OUTER
runner (T8) after the terminal checkpoint succeeds — never inside the
finalizer. The checkpoint DB and application DB never share a transaction.

``complex_boundary`` is the single Phase-3 replacement seam: it is compiled
as a plain node with NO checkpointer of its own, so the Phase-3 subgraph
attached here inherits the supervisor saver (and the shadow run's isolated
saver) with its state, interrupt/resume, and checkpoint namespacing intact.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import json
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4, uuid5

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command  # noqa: F401  (re-exported for T7/T8 runner use)
from pydantic import ValidationError

from .v2.capabilities import (
    AbbreviationCapability,
    CapabilityRegistration,
    CapabilityRegistry,
    CapabilityRuntimeContext,
    DocumentReadCapability,
    DocumentSearchCapability,
    EvidenceBuilder,
    KnowledgeGraphCapability,
    LocatedContent,
    MemoryCapability,
    PeopleCapability,
    PinnedTargetResolver,
    SectionReadCapability,
    build_capability_registry,
)
from .v2.contracts.base import CONTRACT_VERSION
from .v2.contracts.binding import DocumentBindingSet, DocumentDiscoveryCandidate
from .v2.contracts.clarification import ClarificationRequest
from .v2.contracts.conversation import ConversationContext
from .v2.contracts.request import RequestContext
from .v2.contracts.response import FinalResponse
from .v2.contracts.routing import QueryAnalysis, RouteDecision
from .v2.contracts.semantic import SemanticContext, SemanticDraft
from .v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from .v2.contracts.validation import (
    IncompatibleCheckpointError,
    validate_checkpoint_payload,
    validate_supervisor_state,
)
from .v2.nodes.binding import binding_node
from .v2.nodes.clarification import clarify_node as _persist_clarification
from .v2.nodes.clarification import (
    interrupt_for_clarification,
    resume_clarification,  # noqa: F401 (re-exported for the T7/T8 runner)
)
from .v2.nodes.context import context_node, node_context, semantic_finalizer_node
from .v2.nodes.evaluate import AnswerDraftChannel, evaluate_node
from .v2.nodes.execute import execute_node
from .v2.nodes.finalizer import finalizer_node
from .v2.nodes.fast_plan import fast_plan_node
from .v2.nodes.grounding import ground_node
from .v2.nodes.routing import route_node
from .v2.nodes.synthesize import synthesize_node

__all__ = [
    "SUPERVISOR_V2_NODES",
    "SupervisorV2Error",
    "BindingResolverError",
    "V1ServiceUnavailable",
    "normalize_checkpoint_state",
    "build_initial_v2_state",
    "direct_node",
    "complex_unavailable_node",
    "clarify_persist_node",
    "clarify_wait_node",
    "create_supervisor_v2_graph",
    "get_supervisor_v2_graph",
    "set_supervisor_v2_graph",
    "reset_supervisor_v2_graph",
    "supervisor_v2_ready",
    "supervisor_v2_lifespan",
    "DeterministicSemanticAdapter",
    "V1BindingResolver",
    "V1PeopleLookupService",
    "V1DocumentSearchService",
    "V1DocumentContentReader",
    "V1SectionContentReader",
    "V1KnowledgeGraphClient",
    "V1MemoryStore",
    "V1AbbreviationResolverService",
    "V1ServiceBundle",
    "V1_SERVICE_GATES",
    "probe_v1_services",
    "build_v2_capability_registry",
    "build_runtime_services",
    "build_graph_runtime_context",
]


class SupervisorV2Error(RuntimeError):
    """Supervisor v2 composition/lifecycle misuse; fail closed, never default.

    A ``RuntimeError`` (not ``ValueError``): it signals a lifecycle state
    problem (uninitialized singleton, missing route topology), never a bad
    business value. The T7 selector catches this to keep the v1 default.
    """


class BindingResolverError(ValueError):
    """The binding resolver cannot scope this resolution; nothing is pinned."""


class V1ServiceUnavailable(ValueError):
    """A v1 backing service is absent or unusable; the capability stays gated.

    Raised only on the call path (never at construction) so the registry
    availability gate — not an import-time crash — decides dispatch. Callers
    (capability ``execute``) translate it into a typed ``AgentResult`` via
    ``dependency_error``; it never escapes as checkpoint state.
    """


# ---------------------------------------------------------------------------
# Checkpoint normalization (T5 D-1)
# ---------------------------------------------------------------------------

_SLOT_MODELS: dict[str, type] = {
    "request": RequestContext,
    "conversation": ConversationContext,
    "semantic": SemanticContext,
    "bindings": DocumentBindingSet,
    "query_analysis": QueryAnalysis,
    "route_decision": RouteDecision,
    "execution": ExecutionState,
    "clarification": ClarificationRequest,
    "final_response": FinalResponse,
}

#: Slots that may legitimately be ``None`` (everything else must be present).
_NULLABLE_SLOTS = frozenset(
    {"query_analysis", "route_decision", "clarification", "final_response"}
)


def _coerce_slot(value: Any, model: type, *, slot: str) -> Any:
    """Re-validate one checkpoint slot into its frozen contract model.

    Accepts the live model, the Python-mode mapping, or the JSON-mode
    mapping a real checkpoint round-trip delivers (tuples as lists,
    datetimes as ISO strings, UUIDs as strings). Anything else fails closed
    with ``IncompatibleCheckpointError`` — a value that cannot be
    re-validated must never be walked as a contract.

    Live models are NEVER trusted as-is: checkpoint serde revives the
    envelope while leaving nested values as ``list``/``dict`` (verified:
    ``ConversationContext`` revives with plain-dict ``recent_turns``), so
    every model round-trips through JSON validation to rebuild the full
    tree before any node walks it.
    """
    if value is None:
        return None
    if isinstance(value, model):
        try:
            return model.model_validate_json(value.model_dump_json())
        except (ValidationError, TypeError, ValueError) as exc:
            raise IncompatibleCheckpointError(
                f"checkpoint {slot} does not re-validate as "
                f"{model.__name__}: {exc}"
            ) from exc
    if isinstance(value, Mapping):
        payload = dict(value)
        try:
            candidate = model.model_validate(payload)
            return model.model_validate_json(candidate.model_dump_json())
        except ValidationError:
            pass
        try:
            return model.model_validate_json(json.dumps(payload, default=str))
        except (ValidationError, TypeError, ValueError) as exc:
            raise IncompatibleCheckpointError(
                f"checkpoint {slot} is not a valid {model.__name__}: {exc}"
            ) from exc
    raise IncompatibleCheckpointError(
        f"checkpoint {slot} is not a {model.__name__} "
        f"(got {type(value).__name__})"
    )


def normalize_checkpoint_state(state: Mapping[str, Any]) -> SupervisorV2State:
    """Coerce a loaded checkpoint aggregate back into usable contracts.

    Runs the frozen shape gate (``validate_checkpoint_payload``: version +
    required keys + nested envelope versions) then re-validates every nested
    slot. Returns a NEW aggregate; the stored checkpoint is never mutated.
    Full cross-reference validation (``validate_supervisor_state``) is NOT
    applied here — the route→clarify edge legitimately checkpoints a clarify
    route before ``clarify_node`` persists its request (T5 D6 gap, owned by
    the clarify composition below).
    """
    values = dict(state)
    validate_checkpoint_payload(values)
    coerced: dict[str, Any] = {"contract_version": values["contract_version"]}
    for slot, model in _SLOT_MODELS.items():
        value = values.get(slot)
        if value is None and slot not in _NULLABLE_SLOTS:
            raise IncompatibleCheckpointError(
                f"checkpoint payload is missing required slot {slot!r}"
            )
        coerced[slot] = _coerce_slot(value, model, slot=slot)
    return SupervisorV2State(**coerced)  # type: ignore[typeddict-unknown-key]


def _wrap_node(name: str, fn: Callable) -> Callable:
    """Normalize checkpoint state before the node walks any nested value."""

    @functools.wraps(fn)
    async def _node(state: SupervisorV2State, runtime: Any) -> dict:
        return await fn(normalize_checkpoint_state(state), runtime)

    _node.__v2_origin__ = fn  # type: ignore[attr-defined]
    _node.__v2_node__ = name  # type: ignore[attr-defined]
    return _node


def build_initial_v2_state(
    *,
    request: RequestContext,
    conversation: ConversationContext | None = None,
    semantic: SemanticContext | None = None,
    bindings: DocumentBindingSet | None = None,
) -> SupervisorV2State:
    """Build the validated ingress envelope for a new v2 turn (T7 seam).

    Only the envelope shape plus the supplied ``request``/``conversation``
    are content-validated here: a fresh turn has no finalized semantics yet,
    so full-aggregate validation first passes once the owned writer
    boundaries (clarify/direct/finalizer path) have run. Placeholders stay
    type-correct (empty models, ``None`` optionals) so the first checkpoint
    is shape-compatible.
    """
    from .v2.contracts.validation import (
        validate_conversation_context,
        validate_request_context,
    )

    validate_request_context(request)
    conversation = (
        conversation
        if conversation is not None
        else ConversationContext(
            summary="", active_entities=(), last_focus=None, recent_turns=()
        )
    )
    validate_conversation_context(conversation)
    semantic = (
        semantic
        if semantic is not None
        else SemanticContext(
            contextualized_query="",
            normalized_query="",
            abbreviations=(),
            coreferences=(),
            document_refs=(),
            person_refs=(),
            section_refs=(),
            blocking_ambiguities=(),
        )
    )
    state = SupervisorV2State(
        contract_version=CONTRACT_VERSION,
        request=request,
        conversation=conversation,
        semantic=semantic,
        bindings=bindings
        if bindings is not None
        else DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        query_analysis=None,
        route_decision=None,
        execution=ExecutionState(plan=None, task_results=(), evidence_evaluation=None),
        clarification=None,
        final_response=None,
    )
    validate_checkpoint_payload(dict(state))
    return state


# ---------------------------------------------------------------------------
# Thin boundary nodes (owned by this module; no domain business)
# ---------------------------------------------------------------------------


async def direct_node(state: SupervisorV2State, runtime: Any) -> dict:
    """Emit the direct non-factual success boundary with ``plan=None``.

    A resumed turn may carry a stale factual plan; the direct route executes
    no capability, so the stale plan/results/evaluation are cleared rather
    than checkpointed alongside a direct success. The success content itself
    is the finalizer's job (``build_direct_response``). The merged aggregate
    is fully validated (D6) before it can be written.
    """
    node_context(runtime)
    update = {
        "execution": ExecutionState(
            plan=None, task_results=(), evidence_evaluation=None
        )
    }
    validate_supervisor_state({**normalize_checkpoint_state(state), **update})  # type: ignore[typeddict-unknown-key]
    return update


async def complex_unavailable_node(state: SupervisorV2State, runtime: Any) -> dict:
    """Typed-unavailable boundary for complex routes (Phase 3 replaces it).

    Writes nothing: the finalizer owns the typed unavailable response
    (``denied`` for Write, ``error`` for every other complex reason), so a
    wiring bug can never look like an answer. Phase 3 attaches the compiled
    complex-research subgraph at this exact node name; the subgraph is
    compiled WITHOUT its own checkpointer so it inherits this graph's saver.
    """
    node_context(runtime)
    return {}


async def clarify_persist_node(state: SupervisorV2State, runtime: Any) -> dict:
    """Persist the stable clarification request (suspend lives downstream).

    ``clarify_node`` (idempotent: a live persisted request is returned
    unchanged) owns the D6 gap — the clarify route checkpoints here WITH
    its persisted ``ClarificationRequest``, so the merged aggregate fully
    validates BEFORE anything suspends the run. Suspension itself fires in
    the immediately following ``clarify_wait`` node — verified empirically
    that suspending any earlier (inside this node, or in the ``clarify``
    edge) discards this persist update: LangGraph re-runs the suspending
    step from the top on resume, so the suspension checkpoint would carry
    the exact invalid aggregate D6 forbids (clarify route, no request) and
    the stable per-request deadline would be silently extended on rebuild.
    With the split, the suspension checkpoint always carries the request.
    On resume the outer runner drives ``resume_clarification`` +
    ``Command(goto="binding")`` itself (T7/T8); candidate-free replies
    surface as ``ClarificationUnsatisfiable`` there under the fresh-turn
    contract.
    """
    view = normalize_checkpoint_state(state)
    update = await _persist_clarification(view, runtime)
    validate_supervisor_state({**view, **update})  # type: ignore[typeddict-unknown-key]
    return update


async def clarify_wait_node(state: SupervisorV2State, runtime: Any) -> dict:
    """Suspend the clarify turn on the already-checkpointed request.

    Runs strictly after ``clarify_persist_node`` completes AND checkpoints,
    so the suspension checkpoint always carries the persisted
    ``ClarificationRequest`` D6 requires. First pass raises the LangGraph
    interrupt with that payload; on resume the interrupt delivers the outer
    runner's ``Command(resume=...)`` value and the turn falls through to
    ``finalizer`` unless the runner overrode navigation with
    ``goto="binding"`` (the ``resume_clarification`` path). A missing
    request here fails closed — this node never invents one.
    """
    node_context(runtime)
    request = _coerce_slot(
        normalize_checkpoint_state(state).get("clarification"),
        ClarificationRequest,
        slot="clarification",
    )
    if request is None:
        raise SupervisorV2Error(
            "clarify_wait requires the persisted ClarificationRequest; "
            "refusing to suspend without one"
        )
    await interrupt_for_clarification(request)
    return {}


# ---------------------------------------------------------------------------
# Graph composition
# ---------------------------------------------------------------------------

SUPERVISOR_V2_NODES: dict[str, Callable] = {
    "context": _wrap_node("context", context_node),
    "binding": _wrap_node("binding", binding_node),
    "semantic_finalizer": _wrap_node("semantic_finalizer", semantic_finalizer_node),
    "route": _wrap_node("route", route_node),
    "direct": direct_node,
    "clarify": clarify_persist_node,
    "clarify_wait": clarify_wait_node,
    "fast_plan": _wrap_node("fast_plan", fast_plan_node),
    "execute": _wrap_node("execute", execute_node),
    "evaluate": _wrap_node("evaluate", evaluate_node),
    "synthesize": _wrap_node("synthesize", synthesize_node),
    "ground": _wrap_node("ground", ground_node),
    "finalizer": _wrap_node("finalizer", finalizer_node),
    # Phase 3 replaces only this node with the compiled complex-research
    # subgraph. The subgraph is compiled without a checkpointer so it
    # inherits `checkpointer` (and therefore the shadow run's isolated
    # saver) rather than opening its own.
    "complex_boundary": complex_unavailable_node,
}

_ROUTE_BRANCHES = {
    "direct": "direct",
    "clarify": "clarify",
    "fast_domain": "fast_plan",
    "complex_research": "complex_boundary",
}


def _route_branch(state: SupervisorV2State) -> str:
    """Branch on the checkpointed route decision (resume-safe coercion).

    Edges receive raw checkpoint state, so a ``Mapping`` decision left by a
    real round-trip is re-validated before branching; a missing or foreign
    route fails closed instead of falling through to a default topology.
    Returns the ROUTE key (``add_conditional_edges`` maps it to the node
    via ``_ROUTE_BRANCHES``) — never a node name.
    """
    decision = state.get("route_decision")
    if isinstance(decision, Mapping):
        decision = _coerce_slot(decision, RouteDecision, slot="route_decision")
    if not isinstance(decision, RouteDecision):
        raise SupervisorV2Error(
            "route branch requires the checkpointed route decision; "
            "refusing to guess a topology"
        )
    if decision.route not in _ROUTE_BRANCHES:
        raise SupervisorV2Error(
            f"unknown v2 route {decision.route!r}; refusing to guess a topology"
        )
    return decision.route


def _add_supervisor_edges(graph: StateGraph) -> None:
    graph.set_entry_point("context")
    graph.add_edge("context", "binding")
    graph.add_edge("binding", "semantic_finalizer")
    graph.add_edge("semantic_finalizer", "route")
    graph.add_conditional_edges("route", _route_branch, _ROUTE_BRANCHES)
    graph.add_edge("direct", "finalizer")
    graph.add_edge("clarify", "clarify_wait")
    graph.add_edge("clarify_wait", "finalizer")
    graph.add_edge("fast_plan", "execute")
    graph.add_edge("execute", "evaluate")
    graph.add_edge("evaluate", "synthesize")
    graph.add_edge("synthesize", "ground")
    graph.add_edge("ground", "finalizer")
    graph.add_edge("complex_boundary", "finalizer")
    graph.add_edge("finalizer", END)


def create_supervisor_v2_graph(checkpointer: BaseCheckpointSaver) -> Any:
    """Compile the node-based supervisor v2 against ``checkpointer``.

    ``checkpointer`` is any LangGraph checkpoint saver (production: the
    lifespan-owned opened ``AsyncPostgresSaver``; tests: ``InMemorySaver``).
    """
    graph = StateGraph(SupervisorV2State, context_schema=GraphRuntimeContext)
    for name, fn in SUPERVISOR_V2_NODES.items():
        graph.add_node(name, fn)
    _add_supervisor_edges(graph)
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Lifecycle singleton (lifespan owns the saver; T7 owns per-request ingress)
# ---------------------------------------------------------------------------

_supervisor_v2_graph: Any = None
_supervisor_v2_lock = threading.Lock()


def set_supervisor_v2_graph(graph: Any) -> None:
    """Install the lifespan-compiled graph (single writer: the lifespan)."""
    global _supervisor_v2_graph
    with _supervisor_v2_lock:
        _supervisor_v2_graph = graph


def get_supervisor_v2_graph() -> Any:
    """Return the lifespan-compiled graph, or fail fast when v2 is down.

    Never auto-builds: compiling requires the lifespan-owned opened saver,
    so a missing singleton means v2 is unavailable and the caller (T7
    selector) must keep the v1 default. Double-checked locking mirrors v1.
    """
    global _supervisor_v2_graph
    if _supervisor_v2_graph is None:
        with _supervisor_v2_lock:
            if _supervisor_v2_graph is None:
                raise SupervisorV2Error(
                    "supervisor v2 graph is not initialized (lifespan did not "
                    "open the v2 checkpointer); v2 is unavailable, keep v1"
                )
    return _supervisor_v2_graph


def reset_supervisor_v2_graph() -> None:
    """Release the singleton (lifespan shutdown / hot-reload)."""
    global _supervisor_v2_graph
    with _supervisor_v2_lock:
        _supervisor_v2_graph = None


def supervisor_v2_ready() -> bool:
    """True once the lifespan has installed the compiled v2 graph."""
    return _supervisor_v2_graph is not None


@asynccontextmanager
async def supervisor_v2_lifespan(
    checkpoint_dsn: str, *, checkpointer_factory: Any = None
) -> AsyncIterator[Any]:
    """Own exactly ONE opened saver context for the v2 graph lifetime.

    Opens a single ``AsyncPostgresSaver`` via ``checkpointer_factory``
    (default: ``persistence.checkpoint.create_v2_checkpointer``), compiles
    the graph against it, installs the singleton, and releases everything on
    exit. Web startup NEVER calls saver setup/migration here — schema is
    owned by the migration runner; this helper only opens. Failures
    propagate so the caller (``main.lifespan``) can degrade to v1; the
    singleton is never left half-installed.
    """
    if checkpointer_factory is None:
        from .v2.persistence.checkpoint import create_v2_checkpointer

        checkpointer_factory = create_v2_checkpointer
    async with checkpointer_factory(checkpoint_dsn) as saver:
        set_supervisor_v2_graph(create_supervisor_v2_graph(saver))
        try:
            yield get_supervisor_v2_graph()
        finally:
            reset_supervisor_v2_graph()


# ---------------------------------------------------------------------------
# Concrete v1-backed adapters + registry construction (T2-I4)
# ---------------------------------------------------------------------------


def _v1_attr(module_name: str, attr: str) -> Any:
    """Import one v1 symbol lazily, or fail closed with typed unavailability."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise V1ServiceUnavailable(
            f"v1 module {module_name!r} is not importable; "
            f"{attr!r} is unavailable"
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise V1ServiceUnavailable(
            f"v1 module {module_name!r} has no {attr!r}; it is unavailable"
        ) from exc


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class DeterministicSemanticAdapter:
    """Request-scoped semantic draft builder honoring the D3 contract (T6).

    ``preprocess`` is ``(raw_query: str) -> PreprocessingResult`` (sync or
    async), injected per request by T7's ingress: the SAME ``(request,
    conversation)`` MUST yield an equal draft on every call (the draft is
    rebuilt once per node that needs it — context/binding/finalizer turns —
    so new ref ids per call would diverge the binding turn from the
    finalizer turn), and building MUST be read-only (no writes, no side
    effects). ``RequestContext.original_query`` stays the sole raw-query
    owner: it is passed to ``preprocess`` but never copied into the draft.
    The discourse-contextualized form comes from ``conversation.summary``
    (when set), else the preprocessor's normalized query; when neither
    exists the adapter refuses rather than copying the raw query.
    """

    def __init__(self, *, preprocess: Callable[[str], Any]) -> None:
        self._preprocess = preprocess

    async def build_draft(
        self, request: RequestContext, conversation: ConversationContext
    ) -> SemanticDraft:
        from .v2.adapters.semantic import SemanticAdapterError, draft_from_preprocessing

        result = await _maybe_await(self._preprocess(request.original_query))
        summary = conversation.summary.strip() if conversation.summary else ""
        try:
            return draft_from_preprocessing(
                result, contextualized_query=summary or None
            )
        except Exception as exc:
            raise SemanticAdapterError(
                f"v1 preprocessing output cannot become a v2 draft: {exc}"
            ) from exc


class V1BindingResolver:
    """Request-scoped resolver turning draft refs into pinned sets (T6, D5).

    Wraps ``adapters/document.py::resolve_document_bindings`` behind the
    node-facing ``resolve(document_refs, capability_runtime) ->
    DocumentBindingSet`` contract: iterates EVERY workspace in the trusted
    runtime (never collapsed to one), opens a dedicated session per
    workspace through ``session_factory`` (default: the app's
    ``async_session_maker``), and unions the per-workspace sets with
    first-workspace-wins pin identity. Empty references resolve to the empty
    set without opening any session; an empty workspace scope fails closed.
    """

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] | None = None,
        default_role: Any = None,
    ) -> None:
        self._session_factory = session_factory
        self._default_role = default_role

    def _sessions(self) -> Callable[[], Any]:
        if self._session_factory is not None:
            return self._session_factory

        def _default() -> Any:
            from app.core.database import async_session_maker

            return async_session_maker()

        return _default

    async def resolve(
        self,
        document_refs: Any,
        capability_runtime: CapabilityRuntimeContext,
    ) -> DocumentBindingSet:
        from .v2.adapters.document import resolve_document_bindings

        refs = tuple(document_refs)
        if not refs:
            return DocumentBindingSet(bindings=(), revision_requirement_refs=())
        workspaces = tuple(capability_runtime.workspace_ids)
        if not workspaces:
            raise BindingResolverError(
                "binding resolution requires at least one workspace scope; "
                "refusing to pin revisions outside a tenant"
            )
        open_session = self._sessions()
        bindings: dict[str, Any] = {}
        relations: dict[tuple[str, str], Any] = {}
        for workspace_id in workspaces:
            async with open_session() as db:
                resolved = await resolve_document_bindings(
                    db,
                    refs,
                    workspace_id=workspace_id,
                    default_role=self._default_role,
                )
            for binding in resolved.binding_set.bindings:
                bindings.setdefault(binding.binding_id, binding)
            for relation in resolved.binding_set.revision_requirement_refs:
                relations.setdefault(
                    (relation.binding_id, relation.ref_id), relation
                )
        return DocumentBindingSet(
            bindings=tuple(bindings.values()),
            revision_requirement_refs=tuple(relations.values()),
        )


class V1PeopleLookupService:
    """v1-backed People lookup: mongo record → governed mapping (T6).

    Delegates to ``mongo_people_service.search_by_name`` (async generator of
    ``found``/``persons`` dicts). Returns the first found person mapping, or
    ``None`` when unknown; malformed v1 output fails closed so the
    capability maps it to a typed error instead of checkpointing a guess.
    """

    def __init__(self, *, lookup: Any = None, limit: int = 10) -> None:
        self._lookup = lookup
        self._limit = limit

    async def lookup(self, query: str) -> Mapping[str, object] | None:
        lookup = self._lookup
        if lookup is None:
            lookup = _v1_attr("app.services.people.mongo_people_service", "search_by_name")
        produced = lookup(query, limit=self._limit)
        if inspect.isasyncgen(produced):
            async for result in produced:
                if not isinstance(result, Mapping):
                    raise V1ServiceUnavailable(
                        "v1 people lookup yielded a non-mapping result"
                    )
                if result.get("found") and result.get("persons"):
                    persons = result["persons"]
                    if isinstance(persons, (list, tuple)) and persons:
                        first = persons[0]
                        if not isinstance(first, Mapping):
                            raise V1ServiceUnavailable(
                                "v1 people lookup yielded a non-mapping person"
                            )
                        return first
            return None
        result = await _maybe_await(produced)
        if result is None:
            return None
        if not isinstance(result, Mapping):
            raise V1ServiceUnavailable("v1 people lookup returned a non-mapping")
        return result


class V1DocumentSearchService:
    """v1-backed document discovery: hybrid search → pinned candidates (T6).

    Delegates to ``tools.search_documents`` for retrieval, then pins each
    distinct document to its CURRENT revision through
    ``persistence.document_views`` (the v2-blessed immutable source — the v1
    ``ChatSourceChunk`` carries no revision, so minting a candidate without
    this lookup would invent identity). Documents with no current revision
    are skipped; an empty search returns no candidates (typed ``not_found``
    downstream, never an error).
    """

    def __init__(
        self,
        *,
        search: Any = None,
        session_factory: Callable[[], Any] | None = None,
        top_k: int = 10,
    ) -> None:
        self._search = search
        self._session_factory = session_factory
        self._top_k = top_k

    async def search(
        self,
        query: str,
        person_identifier: str | None,
        workspace_ids: tuple[UUID, ...],
    ) -> Any:
        from .v2.persistence import document_views

        search = self._search
        if search is None:
            search = _v1_attr("app.services.agent.tools", "search_documents")
        open_session = self._session_factory
        if open_session is None:

            def open_session() -> Any:  # type: ignore[no-redef]
                from app.core.database import async_session_maker

                return async_session_maker()

        async with open_session() as db:
            found = await search(
                query,
                self._top_k,
                [workspace_id for workspace_id in workspace_ids],
                set(),
                db,
            )
        sources = found.get("sources", ()) if isinstance(found, Mapping) else ()
        document_ids: list[UUID] = []
        seen: set[UUID] = set()
        for source in sources:
            raw_id = (
                source.get("document_id")
                if isinstance(source, Mapping)
                else getattr(source, "document_id", None)
            )
            try:
                document_id = raw_id if isinstance(raw_id, UUID) else UUID(str(raw_id))
            except (ValueError, TypeError, AttributeError):
                continue
            if document_id not in seen:
                seen.add(document_id)
                document_ids.append(document_id)
        candidates: list[DocumentDiscoveryCandidate] = []
        async with open_session() as db:
            for document_id in document_ids:
                try:
                    identity = await document_views.load_current_revision_identity(
                        db, document_id
                    )
                except document_views.RevisionNotReady:
                    continue
                if identity is None:
                    continue
                candidates.append(
                    DocumentDiscoveryCandidate(
                        candidate_id=uuid4(),
                        document_id=document_id,
                        document_revision=str(identity.revision_id),
                    )
                )
        return candidates


class V1DocumentContentReader:
    """Pinned-revision document reader seam (T6 owns the class, T7 the wiring).

    Whole-document reads at a PINNED revision have no verified v1 source:
    v1 retrieval reads the current revision through the vector store, which
    would silently violate the v2 pin guarantee. This class therefore takes
    its ``read`` callable by injection and fails closed
    (``V1ServiceUnavailable``) until T7 wires a revision-exact reader. The
    seam shape matches ``DocumentContentReader.read(binding, locator)`` so
    the wiring is a pure ingress change with no node edits.
    """

    def __init__(self, *, read: Any = None) -> None:
        self._read = read

    async def read(self, binding: Any, locator: Any) -> LocatedContent:
        if self._read is None:
            raise V1ServiceUnavailable(
                "no pinned-revision document reader is wired; T7 ingress "
                "must supply a revision-exact reader (v1 vector reads are "
                "current-revision and cannot serve pinned bindings)"
            )
        return await _maybe_await(self._read(binding, locator))


class V1SectionContentReader:
    """Pinned-revision section reader seam (same T7-wiring contract)."""

    def __init__(self, *, read: Any = None) -> None:
        self._read = read

    async def read(self, binding: Any, locator: Any) -> LocatedContent:
        if self._read is None:
            raise V1ServiceUnavailable(
                "no pinned-revision section reader is wired; T7 ingress "
                "must supply a revision-exact reader"
            )
        return await _maybe_await(self._read(binding, locator))


#: Stable namespace for deterministic v2 ids derived from v1 free text.
_V1_TEXT_NAMESPACE = UUID("7c9e9a1e-0b6c-4a8b-9e2f-3d5a7b9c1e01")


class V1KnowledgeGraphClient:
    """v1-backed KG lookup: workspace query text → stable pairs (T6).

    Delegates to ``KnowledgeGraphService(workspace_id).query`` (returns
    answer text scoped to one workspace). The pair id is a deterministic
    ``uuid5`` of the text so re-runs converge instead of minting fresh
    identity per call. Blank answers yield no matches (typed ``not_found``
    downstream).
    """

    def __init__(self, *, query_fn: Any = None, workspace_id: UUID | None = None) -> None:
        self._query_fn = query_fn
        self._workspace_id = workspace_id

    async def query(self, query: str) -> Any:
        query_fn = self._query_fn
        if query_fn is None:
            if self._workspace_id is None:
                raise V1ServiceUnavailable(
                    "knowledge-graph queries need a workspace scope; "
                    "construct with workspace_id or an explicit query_fn"
                )

            def query_fn(question: str) -> Any:  # type: ignore[no-redef]
                service_cls = _v1_attr(
                    "app.services.kg.knowledge_graph_service", "KnowledgeGraphService"
                )
                return service_cls(workspace_id=self._workspace_id).query(question)

        text = await _maybe_await(query_fn(query))
        if not isinstance(text, str) or not text.strip():
            return ()
        return ((f"kg:{uuid5(_V1_TEXT_NAMESPACE, text.strip())}", text.strip()),)


class V1MemoryStore:
    """v1-backed memory lookup: Graphiti user memory → stable pairs (T6).

    Delegates to ``graphiti_client.search_user_memory`` (user-scoped text).
    Same deterministic-id treatment as the KG client; blank answers yield
    no matches.
    """

    def __init__(
        self,
        *,
        search: Any = None,
        user_id: UUID | None = None,
        top_k: int = 5,
    ) -> None:
        self._search = search
        self._user_id = user_id
        self._top_k = top_k

    async def lookup(self, query: str) -> Any:
        search = self._search
        if search is None:
            if self._user_id is None:
                raise V1ServiceUnavailable(
                    "memory lookup needs a user scope; construct with "
                    "user_id or an explicit search callable"
                )
            search = _v1_attr("app.services.memory.graphiti_client", "search_user_memory")
            text = await search(self._user_id, query, top_k=self._top_k)
        else:
            text = await _maybe_await(search(query))
        if not isinstance(text, str) or not text.strip():
            return ()
        return ((f"graphiti:{uuid5(_V1_TEXT_NAMESPACE, text.strip())}", text.strip()),)


class V1AbbreviationResolverService:
    """Deterministic abbreviation expansion over the v1 table (T6).

    Implements the SYNC ``AbbreviationResolverService.resolve`` protocol, so
    the session must arrive as a sync lookup closure (T7 ingress owns it;
    the v1 query shape mirrors ``tools.search_abbreviation``: active
    ``short_form`` ilike match). Absent wiring fails closed.
    """

    def __init__(self, *, lookup: Callable[[str], str | None] | None = None) -> None:
        self._lookup = lookup

    def resolve(self, token: str) -> str | None:
        if self._lookup is None:
            raise V1ServiceUnavailable(
                "no abbreviation lookup is wired; T7 ingress must supply a "
                "sync closure over the abbreviations table"
            )
        return self._lookup(token)


#: Capability name → deployment service-gate flag consumed by
#: ``build_capability_registry`` (absent/unwired backing ⇒ excluded ⇒ typed
#: ``CapabilityUnavailable`` at dispatch, never a silent fallback).
V1_SERVICE_GATES: dict[str, str] = {
    "people.lookup": "v1-people",
    "document.search": "v1-document-search",
    "document.read": "v1-document-content",
    "section.read": "v1-section-content",
    "knowledge_graph.query": "v1-knowledge-graph",
    "memory.lookup": "v1-memory",
    "abbreviation.resolve": "v1-abbreviation",
}


@dataclass(frozen=True)
class V1ServiceBundle:
    """Injectable v1 backings for registry construction (T6 builds, T7 fills).

    Every field defaults to ``None`` (lazy v1 default or explicit
    unavailability resolved at call time); tests inject fakes. ``evidence``
    and ``resolver`` are the persistence-backed seams (EvidenceBuilder /
    PinnedTargetResolver) owned by T7's per-request ingress — when absent,
    the capabilities that need them are gated out of the registry.
    """

    people_lookup: Any = None
    document_search: Any = None
    document_reader: Any = None
    section_reader: Any = None
    knowledge_graph_client: Any = None
    memory_store: Any = None
    abbreviation_lookup: Any = None
    session_factory: Callable[[], Any] | None = None
    evidence: Any = None
    resolver: Any = None
    user_id: UUID | None = None
    workspace_id: UUID | None = None


def probe_v1_services() -> frozenset[str]:
    """Import-check each v1 backing WITHOUT network or database IO.

    Returns the service-gate flags whose backing modules import. A present
    module means the code path exists — not that the live service answers;
    call-time failures still surface as typed ``V1ServiceUnavailable`` →
    ``dependency_error`` results.
    """
    probes = {
        "v1-people": ("app.services.people.mongo_people_service", "search_by_name"),
        "v1-document-search": ("app.services.agent.tools", "search_documents"),
        "v1-document-content": ("app.services.agents.v2.persistence.document_views", "load_current_revision_identity"),
        "v1-section-content": ("app.services.agent.tools", "search_document_section"),
        "v1-knowledge-graph": ("app.services.kg.knowledge_graph_service", "KnowledgeGraphService"),
        "v1-memory": ("app.services.memory.graphiti_client", "search_user_memory"),
        "v1-abbreviation": ("app.models.abbreviation", "Abbreviation"),
    }
    available: set[str] = set()
    for gate, (module_name, attr) in probes.items():
        try:
            _v1_attr(module_name, attr)
        except V1ServiceUnavailable:
            continue
        available.add(gate)
    return frozenset(available)


def build_v2_capability_registry(
    runtime: CapabilityRuntimeContext,
    *,
    bundle: V1ServiceBundle | None = None,
    evidence: EvidenceBuilder | None = None,
    resolver: PinnedTargetResolver | None = None,
    available_services: frozenset[str] | None = None,
) -> CapabilityRegistry:
    """Construct the request-scoped capability registry (T6 owns, T7 calls).

    All seven Phase-2 capabilities are registered with their v1-backed
    adapters; ``available_services`` (default: ``probe_v1_services()``)
    intersects them. Capabilities whose backing is missing — or whose
    persistence seams (``evidence``/``resolver``) T7 did not supply — are
    gated OUT, so dispatch raises typed ``CapabilityUnavailable`` (mapped to
    ``DEPENDENCY_UNAVAILABLE`` results) instead of silently degrading.
    """
    bundle = bundle if bundle is not None else V1ServiceBundle()
    available = (
        set(available_services)
        if available_services is not None
        else set(probe_v1_services())
    )
    evidence = evidence if evidence is not None else bundle.evidence
    resolver = resolver if resolver is not None else bundle.resolver
    if evidence is None:
        available -= {
            "v1-people",
            "v1-document-content",
            "v1-section-content",
            "v1-knowledge-graph",
            "v1-memory",
        }
    if resolver is None:
        available -= {"v1-document-content", "v1-section-content"}
    session_factory = bundle.session_factory
    abbreviation_source = bundle.abbreviation_lookup
    if abbreviation_source is None or (
        callable(abbreviation_source) and not hasattr(abbreviation_source, "resolve")
    ):
        abbreviation_service: Any = V1AbbreviationResolverService(
            lookup=abbreviation_source
        )
    else:
        abbreviation_service = abbreviation_source
    registrations = [
        CapabilityRegistration(
            PeopleCapability(
                service=bundle.people_lookup or V1PeopleLookupService(),
                evidence=evidence,
            ),
            service=V1_SERVICE_GATES["people.lookup"],
        ),
        CapabilityRegistration(
            DocumentSearchCapability(
                service=bundle.document_search
                or V1DocumentSearchService(session_factory=session_factory),
            ),
            service=V1_SERVICE_GATES["document.search"],
        ),
        CapabilityRegistration(
            DocumentReadCapability(
                reader=bundle.document_reader or V1DocumentContentReader(),
                evidence=evidence,
                resolver=resolver,
            ),
            service=V1_SERVICE_GATES["document.read"],
        ),
        CapabilityRegistration(
            SectionReadCapability(
                reader=bundle.section_reader or V1SectionContentReader(),
                evidence=evidence,
                resolver=resolver,
            ),
            service=V1_SERVICE_GATES["section.read"],
        ),
        CapabilityRegistration(
            KnowledgeGraphCapability(
                client=bundle.knowledge_graph_client
                or V1KnowledgeGraphClient(workspace_id=bundle.workspace_id),
                evidence=evidence,
            ),
            service=V1_SERVICE_GATES["knowledge_graph.query"],
        ),
        CapabilityRegistration(
            MemoryCapability(
                store=bundle.memory_store
                or V1MemoryStore(user_id=bundle.user_id),
                evidence=evidence,
            ),
            service=V1_SERVICE_GATES["memory.lookup"],
        ),
        CapabilityRegistration(
            AbbreviationCapability(service=abbreviation_service),
            service=V1_SERVICE_GATES["abbreviation.resolve"],
        ),
    ]
    return build_capability_registry(
        registrations, runtime, available_services=frozenset(available)
    )


def build_runtime_services(
    *,
    retention_leases: Any = None,
    semantic_adapter: Any = None,
    binding_resolver: Any = None,
    capability_registry: CapabilityRegistry | None = None,
    chat_messages: Any = None,
    authorization: Any = None,
    evidence_hydrator: Any = None,
    answer_draft_channel: AnswerDraftChannel | None = None,
) -> RuntimeServices:
    """Construct the request-scoped ``RuntimeServices`` (single bag, T7 seam).

    Every slot defaults to ``None`` (the nodes fail closed on the services
    they need); T7 supplies the persistence-backed pieces per request. This
    is the ONLY construction site for the bag besides tests — no second
    definition exists anywhere.
    """
    return RuntimeServices(
        retention_leases=retention_leases,
        semantic_adapter=semantic_adapter,
        binding_resolver=binding_resolver,
        capability_registry=capability_registry,
        chat_messages=chat_messages,
        authorization=authorization,
        evidence_hydrator=evidence_hydrator,
        answer_draft_channel=answer_draft_channel,
    )


def build_graph_runtime_context(
    capability_runtime: CapabilityRuntimeContext,
    *,
    services: RuntimeServices | None = None,
    **service_kwargs: Any,
) -> GraphRuntimeContext:
    """Pair trusted runtime authority with its request-scoped services."""
    return GraphRuntimeContext(
        capability_runtime=capability_runtime,
        services=services
        if services is not None
        else build_runtime_services(**service_kwargs),
    )

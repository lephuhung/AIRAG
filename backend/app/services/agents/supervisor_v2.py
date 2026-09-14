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

``DispatchReport.truncated`` (T3-N2) stays runtime-only ON PURPOSE — and,
to be precise, it is NOT visible to any outer runner today: ``execute_node``
builds the scheduler locally and persists results only, discarding the
report. The frozen ``ExecutionState`` has no slot for dispatch metadata,
and adding one would need a contract change owned by the frozen Phase-1
schema. The deliberate-loss record: after a deadline-truncated dispatch the
run can end in a normal typed response with no checkpointed truncation
signal. The T8 outer runner MUST therefore detect incomplete dispatch
itself at the terminal boundary by diffing the checkpointed plan against
its results (``undispatched_tasks`` below): a non-empty remainder with no
active run means the plan was not fully executed — raise rollback/error
from it, never present partial results as complete.

Recorded T7/T8 prerequisite: once conversions are sticky, the common shape
of an early-conversion terminal is an error marker over the INGRESS
placeholder semantic (blank ``contextualized_query``), which the frozen
validator rejects. A terminal checkpoint whose ``final_response.status``
is not ``"success"`` and whose semantic is still the ingress placeholder
must be treated as terminal-errored, not corrupt — never run
``validate_supervisor_state`` on it as a health check.

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

import asyncio
import functools
import hashlib
import importlib
import inspect
import json
import logging
import re
import threading
import warnings
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4, uuid5

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.types import Command  # noqa: F401  (re-exported for T7/T8 runner use)
from pydantic import ValidationError

from .v2.capabilities import (
    AbbreviationCapability,
    CapabilityRegistration,
    CapabilityRegistry,
    CapabilityRuntimeContext,
    DocumentReadCapability,
    DocumentRetrieveCapability,
    DocumentSearchCapability,
    EvidenceBuilder,
    KnowledgeGraphCapability,
    LocatedContent,
    MemoryCapability,
    MultiMatchPeopleLookupService,
    PeopleCapability,
    PinnedTargetResolver,
    SectionReadCapability,
    build_capability_registry,
)
from .v2.capabilities.document import RevisionRetrievedChunk
from .v2.contracts.base import CONTRACT_VERSION
from .v2.contracts.binding import DocumentBindingSet, DocumentDiscoveryCandidate
from .v2.contracts.clarification import ClarificationRequest, ClarificationResolution
from .v2.contracts.conversation import ConversationContext
from .v2.contracts.execution import AgentResult
from .v2.contracts.planning import TaskPlan
from .v2.contracts.request import KnownDocumentResource, RequestContext
from .v2.contracts.response import FinalResponse
from .v2.contracts.routing import QueryAnalysis, RouteDecision
from .v2.contracts.semantic import DocumentReference, SemanticContext, SemanticDraft
from .v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from .v2.contracts.validation import (
    ContractValidationError,
    IncompatibleCheckpointError,
    validate_checkpoint_payload,
    validate_supervisor_state,
)
from .v2.complex_research_graph import (
    build_complex_research_subgraph,
    make_complex_boundary_node,
)
from .v2.nodes.binding import binding_node
from .v2.nodes.clarification import clarify_node as _persist_clarification
from .v2.nodes.clarification import (
    ClarificationError,
    ClarificationExpired,
    ClarificationInvalidSelection,
    ClarificationUnauthorized,
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
    "undispatched_tasks",
    "dedicated_retention_leases",
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
    "V1RevisionAwareRetrievalService",
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


logger = logging.getLogger(__name__)


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
    tree before any node walks it. The round-trip is expected to meet
    serde-shaped values (lists for tuples, ISO strings), so serializer
    warnings from that shape are suppressed narrowly here — genuine
    validation failures still raise. Note the ``default=str`` fallback is
    permissive by construction (an arbitrary object becomes a string and
    can then satisfy a ``str`` field); it is reachable only for mapping
    slots arriving outside checkpoint serde, and the re-validated model is
    still fully checked before use.
    """
    if value is None:
        return None
    if isinstance(value, model):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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


#: User-safe content for owned-boundary validation failures. Internal
#: target/criterion/lease identifiers never reach user-facing responses.
_BOUNDARY_ERROR_CONTENT = "Đã xảy ra lỗi trong quá trình xử lý yêu cầu."

#: Nodes that own clarification semantics and must see the raw persisted
#: request (the T5 idempotence guard handles staleness itself). Every other
#: node gets turn hygiene: a clarification that no longer validates against
#: the current semantic is retired before the node walks the state.
_CLARIFICATION_OWNERS = frozenset({"clarify_persist", "clarify_wait"})

#: Nodes whose entries run before semantics are finalized (or that always
#: overwrite the route decision): normalization + hygiene only, never full
#: aggregate validation at entry.
_UNVALIDATED_ENTRIES = frozenset(
    {"context", "binding", "semantic_finalizer", "route"}
)


def _typed_boundary_error(detail: str) -> dict:
    """Convert an owned-boundary failure into a typed error.

    Returns a terminal ``error`` ``FinalResponse`` update instead of letting
    the failure escape the graph. The detail is logged (operators) but never
    surfaced (users). ``route_decision`` (and its ``query_analysis``
    evidence) is ALWAYS cleared: a converted failure has no validated
    topology left to execute, and a stale route would let the finalizer
    recompute over the marker (masking the error as success) or strand a
    clarify route with no request. Downstream owned boundaries short-circuit
    on the still-invalid aggregate and the finalizer wrapper returns this
    marker, so the turn converges to a typed error — never a success, never
    an exception, never a thread that re-raises on continue.
    """
    logger.warning("supervisor_v2 boundary validation failed: %s", detail)
    return {
        "final_response": FinalResponse(
            contract_version=CONTRACT_VERSION,
            status="error",
            content=_BOUNDARY_ERROR_CONTENT,
            citations=(),
        ),
        "route_decision": None,
        "query_analysis": None,
    }


def _retire_stale_clarification(view: SupervisorV2State) -> bool:
    """Drop a clarification that no longer validates against the semantic.

    Across turns the persisted request can outlive the projection it was
    built from (reply-driven resume, fresh turn over a suspended thread):
    the frozen validator then rejects the whole aggregate. Retiring the
    orphan before any non-clarify node walks the state keeps every later
    owned boundary valid. Returns True when the view was retired (the
    caller propagates ``clarification=None`` into the returned update).
    Failure-OPEN by design: ANY failure to re-validate (un-coercible
    slots, stale references, expired shape) retires the request rather than
    preserving it — dropping a clarification can never crash a turn, while
    keeping a stale one fails the frozen aggregate check downstream. The
    remaining slots are still fail-closed separately by entry validation.
    Skipped for the two clarify nodes (they own clarification semantics)
    and not re-checked inside the four ``_UNVALIDATED_ENTRIES`` nodes
    (pre-finalization states and the always-overwritten route decision
    cannot validate yet — accepted, recorded); every other owned boundary
    validates the retired aggregate on entry.
    """
    from .v2.contracts.validation import validate_clarification_request

    clarification = view.get("clarification")
    if clarification is None:
        return False
    try:
        request = _coerce_slot(
            clarification, ClarificationRequest, slot="clarification"
        )
        semantic = _coerce_slot(view.get("semantic"), SemanticContext, slot="semantic")
        validate_clarification_request(request, semantic)
    except Exception:  # noqa: BLE001 - any staleness retires; validity is enforced later
        return True
    return False


def _has_terminal_marker(state: Mapping[str, Any]) -> bool:
    """True when the state carries a non-success terminal response.

    Accepts the live model and the checkpoint-serde mapping form. Only
    owned conversions and ground's failure path write non-success markers
    mid-flow, so at the route boundary this means an owned failure already
    converted this turn (success markers always re-derive normally).
    """
    marker = state.get("final_response")
    if marker is None:
        return False
    status = getattr(marker, "status", None)
    if status is None and isinstance(marker, Mapping):
        status = marker.get("status")
    return status is not None and status != "success"


def _wrap_node(name: str, fn: Callable) -> Callable:
    """Normalize, hygienize, and boundary-validate around every node.

    - Every entry normalizes checkpoint state (T5 D-1) and — except on the
      two clarify nodes, which own clarification semantics — retires a
      clarification orphaned by a turn change (C1 Proof 2/3).
    - Post-finalization entries (everything past ``route``) fully validate
      the aggregate: entry failures convert WITHOUT running the node, so no
      capability ever dispatches on an invalid aggregate and no exception
      escapes.
    - ``direct``/``clarify``/``clarify_wait``/``fast_plan`` fix an
      expected transient invalidity with their update, so they validate the
      MERGED state instead (conversion on failure).
    - Conversions are STICKY (except ``finalizer``/``clarify_wait``): the
      error marker converges to the finalizer through state (route cleared
      plus the route branch sending marker-states to the finalizer), so no
      downstream node overwrites it and no capability dispatches after a
      failure. ``clarify_wait``/``finalizer`` have no
      static outgoing edge (own ``Command`` navigation / already terminal),
      so their plain-dict conversions terminate in place.
    - The finalizer wrapper never emits success from an invalid aggregate:
      entry failure returns the already-checkpointed marker (unless it is a
      stale success) or a fresh typed error, without running the node.
    """

    sme_merge_nodes = name in ("direct", "clarify", "clarify_wait", "fast_plan")
    # Sticky conversions: a converted failure becomes a route-cleared error
    # marker that NO downstream node overwrites. Stickiness is STATE-based,
    # not navigation-based: a node-returned Command goto UNIONS with static
    # edges on this stack (verified: both targets run), so navigation cannot
    # carry it. Instead (a) every conversion clears route/analysis, (b) the
    # route wrapper skips re-derivation while a non-success marker stands,
    # (c) the route branch sends marker-states straight to the finalizer,
    # (d) context clears stale markers at every turn start, and (e) the
    # finalizer returns a routeless marker as-is. Exempt from NOTHING here:
    # finalizer/clarify_wait conversions are plain dicts that terminate in
    # place (no static edge out of clarify_wait; finalizer is terminal).
    # clarify_wait's own navigation Commands (resume paths) pass through
    # the Command branch below untouched.

    @functools.wraps(fn)
    async def _node(state: SupervisorV2State, runtime: Any) -> dict:
        # Every failure below converts to a STICKY typed terminal response —
        # GraphInterrupt (suspension) is the sole exception and always
        # re-raises; cancellation (BaseException, not Exception) never
        # matches by construction. Stickiness is STATE-based: the error
        # marker always travels with a cleared route, the route wrapper
        # never re-derives over a marker, the route branch sends
        # marker-states straight to the finalizer, and context clears stale
        # markers at every turn start — so no downstream node overwrites the
        # marker and no capability dispatches after a failure. (Navigation
        # cannot carry stickiness: node-returned Command gotos UNION with
        # static edges on this stack.) A mid-chain raise would strand
        # `next=(node,)` on a state that re-raises on continue (poisoned).
        try:
            view = normalize_checkpoint_state(state)
        except GraphInterrupt:
            raise
        except Exception as exc:
            return _typed_boundary_error(f"{name} normalize: {exc}")
        retired = False
        if name not in _CLARIFICATION_OWNERS:
            retired = _retire_stale_clarification(view)
            if retired:
                view = SupervisorV2State(**{**view, "clarification": None})  # type: ignore[typeddict-unknown-key]
        if name == "finalizer":
            try:
                validate_supervisor_state(view)
            except GraphInterrupt:
                raise
            except Exception as exc:
                marker = view.get("final_response")
                if (
                    isinstance(marker, FinalResponse)
                    and marker.status != "success"
                ):
                    logger.warning(
                        "supervisor_v2 finalizer short-circuit: %s", exc
                    )
                    return {"final_response": marker}
                return _typed_boundary_error(f"finalizer entry: {exc}")
            marker = view.get("final_response")
            if (
                isinstance(marker, FinalResponse)
                and marker.status != "success"
                and view.get("route_decision") is None
            ):
                # Conversion marker with nothing left to finalize (an owned
                # boundary retired the route with the request): terminal
                # as-is instead of recomputing. Valid states WITH a route
                # always recompute below (e.g. ground's insufficient).
                logger.warning("supervisor_v2 finalizer returning routed-away marker")
                return {"final_response": marker}
            try:
                return await fn(view, runtime)
            except GraphInterrupt:
                raise
            except Exception as exc:
                # T4 FinalizerError included: "the run must fail, not
                # succeed" becomes the same typed error response.
                return _typed_boundary_error(f"finalizer: {exc}")
        if name == "route" and _has_terminal_marker(view):
            # An owned boundary already converted this turn (conversions
            # always clear the route): never re-derive a topology over the
            # failure. The branch below sends marker-states to the finalizer.
            return {}
        if name in ("direct", "fast_plan", "execute", "evaluate", "synthesize", "ground", "complex_boundary") and _has_terminal_marker(view):
            # Same stickiness downstream: a conversion marker must reach the
            # finalizer untouched — running here would overwrite it (ground
            # re-derives insufficiency over errors; execute could dispatch
            # after a fail-closed failure) and resurrect the turn. Ground's
            # own failure marker is set by ground itself, so it is never
            # skipped into existence: only the finalizer consumes markers.
            return {}
        if name not in _UNVALIDATED_ENTRIES and not sme_merge_nodes:
            try:
                validate_supervisor_state(view)
            except GraphInterrupt:
                raise
            except Exception as exc:
                return _typed_boundary_error(f"{name} entry: {exc}")
        try:
            update = await fn(view, runtime)
        except GraphInterrupt:
            raise
        except Exception as exc:
            return _typed_boundary_error(f"{name}: {exc}")
        if name == "context" and isinstance(update, dict):
            # Fresh turn, fresh response: stale terminal markers from a
            # prior turn must not survive into this turn's flow (they would
            # trip the route skip + branch-error below and stick the new
            # turn in the old error). The finalizer recomputes every turn.
            update = {**update, "final_response": None}
        try:
            if isinstance(update, Command):
                # Navigation-carrying return (clarify_wait resume paths):
                # apply hygiene + merged validation to the payload,
                # preserving goto. A non-dict payload fails closed — it
                # must never silently drop state.
                payload = update.update
                if payload is None:
                    payload = {}
                if not isinstance(payload, dict):
                    raise SupervisorV2Error(
                        f"{name} returned a Command with a non-dict update; "
                        "refusing to drop state silently"
                    )
                payload = dict(payload)
                if retired and "clarification" not in payload:
                    payload = {**payload, "clarification": None}
                validate_supervisor_state({**view, **payload})  # type: ignore[typeddict-unknown-key]
                return Command(update=payload, goto=update.goto)
            if not isinstance(update, dict):
                return update
            if retired and "clarification" not in update:
                update = {**update, "clarification": None}
            if sme_merge_nodes or name not in _UNVALIDATED_ENTRIES:
                validate_supervisor_state({**view, **update})  # type: ignore[typeddict-unknown-key]
            return update
        except GraphInterrupt:
            raise
        except Exception as exc:
            # Wrapper-level corruption of a navigation-carrying return can
            # never continue the planned navigation: force the finalizer.
            return Command(
                update=_typed_boundary_error(f"{name} merged: {exc}"),
                goto="finalizer",
            )

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
    On resume the outer runner passes T5's ``resume_clarification`` output
    through VERBATIM (``Command(resume=...)``, no outer ``goto`` — the
    graph owns navigation); candidate-free replies surface as
    ``ClarificationUnsatisfiable`` there under the fresh-turn contract.
    """
    view = normalize_checkpoint_state(state)
    update = await _persist_clarification(view, runtime)
    validate_supervisor_state({**view, **update})  # type: ignore[typeddict-unknown-key]
    return update


async def clarify_wait_node(state: SupervisorV2State, runtime: Any) -> dict:
    """Suspend on the checkpointed request; on resume, apply the answer.

    Runs strictly after ``clarify_persist_node`` completes AND checkpoints,
    so the suspension checkpoint always carries the persisted
    ``ClarificationRequest`` D6 requires. First pass raises the LangGraph
    interrupt with that payload.

    RESUMED pass (``interrupt()`` returns the outer runner's
    ``Command(resume=...)`` value, byte-for-byte T5's
    ``resume_clarification`` output): the value is treated as the answer —
    coerced to a ``ClarificationResolution`` (JSON-mode aware), checked for
    expiry, membership-validated against the persisted request with the
    frozen validator, and authorized under the CURRENT runtime ACL (T5 typed
    errors throughout). The answered request is then RETIRED
    (``clarification=None``) and the selection is recorded as a
    ``ui_selection`` attachment on the request plus a resolved document
    reference in the semantic state. This node writes NO binding pins
    itself — the re-driven ``binding`` node rebuilds the draft (the
    selection-aware adapter resolves the answered ref from the attachment),
    pins through the resolver, and leases before checkpointing, so every
    key keeps a single writer per step and the standard lease ordering
    holds. Navigation after this update is driven by the RETURNED
    ``Command`` (``goto="binding"`` on success, ``goto="finalizer"`` on
    fallthrough or conversion) — there is deliberately NO static edge out
    of ``clarify_wait``. The runner passes T5's ``resume_clarification``
    output through VERBATIM (``Command(resume=...)`` with NO outer ``goto``
    — the graph owns navigation per the T5 round-2 contract). Verified
    empirically: an outer ``goto="binding"`` pre-schedules ``binding``
    into the SAME super-step as the resumed wait (stale-read pin loss),
    and with the universal conversion in place the corrupted shape now
    terminates typed instead of escaping — but the turn is still destroyed,
    so an outer ``goto`` must never be added. Required T8 call form::

        command = await resume_clarification(message_id, request, runtime)
        await graph.ainvoke(command, config, context=runtime)

    Resume failures (expired/forged/unauthorized/unusable selection,
    un-pinnable document) retire the request and convert to a typed
    ``denied``/``error`` response — never an escape, never a re-ask loop.
    A missing persisted request (runner cleared it) is a typed error: the
    runner can neither keep a stale request nor clear it silently.
    """
    context = node_context(runtime)
    view = normalize_checkpoint_state(state)
    persisted = _coerce_slot(
        view.get("clarification"), ClarificationRequest, slot="clarification"
    )
    if persisted is None:
        return _typed_boundary_error("clarify_wait without persisted request")
    resume_value = await interrupt_for_clarification(persisted)
    if resume_value is None:
        # Defensive: the interrupt resolved without a payload. Fall through
        # to the finalizer, which re-emits the clarify question.
        return Command(update={}, goto="finalizer")
    try:
        update = await _apply_clarification_resolution(
            view, persisted, resume_value, context
        )
        return Command(update=update, goto="binding")
    except ClarificationUnauthorized as exc:
        logger.warning("supervisor_v2 resume denied: %s", exc)
        return Command(
            update={
                "clarification": None,
                "route_decision": None,
                "query_analysis": None,
                "final_response": FinalResponse(
                    contract_version=CONTRACT_VERSION,
                    status="denied",
                    content="Yêu cầu bị từ chối quyền truy cập.",
                    citations=(),
                ),
            },
            goto="finalizer",
        )
    except ClarificationError as exc:
        logger.warning("supervisor_v2 resume failed: %s", exc)
        return Command(
            update={
                "clarification": None,
                **_typed_boundary_error(f"clarify_wait resume: {exc}"),
            },
            goto="finalizer",
        )


def _resolution_from_mapping(value: Any) -> ClarificationResolution | None:
    """Coerce a resume payload into the frozen selection (JSON-mode aware).

    Mirrors T5's ``_request_from_mapping``: the runner's ``Command(resume=…)``
    carries ``model_dump(mode="json")`` output, which strict Python-mode
    validation can reject — so mappings validate in Python mode first, then
    through a JSON round-trip. Garbage yields ``None`` (invalid selection).
    """
    if isinstance(value, ClarificationResolution):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        try:
            return ClarificationResolution.model_validate(payload)
        except ValidationError:
            pass
        try:
            return ClarificationResolution.model_validate_json(
                json.dumps(payload, default=str)
            )
        except (ValidationError, TypeError, ValueError):
            return None
    return None


def _expires_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


async def _apply_clarification_resolution(
    view: SupervisorV2State,
    persisted: ClarificationRequest,
    resume_value: Any,
    context: GraphRuntimeContext,
) -> dict:
    """Validate a resumed selection and fold it into checkpointable state."""
    from .v2.contracts.validation import validate_clarification_resolution

    resolution = _resolution_from_mapping(resume_value)
    if resolution is None:
        raise ClarificationInvalidSelection(
            f"clarification {persisted.clarification_id!r} resume carries "
            "no usable selection"
        )
    if datetime.now(timezone.utc) >= _expires_aware(persisted.expires_at):
        raise ClarificationExpired(persisted.clarification_id)
    try:
        validate_clarification_resolution(persisted, resolution)
    except ContractValidationError as exc:
        # Membership/identity failures are invalid selections: retire the
        # request with the other resume failures instead of crashing.
        raise ClarificationInvalidSelection(
            f"clarification {persisted.clarification_id!r} resume selects "
            f"nothing offered: {exc}"
        ) from exc
    if resolution.selected_candidate_id is None:
        raise ClarificationInvalidSelection(
            f"clarification {persisted.clarification_id!r} has no selected "
            "candidate"
        )
    candidate = next(
        (
            item
            for item in persisted.candidates
            if item.candidate_id == resolution.selected_candidate_id
        ),
        None,
    )
    if candidate is None:  # pragma: no cover - membership proven above
        raise ClarificationInvalidSelection(
            f"selected candidate {resolution.selected_candidate_id!r} is not "
            f"part of clarification {persisted.clarification_id!r}"
        )
    authorization = context.services.authorization
    if authorization is None:
        raise ClarificationError(
            "no authorization service wired on runtime.services; "
            "the selected candidate cannot be authorized"
        )
    try:
        await _maybe_await(
            authorization.require_document(
                candidate.document_id, context.capability_runtime
            )
        )
    except ClarificationError:
        raise
    except PermissionError as error:
        raise ClarificationUnauthorized(
            persisted.clarification_id, candidate.document_id
        ) from error
    request = view["request"]
    known = tuple(request.known_documents)
    if not any(item.resource_id == candidate.ref_id for item in known):
        known = known + (
            KnownDocumentResource(
                resource_id=candidate.ref_id,
                document_id=candidate.document_id,
                source="ui_selection",
            ),
        )
    patched_refs = tuple(
        reference.model_copy(
            update={
                "resolution_status": "resolved",
                "resolved_document_id": candidate.document_id,
                "candidate_document_ids": (),
            }
        )
        if reference.ref_id == candidate.ref_id
        and reference.resolution_status != "resolved"
        else reference
        for reference in view["semantic"].document_refs
    )
    # NOTE: no binding pins are written here — key separation across ticks
    # is what keeps every channel single-writer: this update carries
    # request/semantic/clarification/route/analysis, and the re-driven
    # binding node (next tick, via this node's returned Command) carries
    # bindings. The re-driven binding node rebuilds
    # the draft through the selection-aware adapter (``ui_selection``
    # reconciliation), pins the answered ref, and leases before
    # checkpointing; the transient semantic patch above documents the
    # resolution and keeps this update self-consistent until the
    # finalizer rebuilds semantics from the pinned draft.
    return {
        "clarification": None,
        "route_decision": None,
        "query_analysis": None,
        "request": request.model_copy(update={"known_documents": known}),
        "semantic": view["semantic"].model_copy(
            update={"document_refs": patched_refs}
        ),
    }


# ---------------------------------------------------------------------------
# Graph composition
# ---------------------------------------------------------------------------

SUPERVISOR_V2_NODES: dict[str, Callable] = {
    "context": _wrap_node("context", context_node),
    "binding": _wrap_node("binding", binding_node),
    "semantic_finalizer": _wrap_node("semantic_finalizer", semantic_finalizer_node),
    "route": _wrap_node("route", route_node),
    "direct": _wrap_node("direct", direct_node),
    "clarify": _wrap_node("clarify", clarify_persist_node),
    "clarify_wait": _wrap_node("clarify_wait", clarify_wait_node),
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
    "complex_boundary": _wrap_node("complex_boundary", complex_unavailable_node),
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
    via ``_ROUTE_BRANCHES``) — never a node name — except the converted
    marker, which this branch sends straight to the finalizer (``"error"``
    branch) so a converted failure can never resurrect through
    re-derivation.
    """
    if _has_terminal_marker(state):
        return "error"
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


def _complex_branch(state: SupervisorV2State) -> str:
    """Route after the complex subgraph on the MERGED evidence evaluation (R4).

    ``sufficient`` continues into the existing ``synthesize`` node (which then
    flows ``synthesize -> ground -> finalizer``); every other verdict —
    ``insufficient``, ``contradictory``, ``needs_input``, or a missing
    evaluation (typed unavailable/failure path) — goes straight to the
    ``finalizer``. Synthesis and grounding are never duplicated inside the
    subgraph. Edges receive raw checkpoint state, so ``Mapping`` execution
    and evaluation values left by a real round-trip are read tolerantly; a
    missing execution fails closed to the finalizer, never to synthesis.
    """
    execution = state.get("execution")
    evaluation = None
    if execution is not None:
        if isinstance(execution, Mapping):
            evaluation = execution.get("evidence_evaluation")
        else:
            evaluation = getattr(execution, "evidence_evaluation", None)
    if evaluation is None:
        return "finalizer"
    if isinstance(evaluation, Mapping):
        status = evaluation.get("status")
    else:
        status = getattr(evaluation, "status", None)
    return "synthesize" if status == "sufficient" else "finalizer"


def _add_supervisor_edges(graph: StateGraph) -> None:
    graph.set_entry_point("context")
    graph.add_edge("context", "binding")
    graph.add_edge("binding", "semantic_finalizer")
    graph.add_edge("semantic_finalizer", "route")
    graph.add_conditional_edges(
        "route", _route_branch, {**_ROUTE_BRANCHES, "error": "finalizer"}
    )
    graph.add_edge("direct", "finalizer")
    graph.add_edge("clarify", "clarify_wait")
    # No static edge out of clarify_wait: every pass either suspends
    # (first pass) or returns a Command carrying its own goto (resume
    # paths), so onward navigation originates exactly once (see
    # clarify_wait_node).
    graph.add_edge("fast_plan", "execute")
    graph.add_edge("execute", "evaluate")
    graph.add_edge("evaluate", "synthesize")
    graph.add_edge("synthesize", "ground")
    graph.add_edge("ground", "finalizer")
    graph.add_conditional_edges(
        "complex_boundary",
        _complex_branch,
        {"synthesize": "synthesize", "finalizer": "finalizer"},
    )
    graph.add_edge("finalizer", END)


def create_supervisor_v2_graph(checkpointer: BaseCheckpointSaver) -> Any:
    """Compile the node-based supervisor v2 against ``checkpointer``.

    ``checkpointer`` is any LangGraph checkpoint saver (production: the
    lifespan-owned opened ``AsyncPostgresSaver``; tests: ``InMemorySaver``).
    """
    graph = StateGraph(SupervisorV2State, context_schema=GraphRuntimeContext)
    # Phase 3 replaces only the Phase-2 `complex_boundary` implementation
    # with the compiled complex-research subgraph. It is compiled WITHOUT
    # its own checkpointer so it inherits `checkpointer` (and therefore the
    # shadow run's isolated saver) rather than opening its own.
    # SUPERVISOR_V2_NODES keeps the Phase-2 unavailable entry untouched.
    complex_subgraph = build_complex_research_subgraph()
    complex_boundary_fn = _wrap_node(
        "complex_boundary", make_complex_boundary_node(complex_subgraph)
    )
    for name, fn in SUPERVISOR_V2_NODES.items():
        graph.add_node(name, complex_boundary_fn if name == "complex_boundary" else fn)
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
    The draft's contextualized form is the preprocessor's normalized
    CURRENT query (Task 8 fix round 1: ``conversation.summary`` must
    never substitute for the query — history reaches the turn only via
    ``recent_turns``/``active_entities``/``last_focus``; when no
    normalized form exists the adapter refuses rather than copying the
    raw query). Coreference mentions in the current query resolve only
    against current-turn already-authorized refs (no new identity is
    minted; true ambiguity becomes a ``BlockingAmbiguity``).

    ``ui_selection`` reconciliation: the clarify resume path records an
    answered selection as a ``KnownDocumentResource`` with
    ``source="ui_selection"`` and ``resource_id`` naming the answered
    semantic ``ref_id`` (see ``clarify_wait_node``). After preprocessing,
    every still-unresolved draft ref answered this way is projected to
    ``resolved`` with the selected canonical document id (candidates
    cleared). Only ``ui_selection`` sources reconcile — ``attachment`` /
    ``conversation`` resources are never auto-bound (T1 attachment
    exclusion), and an already-resolved ref is never overridden.
    Deterministic and read-only like the rest of the build.

    ``api_explicit`` projection (P0 factual retrieval): after UI
    reconciliation, every current-turn ``KnownDocumentResource`` with
    ``source="api_explicit"`` is projected into a resolved ``target``
    document reference whose ``ref_id`` is namespaced as
    ``api_explicit:<resource_id>``. The namespace cannot collide with
    preprocessor-generated ``r1``/``r2`` or clarification references, the
    projection never reads document IDs from raw query text or model
    output, and ordinary attachments stay unprojected candidates. The
    Binding Resolver remains the sole owner that pins the projected refs
    under the current ACL.

    T7/T8 LIFETIME REQUIREMENT (recorded): ``ui_selection`` attachments
    live on the thread-persisted ``RequestContext`` while v1 ref ids are
    positional per query, so a STALE attachment (previous turn's ``r1``)
    would force-resolve an unrelated ref of a later query and suppress its
    clarify question. T7/T8 MUST rebuild ``request`` (including
    ``known_documents``) per turn and never reuse a checkpointed request
    for a different query.
    """

    def __init__(
        self,
        *,
        preprocess: Callable[[str], Any],
        identity_resolver: Any = None,
        can_read_people: bool = False,
        session_factory: Callable[[], Any] | None = None,
        workspace_ids: tuple[Any, ...] = (),
    ) -> None:
        self._preprocess = preprocess
        # Phase 4C (Task 8 fix round 1): gates person-mention (``ông ấy``)
        # resolution for this turn. Default-deny; the ingress wires the
        # request's live value. Additive so existing sites are unaffected.
        self._can_read_people = bool(can_read_people)
        # Phase 4B (Task 6): the request-scoped v2
        # ``DocumentIdentityResolver`` (typed v1 ``resolve_candidates()``
        # wrapper). Optional and defaulting to ``None`` so existing
        # construction sites are unaffected. When set (with a session
        # factory + trusted workspace scope), ``build_draft`` resolves
        # still-unresolved draft refs through
        # ``adapters/semantic.py::resolve_draft_identities`` (final review
        # I2: the previously-latent resolver bridge is now live). The
        # existing v2 binding resolver stays the only revision-pin
        # authority. Never checkpointed; read via this attribute only.
        self.identity_resolver = identity_resolver
        self._session_factory = session_factory
        self._workspace_ids = tuple(workspace_ids or ())

    @staticmethod
    def reconcile_ui_selections(
        draft: SemanticDraft, request: RequestContext
    ) -> SemanticDraft:
        """Project answered ``ui_selection`` attachments onto draft refs."""
        selections: dict[str, UUID] = {}
        for known in request.known_documents:
            if known.source == "ui_selection":
                selections.setdefault(known.resource_id, known.document_id)
        if not selections:
            return draft
        refs = tuple(
            reference.model_copy(
                update={
                    "resolution_status": "resolved",
                    "resolved_document_id": selections[reference.ref_id],
                    "candidate_document_ids": (),
                }
            )
            if reference.resolution_status != "resolved"
            and reference.ref_id in selections
            else reference
            for reference in draft.document_refs
        )
        if refs == draft.document_refs:
            return draft
        return draft.model_copy(update={"document_refs": refs})

    #: Namespace prefix for API-explicit hard-scope reference IDs. The
    #: prefix guarantees projected IDs cannot collide with
    #: preprocessor-generated ``r1``/``r2`` or clarification references.
    API_EXPLICIT_REF_PREFIX = "api_explicit:"

    @staticmethod
    def project_api_explicit_targets(
        draft: SemanticDraft, request: RequestContext
    ) -> SemanticDraft:
        """Project current-turn ``api_explicit`` resources into target refs.

        Builds one resolved ``target`` ``DocumentReference`` per
        ``KnownDocumentResource`` with ``source == "api_explicit"``,
        namespaced as ``api_explicit:<resource_id>``. Only resources
        supplied for the current turn are projected; ordinary attachments,
        conversation resources, and ``ui_selection`` answers are never
        projected here, and an already-present namespaced ref is never
        duplicated. Deterministic and read-only like the rest of the build.
        """
        explicit = [
            known
            for known in request.known_documents
            if known.source == "api_explicit"
        ]
        if not explicit:
            return draft
        seen = {reference.ref_id for reference in draft.document_refs}
        additions: list[DocumentReference] = []
        for known in explicit:
            ref_id = (
                f"{DeterministicSemanticAdapter.API_EXPLICIT_REF_PREFIX}"
                f"{known.resource_id}"
            )
            if ref_id in seen:
                continue
            seen.add(ref_id)
            additions.append(
                DocumentReference(
                    ref_id=ref_id,
                    original_span=f"tài liệu {known.resource_id}",
                    normalized_reference=f"tài liệu {known.resource_id}",
                    requested_role="target",
                    revision_requirement=None,
                    resolution_status="resolved",
                    resolved_document_id=known.document_id,
                    candidate_document_ids=(),
                )
            )
        if not additions:
            return draft
        return draft.model_copy(
            update={"document_refs": draft.document_refs + tuple(additions)}
        )

    async def build_draft(
        self, request: RequestContext, conversation: ConversationContext
    ) -> SemanticDraft:
        from .v2.adapters.semantic import SemanticAdapterError, draft_from_preprocessing
        from .v2.semantic.discourse import (
            merge_ambiguities,
            merge_coreferences,
            resolve_coreferences,
        )

        result = await _maybe_await(self._preprocess(request.original_query))
        try:
            # Task 8 fix round 1 (Critical-1): never substitute the
            # session summary for the current query. ``conversation``
            # still flows into the turn via the checkpointed
            # ``ConversationContext`` (recent turns/entities/focus).
            draft = draft_from_preprocessing(result)
        except Exception as exc:
            raise SemanticAdapterError(
                f"v1 preprocessing output cannot become a v2 draft: {exc}"
            ) from exc
        reconciled = self.reconcile_ui_selections(draft, request)
        projected = self.project_api_explicit_targets(reconciled, request)
        # Final review I2: when the request-scoped identity resolver is
        # wired (production ingress), resolve still-unresolved draft refs
        # through the v1 ``resolve_candidates()`` pipeline before the
        # coreference pass below. Already-resolved refs (preprocessor/
        # ui_selection/api_explicit) pass through untouched; the existing
        # v2 binding resolver stays the only revision-pin authority.
        if (
            self.identity_resolver is not None
            and self._session_factory is not None
            and self._workspace_ids
            and any(
                reference.resolution_status != "resolved"
                for reference in projected.document_refs
            )
        ):
            from .v2.adapters.semantic import resolve_draft_identities

            async with self._session_factory() as db:
                projected = await resolve_draft_identities(
                    projected,
                    question=request.original_query,
                    identity_resolver=self.identity_resolver,
                    workspace_ids=self._workspace_ids,
                    db=db,
                    can_read_people=self._can_read_people,
                )
        # Task 8 fix round 1 (Important-1): the production coreference
        # call site. Mentions (``văn bản này``/``điều này``/``ông ấy``/
        # ``file thứ hai``) resolve only to current-turn refs that are
        # already authorized (resolved preprocessor/ui/api refs); the
        # allowed scope is those resolved document IDs — never workspace
        # IDs, never history identity. No binding is minted here.
        allowed_ids = tuple(
            reference.resolved_document_id
            for reference in projected.document_refs
            if reference.resolution_status == "resolved"
            and reference.resolved_document_id is not None
        )
        corefs, coref_ambiguities = resolve_coreferences(
            request.original_query,
            document_refs=projected.document_refs,
            person_refs=projected.person_refs,
            section_refs=projected.section_refs,
            allowed_document_ids=allowed_ids,
            can_read_people=self._can_read_people,
        )
        if not corefs and not coref_ambiguities:
            return projected
        # Merge (never blind-concat): a re-run over the same turn must
        # stay idempotent under the frozen uniqueness rules.
        return projected.model_copy(
            update={
                "coreferences": merge_coreferences(
                    projected.coreferences, corefs
                ),
                "preliminary_ambiguities": merge_ambiguities(
                    projected.preliminary_ambiguities, coref_ambiguities
                ),
            }
        )


class V1BindingResolver:
    """Request-scoped resolver turning draft refs into pinned sets (T6, D5).

    Wraps ``adapters/document.py::resolve_document_bindings`` behind the
    node-facing ``resolve(document_refs, capability_runtime) ->
    DocumentBindingSet`` contract: resolves EACH reference across EVERY
    workspace in the trusted runtime (never collapsed to one) with
    first-authorized-match wins, opening a dedicated session per attempt
    through ``session_factory`` (default: the app's
    ``async_session_maker``). A reference fails only after no workspace can
    resolve it, so references owned by different workspaces union instead of
    failing closed on the first foreign workspace. Empty references resolve
    to the empty set without opening any session; an empty workspace scope
    fails closed.
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
        from .v2.adapters.document import (
            DocumentAdapterError,
            resolve_document_bindings,
        )
        from .v2.persistence.document_views import RevisionNotReady

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
        for reference in refs:
            last_error: Exception | None = None
            pinned = False
            for workspace_id in workspaces:
                async with open_session() as db:
                    try:
                        resolved = await resolve_document_bindings(
                            db,
                            (reference,),
                            workspace_id=workspace_id,
                            default_role=self._default_role,
                        )
                    except (DocumentAdapterError, RevisionNotReady) as exc:
                        # Per-reference union: a workspace that cannot
                        # authorize this reference (foreign workspace,
                        # tombstone, unpublished revision, revision/document
                        # mismatch) only disqualifies THAT workspace for THIS
                        # reference. Keep the last error and try the next
                        # trusted workspace; the reference fails only after no
                        # workspace resolves it. Unexpected errors (DB/IO)
                        # propagate immediately instead of being masked as a
                        # workspace miss.
                        last_error = exc
                        continue
                for binding in resolved.binding_set.bindings:
                    bindings.setdefault(binding.binding_id, binding)
                    pinned = True
                for relation in resolved.binding_set.revision_requirement_refs:
                    relations.setdefault(
                        (relation.binding_id, relation.ref_id), relation
                    )
                if pinned or not resolved.binding_set.bindings:
                    # Bound here (first authorized match wins), or the
                    # reference binds nothing by semantics (unresolved /
                    # ambiguous refs are owned by clarification) — either way
                    # do not probe further workspaces for this reference.
                    last_error = None
                    break
            if last_error is not None:
                raise last_error
        return DocumentBindingSet(
            bindings=tuple(bindings.values()),
            revision_requirement_refs=tuple(relations.values()),
        )


@dataclass(frozen=True)
class PeopleLookupMatch:
    """One distinct person behind a people-identifier query (internal).

    ``fields`` is the minimized canonical mapping the capability persists
    as governed evidence (task-required keys only: ``name``, the queried
    identifier such as ``phone``, and the ``source`` schema label — never
    the raw Mongo record). ``record_id`` is the stable non-PII handle
    (hash of canonical identity + source). ``required_fields`` names the
    keys ``fields`` must carry.
    """

    record_id: str
    fields: Mapping[str, object]
    required_fields: tuple[str, ...] = ("name", "phone", "source")


#: Heterogeneous Mongo name aliases across schemas (bhxh ``hoTen``, vnvc
#: ``fullName``, ...). Public canonicalization: the private
#: ``mongo_people_service`` extractors are never imported here.
PEOPLE_NAME_FIELDS: tuple[str, ...] = (
    "name",
    "hoTen",
    "fullName",
    "TenHoiVien",
    "HO_TEN",
    "ho_ten",
    "tenKhachHang",
)

#: Heterogeneous Mongo phone aliases across schemas (bhxh ``soDienThoai``,
#: vnvc ``mobile``, ...).
PEOPLE_PHONE_FIELDS: tuple[str, ...] = (
    "phone",
    "soDienThoai",
    "mobile",
    "SoDienThoai",
    "DIEN_THOAI_ME",
    "so_dien_thoai",
    "dienThoai",
)

#: ``people_intent_from_query`` intent → v1 ``mongo_people_service`` search.
PEOPLE_INTENT_SEARCH: dict[str, str] = {
    "mongo_search_phone": "search_by_phone",
    "mongo_search_cccd": "search_by_cccd",
    "mongo_search_bhxh": "search_by_bhxh",
    "mongo_search_name": "search_by_name",
    "mongo_search_advanced": "search_by_advanced",
}


def canonicalize_person_record(
    raw: Mapping[str, object],
    *,
    queried_phone: str = "",
    schema: str = "",
) -> dict[str, object]:
    """Normalize one heterogeneous Mongo person doc to task-required fields.

    Public helper (no private ``mongo_people_service`` imports): picks the
    first present name/phone alias, prefers the normalized queried phone
    when the record carries it, and labels the source schema. Only
    ``name``/``phone``/``source`` survive — unrelated DOB/address/CCCD/BHXH
    fields are dropped here and never reach evidence.
    """
    source = schema or str(raw.get("_source_schema", "") or "unknown")
    name = ""
    for field in PEOPLE_NAME_FIELDS:
        value = raw.get(field)
        if value not in (None, "", "None"):
            name = str(value).strip()
            break
    phone = ""
    if queried_phone:
        normalized = re.sub(r"[\s\-\.]+", "", queried_phone)
        for field in PEOPLE_PHONE_FIELDS:
            value = raw.get(field)
            if value not in (None, "", "None") and re.sub(
                r"[\s\-\.]+", "", str(value)
            ) == normalized:
                phone = normalized
                break
    if not phone:
        for field in PEOPLE_PHONE_FIELDS:
            value = raw.get(field)
            if value not in (None, "", "None"):
                phone = str(value).strip()
                break
    return {"name": name, "phone": phone, "source": source}


def stable_people_record_id(
    *, name: str, phone: str, source: str, group: str = "", dob: str = ""
) -> str:
    """Stable non-PII record handle: hash of canonical identity + source.

    ``group`` (the exact Mongo ``_person_group``) is mixed into the digest
    whenever present so distinct Mongo groups sharing one phone/name never
    collapse to one handle downstream; when the group is absent, the
    available DOB widens the digest instead (M5) — DOB lives ONLY in this
    one-way hash, never in persisted evidence. Records with neither keep
    the historical ``name/phone/source`` digest. The digest stays
    deterministic, source-scoped, and free of raw identity substrings.
    """
    name_part = name.strip().lower()
    phone_part = phone.strip()
    source_part = source.strip().lower()
    material = f"{name_part}\x00{phone_part}\x00{source_part}"
    if group.strip():
        material += f"\x00group:{group.strip()}"
    elif dob.strip():
        material += f"\x00dob:{dob.strip()}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"p_{digest}"


#: Heterogeneous Mongo DOB aliases across schemas (bhxh ``ngaySinhHienThi``,
#: lg ``NgaySinh``, vacxin ``NGAY_SINH``, cv19 ``namsinh``, vnvc ``fullNam``
#: — from the ``advanced`` field map in ``mongo_searchable_map``). Used ONLY
#: in the transient dedupe key (never persisted, never in evidence).
PEOPLE_DOB_FIELDS: tuple[str, ...] = (
    "ngaySinhHienThi",
    "NgaySinh",
    "NGAY_SINH",
    "namsinh",
    "fullNam",
)

#: Placeholder display identity for a Mongo match carrying no name (e.g. the
#: ``uids`` phone schema, whose field map has no name field). Non-sensitive
#: and constant: the exact queried phone + source stay in the evidence
#: content, and distinctness comes from ``_person_group`` (see the ``group``
#: salt in :func:`stable_people_record_id`), never from invented identity.
UNKNOWN_PEOPLE_DISPLAY_NAME = "Không rõ tên"


def _people_group_key(
    doc: Mapping[str, object], canonical: Mapping[str, object]
) -> str:
    """Transient dedupe key mirroring Mongo's own grouping.

    The exact ``_person_group`` wins when present; strong identifiers
    (cccd/bhxh digits) next; otherwise ``name|(dob or phone)`` exactly like
    ``mongo_people_service._identity_key`` (DOB widens the key ONLY — it is
    never persisted and never reaches evidence). Name-less docs fall back to
    the Mongo ``_id`` so distinct group-less rows are never collapsed.
    """
    group = doc.get("_person_group")
    if group is not None and str(group).strip() != "":
        return f"group:{group}"
    for key in ("cccd", "bhxh"):
        value = doc.get(key)
        if value not in (None, "", "None"):
            digits = re.sub(r"\D", "", str(value))
            if digits:
                return f"{key}:{digits}"
    name = str(canonical.get("name", "")).lower().strip()
    phone = str(canonical.get("phone", "")).strip()
    if name:
        dob = ""
        for field in PEOPLE_DOB_FIELDS:
            value = doc.get(field)
            if value not in (None, "", "None"):
                dob = str(value).strip()
                break
        return f"np:{name}|{dob or phone}"
    return f"id:{doc.get('_id', '')}"


async def _invoke_people_search(search: Any, query: str, *, limit: int) -> Any:
    """Invoke one v1 people search without guessing its signature (C1).

    Real ``search_by_cccd``/``search_by_bhxh`` take no ``limit`` while
    ``search_by_phone``/``search_by_name``/``search_by_advanced`` do, so an
    unconditional ``limit=`` kwarg is a production ``TypeError`` for
    CCCD/BHXH queries. The ``limit`` kwarg is passed ONLY when the resolved
    callable declares it (or accepts ``**kwargs``); this is decided with
    :func:`inspect.signature` BEFORE the call — an inner ``TypeError`` is
    never caught as a signature mismatch. An uninspectable callable is
    invoked with the query only (safe: every real search has a ``limit``
    default where one exists).
    """
    try:
        parameters = inspect.signature(search).parameters
    except (TypeError, ValueError):
        return search(query)
    accepts_limit = "limit" in parameters or any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )
    if accepts_limit:
        return search(query, limit=limit)
    return search(query)


async def _collect_search_persons(produced: Any) -> list[Mapping[str, object]]:
    """Drain one v1 search result (async-gen or single mapping) to persons.

    Shared single reader for every intent (M7): previously duplicated across
    ``_lookup_many``/``_lookup_many_advanced``. Malformed v1 output fails
    closed with :class:`V1ServiceUnavailable`.
    """
    persons: list[Mapping[str, object]] = []
    if inspect.isasyncgen(produced):
        async for result in produced:
            if not isinstance(result, Mapping):
                raise V1ServiceUnavailable(
                    "v1 people lookup yielded a non-mapping result"
                )
            if result.get("error"):
                raise V1ServiceUnavailable(
                    f"v1 people lookup is unavailable: {result.get('error')}"
                )
            if result.get("found") and result.get("persons"):
                batch = result["persons"]
                if not isinstance(batch, (list, tuple)):
                    raise V1ServiceUnavailable(
                        "v1 people lookup yielded a non-sequence persons"
                    )
                for person in batch:
                    if not isinstance(person, Mapping):
                        raise V1ServiceUnavailable(
                            "v1 people lookup yielded a non-mapping person"
                        )
                    persons.append(person)
    else:
        single = await _maybe_await(produced)
        if single is not None:
            if not isinstance(single, Mapping):
                raise V1ServiceUnavailable(
                    "v1 people lookup returned a non-mapping"
                )
            persons.append(single)
    return persons


def _build_people_matches(
    persons: list[Mapping[str, object]], *, queried_phone: str = ""
) -> list[PeopleLookupMatch]:
    """Canonicalize + dedupe raw Mongo persons into governed matches.

    Shared single builder for every intent (M7). A match without a name
    (``uids`` phone schema) is NEVER dropped (I3): it keeps the placeholder
    :data:`UNKNOWN_PEOPLE_DISPLAY_NAME` with the exact queried phone and
    source, minting a stable non-PII record id salted by ``_person_group``.
    Only task-required keys (``name``/``phone``/``source``) survive — DOB,
    address, CCCD/BHXH numbers and ``_person_group`` itself never enter the
    persisted mapping.
    """
    matches: list[PeopleLookupMatch] = []
    seen: set[str] = set()
    for doc in persons:
        canonical = canonicalize_person_record(
            doc,
            queried_phone=queried_phone,
            schema=str(doc.get("_source_schema", "") or ""),
        )
        name = str(canonical["name"]).strip()
        if not name:
            # I3: name-less Mongo match — preserve with an explicit
            # non-sensitive placeholder instead of answering ``not_found``.
            name = UNKNOWN_PEOPLE_DISPLAY_NAME
        key = _people_group_key(doc, {**canonical, "name": name})
        if key in seen:
            continue
        seen.add(key)
        group_token = doc.get("_person_group")
        group_salt = (
            "" if group_token is None or str(group_token).strip() == ""
            else str(group_token).strip()
        )
        dob_salt = ""
        if not group_salt:
            for field in PEOPLE_DOB_FIELDS:
                value = doc.get(field)
                if value not in (None, "", "None"):
                    dob_salt = str(value).strip()
                    break
        record_id = stable_people_record_id(
            name=name,
            phone=str(canonical["phone"]),
            source=str(canonical["source"]),
            group=group_salt,
            dob=dob_salt,
        )
        matches.append(
            PeopleLookupMatch(
                record_id=record_id,
                fields={
                    "name": name,
                    "phone": canonical["phone"],
                    "source": canonical["source"],
                },
                required_fields=("name", "phone", "source"),
            )
        )
    return matches


class V1PeopleMultiMatchAdapter:
    """Named v1 multi-match seam: intent → signature-safe search → matches.

    Owns the single unified reader (M7): per-intent dispatch through
    :func:`_invoke_people_search` (``limit`` is passed only when the resolved
    callable declares it — C1), collection through
    :func:`_collect_search_persons`, and match building through
    :func:`_build_people_matches` (name-less matches preserved — I3,
    DOB-widened dedupe — M5). The public :meth:`lookup_many` is the explicit
    seam the capability is injected with (M6); :class:`V1PeopleLookupService`
    keeps the frozen ``lookup``-only public surface and reaches this reader
    through its private delegating ``_lookup_many``.
    """

    def __init__(
        self,
        *,
        lookup: Any = None,
        limit: int = 10,
        phone_lookup: Any = None,
        cccd_lookup: Any = None,
        bhxh_lookup: Any = None,
        name_lookup: Any = None,
    ) -> None:
        self._lookup = lookup
        self._limit = limit
        self._overrides: dict[str, Any] = {}
        if phone_lookup is not None:
            self._overrides["search_by_phone"] = phone_lookup
        if cccd_lookup is not None:
            self._overrides["search_by_cccd"] = cccd_lookup
        if bhxh_lookup is not None:
            self._overrides["search_by_bhxh"] = bhxh_lookup
        if name_lookup is not None:
            self._overrides["search_by_name"] = name_lookup

    def _search_fn(self, search_name: str) -> Any:
        if search_name in self._overrides:
            return self._overrides[search_name]
        if search_name == "search_by_name" and self._lookup is not None:
            return self._lookup
        return _v1_attr("app.services.people.mongo_people_service", search_name)

    def _intent_search_name(self, query: str) -> str:
        from app.prompts.agents.supervisor_scope import people_intent_from_query

        try:
            intent = people_intent_from_query(query)
        except Exception:
            intent = "mongo_search_advanced"
        return PEOPLE_INTENT_SEARCH.get(intent, "search_by_name")

    async def lookup_many(self, query: str) -> list[PeopleLookupMatch]:
        """All distinct people behind ``query`` as minimized matches."""
        from app.prompts.agents.supervisor_scope import people_intent_from_query

        search_name = self._intent_search_name(query)
        advanced = search_name == "search_by_advanced"
        # Advanced (unparsed) queries keep the legacy name-search behavior.
        search = self._search_fn("search_by_name" if advanced else search_name)
        produced = await _invoke_people_search(search, query, limit=self._limit)
        persons = await _collect_search_persons(produced)
        if not persons:
            return []
        queried_phone = ""
        if not advanced:
            try:
                if people_intent_from_query(query) == "mongo_search_phone":
                    digits = re.findall(r"\d+", query)
                    tens = [d for d in digits if len(d) == 10]
                    if tens:
                        queried_phone = tens[0]
            except Exception:
                queried_phone = ""
        return _build_people_matches(persons, queried_phone=queried_phone)


class V1PeopleLookupService:
    """v1-backed People lookup: mongo records → governed matches (T6).

    The search dispatched matches the deterministic people intent
    (``people_intent_from_query``): phone/CCCD/BHXH/name queries call
    ``search_by_phone``/``search_by_cccd``/``search_by_bhxh``/``search_by_name``
    respectively instead of always running a name search. :meth:`_lookup_many`
    returns one :class:`PeopleLookupMatch` per distinct person (grouped
    records dedupe to one match; never first-only); :meth:`lookup` keeps the
    legacy first-match mapping contract for read-only callers. Malformed v1
    output fails closed so the capability maps it to a typed error instead
    of checkpointing a guess.
    """

    def __init__(
        self,
        *,
        lookup: Any = None,
        limit: int = 10,
        phone_lookup: Any = None,
        cccd_lookup: Any = None,
        bhxh_lookup: Any = None,
        name_lookup: Any = None,
    ) -> None:
        self._lookup = lookup
        self._limit = limit
        # The shared multi-match reader owns intent dispatch + dedupe. This
        # service keeps the frozen public surface (``lookup`` only); the
        # private ``_lookup_many`` below only forwards for pre-existing
        # callers, and the capability uses an injected adapter instead.
        self._multi_match = V1PeopleMultiMatchAdapter(
            lookup=lookup,
            limit=limit,
            phone_lookup=phone_lookup,
            cccd_lookup=cccd_lookup,
            bhxh_lookup=bhxh_lookup,
            name_lookup=name_lookup,
        )

    async def _lookup_many(self, query: str) -> list[PeopleLookupMatch]:
        """All distinct people behind ``query`` as minimized matches.

        Private delegating reader by design: the shadow R64 surface pins the
        public surface to ``lookup`` only, so the unified logic lives on
        :class:`V1PeopleMultiMatchAdapter` (the capability's injected seam)
        and this method only forwards for pre-existing callers. Legacy
        single-``lookup`` services are unaffected.
        """
        return await self._multi_match.lookup_many(query)

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

    The governed People→Document scalar (``person_identifier``, R92) is
    consumed as an authorized query refinement: the server-materialized
    scalar is appended to the task query and the EXISTING workspace-scoped
    hybrid search runs unchanged. ``workspace_ids`` are never widened, no
    graph/supervisor state is read, and a blank/non-string scalar fails
    closed (``V1ServiceUnavailable`` → typed capability error) instead of
    silently running an unrefined search.
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

        # R92: consume the governed scalar as an authorized refinement of
        # the task query. The scalar is server-materialized after a
        # successful people.lookup (never planner-supplied); combining it
        # into the query text keeps the v1 retrieval call inside the
        # UNCHANGED authorized workspace scope.
        effective_query = query
        if person_identifier is not None:
            if not isinstance(person_identifier, str) or not person_identifier.strip():
                raise V1ServiceUnavailable(
                    "document.search carries a blank people dependency "
                    "scalar; refusing to run an unrefined search"
                )
            scalar = person_identifier.strip()
            effective_query = (
                f"{query} {scalar}" if query and query.strip() else scalar
            )

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
                effective_query,
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
                # Task 5: pin through the workspace-scoped lookup (union over
                # the trusted runtime scope, first authorized match wins), so
                # a foreign-workspace id or a tombstoned document can never be
                # pinned even if the v1 search above returned it. The unscoped
                # loader is deliberately not used here.
                identity = None
                for workspace_id in workspace_ids:
                    try:
                        identity = await document_views.load_current_revision_identity_for_workspace(
                            db, document_id, workspace_id
                        )
                    except document_views.RevisionNotReady:
                        continue
                    if identity is not None:
                        break
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


class V1RevisionAwareRetrievalService:
    """Live revision-manifest chunk retrieval over exact revision namespaces.

    P0 Task 5 server-side dependency behind ``DocumentRetrieveCapability``
    (implements the ``DocumentRetrievalService`` port: ``retrieve(query, *,
    top_k, allowed_targets, workspace_ids)``). For every admitted revision it
    loads the exact manifest identity via ``persistence.document_views``
    (namespace, model hash, dimension, vector artifact version — read from
    that revision's own build manifest, never from current configuration)
    and queries ONLY that namespace with hard ``revision_id`` + ``document_id``
    filters through the existing HTTP embed/rerank provider.

    Fail-closed contract (never a current-config namespace fallback):

    - scoped: each pinned target's revision is resolved workspace-scoped
      (union over ``workspace_ids``, first authorized match wins). A missing
      or unpublished manifest, a foreign-workspace or tombstoned document,
      an incompatible vector artifact version, or a manifest that does not
      match the pinned ``(document_id, document_revision)`` raises
      ``V1ServiceUnavailable`` — the capability maps it to a typed
      ``dependency_error``, never to success with wrong-revision chunks.
    - unscoped: candidates come from the workspace-scoped discovery port and
      each is pinned the same workspace-scoped way; unresolvable candidates
      (foreign, tombstoned, legacy, unpublished, incompatible) are skipped.
    - provider hits are re-verified per hit (revision, document, workspace,
      vector-id shape, non-blank content); anything mismatched or malformed
      is dropped.
    - ``target_id`` hints are never minted here (always ``None``): the
      capability matches hintless chunks by authoritative identity.

    Ports (``discover`` / ``embed_query`` / ``query_namespace`` / ``rerank``)
    are injectable for tests; the defaults are lazy v1/HTTP backings resolved
    at call time so this module stays import-light. No graph/supervisor state
    is read and no scope field is added to ``CapabilityRuntimeContext``.
    """

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] | None = None,
        discover: Any = None,
        embed_query: Any = None,
        query_namespace: Any = None,
        rerank: Any = None,
        discovery_top_k: int = 10,
        over_fetch: int = 3,
    ) -> None:
        self._session_factory = session_factory
        self._discover = discover
        self._embed_query = embed_query
        self._query_namespace = query_namespace
        self._rerank = rerank
        self._discovery_top_k = discovery_top_k
        self._over_fetch = over_fetch

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        allowed_targets: tuple[Any, ...],
        workspace_ids: tuple[UUID, ...],
    ) -> Sequence[RevisionRetrievedChunk]:
        from .v2.contracts.locators import ChunkRangeLocator
        from .v2.persistence import document_views

        if not isinstance(query, str) or not query.strip():
            raise V1ServiceUnavailable(
                "revision retrieval needs a non-blank query"
            )
        workspaces = tuple(workspace_ids or ())
        if not workspaces:
            raise V1ServiceUnavailable(
                "revision retrieval needs an authenticated workspace scope"
            )
        try:
            limit = max(1, min(int(top_k), 20))
        except (TypeError, ValueError) as exc:
            raise V1ServiceUnavailable(
                f"revision retrieval got an unusable top_k {top_k!r}"
            ) from exc
        async with self._open_session() as db:
            if tuple(allowed_targets or ()):
                admitted = await self._load_scoped_identities(
                    db, document_views, tuple(allowed_targets), workspaces
                )
            else:
                admitted = await self._discover_identities(
                    db, document_views, query.strip(), workspaces
                )
        if not admitted:
            return ()
        embedding = await self._embed(query.strip())
        scored: list[tuple[float, str, Any, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for identity in admitted:
            for content, chunk_id, distance in await self._search_identity(
                identity, embedding, limit, workspaces
            ):
                key = (
                    str(identity.revision_id),
                    str(identity.document_id),
                    chunk_id,
                )
                if key in seen:
                    continue
                seen.add(key)
                scored.append((distance, content, identity, chunk_id))
        if not scored:
            return ()
        order = await self._rank(query.strip(), [c for _, c, _, _ in scored])
        chunks: list[RevisionRetrievedChunk] = []
        for rank, (distance, content, identity, chunk_id) in enumerate(scored):
            score = order.get(rank, -float(distance))
            chunks.append(
                RevisionRetrievedChunk(
                    document_id=identity.document_id,
                    document_revision=str(identity.revision_id),
                    locator=ChunkRangeLocator(
                        kind="chunk_range", start=chunk_id, end=chunk_id
                    ),
                    content=content,
                    score=float(score),
                    target_id=None,
                )
            )
        if order:
            chunks.sort(key=lambda c: c.score, reverse=True)
        return tuple(chunks[:limit])

    def _open_session(self) -> Any:
        factory = self._session_factory
        if factory is None:
            from app.core.database import async_session_maker

            factory = async_session_maker
        return factory()

    async def _load_scoped_identities(
        self,
        db: Any,
        document_views: Any,
        allowed_targets: tuple[Any, ...],
        workspaces: tuple[UUID, ...],
    ) -> list[Any]:
        """Pin every planned target to its exact manifest; fail closed."""
        admitted: list[Any] = []
        for target in allowed_targets:
            binding = target.document
            try:
                revision_id = (
                    binding.document_revision
                    if isinstance(binding.document_revision, UUID)
                    else UUID(str(binding.document_revision))
                )
            except (ValueError, TypeError, AttributeError) as exc:
                raise V1ServiceUnavailable(
                    "pinned target carries an unparsable revision; "
                    "refusing to retrieve"
                ) from exc
            identity = None
            last_error: Exception | None = None
            for workspace_id in workspaces:
                try:
                    identity = await document_views.load_revision_identity_for_workspace(
                        db, revision_id, workspace_id, require_vectors=True
                    )
                except document_views.RevisionNotReady as exc:
                    last_error = exc
                    continue
                break
            if identity is None:
                raise V1ServiceUnavailable(
                    f"pinned revision {revision_id} has no usable manifest "
                    f"in this workspace scope ({last_error}); refusing to "
                    "retrieve"
                )
            if (
                identity.document_id != binding.document_id
                or str(identity.revision_id) != str(revision_id)
            ):
                raise V1ServiceUnavailable(
                    "revision manifest does not match the pinned target "
                    "(stale pin?); refusing to retrieve"
                )
            self._check_vector_manifest(document_views, identity)
            admitted.append(identity)
        return admitted

    async def _discover_identities(
        self,
        db: Any,
        document_views: Any,
        query: str,
        workspaces: tuple[UUID, ...],
    ) -> list[Any]:
        """Workspace-scoped discovery, then the same manifest pin per hit."""
        discover = self._discover
        if discover is None:
            search_fn = _v1_attr(
                "app.services.agent.tools", "search_documents"
            )

            async def discover(
                found_query: str,
                found_top_k: int,
                found_workspaces: list,
                found_db: Any,
            ) -> list[UUID]:
                found = await _maybe_await(
                    search_fn(
                        found_query,
                        found_top_k,
                        list(found_workspaces),
                        set(),
                        found_db,
                    )
                )
                sources = (
                    found.get("sources", ())
                    if isinstance(found, Mapping)
                    else ()
                )
                document_ids: list[UUID] = []
                seen: set[UUID] = set()
                for source in sources:
                    raw_id = (
                        source.get("document_id")
                        if isinstance(source, Mapping)
                        else getattr(source, "document_id", None)
                    )
                    try:
                        document_id = (
                            raw_id
                            if isinstance(raw_id, UUID)
                            else UUID(str(raw_id))
                        )
                    except (ValueError, TypeError, AttributeError):
                        continue
                    if document_id not in seen:
                        seen.add(document_id)
                        document_ids.append(document_id)
                return document_ids

        try:
            document_ids = await _maybe_await(
                discover(query, self._discovery_top_k, list(workspaces), db)
            )
        except V1ServiceUnavailable:
            raise
        except Exception as exc:
            raise V1ServiceUnavailable(
                f"revision discovery is unavailable: {exc}"
            ) from exc
        admitted: list[Any] = []
        seen_docs: set[UUID] = set()
        for raw_id in document_ids or ():
            try:
                document_id = (
                    raw_id if isinstance(raw_id, UUID) else UUID(str(raw_id))
                )
            except (ValueError, TypeError, AttributeError):
                continue
            if document_id in seen_docs:
                continue
            seen_docs.add(document_id)
            identity = None
            for workspace_id in workspaces:
                try:
                    identity = await document_views.load_current_revision_identity_for_workspace(
                        db, document_id, workspace_id, require_vectors=True
                    )
                except document_views.RevisionNotReady:
                    continue
                if identity is not None:
                    break
            if identity is None:
                continue
            try:
                self._check_vector_manifest(document_views, identity)
            except V1ServiceUnavailable:
                continue
            admitted.append(identity)
        return admitted

    @staticmethod
    def _check_vector_manifest(document_views: Any, identity: Any) -> None:
        """The manifest must pin a complete, current vector identity."""
        if not identity.vectors_available or not identity.embedding_namespace:
            raise V1ServiceUnavailable(
                "published revision has no complete vector manifest; "
                "refusing to fall back to a current-config namespace"
            )
        if (
            identity.vector_artifact_version
            != document_views.VECTOR_ARTIFACT_VERSION
        ):
            raise V1ServiceUnavailable(
                f"revision vector artifact "
                f"{identity.vector_artifact_version!r} is incompatible; "
                "refusing to retrieve"
            )

    async def _embed(self, query: str) -> list[float]:
        embed = self._embed_query
        if embed is None:
            async def embed(text: str) -> list[float]:
                def _run() -> list[float]:
                    from app.services.embedding.embedder import (
                        get_embedding_service,
                    )

                    return get_embedding_service().embed_query(text)

                return await asyncio.to_thread(_run)

        try:
            vector = await _maybe_await(embed(query))
        except V1ServiceUnavailable:
            raise
        except Exception as exc:
            raise V1ServiceUnavailable(
                f"query embedding is unavailable: {exc}"
            ) from exc
        if not isinstance(vector, (list, tuple)) or not vector:
            raise V1ServiceUnavailable("query embedding returned no vector")
        return list(vector)

    async def _search_identity(
        self,
        identity: Any,
        embedding: list[float],
        limit: int,
        workspaces: tuple[UUID, ...],
    ) -> list[tuple[str, str, float]]:
        """Query one exact manifest namespace with hard revision filters."""
        from .v2.persistence import document_views

        query_ns = self._query_namespace
        if query_ns is None:
            query_ns = _default_revision_namespace_query
        where = {
            "$and": [
                {"revision_id": str(identity.revision_id)},
                {"document_id": str(identity.document_id)},
            ]
        }
        try:
            raw = await _maybe_await(
                query_ns(
                    identity.embedding_namespace,
                    embedding,
                    limit * max(1, self._over_fetch),
                    where,
                )
            )
        except V1ServiceUnavailable:
            raise
        except Exception as exc:
            raise V1ServiceUnavailable(
                f"revision namespace query failed: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise V1ServiceUnavailable("revision query returned no mapping")
        ids = raw.get("ids", []) or []
        documents = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []
        distances = raw.get("distances", []) or []
        if not (
            isinstance(ids, (list, tuple))
            and isinstance(documents, (list, tuple))
            and isinstance(metadatas, (list, tuple))
        ) or len(ids) != len(documents):
            raise V1ServiceUnavailable("revision query returned malformed hits")
        admitted: list[tuple[str, str, float]] = []
        for position, vector_id in enumerate(ids):
            content = documents[position] if position < len(documents) else None
            metadata = metadatas[position] if position < len(metadatas) else None
            distance = (
                distances[position] if position < len(distances) else 0.0
            )
            hit = self._coerce_hit(
                document_views, identity, vector_id, content, metadata,
                workspaces,
            )
            if hit is None:
                continue
            hit_content, chunk_id = hit
            try:
                score_distance = float(distance)
            except (TypeError, ValueError):
                score_distance = 0.0
            admitted.append((hit_content, chunk_id, score_distance))
        return admitted

    def _coerce_hit(
        self,
        document_views: Any,
        identity: Any,
        vector_id: Any,
        content: Any,
        metadata: Any,
        workspaces: tuple[UUID, ...],
    ) -> tuple[str, str] | None:
        """Admit one provider hit only on exact manifest agreement."""
        if not isinstance(content, str) or not content.strip():
            return None
        if not isinstance(metadata, Mapping):
            return None
        if str(metadata.get("revision_id")) != str(identity.revision_id):
            return None
        if str(metadata.get("document_id")) != str(identity.document_id):
            return None
        if str(metadata.get("workspace_id")) not in {
            str(item) for item in workspaces
        }:
            return None
        parsed = document_views.parse_revision_vector_id(
            vector_id if isinstance(vector_id, str) else ""
        )
        if parsed is None or parsed[0] != identity.revision_id:
            return None
        chunk_id = metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            chunk_id = vector_id if isinstance(vector_id, str) else ""
        if not chunk_id:
            return None
        return content, chunk_id

    async def _rank(
        self, query: str, texts: list[str]
    ) -> dict[int, float]:
        """Rerank texts; malformed output falls back to vector order."""
        rerank = self._rerank
        if rerank is None:
            async def rerank(rerank_query: str, rerank_texts: list[str]) -> Any:
                def _run() -> Any:
                    from app.services.retrieval.reranker import (
                        get_reranker_service,
                    )

                    return [
                        (item.index, item.score)
                        for item in get_reranker_service().rerank(
                            rerank_query, list(rerank_texts), top_k=len(rerank_texts)
                        )
                    ]

                return await asyncio.to_thread(_run)

        try:
            ranked = await _maybe_await(rerank(query, texts))
        except Exception:
            logger.warning(
                "revision rerank unavailable; keeping vector order",
                exc_info=True,
            )
            return {}
        order: dict[int, float] = {}
        try:
            entries = list(ranked or ())
        except TypeError:
            return {}
        for entry in entries:
            try:
                index, score = entry
                index = int(index)
                score = float(score)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(texts) and index not in order:
                order[index] = score
        return order


async def _default_revision_namespace_query(
    namespace: str,
    embedding: list[float],
    n_results: int,
    where: dict,
) -> dict:
    """Query one revision-owned collection; never the legacy default.

    The workspace comes from the manifest namespace itself
    (``ws_<workspace>_embed_<hash>_d<dim>``): an unparseable namespace fails
    closed instead of querying the legacy ``kb_<workspace>`` collection,
    which would serve current-config vectors for a pinned revision.
    """
    try:
        head, _, _ = str(namespace).partition("_embed_")
        if not head.startswith("ws_"):
            raise ValueError(f"unexpected namespace {namespace!r}")
        workspace_id = UUID(head[len("ws_"):])
    except (ValueError, TypeError, AttributeError) as exc:
        raise V1ServiceUnavailable(
            f"revision namespace {namespace!r} is not workspace-qualified; "
            "refusing to query"
        ) from exc

    def _run() -> dict:
        from app.services.embedding.vector_store import get_vector_store

        return get_vector_store(workspace_id, namespace=namespace).query(
            query_embedding=embedding, n_results=n_results, where=where
        )

    return await asyncio.to_thread(_run)


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
    "document.retrieve": "v1-revision-retrieval",
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
    document_retrieval: Any = None
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
        # Default-usable without per-request scope: the adapter's lazy v1
        # default is complete (no user/workspace-scoped construction arg).
        "v1-people": ("app.services.people.mongo_people_service", "search_by_name"),
        "v1-document-search": ("app.services.agent.tools", "search_documents"),
        # Revision-manifest retrieval: the namespace query path exists when
        # the vector store backing imports; the manifest loaders live in
        # document_views (same package, always importable). Call-time
        # failures still surface as typed ``V1ServiceUnavailable``.
        "v1-revision-retrieval": (
            "app.services.embedding.vector_store",
            "get_vector_store",
        ),
        # Conservative by design (review M6): KG/memory need a scoped id
        # (workspace_id/user_id) that only per-request ingress knows;
        # abbreviation has no v1 default at all; content readers have no
        # verified pinned-revision v1 source. T7 enables these explicitly
        # via ``available_services`` once wired — the probe never opens a
        # gate the default adapter cannot serve.
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

    All seven Phase-2 capabilities plus the P0 ``document.retrieve`` revision
    capability are registered with their v1-backed adapters;
    ``available_services`` (default: ``probe_v1_services()``) intersects
    them. Capabilities whose backing is missing — or whose persistence seams
    (``evidence``/``resolver``) T7 did not supply — are gated OUT, so
    dispatch raises typed ``CapabilityUnavailable`` (mapped to
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
            # Task 5: revision retrieval persists governed evidence per chunk.
            "v1-revision-retrieval",
        }
    if resolver is None:
        available -= {
            "v1-document-content",
            "v1-section-content",
            # Task 5: scoped retrieval pins every target through the resolver.
            "v1-revision-retrieval",
        }
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
    people_service = bundle.people_lookup or V1PeopleLookupService()
    # Named multi-match seam (M6): only the v1-backed service owns the
    # shared grouped-record reader; legacy/custom single-``lookup`` services
    # keep the capability's first-only path (``multi_match=None``).
    people_multi_match: MultiMatchPeopleLookupService | None = (
        people_service._multi_match
        if isinstance(people_service, V1PeopleLookupService)
        else None
    )
    registrations = [
        CapabilityRegistration(
            PeopleCapability(
                service=people_service,
                evidence=evidence,
                multi_match=people_multi_match,
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
            DocumentRetrieveCapability(
                service=bundle.document_retrieval
                or V1RevisionAwareRetrievalService(
                    session_factory=session_factory
                ),
                evidence=evidence,
                resolver=resolver,
            ),
            service=V1_SERVICE_GATES["document.retrieve"],
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
    pinned_target_resolver: Any = None,
    intent_classifier: Any = None,
    adaptive_planner: Any = None,
    adaptive_replanner: Any = None,
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
        pinned_target_resolver=pinned_target_resolver,
        intent_classifier=intent_classifier,
        adaptive_planner=adaptive_planner,
        adaptive_replanner=adaptive_replanner,
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


def undispatched_tasks(plan: TaskPlan, results: tuple[AgentResult, ...]) -> tuple[str, ...]:
    """Task ids in the checkpointed plan with no result yet (T3-N2 recipe).

    The T8 outer runner calls this at the terminal boundary: a non-empty
    remainder with no active run means a deadline-truncated (or otherwise
    incomplete) dispatch whose ``DispatchReport.truncated`` flag was
    deliberately never checkpointed. Pure function of frozen contracts.
    """
    done = {result.task_id for result in results}
    return tuple(task.task_id for task in plan.tasks if task.task_id not in done)


@asynccontextmanager
async def dedicated_retention_leases(
    session_factory: Callable[[], Any] | None = None,
    *,
    ttl: Any = None,
) -> AsyncIterator[Any]:
    """Yield a lease repo owning a DEDICATED unit of work (M4(a)/I3).

    ``binding_node`` (and the clarify resume path) commit
    ``repo.session`` before the checkpointable update may be written. If
    that session were shared with unrelated application work, the commit
    would also commit that work mid-turn. T7 ingress MUST therefore back
    the request-scoped lease repository with a session used for NOTHING
    else for the whole request: this helper opens exactly such a session
    (default: one fresh ``async_session_maker()`` session — a factory call
    is already an independent session, but it must not be reused or shared
    afterwards) and closes it on exit. Never inject a long-lived or shared
    session here.
    """
    from .v2.persistence.retention_leases import RevisionRetentionLeaseRepository

    if session_factory is None:
        from app.core.database import async_session_maker

        session_factory = async_session_maker
    session = session_factory()
    try:
        if ttl is None:
            yield RevisionRetentionLeaseRepository(session)
        else:
            yield RevisionRetentionLeaseRepository(session, ttl=ttl)
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            await _maybe_await(close())

"""Clarification interrupt/resume node (Phase 2, Task 5).

D6 ruling: ``route_node`` (T1) returns route ``clarify`` with
``clarification=None``, but the frozen aggregate validator
(``contracts/validation.py``, ``validate_supervisor_state``) requires a
persisted ``ClarificationRequest`` when route == clarify. ``clarify_node``
owns that gap: it builds the stable ``ClarificationRequest`` from the
semantic blocking ambiguities/unresolved refs and persists it in the returned
state update so the clarify state can be checkpointed (T6 wires the edge).

Stability: ``clarification_id`` and every candidate UUID/order derive
deterministically (``uuid5``) from the question plus the referenced ref and
document IDs, so rebuilding the same question yields the same identity and
order; only ``expires_at`` is per-request (frozen at build time). The request
carries no runtime secrets or trusted identity and is checkpoint-safe.

Resume: the raw user reply is loaded by ``ChatMessage.id`` through the
request-scoped ``chat_messages`` service and stays authoritative in chat
persistence; ``ClarificationResolution`` stores only the deterministic
selection (never user text). Resume validates expiry (the stable per-request
deadline), candidate membership, and the CURRENT ACL
(``authorization.require_document(candidate.document_id,
capability_runtime)``) — an unauthorized/expired/invalid selection fails
closed with a typed error and never reaches the router. On success resume
returns ``Command(resume=..., goto="binding")`` so the resolved flow restarts
at the Binding Resolver.

Candidate-free requests (``required_document_not_found`` /
``semantic_ambiguity`` with ``candidates == ()``) can never resume via
selection: every reply raises ``ClarificationUnsatisfiable``. The T6/T7
runner MUST treat that error as a fresh user turn (re-run
context → binding on the reply) instead of a bad reply, and MUST NOT retry
``resume_clarification`` on the same request.

Service error contract (T6/T7 wiring): ``chat_messages.get_user_message``
returns an object with a string ``content`` attribute (may be awaitable);
``authorization.require_document`` returns ``None`` on success (may be
awaitable) and raises *any* ``Exception`` subclass on denial — the denial is
wrapped in ``ClarificationUnauthorized`` (the original error chains as
``__cause__``) so T8 streaming can catch one known type. ``BaseException``
(cancellation) is never swallowed.

``interrupt``/``Command`` come from ``langgraph.types``. Capabilities receive
``AgentRequest`` + ``CapabilityRuntimeContext`` and never supervisor/graph
state; this node likewise receives only the semantic slice (build) or the
request-scoped runtime (resume) — never checkpoint state beyond the values it
is given.
"""
from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid5

from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from ..contracts.base import CONTRACT_VERSION
from ..contracts.clarification import (
    ClarificationReason,
    ClarificationRequest,
    ClarificationResolution,
    DocumentCandidate,
)
from ..contracts.semantic import DocumentReference, SemanticContext
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.validation import (
    ContractValidationError,
    validate_clarification_request,
    validate_clarification_resolution,
)
from .context import node_context

__all__ = [
    "ClarificationError",
    "ClarificationExpired",
    "ClarificationInvalidSelection",
    "ClarificationUnauthorized",
    "ClarificationUnsatisfiable",
    "DEFAULT_CLARIFICATION_TTL",
    "build_clarification",
    "clarify_node",
    "interrupt_for_clarification",
    "parse_clarification_resolution",
    "resume_clarification",
]


class ClarificationError(ValueError):
    """The clarification interrupt/resume step cannot proceed; fail closed."""


class ClarificationExpired(ClarificationError):
    """The stable per-request resume deadline passed before the user replied."""

    def __init__(self, clarification_id: str) -> None:
        super().__init__(
            f"clarification {clarification_id!r} expired before resume; "
            "the user must ask again"
        )
        self.clarification_id = clarification_id


class ClarificationInvalidSelection(ClarificationError):
    """The reply selects no offered candidate (unparseable, out of range, or dismissed)."""


class ClarificationUnsatisfiable(ClarificationError):
    """The persisted request offers no selectable candidate.

    Raised for every reply to a candidate-free request
    (``required_document_not_found`` / ``semantic_ambiguity`` with
    ``candidates == ()``): such a request can never resume via selection,
    so the runner must treat the reply as a fresh user turn instead of a
    bad reply (see the module docstring). Distinct from
    ``ClarificationInvalidSelection`` precisely so the runner can tell
    "nothing was offered" apart from "the reply was bad".
    """

    def __init__(self, clarification_id: str, reason: str) -> None:
        super().__init__(
            f"clarification {clarification_id!r} offers no selectable candidate "
            f"(reason {reason!r}); treat the reply as a fresh user turn"
        )
        self.clarification_id = clarification_id
        self.reason = reason


class ClarificationUnauthorized(ClarificationError):
    """The selected candidate is not authorized under the CURRENT ACL.

    Wraps whatever ``authorization.require_document`` raised (chained as
    ``__cause__``): the v2 service slot is ``Any``-typed, so the concrete
    denial type is a T6/T7 wiring detail, but resume always surfaces this
    one known type for T8 streaming to catch. Fail-closed: no ``Command``
    is produced.
    """

    def __init__(
        self, clarification_id: str, document_id: UUID,
    ) -> None:
        super().__init__(
            f"selected document {document_id} of clarification "
            f"{clarification_id!r} is not authorized under the current runtime"
        )
        self.clarification_id = clarification_id
        self.document_id = document_id


#: Stable namespace for the deterministic clarification/candidate UUIDs.
_CLARIFICATION_NAMESPACE = UUID("b3e1a7c4-5d2f-4a6b-8e0c-1d3f5a7b9c1e")

#: Per-request resume window frozen into ``ClarificationRequest.expires_at``.
DEFAULT_CLARIFICATION_TTL = timedelta(minutes=30)

#: Replies that explicitly dismiss the question (no candidate selected).
_DISMISSAL_TOKENS = frozenset(
    {"none", "no", "skip", "neither", "không", "bỏ qua", "khong", "none of these"}
)

#: "option 2" / "lựa chọn 2" / "#2" style display-number replies (1-based).
_DISPLAY_NUMBER_RE = re.compile(
    r"^(?:option|candidate|choice|phương án|lựa chọn|#)?\s*(\d+)\s*$",
    re.IGNORECASE,
)


def _now_aware(value: datetime | None) -> datetime:
    now = value if value is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now


def _blocking_refs(
    semantic: SemanticContext,
) -> tuple[DocumentReference, ...]:
    """Refs this question must clarify, in semantic order (stale pins excluded).

    Only genuinely unresolved refs block: a resolved ref is never listed,
    even when a blocking ambiguity names it (the ambiguity text still enters
    the persisted ``question``). ``SemanticContext`` is the current-turn
    projection, so the result is question-specific by construction.
    """
    return tuple(
        reference
        for reference in semantic.document_refs
        if reference.resolution_status != "resolved"
    )


def _stable_request_id(
    semantic: SemanticContext, unresolved_ref_ids: tuple[str, ...]
) -> str:
    """The deterministic identity ``build_clarification`` would mint."""
    return str(
        uuid5(
            _CLARIFICATION_NAMESPACE,
            f"{semantic.normalized_query}|{','.join(unresolved_ref_ids)}",
        )
    )


def _reason_for(refs: tuple[DocumentReference, ...]) -> ClarificationReason:
    if any(
        reference.resolution_status == "ambiguous"
        and len(set(reference.candidate_document_ids)) >= 2
        for reference in refs
    ):
        return "required_document_ambiguous"
    if refs:
        return "required_document_not_found"
    return "semantic_ambiguity"


def _question_for(
    semantic: SemanticContext, refs: tuple[DocumentReference, ...]
) -> str:
    parts = [
        ambiguity.description.strip()
        for ambiguity in semantic.blocking_ambiguities
        if ambiguity.description.strip()
    ]
    spans = [reference.original_span.strip() for reference in refs]
    spans = [span for span in spans if span]
    if spans:
        parts.append("Unresolved references: " + ", ".join(spans))
    for fallback in ("; ".join(parts), semantic.normalized_query.strip()):
        if fallback:
            return fallback
    contextualized = semantic.contextualized_query.strip()
    if contextualized:
        return contextualized
    return "Clarification required before continuing."


def build_clarification(
    semantic: SemanticContext,
    *,
    now: datetime | None = None,
    ttl: timedelta = DEFAULT_CLARIFICATION_TTL,
) -> ClarificationRequest:
    """Build the stable, checkpoint-safe clarification request for a question.

    Identity/order are deterministic in the question (same semantic projection
    rebuilds the same ``clarification_id`` and candidate UUIDs/order); only
    ``expires_at`` is per-request. The result is validated against the semantic
    it was built from before it is returned.
    """
    at = _now_aware(now)
    refs = _blocking_refs(semantic)
    unresolved_ref_ids = tuple(reference.ref_id for reference in refs)
    clarification_id = _stable_request_id(semantic, unresolved_ref_ids)
    candidates: list[DocumentCandidate] = []
    for reference in refs:
        if reference.resolution_status != "ambiguous":
            continue
        for document_id in sorted(set(reference.candidate_document_ids), key=str):
            ordinal = len(candidates)
            candidates.append(
                DocumentCandidate(
                    candidate_id=str(
                        uuid5(
                            _CLARIFICATION_NAMESPACE,
                            f"{clarification_id}|{reference.ref_id}|{document_id}",
                        )
                    ),
                    ordinal=ordinal,
                    ref_id=reference.ref_id,
                    document_id=document_id,
                    label=f"{reference.normalized_reference} — option {ordinal + 1}",
                )
            )
    request = ClarificationRequest(
        contract_version=CONTRACT_VERSION,
        clarification_id=clarification_id,
        reason=_reason_for(refs),
        question=_question_for(semantic, refs),
        unresolved_ref_ids=unresolved_ref_ids,
        candidates=tuple(candidates),
        expires_at=at + ttl,
    )
    validate_clarification_request(request, semantic)
    return request


def _live_persisted_request(
    state: SupervisorV2State,
) -> ClarificationRequest | None:
    """The already-checkpointed request when it is still usable, else None.

    A persisted request survives only when it is live (not past its stable
    per-request deadline), still identifies this question (deterministic id
    over the current semantic projection), and still validates against the
    current semantic. Checkpoint round-trips may deliver plain mappings, so
    those are re-validated into the contract before comparison.
    """
    semantic = state["semantic"]
    existing = state.get("clarification")
    if isinstance(existing, Mapping):
        try:
            existing = ClarificationRequest.model_validate(existing)
        except ValidationError:
            return None
    if not isinstance(existing, ClarificationRequest):
        return None
    if datetime.now(timezone.utc) >= _now_aware(existing.expires_at):
        return None
    current_ids = tuple(
        reference.ref_id for reference in _blocking_refs(semantic)
    )
    if existing.clarification_id != _stable_request_id(semantic, current_ids):
        return None
    try:
        validate_clarification_request(existing, semantic)
    except ContractValidationError:
        return None
    return existing


async def clarify_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Persist the stable clarification request so the clarify state checkpoints.

    Idempotent: LangGraph re-executes the interrupting node from the top on
    resume, so a live request already persisted in state is returned
    unchanged (the stable per-request deadline is never silently extended).
    Only the runtime shape is validated here — this node needs no services.
    """
    # Fail closed on a malformed runtime even though no service is consumed.
    node_context(runtime)
    live = _live_persisted_request(state)
    if live is not None:
        return {"clarification": live}
    return {"clarification": build_clarification(state["semantic"])}


async def interrupt_for_clarification(
    request: ClarificationRequest,
) -> dict[str, object]:
    """Raise the LangGraph interrupt carrying the checkpointed request payload.

    On resume the runner re-enters here with the ``Command(resume=...)``
    payload, which is returned to the caller.
    """
    return interrupt({"clarification": request.model_dump(mode="json")})


def parse_clarification_resolution(
    content: str, request: ClarificationRequest
) -> ClarificationResolution:
    """Parse the raw reply into the deterministic selection (no user text kept).

    Accepts an exact candidate id, an exact candidate label
    (case-insensitive), or a 1-based display number (``"2"``, ``"option 2"``,
    ``"lựa chọn 2"``, ``"#2"``) over the ordinal display order. Explicit
    dismissal tokens resolve to no selection. Anything else raises
    ``ClarificationInvalidSelection``. A request that offers no candidates
    raises ``ClarificationUnsatisfiable`` for every reply — even past its
    deadline — so the runner never mistakes it for a bad reply.
    """
    if not request.candidates:
        raise ClarificationUnsatisfiable(
            request.clarification_id, request.reason
        )
    text = content.strip()
    lowered = text.casefold()
    if lowered in _DISMISSAL_TOKENS:
        return ClarificationResolution(
            contract_version=CONTRACT_VERSION,
            clarification_id=request.clarification_id,
            selected_candidate_id=None,
        )
    for candidate in request.candidates:
        if text == candidate.candidate_id or lowered == candidate.label.casefold():
            return ClarificationResolution(
                contract_version=CONTRACT_VERSION,
                clarification_id=request.clarification_id,
                selected_candidate_id=candidate.candidate_id,
            )
    match = _DISPLAY_NUMBER_RE.match(text)
    if match is not None:
        ordered = sorted(request.candidates, key=lambda item: item.ordinal)
        index = int(match.group(1)) - 1
        if 0 <= index < len(ordered):
            return ClarificationResolution(
                contract_version=CONTRACT_VERSION,
                clarification_id=request.clarification_id,
                selected_candidate_id=ordered[index].candidate_id,
            )
    raise ClarificationInvalidSelection(
        f"reply selects no candidate of clarification {request.clarification_id!r}"
    )


def _selected_candidate(
    resolution: ClarificationResolution, request: ClarificationRequest
) -> DocumentCandidate:
    """Resolve the validated selection to its offered candidate (fail closed)."""
    validate_clarification_resolution(request, resolution)
    if resolution.selected_candidate_id is None:
        raise ClarificationInvalidSelection(
            f"clarification {request.clarification_id!r} has no selected candidate"
        )
    for candidate in request.candidates:
        if candidate.candidate_id == resolution.selected_candidate_id:
            return candidate
    raise ClarificationInvalidSelection(
        f"selected candidate {resolution.selected_candidate_id!r} is not part "
        f"of clarification {request.clarification_id!r}"
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def resume_clarification(
    message_id: UUID,
    request: ClarificationRequest,
    runtime: GraphRuntimeContext | "Runtime[GraphRuntimeContext]",
) -> Command:
    """Resume a clarification from the raw user reply and restart at binding.

    ``runtime`` is the request-scoped ``GraphRuntimeContext`` or, for graph
    wiring, the framework ``Runtime`` wrapper carrying it (both accepted).
    The reply is loaded by ``ChatMessage.id`` and stays authoritative in chat
    persistence. Expiry uses the stable per-request deadline; membership uses
    the frozen request; authorization uses the CURRENT runtime
    (``require_document`` with the resume-time ``capability_runtime``), so a
    selected candidate that is no longer authorized fails closed with
    ``ClarificationUnauthorized``. Nothing unauthorized is accepted.
    """
    context = node_context(runtime)
    chat_messages = context.services.chat_messages
    if chat_messages is None:
        raise ClarificationError(
            "no chat_messages service wired on runtime.services; "
            "the raw clarification reply cannot be loaded"
        )
    message = await _maybe_await(chat_messages.get_user_message(message_id))
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ClarificationInvalidSelection(
            f"clarification {request.clarification_id!r} reply carries no content"
        )
    resolution = parse_clarification_resolution(content, request)
    if datetime.now(timezone.utc) >= _now_aware(request.expires_at):
        raise ClarificationExpired(request.clarification_id)
    candidate = _selected_candidate(resolution, request)
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
    except Exception as error:
        raise ClarificationUnauthorized(
            request.clarification_id, candidate.document_id
        ) from error
    return Command(resume=resolution.model_dump(mode="json"), goto="binding")

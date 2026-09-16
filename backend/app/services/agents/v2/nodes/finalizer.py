"""Terminal response boundary (Phase 2, Task 4; grounded synthesis: Task 6B).

Only direct non-factual paths and grounded factual paths may emit success:

- ``direct`` → success with the deterministic direct reply (a bounded
  conversational model may supply it later through ``build_direct_response``;
  Phase 2 uses the canned fallback — never a capability, never evidence).
- ``clarify`` → the persisted clarification question (fail-closed when the
  route promises a question the checkpoint does not hold).
- ``fast_domain`` factual → the ``SynthesisCheckpoint`` on the supervisor
  aggregate is the single authority for the LLM synthesis path (spec
  §13.3–13.5): a ``grounded`` phase becomes ``success`` with the
  server-rendered content and the allowlisted ``PublicCitation``
  projection replayed verbatim; every other checkpoint outcome —
  ``failed``, a mid-flight phase, or a corrupt checkpoint — collapses to
  the §12.4 typed ``error`` terminal with the safe Vietnamese message.
  When ``synthesis`` is ``None`` the non-LLM presentation path applies:
  the synthesize boundary ran the existing presentation inline (e.g.
  people_card) and stored its grounded result in the runtime-only
  channel, which this node replays; a channel-recorded failure becomes a
  typed response from the checkpointed task outcomes; a missing entry on
  a ``sufficient`` verdict fails closed as §12.4. Non-sufficient verdicts
  keep their typed terminals (denied/error/insufficient). User-facing
  content carries no internal identifiers.
- ``complex_research`` → the subgraph's verdict decides: ``sufficient``
  runs flow through the same checkpointed synthesis state machine and
  finalize factually here (success when grounded); every other verdict
  (or a missing evaluation) becomes a typed non-success response, never a
  fabricated answer. Write (``simple_write_operation``) stays typed
  unavailable (``denied``).

There is no re-derivation and no extractive fallback on production paths:
``SynthesisCheckpoint`` is authoritative for LLM attempts, handles,
grounded output, and recovery (spec §13.5); the channel only carries the
non-LLM presentation result within the same process.

Terminal lease release is NOT performed here: the outer runner releases the
run's leases only after the terminal checkpoint succeeds (T6 ordering).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Awaitable, Callable

from langgraph.runtime import Runtime

from ..contracts.base import CONTRACT_VERSION
from ..contracts.response import FinalResponse
from ..contracts.routing import RouteDecision
from ..contracts.semantic import SemanticContext
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.synthesis import SynthesisCheckpoint
from ..contracts.validation import validate_final_response
from .context import _context_of
from .evaluate import _channel_of

__all__ = [
    "FinalizerError",
    "DirectResponder",
    "build_direct_response",
    "finalizer_node",
]

#: Phase-2 canned direct replies (a bounded conversational model supplies
#: these once T7 wires it through ``build_direct_response``).
_DIRECT_GREETING = "Xin chào! Tôi có thể giúp gì cho bạn?"
_DIRECT_CONVERSATION = "Tôi đã ghi nhận. Tôi có thể giúp gì thêm cho bạn?"

#: User-safe fallbacks: internal target/criterion identifiers never reach
#: the user-facing response (diagnostics stay in logs and reports).
_INSUFFICIENT_CONTENT = "Không đủ căn cứ đã xác minh."
_NEEDS_INPUT_CONTENT = "Tôi cần thêm thông tin để trả lời."


class FinalizerError(ValueError):
    """No terminal response can be derived; the run must fail, not succeed."""


DirectResponder = Callable[[SemanticContext, RouteDecision], "str | Awaitable[str]"]
"""Bounded-model seam for direct replies: semantic facts in, prose out."""


async def build_direct_response(
    semantic: SemanticContext,
    route: RouteDecision,
    responder: DirectResponder | None = None,
) -> str:
    """Direct non-factual reply without capabilities or evidence.

    The optional bounded responder receives only finalized semantic facts and
    the route decision — never authorization or checkpoint state. Without one,
    the deterministic canned reply for the reason code is returned.
    """
    if responder is not None:
        import inspect

        reply = responder(semantic, route)
        if inspect.isawaitable(reply):
            reply = await reply
        if not isinstance(reply, str) or not reply.strip():
            raise FinalizerError(
                "direct responder returned no usable reply; refusing an "
                "empty direct success"
            )
        return reply
    if route.reason_code == "direct_greeting":
        return _DIRECT_GREETING
    return _DIRECT_CONVERSATION


def _require_route(state: SupervisorV2State) -> RouteDecision:
    route = state["route_decision"]
    if route is None:
        raise FinalizerError(
            "finalizer requires the checkpointed route decision; refusing "
            "to emit a response without a route"
        )
    return route


def _emit(response: FinalResponse) -> dict:
    validate_final_response(response)
    return {"final_response": response}


#: Spec §12.4: the single user-safe message for every synthesis failure —
#: persisted as nonblank assistant content so a reload never recreates a
#: blank row. Internal failure codes stay telemetry-only.
_SYNTHESIS_FAILED_CONTENT = (
    "Không thể tổng hợp câu trả lời đã được kiểm chứng. Vui lòng thử lại."
)


def _synthesis_checkpoint_of(
    state: SupervisorV2State,
) -> SynthesisCheckpoint | None:
    """Coerce the checkpointed synthesis slot (model or serde mapping).

    Post-checkpoint the slot may arrive as a plain mapping whose nested
    tuples revived as lists; the JSON round-trip mirrors ``_coerce_slot``
    (supervisor_v2) so strict-mode validation still succeeds. An
    unparseable value resolves to ``None`` so the caller fails closed with
    the typed synthesis failure instead of raising on corrupt state.
    """
    value = state.get("synthesis")
    if value is None:
        return None
    if isinstance(value, SynthesisCheckpoint):
        return value
    if isinstance(value, Mapping):
        try:
            return SynthesisCheckpoint.model_validate(value)
        except Exception:
            try:
                return SynthesisCheckpoint.model_validate_json(
                    json.dumps(value, default=str)
                )
            except Exception:
                return None
    return None


def _typed_synthesis_failure() -> dict:
    """Spec §12.4: exactly one typed error terminal for synthesis failure.

    Every non-grounded checkpoint outcome — ``failed`` phase, a mid-flight
    phase that reached the finalizer, or a missing/corrupt checkpoint with
    no channel entry on a sufficient verdict — collapses to the same safe
    message. No internal codes, no partial proposal, no extractive
    re-derivation.
    """
    return _emit(
        FinalResponse(
            contract_version=CONTRACT_VERSION,
            status="error",
            content=_SYNTHESIS_FAILED_CONTENT,
            citations=(),
        )
    )


def _typed_outcome_failure(state: SupervisorV2State) -> dict:
    """Translate a channel-recorded synthesis failure into a typed response.

    Used only for the non-LLM presentation path (``synthesis=None``): a
    denied task outcome denies the response; an error outcome errors it;
    anything else (``not_found``, decayed hydration, empty admission) is a
    typed ``insufficient``. No internal identifiers reach the content.
    """
    results = state["execution"].task_results
    if any(result.status == "denied" for result in results):
        status = "denied"
        content = "Yêu cầu bị từ chối quyền truy cập."
    elif any(result.status == "error" for result in results):
        status = "error"
        content = "Đã xảy ra lỗi khi thu thập bằng chứng."
    else:
        status = "insufficient"
        content = _INSUFFICIENT_CONTENT
    return _emit(
        FinalResponse(
            contract_version=CONTRACT_VERSION,
            status=status,  # type: ignore[arg-type]
            content=content,
            citations=(),
        )
    )


#: Statuses that mean the turn cannot proceed without the user naming a
#: document (criterion A: the semantic signal, not the missing verdict, decides
#: clarify vs insufficient).
_NEEDS_INPUT_REF_STATUSES = frozenset({"unresolved", "ambiguous"})


def _semantic_needs_input(state: SupervisorV2State) -> bool:
    """True when an unresolved/ambiguous reference or a blocking ambiguity remains."""
    semantic = state["semantic"]
    if semantic.blocking_ambiguities:
        return True
    return any(
        ref.resolution_status in _NEEDS_INPUT_REF_STATUSES
        for ref in semantic.document_refs
    )


def _typed_missing_verdict(state: SupervisorV2State) -> dict:
    """Turn a factual/complex turn with no checkpointed verdict into a typed reply.

    Never raises, never succeeds: a user-fixable semantic gap clarifies; anything
    else is typed ``insufficient``. Internal identifiers are never surfaced.
    """
    semantic = state["semantic"]
    if _semantic_needs_input(state):
        details = [ambiguity.description for ambiguity in semantic.blocking_ambiguities]
        spans = [
            ref.original_span
            for ref in semantic.document_refs
            if ref.resolution_status in _NEEDS_INPUT_REF_STATUSES
        ]
        if spans:
            details.append("chưa xác định: " + ", ".join(spans))
        content = _NEEDS_INPUT_CONTENT + (" " + " ".join(details) if details else "")
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="clarify",
                content=content,
                citations=(),
            )
        )
    return _emit(
        FinalResponse(
            contract_version=CONTRACT_VERSION,
            status="insufficient",
            content=_INSUFFICIENT_CONTENT,
            citations=(),
        )
    )


async def _finalize_factual(
    state: SupervisorV2State, context: GraphRuntimeContext
) -> dict:
    evaluation = state["execution"].evidence_evaluation
    if evaluation is None:
        return _typed_missing_verdict(state)
    if evaluation.status == "needs_input":
        details = [
            ambiguity.description
            for ambiguity in state["semantic"].blocking_ambiguities
        ]
        content = _NEEDS_INPUT_CONTENT + (
            " " + " ".join(details) if details else ""
        )
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="clarify",
                content=content,
                citations=(),
            )
        )
    if evaluation.status in ("insufficient", "contradictory"):
        # A denied/errored task outcome owns the typed response: the verdict
        # is insufficient because evidence never arrived, and the reason is
        # authorization or infrastructure — not a content gap.
        results = state["execution"].task_results
        if any(result.status == "denied" for result in results):
            return _emit(
                FinalResponse(
                    contract_version=CONTRACT_VERSION,
                    status="denied",
                    content="Yêu cầu bị từ chối quyền truy cập.",
                    citations=(),
                )
            )
        if any(result.status == "error" for result in results):
            return _emit(
                FinalResponse(
                    contract_version=CONTRACT_VERSION,
                    status="error",
                    content="Đã xảy ra lỗi khi thu thập bằng chứng.",
                    citations=(),
                )
            )
        details = [
            f"{contradiction.claim_a} <-> {contradiction.claim_b}"
            for contradiction in evaluation.contradictions
        ]
        content = _INSUFFICIENT_CONTENT + (
            " " + " ".join(details[:2]) if details else ""
        )
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="insufficient",
                content=content,
                citations=(),
            )
        )
    # Sufficient verdict: the checkpointed SynthesisCheckpoint is the single
    # authority for the LLM synthesis path (spec §13.3–13.5). A grounded
    # artifact becomes the success response verbatim — the rendered content
    # and the allowlisted PublicCitation projection are replayed, never
    # rebuilt. Every other checkpoint outcome (failed phase, mid-flight
    # phase, corrupt checkpoint) is the §12.4 typed synthesis failure.
    checkpoint = _synthesis_checkpoint_of(state)
    if checkpoint is not None:
        if checkpoint.phase == "grounded" and checkpoint.grounded is not None:
            return _emit(
                FinalResponse(
                    contract_version=CONTRACT_VERSION,
                    status="success",
                    content=checkpoint.grounded.content,
                    citations=checkpoint.grounded.citations,
                )
            )
        return _typed_synthesis_failure()
    # synthesis=None: the non-LLM presentation path (people_card and other
    # existing specialized presentations) runs inline at the synthesize
    # boundary and stores its grounded result in the runtime-only channel —
    # it never enters the bounded state machine, so no checkpoint exists.
    # The channel is authoritative ONLY for this path; a missing entry on a
    # sufficient verdict means the state machine never ran — fail closed.
    run_id = context.capability_runtime.run_id
    channel = _channel_of(context)
    entry = channel.get(run_id) if channel is not None else None
    if entry is not None and entry.grounded_draft is not None:
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="success",
                content=entry.grounded_draft.content,
                citations=entry.citations,
            )
        )
    if entry is not None and entry.synthesis_error is not None:
        return _typed_outcome_failure(state)
    return _typed_synthesis_failure()


async def finalizer_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Emit the terminal typed response for the checkpointed route."""
    context = _context_of(runtime)
    route = _require_route(state)
    if route.route == "direct":
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="success",
                content=await build_direct_response(state["semantic"], route),
                citations=(),
            )
        )
    if route.route == "clarify":
        clarification = state["clarification"]
        if clarification is None:
            raise FinalizerError(
                "clarify route requires a persisted ClarificationRequest; "
                "refusing to emit a clarification without one"
            )
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="clarify",
                content=clarification.question,
                citations=(),
            )
        )
    if route.route == "fast_domain":
        return await _finalize_factual(state, context)
    if route.reason_code == "simple_write_operation":
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="denied",
                content=(
                    "Thao tác viết (write) chưa được hỗ trợ trong v2 "
                    "giai đoạn này."
                ),
                citations=(),
            )
        )
    if route.route == "complex_research":
        # Phase 3 (R4) + Task 6B: the complex subgraph evaluated; sufficient
        # runs flow through the same checkpointed synthesis state machine
        # (the outer ``synthesize`` boundary) and finalize factually here.
        # Every other verdict — or a missing evaluation — returns the typed
        # missing-verdict reply from `_finalize_factual` (clarify when a
        # semantic gap remains, else insufficient).
        return await _finalize_factual(state, context)
    return _emit(
        FinalResponse(
            contract_version=CONTRACT_VERSION,
            status="error",
            content=(
                "Yêu cầu này hiện chưa được hỗ trợ "
                f"(lý do: {route.reason_code})."
            ),
            citations=(),
        )
    )

"""Terminal response boundary (Phase 2, Task 4).

Only direct non-factual paths and grounded factual paths may emit success:

- ``direct`` → success with the deterministic direct reply (a bounded
  conversational model may supply it later through ``build_direct_response``;
  Phase 2 uses the canned fallback — never a capability, never evidence).
- ``clarify`` → the persisted clarification question (fail-closed when the
  route promises a question the checkpoint does not hold).
- ``fast_domain`` factual → the grounded result consumed from the
  runtime-only channel (this node never synthesizes or grounds on the normal
  path); ``sufficient`` but ungroundable, and every other evaluation status,
  emits a typed non-success response. Synthesis failures become typed
  responses (denied/insufficient/error from the checkpointed task outcomes),
  never escaping exceptions. User-facing content carries no internal
  target/criterion identifiers.
- ``complex_research`` → the subgraph's verdict decides: ``sufficient``
  flowed through ``synthesize``/``ground`` into the shared channel and
  finalizes factually here (success when grounded); every other verdict
  (or a missing evaluation) becomes a typed non-success response, never a
  fabricated answer. Write (``simple_write_operation``) stays typed
  unavailable (``denied``).

On a channel miss (restart between ground and finalizer) the grounded result
is re-derived deterministically ONCE through the shared
``synthesize_and_lease`` helper plus ``ground_answer`` — which also leases
any freshly minted overflow uses — rather than synthesizing on every node.

Terminal lease release is NOT performed here: the outer runner releases the
run's leases only after the terminal checkpoint succeeds (T6 ordering).
"""
from __future__ import annotations

from typing import Awaitable, Callable

from langgraph.runtime import Runtime

from ..contracts.base import CONTRACT_VERSION
from ..contracts.response import FinalResponse
from ..contracts.routing import RouteDecision
from ..contracts.semantic import SemanticContext
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.validation import validate_final_response
from .context import _context_of
from .evaluate import _channel_of
from .execute import require_checkpointed_plan
from .grounding import GroundingInsufficient
from .grounding import ground_answer as _ground_answer
from .synthesize import (
    DEFAULT_SYNTHESIS_BUDGET,
    SynthesisError,
    _synthesis_input_of,
    synthesize_and_lease,
)

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


def _typed_synthesis_failure(state: SupervisorV2State) -> dict:
    """Translate a synthesis failure into a typed response (never a raise).

    A denied task outcome denies the response; an error outcome errors it;
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


async def _finalize_factual(
    state: SupervisorV2State, context: GraphRuntimeContext
) -> dict:
    evaluation = state["execution"].evidence_evaluation
    if evaluation is None:
        raise FinalizerError(
            "factual finalization requires a checkpointed EvidenceEvaluation; "
            "refusing to emit a response without a verdict"
        )
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
        return _typed_synthesis_failure(state)
    # Channel miss: single deterministic re-derivation (synthesize + lease +
    # ground, no reviser); failures become typed responses, never raises.
    plan = require_checkpointed_plan(state)
    try:
        derived = await synthesize_and_lease(
            synthesis_input=_synthesis_input_of(state),
            runtime=context,
            plan=plan,
            bindings=state["bindings"],
            budget=DEFAULT_SYNTHESIS_BUDGET,
        )
        grounded = await _ground_answer(
            draft=derived.draft, evidence=derived.evidence
        )
    except GroundingInsufficient:
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="insufficient",
                content=_INSUFFICIENT_CONTENT,
                citations=(),
            )
        )
    except SynthesisError:
        return _typed_synthesis_failure(state)
    return _emit(
        FinalResponse(
            contract_version=CONTRACT_VERSION,
            status="success",
            content=grounded.draft.content,
            citations=grounded.citations,
        )
    )


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
        # Phase 3 (R4): the complex subgraph evaluated; `sufficient` runs
        # were synthesized + grounded through the shared channel and every
        # other verdict is typed here. An evaluation-less complex turn fails
        # closed inside `_finalize_factual` (converted to a typed error).
        return await _finalize_factual(state, context)
    return _emit(
        FinalResponse(
            contract_version=CONTRACT_VERSION,
            status="error",
            content=(
                "Luồng complex_research chưa khả dụng trong Phase 2 "
                f"(lý do: {route.reason_code})."
            ),
            citations=(),
        )
    )

"""Terminal response boundary (Phase 2, Task 4).

Only direct non-factual paths and grounded factual paths may emit success:

- ``direct`` → success with the deterministic direct reply (a bounded
  conversational model may supply it later through ``build_direct_response``;
  Phase 2 uses the canned fallback — never a capability, never evidence).
- ``clarify`` → the persisted clarification question (fail-closed when the
  route promises a question the checkpoint does not hold).
- ``fast_domain`` factual → the re-derived grounded answer (synthesize +
  ground from the same checkpointed inputs, no reviser wired in Phase 2);
  ``sufficient`` but ungroundable emits ``insufficient``, other evaluation
  statuses map to ``insufficient`` (or ``clarify`` for ``needs_input``).
- ``complex_research`` → never success here: Write (``simple_write_operation``)
  is typed unavailable (``denied`` — T1 already routes it here), and every
  other complex reason is owned by the Phase-3 ``complex_boundary`` (typed
  ``error`` if it ever reaches this node, so a wiring bug cannot look like an
  answer).

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
from .grounding import GroundingInsufficient
from .grounding import ground_answer as _ground_answer
from .synthesize import _synthesis_input_of, synthesize_answer

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
        details = [item.description for item in evaluation.missing]
        for ambiguity in state["semantic"].blocking_ambiguities:
            details.append(ambiguity.description)
        content = "Tôi cần thêm thông tin: " + (
            "; ".join(details) if details else "yêu cầu chưa rõ."
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
        details = [item.description for item in evaluation.missing]
        for contradiction in evaluation.contradictions:
            details.append(
                f"mâu thuẫn: {contradiction.claim_a} <-> {contradiction.claim_b}"
            )
        content = "Không đủ căn cứ đã xác minh" + (
            ": " + "; ".join(details[:3]) if details else "."
        )
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="insufficient",
                content=content,
                citations=(),
            )
        )
    synthesis_input = _synthesis_input_of(state)
    try:
        synthesized = await synthesize_answer(
            synthesis_input=synthesis_input, runtime=context
        )
        grounded = await _ground_answer(
            draft=synthesized.draft, evidence=synthesized.evidence
        )
    except GroundingInsufficient as exc:
        details = list(exc.report.unmapped_assertions) + [
            assertion for assertion, _ in exc.report.ambiguous_assertions
        ]
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="insufficient",
                content="Không đủ căn cứ đã xác minh cho các nội dung: "
                + "; ".join(details[:3]),
                citations=(),
            )
        )
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

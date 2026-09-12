"""Deep-research port: legacy task output → typed v2 result facts (spec §13.3, §25).

This is a **server-internal** port, not the agent-facing tool gateway. Phase-1
``app.services.agents.deep_research`` contracts are legacy inputs/outputs only
(spec §25): a validated subset may be translated at the v2 boundary, but those
models never become v2 canonical contracts and never enter v2 checkpoint state.

The lossless subset this port translates is the task outcome:

- the legacy task status (``ok``/``partial``/``missing``/``ambiguous``/``error``)
  maps onto the frozen v2 ``AgentStatus``, keeping ``not_found`` distinguishable
  from denial/timeout/infrastructure failure;
- an ``error`` outcome carries a typed ``AgentError``.

Two legacy shapes are deliberately **not** translated because doing so would
invent identity/semantics rather than translate:

- ``Evidence`` uses a non-UUID ``evidence_id`` (``"{task_id}:c{N}"``), so it
  cannot become a canonical ``EvidenceRecord`` with stable identity; the legacy
  evidence remains legacy and is re-acquired through a v2 capability;
- ``Coverage`` is an aggregate count without a target id or observed locators, so
  it cannot become a ``CoverageObservation`` without inventing a target.
"""
from __future__ import annotations

from app.services.agents.deep_research.contracts import TaskResult

from ..contracts.base import CONTRACT_VERSION
from ..contracts.execution import AgentError, AgentResult, AgentStatus


class DeepResearchAdapterError(ValueError):
    """A legacy deep-research value cannot be translated into a v2 contract."""


def agent_status_from_legacy(status: str) -> AgentStatus:
    """Map a legacy task status onto the frozen v2 ``AgentStatus``."""
    if status == "ok":
        return "success"
    if status == "partial":
        return "partial"
    if status == "missing":
        return "not_found"
    if status == "ambiguous":
        return "needs_input"
    if status == "error":
        return "error"
    raise DeepResearchAdapterError(
        f"unsupported legacy deep-research task status {status!r}"
    )


def agent_result_from_legacy(result: TaskResult) -> AgentResult:
    """Translate one legacy ``TaskResult`` into a typed v2 ``AgentResult``.

    ``data`` stays ``None``: the legacy result carries no typed
    ``CapabilityOutput``, and a v2 capability boundary never accepts a raw
    provider payload. Evidence/coverage facts are re-derived through a v2
    capability, not copied from the legacy aggregate.
    """
    status = agent_status_from_legacy(result.status)
    error: AgentError | None = None
    if status == "error":
        error = AgentError(
            code="INTERNAL_ERROR",
            message=result.error_detail or "legacy deep-research task failed",
            retryable=False,
        )
    return AgentResult(
        contract_version=CONTRACT_VERSION,
        task_id=result.task_id,
        status=status,
        data=None,
        evidence_uses=(),
        coverage_observations=(),
        error=error,
    )

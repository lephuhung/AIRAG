"""Task execution facts at the capability boundary (spec §13.3).

``AgentRequest`` carries only task execution data: the scheduler resolves
``TaskSpec.capability`` from the current registry and runtime owns request/run IDs.
``AgentResult.status`` describes task execution, not global sufficiency, and
``data`` is only the minimized, schema-validated, checkpoint-safe
``CapabilityOutput``; sensitive payloads are represented by ``EvidenceUseRef``.
"""
from __future__ import annotations

from typing import Literal

from .base import ContractModel, ContractVersion
from .capability import CapabilityInput, CapabilityOutput
from .evaluation import CoverageObservation
from .evidence import EvidenceUseRef

AgentStatus = Literal["success", "partial", "not_found", "needs_input", "denied", "error"]

AgentErrorCode = Literal[
    "INVALID_INPUT",
    "SCOPE_VIOLATION",
    "PERMISSION_DENIED",
    "AMBIGUOUS_ENTITY",
    "DEPENDENCY_UNAVAILABLE",
    "TIMEOUT",
    "CANCELLED",
    "BUDGET_EXHAUSTED",
    "CONTRACT_MISMATCH",
    "INTERNAL_ERROR",
]


class AgentError(ContractModel):
    code: AgentErrorCode
    message: str
    retryable: bool


class AgentRequest(ContractModel):
    """Spec §3/§13.3: the transport boundary into one selected capability."""

    contract_version: ContractVersion
    task_id: str
    objective: str
    input: CapabilityInput


class AgentResult(ContractModel):
    """Spec §3/§13.3: the checkpointed capability result.

    ``task_id`` is retained because asynchronous fan-in, retries, and checkpoint
    association consume it. ``missing requirements`` stay with the evaluator.
    """

    contract_version: ContractVersion
    task_id: str
    status: AgentStatus
    data: CapabilityOutput | None
    evidence_uses: tuple[EvidenceUseRef, ...]
    coverage_observations: tuple[CoverageObservation, ...]
    error: AgentError | None


class TaskExecutionSummary(ContractModel):
    """Spec §16: ephemeral minimal projection of checkpointed ``AgentResult``s.

    It keeps ``not_found`` distinguishable from denial/timeout/infrastructure
    failure for replanning and is never separately persisted.
    """

    task_id: str
    status: AgentStatus
    error_code: AgentErrorCode | None = None


# Spec §3: models embedding discriminated-union aliases rebuild explicitly.
AgentRequest.model_rebuild()
AgentResult.model_rebuild()

"""Checkpoint aggregate versus request-scoped runtime (spec §7, §21).

``SupervisorV2State`` is the mutable LangGraph aggregate whose root
``contract_version`` selects the checkpoint schema; its nested business values
stay strict/frozen ``ContractModel`` values and are replaced rather than mutated.
``GraphRuntimeContext`` and ``RuntimeServices`` are injected per request and are
never checkpointed, so trusted identity and current authorization always replace
historical values on resume.
"""
from __future__ import annotations

from typing import TypedDict

from .base import ContractModel, ContractVersion, RuntimeModel
from .binding import DocumentBindingSet
from .capability import CapabilityRuntimeContext
from .clarification import ClarificationRequest
from .conversation import ConversationContext
from .evaluation import EvidenceEvaluation
from .execution import AgentResult
from .planning import TaskPlan
from .request import RequestContext
from .response import FinalResponse
from .routing import QueryAnalysis, RouteDecision
from .semantic import SemanticContext


class RuntimeServices(RuntimeModel):
    """Spec §7: request-scoped runtime service container; never checkpointed.

    The container is runtime-only and mutable so the runtime can inject the
    services a phase needs (the capability registry in Phase 2, retention leases
    in Task 9). It is never serialized into ``SupervisorV2State``.
    """


class GraphRuntimeContext(ContractModel):
    """Spec §7: injected, request-scoped runtime context."""

    capability_runtime: CapabilityRuntimeContext
    services: RuntimeServices


class ExecutionState(ContractModel):
    """Spec §21: the checkpointed execution aggregate.

    Lives with the checkpoint aggregates rather than in the task-contract module so
    that ``TaskPlan``/``AgentResult`` do not need to import each other. There is no
    aggregate ``evidence_refs``; consumers collect use refs from task results.
    """

    plan: TaskPlan | None
    task_results: tuple[AgentResult, ...]
    evidence_evaluation: EvidenceEvaluation | None


class SupervisorV2State(TypedDict, total=True):
    """Spec §7: the mutable checkpoint aggregate."""

    contract_version: ContractVersion
    request: RequestContext
    conversation: ConversationContext
    semantic: SemanticContext
    bindings: DocumentBindingSet
    query_analysis: QueryAnalysis | None
    route_decision: RouteDecision | None
    execution: ExecutionState
    clarification: ClarificationRequest | None
    final_response: FinalResponse | None

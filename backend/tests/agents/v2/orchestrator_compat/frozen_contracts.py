"""Typed benchmark contracts for the LangGraph v2 Phase 0 parity scenarios.

These classes reproduce the **minimum subset** of contract shapes from the
binding spec (`docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`)
needed to build parity scenarios. They are test fixtures only — production
v2 code imports from `backend/app/services/agents/contracts/` once Phase 1D
introduces them. The spec is the binding authority; this file's structure
follows it verbatim.

Each class header documents the spec section it derives from. All models use
`ConfigDict(extra="forbid", frozen=True, strict=True)` so:
- unknown fields are rejected (typed surface),
- instances are immutable (frozen state),
- `model_validate`/`__init__` reject coercions (strict mode).
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _PlaceholderCapabilityInput(BaseModel):
    """Phase 0 placeholder for CapabilityInput. The discriminated union is
    defined in the contracts module added by Phase 1D; until then a single
    model variant keeps the schema valid (a typed union of `dict` is not
    permitted by Pydantic v2).
    """
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["placeholder"]
    payload: dict[str, Any] = Field(default_factory=dict)


class _PlaceholderCapabilityOutput(BaseModel):
    """Phase 0 placeholder for CapabilityOutput. Same rationale as the input."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["placeholder"]
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Root base — ContractModel is the spec's frozen Pydantic envelope base.
# ---------------------------------------------------------------------------

class ContractModel(BaseModel):
    """Spec §4: every persisted business contract inherits this base."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


# ---------------------------------------------------------------------------
# Capacities & I/O — spec §13.4, only the minimum needed to wire TaskSpec.input.
# We use simple placeholders here because Phase 0 fixtures only carry the
# discriminated-union shape, never real capability data.
# ---------------------------------------------------------------------------

#: Discriminated-union placeholder for capability input variants.
#: Real variants live in the contracts module added by Phase 1D.
CapabilityInput = Annotated[
    _PlaceholderCapabilityInput,
    Field(discriminator="kind"),
]

#: Discriminated-union placeholder for capability output variants.
CapabilityOutput = Annotated[
    _PlaceholderCapabilityOutput,
    Field(discriminator="kind"),
]

#: Spec §18.1 — runtime authorization/scope. Never checkpointed.
class CapabilityRuntimeContext(ContractModel):
    user_id: str
    workspace_ids: tuple[str, ...]
    deadline: datetime | None = None
    feature_flags: tuple[str, ...] = ()


#: Spec §18 — service clients. Never checkpointed.
class RuntimeServices(ContractModel):
    capabilities_registry_id: str  # identity only; the registry itself is ephemeral


#: Spec §7 — request-scoped graph runtime. Never serialized into checkpoint.
class GraphRuntimeContext(ContractModel):
    capability_runtime: CapabilityRuntimeContext
    services: RuntimeServices


# ---------------------------------------------------------------------------
# Capability descriptor — spec §17.2 (referenced as a frozen contract).
# ---------------------------------------------------------------------------

class CapabilityDescriptor(ContractModel):
    name: str
    version: str
    input_model: str  # schema reference, not a Python type
    output_model: str


# ---------------------------------------------------------------------------
# §8 — Request / conversation / semantic contracts.
# ---------------------------------------------------------------------------

class KnownDocumentResource(ContractModel):
    resource_id: str
    document_id: UUID
    source: Literal["attachment", "ui_selection", "conversation", "api_explicit"]


class RequestContext(ContractModel):
    contract_version: Literal["2.0"]
    request_id: str
    thread_id: str
    original_query: str
    known_documents: tuple[KnownDocumentResource, ...] = ()


class ActiveEntity(ContractModel):
    entity_ref: str
    label: str


class EntityReference(ContractModel):
    ref_id: str
    kind: Literal["document", "person", "section", "concept"]
    label: str


class ConversationTurn(ContractModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ConversationContext(ContractModel):
    summary: str
    active_entities: tuple[ActiveEntity, ...] = ()
    last_focus: EntityReference | None = None
    recent_turns: tuple[ConversationTurn, ...] = ()


class DocumentReference(ContractModel):
    ref_id: str
    locator_summary: str


class SemanticContext(ContractModel):
    summary: str
    document_refs: tuple[DocumentReference, ...] = ()


# ---------------------------------------------------------------------------
# §10 — Bindings (minimal shape for fixtures; full body is Phase 1D scope).
# ---------------------------------------------------------------------------

class ScopedDocument(ContractModel):
    document_id: UUID
    document_revision: str
    role: Literal["primary", "supporting", "discovered"]


class DocumentBindingSet(ContractModel):
    bindings: tuple[ScopedDocument, ...] = ()


# ---------------------------------------------------------------------------
# §12 — QueryAnalysis & RouteDecision (typed literals, minimal fields).
# ---------------------------------------------------------------------------

WorkType = Literal[
    "people_lookup", "document_search", "section_read",
    "document_read", "write", "kg_query", "memory_lookup",
    "abbreviation_resolve", "compare", "summarize",
    "evaluate", "compliance", "legal_analysis",
]

Domain = Literal[
    "people", "document", "section", "kg", "memory", "write", "abbreviation",
]

Route = Literal["direct", "clarify", "fast_domain", "complex_research"]
RouteReason = Literal[
    "direct_greeting", "direct_conversation", "essential_ambiguity",
    "unresolved_required_binding", "simple_people_lookup",
    "exact_document_metadata", "exact_section_retrieval",
    "simple_write_operation", "simple_kg_lookup",
    "multi_document_research", "cross_domain_dependency", "comparison",
    "compliance_evaluation", "multi_goal", "runtime_dependency",
    "evidence_replanning_required",
]


class QueryAnalysis(ContractModel):
    contract_version: Literal["2.0"]
    work_type: WorkType
    domains: tuple[Domain, ...]
    dependency_hints: tuple[str, ...] = ()


class RouteDecision(ContractModel):
    contract_version: Literal["2.0"]
    route: Route
    reason: RouteReason


# ---------------------------------------------------------------------------
# §13 — Minimal task contracts.
# ---------------------------------------------------------------------------

class ContentLocator(ContractModel):
    kind: str  # discriminator lives in the Phase 1D contracts module
    locator: dict[str, Any] = Field(default_factory=dict)


class CompletionCriterion(ContractModel):
    kind: str  # spec §13.1 — CoverageCriterion | SemanticCriterion
    description: str


class TargetUnit(ContractModel):
    target_id: str
    binding_id: str
    requested_locator: ContentLocator
    completion_criteria: tuple[CompletionCriterion, ...] = ()


class InitialTaskOrigin(ContractModel):
    kind: Literal["initial"]


class ReplanTaskOrigin(ContractModel):
    kind: Literal["replan"]
    reason: str
    task_ids: tuple[str, ...]
    evidence_use_ids: tuple[UUID, ...]


TaskOrigin = Annotated[InitialTaskOrigin | ReplanTaskOrigin, Field(discriminator="kind")]


class TaskSpec(ContractModel):
    task_id: str
    capability: str
    task_objective: str
    input: CapabilityInput
    depends_on: tuple[str, ...] = ()
    origin: TaskOrigin


class TaskPlan(ContractModel):
    contract_version: Literal["2.0"]
    plan_id: str
    goal: str
    target_units: tuple[TargetUnit, ...] = ()
    tasks: tuple[TaskSpec, ...] = ()


# ---------------------------------------------------------------------------
# §13.3 — Agent request/result types.
# ---------------------------------------------------------------------------

AgentStatus = Literal[
    "success", "partial", "not_found", "needs_input",
    "denied", "error",
]
AgentErrorCode = Literal[
    "INVALID_INPUT", "SCOPE_VIOLATION", "PERMISSION_DENIED",
    "AMBIGUOUS_ENTITY", "DEPENDENCY_UNAVAILABLE", "TIMEOUT",
    "CANCELLED", "BUDGET_EXHAUSTED", "CONTRACT_MISMATCH", "INTERNAL_ERROR",
]


class AgentError(ContractModel):
    code: AgentErrorCode
    message: str
    retryable: bool


class AgentRequest(ContractModel):
    contract_version: Literal["2.0"]
    task_id: str
    objective: str
    input: CapabilityInput


class EvidenceUseRef(ContractModel):
    evidence_use_id: UUID


class CoverageObservation(ContractModel):
    target_id: str
    status: Literal["covered", "partial", "missing"]
    note: str = ""


class AgentResult(ContractModel):
    contract_version: Literal["2.0"]
    task_id: str
    status: AgentStatus
    data: CapabilityOutput | None = None
    evidence_uses: tuple[EvidenceUseRef, ...] = ()
    coverage_observations: tuple[CoverageObservation, ...] = ()
    error: AgentError | None = None


class TaskExecutionSummary(ContractModel):
    task_id: str
    status: AgentStatus
    error_code: AgentErrorCode | None = None


# ---------------------------------------------------------------------------
# §13.1 (truncated) and §15 (evidence) — minimal placeholders for fixtures.
# ---------------------------------------------------------------------------

class EvidenceEvaluation(ContractModel):
    status: Literal["sufficient", "partial", "insufficient"]
    missing: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# §18 — Final response.
# ---------------------------------------------------------------------------

class RenderedCitation(ContractModel):
    ref_id: str
    label: str


class FinalResponse(ContractModel):
    contract_version: Literal["2.0"]
    status: Literal["success", "clarify", "denied", "insufficient", "error"]
    content: str
    citations: tuple[RenderedCitation, ...] = ()


# ---------------------------------------------------------------------------
# §17 — Clarification contract.
# ---------------------------------------------------------------------------

class DocumentCandidate(ContractModel):
    candidate_id: str
    document_id: UUID
    label: str


class ClarificationRequest(ContractModel):
    contract_version: Literal["2.0"]
    clarification_id: str
    reason: Literal[
        "required_document_not_found", "required_document_ambiguous",
        "required_unit_unreadable", "semantic_ambiguity",
    ]
    question: str
    unresolved_ref_ids: tuple[str, ...]
    candidates: tuple[DocumentCandidate, ...] = ()
    expires_at: datetime


# ---------------------------------------------------------------------------
# §21 — ExecutionState and §7 — SupervisorV2State (the checkpointable aggregate).
# ---------------------------------------------------------------------------

class ExecutionState(ContractModel):
    plan: TaskPlan | None = None
    task_results: tuple[AgentResult, ...] = ()
    evidence_evaluation: EvidenceEvaluation | None = None


class SupervisorV2State(TypedDict, total=False):
    """Spec §7: the mutable checkpoint aggregate, NOT a frozen Pydantic model.

    LangGraph's `TypedDict` reducer-mutable aggregate; nested business values
    remain `ContractModel` and are replaced rather than mutated. Frozen here
    means its **schema** is fixed, not the dict instance.
    """
    contract_version: Literal["2.0"]
    request: RequestContext
    conversation: ConversationContext
    semantic: SemanticContext
    bindings: DocumentBindingSet
    query_analysis: QueryAnalysis | None
    route_decision: RouteDecision | None
    execution: ExecutionState
    clarification: ClarificationRequest | None
    final_response: FinalResponse | None

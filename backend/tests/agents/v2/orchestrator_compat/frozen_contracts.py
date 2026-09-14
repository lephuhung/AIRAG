"""Typed benchmark contracts for the LangGraph v2 Phase 0 parity scenarios.

These classes reproduce contract shapes from the binding spec
(`docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`).
They are TEST FIXTURES ONLY. Production v2 code imports these from
`backend/app/services/agents/contracts/` once Phase 1D introduces the real
contracts module.

Architectural rules (binding — see plan `2026-09-11-langgraph-v2-implementation.md`
prohibition #30):

- `CapabilityDescriptor`, `CapabilityInput`, `CapabilityOutput`, and the runtime-only
  `CapabilityRuntimeContext` are FROZEN contracts OWNED BY the contracts layer. They
  are NOT redefined or field-extended in this fixture file. Phase 0 references a
  clearly-named placeholder (`_Phase0CapabilityStandIn`) so different code paths can
  type-check without importing the real types. **Phase 1D REPLACES this placeholder
  with the contract-layer types from `contracts/capability.py`.**
- All `ContractModel`-derived families use
  `ConfigDict(extra="forbid", frozen=True, strict=True)`. The forbidden dict escape
  hatch of `dict[str, Any]` is absent from every locator and I/O contract.
- Every class header carries the spec section it derives from.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Spec §3 — base envelope.
# ---------------------------------------------------------------------------

class ContractModel(BaseModel):
    """Spec §3: every persisted business contract inherits this base."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


# ---------------------------------------------------------------------------
# Spec §11 — CapabilityRuntimeContext (forbidden to redefine).
# Spec §13.4 — CapabilityInput/Output (forbidden to redefine).
# Spec §16 — CapabilityDescriptor (forbidden to redefine).
# ---------------------------------------------------------------------------
#
# Per plan prohibition #30 these four are owned by `contracts/capability.py` and
# must NEVER be redefined or field-extended outside that module. Phase 0 fixtures
# therefore use a single, clearly-named placeholder and rely on the union
# annotation to keep `CapabilityInput`/`CapabilityOutput` width open for Phase 1D.

class _Phase0CapabilityStandIn(ContractModel):
    """Phase 0 placeholder so `GraphRuntimeContext.capability_runtime` and the
    `CapabilityInput` / `CapabilityOutput` unions compile.

    **Phase 1D replaces this with the real types from `contracts/capability.py`.
    Do not import from this fixture file in production code.**
    """
    kind: Literal["phase0_placeholder"]


# Spec §11: request-scoped runtime authorization/scope. Never checkpointed.
# We type the field as the placeholder so the file is self-consistent, but we
# do NOT restate the spec's fields here. Phase 1D swaps in the real contract.
CapabilityRuntimeContext = _Phase0CapabilityStandIn

# Spec §13.4: discriminated unions of capability-specific models. Phase 0
# uses a single-variant placeholder union; Phase 1D expands to the real variants.
CapabilityInput = Annotated[_Phase0CapabilityStandIn, Field(discriminator="kind")]
CapabilityOutput = Annotated[_Phase0CapabilityStandIn, Field(discriminator="kind")]

# Spec §16: registry descriptor. Never redefined here.
CapabilityDescriptor = _Phase0CapabilityStandIn


# ---------------------------------------------------------------------------
# Spec §16 — RuntimeServices (request-scoped, never checkpointed).
# ---------------------------------------------------------------------------

class RuntimeServices(ContractModel):
    """Spec §16 / §18: ephemeral service clients, never serialized into checkpoint."""
    capabilities_registry_id: str  # identity only; the registry itself is ephemeral


class GraphRuntimeContext(ContractModel):
    """Spec §7: request-scoped graph runtime. Never serialized into checkpoint."""
    capability_runtime: CapabilityRuntimeContext
    services: RuntimeServices


# ---------------------------------------------------------------------------
# Spec §8.1 — Request contracts.
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
    known_documents: tuple[KnownDocumentResource, ...]


# ---------------------------------------------------------------------------
# Spec §8.2 — Conversation.
#
# Spec references `ActiveEntity`, `EntityReference`, `ConversationTurn` but
# never defines them; they are spec-internal helpers. Phase 0 fixtures use
# minimal `_Placeholder` shapes named to make their provisional nature obvious.
# Phase 1D either defines them in spec or eliminates the dependency.
# ---------------------------------------------------------------------------

class _ActiveEntityPlaceholder(ContractModel):
    """Spec §8.2 references `ActiveEntity` but does not define it; placeholder."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    entity_ref: str
    label: str


class _EntityReferencePlaceholder(ContractModel):
    """Spec §8.2 / §8.3 references `EntityReference` but does not define it."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    ref_id: str
    kind: Literal["document", "person", "section", "concept"]
    label: str


class _ConversationTurnPlaceholder(ContractModel):
    """Spec §8.2 references `ConversationTurn` but does not define it."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    role: Literal["user", "assistant", "system"]
    content: str


class ConversationContext(ContractModel):
    summary: str
    active_entities: tuple[_ActiveEntityPlaceholder, ...]
    last_focus: _EntityReferencePlaceholder | None
    recent_turns: tuple[_ConversationTurnPlaceholder, ...]


# ---------------------------------------------------------------------------
# Spec §8.3 — Semantic lifecycle.
#
# `AbbreviationResolution`, `CoreferenceResolution`, `SectionReference`,
# `BlockingAmbiguity`, `SemanticDependencyHint` are all referenced but not
# defined in the spec. Phase 0 uses placeholders with a `_Placeholder` suffix.
# ---------------------------------------------------------------------------

class _AbbreviationResolutionPlaceholder(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    abbreviation: str
    resolution: str


class _CoreferenceResolutionPlaceholder(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    mention: str
    resolution: str


class _SectionReferencePlaceholder(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    ref_id: str
    label: str


class _BlockingAmbiguityPlaceholder(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    ambiguity_id: str
    description: str


class DocumentRole_Placeholder(ContractModel):
    """Spec §9: `DocumentRole` is a Literal in the spec; preserved as Literal here."""
    role: Literal["target", "reference", "supporting", "discovered"]


class CurrentRevisionRequirement(ContractModel):
    kind: Literal["current"]


class PinnedRevisionRequirement(ContractModel):
    kind: Literal["pinned"]
    document_revision: str


RevisionRequirement = Annotated[
    CurrentRevisionRequirement | PinnedRevisionRequirement,
    Field(discriminator="kind"),
]


class DocumentReference(ContractModel):
    ref_id: str
    original_span: str
    normalized_reference: str
    requested_role: Literal["target", "reference", "supporting", "discovered"] | None
    revision_requirement: RevisionRequirement | None = None
    resolution_status: Literal["unresolved", "resolved", "ambiguous", "not_found", "error"]
    resolved_document_id: UUID | None
    candidate_document_ids: tuple[UUID, ...] = ()


class SemanticContext(ContractModel):
    contextualized_query: str
    normalized_query: str
    abbreviations: tuple[_AbbreviationResolutionPlaceholder, ...]
    coreferences: tuple[_CoreferenceResolutionPlaceholder, ...]
    document_refs: tuple[DocumentReference, ...]
    person_refs: tuple[_EntityReferencePlaceholder, ...]
    section_refs: tuple[_SectionReferencePlaceholder, ...]
    blocking_ambiguities: tuple[_BlockingAmbiguityPlaceholder, ...]


# ---------------------------------------------------------------------------
# Spec §9 — Document identity / binding / target units.
# ---------------------------------------------------------------------------

class ScopedDocument(ContractModel):
    binding_id: str
    document_id: UUID
    document_revision: str
    role: Literal["target", "reference", "supporting", "discovered"]


class BindingRevisionRequirement(ContractModel):
    binding_id: str
    ref_id: str


class DocumentBindingSet(ContractModel):
    bindings: tuple[ScopedDocument, ...]
    revision_requirement_refs: tuple[BindingRevisionRequirement, ...]


# ---------------------------------------------------------------------------
# Spec §10 — Structured content locators (discriminated union, no dict escape).
# ---------------------------------------------------------------------------

class DocumentLocator(ContractModel):
    kind: Literal["document"]


class SectionLocator(ContractModel):
    kind: Literal["section"]
    structure_node_id: str


class ArticleLocator(ContractModel):
    kind: Literal["article"]
    structure_node_id: str
    article_id: str


class PageRangeLocator(ContractModel):
    kind: Literal["page_range"]
    start: int
    end: int


class ChunkRangeLocator(ContractModel):
    kind: Literal["chunk_range"]
    start: str
    end: str


ContentLocator = Annotated[
    DocumentLocator | SectionLocator | ArticleLocator | PageRangeLocator | ChunkRangeLocator,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Spec §13.1 — CompletionCriterion (discriminated union).
# ---------------------------------------------------------------------------

class CoverageCriterion(ContractModel):
    kind: Literal["coverage"]
    minimum_status: Literal["read_partial", "read_complete"] = "read_complete"
    allow_partial_reason: str | None = None


class SemanticCriterion(ContractModel):
    kind: Literal["semantic"]
    criterion_id: str
    description: str


CompletionCriterion = Annotated[
    CoverageCriterion | SemanticCriterion,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Spec §9 (TargetUnit), §13.2 (TaskPlan / TaskSpec / TaskOrigin).
# ---------------------------------------------------------------------------

class TargetUnit(ContractModel):
    target_id: str
    binding_id: str
    requested_locator: ContentLocator
    completion_criteria: tuple[CompletionCriterion, ...]


class InitialTaskOrigin(ContractModel):
    kind: Literal["initial"]


class ReplanTaskOrigin(ContractModel):
    kind: Literal["replan"]
    reason: str
    task_ids: tuple[str, ...]
    evidence_use_ids: tuple[UUID, ...]


TaskOrigin = Annotated[
    InitialTaskOrigin | ReplanTaskOrigin,
    Field(discriminator="kind"),
]


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
    target_units: tuple[TargetUnit, ...]
    tasks: tuple[TaskSpec, ...]


# ---------------------------------------------------------------------------
# Spec §12 — QueryAnalysis / RouteDecision.
# ---------------------------------------------------------------------------

WorkType = Literal[
    "direct", "lookup", "retrieve", "explain", "summarize",
    "compare", "evaluate", "cross_domain", "multi_goal",
]
Domain = Literal["people", "document", "section", "write", "knowledge_graph", "memory"]
Route = Literal["direct", "clarify", "fast_domain", "complex_research"]
RouteReason = Literal[
    "direct_greeting", "direct_conversation", "essential_ambiguity",
    "unresolved_required_binding", "simple_people_lookup",
    "exact_document_metadata", "exact_section_retrieval",
    "targetless_document_retrieval",
    "simple_write_operation", "simple_kg_lookup",
    "multi_document_research", "cross_domain_dependency", "comparison",
    "compliance_evaluation", "multi_goal", "runtime_dependency",
    "evidence_replanning_required",
]


class _SemanticDependencyHintPlaceholder(ContractModel):
    """Spec §12 references `SemanticDependencyHint` but does not define it."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    hint_id: str
    description: str


class QueryAnalysis(ContractModel):
    work_type: WorkType
    domains: tuple[Domain, ...]
    dependency_hints: tuple[_SemanticDependencyHintPlaceholder, ...] = ()


class RouteDecision(ContractModel):
    route: Route
    reason_code: RouteReason


# ---------------------------------------------------------------------------
# Spec §13.3 — Agent request / result / execution summary.
# ---------------------------------------------------------------------------

AgentStatus = Literal[
    "success", "partial", "not_found", "needs_input", "denied", "error",
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


# Spec §15.2 — EvidenceUseRef is the only graph reference chain.
class EvidenceUseRef(ContractModel):
    use_id: UUID


# Spec §14 — CoverageObservation.
CoverageOutcome = Literal["read", "missing", "unreadable", "truncated"]


class CoverageObservation(ContractModel):
    target_id: str
    observed_locators: tuple[ContentLocator, ...]
    outcome: CoverageOutcome


class AgentResult(ContractModel):
    contract_version: Literal["2.0"]
    task_id: str
    status: AgentStatus
    data: CapabilityOutput | None
    evidence_uses: tuple[EvidenceUseRef, ...]
    coverage_observations: tuple[CoverageObservation, ...]
    error: AgentError | None


class TaskExecutionSummary(ContractModel):
    task_id: str
    status: AgentStatus
    error_code: AgentErrorCode | None = None


# ---------------------------------------------------------------------------
# Spec §18 — Minimal evaluation contract.
# ---------------------------------------------------------------------------

class Contradiction(ContractModel):
    contradiction_id: str
    claim_a: str
    claim_b: str
    evidence_use_ids: tuple[UUID, ...]


class MissingRequirement(ContractModel):
    target_id: str
    criterion_kind: Literal["coverage", "semantic"]
    semantic_criterion_id: str | None = None
    description: str


CoverageStatus = Literal[
    "read_complete", "read_partial", "missing", "unreadable", "truncated",
]


class CoverageItem(ContractModel):
    target_id: str
    observed_locators: tuple[ContentLocator, ...]
    status: CoverageStatus


class Coverage(ContractModel):
    items: tuple[CoverageItem, ...]


class EvidenceEvaluation(ContractModel):
    status: Literal["sufficient", "insufficient", "contradictory", "needs_input"]
    coverage: Coverage
    missing: tuple[MissingRequirement, ...]
    contradictions: tuple[Contradiction, ...]


# ---------------------------------------------------------------------------
# Spec §19 — Synthesis and grounding boundary.
# ---------------------------------------------------------------------------

class RenderedCitation(ContractModel):
    citation_id: str
    evidence_id: UUID
    label: str


class FinalResponse(ContractModel):
    contract_version: Literal["2.0"]
    status: Literal["success", "clarify", "denied", "insufficient", "error"]
    content: str
    citations: tuple[RenderedCitation, ...] = ()


# ---------------------------------------------------------------------------
# Spec §20 — Clarification.
# ---------------------------------------------------------------------------

class DocumentCandidate(ContractModel):
    candidate_id: str
    ordinal: int
    ref_id: str
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
    candidates: tuple[DocumentCandidate, ...]
    expires_at: datetime


# ---------------------------------------------------------------------------
# Spec §21 — ExecutionState; §7 — SupervisorV2State (the checkpointable aggregate).
# ---------------------------------------------------------------------------

class ExecutionState(ContractModel):
    plan: TaskPlan | None
    task_results: tuple[AgentResult, ...]
    evidence_evaluation: EvidenceEvaluation | None


class SupervisorV2State(TypedDict, total=True):
    """Spec §7: mutable checkpoint aggregate. `total=True` (spec default) — every
    key is required by the schema; values may still be replaced at runtime.
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

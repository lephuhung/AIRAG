# LangGraph v2 Contract-First Architecture

**Date:** 2026-09-10

**Status:** Approved design

**Revision basis:** commit `35eeb3ce5ca499cb23d1cd428f43018e128f0429`

**Source direction:** `docs/agent-contract-langgraph-deepagent.md`

**Scope:** architecture and contracts only. This revision does not authorize production runtime changes.

## 1. Objective

Build LangGraph v2 around small, explicit contracts with one authoritative owner per fact. The architecture separates workflow, authorization, conversational meaning, document binding, task execution, evidence identity/use, evaluation, synthesis, and presentation.

The design preserves:

- independent `supervisor_v2.py` composition;
- v1/v2 selection at an external integration boundary;
- independently testable domain subgraphs;
- chat DB as conversation source of truth;
- runtime authorization outside checkpoint state;
- deterministic permission, scope, budget, validation, and hard evidence checks;
- shared domain capabilities for fast and complex paths;
- target/reference distinction and section-level coverage;
- interrupt/resume with current authorization revalidation;
- offline replay, shadow, canary, and gradual cutover;
- v1 until v2 benchmarks pass.

## 2. Minimal-contract principles

```text
Prefer references over copied state.
One fact → one authoritative owner.
Do not store A and B merely to validate A == B.
Derive convenience projections instead of persisting them.
Runtime/security/storage metadata does not belong in semantic contracts.
Presentation metadata does not belong in evidence identity.
```

Before adding a field, record:

```text
Field:
Authoritative owner:
Produced by:
Consumed by:
Persisted? yes/no
Derivable? yes/no
Reason it must exist:
```

If `Derivable=yes`, the default is to omit it from the canonical contract. A denormalized cache is an implementation detail and must not become a second source of truth.

## 3. Versioning policy

`contract_version` exists only on independently persisted or externally transported envelopes. Nested value objects inherit the envelope version.

Versioned independently persisted or transported boundaries:

- the mutable `SupervisorV2State` checkpoint aggregate carries the root checkpoint version;
- `RequestContext` persisted request boundary;
- `ConversationSnapshot` and `SemanticSnapshot`;
- `TaskPlan`;
- `AgentRequest` and `AgentResult` subgraph/transport boundaries;
- `EvidenceUseEnvelope` and `EvidenceStoreRow`;
- `BindingAuditRow`;
- `ClarificationRequest`/`ClarificationResolution`;
- `FinalResponse`.

`DocumentBindingSet` is nested checkpoint state and inherits the root checkpoint version. `EvidenceRecord` is persisted only through `EvidenceStoreRow`; no hypothetical standalone evidence envelope exists.

Leaf types such as `ScopedDocument`, `TargetUnit`, `DocumentReference`, `ContentLocator`, `CoverageItem`, source-identity variants, and runtime-only context do not repeat a version.

The snippets use `from __future__ import annotations` and a shared strict frozen base equivalent to:

```python
from pydantic import BaseModel, ConfigDict

class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
```

Actual modules must order definitions before use where practical. Where mutually referring aliases or models prevent that, the owning module resolves forward annotations and calls `model_rebuild()` after all referenced types are registered. Snippets omit repeated base-class/import boilerplate; omission does not relax strictness or immutability.

`SupervisorV2State` is a mutable LangGraph aggregate updated by reducers, not a frozen Pydantic business envelope. Its root version selects the checkpoint schema. Nested persisted business contracts remain strict/frozen `ContractModel` values and are replaced rather than mutated.

## 4. Terminology and ownership

### 4.1 LangGraph

```text
LangGraph owns:
state lifecycle + routing + subgraph composition + checkpoint + interrupt/resume + stream lifecycle
```

LangGraph primitives never appear in capability results.

### 4.2 ComplexResearchGraph

`ComplexResearchGraph` defines AIRAG complex-research behavior: planning, dependency execution, bounded parallelism, evidence acquisition, evaluation, append-only replan, and synthesis.

Its implementation remains open to a native LangGraph planner/executor, a Deep Agents adapter, or another conforming implementation. Phase 0 benchmarks this choice.

### 4.3 Authoritative owners

| Fact/object | Authoritative owner |
|---|---|
| Current workspace/People permissions | Backend runtime |
| Conversation history | Chat database |
| Short-term discourse context | Conversation layer |
| Final query meaning | Semantic finalizer |
| Document revision requirement | Semantic/reference resolution |
| Document semantic role and pinned revision | Binding Resolver |
| Logical document/read requirement | `TaskPlan.TargetUnit` when a factual route requires document/read coverage |
| Invocation `task_id` | `TaskPlan.TaskSpec` for every factual route |
| Fast/complex route | Deterministic Router |
| Executable DAG | ComplexResearchGraph |
| Domain execution facts | Capability |
| Evidence identity/content/provenance | Evidence Store |
| Run/task/target evidence usage | Evidence Use Store |
| Requirement coverage/sufficiency | Evaluator |
| Answer claims | Answer layer |
| Rendered citations/final response | Grounding + presentation layer |

## 5. Target architecture

```text
Backend ingress
├── authenticate and calculate current authorization
├── calculate People permission
├── persist ORIGINAL message
├── collect KnownDocumentResource[]
└── load ConversationContext
        ↓
Semantic Draft
├── deterministic extraction
├── coreference and abbreviation resolution
├── document references
└── optional small-model output (ephemeral)
        ↓
Binding Resolver
├── resolve authorized document identity
├── pin immutable document revision
├── assign semantic role
└── produce DocumentBindingSet
        ↓
Semantic Finalizer
├── contextualized/normalized query
├── canonical document references + revision requirements
└── persist SemanticSnapshot
        ↓
Deterministic-first QueryAnalysis
        ↓
Deterministic RouteDecision
├── direct
├── clarify
├── fast domain → deterministic one-task TaskPlan → AgentRequest → shared capability → AgentResult
└── complex research → planner-generated TaskPlan DAG → AgentRequest → shared capability → AgentResult
        ↓
EvidenceRecord + EvidenceUse → Evidence Store
        ↓
Coverage → EvidenceEvaluation
        ↓
Synthesis hydration → AnswerClaim[]
        ↓
Grounding → RenderedCitation[] → FinalResponse
```

## 6. Module and graph boundary

```text
backend/app/services/agents/
├── supervisor.py                    # unchanged v1
├── supervisor_v2.py                 # v2 composition only
└── v2/
    ├── contracts/
    ├── context_graph/
    ├── binding_graph/
    ├── routing_graph/
    ├── complex_research_graph/
    ├── people_graph/
    ├── document_graph/
    ├── section_graph/
    ├── write_graph/
    ├── knowledge_graph/
    ├── grounding_graph/
    ├── capabilities/
    ├── evidence_store/
    ├── persistence/
    └── adapters/
```

`supervisor_v2.py` composes subgraphs and declares edges only. Each subgraph has minimal typed input/output and does not read arbitrary root state.

## 7. Checkpoint state versus runtime

```python
class SupervisorV2State(TypedDict):
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

class GraphRuntimeContext(ContractModel):
    capability_runtime: CapabilityRuntimeContext
    services: RuntimeServices
```

`SupervisorV2State` is checkpointable and reducer-mutable; its nested business values remain strict/frozen. `GraphRuntimeContext` is injected, request-scoped, and never checkpointed. On resume, current runtime authorization always replaces historical authorization.

## 8. Request, conversation, and semantic contracts

### 8.1 Request

```python
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
```

`request_id` is the identity of the persisted user request and is consumed by chat/request correlation; it is not an execution-run identity. `run_id` and parent-run lineage belong to LangGraph/backend execution infrastructure. Trusted `user_id` exists only in current `CapabilityRuntimeContext`, never in checkpointed request semantics.

Known resources provide identity, not semantic role or revision. The Binding Resolver resolves the current authorized revision and role. An attachment is only a contextual candidate until semantics binds it.

V2 persists raw `original_query` before expansion or normalization.

### 8.2 Conversation

```python
class ConversationContext(ContractModel):
    summary: str
    active_entities: tuple[ActiveEntity, ...]
    last_focus: EntityReference | None
    recent_turns: tuple[ConversationTurn, ...]

class ConversationSnapshot(ContractModel):
    contract_version: Literal["2.0"]
    thread_id: str
    summary_version: int
    built_through_message_id: str | None
    context: ConversationContext
```

Conversation Context is short-term discourse state (“nghị định này”, “file thứ hai”). Memory is long-term user context (“đơn vị tôi”) and is an optional capability, not a default terminal domain. Pending unresolved user questions are owned by `ClarificationRequest`; no current discourse consumer needs a duplicate `open_questions` collection.

Rolling-summary persistence—not semantic context—owns optimistic locking through `summary_version` and monotonic `built_through_message_id`. `thread_id` appears once on the persisted snapshot.

### 8.3 Semantic lifecycle

```python
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
    requested_role: DocumentRole | None
    revision_requirement: RevisionRequirement | None = None
    resolution_status: Literal["unresolved", "resolved", "ambiguous", "not_found", "error"]
    resolved_document_id: UUID | None
    candidate_document_ids: tuple[UUID, ...] = ()

class SemanticDraft(ContractModel):
    provisional_contextualized_query: str
    abbreviations: tuple[AbbreviationResolution, ...]
    coreferences: tuple[CoreferenceResolution, ...]
    document_refs: tuple[DocumentReference, ...]
    person_refs: tuple[EntityReference, ...]
    section_refs: tuple[SectionReference, ...]
    preliminary_ambiguities: tuple[BlockingAmbiguity, ...]

class SemanticContext(ContractModel):
    contextualized_query: str
    normalized_query: str
    abbreviations: tuple[AbbreviationResolution, ...]
    coreferences: tuple[CoreferenceResolution, ...]
    document_refs: tuple[DocumentReference, ...]
    person_refs: tuple[EntityReference, ...]
    section_refs: tuple[SectionReference, ...]
    blocking_ambiguities: tuple[BlockingAmbiguity, ...]

class SemanticSnapshot(ContractModel):
    contract_version: Literal["2.0"]
    semantic: SemanticContext
```

`SemanticDraft` and small-model output are internal and not persisted. The Binding Resolver runs before semantic finalization. Only the finalized `SemanticSnapshot` is persisted.

`RequestContext.original_query` is the sole authoritative raw query and is immutable; SemanticContext does not copy it. `contextualized_query` resolves discourse references. `normalized_query` adds validated abbreviation/entity normalization. Bindings resolve from `document_refs` by `ref_id`; SemanticContext does not copy resulting binding IDs.

Document-reference invariants:

- `resolved` requires a canonical ID;
- `ambiguous` requires at least two candidates and no canonical ID;
- `not_found` has no canonical ID or candidates;
- `unresolved` means lookup has not completed;
- `error` means resolver/infrastructure failure, not ambiguity.

Small-model reuse uses an ephemeral internal `SemanticModelOutput`, never a persisted business contract. Query Analysis consumes validated facts from it and discards raw prose/reasoning.

### 8.4 Abbreviation reuse

```text
protect identifiers/literals
→ cheap candidate detection
→ batch DB lookup
→ unique normalize | ambiguous conditional disambiguation | unknown preserve
```

Reuse existing Phase-1 abbreviation contracts/validation behind preprocessing and `abbreviation.resolve`. Protect document numbers, CCCD/BHXH/phone/IDs, and quoted literals.

## 9. Document identity, binding, and target units

```python
DocumentRole = Literal["target", "reference", "supporting", "discovered"]

class ScopedDocument(ContractModel):
    binding_id: str
    document_id: UUID
    document_revision: str
    role: DocumentRole

class UserBindingProvenance(ContractModel):
    kind: Literal["user_reference"]
    binding_id: str
    source_ref_id: str

class DiscoveredBindingProvenance(ContractModel):
    kind: Literal["discovered"]
    binding_id: str
    source_task_id: str

class PromotedBindingProvenance(ContractModel):
    kind: Literal["promotion"]
    binding_id: str
    source_binding_id: str
    reason: str

BindingProvenance = Annotated[
    UserBindingProvenance | DiscoveredBindingProvenance | PromotedBindingProvenance,
    Field(discriminator="kind"),
]

class BindingAuditRow(ContractModel):
    contract_version: Literal["2.0"]
    provenance: BindingProvenance

class BindingRevisionRequirement(ContractModel):
    binding_id: str
    ref_id: str

class DocumentBindingSet(ContractModel):
    bindings: tuple[ScopedDocument, ...]
    revision_requirement_refs: tuple[BindingRevisionRequirement, ...]
```

`ScopedDocument` owns document ID, the immutable revision pinned for this run, and role only. `DocumentReference.revision_requirement` is optional and owns only an explicit revision instruction. `None` means an ordinary reference: Binding Resolver resolves it once, pins that revision for the run, and does not rebind merely because a newer revision later appears. `CurrentRevisionRequirement` is emitted only for explicit current/latest/hiện hành semantics and requires freshness revalidation on resume/reuse. `PinnedRevisionRequirement` selects a known specific/historical revision during initial resolution; the resulting `ScopedDocument.document_revision` is thereafter sufficient to enforce that exact pin.

`BindingRevisionRequirement` exists only for a binding whose source reference has `CurrentRevisionRequirement`, because resume/reuse is its sole current consumer. It maps the binding back to that authoritative reference without copying policy. Both IDs must resolve and each current-required binding has exactly one relation. Ordinary, explicitly pinned, and discovered bindings have no relation and validate their exact pinned `ScopedDocument.document_revision`. Promotion preserves the exact pin unless the user supplies a new explicit current/latest reference, which creates a newly resolved binding. Unresolved references are derived from `SemanticContext.document_refs.resolution_status`, not copied into DocumentBindingSet. Downstream reads consume only the pinned binding. Target/reference roles are required; supporting/discovered roles are optional. There is no redundant `required` boolean.

Locator and completion criteria belong to target units:

```python
class TargetUnit(ContractModel):
    target_id: str
    binding_id: str
    requested_locator: ContentLocator
    completion_criteria: tuple[CompletionCriterion, ...]
```

`binding_id` resolves document identity/revision/role from `DocumentBindingSet`. A binding may have multiple target units.

Binding lineage is a persisted audit concern represented by discriminated `BindingProvenance` in `BindingAuditRow`, not copied into hot-path bindings. This is an audit row, not an event-sourcing or replay contract. Each variant keeps only its reconstructable minimal path: user reference, discovery task, or source binding promotion. Only the Binding Resolver creates bindings. Planner outputs addition/promotion proposals; it cannot autonomously create targets.

Discovery lifecycle:

```text
document.search → DocumentDiscoveryCandidate
→ BindingAdditionRequest
→ Binding Resolver creates discovered/supporting binding
→ optional BindingPromotionRequest
→ Binding Resolver creates a new reference/supporting binding + BindingProvenance
```

Discovery never mutates an existing binding. Target addition requires deterministic explicit semantics or user clarification.

The proposal shapes are minimal and reference authoritative IDs:

```python
class DocumentDiscoveryCandidate(ContractModel):
    candidate_id: UUID
    document_id: UUID
    document_revision: str

class BindingAdditionRequest(ContractModel):
    candidate_id: UUID
    requested_role: Literal["discovered", "supporting"]

class BindingPromotionRequest(ContractModel):
    source_binding_id: str
    requested_role: Literal["reference", "supporting"]
```

A discovery candidate is consumed inside its originating `AgentResult`, whose `task_id` already supplies task lineage. `candidate_id` is a UUID allocated when the candidate is created and is globally unique, so aggregation across parallel discovery tasks cannot collide. Reasons and minimal trigger lineage are written once by the Binding Resolver to `BindingProvenance`; evidence IDs are not copied because `EvidenceUse → task_id` reconstructs evidence/task lineage.

Revision is pinned when a binding is created. Binding Resolver MUST resolve an immutable revision identity whose `ContentLocator` coordinate system remains stable for that revision. Downstream reads never silently switch to latest. On resume or reuse, only `CurrentRevisionRequirement` is reevaluated for freshness and may cause explicit rebinding/reacquisition. Ordinary references and explicit pinned references continue using the exact resolved `ScopedDocument.document_revision`. No requirement value is copied onto `ScopedDocument`.

## 10. Structured content locators

```python
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
    DocumentLocator | SectionLocator | ArticleLocator |
    PageRangeLocator | ChunkRangeLocator,
    Field(discriminator="kind"),
]
```

Stable ingestion structure IDs are canonical. Human-readable headings are presentation data. Planning, reading, coverage, evidence, and grounding use this coordinate system.

## 11. Authorization and capability boundary

```python
class CapabilityRuntimeContext(ContractModel):
    request_id: str
    run_id: str
    user_id: UUID
    workspace_ids: tuple[UUID, ...]
    can_read_people: bool
    allowed_capabilities: frozenset[str]
    deadline_at: datetime
```

```text
AgentRequest = requested operation.
CapabilityRuntimeContext = current trusted permission/scope.
Capability = executes request ∩ runtime authorization.
```

`config_revision` is absent from CapabilityRuntimeContext because no capability currently changes authorization or execution behavior from it; deployment revision remains trace/observability metadata.

The request-scoped registry is:

```text
base capabilities ∩ permissions ∩ feature flags ∩ service availability
```

Unauthorized capabilities are absent from planner catalogs and remain denied at execution. Models cannot supply workspace IDs or permission flags. Unauthorized document identity is invisible to clarification; an explicit unauthorized UUID returns `denied` without metadata leakage.

## 12. Minimal QueryAnalysis and RouteDecision

```python
WorkType = Literal[
    "direct", "lookup", "retrieve", "explain", "summarize",
    "compare", "evaluate", "cross_domain", "multi_goal",
]
Domain = Literal["people", "document", "section", "write", "knowledge_graph", "memory"]

class QueryAnalysis(ContractModel):
    work_type: WorkType
    domains: tuple[Domain, ...]
    dependency_hints: tuple[SemanticDependencyHint, ...] = ()

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

class RouteDecision(ContractModel):
    route: Route
    reason_code: RouteReason
```

Capability mapping belongs to router/planner policy, not semantic analysis. Semantic complexity is tracing only. Answer Policy decides synthesis; QueryAnalysis does not duplicate it.

Analysis is deterministic-first. Greetings, People lookup, exact Section retrieval, and bounded Write classify without a model. Uncertain cases may use a small model; if Context already called one, validated ephemeral output is reused.

Routing follows execution topology, not linguistic depth. One bounded domain operation without runtime dependency or iterative acquisition prefers fast path. Fast paths may still synthesize after one bounded read.

Every `fast_domain` route builds and checkpoints a deterministic one-task TaskPlan before dispatch. This uses no planner model, research DAG expansion, iterative acquisition, or replan. The fast-plan builder allocates `plan_id`, one `TaskSpec.task_id`, and only the TargetUnits needed by bounded document/read operations; People/KG/Write operations use `target_units=()`. It validates the same ID, binding, capability, and scope invariants as a complex plan. Fast and complex execution therefore converge on one persisted ownership graph: `task_id → TaskSpec`, `target_id → TargetUnit`, and `binding_id → ScopedDocument`.

A `direct` conversational response such as a greeting executes no capability and keeps `ExecutionState.plan=None`; a direct factual response is not permitted. Synthesis-only reuse builds the same minimal deterministic TaskPlan when it creates new run-specific EvidenceUses, so every new use has a persisted TaskSpec owner. Existing current-run uses may be revalidated and admitted without acquisition or another plan, but their task and target IDs still resolve through the checkpointed plan that originally created them.

## 13. Task planning and capability execution

### 13.1 Typed completion criteria

```python
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
```

Completion criteria belong only to logical `TargetUnit`s; the containing TargetUnit is the authoritative target, so CoverageCriterion does not repeat `target_id`. A target may contain at most one CoverageCriterion, and SemanticCriterion IDs must be unique within that target; plan validation rejects violations. Consequently `MissingRequirement(target_id, criterion_kind="coverage")` identifies at most one criterion, while a semantic missing requirement carries an ID that resolves uniquely in the same target. Coverage criteria are deterministic. `SemanticCriterion` is reserved for a concrete target-level judgment such as legal compliance, is created only by validated trusted planner policy, and goes to the semantic evaluator only as an evaluation objective. It cannot widen scope, select capabilities, issue tool instructions, or modify authorization/DiscoveryPolicy. Partial coverage requires explicit user/objective semantics and a deterministic reason.

Exact lookup and entity-resolution success are already authoritative typed capability status/output semantics. A raw minimum-evidence count has no current quality use case. Those generic criteria are removed rather than creating a mini rule engine.

### 13.2 Minimal task contracts

```python
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
    target_units: tuple[TargetUnit, ...]
    tasks: tuple[TaskSpec, ...]
```

`TaskPlan.goal` is the user-level goal; `TaskSpec.task_objective` is the operation-specific intent consumed by one capability. The deterministic fast-plan builder emits exactly one initial TaskSpec with no dependencies and no replan; the complex planner may emit a validated DAG and bounded append-only replan tasks. Task inputs reference `target_id`/`binding_id` when needed; they do not copy `ScopedDocument`. Runtime resolves IDs from the authoritative plan/binding set. `ReplanTaskOrigin.evidence_use_ids` preserves the triggering run/task/target/purpose context; bare evidence IDs remain appropriate only for content lineage such as DerivedSourceIdentity. Child semantic scope is a subset of parent bindings except policy-authorized reference discovery. Task execution success uses typed capability semantics; user-requirement criteria remain only on TargetUnit.

### 13.3 Minimal capability request/result

```python
AgentStatus = Literal["success", "partial", "not_found", "needs_input", "denied", "error"]

class AgentError(ContractModel):
    code: Literal[
        "INVALID_INPUT", "SCOPE_VIOLATION", "PERMISSION_DENIED",
        "AMBIGUOUS_ENTITY", "DEPENDENCY_UNAVAILABLE", "TIMEOUT",
        "CANCELLED", "BUDGET_EXHAUSTED", "CONTRACT_MISMATCH",
        "INTERNAL_ERROR",
    ]
    message: str
    retryable: bool

class AgentRequest(ContractModel):
    contract_version: Literal["2.0"]
    task_id: str
    objective: str
    input: CapabilityInput

class AgentResult(ContractModel):
    contract_version: Literal["2.0"]
    task_id: str
    status: AgentStatus
    data: CapabilityOutput | None
    evidence_uses: tuple[EvidenceUseRef, ...]
    coverage_observations: tuple[CoverageObservation, ...]
    error: AgentError | None
```

The scheduler resolves `TaskSpec.capability` from the current registry, then invokes the already-selected capability with `AgentRequest`; capability name is therefore not copied into the request. `TaskSpec` owns scheduler dependencies. `AgentRequest` carries only task execution data. Runtime owns request/run IDs. Evaluator—not capability—derives missing requirements.

`AgentResult.task_id` is intentionally retained for asynchronous fan-in, retry, and checkpoint association. `AgentResult.status` describes task execution, not global sufficiency. `success != sufficient`.

Because AgentResult is checkpointed, `AgentResult.data` means only the minimized, schema-validated, checkpoint-safe `CapabilityOutput` produced at the capability boundary. A connector's raw response is an ephemeral internal value: it is minimized before any AgentResult update and is never checkpointed, logged as business state, or returned to the graph. Sensitive or retention-governed content is persisted only as a governed EvidenceRecord and represented in AgentResult by EvidenceUseRef; in that case `data` is `None` or contains only non-sensitive operational facts (for example count/status), never the sensitive payload. Any checkpoint field that remains personal under an explicitly approved domain requirement must receive retention, ACL, encryption, deletion, and audit controls equivalent to the Evidence Store rather than bypassing them.

### 13.4 Typed capability I/O

`CapabilityInput` and `CapabilityOutput` are discriminated unions of domain-specific models:

```python
CapabilityInput = Annotated[
    PeopleLookupInput | DocumentSearchInput | DocumentReadInput |
    SectionReadInput | WriteInput | KnowledgeGraphInput |
    MemoryLookupInput | AbbreviationResolveInput,
    Field(discriminator="kind"),
]

CapabilityOutput = Annotated[
    PeopleLookupOutput | DocumentSearchOutput | DocumentReadOutput |
    SectionReadOutput | WriteOutput | KnowledgeGraphOutput |
    MemoryLookupOutput | AbbreviationResolveOutput,
    Field(discriminator="kind"),
]
```

Each variant carries only domain-required fields and is owned by its capability module. No `dict[str, Any]` escape hatch is allowed. Each capability module's contract spec must classify every output field as checkpoint-safe or evidence-only; unclassified fields fail boundary validation. Raw provider/database output is not a CapabilityOutput.

`document.search` returns discovery candidates. `document.read`/`section.read` produce authoritative coverage observations. Search success never means read-complete coverage.

Fast and complex paths invoke the same capability implementations; only orchestration differs.

## 14. Minimal Coverage

```python
CoverageOutcome = Literal["read", "missing", "unreadable", "truncated"]
CoverageStatus = Literal["read_complete", "read_partial", "missing", "unreadable", "truncated"]

class CoverageObservation(ContractModel):
    target_id: str
    observed_locators: tuple[ContentLocator, ...]
    outcome: CoverageOutcome

class CoverageItem(ContractModel):
    target_id: str
    observed_locators: tuple[ContentLocator, ...]
    status: CoverageStatus

class Coverage(ContractModel):
    items: tuple[CoverageItem, ...]
```

Evaluator resolves `target_id → TargetUnit → binding_id → ScopedDocument` for requested locator, document revision, and role. Those facts are not copied into coverage.

A revision mismatch is found by resolving the EvidenceRecord source behind each use and comparing it with the binding; it cannot complete coverage.

## 15. Evidence identity versus evidence use

### 15.1 Canonical evidence identity

```python
class DocumentSourceIdentity(ContractModel):
    kind: Literal["document"]
    document_id: UUID
    document_revision: str
    locator: ContentLocator

class PeopleSourceIdentity(ContractModel):
    kind: Literal["people"]
    record_id: str

class KnowledgeGraphSourceIdentity(ContractModel):
    kind: Literal["knowledge_graph"]
    entity_or_relation_id: str

class MemorySourceIdentity(ContractModel):
    kind: Literal["memory"]
    memory_id: str

class DerivedSourceIdentity(ContractModel):
    kind: Literal["derived"]
    source_evidence_ids: tuple[UUID, ...]

EvidenceSourceIdentity = Annotated[
    DocumentSourceIdentity | PeopleSourceIdentity | KnowledgeGraphSourceIdentity |
    MemorySourceIdentity | DerivedSourceIdentity,
    Field(discriminator="kind"),
]

class Provenance(ContractModel):
    acquisition_id: UUID
    fetcher: str
    fetched_at: datetime

class EvidenceRecord(ContractModel):
    evidence_id: UUID
    source: EvidenceSourceIdentity
    content: str
    content_hash: str
    provenance: Provenance
```

EvidenceRecord says what the evidence is. It contains no run/task/target usage, storage policy, UI labels, or free-form metadata. `DocumentSourceIdentity.document_id + document_revision` MUST resolve an authoritative revision record that owns workspace membership; ACL checks use that record under current runtime authorization rather than trusting copied workspace metadata.

One acquisition/tool call may yield multiple minimized EvidenceRecords (for example, several source chunks), so `Provenance.acquisition_id` is the intentional grouping identity and is not equivalent to `evidence_id`.

### 15.2 Evidence use

```python
EvidencePurpose = Literal["discovery", "coverage", "supporting"]

class EvidenceUse(ContractModel):
    use_id: UUID
    evidence_id: UUID
    task_id: str
    purpose: EvidencePurpose
    target_id: str | None

class EvidenceUseEnvelope(ContractModel):
    contract_version: Literal["2.0"]
    run_id: str
    use: EvidenceUse

class EvidenceUseRef(ContractModel):
    use_id: UUID
```

EvidenceUse says how the current run/task uses immutable evidence. Resolution is total and deterministic:

- every use belongs to the current `EvidenceUseEnvelope.run_id`, resolves `task_id` to exactly one TaskSpec in the current checkpointed TaskPlan, and resolves `evidence_id` to one current-authorized EvidenceRecord;
- `purpose="coverage"` requires a non-null `target_id` resolving to a TargetUnit in that plan. Document role/revision/locator then derive through its binding and ScopedDocument;
- `purpose="discovery"` requires `target_id=None` and is ineligible for synthesis or claim support;
- a `supporting` use with a target follows the same target/binding path; a targetless supporting use is synthesis-eligible only for a validated targetless TaskSpec, with relevance derived from its task_objective/input and source kind derived from its EvidenceRecord. Its document role is `None` because no canonical binding path exists.

Any unresolved, cross-run, target-incompatible, or purpose-incompatible use is rejected rather than partially hydrated. Cross-run reuse revalidates ACL, retention, current revision requirements, locator, source availability, and semantic compatibility, then creates a **new EvidenceUse**. It does not mutate/copy EvidenceRecord and requires no adoption union in graph contracts. `EvidenceUseRef → EvidenceUse → EvidenceRecord` is the only graph reference chain; no bare EvidenceRef has a current graph consumer.

`EvidenceUseEnvelope.run_id` is retained because the Evidence Use Store is keyed by run; run identity is deliberately not copied into `EvidenceUse`. `binding_id` and `target_id` remain run-local. Semantic identity is resolved from document ID + revision + locator + role.

### 15.3 Evidence Store metadata and governance

```python
class StoragePolicy(ContractModel):
    classification: Literal["normal", "personal", "sensitive_personal"]
    expires_at: datetime | None

class EvidenceStoreRow(ContractModel):
    contract_version: Literal["2.0"]
    record: EvidenceRecord
    storage_policy: StoragePolicy
```

Storage policy belongs to Evidence Store, not semantic evidence. Classification is deterministic and cannot be downgraded by a model. `expires_at` is persisted deliberately as the stable deletion deadline selected at insertion time, so later policy/config changes cannot silently extend retention; no business/model consumer reads it.

Before persistence:

```text
raw capability result
→ Evidence Builder/Minimizer
→ deterministic PII classification
→ EvidenceRecord + StoragePolicy
→ Evidence Store
```

The store enforces authorization-bound access, encryption at rest, TTL/deletion, PII minimization, and audited access. People evidence stores only task-required fields.

Evidence insertion is idempotent by source identity + content hash. EvidenceUse append is idempotent by run/task/evidence/purpose/target. Retry cannot create uncontrolled duplicates.

Derived evidence has no copied document/revision/locator fields. A derived EvidenceRecord is synthesis-eligible only after deterministic/semantic faithfulness validation against every recursively resolved source EvidenceRecord. Failed or unvalidated derivations are excluded. Derived evidence never creates authoritative read coverage by itself; validation state is Evidence Store metadata, not a field on semantic EvidenceRecord.

## 16. Discovery and research policy

```python
class DiscoveryPolicy(ContractModel):
    allow_reference_discovery: bool
    allow_supporting_discovery: bool
    max_discovered_documents: int

class ResearchBudgetView(ContractModel):
    max_tasks_remaining: int
    max_replans_remaining: int
    max_parallel_branches: int

class CapabilityDescriptor(ContractModel):
    name: str
    domain: Domain
    operation_type: Literal["lookup", "search", "read", "transform", "resolve"]
    supports_parallel: bool

class ResearchPlanningInput(ContractModel):
    semantic: SemanticContext
    bindings: DocumentBindingSet
    query_analysis: QueryAnalysis
    capability_catalog: tuple[CapabilityDescriptor, ...]
    discovery_policy: DiscoveryPolicy
    budget: ResearchBudgetView
    prior_evidence: tuple[EvidenceUseRef, ...]
    prior_evaluation: EvidenceEvaluation | None
```

Authorization answers where access is permitted. Bindings answer which documents have semantic roles. DiscoveryPolicy answers only whether/how far research may expand. Workspace search permission derives from current runtime authorization; available discovery tools derive from the request-scoped registry entries with `domain="document"` and `operation_type="search"`. Policy restricts but never grants permission or copies an allowlist.

Planner behavior does not depend on deployment/adapter version, so descriptor version remains registry/config metadata. Document search discovers and document/section read produces coverage by operation contract; duplicate booleans are unnecessary. `supports_parallel` remains because the planner currently uses it to choose safe DAG fan-out and operation type alone does not determine concurrency safety.

`ResearchBudgetView` is ephemeral runtime-derived planner input, never persisted or versioned. The planner consumes `max_tasks_remaining` and `max_replans_remaining` to avoid impossible plans, and consumes `max_parallel_branches` because it explicitly chooses fan-out width; the scheduler still enforces all three.

Planner input excludes full chat history, DB sessions, raw clients, ACL internals, and legacy state.

## 17. Complex research behavior

```text
ResearchPlanningInput
→ typed TaskPlan
→ deterministic validation
→ dispatch dependency-ready TaskSpec
→ AgentRequest + current CapabilityRuntimeContext
→ AgentResult + EvidenceUseRef + CoverageObservation
→ Coverage + EvidenceEvaluation
→ sufficient synthesis | clarify | bounded append-only replan | typed failure
```

Completed tasks are immutable. Replans append tasks with `ReplanTaskOrigin`. Runtime enforces branches, task/tool/replan limits, deadline, and cancellation.

LLM outputs are proposals. Model-produced QueryAnalysis, TaskPlan, binding requests, semantic evaluation, and answer claims require schema plus deterministic policy validation. Models never mutate bindings, coverage, runtime authorization, evidence, or persisted canonical identities.

All retrieved document, People, knowledge-graph, memory, and EvidenceRecord content is untrusted data. Retrieved content cannot alter the system/user objective, authorization, workspace scope, DiscoveryPolicy, capability registry, TaskPlan, or evaluator/grounding policy; it cannot instruct runtime tool execution. Prompt/tool instructions embedded in a PDF or retrieved record remain evidence content to analyze, never control-plane instructions.

Cancellation propagates from the LangGraph run to pending tasks and capability tokens/deadlines. Cancellation prevents replan/synthesis; incomplete work never becomes success.

## 18. Minimal evaluation contract

```python
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

class EvidenceEvaluation(ContractModel):
    status: Literal["sufficient", "insufficient", "contradictory", "needs_input"]
    coverage: Coverage
    missing: tuple[MissingRequirement, ...]
    contradictions: tuple[Contradiction, ...]
```

Evaluator owns sufficiency, missing requirements, and conflicts. Contradictions reference admitted current-run EvidenceUses—not bare records—so target and purpose context remains deterministic when `prior_evaluation` drives replanning. Bare evidence IDs are reserved for content lineage such as `DerivedSourceIdentity`. Each MissingRequirement identifies its authoritative TargetUnit plus criterion kind; `semantic_criterion_id` is required only for `criterion_kind="semantic"` and forbidden for coverage. This avoids a second generic requirement identity and does not introduce a rule engine. Evaluator does not suggest next capabilities; replanner owns next action.

A supported contradiction may be `sufficient` for a comparison objective. `contradictory` is reserved for unresolved conflict preventing a safe conclusion.

## 19. Minimal synthesis and grounding boundary

### 19.1 Runtime hydration

```python
class SynthesisInput(ContractModel):
    semantic: SemanticContext
    evaluation: EvidenceEvaluation
    evidence_uses: tuple[EvidenceUseRef, ...]

class SynthesisRuntimeContext(ContractModel):
    max_evidence_items: int
    max_total_chars: int
    max_total_tokens: int

class SynthesisEvidence(ContractModel):
    use_id: UUID
    content: str
    role: DocumentRole | None
    target_id: str | None
    source_label: str | None
```

`SynthesisInput` is only constructed for factual/domain synthesis after EvidenceEvaluation exists and has `status="sufficient"`; the boundary rejects every other status. Direct non-factual responses such as greetings do not enter factual synthesis. SynthesisInput contains only semantic/evaluation facts and use references consumed by synthesis. Evidence Hydrator—not the model—separately receives the current TaskPlan (deterministic fast or complex), current DocumentBindingSet, current GraphRuntimeContext, and SynthesisRuntimeContext; resolves uses and records; applies the purpose/target rules in §15.2; revalidates current ACL/retention/revision/source availability; applies runtime budget; and produces ephemeral synthesis evidence. Full plans, bindings, and runtime policy are hydrator dependencies, not model-facing synthesis input.

The model does not see workspace IDs, document revisions, locators, or internal provenance. `SynthesisEvidence.target_id` groups evidence by logical unit during comparison; `role` explicitly distinguishes target/reference/supporting prompt semantics that an opaque target ID cannot communicate to the model. `source_label` is an ephemeral human-readable prompt label. All three are computed projections from admitted EvidenceUse plus canonical plan/binding/document stores and are never persisted back into EvidenceRecord.

The hydrator projects each admitted current-run EvidenceUse as `SynthesisEvidence(use_id, ...)` and internally retains its `(use_id, evidence_id)` resolution. Every `AnswerClaim.evidence_use_ids` entry must be a unique member of that admitted use-ID set; discovery-only, omitted, stale, denied, target-incompatible, and merely store-visible uses are rejected. Grounding follows the selected use to its EvidenceRecord, preserving the intended run/task/target/purpose context even when multiple uses point to the same evidence.

Overflow is compacted/map-reduced before final synthesis. Derived summaries are persisted as EvidenceRecords with source lineage and receive a new validated supporting EvidenceUse for the current run; final synthesis receives only their admitted use IDs/content. Grounding recursively validates the derived source lineage.

### 19.2 One claim-to-EvidenceUse mapping

```python
class AnswerClaim(ContractModel):
    claim_id: str
    text: str
    evidence_use_ids: tuple[UUID, ...]

class AnswerDraft(ContractModel):
    content: str
    claims: tuple[AnswerClaim, ...]
```

There is no duplicate `CitationRef` relationship. The model proposes claims and admitted EvidenceUse IDs. Grounding resolves `use_id → EvidenceUse → EvidenceRecord` and validates current-run purpose/target compatibility, claim support, pinned revisions, and locators.

```python
class RenderedCitation(ContractModel):
    citation_id: str
    evidence_id: UUID
    label: str

class FinalResponse(ContractModel):
    contract_version: Literal["2.0"]
    status: Literal["success", "clarify", "denied", "insufficient", "error"]
    content: str
    citations: tuple[RenderedCitation, ...] = ()
```

Grounding uses an internal claim-to-evidence association while validating `AnswerClaim`; it is not a user-facing contract. Current frontend/API citation rendering consumes source identity and label but has no claim-highlighting interaction, so `RenderedCitation` omits `claim_id`. CitationRenderer creates deterministic citation IDs only after Grounding. FinalResponse does not persist a derivable evidence-ID projection.

Grounding flow:

```text
AnswerDraft.claim.evidence_use_ids
→ EvidenceUse → EvidenceRecord
→ source identity + pinned revision + locator
→ semantic support check
→ validated claims
→ CitationRenderer
→ FinalResponse
```

Every material factual assertion in `AnswerDraft.content` must correspond to exactly one declared `AnswerClaim`; Grounding rejects factual prose outside the claim set. A revision request creates a wholly new AnswerDraft and claim mapping, then grounds once more. Unmapped factual assertions or unsupported claims on the second attempt return `insufficient`.

`GroundingResult` is an implementation-internal grounding-graph decision (`pass | revise | insufficient`) and diagnostics object. It is not a versioned business boundary or root checkpoint field because no interrupt, API, or replay consumer currently requires it.

Direct non-factual responses such as greetings require no evidence. Factual fast paths use the same Evidence Store/Use contracts as complex paths.

## 20. Clarification

```python
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

class ClarificationResolution(ContractModel):
    contract_version: Literal["2.0"]
    clarification_id: str
    selected_candidate_id: str | None
```

DocumentBindingSet does not copy the derivable unresolved set; Router derives it from `SemanticContext.document_refs.resolution_status`. `ClarificationRequest.unresolved_ref_ids` remains because it owns the specific persisted subset this question asks the user to resolve rather than the global unresolved projection. Candidate identity/order is stable within a clarification. `selected_candidate_id` is the authoritative deterministic selection; the raw clarification reply remains authoritative in ChatMessage persistence and is not copied here. `ClarificationRequest.expires_at` is retained because resume consumes a stable per-request deadline that must not change after deployment TTL configuration changes. Resume validates expiry, candidate membership, and current ACL, then asks Binding Resolver to create a binding. It does not rerun unconstrained ambiguity resolution.

## 21. Minimal ExecutionState

```python
class ExecutionState(ContractModel):
    plan: TaskPlan | None
    task_results: tuple[AgentResult, ...]
    evidence_evaluation: EvidenceEvaluation | None
```

There is no aggregate `evidence_refs`. Consumers call `collect_evidence_use_refs(task_results)` when needed.

## 22. Canonical routing table

| Query | Route |
|---|---|
| Xin chào | direct |
| CCCD của A là gì | fast / People |
| Điều 5 A nói gì | fast / Section |
| Giải thích Điều 5 A | fast / Section + synthesis |
| Kiểm tra chính tả đoạn này | fast / Write |
| A thuộc đơn vị nào | fast / KG |
| Tóm tắt Chương II A | fast if exact bounded read is supported |
| Tóm tắt toàn bộ tài liệu rất dài | complex map/reduce |
| So sánh Chương II A và III B | complex |
| CCCD của A xuất hiện trong nghị định nào | complex People → Document |
| Kiểm tra F1/F2 theo A | complex target/reference |
| Kiểm tra F1/F2 theo quy định hiện hành | complex with discovery enabled |
| Dựa vào A sửa F1 | complex Document → Write |

Summary complexity depends on required read topology, not `work_type`. Comparison defaults to complex, but when current valid EvidenceUses already cover both bounded inputs, Router may choose synthesis-only without a research DAG.

## 23. Core invariants

1. `RequestContext.original_query` is immutable and is the only raw-query owner.
2. Current authorization is runtime-injected and never checkpointed.
3. LLM cannot create workspace authorization or permission.
4. Context resolution never grants access.
5. Semantic DocumentReference optionally owns an explicit current/pinned RevisionRequirement; `None` means resolve once and pin. Binding Resolver alone owns resolved document role and pinned revision.
6. Target/reference roles are required; supporting/discovered are optional.
7. Attachments are candidates, not automatic targets.
8. Required targets are never silently replaced by discovered documents.
9. Planner cannot autonomously add targets.
10. Child tasks cannot exceed semantic bindings except policy-authorized reference discovery.
11. TaskPlan.TargetUnit owns locator/criteria for every factual document/read requirement; targetless People/KG/Write tasks have no TargetUnit; binding owns document identity/revision/role.
12. Search does not complete read coverage.
13. Coverage references target IDs and does not duplicate binding identity.
14. Revision mismatch cannot complete coverage.
15. Timeout/outage/cancellation/budget exhaustion are not `not_found`.
16. Completed task records are immutable; replans are bounded and append-only.
17. Capability results contain no LangGraph routing.
18. EvidenceRecord and EvidenceUse are distinct authoritative facts.
19. Cross-run reuse creates a new validated EvidenceUse, not copied evidence.
20. Evidence content is minimized before persistence.
21. Evidence Store owns retention, classification, ACL, encryption, and audit.
22. Capability inputs/outputs are typed; no generic dict escape hatch.
23. `AgentResult.success` does not imply `EvidenceEvaluation.sufficient`.
24. Evaluator does not plan next actions.
25. AnswerDraft stores one claim-to-EvidenceUse relationship; content lineage alone uses bare evidence IDs.
26. Citation IDs are deterministic presentation output, not model authority.
27. Grounding runs before factual success.
28. Only outer answer layer streams final response.
29. Trusted content and retrieved content are separate control/data planes; retrieved instructions cannot change route, plan, capability, scope, or policy.
30. Every material factual assertion in AnswerDraft content is represented by an AnswerClaim.
31. Derived evidence must be validated against recursive sources and never creates read coverage.
32. Document revision records authoritatively resolve workspace membership and stable locator coordinates.
33. Every factual EvidenceUse task ID resolves to a TaskSpec, and every non-null target ID resolves to a TargetUnit, in the checkpointed TaskPlan.
34. Factual/domain synthesis requires a sufficient EvidenceEvaluation; direct non-factual responses bypass that boundary.

## 24. Field ownership summary

For every derivable fact the decision is explicit: **REMOVE**, **KEEP AS EPHEMERAL PROJECTION**, or a measured cache exception. This design has no canonical denormalized-cache exception.

| Field/fact | Authoritative owner | Producer | Current consumer | Persisted? | Derivable? | Decision and reason |
|---|---|---|---|---|---|---|
| `RequestContext.request_id/thread_id/original_query/known_documents` | Chat/request persistence | ingress | context, binding, request correlation | yes | no | KEEP: persisted request semantics |
| current `user_id`, run lineage, ACL/deadline | backend runtime | authenticated ingress/runtime | capability and evidence services | no business state | refreshed | REMOVE from RequestContext; runtime only; deployment/config revision is trace metadata |
| `ConversationContext.summary/entities/focus/recent_turns` | conversation layer | context builder | coreference/follow-up resolver | snapshot | no | KEEP: discourse state |
| `ConversationSnapshot.thread_id/summary_version/built_through_message_id` | chat persistence | summary writer | snapshot loader/CAS writer | yes | no | KEEP on persistence envelope only |
| finalized contextualized/normalized query and references | Semantic Finalizer | semantic pipeline | analyzer, router, planner | snapshot | no | KEEP: one final query meaning; raw query remains only in RequestContext |
| optional explicit `RevisionRequirement` | semantic reference resolution | semantic/reference resolver before binding | Binding Resolver; resume/reuse only for current requirement | finalized snapshot | no | KEEP when explicit: resolver validates it before binding; Semantic Finalizer only persists the resolved fact; `None` is ordinary pin-once |
| `binding_id → document/revision/role` | Binding Resolver | binding graph | planner, runtime, evaluator, hydrator | checkpoint | no | KEEP: canonical bound document fact |
| binding provenance variants | binding audit | Binding Resolver | audit/security investigation | yes | no | KEEP in `BindingAuditRow`, not hot path |
| `target_id → binding/locator/criteria` | TaskPlan.TargetUnit | deterministic fast-plan builder or complex planner | document/read capability, evaluator, hydrator | checkpoint | no | KEEP: canonical logical document/read requirement; targetless People/KG/Write tasks have none |
| `plan_id/goal/tasks` | TaskPlan | deterministic fast-plan builder or complex planner | validator, scheduler, evaluator | checkpoint | no | KEEP: canonical factual execution owner; fast emits one task, complex may emit a DAG |
| invocation `task_id` | TaskPlan.TaskSpec | deterministic fast-plan builder or complex planner | scheduler, capability, fan-in, EvidenceUse resolver | checkpoint plus result/use refs | no | KEEP: canonical invocation identity across resume |
| capability/objective/input/dependencies/origin | TaskPlan | deterministic fast-plan builder or complex planner | scheduler; selected capability consumes objective/input | checkpoint | no | KEEP: operation definition; fast has one initial dependency-free task |
| `AgentResult.task_id/status/data/use refs/coverage` | capability boundary | capability | async fan-in, checkpoint, evaluator | checkpoint | data field-specific | KEEP; task ID associates parallel/retried results; raw/sensitive data excluded |
| document details repeated on TargetUnit/Coverage | binding/plan graph | n/a | resolver lookup | no duplicate | yes | REMOVE; follow IDs |
| `evidence_id → source/content/provenance` | Evidence Store | Evidence Builder | hydrator, grounding | yes | no | KEEP: immutable global evidence identity |
| `Provenance.acquisition_id/fetcher/fetched_at` | Evidence Store | Evidence Builder | grounding/audit | yes | no | KEEP: one acquisition can produce multiple records |
| source workspace | document revision record | document store | ACL validator | no evidence copy | yes | REMOVE from DocumentSourceIdentity; authoritative lookup |
| `use_id → evidence/task/target/purpose` and envelope run | Evidence Use Store | validated capability/reuse flow | evaluator, hydrator | yes | no | KEEP: contextual usage; run remains envelope-only |
| bare EvidenceRef / evidence envelope | Evidence Store | n/a | none | no | yes/direct ID | REMOVE: no current boundary consumer |
| storage classification/expiry | Evidence Store | insertion policy | retention/deletion worker | yes | expiry selected once | KEEP in StoragePolicy only; stable deletion deadline |
| DiscoveryPolicy three limits | policy layer | deterministic policy builder | planner/validator | checkpoint with plan input only | no | KEEP: authorized expansion semantics |
| available discovery capabilities/workspace search | runtime registry/ACL | backend runtime | planner filtering | no policy copy | yes | REMOVE copied fields; compute intersection |
| CapabilityDescriptor name/domain/operation/parallel support | capability registry | registry builder | planner | ephemeral | no for parallel safety | KEEP minimal planner catalog; version/derived behavior removed |
| ResearchBudgetView three counters | runtime budget | orchestrator | planner DAG sizing | no | yes per call | KEEP AS EPHEMERAL PROJECTION: planner selects task/replan/fan-out width |
| `SynthesisInput.semantic/evaluation/use refs` | semantic/evaluator/use store | orchestration | synthesizer/hydrator | no separate persistence | no | KEEP minimal factual request; evaluation is mandatory and sufficient; current plan plus bindings/runtime go only to hydrator |
| `SynthesisEvidence.role/target_id/source_label` | plan/binding/document stores | Hydrator/Presentation | synthesizer | no | yes | KEEP AS EPHEMERAL PROJECTION: unit grouping, role semantics, readable prompt source |
| hydrated admitted evidence set | Evidence Hydrator | hydrator | Grounding | no | yes per call | KEEP AS EPHEMERAL PROJECTION: constrain claim IDs |
| `AnswerClaim.claim_id/text/evidence_use_ids` | Answer layer | synthesizer | Grounding | internal draft only | no for selected use context | KEEP: standalone grounding unit preserving admitted current-run task/target/purpose context |
| `RenderedCitation.citation_id/evidence_id/label` | presentation | CitationRenderer | current API/frontend | response | ID/label derivable | KEEP AS RESPONSE PROJECTION; no unused claim ID |
| clarification candidate/selection/expiry | clarification layer | resolver/runtime | UI and resume validator | yes | raw reply in chat | KEEP minimal stable identity/selection/deadline; copied user text removed |
| domain input/output fields | named capability module | capability | named domain consumer | classified per field | field-specific | KEEP only when module contract identifies current consumer; no generic dict |
| `GroundingResult` diagnostics | grounding graph | Grounding | grounding control flow | no/root absent | n/a | KEEP INTERNAL, not a versioned business contract |
| routing semantic complexity and debug ranking | telemetry | analyzer/resolver | observability | trace only | n/a | REMOVE from business contracts |

## 25. Superseded contract disposition

This repository has no production v2 graph, v2 checkpoint, Evidence Store v2 row, or other persisted v2 artifact: all `"2.0"` shapes at commit `26ed74f` were architecture-spec examples only and were never runtime or persistence contracts. “Superseded” below therefore means superseded design prose, not an in-place production data migration.

The following proposed shapes from commit `26ed74f` are superseded by this minimal baseline:

| Superseded shape | Disposition |
|---|---|
| version on every nested model | version only persisted/external envelopes |
| `ScopedDocument.required/locator/revision_policy/lineage` | required derived from role; locator in TargetUnit; optional RevisionRequirement stays with semantic DocumentReference; lineage in BindingProvenance |
| TargetUnit copies document/revision/role | resolve through `binding_id` |
| Coverage copies binding/document/revision/role/request | resolve through `target_id` |
| `ExecutionState.evidence_refs` | derive from task results |
| QueryAnalysis capability/complexity/synthesis fields | policy/tracing/Answer Policy, not semantic contract |
| persisted `SemanticAnalysisHints` | ephemeral internal `SemanticModelOutput` |
| `RouteDecision.domain` | read domains from QueryAnalysis |
| TaskSpec copied ScopedDocuments + completion criteria + three replan fields | ID-bearing typed input + `TaskOrigin`; criteria remain only on TargetUnit and replan reasoning references EvidenceUse IDs |
| AgentRequest scheduler/evaluator/runtime/capability fields | selected capability receives minimal task/objective/input |
| AgentResult request ID + missing requirements | runtime/evaluator ownership |
| EvidenceRecord mixed identity/use/storage/presentation | split EvidenceRecord, EvidenceUse, StoragePolicy, presentation |
| metadata dict | typed fields or telemetry; removed |
| verbose/bare EvidenceRef and AdoptedEvidenceRef | only EvidenceUseRef; reuse creates new EvidenceUse |
| SynthesisBudget in business input | SynthesisRuntimeContext |
| duplicate AnswerClaim + CitationRef mapping | AnswerClaim only; RenderedCitation after grounding |
| FinalResponse evidence-ID projection | derive from citations |
| evaluator research suggestions | replanner ownership |
| candidate ranking/debug/source-task copies | resolver telemetry/container AgentResult; discovery candidate uses globally unique UUID; clarification keeps identity/order/label only |
| EvidenceRecordEnvelope and BindingEventEnvelope | EvidenceStoreRow and BindingAuditRow are the actual persisted boundaries |
| workspace copied into document evidence | resolve workspace from authoritative document revision record |
| ExactLookup/EntityResolution/MinimumEvidence criteria | typed capability semantics; no generic rule-engine/count criteria |
| RenderedCitation.claim_id | internal grounding association only; current presentation has no claim interaction |
| ClarificationResolution.user_text | authoritative raw reply remains in ChatMessage |

Existing Phase-1 `PreprocessingResult`, abbreviation/reference validators, and semantic-context DB column remain reuse candidates through explicit v1-to-v2 adapters. `backend/app/services/agents/deep_research/` contracts are legacy inputs/outputs only: an adapter may translate a validated subset at the v2 boundary, but those models do not become v2 canonical contracts or enter v2 checkpoint state. Phase-1 `RoutingDecision`/SupervisorState additions remain v1 compatibility and are not completed as v2 architecture.

Because no production v2 persisted artifact exists, v2 startup/load code must reject incompatible pre-release fixtures or checkpoints rather than attempt best-effort migration. Test fixtures claiming version `2.0` must conform to this revision's envelope schemas; older spec-derived fixtures are regenerated. V1 production chat data remains authoritative and is read through the named adapters.

## 26. Validation and acceptance scenarios

Contract tests must prove:

- root-versioned mutable SupervisorV2State, strict/frozen nested persisted business contracts, and unversioned runtime context;
- every factual fast route checkpoints a deterministic one-task TaskPlan without planner LLM/replan, while complex routes use planner-generated DAGs;
- ID referential integrity across bindings, targets, tasks, evidence, and uses;
- raw/sensitive capability output cannot enter checkpointed AgentResult.data and evidence-only fields produce refs instead;
- no duplicated canonical identity in TargetUnit/Coverage;
- typed capability input/output without generic dict fallback;
- criteria are target-contained without duplicate target IDs, MissingRequirement structurally references target/criterion kind, and no obsolete resolved coverage state exists;
- search cannot produce read-complete coverage;
- ordinary document references use `revision_requirement=None`, resolve once, and remain pinned when a newer revision appears; explicit current/latest requirements trigger freshness validation, and explicit pinned requirements select the requested immutable revision;
- revision-requirement relation integrity: exactly one relation for every current-required binding, none for ordinary/pinned/discovered bindings, and rejection of dangling or mismatched binding/ref IDs;
- revision mismatch rejects coverage/evidence reuse and authoritative document revision lookup supplies workspace ownership;
- cross-run reuse creates a new EvidenceUse after current ACL/revision/retention checks;
- coverage uses require valid targets, discovery uses cannot synthesize, and targetless supporting uses require a validated targetless task;
- raw People output is minimized before EvidenceRecord persistence;
- factual/domain SynthesisInput requires an existing sufficient EvidenceEvaluation; direct greetings never enter factual synthesis;
- every claim EvidenceUse ID belongs to the hydrated current-run admitted-use set, and two uses of one EvidenceRecord preserve the selected target/purpose context;
- every material factual assertion is represented by an AnswerClaim; an unmapped assertion is revised then becomes insufficient on second failure;
- derived evidence must pass recursive source-faithfulness validation and cannot create read coverage;
- claim/evidence grounding and deterministic citation rendering without a user-facing claim ID;
- direct greetings require no evidence while factual fast paths do;
- runtime context and trusted user/run identity never serialize into RequestContext/checkpoint business state;
- RequestContext alone owns original_query; SemanticDraft and SemanticContext do not copy it, SemanticContext carries no binding-ID projection, DocumentBindingSet has no unresolved projection, and ClarificationRequest persists only its question-specific ref IDs;
- retrieved prompt/tool-injection instructions cause no route, capability, scope, policy, plan, or tool-execution change and remain evidence data only;
- incompatible pre-release v2 fixtures/checkpoints are rejected rather than migrated best-effort.

Required scenario set includes simple People/Section/Write/KG, conversational and ambiguous follow-up, multi-document range comparison, People→Document, target/reference compliance, reference discovery, permission denial, resume ACL change, abbreviation normalization, wrong/partial section coverage, irrelevant attachments, unresolved named documents, discovered-reference promotion, execution-success/evidence-insufficient, document revision change, evidence expiry/reuse denial, search-without-read, fast-path evidence, synthesis-budget overflow, and synthesis-only comparison with already-valid evidence.

Explicit regression scenarios additionally prove:

- fast Section checkpoint/resume resolves both task and target through the persisted one-task TaskPlan without a fast-specific target store;
- a normal document reference (`revision_requirement=None`) pins once, whereas explicit current/latest semantics revalidate freshness;
- when two EvidenceUses point to one EvidenceRecord, an AnswerClaim selects one `use_id` and Grounding preserves its target/purpose;
- evidence-triggered replan records `evidence_use_ids`, preserving execution context;
- parallel discovery tasks generate non-colliding candidate UUIDs.

## 27. Migration and rollout

### Phase 0 — Orchestrator compatibility spike

Benchmark native LangGraph versus Deep Agents adapter against the same minimal contracts. This selects implementation, not contract meaning.

### Phase 1 — Minimal contracts and adapters

**Prerequisite — immutable DocumentRevision persistence.** Before implementing binding, evidence, or coverage, establish an authoritative immutable revision identity such that `(document_id, document_revision)` resolves exactly one revision record owning workspace membership, stable parsed structure, and a stable `ContentLocator` coordinate system. The implementation plan chooses the storage key/design; this architecture does not invent content/parser/structure/ingestion sub-revisions without a demonstrated need.

After that prerequisite, implement envelopes, binding/target identity graph, deterministic fast-plan builder, minimal evidence/use stores, Phase-1 and legacy `deep_research` boundary adapters, raw ingress persistence, and versioned snapshots. Reject incompatible pre-release v2 fixtures/checkpoints; there is no production v2 data migration.

### Phase 2 — supervisor_v2 and fast paths

Compose Context, Binding, Router, People, Document, Section, Write, KG, Evidence, and Grounding subgraphs using shared capabilities. Establish fast-path latency baselines.

### Phase 3+ — complex pilots and rollout

Pilot multi-document comparison, then People→Document, evidence-driven replan, offline replay, shadow, canary, and gradual cutover. Remove v1 only after benchmark gates pass.

Integration setting remains external to both graph builders:

```text
NEXUSRAG_AGENT_GRAPH_VERSION=v1|v2
```

## 28. Final minimal-review disposition and review gate

Every numbered recommendation in the final minimal-contract review was dispositioned rather than applied mechanically:

| # | Disposition | Current-consumer decision |
|---:|---|---|
| 1 | Applied | RequestContext keeps persisted request identity only; trusted user and run lineage are runtime-owned. |
| 2 | Applied | Conversation persistence/CAS fields moved to snapshot; `open_questions` removed because ClarificationRequest is the current owner. |
| 3 | Applied with retained semantic instruction | Resolved binding contains pinned revision only; optional typed RevisionRequirement remains on authoritative DocumentReference, with pin-once `None` semantics and freshness relation only for explicit current/latest. |
| 4 | Applied | Bare EvidenceRef removed; graph follows EvidenceUseRef → EvidenceUse → EvidenceRecord. |
| 5 | Applied | EvidenceRecordEnvelope removed; EvidenceStoreRow is the persisted boundary. |
| 6 | Applied with constraint | Workspace removed from source identity; authoritative document-revision lookup must return workspace ownership. |
| 7 | Applied | SynthesisInput no longer contains bindings; hydrator receives plan/bindings/runtime separately. |
| 8 | Kept as ephemeral projections | Synthesizer currently needs target grouping, explicit role semantics, and readable source label. |
| 9 | Applied | RenderedCitation claim ID removed because current frontend/API has no claim interaction. |
| 10 | Applied/trimmed | Descriptor version and behavior-derivable flags removed; parallel support remains a planner input. |
| 11 | Applied | Discovery policy keeps only expansion choices/limit; capabilities and workspace permission derive from registry/runtime. |
| 12 | Kept ephemeral | Planner chooses task, replan, and fan-out width, so it consumes all three non-persisted budget values. |
| 13 | Applied | Names distinguish TaskPlan.goal from TaskSpec.task_objective. |
| 14 | Applied | Scheduler selects capability before creating AgentRequest; duplicate capability removed. |
| 15 | Kept | AgentResult.task_id is consumed by async fan-in, retries, and checkpoint association. |
| 16 | Applied | TaskSpec completion criteria removed; logical completion belongs to TargetUnit. |
| 17 | Applied | ExactLookup and EntityResolution criteria removed in favor of typed capability semantics. |
| 18 | Applied | MinimumEvidenceCriterion removed; no current count-based quality use case. |
| 19 | Kept with restriction | SemanticCriterion is trusted-policy-only; ID is consumed by missing/evaluation references. |
| 20 | Applied | Discovery candidate source task removed; containing AgentResult owns it. |
| 21 | Applied | Binding provenance is a minimal discriminated audit union without evidence-ID duplication. |
| 22 | Kept | Clarification expiry is consumed on resume and freezes a per-request deadline across config changes. |
| 23 | Applied | Resolution user text removed; ChatMessage is authoritative. |
| 24 | Kept | EvidenceRecord/EvidenceUse separation remains unchanged. |
| 25 | Kept | Acquisition ID groups multiple evidence records from one fetch/tool call. |
| 26 | Kept compact | Optional target plus explicit validators is retained; three use variants would add types without new semantics. |
| 27 | Kept | Envelope run ID is the Evidence Use Store key and is not copied into EvidenceUse. |
| 28 | Kept | Storage expiry is a stable deletion deadline owned only by Evidence Store. |
| 29 | Kept | Global evidence/use UUID identities remain. |
| 30 | Applied behaviorally | Recursive faithfulness and no-derived-coverage invariants added without fields. |
| 31 | Applied behaviorally | Retrieved-content/prompt-injection control-plane boundary and test added without fields. |
| 32 | Applied behaviorally | Grounding requires every material factual assertion to map to a claim. |
| 33 | Kept | Claim text is the current standalone grounding unit; offsets would add more machinery. |
| 34 | Kept | FinalResponse remains status/content/citations only. |
| 35 | Applied | GroundingResult is explicitly internal and absent from root state/versioned boundaries. |
| 36 | Applied | Spec requires one immutable revision identity/stable locator coordinates without prescribing storage sub-revisions. |
| 37 | Applied | Version list now contains actual persisted/transported boundaries only. |
| 38 | Applied | BindingEventEnvelope replaced by BindingAuditRow; no event-sourcing implication. |
| 39 | Applied | Ownership matrix now includes producer, current consumer, persistence, derivability, and explicit decision. |
| 40 | Applied and independently validated | Final acceptance confirmed topology preservation, single ownership, criterion identity, synthesis sufficiency, and checkpoint semantics after the corrective review rounds. |
| 41 | Applied | Recommended minimal core remains the architecture; other types are runtime, persistence, audit, presentation, or internal. |

Approval checks:

- document identity is not duplicated across binding, target, or coverage;
- evidence identity and usage are separate and not duplicated in graph refs;
- no generic metadata dictionary exists;
- runtime ACL/deadline/trusted identity do not enter semantic request state;
- persistence/CAS metadata is outside ConversationContext;
- claim/evidence relationship has one authoritative representation;
- no dead EvidenceRef or hypothetical EvidenceRecord envelope remains;
- factual-claim grounding, derived faithfulness, and prompt-injection boundaries are explicit;
- each retained field category has a current producer, owner, and consumer in §24.

### Final pre-implementation corrections

The final pre-implementation review is closed: factual fast routes now use checkpointed deterministic one-task TaskPlans; ordinary revision references are pin-once; claim and replan reasoning preserve EvidenceUse context; DocumentBindingSet carries no unresolved projection; discovery candidates use UUID identity; Phase 1 is gated on immutable DocumentRevision persistence; and unused runtime config revision was removed. Criteria restrictions, untrusted-content separation, derived-evidence faithfulness, and presentation-only citations remain unchanged.

This revision is **Approved design** and is now under **architecture contract freeze**. Implementation planning may begin. Further contract changes require a concrete implementation blocker, benchmark failure, security issue, or missing supported use case; generic field trimming or future-proof additions are not accepted.

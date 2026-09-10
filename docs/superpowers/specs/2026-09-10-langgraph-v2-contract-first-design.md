# LangGraph v2 Contract-First Architecture

**Date:** 2026-09-10

**Status:** Review / Proposed

**Revision basis:** commit `26ed74f391cc6aaf89f0b510249abcc142346e74`

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

Versioned envelopes:

- `SupervisorV2State` checkpoint envelope;
- `RequestContext`;
- `ConversationSnapshot`;
- `SemanticSnapshot`;
- `DocumentBindingSet` and `BindingEventEnvelope`;
- `TaskPlan`;
- `AgentRequest` and `AgentResult`;
- `EvidenceRecordEnvelope`, `EvidenceUseEnvelope`, and `EvidenceStoreRow`;
- `ClarificationRequest`/`ClarificationResolution`;
- `FinalResponse`.

Leaf types such as `ScopedDocument`, `TargetUnit`, `DocumentReference`, `ContentLocator`, `CoverageItem`, source-identity variants, and runtime-only context do not repeat a version.

The snippets use `from __future__ import annotations` and a shared strict frozen base equivalent to:

```python
from pydantic import BaseModel, ConfigDict

class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
```

Actual modules must order definitions before use where practical. Where mutually referring aliases or models prevent that, the owning module resolves forward annotations and calls `model_rebuild()` after all referenced types are registered. Snippets omit repeated base-class/import boilerplate; omission does not relax strictness or immutability.

Runtime aggregate state evolves through LangGraph updates; immutable business objects are replaced, not mutated.

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
| Document semantic role and pinned revision | Binding Resolver |
| Logical read requirement | `TargetUnit` in `TaskPlan` |
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
├── original/contextualized/normalized query
├── canonical binding references
└── persist SemanticSnapshot
        ↓
Deterministic-first QueryAnalysis
        ↓
Deterministic RouteDecision
├── direct
├── clarify
├── fast domain → shared capability
└── complex research
        ↓
TaskPlan → AgentRequest → shared capability → AgentResult
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

`SupervisorV2State` is checkpointable. `GraphRuntimeContext` is injected, request-scoped, and never checkpointed. On resume, current runtime authorization always replaces historical authorization.

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
    run_id: str
    parent_run_id: str | None
    thread_id: str
    user_id: UUID
    original_query: str
    known_documents: tuple[KnownDocumentResource, ...]
```

Known resources provide identity, not semantic role or revision. The Binding Resolver resolves the current authorized revision and role. An attachment is only a contextual candidate until semantics binds it.

V2 persists raw `original_query` before expansion or normalization.

### 8.2 Conversation

```python
class ConversationContext(ContractModel):
    thread_id: str
    summary: str
    summary_version: int
    active_entities: tuple[ActiveEntity, ...]
    last_focus: EntityReference | None
    open_questions: tuple[str, ...]
    recent_turns: tuple[ConversationTurn, ...]
    built_through_message_id: str | None

class ConversationSnapshot(ContractModel):
    contract_version: Literal["2.0"]
    context: ConversationContext
```

Conversation Context is short-term discourse state (“nghị định này”, “file thứ hai”). Memory is long-term user context (“đơn vị tôi”) and is an optional capability, not a default terminal domain.

Rolling-summary writes use optimistic locking on `summary_version` and monotonic `built_through_message_id`.

### 8.3 Semantic lifecycle

```python
class DocumentReference(ContractModel):
    ref_id: str
    original_span: str
    normalized_reference: str
    requested_role: DocumentRole | None
    resolution_status: Literal["unresolved", "resolved", "ambiguous", "not_found", "error"]
    resolved_document_id: UUID | None
    candidate_document_ids: tuple[UUID, ...] = ()

class SemanticDraft(ContractModel):
    original_query: str
    provisional_contextualized_query: str
    abbreviations: tuple[AbbreviationResolution, ...]
    coreferences: tuple[CoreferenceResolution, ...]
    document_refs: tuple[DocumentReference, ...]
    person_refs: tuple[EntityReference, ...]
    section_refs: tuple[SectionReference, ...]
    preliminary_ambiguities: tuple[BlockingAmbiguity, ...]

class SemanticContext(ContractModel):
    original_query: str
    contextualized_query: str
    normalized_query: str
    abbreviations: tuple[AbbreviationResolution, ...]
    coreferences: tuple[CoreferenceResolution, ...]
    document_refs: tuple[DocumentReference, ...]
    binding_ids: tuple[str, ...]
    person_refs: tuple[EntityReference, ...]
    section_refs: tuple[SectionReference, ...]
    blocking_ambiguities: tuple[BlockingAmbiguity, ...]

class SemanticSnapshot(ContractModel):
    contract_version: Literal["2.0"]
    semantic: SemanticContext
```

`SemanticDraft` and small-model output are internal and not persisted. The Binding Resolver runs before semantic finalization. Only the finalized `SemanticSnapshot` is persisted.

`original_query` is immutable. `contextualized_query` resolves discourse references. `normalized_query` adds validated abbreviation/entity normalization.

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
RevisionPolicy = Literal["pinned", "latest_required"]

class ScopedDocument(ContractModel):
    binding_id: str
    document_id: UUID
    document_revision: str
    revision_policy: RevisionPolicy
    role: DocumentRole

class BindingProvenance(ContractModel):
    binding_id: str
    source_ref_id: str | None
    derived_from_binding_id: str | None
    reason: str
    source_task_ids: tuple[str, ...] = ()
    source_evidence_ids: tuple[UUID, ...] = ()

class BindingEventEnvelope(ContractModel):
    contract_version: Literal["2.0"]
    provenance: BindingProvenance

class DocumentBindingSet(ContractModel):
    contract_version: Literal["2.0"]
    bindings: tuple[ScopedDocument, ...]
    unresolved: tuple[DocumentReference, ...]
```

`ScopedDocument` owns document ID, pinned revision, revision policy, and role only. Target/reference roles are required; supporting/discovered roles are optional. There is no redundant `required` boolean.

Locator and completion criteria belong to target units:

```python
class TargetUnit(ContractModel):
    target_id: str
    binding_id: str
    requested_locator: ContentLocator
    completion_criteria: tuple[CompletionCriterion, ...]
```

`binding_id` resolves document identity/revision/role from `DocumentBindingSet`. A binding may have multiple target units.

Binding lineage is an audit/event concern represented by `BindingProvenance`, not copied into hot-path bindings. Only the Binding Resolver creates bindings. Planner outputs addition/promotion proposals; it cannot autonomously create targets.

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
    candidate_id: str
    document_id: UUID
    document_revision: str
    source_task_id: str

class BindingAdditionRequest(ContractModel):
    candidate_id: str
    requested_role: Literal["discovered", "supporting"]

class BindingPromotionRequest(ContractModel):
    source_binding_id: str
    requested_role: Literal["reference", "supporting"]
```

Reasons and trigger lineage are written once by the Binding Resolver to `BindingProvenance`; they are not copied through hot-path proposal/binding objects.

Revision is pinned when a binding is created. Downstream reads never silently switch to latest. `latest_required` causes explicit rebinding when a newer authorized revision is required.

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
    config_revision: str
```

```text
AgentRequest = requested operation.
CapabilityRuntimeContext = current trusted permission/scope.
Capability = executes request ∩ runtime authorization.
```

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

Every `fast_domain` route deterministically passes through the same minimal TaskPlan builder before capability dispatch; fast means a bounded plan, not no plan. The builder:

1. allocates `plan_id` and one `task_id` from the current run plus a deterministic operation ordinal;
2. emits one `TaskSpec(origin=InitialTaskOrigin())` with no dependencies;
3. emits the smallest required `TargetUnit` set only for logical document/read requirements, allocating each `target_id` from the plan plus target ordinal and putting those IDs in typed capability input;
4. emits no TargetUnit for targetless operations such as People lookup, Write transform, or KG lookup; and
5. validates the plan exactly like a complex plan before constructing `AgentRequest`.

Thus TaskPlan remains authoritative for every executed `task_id` and any `target_id`, including factual fast paths. A `direct` conversational response such as a greeting executes no capability and may keep `ExecutionState.plan=None`; a direct factual response is not permitted. Synthesis-only reuse constructs a minimal TaskPlan with one deterministic `evidence.reuse` TaskSpec. The reuse validator revalidates each source and creates current-run EvidenceUses whose `task_id` points to that task; no acquisition capability runs. The plan owns its objective and target units, and SynthesisInput references only those current-run validated EvidenceUses.

## 13. Task planning and capability execution

### 13.1 Typed completion criteria

```python
class CoverageCriterion(ContractModel):
    kind: Literal["coverage"]
    target_id: str
    minimum_status: Literal["read_partial", "read_complete"] = "read_complete"
    allow_partial_reason: str | None = None

class ExactLookupCriterion(ContractModel):
    kind: Literal["exact_lookup"]
    field_name: str
    require_non_null: bool = True

class MinimumEvidenceCriterion(ContractModel):
    kind: Literal["minimum_evidence"]
    target_id: str | None
    minimum_count: int

class EntityResolutionCriterion(ContractModel):
    kind: Literal["entity_resolution"]
    ref_id: str
    require_unique: bool = True

class SemanticCriterion(ContractModel):
    kind: Literal["semantic"]
    criterion_id: str
    description: str

CompletionCriterion = Annotated[
    CoverageCriterion | ExactLookupCriterion | MinimumEvidenceCriterion |
    EntityResolutionCriterion | SemanticCriterion,
    Field(discriminator="kind"),
]
```

Hard criteria are deterministic. Only `SemanticCriterion` goes to the semantic evaluator. Partial coverage requires explicit user/objective semantics and a deterministic reason. Minimum evidence count never substitutes target/reference coverage.

### 13.2 Minimal task contracts

```python
class InitialTaskOrigin(ContractModel):
    kind: Literal["initial"]

class ReplanTaskOrigin(ContractModel):
    kind: Literal["replan"]
    reason: str
    task_ids: tuple[str, ...]
    evidence_ids: tuple[UUID, ...]

TaskOrigin = Annotated[InitialTaskOrigin | ReplanTaskOrigin, Field(discriminator="kind")]

class TaskSpec(ContractModel):
    task_id: str
    capability: str
    objective: str
    input: CapabilityInput
    depends_on: tuple[str, ...] = ()
    completion_criteria: tuple[CompletionCriterion, ...] = ()
    origin: TaskOrigin

class TaskPlan(ContractModel):
    contract_version: Literal["2.0"]
    plan_id: str
    objective: str
    target_units: tuple[TargetUnit, ...]
    tasks: tuple[TaskSpec, ...]
```

Task inputs reference `target_id`/`binding_id` when needed; they do not copy `ScopedDocument`. Runtime resolves IDs from authoritative plan/binding set. Child semantic scope is a subset of parent bindings except policy-authorized reference discovery.

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
    capability: str
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

`TaskSpec` owns scheduler dependencies and completion criteria. `AgentRequest` carries only execution data. Runtime owns request/run IDs. Evaluator—not capability—derives missing requirements.

`AgentResult.status` describes task execution, not global sufficiency. `success != sufficient`.

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
CoverageOutcome = Literal["resolved", "read", "missing", "unreadable", "truncated"]
CoverageStatus = Literal["resolved", "read_complete", "read_partial", "missing", "unreadable", "truncated"]

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
    workspace_id: UUID
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

class EvidenceRecordEnvelope(ContractModel):
    contract_version: Literal["2.0"]
    record: EvidenceRecord
```

EvidenceRecord says what the evidence is. It contains no run/task/target usage, storage policy, UI labels, or free-form metadata.

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

class EvidenceRef(ContractModel):
    evidence_id: UUID

class EvidenceUseRef(ContractModel):
    use_id: UUID
```

EvidenceUse says how the current run/task uses immutable evidence. Resolution is total and deterministic:

- every use belongs to the current `EvidenceUseEnvelope.run_id`, resolves `task_id` to that run's TaskPlan, and resolves `evidence_id` to one current-authorized EvidenceRecord;
- `purpose="coverage"` requires a non-null `target_id` resolving to that TaskPlan; its document role/revision/locator derive through `target_id → TargetUnit → binding_id → ScopedDocument`;
- `purpose="discovery"` requires `target_id=None` and is ineligible for synthesis or claim support;
- a `supporting` use with a target follows the same target/binding path; a targetless supporting use is synthesis-eligible only for a validated targetless TaskSpec, with relevance derived from `task_id → TaskSpec.objective + typed input` and source kind derived from its EvidenceRecord. Its document role is `None` because no canonical binding path exists.

Any unresolved, cross-run, target-incompatible, or purpose-incompatible use is rejected rather than partially hydrated. Cross-run reuse revalidates ACL, retention, revision policy, locator, source availability, and semantic compatibility, then creates a **new EvidenceUse**. It does not mutate/copy EvidenceRecord and requires no adoption union in graph contracts.

`binding_id` and `target_id` remain run-local. Semantic identity is resolved from document ID + revision + locator + role.

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

Storage policy belongs to Evidence Store, not semantic evidence. Classification is deterministic and cannot be downgraded by a model.

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

## 16. Discovery and research policy

```python
class DiscoveryPolicy(ContractModel):
    allow_reference_discovery: bool
    allow_supporting_discovery: bool
    max_discovered_documents: int
    workspace_search_allowed: bool
    discovery_capabilities: tuple[str, ...]

class ResearchBudgetView(ContractModel):
    max_tasks_remaining: int
    max_replans_remaining: int
    max_parallel_branches: int

class CapabilityDescriptor(ContractModel):
    name: str
    version: str
    domain: Domain
    operation_type: Literal["lookup", "search", "read", "transform", "resolve"]
    can_discover_documents: bool
    produces_coverage: bool
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

Authorization answers where access is permitted. Bindings answer which documents have semantic roles. DiscoveryPolicy answers how research may expand. Effective discovery tools are the request-scoped registry intersected with `discovery_capabilities`; policy restricts but never grants permission.

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

Cancellation propagates from the LangGraph run to pending tasks and capability tokens/deadlines. Cancellation prevents replan/synthesis; incomplete work never becomes success.

## 18. Minimal evaluation contract

```python
class Contradiction(ContractModel):
    contradiction_id: str
    claim_a: str
    claim_b: str
    evidence_ids: tuple[UUID, ...]

class MissingRequirement(ContractModel):
    requirement_id: str
    target_id: str | None
    description: str

class EvidenceEvaluation(ContractModel):
    status: Literal["sufficient", "insufficient", "contradictory", "needs_input"]
    coverage: Coverage
    missing: tuple[MissingRequirement, ...]
    contradictions: tuple[Contradiction, ...]
```

Evaluator owns sufficiency, missing requirements, and conflicts. It does not suggest next capabilities; replanner owns next action.

A supported contradiction may be `sufficient` for a comparison objective. `contradictory` is reserved for unresolved conflict preventing a safe conclusion.

## 19. Minimal synthesis and grounding boundary

### 19.1 Runtime hydration

```python
class SynthesisInput(ContractModel):
    semantic: SemanticContext
    bindings: DocumentBindingSet
    evaluation: EvidenceEvaluation | None
    evidence_uses: tuple[EvidenceUseRef, ...]

class SynthesisRuntimeContext(ContractModel):
    max_evidence_items: int
    max_total_chars: int
    max_total_tokens: int

class SynthesisEvidence(ContractModel):
    evidence_id: UUID
    content: str
    role: DocumentRole | None
    target_id: str | None
    source_label: str | None
```

Evidence Hydrator—not the model—resolves current-run uses and records, applies the purpose/target rules in §15.2, revalidates current ACL/retention/revision/source availability, applies runtime budget, and produces ephemeral synthesis evidence. The model does not see workspace IDs, document revisions, locators, or internal provenance. `SynthesisEvidence.role` and `target_id` are computed projections from the validated EvidenceUse path; they are never persisted back into EvidenceRecord.

The hydrator also returns internally the set of `(use_id, evidence_id)` pairs admitted for this synthesis call. Every `AnswerClaim.evidence_ids` entry must be a member of the evidence-ID projection of that current-run validated set; discovery-only, omitted, stale, denied, target-incompatible, and merely store-visible evidence IDs are rejected. An evidence ID being globally valid is insufficient without an admitted current-run EvidenceUse.

Overflow is compacted/map-reduced before final synthesis. Derived summaries are persisted as EvidenceRecords with source lineage and receive a new validated supporting EvidenceUse for the current run; the final synthesis receives only their admitted IDs/content. Grounding recursively validates the derived source lineage.

### 19.2 One claim-to-evidence mapping

```python
class AnswerClaim(ContractModel):
    claim_id: str
    text: str
    evidence_ids: tuple[UUID, ...]

class AnswerDraft(ContractModel):
    content: str
    claims: tuple[AnswerClaim, ...]
```

There is no duplicate `CitationRef` relationship. The model proposes claims and evidence IDs. Grounding validates claim support against authoritative records, pinned revisions, and locators.

```python
class RenderedCitation(ContractModel):
    citation_id: str
    claim_id: str
    evidence_id: UUID
    label: str

class FinalResponse(ContractModel):
    contract_version: Literal["2.0"]
    status: Literal["success", "clarify", "denied", "insufficient", "error"]
    content: str
    citations: tuple[RenderedCitation, ...] = ()
```

CitationRenderer creates deterministic citation IDs only after Grounding. FinalResponse does not persist a derivable evidence-ID projection.

Grounding flow:

```text
AnswerDraft.claim
→ EvidenceUse/EvidenceRecord
→ source identity + pinned revision + locator
→ semantic support check
→ validated claims
→ CitationRenderer
→ FinalResponse
```

A revision request creates a wholly new AnswerDraft and claim mapping, then grounds once more. Second failure returns `insufficient`.

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
    unresolved_references: tuple[DocumentReference, ...]
    candidates: tuple[DocumentCandidate, ...]
    expires_at: datetime

class ClarificationResolution(ContractModel):
    contract_version: Literal["2.0"]
    clarification_id: str
    selected_candidate_id: str | None
    user_text: str
```

Candidate identity/order is stable within a clarification. Resume validates clarification, candidate membership, and current ACL, then asks Binding Resolver to create a binding. It does not rerun unconstrained ambiguity resolution.

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

1. Raw/original query is immutable.
2. Current authorization is runtime-injected and never checkpointed.
3. LLM cannot create workspace authorization or permission.
4. Context resolution never grants access.
5. Binding Resolver alone owns document role and pinned revision.
6. Target/reference roles are required; supporting/discovered are optional.
7. Attachments are candidates, not automatic targets.
8. Required targets are never silently replaced by discovered documents.
9. Planner cannot autonomously add targets.
10. Child tasks cannot exceed semantic bindings except policy-authorized reference discovery.
11. `TargetUnit` owns locator/criteria; binding owns document identity/revision/role.
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
25. AnswerDraft stores one claim-to-evidence relationship.
26. Citation IDs are deterministic presentation output, not model authority.
27. Grounding runs before factual success.
28. Only outer answer layer streams final response.

## 24. Field ownership summary

| Field/fact | Owner | Consumer | Persisted | Derivable | Keep reason |
|---|---|---|---|---|---|
| `binding_id → document identity/revision/role` | Binding Resolver | planner/runtime/evaluator | yes | no | canonical semantic document identity |
| `target_id → binding/locator/criteria` | TaskPlan | capability/evaluator | checkpoint | no | canonical logical requirement |
| `task_id → operation/dependencies` | TaskPlan | scheduler | checkpoint | no | orchestration identity |
| `evidence_id → source/content/provenance` | Evidence Store | hydrator/grounding | yes | no | immutable evidence identity |
| `use_id → run/task/target/purpose` | Evidence Use Store | evaluator/hydrator | yes | no | contextual evidence use |
| `AgentResult.data` | capability boundary | scheduler/domain consumer | checkpoint only when minimized and non-sensitive | no for approved operational facts | typed execution output; sensitive content is evidence-only |
| EvidenceRecord provenance | Evidence Builder/Store | grounding/audit | yes | no | proves acquisition of immutable content without copying usage/storage metadata |
| runtime authorization/deadline/config | backend runtime | capability/evidence services | no | refreshed per request/resume | trusted execution context; deliberately unversioned business state |
| `SynthesisEvidence.role/target_id/source_label` | Hydrator/Presentation | synthesizer | no | yes | ephemeral projection from admitted EvidenceUse and canonical stores |
| hydrated admitted evidence-ID set | Evidence Hydrator | Grounding | no | yes per synthesis call | enforce claims reference current-run validated uses |
| domain `CapabilityInput`/checkpoint-safe output fields | named capability module contract spec | capability/domain consumer | per field classification | assessed in that module | domain variants own their fields; core spec forbids unclassified/raw fields |
| document details for a target | Binding Resolver | evaluator | no duplicate | yes | resolve through IDs |
| evidence-use aggregate | AgentResults | hydrator | no | yes | collect from task results |
| rendered source label | Presentation | user/UI | response only | yes | derive after validation |
| routing semantic complexity | telemetry | observability | trace only | n/a | not business state |

## 25. Superseded contract disposition

This repository has no production v2 graph, v2 checkpoint, Evidence Store v2 row, or other persisted v2 artifact: all `"2.0"` shapes at commit `26ed74f` were architecture-spec examples only and were never runtime or persistence contracts. “Superseded” below therefore means superseded design prose, not an in-place production data migration.

The following proposed shapes from commit `26ed74f` are superseded by this minimal baseline:

| Superseded shape | Disposition |
|---|---|
| version on every nested model | version only persisted/external envelopes |
| `ScopedDocument.required/locator/lineage` | required derived from role; locator in TargetUnit; lineage in BindingProvenance |
| TargetUnit copies document/revision/role | resolve through `binding_id` |
| Coverage copies binding/document/revision/role/request | resolve through `target_id` |
| `ExecutionState.evidence_refs` | derive from task results |
| QueryAnalysis capability/complexity/synthesis fields | policy/tracing/Answer Policy, not semantic contract |
| persisted `SemanticAnalysisHints` | ephemeral internal `SemanticModelOutput` |
| `RouteDecision.domain` | read domains from QueryAnalysis |
| TaskSpec copied ScopedDocuments + three replan fields | ID-bearing typed input + `TaskOrigin` |
| AgentRequest scheduler/evaluator/runtime fields | minimal task/objective/input/capability |
| AgentResult request ID + missing requirements | runtime/evaluator ownership |
| EvidenceRecord mixed identity/use/storage/presentation | split EvidenceRecord, EvidenceUse, StoragePolicy, presentation |
| metadata dict | typed fields or telemetry; removed |
| verbose EvidenceRef/AdoptedEvidenceRef | minimal EvidenceRef/EvidenceUseRef; reuse creates new EvidenceUse |
| SynthesisBudget in business input | SynthesisRuntimeContext |
| duplicate AnswerClaim + CitationRef mapping | AnswerClaim only; RenderedCitation after grounding |
| FinalResponse evidence-ID projection | derive from citations |
| evaluator research suggestions | replanner ownership |
| candidate ranking/debug fields | resolver telemetry; clarification keeps identity/order/label only |

Existing Phase-1 `PreprocessingResult`, abbreviation/reference validators, and semantic-context DB column remain reuse candidates through explicit v1-to-v2 adapters. `backend/app/services/agents/deep_research/` contracts are legacy inputs/outputs only: an adapter may translate a validated subset at the v2 boundary, but those models do not become v2 canonical contracts or enter v2 checkpoint state. Phase-1 `RoutingDecision`/SupervisorState additions remain v1 compatibility and are not completed as v2 architecture.

Because no production v2 persisted artifact exists, v2 startup/load code must reject incompatible pre-release fixtures or checkpoints rather than attempt best-effort migration. Test fixtures claiming version `2.0` must conform to this revision's envelope schemas; older spec-derived fixtures are regenerated. V1 production chat data remains authoritative and is read through the named adapters.

## 26. Validation and acceptance scenarios

Contract tests must prove:

- envelope-only versioning and immutable boundary models, including versioned SupervisorV2State/BindingEvent/EvidenceStoreRow and unversioned runtime context;
- factual fast routes deterministically build a valid minimal TaskPlan while conversational direct routes may omit it;
- ID referential integrity across bindings, targets, tasks, evidence, and uses;
- raw/sensitive capability output cannot enter checkpointed AgentResult.data and evidence-only fields produce refs instead;
- no duplicated canonical identity in TargetUnit/Coverage;
- typed capability input/output without generic dict fallback;
- hard versus semantic criteria ownership;
- search cannot produce read-complete coverage;
- revision mismatch rejects coverage/evidence reuse;
- cross-run reuse creates a new EvidenceUse after current ACL/revision/retention checks;
- coverage uses require valid targets, discovery uses cannot synthesize, and targetless supporting uses require a validated targetless task;
- raw People output is minimized before EvidenceRecord persistence;
- every claim evidence ID belongs to the hydrated current-run admitted-use set;
- claim/evidence grounding and deterministic citation rendering;
- direct greetings require no evidence while factual fast paths do;
- runtime context never serializes into checkpoint state;
- incompatible pre-release v2 fixtures/checkpoints are rejected rather than migrated best-effort.

Required scenario set includes simple People/Section/Write/KG, conversational and ambiguous follow-up, multi-document range comparison, People→Document, target/reference compliance, reference discovery, permission denial, resume ACL change, abbreviation normalization, wrong/partial section coverage, irrelevant attachments, unresolved named documents, discovered-reference promotion, execution-success/evidence-insufficient, document revision change, evidence expiry/reuse denial, search-without-read, fast-path evidence, synthesis-budget overflow, and synthesis-only comparison with already-valid evidence.

## 27. Migration and rollout

### Phase 0 — Orchestrator compatibility spike

Benchmark native LangGraph versus Deep Agents adapter against the same minimal contracts. This selects implementation, not contract meaning.

### Phase 1 — Minimal contracts and adapters

Implement envelopes, binding/target identity graph, minimal evidence/use stores, Phase-1 and legacy `deep_research` boundary adapters, raw ingress persistence, and versioned snapshots. Reject incompatible pre-release v2 fixtures/checkpoints; there is no production v2 data migration.

### Phase 2 — supervisor_v2 and fast paths

Compose Context, Binding, Router, People, Document, Section, Write, KG, Evidence, and Grounding subgraphs using shared capabilities. Establish fast-path latency baselines.

### Phase 3+ — complex pilots and rollout

Pilot multi-document comparison, then People→Document, evidence-driven replan, offline replay, shadow, canary, and gradual cutover. Remove v1 only after benchmark gates pass.

Integration setting remains external to both graph builders:

```text
NEXUSRAG_AGENT_GRAPH_VERSION=v1|v2
```

## 28. Review gate

This revision is **Review / Proposed** pending review of:

1. field ownership and derivability;
2. minimal contract boundaries;
3. migration impact on Phase-1 and legacy deep-research contracts;
4. security properties after evidence identity/use/storage separation.

Do not write the implementation plan until this review promotes the spec to **Approved design**.

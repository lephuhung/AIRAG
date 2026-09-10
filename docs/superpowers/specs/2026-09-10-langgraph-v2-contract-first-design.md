# LangGraph v2 Contract-First Architecture

**Date:** 2026-09-10

**Status:** Approved design

**Revision basis:** commit `15069012a8ad38a3f814421ade3413da9992f098`

**Source direction:** `docs/agent-contract-langgraph-deepagent.md`

**Scope:** architecture and contracts only; no runtime implementation is authorized by this revision

## 1. Objective

Build a contract-first LangGraph v2 architecture with explicit boundaries between:

- LangGraph orchestration and lifecycle;
- complex research orchestration;
- domain capabilities;
- conversation and semantic context;
- trusted authorization scope;
- business task requests and results;
- evidence, coverage, synthesis, and grounding.

The design must preserve simple-query latency, reuse valuable Phase-1 preprocessing work, and avoid a big-bang rewrite of existing business services.

## 2. Retained architecture decisions

1. Create `backend/app/services/agents/supervisor_v2.py`; do not add v2 behavior to `supervisor.py`.
2. Select v1 or v2 at an integration boundary using a feature/config flag.
3. Compose v2 from independently testable domain subgraphs with typed input/output contracts.
4. Treat the chat database as the conversation source of truth. LangGraph checkpoints support execution, interrupt, and resume only.
5. Keep permission, authorization, scope enforcement, budgets, plan validation, and hard evidence checks deterministic.
6. Never expose MongoDB, vector DB, Neo4j clients, or raw database interfaces directly to a model.
7. Preserve deterministic fast paths for simple requests.
8. Use planning, evidence collection, evidence evaluation, bounded replanning, synthesis, and grounding for complex requests.
9. Preserve citation and provenance through the final answer.
10. Roll out through offline replay, shadow mode, canary, and gradual cutover.
11. Keep v1 until v2 passes explicit correctness, latency, grounding, permission, and cost benchmarks.

## 3. Terminology and framework boundary

### 3.1 LangGraph

LangGraph owns runtime concerns:

```text
LangGraph
= state lifecycle
+ node/subgraph composition
+ Command/Send routing
+ checkpointing
+ interrupt/resume
+ streaming lifecycle
```

LangGraph types do not become business contracts. A capability result never contains `goto`, `Command`, `Send`, or graph topology.

### 3.2 ComplexResearchGraph

This spec replaces the overloaded name “DeepAgent” with `ComplexResearchGraph`.

```text
ComplexResearchGraph
= AIRAG contract and orchestration semantics for complex research.
```

It must support:

- executable planning;
- multi-step and multi-document research;
- domain capability invocation;
- dependency handling and bounded parallelism;
- evidence and target-unit coverage evaluation;
- append-only bounded replanning;
- synthesis after evidence sufficiency;
- clarification when an essential target or ambiguity blocks completion.

This design intentionally does **not** select the implementation framework. Conforming implementations may include:

- a native LangGraph planner/executor;
- a Deep Agents library adapter;
- another implementation that satisfies the same contracts, invariants, lifecycle, and benchmark gates.

Selecting or rejecting the Deep Agents library is a later explicit architecture decision. It requires a compatibility and benchmark spike; absence of the library from initial code is not a decision to reject it.

### 3.3 ResearchOrchestrator

`ResearchOrchestrator` is the implementation-facing interface behind `ComplexResearchGraph`. The graph composition depends on this interface rather than on a particular planner library.

## 4. Updated target architecture

```text
Backend ingress
├── authenticate
├── calculate workspace authorization
├── calculate People permission
├── persist ORIGINAL user message
├── capture attachments and known UI bindings
└── load conversation context
        ↓
Semantic Draft
├── deterministic extraction
├── conversational coreference resolution
├── protected abbreviation resolution
├── document-reference extraction
└── optional small semantic model
        ↓
Binding Resolver
├── resolve document identity
├── validate current authorization
├── assign semantic roles
└── produce DocumentBindingSet
        ↓
Semantic Finalizer
├── contextualized query
├── normalized query
├── canonical binding references
└── persist final semantic snapshot
        ↓
QueryAnalysis
├── work_type
├── domains
├── capability_hints
├── dependency_hints
├── semantic_complexity
└── requires_synthesis
        ↓
Deterministic Router
├── direct
├── clarify
├── fast_domain
│   ├── People
│   ├── Document
│   ├── Section
│   ├── Write
│   └── Knowledge Graph
└── complex_research
        ↓
ComplexResearchGraph
├── ResearchPlanningInput
├── typed executable DAG
├── typed CompletionCriterion
├── request-scoped CapabilityRegistry
├── domain capabilities
├── Evidence Store + EvidenceRef
├── CoverageObservation
├── EvidenceEvaluation
├── bounded append-only replan
└── synthesis
        ↓
Grounding
        ↓
FinalResponse
```

## 5. Module layout and subgraph boundaries

```text
backend/app/services/agents/
├── supervisor.py                    # unchanged v1 implementation
├── supervisor_v2.py                 # v2 composition root
└── v2/
    ├── contracts/
    │   ├── context.py
    │   ├── execution.py
    │   ├── task.py
    │   └── evidence.py
    ├── context_graph/
    ├── routing_graph/
    ├── complex_research_graph/
    ├── people_graph/
    ├── document_graph/
    ├── section_graph/
    ├── write_graph/
    ├── knowledge_graph/
    ├── grounding_graph/
    ├── capabilities/
    ├── persistence/
    └── adapters/
```

`supervisor_v2.py` only composes subgraphs, declares graph edges, and exposes the public graph builder. It does not contain domain logic.

Each subgraph:

- has minimal typed input/output state;
- cannot read or mutate arbitrary root-state fields;
- communicates through shared contracts;
- can be compiled and tested independently;
- may use LangGraph primitives internally without leaking them across business boundaries.

The v2 taxonomy includes People, Document, Section, Write, and Knowledge Graph domains. Initial pilot implementation may stage them, but routing semantics are defined now: bounded single-domain Write and KG operations use fast paths; Write/KG combined with document evidence or runtime dependencies use complex research.

## 6. Data objects and ownership

```text
Raw User Request
    ↓
RequestContext
+
ConversationContext
    ↓
SemanticDraft
    ↓
DocumentBindingSet
    ↓
SemanticContext
    ↓
QueryAnalysis
    ↓
RouteDecision
    ↓
TaskPlan / DAG
    ↓
AgentRequest
+
CapabilityRuntimeContext
    ↓
EvidenceRecord → Evidence Store
+
AgentResult + EvidenceRef + CoverageObservation
    ↓
Coverage + EvidenceEvaluation
    ↓
Synthesis
    ↓
GroundingResult
    ↓
FinalResponse
```

| Object | Owner | Notes |
|---|---|---|
| `RequestContext` / `KnownDocumentResource` | Backend ingress | Raw request metadata plus known identity/source; no semantic role |
| `CapabilityRuntimeContext` | Backend/runtime adapter | Trusted authorization, capability allowlist, deadline |
| `ConversationContext` | Context layer | Short-term discourse state sourced from chat DB |
| `SemanticDraft` / `DocumentReference` | Context layer | Preliminary meaning, raw references, abbreviations, and coreferences |
| `DocumentBindingSet` | Binding resolver | Authoritative resolved bindings plus unresolved references |
| `SemanticContext` | Semantic finalizer | Canonical finalized meaning linked to resolved bindings; persisted only after resolution |
| `QueryAnalysis` | Query analyzer | Semantic structure and dependency hints, not an executable plan |
| `RouteDecision` | Deterministic router | `clarify`, fast domain path, or complex research |
| `DiscoveryPolicy` | Deterministic Research Policy Builder | Bounded semantic research expansion; distinct from authorization |
| `TaskPlan` / DAG | `ComplexResearchGraph` | Executable tasks and dependencies |
| `AgentRequest` | Planner/orchestrator | Business operation requested from a capability |
| `EvidenceRecord` | Evidence Store | Source of truth for full evidence content/provenance |
| `CapabilityOutput` / `AgentResult` / `EvidenceRef` | Capability adapter | Typed domain output plus compact checkpoint-safe task result and evidence references |
| `CoverageObservation` | Capability | Facts observed while executing a task; never authoritative completion |
| `Coverage` / `EvidenceEvaluation` | Evaluator | Authoritative requirement-level completion and semantic sufficiency |
| `SynthesisInput` / `AnswerDraft` / `GroundingResult` / `FinalResponse` | Answer layer | Typed synthesis and validated user-facing output |

## 7. Core contract rules

All boundary and persisted business contracts are versioned immutable Pydantic v2 models by default:

```python
model_config = ConfigDict(extra="forbid", frozen=True)
```

This applies to `RequestContext`, `ConversationContext` snapshots, `SemanticContext`, `DocumentReference`, `DocumentBindingSet`, `ScopedDocument`, `QueryAnalysis`, `RouteDecision`, `TaskSpec`, `TaskPlan`, `AgentRequest`, `AgentResult`, `EvidenceRecord`, `EvidenceRef`, `CoverageObservation`, `Coverage`, `EvidenceEvaluation`, and answer contracts. Any exception must be explicit and justified. Runtime/aggregate state may evolve through LangGraph state updates, but business objects are replaced rather than mutated in place. Runtime-only objects such as DB sessions, cancellation events, LangGraph commands, and raw clients are excluded from persisted business contracts.

### 7.1 Root graph state

```python
class SupervisorV2State(TypedDict):
    request: RequestContext
    conversation: ConversationContext
    semantic: SemanticContext
    document_bindings: DocumentBindingSet
    query_analysis: QueryAnalysis | None
    route_decision: RouteDecision | None
    execution: ExecutionState
    clarification: ClarificationRequest | None
    grounding: GroundingResult | None
    final_response: FinalResponse | None
```

Nested objects cross subgraph boundaries through explicit input/output adapters. A domain subgraph never receives the entire root state unless its input schema explicitly requires every field.

Checkpointable state and trusted runtime are separate:

```python
class GraphRuntimeContext(BaseModel):
    capability_runtime: CapabilityRuntimeContext
    services: RuntimeServices
```

`SupervisorV2State` contains only checkpointable workflow/business state. Nodes and subgraphs obtain `GraphRuntimeContext` through LangGraph runtime/context injection, never from checkpoint data. `GraphRuntimeContext` is current-request-only, never serialized, and recalculated on every request/resume. Context/document resolvers and clarification lookup receive a read-only view derived from its `CapabilityRuntimeContext`, so current ACL always wins.

### 7.2 Request context

```python
class RequestContext(BaseModel):
    contract_version: Literal["2.0"]
    request_id: str
    run_id: str
    parent_run_id: str | None = None
    thread_id: str
    user_id: UUID
    original_query: str
    known_documents: tuple[KnownDocumentResource, ...]
```

Authorization fields are deliberately absent. They belong to trusted runtime context, not the business request. `known_documents` carries identity and source only; backend ingress does not assign semantic roles.

```python
class KnownDocumentResource(BaseModel):
    contract_version: Literal["2.0"]
    resource_id: str
    document_id: UUID
    source: Literal["attachment", "ui_selection", "conversation", "api_explicit"]
```

Named documents mentioned only in text do not need a UUID at ingress. `document identity != semantic role`: the Binding resolver combines known resources, user semantics, and conversation context to create `ScopedDocument` bindings.

## 8. Document roles and required scope

### 8.1 Role-based binding

V2 adopts role-based document binding as the canonical contract:

```python
DocumentRole = Literal["target", "reference", "supporting", "discovered"]

class ScopedDocument(BaseModel):
    contract_version: Literal["2.0"]
    binding_id: str
    document_id: UUID
    role: DocumentRole
    required: bool
    source_ref_id: str | None = None
    locator: ContentLocator | None = None
    derived_from_binding_id: str | None = None
    binding_reason: str | None = None

class DocumentBindingSet(BaseModel):
    contract_version: Literal["2.0"]
    bindings: tuple[ScopedDocument, ...]
    unresolved: tuple[DocumentReference, ...]
```

Semantics:

- `target`: subject being read, analyzed, compared, or evaluated;
- `reference`: normative or comparative basis required for the objective;
- `supporting`: explicitly bound supplemental material;
- `discovered`: material found during research and not silently promoted to target/reference.

A required binding is a completion requirement, not an authorization grant. Authorization still comes only from `CapabilityRuntimeContext.workspace_ids`. `required` is explicit: user-bound targets and references are normally `True`; incidental supporting/discovered bindings are `False`.

Binding lifecycle is append-only. Promotion never mutates an existing binding. If discovered binding `B1` should become a required reference, the planner emits a typed `BindingPromotionRequest`. The Binding resolver validates authorization and lineage, then appends `B2` with `role="reference"`, `required=True`, `derived_from_binding_id="B1"`, and a non-empty `binding_reason`. The Binding resolver remains the sole authoritative writer of `DocumentBindingSet`; the planner only requests reference/supporting additions. Autonomous target promotion is forbidden because target denotes the user’s objective. A new target requires an explicit user-confirmed or deterministic semantic binding, potentially through clarification. Validation rejects `discovered + required=True` and rejects promoted bindings without valid lineage.

### 8.2 Canonical use cases

```text
"So sánh A và B"
A = required target
B = required target

"Kiểm tra F1/F2 theo A"
F1 = required target
F2 = required target
A  = required reference

"Phân tích A và tìm các văn bản liên quan"
A = required target
newly found documents = discovered/supporting

"Kiểm tra F1/F2 theo quy định hiện hành"
F1/F2 = required targets
references = discovered within authorized workspaces, then explicitly bound as references by the plan
```

A required target or reference cannot be silently replaced by a discovered document. The role and `binding_id` must survive task fan-out, evidence fan-in, evaluation, synthesis, and citation rendering.

### 8.3 Reference-to-binding lifecycle

```text
Raw Request
    ↓
RequestContext
├── original_query
├── attachments
└── initial known bindings
    ↓
SemanticDraft + DocumentReference[]
    ↓
Document Binding Resolution
    ↓
DocumentBindingSet
├── bindings: resolved UUID-backed ScopedDocument[]
└── unresolved: DocumentReference[]
    ↓
Semantic Finalization → persisted SemanticContext
    ↓
QueryAnalysis → Router
```

Only the Binding resolver creates UUID-backed bindings for text references. It merges known resources, conversation-derived references, attachments selected by semantics, and newly resolved references. Essential unresolved references route to clarification.

### 8.4 Attachment semantics

```text
attached document != required target/reference
```

Known resources with `source="attachment"` are available as contextual candidates. Attaching A and B while asking a generic corpus question does not require reading A/B. Semantic resolution promotes attachments into immutable `ScopedDocument` bindings only when the request refers to them, for example “kiểm tra hai file tôi vừa gửi.” The flow is `KnownDocumentResource(attachment) → contextual candidates → semantic resolution → DocumentBindingSet`, never automatic attachment-to-target conversion.

### 8.5 Discovery-to-binding lifecycle

Search never creates a binding directly:

```text
document.search
→ DocumentDiscoveryCandidate
→ planner emits BindingAdditionRequest
→ Binding Resolver validates authorization/policy and creates B1
   role=discovered, required=false
→ planner may emit BindingPromotionRequest for B1
→ Binding Resolver creates immutable B2
   role=reference/supporting, with lineage
```

Only the Binding Resolver creates `ScopedDocument`. A planner may request discovery/supporting addition or reference/supporting promotion, never autonomous target promotion.

## 9. Trusted runtime context and capability boundary

### 9.1 CapabilityRuntimeContext

```python
class CapabilityRuntimeContext(BaseModel):
    contract_version: Literal["2.0"]
    request_id: str
    run_id: str
    user_id: UUID
    workspace_ids: tuple[UUID, ...]
    can_read_people: bool
    allowed_capabilities: frozenset[str]
    deadline_at: datetime
    config_revision: str
```

This object is constructed by trusted backend/runtime code. Model output cannot create or alter it.

### 9.2 Standard capability signature

```python
async def execute(
    request: AgentRequest,
    runtime: CapabilityRuntimeContext,
) -> AgentResult:
    ...
```

```text
AgentRequest
= what the model/task asks to do.

CapabilityRuntimeContext
= what the backend permits, where, and until when.

Capability
= executes request ∩ runtime authorization.
```

The capability adapter injects `workspace_ids`; tool arguments generated by a model never contain trusted authorization scope. A capability validates its name against `allowed_capabilities` and applies People permission defense in depth.

### 9.3 People permission

- `can_read_people=false`: People capabilities are omitted from the request-scoped registry; direct execution returns `denied`.
- `can_read_people=true`: People lookup, including CCCD/BHXH, is allowed.

The backend recalculates authorization and People permission for each request and resume.

## 10. ConversationContext and SemanticContext

### 10.1 ConversationContext

```python
class ConversationContext(BaseModel):
    contract_version: Literal["2.0"]
    thread_id: str
    summary: str
    summary_version: int
    active_entities: tuple[ActiveEntity, ...]
    last_focus: EntityReference | None
    open_questions: tuple[str, ...]
    recent_turns: tuple[ConversationTurn, ...]
    built_through_message_id: str | None
```

It is structured short-term discourse state. It avoids sending an unbounded history to every agent.

### 10.2 Conversation context is not memory

```text
Conversation Context
= short-term discourse state.

Memory
= long-term user-related context.
```

Resolve from conversation context:

- “nghị định này”;
- “văn bản trên”;
- “người vừa nói”;
- “file thứ hai”;
- “điều vừa nói”.

Long-term memory may be consulted for statements such as “đơn vị tôi” or durable user information, subject to its own permission and relevance policy. It must not replace conversation resolution when the antecedent is present in the thread.

### 10.3 SemanticContext

```python
class DocumentReference(BaseModel):
    contract_version: Literal["2.0"]
    ref_id: str
    original_span: str
    normalized_reference: str
    requested_role: DocumentRole | None = None
    required: bool
    locator: ContentLocator | None = None
    resolution_status: Literal["unresolved", "resolved", "ambiguous", "not_found", "error"]
    match_type: Literal[
        "exact_id", "exact_number", "exact_title", "conversation_coreference",
        "fuzzy_title", "semantic_match",
    ] | None = None
    resolved_document_id: UUID | None = None
    candidate_document_ids: tuple[UUID, ...] = ()

class SemanticDraft(BaseModel):
    contract_version: Literal["2.0"]
    original_query: str
    provisional_contextualized_query: str
    abbreviations: tuple[AbbreviationResolution, ...]
    coreferences: tuple[CoreferenceResolution, ...]
    document_refs: tuple[DocumentReference, ...]
    person_refs: tuple[EntityReference, ...]
    section_refs: tuple[SectionReference, ...]
    preliminary_ambiguities: tuple[BlockingAmbiguity, ...]

class SemanticContext(BaseModel):
    contract_version: Literal["2.0"]
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
```

The three query forms are distinct:

```text
original_query
= immutable raw user input.

contextualized_query
= conversational references resolved.

normalized_query
= contextualized query with validated abbreviation/entity/document normalization.
```

Example:

```text
Previous turn:
"Nghị định 13/2023/NĐ-CP quy định gì?"

Current original_query:
"NĐ này có quy định về DLCN không?"

contextualized_query:
"NĐ 13/2023/NĐ-CP có quy định về DLCN không?"

normalized_query:
"Nghị định 13/2023/NĐ-CP có quy định về dữ liệu cá nhân không?"
```

`SemanticDraft` is never persisted as the final semantic snapshot. The Binding resolver first resolves identity and authorization; the Semantic finalizer then writes canonical resolved IDs/binding IDs into `SemanticContext` and persists it. Query analysis runs only after finalization.

Exact unique matches bind deterministically. Fuzzy/semantic matches must satisfy a versioned policy threshold; multiple plausible matches clarify. Confidence alone is insufficient without `match_type`. `DocumentReference` invariants are deterministic: `resolved` requires non-null `resolved_document_id` and, when candidates are retained, that ID must be a member of `candidate_document_ids` (the canonical resolved representation should retain a singleton matching candidate); `ambiguous` requires at least two plausible candidates and no canonical ID; `not_found` requires no canonical ID and an empty candidate set; `unresolved` means lookup has not completed and has no canonical ID; and `error` represents resolver/infrastructure failure, never semantic ambiguity.

### 10.4 Context resolution does not grant access

```text
ConversationContext
    ↓
resolve "nghị định này" → A
    ↓
current runtime workspace authorization
    ↓
authorized capability execution
```

An entity seen in a previous turn may no longer be accessible. Current authorization always wins.

## 11. Semantic preprocessing and abbreviation reuse

The semantic preprocessing stage always exists, but expensive semantic-model work is conditional.

### 11.1 Abbreviation pipeline

```text
raw query
 ↓
protect identifiers and literals
 ↓
cheap candidate detection
 ↓
no candidate ─────────────→ continue
 ↓
batch DB lookup
 ├── unique ──────────────→ normalize
 ├── ambiguous ───────────→ conditional disambiguation
 └── unknown ─────────────→ preserve original
```

Protected spans include:

- legal document numbers such as `12/2024/NĐ-CP`;
- CCCD, BHXH, phone numbers, and other IDs;
- quoted literals;
- validated document identifiers.

Abbreviations are never expanded blindly. The contract retains:

```python
class AbbreviationResolution(BaseModel):
    span: str
    short_form: str
    chosen: str | None
    candidates: tuple[AbbreviationCandidate, ...]
    status: Literal["resolved", "ambiguous", "unknown", "not_in_db"]
    source: str
    confidence: float | None
```

Complex research may call the shared `abbreviation.resolve` capability for abbreviations discovered during research. It must not duplicate preprocessing logic.

### 11.2 Conditional small semantic model

```text
deterministic extraction
       ↓
context/semantic complexity gate
       ├── fully resolved + simple
       │      ↓
       │   skip semantic LLM
       │
       └── ambiguous/compound/context-dependent
              ↓
          small semantic model
              ↓
          deterministic reconciliation
```

Examples that skip the semantic model when deterministically resolved:

- “Xin chào”;
- a validated direct CCCD lookup;
- exact document metadata lookup;
- exact document-and-section retrieval.

The semantic model may select or describe validated candidates but cannot create trusted document IDs, workspace authorization, or permissions.

## 12. Query analysis and deterministic routing

`QueryAnalysis` describes semantic structure; it is not an executable planner. Analysis is deterministic-first: rules classify confident greetings, People lookup, exact Section retrieval, and bounded Write operations. Only uncertain cases call the small model. If the Context layer already ran that model, Query Analysis reuses its structured semantic output rather than making a duplicate call. Simple requests require zero planner calls.

```python
class QueryAnalysis(BaseModel):
    contract_version: Literal["2.0"]
    work_type: Literal[
        "direct", "lookup", "retrieve", "explain", "summarize",
        "compare", "evaluate", "cross_domain", "multi_goal",
    ]
    domains: tuple[Literal[
        "people", "document", "section", "write",
        "knowledge_graph", "memory",
    ], ...]
    capability_hints: tuple[str, ...]
    dependency_hints: tuple[SemanticDependencyHint, ...]
    semantic_complexity: Literal["simple", "compound", "deep"]
    requires_synthesis: bool
```

`work_type` describes what the user wants done; `domains` identifies relevant data/domain boundaries; capability and dependency hints describe likely operations and semantic ordering; `semantic_complexity` describes language/meaning complexity, not the final route. For example, “People result is required before document search” is a dependency hint. It has no task ID and is not executable.

Examples:

```text
"CCCD của A là gì?"
work_type=lookup
 domains=[people]
capability_hints=[people.lookup]
semantic_complexity=simple

"Nghị định A quy định gì về X?"
work_type=explain
domains=[document]
capability_hints=[document.search]

"CCCD của A xuất hiện trong nghị định nào?"
work_type=cross_domain
domains=[people, document]
capability_hints=[people.lookup, document.search]
dependency_hints=[people result required before document search]
```

Memory is normally a supporting context capability, not a terminal domain route. “Tôi đã nói đơn vị tôi là gì?” may use direct/memory lookup; “đơn vị tôi có thuộc diện này không?” is memory enrichment followed by document reasoning and may be cross-domain.

Only `ComplexResearchGraph` creates the executable DAG containing task IDs, `depends_on`, capability names, document bindings, and completion criteria.

The deterministic router owns `RouteDecision`. It receives `SemanticContext`, `QueryAnalysis`, `DocumentBindingSet`, and `CapabilityRuntimeContext`:

```python
RouteReason = Literal[
    "direct_greeting", "direct_conversation", "essential_ambiguity",
    "unresolved_required_binding", "simple_people_lookup",
    "exact_document_metadata", "exact_section_retrieval",
    "simple_write_operation", "simple_kg_lookup",
    "multi_document_research", "cross_domain_dependency", "comparison",
    "compliance_evaluation", "multi_goal", "runtime_dependency",
    "evidence_replanning_required",
]

class RouteDecision(BaseModel):
    contract_version: Literal["2.0"]
    route: Literal["direct", "clarify", "fast_domain", "complex_research"]
    reason_code: RouteReason
    domain: Literal[
        "people", "document", "section", "write", "knowledge_graph",
    ] | None = None
```

Routing priority:

1. Greeting or deterministic conversational direct response → direct.
2. Essential ambiguity or unresolved required binding → clarify.
3. One domain + bounded operation + no runtime dependency + no iterative evidence acquisition → prefer fast domain.
4. Simple People lookup → fast People path after permission gate.
5. Document metadata/listing or exact section retrieval → fast Document/Section path.
6. Grammar check, suggest edits, format check, and rewrite/summarize supplied text → fast Write path.
7. Bounded KG lookup → fast Knowledge Graph path.
8. Multi-document independent research, cross-domain dependency, multi-evidence comparison, compliance/evaluation, multi-goal work, runtime task dependency, evidence-driven replanning, or hierarchical/map-reduce research → complex research.

Routing follows **execution complexity, not linguistic complexity**. A linguistically deep explanation of one exact section may use fast Section retrieval followed by synthesis without a research DAG. A short People → Document question uses complex research because its execution topology has a runtime dependency. `semantic_complexity == "deep"` alone never selects complex research.

Canonical acceptance table:

| Query | Route |
|---|---|
| Xin chào | direct |
| CCCD của A là gì | fast / People |
| Điều 5 A nói gì | fast / Section |
| Giải thích Điều 5 A | fast / Section + synthesis |
| Kiểm tra chính tả đoạn này | fast / Write |
| A thuộc đơn vị nào | fast / KG |
| Tóm tắt Chương II A | fast when exact bounded read is supported |
| Tóm tắt toàn bộ A rất dài | complex hierarchical/map-reduce |
| So sánh Chương II A và III B | complex |
| CCCD của A xuất hiện trong nghị định nào | complex |
| Kiểm tra F1/F2 theo A | complex |
| Kiểm tra F1/F2 theo quy định hiện hành | complex + discovery enabled |
| Dựa vào A sửa F1 | complex Document → Write |

Summary complexity is determined by required read topology, not `work_type` alone. Comparison defaults to complex, but if current authorized EvidenceRefs already prove complete bounded evidence for both sides, the router may choose synthesis-only comparison without new planner/research work; reused evidence is revalidated first.

## 13. Task plan and capability contracts

### 13.1 Executable plan

```python
class CoverageCriterion(BaseModel):
    kind: Literal["coverage"]
    target_id: str
    minimum_status: Literal["read_partial", "read_complete"] = "read_complete"
    allow_partial_reason: str | None = None

class ExactLookupCriterion(BaseModel):
    kind: Literal["exact_lookup"]
    field_name: str
    require_non_null: bool = True

class MinimumEvidenceCriterion(BaseModel):
    kind: Literal["minimum_evidence"]
    target_id: str | None = None
    minimum_count: int

class EntityResolutionCriterion(BaseModel):
    kind: Literal["entity_resolution"]
    ref_id: str
    require_unique: bool = True

class SemanticCriterion(BaseModel):
    kind: Literal["semantic"]
    criterion_id: str
    description: str

CompletionCriterion = Annotated[
    CoverageCriterion | ExactLookupCriterion | MinimumEvidenceCriterion |
    EntityResolutionCriterion | SemanticCriterion,
    Field(discriminator="kind"),
]

class TargetUnit(BaseModel):
    contract_version: Literal["2.0"]
    target_id: str
    binding_id: str
    document_id: UUID
    role: DocumentRole
    requested_locator: ContentLocator
    completion_criteria: tuple[CompletionCriterion, ...]

class TaskPlan(BaseModel):
    contract_version: Literal["2.0"]
    plan_id: str
    objective: str
    target_units: tuple[TargetUnit, ...]
    tasks: tuple[TaskSpec, ...]

class TaskSpec(BaseModel):
    contract_version: Literal["2.0"]
    task_id: str
    capability: str
    objective: str
    document_bindings: tuple[ScopedDocument, ...]
    input: CapabilityInput
    depends_on: tuple[str, ...]
    completion_criteria: tuple[CompletionCriterion, ...]
    replan_reason: str | None = None
    triggered_by_task_ids: tuple[str, ...] = ()
    triggered_by_evidence_ids: tuple[str, ...] = ()
```

Deterministic completion criteria (`coverage`, `exact_lookup`, `minimum_evidence`, `entity_resolution`) are enforced by the hard evaluator; only `semantic` criteria go to the semantic evaluator. `read_complete` is mandatory by default for legal comparison, compliance, and exact-section summary. `read_partial` is accepted only when the user/objective explicitly requests sampling/preview or a deterministic policy builder authorizes it and records `allow_partial_reason`; planner output cannot downgrade the default by itself. Runtime validates unique `target_id` values, target-unit bindings/locators, task IDs, acyclic dependencies, capability allowlist, document roles, budgets, and scope before execution. Every `CoverageObservation.target_id`, `CoverageItem.target_id`, and bound `EvidenceRecord.target_id`/`EvidenceRef.target_id` must reference a declared `TaskPlan.target_units` entry.

### 13.2 Business task request

```python
class AgentRequest(BaseModel):
    contract_version: Literal["2.0"]
    request_id: str
    task_id: str
    parent_task_id: str | None = None
    capability: str
    objective: str
    document_bindings: tuple[ScopedDocument, ...]
    input: CapabilityInput
    completion_criteria: tuple[CompletionCriterion, ...]
```

`CapabilityInput` is a discriminated union of capability-specific Pydantic models such as `PeopleLookupInput`, `DocumentSearchInput`, and `SectionReadInput`; it is not a free-form dictionary. Every input model forbids extra fields and excludes reserved trusted keys (`workspace_ids`, permission flags, capability allowlists, deadlines, runtime IDs). Document identifiers in an input must reference validated `document_bindings`. The same typed input rule applies to `TaskSpec`. `AgentRequest` excludes workspace authorization and permission fields.

### 13.3 Domain result

Capability outputs are a discriminated union; both input and output are typed:

```python
class PeopleLookupOutput(BaseModel):
    contract_version: Literal["2.0"]
    kind: Literal["people_lookup"]
    records: tuple[PersonResult, ...]

class DocumentSearchOutput(BaseModel):
    contract_version: Literal["2.0"]
    kind: Literal["document_search"]
    matches: tuple[DocumentDiscoveryCandidate, ...]

class DocumentReadOutput(BaseModel):
    contract_version: Literal["2.0"]
    kind: Literal["document_read"]
    locator: ContentLocator
    evidence_refs: tuple[EvidenceRef, ...]

class SectionReadOutput(BaseModel):
    contract_version: Literal["2.0"]
    kind: Literal["section_read"]
    locator: ContentLocator
    evidence_refs: tuple[EvidenceRef, ...]

class KnowledgeGraphOutput(BaseModel):
    contract_version: Literal["2.0"]
    kind: Literal["knowledge_graph"]
    facts: tuple[KnowledgeGraphFact, ...]

class WriteOutput(BaseModel):
    contract_version: Literal["2.0"]
    kind: Literal["write"]
    content: str

class MemoryLookupOutput(BaseModel):
    contract_version: Literal["2.0"]
    kind: Literal["memory_lookup"]
    memories: tuple[MemoryResult, ...]

class AbbreviationResolveOutput(BaseModel):
    contract_version: Literal["2.0"]
    kind: Literal["abbreviation_resolve"]
    resolutions: tuple[AbbreviationResolution, ...]

CapabilityOutput = Annotated[
    PeopleLookupOutput | DocumentSearchOutput | DocumentReadOutput |
    SectionReadOutput | KnowledgeGraphOutput | WriteOutput |
    MemoryLookupOutput | AbbreviationResolveOutput,
    Field(discriminator="kind"),
]

class AgentResult(BaseModel):
    contract_version: Literal["2.0"]
    request_id: str
    task_id: str
    status: Literal[
        "success", "partial", "not_found",
        "needs_input", "denied", "error",
    ]
    data: CapabilityOutput | None
    evidence_refs: tuple[EvidenceRef, ...]
    coverage_observations: tuple[CoverageObservation, ...]
    missing: tuple[MissingRequirement, ...]
    error: AgentError | None
```

`AgentRequest` is the capability execution contract and intentionally omits scheduler topology such as `depends_on`; `TaskSpec` is the orchestration contract, and the scheduler invokes a capability only after dependencies complete. `answer_mode` is not part of `AgentResult`. Template rendering versus model synthesis is Answer Policy owned by orchestration/presentation code. Capabilities report only `CoverageObservation` facts (resolved range, bytes/pages/chunks read, truncation, failure); the evaluator alone constructs authoritative `Coverage` by matching those observations and evidence against the requested target units.

Status semantics are strict:

- `success`: capability objective completed within runtime authorization;
- `partial`: useful result exists but declared requirements remain incomplete;
- `not_found`: lookup completed normally within scope and found nothing;
- `needs_input`: user/caller input is required;
- `denied`: deterministic permission or authorization gate rejected execution;
- `error`: runtime, dependency, timeout, or infrastructure failure.

Timeout and backend outage never become `not_found`.

`AgentResult.status` describes the execution outcome of one declared capability/task operation. `EvidenceEvaluation.status` describes whether the accumulated evidence is sufficient for the user objective. They are independent: `AgentResult.status="success"` may coexist with `CoverageItem.status="read_partial"` and `EvidenceEvaluation.status="insufficient"`. `AgentResult.status="partial"` is reserved for a capability that completed only part of its own declared operation; it does not represent global answer completeness. Therefore `success != sufficient`.

## 14. Coverage by logical target unit

Coverage is measured against required logical units, not only document UUIDs.

```python
class DocumentLocator(BaseModel):
    kind: Literal["document"]

class SectionLocator(BaseModel):
    kind: Literal["section"]
    structure_node_id: str
    heading_path: tuple[str, ...]

class ArticleLocator(BaseModel):
    kind: Literal["article"]
    structure_node_id: str
    article_id: str

class PageRangeLocator(BaseModel):
    kind: Literal["page_range"]
    start: int
    end: int

class ChunkRangeLocator(BaseModel):
    kind: Literal["chunk_range"]
    start: str
    end: str

ContentLocator = Annotated[
    DocumentLocator | SectionLocator | ArticleLocator |
    PageRangeLocator | ChunkRangeLocator,
    Field(discriminator="kind"),
]

class CoverageObservation(BaseModel):
    contract_version: Literal["2.0"]
    target_id: str
    binding_id: str
    document_id: UUID
    observed_locators: tuple[ContentLocator, ...]
    outcome: Literal["resolved", "read", "missing", "unreadable", "truncated"]

class CoverageItem(BaseModel):
    contract_version: Literal["2.0"]
    target_id: str
    binding_id: str
    document_id: UUID
    role: DocumentRole
    requested_locator: ContentLocator
    observed_locators: tuple[ContentLocator, ...]
    status: Literal[
        "resolved", "read_complete", "read_partial",
        "missing", "unreadable", "truncated",
    ]

class Coverage(BaseModel):
    contract_version: Literal["2.0"]
    items: tuple[CoverageItem, ...]
```

A target unit may represent:

- an entire document;
- Chapter II of document A;
- Article 5 of document A;
- an uploaded file;
- a required reference range.

Locator variants reject impossible field combinations by construction. Page/chunk endpoints must be ordered. Section/article locators require stable ingestion `structure_node_id`; human-readable heading/article text is descriptive, not canonical identity. The evaluator computes containment/coverage from structured locators and document structure metadata. Reading Chapter I of A cannot complete a requirement for Chapter II of A; it deterministically produces `missing` for that requested locator. Reading only part of Chapter II produces `read_partial`. A required unit is sufficient only when the requested locator is completely covered or the completion criteria explicitly allow partial coverage.

## 15. Evidence contract

```python
EvidencePurpose = Literal["discovery", "coverage", "supporting"]

class EvidenceRef(BaseModel):
    contract_version: Literal["2.0"]
    evidence_id: str
    task_id: str
    source_type: Literal["document", "knowledge_graph", "people", "memory"]
    role: DocumentRole | None
    purpose: EvidencePurpose
    target_id: str | None
    locator: ContentLocator | None
    document_revision: str | None
    content_hash: str

class EvidenceRecord(BaseModel):
    contract_version: Literal["2.0"]
    evidence_id: str
    request_id: str
    run_id: str
    parent_run_id: str | None = None
    task_id: str
    source_type: Literal["document", "knowledge_graph", "people", "memory"]
    role: DocumentRole | None
    binding_id: str | None
    target_id: str | None
    document_id: UUID | None
    workspace_id: UUID | None
    purpose: EvidencePurpose
    locator: ContentLocator | None
    display_section: str | None = None
    content: str
    content_hash: str
    source_identity: EvidenceSourceIdentity
    retention: EvidenceRetentionPolicy
    metadata: dict[str, Any]
    provenance: Provenance
```

Capabilities create full `EvidenceRecord` objects. Before graph state updates, the capability adapter validates and writes them to the Evidence Store, then returns compact `EvidenceRef` objects in `AgentResult`. The Evidence Store is authoritative for content/provenance; checkpoints contain IDs and compact metadata only.

Rules:

- `source_type=document` requires verified `document_id`, `workspace_id`, immutable document revision in `DocumentSourceIdentity`, and canonical `ContentLocator`;
- document evidence with role `target` or `reference` also requires `binding_id` and `target_id`, each matching a declared target unit;
- `discovered`/`supporting` document evidence may omit `target_id` only when it is not attached to a required unit;
- People, memory, and KG evidence may omit document fields but must carry identity in the discriminated `EvidenceSourceIdentity` union; free-form `metadata` is supplemental and never authoritative for identity or authorization;
- role, binding, task, document, and section provenance survive fan-out/fan-in;
- synthesis never infers source identity from evidence text;
- evidence from different documents is not deduplicated into one provenance record merely because content hashes match;
- `source_type` must match `source_identity.kind`; for document evidence, top-level `document_id`/`workspace_id` must equal `DocumentSourceIdentity.document_id`/`workspace_id`;
- an `EvidenceRef` is a deterministic projection of one authoritative `EvidenceRecord`: IDs, task, source type, role, purpose, target, locator, document revision, and content hash must match exactly;
- citation metadata must map to verified evidence, immutable document revision, and the same canonical locator coordinate system used by `TargetUnit` and `CoverageObservation`;
- `purpose="discovery"` search evidence can guide planning but cannot produce `read_complete`; only authoritative `document.read`/`section.read` evidence with `purpose="coverage"` can support coverage completion.

### 15.1 Evidence Store lifecycle

Evidence payloads are purpose-limited: store only fields required for the task, never a full People/Mongo record when one field and provenance suffice. The Evidence Store enforces authorization-bound reads, encryption at rest, source-specific retention/TTL, expiry and deletion behavior, PII minimization, and audited access. Every record carries an `EvidenceRetentionPolicy` classification and expiry.

Every record is keyed and indexed by `request_id`, `run_id`, `task_id`, and `evidence_id`; it also records optional `parent_run_id`, retention state, and validation state. Document evidence pins an immutable `document_revision`; a changed content revision makes old refs stale and forces reacquisition. Re-indexing may reuse evidence only under an explicit compatibility rule proving the content revision and locator coordinate system unchanged. Interrupt/resume in the same run reuses validated refs. Replans append new records and never mutate prior evidence. An expired run follows retention policy without breaking a still-valid checkpoint. A replacement run imports only `EvidenceRef` links to previously validated records, records lineage, and revalidates current authorization plus source availability before use; it never blindly copies raw payloads. Missing/expired evidence forces reacquisition or an insufficient result.

## 16. ComplexResearchGraph behavior

Planner input is explicit and minimal:

```python
class DiscoveryPolicy(BaseModel):
    contract_version: Literal["2.0"]
    allow_reference_discovery: bool
    allow_supporting_discovery: bool
    max_discovered_documents: int
    workspace_search_allowed: bool
    allowed_capabilities: tuple[str, ...]

class ResearchBudgetView(BaseModel):
    contract_version: Literal["2.0"]
    max_tasks_remaining: int
    max_replans_remaining: int
    max_parallel_branches: int

class CapabilityDescriptor(BaseModel):
    contract_version: Literal["2.0"]
    name: str
    version: str
    description: str
    domain: Literal["people", "document", "section", "write", "knowledge_graph", "memory"]
    operation_type: Literal["lookup", "search", "read", "transform", "resolve"]
    can_discover_documents: bool = False
    produces_coverage: bool = False
    supports_parallel: bool = False

class ResearchPlanningInput(BaseModel):
    contract_version: Literal["2.0"]
    semantic: SemanticContext
    bindings: DocumentBindingSet
    query_analysis: QueryAnalysis
    capability_catalog: tuple[CapabilityDescriptor, ...]
    discovery_policy: DiscoveryPolicy
    budget: ResearchBudgetView
    prior_evidence: tuple[EvidenceRef, ...]
    prior_evaluation: EvidenceEvaluation | None
```

It excludes DB sessions, raw clients, full chat history, ACL internals, and legacy `SupervisorState`. `ResearchBudgetView` exposes planning limits, not mutable runtime counters; runtime enforces them independently. The request-scoped capability registry/catalog is built as `base registrations ∩ permissions ∩ feature flags ∩ environment availability`; unavailable or unauthorized capabilities do not appear to the planner, while execute-time checks remain defense in depth. Descriptor flags tell the planner whether a capability may discover documents or produce authoritative coverage.

Authorization scope, semantic bindings, and discovery policy are separate: authorization says where reads are permitted; `DocumentBindingSet` says which documents/objective roles are in scope; `DiscoveryPolicy` says whether and how research may expand. A deterministic Research Policy Builder derives `DiscoveryPolicy` from finalized semantics and `QueryAnalysis`. “Compare A and B” normally disables discovery; “check A against current regulations” enables bounded reference discovery. The planner cannot exceed this policy.

Task semantic scope is local: each `TaskSpec.document_bindings` contains only bindings needed by that task, and `child task bindings ⊆ parent resolved bindings`. Reference-discovery tasks may search authorized workspaces only through an explicitly discovery-capable input/capability.

Capability semantics distinguish `document.search` (candidate passages/documents) from `document.read`/`section.read` (authoritative requested-unit read). Search success never marks target coverage `read_complete`; only a read capability with structured locator observations can do so.

```text
semantic objective + dependency hints
→ create executable TaskPlan/DAG
→ deterministic plan validation
→ dispatch ready tasks in parallel
→ call domain capabilities with AgentRequest + CapabilityRuntimeContext
→ persist EvidenceRecord in Evidence Store
→ collect AgentResult + EvidenceRef + CoverageObservation
→ deterministic evidence checks
→ optional small-model semantic evidence evaluation
    ├── sufficient → synthesis
    ├── required binding blocked → clarification
    ├── evidence gap → bounded append-only replan
    └── terminal failure → typed failure response
```

Plan rules:

- task IDs are unique;
- dependencies are acyclic;
- capability names must be request-authorized;
- child tasks cannot exceed authorized workspaces;
- completed task records are immutable;
- replanning only appends tasks and records `replan_reason`, `triggered_by_task_ids`, and `triggered_by_evidence_ids`; at least one trigger is required for every appended task;
- maximum branches, tasks, tool calls, replan rounds, deadlines, and cancellation are runtime-enforced;
- only the outer synthesis layer streams final-answer content.

### 16.1 Model-output trust boundary

LLM output is always a proposal. Model-produced `QueryAnalysis`, `TaskPlan`, binding requests, semantic `EvidenceEvaluation`, and `AnswerDraft` require Pydantic validation plus deterministic policy validation and authorization/scope validation where applicable. Models never mutate `DocumentBindingSet`, authoritative `Coverage`, `CapabilityRuntimeContext`, `EvidenceRecord`, or persisted canonical conversation identities.

### 16.2 Idempotency

Retries/resumes use stable `request_id`, `run_id`, `task_id`, `tool_call_id`, and `evidence_id`. Evidence insertion uses an idempotency key equivalent to `run_id + task_id + source identity + document revision + locator + content hash`. Repeating the same task/call cannot create uncontrolled duplicate evidence or completion records; append operations are idempotent and conflict-safe.

### 16.3 Cancellation

```text
user cancellation
→ cancel LangGraph run and pending tasks
→ propagate cancellation token/deadline to capabilities
→ stop safely and do not start replan/synthesis
```

Completed evidence follows retention policy. Incomplete work never becomes `success`; capability errors distinguish `CANCELLED`, `TIMEOUT`, and `BUDGET_EXHAUSTED`.

### 16.4 Shared capability implementations

Fast and complex paths share the same domain capabilities:

```text
Fast People Graph ─────┐
                       ├→ People Capability
ComplexResearchGraph ──┘
```

The same rule applies to Document, Section, Write, and Knowledge Graph. Paths differ in orchestration, not domain business logic.

## 17. Evidence evaluation, Answer Policy, and grounding

### 17.1 Hard deterministic evaluation

Validate:

- every required target/reference unit is resolved and read at the requested structured locator;
- every deterministic completion criterion (`coverage`, `exact_lookup`, `minimum_evidence`, `entity_resolution`) is satisfied;
- evidence provenance and content hashes are valid;
- citation references map to evidence;
- capability permission and status semantics are valid;
- timeout, error, and truncation are represented honestly.

### 17.2 Semantic evaluation

A small model may return a structured `EvidenceEvaluation` for:

- relevance to the objective;
- semantic coverage gaps;
- contradictions;
- sufficiency for synthesis;
- targeted research suggestions.

The semantic evaluator receives only `SemanticCriterion` entries and reports their satisfaction in structured form. The hard evaluator combines those structured semantic outcomes with its own deterministic-criterion results to produce final `EvidenceEvaluation`; it never interprets natural-language criteria itself. The model cannot modify evidence, coverage, authorization, or completion records.

### 17.3 Typed synthesis boundary

```python
class SynthesisInput(BaseModel):
    contract_version: Literal["2.0"]
    mode: Literal["fast", "complex"]
    semantic: SemanticContext
    bindings: DocumentBindingSet
    evaluation: EvidenceEvaluation | None
    evidence_refs: tuple[EvidenceRef, ...]

class CitationRef(BaseModel):
    citation_id: str
    evidence_id: str

class AnswerDraft(BaseModel):
    contract_version: Literal["2.0"]
    content: str
    citations: tuple[CitationRef, ...]
```

```text
EvidenceEvaluation → SynthesisInput → main model → AnswerDraft → Grounding → FinalResponse
```

For `mode="complex"`, validation requires `evaluation.status="sufficient"`. For `mode="fast"`, `evaluation` may be absent because deterministic route/capability/coverage prechecks supply synthesis readiness; all cited evidence still passes Grounding. The answer model emits typed `CitationRef` values, not authoritative free-form citation markers. Grounding follows each citation through `EvidenceRecord → DocumentSourceIdentity → document_revision → ContentLocator`. Unsupported citation yields `GroundingResult.revise`; failed revision yields `FinalResponse.insufficient`.

### 17.4 Answer Policy

Answer Policy—not the capability—decides between:

- deterministic template/verbatim rendering;
- main-model synthesis;
- clarification;
- insufficient-evidence fallback.

Fast People, metadata/listing, and exact section retrieval may use deterministic rendering. A fast path may still call synthesis—for example, exact Section retrieval followed by explanation—without invoking planner/evaluator/replanner. Compare, compliance, and cross-domain requests require synthesis after complex evidence collection.

### 17.5 Grounding

Grounding runs before successful completion of **every** answer. Deterministic/template responses use the hard validation subset; synthesized responses additionally run unsupported-claim checks and bounded revision:

- verify required-unit coverage;
- map citations to evidence and document metadata;
- detect unsupported claims or references;
- allow at most one bounded revision;
- return a transparent insufficient-evidence response if validation still fails.

## 18. Clarification and resume

The following conditions block synthesis:

- required target unresolved;
- required reference unresolved when the objective depends on it;
- required binding ambiguous;
- required unit unreadable;
- essential semantic ambiguity.

```python
class ClarificationResolution(BaseModel):
    contract_version: Literal["2.0"]
    clarification_id: str
    selected_candidate_id: str | None
    user_text: str

class ClarificationRequest(BaseModel):
    contract_version: Literal["2.0"]
    clarification_id: str
    reason: Literal[
        "required_document_not_found",
        "required_document_ambiguous",
        "required_unit_unreadable",
        "semantic_ambiguity",
    ]
    question: str
    unresolved_references: tuple[DocumentReference, ...]
    candidates: tuple[DocumentCandidate, ...]
    resumable: bool
    expires_at: datetime
```

Candidate lookup receives a trusted resolver view of the current `CapabilityRuntimeContext`; candidates are filtered in-query by authorized `workspace_ids` before any title or metadata is returned. Unauthorized identities are invisible to semantic clarification. If the user explicitly supplies an unauthorized UUID, return `denied` without title/metadata leakage, not `not_found` or an ambiguity candidate.

Candidates remain candidates until confirmation. Candidate ordering is persisted and stable within a clarification as `(clarification_id, candidate_id, ordinal, document_id)`. A response such as “cái thứ hai” maps deterministically against the persisted ordinal, never a newly searched ordering. Resume binds `ClarificationResolution.clarification_id + selected_candidate_id + user_text` deterministically; it validates that the candidate belonged to that clarification and then asks the Binding resolver to append the binding. It does not rerun unconstrained ambiguity selection. The main model cannot guess among equally plausible references.

The graph prefers LangGraph interrupt/resume. On resume:

1. backend recalculates current workspace authorization and People permission;
2. stored entity/document references are revalidated;
3. compatible checkpoints resume;
4. expired, failed, or contract-incompatible checkpoints create a replacement run linked by `parent_run_id`;
5. only validated semantic and clarification data is imported into the replacement run.

## 19. Conversation persistence, concurrency, and versioning

### 19.1 Raw user message invariant

V2 persists the raw ingress message unchanged before semantic preprocessing and assigns the same value to `RequestContext.original_query`:

```text
raw request.message / RequestContext.original_query
    ↓
persist raw/original ChatMessage.content
    ↓
Semantic Draft → Binding Resolver → Semantic Finalizer
    ↓
persist finalized semantic snapshot separately
```

V2 must not persist expanded or normalized text in place of raw user content. `chat_messages.semantic_context` stores only the versioned finalized `SemanticContext`, including canonical `document_id`/`binding_id` links. A draft is never labeled or persisted as final.

### 19.2 Per-message semantic snapshot

Persist:

- original, contextualized, and normalized query;
- resolved entities and provenance;
- abbreviation/coreference results;
- blocking ambiguities;
- contract, model, and config revisions.

### 19.3 Rolling thread summary

Persist:

- concise summary;
- structured active entities and last focus;
- open questions;
- `summary_version`;
- `built_through_message_id`.

### 19.4 Concurrent update policy

Rolling-summary writes use optimistic concurrency:

```text
read summary_version=N
→ build update through message M
→ UPDATE ... WHERE summary_version=N
→ success: version=N+1
→ conflict: reload latest summary and reconstruct/merge from missing ordered messages
```

A stale request cannot overwrite a newer summary. `built_through_message_id` must advance monotonically according to persisted message order. If compare-and-swap fails repeatedly, the request may continue with its per-message semantic snapshot while scheduling deterministic summary reconstruction; summary conflict is not permission to discard messages.

## 20. Existing Phase-1 disposition

The current Phase-1 code contains useful contracts and validation. V2 must reuse or adapt them rather than rewriting equivalent logic without reason.

| Existing artifact | V2 disposition |
|---|---|
| `semantic_preprocessor.PreprocessingResult` | Migrate concepts and validators into v2 `SemanticContext`; provide an adapter during transition |
| `AbbreviationEntry` | Reuse/adapt as `AbbreviationResolution`; preserve span, candidates, status, source, and confidence |
| `DocumentRefEntry` | Reuse/adapt into `DocumentReference` and then explicit `ScopedDocument` bindings |
| `BlockingAmbiguity` | Reuse/adapt; essential ambiguity remains a deterministic clarify gate |
| `TraceEvent` | Reuse/adapt for preprocessing observability; do not expose as business task data |
| Phase-1 `RoutingDecision` | Do not complete the old design for v2; v2 uses semantic `QueryAnalysis` plus deterministic `RouteDecision` |
| Phase-1 additions to `SupervisorState` | V1 compatibility only; not the v2 business interface |
| `chat_messages.semantic_context` 1.x | Define version-aware reader/adapter and migration to v2 snapshots |
| Existing abbreviation lookup/disambiguation | Reuse behind preprocessing and shared `abbreviation.resolve`; do not duplicate |

No Phase 1B router/preprocessor implementation should proceed in parallel under the superseded design. Work must first be classified as v1 compatibility or v2 contract migration.

## 21. Required use cases and expected flow

### 21.1 Simple People

```text
"CCCD của A là gì?"
→ deterministic/simple analysis
→ People permission gate
→ fast People capability
→ deterministic answer policy
```

### 21.2 Simple document retrieval

```text
"Điều 5 A nói gì?"
→ exact document + section resolution
→ fast Section capability
→ verbatim/template answer with citation
```

### 21.3 Conversational follow-up

```text
Turn 1: asks about A
Turn 2: "nghị định này..."
→ ConversationContext resolves A
→ current authorization revalidation
→ semantic normalization
→ routing
```

### 21.4 Ambiguous follow-up

```text
Turn 1: compares A and B
Turn 2: "nghị định này có hiệu lực khi nào?"
→ A and B both plausible
→ ClarificationRequest
```

### 21.5 Multi-document range comparison

```text
"So sánh Chương II A với Chương III B"
→ two required target units
→ complex DAG/fan-out
→ coverage requires the exact chapters
→ comparison synthesis
```

### 21.6 Cross-domain dependency

```text
"CCCD của A xuất hiện trong nghị định nào?"
→ People lookup task
→ CCCD result
→ dependent document research task
→ evidence evaluation and synthesis
```

### 21.7 Target/reference compliance

```text
"Kiểm tra F1/F2 có đúng quy định A"
→ F1/F2 required targets
→ A required reference
→ role-preserving evidence
→ compliance synthesis
```

### 21.8 Reference discovery

```text
"Kiểm tra F1/F2 theo quy định hiện hành"
→ F1/F2 required targets remain immutable
→ discover candidate references in authorized workspaces
→ bind selected references explicitly
→ evaluate target units against reference units
```

### 21.9 People permission denial

```text
can_read_people=false
→ People capability not exposed
→ direct invocation also denied
```

### 21.10 Resume after authorization change

```text
A was previously accessible
→ resume recalculates workspace authorization
→ current authorization wins
→ deny/clarify if A is no longer accessible
```

### 21.11 Abbreviation

```text
"NĐ 13 quy định gì về DLCN?"
→ original preserved
→ protected document identifier
→ shared abbreviation lookup
→ normalized SemanticContext
→ routing
```

### 21.12 Incomplete evidence

```text
many sources exist, but required section is missing
→ CoverageItem is missing/read_partial
→ EvidenceEvaluation insufficient
→ no complete synthesis
```

### 21.13 Simple Write

```text
"Kiểm tra chính tả đoạn này"
→ work_type=evaluate, domains=[write]
→ fast Write path
```

### 21.14 Write with document dependency

```text
"Dựa vào Nghị định A, sửa nội dung file F1"
→ required reference A + required target F1
→ document evidence → write dependency
→ complex research
```

### 21.15 Simple Knowledge Graph

```text
"A thuộc đơn vị nào?"
→ bounded KG lookup
→ fast Knowledge Graph path
```

### 21.16 KG with document dependency

```text
"So sánh quan hệ của A trong KG với quy định tại B"
→ KG + Document dependency
→ complex research
```

### 21.17 Irrelevant attachments

```text
attach A/B; ask a generic question
→ A/B remain contextual candidates
→ no automatic required binding
```

### 21.18 Unresolved named documents

```text
"So sánh Nghị định A và B"
→ RequestContext contains no fabricated UUIDs
→ semantic extraction creates DocumentReference entries
→ Binding resolver creates bindings or clarification
```

### 21.19 Wrong-section coverage

```text
request Chapter II; worker reads Chapter I
→ structured locator mismatch
→ CoverageItem.status=missing
→ EvidenceEvaluation.status=insufficient
```

### 21.20 Partial-section coverage

```text
request Chapter II; only half is read
→ CoverageItem.status=read_partial
→ EvidenceEvaluation.status=insufficient unless criteria explicitly allow partial
```

### 21.21 Discovered-reference promotion

```text
discover binding B1 for C
→ planner emits BindingPromotionRequest
→ Binding resolver validates and appends immutable B2 role=reference, required=true
→ B2.derived_from_binding_id=B1
→ B2.binding_reason recorded
```

### 21.22 Execution success with insufficient evidence

```text
AgentResult.status=success
+ CoverageItem.status=read_partial
→ EvidenceEvaluation.status=insufficient
```

## 22. Explicit invariants

1. `original_query` and raw persisted user content are immutable.
2. An LLM cannot create or widen workspace authorization.
3. An LLM cannot grant People or other permissions.
4. Context or coreference resolution does not grant access.
5. Required targets/references cannot be silently replaced by discovered documents.
6. Child tasks cannot exceed authorized workspace scope.
7. Target/reference roles and binding IDs survive fan-out/fan-in.
8. Document evidence requires a verified `document_id`.
9. Section/range requirements complete only when the requested unit is read.
10. Timeout or dependency outage is not `not_found`.
11. Essential ambiguity triggers clarification, not model guessing.
12. Completed task records are immutable.
13. Replanning is append-only and bounded.
14. Capability results contain no LangGraph routing instruction.
15. Only the outer synthesis layer streams final-answer content.
16. Grounding and citation validation occur before successful completion.
17. Current request/resume authorization overrides persisted conversation context.
18. Capabilities receive trusted scope only through `CapabilityRuntimeContext`.
19. Evidence deduplication never destroys source/task/document provenance.
20. Query analysis provides hints; only `ComplexResearchGraph` owns executable planning.
21. Text-named documents do not require resolved UUIDs in `RequestContext`; only the Binding resolver creates their bindings.
22. Attachments are contextual candidates and never become required bindings automatically.
23. Coverage completion is computed from structured `ContentLocator` values, not opaque range text or document identity alone.
24. `AgentResult.success` does not imply `EvidenceEvaluation.sufficient`.
25. Discovered-document promotion is requested by the planner but validated/appended only by the Binding resolver; it creates a new immutable binding with lineage and never mutates the original binding.
26. Routing follows execution topology, not `semantic_complexity` alone.
27. One bounded domain operation without runtime dependency or iterative acquisition prefers a fast path.
28. Business boundary contracts are immutable by default.
29. Trusted runtime authorization is injected and never stored in checkpointable graph state.
30. Only finalized post-binding `SemanticContext` is persisted as the semantic snapshot.
31. Hard completion criteria are typed; natural-language semantic criteria go only to the semantic evaluator.
32. Planner cannot autonomously promote a discovered document to `target`.
33. Known document identity at ingress carries no semantic role.
34. Full evidence content lives in the Evidence Store; checkpoints contain compact `EvidenceRef` values.
35. Search results cannot complete read coverage; authoritative read observations are required.
36. Unauthorized document identity is invisible to clarification; explicit unauthorized UUID access is `denied`.
37. Child task semantic bindings are a subset of parent resolved bindings except explicit authorized reference discovery.

## 23. Supporting boundary contracts

The following first-class objects are versioned Pydantic contracts, not implied dictionaries. `GroundingResult.citations` and `FinalResponse.citations` preserve the validated `citation_id → evidence_id` mapping emitted by `AnswerDraft`; `evidence_ids` is a rebuildable convenience projection:

```python
class ExecutionState(BaseModel):
    contract_version: Literal["2.0"]
    plan: TaskPlan | None
    task_results: tuple[AgentResult, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    evidence_evaluation: EvidenceEvaluation | None

class EvidenceEvaluation(BaseModel):
    contract_version: Literal["2.0"]
    status: Literal["sufficient", "insufficient", "contradictory", "needs_input"]
    coverage: Coverage
    missing: tuple[MissingRequirement, ...]
    contradictions: tuple[str, ...]
    suggested_research: tuple[str, ...]

class DocumentSourceIdentity(BaseModel):
    kind: Literal["document"]
    document_id: UUID
    document_revision: str
    ingestion_revision: str | None = None
    workspace_id: UUID

class PeopleSourceIdentity(BaseModel):
    kind: Literal["people"]
    record_id: str

class KnowledgeGraphSourceIdentity(BaseModel):
    kind: Literal["knowledge_graph"]
    entity_or_relation_id: str

class MemorySourceIdentity(BaseModel):
    kind: Literal["memory"]
    memory_id: str

EvidenceSourceIdentity = Annotated[
    DocumentSourceIdentity | PeopleSourceIdentity |
    KnowledgeGraphSourceIdentity | MemorySourceIdentity,
    Field(discriminator="kind"),
]

class DocumentDiscoveryCandidate(BaseModel):
    contract_version: Literal["2.0"]
    document_id: UUID
    source_task_id: str
    evidence_ids: tuple[str, ...]
    reason: str

class BindingAdditionRequest(BaseModel):
    contract_version: Literal["2.0"]
    document_id: UUID
    requested_role: Literal["discovered", "supporting"]
    reason: str
    triggered_by_task_ids: tuple[str, ...]
    triggered_by_evidence_ids: tuple[str, ...]

class BindingPromotionRequest(BaseModel):
    contract_version: Literal["2.0"]
    source_binding_id: str
    requested_role: Literal["reference", "supporting"]
    required: bool
    binding_reason: str
    triggered_by_task_ids: tuple[str, ...]
    triggered_by_evidence_ids: tuple[str, ...]

class DocumentCandidate(BaseModel):
    contract_version: Literal["2.0"]
    candidate_id: str
    ordinal: int
    ref_id: str
    document_id: UUID
    label: str
    match_basis: str
    confidence: float

class EvidenceRetentionPolicy(BaseModel):
    contract_version: Literal["2.0"]
    classification: Literal["normal", "personal", "sensitive_personal"]
    expires_at: datetime | None

class Provenance(BaseModel):
    contract_version: Literal["2.0"]
    source_id: str
    fetcher: str
    fetched_at: datetime
    tool_call_id: str | None
    run_id: str

class MissingRequirement(BaseModel):
    contract_version: Literal["2.0"]
    requirement_id: str
    description: str
    target_id: str | None

class AgentError(BaseModel):
    contract_version: Literal["2.0"]
    code: Literal[
        "INVALID_INPUT", "SCOPE_VIOLATION", "PERMISSION_DENIED",
        "AMBIGUOUS_ENTITY", "DEPENDENCY_UNAVAILABLE", "TIMEOUT",
        "CANCELLED", "BUDGET_EXHAUSTED", "CONTRACT_MISMATCH",
        "INTERNAL_ERROR",
    ]
    message: str
    retryable: bool

class GroundingResult(BaseModel):
    contract_version: Literal["2.0"]
    status: Literal["pass", "revise", "insufficient"]
    citations: tuple[CitationRef, ...]
    unsupported_claims: tuple[str, ...]
    citation_errors: tuple[str, ...]

class FinalResponse(BaseModel):
    contract_version: Literal["2.0"]
    status: Literal["success", "clarify", "denied", "insufficient", "error"]
    content: str
    citations: tuple[CitationRef, ...]
    evidence_ids: tuple[str, ...]
```

`AgentResult.evidence_refs` is the task-level source of truth. `ExecutionState.evidence_refs` is only a deterministic aggregate/cache and must equal the deduplicated union of `task_results[*].evidence_refs`; it is rebuildable and never appended independently.

Checkpoint compatibility is determined from graph version plus the versions of root and nested contracts. Version adapters may read older persisted snapshots; runtime-only state is never treated as durable business data.

## 24. Testing and benchmark gates

### 24.1 Contract tests

- reject unsupported versions and extra fields;
- reject model/tool attempts to inject trusted scope;
- reject plan cycles, unknown capabilities, invalid document roles, free-text hard criteria, and autonomous target promotion;
- require verified document identity for document evidence;
- preserve role/binding/task/evidence-store provenance;
- reject completion when any required target unit is missing or wrong-range;
- preserve distinct result status semantics;
- validate v1 semantic-snapshot migration/read compatibility;
- validate `DocumentReference` status invariants and post-binding semantic finalization;
- validate all `CompletionCriterion` variants and hard/semantic evaluator ownership;
- validate `ContentLocator` discriminators and stable structure-node matching;
- validate stable `RouteReason` values;
- validate every capability output variant is versioned, typed, and registered, including Memory and Abbreviation;
- validate `EvidenceRef` is an exact projection of its `EvidenceRecord` and document source identities cannot disagree;
- validate fast/complex `SynthesisInput` invariants and citation mapping preservation through `AnswerDraft → GroundingResult → FinalResponse`.

### 24.2 Subgraph tests

Compile and test each subgraph independently for:

- input/output validation;
- deterministic routing and terminal state;
- conditional semantic-model invocation;
- malformed structured model output;
- permission denial;
- timeout, retry, cancellation, and budget exhaustion;
- clarification interrupt/resume and deterministic candidate binding;
- conversation-summary compare-and-swap conflicts;
- runtime context absent from serialized checkpoints;
- EvidenceRef compaction, store retention, replacement-run revalidation;
- search success without read-complete coverage;
- deterministic QueryAnalysis with zero small-model/planner calls for clear direct/People/Section/Write cases;
- memory direct lookup versus memory-as-supporting-dependency routing;
- explicit unauthorized UUID denial without candidate metadata leakage;
- task-local binding subsets and explicit reference-discovery scope.

### 24.3 End-to-end tests

All twenty-two flows in Section 21 are mandatory acceptance scenarios.

### 24.4 Benchmarks

Track separately for fast and complex paths:

- routing accuracy;
- coreference and abbreviation accuracy;
- required-unit coverage;
- citation faithfulness;
- permission and scope violations;
- p50/p95 latency;
- token/tool-call cost;
- replan and clarification rate;
- answer correctness against v1 baselines.

Fast-path latency must not regress materially because of conversation or semantic processing. The exact threshold is an implementation-plan decision backed by baseline measurement.

## 25. Observability

Record:

- graph, contract, model, and config versions;
- semantic gate outcome and whether a small model ran;
- reconciliation conflicts;
- route and reason code;
- initial plan and append-only replans;
- task status and latency;
- required-unit coverage;
- document role/binding IDs;
- People permission denials;
- evidence IDs and grounding failures;
- interrupt/resume lineage;
- summary concurrency conflicts;
- token, branch, task, tool-call, and replan budgets.

Logs and traces redact or hash CCCD, BHXH, phone numbers, and other sensitive values.

## 26. Migration strategy

Do not rewrite business services in one step.

### Phase 0 — Orchestrator compatibility and benchmark spike

- compare native LangGraph planner/executor with a Deep Agents adapter against the same contract fixtures;
- verify structured planning, interrupts, streaming ownership, capability injection, checkpoint behavior, and bounded replanning;
- produce a recorded architecture decision selecting the initial implementation;
- keep all spike code isolated and non-production.

This bounded spike produces the architecture decision required before Phase 3 complex-orchestrator implementation. The contract and behavior design may be promoted to **Approved design** while the implementation choice remains open; approval does not imply selection of native LangGraph or a Deep Agents adapter.

### Phase 1 — Contracts, context semantics, and adapters

- define v2 contracts, discriminated content locators/completion criteria, and semantic-draft/binding/finalization lifecycle;
- define Evidence Store/EvidenceRef persistence and retention;
- add v1-to-v2 semantic adapters;
- preserve raw ingress content;
- define versioned semantic snapshot reads/writes;
- reuse abbreviation preprocessing.

### Phase 2 — Supervisor v2 composition and fast paths

- build `supervisor_v2.py` composition with injected `GraphRuntimeContext` outside checkpoints;
- integrate Context, Binding Resolution, Routing, People, Document, Section, Write, and bounded KG subgraphs;
- use existing services through capability adapters;
- establish fast-path latency baselines and gates.

### Phase 3 — Complex research comparison pilot

- implement the orchestrator selected by the Phase 0 architecture decision;
- pilot multi-document/range comparison;
- validate role-based binding and target-unit coverage.

### Phase 4 — Cross-domain People to Documents

- add validated dependent People → Document research;
- enforce People gating and evidence separation.

### Phase 5 — Evidence evaluator and bounded replanning

- add hard evaluator;
- add conditional semantic evaluator;
- enable append-only bounded replanning.

### Phase 6 — Shadow and canary

- offline replay;
- shadow semantic/routing decisions;
- canary by user/cohort;
- gradual traffic increase.

### Phase 7 — V1 disposition

Only after benchmark gates pass, consider deprecating the custom v1 supervisor/ReAct path in a separate reviewed change.

## 27. Rollout configuration

Use an integration-boundary setting such as:

```text
NEXUSRAG_AGENT_GRAPH_VERSION=v1|v2
```

The selector lives outside `supervisor.py` and `supervisor_v2.py`. Architecture-changing implementation must update `CLAUDE.md`, `.env.example`, relevant harness documentation, and this spec in the same change. Re-index GitNexus after structural changes.

## 28. Final contract data flow

```text
Backend
├── authenticate/current authorization/People permission
├── persist raw message
└── KnownDocumentResource[]
        ↓
ConversationContext → SemanticDraft
        ↓
Binding Resolver → DocumentBindingSet
        ↓
Semantic Finalizer → SemanticContext
        ↓
Deterministic-first QueryAnalysis → RouteDecision
        ├── direct
        ├── clarify
        ├── fast domain → shared capability
        └── complex research
               ↓
        ResearchPlanningInput
        ├── SemanticContext
        ├── DocumentBindingSet
        ├── QueryAnalysis
        ├── DiscoveryPolicy
        ├── CapabilityDescriptor[]
        └── ResearchBudgetView
               ↓
        TaskPlan → AgentRequest + CapabilityRuntimeContext
               ↓
        shared typed Capability → CapabilityOutput
        + EvidenceRecord + CoverageObservation
               ↓
        Evidence Store → EvidenceRef[]
               ↓
        Coverage / EvidenceEvaluation
               ↓
        SynthesisInput → AnswerDraft
               ↓
        Grounding → FinalResponse
```

## 29. Final ownership principle

```text
LangGraph owns workflow.
Backend owns trust and authorization.
Conversation layer owns discourse context.
Semantic layer owns query meaning.
Binding Resolver owns document semantic bindings.
Router owns fast vs complex execution.
ComplexResearchGraph owns research planning.
Capabilities own domain operations.
Evidence Store owns evidence payload/provenance.
Evaluator owns sufficiency.
Answer layer owns synthesis and grounding.
```

LLM/subagent output is never authoritative for scope, permission, document identity, binding state, coverage, or evidence integrity.

## 30. Open decision and approval gate

This revision is **Approved design**. It closes all final consistency gates:

1. clarification uses stable `candidate_id` and persisted ordinal mapping;
2. document evidence pins immutable revision identity;
3. Evidence and Coverage share `ContentLocator`;
4. capability inputs and outputs are discriminated typed unions;
5. discovery → addition → promotion is explicit and Binding-Resolver-owned;
6. partial coverage requires deterministic authorization and reason;
7. `DiscoveryPolicy` is separate from authorization and bindings;
8. Evidence Store owns security, retention, minimization, audit, and payload provenance;
9. `SynthesisInput → AnswerDraft → Grounding → FinalResponse` is typed.

The following implementation decision intentionally remains open for the Phase 0 compatibility/benchmark spike:

1. `ComplexResearchGraph` implementation: native LangGraph, Deep Agents adapter, or another conforming orchestrator.

Phase 1/2 contract, adapter, and fast-path planning may proceed from this approved baseline. Phase 3 complex-orchestrator implementation additionally requires the Phase 0 architecture decision.

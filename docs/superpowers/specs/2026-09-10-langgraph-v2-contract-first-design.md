# LangGraph v2 Contract-First Architecture

**Date:** 2026-09-10

**Status:** Review / Proposed

**Revision basis:** commit `977e1f12b87efe49a70e25ab2b287986e5c3f2bd`

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
└── load conversation context
        ↓
Context / Semantic Subgraph
├── deterministic extraction
├── conversational coreference resolution
├── protected abbreviation resolution
├── optional small semantic model
├── reconciliation
└── persist semantic snapshot
        ↓
Query Analysis
        ↓
Deterministic Router
├── clarify
├── fast domain path
└── complex research
        ↓
ComplexResearchGraph
├── create executable DAG
├── validate DAG
├── dispatch domain tasks
├── AgentRequest + CapabilityRuntimeContext
├── AgentResult + Evidence + Coverage
├── coverage/evidence evaluation
├── bounded append-only replan
└── synthesis
        ↓
Grounding
├── required-target coverage
├── evidence/citation validation
└── unsupported-claim handling
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

Initial domains are People, Document, and Section. KG becomes a separate subgraph only for a validated use case.

## 6. Data objects and ownership

```text
Raw User Request
    ↓
RequestContext
+
ConversationContext
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
AgentResult
+
Evidence
+
Coverage
    ↓
EvidenceEvaluation
    ↓
Synthesis
    ↓
GroundingResult
    ↓
FinalResponse
```

| Object | Owner | Notes |
|---|---|---|
| `RequestContext` | Backend ingress | Authenticated request metadata and initial required-document bindings |
| `CapabilityRuntimeContext` | Backend/runtime adapter | Trusted authorization, capability allowlist, deadline |
| `ConversationContext` | Context layer | Short-term discourse state sourced from chat DB |
| `SemanticContext` | Context layer | Meaning of the current request after resolution/normalization |
| `QueryAnalysis` | Query analyzer | Semantic structure and dependency hints, not an executable plan |
| `RouteDecision` | Deterministic router | `clarify`, fast domain path, or complex research |
| `TaskPlan` / DAG | `ComplexResearchGraph` | Executable tasks and dependencies |
| `AgentRequest` | Planner/orchestrator | Business operation requested from a capability |
| `AgentResult` / `Evidence` | Capability | Business result and provenance-anchored evidence |
| `CoverageObservation` | Capability | Facts observed while executing a task; never authoritative completion |
| `Coverage` / `EvidenceEvaluation` | Evaluator | Authoritative requirement-level completion and semantic sufficiency |
| `GroundingResult` / `FinalResponse` | Answer layer | Validated user-facing output |

## 7. Core contract rules

All boundary and persisted contracts are versioned Pydantic v2 models with forbidden extra fields. Runtime-only objects such as DB sessions, cancellation events, LangGraph commands, and raw clients are excluded from persisted business contracts.

### 7.1 Root graph state

```python
class SupervisorV2State(TypedDict):
    request: RequestContext
    runtime: CapabilityRuntimeContext
    conversation: ConversationContext
    semantic: SemanticContext
    query_analysis: QueryAnalysis | None
    route_decision: RouteDecision | None
    execution: ExecutionState
    clarification: ClarificationRequest | None
    grounding: GroundingResult | None
    final_response: FinalResponse | None
```

Nested objects cross subgraph boundaries through explicit input/output adapters. A domain subgraph never receives the entire root state unless its input schema explicitly requires every field. `runtime` is trusted, request-scoped, and never persisted or exposed as model-controlled state. Context/document resolvers and clarification-candidate lookup receive a read-only resolver view derived from `CapabilityRuntimeContext`, so every candidate query is filtered by current `workspace_ids` before identity or metadata is returned.

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
    required_documents: tuple[ScopedDocument, ...]
    attached_document_ids: tuple[UUID, ...]
```

Authorization fields are deliberately absent. They belong to trusted runtime context, not the business request.

## 8. Document roles and required scope

### 8.1 Role-based binding

V2 adopts role-based document binding as the canonical contract:

```python
class ScopedDocument(BaseModel):
    contract_version: Literal["2.0"]
    binding_id: str
    document_id: UUID
    role: Literal["target", "reference", "supporting", "discovered"]
    required: bool
    source_ref_id: str | None = None
    section_ref: str | None = None
```

Semantics:

- `target`: subject being read, analyzed, compared, or evaluated;
- `reference`: normative or comparative basis required for the objective;
- `supporting`: explicitly bound supplemental material;
- `discovered`: material found during research and not silently promoted to target/reference.

A required binding is a completion requirement, not an authorization grant. Authorization still comes only from `CapabilityRuntimeContext.workspace_ids`. `required` is explicit: user-bound targets and references are normally `True`; incidental supporting/discovered bindings are `False` unless a validated plan deliberately promotes them into a new required binding. Role-sensitive validation rejects an accidental `discovered + required=True` binding without an explicit promotion reason.

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
class SemanticContext(BaseModel):
    contract_version: Literal["2.0"]
    original_query: str
    contextualized_query: str
    normalized_query: str
    abbreviations: tuple[AbbreviationResolution, ...]
    coreferences: tuple[CoreferenceResolution, ...]
    document_refs: tuple[DocumentReference, ...]
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

Query analysis runs only after this stage.

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

`QueryAnalysis` describes semantic structure; it is not an executable planner.

```python
class QueryAnalysis(BaseModel):
    contract_version: Literal["2.0"]
    work_type: Literal[
        "direct", "lookup", "retrieve", "explain", "compare",
        "summarize", "cross_domain", "multi_goal",
    ]
    semantic_complexity: Literal["simple", "compound", "deep"]
    capability_hints: tuple[str, ...]
    dependency_hints: tuple[SemanticDependencyHint, ...]
    requires_synthesis: bool
```

For example, “People result is required before document search” is a dependency hint. It has no task ID and is not executable.

Only `ComplexResearchGraph` creates the executable DAG containing task IDs, `depends_on`, capability names, document bindings, and completion criteria.

The deterministic router owns `RouteDecision`:

```python
class RouteDecision(BaseModel):
    contract_version: Literal["2.0"]
    route: Literal["direct", "clarify", "fast_domain", "complex_research"]
    reason_code: str
    domain: str | None = None
```

Routing priority:

1. Greeting or deterministic conversational direct response → direct.
2. Essential ambiguity or unresolved required binding → clarify.
3. Simple People lookup → permission gate and fast People path.
4. Document metadata/listing → fast Document path.
5. Exact section retrieval → fast Section path.
6. Compare, cross-domain, multi-goal, dependent research, compliance, or unresolved evidence needs → complex research.
7. Other bounded work → domain path and Answer Policy.

## 13. Task plan and capability contracts

### 13.1 Executable plan

```python
class TaskPlan(BaseModel):
    contract_version: Literal["2.0"]
    plan_id: str
    objective: str
    tasks: tuple[TaskSpec, ...]

class TaskSpec(BaseModel):
    contract_version: Literal["2.0"]
    task_id: str
    capability: str
    objective: str
    document_bindings: tuple[ScopedDocument, ...]
    input: CapabilityInput
    depends_on: tuple[str, ...]
    completion_criteria: tuple[str, ...]
    replan_reason: str | None = None
    triggered_by_task_ids: tuple[str, ...] = ()
    triggered_by_evidence_ids: tuple[str, ...] = ()
```

Runtime validates task IDs, acyclic dependencies, capability allowlist, document roles, budgets, and scope before execution.

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
    depends_on: tuple[str, ...]
    completion_criteria: tuple[str, ...]
```

`CapabilityInput` is a discriminated union of capability-specific Pydantic models such as `PeopleLookupInput`, `DocumentSearchInput`, and `SectionReadInput`; it is not a free-form dictionary. Every input model forbids extra fields and excludes reserved trusted keys (`workspace_ids`, permission flags, capability allowlists, deadlines, runtime IDs). Document identifiers in an input must reference validated `document_bindings`. The same typed input rule applies to `TaskSpec`. `AgentRequest` excludes workspace authorization and permission fields.

### 13.3 Domain result

```python
class AgentResult(BaseModel):
    contract_version: Literal["2.0"]
    request_id: str
    task_id: str
    status: Literal[
        "success", "partial", "not_found",
        "needs_input", "denied", "error",
    ]
    data: dict[str, Any] | None
    evidence: tuple[Evidence, ...]
    coverage_observations: tuple[CoverageObservation, ...]
    missing: tuple[MissingRequirement, ...]
    error: AgentError | None
```

`answer_mode` is not part of `AgentResult`. Template rendering versus model synthesis is Answer Policy owned by orchestration/presentation code. Capabilities report only `CoverageObservation` facts (resolved range, bytes/pages/chunks read, truncation, failure); the evaluator alone constructs authoritative `Coverage` by matching those observations and evidence against the requested target units.

Status semantics are strict:

- `success`: capability objective completed within runtime authorization;
- `partial`: useful result exists but declared requirements remain incomplete;
- `not_found`: lookup completed normally within scope and found nothing;
- `needs_input`: user/caller input is required;
- `denied`: deterministic permission or authorization gate rejected execution;
- `error`: runtime, dependency, timeout, or infrastructure failure.

Timeout and backend outage never become `not_found`.

## 14. Coverage by logical target unit

Coverage is measured against required logical units, not only document UUIDs.

```python
class CoverageObservation(BaseModel):
    contract_version: Literal["2.0"]
    target_id: str
    binding_id: str
    document_id: UUID
    observed_range: str | None = None
    outcome: Literal["resolved", "read", "missing", "unreadable", "truncated"]

class CoverageItem(BaseModel):
    contract_version: Literal["2.0"]
    target_id: str
    binding_id: str
    document_id: UUID
    role: Literal["target", "reference", "supporting", "discovered"]
    section_ref: str | None = None
    requested_range: str | None = None
    observed_range: str | None = None
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

Reading Chapter I of A cannot complete a requirement for Chapter II of A. A required unit is sufficient only when the requested range is read completely or the completion criteria explicitly allow partial coverage.

## 15. Evidence contract

```python
class Evidence(BaseModel):
    contract_version: Literal["2.0"]
    evidence_id: str
    task_id: str
    source_type: Literal["document", "knowledge_graph", "people", "memory"]
    role: Literal["target", "reference", "discovered", "supporting"]
    binding_id: str | None
    target_id: str | None
    document_id: UUID | None
    workspace_id: UUID | None
    section_path: str | None
    page_or_chunk: str | None
    content: str
    content_hash: str
    metadata: dict[str, Any]
    provenance: Provenance
```

Rules:

- `source_type=document` requires verified `document_id` and `workspace_id`;
- document evidence with role `target` or `reference` also requires `binding_id` and `target_id`, each matching a declared target unit;
- `discovered`/`supporting` document evidence may omit `target_id` only when it is not attached to a required unit;
- People, memory, and KG evidence may omit document fields but must carry source-specific identity in typed provenance/metadata;
- role, binding, task, document, and section provenance survive fan-out/fan-in;
- synthesis never infers source identity from evidence text;
- evidence from different documents is not deduplicated into one provenance record merely because content hashes match;
- citation metadata must map to verified evidence and document metadata.

## 16. ComplexResearchGraph behavior

```text
semantic objective + dependency hints
→ create executable TaskPlan/DAG
→ deterministic plan validation
→ dispatch ready tasks in parallel
→ call domain capabilities with AgentRequest + CapabilityRuntimeContext
→ collect AgentResult + Evidence + Coverage
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

## 17. Evidence evaluation, Answer Policy, and grounding

### 17.1 Hard deterministic evaluation

Validate:

- every required target/reference unit is resolved and read at the requested range;
- task completion criteria are satisfied;
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

The model cannot modify evidence, coverage, authorization, or completion records.

### 17.3 Answer Policy

Answer Policy—not the capability—decides between:

- deterministic template/verbatim rendering;
- main-model synthesis;
- clarification;
- insufficient-evidence fallback.

Fast People, metadata/listing, and exact section retrieval may use deterministic rendering. Explain, compare, compliance, summarize, and cross-domain requests require synthesis.

### 17.4 Grounding

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
    unresolved_bindings: tuple[ScopedDocumentCandidate, ...]
    candidates: tuple[DocumentCandidate, ...]
    resumable: bool
    expires_at: datetime
```

Candidate lookup receives a trusted resolver view of the current `CapabilityRuntimeContext`; candidates are filtered in-query by authorized `workspace_ids` before any title or metadata is returned. Candidates remain candidates until the user confirms a binding. The main model cannot guess among equally plausible conversational references.

The graph prefers LangGraph interrupt/resume. On resume:

1. backend recalculates current workspace authorization and People permission;
2. stored entity/document references are revalidated;
3. compatible checkpoints resume;
4. expired, failed, or contract-incompatible checkpoints create a replacement run linked by `parent_run_id`;
5. only validated semantic and clarification data is imported into the replacement run.

## 19. Conversation persistence, concurrency, and versioning

### 19.1 Raw user message invariant

V2 persists `request.message` unchanged before semantic preprocessing:

```text
request.message
    ↓
persist raw/original ChatMessage.content
    ↓
Context / Semantic Subgraph
    ↓
persist semantic snapshot separately
```

V2 must not persist expanded or normalized text in place of raw user content. `chat_messages.semantic_context` stores the versioned semantic snapshot.

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

## 23. Supporting boundary contracts

The following first-class objects are versioned Pydantic contracts, not implied dictionaries:

```python
class ExecutionState(BaseModel):
    contract_version: Literal["2.0"]
    plan: TaskPlan | None
    task_results: tuple[AgentResult, ...]
    evidence_evaluation: EvidenceEvaluation | None

class EvidenceEvaluation(BaseModel):
    contract_version: Literal["2.0"]
    status: Literal["sufficient", "insufficient", "contradictory", "needs_input"]
    coverage: Coverage
    missing: tuple[MissingRequirement, ...]
    contradictions: tuple[str, ...]
    suggested_research: tuple[str, ...]

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
    code: str
    message: str
    retryable: bool

class GroundingResult(BaseModel):
    contract_version: Literal["2.0"]
    status: Literal["pass", "revise", "insufficient"]
    unsupported_claims: tuple[str, ...]
    citation_errors: tuple[str, ...]

class FinalResponse(BaseModel):
    contract_version: Literal["2.0"]
    status: Literal["success", "clarify", "denied", "insufficient", "error"]
    content: str
    evidence_ids: tuple[str, ...]
```

Checkpoint compatibility is determined from graph version plus the versions of root and nested contracts. Version adapters may read older persisted snapshots; runtime-only state is never treated as durable business data.

## 24. Testing and benchmark gates

### 24.1 Contract tests

- reject unsupported versions and extra fields;
- reject model/tool attempts to inject trusted scope;
- reject plan cycles, unknown capabilities, and invalid document roles;
- require verified document identity for document evidence;
- preserve role/binding/task provenance;
- reject completion when any required target unit is missing or wrong-range;
- preserve distinct result status semantics;
- validate v1 semantic-snapshot migration/read compatibility.

### 24.2 Subgraph tests

Compile and test each subgraph independently for:

- input/output validation;
- deterministic routing and terminal state;
- conditional semantic-model invocation;
- malformed structured model output;
- permission denial;
- timeout, retry, cancellation, and budget exhaustion;
- clarification interrupt/resume;
- conversation-summary compare-and-swap conflicts.

### 24.3 End-to-end tests

The twelve flows in Section 21 are mandatory acceptance scenarios.

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

This bounded spike is explicitly exempt from the no-runtime gate only as disposable evaluation code. No production graph composition begins until the architecture decision is approved and this spec is promoted to **Approved design**.

### Phase 1 — Contracts, context semantics, and adapters

- define v2 contracts;
- add v1-to-v2 semantic adapters;
- preserve raw ingress content;
- define versioned semantic snapshot reads/writes;
- reuse abbreviation preprocessing.

### Phase 2 — Supervisor v2 composition and fast paths

- build `supervisor_v2.py` composition;
- integrate Context, Routing, People, Document, and Section subgraphs;
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

## 28. Open decisions and approval gate

This revision is **Review / Proposed**, not implementation-approved, because the following decision remains open:

1. `ComplexResearchGraph` implementation: native LangGraph, Deep Agents adapter, or another conforming orchestrator. Resolve through a bounded compatibility/benchmark spike before implementation selection.

The following design choices are fixed by this revision:

- role-based `ScopedDocument` bindings for target/reference/supporting/discovered semantics;
- conditional, not mandatory, small semantic-model execution;
- QueryAnalysis as semantic hints rather than an executable planner;
- typed `CapabilityRuntimeContext` separate from `AgentRequest`;
- target-unit/range-level coverage;
- Answer Policy outside `AgentResult`.

No runtime code should be implemented merely to match this spec until the open orchestrator decision is reviewed and the status is explicitly promoted to **Approved design**.

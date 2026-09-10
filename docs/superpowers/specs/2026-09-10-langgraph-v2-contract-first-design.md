# LangGraph v2 Contract-First Architecture

**Date:** 2026-09-10  
**Status:** Approved design  
**Source direction:** `docs/agent-contract-langgraph-deepagent.md`  
**Supersedes for this initiative:** the previous DeepAgent implementation specs and plans

## 1. Objective

Build a new contract-first LangGraph architecture without refactoring the existing `supervisor.py` in place. The new graph is composed from independently testable domain subgraphs, preserves deterministic fast paths, and uses a DeepAgent subgraph for dependent, multi-step research.

“DeepAgent” in this design means an AIRAG orchestration subgraph implemented with LangGraph primitives. It is not an external library.

## 2. Decisions

1. Create `backend/app/services/agents/supervisor_v2.py`; do not add v2 behavior to `supervisor.py`.
2. Select v1 or v2 at an external integration boundary using `NEXUSRAG_AGENT_GRAPH_VERSION=v1|v2`.
3. Organize v2 as domain subgraphs with minimal typed input/output state.
4. Use the existing chat-history database as the conversation source of truth. LangGraph checkpoints support execution, interrupt, and resume only.
5. Run deterministic semantic extraction and a small semantic model in parallel, then reconcile their outputs.
6. Persist a rolling thread summary and a semantic snapshot for each user message.
7. Use a small model for semantic analysis, query analysis, conversation summary, and semantic evidence evaluation.
8. Use deterministic code for permission gates, routing policy, scope enforcement, budgets, contract validation, and hard evidence checks.
9. Use the main model for complex planning, evidence reasoning, and final-answer synthesis.
10. Allow template responses for simple People lookup, document metadata/listing, and exact section retrieval.
11. Use a hybrid DeepAgent planner: initial dependency DAG, parallel ready-task execution, evidence evaluation, and bounded append-only replanning.
12. A required target that cannot be resolved or read blocks synthesis and triggers clarification with candidate documents.

## 3. Authorization and scope semantics

The backend authenticates the caller and passes:

```python
workspace_ids: list[UUID]
can_read_people: bool
```

`workspace_ids` is the complete document-search authorization boundary for the request. A user may read all documents within those workspaces. DeepAgent does not perform ACL reasoning. Capability adapters inject the trusted workspace list; an LLM cannot supply or modify it in tool input.

`target_document_ids` has business—not authorization—semantics. It identifies documents that the user explicitly selected, attached, quoted, or required the system to read. DeepAgent may discover supporting documents within `workspace_ids`, but discovered documents do not satisfy an unread required target.

People data uses one deterministic gate:

- `can_read_people=false`: People capabilities are not exposed; direct invocation is denied by the service as defense in depth.
- `can_read_people=true`: People lookup, including CCCD/BHXH, is permitted.

Permission values are recalculated by the backend on resume. They are never inferred by a model or persisted as conversational memory.

## 4. Top-level architecture

```text
Backend ingress
├── authenticate caller
├── calculate workspace_ids
├── calculate can_read_people
└── load conversation history
          ↓
supervisor_v2.py
          ↓
Context Subgraph
├── deterministic extraction
├── small-model semantic analysis
├── reconciliation
└── semantic-context persistence
          ↓
Routing Subgraph
├── small-model QueryAnalysis
└── deterministic fast/deep/clarify routing
          ↓
   ┌──────┼────────────────┐
   ↓      ↓                ↓
Clarify  Fast path       DeepAgent Subgraph
         ├── People      ├── initial plan
         ├── Document    ├── validate DAG
         └── Section     ├── dispatch domain subgraphs
                         ├── collect evidence
                         ├── evaluate evidence
                         ├── bounded replan
                         └── main-model synthesis
                              ↓
                       Grounding Subgraph
                       ├── citation validation
                       ├── target coverage
                       └── final response
```

## 5. Module layout

```text
backend/app/services/agents/
├── supervisor.py                    # unchanged v1
├── supervisor_v2.py                 # v2 composition root
└── v2/
    ├── contracts/
    │   ├── context.py
    │   ├── execution.py
    │   ├── task.py
    │   └── evidence.py
    ├── context_graph/
    ├── routing_graph/
    ├── deep_agent_graph/
    ├── people_graph/
    ├── document_graph/
    ├── section_graph/
    ├── grounding_graph/
    ├── capabilities/
    ├── persistence/
    └── adapters/
```

`supervisor_v2.py` only composes subgraphs, declares edges, and exposes the public graph builder. Domain logic belongs in focused modules.

Each subgraph:

- has its own minimal input/output schema;
- does not read arbitrary root-state fields;
- communicates through shared business contracts;
- can be compiled and tested independently;
- may use LangGraph `Command` or `Send` internally, but those routing primitives never appear in business contracts.

Initial domains are People, Document, and Section. KG becomes a separate subgraph only when required by a validated use case.

## 6. Core contracts

All persisted and boundary contracts are versioned Pydantic v2 models with forbidden extra fields. Runtime-only LangGraph state remains distinct from business contracts.

### 6.1 Root state

```python
class SupervisorV2State(TypedDict):
    request: RequestContext
    conversation: ConversationContext
    semantic: SemanticContext
    query_analysis: QueryAnalysis | None
    execution: ExecutionState
    clarification: ClarificationRequest | None
    final_response: FinalResponse | None
```

### 6.2 Request context

```python
class RequestContext(BaseModel):
    contract_version: Literal["2.0"]
    request_id: str
    run_id: str
    parent_run_id: str | None = None
    thread_id: str
    user_id: UUID
    original_query: str
    workspace_ids: list[UUID]
    target_document_ids: list[UUID]
    attached_document_ids: list[UUID]
    can_read_people: bool
    deadline_at: datetime
```

Trusted fields are built by backend adapters and cannot be produced by model output.

### 6.3 Conversation and semantic context

```python
class ConversationContext(BaseModel):
    contract_version: Literal["2.0"]
    thread_id: str
    summary: str
    active_entities: list[ActiveEntity]
    last_focus: EntityReference | None
    open_questions: list[str]
    recent_turns: list[ConversationTurn]
    built_through_message_id: str | None

class SemanticContext(BaseModel):
    contract_version: Literal["2.0"]
    original_query: str
    contextualized_query: str
    normalized_query: str
    abbreviations: list[AbbreviationResolution]
    coreferences: list[CoreferenceResolution]
    document_refs: list[DocumentReference]
    person_refs: list[EntityReference]
    section_refs: list[SectionReference]
    blocking_ambiguities: list[BlockingAmbiguity]
```

Identifiers suggested by the semantic model are accepted only after reconciliation against deterministic candidates or database records. Exact deterministic matches take precedence. A conflict affecting target identity, scope, or objective triggers clarification.

### 6.4 Query analysis

```python
class QueryAnalysis(BaseModel):
    work_type: Literal[
        "lookup", "retrieve", "explain", "compare",
        "summarize", "cross_domain", "multi_goal",
    ]
    required_capabilities: list[str]
    dependencies: list[TaskDependency]
    semantic_complexity: Literal["simple", "compound", "deep"]
    requires_synthesis: bool
```

The small model describes the request. Deterministic routing code chooses fast, deep, or clarify and validates all named capabilities.

### 6.5 Agent request and result

```python
class AgentRequest(BaseModel):
    contract_version: Literal["2.0"]
    request_id: str
    task_id: str
    parent_task_id: str | None = None
    capability: str
    objective: str
    inputs: dict[str, Any]
    required_target_ids: list[UUID]
    discovered_document_refs: list[DocumentReference]
    depends_on: list[str]
    completion_criteria: list[str]

class AgentResult(BaseModel):
    contract_version: Literal["2.0"]
    request_id: str
    task_id: str
    status: Literal[
        "success", "partial", "not_found",
        "needs_input", "denied", "error",
    ]
    data: dict[str, Any] | None
    evidence: list[Evidence]
    coverage: Coverage
    missing: list[MissingRequirement]
    answer_mode: Literal["template", "synthesis_required"]
    error: AgentError | None
```

`AgentRequest` intentionally excludes `workspace_ids` and People permission. The capability boundary receives trusted runtime context separately.

Status semantics are strict:

- `success`: objective completed within the supplied runtime scope;
- `partial`: useful result exists but non-blocking requirements remain;
- `not_found`: lookup completed normally and found nothing;
- `needs_input`: user input is required;
- `denied`: permission gate rejected the operation;
- `error`: runtime or infrastructure failure.

Timeout and backend outage must never become `not_found`.

### 6.6 Coverage and evidence

```python
class Coverage(BaseModel):
    required: list[UUID]
    resolved: list[UUID]
    read: list[UUID]
    unreadable: list[UUID]
    missing: list[UUID]
    truncated: list[UUID]

class Evidence(BaseModel):
    evidence_id: str
    task_id: str
    source_type: Literal["document", "knowledge_graph", "people", "memory"]
    role: Literal["target", "discovered", "supporting"]
    document_id: UUID | None
    workspace_id: UUID | None
    section_path: str | None
    page_or_chunk: str | None
    content: str
    content_hash: str
    metadata: dict[str, Any]
    provenance: Provenance
```

All document evidence requires a verified `document_id`. Citation metadata must map to evidence and verified document metadata. Evidence preserves task and source provenance through fan-out and fan-in.

## 7. Context and persistence design

The chat database is the source of truth. LangGraph checkpoint state is a resumable execution snapshot, not the authoritative conversation record.

For each user message, persist a semantic snapshot containing:

- original, contextualized, and normalized query;
- resolved entities and their provenance;
- abbreviations and coreferences;
- ambiguities;
- contract, model, and configuration revisions.

For each thread, persist a rolling structured summary containing:

- concise text summary;
- active documents, people, sections, and files;
- last focus;
- open questions;
- `built_through_message_id`;
- schema version.

The Context subgraph reads the persisted context and recent turns. It reconstructs from history only when data is missing or incompatible. It runs deterministic extraction and a small-model semantic pass concurrently, reconciles them, then persists the resulting snapshot. Conversation references never grant permission; current `workspace_ids` remain authoritative.

## 8. Routing and fast paths

Deterministic routing applies the following priority:

1. Missing or ambiguous required target/objective: clarify.
2. Simple People lookup: People gate, capability, template response.
3. Document metadata/listing: Document subgraph, template response.
4. Exact resolved section retrieval: Section subgraph, verbatim/template response with citation.
5. Compare, cross-domain dependency, multi-goal, dependent research, or insufficient single-step evidence: DeepAgent.
6. Other bounded domain work: domain subgraph, followed by synthesis only if requested.

Exact section retrieval bypasses the main model only when the user requests retrieval. Explain, compare, evaluate, or summarize requests still require synthesis.

## 9. DeepAgent subgraph

DeepAgent uses a hybrid planning loop:

```text
initial plan
→ deterministic plan validation
→ dispatch ready tasks in parallel with Send
→ execute domain subgraphs
→ collect AgentResult and Evidence
→ deterministic evidence validation
→ small-model semantic evaluation
    ├── sufficient → synthesis
    ├── required target missing → clarification
    ├── evidence gap → bounded replan
    └── terminal failure → typed failure response
```

Plan rules:

- task IDs are unique;
- dependencies form an acyclic graph;
- capabilities must exist in the request-scoped registry;
- completed task records are immutable;
- replanning appends tasks and records a reason plus evidence/task dependencies;
- parallel branches receive minimal task state;
- runtime enforces maximum branches, tasks, tool calls, replan rounds, deadlines, and cancellation.

The main model creates the initial plan and bounded additions. Runtime validation, not the model, decides whether a plan can execute.

## 10. Evidence evaluation and grounding

Evidence evaluation has two layers.

### Hard deterministic checks

- required-target resolution and read coverage;
- provenance and content-hash integrity;
- citation-to-evidence mapping;
- task completion criteria;
- permission and capability status;
- error, timeout, and truncation semantics.

### Small-model semantic evaluation

The model returns structured judgments for:

- relevance to the objective;
- missing aspects;
- contradictions;
- semantic sufficiency for synthesis.

The small model cannot modify evidence or claim that unread material was read.

The Grounding subgraph validates every synthesized answer. Unsupported citations are rejected. An ungrounded answer receives at most one revision; if it remains invalid, the system returns a transparent insufficient-evidence response.

## 11. Blocking clarification and resume

An unreadable, unresolved, or ambiguous required target prevents synthesis. The system searches for candidate documents inside `workspace_ids` and emits:

```python
class ClarificationRequest(BaseModel):
    contract_version: Literal["2.0"]
    clarification_id: str
    reason: Literal[
        "target_not_found", "target_ambiguous",
        "target_unreadable", "semantic_ambiguity",
    ]
    question: str
    unresolved_targets: list[TargetReference]
    candidates: list[DocumentCandidate]
    resumable: bool
    expires_at: datetime
```

Candidate documents are suggestions only. They become required targets only after user confirmation.

The graph prefers LangGraph interrupt/resume. On every resume, the backend recalculates permissions. If the checkpoint expired, failed, or uses an incompatible contract version, the backend creates a replacement run linked with `parent_run_id` and imports only validated semantic and clarification data.

## 12. Testing

### Contract tests

- unsupported versions and extra fields are rejected;
- plan cycles and unknown capabilities are rejected;
- model output cannot inject trusted scope fields;
- citation must map to verified evidence;
- required-target coverage blocks synthesis when incomplete;
- all result statuses retain distinct semantics.

### Subgraph tests

Each subgraph is compiled and tested independently for:

- input/output validation;
- routing and terminal states;
- malformed model output;
- permission denial;
- timeout, retry, cancellation, and budget exhaustion;
- interrupt and resume behavior.

### End-to-end cases

1. Allowed and denied People lookup.
2. Document metadata/listing template response.
3. Exact section verbatim response with citation.
4. Resolved conversational follow-up.
5. Ambiguous follow-up clarification.
6. Multi-document comparison.
7. People-to-document dependency.
8. Missing target, candidate selection, interrupt, and resume.
9. Abundant but semantically irrelevant evidence.
10. Unsupported citation generated by the main model.
11. Workspace access changed before resume.
12. Expired checkpoint replaced by a linked run.

## 13. Observability

Record:

- graph, contract, model, and configuration versions;
- semantic reconciliation conflicts;
- routing decision and reason;
- initial plan and append-only replans;
- task status and latency;
- required-target coverage;
- People permission denials;
- evidence IDs and grounding failures;
- interrupt/resume lineage;
- token, task, branch, tool-call, and replan budgets.

Logs and traces redact or hash CCCD, BHXH, phone numbers, and other sensitive values.

## 14. Rollout

Use `NEXUSRAG_AGENT_GRAPH_VERSION=v1|v2` at an integration boundary outside both supervisor implementations.

1. Build and test v2 independently.
2. Replay curated and historical cases offline.
3. Shadow semantic analysis and routing without serving v2 output.
4. Canary v2 by user/cohort.
5. Compare correctness, target coverage, citation faithfulness, permission behavior, latency, and cost.
6. Increase v2 traffic gradually.
7. Set v2 as default after gates pass.
8. Remove v1 in a separate, explicitly reviewed change.

The v2 implementation must update `CLAUDE.md`, `.env.example`, relevant harness documentation, and this architecture document in the same change. Re-index GitNexus after structural changes.

## 15. Explicit non-goals for initial v2

- Refactoring `supervisor.py` in place.
- Reusing `SupervisorState` as the v2 business interface.
- Exposing database/vector/KG clients directly to models.
- Letting an LLM decide permissions or construct workspace scope.
- Adding every possible domain subgraph before a validated use case.
- Removing v1 before canary evidence supports cutover.

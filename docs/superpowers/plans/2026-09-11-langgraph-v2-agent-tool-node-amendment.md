# LangGraph v2 Agent / Tool / Node Architecture Amendment

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this amendment together with the Phase 2 and Phase 3 plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LangGraph v2 implementation ownership unambiguous: LangGraph nodes own workflow/state lifecycle, one complex-research agent may propose adaptive plans/replans, atomic domain work lives in typed capabilities, task know-how lives in skills/workflows, and every capability execution remains governed by the frozen `TaskPlan -> validation -> checkpoint -> scheduler` path.

**Architecture:** This amendment is normative for implementation ownership and package layout. It does **not** add or change frozen business-contract fields. It removes the old “one use-case = one agent/domain graph” model, makes Phase 2 node/capability based, and constrains Phase 3 agent-facing tools so they can propose work but can never bypass plan validation, checkpoint ownership, authorization, evidence governance, or grounding.

**Tech Stack:** Python 3.11, LangGraph Phase-0 winner, optional Deep Agents adapter if it wins Phase 0, Pydantic v2, request-scoped `CapabilityRegistry`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

**Applies to:**
- `docs/superpowers/plans/2026-09-11-langgraph-v2-phase2-fast-paths.md`
- `docs/superpowers/plans/2026-09-11-langgraph-v2-phase3-rollout.md`

## Global Constraints

- Frozen v2 contracts remain authoritative; this amendment changes implementation ownership, package layout, and internal runtime adapters only.
- **Agent** means a runtime component that may propose an initial `TaskPlan`, propose append-only replans, and decide whether more research is useful from validated observations. It does not directly execute capabilities.
- **Node** means a LangGraph orchestration/state-lifecycle step with a predetermined responsibility. Nodes may route, validate, checkpoint, evaluate, ground, interrupt/resume, synthesize, or invoke the scheduler.
- **Capability/tool** means one bounded typed domain operation. It accepts `AgentRequest` plus `CapabilityRuntimeContext`, cannot read arbitrary supervisor state, and returns validated `AgentResult`/evidence references.
- **Skill** means task strategy/know-how such as summarize, compare, legal analysis, or compliance. A skill owns no authorization, persistence, TaskPlan, EvidenceUse, or FinalResponse boundary.
- **Workflow/subgraph** means a predetermined multi-step algorithm, such as hierarchical large-document summarization. It may be callable through the governed execution boundary, but it is not an autonomous agent.
- Fast paths and complex research use the **same capability implementations** from one request-scoped registry. Duplicate People/document/KG business implementations are forbidden.
- Current authorization, workspace scope, People permission, deadlines, feature flags, service availability, and cancellation are runtime-only and cannot be supplied or widened by model-generated input.
- Do not create v2 `people_agent`, `summary_agent`, `comparison_agent`, `document_agent`, `section_agent`, `kg_agent`, `evaluation_agent`, `grounding_agent`, or equivalent wrappers under different filenames.
- Write remains outside this v2 rollout and stays v1-owned until a separate approved plan defines its semantics.

---

## 1. Normative Taxonomy

Use this decision rule before creating a v2 module:

```text
Does it adaptively propose what research to do next from validated observations?
├── yes -> complex-research agent/planner boundary only
└── no
    ├── does it own LangGraph state/routing/lifecycle? -> node
    ├── does it perform one bounded domain operation? -> capability
    ├── does it encode task strategy/instructions? -> skill
    └── does it run a fixed multi-step algorithm? -> workflow/subgraph
```

| Concern | Correct v2 role | Not allowed |
|---|---|---|
| People lookup | `people.lookup` capability | `people_agent` |
| Document search | `document.search` capability | generic `rag_agent` |
| Document/section read | `document.read` / `section.read` capability | `document_agent`, `section_agent` |
| KG lookup | `knowledge_graph.query` capability | `kg_agent` |
| Initial document identity resolution | Binding Resolver node/service | `resolve_doc_agent` |
| Query preprocessing/coreference/abbreviation normalization | Context/Semantic nodes + reusable services/capabilities | `semantic_agent` |
| Evidence sufficiency | Evaluator node | `evaluation_agent` |
| Grounding/citations | Grounding node | `grounding_agent` |
| Bounded summary | document/section read + synthesis | `summary_agent` |
| Comparison | complex-research skill/policy | `comparison_agent` |
| Compliance/legal evaluation | complex skill/policy + evaluator criteria | `compliance_agent` |
| Large map/reduce summary | deterministic workflow | autonomous summary agent |
| Multi-step adaptive research | complex-research agent | handoff chain of domain agents |

---

## 2. Hard Execution Invariant: Tools Never Bypass TaskPlan

The phrase **“agent chooses a tool”** means the agent proposes a `TaskSpec.capability` or append-only task through the governed planning boundary. It never means `LLM -> capability.execute()`.

All factual capability execution, fast or complex, must follow:

```text
Fast deterministic builder OR Complex research agent
                    |
                    v
          TaskPlan / appended TaskSpec proposal
                    |
                    v
      validate_task_plan / validate_replan
                    |
                    v
       checkpoint authoritative TaskPlan
                    |
                    v
              TaskScheduler
                    |
                    v
          CapabilityRegistry lookup
                    |
                    v
           Capability.execute(
               AgentRequest,
               CapabilityRuntimeContext,
           )
                    |
                    v
              AgentResult
                    |
                    v
      EvidenceRecord / EvidenceUse / Coverage
```

Rules:

1. `TaskPlan` is the authoritative executable intent for every factual invocation.
2. A task must exist in the validated checkpointed plan before the scheduler can execute it.
3. Only the scheduler may dispatch a capability.
4. Agent-facing tool adapters do not call capability implementations directly.
5. Deep Agents/native framework tool calls, if used, are interpreted as **task proposals** and routed through the same validator/checkpoint/scheduler path.
6. Unknown, unauthorized, stale, or unplanned tool/capability references fail closed.
7. Model-generated workspace IDs, ACLs, deadlines, service objects, EvidenceUse IDs, or authorization flags are never accepted as execution authority.

### Framework-neutral internal gateway

The selected orchestrator may expose an internal gateway such as:

```python
@dataclass(frozen=True)
class CapabilityInvocationProposal:
    capability: str
    objective: str
    input: CapabilityInput
    depends_on: tuple[str, ...] = ()

class AgentToolGateway(Protocol):
    async def invoke(
        self,
        proposal: CapabilityInvocationProposal,
        current_plan: TaskPlan,
        runtime: GraphRuntimeContext,
    ) -> "AgentToolObservation": ...
```

`AgentToolGateway.invoke()` performs exactly:

```text
proposal
-> convert to new TaskSpec
-> validate append-only plan/replan
-> checkpoint updated plan
-> scheduler executes ready tasks
-> validate AgentResult/EvidenceUse
-> project a safe AgentToolObservation
```

The gateway is orchestration infrastructure, not a business capability.

---

## 3. Model Observation Boundary

`CapabilityOutput` is **not** model-visible by default. Capability execution and model observation are separate boundaries.

```text
Capability
   |
   v
AgentResult + EvidenceUse
   |
   +---------------------> scheduler/materializer/evaluator sees governed result
   |
   v
ObservationProjector
   |
   v
AgentToolObservation
   |
   v
Complex research agent
```

Use an implementation-only projection such as:

```python
@dataclass(frozen=True)
class AgentToolObservation:
    task_id: str
    status: AgentStatus
    evidence_use_ids: tuple[UUID, ...]
    coverage: tuple[CoverageObservation, ...]
    result_kind: str
    safe_metadata: Mapping[str, str]
```

Rules:

- No raw connector/database payload is returned to the model.
- No runtime secrets, service clients, ACL/workspace authority, deadlines, storage keys, or encryption metadata are model-visible.
- Retrieved document text remains untrusted data and is not automatically returned to the planner. The planner primarily receives task status, coverage/evaluation gaps, candidate metadata explicitly admitted by policy, and EvidenceUse identities.
- Full evidence content is hydrated only by governed evaluator/synthesis/grounding paths.
- People and other sensitive capabilities use stricter projection: raw CCCD, DOB, addresses, phone/email, personnel records, or unrelated fields never enter planner observation or checkpoint.
- People -> Document scalar transfer remains a deterministic governed dependency materializer, not a planner observation.
- Memory/personal-data capabilities follow the same minimization rule.

Required security test:

```text
people.lookup returns a governed People EvidenceUse
-> planner observation contains status/use IDs only
-> CCCD required by DocumentSearchInput is materialized server-side
-> raw People record never enters planner prompt/checkpoint
```

---

## 4. Canonical Package Ownership

Phase 2 and Phase 3 must use one physical layout:

```text
backend/app/services/agents/v2/
├── nodes/
│   ├── context.py
│   ├── binding.py
│   ├── routing.py
│   ├── fast_plan.py
│   ├── execute.py
│   ├── evaluate.py
│   ├── synthesize.py
│   ├── grounding.py
│   ├── clarification.py
│   └── finalizer.py
│
├── capabilities/
│   ├── people.py
│   ├── document.py
│   ├── section.py
│   ├── knowledge_graph.py
│   ├── abbreviation.py
│   └── memory.py
│
├── adapters/                  # server-internal typed ports to v1 services; never model/tool visible
│   ├── semantic.py
│   ├── conversation.py
│   ├── document.py
│   └── deep_research.py
│
├── tools/                     # agent-facing governed gateway (Phase 3 only)
│   ├── adapters.py
│   ├── gateway.py
│   └── observations.py
│
├── execution/
│   ├── __init__.py
│   └── scheduler.py
│
├── dependencies/
│   └── people_document.py
│
├── skills/                    # framework-neutral policy; native skill files may mirror after Phase 0
│   ├── summarize/policy.py
│   ├── compare/policy.py
│   ├── legal_analysis/policy.py
│   └── compliance/policy.py
│
├── replanning.py
├── discovery.py
└── complex_research_graph.py
```

Framework-specific skill representation is selected only after Phase 0; `skills/<name>/policy.py` is the framework-neutral source of truth and, if Deep Agents wins, native skill files may mirror it inside the same directory. `adapters/` is the server-internal port layer to legacy v1 services and is distinct from `tools/`: `tools/adapters.py` is agent-facing and must never call `capability.execute(...)` directly, while `adapters/` is never exposed to the model or the tool catalog. Phase 1 owns `adapters/` and `capabilities/__init__.py`; Phase 2 adds capability implementations and must not create a second adapter package. Business ownership remains unchanged.

### Old -> canonical path migration

The following Phase-2 names are retired:

```text
context_graph.py                 -> nodes/context.py
binding_graph.py                 -> nodes/binding.py
routing_graph.py                 -> nodes/routing.py
planning.py                      -> nodes/fast_plan.py
evaluation.py                    -> nodes/evaluate.py
grounding.py                     -> nodes/grounding.py
clarification.py                 -> nodes/clarification.py
domain/people_graph.py           -> capabilities/people.py
domain/document_graph.py         -> capabilities/document.py
domain/section_graph.py          -> capabilities/section.py
domain/knowledge_graph.py        -> capabilities/knowledge_graph.py
```

No implementation task may create both sides of one mapping.

---

## 5. Phase 2 Semantics: Nodes + Shared Capabilities

Fast routing remains deterministic:

```text
simple People lookup                    -> fast_domain -> people.lookup
exact document metadata/read            -> fast_domain -> document.*
exact section retrieval                 -> fast_domain -> section.read
simple KG lookup                        -> fast_domain -> knowledge_graph.query
bounded one-document summary            -> fast_domain -> document.read -> evaluate -> synthesize -> ground
comparison / cross-document summary     -> complex_research
cross-domain dependency                  -> complex_research
compliance / multi-goal / iterative RAG -> complex_research
```

Fast path:

```text
semantic/routing nodes
-> deterministic TaskPlan
-> checkpoint
-> execute node
-> shared scheduler
-> shared capability
-> evaluator
-> synthesis when needed
-> grounding
-> finalizer
```

There are no independently reasoning People/Document/Section/KG subgraphs.

Required Phase-2 tests:

```text
test_v2_has_no_domain_agent_or_domain_graph_wrappers
test_fast_people_uses_shared_capability_registry
test_fast_document_read_uses_shared_capability_registry
test_capability_receives_capability_runtime_context_only
test_capability_cannot_read_supervisor_root_state
test_model_input_cannot_supply_workspace_or_acl
test_bounded_summary_is_read_evaluate_synthesize_ground
test_compare_never_routes_to_fast_domain
test_task_must_be_checkpointed_before_capability_dispatch
```

---

## 6. Phase 3 Semantics: One Adaptive Complex-Research Agent

`ComplexResearchGraph` is the single adaptive planning/replanning boundary. The outer LangGraph graph, validator, scheduler, evaluator, and grounding remain authoritative.

Canonical research loop:

```text
ResearchPlanningInput
        |
        v
Complex research agent proposes initial TaskPlan
        |
        v
validate_task_plan
        |
        v
checkpoint plan
        |
        v
scheduler executes ready tasks
        |
        v
EvidenceEvaluation
        |
        +---- sufficient/terminal ----> synthesis -> grounding -> finalizer
        |
        v
ResearchPlanningInput(current_plan, task_outcomes,
                      prior_evidence_uses, prior_evaluation)
        |
        v
agent proposes append-only replan
        |
        v
validate_replan
        |
        v
checkpoint updated plan
        |
        +------------------------------> scheduler
```

The agent owns only:

```text
initial plan proposal
append-only replan proposal
research stop/continue recommendation
skill selection/strategy
```

It does **not** own:

```text
authorization
binding truth
revision identity
capability execution
EvidenceUse creation
coverage truth
evidence sufficiency
contradiction authority
retention
citation grounding
FinalResponse success
```

### Compare

```text
compare skill
-> planner proposes reads for both sides
-> scheduler executes shared document/section capabilities
-> evaluator verifies both target coverages
-> synthesizer creates use-bound claims
-> grounding renders citations
```

No `comparison_agent` exists.

### Summary

```text
summarize skill
-> bounded case stays fast
-> large/iterative case uses complex planner
-> deterministic map/reduce workflow may be proposed as governed work
-> evaluator owns coverage
-> synthesis/grounding remain outside agent authority
```

No `summary_agent` exists.

### People -> Document

`depends_on` expresses ordering only. The planner never sees raw People output.

```text
T1 people.lookup
-> governed People EvidenceUse
-> PeopleDocumentDependencyAdapter
-> current ACL/expiry/minimization check
-> exact allowed scalar
-> concrete DocumentSearchInput
-> T2 scheduler dispatch
```

No agent-to-agent handoff exists.

Required Phase-3 tests:

```text
test_complex_agent_uses_request_scoped_tool_catalog
test_fast_and_complex_share_same_capability_instance_or_factory
test_agent_tool_call_creates_validated_task_before_dispatch
test_tool_adapter_cannot_call_capability_directly
test_compare_is_skill_not_agent_route
test_summary_is_skill_not_agent_route
test_people_document_dependency_is_not_agent_handoff
test_people_observation_does_not_expose_raw_record
test_unknown_or_unauthorized_tool_is_rejected_at_execution
test_replan_can_add_tasks_but_cannot_widen_authorization
test_planner_never_receives_runtime_secrets
```

---

## 7. Capability Runtime Contract

Capabilities receive only the narrow runtime contract:

```python
class Capability(Protocol):
    descriptor: CapabilityDescriptor

    async def execute(
        self,
        request: AgentRequest,
        runtime: CapabilityRuntimeContext,
    ) -> AgentResult: ...
```

`CapabilityDescriptor`, `CapabilityInput`, `CapabilityOutput`, and the runtime-only `CapabilityRuntimeContext` are **frozen contract types owned by the contracts layer** (`contracts/capability.py`). This amendment and every plan import them; none may redefine, rename, or add fields to them. The capability package owns only the `Capability` protocol and the request-scoped registry, which is runtime/config metadata rather than a new contract.

`GraphRuntimeContext` belongs to nodes/scheduler/orchestration. A capability may not receive it as a substitute for `CapabilityRuntimeContext`.

Agent/model-visible tool schema contains only allowed `CapabilityInput` fields. Runtime fields are injected server-side and cannot appear as model-supplied parameters.

---

## 8. Evaluator Ownership

Evaluator is a node-owned authority, not an agent. “Node-owned” does not require every judgment to be purely algorithmic.

```text
Deterministic rules own:
- coverage/read-complete semantics
- revision/locator compatibility
- ACL/expiry/source validity
- criterion presence
- status precedence

Bounded model judgment MAY assist:
- semantic criterion evaluation
- contradiction interpretation
- legal semantic comparison
```

Any model used inside evaluator:

- receives only governed evidence;
- cannot plan or call tools;
- cannot change authorization/bindings/revisions;
- returns a typed proposal validated by evaluator rules;
- cannot self-certify final factual success outside `EvidenceEvaluation`.

---

## 9. Subagent Rule

Subagents are optional advisory/context-isolation helpers only. They are not domain agents and have no execution authority.

A subagent MAY:

```text
analyze already-authorized bounded context
return structured recommendation
propose strategy to the parent complex agent
perform isolated reasoning over sanitized inputs
```

A subagent MUST NOT:

```text
own or checkpoint TaskPlan
append authoritative TaskSpec directly
execute a Capability
bind/promote documents
widen the tool catalog or ACL
create EvidenceUse
read raw People records/runtime secrets
decide authoritative sufficiency
emit FinalResponse
```

The parent complex-research boundary remains the only adaptive planning owner.

---

## 10. Existing AIRAG -> v2 Ownership Map

| Existing module/concept | v2 disposition |
|---|---|
| `services/agents/people_agent.py` | People capability/service + execute node; no v2 People agent |
| `services/agents/rag_agent.py` | `document.search`, `document.read`, `section.read`, KG capabilities |
| `services/agents/resolve_doc_agent.py` | Binding Resolver node/service |
| `services/agents/write_agent.py` | Remains v1-owned/out of this rollout |
| `semantic_preprocessor.py` | Context/semantic nodes + reusable normalization services |
| result evaluator behavior | evaluator node, deterministic rules + bounded typed judgment |
| `react_executor` behavior | consolidated into the one complex-research planning boundary |
| proposed summary/comparison agents | do not create; skills over shared capabilities |
| `supervisor_v2.py` | graph composition only; no domain business implementation |

---

## 11. Execution Order and Plan Authority

Apply this amendment after Phase 0 semantics are known and Phase 1 foundation passes, before Phase 2 implementation starts:

```text
Phase 0 benchmark/winner
        |
Phase 1 frozen foundation
        |
THIS AMENDMENT
        |
Phase 2 nodes + capabilities
        |
Phase 3 complex agent + governed tool gateway + skills
```

Phase 2 and Phase 3 plans must directly use the package paths and execution invariants in this amendment. Do not rely on a worker mentally merging contradictory documents.

If an older plan statement conflicts with this amendment on **agent/node/capability/skill ownership, package path, or tool execution path**, the synchronized Phase 2/3 plan text is the source of truth. Frozen business contracts remain higher authority than all plans.

---

## 12. Amendment Acceptance Gate

Before Phase 2/3 implementation is considered synchronized, prove:

| # | Proof |
|---|---|
| 1 | No v2 People/Summary/Comparison/Document/Section/KG domain agent or domain graph wrapper exists. |
| 2 | Fast and complex paths resolve the same capability implementation from the request-scoped registry. |
| 3 | Every factual capability dispatch references a validated checkpointed TaskSpec. |
| 4 | Agent-facing tool adapters cannot directly invoke capabilities. |
| 5 | Capabilities receive `CapabilityRuntimeContext`, never arbitrary root/graph state. |
| 6 | Model-generated input cannot widen workspace, People permission, ACL, deadline, feature/service authority, or cancellation state. |
| 7 | Sensitive capability outputs use explicit observation projections; raw People records never enter planner/checkpoint. |
| 8 | Bounded summary is read -> evaluate -> synthesize -> ground, not summary-agent dispatch. |
| 9 | Comparison routes to one complex planner and shared document/section capabilities. |
| 10 | People -> Document remains deterministic scheduler/materializer execution. |
| 11 | Replan is append-only and cannot widen authorization. |
| 12 | Skills/subagents cannot bypass TaskPlan validation, evaluator, evidence governance, or grounding. |
| 13 | `CapabilityDescriptor`/`CapabilityInput`/`CapabilityOutput`/`CapabilityRuntimeContext` are imported from frozen contracts and never redefined or field-extended outside `contracts/capability.py`. |

Static architecture guards:

```bash
set -e
! find backend/app/services/agents/v2 -type f \
  \( -name 'people_agent.py' -o -name 'summary_agent.py' -o -name 'comparison_agent.py' \
     -o -name 'document_agent.py' -o -name 'section_agent.py' -o -name 'kg_agent.py' \
     -o -path '*/domain/*_graph.py' \) | grep .

! rg -n \
  'class\s+(People|Summary|Comparison|Document|Section|KG|Evaluation|Grounding).*Agent|\b(People|Summary|Comparison|Document|Section|KG)Agent\b' \
  backend/app/services/agents/v2

! rg -n 'capability\.execute\(' backend/app/services/agents/v2/tools

! rg -n 'class\s+Capability(Descriptor|Input|Output|RuntimeContext)' \
  backend/app/services/agents/v2 --glob '!**/contracts/**'
```

Focused validation:

```bash
cd backend
pytest tests/agents/v2/fast_paths tests/agents/v2/complex -q
```

Expected: taxonomy, plan-ownership, sensitive-observation, contract/evidence, and shared-capability tests pass with v1 still the production default.

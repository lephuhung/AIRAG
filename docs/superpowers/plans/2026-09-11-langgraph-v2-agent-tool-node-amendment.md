# LangGraph v2 Agent / Tool / Node Architecture Amendment

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this amendment together with the Phase 2 and Phase 3 plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ambiguous “one use-case = one agent” ownership from LangGraph v2 and make the execution model explicit: dynamic planning belongs to the complex-research agent, deterministic orchestration belongs to LangGraph nodes, atomic domain operations belong to typed capabilities/tools, and task-specific know-how such as summarize/compare/compliance belongs to skills or deterministic workflows.

**Architecture:** This is a normative implementation amendment to the existing frozen v2 contracts. It does **not** add or change business-contract fields. It changes implementation ownership and naming so fast paths and complex research share the same capability implementations, while only complex research may dynamically plan/replan or select tools.

**Tech Stack:** Python 3.11, LangGraph Phase-0 winner, optional Deep Agents adapter if it wins Phase 0, Pydantic v2, request-scoped `CapabilityRegistry`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- The frozen v2 contracts remain authoritative; this amendment changes implementation boundaries and terminology only.
- **Agent** means a runtime component that can dynamically create/update a plan and choose the next tool/capability from observations. An LLM call alone does not make a component an agent.
- **Node** means a LangGraph orchestration/state-lifecycle step with a predetermined responsibility. Nodes may route, validate, checkpoint, evaluate, ground, interrupt/resume, or invoke a previously selected capability; they do not own free-form domain planning.
- **Capability/tool** means an atomic typed domain operation. It accepts typed input plus trusted runtime context, does not read arbitrary supervisor root state, and returns a minimized typed result/evidence reference.
- **Skill** means task strategy/know-how such as summarize, compare, legal analysis, compliance evaluation, or drafting. A skill is not an independently routed agent and owns no authorization or persistence boundary.
- **Workflow/subgraph** means a fixed multi-step algorithm used when deterministic execution is preferable (for example hierarchical large-document summarization). It may be exposed to the complex agent as one tool, but it does not gain an agent identity.
- Fast paths invoke capabilities directly through the shared registry/scheduler. Complex research sees agent-facing tool adapters over the **same** capabilities. No duplicate business implementation is allowed.
- Authorization, workspace scope, People permission, deadlines, feature flags, and service availability are injected by runtime. A model/tool call cannot supply or widen them.
- Do not create v2 `people_agent`, `summary_agent`, `comparison_agent`, `document_agent`, `section_agent`, `kg_agent`, or equivalent domain-agent wrappers.

---

## 1. Normative Taxonomy

Use this decision rule before creating any v2 module:

```text
Does the component dynamically decide what to do next from observations?
├── yes → agent/planner boundary (complex research only)
└── no
    ├── does it mutate/read LangGraph execution state or control routing? → node
    ├── does it perform one bounded domain operation? → capability/tool
    ├── does it encode task strategy/instructions? → skill
    └── does it run a predetermined multi-step algorithm? → workflow/subgraph
```

Examples:

| Concern | Correct v2 role | Not allowed |
|---|---|---|
| People lookup | `people.lookup` capability/tool | `people_agent` |
| Document search | `document.search` capability/tool | generic `rag_agent` |
| Document/section read | `document.read` / `section.read` capability/tool | `document_agent`, `section_agent` |
| Initial document identity resolution | Binding Resolver node/service | `resolve_doc_agent` |
| Query preprocessing/coreference/abbreviation normalization | Context/Semantic nodes + reusable capabilities | `semantic_agent` |
| Evidence sufficiency | Evaluator node | `evaluation_agent` |
| Grounding/citations | Grounding node | `grounding_agent` |
| Summarize | work type + skill; bounded read may use fast path | `summary_agent` |
| Compare | complex-research skill/policy | `comparison_agent` |
| Compliance/legal evaluation | complex-research skill/policy + evaluator criteria | `compliance_agent` |
| Large deterministic map-reduce summary | workflow exposed as a tool when needed | autonomous summary agent |
| Multi-step adaptive research | complex-research agent | handoff chain of domain agents |

---

## 2. Existing AIRAG → v2 Ownership Map

The current v1 modules remain untouched until v2 cutover, but v2 must not reproduce their naming model.

| Existing concept/module | v2 disposition |
|---|---|
| `services/agents/people_agent.py` | Split into People capability/service plus deterministic execute node. Permission enforcement remains at capability boundary. |
| `services/agents/rag_agent.py` | Split into `document.search`, `document.read`, `section.read`, and KG/retrieval capabilities. No generic RAG agent. |
| `services/agents/resolve_doc_agent.py` | Move ownership to Binding Resolver node/service. Discovery-time resolution may be exposed as a bounded tool, but initial binding remains deterministic graph ownership. |
| `services/agents/write_agent.py` | Remains v1-owned and out of this v2 rollout. A future v2 Write design must choose capability/workflow ownership, not recreate a domain agent by default. |
| `semantic_preprocessor.py` | Context/semantic nodes plus reusable abbreviation/entity capabilities. |
| `result_evaluator` behavior | Deterministic evaluator node. |
| `react_executor` behavior | Replaced by/consolidated into the single complex-research agent/planner boundary. |
| proposed `summary_agent` | Do not create. Summary is a work type/skill over document-read tools and synthesis. |
| proposed `comparison_agent` | Do not create. Comparison is a skill executed by the complex-research agent over multiple evidence-producing tools. |
| `supervisor_v2.py` | Composition/orchestration graph only; it is not a domain agent and contains no business tool implementation. |

---

## 3. Target Package Ownership

The frozen contracts do not require a specific Python package layout, but implementation should converge on the following ownership model:

```text
backend/app/services/agents/v2/
├── nodes/
│   ├── context.py
│   ├── binding.py
│   ├── routing.py
│   ├── fast_plan.py
│   ├── execute.py
│   ├── evaluate.py
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
├── tools/
│   └── adapters.py          # agent-facing adapters over CapabilityRegistry; no duplicated business logic
│
├── execution/
│   └── scheduler.py
│
├── skills/
│   ├── summarize.*
│   ├── compare.*
│   ├── legal_analysis.*
│   └── compliance.*
│
└── complex_research_graph.py   # the only adaptive planning/tool-selection boundary
```

The physical representation of `skills/` follows the Phase-0 winner. If Deep Agents wins, these may become native `SKILL.md`/skill packages. If native LangGraph wins, the same strategy contracts are loaded by the planner policy. Do not hard-code framework-specific skill loading before Phase 0 selects the orchestrator.

`tools/adapters.py` is intentionally thin. It converts the request-scoped `CapabilityRegistry` entries into whatever tool interface the Phase-0 winner requires. It must never reimplement People/document/KG logic.

---

## 4. Phase 2 Amendment — Fast Paths Are Nodes + Capabilities

This section supersedes any Phase-2 wording that calls People, Document, Section, or KG an “agent” or an independently reasoning “domain graph”.

### Phase-2 Task 1 routing rule

Keep deterministic-first routing, with these clarifications:

```text
simple People lookup                    → fast_domain → people.lookup
exact document metadata/read            → fast_domain → document.*
exact section retrieval                 → fast_domain → section.read
simple KG lookup                        → fast_domain → knowledge_graph.query
bounded one-document summary            → fast_domain → document.read → synthesis
comparison / cross-document summary     → complex_research
cross-domain dependency                  → complex_research
compliance / multi-goal / iterative RAG → complex_research
```

`summary` and `compare` remain `WorkType`s. They are not converted into agent route names.

### Phase-2 Task 3 path replacement

The following planned create paths are superseded:

```text
REMOVE FROM PLAN                         REPLACE WITH
v2/domain/people_graph.py               v2/capabilities/people.py
v2/domain/document_graph.py             v2/capabilities/document.py
v2/domain/section_graph.py              v2/capabilities/section.py
v2/domain/knowledge_graph.py            v2/capabilities/knowledge_graph.py
                                         v2/nodes/execute.py
```

`test_domain_paths.py` remains valid but should assert capability behavior and scheduler/node integration rather than autonomous subgraph behavior.

The generic execute node performs only orchestration:

```python
async def execute_node(state: SupervisorV2State, runtime: GraphRuntimeContext) -> dict:
    plan = require_checkpointed_plan(state)
    results = await execute_ready_tasks(
        plan=plan,
        results=state["execution"].results,
        registry=runtime.services.capability_registry,
        runtime=runtime,
    )
    return execution_update(results)
```

Domain behavior remains inside capability implementations selected by `TaskSpec.capability`.

### Required Phase-2 tests

Add or retain named tests proving:

```text
test_v2_has_no_people_summary_comparison_domain_agents
test_fast_people_uses_shared_capability_registry
test_fast_document_read_uses_shared_capability_registry
test_capability_cannot_read_supervisor_root_state
test_model_input_cannot_supply_workspace_or_acl
test_bounded_summary_is_read_plus_synthesis_not_summary_agent
test_compare_never_routes_to_fast_domain_agent
```

---

## 5. Phase 3 Amendment — One Adaptive Complex-Research Agent

`ComplexResearchGraph` is the single v2 boundary allowed to dynamically plan/replan and choose tools. Phase 0 still determines whether its concrete implementation is Deep Agents or native LangGraph; the behavior contract remains the same.

For a comparison request:

```text
Goal: compare A with B
    ↓
complex-research agent
    ├── resolve/check bindings as required
    ├── plan reads for A
    ├── plan reads for B
    ├── execute document/section tools
    ├── inspect evidence/coverage
    ├── bounded replan if permitted
    └── synthesize grounded comparison
```

There is no `comparison_agent` handoff.

For a complex summary request:

```text
Goal: summarize document(s) under requested focus
    ↓
complex-research agent + summarize skill
    ├── inspect structure
    ├── choose bounded target units
    ├── read required sections
    ├── call deterministic large-summary workflow when justified
    ├── evaluate coverage
    └── synthesize grounded summary
```

There is no `summary_agent` handoff.

People→Document stays a scheduler dependency materialization problem, not an agent-to-agent handoff. The complex planner can create T1/T2 dependencies, but the scheduler and governed adapter materialize only allowed typed values.

Subagents are permitted only when the selected orchestrator needs **context isolation, bounded parallel research, or specialist reasoning with a strict input/output contract**. Do not create one subagent per domain merely to mirror tools.

### Agent-facing tool catalog

At each complex-research run:

```text
base capability catalog
∩ current runtime permissions
∩ feature flags
∩ service availability
= request-scoped tool catalog
```

If `can_read_people=False`, no People tool is exposed to the planner and execution remains fail-closed if a stale plan references it.

### Required Phase-3 tests

Add or retain named tests proving:

```text
test_complex_agent_uses_request_scoped_tool_catalog
test_fast_and_complex_paths_share_same_capability_implementation
test_compare_is_skill_not_subagent_route
test_summary_is_skill_not_subagent_route
test_people_document_dependency_is_not_agent_handoff
test_unknown_or_unauthorized_tool_is_rejected_at_execution
test_replan_can_add_tasks_but_cannot_widen_authorization
test_subagent_cannot_receive_raw_people_record_or_runtime_secrets
```

---

## 6. Capability / Tool Contract Rules

Every atomic capability/tool must satisfy all of the following:

```text
1. Typed domain input from CapabilityInput union.
2. Trusted authorization/scope from CapabilityRuntimeContext only.
3. No arbitrary SupervisorV2State access.
4. No raw connector/database payload in checkpoint state.
5. Minimized typed CapabilityOutput or EvidenceUse references.
6. Stable AgentResult.task_id association.
7. Cancellation/deadline propagation.
8. Same implementation callable from deterministic fast path and complex-agent tool adapter.
```

The agent-facing adapter may add schema/description metadata for the selected framework, but must not alter authorization, scope, evidence, or business behavior.

---

## 7. Summary / Compare / Compliance Are Task Strategies

These are semantic goals, not runtime identities.

### Summarize

- Bounded one-document/one-section summary may remain a deterministic fast route: read evidence once, evaluate coverage, synthesize.
- Large, multi-document, focused, or iterative summary routes to complex research.
- A large-document map/reduce implementation is a deterministic workflow. The complex agent may invoke it as one tool when policy allows.

### Compare

- Always complex when two or more independently evidenced targets must be aligned.
- The skill defines comparison procedure and output expectations; document/section capabilities acquire evidence.
- Evaluator verifies required coverage for every comparison side before synthesis.

### Compliance / legal evaluation

- Complex research owns planning and evidence acquisition.
- `SemanticCriterion` may encode validated target-level judgment requirements.
- The evaluator owns sufficiency/contradiction; the agent does not self-certify evidence completeness.

---

## 8. Execution Order

This amendment is applied **after Phase 0 selection semantics are known and before Phase 2 implementation begins**. It does not require a new migration or contract version.

```text
Phase 0 benchmark/winner
        ↓
Phase 1 frozen foundation
        ↓
THIS AMENDMENT
        ↓
Phase 2 nodes + shared capabilities
        ↓
Phase 3 single complex-research agent + tools/skills
```

If Phase 2 implementation has already begun when this amendment is applied, stop before creating any `v2/domain/*_graph.py` domain-agent wrappers and migrate the uncommitted work to the capability/node ownership above.

---

## 9. Amendment Acceptance Gate

Before Phase 2 is considered ready for complex pilots, prove:

| # | Proof |
|---|---|
| 1 | No v2 People/Summary/Comparison/Document/Section/KG domain agent exists. |
| 2 | Fast and complex paths resolve the same capability implementation from the registry. |
| 3 | Supervisor nodes own graph state/routing only; capabilities do not read arbitrary root state. |
| 4 | Model-generated tool calls cannot provide/widen workspace IDs, People permission, ACL, deadlines, or service clients. |
| 5 | Bounded summary is document/section evidence acquisition plus synthesis, not a summary-agent dispatch. |
| 6 | Comparison routes to the complex-research planner and uses document/section tools. |
| 7 | People→Document dependency is scheduler/materializer execution, not agent handoff. |
| 8 | The complex-research agent receives only the request-scoped authorized tool catalog. |
| 9 | Skills contain task strategy only and cannot bypass TaskPlan validation, evaluator, evidence governance, or grounding. |

Suggested static guard:

```bash
! find backend/app/services/agents/v2 -type f \
  \( -name 'people_agent.py' -o -name 'summary_agent.py' -o -name 'comparison_agent.py' \
     -o -name 'document_agent.py' -o -name 'section_agent.py' -o -name 'kg_agent.py' \) | grep .
```

Suggested focused validation after implementation:

```bash
cd backend
pytest tests/agents/v2/fast_paths tests/agents/v2/complex -q
```

Expected: taxonomy tests and existing v2 contract/evidence tests pass with v1 still the production default.

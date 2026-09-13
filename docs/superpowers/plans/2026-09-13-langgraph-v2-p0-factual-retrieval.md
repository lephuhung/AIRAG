# LangGraph v2 P0 Factual Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make reference-free and hard-document-scoped factual v2 queries execute revision-aware retrieval and return grounded citations.

**Architecture:** Add an additive `document.retrieve` contract and capability. Project ACL-filtered `api_explicit` resources into deterministic semantic targets, create a validated/checkpointed retrieve plan, execute only through `TaskScheduler`, and persist retrieved chunks as governed evidence.

**Tech Stack:** Python 3.11, Pydantic v2, LangGraph, async SQLAlchemy, HTTP embed/rerank service, pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-13-langgraph-v2-factual-retrieval-reindex-design.md`

## Global Constraints

- `document_ids` are a hard scope after API ACL filtering.
- Preserve `proposal → validate → lease → checkpoint → TaskScheduler → capability`.
- No second scheduler and no direct capability execution from tools/planner/nodes.
- Raw retrieved content lives only in Evidence records, never capability output or planner observations.
- Every accepted source must match current authorization and an immutable published revision.
- Existing contract variants/checkpoints remain readable.
- v1 remains the default rollback arm; do not restart vLLM engines.
- Run GitNexus impact before every symbol edit and `detect-changes` before every commit.

---

### Task 1: Add the additive retrieval contract

**Files:**
- Modify: `backend/app/services/agents/v2/contracts/capability.py`
- Modify: `backend/app/services/agents/v2/contracts/validation.py`
- Modify: `backend/tests/agents/v2/contracts/test_capability_contracts.py`
- Modify: `backend/tests/agents/v2/orchestrator_compat/test_contract_parity.py`

**Interfaces:**
- Produces: `DocumentRetrieveInput`, `DocumentRetrieveOutput`, additive union membership.

- [ ] **Step 1: Write failing contract tests**

```python
def test_document_retrieve_contract_is_strict_and_bounded():
    value = DocumentRetrieveInput(kind="document.retrieve", query="lan bmnn", target_ids=("t1",), top_k=8)
    assert value.target_ids == ("t1",)
    with pytest.raises(ValidationError):
        DocumentRetrieveInput(kind="document.retrieve", query="x", top_k=21)


def test_old_capability_payloads_still_round_trip():
    payload = DocumentReadInput(kind="document.read", target_ids=("t1",)).model_dump_json()
    assert DocumentReadInput.model_validate_json(payload).kind == "document.read"
```

- [ ] **Step 2: Run RED**

Run: `cd backend && PYTHONPATH=. pytest tests/agents/v2/contracts/test_capability_contracts.py tests/agents/v2/orchestrator_compat/test_contract_parity.py -q`

Expected: FAIL because retrieval variants do not exist.

- [ ] **Step 3: Add minimal strict models and union members**

```python
class DocumentRetrieveInput(ContractModel):
    kind: Literal["document.retrieve"]
    query: str = Field(min_length=1)
    target_ids: tuple[str, ...] = ()
    top_k: int = Field(default=8, ge=1, le=20)

class DocumentRetrieveOutput(ContractModel):
    kind: Literal["document.retrieve"]
    retrieved_unit_count: int = Field(ge=0)
```

Add both to `CapabilityInput`/`CapabilityOutput`; extend validators without weakening existing cases.

- [ ] **Step 4: Run GREEN and regressions**

Run: `cd backend && PYTHONPATH=. pytest tests/agents/v2/contracts tests/agents/v2/orchestrator_compat -q`

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/agents/v2/contracts backend/tests/agents/v2/contracts backend/tests/agents/v2/orchestrator_compat
git commit -m "feat(v2): add document retrieval contract"
```

### Task 2: Implement revision-aware retrieval capability

**Files:**
- Modify: `backend/app/services/agents/v2/capabilities/document.py`
- Modify: `backend/app/services/agents/v2/capabilities/__init__.py`
- Modify: `backend/app/services/agents/v2/tools/observations.py`
- Modify: `backend/tests/agents/v2/fast_paths/test_capabilities.py`
- Modify: `backend/tests/agents/v2/complex/test_tool_gateway.py`

**Interfaces:**
- Consumes: `DocumentRetrieveInput` and `PinnedTargetResolver`.
- Produces: `DocumentRetrieveCapability`, `RevisionRetrievedChunk`, `DocumentRetrievalService.retrieve(...)`.

- [ ] **Step 1: Write failing capability tests**

Add tests proving: scoped target resolution; outside-scope/revision-mismatched chunks are dropped; missing targets are denied; accepted chunks produce EvidenceUses and count-only output; unknown result observation fails closed until an explicit count-only projector exists.

```python
result = await capability.execute(request, runtime)
assert result.status == "success"
assert result.data.retrieved_unit_count == 1
assert len(result.evidence_uses) == 1
assert "secret chunk" not in result.model_dump_json()
```

- [ ] **Step 2: Run RED**

Run: `cd backend && PYTHONPATH=. pytest tests/agents/v2/fast_paths/test_capabilities.py tests/agents/v2/complex/test_tool_gateway.py -q`

- [ ] **Step 3: Implement typed retrieval boundary**

```python
@dataclass(frozen=True)
class RevisionRetrievedChunk:
    document_id: UUID
    document_revision: str
    locator: ContentLocator
    content: str
    score: float
    target_id: str | None = None

class DocumentRetrievalService(Protocol):
    async def retrieve(self, query: str, *, top_k: int, allowed_targets: tuple[ResolvedTarget, ...], workspace_ids: tuple[UUID, ...]) -> Sequence[RevisionRetrievedChunk]: ...
```

`DocumentRetrieveCapability.execute()` must resolve non-empty target IDs, call the service, reject mismatches, persist each accepted chunk with `DocumentSourceIdentity`, emit target-bound coverage-purpose EvidenceUses only for matched explicit targets, and return `not_found` when none survive. It deliberately emits no read-only `CoverageObservation`; Task 4 owns evaluator integration for these uses.

- [ ] **Step 4: Add count-only observation and run GREEN**

```python
class DocumentRetrieveObservation(ContractModel):
    kind: Literal["document.retrieve"] = "document.retrieve"
    retrieved_unit_count: int
```

Run the focused tests plus `tests/agents/v2/fast_paths/test_scheduler.py`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/agents/v2/capabilities backend/app/services/agents/v2/tools/observations.py backend/tests/agents/v2
git commit -m "feat(v2): add revision-aware retrieval capability"
```

### Task 3: Project API-explicit documents as hard-scoped targets

**Files:**
- Modify: `backend/app/services/agents/supervisor_v2.py`
- Modify: `backend/app/services/agents/v2/nodes/routing.py`
- Modify: `backend/tests/agents/v2/test_context_binding_routes.py`
- Modify: `backend/tests/agents/v2/test_supervisor_v2.py`

**Interfaces:**
- Produces: deterministic `api_explicit:<resource_id>` references and scoped complex routing.

- [ ] **Step 1: Write failing tests**

Prove one/multiple API-explicit resources are projected, ordinary attachments remain candidates, IDs cannot collide with `r1`, and one scoped factual document routes complex instead of fast `document.read`.

- [ ] **Step 2: Run RED**

Run: `cd backend && PYTHONPATH=. pytest tests/agents/v2/test_context_binding_routes.py tests/agents/v2/test_supervisor_v2.py -q`

- [ ] **Step 3: Implement deterministic projection**

Add `DeterministicSemanticAdapter.project_api_explicit_targets(draft, request)` and call it after UI reconciliation. Build resolved references only from `request.known_documents` whose source is `api_explicit`. Extend `decide_route(..., request=None)` so current explicit IDs force factual `retrieve` to `complex_research/multi_document_research`; `route_node` passes `state["request"]`.

- [ ] **Step 4: Run GREEN and commit**

```bash
cd backend && PYTHONPATH=. pytest tests/agents/v2/test_context_binding_routes.py tests/agents/v2/test_supervisor_v2.py -q
git add backend/app/services/agents/supervisor_v2.py backend/app/services/agents/v2/nodes/routing.py backend/tests/agents/v2
git commit -m "feat(v2): enforce explicit document hard scope"
```

### Task 4: Add deterministic retrieve planning and evaluator integration

**Files:**
- Create: `backend/app/services/agents/v2/skills/retrieve/__init__.py`
- Create: `backend/app/services/agents/v2/skills/retrieve/policy.py`
- Modify: `backend/app/services/agents/v2/complex_research_graph.py`
- Modify: `backend/app/services/agents/v2/nodes/evaluate.py`
- Create: `backend/tests/agents/v2/complex/test_retrieval.py`
- Modify: `backend/tests/agents/v2/test_evaluation_grounding.py`

**Interfaces:**
- Produces: `build_retrieve_plan(ResearchPlanningInput) -> TaskPlan`, initial proposal selection for `retrieve`, and evaluator-owned sufficiency/coverage semantics for admitted retrieval evidence.

- [ ] **Step 1: Write failing real-graph and evaluator tests**

Assert reference-free retrieval checkpoints one targetless task before dispatch; explicit pins create target units and task target IDs; missing catalog entry produces typed unavailable; actual scheduler call count is one. Prove an unscoped retrieve-only task with one admitted supporting use passes the every-expecting-task evidence gate. Prove a scoped document-level retrieve target with an admitted target-bound coverage use on the pinned revision produces `read_partial`, satisfies an explicit `minimum_status="read_partial"` criterion, and rejects wrong-target, wrong-revision, and locator-incompatible uses.

- [ ] **Step 2: Run RED**

Run: `cd backend && PYTHONPATH=. pytest tests/agents/v2/complex/test_retrieval.py -q`

- [ ] **Step 3: Implement policy**

```python
def build_retrieve_plan(input: ResearchPlanningInput) -> TaskPlan:
    # build TargetUnit values for current explicit target bindings; otherwise ()
    # create exactly one DocumentRetrieveInput(query=semantic.normalized_query,
    # target_ids=tuple(unit.target_id for unit in targets), top_k=8)
    # validate_task_plan before return
```

Select it in `build_initial_proposal()` for `work_type == "retrieve"`; preserve compare/summarize/cross-domain branches. Add `document.retrieve` to `_EVIDENCE_SUPPLYING_CAPABILITIES`. Extend `build_coverage()` so a governed, hydrated `coverage` use produced by a `document.retrieve` task establishes retrieval coverage without fabricating a read observation: a chunk from the pinned revision is `read_partial` for a document-level target, while locator-specific targets must satisfy `locator_covers`. The retrieve policy must set document-level target coverage criteria to `minimum_status="read_partial"`. Keep existing `document.read` coverage semantics unchanged.

- [ ] **Step 4: Run GREEN and commit**

```bash
cd backend && PYTHONPATH=. pytest tests/agents/v2/complex tests/agents/v2/test_evaluation_grounding.py tests/agents/v2/fast_paths/test_scheduler.py -q
git add backend/app/services/agents/v2/skills/retrieve backend/app/services/agents/v2/complex_research_graph.py backend/app/services/agents/v2/nodes/evaluate.py backend/tests/agents/v2/complex/test_retrieval.py backend/tests/agents/v2/test_evaluation_grounding.py
git commit -m "feat(v2): plan and evaluate factual retrieval"
```

### Task 5: Wire the live revision-manifest retrieval service

**Files:**
- Modify: `backend/app/services/agent/runtime_selector.py`
- Modify: `backend/app/services/agents/supervisor_v2.py`
- Modify: `backend/app/services/agents/v2/persistence/document_views.py`
- Modify: `backend/app/services/agents/v2/tools/adapters.py`
- Modify: `backend/tests/agents/v2/test_supervisor_v2_lifespan.py`
- Modify: `backend/tests/agents/v2/complex/test_tool_gateway.py`
- Modify: `backend/tests/api/test_agent_v2_streaming.py`

**Interfaces:**
- Produces: `V1RevisionAwareRetrievalService`; registry includes `document.retrieve` only when manifest/provider dependencies are live.

- [ ] **Step 1: Write failing wiring tests**

Prove manifest namespace is loaded per revision, hard filters reach the provider, missing/incompatible manifests fail closed, and all four ingress callers construct equivalent scope.

- [ ] **Step 2: Run RED**

Run: `cd backend && PYTHONPATH=. pytest tests/agents/v2/test_supervisor_v2_lifespan.py tests/api/test_agent_v2_streaming.py -q`

- [ ] **Step 3: Implement request-scoped service**

The service loads exact manifest identity via `document_views`, invokes the existing HTTP embed/rerank client with workspace/document/revision namespace filters, and converts only matching results to `RevisionRetrievedChunk`. Register the capability in `build_v2_capability_registry`; do not add scope fields to `CapabilityRuntimeContext`. Add `document.retrieve` to the existing model-facing `_TOOL_INPUT_TYPES` schema map so a registered visible tool cannot raise `UnknownAgentTool`; test the real registry/schema projection path.

- [ ] **Step 4: Run GREEN and commit**

```bash
cd backend && PYTHONPATH=. pytest tests/agents/v2 tests/api/test_agent_v2_streaming.py tests/api/test_agent_runtime_selector.py -q
git add backend/app/services/agent/runtime_selector.py backend/app/services/agents backend/tests/agents/v2 backend/tests/api
git commit -m "feat(v2): wire live revision retrieval"
```

### Task 6: Add observability, documentation, and P0 live gate

**Files:**
- Modify: `backend/app/services/agent/rollout_metrics.py`
- Modify: `backend/scripts/collect_v2_rollout_report.py`
- Modify: `backend/tests/agents/v2/test_rollout_metrics.py`
- Modify: `docs/harness.md`
- Modify: `docs/embedding.md`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Produces: zero-dispatch factual regression signal and documented authenticated gate.

- [ ] **Step 1: Write failing metric tests**

Assert a factual complex terminal with zero capability calls is gate-invalid; explicitly unsupported/denied routes remain classified by their typed outcome; no raw content enters metrics.

- [ ] **Step 2: Run RED, implement minimal counters, run GREEN**

Run: `cd backend && PYTHONPATH=. pytest tests/agents/v2/test_rollout_metrics.py tests/scripts/test_v2_ab_replay.py -q`

- [ ] **Step 3: Run static and regression gates**

```bash
! rg 'capability\.execute\(' backend/app/services/agents/v2/tools
! rg 'TaskScheduler|scheduler\.execute\(|checkpointer' backend/app/services/agents/v2/tools
cd backend && PYTHONPATH=. pytest tests/agents/v2 tests/api/test_agent_v2_streaming.py -q
git diff --check
```

- [ ] **Step 4: Commit and perform live gate without restarting vLLM**

```bash
git add backend/app/services/agent/rollout_metrics.py backend/scripts/collect_v2_rollout_report.py backend/tests docs/harness.md docs/embedding.md CLAUDE.md README.md
git commit -m "test(v2): gate factual retrieval execution"
```

Use the authenticated harness for one unscoped and one hard-scoped query. Required evidence: non-zero retrieval call/task/EvidenceUse; scoped citations are a subset of requested documents; no approximately-100-ms zero-dispatch terminal.

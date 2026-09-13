# LangGraph v2 Factual Retrieval and Revision-Safe Reindex — Design

**Date:** 2026-09-13

**Status:** Approved

**Branch:** `feat/langgraph-v2`

**Supersedes only:** the Phase-3 assumption that generic `retrieve` / `multi_document_research` may terminate as unsupported. All existing execution, authorization, checkpoint, evidence, grounding, shadow-isolation, and rollout invariants remain authoritative.

**Related documents:**

- `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`
- `docs/superpowers/plans/2026-09-11-langgraph-v2-agent-tool-node-amendment.md`
- `docs/superpowers/plans/2026-09-11-langgraph-v2-phase3-rollout.md`
- `docs/superpowers/specs/2026-09-13-langgraph-v2-defect-repair-design.md`
- `docs/reports/2026-09-13-langgraph-v2-live-test-report.md`

## 1. Objective

Restore factual answering in the v2 serving arm and make the original multi-step goal operational:

1. An open factual query must execute retrieval instead of terminating with no plan, no capability call, and typed `insufficient` in roughly 100 ms.
2. API/UI `document_ids` are a hard retrieval scope, not passive contextual candidates.
3. Retrieval evidence must be tied to immutable published revisions and produce grounded citations.
4. Reindex must publish a new revision only after every stage and required artifact for that same revision is complete.
5. Complex multi-step questions may use an LLM planner, while execution authority remains exclusively with validation, checkpointing, and the shared scheduler.

The delivery order is P0 factual retrieval, P1 revision-safe reindex, then P2 adaptive multi-step planning. Each stage must be independently testable and releasable.

## 2. Confirmed root causes

### 2.1 Generic factual work has no complex plan policy

`nodes/routing.py::decide_route()` maps a reference-free factual query to:

```text
QueryAnalysis(work_type="retrieve", domains=("document",))
→ RouteDecision(route="complex_research", reason_code="multi_document_research")
```

The production `create_supervisor_v2_graph()` does replace the old Phase-2 stub with `build_complex_research_subgraph()`. The failure is therefore not a live stub. Inside the subgraph, `build_initial_proposal()` supports only comparison, summarization, and People→Document cross-domain work. `retrieve` produces no plan; `decide_node()` records unavailable; the parent receives no plan or evidence evaluation and finalizes as typed `insufficient`.

This explains the observed 101–138 ms turns with zero embedding calls, zero LLM calls, and zero citations.

### 2.2 Explicit document IDs are not semantic targets

Ingress stores filtered `document_ids` as `KnownDocumentResource(source="api_explicit")`. `DeterministicSemanticAdapter.reconcile_ui_selections()` intentionally resolves only `ui_selection`; `api_explicit` resources are never projected into semantic document references. Consequently the Binding Resolver sees no references, `_current_bound_count()` is zero, and pinning documents at the API does not make a factual route executable.

### 2.3 Discovery is not factual retrieval

`document.search` intentionally returns opaque `DocumentDiscoveryCandidate` values and no factual evidence or read coverage. Reusing it as a chunk-retrieval capability would break its existing ownership and observation semantics. `document.read` reads an already-planned exact document or section and is also not an adequate semantic chunk-retrieval operation.

A separate factual `document.retrieve` capability is required.

### 2.4 The complex boundary is deterministic-only

The current complex graph is structurally adaptive but its initial and replan proposals are deterministic policy functions. No LLM planner participates in a generic multi-step request. This is acceptable for P0's deterministic retrieval baseline but does not satisfy the stated multi-step objective; P2 adds the governed model planner.

### 2.5 Reindex completion is inferred from document-global mirror flags

The live database currently contains 400 non-deleted documents, 398 without `current_revision_id`; revision statuses include only two published revisions and multiple failed revisions. The observed failures include `verify:RevisionArtifactsIncomplete`.

`check_and_finalize()` decides that all work is complete using `Document.embed_done`, `captions_done`, and `kg_done`. These are document-global v1/UI mirrors. A new generation can observe values left by an earlier generation and call `finalize_revision_if_complete(expect_complete=True)` before its own artifacts exist, terminalizing the new revision as failed.

Resetting the global flags during allocation is a possible mitigation but is not a sound revision-ownership boundary. It also has a CRITICAL GitNexus blast radius across upload, reindex, clone, batch, and retry flows. The durable fix is revision-owned stage state.

## 3. Normative contract amendment

The prior capability contract freeze is reopened only for this additive retrieval operation. Existing variants and checkpoint fields are not renamed or removed.

### 3.1 New capability types

Add these variants to the discriminated capability unions:

```python
class DocumentRetrieveInput(ContractModel):
    kind: Literal["document.retrieve"]
    query: str
    target_ids: tuple[str, ...] = ()
    top_k: int = Field(default=8, ge=1, le=20)


class DocumentRetrieveOutput(ContractModel):
    kind: Literal["document.retrieve"]
    retrieved_unit_count: int = Field(ge=0)
```

Semantics:

- Empty `target_ids` means retrieval over the current authenticated workspace scope.
- Non-empty `target_ids` means every target resolves through the authoritative checkpointed plan and bindings. It never contains raw document or revision IDs.
- `query` is model-supplied only through a governed proposal; authorization, workspaces, document IDs, revision IDs, vector namespaces, deadlines, and service clients remain runtime-owned.
- `DocumentRetrieveOutput` contains only counts. Content exists only in governed Evidence records referenced by `AgentResult.evidence_uses`.

The implementation contract version advances additively. Compatibility tests must prove that old serialized capability variants and existing supervisor checkpoints still deserialize unchanged. No database migration is required for the capability-union extension.

### 3.2 Unchanged invariants

```text
proposal
→ validate
→ acquire/commit revision leases
→ authoritative LangGraph checkpoint
→ shared TaskScheduler
→ DocumentRetrieveCapability.execute
→ persist/lease EvidenceUse
→ checkpoint result
→ evaluate
→ synthesize
→ ground
→ final response
```

The following remain forbidden:

- a second scheduler;
- capability execution from a tool adapter, planner, or graph node other than through `TaskScheduler`;
- raw chunks in planner observations or checkpointed capability output;
- model-supplied ACL, workspace, document, revision, namespace, deadline, or service values;
- success without a sufficient evidence verdict, grounded claims, and valid citations.

## 4. P0 — Factual retrieval

### 4.1 Explicit document hard scope

After the existing API ACL filter, every `KnownDocumentResource(source="api_explicit")` is projected deterministically into a resolved `DocumentReference` in the semantic draft. Its reference ID is stable and namespaced, for example `api_explicit:<resource_id>`, so it cannot collide with preprocessor-generated `r1`, `r2`, or clarification references.

Rules:

1. Only `api_explicit` resources supplied for the current turn are projected.
2. The projection never accepts document IDs from raw query text or model output.
3. The Binding Resolver remains the sole owner that loads and pins `current_revision_id` under current ACL.
4. A missing, unpublished, deleted, or unauthorized resource fails closed before dispatch.
5. Explicit documents are targets, not supporting or discovered documents.
6. When any explicit target exists, retrieval results outside those targets are rejected even if the underlying provider returns them.

`route_node` receives the full supervisor state and may compare resolved semantic document IDs with the current request's `api_explicit` resources. A scoped factual query routes to `complex_research`, including the one-document case; it must not be mistaken for the old exact-document metadata/read fast path.

### 4.2 Deterministic retrieve policy

P0 adds a deterministic initial policy for `QueryAnalysis.work_type == "retrieve"`:

- Scoped request: create one `document.retrieve` task referencing all current explicit target IDs.
- Unscoped request: create one targetless `document.retrieve` task over authenticated workspace scope.
- The task objective and initial query come from finalized semantics.
- The policy uses only `document.retrieve` when it is present in the request-scoped capability catalog; otherwise it produces a typed dependency-unavailable result and dispatches nothing.
- The initial plan is validated, leased where it pins targets, and checkpointed by the existing `validate_checkpoint_node` before execution.

P0 may perform one deterministic coverage-driven retry using the existing replan budget. It may reformulate only from finalized semantics and validated evaluation gaps; it may not use raw evidence or invent document targets. General LLM planning belongs to P2.

### 4.3 Document retrieval capability

`DocumentRetrieveCapability` is an atomic factual capability. It receives `AgentRequest + CapabilityRuntimeContext` and constructor-injected request-scoped services.

For scoped retrieval it resolves every target through `PinnedTargetResolver`, producing an allow-set of `(document_id, document_revision)`. For unscoped retrieval it uses only `CapabilityRuntimeContext.workspace_ids` and admits sources only after resolving a published current revision in one of those workspaces.

The retrieval service must:

1. Load `DocumentRevisionBuild` for the pinned or resolved revision.
2. Read `embedding_namespace`, `embedding_model_hash`, `embedding_dimension`, and `vector_artifact_version` from that manifest.
3. Query the embedding/rerank provider using that revision-owned namespace and the hard document allow-set when present.
4. Return typed located chunks carrying document ID, revision ID, locator, content, and provider score.
5. Reject or drop chunks whose document, revision, namespace, or workspace does not match the authorized request.
6. Persist accepted chunks through `EvidenceBuilder` as `DocumentSourceIdentity` records and return only `EvidenceUseRef` values plus `DocumentRetrieveOutput`.
7. Report retrieval coverage only for explicit target IDs actually represented by admitted chunks. An unscoped targetless retrieval supplies evidence but does not fabricate target coverage.

Retrieval coverage is evaluator-owned and is derived from admitted, target-bound `EvidenceUse(purpose="coverage")` records after governed hydration. `document.retrieve` does not emit the read-only `CoverageObservation` contract. The evaluator treats `document.retrieve` as an evidence-supplying capability. For an explicit document-level retrieval target, at least one admitted chunk on the pinned document revision establishes `read_partial`; the deterministic retrieve policy sets that target's `CoverageCriterion.minimum_status` to `read_partial`. Locator-specific targets retain the existing `locator_covers` compatibility rule and must not receive coverage from a mismatched locator. This rule preserves the frozen coverage contracts while making both scoped and targetless retrieve-only plans capable of reaching a sufficient verdict.

The provider adapter may call the existing embed/rerank HTTP service, but it must not fall back to a current-config namespace when the revision manifest is absent or incompatible.

### 4.4 Evaluation and synthesis

The shared evaluator remains authoritative:

- A scoped plan is sufficient only when required target coverage and semantic evidence criteria pass. For `document.retrieve`, target coverage is computed from admitted target-bound coverage uses under the retrieval rule in §4.3; it does not require a read-only `CoverageObservation`.
- An unscoped plan is sufficient only when at least one admitted EvidenceUse exists and the semantic judge finds the evidence adequate. `document.retrieve` is included in the evaluator's evidence-supplying capability set so this gate is reachable.
- Empty, denied, timed-out, stale-revision, or out-of-scope retrieval results cannot become success.

The existing synthesis and grounding nodes hydrate EvidenceUses, enforce budgets, validate claims, and emit citations. No synthesis or grounding logic is added to the capability or complex subgraph.

## 5. P1 — Revision-owned stage completion and reindex

### 5.1 Revision stage state

Add revision-owned state for the canonical stages `parse`, `embed`, `caption`, and `kg`. The implementation may use a dedicated `document_revision_stages` table or equivalently constrained revision-owned rows, but the persisted key and behavior are fixed:

```text
(revision_id, stage) UNIQUE
state ∈ {pending, running, completed, skipped, failed}
attempt_count >= 0
updated_at
failure_class nullable
```

A migration creates the structure before any worker consumer is deployed. Existing revisions are not inferred complete from `Document.*_done`; only newly allocated/retried revisions use the new stage gate. Legacy rows remain readable and fail closed until reindexed.

### 5.2 Worker ownership

- Allocation creates the required stage rows according to immutable `RevisionBuildProfile`.
- A worker message may update only the named `revision_id` and its own stage.
- Redelivery is idempotent.
- A delayed message for an older revision cannot update stage state for the current or newer revision.
- `skipped` is explicit and allowed only by the build profile.
- A failed stage terminalizes only its own revision and never rewrites a newer document pointer or mirror state.

### 5.3 Finalization gate

`check_and_finalize(..., revision_id=...)` stops using document-global completion flags as authoritative input. It loads:

1. the revision's immutable build profile;
2. every required revision-stage row;
3. the revision's `DocumentRevisionBuild` manifest;
4. current source tombstone and generation state.

It calls `finalize_revision_if_complete(expect_complete=True)` only when all required stages for that revision are `completed` or profile-authorized `skipped`. Incomplete stages return `NOT_READY`, not `FAILED`. Once stage completion is authoritative, a missing required artifact is a real integrity failure and may terminalize that revision.

`Document.embed_done`, `captions_done`, `kg_done`, and `status` remain v1/UI mirrors updated from revision outcomes. They never authorize revision publication.

The current uncommitted reset-at-allocation change is treated as a candidate mitigation, not an accepted design requirement. It must not be committed as the sole fix. If retained for UI compatibility, tests must prove it cannot affect a published revision, concurrent generation, clone, retry, or delayed worker flow.

### 5.4 Controlled reindex

After code and migrations pass:

1. Select one accessible non-deleted document lacking `current_revision_id`.
2. Allocate a copy-on-write reindex generation.
3. Observe revision stage transitions and artifact-manifest writes.
4. Verify `draft → building → verified → published`.
5. Verify `Document.current_revision_id` advances only after publication.
6. Run a hard-scoped factual query against that document.
7. Expand to one small workspace only after the single-document probe succeeds.

Never set `current_revision_id` manually, reuse legacy artifacts as revision-ready, destructively recreate shared collections, or restart vLLM engines.

## 6. P2 — Governed adaptive multi-step planning

P2 introduces a request-scoped planner service into `RuntimeServices`; it remains runtime-only and is never checkpointed. The complex graph invokes it only inside the proposal-owning validation node so no unvalidated proposal crosses a checkpoint boundary.

The planner receives only:

- finalized semantic context;
- current bindings and redacted current plan;
- narrowed capability catalog;
- typed safe observations;
- task outcomes and evaluation gaps;
- remaining task/replan/parallel/deadline budgets.

It may propose `document.retrieve` and existing authorized capability tasks. It may not execute, bind, checkpoint, create EvidenceUses, decide sufficiency, synthesize, ground, or emit a final response.

Every proposal passes through the existing governed append constructor and frozen validation rules before lease/checkpoint. Invalid, unauthorized, over-budget, cyclic, or widening proposals are rejected. Replans are append-only and bounded by deployment settings.

The deterministic P0 retrieve policy remains the fallback when the planner is disabled or unavailable, ensuring ordinary factual retrieval does not depend on an LLM planning call. Multi-step acceptance tests must separately prove that an enabled complex request makes a planner LLM call and executes at least two causally related tasks.

## 7. Failure handling and observability

### 7.1 Typed outcomes

- No published revision or no admitted evidence: `insufficient`.
- Required explicit document unresolved/unpublished: `clarify` when user action can resolve it; otherwise `insufficient` with an operator-visible reason.
- Capability/service unavailable: typed dependency error, never success.
- Permission or hard-scope violation: `denied`, with no leaked document identity.
- Planner unavailable: deterministic P0 policy; if that is also unavailable, typed dependency error.
- Invalid checkpoint/plan/evidence: typed error and no further dispatch.

### 7.2 Metrics and trace requirements

Each factual turn records:

- resolved route and reason;
- plan/task count;
- capability call count by capability;
- embed/rerank call count and latency;
- retrieved/admitted/dropped chunk counts;
- drop reason: ACL, hard scope, revision mismatch, missing manifest, stale namespace, malformed source;
- EvidenceUse and citation counts;
- planner call count, latency, and fallback reason;
- terminal status.

A factual complex turn that reaches terminal state with zero capability calls is an internal regression unless it is an explicitly unsupported/denied route. It must be counted as an error for rollout gates, not a normal insufficient answer.

No metric or log contains raw evidence content, personal values, tokens, or runtime secrets.

## 8. Testing strategy

Implementation follows TDD: failing behavior test, observed failure, minimal implementation, passing focused test, regression suite, then commit.

### 8.1 P0 tests

1. Reference-free factual query routes to an executable retrieve plan.
2. One and multiple `api_explicit` documents become deterministic target references and revision pins.
3. Explicit targets force complex retrieval rather than the exact-document fast path.
4. Hard scope rejects a provider source outside `document_ids`.
5. Revision mismatch and missing manifest are excluded and cannot produce citations.
6. Old capability/checkpoint payloads deserialize after adding `document.retrieve`.
7. Plan is checkpointed before the shared scheduler calls the capability.
8. EvidenceUse is leased before its result checkpoint.
9. Authenticated standalone, session-SSE, admin evaluation, and Telegram ingress preserve the same hard-scope semantics.
10. End-to-end query produces an embed/retrieval call, task result, EvidenceUse, grounded answer, and citation wholly inside the requested document set.
11. Reference-free end-to-end query performs retrieval instead of terminating before dispatch.
12. No second scheduler or direct capability execution appears.

### 8.2 P1 tests

1. Allocation creates correct revision-stage rows for FULL, CHAT_UPLOAD, and PARSE_ONLY profiles.
2. Stale document mirror flags cannot finalize a new revision.
3. Out-of-order and duplicate worker messages are idempotent.
4. A delayed old-generation worker cannot alter a newer revision's stage or document mirror.
5. Finalization waits while any required stage is pending/running.
6. Authorized skipped stages satisfy only their matching profile.
7. Complete stages plus complete manifest publish exactly once.
8. Complete stages plus incomplete manifest fail only that revision.
9. Concurrent finalizers preserve generation CAS and one current pointer.
10. Upload, clone, explicit reindex, batch, and retry flows retain their expected behavior.
11. A real single-document reindex publishes and becomes retrievable through v2.

### 8.3 P2 tests

1. Enabled multi-step query calls the planner LLM.
2. Planner sees only narrowed catalog and redacted typed observations.
3. Proposed tasks are validated and checkpointed before dispatch.
4. At least two causally related tasks execute for the multi-step golden case.
5. Unauthorized, cyclic, over-budget, and scope-widening proposals are rejected.
6. Planner timeout falls back to deterministic P0 retrieval.
7. Replan terminates at configured limits.
8. Shadow runs use the same planner/retrieval behavior without production writes or outbound events.

## 9. Delivery gates

### P0 gate

- Focused and v2 regression tests pass.
- Authenticated factual probes show non-zero retrieval calls and task results.
- Scoped success has at least one citation and every citation belongs to the explicit document allow-set.
- Unscoped factual retrieval no longer terminates in approximately 100 ms with zero capability calls.
- Rollout remains v1-default unless explicitly configured otherwise.

### P1 gate

- Migration applies idempotently before worker consumers.
- Worker/revision concurrency suite passes.
- One real document reaches `published` and receives `current_revision_id` without manual DB edits.
- The P0 hard-scoped probe answers from that revision.
- No vLLM service is restarted.

### P2 gate

- Golden multi-step cases prove planner calls and multiple executed tasks.
- Security, checkpoint-before-dispatch, grounding, cancellation, and shadow-isolation gates pass.
- Canary promotion remains blocked until live sample and duration thresholds in the canonical rollout plan pass.

## 10. Files and ownership

Expected implementation areas:

- Contract amendment: `backend/app/services/agents/v2/contracts/capability.py` and compatibility tests.
- Semantic hard scope: `backend/app/services/agents/supervisor_v2.py`, routing/context tests, and ingress consistency tests.
- Retrieval: `backend/app/services/agents/v2/capabilities/document.py`, request-scoped service wiring in `backend/app/services/agent/runtime_selector.py`, registry construction, planner policy, evaluator coverage tests.
- Complex policy/planner: `backend/app/services/agents/v2/complex_research_graph.py` plus a focused retrieval policy module and P2 planner adapter.
- Revision stages: raw migration, ORM mapping after migration, revision repository, worker utilities and worker call sites.
- Documentation: `CLAUDE.md`, `README.md` pointer, `docs/embedding.md`, `docs/workers.md`, `docs/harness.md`, and `.env.example` for any new P2 settings.

One writer owns each shared file per implementation task. P0, P1, and P2 are separate review/commit series. No task may absorb unrelated cleanup.

## 11. Blast radius and safety controls

GitNexus analysis before planning reported:

- `decide_route`: LOW.
- `build_initial_proposal`: LOW.
- `DeterministicSemanticAdapter.reconcile_ui_selections`: LOW.
- `build_v2_ingress`: HIGH; affects admin evaluation, standalone chat, session chat, and Telegram.
- `create_supervisor_v2_graph`: HIGH; affects production and shadow composition.
- `_allocate_explicit_revision`: CRITICAL; affects upload, reindex, clone, batch, and retry flows.

Implementation must rerun exact upstream impact immediately before editing every symbol. HIGH or CRITICAL edits require an explicit warning and focused regression matrix. Before each commit, run `detect_changes --scope compare --base-ref main`, inspect affected flows, and stage only task-owned files.

## 12. Rollback

- P0 can be disabled by removing `document.retrieve` from the request-scoped registry; old capability variants remain valid.
- P1 migration is additive. Worker consumers are deployed only after schema readiness passes. Rollback stops new stage-state consumers but does not delete recorded revision stage rows.
- P2 has a deployment flag defaulting off; deterministic P0 retrieval remains available.
- Failed reindex generations are marked failed/abandoned; the previous published pointer remains unchanged.
- Kill switch and existing canary controls remain authoritative for all new v2 traffic.

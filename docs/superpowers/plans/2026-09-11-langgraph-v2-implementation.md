# LangGraph v2 Implementation Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the frozen LangGraph v2 architecture through compatibility discovery, immutable revision foundations, explicit Agent/Node/Capability/Skill ownership, deterministic fast paths, one governed complex-research planning boundary, and controlled rollout while v1 remains production default.

**Architecture:** The suite has four executable phase plans plus one normative implementation amendment. Phase 0 discovers exact orchestration/checkpoint versions; Phase 1 establishes immutable revisions/contracts/evidence/checkpointing; the Agent/Tool/Node amendment fixes implementation ownership without changing frozen business contracts; Phase 2 builds node-based deterministic fast execution over shared typed capabilities; Phase 3 adds one adaptive complex planner whose tool proposals still pass through TaskPlan validation, checkpointing, the shared scheduler, evaluator, synthesis, and grounding.

**Tech Stack:** Python 3.11, Pydantic v2, discovered/pinned LangGraph and checkpoint stack, FastAPI, async SQLAlchemy/PostgreSQL 15, RabbitMQ, MinIO, Chroma, Neo4j, Redis, pytest, React/Vitest, Docker Compose, GitNexus.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- Architecture remains **Approved design** with contract freeze active; plans do not add/change frozen business-contract fields.
- Keep `backend/app/services/agents/supervisor.py` as v1 and create independent `backend/app/services/agents/supervisor_v2.py`.
- Apply [`2026-09-11-langgraph-v2-agent-tool-node-amendment.md`](2026-09-11-langgraph-v2-agent-tool-node-amendment.md) before Phase 2. It is normative for implementation taxonomy, package ownership, and tool execution path.
- The frozen spec carries a **non-contract §6 erratum** (module/graph boundary) pointing to that amendment; the spec's business contracts remain higher authority and the architecture status stays Approved/frozen.
- **Agent:** adaptive plan/replan proposal only. **Node:** LangGraph state/routing/lifecycle. **Capability:** atomic typed domain operation. **Skill:** task strategy. **Workflow/subgraph:** predetermined multi-step algorithm.
- Do not create v2 People/RAG/Document/Section/KG/Summary/Comparison/Evaluation/Grounding domain agents or domain graph wrappers.
- `CapabilityDescriptor`, `CapabilityInput`, `CapabilityOutput`, and the runtime-only `CapabilityRuntimeContext` remain frozen contracts owned by the contracts layer; plans import them and never redefine or field-extend them.
- Fast and complex paths share the same capability implementations and request-scoped `CapabilityRegistry`.
- “Agent chooses a tool” means it proposes `TaskSpec.capability`; it does not directly invoke a capability.
- Every factual capability dispatch must follow `TaskPlan/TaskSpec proposal -> validation -> authoritative checkpoint -> scheduler -> capability`.
- Capability model input cannot supply workspace IDs, ACL/People permissions, deadlines, feature flags, service clients, registry construction, or cancellation authority.
- `CapabilityOutput` is not model observation by default; sensitive outputs require explicit safe observation projection.
- Chat DB remains conversation truth; checkpoints hold execution/interrupt/resume only.
- Never checkpoint current identity, ACL/workspace scope, deadline, DB/client/service objects, or capability registry.
- Persist raw user text before semantic normalization.
- Direct non-factual responses own no plan; every factual invocation owns a checkpointed TaskPlan.
- Do not implement binding/evidence/coverage/reuse before immutable revision publication passes.
- Write remains v1-owned/outside this v2 rollout.
- Keep API, SSE, cancellation, citations, and frontend behavior compatible.
- Before every phase run repository-drift preflight; before existing-symbol edits run exact GitNexus impact; before each commit run compare-scope detect-changes and stage narrow paths.
- Keep `.orca/` untracked.

## Exact Dependency Order

1. [`2026-09-11-langgraph-v2-phase0-benchmark.md`](2026-09-11-langgraph-v2-phase0-benchmark.md)
   - discover exact LangGraph version supporting `StateGraph(..., context_schema=...)`;
   - discover compatible checkpoint-postgres/psycopg stack and prove real DSN/setup round-trip;
   - benchmark native vs Deep Agents using frozen contracts; pin only winner stack.

2. [`2026-09-11-langgraph-v2-phase1-foundation.md`](2026-09-11-langgraph-v2-phase1-foundation.md)
   - **1A:** raw-SQL migration runner only;
   - **1B:** post-migration ORM/readiness mappings;
   - **1C:** immutable revision-owned build state/artifacts and legacy-v1 compatibility;
   - **1D:** frozen contracts, evidence/checkpoint governance, adapters.

3. [`2026-09-11-langgraph-v2-agent-tool-node-amendment.md`](2026-09-11-langgraph-v2-agent-tool-node-amendment.md)
   - define Agent vs Node vs Capability vs Skill vs Workflow;
   - define canonical `v2/nodes`, `v2/capabilities`, `v2/tools`, `v2/skills` layout;
   - prohibit domain-agent/domain-graph wrappers;
   - require `TaskPlan -> validate -> checkpoint -> scheduler` for all tool execution;
   - require explicit sensitive-output observation projection.

4. [`2026-09-11-langgraph-v2-phase2-fast-paths.md`](2026-09-11-langgraph-v2-phase2-fast-paths.md)
   - Context/Binding/Routing/FastPlan/Execute/Evaluate/Synthesize/Grounding/Clarification/Finalizer nodes;
   - request-scoped shared People/Document/Section/KG capabilities;
   - deterministic one-task fast plans and shared scheduler;
   - independent supervisor, selector, SSE compatibility;
   - no complex adaptive planning yet.

5. [`2026-09-11-langgraph-v2-phase3-rollout.md`](2026-09-11-langgraph-v2-phase3-rollout.md)
   - golden shared-evaluator A/B preflight;
   - governed `AgentToolGateway` and `AgentToolObservation` projection;
   - one adaptive `ComplexResearchGraph` planning/replanning boundary as a checkpointed LangGraph subgraph;
   - comparison/summarization as skills over shared capabilities, with an explicit supported work-type scope (compare + summarize in; evaluate/legal/compliance/write out, typed unavailable);
   - deterministic People→Document dependency materialization;
   - bounded replan/discovery;
   - isolated shadow graph;
   - rollout schema migration-first Task 7A, then consumers Task 7B;
   - live canary/kill switch/24-hour gates.

## Hard Gates

| Gate | Required evidence | Blocks |
|---|---|---|
| Phase 0 | exact versions, `context_schema` proof, real AsyncPostgresSaver round-trip, frozen-contract parity, winner report | dependency changes |
| Phase 1A | advisory-locked populated-legacy migration, no v2 ORM import | Phase 1B |
| Phase 1B | exact readiness; mapped pre-existing schema; startup cannot emit v2 DDL | revision producers |
| Phase 1C | revision owns build state/artifacts; R1 survives R2; legacy remains v1-only until ready | bindings/evidence |
| Phase 1D | frozen contracts, EvidenceRecord/Use, ACL/retention/audit, checkpoint/adapters | Phase 2 |
| Ownership amendment | one canonical node/capability layout; no domain agents; no direct tool execution; sensitive observation projection | Phase 2/3 implementation |
| Phase 2 | checkpoint-before-dispatch, shared capabilities, all evaluation statuses, clarification/resume, SSE/API parity, v1 default | complex pilots |
| Complex research | tool proposal becomes validated checkpointed TaskSpec; fast/complex share capabilities; People raw data absent from planner; replan bounded | shadow/canary |
| Golden A/B | functional + shared-evaluator quality comparison, same evaluator version both arms | shadow/canary |
| Shadow | separately compiled graph + isolated saver/stores; zero production writes/events | canary |
| Live canary | DB metrics, ≥200 samples/arm, ≥24h continuous, zero security violations, latency/error/cancellation gates | promotion |

## Prohibited Sequencing and Ownership

1. Hardcoding package pins before Phase-0 compatibility discovery.
2. Registering v2 ORM before Phase-1A migration is applied/verified.
3. Using legacy Document completion flags to decide revision work.
4. Deleting revision artifacts by document during replacement build.
5. Treating legacy Chroma/KG artifacts as revision-ready without full revision-aware reindex.
6. Recreating vector collections destructively after published revisions exist.
7. Persisting EvidenceUse before authoritative TaskPlan checkpoint.
8. Exposing raw People rows to planner/checkpoint or fabricating People→Document input.
9. Compiling shadow with production checkpointer/writable stores.
10. Treating batch A/B as live 24h canary evidence.
11. Inner nodes/capabilities streaming final user prose.
12. Publishing lower-generation revision over newer current revision.
13. Creating duplicate draft for repeated ingest attempt.
14. Physically deleting document revision artifacts before tombstone + independent artifact-GC eligibility.
15. Clone/reindex mutating published current artifacts in place.
16. Treating absent security metric as secure.
17. Deploying schema-2 consumers before migration 1→2 is applied/verified.
18. Publishing/promoting revision for tombstoned source.
19. Sharing one GC predicate between evidence retention and revision artifact reclamation.
20. Reusing terminal-failed revision for retry instead of allocating bounded new generation.
21. Using non-canonical source-object identity.
22. Comparing v2-internal grounding-completeness ratio as cross-arm live quality.
23. Blind-delete shared KG canonical entities during revision GC.
24. Recreating v1-style People/RAG/Summary/Comparison/etc. agents or `v2/domain/*_graph.py` wrappers.
25. Giving model control of runtime authorization/scope/deadline/service fields.
26. Implementing separate fast vs complex business logic for same capability.
27. Letting framework-native tool call execute capability before TaskSpec validation/checkpoint.
28. Returning raw `CapabilityOutput` to planner by default.
29. Allowing subagent to own TaskPlan, dispatch capability, create EvidenceUse, decide sufficiency, or emit FinalResponse.
30. Redefining, renaming, or field-extending the frozen `CapabilityDescriptor`/`CapabilityInput`/`CapabilityOutput`/`CapabilityRuntimeContext` outside `contracts/capability.py`.
31. Defining a second scheduler or dispatching a capability outside the single shared Phase-2 `TaskScheduler`.
32. Compiling the complex-research subgraph with its own production/shadow checkpointer or as a module-level singleton — it must inherit the supervisor saver as the `complex_boundary` node.
33. Introducing a `plan_checkpoint` service or an `EvidenceEvaluator` service/class — plan persistence is LangGraph state plus the supervisor saver, and evaluation is the shared `evaluate_evidence(...)` node function; `RuntimeServices` is defined once in Phase 2 and has no such fields.
34. Letting `AgentToolGateway` execute a capability or checkpoint a plan — it is a proposal + observation adapter; only `TaskScheduler` executes and only graph nodes/saver checkpoint.
35. Carrying model-visible observation through a free-form `Mapping`/dict (`safe_metadata`) instead of a typed per-capability projection; unknown result kinds must fail closed.
36. Using the webhook `arrival_identity` as the attempt key, treating an etag as content identity, or decoding S3 event keys zero/multiple times instead of exactly once before `normalize_object_key`.
37. Leaving a `verified` revision non-terminal forever when its source is tombstoned — it must become terminal `abandoned` (with `abandon_reason`) so Predicate B can reclaim its artifacts; `abandoned` is immutable and never current.
38. Reclaiming a revision or evidence payload while a checkpoint retention lease is active, or failing to write/release a lease for a checkpoint that pins a revision/EvidenceUse — a resumable run must never lose pinned artifacts, and an expired lease must never block GC.
39. Creating an out-of-scope Phase-3 skill (`legal_analysis`, `compliance`) or a domain-agent wrapper, or silently routing an unsupported `WorkType`/`RouteReason` to `fast_domain` instead of failing closed with the typed unavailable response.
40. Building a second ownership/execution path: a duplicate scheduler, a plain-Python complex execute loop, a dispatching `AgentToolGateway`, a `plan_checkpoint`/`EvidenceEvaluator` service, a subgraph-owned checkpointer, or scheduler-side `TaskSpec.input` materialization.

## Final Execution Gate

The implementation suite is ready only when all are proven:

| # | Proof | Primary gate |
|---|---|---|
| 1 | Revision publication monotonic under concurrency | Phase 1 |
| 2 | Ingestion triggers converge atomically to one attempt | Phase 1 |
| 3 | Tombstone precedes independent artifact GC | Phase 1 `test_delete_tombstones_before_gc`, `test_publish_after_tombstone_does_not_resurrect_current`, `test_verified_revision_blocked_by_tombstone_becomes_abandoned`, `test_tombstone_abandons_non_published_revisions`, `test_expired_evidence_alone_does_not_release_revision_artifacts`, `test_revision_artifact_gc_requires_no_retained_references` |
| 4 | Current document view resolves immutable current revision | Phase 1 |
| 5 | Clone/reindex cannot bypass revision lifecycle | Phase 1 |
| 6 | KG facts/provenance isolated by revision | Phase 1 |
| 7 | Historical vectors resolve recorded build namespace | Phase 1 |
| 8 | Write has explicit v1 owner | Phase 2 |
| 9 | Evidence encryption/key rotation fail closed | Phase 1 |
| 10 | Failed ingestion retry allocates bounded new generation | Phase 1 |
| 11 | Canonical source identity stable across triggers | Phase 1 `test_source_object_identity_is_canonical`, `test_s3_event_key_is_decoded_once_and_matches_storage_key`, `test_etag_is_a_version_selector_not_content_identity` |
| 12 | No v2 domain-agent/domain-graph wrappers exist | Amendment + Phase 2 |
| 13 | Fast and complex use same capability implementation | Phase 2/3 |
| 14 | Every factual dispatch references validated checkpointed TaskSpec | Phase 2/3 |
| 15 | Capability receives `CapabilityRuntimeContext` only | Phase 2 |
| 16 | Agent-facing tool adapter cannot call capability directly | Phase 3 |
| 17 | Sensitive capability output uses safe observation projection | Phase 3 |
| 18 | People→Document remains deterministic materialization | Phase 3 |
| 19 | Summary/compare/compliance are skills/work types, not agents | Amendment + Phase 3 |
| 20 | Replan is append-only and cannot widen current authorization | Phase 3 |
| 21 | Subagents have no execution/persistence/sufficiency authority | Phase 3 |
| 22 | Canary security metrics have authoritative producers | Phase 3 Task 7B |
| 23 | Every migration precedes consumers requiring its version | Phase 1 + Phase 3 Task 7A/7B |
| 24 | Quality comparison uses same arm-neutral evaluator only in golden preflight | Phase 3 Task 1 |
| 25 | Frozen `Capability*` contract types are imported, never redefined or field-extended | Amendment gate 13 + Phase 2 static guard |
| 26 | One shared `TaskScheduler` defines the only capability-dispatch path | Phase 2 Task 3 `test_task_scheduler_is_defined_once_and_shared` |
| 27 | Complex research is a checkpointed LangGraph subgraph inheriting the supervisor saver | Phase 3 Task 3 `test_complex_subgraph_is_checkpointed_under_supervisor_saver`, `test_complex_subgraph_does_not_open_its_own_checkpointer` |
| 28 | `RuntimeServices` is defined once and excludes `plan_checkpoint`/`EvidenceEvaluator` | Phase 2 Task 3 `test_runtime_services_excludes_plan_checkpoint_and_evaluator_service` |
| 29 | `AgentToolGateway` is proposal + observation only, never a second executor | Phase 3 Task 2 `test_tool_gateway_is_proposal_and_observation_only` + tools guard |
| 30 | Model observation is a typed per-capability projection, never a `Mapping` | Phase 3 Task 2 `test_observation_projection_is_typed_no_mapping`, `test_unknown_result_kind_projection_fails_closed` + tools guard |
| 31 | Checkpoint/revision retention leases block GC for resumable runs | Phase 1 `test_checkpoint_pinning_revision_writes_lease`, `test_active_lease_blocks_artifact_and_evidence_gc`, `test_expired_lease_does_not_block_gc`, `test_terminal_run_releases_lease`, `test_resumable_interrupt_keeps_lease_until_expiry` |
| 32 | Phase-3 supported skill/work-type scope is explicit; out-of-scope work types fail closed | Phase 3 scope table + `test_phase3_scope_excludes_legal_and_compliance_skills`, `test_unsupported_work_type_returns_typed_unavailable` |
| 33 | Parent/child states are connected by explicit adapters and exactly one ownership chain exists | Phase 3 `test_complex_boundary_maps_parent_to_child_state`, `test_complex_boundary_maps_child_result_back_to_execution_state`, `test_complex_subgraph_does_not_require_root_state_schema` |
| 34 | People→Document input is materialized before T2 checkpoint; scheduler never mutates `TaskSpec.input` | Phase 3 `test_people_document_materializes_before_task_append`, `test_t2_checkpoint_contains_final_document_search_input`, `test_scheduler_never_rewrites_task_input` |
| 35 | Phase 1A creates every schema-v1 table mapped in 1B (`after - before == V2_SCHEMA_V1_TABLES`); tombstone clears the current pointer; one GC retention anchor | Phase 1 migration control test, `test_tombstone_clears_current_revision_pointer`, `test_failed_revision_has_gc_retention_anchor` |
| 36 | Document search observation exposes opaque `candidate_ids` resolved by a request-scoped `DiscoveryCandidateRegistry` | Phase 3 `test_document_search_observation_exposes_candidate_ids_only`, `test_planner_cannot_select_document_revision_directly` |
| 37 | 100% rollout means 100% of v2-eligible traffic (`write` is a Domain, selector is not a second classifier) | Phase 3 `test_rollout_100_percent_still_routes_write_to_v1`, `test_rollout_100_percent_still_routes_evaluate_to_v1`, `test_supported_compare_uses_v2_at_100_percent_eligible_rollout` |
| 38 | `ResearchPlanningInput` is an ephemeral projection, never checkpointed | Phase 3 `test_planning_input_is_ephemeral_never_checkpointed` |
| 39 | Binding pin acquires a lease before checkpoint; runner releases only after the terminal checkpoint | Phase 2 `test_binding_pin_acquires_lease_before_checkpoint`, `test_resume_refreshes_lease_before_continuing` |
| 40 | Repositories mutate+flush only; the service/UoW owns the transaction | Phase 1 publish tests (`test_tombstone_race_commits_abandoned_before_error`) |

## Whole-Suite Validation

```bash
git diff --check
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
docker exec hrag-backend pytest tests/agents tests/migrations tests/services tests/api tests/workers -q
make test-recall
make test-section
make test-validity
make fe-lint
make fe-build
cd frontend && pnpm test -- ChatPanel.rollback
```

Expected: all commands exit 0; frozen spec stays Approved; v1 remains default until persisted Phase-3 rollout control promotes v2.

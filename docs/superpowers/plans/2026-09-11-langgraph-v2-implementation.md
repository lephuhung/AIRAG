# LangGraph v2 Implementation Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the frozen LangGraph v2 architecture through compatibility discovery, immutable revision foundations, explicit agent/tool/node ownership, fast paths, complex pilots, and controlled rollout while v1 remains production default.

**Architecture:** The suite has four executable phase plans plus one normative implementation amendment. Phase 0 discovers exact packages that support `context_schema` and proves frozen-contract parity; Phase 1 deploys migration, mappings, revision-owned ingestion, then contracts/stores/adapters in four ordered releases; the agent/tool/node amendment fixes implementation ownership without changing frozen business contracts; Phase 2 composes v2 deterministic nodes and shared capabilities; Phase 3 adds the single adaptive complex-research agent, task skills, complex pilots, and isolated rollout.

**Tech Stack:** Python 3.11, Pydantic v2, discovered/pinned LangGraph and checkpoint stack, FastAPI, async SQLAlchemy/PostgreSQL 15, RabbitMQ, MinIO, Chroma, Neo4j, Redis, pytest, React/Vitest, Docker Compose, GitNexus.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- The architecture remains **Approved design** with contract freeze active; this suite does not edit the spec or add business-contract fields.
- Keep `backend/app/services/agents/supervisor.py` as v1 and create independent `backend/app/services/agents/supervisor_v2.py`.
- Apply [`2026-09-11-langgraph-v2-agent-tool-node-amendment.md`](2026-09-11-langgraph-v2-agent-tool-node-amendment.md) before Phase 2 implementation. Its taxonomy is normative for implementation ownership and supersedes older plan wording that calls People, Document, Section, KG, Summary, or Comparison independent agents/domain graphs.
- In v2, only a component that dynamically plans/replans and selects the next capability from observations is an **agent**. Deterministic graph lifecycle/routing/evaluation steps are **nodes**; atomic domain operations are typed **capabilities/tools**; summarize/compare/compliance are **task strategies/skills**; predetermined multi-step algorithms are **workflows/subgraphs**.
- Do not create v2 `people_agent`, `summary_agent`, `comparison_agent`, `document_agent`, `section_agent`, `kg_agent`, or equivalent domain-agent wrappers. Fast and complex paths must share the same capability implementations.
- Keep chat DB as conversation truth; checkpoints hold execution/interrupt/resume only.
- Never checkpoint current identity, ACL/workspace scope, deadline, DB/client/service objects, or capability registry.
- Persist raw user text before semantic normalization.
- Every factual execution owns a checkpointed `TaskPlan`; direct non-factual responses own no plan.
- Do not implement binding/evidence/coverage/reuse until immutable revision publication passes.
- Write (pasted-text grammar/proofread/rewrite) is explicitly outside the v2 rollout scope; v1 remains its owner until a separate approved implementation plan exists.
- Keep API, SSE payloads, cancellation, citations, and frontend behavior compatible.
- Before every phase, run its repository-drift preflight: verify Modify paths and symbols exist, Create paths do not conflict, and selected dependency imports/APIs execute.
- Before editing an existing symbol, run exact GitNexus upstream impact and stop for HIGH/CRITICAL review.
- Before every commit run `node .gitnexus/run.cjs detect-changes --scope compare --base-ref main`; stage only task paths.
- Keep `.orca/` untracked.

## Exact Dependency Order

1. [`2026-09-11-langgraph-v2-phase0-benchmark.md`](2026-09-11-langgraph-v2-phase0-benchmark.md)
   - discover an exact LangGraph version exposing `StateGraph(..., context_schema=...)`;
   - discover compatible `langgraph-checkpoint-postgres` and `psycopg` versions and prove a real psycopg DSN;
   - benchmark native versus Deep Agents with actual frozen contracts; pin only the winner stack.
2. [`2026-09-11-langgraph-v2-phase1-foundation.md`](2026-09-11-langgraph-v2-phase1-foundation.md)
   - **1A:** deploy isolated raw-SQL migration runner only; no v2 ORM registration;
   - **1B:** after migration, deploy ORM/readiness mappings; startup performs no v2 DDL;
   - **1C:** move all build state and artifacts to immutable revisions, use FULL/CHAT_UPLOAD/PARSE_ONLY profiles, preserve v1 legacy retrieval until full revision-aware reindex;
   - **1D:** implement frozen contracts, evidence/checkpoint persistence, governance, and adapters.
3. [`2026-09-11-langgraph-v2-agent-tool-node-amendment.md`](2026-09-11-langgraph-v2-agent-tool-node-amendment.md)
   - define the normative `agent` vs `node` vs `capability/tool` vs `skill` vs `workflow` boundary;
   - map current AIRAG `people_agent`, `rag_agent`, `resolve_doc_agent`, evaluator, summary, and comparison concepts to v2 ownership;
   - prohibit one-domain/one-use-case agent wrappers;
   - require fast and complex execution to reuse the same capability registry;
   - reserve adaptive plan/replan/tool selection for the complex-research boundary selected in Phase 0.
4. [`2026-09-11-langgraph-v2-phase2-fast-paths.md`](2026-09-11-langgraph-v2-phase2-fast-paths.md)
   - context/binding/finalization/routing nodes;
   - deterministic one-task fast plans;
   - shared typed capabilities/tools for People, Document, Section, and KG instead of domain agents;
   - shared execution package, four-status evaluator, governed hydration/grounding, clarification;
   - independent supervisor, selector, and SSE compatibility.
5. [`2026-09-11-langgraph-v2-phase3-rollout.md`](2026-09-11-langgraph-v2-phase3-rollout.md)
   - golden A/B preflight;
   - single adaptive complex-research agent/planner using the request-scoped authorized tool catalog;
   - comparison as a task skill/policy over document/section capabilities, not a `comparison_agent`;
   - deterministic governed People→Document dependency materialization, not agent handoff;
   - bounded replan/discovery;
   - separately compiled shadow graph with isolated saver/stores;
   - live canary metrics, kill switch, and 24-hour promotion gates; rollout-control schema 1→2 is released migration-first (Task 6A) before the consumers that require it (Task 6B).

## Hard Gates

| Gate | Required evidence | Blocks |
|---|---|---|
| Phase 0 | exact versions, `inspect.signature(StateGraph)` `context_schema` proof, real AsyncPostgresSaver DSN/setup round-trip, frozen-contract parity, committed winner report | production dependency changes |
| Phase 1A | advisory-locked migration succeeds on populated legacy DB without importing v2 ORM | Phase 1B |
| Phase 1B | exact schema readiness; v2 models map pre-existing schema; legacy `AUTO_CREATE_TABLES` cannot emit v2 DDL | revision-aware producers |
| Phase 1C | revision owns every build state/artifact; R1 survives R2; dimension mismatch cannot delete published vectors; legacy remains v1-only until revision-ready | bindings/evidence/coverage |
| Phase 1D | frozen contract, EvidenceRecord/Use, ACL/retention/audit, checkpointer DSN and adapters pass | agent/tool/node amendment + `supervisor_v2.py` |
| Agent/tool/node amendment | no v2 domain/use-case agents; fast/complex share one capability implementation; complex planner receives only request-scoped authorized tool catalog | Phase 2 execution implementation |
| Phase 2 | all four evaluation statuses, blocking ambiguity finalization, checkpoint/resume, SSE/API parity, v1 default, taxonomy tests pass | complex pilots |
| Golden A/B preflight | functional comparison plus a shared-evaluator quality comparison (same evaluator version on both arms); no 24-hour claim | shadow/canary |
| Shadow | separately compiled graph + isolated saver, zero production checkpoint/evidence/audit/chat/memory/title/event writes | canary |
| Live canary | metrics-table reports, ≥200 completed samples per arm, ≥24 continuous hours, zero security violations, latency/error/cancellation gates (no quality field; quality is gated in the shared-evaluator preflight) | percentage promotion |

## Prohibited Sequencing

1. Hardcoding package pins before API compatibility discovery.
2. Registering v2 ORM before release 1A migration is applied and verified.
3. Using `Document.embed_done/captions_done/kg_done/status` to drive revision work.
4. Deleting images, tables, chunks, objects, vectors, or KG facts by document during revision build.
5. Associating legacy Chroma/KG artifacts with a baseline revision without full revision-aware reindex.
6. Recreating a workspace vector collection on dimension mismatch after published revisions exist.
7. Persisting EvidenceUse before owning TaskPlan checkpoint.
8. Exposing raw People rows to planner/checkpoint or fabricating People→Document input.
9. Compiling shadow with production checkpointer or production writable stores.
10. Treating short golden A/B output as a continuous 24-hour canary report.
11. Inner graph/domain streaming user-facing prose.
12. Publishing a lower-generation revision over a newer current revision (`current_revision_id` must be monotonic under concurrency).
13. Allocating a second draft revision for a repeated ingest trigger (`document_id` + `source_object_identity` + `build_profile` converges to one attempt).
14. Physically deleting a document's vectors/KG/objects/rows before tombstone + eligibility + GC.
15. Cross-workspace clone or reindex mutating current revision artifacts in place instead of building a target revision.
16. Treating an absent/default canary security metric as `False`/secure.
17. Deploying code that requires schema version 2 before the 1→2 migration is applied and verified.
18. Publishing a revision to a tombstoned document, or advancing `current_revision_id` without the `documents.source_deleted_at IS NULL` CAS guard (must fail closed, never resurrect).
19. Sharing one GC eligibility predicate between evidence retention and revision artifacts — evidence expiry must never by itself make revision artifacts deletable, and artifact reclamation must never delete evidence rows.
20. Resuming a terminal-`failed` revision in place, or retrying without allocating a new generation — retries must be bounded, produce `retry_of_revision_id` provenance, and a failed revision stays immutable.
21. Deriving `source_object_identity` from a user filename, client header, or otherwise non-canonical key — it must be `bucket + verbatim storage object key + version_id/etag + size + streamed sha256`, with a multipart etag never treated as a content hash.
22. Comparing a v2-internal grounded/total ratio across arms as a live quality gate (tautological and lacking a v1 counterpart) — quality must be measured by a shared evaluator on both arms in the golden preflight, and the live gate must carry no quality field.
18. Recreating v1-style People/RAG/Summary/Comparison domain agents inside v2 instead of nodes + shared capabilities/tools + task skills.
19. Giving the model control of workspace IDs, People permission, ACL, deadlines, service clients, or capability-registry construction.
20. Implementing separate fast-path and complex-agent business logic for the same capability.

## Final Execution Gate (post-amendment)

The suite is ready to execute only when the revised plans prove:

| # | Proof | Verified by |
|---|---|---|
| 1 | Revision publication is monotonic under concurrency | Phase 1 `test_concurrent_revision_publish_does_not_regress_current` |
| 2 | All ingestion triggers converge to one revision attempt | Phase 1 `test_webhook_and_confirm_create_one_revision`, `test_duplicate_webhook_is_idempotent`, `test_chat_upload_webhook_profile_is_preserved`, `test_concurrent_get_or_create_ingestion_attempt_is_atomic` |
| 3 | Delete is logical/tombstoned before GC | Phase 1 `test_delete_tombstones_before_gc`, `test_publish_after_tombstone_does_not_resurrect_current`, `test_expired_evidence_alone_does_not_release_revision_artifacts`, `test_revision_artifact_gc_requires_no_retained_references` + Task 9 independent predicates |
| 4 | Viewer APIs resolve the current revision correctly | Phase 1 `test_current_document_view_uses_current_revision` |
| 5 | Clone cannot bypass the revision lifecycle | Phase 1 Task 4 clone rule + pipeline test |
| 6 | KG facts are isolated by revision | Phase 1 `test_revision_kg_does_not_leak_old_fact` |
| 7 | Historical vectors resolve from recorded artifact metadata | Phase 1 `test_historical_revision_uses_recorded_embedding_namespace` |
| 8 | Write scope has an explicit owner | Phase 2 Global Constraints (v1 owns; out of v2 scope) |
| 9 | Evidence encryption has real key-management semantics | Phase 1 Task 8 keyring/key-ID/rotation tests |
| 10 | Canary security metrics have deterministic producers | Phase 3 Task 6B producer mapping, non-null metrics |
| 11 | Every schema migration is deployed before code requiring that version | Phase 1 A/B/C/D releases and Phase 3 Task 6A (migration-only commit) before Task 6B (consumers) |
| 12 | Failed ingestion has explicit, bounded retry semantics | Phase 1 `test_failed_revision_is_terminal_and_immutable`, `test_queue_redelivery_of_failed_revision_is_noop`, `test_retry_after_failure_allocates_new_generation`, `test_retry_is_bounded_and_exhausts` |
| 13 | `source_object_identity` is canonical and stable across triggers | Phase 1 `test_source_object_identity_is_canonical`, `test_multipart_etag_is_not_a_content_hash`, `test_overwrite_same_key_changes_identity_and_creates_new_attempt`, `test_duplicate_webhook_arrival_is_idempotent`, `test_build_profile_resolution_is_deterministic` |
| 14 | Live quality comparison is not tautological and is shared-evaluator gated | Phase 3 Task 6B excludes any quality field from live metrics; Task 1 shared evaluator + `v2-evaluator-version-fail.json` schema rejection |
| 12 | v2 domain/use-case ownership is normalized to nodes + shared capabilities/tools + skills | Agent/tool/node amendment acceptance gate |
| 13 | Fast and complex paths use the same capability implementation and runtime authorization boundary | Amendment tests + Phase 2/3 integration tests |
| 14 | Summary and comparison are task strategies, not independent agents | `test_bounded_summary_is_read_plus_synthesis_not_summary_agent`, `test_compare_is_skill_not_subagent_route` |

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

Expected: every command exits 0; the spec remains Approved/frozen and v1 remains default until a Phase-3 live gate promotes a persisted rollout control.
# LangGraph v2 Implementation Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the frozen LangGraph v2 architecture through compatibility discovery, immutable revision foundations, fast paths, complex pilots, and controlled rollout while v1 remains production default.

**Architecture:** The suite has four executable phase plans. Phase 0 discovers exact packages that support `context_schema` and proves frozen-contract parity; Phase 1 deploys migration, mappings, revision-owned ingestion, then contracts/stores/adapters in four ordered releases; Phase 2 composes v2 fast execution; Phase 3 adds complex pilots and isolated rollout.

**Tech Stack:** Python 3.11, Pydantic v2, discovered/pinned LangGraph and checkpoint stack, FastAPI, async SQLAlchemy/PostgreSQL 15, RabbitMQ, MinIO, Chroma, Neo4j, Redis, pytest, React/Vitest, Docker Compose, GitNexus.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- The architecture remains **Approved design** with contract freeze active; this suite does not edit the spec or add business-contract fields.
- Keep `backend/app/services/agents/supervisor.py` as v1 and create independent `backend/app/services/agents/supervisor_v2.py`.
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
3. [`2026-09-11-langgraph-v2-phase2-fast-paths.md`](2026-09-11-langgraph-v2-phase2-fast-paths.md)
   - context/binding/finalization/routing;
   - deterministic one-task fast plans;
   - shared execution package, four-status evaluator, governed hydration/grounding, clarification;
   - independent supervisor, selector, and SSE compatibility.
4. [`2026-09-11-langgraph-v2-phase3-rollout.md`](2026-09-11-langgraph-v2-phase3-rollout.md)
   - golden A/B preflight;
   - comparison pilot;
   - deterministic governed People→Document dependency materialization;
   - bounded replan/discovery;
   - separately compiled shadow graph with isolated saver/stores;
   - live canary metrics, kill switch, and 24-hour promotion gates.

## Hard Gates

| Gate | Required evidence | Blocks |
|---|---|---|
| Phase 0 | exact versions, `inspect.signature(StateGraph)` `context_schema` proof, real AsyncPostgresSaver DSN/setup round-trip, frozen-contract parity, committed winner report | production dependency changes |
| Phase 1A | advisory-locked migration succeeds on populated legacy DB without importing v2 ORM | Phase 1B |
| Phase 1B | exact schema readiness; v2 models map pre-existing schema; legacy `AUTO_CREATE_TABLES` cannot emit v2 DDL | revision-aware producers |
| Phase 1C | revision owns every build state/artifact; R1 survives R2; dimension mismatch cannot delete published vectors; legacy remains v1-only until revision-ready | bindings/evidence/coverage |
| Phase 1D | frozen contract, EvidenceRecord/Use, ACL/retention/audit, checkpointer DSN and adapters pass | `supervisor_v2.py` |
| Phase 2 | all four evaluation statuses, blocking ambiguity finalization, checkpoint/resume, SSE/API parity, v1 default | complex pilots |
| Golden A/B preflight | functional and quality comparison only; no 24-hour claim | shadow/canary |
| Shadow | separately compiled graph + isolated saver, zero production checkpoint/evidence/audit/chat/memory/title/event writes | canary |
| Live canary | metrics-table reports, ≥200 completed samples per arm, ≥24 continuous hours, zero security violations, latency/error/quality/cancellation gates | percentage promotion |

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

## Final Execution Gate (post-amendment)

The suite is ready to execute only when the revised plans prove:

| # | Proof | Verified by |
|---|---|---|
| 1 | Revision publication is monotonic under concurrency | Phase 1 `test_concurrent_revision_publish_does_not_regress_current` |
| 2 | All ingestion triggers converge to one revision attempt | Phase 1 `test_webhook_and_confirm_create_one_revision`, `test_duplicate_webhook_is_idempotent`, `test_chat_upload_webhook_profile_is_preserved` |
| 3 | Delete is logical/tombstoned before GC | Phase 1 `test_delete_tombstones_before_gc` + Task 9 eligibility rules |
| 4 | Viewer APIs resolve the current revision correctly | Phase 1 `test_current_document_view_uses_current_revision` |
| 5 | Clone cannot bypass the revision lifecycle | Phase 1 Task 4 clone rule + pipeline test |
| 6 | KG facts are isolated by revision | Phase 1 `test_revision_kg_does_not_leak_old_fact` |
| 7 | Historical vectors resolve from recorded artifact metadata | Phase 1 `test_historical_revision_uses_recorded_embedding_namespace` |
| 8 | Write scope has an explicit owner | Phase 2 Global Constraints (v1 owns; out of v2 scope) |
| 9 | Evidence encryption has real key-management semantics | Phase 1 Task 8 keyring/key-ID/rotation tests |
| 10 | Canary security metrics have deterministic producers | Phase 3 Task 6 producer mapping, non-null metrics |
| 11 | Every schema migration is deployed before code requiring that version | Phase 1 A/B/C/D releases and Phase 3 Task 6 1→2 two-release discipline |

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

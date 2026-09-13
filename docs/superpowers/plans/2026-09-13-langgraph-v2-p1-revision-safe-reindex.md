# LangGraph v2 P1 Revision-Safe Reindex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make revision-owned stage completion—not document-global mirror flags—the only authorization for revision verification/publication.

**Architecture:** Add migration-first per-revision stage rows, update workers idempotently by message revision, and gate finalization on stage rows plus the existing artifact manifest. Keep `Document.*_done` as v1/UI mirrors only.

**Tech Stack:** PostgreSQL 15, raw SQL migration runner, SQLAlchemy async ORM, RabbitMQ workers, pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-13-langgraph-v2-factual-retrieval-reindex-design.md`

## Global Constraints

- Migration lands and is verified before worker consumers.
- `(revision_id, stage)` is unique; state is pending/running/completed/skipped/failed.
- Worker redelivery is idempotent and cannot alter another generation.
- `Document.embed_done/captions_done/kg_done` never authorize v2 publication.
- Existing published pointers/artifacts remain copy-on-write and immutable.
- Do not commit the current reset-at-allocation diff as the sole fix.
- `_allocate_explicit_revision` has CRITICAL blast radius; rerun impact and the upload/reindex/clone/retry matrix before edits.
- Do not restart vLLM engines.

---

### Task 1: Add revision-stage schema and repository

**Files:**
- Modify: `backend/app/services/agents/v2/persistence/migrate.py`
- Create: `backend/app/models/document_revision_stage.py`
- Modify: `backend/app/models/v2_registry.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/app/services/agents/v2/persistence/document_revisions.py`
- Test: `backend/tests/migrations/v2/test_revision_stage_migration.py`
- Test: `backend/tests/agents/v2/persistence/test_revision_stages.py`

**Interfaces:**
- Produces: `DocumentRevisionStage` and repository methods `initialize_stages`, `mark_stage_running`, `mark_stage_retry_pending`, `mark_stage_completed`, `mark_stage_skipped`, `mark_stage_failed`, `required_stages_complete`.

- [ ] **Step 1: Write RED migration/repository tests**

```python
await repo.initialize_stages(revision_id, RevisionBuildProfile.FULL)
assert await repo.required_stages_complete(revision_id) is False
await repo.mark_stage_completed(revision_id, "parse")
await repo.mark_stage_completed(revision_id, "embed")
await repo.mark_stage_completed(revision_id, "caption")
await repo.mark_stage_completed(revision_id, "kg")
assert await repo.required_stages_complete(revision_id) is True
```

Also assert duplicate/redelivered transitions converge, unknown stages fail, and CHAT_UPLOAD/PARSE_ONLY use explicit skipped rows. The explicit retry edge is `running → pending` without resetting `attempt_count`; the next `pending → running` increments it. `completed`, `skipped`, and exhausted `failed` remain terminal.

- [ ] **Step 2: Run RED**

Run: `cd backend && PYTHONPATH=. pytest tests/migrations/v2/test_revision_stage_migration.py tests/agents/v2/persistence/test_revision_stages.py -q`

- [ ] **Step 3: Add idempotent raw migration and repository**

Schema columns: `revision_id UUID REFERENCES document_revisions ON DELETE RESTRICT`, `stage TEXT`, `state TEXT`, `attempt_count INTEGER`, `updated_at TIMESTAMPTZ`, `failure_class TEXT NULL`; primary/unique key `(revision_id, stage)` and CHECK constraints for exact literals/non-negative attempts.

- [ ] **Step 4: Run GREEN and commit**

```bash
cd backend && PYTHONPATH=. pytest tests/migrations/v2 tests/agents/v2/persistence -q
git add backend/app/services/agents/v2/persistence backend/app/models backend/tests/migrations/v2 backend/tests/agents/v2/persistence
git commit -m "feat(v2): add revision-owned stage state"
```

### Task 2: Initialize stage rows on allocation and retry

**Files:**
- Modify: `backend/app/queue/publisher.py`
- Modify: `backend/app/api/workers.py`
- Modify: `backend/tests/workers/test_revision_pipeline.py`

**Interfaces:**
- Consumes: `initialize_stages(revision_id, profile)`.
- Produces: atomic revision+stage initialization for ingest, reindex, clone, and retry.

- [ ] **Step 1: Preserve and inspect the pre-existing uncommitted patch**

Save its diff in this plan's ignored SDD workspace. Do not revert user files. Tests must first demonstrate that resetting global flags alone does not establish generation ownership.

- [ ] **Step 2: Write RED tests**

Cover FULL/CHAT_UPLOAD/PARSE_ONLY allocations, clone, retry, concurrent generation allocation, and stale global mirror flags. Assert new stage rows belong only to the allocated revision.

- [ ] **Step 3: Run RED, implement initialization, run GREEN**

Allocation and retry call the repository before publishing worker messages. Transaction rollback must remove both revision and stage rows. Do not let global flags satisfy `required_stages_complete`.

Run: `cd backend && PYTHONPATH=. pytest tests/workers/test_revision_pipeline.py tests/api/test_workers.py -q`

- [ ] **Step 4: Commit narrow paths**

```bash
git add backend/app/queue/publisher.py backend/app/api/workers.py backend/tests/workers/test_revision_pipeline.py backend/tests/api/test_workers.py
git commit -m "feat(worker): initialize revision stage state"
```

### Task 3: Make workers report revision-owned stages

**Files:**
- Modify: `backend/app/workers/parse_worker.py`
- Modify: `backend/app/workers/embed_worker.py`
- Modify: `backend/app/workers/caption_worker.py`
- Modify: `backend/app/workers/kg_worker.py`
- Modify: `backend/app/queue/connection.py`
- Modify: `backend/app/services/agents/v2/persistence/document_revisions.py`
- Modify: `backend/tests/agents/v2/persistence/test_revision_stages.py`
- Modify: `backend/tests/workers/test_revision_pipeline.py`

**Interfaces:**
- Produces: each worker transitions only its message revision/stage.

- [ ] **Step 1: Write RED tests**

For each worker assert running→completed, retry increments attempt count, exhausted retry marks failed, duplicate completion is idempotent, and an old revision message cannot touch the newer revision.

- [ ] **Step 2: Run RED and implement minimal calls**

Each worker records running before work and completed only after its artifact transaction succeeds. Workers acquire the running edge through an atomic repository claim that distinguishes a newly claimed `pending → running` attempt from an already-`running` duplicate. A duplicate/in-flight or terminal-stage delivery returns before performing work or mutating mirrors; it must not swallow a transition result/error and continue. Profile skips are initialized, not guessed by workers. A retryable worker exception must not terminalize the revision before the queue decides exhaustion. Before requeue, the queue path calls `mark_stage_retry_pending` only for that message revision/stage; the next worker remains executable and `mark_stage_running` increments `attempt_count`. On exhaustion, the queue atomically calls `mark_stage_failed` and terminalizes only that revision. Queue retry paths update only the failed message revision. Handler-level tests—not source-string checks alone—must exercise all four workers' terminal no-op behavior, running/work/commit/completed ordering, retry attempt bump, and queue stage mapping.

- [ ] **Step 3: Run GREEN and commit**

```bash
cd backend && PYTHONPATH=. pytest tests/workers -q
git add backend/app/workers backend/app/queue/connection.py backend/tests/workers
git commit -m "feat(worker): report revision-owned completion"
```

### Task 4: Gate finalization on revision stages and run reindex proof

**Files:**
- Modify: `backend/app/workers/utils.py`
- Modify: `backend/tests/workers/test_revision_pipeline.py`
- Modify: `docs/runbooks/langgraph-v2-corpus-reindex.md`
- Modify: `docs/workers.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Produces: revision-stage-authoritative `check_and_finalize`.

- [ ] **Step 1: Write RED concurrency tests**

Assert pending/running returns NOT_READY; all required complete + manifest publishes once; complete + incomplete manifest fails only that revision; stale flags never finalize; concurrent finalizers preserve CAS/current pointer.

- [ ] **Step 2: Implement finalization gate**

When `revision_id` is supplied, load profile/stage rows/manifest and call `finalize_revision_if_complete(expect_complete=True)` only after `required_stages_complete` is true. Legacy calls without revision retain mirror behavior.

- [ ] **Step 3: Run full worker/migration gate and commit**

```bash
cd backend && PYTHONPATH=. pytest tests/workers tests/migrations/v2 tests/agents/v2/persistence -q
git add backend/app/workers/utils.py backend/tests/workers/test_revision_pipeline.py docs/runbooks/langgraph-v2-corpus-reindex.md docs/workers.md CLAUDE.md
git commit -m "fix(v2): finalize only complete revisions"
```

- [ ] **Step 4: Operational proof**

Apply migration to the test stack, rebuild/restart backend and relevant workers only, reindex one non-deleted document, verify stage rows + complete manifest + published pointer, then run P0 hard-scoped factual SSE. Never manually set the pointer and never restart vLLM.

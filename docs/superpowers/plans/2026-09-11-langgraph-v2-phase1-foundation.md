# LangGraph v2 Phase 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish immutable revision storage, canonical v2 contracts, governed evidence/checkpoint persistence, and typed adapters before any v2 graph executes.

**Architecture:** Apply the first v2 DDL from an isolated raw-SQL migration before importing/registering any v2 ORM model into startup metadata. On a populated legacy database the locked migration adds nullable revision links, allocates and backfills one baseline revision per reconstructable document (including images/tables), validates the backfill, and only then installs foreign keys/non-null/current-pointer constraints; a subsequent code step registers ORM mappings that match the already-migrated schema. Ingestion then allocates drafts, builds revision-qualified artifacts, verifies them, and atomically publishes; retrieval always selects an explicit published revision.

**Tech Stack:** Python 3.11, Pydantic v2, async SQLAlchemy/PostgreSQL 15, RabbitMQ, MinIO, Chroma, AsyncPostgresSaver, pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- Phase 0 winner and exact runtime/checkpointer pins must be committed first.
- Do not implement binding/evidence/coverage against mutable current-document artifacts.
- The initial v2 migration must run before v2 models or revision columns are imported into `Base.metadata`; unlocked startup `create_all()` must never create/mutate v2 tables or columns.
- Every path that can select v2 must call the same schema compatibility check first.
- Published revisions and their artifacts are immutable; reindex is copy-on-write.
- Before editing an existing symbol, run the qualified GitNexus impact command named by the task; stop on HIGH/CRITICAL risk.
- Before each commit run compare-scope `detect-changes` and stage only that task's paths.

---

### Task 1: Install the Isolated Populated-Database Migration Before ORM Registration

**Files:**
- Create: `backend/app/services/agents/v2/persistence/migrate.py`
- Create: `backend/tests/migrations/v2/test_populated_legacy_migration.py`
- Test: `backend/tests/migrations/v2/test_migration_control.py`

**Interfaces:**
- Produces: `V2_SCHEMA_VERSION = 1`, `check_v2_schema(engine) -> SchemaCheck`, and `apply_v2_schema(engine) -> None` implemented without importing `app.models.v2_registry` or adding v2 tables/columns to startup metadata.

- [ ] **Step 1: Write the populated-legacy-schema integration test**

Create a PostgreSQL test schema using the pre-v2 `documents`, `document_images`, and `document_tables` shape; insert two documents plus image/table children and current mutable artifact metadata. Run `apply_v2_schema`, then assert: one published baseline revision per reconstructable document; every child row points to its document's baseline revision; `documents.current_revision_id` points to it; all three link columns are non-null and foreign-keyed only after backfill; an unreconstructable document is explicitly marked unavailable rather than falsely pinned; rerun is idempotent. Instrument executed SQL and assert `ADD COLUMN ... NULL` occurs before baseline allocation/update, validation queries return zero or abort, and `SET NOT NULL`/FK/current-pointer enforcement occurs last.

Run `cd backend && pytest tests/migrations/v2/test_populated_legacy_migration.py tests/migrations/v2/test_migration_control.py -q`; expect import failure.

- [ ] **Step 2: Implement the locked migration in exact safe order**

`apply_v2_schema` uses SQLAlchemy text/DDL only, takes `pg_advisory_xact_lock`, and never imports `app.models` or calls `Base.metadata.create_all`. In one transaction it: (1) creates version/revision/structure tables; (2) adds nullable `documents.current_revision_id`, `document_images.revision_id`, and `document_tables.revision_id`; (3) allocates a deterministic baseline revision for each reconstructable existing document and snapshots immutable workspace/artifact/structure ownership; (4) updates every existing image/table and current pointer; (5) aborts unless orphan/null/mismatched-workspace validation queries return zero; (6) installs FKs, unique indexes, and `NOT NULL` on image/table revision links; and (7) writes schema version 1. Documents that cannot produce trustworthy stable locators remain unavailable to v2 and have a null current pointer; the document pointer therefore remains nullable by design.

- [ ] **Step 3: Prove no startup metadata registration exists yet**

Add a source assertion that importing `app.models` before migration exposes no v2 table and no mapped revision columns. This is the deploy gate: run `python -m app.services.agents.v2.persistence.migrate --apply` successfully on the populated database before deploying Task 2 code.

- [ ] **Step 4: Run and commit**

```bash
cd backend && pytest tests/migrations/v2/test_populated_legacy_migration.py tests/migrations/v2/test_migration_control.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/persistence/migrate.py backend/tests/migrations/v2/test_populated_legacy_migration.py backend/tests/migrations/v2/test_migration_control.py
git commit -m "feat: migrate populated databases to revision storage"
```

---

### Task 2: Register ORM Models Only After Schema Version 1 Is Applied

**Files:**
- Create: `backend/app/models/v2_registry.py`
- Create: `backend/app/models/document_revision.py`
- Create: `backend/app/models/document_revision_chunk.py`
- Create: `backend/app/models/conversation_snapshot.py`
- Create: `backend/app/models/semantic_snapshot.py`
- Create: `backend/app/models/binding_audit.py`
- Create: `backend/app/models/evidence_record.py`
- Create: `backend/app/models/evidence_use.py`
- Modify: `backend/app/models/document.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/migrations/v2/test_model_metadata.py`

**Interfaces:**
- Consumes: a database already at exact v2 schema version 1 from Task 1.
- Produces: ORM mappings matching the migrated schema and a startup legacy-table allowlist that cannot issue v2 DDL.

- [ ] **Step 1: Impact-check startup and legacy model mappings**

```bash
impact({target: "app.main.lifespan", direction: "upstream"})
impact({target: "app.models.document.Document", direction: "upstream"})
impact({target: "app.models.document.DocumentImage", direction: "upstream"})
impact({target: "app.models.document.DocumentTable", direction: "upstream"})
```

- [ ] **Step 2: Write metadata/readiness tests**

Assert model nullability matches the completed migration (`Document.current_revision_id` nullable; image/table revision IDs non-null), all FKs resolve, exact schema version 1 is required before application startup imports/uses repositories, and `LEGACY_STARTUP_TABLES` excludes every v2 table. Against a fresh legacy schema, importing models must not be followed by startup mutation: lifespan reports the migration command and exits before `create_all` if required mapped columns/version are absent.

- [ ] **Step 3: Register post-migration ORM mappings and gate startup**

Define the revision and v2 persistence models, map the already-existing columns, and register them in `app.models.__init__` only in this post-migration deployment step. Replace unrestricted `Base.metadata.create_all()` with `Base.metadata.create_all(tables=LEGACY_STARTUP_TABLES)`. `lifespan` checks exact v2 schema compatibility before model-backed v2 services initialize; it never invokes v2 migration or DDL. EvidenceUse null-safe uniqueness is represented as an index matching Task-1 SQL.

- [ ] **Step 4: Test and commit**

```bash
cd backend && pytest tests/migrations/v2/test_populated_legacy_migration.py tests/migrations/v2/test_model_metadata.py tests/migrations/v2/test_migration_control.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/models/v2_registry.py backend/app/models/document_revision.py backend/app/models/document_revision_chunk.py backend/app/models/conversation_snapshot.py backend/app/models/semantic_snapshot.py backend/app/models/binding_audit.py backend/app/models/evidence_record.py backend/app/models/evidence_use.py backend/app/models/document.py backend/app/models/__init__.py backend/app/main.py backend/tests/migrations/v2/test_model_metadata.py
git commit -m "feat: register post-migration v2 models"
```

---

### Task 3: Implement Draft Allocation and Atomic Publication Repository

**Files:**
- Create: `backend/app/services/agents/v2/persistence/document_revisions.py`
- Test: `backend/tests/agents/v2/persistence/test_document_revisions.py`

**Interfaces:**
- Produces: `allocate_draft`, `record_artifacts`, `verify_draft`, `publish`, `get_published`, `mark_source_deleted`.

- [ ] **Step 1: Write lifecycle tests**

Test `allocate → build metadata → verify → publish`, rejection when any object/vector/structure artifact is missing, publish transaction updating `Document.current_revision_id`, immutable published columns, failed draft leaving prior current revision unchanged, and tombstone lookup.

- [ ] **Step 2: Implement explicit lifecycle**

```python
async def publish(self, revision_id: UUID) -> DocumentRevision:
    revision = await self.get_for_update(revision_id)
    if revision.state != "verified":
        raise RevisionNotPublishable(str(revision_id))
    document = await self.get_document_for_update(revision.document_id)
    revision.state = "published"
    revision.published_at = datetime.now(timezone.utc)
    document.current_revision_id = revision.revision_id
    await self.session.flush()
    return revision
```

`verify_draft` checks raw, markdown, structure manifest, images/tables/chunks ownership, and exact vector namespace count before transitioning to verified.

- [ ] **Step 3: Test and commit**

```bash
cd backend && pytest tests/agents/v2/persistence/test_document_revisions.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/persistence/document_revisions.py backend/tests/agents/v2/persistence/test_document_revisions.py
git commit -m "feat: add immutable revision publication lifecycle"
```

---

### Task 4: Propagate revision_id Through Every Ingestion Producer and Worker

**Files:**
- Modify: `backend/app/api/documents.py`
- Modify: `backend/app/api/minio_events.py`
- Modify: `backend/app/api/rag.py`
- Modify: `backend/app/queue/messages.py`
- Modify: `backend/app/queue/publisher.py`
- Modify: `backend/app/workers/parse_worker.py`
- Modify: `backend/app/workers/caption_worker.py`
- Modify: `backend/app/workers/embed_worker.py`
- Modify: `backend/app/workers/kg_worker.py`
- Modify: `backend/app/workers/utils.py`
- Test: `backend/tests/workers/test_revision_pipeline.py`

**Interfaces:**
- Produces: fresh upload/reindex/minio event allocates one draft before publishing `ParseMessage`; all child messages preserve the same required UUID.

- [ ] **Step 1: Impact-check all producers/consumers with qualified names**

```bash
impact({target: "app.api.documents.upload_document", direction: "upstream"})
impact({target: "app.api.documents.presign_upload", direction: "upstream"})
impact({target: "app.api.documents.confirm_upload", direction: "upstream"})
impact({target: "app.api.documents.process_document_background", direction: "upstream"})
impact({target: "app.api.documents._clone_document_to_workspace", direction: "upstream"})
impact({target: "app.api.documents.delete_document", direction: "upstream"})
impact({target: "app.api.minio_events.handle_minio_event", direction: "upstream"})
impact({target: "app.api.rag.reindex_document", direction: "upstream"})
impact({target: "app.queue.messages.ParseMessage", direction: "upstream"})
impact({target: "app.queue.messages.CaptionMessage", direction: "upstream"})
impact({target: "app.queue.messages.EmbedMessage", direction: "upstream"})
impact({target: "app.queue.messages.KGMessage", direction: "upstream"})
impact({target: "app.queue.publisher.publish_parse_task", direction: "upstream"})
impact({target: "app.workers.parse_worker.handle_parse", direction: "upstream"})
impact({target: "app.workers.caption_worker.handle_caption", direction: "upstream"})
impact({target: "app.workers.embed_worker.handle_embed", direction: "upstream"})
impact({target: "app.workers.kg_worker.handle_kg", direction: "upstream"})
impact({target: "app.workers.utils.check_and_finalize", direction: "upstream"})
```

- [ ] **Step 2: Write constructor and stale-message tests**

Discover every constructor with `rg 'ParseMessage\(|CaptionMessage\(|EmbedMessage\(|KGMessage\(' backend/app` and make the test fail if an unclassified constructor remains. Parameterize `upload_document`, `presign_upload`→`confirm_upload`, `process_document_background`, `_clone_document_to_workspace`, `handle_minio_event`, `reindex_document`, `publish_parse_task`, and every worker child publish, asserting required `revision_id`. Assert each fresh upload/confirmed upload/clone/reindex allocates a draft before first publish and stale messages cannot publish a newer revision.

- [ ] **Step 3: Implement allocation and propagation**

All four message models require `revision_id: UUID` with no default. `documents.py`, `minio_events.py`, and `rag.py` call `allocate_draft` before the first queue publish. Workers load exactly that draft, write revision-owned rows/artifacts, and publish child messages with the same UUID. Finalization executes `record_artifacts → verify_draft → publish`; failures mark only that draft failed.

- [ ] **Step 4: Compile, test, and commit**

```bash
cd backend && python -m compileall app/api app/queue app/workers
cd backend && pytest tests/workers/test_revision_pipeline.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/api/documents.py backend/app/api/minio_events.py backend/app/api/rag.py backend/app/queue/messages.py backend/app/queue/publisher.py backend/app/workers/parse_worker.py backend/app/workers/caption_worker.py backend/app/workers/embed_worker.py backend/app/workers/kg_worker.py backend/app/workers/utils.py backend/tests/workers/test_revision_pipeline.py
git commit -m "feat: propagate revision through ingestion pipeline"
```

---

### Task 5: Make Artifact Storage and Retrieval Revision-Selected

**Files:**
- Modify: `backend/app/services/storage_service.py`
- Modify: `backend/app/services/embedding/vector_store.py`
- Modify: `backend/app/services/retrieval/deep_retriever.py`
- Modify: `backend/app/services/retrieval/rag_service.py`
- Modify: `backend/app/services/retrieval/hrag_service.py`
- Modify: `backend/app/services/agent/tools.py`
- Modify: `backend/app/services/kg/legal_kg_service.py`
- Test: `backend/tests/agents/v2/persistence/test_revision_artifacts.py`
- Test: `backend/tests/agents/v2/persistence/test_revision_live_callers.py`
- Test: `backend/tests/api/test_revision_reindex_delete.py`

**Interfaces:**
- Produces: revision-qualified keys/IDs, selected-revision retrieval, stable ContentLocator reconstruction, non-destructive reindex.

- [ ] **Step 1: Impact-check storage/vector/retrieval/delete symbols**

```bash
impact({target: "app.services.storage_service.StorageService.upload_markdown", direction: "upstream"})
impact({target: "app.services.storage_service.StorageService.delete_markdown", direction: "upstream"})
impact({target: "app.services.storage_service.StorageService.delete_file", direction: "upstream"})
impact({target: "app.services.embedding.vector_store.VectorStore.add_documents", direction: "upstream"})
impact({target: "app.services.embedding.vector_store.VectorStore.query", direction: "upstream"})
impact({target: "app.services.embedding.vector_store.VectorStore.delete_by_document_id", direction: "upstream"})
impact({target: "app.services.retrieval.deep_retriever.DeepRetriever._vector_query", direction: "upstream"})
impact({target: "app.services.retrieval.deep_retriever.DeepRetriever._bm25_query", direction: "upstream"})
impact({target: "app.services.retrieval.deep_retriever.DeepRetriever._rrf_merge", direction: "upstream"})
impact({target: "app.services.retrieval.deep_retriever.DeepRetriever._assemble_context", direction: "upstream"})
impact({target: "app.services.retrieval.rag_service.RAGService.delete_document", direction: "upstream"})
impact({target: "app.services.retrieval.hrag_service.HRAGService.query", direction: "upstream"})
impact({target: "app.services.retrieval.hrag_service.HRAGService.query_deep", direction: "upstream"})
impact({target: "app.services.agent.tools.search_documents", direction: "upstream"})
impact({target: "app.services.agent.tools.summarize_document", direction: "upstream"})
impact({target: "app.services.agent.tools.get_documents_content", direction: "upstream"})
impact({target: "app.services.agent.tools.search_document_section", direction: "upstream"})
impact({target: "app.services.kg.legal_kg_service.LegalKGService.ingest", direction: "upstream"})
impact({target: "app.services.kg.legal_kg_service.LegalKGService.get_relevant_context", direction: "upstream"})
impact({target: "app.services.kg.legal_kg_service.LegalKGService.delete_document", direction: "upstream"})
impact({target: "app.api.documents.delete_document", direction: "upstream"})
```

- [ ] **Step 2: Write revision-selection tests**

Tests assert old artifacts survive reindex; failed draft does not replace current; exact pinned retrieval filters SQL/Chroma/BM25 by selected `revision_id`; current retrieval first resolves `Document.current_revision_id`; image/table/chunk rows belong to that revision; mixed-revision merge is rejected; locator reconstructs from stable `structure_node_id`. `test_revision_live_callers.py` directly exercises `HRAGService.query`, `HRAGService.query_deep`, `search_documents`, `summarize_document`, `get_documents_content`, and `search_document_section`: v2 callers must pass the selected revision through every layer, while the v1 adapter must resolve `Document.current_revision_id` once and preserve today's current-document behavior without accepting an arbitrary client revision.

- [ ] **Step 3: Implement revision-qualified artifacts**

Use `documents/{document_id}/revisions/{revision_id}/...` object keys and `doc_{document_id}_rev_{revision_id}_node_{structure_node_id}` vector IDs. Every vector metadata row contains revision and structure node. Adapt `HRAGService` and agent tool entrypoints to call a shared revision-selected retrieval port. The v2 port requires `revision_id`; a named `CurrentDocumentRetrievalAdapter` used only by v1 resolves `Document.current_revision_id` server-side and then calls that same port. Every vector/BM25/KG retrieval entrypoint filters the selected revision; no fallback to mutable chunks or document-wide KG facts occurs after selection. Reindex no longer deletes old markdown/vector artifacts before build.

- [ ] **Step 4: Test and commit**

```bash
cd backend && pytest tests/agents/v2/persistence/test_revision_artifacts.py tests/agents/v2/persistence/test_revision_live_callers.py tests/api/test_revision_reindex_delete.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/storage_service.py backend/app/services/embedding/vector_store.py backend/app/services/retrieval/deep_retriever.py backend/app/services/retrieval/rag_service.py backend/app/services/retrieval/hrag_service.py backend/app/services/agent/tools.py backend/app/services/kg/legal_kg_service.py backend/tests/agents/v2/persistence/test_revision_artifacts.py backend/tests/agents/v2/persistence/test_revision_live_callers.py backend/tests/api/test_revision_reindex_delete.py
git commit -m "feat: select immutable revisions for retrieval"
```

---

### Task 6: Implement Canonical Contracts and Pure Validators

**Files:**
- Create: `backend/app/services/agents/v2/contracts/`
- Create: `backend/tests/agents/v2/contracts/`

**Interfaces:**
- Produces: exact frozen contracts and `validate_*` functions from the approved spec.

- [ ] **Step 1: Write contract tests**

Cover strict/frozen/envelope versioning, optional revision semantics, ID/DAG/relation integrity, criteria uniqueness, EvidenceRecord/EvidenceUse separation, purpose/target rules, admitted claim use IDs, TaskExecutionSummary/replan context, and incompatible checkpoint rejection.

- [ ] **Step 2: Implement owner-focused modules**

Create `base.py`, `request.py`, `conversation.py`, `semantic.py`, `locators.py`, `binding.py`, `routing.py`, `planning.py`, `execution.py`, `evidence.py`, `evaluation.py`, `synthesis.py`, `clarification.py`, `response.py`, `state.py`, and `validation.py`. Define every union only after its variants and call `model_rebuild()` explicitly.

- [ ] **Step 3: Test and commit**

```bash
cd backend && pytest tests/agents/v2/contracts -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/contracts backend/tests/agents/v2/contracts
git commit -m "feat: add canonical v2 contracts"
```

---

### Task 7: Implement Snapshot and Audit Repositories

**Files:**
- Create: `backend/app/services/agents/v2/persistence/snapshots.py`
- Create: `backend/app/services/agents/v2/persistence/binding_audit.py`
- Test: `backend/tests/agents/v2/persistence/test_snapshots.py`

**Interfaces:**
- Produces: ConversationSnapshot CAS, SemanticSnapshot persistence, binding audit append/read.

- [ ] **Step 1: Write CAS/version/audit tests**

Test stale summary versions fail, raw ChatMessage remains authoritative, incompatible snapshot versions reject, and binding audit is not needed to resolve hot-path revision policy.

- [ ] **Step 2: Implement repositories and commit**

```bash
cd backend && pytest tests/agents/v2/persistence/test_snapshots.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/persistence/snapshots.py backend/app/services/agents/v2/persistence/binding_audit.py backend/tests/agents/v2/persistence/test_snapshots.py
git commit -m "feat: add v2 snapshot persistence"
```

---

### Task 8: Implement Evidence Repositories and Governance

**Files:**
- Create: `backend/app/services/agents/v2/persistence/evidence.py`
- Create: `backend/app/services/agents/v2/evidence_store/governance.py`
- Test: `backend/tests/agents/v2/persistence/test_evidence_governance.py`

**Interfaces:**
- Produces: idempotent record/use insertion; minimization/classification/encryption; governed audited hydration.

- [ ] **Step 1: Write governance tests**

Test null-safe EvidenceUse uniqueness, People minimization, encryption, expiry, ACL, tombstone, revision mismatch, derived validation state, and audited allow/deny reads.

- [ ] **Step 2: Implement fail-closed repositories**

Insert EvidenceUse with `ON CONFLICT (run_id, task_id, evidence_id, purpose, target_id) DO NOTHING`; PostgreSQL infers the matching `NULLS NOT DISTINCT` unique index, including targetless uses. Then select and return the existing UUID by the same five-column null-safe key. Resolve document workspace from immutable revision, never copied evidence metadata. Decrypt only after ACL/expiry/source/validation checks.

- [ ] **Step 3: Test and commit**

```bash
cd backend && pytest tests/agents/v2/persistence/test_evidence_governance.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/persistence/evidence.py backend/app/services/agents/v2/evidence_store/governance.py backend/tests/agents/v2/persistence/test_evidence_governance.py
git commit -m "feat: add governed evidence persistence"
```

---

### Task 9: Implement Executable Evidence and Revision Artifact GC

**Files:**
- Create: `backend/app/services/agents/v2/evidence_store/gc.py`
- Create: `backend/app/workers/evidence_gc_worker.py`
- Modify: `backend/app/services/storage_service.py`
- Modify: `backend/app/services/embedding/vector_store.py`
- Modify: `docker-compose.services.yml`
- Test: `backend/tests/agents/v2/persistence/test_evidence_gc.py`
- Test: `backend/tests/workers/test_evidence_gc_worker.py`

**Interfaces:**
- Produces: `run_gc_batch`; concrete object/vector artifact delete methods; scheduled one-shot worker.

- [ ] **Step 1: Impact-check destructive methods**

```bash
impact({target: "app.services.storage_service.StorageService.delete_markdown", direction: "upstream"})
impact({target: "app.services.storage_service.StorageService.delete_file", direction: "upstream"})
impact({target: "app.services.embedding.vector_store.VectorStore.delete_by_document_id", direction: "upstream"})
```

- [ ] **Step 2: Write deletion-order/idempotency tests**

Test advisory lock + `FOR UPDATE SKIP LOCKED`; expired ciphertext/use deletion and audit; revision artifacts retained while an unexpired evidence record references them; object/manifest/image/table/vector deletion only after eligibility; retry after partial object-store failure; two workers do not double-delete.

- [ ] **Step 3: Implement concrete deletion APIs and worker**

Add `StorageService.delete_revision_artifacts(document_id, revision_id)` and `VectorStore.delete_revision(document_id, revision_id)`. `run_gc_batch` marks a GC lease in DB, deletes external artifacts idempotently, records each completion, then finalizes DB deletion. `python -m app.workers.evidence_gc_worker --once --batch-size 100` is the executable entrypoint. Add a dedicated `evidence-gc` Compose service with a documented periodic command/interval and migration dependency.

- [ ] **Step 4: Test and commit**

```bash
cd backend && pytest tests/agents/v2/persistence/test_evidence_gc.py tests/workers/test_evidence_gc_worker.py -q
docker compose -f docker-compose.services.yml config --quiet
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/evidence_store/gc.py backend/app/workers/evidence_gc_worker.py backend/app/services/storage_service.py backend/app/services/embedding/vector_store.py docker-compose.services.yml backend/tests/agents/v2/persistence/test_evidence_gc.py backend/tests/workers/test_evidence_gc_worker.py
git commit -m "feat: add evidence and revision garbage collection"
```

---

### Task 10: Install and Initialize Async PostgreSQL Checkpointing

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/app/services/agents/v2/persistence/checkpoint.py`
- Test: `backend/tests/agents/v2/persistence/test_checkpoint.py`

**Interfaces:**
- Produces: `create_v2_checkpointer(database_url) -> AsyncPostgresSaver`, `setup_v2_checkpointer(database_url) -> None`.

- [ ] **Step 1: Verify Phase-0 exact pins and write failing tests**

Assert `backend/requirements.txt` contains the Phase-0-recorded exact `langgraph-checkpoint-postgres` and `psycopg[binary,pool]` pins. Tests reject non-Postgres URLs, call saver `setup()` during migration/deployment, pass `thread_id` via configurable metadata, and reject incompatible state versions.

- [ ] **Step 2: Implement async lifecycle**

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

@asynccontextmanager
async def create_v2_checkpointer(database_url: str):
    async with AsyncPostgresSaver.from_conn_string(database_url) as saver:
        yield saver

async def setup_v2_checkpointer(database_url: str) -> None:
    async with create_v2_checkpointer(database_url) as saver:
        await saver.setup()
```

Add `checkpoint --setup` and `checkpoint --check` CLI modes; check verifies saver tables without mutating them. Never instantiate or migrate at module import.

- [ ] **Step 3: Test and commit**

```bash
cd backend && python -m pip check
cd backend && pytest tests/agents/v2/persistence/test_checkpoint.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/requirements.txt backend/app/services/agents/v2/persistence/checkpoint.py backend/tests/agents/v2/persistence/test_checkpoint.py
git commit -m "feat: add async PostgreSQL v2 checkpointing"
```

---

### Task 11: Add Typed Legacy Adapters and Capability Ports

**Files:**
- Create: `backend/app/services/agents/v2/adapters/`
- Create: `backend/app/services/agents/v2/capabilities/`
- Test: `backend/tests/agents/v2/test_adapters_and_registry.py`

**Interfaces:**
- Produces: typed semantic/conversation/document/deep-research adapters and ACL-filtered capability registry.

- [ ] **Step 1: Write adapter tests**

Require raw query ownership, Draft→Binding→Finalizer flow, immutable revision lookup, typed outputs without dictionaries, runtime capability intersection, and validation rather than v1 `model_construct()` loading.

- [ ] **Step 2: Implement protocols/adapters and commit**

```python
class Capability(Protocol):
    descriptor: CapabilityDescriptor
    async def execute(self, request: AgentRequest, runtime: CapabilityRuntimeContext) -> AgentResult: ...
```

```bash
cd backend && pytest tests/agents/v2/test_adapters_and_registry.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/adapters backend/app/services/agents/v2/capabilities backend/tests/agents/v2/test_adapters_and_registry.py
git commit -m "feat: add typed v2 adapters and capabilities"
node .gitnexus/run.cjs analyze
```

## Phase 1 §26 Gate

Phase 1 owns contract/persistence acceptance only; People→Document, synthesis budget overflow, and synthesis-only reuse are not claimed here.

```bash
docker exec hrag-backend python -m app.services.agents.v2.persistence.migrate --check
docker exec hrag-backend python -m app.services.agents.v2.persistence.checkpoint --check
docker exec hrag-backend pytest tests/migrations/v2 tests/agents/v2/contracts tests/agents/v2/persistence tests/workers/test_revision_pipeline.py tests/workers/test_evidence_gc_worker.py -q
docker exec hrag-backend python -m compileall app/services/agents/v2
docker compose -f docker-compose.services.yml config --quiet
```

Required named coverage: envelope/version strictness, binding/target/task/evidence/use integrity, immutable revision pin/current semantics, revision-requirement relation, EvidenceUse purpose/target rules, governance/expiry/ACL/audit, prompt data isolation at adapters, and incompatible checkpoint rejection.

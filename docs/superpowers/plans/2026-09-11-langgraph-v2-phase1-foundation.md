# LangGraph v2 Phase 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish immutable revision storage, canonical v2 contracts, governed evidence/checkpoint persistence, and typed adapters before any v2 graph executes.

**Architecture:** Deliver four ordered releases: 1A deploys an isolated raw-SQL migration runner with no v2 ORM registration; 1B registers mappings only after schema readiness and never runs v2 startup DDL; 1C moves processing state and all SQL/object/vector/KG artifacts to immutable revisions with explicit build profiles; 1D adds frozen contracts, governed evidence/checkpoint persistence, and adapters. Existing legacy artifacts remain v1-only until a full revision-aware reindex publishes the first v2-ready revision.

**Tech Stack:** Python 3.11, Pydantic v2, async SQLAlchemy/PostgreSQL 15, RabbitMQ, MinIO, Chroma, AsyncPostgresSaver, pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- Phase 0 winner and exact runtime/checkpointer pins must be committed first.
- Do not implement binding/evidence/coverage against mutable current-document artifacts.
- Release 1A migration runner must deploy and apply before release 1B imports v2 mappings into `Base.metadata`; unlocked startup `create_all()` must never create/mutate v2 tables or columns.
- Every path that can select v2 must call the same schema compatibility check first.
- Published revisions and their artifacts are immutable; reindex is copy-on-write.
- Before editing an existing symbol, run the qualified GitNexus impact command named by the task; stop on HIGH/CRITICAL risk.
- Before each commit run compare-scope `detect-changes` and stage only that task's paths.

---

### Task 0: Verify Phase-1 Repository and Dependency Preconditions

**Files:**
- Read: all Modify paths and named symbols in this plan
- Test: shell preflight only

**Interfaces:**
- Produces: a recorded preflight manifest proving paths/symbols/imports before Release 1A begins.

- [ ] **Step 1: Verify repository paths and module conflicts**

```bash
set -e
for path in backend/app/main.py backend/app/models/document.py backend/app/api/documents.py backend/app/api/minio_events.py backend/app/api/rag.py backend/app/queue/messages.py backend/app/workers/parse_worker.py backend/app/workers/caption_worker.py backend/app/workers/embed_worker.py backend/app/workers/kg_worker.py backend/app/services/storage_service.py backend/app/services/embedding/vector_store.py backend/app/services/retrieval/hrag_service.py backend/app/services/agent/tools.py; do test -e "$path"; done
test ! -e backend/app/services/agents/v2/persistence/migrate.py
test ! -e backend/app/models/document_revision.py
rg -n 'embed_done|captions_done|kg_done|raw_chunks_json|markdown_s3_key|delete.*Document(Image|Table)|delete_by_document_id|dimension' backend/app
python - <<'PY'
import re, pathlib
plan = pathlib.Path('docs/superpowers/plans/2026-09-11-langgraph-v2-phase1-foundation.md').read_text()
modify = [p.split(':')[0] for p in re.findall(r'^- Modify: `([^`]+)`', plan, re.M)]
create = [p.split(':')[0] for p in re.findall(r'^- Create: `([^`]+)`', plan, re.M)]
missing = [p for p in modify if not pathlib.Path(p).exists()]
conflict = [p for p in create if pathlib.Path(p).exists()]
assert not missing and not conflict, {'missing': missing, 'conflict': conflict}
print(f'phase1 paths ok: {len(set(modify))} modify, {len(set(create))} create')
PY
```

Expected: every Modify path exists, Create paths do not conflict, and output captures all legacy state/destructive seams. Stop on drift.

- [ ] **Step 2: Verify selected dependencies and exact symbols**

```bash
cd backend && python - <<'PY'
from inspect import signature
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
assert "context_schema" in signature(StateGraph).parameters
print(AsyncPostgresSaver)
PY
node .gitnexus/run.cjs query --query "document upload parse caption embed KG reindex retrieval deletion lifecycle" || true
```

Expected: imports/API pass; record exact GitNexus-returned symbol names and use those—not guessed names—in each later impact command.

- [ ] **Step 3: Confirm clean phase start**

```bash
git diff --check
git status --short
```

Expected: only intentional plan/execution changes; `.orca/` remains untracked.

---

## Release 1A — Raw-SQL Migration Runner Only

### Task 1: Install the Isolated Populated-Database Migration Before ORM Registration

**Files:**
- Create: `backend/app/services/agents/v2/persistence/migrate.py`
- Create: `backend/tests/migrations/v2/test_populated_legacy_migration.py`
- Test: `backend/tests/migrations/v2/test_migration_control.py`

**Interfaces:**
- Produces: `V2_SCHEMA_VERSION = 1`, `check_v2_schema(engine) -> SchemaCheck`, and `apply_v2_schema(engine) -> None` implemented without importing `app.models.v2_registry` or adding v2 tables/columns to startup metadata.

- [ ] **Step 1: Write the populated-legacy-schema integration test**

Create a PostgreSQL test schema using populated pre-v2 `documents`, `document_images`, and `document_tables`. Run `apply_v2_schema`, then assert: schema/version and empty revision tables exist; nullable revision links/current pointer exist; every legacy row remains untouched and v1-readable; no legacy Chroma/KG/object/SQL artifact is falsely marked revision-ready; all legacy `current_revision_id` values remain null until full revision-aware reindex; rerun is idempotent. Instrument SQL and assert the migration takes its advisory lock before DDL and never imports ORM metadata, calls `create_all`, deletes legacy rows, or invents a baseline published revision.

Run `cd backend && pytest tests/migrations/v2/test_populated_legacy_migration.py tests/migrations/v2/test_migration_control.py -q`; expect import failure.

- [ ] **Step 2: Implement the locked migration in exact safe order**

`apply_v2_schema` uses SQLAlchemy text/DDL only, takes `pg_advisory_xact_lock`, and never imports `app.models` or calls `Base.metadata.create_all`. In one transaction it: (1) creates version/revision/build/structure tables, including implementation-only metadata: `document_revisions.generation BIGINT` (per-document monotonic allocation counter, unique per document), `document_revisions.retry_of_revision_id UUID NULL` (retry provenance), `document_revisions.failed_at TIMESTAMPTZ NULL`/`failure_stage TEXT NULL`/`failure_class TEXT NULL` (terminal failure), `document_revisions.abandoned_at TIMESTAMPTZ NULL`/`abandon_reason TEXT NULL` (terminal tombstone/GC-blocked), `document_revisions.superseded_at`/`artifacts_purged_at` lifecycle metadata, `revision_ingestion_attempts` with unique constraint `uq_revision_ingestion_attempt_key` on `(document_id, source_object_identity, build_profile)` (the `ON CONFLICT` arbiter, not a read-then-write check), canonical source component columns (`source_scheme`, `source_bucket`, `source_object_key`, `source_version_id`, `source_etag`, `source_size`, `source_sha256`), a bounded `attempt_generation INT NOT NULL DEFAULT 1` retry counter plus `exhausted_at TIMESTAMPTZ NULL`, `source_arrivals` (webhook staging: `bucket`, `object_key`, `version_id`, `etag`, `size_bytes`, `arrival_identity`, `received_at`, `processed_at`, unique on `arrival_identity`), `revision_retention_leases` (`lease_id UUID PK`, `run_id TEXT`, `revision_id UUID` FK, `evidence_use_id UUID NULL`, `acquired_at`, `expires_at`, `released_at TIMESTAMPTZ NULL`, `release_reason TEXT NULL`, null-safe unique `(run_id, revision_id, evidence_use_id)`, partial index `(revision_id, expires_at) WHERE released_at IS NULL`), revision build manifest columns (`embedding_namespace`, `embedding_model_hash`, `embedding_dimension`, `vector_artifact_version`), revision lifecycle GC metadata (`superseded_at TIMESTAMPTZ NULL`, `artifacts_purged_at TIMESTAMPTZ NULL`), and evidence encryption metadata columns (`ciphertext`, `encryption_key_id`, `nonce`, `encryption_algorithm`, `payload_purged_at TIMESTAMPTZ NULL`); (2) adds nullable `documents.current_revision_id`, `documents.source_deleted_at TIMESTAMPTZ NULL` (plus a partial index for tombstone lookup), `document_images.revision_id`, and `document_tables.revision_id`; (3) installs foreign keys and revision uniqueness without imposing NOT NULL on legacy child rows, and verifies that cascading deletes from `documents` cannot orphan or destroy `document_revisions`, retained evidence lineage, or revision artifacts; (4) installs DB constraints/triggers requiring revision IDs for rows written through the revision-owned pipeline; (5) verifies existing row counts/checksums are unchanged; and (6) writes schema version 1. Legacy rows stay v1-only and cannot be selected by v2. A full revision-aware reindex later creates new revision-owned child rows and atomically sets `current_revision_id`.

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

## Release 1B — Post-Migration ORM and Readiness

Deploy Release 1A, run the migration command, and verify schema version 1 before building/deploying this release. `AUTO_CREATE_TABLES` remains available only for the explicit legacy-table allowlist; it cannot create or alter v2 tables.

### Task 2: Register ORM Models Only After Schema Version 1 Is Applied

**Files:**
- Create: `backend/app/models/v2_registry.py`
- Create: `backend/app/models/document_revision.py`
- Create: `backend/app/models/document_revision_build.py`
- Create: `backend/app/models/document_revision_chunk.py`
- Create: `backend/app/models/document_ingestion_attempt.py`
- Create: `backend/app/models/revision_retention_lease.py`
- Create: `backend/app/models/source_arrival.py`
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

Assert mappings match completed migration (`Document.current_revision_id` and legacy image/table revision IDs remain nullable; revision-pipeline inserts require revision ownership through DB constraints), all FKs resolve, exact schema version 1 is required before v2 repositories initialize, and `LEGACY_STARTUP_TABLES` excludes every v2 table. Against a fresh legacy schema, lifespan reports the Release-1A migration command and exits before startup table creation. With schema version 1, legacy `AUTO_CREATE_TABLES` may create only allowlisted legacy tables and emits no v2 DDL.

- [ ] **Step 3: Register post-migration ORM mappings and gate startup**

Define revision/build and v2 persistence models, map only already-existing columns, and register them in `app.models.__init__` only in this post-migration release. Replace unrestricted `Base.metadata.create_all()` with `Base.metadata.create_all(tables=LEGACY_STARTUP_TABLES)`. `lifespan` checks exact compatibility before v2 services initialize; it never invokes v2 migration or DDL. EvidenceUse null-safe uniqueness maps the Task-1 SQL index. `DocumentRevision.generation` maps the monotonic per-document allocation column with its unique constraint; `DocumentIngestionAttempt` maps the unique `(document_id, source_object_identity, build_profile)` key; `DocumentRevisionBuild` maps the embedding namespace/model hash/dimension/vector-artifact-version manifest; `Document.source_deleted_at` maps the tombstone column; `EvidenceRecord` maps `ciphertext`/`encryption_key_id`/`nonce`/`encryption_algorithm`/`payload_purged_at` and exposes no plaintext column. `DocumentRevision` also maps the independent GC metadata `superseded_at` and `artifacts_purged_at`, the terminal-failure fields `failed_at`/`failure_stage`/`failure_class`, the terminal-abandon fields `abandoned_at`/`abandon_reason`, and the retry provenance `retry_of_revision_id`; `DocumentIngestionAttempt` maps `attempt_generation` and `exhausted_at`, plus the canonical source columns `source_scheme`/`source_bucket`/`source_object_key`/`source_version_id`/`source_etag`/`source_size`/`source_sha256`; `SourceArrival` maps the webhook staging row; `RevisionRetentionLease` maps the checkpoint/revision retention lease.

- [ ] **Step 4: Test and commit**

```bash
cd backend && pytest tests/migrations/v2/test_populated_legacy_migration.py tests/migrations/v2/test_model_metadata.py tests/migrations/v2/test_migration_control.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/models/v2_registry.py backend/app/models/document_revision.py backend/app/models/document_revision_build.py backend/app/models/document_revision_chunk.py backend/app/models/document_ingestion_attempt.py backend/app/models/revision_retention_lease.py backend/app/models/source_arrival.py backend/app/models/conversation_snapshot.py backend/app/models/semantic_snapshot.py backend/app/models/binding_audit.py backend/app/models/evidence_record.py backend/app/models/evidence_use.py backend/app/models/document.py backend/app/models/__init__.py backend/app/main.py backend/tests/migrations/v2/test_model_metadata.py
git commit -m "feat: register post-migration v2 models"
```

---

## Release 1C — Revision-Owned Build State and Artifacts

Deploy only after Release 1B readiness passes. From this release onward every worker decision is keyed by `revision_id`; legacy Document status fields may remain UI/v1 projections but never determine whether revision work runs or skips.

### Task 3: Implement Revision-Owned Build Profiles, State, and Atomic Publication

**Files:**
- Create: `backend/app/services/agents/v2/persistence/document_revisions.py`
- Create: `backend/app/services/agents/v2/persistence/source_identity.py`
- Test: `backend/tests/agents/v2/persistence/test_document_revisions.py`
- Test: `backend/tests/agents/v2/persistence/test_source_identity.py`

**Interfaces:**
- Produces: internal `RevisionBuildProfile = FULL | CHAT_UPLOAD | PARSE_ONLY`, canonical `compute_source_object_identity`/`normalize_object_key`/`resolve_build_profile`, monotonic `allocate_draft`/`generation`, failure-aware idempotent `get_or_create_ingestion_attempt`, bounded `retry_ingestion_attempt` (new generation per retry), terminal `mark_failed`, terminal `abandon_revision`, `record_worker_state`, `record_artifacts`, `verify_draft`, generation-safe and tombstone-guarded `publish`, `get_published`, and tombstone `mark_source_deleted` (sets `Document.source_deleted_at`, abandons non-published revisions, and never deletes revision rows).

- [ ] **Step 1: Write lifecycle tests**

Test `allocate → revision-owned build → verify → publish`, profile-specific required artifacts, publish transaction updating `Document.current_revision_id`, immutable published columns, failed draft leaving prior current revision unchanged, and tombstone lookup. Assert R2 does not inherit R1 worker completion. `FULL` requires markdown+structure+vectors and configured caption/KG results; `CHAT_UPLOAD` requires markdown+structure+vectors and records caption/KG as intentionally skipped; `PARSE_ONLY` requires markdown+structure and records vector/caption/KG as intentionally skipped. Skips are not failures. Add monotonic/idempotency/manifest tests: `test_concurrent_revision_publish_does_not_regress_current` (allocate R2 then R3; publish R3 first, then R2; current stays R3 while R2 remains a historical published revision), `test_get_or_create_ingestion_attempt_is_idempotent` (same document/source/profile returns one revision), `test_concurrent_get_or_create_ingestion_attempt_is_atomic` (two concurrent sessions racing the same attempt key while neither sees an uncommitted row: exactly one `revision_ingestion_attempts` row, exactly one draft revision, both callers receive the same `revision_id`, no unhandled `UniqueViolation`, and the loser's savepoint leaves no orphan revision or advanced generation), `test_publish_after_tombstone_does_not_resurrect_current` (publish R1, tombstone the document, then publish a verified R2 → `DocumentTombstoned`, `documents.current_revision_id` unchanged, and a concurrent tombstone that commits between verify and CAS also leaves current unset), `test_verified_revision_blocked_by_tombstone_becomes_abandoned` (publish R1, tombstone, then publish a verified R2 → R2 is terminal `abandoned` with `abandon_reason='document_tombstoned'`, current stays R1, and R2's artifacts become Predicate-B eligible), `test_tombstone_abandons_non_published_revisions` (`mark_source_deleted` moves draft/building/verified revisions to `abandoned` while already-published revisions stay historical), and `test_record_artifacts_persists_embedding_manifest` (the build manifest records namespace/model hash/dimension/vector artifact version). Add explicit failure/retry tests: `test_failed_revision_is_terminal_and_immutable` (a failed revision can never transition to verified/published and leaves the prior current revision unchanged), `test_queue_redelivery_of_failed_revision_is_noop` (redelivering a failed revision neither re-runs a stage nor allocates a new generation), `test_retry_after_failure_allocates_new_generation` (failed R1 → retry yields R2 with `generation > R1` and `retry_of_revision_id = R1`, and R1 becomes superseded), and `test_retry_is_bounded_and_exhausts` (after `MAX_REVISION_RETRIES` the attempt is marked permanently failed and `RevisionRetriesExhausted` is raised). Add canonical identity tests in `tests/agents/v2/persistence/test_source_identity.py`: `test_source_object_identity_is_canonical` (the storage key is used verbatim; the user filename and case never change the identity; an S3 event key is form-decoded exactly once before normalization), `test_multipart_etag_is_not_a_content_hash` (same etag/size but different `sha256` yields different attempt identities while sharing one arrival key), `test_s3_event_key_is_decoded_once_and_matches_storage_key` (`doc+1%20b.pdf` from the event canonicalizes to the same key as the storage-reported `doc 1 b.pdf`), `test_etag_is_a_version_selector_not_content_identity` (identity and arrival strings are stable across callers and sha256 is the only content discriminator), `test_overwrite_same_key_changes_identity_and_creates_new_attempt` (new version/etag/sha → new identity → new attempt), `test_duplicate_webhook_arrival_is_idempotent` (`source_arrivals` upsert), and `test_build_profile_resolution_is_deterministic` (`doc_*` → `FULL`, `chat_file_*` → `CHAT_UPLOAD`, explicit parse-only flag overrides).

- [ ] **Step 2: Implement canonical source identity and explicit lifecycle**

`source_object_identity` is a canonical, immutable, versioned string derived from the stored object — never from a user filename or a client header:

```python
SOURCE_SCHEME = "s3v1"

def normalize_object_key(raw_key: str) -> str:
    """Key that is already decoded (storage/list API or parsed request): used verbatim.
    No case folding and no filename reconstruction; strip exactly one leading '/'; reject
    empty/absolute/'..'."""
    key = raw_key[1:] if raw_key.startswith("/") else raw_key
    if not key or key.startswith("..") or key.startswith("/"):
        raise InvalidSourceObjectKey(raw_key)
    return key

def object_key_from_s3_event(event_key: str) -> str:
    """S3/MinIO ObjectCreated keys are application/x-www-form-urlencoded: spaces arrive as
    '+', non-ASCII as UTF-8 %XX. Decode exactly once with form semantics, then normalize."""
    return normalize_object_key(unquote_plus(event_key, encoding="utf-8"))

def _version_token(version_id: str | None, etag: str | None) -> str:
    """A version SELECTOR, never a content identity. versionId wins when versioning is on;
    otherwise the storage etag is a best-effort object-version surrogate (a multipart etag
    '<md5>-<n>' is kept verbatim and is still not a content hash)."""
    if version_id:
        return f"version:{version_id}"
    if etag:
        return f"etag:{etag.lower()}"
    raise MissingObjectVersion()

def compute_source_object_identity(
    bucket: str, object_key: str, version_id: str | None, etag: str | None,
    size_bytes: int, content_sha256: str,
) -> str:
    """The ONLY attempt key. It always includes the streamed sha256, so two objects that
    share a version selector (no versioning, same bytes/parts) still separate when content
    differs."""
    key = normalize_object_key(object_key)
    return f"{SOURCE_SCHEME}|{bucket}|{key}|{_version_token(version_id, etag)}|{size_bytes}|{content_sha256}"

def arrival_identity(
    bucket: str, object_key: str, version_id: str | None, etag: str | None, size_bytes: int,
) -> str:
    """Webhook-only staging key; intentionally omits sha256. It matches a webhook to /confirm
    for the same event and is NEVER used as an attempt key."""
    key = normalize_object_key(object_key)
    return f"{SOURCE_SCHEME}|{bucket}|{key}|{_version_token(version_id, etag)}|{size_bytes}"

def resolve_build_profile(object_key: str, flags: IngestFlags) -> RevisionBuildProfile:
    """Deterministic from the object, so the same object always yields one attempt key."""
    if flags.parse_only:
        return "PARSE_ONLY"
    if normalize_object_key(object_key).startswith("chat_file_"):
        return "CHAT_UPLOAD"
    return "FULL"
```

Canonicalization rules, applied identically by every trigger:

- **S3 event keys are form-encoded, storage keys are not.** `object_key_from_s3_event` form-decodes exactly once (`+` → space, `%XX` → UTF-8 byte) and then normalizes; every other trigger (`/confirm`, direct upload, chat upload, reindex, queue retry) passes an already-decoded storage/list key through `normalize_object_key`. Never decode twice and never reconstruct the key from a user filename, so the webhook and `/confirm` converge on one canonical key.
- **The webhook is metadata-only.** On `ObjectCreated` it issues a HEAD for `bucket`/`key`/`versionId`/`etag`/`size` (no body read) and writes `source_arrivals.arrival_identity`; it never creates a revision.
- **ETag is a version selector, not a content identity.** `version_id` wins when storage versioning is on; otherwise the lowercased etag is a best-effort object-version surrogate. A multipart etag (`<md5>-<n>`) is kept verbatim and is still not a content hash. `content_sha256`, streamed by `/confirm`/direct/chat/reindex and never trusted from a client header, is always part of `compute_source_object_identity`.
- **Only `compute_source_object_identity` is an attempt key.** `arrival_identity` deliberately omits sha256 and exists only to match a webhook to its `/confirm`. An arrival-key collision from an identical `etag+size` (versioning off) is acceptable: it groups one ingest event, while differing bytes still yield a different attempt identity through sha256. Identical bytes re-uploaded with versioning off converge to the same attempt (idempotent). Overwriting the same key with different bytes changes version/etag/sha → new identity → new attempt.

```python
class AttemptAlreadyClaimed(Exception):
    """Sentinel: another transaction owns this ingest event's attempt key."""

class RevisionRetriesExhausted(Exception):
    """Terminal failure retry budget for an ingest attempt is exhausted."""

async def get_or_create_ingestion_attempt(
    self, document_id: UUID, source_object_identity: str, build_profile: RevisionBuildProfile
) -> tuple[DocumentRevision, bool]:
    """Idempotent and failure-aware: one ACTIVE revision per (document, source object, profile).

    A read-then-insert races when two transactions both miss an uncommitted row,
    so the unique index is the arbiter, not a SELECT. Retry is an in-place UPDATE
    of the attempt row (bumped attempt_generation) that allocates a NEW revision
    generation; a terminal failed revision is never resumed.
    """
    # 1. Serialize per document. This makes generation allocation monotonic, makes
    #    the attempt re-read see committed rows, and serializes retry updates.
    await self.lock_document_for_update(document_id)  # SELECT id FROM documents WHERE documents.id = :id FOR UPDATE

    # 2. Converge on an existing attempt for this ingest event.
    attempt = await self.get_attempt_for_update(document_id, source_object_identity, build_profile)
    if attempt is not None:
        active = await self.get(attempt.revision_id)
        if active is not None and active.state == "failed":
            return await self.retry_ingestion_attempt(attempt, build_profile)  # terminal -> new generation
        return active, False

    # 3. First attempt: create the candidate revision in a savepoint so a lost race
    #    leaves no orphan, then claim the key atomically.
    revision: DocumentRevision
    try:
        async with self.session.begin_nested():
            revision = await self._new_draft(document_id, source_object_identity, build_profile)
            claimed = (await self.session.execute(
                pg_insert(DocumentIngestionAttempt)
                .values(
                    document_id=document_id,
                    source_object_identity=source_object_identity,
                    build_profile=build_profile,
                    revision_id=revision.revision_id,
                    attempt_generation=1,
                )
                .on_conflict_do_nothing(constraint="uq_revision_ingestion_attempt_key")
                .returning(DocumentIngestionAttempt.revision_id)
            )).scalar_one_or_none()
            if claimed is None:
                raise AttemptAlreadyClaimed  # rolls back this savepoint, discarding the draft
    except AttemptAlreadyClaimed:
        pass
    else:
        return revision, True

    # 4. Lost the create race: converge on the committed winner, or retry it if it
    #    already reached a terminal failure. Safe under READ COMMITTED because the
    #    conflicting insert has committed and the document lock prevents a second winner.
    attempt = await self.get_attempt_for_update(document_id, source_object_identity, build_profile)
    active = await self.get(attempt.revision_id)
    if active.state == "failed":
        return await self.retry_ingestion_attempt(attempt, build_profile)
    return active, False


async def retry_ingestion_attempt(
    self, attempt: DocumentIngestionAttempt, build_profile: RevisionBuildProfile
) -> tuple[DocumentRevision, bool]:
    """Bounded automatic retry. A terminal failed revision is NEVER resumed.

    Every retry allocates a NEW generation with explicit provenance, marks the
    failed revision superseded, and bumps the attempt counter in place. The
    attempt key stays unique; only its revision pointer and counter advance.
    """
    if attempt.attempt_generation >= self.MAX_REVISION_RETRIES:
        await self.mark_attempt_exhausted(attempt)  # permanent failure, surfaced to UI
        raise RevisionRetriesExhausted(str(attempt.document_id))
    failed = await self.get(attempt.revision_id)
    revision = await self._new_draft(
        attempt.document_id,
        attempt.source_object_identity,
        build_profile,
        retry_of_revision_id=failed.revision_id,
    )
    await self.mark_superseded(failed.revision_id, superseded_by=revision.revision_id)
    attempt.revision_id = revision.revision_id
    attempt.attempt_generation += 1
    await self.session.flush()
    return revision, True


async def mark_failed(self, revision_id: UUID, stage: str, error_class: str) -> DocumentRevision:
    """Terminal failure. Failed revisions are immutable and can never be published."""
    revision = await self.get_for_update(revision_id)
    if revision.state in {"published", "verified"}:
        raise RevisionNotFailable(str(revision_id))
    revision.state = "failed"
    revision.failed_at = datetime.now(timezone.utc)
    revision.failure_stage = stage
    revision.failure_class = error_class
    await self.session.flush()
    return revision


async def abandon_revision(self, revision: DocumentRevision, reason: str) -> DocumentRevision:
    """Terminal tombstone/GC-blocked state. A verified revision that can never be published
    because its source was tombstoned must not linger non-terminal: abandoned is immutable and
    eligible for artifact GC, so its build output is reclaimed instead of leaking."""
    if revision.state == "published":
        raise RevisionNotAbandonable(str(revision.revision_id))
    revision.state = "abandoned"
    revision.abandoned_at = datetime.now(timezone.utc)
    revision.abandon_reason = reason
    await self.session.flush()
    return revision


async def mark_source_deleted(self, document_id: UUID, reason: str = "user_deleted") -> Document:
    """Tombstone the document and abandon every non-published revision in one transaction."""
    document = await self.get_document_for_update(document_id)
    if document.source_deleted_at is None:
        document.source_deleted_at = datetime.now(timezone.utc)
    for revision in await self.list_non_published_for_update(document_id):  # draft|building|verified
        await self.abandon_revision(revision, reason="document_tombstoned")
    await self.session.flush()
    return document


async def publish(self, revision_id: UUID) -> DocumentRevision:
    revision = await self.get_for_update(revision_id)
    if revision.state != "verified":
        raise RevisionNotPublishable(str(revision_id))
    document = await self.get_document_for_update(revision.document_id)
    if document.source_deleted_at is not None:
        await self.abandon_revision(revision, reason="document_tombstoned")
        raise DocumentTombstoned(str(document.id))  # never resurrect a deleted source
    revision.state = "published"
    revision.published_at = datetime.now(timezone.utc)
    advanced = await self.cas_advance_current(
        document_id=document.id,          # documents.id is the PK, not documents.document_id
        revision_id=revision.revision_id,
        generation=revision.generation,
    )
    if advanced == 0:
        # Either current already has a newer generation, or a concurrent tombstone won.
        # Re-read so a tombstone fails closed; otherwise this is a historical publish.
        if await self.is_source_deleted(document.id):
            await self.abandon_revision(revision, reason="document_tombstoned")
            raise DocumentTombstoned(str(document.id))
    await self.session.flush()
    return revision
```

**Failure, abandonment, and retry semantics (explicit).** The revision state machine is `draft → building → verified → published`, with `draft|building → failed` and `draft|building|verified → abandoned` both terminal. `failed` means the build itself failed; `abandoned` means the source/document was tombstoned (or otherwise permanently blocked), so the revision can never become current. Both terminals are immutable and can never transition to `verified`/`published`; `publish` and `mark_source_deleted` commit the terminal transition before raising or returning, so an abandoned revision is durable even though `DocumentTombstoned` still propagates. Three retry paths are distinct: (1) **stage redelivery** — a queue message whose revision is still `draft|building` re-runs only the incomplete stage idempotently, same `revision_id`; if the revision is already complete the message is a no-op; if it is `failed` the message is a no-op dead-letter that never allocates. (2) **automatic retry** — `get_or_create_ingestion_attempt` on a terminal-failed attempt calls `retry_ingestion_attempt`, which allocates a NEW generation with `retry_of_revision_id` provenance, marks the failed revision superseded, and bumps `attempt_generation`, bounded by `MAX_REVISION_RETRIES` (implementation setting); at the budget it calls `mark_attempt_exhausted` and raises `RevisionRetriesExhausted`. (3) **explicit user retry/reindex** — always allocates a new generation and never mutates a prior revision. Monotonic generation guarantees a retry can only become current if it is newer.

The DB-level CAS is a single conditional update, not read-then-write, and uses the real `documents` primary key (`documents.id`) plus a tombstone guard:

```sql
UPDATE documents
SET current_revision_id = :rid
WHERE documents.id = :did
  AND documents.source_deleted_at IS NULL
  AND (documents.current_revision_id IS NULL
       OR :generation > (SELECT generation
                         FROM document_revisions
                         WHERE revision_id = documents.current_revision_id))
```

`rowcount = 0` means either a newer generation is already current or the source was tombstoned concurrently; the caller discriminates with `is_source_deleted` and raises `DocumentTombstoned` rather than resurrecting a deleted document. The losing revision stays `published` but historical. A verified revision whose source was tombstoned before or during CAS becomes terminal `abandoned` (never current) and is artifact-GC-eligible under Predicate B. `lock_document_for_update` raises `DocumentNotFound` when `SELECT id FROM documents WHERE documents.id = :id FOR UPDATE` returns no row, so an attempt can never be created for a nonexistent document. `record_artifacts` persists the build manifest (embedding namespace, model identity/hash, dimension, vector artifact version) so later retrieval never guesses its embedding location from current configuration. `DocumentRevision` plus revision-build rows are authoritative for processing state, `embed_done`, `captions_done`, `kg_done`, raw chunk/build manifest, markdown artifact identity, artifact verification, failures, and publication. `Document` may mirror current status for v1/UI only. `verify_draft` consults the immutable profile selected at allocation and checks exactly its required artifacts before transitioning to verified; no worker reads Document completion flags to skip work.

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
- Produces: triggers of one ingest event converge on `get_or_create_ingestion_attempt(document_id, source_object_identity, build_profile)`; explicit reindex allocates a new generation; queue retry/redelivery reuses the message `revision_id`; all workers load/write by the same required revision UUID and never use Document completion flags.

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

Discover every message constructor and make the test fail if an unclassified caller remains. Parameterize full upload/reindex as `FULL`, chat upload as `CHAT_UPLOAD`, and parse-only entrypoints as `PARSE_ONLY`; assert allocation occurs before first publish and child messages preserve required `revision_id` and profile. Publish R1 then build R2 and assert parse deletes/replaces only `(document_id, R2)` image/table/chunk rows; R1 SQL children remain readable. Assert R1 completion flags cannot make R2 workers skip and stale messages cannot publish a newer revision. Add `test_webhook_and_confirm_create_one_revision` (MinIO `ObjectCreated` and `/confirm` race to one draft and one parse execution identity), `test_duplicate_webhook_is_idempotent` (redelivery returns the same revision with no duplicate draft), `test_chat_upload_webhook_profile_is_preserved` (with the webhook enabled, a `chat_file_<document_id>` object still yields `CHAT_UPLOAD`, never `FULL`), `test_reindex_allocates_new_revision_for_same_source_object` (explicit reindex over the same `upload_s3_key` produces a new higher generation, not the cached attempt revision), and a clone test proving `_clone_document_to_workspace` creates a target-workspace revision and never marks a Document indexed without one.

- [ ] **Step 3: Implement allocation and propagation**

All four message models require `revision_id: UUID` with no default. Producers publish only a revision returned by the lifecycle. Every trigger first computes the canonical `compute_source_object_identity(...)` (bucket + canonical object key with S3 event keys form-decoded exactly once + version selector + size + streamed sha256) and `resolve_build_profile(...)`, then **direct upload**, **presigned confirm**, and **chat upload** call the lock-then-upsert `get_or_create_ingestion_attempt(document_id, identity, profile)` (unique-index arbiter, no read-then-insert race); the `ObjectCreated` webhook writes only a `source_arrivals` row keyed by `arrival_identity` and never creates a revision. **explicit reindex** is not an idempotent redelivery and calls `allocate_draft(...)` directly to create a new monotonic generation even when `upload_s3_key` is unchanged, recording `reindex_of_revision_id`; **queue retry/redelivery** never re-allocates on its own: it re-runs only an incomplete stage of a non-terminal (`draft|building`) revision, and is a no-op dead-letter if that revision is already complete or terminal-`failed`; only the bounded `retry_ingestion_attempt` path or an explicit reindex allocates a new generation. Trigger ownership is explicit: for **presigned upload**, the MinIO `ObjectCreated` webhook records object arrival only (a `source_arrivals` row) and `/confirm` is authoritative — it matches the staged arrival, hashes the object, selects the profile, and calls `get_or_create_ingestion_attempt` with the canonical identity; for **chat upload** the `chat_file_<document_id>` object is normalized to the same attempt key with `CHAT_UPLOAD`, and webhook matching must recognize both `doc_<document_id>` and `chat_file_<document_id>` shapes. Redelivery of the same ingest event (duplicate webhook, `/confirm` racing its webhook, retried API call) is a no-op returning the existing revision while it is non-terminal; if that revision is terminal-`failed`, the failure-aware path allocates a bounded new generation instead of silently re-running it. Workers load exactly that revision/build record, scope image/table/chunk replacement to `(document_id, revision_id)`, publish child messages with the same UUID, and update only revision-owned completion/failure state. They never query `Document.embed_done`, `captions_done`, `kg_done`, `raw_chunks_json`, `markdown_s3_key`, or status to decide revision execution. Finalization executes `record_artifacts → profile-aware verify_draft → publish`; failures mark only that draft failed. `_clone_document_to_workspace` never copies markdown/vectors or sets Document completion flags: it allocates a target-workspace revision, records explicit `cloned_from_revision_id` provenance, and runs the revision-aware build (reusing raw content/hash only), so the target binding revision is never another workspace's revision identity.

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
- Create: `backend/app/services/agents/v2/persistence/document_views.py`
- Modify: `backend/app/services/storage_service.py`
- Modify: `backend/app/services/embedding/vector_store.py`
- Modify: `backend/app/services/retrieval/deep_retriever.py`
- Modify: `backend/app/services/retrieval/rag_service.py`
- Modify: `backend/app/services/retrieval/hrag_service.py`
- Modify: `backend/app/api/rag.py`
- Modify: `backend/app/api/documents.py`
- Modify: `backend/app/services/agent/tools.py`
- Modify: `backend/app/services/kg/legal_kg_service.py`
- Test: `backend/tests/agents/v2/persistence/test_revision_artifacts.py`
- Test: `backend/tests/agents/v2/persistence/test_revision_live_callers.py`
- Test: `backend/tests/api/test_revision_reindex_delete.py`

**Interfaces:**
- Produces: revision-qualified SQL/object/vector/KG identities, selected-revision retrieval, tombstone-first deletion, `CurrentDocumentViewAdapter` in `backend/app/services/agents/v2/persistence/document_views.py`, explicit legacy-v1/not-ready policy, stable locators, dimension-safe non-destructive reindex, and revision-scoped KG facts.

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

Tests publish R1, build/publish R2, and prove R1 images/tables/chunks/markdown/vectors/KG remain readable. Exact retrieval filters SQL/Chroma/BM25/KG by selected revision; mixed-revision merge fails; locators reconstruct from stable structure-node IDs. A populated legacy document with only old Chroma/KG/object artifacts remains v1-readable but v2 binding/retrieval returns `REVISION_NOT_READY`; after full revision-aware reindex v2 succeeds. Simulate a new embedding dimension while R1 vectors exist and assert R2 build fails closed or uses a new dimension namespace without deleting/recreating R1's collection. Prove the destructive reindex seams are gone: `reindex_document` must not pre-delete current revision artifacts or reset Document completion flags before a replacement revision publishes, and `reindex_workspace` must never call `VectorStore.delete_collection()` while published revisions exist. Add viewer/delete/KG tests: `test_current_document_view_uses_current_revision` (publish R1 then R2 → viewer shows R2, no mixed R1/R2 chunks/images, R1 still readable internally), `test_delete_tombstones_before_gc` (delete a document with retained evidence → normal lookup denied/not found; evidence grounding stays valid until retention expiry; GC removes physical artifacts only after all references expire), `test_revision_kg_does_not_leak_old_fact` (R1 has fact F1, R2 removes/changes it → R1 query returns F1, R2 query must not return the R1-only F1), and `test_historical_revision_uses_recorded_embedding_namespace` (R1 retrieval uses its recorded namespace/model/dimension even after current embedding config changes). Live-caller tests exercise HRAG and agent tools with selected revision; v1 stays on its named legacy/current adapter.

- [ ] **Step 3: Implement revision-qualified artifacts**

Use revision-qualified object keys, vector IDs/metadata, BM25 corpus entries, and KG provenance. Use an embedding-model/dimension-qualified collection namespace such as `ws_{workspace}_embed_{model_hash}_d{dimension}`; dimension mismatch never calls collection delete/recreate and instead selects a compatible namespace or raises `EmbeddingMigrationRequired`. The v2 port requires a published, artifact-verified revision; it never falls back to legacy chunks/KG. Legacy documents continue through the unchanged v1 retrieval adapter until full revision-aware reindex publishes a ready revision and atomically sets current pointer. Reindex and delete never remove old revision artifacts; GC is the only eligible deletion path. Remove the unconditional workspace collection delete from `reindex_workspace` and the pre-emptive `rag_service.delete_document`/artifact purge from `reindex_document`: both must allocate a new draft revision and publish atomically instead of mutating the current revision in place. Wire `delete_document` to tombstone-first deletion: mark the source/document deleted via `mark_source_deleted`, block new bindings and retrieval, hide it from normal/current lookup, and preserve every revision and artifact still referenced by a retained `EvidenceRecord`; only GC physically reclaims objects/vectors/KG/rows, and only under the independent revision-artifact predicate in Task 9 (evidence expiry alone never releases revision artifacts). Audit FK/cascade so tombstoning or deleting a `Document` cannot cascade-destroy `DocumentRevision`, retained evidence lineage, or retained revision artifacts. Add `CurrentDocumentViewAdapter`, which resolves `Document.current_revision_id` and loads revision markdown/images/chunks for `/document/{id}/markdown`, `/document/{id}/chunk-context`, and `/document/{id}/images` while keeping the external frontend contract stable; it must never fall back to legacy `Document.markdown_s3_key`/`chunk_count` or `doc_<document>_chunk_<index>` when a current revision exists. For KG, allow canonical entities to be shared but scope every document-derived fact, edge, membership, and provenance row to the producing revision so a revision query can never leak a fact that only another revision contains. Historical revision retrieval resolves the embedding namespace/model/dimension/vector-artifact version recorded in that revision's build manifest instead of current configuration.

- [ ] **Step 4: Test and commit**

```bash
cd backend && pytest tests/agents/v2/persistence/test_revision_artifacts.py tests/agents/v2/persistence/test_revision_live_callers.py tests/api/test_revision_reindex_delete.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/persistence/document_views.py backend/app/services/storage_service.py backend/app/services/embedding/vector_store.py backend/app/services/retrieval/deep_retriever.py backend/app/services/retrieval/rag_service.py backend/app/services/retrieval/hrag_service.py backend/app/api/rag.py backend/app/api/documents.py backend/app/services/agent/tools.py backend/app/services/kg/legal_kg_service.py backend/tests/agents/v2/persistence/test_revision_artifacts.py backend/tests/agents/v2/persistence/test_revision_live_callers.py backend/tests/api/test_revision_reindex_delete.py
git commit -m "feat: select immutable revisions for retrieval"
```

---

## Release 1D — Frozen Contracts, Stores, Checkpoint, and Adapters

Begin only after the Release-1C gate proves at least one revision-ready document and legacy-not-ready rejection. This release does not activate a v2 graph.

### Task 6: Implement Canonical Contracts and Pure Validators

**Files:**
- Create: `backend/app/services/agents/v2/contracts/`
- Create: `backend/tests/agents/v2/contracts/`

**Interfaces:**
- Produces: exact frozen contracts and `validate_*` functions from the approved spec.

- [ ] **Step 1: Write contract tests**

Cover strict/frozen/envelope versioning, optional revision semantics, ID/DAG/relation integrity, criteria uniqueness, EvidenceRecord/EvidenceUse separation, purpose/target rules, admitted claim use IDs, TaskExecutionSummary/replan context, and incompatible checkpoint rejection.

- [ ] **Step 2: Implement owner-focused modules**

Create `base.py`, `request.py`, `conversation.py`, `semantic.py`, `locators.py`, `binding.py`, `routing.py`, `planning.py`, `execution.py`, `evidence.py`, `evaluation.py`, `synthesis.py`, `clarification.py`, `response.py`, `capability.py`, `state.py`, and `validation.py`. Define every union only after its variants and call `model_rebuild()` explicitly. `capability.py` owns the frozen `CapabilityDescriptor`, `CapabilityInput`, `CapabilityOutput`, and the runtime-only `CapabilityRuntimeContext`; the capability package imports these types and never redefines or field-extends them.

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
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Test: `backend/tests/agents/v2/persistence/test_evidence_governance.py`

**Interfaces:**
- Produces: idempotent record/use insertion; minimization/classification; fail-closed AES-256-GCM encryption with runtime key management; governed audited hydration.

- [ ] **Step 1: Write governance tests**

Test null-safe EvidenceUse uniqueness, People minimization, encryption, expiry, ACL, tombstone, revision mismatch, derived validation state, and audited allow/deny reads. Add `test_evidence_key_unavailable_fails_closed` (missing key ring/key ID raises `EvidenceKeyUnavailable` and returns no plaintext), `test_evidence_wrong_key_does_not_leak_plaintext` (ciphertext decrypted with the wrong key ID fails closed), and `test_evidence_key_rotation_keeps_old_records_readable` (old records stay readable by their recorded key ID after the active key rotates).

- [ ] **Step 2: Implement fail-closed repositories**

Insert EvidenceUse with `ON CONFLICT (run_id, task_id, evidence_id, purpose, target_id) DO NOTHING`; PostgreSQL infers the matching `NULLS NOT DISTINCT` unique index, including targetless uses. Then select and return the existing UUID by the same five-column null-safe key. Resolve document workspace from immutable revision, never copied evidence metadata. Decrypt only after ACL/expiry/source/validation checks.

Encryption is AES-256-GCM. A record persists only `ciphertext`, `encryption_key_id`, `nonce`, and `encryption_algorithm`; there is no plaintext column. Keys come from runtime secret management (`EVIDENCE_ENCRYPTION_KEYS` keyring plus `EVIDENCE_ENCRYPTION_ACTIVE_KEY_ID`), never from DB business state. New writes use the active key ID. Authorization to decrypt is the same ACL/expiry/source/revision/derived-validation check as hydration; a missing key ID or unavailable keyring fails closed with `EvidenceKeyUnavailable` and never returns plaintext. Rotation adds a new active key ID while retaining prior IDs, so already-retained evidence remains readable with its recorded key ID; in-place re-encryption requires an explicit keyed migration and is out of scope here.

- [ ] **Step 3: Test and commit**

```bash
cd backend && pytest tests/agents/v2/persistence/test_evidence_governance.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/persistence/evidence.py backend/app/services/agents/v2/evidence_store/governance.py backend/app/core/config.py .env.example backend/tests/agents/v2/persistence/test_evidence_governance.py
git commit -m "feat: add governed evidence persistence"
```

---

### Task 9: Implement Executable Evidence and Revision Artifact GC

**Files:**
- Create: `backend/app/services/agents/v2/evidence_store/gc.py`
- Create: `backend/app/services/agents/v2/persistence/revision_gc.py`
- Create: `backend/app/services/agents/v2/persistence/retention_leases.py`
- Create: `backend/app/workers/evidence_gc_worker.py`
- Modify: `backend/app/services/storage_service.py`
- Modify: `backend/app/services/embedding/vector_store.py`
- Modify: `backend/app/services/kg/legal_kg_service.py`
- Modify: `docker-compose.services.yml`
- Test: `backend/tests/agents/v2/persistence/test_evidence_gc.py`
- Test: `backend/tests/agents/v2/persistence/test_revision_kg_gc.py`
- Test: `backend/tests/workers/test_evidence_gc_worker.py`

**Interfaces:**
- Produces: two independent batchers — `run_evidence_payload_gc_batch` (evidence retention only) and `run_revision_artifact_gc_batch` (revision artifacts only) — the checkpoint/revision retention-lease API (`acquire_or_refresh`, `release`, `sweep_expired`, `has_active_lease`), concrete object/vector/revision-scoped-KG artifact delete methods, and a scheduled one-shot worker that drives both.

- [ ] **Step 1: Impact-check destructive methods**

```bash
impact({target: "app.services.storage_service.StorageService.delete_markdown", direction: "upstream"})
impact({target: "app.services.storage_service.StorageService.delete_file", direction: "upstream"})
impact({target: "app.services.embedding.vector_store.VectorStore.delete_by_document_id", direction: "upstream"})
impact({target: "app.services.kg.legal_kg_service.LegalKGService.delete_document", direction: "upstream"})
```

- [ ] **Step 2: Write deletion-order/idempotency tests**

Test advisory lock + `FOR UPDATE SKIP LOCKED`; retry after partial object-store failure; two workers do not double-delete. Add the two independence tests: `test_expired_evidence_alone_does_not_release_revision_artifacts` (evidence `expires_at` passed and payload purged, but a live source inside retention keeps the revision's objects/vectors/KG/images/tables/chunks intact), and `test_revision_artifact_gc_requires_no_retained_references` (a superseded revision whose only unexpired evidence reference is still retained is never artifact-eligible; once that reference is expired+purged or absent, the artifact predicate can pass without any evidence needing to expire first, e.g. a tombstoned source with zero references). Also assert the evidence batcher never deletes revision artifacts and the artifact batcher never deletes evidence rows. Add `test_revision_kg_gc_preserves_shared_entities` (R1 and R2 both reference canonical entity E; GC of R1 removes only R1-scoped facts/edges/memberships/provenance, keeps E because R2 still references it and leaves R2 facts intact, prunes an entity that was R1-only, and a second run deletes zero rows — idempotent). Add lease-invariant tests: `test_checkpoint_pinning_revision_writes_lease` (a checkpoint that pins a revision writes/refreshes a lease in the same transaction), `test_active_lease_blocks_artifact_and_evidence_gc`, `test_expired_lease_does_not_block_gc`, `test_terminal_run_releases_lease`, and `test_resumable_interrupt_keeps_lease_until_expiry`.

- [ ] **Step 3: Implement two independent GC predicates and the worker**

Add `StorageService.delete_revision_artifacts(document_id, revision_id)`, `VectorStore.delete_revision(document_id, revision_id)`, and `LegalKGService.delete_revision_artifacts(document_id, revision_id) -> int`. Implement two predicates that never share eligibility:

**Predicate A — evidence payload retention** (`run_evidence_payload_gc_batch`, evidence_store/gc.py): scope is `evidence_records.ciphertext` only. Eligible when `expires_at <= now()` AND `payload_purged_at IS NULL` AND no legal hold AND no active `revision_retention_lease` references the evidence use/revision (`NOT EXISTS (revision_retention_leases WHERE (evidence_use_id = ... OR revision_id = ...) AND released_at IS NULL AND expires_at > now())`), because a resumable run may still hydrate it. Action: purge ciphertext, set `payload_purged_at`, keep the row/key-id metadata for lineage, and audit the purge. This batcher must never delete objects/vectors/KG/images/tables/chunks/markdown or any `DocumentRevision`.

**Predicate B — revision artifact reclamation** (`run_revision_artifact_gc_batch`, persistence/revision_gc.py): scope is one `DocumentRevision`'s artifacts. Eligible only when ALL hold, evaluated independently of Predicate A: (1) `state IN ('published','failed','abandoned')` and the revision is not `documents.current_revision_id` (an `abandoned` revision was never current); (2) `artifacts_purged_at IS NULL`; (3) no *retained* reference exists — `NOT EXISTS (evidence_records WHERE revision_id = revision.revision_id AND expires_at > now() AND payload_purged_at IS NULL)`; (4) source-resolved or retention-elapsed — `documents.source_deleted_at IS NOT NULL` OR `now() >= revision.superseded_at + retention_window`; (5) no active retention lease — `NOT EXISTS (revision_retention_leases WHERE revision_id = revision.revision_id AND released_at IS NULL AND expires_at > now())`. Action: delete external artifacts idempotently — object/markdown via `StorageService.delete_revision_artifacts`, vectors via `VectorStore.delete_revision`, and revision-scoped KG via `LegalKGService.delete_revision_artifacts` — then set `artifacts_purged_at`; the `DocumentRevision` row is retained as lineage and is never `DELETE`d here. A merely superseded/non-current revision that is still referenced, still current, or whose live source is inside retention is not eligible. `LegalKGService.delete_revision_artifacts` must delete only rows carrying the producing `revision_id` (document-derived facts, edges, memberships, provenance) and must never blind-`DETACH DELETE` a MERGE-shared canonical entity: it removes an entity only when it has no remaining membership in any other revision, and it returns the deleted count and is idempotent (a second call deletes 0).

The only coupling between the predicates is that an unexpired evidence reference *blocks* Predicate B (condition 3). Evidence expiry never triggers artifact deletion; artifact eligibility never requires evidence to have expired.

**Checkpoint/revision retention-lease invariant.** A checkpoint is the only thing that can pin immutable artifacts for a run that may resume, so every checkpoint write that pins a `revision_id` (TaskPlan TargetUnit or DocumentBindingSet) or retains an `EvidenceUse` for later hydration must `acquire_or_refresh` a `revision_retention_leases` row in the same transaction as the checkpoint. Leases are control state and are never checkpointed. `acquire_or_refresh(run_id, revision_id, evidence_use_id=None, ttl=CHECKPOINT_RETENTION_LEASE_TTL)` upserts and extends `expires_at`; terminal finalization, cancellation, or clarification expiry calls `release(...)`; `sweep_expired()` is the crash-safe fallback that marks `released_at` for `expires_at <= now()`. GC reclaims a revision or evidence payload only when no lease is active (`released_at IS NULL AND expires_at > now()`); an expired lease never blocks. The TTL is bounded (config `CHECKPOINT_RETENTION_LEASE_TTL_HOURS`, at least max run duration plus resume grace) so a crashed or abandoned run cannot pin artifacts forever.

`run_revision_artifact_gc_batch` and `run_evidence_payload_gc_batch` each take the advisory lock and `FOR UPDATE SKIP LOCKED` leases, delete idempotently, and record completion. `python -m app.workers.evidence_gc_worker --once --batch-size 100` runs Predicate A then Predicate B as separate transactions; a failure in one does not abort the other. Add a dedicated `evidence-gc` Compose service with a documented periodic command/interval and migration dependency.

- [ ] **Step 4: Test and commit**

```bash
cd backend && pytest tests/agents/v2/persistence/test_evidence_gc.py tests/workers/test_evidence_gc_worker.py -q
docker compose -f docker-compose.services.yml config --quiet
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/evidence_store/gc.py backend/app/services/agents/v2/persistence/revision_gc.py backend/app/workers/evidence_gc_worker.py backend/app/services/storage_service.py backend/app/services/embedding/vector_store.py backend/app/services/kg/legal_kg_service.py docker-compose.services.yml backend/tests/agents/v2/persistence/test_evidence_gc.py backend/tests/agents/v2/persistence/test_revision_kg_gc.py backend/tests/workers/test_evidence_gc_worker.py
git commit -m "feat: add evidence and revision garbage collection"
```

---

### Task 10: Install and Initialize Async PostgreSQL Checkpointing

**Files:**
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Create: `backend/app/services/agents/v2/persistence/checkpoint.py`
- Test: `backend/tests/agents/v2/persistence/test_checkpoint.py`

**Interfaces:**
- Produces: validated `CHECKPOINT_DATABASE_URL`, `create_v2_checkpointer(checkpoint_dsn) -> AsyncPostgresSaver`, and `setup_v2_checkpointer(checkpoint_dsn) -> None`.

- [ ] **Step 1: Verify Phase-0 exact pins and write failing tests**

Assert production requirements contain Phase-0 exact pins. Settings requires a psycopg-compatible `CHECKPOINT_DATABASE_URL=postgresql://...`; tests reject `postgresql+asyncpg://`, absent credentials, and accidental direct reuse of `settings.DATABASE_URL`. Against disposable PostgreSQL, open `AsyncPostgresSaver.from_conn_string(settings.CHECKPOINT_DATABASE_URL)`, run setup, write/read a checkpoint with configurable thread ID, and reject incompatible state versions.

- [ ] **Step 2: Implement async lifecycle**

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

@asynccontextmanager
async def create_v2_checkpointer(checkpoint_dsn: str):
    validate_psycopg_dsn(checkpoint_dsn)
    async with AsyncPostgresSaver.from_conn_string(checkpoint_dsn) as saver:
        yield saver

async def setup_v2_checkpointer(checkpoint_dsn: str) -> None:
    async with create_v2_checkpointer(checkpoint_dsn) as saver:
        await saver.setup()
```

Add `checkpoint --setup` and `checkpoint --check` CLI modes; check verifies saver tables without mutating them. Never instantiate or migrate at module import.

- [ ] **Step 3: Test and commit**

```bash
cd backend && python -m pip check
cd backend && pytest tests/agents/v2/persistence/test_checkpoint.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/core/config.py .env.example backend/app/services/agents/v2/persistence/checkpoint.py backend/tests/agents/v2/persistence/test_checkpoint.py
git commit -m "feat: add async PostgreSQL v2 checkpointing"
```

---

### Task 11: Add Typed Legacy Adapters and Capability Ports

**Files:**
- Create: `backend/app/services/agents/v2/adapters/__init__.py`
- Create: `backend/app/services/agents/v2/adapters/semantic.py`
- Create: `backend/app/services/agents/v2/adapters/conversation.py`
- Create: `backend/app/services/agents/v2/adapters/document.py`
- Create: `backend/app/services/agents/v2/adapters/deep_research.py`
- Create: `backend/app/services/agents/v2/capabilities/__init__.py`
- Test: `backend/tests/agents/v2/test_adapters_and_registry.py`

**Interfaces:**
- Produces: the canonical `Capability` protocol in `capabilities/__init__.py` (importing `CapabilityDescriptor`, `CapabilityInput`, `CapabilityOutput`, and the runtime-only `CapabilityRuntimeContext` from the frozen v2 contracts), typed server-internal semantic/conversation/document/deep-research ports in `adapters/`, and the ACL-filtered capability registry. `adapters/` is the server-internal port layer, not the agent-facing `tools/` gateway created in Phase 3.

- [ ] **Step 1: Write adapter tests**

Require raw query ownership, Draft→Binding→Finalizer flow, immutable revision lookup, typed outputs without dictionaries, runtime capability intersection, and validation rather than v1 `model_construct()` loading.

- [ ] **Step 2: Implement protocols/adapters and commit**

```python
# Defined once, in backend/app/services/agents/v2/capabilities/__init__.py.
# `CapabilityDescriptor`, `CapabilityInput`, `CapabilityOutput`, and
# `CapabilityRuntimeContext` are frozen contracts imported from contracts/capability.py.
class Capability(Protocol):
    descriptor: CapabilityDescriptor
    async def execute(self, request: AgentRequest, runtime: CapabilityRuntimeContext) -> AgentResult: ...
```

Phase 2 imports and re-exports this protocol; it must not define a second `Capability`.

```bash
cd backend && pytest tests/agents/v2/test_adapters_and_registry.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/adapters backend/app/services/agents/v2/capabilities backend/tests/agents/v2/test_adapters_and_registry.py
git commit -m "feat: add typed v2 adapters and capabilities"
node .gitnexus/run.cjs analyze
```

## Phase 1 Release and §26 Gates

Deploy in three separate application releases after the Phase-0 dependency release: **A)** migration-runner code only, no v2 ORM registration; run `migrate --apply` and verify schema version; **B)** ORM/readiness code, startup refuses incompatible schema and performs legacy allowlisted `create_all` only; **C)** revision-aware producers/workers with v2 still unselectable; then **D)** contracts/stores/checkpoint/adapters. Never assume migration-only and post-migration mappings deploy atomically.

Phase 1 owns contract/persistence acceptance only; People→Document, synthesis budget overflow, and synthesis-only reuse are not claimed here.

```bash
docker exec hrag-backend python -m app.services.agents.v2.persistence.migrate --check
docker exec hrag-backend python -m app.services.agents.v2.persistence.checkpoint --check
docker exec hrag-backend pytest tests/migrations/v2 tests/agents/v2/contracts tests/agents/v2/persistence tests/api/test_revision_reindex_delete.py tests/workers/test_revision_pipeline.py tests/workers/test_evidence_gc_worker.py -q
docker exec hrag-backend python -m compileall app/services/agents/v2
docker compose -f docker-compose.services.yml config --quiet
```

Required named tests: `test_concurrent_revision_publish_does_not_regress_current`, `test_publish_after_tombstone_does_not_resurrect_current`, `test_verified_revision_blocked_by_tombstone_becomes_abandoned`, `test_tombstone_abandons_non_published_revisions`, `test_webhook_and_confirm_create_one_revision`, `test_duplicate_webhook_is_idempotent`, `test_chat_upload_webhook_profile_is_preserved`, `test_reindex_allocates_new_revision_for_same_source_object`, `test_concurrent_get_or_create_ingestion_attempt_is_atomic`, `test_delete_tombstones_before_gc`, `test_current_document_view_uses_current_revision`, `test_revision_kg_does_not_leak_old_fact`, `test_historical_revision_uses_recorded_embedding_namespace`, `test_evidence_key_unavailable_fails_closed`, `test_evidence_key_rotation_keeps_old_records_readable`, `test_expired_evidence_alone_does_not_release_revision_artifacts`, `test_revision_artifact_gc_requires_no_retained_references`, `test_revision_kg_gc_preserves_shared_entities`, `test_checkpoint_pinning_revision_writes_lease`, `test_active_lease_blocks_artifact_and_evidence_gc`, `test_expired_lease_does_not_block_gc`, `test_terminal_run_releases_lease`, `test_resumable_interrupt_keeps_lease_until_expiry`, `test_failed_revision_is_terminal_and_immutable`, `test_queue_redelivery_of_failed_revision_is_noop`, `test_retry_after_failure_allocates_new_generation`, `test_retry_is_bounded_and_exhausts`, `test_source_object_identity_is_canonical`, `test_multipart_etag_is_not_a_content_hash`, `test_s3_event_key_is_decoded_once_and_matches_storage_key`, `test_etag_is_a_version_selector_not_content_identity`, `test_overwrite_same_key_changes_identity_and_creates_new_attempt`, `test_duplicate_webhook_arrival_is_idempotent`, `test_build_profile_resolution_is_deterministic`. Broader required coverage: populated-legacy no-fake-baseline migration; exact readiness; FULL/CHAT_UPLOAD/PARSE_ONLY verification; revision-owned worker flags; R1 SQL/object/vector/KG survival while building R2; legacy v1 works while v2 rejects not-ready; dimension mismatch preserves R1; envelope/version strictness; binding/target/task/evidence/use integrity; revision pin/current semantics; governance/expiry/ACL/audit; psycopg checkpoint DSN round-trip; prompt data isolation; and incompatible checkpoint rejection.

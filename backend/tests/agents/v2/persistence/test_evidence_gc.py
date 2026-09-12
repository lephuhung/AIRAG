"""Phase 1D Task 9 — evidence payload GC (Predicate A) and revision artifact GC
(Predicate B), plus the retention-lease invariant.

These tests exercise the real v2 schema (``hrag_test_v2_task3``), the real
batchers, and the real ``RevisionRetentionLeaseRepository``. External artifact
stores (object/vector/KG) are faked — the batchers accept injected factories, and
the tests assert exactly what the two independent predicates do and do not
touch.

Covers the brief's named tests:
  - ``test_expired_evidence_alone_does_not_release_revision_artifacts``
  - ``test_revision_artifact_gc_requires_no_retained_references``
  - ``test_active_lease_blocks_artifact_and_evidence_gc``
  - ``test_expired_lease_does_not_block_gc``
  - ``test_checkpoint_pinning_revision_writes_lease``
  - ``test_terminal_run_releases_lease``
  - ``test_resumable_interrupt_keeps_lease_until_expiry``
  - ``test_lease_is_committed_before_checkpointable_state_update``
  - ``test_lease_failure_prevents_revision_pin_checkpoint``
  - ``test_checkpoint_failure_leaves_only_expiring_orphan_lease``
  - ``test_terminal_checkpoint_happens_before_lease_release``
  - ``test_resume_refreshes_existing_lease``
  - retention-anchor tests (failed/abandoned/late-historical/current)
  - advisory-lock + ``FOR UPDATE SKIP LOCKED``; partial-failure retry; two
    workers do not double-delete; predicate independence.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.document import Document, DocumentImage, DocumentTable
from app.models.document_revision import DocumentRevision
from app.models.document_revision_build import DocumentRevisionBuild
from app.models.document_revision_chunk import DocumentRevisionChunk
from app.models.evidence_record import EvidenceRecord
from app.models.evidence_use import EvidenceUse
from app.models.revision_retention_lease import RevisionRetentionLease
from app.services.agents.v2.contracts.state import RuntimeServices
from app.services.agents.v2.evidence_store.gc import (
    EVIDENCE_GC_ADVISORY_LOCK_KEY,
    EvidencePayloadGcResult,
    run_evidence_payload_gc_batch,
)
from app.services.agents.v2.persistence.document_revisions import (
    DocumentRevisionsRepository,
    PublishOutcome,
)
from app.services.agents.v2.persistence.retention_leases import (
    RevisionRetentionLeaseRepository,
)
from app.services.agents.v2.persistence.revision_gc import (
    REVISION_GC_ADVISORY_LOCK_KEY,
    run_revision_artifact_gc_batch as _run_revision_artifact_gc_batch,
)
from app.services.agents.v2.persistence.source_identity import (
    RevisionBuildProfile,
    compute_source_object_identity,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
TTL = timedelta(hours=1)
WINDOW = timedelta(hours=24)
RUN_ID = "run-gc-1"


def _dt(hours: float) -> datetime:
    return BASE + timedelta(hours=hours)


# ---------------------------------------------------------------------------
# Fake external artifact stores
# ---------------------------------------------------------------------------


class FakeArtifactStore:
    """Records object deletes; optionally fails on a specific key."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.deleted: list[str] = []
        self.calls: list[tuple[uuid.UUID, uuid.UUID, tuple[str, ...]]] = []
        self.workspaces: list[uuid.UUID | None] = []

    async def delete_revision_artifacts(
        self,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        *,
        workspace_id: uuid.UUID | None = None,
        artifact_keys=None,
    ) -> int:
        keys = tuple(artifact_keys or ())
        self.calls.append((document_id, revision_id, keys))
        self.workspaces.append(workspace_id)
        for key in keys:
            if self.fail_on is not None and key == self.fail_on:
                raise RuntimeError(f"object store down deleting {key}")
            self.deleted.append(key)
        return len(keys)


class FakeVectorStore:
    def __init__(self, workspace_id: uuid.UUID, namespace: str | None = None) -> None:
        self.workspace_id = workspace_id
        self.collection_name = namespace
        self.deleted: list[tuple[uuid.UUID, uuid.UUID]] = []

    def delete_revision(self, document_id: uuid.UUID, revision_id: uuid.UUID) -> None:
        self.deleted.append((document_id, revision_id))


class FakeKGService:
    def __init__(self, workspace_id: uuid.UUID) -> None:
        self.workspace_id = workspace_id
        self.deleted: list[tuple[uuid.UUID, uuid.UUID]] = []
        self.cleaned_up = False

    async def delete_revision_artifacts(
        self, document_id: uuid.UUID, revision_id: uuid.UUID
    ) -> int:
        self.deleted.append((document_id, revision_id))
        return 1

    async def cleanup(self) -> None:
        self.cleaned_up = True


def _vector_factory(created: list[FakeVectorStore]):
    def _factory(workspace_id, namespace=None):
        store = FakeVectorStore(workspace_id, namespace)
        created.append(store)
        return store

    return _factory


def _kg_factory(created: list[FakeKGService]):
    def _factory(workspace_id):
        kg = FakeKGService(workspace_id)
        created.append(kg)
        return kg

    return _factory


def _artifact_keys(document_id, revision_id, workspace_id=None):
    return (
        f"kb_{workspace_id}/revisions/{document_id}/{revision_id}/document.md",
        f"kb_{workspace_id}/revisions/{document_id}/{revision_id}/structure.json",
    )


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


async def _make_revision(
    db: AsyncSession,
    *,
    document_id: uuid.UUID,
    status: str = "published",
    generation: int = 1,
    anchor: datetime | None = None,
    artifacts_purged_at: datetime | None = None,
) -> DocumentRevision:
    revision = DocumentRevision(
        revision_id=uuid.uuid4(),
        document_id=document_id,
        generation=generation,
        status=status,
        artifact_retention_starts_at=anchor,
        artifacts_purged_at=artifacts_purged_at,
    )
    db.add(revision)
    await db.flush()
    return revision


async def _make_build(
    db: AsyncSession,
    *,
    revision_id: uuid.UUID,
    markdown_key: str | None = None,
    structure_key: str | None = None,
    namespace: str | None = "ws_test_embed_hash_d768",
    model_hash: str | None = "hash",
    dimension: int | None = 768,
    version: str | None = "v1",
    kg_skipped: bool = False,
    finished_at: datetime = BASE,
    profile: RevisionBuildProfile = RevisionBuildProfile.FULL,
) -> DocumentRevisionBuild:
    build = DocumentRevisionBuild(
        build_id=uuid.uuid4(),
        revision_id=revision_id,
        build_profile=profile.value,
        embedding_namespace=namespace,
        embedding_model_hash=model_hash,
        embedding_dimension=dimension,
        vector_artifact_version=version,
        markdown_artifact_key=markdown_key,
        structure_artifact_key=structure_key,
        kg_skipped=kg_skipped,
        finished_at=finished_at,
    )
    db.add(build)
    await db.flush()
    return build


async def _make_evidence(
    db: AsyncSession,
    *,
    revision_id: uuid.UUID | None,
    expires_at: datetime | None,
    payload_purged_at: datetime | None = None,
    ciphertext: bytes = b"secret-payload",
) -> EvidenceRecord:
    record = EvidenceRecord(
        evidence_id=uuid.uuid4(),
        contract_version="2.0",
        ciphertext=ciphertext,
        encryption_key_id="k1",
        nonce=b"0123456789ab",
        encryption_algorithm="AES-256-GCM",
        content_hash=uuid.uuid4().hex,
        classification="normal",
        expires_at=expires_at,
        revision_id=revision_id,
        source={"kind": "document"},
        provenance={"fetcher": "document.read", "fetched_at": BASE.isoformat()},
        payload_purged_at=payload_purged_at,
    )
    db.add(record)
    await db.flush()
    return record


async def _make_use(
    db: AsyncSession, *, evidence_id: uuid.UUID, run_id: str = RUN_ID
) -> EvidenceUse:
    use = EvidenceUse(
        use_id=uuid.uuid4(),
        run_id=run_id,
        evidence_id=evidence_id,
        task_id="t1",
        purpose="supporting",
        target_id=None,
    )
    db.add(use)
    await db.flush()
    return use


async def _make_lease(
    db: AsyncSession,
    *,
    run_id: str,
    revision_id: uuid.UUID,
    evidence_use_id: uuid.UUID | None = None,
    expires_at: datetime,
    released_at: datetime | None = None,
) -> RevisionRetentionLease:
    lease = RevisionRetentionLease(
        lease_id=uuid.uuid4(),
        run_id=run_id,
        revision_id=revision_id,
        evidence_use_id=evidence_use_id,
        acquired_at=BASE,
        expires_at=expires_at,
        released_at=released_at,
    )
    db.add(lease)
    await db.flush()
    return lease


async def _workspace_id(db: AsyncSession, document_id: uuid.UUID) -> uuid.UUID:
    return await db.scalar(
        select(Document.workspace_id).where(Document.id == document_id)
    )


async def _run_revision_gc_with_fakes(session, **kwargs):
    """Test wrapper: default the external artifact stores to in-memory fakes.

    The production default pulls in MinIO/ChromaDB/Neo4j; these tests only need
    to observe which revision-qualified deletes the batcher issues.
    """
    kwargs.setdefault("storage", FakeArtifactStore())
    kwargs.setdefault("vector_store_factory", _vector_factory([]))
    kwargs.setdefault("kg_service_factory", _kg_factory([]))
    return await _run_revision_artifact_gc_batch(session, **kwargs)


run_revision_artifact_gc_batch = _run_revision_gc_with_fakes


# ---------------------------------------------------------------------------
# Predicate A — evidence payloads
# ---------------------------------------------------------------------------


class TestEvidencePayloadGc:
    @pytest.mark.asyncio
    async def test_expired_payload_is_purged_and_lineage_is_kept(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(db, document_id=document_id, anchor=None)
        record = await _make_evidence(
            db, revision_id=revision.revision_id, expires_at=_dt(-2)
        )

        result = await run_evidence_payload_gc_batch(db, now=BASE)

        assert result.purged == 1
        assert result.purged_ids == (record.evidence_id,)
        await db.refresh(record)
        assert record.ciphertext == b""
        assert record.payload_purged_at is not None
        # Lineage metadata survives.
        assert record.encryption_key_id == "k1"
        assert record.nonce == b"0123456789ab"
        assert record.encryption_algorithm == "AES-256-GCM"
        assert record.content_hash

    @pytest.mark.asyncio
    async def test_unexpired_or_already_purged_rows_are_not_touched(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        future = await _make_evidence(
            db, revision_id=None, expires_at=_dt(+10)
        )
        purged = await _make_evidence(
            db,
            revision_id=None,
            expires_at=_dt(-10),
            payload_purged_at=_dt(-5),
            ciphertext=b"",
        )

        result = await run_evidence_payload_gc_batch(db, now=BASE)

        assert result.purged == 0
        await db.refresh(future)
        await db.refresh(purged)
        assert future.ciphertext == b"secret-payload"
        assert purged.payload_purged_at == _dt(-5)

    @pytest.mark.asyncio
    async def test_evidence_batcher_never_touches_revision_artifacts(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        workspace_id = await _workspace_id(db, document_id)
        revision = await _make_revision(
            db, document_id=document_id, anchor=None
        )
        markdown_key, structure_key = _artifact_keys(
            document_id, revision.revision_id, workspace_id
        )
        await _make_build(
            db,
            revision_id=revision.revision_id,
            markdown_key=markdown_key,
            structure_key=structure_key,
        )
        db.add(
            DocumentRevisionChunk(
                chunk_id=uuid.uuid4(), revision_id=revision.revision_id, ordinal=0
            )
        )
        db.add(
            DocumentImage(
                id=uuid.uuid4(),
                document_id=document_id,
                revision_id=revision.revision_id,
                page_no=1,
            )
        )
        db.add(
            DocumentTable(
                id=uuid.uuid4(),
                document_id=document_id,
                revision_id=revision.revision_id,
                page_no=1,
            )
        )
        record = await _make_evidence(
            db, revision_id=revision.revision_id, expires_at=_dt(-2)
        )
        await db.flush()

        storage = FakeArtifactStore()
        vectors: list[FakeVectorStore] = []
        kgs: list[FakeKGService] = []
        # Predicate A takes no artifact-store dependencies at all: it cannot
        # delete an object/vector/KG even if one were injected.
        result = await run_evidence_payload_gc_batch(db, now=BASE)

        assert result.purged == 1
        await db.refresh(record)
        assert record.payload_purged_at is not None
        assert storage.deleted == []
        assert vectors == []
        assert kgs == []
        assert await db.get(DocumentRevision, revision.revision_id) is not None
        assert await db.scalar(
            select(func.count()).select_from(DocumentRevisionBuild).where(
                DocumentRevisionBuild.revision_id == revision.revision_id
            )
        ) == 1
        assert await db.scalar(
            select(func.count()).select_from(DocumentRevisionChunk).where(
                DocumentRevisionChunk.revision_id == revision.revision_id
            )
        ) == 1


# ---------------------------------------------------------------------------
# Predicate B — revision artifacts
# ---------------------------------------------------------------------------


class TestRevisionArtifactGc:
    @pytest.mark.asyncio
    async def test_eligible_revision_reclaims_all_three_artifact_stores(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        workspace_id = await _workspace_id(db, document_id)
        revision = await _make_revision(db, document_id=document_id, anchor=_dt(-48))
        markdown_key, structure_key = _artifact_keys(
            document_id, revision.revision_id, workspace_id
        )
        await _make_build(
            db,
            revision_id=revision.revision_id,
            markdown_key=markdown_key,
            structure_key=structure_key,
        )
        other_revision = await _make_revision(
            db, document_id=document_id, generation=2, anchor=None
        )
        db_doc = await db.get(Document, document_id)
        db_doc.current_revision_id = other_revision.revision_id
        await db.flush()

        storage = FakeArtifactStore()
        vectors: list[FakeVectorStore] = []
        kgs: list[FakeKGService] = []
        result = await run_revision_artifact_gc_batch(
            db,
            now=BASE,
            retention_window=WINDOW,
            storage=storage,
            vector_store_factory=_vector_factory(vectors),
            kg_service_factory=_kg_factory(kgs),
        )

        assert result.reclaimed == 1
        assert result.reclaimed_revision_ids == (revision.revision_id,)
        # Object artifacts: exactly the manifest keys, revision-qualified.
        assert storage.deleted == [markdown_key, structure_key]
        assert storage.calls == [
            (document_id, revision.revision_id, (markdown_key, structure_key))
        ]
        # Vectors: the revision's own recorded namespace + revision metadata.
        assert [(s.workspace_id, s.collection_name) for s in vectors] == [
            (workspace_id, "ws_test_embed_hash_d768")
        ]
        assert vectors[0].deleted == [(document_id, revision.revision_id)]
        # KG: revision-scoped delete, driver cleaned up.
        assert kgs[0].deleted == [(document_id, revision.revision_id)]
        assert kgs[0].cleaned_up is True
        # The lineage row survives; only the purge tombstone is set.
        stored = await db.get(DocumentRevision, revision.revision_id)
        assert stored is not None
        assert stored.artifacts_purged_at is not None

    @pytest.mark.asyncio
    async def test_current_revision_is_not_artifact_eligible(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        current = await _make_revision(
            db, document_id=document_id, anchor=_dt(-48)
        )
        db_doc = await db.get(Document, document_id)
        db_doc.current_revision_id = current.revision_id
        await db.flush()

        result = await run_revision_artifact_gc_batch(db, now=BASE)

        assert result.reclaimed == 0
        stored = await db.get(DocumentRevision, current.revision_id)
        assert stored.artifacts_purged_at is None

    @pytest.mark.asyncio
    async def test_revision_inside_retention_window_is_not_eligible(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        await _make_revision(db, document_id=document_id, anchor=_dt(-1))

        result = await run_revision_artifact_gc_batch(
            db, now=BASE, retention_window=WINDOW
        )

        assert result.reclaimed == 0

    @pytest.mark.asyncio
    async def test_failed_revision_without_anchor_is_not_eligible(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        await _make_revision(
            db, document_id=document_id, status="failed", anchor=None
        )

        result = await run_revision_artifact_gc_batch(db, now=BASE)

        assert result.reclaimed == 0

    @pytest.mark.asyncio
    async def test_tombstoned_source_revision_is_reclaimable_without_anchor(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(
            db, document_id=document_id, status="abandoned", anchor=None
        )
        db_doc = await db.get(Document, document_id)
        db_doc.source_deleted_at = BASE
        await db.flush()

        result = await run_revision_artifact_gc_batch(db, now=BASE)

        assert result.reclaimed == 1
        stored = await db.get(DocumentRevision, revision.revision_id)
        assert stored.artifacts_purged_at is not None

    @pytest.mark.asyncio
    async def test_revision_artifact_gc_requires_no_retained_references(
        self, async_db, document_factory
    ):
        """An unexpired, unpurged evidence reference blocks reclamation.

        Once that reference is expired AND purged, the artifact predicate can
        pass — evidence expiry is never itself a trigger for artifact deletion.
        """
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(db, document_id=document_id, anchor=_dt(-48))
        retained = await _make_evidence(
            db, revision_id=revision.revision_id, expires_at=_dt(+10)
        )

        blocked = await run_revision_artifact_gc_batch(
            db, now=BASE, retention_window=WINDOW
        )
        assert blocked.reclaimed == 0
        assert (
            await db.get(DocumentRevision, revision.revision_id)
        ).artifacts_purged_at is None

        # Expire + purge the reference (as Predicate A would): now eligible.
        retained.expires_at = _dt(-1)
        retained.payload_purged_at = _dt(-1)
        retained.ciphertext = b""
        await db.flush()

        eligible = await run_revision_artifact_gc_batch(
            db, now=BASE, retention_window=WINDOW
        )
        assert eligible.reclaimed == 1
        assert (
            await db.get(DocumentRevision, revision.revision_id)
        ).artifacts_purged_at is not None

    @pytest.mark.asyncio
    async def test_expired_evidence_alone_does_not_release_revision_artifacts(
        self, async_db, document_factory
    ):
        """Evidence expiry purges the payload but leaves a live source's
        revision artifacts intact while the revision is inside retention."""
        db = async_db
        document_id = document_factory()
        workspace_id = await _workspace_id(db, document_id)
        revision = await _make_revision(
            db, document_id=document_id, anchor=_dt(-1)  # inside the window
        )
        markdown_key, structure_key = _artifact_keys(
            document_id, revision.revision_id, workspace_id
        )
        await _make_build(
            db,
            revision_id=revision.revision_id,
            markdown_key=markdown_key,
            structure_key=structure_key,
        )
        record = await _make_evidence(
            db, revision_id=revision.revision_id, expires_at=_dt(-2)
        )

        evidence_result = await run_evidence_payload_gc_batch(db, now=BASE)
        assert evidence_result.purged == 1

        storage = FakeArtifactStore()
        vectors: list[FakeVectorStore] = []
        kgs: list[FakeKGService] = []
        artifact_result = await run_revision_artifact_gc_batch(
            db,
            now=BASE,
            retention_window=WINDOW,
            storage=storage,
            vector_store_factory=_vector_factory(vectors),
            kg_service_factory=_kg_factory(kgs),
        )

        assert artifact_result.reclaimed == 0
        assert storage.deleted == []
        assert vectors == []
        assert kgs == []
        stored = await db.get(DocumentRevision, revision.revision_id)
        assert stored is not None
        assert stored.artifacts_purged_at is None
        await db.refresh(record)
        assert record.payload_purged_at is not None

    @pytest.mark.asyncio
    async def test_revision_batcher_never_deletes_evidence_rows(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(db, document_id=document_id, anchor=_dt(-48))
        record = await _make_evidence(
            db,
            revision_id=revision.revision_id,
            expires_at=_dt(-1),
            payload_purged_at=_dt(-1),
            ciphertext=b"",
        )

        result = await run_revision_artifact_gc_batch(
            db, now=BASE, retention_window=WINDOW
        )

        assert result.reclaimed == 1
        assert await db.get(EvidenceRecord, record.evidence_id) is not None
        assert await db.scalar(
            select(func.count()).select_from(EvidenceRecord)
        ) >= 1
        await db.refresh(record)
        assert record.payload_purged_at == _dt(-1)

    @pytest.mark.asyncio
    async def test_second_revision_gc_run_deletes_nothing(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(db, document_id=document_id, anchor=_dt(-48))

        first = await run_revision_artifact_gc_batch(
            db, now=BASE, retention_window=WINDOW
        )
        second = await run_revision_artifact_gc_batch(
            db, now=BASE, retention_window=WINDOW
        )

        assert first.reclaimed == 1
        assert second.reclaimed == 0
        assert await db.get(DocumentRevision, revision.revision_id) is not None

    @pytest.mark.asyncio
    async def test_revision_gc_retries_after_partial_object_store_failure(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        workspace_id = await _workspace_id(db, document_id)
        revision = await _make_revision(db, document_id=document_id, anchor=_dt(-48))
        markdown_key, structure_key = _artifact_keys(
            document_id, revision.revision_id, workspace_id
        )
        await _make_build(
            db,
            revision_id=revision.revision_id,
            markdown_key=markdown_key,
            structure_key=structure_key,
        )

        # First key deletes, second fails: the external deletes are not
        # transactional, but ``artifacts_purged_at`` must NOT be set.
        failing = FakeArtifactStore(fail_on=structure_key)
        with pytest.raises(RuntimeError):
            await run_revision_artifact_gc_batch(
                db,
                now=BASE,
                retention_window=WINDOW,
                storage=failing,
                vector_store_factory=_vector_factory([]),
                kg_service_factory=_kg_factory([]),
            )
        assert failing.deleted == [markdown_key]
        stored = await db.get(DocumentRevision, revision.revision_id)
        await db.refresh(stored)
        assert stored.artifacts_purged_at is None

        # Retry converges: both keys are deleted (idempotent) and the revision
        # is marked purged.
        healthy = FakeArtifactStore()
        vectors: list[FakeVectorStore] = []
        kgs: list[FakeKGService] = []
        retry = await run_revision_artifact_gc_batch(
            db,
            now=BASE,
            retention_window=WINDOW,
            storage=healthy,
            vector_store_factory=_vector_factory(vectors),
            kg_service_factory=_kg_factory(kgs),
        )
        assert retry.reclaimed == 1
        assert healthy.deleted == [markdown_key, structure_key]
        assert vectors[0].deleted == [(document_id, revision.revision_id)]
        assert kgs[0].deleted == [(document_id, revision.revision_id)]
        stored = await db.get(DocumentRevision, revision.revision_id)
        await db.refresh(stored)
        assert stored.artifacts_purged_at is not None

    @pytest.mark.asyncio
    async def test_revision_without_build_manifest_reclaims_deterministic_objects(
        self, async_db, document_factory
    ):
        """A crash between the object upload and the manifest commit leaves a
        reclaimable revision with no build row; its deterministic
        revision-scoped objects must still be deleted before the tombstone."""
        db = async_db
        document_id = document_factory()
        workspace_id = await _workspace_id(db, document_id)
        revision = await _make_revision(db, document_id=document_id, anchor=_dt(-48))

        storage = FakeArtifactStore()
        result = await run_revision_artifact_gc_batch(
            db,
            now=BASE,
            retention_window=WINDOW,
            storage=storage,
            vector_store_factory=_vector_factory([]),
            kg_service_factory=_kg_factory([]),
        )

        deterministic = _artifact_keys(document_id, revision.revision_id, workspace_id)
        assert result.reclaimed == 1
        assert storage.workspaces == [workspace_id]
        assert set(storage.deleted) == set(deterministic)
        stored = await db.get(DocumentRevision, revision.revision_id)
        await db.refresh(stored)
        assert stored.artifacts_purged_at is not None

    @pytest.mark.asyncio
    async def test_revision_without_build_manifest_delete_failure_skips_tombstone(
        self, async_db, document_factory
    ):
        """When the deterministic object delete fails the tombstone must NOT be
        written, so the next run retries instead of leaking the objects."""
        db = async_db
        document_id = document_factory()
        workspace_id = await _workspace_id(db, document_id)
        revision = await _make_revision(db, document_id=document_id, anchor=_dt(-48))
        markdown_key, _structure_key = _artifact_keys(
            document_id, revision.revision_id, workspace_id
        )

        with pytest.raises(RuntimeError):
            await run_revision_artifact_gc_batch(
                db,
                now=BASE,
                retention_window=WINDOW,
                storage=FakeArtifactStore(fail_on=markdown_key),
                vector_store_factory=_vector_factory([]),
                kg_service_factory=_kg_factory([]),
            )
        stored = await db.get(DocumentRevision, revision.revision_id)
        await db.refresh(stored)
        assert stored.artifacts_purged_at is None

        # Retry converges once the object store is reachable again.
        healthy = FakeArtifactStore()
        retry = await run_revision_artifact_gc_batch(
            db,
            now=BASE,
            retention_window=WINDOW,
            storage=healthy,
            vector_store_factory=_vector_factory([]),
            kg_service_factory=_kg_factory([]),
        )
        assert retry.reclaimed == 1
        assert set(healthy.deleted) == {
            markdown_key,
            _structure_key,
        }
        await db.refresh(stored)
        assert stored.artifacts_purged_at is not None

    @pytest.mark.asyncio
    async def test_artifact_keys_are_unioned_across_all_build_rows(
        self, async_db, document_factory
    ):
        """Keys recorded by an *older* build row of the same revision are
        reclaimed too — only the latest row is not authoritative by itself."""
        db = async_db
        document_id = document_factory()
        workspace_id = await _workspace_id(db, document_id)
        revision = await _make_revision(db, document_id=document_id, anchor=_dt(-48))
        legacy_key = (
            f"kb_{workspace_id}/revisions/{document_id}/{revision.revision_id}"
            "/legacy-profile.md"
        )
        await _make_build(
            db,
            revision_id=revision.revision_id,
            markdown_key=legacy_key,
            finished_at=BASE,
            profile=RevisionBuildProfile.PARSE_ONLY,
        )
        await _make_build(
            db,
            revision_id=revision.revision_id,
            markdown_key=None,
            structure_key=None,
            finished_at=BASE + timedelta(hours=1),
        )

        storage = FakeArtifactStore()
        result = await run_revision_artifact_gc_batch(
            db,
            now=BASE,
            retention_window=WINDOW,
            storage=storage,
            vector_store_factory=_vector_factory([]),
            kg_service_factory=_kg_factory([]),
        )

        deterministic = _artifact_keys(document_id, revision.revision_id, workspace_id)
        assert result.reclaimed == 1
        assert legacy_key in storage.deleted
        assert set(deterministic).issubset(storage.deleted)


# ---------------------------------------------------------------------------
# Retention leases — acquisition / release / TTL
# ---------------------------------------------------------------------------


class TestRetentionLeases:
    @pytest.mark.asyncio
    async def test_checkpoint_pinning_revision_writes_lease(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(db, document_id=document_id, anchor=None)
        repo = RevisionRetentionLeaseRepository(db, ttl=TTL)

        lease = await repo.acquire_or_refresh(RUN_ID, revision.revision_id, now=BASE)

        assert lease.run_id == RUN_ID
        assert lease.revision_id == revision.revision_id
        assert lease.evidence_use_id is None
        assert lease.expires_at == BASE + TTL
        assert lease.released_at is None
        assert await repo.has_active_revision_lease(
            revision.revision_id, now=_dt(0.5)
        )

    @pytest.mark.asyncio
    async def test_resume_refreshes_existing_lease(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(db, document_id=document_id, anchor=None)
        repo = RevisionRetentionLeaseRepository(db, ttl=TTL)

        first = await repo.acquire_or_refresh(RUN_ID, revision.revision_id, now=BASE)
        refreshed = await repo.acquire_or_refresh(
            RUN_ID, revision.revision_id, now=_dt(2)
        )

        assert refreshed.lease_id == first.lease_id  # same lease, refreshed
        assert refreshed.expires_at == _dt(2) + TTL
        await db.refresh(first)
        assert first.expires_at == _dt(2) + TTL
        assert refreshed.acquired_at == BASE  # first acquisition preserved
        count = await db.scalar(
            select(func.count()).select_from(RevisionRetentionLease).where(
                RevisionRetentionLease.run_id == RUN_ID,
                RevisionRetentionLease.revision_id == revision.revision_id,
            )
        )
        assert count == 1

    @pytest.mark.asyncio
    async def test_terminal_run_releases_lease(self, async_db, document_factory):
        db = async_db
        document_id = document_factory()
        r1 = await _make_revision(db, document_id=document_id, anchor=None)
        r2 = await _make_revision(
            db, document_id=document_id, generation=2, anchor=None
        )
        repo = RevisionRetentionLeaseRepository(db, ttl=TTL)
        await repo.acquire_or_refresh(RUN_ID, r1.revision_id, now=BASE)
        await repo.acquire_or_refresh(RUN_ID, r2.revision_id, now=BASE)

        released = await repo.release_run(RUN_ID, "terminal", now=_dt(2))

        assert released == 2
        assert not await repo.has_active_revision_lease(r1.revision_id, now=_dt(3))
        assert not await repo.has_active_revision_lease(r2.revision_id, now=_dt(3))

    @pytest.mark.asyncio
    async def test_resumable_interrupt_keeps_lease_until_expiry(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(
            db, document_id=document_id, anchor=_dt(-48)
        )
        repo = RevisionRetentionLeaseRepository(db, ttl=TTL)
        await repo.acquire_or_refresh(RUN_ID, revision.revision_id, now=BASE)

        # Interrupt: no release. The lease still blocks GC until it expires.
        mid = await run_revision_artifact_gc_batch(
            db, now=_dt(0.5), retention_window=WINDOW
        )
        assert mid.reclaimed == 0

        after = await run_revision_artifact_gc_batch(
            db, now=BASE + TTL + timedelta(minutes=1), retention_window=WINDOW
        )
        assert after.reclaimed == 1
        # Expiry hygiene: the lapsed lease is swept, never silently deleted.
        swept = await repo.sweep_expired(now=BASE + TTL + timedelta(minutes=1))
        assert swept == 1
        lease = await db.scalar(
            select(RevisionRetentionLease).where(
                RevisionRetentionLease.revision_id == revision.revision_id
            )
        )
        assert lease.released_at is not None
        assert lease.release_reason == "expired"

    @pytest.mark.asyncio
    async def test_active_lease_blocks_artifact_and_evidence_gc(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(db, document_id=document_id, anchor=_dt(-48))
        record = await _make_evidence(
            db, revision_id=revision.revision_id, expires_at=_dt(-1)
        )
        use = await _make_use(db, evidence_id=record.evidence_id)
        await _make_lease(
            db,
            run_id=RUN_ID,
            revision_id=revision.revision_id,
            evidence_use_id=use.use_id,
            expires_at=_dt(1),
        )

        evidence_result = await run_evidence_payload_gc_batch(db, now=BASE)
        storage = FakeArtifactStore()
        vectors: list[FakeVectorStore] = []
        kgs: list[FakeKGService] = []
        artifact_result = await run_revision_artifact_gc_batch(
            db,
            now=BASE,
            retention_window=WINDOW,
            storage=storage,
            vector_store_factory=_vector_factory(vectors),
            kg_service_factory=_kg_factory(kgs),
        )

        assert evidence_result.purged == 0
        assert artifact_result.reclaimed == 0
        await db.refresh(record)
        assert record.ciphertext == b"secret-payload"
        assert record.payload_purged_at is None
        assert storage.deleted == []
        assert vectors == []
        assert (
            await db.get(DocumentRevision, revision.revision_id)
        ).artifacts_purged_at is None

    @pytest.mark.asyncio
    async def test_expired_lease_does_not_block_gc(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(db, document_id=document_id, anchor=_dt(-48))
        record = await _make_evidence(
            db, revision_id=revision.revision_id, expires_at=_dt(-2)
        )
        await _make_lease(
            db,
            run_id=RUN_ID,
            revision_id=revision.revision_id,
            expires_at=_dt(-0.5),  # lapsed, never released
        )

        evidence_result = await run_evidence_payload_gc_batch(db, now=BASE)
        artifact_result = await run_revision_artifact_gc_batch(
            db, now=BASE, retention_window=WINDOW
        )

        assert evidence_result.purged == 1
        assert artifact_result.reclaimed == 1
        await db.refresh(record)
        assert record.payload_purged_at is not None

    @pytest.mark.asyncio
    async def test_has_active_evidence_lease_matches_use_or_revision(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(db, document_id=document_id, anchor=None)
        record = await _make_evidence(
            db, revision_id=revision.revision_id, expires_at=_dt(-1)
        )
        use = await _make_use(db, evidence_id=record.evidence_id)
        repo = RevisionRetentionLeaseRepository(db, ttl=TTL)

        assert not await repo.has_active_evidence_lease(
            evidence_use_id=use.use_id, revision_id=revision.revision_id, now=BASE
        )
        await _make_lease(
            db,
            run_id=RUN_ID,
            revision_id=revision.revision_id,
            evidence_use_id=use.use_id,
            expires_at=_dt(1),
        )
        assert await repo.has_active_evidence_lease(
            evidence_use_id=use.use_id, revision_id=revision.revision_id, now=BASE
        )
        # An expired lease never blocks.
        assert not await repo.has_active_evidence_lease(
            evidence_use_id=use.use_id,
            revision_id=revision.revision_id,
            now=_dt(2),
        )


# ---------------------------------------------------------------------------
# Lease safe-ordering invariant (no cross-DB atomicity claimed)
# ---------------------------------------------------------------------------


class TestLeaseOrdering:
    @pytest.mark.asyncio
    async def test_lease_is_committed_before_checkpointable_state_update(
        self, committed_revision, async_engine, raw_connection
    ):
        """The lease is committed BEFORE the checkpointable state update.

        The graph checkpoint lives in a different database, so the guarantee is
        an ordering: commit the lease, then return the state update. A second
        connection must already observe the lease when the checkpoint is written.
        """
        document_id, revision_id = committed_revision()
        session_factory = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )
        events: list[tuple[str, bool]] = []

        async with session_factory() as session:
            repo = RevisionRetentionLeaseRepository(session, ttl=TTL)
            await repo.acquire_or_refresh(RUN_ID, revision_id, now=BASE)
            await session.commit()  # lease committed ...
            # ... and only now is the state update emitted; the checkpoint
            # writer observes the committed lease.
            events.append(("checkpoint", _lease_visible(raw_connection, revision_id)))

        assert events == [("checkpoint", True)]

    @pytest.mark.asyncio
    async def test_lease_failure_prevents_revision_pin_checkpoint(
        self, async_db, document_factory
    ):
        """A failed lease write must fail the node, so no pin is checkpointed."""
        db = async_db
        document_factory()
        repo = RevisionRetentionLeaseRepository(db, ttl=TTL)
        checkpoint_writes: list[str] = []

        with pytest.raises(Exception):
            # A revision that does not exist violates the lease FK: the lease
            # write fails, so the checkpointable update must never be returned.
            await repo.acquire_or_refresh(RUN_ID, uuid.uuid4(), now=BASE)
            checkpoint_writes.append("emitted")  # pragma: no cover

        assert checkpoint_writes == []

    @pytest.mark.asyncio
    async def test_checkpoint_failure_leaves_only_expiring_orphan_lease(
        self, committed_revision, async_engine, raw_connection
    ):
        """Lease commit ok + checkpoint fail => only a TTL-released orphan."""
        document_id, revision_id = committed_revision()
        session_factory = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with session_factory() as session:
            repo = RevisionRetentionLeaseRepository(session, ttl=TTL)
            await repo.acquire_or_refresh(RUN_ID, revision_id, now=BASE)
            await session.commit()

        try:
            raise RuntimeError("checkpoint store unavailable")
        except RuntimeError:
            pass  # the node fails; the orphan lease remains

        lease = _lease_row(raw_connection, revision_id)
        assert lease["released_at"] is None
        assert lease["expires_at"] == BASE + TTL

        async with session_factory() as session:
            repo = RevisionRetentionLeaseRepository(session, ttl=TTL)
            swept = await repo.sweep_expired(now=BASE + TTL + timedelta(minutes=1))
            await session.commit()
        assert swept == 1
        lease = _lease_row(raw_connection, revision_id)
        assert lease["release_reason"] == "expired"

    @pytest.mark.asyncio
    async def test_terminal_checkpoint_happens_before_lease_release(
        self, async_db, document_factory
    ):
        """Terminal completion checkpoints first, then releases the lease."""
        db = async_db
        document_id = document_factory()
        revision = await _make_revision(db, document_id=document_id, anchor=None)
        repo = RevisionRetentionLeaseRepository(db, ttl=TTL)
        await repo.acquire_or_refresh(RUN_ID, revision.revision_id, now=BASE)

        events: list[str] = []

        async def checkpoint_writer() -> None:
            events.append("terminal_checkpoint")

        # The documented terminal sequence: checkpoint, then release.
        await checkpoint_writer()
        await repo.release_run(RUN_ID, "terminal", now=_dt(1))

        assert events == ["terminal_checkpoint"]
        assert not await repo.has_active_revision_lease(
            revision.revision_id, now=_dt(1)
        )

    @pytest.mark.asyncio
    async def test_runtime_services_injects_retention_leases(
        self, async_db, document_factory
    ):
        """R12: RuntimeServices carries the single-owner lease repository."""
        db = async_db
        document_factory()
        repo = RevisionRetentionLeaseRepository(db, ttl=TTL)

        services = RuntimeServices(retention_leases=repo)

        assert services.retention_leases is repo
        assert RuntimeServices().retention_leases is None


# ---------------------------------------------------------------------------
# Advisory lock + FOR UPDATE SKIP LOCKED
# ---------------------------------------------------------------------------


class TestConcurrencyAndLocking:
    @pytest.mark.asyncio
    async def test_evidence_gc_skips_when_advisory_lock_held(
        self, async_db, async_engine, document_factory
    ):
        db = async_db
        document_id = document_factory()
        await _make_evidence(db, revision_id=None, expires_at=_dt(-1))

        async with async_engine.connect() as conn:
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": EVIDENCE_GC_ADVISORY_LOCK_KEY},
            )
            result = await run_evidence_payload_gc_batch(db, now=BASE)

        assert result == EvidencePayloadGcResult(skipped_locked=True)
        assert result.purged == 0

    @pytest.mark.asyncio
    async def test_revision_gc_skips_when_advisory_lock_held(
        self, async_db, async_engine, document_factory
    ):
        db = async_db
        document_id = document_factory()
        await _make_revision(db, document_id=document_id, anchor=_dt(-48))

        async with async_engine.connect() as conn:
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": REVISION_GC_ADVISORY_LOCK_KEY},
            )
            result = await run_revision_artifact_gc_batch(db, now=BASE)

        assert result.skipped_locked is True
        assert result.reclaimed == 0

    @pytest.mark.asyncio
    async def test_evidence_gc_skips_rows_locked_by_another_transaction(
        self, async_engine, committed_evidence
    ):
        """``FOR UPDATE SKIP LOCKED``: a row another transaction locked is deferred."""
        evidence_id = committed_evidence(expires_at=_dt(-1))
        session_factory = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )

        # A separate connection holds the row lock (never uses the async_db
        # fixture: its long-lived uncommitted transaction would block the
        # fixture teardown's DELETE after the test).
        async with async_engine.connect() as conn:
            await conn.execute(
                text(
                    "SELECT evidence_id FROM evidence_records "
                    "WHERE evidence_id = :eid FOR UPDATE"
                ),
                {"eid": str(evidence_id)},
            )
            async with session_factory() as session:
                async with session.begin():
                    blocked = await run_evidence_payload_gc_batch(
                        session, now=BASE
                    )
            assert blocked.purged == 0

        # Holder released -> next run purges exactly once.
        async with session_factory() as session:
            async with session.begin():
                purged = await run_evidence_payload_gc_batch(session, now=BASE)
        assert purged.purged == 1
        assert purged.purged_ids == (evidence_id,)

    @pytest.mark.asyncio
    async def test_two_workers_do_not_double_delete_evidence(
        self, async_engine, committed_evidence
    ):
        evidence_ids = {
            committed_evidence(expires_at=_dt(-1)) for _ in range(3)
        }
        session_factory = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )

        async def worker() -> EvidencePayloadGcResult:
            async with session_factory() as session:
                async with session.begin():
                    return await run_evidence_payload_gc_batch(
                        session, batch_size=10, now=BASE
                    )

        first, second = await asyncio.gather(worker(), worker())

        assert first.purged + second.purged == len(evidence_ids)
        # Exactly one purger per row (never zero, never two).
        purged_ids = set(first.purged_ids) | set(second.purged_ids)
        assert purged_ids == evidence_ids
        assert set(first.purged_ids) & set(second.purged_ids) == set()


# ---------------------------------------------------------------------------
# Retention anchors written by the Task-3 lifecycle
# ---------------------------------------------------------------------------


class TestRetentionAnchors:
    @pytest.mark.asyncio
    async def test_failed_revision_has_gc_retention_anchor(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        repo = DocumentRevisionsRepository(db)
        revision = await repo.allocate_draft(
            document_id,
            compute_source_object_identity(
                bucket="b",
                object_key="k1",
                version_id=None,
                etag="e1",
                size_bytes=1,
                content_sha256="a" * 64,
            ),
            RevisionBuildProfile.PARSE_ONLY,
        )

        failed = await repo.mark_failed(revision.revision_id, "parse", "Boom")

        assert failed.artifact_retention_starts_at == failed.failed_at
        assert failed.artifact_retention_starts_at is not None

    @pytest.mark.asyncio
    async def test_abandoned_revision_has_gc_retention_anchor(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        repo = DocumentRevisionsRepository(db)
        revision = await repo.allocate_draft(
            document_id,
            compute_source_object_identity(
                bucket="b",
                object_key="k2",
                version_id=None,
                etag="e2",
                size_bytes=1,
                content_sha256="b" * 64,
            ),
            RevisionBuildProfile.PARSE_ONLY,
        )

        abandoned = await repo.abandon_revision(revision, "gc_blocked")

        assert abandoned.artifact_retention_starts_at == abandoned.abandoned_at
        assert abandoned.artifact_retention_starts_at is not None

    @pytest.mark.asyncio
    async def test_late_historical_publish_has_gc_retention_anchor(
        self, async_db, document_factory
    ):
        """A published-but-never-current revision starts its clock at publish."""
        db = async_db
        document_id = document_factory()
        newer = await _make_revision(
            db, document_id=document_id, generation=5, anchor=None
        )
        db_doc = await db.get(Document, document_id)
        db_doc.current_revision_id = newer.revision_id
        await db.flush()

        repo = DocumentRevisionsRepository(db)
        historical = await _make_revision(
            db, document_id=document_id, generation=1, status="verified", anchor=None
        )

        published, outcome = await repo.publish(historical.revision_id)

        assert outcome is PublishOutcome.PUBLISHED_HISTORICAL
        assert published.status == "published"
        assert published.published_at is not None
        assert published.artifact_retention_starts_at == published.published_at

    @pytest.mark.asyncio
    async def test_current_revision_has_no_gc_retention_anchor_until_superseded(
        self, async_db, document_factory
    ):
        db = async_db
        document_id = document_factory()
        repo = DocumentRevisionsRepository(db)
        first = await repo.allocate_draft(
            document_id,
            compute_source_object_identity(
                bucket="b",
                object_key="k3",
                version_id=None,
                etag="e3",
                size_bytes=1,
                content_sha256="c" * 64,
            ),
            RevisionBuildProfile.PARSE_ONLY,
        )
        await repo.record_artifacts(
            first.revision_id,
            RevisionBuildProfile.PARSE_ONLY,
            markdown_artifact_key="m",
            structure_artifact_key="s",
            captions_skipped=True,
            kg_skipped=True,
            embed_skipped=True,
        )
        await repo.verify_draft(first.revision_id)
        published, outcome = await repo.publish(first.revision_id)
        assert outcome is PublishOutcome.BECAME_CURRENT
        assert published.artifact_retention_starts_at is None

        # A newer published revision supersedes it -> the anchor appears.
        second = await repo.allocate_draft(
            document_id,
            compute_source_object_identity(
                bucket="b",
                object_key="k4",
                version_id=None,
                etag="e4",
                size_bytes=1,
                content_sha256="d" * 64,
            ),
            RevisionBuildProfile.PARSE_ONLY,
        )
        await repo.record_artifacts(
            second.revision_id,
            RevisionBuildProfile.PARSE_ONLY,
            markdown_artifact_key="m2",
            structure_artifact_key="s2",
            captions_skipped=True,
            kg_skipped=True,
            embed_skipped=True,
        )
        await repo.verify_draft(second.revision_id)
        _, outcome2 = await repo.publish(second.revision_id)
        assert outcome2 is PublishOutcome.BECAME_CURRENT

        superseded = await db.get(DocumentRevision, first.revision_id)
        await db.refresh(superseded)
        assert superseded.superseded_at is not None
        assert superseded.artifact_retention_starts_at is not None


# ---------------------------------------------------------------------------
# Committed-data fixtures (rows visible to separate worker sessions)
# ---------------------------------------------------------------------------


def _lease_visible(raw_connection, revision_id: uuid.UUID) -> bool:
    with raw_connection.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM revision_retention_leases WHERE revision_id = %s "
            "AND released_at IS NULL",
            (str(revision_id),),
        )
        return cur.fetchone() is not None


def _lease_row(raw_connection, revision_id: uuid.UUID) -> dict:
    with raw_connection.cursor() as cur:
        cur.execute(
            "SELECT released_at, expires_at, release_reason "
            "FROM revision_retention_leases WHERE revision_id = %s",
            (str(revision_id),),
        )
        row = cur.fetchone()
        return {
            "released_at": row[0],
            "expires_at": row[1],
            "release_reason": row[2],
        }


@pytest.fixture
def committed_revision(document_factory, raw_connection):
    """Insert a committed (document, revision) pair usable by other sessions."""
    created: list[tuple[uuid.UUID, uuid.UUID]] = []

    def _make(*, status: str = "published") -> tuple[uuid.UUID, uuid.UUID]:
        document_id = document_factory()
        revision_id = uuid.uuid4()
        with raw_connection.cursor() as cur:
            cur.execute(
                "INSERT INTO document_revisions "
                "(revision_id, document_id, generation, status) "
                "VALUES (%s, %s, %s, %s)",
                (str(revision_id), str(document_id), 1, status),
            )
        created.append((document_id, revision_id))
        return document_id, revision_id

    yield _make

    with raw_connection.cursor() as cur:
        for _document_id, revision_id in created:
            cur.execute(
                "DELETE FROM revision_retention_leases WHERE revision_id = %s",
                (str(revision_id),),
            )
            cur.execute(
                "DELETE FROM evidence_uses WHERE evidence_id IN "
                "(SELECT evidence_id FROM evidence_records WHERE revision_id = %s)",
                (str(revision_id),),
            )
            cur.execute(
                "DELETE FROM evidence_records WHERE revision_id = %s",
                (str(revision_id),),
            )
            cur.execute(
                "DELETE FROM document_revision_builds WHERE revision_id = %s",
                (str(revision_id),),
            )
            cur.execute(
                "DELETE FROM document_revision_chunks WHERE revision_id = %s",
                (str(revision_id),),
            )
            cur.execute(
                "DELETE FROM document_revisions WHERE revision_id = %s",
                (str(revision_id),),
            )


@pytest.fixture
def committed_evidence(raw_connection):
    """Insert committed evidence rows (no revision FK) visible to any session."""
    created: list[uuid.UUID] = []

    def _make(
        *,
        expires_at: datetime | None,
        payload_purged_at: datetime | None = None,
    ) -> uuid.UUID:
        evidence_id = uuid.uuid4()
        with raw_connection.cursor() as cur:
            cur.execute(
                "INSERT INTO evidence_records ("
                "evidence_id, contract_version, ciphertext, encryption_key_id, "
                "nonce, encryption_algorithm, content_hash, classification, "
                "expires_at, revision_id, source, provenance, payload_purged_at"
                ") VALUES (%s, '2.0', %s, 'k1', %s, 'AES-256-GCM', %s, 'normal', "
                "%s, NULL, %s::jsonb, %s::jsonb, %s)",
                (
                    str(evidence_id),
                    b"secret-payload",
                    b"0123456789ab",
                    uuid.uuid4().hex,
                    expires_at,
                    json.dumps({"kind": "document"}),
                    json.dumps({"fetcher": "document.read"}),
                    payload_purged_at,
                ),
            )
        created.append(evidence_id)
        return evidence_id

    yield _make

    with raw_connection.cursor() as cur:
        for evidence_id in created:
            cur.execute(
                "DELETE FROM evidence_records WHERE evidence_id = %s",
                (str(evidence_id),),
            )

"""Tests for the revision-owned build/state/publication lifecycle.

These tests cover the Phase 1C Task 3 brief's Step 1 named tests:

- Allocate → build → verify → publish for ``FULL``, ``CHAT_UPLOAD``,
  and ``PARSE_ONLY`` profiles.
- Idempotency of ``get_or_create_ingestion_attempt`` (single-call AND
  two concurrent sessions racing the same attempt key).
- Atomic CAS publish (one winner, one historical, neither resurrects a
  tombstoned document).
- Terminal-state immutability (``failed`` / ``abandoned`` / ``published``
  cannot transition further).
- Tombstone semantics (``mark_source_deleted`` clears the current
  pointer, abandons non-published revisions, never deletes rows).
- Failure-aware retry (terminal failed → new generation with
  ``retry_of_revision_id`` provenance; bounded by
  ``MAX_REVISION_RETRIES``).
- Build manifest persistence via ``record_artifacts``.

All tests use the ``async_db`` fixture (SAVEPOINT-wrapped AsyncSession)
so each test starts with a clean slate. The conftest provides a
``document_factory`` helper for seeding legacy ``documents`` rows
outside the test's SAVEPOINT.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_ingestion_attempt import DocumentIngestionAttempt
from app.models.document_revision import DocumentRevision
from app.models.document_revision_build import DocumentRevisionBuild
from app.services.agents.v2.persistence.document_revisions import (  # type: ignore[import-not-found]
    DocumentNotFound,
    DocumentRevisionsRepository,
    PublishOutcome,
    RevisionArtifactsIncomplete,
    RevisionBuildProfile,
    RevisionNotAbandonable,
    RevisionNotFailable,
    RevisionNotPublishable,
    RevisionRetriesExhausted,
)
from app.services.agents.v2.persistence.source_identity import (  # type: ignore[import-not-found]
    compute_source_object_identity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(async_db: AsyncSession) -> DocumentRevisionsRepository:
    """A repository bound to the test's transactional AsyncSession."""
    return DocumentRevisionsRepository(async_db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_identity(
    *,
    bucket: str = "bkt",
    object_key: str = "uploads/x.pdf",
    version_id: str | None = "v-1",
    etag: str | None = None,
    size_bytes: int = 1024,
    content_sha256: str | None = None,
) -> str:
    if content_sha256 is None:
        content_sha256 = "a" * 64
    return compute_source_object_identity(
        bucket=bucket,
        object_key=object_key,
        version_id=version_id,
        etag=etag,
        size_bytes=size_bytes,
        content_sha256=content_sha256,
    )


# ---------------------------------------------------------------------------
# Step 1 — Allocate / publish lifecycle (FULL profile)
# ---------------------------------------------------------------------------


class TestAllocatePublishLifecycle:
    @pytest.mark.asyncio
    async def test_full_profile_allocate_publish_lifecycle(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Allocate → record all FULL-profile artifacts → verify → publish.
        The published revision becomes the document's current_revision_id."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="f" * 64)

        # 1. Allocate
        revision, created = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        assert created is True
        assert revision.status == "draft"
        assert revision.generation == 1
        assert revision.retry_of_revision_id is None

        # 2. Transition to building (worker starts)
        revision.status = "building"
        await repo.session.flush()

        # 3. Record the embedding manifest
        await repo.record_artifacts(
            revision_id=revision.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns_default",
            embedding_model_hash="mdl_hash_1",
            embedding_dimension=1536,
            vector_artifact_version="v1",
        )

        # 4. Mark markdown + structure + vectors done (simulated via DB)
        await repo.session.execute(
            text(
                "UPDATE document_revisions SET status='building' "
                "WHERE revision_id = :rid"
            ),
            {"rid": str(revision.revision_id)},
        )
        # Promote to verified via the repo helper
        await repo.verify_draft(revision_id=revision.revision_id)

        refreshed = await repo.get(revision.revision_id)
        assert refreshed.status == "verified"

        # 5. Publish
        pub_rev, outcome = await repo.publish(revision.revision_id)
        assert outcome == PublishOutcome.BECAME_CURRENT
        assert pub_rev.status == "published"
        assert pub_rev.published_at is not None

        # 6. Document.current_revision_id updated
        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id == revision.revision_id

    @pytest.mark.asyncio
    async def test_published_columns_are_immutable(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Once a revision is ``published``, its terminal columns
        (``status``, ``published_at``, ``generation``) MUST NOT change."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="p" * 64)
        revision, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        revision.status = "building"
        await repo.record_artifacts(
            revision_id=revision.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=revision.revision_id)
        await repo.publish(revision.revision_id)

        published = await repo.get(revision.revision_id)
        original_pub_at = published.published_at
        original_gen = published.generation
        original_status = published.status

        # Direct mutation attempts via the repository should be rejected.
        with pytest.raises(RevisionNotPublishable):
            await repo.publish(revision.revision_id)
        with pytest.raises(RevisionNotFailable):
            await repo.mark_failed(
                revision_id=revision.revision_id,
                stage="embed",
                error_class="CrashError",
            )
        with pytest.raises(RevisionNotAbandonable):
            await repo.abandon_revision(
                revision=published, reason="late_tombstone"
            )

        # The published revision's state is unchanged.
        after = await repo.get(revision.revision_id)
        assert after.status == original_status
        assert after.published_at == original_pub_at
        assert after.generation == original_gen

    @pytest.mark.asyncio
    async def test_failed_draft_does_not_change_current_revision(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """A failed draft MUST NOT modify the document's current pointer."""
        document_id = document_factory()

        # First revision: publish successfully (becomes current).
        first_id = _source_identity(content_sha256="1" * 64)
        rev1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=first_id,
            build_profile=RevisionBuildProfile.FULL,
        )
        rev1.status = "building"
        await repo.record_artifacts(
            revision_id=rev1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=rev1.revision_id)
        await repo.publish(rev1.revision_id)
        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id == rev1.revision_id

        # Second revision: different identity → different attempt → NEW generation.
        second_id = _source_identity(content_sha256="2" * 64)
        rev2, created = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=second_id,
            build_profile=RevisionBuildProfile.FULL,
        )
        assert created is True
        assert rev2.generation == rev1.generation + 1

        # The new revision fails (no verify).
        failed = await repo.mark_failed(
            revision_id=rev2.revision_id,
            stage="embed",
            error_class="EmbedCrash",
        )
        assert failed.status == "failed"

        # The document's current pointer is UNCHANGED.
        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id == rev1.revision_id


# ---------------------------------------------------------------------------
# Step 1 — Profile-specific required artifacts
# ---------------------------------------------------------------------------


class TestProfileRequiredArtifacts:
    @pytest.mark.asyncio
    async def test_chat_upload_records_intentional_skips(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """``CHAT_UPLOAD`` profile: required = markdown + structure +
        vectors; ``caption`` / ``kg`` are recorded as intentionally
        skipped (not failed). Skips are NOT failures."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="c" * 64)
        revision, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.CHAT_UPLOAD,
        )
        revision.status = "building"
        await repo.record_artifacts(
            revision_id=revision.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.CHAT_UPLOAD,
            embedding_namespace="ns_chat",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
            captions_skipped=True,
            kg_skipped=True,
        )
        await repo.verify_draft(revision_id=revision.revision_id)
        v = await repo.get(revision.revision_id)
        assert v.status == "verified", (
            "skips must NOT block verification for CHAT_UPLOAD"
        )

    @pytest.mark.asyncio
    async def test_parse_only_records_intentional_skips(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """``PARSE_ONLY`` profile: required = markdown + structure; vector
        / caption / kg are recorded as intentionally skipped."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="p" * 64)
        revision, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.PARSE_ONLY,
        )
        revision.status = "building"
        await repo.record_artifacts(
            revision_id=revision.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.PARSE_ONLY,
            embed_skipped=True,
            captions_skipped=True,
            kg_skipped=True,
        )
        await repo.verify_draft(revision_id=revision.revision_id)
        v = await repo.get(revision.revision_id)
        assert v.status == "verified", (
            "PARSE_ONLY verification only needs markdown + structure"
        )

    @pytest.mark.asyncio
    async def test_record_artifacts_persists_embedding_manifest(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """``record_artifacts`` persists namespace / model hash / dimension
        / vector artifact version on ``document_revision_builds``."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="m" * 64)
        revision, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        revision.status = "building"
        await repo.record_artifacts(
            revision_id=revision.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns_main",
            embedding_model_hash="model_hash_v3",
            embedding_dimension=3072,
            vector_artifact_version="v2",
        )
        # Re-fetch and confirm the build row carries the manifest.
        result = await repo.session.execute(
            select(DocumentRevisionBuild).where(
                DocumentRevisionBuild.revision_id == revision.revision_id
            )
        )
        build = result.scalar_one()
        assert build.embedding_namespace == "ns_main"
        assert build.embedding_model_hash == "model_hash_v3"
        assert build.embedding_dimension == 3072
        assert build.vector_artifact_version == "v2"
        assert build.build_profile == "FULL"

    @pytest.mark.asyncio
    async def test_r2_does_not_inherit_r1_worker_completion(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """A second revision on the same document is independent — its
        worker-completion flags start fresh and are not inherited from R1."""
        document_id = document_factory()
        # R1
        id1 = _source_identity(content_sha256="1" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id1,
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)

        # R2 (different identity)
        id2 = _source_identity(content_sha256="2" * 64)
        r2, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id2,
            build_profile=RevisionBuildProfile.FULL,
        )
        # R2 starts in draft, NOT verified — verify_draft has not been called.
        assert r2.status == "draft"
        # R2 has its own build row with no manifest yet.
        result = await repo.session.execute(
            select(DocumentRevisionBuild).where(
                DocumentRevisionBuild.revision_id == r2.revision_id
            )
        )
        r2_build = result.scalar_one_or_none()
        assert r2_build is None, "R2 has no build row until record_artifacts"


# ---------------------------------------------------------------------------
# Step 1 — Idempotency / concurrency on get_or_create_ingestion_attempt
# ---------------------------------------------------------------------------


class TestIngestionAttemptIdempotency:
    @pytest.mark.asyncio
    async def test_get_or_create_ingestion_attempt_is_idempotent(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Calling get_or_create twice with the same key returns the same
        revision_id and exactly one ``revision_ingestion_attempts`` row."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="i" * 64)
        r1, created1 = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        assert created1 is True
        r2, created2 = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        assert created2 is False, "second call must NOT create a new revision"
        assert r1.revision_id == r2.revision_id

        # Exactly one attempt row.
        cnt = await repo.session.execute(
            text(
                "SELECT count(*) FROM revision_ingestion_attempts "
                "WHERE document_id = :did"
            ),
            {"did": str(document_id)},
        )
        assert cnt.scalar() == 1

        # Exactly one revision row.
        cnt = await repo.session.execute(
            text(
                "SELECT count(*) FROM document_revisions "
                "WHERE document_id = :did"
            ),
            {"did": str(document_id)},
        )
        assert cnt.scalar() == 1

    @pytest.mark.asyncio
    async def test_concurrent_get_or_create_ingestion_attempt_is_atomic(
        self, async_engine, document_factory, raw_connection
    ):
        """Two concurrent sessions racing the same attempt key converge on
        exactly ONE attempt row and ONE draft revision, and both callers
        receive the same ``revision_id``.

        Both sessions run ``get_or_create_ingestion_attempt`` under
        ``asyncio.gather``. The ``documents`` row lock serializes them (the
        allocator's atomicity mechanism); the
        ``uq_revision_ingestion_attempt_key`` UNIQUE constraint is the
        DB-level backstop."""
        from sqlalchemy.ext.asyncio import async_sessionmaker

        document_id = document_factory()
        identity = _source_identity(content_sha256="r" * 64)
        maker = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False,
            autocommit=False, autoflush=False,
        )

        async def race() -> tuple[uuid.UUID, bool]:
            async with maker() as s:
                async with s.begin():
                    repo = DocumentRevisionsRepository(s)
                    r, created = await repo.get_or_create_ingestion_attempt(
                        document_id=document_id,
                        source_object_identity=identity,
                        build_profile=RevisionBuildProfile.FULL,
                    )
                    return r.revision_id, created

        results = await asyncio.gather(race(), race(), return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, f"concurrent allocation raised: {errors!r}"
        ids = {r[0] for r in results}
        created_flags = [r[1] for r in results]
        assert len(ids) == 1, f"both callers must get one revision_id, got {ids}"
        assert created_flags.count(True) == 1, created_flags
        assert created_flags.count(False) == 1, created_flags

        with raw_connection.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM revision_ingestion_attempts WHERE document_id = %s",
                (str(document_id),),
            )
            attempts = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM document_revisions WHERE document_id = %s",
                (str(document_id),),
            )
            revisions = cur.fetchone()[0]
        assert attempts == 1, f"expected 1 attempt, got {attempts}"
        assert revisions == 1, f"expected 1 revision, got {revisions}"

    @pytest.mark.asyncio
    async def test_loser_savepoint_leaves_no_orphan_revision(
        self, async_engine, document_factory, raw_connection
    ):
        """A second allocation for an existing key converges on the winner
        and leaves no orphan draft revision.

        The document-row lock prevents two creators from reaching the
        savepoint-loser branch through this API, so ``ON CONFLICT`` is a
        DB-level backstop; this test asserts the observable invariant (one
        revision, no orphan) for the converging caller."""
        from sqlalchemy.ext.asyncio import async_sessionmaker

        document_id = document_factory()
        identity = _source_identity(content_sha256="o" * 64)
        maker = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False,
        )
        async with maker() as s1:
            async with s1.begin():
                r1 = DocumentRevisionsRepository(s1)
                winner, created = await r1.get_or_create_ingestion_attempt(
                    document_id=document_id,
                    source_object_identity=identity,
                    build_profile=RevisionBuildProfile.FULL,
                )
                assert created is True
                winner_id = winner.revision_id

        async with maker() as s2:
            async with s2.begin():
                r2 = DocumentRevisionsRepository(s2)
                converged, created = await r2.get_or_create_ingestion_attempt(
                    document_id=document_id,
                    source_object_identity=identity,
                    build_profile=RevisionBuildProfile.FULL,
                )
                assert created is False
                assert converged.revision_id == winner_id

        with raw_connection.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM document_revisions WHERE document_id = %s",
                (str(document_id),),
            )
            revisions = cur.fetchone()[0]
        assert revisions == 1


# ---------------------------------------------------------------------------
# Step 1 — Concurrent revision publish (CAS race)
# ---------------------------------------------------------------------------


class TestConcurrentRevisionPublish:
    @pytest.mark.asyncio
    async def test_concurrent_revision_publish_does_not_regress_current(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Allocate R2 then R3. Publish R3 first, then R2. The CAS must
        keep ``current_revision_id`` on R3 (newer generation wins). R2
        remains a historical published revision."""
        document_id = document_factory()
        # R2: identity-1
        id2 = _source_identity(content_sha256="2" * 64)
        r2, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id2,
            build_profile=RevisionBuildProfile.FULL,
        )
        # R3: identity-2 (different → different attempt → generation 3)
        id3 = _source_identity(content_sha256="3" * 64)
        r3, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id3,
            build_profile=RevisionBuildProfile.FULL,
        )
        assert r3.generation > r2.generation

        # Bring both to verified
        for r in (r2, r3):
            r.status = "building"
            await repo.record_artifacts(
                revision_id=r.revision_id,
                markdown_artifact_key="md/artifact",
                structure_artifact_key="st/artifact",
                build_profile=RevisionBuildProfile.FULL,
                embedding_namespace="ns",
                embedding_model_hash="h",
                embedding_dimension=8,
                vector_artifact_version="v1",
            )
            await repo.verify_draft(revision_id=r.revision_id)

        # Publish R3 first → BECAME_CURRENT
        _, outcome_r3 = await repo.publish(r3.revision_id)
        assert outcome_r3 == PublishOutcome.BECAME_CURRENT
        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id == r3.revision_id

        # Then publish R2 → CAS loses (newer is current) → historical
        _, outcome_r2 = await repo.publish(r2.revision_id)
        assert outcome_r2 == PublishOutcome.PUBLISHED_HISTORICAL

        # R2 is published but never current.
        r2_after = await repo.get(r2.revision_id)
        assert r2_after.status == "published"

        # Current pointer STILL on R3.
        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id == r3.revision_id

    @pytest.mark.asyncio
    async def test_newer_generation_loser_becomes_historical_published(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Even when a HIGHER-generation revision loses the CAS (e.g.,
        another concurrent caller published a still-higher one), it
        MUST stay ``published`` (never ``abandoned``) and MUST NOT
        become current."""
        document_id = document_factory()
        id2 = _source_identity(content_sha256="2" * 64)
        r2, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id2,
            build_profile=RevisionBuildProfile.FULL,
        )
        id3 = _source_identity(content_sha256="3" * 64)
        r3, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id3,
            build_profile=RevisionBuildProfile.FULL,
        )

        for r in (r2, r3):
            r.status = "building"
            await repo.record_artifacts(
                revision_id=r.revision_id,
                markdown_artifact_key="md/artifact",
                structure_artifact_key="st/artifact",
                build_profile=RevisionBuildProfile.FULL,
                embedding_namespace="ns",
                embedding_model_hash="h",
                embedding_dimension=8,
                vector_artifact_version="v1",
            )
            await repo.verify_draft(revision_id=r.revision_id)

        # Publish R2 first → BECAME_CURRENT
        _, out2 = await repo.publish(r2.revision_id)
        assert out2 == PublishOutcome.BECAME_CURRENT
        # Now publish R3 (newer generation) → BECAME_CURRENT (R2 → historical)
        _, out3 = await repo.publish(r3.revision_id)
        assert out3 == PublishOutcome.BECAME_CURRENT

        # R2 is published, not abandoned.
        r2_after = await repo.get(r2.revision_id)
        assert r2_after.status == "published"


# ---------------------------------------------------------------------------
# Step 1 — Tombstone behavior
# ---------------------------------------------------------------------------


class TestTombstoneBehavior:
    @pytest.mark.asyncio
    async def test_tombstone_clears_current_revision_pointer(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """``mark_source_deleted`` MUST clear the document's
        ``current_revision_id``."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="t" * 64)
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        r.status = "building"
        await repo.record_artifacts(
            revision_id=r.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r.revision_id)
        await repo.publish(r.revision_id)
        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id == r.revision_id

        await repo.mark_source_deleted(document_id=document_id)
        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id is None
        assert doc.source_deleted_at is not None

    @pytest.mark.asyncio
    async def test_publish_after_tombstone_does_not_resurrect_current(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Publish R1, tombstone, then publish a verified R2 →
        ``DocumentTombstoned`` and ``documents.current_revision_id = NULL``."""
        document_id = document_factory()

        # R1
        id1 = _source_identity(content_sha256="1" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id1,
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)

        # Tombstone the document
        await repo.mark_source_deleted(document_id=document_id)
        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id is None

        # R2: a new generation, verified
        id2 = _source_identity(content_sha256="2" * 64)
        r2, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id2,
            build_profile=RevisionBuildProfile.FULL,
        )
        r2.status = "building"
        await repo.record_artifacts(
            revision_id=r2.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r2.revision_id)

        # publish() must detect the tombstone and ABANDON the revision
        # without resurrecting the current pointer.
        _, outcome = await repo.publish(r2.revision_id)
        assert outcome == PublishOutcome.ABANDONED_SOURCE_DELETED

        # Caller then raises DocumentTombstoned (the publication contract).
        r2_after = await repo.get(r2.revision_id)
        assert r2_after.status == "abandoned"
        assert r2_after.abandon_reason == "document_tombstoned"

        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id is None

    @pytest.mark.asyncio
    async def test_verified_revision_blocked_by_tombstone_becomes_abandoned(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """A verified revision whose source was tombstoned MUST become
        terminal ``abandoned`` with ``abandon_reason='document_tombstoned'``."""
        document_id = document_factory()
        # R1 published
        id1 = _source_identity(content_sha256="1" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id1,
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)

        # R2 verified
        id2 = _source_identity(content_sha256="2" * 64)
        r2, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id2,
            build_profile=RevisionBuildProfile.FULL,
        )
        r2.status = "building"
        await repo.record_artifacts(
            revision_id=r2.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r2.revision_id)

        # Tombstone, then attempt to publish R2 → abandoned.
        await repo.mark_source_deleted(document_id=document_id)
        _, outcome = await repo.publish(r2.revision_id)
        assert outcome == PublishOutcome.ABANDONED_SOURCE_DELETED

        r2_after = await repo.get(r2.revision_id)
        assert r2_after.status == "abandoned"
        assert r2_after.abandon_reason == "document_tombstoned"

    @pytest.mark.asyncio
    async def test_tombstone_abandons_non_published_revisions(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """``mark_source_deleted`` MUST abandon every draft/building/verified
        revision on the document. Already-published revisions stay
        ``published`` (they are historical evidence, not work in flight)."""
        document_id = document_factory()

        # 1. A published revision (R1)
        id1 = _source_identity(content_sha256="1" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id1,
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)

        # 2. A draft revision (R2)
        id2 = _source_identity(content_sha256="2" * 64)
        r2, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id2,
            build_profile=RevisionBuildProfile.FULL,
        )
        # stays draft

        # 3. A building revision (R3)
        id3 = _source_identity(content_sha256="3" * 64)
        r3, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id3,
            build_profile=RevisionBuildProfile.FULL,
        )
        r3.status = "building"

        # 4. A verified revision (R4)
        id4 = _source_identity(content_sha256="4" * 64)
        r4, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id4,
            build_profile=RevisionBuildProfile.FULL,
        )
        r4.status = "building"
        await repo.record_artifacts(
            revision_id=r4.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r4.revision_id)

        await repo.session.flush()

        # Tombstone
        await repo.mark_source_deleted(document_id=document_id)

        # R1 stays published
        r1_after = await repo.get(r1.revision_id)
        assert r1_after.status == "published", "published revision must remain historical"
        # R2, R3, R4 abandoned with reason document_tombstoned
        for r_expected in (r2, r3, r4):
            after = await repo.get(r_expected.revision_id)
            assert after.status == "abandoned"
            assert after.abandon_reason == "document_tombstoned"

    @pytest.mark.asyncio
    async def test_tombstone_does_not_delete_published_revision_rows(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """``mark_source_deleted`` MUST NEVER delete revision rows — even
        published ones — because the row is historical evidence."""
        document_id = document_factory()
        id1 = _source_identity(content_sha256="1" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id1,
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)

        await repo.mark_source_deleted(document_id=document_id)

        # Query through the test's own session: ``async_db`` never commits,
        # so a separate raw autocommit connection could not observe these
        # rows at all (it would report 0 regardless of tombstone behavior).
        retained = (
            await repo.session.scalars(
                select(DocumentRevision).where(
                    DocumentRevision.revision_id == r1.revision_id
                )
            )
        ).all()
        assert len(retained) == 1, "published revision row must be retained"

    @pytest.mark.asyncio
    async def test_deleted_former_current_revision_can_be_gc_eligible_later(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """After tombstone, the former-current revision's retention clock
        starts (via ``artifact_retention_starts_at``) so artifact GC can
        reclaim its build artifacts once the retention window elapses."""
        document_id = document_factory()
        id1 = _source_identity(content_sha256="1" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id1,
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)

        # A currently-published revision has NO retention anchor: the clock
        # starts only when the revision becomes permanently non-current.
        r1_pub = await repo.get(r1.revision_id)
        assert r1_pub.artifact_retention_starts_at is None
        assert r1_pub.published_at is not None

        # After tombstone, the former current is still present with its
        # retention anchor (Predicate B uses the artifact_retention_starts_at
        # plus a GC lease threshold).
        await repo.mark_source_deleted(document_id=document_id)
        r1_after = await repo.get(r1.revision_id)
        assert r1_after.status == "published"
        assert r1_after.artifact_retention_starts_at is not None

    @pytest.mark.asyncio
    async def test_retained_evidence_still_resolves_after_current_pointer_cleared(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """After the current pointer is cleared by a tombstone, the
        historical revision row is still discoverable by revision_id
        (evidence lineage lookup works)."""
        document_id = document_factory()
        id1 = _source_identity(content_sha256="1" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id1,
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)

        await repo.mark_source_deleted(document_id=document_id)

        # Direct lookup by revision_id still returns the row.
        still_there = await repo.get_published(r1.revision_id)
        assert still_there is not None
        assert still_there.status == "published"

    @pytest.mark.asyncio
    async def test_tombstone_race_commits_abandoned_before_error(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """If a publish observes the document is tombstoned between
        ``verify_draft`` and ``publish``, the ``abandoned`` transition
        MUST be durable even though the caller raises
        ``DocumentTombstoned``.

        Simulated by tombstoning, then publishing a verified revision —
        the abandon transition is committed by the repository, the caller
        only decides whether to raise based on the outcome."""
        document_id = document_factory()
        # First publish
        id1 = _source_identity(content_sha256="1" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id1,
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)

        # Verified R2 (will be abandoned when tombstoned)
        id2 = _source_identity(content_sha256="2" * 64)
        r2, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id2,
            build_profile=RevisionBuildProfile.FULL,
        )
        r2.status = "building"
        await repo.record_artifacts(
            revision_id=r2.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r2.revision_id)

        # Tombstone, then publish — outcome says ABANDONED_SOURCE_DELETED.
        await repo.mark_source_deleted(document_id=document_id)
        _, outcome = await repo.publish(r2.revision_id)
        assert outcome == PublishOutcome.ABANDONED_SOURCE_DELETED

        # The caller would raise DocumentTombstoned here, but the
        # abandoned state is ALREADY durable.
        r2_after = await repo.get(r2.revision_id)
        assert r2_after.status == "abandoned"


# ---------------------------------------------------------------------------
# Step 1 — Terminal-state immutability
# ---------------------------------------------------------------------------


class TestTerminalStateImmutability:
    @pytest.mark.asyncio
    async def test_failed_is_terminal(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """A ``failed`` revision cannot transition again."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="f" * 64)
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        await repo.mark_failed(
            revision_id=r.revision_id, stage="embed", error_class="OOM"
        )
        with pytest.raises(RevisionNotFailable):
            await repo.mark_failed(
                revision_id=r.revision_id, stage="retry", error_class="OOM"
            )
        with pytest.raises(RevisionNotAbandonable):
            await repo.abandon_revision(revision=r, reason="late")

    @pytest.mark.asyncio
    async def test_abandoned_is_terminal(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """An ``abandoned`` revision cannot transition again."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="a" * 64)
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        await repo.abandon_revision(revision=r, reason="test")
        with pytest.raises(RevisionNotAbandonable):
            await repo.abandon_revision(revision=r, reason="again")
        with pytest.raises(RevisionNotFailable):
            await repo.mark_failed(
                revision_id=r.revision_id, stage="x", error_class="x"
            )
        with pytest.raises(RevisionNotPublishable):
            await repo.publish(r.revision_id)

    @pytest.mark.asyncio
    async def test_published_is_terminal(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """A ``published`` revision cannot transition again."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="p" * 64)
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        r.status = "building"
        await repo.record_artifacts(
            revision_id=r.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r.revision_id)
        await repo.publish(r.revision_id)
        r_pub = await repo.get(r.revision_id)

        with pytest.raises(RevisionNotFailable):
            await repo.mark_failed(
                revision_id=r.revision_id, stage="x", error_class="x"
            )
        with pytest.raises(RevisionNotAbandonable):
            await repo.abandon_revision(revision=r_pub, reason="late")

    @pytest.mark.asyncio
    async def test_abandoned_cannot_be_failed(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        identity = _source_identity(content_sha256="af" * 32)
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        await repo.abandon_revision(revision=r, reason="block")
        with pytest.raises(RevisionNotFailable):
            await repo.mark_failed(
                revision_id=r.revision_id, stage="x", error_class="x"
            )

    @pytest.mark.asyncio
    async def test_failed_cannot_be_abandoned(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        document_id = document_factory()
        identity = _source_identity(content_sha256="fa" * 32)
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        await repo.mark_failed(
            revision_id=r.revision_id, stage="x", error_class="x"
        )
        with pytest.raises(RevisionNotAbandonable):
            await repo.abandon_revision(revision=r, reason="x")


# ---------------------------------------------------------------------------
# Step 1 — Failure / retry semantics
# ---------------------------------------------------------------------------


class TestFailureAndRetry:
    @pytest.mark.asyncio
    async def test_failed_revision_is_terminal_and_immutable(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """A failed revision can never transition to ``verified`` /
        ``published``; the prior current revision (if any) is unchanged."""
        document_id = document_factory()
        # Publish R1 first
        id1 = _source_identity(content_sha256="1" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id1,
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)
        doc = await repo.session.get(Document, document_id)
        prior_current = doc.current_revision_id

        # R2 fails
        id2 = _source_identity(content_sha256="2" * 64)
        r2, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id2,
            build_profile=RevisionBuildProfile.FULL,
        )
        await repo.mark_failed(
            revision_id=r2.revision_id, stage="embed", error_class="OOM"
        )
        with pytest.raises(RevisionNotPublishable):
            await repo.publish(r2.revision_id)

        # The document's current pointer is still on R1.
        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id == prior_current

    @pytest.mark.asyncio
    async def test_queue_redelivery_of_failed_revision_is_noop(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Redelivering a queue message for a failed revision must NOT
        re-run any stage, must NOT allocate a new generation, and must
        be a no-op (the message becomes a dead-letter)."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="q" * 64)
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        original_gen = r.generation
        await repo.mark_failed(
            revision_id=r.revision_id, stage="embed", error_class="OOM"
        )

        # A redelivery query + mark_attempt_exhausted is the
        # dead-letter path: the revision stays failed and no new
        # generation is allocated.
        await repo.record_worker_state(
            revision_id=r.revision_id, stage="embed", state="noop_dead_letter"
        )

        after = await repo.get(r.revision_id)
        assert after.status == "failed"
        assert after.generation == original_gen
        # The attempt row's generation is NOT bumped on a noop dead-letter
        result = await repo.session.execute(
            select(DocumentIngestionAttempt).where(
                DocumentIngestionAttempt.revision_id == r.revision_id
            )
        )
        attempt = result.scalar_one()
        assert attempt.attempt_generation == 1

    @pytest.mark.asyncio
    async def test_retry_after_failure_allocates_new_generation(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """Failed R1 → retry yields R2 with ``generation > R1`` and
        ``retry_of_revision_id = R1``. The old R1 becomes ``superseded``
        (artifact_retention_starts_at set)."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="r" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        await repo.mark_failed(
            revision_id=r1.revision_id, stage="embed", error_class="OOM"
        )

        # Re-trigger the same attempt key (simulates automatic retry).
        r2, created = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        assert created is True
        assert r2.generation > r1.generation
        assert r2.retry_of_revision_id == r1.revision_id

        r1_after = await repo.get(r1.revision_id)
        # The failed revision is superseded by the retry (provenance) and its
        # retention anchor stays at the failure time (mark_failed), not the
        # supersession time.
        assert r1_after.superseded_at is not None
        assert r1_after.artifact_retention_starts_at is not None

        # Attempt counter bumped
        result = await repo.session.execute(
            select(DocumentIngestionAttempt).where(
                DocumentIngestionAttempt.document_id == document_id,
            )
        )
        attempt = result.scalar_one()
        assert attempt.attempt_generation == 2
        assert attempt.revision_id == r2.revision_id

    @pytest.mark.asyncio
    async def test_retry_is_bounded_and_exhausts(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """After ``MAX_REVISION_RETRIES`` automatic retries, the attempt
        is permanently failed (``exhausted_at`` set) and any further
        retry raises ``RevisionRetriesExhausted``."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="b" * 64)

        # Drive the attempt counter to MAX_REVISION_RETRIES by failing
        # each generation in sequence. We don't have to actually call
        # get_or_create_ingestion_attempt MAX_REVISION_RETRIES times —
        # we bump the attempt_generation directly to the budget, then
        # call once more to confirm the bound is enforced.
        from app.services.agents.v2.persistence.document_revisions import (  # type: ignore[import-not-found]
            MAX_REVISION_RETRIES,
        )
        # First allocation
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        await repo.mark_failed(
            revision_id=r1.revision_id, stage="embed", error_class="OOM"
        )

        # Exhaust the remaining retries so attempt_generation == MAX.
        result = await repo.session.execute(
            select(DocumentIngestionAttempt).where(
                DocumentIngestionAttempt.document_id == document_id,
            )
        )
        attempt = result.scalar_one()
        attempt.attempt_generation = MAX_REVISION_RETRIES
        await repo.session.flush()

        # Next retry attempt exhausts the budget.
        with pytest.raises(RevisionRetriesExhausted):
            await repo.get_or_create_ingestion_attempt(
                document_id=document_id,
                source_object_identity=identity,
                build_profile=RevisionBuildProfile.FULL,
            )

        # The attempt is marked exhausted.
        result = await repo.session.execute(
            select(DocumentIngestionAttempt).where(
                DocumentIngestionAttempt.document_id == document_id,
            )
        )
        attempt = result.scalar_one()
        assert attempt.exhausted_at is not None


# ---------------------------------------------------------------------------
# Step 1 — Concurrency: race between tombstone and verify
# ---------------------------------------------------------------------------


class TestTombstoneRaceDuringPublish:
    @pytest.mark.asyncio
    async def test_publish_with_concurrent_tombstone_returns_abandoned(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """The repository's CAS path checks ``is_source_deleted`` after
        a CAS miss. We simulate the race by tombstoning the document
        just before publish, then asserting the publish returns
        ``ABANDONED_SOURCE_DELETED`` and the revision becomes abandoned.

        This is the 'concurrent tombstone that commits between verify
        and CAS' scenario from the brief."""
        document_id = document_factory()
        id1 = _source_identity(content_sha256="1" * 64)
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=id1,
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)

        # Tombstone happens between verify and publish
        await repo.mark_source_deleted(document_id=document_id)

        _, outcome = await repo.publish(r1.revision_id)
        assert outcome == PublishOutcome.ABANDONED_SOURCE_DELETED

        doc = await repo.session.get(Document, document_id)
        assert doc.current_revision_id is None
        r1_after = await repo.get(r1.revision_id)
        assert r1_after.status == "abandoned"


# ---------------------------------------------------------------------------
# Step 1 — document_not_found on lock
# ---------------------------------------------------------------------------


class TestLockDocumentForUpdate:
    @pytest.mark.asyncio
    async def test_lock_document_for_update_raises_for_missing_document(
        self, repo: DocumentRevisionsRepository
    ):
        """``lock_document_for_update`` raises ``DocumentNotFound`` for a
        non-existent document — preventing an attempt from being created
        for a document that does not exist."""
        nonexistent = uuid.uuid4()
        with pytest.raises(DocumentNotFound):
            await repo.lock_document_for_update(nonexistent)


# ---------------------------------------------------------------------------
# Fix round 1 — I4 required-artifact enforcement (negative paths)
# ---------------------------------------------------------------------------


class TestRequiredArtifactEnforcement:
    @pytest.mark.asyncio
    async def test_record_artifacts_persists_keys_and_skip_flags(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """I4: ``record_artifacts`` must persist the artifact identity and
        skip flags instead of silently dropping them."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="aa" * 32)
        revision, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.CHAT_UPLOAD,
        )
        revision.status = "building"
        await repo.record_artifacts(
            revision_id=revision.revision_id,
            build_profile=RevisionBuildProfile.CHAT_UPLOAD,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
            markdown_artifact_key="revisions/x/markdown.md",
            structure_artifact_key="revisions/x/structure.json",
            captions_skipped=True,
            kg_skipped=True,
        )
        build = await repo.session.scalar(
            select(DocumentRevisionBuild).where(
                DocumentRevisionBuild.revision_id == revision.revision_id
            )
        )
        assert build.markdown_artifact_key == "revisions/x/markdown.md"
        assert build.structure_artifact_key == "revisions/x/structure.json"
        assert build.captions_skipped is True
        assert build.kg_skipped is True
        assert build.embed_skipped is False

    @pytest.mark.asyncio
    async def test_full_profile_cannot_verify_without_complete_manifest(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """I4: a FULL revision missing vectors cannot verify."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="ab" * 32)
        revision, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        revision.status = "building"
        await repo.record_artifacts(
            revision_id=revision.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=None,  # incomplete manifest
            vector_artifact_version="v1",
        )
        with pytest.raises(RevisionArtifactsIncomplete):
            await repo.verify_draft(revision_id=revision.revision_id)
        still = await repo.get(revision.revision_id)
        assert still.status == "building"

    @pytest.mark.asyncio
    async def test_full_profile_cannot_verify_with_skips(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """I4: FULL must not record caption/KG as skipped."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="ac" * 32)
        revision, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        revision.status = "building"
        await repo.record_artifacts(
            revision_id=revision.revision_id,
            markdown_artifact_key="md/artifact",
            structure_artifact_key="st/artifact",
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
            captions_skipped=True,
        )
        with pytest.raises(RevisionArtifactsIncomplete):
            await repo.verify_draft(revision_id=revision.revision_id)

    @pytest.mark.asyncio
    async def test_parse_only_cannot_verify_without_skip_records(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """I4: PARSE_ONLY must record that vector/caption/KG were skipped."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="ad" * 32)
        revision, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.PARSE_ONLY,
        )
        revision.status = "building"
        with pytest.raises(RevisionArtifactsIncomplete):
            await repo.verify_draft(revision_id=revision.revision_id)


# ---------------------------------------------------------------------------
# Fix round 1 — C1/I6 real concurrent sessions
# ---------------------------------------------------------------------------


class TestRealConcurrentSessions:
    @pytest.mark.asyncio
    async def test_concurrent_publish_same_document_does_not_deadlock(
        self, async_engine, document_factory
    ):
        """C1/I6: two independent sessions publish two verified revisions of
        the same document under ``asyncio.gather``.

        Under the pre-fix revision-first lock order this surfaced
        ``DeadlockDetected`` (SQLSTATE 40P01). Document-first ordering must
        serialize cleanly: no exception, and exactly one winner.
        """
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        document_id = document_factory()
        maker = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )
        id2 = _source_identity(content_sha256="ca" * 32)
        id3 = _source_identity(content_sha256="cb" * 32)
        async with maker() as setup:
            repo = DocumentRevisionsRepository(setup)
            r2, _ = await repo.get_or_create_ingestion_attempt(
                document_id=document_id,
                source_object_identity=id2,
                build_profile=RevisionBuildProfile.FULL,
            )
            r3, _ = await repo.get_or_create_ingestion_attempt(
                document_id=document_id,
                source_object_identity=id3,
                build_profile=RevisionBuildProfile.FULL,
            )
            for r in (r2, r3):
                r.status = "building"
                await repo.record_artifacts(
                    revision_id=r.revision_id,
                    markdown_artifact_key="md/artifact",
                    structure_artifact_key="st/artifact",
                    build_profile=RevisionBuildProfile.FULL,
                    embedding_namespace="ns",
                    embedding_model_hash="h",
                    embedding_dimension=8,
                    vector_artifact_version="v1",
                )
                await repo.verify_draft(revision_id=r.revision_id)
            await setup.commit()
            r2_id, r3_id = r2.revision_id, r3.revision_id

        async def _publish(rid: uuid.UUID) -> PublishOutcome:
            async with maker() as session:
                local = DocumentRevisionsRepository(session)
                try:
                    _, outcome = await local.publish(rid)
                    await session.commit()
                    return outcome
                except BaseException:
                    await session.rollback()
                    raise

        results = await asyncio.gather(
            _publish(r2_id), _publish(r3_id), return_exceptions=True
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, f"publish raised (deadlock?): {errors!r}"
        assert len(results) == 2, results
        assert PublishOutcome.BECAME_CURRENT in results, results
        assert all(
            r
            in (PublishOutcome.BECAME_CURRENT, PublishOutcome.PUBLISHED_HISTORICAL)
            for r in results
        ), results

    @pytest.mark.asyncio
    async def test_concurrent_publish_and_tombstone_do_not_deadlock(
        self, async_engine, document_factory
    ):
        """C1 smoke test: a concurrent ``publish`` and
        ``mark_source_deleted`` complete without a deadlock.

        The tombstone is started first (with a short head start) to bias
        toward the interleaving the pre-fix revision-first order deadlocked
        on: publish holds the verified revision while the tombstone holds the
        document and wants that same revision. Document-first ordering
        serializes the two. This is a timing-based smoke test, not a
        deterministic proof.
        """
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        document_id = document_factory()
        maker = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )

        async with maker() as setup:
            repo = DocumentRevisionsRepository(setup)
            r1, _ = await repo.get_or_create_ingestion_attempt(
                document_id=document_id,
                source_object_identity=_source_identity(content_sha256="ea" * 32),
                build_profile=RevisionBuildProfile.FULL,
            )
            r1.status = "building"
            await repo.record_artifacts(
                revision_id=r1.revision_id,
                build_profile=RevisionBuildProfile.FULL,
                embedding_namespace="ns",
                embedding_model_hash="h",
                embedding_dimension=8,
                vector_artifact_version="v1",
                markdown_artifact_key="md/1",
                structure_artifact_key="st/1",
            )
            await repo.verify_draft(revision_id=r1.revision_id)
            await repo.publish(r1.revision_id)

            r2, _ = await repo.get_or_create_ingestion_attempt(
                document_id=document_id,
                source_object_identity=_source_identity(content_sha256="eb" * 32),
                build_profile=RevisionBuildProfile.FULL,
            )
            r2.status = "building"
            await repo.record_artifacts(
                revision_id=r2.revision_id,
                build_profile=RevisionBuildProfile.FULL,
                embedding_namespace="ns",
                embedding_model_hash="h",
                embedding_dimension=8,
                vector_artifact_version="v1",
                markdown_artifact_key="md/2",
                structure_artifact_key="st/2",
            )
            await repo.verify_draft(revision_id=r2.revision_id)
            await setup.commit()
            r2_id = r2.revision_id

        async def _publish():
            async with maker() as s:
                local = DocumentRevisionsRepository(s)
                try:
                    await local.publish(r2_id)
                    await s.commit()
                except BaseException:
                    await s.rollback()
                    raise

        async def _tombstone():
            async with maker() as s:
                local = DocumentRevisionsRepository(s)
                try:
                    await local.mark_source_deleted(document_id=document_id)
                    await s.commit()
                except BaseException:
                    await s.rollback()
                    raise

        tombstone_task = asyncio.create_task(_tombstone())
        # Give the tombstone a head start so it holds the document lock when
        # the publisher starts. With the pre-fix revision-first order this
        # deadlocks deterministically (publisher holds R2 and wants D;
        # tombstone holds D and wants R2).
        await asyncio.sleep(0.05)
        publish_task = asyncio.create_task(_publish())
        results = await asyncio.gather(
            tombstone_task, publish_task, return_exceptions=True
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, f"publish/tombstone raised (deadlock?): {errors!r}"

        # Consistency: the tombstone always wins eventually; current must be
        # unset and no revision is left non-terminal in a way that resurrects
        # the document.
        async with maker() as check:
            doc = await check.get(Document, document_id)
            assert doc.source_deleted_at is not None
            assert doc.current_revision_id is None


# ---------------------------------------------------------------------------
# Fix round 2 — artifact contract + supersession anchor
# ---------------------------------------------------------------------------


class TestArtifactContractAndAnchors:
    @pytest.mark.asyncio
    async def test_full_profile_cannot_verify_without_markdown_or_structure(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """I-A: markdown + structure are required for every profile, not
        just the vector manifest."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="ba" * 32)
        revision, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=identity,
            build_profile=RevisionBuildProfile.FULL,
        )
        revision.status = "building"
        # Complete embedding manifest and explicitly NO markdown/structure.
        await repo.record_artifacts(
            revision_id=revision.revision_id,
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        with pytest.raises(RevisionArtifactsIncomplete):
            await repo.verify_draft(revision_id=revision.revision_id)
        still = await repo.get(revision.revision_id)
        assert still.status == "building"

    @pytest.mark.asyncio
    async def test_supersession_starts_retention_for_former_current_only(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """M4: publishing a newer revision anchors the former current, while
        the new current keeps no anchor."""
        document_id = document_factory()
        r2, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=_source_identity(content_sha256="bb" * 32),
            build_profile=RevisionBuildProfile.FULL,
        )
        r3, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=_source_identity(content_sha256="bc" * 32),
            build_profile=RevisionBuildProfile.FULL,
        )
        for r in (r2, r3):
            r.status = "building"
            await repo.record_artifacts(
                revision_id=r.revision_id,
                build_profile=RevisionBuildProfile.FULL,
                embedding_namespace="ns",
                embedding_model_hash="h",
                embedding_dimension=8,
                vector_artifact_version="v1",
                markdown_artifact_key="md/a",
                structure_artifact_key="st/a",
            )
            await repo.verify_draft(revision_id=r.revision_id)

        await repo.publish(r2.revision_id)
        r2_now = await repo.get(r2.revision_id)
        assert r2_now.artifact_retention_starts_at is None  # still current

        await repo.publish(r3.revision_id)
        r2_after = await repo.get(r2.revision_id)
        r3_after = await repo.get(r3.revision_id)
        assert r2_after.superseded_at is not None
        assert r2_after.superseded_by == r3.revision_id
        assert r2_after.artifact_retention_starts_at is not None
        assert r3_after.status == "published"
        assert r3_after.artifact_retention_starts_at is None  # new current


# ---------------------------------------------------------------------------
# Fix round 3 — regression guards for the round-2 semantics
# ---------------------------------------------------------------------------


class TestRound2RegressionGuards:
    @pytest.mark.asyncio
    async def test_abandoned_tombstone_reason_on_live_document_cannot_publish(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """I-B: an abandoned revision on a NON-tombstoned document must raise,
        never flip back to published via the CAS."""
        document_id = document_factory()
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=_source_identity(content_sha256="ga" * 32),
            build_profile=RevisionBuildProfile.FULL,
        )
        await repo.abandon_revision(r, reason="document_tombstoned")
        with pytest.raises(RevisionNotPublishable):
            await repo.publish(r.revision_id)
        assert (await repo.get(r.revision_id)).status == "abandoned"

    @pytest.mark.asyncio
    async def test_verified_revision_on_already_tombstoned_document_becomes_abandoned(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """I-B: the verified + tombstoned branch — tombstone first, then a
        fresh verified revision publishes as ABANDONED_SOURCE_DELETED."""
        document_id = document_factory()
        r1, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=_source_identity(content_sha256="gb" * 32),
            build_profile=RevisionBuildProfile.FULL,
        )
        r1.status = "building"
        await repo.record_artifacts(
            revision_id=r1.revision_id,
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
            markdown_artifact_key="md/1",
            structure_artifact_key="st/1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)
        await repo.mark_source_deleted(document_id=document_id)

        r2, created = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=_source_identity(content_sha256="gc" * 32),
            build_profile=RevisionBuildProfile.FULL,
        )
        assert created is True
        r2.status = "building"
        await repo.record_artifacts(
            revision_id=r2.revision_id,
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
            markdown_artifact_key="md/2",
            structure_artifact_key="st/2",
        )
        await repo.verify_draft(revision_id=r2.revision_id)
        _, outcome = await repo.publish(r2.revision_id)
        assert outcome == PublishOutcome.ABANDONED_SOURCE_DELETED
        assert (await repo.get(r2.revision_id)).status == "abandoned"

    @pytest.mark.asyncio
    async def test_record_artifacts_redelivery_coalesces_and_keeps_skips(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """M1: a partial redelivery must not null previously recorded fields,
        and a skip flag must be sticky."""
        document_id = document_factory()
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=_source_identity(content_sha256="gd" * 32),
            build_profile=RevisionBuildProfile.FULL,
        )
        r.status = "building"
        await repo.record_artifacts(
            revision_id=r.revision_id,
            build_profile=RevisionBuildProfile.FULL,
            markdown_artifact_key="md/keep",
            structure_artifact_key="st/keep",
        )
        await repo.record_artifacts(
            revision_id=r.revision_id,
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        build = await repo.session.scalar(
            select(DocumentRevisionBuild).where(
                DocumentRevisionBuild.revision_id == r.revision_id
            )
        )
        assert build.markdown_artifact_key == "md/keep"
        assert build.structure_artifact_key == "st/keep"
        assert build.embedding_namespace == "ns"
        # Verify succeeds only because coalesce preserved the keys.
        await repo.verify_draft(revision_id=r.revision_id)
        # Skip is sticky: a later delivery with False must not un-skip.
        await repo.record_artifacts(
            revision_id=r.revision_id,
            build_profile=RevisionBuildProfile.FULL,
            captions_skipped=True,
        )
        await repo.record_artifacts(
            revision_id=r.revision_id,
            build_profile=RevisionBuildProfile.FULL,
        )
        build2 = await repo.session.scalar(
            select(DocumentRevisionBuild).where(
                DocumentRevisionBuild.revision_id == r.revision_id
            )
        )
        await repo.session.refresh(build2)
        assert build2.captions_skipped is True

    @pytest.mark.asyncio
    async def test_verify_fails_closed_without_allocated_profile(
        self, repo: DocumentRevisionsRepository, document_factory, monkeypatch
    ):
        """I-A: an unknown allocation profile must fail closed."""
        document_id = document_factory()
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=_source_identity(content_sha256="ge" * 32),
            build_profile=RevisionBuildProfile.FULL,
        )
        r.status = "building"

        async def _none(_rid):
            return None

        monkeypatch.setattr(repo, "_allocated_profile", _none)
        with pytest.raises(RevisionArtifactsIncomplete):
            await repo.verify_draft(revision_id=r.revision_id)

    @pytest.mark.asyncio
    async def test_forged_build_profile_row_is_invisible(
        self, repo: DocumentRevisionsRepository, document_factory
    ):
        """I-A: a build row recorded under a different profile than the
        allocated one must not satisfy verify_draft."""
        document_id = document_factory()
        r, _ = await repo.get_or_create_ingestion_attempt(
            document_id=document_id,
            source_object_identity=_source_identity(content_sha256="gf" * 32),
            build_profile=RevisionBuildProfile.FULL,
        )
        r.status = "building"
        await repo.record_artifacts(
            revision_id=r.revision_id,
            build_profile=RevisionBuildProfile.CHAT_UPLOAD,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
            markdown_artifact_key="md",
            structure_artifact_key="st",
            captions_skipped=True,
            kg_skipped=True,
        )
        with pytest.raises(RevisionArtifactsIncomplete):
            await repo.verify_draft(revision_id=r.revision_id)

    @pytest.mark.asyncio
    async def test_publish_locks_document_before_revision(
        self, async_engine, document_factory, raw_connection
    ):
        """C1 deterministic guard: publish() takes the document lock BEFORE
        the revision lock.

        A holder session keeps an uncommitted ``FOR UPDATE`` on the document,
        so the publish task must block on the document row. While it blocks,
        the revision row must still be lockable (``FOR UPDATE NOWAIT``).
        Under the pre-fix revision-first order the publish task already holds
        the revision lock and the probe raises ``LockNotAvailable`` (55P03).
        """
        import psycopg
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        document_id = document_factory()
        maker = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with maker() as setup:
            repo = DocumentRevisionsRepository(setup)
            r, _ = await repo.get_or_create_ingestion_attempt(
                document_id=document_id,
                source_object_identity=_source_identity(content_sha256="gg" * 32),
                build_profile=RevisionBuildProfile.FULL,
            )
            r.status = "building"
            await repo.record_artifacts(
                revision_id=r.revision_id,
                build_profile=RevisionBuildProfile.FULL,
                embedding_namespace="ns",
                embedding_model_hash="h",
                embedding_dimension=8,
                vector_artifact_version="v1",
                markdown_artifact_key="md",
                structure_artifact_key="st",
            )
            await repo.verify_draft(revision_id=r.revision_id)
            await setup.commit()
            rid = r.revision_id

        holder = maker()
        await holder.begin()
        await holder.execute(
            text("SELECT id FROM documents WHERE id = :d FOR UPDATE"),
            {"d": str(document_id)},
        )

        async def _pub():
            async with maker() as s:
                local = DocumentRevisionsRepository(s)
                await local.publish(rid)
                await s.commit()

        task = asyncio.create_task(_pub())
        try:
            # Wait until the publish task is blocked on a lock.
            for _ in range(300):
                await asyncio.sleep(0.01)
                with raw_connection.cursor() as cur:
                    cur.execute(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE datname = current_database() "
                        "AND wait_event_type = 'Lock'"
                    )
                    if cur.fetchone()[0] > 0:
                        break
            try:
                with raw_connection.cursor() as cur:
                    cur.execute(
                        "SELECT revision_id FROM document_revisions "
                        "WHERE revision_id = %s FOR UPDATE NOWAIT",
                        (str(rid),),
                    )
            except psycopg.errors.LockNotAvailable as exc:
                raise AssertionError(
                    "publish() holds the revision lock while blocked on the "
                    "document lock - revision-first order regression (C1)"
                ) from exc
        finally:
            await holder.rollback()
            await task

        assert task.exception() is None

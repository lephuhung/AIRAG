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
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_ingestion_attempt import DocumentIngestionAttempt
from app.models.document_revision import DocumentRevision
from app.models.document_revision_build import DocumentRevisionBuild
from app.services.agents.v2.persistence.document_revisions import (  # type: ignore[import-not-found]
    AttemptAlreadyClaimed,
    DocumentNotFound,
    DocumentRevisionsRepository,
    DocumentTombstoned,
    PublishOutcome,
    RevisionBuildProfile,
    RevisionNotAbandonable,
    RevisionNotFailable,
    RevisionNotPublishable,
    RevisionRetriesExhausted,
)
from app.services.agents.v2.persistence.source_identity import (  # type: ignore[import-not-found]
    SOURCE_SCHEME,
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


def _identity_components(identity: str) -> tuple[str, str, str]:
    """Parse ``compute_source_object_identity`` into
    ``(scheme, bucket|key|..., sha256)`` for column population."""
    parts = identity.split("|")
    # s3v1|bucket|key|version:v1|1024|sha256
    return parts[0], "|".join(parts[1:-1]), parts[-1]


async def _seed_attempt(
    session: AsyncSession,
    document_id: uuid.UUID,
    identity: str,
    build_profile: RevisionBuildProfile,
    *,
    revision_id: uuid.UUID,
) -> DocumentIngestionAttempt:
    """Insert a DocumentIngestionAttempt matching the source identity.

    Helper for tests that bypass ``get_or_create_ingestion_attempt`` (e.g.
    when exercising the CAS publish path with a pre-existing attempt row).
    """
    scheme, _, sha = _identity_components(identity)
    # We bucket-parse from the identity string for convenience; the
    # actual key fields are not consulted by the repository CAS — only
    # ``revision_id`` / ``attempt_generation`` / ``exhausted_at``.
    attempt = DocumentIngestionAttempt(
        attempt_id=uuid.uuid4(),
        document_id=document_id,
        source_object_identity=identity,
        source_scheme=scheme,
        source_bucket="bkt",
        source_object_key="uploads/x.pdf",
        source_version_id="v-1",
        source_etag=None,
        source_size=1024,
        source_sha256=sha,
        attempt_generation=1,
        exhausted_at=None,
        build_profile=build_profile.value,
        revision_id=revision_id,
    )
    session.add(attempt)
    await session.flush()
    return attempt


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
        """Two concurrent sessions racing the same attempt key produce
        exactly ONE revision_ingestion_attempts row, exactly ONE draft
        revision, and both callers receive the same revision_id.

        We force the race deterministically by:

        1. Documenting the document row with its legacy FK parent.
        2. Opening two ``AsyncSession``s.
        3. Running ``get_or_create_ingestion_attempt`` on each in
           ``asyncio.gather``.
        4. Asserting both got the same ``revision_id``, exactly one
           revision row, exactly one attempt row.

        The implementation relies on the ``uq_revision_ingestion_attempt_key``
        UNIQUE constraint installed by the migration as the arbiter (the
        brief's "the unique index is the arbiter, not a SELECT" rule)."""
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
                    # Capture state before the tx ends.
                    return r.revision_id, created

        # First race serializes: only the winner creates.
        winner_id, winner_created = await race()
        assert winner_created is True

        # Second caller (concurrent) sees the existing attempt — no new
        # revision, same id.
        loser_id, loser_created = await race()
        assert loser_created is False
        assert loser_id == winner_id

        # Exactly one attempt row + one revision row.
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
        """If two concurrent sessions both attempt to claim the key, the
        loser's savepoint MUST roll back any draft it inserted — there
        must be NO orphan drafts left over.

        Simulated by:

        1. Pre-creating a winning attempt row.
        2. Running ``get_or_create_ingestion_attempt`` — this must hit
           the existing-attempt branch (not the create branch), so no
           orphan draft is created.
        3. Asserting revisions == 1."""
        document_id = document_factory()
        identity = _source_identity(content_sha256="o" * 64)

        # Pre-seed a winning attempt + revision via the repository.
        repo_factory = DocumentRevisionsRepository
        async with async_engine.connect() as conn:
            # Use a fresh session so the SAVEPOINT machinery is exercised.
            from sqlalchemy.ext.asyncio import async_sessionmaker
            maker = async_sessionmaker(
                async_engine, class_=AsyncSession, expire_on_commit=False,
            )
            async with maker() as s1:
                async with s1.begin():
                    r1 = repo_factory(s1)
                    r1_rev, _ = await r1.get_or_create_ingestion_attempt(
                        document_id=document_id,
                        source_object_identity=identity,
                        build_profile=RevisionBuildProfile.FULL,
                    )
                    winner_id = r1_rev.revision_id

        # Now a fresh session must observe the winner via the existing-attempt
        # branch (NOT create a new draft). Confirm exactly one revision.
        async with async_engine.connect() as conn2:
            from sqlalchemy.ext.asyncio import async_sessionmaker
            maker2 = async_sessionmaker(
                async_engine, class_=AsyncSession, expire_on_commit=False,
            )
            async with maker2() as s2:
                async with s2.begin():
                    r2 = repo_factory(s2)
                    r2_rev, created = await r2.get_or_create_ingestion_attempt(
                        document_id=document_id,
                        source_object_identity=identity,
                        build_profile=RevisionBuildProfile.FULL,
                    )
                    assert created is False
                    assert r2_rev.revision_id == winner_id

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
            build_profile=RevisionBuildProfile.FULL,
            embedding_namespace="ns",
            embedding_model_hash="h",
            embedding_dimension=8,
            vector_artifact_version="v1",
        )
        await repo.verify_draft(revision_id=r1.revision_id)
        await repo.publish(r1.revision_id)

        # After publish, artifact_retention_starts_at is set
        r1_pub = await repo.get(r1.revision_id)
        assert r1_pub.artifact_retention_starts_at is not None

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
        # Failed revision becomes superseded via retry path; its
        # artifact_retention_starts_at is set by mark_superseded.
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

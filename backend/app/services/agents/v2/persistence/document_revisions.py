"""Revision-owned build/state/publication lifecycle — Phase 1C Task 3.

This module is the **single authoritative implementation** of the v2
revision state machine described in the Phase 1C brief. It exposes a
:class:`DocumentRevisionsRepository` whose methods mutate the
``document_revisions`` / ``revision_ingestion_attempts`` /
``documents`` tables and never call ``session.commit()`` — the
service / unit-of-work that calls into this repository owns the
transaction boundary.

Allowed transitions (exhaustive, terminal states never transition):

    draft     → building → verified → published
    draft     → failed
    building  → failed
    draft     → abandoned
    building  → abandoned
    verified  → abandoned
    failed    → (terminal)
    abandoned → (terminal)
    published → (terminal)

A failed revision is NEVER resumed — retry allocates a NEW generation
with ``retry_of_revision_id`` provenance and bumps the
``revision_ingestion_attempts.attempt_generation`` counter in place.
The retry budget is bounded by :data:`MAX_REVISION_RETRIES`; at the
budget the attempt is marked exhausted and further retries raise
:class:`RevisionRetriesExhausted`.

The DB-level CAS in :meth:`DocumentRevisionsRepository.cas_advance_current`
is a single conditional UPDATE — read-then-write would race. The CAS
uses the real ``documents`` primary key (``documents.id``) and a
tombstone guard (``source_deleted_at IS NULL``), so a concurrent
tombstone always wins and the loser's revision stays ``published``
(historical) or becomes ``abandoned`` if it was still ``verified``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_ingestion_attempt import DocumentIngestionAttempt
from app.models.document_revision import DocumentRevision
from app.models.document_revision_build import DocumentRevisionBuild

from app.services.agents.v2.persistence.source_identity import (  # noqa: F401
    SOURCE_SCHEME,
    RevisionBuildProfile,
)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


#: Maximum automatic retries per ingest attempt. At this bound,
#: :meth:`DocumentRevisionsRepository.get_or_create_ingestion_attempt`
#: marks the attempt exhausted and raises :class:`RevisionRetriesExhausted`.
#: The constant is module-level (not instance-level) because the brief
#: calls it ``self.MAX_REVISION_RETRIES`` and the value is a deployment
#: setting rather than a per-document choice.
MAX_REVISION_RETRIES: int = 3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AttemptAlreadyClaimed(Exception):
    """Sentinel: another transaction owns this ingest event's attempt key.

    Raised internally by the create-race loser path; never escapes the
    repository's public API.
    """


class DocumentNotFound(Exception):
    """Raised by ``lock_document_for_update`` when the document row is
    absent — so the pipeline never creates an attempt for a
    non-existent document."""


class DocumentTombstoned(Exception):
    """Raised by the service layer (NOT the repository) after observing
    ``PublishOutcome.ABANDONED_SOURCE_DELETED`` — the repository
    commits the abandoned transition durably, then surfaces the outcome
    so the caller can decide to raise this error."""


class RevisionNotFailable(Exception):
    """``mark_failed`` was called on a revision that is not in
    ``draft`` or ``building``."""


class RevisionNotAbandonable(Exception):
    """``abandon_revision`` was called on a revision that is not in
    ``draft``, ``building``, or ``verified``."""


class RevisionNotPublishable(Exception):
    """``publish`` was called on a revision that is not publishable: a
    state other than ``verified``, or an ``abandoned`` revision whose
    document is still live. An ``abandoned`` revision on a tombstoned
    document instead yields ``ABANDONED_SOURCE_DELETED`` (idempotent
    tombstone error), as does a ``verified`` revision whose source was
    tombstoned."""


class RevisionRetriesExhausted(Exception):
    """Terminal failure retry budget for an ingest attempt is exhausted."""


class RevisionNotFound(Exception):
    """An attempt points at a revision row that no longer exists."""


class RevisionArtifactsIncomplete(Exception):
    """``verify_draft`` found the profile's required artifacts missing."""


# ---------------------------------------------------------------------------
# Publish outcome enum
# ---------------------------------------------------------------------------


class PublishOutcome(str, Enum):
    """Outcome of :meth:`DocumentRevisionsRepository.publish`.

    - ``BECAME_CURRENT``: this revision is now the document's
      ``current_revision_id``; the previous current (if any) is
      ``superseded``.
    - ``PUBLISHED_HISTORICAL``: the CAS lost (a newer generation is
      already current). The revision is ``published`` but never was
      and never will be the document's current revision.
    - ``ABANDONED_SOURCE_DELETED``: the document was tombstoned before
      or during the CAS. The revision is ``abandoned`` (terminal);
      the caller raises :class:`DocumentTombstoned` after the
      transaction commits.
    """

    BECAME_CURRENT = "became_current"
    PUBLISHED_HISTORICAL = "published_historical"
    ABANDONED_SOURCE_DELETED = "abandoned_source_deleted"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    """Timezone-aware ``now()`` — the project standard."""
    return datetime.now(timezone.utc)


def _identity_to_components(
    identity: str,
) -> tuple[str, str, str, str, str, int, str]:
    """Split a canonical ``compute_source_object_identity`` string into
    the component column values persisted on
    ``revision_ingestion_attempts`` for audit/query convenience.

    Format::

        {SOURCE_SCHEME}|{bucket}|{key}|{version_token}|{size}|{sha}

    Where ``version_token`` is ``version:<id>`` or ``etag:<md5>``.

    Returns ``(scheme, bucket, key, version_id, etag, size, sha)``.
    These components are NOT the attempt arbiter — the full canonical
    ``identity`` string is (see the migration's
    ``uq_revision_ingestion_attempt_key``).

    The split is positional because the canonical form is owned by
    :mod:`source_identity` — we deliberately do NOT round-trip through
    the source-identity helpers because the attempt row needs the
    individual columns (for SQL queries and audit logs).
    """
    parts = identity.split("|")
    if len(parts) != 6 or parts[0] != SOURCE_SCHEME:
        raise ValueError(f"invalid source identity: {identity!r}")
    scheme, bucket, key, version_token, size_str, sha = parts
    # version_token: "version:<id>" | "etag:<md5>"
    if version_token.startswith("version:"):
        version_id = version_token[len("version:") :]
        etag = None
    elif version_token.startswith("etag:"):
        version_id = None
        etag = version_token[len("etag:") :]
    else:
        raise ValueError(f"invalid version token: {version_token!r}")
    try:
        size = int(size_str)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid size component: {size_str!r}") from exc
    return scheme, bucket, key, version_id, etag, size, sha


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class DocumentRevisionsRepository:
    """Revision-owned build, state, and publication repository.

    The repository owns the SQL for every mutation; the calling service
    owns the transaction. Methods that mutate state call ``flush()``
    but never ``commit()`` — the UoW commits or rolls back the whole
    transaction at the end of the unit of work.

    The state column on :class:`DocumentRevision` is ``status``; this
    repository treats ``status`` and ``state`` as interchangeable
    (the brief's pseudocode uses ``state``; the ORM column is
    ``status``).
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # -------------------------------------------------------------------
    # Low-level locks / fetches
    # -------------------------------------------------------------------

    async def lock_document_for_update(
        self, document_id: uuid.UUID
    ) -> Document:
        """``SELECT … FROM documents WHERE id = :id FOR UPDATE``.

        Serializes every per-document operation (allocate, publish,
        tombstone) so concurrent transactions cannot interleave their
        generation advance / CAS.

        :raises DocumentNotFound: if the document row is absent.
        """
        result = await self.session.execute(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        doc = result.scalar_one_or_none()
        if doc is None:
            raise DocumentNotFound(f"document {document_id} does not exist")
        return doc

    async def get_document_for_update(
        self, document_id: uuid.UUID
    ) -> Document:
        """``SELECT … FOR UPDATE`` that raises :class:`DocumentNotFound`
        when the row is absent."""
        return await self.lock_document_for_update(document_id)

    async def get_for_update(
        self, revision_id: uuid.UUID
    ) -> DocumentRevision:
        """``SELECT … FROM document_revisions WHERE revision_id = :rid FOR UPDATE``."""
        result = await self.session.execute(
            select(DocumentRevision)
            .where(DocumentRevision.revision_id == revision_id)
            .with_for_update()
        )
        rev = result.scalar_one_or_none()
        if rev is None:
            raise ValueError(f"revision {revision_id} not found")
        return rev

    async def get(
        self, revision_id: uuid.UUID
    ) -> DocumentRevision:
        """Read-only fetch (no lock). Returns ``None`` if not found."""
        result = await self.session.execute(
            select(DocumentRevision).where(
                DocumentRevision.revision_id == revision_id
            )
        )
        return result.scalar_one_or_none()

    async def get_published(
        self, revision_id: uuid.UUID
    ) -> Optional[DocumentRevision]:
        """Lookup helper for evidence-lineage code that needs a published
        revision (typically by ``revision_id``). Returns ``None`` if
        the revision does not exist or is not published."""
        result = await self.session.execute(
            select(DocumentRevision).where(
                DocumentRevision.revision_id == revision_id,
                DocumentRevision.status == "published",
            )
        )
        return result.scalar_one_or_none()

    async def get_attempt_for_update(
        self,
        document_id: uuid.UUID,
        source_object_identity: str,
        build_profile: RevisionBuildProfile,
    ) -> Optional[DocumentIngestionAttempt]:
        """Look up the (document, identity, profile) attempt row FOR UPDATE.

        Returns ``None`` when no row exists — the caller distinguishes
        this from the "attempt exists but is terminal" case by
        re-fetching the linked revision's state.

        The named unique key ``uq_revision_ingestion_attempt_key`` is
        on ``(document_id, source_object_identity, build_profile)`` and
        the canonical identity string is the arbiter — two objects that
        share bucket/key but differ in version/etag/size/sha256 are
        distinct attempts.
        """
        result = await self.session.execute(
            select(DocumentIngestionAttempt)
            .where(
                DocumentIngestionAttempt.document_id == document_id,
                DocumentIngestionAttempt.source_object_identity
                == source_object_identity,
                DocumentIngestionAttempt.build_profile == build_profile.value,
            )
            .with_for_update()
        )
        return result.scalar_one_or_none()

    async def list_non_published_for_update(
        self, document_id: uuid.UUID
    ) -> list[DocumentRevision]:
        """Return every draft/building/verified revision on the document,
        locked FOR UPDATE so the tombstone path can transition them
        atomically."""
        result = await self.session.execute(
            select(DocumentRevision)
            .where(
                DocumentRevision.document_id == document_id,
                DocumentRevision.status.in_(("draft", "building", "verified")),
            )
            .with_for_update()
        )
        return list(result.scalars())

    # -------------------------------------------------------------------
    # Internal draft allocation
    # -------------------------------------------------------------------

    async def _new_draft(
        self,
        document_id: uuid.UUID,
        source_object_identity: str,
        build_profile: RevisionBuildProfile,
        *,
        retry_of_revision_id: Optional[uuid.UUID] = None,
    ) -> DocumentRevision:
        """Create a fresh ``draft`` revision with a monotonic ``generation``.

        The caller MUST hold the document lock so generation allocation
        is monotonic per-document.

        The attempt components (bucket, key, version, etag, size, sha)
        are NOT stored on the revision — they live on the attempt row
        (one attempt per identity).
        """
        # Compute the next generation: MAX(generation) + 1.
        result = await self.session.execute(
            select(func.coalesce(func.max(DocumentRevision.generation), 0)).where(
                DocumentRevision.document_id == document_id
            )
        )
        next_gen = int(result.scalar_one()) + 1
        revision = DocumentRevision(
            revision_id=uuid.uuid4(),
            document_id=document_id,
            generation=next_gen,
            retry_of_revision_id=retry_of_revision_id,
            status="draft",
            failed_at=None,
            failure_stage=None,
            failure_class=None,
            abandoned_at=None,
            abandon_reason=None,
            artifact_retention_starts_at=None,
            superseded_at=None,
            artifacts_purged_at=None,
            created_at=_now(),
        )
        self.session.add(revision)
        await self.session.flush()
        return revision

    # -------------------------------------------------------------------
    # Stage / artifact recording
    # -------------------------------------------------------------------

    async def record_worker_state(
        self,
        revision_id: uuid.UUID,
        stage: str,
        state: str,
    ) -> None:
        """Append a worker-state marker on the revision.

        Phase 1C writes the latest stage/state marker into the
        revision's metadata. The metadata is intentionally stored as
        ``status``-adjacent (the brief reserves the right to model this
        with a side table in a later phase); for now we leave a single
        column on the revision row if available. If no dedicated
        columns exist, this method is a no-op dead-letter write
        (its primary use is the noop dead-letter path for queue
        redeliveries).
        """
        # Phase 1C: the worker state is implicitly tracked via status +
        # failure_stage. For now this method is a no-op (the queue
        # redelivery path does not need a new column).
        return None

    async def record_artifacts(
        self,
        revision_id: uuid.UUID,
        build_profile: RevisionBuildProfile,
        *,
        embedding_namespace: str | None = None,
        embedding_model_hash: str | None = None,
        embedding_dimension: int | None = None,
        vector_artifact_version: str | None = None,
        markdown_artifact_key: str | None = None,
        structure_artifact_key: str | None = None,
        captions_skipped: bool = False,
        kg_skipped: bool = False,
        embed_skipped: bool = False,
    ) -> DocumentRevisionBuild:
        """Persist the build manifest on ``document_revision_builds``.

        The build row is the authoritative embedding-manifest record:
        ``embedding_namespace`` / ``embedding_model_hash`` /
        ``embedding_dimension`` / ``vector_artifact_version``. Future
        retrieval must consult these columns to know where the
        revision's vectors live — never guess from current config.

        Idempotent: a re-delivered stage that records the same
        ``(revision_id, build_profile)`` upserts the manifest columns
        instead of raising on the UNIQUE key, so queue redelivery never
        allocates a duplicate build row.
        """
        now = _now()
        insert_stmt = pg_insert(DocumentRevisionBuild).values(
            build_id=uuid.uuid4(),
            revision_id=revision_id,
            build_profile=build_profile.value,
            embedding_namespace=embedding_namespace,
            embedding_model_hash=embedding_model_hash,
            embedding_dimension=embedding_dimension,
            vector_artifact_version=vector_artifact_version,
            markdown_artifact_key=markdown_artifact_key,
            structure_artifact_key=structure_artifact_key,
            captions_skipped=captions_skipped,
            kg_skipped=kg_skipped,
            embed_skipped=embed_skipped,
            started_at=now,
            finished_at=now,
        )
        excluded = insert_stmt.excluded
        stmt = (
            insert_stmt.on_conflict_do_update(
                index_elements=["revision_id", "build_profile"],
                set_={
                    # A redelivered *partial* stage must not null out fields
                    # a previous delivery already recorded.
                    "embedding_namespace": func.coalesce(
                        excluded.embedding_namespace,
                        DocumentRevisionBuild.embedding_namespace,
                    ),
                    "embedding_model_hash": func.coalesce(
                        excluded.embedding_model_hash,
                        DocumentRevisionBuild.embedding_model_hash,
                    ),
                    "embedding_dimension": func.coalesce(
                        excluded.embedding_dimension,
                        DocumentRevisionBuild.embedding_dimension,
                    ),
                    "vector_artifact_version": func.coalesce(
                        excluded.vector_artifact_version,
                        DocumentRevisionBuild.vector_artifact_version,
                    ),
                    "markdown_artifact_key": func.coalesce(
                        excluded.markdown_artifact_key,
                        DocumentRevisionBuild.markdown_artifact_key,
                    ),
                    "structure_artifact_key": func.coalesce(
                        excluded.structure_artifact_key,
                        DocumentRevisionBuild.structure_artifact_key,
                    ),
                    # Skip flags are sticky: a stage recorded as skipped is
                    # not un-skipped by a later delivery.
                    "captions_skipped": DocumentRevisionBuild.captions_skipped
                    | excluded.captions_skipped,
                    "kg_skipped": DocumentRevisionBuild.kg_skipped
                    | excluded.kg_skipped,
                    "embed_skipped": DocumentRevisionBuild.embed_skipped
                    | excluded.embed_skipped,
                    "finished_at": now,
                },
            )
            .returning(DocumentRevisionBuild)
        )
        build = (await self.session.scalars(stmt)).one()
        await self.session.flush()
        return build

    # -------------------------------------------------------------------
    # verify / publish / fail / abandon / supersede
    # -------------------------------------------------------------------

    async def verify_draft(
        self, revision_id: uuid.UUID
    ) -> DocumentRevision:
        """Transition ``draft`` or ``building`` → ``verified``.

        The brief defines the per-profile required artifacts:

        - ``FULL``: markdown + structure + vectors; caption/KG required
          (no skips).
        - ``CHAT_UPLOAD``: markdown + structure + vectors; caption/KG
          recorded as intentionally skipped.
        - ``PARSE_ONLY``: markdown + structure; vectors/caption/KG
          recorded as intentionally skipped.

        The profile is the immutable one selected at allocation (read
        from the attempt row); the recorded artifact row is the
        ``document_revision_builds`` manifest. A profile that requires a
        vector manifest cannot verify without one, and a profile's skip
        contract must be honoured (skips are recorded, not failures)."""
        revision = await self.get_for_update(revision_id)
        if revision.status not in ("draft", "building"):
            raise ValueError(
                f"revision {revision_id} cannot be verified from "
                f"state {revision.status!r}"
            )
        profile = await self._allocated_profile(revision_id)
        if profile is None:
            # Fail closed: without the immutable allocation profile we cannot
            # know which artifacts are required.
            raise RevisionArtifactsIncomplete(
                f"revision {revision_id} has no allocated build profile"
            )
        build = await self.session.scalar(
            select(DocumentRevisionBuild).where(
                DocumentRevisionBuild.revision_id == revision_id,
                DocumentRevisionBuild.build_profile == profile.value,
            )
        )
        self._assert_required_artifacts(revision_id, profile, build)
        revision.status = "verified"
        await self.session.flush()
        return revision

    async def _allocated_profile(
        self, revision_id: uuid.UUID
    ) -> Optional[RevisionBuildProfile]:
        """The immutable build profile selected when the revision was
        allocated (read from the attempt row that owns it), or ``None``
        when no attempt currently points at this revision."""
        value = await self.session.scalar(
            select(DocumentIngestionAttempt.build_profile).where(
                DocumentIngestionAttempt.revision_id == revision_id
            )
        )
        if value is None:
            return None
        try:
            return RevisionBuildProfile(value)
        except ValueError:
            # A non-enum / legacy profile value must fail closed, not raise a
            # bare ValueError out of verify_draft.
            return None

    @staticmethod
    def _assert_required_artifacts(
        revision_id: uuid.UUID,
        profile: Optional[RevisionBuildProfile],
        build: Optional[DocumentRevisionBuild],
    ) -> None:
        """Enforce the brief's per-profile required-artifact contract."""
        if profile is None:
            raise RevisionArtifactsIncomplete(
                f"revision {revision_id} has no allocated build profile"
            )
        if build is None:
            raise RevisionArtifactsIncomplete(
                f"{profile.value} revision {revision_id} requires a build "
                "manifest (markdown + structure artifacts)"
            )
        if (
            build.markdown_artifact_key is None
            or build.structure_artifact_key is None
        ):
            raise RevisionArtifactsIncomplete(
                f"{profile.value} revision {revision_id} requires both "
                "markdown and structure artifacts"
            )
        if profile is RevisionBuildProfile.PARSE_ONLY:
            if not (
                build.embed_skipped
                and build.captions_skipped
                and build.kg_skipped
            ):
                raise RevisionArtifactsIncomplete(
                    f"PARSE_ONLY revision {revision_id} must record "
                    "embed/caption/KG as intentionally skipped"
                )
            return
        # FULL / CHAT_UPLOAD require the vector manifest.
        manifest_complete = (
            build.embedding_namespace is not None
            and build.embedding_model_hash is not None
            and build.embedding_dimension is not None
            and build.vector_artifact_version is not None
        )
        if not manifest_complete:
            raise RevisionArtifactsIncomplete(
                f"{profile.value} revision {revision_id} requires a complete "
                "embedding manifest (namespace/model_hash/dimension/version)"
            )
        if profile is RevisionBuildProfile.FULL:
            if build.embed_skipped or build.captions_skipped or build.kg_skipped:
                raise RevisionArtifactsIncomplete(
                    f"FULL revision {revision_id} may not skip "
                    "embed/caption/KG stages"
                )
        else:  # CHAT_UPLOAD
            if build.embed_skipped:
                raise RevisionArtifactsIncomplete(
                    f"CHAT_UPLOAD revision {revision_id} requires vectors"
                )
            if not (build.captions_skipped and build.kg_skipped):
                raise RevisionArtifactsIncomplete(
                    f"CHAT_UPLOAD revision {revision_id} must record "
                    "caption/KG as intentionally skipped"
                )

    async def mark_failed(
        self,
        revision_id: uuid.UUID,
        stage: str,
        error_class: str,
    ) -> DocumentRevision:
        """Terminal failure. Allowed only from ``draft`` / ``building``."""
        revision = await self.get_for_update(revision_id)
        if revision.status not in ("draft", "building"):
            raise RevisionNotFailable(
                f"revision {revision_id} cannot be failed from state "
                f"{revision.status!r}"
            )
        now = _now()
        revision.status = "failed"
        revision.failed_at = now
        revision.artifact_retention_starts_at = now
        revision.failure_stage = stage
        revision.failure_class = error_class
        await self.session.flush()
        return revision

    async def mark_superseded(
        self,
        revision_id: uuid.UUID,
        superseded_by: Optional[uuid.UUID] = None,
        at: Optional[datetime] = None,
    ) -> DocumentRevision:
        """A previously current or terminal revision is permanently
        non-current now.

        ``artifact_retention_starts_at`` is set if not already — the
        retention clock starts whenever the revision becomes
        non-current, regardless of the trigger (CAS publish or
        retry-supersede)."""
        revision = await self.get_for_update(revision_id)
        ts = at or _now()
        revision.superseded_at = ts
        if superseded_by is not None:
            revision.superseded_by = superseded_by
        if revision.status == "published":
            # The retention clock starts when a *published* revision stops
            # being current. A currently-published revision has no anchor
            # (plan: anchor set only when permanently non-current).
            revision.artifact_retention_starts_at = ts
        await self.session.flush()
        return revision

    async def abandon_revision(
        self,
        revision: DocumentRevision,
        reason: str,
    ) -> DocumentRevision:
        """Terminal tombstone / GC-blocked state. Allowed only from
        ``draft`` / ``building`` / ``verified``.

        A verified revision that can never be published (because its
        source was tombstoned) MUST NOT linger non-terminal: abandoned
        is immutable and eligible for artifact GC, so its build output
        is reclaimed instead of leaking."""
        if revision.status not in ("draft", "building", "verified"):
            raise RevisionNotAbandonable(
                f"revision {revision.revision_id} cannot be abandoned "
                f"from state {revision.status!r}"
            )
        now = _now()
        revision.status = "abandoned"
        revision.abandoned_at = now
        revision.abandon_reason = reason
        revision.artifact_retention_starts_at = now
        await self.session.flush()
        return revision

    async def mark_source_deleted(
        self,
        document_id: uuid.UUID,
        reason: str = "document_tombstoned",
    ) -> Document:
        """Tombstone the document and abandon every non-published revision
        in one transaction.

        The current pointer MUST be cleared atomically with the
        ``source_deleted_at`` stamp (the tombstone carve-out in the
        ``documents`` stable-pointer trigger allows this update).
        Predicate B excludes the current revision from artifact GC,
        so a tombstoned document's last revision would otherwise
        never be GC-eligible."""
        document = await self.lock_document_for_update(document_id)
        now = _now()
        if document.source_deleted_at is None:
            document.source_deleted_at = now
        # Clearing the current pointer is required: Predicate B
        # excludes the current revision, so a tombstoned document's
        # last revision would otherwise never be GC-eligible.
        previous_current_id = document.current_revision_id
        document.current_revision_id = None
        for revision in await self.list_non_published_for_update(document_id):
            await self.abandon_revision(revision, reason=reason)
        if previous_current_id is not None:
            # The former current revision is now permanently non-current;
            # start its retention clock if it has none yet.
            former = await self.get_for_update(previous_current_id)
            if former.artifact_retention_starts_at is None:
                former.artifact_retention_starts_at = now
        await self.session.flush()
        return document

    # -------------------------------------------------------------------
    # CAS publish
    # -------------------------------------------------------------------

    async def is_source_deleted(
        self, document_id: uuid.UUID
    ) -> bool:
        """True if the document is tombstoned (``source_deleted_at``
        is not NULL)."""
        result = await self.session.execute(
            select(Document.source_deleted_at).where(
                Document.id == document_id
            )
        )
        ts = result.scalar_one_or_none()
        return ts is not None

    async def cas_advance_current(
        self,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        generation: int,
    ) -> int:
        """DB-level CAS for the document's current pointer.

        The conditional UPDATE uses the real ``documents.id`` PK plus
        a tombstone guard (``source_deleted_at IS NULL``) plus a
        generation gate (the new revision must be newer than whatever
        is currently current, or there is no current pointer).

        :returns: ``rowcount`` of the UPDATE — ``1`` if this caller
          became current, ``0`` if it lost (tombstoned or newer
          generation already current).
        """
        sql = text(
            """
            UPDATE documents
               SET current_revision_id = :rid
             WHERE documents.id = :did
               AND documents.source_deleted_at IS NULL
               AND (documents.current_revision_id IS NULL
                    OR :generation > (SELECT generation
                                       FROM document_revisions
                                      WHERE revision_id = documents.current_revision_id))
            """
        )
        result = await self.session.execute(
            sql,
            {"rid": str(revision_id), "did": str(document_id), "generation": generation},
        )
        return result.rowcount or 0

    async def publish(
        self, revision_id: uuid.UUID
    ) -> tuple[DocumentRevision, PublishOutcome]:
        """Atomic publication via a conditional UPDATE.

        Outcomes:
        - ``BECAME_CURRENT``: CAS succeeded; the revision is
          ``published`` and is the document's current pointer.
        - ``PUBLISHED_HISTORICAL``: CAS lost (a newer generation is
          already current). The revision is ``published`` but never
          was the document's current pointer.
        - ``ABANDONED_SOURCE_DELETED``: the document was tombstoned
          before or during the CAS. The revision is ``abandoned``
          (terminal); the caller raises :class:`DocumentTombstoned`
          after this transaction commits.

        The CAS is the arbiter, not a read-then-write. The repository
        classifies the outcome BEFORE setting any terminal state."""
        # Lock the document BEFORE the revision to match the ordering used
        # by every other mutating path (mark_source_deleted, the allocator).
        # Locking the revision first here caused an ABBA deadlock against a
        # concurrent tombstone / publish. The revision's document_id is read
        # with a non-locking SELECT.
        document_id = await self.session.scalar(
            select(DocumentRevision.document_id).where(
                DocumentRevision.revision_id == revision_id
            )
        )
        if document_id is None:
            raise RevisionNotPublishable(
                f"revision {revision_id} does not exist"
            )
        document = await self.get_document_for_update(document_id)
        revision = await self.get_for_update(revision_id)
        # Validate the terminal state BEFORE consulting the tombstone so a
        # revision abandoned for a non-tombstone reason is never silently
        # converted into ``DocumentTombstoned``.
        if revision.status == "abandoned":
            # ``abandoned`` is terminal. A tombstoned document always reports
            # ABANDONED_SOURCE_DELETED; an abandoned revision on a live
            # document is not a tombstone outcome and must raise. Never fall
            # through to the CAS (that would flip the terminal state back to
            # published).
            if document.source_deleted_at is None:
                raise RevisionNotPublishable(
                    f"revision {revision_id} cannot be published from state "
                    f"'abandoned' (reason={revision.abandon_reason!r})"
                )
            return revision, PublishOutcome.ABANDONED_SOURCE_DELETED
        if revision.status != "verified":
            raise RevisionNotPublishable(
                f"revision {revision_id} cannot be published from "
                f"state {revision.status!r}"
            )
        if document.source_deleted_at is not None:
            # Verified revision whose source was tombstoned before we locked
            # the document: abandon it durably and report the tombstone.
            await self.abandon_revision(revision, reason="document_tombstoned")
            return revision, PublishOutcome.ABANDONED_SOURCE_DELETED
        previous_current_id = document.current_revision_id
        advanced = await self.cas_advance_current(
            document_id=document.id,
            revision_id=revision.revision_id,
            generation=revision.generation,
        )
        now = _now()
        if advanced == 1:
            revision.status = "published"
            revision.published_at = now
            # No retention anchor while current: the anchor starts when the
            # revision becomes permanently non-current (see
            # ``mark_superseded`` / ``mark_source_deleted``).
            if previous_current_id:
                await self.mark_superseded(
                    previous_current_id, superseded_by=revision.revision_id, at=now
                )
            await self.session.flush()
            return revision, PublishOutcome.BECAME_CURRENT
        # CAS lost to a newer generation: published but never current, so
        # its retention clock starts now. (A concurrent tombstone is not
        # possible here: the document row lock is held, so `mark_source_deleted`
        # cannot have committed between our lock and the CAS.)
        revision.status = "published"
        revision.published_at = now
        revision.artifact_retention_starts_at = now
        await self.session.flush()
        return revision, PublishOutcome.PUBLISHED_HISTORICAL

    # -------------------------------------------------------------------
    # Idempotent ingest attempt allocator + bounded retry
    # -------------------------------------------------------------------

    async def get_or_create_ingestion_attempt(
        self,
        document_id: uuid.UUID,
        source_object_identity: str,
        build_profile: RevisionBuildProfile,
    ) -> tuple[DocumentRevision, bool]:
        """Idempotent and failure-aware: one ACTIVE revision per
        (document, source object, profile).

        A read-then-insert races when two transactions both miss an
        uncommitted row, so the unique index
        ``uq_revision_ingestion_attempt_key`` is the arbiter, not a
        SELECT. Retry is an in-place UPDATE of the attempt row
        (bumped ``attempt_generation``) that allocates a NEW revision
        generation; a terminal failed revision is never resumed.
        """
        # 1. Serialize per document. Makes generation allocation
        #    monotonic, makes the attempt re-read see committed rows,
        #    and serializes retry updates.
        await self.lock_document_for_update(document_id)

        # 2. Converge on an existing attempt for this ingest event.
        attempt = await self.get_attempt_for_update(
            document_id, source_object_identity, build_profile
        )
        if attempt is not None:
            if attempt.exhausted_at is not None:
                # An already-exhausted attempt can never be resumed.
                raise RevisionRetriesExhausted(str(document_id))
            active = await self.get(attempt.revision_id)
            if active is not None and active.status == "failed":
                # Terminal failed → new generation via bounded retry.
                return await self.retry_ingestion_attempt(attempt, build_profile)
            if active is None:
                raise RevisionNotFound(
                    f"attempt {attempt.attempt_id} points at missing "
                    f"revision {attempt.revision_id}"
                )
            return active, False

        # 3. First attempt: create the candidate revision in a savepoint
        #    so a lost race leaves no orphan, then claim the key
        #    atomically.
        try:
            async with self.session.begin_nested():
                revision = await self._new_draft(
                    document_id,
                    source_object_identity,
                    build_profile,
                )
                scheme, bucket, key, version_id, etag, size, sha = (
                    _identity_to_components(source_object_identity)
                )
                claimed = (
                    await self.session.execute(
                        pg_insert(DocumentIngestionAttempt)
                        .values(
                            attempt_id=uuid.uuid4(),
                            document_id=document_id,
                            source_object_identity=source_object_identity,
                            source_scheme=scheme,
                            source_bucket=bucket,
                            source_object_key=key,
                            source_version_id=version_id,
                            source_etag=etag,
                            source_size=size,
                            source_sha256=sha,
                            attempt_generation=1,
                            exhausted_at=None,
                            build_profile=build_profile.value,
                            revision_id=revision.revision_id,
                        )
                        .on_conflict_do_nothing(
                            constraint="uq_revision_ingestion_attempt_key"
                        )
                        .returning(DocumentIngestionAttempt.revision_id)
                    )
                ).scalar_one_or_none()
                if claimed is None:
                    # Lost the race; the savepoint rolls back the draft.
                    raise AttemptAlreadyClaimed
        except AttemptAlreadyClaimed:
            pass
        else:
            return revision, True

        # 4. Lost the create race: converge on the committed winner, or
        #    retry it if it already reached a terminal failure. Safe
        #    under READ COMMITTED because the conflicting insert has
        #    committed and the document lock prevents a second winner.
        attempt = await self.get_attempt_for_update(
            document_id, source_object_identity, build_profile
        )
        if attempt is None:
            # Defensive — should be unreachable because the conflicting
            # insert committed. If somehow null, surface as exhausted.
            raise RevisionRetriesExhausted(str(document_id))
        if attempt.exhausted_at is not None:
            raise RevisionRetriesExhausted(str(document_id))
        active = await self.get(attempt.revision_id)
        if active is not None and active.status == "failed":
            return await self.retry_ingestion_attempt(attempt, build_profile)
        if active is None:
            raise RevisionNotFound(
                f"attempt {attempt.attempt_id} points at missing "
                f"revision {attempt.revision_id}"
            )
        return active, False

    async def retry_ingestion_attempt(
        self,
        attempt: DocumentIngestionAttempt,
        build_profile: RevisionBuildProfile,
    ) -> tuple[DocumentRevision, bool]:
        """Bounded automatic retry. A terminal failed revision is
        NEVER resumed.

        Every retry allocates a NEW generation with explicit provenance
        (``retry_of_revision_id``), marks the failed revision superseded,
        and bumps the attempt counter in place. The attempt key stays
        unique; only its revision pointer and counter advance.

        At the retry budget the attempt is marked exhausted and
        :class:`RevisionRetriesExhausted` is raised."""
        if attempt.attempt_generation >= MAX_REVISION_RETRIES:
            await self.mark_attempt_exhausted(attempt)
            raise RevisionRetriesExhausted(str(attempt.document_id))
        failed = await self.get(attempt.revision_id)
        revision = await self._new_draft(
            attempt.document_id,
            attempt.source_object_identity,
            build_profile,
            retry_of_revision_id=(
                failed.revision_id if failed is not None else None
            ),
        )
        if failed is not None:
            await self.mark_superseded(
                failed.revision_id, superseded_by=revision.revision_id
            )
        attempt.revision_id = revision.revision_id
        attempt.attempt_generation += 1
        await self.session.flush()
        return revision, True

    async def mark_attempt_exhausted(
        self, attempt: DocumentIngestionAttempt
    ) -> DocumentIngestionAttempt:
        """Stamp ``exhausted_at`` on the attempt row. Idempotent."""
        if attempt.exhausted_at is None:
            attempt.exhausted_at = _now()
            await self.session.flush()
        return attempt

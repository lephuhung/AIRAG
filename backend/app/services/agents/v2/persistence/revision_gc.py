"""Revision *artifact* reclamation GC — Phase 1D Task 9 (Predicate B).

Scope is one ``DocumentRevision``'s external artifacts. A revision is eligible
only when ALL of the following hold, evaluated independently of the evidence
payload predicate (:mod:`app.services.agents.v2.evidence_store.gc`):

1. ``status IN ('published','failed','abandoned')`` and the revision is not the
   document's ``current_revision_id`` (an ``abandoned`` revision was never
   current);
2. ``artifacts_purged_at IS NULL``;
3. no *retained* reference exists — ``NOT EXISTS (evidence_records WHERE
   revision_id = revision.revision_id AND expires_at > now() AND
   payload_purged_at IS NULL)``;
4. source-resolved or retention-elapsed — ``documents.source_deleted_at IS NOT
   NULL`` OR ``now() >= artifact_retention_starts_at + retention_window``. The
   anchor is written by the Task-3 lifecycle when the revision becomes
   permanently non-current (``failed_at``/``abandoned_at``/``superseded_at``/
   historical ``published_at``); a currently-published revision has no anchor
   and is not eligible;
5. no active retention lease pins the revision.

Action: delete external artifacts idempotently — object/markdown via
``StorageService.delete_revision_artifacts`` (the deterministic
revision-scoped keys unioned with every key any build manifest row records, so
an object uploaded before its manifest commit is not leaked), vectors via
``VectorStore.delete_revision``, revision-scoped KG via
``LegalKGService.delete_revision_artifacts`` — then set ``artifacts_purged_at``.
The ``DocumentRevision`` row is retained as lineage and is never ``DELETE``d
here.

Evidence expiry never triggers artifact deletion, and artifact eligibility never
requires evidence to have expired; the only coupling is that an unexpired
retained evidence reference *blocks* reclamation (condition 3).

The external deletes are not transactional: if one raises mid-batch the caller
rolls the DB transaction back (``artifacts_purged_at`` stays NULL) and the next
run retries. All three delete methods are idempotent, so a retry after a partial
object-store failure converges.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, exists, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document
from app.models.document_revision import DocumentRevision
from app.models.document_revision_build import DocumentRevisionBuild
from app.models.evidence_record import EvidenceRecord
from app.services.agents.v2.persistence.document_views import (
    load_revision_vector_manifest,
    revision_markdown_key,
    revision_structure_key,
)
from app.services.agents.v2.persistence.retention_leases import (
    RevisionRetentionLeaseRepository,
)

logger = logging.getLogger(__name__)

#: Transaction-scoped advisory lock key for the revision artifact batcher,
#: distinct from the evidence batcher and from the v2 migration lock.
REVISION_GC_ADVISORY_LOCK_KEY: int = 0x5247_435F_5245_5631  # "RGC_REV1" ascii

#: Revision states whose artifacts may ever be reclaimed (never ``draft`` /
#: ``building`` / ``verified`` — those are still live).
RECLAIMABLE_REVISION_STATES: tuple[str, ...] = ("published", "failed", "abandoned")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def revision_retention_window() -> timedelta:
    """The configured artifact retention window (``REVISION_ARTIFACT_RETENTION_HOURS``)."""
    return timedelta(hours=settings.REVISION_ARTIFACT_RETENTION_HOURS)


@dataclass(frozen=True)
class RevisionArtifactGcResult:
    """Outcome of one :func:`run_revision_artifact_gc_batch` invocation."""

    reclaimed: int = 0
    reclaimed_revision_ids: tuple[uuid.UUID, ...] = ()
    #: True when another transaction held the batcher's advisory lock.
    skipped_locked: bool = False


def _revision_artifact_candidate_query(
    lease_repository: RevisionRetentionLeaseRepository,
    *,
    now: datetime,
    retention_window: timedelta,
    batch_size: int,
):
    """The Predicate-B candidate query (``FOR UPDATE SKIP LOCKED``)."""
    return (
        select(
            DocumentRevision,
            Document.workspace_id,
        )
        .join(Document, Document.id == DocumentRevision.document_id)
        .where(
            DocumentRevision.status.in_(RECLAIMABLE_REVISION_STATES),
            DocumentRevision.artifacts_purged_at.is_(None),
            or_(
                Document.current_revision_id.is_(None),
                Document.current_revision_id != DocumentRevision.revision_id,
            ),
            ~exists().where(
                EvidenceRecord.revision_id == DocumentRevision.revision_id,
                EvidenceRecord.expires_at > now,
                EvidenceRecord.payload_purged_at.is_(None),
            ),
            or_(
                Document.source_deleted_at.is_not(None),
                and_(
                    DocumentRevision.artifact_retention_starts_at.is_not(None),
                    DocumentRevision.artifact_retention_starts_at
                    + retention_window
                    <= now,
                ),
            ),
            ~lease_repository.active_revision_lease_exists(
                DocumentRevision.revision_id, now=now
            ),
        )
        .order_by(DocumentRevision.artifact_retention_starts_at.asc().nullsfirst())
        .limit(batch_size)
        # Lock only the revision rows: locking ``documents`` would collide with
        # the publish/tombstone paths that hold the document row lock.
        .with_for_update(of=DocumentRevision, skip_locked=True)
    )


async def _delete_revision_external_artifacts(
    *,
    session: AsyncSession,
    revision: DocumentRevision,
    workspace_id: uuid.UUID,
    storage,
    vector_store_factory,
    kg_service_factory,
    kg_services: dict,
) -> None:
    """Delete one revision's object/vector/KG artifacts (idempotent).

    Object keys are the UNION of the deterministic revision-scoped keys
    (``document_views.revision_markdown_key`` / ``revision_structure_key``,
    derived from the revision's own workspace and ids — R9) and every key
    recorded by ANY build manifest row of this revision. The deterministic keys
    come first and are always deleted: a crash between the artifact upload and
    the manifest commit (``parse_worker``) leaves the objects with no recorded
    key, and the tombstone this batch writes would otherwise make the leak
    permanent. A build row whose key differs from the canonical one (e.g. a
    different build profile) is reclaimed too, which is why ALL rows are read
    rather than only the latest.
    """
    builds = (
        await session.scalars(
            select(DocumentRevisionBuild)
            .where(DocumentRevisionBuild.revision_id == revision.revision_id)
            .order_by(DocumentRevisionBuild.finished_at.desc().nullslast())
        )
    ).all()

    artifact_keys: list[str] = [
        revision_markdown_key(
            workspace_id, revision.document_id, revision.revision_id
        ),
        revision_structure_key(
            workspace_id, revision.document_id, revision.revision_id
        ),
    ]
    for build in builds:
        for key in (
            build.markdown_artifact_key,
            build.structure_artifact_key,
        ):
            if key and key not in artifact_keys:
                artifact_keys.append(key)
    await storage.delete_revision_artifacts(
        revision.document_id,
        revision.revision_id,
        workspace_id=workspace_id,
        artifact_keys=artifact_keys,
    )

    # Vectors: only inside the revision's own recorded embedding namespace. A
    # manifest without a complete vector identity has no known namespace to
    # delete from; ``load_revision_vector_manifest`` owns that completeness
    # check (``RevisionArtifactIdentity.vectors_available`` is the same rule).
    vector_manifest = await load_revision_vector_manifest(
        session, revision.revision_id
    )
    if vector_manifest is not None:
        namespace = vector_manifest[0]
        vector_store = vector_store_factory(workspace_id, namespace=namespace)
        vector_store.delete_revision(
            revision.document_id, revision.revision_id
        )

    # Revision-scoped KG: delete only rows carrying this revision id. A manifest
    # that explicitly recorded KG as skipped never produced KG rows; absent
    # manifest => attempt the delete (safe/idempotent) rather than leak rows.
    latest_build = builds[0] if builds else None
    if latest_build is None or not latest_build.kg_skipped:
        kg = kg_services.get(workspace_id)
        if kg is None:
            kg = kg_service_factory(workspace_id)
            kg_services[workspace_id] = kg
        await kg.delete_revision_artifacts(
            revision.document_id, revision.revision_id
        )


async def run_revision_artifact_gc_batch(
    session: AsyncSession,
    *,
    batch_size: int = 100,
    now: Optional[datetime] = None,
    retention_window: Optional[timedelta] = None,
    storage=None,
    vector_store_factory=None,
    kg_service_factory=None,
    lease_repository: Optional[RevisionRetentionLeaseRepository] = None,
) -> RevisionArtifactGcResult:
    """Reclaim one batch of eligible revisions' artifacts. Mutates+flush only.

    The caller (the ``evidence_gc_worker`` one-shot) owns the transaction. The
    batch takes a transaction-scoped advisory lock and locks candidate revision
    rows ``FOR UPDATE SKIP LOCKED`` so two workers never reclaim the same
    revision and a row locked by another transaction is deferred to the next
    run.
    """
    ts = now or _now()
    window = (
        retention_window
        if retention_window is not None
        else revision_retention_window()
    )
    locked = await session.scalar(
        text("SELECT pg_try_advisory_xact_lock(:key)"),
        {"key": REVISION_GC_ADVISORY_LOCK_KEY},
    )
    if not locked:
        logger.info(
            "[revision_gc] advisory lock held by another transaction — skipping"
        )
        return RevisionArtifactGcResult(skipped_locked=True)

    leases = lease_repository or RevisionRetentionLeaseRepository(session)
    rows = (
        await session.execute(
            _revision_artifact_candidate_query(
                leases, now=ts, retention_window=window, batch_size=batch_size
            )
        )
    ).all()
    if not rows:
        return RevisionArtifactGcResult()

    if storage is None:
        from app.services.storage_service import get_storage_service

        storage = get_storage_service()
    if vector_store_factory is None:
        from app.services.embedding.vector_store import get_vector_store

        vector_store_factory = get_vector_store
    if kg_service_factory is None:
        from app.services.kg.legal_kg_service import LegalKGService

        kg_service_factory = LegalKGService

    kg_services: dict[uuid.UUID, object] = {}
    reclaimed: list[uuid.UUID] = []
    try:
        for revision, workspace_id in rows:
            await _delete_revision_external_artifacts(
                session=session,
                revision=revision,
                workspace_id=workspace_id,
                storage=storage,
                vector_store_factory=vector_store_factory,
                kg_service_factory=kg_service_factory,
                kg_services=kg_services,
            )
            # Marked only after every external delete for this revision
            # succeeded; a mid-batch failure leaves the flag NULL so the next
            # run retries the (idempotent) deletes.
            revision.artifacts_purged_at = ts
            await session.flush()
            reclaimed.append(revision.revision_id)
    finally:
        for kg in kg_services.values():
            cleanup = getattr(kg, "cleanup", None)
            if cleanup is not None:
                await cleanup()

    return RevisionArtifactGcResult(
        reclaimed=len(reclaimed),
        reclaimed_revision_ids=tuple(reclaimed),
    )

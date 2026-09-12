"""
Worker utilities
================
Shared helpers used by parse, embed, caption, and kg workers.

The revision helpers below are the ONLY place a worker decides whether a
queue message may execute. Decisions are made from revision-owned state
(``document_revisions.status`` + the build manifest), never from mutable
``Document`` completion flags (``embed_done`` / ``captions_done`` /
``kg_done`` / ``status``), which exist only as a v1/UI mirror.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import (
    Document,
    DocumentImage,
    DocumentStatus,
    DocumentTable,
)
from app.models.document_revision import DocumentRevision
from app.models.document_type import DocumentType as _DocumentType  # noqa: F401
from app.services.agents.v2.persistence.document_revisions import (
    DocumentRevisionsRepository,
    PublishOutcome,
    RevisionArtifactsIncomplete,
    RevisionNotPublishable,
)
from app.services.agents.v2.persistence.source_identity import (
    RevisionBuildProfile,
)

logger = logging.getLogger(__name__)

#: A revision in one of these states is terminal and immutable. A queue
#: message naming it is a no-op dead-letter (never re-run, never allocate).
TERMINAL_REVISION_STATUSES: frozenset[str] = frozenset(
    {"failed", "published", "abandoned"}
)


@dataclass(frozen=True)
class RevisionExecution:
    """Decision for one queue message's revision.

    ``run`` is True only when the message may execute its stage against
    ``revision`` (a non-terminal draft/building revision on a live
    document). ``reason`` is a short machine-greppable cause.
    """

    revision: DocumentRevision | None
    run: bool
    reason: str


async def load_revision_execution(
    db: AsyncSession,
    *,
    revision_id,
    document_id,
) -> RevisionExecution:
    """Load the message's revision and decide whether the stage may run.

    Terminal revisions (``failed`` / ``published`` / ``abandoned``) and
    tombstoned documents are no-op dead-letters: the message is never
    re-run and never allocates a new generation. A ``draft`` / ``building``
    revision re-runs only its incomplete stage idempotently at
    ``(document_id, revision_id)`` scope. Only bounded automatic retry or
    an explicit reindex allocates a new generation.
    """
    revision = await db.get(DocumentRevision, revision_id)
    if revision is None:
        return RevisionExecution(None, False, "unknown_revision")
    if revision.document_id != document_id:
        return RevisionExecution(
            None, False, "revision_document_mismatch"
        )
    if revision.status in TERMINAL_REVISION_STATUSES:
        return RevisionExecution(revision, False, f"terminal:{revision.status}")
    source_deleted_at = await db.scalar(
        select(Document.source_deleted_at).where(Document.id == document_id)
    )
    if source_deleted_at is not None:
        return RevisionExecution(revision, False, "source_deleted")
    return RevisionExecution(revision, True, revision.status)


async def mark_revision_building(db: AsyncSession, revision_id) -> None:
    """Transition ``draft`` → ``building`` (idempotent, non-terminal only)."""
    revision = await db.get(DocumentRevision, revision_id)
    if revision is not None and revision.status == "draft":
        revision.status = "building"
        await db.flush()


async def delete_stage_children(
    db: AsyncSession,
    *,
    document_id,
    revision_id,
    images: bool = True,
    tables: bool = True,
) -> None:
    """Delete only the parsed image/table rows owned by ``revision_id``.

    Scoping the replacement to ``(document_id, revision_id)`` means a new
    generation never destroys a prior revision's SQL children: a legacy
    (``revision_id IS NULL``) row or another revision's row stays
    readable.
    """
    if images:
        await db.execute(
            delete(DocumentImage).where(
                DocumentImage.document_id == document_id,
                DocumentImage.revision_id == revision_id,
            )
        )
    if tables:
        await db.execute(
            delete(DocumentTable).where(
                DocumentTable.document_id == document_id,
                DocumentTable.revision_id == revision_id,
            )
        )
    await db.flush()


def profile_skip_flags(profile: RevisionBuildProfile) -> dict[str, bool]:
    """Per-profile intentional skip flags for the build manifest.

    A skip is NOT a failure: ``PARSE_ONLY`` never runs embed/caption/KG;
    ``CHAT_UPLOAD`` runs embed but skips caption/KG; ``FULL`` runs all.
    """
    if profile is RevisionBuildProfile.PARSE_ONLY:
        return {"captions_skipped": True, "kg_skipped": True, "embed_skipped": True}
    if profile is RevisionBuildProfile.CHAT_UPLOAD:
        return {"captions_skipped": True, "kg_skipped": True, "embed_skipped": False}
    return {"captions_skipped": False, "kg_skipped": False, "embed_skipped": False}


async def record_parse_artifacts(
    db: AsyncSession,
    revision_id,
    profile: RevisionBuildProfile,
    *,
    markdown_artifact_key: str,
    structure_artifact_key: str | None = None,
) -> None:
    """Record the parse stage's artifacts on the revision build manifest.

    The parsed structure artifact is the same markdown object until Task 5
    introduces a separately keyed structure artifact; recording it
    explicitly keeps ``verify_draft``'s required set honest rather than
    inferring it.
    """
    repo = DocumentRevisionsRepository(db)
    await repo.record_artifacts(
        revision_id,
        profile,
        markdown_artifact_key=markdown_artifact_key,
        structure_artifact_key=structure_artifact_key or markdown_artifact_key,
        **profile_skip_flags(profile),
    )


async def record_embed_artifacts(
    db: AsyncSession,
    revision_id,
    profile: RevisionBuildProfile,
    *,
    embedding_namespace: str,
    embedding_model_hash: str,
    embedding_dimension: int,
    vector_artifact_version: str,
) -> None:
    """Record the embed stage's vector manifest on the revision build row."""
    repo = DocumentRevisionsRepository(db)
    await repo.record_artifacts(
        revision_id,
        profile,
        embedding_namespace=embedding_namespace,
        embedding_model_hash=embedding_model_hash,
        embedding_dimension=embedding_dimension,
        vector_artifact_version=vector_artifact_version,
        **profile_skip_flags(profile),
    )


async def finalize_revision_if_complete(revision_id) -> None:
    """Opportunistically verify + publish a revision whose artifacts are complete.

    Safe to call from every worker stage: it is a no-op when the build is
    incomplete, the revision is already terminal, or the document was
    tombstoned. ``verify_draft`` and ``publish`` intentionally run in
    SEPARATE transactions (the Task-3 residual ABBA is avoided), and a
    failed draft is marked only for its own revision.
    """
    from app.core.database import async_session_maker

    # Transaction 1: verify (draft|building -> verified) when not already
    # verified. A missing required artifact is "not ready yet", not a failure.
    async with async_session_maker() as db:
        repo = DocumentRevisionsRepository(db)
        revision = await repo.get(revision_id)
        if revision is None or revision.status in TERMINAL_REVISION_STATUSES:
            return
        if revision.status in ("draft", "building"):
            try:
                await repo.verify_draft(revision_id)
                await db.commit()
            except RevisionArtifactsIncomplete:
                return
            except ValueError:
                # Non-verifiable state (e.g. raced to abandoned) — stop.
                return

    # Transaction 2: publish (verified -> published) via the generation CAS.
    async with async_session_maker() as db:
        repo = DocumentRevisionsRepository(db)
        revision = await repo.get(revision_id)
        if revision is None or revision.status in TERMINAL_REVISION_STATUSES:
            return
        if revision.status != "verified":
            return
        try:
            revision, outcome = await repo.publish(revision_id)
            await db.commit()
        except RevisionNotPublishable:
            return
        if outcome is PublishOutcome.ABANDONED_SOURCE_DELETED:
            logger.info(
                "[finalize] revision %s abandoned — source tombstoned",
                revision_id,
            )



async def check_and_finalize(
    document: Document, db: AsyncSession, *, revision_id=None
) -> None:
    """
    Transition document status based on sub-task completion:

      - is_chat_upload + embed_done → INDEXED (chat temp files: parse → embed only)
      - embed_done + captions_done + kg_done → INDEXED (full pipeline)
      - embed_done + captions_done (kg still running) → BUILDING_KG
      - otherwise → no change (still EMBEDDING or CHUNKING)

    Chat-upload documents skip KG and caption workers, so only embed_done
    is needed before marking INDEXED.

    Opens a *separate* session so it always reads the latest committed values
    from the other workers (avoids stale snapshot from the caller's long-lived
    transaction).  SELECT FOR UPDATE serialises concurrent calls so only one
    worker promotes the document.

    The status transition above is the **v1/UI mirror** of pipeline progress.
    When ``revision_id`` is supplied, reaching the completion condition also
    finalises the owning revision (``verify_draft`` then ``publish`` in
    separate transactions, see :func:`finalize_revision_if_complete`). This
    function is therefore the single publication choke point for a
    multi-stage build; individual stage execution decisions are made from
    revision-owned state by each worker, never from these flags.
    """
    from app.core.database import async_session_maker

    completed = False
    async with async_session_maker() as fresh_db:
        result = await fresh_db.execute(
            select(Document)
            .where(Document.id == document.id)
            .with_for_update()
        )
        fresh = result.scalar_one_or_none()
        if fresh is None:
            return

        # FAILED is a terminal state — only admin retry can clear it
        if fresh.status == DocumentStatus.FAILED:
            return

        changed = False

        # Chat-upload documents: skip KG and caption workers, so only embed_done is needed
        if fresh.is_chat_upload:
            if fresh.embed_done:
                completed = True
                if fresh.raw_chunks_json is not None:
                    fresh.raw_chunks_json = None
                    changed = True
                if fresh.status != DocumentStatus.INDEXED:
                    fresh.status = DocumentStatus.INDEXED
                    changed = True
                    logger.info(
                        f"[finalize] doc={fresh.id} → INDEXED "
                        f"(chat-upload: embed✓)"
                    )
                if changed:
                    await fresh_db.commit()
        elif fresh.embed_done and fresh.captions_done:
            # captions_done is set AFTER the caption re-embed ran, so no worker
            # needs the raw chunks anymore — free the (potentially large) column.
            # The KG worker reads markdown from MinIO, not from raw_chunks_json.
            if fresh.raw_chunks_json is not None:
                fresh.raw_chunks_json = None
                changed = True
            if fresh.kg_done:
                # All three done → INDEXED
                completed = True
                if fresh.status != DocumentStatus.INDEXED:
                    fresh.status = DocumentStatus.INDEXED
                    changed = True
                    logger.info(
                        f"[finalize] doc={fresh.id} → INDEXED "
                        f"(embed✓ captions✓ kg✓)"
                    )
            else:
                # embed+captions done, KG still running → BUILDING_KG
                if fresh.status not in (DocumentStatus.BUILDING_KG, DocumentStatus.INDEXED):
                    fresh.status = DocumentStatus.BUILDING_KG
                    changed = True
                    logger.info(
                        f"[finalize] doc={fresh.id} → BUILDING_KG "
                        f"(embed✓ captions✓ kg⟳)"
                    )
            if changed:
                await fresh_db.commit()

    # Finalise the revision OUTSIDE the mirror session: verify_draft and
    # publish take their own document/revision locks and must not nest inside
    # the FOR UPDATE transaction above.
    if completed and revision_id is not None:
        await finalize_revision_if_complete(revision_id)

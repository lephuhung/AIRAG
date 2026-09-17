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
from enum import Enum

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import (
    Document,
    DocumentImage,
    DocumentStatus,
    DocumentTable,
)
from app.models.document_revision import DocumentRevision
from app.models.document_revision_build import DocumentRevisionBuild
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


class FinalizeOutcome(str, Enum):
    """Outcome of one opportunistic revision finalization.

    - ``PUBLISHED``: the revision is published (it just published, or was
      already published by a racing stage).
    - ``NOT_READY``: the build is still incomplete, or another worker raced the
      revision into a non-ready state. Nothing to do yet.
    - ``FAILED``: the revision is terminal ``failed``.
    - ``ABANDONED``: the revision was abandoned (the source was tombstoned).
    - ``NO_REVISION``: no revision row exists for the id.
    """

    PUBLISHED = "published"
    NOT_READY = "not_ready"
    FAILED = "failed"
    ABANDONED = "abandoned"
    NO_REVISION = "no_revision"


@dataclass(frozen=True)
class FinalizeResult:
    """Result of :func:`finalize_revision_if_complete`.

    ``failure_stage`` / ``failure_class`` are set for ``FAILED`` so the caller
    can mirror the reason onto the ``Document`` row.
    """

    outcome: FinalizeOutcome
    failure_stage: str | None = None
    failure_class: str | None = None


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


async def mark_revision_failed(
    db: AsyncSession,
    revision_id,
    *,
    stage: str,
    error_class: str,
) -> None:
    """Terminalize a revision after a hard stage failure.

    A failed revision is immutable and never resumed: the stage messages still
    in flight dead-letter on :func:`load_revision_execution`, and only a bounded
    automatic retry or an explicit reindex allocates a new generation.

    Best-effort by design: the caller is already unwinding an exception, so a
    failure here is logged and swallowed (it must not mask the original error).
    Already-terminal revisions are left untouched.
    """
    try:
        repo = DocumentRevisionsRepository(db)
        revision = await repo.get(revision_id)
        if revision is None or revision.status in TERMINAL_REVISION_STATUSES:
            return
        await repo.mark_failed(revision_id, stage=stage, error_class=error_class)
        await db.commit()
    except Exception as exc:  # noqa: BLE001 - must not mask the stage error
        logger.warning(
            "[revision] could not mark revision %s failed (stage=%s): %s",
            revision_id,
            stage,
            exc,
        )
        await db.rollback()


async def load_revision_caption_targets(
    db: AsyncSession,
    *,
    document_id,
    revision_id,
) -> tuple[list[DocumentImage], list[DocumentTable]]:
    """Load the image/table rows owned by ``(document_id, revision_id)``.

    Captioning a published R1 must never consume R2's rows (or vice versa):
    the stage operates on exactly the revision its message names. Legacy
    ``revision_id IS NULL`` rows are intentionally excluded — a revision-aware
    build owns its own children, and re-captioning a legacy row would mutate
    another revision's artifact.
    """
    images = list(
        (
            await db.scalars(
                select(DocumentImage).where(
                    DocumentImage.document_id == document_id,
                    DocumentImage.revision_id == revision_id,
                )
            )
        ).all()
    )
    tables = list(
        (
            await db.scalars(
                select(DocumentTable).where(
                    DocumentTable.document_id == document_id,
                    DocumentTable.revision_id == revision_id,
                )
            )
        ).all()
    )
    return images, tables


async def load_revision_chunk_payloads(
    db: AsyncSession, revision_id
) -> list[dict] | None:
    """Load the revision's OWN chunk payloads from its structure artifact.

    The structure artifact is the authoritative revision-qualified chunk
    store; ``Document.raw_chunks_json`` is only a v1/UI mirror. Returns
    ``None`` when the revision has no structure artifact (a legacy in-flight
    message, or a parse stage that predates the artifact) so the caller can
    fall back to the mirror.
    """
    from app.services.agents.v2.persistence.document_views import (
        parse_structure_artifact,
    )

    build = await db.scalar(
        select(DocumentRevisionBuild)
        .where(DocumentRevisionBuild.revision_id == revision_id)
        .order_by(DocumentRevisionBuild.finished_at.desc().nullslast())
        .limit(1)
    )
    if build is None or not build.structure_artifact_key:
        return None
    from app.services.storage_service import get_storage_service

    raw = await get_storage_service().download_markdown(
        build.structure_artifact_key
    )
    return [
        {
            "chunk_id": record.chunk_id,
            "content": record.content,
            "chunk_index": record.ordinal,
            "source_file": record.source_file,
            "page_no": record.page_no,
            "heading_path": list(record.heading_path),
            "image_refs": list(record.image_refs),
            "table_refs": list(record.table_refs),
            "has_table": record.has_table,
            "has_code": record.has_code,
            "khoan_nos": list(record.khoan_nos),
            "diem_labels": list(record.diem_labels),
            "subdivision_refs": list(record.subdivision_refs),
            "subdivision_schema_version": record.subdivision_schema_version,
        }
        for record in parse_structure_artifact(raw)
    ]


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


def _terminal_finalize_result(revision: DocumentRevision) -> FinalizeResult:
    """Classify an already-terminal revision for the finalization caller."""
    if revision.status == "failed":
        return FinalizeResult(
            FinalizeOutcome.FAILED,
            failure_stage=revision.failure_stage,
            failure_class=revision.failure_class,
        )
    if revision.status == "abandoned":
        return FinalizeResult(FinalizeOutcome.ABANDONED)
    return FinalizeResult(FinalizeOutcome.PUBLISHED)


async def finalize_revision_if_complete(
    revision_id, *, expect_complete: bool = False
) -> FinalizeResult:
    """Opportunistically verify + publish a revision whose artifacts are complete.

    Safe to call from every worker stage: it is a no-op when the build is
    incomplete, the revision is already terminal, or the document was
    tombstoned. ``verify_draft`` and ``publish`` intentionally run in
    SEPARATE transactions (the Task-3 residual ABBA is avoided), and a
    failed draft is marked only for its own revision.

    ``expect_complete`` is the caller's assertion that every stage the build
    profile requires has reported done (the v1/UI mirror reached completion).
    In that case a ``RevisionArtifactsIncomplete`` is NOT "not ready yet": no
    later stage will arrive, so the revision would otherwise stay ``building``
    forever. It is logged at error and terminalized via ``mark_failed`` so the
    failure is visible and the revision is never resumed.

    Returns the :class:`FinalizeResult` the caller must mirror onto the
    ``Document`` row (see :func:`apply_finalize_outcome`): a worker must never
    mark a document ``INDEXED`` when its revision did not publish.
    """
    from app.core.database import async_session_maker

    # Transaction 1: verify (draft|building -> verified) when not already
    # verified. A missing required artifact is "not ready yet", not a failure —
    # UNLESS the caller has asserted completion.
    async with async_session_maker() as db:
        repo = DocumentRevisionsRepository(db)
        revision = await repo.get(revision_id)
        if revision is None:
            return FinalizeResult(FinalizeOutcome.NO_REVISION)
        if revision.status in TERMINAL_REVISION_STATUSES:
            return _terminal_finalize_result(revision)
        if revision.status in ("draft", "building"):
            try:
                await repo.verify_draft(revision_id)
                await db.commit()
            except RevisionArtifactsIncomplete as exc:
                if expect_complete:
                    logger.error(
                        "[finalize] revision %s reached document completion but "
                        "its artifacts are incomplete — marking failed: %s",
                        revision_id,
                        exc,
                    )
                    failure_class = type(exc).__name__
                    try:
                        await repo.mark_failed(
                            revision_id,
                            stage="verify",
                            error_class=failure_class,
                        )
                        await db.commit()
                    except Exception:
                        await db.rollback()
                        return FinalizeResult(FinalizeOutcome.NOT_READY)
                    return FinalizeResult(
                        FinalizeOutcome.FAILED,
                        failure_stage="verify",
                        failure_class=failure_class,
                    )
                return FinalizeResult(FinalizeOutcome.NOT_READY)
            except ValueError:
                # Non-verifiable state (e.g. raced to abandoned) — stop.
                return FinalizeResult(FinalizeOutcome.NOT_READY)

    # Transaction 2: publish (verified -> published) via the generation CAS.
    async with async_session_maker() as db:
        repo = DocumentRevisionsRepository(db)
        revision = await repo.get(revision_id)
        if revision is None:
            return FinalizeResult(FinalizeOutcome.NO_REVISION)
        if revision.status in TERMINAL_REVISION_STATUSES:
            return _terminal_finalize_result(revision)
        if revision.status != "verified":
            return FinalizeResult(FinalizeOutcome.NOT_READY)
        try:
            revision, outcome = await repo.publish(revision_id)
            await db.commit()
        except RevisionNotPublishable:
            return FinalizeResult(FinalizeOutcome.NOT_READY)
        if outcome is PublishOutcome.ABANDONED_SOURCE_DELETED:
            logger.info(
                "[finalize] revision %s abandoned — source tombstoned",
                revision_id,
            )
            return FinalizeResult(FinalizeOutcome.ABANDONED)
        return FinalizeResult(FinalizeOutcome.PUBLISHED)


async def apply_finalize_outcome(
    document_id, result: FinalizeResult, *, revision_id=None
) -> None:
    """Mirror a finalization outcome onto the ``Document`` row (v1/UI mirror).

    The revision is authoritative: ``INDEXED`` is written ONLY for
    ``PUBLISHED``, and a revision terminalized ``failed`` mirrors ``FAILED``
    with the failure stage/class — so a document is never reported indexed
    while its revision failed. ``NOT_READY`` / ``ABANDONED`` / ``NO_REVISION``
    leave the mirror untouched (the pipeline outcome is not final).

    ``revision_id`` is the generation guard: once the document's stable
    pointer (``current_revision_id``) names a different revision, this
    revision's late outcome no longer describes the document. A superseded
    build's failure must never mark a live, newer document permanently
    ``FAILED`` (nor a losing ``PUBLISHED`` rewrite the pointer's status), so
    the mirror is left untouched in that case.
    """
    from app.core.database import async_session_maker

    async with async_session_maker() as db:
        row = await db.execute(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        fresh = row.scalar_one_or_none()
        if fresh is None:
            return
        if (
            revision_id is not None
            and fresh.current_revision_id is not None
            and fresh.current_revision_id != revision_id
        ):
            logger.info(
                f"[finalize] doc={document_id} ignoring {result.outcome.value} "
                f"from superseded revision {revision_id} "
                f"(current={fresh.current_revision_id})"
            )
            return
        # FAILED is terminal — only an admin retry clears it, and its error
        # message (the original stage failure) is more useful than ours.
        if fresh.status == DocumentStatus.FAILED:
            return
        if result.outcome is FinalizeOutcome.PUBLISHED:
            if fresh.status != DocumentStatus.INDEXED:
                fresh.status = DocumentStatus.INDEXED
                await db.commit()
                logger.info(
                    f"[finalize] doc={document_id} → INDEXED (revision published)"
                )
            return
        if result.outcome is FinalizeOutcome.FAILED:
            fresh.status = DocumentStatus.FAILED
            fresh.error_message = (
                f"revision_failed: {result.failure_stage or 'verify'}: "
                f"{result.failure_class or 'unknown'}"
            )[:500]
            await db.commit()
            logger.error(
                f"[finalize] doc={document_id} → FAILED "
                f"(revision {result.failure_stage}:{result.failure_class})"
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
    When ``revision_id`` is supplied, the revision's OWN stage rows — not the
    ``Document`` mirror flags — decide whether the owning revision may
    finalise. ``required_stages_complete`` loads the revision's immutable
    allocated profile plus every stage row; a pending/running/incomplete
    stage returns NOT_READY without calling
    :func:`finalize_revision_if_complete`. Only when every required stage is
    ``completed`` (or profile-authorized ``skipped``) does the revision
    finalise (``verify_draft`` then ``publish`` in separate transactions,
    see :func:`finalize_revision_if_complete`), and the mirror status follows
    the REVISION outcome (:func:`apply_finalize_outcome`): ``INDEXED`` is
    written only when the revision actually published, and ``FAILED`` when
    the revision was terminalized by verification. The manifest check inside
    ``verify_draft``, the tombstone handling inside ``publish``, and the
    generation CAS/guard stay authoritative alongside the stage rows — a
    completed stage row never certifies an absent artifact, and an older
    generation never rewrites a newer pointer/mirror. The inline ``INDEXED``
    writes below are the legacy path for a message that carries no revision.
    This function is therefore the single publication choke point for a
    multi-stage build; individual stage execution decisions are made from
    revision-owned state by each worker, never from these flags.
    """
    from app.core.database import async_session_maker

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

        # A message naming a revision must NOT flip the mirror to INDEXED by
        # itself: the revision is authoritative and may turn out to be
        # unpublishable. ``apply_finalize_outcome`` writes the final status.
        mirror_completes = revision_id is None

        # Chat-upload documents: skip KG and caption workers, so only embed_done is needed
        if fresh.is_chat_upload:
            if fresh.embed_done:
                if fresh.raw_chunks_json is not None:
                    fresh.raw_chunks_json = None
                    changed = True
                if mirror_completes and fresh.status != DocumentStatus.INDEXED:
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
                # All three done → the stage gate below decides; the mirror
                # status follows the revision outcome via apply_finalize_outcome.
                if mirror_completes and fresh.status != DocumentStatus.INDEXED:
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
    # the FOR UPDATE transaction above. The revision-owned stage gate (not
    # the mirror flags read above) authorizes finalization: stale
    # ``Document.*_done`` flags neither authorize it (a fast stage must not
    # terminalize its generation via ``expect_complete`` while siblings are
    # still working) nor block it (a slow mirror write must not hold back a
    # fully-built revision). Reaching stage completion means every required
    # stage reported done, so an incomplete artifact set is a real failure
    # (not "not ready yet") — the manifest check stays authoritative.
    if revision_id is not None:
        async with async_session_maker() as gate_db:
            gate_repo = DocumentRevisionsRepository(gate_db)
            stages_complete = await gate_repo.required_stages_complete(
                revision_id
            )
        if not stages_complete:
            return
        outcome = await finalize_revision_if_complete(
            revision_id, expect_complete=True
        )
        await apply_finalize_outcome(document.id, outcome, revision_id=revision_id)

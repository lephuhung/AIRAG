"""
Queue Publisher
===============
High-level helpers called by the FastAPI ingestion producers and by
the worker pipeline when it fans out child messages.

Every producer publishes only a ``revision_id`` returned by the
revision lifecycle: the direct-upload / presigned-confirm / chat paths
converge via the idempotent
:meth:`DocumentRevisionsRepository.get_or_create_ingestion_attempt`,
while an explicit reindex/clone allocates a NEW generation via
:meth:`DocumentRevisionsRepository.allocate_draft`. Queue retry /
redelivery never re-allocates on its own — it reuses the message's
``revision_id``.
"""

from __future__ import annotations

import uuid

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document_revision import DocumentRevision
from app.models.source_arrival import SourceArrival
from app.queue.messages import MemorySaveMessage, ParseMessage
from app.services.agents.v2.persistence.document_revisions import (
    DocumentRevisionsRepository,
)
from app.services.agents.v2.persistence.source_identity import (
    IngestFlags,
    RevisionBuildProfile,
    arrival_identity,
    compute_source_object_identity,
    normalize_object_key,
    resolve_build_profile,
)


def _profile_value(build_profile: RevisionBuildProfile | str) -> str:
    return (
        build_profile.value
        if isinstance(build_profile, RevisionBuildProfile)
        else str(build_profile)
    )


async def record_source_arrival(
    db: AsyncSession,
    *,
    bucket: str,
    object_key: str,
    version_id: str | None,
    etag: str | None,
    size_bytes: int | None,
) -> str:
    """Record a webhook ``ObjectCreated`` arrival (metadata-only).

    Writes/keeps exactly one ``source_arrivals`` row keyed by
    :func:`arrival_identity` and returns that key. It never reads the
    object body and never allocates a revision — ``/confirm`` is the
    authoritative producer for the presigned flow.

    Idempotent: a duplicate webhook for the same arrival is a no-op.
    """
    identity = arrival_identity(
        bucket, object_key, version_id, etag, size_bytes or 0
    )
    stmt = (
        pg_insert(SourceArrival)
        .values(
            arrival_id=uuid.uuid4(),
            bucket=bucket,
            object_key=normalize_object_key(object_key),
            version_id=version_id,
            etag=etag,
            size_bytes=size_bytes,
            arrival_identity=identity,
        )
        .on_conflict_do_nothing(index_elements=["arrival_identity"])
    )
    await db.execute(stmt)
    await db.flush()
    return identity


async def allocate_ingest_revision(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    object_key: str,
    size_bytes: int,
    content_sha256: str,
    bucket: str | None = None,
    version_id: str | None = None,
    etag: str | None = None,
    parse_only: bool = False,
) -> tuple[DocumentRevision, RevisionBuildProfile, bool]:
    """Idempotently allocate/return the revision for one ingest event.

    Computes the canonical attempt identity (the unique index is the
    arbiter, so concurrent direct-upload / ``/confirm`` / webhook-derived
    callers converge) and the deterministic build profile, then delegates
    to ``get_or_create_ingestion_attempt``.

    Returns ``(revision, build_profile, created)``. The caller must not
    commit ownership to this helper — it flushes and lets the request
    unit-of-work commit.

    :raises MissingObjectVersion: if neither ``version_id`` nor ``etag``
      is supplied (a version selector is required).
    """
    bucket = bucket or settings.MINIO_BUCKET_UPLOADS
    identity = compute_source_object_identity(
        bucket, object_key, version_id, etag, size_bytes, content_sha256
    )
    profile = resolve_build_profile(object_key, IngestFlags(parse_only=parse_only))
    repo = DocumentRevisionsRepository(db)
    revision, created = await repo.get_or_create_ingestion_attempt(
        document_id, identity, profile
    )
    return revision, profile, created


async def _allocate_explicit_revision(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    object_key: str,
    size_bytes: int,
    content_sha256: str,
    bucket: str | None = None,
    version_id: str | None = None,
    etag: str | None = None,
    reindex_of_revision_id: uuid.UUID | None = None,
    cloned_from_revision_id: uuid.UUID | None = None,
    parse_only: bool = False,
) -> tuple[DocumentRevision, RevisionBuildProfile]:
    bucket = bucket or settings.MINIO_BUCKET_UPLOADS
    identity = compute_source_object_identity(
        bucket, object_key, version_id, etag, size_bytes, content_sha256
    )
    profile = resolve_build_profile(object_key, IngestFlags(parse_only=parse_only))
    repo = DocumentRevisionsRepository(db)
    revision = await repo.allocate_draft(
        document_id,
        identity,
        profile,
        reindex_of_revision_id=reindex_of_revision_id,
        cloned_from_revision_id=cloned_from_revision_id,
    )
    return revision, profile


async def allocate_reindex_revision(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    object_key: str,
    size_bytes: int,
    content_sha256: str,
    bucket: str | None = None,
    version_id: str | None = None,
    etag: str | None = None,
    reindex_of_revision_id: uuid.UUID | None = None,
    parse_only: bool = False,
) -> tuple[DocumentRevision, RevisionBuildProfile]:
    """Explicit user reindex: always a NEW monotonic generation.

    Not idempotent and not a redelivery: even when ``object_key`` (the
    unchanged ``upload_s3_key``) and content are identical, this
    allocates a higher generation and records ``reindex_of_revision_id``
    provenance.
    """
    return await _allocate_explicit_revision(
        db,
        document_id,
        object_key=object_key,
        size_bytes=size_bytes,
        content_sha256=content_sha256,
        bucket=bucket,
        version_id=version_id,
        etag=etag,
        reindex_of_revision_id=reindex_of_revision_id,
        parse_only=parse_only,
    )


async def allocate_clone_revision(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    object_key: str,
    size_bytes: int,
    content_sha256: str,
    bucket: str | None = None,
    version_id: str | None = None,
    etag: str | None = None,
    cloned_from_revision_id: uuid.UUID | None = None,
    parse_only: bool = False,
) -> tuple[DocumentRevision, RevisionBuildProfile]:
    """Workspace clone: a NEW target-workspace generation with provenance.

    The clone reuses the source document's raw object/hash only; it does
    not copy markdown/vectors, and it never reuses another workspace's
    revision identity as the target binding revision.
    """
    return await _allocate_explicit_revision(
        db,
        document_id,
        object_key=object_key,
        size_bytes=size_bytes,
        content_sha256=content_sha256,
        bucket=bucket,
        version_id=version_id,
        etag=etag,
        cloned_from_revision_id=cloned_from_revision_id,
        parse_only=parse_only,
    )


async def publish_parse_task(
    document_id: uuid.UUID,
    workspace_id: uuid.UUID,
    minio_key: str,
    original_filename: str,
    *,
    revision_id: uuid.UUID,
    build_profile: RevisionBuildProfile | str,
) -> None:
    """Publish a ParseMessage for an already-allocated revision."""
    profile = _profile_value(build_profile)
    is_chat_upload = profile == RevisionBuildProfile.CHAT_UPLOAD.value
    # Lazy import: the broker client (aio_pika) is only needed when a message is
    # actually published, so the revision-allocation helpers stay importable
    # (and unit-testable) without a live RabbitMQ dependency.
    from app.queue import connection as mq

    await mq.publish(
        mq.EXCHANGE_PARSE,
        "parse",
        ParseMessage(
            document_id=document_id,
            workspace_id=workspace_id,
            revision_id=revision_id,
            build_profile=profile,
            minio_key=minio_key,
            original_filename=original_filename,
            is_chat_upload=is_chat_upload,
        ).model_dump(mode="json"),
    )


async def allocate_commit_and_publish_parse(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    workspace_id: uuid.UUID,
    object_key: str,
    original_filename: str,
    size_bytes: int,
    content_sha256: str,
    bucket: str | None = None,
    version_id: str | None = None,
    etag: str | None = None,
    parse_only: bool = False,
) -> tuple[DocumentRevision, RevisionBuildProfile, bool]:
    """Allocate → COMMIT → publish the parse task for one ingest event.

    The allocation is idempotent (``get_or_create_ingestion_attempt``), so a
    duplicate ``/confirm`` / direct upload racing a webhook-derived caller
    converges on the same draft. The commit is NOT optional: ``AsyncSession``
    is not autocommit and the request unit-of-work rolls back on return, so a
    merely-flushed revision would vanish and the worker would dead-letter the
    published message as ``unknown_revision``. Committing first also makes the
    draft durable before another connection (the worker) can observe it.

    On publish failure the committed draft is marked ``failed`` (terminal) and
    the exception is re-raised, so the caller can mirror the failure onto the
    ``Document`` row. Returns ``(revision, build_profile, created)``.
    """
    revision, profile, created = await allocate_ingest_revision(
        db,
        document_id,
        object_key=object_key,
        size_bytes=size_bytes,
        content_sha256=content_sha256,
        bucket=bucket,
        version_id=version_id,
        etag=etag,
        parse_only=parse_only,
    )
    await db.commit()
    try:
        await publish_parse_task(
            document_id=document_id,
            workspace_id=workspace_id,
            minio_key=object_key,
            original_filename=original_filename,
            revision_id=revision.revision_id,
            build_profile=profile,
        )
    except Exception as exc:
        # The draft is already committed: terminalize ONLY this revision so it
        # is never resumed, then let the caller mirror Document FAILED.
        try:
            repo = DocumentRevisionsRepository(db)
            await repo.mark_failed(
                revision.revision_id,
                stage="publish",
                error_class=type(exc).__name__,
            )
            await db.commit()
        except Exception:
            await db.rollback()
        raise
    return revision, profile, created


async def publish_memory_save_task(
    user_id: uuid.UUID,
    user_message: str,
    assistant_message: str = "",
    session_id: str | None = None,
) -> None:
    """Publish a MemorySaveMessage to the durable hrag.memory queue.

    Replaces the old fire-and-forget ``asyncio.create_task`` save: the memory
    worker does the LLM fact-extraction + Graphiti write, and RabbitMQ retries
    transient failures (5s/15s/60s) before dead-lettering — so a fact is not
    lost if the worker, Neo4j, or the LLM is briefly unavailable.
    """
    from app.queue import connection as mq

    await mq.publish(
        mq.EXCHANGE_MEMORY,
        "memory",
        MemorySaveMessage(
            user_id=user_id,
            user_message=user_message,
            assistant_message=assistant_message,
            session_id=session_id,
        ).model_dump(mode="json"),
    )

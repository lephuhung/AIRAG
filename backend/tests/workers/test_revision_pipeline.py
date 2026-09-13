"""Revision propagation through ingestion producers and workers — Task 4.

These tests exercise the seam the Task-4 brief defines: every ingestion
trigger allocates a revision through the lifecycle, every queue message
carries that revision UUID, and worker execution decisions come from
revision-owned state rather than mutable ``Document`` completion flags.

The FastAPI endpoints themselves are thin wrappers over the producer
helpers in :mod:`app.queue.publisher` (``record_source_arrival``,
``allocate_ingest_revision``, ``allocate_reindex_revision``,
``allocate_clone_revision``) and the worker control helpers in
:mod:`app.workers.utils`; this suite drives those helpers directly against
the real Postgres v2 schema (the same bootstrap Task 3 uses), which is what
the endpoint/webhook code calls.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.document import Document, DocumentImage, DocumentStatus, DocumentTable
from app.models.document_ingestion_attempt import DocumentIngestionAttempt
from app.models.document_revision import DocumentRevision
from app.models.source_arrival import SourceArrival
from app.queue.messages import CaptionMessage, EmbedMessage, KGMessage, ParseMessage
from app.queue.publisher import (
    allocate_clone_revision,
    allocate_commit_and_publish_parse,
    allocate_commit_and_publish_retry,
    allocate_ingest_revision,
    allocate_reindex_revision,
    record_source_arrival,
)
from app.services.agents.v2.persistence.document_revisions import (
    DocumentRevisionsRepository,
    PublishOutcome,
)
from app.services.agents.v2.persistence.source_identity import (
    IngestFlags,
    RevisionBuildProfile,
    compute_source_object_identity,
    resolve_build_profile,
)
from app.workers.utils import (
    FinalizeOutcome,
    apply_finalize_outcome,
    check_and_finalize,
    delete_stage_children,
    finalize_revision_if_complete,
    load_revision_caption_targets,
    load_revision_execution,
    mark_revision_failed,
    profile_skip_flags,
    record_embed_artifacts,
    record_parse_artifacts,
)

BUCKET = "hrag-uploads"

FULL = RevisionBuildProfile.FULL
CHAT = RevisionBuildProfile.CHAT_UPLOAD
PARSE_ONLY = RevisionBuildProfile.PARSE_ONLY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _workspace_id(db, document_id: uuid.UUID) -> uuid.UUID:
    return await db.scalar(
        select(Document.workspace_id).where(Document.id == document_id)
    )


def _doc_key(workspace_id: uuid.UUID, document_id: uuid.UUID) -> str:
    return f"kb_{workspace_id}/doc_{document_id}.pdf"


def _chat_key(workspace_id: uuid.UUID, document_id: uuid.UUID) -> str:
    return f"kb_{workspace_id}/chat_file_{document_id}.pdf"


async def _revision_count(db, document_id: uuid.UUID) -> int:
    return int(
        await db.scalar(
            select(func.count())
            .select_from(DocumentRevision)
            .where(DocumentRevision.document_id == document_id)
        )
    )


async def _attempt_count(db, document_id: uuid.UUID) -> int:
    return int(
        await db.scalar(
            select(func.count())
            .select_from(DocumentIngestionAttempt)
            .where(DocumentIngestionAttempt.document_id == document_id)
        )
    )


async def _record_complete_artifacts(db, revision, profile) -> None:
    """Record a build manifest that satisfies ``verify_draft`` for ``profile``."""
    repo = DocumentRevisionsRepository(db)
    flags = profile_skip_flags(profile)
    manifest = {}
    if profile in (FULL, CHAT):
        manifest = dict(
            embedding_namespace="ws_test",
            embedding_model_hash="hash",
            embedding_dimension=1024,
            vector_artifact_version="v1",
        )
    await repo.record_artifacts(
        revision.revision_id,
        profile,
        markdown_artifact_key="kb_x/doc.md",
        structure_artifact_key="kb_x/doc.structure",
        **flags,
        **manifest,
    )


async def _build_and_publish(db, revision, profile) -> None:
    await _record_complete_artifacts(db, revision, profile)
    repo = DocumentRevisionsRepository(db)
    await repo.verify_draft(revision.revision_id)
    await repo.publish(revision.revision_id)


async def _allocate_full(db, document_id, *, key, sha, size=11, version_id="v-1"):
    revision, profile, created = await allocate_ingest_revision(
        db,
        document_id,
        object_key=key,
        size_bytes=size,
        content_sha256=sha,
        version_id=version_id,
    )
    assert profile is FULL
    return revision, created


# ---------------------------------------------------------------------------
# Message constructors — revision_id is required with no default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [ParseMessage, EmbedMessage, CaptionMessage, KGMessage])
def test_message_constructors_require_revision_id(cls):
    from pydantic import ValidationError

    minimal = {"document_id": uuid.uuid4(), "workspace_id": uuid.uuid4()}
    if cls is ParseMessage:
        minimal.update(minio_key="kb_x/doc.md", original_filename="doc.pdf")
    with pytest.raises(ValidationError):
        cls(**minimal)
    revision_id = uuid.uuid4()
    msg = cls(
        revision_id=revision_id,
        build_profile="FULL",
        **minimal,
    )
    assert msg.revision_id == revision_id
    assert msg.build_profile == "FULL"


def _balanced_call(text: str, start: int) -> str:
    """Return the source of the parenthesized call beginning at ``start``."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _function_source(source: str, name: str) -> str:
    """Return the source of a module-level function definition."""
    match = re.search(rf"^async def {re.escape(name)}\(", source, re.MULTILINE)
    assert match is not None, f"{name} not found"
    following = re.search(r"^(?:@|async def |def )", source[match.end() :], re.MULTILINE)
    end = match.end() + (following.start() if following else len(source))
    return source[match.start() : end]


def test_all_pipeline_message_constructors_supply_revision_id():
    """No unclassified producer/worker constructor may omit ``revision_id``.

    Scans every ``ParseMessage(...)`` / ``EmbedMessage(...)`` /
    ``CaptionMessage(...)`` / ``KGMessage(...)`` call in ``app/`` and fails
    when the call does not name a ``revision_id``. Class definitions and the
    worker deserialization ``Cls(**payload)`` form are excluded (the payload
    itself carries the required field).
    """
    app_root = Path(__file__).resolve().parents[2] / "app"
    pattern = re.compile(r"\b(?:ParseMessage|EmbedMessage|CaptionMessage|KGMessage)\(")
    offenders: list[str] = []
    for path in sorted(app_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in pattern.finditer(source):
            line_start = source.rfind("\n", 0, match.start()) + 1
            line = source[line_start : source.find("\n", match.start())]
            if line.lstrip().startswith("class "):
                continue
            call = _balanced_call(source, match.end() - 1)
            if "**payload" in call:
                continue
            if "revision_id=" not in call:
                line_no = source.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(app_root.parent)}:{line_no}")
    assert not offenders, (
        "message constructors missing revision_id=: " + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Producer allocation — direct upload / confirm / chat upload converge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_and_confirm_create_one_revision(async_db, document_factory):
    """MinIO ``ObjectCreated`` and ``/confirm`` converge on one draft.

    The webhook is metadata-only: it stages one ``source_arrivals`` row and
    creates no revision. Two ``/confirm``-style allocations (a duplicate
    callback racing the webhook-derived caller) converge on one revision and
    one attempt.
    """
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    arrival_a = await record_source_arrival(
        async_db,
        bucket=BUCKET,
        object_key=key,
        version_id="v-1",
        etag=None,
        size_bytes=11,
    )
    arrival_b = await record_source_arrival(
        async_db,
        bucket=BUCKET,
        object_key=key,
        version_id="v-1",
        etag=None,
        size_bytes=11,
    )
    assert arrival_a == arrival_b
    # Webhook alone created no revision.
    assert await _revision_count(async_db, doc_id) == 0

    revision_a, profile_a, created_a = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="a" * 64,
        version_id="v-1",
    )
    revision_b, profile_b, created_b = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="a" * 64,
        version_id="v-1",
    )
    assert revision_a.revision_id == revision_b.revision_id
    assert (created_a, created_b) == (True, False)
    assert profile_a is FULL and profile_b is FULL
    assert await _revision_count(async_db, doc_id) == 1
    assert await _attempt_count(async_db, doc_id) == 1


@pytest.mark.asyncio
async def test_duplicate_webhook_is_idempotent(async_db, document_factory):
    """A redelivered webhook returns the same staging key, no duplicate draft."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    first = await record_source_arrival(
        async_db, bucket=BUCKET, object_key=key, version_id="v-9", etag=None, size_bytes=5
    )
    second = await record_source_arrival(
        async_db, bucket=BUCKET, object_key=key, version_id="v-9", etag=None, size_bytes=5
    )
    assert first == second
    arrivals = await async_db.scalar(
        select(func.count())
        .select_from(SourceArrival)
        .where(SourceArrival.arrival_identity == first)
    )
    assert arrivals == 1
    assert await _revision_count(async_db, doc_id) == 0


@pytest.mark.asyncio
async def test_chat_upload_webhook_profile_is_preserved(async_db, document_factory):
    """A ``chat_file_<document_id>`` object yields ``CHAT_UPLOAD``, never ``FULL``.

    Real storage keys are namespaced ``kb_<workspace>/chat_file_<document>``;
    the canonical resolver matches the ``chat_file_`` prefix on the file-name
    segment of the key.
    """
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _chat_key(ws, doc_id)

    assert resolve_build_profile(key, IngestFlags()) is CHAT

    revision, profile, _created = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=7,
        content_sha256="b" * 64,
        version_id="v-2",
    )
    assert profile is CHAT
    attempt_profile = await async_db.scalar(
        select(DocumentIngestionAttempt.build_profile).where(
            DocumentIngestionAttempt.revision_id == revision.revision_id
        )
    )
    assert attempt_profile == CHAT.value


@pytest.mark.asyncio
async def test_parse_only_entrypoint_yields_parse_only_profile(async_db, document_factory):
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    _revision, profile, _created = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=3,
        content_sha256="c" * 64,
        version_id="v-3",
        parse_only=True,
    )
    assert profile is PARSE_ONLY
    # parse_only wins even over the chat prefix.
    assert resolve_build_profile(key, IngestFlags(parse_only=True)) is PARSE_ONLY


# ---------------------------------------------------------------------------
# Explicit reindex — always a new monotonic generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reindex_allocates_new_revision_for_same_source_object(
    async_db, document_factory
):
    """Explicit reindex over an unchanged ``upload_s3_key`` is a new generation."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    first, created = await _allocate_full(async_db, doc_id, key=key, sha="d" * 64)
    assert created is True and first.generation == 1
    await _build_and_publish(async_db, first, FULL)

    second, profile = await allocate_reindex_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="d" * 64,
        version_id="v-1",
        reindex_of_revision_id=first.revision_id,
    )
    assert profile is FULL
    assert second.revision_id != first.revision_id
    assert second.generation > first.generation
    assert second.reindex_of_revision_id == first.revision_id
    assert await _revision_count(async_db, doc_id) == 2

    # Redelivery of the same ingest event now converges on the explicit revision.
    redelivered, _profile, created_again = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="d" * 64,
        version_id="v-1",
    )
    assert redelivered.revision_id == second.revision_id
    assert created_again is False


@pytest.mark.asyncio
async def test_clone_allocates_target_workspace_revision(async_db, document_factory):
    """A clone creates a target-workspace revision; it never reuses source identity."""
    source_doc = document_factory()
    target_doc = document_factory()  # a DIFFERENT workspace/document
    ws_src = await _workspace_id(async_db, source_doc)
    ws_tgt = await _workspace_id(async_db, target_doc)

    source_rev, _created = await _allocate_full(
        async_db, source_doc, key=_doc_key(ws_src, source_doc), sha="e" * 64
    )
    await _build_and_publish(async_db, source_rev, FULL)

    target = await async_db.get(Document, target_doc)
    target.upload_s3_key = _doc_key(ws_tgt, target_doc)
    target.content_hash = "e" * 64
    target.file_size = 11
    await async_db.flush()

    clone_rev, profile = await allocate_clone_revision(
        async_db,
        target_doc,
        object_key=target.upload_s3_key,
        size_bytes=11,
        content_sha256="e" * 64,
        etag="e" * 32,
        cloned_from_revision_id=source_rev.revision_id,
    )
    assert profile is FULL
    assert clone_rev.document_id == target_doc
    assert clone_rev.document_id != source_rev.document_id
    assert clone_rev.cloned_from_revision_id == source_rev.revision_id
    assert clone_rev.generation == 1
    # The clone target is NOT marked indexed and carries no completion flags.
    assert target.status.value != "indexed"
    assert target.embed_done is False
    assert target.captions_done is False
    assert target.kg_done is False


# ---------------------------------------------------------------------------
# Worker scoping and execution decisions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_replaces_only_new_revision_children(async_db, document_factory):
    """Parsing R2 deletes only ``(document_id, R2)`` image/table rows.

    R1's SQL children and legacy (``revision_id IS NULL``) rows stay readable.
    """
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    r1, _ = await _allocate_full(async_db, doc_id, key=key, sha="f" * 64)
    r2, _profile = await allocate_reindex_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="f" * 64,
        version_id="v-1",
        reindex_of_revision_id=r1.revision_id,
    )
    for revision_id in (r1.revision_id, r2.revision_id, None):
        async_db.add(
            DocumentImage(
                document_id=doc_id,
                revision_id=revision_id,
                image_id=f"img-{revision_id}",
                page_no=1,
            )
        )
        async_db.add(
            DocumentTable(
                document_id=doc_id,
                revision_id=revision_id,
                table_id=f"tbl-{revision_id}",
                page_no=1,
                content_markdown="| a |",
            )
        )
    await async_db.flush()

    await delete_stage_children(
        async_db, document_id=doc_id, revision_id=r2.revision_id
    )

    remaining_images = [
        row.revision_id
        for row in (
            await async_db.scalars(
                select(DocumentImage).where(DocumentImage.document_id == doc_id)
            )
        ).all()
    ]
    remaining_tables = [
        row.revision_id
        for row in (
            await async_db.scalars(
                select(DocumentTable).where(DocumentTable.document_id == doc_id)
            )
        ).all()
    ]
    assert set(remaining_images) == {r1.revision_id, None}
    assert set(remaining_tables) == {r1.revision_id, None}


@pytest.mark.asyncio
async def test_terminal_revision_message_is_noop_dead_letter(
    async_db, document_factory
):
    """A message naming a terminal revision never runs and never re-allocates."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    draft, _ = await _allocate_full(async_db, doc_id, key=key, sha="1" * 64)
    decision = await load_revision_execution(
        async_db, revision_id=draft.revision_id, document_id=doc_id
    )
    assert decision.run is True

    repo = DocumentRevisionsRepository(async_db)
    await repo.mark_failed(draft.revision_id, stage="parse", error_class="Boom")
    failed = await load_revision_execution(
        async_db, revision_id=draft.revision_id, document_id=doc_id
    )
    assert failed.run is False
    assert failed.reason == "terminal:failed"
    assert await _revision_count(async_db, doc_id) == 1

    # Unknown revision is also a no-op (never an implicit allocation).
    unknown = await load_revision_execution(
        async_db, revision_id=uuid.uuid4(), document_id=doc_id
    )
    assert unknown.run is False
    assert unknown.reason == "unknown_revision"


@pytest.mark.asyncio
async def test_document_completion_flags_do_not_gate_revision_stage(
    async_db, document_factory
):
    """Stale ``Document`` mirror flags cannot make a new revision's stage skip.

    The revision is authoritative: a fresh draft runs even though the
    document still carries ``embed_done/captions_done/kg_done`` from R1.
    """
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    document = await async_db.get(Document, doc_id)
    document.embed_done = True
    document.captions_done = True
    document.kg_done = True
    await async_db.flush()

    r2, _profile = await allocate_reindex_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="2" * 64,
        version_id="v-1",
    )
    decision = await load_revision_execution(
        async_db, revision_id=r2.revision_id, document_id=doc_id
    )
    assert decision.run is True, "new draft must run despite stale Document flags"


@pytest.mark.asyncio
async def test_stale_message_cannot_publish_newer_revision(async_db, document_factory):
    """R1 finishing late cannot regress a document whose current revision is R2.

    R1 and R2 are distinct ingest events (an overwrite produced a new
    identity), each with its own attempt row, so both can be verified before
    either publishes. Publishing the winner first must leave the loser
    ``published`` but historical.
    """
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    r1, _ = await _allocate_full(async_db, doc_id, key=key, sha="3" * 64)
    r2, _created = await _allocate_full(async_db, doc_id, key=key, sha="6" * 64)
    assert r2.generation > r1.generation
    await _record_complete_artifacts(async_db, r1, FULL)
    await _record_complete_artifacts(async_db, r2, FULL)
    repo = DocumentRevisionsRepository(async_db)
    await repo.verify_draft(r1.revision_id)
    await repo.verify_draft(r2.revision_id)

    published, outcome = await repo.publish(r2.revision_id)
    assert outcome is PublishOutcome.BECAME_CURRENT

    stale, stale_outcome = await repo.publish(r1.revision_id)
    assert stale_outcome is PublishOutcome.PUBLISHED_HISTORICAL
    document = await async_db.get(Document, doc_id)
    assert document.current_revision_id == r2.revision_id
    assert stale.status == "published"


@pytest.mark.asyncio
async def test_worker_artifact_recording_satisfies_profile_verify(
    async_db, document_factory
):
    """The artifacts each worker records satisfy ``verify_draft`` for its profile.

    PARSE_ONLY: parse records markdown + the three intentional skips.
    CHAT_UPLOAD: parse records the skips, embed records the vector manifest.
    """
    # PARSE_ONLY
    parse_doc = document_factory()
    ws = await _workspace_id(async_db, parse_doc)
    parse_rev, parse_profile, _created = await allocate_ingest_revision(
        async_db,
        parse_doc,
        object_key=_doc_key(ws, parse_doc),
        size_bytes=11,
        content_sha256="7" * 64,
        version_id="v-1",
        parse_only=True,
    )
    assert parse_profile is PARSE_ONLY
    await record_parse_artifacts(
        async_db, parse_rev.revision_id, PARSE_ONLY, markdown_artifact_key="md/key"
    )
    repo = DocumentRevisionsRepository(async_db)
    await repo.verify_draft(parse_rev.revision_id)
    _rev, outcome = await repo.publish(parse_rev.revision_id)
    assert outcome is PublishOutcome.BECAME_CURRENT

    # CHAT_UPLOAD
    chat_doc = document_factory()
    ws2 = await _workspace_id(async_db, chat_doc)
    chat_rev, chat_profile, _created = await allocate_ingest_revision(
        async_db,
        chat_doc,
        object_key=_chat_key(ws2, chat_doc),
        size_bytes=11,
        content_sha256="8" * 64,
        version_id="v-1",
    )
    assert chat_profile is CHAT
    await record_parse_artifacts(
        async_db, chat_rev.revision_id, CHAT, markdown_artifact_key="md/key"
    )
    await record_embed_artifacts(
        async_db,
        chat_rev.revision_id,
        CHAT,
        embedding_namespace="ws_test",
        embedding_model_hash="hash",
        embedding_dimension=8,
        vector_artifact_version="v1",
    )
    await repo.verify_draft(chat_rev.revision_id)
    _rev, outcome = await repo.publish(chat_rev.revision_id)
    assert outcome is PublishOutcome.BECAME_CURRENT


# ---------------------------------------------------------------------------
# Canonical identity discipline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_content_same_version_converges_but_overwrite_does_not(
    async_db, document_factory
):
    """Identical re-uploads converge; new bytes produce a new attempt identity."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    first, _created = await _allocate_full(async_db, doc_id, key=key, sha="4" * 64)
    same, _created2 = await _allocate_full(async_db, doc_id, key=key, sha="4" * 64)
    assert same.revision_id == first.revision_id

    identity_a = compute_source_object_identity(
        BUCKET, key, "v-1", None, 11, "4" * 64
    )
    identity_b = compute_source_object_identity(
        BUCKET, key, "v-1", None, 11, "5" * 64
    )
    assert identity_a != identity_b
    # The canonical resolver keys off the file-name segment, so a namespaced
    # ``doc_`` key is FULL regardless of the workspace prefix.
    assert resolve_build_profile(key, IngestFlags()) is FULL


# ---------------------------------------------------------------------------
# Producer ordering — allocation is COMMITTED before the parse publish
# ---------------------------------------------------------------------------


def _session_maker(engine):
    """A real (non-SAVEPOINT) sessionmaker for cross-connection assertions."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
    )


@pytest.mark.asyncio
async def test_allocation_committed_before_parse_publish(
    async_engine, document_factory, monkeypatch
):
    """The ingest revision is committed BEFORE ``publish_parse_task`` runs.

    The worker loads the revision from its own connection. A merely-flushed
    allocation rolls back when the request unit-of-work returns, so the
    published message dead-letters as ``unknown_revision``. A *separate*
    session must therefore see the revision at the moment the publish happens.
    """
    maker = _session_maker(async_engine)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
    key = _doc_key(ws, doc_id)
    observed: dict = {}

    async def spy_publish_parse_task(
        *, document_id, workspace_id, minio_key, original_filename,
        revision_id, build_profile,
    ):
        # A fresh connection — proves the allocation was COMMITTED, not just
        # flushed on the caller's open transaction.
        async with maker() as observer:
            observed["revision"] = await observer.get(DocumentRevision, revision_id)
        observed["profile"] = build_profile

    monkeypatch.setattr(
        "app.queue.publisher.publish_parse_task", spy_publish_parse_task
    )

    async with maker() as db:
        revision, profile, created = await allocate_commit_and_publish_parse(
            db,
            doc_id,
            workspace_id=ws,
            object_key=key,
            original_filename="doc.pdf",
            size_bytes=11,
            content_sha256="a" * 64,
            version_id="v-1",
        )
        assert created is True

    assert observed.get("revision") is not None, (
        "revision must be committed before publish_parse_task is invoked"
    )
    assert observed["revision"].status == "draft"
    assert observed["profile"] is profile


@pytest.mark.asyncio
async def test_publish_failure_terminalizes_committed_draft(
    async_engine, document_factory, monkeypatch
):
    """A publish failure marks the already-committed draft ``failed``.

    The draft must not be left resumable, and the failure must be visible
    from a separate connection (it was committed before the exception was
    surfaced to the caller).
    """
    maker = _session_maker(async_engine)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
    key = _doc_key(ws, doc_id)
    captured: dict = {}

    async def boom_publish_parse_task(
        *, document_id, workspace_id, minio_key, original_filename,
        revision_id, build_profile,
    ):
        captured["revision_id"] = revision_id
        raise RuntimeError("broker down")

    monkeypatch.setattr(
        "app.queue.publisher.publish_parse_task", boom_publish_parse_task
    )

    async with maker() as db:
        with pytest.raises(RuntimeError):
            await allocate_commit_and_publish_parse(
                db,
                doc_id,
                workspace_id=ws,
                object_key=key,
                original_filename="doc.pdf",
                size_bytes=11,
                content_sha256="a" * 64,
                version_id="v-1",
            )

    async with maker() as db:
        row = await db.get(DocumentRevision, captured["revision_id"])
    assert row.status == "failed"
    assert row.failure_stage == "publish"
    assert row.failure_class == "RuntimeError"


# ---------------------------------------------------------------------------
# Caption worker — reads are scoped to the message's revision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caption_targets_scoped_to_revision(async_db, document_factory):
    """Captioning R2 loads only R2's image/table rows, never R1's or legacy rows."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    r1, _ = await _allocate_full(async_db, doc_id, key=key, sha="9" * 64)
    r2, _profile = await allocate_reindex_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="9" * 64,
        version_id="v-1",
        reindex_of_revision_id=r1.revision_id,
    )
    for revision_id in (r1.revision_id, r2.revision_id, None):
        async_db.add(
            DocumentImage(
                document_id=doc_id,
                revision_id=revision_id,
                image_id=f"img-{revision_id}",
                page_no=1,
            )
        )
        async_db.add(
            DocumentTable(
                document_id=doc_id,
                revision_id=revision_id,
                table_id=f"tbl-{revision_id}",
                page_no=1,
                content_markdown="| a |",
            )
        )
    await async_db.flush()

    images, tables = await load_revision_caption_targets(
        async_db, document_id=doc_id, revision_id=r2.revision_id
    )
    assert {img.revision_id for img in images} == {r2.revision_id}
    assert {tbl.revision_id for tbl in tables} == {r2.revision_id}


# ---------------------------------------------------------------------------
# Terminal failure — failed revisions are immutable dead-letters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_revision_failed_terminalizes_and_dead_letters(
    async_db, document_factory
):
    """A worker terminal failure marks only that revision and dead-letters it."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _ = await _allocate_full(async_db, doc_id, key=key, sha="a" * 64)

    await mark_revision_failed(
        async_db, revision.revision_id, stage="embed", error_class="Boom"
    )

    fresh = await async_db.get(DocumentRevision, revision.revision_id)
    assert fresh.status == "failed"
    assert fresh.failure_stage == "embed"
    assert fresh.failure_class == "Boom"

    decision = await load_revision_execution(
        async_db, revision_id=revision.revision_id, document_id=doc_id
    )
    assert decision.run is False
    assert decision.reason == "terminal:failed"

    # Idempotent: re-terminalizing a terminal revision is a silent no-op.
    await mark_revision_failed(
        async_db, revision.revision_id, stage="embed", error_class="Boom"
    )
    assert (await async_db.get(DocumentRevision, revision.revision_id)).status == "failed"


# ---------------------------------------------------------------------------
# Finalization — mirror complete + incomplete artifacts is a real failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_with_expect_complete_fails_incomplete_revision(
    async_engine, document_factory, monkeypatch
):
    """Mirror-complete + ``RevisionArtifactsIncomplete`` terminalizes the draft.

    Without this, a message whose stage can never record its artifact (e.g. an
    embed message with empty ``raw_chunks_json`` for a FULL/CHAT_UPLOAD build)
    leaves the revision ``building`` forever.
    """
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="d" * 64,
            version_id="v-1",
        )
        await db.commit()
        revision_id = revision.revision_id
    assert profile is FULL

    await finalize_revision_if_complete(revision_id, expect_complete=True)

    async with maker() as db:
        row = await db.get(DocumentRevision, revision_id)
    assert row.status == "failed"
    assert row.failure_stage == "verify"
    assert row.failure_class == "RevisionArtifactsIncomplete"


@pytest.mark.asyncio
async def test_finalize_without_expect_complete_leaves_incomplete_draft(
    async_engine, document_factory, monkeypatch
):
    """The opportunistic path still treats incomplete artifacts as "not ready"."""
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, _profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="e" * 64,
            version_id="v-1",
        )
        await db.commit()
        revision_id = revision.revision_id

    await finalize_revision_if_complete(revision_id)

    async with maker() as db:
        row = await db.get(DocumentRevision, revision_id)
    assert row.status == "draft"


@pytest.mark.asyncio
async def test_finalize_with_expect_complete_publishes_complete_revision(
    async_engine, document_factory, monkeypatch
):
    """A complete revision still verifies + publishes when expect_complete is set."""
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="f" * 64,
            version_id="v-1",
        )
        await _record_complete_artifacts(db, revision, profile)
        await db.commit()
        revision_id = revision.revision_id

    await finalize_revision_if_complete(revision_id, expect_complete=True)

    async with maker() as db:
        row = await db.get(DocumentRevision, revision_id)
        document = await db.get(Document, doc_id)
    assert row.status == "published"
    assert document.current_revision_id == revision_id


# ---------------------------------------------------------------------------
# Admin retry producers — explicit recovery always allocates a new generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_retry_publishes_new_revision_generation(
    async_engine, document_factory, monkeypatch
):
    """An operator retry allocates a HIGHER generation and publishes its UUID.

    ``ParseMessage`` requires ``revision_id`` + ``build_profile`` with no
    default, so the superadmin recovery path must publish a lifecycle revision:
    reusing the prior one would mutate it, and a revision-less payload would
    raise ValidationError and dead-letter (the operator recovery path would be
    broken). The allocation must also be committed before the publish, because
    the worker loads the revision from its own connection.
    """
    maker = _session_maker(async_engine)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        key = _doc_key(ws, doc_id)
        document = await db.get(Document, doc_id)
        document.upload_s3_key = key
        document.content_hash = "a" * 64
        prior, _created = await _allocate_full(db, doc_id, key=key, sha="a" * 64)
        await _build_and_publish(db, prior, FULL)
        await db.commit()
        prior_id = prior.revision_id
        prior_generation = prior.generation
    assert prior_generation == 1

    published: dict = {}

    async def spy_publish_parse_task(
        *, document_id, workspace_id, minio_key, original_filename,
        revision_id, build_profile,
    ):
        published.update(
            document_id=document_id,
            revision_id=revision_id,
            build_profile=build_profile,
            minio_key=minio_key,
        )
        # A separate connection — proves the allocation was COMMITTED.
        async with maker() as observer:
            published["revision"] = await observer.get(DocumentRevision, revision_id)

    monkeypatch.setattr(
        "app.queue.publisher.publish_parse_task", spy_publish_parse_task
    )

    async with maker() as db:
        revision, profile = await allocate_commit_and_publish_retry(
            db,
            doc_id,
            workspace_id=ws,
            object_key=key,
            original_filename="orig.pdf",
            size_bytes=1024,
            content_sha256="a" * 64,
            etag="a" * 64,
            previous_revision_id=prior_id,
        )

    assert revision.revision_id is not None
    assert revision.generation > prior_generation
    assert revision.reindex_of_revision_id == prior_id
    assert profile is FULL
    # The published payload carries the new revision and a valid profile.
    assert published["revision_id"] == revision.revision_id
    assert RevisionBuildProfile(published["build_profile"]) is FULL
    assert published["minio_key"] == key
    assert published["revision"] is not None, (
        "the retry revision must be committed before publish_parse_task runs"
    )
    assert published["revision"].status == "draft"
    # And it is exactly the payload shape the parser now accepts.
    ParseMessage(
        document_id=published["document_id"],
        workspace_id=ws,
        revision_id=published["revision_id"],
        build_profile=published["build_profile"],
        minio_key=published["minio_key"],
        original_filename="orig.pdf",
    )
    # Explicit retry never mutates the prior revision.
    async with maker() as db:
        prior_row = await db.get(DocumentRevision, prior_id)
    assert prior_row.status == "published"


@pytest.mark.asyncio
async def test_admin_retry_resolves_chat_profile_from_real_object_key(
    async_engine, document_factory, monkeypatch
):
    """A ``chat_file_<document_id>`` key retries as ``CHAT_UPLOAD``, never FULL."""
    maker = _session_maker(async_engine)
    doc_id = document_factory(is_chat_upload=True)
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        key = _chat_key(ws, doc_id)
        document = await db.get(Document, doc_id)
        document.upload_s3_key = key
        document.content_hash = "b" * 64
        await db.commit()

    published: dict = {}

    async def spy_publish_parse_task(
        *, document_id, workspace_id, minio_key, original_filename,
        revision_id, build_profile,
    ):
        published.update(
            revision_id=revision_id,
            build_profile=build_profile,
            minio_key=minio_key,
        )

    monkeypatch.setattr(
        "app.queue.publisher.publish_parse_task", spy_publish_parse_task
    )

    async with maker() as db:
        revision, profile = await allocate_commit_and_publish_retry(
            db,
            doc_id,
            workspace_id=ws,
            object_key=key,
            original_filename="orig.pdf",
            size_bytes=1024,
            content_sha256="b" * 64,
            etag="b" * 64,
        )

    assert profile is CHAT
    assert revision.revision_id is not None
    assert RevisionBuildProfile(published["build_profile"]) is CHAT


def test_retry_endpoints_queue_through_revision_allocator():
    """Every superadmin retry endpoint publishes a revision-aware payload.

    The endpoints live in ``app/api/workers.py`` (imports FastAPI, not
    available in this bench venv), so this asserts the source-level contract:
    each of the three recovery endpoints queues through the revision-aware
    helper, and no ``EXCHANGE_PARSE`` publish in the module is a raw
    revision-less dict.
    """
    source = (
        Path(__file__).resolve().parents[2] / "app" / "api" / "workers.py"
    ).read_text(encoding="utf-8")

    for endpoint in (
        "retry_all_failed",
        "retry_single_failed",
        "retry_stuck_documents",
    ):
        body = _function_source(source, endpoint)
        assert "_reset_and_queue_document_retry(" in body, endpoint
        assert "EXCHANGE_PARSE" not in body, endpoint

    offenders: list[str] = []
    for match in re.finditer(r"(?<![\w.])publish\(", source):
        call = _balanced_call(source, match.end() - 1)
        if "EXCHANGE_PARSE" in call and "revision_id" not in call:
            line_no = source.count("\n", 0, match.start()) + 1
            offenders.append(f"app/api/workers.py:{line_no}")
    assert not offenders, (
        "EXCHANGE_PARSE publish without revision_id: " + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Document mirror vs revision outcome — never INDEXED with a failed revision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_failure_mirrors_document_failed_not_indexed(
    async_engine, document_factory, monkeypatch
):
    """A revision terminalized by verify must NOT leave its Document INDEXED.

    The parse-only worker path used to set ``INDEXED`` unconditionally right
    after the finalize call, so a build whose artifacts were incomplete ended
    up reported indexed while its revision was terminal ``failed``.
    """
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="c" * 64,
            version_id="v-1",
            parse_only=True,
        )
        await db.commit()
        revision_id = revision.revision_id
    assert profile is PARSE_ONLY

    # No artifacts recorded → verify fails when completion is asserted.
    result = await finalize_revision_if_complete(revision_id, expect_complete=True)
    assert result.outcome is FinalizeOutcome.FAILED
    assert result.failure_stage == "verify"
    await apply_finalize_outcome(doc_id, result)

    async with maker() as db:
        document = await db.get(Document, doc_id)
        row = await db.get(DocumentRevision, revision_id)
    assert row.status == "failed"
    assert document.status == DocumentStatus.FAILED
    assert document.status != DocumentStatus.INDEXED
    assert "revision_failed" in document.error_message


@pytest.mark.asyncio
async def test_finalize_success_mirrors_document_indexed(
    async_engine, document_factory, monkeypatch
):
    """A published revision still promotes the Document mirror to INDEXED."""
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="d" * 64,
            version_id="v-1",
            parse_only=True,
        )
        await _record_complete_artifacts(db, revision, profile)
        await db.commit()
        revision_id = revision.revision_id

    result = await finalize_revision_if_complete(revision_id, expect_complete=True)
    assert result.outcome is FinalizeOutcome.PUBLISHED
    await apply_finalize_outcome(doc_id, result)

    async with maker() as db:
        document = await db.get(Document, doc_id)
        row = await db.get(DocumentRevision, revision_id)
    assert row.status == "published"
    assert document.status == DocumentStatus.INDEXED
    assert document.current_revision_id == revision_id


@pytest.mark.asyncio
async def test_check_and_finalize_never_indexes_a_failed_revision(
    async_engine, document_factory, monkeypatch
):
    """The multi-stage choke point commits INDEXED only for a published revision.

    The revision's required stages complete (chat upload: parse + embed)
    while the revision has no vector manifest, so ``verify_draft`` fails.
    The Document must end up FAILED, never INDEXED. (P1 Task 4: stage rows
    authorize finalization — the mirror ``embed_done`` flag alone no longer
    reaches ``finalize_revision_if_complete``.)
    """
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    doc_id = document_factory(is_chat_upload=True)
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        document = await db.get(Document, doc_id)
        document.embed_done = True
        revision, profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_chat_key(ws, doc_id),
            size_bytes=11,
            content_sha256="e" * 64,
            version_id="v-1",
        )
        await _task4_complete_stages(
            db, revision.revision_id, ("parse", "embed")
        )
        await db.commit()
        revision_id = revision.revision_id
    assert profile is CHAT

    async with maker() as db:
        document = await db.get(Document, doc_id)
        await check_and_finalize(document, db, revision_id=revision_id)

    async with maker() as db:
        fresh = await db.get(Document, doc_id)
        row = await db.get(DocumentRevision, revision_id)
    assert row.status == "failed"
    assert fresh.status == DocumentStatus.FAILED
    assert fresh.status != DocumentStatus.INDEXED


@pytest.mark.asyncio
async def test_check_and_finalize_indexes_a_published_revision(
    async_engine, document_factory, monkeypatch
):
    """The success path is unchanged: stages complete + publishable → INDEXED."""
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    doc_id = document_factory(is_chat_upload=True)
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        document = await db.get(Document, doc_id)
        document.embed_done = True
        revision, profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_chat_key(ws, doc_id),
            size_bytes=11,
            content_sha256="f" * 64,
            version_id="v-1",
        )
        await _task4_complete_stages(
            db, revision.revision_id, ("parse", "embed")
        )
        await _record_complete_artifacts(db, revision, profile)
        await db.commit()
        revision_id = revision.revision_id

    async with maker() as db:
        document = await db.get(Document, doc_id)
        await check_and_finalize(document, db, revision_id=revision_id)

    async with maker() as db:
        fresh = await db.get(Document, doc_id)
        row = await db.get(DocumentRevision, revision_id)
    assert row.status == "published"
    assert fresh.status == DocumentStatus.INDEXED
    assert fresh.current_revision_id == revision_id


@pytest.mark.asyncio
async def test_stale_revision_failure_does_not_fail_a_live_document(
    async_engine, document_factory, monkeypatch
):
    """A superseded revision's late failure must not touch the live mirror.

    The document's current pointer names a newer published revision; the older
    revision then terminalizes ``failed`` (e.g. its own finalize finds the
    artifact set incomplete). Without the generation guard the mirror would be
    rewritten to FAILED, permanently reporting a healthy, indexed document as
    failed.
    """
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        stale, _stale_profile = await allocate_reindex_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="1" * 64,
            version_id="v-1",
        )
        current, current_profile = await allocate_reindex_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="1" * 64,
            version_id="v-2",
        )
        await _record_complete_artifacts(db, current, current_profile)
        await db.commit()
        stale_id = stale.revision_id
        current_id = current.revision_id

    # The newer revision publishes and becomes the document's current pointer.
    result = await finalize_revision_if_complete(current_id, expect_complete=True)
    assert result.outcome is FinalizeOutcome.PUBLISHED
    await apply_finalize_outcome(doc_id, result, revision_id=current_id)

    async with maker() as db:
        document = await db.get(Document, doc_id)
    assert document.status == DocumentStatus.INDEXED
    assert document.current_revision_id == current_id

    # The stale revision's late finalize failure arrives afterwards.
    stale_result = await finalize_revision_if_complete(
        stale_id, expect_complete=True
    )
    assert stale_result.outcome is FinalizeOutcome.FAILED
    await apply_finalize_outcome(doc_id, stale_result, revision_id=stale_id)

    async with maker() as db:
        fresh = await db.get(Document, doc_id)
        stale_row = await db.get(DocumentRevision, stale_id)
    assert stale_row.status == "failed"
    assert fresh.status == DocumentStatus.INDEXED
    assert fresh.status != DocumentStatus.FAILED


# ---------------------------------------------------------------------------
# P1 Task 2 — stage rows are initialized atomically on allocation
# ---------------------------------------------------------------------------


async def _stage_states(db, revision_id) -> dict:
    """Stage -> state mapping for one revision (read-only)."""
    repo = DocumentRevisionsRepository(db)
    return {row.stage: row.state for row in await repo.get_stages(revision_id)}


@pytest.mark.asyncio
async def test_ingest_allocation_initializes_full_stage_rows(
    async_db, document_factory
):
    """A FULL ingest allocation owns four ``pending`` stage rows."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    revision, profile, created = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="7" * 64,
        version_id="v-7",
    )
    assert created is True
    assert profile is FULL
    assert await _stage_states(async_db, revision.revision_id) == {
        "parse": "pending",
        "embed": "pending",
        "caption": "pending",
        "kg": "pending",
    }
    repo = DocumentRevisionsRepository(async_db)
    assert await repo.required_stages_complete(revision.revision_id) is False


@pytest.mark.asyncio
async def test_ingest_allocation_initializes_chat_upload_stage_rows(
    async_db, document_factory
):
    """A CHAT_UPLOAD ingest allocation pends parse/embed and skips caption/kg."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _chat_key(ws, doc_id)

    revision, profile, _created = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=7,
        content_sha256="8" * 64,
        version_id="v-8",
    )
    assert profile is CHAT
    assert await _stage_states(async_db, revision.revision_id) == {
        "parse": "pending",
        "embed": "pending",
        "caption": "skipped",
        "kg": "skipped",
    }
    repo = DocumentRevisionsRepository(async_db)
    assert await repo.required_stages_complete(revision.revision_id) is False
    await repo.mark_stage_completed(revision.revision_id, "parse")
    await repo.mark_stage_completed(revision.revision_id, "embed")
    assert await repo.required_stages_complete(revision.revision_id) is True


@pytest.mark.asyncio
async def test_ingest_allocation_initializes_parse_only_stage_rows(
    async_db, document_factory
):
    """A PARSE_ONLY ingest allocation pends only parse; the rest are skipped."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    revision, profile, _created = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=3,
        content_sha256="9" * 64,
        version_id="v-9",
        parse_only=True,
    )
    assert profile is PARSE_ONLY
    assert await _stage_states(async_db, revision.revision_id) == {
        "parse": "pending",
        "embed": "skipped",
        "caption": "skipped",
        "kg": "skipped",
    }
    repo = DocumentRevisionsRepository(async_db)
    await repo.mark_stage_completed(revision.revision_id, "parse")
    assert await repo.required_stages_complete(revision.revision_id) is True


@pytest.mark.asyncio
async def test_reindex_allocation_initializes_stage_rows(async_db, document_factory):
    """A reindex generation starts with its own four ``pending`` stage rows."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    first, _created = await _allocate_full(async_db, doc_id, key=key, sha="a" * 64)
    await _build_and_publish(async_db, first, FULL)

    second, profile = await allocate_reindex_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="a" * 64,
        version_id="v-1",
        reindex_of_revision_id=first.revision_id,
    )
    assert profile is FULL
    assert await _stage_states(async_db, second.revision_id) == {
        "parse": "pending",
        "embed": "pending",
        "caption": "pending",
        "kg": "pending",
    }
    # The new rows belong only to the new generation; the older generation's
    # ownership is asserted precisely by
    # ``test_reindex_stage_rows_are_owned_by_new_revision_only`` below.
    repo = DocumentRevisionsRepository(async_db)
    assert await repo.required_stages_complete(second.revision_id) is False


@pytest.mark.asyncio
async def test_clone_allocation_initializes_stage_rows(async_db, document_factory):
    """A clone allocation initializes stage rows on the TARGET revision."""
    source_doc = document_factory()
    target_doc = document_factory()
    ws_src = await _workspace_id(async_db, source_doc)
    ws_tgt = await _workspace_id(async_db, target_doc)

    source_rev, _created = await _allocate_full(
        async_db, source_doc, key=_doc_key(ws_src, source_doc), sha="b" * 64
    )
    await _build_and_publish(async_db, source_rev, FULL)

    target = await async_db.get(Document, target_doc)
    target.upload_s3_key = _doc_key(ws_tgt, target_doc)
    target.content_hash = "b" * 64
    target.file_size = 11
    await async_db.flush()

    clone_rev, profile = await allocate_clone_revision(
        async_db,
        target_doc,
        object_key=target.upload_s3_key,
        size_bytes=11,
        content_sha256="b" * 64,
        etag="b" * 32,
        cloned_from_revision_id=source_rev.revision_id,
    )
    assert profile is FULL
    assert clone_rev.document_id == target_doc
    assert await _stage_states(async_db, clone_rev.revision_id) == {
        "parse": "pending",
        "embed": "pending",
        "caption": "pending",
        "kg": "pending",
    }


@pytest.mark.asyncio
async def test_duplicate_ingest_allocation_preserves_progressed_stages(
    async_db, document_factory
):
    """A redelivered ingest event converges WITHOUT resetting stage progress.

    Resetting stages at allocation would wipe a ``completed`` mark the first
    delivery already recorded; ``initialize_stages`` is ``ON CONFLICT DO
    NOTHING`` so the duplicate converges on the same rows untouched.
    """
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    first, _profile, created = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="c" * 64,
        version_id="v-1",
    )
    assert created is True
    repo = DocumentRevisionsRepository(async_db)
    await repo.mark_stage_completed(first.revision_id, "parse")

    redelivered, _profile2, created_again = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="c" * 64,
        version_id="v-1",
    )
    assert created_again is False
    assert redelivered.revision_id == first.revision_id
    assert await _stage_states(async_db, first.revision_id) == {
        "parse": "completed",
        "embed": "pending",
        "caption": "pending",
        "kg": "pending",
    }
    assert len(await repo.get_stages(first.revision_id)) == 4


@pytest.mark.asyncio
async def test_concurrent_ingest_allocation_converges_with_one_stage_set(
    async_engine, document_factory
):
    """Two racing ingest allocations converge on one revision + one stage set."""
    maker = _session_maker(async_engine)
    doc_id = document_factory()
    async with maker() as probe:
        ws = await _workspace_id(probe, doc_id)
    key = _doc_key(ws, doc_id)
    results: dict = {}

    async def _allocate(tag: str) -> None:
        async with maker() as db:
            revision, _profile, created = await allocate_ingest_revision(
                db,
                doc_id,
                object_key=key,
                size_bytes=11,
                content_sha256="d" * 64,
                version_id="v-1",
            )
            await db.commit()
            results[tag] = (revision.revision_id, created)

    await asyncio.gather(_allocate("a"), _allocate("b"))

    assert results["a"][0] == results["b"][0]
    assert sorted([results["a"][1], results["b"][1]]) == [False, True]
    async with maker() as db:
        assert await _stage_states(db, results["a"][0]) == {
            "parse": "pending",
            "embed": "pending",
            "caption": "pending",
            "kg": "pending",
        }
        repo = DocumentRevisionsRepository(db)
        assert len(await repo.get_stages(results["a"][0])) == 4


@pytest.mark.asyncio
async def test_reindex_stage_rows_are_owned_by_new_revision_only(
    async_db, document_factory
):
    """Completing R1's stages then reindexing leaves R1 done and R2 pending."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    first, _created = await _allocate_full(async_db, doc_id, key=key, sha="e" * 64)
    repo = DocumentRevisionsRepository(async_db)
    for stage in ("parse", "embed", "caption", "kg"):
        await repo.mark_stage_completed(first.revision_id, stage)
    assert await repo.required_stages_complete(first.revision_id) is True

    second, _profile = await allocate_reindex_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="e" * 64,
        version_id="v-1",
        reindex_of_revision_id=first.revision_id,
    )
    assert await _stage_states(async_db, second.revision_id) == {
        "parse": "pending",
        "embed": "pending",
        "caption": "pending",
        "kg": "pending",
    }
    # The older generation is untouched by the new allocation.
    assert await _stage_states(async_db, first.revision_id) == {
        "parse": "completed",
        "embed": "completed",
        "caption": "completed",
        "kg": "completed",
    }


@pytest.mark.asyncio
async def test_stale_mirror_flags_neither_satisfy_nor_block_stage_gate(
    async_db, document_factory
):
    """``Document.*_done`` mirrors never satisfy ``required_stages_complete``.

    Stale ``True`` mirrors from the previous generation must not complete the
    new generation's gate, and ``False`` mirrors must not block a genuinely
    complete revision — the gate reads only revision-owned stage rows.
    """
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)

    document = await async_db.get(Document, doc_id)
    document.embed_done = True
    document.captions_done = True
    document.kg_done = True
    await async_db.flush()

    revision, _profile, _created = await allocate_ingest_revision(
        async_db,
        doc_id,
        object_key=key,
        size_bytes=11,
        content_sha256="f" * 64,
        version_id="v-1",
    )
    repo = DocumentRevisionsRepository(async_db)
    assert await repo.required_stages_complete(revision.revision_id) is False

    document.embed_done = False
    document.captions_done = False
    document.kg_done = False
    await async_db.flush()
    for stage in ("parse", "embed", "caption", "kg"):
        await repo.mark_stage_completed(revision.revision_id, stage)
    assert await repo.required_stages_complete(revision.revision_id) is True


@pytest.mark.asyncio
async def test_allocation_rollback_removes_revision_and_stage_rows(
    async_engine, document_factory
):
    """Stage rows are allocated in the revision's own transaction: a rollback
    removes both, so a failed allocation never leaves orphaned stage rows."""
    maker = _session_maker(async_engine)
    doc_id = document_factory()
    async with maker() as probe:
        ws = await _workspace_id(probe, doc_id)
    key = _doc_key(ws, doc_id)

    async with maker() as db:
        revision, _profile = await allocate_reindex_revision(
            db,
            doc_id,
            object_key=key,
            size_bytes=11,
            content_sha256="1" * 64,
            version_id="v-1",
        )
        revision_id = revision.revision_id
        repo = DocumentRevisionsRepository(db)
        assert len(await repo.get_stages(revision_id)) == 4
        await db.rollback()

    async with maker() as db:
        assert await db.get(DocumentRevision, revision_id) is None
        repo = DocumentRevisionsRepository(db)
        assert await repo.get_stages(revision_id) == []


@pytest.mark.asyncio
async def test_ingest_stages_committed_before_parse_publish(
    async_engine, document_factory, monkeypatch
):
    """``allocate_commit_and_publish_parse`` commits stage rows before publish.

    The worker loads the revision from its own connection; stage rows that
    are merely flushed would vanish with the request unit-of-work and the
    worker would fail closed on ``UnknownRevisionStage``.
    """
    maker = _session_maker(async_engine)
    doc_id = document_factory()
    async with maker() as probe:
        ws = await _workspace_id(probe, doc_id)
    key = _doc_key(ws, doc_id)
    observed: dict = {}

    async def spy_publish_parse_task(
        *, document_id, workspace_id, minio_key, original_filename,
        revision_id, build_profile,
    ):
        async with maker() as observer:
            observed["revision"] = await observer.get(DocumentRevision, revision_id)
            repo = DocumentRevisionsRepository(observer)
            observed["stages"] = {
                row.stage: row.state for row in await repo.get_stages(revision_id)
            }

    monkeypatch.setattr(
        "app.queue.publisher.publish_parse_task", spy_publish_parse_task
    )

    async with maker() as db:
        revision, _profile, _created = await allocate_commit_and_publish_parse(
            db,
            doc_id,
            workspace_id=ws,
            object_key=key,
            original_filename="doc.pdf",
            size_bytes=11,
            content_sha256="2" * 64,
            version_id="v-1",
        )

    assert observed.get("revision") is not None
    assert observed["revision"].revision_id == revision.revision_id
    assert observed["stages"] == {
        "parse": "pending",
        "embed": "pending",
        "caption": "pending",
        "kg": "pending",
    }


@pytest.mark.asyncio
async def test_operator_retry_allocation_initializes_stage_rows(
    async_engine, document_factory, monkeypatch
):
    """An operator retry generation owns fresh stage rows, committed pre-publish."""
    maker = _session_maker(async_engine)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        key = _doc_key(ws, doc_id)
        document = await db.get(Document, doc_id)
        document.upload_s3_key = key
        document.content_hash = "3" * 64
        prior, _created = await _allocate_full(db, doc_id, key=key, sha="3" * 64)
        await _build_and_publish(db, prior, FULL)
        await db.commit()
        prior_id = prior.revision_id
        prior_generation = prior.generation

    async def spy_publish_parse_task(
        *, document_id, workspace_id, minio_key, original_filename,
        revision_id, build_profile,
    ):
        pass

    monkeypatch.setattr(
        "app.queue.publisher.publish_parse_task", spy_publish_parse_task
    )

    async with maker() as db:
        revision, profile = await allocate_commit_and_publish_retry(
            db,
            doc_id,
            workspace_id=ws,
            object_key=key,
            original_filename="orig.pdf",
            size_bytes=1024,
            content_sha256="3" * 64,
            etag="3" * 64,
            previous_revision_id=prior_id,
        )
        assert profile is FULL
        assert revision.generation > prior_generation
        assert revision.reindex_of_revision_id == prior_id

    async with maker() as db:
        assert await _stage_states(db, revision.revision_id) == {
            "parse": "pending",
            "embed": "pending",
            "caption": "pending",
            "kg": "pending",
        }
        repo = DocumentRevisionsRepository(db)
        assert await repo.required_stages_complete(revision.revision_id) is False


# ---------------------------------------------------------------------------
# P1 Task 3 — workers report revision-owned stage completion
# ---------------------------------------------------------------------------

_TASK3_STAGES = ("parse", "embed", "caption", "kg")

_WORKER_STAGE_FILES = {
    "parse": "app/workers/parse_worker.py",
    "embed": "app/workers/embed_worker.py",
    "caption": "app/workers/caption_worker.py",
    "kg": "app/workers/kg_worker.py",
}


def _task3_worker_source(stage: str) -> str:
    backend_root = Path(__file__).resolve().parents[2]
    return (backend_root / _WORKER_STAGE_FILES[stage]).read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", _TASK3_STAGES)
async def test_task3_stage_running_before_work_completed_after_artifact(
    async_db, document_factory, stage
):
    """Each message revision/stage goes pending -> running before work.

    The running mark lands (attempt 1) while the gate is still unsatisfied;
    completion is recorded only after that stage's artifact transaction
    succeeds. Profile skips stay initialized rows — this test never skips.
    """
    from app.workers.utils import record_embed_artifacts, record_parse_artifacts

    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _created = await _allocate_full(async_db, doc_id, key=key, sha="a" * 64)
    repo = DocumentRevisionsRepository(async_db)

    assert (await _stage_states(async_db, revision.revision_id))[stage] == "pending"
    running = await repo.mark_stage_running(revision.revision_id, stage)
    assert running.state == "running"
    assert running.attempt_count == 1
    assert await repo.required_stages_complete(revision.revision_id) is False

    if stage == "parse":
        await record_parse_artifacts(
            async_db,
            revision.revision_id,
            FULL,
            markdown_artifact_key="kb_x/doc.md",
            structure_artifact_key="kb_x/doc.structure",
        )
    elif stage == "embed":
        await record_embed_artifacts(
            async_db,
            revision.revision_id,
            FULL,
            embedding_namespace="ws_test",
            embedding_model_hash="hash",
            embedding_dimension=1024,
            vector_artifact_version="v1",
        )
    # caption/kg stages commit mirror rows (no manifest recorder); the
    # completion below models the post-commit report, not a skip guess.
    completed = await repo.mark_stage_completed(revision.revision_id, stage)
    assert completed.state == "completed"
    assert completed.attempt_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", _TASK3_STAGES)
async def test_task3_duplicate_running_and_completion_converge(
    async_db, document_factory, stage
):
    """Redelivered running/completion marks converge without side effects."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _created = await _allocate_full(async_db, doc_id, key=key, sha="b" * 64)
    repo = DocumentRevisionsRepository(async_db)

    first = await repo.mark_stage_running(revision.revision_id, stage)
    second = await repo.mark_stage_running(revision.revision_id, stage)
    assert (first.state, second.state) == ("running", "running")
    assert second.attempt_count == 1

    done_once = await repo.mark_stage_completed(revision.revision_id, stage)
    done_twice = await repo.mark_stage_completed(revision.revision_id, stage)
    assert (done_once.state, done_twice.state) == ("completed", "completed")
    assert done_twice.attempt_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", _TASK3_STAGES)
async def test_task3_retry_edge_preserves_attempt_then_bumps(
    async_db, document_factory, stage
):
    """running -> pending preserves the attempt; next running bumps it."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _created = await _allocate_full(async_db, doc_id, key=key, sha="c" * 64)
    repo = DocumentRevisionsRepository(async_db)

    await repo.mark_stage_running(revision.revision_id, stage)
    pending = await repo.mark_stage_retry_pending(revision.revision_id, stage)
    assert pending.state == "pending"
    assert pending.attempt_count == 1
    rerun = await repo.mark_stage_running(revision.revision_id, stage)
    assert rerun.state == "running"
    assert rerun.attempt_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", _TASK3_STAGES)
async def test_task3_exhausted_stage_failed_is_terminal(
    async_db, document_factory, stage
):
    """Exhausted retries mark that stage failed; terminal stages never move."""
    from app.services.agents.v2.persistence.document_revisions import (
        InvalidStageTransition,
    )

    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _created = await _allocate_full(async_db, doc_id, key=key, sha="d" * 64)
    repo = DocumentRevisionsRepository(async_db)

    await repo.mark_stage_running(revision.revision_id, stage)
    failed = await repo.mark_stage_failed(
        revision.revision_id, stage, failure_class="TimeoutError"
    )
    assert failed.state == "failed"
    assert failed.failure_class == "TimeoutError"
    for coro in (
        repo.mark_stage_running(revision.revision_id, stage),
        repo.mark_stage_retry_pending(revision.revision_id, stage),
        repo.mark_stage_completed(revision.revision_id, stage),
    ):
        with pytest.raises(InvalidStageTransition):
            await coro
    assert (await _stage_states(async_db, revision.revision_id))[stage] == "failed"


@pytest.mark.asyncio
async def test_task3_stale_revision_completion_isolated_to_own_generation(
    async_db, document_factory
):
    """A late R1 completion cannot touch the newer R2 generation's rows."""
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    r1, _ = await _allocate_full(async_db, doc_id, key=key, sha="e" * 64)
    r2, _created = await _allocate_full(async_db, doc_id, key=key, sha="f" * 64)
    assert r2.generation > r1.generation
    repo = DocumentRevisionsRepository(async_db)

    for stage in _TASK3_STAGES:
        await repo.mark_stage_running(r1.revision_id, stage)
        await repo.mark_stage_completed(r1.revision_id, stage)
    assert await repo.required_stages_complete(r1.revision_id) is True
    assert await _stage_states(async_db, r2.revision_id) == {
        "parse": "pending",
        "embed": "pending",
        "caption": "pending",
        "kg": "pending",
    }
    assert await repo.required_stages_complete(r2.revision_id) is False


def test_task3_queue_stage_mapping_covers_all_pipeline_queues():
    """The queue retry path maps every pipeline queue to exactly one stage."""
    from app.queue.connection import stage_for_queue

    assert stage_for_queue("hrag.parse", "hrag.parse") == "parse"
    assert stage_for_queue("hrag.embed", "hrag.embed") == "embed"
    assert stage_for_queue("hrag.caption", "hrag.caption") == "caption"
    assert stage_for_queue("hrag.kg.00000000-0000-0000-0000-000000000000", "hrag.kg") == "kg"
    assert stage_for_queue("hrag.memory", "hrag.memory") is None


@pytest.mark.asyncio
async def test_task3_queue_retry_note_touches_only_failed_message_stage(
    async_db, document_factory
):
    """Retry requeue marks pending ONLY the failed message revision/stage."""
    from app.queue.connection import note_stage_retry_pending

    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _created = await _allocate_full(async_db, doc_id, key=key, sha="g" * 64)
    repo = DocumentRevisionsRepository(async_db)
    await repo.mark_stage_running(revision.revision_id, "embed")

    assert await note_stage_retry_pending(
        revision.revision_id, "embed", db=async_db
    ) is True
    assert await _stage_states(async_db, revision.revision_id) == {
        "parse": "pending",
        "embed": "pending",
        "caption": "pending",
        "kg": "pending",
    }
    rows = await repo.get_stages(revision.revision_id)
    assert {r.stage: r.attempt_count for r in rows}["embed"] == 1
    # Unknown revisions and already-terminal stages are left untouched,
    # never raised through the retry path.
    assert await note_stage_retry_pending(uuid.uuid4(), "embed", db=async_db) is False
    await repo.mark_stage_completed(revision.revision_id, "parse")
    assert await note_stage_retry_pending(
        revision.revision_id, "parse", db=async_db
    ) is False
    assert (await _stage_states(async_db, revision.revision_id))["parse"] == "completed"


@pytest.mark.asyncio
async def test_task3_queue_exhausted_note_marks_stage_failed_terminal(
    async_db, document_factory
):
    """Exhausted queue retries mark that stage failed and stay terminal."""
    from app.queue.connection import note_stage_exhausted
    from app.services.agents.v2.persistence.document_revisions import (
        InvalidStageTransition,
    )

    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _created = await _allocate_full(async_db, doc_id, key=key, sha="h" * 64)
    repo = DocumentRevisionsRepository(async_db)
    await repo.mark_stage_running(revision.revision_id, "kg")

    assert await note_stage_exhausted(
        revision.revision_id, "kg", failure_class="TimeoutError", db=async_db
    ) is True
    assert (await _stage_states(async_db, revision.revision_id))["kg"] == "failed"
    # Exhaustion terminalizes the revision in the same transaction.
    failed_rev = await repo.get(revision.revision_id)
    assert failed_rev.status == "failed"
    assert failed_rev.failure_stage == "kg"
    assert failed_rev.failure_class == "TimeoutError"
    with pytest.raises(InvalidStageTransition):
        await repo.mark_stage_completed(revision.revision_id, "kg")
    # A completed stage is immutable even to the exhausted path.
    await repo.mark_stage_completed(revision.revision_id, "parse")
    assert await note_stage_exhausted(
        revision.revision_id, "parse", failure_class="ValueError", db=async_db
    ) is False
    assert (await _stage_states(async_db, revision.revision_id))["parse"] == "completed"


@pytest.mark.parametrize("stage", _TASK3_STAGES)
def test_task3_workers_report_only_their_own_stage(stage):
    """Each worker claims/completes exactly its message stage.

    The running edge is owned by the atomic ``claim_stage_running`` claim
    (which returns whether this delivery newly claimed pending → running);
    workers never call ``mark_stage_running`` directly, so a swallowed
    transition can no longer fall through into heavy work. Handler-level
    tests below drive the real handlers — this pins the call sites.
    """
    source = _task3_worker_source(stage)
    # The claim call spans lines; normalize whitespace before matching.
    flat = re.sub(r"\s+", "", source)
    assert f'claim_stage_running(msg.revision_id,"{stage}")' in flat
    assert f'mark_stage_completed(msg.revision_id,"{stage}")' in flat
    for other in _TASK3_STAGES:
        if other == stage:
            continue
        assert f'claim_stage_running(msg.revision_id,"{other}")' not in flat
        assert f'mark_stage_completed(msg.revision_id,"{other}")' not in flat
    assert "mark_stage_running(" not in flat
    # Skips are initialized rows (Task 2), never guessed by workers; workers
    # never fail a stage directly — exhaustion is owned by the queue path.
    assert "mark_stage_skipped" not in source
    assert "mark_stage_failed" not in source


@pytest.mark.parametrize(
    "stage,artifact_anchor",
    [
        ("parse", "record_parse_artifacts"),
        ("embed", "record_embed_artifacts"),
        ("caption", "captions_done = True"),
        ("kg", "kg_done = True"),
    ],
)
def test_task3_worker_completed_only_after_artifact_transaction(
    stage, artifact_anchor
):
    """Running lands before work; completed lands after the artifact commit."""
    source = _task3_worker_source(stage)
    flat = re.sub(r"\s+", "", source)
    running_call = f'claim_stage_running(msg.revision_id,"{stage}")'
    source = flat
    completed_call = f'mark_stage_completed(msg.revision_id,"{stage}")'
    running_at = source.index(running_call)
    first_completed_at = source.index(completed_call)
    last_completed_at = source.rindex(completed_call)
    # ``await``-prefixed call anchors match the execution/call sites, never
    # the module import block; mirror-assignment anchors match as written.
    # All needles are whitespace-normalized to match the flattened source.
    if "=" in artifact_anchor:
        # Mirror writes: scope to the Document assignment so the module
        # docstring ("Set captions_done=True") never matches.
        await_anchor = "document." + re.sub(r"\s+", "", artifact_anchor)
    else:
        await_anchor = f"await{artifact_anchor}"
    anchor_at = source.index(await_anchor)
    gate_at = source.index("awaitload_revision_execution(")
    # Running lands after the execution gate and before any work; the main
    # artifact path completes after its transaction (early-return paths
    # complete after their own mirror commits, hence the last-site bound).
    assert gate_at < running_at < anchor_at
    assert running_at < first_completed_at
    assert anchor_at < last_completed_at


def test_task3_queue_retry_branch_reports_stage_before_requeue():
    """Retry requeues report the failed revision/stage; exhaustion fails it."""
    backend_root = Path(__file__).resolve().parents[2]
    source = (backend_root / "app/queue/connection.py").read_text(encoding="utf-8")
    assert "note_stage_retry_pending" in source
    assert "note_stage_exhausted" in source
    assert "stage_for_queue" in source


# ---------------------------------------------------------------------------
# Task 3 fix-1 — handler-level claim/order/retry/exhaustion tests
# ---------------------------------------------------------------------------
# Source-string assertions above pin the call sites; the tests below drive
# the REAL handlers (heavy work stubbed, DB real) so a behavioral mutation
# — swallowed transition that keeps working, completion before the artifact
# commit, a wrong queue→stage map, a worker-owned revision failure — fails.
# Each handler gets: claim → work → artifact → completed ordering, duplicate
# + terminal (completed/failed) no-op without work or mirror mutation, retry
# re-claim with attempt bump, and stale-revision isolation.

from app.models.document_revision_build import DocumentRevisionBuild
from app.queue import connection as _conn
from app.workers import caption_worker as _caption_worker
from app.workers import embed_worker as _embed_worker
from app.workers import kg_worker as _kg_worker
from app.workers import parse_worker as _parse_worker


def _handler_maker(async_engine):
    return async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False,
        autocommit=False,
    )


async def _handler_setup_full(maker, doc_id, *, sha):
    """Allocate a FULL revision with initialized stages; commit for handler."""
    async with maker() as db:
        ws = await db.scalar(
            select(Document.workspace_id).where(Document.id == doc_id)
        )
        key = _doc_key(ws, doc_id)
        revision, _created = await _allocate_full(db, doc_id, key=key, sha=sha)
        await db.commit()
        return ws, revision.revision_id


async def _handler_stage_rows(maker, revision_id):
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        rows = await repo.get_stages(revision_id)
        return {r.stage: (r.state, r.attempt_count) for r in rows}


async def _handler_mirrors(maker, doc_id):
    async with maker() as db:
        doc = await db.get(Document, doc_id)
        return (
            doc.status, doc.embed_done, doc.captions_done, doc.kg_done,
            doc.markdown_s3_key, doc.error_message,
        )


async def _handler_revision_status(maker, revision_id):
    async with maker() as db:
        return (await DocumentRevisionsRepository(db).get(revision_id)).status


def _patch_config_watch(monkeypatch):
    import app.workers.config_watch as config_watch

    async def _fresh():
        return None

    monkeypatch.setattr(config_watch, "ensure_fresh_config", _fresh)


def _patch_finalize_recorder(monkeypatch, module):
    calls = []

    async def _record(document, db, *, revision_id=None):
        calls.append(revision_id)

    monkeypatch.setattr(module, "check_and_finalize", _record)
    return calls


# -- parse stubs ------------------------------------------------------------

class _FakeParseChunk:
    def __init__(self, content="hello world"):
        self.content = content
        self.chunk_index = 0
        self.page_no = 1
        self.heading_path = []
        self.source_file = "doc.pdf"
        self.image_refs = []
        self.table_refs = []
        self.has_table = False
        self.has_code = False


class _FakeParsedDoc:
    def __init__(self):
        self.markdown = "# Title\n\nBody text."
        self.page_count = 1
        self.tables_count = 0
        self.parser = "ocr"  # skips the page-1 re-OCR block
        self.chunks = [_FakeParseChunk()]
        self.images = []
        self.tables = []


class _FakeParseParser:
    def __init__(self, workspace_id=None):
        pass

    async def parse_structure(self, **kwargs):
        return _FakeParsedDoc()


class _RecordingParseStore:
    def __init__(self, fail_download=False):
        self.downloads = []
        self.uploads = {}
        self.artifacts = {}
        self.fail_download = fail_download

    async def download_file(self, key):
        self.downloads.append(key)
        if self.fail_download:
            raise RuntimeError("minio boom")
        return b"fake-bytes"

    async def upload_markdown(self, *, workspace_id, document_id, content, key=None):
        self.uploads[key] = content
        return key

    async def upload_artifact(self, key, payload, content_type):
        self.artifacts[key] = payload
        return key

    async def download_markdown(self, key):
        return "# Title\n\nBody text for the knowledge graph."


def _patch_parse(monkeypatch, maker, store):
    monkeypatch.setattr(_parse_worker, "async_session_maker", maker)
    monkeypatch.setattr(_parse_worker, "get_storage_service", lambda: store)
    monkeypatch.setattr(_parse_worker, "DeepDocumentParser", _FakeParseParser)
    import app.services.document_type_classifier as _dtc
    import app.services.legal.validity_service as _val

    async def _classify(text):
        return {}

    async def _validity(db, document, markdown):
        return None

    monkeypatch.setattr(_dtc, "classify_with_llm", _classify)
    monkeypatch.setattr(_val, "apply_validity", _validity)
    published = []

    async def _publish(exchange, routing_key, payload):
        published.append((exchange, routing_key))

    async def _ensure_kg(*args, **kwargs):
        return None

    monkeypatch.setattr(_conn, "publish", _publish)
    monkeypatch.setattr(_conn, "ensure_kg_queue", _ensure_kg)
    return published


def _parse_payload(doc_id, ws, rev_id):
    return ParseMessage(
        document_id=doc_id,
        workspace_id=ws,
        revision_id=rev_id,
        build_profile=FULL.value,
        minio_key="kb_x/doc.txt",
        original_filename="doc.txt",
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_task3_handle_parse_claims_runs_completes(
    async_engine, document_factory, monkeypatch
):
    """Real parse handler: claim → work → artifact commit → completed."""
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="p" * 64)
    store = _RecordingParseStore()
    published = _patch_parse(monkeypatch, maker, store)

    await _parse_worker.handle_parse(_parse_payload(doc_id, ws, rev_id))

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["parse"] == ("completed", 1)
    assert rows["embed"] == ("pending", 0)
    assert rows["caption"] == ("pending", 0)
    assert rows["kg"] == ("pending", 0)
    # Artifact work actually ran before completion: markdown uploaded at the
    # revision key, structure artifact stored, manifest recorded.
    assert len(store.downloads) == 1
    assert len(store.uploads) == 1
    assert len(store.artifacts) == 1
    async with maker() as db:
        build = await db.scalar(
            select(DocumentRevisionBuild).where(
                DocumentRevisionBuild.revision_id == rev_id
            )
        )
        assert build is not None and build.markdown_artifact_key is not None
    mirrors = await _handler_mirrors(maker, doc_id)
    assert mirrors[4] is not None  # markdown_s3_key mirror written
    # FULL profile dispatched all three child stages.
    assert sorted(e for e, _ in published) == sorted(
        [_conn.EXCHANGE_EMBED, _conn.EXCHANGE_CAPTION, _conn.EXCHANGE_KG]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["running", "completed", "failed"])
async def test_task3_handle_parse_duplicate_and_terminal_noop(
    async_engine, document_factory, monkeypatch, preset
):
    """Duplicate/terminal parse delivery: no work, no mirror mutation."""
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="q" * 64)
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        if preset == "running":
            await repo.claim_stage_running(rev_id, "parse")
        elif preset == "completed":
            await repo.claim_stage_running(rev_id, "parse")
            await repo.mark_stage_completed(rev_id, "parse")
        else:
            await repo.claim_stage_running(rev_id, "parse")
            await repo.mark_stage_failed(rev_id, "parse", failure_class="E")
        await db.commit()
    before_rows = await _handler_stage_rows(maker, rev_id)
    before_mirrors = await _handler_mirrors(maker, doc_id)
    store = _RecordingParseStore()
    _patch_parse(monkeypatch, maker, store)

    await _parse_worker.handle_parse(_parse_payload(doc_id, ws, rev_id))

    assert store.downloads == [] and store.uploads == {} and store.artifacts == {}
    assert await _handler_stage_rows(maker, rev_id) == before_rows
    assert await _handler_mirrors(maker, doc_id) == before_mirrors
    assert await _handler_revision_status(maker, rev_id) == "draft"


@pytest.mark.asyncio
async def test_task3_handle_parse_retry_reclaims_attempt(
    async_engine, document_factory, monkeypatch
):
    """Queue retry edge keeps parse executable: next claim bumps to 2."""
    from app.queue.connection import note_stage_retry_pending

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="r" * 64)
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        _, claimed = await repo.claim_stage_running(rev_id, "parse")
        assert claimed is True
        await db.commit()
    async with maker() as db:
        assert await note_stage_retry_pending(rev_id, "parse", db=db) is True
        await db.commit()
    store = _RecordingParseStore()
    _patch_parse(monkeypatch, maker, store)

    await _parse_worker.handle_parse(_parse_payload(doc_id, ws, rev_id))

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["parse"] == ("completed", 2)
    assert len(store.downloads) == 1


@pytest.mark.asyncio
async def test_task3_handle_parse_failure_leaves_revision_live(
    async_engine, document_factory, monkeypatch
):
    """Retryable parse exception must NOT terminalize the revision (I3).

    The stage stays running, the revision stays draft, and after the queue
    retry edge the redelivery re-claims (attempt 2) and completes.
    """
    from app.queue.connection import note_stage_retry_pending

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="s" * 64)
    store = _RecordingParseStore(fail_download=True)
    _patch_parse(monkeypatch, maker, store)

    with pytest.raises(RuntimeError, match="minio boom"):
        await _parse_worker.handle_parse(_parse_payload(doc_id, ws, rev_id))

    # Live (draft/building), never terminalized: the queue owns exhaustion.
    assert await _handler_revision_status(maker, rev_id) in ("draft", "building")
    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["parse"] == ("running", 1)

    async with maker() as db:
        assert await note_stage_retry_pending(rev_id, "parse", db=db) is True
        await db.commit()
    store.fail_download = False
    await _parse_worker.handle_parse(_parse_payload(doc_id, ws, rev_id))

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["parse"] == ("completed", 2)


# -- embed stubs ------------------------------------------------------------

class _FakeEmbedder:
    def __init__(self):
        self.calls = []

    model_name = "fake-model"
    dimension = 4

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[0.1] * 4 for _ in texts]


class _FakeVectorStore:
    def __init__(self):
        self.collection_name = "ws_fake_collection"
        self.added = []

    def add_documents(self, *, ids, embeddings, documents, metadatas):
        self.added.append((list(ids), list(documents)))


_CHUNK = {
    "chunk_id": "c1",
    "content": "hello world",
    "chunk_index": 0,
    "source_file": "doc.pdf",
    "page_no": 1,
    "heading_path": [],
    "image_refs": [],
    "table_refs": [],
    "has_table": False,
    "has_code": False,
}


async def _seed_raw_chunks(maker, doc_id):
    import json as _json

    async with maker() as db:
        doc = await db.get(Document, doc_id)
        doc.raw_chunks_json = _json.dumps([dict(_CHUNK)])
        await db.commit()


def _patch_embed(monkeypatch, maker, embedder, store):
    monkeypatch.setattr(_embed_worker, "async_session_maker", maker)
    monkeypatch.setattr(
        _embed_worker, "get_embedding_service", lambda: embedder
    )
    monkeypatch.setattr(
        _embed_worker, "get_vector_store", lambda ws, namespace: store
    )
    _patch_config_watch(monkeypatch)


def _embed_payload(doc_id, ws, rev_id):
    return EmbedMessage(
        document_id=doc_id,
        workspace_id=ws,
        revision_id=rev_id,
        build_profile=FULL.value,
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_task3_handle_embed_claims_runs_completes(
    async_engine, document_factory, monkeypatch
):
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="t" * 64)
    await _seed_raw_chunks(maker, doc_id)
    embedder, store = _FakeEmbedder(), _FakeVectorStore()
    _patch_embed(monkeypatch, maker, embedder, store)
    finalize_calls = _patch_finalize_recorder(monkeypatch, _embed_worker)

    await _embed_worker.handle_embed(_embed_payload(doc_id, ws, rev_id))

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["embed"] == ("completed", 1)
    assert rows["parse"] == ("pending", 0)
    assert len(embedder.calls) == 1 and embedder.calls[0] == ["hello world"]
    assert len(store.added) == 1
    async with maker() as db:
        ns = await db.scalar(
            select(DocumentRevisionBuild.embedding_namespace).where(
                DocumentRevisionBuild.revision_id == rev_id
            )
        )
        assert ns == "ws_fake_collection"
    mirrors = await _handler_mirrors(maker, doc_id)
    assert mirrors[1] is True  # embed_done mirror
    assert finalize_calls == [rev_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["running", "completed", "failed"])
async def test_task3_handle_embed_duplicate_and_terminal_noop(
    async_engine, document_factory, monkeypatch, preset
):
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="u" * 64)
    await _seed_raw_chunks(maker, doc_id)
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        await repo.claim_stage_running(rev_id, "embed")
        if preset == "completed":
            await repo.mark_stage_completed(rev_id, "embed")
        elif preset == "failed":
            await repo.mark_stage_failed(rev_id, "embed", failure_class="E")
        await db.commit()
    before_rows = await _handler_stage_rows(maker, rev_id)
    before_mirrors = await _handler_mirrors(maker, doc_id)
    embedder, store = _FakeEmbedder(), _FakeVectorStore()
    _patch_embed(monkeypatch, maker, embedder, store)

    await _embed_worker.handle_embed(_embed_payload(doc_id, ws, rev_id))

    assert embedder.calls == [] and store.added == []
    assert await _handler_stage_rows(maker, rev_id) == before_rows
    assert await _handler_mirrors(maker, doc_id) == before_mirrors
    assert await _handler_revision_status(maker, rev_id) == "draft"


@pytest.mark.asyncio
async def test_task3_handle_embed_retry_reclaims_attempt(
    async_engine, document_factory, monkeypatch
):
    from app.queue.connection import note_stage_retry_pending

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="v" * 64)
    await _seed_raw_chunks(maker, doc_id)
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        _, claimed = await repo.claim_stage_running(rev_id, "embed")
        assert claimed is True
        await db.commit()
    async with maker() as db:
        assert await note_stage_retry_pending(rev_id, "embed", db=db) is True
        await db.commit()
    embedder, store = _FakeEmbedder(), _FakeVectorStore()
    _patch_embed(monkeypatch, maker, embedder, store)
    _patch_finalize_recorder(monkeypatch, _embed_worker)

    await _embed_worker.handle_embed(_embed_payload(doc_id, ws, rev_id))

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["embed"] == ("completed", 2)
    assert len(embedder.calls) == 1


@pytest.mark.asyncio
async def test_task3_handle_embed_failure_leaves_revision_live(
    async_engine, document_factory, monkeypatch
):
    """Retryable embed exception must NOT terminalize the revision (I3)."""
    from app.queue.connection import note_stage_retry_pending

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="w" * 64)
    await _seed_raw_chunks(maker, doc_id)

    class _BoomEmbedder(_FakeEmbedder):
        def embed_texts(self, texts):
            raise RuntimeError("chroma boom")

    _patch_embed(monkeypatch, maker, _BoomEmbedder(), _FakeVectorStore())
    _patch_finalize_recorder(monkeypatch, _embed_worker)

    with pytest.raises(RuntimeError, match="chroma boom"):
        await _embed_worker.handle_embed(_embed_payload(doc_id, ws, rev_id))

    # Live, never terminalized: the queue owns exhaustion.
    assert await _handler_revision_status(maker, rev_id) in ("draft", "building")
    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["embed"] == ("running", 1)

    async with maker() as db:
        assert await note_stage_retry_pending(rev_id, "embed", db=db) is True
        await db.commit()
    _patch_embed(monkeypatch, maker, _FakeEmbedder(), _FakeVectorStore())
    await _embed_worker.handle_embed(_embed_payload(doc_id, ws, rev_id))

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["embed"] == ("completed", 2)


# -- caption stubs ----------------------------------------------------------

class _RecordingCaptionStore:
    def __init__(self):
        self.downloads = []
        self.uploads = []

    async def download_markdown(self, key):
        self.downloads.append(key)
        return "# tài liệu"

    async def upload_markdown(self, *, workspace_id, document_id, content, key=None):
        self.uploads.append(key)
        return key


class _FakeCaptionParser:
    def __init__(self, workspace_id=None):
        pass

    def _inject_table_captions(self, markdown, tables):
        return markdown + "\n<!-- injected -->"


async def _seed_caption_table(maker, doc_id, rev_id):
    async with maker() as db:
        db.add(
            DocumentTable(
                document_id=doc_id,
                revision_id=rev_id,
                table_id="tbl-1",
                page_no=1,
                content_markdown="| a |",
                num_rows=1,
                num_cols=1,
            )
        )
        await db.commit()


def _patch_caption(monkeypatch, maker, store):
    from app.services.agents.v2.persistence import document_views as _views

    monkeypatch.setattr(_caption_worker, "async_session_maker", maker)

    async def _caption_tables(tables):
        for table in tables:
            table.caption = "bảng đã chú thích"

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(_caption_worker, "_caption_tables_concurrent", _caption_tables)
    monkeypatch.setattr(_caption_worker, "_reenrich_embeddings", _noop)
    monkeypatch.setattr(_caption_worker, "get_storage_service", lambda: store)
    monkeypatch.setattr(_caption_worker, "DeepDocumentParser", _FakeCaptionParser)
    monkeypatch.setattr(
        _caption_worker.settings, "HRAG_ENABLE_TABLE_CAPTIONING", True
    )
    monkeypatch.setattr(
        _caption_worker.settings, "HRAG_ENABLE_IMAGE_CAPTIONING", False
    )
    _patch_config_watch(monkeypatch)
    return _views


def _caption_payload(doc_id, ws, rev_id):
    return CaptionMessage(
        document_id=doc_id,
        workspace_id=ws,
        revision_id=rev_id,
        build_profile=FULL.value,
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_task3_handle_caption_claims_runs_completes(
    async_engine, document_factory, monkeypatch
):
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="x" * 64)
    await _seed_caption_table(maker, doc_id, rev_id)
    store = _RecordingCaptionStore()
    views = _patch_caption(monkeypatch, maker, store)
    finalize_calls = _patch_finalize_recorder(monkeypatch, _caption_worker)

    await _caption_worker.handle_caption(_caption_payload(doc_id, ws, rev_id))

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["caption"] == ("completed", 1)
    assert rows["parse"] == ("pending", 0)
    expected = views.revision_markdown_key(ws, doc_id, rev_id)
    assert store.downloads == [expected]
    assert store.uploads == [expected]
    mirrors = await _handler_mirrors(maker, doc_id)
    assert mirrors[2] is True  # captions_done mirror
    assert finalize_calls == [rev_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["running", "completed", "failed"])
async def test_task3_handle_caption_duplicate_and_terminal_noop(
    async_engine, document_factory, monkeypatch, preset
):
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="y" * 64)
    await _seed_caption_table(maker, doc_id, rev_id)
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        await repo.claim_stage_running(rev_id, "caption")
        if preset == "completed":
            await repo.mark_stage_completed(rev_id, "caption")
        elif preset == "failed":
            await repo.mark_stage_failed(rev_id, "caption", failure_class="E")
        await db.commit()
    before_rows = await _handler_stage_rows(maker, rev_id)
    before_mirrors = await _handler_mirrors(maker, doc_id)
    store = _RecordingCaptionStore()
    _patch_caption(monkeypatch, maker, store)

    await _caption_worker.handle_caption(_caption_payload(doc_id, ws, rev_id))

    assert store.downloads == [] and store.uploads == []
    assert await _handler_stage_rows(maker, rev_id) == before_rows
    assert await _handler_mirrors(maker, doc_id) == before_mirrors
    assert await _handler_revision_status(maker, rev_id) == "draft"


@pytest.mark.asyncio
async def test_task3_handle_caption_retry_reclaims_attempt(
    async_engine, document_factory, monkeypatch
):
    from app.queue.connection import note_stage_retry_pending

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="z" * 64)
    await _seed_caption_table(maker, doc_id, rev_id)
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        _, claimed = await repo.claim_stage_running(rev_id, "caption")
        assert claimed is True
        await db.commit()
    async with maker() as db:
        assert await note_stage_retry_pending(rev_id, "caption", db=db) is True
        await db.commit()
    store = _RecordingCaptionStore()
    _patch_caption(monkeypatch, maker, store)
    _patch_finalize_recorder(monkeypatch, _caption_worker)

    await _caption_worker.handle_caption(_caption_payload(doc_id, ws, rev_id))

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["caption"] == ("completed", 2)
    assert len(store.downloads) == 1


# -- kg stubs ---------------------------------------------------------------

class _RecordingKGService:
    def __init__(self):
        self.ingests = []

    async def ingest(self, markdown, *, document_id, revision_id):
        self.ingests.append((markdown, revision_id))


class _RecordingKGStore:
    async def download_markdown(self, key):
        return "# Title\n\nBody text for the knowledge graph."


def _patch_kg(monkeypatch, maker, service, *, patch_store=True):
    import app.services.storage_service as _storage

    monkeypatch.setattr(_kg_worker, "async_session_maker", maker)
    monkeypatch.setattr(_kg_worker, "get_kg_service", lambda workspace_id: service)
    if patch_store:
        monkeypatch.setattr(
            _storage, "get_storage_service", lambda: _RecordingKGStore()
        )
    _patch_config_watch(monkeypatch)


def _kg_payload(doc_id, ws, rev_id):
    return KGMessage(
        document_id=doc_id,
        workspace_id=ws,
        revision_id=rev_id,
        build_profile=FULL.value,
        markdown_s3_key="kb_x/rev/document.md",
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_task3_handle_kg_claims_runs_completes(
    async_engine, document_factory, monkeypatch
):
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="1" * 64)
    service = _RecordingKGService()
    _patch_kg(monkeypatch, maker, service)
    finalize_calls = _patch_finalize_recorder(monkeypatch, _kg_worker)

    await _kg_worker.handle_kg(_kg_payload(doc_id, ws, rev_id))

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["kg"] == ("completed", 1)
    assert rows["parse"] == ("pending", 0)
    assert len(service.ingests) == 1
    assert service.ingests[0][1] == rev_id
    mirrors = await _handler_mirrors(maker, doc_id)
    assert mirrors[3] is True  # kg_done mirror
    assert finalize_calls == [rev_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["running", "completed", "failed"])
async def test_task3_handle_kg_duplicate_and_terminal_noop(
    async_engine, document_factory, monkeypatch, preset
):
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="2" * 64)
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        await repo.claim_stage_running(rev_id, "kg")
        if preset == "completed":
            await repo.mark_stage_completed(rev_id, "kg")
        elif preset == "failed":
            await repo.mark_stage_failed(rev_id, "kg", failure_class="E")
        await db.commit()
    before_rows = await _handler_stage_rows(maker, rev_id)
    before_mirrors = await _handler_mirrors(maker, doc_id)
    service = _RecordingKGService()
    _patch_kg(monkeypatch, maker, service)

    await _kg_worker.handle_kg(_kg_payload(doc_id, ws, rev_id))

    assert service.ingests == []
    assert await _handler_stage_rows(maker, rev_id) == before_rows
    assert await _handler_mirrors(maker, doc_id) == before_mirrors
    assert await _handler_revision_status(maker, rev_id) == "draft"


@pytest.mark.asyncio
async def test_task3_handle_kg_retry_reclaims_attempt(
    async_engine, document_factory, monkeypatch
):
    from app.queue.connection import note_stage_retry_pending

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="3" * 64)
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        _, claimed = await repo.claim_stage_running(rev_id, "kg")
        assert claimed is True
        await db.commit()
    async with maker() as db:
        assert await note_stage_retry_pending(rev_id, "kg", db=db) is True
        await db.commit()
    service = _RecordingKGService()
    _patch_kg(monkeypatch, maker, service)
    _patch_finalize_recorder(monkeypatch, _kg_worker)

    await _kg_worker.handle_kg(_kg_payload(doc_id, ws, rev_id))

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["kg"] == ("completed", 2)
    assert len(service.ingests) == 1


# -- stale-revision isolation across all four handlers ----------------------

@pytest.mark.asyncio
async def test_task3_handlers_isolate_stale_revision(
    async_engine, document_factory, monkeypatch
):
    """R1 handlers touch only R1 rows after R2 is allocated."""
    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, r1 = await _handler_setup_full(maker, doc_id, sha="4" * 64)
    async with maker() as db:
        key = _doc_key(ws, doc_id)
        r2, _profile = await allocate_reindex_revision(
            db,
            doc_id,
            object_key=key,
            size_bytes=11,
            content_sha256="5" * 64,
            version_id="v-1",
        )
        r2_id = r2.revision_id
        await db.commit()
    assert r2_id != r1
    await _seed_raw_chunks(maker, doc_id)
    await _seed_caption_table(maker, doc_id, r1)

    import app.services.storage_service as _storage_for_stale

    class _StaleStore(_RecordingParseStore):
        async def download_markdown(self, key):
            if key.endswith("structure.json"):
                import json as _json

                return _json.dumps(
                    {
                        "chunks": [
                            {
                                "chunk_id": "c1",
                                "ordinal": 0,
                                "content": "hello world",
                                "page_no": 1,
                                "heading_path": [],
                                "source_file": "doc.pdf",
                                "image_refs": [],
                                "table_refs": [],
                                "has_table": False,
                                "has_code": False,
                            }
                        ]
                    }
                )
            return await super().download_markdown(key)

    store = _StaleStore()
    # One combined fake serves parse uploads AND the embed structure-artifact
    # read (utils imports the getter from the storage module directly).
    monkeypatch.setattr(
        _storage_for_stale, "get_storage_service", lambda: store
    )
    _patch_parse(monkeypatch, maker, store)
    await _parse_worker.handle_parse(_parse_payload(doc_id, ws, r1))

    embedder = _FakeEmbedder()
    _patch_embed(monkeypatch, maker, embedder, _FakeVectorStore())
    _patch_finalize_recorder(monkeypatch, _embed_worker)
    await _embed_worker.handle_embed(_embed_payload(doc_id, ws, r1))

    _patch_caption(monkeypatch, maker, _RecordingCaptionStore())
    _patch_finalize_recorder(monkeypatch, _caption_worker)
    await _caption_worker.handle_caption(_caption_payload(doc_id, ws, r1))

    service = _RecordingKGService()
    _patch_kg(monkeypatch, maker, service, patch_store=False)
    _patch_finalize_recorder(monkeypatch, _kg_worker)
    await _kg_worker.handle_kg(_kg_payload(doc_id, ws, r1))

    r1_rows = await _handler_stage_rows(maker, r1)
    assert all(state == "completed" for state, _ in r1_rows.values())
    # The newer generation is untouched: still pending, gate unsatisfied.
    assert await _handler_stage_rows(maker, r2_id) == {
        "parse": ("pending", 0),
        "embed": ("pending", 0),
        "caption": ("pending", 0),
        "kg": ("pending", 0),
    }
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        assert await repo.required_stages_complete(r2_id) is False


# -- queue failure-note mapping + atomic exhaustion -------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "queue,exchange,stage",
    [
        ("hrag.parse", "hrag.parse", "parse"),
        ("hrag.embed", "hrag.embed", "embed"),
        ("hrag.caption", "hrag.caption", "caption"),
        ("hrag.kg.00000000-0000-0000-0000-000000000001", "hrag.kg", "kg"),
    ],
)
async def test_task3_note_failure_stage_retry_touches_only_mapped_stage(
    async_db, document_factory, queue, exchange, stage
):
    """The consolidated failure note resolves the stage from the queue.

    A hardcoded-stage mutation (e.g. always noting ``parse``) moves the
    wrong row and fails here — the note helpers are never called with a
    literal stage from the branch code.
    """
    from app.queue.connection import _note_failure_stage

    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _created = await _allocate_full(async_db, doc_id, key=key, sha="j" * 64)
    repo = DocumentRevisionsRepository(async_db)
    for s in ("parse", "embed", "caption", "kg"):
        await repo.mark_stage_running(revision.revision_id, s)

    resolved = await _note_failure_stage(
        queue_name=queue,
        exchange_name=exchange,
        revision_id=revision.revision_id,
        retry_count=0,
        failure_class="ValueError",
        db=async_db,
    )

    assert resolved == stage
    states = await _stage_states(async_db, revision.revision_id)
    for s in ("parse", "embed", "caption", "kg"):
        assert states[s] == ("pending" if s == stage else "running"), s
    rows = await repo.get_stages(revision.revision_id)
    assert {r.stage: r.attempt_count for r in rows}[stage] == 1
    # The revision stays live on retry so the redelivery can re-claim.
    assert (await repo.get(revision.revision_id)).status == "draft"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "queue,exchange,stage",
    [
        ("hrag.parse", "hrag.parse", "parse"),
        ("hrag.embed", "hrag.embed", "embed"),
        ("hrag.caption", "hrag.caption", "caption"),
        ("hrag.kg.00000000-0000-0000-0000-000000000001", "hrag.kg", "kg"),
    ],
)
async def test_task3_note_failure_stage_exhaustion_fails_stage_and_revision(
    async_db, document_factory, queue, exchange, stage
):
    """Exhaustion atomically fails the exact stage AND the revision."""
    from app.queue.connection import _note_failure_stage

    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _created = await _allocate_full(async_db, doc_id, key=key, sha="k" * 64)
    repo = DocumentRevisionsRepository(async_db)
    await repo.mark_stage_running(revision.revision_id, stage)

    resolved = await _note_failure_stage(
        queue_name=queue,
        exchange_name=exchange,
        revision_id=revision.revision_id,
        retry_count=_conn.MAX_RETRIES,
        failure_class="TimeoutError",
        db=async_db,
    )

    assert resolved == stage
    assert (await _stage_states(async_db, revision.revision_id))[stage] == "failed"
    failed_rev = await repo.get(revision.revision_id)
    assert failed_rev.status == "failed"
    assert failed_rev.failure_stage == stage
    assert failed_rev.failure_class == "TimeoutError"
    # Sibling stages of the same revision are untouched.
    _final_states = await _stage_states(async_db, revision.revision_id)
    for s in ("parse", "embed", "caption", "kg"):
        if s != stage:
            assert _final_states[s] == "pending"


@pytest.mark.asyncio
async def test_task3_note_failure_stage_publish_failure_is_honest(
    async_db, document_factory
):
    """A failed retry publish (no redelivery coming) exhausts honestly."""
    from app.queue.connection import _note_failure_stage

    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _created = await _allocate_full(async_db, doc_id, key=key, sha="m" * 64)
    repo = DocumentRevisionsRepository(async_db)
    await repo.mark_stage_running(revision.revision_id, "embed")

    resolved = await _note_failure_stage(
        queue_name="hrag.embed",
        exchange_name="hrag.embed",
        revision_id=revision.revision_id,
        retry_count=0,
        failure_class="ValueError",
        force_exhausted=True,
        db=async_db,
    )

    assert resolved == "embed"
    assert (await _stage_states(async_db, revision.revision_id))["embed"] == "failed"
    assert (await repo.get(revision.revision_id)).status == "failed"


@pytest.mark.asyncio
async def test_task3_note_failure_stage_ignores_non_pipeline(
    async_db, document_factory
):
    """Memory queues and stageless messages resolve to None, untouched."""
    from app.queue.connection import _note_failure_stage

    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _doc_key(ws, doc_id)
    revision, _created = await _allocate_full(async_db, doc_id, key=key, sha="n" * 64)
    repo = DocumentRevisionsRepository(async_db)
    await repo.mark_stage_running(revision.revision_id, "parse")

    assert await _note_failure_stage(
        queue_name="hrag.memory",
        exchange_name="hrag.memory",
        revision_id=revision.revision_id,
        retry_count=0,
        failure_class="ValueError",
        db=async_db,
    ) is None
    assert await _note_failure_stage(
        queue_name="hrag.embed",
        exchange_name="hrag.embed",
        revision_id=None,
        retry_count=0,
        failure_class="ValueError",
        db=async_db,
    ) is None
    assert await _stage_states(async_db, revision.revision_id) == {
        "parse": "running",
        "embed": "pending",
        "caption": "pending",
        "kg": "pending",
    }
    assert (await repo.get(revision.revision_id)).status == "draft"
    assert (await repo.get(revision.revision_id)).status == "draft"


@pytest.mark.asyncio
async def test_task3_exhausted_note_commits_stage_and_revision_atomically(
    async_engine, document_factory
):
    """Private-session exhaustion commits stage-failed + revision-failed."""
    from app.queue.connection import note_stage_exhausted

    maker = _handler_maker(async_engine)
    doc_id = document_factory()
    ws, rev_id = await _handler_setup_full(maker, doc_id, sha="p" * 64)
    async with maker() as db:
        repo = DocumentRevisionsRepository(db)
        await repo.mark_stage_running(rev_id, "embed")
        await db.commit()

    import app.core.database as _dbmod

    _real_maker = _dbmod.async_session_maker
    _dbmod.async_session_maker = maker
    try:
        assert await note_stage_exhausted(
            rev_id, "embed", failure_class="ValueError"
        ) is True
    finally:
        _dbmod.async_session_maker = _real_maker

    rows = await _handler_stage_rows(maker, rev_id)
    assert rows["embed"][0] == "failed"
    assert await _handler_revision_status(maker, rev_id) == "failed"


# ---------------------------------------------------------------------------
# P1 Task 4 — finalization is gated on revision stage rows, not mirrors
# ---------------------------------------------------------------------------


async def _task4_complete_stages(db, revision_id, stages) -> None:
    """Mark every named stage completed (pending -> completed converges)."""
    repo = DocumentRevisionsRepository(db)
    for stage in stages:
        await repo.mark_stage_completed(revision_id, stage)


def _task4_patch_finalize_spy(monkeypatch):
    """Spy on finalize_revision_if_complete through the real implementation."""
    import app.workers.utils as utils

    calls: list = []
    real = utils.finalize_revision_if_complete

    async def _spy(revision_id, *, expect_complete=False):
        calls.append(
            {"revision_id": revision_id, "expect_complete": expect_complete}
        )
        return await real(revision_id, expect_complete=expect_complete)

    monkeypatch.setattr(utils, "finalize_revision_if_complete", _spy)
    return calls


@pytest.mark.asyncio
async def test_task4_pending_stages_never_call_finalize(
    async_engine, document_factory, monkeypatch
):
    """Pending stages + stale-True mirrors return NOT_READY without finalize.

    The pre-Task-4 bug: a new generation observed the previous generation's
    ``embed_done/captions_done/kg_done`` mirrors, called
    ``finalize_revision_if_complete(expect_complete=True)`` and terminalized
    itself ``failed`` while its own stages were still pending.
    """
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    finalize_calls = _task4_patch_finalize_spy(monkeypatch)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="t4a" + "0" * 61,
            version_id="v-1",
        )
        # Stale mirrors from a previous generation: all done.
        document = await db.get(Document, doc_id)
        document.embed_done = True
        document.captions_done = True
        document.kg_done = True
        await db.commit()
        revision_id = revision.revision_id
    assert profile is FULL

    async with maker() as db:
        document = await db.get(Document, doc_id)
        await check_and_finalize(document, db, revision_id=revision_id)

    assert finalize_calls == [], "pending stages MUST NOT call finalize"
    async with maker() as db:
        row = await db.get(DocumentRevision, revision_id)
        fresh = await db.get(Document, doc_id)
    assert row.status in ("draft", "building")
    assert fresh.status != DocumentStatus.INDEXED


@pytest.mark.asyncio
async def test_task4_running_stage_never_calls_finalize(
    async_engine, document_factory, monkeypatch
):
    """A running stage + stale-True mirrors return NOT_READY without finalize."""
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    finalize_calls = _task4_patch_finalize_spy(monkeypatch)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, _profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="t4b" + "0" * 61,
            version_id="v-1",
        )
        repo = DocumentRevisionsRepository(db)
        await repo.mark_stage_completed(revision.revision_id, "parse")
        await repo.mark_stage_completed(revision.revision_id, "embed")
        await repo.mark_stage_running(revision.revision_id, "caption")
        document = await db.get(Document, doc_id)
        document.embed_done = True
        document.captions_done = True
        document.kg_done = True
        await db.commit()
        revision_id = revision.revision_id

    async with maker() as db:
        document = await db.get(Document, doc_id)
        await check_and_finalize(document, db, revision_id=revision_id)

    assert finalize_calls == [], "a running stage MUST NOT call finalize"
    async with maker() as db:
        row = await db.get(DocumentRevision, revision_id)
    assert row.status in ("draft", "building")


@pytest.mark.asyncio
async def test_task4_partial_stages_never_call_finalize(
    async_engine, document_factory, monkeypatch
):
    """Three of four FULL stages complete + stale mirrors: still NOT_READY."""
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    finalize_calls = _task4_patch_finalize_spy(monkeypatch)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, _profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="t4c" + "0" * 61,
            version_id="v-1",
        )
        await _task4_complete_stages(
            db, revision.revision_id, ("parse", "embed", "caption")
        )
        document = await db.get(Document, doc_id)
        document.embed_done = True
        document.captions_done = True
        document.kg_done = True
        await db.commit()
        revision_id = revision.revision_id

    async with maker() as db:
        document = await db.get(Document, doc_id)
        await check_and_finalize(document, db, revision_id=revision_id)

    assert finalize_calls == [], "an incomplete stage MUST NOT call finalize"
    async with maker() as db:
        row = await db.get(DocumentRevision, revision_id)
    assert row.status in ("draft", "building")


@pytest.mark.asyncio
async def test_task4_complete_stages_publish_despite_stale_false_mirrors(
    async_engine, document_factory, monkeypatch
):
    """All stages complete + manifest complete publishes with mirrors False.

    Stale-False mirrors must neither block finalization nor stop the mirror
    from reaching INDEXED: the revision outcome authorizes the mirror.
    """
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    finalize_calls = _task4_patch_finalize_spy(monkeypatch)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="t4d" + "0" * 61,
            version_id="v-1",
        )
        await _task4_complete_stages(
            db, revision.revision_id, ("parse", "embed", "caption", "kg")
        )
        await _record_complete_artifacts(db, revision, profile)
        document = await db.get(Document, doc_id)
        assert document.embed_done is False
        assert document.captions_done is False
        assert document.kg_done is False
        await db.commit()
        revision_id = revision.revision_id

    async with maker() as db:
        document = await db.get(Document, doc_id)
        await check_and_finalize(document, db, revision_id=revision_id)

    assert len(finalize_calls) == 1
    assert finalize_calls[0]["revision_id"] == revision_id
    assert finalize_calls[0]["expect_complete"] is True
    async with maker() as db:
        row = await db.get(DocumentRevision, revision_id)
        fresh = await db.get(Document, doc_id)
    assert row.status == "published"
    assert fresh.status == DocumentStatus.INDEXED
    assert fresh.current_revision_id == revision_id


@pytest.mark.asyncio
async def test_task4_complete_stages_incomplete_manifest_fails_only_that_revision(
    async_engine, document_factory, monkeypatch
):
    """Authoritative stage completion + missing artifacts fails that revision.

    Stage rows alone never certify absent artifacts (a caption/KG warning
    completion cannot stand in for the manifest): ``verify_draft`` still
    owns the manifest check, and the failure terminalizes only the revision
    — the document pointer stays untouched.
    """
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    finalize_calls = _task4_patch_finalize_spy(monkeypatch)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, _profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="t4e" + "0" * 61,
            version_id="v-1",
        )
        await _task4_complete_stages(
            db, revision.revision_id, ("parse", "embed", "caption", "kg")
        )
        document = await db.get(Document, doc_id)
        document.embed_done = True
        document.captions_done = True
        document.kg_done = True
        await db.commit()
        revision_id = revision.revision_id

    async with maker() as db:
        document = await db.get(Document, doc_id)
        await check_and_finalize(document, db, revision_id=revision_id)

    assert len(finalize_calls) == 1
    assert finalize_calls[0]["expect_complete"] is True
    async with maker() as db:
        row = await db.get(DocumentRevision, revision_id)
        fresh = await db.get(Document, doc_id)
    assert row.status == "failed"
    assert row.failure_stage == "verify"
    assert row.failure_class == "RevisionArtifactsIncomplete"
    assert fresh.status == DocumentStatus.FAILED
    assert fresh.status != DocumentStatus.INDEXED
    assert fresh.current_revision_id is None


@pytest.mark.asyncio
async def test_task4_parse_only_profile_skips_satisfy_gate(
    async_engine, document_factory, monkeypatch
):
    """PARSE_ONLY: parse completed + profile-skipped rows satisfy the gate."""
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    finalize_calls = _task4_patch_finalize_spy(monkeypatch)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="t4f" + "0" * 61,
            version_id="v-1",
            parse_only=True,
        )
        assert profile is PARSE_ONLY
        await _task4_complete_stages(db, revision.revision_id, ("parse",))
        await _record_complete_artifacts(db, revision, profile)
        await db.commit()
        revision_id = revision.revision_id

    async with maker() as db:
        document = await db.get(Document, doc_id)
        await check_and_finalize(document, db, revision_id=revision_id)

    assert len(finalize_calls) == 1
    assert finalize_calls[0]["expect_complete"] is True
    async with maker() as db:
        row = await db.get(DocumentRevision, revision_id)
        fresh = await db.get(Document, doc_id)
    assert row.status == "published"
    assert fresh.status == DocumentStatus.INDEXED
    assert fresh.current_revision_id == revision_id


@pytest.mark.asyncio
async def test_task4_concurrent_finalizers_publish_once(
    async_engine, document_factory, monkeypatch
):
    """Concurrent finalizers preserve the CAS pointer and publish once."""
    import asyncio as _asyncio

    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        revision, profile, _created = await allocate_ingest_revision(
            db,
            doc_id,
            object_key=_doc_key(ws, doc_id),
            size_bytes=11,
            content_sha256="t4g" + "0" * 61,
            version_id="v-1",
        )
        await _task4_complete_stages(
            db, revision.revision_id, ("parse", "embed", "caption", "kg")
        )
        await _record_complete_artifacts(db, revision, profile)
        await db.commit()
        revision_id = revision.revision_id

    async with maker() as db:
        document = await db.get(Document, doc_id)

        async def _one(_i):
            await check_and_finalize(document, db, revision_id=revision_id)

        await _asyncio.gather(*(_one(i) for i in range(5)))

    async with maker() as db:
        row = await db.get(DocumentRevision, revision_id)
        fresh = await db.get(Document, doc_id)
        published_count = await db.scalar(
            select(func.count())
            .select_from(DocumentRevision)
            .where(
                DocumentRevision.document_id == doc_id,
                DocumentRevision.status == "published",
            )
        )
    assert row.status == "published"
    assert published_count == 1
    assert fresh.current_revision_id == revision_id
    assert fresh.status == DocumentStatus.INDEXED


@pytest.mark.asyncio
async def test_task4_older_generation_cannot_rewrite_newer_pointer(
    async_engine, document_factory, monkeypatch
):
    """A stale generation's late finalize never moves the current pointer.

    R2 publishes first; R1 then completes its stages but has no manifest.
    R1 terminalizes ``failed`` while the document keeps pointing at R2.
    """
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    doc_id = document_factory()
    async with maker() as db:
        ws = await _workspace_id(db, doc_id)
        key = _doc_key(ws, doc_id)
        stale, _stale_profile = await allocate_reindex_revision(
            db,
            doc_id,
            object_key=key,
            size_bytes=11,
            content_sha256="t4h" + "0" * 61,
            version_id="v-1",
        )
        current, current_profile = await allocate_reindex_revision(
            db,
            doc_id,
            object_key=key,
            size_bytes=11,
            content_sha256="t4h" + "0" * 61,
            version_id="v-2",
        )
        await _task4_complete_stages(
            db, current.revision_id, ("parse", "embed", "caption", "kg")
        )
        await _record_complete_artifacts(db, current, current_profile)
        await db.commit()
        stale_id = stale.revision_id
        current_id = current.revision_id

    async with maker() as db:
        document = await db.get(Document, doc_id)
        await check_and_finalize(document, db, revision_id=current_id)
    async with maker() as db:
        fresh = await db.get(Document, doc_id)
    assert fresh.status == DocumentStatus.INDEXED
    assert fresh.current_revision_id == current_id

    # The older generation finishes late with an incomplete manifest.
    async with maker() as db:
        await _task4_complete_stages(
            db, stale_id, ("parse", "embed", "caption", "kg")
        )
        await db.commit()
    async with maker() as db:
        document = await db.get(Document, doc_id)
        await check_and_finalize(document, db, revision_id=stale_id)

    async with maker() as db:
        fresh = await db.get(Document, doc_id)
        stale_row = await db.get(DocumentRevision, stale_id)
    assert stale_row.status == "failed"
    assert fresh.status == DocumentStatus.INDEXED
    assert fresh.current_revision_id == current_id


@pytest.mark.asyncio
async def test_task4_legacy_call_without_revision_retains_mirror_behavior(
    async_engine, document_factory, monkeypatch
):
    """No revision: the v1/UI mirror path is unchanged (authorize + block)."""
    maker = _session_maker(async_engine)
    monkeypatch.setattr("app.core.database.async_session_maker", maker)
    finalize_calls = _task4_patch_finalize_spy(monkeypatch)
    chat_id = document_factory(is_chat_upload=True)
    plain_id = document_factory()
    async with maker() as db:
        chat = await db.get(Document, chat_id)
        chat.embed_done = True
        plain = await db.get(Document, plain_id)
        assert plain.embed_done is False
        await db.commit()

    async with maker() as db:
        chat = await db.get(Document, chat_id)
        await check_and_finalize(chat, db)
        plain = await db.get(Document, plain_id)
        await check_and_finalize(plain, db)

    assert finalize_calls == [], "legacy path never calls finalize"
    async with maker() as db:
        fresh_chat = await db.get(Document, chat_id)
        fresh_plain = await db.get(Document, plain_id)
    assert fresh_chat.status == DocumentStatus.INDEXED
    assert fresh_plain.status != DocumentStatus.INDEXED

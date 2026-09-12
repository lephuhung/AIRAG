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

import re
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.models.document import Document, DocumentImage, DocumentTable
from app.models.document_ingestion_attempt import DocumentIngestionAttempt
from app.models.document_revision import DocumentRevision
from app.models.source_arrival import SourceArrival
from app.queue.messages import CaptionMessage, EmbedMessage, KGMessage, ParseMessage
from app.queue.publisher import (
    allocate_clone_revision,
    allocate_ingest_revision,
    allocate_reindex_revision,
    record_source_arrival,
    resolve_ingest_profile,
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
    delete_stage_children,
    load_revision_execution,
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
    the producer must derive ``CHAT_UPLOAD`` from the file-name segment even
    though the raw ``resolve_build_profile`` rule keys off the key start.
    """
    doc_id = document_factory()
    ws = await _workspace_id(async_db, doc_id)
    key = _chat_key(ws, doc_id)

    assert resolve_ingest_profile(key) is CHAT

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
    assert resolve_ingest_profile(key, parse_only=True) is PARSE_ONLY


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
    # The raw ``resolve_build_profile`` keys off the file-name segment only,
    # so namespaced storage keys need the Task-4 producer helper.
    assert resolve_build_profile(key, IngestFlags()) is FULL
    assert resolve_ingest_profile(key) is FULL

"""Phase 1C Task 5 — revision-qualified artifact identity and the current
revision viewer adapter.

Pure persistence tests: they exercise the real v2 schema (``hrag_test_v2_task3``)
and the real :class:`DocumentRevisionsRepository`, with only the object store
faked (a structure artifact is JSON, so the fake is a dict lookup — no MinIO
round trip).

Covers the brief's named tests:
  - ``test_current_document_view_uses_current_revision``
  - ``test_historical_revision_uses_recorded_embedding_namespace``
  - ``test_revision_kg_does_not_leak_old_fact``
plus R1-artifact survival, dimension qualification, stable locators, and the
explicit legacy/``REVISION_NOT_READY`` policy.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import select, text

from app.models.document import Document, DocumentImage
from app.models.document_revision_build import DocumentRevisionBuild
from app.models.document_revision_chunk import DocumentRevisionChunk
from app.services.agents.v2.persistence.document_revisions import (
    DocumentRevisionsRepository,
)
from app.services.agents.v2.persistence.document_views import (
    REVISION_NOT_READY,
    ChunkRecord,
    CurrentDocumentViewAdapter,
    EmbeddingMigrationRequired,
    RevisionNotReady,
    build_structure_artifact,
    embedding_namespace,
    legacy_vector_id,
    load_current_revision_identity,
    load_current_revision_identity_for_workspace,
    load_revision_chunks,
    load_revision_identity,
    load_revision_identity_for_workspace,
    parse_revision_vector_id,
    parse_structure_artifact,
    record_revision_chunk_rows,
    resolve_document_targets,
    resolve_retrieval_revisions,
    revision_kg_scope,
    revision_markdown_key,
    revision_structure_key,
    revision_vector_id,
)
from app.services.agents.v2.persistence.source_identity import (
    RevisionBuildProfile,
    compute_source_object_identity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeArtifactStore:
    """Minimal object store: ``key -> text`` in memory."""

    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    async def upload_artifact(self, key: str, content: str, content_type: str) -> str:
        self.objects[key] = content
        return key

    async def download_markdown(self, key: str) -> str:
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]


def _identity(key: str, sha: str) -> str:
    return compute_source_object_identity(
        bucket="hrag-uploads",
        object_key=key,
        version_id=None,
        etag="abc123",
        size_bytes=1024,
        content_sha256=sha,
    )


async def _publish_revision(
    db,
    *,
    document_id: uuid.UUID,
    workspace_id: uuid.UUID,
    object_key: str,
    sha: str,
    chunks: list[ChunkRecord],
    storage: FakeArtifactStore,
    namespace: str,
    model_hash: str = "modelhash1",
    dimension: int = 768,
    profile: RevisionBuildProfile = RevisionBuildProfile.FULL,
    vector_artifact_version: str = "v1",
) -> uuid.UUID:
    """Allocate → record artifacts → verify → publish one revision.

    Writes the revision's markdown + structure artifacts into ``storage`` under
    the revision-qualified keys, exactly as the parse worker does.
    """
    repo = DocumentRevisionsRepository(db)
    revision = await repo.allocate_draft(
        document_id, _identity(object_key, sha), profile
    )
    rid = revision.revision_id

    markdown_key = revision_markdown_key(workspace_id, document_id, rid)
    structure_key = revision_structure_key(workspace_id, document_id, rid)
    await storage.upload_artifact(markdown_key, f"# markdown for {rid}", "text/markdown")
    await storage.upload_artifact(
        structure_key,
        build_structure_artifact(rid, document_id, chunks),
        "application/json",
    )
    await record_revision_chunk_rows(db, rid, chunks)

    await repo.record_artifacts(
        rid,
        profile,
        embedding_namespace=namespace,
        embedding_model_hash=model_hash,
        embedding_dimension=dimension,
        vector_artifact_version=vector_artifact_version,
        markdown_artifact_key=markdown_key,
        structure_artifact_key=structure_key,
        captions_skipped=(profile is not RevisionBuildProfile.FULL),
        kg_skipped=(profile is not RevisionBuildProfile.FULL),
    )
    await repo.verify_draft(rid)
    revision, outcome = await repo.publish(rid)
    await db.commit()
    assert outcome.value in ("became_current", "published_historical")
    return rid


def _chunks(prefix: str, count: int = 2) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            ordinal=i,
            content=f"{prefix} chunk {i}",
            page_no=i + 1,
            heading_path=[f"Điều {i + 1}"],
            source_file=f"{prefix}.pdf",
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Identity scheme (pure)
# ---------------------------------------------------------------------------


def test_revision_object_keys_are_revision_qualified():
    ws, doc, rev = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    md = revision_markdown_key(ws, doc, rev)
    struct = revision_structure_key(ws, doc, rev)
    assert str(rev) in md and str(doc) in md and str(ws) in md
    assert str(rev) in struct
    assert md != struct
    # A different revision never shares an object key.
    assert revision_markdown_key(ws, doc, uuid.uuid4()) != md


def test_vector_ids_round_trip_to_stable_locator():
    rev = uuid.uuid4()
    vid = revision_vector_id(rev, 7)
    assert parse_revision_vector_id(vid) == (rev, 7)
    # A legacy document-scoped id is explicitly NOT a revision locator.
    assert parse_revision_vector_id(legacy_vector_id(uuid.uuid4(), 7)) is None
    assert parse_revision_vector_id("nonsense") is None


def test_embedding_namespace_is_model_and_dimension_qualified():
    ws = uuid.uuid4()
    ns_768 = embedding_namespace(ws, "hashA", 768)
    ns_1024 = embedding_namespace(ws, "hashA", 1024)
    ns_other_model = embedding_namespace(ws, "hashB", 768)
    assert ns_768 == f"ws_{ws}_embed_hashA_d768"
    # A dimension or model change yields a DIFFERENT namespace — never a
    # delete/recreate of the existing collection.
    assert len({ns_768, ns_1024, ns_other_model}) == 3
    with pytest.raises(ValueError):
        embedding_namespace(ws, "", 768)
    with pytest.raises(ValueError):
        embedding_namespace(ws, "hashA", 0)


def test_embedding_migration_required_is_exported_from_the_identity_module():
    # One spelling: the vector store raises the exception the identity module
    # publishes, so Task 9's consumers import it from either place.
    assert EmbeddingMigrationRequired.__name__ == "EmbeddingMigrationRequired"


def test_structure_artifact_round_trips_chunk_locators():
    rev, doc = uuid.uuid4(), uuid.uuid4()
    chunks = _chunks("R1", 3)
    raw = build_structure_artifact(rev, doc, chunks)
    payload = json.loads(raw)
    assert payload["revision_id"] == str(rev)
    assert payload["document_id"] == str(doc)
    parsed = parse_structure_artifact(raw)
    assert [(c.chunk_id, c.ordinal, c.content) for c in parsed] == [
        (c.chunk_id, c.ordinal, c.content) for c in chunks
    ]
    # A blank/missing artifact degrades to "no chunks" instead of raising.
    assert parse_structure_artifact("") == []


# ---------------------------------------------------------------------------
# Repository-backed artifact identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revision_chunk_rows_are_revision_scoped(async_db, document_factory):
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    repo = DocumentRevisionsRepository(async_db)
    r1 = (
        await repo.allocate_draft(
            doc_id,
            _identity(f"kb_{ws}/doc_{doc_id}.pdf", "1" * 64),
            RevisionBuildProfile.FULL,
        )
    ).revision_id
    r2 = (
        await repo.allocate_draft(
            doc_id,
            _identity(f"kb_{ws}/doc_{doc_id}.pdf", "2" * 64),
            RevisionBuildProfile.FULL,
        )
    ).revision_id
    await record_revision_chunk_rows(async_db, r1, _chunks("R1", 2))
    await record_revision_chunk_rows(async_db, r2, _chunks("R2", 3))
    await async_db.commit()

    rows = (
        await async_db.scalars(
            select(DocumentRevisionChunk).where(
                DocumentRevisionChunk.revision_id.in_([r1, r2])
            )
        )
    ).all()
    assert len(rows) == 5
    assert {r.revision_id for r in rows} == {r1, r2}
    # Re-recording R1 replaces only R1's rows.
    await record_revision_chunk_rows(async_db, r1, _chunks("R1b", 1))
    await async_db.commit()
    assert (
        await async_db.scalar(
            select(DocumentRevisionChunk.revision_id).where(
                DocumentRevisionChunk.revision_id == r1
            )
        )
    ) == r1
    assert len(
        (
            await async_db.scalars(
                select(DocumentRevisionChunk).where(
                    DocumentRevisionChunk.revision_id == r1
                )
            )
        ).all()
    ) == 1
    # R2 untouched.
    assert len(
        (
            await async_db.scalars(
                select(DocumentRevisionChunk).where(
                    DocumentRevisionChunk.revision_id == r2
                )
            )
        ).all()
    ) == 3
    assert doc_id is not None


@pytest.mark.asyncio
async def test_historical_revision_uses_recorded_embedding_namespace(
    async_db, document_factory
):
    """R1 retrieval resolves R1's own manifest, not current config."""
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()

    r1_ns = embedding_namespace(ws, "hashOLD", 768)
    r2_ns = embedding_namespace(ws, "hashNEW", 1024)
    r1 = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="a" * 64,
        chunks=_chunks("R1"),
        storage=storage,
        namespace=r1_ns,
        model_hash="hashOLD",
        dimension=768,
    )
    r2 = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="b" * 64,
        chunks=_chunks("R2"),
        storage=storage,
        namespace=r2_ns,
        model_hash="hashNEW",
        dimension=1024,
    )

    # The document's CURRENT revision is R2 …
    current = await load_current_revision_identity(async_db, doc_id)
    assert current is not None and current.revision_id == r2
    assert current.embedding_namespace == r2_ns
    assert current.embedding_dimension == 1024

    # … but R1 (historical) still resolves ITS recorded namespace/model/dim.
    historical = await load_revision_identity(async_db, r1)
    assert historical.embedding_namespace == r1_ns
    assert historical.embedding_model_hash == "hashOLD"
    assert historical.embedding_dimension == 768
    assert historical.vector_artifact_version == "v1"
    assert historical.vectors_available is True

    # And the v2 binding resolves the current revision, not the historical one.
    resolved = await resolve_retrieval_revisions(async_db, [doc_id])
    assert [i.revision_id for i in resolved] == [r2]


@pytest.mark.asyncio
async def test_r1_artifacts_survive_r2_publication(async_db, document_factory):
    """Copy-on-write: publishing R2 mutates nothing belonging to R1."""
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()
    ns = embedding_namespace(ws, "hashA", 768)

    r1 = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="a" * 64,
        chunks=_chunks("R1"),
        storage=storage,
        namespace=ns,
    )
    r1_build_before = await async_db.scalar(
        select(DocumentRevisionBuild).where(DocumentRevisionBuild.revision_id == r1)
    )
    r1_snapshot = (
        r1_build_before.markdown_artifact_key,
        r1_build_before.structure_artifact_key,
        r1_build_before.embedding_namespace,
    )

    r2 = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="b" * 64,
        chunks=_chunks("R2"),
        storage=storage,
        namespace=ns,
    )
    assert r2 != r1

    # R1's artifacts are byte-identical and still readable.
    r1_build_after = await async_db.scalar(
        select(DocumentRevisionBuild).where(DocumentRevisionBuild.revision_id == r1)
    )
    assert (
        r1_build_after.markdown_artifact_key,
        r1_build_after.structure_artifact_key,
        r1_build_after.embedding_namespace,
    ) == r1_snapshot
    identity_r1 = await load_revision_identity(async_db, r1)
    r1_chunks = await load_revision_chunks(async_db, identity_r1, storage=storage)
    assert [c.content for c in r1_chunks] == ["R1 chunk 0", "R1 chunk 1"]

    # R1 is still published (historical), never purged by a reindex/publish.
    from app.models.document_revision import DocumentRevision

    r1_row = await async_db.get(DocumentRevision, r1)
    assert r1_row.status == "published"
    assert r1_row.artifacts_purged_at is None
    assert r1_row.artifact_retention_starts_at is not None


@pytest.mark.asyncio
async def test_current_document_view_uses_current_revision(async_db, document_factory):
    """Publish R1 then R2 → the viewer serves R2 only; R1 stays readable."""
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()
    ns = embedding_namespace(ws, "hashA", 768)

    r1 = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="a" * 64,
        chunks=_chunks("R1", 3),
        storage=storage,
        namespace=ns,
    )
    r2 = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="b" * 64,
        chunks=_chunks("R2", 2),
        storage=storage,
        namespace=ns,
    )

    # Images: one owned by R1, one by R2, one legacy (revision_id NULL).
    async_db.add_all(
        [
            DocumentImage(
                document_id=doc_id,
                revision_id=r1,
                image_id="img-r1",
                page_no=1,
                file_path="/tmp/r1.png",
            ),
            DocumentImage(
                document_id=doc_id,
                revision_id=r2,
                image_id="img-r2",
                page_no=1,
                file_path="/tmp/r2.png",
            ),
            DocumentImage(
                document_id=doc_id,
                revision_id=None,
                image_id="img-legacy",
                page_no=1,
                file_path="/tmp/legacy.png",
            ),
        ]
    )
    # The legacy mirror points at R1's object; the viewer must ignore it.
    await async_db.execute(
        text("UPDATE documents SET markdown_s3_key = :k WHERE id = :d"),
        {"k": revision_markdown_key(ws, doc_id, r1), "d": str(doc_id)},
    )
    await async_db.commit()

    adapter = CurrentDocumentViewAdapter(async_db, storage=storage)
    identity = await adapter.current_identity(doc_id)
    assert identity is not None and identity.revision_id == r2

    markdown = await adapter.load_markdown(doc_id)
    assert markdown == f"# markdown for {r2}"
    assert str(r1) not in markdown

    ctx = await adapter.load_chunk_context(doc_id, context_window=5)
    assert ctx["revision_id"] == str(r2)
    assert ctx["legacy"] is False
    assert ctx["total_chunks"] == 2
    assert [c["chunk_index"] for c in ctx["chunks"]] == [0, 1]
    assert all("R2 chunk" in c["content"] for c in ctx["chunks"])
    assert not any("R1 chunk" in c["content"] for c in ctx["chunks"])
    # Stable locators reconstruct from the stored vector id.
    assert ctx["chunks"][0]["vector_id"] == revision_vector_id(r2, 0)

    images = await adapter.load_images(doc_id)
    assert [i.image_id for i in images] == ["img-r2"]

    # R1 remains readable internally (evidence/historical retrieval).
    r1_identity = await load_revision_identity(async_db, r1)
    r1_ctx_chunks = await load_revision_chunks(
        async_db, r1_identity, storage=storage
    )
    assert [c.content for c in r1_ctx_chunks] == [
        "R1 chunk 0",
        "R1 chunk 1",
        "R1 chunk 2",
    ]
    r1_images = (
        await async_db.scalars(
            select(DocumentImage).where(DocumentImage.revision_id == r1)
        )
    ).all()
    assert [i.image_id for i in r1_images] == ["img-r1"]


@pytest.mark.asyncio
async def test_chunk_context_resolves_locator_from_heading(async_db, document_factory):
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()
    chunks = [
        ChunkRecord(chunk_id=str(uuid.uuid4()), ordinal=0, content="a",
                    heading_path=["Điều 1"], page_no=1),
        ChunkRecord(chunk_id=str(uuid.uuid4()), ordinal=1, content="b",
                    heading_path=["Điều 2"], page_no=2),
        ChunkRecord(chunk_id=str(uuid.uuid4()), ordinal=2, content="c",
                    heading_path=["Điều 3"], page_no=3),
    ]
    await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="a" * 64,
        chunks=chunks,
        storage=storage,
        namespace=embedding_namespace(ws, "hashA", 768),
    )
    adapter = CurrentDocumentViewAdapter(async_db, storage=storage)
    ctx = await adapter.load_chunk_context(
        doc_id, heading_path="Điều 2", context_window=1
    )
    assert ctx["target_chunk_index"] == 1
    assert [c["chunk_index"] for c in ctx["chunks"]] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Legacy / REVISION_NOT_READY policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_document_is_not_ready_and_has_no_revision(
    async_db, document_factory
):
    """A populated legacy document has no current revision → REVISION_NOT_READY."""
    doc_id = document_factory(content_hash="c" * 64)
    await async_db.execute(
        text(
            "UPDATE documents SET markdown_s3_key = :k, status = 'indexed', "
            "chunk_count = 4 WHERE id = :d"
        ),
        {"k": f"kb_legacy/doc_{doc_id}.md", "d": str(doc_id)},
    )
    await async_db.commit()

    assert await load_current_revision_identity(async_db, doc_id) is None
    targets = await resolve_document_targets(async_db, [doc_id])
    assert targets[0].is_legacy is True

    with pytest.raises(RevisionNotReady) as exc:
        await resolve_retrieval_revisions(async_db, [doc_id])
    assert exc.value.code == REVISION_NOT_READY
    assert exc.value.document_id == str(doc_id)

    # The viewer never falls back to the legacy key when a revision exists; with
    # no revision at all the legacy key is the v1 adapter's path.
    storage = FakeArtifactStore()
    storage.objects[f"kb_legacy/doc_{doc_id}.md"] = "# legacy markdown"
    adapter = CurrentDocumentViewAdapter(async_db, storage=storage)
    assert await adapter.load_markdown(doc_id) == "# legacy markdown"


@pytest.mark.asyncio
async def test_revision_aware_reindex_publishes_a_ready_revision(
    async_db, document_factory
):
    """After a revision-aware reindex the v2 binding succeeds."""
    doc_id = document_factory(content_hash="c" * 64)
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()
    with pytest.raises(RevisionNotReady):
        await resolve_retrieval_revisions(async_db, [doc_id])

    rid = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="d" * 64,
        chunks=_chunks("R1"),
        storage=storage,
        namespace=embedding_namespace(ws, "hashA", 768),
    )
    resolved = await resolve_retrieval_revisions(async_db, [doc_id])
    assert [i.revision_id for i in resolved] == [rid]


@pytest.mark.asyncio
async def test_parse_only_revision_has_no_vectors_and_fails_closed(
    async_db, document_factory
):
    """A published PARSE_ONLY revision is viewer-ready but not vector-ready."""
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()
    repo = DocumentRevisionsRepository(async_db)
    revision = await repo.allocate_draft(
        doc_id,
        _identity(f"kb_{ws}/doc_{doc_id}.pdf", "e" * 64),
        RevisionBuildProfile.PARSE_ONLY,
    )
    rid = revision.revision_id
    md_key = revision_markdown_key(ws, doc_id, rid)
    struct_key = revision_structure_key(ws, doc_id, rid)
    await storage.upload_artifact(md_key, "# parse only", "text/markdown")
    await storage.upload_artifact(
        struct_key, build_structure_artifact(rid, doc_id, _chunks("P", 1)), "application/json"
    )
    await repo.record_artifacts(
        rid,
        RevisionBuildProfile.PARSE_ONLY,
        markdown_artifact_key=md_key,
        structure_artifact_key=struct_key,
        captions_skipped=True,
        kg_skipped=True,
        embed_skipped=True,
    )
    await repo.verify_draft(rid)
    await repo.publish(rid)
    await async_db.commit()

    identity = await load_revision_identity(async_db, rid)
    assert identity.vectors_available is False
    adapter = CurrentDocumentViewAdapter(async_db, storage=storage)
    assert await adapter.load_markdown(doc_id) == "# parse only"
    with pytest.raises(RevisionNotReady):
        await load_revision_identity(async_db, rid, require_vectors=True)


# ---------------------------------------------------------------------------
# KG revision provenance lives in ``test_revision_live_callers.py`` (it needs
# the full dependency set: ``legal_kg_service`` imports the LLM providers).
# ---------------------------------------------------------------------------


def test_kg_revision_scope_is_the_revision_id():
    rev = uuid.uuid4()
    assert revision_kg_scope(rev) == str(rev)


# ---------------------------------------------------------------------------
# Workspace / tombstone guards on caller-supplied documents and revisions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revision_identity_is_rejected_outside_its_workspace(
    async_db, document_factory
):
    """A caller-supplied revision id must be joined to its document/workspace."""
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()
    rid = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="f" * 64,
        chunks=_chunks("R1"),
        storage=storage,
        namespace=embedding_namespace(ws, "hashA", 768),
    )

    identity = await load_revision_identity_for_workspace(
        async_db, rid, ws, require_vectors=True
    )
    assert identity.revision_id == rid

    # Another workspace that merely learns the revision id must be rejected.
    with pytest.raises(RevisionNotReady):
        await load_revision_identity_for_workspace(
            async_db, rid, uuid.uuid4(), require_vectors=True
        )
    # An unknown revision is rejected the same way.
    with pytest.raises(RevisionNotReady):
        await load_revision_identity_for_workspace(async_db, uuid.uuid4(), ws)


@pytest.mark.asyncio
async def test_tombstoned_document_has_no_retrievable_revision(
    async_db, document_factory
):
    """A tombstoned document's published revision must not be selectable."""
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()
    rid = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="g" * 64,
        chunks=_chunks("R1"),
        storage=storage,
        namespace=embedding_namespace(ws, "hashA", 768),
    )

    await DocumentRevisionsRepository(async_db).mark_source_deleted(
        doc_id, reason="document_deleted"
    )
    await async_db.commit()

    with pytest.raises(RevisionNotReady):
        await load_revision_identity_for_workspace(async_db, rid, ws)

    targets = await resolve_document_targets(async_db, [doc_id], workspace_id=ws)
    assert targets[0].eligible is False
    assert targets[0].identity is None
    # A tombstoned document is NOT a legacy document to fall back on.
    assert targets[0].is_legacy is False


@pytest.mark.asyncio
async def test_resolve_document_targets_marks_foreign_documents_ineligible(
    async_db, document_factory
):
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()
    rid = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="h" * 64,
        chunks=_chunks("R1"),
        storage=storage,
        namespace=embedding_namespace(ws, "hashA", 768),
    )

    own = await resolve_document_targets(async_db, [doc_id], workspace_id=ws)
    assert own[0].eligible is True
    assert own[0].identity is not None and own[0].identity.revision_id == rid

    foreign = await resolve_document_targets(
        async_db, [doc_id], workspace_id=uuid.uuid4()
    )
    assert foreign[0].eligible is False
    assert foreign[0].is_legacy is False


# ---------------------------------------------------------------------------
# P0 Task 3 fix round 2 (R2-I3): mutation-sensitive DB tests for the
# workspace-scoped CURRENT-revision guard.
#
# Unlike the stub-level adapter tests, these mutate real rows: publish a
# revision (sets ``documents.current_revision_id``), move the document to a
# foreign workspace / tombstone it, and prove the guarded lookup rejects
# while the owned legacy pointer still returns ``None``.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_current_lookup_rejects_foreign_workspace(
    async_db, document_factory
):
    """R2-I3: the current-revision pointer of another workspace is rejected."""
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()
    rid = await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="1" * 64,
        chunks=_chunks("W1"),
        storage=storage,
        namespace=embedding_namespace(ws, "hashA", 768),
    )
    owned = await load_current_revision_identity_for_workspace(
        async_db, doc_id, ws
    )
    assert owned is not None and owned.revision_id == rid

    with pytest.raises(RevisionNotReady):
        await load_current_revision_identity_for_workspace(
            async_db, doc_id, uuid.uuid4()
        )


@pytest.mark.asyncio
async def test_workspace_current_lookup_rejects_tombstoned_document(
    async_db, document_factory
):
    """R2-I3: tombstoning the owned document rejects its current revision."""
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = FakeArtifactStore()
    await _publish_revision(
        async_db,
        document_id=doc_id,
        workspace_id=ws,
        object_key=f"kb_{ws}/doc_{doc_id}.pdf",
        sha="2" * 64,
        chunks=_chunks("W2"),
        storage=storage,
        namespace=embedding_namespace(ws, "hashA", 768),
    )
    assert (
        await load_current_revision_identity_for_workspace(async_db, doc_id, ws)
    ) is not None

    await DocumentRevisionsRepository(async_db).mark_source_deleted(
        doc_id, reason="document_deleted"
    )
    await async_db.commit()

    with pytest.raises(RevisionNotReady):
        await load_current_revision_identity_for_workspace(async_db, doc_id, ws)


@pytest.mark.asyncio
async def test_workspace_current_lookup_returns_none_for_owned_legacy(
    async_db, document_factory
):
    """R2-I3: an owned document with ``current_revision_id=None`` returns None."""
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    assert (
        await load_current_revision_identity_for_workspace(async_db, doc_id, ws)
    ) is None


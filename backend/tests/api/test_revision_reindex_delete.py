"""Phase 1C Task 5 — tombstone-first delete and copy-on-write reindex.

Runs in the full-dependency container (``fastapi``/``chromadb``/``aio_pika``).

Covers the brief's named test ``test_delete_tombstones_before_gc`` plus the
reindex/delete seam guards: the destructive pre-delete, the completion-flag
reset, and the unconditional workspace ``VectorStore.delete_collection()`` must
all be gone, and a dimension change must not destroy a collection that holds
published revisions' vectors.
"""

from __future__ import annotations

import inspect
import uuid

import pytest
from sqlalchemy import select, text

from app.models.document import Document, DocumentImage
from app.models.document_revision import DocumentRevision
from app.models.document_revision_build import DocumentRevisionBuild
from app.models.document_revision_chunk import DocumentRevisionChunk
from app.services.agents.v2.persistence.document_revisions import (
    DocumentRevisionsRepository,
)
from app.services.agents.v2.persistence.document_views import (
    ChunkRecord,
    build_structure_artifact,
    embedding_namespace,
    record_revision_chunk_rows,
    revision_markdown_key,
    revision_structure_key,
)
from app.services.agents.v2.persistence.source_identity import (
    RevisionBuildProfile,
    compute_source_object_identity,
)


class _FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    async def upload_artifact(self, key, content, content_type) -> str:
        self.objects[key] = content
        return key

    async def download_markdown(self, key) -> str:
        return self.objects[key]


async def _publish(
    db,
    *,
    document_id,
    workspace_id,
    sha: str,
    chunk_count: int = 2,
    store: "_FakeStore | None" = None,
):
    repo = DocumentRevisionsRepository(db)
    identity = compute_source_object_identity(
        bucket="hrag-uploads",
        object_key=f"kb_{workspace_id}/doc_{document_id}.pdf",
        version_id=None,
        etag="etag1",
        size_bytes=10,
        content_sha256=sha,
    )
    revision = await repo.allocate_draft(
        document_id, identity, RevisionBuildProfile.FULL
    )
    rid = revision.revision_id
    store = store or _FakeStore()
    md = revision_markdown_key(workspace_id, document_id, rid)
    struct = revision_structure_key(workspace_id, document_id, rid)
    chunks = [
        ChunkRecord(chunk_id=str(uuid.uuid4()), ordinal=i, content=f"c{i}")
        for i in range(chunk_count)
    ]
    await store.upload_artifact(md, "# md", "text/markdown")
    await store.upload_artifact(
        struct, build_structure_artifact(rid, document_id, chunks), "application/json"
    )
    await record_revision_chunk_rows(db, rid, chunks)
    await repo.record_artifacts(
        rid,
        RevisionBuildProfile.FULL,
        embedding_namespace=embedding_namespace(workspace_id, "hashA", 768),
        embedding_model_hash="hashA",
        embedding_dimension=768,
        vector_artifact_version="v1",
        markdown_artifact_key=md,
        structure_artifact_key=struct,
    )
    await repo.verify_draft(rid)
    await repo.publish(rid)
    await db.commit()
    return rid


# ---------------------------------------------------------------------------
# test_delete_tombstones_before_gc
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_tombstones_before_gc(async_db, document_factory):
    """Delete tombstones and preserves artifacts; only GC may reclaim them."""
    doc_id = document_factory(content_hash="a" * 64)
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    r1 = await _publish(
        async_db, document_id=doc_id, workspace_id=ws, sha="a" * 64
    )
    async_db.add(
        DocumentImage(
            document_id=doc_id,
            revision_id=r1,
            image_id="img-r1",
            page_no=1,
            file_path="/tmp/r1.png",
        )
    )
    await async_db.commit()

    repo = DocumentRevisionsRepository(async_db)
    document = await repo.mark_source_deleted(doc_id, reason="document_deleted")
    await async_db.commit()

    # Normal/current lookup is denied.
    assert document.source_deleted_at is not None
    assert document.current_revision_id is None
    visible = await async_db.scalar(
        select(Document.id).where(
            Document.id == doc_id, Document.source_deleted_at.is_(None)
        )
    )
    assert visible is None

    # Every revision and artifact is retained for evidence grounding.
    r1_row = await async_db.get(DocumentRevision, r1)
    assert r1_row is not None
    assert r1_row.status == "published"
    assert r1_row.artifacts_purged_at is None
    assert r1_row.artifact_retention_starts_at is not None
    build = await async_db.scalar(
        select(DocumentRevisionBuild).where(DocumentRevisionBuild.revision_id == r1)
    )
    assert build is not None and build.markdown_artifact_key is not None
    assert build.embedding_namespace is not None
    assert (
        await async_db.scalar(
            select(DocumentRevisionChunk.chunk_id).where(
                DocumentRevisionChunk.revision_id == r1
            )
        )
    ) is not None
    assert (
        await async_db.scalar(
            select(DocumentImage.image_id).where(DocumentImage.revision_id == r1)
        )
    ) == "img-r1"

    # Idempotent: a second tombstone neither fails nor changes anything.
    again = await repo.mark_source_deleted(doc_id, reason="document_deleted")
    await async_db.commit()
    assert again.source_deleted_at == document.source_deleted_at


@pytest.mark.asyncio
async def test_tombstone_abandons_unpublished_drafts_only(
    async_db, document_factory
):
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    published = await _publish(
        async_db, document_id=doc_id, workspace_id=ws, sha="b" * 64
    )
    repo = DocumentRevisionsRepository(async_db)
    draft = (
        await repo.allocate_draft(
            doc_id,
            compute_source_object_identity(
                bucket="b",
                object_key=f"kb_{ws}/doc_{doc_id}.pdf",
                version_id=None,
                etag="etag2",
                size_bytes=11,
                content_sha256="c" * 64,
            ),
            RevisionBuildProfile.FULL,
        )
    ).revision_id
    await async_db.commit()

    await repo.mark_source_deleted(doc_id, reason="document_deleted")
    await async_db.commit()

    assert (await async_db.get(DocumentRevision, published)).status == "published"
    assert (await async_db.get(DocumentRevision, draft)).status == "abandoned"


# ---------------------------------------------------------------------------
# The destructive seams must be gone
# ---------------------------------------------------------------------------


def _source(fn) -> str:
    return inspect.getsource(fn)


def test_delete_document_endpoint_is_tombstone_only():
    from app.api import documents as documents_api

    src = _source(documents_api.delete_document)
    assert "mark_source_deleted" in src
    # No physical reclamation from the endpoint.
    assert "db.delete(document)" not in src
    assert "delete_file(" not in src
    assert "delete_markdown(" not in src
    assert "rag_service.delete_document" not in src
    assert "kg_service.delete_document" not in src


def test_reindex_document_does_not_pre_delete_or_reset_flags():
    from app.api import rag as rag_api

    src = _source(rag_api.reindex_document)
    assert "allocate_reindex_revision" in src
    # The pre-emptive purge is gone.
    assert "rag_service.delete_document" not in src
    assert "delete_markdown" not in src
    assert "delete_collection" not in src
    # Document completion flags / the markdown mirror are not reset before the
    # replacement revision publishes.
    for forbidden in (
        "embed_done = False",
        "captions_done = False",
        "kg_done = False",
        "markdown_s3_key = None",
        "chunk_count = 0",
        "image_count = 0",
        "table_count = 0",
    ):
        assert forbidden not in src, forbidden


def test_reindex_workspace_never_deletes_the_collection_or_purges():
    from app.api import rag as rag_api

    src = _source(rag_api.reindex_workspace)
    assert "allocate_reindex_revision" in src
    assert "delete_collection" not in src
    assert "rag_service.delete_document" not in src
    for forbidden in (
        "embed_done = False",
        "captions_done = False",
        "kg_done = False",
    ):
        assert forbidden not in src, forbidden


@pytest.mark.asyncio
async def test_reindex_allocation_keeps_the_current_revision_current(
    async_db, document_factory
):
    """Allocating a replacement revision changes nothing until it publishes."""
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    r1 = await _publish(
        async_db, document_id=doc_id, workspace_id=ws, sha="d" * 64
    )
    r1_build = await async_db.scalar(
        select(DocumentRevisionBuild).where(DocumentRevisionBuild.revision_id == r1)
    )
    snapshot = (r1_build.markdown_artifact_key, r1_build.embedding_namespace)

    repo = DocumentRevisionsRepository(async_db)
    replacement = await repo.allocate_draft(
        doc_id,
        compute_source_object_identity(
            bucket="hrag-uploads",
            object_key=f"kb_{ws}/doc_{doc_id}.pdf",
            version_id=None,
            etag="etag1",
            size_bytes=10,
            content_sha256="d" * 64,
        ),
        RevisionBuildProfile.FULL,
        reindex_of_revision_id=r1,
    )
    await async_db.commit()

    current = await async_db.scalar(
        select(Document.current_revision_id).where(Document.id == doc_id)
    )
    assert current == r1  # the published revision is still current
    assert (await async_db.get(DocumentRevision, replacement.revision_id)).status == (
        "draft"
    )
    r1_build_after = await async_db.scalar(
        select(DocumentRevisionBuild).where(DocumentRevisionBuild.revision_id == r1)
    )
    assert (
        r1_build_after.markdown_artifact_key,
        r1_build_after.embedding_namespace,
    ) == snapshot


# ---------------------------------------------------------------------------
# Dimension preservation
# ---------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, *, raise_dimension: bool = False):
        self.raise_dimension = raise_dimension
        self.add_calls = 0
        self.delete_calls = 0

    def add(self, **kwargs):
        self.add_calls += 1
        if self.raise_dimension:
            raise Exception("dimension mismatch: expected 768 got 1024")

    def delete(self, **kwargs):
        self.delete_calls += 1


class _FakeChromaClient:
    def __init__(self):
        self.delete_collection_calls: list[str] = []
        self.get_or_create_calls: list[str] = []

    def delete_collection(self, name):
        self.delete_collection_calls.append(name)

    def get_or_create_collection(self, name, metadata=None):
        self.get_or_create_calls.append(name)
        return _FakeCollection()

    def heartbeat(self):
        return 1


def test_dimension_change_never_deletes_a_collection_with_published_vectors(
    monkeypatch,
):
    """A new embedding dimension selects a new namespace, never a delete."""
    from app.services.embedding import vector_store as vs

    ws = uuid.uuid4()
    r1_ns = embedding_namespace(ws, "hashOLD", 768)
    r2_ns = embedding_namespace(ws, "hashNEW", 1024)
    assert r1_ns != r2_ns

    fake_client = _FakeChromaClient()
    monkeypatch.setattr(vs, "get_chroma_client", lambda: fake_client)

    # R1's collection is written under its own namespace.
    r1_store = vs.get_vector_store(ws, namespace=r1_ns)
    r1_store.add_documents(
        ids=["rev_r1_chunk_0"], embeddings=[[0.0] * 768], documents=["x"]
    )
    assert fake_client.get_or_create_calls == [r1_ns]

    # A dimension change writes to a NEW namespace; the old one is untouched.
    r2_store = vs.get_vector_store(ws, namespace=r2_ns)
    r2_store.add_documents(
        ids=["rev_r2_chunk_0"], embeddings=[[0.0] * 1024], documents=["y"]
    )
    assert fake_client.get_or_create_calls == [r1_ns, r2_ns]
    assert fake_client.delete_collection_calls == []

    # Even a same-namespace mismatch must fail closed, not delete/recreate.
    mismatched = vs.get_vector_store(ws, namespace=r1_ns)
    mismatched._collection = _FakeCollection(raise_dimension=True)
    with pytest.raises(vs.EmbeddingMigrationRequired):
        mismatched.add_documents(
            ids=["rev_r1_chunk_1"], embeddings=[[0.0] * 1024], documents=["z"]
        )
    assert fake_client.delete_collection_calls == []


@pytest.mark.asyncio
async def test_tombstoned_document_is_hidden_from_normal_lookup(
    async_db, document_factory
):
    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    await _publish(async_db, document_id=doc_id, workspace_id=ws, sha="e" * 64)
    await async_db.execute(
        text(
            "UPDATE documents SET source_deleted_at = NOW(), "
            "current_revision_id = NULL WHERE id = :d"
        ),
        {"d": str(doc_id)},
    )
    await async_db.commit()
    assert (
        await async_db.scalar(
            select(Document.id).where(
                Document.id == doc_id, Document.source_deleted_at.is_(None)
            )
        )
    ) is None


# ---------------------------------------------------------------------------
# Sub-resource endpoints: tombstone blocks, current revision serves
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tombstoned_document_subresources_are_not_found(
    async_db, document_factory
):
    """Tombstoning hides markdown/images/chunk-context/download too."""
    from app.api import documents as documents_api
    from app.core.exceptions import NotFoundError

    doc_id = document_factory(content_hash="f" * 64)
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    store = _FakeStore()
    await _publish(
        async_db, document_id=doc_id, workspace_id=ws, sha="f" * 64, store=store
    )
    await DocumentRevisionsRepository(async_db).mark_source_deleted(
        doc_id, reason="document_deleted"
    )
    await async_db.commit()

    for endpoint in (
        documents_api.get_document_markdown,
        documents_api.get_document_images,
        documents_api.get_chunk_context,
        documents_api.download_document,
    ):
        with pytest.raises(NotFoundError):
            await endpoint(doc_id, db=async_db, user=None)


@pytest.mark.asyncio
async def test_markdown_endpoint_serves_the_current_revision(
    async_db, document_factory, monkeypatch
):
    """The viewer endpoint reads the revision artifact, not the v1 mirror."""
    from app.api import documents as documents_api
    from app.services import storage_service

    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    store = _FakeStore()
    await _publish(
        async_db, document_id=doc_id, workspace_id=ws, sha="i" * 64, store=store
    )
    # The legacy mirror points at DIFFERENT content; the endpoint must ignore it.
    store.objects["kb_legacy/legacy.md"] = "# LEGACY MIRROR"
    await async_db.execute(
        text(
            "UPDATE documents SET markdown_s3_key = :k, status = 'indexed' "
            "WHERE id = :d"
        ),
        {"k": "kb_legacy/legacy.md", "d": str(doc_id)},
    )
    await async_db.commit()
    monkeypatch.setattr(storage_service, "get_storage_service", lambda: store)

    response = await documents_api.get_document_markdown(
        doc_id, db=async_db, user=None
    )
    assert response.body.decode() == "# md"
    assert "LEGACY" not in response.body.decode()

    images = await documents_api.get_document_images(
        doc_id, db=async_db, user=None
    )
    assert images == []

    context = await documents_api.get_chunk_context(
        doc_id, db=async_db, user=None
    )
    assert context["total_chunks"] == 2
    assert context["revision_id"] is not None
    assert [c["content"] for c in context["chunks"]] == ["c0", "c1"]

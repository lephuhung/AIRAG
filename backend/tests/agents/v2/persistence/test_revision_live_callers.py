"""Phase 1C Task 5 — revision-selected retrieval through the live callers.

Runs in the full-dependency container (``chromadb``/``numpy``/``fastapi``):
these tests import the retriever, the RAG services, the vector store and the
KG service, none of which are importable in the benchmark venv.

Covers the brief's named tests:
  - ``test_revision_kg_does_not_leak_old_fact``
plus exact retrieval filters (vector + BM25), mixed-revision merge failure,
historical-revision namespace resolution, the unchanged v1 path, and
dimension-safe non-destructive vector writes.
"""

from __future__ import annotations

import uuid

import pytest

# Full-dependency suite: the retriever, the RAG services and the vector store
# import chromadb/numpy. Skip cleanly in the benchmark venv (controller ruling
# R15) so the pure persistence command stays green.
pytest.importorskip("chromadb", reason="revision-selected retrieval needs chromadb")
pytest.importorskip("numpy", reason="legal_kg_service imports the LLM providers")

from app.services.agents.v2.persistence.document_views import (  # noqa: E402
    MixedRevisionMerge,
    RevisionArtifactIdentity,
    embedding_namespace,
    revision_vector_id,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    model_name = "fake-embed"
    dimension = 768

    def embed_query(self, question: str) -> list[float]:
        return [0.0] * self.dimension

    def embed_texts(self, texts):
        return [[0.0] * self.dimension for _ in texts]


class RecordingVectorStore:
    """Records every query so the test can assert the exact Chroma filter."""

    def __init__(self, workspace_id, namespace, hits=None):
        self.workspace_id = workspace_id
        self.collection_name = namespace or f"kb_{workspace_id}"
        self.calls: list[dict] = []
        self._hits = hits or []

    def query(self, query_embedding, n_results, where=None, include=None):
        self.calls.append({"where": where, "n_results": n_results})
        return {
            "ids": [h[0] for h in self._hits],
            "documents": [h[1] for h in self._hits],
            "metadatas": [h[2] for h in self._hits],
            "distances": [0.1 for _ in self._hits],
        }


def _identity(
    revision_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    namespace: str,
    model_hash: str = "hashA",
    dimension: int = 768,
) -> RevisionArtifactIdentity:
    return RevisionArtifactIdentity(
        revision_id=revision_id,
        document_id=document_id,
        generation=1,
        build_profile="FULL",
        markdown_artifact_key=f"kb_x/revisions/{document_id}/{revision_id}/document.md",
        structure_artifact_key=(
            f"kb_x/revisions/{document_id}/{revision_id}/structure.json"
        ),
        embedding_namespace=namespace,
        embedding_model_hash=model_hash,
        embedding_dimension=dimension,
        vector_artifact_version="v1",
    )


def _retriever(workspace_id, store, monkeypatch):
    from app.services.retrieval import deep_retriever as dr

    created: list[RecordingVectorStore] = []

    def _factory(ws_id, namespace=None):
        created.append(RecordingVectorStore(ws_id, namespace))
        return created[-1]

    monkeypatch.setattr(dr, "get_vector_store", _factory)
    retriever = dr.DeepRetriever(
        workspace_id=workspace_id,
        kg_service=None,
        vector_store=store,
        embedder=FakeEmbedder(),
        db=None,
        reranker=object(),
    )
    return retriever, created


# ---------------------------------------------------------------------------
# Exact vector filters
# ---------------------------------------------------------------------------


def test_vector_query_uses_revision_namespace_and_exact_filter(monkeypatch):
    """R2 retrieval reads R2's recorded namespace, filtered to R2 exactly."""
    ws, doc = uuid.uuid4(), uuid.uuid4()
    r1, r2 = uuid.uuid4(), uuid.uuid4()
    r1_ns = embedding_namespace(ws, "hashOLD", 768)
    r2_ns = embedding_namespace(ws, "hashNEW", 1024)

    legacy_store = RecordingVectorStore(ws, None)
    retriever, created = _retriever(ws, legacy_store, monkeypatch)

    identity_r2 = _identity(
        r2, doc, namespace=r2_ns, model_hash="hashNEW", dimension=1024
    )
    retriever._vector_query("câu hỏi", 5, [doc], [identity_r2])

    assert len(created) == 1
    assert created[0].collection_name == r2_ns
    where = created[0].calls[0]["where"]
    assert where == {
        "$and": [
            {"revision_id": {"$in": [str(r2)]}},
            {"document_id": {"$in": [str(doc)]}},
        ]
    }
    # R1 and its namespace are never touched.
    assert str(r1) not in str(where)
    assert r1_ns not in [s.collection_name for s in created]


def test_historical_revision_reads_its_own_namespace(monkeypatch):
    """A historical R1 identity is queried in R1's recorded namespace."""
    ws, doc = uuid.uuid4(), uuid.uuid4()
    r1 = uuid.uuid4()
    r1_ns = embedding_namespace(ws, "hashOLD", 768)
    legacy_store = RecordingVectorStore(ws, None)
    retriever, created = _retriever(ws, legacy_store, monkeypatch)

    retriever._vector_query(
        "câu hỏi", 5, [doc], [_identity(r1, doc, namespace=r1_ns)]
    )
    assert [s.collection_name for s in created] == [r1_ns]
    assert created[0].calls[0]["where"] == {
        "$and": [
            {"revision_id": {"$in": [str(r1)]}},
            {"document_id": {"$in": [str(doc)]}},
        ]
    }


def test_legacy_path_is_unchanged_and_never_revision_filtered(monkeypatch):
    """v1: no revision scope → the legacy document-scoped store, no revision filter."""
    ws, doc = uuid.uuid4(), uuid.uuid4()
    legacy_store = RecordingVectorStore(ws, None)
    retriever, created = _retriever(ws, legacy_store, monkeypatch)

    retriever._vector_query("câu hỏi", 5, [doc], None)
    assert created == []  # no namespace store created
    assert legacy_store.calls[0]["where"] == {"document_id": {"$in": [str(doc)]}}
    assert "revision_id" not in str(legacy_store.calls[0]["where"])


def test_vector_hits_carry_revision_and_stable_vector_id(monkeypatch):
    ws, doc, rev = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    ns = embedding_namespace(ws, "hashA", 768)
    legacy_store = RecordingVectorStore(ws, None)
    retriever, created = _retriever(ws, legacy_store, monkeypatch)

    from app.services.retrieval import deep_retriever as dr

    def _factory(ws_id, namespace=None):
        store = RecordingVectorStore(
            ws_id,
            namespace,
            hits=[
                (
                    revision_vector_id(rev, 0),
                    "nội dung",
                    {
                        "document_id": str(doc),
                        "revision_id": str(rev),
                        "chunk_id": "chunk-uuid-0",
                        "chunk_index": 0,
                        "page_no": 1,
                        "heading_path": "Điều 1",
                    },
                )
            ],
        )
        created.append(store)
        return store

    monkeypatch.setattr(dr, "get_vector_store", _factory)
    chunks, citations = retriever._vector_query(
        "q", 5, [doc], [_identity(rev, doc, namespace=ns)]
    )
    assert chunks[0].revision_id == str(rev)
    assert chunks[0].vector_id == revision_vector_id(rev, 0)
    assert chunks[0].chunk_id == "chunk-uuid-0"
    assert citations[0].document_id == str(doc)


# ---------------------------------------------------------------------------
# Mixed-revision merge failure
# ---------------------------------------------------------------------------


def test_mixed_revision_merge_fails(monkeypatch):
    """A hit from another revision fails the RRF merge instead of leaking."""
    from app.services.models.parsed_document import Citation, EnrichedChunk
    from app.services.retrieval import deep_retriever as dr

    ws = uuid.uuid4()
    r1, r2 = uuid.uuid4(), uuid.uuid4()
    store = RecordingVectorStore(ws, None)
    retriever, _ = _retriever(ws, store, monkeypatch)

    chunk = EnrichedChunk(
        content="R1 only fact",
        chunk_index=0,
        source_file="a.pdf",
        document_id=1,
        revision_id=str(r1),
        vector_id=revision_vector_id(r1, 0),
    )
    citation = Citation(source_file="a.pdf", document_id=1)

    with pytest.raises(MixedRevisionMerge) as exc:
        retriever._rrf_merge([chunk], [citation], [], revision_ids=[r2])
    assert str(r1) in str(exc.value)

    # A BM25-only hit from another revision fails the same way.
    with pytest.raises(MixedRevisionMerge):
        retriever._rrf_merge(
            [],
            [],
            [{"id": revision_vector_id(r1, 1), "metadata": {"revision_id": str(r1)}}],
            revision_ids=[r2],
        )

    # In-scope hits merge normally.
    merged, _ = retriever._rrf_merge(
        [chunk], [citation], [], revision_ids=[r1]
    )
    assert merged == [chunk]


def test_legacy_hit_without_revision_fails_under_a_scope(monkeypatch):
    """Under an explicit scope a legacy (no-provenance) hit is a leak too."""
    from app.services.models.parsed_document import Citation, EnrichedChunk
    from app.services.retrieval import deep_retriever as dr

    ws, r2 = uuid.uuid4(), uuid.uuid4()
    retriever, _ = _retriever(ws, RecordingVectorStore(ws, None), monkeypatch)
    chunk = EnrichedChunk(
        content="legacy", chunk_index=0, source_file="a.pdf", document_id=1
    )
    with pytest.raises(MixedRevisionMerge):
        retriever._rrf_merge(
            [chunk], [Citation(source_file="a.pdf", document_id=1)], [],
            revision_ids=[r2],
        )


# ---------------------------------------------------------------------------
# RAG services honour the selected revision
# ---------------------------------------------------------------------------


def test_rag_service_query_selects_revision_namespace(monkeypatch):
    from app.services.retrieval import rag_service as rs

    ws, doc, rev = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    ns = embedding_namespace(ws, "hashA", 768)
    created: list[RecordingVectorStore] = []

    def _factory(ws_id, namespace=None):
        created.append(RecordingVectorStore(ws_id, namespace))
        return created[-1]

    monkeypatch.setattr(rs, "get_vector_store", _factory)
    service = rs.RAGService.__new__(rs.RAGService)
    service.workspace_id = ws
    service.embedder = FakeEmbedder()
    service.vector_store = RecordingVectorStore(ws, None)

    result = service.query(
        "q", top_k=3, document_ids=[doc], revision_identities=[_identity(rev, doc, namespace=ns)]
    )
    assert [s.collection_name for s in created] == [ns]
    assert created[0].calls[0]["where"] == {
        "$and": [
            {"revision_id": str(rev)},
            {"document_id": {"$in": [str(doc)]}},
        ]
    }
    assert service.vector_store.calls == []  # legacy store never used
    assert result.query == "q"


@pytest.mark.asyncio
async def test_hrag_query_deep_passes_revision_scope(monkeypatch):
    from app.services.retrieval import hrag_service as hs

    recorded: dict = {}

    class FakeRetriever:
        async def query(self, **kwargs):
            recorded.update(kwargs)
            from app.services.models.parsed_document import DeepRetrievalResult

            return DeepRetrievalResult(chunks=[], citations=[], context="", query="q")

    service = hs.HRAGService.__new__(hs.HRAGService)
    service.workspace_id = uuid.uuid4()
    service.retriever = FakeRetriever()

    rev, doc = uuid.uuid4(), uuid.uuid4()
    identity = _identity(rev, doc, namespace=embedding_namespace(service.workspace_id, "hashA", 768))
    await service.query_deep("q", top_k=2, revision_identities=[identity])
    assert recorded["revision_identities"] == [identity]


# ---------------------------------------------------------------------------
# Dimension safety
# ---------------------------------------------------------------------------


class _DimensionMismatchCollection:
    def __init__(self):
        self.add_calls = 0
        self.deleted = False

    def add(self, **kwargs):
        self.add_calls += 1
        raise Exception(
            "Embedding dimension 1024 does not match collection dimensionality 768"
        )

    def delete(self, **kwargs):  # pragma: no cover - must never run
        self.deleted = True


def test_dimension_mismatch_raises_without_deleting_the_collection(monkeypatch):
    """A dimension change must never destroy a collection holding published vectors."""
    from app.services.embedding.vector_store import (
        EmbeddingMigrationRequired,
        VectorStore,
    )

    ws = uuid.uuid4()
    store = VectorStore(ws, namespace=embedding_namespace(ws, "hashA", 768))
    fake = _DimensionMismatchCollection()
    store._collection = fake

    with pytest.raises(EmbeddingMigrationRequired):
        store.add_documents(
            ids=[revision_vector_id(uuid.uuid4(), 0)],
            embeddings=[[0.0] * 1024],
            documents=["x"],
        )
    assert fake.add_calls == 1  # no retry after a destructive recreate
    assert fake.deleted is False
    # The destructive in-place migration path is gone entirely.
    assert not hasattr(VectorStore, "_recreate_collection")


def test_namespaced_store_uses_the_namespace_as_collection_name():
    from app.services.embedding.vector_store import get_vector_store

    ws = uuid.uuid4()
    ns = embedding_namespace(ws, "hashA", 768)
    assert get_vector_store(ws, namespace=ns).collection_name == ns
    assert get_vector_store(ws).collection_name == f"kb_{ws}"


# ---------------------------------------------------------------------------
# KG revision provenance
# ---------------------------------------------------------------------------


class _CapturingResult:
    def __init__(self, rows=None):
        self._rows = rows or []

    async def data(self):
        return self._rows

    async def single(self):
        return {"node_id": "1"} if not self._rows else self._rows[0]

    async def consume(self):
        return None


class _CapturingSession:
    def __init__(self, sink, rows=None):
        self.sink = sink
        self._rows = rows

    async def run(self, cypher, **params):
        self.sink.append((cypher, params))
        return _CapturingResult(self._rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _CapturingDriver:
    def __init__(self, rows=None):
        self.calls: list[tuple[str, dict]] = []
        self._rows = rows

    def session(self):
        return _CapturingSession(self.calls, self._rows)


@pytest.mark.asyncio
async def test_revision_kg_does_not_leak_old_fact():
    """R1's fact is stamped R1; an R2-scoped read can only see R2's scope.

    Neo4j is not reachable from the test environment, so this asserts the two
    mechanisms that make the leak impossible: (1) every node/relation write
    carries the producing revision's ``revision_id``/``revision_ids``
    provenance — including via the ingest-scoped ContextVar that
    ``LegalKGService.ingest`` sets — and (2) ``get_relevant_context`` generates
    a revision-scope predicate over those ownership lists, parameterised with
    the requested revisions.
    """
    from app.services.kg import legal_kg_service as kg

    service = kg.LegalKGService(uuid.uuid4())
    r1, r2 = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()

    # (1a) Explicit revision_id stamps node + relation provenance.
    driver = _CapturingDriver()
    session = driver.session()
    await service._upsert_node(
        session, "Cục Thuế", "Organization", "mô tả", str(doc),
        revision_id=str(r1),
    )
    await service._upsert_relation(
        session, "Cục Thuế", "BAN_HANH", "Nghị định 53/2022", "mô tả", str(doc),
        source_type="Organization", target_type="Document",
        revision_id=str(r1),
    )
    node_cypher, node_params = driver.calls[0]
    rel_cypher, rel_params = driver.calls[1]
    assert "revision_ids" in node_cypher
    assert node_params["revision_id"] == str(r1)
    assert "revision_ids" in rel_cypher
    assert rel_params["revision_id"] == str(r1)

    # (1b) The ingest-scoped ContextVar is what makes every call site scoped
    # without threading the revision through: set it as ingest() does and call
    # the same helper with no explicit revision.
    token = kg._kg_revision_ctx.set(str(r1))
    try:
        driver_scoped = _CapturingDriver()
        await service._upsert_node(
            driver_scoped.session(), "Cục Thuế", "Organization", "mô tả", str(doc)
        )
    finally:
        kg._kg_revision_ctx.reset(token)
    assert driver_scoped.calls[0][1]["revision_id"] == str(r1)

    # (2) An R2-scoped read is scoped by predicate + parameter.
    driver_read = _CapturingDriver(rows=[])
    service._driver = driver_read
    await service.get_relevant_context("nội dung", revision_ids=[r2])
    read_cypher, read_params = driver_read.calls[-1]
    assert "any(rid IN n.revision_ids WHERE rid IN $revision_ids)" in read_cypher
    assert "any(rid IN r.revision_ids WHERE rid IN $revision_ids)" in read_cypher
    assert read_params["revision_ids"] == [str(r2)]
    # R1 is NOT in the R2 scope → R1's fact cannot be returned.
    assert str(r1) not in read_params["revision_ids"]

    # Unscoped (v1) reads keep the pre-Task-5 behaviour: no revision predicate.
    driver_legacy = _CapturingDriver(rows=[])
    service._driver = driver_legacy
    await service.get_relevant_context("nội dung")
    legacy_cypher, legacy_params = driver_legacy.calls[-1]
    assert "$revision_ids" not in legacy_cypher
    assert "revision_ids" not in legacy_params

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
from sqlalchemy import select

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


def _kg_fact_row(r1: uuid.UUID, r2: uuid.UUID, **overrides) -> dict:
    """One fabricated ``get_relevant_context`` row for a shared fact pair.

    ``entity_desc``/``rel_desc`` are the canonical (last-writer) descriptions,
    while ``entity_facts``/``rel_facts`` hold the producing revision's text.
    """
    row = {
        "entity_name": "Cục Thuế",
        "entity_type": "Organization",
        "entity_desc": "R1 fact",  # canonical = last writer (R1)
        "entity_facts": [_fact(r1, "R1 fact"), _fact(r2, "R2 fact")],
        "rel_type": "BAN_HANH",
        "rel_desc": "R1 edge fact",
        "rel_facts": [_fact(r1, "R1 edge fact"), _fact(r2, "R2 edge fact")],
        "rel_src": "Cục Thuế",
        "rel_tgt": "Nghị định 53/2022",
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_revision_kg_scoped_read_does_not_leak_last_writer_fact():
    """Fact TEXT must be per-revision, not last-writer-wins on the shared row.

    A canonical entity/relationship is shared across revisions, so its
    ``description`` is whatever revision wrote last. An R2-scoped read whose
    membership set includes an R1-only fact would return it under the old
    predicate-only implementation (which returned ``n.description`` /
    ``r.description`` verbatim) — this test fails there.
    """
    from app.services.kg import legal_kg_service as kg

    service = kg.LegalKGService(uuid.uuid4())
    r1, r2 = uuid.uuid4(), uuid.uuid4()

    # An R2-scoped read gets R2's fact text, never R1's.
    service._driver = _CapturingDriver(rows=[_kg_fact_row(r1, r2)])
    out_r2 = await service.get_relevant_context("cục thuế", revision_ids=[r2])
    assert "R2 fact" in out_r2
    assert "R1 fact" not in out_r2
    assert "R2 edge fact" in out_r2
    assert "R1 edge fact" not in out_r2

    # A shared row with no in-scope fact yields no text (fail closed) instead
    # of leaking the canonical description another revision wrote.
    service._driver = _CapturingDriver(
        rows=[
            _kg_fact_row(
                r1,
                r2,
                entity_facts=[_fact(r1, "R1 fact")],
                rel_type=None,
                rel_desc=None,
                rel_facts=None,
                rel_src=None,
                rel_tgt=None,
            )
        ]
    )
    out_other = await service.get_relevant_context("cục thuế", revision_ids=[r2])
    assert "R1 fact" not in out_other

    # The unscoped v1 read keeps the canonical description.
    service._driver = _CapturingDriver(rows=[_kg_fact_row(r1, r2)])
    out_v1 = await service.get_relevant_context("cục thuế")
    assert "R1 fact" in out_v1


def _fact(revision_id, text) -> str:
    """Encode one revision fact the way the KG writer stores it.

    Neo4j property values must be primitives or arrays thereof, so a
    ``revision_facts`` entry is the primitive string
    ``"<revision_id><sep><description>"``.
    """
    from app.services.kg.legal_kg_service import _REVISION_FACT_SEP

    return f"{revision_id}{_REVISION_FACT_SEP}{text}"


@pytest.mark.asyncio
async def test_revision_kg_writes_store_fact_text_per_revision():
    """Each ingest stores its fact text under its own revision entry."""
    from app.services.kg import legal_kg_service as kg

    service = kg.LegalKGService(uuid.uuid4())
    r1 = uuid.uuid4()
    doc = uuid.uuid4()
    sep = kg._REVISION_FACT_SEP

    driver = _CapturingDriver()
    await service._upsert_node(
        driver.session(), "Cục Thuế", "Organization", "R1 fact", str(doc),
        revision_id=str(r1),
    )
    await service._upsert_relation(
        driver.session(), "Cục Thuế", "BAN_HANH", "Nghị định 53/2022",
        "R1 edge fact", str(doc),
        source_type="Organization", target_type="Document", revision_id=str(r1),
    )
    node_cypher = driver.calls[0][0]
    rel_cypher = driver.calls[1][0]
    assert "revision_facts" in node_cypher
    assert "revision_facts" in rel_cypher
    # The append is revision-keyed and never overwrites another revision's entry.
    assert f"head(split(f, '{sep}')) <> $revision_id" in node_cypher
    assert f"head(split(f, '{sep}')) <> $revision_id" in rel_cypher
    # Primitive-string encoding on BOTH branches (CREATE and MATCH).
    assert f"[$revision_id + '{sep}' + $description]" in node_cypher
    assert f"[$revision_id + '{sep}' + $desc]" in rel_cypher


@pytest.mark.asyncio
async def test_revision_kg_write_cypher_uses_only_primitive_property_values():
    """Neo4j rejects a map property value — a list of maps made EVERY
    revision-scoped KG write fail at runtime, and ``kg_worker`` swallows an
    ingest failure, so the v2 graph was silently empty while the document
    still reached INDEXED. Every generated write must encode per-revision fact
    text as a primitive string.
    """
    from app.services.kg import legal_kg_service as kg

    service = kg.LegalKGService(uuid.uuid4())
    doc = uuid.uuid4()
    sep = kg._REVISION_FACT_SEP
    driver = _CapturingDriver()
    await service._upsert_document_root(
        driver.session(), "Nghị định 53/2022", "Nghị định 53/2022",
        str(doc), "root fact", revision_id=str(uuid.uuid4()),
    )
    await service._upsert_node(
        driver.session(), "Cục Thuế", "Organization", "fact", str(doc),
        revision_id=str(uuid.uuid4()),
    )
    await service._upsert_relation(
        driver.session(), "Cục Thuế", "BAN_HANH", "Nghị định 53/2022",
        "edge fact", str(doc),
        source_type="Organization", target_type="Document",
        revision_id=str(uuid.uuid4()),
    )

    for cypher, _params in driver.calls:
        assert "revision_facts" in cypher
        # A map literal is not a legal Neo4j property value.
        assert "{revision_id:" not in cypher
        assert "revision_id:" not in cypher
        # The primitive string encoding is used on both branches.
        assert f"[$revision_id + '{sep}' +" in cypher
        assert f"head(split(f, '{sep}')) <> $revision_id" in cypher


@pytest.mark.asyncio
async def test_revision_kg_fact_round_trip_against_live_neo4j():
    """Live Neo4j: the primitive encoder is written and read back per revision.

    This is the test that catches the class of error the ``revision_facts``
    list-of-maps property had: Neo4j rejects it at *statement execution*, so
    only a real write can prove the encoding is accepted. Runs on an isolated
    random workspace label (no interference with real graphs) and skips when
    Neo4j is unreachable (e.g. the benchmark venv).
    """
    from app.services.kg import legal_kg_service as kg

    ws = uuid.uuid4()
    r1, r2 = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    service = kg.LegalKGService(ws)
    driver = None
    try:
        try:
            probe = await service._get_driver()
            await probe.verify_connectivity()
        except Exception as exc:  # pragma: no cover - env dependent
            # Nothing to clean up: the probe never connected (and the driver
            # is lazy, so no session was opened).
            await service.cleanup()
            pytest.skip(f"Neo4j unreachable: {exc}")
        driver = probe

        async def _write(revision_id, fact, edge_fact):
            # Same ContextVar wiring ingest() uses.
            token = kg._kg_revision_ctx.set(str(revision_id))
            try:
                async with driver.session() as session:
                    await service._upsert_document_root(
                        session, "Nghị định 9999/2099", "Nghị định 9999/2099",
                        str(doc), "root fact",
                    )
                    await service._upsert_node(
                        session, "Cục Thuế Live Probe", "Organization", fact,
                        str(doc),
                    )
                    await service._upsert_relation(
                        session, "Cục Thuế Live Probe", "BAN_HANH",
                        "Nghị định 9999/2099", edge_fact, str(doc),
                        source_type="Organization", target_type="Document",
                    )
            finally:
                kg._kg_revision_ctx.reset(token)

        await _write(r1, "R1 live fact", "R1 live edge")
        await _write(r2, "R2 live fact", "R2 live edge")

        out_r1 = await service.get_relevant_context(
            "cục thuế live probe", revision_ids=[r1]
        )
        out_r2 = await service.get_relevant_context(
            "cục thuế live probe", revision_ids=[r2]
        )
        assert "R1 live fact" in out_r1
        assert "R2 live fact" not in out_r1
        assert "R2 live fact" in out_r2
        assert "R1 live fact" not in out_r2
        assert "R1 live edge" in out_r1
        assert "R2 live edge" not in out_r1
        assert "R2 live edge" in out_r2
    finally:
        if driver is not None:
            try:
                async with driver.session() as session:
                    await session.run(
                        f"MATCH (n:`{service._label}`) DETACH DELETE n"
                    )
            finally:
                await service.cleanup()


# ---------------------------------------------------------------------------
# Recency boost reads the revision's recorded namespace
# ---------------------------------------------------------------------------


class _FakeChromaCollection:
    def __init__(self) -> None:
        self.get_calls: list[list[str]] = []

    def get(self, ids, include=None):
        self.get_calls.append(list(ids))
        return {
            "ids": list(ids),
            "metadatas": [{"published_date": "01/01/2026"} for _ in ids],
        }


def test_recency_boost_reads_the_revision_namespace(monkeypatch):
    """A revision-qualified chunk id must be resolved in its own namespace.

    ``self.vector_store`` is the legacy ``kb_<workspace>`` collection; a
    revision-qualified id (``rev_<rev>_chunk_<n>``) can never be found there,
    so the boost would always miss (and lazily create an empty legacy
    collection).
    """
    from app.services.models.parsed_document import Citation, EnrichedChunk
    from app.services.retrieval import deep_retriever as dr

    ws, doc, rev = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    ns = embedding_namespace(ws, "hashA", 768)
    created: list[RecordingVectorStore] = []

    def _factory(ws_id, namespace=None):
        store = RecordingVectorStore(ws_id, namespace)
        store.collection = _FakeChromaCollection()
        created.append(store)
        return store

    class _ExplodingLegacyCollection:
        def get(self, *args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("legacy collection must not be read")

    class _ExplodingLegacyStore:
        def __init__(self):
            self.collection = _ExplodingLegacyCollection()

    monkeypatch.setattr(dr, "get_vector_store", _factory)
    retriever = dr.DeepRetriever(
        workspace_id=ws,
        kg_service=None,
        vector_store=_ExplodingLegacyStore(),
        embedder=FakeEmbedder(),
        db=None,
        reranker=object(),
    )
    chunk = EnrichedChunk(
        content="x",
        chunk_index=0,
        source_file="a.pdf",
        document_id=doc,
        revision_id=str(rev),
        vector_id=revision_vector_id(rev, 0),
        score=0.5,
    )
    citations = [Citation(source_file="a.pdf", document_id=doc)]
    identity = _identity(rev, doc, namespace=ns)

    boosted, _ = retriever._apply_recency_boost([chunk], citations, [identity])

    assert [s.collection_name for s in created] == [ns]
    assert created[0].collection.get_calls == [[revision_vector_id(rev, 0)]]
    assert boosted[0].score >= 0.5


# ---------------------------------------------------------------------------
# Agent read tools are revision-selected
# ---------------------------------------------------------------------------


class _ToolFakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    async def upload_artifact(self, key, content, content_type) -> str:
        self.objects[key] = content
        return key

    async def download_markdown(self, key) -> str:
        return self.objects[key]


async def _publish_current_revision(async_db, document_id, workspace_id, storage):
    """Publish one FULL revision whose markdown artifact is distinct from v1."""
    from app.models.document import Document  # noqa: F401
    from app.services.agents.v2.persistence.document_revisions import (
        DocumentRevisionsRepository,
    )
    from app.services.agents.v2.persistence.document_views import (
        ChunkRecord,
        build_structure_artifact,
        record_revision_chunk_rows,
        revision_markdown_key,
        revision_structure_key,
    )
    from app.services.agents.v2.persistence.source_identity import (
        RevisionBuildProfile,
        compute_source_object_identity,
    )

    repo = DocumentRevisionsRepository(async_db)
    revision = await repo.allocate_draft(
        document_id,
        compute_source_object_identity(
            bucket="b",
            object_key=f"kb_{workspace_id}/doc_{document_id}.pdf",
            version_id=None,
            etag="tool-etag",
            size_bytes=1,
            content_sha256="a" * 64,
        ),
        RevisionBuildProfile.FULL,
    )
    rid = revision.revision_id
    md_key = revision_markdown_key(workspace_id, document_id, rid)
    struct_key = revision_structure_key(workspace_id, document_id, rid)
    chunks = [
        ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            ordinal=0,
            content="revision chunk",
            page_no=1,
            heading_path=["Điều 1"],
        )
    ]
    await storage.upload_artifact(md_key, "# REVISION MARKDOWN", "text/markdown")
    await storage.upload_artifact(
        struct_key,
        build_structure_artifact(rid, document_id, chunks),
        "application/json",
    )
    await record_revision_chunk_rows(async_db, rid, chunks)
    await repo.record_artifacts(
        rid,
        RevisionBuildProfile.FULL,
        embedding_namespace=embedding_namespace(workspace_id, "hashA", 768),
        embedding_model_hash="hashA",
        embedding_dimension=768,
        vector_artifact_version="v1",
        markdown_artifact_key=md_key,
        structure_artifact_key=struct_key,
    )
    await repo.verify_draft(rid)
    await repo.publish(rid)
    await async_db.commit()
    return rid


@pytest.mark.asyncio
async def test_summarize_document_reads_the_current_revision(
    async_db, document_factory, monkeypatch
):
    from sqlalchemy import text

    from app.models.document import Document
    from app.services import llm as llm_module
    from app.services import storage_service
    from app.services.agent import tools

    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = _ToolFakeStore()
    await _publish_current_revision(async_db, doc_id, ws, storage)
    storage.objects["kb_legacy/legacy.md"] = "# LEGACY MIRROR"
    await async_db.execute(
        text(
            "UPDATE documents SET markdown_s3_key = :k, status = 'indexed' "
            "WHERE id = :d"
        ),
        {"k": "kb_legacy/legacy.md", "d": str(doc_id)},
    )
    await async_db.commit()
    monkeypatch.setattr(storage_service, "get_storage_service", lambda: storage)

    captured: dict = {}

    class _FakeLLM:
        async def acomplete(self, messages, **kwargs):
            captured["prompt"] = messages[0].content
            return "TÓM TẮT"

    monkeypatch.setattr(llm_module, "get_llm_provider", lambda: _FakeLLM())

    result = await tools.summarize_document(doc_id, async_db)

    assert result["text"] == "TÓM TẮT"
    assert "REVISION MARKDOWN" in captured["prompt"]
    assert "LEGACY MIRROR" not in captured["prompt"]


@pytest.mark.asyncio
async def test_get_documents_content_reads_the_current_revision(
    async_db, document_factory, monkeypatch
):
    from sqlalchemy import text

    from app.models.document import Document
    from app.services import storage_service
    from app.services.agent import tools

    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = _ToolFakeStore()
    await _publish_current_revision(async_db, doc_id, ws, storage)
    storage.objects["kb_legacy/legacy.md"] = "# LEGACY MIRROR"
    await async_db.execute(
        text(
            "UPDATE documents SET markdown_s3_key = :k, status = 'indexed' "
            "WHERE id = :d"
        ),
        {"k": "kb_legacy/legacy.md", "d": str(doc_id)},
    )
    await async_db.commit()
    monkeypatch.setattr(storage_service, "get_storage_service", lambda: storage)

    result = await tools.get_documents_content([doc_id], async_db)

    assert result["documents"][0]["content"] == "# REVISION MARKDOWN"


@pytest.mark.asyncio
async def test_agent_read_tools_hide_tombstoned_documents(
    async_db, document_factory, monkeypatch
):
    from app.models.document import Document
    from app.services import storage_service
    from app.services.agent import tools
    from app.services.agents.v2.persistence.document_revisions import (
        DocumentRevisionsRepository,
    )

    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = _ToolFakeStore()
    await _publish_current_revision(async_db, doc_id, ws, storage)
    from sqlalchemy import text

    await async_db.execute(
        text("UPDATE documents SET status = 'indexed' WHERE id = :d"),
        {"d": str(doc_id)},
    )
    await async_db.commit()
    listed = await tools.list_documents([ws], async_db)
    assert listed["document_count"] == 1

    await DocumentRevisionsRepository(async_db).mark_source_deleted(
        doc_id, reason="document_deleted"
    )
    await async_db.commit()
    monkeypatch.setattr(storage_service, "get_storage_service", lambda: storage)

    assert (await tools.list_documents([ws], async_db))["document_count"] == 0

    summary = await tools.summarize_document(doc_id, async_db)
    assert "Không tìm thấy" in summary["text"]

    content = await tools.get_documents_content([doc_id], async_db)
    assert content["documents"][0]["content"] is None
    assert content["documents"][0]["error"] is not None

    from app.core import database as db_module

    class _SessionCtx:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            return self._session

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        db_module, "async_session_maker", lambda: _SessionCtx(async_db)
    )
    section = await tools.search_document_section(
        "Điều 1", [str(ws)], [str(doc_id)]
    )
    assert section["sources"] == []


@pytest.mark.asyncio
async def test_search_document_section_uses_the_current_revision(
    async_db, document_factory, monkeypatch
):
    from app.models.document import Document
    from app.services.agent import tools
    from app.core import database as db_module
    from app.services.embedding import vector_store as vs_module

    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    storage = _ToolFakeStore()
    rid = await _publish_current_revision(async_db, doc_id, ws, storage)

    created: list = []

    class _FakeVStore:
        def __init__(self, ws_id, namespace):
            self.workspace_id = ws_id
            self.collection_name = namespace or f"kb_{ws_id}"
            self.queries: list = []

        def get_by_metadata(self, where=None):
            self.queries.append(where)
            return {
                "documents": ["nội dung Điều 1"],
                "metadatas": [
                    {
                        "document_id": str(doc_id),
                        "heading_path": "Điều 1",
                        "page_no": 1,
                        "chunk_index": 0,
                        "revision_id": str(rid),
                        "chunk_id": "chunk-1",
                    }
                ],
            }

        def query(self, **kwargs):  # pragma: no cover - metadata hit short-circuits
            return {"documents": [], "metadatas": []}

    def _factory(ws_id, namespace=None):
        store = _FakeVStore(ws_id, namespace)
        created.append(store)
        return store

    monkeypatch.setattr(vs_module, "get_vector_store", _factory)

    class _SessionCtx:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            return self._session

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(db_module, "async_session_maker", lambda: _SessionCtx(async_db))

    result = await tools.search_document_section(
        "Điều 1", [str(ws)], [str(doc_id)]
    )

    assert result["sources"]
    assert len(created) == 1
    assert created[0].collection_name == embedding_namespace(ws, "hashA", 768)
    assert created[0].queries == [
        {"$and": [{"document_id": str(doc_id)}, {"revision_id": str(rid)}]}
    ]


@pytest.mark.asyncio
async def test_search_document_section_fails_closed_on_unready_revision(
    async_db, document_factory, monkeypatch
):
    """A current pointer that is not published must not fall back to the
    legacy document-scoped store (nor raise): the section search returns no
    sources."""
    from sqlalchemy import text

    from app.core import database as db_module
    from app.models.document import Document
    from app.services.agent import tools
    from app.services.agents.v2.persistence.document_revisions import (
        DocumentRevisionsRepository,
    )
    from app.services.agents.v2.persistence.source_identity import (
        RevisionBuildProfile,
        compute_source_object_identity,
    )

    doc_id = document_factory()
    ws = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    draft = await DocumentRevisionsRepository(async_db).allocate_draft(
        doc_id,
        compute_source_object_identity(
            bucket="b",
            object_key=f"kb_{ws}/doc_{doc_id}.pdf",
            version_id=None,
            etag="draft-etag",
            size_bytes=1,
            content_sha256="b" * 64,
        ),
        RevisionBuildProfile.FULL,
    )
    await async_db.execute(
        text("UPDATE documents SET current_revision_id = :r WHERE id = :d"),
        {"r": str(draft.revision_id), "d": str(doc_id)},
    )
    await async_db.commit()

    class _SessionCtx:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            return self._session

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        db_module, "async_session_maker", lambda: _SessionCtx(async_db)
    )

    result = await tools.search_document_section(
        "Điều 1", [str(ws)], [str(doc_id)]
    )
    assert result["sources"] == []

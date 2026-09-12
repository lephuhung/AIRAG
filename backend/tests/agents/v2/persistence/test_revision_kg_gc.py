"""Phase 1D Task 9 — revision-scoped KG GC.

``LegalKGService.delete_revision_artifacts(document_id, revision_id)`` must
delete only rows carrying the producing ``revision_id`` and must never
blind-``DETACH DELETE`` a MERGE-shared canonical entity: an entity/relationship
is pruned only when it has no remaining revision membership (and no remaining
other-document ownership). It returns the deleted count and is idempotent.

Neo4j is unreachable from the benchmark venv, so the always-running tests assert
the generated Cypher's semantics (the same technique Task 5 uses for the
revision-scoped read). ``test_revision_kg_gc_preserves_shared_entities`` runs
against a live Neo4j when the driver is installed and reachable, and skips
otherwise (e.g. the benchmark venv); the R15 full-dependency container runs it.
"""

from __future__ import annotations

import types
import uuid

import pytest

# ``legal_kg_service`` imports the LLM providers (numpy/aioboto3/...), which the
# benchmark venv does not carry. Skip the whole module there, exactly as
# ``test_revision_live_callers.py`` does; the R15 full-dependency container
# exercises it (including the live-Neo4j test).
pytest.importorskip("numpy", reason="legal_kg_service imports the LLM providers")

from app.services.kg import legal_kg_service as kg


class _Summary:
    def __init__(self, nodes: int = 0, rels: int = 0) -> None:
        self.counters = types.SimpleNamespace(
            nodes_deleted=nodes, relationships_deleted=rels
        )


class _Result:
    def __init__(self, summary: _Summary) -> None:
        self._summary = summary

    async def consume(self) -> _Summary:
        return self._summary


class _Session:
    def __init__(self, calls, *, nodes: int = 0, rels: int = 0) -> None:
        self.calls = calls
        self._nodes = nodes
        self._rels = rels

    async def run(self, cypher, **params) -> _Result:
        self.calls.append((cypher, params))
        # First statement is the relationship pass, second the node pass.
        is_rel = "-[r]" in cypher
        return _Result(_Summary(rels=self._rels if is_rel else 0,
                                nodes=self._nodes if not is_rel else 0))

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _CapturingDriver:
    def __init__(self, *, nodes: int = 0, rels: int = 0) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._nodes = nodes
        self._rels = rels

    def session(self) -> _Session:
        return _Session(self.calls, nodes=self._nodes, rels=self._rels)


async def _run_capture(node_deleted=0, rel_deleted=0):
    service = kg.LegalKGService(uuid.uuid4())
    driver = _CapturingDriver(nodes=node_deleted, rels=rel_deleted)
    service._driver = driver
    document_id, revision_id = uuid.uuid4(), uuid.uuid4()
    count = await service.delete_revision_artifacts(document_id, revision_id)
    return count, driver.calls, document_id, revision_id


# ---------------------------------------------------------------------------
# Generated-Cypher contract (always runs)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revision_kg_gc_scopes_deletes_to_the_producing_revision():
    count, calls, document_id, revision_id = await _run_capture()

    assert len(calls) == 2  # relationships first, then nodes
    for cypher, params in calls:
        assert "$revision_id IN coalesce(r.revision_ids, [])" in cypher or (
            "$revision_id IN coalesce(n.revision_ids, [])" in cypher
        )
        assert params["revision_id"] == str(revision_id)
        assert params["document_id"] == str(document_id)
    # Rows that never carried the revision (legacy/v1 rows with no
    # revision_ids) are never matched.
    assert all("revision_ids, [])" in cypher for cypher, _ in calls)


@pytest.mark.asyncio
async def test_revision_kg_gc_removes_membership_and_revision_facts_only():
    _count, calls, _document_id, revision_id = await _run_capture()
    rel_cypher, node_cypher = calls[0][0], calls[1][0]

    for cypher in (rel_cypher, node_cypher):
        # The reclaimed revision is dropped from the ownership list ...
        assert "[rid IN coalesce" in cypher and "WHERE rid <> $revision_id]" in cypher
        # ... and its per-revision fact text is dropped.
        assert "revision_facts" in cypher
        assert "head(split(f," in cypher and "<> $revision_id]" in cypher
    assert f"${'revision_id'}" in rel_cypher


@pytest.mark.asyncio
async def test_revision_kg_gc_only_prunes_entities_with_no_remaining_revision():
    """A shared canonical entity (another revision references it) survives."""
    _count, calls, _document_id, _revision_id = await _run_capture()
    node_cypher = calls[1][0]

    # Prune only when the ownership list became empty AND no other document
    # still owns the entity — never an unconditional DETACH DELETE.
    assert "WHERE size(remaining) = 0 AND size(other_docs) = 0" in node_cypher
    assert "DETACH DELETE n" in node_cypher
    # The guard is *before* the delete clause, not a bare DETACH DELETE.
    assert node_cypher.index("size(other_docs) = 0") < node_cypher.index(
        "DETACH DELETE n"
    )
    # A shared entity keeps its remaining memberships unchanged.
    assert "ELSE n.document_ids END" in node_cypher


@pytest.mark.asyncio
async def test_revision_kg_gc_returns_deleted_node_and_relation_count():
    count, calls, _document_id, _revision_id = await _run_capture(
        node_deleted=2, rel_deleted=3
    )
    assert count == 5
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_revision_kg_gc_second_call_returns_zero():
    """Idempotent: a second delete for the same revision returns 0."""
    service = kg.LegalKGService(uuid.uuid4())
    service._driver = _CapturingDriver(nodes=0, rels=0)
    document_id, revision_id = uuid.uuid4(), uuid.uuid4()
    first = await service.delete_revision_artifacts(document_id, revision_id)
    second = await service.delete_revision_artifacts(document_id, revision_id)
    assert first == 0 and second == 0


# ---------------------------------------------------------------------------
# Live Neo4j behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revision_kg_gc_preserves_shared_entities():
    """Live Neo4j: R1 GC keeps an entity R2 still references.

    R1 and R2 both reference canonical entity E; GC of R1 removes only R1-scoped
    facts/edges/membership, keeps E (R2 still references it) and leaves R2's
    fact intact, prunes an entity that was R1-only, and a second run deletes
    zero rows. Skips when the neo4j driver is unavailable (benchmark venv) or
    Neo4j is unreachable.
    """
    neo4j = pytest.importorskip("neo4j", reason="KG GC live test needs the neo4j driver")

    from app.core.config import settings

    workspace_id = uuid.uuid4()
    document_id = uuid.uuid4()
    r1, r2 = uuid.uuid4(), uuid.uuid4()
    sep = kg._REVISION_FACT_SEP
    service = kg.LegalKGService(workspace_id)
    try:
        try:
            driver = await service._get_driver()
            await driver.verify_connectivity()
        except Exception as exc:  # pragma: no cover - env dependent
            await service.cleanup()
            pytest.skip(f"Neo4j unreachable: {exc}")

        async def _write(revision_id: uuid.UUID, *, r1_only: bool = False) -> None:
            # Same ContextVar wiring ingest() uses.
            token = kg._kg_revision_ctx.set(str(revision_id))
            try:
                async with driver.session() as session:
                    await service._upsert_node(
                        session, "Cục Thuế GC Probe", "Organization",
                        "R-N fact", str(document_id),
                    )
                    if r1_only:
                        await service._upsert_node(
                            session, "Chỉ R1 GC Probe", "Organization",
                            "R1-only fact", str(document_id),
                        )
                    # The relation MATCHes its endpoints, so both must exist.
                    await service._upsert_node(
                        session, "Nghị định 9998/2099", "Document",
                        "target", str(document_id),
                    )
                    await service._upsert_relation(
                        session, "Cục Thuế GC Probe", "BAN_HANH",
                        "Nghị định 9998/2099", "R-N edge", str(document_id),
                        source_type="Organization", target_type="Document",
                    )
            finally:
                kg._kg_revision_ctx.reset(token)

        async def _read(cypher, **params):
            async with driver.session() as session:
                result = await session.run(cypher, **params)
                return [dict(record) async for record in result]

        try:
            await _write(r1, r1_only=True)
            await _write(r2)

            shared_before = await _read(
                f"MATCH (n:`{service._label}` {{display_name: 'Cục Thuế GC Probe'}}) "
                "RETURN n.revision_ids AS revision_ids, n.revision_facts AS facts"
            )
            assert shared_before, "probe entity was not written"
            assert {str(r1), str(r2)} <= set(shared_before[0]["revision_ids"])

            deleted = await service.delete_revision_artifacts(document_id, r1)

            shared_after = await _read(
                f"MATCH (n:`{service._label}` {{display_name: 'Cục Thuế GC Probe'}}) "
                "RETURN n.revision_ids AS revision_ids, n.revision_facts AS facts"
            )
            assert shared_after, "shared entity E was deleted by R1 GC"
            assert str(r2) in shared_after[0]["revision_ids"]
            assert str(r1) not in shared_after[0]["revision_ids"]
            facts = shared_after[0]["facts"]
            assert f"{r2}{sep}R-N fact" in facts
            assert f"{r1}{sep}R-N fact" not in facts
            # R1-only entity is pruned.
            assert await _read(
                f"MATCH (n:`{service._label}` {{display_name: 'Chỉ R1 GC Probe'}}) "
                "RETURN n"
            ) == []
            # The shared edge keeps R2's provenance.
            edges = await _read(
                f"MATCH ()-[r:BAN_HANH]->() "
                f"WHERE r.description = 'R-N edge' AND $r2 IN r.revision_ids "
                "RETURN r.revision_ids AS revision_ids",
                r2=str(r2),
            )
            assert edges and str(r2) in edges[0]["revision_ids"]

            # Idempotent: a second run deletes nothing.
            assert await service.delete_revision_artifacts(document_id, r1) == 0
            assert deleted >= 1
        finally:
            async with driver.session() as session:
                await session.run(
                    f"MATCH (n:`{service._label}`) DETACH DELETE n"
                )
    finally:
        await service.cleanup()

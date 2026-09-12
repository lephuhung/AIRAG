"""Phase 1 final-review I1 — lightrag KG must satisfy the revision interface.

``KnowledgeGraphService`` (``HRAG_KG_MODE=lightrag``, non-default) keeps one
workspace-level graph and cannot provide per-revision provenance. The v2
callers still pass the revision kwargs — ``kg_worker`` calls
``ingest(..., revision_id=...)``, ``deep_retriever`` calls
``get_relevant_context(..., revision_ids=...)``, revision GC calls
``delete_revision_artifacts(...)``. The service must accept and ignore them
instead of raising ``TypeError``: ``kg_worker`` swallows an ingest failure, so
the v2 graph would be silently empty, and Predicate-B GC would stall.

Needs ``numpy`` (imported by the service); the benchmark venv lacks it, so this
module skips there and runs in the full-dependency container.
"""

from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy", reason="KnowledgeGraphService imports numpy")

from app.services.kg import knowledge_graph_service as kg_module
from app.services.kg.knowledge_graph_service import KnowledgeGraphService


@pytest.fixture(autouse=True)
def _reset_warning_flag():
    kg_module._revision_kwargs_warned.clear()
    yield
    kg_module._revision_kwargs_warned.clear()


class _Graph:
    async def get_all_nodes(self):
        return [{"id": "1"}]


class _StubRag:
    def __init__(self) -> None:
        self.inserted: list[str] = []
        self.chunk_entity_relation_graph = _Graph()

    async def ainsert(self, content: str) -> None:
        self.inserted.append(content)


@pytest.mark.asyncio
async def test_lightrag_ingest_accepts_and_ignores_revision_id(
    monkeypatch, caplog
):
    service = KnowledgeGraphService(uuid.uuid4())
    stub = _StubRag()

    async def _get_rag():
        return stub

    monkeypatch.setattr(service, "_get_rag", _get_rag)

    with caplog.at_level(logging.WARNING, logger=kg_module.__name__):
        await service.ingest("nội dung", revision_id=uuid.uuid4())
        await service.ingest("nội dung 2", revision_id=uuid.uuid4())

    assert stub.inserted == ["nội dung", "nội dung 2"]
    warnings = [
        record for record in caplog.records if "revision-scoped" in record.getMessage()
    ]
    assert len(warnings) == 1  # one-time per operation kind


@pytest.mark.asyncio
async def test_lightrag_query_accepts_and_ignores_revision_ids(monkeypatch):
    service = KnowledgeGraphService(uuid.uuid4())
    monkeypatch.setattr(
        kg_module, "settings", SimpleNamespace(HRAG_KG_GRAPH_BACKEND="networkx")
    )
    calls: list[tuple] = []

    async def _networkx(keywords, max_entities, max_relationships):
        calls.append((keywords, max_entities, max_relationships))
        return "ctx"

    monkeypatch.setattr(service, "_get_relevant_context_networkx", _networkx)

    out = await service.get_relevant_context("cục thuế", revision_ids=[uuid.uuid4()])
    assert out == "ctx"
    assert calls


@pytest.mark.asyncio
async def test_lightrag_delete_revision_artifacts_is_a_zero_noop():
    service = KnowledgeGraphService(uuid.uuid4())
    assert (
        await service.delete_revision_artifacts(uuid.uuid4(), uuid.uuid4())
    ) == 0

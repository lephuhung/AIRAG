"""Phase 1 final-review M2 — agent tools must pass the caller's workspace_id.

``search_document_section`` resolves caller-named ``document_ids`` through
``resolve_document_targets``. It used to omit ``workspace_id``, so the
eligibility check could never reject a document outside the requested
workspace (``api/rag.py`` passes it) and a target's revision could be paired
with another workspace's vector store.

Runs in the full-dependency container (``chromadb`` via the embedding
services); skips in the benchmark venv.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("chromadb", reason="embedding vector store needs chromadb")

from app.services.agents.v2.persistence import document_views as dv
from app.services.agent import tools as tools_module  # noqa: E402


class _FakeStore:
    collection_name = "fake"

    def get_by_metadata(self, where=None):
        return {"documents": [], "metadatas": []}

    def query(self, **kwargs):  # pragma: no cover - no chunks to fall back to
        return {"documents": [], "metadatas": []}


class _FakeSession:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_search_document_section_resolves_targets_per_workspace(monkeypatch):
    import app.core.database as database_module
    import app.services.embedding.vector_store as vector_store_module

    ws_a, ws_b = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    seen_workspaces: list[uuid.UUID] = []

    async def fake_resolve(_db, document_ids, *, workspace_id=None):
        seen_workspaces.append(workspace_id)
        return [
            dv.DocumentRetrievalTarget(
                document_id=doc, identity=None, eligible=False
            )
        ]

    monkeypatch.setattr(dv, "resolve_document_targets", fake_resolve)
    monkeypatch.setattr(database_module, "async_session_maker", _FakeSession)
    monkeypatch.setattr(
        vector_store_module,
        "get_vector_store",
        lambda ws_id, namespace=None: _FakeStore(),
    )

    result = await tools_module.search_document_section(
        section_reference="Điều 5",
        workspace_ids=[str(ws_a), str(ws_b)],
        document_ids=[str(doc)],
    )

    # Resolved once per requested workspace, each scoped to its own workspace.
    assert seen_workspaces == [ws_a, ws_b]
    assert result["sources"] == []

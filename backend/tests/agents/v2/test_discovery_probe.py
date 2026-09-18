"""Regression tests for ``_discover_workspace_document_ids`` (P1 recall fix).

The discovery contract is ``top_k`` DISTINCT pinnable documents, but Chroma
returns top_k CHUNKS. Two starvation modes are pinned here:

- a dominant document fills the whole chunk window (all top-K chunks belong
  to document A → document B at rank K+1 was previously invisible);
- tombstoned/unpublished hits consume window slots and are dropped
  post-fetch without backfill.

The fix over-fetches chunks, filters to pinnable document ids, and grows
the window (bounded) until enough distinct documents are found or the
workspace's hits are exhausted.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.services.agents.supervisor_v2 import _discover_workspace_document_ids


class _Result:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def all(self) -> list[tuple]:
        return self._rows


class _DB:
    """Fake session: ``execute`` returns the pinnable (doc_id, ws) rows."""

    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    async def execute(self, _stmt: Any) -> _Result:
        return _Result(self._rows)


class _Store:
    """Fake vector store returning a fixed ranked hit list, windowed."""

    def __init__(self, metadatas: list[dict], distances: list[float]) -> None:
        self._metadatas = metadatas
        self._distances = distances
        self.n_results_seen: list[int] = []

    def query(self, *, query_embedding, n_results, include):
        self.n_results_seen.append(n_results)
        return {
            "metadatas": self._metadatas[:n_results],
            "distances": self._distances[:n_results],
        }


def _patch_store(monkeypatch: pytest.MonkeyPatch, store: _Store) -> None:
    monkeypatch.setattr(
        "app.services.embedding.vector_store.get_vector_store",
        lambda _ws: store,
    )


async def _embed(_query: str) -> list[float]:
    return [0.0]


@pytest.mark.asyncio
async def test_discovery_recovers_document_beyond_chunk_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All top-K chunks belong to doc A; doc B sits at rank K+1.

    With chunk-level n_results=top_k the old code returned only A. The
    over-fetch window must surface B.
    """
    ws, doc_a, doc_b = uuid4(), uuid4(), uuid4()
    db = _DB([(doc_a, ws), (doc_b, ws)])
    metadatas = [{"document_id": str(doc_a)}] * 8 + [
        {"document_id": str(doc_b)}
    ]
    distances = [0.01 * i for i in range(len(metadatas))]
    store = _Store(metadatas, distances)
    _patch_store(monkeypatch, store)

    ids = await _discover_workspace_document_ids(
        db, "q", 2, (ws,), embed=_embed
    )

    assert ids == [doc_a, doc_b]
    # The first probe must over-fetch beyond top_k chunks.
    assert store.n_results_seen[0] > 2


@pytest.mark.asyncio
async def test_discovery_backfills_past_tombstoned_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tombstoned/unpublished hits are dropped and the window grows to
    backfill with pinnable documents instead of returning a shortlist."""
    ws, doc_a, doc_b, tomb = uuid4(), uuid4(), uuid4(), uuid4()
    # Only A and B are pinnable; tomb has no current revision.
    db = _DB([(doc_a, ws), (doc_b, ws)])
    metadatas = (
        [{"document_id": str(doc_a)}]
        + [{"document_id": str(tomb)}] * 6
        + [{"document_id": str(doc_b)}]
    )
    distances = [0.01 * i for i in range(len(metadatas))]
    store = _Store(metadatas, distances)
    _patch_store(monkeypatch, store)

    ids = await _discover_workspace_document_ids(
        db, "q", 2, (ws,), embed=_embed
    )

    assert ids == [doc_a, doc_b]
    assert tomb not in ids


@pytest.mark.asyncio
async def test_discovery_stops_when_hits_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A workspace with fewer hits than the window returns what it has —
    no infinite growth."""
    ws, doc_a = uuid4(), uuid4()
    db = _DB([(doc_a, ws)])
    metadatas = [{"document_id": str(doc_a)}] * 3
    distances = [0.01 * i for i in range(3)]
    store = _Store(metadatas, distances)
    _patch_store(monkeypatch, store)

    ids = await _discover_workspace_document_ids(
        db, "q", 5, (ws,), embed=_embed
    )

    assert ids == [doc_a]
    # One probe only: len(raw)=3 < window → no re-query.
    assert len(store.n_results_seen) == 1


@pytest.mark.asyncio
async def test_discovery_skips_workspace_without_pinnable_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A workspace whose documents are all tombstoned is never probed."""
    ws, tomb = uuid4(), uuid4()
    db = _DB([])  # no pinnable rows at all
    store = _Store([{"document_id": str(tomb)}], [0.0])
    _patch_store(monkeypatch, store)

    ids = await _discover_workspace_document_ids(
        db, "q", 5, (ws,), embed=_embed
    )

    assert ids == []
    assert store.n_results_seen == []

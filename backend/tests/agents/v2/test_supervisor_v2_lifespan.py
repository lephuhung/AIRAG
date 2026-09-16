"""Task 6 — supervisor v2 lifespan: one saver, v1 default, singleton helpers.

Red-phase first: imports ``supervisor_v2_lifespan`` and the singleton
helpers from ``supervisor_v2``, which do not exist yet.
"""
from __future__ import annotations

import logging
import sys
import types
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from langgraph.checkpoint.memory import InMemorySaver

from app.services.agents import supervisor_v2
from app.services.agents.supervisor_v2 import (
    create_supervisor_v2_graph,
    get_supervisor_v2_graph,
    reset_supervisor_v2_graph,
    supervisor_v2_lifespan,
)


class FakeSaverContext:
    """Counts saver opens/closes around a real in-memory saver."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.saver = InMemorySaver()

    async def __aenter__(self) -> InMemorySaver:
        self._events.append("open")
        return self.saver

    async def __aexit__(self, *args: Any) -> None:
        self._events.append("close")


def test_import_builds_no_graph() -> None:
    reset_supervisor_v2_graph()
    # Importing the module must not construct or cache any graph.
    import importlib

    importlib.reload(supervisor_v2)
    with pytest.raises(RuntimeError):
        get_supervisor_v2_graph()
    reset_supervisor_v2_graph()


def test_singleton_set_get_reset() -> None:
    reset_supervisor_v2_graph()
    try:
        with pytest.raises(RuntimeError):
            get_supervisor_v2_graph()
        graph = create_supervisor_v2_graph(None)  # type: ignore[arg-type]
        from app.services.agents.supervisor_v2 import set_supervisor_v2_graph

        set_supervisor_v2_graph(graph)
        assert get_supervisor_v2_graph() is graph
    finally:
        reset_supervisor_v2_graph()


@pytest.mark.asyncio
async def test_lifespan_opens_exactly_one_saver_context() -> None:
    from contextlib import asynccontextmanager

    reset_supervisor_v2_graph()
    events: list[str] = []

    @asynccontextmanager
    async def factory(dsn: str):
        assert dsn == "postgresql://user:pass@host:5432/checkpoints"
        cm = FakeSaverContext(events)
        async with cm as saver:
            yield saver

    try:
        async with supervisor_v2_lifespan(
            "postgresql://user:pass@host:5432/checkpoints",
            checkpointer_factory=factory,  # type: ignore[arg-type]
        ) as graph:
            assert events == ["open"], f"expected exactly one opened saver, got {events}"
            assert get_supervisor_v2_graph() is graph
        assert events == ["open", "close"]
        # Singleton is released with the lifespan.
        with pytest.raises(RuntimeError):
            get_supervisor_v2_graph()
    finally:
        reset_supervisor_v2_graph()


def test_lifespan_helper_never_migrates() -> None:
    import inspect

    source = inspect.getsource(supervisor_v2_lifespan)
    assert "setup(" not in source
    assert "migrate" not in source.lower()


@pytest.mark.asyncio
async def test_lifespan_failure_propagates_for_outer_guard() -> None:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def failing_factory(dsn: str):
        raise ConnectionError("checkpoint db unreachable")
        yield  # pragma: no cover

    reset_supervisor_v2_graph()
    try:
        # The helper fails fast; main.py's lifespan catches it so v1 stays up.
        with pytest.raises(ConnectionError):
            async with supervisor_v2_lifespan(
                "postgresql://user:pass@host:5432/checkpoints",
                checkpointer_factory=failing_factory,  # type: ignore[arg-type]
            ):
                pass  # pragma: no cover
        with pytest.raises(RuntimeError):
            get_supervisor_v2_graph()
    finally:
        reset_supervisor_v2_graph()


def test_main_lifespan_wiring_pins() -> None:
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[3] / "app" / "main.py").read_text()
    assert "supervisor_v2_lifespan" in source
    assert "setup_v2_checkpointer" not in source
    assert "saver.setup" not in source
    assert source.count("create_v2_checkpointer") <= 1
    # M5 (round 2): the stack is a real context manager around the yield,
    # so cancellation at the yield still closes the saver context.
    assert "AsyncExitStack" in source
    # NEW-I2: BOTH v2 imports sit inside a try in lifespan — an import
    # failure degrades to v1 instead of aborting web startup. AST pin
    # (a text pin cannot see guard structure).
    tree = ast.parse(source)
    lifespan = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )

    def _guarded(import_node: ast.AST) -> bool:
        parent: ast.AST | None = getattr(import_node, "_parent", None)
        while parent is not None and parent is not lifespan:
            if isinstance(parent, ast.Try):
                return True
            parent = getattr(parent, "_parent", None)
        return False

    for node in ast.walk(lifespan):
        for child in ast.iter_child_nodes(node):
            setattr(child, "_parent", node)
    v2_imports = [
        node
        for node in ast.walk(lifespan)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(
            (alias.name == "AsyncExitStack" or "supervisor_v2" in (getattr(alias, "name", "") or ""))
            for alias in node.names
        )
    ]
    assert v2_imports, "expected v2 imports inside lifespan"
    assert all(_guarded(node) for node in v2_imports), (
        "v2 imports must be inside try (v1-degradation invariant)"
    )


# ── NEW-I2a behavioral pins (round 4) ─────────────────────────────────────
# The AST pin above proves the v2 imports sit inside a ``try``; it CANNOT
# prove the lifespan ``yield`` is reachable on the import-failure path (the
# round-3 defect: ``yield`` inside the ``try``'s ``else:``). These tests
# drive the REAL ``app.main.lifespan`` with every pre-v2 side effect
# stubbed and assert the v1-degradation invariant behaviorally: startup
# enters AND exits cleanly — no ``RuntimeError: generator didn't yield``
# from ``@asynccontextmanager`` — with ``supervisor_v2_ready`` False, the
# v1 readiness gate still executed, and v1 teardown (engine dispose) run.


class _StubURLEndpoint:
    def render_as_string(self, hide_password: bool = False) -> str:
        return "postgresql+asyncpg://user:pass@host:5432/hrag"


class _StubEngine:
    """Stands in for ``app.main.engine``: records v1 teardown."""

    def __init__(self) -> None:
        self.url = _StubURLEndpoint()
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


def _install_main_lifespan_stubs(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Make the real ``app.main.lifespan`` run IO-free up to the v2 block.

    Returns {"engine": _StubEngine, "readiness_calls": list}.
    """
    import app.main

    monkeypatch.setenv("AUTO_CREATE_TABLES", "false")
    monkeypatch.setenv("WEB_CONCURRENCY", "1")

    engine = _StubEngine()
    monkeypatch.setattr(app.main, "engine", engine)

    readiness_calls: list = []

    def _fake_readiness(check: Any) -> None:
        readiness_calls.append(check)

    monkeypatch.setattr(app.main, "assert_v2_readiness", _fake_readiness)

    class _SyncEngine:
        def dispose(self) -> None:
            pass

    def _attach(name: str, **attrs: Any) -> None:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)

    async def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    def _noop_sync(*args: Any, **kwargs: Any) -> None:
        return None

    _attach(
        "app.services.agents.v2.persistence.migrate",
        make_engine=lambda dsn: _SyncEngine(),
        check_v2_schema=lambda eng: object(),
    )

    class _Storage:
        async def ensure_bucket(self) -> None:
            return None

        async def ensure_uploads_bucket(self) -> None:
            return None

    _attach("app.services.storage_service", get_storage_service=lambda: _Storage())
    _attach("app.services.models.loader", preload_models=_noop_sync)
    _attach("app.services.memory.graphiti_client", initialize_graphiti=_noop)
    _attach(
        "app.services.people.mongo_client",
        get_mongo_client=lambda: SimpleNamespace(
            admin=SimpleNamespace(command=lambda *a, **k: {"ok": 1})
        ),
        close_mongo_client=_noop_sync,
    )
    _attach(
        "app.services.runtime_config",
        refresh_snapshot=_make_async_true(),
        ensure_default_connections=_make_async_empty_list(),
    )
    _attach(
        "app.core.redis_client",
        ping_redis=_make_async_false(),
        close_redis=_noop,
    )
    return {"engine": engine, "readiness_calls": readiness_calls}


def _make_async_true() -> Any:
    async def _inner(*args: Any, **kwargs: Any) -> bool:
        return True

    return _inner


def _make_async_false() -> Any:
    async def _inner(*args: Any, **kwargs: Any) -> bool:
        return False

    return _inner


def _make_async_empty_list() -> Any:
    async def _inner(*args: Any, **kwargs: Any) -> list:
        return []

    return _inner


@pytest.mark.asyncio
async def test_main_lifespan_v2_import_failure_still_serves_v1(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Forced v2 import failure must enter AND exit lifespan (no RuntimeError)."""
    stubs = _install_main_lifespan_stubs(monkeypatch)
    # Force ``from app.services.agents.supervisor_v2 import ...`` to raise
    # ImportError, as a missing/older v2-only dependency would.
    monkeypatch.setitem(sys.modules, "app.services.agents.supervisor_v2", None)
    import app.main
    from fastapi import FastAPI

    probe = FastAPI()
    with caplog.at_level(logging.WARNING, logger="app.main"):
        async with app.main.lifespan(probe):
            # Startup entered: v2 unavailable, v1 gate still executed.
            assert probe.state.supervisor_v2_ready is False
            assert len(stubs["readiness_calls"]) == 1
    # Clean exit: v1 teardown ran (no "generator didn't yield").
    assert stubs["engine"].disposed is True
    assert "v1 default unaffected" in caplog.text


@pytest.mark.asyncio
async def test_main_lifespan_saver_open_failure_yields_with_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saver-open failure must still reach the lifespan yield (v1 serves)."""
    stubs = _install_main_lifespan_stubs(monkeypatch)

    @asynccontextmanager
    async def _failing_factory(dsn: str) -> Any:
        raise ConnectionError("checkpoint db unreachable")
        yield  # pragma: no cover

    mod = types.ModuleType("app.services.agents.supervisor_v2")
    mod.supervisor_v2_lifespan = _failing_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.services.agents.supervisor_v2", mod)
    import app.main
    from fastapi import FastAPI

    probe = FastAPI()
    async with app.main.lifespan(probe):
        assert probe.state.supervisor_v2_ready is False
    assert stubs["engine"].disposed is True


@pytest.mark.asyncio
async def test_main_lifespan_saver_close_failure_still_tears_down(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A saver-close failure must be contained: teardown (engine dispose) runs."""
    stubs = _install_main_lifespan_stubs(monkeypatch)
    events: list[str] = []

    @asynccontextmanager
    async def _close_failing_factory(dsn: str) -> Any:
        events.append("open")
        try:
            yield object()
        finally:
            events.append("close")
            raise RuntimeError("saver close failed")

    mod = types.ModuleType("app.services.agents.supervisor_v2")
    mod.supervisor_v2_lifespan = _close_failing_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.services.agents.supervisor_v2", mod)
    import app.main
    from fastapi import FastAPI

    probe = FastAPI()
    with caplog.at_level(logging.WARNING, logger="app.main"):
        async with app.main.lifespan(probe):
            assert probe.state.supervisor_v2_ready is True
    assert events == ["open", "close"]
    assert "scope teardown failed" in caplog.text
    assert stubs["engine"].disposed is True


# ── P0 Task 5: live revision-manifest retrieval wiring (RED phase) ──────────
# Proves the request-scoped ``V1RevisionAwareRetrievalService`` loads the exact
# per-revision manifest identity, queries only that manifest's namespace with
# hard document/revision filters, and fails closed (never a current-config
# namespace fallback) on missing/incompatible manifests, foreign workspaces,
# tombstones, stale pins, and malformed provider output.


TASK5_WS = __import__("uuid").UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
TASK5_OTHER_WS = __import__("uuid").UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
TASK5_DOC = __import__("uuid").UUID("11111111-1111-1111-1111-111111111111")
TASK5_FOREIGN_DOC = __import__("uuid").UUID("22222222-2222-2222-2222-222222222222")
TASK5_REV = __import__("uuid").UUID("33333333-3333-3333-3333-333333333333")
TASK5_NS = f"ws_{TASK5_WS}_embed_hashA_d768"


def _task5_identity(**overrides: Any) -> Any:
    from app.services.agents.v2.persistence import document_views as _dv

    fields: dict[str, Any] = {
        "revision_id": TASK5_REV,
        "document_id": TASK5_DOC,
        "generation": 1,
        "build_profile": "FULL",
        "markdown_artifact_key": "kb/rev/doc.md",
        "structure_artifact_key": "kb/rev/structure.json",
        "embedding_namespace": TASK5_NS,
        "embedding_model_hash": "hashA",
        "embedding_dimension": 768,
        "vector_artifact_version": "v1",
    }
    fields.update(overrides)
    return _dv.RevisionArtifactIdentity(**fields)


def _task5_target(
    document_id: Any = None, revision: str | None = None
) -> Any:
    from app.services.agents.v2.capabilities import ResolvedTarget
    from app.services.agents.v2.contracts.binding import ScopedDocument
    from app.services.agents.v2.contracts.locators import DocumentLocator
    from app.services.agents.v2.contracts.planning import TargetUnit

    return ResolvedTarget(
        target_unit=TargetUnit(
            target_id="t1",
            binding_id="b1",
            requested_locator=DocumentLocator(kind="document"),
            completion_criteria=(),
        ),
        document=ScopedDocument(
            binding_id="b1",
            document_id=document_id or TASK5_DOC,
            document_revision=revision or str(TASK5_REV),
            role="target",
        ),
    )


class _Task5Session:
    """Stub async session context manager (manifest loaders are patched)."""

    def __init__(self, db: Any) -> None:
        self._db = db

    async def __aenter__(self) -> Any:
        return self._db

    async def __aexit__(self, *args: object) -> None:
        return None


class _OwnerMapResult:
    """Minimal execute() result: only .all() is consumed by owner lookups."""

    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return self._rows


class _OwnerMapDB:
    """Stub db: owner-map queries resolve every known doc to its workspace.

    The service batches ``Document.id → Document.workspace_id`` resolution
    into one ``db.execute``; extra rows are harmless because only queried
    ids are looked up. TASK5_FOREIGN_DOC maps to the in-scope workspace so
    the patched scoped loader still rejects it (same probe as before).
    """

    def __init__(self, rows: list) -> None:
        self._rows = rows

    async def execute(self, *args: Any, **kwargs: Any) -> _OwnerMapResult:
        return _OwnerMapResult(self._rows)


_TASK5_OWNER_DB = _OwnerMapDB(
    [(TASK5_DOC, TASK5_WS), (TASK5_FOREIGN_DOC, TASK5_WS)]
)


def _task5_session_factory(db: Any = None) -> Any:
    sentinel: Any = _TASK5_OWNER_DB if db is None else db

    def _open() -> _Task5Session:
        return _Task5Session(sentinel)

    return _open

class _Task5Provider:
    """Injectable embed/namespace-query/rerank ports with call recording."""

    def __init__(self, hits: list[dict] | None = None) -> None:
        self._hits = hits if hits is not None else []
        self.embed_calls: list[str] = []
        self.query_calls: list[dict] = []
        self.rerank_calls: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [0.1, 0.2, 0.3]

    async def query_namespace(
        self,
        namespace: str,
        embedding: list[float],
        n_results: int,
        where: dict,
    ) -> dict:
        self.query_calls.append(
            {"namespace": namespace, "n_results": n_results, "where": where}
        )
        return {
            "ids": [h["id"] for h in self._hits],
            "documents": [h["content"] for h in self._hits],
            "metadatas": [h["metadata"] for h in self._hits],
            "distances": [h.get("distance", 0.1) for h in self._hits],
        }

    async def rerank(
        self, query: str, texts: list[str]
    ) -> list[tuple[int, float]]:
        self.rerank_calls.append(query)
        return [(i, 0.9 - i * 0.01) for i in range(len(texts))]


def _task5_hit(
    *,
    vector_id: str | None = None,
    content: str = "pinned chunk text",
    document_id: Any = None,
    revision_id: Any = None,
    workspace_id: Any = None,
    chunk_id: str = "chunk-1",
) -> dict:
    revision_id = revision_id or TASK5_REV
    return {
        "id": vector_id or f"rev_{revision_id}_chunk_0",
        "content": content,
        "metadata": {
            "document_id": str(document_id or TASK5_DOC),
            "workspace_id": str(workspace_id or TASK5_WS),
            "revision_id": str(revision_id),
            "chunk_id": chunk_id,
            "ordinal": 0,
        },
    }


def _task5_service(provider: _Task5Provider, **kwargs: Any) -> Any:
    return supervisor_v2.V1RevisionAwareRetrievalService(
        session_factory=_task5_session_factory(),
        embed_query=provider.embed_query,
        query_namespace=provider.query_namespace,
        rerank=provider.rerank,
        **kwargs,
    )


def _patch_task5_manifests(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scoped: Any = None,
    scoped_error: BaseException | None = None,
    current: Any = None,
) -> dict:
    """Patch the workspace-scoped manifest loaders; forbid the unscoped one."""
    from app.services.agents.v2.persistence import document_views as _dv

    calls: dict[str, list] = {"scoped": [], "current": [], "unscoped": []}

    async def _scoped(
        db: Any, revision_id: Any, workspace_id: Any, **kwargs: Any
    ) -> Any:
        calls["scoped"].append((revision_id, workspace_id))
        if scoped_error is not None:
            raise scoped_error
        return scoped

    async def _current(
        db: Any, document_id: Any, workspace_id: Any, **kwargs: Any
    ) -> Any:
        calls["current"].append((document_id, workspace_id))
        if isinstance(current, BaseException):
            raise current
        if callable(current):
            return current(document_id, workspace_id)
        return current

    async def _unscoped(*args: Any, **kwargs: Any) -> Any:
        calls["unscoped"].append(args)
        raise AssertionError(
            "unscoped manifest lookup must not be used by the live service"
        )

    monkeypatch.setattr(_dv, "load_revision_identity_for_workspace", _scoped)
    monkeypatch.setattr(
        _dv, "load_current_revision_identity_for_workspace", _current
    )
    monkeypatch.setattr(_dv, "load_revision_identity", _unscoped)
    return calls


@pytest.mark.asyncio
async def test_task5_service_queries_exact_manifest_namespace_with_hard_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Task5Provider(hits=[_task5_hit()])
    service = _task5_service(provider)
    _patch_task5_manifests(monkeypatch, scoped=_task5_identity())

    chunks = await service.retrieve(
        "what does the rule say",
        top_k=8,
        allowed_targets=(_task5_target(),),
        workspace_ids=(TASK5_WS,),
    )

    assert provider.embed_calls == ["what does the rule say"]
    assert len(provider.query_calls) == 1
    call = provider.query_calls[0]
    assert call["namespace"] == TASK5_NS
    where = call["where"]
    assert str(TASK5_REV) in str(where)
    assert str(TASK5_DOC) in str(where)
    assert len(chunks) == 1
    assert chunks[0].document_id == TASK5_DOC
    assert chunks[0].document_revision == str(TASK5_REV)
    assert chunks[0].content == "pinned chunk text"


@pytest.mark.asyncio
async def test_task5_service_fail_closed_on_missing_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.agents.v2.persistence import document_views as _dv

    provider = _Task5Provider(hits=[_task5_hit()])
    service = _task5_service(provider)
    _patch_task5_manifests(
        monkeypatch,
        scoped_error=_dv.RevisionNotReady(TASK5_DOC, "no manifest"),
    )

    with pytest.raises(supervisor_v2.V1ServiceUnavailable):
        await service.retrieve(
            "query",
            top_k=8,
            allowed_targets=(_task5_target(),),
            workspace_ids=(TASK5_WS,),
        )
    # No fallback to a current-config namespace: the provider is never touched.
    assert provider.query_calls == []


@pytest.mark.asyncio
async def test_task5_service_fail_closed_on_incompatible_vector_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Task5Provider(hits=[_task5_hit()])
    service = _task5_service(provider)
    _patch_task5_manifests(
        monkeypatch, scoped=_task5_identity(vector_artifact_version="v0")
    )

    with pytest.raises(supervisor_v2.V1ServiceUnavailable):
        await service.retrieve(
            "query",
            top_k=8,
            allowed_targets=(_task5_target(),),
            workspace_ids=(TASK5_WS,),
        )
    assert provider.query_calls == []


@pytest.mark.asyncio
async def test_task5_service_fail_closed_on_stale_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Task5Provider(hits=[_task5_hit()])
    service = _task5_service(provider)
    # Manifest belongs to a different document than the pinned target.
    _patch_task5_manifests(
        monkeypatch, scoped=_task5_identity(document_id=TASK5_FOREIGN_DOC)
    )

    with pytest.raises(supervisor_v2.V1ServiceUnavailable):
        await service.retrieve(
            "query",
            top_k=8,
            allowed_targets=(_task5_target(),),
            workspace_ids=(TASK5_WS,),
        )
    assert provider.query_calls == []


@pytest.mark.asyncio
async def test_task5_service_drops_foreign_and_malformed_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID as _UUID

    other_rev = _UUID("44444444-4444-4444-4444-444444444444")
    # Every rejected hit carries a DISTINCT dedupe key
    # ``(revision_id, document_id, chunk_id)`` (see ``retrieve``) and distinct
    # content, so a rejection is observable only through its own ``_coerce_hit``
    # guard: dedupe can never hide a leaked hit, and the content assertion can
    # never mistake a survivor for the admitted chunk.
    provider = _Task5Provider(
        hits=[
            _task5_hit(),  # admitted (chunk-1 / "pinned chunk text")
            _task5_hit(
                revision_id=other_rev,
                chunk_id="chunk-stale",
                content="stale revision chunk",
            ),  # stale revision → drop
            _task5_hit(
                document_id=TASK5_FOREIGN_DOC,
                chunk_id="chunk-foreign-doc",
                content="foreign document chunk",
            ),  # foreign doc → drop
            _task5_hit(
                workspace_id=TASK5_OTHER_WS,
                chunk_id="chunk-foreign-ws",
                content="foreign workspace chunk",
            ),  # foreign ws → drop (only the workspace check can drop it)
            _task5_hit(
                content="   ", chunk_id="chunk-blank"
            ),  # blank → drop
            _task5_hit(
                vector_id=f"doc_{TASK5_DOC}_chunk_0",
                chunk_id="chunk-legacy",
                content="legacy vector id chunk",
            ),  # legacy id shape → drop (metadata matches, so only the
            # ``parse_revision_vector_id`` shape check can drop it)
        ]
    )
    service = _task5_service(provider)
    _patch_task5_manifests(monkeypatch, scoped=_task5_identity())

    chunks = await service.retrieve(
        "query",
        top_k=8,
        allowed_targets=(_task5_target(),),
        workspace_ids=(TASK5_WS,),
    )

    # Exact admitted count AND content: removing any single ``_coerce_hit``
    # guard admits an extra distinctly-identified chunk and fails here.
    assert len(chunks) == 1
    assert [c.content for c in chunks] == ["pinned chunk text"]
    assert chunks[0].locator.start == "chunk-1"
    assert chunks[0].locator.end == "chunk-1"


@pytest.mark.asyncio
async def test_task5_service_unscoped_skips_unresolvable_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.agents.v2.persistence import document_views as _dv

    async def _discover(
        query: str, top_k: int, workspace_ids: tuple, db: Any
    ) -> list:
        assert list(workspace_ids) == [TASK5_WS]
        return [TASK5_DOC, TASK5_FOREIGN_DOC]

    provider = _Task5Provider(hits=[_task5_hit()])
    service = _task5_service(provider, discover=_discover)

    def _current(document_id: Any, workspace_id: Any) -> Any:
        if document_id == TASK5_DOC:
            return _task5_identity()
        raise _dv.RevisionNotReady(document_id, "foreign workspace")

    calls = _patch_task5_manifests(monkeypatch, current=_current)

    chunks = await service.retrieve(
        "open question", top_k=8, allowed_targets=(), workspace_ids=(TASK5_WS,)
    )

    assert [c.document_id for c in chunks] == [TASK5_DOC]
    # Only the owned identity's namespace is ever queried.
    assert [c["namespace"] for c in provider.query_calls] == [TASK5_NS]
    assert calls["unscoped"] == []


@pytest.mark.asyncio
async def test_task5_default_namespace_query_uses_manifest_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production default query path passes the manifest namespace through."""
    from app.services.embedding import vector_store as _vs

    provider = _Task5Provider()
    # No ``query_namespace`` port → the production
    # ``_default_revision_namespace_query`` default is exercised.
    service = supervisor_v2.V1RevisionAwareRetrievalService(
        session_factory=_task5_session_factory(),
        embed_query=provider.embed_query,
        rerank=provider.rerank,
    )
    _patch_task5_manifests(monkeypatch, scoped=_task5_identity())

    store_calls: list[dict] = []

    class _FakeStore:
        def query(
            self,
            *,
            query_embedding: list[float],
            n_results: int,
            where: dict | None = None,
        ) -> dict:
            store_calls.append(
                {"n_results": n_results, "where": where}
            )
            return {
                "ids": [f"rev_{TASK5_REV}_chunk_0"],
                "documents": ["pinned chunk text"],
                "metadatas": [
                    {
                        "document_id": str(TASK5_DOC),
                        "workspace_id": str(TASK5_WS),
                        "revision_id": str(TASK5_REV),
                        "chunk_id": "chunk-1",
                        "ordinal": 0,
                    }
                ],
                "distances": [0.1],
            }

    seen: list[tuple] = []

    def _fake_get_store(workspace_id: Any, namespace: Any = None) -> Any:
        seen.append((workspace_id, namespace))
        return _FakeStore()

    monkeypatch.setattr(_vs, "get_vector_store", _fake_get_store)

    chunks = await service.retrieve(
        "query",
        top_k=8,
        allowed_targets=(_task5_target(),),
        workspace_ids=(TASK5_WS,),
    )

    # The default path reached the real vector-store seam (not the injected
    # provider port) with the EXACT manifest namespace; the workspace is
    # derived from that namespace, and the hard filters travel with it.
    assert provider.query_calls == []
    assert seen == [(TASK5_WS, TASK5_NS)]
    assert len(store_calls) == 1
    assert str(TASK5_REV) in str(store_calls[0]["where"])
    assert str(TASK5_DOC) in str(store_calls[0]["where"])
    assert len(chunks) == 1
    assert chunks[0].content == "pinned chunk text"


@pytest.mark.asyncio
async def test_task5_default_namespace_query_rejects_unqualified_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy/unqualified namespaces fail closed before any store IO."""
    from app.services.embedding import vector_store as _vs

    def _forbidden_store(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "an unqualified namespace must never reach the vector store"
        )

    monkeypatch.setattr(_vs, "get_vector_store", _forbidden_store)

    # Direct seam: the legacy ``kb_<workspace>`` (current-config) collection
    # name and a bare unqualified name both raise.
    with pytest.raises(supervisor_v2.V1ServiceUnavailable):
        await supervisor_v2._default_revision_namespace_query(
            f"kb_{TASK5_WS}", [0.1, 0.2, 0.3], 8, {"$and": []}
        )
    with pytest.raises(supervisor_v2.V1ServiceUnavailable):
        await supervisor_v2._default_revision_namespace_query(
            "not-a-namespace", [0.1, 0.2, 0.3], 8, {"$and": []}
        )

    # Service level: a manifest carrying a legacy namespace fails closed
    # through the same default path with zero provider/store calls.
    provider = _Task5Provider(hits=[_task5_hit()])
    service = supervisor_v2.V1RevisionAwareRetrievalService(
        session_factory=_task5_session_factory(),
        embed_query=provider.embed_query,
        rerank=provider.rerank,
    )
    _patch_task5_manifests(
        monkeypatch,
        scoped=_task5_identity(embedding_namespace=f"kb_{TASK5_WS}"),
    )
    with pytest.raises(supervisor_v2.V1ServiceUnavailable):
        await service.retrieve(
            "query",
            top_k=8,
            allowed_targets=(_task5_target(),),
            workspace_ids=(TASK5_WS,),
        )
    assert provider.query_calls == []


def test_task5_probe_reports_revision_retrieval_gate_by_default() -> None:
    """Default probe path: the vector-store backing imports → gate present."""
    assert "v1-revision-retrieval" in supervisor_v2.probe_v1_services()


def test_task5_probe_omits_revision_retrieval_gate_when_backing_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe default path fails closed when the vector-store seam is gone."""
    real_v1_attr = supervisor_v2._v1_attr

    def _missing_backing(module_name: str, attr: str) -> Any:
        if (module_name, attr) == (
            "app.services.embedding.vector_store",
            "get_vector_store",
        ):
            raise supervisor_v2.V1ServiceUnavailable(
                "app.services.embedding.vector_store is not importable"
            )
        return real_v1_attr(module_name, attr)

    monkeypatch.setattr(supervisor_v2, "_v1_attr", _missing_backing)
    assert "v1-revision-retrieval" not in supervisor_v2.probe_v1_services()


def test_task5_registry_includes_document_retrieve_only_when_service_live() -> None:
    import uuid as _uuid_module
    from datetime import datetime, timezone

    from app.services.agents.v2.capabilities import CapabilityUnavailable
    from app.services.agents.v2.contracts.capability import (
        CapabilityRuntimeContext,
    )

    runtime = CapabilityRuntimeContext(
        request_id="req-task5",
        run_id="run-task5",
        user_id=_uuid_module.uuid4(),
        workspace_ids=(TASK5_WS,),
        can_read_people=False,
        allowed_capabilities=frozenset({"document.retrieve"}),
        deadline_at=datetime.now(timezone.utc),
    )

    class _FakeRetrieval:
        pass

    live_bundle = supervisor_v2.V1ServiceBundle(
        session_factory=_task5_session_factory(),
        document_retrieval=_FakeRetrieval(),
    )
    live = supervisor_v2.build_v2_capability_registry(
        runtime,
        bundle=live_bundle,
        evidence=object(),
        resolver=object(),
        available_services=frozenset({"v1-revision-retrieval"}),
    )
    assert "document.retrieve" in live.capability_names()

    gated_out = supervisor_v2.build_v2_capability_registry(
        runtime,
        bundle=live_bundle,
        evidence=object(),
        resolver=object(),
        available_services=frozenset(),
    )
    assert "document.retrieve" not in gated_out.capability_names()
    with pytest.raises(CapabilityUnavailable):
        gated_out.get("document.retrieve")

    # Missing persistence seams gate the capability out even when flagged live.
    no_seams = supervisor_v2.build_v2_capability_registry(
        runtime,
        bundle=live_bundle,
        evidence=None,
        resolver=None,
        available_services=frozenset({"v1-revision-retrieval"}),
    )
    assert "document.retrieve" not in no_seams.capability_names()


def test_task5_default_allowed_capabilities_include_document_retrieve() -> None:
    from app.services.agent.runtime_selector import (
        DEFAULT_V2_ALLOWED_CAPABILITIES,
    )

    assert "document.retrieve" in DEFAULT_V2_ALLOWED_CAPABILITIES


@pytest.mark.asyncio
async def test_task5_document_search_pins_manifests_in_workspace_scope_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.agents.v2.persistence import document_views as _dv

    async def _search(
        query: str,
        top_k: int,
        workspace_ids: list,
        existing: set,
        db: Any,
    ) -> dict:
        return {
            "sources": [
                {"document_id": str(TASK5_DOC)},
                {"document_id": str(TASK5_FOREIGN_DOC)},
            ]
        }

    service = supervisor_v2.V1DocumentSearchService(
        search=_search, session_factory=_task5_session_factory()
    )

    seen: list[tuple] = []

    async def _scoped_current(
        db: Any, document_id: Any, workspace_id: Any, **kwargs: Any
    ) -> Any:
        seen.append((document_id, workspace_id))
        if document_id == TASK5_DOC and workspace_id == TASK5_WS:
            return _task5_identity()
        raise _dv.RevisionNotReady(document_id, "not in workspace")

    async def _forbid_unscoped(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("discovery must use the workspace-scoped lookup")

    monkeypatch.setattr(
        _dv,
        "load_current_revision_identity_for_workspace",
        _scoped_current,
    )
    monkeypatch.setattr(_dv, "load_current_revision_identity", _forbid_unscoped)

    candidates = await service.search("query", None, (TASK5_WS,))

    assert [c.document_id for c in candidates] == [TASK5_DOC]
    assert (TASK5_FOREIGN_DOC, TASK5_WS) in seen


@pytest.mark.asyncio
async def test_task5_registry_capability_executes_through_real_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry → capability → live service executes with zero new dispatch."""
    import uuid as _uuid_module
    from datetime import datetime, timezone

    from app.services.agents.v2.contracts.capability import (
        CapabilityRuntimeContext,
        DocumentRetrieveInput,
    )
    from app.services.agents.v2.contracts.evidence import EvidenceUseRef
    from app.services.agents.v2.contracts.execution import AgentRequest

    provider = _Task5Provider(hits=[_task5_hit()])
    service = _task5_service(provider)
    _patch_task5_manifests(monkeypatch, scoped=_task5_identity())

    pinned = _task5_target()

    class _Resolver:
        def resolve(self, target_id: str) -> Any:
            return pinned if target_id == "t1" else None

    class _Evidence:
        def __init__(self) -> None:
            self.uses: list[dict] = []

        async def persist_use(self, **kwargs: Any) -> Any:
            self.uses.append(kwargs)
            return EvidenceUseRef(use_id=_uuid_module.uuid4())

    evidence = _Evidence()
    runtime = CapabilityRuntimeContext(
        request_id="req-task5-e2e",
        run_id="run-task5-e2e",
        user_id=_uuid_module.uuid4(),
        workspace_ids=(TASK5_WS,),
        can_read_people=False,
        allowed_capabilities=frozenset({"document.retrieve"}),
        deadline_at=datetime.now(timezone.utc),
    )
    registry = supervisor_v2.build_v2_capability_registry(
        runtime,
        bundle=supervisor_v2.V1ServiceBundle(
            session_factory=_task5_session_factory(),
            document_retrieval=service,
        ),
        evidence=evidence,
        resolver=_Resolver(),
        available_services=frozenset({"v1-revision-retrieval"}),
    )
    capability = registry.get("document.retrieve")
    result = await capability.execute(
        AgentRequest(
            contract_version="2.0",
            task_id="T1",
            objective="answer the factual question",
            input=DocumentRetrieveInput(
                kind="document.retrieve", query="what changed", target_ids=("t1",)
            ),
        ),
        runtime,
    )

    assert result.status == "success"
    assert result.data.retrieved_unit_count == 1
    assert len(result.evidence_uses) == 1
    assert evidence.uses[0]["purpose"] == "coverage"
    assert evidence.uses[0]["target_id"] == "t1"
    assert "pinned chunk text" not in result.model_dump_json()

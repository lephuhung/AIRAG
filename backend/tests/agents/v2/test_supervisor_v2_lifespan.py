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

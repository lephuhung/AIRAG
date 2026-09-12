"""Task 6 — supervisor v2 lifespan: one saver, v1 default, singleton helpers.

Red-phase first: imports ``supervisor_v2_lifespan`` and the singleton
helpers from ``supervisor_v2``, which do not exist yet.
"""
from __future__ import annotations

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

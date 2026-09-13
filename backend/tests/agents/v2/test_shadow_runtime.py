"""Task 6 — side-effect-free v2 shadow execution (R54/R55/R56/R57).

Red-phase first: imports ``shadow_runtime`` and ``shadow_checkpoint``,
which do not exist yet, so collection must fail until the implementation
lands. All runs use fakes + an isolated ``InMemorySaver`` (no database,
no network); the only production imports under test are the frozen
contract layer plus ``create_supervisor_v2_graph``.
"""
from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")

PRODUCTION_TABLES = (
    "checkpoint",
    "evidence",
    "evidence_use",
    "audit",
    "chat",
    "memory",
    "title",
)


def make_production_rows() -> dict[str, dict[str, Any]]:
    """Sentinel production rows; the shadow run must leave them untouched."""
    return {
        table: {f"{table}-row-1": {"id": f"{table}-row-1", "n": 1}}
        for table in PRODUCTION_TABLES
    }


def snapshot(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return copy.deepcopy(rows)


@pytest.mark.asyncio
async def test_shadow_never_calls_production_graph_resolver() -> None:
    """R54: shadow compiles its own graph; production getter stays silent."""
    import app.services.agents.supervisor_v2 as supervisor_v2
    from app.services.agent.shadow_runtime import build_shadow_bundle

    calls: list[str] = []
    real_getter = supervisor_v2.get_supervisor_v2_graph

    def _spy() -> Any:
        calls.append("get_supervisor_v2_graph")
        return real_getter()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(supervisor_v2, "get_supervisor_v2_graph", _spy)
    try:
        bundle = build_shadow_bundle(raw_query="Xin chào", thread_id="shadow-t1")
        await bundle.run()
    finally:
        monkeypatch.undo()
    assert calls == [], f"shadow called production graph resolver: {calls}"


@pytest.mark.asyncio
async def test_shadow_never_constructs_production_saver() -> None:
    """R54: shadow must not open the production Postgres saver factory."""
    import ast
    import inspect

    import app.services.agent.shadow_runtime as shadow_runtime
    from app.services.agent.shadow_runtime import build_shadow_bundle

    # Static guarantee over CODE references (docstrings excluded): the
    # shadow path must never reference the production graph resolver,
    # the production saver factory, or the production graph selector.
    tree = ast.parse(inspect.getsource(shadow_runtime))
    referenced: set[str] = set()
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    for forbidden in (
        "create_v2_checkpointer",  # production saver factory
        "get_supervisor_v2_graph",  # production graph singleton
        "resolve_agent_graph",  # production graph selector
        "runtime_selector",
    ):
        assert forbidden not in referenced, (
            f"shadow path references production symbol {forbidden!r}"
        )
    for module in imported_modules:
        assert module != "app.services.agents.v2.persistence.checkpoint", (
            "shadow path imports the production saver factory module"
        )
    try:
        from app.services.agents.v2.persistence import (
            checkpoint as production_checkpoint,
        )
    except ModuleNotFoundError:
        # No Postgres driver in this environment: the static guarantee
        # above (no production reference on the shadow path) plus the
        # isolated-saver branding asserted elsewhere is the proof.
        bundle = build_shadow_bundle(raw_query="Xin chào", thread_id="shadow-t2")
        from app.services.agents.v2.persistence.shadow_checkpoint import (
            is_shadow_saver,
        )

        assert is_shadow_saver(bundle.checkpoint_bundle.saver)
        await bundle.run()
        return
    calls: list[str] = []
    real_factory = production_checkpoint.create_v2_checkpointer

    def _spy(dsn: str) -> Any:
        calls.append(dsn)
        return real_factory(dsn)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        production_checkpoint, "create_v2_checkpointer", _spy
    )
    try:
        bundle = build_shadow_bundle(raw_query="Xin chào", thread_id="shadow-t2")
        await bundle.run()
    finally:
        monkeypatch.undo()
    assert calls == [], f"shadow constructed production saver: {calls}"


@pytest.mark.asyncio
async def test_shadow_leaves_production_rows_unchanged() -> None:
    """R54 isolation proof: production rows are identical after a shadow run."""
    from app.services.agent.shadow_runtime import build_shadow_bundle

    production_rows = make_production_rows()
    before = snapshot(production_rows)
    bundle = build_shadow_bundle(
        raw_query="Xin chào",
        thread_id="shadow-t3",
        production_rows=production_rows,
    )
    metrics = await bundle.run()
    assert production_rows == before, "shadow mutated production rows"
    assert metrics.status in ("success", "error", "clarify", "denied")


@pytest.mark.asyncio
async def test_shadow_emits_no_outbound_events() -> None:
    """R55: no SSE/webhook/Telegram/chat event leaves a shadow run."""
    import app.services.agents.v2.events as v2_events
    import app.services.agent.shadow_runtime as shadow_runtime
    from app.services.agent.shadow_runtime import build_shadow_bundle

    outbound_calls: list[tuple[str, Any]] = []
    real_format = v2_events.format_sse_event
    real_funnel = shadow_runtime.emit_outbound_event

    def _sse_spy(event: str, data: dict) -> str:
        outbound_calls.append(("sse", event))
        return real_format(event, data)

    def _funnel_spy(*args: Any, **kwargs: Any) -> None:
        outbound_calls.append(("funnel", args))
        return real_funnel(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(v2_events, "format_sse_event", _sse_spy)
    monkeypatch.setattr(shadow_runtime, "emit_outbound_event", _funnel_spy)
    try:
        bundle = build_shadow_bundle(raw_query="Xin chào", thread_id="shadow-t4")
        await bundle.run()
    finally:
        monkeypatch.undo()
    assert outbound_calls == [], f"shadow emitted outbound events: {outbound_calls}"


def test_shadow_source_adapters_are_read_only() -> None:
    """R57: no write path is reachable from the shadow bundle."""
    from app.services.agent.shadow_runtime import (
        ShadowIsolationError,
        build_shadow_bundle,
    )

    bundle = build_shadow_bundle(raw_query="Xin chào", thread_id="shadow-t5")
    for adapter in bundle.source_adapters:
        for write_attr in (
            "write",
            "save",
            "commit",
            "persist",
            "delete",
            "update",
            "append",
            "insert",
        ):
            assert not hasattr(adapter, write_attr), (
                f"shadow adapter {type(adapter).__name__!r} exposes "
                f"write path {write_attr!r}"
            )
        with pytest.raises(ShadowIsolationError):
            adapter.write({"anything": "goes"})


def test_shadow_config_defaults_are_safe() -> None:
    """R56: shadow is disabled by default (v2 disabled, percent 0)."""
    from app.core.config import settings

    assert settings.NEXUSRAG_AGENT_V2_SHADOW_PERCENT == 0
    assert settings.NEXUSRAG_AGENT_V2_SHADOW_ENABLED is False


def test_shadow_percent_validation_rejects_out_of_range() -> None:
    """R56: a percent outside 0..100 fails fast at config validation."""
    from app.core.config import Settings

    with pytest.raises(ValueError):
        Settings(NEXUSRAG_AGENT_V2_SHADOW_PERCENT=101)
    with pytest.raises(ValueError):
        Settings(NEXUSRAG_AGENT_V2_SHADOW_PERCENT=-1)


def test_shadow_output_is_redacted_metrics_only() -> None:
    """Shadow output carries redacted metrics — never response content."""
    from app.services.agent.shadow_runtime import ShadowMetrics

    metrics = ShadowMetrics(
        status="success",
        route="direct",
        task_count=0,
        duration_ms=12,
    )
    payload = metrics.redacted()
    assert "content" not in payload
    assert "answer" not in payload
    assert payload["status"] == "success"
    assert payload["route"] == "direct"


@pytest.mark.asyncio
async def test_shadow_cancellation_follows_primary() -> None:
    """Cancellation of the primary run cancels the shadow run promptly."""
    import asyncio as _asyncio

    from app.services.agent.shadow_runtime import build_shadow_bundle

    bundle = build_shadow_bundle(raw_query="Xin chào", thread_id="shadow-t6")
    task = _asyncio.ensure_future(bundle.run_forever())
    await _asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(_asyncio.CancelledError):
        await task
    assert bundle.metrics is None or bundle.metrics.status != "success"


def test_shadow_module_never_redefines_frozen_contracts() -> None:
    """Frozen contracts are imported, never redefined, in shadow modules."""
    import inspect

    import app.services.agent.shadow_runtime as shadow_runtime
    import app.services.agents.v2.persistence.shadow_checkpoint as shadow_checkpoint

    for module in (shadow_runtime, shadow_checkpoint):
        source = inspect.getsource(module)
        assert "class TaskPlan" not in source
        assert "class AgentRequest" not in source
        assert "class SupervisorV2State" not in source

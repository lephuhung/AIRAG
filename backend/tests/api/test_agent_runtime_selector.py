"""Task 7 — lazy v1/v2 runtime selector tests.

Covers: default v1, invalid config fail-fast, lazy factories (no graph
construction at import), all entrypoints through the resolver, v2
schema/checkpoint readiness preceding selection (fail closed, never silent
v1 fallback), scope intersection, ordinary headers ignored, and the
admin-only evaluation override as the sole per-request arm override.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _backend_root() -> Path:
    # In the harness container the worktree is mounted at /app, so the
    # host-side BACKEND_ROOT does not exist; fall back to the live cwd.
    if BACKEND_ROOT.exists():
        return BACKEND_ROOT
    return Path.cwd()


def _read(relative: str) -> str:
    return (_backend_root() / relative).read_text()


# ---------------------------------------------------------------------------
# Config: default v1, invalid fails fast
# ---------------------------------------------------------------------------


def test_config_default_is_v1():
    from app.core.config import Settings

    assert Settings().NEXUSRAG_AGENT_GRAPH_VERSION == "v1"


def test_configured_version_defaults_to_v1(monkeypatch):
    import app.services.agent.runtime_selector as selector
    from app.core import config as config_module

    monkeypatch.setattr(
        config_module.settings, "NEXUSRAG_AGENT_GRAPH_VERSION", "v1"
    )
    assert selector.configured_agent_version() == "v1"


def test_config_invalid_version_fails_fast():
    import pydantic

    from app.core.config import Settings

    with pytest.raises(pydantic.ValidationError):
        Settings(NEXUSRAG_AGENT_GRAPH_VERSION="v3")


def test_configured_version_rejects_invalid_value(monkeypatch):
    import app.services.agent.runtime_selector as selector
    from app.core import config as config_module

    monkeypatch.setattr(
        config_module.settings, "NEXUSRAG_AGENT_GRAPH_VERSION", "v9"
    )
    with pytest.raises(ValueError):
        selector.configured_agent_version()


def test_unknown_version_rejected():
    import asyncio

    import app.services.agent.runtime_selector as selector

    with pytest.raises(ValueError):
        asyncio.run(selector.resolve_agent_graph("v9"))


# ---------------------------------------------------------------------------
# Laziness: no graph construction at import time
# ---------------------------------------------------------------------------


def test_selector_import_builds_no_graph():
    """Importing the selector must construct neither graph.

    Transitive module imports (via the ``app.services.agent`` package init)
    are pre-existing and out of scope; what the lazy-factory contract
    forbids is graph CONSTRUCTION at import time, pinned here through both
    singletons staying empty.
    """
    code = (
        "import sys; "
        "import app.services.agent.runtime_selector; "
        "from app.services.agents import supervisor as _v1; "
        "assert _v1._supervisor_graph is None, 'v1 graph built at import'; "
        "_v2 = sys.modules.get('app.services.agents.supervisor_v2'); "
        "assert _v2 is None or _v2._supervisor_v2_graph is None, "
        "'v2 graph built at import'; "
        "print('lazy-ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_backend_root()),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "lazy-ok" in proc.stdout


def test_selector_graph_imports_are_function_local():
    """Both supervisor imports must live inside functions, never at top level."""
    tree = ast.parse(_read("app/services/agent/runtime_selector.py"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                names = [f"{node.module}.{a.name}" for a in node.names]
            joined = " ".join(names)
            assert "agents.supervisor" not in joined, (
                f"module-level supervisor import forbidden: {joined}"
            )


# ---------------------------------------------------------------------------
# Lazy resolution: v1 direct, v2 gated
# ---------------------------------------------------------------------------


def test_resolve_v1_uses_lazy_factory(monkeypatch):
    import asyncio

    import app.services.agent.runtime_selector as selector

    sentinel = object()
    calls: list[str] = []

    class _FakeV1Module:
        @staticmethod
        def get_supervisor_graph():
            calls.append("v1")
            return sentinel

    monkeypatch.setitem(
        sys.modules, "app.services.agents.supervisor", _FakeV1Module()
    )
    graph = asyncio.run(selector.resolve_agent_graph("v1"))
    assert graph is sentinel
    assert calls == ["v1"]


def test_resolve_v2_runs_readiness_gate_first(monkeypatch):
    import asyncio

    import app.services.agent.runtime_selector as selector

    order: list[str] = []

    async def _fake_gate(**kwargs):
        order.append("gate")

    class _FakeV2Module:
        @staticmethod
        def get_supervisor_v2_graph():
            order.append("graph")
            return object()

    monkeypatch.setattr(selector, "require_v2_schema_ready", _fake_gate)
    monkeypatch.setitem(
        sys.modules, "app.services.agents.supervisor_v2", _FakeV2Module()
    )
    asyncio.run(selector.resolve_agent_graph("v2"))
    assert order == ["gate", "graph"], order


def test_resolve_v2_fails_closed_when_not_ready(monkeypatch):
    import asyncio

    import app.services.agent.runtime_selector as selector

    async def _refuse(**kwargs):
        raise selector.V2NotReadyError("schema not ready")

    monkeypatch.setattr(selector, "require_v2_schema_ready", _refuse)
    with pytest.raises(selector.V2NotReadyError):
        asyncio.run(selector.resolve_agent_graph("v2"))


def test_v2_unselectable_by_default(monkeypatch):
    """Negative readiness must fail closed, never silently fall back to v1."""
    import asyncio

    import app.services.agent.runtime_selector as selector
    from app.services.agents.v2.persistence.checkpoint import CheckpointCheck
    from app.services.agents.v2.persistence.migrate import SchemaCheck

    schema = SchemaCheck(
        applied=False, version=None, missing_tables=frozenset({"x"}),
        extra_tables=frozenset(),
    )
    checkpoint = CheckpointCheck(
        present=frozenset(), missing=frozenset({"checkpoints"})
    )
    with pytest.raises(selector.V2NotReadyError):
        asyncio.run(
            selector.require_v2_schema_ready(
                schema_check=schema, checkpoint_check=checkpoint
            )
        )


def test_require_v2_schema_ready_passes_when_clean():
    import asyncio

    import app.services.agent.runtime_selector as selector
    from app.services.agents.v2.persistence.checkpoint import CheckpointCheck
    from app.services.agents.v2.persistence.migrate import SchemaCheck

    schema = SchemaCheck(
        applied=True, version=2, missing_tables=frozenset(),
        extra_tables=frozenset(), shape_errors=frozenset(),
    )
    checkpoint = CheckpointCheck(
        present=frozenset(
            {"checkpoints", "checkpoint_blobs", "checkpoint_writes",
             "checkpoint_migrations"}
        ),
        missing=frozenset(),
    )
    asyncio.run(
        selector.require_v2_schema_ready(
            schema_check=schema, checkpoint_check=checkpoint
        )
    )


# ---------------------------------------------------------------------------
# Per-request version: configured default wins; admin override is the only arm
# ---------------------------------------------------------------------------


class _User:
    def __init__(self, *, is_superadmin: bool):
        self.is_superadmin = is_superadmin


def test_request_version_defaults_to_configured(monkeypatch):
    import app.services.agent.runtime_selector as selector
    from app.core import config as config_module

    monkeypatch.setattr(
        config_module.settings, "NEXUSRAG_AGENT_GRAPH_VERSION", "v1"
    )
    assert (
        selector.resolve_request_version(user=_User(is_superadmin=False)) == "v1"
    )


def test_request_version_accepts_no_headers():
    """Ordinary graph-version headers are ignored: no header param exists."""
    import inspect

    import app.services.agent.runtime_selector as selector

    params = set(inspect.signature(selector.resolve_request_version).parameters)
    assert params == {"user", "admin_override"}, params
    for name in params:
        assert "header" not in name.lower()


def test_admin_override_selects_other_arm_for_admin(monkeypatch):
    import app.services.agent.runtime_selector as selector
    from app.core import config as config_module

    monkeypatch.setattr(
        config_module.settings, "NEXUSRAG_AGENT_GRAPH_VERSION", "v1"
    )
    assert (
        selector.resolve_request_version(
            user=_User(is_superadmin=True), admin_override="v2"
        )
        == "v2"
    )


def test_admin_override_rejected_for_non_admin(monkeypatch):
    from fastapi import HTTPException

    import app.services.agent.runtime_selector as selector
    from app.core import config as config_module

    monkeypatch.setattr(
        config_module.settings, "NEXUSRAG_AGENT_GRAPH_VERSION", "v1"
    )
    with pytest.raises(HTTPException) as exc:
        selector.resolve_request_version(
            user=_User(is_superadmin=False), admin_override="v2"
        )
    assert exc.value.status_code == 403


def test_admin_override_rejected_when_anonymous(monkeypatch):
    from fastapi import HTTPException

    import app.services.agent.runtime_selector as selector
    from app.core import config as config_module

    monkeypatch.setattr(
        config_module.settings, "NEXUSRAG_AGENT_GRAPH_VERSION", "v1"
    )
    with pytest.raises(HTTPException) as exc:
        selector.resolve_request_version(user=None, admin_override="v2")
    assert exc.value.status_code == 401


def test_admin_override_invalid_value_fails_fast():
    from fastapi import HTTPException

    import app.services.agent.runtime_selector as selector

    with pytest.raises(HTTPException) as exc:
        selector.resolve_request_version(
            user=_User(is_superadmin=True), admin_override="v9"
        )
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Scope: authenticated ∩ requested, never widened
# ---------------------------------------------------------------------------


def test_runtime_scope_is_intersection():
    import app.services.agent.runtime_selector as selector

    a, b, c = uuid4(), uuid4(), uuid4()
    assert selector.resolve_runtime_scope(
        authenticated_ids=[a, b], requested_ids=[b, c]
    ) == (b,)
    # Order follows the authenticated scope.
    assert selector.resolve_runtime_scope(
        authenticated_ids=[b, a], requested_ids=[a, b]
    ) == (b, a)


def test_runtime_scope_defaults_to_authenticated():
    import app.services.agent.runtime_selector as selector

    a, b = uuid4(), uuid4()
    assert selector.resolve_runtime_scope(
        authenticated_ids=[a, b], requested_ids=None
    ) == (a, b)


def test_runtime_scope_never_widens():
    import app.services.agent.runtime_selector as selector

    a, b = uuid4(), uuid4()
    scoped = selector.resolve_runtime_scope(
        authenticated_ids=[a], requested_ids=[a, b]
    )
    assert scoped == (a,)
    assert set(scoped) <= {a}


def test_runtime_scope_disjoint_is_empty():
    import app.services.agent.runtime_selector as selector

    assert selector.resolve_runtime_scope(
        authenticated_ids=[uuid4()], requested_ids=[uuid4()]
    ) == ()


# ---------------------------------------------------------------------------
# Entrypoints: standalone, session, Telegram all go through the resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "app/api/chat_agent_lg.py",
        "app/api/chat_session.py",
        "app/services/integrations/telegram_service.py",
    ],
)
def test_entrypoint_uses_resolver_not_direct_construction(relative):
    source = _read(relative)
    assert "resolve_agent_graph" in source, f"{relative} bypasses the resolver"
    assert "get_supervisor_graph(" not in source, (
        f"{relative} still constructs the graph directly"
    )
    assert "get_supervisor_v2_graph(" not in source, (
        f"{relative} still constructs the v2 graph directly"
    )

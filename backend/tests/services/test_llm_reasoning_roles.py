"""Task 0 — semantic_router/planner LLM roles inherit effective thinking config.

Red-phase tests for the Phase-4 prerequisite:
- new roles default to the EFFECTIVE thinking connection/model (incl. DB overrides);
- explicit assignment wins over inheritance;
- reasoning-role provider factories are isolated per role/config version;
- admin API accepts the new roles, rejects unknown roles, reports truthful
  inherited source/connection, and never leaks plaintext API keys;
- frontend LlmRole typing contains both new roles.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services import runtime_config
from app.services.runtime_config import EffectiveLLMConfig


@pytest.fixture
def _clean_runtime():
    """Isolate the in-process snapshot + provider caches per test."""
    import app.services.llm as llm_module

    saved_snap = dict(runtime_config._snapshot)
    saved_ver = runtime_config._snapshot_version
    saved_providers = dict(llm_module._PROVIDERS)
    runtime_config._snapshot = {}
    runtime_config._snapshot_version = -1
    llm_module._PROVIDERS.clear()
    try:
        yield
    finally:
        runtime_config._snapshot = saved_snap
        runtime_config._snapshot_version = saved_ver
        llm_module._PROVIDERS.clear()
        llm_module._PROVIDERS.update(saved_providers)


def _thinking_db_cfg() -> EffectiveLLMConfig:
    return EffectiveLLMConfig(
        provider="openai_compatible",
        base_url="http://thinking-conn:8000/v1",
        model="thinking-model-v2",
        api_key="secret-thinking-key",
        extra={},
        source="db",
    )


# ---------------------------------------------------------------------------
# Defaults: inherit EFFECTIVE thinking (env-level, no snapshot yet)
# ---------------------------------------------------------------------------

def test_default_semantic_router_equals_effective_thinking(_clean_runtime) -> None:
    assert runtime_config.get_effective_sync("semantic_router") == runtime_config.get_effective_sync("thinking")


def test_default_planner_equals_effective_thinking(_clean_runtime) -> None:
    assert runtime_config.get_effective_sync("planner") == runtime_config.get_effective_sync("thinking")


def test_new_roles_registered(_clean_runtime) -> None:
    assert "semantic_router" in runtime_config.ROLES
    assert "planner" in runtime_config.ROLES


# ---------------------------------------------------------------------------
# DB-backed resolution: _load_effective with a faked system_settings store
# ---------------------------------------------------------------------------

class _FakeDB:
    """Minimal async-session stand-in for runtime_config._load_effective."""

    def __init__(self, rows: dict[str, dict | None]) -> None:
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def scalar(self, *args, **kwargs):
        raise AssertionError("_FakeDB.scalar must be bypassed via _fetch_row patch")


def _patch_settings_store(monkeypatch: pytest.MonkeyPatch, role_rows: dict, conn_rows: dict):
    """Patch _fetch_row so _load_effective resolves without a real database."""
    async def _fake_fetch_row(db, key: str):
        if key.startswith("llm_role."):
            doc = role_rows.get(key.removeprefix("llm_role."))
        elif key.startswith("llm_conn."):
            doc = conn_rows.get(key.removeprefix("llm_conn."))
        else:
            doc = None
        if doc is None:
            return None
        return SimpleNamespace(value_enc=json.dumps(doc))

    class _FakeSessionMaker:
        def __call__(self):
            return _FakeDB({})

    monkeypatch.setattr(runtime_config, "_fetch_row", _fake_fetch_row)
    monkeypatch.setattr("app.core.database.async_session_maker", _FakeSessionMaker())


def _thinking_conn_rows(base_url="http://thinking-conn:8000/v1"):
    return {
        "think-conn": {
            "name": "Thinking",
            "provider": "openai_compatible",
            "base_url": base_url,
            "api_key_enc": runtime_config._encrypt("secret-thinking-key"),
            "extra": {},
        }
    }


@pytest.mark.asyncio
async def test_changing_thinking_db_assignment_changes_inherited_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inherited roles track the EFFECTIVE thinking assignment (not just .env)."""
    role_rows: dict = {
        "thinking": {"conn_id": "think-conn", "model": "thinking-model-v2"},
    }
    _patch_settings_store(monkeypatch, role_rows, _thinking_conn_rows())

    thinking = await runtime_config._load_effective("thinking")
    router = await runtime_config._load_effective("semantic_router")
    planner = await runtime_config._load_effective("planner")
    assert (router.base_url, router.model) == (thinking.base_url, thinking.model)
    assert (planner.base_url, planner.model) == (thinking.base_url, thinking.model)
    assert thinking.source == "db"

    # Admin re-points thinking at a new connection/model → inherited follows.
    _patch_settings_store(
        monkeypatch, role_rows,
        _thinking_conn_rows(base_url="http://thinking-conn-v2:8000/v1"),
    )
    role_rows["thinking"] = {"conn_id": "think-conn", "model": "thinking-model-v3"}
    thinking2 = await runtime_config._load_effective("thinking")
    assert (await runtime_config._load_effective("semantic_router")).model == thinking2.model
    assert (await runtime_config._load_effective("planner")).base_url == thinking2.base_url


@pytest.mark.asyncio
async def test_explicit_assignment_wins_over_thinking_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    role_rows: dict = {
        "thinking": {"conn_id": "think-conn", "model": "thinking-model-v2"},
        "semantic_router": {"conn_id": "@env", "model": "custom-router-model"},
        "planner": {"conn_id": "think-conn", "model": "custom-planner-model"},
    }
    _patch_settings_store(monkeypatch, role_rows, _thinking_conn_rows())

    router = await runtime_config._load_effective("semantic_router")
    planner = await runtime_config._load_effective("planner")
    assert router.model == "custom-router-model"
    assert planner.model == "custom-planner-model"

    # Thinking moves on; explicit assignments must not follow.
    role_rows["thinking"] = {"conn_id": "think-conn", "model": "thinking-model-v9"}
    assert (await runtime_config._load_effective("semantic_router")).model == "custom-router-model"
    assert (await runtime_config._load_effective("planner")).model == "custom-planner-model"


# ---------------------------------------------------------------------------
# Provider factories: shared reasoning-role factory, distinct labels, isolation
# ---------------------------------------------------------------------------

def test_reasoning_provider_factories_exist_with_distinct_labels(_clean_runtime) -> None:
    from app.services.llm import get_planner_provider, get_semantic_router_provider

    router = get_semantic_router_provider()
    planner = get_planner_provider()
    assert getattr(router, "_label", None) == "semantic_router_llm"
    assert getattr(planner, "_label", None) == "planner_llm"


def test_thinking_provider_backward_compatible(_clean_runtime) -> None:
    from app.services.llm import get_thinking_provider

    thinking = get_thinking_provider()
    assert getattr(thinking, "_label", None) == "thinking_llm"


def test_provider_caches_isolated_by_role_and_config_version(_clean_runtime) -> None:
    import app.services.llm as llm_module
    from app.services.llm import get_planner_provider, get_semantic_router_provider

    runtime_config._snapshot_version = 7
    r1 = get_semantic_router_provider()
    p1 = get_planner_provider()
    assert r1 is get_semantic_router_provider()
    assert p1 is get_planner_provider()
    assert r1 is not p1
    assert set(llm_module._PROVIDERS) >= {"role:semantic_router", "role:planner"}

    # Config version bump rebuilds both role providers independently.
    runtime_config._snapshot_version = 8
    r2 = get_semantic_router_provider()
    p2 = get_planner_provider()
    assert r2 is not r1
    assert p2 is not p1
    assert r2 is not p2


# ---------------------------------------------------------------------------
# Admin API: accept new roles, reject unknown, truthful inheritance, no key leak
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_accepts_new_roles_and_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from app.api import llm_config as api_module

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(runtime_config, "set_override", AsyncMock(return_value=None))
    monkeypatch.setattr(
        runtime_config, "get_effective_sync",
        lambda role: EffectiveLLMConfig(
            provider="openai_compatible", base_url="http://x/v1",
            model="m", api_key="sekret", source="db",
        ),
    )
    monkeypatch.setattr(runtime_config, "snapshot_version", lambda: 3)
    monkeypatch.setattr(
        "app.services.audit_service.AuditService.record_for_actor",
        AsyncMock(return_value=None),
    )
    body = SimpleNamespace(conn_id="think-conn", model="m", extra=None)
    user = SimpleNamespace(id="u1")

    for role in ("semantic_router", "planner"):
        res = await api_module.assign_role(role, body, user)
        assert res["ok"] is True and res["role"] == role

    with pytest.raises(HTTPException) as exc:
        await api_module.assign_role("definitely_not_a_role", body, user)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_inherited_role_response_reports_truthful_source_and_hides_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import llm_config as api_module

    thinking = _thinking_db_cfg()

    def _fake_effective(role: str):
        if role == "thinking":
            return thinking
        return thinking  # inherited roles resolve to effective thinking

    monkeypatch.setattr(runtime_config, "get_effective_sync", _fake_effective)
    monkeypatch.setattr(runtime_config, "list_overrides", AsyncMock(return_value={
        "thinking": {"override": True, "conn_id": "think-conn", "model": thinking.model},
    }))
    monkeypatch.setattr(runtime_config, "list_connections", AsyncMock(return_value={}))
    monkeypatch.setattr(runtime_config, "snapshot_version", lambda: 5)
    user = SimpleNamespace(id="u1")

    payload = await api_module.get_llm_config(user)
    for role in ("semantic_router", "planner"):
        status = payload["roles"][role]
        # Truthful: effective DB-backed thinking connection/model, NOT "@env".
        assert status["conn_id"] == "think-conn", status
        assert status["model"] == thinking.model, status
        assert status["source"] == "db", status
        assert status["resolved"]["base_url"] == thinking.base_url, status

    dumped = json.dumps(payload)
    assert "secret-thinking-key" not in dumped
    assert "api_key" not in dumped.replace("masked_api_key", "")


# ---------------------------------------------------------------------------
# Frontend typing
# ---------------------------------------------------------------------------

def test_frontend_llm_role_contains_new_roles() -> None:
    here = Path(__file__).resolve()
    repo = next(
        (p for p in [here.parent, *here.parents]
         if (p / "frontend" / "src" / "types" / "llmConfig.ts").exists()),
        here.parents[4],
    )
    ts = (repo / "frontend" / "src" / "types" / "llmConfig.ts").read_text()
    for role in ('"semantic_router"', '"planner"'):
        assert role in ts, f"frontend LlmRole missing {role}"

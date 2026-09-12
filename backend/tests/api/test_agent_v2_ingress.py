"""Task 7 — v2 ingress tests.

Covers: raw user text persisted before normalization, scope intersection
at ingress, the per-request RuntimeServices/registry construction (real
v1-backed adapters, D5 binding resolver, D3 determinism, M4a dedicated
lease session), the runtime-only truncation channel, the
ClarificationUnsatisfiable runner contract, the placeholder-terminal
validation rule, and the admin evaluation surface guards.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
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
# Raw text persists before normalization/semantic work
# ---------------------------------------------------------------------------


def test_raw_message_helper_persists_verbatim():
    import asyncio

    import app.services.agent.runtime_selector as selector

    added: list = []

    class _FakeDB:
        def add(self, row):
            added.append(row)

        async def commit(self):
            return None

    db = _FakeDB()
    row = asyncio.run(
        selector.persist_raw_user_message(
            db,
            session_id=str(uuid4()),
            user_id=uuid4(),
            raw_text=" decide 13/2023/NĐ-CP ",
        )
    )
    assert row.content == " decide 13/2023/NĐ-CP "
    assert row.role == "user"
    assert added == [row]


def test_standalone_entrypoint_persists_before_expansion():
    """chat_agent_lg must persist the raw text before abbreviation expansion."""
    source = _read("app/api/chat_agent_lg.py")
    persist_at = source.find("persist_raw_user_message")
    expand_at = source.find("expand_ab_in_text")
    assert persist_at != -1, "standalone path must persist the raw user text"
    assert expand_at != -1
    assert persist_at < expand_at, (
        "raw user text must persist BEFORE abbreviation expansion"
    )


def test_session_entrypoint_persists_raw_text():
    source = _read("app/api/chat_session.py")
    # The persisted user content must be the raw request text, never a
    # normalized/expanded form.
    assert "persist_raw_user_message" in source
    assert "raw_text=request.message" in source


def test_telegram_entrypoint_persists_raw_text():
    source = _read("app/services/integrations/telegram_service.py")
    assert "persist_raw_user_message" in source or "content=question" in source


# ---------------------------------------------------------------------------
# Ingress construction: scope, services, sessions
# ---------------------------------------------------------------------------


def _ingress_kwargs(**overrides):
    from uuid import UUID

    base = {
        "user_id": uuid4(),
        "authenticated_workspace_ids": [uuid4(), uuid4()],
        "requested_workspace_ids": None,
        "raw_query": "quyết định 13 là gì?",
        "thread_id": f"thread-{uuid4().hex[:8]}",
        "can_read_people": False,
    }
    base.update(overrides)
    return base


def test_ingress_scope_is_authenticated_intersect_requested():
    import asyncio

    import app.services.agent.runtime_selector as selector

    a, b, c = uuid4(), uuid4(), uuid4()

    async def _run():
        async with selector.build_v2_ingress(
            **_ingress_kwargs(
                authenticated_workspace_ids=[a, b],
                requested_workspace_ids=[b, c],
            )
        ) as ingress:
            return ingress

    ingress = asyncio.run(_run())
    assert ingress.runtime_context.capability_runtime.workspace_ids == (b,)


def test_ingress_services_are_wired():
    import asyncio

    import app.services.agent.runtime_selector as selector
    from app.services.agents.supervisor_v2 import (
        DeterministicSemanticAdapter,
        V1BindingResolver,
    )
    from app.services.agents.v2.nodes.evaluate import AnswerDraftChannel

    async def _run():
        async with selector.build_v2_ingress(**_ingress_kwargs()) as ingress:
            return ingress

    ingress = asyncio.run(_run())
    services = ingress.runtime_context.services
    assert isinstance(services.semantic_adapter, DeterministicSemanticAdapter)
    assert isinstance(services.binding_resolver, V1BindingResolver)
    assert services.capability_registry is not None
    assert isinstance(services.answer_draft_channel, AnswerDraftChannel)
    assert services.evidence_hydrator is not None
    assert services.chat_messages is not None
    assert services.authorization is not None
    assert services.retention_leases is not None
    # The initial envelope carries the raw query; nothing is normalized yet.
    assert ingress.initial_state["request"].original_query.startswith("quyết định")


def test_ingress_lease_session_is_dedicated():
    """M4a: the lease repo owns a session used for nothing else."""
    import asyncio

    import app.services.agent.runtime_selector as selector

    seen: list[str] = []

    class _FakeSession:
        def __init__(self, name: str):
            self._name = name

        async def close(self):
            seen.append(self._name)

    def _factory(name: str):
        return _FakeSession(name)

    async def _run():
        async with selector.build_v2_ingress(
            **_ingress_kwargs(),
            session_factory=lambda: _factory("evidence"),
            lease_session_factory=lambda: _factory("lease"),
        ) as ingress:
            lease_session = ingress.lease_session
            evidence_session = ingress.evidence_session
            assert lease_session is not evidence_session
            return ingress

    asyncio.run(_run())
    assert seen == ["lease", "evidence"] or set(seen) == {"lease", "evidence"}


def test_ingress_truncation_channel_is_fresh_and_runtime_only():
    import asyncio

    import app.services.agent.runtime_selector as selector

    async def _run():
        async with selector.build_v2_ingress(**_ingress_kwargs()) as first:
            channel_one = first.runtime_context.services.answer_draft_channel
        async with selector.build_v2_ingress(**_ingress_kwargs()) as second:
            channel_two = second.runtime_context.services.answer_draft_channel
        return channel_one, channel_two

    one, two = asyncio.run(_run())
    assert one is not two


def test_semantic_adapter_is_deterministic():
    import asyncio

    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter
    from app.services.agents.v2.contracts.conversation import ConversationContext
    from app.services.agents.v2.contracts.request import RequestContext

    async def _fake_preprocess(raw_query: str):
        from app.services.agents.semantic_preprocessor import PreprocessingResult

        return PreprocessingResult(
            original_query=raw_query,
            normalized_query=raw_query.strip().lower(),
            preprocessing_status="ok",
            preprocessor_trace=[],
        )

    adapter = DeterministicSemanticAdapter(preprocess=_fake_preprocess)
    request = RequestContext(
        contract_version="2.0",
        request_id="r1",
        thread_id="t1",
        original_query="  Tra cứu NĐ 13?  ",
        known_documents=(),
    )
    conversation = ConversationContext(
        summary="", active_entities=(), last_focus=None, recent_turns=()
    )
    first = asyncio.run(adapter.build_draft(request, conversation))
    second = asyncio.run(adapter.build_draft(request, conversation))
    assert first == second


def test_plan_binding_resolver_fails_closed_until_fed():
    import app.services.agent.runtime_selector as selector

    resolver = selector.PlanBindingResolver()
    assert resolver.resolve("missing-target") is None


def test_undispatched_tasks_recipe():
    """T3-N2: truncation stays runtime-only; the runner diffs plan vs results."""
    import app.services.agent.runtime_selector as selector

    plan = SimpleNamespace(tasks=(SimpleNamespace(task_id="a"), SimpleNamespace(task_id="b")))
    results = (SimpleNamespace(task_id="a"),)
    assert selector.undispatched_tasks(plan, results) == ("b",)
    assert selector.undispatched_tasks(plan, results + (SimpleNamespace(task_id="b"),)) == ()


# ---------------------------------------------------------------------------
# ClarificationUnsatisfiable runner contract + placeholder-terminal rule
# ---------------------------------------------------------------------------


def test_candidate_free_reply_is_fresh_turn_contract():
    import app.services.agent.runtime_selector as selector
    from app.services.agents.v2.nodes.clarification import (
        ClarificationInvalidSelection,
        ClarificationUnsatisfiable,
        parse_clarification_resolution,
    )

    request = SimpleNamespace(
        candidates=(), clarification_id="c1", reason="required_document_not_found"
    )
    with pytest.raises(ClarificationUnsatisfiable) as exc:
        parse_clarification_resolution("1", request)
    assert selector.clarification_reply_is_fresh_turn(exc.value) is True
    assert selector.clarification_reply_is_fresh_turn(
        ClarificationInvalidSelection("bad")
    ) is False


def test_placeholder_terminal_is_never_validated():
    """Non-success terminals over the ingress placeholder semantic are
    terminal-errored, not corrupt: the runner must not validate them."""
    import app.services.agent.runtime_selector as selector
    from app.services.agents.supervisor_v2 import build_initial_v2_state
    from app.services.agents.v2.contracts.base import CONTRACT_VERSION
    from app.services.agents.v2.contracts.request import RequestContext
    from app.services.agents.v2.contracts.response import FinalResponse

    request = RequestContext(
        contract_version=CONTRACT_VERSION,
        request_id="r1",
        thread_id="t1",
        original_query="raw query",
        known_documents=(),
    )
    state = build_initial_v2_state(request=request)
    state["final_response"] = FinalResponse(
        contract_version=CONTRACT_VERSION,
        status="error",
        content="không đủ căn cứ",
        citations=(),
    )
    assert selector.terminal_state_is_error(state) is True
    assert selector.terminal_state_is_error(build_initial_v2_state(request=request)) is False


# ---------------------------------------------------------------------------
# Admin evaluation surface
# ---------------------------------------------------------------------------


def test_admin_router_is_registered():
    source = _read("app/api/router.py")
    assert "agent_admin" in source


def test_admin_evaluate_requires_superadmin():
    """The evaluation endpoint is authenticated-admin-only."""
    from fastapi.routing import APIRoute

    from app.api.agent_admin import router
    from app.core.deps import require_superadmin

    routes = [r for r in router.routes if isinstance(r, APIRoute)]
    assert routes, "agent_admin must expose at least one route"
    evaluate = next(r for r in routes if "evaluat" in r.path)
    deps = list(evaluate.dependencies)
    assert any(getattr(d, "dependency", None) is require_superadmin for d in deps), (
        "evaluation route must depend on require_superadmin"
    )


def test_admin_evaluate_rejects_invalid_version():
    import asyncio

    from fastapi import HTTPException

    from app.api.agent_admin import run_admin_evaluation

    admin = SimpleNamespace(is_superadmin=True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            run_admin_evaluation(
                message="hello",
                version="v9",
                user=admin,
                workspace_ids=None,
            )
        )
    assert exc.value.status_code == 400


def test_admin_evaluate_rejects_non_admin():
    import asyncio

    from fastapi import HTTPException

    from app.api.agent_admin import run_admin_evaluation

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            run_admin_evaluation(
                message="hello",
                version="v1",
                user=SimpleNamespace(is_superadmin=False),
                workspace_ids=None,
            )
        )
    assert exc.value.status_code == 403


def test_single_runtime_services_definition():
    """RuntimeServices keeps exactly ONE definition (contracts/state.py)."""
    import subprocess

    proc = subprocess.run(
        ["grep", "-rn", "class RuntimeServices", "app", "--include=*.py"],
        cwd=str(_backend_root()),
        capture_output=True,
        text=True,
        timeout=60,
    )
    matches = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(matches) == 1, matches
    assert "v2/contracts/state.py" in matches[0]

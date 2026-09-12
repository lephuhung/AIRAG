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
    # Pinned to the helper call with the verbatim question (a bare
    # `"content=question" in source` check would also pass on pre-change code).
    assert "persist_raw_user_message" in source
    assert "raw_text=question" in source


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


# ---------------------------------------------------------------------------
# I1 — session v2 cleanup: raising/cancelled streams still close sessions
# and roll back the partial evidence transaction (v1 unaffected)
# ---------------------------------------------------------------------------


def _fake_session_factory(recorder, name):
    class _FakeSession:
        async def commit(self):
            recorder.append((name, "commit"))

        async def rollback(self):
            recorder.append((name, "rollback"))

        async def close(self):
            recorder.append((name, "close"))

    def _factory():
        return _FakeSession()

    return _factory


class _BoomGraph:
    """Fake v2 graph whose turn raises mid-stream (I1 error path)."""

    async def ainvoke(self, state, config, context=None):
        raise RuntimeError("stream blew up")


def _session_v2_kwargs(recorder, **overrides):
    from uuid import uuid4

    async def _fake_preprocess(raw_query: str):
        return SimpleNamespace(normalized_query=raw_query)

    kwargs = {
        "graph": _BoomGraph(),
        "raw_message": "halo?",
        "workspace_ids": [uuid4()],
        "document_ids": (),
        "user_id": uuid4(),
        "can_read_people": False,
        "thread_id": "thread-i1",
        "session_factory": _fake_session_factory(recorder, "evidence"),
        "lease_session_factory": _fake_session_factory(recorder, "lease"),
        "preprocess": _fake_preprocess,
        "available_services": frozenset(),
    }
    kwargs.update(overrides)
    return kwargs


def test_session_v2_run_rolls_back_and_closes_on_error():
    import asyncio

    from app.api.chat_session import _session_v2_run

    recorder: list = []

    async def _boom(agen):
        async for _ in agen:
            pass
        raise RuntimeError("stream blew up")

    async def _main():
        with pytest.raises(RuntimeError, match="stream blew up"):
            async with _session_v2_run(**_session_v2_kwargs(recorder)) as (
                _ingress,
                agen,
            ):
                await _boom(agen)

    asyncio.run(_main())
    # Partial evidence work is rolled back, and BOTH sessions close.
    assert ("evidence", "rollback") in recorder
    assert ("evidence", "commit") not in recorder
    assert ("evidence", "close") in recorder
    assert ("lease", "close") in recorder


def test_session_v2_run_closes_without_commit_on_cancel():
    import asyncio

    from app.api.chat_session import _session_v2_run

    recorder: list = []
    entered = asyncio.Event()

    async def _hang(agen):
        entered.set()
        await asyncio.sleep(3600)

    async def _victim():
        async with _session_v2_run(**_session_v2_kwargs(recorder)) as (
            _ingress,
            agen,
        ):
            await _hang(agen)

    async def _main():
        task = asyncio.create_task(_victim())
        await asyncio.wait_for(entered.wait(), timeout=10)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled() or task.done()

    asyncio.run(_main())
    # Cancellation skips the commit (session close rolls the partial turn
    # back), but both sessions still close deterministically — no GC wait.
    assert ("evidence", "commit") not in recorder
    assert ("evidence", "close") in recorder
    assert ("lease", "close") in recorder


def test_session_v2_run_commits_on_success():
    import asyncio

    from app.api.chat_session import _session_v2_run
    from app.services.agents.v2.contracts.base import CONTRACT_VERSION
    from app.services.agents.v2.contracts.response import FinalResponse

    recorder: list = []
    seen: list = []

    class _Graph:
        async def ainvoke(self, state, config, context=None):
            return {
                "final_response": FinalResponse(
                    contract_version=CONTRACT_VERSION,
                    status="success",
                    content="done",
                    citations=(),
                )
            }

    async def _drain(agen):
        async for sse in agen:
            seen.append(sse)

    async def _main():
        async with _session_v2_run(
            **_session_v2_kwargs(recorder, graph=_Graph())
        ) as (_ingress, agen):
            await _drain(agen)

    asyncio.run(_main())
    assert ("evidence", "commit") in recorder
    assert ("evidence", "close") in recorder
    assert ("lease", "close") in recorder
    assert any(s.startswith("event: complete") for s in seen)


# ---------------------------------------------------------------------------
# Final-review fix wave (F1/F2/F4/F8)
# ---------------------------------------------------------------------------


def test_production_binding_resolver_pins_role_less_reference(monkeypatch):
    """F1: the PRODUCTION resolver wiring pins a role-less resolved ref.

    Goes through ``build_v2_ingress`` services (not a hand-built resolver):
    the production semantic adapter emits ``requested_role=None`` for every
    reference, so the wired ``V1BindingResolver`` must supply the role
    policy itself (``default_role="target"``).
    """
    import asyncio
    from types import SimpleNamespace as _NS
    from uuid import uuid4

    import app.services.agent.runtime_selector as selector
    from app.services.agents.v2.contracts.semantic import DocumentReference
    from app.services.agents.v2.persistence import document_views

    workspace_id = uuid4()
    document_id = uuid4()
    revision_id = uuid4()

    async def _fake_current(db, document_id_arg, *, require_vectors=False):
        assert document_id_arg == document_id
        return document_views.RevisionArtifactIdentity(
            revision_id=revision_id,
            document_id=document_id,
            generation=1,
            build_profile="FULL",
            markdown_artifact_key="markdown.md",
            structure_artifact_key="structure.json",
            embedding_namespace=None,
            embedding_model_hash=None,
            embedding_dimension=None,
            vector_artifact_version=None,
        )

    monkeypatch.setattr(
        document_views, "load_current_revision_identity", _fake_current
    )

    class _FakeSession:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *exc):
            return False

        async def commit(self):
            return None

        async def rollback(self):
            return None

        async def close(self):
            return None

    async def _fake_preprocess(raw_query: str):
        return _NS(normalized_query=raw_query)

    async def _main():
        async with selector.build_v2_ingress(
            user_id=uuid4(),
            authenticated_workspace_ids=[workspace_id],
            requested_workspace_ids=None,
            raw_query="quyết định 13 là gì?",
            thread_id=f"thread-{uuid4().hex[:8]}",
            session_factory=lambda: _FakeSession(),
            lease_session_factory=lambda: _FakeSession(),
            preprocess=_fake_preprocess,
        ) as ingress:
            resolver = ingress.runtime_context.services.binding_resolver
            ref = DocumentReference(
                ref_id="r1",
                original_span="quyết định 13",
                normalized_reference="quyết định 13",
                requested_role=None,
                resolution_status="resolved",
                resolved_document_id=document_id,
            )
            return await resolver.resolve(
                (ref,), ingress.runtime_context.capability_runtime
            )

    binding_set = asyncio.run(_main())
    assert len(binding_set.bindings) == 1
    assert binding_set.bindings[0].binding_id == "b_r1"
    assert binding_set.bindings[0].document_revision == str(revision_id)
    assert binding_set.bindings[0].role == "target"


def _final_fix_clarification_request():
    from datetime import datetime, timedelta, timezone

    from app.services.agents.v2.contracts.base import CONTRACT_VERSION
    from app.services.agents.v2.contracts.clarification import ClarificationRequest

    return ClarificationRequest(
        contract_version=CONTRACT_VERSION,
        clarification_id="clr-admin-1",
        reason="required_document_ambiguous",
        question="Which document?",
        unresolved_ref_ids=("r1",),
        candidates=(),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


class _FinalFixLeaseRepo:
    """Stand-in for the retention-lease repository (records releases)."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self.released: list[tuple[str, str]] = []
        self.commits = 0
        self.session = self

    async def release_run(self, run_id: str, reason: str = "terminal") -> int:
        self.released.append((run_id, reason))
        return 1

    async def commit(self) -> None:
        self.commits += 1


class _FinalFixGraph:
    """Scripted v2 graph with a checkpoint snapshot (suspend-aware)."""

    def __init__(self, result: dict, *, checkpoint: dict, next_nodes: tuple = ()) -> None:
        self._result = result
        self._checkpoint = dict(checkpoint)
        self._next_nodes = tuple(next_nodes)

    async def aget_state(self, config: Any = None) -> Any:
        from types import SimpleNamespace as _NS

        return _NS(values=dict(self._checkpoint), next=tuple(self._next_nodes))

    async def ainvoke(self, payload: Any, config: Any, context: Any = None) -> Any:
        return self._result


class _FinalFixIngress:
    """Minimal ``build_v2_ingress`` stand-in for the admin eval path."""

    def __init__(self, runtime_context: Any, plan_resolver: Any) -> None:
        self.runtime_context = runtime_context
        self.initial_state = {"request": "admin eval"}
        self.plan_resolver = plan_resolver
        self.committed = 0
        self.rolled_back = 0

    async def __aenter__(self) -> "_FinalFixIngress":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def commit_evidence(self) -> None:
        self.committed += 1

    async def rollback_evidence(self) -> None:
        self.rolled_back += 1


def _run_admin_v2_eval_with_graph(monkeypatch, graph):
    import asyncio
    from types import SimpleNamespace as _NS
    from uuid import uuid4

    import app.api.agent_admin as agent_admin
    import app.services.agent.runtime_selector as selector

    workspace_id = uuid4()
    leases = _FinalFixLeaseRepo(run_id="run-admin-eval")
    runtime_context = _NS(
        capability_runtime=_NS(
            run_id="run-admin-eval",
            workspace_ids=(workspace_id,),
            deadline_at=None,
        ),
        services=_NS(retention_leases=leases),
    )

    class _Resolver:
        def feed(self, *args: Any) -> None:
            return None

    ingress = _FinalFixIngress(runtime_context, _Resolver())

    async def _scope(db: Any, user: Any) -> list:
        return [workspace_id]

    def _fake_ingress(**kwargs: Any) -> _FinalFixIngress:
        return ingress

    monkeypatch.setattr(agent_admin, "_accessible_scope", _scope)
    monkeypatch.setattr(selector, "build_v2_ingress", _fake_ingress)
    user = _NS(id=uuid4(), is_superadmin=True)
    events = asyncio.run(
        agent_admin._run_v2_eval(graph, object(), user, "quyết định nào?", None)
    )
    return events, ingress, leases


def test_admin_v2_eval_surfaces_clarify_without_502(monkeypatch):
    """F4: a top-level suspend (RETURNED ``__interrupt__``) is a question."""
    request = _final_fix_clarification_request()
    graph = _FinalFixGraph(
        {"__interrupt__": ({"clarify": True},), "clarification": request},
        checkpoint={"clarification": request},
        next_nodes=("clarify_wait",),
    )
    events, ingress, leases = _run_admin_v2_eval_with_graph(monkeypatch, graph)
    terminals = [ev for ev in events if ev["event"] in ("complete", "error")]
    assert len(terminals) == 1
    assert terminals[0]["event"] == "complete"
    assert terminals[0]["data"]["answer"] == "Which document?"
    # Suspension keeps leases active: never released on interrupt.
    assert leases.released == []
    assert ingress.committed == 1


def test_admin_v2_eval_releases_leases_on_terminal(monkeypatch):
    """F4: a terminal admin turn releases the run's leases (no linger)."""
    import asyncio

    from app.services.agents.v2.contracts.base import CONTRACT_VERSION
    from app.services.agents.v2.contracts.response import FinalResponse

    graph = _FinalFixGraph(
        {
            "final_response": FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="success",
                content="Verified answer.",
                citations=(),
            )
        },
        checkpoint={},
        next_nodes=(),
    )
    events, ingress, leases = _run_admin_v2_eval_with_graph(monkeypatch, graph)
    terminals = [ev for ev in events if ev["event"] in ("complete", "error")]
    assert len(terminals) == 1
    assert terminals[0]["event"] == "complete"
    assert leases.released == [("run-admin-eval", "terminal")]
    assert ingress.committed == 1


def test_standalone_stream_has_no_dead_scope_override():
    """F2: /agent-lg/stream serves the authenticated scope (no dead read)."""
    source = _read("app/api/chat_agent_lg.py")
    assert 'getattr(request, "workspace_ids"' not in source
    assert 'hasattr(request, "workspace_ids")' not in source


def test_dead_v2_turn_runner_removed():
    """F8: the uncalled interim ``run_v2_turn_sse`` runner is gone."""
    import app.services.agent.runtime_selector as selector

    assert not hasattr(selector, "run_v2_turn_sse")

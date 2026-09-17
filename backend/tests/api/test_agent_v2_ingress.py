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

    async def _fake_current(db, document_id_arg, workspace_id_arg, *, require_vectors=False):
        assert document_id_arg == document_id
        assert workspace_id_arg == workspace_id
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
        document_views, "load_current_revision_identity_for_workspace", _fake_current
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


# ---------------------------------------------------------------------------
# P0 live-gate fix round 1 (F1/F2): pinned-target wiring coverage.
#
# F1 kills deleting ``pinned_target_resolver=plan_resolver`` in
# ``build_v2_ingress``: the identity assertion fails and the fresh scoped
# dispatch below denies without ever reaching the provider. F2 kills
# deleting the ``build_runtime_services`` pass-through seam the same way.
# ---------------------------------------------------------------------------


def test_build_runtime_services_passes_through_pinned_resolver():
    """F2: the single construction seam forwards the resolver by identity."""
    from app.services.agents.supervisor_v2 import build_runtime_services

    sentinel = object()
    services = build_runtime_services(pinned_target_resolver=sentinel)
    assert services.pinned_target_resolver is sentinel


def test_build_runtime_services_defaults_pinned_resolver_to_none():
    """F2: absent resolver defaults to None (targetless plans stay supported)."""
    from app.services.agents.supervisor_v2 import build_runtime_services

    assert build_runtime_services().pinned_target_resolver is None


def test_ingress_pinned_resolver_wired_to_dispatch():
    """F1: ingress wires the exact resolver; a fresh scoped turn reaches the provider.

    Builds the REAL ``build_v2_ingress`` (no ingress bypass), asserts the
    runtime-only ``pinned_target_resolver`` service ``is`` the ingress
    ``plan_resolver`` the document capabilities resolve through, compiles the
    real supervisor graph against ``InMemorySaver`` (sole-scheduler shape),
    then dispatches a fresh non-resume hard-scoped ``document.retrieve``
    request through the WIRED registry via the shared ``TaskScheduler`` with
    test-double IO backings (retrieval/evidence/leases/semantic/binding)
    injected around — never instead of — the wired resolver.
    """
    import asyncio
    from uuid import UUID, uuid4

    from langgraph.checkpoint.memory import InMemorySaver

    import app.services.agent.runtime_selector as selector
    from app.services.agents.supervisor_v2 import create_supervisor_v2_graph
    from app.services.agents.v2.capabilities.document import RevisionRetrievedChunk
    from app.services.agents.v2.contracts.binding import (
        DocumentBindingSet,
        ScopedDocument,
    )
    from app.services.agents.v2.contracts.capability import DocumentRetrieveInput
    from app.services.agents.v2.contracts.evidence import EvidenceUseRef
    from app.services.agents.v2.contracts.locators import (
        ChunkRangeLocator,
        SectionLocator,
    )
    from app.services.agents.v2.contracts.planning import (
        InitialTaskOrigin,
        TargetUnit,
        TaskPlan,
        TaskSpec,
    )
    from app.services.agents.v2.execution.scheduler import TaskScheduler

    workspace_id = uuid4()
    document_id = UUID("11111111-1111-1111-1111-111111111111")
    revision = "22222222-2222-2222-2222-222222222222"

    class _FakeSession:
        async def commit(self):
            return None

        async def rollback(self):
            return None

        async def close(self):
            return None

    async def _fake_preprocess(raw_query: str):
        from types import SimpleNamespace as _NS

        return _NS(normalized_query=raw_query)

    class _FakeRetrieval:
        def __init__(self, chunks):
            self._chunks = tuple(chunks)
            self.calls: list[dict] = []

        async def retrieve(self, query, *, top_k, allowed_targets, workspace_ids):
            self.calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "allowed_targets": allowed_targets,
                    "workspace_ids": workspace_ids,
                }
            )
            return self._chunks

    class _FakeEvidence:
        def __init__(self):
            self.calls: list[dict] = []

        async def persist_use(
            self, *, source, content, provenance, task_id, purpose, target_id
        ):
            ref = EvidenceUseRef(use_id=uuid4())
            self.calls.append(
                {
                    "task_id": task_id,
                    "purpose": purpose,
                    "target_id": target_id,
                    "use_id": ref.use_id,
                }
            )
            return ref

    class _FakeLeaseSession:
        def __init__(self, events):
            self._events = events

        async def commit(self):
            self._events.append("commit")

    class _FakeLeases:
        def __init__(self):
            self.events: list[str] = []
            self.calls: list[tuple] = []
            self.session = _FakeLeaseSession(self.events)

        async def acquire_or_refresh(self, run_id, revision_id=None, evidence_use_id=None, **kwargs):
            self.calls.append((run_id, revision_id, evidence_use_id))
            self.events.append(f"acquire:{evidence_use_id}")
            return None

    async def _main():
        async with selector.build_v2_ingress(
            user_id=uuid4(),
            authenticated_workspace_ids=[workspace_id],
            requested_workspace_ids=None,
            raw_query="\u0110i\u1ec1u 5 n\u00f3i g\u00ec?",
            thread_id=f"thread-{uuid4().hex[:8]}",
            session_factory=lambda: _FakeSession(),
            lease_session_factory=lambda: _FakeSession(),
            preprocess=_fake_preprocess,
            available_services=frozenset(
                {"v1-people", "v1-document-search", "v1-revision-retrieval"}
            ),
        ) as ingress:
            # The wiring under repair: exact instance identity.
            assert (
                ingress.runtime_context.services.pinned_target_resolver
                is ingress.plan_resolver
            )
            # Compiled-graph shape: the sole-scheduler dispatch path exists.
            compiled = create_supervisor_v2_graph(InMemorySaver())
            assert "execute" in compiled.get_graph().nodes
            # Inject IO doubles around (never instead of) the wired resolver.
            registry = ingress.runtime_context.services.capability_registry
            capability = registry.get("document.retrieve")
            assert capability._resolver is ingress.plan_resolver
            chunks = (
                RevisionRetrievedChunk(
                    document_id=document_id,
                    document_revision=revision,
                    locator=ChunkRangeLocator(
                        kind="chunk_range", start="c1", end="c1"
                    ),
                    content="secret chunk",
                    score=0.9,
                    target_id="t1",
                ),
            )
            retrieval = _FakeRetrieval(chunks)
            evidence = _FakeEvidence()
            capability._service = retrieval
            capability._evidence = evidence
            leases = _FakeLeases()
            ingress.runtime_context.services.retention_leases = leases
            plan = TaskPlan(
                contract_version="2.0",
                plan_id="plan-ingress-pinned",
                goal="factual query",
                target_units=(
                    TargetUnit(
                        target_id="t1",
                        binding_id="b_t1",
                        requested_locator=SectionLocator(
                            kind="section", structure_node_id="node-5"
                        ),
                        completion_criteria=(),
                    ),
                ),
                tasks=(
                    TaskSpec(
                        task_id="T1",
                        capability="document.retrieve",
                        task_objective="factual query",
                        input=DocumentRetrieveInput(
                            kind="document.retrieve",
                            query="lan bmnn",
                            target_ids=("t1",),
                        ),
                        depends_on=(),
                        origin=InitialTaskOrigin(kind="initial"),
                    ),
                ),
            )
            bindings = DocumentBindingSet(
                bindings=(
                    ScopedDocument(
                        binding_id="b_t1",
                        document_id=document_id,
                        document_revision=revision,
                        role="target",
                    ),
                ),
                revision_requirement_refs=(),
            )
            report = await TaskScheduler(registry).execute(
                plan, ingress.runtime_context, bindings=bindings
            )
            return report, retrieval, evidence

    report, retrieval, evidence = asyncio.run(_main())
    assert len(report.results) == 1
    result = report.results[0]
    assert result.status == "success"
    assert result.data is not None
    assert result.data.kind == "document.retrieve"
    assert result.data.retrieved_unit_count == 1
    assert len(retrieval.calls) == 1
    assert len(result.evidence_uses) == 1
    assert evidence.calls[0]["purpose"] == "coverage"
    assert evidence.calls[0]["target_id"] == "t1"
    assert "secret chunk" not in result.model_dump_json()


def test_fresh_resolver_graph_turn_reaches_provider():
    """N2: a TRUE fresh (non-resume) scoped turn through the compiled graph.

    Actual ``build_v2_ingress`` runtime + ``create_supervisor_v2_graph``
    (``InMemorySaver``), driven with ``graph.ainvoke`` — no manual
    ``TaskScheduler`` in this test. The turn flows through the real nodes
    (context -> binding -> semantic -> route -> complex plan -> shared
    execute) into the REAL ``DocumentRetrieveCapability`` with test-double
    IO backings (retrieval provider / leases / hydrator / binding
    resolver) injected around — never instead of — the wired resolver.
    Evidence goes through the REAL ``GovernorEvidenceBuilder`` over a
    recording governor that enforces the production revision invariant
    (document evidence REQUIRES the authoritative ``revision_id``), so
    deleting the ``revision_id`` kwarg from the builder turns this
    terminal success into a failed turn. The API-explicit hard scope
    routes to ``complex_research`` (never the read fast path), the
    provider is reached exactly once, the task succeeds with target-bound
    coverage evidence, the terminal response succeeds with a citation,
    and no raw chunk leaks.

    Kills all four seams: deleting the ingress wiring, the supervisor
    pass-through, the scheduler feed, or the builder ``revision_id``
    kwarg turns this terminal success into a raise or a denied/failed
    turn.
    """
    import asyncio
    from types import SimpleNamespace
    from uuid import UUID, uuid4

    from langgraph.checkpoint.memory import InMemorySaver

    import app.services.agent.runtime_selector as selector
    from app.services.agents.supervisor_v2 import create_supervisor_v2_graph
    from app.services.agents.v2.capabilities.document import RevisionRetrievedChunk
    from app.services.agents.v2.contracts.binding import (
        DocumentBindingSet,
        ScopedDocument,
    )
    from app.services.agents.v2.contracts.evidence import (
        DocumentSourceIdentity,
        EvidenceUseRef,
    )
    from app.services.agents.v2.contracts.locators import ChunkRangeLocator
    from app.services.agents.v2.contracts.request import KnownDocumentResource
    from app.services.agents.v2.nodes.evaluate import (
        AnswerDraftChannel,
        HydratedEvidence,
    )
    from app.services.agents.v2.contracts.synthesis import (
        ParsedCandidate,
        ParsedClaim,
    )
    from app.services.agents.v2.synthesis.citations import (
        ResolvedDocumentCitation,
    )

    workspace_id = uuid4()
    document_id = UUID("11111111-1111-1111-1111-111111111111")
    revision = "22222222-2222-2222-2222-222222222222"
    thread_id = f"thread-{uuid4().hex[:8]}"

    class _FakeSession:
        async def commit(self):
            return None

        async def rollback(self):
            return None

        async def close(self):
            return None

    async def _fake_preprocess(raw_query: str):
        return SimpleNamespace(
            normalized_query="Điều 5 của A nói gì?",
            abbreviations=(),
            document_refs=(),
            blocking_ambiguities=(),
        )

    class _FakeRetrieval:
        def __init__(self, chunks):
            self._chunks = tuple(chunks)
            self.calls: list[dict] = []

        async def retrieve(self, query, *, top_k, allowed_targets, workspace_ids):
            self.calls.append(
                {"query": query, "allowed_targets": allowed_targets}
            )
            return self._chunks

    class _RecordingRepository:
        def __init__(self):
            self.envelopes: list = []

        async def append_use(self, envelope):
            self.envelopes.append(envelope)
            return envelope.use.use_id

    class _RecordingGovernor:
        """Recording governor enforcing the production revision invariant.

        Mirrors ``EvidenceGovernor._require_valid_governance_fields`` for
        the revision rule so deleting the builder ``revision_id`` kwarg
        fails this test exactly like the live ``EvidenceValidationError``.
        """

        def __init__(self):
            from app.services.agents.v2.evidence_store.governance import (
                EvidenceValidationError,
            )

            self._validation_error = EvidenceValidationError
            self.record_calls: list[dict] = []
            self.repository = _RecordingRepository()

        async def persist_record(
            self, *, source, content, provenance, revision_id=None, **kwargs
        ):
            from app.services.agents.v2.contracts.evidence import (
                DocumentSourceIdentity,
            )

            if isinstance(source, DocumentSourceIdentity):
                if revision_id is None:
                    raise self._validation_error(
                        "document evidence must reference the authoritative "
                        "revision_id it was read from (workspace resolves "
                        "through the revision, never a copied allowlist)"
                    )
            elif revision_id is not None:
                raise self._validation_error(
                    f"{source.kind!r} evidence must not carry a document "
                    "revision_id"
                )
            self.record_calls.append(
                {"source": source, "revision_id": revision_id}
            )
            return uuid4()

        async def persist_people_evidence(self, **kwargs):
            raise AssertionError("retrieve pilot persists document evidence")

    class _FakeLeaseSession:
        def __init__(self, events):
            self._events = events
            self.commits = 0

        async def commit(self):
            self.commits += 1
            self._events.append("commit")

    class _FakeLeases:
        def __init__(self):
            self.events: list[str] = []
            self.calls: list[tuple] = []
            self.session = _FakeLeaseSession(self.events)

        async def acquire_or_refresh(
            self, run_id, revision_id=None, evidence_use_id=None, **kwargs
        ):
            self.calls.append((run_id, revision_id, evidence_use_id))
            return {"run_id": run_id}

        async def release_run(self, run_id, reason="terminal"):
            return 0

    class _FakeBindings:
        async def resolve(self, document_refs, capability_runtime):
            assert len(document_refs) == 1
            assert document_refs[0].ref_id == "api_explicit:doc1"
            return DocumentBindingSet(
                bindings=(
                    ScopedDocument(
                        binding_id="b_api_explicit:doc1",
                        document_id=document_id,
                        document_revision=revision,
                        role="target",
                    ),
                ),
                revision_requirement_refs=(),
            )

    class _FakeHydrator:
        def __init__(self, governor):
            self._governor = governor

        async def hydrate_for_evaluation(
            self, use_refs, *, runtime, plan, bindings
        ):
            admitted = []
            for ref in use_refs:
                envelope = next(
                    item
                    for item in self._governor.repository.envelopes
                    if item.use.use_id == ref.use_id
                )
                use = envelope.use
                unit = plan.target_units[0]
                binding = next(
                    item
                    for item in bindings.bindings
                    if item.binding_id == unit.binding_id
                )
                admitted.append(
                    HydratedEvidence(
                        use_id=ref.use_id,
                        evidence_id=uuid4(),
                        task_id=use.task_id,
                        purpose=use.purpose,
                        target_id=use.target_id,
                        content="content of t1",
                        role=binding.role,
                        source_label="t1",
                        source_identity=DocumentSourceIdentity(
                            kind="document",
                            document_id=binding.document_id,
                            document_revision=binding.document_revision,
                            locator=unit.requested_locator,
                        ),
                        classification="normal",
                        locator=unit.requested_locator,
                        document_revision=binding.document_revision,
                    )
                )
            return tuple(admitted)

        async def hydrate_for_synthesis(
            self, use_refs, *, runtime, plan, bindings, budget
        ):
            return await self.hydrate_for_evaluation(
                use_refs, runtime=runtime, plan=plan, bindings=bindings
            )

        async def persist_derived_summary(self, **kwargs):
            raise AssertionError("retrieve pilot never persists derived summaries")

    async def _main():
        async with selector.build_v2_ingress(
            user_id=uuid4(),
            authenticated_workspace_ids=[workspace_id],
            requested_workspace_ids=None,
            raw_query="Điều 5 của A nói gì?",
            thread_id=thread_id,
            session_factory=lambda: _FakeSession(),
            lease_session_factory=lambda: _FakeSession(),
            preprocess=_fake_preprocess,
            known_documents=(
                KnownDocumentResource(
                    resource_id="doc1",
                    document_id=document_id,
                    source="api_explicit",
                ),
            ),
            available_services=frozenset(
                {"v1-people", "v1-document-search", "v1-revision-retrieval"}
            ),
        ) as ingress:
            # The wiring under repair: exact instance identity.
            assert (
                ingress.runtime_context.services.pinned_target_resolver
                is ingress.plan_resolver
            )
            # Inject IO doubles around (never instead of) the wired resolver.
            registry = ingress.runtime_context.services.capability_registry
            capability = registry.get("document.retrieve")
            assert capability._resolver is ingress.plan_resolver
            chunks = (
                RevisionRetrievedChunk(
                    document_id=document_id,
                    document_revision=revision,
                    locator=ChunkRangeLocator(
                        kind="chunk_range", start="c1", end="c1"
                    ),
                    content="secret chunk",
                    score=0.9,
                    target_id="t1",
                ),
            )
            retrieval = _FakeRetrieval(chunks)
            governor = _RecordingGovernor()
            evidence = selector.GovernorEvidenceBuilder(
                governor,
                run_id=ingress.runtime_context.capability_runtime.run_id,
            )
            capability._service = retrieval
            capability._evidence = evidence
            ingress.runtime_context.services.retention_leases = _FakeLeases()
            ingress.runtime_context.services.binding_resolver = _FakeBindings()
            ingress.runtime_context.services.evidence_hydrator = _FakeHydrator(
                governor
            )
            ingress.runtime_context.services.answer_draft_channel = (
                AnswerDraftChannel()
            )
            # The grounded-LLM synthesis subgraph runs for real (manifest,
            # claim-first grounding, CitationProjector, renderer); only the
            # two new runtime seams are deterministic doubles — the live
            # model and the document store are not part of this test's
            # contract.
            class _FakeDraftBuilder:
                async def build(
                    self,
                    query_text,
                    evidence_items,
                    *,
                    repair_context=None,
                    on_claim=None,
                    on_delta=None,
                ):
                    return ParsedCandidate(
                        claims=(
                            ParsedClaim(
                                claim_id="claim-1",
                                text=(
                                    "Nội dung đã được truy xuất từ tài liệu."
                                ),
                                handles=("E1",),
                                presentation="summary",
                            ),
                        )
                    )

            class _FakeCitationResolver:
                async def resolve_document(self, source, *, content):
                    return ResolvedDocumentCitation(
                        document_id=str(source.document_id),
                        document_revision=str(source.document_revision),
                        chunk_id="c1",
                        content=content,
                        source_file="a.pdf",
                    )

                async def resolve_lineage(self, evidence_id):
                    return None

            ingress.runtime_context.services.answer_draft_builder = (
                _FakeDraftBuilder()
            )
            ingress.runtime_context.services.citation_resolver = (
                _FakeCitationResolver()
            )
            graph = create_supervisor_v2_graph(InMemorySaver())
            out = await graph.ainvoke(
                ingress.initial_state,
                {"configurable": {"thread_id": thread_id}},
                context=ingress.runtime_context,
            )
            return out, retrieval, governor

    out, retrieval, governor = asyncio.run(_main())
    # The API-explicit hard scope routes to complex research, never fast read.
    assert out["route_decision"].route == "complex_research"
    assert out["query_analysis"].work_type == "retrieve"
    # The shared scheduler fed the wired resolver: provider reached once.
    assert len(retrieval.calls) == 1
    # Task success with target-bound coverage evidence.
    results = out["execution"].task_results
    assert len(results) == 1
    assert results[0].status == "success"
    assert results[0].data.kind == "document.retrieve"
    assert len(results[0].evidence_uses) == 1
    assert len(governor.record_calls) == 1
    assert governor.record_calls[0]["revision_id"] == UUID(revision)
    assert len(governor.repository.envelopes) == 1
    assert governor.repository.envelopes[0].use.purpose == "coverage"
    assert governor.repository.envelopes[0].use.target_id == "t1"
    # Terminal citation success with no raw chunk anywhere.
    final = out["final_response"]
    status = final.status if hasattr(final, "status") else final["status"]
    assert status == "success"
    citations = (
        final.citations if hasattr(final, "citations") else final["citations"]
    )
    assert len(citations) >= 1
    assert "secret chunk" not in final.model_dump_json()
    assert "secret chunk" not in results[0].model_dump_json()

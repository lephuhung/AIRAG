"""Task 8 — v2 SSE / cancellation / resume compatibility tests.

Preserves the EXISTING SSE contract used by the frontend
(``frontend/src/hooks/useRAGChatStream.ts``): event types
status/thinking/sources/images/token/token_rollback/
potential_abbreviations/people_data/error/complete.

Rulings pinned here:

- Only the OUTER adapter streams user-facing prose (chunked terminal
  content); v2 nodes never push token events.
- Exactly ONE terminal event per run (``complete`` for success/clarify,
  ``error`` for denied/insufficient/error).
- ``token_rollback`` clears every accumulator; a deadline-truncated
  dispatch (``undispatched_tasks`` non-empty) produces
  token_rollback + error, never a normal success.
- Disconnect/cancel prevents success: ``CancelledError`` propagates, no
  terminal event is emitted, leases release as ``cancelled``.
- Stable thread id resumes from the checkpoint via
  ``graph.ainvoke(Command(resume=<resolution>), config, context=runtime)``
  — graph owns navigation (no outer ``goto``). On resume the runner
  refreshes existing leases and feeds the plan resolver from the
  checkpointed plan/bindings, and the CURRENT runtime ACL is reinjected.
- Terminal lease release (``release_run``) happens in the outer runner
  only AFTER the terminal checkpoint succeeds — never in the finalizer,
  never on interrupt.
- ``ClarificationUnsatisfiable`` means fresh turn, not a resume retry.
- ``validate_supervisor_state`` never runs on a non-success terminal
  whose semantic is still the ingress placeholder.

All fakes (no DB, no network).
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.response import FinalResponse


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _future_deadline() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=10)


class FakeLeaseRepo:
    """Stand-in for ``RevisionRetentionLeaseRepository`` (records calls)."""

    def __init__(self) -> None:
        self.released: list[tuple[str, str]] = []
        self.refreshed: list[tuple[str, Any, Any]] = []
        self.commits = 0
        self.session = self

    async def release_run(self, run_id: str, reason: str = "terminal") -> int:
        self.released.append((run_id, reason))
        return 1

    async def acquire_or_refresh(
        self,
        run_id: str,
        revision_id: Any = None,
        evidence_use_id: Any = None,
    ) -> Any:
        self.refreshed.append((run_id, revision_id, evidence_use_id))
        return SimpleNamespace(lease_id=uuid4())

    async def commit(self) -> None:
        self.commits += 1

    async def flush(self) -> None:  # pragma: no cover - interface parity
        return None


def _runtime(
    *,
    run_id: str = "run-t8-test",
    workspace_ids: tuple = (),
    leases: FakeLeaseRepo | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        capability_runtime=SimpleNamespace(
            run_id=run_id,
            workspace_ids=tuple(workspace_ids or (uuid4(),)),
            deadline_at=_future_deadline(),
        ),
        services=SimpleNamespace(retention_leases=leases or FakeLeaseRepo()),
    )


def _success_response(content: str = "Verified answer.") -> FinalResponse:
    return FinalResponse(
        contract_version=CONTRACT_VERSION,
        status="success",
        content=content,
        citations=(),
    )


class FakeGraph:
    """Scripted stand-in for the compiled supervisor v2 graph.

    ``script`` items: ``("return", state)`` | ``("interrupt", exc)`` |
    ``("cancel",)``. Records every ``(payload, config, context)`` call.
    """

    def __init__(self, script: list, *, checkpoint: dict | None = None) -> None:
        self.script = list(script)
        self.calls: list[tuple[Any, dict, Any]] = []
        self.checkpoint = checkpoint or {}

    async def ainvoke(self, payload: Any, config: Any, context: Any = None) -> Any:
        self.calls.append((payload, config, context))
        action = self.script.pop(0)
        if action[0] == "return":
            return action[1]
        if action[0] == "interrupt":
            raise action[1]
        if action[0] == "cancel":
            raise asyncio.CancelledError()
        raise AssertionError(f"unknown script action {action[0]!r}")

    async def get_state(self, config: Any) -> Any:
        self.last_get_state_config = config
        return SimpleNamespace(values=dict(self.checkpoint))


def _terminal_events(events: list[dict]) -> list[dict]:
    return [ev for ev in events if ev["event"] in ("complete", "error")]


async def _collect(agen) -> list[dict]:
    return [ev async for ev in agen]


# ---------------------------------------------------------------------------
# 1. SSE vocabulary: all nine event types survive the adapter protocol
# ---------------------------------------------------------------------------


def test_v2_complete_carries_all_frontend_keys():
    from app.services.agent import streaming as convert_streaming

    # The complete payload is the v1 shape (frontend reads these keys) plus
    # additive v2 fields — never a renamed contract.
    event, data = convert_streaming._v2_terminal_event(_success_response("hi"))
    assert event == "complete"
    for key in (
        "answer",
        "sources",
        "images",
        "potential_abbreviations",
        "people_data",
    ):
        assert key in data, f"v1 complete key {key!r} missing"
    assert data["answer"] == "hi"
    # Additive v2 fields must not break the hook (hook ignores unknowns).
    assert data["status"] == "success"
    assert "citations" in data


def test_v2_sse_formatter_round_trips_all_nine_event_types():
    from app.services.agent import streaming as convert_streaming

    import json

    cases = {
        "status": {"step": "generating", "detail": "..."},
        "thinking": {"text": "hmm"},
        "sources": {"sources": [{"document_id": "d1"}]},
        "images": {"image_refs": []},
        "token": {"text": "abc"},
        "token_rollback": {},
        "potential_abbreviations": {"abbreviations": ["BMNN"]},
        "people_data": {"people": [{"id": "p1"}]},
        "error": {"message": "boom"},
        "complete": {"answer": "done", "sources": [], "images": []},
    }
    for event, data in cases.items():
        raw = convert_streaming._sse(event, data)
        assert raw.startswith(f"event: {event}\n")
        assert json.loads(raw.split("data: ", 1)[1]) == data


def test_only_outer_adapter_streams_prose():
    """No v2 node pushes token events; adapter tokens == chunked content."""
    import asyncio
    from pathlib import Path

    import app.services.agent.streaming as convert_streaming

    v2_root = Path(convert_streaming.__file__).resolve().parents[1] / "agents" / "v2"
    offenders = []
    for path in sorted(v2_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        if 'push_event' in text and '"token"' in text.replace("'", '"'):
            offenders.append(str(path))
    assert offenders == [], f"v2 nodes must never push token events: {offenders}"

    content = "x" * 300
    graph = FakeGraph([("return", {"final_response": _success_response(content)})])
    runtime = _runtime()
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=runtime,
                thread_id="thread-outer-prose",
                initial_state={"request": "q"},
                token_chunk_size=100,
            )
        )
    )
    tokens = [ev for ev in events if ev["event"] == "token"]
    assert tokens, "outer adapter must stream the terminal prose as tokens"
    assert "".join(t["data"]["text"] for t in tokens) == content
    assert all(len(t["data"]["text"]) <= 100 for t in tokens)


# ---------------------------------------------------------------------------
# 2. Exactly one terminal event per run
# ---------------------------------------------------------------------------


def test_success_emits_exactly_one_complete_terminal():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    graph = FakeGraph([("return", {"final_response": _success_response("ok")})])
    runtime = _runtime()
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=runtime,
                thread_id="thread-one-terminal",
                initial_state={"request": "q"},
            )
        )
    )
    assert _terminal_events(events) == [events[-1]]
    assert events[-1]["event"] == "complete"
    assert events[-1]["data"]["answer"] == "ok"


def test_clarify_terminal_is_complete_with_question():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    clarify = FinalResponse(
        contract_version=CONTRACT_VERSION,
        status="clarify",
        content="Which document do you mean?",
        citations=(),
    )
    graph = FakeGraph([("return", {"final_response": clarify})])
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-clarify",
                initial_state={"request": "q"},
            )
        )
    )
    terminals = _terminal_events(events)
    assert len(terminals) == 1
    assert terminals[0]["event"] == "complete"
    assert terminals[0]["data"]["answer"] == "Which document do you mean?"


def test_denied_terminal_is_single_error():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    denied = FinalResponse(
        contract_version=CONTRACT_VERSION,
        status="denied",
        content="Access denied.",
        citations=(),
    )
    graph = FakeGraph([("return", {"final_response": denied})])
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-denied",
                initial_state={"request": "q"},
            )
        )
    )
    terminals = _terminal_events(events)
    assert len(terminals) == 1
    assert terminals[0]["event"] == "error"
    assert terminals[0]["data"]["message"] == "Access denied."


# ---------------------------------------------------------------------------
# 3. Rollback clears accumulators; truncation never succeeds
# ---------------------------------------------------------------------------


def test_rollback_clears_every_accumulator():
    from app.services.agent import streaming as convert_streaming

    acc = convert_streaming._V2StreamAccumulators()
    acc.on_token("draft ")
    acc.on_sources([{"document_id": "A"}])
    acc.on_images([{"id": "img1"}])
    acc.on_people_data([{"id": "p1"}])
    acc.on_potential_abbreviations(["BMNN"])
    assert acc.answer_text != ""
    assert acc.sources != []

    cleared = acc.on_rollback()
    assert cleared["event"] == "token_rollback"
    assert acc.answer_text == ""
    assert acc.sources == []
    assert acc.images == []
    assert acc.people_data == []
    assert acc.potential_abbreviations == []


def test_deadline_truncated_dispatch_produces_rollback_then_error():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    plan = SimpleNamespace(
        tasks=(SimpleNamespace(task_id="t1"), SimpleNamespace(task_id="t2"))
    )
    results = (SimpleNamespace(task_id="t1"),)
    state = {
        "final_response": _success_response("partial prose must not win"),
        "execution": SimpleNamespace(plan=plan, task_results=results),
    }
    graph = FakeGraph([("return", state)])
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-truncated",
                initial_state={"request": "q"},
            )
        )
    )
    kinds = [ev["event"] for ev in events]
    assert "token_rollback" in kinds
    assert "complete" not in kinds
    terminals = _terminal_events(events)
    assert len(terminals) == 1
    assert terminals[0]["event"] == "error"


# ---------------------------------------------------------------------------
# 4. Cancellation / disconnect never succeed
# ---------------------------------------------------------------------------


def test_cancel_prevents_success_and_releases_as_cancelled():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    leases = FakeLeaseRepo()
    graph = FakeGraph([("cancel",)])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            _collect(
                convert_streaming.stream_v2_turn_events(
                    graph=graph,
                    runtime_context=_runtime(leases=leases),
                    thread_id="thread-cancel",
                    initial_state={"request": "q"},
                )
            )
        )
    assert leases.released == [("run-t8-test", "cancelled")]


def test_disconnect_mid_run_emits_no_terminal():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    leases = FakeLeaseRepo()

    async def _run() -> list[dict]:
        started = asyncio.Event()

        class SlowGraph(FakeGraph):
            async def ainvoke(self, payload, config, context=None):
                self.calls.append((payload, config, context))
                started.set()
                await asyncio.sleep(30)
                return {"final_response": _success_response("late")}

        agen = convert_streaming.stream_v2_turn_events(
            graph=SlowGraph([]),
            runtime_context=_runtime(leases=leases),
            thread_id="thread-disconnect",
            initial_state={"request": "q"},
        )
        seen: list[dict] = []
        try:
            async for ev in agen:
                seen.append(ev)
                if started.is_set():
                    await agen.aclose()
                    break
        finally:
            pass
        return seen

    seen = asyncio.run(_run())
    assert _terminal_events(seen) == []
    assert leases.released == [("run-t8-test", "cancelled")]


# ---------------------------------------------------------------------------
# 5. Stable thread resume from the checkpoint (graph owns navigation)
# ---------------------------------------------------------------------------


def test_interrupt_suspends_with_clarify_complete_and_no_release():
    import asyncio

    from langgraph.errors import GraphInterrupt

    import app.services.agent.streaming as convert_streaming
    from app.services.agents.v2.contracts.clarification import ClarificationRequest

    request = ClarificationRequest(
        contract_version=CONTRACT_VERSION,
        clarification_id="clr-1",
        reason="required_document_ambiguous",
        question="Which document?",
        unresolved_ref_ids=("r1",),
        candidates=(),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    graph = FakeGraph(
        [("interrupt", GraphInterrupt([]))],
        checkpoint={"clarification": request},
    )
    leases = FakeLeaseRepo()
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(leases=leases),
                thread_id="thread-stable-1",
                initial_state={"request": "q"},
            )
        )
    )
    terminals = _terminal_events(events)
    assert len(terminals) == 1
    assert terminals[0]["event"] == "complete"
    assert terminals[0]["data"]["answer"] == "Which document?"
    # Suspension keeps leases active: never released on interrupt.
    assert leases.released == []
    # Stable thread id on the wire.
    assert graph.calls[0][1] == {"configurable": {"thread_id": "thread-stable-1"}}


def test_resume_uses_command_verbatim_on_stable_thread():
    import asyncio

    from langgraph.types import Command

    import app.services.agent.streaming as convert_streaming

    command = Command(resume={"clarification_id": "clr-1"})
    plan = SimpleNamespace(tasks=())
    checkpoint = {
        "execution": SimpleNamespace(plan=plan, task_results=()),
        "bindings": SimpleNamespace(bindings=()),
    }
    graph = FakeGraph(
        [("return", {"final_response": _success_response("resumed")})],
        checkpoint=checkpoint,
    )
    fed: list = []

    class _Resolver:
        def feed(self, plan_arg, bindings_arg):
            fed.append((plan_arg, bindings_arg))

    runtime = _runtime()
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=runtime,
                thread_id="thread-stable-1",
                resume_command=command,
                plan_resolver=_Resolver(),
            )
        )
    )
    assert len(graph.calls) == 1
    payload, config, context = graph.calls[0]
    # Verbatim resume: the exact Command object, same thread, live runtime.
    assert payload is command
    assert config == {"configurable": {"thread_id": "thread-stable-1"}}
    assert context is runtime
    # Resolver fed from the checkpointed plan/bindings before dispatch.
    assert fed == [(plan, checkpoint["bindings"])]
    assert _terminal_events(events)[0]["event"] == "complete"


def test_resume_refreshes_leases_before_dispatch_and_reinjects_acl():
    import asyncio

    from langgraph.types import Command

    import app.services.agent.streaming as convert_streaming

    use_id = uuid4()
    plan = SimpleNamespace(tasks=(SimpleNamespace(task_id="t1"),))
    results = (
        SimpleNamespace(task_id="t1", evidence_uses=(SimpleNamespace(use_id=use_id),)),
    )
    checkpoint = {
        "execution": SimpleNamespace(plan=plan, task_results=results),
        "bindings": SimpleNamespace(bindings=()),
    }
    graph = FakeGraph(
        [("return", {"final_response": _success_response("ok")})],
        checkpoint=checkpoint,
    )
    order: list[str] = []
    leases = FakeLeaseRepo()
    orig_refresh = leases.acquire_or_refresh

    async def _tracked(run_id, revision_id=None, evidence_use_id=None):
        order.append("refresh")
        return await orig_refresh(run_id, revision_id, evidence_use_id)

    leases.acquire_or_refresh = _tracked  # type: ignore[method-assign]
    orig_ainvoke = graph.ainvoke

    async def _tracked_invoke(payload, config, context=None):
        order.append("invoke")
        return await orig_ainvoke(payload, config, context)

    graph.ainvoke = _tracked_invoke  # type: ignore[method-assign]

    narrowed = (uuid4(),)
    runtime = _runtime(workspace_ids=narrowed, leases=leases)
    asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=runtime,
                thread_id="thread-acl",
                resume_command=Command(resume={"clarification_id": "clr-1"}),
            )
        )
    )
    assert order[0] == "refresh"
    assert "invoke" in order
    assert (leases.refreshed[0][0], leases.refreshed[0][2]) == (
        "run-t8-test",
        use_id,
    )
    # Current ACL is reinjected: the graph sees the resume-time runtime,
    # never checkpointed scope.
    assert graph.calls[0][2].capability_runtime.workspace_ids == narrowed


def test_adapter_resume_form_owns_no_goto():
    import inspect

    import app.services.agent.streaming as convert_streaming

    source = inspect.getsource(convert_streaming.stream_v2_turn_events)
    assert "Command(resume=" in source or "resume_command" in source
    assert "goto=" not in source
    # The adapter passes the resume Command through verbatim.
    assert "context=runtime_context" in source


# ---------------------------------------------------------------------------
# 6. Runner contracts: fresh turn, placeholder terminals, lease release
# ---------------------------------------------------------------------------


def test_unsatisfiable_reply_is_fresh_turn_not_resume_retry():
    import asyncio

    import app.services.agent.streaming as convert_streaming
    from app.services.agents.v2.contracts.clarification import ClarificationRequest
    from app.services.agents.v2.nodes.clarification import ClarificationUnsatisfiable

    request = ClarificationRequest(
        contract_version=CONTRACT_VERSION,
        clarification_id="clr-empty",
        reason="semantic_ambiguity",
        question="What do you mean?",
        unresolved_ref_ids=(),
        candidates=(),  # candidate-free: every reply is unsatisfiable
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    from app.services.agents.v2.contracts.capability import (
        CapabilityRuntimeContext,
    )
    from app.services.agents.v2.contracts.state import (
        GraphRuntimeContext,
        RuntimeServices,
    )

    class _Messages:
        async def get_user_message(self, message_id):
            return SimpleNamespace(content="whatever the user said")

    runtime = GraphRuntimeContext(
        capability_runtime=CapabilityRuntimeContext(
            request_id="req-t8-unsat",
            run_id="run-t8-test",
            user_id=uuid4(),
            workspace_ids=(),
            can_read_people=False,
            allowed_capabilities=frozenset(),
            deadline_at=_future_deadline(),
        ),
        services=RuntimeServices(
            chat_messages=_Messages(),
            authorization=None,
        ),
    )
    with pytest.raises(convert_streaming.V2FreshTurnRequired):
        asyncio.run(
            convert_streaming.prepare_v2_resume_command(
                message_id=uuid4(),
                request=request,
                runtime_context=runtime,
            )
        )
    # The runner contract behind it: candidate-free means unsatisfiable.
    with pytest.raises(ClarificationUnsatisfiable):
        from app.services.agents.v2.nodes.clarification import (
            parse_clarification_resolution,
        )

        parse_clarification_resolution("anything", request)


def test_placeholder_terminal_never_validates():
    import asyncio
    import inspect

    import app.services.agent.streaming as convert_streaming

    source = inspect.getsource(convert_streaming.stream_v2_turn_events)
    assert "validate_supervisor_state" not in source

    # Non-success terminal over the ingress placeholder semantic surfaces
    # as an error without raising a validation error.
    denied = FinalResponse(
        contract_version=CONTRACT_VERSION,
        status="denied",
        content="Denied.",
        citations=(),
    )
    state = {
        "final_response": denied,
        "semantic": {"contextualized_query": ""},
        "execution": SimpleNamespace(plan=None, task_results=()),
    }
    graph = FakeGraph([("return", state)])
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-placeholder",
                initial_state={"request": "q"},
            )
        )
    )
    assert _terminal_events(events)[0]["event"] == "error"


def test_terminal_release_runs_after_checkpoint_success():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    leases = FakeLeaseRepo()
    graph = FakeGraph([("return", {"final_response": _success_response("ok")})])
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(leases=leases),
                thread_id="thread-release",
                initial_state={"request": "q"},
            )
        )
    )
    assert _terminal_events(events)[0]["event"] == "complete"
    assert leases.released == [("run-t8-test", "terminal")]
    assert leases.commits >= 1


def test_finalizer_never_releases_leases():
    import inspect

    from app.services.agents.v2.nodes import finalizer as finalizer_node

    source = inspect.getsource(finalizer_node)
    assert "release_run" not in source

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
        self.events: list[str] = []
        self.session = self

    async def release_run(self, run_id: str, reason: str = "terminal") -> int:
        self.released.append((run_id, reason))
        self.events.append(f"release:{run_id}:{reason}")
        return 1

    async def acquire_or_refresh(
        self,
        run_id: str,
        revision_id: Any = None,
        evidence_use_id: Any = None,
    ) -> Any:
        self.refreshed.append((run_id, revision_id, evidence_use_id))
        self.events.append(f"acquire:{revision_id}:{evidence_use_id}")
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
    ``("cancel",)`` | ``("raise", exc)``. Records every
    ``(payload, config, context)`` call.
    """

    def __init__(
        self,
        script: list,
        *,
        checkpoint: dict | None = None,
        next_nodes: tuple = (),
    ) -> None:
        self.script = list(script)
        self.calls: list[tuple[Any, dict, Any]] = []
        self.checkpoint = checkpoint or {}
        self.next_nodes = tuple(next_nodes)

    def _snapshot(self, config: Any) -> Any:
        self.last_get_state_config = config
        return SimpleNamespace(
            values=dict(self.checkpoint), next=tuple(self.next_nodes)
        )

    def get_state(self, config: Any) -> Any:
        """Sync snapshot (mirrors the real langgraph 1.0.0 API)."""
        return self._snapshot(config)

    async def aget_state(self, config: Any) -> Any:
        """Async snapshot (mirrors the real langgraph 1.0.0 API)."""
        return self._snapshot(config)

    async def ainvoke(self, payload: Any, config: Any, context: Any = None) -> Any:
        self.calls.append((payload, config, context))
        action = self.script.pop(0)
        if action[0] == "return":
            return action[1]
        if action[0] == "interrupt":
            raise action[1]
        if action[0] == "cancel":
            raise asyncio.CancelledError()
        if action[0] == "raise":
            raise action[1]
        raise AssertionError(f"unknown script action {action[0]!r}")


class FakeStreamingGraph(FakeGraph):
    """FakeGraph with a scripted ``astream`` (langgraph 1.2.x shape).

    ``stream_script`` items are ``(ns, mode, chunk)`` 3-tuples, or
    ``("raise", exc)`` to fail mid-stream. When ``gate`` is given the
    stream blocks on it after the first item (cancel test).
    """

    def __init__(
        self,
        script: list,
        *,
        stream_script: list | None = None,
        gate: "asyncio.Event | None" = None,
        checkpoint: dict | None = None,
        next_nodes: tuple = (),
    ) -> None:
        super().__init__(script, checkpoint=checkpoint, next_nodes=next_nodes)
        self.stream_script = list(stream_script or [])
        self.stream_calls: list[tuple] = []
        self.gate = gate
        self.stream_started = asyncio.Event()

    async def astream(
        self,
        payload,
        config,
        context=None,
        stream_mode=None,
        subgraphs=False,
    ):
        self.stream_calls.append((payload, config, context, stream_mode, subgraphs))
        self.stream_started.set()
        if self.gate is not None:
            await self.gate.wait()
        for item in self.stream_script:
            if item[0] == "raise":
                raise item[1]
            yield item


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


def test_people_snapshot_emits_people_data_before_tokens_and_in_complete():
    """v1 parity: a people lookup surfaces the card records on the wire.

    The request-scoped ``services.people_lookup`` snapshot (raw sanitized
    v1 records captured during dispatch) produces exactly one
    ``people_data`` event BEFORE the first ``token``, and the terminal
    ``complete`` repeats the identical set — the frontend card and the
    session persistence both read these.
    """
    import asyncio

    import app.services.agent.streaming as convert_streaming

    persons = [
        {"hoTen": "Nguyen Van A", "soDienThoai": "0901234567", "_person_group": 1},
        {"hoTen": "Nguyen Van A", "maSoBhxh": "7900012345", "_person_group": 1},
    ]

    class _PeopleLookup:
        def people_display_snapshot(self):
            return persons, "display text"

    runtime = _runtime()
    runtime.services.people_lookup = _PeopleLookup()
    graph = FakeGraph([("return", {"final_response": _success_response("ok")})])
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=runtime,
                thread_id="thread-people-card",
                initial_state={"request": "q"},
            )
        )
    )
    people_events = [ev for ev in events if ev["event"] == "people_data"]
    assert len(people_events) == 1
    assert people_events[0]["data"]["people"] == persons
    first_token = next(i for i, ev in enumerate(events) if ev["event"] == "token")
    assert events.index(people_events[0]) < first_token
    complete = events[-1]
    assert complete["event"] == "complete"
    assert complete["data"]["people_data"] == persons


def test_no_people_snapshot_emits_no_people_data():
    """A non-people turn (or an empty snapshot) emits no people_data."""
    import asyncio

    import app.services.agent.streaming as convert_streaming

    class _PeopleLookup:
        def people_display_snapshot(self):
            return None

    runtime = _runtime()
    runtime.services.people_lookup = _PeopleLookup()
    graph = FakeGraph([("return", {"final_response": _success_response("ok")})])
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=runtime,
                thread_id="thread-no-people",
                initial_state={"request": "q"},
            )
        )
    )
    assert not [ev for ev in events if ev["event"] == "people_data"]
    assert events[-1]["data"]["people_data"] == []


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


def test_interrupt_emits_structured_clarification_required_before_terminal():
    """Task 9 fix round 1 (C1): the suspend turn produces a real
    ``clarification_required`` frame (server-issued option IDs + resume
    metadata) ahead of its single terminal ``complete``.
    """
    import asyncio

    from langgraph.errors import GraphInterrupt

    import app.services.agent.streaming as convert_streaming
    from app.services.agents.v2.contracts.clarification import (
        ClarificationRequest,
        DocumentCandidate,
    )

    request = ClarificationRequest(
        contract_version=CONTRACT_VERSION,
        clarification_id="clr-9",
        reason="required_document_ambiguous",
        question="Which document?",
        unresolved_ref_ids=("r1",),
        candidates=(
            DocumentCandidate(
                candidate_id="cand-1",
                ordinal=1,
                ref_id="r1",
                document_id=uuid4(),
                label="Doc A",
            ),
        ),
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
                thread_id="thread-stable-9",
                initial_state={"request": "q"},
            )
        )
    )
    required = [ev for ev in events if ev["event"] == "clarification_required"]
    assert len(required) == 1
    data = required[0]["data"]
    assert data["clarification_id"] == "clr-9"
    assert data["options"] == [{"option_id": "cand-1", "label": "Doc A"}]
    assert data["resume"]["thread_id"] == "thread-stable-9"
    # Still exactly one terminal, and leases stay active on suspension.
    assert _terminal_events(events)[0]["event"] == "complete"
    assert leases.released == []


def test_resume_uses_command_verbatim_on_stable_thread():
    import asyncio

    from langgraph.types import Command

    import app.services.agent.streaming as convert_streaming
    from app.services.agents.v2.contracts.binding import DocumentBindingSet
    from app.services.agents.v2.contracts.planning import TaskPlan

    command = Command(resume={"clarification_id": "clr-1"})
    plan = TaskPlan(
        contract_version=CONTRACT_VERSION,
        plan_id="plan-1",
        goal="goal",
        target_units=(),
        tasks=(),
    )
    bindings = DocumentBindingSet(bindings=(), revision_requirement_refs=())
    checkpoint = {
        "execution": SimpleNamespace(plan=plan, task_results=()),
        "bindings": bindings,
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
    # Verbatim resume: the exact Command object, same thread, live runtime,
    # and no outer navigation (the graph owns it).
    assert payload is command
    assert not getattr(payload, "goto", None)
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
    order: list[str] = []
    orig_release = leases.release_run

    async def _tracked_release(run_id: str, reason: str = "terminal") -> int:
        order.append("release")
        return await orig_release(run_id, reason)

    leases.release_run = _tracked_release  # type: ignore[method-assign]
    graph = FakeGraph([("return", {"final_response": _success_response("ok")})])
    orig_ainvoke = graph.ainvoke

    async def _tracked_ainvoke(payload, config, context=None):
        result = await orig_ainvoke(payload, config, context)
        order.append("checkpoint")
        return result

    graph.ainvoke = _tracked_ainvoke  # type: ignore[method-assign]
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
    # Release runs after the terminal checkpoint succeeded, never before.
    assert order == ["checkpoint", "release"]


def test_finalizer_never_releases_leases():
    import inspect

    from app.services.agents.v2.nodes import finalizer as finalizer_node

    source = inspect.getsource(finalizer_node)
    assert "release_run" not in source


# ---------------------------------------------------------------------------
# Round 1 — review fixes
# ---------------------------------------------------------------------------


def test_unexpected_graph_failure_emits_rollback_error_and_releases():
    """I1: a raw graph exception becomes a terminal error, never an escape."""
    import asyncio

    import app.services.agent.streaming as convert_streaming

    leases = FakeLeaseRepo()
    graph = FakeGraph([("raise", RuntimeError("provider exploded"))])
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(leases=leases),
                thread_id="thread-unexpected",
                initial_state={"request": "q"},
            )
        )
    )
    kinds = [ev["event"] for ev in events]
    assert kinds == ["status", "token_rollback", "error"]
    assert events[-1]["data"]["message"] == (
        convert_streaming._V2_UNEXPECTED_ERROR_MESSAGE
    )
    assert "provider exploded" not in events[-1]["data"]["message"]
    assert leases.released == [("run-t8-test", "terminal")]


def test_refresh_pairs_anchor_revision_like_scheduler():
    """I2: resume refresh re-acquires the scheduler's (revision, use) rows."""
    from app.services.agents.v2.execution.scheduler import (
        refresh_pairs_for_checkpoint,
    )

    revision = uuid4()
    use_id = uuid4()
    plan = SimpleNamespace(
        target_units=(SimpleNamespace(target_id="t1", binding_id="b1"),),
        tasks=(
            SimpleNamespace(
                task_id="k1", input=SimpleNamespace(target_ids=("t1",))
            ),
        ),
    )
    bindings = SimpleNamespace(
        bindings=(
            SimpleNamespace(
                binding_id="b1",
                document_id=uuid4(),
                document_revision=str(revision),
            ),
        )
    )
    results = (
        SimpleNamespace(
            task_id="k1",
            evidence_uses=(SimpleNamespace(use_id=use_id),),
        ),
    )
    assert refresh_pairs_for_checkpoint(plan, bindings, results) == (
        (revision, use_id),
    )


def test_refresh_pairs_tolerate_serde_mappings_and_targetless_uses():
    """I2: mappings (checkpoint serde) resolve; targetless uses stay (None, use)."""
    from app.services.agents.v2.execution.scheduler import (
        refresh_pairs_for_checkpoint,
    )

    revision = uuid4()
    anchored_use = uuid4()
    lonely_use = uuid4()
    plan = {
        "target_units": [{"target_id": "t1", "binding_id": "b1"}],
        "tasks": [
            {"task_id": "k1", "input": {"target_ids": ["t1"]}},
            {"task_id": "k2", "input": {"target_ids": []}},
        ],
    }
    bindings = {
        "bindings": [
            {
                "binding_id": "b1",
                "document_id": str(uuid4()),
                "document_revision": str(revision),
            }
        ]
    }
    results = (
        {"task_id": "k1", "evidence_uses": [{"use_id": str(anchored_use)}]},
        {"task_id": "k2", "evidence_uses": [{"use_id": str(lonely_use)}]},
        {"task_id": "k1", "evidence_uses": [{"use_id": str(anchored_use)}]},
    )
    assert refresh_pairs_for_checkpoint(plan, bindings, results) == (
        (revision, anchored_use),
        (None, lonely_use),
    )


def test_clarify_terminal_streams_no_prose_tokens():
    """I3: the T7 non-success rule gates prose — clarify arrives whole."""
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
                thread_id="thread-clarify-prose",
                initial_state={"request": "q"},
            )
        )
    )
    assert [ev["event"] for ev in events if ev["event"] == "token"] == []
    terminals = _terminal_events(events)
    assert len(terminals) == 1
    assert terminals[0]["data"]["answer"] == "Which document do you mean?"


def test_success_projects_unknown_abbreviations_and_fabricates_nothing():
    """I5: potential_abbreviations derive from semantics; sources/images/people absent."""
    import asyncio

    import app.services.agent.streaming as convert_streaming

    state = {
        "final_response": _success_response("ok"),
        "semantic": {
            "abbreviations": [
                {"abbreviation": "BMNN", "expansion": None},
                {"abbreviation": "HĐ", "expansion": "hợp đồng"},
            ]
        },
        "execution": SimpleNamespace(plan=None, task_results=()),
    }
    graph = FakeGraph([("return", state)])
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-abbrev",
                initial_state={"request": "q"},
            )
        )
    )
    abbrev = [ev for ev in events if ev["event"] == "potential_abbreviations"]
    assert len(abbrev) == 1
    assert abbrev[0]["data"] == {"abbreviations": ["BMNN"]}
    # No v2 projection exists for these yet (Phase-3-owned) — the adapter
    # must not fabricate them. ``citation`` is the Task-6 grounded-synthesis
    # frame emitted before the first token (empty set on citation-free
    # success); it is part of the pinned public order, not a fabrication.
    assert {
        ev["event"] for ev in events
    } <= {"status", "citation", "token", "potential_abbreviations", "complete"}


def test_v2_entrypoints_use_production_adapter():
    """I4: all three entrypoints stream through the reviewed adapter + resume."""
    from pathlib import Path

    import app.services.agent.streaming as convert_streaming

    root = Path(convert_streaming.__file__).resolve().parents[3]
    v2_sse = (root / "app" / "api" / "chat_agent_lg.py").read_text()
    v2_session = (root / "app" / "api" / "chat_session.py").read_text()
    v2_telegram = (
        root / "app" / "services" / "integrations" / "telegram_service.py"
    ).read_text()
    assert "stream_v2_turn_to_sse" in v2_sse
    assert "stream_v2_turn_to_sse" in v2_session
    assert "stream_v2_turn_events" in v2_telegram
    for source in (v2_sse, v2_session, v2_telegram):
        assert "run_v2_turn_sse" not in source
        assert "resolve_v2_resume_command" in source
        assert "plan_resolver" in source


def test_load_pending_clarification_needs_suspended_thread():
    """Entrypoint resume: pending loads only on a clarify_wait thread."""
    import asyncio

    import app.services.agent.streaming as convert_streaming
    from app.services.agents.v2.contracts.clarification import ClarificationRequest

    request = ClarificationRequest(
        contract_version=CONTRACT_VERSION,
        clarification_id="clr-pend",
        reason="required_document_ambiguous",
        question="Which document?",
        unresolved_ref_ids=("r1",),
        candidates=(),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    suspended = FakeGraph(
        [], checkpoint={"clarification": request}, next_nodes=("clarify_wait",)
    )
    pending = asyncio.run(
        convert_streaming.load_v2_pending_clarification(suspended, "thread-x")
    )
    assert pending is not None
    assert pending.clarification_id == "clr-pend"
    terminal = FakeGraph([], checkpoint={"clarification": request}, next_nodes=())
    assert (
        asyncio.run(
            convert_streaming.load_v2_pending_clarification(terminal, "thread-x")
        )
        is None
    )


def test_resolve_resume_command_valid_reply_and_fresh_turn_fallback():
    """Entrypoint resume: a valid reply builds the Command; unsatisfiable → None."""
    import asyncio

    from app.services.agents.v2.contracts.capability import (
        CapabilityRuntimeContext,
    )
    from app.services.agents.v2.contracts.clarification import (
        ClarificationRequest,
        DocumentCandidate,
    )
    from app.services.agents.v2.contracts.state import (
        GraphRuntimeContext,
        RuntimeServices,
    )

    import app.services.agent.streaming as convert_streaming

    doc_id = uuid4()
    offered = ClarificationRequest(
        contract_version=CONTRACT_VERSION,
        clarification_id="clr-offer",
        reason="required_document_ambiguous",
        question="Which document?",
        unresolved_ref_ids=("r1",),
        candidates=(
            DocumentCandidate(
                candidate_id="c1",
                ordinal=1,
                ref_id="r1",
                document_id=doc_id,
                label="Doc A",
            ),
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    empty = offered.model_copy(update={"candidates": (), "clarification_id": "clr-empty"})

    def _runtime_for(contents: dict) -> Any:
        class _Messages:
            async def get_user_message(self, message_id):
                return SimpleNamespace(content=contents[message_id])

        class _Auth:
            async def require_document(self, document_id, capability_runtime):
                return None

        return GraphRuntimeContext(
            capability_runtime=CapabilityRuntimeContext(
                request_id="req-t8-resolve",
                run_id="run-t8-test",
                user_id=uuid4(),
                workspace_ids=(uuid4(),),
                can_read_people=False,
                allowed_capabilities=frozenset(),
                deadline_at=_future_deadline(),
            ),
            services=RuntimeServices(
                chat_messages=_Messages(), authorization=_Auth()
            ),
        )

    reply_id = uuid4()
    graph = FakeGraph(
        [], checkpoint={"clarification": offered}, next_nodes=("clarify_wait",)
    )
    command = asyncio.run(
        convert_streaming.resolve_v2_resume_command(
            graph=graph,
            thread_id="thread-resolve",
            message_id=reply_id,
            runtime_context=_runtime_for({reply_id: "1"}),
        )
    )
    assert command is not None
    assert not getattr(command, "goto", None)

    graph_empty = FakeGraph(
        [], checkpoint={"clarification": empty}, next_nodes=("clarify_wait",)
    )
    assert (
        asyncio.run(
            convert_streaming.resolve_v2_resume_command(
                graph=graph_empty,
                thread_id="thread-resolve",
                message_id=reply_id,
                runtime_context=_runtime_for({reply_id: "whatever"}),
            )
        )
        is None
    )

    graph_idle = FakeGraph([], checkpoint={}, next_nodes=())
    assert (
        asyncio.run(
            convert_streaming.resolve_v2_resume_command(
                graph=graph_idle,
                thread_id="thread-resolve",
                message_id=reply_id,
                runtime_context=_runtime_for({reply_id: "1"}),
            )
        )
        is None
    )


# ---------------------------------------------------------------------------
# Round 1 C1/C2 — real compiled graph (InMemorySaver), no framework fakes
# ---------------------------------------------------------------------------


def _t8_ambiguous_preprocessing(doc_a, doc_b):
    from app.services.agents.semantic_preprocessor import (
        DocumentCandidate as LegacyCandidate,
    )
    from app.services.agents.semantic_preprocessor import (
        DocumentRefEntry,
        PreprocessingResult,
    )

    return PreprocessingResult(
        original_query="Mở Nghị định 12",
        normalized_query="mở nghị định 12",
        abbreviations=[],
        document_refs=[
            DocumentRefEntry(
                ref_id="r1",
                original_span="Nghị định 12",
                span_offset=(3, 15),
                reference="nghị định 12",
                candidates=[
                    LegacyCandidate(
                        document_id=doc_a,
                        match_basis="exact_number",
                        confidence=0.9,
                    ),
                    LegacyCandidate(
                        document_id=doc_b,
                        match_basis="exact_number",
                        confidence=0.8,
                    ),
                ],
                resolution_status="ambiguous",
            )
        ],
        blocking_ambiguities=[],
        preprocessing_status="ok",
        preprocessor_trace=[],
    )


class _T8RecordingLeases:
    """Lease repo recording anchored identities + releases (real-graph use)."""

    def __init__(self) -> None:
        self.acquired: list[tuple[Any, Any]] = []
        self.released: list[tuple[str, str]] = []
        self.commits = 0
        self.session = self

    async def acquire_or_refresh(
        self, run_id: str, revision_id: Any = None, evidence_use_id: Any = None
    ) -> Any:
        self.acquired.append((revision_id, evidence_use_id))
        return SimpleNamespace(lease_id=uuid4())

    async def release_run(self, run_id: str, reason: str = "terminal") -> int:
        self.released.append((run_id, reason))
        return 1

    async def commit(self) -> None:
        self.commits += 1


class _T8EchoBindings:
    """Pins whatever resolved refs it is given (T6 stand-in)."""

    def __init__(self, revision) -> None:
        self._revision = revision

    async def resolve(self, document_refs: Any, capability_runtime: Any) -> Any:
        from app.services.agents.v2.contracts.binding import (
            DocumentBindingSet,
            ScopedDocument,
        )

        pins = tuple(
            ScopedDocument(
                binding_id=f"b_{ref.ref_id}",
                document_id=ref.resolved_document_id,
                document_revision=str(self._revision),
                role="target",
            )
            for ref in document_refs
            if ref.resolution_status == "resolved"
            and ref.resolved_document_id is not None
        )
        return DocumentBindingSet(bindings=pins, revision_requirement_refs=())


def _t8_real_setup(thread_id: str) -> dict:
    """Real compiled v2 graph + real runtime over an ambiguous query."""
    from datetime import UTC

    from langgraph.checkpoint.memory import InMemorySaver

    from app.services.agents.supervisor_v2 import (
        DeterministicSemanticAdapter,
        build_initial_v2_state,
        create_supervisor_v2_graph,
    )
    from app.services.agents.v2.contracts.capability import (
        CapabilityRuntimeContext,
    )
    from app.services.agents.v2.contracts.request import RequestContext
    from app.services.agents.v2.contracts.state import (
        GraphRuntimeContext,
        RuntimeServices,
    )
    from app.services.agents.v2.nodes.evaluate import AnswerDraftChannel

    doc_a, doc_b, revision = uuid4(), uuid4(), uuid4()
    workspace_id = uuid4()
    leases = _T8RecordingLeases()
    runtime = GraphRuntimeContext(
        capability_runtime=CapabilityRuntimeContext(
            request_id="req-t8-real",
            run_id="run-t8-real",
            user_id=uuid4(),
            workspace_ids=(workspace_id,),
            can_read_people=False,
            allowed_capabilities=frozenset(
                {
                    "people.lookup",
                    "document.search",
                    "document.read",
                    "section.read",
                    "knowledge_graph.query",
                    "memory.lookup",
                    "abbreviation.resolve",
                }
            ),
            deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        services=RuntimeServices(
            retention_leases=leases,
            semantic_adapter=DeterministicSemanticAdapter(
                preprocess=lambda query: _t8_ambiguous_preprocessing(doc_a, doc_b)
            ),
            binding_resolver=_T8EchoBindings(revision),
            answer_draft_channel=AnswerDraftChannel(),
        ),
    )
    graph = create_supervisor_v2_graph(InMemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = build_initial_v2_state(
        request=RequestContext(
            contract_version=CONTRACT_VERSION,
            request_id="req-t8-real",
            thread_id=thread_id,
            original_query="Mở Nghị định 12",
            known_documents=(),
        )
    )
    return {
        "graph": graph,
        "config": config,
        "runtime": runtime,
        "initial_state": initial_state,
        "leases": leases,
        "workspace_id": workspace_id,
        "doc_a": doc_a,
        "doc_b": doc_b,
        "revision": revision,
    }


@pytest.mark.asyncio
async def test_c1_real_clarify_suspend_is_complete_without_release():
    """C1: a real suspend returns (not raises) → question complete, leases kept."""
    import app.services.agent.streaming as convert_streaming

    shared = _t8_real_setup("t8-real-suspend")
    graph, runtime = shared["graph"], shared["runtime"]
    events = [
        ev
        async for ev in convert_streaming.stream_v2_turn_events(
            graph=graph,
            runtime_context=runtime,
            thread_id="t8-real-suspend",
            initial_state=shared["initial_state"],
        )
    ]
    kinds = [ev["event"] for ev in events]
    assert "error" not in kinds, f"suspend must not error: {events[-1:]}"
    terminals = [ev for ev in events if ev["event"] in ("complete", "error")]
    assert len(terminals) == 1
    assert terminals[0]["event"] == "complete"
    stored = graph.get_state(shared["config"])
    assert stored.next == ("clarify_wait",)
    clarification = (stored.values or {}).get("clarification")
    assert clarification is not None
    candidates = (
        clarification.get("candidates")
        if isinstance(clarification, dict)
        else getattr(clarification, "candidates", None)
    )
    question = (
        clarification.get("question")
        if isinstance(clarification, dict)
        else getattr(clarification, "question", None)
    )
    assert candidates is not None and len(candidates) == 2
    assert terminals[0]["data"]["answer"] == question
    # Suspension keeps leases active: never released.
    assert shared["leases"].released == []


@pytest.mark.asyncio
async def test_c2_real_resume_advances_feeds_resolver_and_anchored_refresh():
    """C1+C2 on the real graph: resume advances, then a terminal-thread
    resume feeds the resolver from the real checkpointed plan and refreshes
    the scheduler-anchored (revision, use) pairs."""
    from app.services.agents.v2.capabilities import (
        CapabilityDescriptor,
        CapabilityRegistration,
        build_capability_registry,
    )
    from app.services.agents.v2.contracts.binding import DocumentBindingSet
    from app.services.agents.v2.contracts.capability import DocumentReadOutput
    from app.services.agents.v2.contracts.evidence import (
        DerivedSourceIdentity,
        EvidenceUseRef,
    )
    from app.services.agents.v2.contracts.evaluation import CoverageObservation
    from app.services.agents.v2.contracts.execution import AgentRequest, AgentResult
    from app.services.agents.v2.contracts.locators import DocumentLocator
    from app.services.agents.v2.contracts.planning import TaskPlan
    from app.services.agents.v2.nodes.evaluate import HydratedEvidence

    import app.services.agent.streaming as convert_streaming
    from app.services.agent.runtime_selector import PlanBindingResolver

    shared = _t8_real_setup("t8-real-resume")
    graph, runtime, config = (
        shared["graph"],
        shared["runtime"],
        shared["config"],
    )
    first = [
        ev
        async for ev in convert_streaming.stream_v2_turn_events(
            graph=graph,
            runtime_context=runtime,
            thread_id="t8-real-resume",
            initial_state=shared["initial_state"],
        )
    ]
    assert [ev for ev in first if ev["event"] in ("complete", "error")][
        0
    ]["event"] == "complete"

    use_id = uuid4()

    class _ChatMessages:
        def __init__(self, contents: dict) -> None:
            self.contents = dict(contents)

        async def get_user_message(self, message_id):
            return SimpleNamespace(content=self.contents[message_id])

    class _Authorization:
        def __init__(self, grants: dict) -> None:
            self.grants = grants

        async def require_document(self, document_id, capability_runtime):
            allowed = any(
                document_id in self.grants.get(workspace, set())
                for workspace in capability_runtime.workspace_ids
            )
            if not allowed:
                raise PermissionError(f"document {document_id} is not authorized")

    class _DocRead:
        def __init__(self) -> None:
            self.descriptor = CapabilityDescriptor(
                name="document.read",
                domain="document",  # type: ignore[arg-type]
                operation_type="read",
                supports_parallel=True,
            )
            self.calls: list = []

        async def execute(self, request: AgentRequest, context: Any) -> AgentResult:
            self.calls.append(request.task_id)
            return AgentResult(
                contract_version=CONTRACT_VERSION,
                task_id=request.task_id,
                status="success",
                data=DocumentReadOutput(
                    kind="document.read", read_unit_count=1
                ),
                evidence_uses=(EvidenceUseRef(use_id=use_id),),
                coverage_observations=(
                    CoverageObservation(
                        target_id="t_b_r1",
                        observed_locators=(DocumentLocator(kind="document"),),
                        outcome="read",
                    ),
                ),
                error=None,
            )

    class _Hydrator:
        async def hydrate_for_evaluation(
            self, use_refs, *, runtime, plan, bindings
        ):
            task_id = plan.tasks[0].task_id if plan.tasks else "t0"
            return tuple(
                HydratedEvidence(
                    use_id=ref.use_id,
                    evidence_id=uuid4(),
                    task_id=task_id,
                    purpose="supporting",
                    target_id=None,
                    content="Nội dung tài liệu.",
                    role=None,
                    source_label="doc",
                    source_identity=DerivedSourceIdentity(
                        kind="derived", source_evidence_ids=(ref.use_id,)
                    ),
                    classification="official",
                    locator=None,
                )
                for ref in use_refs
            )

        async def hydrate_for_synthesis(
            self, use_refs, *, runtime, plan, bindings, budget
        ):
            return await self.hydrate_for_evaluation(
                use_refs, runtime=runtime, plan=plan, bindings=bindings
            )

    reply_id = uuid4()
    runtime.services.chat_messages = _ChatMessages({reply_id: "1"})
    runtime.services.authorization = _Authorization(
        grants={
            shared["workspace_id"]: {shared["doc_a"], shared["doc_b"]},
        }
    )
    reader = _DocRead()
    runtime.services.capability_registry = build_capability_registry(
        [CapabilityRegistration(capability=reader)],
        runtime.capability_runtime,
    )
    runtime.services.evidence_hydrator = _Hydrator()

    stored = graph.get_state(config)
    pending_request = (stored.values or {})["clarification"]
    command = await convert_streaming.prepare_v2_resume_command(
        message_id=reply_id,
        request=pending_request,
        runtime_context=runtime,
    )
    assert not getattr(command, "goto", None)

    resolver = PlanBindingResolver()
    # Mirror production ingress: the resume pre-feed and the scheduler feed
    # resolve through the SAME request-scoped instance (round 2 N1 raises on
    # targeted plans without a wired resolver).
    runtime.services.pinned_target_resolver = resolver
    second = [
        ev
        async for ev in convert_streaming.stream_v2_turn_events(
            graph=graph,
            runtime_context=runtime,
            thread_id="t8-real-resume",
            resume_command=command,
            plan_resolver=resolver,
        )
    ]
    terminals = [ev for ev in second if ev["event"] in ("complete", "error")]
    assert len(terminals) == 1
    # The resume must not re-suspend and must retire the request.
    assert graph.get_state(config).next == ()
    assert (graph.get_state(config).values or {}).get("clarification") is None
    # Terminal lease release discharged at runtime by the adapter.
    assert ("run-t8-real", "terminal") in shared["leases"].released
    # The resumed dispatch leased the scheduler-anchored (revision, use) row.
    assert (shared["revision"], use_id) in shared["leases"].acquired

    # C2: the terminal thread now checkpoints a real plan — a further resume
    # attempt reads those real values, feeds the resolver, and refreshes the
    # anchored pairs before failing closed with a single error terminal.
    fed: list = []
    orig_feed = resolver.feed

    def _tracked_feed(plan, bindings):
        fed.append((plan, bindings))
        return orig_feed(plan, bindings)

    resolver.feed = _tracked_feed  # type: ignore[method-assign]
    acquired_before = list(shared["leases"].acquired)
    third = [
        ev
        async for ev in convert_streaming.stream_v2_turn_events(
            graph=graph,
            runtime_context=runtime,
            thread_id="t8-real-resume",
            resume_command=command,
            plan_resolver=resolver,
        )
    ]
    third_terminals = [ev for ev in third if ev["event"] in ("complete", "error")]
    assert len(third_terminals) == 1
    assert third_terminals[0]["event"] == "error"
    assert len(fed) == 1
    assert isinstance(fed[0][0], TaskPlan)
    assert isinstance(fed[0][1], DocumentBindingSet)
    # The refresh re-acquires the SAME anchored row the scheduler leased
    # (expiry extended): the pair recurs after the snapshot.
    assert len(shared["leases"].acquired) > len(acquired_before)
    assert (shared["revision"], use_id) in shared["leases"].acquired


def test_task5_v2_ingress_callers_construct_equivalent_trusted_scope():
    """Task 5: standalone/session/admin/telegram build equivalent v2 scope.

    Every production ``build_v2_ingress`` call site passes the authenticated
    workspace scope as ``authenticated_workspace_ids``; only the
    authenticated-admin evaluation surface may narrow it via
    ``requested_workspace_ids`` (intersected inside, 403 when empty); and
    ``api_explicit`` known-documents are built only from server-filtered
    ``document_ids``.
    """
    import ast
    from pathlib import Path

    import app.services.agent.streaming as convert_streaming

    root = Path(convert_streaming.__file__).resolve().parents[3]
    files = {
        "standalone": root / "app" / "api" / "chat_agent_lg.py",
        "session": root / "app" / "api" / "chat_session.py",
        "admin": root / "app" / "api" / "agent_admin.py",
        "telegram": root
        / "app"
        / "services"
        / "integrations"
        / "telegram_service.py",
    }
    calls: dict[str, list[dict]] = {}
    for name, path in files.items():
        tree = ast.parse(path.read_text())
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                label = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", "")
                )
                if label == "build_v2_ingress":
                    found.append(
                        {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
                    )
        assert found, f"{name} ({path}) must build the v2 ingress"
        calls[name] = found

    for name, sites in calls.items():
        for kw in sites:
            assert "authenticated_workspace_ids" in kw, (
                f"{name}: ingress must pass the authenticated scope"
            )
            assert "requested_workspace_ids" in kw or name != "admin", (
                f"{name}: only admin may narrow scope; others default to full"
            )
    # Only the admin surface narrows by a caller-supplied workspace list.
    narrowers = [
        name
        for name, sites in calls.items()
        for kw in sites
        if kw.get("requested_workspace_ids") not in (None, "None")
    ]
    assert narrowers == ["admin"], (
        f"only admin may narrow scope, got {narrowers}"
    )
    # api_explicit resources are built only from server-filtered document_ids.
    for name in ("standalone", "session"):
        source = files[name].read_text()
        assert 'source="api_explicit"' in source
    assert 'source="api_explicit"' not in files["telegram"].read_text()


# ---------------------------------------------------------------------------
# P0 Task 6 fix round 1 (F1): observed serving telemetry drives the
# factual zero-dispatch sentinel through the real streaming/metric path.
# ---------------------------------------------------------------------------


def _task6_insufficient_response(content: str = "Not enough evidence.") -> FinalResponse:
    return FinalResponse(
        contract_version=CONTRACT_VERSION,
        status="insufficient",
        content=content,
        citations=(),
    )


def _task6_denied_response(content: str = "Access denied.") -> FinalResponse:
    return FinalResponse(
        contract_version=CONTRACT_VERSION,
        status="denied",
        content=content,
        citations=(),
    )


def _task6_factual_state(*, response, task_results_present: bool, task_results) -> dict:
    state: dict = {
        "route_decision": {"route": "complex_research"},
        "final_response": response,
    }
    if task_results_present:
        state["execution"] = {"plan": None, "task_results": task_results}
    return state


class _Task6FakeDB:
    """Minimal append-only stand-in for the metric emission DB session."""

    def __init__(self) -> None:
        self.added: list = []

    def add(self, row) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


def test_task6_streaming_zero_task_results_persists_factual_zero_dispatch():
    """Factual v2 terminal with zero task results -> factual_zero_dispatch.

    Drives the REAL streaming adapter (observed route + count in the
    caller-owned terminal_info) into the REAL metric emission: the stored
    terminal must be the sentinel.
    """
    import app.services.agent.streaming as convert_streaming
    from app.services.agent import rollout_metrics as metrics

    state = _task6_factual_state(
        response=_task6_insufficient_response(),
        task_results_present=True,
        task_results=(),
    )
    graph = FakeGraph([("return", state)])
    terminal_info: dict = {}
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-task6-zero",
                initial_state={"request": "q"},
                terminal_info=terminal_info,
            )
        )
    )
    assert [ev["event"] for ev in events].count("error") == 1
    assert terminal_info.get("route") == "complex_research"
    assert terminal_info.get("capability_call_count") == 0
    # The error-terminal status is observed via the seam (the SSE error
    # payload carries no status, so the ingress cannot read it from the
    # complete event).
    assert terminal_info.get("response_status") == "insufficient"

    db = _Task6FakeDB()

    async def _run():
        return await metrics.try_emit_terminal_rollout_metric(
            db,
            arm="v2",
            request_id="req-task6-zero",
            workspace_ids=["ws-1"],
            started_at=datetime.now(timezone.utc) - timedelta(seconds=2),
            terminal_status="success",
            citation_count=0,
            cancelled=False,
            answer_text="grounded answer",
            factual_expected=True,
            served_document_ids=[],
            allowed_document_ids=None,
            production_write_count=0,
            scope_bound=True,
            route=terminal_info.get("route"),
            response_status=terminal_info.get("response_status"),
            capability_call_count=terminal_info.get("capability_call_count"),
        )

    row = asyncio.run(_run())
    assert row is not None
    assert row.terminal_status == metrics.FACTUAL_ZERO_DISPATCH_TERMINAL


def test_task6_streaming_nonzero_task_results_does_not_remap():
    """Factual v2 terminal with one task result stays a normal terminal."""
    import app.services.agent.streaming as convert_streaming
    from app.services.agent import rollout_metrics as metrics

    state = _task6_factual_state(
        response=_task6_insufficient_response(),
        task_results_present=True,
        task_results=({"task_id": "t1"},),
    )
    graph = FakeGraph([("return", state)])
    terminal_info: dict = {}
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-task6-nonzero",
                initial_state={"request": "q"},
                terminal_info=terminal_info,
            )
        )
    )
    assert [ev["event"] for ev in events].count("error") == 1
    assert terminal_info.get("route") == "complex_research"
    assert terminal_info.get("capability_call_count") == 1
    assert terminal_info.get("response_status") == "insufficient"

    db = _Task6FakeDB()

    async def _run():
        return await metrics.try_emit_terminal_rollout_metric(
            db,
            arm="v2",
            request_id="req-task6-nonzero",
            workspace_ids=["ws-1"],
            started_at=datetime.now(timezone.utc) - timedelta(seconds=2),
            terminal_status="success",
            citation_count=0,
            cancelled=False,
            answer_text="grounded answer",
            factual_expected=True,
            served_document_ids=[],
            allowed_document_ids=None,
            production_write_count=0,
            scope_bound=True,
            route=terminal_info.get("route"),
            response_status=terminal_info.get("response_status"),
            capability_call_count=terminal_info.get("capability_call_count"),
        )

    row = asyncio.run(_run())
    assert row is not None
    assert row.terminal_status == "success"


def test_task6_streaming_missing_execution_stays_unobservable():
    """Missing execution.task_results is None (never inferred as zero)."""
    import app.services.agent.streaming as convert_streaming
    from app.services.agent import rollout_metrics as metrics

    state = _task6_factual_state(
        response=_task6_insufficient_response(),
        task_results_present=False,
        task_results=None,
    )
    graph = FakeGraph([("return", state)])
    terminal_info: dict = {}
    asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-task6-missing",
                initial_state={"request": "q"},
                terminal_info=terminal_info,
            )
        )
    )
    assert terminal_info.get("route") == "complex_research"
    assert "capability_call_count" not in terminal_info
    assert terminal_info.get("response_status") == "insufficient"

    db = _Task6FakeDB()

    async def _run():
        return await metrics.try_emit_terminal_rollout_metric(
            db,
            arm="v2",
            request_id="req-task6-missing",
            workspace_ids=["ws-1"],
            started_at=datetime.now(timezone.utc) - timedelta(seconds=2),
            terminal_status="success",
            citation_count=0,
            cancelled=False,
            answer_text="grounded answer",
            factual_expected=True,
            served_document_ids=[],
            allowed_document_ids=None,
            production_write_count=0,
            scope_bound=True,
            route=terminal_info.get("route"),
            response_status=terminal_info.get("response_status"),
            capability_call_count=terminal_info.get("capability_call_count"),
        )

    row = asyncio.run(_run())
    assert row is not None
    assert row.terminal_status == "success"


def test_task6_streaming_denied_and_security_sentinel_not_remapped():
    """Typed denied stays typed; security-unobservable keeps precedence."""
    import app.services.agent.streaming as convert_streaming
    from app.services.agent import rollout_metrics as metrics

    denied_state = _task6_factual_state(
        response=_task6_denied_response(),
        task_results_present=True,
        task_results=(),
    )
    denied_graph = FakeGraph([("return", denied_state)])
    denied_info: dict = {}
    asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=denied_graph,
                runtime_context=_runtime(),
                thread_id="thread-task6-denied",
                initial_state={"request": "q"},
                terminal_info=denied_info,
            )
        )
    )
    assert denied_info.get("capability_call_count") == 0
    assert denied_info.get("response_status") == "denied"
    denied_db = _Task6FakeDB()

    async def _run_denied():
        return await metrics.try_emit_terminal_rollout_metric(
            denied_db,
            arm="v2",
            request_id="req-task6-denied",
            workspace_ids=["ws-1"],
            started_at=datetime.now(timezone.utc) - timedelta(seconds=2),
            terminal_status="success",
            citation_count=0,
            cancelled=False,
            answer_text="Access denied.",
            factual_expected=False,
            served_document_ids=[],
            allowed_document_ids=["doc-a"],
            production_write_count=0,
            scope_bound=True,
            route=denied_info.get("route"),
            response_status=denied_info.get("response_status"),
            capability_call_count=denied_info.get("capability_call_count"),
        )

    denied_row = asyncio.run(_run_denied())
    assert denied_row is not None
    assert denied_row.terminal_status == "success"

    sentinel_db = _Task6FakeDB()

    async def _run_sentinel():
        return await metrics.try_emit_terminal_rollout_metric(
            sentinel_db,
            arm="v2",
            request_id="req-task6-sentinel",
            workspace_ids=["ws-1"],
            started_at=datetime.now(timezone.utc) - timedelta(seconds=2),
            terminal_status="success",
            citation_count=0,
            cancelled=False,
            answer_text=None,
            factual_expected=True,
            served_document_ids=[],
            allowed_document_ids=["doc-a"],
            production_write_count=0,
            route="complex_research",
            response_status="insufficient",
            capability_call_count=0,
        )

    sentinel_row = asyncio.run(_run_sentinel())
    assert sentinel_row is not None
    assert sentinel_row.terminal_status == metrics.SECURITY_UNOBSERVABLE_TERMINAL


# ---------------------------------------------------------------------------
# Round 2 — error-terminal persistence is a v2-only contract (spec §12.4)
#
# The v2 typed error terminal carries a SAFE message that must persist as
# nonblank assistant content (a reload never recreates a blank row). The v1
# stream emits ``str(exc)`` — internal exception detail (DB errors, file
# paths, provider messages) — which must NEVER persist as assistant content
# or feed the memory pipeline. The fill is gated on the serving arm, never
# on the wire event alone.
# ---------------------------------------------------------------------------


class _ErrPersistResult:
    """``db.execute`` result: ``_first_obj`` once, then always empty."""

    def __init__(self, first_obj: Any) -> None:
        self._first_obj = first_obj
        self._calls = 0

    def scalar_one_or_none(self) -> Any:
        self._calls += 1
        if self._calls == 1:
            return self._first_obj
        return None

    def scalars(self) -> Any:
        return SimpleNamespace(all=lambda: [])

    def first(self) -> Any:
        return None


class _ErrPersistDB:
    """Minimal ``AsyncSession`` stand-in capturing ``db.add`` rows."""

    def __init__(self, first_obj: Any = None, added: list | None = None) -> None:
        self._first_obj = first_obj
        self.added = added if added is not None else []

    async def execute(self, *_a: Any, **_k: Any) -> _ErrPersistResult:
        return _ErrPersistResult(self._first_obj)

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def __aenter__(self) -> "_ErrPersistDB":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


def _err_sse(message: str) -> str:
    """One raw v1-wire ``error`` SSE frame."""
    import json as _json

    return f"event: error\ndata: {_json.dumps({'message': message})}\n\n"


def _patch_standalone_deps(
    monkeypatch: pytest.MonkeyPatch, *, arm: str, error_message: str
) -> None:
    """Patch every dependency of ``langgraph_chat_stream`` (all fakes)."""
    import app.api.chat_agent_lg as lg
    import app.services.agent.runtime_selector as selector
    import app.services.agent.streaming as streaming
    import app.services.agent.rollout_metrics as metrics
    import app.queue.publisher as publisher
    from app.services.abbreviation_service import AbbreviationService

    async def _v1_stream(_graph: Any, _state: Any):
        yield _err_sse(error_message)

    async def _v2_stream(**_kwargs: Any):
        yield _err_sse(error_message)

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(lg, "_resolve_system_prompt", _noop)
    monkeypatch.setattr(selector, "resolve_runtime_scope", lambda **kw: kw["authenticated_ids"])
    monkeypatch.setattr(selector, "resolve_serving_arm", lambda **kw: asyncio.sleep(0, arm))
    monkeypatch.setattr(selector, "resolve_agent_graph", lambda _v: asyncio.sleep(0, object()))
    monkeypatch.setattr(selector, "persist_raw_user_message", _noop)
    monkeypatch.setattr(streaming, "stream_agent_to_sse", _v1_stream)
    monkeypatch.setattr(streaming, "build_initial_state", lambda **_kw: {})
    monkeypatch.setattr(lg, "_stream_v2_standalone", _v2_stream)
    monkeypatch.setattr(AbbreviationService, "expand_ab_in_text", _noop)
    monkeypatch.setattr(metrics, "try_emit_terminal_rollout_metric", _noop)
    monkeypatch.setattr(metrics, "was_cancel_requested", lambda _rid: asyncio.sleep(0, False))
    monkeypatch.setattr(publisher, "publish_memory_save_task", _noop)


def _patch_session_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm: str,
    error_message: str,
    added: list,
) -> None:
    """Patch every dependency of ``chat_stream_session`` (all fakes)."""
    from contextlib import asynccontextmanager

    import app.api.chat_agent as chat_agent
    import app.api.chat_session as cs
    import app.core.database as core_db
    import app.services.agent.runtime_selector as selector
    import app.services.agent.streaming as streaming
    import app.services.agent.rollout_metrics as metrics
    import app.services.memory.conversation_summary_service as summary_mod

    async def _v1_stream(_graph: Any, _state: Any):
        yield _err_sse(error_message)

    async def _agen():
        yield _err_sse(error_message)

    @asynccontextmanager
    async def _fake_v2_run(**_kwargs: Any):
        yield None, _agen()

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    class _SummarySvc:
        async def get_context_for_session(self, *_a: Any, **_k: Any) -> str:
            return ""

        async def save_exchange_summary(self, **_kw: Any) -> None:
            return None

    monkeypatch.setattr(cs, "_get_accessible_workspaces", lambda *_a, **_k: asyncio.sleep(0, [uuid4()]))
    monkeypatch.setattr(cs, "_filter_accessible_document_ids", lambda *_a, **_k: asyncio.sleep(0, []))
    monkeypatch.setattr(cs, "_maybe_launch_shadow_turn", lambda **_kw: None)
    monkeypatch.setattr(cs, "_session_v2_run", _fake_v2_run)
    monkeypatch.setattr(selector, "resolve_runtime_scope", lambda **kw: kw["authenticated_ids"])
    monkeypatch.setattr(selector, "resolve_serving_arm", lambda **kw: asyncio.sleep(0, arm))
    monkeypatch.setattr(selector, "resolve_agent_graph", lambda _v: asyncio.sleep(0, object()))
    monkeypatch.setattr(selector, "persist_raw_user_message", _noop)
    monkeypatch.setattr(streaming, "stream_agent_to_sse", _v1_stream)
    monkeypatch.setattr(streaming, "build_initial_state", lambda **_kw: {})
    monkeypatch.setattr(core_db, "async_session_maker", lambda: _ErrPersistDB(added=added))
    monkeypatch.setattr(summary_mod, "get_conversation_summary_service", lambda: _SummarySvc())
    monkeypatch.setattr(chat_agent, "sse_with_heartbeat", lambda agen: agen)
    monkeypatch.setattr(metrics, "try_emit_terminal_rollout_metric", _noop)
    monkeypatch.setattr(metrics, "was_cancel_requested", lambda _rid: asyncio.sleep(0, False))


def _assistant_rows(added: list) -> list:
    return [
        row
        for row in added
        if getattr(row, "role", None) == "assistant"
    ]


@pytest.mark.asyncio
async def test_standalone_v1_error_event_keeps_assistant_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V1 ``str(exc)`` error text must NOT persist as assistant content."""
    import app.api.chat_agent_lg as lg
    from app.schemas.rag import ChatRequest

    _patch_standalone_deps(
        monkeypatch, arm="v1", error_message="psycopg2: relation 'documents' missing at /srv/app/db.py:42"
    )
    db = _ErrPersistDB(first_obj=SimpleNamespace(system_prompt=None))
    frames = [
        frame
        async for frame in lg.langgraph_chat_stream(
            [uuid4()],
            ChatRequest(message="hello"),
            db,
            uuid4(),
            "u@example.com",
            False,
            None,
        )
    ]
    assert any(frame.startswith("event: error") for frame in frames)
    rows = _assistant_rows(db.added)
    assert len(rows) == 1
    assert rows[0].content == ""


@pytest.mark.asyncio
async def test_standalone_v2_typed_error_persists_safe_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V2 typed error terminal persists its safe message nonblank (§12.4)."""
    import app.api.chat_agent_lg as lg
    from app.schemas.rag import ChatRequest

    _patch_standalone_deps(
        monkeypatch, arm="v2", error_message="The answer could not be completed safely."
    )
    db = _ErrPersistDB(first_obj=SimpleNamespace(system_prompt=None))
    frames = [
        frame
        async for frame in lg.langgraph_chat_stream(
            [uuid4()],
            ChatRequest(message="hello"),
            db,
            uuid4(),
            "u@example.com",
            False,
            None,
        )
    ]
    assert any(frame.startswith("event: error") for frame in frames)
    rows = _assistant_rows(db.added)
    assert len(rows) == 1
    assert rows[0].content == "The answer could not be completed safely."


@pytest.mark.asyncio
async def test_session_v1_error_event_keeps_assistant_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session path: v1 ``str(exc)`` never persists as assistant content."""
    import app.api.chat_session as cs
    from app.schemas.rag import ChatRequest

    added: list = []
    _patch_session_deps(
        monkeypatch,
        arm="v1",
        error_message="psycopg2: relation 'documents' missing at /srv/app/db.py:42",
        added=added,
    )
    session_db = _ErrPersistDB(first_obj=SimpleNamespace(id="sess-1"))
    user = SimpleNamespace(id=uuid4(), is_superadmin=False)
    response = await cs.chat_stream_session(
        "sess-1", ChatRequest(message="hello"), session_db, user
    )
    frames = [frame async for frame in response.body_iterator]
    assert any(str(frame).startswith("event: error") for frame in frames)
    rows = _assistant_rows(added)
    assert len(rows) == 1
    assert rows[0].content == ""


@pytest.mark.asyncio
async def test_session_v2_typed_error_persists_safe_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session path: v2 typed error persists its safe message nonblank."""
    import app.api.chat_session as cs
    from app.schemas.rag import ChatRequest

    added: list = []
    _patch_session_deps(
        monkeypatch,
        arm="v2",
        error_message="The answer could not be completed safely.",
        added=added,
    )
    session_db = _ErrPersistDB(first_obj=SimpleNamespace(id="sess-2"))
    user = SimpleNamespace(id=uuid4(), is_superadmin=False)
    response = await cs.chat_stream_session(
        "sess-2", ChatRequest(message="hello"), session_db, user
    )
    frames = [frame async for frame in response.body_iterator]
    assert any(str(frame).startswith("event: error") for frame in frames)
    rows = _assistant_rows(added)
    assert len(rows) == 1
    assert rows[0].content == "The answer could not be completed safely."


# ---------------------------------------------------------------------------
# 6. In-flight progress: astream node starts -> v1 status steps
# ---------------------------------------------------------------------------


def test_v2_progress_event_mapping():
    import app.services.agent.streaming as convert_streaming

    assert convert_streaming._v2_progress_event("route", "analyzing") is None
    assert (
        convert_streaming._v2_progress_event("evaluate", "retrieved") is None
    )
    ev = convert_streaming._v2_progress_event("execute", "analyzing")
    assert ev == {
        "event": "status",
        "data": {"step": "searching", "detail": "Đang tra cứu tài liệu..."},
    }


def test_v2_astream_projects_node_starts_as_status_progress():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    script = [
        ((), "tasks", {"name": "route", "input": {}}),
        ((), "tasks", {"name": "execute", "input": {}}),
        (("complex_boundary:x",), "tasks", {"name": "execute", "input": {}}),
        ((), "tasks", {"name": "execute", "result": {}}),
        ((), "tasks", {"name": "evaluate", "input": {}}),
        (("complex_boundary:x",), "tasks", {"name": "reduce", "input": {}}),
        ((), "tasks", {"name": "synthesize", "input": {}}),
        (("synthesis",), "tasks", {"name": "generate", "input": {}}),
        ((), "updates", {"synthesize": {}}),
        ((), "values", {"final_response": _success_response("ok")}),
    ]
    graph = FakeStreamingGraph([], stream_script=script)
    runtime = _runtime()
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=runtime,
                thread_id="thread-astream-progress",
                initial_state={"request": "q"},
            )
        )
    )
    steps = [
        ev["data"]["step"] for ev in events if ev["event"] == "status"
    ]
    assert steps == ["analyzing", "searching", "retrieved", "generating"]
    kinds = [ev["event"] for ev in events]
    assert "citation" in kinds
    tokens = [ev for ev in events if ev["event"] == "token"]
    assert "".join(t["data"]["text"] for t in tokens) == "ok"
    terminals = _terminal_events(events)
    assert len(terminals) == 1
    assert terminals[0]["event"] == "complete"
    assert terminals[0]["data"]["answer"] == "ok"
    payload, cfg, ctx, stream_mode, subgraphs = graph.stream_calls[0]
    assert stream_mode == ["updates", "tasks", "values", "custom"]
    assert subgraphs is True
    assert ctx is runtime
    assert graph.calls == []


def test_v2_astream_interrupt_values_chunk_is_suspension():
    import asyncio

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
    script = [
        ((), "tasks", {"name": "clarify", "input": {}}),
        (
            (),
            "values",
            {"__interrupt__": (True,), "clarification": request},
        ),
    ]
    graph = FakeStreamingGraph(
        [],
        stream_script=script,
        checkpoint={"clarification": request},
        next_nodes=("clarify_wait",),
    )
    leases = FakeLeaseRepo()
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(leases=leases),
                thread_id="thread-astream-suspend",
                initial_state={"request": "q"},
            )
        )
    )
    assert any(ev["event"] == "clarification_required" for ev in events)
    terminals = _terminal_events(events)
    assert len(terminals) == 1
    assert terminals[0]["event"] == "complete"
    assert terminals[0]["data"]["status"] == "clarify"
    assert terminals[0]["data"]["answer"] == "Which document?"
    assert leases.released == []


def test_v2_astream_failure_is_terminal_error():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    script = [
        ((), "tasks", {"name": "execute", "input": {}}),
        ("raise", RuntimeError("boom")),
    ]
    graph = FakeStreamingGraph([], stream_script=script)
    leases = FakeLeaseRepo()
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(leases=leases),
                thread_id="thread-astream-fail",
                initial_state={"request": "q"},
            )
        )
    )
    assert events[-2]["event"] == "token_rollback"
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["message"] == (
        convert_streaming._V2_UNEXPECTED_ERROR_MESSAGE
    )
    assert len(_terminal_events(events)) == 1
    assert leases.released == [("run-t8-test", "terminal")]


def test_v2_astream_cancel_mid_run_releases_cancelled():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    leases = FakeLeaseRepo()

    async def _driver() -> list[dict]:
        gate = asyncio.Event()
        graph = FakeStreamingGraph([], stream_script=[], gate=gate)
        agen = convert_streaming.stream_v2_turn_events(
            graph=graph,
            runtime_context=_runtime(leases=leases),
            thread_id="thread-astream-cancel",
            initial_state={"request": "q"},
        )
        seen: list[dict] = []
        # Consume the first status event, resume the generator so the
        # stream task starts (it then blocks on the gate inside astream),
        # wait for that, then disconnect (aclose -> GeneratorExit).
        ev = await agen.__anext__()
        seen.append(ev)
        pending = asyncio.ensure_future(agen.__anext__())
        await asyncio.wait_for(graph.stream_started.wait(), timeout=5)
        # Cancel the consumer task: CancelledError propagates through the
        # adapter, the graph task is cancelled, leases release cancelled.
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        return seen

    seen = asyncio.run(_driver())
    assert _terminal_events(seen) == []
    assert leases.released == [("run-t8-test", "cancelled")]


def test_v2_graph_without_astream_falls_back_to_ainvoke():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    graph = FakeGraph([("return", {"final_response": _success_response("ok")})])
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-ainvoke-fallback",
                initial_state={"request": "q"},
            )
        )
    )
    assert len(graph.calls) == 1
    steps = [
        ev["data"]["step"] for ev in events if ev["event"] == "status"
    ]
    assert steps == ["analyzing", "generating"]
    assert "token_rollback" not in [ev["event"] for ev in events]
    assert events[-1]["event"] == "complete"


# ---------------------------------------------------------------------------
# 7. Speculative claim streaming: custom chunks -> token/token_rollback
# ---------------------------------------------------------------------------


def _spec_claim(index: int, presentation: str, text: str) -> tuple:
    return (
        ("synthesis",),
        "custom",
        {
            "kind": "synthesis.speculative_claim",
            "index": index,
            "presentation": presentation,
            "text": text,
        },
    )


def _spec_reset() -> tuple:
    return (("synthesis",), "custom", {"kind": "synthesis.speculative_reset"})


def test_v2_speculative_claims_render_then_rollback_before_final():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    script = [
        _spec_claim(0, "summary", "Tóm tắt."),
        _spec_claim(1, "detail", "Chi tiết 1."),
        _spec_claim(2, "detail", "Chi tiết 2."),
        _spec_claim(3, "caveat", "Còn thiếu."),
        ((), "values", {"final_response": _success_response("FINAL")}),
    ]
    graph = FakeStreamingGraph([], stream_script=script)
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-spec-1",
                initial_state={"request": "q"},
            )
        )
    )
    kinds = [ev["event"] for ev in events]
    assert kinds.count("token_rollback") == 1
    rb = kinds.index("token_rollback")
    spec = "".join(ev["data"]["text"] for ev in events[:rb] if ev["event"] == "token")
    assert spec == (
        "Tóm tắt.\n\n- Chi tiết 1.\n- Chi tiết 2.\n\n- **Lưu ý:** Còn thiếu."
    )
    # Rollback sits after the last speculative token and before citation.
    assert kinds.index("citation") > rb
    final_tokens = "".join(
        ev["data"]["text"] for ev in events[rb + 1 :] if ev["event"] == "token"
    )
    assert final_tokens == "FINAL"
    assert events[-1]["data"]["answer"] == "FINAL"


def test_v2_speculative_reset_rolls_back_mid_run():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    script = [
        _spec_claim(0, "summary", "Nháp một."),
        _spec_reset(),
        _spec_claim(0, "summary", "Nháp hai."),
        _spec_claim(1, "detail", "Chi tiết hai."),
        ((), "values", {"final_response": _success_response("FINAL")}),
    ]
    graph = FakeStreamingGraph([], stream_script=script)
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-spec-2",
                initial_state={"request": "q"},
            )
        )
    )
    kinds = [ev["event"] for ev in events]
    assert kinds.count("token_rollback") == 2
    first_rb = kinds.index("token_rollback")
    second_rb = kinds.index("token_rollback", first_rb + 1)
    between = "".join(
        ev["data"]["text"]
        for ev in events[first_rb + 1 : second_rb]
        if ev["event"] == "token"
    )
    assert between == "Nháp hai.\n\n- Chi tiết hai."


def test_v2_speculative_claims_retracted_before_error_terminal():
    import asyncio

    import app.services.agent.streaming as convert_streaming
    from app.services.agents.v2.contracts.response import FinalResponse

    denied = FinalResponse(
        contract_version=CONTRACT_VERSION,
        status="error",
        content="Thất bại.",
        citations=(),
    )
    script = [
        _spec_claim(0, "summary", "Nháp."),
        ((), "values", {"final_response": denied}),
    ]
    graph = FakeStreamingGraph([], stream_script=script)
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-spec-3",
                initial_state={"request": "q"},
            )
        )
    )
    kinds = [ev["event"] for ev in events]
    assert "citation" not in kinds
    assert kinds[-2:] == ["token_rollback", "error"]


def test_v2_speculative_claims_retracted_before_suspend():
    import asyncio

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
    script = [
        _spec_claim(0, "summary", "Nháp."),
        ((), "values", {"__interrupt__": (True,), "clarification": request}),
    ]
    graph = FakeStreamingGraph(
        [],
        stream_script=script,
        checkpoint={"clarification": request},
        next_nodes=("clarify_wait",),
    )
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-spec-4",
                initial_state={"request": "q"},
            )
        )
    )
    kinds = [ev["event"] for ev in events]
    assert "clarification_required" in kinds
    assert kinds.index("token_rollback") < kinds.index("clarification_required")


def test_v2_malformed_custom_chunks_emit_nothing():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    script = [
        (("synthesis",), "custom", "not a dict"),
        (("synthesis",), "custom", {"kind": "other"}),
        (("synthesis",), "custom", {
            "kind": "synthesis.speculative_claim",
            "index": 0, "presentation": "summary", "text": "",
        }),
        (("synthesis",), "custom", {
            "kind": "synthesis.speculative_claim",
            "index": 1, "presentation": "x", "text": "abc",
        }),
        ((), "values", {"final_response": _success_response("ok")}),
    ]
    graph = FakeStreamingGraph([], stream_script=script)
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-spec-5",
                initial_state={"request": "q"},
            )
        )
    )
    kinds = [ev["event"] for ev in events]
    assert "token_rollback" not in kinds
    # Only the final grounded tokens, no speculative text.
    assert "".join(ev["data"]["text"] for ev in events if ev["event"] == "token") == "ok"


def test_v2_reset_without_speculative_text_emits_no_rollback():
    import asyncio

    import app.services.agent.streaming as convert_streaming

    script = [
        _spec_reset(),
        ((), "values", {"final_response": _success_response("ok")}),
    ]
    graph = FakeStreamingGraph([], stream_script=script)
    events = asyncio.run(
        _collect(
            convert_streaming.stream_v2_turn_events(
                graph=graph,
                runtime_context=_runtime(),
                thread_id="thread-spec-6",
                initial_state={"request": "q"},
            )
        )
    )
    assert "token_rollback" not in [ev["event"] for ev in events]


def test_speculative_chunk_matches_grounded_render_minus_markers():
    """Speculative rendering == render_grounded_claims output with the
    ``[idx]`` markers stripped (canonical summary/detail/caveat order)."""
    import re
    from uuid import uuid4

    import app.services.agent.streaming as convert_streaming
    from app.services.agents.v2.contracts.evidence import EvidenceUseRef
    from app.services.agents.v2.contracts.synthesis import GroundedClaim
    from app.services.agents.v2.synthesis import render
    from app.services.agents.v2.synthesis.citations import CitationProjection

    assert convert_streaming._V2_SPECULATIVE_CAVEAT_PREFIX == render._CAVEAT_PREFIX

    claims = (
        GroundedClaim(
            claim_id="claim-1", text="Tóm tắt.",
            uses=(EvidenceUseRef(use_id=uuid4()),), presentation="summary",
        ),
        GroundedClaim(
            claim_id="claim-2", text="Chi tiết 1.",
            uses=(EvidenceUseRef(use_id=uuid4()),), presentation="detail",
        ),
        GroundedClaim(
            claim_id="claim-3", text="Chi tiết 2.",
            uses=(EvidenceUseRef(use_id=uuid4()),), presentation="detail",
        ),
        GroundedClaim(
            claim_id="claim-4", text="Còn thiếu.",
            uses=(EvidenceUseRef(use_id=uuid4()),), presentation="caveat",
        ),
    )
    projection = CitationProjection(
        citations=(),
        claim_indexes={c.claim_id: (f"a{i}b{i}"[:4],) for i, c in enumerate(claims)},
    )
    grounded = render.render_grounded_claims(claims, projection)
    grounded = re.sub(r"\[[a-z0-9]{4}\]", "", grounded)

    prev = None
    speculative = ""
    for c in claims:
        chunk = convert_streaming._speculative_claim_chunk(
            prev, c.presentation, c.text
        )
        speculative += chunk
        prev = c.presentation
    assert speculative == grounded

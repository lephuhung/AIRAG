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
    # must not fabricate them.
    assert {
        ev["event"] for ev in events
    } <= {"status", "token", "potential_abbreviations", "complete"}


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
    from app.services.agents.v2.contracts.evidence import EvidenceUseRef
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

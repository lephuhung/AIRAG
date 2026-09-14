"""Task 9 fix round 1 (I5): the v1 arm must emit its clarification event.

``ask_user_clarification`` pushes a ``("clarification", {...})`` tuple into
the SSE queue, but ``stream_agent_events`` had no branch for it — the frame
was dropped silently and no wire (raw or normalized) ever carried the v1
clarification request. These tests drive the REAL emitter with the stub
graph pattern (no LLM, no DB) and pin that the event is forwarded.
"""
from __future__ import annotations

import pytest

from app.services.agent.streaming import stream_agent_events


class _StubGraph:
    def __init__(self, events_to_push):
        self._events = events_to_push

    async def ainvoke(self, initial_state, config=None, debug=False):
        from app.services.agent.streaming import push_event

        for ev_type, ev_data in self._events:
            await push_event({}, ev_type, ev_data)
        await push_event({}, "done", None)


@pytest.mark.asyncio
async def test_v1_emitter_forwards_clarification_event():
    graph = _StubGraph([
        (
            "clarification",
            {
                "message": "Pick one",
                "options": ["Alpha", "Beta"],
                "context": {},
            },
        ),
    ])
    out = []
    async for ev in stream_agent_events(graph, {"messages": []}):
        out.append(ev)
    frames = [e for e in out if e["event"] == "clarification"]
    assert len(frames) == 1
    assert frames[0]["data"]["message"] == "Pick one"
    assert frames[0]["data"]["options"] == ["Alpha", "Beta"]

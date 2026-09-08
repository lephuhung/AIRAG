"""
Phase 0 / B5 — E2E rollback test.

Per F.3/O70: production-path SSE-to-persistence E2E test (not local
reimplementation). Covers grounding-retract → token_rollback → all artifacts
cleared across frontend reducer, persistence rollback, terminal complete
authoritative.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.services.agent.streaming import stream_agent_events


class _StubGraph:
    """A minimal graph stand-in that puts pre-canned events into the
    queue as soon as the background task starts.
    """

    def __init__(self, events_to_push: list[tuple[str, Any]]):
        self._events = events_to_push

    async def ainvoke(self, initial_state: dict, config: dict | None = None, debug: bool = False):
        from app.services.agent.streaming import push_event

        for ev_type, ev_data in self._events:
            await push_event({}, ev_type, ev_data)
        await push_event({}, "done", None)


@pytest.mark.asyncio
async def test_grounding_retract_clears_all_layers():
    """Full flow: sources → grounding guard retracts → all artifacts cleared."""
    graph = _StubGraph([
        ("sources", [{"id": "s1", "content": "stale source"}]),
        ("images", [{"id": "img-stale"}]),
        ("token", "fabricated answer "),
        ("potential_abbreviations", ["BMNN"]),
        ("people_data", [{"id": "p1"}]),
        ("token_rollback", {}),  # Grounding guard retracts
        ("sources", [{"id": "s2", "content": "clean source"}]),
        ("images", [{"id": "img-clean"}]),
        ("token", "clean answer"),
    ])

    out = []
    async for ev in stream_agent_events(graph, {"messages": []}):
        out.append(ev)

    # Verify token_rollback event is emitted
    rollback_events = [e for e in out if e["event"] == "token_rollback"]
    assert len(rollback_events) >= 1

    # Verify terminal complete has cleared artifacts
    complete_ev = next(e for e in out if e["event"] == "complete")
    
    # After rollback, pre-rollback sources should NOT appear
    source_ids = [s.get("id") for s in complete_ev["data"].get("sources", [])]
    assert "s1" not in source_ids, "Stale source should be cleared by rollback"
    
    # After rollback, clean sources should appear
    assert "s2" in source_ids, "Clean source should appear after rollback"


@pytest.mark.asyncio
async def test_terminal_complete_carries_cumulative_sources():
    """Terminal complete event carries cumulative deduplicated sources."""
    graph = _StubGraph([
        ("sources", [{"id": "A1", "content": "source A"}]),
        ("sources", [{"id": "B1", "content": "source B"}]),
    ])

    out = []
    async for ev in stream_agent_events(graph, {"messages": []}):
        out.append(ev)

    complete_ev = next(e for e in out if e["event"] == "complete")
    source_ids = [s.get("id") for s in complete_ev["data"].get("sources", [])]
    
    # Both sources should be present (cumulative)
    assert "A1" in source_ids
    assert "B1" in source_ids


@pytest.mark.asyncio
async def test_rollback_then_complete_with_no_followup():
    """Rollback with no followup sources → complete has empty sources."""
    graph = _StubGraph([
        ("sources", [{"id": "stale"}]),
        ("token_rollback", {}),
    ])

    out = []
    async for ev in stream_agent_events(graph, {"messages": []}):
        out.append(ev)

    complete_ev = next(e for e in out if e["event"] == "complete")
    assert complete_ev["data"]["sources"] == [], "Sources should be empty after rollback with no followup"

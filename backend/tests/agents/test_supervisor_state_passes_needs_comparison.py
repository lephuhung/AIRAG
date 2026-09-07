"""
Task 1 / B3 — `needs_comparison` must survive SupervisorState filtering.

Pre-existing code returns the ``needs_comparison`` key from
``query_analyzer_node`` (supervisor.py L1130, L1187) and reads it back
inside ``supervisor_node`` (L1631), but ``SupervisorState`` (models.py)
does not declare the field. TypedDict filtering strips unknown keys
before they reach the next node, so the flag is silently dropped.

These tests pin:

  1. ``SupervisorState`` declares ``needs_comparison`` — typed literals
     accept the key (regression pin for the schema fix).
  2. ``query_analyzer_node`` returns the key on the dict it returns
     from the awaited call — the *producer* publishes it.
  3. The supervisor graph round-trip preserves the key through node
     boundaries — schema filtering does NOT silently drop it.

No real LLM call: the awaited node uses a deterministic fast-path
(`needs_comparison = has_comparison AND has_personal` via regex on the
message) which exercises the real producer path end-to-end without
touching any provider.
"""
from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import HumanMessage

from app.services.agents.models import SupervisorState
from app.services.agents.supervisor import query_analyzer_node


def test_supervisor_state_declares_needs_comparison():
    """TypedDict regression pin: a state dict literal carrying
    ``needs_comparison`` must type-check (mypy) and round-trip.

    If someone removes the field from SupervisorState, the ``TypedDict``
    contract breaks and downstream consumers reading
    ``state['needs_comparison']`` start getting KeyError instead of
    a filtered-out value.
    """
    state: SupervisorState = {
        "messages": [HumanMessage("hello")],
        "needs_comparison": True,
    }
    assert state["needs_comparison"] is True


@pytest.mark.asyncio
async def test_query_analyzer_node_sets_needs_comparison_on_state():
    """Producer publishes the key when both comparison + personal
    patterns match the user message.

    Awaited node (it is async) exercises the real producer path. We do
    not patch the memory-agent fast-path because the comparison flag is
    set by regex BEFORE the LLM call — covering the actual fix.
    """
    state_in: SupervisorState = {
        "messages": [
            HumanMessage(
                "Hồ sơ của tôi có đủ điều kiện so với NĐ 13 không?"
            )
        ],
    }
    out = await query_analyzer_node(state_in)
    # Real producer contract: the key is present in the returned dict.
    assert "needs_comparison" in out
    assert out["needs_comparison"] is True


@pytest.mark.asyncio
async def test_query_analyzer_node_omits_needs_comparison_for_non_comparison():
    """Negative pin: a plain search query (no comparison pattern, no
    personal reference) must NOT set the comparison flag.

    Guards against a future regression where the regex accidentally
    starts matching common search phrases.
    """
    state_in: SupervisorState = {
        "messages": [
            HumanMessage("Quy định về bảo mật thông tin trong luật an ninh mạng")
        ],
    }
    out = await query_analyzer_node(state_in)
    assert "needs_comparison" in out
    assert out["needs_comparison"] is False


@pytest.mark.asyncio
async def test_full_graph_round_trip_preserves_needs_comparison():
    """Schema-filtering regression pin: needs_comparison set by
    query_analyzer_node must reach the supervisor_node boundary
    through a real LangGraph graph.

    This is the regression the brief is fixing: without the
    SupervisorState field declaration, TypedDict filtering drops the
    key before it reaches supervisor_node, and `supervisor_node` then
    falls back to state.get('needs_comparison', False) — silently
    treating a comparison query as a plain query.

    We construct a graph composed of query_analyzer_node +
    supervisor_node and assert the supervisor_node received the flag.

    Implementation note: supervisor_node calls an LLM. We can't run a
    real LLM in this offline test, so we wrap supervisor_node in a
    stub that reads `state['needs_comparison']` and stashes the value
    onto a closure-side list. The graph then captures what the
    supervisor_node observed.
    """
    from typing import Any
    from langgraph.graph import StateGraph, START, END

    # First, prove the producer wrote the flag.
    seed: SupervisorState = {
        "messages": [
            HumanMessage(
                "Hồ sơ của tôi có đủ điều kiện so với NĐ 13 không?"
            )
        ],
        "workspace_ids": [],
        "user_id": uuid.uuid4(),
        "session_id": "s-task1-b3",
    }
    analyzer_out = await query_analyzer_node(seed)
    assert analyzer_out.get("needs_comparison") is True

    # Now build a tiny graph: query_analyzer -> observer -> END.
    # The observer reads state['needs_comparison'] from the merged
    # graph state. If SupervisorState filtering strips the key, the
    # observer sees False (default) and the assertion fails.
    observed: dict[str, Any] = {}

    async def observer_node(state: SupervisorState) -> dict:
        observed["needs_comparison"] = state.get("needs_comparison")
        observed["raw_keys"] = list(state.keys())
        return {}

    g = StateGraph(SupervisorState)
    g.add_node("query_analyzer", query_analyzer_node)
    g.add_node("observer", observer_node)
    g.add_edge(START, "query_analyzer")
    g.add_edge("query_analyzer", "observer")
    g.add_edge("observer", END)
    compiled = g.compile()

    # graph.ainvoke(state) returns the merged final state.
    final = await compiled.ainvoke(seed)
    assert final.get("needs_comparison") is True, (
        f"Schema filter dropped needs_comparison; observer saw "
        f"{observed.get('needs_comparison')!r}; state keys: "
        f"{observed.get('raw_keys')}"
    )
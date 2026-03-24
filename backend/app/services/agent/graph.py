"""
LangGraph Agent Graph Builder
==============================

Builds and compiles the NexusRAG chat StateGraph.

Graph topology:
    START
      → memory_recall
      → intent_classifier
      → [direct_answer | tool_executor]
      → [answer_generator]
      → END

Usage::

    graph = build_agent_graph()
    initial_state = {**DEFAULT_STATE, "messages": [...], "workspace_ids": [...]}
    async for event in graph.astream_events(initial_state, version="v2"):
        ...
"""
from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END

from app.services.agent.state import AgentState, VALID_INTENTS
from app.services.agent.nodes import (
    memory_recall,
    intent_classifier,
    tool_executor,
    answer_generator,
    direct_answer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing function — decides next node after intent_classifier
# ---------------------------------------------------------------------------

def _route_by_intent(state: AgentState) -> str:
    """
    Conditional edge: route to direct_answer for greetings,
    or tool_executor for everything else.
    """
    intent = state.get("intent", "search")

    if intent == "greeting":
        logger.debug("[router] intent=greeting → direct_answer")
        return "direct_answer"

    logger.debug(f"[router] intent={intent!r} → tool_executor")
    return "tool_executor"


# ---------------------------------------------------------------------------
# Guard: check iteration limit after tool_executor
# ---------------------------------------------------------------------------

def _should_continue_after_tool(state: AgentState) -> str:
    """
    After tool execution, always proceed to answer_generator.
    Guard against excessive iterations.
    """
    from app.core.config import settings

    max_iter = getattr(settings, "NEXUSRAG_LG_MAX_ITERATIONS", 3)
    iterations = state.get("iterations", 0)

    if iterations >= max_iter:
        logger.warning(f"[router] Max iterations ({max_iter}) reached — forcing answer_generator")

    return "answer_generator"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """
    Build and compile the NexusRAG LangGraph agent.

    Returns a compiled graph ready for .invoke() or .astream_events().
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    graph.add_node("memory_recall", memory_recall)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("answer_generator", answer_generator)
    graph.add_node("direct_answer", direct_answer)

    # ── Add edges ──────────────────────────────────────────────────────────
    # Linear: START → memory_recall → intent_classifier
    graph.add_edge(START, "memory_recall")
    graph.add_edge("memory_recall", "intent_classifier")

    # Conditional: intent_classifier → [direct_answer | tool_executor]
    graph.add_conditional_edges(
        "intent_classifier",
        _route_by_intent,
        {
            "direct_answer": "direct_answer",
            "tool_executor": "tool_executor",
        },
    )

    # After tool: tool_executor → answer_generator
    graph.add_conditional_edges(
        "tool_executor",
        _should_continue_after_tool,
        {"answer_generator": "answer_generator"},
    )

    # Terminal nodes → END
    graph.add_edge("answer_generator", END)
    graph.add_edge("direct_answer", END)

    # ── Compile ────────────────────────────────────────────────────────────
    compiled = graph.compile()

    logger.info(
        "[agent_graph] Graph compiled: "
        "memory_recall → intent_classifier → [direct_answer | tool_executor → answer_generator]"
    )
    return compiled


# Module-level singleton — built once, reused across requests
_agent_graph = None


def get_agent_graph() -> StateGraph:
    """Return cached compiled graph (thread-safe singleton)."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph

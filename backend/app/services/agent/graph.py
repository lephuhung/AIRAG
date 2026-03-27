"""
LangGraph Agent Graph Builder
==============================

Builds and compiles the NexusRAG chat StateGraph.

Graph topology:
    START
      → abbr_expander              ← NEW: expand abbreviations immediately
      → memory_recall
      → intent_classifier
      → [direct_answer | write_executor | agent_rag_executor]
      → [answer_generator | write_executor]   ← write_executor also receives summarize
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
    abbr_expander,
    agent_rag_executor,
    answer_generator,
    direct_answer,
    write_executor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing function — decides next node after abbr_expander
# ---------------------------------------------------------------------------


def _route_by_intent(state: AgentState) -> str:
    """
    Conditional edge: route to direct_answer for greetings,
    write_executor for write intents, or agent_rag_executor for everything else.
    """
    intent = state.get("intent", "search")

    if intent in ("greeting", "personal"):
        logger.debug(f"[router] intent={intent!r} → direct_answer")
        return "direct_answer"

    if intent in ("write_summarize", "write_suggest_edits", "write_grammar_check"):
        logger.debug(f"[router] intent={intent!r} → write_executor")
        return "write_executor"

    logger.debug(f"[router] intent={intent!r} → agent_rag_executor")
    return "agent_rag_executor"


# ---------------------------------------------------------------------------
# Guard: check iteration limit after tool_executor
# ---------------------------------------------------------------------------


def _should_continue_after_rag(state: AgentState) -> str:
    """
    After agent_rag_executor:
    - If intent = summarize: route → write_executor (RAG fetched raw doc, Write will summarize)
    - If expanded_query is set (abbreviation + context detected), route back to agent_rag_executor
    - Otherwise proceed to answer_generator.
    Guard against excessive iterations.
    """
    from app.core.config import settings

    max_iter = getattr(settings, "NEXUSRAG_LG_MAX_ITERATIONS", 3)
    iterations = state.get("iterations", 0)
    intent = state.get("intent", "search")

    if iterations >= max_iter:
        logger.warning(
            f"[router] Max iterations ({max_iter}) reached — forcing answer_generator"
        )
        return "answer_generator"

    # summarize intent: RAG đã lấy raw doc content → route sang write_executor để tóm tắt
    if intent == "summarize":
        kg_summaries = state.get("kg_summaries", [])
        if kg_summaries:
            logger.info(
                f"[router] intent=summarize, kg_summaries present → write_executor"
            )
            return "write_executor"

    # Check if abbreviation expansion should trigger re-routing to agent_rag_executor
    expanded_query = state.get("expanded_query")
    if expanded_query and iterations < max_iter:
        logger.info(
            f"[router] Routing to agent_rag_executor with expanded query: {expanded_query!r}"
        )
        return "agent_rag_executor"

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
    graph.add_node("abbr_expander", abbr_expander)
    graph.add_node("agent_rag_executor", agent_rag_executor)
    graph.add_node("answer_generator", answer_generator)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("write_executor", write_executor)

    # ── Add edges ──────────────────────────────────────────────────────────
    # Linear: START → abbr_expander → memory_recall → intent_classifier
    graph.add_edge(START, "abbr_expander")
    graph.add_edge("abbr_expander", "memory_recall")
    graph.add_edge("memory_recall", "intent_classifier")

    # Conditional: intent_classifier → [direct_answer | write_executor | agent_rag_executor]
    graph.add_conditional_edges(
        "intent_classifier",
        _route_by_intent,
        {
            "direct_answer": "direct_answer",
            "write_executor": "write_executor",
            "agent_rag_executor": "agent_rag_executor",
        },
    )

    # After RAG: agent_rag_executor → [agent_rag_executor (retry) | write_executor | answer_generator]
    graph.add_conditional_edges(
        "agent_rag_executor",
        _should_continue_after_rag,
        {
            "agent_rag_executor": "agent_rag_executor",
            "write_executor": "write_executor",
            "answer_generator": "answer_generator",
        },
    )

    # Terminal nodes → END
    graph.add_edge("answer_generator", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("write_executor", END)

    # ── Compile ────────────────────────────────────────────────────────────
    compiled = graph.compile()

    logger.info(
        "[agent_graph] Graph compiled: "
        "memory_recall → intent_classifier → abbr_expander → "
        "[direct_answer | write_executor | agent_rag_executor → (write_executor | answer_generator)]"
    )
    return compiled


# Module-level singleton — built once, reused across requests
_agent_graph = None


def get_agent_graph() -> StateGraph:
    """Return cached compiled graph (thread-safe singleton). Call reset_agent_graph() to force rebuild."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def reset_agent_graph() -> None:
    """Force the singleton to rebuild on next call (e.g. after hot-reload in dev)."""
    global _agent_graph
    _agent_graph = None


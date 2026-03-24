"""
LangGraph Agent Module
======================

Semi-agentic chat pipeline implemented as a LangGraph StateGraph.

Graph flow:
    START
      → memory_recall         (load user memories from pgvector)
      → intent_classifier     (Qwen3-4B: classify + rewrite query)
      → [direct_answer]       (for greetings/chitchat)
      → [tool_executor]       (for search/list/summarize/kg_query)
      → [answer_generator]    (main LLM: generate answer with citations)
      → END

Usage::

    from app.services.agent import build_agent_graph

    app = build_agent_graph()
    async for event in app.astream_events(state, version="v2"):
        ...
"""

from app.services.agent.graph import build_agent_graph
from app.services.agent.state import AgentState

__all__ = ["build_agent_graph", "AgentState"]

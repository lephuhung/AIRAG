"""
LangGraph Agent Module
======================

Semi-agentic chat pipeline implemented as a LangGraph StateGraph.

Graph flow:
    START
      → memory_recall         (load user memories from Graphiti)
      → intent_classifier     (Qwen3-4B: classify + rewrite query)
      → abbr_expander         (global: expand abbreviations in query before routing)
      → [direct_answer]       (for greetings/chitchat/personal)
      → [write_executor]      (for write_summarize/suggest_edits/grammar_check)
      → [agent_rag_executor]  (for search/list/summarize/kg_query/doc_num/abbr)
          → [write_executor]  (when intent=summarize: RAG fetches doc, Write summarizes)
          → [answer_generator](for search/list/kg_query/doc_num/abbr)
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


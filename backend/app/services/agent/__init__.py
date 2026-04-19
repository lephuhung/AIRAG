"""
LangGraph Agent Module
======================

Supervisor-based multi-agent architecture.

Graph flow:
    START → supervisor (classify intent + route) → [rag | write | direct] → answer_generator → END

Usage::

    from app.services.agents import get_supervisor_graph

    app = get_supervisor_graph()
"""

from app.services.agents import get_supervisor_graph

# Re-export for backward compatibility
build_agent_graph = get_supervisor_graph

from app.services.agents.models import SupervisorState as AgentState

__all__ = ["build_agent_graph", "AgentState", "get_supervisor_graph"]
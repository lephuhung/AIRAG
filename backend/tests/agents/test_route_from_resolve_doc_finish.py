"""
Task 1 / B2 — `route_from_resolve_doc` FINISH precedence regression.

Pre-existing code at supervisor.py L3431 already sets ``target = "END"``
when ``next_agent == AgentType.FINISH`` (verified by reading the source).
The brief adds a regression test to lock the contract: an ambiguous /
not-found resolve_doc that finishes MUST end the graph even when other
fields (pending_intent, intent) would otherwise route to rag or
answer_generator.

This guards against a future refactor that reorders the priority
branches and accidentally routes a finished resolve_doc into another
agent — a known failure mode (the agent has already streamed its
final answer; routing it into rag / answer_generator would re-emit or
duplicate output).
"""
from __future__ import annotations

from langgraph.graph import END

from app.services.agents.supervisor import (
    AgentType,
    route_from_resolve_doc,
)


def test_finish_with_pending_search_returns_end():
    """FINISH takes precedence over pending_intent='search_section' +
    section_reference. Pre-fix the priority order would route to rag."""
    state = {
        "next_agent": AgentType.FINISH,
        "intent": "resolve_doc",
        "section_reference": "Điều 5",
        "pending_intent": "search_section",
        "document_ids": [],
    }
    assert route_from_resolve_doc(state) is END


def test_finish_with_pending_summarize_returns_end():
    """FINISH takes precedence over pending_intent='summarize'. Pre-fix
    this would route to answer_generator, double-answering a query
    whose resolve_doc has already streamed its final response."""
    state = {
        "next_agent": AgentType.FINISH,
        "intent": "resolve_doc",
        "section_reference": "",
        "pending_intent": "summarize",
        "document_ids": [],
    }
    assert route_from_resolve_doc(state) is END


def test_non_finish_with_pending_search_section_routes_to_rag():
    """Non-FINISH + pending_intent='search_section' + section_reference
    must route to rag (the normal continuation path). This is the
    positive control: FINISH precedence must not break the normal
    flow."""
    state = {
        "next_agent": AgentType.RAG,
        "intent": "resolve_doc",
        "section_reference": "Điều 5",
        "pending_intent": "search_section",
        "document_ids": [],
    }
    target = route_from_resolve_doc(state)
    assert target == "rag"
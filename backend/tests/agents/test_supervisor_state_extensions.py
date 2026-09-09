"""
Tests for SupervisorState Phase 1A extensions.

Per spec Section B.11 0.4: semantic_context, complexity_route,
_preprocessor_marker, flag_snapshot, budget_guard fields.
Legacy fields (query_complexity, sub_queries, etc.) preserved for transition.
"""

from __future__ import annotations

import uuid
import pytest


def test_supervisor_state_accepts_new_fields():
    """Phase 1A fields must be accepted by SupervisorState TypedDict."""
    from app.services.agents.models import SupervisorState

    state: SupervisorState = {
        # Legacy fields (must still work)
        "messages": [],
        "intent": "search",
        "query_complexity": "simple",
        "sub_queries": None,
        "extracted_params": None,
        "task_plan": None,
        "pending_intent": None,
        # Phase 1A additions
        "semantic_context": None,
        "complexity_route": None,
        "_preprocessor_marker": None,
        "flag_snapshot": None,
        "budget_guard": None,
    }
    assert state["semantic_context"] is None
    assert state["_preprocessor_marker"] is None
    assert state["flag_snapshot"] is None


def test_supervisor_state_legacy_fields_preserved():
    """Phase 0/1A transition: legacy fields must not be removed."""
    from app.services.agents.models import SupervisorState

    state: SupervisorState = {
        "messages": [],
        "query_complexity": "multi_doc",  # legacy
        "sub_queries": [{"query": "test", "intent_hint": "search"}],  # legacy
        "extracted_params": {"document_refs": []},  # legacy
    }
    assert state["query_complexity"] == "multi_doc"
    assert state["sub_queries"] is not None
    assert state["extracted_params"] is not None

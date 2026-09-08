"""
Phase 0 / B3 — `needs_comparison` prompt consumption regression test.

Per F.3/O68: B3 schema/producer fixed by Task-1 but prompt consumption
path NOT regression-tested. This test verifies the answer-generator prompt
assembly correctly includes comparison instruction when needs_comparison=True
and excludes it when needs_comparison=False.

The comparison instruction is injected via get_instructions_for_intent() in
nodes.py:answer_generator at lines ~918-927.
"""
from __future__ import annotations

from app.prompts.agents.answer_instructions import get_instructions_for_intent


def test_answer_generator_includes_comparison_when_flag_true():
    """state with needs_comparison=True → prompt contains 'COMPARISON TASK' instruction."""
    instructions = get_instructions_for_intent(
        intent="search",
        enable_thinking=False,
        needs_comparison=True
    )
    assert "COMPARISON TASK" in instructions
    assert "so sánh" in instructions.lower() or "comparison" in instructions.lower()
    assert "USER CONTEXT" in instructions
    assert "REQUIREMENTS" in instructions


def test_answer_generator_excludes_comparison_when_flag_false():
    """state with needs_comparison=False → prompt does NOT contain comparison instruction."""
    instructions = get_instructions_for_intent(
        intent="search",
        enable_thinking=False,
        needs_comparison=False
    )
    assert "COMPARISON TASK" not in instructions
    assert "so sánh user context" not in instructions.lower()
    assert "compare user context vs document requirements" not in instructions.lower()


def test_answer_generator_includes_thinking_directive_when_enabled():
    """enable_thinking=True → prompt contains thinking directive."""
    instructions = get_instructions_for_intent(
        intent="search",
        enable_thinking=True,
        needs_comparison=False
    )
    assert "Yêu cầu cho phần suy nghĩ" in instructions or "Thinking" in instructions


def test_answer_generator_rag_instructions_for_search_intent():
    """search intent → RAG instructions with citation rules."""
    instructions = get_instructions_for_intent(
        intent="search",
        enable_thinking=False,
        needs_comparison=False
    )
    assert "Cite sources" in instructions
    assert "[a3z9]" in instructions


def test_answer_generator_mongo_instructions_for_phone_search():
    """mongo_search_phone intent → mongo instructions with strict anti-hallucination rules."""
    instructions = get_instructions_for_intent(
        intent="mongo_search_phone",
        enable_thinking=False,
        needs_comparison=False
    )
    assert "phone records" in instructions.lower()
    assert "hallucination" in instructions.lower()


def test_answer_generator_abbreviation_instructions():
    """search_abbr intent → minimal abbreviation instructions."""
    instructions = get_instructions_for_intent(
        intent="search_abbr",
        enable_thinking=False,
        needs_comparison=False
    )
    assert "abbreviation meaning" in instructions.lower()

"""
AgentState — LangGraph state definition for the NexusRAG chat agent.
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state flowing through the LangGraph graph nodes.

    Fields use reducers where accumulation is needed (messages, sources, images).
    Plain assignment (no reducer) for scalar/control fields.
    """

    # ── Conversation ─────────────────────────────────────────────────────────
    # add_messages reducer: appends new messages, never overwrites.
    messages: Annotated[list, add_messages]

    # ── Request context ───────────────────────────────────────────────────────
    workspace_ids: list[int]
    document_ids: Optional[list[int]]
    user_id: Optional[int]
    session_id: Optional[str]
    system_prompt: str
    enable_thinking: bool

    # ── Retrieval accumulator (reducer: extend lists) ─────────────────────────
    sources: Annotated[list, lambda a, b: a + b]  # ChatSourceChunk dicts
    images: Annotated[list, lambda a, b: a + b]  # ChatImageRef dicts
    image_parts: Annotated[list, lambda a, b: a + b]  # raw bytes for vision LLM
    kg_summaries: Annotated[list, lambda a, b: a + b]  # KG insight strings

    # Shared citation ID registry — plain set, nodes mutate in-place
    # (LangGraph copies state per node, so we use a dict wrapper trick)
    existing_citation_ids: set

    # ── Agent control ─────────────────────────────────────────────────────────
    intent: str  # "greeting" | "search" | "list_docs" | "summarize" | "kg_query"
    rewritten_query: str  # query after Qwen3-4B rewrite (used by tool_executor)
    iterations: int  # loop guard — nodes increment, graph checks max
    tool_called: bool  # True after first tool execution

    # ── Memory ───────────────────────────────────────────────────────────────
    user_memory_context: str  # formatted memories injected into system prompt

    # ── Output ───────────────────────────────────────────────────────────────
    final_answer: str
    citation_map: dict  # citation_id → {source_file, page_no, ...}

    # ── Abbreviation search results ───────────────────────────────────────
    abbreviation_results: list[
        dict
    ]  # [{"short_form": ..., "full_form": ..., "description": ...}]
    expanded_query: str  # query expanded with abbreviation full_form for routing


# Valid intents recognised by the classifier
VALID_INTENTS = {
    "greeting",
    "personal",
    "search",
    "list_docs",
    "summarize",
    "kg_query",
    "search_doc_num",
    "search_abbr",
}

# Default initial values — merge with per-request values when building state
DEFAULT_STATE: dict = {
    "messages": [],
    "workspace_ids": [],
    "document_ids": None,
    "user_id": None,
    "session_id": None,
    "system_prompt": "",
    "enable_thinking": False,
    "sources": [],
    "images": [],
    "image_parts": [],
    "kg_summaries": [],
    "existing_citation_ids": set(),
    "intent": "search",
    "rewritten_query": "",
    "iterations": 0,
    "tool_called": False,
    "user_memory_context": "",
    "final_answer": "",
    "citation_map": {},
}

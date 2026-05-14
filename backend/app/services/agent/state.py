"""
AgentState — LangGraph state definition for the NexusRAG chat agent.
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


import uuid
import operator

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
    workspace_ids: list[uuid.UUID]
    document_ids: Optional[list[uuid.UUID]]
    user_id: Optional[uuid.UUID]
    session_id: Optional[str]
    system_prompt: str
    enable_thinking: bool

    # ── Retrieval accumulator (reducer: extend lists) ─────────────────────────
    # We use list[ChatSourceChunk] if imported, but TypedDict doesn't strictly enforce.
    # Annotated[list, operator.add] is the LangGraph way to extend lists instead of overwriting.
    sources: Annotated[list, operator.add]  # ChatSourceChunk objects
    images: Annotated[list, operator.add]  # ChatImageRef objects
    image_parts: Annotated[list, operator.add]  # raw bytes for vision LLM
    kg_summaries: Annotated[list, operator.add]  # KG insight strings
    abbreviation_results: Annotated[list, operator.add] # List[dict]
    mongo_results: Annotated[list, operator.add] # List[dict]
    potential_abbreviations: Annotated[list, operator.add] # List[str]

    # Shared citation ID registry — dict wrapper (set-like, survives LangGraph immutability)
    existing_citation_ids: dict

    # ── Agent control ─────────────────────────────────────────────────────────
    intent: str
    rewritten_query: str
    original_query: str
    iterations: int
    tool_called: bool
    should_loop_back: bool
    needs_memory: bool

    # ── Memory ───────────────────────────────────────────────────────────────
    user_memory_context: str

    # ── Output ───────────────────────────────────────────────────────────────
    final_answer: str
    citation_map: dict

    # ── Section search results ───────────────────────────────────────
    section_reference: str | None


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
    # write intents
    "write_summarize",
    "write_suggest_edits",
    "write_grammar_check",
    "write_format_check",
    # mongo people search intents
    "mongo_search_cccd",
    "mongo_search_name",
    "mongo_search_bhxh",
    "mongo_search_phone",
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
    "existing_citation_ids": {},
    "intent": "search",
    "rewritten_query": "",
    "original_query": "",
    "iterations": 0,
    "tool_called": False,
    "user_memory_context": "",
    "final_answer": "",
    "citation_map": {},
    "write_action": "",
    "text_input": "",
    "format_data": None,
    "file_name": None,
    "abbreviation_results": [],
    "expanded_query": "",
    "mongo_results": [],
    "potential_abbreviations": [],
}

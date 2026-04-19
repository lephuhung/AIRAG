"""
Shared Models for Supervisor-Based Multi-Agent Architecture
=========================================================

Contains:
- Intent constants (16 intent categories)
- AgentType constants (rag, write, direct, finish)
- SupervisorState TypedDict
- INTENT_TO_AGENT mapping
"""

from typing import Literal, TypedDict


# =============================================================================
# Intent Definitions (16 intents from nodes.py:_CLASSIFIER_SYSTEM)
# =============================================================================

class Intent:
    GREETING = "greeting"
    PERSONAL = "personal"
    SEARCH = "search"
    LIST_DOCS = "list_docs"
    SUMMARIZE = "summarize"
    KG_QUERY = "kg_query"
    SEARCH_DOC_NUM = "search_doc_num"
    SEARCH_ABBR = "search_abbr"
    WRITE_SUMMARIZE = "write_summarize"
    WRITE_SUGGEST_EDITS = "write_suggest_edits"
    WRITE_GRAMMAR_CHECK = "write_grammar_check"
    WRITE_FORMAT_CHECK = "write_format_check"
    MONGO_SEARCH_CCCD = "mongo_search_cccd"
    MONGO_SEARCH_NAME = "mongo_search_name"
    MONGO_SEARCH_BHxh = "mongo_search_bhxh"
    MONGO_SEARCH_PHONE = "mongo_search_phone"
    MONGO_SEARCH_ADVANCED = "mongo_search_advanced"

    ALL = {
        GREETING, PERSONAL, SEARCH, LIST_DOCS, SUMMARIZE, KG_QUERY,
        SEARCH_DOC_NUM, SEARCH_ABBR, WRITE_SUMMARIZE, WRITE_SUGGEST_EDITS,
        WRITE_GRAMMAR_CHECK, WRITE_FORMAT_CHECK, MONGO_SEARCH_CCCD,
        MONGO_SEARCH_NAME, MONGO_SEARCH_BHxh, MONGO_SEARCH_PHONE,
        MONGO_SEARCH_ADVANCED,
    }


# =============================================================================
# Agent Types
# =============================================================================

class AgentType:
    RAG = "rag"
    WRITE = "write"
    DIRECT = "direct"
    PEOPLE = "people"
    FINISH = "finish"


# =============================================================================
# Intent → Agent Mapping
# =============================================================================

# INTENT_TO_AGENT: reference mapping — không dùng trong runtime routing.
# Supervisor hỏi LLM trực tiếp để lấy next_agent.
# Giữ lại để documentation và là nguồn tham chiếu khi thêm intent mới.
INTENT_TO_AGENT: dict[str, str] = {
    Intent.GREETING: AgentType.DIRECT,
    Intent.PERSONAL: AgentType.DIRECT,
    Intent.SEARCH: AgentType.RAG,
    Intent.LIST_DOCS: AgentType.RAG,
    Intent.SUMMARIZE: AgentType.RAG,
    Intent.KG_QUERY: AgentType.RAG,
    Intent.SEARCH_DOC_NUM: AgentType.RAG,
    Intent.SEARCH_ABBR: AgentType.RAG,
    Intent.MONGO_SEARCH_CCCD: AgentType.PEOPLE,
    Intent.MONGO_SEARCH_NAME: AgentType.PEOPLE,
    Intent.MONGO_SEARCH_BHxh: AgentType.PEOPLE,
    Intent.MONGO_SEARCH_PHONE: AgentType.PEOPLE,
    Intent.MONGO_SEARCH_ADVANCED: AgentType.PEOPLE,
    Intent.WRITE_SUMMARIZE: AgentType.WRITE,
    Intent.WRITE_SUGGEST_EDITS: AgentType.WRITE,
    Intent.WRITE_GRAMMAR_CHECK: AgentType.WRITE,
    Intent.WRITE_FORMAT_CHECK: AgentType.WRITE,
}


# =============================================================================
# Supervisor State
# =============================================================================

class SupervisorState(TypedDict, total=False):
    """State shared across all agents in the supervisor graph."""

    # Conversation
    messages: list

    # Intent classification results
    intent: str
    rewritten_query: str
    original_query: str

    # Workspace context
    workspace_ids: list[int]
    document_ids: list[int] | None
    user_id: int | None
    session_id: str | None
    system_prompt: str
    enable_thinking: bool

    # Accumulated retrieval results
    sources: list
    images: list
    image_parts: list
    kg_summaries: list
    abbreviation_results: list
    mongo_results: list

    # Write agent inputs
    write_action: str
    text_input: str
    format_data: dict | None
    file_name: str | None

    # Format evaluation (Option A - RAG-based evaluation)
    # Using dict | None instead of TypedDict references to avoid forward-reference NameError.
    extracted_format_info: dict | None
    format_evaluation_result: dict | None

    # Output
    final_answer: str | None

    # Supervisor control fields
    next_agent: Literal["rag", "write", "people", "direct", "finish"] | None
    iterations: int
    user_memory_context: str
    potential_abbreviations: list[str]
    expanded_query: str


# =============================================================================
# Write Action Constants
# =============================================================================

# =============================================================================
# Format Evaluation Types (Option A)
# =============================================================================

class ExtractedFormatInfo(TypedDict, total=False):
    """Typed format information extracted from a Word document."""
    file_name: str
    margins: dict  # {top, bottom, left, right}
    font_sizes: list[int]
    fonts: list[str]
    line_spacing: list[dict]
    paragraph_count: int
    table_count: int


class FormatEvaluationResult(TypedDict, total=False):
    """Result from RAG-based format evaluation against 30/2020/NĐ-CP."""
    is_valid: bool
    overall_status: str  # "đạt", "yếu", "cần cải thiện"
    issues: list[dict]  # [{severity, category, description, suggestion}]
    standard_reference: str  # Relevant section from 30/2020/NĐ-CP


# =============================================================================
# Write Action Constants
# =============================================================================

class WriteAction:
    SUMMARIZE = "summarize"
    EXTRACT_KEY_POINTS = "extract_key_points"
    SUGGEST_EDITS = "suggest_edits"
    GRAMMAR_CHECK = "grammar_check"
    FORMAT_CHECK = "format_check"

    ALL = {SUMMARIZE, EXTRACT_KEY_POINTS, SUGGEST_EDITS, GRAMMAR_CHECK, FORMAT_CHECK}

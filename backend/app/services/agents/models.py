"""
Shared Models for Supervisor-Based Multi-Agent Architecture
=========================================================

Contains:
- Intent constants (16 intent categories)
- AgentType constants (rag, write, direct, finish)
- SupervisorState TypedDict
- INTENT_TO_AGENT mapping
"""

from typing import Literal, TypedDict, Annotated
import uuid
import operator
from app.schemas.rag import ChatSourceChunk, ChatImageRef


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
    SEARCH_SECTION = "search_section"
    RESOLVE_DOC = "resolve_doc"
    WRITE_SUMMARIZE = "write_summarize"
    WRITE_SUGGEST_EDITS = "write_suggest_edits"
    WRITE_GRAMMAR_CHECK = "write_grammar_check"
    WRITE_FORMAT_CHECK = "write_format_check"
    MONGO_SEARCH_CCCD = "mongo_search_cccd"
    MONGO_SEARCH_NAME = "mongo_search_name"
    MONGO_SEARCH_BHXH = "mongo_search_bhxh"
    MONGO_SEARCH_PHONE = "mongo_search_phone"
    MONGO_SEARCH_ADVANCED = "mongo_search_advanced"

    ALL = {
        GREETING, PERSONAL, SEARCH, LIST_DOCS, SUMMARIZE, KG_QUERY,
        SEARCH_DOC_NUM, SEARCH_ABBR, SEARCH_SECTION, RESOLVE_DOC,
        WRITE_SUMMARIZE, WRITE_SUGGEST_EDITS,
        WRITE_GRAMMAR_CHECK, WRITE_FORMAT_CHECK, MONGO_SEARCH_CCCD,
        MONGO_SEARCH_NAME, MONGO_SEARCH_BHXH, MONGO_SEARCH_PHONE,
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
    ANSWER_GENERATOR = "answer_generator"
    RESOLVE_DOC = "resolve_doc"  # Phase 2: dedicated resolve_doc agent (future)


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
    Intent.SEARCH_SECTION: AgentType.RAG,
    Intent.MONGO_SEARCH_CCCD: AgentType.PEOPLE,
    Intent.MONGO_SEARCH_NAME: AgentType.PEOPLE,
    Intent.MONGO_SEARCH_BHXH: AgentType.PEOPLE,
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
    workspace_ids: list[uuid.UUID]
    document_ids: list[uuid.UUID] | None
    user_id: uuid.UUID | None
    session_id: str | None
    system_prompt: str
    enable_thinking: bool

    # Accumulated retrieval results
    sources: Annotated[list[ChatSourceChunk], operator.add]
    images: Annotated[list[ChatImageRef], operator.add]
    image_parts: Annotated[list, operator.add]
    kg_summaries: Annotated[list, operator.add]
    abbreviation_results: Annotated[list, operator.add]
    mongo_results: Annotated[list, operator.add]

    # Section search results
    section_reference: str | None

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

    # Phase 1: Smart RAG Search Routing
    # Controls which retrieval mode is used: "vector" | "kg" | "hybrid"
    # Set by supervisor_node based on intent, consumed by _tool_search
    search_mode: str  # "vector" | "kg" | "hybrid"

    # Phase 4: Query Clarification & Smart Abbreviation Detection
    suspected_abbreviations: Annotated[list[str], operator.add]   # Từ nghi ngờ là viết tắt (từ thinking)
    clarification_needed: bool            # True khi cần hỏi user thêm thông tin
    clarification_message: str            # Nội dung câu hỏi clarification gửi cho user

    # Supervisor control fields
    next_agent: Literal["rag", "write", "people", "direct", "finish"] | None
    iterations: int
    user_memory_context: str
    potential_abbreviations: Annotated[list[str], operator.add]
    expanded_query: str
    should_loop_back: bool  # True when abbreviation was found → re-classify with full form

    # Phase 3: Smart Memory Recall
    # True when supervisor detects personal reference in query (intent=personal OR keywords like "tôi", "đơn vị tôi")
    # Controls whether memory_recall → query_enricher runs before target agent
    needs_memory: bool

    # Phase 4: Plan-Aware Supervisor
    # task_plan: ordered list of intents the supervisor plans to execute
    # e.g. ["resolve_doc", "search_section"] for "Tóm tắt điều 3 Luật ANM 2018"
    # First item = current step (= intent), remaining = pending steps
    task_plan: list[str] | None
    # pending_intent: the user's FINAL GOAL intent, preserved while prerequisite
    # steps execute. e.g. "summarize" while resolve_doc runs first.
    pending_intent: str | None

    # Phase 5: Query Analyzer output (set by query_analyzer_node)
    # sub_queries: decomposed sub-questions for multi-step execution
    # e.g. [{"query": "NĐ 13 về DLCN", "intent_hint": "resolve_doc"}, ...]
    sub_queries: list[dict] | None
    # extracted_params: structured params extracted from the user query
    # e.g. {"document_refs": ["NĐ 13", "Luật ANM"], "sections": ["Điều 5"]}
    extracted_params: dict | None
    # query_complexity: "simple"|"multi_doc"|"multi_section"|"cross_agent"|"comparison"
    query_complexity: str | None

    # Phase 5: Multi-step execution tracking
    # current_step_index: which step in sub_queries are we executing
    current_step_index: int
    # accumulated_results: results from completed sub-query steps
    # Each entry: {"step_intent": str, "sources": [...], "kg_summaries": [...]}
    accumulated_results: Annotated[list[dict], operator.add]
    # retry_count: how many retries for current step (max 2)
    retry_count: int
    # retry_strategy: what fallback strategy is being used
    retry_strategy: str | None


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

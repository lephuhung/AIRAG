"""
Agent Write
==========

Writing and text processing capabilities for the NexusRAG agent system.
This agent handles summarization, editing suggestions, and grammar checks.

State: AgentWriteState
Tools:
    - summarize_text: Summarize a text passage
    - suggest_edits: Suggest improvements to text
    - check_grammar: Check grammar and style
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.services.llm import get_llm_provider
from app.services.llm.types import LLMMessage

logger = logging.getLogger(__name__)


# =============================================================================
# State
# =============================================================================


class AgentWriteState(TypedDict, total=False):
    """State for the write agent."""

    messages: list[dict]
    user_id: int
    workspace_ids: list[int]
    text_input: str
    write_action: str
    result: str
    error: str | None


# =============================================================================
# Tool Nodes
# =============================================================================


async def summarize_text_node(state: AgentWriteState) -> AgentWriteState:
    """Summarize the provided text."""
    text = state.get("text_input", "")
    action = state.get("write_action", "summarize")

    if not text:
        return {
            **state,
            "error": "No text provided",
            "result": "Vui lòng cung cấp văn bản cần xử lý.",
        }

    try:
        llm = get_llm_provider()

        if action == "summarize":
            prompt = (
                "Hãy tóm tắt văn bản sau bằng tiếng Việt một cách ngắn gọn và súc tích. "
                "Giữ lại các ý chính và thông tin quan trọng.\n\n"
                f"Văn bản:\n{text}"
            )
        elif action == "extract_key_points":
            prompt = (
                "Hãy trích xuất các điểm chính từ văn bản sau bằng tiếng Việt. "
                "Liệt kê các ý quan trọng nhất.\n\n"
                f"Văn bản:\n{text}"
            )
        else:
            prompt = (
                f"Hãy thực hiện yêu cầu '{action}' trên văn bản sau bằng tiếng Việt.\n\n"
                f"Văn bản:\n{text}"
            )

        result = await llm.acomplete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=512,
        )

        result_text = (
            result
            if isinstance(result, str)
            else getattr(result, "content", str(result))
        )

        return {**state, "result": result_text, "error": None}

    except Exception as e:
        logger.error(f"[summarize_text_node] Failed: {e}")
        return {
            **state,
            "error": str(e),
            "result": "Không thể tóm tắt văn bản. Vui lòng thử lại.",
        }


async def suggest_edits_node(state: AgentWriteState) -> AgentWriteState:
    """Suggest improvements to the provided text."""
    text = state.get("text_input", "")

    if not text:
        return {
            **state,
            "error": "No text provided",
            "result": "Vui lòng cung cấp văn bản cần chỉnh sửa.",
        }

    try:
        llm = get_llm_provider()

        prompt = (
            "Hãy phân tích và đề xuất chỉnh sửa cho văn bản sau. "
            "Tập trung vào:\n"
            "1. Cải thiện cấu trúc và luồng nội dung\n"
            "2. Làm rõ các câu ý nghĩa mơ hồ\n"
            "3. Đề xuất cách diễn đạt tốt hơn\n"
            "4. Cách diễn đạt phù hợp ngữ cảnh hơn\n\n"
            "Trả lời bằng tiếng Việt.\n\n"
            f"Văn bản:\n{text}"
        )

        result = await llm.acomplete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=1024,
        )

        result_text = (
            result
            if isinstance(result, str)
            else getattr(result, "content", str(result))
        )

        return {**state, "result": result_text, "error": None}

    except Exception as e:
        logger.error(f"[suggest_edits_node] Failed: {e}")
        return {
            **state,
            "error": str(e),
            "result": "Không thể đề xuất chỉnh sửa. Vui lòng thử lại.",
        }


async def check_grammar_node(state: AgentWriteState) -> AgentWriteState:
    """Check grammar and style of the provided text."""
    text = state.get("text_input", "")

    if not text:
        return {
            **state,
            "error": "No text provided",
            "result": "Vui lòng cung cấp văn bản cần kiểm tra.",
        }

    try:
        llm = get_llm_provider()

        prompt = (
            "Hãy kiểm tra ngữ pháp và phong cách viết của văn bản sau. "
            "Chỉ ra các lỗi ngữ pháp, chính tả, và đề xuất cách sửa. "
            "Nếu văn bản tốt, hãy xác nhận điều đó.\n\n"
            "Trả lời bằng tiếng Việt.\n\n"
            f"Văn bản:\n{text}"
        )

        result = await llm.acomplete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.2,
            max_tokens=512,
        )

        result_text = (
            result
            if isinstance(result, str)
            else getattr(result, "content", str(result))
        )

        return {**state, "result": result_text, "error": None}

    except Exception as e:
        logger.error(f"[check_grammar_node] Failed: {e}")
        return {
            **state,
            "error": str(e),
            "result": "Không thể kiểm tra ngữ pháp. Vui lòng thử lại.",
        }


async def answer_node(state: AgentWriteState) -> AgentWriteState:
    """Generate the final answer based on the tool result."""
    result = state.get("result", "")
    error = state.get("error")

    if error:
        return {**state, "messages": [{"role": "assistant", "content": error}]}

    return {
        **state,
        "messages": [{"role": "assistant", "content": result}],
    }


# =============================================================================
# Graph Builder
# =============================================================================


def create_agent_write() -> StateGraph:
    """
    Create the write agent graph.

    The graph routes to different tool nodes based on write_action:
    - summarize -> summarize_text_node
    - suggest_edits -> suggest_edits_node
    - grammar_check -> check_grammar_node
    """

    graph = StateGraph(AgentWriteState)

    # Add nodes
    graph.add_node("summarize_text", summarize_text_node)
    graph.add_node("suggest_edits", suggest_edits_node)
    graph.add_node("check_grammar", check_grammar_node)
    graph.add_node("answer", answer_node)

    # Set entry point - route based on write_action
    def route_write_action(state: AgentWriteState) -> str:
        action = state.get("write_action", "summarize")

        if action in ("summarize", "extract_key_points"):
            return "summarize_text"
        elif action == "suggest_edits":
            return "suggest_edits"
        elif action == "grammar_check":
            return "check_grammar"
        else:
            # Default to summarize
            return "summarize_text"

    graph.set_conditional_entry_point(
        route_write_action,
        {
            "summarize_text": "summarize_text",
            "suggest_edits": "suggest_edits",
            "check_grammar": "check_grammar",
        },
    )

    # All paths lead to answer
    graph.add_edge("summarize_text", "answer")
    graph.add_edge("suggest_edits", "answer")
    graph.add_edge("check_grammar", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


# Export the compiled graph
agent_write_graph = create_agent_write()

"""
Agent Write
==========

Writing and text processing capabilities for the NexusRAG agent system.
This agent handles summarization, editing suggestions, grammar checks,
and document format checking.

State: AgentWriteState
Tools:
    - summarize_text: Summarize a text passage
    - suggest_edits: Suggest improvements to text
    - check_grammar: Check grammar and style
    - check_format: Check document formatting against Vietnamese standards
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from app.services.llm import get_llm_provider
from app.services.llm.types import LLMMessage
from app.services.agents.docx_formatter_tools import (
    analyze_format_issues,
    rag_lookup_format_standards,
)
from app.services.agent.streaming import push_event, get_current_db

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
    # For format checking
    format_data: dict[str, Any] | None
    file_name: str | None


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

        result_text = ""
        try:
            async for chunk in llm.astream(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=2048,
            ):
                if chunk.type == "text" and chunk.text:
                    result_text += chunk.text
                    await push_event(state, "token", chunk.text)
        except Exception as e:
            logger.warning(f"[summarize_text_node] astream failed, falling back to acomplete: {e}")
            result = await llm.acomplete(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=2048,
            )
            result_text = result if isinstance(result, str) else getattr(result, "content", str(result))
            # Fallback: push all at once (not streaming)
            if result_text:
                await push_event(state, "token", result_text)

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

        result_text = ""
        try:
            async for chunk in llm.astream(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=1024,
            ):
                if chunk.type == "text" and chunk.text:
                    result_text += chunk.text
                    await push_event(state, "token", chunk.text)
        except Exception as e:
            logger.warning(f"[suggest_edits_node] astream failed, falling back to acomplete: {e}")
            result = await llm.acomplete(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=1024,
            )
            result_text = result if isinstance(result, str) else getattr(result, "content", str(result))
            if result_text:
                await push_event(state, "token", result_text)

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

        result_text = ""
        try:
            async for chunk in llm.astream(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.2,
                max_tokens=1024,
            ):
                if chunk.type == "text" and chunk.text:
                    result_text += chunk.text
                    await push_event(state, "token", chunk.text)
        except Exception as e:
            logger.warning(f"[check_grammar_node] astream failed, falling back to acomplete: {e}")
            result = await llm.acomplete(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.2,
                max_tokens=1024,
            )
            result_text = result if isinstance(result, str) else getattr(result, "content", str(result))
            if result_text:
                await push_event(state, "token", result_text)

        return {**state, "result": result_text, "error": None}

    except Exception as e:
        logger.error(f"[check_grammar_node] Failed: {e}")
        return {
            **state,
            "error": str(e),
            "result": "Không thể kiểm tra ngữ pháp. Vui lòng thử lại.",
        }


async def check_format_node(state: AgentWriteState) -> AgentWriteState:
    """Check document formatting against Vietnamese government standards."""
    format_data = state.get("format_data")
    file_name = state.get("file_name", "tài liệu")

    if not format_data:
        # Safety net — should be caught earlier in write_executor
        return {
            **state,
            "error": "No format data provided",
            "result": (
                "Không có dữ liệu định dạng để kiểm tra.\n\n"
                "Vui lòng đính kèm một file Word (.docx) và hỏi lại."
            ),
        }

    try:
        llm = get_llm_provider()

        # Analyze issues
        issues = analyze_format_issues(format_data)

        margins = format_data.get("margins", {})
        font_samples = format_data.get("font_samples", [])
        format_data.get("line_spacing", [])  # kept for future use
        paragraph_count = format_data.get("paragraph_count", 0)
        table_count = format_data.get("table_count", 0)

        # Get standards via RAG
        workspace_ids = state.get("workspace_ids", [])
        standards_contexts = []
        if workspace_ids:
            try:
                from app.services.agent.streaming import get_current_db
                db = get_current_db()
                if db:
                    standards_contexts = await rag_lookup_format_standards(
                        query="quy chuẩn trình bày văn bản hành chính Việt Nam căn lề giãn dòng cỡ chữ",
                        workspace_ids=workspace_ids,
                        db=db,
                        top_k=5,
                    )
            except Exception as e:
                logger.warning(f"[check_format_node] RAG lookup failed: {e}")

        # Build font info
        font_sizes = [f["font_size"] for f in font_samples if f.get("font_size")]
        from collections import Counter
        size_counts = Counter(font_sizes)
        most_common_sizes = size_counts.most_common(3)

        unique_fonts = set(f["font_name"] for f in font_samples if f.get("font_name"))

        # Build prompt
        prompt = f"""Phân tích thông tin định dạng sau và đưa ra báo cáo kiểm tra chi tiết.

## TÊN TỆP: {file_name}

## THÔNG TIN ĐỊNH DẠNG ĐÃ TRÍCH XUẤT:

### 1. Căn lề (cm):
- Trên (top): {margins.get('top', 'N/A')} cm
- Dưới (bottom): {margins.get('bottom', 'N/A')} cm
- Trái (left): {margins.get('left', 'N/A')} cm
- Phải (right): {margins.get('right', 'N/A')} cm
- Chuẩn thông thường: Top 2cm, Bottom 2cm, Left 3cm, Right 2cm

### 2. Cỡ chữ:
"""
        # Font info - report actual values or explicitly state missing
        if most_common_sizes:
            prompt += f"- Cỡ chữ phổ biến: {', '.join([f'{s}pt ({c} lần)' for s, c in most_common_sizes])} pt\n"
        else:
            prompt += "- Cỡ chữ: Không trích xuất được (yêu cầu kiểm tra thủ công)\n"
        # Line spacing - extract actual values or mark as missing
        line_spacing_data = format_data.get("line_spacing", [])
        has_line_spacing = any(ls.get("line_spacing_value") for ls in line_spacing_data)
        if has_line_spacing:
            spacing_values = [f"{ls.get('line_spacing_value')} ({ls.get('line_spacing_type', 'unknown')})"
                            for ls in line_spacing_data if ls.get("line_spacing_value")]
            prompt += f"- Khoảng cách dòng: {', '.join(set(spacing_values[:3]))}\n"
        else:
            prompt += "- Khoảng cách dòng: Không trích xuất được (mặc định nên dùng 1.5 lines)\n"

        prompt += f"- Chuẩn: 13pt cho nội dung, 14pt cho tiêu đề, 1.5 lines cho khoảng cách dòng\n"

        if unique_fonts:
            prompt += f"- Font chữ: {', '.join(list(unique_fonts)[:5])}\n"
        else:
            prompt += "- Font chữ: Không trích xuất được (mặc định nên dùng Times New Roman)\n"

        prompt += f"\n### 3. Số đoạn văn: {paragraph_count}\n"
        prompt += f"### 4. Số bảng: {table_count}\n"

        # Add issues
        prompt += "\n## CÁC VẤN ĐỀ PHÁT HIỆN:\n"
        if issues:
            for i, issue in enumerate(issues, 1):
                prompt += f"{i}. [{issue['severity'].upper()}] {issue['detail']}\n"
                prompt += f"   → {issue['suggestion']}\n"
        else:
            prompt += "Không phát hiện vấn đề định dạng nghiêm trọng.\n"

        # Add standards from RAG
        if standards_contexts:
            prompt += "\n## TIÊU CHUẨN LIÊN QUAN (từ RAG):\n"
            for i, ctx in enumerate(standards_contexts[:3], 1):
                preview = ctx[:500] + "..." if len(ctx) > 500 else ctx
                prompt += f"{i}. {preview}\n\n"

        prompt += """
## YÊU CẦU:
1. Đưa ra đánh giá TỔNG QUAN về định dạng văn bản (đạt/yếu/cần cải thiện)
2. Liệt kê các vấn đề cụ thể theo mức độ nghiêm trọng (error/warning/info)
3. Đề xuất cách SỬA chữa cụ thể cho từng vấn đề
4. Tham khảo tiêu chuẩn đã tìm kiếm được (nếu có)

Trả lời bằng tiếng Việt, rõ ràng và có cấu trúc.
"""

        result_text = ""
        try:
            async for chunk in llm.astream(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=2048,
            ):
                if chunk.type == "text" and chunk.text:
                    result_text += chunk.text
                    await push_event(state, "token", chunk.text)
        except Exception as e:
            logger.warning(f"[check_format_node] astream failed, falling back to acomplete: {e}")
            result = await llm.acomplete(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=2048,
            )
            result_text = result if isinstance(result, str) else getattr(result, "content", str(result))
            if result_text:
                await push_event(state, "token", result_text)

        return {**state, "result": result_text, "error": None}

    except Exception as e:
        logger.error(f"[check_format_node] Failed: {e}")
        return {
            **state,
            "error": str(e),
            "result": "Không thể kiểm tra định dạng. Vui lòng thử lại.",
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
    - format_check -> check_format_node
    """

    graph = StateGraph(AgentWriteState)

    # Add nodes
    graph.add_node("summarize_text", summarize_text_node)
    graph.add_node("suggest_edits", suggest_edits_node)
    graph.add_node("check_grammar", check_grammar_node)
    graph.add_node("check_format", check_format_node)
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
        elif action == "format_check":
            return "check_format"
        else:
            # Default to summarize
            return "summarize_text"

    graph.set_conditional_entry_point(
        route_write_action,
        {
            "summarize_text": "summarize_text",
            "suggest_edits": "suggest_edits",
            "check_grammar": "check_grammar",
            "check_format": "check_format",
        },
    )

    # All paths lead to answer
    graph.add_edge("summarize_text", "answer")
    graph.add_edge("suggest_edits", "answer")
    graph.add_edge("check_grammar", "answer")
    graph.add_edge("check_format", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


# Export the compiled graph
agent_write_graph = create_agent_write()

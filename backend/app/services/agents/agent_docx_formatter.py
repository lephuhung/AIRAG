"""
Agent Docx Formatter
====================

LangGraph agent for checking document formatting and consulting
Vietnamese government document standards via RAG.

State: AgentDocxFormatterState
Nodes:
    - extract_format: Extract formatting from uploaded .docx file
    - analyze_issues: Analyze formatting issues
    - rag_lookup: Look up formatting standards via RAG
    - generate_report: Generate final formatting report

Usage:
    graph = create_agent_docx_formatter()
    async for event in graph.astream_events(initial_state):
        ...
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any, Optional

from langgraph.graph import StateGraph, END

from app.services.llm import get_llm_provider
from app.services.llm.types import LLMMessage
from app.services.agents.docx_formatter_tools import (
    extract_docx_format,
    analyze_format_issues,
    rag_lookup_format_standards,
)

logger = logging.getLogger(__name__)


class AgentDocxFormatterState(TypedDict, total=False):
    """State for the docx formatter agent."""

    messages: list[dict]
    user_id: Optional[int]
    workspace_ids: list[int]

    file_path: str
    file_name: str

    extracted_format: dict[str, Any]
    issues: list[dict]
    standards_contexts: list[str]

    report: str
    error: str | None


async def extract_format_node(
    state: AgentDocxFormatterState,
) -> AgentDocxFormatterState:
    """Extract formatting information from the uploaded .docx file."""
    file_path = state.get("file_path", "")
    file_name = state.get("file_name", "")

    if not file_path:
        return {
            **state,
            "error": "No file provided",
            "report": "Vui lòng cung cấp tệp Word (.docx) để kiểm tra định dạng.",
        }

    logger.info(f"[extract_format_node] Extracting format from: {file_path}")

    try:
        extracted = await extract_docx_format(file_path)

        if extracted.get("error"):
            return {
                **state,
                "error": extracted["error"],
                "extracted_format": extracted,
                "report": f"Không thể đọc tệp {file_name}. Vui lòng đảm bảo đây là tệp Word (.docx) hợp lệ.",
            }

        logger.info(
            f"[extract_format_node] Extracted: margins={extracted.get('margins')}, "
            f"paragraphs={extracted.get('paragraph_count')}, "
            f"tables={extracted.get('table_count')}"
        )

        return {
            **state,
            "extracted_format": extracted,
            "error": None,
        }

    except Exception as e:
        logger.error(f"[extract_format_node] Failed: {e}")
        return {
            **state,
            "error": str(e),
            "report": f"Không thể đọc tệp {file_name}. Vui lòng đảm bảo đây là tệp Word (.docx) hợp lệ.",
        }


async def analyze_issues_node(
    state: AgentDocxFormatterState,
) -> AgentDocxFormatterState:
    """Analyze extracted formatting and identify issues compared to standards."""
    extracted = state.get("extracted_format", {})

    if not extracted or extracted.get("error"):
        return {**state, "issues": [], "error": "No format data to analyze"}

    try:
        issues = analyze_format_issues(extracted)

        logger.info(f"[analyze_issues_node] Found {len(issues)} issues")
        for issue in issues:
            logger.info(f"  - [{issue['severity']}] {issue['type']}: {issue['detail']}")

        return {
            **state,
            "issues": issues,
            "error": None,
        }

    except Exception as e:
        logger.error(f"[analyze_issues_node] Failed: {e}")
        return {**state, "issues": [], "error": str(e)}


async def rag_lookup_node(
    state: AgentDocxFormatterState,
    db: "AsyncSession",
) -> AgentDocxFormatterState:
    """Look up Vietnamese government document formatting standards via RAG."""
    workspace_ids = state.get("workspace_ids", [])
    extracted = state.get("extracted_format", {})

    if not workspace_ids:
        logger.info("[rag_lookup_node] No workspace IDs provided, skipping RAG lookup")
        return {**state, "standards_contexts": []}

    try:
        standards_query = (
            "quy chuẩn trình bày văn bản hành chính Việt Nam căn lề giãn dòng cỡ chữ"
        )

        contexts = await rag_lookup_format_standards(
            query=standards_query,
            workspace_ids=workspace_ids,
            db=db,
            top_k=5,
        )

        logger.info(
            f"[rag_lookup_node] Found {len(contexts)} relevant standards contexts"
        )

        return {
            **state,
            "standards_contexts": contexts,
            "error": None,
        }

    except Exception as e:
        logger.error(f"[rag_lookup_node] Failed: {e}")
        return {**state, "standards_contexts": [], "error": str(e)}


async def generate_report_node(
    state: AgentDocxFormatterState,
) -> AgentDocxFormatterState:
    """Generate comprehensive formatting report using LLM."""
    extracted = state.get("extracted_format", {})
    issues = state.get("issues", [])
    standards_contexts = state.get("standards_contexts", [])
    file_name = state.get("file_name", "tài liệu")

    if not extracted:
        return {
            **state,
            "report": "Không có dữ liệu định dạng để tạo báo cáo.",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Không có dữ liệu định dạng để tạo báo cáo.",
                }
            ],
        }

    try:
        llm = get_llm_provider()

        margins = extracted.get("margins", {})
        line_spacing = extracted.get("line_spacing", [])
        font_samples = extracted.get("font_samples", [])

        # Build prompt
        prompt = f"""Phân tích thông tin định dạng sau và đưa ra báo cáo kiểm tra chi tiết.

## TÊN TỆP: {file_name}

## THÔNG TIN ĐỊNH DẠNG ĐÃ TRÍCH XUẤT:

### 1. Căn lề (cm):
- Trên (top): {margins.get("top", "N/A")} cm
- Dưới (bottom): {margins.get("bottom", "N/A")} cm  
- Trái (left): {margins.get("left", "N/A")} cm
- Phải (right): {margins.get("right", "N/A")} cm
- Chuẩn thông thường: Top 2cm, Bottom 2cm, Left 3cm, Right 2cm

### 2. Cỡ chữ:
"""

        # Add font samples
        font_sizes = [f["font_size"] for f in font_samples if f.get("font_size")]
        if font_sizes:
            from collections import Counter

            size_counts = Counter(font_sizes)
            most_common = size_counts.most_common(3)
            prompt += f"- Cỡ chữ phổ biến: {', '.join([f'{s}pt ({c} lần)' for s, c in most_common])} pt\n"
            prompt += "- Chuẩn: 13pt cho nội dung, 14pt cho tiêu đề\n"

        # Add unique fonts
        unique_fonts = set(f["font_name"] for f in font_samples if f.get("font_name"))
        if unique_fonts:
            prompt += f"- Font chữ: {', '.join(list(unique_fonts)[:5])}\n"

        prompt += f"\n### 3. Số đoạn văn: {extracted.get('paragraph_count', 'N/A')}\n"
        prompt += f"### 4. Số bảng: {extracted.get('table_count', 'N/A')}\n"

        # Add header/footer info
        has_header = extracted.get("has_header", False)
        has_footer = extracted.get("has_footer", False)
        prompt += f"### 5. Đầu trang/Chân trang: {'Có' if has_header else 'Không có'} đầu trang, {'Có' if has_footer else 'Không có'} chân trang\n"

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

        result = await llm.acomplete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=2048,
        )

        report = (
            result
            if isinstance(result, str)
            else getattr(result, "content", str(result))
        )

        return {
            **state,
            "report": report,
            "messages": [{"role": "assistant", "content": report}],
            "error": None,
        }

    except Exception as e:
        logger.error(f"[generate_report_node] Failed: {e}")
        return {
            **state,
            "error": str(e),
            "report": "Không thể tạo báo cáo. Vui lòng thử lại.",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Không thể tạo báo cáo. Vui lòng thử lại.",
                }
            ],
        }


def create_agent_docx_formatter() -> StateGraph:
    """
    Create the docx formatter agent graph.

    Flow:
        extract_format → analyze_issues → rag_lookup → generate_report → END
    """
    graph = StateGraph(AgentDocxFormatterState)

    graph.add_node("extract_format", extract_format_node)
    graph.add_node("analyze_issues", analyze_issues_node)
    graph.add_node("rag_lookup", rag_lookup_node)
    graph.add_node("generate_report", generate_report_node)

    graph.set_entry_point("extract_format")
    graph.add_edge("extract_format", "analyze_issues")
    graph.add_edge("analyze_issues", "rag_lookup")
    graph.add_edge("rag_lookup", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


agent_docx_formatter_graph = create_agent_docx_formatter()

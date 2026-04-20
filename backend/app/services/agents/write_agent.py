"""
Write Agent
===========

Single-file agent handling text processing operations:
- summarize      → summarize provided text
- suggest_edits  → editing suggestions
- grammar_check  → grammar/style checking
- format_check   → Word document format checking

Each operation is a prompt-based LLM call.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agents.models import SupervisorState

logger = logging.getLogger(__name__)

# =============================================================================
# Static Standards Path
# =============================================================================

STANDARD_30_PATH = os.path.join(
    os.path.dirname(__file__), "../../docs/30-ND.md"
)


FALLBACK_STANDARD = """\
30/2020/NĐ-CP QUY ĐỊNH VỀ THỂ THỨC VĂN BẢN:

1. CĂN LỀ: Trên 2cm, Dưới 2cm, Trái 3cm, Phải 2cm
2. CỠ CHỮ: 13pt cho nội dung, 14pt cho tiêu đề
3. FONT: Times New Roman, Arial
4. KHOẢNG CÁCH DÒNG: 1.5 dòng
"""


def _load_30_standard_from_file() -> str:
    """Load 30/2020/NĐ-CP standards directly from static file."""
    try:
        with open(STANDARD_30_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"[_load_30_standard_from_file] Failed to read {STANDARD_30_PATH}: {e}")
        return FALLBACK_STANDARD


async def _get_30_2020_standard(state: SupervisorState) -> str:
    """
    Load 30/2020/NĐ-CP standards from static file.
    No RAG dependency - works even when documents are not indexed.
    """
    return _load_30_standard_from_file()


# =============================================================================
# Write Prompts
# =============================================================================

WRITE_PROMPTS = {
    "summarize": (
        "Hãy tóm tắt văn bản sau bằng tiếng Việt một cách ngắn gọn và súc tích. "
        "Giữ lại các ý chính và thông tin quan trọng.\n\n"
        "Văn bản:\n{text}"
    ),
    "extract_key_points": (
        "Hãy trích xuất các điểm chính từ văn bản sau bằng tiếng Việt. "
        "Liệt kê các ý quan trọng nhất.\n\n"
        "Văn bản:\n{text}"
    ),
    "suggest_edits": (
        "Hãy phân tích và đề xuất chỉnh sửa cho văn bản sau. "
        "Tập trung vào:\n"
        "1. Cải thiện cấu trúc và luồng nội dung\n"
        "2. Làm rõ các câu ý nghĩa mơ hồ\n"
        "3. Đề xuất cách diễn đạt tốt hơn\n"
        "4. Cách diễn đạt phù hợp ngữ cảnh hơn\n\n"
        "Trả lời bằng tiếng Việt.\n\n"
        "Văn bản:\n{text}"
    ),
    "grammar_check": (
        "Hãy kiểm tra ngữ pháp và phong cách viết của văn bản sau. "
        "Chỉ ra các lỗi ngữ pháp, chính tả, và đề xuất cách sửa. "
        "Nếu văn bản tốt, hãy xác nhận điều đó.\n\n"
        "Trả lời bằng tiếng Việt.\n\n"
        "Văn bản:\n{text}"
    ),
}

FORMAT_CHECK_PROMPT = """\
Phân tích thông tin định dạng sau và đưa ra báo cáo kiểm tra chi tiết.

## TÊN TỆP: {file_name}

## THÔNG TIN ĐỊNH DẠNG ĐÃ TRÍCH XUẤT:

### 1. Khổ giấy:
- Chiều rộng: {page_width} cm
- Chiều cao: {page_height} cm

### 2. Căn lề (cm):
- Trên (top): {top} cm
- Dưới (bottom): {bottom} cm
- Trái (left): {left} cm
- Phải (right): {right} cm

### 3. Cỡ chữ:
{font_sizes}

### 4. Font chữ:
{fonts}

### 5. Khoảng cách dòng:
{line_spacing}

### 6. Số đoạn văn: {paragraph_count}
### 7. Số bảng: {table_count}

## TIÊU CHUẨN VĂN BẢN (30/2020/NĐ-CP):
{standard_content}

## YÊU CẦU:
1. So sánh thông tin định dạng đã trích xuất với tiêu chuẩn 30/2020/NĐ-CP
2. Đưa ra đánh giá TỔNG QUAN (đạt/yếu/cần cải thiện)
3. Liệt kê các vấn đề cụ thể theo mức độ nghiêm trọng (nghiêm trọng/thấp)
4. Đề xuất cách SỬA chữa cụ thể cho từng vấn đề
5. Trích dẫn điều khoản cụ thể từ 30/2020/NĐ-CP cho mỗi vấn đề

Trả lời bằng tiếng Việt, rõ ràng và có cấu trúc.
"""


# =============================================================================
# Write Agent Node
# =============================================================================

async def write_agent_node(state: SupervisorState) -> dict:
    """
    Execute write operation based on write_action.

    Flow:
    1. Determine write action (summarize/suggest_edits/grammar_check/format_check)
    2. Get text input (from state.text_input or kg_summaries)
    3. Call appropriate LLM
    4. Return final answer
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage
    from app.services.agent.nodes import strip_thinking_tags
    from app.services.agents.models import AgentType
    from app.services.agent.streaming import push_event

    write_action = state.get("write_action", "summarize")
    text_input = state.get("text_input", "")

    logger.info(f"[write_agent] action={write_action!r}, text_input_len={len(text_input)}")

    # Emit status so frontend shows progress immediately
    await push_event(state, "status", {"step": "generating", "detail": "Đang xử lý văn bản..."})

    # Handle format_check separately
    if write_action == "format_check":
        return await _handle_format_check(state)

    # Guard: need text input
    if not text_input:
        # Try to get from kg_summaries (for summarize intent from RAG)
        kg_summaries = state.get("kg_summaries", [])
        if kg_summaries:
            text_input = kg_summaries[0]
        else:
            return {
                "final_answer": "Vui lòng cung cấp văn bản để xử lý.",
                "next_agent": AgentType.FINISH,
            }

    # Get prompt
    prompt_template = WRITE_PROMPTS.get(write_action, WRITE_PROMPTS["summarize"])
    prompt = prompt_template.format(text=text_input)

    # Call LLM
    provider = get_llm_provider()
    answer_parts = []

    try:
        async for chunk in provider.astream(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=2048,
        ):
            if chunk.type == "text" and chunk.text:
                answer_parts.append(chunk.text)
                # Emit token for realtime SSE streaming
                await push_event(state, "token", chunk.text)
    except Exception as e:
        logger.error(f"[write_agent] LLM call failed: {e}")
        return {
            "final_answer": f"Lỗi xử lý văn bản: {e}",
            "next_agent": AgentType.FINISH,
        }

    final_answer = strip_thinking_tags("".join(answer_parts))

    return {
        "final_answer": final_answer,
        "next_agent": AgentType.FINISH,
        "iterations": state.get("iterations", 0) + 1,
    }


async def _handle_format_check(state: SupervisorState) -> dict:
    """
    Handle Word document format checking with RAG-based evaluation.

    Flow (Option A + B):
    1. Extract format metadata from docx (unchanged)
    2. Call RAG to get standard from 30/2020/NĐ-CP (NEW)
    3. Build comprehensive evaluation prompt with both
    4. LLM evaluates and returns detailed report
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage
    from app.services.agent.nodes import strip_thinking_tags
    from app.services.agents.models import AgentType
    from app.services.agent.streaming import push_event

    format_data = state.get("format_data") or {}
    file_name = state.get("file_name", "tài liệu")

    # Emit status
    await push_event(state, "status", {"step": "generating", "detail": "Đang kiểm tra định dạng..."})

    if not format_data:
        return {
            "final_answer": (
                "Không có dữ liệu định dạng để kiểm tra.\n\n"
                "Vui lòng đính kèm một file Word (.docx) và hỏi lại."
            ),
            "next_agent": AgentType.FINISH,
        }

    # Extract format info
    margins = format_data.get("margins", {})
    page_size = format_data.get("page_size", {})
    font_samples = format_data.get("font_samples", [])
    line_spacing_data = format_data.get("line_spacing", [])

    # Build font sizes info
    from collections import Counter
    raw_font_sizes = [f.get("font_size") for f in font_samples if f.get("font_size")]
    size_counts = Counter(raw_font_sizes)
    most_common_sizes = size_counts.most_common(5)

    # Build font list - deduplicate and limit
    unique_fonts = []
    seen = set()
    for f in font_samples:
        name = f.get("font_name")
        if name and name not in seen:
            seen.add(name)
            unique_fonts.append(name)

    # Analyze line spacing
    spacing_types = {}
    for ls in line_spacing_data:
        ls_type = ls.get("line_spacing_type")
        ls_val = ls.get("line_spacing_value")
        if ls_type and ls_val is not None:
            key = f"{ls_type}_{ls_val}"
            spacing_types[key] = spacing_types.get(key, 0) + 1

    if spacing_types:
        most_common_spacing = max(spacing_types, key=spacing_types.get)
        line_spacing_display = f"{most_common_spacing} (gặp {max(spacing_types.values())} lần)"
    else:
        line_spacing_display = "Không trích xuất được"

    # Step 1: Get standard from file (no RAG dependency)
    logger.info("[_handle_format_check] Loading 30/2020/NĐ-CP standards from file...")
    standard_content = await _get_30_2020_standard(state)
    logger.info(f"[_handle_format_check] Got standard content: {len(standard_content)} chars")

    # Step 2: Build comprehensive evaluation prompt with standard
    prompt = FORMAT_CHECK_PROMPT.format(
        file_name=file_name,
        page_width=page_size.get("width", "N/A"),
        page_height=page_size.get("height", "N/A"),
        top=margins.get("top", "N/A"),
        bottom=margins.get("bottom", "N/A"),
        left=margins.get("left", "N/A"),
        right=margins.get("right", "N/A"),
        font_sizes="\n".join([f"- {s}pt ({c} lần)" for s, c in most_common_sizes]) if most_common_sizes else "- Không trích xuất được",
        fonts=", ".join(unique_fonts[:5]) if unique_fonts else "Không trích xuất được",
        line_spacing=line_spacing_display,
        paragraph_count=format_data.get("paragraph_count", 0),
        table_count=format_data.get("table_count", 0),
        standard_content=standard_content,
    )

    # Call LLM
    provider = get_llm_provider()
    answer_parts = []

    try:
        async for chunk in provider.astream(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=2048,
        ):
            if chunk.type == "text" and chunk.text:
                answer_parts.append(chunk.text)
                # Emit token for realtime SSE streaming
                await push_event(state, "token", chunk.text)
    except Exception as e:
        logger.error(f"[write_agent:format_check] LLM call failed: {e}")
        return {
            "final_answer": f"Lỗi kiểm tra định dạng: {e}",
            "next_agent": AgentType.FINISH,
        }

    final_answer = strip_thinking_tags("".join(answer_parts))

    return {
        "final_answer": final_answer,
        "next_agent": AgentType.FINISH,
        "iterations": state.get("iterations", 0) + 1,
        "format_evaluation_result": {
            "is_valid": "đạt" in final_answer.lower() or "phù hợp" in final_answer.lower(),
            "overall_status": (
                "đạt" if "đạt" in final_answer.lower()
                else "yếu" if "yếu" in final_answer.lower()
                else "cần cải thiện"
            ),
            "standard_reference": "30/2020/NĐ-CP",
        },
    }

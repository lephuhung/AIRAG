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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agents.models import SupervisorState

logger = logging.getLogger(__name__)

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

### 1. Căn lề (cm):
- Trên (top): {top} cm
- Dưới (bottom): {bottom} cm
- Trái (left): {left} cm
- Phải (right): {right} cm

### 2. Cỡ chữ: {font_sizes}
### 3. Font chữ: {fonts}
### 4. Khoảng cách dòng: {line_spacing}
### 5. Số đoạn văn: {paragraph_count}
### 6. Số bảng: {table_count}

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
# RAG-based Format Evaluation (Option A + B)
# =============================================================================

RAG_FORMAT_EVALUATION_PROMPT = """\
Dựa trên thông tin thể thức văn bản sau, hãy đánh giá và so sánh với tiêu chuẩn 30/2020/NĐ-CP.

## THÔNG TIN THỂ THỨC VĂN BẢN CẦN ĐÁNH GIÁ:
{format_summary}

## YÊU CẦU:
1. Tra cứu các quy định về thể thức trong 30/2020/NĐ-CP liên quan đến:
   - Căn lề (margins)
   - Cỡ chữ và font chữ
   - Khoảng cách dòng
   - Cấu trúc văn bản (tiêu đề, đề mục, đoạn văn)
2. So sánh văn bản cần đánh giá với tiêu chuẩn
3. Chỉ ra các điểm sai hoặc không phù hợp
4. Đề xuất cách sửa cụ thể

Trả lời bằng tiếng Việt, có cấu trúc rõ ràng.
"""


async def _get_30_2020_standard(state: SupervisorState) -> str:
    """
    Call RAG to get relevant standards from 30/2020/NĐ-CP for format evaluation.

    This implements the pattern: Write Agent -> RAG Agent -> Evaluate
    (Option B: direct edge for format_check flow)
    """
    from app.services.hrag_service import NexusRAGService
    from app.services.agent.streaming import get_current_db

    workspace_ids = state.get("workspace_ids", [])
    if not workspace_ids:
        return ""

    try:
        db = get_current_db()
        rag_service = NexusRAGService()

        # Query to get relevant sections from 30/2020/NĐ-CP about format standards
        query = (
            "thể thức văn bản hành chính 30/2020/NĐ-CP căn lề cỡ chữ khoảng cách dòng "
            "cấu trúc tiêu đề đề mục quy định trình bày"
        )

        # Use the same deep retrieval as the main RAG pipeline
        result = await rag_service.query_deep(
            query=query,
            workspace_ids=workspace_ids,
            top_k=3,
            rerank=True,
            db=db,
        )

        # Extract relevant text from sources
        standard_parts = []
        for src in result.get("sources", []):
            content = src.get("content", "")
            if content:
                standard_parts.append(content)

        if standard_parts:
            standard_content = "\n\n---\n\n".join(standard_parts[:3])  # Limit to top 3
            logger.info(f"[_get_30_2020_standard] Retrieved {len(standard_parts)} standard sections")
            return standard_content

        # Fallback: return generic 30/2020/NĐ-CP standards if RAG returns nothing
        return _get_generic_30_2020_standards()

    except Exception as e:
        logger.warning(f"[_get_30_2020_standard] RAG call failed: {e}, using generic standards")
        return _get_generic_30_2020_standards()


def _get_generic_30_2020_standards() -> str:
    """Fallback generic standards from 30/2020/NĐ-CP when RAG is unavailable."""
    return """\
30/2020/NĐ-CP QUY ĐỊNH VỀ THỂ THỨC VĂN BẢN:

1. CĂN LỀ (theo Điều 12):
   - Trang giấy A4: Trên 2cm, Dưới 2cm, Trái 3cm, Phải 2cm
   - Văn bản khổ A5: Trên 2cm, Dưới 2cm, Trái 2.5cm, Phải 2cm

2. CỠ CHỮ VÀ FONT (theo Điều 11):
   - Khuyến khích dùng font Times New Roman, Arial, Courier
   - Cỡ chữ: 13pt hoặc 14pt cho văn bản thông thường
   - Tiêu đề có thể dùng 16pt-20pt

3. KHOẢNG CÁCH DÒNG (theo Điều 13):
   - Dùng cách 1 dòng rưỡi (1.5 line spacing) cho văn bản thông thường
   - Có thể dùng cách đơn (single spacing) cho phụ lục, trích dẫn

4. CẤU TRÚC VĂN BẢN:
   - Tiêu đề văn bản: IN HOA, không gạch chân, căn giữa
   - Đề mục: số thứ tự La Mã, chữ cái, số Ả Rập
   - Danh sách: dùng dấu gạch đầu dòng (-) hoặc số thứ tự
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
    font_samples = format_data.get("font_samples", [])

    # Build font info
    font_sizes = format_data.get("font_sizes", [])
    unique_fonts = set(f.get("font_name") for f in font_samples if f.get("font_name"))

    from collections import Counter
    size_counts = Counter(font_sizes)
    most_common_sizes = size_counts.most_common(3)

    line_spacing_data = format_data.get("line_spacing", [])
    has_line_spacing = any(ls.get("line_spacing_value") for ls in line_spacing_data)

    # Step 1: Get standard from RAG (Option B - direct edge pattern)
    logger.info("[_handle_format_check] Calling RAG for 30/2020/NĐ-CP standards...")
    standard_content = await _get_30_2020_standard(state)
    logger.info(f"[_handle_format_check] Got standard content: {len(standard_content)} chars")

    # Step 2: Build comprehensive evaluation prompt with standard
    prompt = FORMAT_CHECK_PROMPT.format(
        file_name=file_name,
        top=margins.get("top", "N/A"),
        bottom=margins.get("bottom", "N/A"),
        left=margins.get("left", "N/A"),
        right=margins.get("right", "N/A"),
        font_sizes=", ".join([f"{s}pt ({c} lần)" for s, c in most_common_sizes]) if most_common_sizes else "Không trích xuất được",
        fonts=", ".join(list(unique_fonts)[:5]) if unique_fonts else "Không trích xuất được",
        line_spacing="Có" if has_line_spacing else "Không trích xuất được (mặc định nên dùng 1.5 lines)",
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

"""
Write Agent Prompts
====================

Text processing prompts for the Write Agent:
- summarize      → summarize provided text
- suggest_edits  → editing suggestions
- grammar_check  → grammar/style checking
- format_check   → Word document format checking (30/2020/NĐ-CP)

Location: app/services/agents/write_agent.py (lines 34-136)

References:
  - WRITE_PROMPTS: lines 66-94
  - FORMAT_CHECK_PROMPT: lines 96-136
  - FALLBACK_STANDARD: lines 34-41
  - _load_30_standard_from_file: lines 44-51
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# =============================================================================
# Static Standards Path
# =============================================================================

# Path: app/prompts/agents/write_agent_prompt.py -> app/prompts/agents -> app/prompts -> app -> backend/
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
STANDARD_30_PATH = os.path.join(_BACKEND_ROOT, "docs", "30-ND.md")


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
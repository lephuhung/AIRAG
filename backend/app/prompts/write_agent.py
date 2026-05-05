"""
Write Agent Prompts
====================
Single-file agent handling text processing operations.

Referenced by: backend/app/services/agents/write_agent.py

Operations:
  - summarize      → summarize provided text
  - suggest_edits  → editing suggestions
  - grammar_check  → grammar/style checking
  - format_check   → Word document format checking

See: prompts/write_agent.md
"""

import os

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
        import logging
        logger = logging.getLogger(__name__)
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

{themed_content}

## NỘI DUNG ĐỊNH DẠNG:
{format_data}

## TIÊU CHUẨN THAM CHIẾU (30/2020/NĐ-CP):
{standard_content}

Hãy kiểm tra và đưa ra báo cáo chi tiết theo format sau:

### 1. CĂN LỀ (Margins)
- Trên: {margin_top} (chuẩn: 2cm)
- Dưới: {margin_bottom} (chuẩn: 2cm)
- Trái: {margin_left} (chuẩn: 3cm)
- Phải: {margin_right} (chuẩn: 2cm)
**Đánh giá:** [ĐẠT / KHÔNG ĐẠT] - Mô tả cụ thể

### 2. CỠ CHỮ (Font Size)
- Cỡ chữ hiện tại: {font_size}pt
- Cỡ chữ tiêu đề: {heading_font_size}pt
**Đánh giá:** [ĐẠT / KHÔNG ĐẠT]

### 3. FONT CHỮ
- Font hiện tại: {font_name}
**Đánh giá:** [ĐẠT / KHÔNG ĐẠT] - Font chuẩn: Times New Roman, Arial

### 4. KHOẢNG CÁCH DÒNG
- Khoảng cách dòng hiện tại: {line_spacing}
**Đánh giá:** [ĐẠT / KHÔNG ĐẠT] - Chuẩn: 1.5 dòng

### 5. ĐÁNH GIÁ TỔNG HỢP
**Tổng thể:** [ĐẠT / CẦN CẢI THIỆN / KHÔNG ĐẠT]
**Số lỗi:** {error_count}
**Mô tả:** {overall_assessment}
"""
"""
Disambiguation System Prompt
============================

Prompt for LLM-based abbreviation disambiguation when a written abbreviation
has multiple meanings in the abbreviation DB.

Used by: app/services/agents/supervisor.py :: _disambiguate_multi_meaning_abbrs

Location: imported via from app.prompts.agents.disambiguation_prompt import
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "Bạn là trợ lý xác định nghĩa từ viết tắt tiếng Việt (chuyên văn bản hành chính, pháp luật). "
    "Với mỗi từ viết tắt có nhiều nghĩa, hãy chọn nghĩa PHÙ HỢP NHẤT với ngữ cảnh câu hỏi.\n\n"
    "CÁCH CHỌN:\n"
    "1. Đọc các DANH TỪ / CHỦ ĐỀ đi kèm ngay quanh từ viết tắt (vd: 'tài liệu', 'văn bản', "
    "'giao nhận', 'mật', 'bảo vệ', 'công tác', 'lực lượng', 'phòng', 'cục'...). Chúng cho biết "
    "lĩnh vực đang nói tới.\n"
    "2. So khớp lĩnh vực đó với phần mô tả (— …) của TỪNG nghĩa; chọn nghĩa có mô tả khớp nhất.\n"
    "3. Ưu tiên nghĩa hợp lý về NGHĨA (một 'tài liệu' có thể là 'bí mật nhà nước' nhưng KHÔNG thể "
    "là 'bộ máy nhà nước'; hãy loại nghĩa vô lý về mặt ngữ pháp/logic).\n"
    "4. CHỈ trả 'low' khi câu thực sự không có manh mối nào để phân biệt; nếu ngữ cảnh nghiêng rõ "
    "về một nghĩa thì trả 'high'.\n\n"
    "QUAN TRỌNG: trường \"chosen\" phải là dạng đầy đủ (full_form) CHÍNH XÁC như đã liệt kê — "
    "chỉ chép cụm nghĩa, TUYỆT ĐỐI không kèm phần mô tả sau dấu '—' và không thêm chữ nào khác.\n\n"
    "CHỈ xuất JSON hợp lệ, không kèm giải thích ngoài JSON."
)

USER_PROMPT_TEMPLATE = (
    'Từ viết tắt "{abbr}" có các nghĩa sau:\n{meanings_text}\n\n'
    'Câu hỏi của user: "{user_message}"\n\n'
    'Dựa vào ngữ cảnh câu hỏi, chọn nghĩa phù hợp nhất.\n'
    'Nếu ngữ cảnh không đủ rõ để chọn, trả về confidence: "low".\n\n'
    'Output JSON: {{"chosen": "<full_form>", "confidence": "high" or "low", "reasoning": "<1 sentence>"}}'
)


def build_disambiguation_prompt(
    abbr: str,
    meanings: list[dict],
    user_message: str,
) -> tuple[str, str]:
    """Build the disambiguation prompt pair.

    Args:
        abbr: The abbreviation to disambiguate (e.g. "ANM")
        meanings: List of {full_form, description} dicts from DB
        user_message: The original user query

    Returns:
        (system_prompt, user_prompt) tuple
    """
    meanings_text = "\n".join(
        f"  {i+1}. {m['full_form']}" + (f" — {m['description']}" if m.get('description') else "")
        for i, m in enumerate(meanings)
    )
    return (
        SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE.format(
            abbr=abbr,
            meanings_text=meanings_text,
            user_message=user_message,
        ),
    )


# =============================================================================
# Batch Disambiguation (Optimized — single LLM call for all abbreviations)
# =============================================================================

BATCH_USER_PROMPT_TEMPLATE = (
    'Câu hỏi của user: "{user_message}"\n\n'
    'Các từ viết tắt cần xác định nghĩa:\n{abbreviations_block}\n\n'
    'Dựa vào ngữ cảnh câu hỏi, chọn nghĩa phù hợp nhất cho TỪNG từ viết tắt.\n'
    'Nếu ngữ cảnh không đủ rõ để chọn, trả về confidence: "low" cho từ đó.\n\n'
    'Output JSON: {{"results": [{{"abbr": "<viết_tắt>", "chosen": "<full_form>", '
    '"confidence": "high" or "low", "reasoning": "<1 sentence>"}}, ...]}}'
)


def build_batch_disambiguation_prompt(
    multi_meaning_map: dict[str, list],
    user_message: str,
) -> tuple[str, str]:
    """Build a SINGLE disambiguation prompt for ALL abbreviations at once.

    Instead of calling LLM once per abbreviation (N sequential calls),
    this batches everything into one prompt (1 call total).

    Args:
        multi_meaning_map: {abbr: [{full_form, description}, ...], ...}
        user_message: The original user query

    Returns:
        (system_prompt, user_prompt) tuple
    """
    blocks = []
    for abbr, meanings in multi_meaning_map.items():
        meanings_text = "\n".join(
            f"    {i+1}. {m['full_form']}" + (f" — {m['description']}" if m.get('description') else "")
            for i, m in enumerate(meanings)
        )
        blocks.append(f'  "{abbr}":\n{meanings_text}')

    abbreviations_block = "\n\n".join(blocks)

    return (
        SYSTEM_PROMPT,
        BATCH_USER_PROMPT_TEMPLATE.format(
            user_message=user_message,
            abbreviations_block=abbreviations_block,
        ),
    )

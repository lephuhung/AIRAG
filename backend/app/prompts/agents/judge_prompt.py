"""
Answer Judge Prompt (ReAct group)
=================================

Drives the LLM-as-judge gate inside ``react_executor_node``. After the model
produces a DRAFT answer (stops calling tools), the judge scores it against the
collected sources and the query plan, and decides whether to finalise or send
the loop back to gather more.

The judge NEVER writes the user-facing answer — it only returns a verdict JSON.
"""

from __future__ import annotations

JUDGE_SYSTEM_PROMPT = """\
Bạn là GIÁM KHẢO chất lượng cho một trợ lý hỏi-đáp văn bản pháp luật Việt Nam. \
Nhiệm vụ của bạn KHÔNG phải viết câu trả lời, mà là CHẤM bản nháp câu trả lời dựa \
trên NGUỒN đã thu thập và KẾ HOẠCH (nếu có), rồi quyết định: chấp nhận hay yêu cầu tra thêm.

Chấm theo các tiêu chí:
1. CĂN CỨ: Mọi khẳng định pháp lý trong bản nháp có dựa trên nội dung NGUỒN không? \
Có chỗ nào bịa/suy đoán ngoài nguồn không?
2. TRÍCH DẪN: Các mã trích dẫn [xxxx] trong bản nháp có KHỚP với mã có thật trong NGUỒN \
không? (bịa mã = lỗi nặng).
3. ĐẦY ĐỦ: Nếu có KẾ HOẠCH, bản nháp đã trả lời HẾT mọi mục chưa? Nếu không có kế hoạch, \
bản nháp đã trả lời đúng trọng tâm câu hỏi chưa?
4. Nếu NGUỒN thực sự rỗng/không liên quan và bản nháp đã nói rõ "không tìm thấy" một cách \
trung thực → đó là ĐẠT (không bắt tra thêm vô ích).

Quyết định:
- "pass": bản nháp đủ tốt để gửi người dùng.
- "revise": còn thiếu căn cứ / thiếu mục kế hoạch / có dấu hiệu bịa → cần gọi thêm công cụ. \
Khi "revise", nêu CỤ THỂ trong "feedback" cần tra thêm gì (từ khoá, văn bản, khía cạnh) để \
lần sau khắc phục được.

CHỈ trả về JSON hợp lệ, không kèm giải thích ngoài JSON:
{"verdict": "pass" | "revise", "score": <số 0..1>, "missing": ["..."], "feedback": "..."}

BẮT BUỘC NGẮN GỌN (JSON dài sẽ bị cắt cụt và trở nên VÔ DỤNG):
- "missing": TỐI ĐA 3 mục, mỗi mục ≤ 15 từ (chỉ nêu CẦN TRA GÌ, không phân tích dài).
- "feedback": TỐI ĐA 2 câu."""


def build_judge_user_prompt(
    question: str,
    draft: str,
    sources_digest: str,
    plan_lines: list[str] | None = None,
) -> str:
    """Assemble the judge's user turn from the question, plan, sources and draft."""
    plan_block = ""
    if plan_lines:
        plan_block = "KẾ HOẠCH (các phần cần trả lời):\n" + "\n".join(
            f"- {p}" for p in plan_lines if p
        ) + "\n\n"
    src = sources_digest.strip() or "(không có nguồn nào được thu thập)"
    return (
        f"CÂU HỎI NGƯỜI DÙNG:\n{question.strip()}\n\n"
        f"{plan_block}"
        f"NGUỒN ĐÃ THU THẬP:\n{src}\n\n"
        f"BẢN NHÁP CÂU TRẢ LỜI:\n{draft.strip() or '(trống)'}\n\n"
        f"Hãy chấm và trả về JSON verdict."
    )

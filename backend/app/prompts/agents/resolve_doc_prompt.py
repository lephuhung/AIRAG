"""
Resolve Doc LLM Extraction Prompt
==================================

System prompt for LLM-based document reference extraction when regex fails.

Used by: app/services/agents/resolve_doc_agent.py :: _extract_by_llm()
Context: Called ONLY when regex + DB query return 0 results (fallback strategy).
The memory agent (gemma-4-E4B) is used for this extraction to minimize latency.

Location: app/prompts/agents/resolve_doc_prompt.py
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "Bạn là chuyên gia tìm kiếm văn bản pháp luật Việt Nam.\n"
    "Trích xuất metadata để search database. Trả về JSON duy nhất.\n\n"
    "DATABASE SCHEMA (bảng documents):\n"
    "  document_number  : Số ký hiệu, vd: 24/2018/QH14, 15/TT-BCA\n"
    "  document_title   : Tiêu đề, vd: Luật An ninh mạng\n"
    "  issuing_agency   : Cơ quan, vd: Bộ Công an, Chính phủ\n"
    "  published_date   : Năm/ngày, vd: 2018, 2026-03-20\n"
    "  document_type    : luat|bo_luat|nghi_dinh|thong_tu|quyet_dinh|nghi_quyet|phap_lenh|chi_thi|thong_tu_lien_tich\n\n"
    "QUY TẮC SỐ VĂN BẢN VIỆT NAM:\n"
    "  Luật (Quốc hội)      : [số]/[năm]/QH15  vd: 24/2018/QH14, 129/2025/QH15\n"
    "  Nghị định (CP)       : [số]/[năm]/NĐ-CP vd: 83/2026/NĐ-CP\n"
    "  Thông tư Bộ X        : [số]/[năm]/TT-[MÃ] vd: 15/2026/TT-BCA\n"
    "  Quyết định UBND      : [số]/[năm]/QĐ-UBND\n"
    "  Nghị quyết HĐND      : [số]/[năm]/NQ-HĐND\n"
    "  Thông tư liên tịch   : [số]/[năm]/TTLT-[BỘ1]-[BỘ2]\n\n"
    "MÃ CƠ QUAN: BCA=Công an|BTC=Tài chính|BCT=Công Thương|BTP=Tư pháp\n"
    "            BYT=Y tế|BNV=Nội vụ|BGDĐT=Giáo dục|BXD=Xây dựng\n"
    "            BGTVT=Giao thông|NHNN=Ngân hàng NN|CP=Chính phủ|TTg=Thủ tướng\n\n"
    "QUAN TRỌNG: nếu câu hỏi ĐÃ chứa số ký hiệu ĐẦY ĐỦ (vd '100/2019/NĐ-CP', "
    "'12/2024/NQ-HĐND', '24/2018/QH14') → CHÉP NGUYÊN VĂN vào document_number, KHÔNG "
    "để trống. doc_number_candidates chỉ dùng khi phải SUY ĐOÁN số (câu hỏi chỉ nhớ "
    "số lẻ hoặc tên, không có số ký hiệu đầy đủ).\n"
)

EXAMPLES = (
    "Ví dụ 1 - nhớ số: \"Thông tư 15 của Bộ Công an\"\n"
    '{"doc_type_slug":"thong_tu","document_number":"15/TT-BCA",'
    '"doc_number_candidates":["15/TT-BCA","15/2025/TT-BCA","15/2026/TT-BCA"],'
    '"title_keywords":[],"issuing_agency_text":"Bộ Công an","year":"","section_reference":""}\n'
    "Ví dụ 2 - nhớ tên: \"Luật An ninh mạng 2018\"\n"
    '{"doc_type_slug":"luat","document_number":"24/2018/QH14",'
    '"doc_number_candidates":["24/2018/QH14"],'
    '"title_keywords":["an ninh","mạng"],"issuing_agency_text":"Quốc hội","year":"2018","section_reference":""}\n'
    "Ví dụ 3 - nhớ nội dung: \"Nghị định về xử phạt vi phạm giao thông\"\n"
    '{"doc_type_slug":"nghi_dinh","document_number":"","doc_number_candidates":[],'
    '"title_keywords":["xử phạt","vi phạm","giao thông"],"issuing_agency_text":"Chính phủ","year":"","section_reference":""}\n'
    "Ví dụ 4 - có SỐ ĐẦY ĐỦ (chép nguyên văn, không để trống): \"Nghị định 100/2019/NĐ-CP về xử phạt giao thông\"\n"
    '{"doc_type_slug":"nghi_dinh","document_number":"100/2019/NĐ-CP",'
    '"doc_number_candidates":["100/2019/NĐ-CP"],'
    '"title_keywords":["xử phạt","giao thông"],"issuing_agency_text":"Chính phủ","year":"2019","section_reference":""}'
)


def build_extract_prompt(reference: str) -> str:
    """Build the LLM extraction prompt for document reference.

    Args:
        reference: The user's document reference string (e.g. "Luật An ninh mạng 2018")

    Returns:
        Single prompt string combining system + reference + examples
    """
    return (
        SYSTEM_PROMPT
        + f'Câu hỏi: "{reference}"\n\n'
        + "Trả về JSON (chỉ JSON):\n"
        + '{"doc_type_slug":"","document_number":"","doc_number_candidates":[],'
        + '"title_keywords":[],"issuing_agency_text":"","year":"","section_reference":""}\n\n'
        + EXAMPLES
    )

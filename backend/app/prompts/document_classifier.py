"""
Document Type Classifier Prompts
==================================
Classify Vietnamese administrative/legal documents from markdown content.

Referenced by: backend/app/services/document_type_classifier.py

The classifier uses an LLM (gemma-4-E4B via vLLM / memory agent) to extract:
  - slug (document type from database)
  - document_number (official document reference number)
  - location (Province or city of issuance)
  - issuing_agency (Direct issuing body)
  - parent_agency (Parent organization)
  - published_date (Date of publication)
  - document_title (Document title/name)

See: prompts/document_classifier.md
"""

from typing import List, Dict


def _build_llm_system_prompt(doc_types: List[Dict]) -> str:
    """Build the LLM system prompt dynamically using the given doc type list."""
    types_list = "\n".join(
        f"  - {d['slug']}: {d['name']} — {d['description']}" for d in doc_types
    )
    return (
        "Bạn là chuyên gia phân tích siêu dữ liệu văn bản hành chính Việt Nam.\n"
        "Nhiệm vụ: Đọc phần đầu của văn bản (markdown hoặc text OCR trang đầu) và bóc tách các trường thông tin tiêu chuẩn của Header.\n"
        "Trả về JSON thuần với đúng 7 trường sau (không dùng markdown):\n\n"
        "1. \"slug\": loại văn bản (chọn từ danh sách bên dưới, hoặc \"unknown\")\n"
        "2. \"document_number\": số hiệu chính thức, thường bắt đầu bằng Luật số:, số: (ví dụ: \"13/2023/NĐ-CP\", \"29/2018/QH14\",\"23/BC-VPUB\"). Bỏ qua chữ 'Số:', 'Luật số:'. Thường nằm bên dưới parent_agency\n"
        "3. \"document_title\": Tên/Tiêu đề văn bản, thường nằm ngay dưới số ký hiệu (VD: \"Luật Bảo vệ Bí mật nhà nước\", \"Kế hoạch triển khai thực hiện\", \"Giấy mời tham gia\").\n"
        "4. \"parent_agency\": Tên cơ quan chủ quản / cấp trên, thường ở góc TRÊN CÙNG BÊN TRÁI, có trường hợp nằm trên 2 dòng (VD: \"UBND TỈNH HÀ TĨNH\", \"Ủy ban nhân dân \n Tỉnh Hà Tĩnh\").\n"
        "5. \"issuing_agency\": Tên đơn vị ban hành trực tiếp, thường nằm ngay dưới parent_agency (VD: \"VĂN PHÒNG\").\n"
        "6. \"location\": Địa danh ban hành ở góc TRÊN CÙNG BÊN PHẢI (VD: \"Hà Tĩnh\", \"Hà Nội\").\n"
        "7. \"published_date\": Ngày tháng năm ban hành (VD: \"15/01/2026\").\n\n"
        "Các slug hợp lệ:\n"
        + types_list
        + "\n\n"
        "Quy tắc:\n"
        "- Chỉ trả về duy nhất chuỗi JSON, không giải thích.\n"
        "- Nếu giá trị nào không có, đặt là `null`.\n"
        "- Mọi text nên giữ nguyên case gốc nếu được hoặc chuẩn hoá Title Case.\n"
        "Ví dụ đầu ra:\n"
        "{\"slug\": \"luat\", \"document_number\": \"13/2024/QH15\", \"document_title\": \"Luật Bảo vệ Bí mật nhà nước\", \"parent_agency\": \"QUỐC HỘI\", \"issuing_agency\": \"VP QUỐC HỘI\", \"location\": \"Hà Nội\", \"published_date\": \"15/06/2024\"}"
    )
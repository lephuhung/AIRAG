"""
Legal Knowledge Graph Prompts
==============================

Vietnamese legal/administrative document extraction prompts for LegalKGService.

Two prompt variants:
  - LEGAL_KG_SYSTEM_PROMPT + LEGAL_KG_USER_PROMPT:
      General legal documents (decrees, circulars, decisions with tasks/org assignments).
  - PERSON_EXTRACT_SYSTEM_PROMPT + PERSON_EXTRACT_USER_PROMPT:
      Personnel decisions (appointment, transfer, discipline, reward, retirement).
  - PREAMBLE_SYSTEM_PROMPT + PREAMBLE_USER_PROMPT:
      Extracts CAN_CU (legal bases) from the document preamble block.
"""

# ---------------------------------------------------------------------------
# General Legal Document Extraction
# ---------------------------------------------------------------------------

LEGAL_KG_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản hành chính/pháp luật Việt Nam.
Nhiệm vụ của bạn là trích xuất các thực thể (entities) và mối quan hệ (relations) từ một Điều/Khoản của văn bản được cung cấp.

## Các loại thực thể được phép (entity types):
- Article: Điều, Khoản, Điểm của văn bản hiện tại. Ví dụ: "Điều 5", "Khoản 2 Điều 3"
- Document: Văn bản pháp luật được viện dẫn. Ví dụ: "Nghị định 123/2024/NĐ-CP"
- Organization: Cơ quan, tổ chức. PHẢI viết tên đầy đủ dựa vào thông tin văn bản (document_meta). KHÔNG dùng tên tắt. Ví dụ: "UBND Tỉnh Nghệ An" (không dùng "UBND tỉnh")
- Person: Cá nhân. PHẢI dùng Composite Key theo quy tắc ưu tiên (xem bên dưới).
- Task: Nhiệm vụ, công việc cụ thể được giao. Ví dụ: "lập kế hoạch thanh tra hàng năm"

## Quy tắc Composite Key cho Person (theo thứ tự ưu tiên):
1. "[Họ Tên] (DD/MM/YYYY)" — nếu có ngày sinh
2. "[Họ Tên] (Số CCCD)" — nếu có số CCCD/định danh cá nhân
3. "[Họ Tên] ([Đơn vị công tác rõ nhất])" — ví dụ: "Nguyễn Văn A (Sở Tài chính Nghệ An)"
4. "[Họ Tên] (không xác định)" — nếu không có thông tin định danh nào

## Quy tắc Canonicalization cho Organization:
- Luôn dùng tên đầy đủ. Ví dụ: "UBND Tỉnh Nghệ An", không dùng "UBND tỉnh" hay "UBND"
- Sử dụng document_meta (thông tin văn bản) để suy diễn tên đầy đủ khi văn bản dùng tên tắt

## Các loại quan hệ được phép (PHẢI dùng chính xác tên sau):
- CAN_CU: Văn bản hiện tại căn cứ vào/dựa trên văn bản pháp lý khác. Source: Document → Target: Document
- VIEN_DAN: Điều khoản viện dẫn/tham chiếu một quy định khác. Source: Article → Target: Document/Article
- SUA_DOI: Văn bản sửa đổi, bổ sung văn bản khác. Source: Document → Target: Document
- CHU_TRI: Đơn vị, cơ quan, cá nhân chủ trì thực hiện. Source: Task/Article → Target: Organization/Person
- PHOI_HOP: Đơn vị, cơ quan, cá nhân phối hợp thực hiện. Source: Task/Article → Target: Organization/Person
- CHIU_TRACH_NHIEM: Đơn vị chịu trách nhiệm thi hành hoặc giám sát. Source: Task/Article → Target: Organization/Person
- PART_OF: Điều/Khoản thuộc cấu trúc của văn bản. Source: Article → Target: Document
- REFERENCES: Điều/Khoản tham chiếu chung đến văn bản/điều khác. Source: Article → Target: Document/Article
- KY: Người ký ban hành văn bản. Source: Document → Target: Person

## QUY TẮC NGHIÊM NGẶT:
1. CHỈ trả về JSON hợp lệ, không có markdown, không có giải thích thêm.
2. KHÔNG được tạo ra bất kỳ loại quan hệ nào ngoài danh sách trên.
3. KHÔNG được tạo entity type ngoài danh sách trên.
4. XỬ LÝ TỰ THAM CHIẾU: TUYỆT ĐỐI KHÔNG trích xuất các cụm từ "quy định này", "quyết định này", "văn bản này" làm thực thể độc lập. Khi gặp câu "Điều X của quy định/quyết định này", BỎ QUA cụm từ chỉ văn bản, CHỈ lấy "Điều X" (loại Article). Nếu văn bản nói "Sở này", "cơ quan này", tìm ngữ cảnh trước đó để ghi TÊN ĐẦY ĐỦ.
5. Nếu không trích xuất được gì, trả về: {"entities": [], "relations": []}
"""

LEGAL_KG_USER_PROMPT = """Thông tin văn bản (document_meta):
{document_meta}

Nội dung cần phân tích:
{article_text}

Hãy trích xuất entities và relations theo đúng schema đã quy định.
Trả về JSON có dạng:
{{
  "entities": [
    {{"name": "...", "type": "Article|Document|Organization|Person|Task", "description": "..."}}
  ],
  "relations": [
    {{"source": "...", "relation": "CAN_CU|VIEN_DAN|SUA_DOI|CHU_TRI|PHOI_HOP|CHIU_TRACH_NHIEM|PART_OF|REFERENCES|KY", "target": "...", "description": "..."}}
  ]
}}"""

# ---------------------------------------------------------------------------
# Personnel Decision Extraction (BO_NHIEM, DIEU_DONG, KHEN_THUONG, etc.)
# ---------------------------------------------------------------------------

PERSON_EXTRACT_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích quyết định nhân sự trong văn bản hành chính Việt Nam.
Nhiệm vụ: Trích xuất thông tin cá nhân và quyết định liên quan từ văn bản hoặc điều khoản được cung cấp.

## Quy tắc Composite Key cho Person (PHẢI tuân thủ thứ tự ưu tiên):
1. "[Họ Tên] (DD/MM/YYYY)" — nếu có ngày sinh (chuẩn hóa về DD/MM/YYYY)
2. "[Họ Tên] (Số CCCD)" — nếu có số CCCD/định danh cá nhân
3. "[Họ Tên] ([Đơn vị công tác rõ nhất])" — ví dụ: "Nguyễn Văn A (Sở Tài chính Nghệ An)"
4. "[Họ Tên] (không xác định)" — nếu không có thông tin định danh nào

## Loại quan hệ Person:
- BO_NHIEM: Bổ nhiệm vào chức vụ mới
- MIEN_NHIEM: Miễn nhiệm khỏi chức vụ
- DIEU_DONG: Điều chuyển sang đơn vị khác
- NGHI_HUU: Nghỉ hưu theo chế độ
- KHEN_THUONG: Khen thưởng (bằng khen, huân chương...)
- KY_LUAT: Kỷ luật (cảnh cáo, khiển trách...)
- PHE_DUYET: Phê duyệt hồ sơ/đề án liên quan đến cá nhân
- LIEN_QUAN: Đề cập đến cá nhân trong điều khoản
- KY: Người ký ban hành văn bản

## Thuộc tính edge Person (chỉ điền nếu văn bản có thông tin):
ho_ten, ngay_sinh (DD/MM/YYYY), gioi_tinh, dan_toc, que_quan, noi_o,
chuc_vu_cu, chuc_vu_moi, don_vi_cu, don_vi_moi, ngach_luong,
trinh_do_cm, trinh_do_ct, dang_vien (true/false), so_the_dang,
hinh_thuc, ly_do, ngay_hieu_luc (DD/MM/YYYY), description, document_id, article_ref

## QUY TẮC NGHIÊM NGẶT:
1. CHỈ trả về JSON hợp lệ, không có markdown, không có giải thích thêm.
2. Trường "description" trên edge phải có (bắt buộc).
3. "person_props" là dict tùy chọn — chỉ điền các field thực sự xuất hiện trong văn bản.
4. XỬ LÝ TỰ THAM CHIẾU: TUYỆT ĐỐI KHÔNG trích xuất các cụm từ "quyết định này", "văn bản này" làm thực thể độc lập. Khi gặp "Điều X của quyết định này", CHỈ trích xuất "Điều X".
5. Nếu không trích xuất được gì, trả về: {"entities": [], "relations": []}
"""

PERSON_EXTRACT_USER_PROMPT = """Thông tin văn bản (document_meta):
{document_meta}

Nội dung cần phân tích:
{article_text}

Trả về JSON theo dạng:
{{
  "entities": [
    {{"name": "[Họ Tên] (Composite Key)", "type": "Person", "description": "..."}}
  ],
  "relations": [
    {{
      "source": "Tên văn bản/điều khoản",
      "relation": "BO_NHIEM|MIEN_NHIEM|DIEU_DONG|NGHI_HUU|KHEN_THUONG|KY_LUAT|PHE_DUYET|LIEN_QUAN|KY",
      "target": "[Họ Tên] (Composite Key)",
      "description": "Mô tả tổng hợp bắt buộc",
      "person_props": {{
        "ngay_sinh": "DD/MM/YYYY",
        "chuc_vu_moi": "..."
      }}
    }}
  ]
}}"""

# ---------------------------------------------------------------------------
# Preamble / CAN_CU Extraction
# ---------------------------------------------------------------------------

PREAMBLE_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam.
Nhiệm vụ: Trích xuất danh sách căn cứ pháp lý (CAN_CU) từ phần đầu văn bản ("căn cứ vào...", "theo...", v.v.).
Mỗi căn cứ là một văn bản pháp lý khác mà văn bản hiện tại dựa trên.

Trả về JSON hợp lệ duy nhất, không có markdown:
{{"can_cu_list": ["Tên văn bản 1", "Tên văn bản 2", ...]}}

Nếu không tìm thấy căn cứ nào: {{"can_cu_list": []}}"""

PREAMBLE_USER_PROMPT = """Phần đầu văn bản:
{preamble_text}

Tên văn bản hiện tại: {document_name}

Trích xuất tất cả các căn cứ pháp lý (văn bản được viện dẫn trong phần căn cứ)."""

# ---------------------------------------------------------------------------
# Keyword triggers for personnel decision detection
# ---------------------------------------------------------------------------

PERSON_DOCUMENT_TRIGGERS = [
    "bổ nhiệm", "bổ nhiêm", "miễn nhiệm", "điều động", "điều chuyển",
    "nghỉ hưu", "khen thưởng", "kỷ luật", "kỉ luật", "phê duyệt danh sách",
    "tiếp nhận và bổ nhiệm", "hưu trí", "thôi việc",
]

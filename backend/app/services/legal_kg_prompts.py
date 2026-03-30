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
- Organization: Cơ quan, tổ chức cụ thể, không chung chung. PHẢI viết tên đầy đủ dựa vào thông tin văn bản (document_meta). KHÔNG dùng tên tắt. Ví dụ: "UBND Tỉnh Nghệ An" (không dùng "UBND tỉnh"). Các tổ chức chung chung như "Người có trách nhiệm", "Cơ quan có thẩm quyền", "Cơ quan, tổ chức có liên quan" không được coi là thực thể.
- Person: Cá nhân. PHẢI dùng Composite Key theo quy tắc ưu tiên (xem bên dưới).
- Task: Nhiệm vụ, công việc cụ thể được giao. Ví dụ: "lập kế hoạch thanh tra hàng năm"

## Quy tắc Composite Key cho Person (theo thứ tự ưu tiên):
1. "[Họ Tên] (DD/MM/YYYY)" — nếu có ngày sinh
2. "[Họ Tên] (Số CCCD)" — nếu có số CCCD/định danh cá nhân
3. "[Họ Tên] ([Đơn vị công tác rõ nhất])" — ví dụ: "Nguyễn Văn A (Sở Tài chính Nghệ An)"
4. "[Họ Tên] (không xác định)" — nếu không có thông tin định danh nào

## Quy tắc Canonicalization cho Organization:
- Các entity name được format không có các ký tự đặc biệt như: #, ?, *, ...
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

LEGAL_KG_USER_PROMPT = """## Thông tin văn bản (document_meta)
Tiêu đề văn bản: "{document_title}"
Số hiệu: {document_number}
Cơ quan ban hành: {issuing_agency}
Ngày ban hành: {published_date}

Nội dung cần phân tích:
{article_text}

## Lưu ý quan trọng khi trích xuất:
- **Phân biệt Document vs Organization**: Tiêu đề văn bản (document_title) thường là TÊN ĐẦY ĐỦ của văn bản pháp luật (VD: "Luật Bảo vệ Bí mật nhà nước", "Kế hoạch triển khai"). Nếu entity trùng hoặc gần trùng với tiêu đề → đây là Document (văn bản), KHÔNG phải Organization.
- **Số hiệu**: Nếu entity chứa số hiệu văn bản (VD: "13/2024/QH15") → đây là Document.
- **Organization**: Là cơ quan/tổ chức CỤ THỂ được nhắc đến trong điều khoản, không phải tên văn bản.

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

PERSON_EXTRACT_USER_PROMPT = """## Thông tin văn bản (document_meta)
Tiêu đề văn bản: "{document_title}"
Số hiệu: {document_number}
Cơ quan ban hành: {issuing_agency}
Ngày ban hành: {published_date}

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

# ---------------------------------------------------------------------------
# Entity Resolution (Post-Extraction Deduplication)
# ---------------------------------------------------------------------------

ENTITY_RESOLVE_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích đồng thuật thực thể (Entity Consensus) trong văn bản pháp luật Việt Nam.
Nhiệm vụ: Với danh sách các thực thể đã được trích xuất từ nhiều điều khoản khác nhau của cùng một văn bản, hãy phân tích và gộp các thực thể trùng lặp.

## Đầu vào:
Một danh sách các thực thể, mỗi thực thể có:
- name: tên thực thể (sau khi đã normalize)
- type: loại thực thể (Organization, Person, Document, Article, Location, Task)
- article_ref: điều khoản nguồn

## Lưu ý quan trọng: Ưu tiên Node Cha (Document Root)

Trong quá trình extract, các điều khoản thường viện dẫn trực tiếp tên văn bản hiện tại:
- "theo Nghị định này"
- "quy định tại Điều 5 Nghị định 13/2024/NĐ-CP"
- "Căn cứ Nghị định 13/2024/NĐ-CP"

Khi gặp các trường hợp này:
1. **Ưu tiên Node Cha** — nếu entity trùng tên, số hiệu hoặc gần trùng với Document Root đã được tạo ở bước trước,
   KHÔNG tạo node mới. Đánh dấu entity đó là tham chiếu tới Document Root.
   ví dụ: Luật Bảo Vệ Bí Mật Nhà Nước Số 29/2018/qh14" trùng "29/2018/qh14" và trùng "Luật Bảo Vệ Bí Mật Nhà Nước 2018"
   
2. **Chỉ tạo node mới** khi entity thực sự là một văn bản PHÂN BIỆT (khác số hiệu, khác ngày ban hành)
3. **Sai type**: nếu trích xuất "Nghị Định 13/2024" làm Document nhưng trùng node cha →
   BỎ entity này (đã có node Document cha rồi), KHÔNG tạo duplicate

## Các loại xung đột cần xử lý:

### 1. Trùng tên, khác type:
Ví dụ:
- "Bộ Tài Chính" (Organization) ở Điều 3
- "Bộ Tài Chính" (Document) ở Điều 7
→ PHẢI giữ loại đúng theo ngữ cảnh văn bản. Thường "Bộ Tài Chính" là Organization (cơ quan), không phải Document (văn bản).

### 2. Cùng type, khác tên gần giống (alias):
Ví dụ:
- "Hai node Document gần giống nhau: Luật Bảo Vệ Bí Mật Nhà Nước Số 29/2018/qh14" trùng "29/2018/qh14" (do trùng số hiệu văn bản) và trùng "Luật Bảo Vệ Bí Mật Nhà Nước 2018" (do trùng tên)
- "Sở Thông Tin Và Truyền Thông" (Organization)
- "Sở Thông tin và Truyền thông" (Organization)
→ Cùng một cơ quan, giữ tên đầy đủ nhất, bỏ phần duplicates.

### 3. Cùng tên, cùng type:
Giữ một bản duy nhất và luôn ưu tiên Document Root, gộp descriptions.

## Ví dụ thực tế:

**Input:**
- Document Root: "Nghị Định 13/2024/NĐ-CP (UBND Tỉnh Hà Tĩnh, 2024)"
- Entity from Điều 3: "Nghị Định 13/2024/NĐ-CP" (type=Document)
- Entity from Điều 5: "Nghị định này" (type=Document)

**Expected Output:**
- "Nghị Định 13/2024/NĐ-CP" → MERGE vào Document Root (cùng một văn bản)
- "Nghị định này" → BỎ, đã có Document Root

**Input:**
- Document Root: "Nghị Định 13/2024/NĐ-CP (UBND Tỉnh Hà Tĩnh, 2024)"
- Entity from preamble: "Nghị Định 100/2019/NĐ-CP" (type=Document)

**Expected Output:**
- "Nghị Định 100/2019/NĐ-CP" → TẠO NODE MỚI (văn bản khác, có số hiệu riêng)

## Quy tắc nghiêm ngặt:
1. Mỗi thực thể thực sự chỉ tạo MỘT node trong graph
2. Nếu cùng một tên nhưng type khác nhau → CHỌN MỘT type đúng, bỏ cái còn lại
3. Organization: Kiểm tra xem có phải là tên viết tắt của Organization khác không
4. Document: Chỉ giữ type=Document nếu entity_name chứa số hiệu văn bản (VD: "Nghị định 123/2024")
5. Trả về JSON hợp lệ, không markdown, không giải thích

## Output format:
{{
  "resolved_entities": [
    {{
      "canonical_name": "Tên chuẩn hóa cuối cùng",
      "type": "Organization|Person|Document|Article|Location|Task",
      "representative_description": "Mô tả tổng hợp từ tất cả bản",
      "source_articles": ["Điều 3", "Điều 7"],
      "merged_from": ["Tên gốc 1 (từ Điều 3)", "Tên gốc 2 (từ Điều 7)"]
    }}
  ],
  "dropped_entities": [
    {{
      "name": "Tên entity bị loại hoàn toàn",
      "reason": "Lý do loại bỏ: trùng Document Root, sai type, không phải thực thể hợp lệ..."
    }}
  ],
  "type_conflicts": [
    {{
      "name": "Tên entity",
      "types": ["Organization", "Document"],
      "resolved_type": "Organization",
      "reason": "Giải thích tại sao chọn type này"
    }}
  ]
}}

Lưu ý:
- resolved_entities: những entity được GIỮ LẠI và tạo node trong graph
- merged_from: danh sách TẤT CẢ các tên gốc đã được gộp vào canonical này (bao gồm cả canonical gốc)
- dropped_entities: những entity bị LOẠI BỎ hoàn toàn, KHÔNG tạo node, KHÔNG tạo relation
- Nếu một entity được gộp (merged), nó phải xuất hiện trong merged_from của entity canonical, KHÔNG nằm trong dropped_entities"""

ENTITY_RESOLVE_USER_PROMPT = """## Document Root (Node Cha)

Tên đầy đủ của Document Root trong graph: "{doc_name}"

Các entity SAU ĐÂY đã được tự động phát hiện là tham chiếu tới Document Root (đã xử lý bằng code):
- Entity chứa cùng số hiệu văn bản (VD: "Nghị Định 13/2024/NĐ-CP" → Document Root)
- Entity là self-reference rõ ràng ("văn bản này", "quyết định này", "Nghị định này")
- Entity trùng hoặc gần trùng với tiêu đề văn bản

## Nhiệm vụ

Với danh sách các entity CÒN LẠI bên dưới (không phải Document Root), hãy:
1. Gộp các entity trùng lặp (cùng tên, cùng type)
2. Resolve type conflicts (cùng tên, khác type) — **ưu tiên Organization cho các cơ quan nhà nước thông thường**
3. Đánh dấu entity nào bị DROP (sai type, không hợp lệ, trùng Document Root)

## Metadata văn bản
Tiêu đề: {document_title}
{doc_meta}

## Danh sách entity cần phân tích
{entity_list}

Trả về JSON theo format đã quy định."""

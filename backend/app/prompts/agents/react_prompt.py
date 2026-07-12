"""
ReAct Executor System Prompt (RAG group)
========================================

Drives the tool-calling loop in ``react_executor_node``. The model is given the
RAG tool schemas (see app/services/agents/react_tools.py) and decides which to
call, how many times, and in parallel — replacing the static intent→tool routing.

Query-type classification (regex, deterministic) selects which prompt sections
and tool schemas are sent, so the LLM gets a focused, minimal instruction set
instead of all 140+ lines every time.
"""

from __future__ import annotations

import re

# ──────────────────────────────────────────────────────────────────────────
# Regex constants (used across the executor)
# ──────────────────────────────────────────────────────────────────────────

PLAN_LINE_RE: re.Pattern[str] = re.compile(
    r"^\s*KẾ\s*HOẠCH\s*:[^\n]*\n?", re.IGNORECASE | re.UNICODE
)

READY_SIGNAL = "SẴN SÀNG TRẢ LỜI"
READY_SIGNAL_RE: re.Pattern[str] = re.compile(
    r"SẴN\s*SÀNG?\s*TRẢ\s*LỜI", re.IGNORECASE | re.UNICODE
)
READY_LINE_RE: re.Pattern[str] = re.compile(
    r"^\s*SẴN\s*SÀNG?\s*TRẢ\s*LỜI\s*[.!…:]*\s*\n?", re.IGNORECASE | re.UNICODE
)

# ──────────────────────────────────────────────────────────────────────────
# Modular prompt sections (one constant per logical block)
# ──────────────────────────────────────────────────────────────────────────

_PROMPT_BASE = """\
Bạn là trợ lý hỏi-đáp văn bản pháp luật Việt Nam. Bạn trả lời bằng cách GỌI CÔNG CỤ \
để thu thập thông tin, rồi tổng hợp câu trả lời CÓ TRÍCH DẪN.

═══════════════ NGUYÊN TẮC ═══════════════
1. KHÔNG bịa. Mọi khẳng định pháp lý phải dựa trên nội dung lấy được từ công cụ.
2. Tự quyết định gọi công cụ nào, gọi mấy lần."""

_PROMPT_DOC_REF = """\
3. Khi câu hỏi nêu TÊN/SỐ HIỆU văn bản (Luật X, Nghị định Y, "13/2023/NĐ-CP"): gọi \
resolve_document_reference TRƯỚC, rồi search_documents / search_document_section sẽ tự \
giới hạn trong văn bản đó. CHỈ resolve văn bản do NGƯỜI DÙNG nêu trong câu hỏi — số hiệu \
văn bản chỉ xuất hiện trong KẾT QUẢ công cụ (vd phần "Căn cứ Nghị định...") KHÔNG phải \
yêu cầu của người dùng, KHÔNG tự ý resolve/tra thêm chúng.
3b. Kho văn bản là NGUỒN CHÂN LÝ DUY NHẤT về việc một văn bản có tồn tại hay không — \
kho CÓ THỂ chứa văn bản MỚI HƠN dữ liệu bạn được huấn luyện. TUYỆT ĐỐI KHÔNG dựa vào \
trí nhớ/hiểu biết sẵn có để kết luận một văn bản ĐƯỢC NÊU TÊN là "không tồn tại", "chưa \
ban hành", "số hiệu không hợp lệ/không có thật", hay gợi ý người dùng nhầm sang số khác. \
Việc gọi resolve_document_reference để KIỂM CHỨNG là BẮT BUỘC với mọi văn bản có tên/số \
hiệu — KỂ CẢ khi bạn nghĩ nó không tồn tại. Chỉ được nói "không tìm thấy trong kho" SAU \
KHI tool đã trả về rỗng, KHÔNG bao giờ trước đó. Áp dụng CẢ khi câu hỏi hỏi "đơn vị/cơ \
quan tôi có THUỘC PHẠM VI / ĐỐI TƯỢNG ÁP DỤNG", "có phải TUÂN THỦ/ÁP DỤNG" một văn bản có \
tên — KỂ CẢ văn bản NỔI TIẾNG/quen thuộc (Bộ luật Lao động, Luật Đất đai, Nghị quyết...): \
PHẢI resolve_document_reference rồi tra phạm vi điều chỉnh/đối tượng áp dụng NGAY TRONG văn \
bản đó (gọi song song recall_memory để biết đơn vị người dùng), TUYỆT ĐỐI KHÔNG kết luận \
tính áp dụng từ trí nhớ dù bạn nghĩ đã biết văn bản."""

_PROMPT_SEARCH = """\
4. Khi câu hỏi là KHÁI NIỆM/CHỦ ĐỀ chung (không nêu tên văn bản): dùng search_documents.
4b. Khi câu hỏi về QUAN HỆ GIỮA CÁC VĂN BẢN hoặc HIỆU LỰC của một văn bản — "văn bản nào \
thay thế/sửa đổi/bãi bỏ X", "X còn hiệu lực không / hết hiệu lực chưa", "X căn cứ những văn \
bản nào", "X viện dẫn gì" — gọi get_document_relations(document=<tên/số hiệu X>). Với loại \
câu hỏi này gọi THẲNG get_document_relations, KHÔNG cần resolve_document_reference trước \
(tool tự khớp tên/số hiệu — đây là NGOẠI LỆ của quy tắc 3). Ví dụ: "Luật An ninh mạng 24/2018 \
còn hiệu lực không?" → get_document_relations(document="24/2018/QH14"); "Nghị định 361/2025 \
căn cứ trên những văn bản nào?" → get_document_relations(document="361/2025/NĐ-CP"). \
KHÔNG dùng search_documents cho loại câu hỏi này (nội dung chunk không nói được trạng thái \
hiệu lực hiện tại)."""

_PROMPT_AMBIGUOUS = """\
5. Khi mơ hồ (nhiều văn bản trùng, không rõ ý): gọi ask_user thay vì đoán."""

_PROMPT_PERSONAL = """\
6. CHỈ gọi recall_memory khi câu hỏi THỰC SỰ cần thông tin cá nhân/tổ chức của người dùng ĐỂ \
trả lời — ví dụ: "đơn vị tôi có phải tuân thủ...", "thiết bị của tôi có đáp ứng...", "tôi là \
cán bộ cấp xã thì...". Ngược lại, "cho tôi", "gửi tôi", "làm ơn cho tôi" chỉ là KÍNH NGỮ lịch \
sự, KHÔNG phải yêu cầu tra cứu thông tin cá nhân → TUYỆT ĐỐI KHÔNG gọi recall_memory cho \
những câu dạng này. Tương tự, chỉ gọi save_memory khi người dùng CUNG CẤP thông tin cá nhân \
MỚI (vd "Tôi đang dùng Macbook Pro 2021...", "Tôi làm ở Phòng Tài chính..."). Một câu hỏi \
thuần về văn bản/tài liệu (kể cả có từ "tôi") → KHÔNG dùng recall_memory hay save_memory."""

_PROMPT_LOOP = """\
7. Nếu công cụ trả về rỗng/không khớp: thử lại với từ khoá/công cụ khác TỐI ĐA 2-3 lần; \
nếu vẫn không có văn bản/nội dung được hỏi trong kho thì KẾT LUẬN "không tìm thấy trong kho" \
NGAY, KHÔNG tiếp tục đoán số hiệu văn bản khác.
8. KHI CÒN GỌI CÔNG CỤ mà câu hỏi có TỪ HAI Ý trở lên, HOẶC đây là lượt gọi công cụ tiếp \
theo sau khi đã có kết quả: kèm ĐÚNG MỘT dòng kế hoạch dạng \
`KẾ HOẠCH: <ý cần trả lời 1> [đã có/thiếu] | <ý 2> [đã có/thiếu] | ...` ngay trước các lời \
gọi công cụ — ý [thiếu] phải khớp với công cụ bạn gọi. NGOÀI dòng đó, TUYỆT ĐỐI không viết \
lời dẫn/giải thích nào khác. Dòng KẾ HOẠCH giúp các lượt sau biết đang theo đuổi gì và \
khi nào NÊN DỪNG. (Câu hỏi MỘT ý ở lượt đầu tiên có thể gọi công cụ trực tiếp, không cần \
dòng kế hoạch.) Chỉ chuyển sang bước trả lời khi đã đủ thông tin (lúc đó không gọi công cụ \
nữa và KHÔNG kèm dòng KẾ HOẠCH — xem mục KHI ĐÃ ĐỦ THÔNG TIN).
9. ƯU TIÊN TRẢ LỜI SỚM: ngay khi dữ liệu đã thu thập ĐỦ để trả lời câu hỏi CHÍNH, hãy DỪNG \
gọi công cụ và chuyển sang bước trả lời (mục KHI ĐÃ ĐỦ THÔNG TIN) — KHÔNG quét thêm chỉ để \
cho "đầy đủ". Chỉ tra tiếp khi còn thiếu một phần CỤ THỂ và THIẾT YẾU cho câu hỏi."""

_PROMPT_ATTACHED = """\
═══════════════ KHI NGƯỜI DÙNG ĐÍNH KÈM FILE ═══════════════
Nếu lượt này có VĂN BẢN ĐÍNH KÈM, chọn công cụ theo MỤC ĐÍCH của người dùng:
• TÓM TẮT TOÀN BỘ file → summarize_long_document (an toàn cho file DÀI, không bị cắt). \
Chỉ dùng read_uploaded_document khi file ngắn hoặc cần trích nguyên văn.
• TÌM / TÓM TẮT một SỐ THÔNG TIN CỤ THỂ trong file → search_documents(query=<thông tin cần>, \
scope="uploaded"); nếu rỗng (file vừa tải lên chưa lập chỉ mục) thì fallback read_uploaded_document.
• TÓM TẮT TOÀN BỘ nhưng chỉ một KHÍA CẠNH (vd 'các điều khoản xử phạt') → summarize_long_document(focus=<khía cạnh>).
• HỎI ĐÁP nội dung TRONG file → read_uploaded_document (đọc rồi trả lời).
• KIỂM TRA CHÍNH TẢ / NGỮ PHÁP → read_uploaded_document rồi TỰ rà soát, chỉ ra lỗi + sửa.
• KIỂM TRA THỂ THỨC / ĐỊNH DẠNG (căn lề, cỡ chữ, .docx) → check_document_format.
• ĐỐI CHIẾU file với QUY ĐỊNH trong kho (đúng/sai, có phù hợp luật không) → gọi SONG SONG:
    read_uploaded_document  +  search_documents(query=<điểm cần kiểm>, scope="knowledge_base")
  rồi SO SÁNH nội dung file với căn cứ tìm được, nêu rõ chỗ khớp/lệch + trích dẫn.
• HỎI ĐÁP xuyên CẢ file LẪN kho → gọi search_documents 2 lần: scope="uploaded" và scope="knowledge_base".
LƯU Ý: mặc định search_documents KHI có file sẽ chỉ tìm trong file; muốn tìm toàn kho PHẢI đặt scope="knowledge_base".
QUAN TRỌNG: nếu người dùng KHÔNG yêu cầu đối chiếu/tìm toàn kho mà nội dung họ hỏi KHÔNG có \
trong file đính kèm → trả lời rõ "Nội dung bạn hỏi không có trong file đính kèm." KHÔNG tự ý mở \
rộng sang toàn kho, KHÔNG suy đoán, KHÔNG lấy thông tin ngoài file.
QUAN TRỌNG: khi người dùng đính kèm file và câu hỏi là VỀ NỘI DUNG file (tóm tắt, trích xuất, \
lập bảng, hỏi thông tin trong file...), từ "tôi" trong câu (vd "cho tôi bảng...") CHỈ là kính \
ngữ — KHÔNG gọi recall_memory, KHÔNG gọi save_memory. File đính kèm là nguồn thông tin duy \
nhất cần dùng, thông tin cá nhân người dùng không liên quan đến câu hỏi về file.

CẢNH BÁO — SỐ HIỆU VĂN BẢN TRÍCH DẪN TRONG FILE: \
Các số hiệu văn bản (vd "Nghị định 13/2023/NĐ-CP", "Thông tư 03/2017/TT-BTTTT", "Luật X...") \
xuất hiện TRONG NỘI DUNG file đính kèm (phần "Căn cứ pháp lý", trích dẫn, viện dẫn...) CHỈ là \
tài liệu tham khảo được file trích dẫn — KHÔNG PHẢI yêu cầu của người dùng. TUYỆT ĐỐI KHÔNG \
được gọi resolve_document_reference hay search_documents cho các số hiệu này. Khi người dùng \
yêu cầu tóm tắt/hỏi về NỘI DUNG file, hãy ĐỌC nội dung đã có và trả lời — không cần tra cứu \
thêm các văn bản được trích dẫn bên trong."""

_PROMPT_PARALLEL = """\
═══════════════ GỌI SONG SONG (QUAN TRỌNG) ═══════════════
Khi cần NHIỀU mẩu thông tin ĐỘC LẬP (không cái nào phụ thuộc kết quả của cái kia), BẮT BUỘC \
gọi TẤT CẢ trong CÙNG MỘT lượt (nhiều tool_call song song) — KHÔNG gọi lần lượt từng lượt.
Ví dụ: "Thiết bị máy tính của tôi có dùng để soạn thảo tài liệu BMNN không?" → trong MỘT lượt \
viết dòng kế hoạch (rule 8) rồi gọi đồng thời 3 công cụ:
  KẾ HOẠCH: thiết bị của người dùng [thiếu] | nghĩa BMNN [thiếu] | điều kiện soạn thảo tài liệu BMNN [thiếu]
  • recall_memory(query="thiết bị máy tính của người dùng")
  • search_abbreviation(abbreviation="BMNN")
  • search_documents(query="điều kiện/thiết bị để soạn thảo tài liệu bí mật nhà nước")
CHỈ gọi tuần tự (nhiều lượt) khi tool sau CẦN kết quả của tool trước — ví dụ \
resolve_document_reference (để biết văn bản) RỒI mới search_document_section trong văn bản đó."""

_PROMPT_READY = """\
═══════════════ KHI ĐÃ ĐỦ THÔNG TIN ═══════════════
Bước trả lời gồm HAI lượt:
1) Lượt báo hiệu: khi dữ liệu đã đủ, ngừng gọi công cụ và trả lời bằng dòng: SẴN SÀNG TRẢ LỜI \
— chưa cần soạn nội dung ở lượt này.
2) Lượt soạn thảo: hệ thống sẽ yêu cầu bạn viết câu trả lời đầy đủ ngay sau đó.
(Các lượt CÒN GỌI CÔNG CỤ không thay đổi: vẫn kèm dòng KẾ HOẠCH theo quy tắc 8.)

Khi soạn câu trả lời cuối cùng (lượt soạn thảo), viết bằng TIẾNG VIỆT:
- Trích dẫn nguồn bằng cách chèn ĐÚNG mã trong ngoặc vuông xuất hiện ở kết quả công cụ — \
ví dụ kết quả ghi "Nguồn [a3z9]" thì viết [a3z9]. KHÔNG tự bịa mã (không viết [idXX], [idKG], \
[id12], [1]...); chỉ dùng mã CÓ THẬT trong kết quả. Ý nào không có mã nguồn kèm theo thì không gắn trích dẫn.
- Ngắn gọn, đúng trọng tâm, đúng nội dung văn bản; không thêm thông tin ngoài nguồn.
- Nếu không tìm được căn cứ, nói rõ "Không tìm thấy thông tin trong kho văn bản" thay vì đoán."""

# ── Legacy monolithic prompt (kept for eval suite compatibility) ──────────
REACT_SYSTEM_PROMPT = "\n\n".join([
    _PROMPT_BASE,
    _PROMPT_DOC_REF,
    _PROMPT_SEARCH,
    _PROMPT_AMBIGUOUS,
    _PROMPT_PERSONAL,
    _PROMPT_LOOP,
    _PROMPT_ATTACHED,
    _PROMPT_PARALLEL,
    _PROMPT_READY,
    "{memory_context}",
])

# ──────────────────────────────────────────────────────────────────────────
# Query-type classification (regex, deterministic — no LLM)
# ──────────────────────────────────────────────────────────────────────────

# Legal document number pattern: ordinal/year/type
_LEGAL_DOC_NUM_RE = re.compile(
    r"\b\d{1,4}\s*/\s*((?:19|20)\d{2})\s*/\s*[A-ZĐ]", re.UNICODE
)

# Legal document type keywords
_DOC_TYPE_KW_RE = re.compile(
    r"\b(?:Nghị\s*định|Thông\s*tư|Luật|Quyết\s*định|Chỉ\s*thị|Nghị\s*quyết"
    r"|Công\s*văn|Bộ\s*luật|Hiến\s*pháp|Pháp\s*lệnh)\b",
    re.IGNORECASE | re.UNICODE,
)

# Doc relations cues: asking about validity, replacements, basis
_DOC_RELATIONS_RE = re.compile(
    r"\b(?:hiệu\s*lực|còn\s*.*\s*lực|hết\s*.*\s*lực|thay\s*thế|sửa\s*đổi"
    r"|bãi\s*bỏ|hủy\s*bỏ|căn\s*cứ|viện\s*dẫn|dẫn\s*chiếu|quan\s*hệ.*văn\s*bản"
    r"|văn\s*bản\s*nào)\b",
    re.IGNORECASE | re.UNICODE,
)

# Personal-context cues: actually asking about user's own situation
_PERSONAL_CONTEXT_RE = re.compile(
    r"\b(?:đơn\s*vị\s+(?:tôi|em|mình|ta)|cơ\s*quan\s+(?:tôi|em|mình|ta)"
    r"|thiết\s*bị\s+của\s+(?:tôi|em|mình|ta)"
    r"|tôi\s+là\s+|tôi\s+đang\s+dùng|tôi\s+ở\s+|tôi\s+làm\s+ở"
    r"|cán\s*bộ\s+cấp|thuộc\s+phạm\s+vi.*tôi|tuân\s+thủ.*tôi)\b",
    re.IGNORECASE | re.UNICODE,
)

# Polite pronoun only (not personal context)
_POLITE_PRONOUN_RE = re.compile(
    r"\b(?:cho\s+tôi|gửi\s+tôi|làm\s+ơn\s+cho\s+tôi|cho\s+mình|cho\s+em)\b",
    re.IGNORECASE | re.UNICODE,
)


def classify_query_type(
    user_message: str,
    has_attached_file: bool = False,
) -> str:
    """Classify the user's question into a query type using regex only.

    Returns one of:
      - ``"attached_file"`` — file is attached, question is about file content
      - ``"doc_relations"`` — asking about validity / replacements / relations
      - ``"doc_reference"`` — names a specific legal document (resolve + search)
      - ``"personal"`` — genuinely asks about user's own situation / context
      - ``"general"`` — concept / keyword / fallback (no named doc, no file)
    """
    m = (user_message or "").strip()

    # 1. Attached file takes priority — the tools + instructions are specialised
    if has_attached_file:
        return "attached_file"

    # 2. Doc relations (hiệu lực, thay thế, bãi bỏ, căn cứ...)
    if _DOC_RELATIONS_RE.search(m):
        return "doc_relations"

    # 3. Personal context (actually asking about user's situation)
    if _PERSONAL_CONTEXT_RE.search(m):
        return "personal"

    # 4. Named document reference (has doc type keyword + possibly doc number)
    if _DOC_TYPE_KW_RE.search(m) or _LEGAL_DOC_NUM_RE.search(m):
        return "doc_reference"

    # 5. Fallback — general concept / keyword search
    return "general"


# ──────────────────────────────────────────────────────────────────────────
# Prompt assembly by query type
# ──────────────────────────────────────────────────────────────────────────

# Sections to include per query type. Every type gets BASE + LOOP + READY.
_QUERY_TYPE_SECTIONS: dict[str, list[str]] = {
    "attached_file": [
        _PROMPT_BASE,
        _PROMPT_LOOP,
        _PROMPT_ATTACHED,
        _PROMPT_READY,
    ],
    "doc_relations": [
        _PROMPT_BASE,
        _PROMPT_SEARCH,
        _PROMPT_AMBIGUOUS,
        _PROMPT_LOOP,
        _PROMPT_PARALLEL,
        _PROMPT_READY,
    ],
    "doc_reference": [
        _PROMPT_BASE,
        _PROMPT_DOC_REF,
        _PROMPT_AMBIGUOUS,
        _PROMPT_LOOP,
        _PROMPT_PARALLEL,
        _PROMPT_READY,
    ],
    "personal": [
        _PROMPT_BASE,
        _PROMPT_DOC_REF,
        _PROMPT_SEARCH,
        _PROMPT_AMBIGUOUS,
        _PROMPT_PERSONAL,
        _PROMPT_LOOP,
        _PROMPT_PARALLEL,
        _PROMPT_READY,
    ],
    "general": [
        _PROMPT_BASE,
        _PROMPT_DOC_REF,
        _PROMPT_SEARCH,
        _PROMPT_AMBIGUOUS,
        _PROMPT_LOOP,
        _PROMPT_PARALLEL,
        _PROMPT_READY,
    ],
}

# ──────────────────────────────────────────────────────────────────────────
# Shared helpers (history rendering, plan block, nudge prompt)
# ──────────────────────────────────────────────────────────────────────────

SUFFICIENCY_NUDGE_PROMPT = (
    "Bạn đã có dữ liệu ban đầu ở trên. Hãy rà soát bằng KẾ HOẠCH tường minh: "
    "liệt kê các Ý CHÍNH mà câu hỏi cần trả lời, ý nào ĐÃ CÓ căn cứ trong "
    "kết quả (kèm mã nguồn), ý nào CÒN THIẾU.\n"
    "- Nếu KHÔNG còn ý thiếu → trả về đúng dòng `SẴN SÀNG TRẢ LỜI` (không gọi "
    "công cụ, không kèm dòng KẾ HOẠCH, KHÔNG tự viết câu trả lời). Không cần "
    "tra cho hết mọi khía cạnh nếu ý chính đã đủ.\n"
    "- Nếu còn ý THIẾU THIẾT YẾU → ghi dòng `KẾ HOẠCH:` (đánh dấu ý "
    "[đã có]/[thiếu]) rồi gọi ĐÚNG công cụ cho ý thiếu đó.\n"
    "- TUYỆT ĐỐI KHÔNG resolve/tra thêm văn bản có số hiệu chỉ xuất hiện "
    "trong KẾT QUẢ công cụ (vd phần 'Căn cứ...') mà người dùng không nhắc "
    "— đó không phải yêu cầu của người dùng."
)


def _render_plan_block(
    plan: list[dict] | None = None,
    extracted_params: dict | None = None,
) -> str:
    """Render the query_analyzer plan as an explicit checklist for the ReAct loop."""
    if not plan:
        return ""
    items = []
    for i, sq in enumerate(plan, 1):
        q = (sq.get("query") or "").strip() if isinstance(sq, dict) else str(sq).strip()
        if q:
            items.append(f"  {i}. {q}")
    if not items:
        return ""
    block = (
        "\n\n═══════════════ GỢI Ý KẾ HOẠCH (định hướng, KHÔNG bắt buộc quét hết) ═══════════════\n"
        "Câu hỏi có thể gồm các phần dưới đây — dùng để ĐỊNH HƯỚNG tra cứu. "
        "ƯU TIÊN trả lời ngay khi đã đủ ý chính; KHÔNG cần tra cho hết mọi phần "
        "nếu câu hỏi chính đã được trả lời thoả đáng:\n"
        + "\n".join(items)
    )
    refs = (extracted_params or {}).get("document_refs") if extracted_params else None
    if refs:
        block += (
            "\nVăn bản được nhắc tới (resolve_document_reference trước nếu cần): "
            + ", ".join(str(r) for r in refs if r)
        )
    return block


_STALE_CITATION_RE = re.compile(r"\[[a-zA-Z0-9]{4}\]")
_DOC_TAG_RE = re.compile(r"<document_id=[^>]+>", re.IGNORECASE)
_HISTORY_USER_CAP = 400
_HISTORY_ASSISTANT_CAP = 700


def _render_history_block(history: list[tuple[str, str]] | None) -> str:
    """Render prior conversation turns as a digest block."""
    lines: list[str] = []
    for role, content in history or []:
        content = str(content or "").strip()
        if not content:
            continue
        if role == "user":
            content = _DOC_TAG_RE.sub("", content).strip()
            speaker, cap = "Người dùng", _HISTORY_USER_CAP
        else:
            content = _STALE_CITATION_RE.sub("", content)
            speaker, cap = "Trợ lý", _HISTORY_ASSISTANT_CAP
        if len(content) > cap:
            content = content[:cap] + "…"
        if content:
            lines.append(f"{speaker}: {content}")
    if not lines:
        return ""
    return (
        "\n\n═══════════════ HỘI THOẠI TRƯỚC ĐÓ (chỉ để hiểu câu hỏi) ═══════════════\n"
        + "\n".join(lines)
        + "\nDùng đoạn hội thoại trên CHỈ để hiểu câu hỏi hiện tại (giải nghĩa "
        '"văn bản này", "điều đó", câu hỏi nối tiếp...). Nó KHÔNG phải nguồn trả lời: '
        "mọi căn cứ pháp lý vẫn phải lấy từ CÔNG CỤ gọi trong lượt này — quy tắc 1 và 3b "
        "giữ nguyên hiệu lực, KHÔNG dùng lại mã nguồn của các lượt trước."
    )


# ──────────────────────────────────────────────────────────────────────────
# build_react_system_prompt (with optional query-type filtering)
# ──────────────────────────────────────────────────────────────────────────


def build_react_system_prompt(
    memory_context: str = "",
    plan: list[dict] | None = None,
    extracted_params: dict | None = None,
    history: list[tuple[str, str]] | None = None,
    query_type: str | None = None,
    has_attached_file: bool = False,
) -> str:
    """Render the system prompt.

    When ``query_type`` is provided (or auto-detected via ``has_attached_file``),
    only the sections relevant to that type are included — cutting the prompt to
    ~30–60% of its full size. Pass ``query_type`` from ``classify_query_type``
    for deterministic classification, or leave ``None`` for the full legacy prompt.
    """
    # Auto-classify when no explicit type given
    if query_type is None:
        # We don't have user_message here, but we can infer attached_file from flag
        if has_attached_file:
            query_type = "attached_file"
        else:
            query_type = "general"

    sections = _QUERY_TYPE_SECTIONS.get(query_type, _QUERY_TYPE_SECTIONS["general"])

    block = _render_plan_block(plan, extracted_params)
    if memory_context and memory_context.strip():
        block += (
            "\n\n═══════════════ NGỮ CẢNH CÁ NHÂN (đã truy hồi) ═══════════════\n"
            + memory_context.strip()
            + "\nDùng ngữ cảnh này khi câu hỏi liên quan tới người dùng; nếu cần thêm, gọi recall_memory."
        )
    block += _render_history_block(history)

    prompt_body = "\n\n".join(list(sections) + [block])
    return prompt_body

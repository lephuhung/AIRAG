"""
Document Type Classifier
========================
Classify Vietnamese administrative / legal documents from their markdown content
using an LLM (Qwen3-4B via vLLM / memory agent).

Return a dictionary with:
  - slug             matches DocumentType.slug in DB, or None.
  - document_number  official document reference number (e.g. "13/2023/NĐ-CP").
  - location         Province or city of issuance.
  - issuing_agency   Direct issuing body.
  - parent_agency    Parent organization.
  - published_date   Date of publication.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document-type definitions  (slug + metadata only — no regex patterns)
# ---------------------------------------------------------------------------

@dataclass
class _DocType:
    slug: str
    name: str
    description: str


_DOC_TYPES: list[_DocType] = [
    _DocType("luat",        "Luật",         "Luật do Quốc hội ban hành"),
    _DocType("phap_lenh",   "Pháp lệnh",    "Pháp lệnh của Ủy ban Thường vụ Quốc hội"),
    _DocType("nghi_quyet",  "Nghị quyết",   "Nghị quyết của Quốc hội, HĐND, Đảng, tổ chức"),
    _DocType("nghi_dinh",   "Nghị định",    "Nghị định của Chính phủ"),
    _DocType("quyet_dinh",  "Quyết định",   "Quyết định hành chính của cơ quan nhà nước"),
    _DocType("thong_tu",    "Thông tư",     "Thông tư (bao gồm Thông tư liên tịch) của Bộ, cơ quan ngang Bộ"),
    _DocType("chi_thi",     "Chỉ thị",      "Chỉ thị của Thủ tướng, Bộ trưởng"),
    _DocType("cong_van",    "Công văn",     "Công văn hành chính"),
    _DocType("thong_bao",   "Thông báo",    "Thông báo hành chính"),
    _DocType("bao_cao",     "Báo cáo",      "Báo cáo định kỳ, chuyên đề, tài chính"),
    _DocType("to_trinh",    "Tờ trình",     "Tờ trình lên cấp trên"),
    _DocType("bien_ban",    "Biên bản",     "Biên bản họp, làm việc, kiểm tra"),
    _DocType("hop_dong",    "Hợp đồng",     "Hợp đồng kinh tế, dân sự, lao động"),
    _DocType("ke_hoach",    "Kế hoạch",     "Kế hoạch công tác, triển khai"),
    _DocType("huong_dan",   "Hướng dẫn",    "Hướng dẫn thực hiện, nghiệp vụ"),
    _DocType("don_tu",      "Đơn, Tờ khai", "Đơn xin, đơn đề nghị, tờ khai"),
]

# Slug set for LLM response validation
_VALID_SLUGS: frozenset[str] = frozenset(d.slug for d in _DOC_TYPES)

# ---------------------------------------------------------------------------
# LLM classification prompt
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân tích siêu dữ liệu văn bản hành chính Việt Nam.\n"
    "Nhiệm vụ: Đọc phần đầu của văn bản (markdown hoặc text OCR trang đầu) và bóc tách các trường thông tin tiêu chuẩn của Header.\n"
    "Trả về JSON thuần với đúng 6 trường sau (không dùng markdown):\n\n"
    "1. \"slug\": loại văn bản (chọn từ danh sách bên dưới, hoặc \"unknown\")\n"
    "2. \"document_number\": số hiệu chính thức (ví dụ: \"13/2023/NĐ-CP\", \"23/BC-VPUB\"). Bỏ qua chữ 'Số:'.\n"
    "3. \"parent_agency\": Tên cơ quan chủ quản / cấp trên, thường ở góc TRÊN CÙNG BÊN TRÁI (VD: \"UBND TỈNH HÀ TĨNH\").\n"
    "4. \"issuing_agency\": Tên đơn vị ban hành trực tiếp, thường nằm ngay dưới parent_agency (VD: \"VĂN PHÒNG\").\n"
    "5. \"location\": Địa danh ban hành ở góc TRÊN CÙNG BÊN PHẢI (VD: \"Hà Tĩnh\", \"Hà Nội\").\n"
    "6. \"published_date\": Ngày tháng năm ban hành (VD: \"15/01/2026\").\n\n"
    "Các slug hợp lệ:\n"
    + "\n".join(f"  - {d.slug}: {d.name} — {d.description}" for d in _DOC_TYPES)
    + "\n\n"
    "Quy tắc:\n"
    "- Chỉ trả về duy nhất chuỗi JSON, không giải thích.\n"
    "- Nếu giá trị nào không có, đặt là `null`.\n"
    "- Mọi text nên giữ nguyên case gốc nếu được hoặc chuẩn hoá Title Case.\n"
    "Ví dụ đầu ra:\n"
    "{\"slug\": \"bao_cao\", \"document_number\": \"23/BC-VPUB\", \"parent_agency\": \"UBND TỈNH HÀ TĨNH\", \"issuing_agency\": \"VĂN PHÒNG\", \"location\": \"Hà Tĩnh\", \"published_date\": \"15/01/2026\"}"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def classify_with_llm(markdown_text: str) -> dict:
    """
    Classify a document and extract rich header metadata via memory agent LLM.

    Returns:
        A dict containing: slug, document_number, location, issuing_agency, parent_agency, published_date.
    """
    from app.services.llm import get_memory_agent
    from app.services.llm.types import LLMMessage

    default_result = {
        "slug": None, "document_number": None, "location": None,
        "issuing_agency": None, "parent_agency": None, "published_date": None
    }
    if not markdown_text:
        return default_result

    content_str = ""
    try:
        llm = get_memory_agent()
        preview = markdown_text[:1500].strip()
        messages: list[LLMMessage] = [
            LLMMessage(role="user", content=f"Nội dung văn bản (markdown):\n\n{preview}")
        ]
        result = await llm.acomplete(
            messages,
            system_prompt=_LLM_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=200,
        )
        content_str = result if isinstance(result, str) else result.content
        if not content_str:
            return default_result

        # Strip markdown fences if the model wrapped the JSON
        content_str = content_str.strip()
        if content_str.startswith("```"):
            content_str = re.sub(r"^```[a-z]*\n?", "", content_str)
            content_str = content_str.rstrip("`").strip()

        parsed = json.loads(content_str)

        # --- slug ---
        raw_slug = str(parsed.get("slug", "")).strip().lower()
        raw_slug = re.sub(r"[^a-z_]", "", raw_slug)
        slug = raw_slug if raw_slug in _VALID_SLUGS else None

        # Clean string helper
        def _clean_str(val):
            if not val or str(val).lower() == "null" or str(val).lower() == "none":
                return None
            return str(val).strip()

        document_number = _clean_str(parsed.get("document_number"))
        if document_number and len(document_number) > 60:
            document_number = None

        final_res = {
            "slug": slug,
            "document_number": document_number,
            "location": _clean_str(parsed.get("location")),
            "issuing_agency": _clean_str(parsed.get("issuing_agency")),
            "parent_agency": _clean_str(parsed.get("parent_agency")),
            "published_date": _clean_str(parsed.get("published_date"))
        }

        logger.info(
            f"[classifier] Extracted rich metadata: {final_res}"
        )
        return final_res

    except (json.JSONDecodeError, KeyError, TypeError) as parse_err:
        logger.debug(f"[classifier] JSON parse failed: {parse_err} — raw: {content_str!r}")
    except Exception as err:
        logger.warning(f"[classifier] LLM classification failed (non-fatal): {err}")

    return default_result


def get_all_document_types() -> list[dict]:
    """
    Return all supported document types.
    Used to seed the document_types table on startup (idempotent).
    """
    return [
        {"slug": d.slug, "name": d.name, "description": d.description}
        for d in _DOC_TYPES
    ]

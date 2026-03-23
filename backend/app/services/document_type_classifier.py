"""
Document Type Classifier
========================
Classify Vietnamese administrative / legal documents from their markdown content
using an LLM (Qwen3-4B via vLLM / memory agent).

Returns a tuple (slug, document_number):
  - slug          matches DocumentType.slug in DB, or None if unrecognised.
  - document_number  the official document reference number (e.g. "13/2023/NĐ-CP"),
                  or None if not found.
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
    "Bạn là chuyên gia phân loại văn bản hành chính và pháp luật Việt Nam.\n"
    "Nhiệm vụ: đọc phần đầu nội dung văn bản (markdown) và trả về JSON với 2 trường:\n\n"
    "1. \"slug\": loại văn bản (chọn từ danh sách bên dưới, hoặc \"unknown\")\n"
    "2. \"document_number\": số hiệu văn bản chính thức (ví dụ: \"13/2023/NĐ-CP\", "
    "\"1234/CV-BTC\", \"45/2021/QH15\"). Nếu không có thì null.\n\n"
    "Các slug hợp lệ:\n"
    + "\n".join(f"  - {d.slug}: {d.name} — {d.description}" for d in _DOC_TYPES)
    + "\n\n"
    "Quy tắc:\n"
    "- Chỉ trả về JSON thuần, không giải thích, không markdown.\n"
    "- document_number phải là chuỗi số hiệu gốc trong văn bản, "
    "KHÔNG tự suy luận hay bịa đặt.\n"
    "- Nếu số hiệu xuất hiện nhiều lần, lấy lần đầu tiên.\n\n"
    "Ví dụ đầu vào:\n"
    "\"NGHỊ ĐỊNH\\nSố: 13/2023/NĐ-CP\\nQuy định về bảo vệ dữ liệu cá nhân\"\n\n"
    "Ví dụ đầu ra:\n"
    "{\"slug\": \"nghi_dinh\", \"document_number\": \"13/2023/NĐ-CP\"}"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def classify_with_llm(markdown_text: str) -> tuple[str | None, str | None]:
    """
    Classify a document and extract its official number using the memory agent LLM.

    Args:
        markdown_text: Parsed markdown content (first ~1 500 chars used).

    Returns:
        Tuple of (slug, document_number). Either may be None if not found/recognised.
    """
    from app.services.llm import get_memory_agent
    from app.services.llm.types import LLMMessage

    if not markdown_text:
        return None, None

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
            max_tokens=60,
        )
        content_str = result if isinstance(result, str) else result.content
        if not content_str:
            return None, None

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

        # --- document_number ---
        raw_number = parsed.get("document_number")
        document_number = str(raw_number).strip() if raw_number else None
        # Reject suspiciously long or empty strings
        if document_number and (len(document_number) > 60 or document_number == "null"):
            document_number = None

        logger.info(
            f"[classifier] slug={slug!r} document_number={document_number!r}"
        )
        return slug, document_number

    except (json.JSONDecodeError, KeyError, TypeError) as parse_err:
        logger.debug(f"[classifier] JSON parse failed: {parse_err} — raw: {content_str!r}")
    except Exception as err:
        logger.warning(f"[classifier] LLM classification failed (non-fatal): {err}")

    return None, None


def get_all_document_types() -> list[dict]:
    """
    Return all supported document types.
    Used to seed the document_types table on startup (idempotent).
    """
    return [
        {"slug": d.slug, "name": d.name, "description": d.description}
        for d in _DOC_TYPES
    ]

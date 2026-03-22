"""
Document Type Classifier
========================
Classify Vietnamese administrative / legal documents from their markdown content.

Flow:
  1. Regex pass on first ~1 000 chars of markdown content.
  2. If no regex match → async LLM fallback (returns the closest slug or None).

Returns `slug` matching `DocumentType.slug` in DB, or `None` if unrecognised.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Broad document-type definitions
# ---------------------------------------------------------------------------

@dataclass
class _DocTypePattern:
    slug: str
    name: str
    description: str
    patterns: list[str]   # case-insensitive, UNICODE regex patterns


# Ordered most-specific → most-general so the first match wins.
_PATTERNS: list[_DocTypePattern] = [
    _DocTypePattern(
        slug="luat",
        name="Luật",
        description="Luật do Quốc hội ban hành",
        patterns=[
            r"LUẬT\s+[A-ZĐÀÁẢÃẠĂẮẶẶẢẸẺẼẸÍÌỈĨỊ]",
            r"(?:^|[\s_\-])lu[ậa]t[\s_\-]",
            r"(?:^|[\s_\-])luat[\s_\-]",
        ],
    ),
    _DocTypePattern(
        slug="phap_lenh",
        name="Pháp lệnh",
        description="Pháp lệnh của Ủy ban Thường vụ Quốc hội",
        patterns=[
            r"PHÁP\s+LỆNH",
            r"\bph[áa]p\s*l[ệe]nh\b",
            r"\bphap[\s_\-]?lenh\b",
        ],
    ),
    _DocTypePattern(
        slug="nghi_quyet",
        name="Nghị quyết",
        description="Nghị quyết của Quốc hội, HĐND, Đảng, tổ chức",
        patterns=[
            r"NGHỊ\s+QUYẾT",
            r"\bngh[ịi]\s*quy[ếe]t\b",
            r"\bnghi[\s_\-]?quyet\b",
            r"\bNQ[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="nghi_dinh",
        name="Nghị định",
        description="Nghị định của Chính phủ",
        patterns=[
            r"NGHỊ\s+ĐỊNH",
            r"\bngh[ịi]\s*[đd][ịi]nh\b",
            r"\bnghi[\s_\-]?dinh\b",
            r"\bND[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="quyet_dinh",
        name="Quyết định",
        description="Quyết định hành chính của cơ quan nhà nước",
        patterns=[
            r"QUYẾT\s+ĐỊNH",
            r"\bquy[ếe]t\s*[đd][ịi]nh\b",
            r"\bquyet[\s_\-]?dinh\b",
            r"\bQĐ[\s\-_/]\d+",
            r"\bQD[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="thong_tu",
        name="Thông tư",
        description="Thông tư (bao gồm Thông tư liên tịch) của Bộ, cơ quan ngang Bộ",
        patterns=[
            r"THÔNG\s+TƯ",
            r"\bth[ôo]ng\s*t[ưu]\b",
            r"\bthong[\s_\-]?tu\b",
            r"\bTTLT[\s\-_/]\d+",
            r"\bTT[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="chi_thi",
        name="Chỉ thị",
        description="Chỉ thị của Thủ tướng, Bộ trưởng",
        patterns=[
            r"CHỈ\s+THỊ",
            r"\bch[ỉi]\s*th[ịi]\b",
            r"\bchi[\s_\-]?thi\b",
            r"\bCT[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="cong_van",
        name="Công văn",
        description="Công văn hành chính",
        patterns=[
            r"CÔNG\s+VĂN",
            r"\bc[ôo]ng\s*v[ăa]n\b",
            r"\bcong[\s_\-]?van\b",
            r"\bCV[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="thong_bao",
        name="Thông báo",
        description="Thông báo hành chính",
        patterns=[
            r"THÔNG\s+BÁO",
            r"\bth[ôo]ng\s*b[áa]o\b",
            r"\bthong[\s_\-]?bao\b",
            r"\bTB[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="bao_cao",
        name="Báo cáo",
        description="Báo cáo định kỳ, chuyên đề, tài chính",
        patterns=[
            r"BÁO\s+CÁO",
            r"\bb[áa]o\s*c[áa]o\b",
            r"\bbao[\s_\-]?cao\b",
            r"\bBC[\s\-_/]\d+",
            r"\bbctc\b",
            r"\bfinancial\s+(report|statement)\b",
        ],
    ),
    _DocTypePattern(
        slug="to_trinh",
        name="Tờ trình",
        description="Tờ trình lên cấp trên",
        patterns=[
            r"TỜ\s+TRÌNH",
            r"\bt[ờo]\s*tr[ìi]nh\b",
            r"\bto[\s_\-]?trinh\b",
            r"\bTTr[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="bien_ban",
        name="Biên bản",
        description="Biên bản họp, làm việc, kiểm tra",
        patterns=[
            r"BIÊN\s+BẢN",
            r"\bbi[êe]n\s*b[ảa]n\b",
            r"\bbien[\s_\-]?ban\b",
            r"\bBB[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="hop_dong",
        name="Hợp đồng",
        description="Hợp đồng kinh tế, dân sự, lao động",
        patterns=[
            r"HỢP\s+ĐỒNG",
            r"\bh[ợo]p\s*[đd][ồo]ng\b",
            r"\bhop[\s_\-]?dong\b",
            r"\bHĐ[\s\-_/]\d+",
            r"\bHD[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="ke_hoach",
        name="Kế hoạch",
        description="Kế hoạch công tác, triển khai",
        patterns=[
            r"KẾ\s+HOẠCH",
            r"\bk[ếe]\s*ho[ạa]ch\b",
            r"\bke[\s_\-]?hoach\b",
            r"\bKH[\s\-_/]\d+",
        ],
    ),
    _DocTypePattern(
        slug="huong_dan",
        name="Hướng dẫn",
        description="Hướng dẫn thực hiện, nghiệp vụ",
        patterns=[
            r"HƯỚNG\s+DẪN",
            r"\bh[ướu]ng\s*d[ẫa]n\b",
            r"\bhuong[\s_\-]?dan\b",
        ],
    ),
    _DocTypePattern(
        slug="don_tu",
        name="Đơn, Tờ khai",
        description="Đơn xin, đơn đề nghị, tờ khai",
        patterns=[
            r"ĐƠN\s+(XIN|ĐỀ\s+NGHỊ)",
            r"\b[đd][ơo]n\s*(xin|[đd][ềe]\s*ngh[ịi]|khi[ếe]u\s*n[ạa]i)\b",
            r"\bdon[\s_\-]?(xin|de[\s_\-]nghi)\b",
            r"\bt[ờo]\s*khai\b",
        ],
    ),
]

# Compile once at module load
_COMPILED: list[tuple[str, str, list[re.Pattern]]] = [
    (
        p.slug,
        p.name,
        [re.compile(pat, re.IGNORECASE | re.UNICODE) for pat in p.patterns],
    )
    for p in _PATTERNS
]

# Slug set for LLM response validation
_VALID_SLUGS: frozenset[str] = frozenset(p.slug for p in _PATTERNS)

# Prompt for LLM fallback
_LLM_SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân loại văn bản hành chính và pháp luật Việt Nam. "
    "Nhiệm vụ: xác định loại văn bản từ nội dung được cung cấp.\n\n"
    "Các loại văn bản hợp lệ (trả về đúng slug sau đây):\n"
    + "\n".join(f"  - {p.slug}: {p.name} — {p.description}" for p in _PATTERNS)
    + "\n\nNếu không xác định được loại văn bản, trả về: unknown\n"
    "Chỉ trả về một slug duy nhất, không giải thích thêm."
)


def classify_document_type(markdown_text: str) -> str | None:
    """
    Regex-based classification on markdown content.

    Args:
        markdown_text: First ~1 000 chars of parsed markdown content.

    Returns:
        Matching slug or None (no match).
    """
    if not markdown_text:
        return None

    for slug, _name, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(markdown_text):
                logger.debug(
                    f"[classifier] matched '{slug}' via pattern '{pat.pattern[:40]}'"
                )
                return slug

    logger.debug("[classifier] no regex match")
    return None


async def classify_with_llm(markdown_text: str) -> str | None:
    """
    LLM fallback classification when regex returns None.

    Args:
        markdown_text: First ~1 000 chars of parsed markdown content.

    Returns:
        Matching slug, or None if LLM also cannot classify.
    """
    from app.services.llm import get_memory_agent
    from app.services.llm.types import LLMMessage

    try:
        # Use the dedicated memory agent (e.g. Qwen3.5-0.8B) for classification
        llm = get_memory_agent()
        preview = markdown_text[:1500].strip()  # Increased preview length for better context
        messages: list[LLMMessage] = [
            LLMMessage(role="user", content=f"Nội dung văn bản (markdown):\n\n{preview}")
        ]
        result = await llm.acomplete(
            messages,
            system_prompt=_LLM_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=20,
        )
        # Handle both string and content object results
        content_str = result if isinstance(result, str) else result.content
        slug = content_str.strip().lower() if content_str else ""
        # Sanitise: remove surrounding punctuation/whitespace
        slug = re.sub(r"[^a-z_]", "", slug)
        if slug in _VALID_SLUGS:
            logger.info(f"[classifier] LLM classified as '{slug}'")
            return slug
        logger.debug(f"[classifier] LLM returned unrecognised slug: {slug!r}")
    except Exception as _err:
        logger.warning(f"[classifier] LLM fallback failed (non-fatal): {_err}")

    return None


def get_all_document_types() -> list[dict]:
    """
    Return all supported document types.
    Used to seed the document_types table on startup (idempotent).
    """
    return [
        {
            "slug": p.slug,
            "name": p.name,
            "description": p.description,
        }
        for p in _PATTERNS
    ]

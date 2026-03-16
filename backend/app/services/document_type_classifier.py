"""
Document Type Classifier
========================
Regex-based classifier cho các loại văn bản hành chính / pháp luật Việt Nam.

Phân tích tên file (original_filename) và nội dung đầu văn bản (first ~500 chars)
để xác định loại văn bản.

Thứ tự ưu tiên: tên file → tiêu đề văn bản → nội dung đầu tiên.

Trả về `slug` khớp với `DocumentType.slug` trong DB, hoặc `None` nếu không nhận ra.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Định nghĩa các loại văn bản (slug, tên tiếng Việt, các pattern regex)
# ---------------------------------------------------------------------------

@dataclass
class _DocTypePattern:
    slug: str
    name: str                  # Tên tiếng Việt đầy đủ
    description: str
    patterns: list[str]        # list regex patterns (case-insensitive, UNICODE)


# Patterns được sắp xếp từ đặc thù nhất → tổng quát nhất
_PATTERNS: list[_DocTypePattern] = [
    _DocTypePattern(
        slug="nghi_quyet",
        name="Nghị quyết",
        description="Nghị quyết của Quốc hội, HĐND, Đảng, tổ chức",
        patterns=[
            r"\bngh[ịi]\s*quy[ếe]t\b",
            r"\bnghi\s*quyet\b",
            r"\bNQ[\s\-_/]\d+",
            r"NGHỊ\s+QUYẾT",
        ],
    ),
    _DocTypePattern(
        slug="nghi_dinh",
        name="Nghị định",
        description="Nghị định của Chính phủ",
        patterns=[
            r"\bngh[ịi]\s*[đd][ịi]nh\b",
            r"\bnghi[\s_\-]?dinh\b",
            r"\bND[\s\-_/]\d+",
            r"NGHỊ\s+ĐỊNH",
        ],
    ),
    _DocTypePattern(
        slug="quyet_dinh",
        name="Quyết định",
        description="Quyết định hành chính của cơ quan nhà nước",
        patterns=[
            r"\bquy[ếe]t\s*[đd][ịi]nh\b",
            r"\bquyet[\s_\-]?dinh\b",
            r"\bQD[\s\-_/]\d+",
            r"\bQĐ[\s\-_/]\d+",
            r"QUYẾT\s+ĐỊNH",
        ],
    ),
    _DocTypePattern(
        slug="thong_tu_lien_tich",
        name="Thông tư liên tịch",
        description="Thông tư liên tịch giữa các Bộ",
        patterns=[
            r"\bth[ôo]ng\s*t[ưu]\s*li[eê]n\s*t[ịi]ch\b",
            r"\bthong[\s_\-]?tu[\s_\-]?lien[\s_\-]?tich\b",
            r"\bTTLT[\s\-_/]\d+",
            r"THÔNG\s+TƯ\s+LIÊN\s+TỊCH",
        ],
    ),
    _DocTypePattern(
        slug="thong_tu",
        name="Thông tư",
        description="Thông tư của Bộ, cơ quan ngang Bộ",
        patterns=[
            r"\bth[ôo]ng\s*t[ưu]\b",
            r"\bthong[\s_\-]?tu\b",
            r"\bTT[\s\-_/]\d+",
            r"THÔNG\s+TƯ",
        ],
    ),
    _DocTypePattern(
        slug="chi_thi",
        name="Chỉ thị",
        description="Chỉ thị của Thủ tướng, Bộ trưởng",
        patterns=[
            r"\bch[ỉi]\s*th[ịi]\b",
            r"\bchi[\s_\-]?thi\b",
            r"\bCT[\s\-_/]\d+",
            r"CHỈ\s+THỊ",
        ],
    ),
    _DocTypePattern(
        slug="phap_lenh",
        name="Pháp lệnh",
        description="Pháp lệnh của Ủy ban Thường vụ Quốc hội",
        patterns=[
            r"\bph[áa]p\s*l[ệe]nh\b",
            r"\bphap[\s_\-]?lenh\b",
            r"PHÁP\s+LỆNH",
        ],
    ),
    _DocTypePattern(
        slug="luat",
        name="Luật",
        description="Luật do Quốc hội ban hành",
        patterns=[
            r"(?:^|[\s_\-])lu[ậa]t[\s_\-]",
            r"(?:^|[\s_\-])luat[\s_\-]",
            r"\bLUẬT\s+[A-ZĐÀÁẢÃẠĂẮẶẴẶẢẸẺẼẸÍÌỈĨỊ]",
        ],
    ),
    _DocTypePattern(
        slug="cong_van",
        name="Công văn",
        description="Công văn hành chính",
        patterns=[
            r"\bc[ôo]ng\s*v[ăa]n\b",
            r"\bcong[\s_\-]?van\b",
            r"\bCV[\s\-_/]\d+",
            r"CÔNG\s+VĂN",
        ],
    ),
    _DocTypePattern(
        slug="to_trinh",
        name="Tờ trình",
        description="Tờ trình lên cấp trên",
        patterns=[
            r"\bt[ờo]\s*tr[ìi]nh\b",
            r"\bto[\s_\-]?trinh\b",
            r"\bTTr[\s\-_/]\d+",
            r"TỜ\s+TRÌNH",
        ],
    ),
    _DocTypePattern(
        slug="bao_cao_tai_chinh",
        name="Báo cáo tài chính",
        description="Báo cáo tài chính, kết quả kinh doanh",
        patterns=[
            r"\bb[áa]o\s*c[áa]o\s*t[àa]i\s*ch[íi]nh\b",
            r"\bbao[\s_\-]?cao[\s_\-]?tai[\s_\-]?chinh\b",
            r"\bbctc\b",
            r"\bfinancial\s+(report|statement)\b",
            r"\bk[ếe]t\s*qu[ảa]\s*kinh\s*doanh\b",
        ],
    ),
    _DocTypePattern(
        slug="bao_cao",
        name="Báo cáo",
        description="Báo cáo định kỳ hoặc chuyên đề",
        patterns=[
            r"\bb[áa]o\s*c[áa]o\b",
            r"\bbao[\s_\-]?cao\b",
            r"\bBC[\s\-_/]\d+",
            r"BÁO\s+CÁO",
        ],
    ),
    _DocTypePattern(
        slug="thong_bao",
        name="Thông báo",
        description="Thông báo hành chính",
        patterns=[
            r"\bth[ôo]ng\s*b[áa]o\b",
            r"\bthong[\s_\-]?bao\b",
            r"\bTB[\s\-_/]\d+",
            r"THÔNG\s+BÁO",
        ],
    ),
    _DocTypePattern(
        slug="bien_ban",
        name="Biên bản",
        description="Biên bản họp, làm việc, kiểm tra",
        patterns=[
            r"\bbi[êe]n\s*b[ảa]n\b",
            r"\bbien[\s_\-]?ban\b",
            r"\bBB[\s\-_/]\d+",
            r"BIÊN\s+BẢN",
        ],
    ),
    _DocTypePattern(
        slug="hop_dong",
        name="Hợp đồng",
        description="Hợp đồng kinh tế, dân sự, lao động",
        patterns=[
            r"\bh[ợo]p\s*[đd][ồo]ng\b",
            r"\bhop[\s_\-]?dong\b",
            r"\bHD[\s\-_/]\d+",
            r"\bHĐ[\s\-_/]\d+",
            r"HỢP\s+ĐỒNG",
        ],
    ),
    _DocTypePattern(
        slug="ke_hoach",
        name="Kế hoạch",
        description="Kế hoạch công tác, triển khai",
        patterns=[
            r"\bk[ếe]\s*ho[ạa]ch\b",
            r"\bke[\s_\-]?hoach\b",
            r"\bKH[\s\-_/]\d+",
            r"KẾ\s+HOẠCH",
        ],
    ),
    _DocTypePattern(
        slug="huong_dan",
        name="Hướng dẫn",
        description="Hướng dẫn thực hiện, nghiệp vụ",
        patterns=[
            r"\bh[ướu]ng\s*d[ẫa]n\b",
            r"\bhuong[\s_\-]?dan\b",
            r"\bHD[\s\-_/]\d+",
            r"HƯỚNG\s+DẪN",
        ],
    ),
    _DocTypePattern(
        slug="tai_lieu_ky_thuat",
        name="Tài liệu kỹ thuật",
        description="Tài liệu kỹ thuật, đặc tả, thiết kế",
        patterns=[
            r"\bt[àa]i\s*li[ệe]u\s*k[ỹy]\s*thu[ậa]t\b",
            r"\btai[\s_\-]?lieu[\s_\-]?ky[\s_\-]?thuat\b",
            r"\bspecification\b",
            r"\bdesign[\s_\-]doc",
            r"\btechnical[\s_\-]report\b",
        ],
    ),
    _DocTypePattern(
        slug="don_tu",
        name="Đơn, Tờ khai",
        description="Đơn xin, đơn đề nghị, tờ khai",
        patterns=[
            r"\b[đd][ơo]n\s*(xin|[đd][ềe]\s*ngh[ịi]|khi[ếe]u\s*n[ạa]i)\b",
            r"\bdon[\s_\-]?(xin|de[\s_\-]nghi)\b",
            r"\bt[ờo]\s*khai\b",
            r"\bto[\s_\-]?khai\b",
            r"ĐƠN\s+(XIN|ĐỀ\s+NGHỊ)",
        ],
    ),
]

# Compile tất cả patterns một lần khi module load
_COMPILED: list[tuple[str, str, list[re.Pattern]]] = [
    (
        p.slug,
        p.name,
        [re.compile(pat, re.IGNORECASE | re.UNICODE) for pat in p.patterns],
    )
    for p in _PATTERNS
]


def classify_document_type(
    filename: str,
    text_preview: str = "",
) -> str | None:
    """
    Trả về slug của loại văn bản hoặc None.

    Args:
        filename: Tên file gốc (original_filename)
        text_preview: ~600 ký tự đầu của nội dung văn bản (markdown/text)
    """
    # Normalize filename: thay _ và - bằng space để \b hoạt động đúng
    filename_normalized = re.sub(r"[_\-]", " ", filename)
    # Bỏ extension
    filename_no_ext = re.sub(r"\.\w+$", "", filename_normalized)

    # Kết hợp: filename (ưu tiên cao) + preview nội dung
    haystack = f"{filename_no_ext}\n{text_preview}"

    for slug, _name, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(haystack):
                logger.debug(
                    f"[classifier] matched '{slug}' via '{pat.pattern}' "
                    f"in '{filename}'"
                )
                return slug

    logger.debug(f"[classifier] no match for '{filename}'")
    return None


def get_all_document_types() -> list[dict]:
    """
    Trả về danh sách tất cả loại văn bản được hỗ trợ.
    Dùng để seed bảng document_types khi khởi động.
    """
    return [
        {
            "slug": p.slug,
            "name": p.name,
            "description": p.description,
        }
        for p in _PATTERNS
    ]

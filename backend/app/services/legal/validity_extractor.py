"""
Trích xuất thông tin HIỆU LỰC từ văn bản pháp luật (regex-first, không LLM).

Văn bản luật/nghị định/thông tư gần như luôn tuyên bố hiệu lực ở các điều
cuối ("Điều N. Hiệu lực thi hành", chương "ĐIỀU KHOẢN THI HÀNH"):

    "Nghị định này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2016."
    "Luật An ninh mạng số 24/2018/QH14 hết hiệu lực kể từ ngày Luật này..."
    "các quy định liên quan đến vị trí việc làm công chức tại Nghị định số
     62/2020/NĐ-CP ... hết hiệu lực thi hành."          (một phần)
    "Bãi bỏ khoản 3 Điều 49 của Luật Thư viện số 46/2019/QH14."  (một phần)

Đây là dữ kiện pháp lý quan trọng nên đi đường regex trên vùng "điều khoản
thi hành" (định vị bằng ``find_headings`` — cùng nền với chunker/heading_path)
thay vì phụ thuộc LLM; trích xuất KG (THAY_THE/BAI_BO) chỉ là lớp bổ sung.

Dùng ở: parse_worker (văn bản mới), scripts/backfill_validity.py (kho cũ).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.parsing.heading_path import find_headings

# Loại văn bản đứng trước "này" / "số N" trong các câu tuyên bố hiệu lực.
_DOC_TYPES = (
    r"(?:Nghị\s+định|Luật|Bộ\s+luật|Thông\s+tư(?:\s+liên\s+tịch)?|"
    r"Quyết\s+định|Nghị\s+quyết|Pháp\s+lệnh|Chỉ\s+thị)"
)

# Số hiệu văn bản: "62/2020/NĐ-CP", "24/2018/QH14", "46/2019/QH14"...
_DOC_NUMBER_RE = re.compile(r"\b(\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9.\-]{1,15})\b")

# Điều thuộc nhóm "điều khoản thi hành" — chỉ quét hiệu lực trong các vùng này
# để "thay thế"/"bãi bỏ" trong thân văn bản (nghĩa thường) không gây nhiễu.
_ENFORCEMENT_TITLE_RE = re.compile(
    r"hiệu\s+lực|thi\s+hành|chuyển\s+tiếp|sửa\s+đổi.{0,15}bổ\s+sung|bãi\s+bỏ",
    re.IGNORECASE,
)

# "Nghị định này có hiệu lực (thi hành) (kể) từ ngày ..."
_EFFECTIVE_RE = re.compile(
    rf"{_DOC_TYPES}\s+này\s+có\s+hiệu\s+lực(?:\s+thi\s+hành)?\s*(?:kể\s+)?từ\s+ngày\s+"
    r"(?:(?P<d>\d{1,2})\s+tháng\s+(?P<m>\d{1,2})\s+năm\s+(?P<y>\d{4})"
    r"|(?P<dmy>\d{1,2}[/.-]\d{1,2}[/.-]\d{4})"
    r"|(?P<sign>ký(?:\s+ban\s+hành)?))",
    re.IGNORECASE,
)

_TRIGGER_RE = re.compile(r"hết\s+hiệu\s+lực|thay\s+thế|bãi\s+bỏ", re.IGNORECASE)

# Số hiệu là VĂN BẢN CÔNG CỤ chứ không phải đối tượng ("...đã được sửa đổi, bổ
# sung một số điều theo Luật số 35/2018/QH14") — nhìn ngược tối đa 50 ký tự.
_INSTRUMENT_BACKREF_RE = re.compile(
    rf"theo\s+(?:{_DOC_TYPES}\s+)?số\s*$", re.IGNORECASE
)

# Dấu hiệu "một phần": trước tham chiếu văn bản có giới hạn phạm vi
# ("các quy định liên quan đến X tại...", "khoản 3 Điều 49 của...",
# "thay thế/bãi bỏ (một số) cụm từ...").
_PARTIAL_RE = re.compile(
    r"các\s+quy\s+định|quy\s+định\s+(?:về|liên\s+quan|tại)|"
    r"(?:khoản|điểm|điều|mục|chương)\s+\d|"
    r"cụm\s+từ|một\s+số\s+(?:khoản|điều|điểm)",
    re.IGNORECASE,
)

# Vùng liệt kê VĂN BẢN CÔNG CỤ: "...đã được sửa đổi, bổ sung (một số điều)
# theo Luật số A, Luật số B; ..." — mọi số hiệu từ "theo" tới dấu ';' (hoặc
# hết câu) là văn bản sửa đổi, KHÔNG phải đối tượng bị ảnh hưởng.
_INSTRUMENT_ZONE_RE = re.compile(
    r"sửa\s+đổi[^;]{0,80}?\btheo\b", re.IGNORECASE
)

# Markdown đường admin-layout bọc nội dung trong thẻ <p class="ocr-*">; đưa về
# dạng dòng trơn để find_headings/regex câu hoạt động như văn bản thường.
_HTML_TAG_RE = re.compile(r"<br\s*/?>|</?p[^>]*>", re.IGNORECASE)


@dataclass
class ValidityEvent:
    """Một tuyên bố về hiệu lực của văn bản KHÁC trong văn bản đang xét."""
    kind: str            # 'thay_the' | 'bai_bo' | 'het_hieu_luc'
    target_number: str   # số hiệu văn bản bị ảnh hưởng, vd "24/2018/QH14"
    scope: str           # 'full' | 'partial'
    quote: str           # câu gốc (đã rút gọn) — phục vụ audit/log

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "target_number": self.target_number,
            "scope": self.scope,
            "quote": self.quote,
        }


@dataclass
class ValidityInfo:
    """Kết quả trích xuất hiệu lực của MỘT văn bản."""
    effective_date: str | None = None   # "dd/mm/yyyy" | "sign_date" | None
    events: list[ValidityEvent] = field(default_factory=list)


def extract_validity(markdown: str) -> ValidityInfo:
    """Trích hiệu lực từ markdown đã parse của một văn bản pháp luật.

    Văn bản không có cấu trúc Điều (công văn, tờ trình) trả về ValidityInfo
    rỗng — không đoán mò.
    """
    info = ValidityInfo()
    text = _HTML_TAG_RE.sub("\n", markdown or "")
    if not text.strip():
        return info

    for span in _enforcement_spans(text):
        if info.effective_date is None:
            m = _EFFECTIVE_RE.search(span)
            if m:
                if m.group("sign"):
                    info.effective_date = "sign_date"
                elif m.group("dmy"):
                    info.effective_date = re.sub(r"[.-]", "/", m.group("dmy"))
                else:
                    info.effective_date = (
                        f"{int(m.group('d')):02d}/{int(m.group('m')):02d}/{m.group('y')}"
                    )
        for event in _extract_events(span):
            if all(e.target_number != event.target_number or e.kind != event.kind
                   for e in info.events):
                info.events.append(event)
    return info


def _enforcement_spans(text: str) -> list[str]:
    """Nội dung các Điều thuộc nhóm 'điều khoản thi hành'.

    KHÔNG gate theo vị trí trong văn bản: nghị định kèm phụ lục lớn (vd
    361/2025 — 90% độ dài là phụ lục) có điều thi hành ngay ở ~7% đầu file.
    Mục lục trùng tiêu đề vô hại: span mục lục không chứa câu tuyên bố.
    """
    headings = find_headings(text)
    dieu = [h for h in headings if h.level == 4]
    spans: list[str] = []
    for i, h in enumerate(dieu):
        if not _ENFORCEMENT_TITLE_RE.search(h.title):
            continue
        end = dieu[i + 1].start if i + 1 < len(dieu) else len(text)
        spans.append(text[h.start:end])
    return spans


def _extract_events(span: str) -> list[ValidityEvent]:
    events: list[ValidityEvent] = []
    for sentence in _sentences(span):
        trigger = _TRIGGER_RE.search(sentence)
        if not trigger:
            continue
        kw = re.sub(r"\s+", " ", trigger.group(0)).lower()
        kind = {"hết hiệu lực": "het_hieu_luc", "thay thế": "thay_the",
                "bãi bỏ": "bai_bo"}[kw]
        instrument_zones = []
        for zm in _INSTRUMENT_ZONE_RE.finditer(sentence):
            zone_end = sentence.find(";", zm.end())
            instrument_zones.append(
                (zm.end(), zone_end if zone_end != -1 else len(sentence))
            )
        for num_match in _DOC_NUMBER_RE.finditer(sentence):
            if any(a <= num_match.start() < b for a, b in instrument_zones):
                continue  # văn bản công cụ trong vùng "sửa đổi... theo ..."
            lookback = sentence[max(0, num_match.start() - 50):num_match.start()]
            if _INSTRUMENT_BACKREF_RE.search(lookback):
                continue  # văn bản công cụ ("theo Luật số ..."), không phải đối tượng
            # Marker "một phần" chỉ xét trong ĐOẠN chấm-phẩy chứa số hiệu:
            # câu liệt kê "Luật A ... sửa đổi một số điều theo ...; Luật B hết
            # hiệu lực" không được để marker của đoạn A lây sang đoạn B.
            seg_start = sentence.rfind(";", 0, num_match.start()) + 1
            scope = "partial" if _PARTIAL_RE.search(
                sentence[seg_start:num_match.start()]
            ) else "full"
            quote = re.sub(r"\s+", " ", sentence).strip()
            events.append(ValidityEvent(
                kind=kind,
                target_number=num_match.group(1),
                scope=scope,
                quote=quote[:300],
            ))
    return events


def _sentences(span: str) -> list[str]:
    """Tách câu tuyên bố: theo dấu chấm cuối câu / xuống dòng kép / đầu khoản.

    KHÔNG tách theo ';' — một câu có thể liệt kê nhiều văn bản cùng hết hiệu
    lực ngăn cách bằng chấm phẩy ("Luật A số ...; Luật B số ... hết hiệu lực").
    """
    # Bỏ dòng heading để tiêu đề điều không dính vào câu đầu.
    body = re.sub(r"(?m)^[ \t]*#{0,6}[ \t]*Điều\s+\d+[^\n]*$", "", span)
    parts = re.split(r"(?m)\.\s*(?:\n|$)|\.\s+(?=[A-ZĐÀ-Ỹ0-9])|\n{2,}|^\s*\d+\.\s", body)
    return [p.strip() for p in parts if p and p.strip()]

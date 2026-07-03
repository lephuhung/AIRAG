"""
Suy ``heading_path`` (Phần/Chương/Mục/Điều) từ NỘI DUNG chunk văn bản pháp luật.

Bối cảnh: heading_path vốn lấy từ ``chunk.meta.headings`` của Docling, nhưng
Docling không nhận "Điều N." là heading với văn bản luật VN, và đường OCR/legacy
(``_parse_legacy``) không set gì cả → heading_path rỗng 100% toàn kho, giết chết
nhánh tra cứu cấu trúc của ``search_document_section``. Cấu trúc vẫn còn nguyên
trong text chunk ("## Điều 17. Hồ sơ...") — module này khôi phục nó.

Dùng ở 2 chỗ:
- parse time: ``DeepDocumentParser._parse_legacy`` / ``_chunk_document`` (fallback
  khi Docling không trả headings);
- backfill kho hiện có: ``scripts/backfill_heading_path.py`` (chỉ update metadata
  ChromaDB, không re-embed).

Quy ước heading_path của một chunk: các cấp trên (Phần > Chương > Mục) theo trạng
thái carry-forward, cộng MỌI "Điều N." xuất hiện trong chunk (chunk gộp nhiều điều
ngắn thì liệt kê hết để tra cứu điều nào cũng trúng); chunk không chứa header nào
thì thừa hưởng Điều đang mở từ chunk trước.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Cấp bậc cấu trúc; số nhỏ = cấp cao. Heading mới ở cấp L xoá state các cấp >= L.
_LEVELS = {"phần": 1, "chương": 2, "mục": 3, "điều": 4}
_DIEU_LEVEL = 4

# Một DÒNG heading trong markdown/plain-text đã parse. Khoan dung với thực tế kho:
# prefix # không đồng nhất ("#"/"##"/"###", có khi 2 space sau #), "Chương III"
# số La Mã tiêu đề cùng dòng HOẶC dòng sau ("## Chương V"), "Phần 1."/"PHẦN THỨ
# NHẤT", "Mục 2". Docling chunk text không còn dấu # nên prefix là tuỳ chọn.
_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?"
    r"(?P<kw>Phần|Chương|Mục|Điều)[ \t]+"
    r"(?P<num>thứ[ \t]+\w+|[IVXLCDM]+\b|\d+[a-zA-Z]?)"
    r"(?P<rest>[^\n]*)"
)

# "Điều" chỉ được coi là heading khi sau số là "." hoặc ":" ("Điều 17. Tiêu đề")
# — chặn dòng thân văn bản tình cờ mở đầu bằng tham chiếu ("Điều 5 và Điều 6
# Nghị định này..."). Chương/Mục/Phần ít bị nhiễu hơn nên chấp nhận cả tiêu đề
# ngăn cách bằng khoảng trắng lẫn dòng trống phía sau.
_DIEU_REST_RE = re.compile(r"^[ \t]*[.:]")

_WS_RE = re.compile(r"\s+")

# Chunk gộp rất nhiều điều ngắn (phụ lục biểu mẫu) — trần số component Điều để
# metadata không phình vô hạn.
_MAX_DIEU_PER_CHUNK = 8


@dataclass
class _Heading:
    start: int
    level: int
    title: str


def _iter_headings(text: str) -> list[_Heading]:
    out: list[_Heading] = []
    for m in _HEADING_RE.finditer(text or ""):
        kw = m.group("kw").lower()
        level = _LEVELS[kw]
        rest = m.group("rest") or ""
        if level == _DIEU_LEVEL and not _DIEU_REST_RE.match(rest):
            continue
        title = _WS_RE.sub(" ", f"{m.group('kw')} {m.group('num')}{rest}").strip()
        title = title.rstrip(" .:")  # "Chương V" trần / "Điều 17. Tiêu đề." gọn đuôi
        if title:
            out.append(_Heading(start=m.start(), level=level, title=title))
    return out


def derive_heading_paths(chunk_texts: list[str]) -> list[list[str]]:
    """Trả về heading_path cho TỪNG chunk (cùng thứ tự với ``chunk_texts``).

    Yêu cầu chunk theo ĐÚNG thứ tự văn bản (chunk_index tăng dần) — trạng thái
    Phần/Chương/Mục/Điều được carry-forward qua các chunk không chứa header.
    """
    state: dict[int, str] = {}
    paths: list[list[str]] = []
    for text in chunk_texts:
        headings = _iter_headings(text)
        dieu_at_start = state.get(_DIEU_LEVEL)
        # Chunk có nội dung TRƯỚC header đầu tiên → phần đầu vẫn thuộc Điều đang
        # mở của chunk trước, giữ nó trong path để tra Điều đó vẫn trúng chunk này.
        starts_mid_article = bool(
            headings and dieu_at_start and text[: headings[0].start].strip()
        )
        dieu_in_chunk: list[str] = []
        for h in headings:
            state = {lv: t for lv, t in state.items() if lv < h.level}
            state[h.level] = h.title
            if h.level == _DIEU_LEVEL:
                dieu_in_chunk.append(h.title)

        upper = [state[lv] for lv in (1, 2, 3) if lv in state]
        dieu_comps: list[str] = []
        if dieu_at_start and (not headings or starts_mid_article):
            dieu_comps.append(dieu_at_start)
        dieu_comps.extend(dieu_in_chunk)
        # dedup giữ thứ tự (Điều mở đầu có thể trùng header đầu chunk)
        seen: set[str] = set()
        dieu_comps = [d for d in dieu_comps if not (d in seen or seen.add(d))]
        paths.append(upper + dieu_comps[:_MAX_DIEU_PER_CHUNK])
    return paths

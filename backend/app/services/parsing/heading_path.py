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
# prefix # không đồng nhất ("#"/"##"/"###", có khi 2 space sau #), heading IN ĐẬM
# ("**Điều 12.** Phạm vi" — Docling PDF số thường bold legal heading thay vì #),
# "Chương III" số La Mã tiêu đề cùng dòng HOẶC dòng sau ("## Chương V"),
# "Phần 1."/"PHẦN THỨ NHẤT", "Mục 2". Docling chunk text không còn dấu # nên
# prefix là tuỳ chọn. Số La Mã bắt buộc VIẾT HOA ((?-i:...)) — nếu không, cờ (?i)
# khiến 'c' thường trong "Mục c khoản 2..." match [IVXLCDM].
_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:[*_]{1,3}[ \t]*)?"
    r"(?P<kw>Phần|Chương|Mục|Điều)[ \t]+"
    r"(?P<num>thứ[ \t]+\w+|(?-i:[IVXLCDM]+)\b|\d+[a-zA-Z]?)"
    r"(?P<rest>[^\n]*)"
)

# "Điều" chỉ được coi là heading khi sau số là "." hoặc ":" ("Điều 17. Tiêu đề",
# chấp nhận marker đậm đóng trước dấu: "**Điều 17**. ...") — chặn dòng thân văn
# bản tình cờ mở đầu bằng tham chiếu ("Điều 5 và Điều 6 Nghị định này...").
# "Điều N" TRẦN (OCR rơi dấu chấm) được cứu riêng trong _iter_headings bằng
# lookahead dòng kế tiếp.
_DIEU_REST_RE = re.compile(r"^[ \t]*[*_]{0,3}[ \t]*[.:]")

# Đuôi "rest" trông như TRÍCH DẪN cấu trúc tiếp diễn ("Chương III Nghị định
# này", "khoản 2 Điều 7") — keyword theo sau bởi SỐ. Phân biệt với tiêu đề thật
# chứa keyword nhưng không có số ("Chương IX ĐIỀU KHOẢN THI HÀNH").
_REF_CONT_RE = re.compile(
    r"(?i)^(?:Phần|Chương|Mục|Điều|khoản|điểm)[ \t]+(?:thứ[ \t]+\w+|(?-i:[IVXLCDM]+)\b|\d)"
)

# Marker nhấn mạnh markdown (bold/italic) — loại khỏi title và các phép thử gate.
_EMPH_RE = re.compile(r"[*_]+")

_WS_RE = re.compile(r"\s+")

# Chunk gộp rất nhiều điều ngắn (phụ lục biểu mẫu) — trần số component Điều để
# metadata không phình vô hạn.
_MAX_DIEU_PER_CHUNK = 8


@dataclass
class Heading:
    """Một heading cấu trúc tìm thấy trong text: vị trí, cấp và tiêu đề đã chuẩn hoá."""
    start: int
    level: int
    title: str


# Giữ alias cũ cho nội bộ module
_Heading = Heading

# Component heading_path của một Điều: "Điều 17. Hồ sơ..." / "Điều 5a: ..."
_DIEU_COMPONENT_RE = re.compile(r"(?i)^\s*Điều\s+(\d+[a-zA-Z]?)\b")


def extract_article_nos(heading_path: list[str] | str | None) -> list[str]:
    """Rút SỐ ĐIỀU từ heading_path của chunk → ["17", "18"].

    Nguồn của metadata ``article_nos`` (pipe-separated) trên chunk ChromaDB —
    cho ``search_document_section`` match chính xác "Điều 17" không cần regex
    trên chuỗi heading_path (né luôn lỗi "Điều 3" trúng "Điều 30").
    """
    if isinstance(heading_path, str):
        components = [c.strip() for c in heading_path.split(">")]
    else:
        components = [str(c) for c in (heading_path or [])]
    out: list[str] = []
    for comp in components:
        m = _DIEU_COMPONENT_RE.match(comp)
        if m:
            no = m.group(1).lower()
            if no not in out:
                out.append(no)
    return out


def find_headings(text: str) -> list[Heading]:
    """Public API: liệt kê heading Phần/Chương/Mục/Điều trong ``text`` theo thứ tự.

    Dùng bởi ``derive_heading_paths`` (suy path per-chunk) và
    ``LegalDocumentChunker`` (cắt chunk theo ranh giới cấu trúc).
    """
    return _iter_headings(text)


def _upper_rest_ok(rest: str) -> bool:
    """Gate cho Phần/Chương/Mục: đuôi dòng phải giống TIÊU ĐỀ, không phải trích dẫn.

    Chấp nhận: rỗng ("Mục 1"), dấu câu ("Phần 1. ..."), chữ hoa ("Chương II
    NHỮNG QUY ĐỊNH CHUNG"). Bác bỏ: chữ thường ("Chương V của Luật này..." —
    trích dẫn bị wrap dòng) và tham chiếu cấu trúc nối tiếp ("Mục 2 Chương III
    Nghị định này").
    """
    rest = _EMPH_RE.sub("", rest).strip()
    if not rest:
        return True
    if rest[0] in ".:-–—":
        return True
    if _REF_CONT_RE.match(rest):
        return False
    return not rest[0].islower()


def _dieu_rest_ok(rest: str, text: str, match_end: int) -> bool:
    """Gate cho Điều: có [.:] sau số, HOẶC "Điều N" trần cuối dòng mà dòng kế
    tiếp mở đầu bằng chữ hoa (tiêu đề bị OCR tách dòng / rơi dấu chấm)."""
    if _DIEU_REST_RE.match(rest):
        return True
    if _EMPH_RE.sub("", rest).strip():
        return False  # còn chữ sau số nhưng không phải [.:] → tham chiếu
    # rest rỗng → nhìn dòng kế tiếp: "Phạm vi điều chỉnh" (hoa) = tiêu đề,
    # "và khoản 2..." (thường) / hết text = trích dẫn wrap dòng → bỏ.
    for line in text[match_end:].split("\n")[1:]:
        s = _EMPH_RE.sub("", line).strip()
        if s:
            return s[0].isupper() and not _REF_CONT_RE.match(s)
    return False


def _iter_headings(text: str) -> list[_Heading]:
    text = text or ""
    out: list[_Heading] = []
    for m in _HEADING_RE.finditer(text):
        kw = m.group("kw").lower()
        level = _LEVELS[kw]
        rest = m.group("rest") or ""
        if level == _DIEU_LEVEL:
            if not _dieu_rest_ok(rest, text, m.end()):
                continue
        elif not _upper_rest_ok(rest):
            continue
        title = _EMPH_RE.sub(" ", f"{m.group('kw')} {m.group('num')}{rest}")
        title = _WS_RE.sub(" ", title).strip()
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

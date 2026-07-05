"""
Unit tests: LegalDocumentChunker + derive_heading_paths (pure functions, no stack).

Guards the legal-structure chunking added 2026-07-03: chunks must break at
Phần/Chương/Mục/Điều boundaries (never mix the tail of one điều with the head
of the next), long articles size-split WITHIN their section, and non-legal
documents (công văn) must not trigger legal chunking at all.
"""
from __future__ import annotations

import re

from app.services.embedding.chunker import LegalDocumentChunker
from app.services.parsing.heading_path import derive_heading_paths, find_headings

LEGAL_MD = """QUỐC HỘI
CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM

# Chương I QUY ĐỊNH CHUNG

## Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về việc kiểm thử hệ thống.

## Điều 2. Đối tượng áp dụng
Áp dụng với cơ quan, tổ chức, cá nhân liên quan.

# Chương II TRÌNH TỰ THỰC HIỆN

## Điều 3. Hồ sơ đề nghị
""" + ("Nội dung khoản rất dài. " * 200) + """

## Điều 4. Hiệu lực thi hành
Nghị định có hiệu lực từ ngày ký.
"""

CONG_VAN = """UBND TỈNH
Kính gửi: Các sở, ban, ngành.
Về việc triển khai kế hoạch quý III.
Đề nghị các đơn vị nghiêm túc thực hiện.
"""

_DIEU_HDR = re.compile(r"(?m)^#{0,6}\s*Điều\s+\d+[a-zA-Z]?\s*[.:]")


def test_detects_legal_structure():
    assert LegalDocumentChunker.has_legal_structure(LEGAL_MD)
    assert not LegalDocumentChunker.has_legal_structure(CONG_VAN)


def test_chunks_never_mix_articles():
    chunks = LegalDocumentChunker(max_chars=1800).split_text(LEGAL_MD, source="t.md")
    assert chunks, "phải ra chunk"
    for c in chunks:
        m = _DIEU_HDR.search(c.content)
        if m:
            # Header Điều chỉ được đứng ĐẦU chunk — không dính nội dung điều trước
            assert not c.content[: m.start()].strip(), (
                f"chunk {c.chunk_index} trộn nội dung trước header: "
                f"{c.content[:80]!r}"
            )
        # Mỗi chunk chứa tối đa 1 header Điều (điều ngắn không bị gộp)
        assert len(_DIEU_HDR.findall(c.content)) <= 1


def test_long_article_splits_within_section():
    chunks = LegalDocumentChunker(max_chars=1800).split_text(LEGAL_MD, source="t.md")
    d3 = [c for c in chunks if "Điều 3" in c.content or "Nội dung khoản rất dài" in c.content]
    assert len(d3) >= 2, "Điều 3 dài phải được size-split thành nhiều chunk"
    # Không sub-chunk nào của Điều 3 tràn sang Điều 4
    assert all("Điều 4" not in c.content for c in d3)
    # char offsets phải khớp text gốc tuyệt đối
    for c in chunks:
        assert LEGAL_MD[c.char_start:c.char_end] == c.content


def test_heading_paths_carry_forward():
    chunks = LegalDocumentChunker(max_chars=1800).split_text(LEGAL_MD, source="t.md")
    paths = derive_heading_paths([c.content for c in chunks])
    by_content = dict(zip([c.content for c in chunks], paths))
    for content, path in by_content.items():
        if "Nội dung khoản rất dài" in content:
            # kể cả sub-chunk giữa Điều 3 (không chứa header) phải mang path Điều 3
            assert any("Điều 3" in comp for comp in path), path
            assert any("Chương II" in comp for comp in path), path


def test_find_headings_ignores_body_references():
    # Tham chiếu trong thân văn bản ("Điều 5 và Điều 6 Nghị định...") không có
    # dấu chấm sau số → KHÔNG phải heading
    text = "Điều 5 và Điều 6 Nghị định này quy định chi tiết.\n## Điều 7. Tiêu đề\nNội dung."
    titles = [h.title for h in find_headings(text)]
    assert titles == ["Điều 7. Tiêu đề"]

"""
Unit tests: LegalDocumentChunker + derive_heading_paths (pure functions, no stack).

Guards the legal-structure chunking added 2026-07-03: chunks must break at
Phần/Chương/Mục/Điều boundaries (never mix the tail of one điều with the head
of the next), long articles size-split WITHIN their section, and non-legal
documents (công văn) must not trigger legal chunking at all.
"""
from __future__ import annotations

import re

from app.services.embedding.chunker import DocumentChunker, LegalDocumentChunker
from app.services.parsing.heading_path import (
    SubdivisionMetadata,
    derive_heading_paths,
    derive_subdivision_metadata,
    find_headings,
    find_subdivisions,
    parse_subdivisions,
)

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


def test_parse_subdivisions_basic_khoan_diem():
    text = (
        "## Điều 8. Hồ sơ\n"
        "1. Khoản một nội dung.\n"
        "a) điểm a;\n"
        "đ) điểm đ;\n"
        "i) điểm i;\n"
        "2. Khoản hai nội dung.\n"
        "a) điểm a khoản 2;\n"
        "b) điểm b khoản 2.\n"
    )
    res = parse_subdivisions(text)
    kinds = [(s.kind, s.label, s.article_no, s.parent_khoan) for s in res.subdivisions]
    assert kinds == [
        ("khoan", "1", "8", None),
        ("diem", "a", "8", "1"),
        ("diem", "đ", "8", "1"),
        ("diem", "i", "8", "1"),
        ("khoan", "2", "8", None),
        ("diem", "a", "8", "2"),
        ("diem", "b", "8", "2"),
    ]
    assert res.stats.khoan_candidates == 2
    assert res.stats.khoan_accepted == 2
    assert res.stats.diem_candidates == 5
    assert res.stats.diem_accepted == 5
    assert res.stats.ambiguous_rejected == 0
    assert find_subdivisions(text) == list(res.subdivisions)


def test_parse_subdivisions_docling_bullet_diem():
    # Docling export điểm dưới dạng markdown list ("- a) ...") — candidate regex
    # phải chấp nhận bullet prefix; state machine validate như marker trần.
    text = (
        "## Điều 8. Hồ sơ\n"
        "1. Khoản một nội dung.\n"
        "- a) điểm a;\n"
        "- b) điểm b;\n"
        "2. Khoản hai nội dung.\n"
        "- a) điểm a khoản 2.\n"
    )
    res = parse_subdivisions(text)
    kinds = [(s.kind, s.label, s.article_no, s.parent_khoan) for s in res.subdivisions]
    assert kinds == [
        ("khoan", "1", "8", None),
        ("diem", "a", "8", "1"),
        ("diem", "b", "8", "1"),
        ("khoan", "2", "8", None),
        ("diem", "a", "8", "2"),
    ]
    assert res.stats.diem_candidates == 3
    assert res.stats.diem_accepted == 3
    assert res.stats.ambiguous_rejected == 0


def test_parse_subdivisions_vbhn_footnote_glue():
    # VBHN: superscript chú thích dính vào marker Điểm — "3. đ)" là footnote 3 +
    # điểm đ (khoản thật không mở đầu bằng marker chữ cái). Số ảo này không được
    # tạo khoản phantom làm gãy chuỗi điểm/khoản thật phía sau.
    text = (
        "## Điều 4. Giải thích từ ngữ\n"
        "1. Trong Bộ luật này, các từ ngữ dưới đây được hiểu như sau:\n"
        "- a) Cơ quan có thẩm quyền;\n"
        "- b) Người có thẩm quyền;\n"
        "- c) Người tham gia tố tụng;\n"
        "- d) Nguồn tin về tội phạm;\n"
        "3. đ) Người bị buộc tội gồm người bị bắt, bị can, bị cáo.\n"
        "- e) Người thân thích;\n"
        "8. g ) Đương sự.\n"
        "2. Trong Bộ luật này, những từ ngữ dưới đây được gọi như sau:\n"
        "## a) 3 (được bãi bỏ)\n"
        "- b) Cơ quan điều tra cấp tỉnh.\n"
    )
    res = parse_subdivisions(text)
    kinds = [(s.kind, s.label, s.parent_khoan) for s in res.subdivisions]
    assert kinds == [
        ("khoan", "1", None),
        ("diem", "a", "1"),
        ("diem", "b", "1"),
        ("diem", "c", "1"),
        ("diem", "d", "1"),
        ("diem", "đ", "1"),   # cứu từ "3. đ)" — footnote 3 bị bỏ qua
        ("diem", "e", "1"),
        ("diem", "g", "1"),   # cứu từ "8. g )" — paren tách bởi space
        ("khoan", "2", None), # khoản 2 thật không bị phantom "3" chặn
        ("diem", "a", "2"),   # cứu từ "## a) 3 (được bãi bỏ)" — điểm bị bãi bỏ
        ("diem", "b", "2"),
    ]
    assert res.stats.khoan_candidates == 2
    assert res.stats.khoan_accepted == 2


def test_parse_subdivisions_docling_heading_marker():
    # Docling render khoản/điểm in đậm thành heading: "## 1." là khoản thật,
    # "## a) 3 (được bãi bỏ)" là điểm bị bãi bỏ (vẫn chiếm vị trí chữ cái).
    text = (
        "## Điều 9. Quyền\n"
        "## 1. Người tố cáo có quyền:\n"
        "## a) 3 (được bãi bỏ)\n"
        "- b) Yêu cầu giữ bí mật họ tên.\n"
        "2. Nghĩa vụ khác.\n"
    )
    res = parse_subdivisions(text)
    kinds = [(s.kind, s.label, s.parent_khoan) for s in res.subdivisions]
    assert kinds == [
        ("khoan", "1", None),
        ("diem", "a", "1"),
        ("diem", "b", "1"),
        ("khoan", "2", None),
    ]


def test_parse_subdivisions_parent_refs_exact():
    text = (
        "## Điều 8. Hồ sơ\n"
        "1. Khoản một.\n"
        "a) điểm a;\n"
        "2. Khoản hai.\n"
        "a) điểm a k2;\n"
        "b) điểm b k2.\n"
    )
    metadata = derive_subdivision_metadata([text])[0]
    assert metadata.subdivision_refs == (
        "khoan:1",
        "khoan:1/diem:a",
        "khoan:2",
        "khoan:2/diem:a",
        "khoan:2/diem:b",
    )
    assert metadata.khoan_nos == ("1", "2")
    assert metadata.diem_labels == ("a", "b")


def test_parse_subdivisions_state_resets():
    text = (
        "## Điều 1. Một\n"
        "1. Khoản một điều 1.\n"
        "a) điểm a;\n"
        "2. Khoản hai điều 1.\n"
        "a) điểm a k2 điều 1 — điểm reset theo khoản mới;\n"
        "## Điều 2. Hai\n"
        "1. Khoản một điều 2 — khoản reset theo điều mới.\n"
        "a) điểm a điều 2.\n"
    )
    subs = parse_subdivisions(text).subdivisions
    assert [(s.kind, s.label, s.article_no) for s in subs] == [
        ("khoan", "1", "1"),
        ("diem", "a", "1"),
        ("khoan", "2", "1"),
        ("diem", "a", "1"),
        ("khoan", "1", "2"),
        ("diem", "a", "2"),
    ]


def test_parse_subdivisions_forward_jump_and_first_must_be_one():
    text = "## Điều 1. T\n1. một\n3. ba — nhảy vọt chấp nhận\n"
    subs = find_subdivisions(text)
    assert [(s.kind, s.label) for s in subs] == [("khoan", "1"), ("khoan", "3")]

    res = parse_subdivisions("## Điều 1. T\n3. ba\n")
    assert res.subdivisions == ()
    assert res.stats.khoan_candidates == 1
    assert res.stats.khoan_accepted == 0
    assert res.stats.ambiguous_rejected == 1


def test_parse_subdivisions_negative_candidates_inside_dieu():
    text = (
        "## Điều 5. Năm\n"
        "1. Khoản một hợp lệ.\n"
        "1. Lặp lại số một — equal bị loại.\n"
        "0. Số không ngoài dải.\n"
        "45. Số quá 30 (năm/số hiệu) bị loại.\n"
        "    2. Thụt 4 space — không phải candidate.\n"
        "> 3. Blockquote — không phải candidate.\n"
        "Theo khoản 2 Điều 7 thì mid-sentence ref không tính.\n"
        "2. Khoản hai hợp lệ.\n"
        "b) điểm b đầu tiên — phải là a nên bị loại.\n"
        "z) chữ cái ngoài bảng tiếng Việt — bị loại.\n"
        "a) điểm a hợp lệ.\n"
    )
    res = parse_subdivisions(text)
    assert [(s.kind, s.label) for s in res.subdivisions] == [
        ("khoan", "1"),
        ("khoan", "2"),
        ("diem", "a"),
    ]
    assert res.stats.khoan_candidates == 5
    assert res.stats.khoan_accepted == 2
    assert res.stats.diem_candidates == 3
    assert res.stats.diem_accepted == 1
    assert res.stats.ambiguous_rejected == 5


def test_parse_subdivisions_inactive_outside_dieu():
    res = parse_subdivisions("1. một\na) điểm a\n")
    assert res.subdivisions == ()
    assert res.stats.khoan_candidates == 1
    assert res.stats.diem_candidates == 1


def test_derive_subdivision_metadata_continuation_inherits():
    chunks = [
        "## Điều 3. Ba\n1. Khoản một mở đầu.\na) điểm a;\n",
        "nội dung tiếp của khoản 1, không marker nào",
        "2. Khoản hai.\n",
    ]
    metas = derive_subdivision_metadata(chunks)
    assert metas[0].subdivision_refs == ("khoan:1", "khoan:1/diem:a")
    assert metas[1].subdivision_refs == ("khoan:1", "khoan:1/diem:a")
    assert metas[1].khoan_nos == ("1",)
    assert metas[1].diem_labels == ("a",)
    assert metas[2].subdivision_refs == ("khoan:2",)


def test_derive_subdivision_metadata_heading_paths_fallback():
    chunks = ["1. Khoản một của điều năm.\na) điểm a.\n"]
    paths = [["Chương II", "Điều 5. Hiệu lực"]]
    metas = derive_subdivision_metadata(chunks, paths)
    assert metas[0].subdivision_refs == ("khoan:1", "khoan:1/diem:a")

    metas_no_path = derive_subdivision_metadata(chunks)
    assert metas_no_path[0].subdivision_refs == ()


def test_derive_subdivision_metadata_rejects_ambiguous_heading_path():
    metas = derive_subdivision_metadata(
        ["1. Không được gán mơ hồ.\na) Điểm a.\n"],
        [["Điều 5. Một", "Điều 6. Hai"]],
    )
    assert metas[0] == SubdivisionMetadata()


_LONG = "Nội dung rất dài để vượt ngưỡng max chars của chunker. " * 8


def _legal_doc(d2_body: str) -> str:
    return (
        "## Điều 1. Phạm vi\nNội dung điều một.\n\n"
        "## Điều 2. Quy định\n" + d2_body + "\n\n"
        "## Điều 3. Hiệu lực\nHiệu lực thi hành.\n"
    )


def test_short_khoan_never_cut_and_packed():
    body = (
        "1. " + "Khoản một ngắn gọn. " * 6 + "\n"
        "2. " + "Khoản hai ngắn gọn. " * 6 + "\n"
        "3. " + "Khoản ba ngắn gọn. " * 6 + "\n"
    )
    doc = _legal_doc(body)
    chunks = LegalDocumentChunker(max_chars=300).split_text(doc, source="t.md")
    for c in chunks:
        assert doc[c.char_start:c.char_end] == c.content
    d2 = [c for c in chunks if "Quy định" in c.content or "Khoản" in c.content]
    for label in ("Khoản một", "Khoản hai", "Khoản ba"):
        owners = [c for c in d2 if label in c.content]
        assert len(owners) == 1
        assert owners[0].content.count("Khoản") >= 1
    assert len(d2) < 4
    packed = [c for c in d2 if len(c.metadata["khoan_nos"]) > 1]
    assert packed, "phải có chunk gộp nhiều khoản"
    for c in packed:
        for k in c.metadata["khoan_nos"]:
            assert f"khoan:{k}" in c.metadata["subdivision_refs"]
        assert c.metadata["subdivision_schema_version"] == 1


def test_long_khoan_subchunks_inherit_khoan():
    body = (
        "1. Khoản một ngắn.\n"
        "2. " + _LONG * 4 + "\n"
        "3. Khoản ba ngắn.\n"
    )
    doc = _legal_doc(body)
    chunks = LegalDocumentChunker(max_chars=600).split_text(doc)
    k2 = [c for c in chunks if "khoan:2" in c.metadata["subdivision_refs"]]
    assert len(k2) >= 2, "khoản 2 dài phải được sub-split"
    cont = [c for c in k2 if not re.search(r"(?m)^2\.\s", c.content)]
    assert cont, "phải có continuation chunk của khoản 2"
    for c in cont:
        assert c.metadata["khoan_nos"] == ["2"]
    k3 = [c for c in chunks if "khoan:3" in c.metadata["subdivision_refs"]]
    assert len(k3) == 1
    assert "Nội dung rất dài" not in k3[0].content or k3[0].content.index(
        "Nội dung rất dài"
    ) > k3[0].content.index("3.")
    for c in chunks:
        assert doc[c.char_start:c.char_end] == c.content


def test_long_diem_subchunks_inherit_refs():
    body = (
        "1. Khoản một.\n"
        "a) điểm a ngắn;\n"
        "2. Khoản hai.\n"
        "a) " + _LONG * 4 + "\n"
        "3. Khoản ba.\n"
    )
    doc = _legal_doc(body)
    chunks = LegalDocumentChunker(max_chars=600).split_text(doc)
    d2a = [c for c in chunks if "khoan:2/diem:a" in c.metadata["subdivision_refs"]]
    assert len(d2a) >= 2, "điểm a khoản 2 dài phải được sub-split"
    cont = [c for c in d2a if not re.search(r"(?m)^a\)\s", c.content)]
    assert cont, "phải có continuation chunk của điểm a"
    for c in cont:
        assert c.metadata["khoan_nos"] == ["2"]
        assert c.metadata["diem_labels"] == ["a"]
        assert c.metadata["subdivision_refs"] == ["khoan:2", "khoan:2/diem:a"]
    for c in chunks:
        assert doc[c.char_start:c.char_end] == c.content


def test_no_cross_product_refs():
    body = (
        "1. Khoản một.\n"
        "a) điểm a k1;\n"
        "b) điểm b k1;\n"
        "2. Khoản hai.\n"
        "a) điểm a k2.\n"
    )
    doc = _legal_doc(body)
    chunks = LegalDocumentChunker(max_chars=1800).split_text(doc)
    both = [
        c for c in chunks
        if "khoan:1/diem:b" in c.metadata["subdivision_refs"]
        and "khoan:2" in c.metadata["subdivision_refs"]
    ]
    assert both, "kỳ vọng một chunk chứa cả khoản 1 lẫn khoản 2"
    for c in both:
        assert "khoan:2/diem:b" not in c.metadata["subdivision_refs"]
        assert "khoan:2/diem:a" in c.metadata["subdivision_refs"]


def test_oversized_without_khoan_falls_back():
    doc = _legal_doc("Không có khoản nào. " + _LONG * 4 + "\n")
    chunks = LegalDocumentChunker(max_chars=600).split_text(doc)
    d2 = [
        c
        for c in chunks
        if "Không có khoản" in c.content or "Nội dung rất dài" in c.content
    ]
    assert len(d2) >= 2
    for c in chunks:
        assert doc[c.char_start:c.char_end] == c.content
        assert c.metadata["subdivision_schema_version"] == 1
    assert all(c.metadata["khoan_nos"] == [] for c in d2)


def test_non_legal_chunker_unchanged():
    chunks = DocumentChunker(chunk_size=200, chunk_overlap=20).split_text(
        "Văn bản thường. " * 60
    )
    assert chunks
    for c in chunks:
        assert "subdivision_schema_version" not in c.metadata
        assert "khoan_nos" not in c.metadata

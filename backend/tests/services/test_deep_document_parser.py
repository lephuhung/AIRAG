"""Pure tests: Docling flat chunk export + legal chunking path.

No real Docling, model, or GPU — fake documents/converters + monkeypatch only.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.config import settings
from app.services.models.parsed_document import (
    EnrichedChunk,
    ExtractedImage,
    ExtractedTable,
)
from app.services.parsing.deep_document_parser import (
    _DOCLING_CHUNK_PAGE_BREAK,
    DeepDocumentParser,
    _contains_markdown_table,
)


class _FakeDoc:
    def __init__(
        self,
        export_text: str,
        supports_placeholder: bool = True,
        pages=None,
    ):
        self._export_text = export_text
        self._supports_placeholder = supports_placeholder
        self.pages = pages if pages is not None else {}
        self.pictures = []
        self.tables = []

    def export_to_markdown(self, page_break_placeholder=None):
        if page_break_placeholder is not None:
            if not self._supports_placeholder:
                raise TypeError("unexpected keyword argument")
            return self._export_text.replace("<PB>", page_break_placeholder)
        return self._export_text.replace("<PB>", "\n\n---\n\n")


class _FakeConvResult:
    def __init__(self, doc):
        self.document = doc


class _FakeConverter:
    def __init__(self, doc):
        self._doc = doc

    def convert(self, _path):
        return _FakeConvResult(self._doc)


@pytest.fixture
def parser(tmp_path):
    return DeepDocumentParser(workspace_id=1, output_dir=tmp_path)


@pytest.fixture(autouse=True)
def _no_image_extraction(monkeypatch):
    monkeypatch.setattr(settings, "HRAG_ENABLE_IMAGE_EXTRACTION", False)


_DIEU_HDR = re.compile(r"(?m)^#{0,6}\s*Điều\s+\d+[a-zA-Z]?\s*[.:]")


def test_export_chunk_markdown_enumerates_pages_and_spans(parser):
    doc = _FakeDoc("Trang một nội dung<PB>Trang hai nội dung")
    content, spans = parser._export_chunk_markdown(doc)
    assert _DOCLING_CHUNK_PAGE_BREAK not in content
    assert [p for p, _, _ in spans] == [1, 2]
    for pno, start, end in spans:
        part = content[start:end]
        assert part.startswith(f"<!-- page {pno} -->\n\n")
    assert spans[1][1] == spans[0][2] + 2
    assert content[spans[0][1] : spans[0][2]].endswith("Trang một nội dung")
    assert content[spans[1][1] : spans[1][2]].endswith("Trang hai nội dung")


def test_export_chunk_markdown_horizontal_rule_not_page_break(parser):
    doc = _FakeDoc("Đoạn trên\n\n---\n\nĐoạn dưới")
    content, spans = parser._export_chunk_markdown(doc)
    assert len(spans) == 1
    assert spans[0][0] == 1
    assert "---" in content


def test_export_chunk_markdown_typeerror_fallback(parser):
    doc = _FakeDoc("Văn bản không placeholder", supports_placeholder=False)
    content, spans = parser._export_chunk_markdown(doc)
    assert spans == []
    assert content == "Văn bản không placeholder"


def test_export_chunk_markdown_empty_export_safe(parser):
    content, spans = parser._export_chunk_markdown(_FakeDoc(""))
    assert spans == [(1, 0, len(content))]


def test_contains_markdown_table():
    assert _contains_markdown_table("| a | b |\n|---|---|\n| 1 | 2 |")
    assert not _contains_markdown_table("Câu có | dấu gạch đứng bình thường.")
    assert not _contains_markdown_table("| a | b |\nkhông phải separator")


def _two_page_legal_doc() -> _FakeDoc:
    text = (
        "QUỐC HỘI\nCỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"
        "<PB>"
        "Điều 1. Một\nNội dung điều một.\n\n"
        "Điều 2. Hai\nNội dung điều hai.\n\n"
        "Điều 3. Ba\nNội dung điều ba.\n"
    )
    return _FakeDoc(text, pages={1: object(), 2: object()})


def test_legal_chunks_article_contained_and_page_spans(parser, monkeypatch):
    monkeypatch.setattr(settings, "HRAG_LEGAL_CHUNK_MAX_CHARS", 1800)
    doc = _two_page_legal_doc()
    chunk_markdown, page_spans = parser._export_chunk_markdown(doc)
    chunks = parser._chunk_legal_markdown(
        chunk_markdown, page_spans, 42, "luat.md"
    )
    assert chunks
    for c in chunks:
        assert len(_DIEU_HDR.findall(c.content)) <= 1
    preamble = chunks[0]
    assert "QUỐC HỘI" in preamble.content
    assert preamble.page_no == 1
    d1 = [c for c in chunks if "Điều 1" in c.content][0]
    assert d1.page_no == 2
    d3 = [c for c in chunks if "Điều 3" in c.content][0]
    assert d3.page_no == 2
    assert d1.heading_path and any("Điều 1" in h for h in d1.heading_path)
    assert d1.contextualized.startswith(" > ".join(d1.heading_path))


def test_preamble_marker_only_overlap_steals_no_page2_assets(
    parser, monkeypatch
):
    monkeypatch.setattr(settings, "HRAG_LEGAL_CHUNK_MAX_CHARS", 1800)
    doc = _two_page_legal_doc()
    chunk_markdown, page_spans = parser._export_chunk_markdown(doc)
    images = [
        ExtractedImage("img-p1", 42, 1, "/tmp/a.png", caption="hình một"),
        ExtractedImage("img-p2", 42, 2, "/tmp/b.png", caption="hình hai"),
    ]
    tables = [
        ExtractedTable("tbl-p2", 42, 2, "| a |\n|---|\n| 1 |", caption="bảng hai"),
    ]
    chunks = parser._chunk_legal_markdown(
        chunk_markdown, page_spans, 42, "luat.md", images, tables
    )
    preamble = chunks[0]
    assert "QUỐC HỘI" in preamble.content
    assert "<!-- page 2 -->" in preamble.content
    assert preamble.page_no == 1
    assert preamble.image_refs == ["img-p1"]
    assert preamble.table_refs == []
    d1 = [c for c in chunks if "Điều 1" in c.content][0]
    assert d1.page_no == 2
    assert d1.image_refs == ["img-p2"]
    assert d1.table_refs == ["tbl-p2"]
    assert "[Image on page 2]: hình hai" in d1.content
    all_img = [r for c in chunks for r in c.image_refs]
    all_tbl = [r for c in chunks for r in c.table_refs]
    assert sorted(all_img) == ["img-p1", "img-p2"]
    assert all_tbl == ["tbl-p2"]


def test_legal_chunk_spanning_pages_gets_assets_once(parser, monkeypatch):
    monkeypatch.setattr(settings, "HRAG_LEGAL_CHUNK_MAX_CHARS", 1800)
    doc = _FakeDoc(
        "Điều 1. Một\nNội dung điều một phần đầu"
        "<PB>"
        "nội dung điều một phần cuối.\n\n"
        "Điều 2. Hai\nNội dung điều hai.\n\n"
        "Điều 3. Ba\nNội dung điều ba.\n",
        pages={1: object(), 2: object()},
    )
    chunk_markdown, page_spans = parser._export_chunk_markdown(doc)
    images = [
        ExtractedImage("img-p1", 42, 1, "/tmp/a.png", caption="hình một"),
        ExtractedImage("img-p2", 42, 2, "/tmp/b.png", caption="hình hai"),
    ]
    tables = [
        ExtractedTable("tbl-p2", 42, 2, "| a |\n|---|\n| 1 |", caption="bảng hai"),
    ]
    chunks = parser._chunk_legal_markdown(
        chunk_markdown, page_spans, 42, "luat.md", images, tables
    )
    spanning = [c for c in chunks if "Điều 1" in c.content][0]
    assert "phần cuối" in spanning.content
    assert spanning.page_no == 1
    assert sorted(spanning.image_refs) == ["img-p1", "img-p2"]
    assert spanning.table_refs == ["tbl-p2"]
    assert "[Image on page 1]: hình một" in spanning.content
    assert "[Image on page 2]: hình hai" in spanning.content
    assert "[Table on page 2 (0x0)]: bảng hai" in spanning.content
    all_img = [r for c in chunks for r in c.image_refs]
    all_tbl = [r for c in chunks for r in c.table_refs]
    assert sorted(all_img) == ["img-p1", "img-p2"]
    assert all_tbl == ["tbl-p2"]


def test_legal_chunk_no_spans_page_zero(parser, monkeypatch):
    monkeypatch.setattr(settings, "HRAG_LEGAL_CHUNK_MAX_CHARS", 1800)
    md = (
        "Điều 1. Một\nA.\n\nĐiều 2. Hai\nB.\n\nĐiều 3. Ba\nC.\n"
    )
    chunks = parser._chunk_legal_markdown(md, [], 7, "x.md")
    assert all(c.page_no == 0 for c in chunks)


def _run_parse_with_docling(parser, monkeypatch, doc, preserve_layout):
    monkeypatch.setattr(settings, "HRAG_DOCLING_PRESERVE_LAYOUT", preserve_layout)
    monkeypatch.setattr(settings, "HRAG_LEGAL_CHUNKING", True)
    monkeypatch.setattr(
        parser, "_get_converter", lambda **_: _FakeConverter(doc)
    )
    monkeypatch.setattr(
        parser,
        "_export_layout_markdown",
        lambda d, urls: "<html>LAYOUT SENTINEL</html>",
    )
    return parser._parse_with_docling(Path("/tmp/fake.pdf"), 42, "luat.pdf")


@pytest.mark.parametrize("preserve_layout", [False, True])
def test_parse_with_docling_legal_uses_flat_chunk_source(
    parser, monkeypatch, preserve_layout
):
    result = _run_parse_with_docling(
        parser, monkeypatch, _two_page_legal_doc(), preserve_layout
    )
    assert result.parser == "docling"
    if preserve_layout:
        assert result.markdown == "<html>LAYOUT SENTINEL</html>"
    else:
        assert result.markdown != "<html>LAYOUT SENTINEL</html>"
        assert "Điều 1" in result.markdown
    assert result.chunks
    for c in result.chunks:
        assert len(_DIEU_HDR.findall(c.content)) <= 1
    assert any(
        h for c in result.chunks for h in c.heading_path if "Điều" in h
    )


def test_parse_with_docling_layout_html_never_chunked(parser, monkeypatch):
    doc = _two_page_legal_doc()
    seen = {}
    orig = DeepDocumentParser._chunk_legal_markdown

    def spy(self, markdown, page_spans, *a, **k):
        seen["markdown"] = markdown
        return orig(self, markdown, page_spans, *a, **k)

    monkeypatch.setattr(DeepDocumentParser, "_chunk_legal_markdown", spy)
    _run_parse_with_docling(parser, monkeypatch, doc, preserve_layout=True)
    assert "<html>" not in seen["markdown"]


def test_parse_with_docling_non_legal_delegates(parser, monkeypatch):
    doc = _FakeDoc("Công văn thông thường, không cấu trúc điều khoản.")
    sentinel = EnrichedChunk(
        content="x", chunk_index=0, source_file="cv.pdf", document_id=9
    )
    called = {}

    def fake_chunk_document(self, d, document_id, filename, images, tables):
        called["hit"] = True
        return [sentinel]

    monkeypatch.setattr(settings, "HRAG_LEGAL_CHUNKING", True)
    monkeypatch.setattr(parser, "_get_converter", lambda **_: _FakeConverter(doc))
    monkeypatch.setattr(
        DeepDocumentParser, "_chunk_document", fake_chunk_document
    )
    result = parser._parse_with_docling(Path("/tmp/fake.pdf"), 9, "cv.pdf")
    assert called.get("hit")
    assert result.chunks == [sentinel]


def test_chunk_legal_markdown_carries_subdivision_metadata(parser, monkeypatch):
    """TextChunk.metadata subdivision keys land on the EnrichedChunk fields."""
    monkeypatch.setattr(settings, "HRAG_LEGAL_CHUNK_MAX_CHARS", 1800)
    md = (
        "Điều 1. Một\n"
        "1. Khoản một điều một.\n"
        "a) điểm a khoản một;\n"
        "2. Khoản hai điều một.\n\n"
        "Điều 2. Hai\nNội dung điều hai.\n\n"
        "Điều 3. Ba\nNội dung điều ba.\n"
    )
    chunks = parser._chunk_legal_markdown(md, [], 7, "x.md")
    assert chunks
    assert all(c.subdivision_schema_version == 1 for c in chunks)
    d1 = [c for c in chunks if "Khoản một" in c.content]
    assert d1
    assert d1[0].khoan_nos == ["1", "2"]
    assert d1[0].diem_labels == ["a"]
    assert d1[0].subdivision_refs == [
        "khoan:1",
        "khoan:1/diem:a",
        "khoan:2",
    ]
    d2 = [c for c in chunks if "Điều 2" in c.content]
    assert d2
    assert d2[0].khoan_nos == []
    assert d2[0].diem_labels == []
    assert d2[0].subdivision_refs == []
    assert d2[0].subdivision_schema_version == 1


def test_parse_legacy_legal_chunks_carry_subdivision_metadata(
    parser, tmp_path, monkeypatch
):
    """The OCR/legacy wrapper copies TextChunk.metadata into EnrichedChunk."""
    monkeypatch.setattr(settings, "HRAG_LEGAL_CHUNKING", True)
    md = (
        "Điều 1. Một\n"
        "1. Khoản một điều một.\n"
        "a) điểm a khoản một;\n"
        "2. Khoản hai điều một.\n\n"
        "Điều 2. Hai\nNội dung điều hai.\n\n"
        "Điều 3. Ba\nNội dung điều ba.\n"
    )
    path = tmp_path / "luat.md"
    path.write_text(md, encoding="utf-8")
    result = parser._parse_legacy(path, 7, "luat.md")
    assert result.chunks
    assert all(c.subdivision_schema_version == 1 for c in result.chunks)
    d1 = [c for c in result.chunks if "Khoản một" in c.content]
    assert d1
    assert d1[0].khoan_nos == ["1", "2"]
    assert d1[0].diem_labels == ["a"]
    assert "khoan:1/diem:a" in d1[0].subdivision_refs


def test_parse_legacy_non_legal_chunks_default_subdivision_metadata(
    parser, tmp_path, monkeypatch
):
    """Non-legal chunks (DocumentChunker has no keys) default to [] / 0."""
    monkeypatch.setattr(settings, "HRAG_LEGAL_CHUNKING", True)
    path = tmp_path / "cv.txt"
    path.write_text("Công văn thông thường. " * 60, encoding="utf-8")
    result = parser._parse_legacy(path, 7, "cv.txt")
    assert result.chunks
    for c in result.chunks:
        assert c.khoan_nos == []
        assert c.diem_labels == []
        assert c.subdivision_refs == []
        assert c.subdivision_schema_version == 0

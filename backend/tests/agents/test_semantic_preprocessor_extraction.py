"""Regression coverage for document-reference extraction arbitration (defect #1)."""
from __future__ import annotations

import pytest

from app.services.agents.semantic_preprocessor import (
    DocumentRefEntry,
    PreprocessingResult,
    build_normalized_match_view,
    extract_document_references,
)


def _extract(query: str):
    normalized, _ = build_normalized_match_view(query)
    return extract_document_references(normalized, query)


def test_official_document_number_emits_one_non_overlapping_ref() -> None:
    refs = _extract("53/2022/NĐ-CP quy định những nội dung gì?")
    assert [(r.span_offset, r.parse_basis) for r in refs] == [
        ((0, 13), "regex_doc_num")
    ]


def test_bare_abbreviation_is_kept_as_a_single_ref() -> None:
    refs = _extract("NĐ là gì?")
    assert [(r.span_offset, r.reference) for r in refs] == [((0, 2), "nđ")]


@pytest.mark.parametrize(
    "query",
    [
        "53/2022/NĐ-CP quy định những nội dung gì?",
        "NĐ là gì?",
        "So sánh 85/2016/NĐ-CP và Luật An ninh mạng",
        "So sánh Luật An ninh mạng và Luật Bảo vệ dữ liệu cá nhân",
    ],
)
def test_arbitrated_refs_are_pairwise_non_overlapping(query: str) -> None:
    spans = sorted(r.span_offset for r in _extract(query))
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert end <= next_start, f"{query!r}: {spans}"


def test_number_query_constructs_a_valid_preprocessing_result() -> None:
    query = "53/2022/NĐ-CP quy định những nội dung gì?"
    normalized, _ = build_normalized_match_view(query)
    entries = [
        DocumentRefEntry(
            ref_id=r.ref_id,
            original_span=r.original_span,
            span_offset=r.span_offset,
            reference=r.reference,
            section_reference=r.section_reference,
            resolution_status="deferred",
        )
        for r in extract_document_references(normalized, query)
    ]
    result = PreprocessingResult(
        original_query=query,
        normalized_query=normalized,
        document_refs=entries,
        preprocessing_status="ok",
        preprocessor_trace=[],
    )
    assert len(result.document_refs) == 1

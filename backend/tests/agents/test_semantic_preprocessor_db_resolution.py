"""Regression coverage for DB-backed semantic-preprocessor resolution."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from app.services.agents.semantic_preprocessor import (
    AbbreviationEntry,
    RefExtraction,
    _call_llm_for_disambiguation,
    _lookup_by_alias,
    _resolve_abbreviations_db,
)

WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _context() -> SimpleNamespace:
    return SimpleNamespace(allowed_workspace_ids=[WORKSPACE_ID])


@pytest.mark.asyncio
async def test_abbreviation_db_lookup_returns_the_resolved_expansion() -> None:
    """Missing document-model resolution must not turn a known abbreviation into an error."""
    entry = AbbreviationEntry(
        span="NĐ",
        span_offset=(0, 2),
        short_form="nđ",
        status="unknown",
    )
    db_result = MagicMock()
    db_result.fetchall.return_value = [
        SimpleNamespace(
            alias_text="nđ",
            document_number="Nghị định 12/2020/NĐ-CP",
            document_title=None,
        )
    ]
    session = MagicMock(execute=AsyncMock(return_value=db_result))

    resolved = await _resolve_abbreviations_db([entry], _context(), session)

    assert resolved[0].chosen == "Nghị định 12/2020/NĐ-CP"
    assert resolved[0].status == "resolved"
    assert resolved[0].source == "db_single"


@pytest.mark.asyncio
async def test_alias_lookup_returns_the_workspace_scoped_document() -> None:
    """A known document alias must resolve instead of being deferred by a SQL helper error."""
    document = SimpleNamespace(
        id=DOCUMENT_ID,
        workspace_id=WORKSPACE_ID,
        updated_at=None,
        content_hash=None,
    )
    db_result = MagicMock()
    db_result.all.return_value = [(document, SimpleNamespace())]
    session = MagicMock(execute=AsyncMock(return_value=db_result))
    reference = RefExtraction(
        ref_id="r1",
        original_span="Luật An ninh mạng",
        span_offset=(0, 19),
        reference="Luật An ninh mạng",
        section_reference=None,
        parse_basis="regex_named_doc",
    )

    resolved = await _lookup_by_alias(reference, _context(), session)

    assert resolved.resolution_status == "resolved"
    assert resolved.document_handle == DOCUMENT_ID
    assert resolved.metadata.workspace_id == WORKSPACE_ID


@pytest.mark.asyncio
async def test_llm_abbreviation_fallback_includes_the_abbreviation_in_its_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved abbreviation must reach the LLM fallback and return its expansion."""
    class FakeLLM:
        prompt = ""

        async def acomplete(self, messages, **_kwargs):
            self.prompt = messages[0].content
            return (
                '[{"short_form": "tlk", "full_form": "Tài liệu khác", '
                '"confidence": 0.9, "reasoning": "test"}]'
            )

    fake_llm = FakeLLM()
    import app.services.llm as llm_module

    # Role ownership (Phase-4 Task 0): abbreviation disambiguation is owned by
    # the semantic_router role, not the main answer model.
    monkeypatch.setattr(llm_module, "get_semantic_router_provider", lambda: fake_llm)
    abbreviation = AbbreviationEntry(
        span="TLK",
        span_offset=(0, 3),
        short_form="tlk",
        status="unknown",
    )

    resolved = await _call_llm_for_disambiguation(
        [abbreviation], "TLK là gì?", timeout_sec=1.0
    )

    assert "tlk (original: 'TLK')" in fake_llm.prompt
    assert resolved[0].chosen == "Tài liệu khác"
    assert resolved[0].status == "resolved"

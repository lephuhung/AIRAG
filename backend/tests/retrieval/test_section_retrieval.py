"""
Regression: ``search_document_section`` must find "Điều N" that EXISTS in the doc.

Reproduces the 2026-07-03 bug: "Tóm tắt điều 17 của Nghị định 85/2016" →
"không tìm thấy Điều 17" although the parsed markdown AND the ChromaDB chunks
both contain the heading "## Điều 17. Hồ sơ phê duyệt đề xuất cấp độ" verbatim
(doc 81ccfe99, chunk 43). Root cause is the TOOL, not the LLM and not the data:

1. Structural lookup matches only the ``heading_path`` chunk metadata — which is
   empty for 100% of chunks corpus-wide (Docling never captures "Điều N" as a
   heading level on these documents), so it can never match.
2. The fallback embeds the literal string "Điều 17" and does a semantic top-10 —
   a bare article number carries no semantics, so it returns unrelated chunks
   (page-1 letterhead, Điều 11, training paragraphs) and the LLM truthfully
   answers "not found".

The fix (pending): match the section reference against CHUNK CONTENT (the
markdown headings survive verbatim in the chunk text) instead of the empty
heading_path metadata. This test stays RED until that lands.

Needs the live stack (PostgreSQL + ChromaDB + embedding models) — run inside
the backend container, gated by RETRIEVAL_EVAL=1:

    docker exec -e RETRIEVAL_EVAL=1 -w /app/backend hrag-backend \
        pytest tests/retrieval -q
"""
from __future__ import annotations

import os
import re

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.skipif(
    os.getenv("RETRIEVAL_EVAL") != "1",
    reason="set RETRIEVAL_EVAL=1 to run live retrieval regression tests",
)

# (document_number pattern, section reference) — extend with new regressions.
SECTION_CASES = [
    ("85/2016%", "Điều 17"),    # OCR path, the original bug report
    ("361/2025%", "Điều 9"),    # OCR path, 812-chunk doc (mostly appendix tables)
    ("116/2025%", "Điều 3"),    # OCR path, luật
    ("24/2018%", "Điều 10"),    # Docling path — heading_path from meta.headings
]


async def _find_doc(number_pattern: str) -> tuple[str, str] | None:
    """Locate an indexed document by number → (document_id, workspace_id)."""
    from app.core.database import async_session_maker

    async with async_session_maker() as db:
        row = (await db.execute(
            text(
                "SELECT id, workspace_id FROM documents "
                "WHERE document_number ILIKE :pat AND status = 'indexed' LIMIT 1"
            ),
            {"pat": number_pattern},
        )).fetchone()
    return (str(row[0]), str(row[1])) if row else None


@pytest.mark.parametrize("number_pattern,section_ref", SECTION_CASES)
async def test_section_lookup_finds_existing_article(number_pattern, section_ref):
    from app.services.agent.tools import search_document_section

    found = await _find_doc(number_pattern)
    if not found:
        pytest.skip(f"no indexed document matching {number_pattern!r} in this environment")
    doc_id, ws_id = found

    res = await search_document_section(
        section_reference=section_ref,
        workspace_ids=[ws_id],
        document_ids=[doc_id],
    )
    sources = res.get("sources") or []
    # Whole-word match ("Điều 17" must not be satisfied by "Điều 170"). Checked
    # against content OR heading_path: the OCR path keeps the heading line inside
    # the chunk text, while the Docling path stores it only in heading_path
    # metadata (the chunk body does not repeat the heading).
    ref_re = re.compile(rf"{re.escape(section_ref)}(?!\d)")
    def _matches(s) -> bool:
        if ref_re.search(getattr(s, "content", "") or ""):
            return True
        hp = getattr(s, "heading_path", None) or []
        return bool(ref_re.search(" > ".join(str(c) for c in hp)))
    hit = any(_matches(s) for s in sources)
    assert hit, (
        f"{section_ref} exists verbatim in doc {doc_id} chunks, but "
        f"search_document_section returned {len(sources)} source(s) without it "
        f"(heading_path metadata is empty corpus-wide; semantic fallback on a bare "
        f"article number returns unrelated chunks)"
    )

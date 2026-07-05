"""
Regression: hiệu lực văn bản trong hybrid retrieval (``DeepRetriever``).

Kho có cặp thật: Luật An ninh mạng 24/2018/QH14 (superseded) và 116/2025/QH15
(effective) cùng workspace — 116/2025 Điều 44 tuyên bố 24/2018 hết hiệu lực,
``validity_service`` đã đánh dấu qua backfill/parse.

Hai hành vi phải giữ:
1. Query KHÔNG scope: chunk văn bản superseded bị demote
   (HRAG_SUPERSEDED_DEMOTE) — văn bản còn hiệu lực phải thắng ở top đầu.
2. Query scope ĐÍCH DANH văn bản superseded: vẫn trả chunk (user chủ động
   hỏi văn bản cũ) nhưng context PHẢI chứa cảnh báo hết hiệu lực + số hiệu
   văn bản thay thế.

Cần live stack:

    docker exec -e RETRIEVAL_EVAL=1 -w /app/backend hrag-backend \
        pytest tests/retrieval/test_validity.py -q
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.skipif(
    os.getenv("RETRIEVAL_EVAL") != "1",
    reason="set RETRIEVAL_EVAL=1 to run live retrieval regression tests",
)

SUPERSEDED_NUMBER = "24/2018/QH14"
QUERY = "Các biện pháp bảo vệ an ninh mạng"


async def _find_superseded_doc():
    from app.core.database import async_session_maker

    async with async_session_maker() as db:
        row = (await db.execute(
            text(
                "SELECT id, workspace_id, superseded_by_number FROM documents "
                "WHERE document_number = :num AND validity_status = 'superseded' "
                "AND embed_done = true LIMIT 1"
            ),
            {"num": SUPERSEDED_NUMBER},
        )).fetchone()
    return row


async def test_unscoped_query_demotes_superseded_document():
    from app.core.database import async_session_maker
    from app.services.retrieval.hrag_service import HRAGService

    row = await _find_superseded_doc()
    if not row:
        pytest.skip(f"no superseded {SUPERSEDED_NUMBER} in this environment")
    doc_id, ws_id, _ = row

    async with async_session_maker() as db:
        svc = HRAGService(db, uuid.UUID(str(ws_id)))
        res = await svc.query_deep(QUERY, top_k=8, include_images=False)

    assert res.chunks, "retrieval returned nothing"
    # Văn bản còn hiệu lực phải chiếm vị trí #1; chunk superseded nếu còn sót
    # trong top-k phải mang annotation để tầng trên cảnh báo được.
    assert res.chunks[0].validity_status != "superseded", (
        f"top-1 chunk vẫn thuộc văn bản superseded (doc={res.chunks[0].document_id})"
    )
    for c in res.chunks:
        if str(c.document_id) == str(doc_id):
            assert c.validity_status == "superseded"


async def test_scoped_query_returns_superseded_with_warning():
    from app.core.database import async_session_maker
    from app.services.retrieval.hrag_service import HRAGService

    row = await _find_superseded_doc()
    if not row:
        pytest.skip(f"no superseded {SUPERSEDED_NUMBER} in this environment")
    doc_id, ws_id, superseded_by = row

    async with async_session_maker() as db:
        svc = HRAGService(db, uuid.UUID(str(ws_id)))
        res = await svc.query_deep(
            QUERY, top_k=5, document_ids=[doc_id], include_images=False
        )

    assert res.chunks, "scope đích danh phải vẫn trả chunk của văn bản cũ"
    assert all(str(c.document_id) == str(doc_id) for c in res.chunks)
    assert "VĂN BẢN NÀY ĐÃ HẾT HIỆU LỰC" in res.context
    if superseded_by:
        assert superseded_by in res.context, (
            "cảnh báo phải nêu số hiệu văn bản thay thế"
        )

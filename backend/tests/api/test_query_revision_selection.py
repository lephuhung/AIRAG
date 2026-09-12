"""Phase 1C Task 5 — the query endpoint's ``document_ids`` branch fails typed.

``api/rag.py: query_documents`` resolves caller-named ``document_ids`` through
``resolve_document_targets``, which raises ``RevisionNotReady`` when a
document's current pointer is not a published/artifact-complete revision (e.g.
a crash between the pointer write and publish). That must surface as
``409 REVISION_NOT_READY`` — exactly like the sibling caller-supplied
``revision_ids`` branch — never as a generic 500.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import select, text

from app.models.document import Document
from app.schemas.rag import RAGQueryRequest
from app.services.agents.v2.persistence.document_revisions import (
    DocumentRevisionsRepository,
)
from app.services.agents.v2.persistence.source_identity import (
    RevisionBuildProfile,
    compute_source_object_identity,
)


@pytest.mark.asyncio
async def test_query_document_ids_with_unready_current_revision_is_409(
    async_db, document_factory, monkeypatch
):
    from app.api import rag as rag_api

    doc_id = document_factory()
    workspace_id = await async_db.scalar(
        select(Document.workspace_id).where(Document.id == doc_id)
    )
    repo = DocumentRevisionsRepository(async_db)
    draft = await repo.allocate_draft(
        doc_id,
        compute_source_object_identity(
            bucket="hrag-uploads",
            object_key=f"kb_{workspace_id}/doc_{doc_id}.pdf",
            version_id=None,
            etag="etag1",
            size_bytes=10,
            content_sha256="7" * 64,
        ),
        RevisionBuildProfile.FULL,
    )
    await async_db.execute(
        text("UPDATE documents SET current_revision_id = :r WHERE id = :d"),
        {"r": str(draft.revision_id), "d": str(doc_id)},
    )
    await async_db.commit()

    # The retrieval service is irrelevant: the request must fail before any
    # retrieval leg runs. Abbreviation expansion is likewise irrelevant (and its
    # table is not part of the v2 test bootstrap).
    monkeypatch.setattr(rag_api, "get_rag_service", lambda db, ws_id: object())

    async def _passthrough(_db, text):
        return text

    monkeypatch.setattr(
        rag_api.AbbreviationService, "expand_ab_in_text", staticmethod(_passthrough)
    )

    with pytest.raises(HTTPException) as exc:
        await rag_api.query_documents(
            workspace_id,
            RAGQueryRequest(question="câu hỏi", document_ids=[doc_id]),
            db=async_db,
            user=None,
        )
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "REVISION_NOT_READY"

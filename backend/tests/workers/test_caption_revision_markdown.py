"""Phase 1 final-review I2 — caption enrichment must stay revision-scoped.

``caption_worker`` re-uploads the caption-enriched markdown. It used to call
``upload_markdown`` with no ``key``, so the legacy ``kb_<ws>/doc_<doc>.md``
object was (re)written and the revision's own artifact
(``kb_<ws>/revisions/<doc>/<rev>/document.md``) was never updated — the served
revision stayed stale while a legacy object a v2 build must not touch was
created. The enriched copy belongs at the revision-qualified key that
``parse_worker`` uploaded.

Needs the storage stack (``aioboto3``); the benchmark venv lacks it, so this
module skips there and runs in the full-dependency container.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("aioboto3", reason="caption_worker imports storage_service")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.document import Document, DocumentTable
from app.queue.messages import CaptionMessage
from app.services.agents.v2.persistence import document_views
from app.services.agents.v2.persistence.document_revisions import (
    DocumentRevisionsRepository,
)
from app.services.agents.v2.persistence.source_identity import (
    RevisionBuildProfile,
    compute_source_object_identity,
)
from app.workers import caption_worker


def _session_maker(engine):
    return async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autocommit=False
    )


class _RecordingStore:
    def __init__(self) -> None:
        self.downloads: list[str] = []
        self.uploads: list[str | None] = []

    async def download_markdown(self, key: str) -> str:
        self.downloads.append(key)
        return "# tài liệu"

    async def upload_markdown(
        self, *, workspace_id, document_id, content, key=None
    ) -> str:
        self.uploads.append(key)
        return key


class _FakeParser:
    def __init__(self, workspace_id=None) -> None:
        pass

    def _inject_table_captions(self, markdown, tables) -> str:
        return markdown + "\n<!-- injected -->"


@pytest.mark.asyncio
async def test_caption_worker_writes_back_to_revision_markdown_key(
    async_engine, document_factory, monkeypatch
):
    maker = _session_maker(async_engine)
    monkeypatch.setattr(caption_worker, "async_session_maker", maker)
    doc_id = document_factory()
    async with maker() as db:
        ws = await db.scalar(
            select(Document.workspace_id).where(Document.id == doc_id)
        )
        revision = await DocumentRevisionsRepository(db).allocate_draft(
            doc_id,
            compute_source_object_identity(
                bucket="hrag-uploads",
                object_key=f"kb_{ws}/doc_{doc_id}.pdf",
                version_id=None,
                etag="e1",
                size_bytes=10,
                content_sha256="a" * 64,
            ),
            RevisionBuildProfile.FULL,
        )
        rev_id = revision.revision_id
        db.add(
            DocumentTable(
                document_id=doc_id,
                revision_id=rev_id,
                table_id="tbl-1",
                page_no=1,
                content_markdown="| a |",
                num_rows=1,
                num_cols=1,
            )
        )
        await db.commit()

    store = _RecordingStore()

    async def _caption_tables(tables):
        for table in tables:
            table.caption = "bảng đã chú thích"

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(caption_worker, "_caption_tables_concurrent", _caption_tables)
    monkeypatch.setattr(caption_worker, "_reenrich_embeddings", _noop)
    monkeypatch.setattr(caption_worker, "check_and_finalize", _noop)
    monkeypatch.setattr(caption_worker, "get_storage_service", lambda: store)
    monkeypatch.setattr(caption_worker, "DeepDocumentParser", _FakeParser)
    monkeypatch.setattr(caption_worker.settings, "HRAG_ENABLE_TABLE_CAPTIONING", True)
    monkeypatch.setattr(caption_worker.settings, "HRAG_ENABLE_IMAGE_CAPTIONING", False)

    import app.workers.config_watch as config_watch

    async def _fresh():
        return None

    monkeypatch.setattr(config_watch, "ensure_fresh_config", _fresh)

    await caption_worker.handle_caption(
        CaptionMessage(
            document_id=doc_id,
            workspace_id=ws,
            revision_id=rev_id,
            build_profile=RevisionBuildProfile.FULL.value,
        ).model_dump(mode="json")
    )

    expected = document_views.revision_markdown_key(ws, doc_id, rev_id)
    legacy = f"kb_{ws}/doc_{doc_id}.md"
    assert store.downloads == [expected]
    assert store.uploads == [expected]
    assert legacy not in store.uploads

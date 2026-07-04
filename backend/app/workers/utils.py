"""
Worker utilities
================
Shared helpers used by embed, caption, and kg workers.
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentStatus
from app.models.document_type import DocumentType as _DocumentType  # noqa: F401

logger = logging.getLogger(__name__)


async def check_and_finalize(document: Document, db: AsyncSession) -> None:
    """
    Transition document status based on sub-task completion:

      - is_chat_upload + embed_done → INDEXED (chat temp files: parse → embed only)
      - embed_done + captions_done + kg_done → INDEXED (full pipeline)
      - embed_done + captions_done (kg still running) → BUILDING_KG
      - otherwise → no change (still EMBEDDING or CHUNKING)

    Chat-upload documents skip KG and caption workers, so only embed_done
    is needed before marking INDEXED.

    Opens a *separate* session so it always reads the latest committed values
    from the other workers (avoids stale snapshot from the caller's long-lived
    transaction).  SELECT FOR UPDATE serialises concurrent calls so only one
    worker promotes the document.
    """
    from app.core.database import async_session_maker

    async with async_session_maker() as fresh_db:
        result = await fresh_db.execute(
            select(Document)
            .where(Document.id == document.id)
            .with_for_update()
        )
        fresh = result.scalar_one_or_none()
        if fresh is None:
            return

        # FAILED is a terminal state — only admin retry can clear it
        if fresh.status == DocumentStatus.FAILED:
            return

        changed = False

        # Chat-upload documents: skip KG and caption workers, so only embed_done is needed
        if fresh.is_chat_upload:
            if fresh.embed_done:
                if fresh.raw_chunks_json is not None:
                    fresh.raw_chunks_json = None
                    changed = True
                if fresh.status != DocumentStatus.INDEXED:
                    fresh.status = DocumentStatus.INDEXED
                    changed = True
                    logger.info(
                        f"[finalize] doc={fresh.id} → INDEXED "
                        f"(chat-upload: embed✓)"
                    )
                if changed:
                    await fresh_db.commit()
            return

        if fresh.embed_done and fresh.captions_done:
            # captions_done is set AFTER the caption re-embed ran, so no worker
            # needs the raw chunks anymore — free the (potentially large) column.
            # The KG worker reads markdown from MinIO, not from raw_chunks_json.
            if fresh.raw_chunks_json is not None:
                fresh.raw_chunks_json = None
                changed = True
            if fresh.kg_done:
                # All three done → INDEXED
                if fresh.status != DocumentStatus.INDEXED:
                    fresh.status = DocumentStatus.INDEXED
                    changed = True
                    logger.info(
                        f"[finalize] doc={fresh.id} → INDEXED "
                        f"(embed✓ captions✓ kg✓)"
                    )
            else:
                # embed+captions done, KG still running → BUILDING_KG
                if fresh.status not in (DocumentStatus.BUILDING_KG, DocumentStatus.INDEXED):
                    fresh.status = DocumentStatus.BUILDING_KG
                    changed = True
                    logger.info(
                        f"[finalize] doc={fresh.id} → BUILDING_KG "
                        f"(embed✓ captions✓ kg⟳)"
                    )
            if changed:
                await fresh_db.commit()

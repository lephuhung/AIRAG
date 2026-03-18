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

      - embed_done + captions_done + kg_done → INDEXED
      - embed_done + captions_done (kg still running) → BUILDING_KG
      - otherwise → no change (still EMBEDDING or CHUNKING)

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

        if fresh.embed_done and fresh.captions_done:
            if fresh.kg_done:
                # All three done → INDEXED
                if fresh.status != DocumentStatus.INDEXED:
                    fresh.status = DocumentStatus.INDEXED
                    await fresh_db.commit()
                    logger.info(
                        f"[finalize] doc={fresh.id} → INDEXED "
                        f"(embed✓ captions✓ kg✓)"
                    )
            else:
                # embed+captions done, KG still running → BUILDING_KG
                if fresh.status not in (DocumentStatus.BUILDING_KG, DocumentStatus.INDEXED):
                    fresh.status = DocumentStatus.BUILDING_KG
                    await fresh_db.commit()
                    logger.info(
                        f"[finalize] doc={fresh.id} → BUILDING_KG "
                        f"(embed✓ captions✓ kg⟳)"
                    )

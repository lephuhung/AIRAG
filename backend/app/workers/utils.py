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
    Transition document to INDEXED when embed + captions sub-tasks are complete.

    KG ingestion is intentionally excluded from the INDEXED condition:
      - KG extraction via LLM can take many minutes (large docs, slow models).
      - The document is fully searchable once embed_done + captions_done are True.
      - KG continues running in the background; kg_done is tracked separately
        and surfaced in the frontend as a non-blocking background indicator.

    Opens a *separate* session so it always reads the latest committed values
    from the other workers (avoids stale snapshot from the caller's long-lived
    transaction).  SELECT FOR UPDATE serialises concurrent calls so only one
    worker promotes the document to INDEXED.
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

        # Transition to INDEXED as soon as embed + captions are done.
        # KG is a non-blocking background task — it keeps running after INDEXED.
        if fresh.embed_done and fresh.captions_done:
            if fresh.status != DocumentStatus.INDEXED:
                fresh.status = DocumentStatus.INDEXED
                await fresh_db.commit()
                logger.info(
                    f"[finalize] doc={fresh.id} → INDEXED "
                    f"(embed✓ captions✓ | kg={'✓' if fresh.kg_done else '⟳ background'})"
                )

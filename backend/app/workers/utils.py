"""
Worker utilities
================
Shared helpers used by embed, caption, and kg workers.
"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)


async def check_and_finalize(document: Document, db: AsyncSession) -> None:
    """
    Transition document to INDEXED when all three sub-tasks are complete.
    Called at the end of embed_worker, caption_worker, and kg_worker.
    """
    if document.embed_done and document.captions_done and document.kg_done:
        document.status = DocumentStatus.INDEXED
        logger.info(
            f"[finalize] doc={document.id} fully INDEXED "
            f"(embed✓ captions✓ kg✓)"
        )
        await db.commit()

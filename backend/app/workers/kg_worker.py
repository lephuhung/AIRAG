"""
KG Worker
=========
Consumes nexusrag.kg.<workspace_id> queue.

Key design decisions:
  - routing_key = workspace_id → RabbitMQ delivers all docs for a workspace
    to the SAME queue, processed ONE AT A TIME (prefetch_count=1).
    This prevents concurrent LightRAG writes to the same graph files.
  - llm_model_max_async=3 inside LightRAG → max 3 chunk LLM calls at once.
  - Semaphore + exponential-backoff retry in _kg_llm_complete for rate limits.
  - KG failure does NOT fail the document — it stays INDEXED_PARTIAL
    and captions_done/embed_done are unaffected.
"""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from app.core.database import async_session_maker
from app.models.document_type import DocumentType as _DocumentType  # noqa: F401
from app.models.document import Document
from app.queue.messages import KGMessage
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.workers.utils import check_and_finalize

logger = logging.getLogger(__name__)


async def handle_kg(payload: dict) -> None:
    msg = KGMessage(**payload)
    logger.info(
        f"[kg_worker] doc={msg.document_id} workspace={msg.workspace_id} "
        f"markdown_len={len(msg.markdown)}"
    )

    async with async_session_maker() as db:
        result = await db.execute(select(Document).where(Document.id == msg.document_id))
        document = result.scalar_one_or_none()
        if document is None:
            logger.error(f"[kg_worker] doc={msg.document_id} not found")
            return

        if not msg.markdown.strip():
            logger.warning(f"[kg_worker] doc={msg.document_id} empty markdown — skipping KG")
            document.kg_done = True
            await db.commit()
            await check_and_finalize(document, db)
            return

        try:
            kg_service = KnowledgeGraphService(workspace_id=msg.workspace_id)
            await kg_service.ingest(msg.markdown)

            document.kg_done = True
            await db.commit()
            logger.info(f"[kg_worker] doc={msg.document_id} KG ingest done")
            await check_and_finalize(document, db)

        except Exception as e:
            logger.error(
                f"[kg_worker] doc={msg.document_id} KG ingest FAILED: {e}",
                exc_info=True,
            )
            # KG failure is non-fatal: mark done so document can reach INDEXED
            document.kg_done = True
            document.error_message = f"kg_warning: {str(e)[:400]}"
            await db.commit()
            await check_and_finalize(document, db)
        finally:
            # Release cached GPU memory after each document so other workers
            # (or the next document in this worker) can reclaim the blocks.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

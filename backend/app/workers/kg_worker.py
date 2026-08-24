"""
KG Worker
=========
Consumes hrag.kg.<workspace_id> queue.

Key design decisions:
  - routing_key = workspace_id → RabbitMQ delivers all docs for a workspace
    to the SAME queue, processed ONE AT A TIME (prefetch_count=1).
    This prevents concurrent LightRAG writes to the same graph files.
  - llm_model_max_async=3 inside LightRAG → max 3 chunk LLM calls at once.
  - Semaphore + exponential-backoff retry in _kg_llm_complete for rate limits.
  - KG failure does NOT fail the document — it stays BUILDING_KG
    and captions_done/embed_done are unaffected.
  - Idempotent: checks kg_done at start to skip already-processed documents
    (handles redelivered messages after crash-before-ack).
"""
from __future__ import annotations

import logging

from sqlalchemy import select

from app.core.database import async_session_maker
from app.models.document import Document
from app.queue.messages import KGMessage
from app.services.kg.knowledge_graph_service import get_kg_service
from app.workers.utils import check_and_finalize

logger = logging.getLogger(__name__)


async def handle_kg(payload: dict) -> None:
    # Config watch first: pick up WebUI LLM changes before resolving any
    # provider (kg_extract / main via legal_kg + knowledge_graph services).
    # Fail-open — never blocks the message.
    from app.workers.config_watch import ensure_fresh_config
    await ensure_fresh_config()

    msg = KGMessage(**payload)
    logger.info(
        f"[kg_worker] doc={msg.document_id} workspace={msg.workspace_id} "
        f"markdown_s3_key={msg.markdown_s3_key or '-'}"
    )

    async with async_session_maker() as db:
        result = await db.execute(select(Document).where(Document.id == msg.document_id))
        document = result.scalar_one_or_none()
        if document is None:
            logger.error(f"[kg_worker] doc={msg.document_id} not found")
            return

        # Idempotency: if kg_done=True the message was already processed
        # (e.g. redelivered after a crash-before-ack). Skip silently.
        if document.kg_done:
            logger.info(f"[kg_worker] doc={msg.document_id} already has kg_done=True — skipping")
            return

        try:
            # ── Load markdown ────────────────────────────────────────────────
            # New messages carry only markdown_s3_key (small broker payload);
            # the inline `markdown` field is a fallback for in-flight/DLQ
            # messages published before the key existed.
            markdown = msg.markdown
            if msg.markdown_s3_key:
                from app.services.storage_service import get_storage_service

                markdown = await get_storage_service().download_markdown(
                    msg.markdown_s3_key
                )

            # OCR-path markdown carries administrative-layout HTML (alignment
            # divs, data-bbox). That markup is token waste + extraction noise
            # for the KG LLM — strip it (no-op for Docling documents).
            from app.services.parsing.ocr_service import strip_ocr_layout

            markdown = strip_ocr_layout(markdown)

            if not markdown.strip():
                logger.warning(f"[kg_worker] doc={msg.document_id} empty markdown — skipping KG")
                document.kg_done = True
                await db.commit()
                await check_and_finalize(document, db)
                return

            kg_service = get_kg_service(workspace_id=msg.workspace_id)
            logger.info(
                f"[kg_worker] doc={msg.document_id} starting KG ingest "
                f"(markdown_len={len(markdown)})..."
            )
            await kg_service.ingest(markdown, document_id=msg.document_id)

            document.kg_done = True
            # A successful ingest supersedes any stale KG/timeout warning left
            # by an earlier failed attempt — don't keep scaring the UI.
            if document.error_message and document.error_message.startswith(
                ("kg_", "timeout_retry")
            ):
                document.error_message = None
            await db.commit()
            logger.info(f"[kg_worker] doc={msg.document_id} KG ingest done")
            await check_and_finalize(document, db)

        except Exception as e:
            # NOTE: a handler-level timeout from connection.py arrives here as
            # CancelledError (BaseException) and is NOT caught — connection.py
            # then resets kg_done=False and requeues for a real retry. This
            # block only handles failures raised by the ingest itself, which
            # are non-fatal by design: mark done so the document can reach
            # INDEXED, and record honestly that the KG was skipped.
            is_timeout = isinstance(e, TimeoutError)
            logger.error(
                f"[kg_worker] doc={msg.document_id} KG ingest FAILED "
                f"({type(e).__name__}): {e}",
                exc_info=True,
            )
            # Roll back first: the session may be aborted after a DB error.
            await db.rollback()
            document.kg_done = True
            document.error_message = (
                f"kg_timeout: ingest timed out — KG skipped"
                if is_timeout
                else f"kg_warning: {str(e)[:400]}"
            )
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

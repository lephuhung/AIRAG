"""
Caption Worker
==============
Consumes nexusrag.caption queue.

Responsibilities:
  1. Load images (no caption) + tables (no caption) from DB
  2. Run Vision LLM on images concurrently (asyncio.gather + semaphore)
  3. Run Text LLM on tables concurrently (asyncio.gather + semaphore)
  4. Update captions in DB
  5. Re-embed enriched chunks in ChromaDB (image/table descriptions added)
  6. Update markdown in MinIO with injected table captions
  7. Set captions_done=True
  8. Check if fully INDEXED
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from sqlalchemy import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.document_type import DocumentType as _DocumentType  # noqa: F401
from app.models.document import Document, DocumentImage, DocumentStatus, DocumentTable
from app.queue.messages import CaptionMessage
from app.services.deep_document_parser import DeepDocumentParser
from app.services.embedder import get_embedding_service
from app.services.models.parsed_document import ExtractedImage, ExtractedTable
from app.services.storage_service import get_storage_service
from app.services.vector_store import get_vector_store
from app.workers.utils import check_and_finalize

logger = logging.getLogger(__name__)

# Max concurrent LLM calls for captioning
_CAPTION_SEMAPHORE = asyncio.Semaphore(4)


async def handle_caption(payload: dict) -> None:
    msg = CaptionMessage(**payload)
    logger.info(f"[caption_worker] doc={msg.document_id}")

    async with async_session_maker() as db:
        result = await db.execute(select(Document).where(Document.id == msg.document_id))
        document = result.scalar_one_or_none()
        if document is None:
            logger.error(f"[caption_worker] doc={msg.document_id} not found")
            return

        has_images  = settings.NEXUSRAG_ENABLE_IMAGE_CAPTIONING
        has_tables  = settings.NEXUSRAG_ENABLE_TABLE_CAPTIONING

        try:
            # ── Load images and tables from DB ──────────────────────────────
            img_result = await db.execute(
                select(DocumentImage).where(DocumentImage.document_id == msg.document_id)
            )
            db_images: list[DocumentImage] = img_result.scalars().all()

            tbl_result = await db.execute(
                select(DocumentTable).where(DocumentTable.document_id == msg.document_id)
            )
            db_tables: list[DocumentTable] = tbl_result.scalars().all()

            if not db_images and not db_tables:
                logger.info(f"[caption_worker] doc={msg.document_id} no images/tables — done")
                document.captions_done = True
                await db.commit()
                await check_and_finalize(document, db)
                return

            # ── Caption images concurrently ─────────────────────────────────
            if has_images and db_images:
                ext_images = [
                    ExtractedImage(
                        image_id=img.image_id,
                        document_id=img.document_id,
                        page_no=img.page_no,
                        file_path=img.file_path,
                        caption=img.caption or "",
                        width=img.width,
                        height=img.height,
                        mime_type=img.mime_type,
                    )
                    for img in db_images
                    if not img.caption   # skip already-captioned
                ]
                if ext_images:
                    await _caption_images_concurrent(ext_images)
                    # Flush captions back to DB
                    img_by_id = {img.image_id: img for img in db_images}
                    for ext in ext_images:
                        if ext.caption and ext.image_id in img_by_id:
                            img_by_id[ext.image_id].caption = ext.caption
                    await db.commit()
                    logger.info(
                        f"[caption_worker] doc={msg.document_id} captioned "
                        f"{len(ext_images)} images"
                    )

            # ── Caption tables concurrently ─────────────────────────────────
            if has_tables and db_tables:
                ext_tables = [
                    ExtractedTable(
                        table_id=tbl.table_id,
                        document_id=tbl.document_id,
                        page_no=tbl.page_no,
                        content_markdown=tbl.content_markdown,
                        caption=tbl.caption or "",
                        num_rows=tbl.num_rows,
                        num_cols=tbl.num_cols,
                    )
                    for tbl in db_tables
                    if not tbl.caption
                ]
                if ext_tables:
                    await _caption_tables_concurrent(ext_tables)
                    tbl_by_id = {tbl.table_id: tbl for tbl in db_tables}
                    for ext in ext_tables:
                        if ext.caption and ext.table_id in tbl_by_id:
                            tbl_by_id[ext.table_id].caption = ext.caption
                    await db.commit()
                    logger.info(
                        f"[caption_worker] doc={msg.document_id} captioned "
                        f"{len(ext_tables)} tables"
                    )

            # ── Update markdown in MinIO with table captions injected ───────
            if has_tables and db_tables and document.markdown_s3_key:
                all_ext_tables = [
                    ExtractedTable(
                        table_id=tbl.table_id,
                        document_id=tbl.document_id,
                        page_no=tbl.page_no,
                        content_markdown=tbl.content_markdown,
                        caption=tbl.caption or "",
                        num_rows=tbl.num_rows,
                        num_cols=tbl.num_cols,
                    )
                    for tbl in db_tables
                ]
                storage = get_storage_service()
                current_md = await storage.download_markdown(document.markdown_s3_key)
                parser = DeepDocumentParser(workspace_id=msg.workspace_id)
                updated_md = parser._inject_table_captions(current_md, all_ext_tables)
                await storage.upload_markdown(
                    workspace_id=msg.workspace_id,
                    document_id=msg.document_id,
                    content=updated_md,
                )
                # key unchanged — no DB update needed

            # ── Re-embed chunks enriched with captions ──────────────────────
            # Only if there were actual captions generated
            if (has_images and db_images) or (has_tables and db_tables):
                await _reenrich_embeddings(msg.document_id, msg.workspace_id, db_images, db_tables, document)

            # ── Done ────────────────────────────────────────────────────────
            document.captions_done = True
            await db.commit()
            logger.info(f"[caption_worker] doc={msg.document_id} captions done")
            await check_and_finalize(document, db)

        except Exception as e:
            logger.error(
                f"[caption_worker] doc={msg.document_id} FAILED: {e}", exc_info=True
            )
            # Caption failure does NOT fail the whole document —
            # it stays INDEXED_PARTIAL if embed already done
            document.captions_done = True   # mark done to unblock INDEXED transition
            document.error_message = f"caption_warning: {str(e)[:400]}"
            await db.commit()
            await check_and_finalize(document, db)


async def _caption_images_concurrent(images: list[ExtractedImage]) -> None:
    """Caption all images with Vision LLM, max 4 concurrent."""
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMImagePart, LLMMessage

    _CAPTION_PROMPT = (
        "Describe ONLY what you can directly see in this image. "
        "Do NOT infer, assume, or add any information not visible.\n\n"
        "Include:\n"
        "- Type of visual (chart, table, diagram, photo, screenshot, etc.)\n"
        "- ALL specific numbers, percentages, and labels that are VISIBLE\n"
        "- Axis labels, legend text, and category names exactly as shown\n"
        "- Trends or comparisons that are visually obvious\n\n"
        "RULES:\n"
        "- Write 2-4 concise sentences, max 400 characters.\n"
        "- Do NOT start with 'This image shows' or 'Here is'.\n"
        "- Do NOT add data beyond what is visible.\n"
        "- If text is not clearly readable, say so."
    )

    provider = get_llm_provider()
    if not provider.supports_vision():
        logger.warning("[caption_worker] LLM does not support vision — skipping image captions")
        return

    async def caption_one(img: ExtractedImage) -> None:
        async with _CAPTION_SEMAPHORE:
            try:
                image_path = Path(img.file_path)
                if not image_path.exists():
                    return
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                message = LLMMessage(
                    role="user",
                    content=_CAPTION_PROMPT,
                    images=[LLMImagePart(data=image_bytes, mime_type=img.mime_type)],
                )
                result = await asyncio.to_thread(provider.complete, [message])
                if result:
                    img.caption = " ".join(str(result).strip().split())[:500]
            except Exception as e:
                logger.debug(f"[caption_worker] image {img.image_id} caption failed: {e}")

    await asyncio.gather(*[caption_one(img) for img in images])


async def _caption_tables_concurrent(tables: list[ExtractedTable]) -> None:
    """Caption all tables with Text LLM, max 4 concurrent."""
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage

    _TABLE_PROMPT = (
        "You are a document analysis assistant. Given a markdown table, "
        "write a concise description that covers:\n"
        "- The purpose/topic of the table\n"
        "- Key column names and what they represent\n"
        "- Notable values, trends, or outliers\n\n"
        "RULES:\n"
        "- Write 2-4 sentences, max 500 characters.\n"
        "- Be factual — describe only what is in the table.\n"
        "- Write in the SAME LANGUAGE as the table content.\n\n"
        "Table:\n"
    )
    provider = get_llm_provider()

    async def caption_one(tbl: ExtractedTable) -> None:
        async with _CAPTION_SEMAPHORE:
            try:
                table_md = tbl.content_markdown
                if len(table_md) > settings.NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS:
                    table_md = table_md[:settings.NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS] + "\n...(truncated)"
                from app.services.llm.types import LLMMessage
                message = LLMMessage(role="user", content=_TABLE_PROMPT + table_md)
                result = await asyncio.to_thread(provider.complete, [message])
                if result:
                    tbl.caption = " ".join(str(result).strip().split())[:500]
            except Exception as e:
                logger.debug(f"[caption_worker] table {tbl.table_id} caption failed: {e}")

    await asyncio.gather(*[caption_one(tbl) for tbl in tables])


async def _reenrich_embeddings(
    document_id: int,
    workspace_id: int,
    db_images: list[DocumentImage],
    db_tables: list[DocumentTable],
    document: Document,
) -> None:
    """
    Re-embed chunks that now have image/table caption descriptions.
    Only re-embeds chunks whose image_refs or table_refs have captions.
    """
    import json as _json

    if not document.raw_chunks_json:
        # Already cleared by embed_worker — need to re-fetch from ChromaDB
        # and update only the enriched text in-place
        logger.debug(f"[caption_worker] doc={document_id} raw_chunks_json gone — skipping re-embed")
        return

    try:
        chunks_data: list[dict] = _json.loads(document.raw_chunks_json)
    except Exception:
        return

    if not chunks_data:
        return

    img_captions  = {img.image_id: img.caption for img in db_images if img.caption}
    tbl_captions  = {tbl.table_id: tbl.caption  for tbl in db_tables if tbl.caption}

    # Rebuild enriched texts
    updated_texts: dict[str, str] = {}   # chunk_id → new text
    for c in chunks_data:
        extra: list[str] = []
        for iid in c.get("image_refs", []):
            cap = img_captions.get(iid)
            if cap:
                extra.append(f"[Image on page {c['page_no']}]: {cap}")
        for tid in c.get("table_refs", []):
            cap = tbl_captions.get(tid)
            if cap:
                extra.append(f"[Table on page {c['page_no']}]: {cap}")
        if extra:
            enriched = c["content"] + "\n\n" + "\n".join(extra)
            chunk_id = f"doc_{document_id}_chunk_{c['chunk_index']}"
            updated_texts[chunk_id] = enriched

    if not updated_texts:
        return

    embedder     = get_embedding_service()
    vector_store = get_vector_store(workspace_id)

    ids   = list(updated_texts.keys())
    texts = list(updated_texts.values())
    new_embeddings = embedder.embed_texts(texts)

    # ChromaDB upsert — update existing vectors with enriched text
    vector_store.update_documents(ids=ids, embeddings=new_embeddings, documents=texts)
    logger.info(
        f"[caption_worker] doc={document_id} re-embedded "
        f"{len(ids)} chunks with captions"
    )

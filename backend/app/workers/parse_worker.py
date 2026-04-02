"""
Parse Worker
============
Consumes hrag.parse queue.

Responsibilities:
  1. Download raw file from MinIO (hrag-uploads bucket)
  2. Run Docling / HunyuanOCR — zero LLM calls
  3. Save markdown, images, tables to DB  →  status = CHUNKING
  4. Dispatch three independent messages:
       EmbedMessage   → hrag.embed
       CaptionMessage → hrag.caption
       KGMessage      → hrag.kg  (routing_key = workspace_id)
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path

from sqlalchemy import delete, select

from app.core.database import async_session_maker
from app.models.document_type import DocumentType as _DocumentType  # noqa: F401 — ensures SQLAlchemy mapper resolves "DocumentType" relationship
from app.models.document import Document, DocumentImage, DocumentStatus, DocumentTable
from app.queue import connection as mq
from app.queue.messages import CaptionMessage, EmbedMessage, KGMessage, ParseMessage
from app.services.deep_document_parser import DeepDocumentParser
from app.services.storage_service import get_storage_service

logger = logging.getLogger(__name__)


async def handle_parse(payload: dict) -> None:
    msg = ParseMessage(**payload)
    logger.info(f"[parse_worker] doc={msg.document_id} file={msg.original_filename}")
    start = time.time()

    async with async_session_maker() as db:
        result = await db.execute(select(Document).where(Document.id == msg.document_id))
        document = result.scalar_one_or_none()
        if document is None:
            logger.error(f"[parse_worker] doc={msg.document_id} not found — skipping")
            return

        tmp_path: Path | None = None
        try:
            document.status = DocumentStatus.PARSING
            await db.commit()

            # ── Download raw file from MinIO ────────────────────────────────
            storage = get_storage_service()
            try:
                file_bytes = await storage.download_file(msg.minio_key)
                logger.info(
                    f"[parse_worker] doc={msg.document_id} downloaded "
                    f"{len(file_bytes)} bytes from MinIO key={msg.minio_key}"
                )
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                logger.error(
                    f"[parse_worker] doc={msg.document_id} MINIO DOWNLOAD FAILED "
                    f"({type(e).__name__}): key={msg.minio_key} — {e}",
                    exc_info=True,
                )
                raise
            ext = Path(msg.minio_key).suffix.lower()

            # Write to temp file (Docling requires a file path)
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)

            # ── Check if scanned PDF → switch to OCRING ─────────────────────
            is_scanned = False
            if ext == ".pdf":
                try:
                    from app.services.ocr_service import get_ocr_service
                    ocr_svc = get_ocr_service()
                    is_scanned = await asyncio.to_thread(
                        ocr_svc.is_scanned_pdf, str(tmp_path)
                    )
                    if is_scanned:
                        document.status = DocumentStatus.OCRING
                        await db.commit()
                except Exception:
                    pass  # fall back to PARSING

            # ── Extract digital signatures (native PDF only) ────────────────
            if ext == ".pdf" and not is_scanned:
                try:
                    from app.services.ocr_service import get_ocr_service as _get_ocr  # noqa: F811
                    from app.services.digital_signature_service import extract_digital_signatures
                    sigs = await asyncio.to_thread(
                        extract_digital_signatures, str(tmp_path)
                    )
                    if sigs:
                        document.digital_signatures = sigs
                        await db.commit()
                        logger.info(
                            f"[parse_worker] doc={msg.document_id} "
                            f"found {len(sigs)} digital signature(s)"
                        )
                except Exception as _sig_err:
                    logger.warning(
                        f"[parse_worker] doc={msg.document_id} "
                        f"digital signature extraction failed (non-fatal): {_sig_err}"
                    )

            # ── Phase: structural parse (ZERO LLM) ─────────────────────────
            parser = DeepDocumentParser(workspace_id=msg.workspace_id)
            parsed = await parser.parse_structure(
                file_path=str(tmp_path),
                document_id=msg.document_id,
                original_filename=msg.original_filename,
            )

            # ── Persist markdown + counts ───────────────────────────────────
            s3_key = await storage.upload_markdown(
                workspace_id=msg.workspace_id,
                document_id=msg.document_id,
                content=parsed.markdown,
            )
            document.markdown_s3_key = s3_key
            document.page_count       = parsed.page_count
            document.table_count      = parsed.tables_count
            document.parser_version   = (
                "docling"
                if DeepDocumentParser.is_docling_supported(str(tmp_path))
                else "legacy"
            )
            await db.commit()

            # ── Classify document type & extract rich header ────────────────────────
            try:
                from app.services.document_type_classifier import classify_with_llm
                from app.models.document_type import DocumentType as _DT
                
                # If PDF, attempt 1st page OCR for perfect header extraction
                text_for_llm = parsed.markdown
                if str(tmp_path).lower().endswith(".pdf"):
                    try:
                        import fitz
                        from app.services.ocr_service import get_ocr_service
                        logger.info(f"[parse_worker] doc={msg.document_id} extracting page 1 for reliable header OCR")
                        doc_fitz = fitz.open(str(tmp_path))
                        if doc_fitz.page_count > 0:
                            page_pixmap = doc_fitz[0].get_pixmap(matrix=fitz.Matrix(150/72, 150/72), alpha=False)
                            img_bytes = page_pixmap.tobytes("png")
                            doc_fitz.close()
                            
                            ocr_svc = get_ocr_service()
                            if ocr_svc._local:
                                page_texts = await ocr_svc._ocr_pages_local([img_bytes])
                            else:
                                page_texts = await ocr_svc._ocr_pages_api([img_bytes])
                                
                            if page_texts and page_texts[0].strip():
                                text_for_llm = page_texts[0]
                                logger.info(f"[parse_worker] doc={msg.document_id} page 1 OCR successful ({len(text_for_llm)} chars)")
                    except Exception as e_pdf:
                        logger.warning(f"[parse_worker] doc={msg.document_id} page 1 OCR failed, fallback to markdown: {e_pdf}")

                meta_res = await classify_with_llm(text_for_llm) if text_for_llm else {}
                slug = meta_res.get("slug")
                
                if slug:
                    dt_result = await db.execute(
                        select(_DT).where(_DT.slug == slug, _DT.is_active.is_(True))
                    )
                    dt = dt_result.scalar_one_or_none()
                    if dt:
                        document.document_type_id = dt.id
                        
                # Update all rich fields
                document.document_number = meta_res.get("document_number")
                document.document_title = meta_res.get("document_title")
                document.location = meta_res.get("location")
                document.issuing_agency = meta_res.get("issuing_agency")
                document.parent_agency = meta_res.get("parent_agency")
                document.published_date = meta_res.get("published_date")

                await db.commit()
                logger.info(f"[parse_worker] doc={msg.document_id} metadata classified: {meta_res}")
            except Exception as _cls_err:
                logger.warning(
                    f"[parse_worker] doc={msg.document_id} "
                    f"document type classification failed (non-fatal): {_cls_err}"
                )

            # ── Persist images (no captions yet) ───────────────────────────
            await db.execute(
                delete(DocumentImage).where(DocumentImage.document_id == msg.document_id)
            )
            await db.commit()
            for img in parsed.images:
                db.add(DocumentImage(
                    document_id=msg.document_id,
                    image_id=img.image_id,
                    page_no=img.page_no,
                    file_path=img.file_path,
                    caption=img.caption,   # empty at this point
                    width=img.width,
                    height=img.height,
                    mime_type=img.mime_type,
                ))
            if parsed.images:
                document.image_count = len(parsed.images)
                await db.commit()

            # ── Persist tables (no captions yet) ───────────────────────────
            await db.execute(
                delete(DocumentTable).where(DocumentTable.document_id == msg.document_id)
            )
            await db.commit()
            for tbl in parsed.tables:
                db.add(DocumentTable(
                    document_id=msg.document_id,
                    table_id=tbl.table_id,
                    page_no=tbl.page_no,
                    content_markdown=tbl.content_markdown,
                    caption="",   # empty at this point
                    num_rows=tbl.num_rows,
                    num_cols=tbl.num_cols,
                ))
            if parsed.tables:
                await db.commit()

            # ── Store raw chunks in ChromaDB (via EmbedMessage) ────────────
            # Attach chunk data into a JSON column for the embed worker
            # so we don't need to re-parse the file
            import json
            document.raw_chunks_json = json.dumps([
                {
                    "content":         c.content,
                    "chunk_index":     c.chunk_index,
                    "source_file":     c.source_file,
                    "page_no":         c.page_no,
                    "heading_path":    c.heading_path,
                    "image_refs":      c.image_refs,
                    "table_refs":      c.table_refs,
                    "has_table":       c.has_table,
                    "has_code":        c.has_code,
                    "document_number": document.document_number or "",
                }
                for c in parsed.chunks
            ])
            document.status = DocumentStatus.CHUNKING
            elapsed_ms = int((time.time() - start) * 1000)
            document.processing_time_ms = elapsed_ms
            await db.commit()
            logger.info(
                f"[parse_worker] doc={msg.document_id} parsed in {elapsed_ms}ms "
                f"— {len(parsed.chunks)} chunks, {len(parsed.images)} images, "
                f"{parsed.tables_count} tables"
            )

            # ── Dispatch 3 independent sub-tasks ───────────────────────────
            await mq.publish(
                mq.EXCHANGE_EMBED, "embed",
                EmbedMessage(
                    document_id=msg.document_id,
                    workspace_id=msg.workspace_id,
                ).model_dump(mode="json"),
            )
            await mq.publish(
                mq.EXCHANGE_CAPTION, "caption",
                CaptionMessage(
                    document_id=msg.document_id,
                    workspace_id=msg.workspace_id,
                ).model_dump(mode="json"),
            )
            await mq.publish(
                mq.EXCHANGE_KG, str(msg.workspace_id),
                KGMessage(
                    document_id=msg.document_id,
                    workspace_id=msg.workspace_id,
                    markdown=parsed.markdown,
                ).model_dump(mode="json"),
            )
            logger.info(
                f"[parse_worker] doc={msg.document_id} dispatched "
                f"embed + caption + kg messages"
            )

        except Exception as e:
            logger.error(f"[parse_worker] doc={msg.document_id} FAILED: {e}", exc_info=True)
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)[:500]
            await db.commit()
            raise
        finally:
            # Always clean up temp file
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

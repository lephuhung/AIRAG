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

from sqlalchemy import select

from app.core.database import async_session_maker
from app.models.document_type import DocumentType as _DocumentType  # noqa: F401 — ensures SQLAlchemy mapper resolves "DocumentType" relationship
from app.models.document import Document, DocumentImage, DocumentStatus, DocumentTable
from app.queue import connection as mq
from app.queue.messages import CaptionMessage, EmbedMessage, KGMessage, ParseMessage
from app.services.agents.v2.persistence.source_identity import (
    RevisionBuildProfile,
)
from app.services.agents.v2.persistence.document_views import (
    ChunkRecord,
    build_structure_artifact,
    record_revision_chunk_rows,
    revision_markdown_key,
    revision_structure_key,
)
from app.services.parsing.deep_document_parser import DeepDocumentParser
from app.services.storage_service import get_storage_service
from app.workers.utils import (
    FinalizeOutcome,
    apply_finalize_outcome,
    delete_stage_children,
    finalize_revision_if_complete,
    load_revision_execution,
    mark_revision_building,
    mark_revision_failed,
    record_parse_artifacts,
)

logger = logging.getLogger(__name__)


async def handle_parse(payload: dict) -> None:
    msg = ParseMessage(**payload)
    logger.info(
        f"[parse_worker] doc={msg.document_id} rev={msg.revision_id} "
        f"profile={msg.build_profile} file={msg.original_filename}"
    )
    start = time.time()
    profile = RevisionBuildProfile(msg.build_profile)

    async with async_session_maker() as db:
        result = await db.execute(
            select(Document).where(Document.id == msg.document_id)
        )
        document = result.scalar_one_or_none()
        if document is None:
            logger.error(f"[parse_worker] doc={msg.document_id} not found — skipping")
            return

        # Revision-owned execution decision: a terminal revision or a
        # tombstoned source is a no-op dead-letter (never re-run).
        execution = await load_revision_execution(
            db, revision_id=msg.revision_id, document_id=msg.document_id
        )
        if not execution.run:
            logger.info(
                f"[parse_worker] doc={msg.document_id} rev={msg.revision_id} "
                f"no-op ({execution.reason})"
            )
            return

        await mark_revision_building(db, msg.revision_id)
        document.is_chat_upload = msg.is_chat_upload
        await db.commit()

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

            # ── Extract digital signatures (native PDF only) ────────────────
            if ext == ".pdf":
                try:
                    from app.services.parsing.ocr_service import get_ocr_service as _get_ocr  # noqa: F811
                    from app.services.parsing.digital_signature_service import (
                        extract_digital_signatures,
                    )

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
            # The markdown object is keyed by THIS revision (copy-on-write): a
            # reindex uploads a new object and never overwrites a published
            # revision's markdown. ``Document.markdown_s3_key`` is kept as the
            # v1/UI projection only — it never decides v2 retrieval.
            s3_key = await storage.upload_markdown(
                workspace_id=msg.workspace_id,
                document_id=msg.document_id,
                content=parsed.markdown,
                key=revision_markdown_key(
                    msg.workspace_id, msg.document_id, msg.revision_id
                ),
            )
            document.markdown_s3_key = s3_key
            document.page_count = parsed.page_count
            document.table_count = parsed.tables_count
            # Reflect the pipeline that ACTUALLY ran (ocr / docling / legacy),
            # not merely whether the format is Docling-capable.
            document.parser_version = parsed.parser
            await db.commit()

            # ── Structure artifact + stable chunk locators ──────────────────
            # ``document_revision_chunks`` holds the stable locator rows and the
            # structure artifact holds the payloads, so a viewer can rebuild a
            # locator from a stored id without reading mutable document state.
            non_empty_chunks = [
                c for c in parsed.chunks if c.content and c.content.strip()
            ]
            chunk_records = [
                ChunkRecord(
                    chunk_id="",
                    ordinal=c.chunk_index,
                    content=c.content,
                    page_no=c.page_no,
                    heading_path=list(c.heading_path),
                    source_file=c.source_file,
                    image_refs=list(c.image_refs),
                    table_refs=list(c.table_refs),
                    has_table=c.has_table,
                    has_code=c.has_code,
                )
                for c in non_empty_chunks
            ]
            chunk_records = await record_revision_chunk_rows(
                db, msg.revision_id, chunk_records
            )
            structure_key = revision_structure_key(
                msg.workspace_id, msg.document_id, msg.revision_id
            )
            await storage.upload_artifact(
                structure_key,
                build_structure_artifact(
                    msg.revision_id, msg.document_id, chunk_records
                ),
                "application/json",
            )
            await db.commit()

            # ── Record the parse-stage artifacts on the REVISION manifest ──
            await record_parse_artifacts(
                db,
                msg.revision_id,
                profile,
                markdown_artifact_key=s3_key,
                structure_artifact_key=structure_key,
            )
            await db.commit()

            # ── Classify document type & extract rich header ────────────────────────
            try:
                from app.services.document_type_classifier import classify_with_llm
                from app.services.parsing.ocr_service import strip_ocr_layout
                from app.models.document_type import DocumentType as _DT

                # OCR-path markdown carries administrative-layout HTML — feed the
                # classifier clean text (no-op for Docling/native documents).
                text_for_llm = strip_ocr_layout(parsed.markdown)
                # If PDF parsed via Docling/legacy, re-OCR page 1 for reliable
                # header extraction (native PDF text extraction may mangle the
                # header layout). When the whole doc already went through the
                # OCR pipeline the markdown IS OCR output — skip the extra call.
                if str(tmp_path).lower().endswith(".pdf") and parsed.parser != "ocr":
                    try:
                        import fitz
                        from app.services.parsing.ocr_service import get_ocr_service

                        logger.info(
                            f"[parse_worker] doc={msg.document_id} extracting page 1 for reliable header OCR"
                        )
                        doc_fitz = fitz.open(str(tmp_path))
                        if doc_fitz.page_count > 0:
                            page_pixmap = doc_fitz[0].get_pixmap(
                                matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False
                            )
                            img_bytes = page_pixmap.tobytes("png")
                            doc_fitz.close()

                            ocr_svc = get_ocr_service()
                            if ocr_svc._local:
                                page_texts = await ocr_svc._ocr_pages_local([img_bytes])
                            else:
                                page_texts = await ocr_svc._ocr_pages_api([img_bytes])

                            if page_texts and page_texts[0].strip():
                                text_for_llm = page_texts[0]
                                logger.info(
                                    f"[parse_worker] doc={msg.document_id} page 1 OCR successful ({len(text_for_llm)} chars)"
                                )
                    except Exception as e_pdf:
                        logger.warning(
                            f"[parse_worker] doc={msg.document_id} page 1 OCR failed, fallback to markdown: {e_pdf}"
                        )

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
                logger.info(
                    f"[parse_worker] doc={msg.document_id} metadata classified: {meta_res}"
                )
            except Exception as _cls_err:
                logger.warning(
                    f"[parse_worker] doc={msg.document_id} "
                    f"document type classification failed (non-fatal): {_cls_err}"
                )

            # ── Hiệu lực pháp lý (validity) ─────────────────────────────────
            # Sau classifier để có document_number cho cross-match 2 chiều.
            try:
                from app.services.legal.validity_service import apply_validity

                await apply_validity(db, document, parsed.markdown)
            except Exception as _val_err:
                logger.warning(
                    f"[parse_worker] doc={msg.document_id} "
                    f"validity extraction failed (non-fatal): {_val_err}"
                )

            # ── Persist images (no captions yet) ───────────────────────────
            # Scope replacement to THIS revision: a prior revision's (or a
            # legacy NULL-revision) image rows stay readable.
            await delete_stage_children(
                db,
                document_id=msg.document_id,
                revision_id=msg.revision_id,
                images=True,
                tables=True,
            )
            await db.commit()
            for img in parsed.images:
                db.add(
                    DocumentImage(
                        document_id=msg.document_id,
                        revision_id=msg.revision_id,
                        image_id=img.image_id,
                        page_no=img.page_no,
                        file_path=img.file_path,
                        caption=img.caption,  # empty at this point
                        width=img.width,
                        height=img.height,
                        mime_type=img.mime_type,
                    )
                )
            if parsed.images:
                document.image_count = len(parsed.images)
                await db.commit()

            # ── Persist tables (no captions yet) ───────────────────────────
            for tbl in parsed.tables:
                db.add(
                    DocumentTable(
                        document_id=msg.document_id,
                        revision_id=msg.revision_id,
                        table_id=tbl.table_id,
                        page_no=tbl.page_no,
                        content_markdown=tbl.content_markdown,
                        caption="",  # empty at this point
                        num_rows=tbl.num_rows,
                        num_cols=tbl.num_cols,
                    )
                )
            if parsed.tables:
                await db.commit()

            # ── Store raw chunks in ChromaDB (via EmbedMessage) ────────────
            # ``raw_chunks_json`` is the document-level v1/UI mirror only; the
            # authoritative revision-qualified payload is the structure
            # artifact written above. The embed worker prefers the revision's
            # own structure artifact and falls back to this column for legacy
            # in-flight messages.
            import json

            document.raw_chunks_json = json.dumps(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "chunk_index": c.ordinal,
                        "source_file": c.source_file,
                        "page_no": c.page_no,
                        "heading_path": c.heading_path,
                        "image_refs": c.image_refs,
                        "table_refs": c.table_refs,
                        "has_table": c.has_table,
                        "has_code": c.has_code,
                        "document_number": document.document_number or "",
                    }
                    for c in chunk_records
                ]
            )
            document.status = DocumentStatus.CHUNKING
            elapsed_ms = int((time.time() - start) * 1000)
            document.processing_time_ms = elapsed_ms
            await db.commit()
            logger.info(
                f"[parse_worker] doc={msg.document_id} parsed in {elapsed_ms}ms "
                f"— {len(chunk_records)} chunks (filtered {len(parsed.chunks) - len(chunk_records)} empty), "
                f"{len(parsed.images)} images, {parsed.tables_count} tables"
            )

            # ── Dispatch sub-tasks OR publish (parse-only / chat-upload mode) ─────
            if profile is RevisionBuildProfile.PARSE_ONLY:
                # Parse-only: no embed/caption/KG child stages. Verify + publish
                # the revision from its recorded manifest. This call IS the
                # final one for the profile, so incomplete artifacts are a real
                # failure (verified with expect_complete).
                await db.commit()
                result = await finalize_revision_if_complete(
                    msg.revision_id, expect_complete=True
                )
                # The document must never read INDEXED unless the revision
                # actually published; a verify failure mirrors FAILED — but only
                # while THIS revision is still the document's current pointer
                # (a superseded build's late failure must not fail a live one).
                await apply_finalize_outcome(
                    msg.document_id, result, revision_id=msg.revision_id
                )
                if result.outcome is FinalizeOutcome.PUBLISHED:
                    logger.info(
                        f"[parse_worker] doc={msg.document_id} rev={msg.revision_id} "
                        f"parse-only — published in {int((time.time() - start) * 1000)}ms"
                    )
                else:
                    logger.error(
                        f"[parse_worker] doc={msg.document_id} rev={msg.revision_id} "
                        f"parse-only — finalize outcome={result.outcome.value} "
                        f"({result.failure_stage}:{result.failure_class})"
                    )
            elif profile is RevisionBuildProfile.CHAT_UPLOAD:
                # Chat-upload: parse → embed (skip KG and caption for speed)
                await mq.publish(
                    mq.EXCHANGE_EMBED,
                    "embed",
                    EmbedMessage(
                        document_id=msg.document_id,
                        workspace_id=msg.workspace_id,
                        revision_id=msg.revision_id,
                        build_profile=msg.build_profile,
                    ).model_dump(mode="json"),
                )
                logger.info(
                    f"[parse_worker] doc={msg.document_id} rev={msg.revision_id} "
                    f"chat-upload — dispatched embed (skip caption+kg) in "
                    f"{int((time.time() - start) * 1000)}ms"
                )
            else:
                # Full pipeline: embed + caption + KG
                await mq.publish(
                    mq.EXCHANGE_EMBED,
                    "embed",
                    EmbedMessage(
                        document_id=msg.document_id,
                        workspace_id=msg.workspace_id,
                        revision_id=msg.revision_id,
                        build_profile=msg.build_profile,
                    ).model_dump(mode="json"),
                )
                await mq.publish(
                    mq.EXCHANGE_CAPTION,
                    "caption",
                    CaptionMessage(
                        document_id=msg.document_id,
                        workspace_id=msg.workspace_id,
                        revision_id=msg.revision_id,
                        build_profile=msg.build_profile,
                    ).model_dump(mode="json"),
                )
                # New workspace → its KG queue may not exist yet (the kg
                # worker's poller discovers workspaces every ~30s) and an
                # unroutable publish is silently DROPPED. Declare first.
                await mq.ensure_kg_queue(msg.workspace_id)
                await mq.publish(
                    mq.EXCHANGE_KG,
                    str(msg.workspace_id),
                    KGMessage(
                        document_id=msg.document_id,
                        workspace_id=msg.workspace_id,
                        revision_id=msg.revision_id,
                        build_profile=msg.build_profile,
                        markdown_s3_key=s3_key,
                    ).model_dump(mode="json"),
                )
                logger.info(
                    f"[parse_worker] doc={msg.document_id} rev={msg.revision_id} "
                    f"dispatched embed + caption + kg messages"
                )

        except Exception as e:
            logger.error(
                f"[parse_worker] doc={msg.document_id} FAILED: {e}", exc_info=True
            )
            # The session may be in an aborted state if the failure came from a
            # DB flush — roll back first so the FAILED status can be written.
            await db.rollback()
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)[:500]
            await db.commit()
            # Terminalize the revision too: it is immutable and never resumed,
            # so the child stage messages still in flight dead-letter instead of
            # rebuilding a failed revision.
            await mark_revision_failed(
                db, msg.revision_id, stage="parse", error_class=type(e).__name__
            )
            raise
        finally:
            # Always clean up temp file
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

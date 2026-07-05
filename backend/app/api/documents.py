from __future__ import annotations

import os
import io
import re
import uuid
import asyncio
import hashlib
import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    status,
    UploadFile,
    File,
    Body,
)
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.deps import (
    get_db,
    get_current_active_user,
    verify_workspace_access,
    verify_workspace_manage_access,
)
from app.core.exceptions import NotFoundError
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentImage, DocumentStatus
from app.models.user import User
from app.schemas.document import (
    DocumentResponse,
    DocumentUploadResponse,
    DocumentUpdate,
)
from app.schemas.rag import DocumentImageResponse

logger = logging.getLogger(__name__)


async def _find_duplicate_document(
    db: AsyncSession,
    workspace_id: uuid.UUID,
    content_hash: str,
    file_size: int,
) -> Document | None:
    """Find an existing INDEXED document with the same content within scope.

    Scope rules (visibility-aware) — pick the broadest sharing boundary that
    still respects confidentiality, in priority order:
    - PUBLIC target → match any INDEXED doc in any other PUBLIC workspace.
      Public content is shared knowledge and figures are served from an
      unauthenticated static mount, so reusing across owners is safe.
    - Has tenant_id → match any INDEXED doc whose workspace shares that
      tenant_id (Phase 2 behaviour).
    - PERSONAL (no tenant) → match across the SAME owner's workspaces only, so
      one user's parsed file is reused only by that same user, never leaked.
    - No owner/tenant (legacy) → restrict to the SAME workspace.

    A match in the same workspace is preferred (ordered first). file_size is
    compared alongside the hash as a cheap guard against hash collisions.
    """
    ws = await db.get(KnowledgeBase, workspace_id)
    tenant_id = ws.tenant_id if ws is not None else None
    visibility = ws.visibility if ws is not None else None
    owner_id = ws.owner_id if ws is not None else None

    stmt = (
        select(Document)
        .join(KnowledgeBase, Document.workspace_id == KnowledgeBase.id)
        .where(
            Document.content_hash == content_hash,
            Document.file_size == file_size,
            Document.status == DocumentStatus.INDEXED,
        )
    )
    if visibility == "public":
        stmt = stmt.where(KnowledgeBase.visibility == "public")
    elif tenant_id is not None:
        stmt = stmt.where(KnowledgeBase.tenant_id == tenant_id)
    elif owner_id is not None:
        stmt = stmt.where(KnowledgeBase.owner_id == owner_id)
    else:
        stmt = stmt.where(Document.workspace_id == workspace_id)

    # Prefer a match that already lives in the target workspace.
    stmt = stmt.order_by(
        (Document.workspace_id == workspace_id).desc(),
        Document.created_at.asc(),
    )
    result = await db.execute(stmt)
    return result.scalars().first()


async def _is_public_workspace(db: AsyncSession, workspace_id: uuid.UUID) -> bool:
    """True if the workspace is publicly visible."""
    ws = await db.get(KnowledgeBase, workspace_id)
    return ws is not None and ws.visibility == "public"


async def _public_duplicate_response(
    db: AsyncSession, dup: Document
) -> DocumentUploadResponse:
    """Build the warning response for a duplicate found in another PUBLIC
    workspace. The caller is responsible for deleting the uploaded file and the
    pending record; here we only assemble the user-facing payload pointing at the
    existing public copy."""
    src_ws = await db.get(KnowledgeBase, dup.workspace_id)
    ws_name = src_ws.name if src_ws is not None else None
    where = f' ("{ws_name}")' if ws_name else ""
    return DocumentUploadResponse(
        id=dup.id,
        filename=dup.original_filename,
        status=dup.status,
        message=(
            f"Tài liệu này đã tồn tại trong một không gian công khai khác{where} "
            f"— không xử lý lại. Bạn có thể dùng trực tiếp bản đã có."
        ),
        duplicate=True,
        duplicate_document_id=dup.id,
        duplicate_workspace_id=dup.workspace_id,
        duplicate_workspace_name=ws_name,
    )


def _copy_vector_chunks(
    source_workspace_id: uuid.UUID,
    source_document_id: uuid.UUID,
    target_workspace_id: uuid.UUID,
    new_document_id: uuid.UUID,
) -> int:
    """Copy a document's vector chunks from one workspace collection to another.

    Re-keys ids to the new document and rewrites document_id / workspace_id in
    the metadata. Image URLs in the metadata are intentionally LEFT UNCHANGED —
    they keep pointing at the source workspace's static path, which resolves for
    same-tenant users (the static mount is shared, unauthenticated), so figures
    render without copying any image files.

    Runs synchronously (ChromaDB client is blocking) — call via asyncio.to_thread.
    Returns the number of chunks copied.
    """
    from app.services.embedding.vector_store import get_vector_store

    src = get_vector_store(source_workspace_id)
    data = src.get_document_chunks(source_document_id, include_embeddings=True)
    ids = data["ids"]
    if not ids:
        return 0

    embeddings = data["embeddings"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    new_ids: list[str] = []
    new_metas: list[dict] = []
    for meta in metadatas:
        m = dict(meta)
        chunk_index = m.get("chunk_index")
        new_ids.append(f"doc_{new_document_id}_chunk_{chunk_index}")
        m["document_id"] = str(new_document_id)
        m["workspace_id"] = str(target_workspace_id)
        new_metas.append(m)

    tgt = get_vector_store(target_workspace_id)
    tgt.add_documents(
        ids=new_ids,
        embeddings=[list(e) for e in embeddings],
        documents=documents,
        metadatas=new_metas,
    )

    # The BM25 index for the target workspace is built from ChromaDB and cached;
    # force a rebuild so the freshly copied chunks become lexically searchable.
    try:
        from app.services.retrieval.bm25_index import invalidate_cache

        invalidate_cache(target_workspace_id)
    except Exception as e:
        logger.warning(f"[clone] BM25 invalidate failed for {target_workspace_id}: {e}")

    return len(new_ids)


async def _clone_document_to_workspace(
    db: AsyncSession,
    source_doc: Document,
    new_doc: Document,
) -> None:
    """Populate ``new_doc`` (already created in the target workspace) by reusing
    the parsed artifacts of ``source_doc`` instead of re-running the pipeline.

    Reuses: parsed markdown (copied to the new doc's key) + vector chunks
    (copied across collections with their existing embeddings). Skips parse,
    embed, caption and KG entirely. The new document is marked INDEXED.
    """
    from app.services.storage_service import get_storage_service

    storage = get_storage_service()

    # 1. Copy parsed markdown to the new doc's key (best-effort).
    if source_doc.markdown_s3_key:
        try:
            md = await storage.download_markdown(source_doc.markdown_s3_key)
            new_doc.markdown_s3_key = await storage.upload_markdown(
                new_doc.workspace_id, new_doc.id, md
            )
        except Exception as e:
            logger.warning(
                f"[clone] markdown copy failed src={source_doc.id} "
                f"new={new_doc.id}: {e}"
            )

    # 2. Copy vector chunks (with embeddings) into the target collection.
    chunk_count = await asyncio.to_thread(
        _copy_vector_chunks,
        source_doc.workspace_id,
        source_doc.id,
        new_doc.workspace_id,
        new_doc.id,
    )

    # 3. Carry over parsed counts/metadata and mark the doc fully indexed.
    new_doc.chunk_count = chunk_count or source_doc.chunk_count
    new_doc.page_count = source_doc.page_count
    new_doc.image_count = source_doc.image_count
    new_doc.table_count = source_doc.table_count
    new_doc.parser_version = source_doc.parser_version
    new_doc.document_type_id = source_doc.document_type_id
    new_doc.document_number = source_doc.document_number
    new_doc.document_title = source_doc.document_title
    new_doc.location = source_doc.location
    new_doc.issuing_agency = source_doc.issuing_agency
    new_doc.parent_agency = source_doc.parent_agency
    new_doc.published_date = source_doc.published_date
    new_doc.digital_signatures = source_doc.digital_signatures
    new_doc.embed_done = True
    new_doc.captions_done = True
    new_doc.kg_done = True
    new_doc.status = DocumentStatus.INDEXED
    await db.commit()
    logger.info(
        f"[clone] new doc={new_doc.id} in ws={new_doc.workspace_id} cloned from "
        f"src={source_doc.id} (ws={source_doc.workspace_id}), {new_doc.chunk_count} chunks"
    )


def _inject_images_from_db(
    markdown: str,
    images: list[DocumentImage],
    workspace_id: uuid.UUID,
) -> str:
    """Replace remaining <!-- image --> placeholders with real image markdown.

    Used as a safety net when the parser didn't inject them during processing.
    Images are matched in insertion order (by primary key) which mirrors the
    order of pictures in the original Docling document.
    """
    img_iter = iter(images)

    def _replacer(match):
        try:
            img = next(img_iter)
            url = f"/static/doc-images/kb_{workspace_id}/images/{img.image_id}.png"
            caption = (img.caption or "").replace("[", "").replace("]", "")
            return f"\n![{caption}]({url})\n"
        except StopIteration:
            return ""

    return re.sub(r"<!--\s*image\s*-->", _replacer, markdown)


router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = settings.BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# MIME type mapping for common extensions
_EXT_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _mime_for_ext(ext: str) -> str:
    return _EXT_TO_MIME.get(ext.lower(), "application/octet-stream")


@router.get("/workspace/{workspace_id}", response_model=list[DocumentResponse])
async def list_documents(
    workspace_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """List all documents in a knowledge base."""
    await verify_workspace_access(workspace_id, user, db)

    result = await db.execute(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
    )
    return result.scalars().all()


async def process_document_background(
    document_id: uuid.UUID, file_path: str, workspace_id: uuid.UUID
):
    """Legacy fallback: process document inline when RabbitMQ is unavailable."""
    from app.core.database import async_session_maker
    from app.services.retrieval.rag_service import get_rag_service

    async with async_session_maker() as db:
        try:
            rag_service = get_rag_service(db, workspace_id)
            await rag_service.process_document(document_id, file_path)
            logger.info(
                f"Document {document_id} processed successfully (fallback mode)"
            )
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")


@router.post("/upload/{workspace_id}", response_model=DocumentUploadResponse)
async def upload_document(
    workspace_id: uuid.UUID,
    file: UploadFile = File(...),
    x_chat_upload: bool = Header(default=False, alias="X-Chat-Upload"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Upload a document to a knowledge base and store the raw file in MinIO.

    Header X-Chat-Upload: set to true when uploading from chat context.
    This causes the file to be processed in parse-only mode (skip KG/caption
    workers) for faster chat attachment.
    """
    await verify_workspace_access(workspace_id, user, db)

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}",
        )

    try:
        content = await file.read()
    except (ConnectionResetError, OSError) as exc:
        # Client disconnected mid-upload (Broken pipe / Connection reset)
        logger.warning(
            f"Client disconnected during file upload for workspace {workspace_id}: {exc}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload interrupted — client disconnected. Please retry.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB",
        )

    # Duplicate detection: hash the raw bytes and look for an existing INDEXED
    # document with the same content in scope. A match in the SAME workspace is
    # returned as-is (idempotent) — no re-upload, no re-processing.
    content_hash = hashlib.sha256(content).hexdigest()
    dup = await _find_duplicate_document(db, workspace_id, content_hash, len(content))
    if dup is not None and dup.workspace_id == workspace_id:
        logger.info(
            f"[upload] duplicate of doc={dup.id} in same workspace "
            f"{workspace_id} (hash={content_hash[:12]}) — returning existing"
        )
        return DocumentUploadResponse(
            id=dup.id,
            filename=dup.original_filename,
            status=dup.status,
            message="Document already exists in this knowledge base.",
        )
    # Cross-workspace duplicate in another PUBLIC workspace: don't duplicate the
    # work — warn the user and stop. No MinIO object has been written yet in this
    # flow, so there is nothing to delete.
    if dup is not None and await _is_public_workspace(db, workspace_id):
        logger.info(
            f"[upload] duplicate of doc={dup.id} in another public workspace "
            f"{dup.workspace_id} (hash={content_hash[:12]}) — warn + skip"
        )
        return await _public_duplicate_response(db, dup)
    # Cross-workspace duplicate within the same tenant/owner: reuse its parsed
    # artifacts (markdown + vector chunks) instead of re-running the pipeline.
    reuse_source = dup if dup is not None else None

    # Sanitize original filename: keep alphanumeric, dots, dashes, underscores
    import re as _re

    safe_stem = _re.sub(r"[^\w\-.]", "_", Path(file.filename).stem)
    filename = f"{safe_stem}{ext}"

    # Create DB record first to get document.id for the MinIO key
    document = Document(
        workspace_id=workspace_id,
        filename=filename,
        original_filename=file.filename,
        file_type=ext[1:],
        file_size=len(content),
        content_hash=content_hash,
        status=DocumentStatus.PENDING,
        uploaded_by=user.id,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Upload raw file to MinIO hrag-uploads bucket
    from app.services.storage_service import get_storage_service

    storage = get_storage_service()
    upload_key = storage._make_upload_key(
        workspace_id, document.id, ext, is_chat_upload=x_chat_upload
    )
    try:
        await storage.upload_file(
            key=upload_key,
            data=content,
            content_type=_mime_for_ext(ext),
        )
        document.upload_s3_key = upload_key
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to upload file to MinIO for doc {document.id}: {e}")
        # Rollback document status to failed
        document.status = "failed"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file to storage: {str(e)}",
        )

    # Cross-workspace duplicate (same tenant): clone parsed artifacts instead of
    # queuing the parse pipeline. Raw file is still stored above so the new doc
    # is self-contained (e.g. for future re-index).
    if reuse_source is not None:
        try:
            await _clone_document_to_workspace(db, reuse_source, document)
            return DocumentUploadResponse(
                id=document.id,
                filename=document.original_filename,
                status=document.status,
                message="Document reused from an existing copy in your tenant.",
            )
        except Exception as e:
            # Clone failed — fall back to the normal parse pipeline rather than
            # leaving the document stuck.
            logger.warning(
                f"[upload] clone from doc={reuse_source.id} failed for "
                f"doc={document.id}: {e} — falling back to full parse"
            )
            document.status = DocumentStatus.PENDING
            document.embed_done = False
            document.captions_done = False
            document.kg_done = False
            await db.commit()

    # Publish parse task (immediate if webhook disabled, else MinIO event fires)
    if not settings.MINIO_WEBHOOK_ENABLED:
        try:
            from app.queue.publisher import publish_parse_task

            await publish_parse_task(
                document_id=document.id,
                workspace_id=workspace_id,
                minio_key=upload_key,
                original_filename=file.filename,
            )
            logger.info(
                f"Document {document.id} queued for processing (direct publish)"
            )
        except Exception as e:
            logger.error(
                f"Failed to publish parse task for doc {document.id}: {e}. "
                f"Rolling back document to FAILED."
            )
            document.status = DocumentStatus.FAILED
            document.error_message = f"Publish failed: {e}"
            await db.commit()
    else:
        logger.info(
            f"Document {document.id} uploaded to MinIO — "
            f"waiting for webhook event to trigger parse"
        )

    return DocumentUploadResponse(
        id=document.id,
        filename=document.original_filename,
        status=document.status,
        message="Document uploaded and queued for processing.",
    )


# ---------------------------------------------------------------------------
# Presigned-upload flow (frontend uploads directly to MinIO)
# ---------------------------------------------------------------------------


class PresignRequest(BaseModel):
    filename: str
    file_size: int
    content_type: str | None = None


class PresignResponse(BaseModel):
    document_id: uuid.UUID
    upload_url: str  # Presigned PUT URL pointing directly to MinIO
    minio_key: str  # Object key — needed for /confirm call


class ConfirmRequest(BaseModel):
    document_id: uuid.UUID


@router.post("/upload/{workspace_id}/presign", response_model=PresignResponse)
async def presign_upload(
    workspace_id: uuid.UUID,
    body: PresignRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Step 1 of direct-to-MinIO upload.

    Creates the Document record in PENDING state and returns a presigned PUT
    URL.  The frontend must PUT the file bytes directly to that URL, then call
    ``/confirm`` to trigger the parse pipeline.
    """
    await verify_workspace_access(workspace_id, user, db)

    ext = Path(body.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}",
        )

    if body.file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB",
        )

    import re as _re

    safe_stem = _re.sub(r"[^\w\-.]", "_", Path(body.filename).stem)
    filename = f"{safe_stem}{ext}"

    document = Document(
        workspace_id=workspace_id,
        filename=filename,
        original_filename=body.filename,
        file_type=ext[1:],
        file_size=body.file_size,
        status=DocumentStatus.PENDING,
        uploaded_by=user.id,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    from app.services.storage_service import get_storage_service

    storage = get_storage_service()
    upload_key = storage._make_upload_key(workspace_id, document.id, ext)

    content_type = body.content_type or _mime_for_ext(ext)
    try:
        presigned_url = await storage.generate_presigned_upload_url(
            key=upload_key,
            content_type=content_type,
        )
    except Exception as e:
        # Roll back the document record so the client can retry cleanly
        await db.delete(document)
        await db.commit()
        logger.error(f"Failed to generate presigned URL for doc {document.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable — could not generate upload URL.",
        )

    # DO NOT set upload_s3_key here — file is not uploaded yet.
    # It will be set in confirm_upload ONLY after MinIO verifies the object exists.
    # This prevents orphaned records when frontend fails to complete the upload.
    return PresignResponse(
        document_id=document.id,
        upload_url=presigned_url,
        minio_key=upload_key,
    )


@router.post("/upload/{workspace_id}/confirm", response_model=DocumentUploadResponse)
async def confirm_upload(
    workspace_id: uuid.UUID,
    body: ConfirmRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Step 2 of direct-to-MinIO upload.

    Call this after the frontend has successfully PUT the file to the presigned
    URL.  Verifies the object exists in MinIO then publishes a ParseMessage to
    kick off the pipeline.
    """
    result = await db.execute(
        select(Document).where(
            Document.id == body.document_id,
            Document.workspace_id == workspace_id,
        )
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise NotFoundError("Document", body.document_id)

    await verify_workspace_access(workspace_id, user, db)

    if document.status != DocumentStatus.PENDING:
        # Already queued / processing — idempotent response
        return DocumentUploadResponse(
            id=document.id,
            filename=document.original_filename,
            status=document.status,
            message="Document already queued for processing.",
        )

    # Verify the file actually landed in MinIO before queuing
    from app.services.storage_service import get_storage_service

    storage = get_storage_service()
    minio_key = document.upload_s3_key  # May be None if set by presign flow
    if not minio_key:
        # Fallback: reconstruct from document_id (presign no longer saves it prematurely)
        minio_key = storage._make_upload_key(
            workspace_id, document.id, Path(document.filename).suffix.lower()
        )
    try:
        exists = await storage.object_exists(minio_key)
    except Exception as e:
        logger.error(f"MinIO object_exists check failed for doc {document.id}: {e}")
        exists = True  # optimistic — proceed anyway
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File not found in storage. Please retry the upload.",
        )

    # Duplicate detection: the bytes are only on the server now (presign uploads
    # go straight to MinIO), so hash them here. If the same content already lives
    # in this workspace, drop the freshly uploaded object + pending record and
    # return the existing document.
    try:
        raw_bytes = await storage.download_file(minio_key)
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
    except Exception as e:
        logger.warning(
            f"[confirm_upload] could not hash doc={document.id} for dedup: {e} "
            f"— proceeding without dedup"
        )
        content_hash = None

    reuse_source: Document | None = None
    if content_hash is not None:
        dup = await _find_duplicate_document(
            db, workspace_id, content_hash, document.file_size
        )
        if dup is not None and dup.workspace_id == workspace_id:
            logger.info(
                f"[confirm_upload] duplicate of doc={dup.id} in same workspace "
                f"{workspace_id} (hash={content_hash[:12]}) — discarding upload"
            )
            try:
                await storage.delete_file(minio_key)
            except Exception as del_err:
                logger.warning(
                    f"[confirm_upload] failed to delete duplicate MinIO object "
                    f"{minio_key}: {del_err}"
                )
            await db.delete(document)
            await db.commit()
            return DocumentUploadResponse(
                id=dup.id,
                filename=dup.original_filename,
                status=dup.status,
                message="Document already exists in this knowledge base.",
            )
        # Cross-workspace duplicate in another PUBLIC workspace: delete the
        # freshly uploaded MinIO object + pending record, warn, and stop.
        if dup is not None and await _is_public_workspace(db, workspace_id):
            logger.info(
                f"[confirm_upload] duplicate of doc={dup.id} in another public "
                f"workspace {dup.workspace_id} (hash={content_hash[:12]}) — "
                f"deleting upload + skip"
            )
            try:
                await storage.delete_file(minio_key)
            except Exception as del_err:
                logger.warning(
                    f"[confirm_upload] failed to delete duplicate MinIO object "
                    f"{minio_key}: {del_err}"
                )
            resp = await _public_duplicate_response(db, dup)
            await db.delete(document)
            await db.commit()
            return resp
        document.content_hash = content_hash
        reuse_source = dup  # cross-workspace tenant/owner match (or None)

    # Now that MinIO has confirmed the file, record the key and queue the parse task
    document.upload_s3_key = minio_key
    await db.commit()

    # Cross-workspace duplicate (same tenant): clone parsed artifacts instead of
    # re-running the parse pipeline.
    if reuse_source is not None:
        try:
            await _clone_document_to_workspace(db, reuse_source, document)
            return DocumentUploadResponse(
                id=document.id,
                filename=document.original_filename,
                status=document.status,
                message="Document reused from an existing copy in your tenant.",
            )
        except Exception as e:
            logger.warning(
                f"[confirm_upload] clone from doc={reuse_source.id} failed for "
                f"doc={document.id}: {e} — falling back to full parse"
            )
            document.status = DocumentStatus.PENDING
            document.embed_done = False
            document.captions_done = False
            document.kg_done = False
            await db.commit()

    if not settings.MINIO_WEBHOOK_ENABLED:
        try:
            from app.queue.publisher import publish_parse_task

            await publish_parse_task(
                document_id=document.id,
                workspace_id=workspace_id,
                minio_key=minio_key,
                original_filename=document.original_filename,
            )
            logger.info(
                f"Document {document.id} queued for processing (presign confirm)"
            )
        except Exception as e:
            logger.error(
                f"Failed to publish parse task for doc {document.id}: {e}. "
                f"Rolling back document to FAILED."
            )
            document.status = DocumentStatus.FAILED
            document.error_message = f"Publish failed: {e}"
            await db.commit()
            # Clean up MinIO file so it doesn't become orphaned
            try:
                await storage.delete_file(minio_key)
                logger.info(f"[confirm_upload] Cleaned up orphaned MinIO file: {minio_key}")
            except Exception as del_err:
                logger.warning(
                    f"[confirm_upload] Failed to delete MinIO file after publish failure: "
                    f"{del_err} (doc={document.id}, key={minio_key})"
                )
    else:
        logger.info(
            f"Document {document.id} confirmed in MinIO — "
            "waiting for webhook event to trigger parse"
        )

    return DocumentUploadResponse(
        id=document.id,
        filename=document.original_filename,
        status=document.status,
        message="Document queued for processing.",
    )


@router.get("/{document_id}/chunk-context")
async def get_chunk_context(
    document_id: uuid.UUID,
    chunk_index: int | None = None,
    page_no: int | None = None,
    heading_path: str | None = None,
    context_window: int = 2,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get a focused markdown snippet around a specific chunk.

    Instead of loading the full document markdown, this returns only the
    target chunk and its immediate neighbors (context_window chunks before
    and after).  Used by the frontend for lightweight source viewing.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    await verify_workspace_access(document.workspace_id, user, db)

    if document.status not in (DocumentStatus.INDEXED, DocumentStatus.BUILDING_KG):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document is not yet indexed.",
        )

    from app.services.embedding.vector_store import get_vector_store

    vector_store = get_vector_store(document.workspace_id)

    # --- Determine the target chunk_index ---
    target_index = chunk_index

    if target_index is None and (heading_path or page_no is not None):
        # Build a metadata filter to find the target chunk
        where_filter: dict = {"document_id": str(document_id)}
        if heading_path:
            # Note: ChromaDB 'where' filter requires exact matches for metadata
            where_filter = {
                "$and": [
                    {"document_id": str(document_id)},
                    {"heading_path": heading_path},
                ]
            }
            if page_no is not None:
                where_filter["$and"].append({"page_no": page_no})
        elif page_no is not None:
            where_filter = {
                "$and": [
                    {"document_id": str(document_id)},
                    {"page_no": page_no},
                ]
            }

        try:
            found = vector_store.get_by_metadata(where=where_filter)
            if found["metadatas"]:
                # Pick the chunk with the smallest chunk_index
                indices = [
                    m.get("chunk_index", 9999) for m in found["metadatas"]
                    if m.get("chunk_index") is not None
                ]
                if indices:
                    target_index = min(indices)
        except Exception as e:
            logger.warning(f"[chunk-context] metadata lookup failed: {e}")

    if target_index is None:
        target_index = 0

    # --- Fetch target + neighbors ---
    start = max(0, target_index - context_window)
    end = min(document.chunk_count - 1 if document.chunk_count > 0 else 0, target_index + context_window)
    if start > end:
        start = end

    chunk_ids = [
        f"doc_{document_id}_chunk_{i}" for i in range(start, end + 1)
    ]

    try:
        results = vector_store.get_by_ids(chunk_ids)
    except Exception as e:
        logger.error(f"[chunk-context] get_by_ids failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {str(e)}",
        )

    chunks = []
    for i in range(len(results.get("ids", []))):
        meta = results["metadatas"][i] if results.get("metadatas") else {}
        chunks.append({
            "chunk_id": results["ids"][i],
            "chunk_index": meta.get("chunk_index", start + i),
            "content": results["documents"][i] if results.get("documents") else "",
            "page_no": meta.get("page_no"),
            "heading_path": meta.get("heading_path", ""),
            "source": meta.get("source", ""),
        })

    chunks.sort(key=lambda c: c["chunk_index"])

    md_parts: list[str] = []
    last_heading = ""
    last_page = -1

    for chunk in chunks:
        hp = chunk.get("heading_path", "")
        pg = chunk.get("page_no")
        if hp and hp != last_heading:
            md_parts.append(f"\n### 📍 {hp}\n")
            last_heading = hp
        if pg is not None and pg != last_page:
            # Use the standard format that frontend expects
            md_parts.append(f"\n<!-- page {pg} -->\n")
            last_page = pg
        md_parts.append(chunk["content"])

    combined_markdown = "\n\n".join(md_parts)

    return {
        "document_id": str(document_id),
        "target_chunk_index": target_index,
        "chunk_range": [start, end],
        "total_chunks": document.chunk_count,
        "chunks": chunks,
        "markdown": combined_markdown,
    }



@router.get("/{document_id}/markdown")
async def get_document_markdown(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get the full structured markdown content of a document (HRAG parsed)."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    await verify_workspace_access(document.workspace_id, user, db)

    if not document.markdown_s3_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No markdown content available. Document may not have been processed with HRAG.",
        )

    from app.services.storage_service import get_storage_service

    try:
        markdown = await get_storage_service().download_markdown(
            document.markdown_s3_key
        )
    except Exception as e:
        logger.error(f"Failed to fetch markdown from MinIO for doc {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Markdown storage is temporarily unavailable.",
        )

    # Safety net: if image placeholders remain, inject real references on-the-fly
    if "<!-- image" in markdown:
        img_result = await db.execute(
            select(DocumentImage)
            .where(DocumentImage.document_id == document_id)
            .order_by(DocumentImage.id)
        )
        images = img_result.scalars().all()
        if images:
            markdown = _inject_images_from_db(markdown, images, document.workspace_id)

    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown",
    )




@router.get("/{document_id}/images", response_model=list[DocumentImageResponse])
async def get_document_images(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """List all extracted images for a document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    await verify_workspace_access(document.workspace_id, user, db)

    result = await db.execute(
        select(DocumentImage)
        .where(DocumentImage.document_id == document_id)
        .order_by(DocumentImage.page_no)
    )
    images = result.scalars().all()

    return [
        DocumentImageResponse(
            image_id=img.image_id,
            document_id=img.document_id,
            page_no=img.page_no,
            caption=img.caption or "",
            width=img.width,
            height=img.height,
            url=f"/static/doc-images/kb_{document.workspace_id}/images/{img.image_id}.png",
        )
        for img in images
    ]


@router.get("/{document_id}/download")
async def download_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Download the original uploaded file from MinIO."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    await verify_workspace_access(document.workspace_id, user, db)

    if not document.upload_s3_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original file not available in storage.",
        )

    from app.services.storage_service import get_storage_service

    storage = get_storage_service()

    try:
        file_bytes = await storage.download_file(document.upload_s3_key)
    except Exception as e:
        logger.error(f"Failed to download file from MinIO for doc {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service is temporarily unavailable.",
        )

    ext = Path(document.original_filename).suffix.lower()
    content_type = _mime_for_ext(ext)

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{document.original_filename}"',
            "Content-Length": str(len(file_bytes)),
        },
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get document by ID"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    await verify_workspace_access(document.workspace_id, user, db)

    return document


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: uuid.UUID,
    body: DocumentUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Update document metadata (document_number, signer_name)."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    # Check workspace access
    workspace_result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == document.workspace_id)
    )
    workspace = workspace_result.scalar_one_or_none()
    if workspace is None:
        raise NotFoundError("KnowledgeBase", document.workspace_id)

    await verify_workspace_access(workspace.id, user, db)

    # Update fields
    if body.document_number is not None:
        document.document_number = body.document_number
    if body.document_title is not None:
        document.document_title = body.document_title
    if body.signer_name is not None:
        document.signer_name = body.signer_name
    if body.published_date is not None:
        document.published_date = body.published_date
    if body.issuing_agency is not None:
        document.issuing_agency = body.issuing_agency

    await db.commit()
    await db.refresh(document)

    # Update LegalKG (Neo4j) if document was indexed
    logger.info(
        f"update_document: doc_id={document_id}, status={document.status}, workspace_id={document.workspace_id}, kg_root_entity_id={document.kg_root_entity_id}"
    )
    if document.status == DocumentStatus.INDEXED:
        try:
            from app.services.kg.legal_kg_service import LegalKGService

            kg_service = LegalKGService(document.workspace_id)
            logger.info(
                f"Calling LegalKG update_document_metadata for doc_id={document_id}"
            )
            await kg_service.update_document_metadata(
                document_id=document.id,
                doc_number=document.document_number,
                doc_title=document.document_title,
                signer_name=document.signer_name,
                issuing_agency=document.issuing_agency,
                published_date=document.published_date,
            )
        except Exception as e:
            logger.warning(f"Failed to update LegalKG metadata: {e}")
    else:
        logger.info(
            f"Skipping LegalKG update - document status is {document.status}, not INDEXED"
        )

    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Delete a document and its chunks from vector store"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    # Only the owner, a tenant admin (tenant-visible), or a superadmin may delete.
    kb = await verify_workspace_access(document.workspace_id, user, db)
    await verify_workspace_manage_access(kb, user, db)

    # Always clean up vector store + KG, regardless of status. Chunks/KG nodes
    # can exist whenever the document was (partially) processed — e.g. embed_done
    # while still BUILDING_KG, or a FAILED doc that already embedded. Gating this
    # on status == INDEXED left orphaned chunks in ChromaDB on delete.
    try:
        from app.services.retrieval.rag_service import get_rag_service

        rag_service = get_rag_service(db, document.workspace_id)
        await rag_service.delete_document(document_id)
    except Exception as e:
        logger.warning(f"Failed to delete chunks from vector store: {e}")

    # Also delete from LegalKG (Neo4j) if KG was built
    try:
        from app.services.kg.legal_kg_service import LegalKGService

        kg_service = LegalKGService(document.workspace_id)
        await kg_service.delete_document(document_id)
    except Exception as e:
        logger.warning(f"Failed to delete document from LegalKG (Neo4j): {e}")

    # Delete local file if it still exists (legacy / backward compat)
    file_path = UPLOAD_DIR / document.filename
    if file_path.exists():
        os.remove(file_path)

    from app.services.storage_service import get_storage_service

    storage = get_storage_service()

    # Delete raw upload from MinIO
    if document.upload_s3_key:
        try:
            await storage.delete_file(document.upload_s3_key)
        except Exception as e:
            logger.warning(
                f"Failed to delete upload MinIO object for doc {document_id}: {e}"
            )

    # Delete markdown object from MinIO
    if document.markdown_s3_key:
        try:
            await storage.delete_markdown(document.markdown_s3_key)
        except Exception as e:
            logger.warning(
                f"Failed to delete markdown MinIO object for doc {document_id}: {e}"
            )

    await db.delete(document)
    await db.commit()

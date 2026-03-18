from __future__ import annotations

import os
import re
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Body
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentImage, DocumentStatus
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.schemas.rag import DocumentImageResponse

logger = logging.getLogger(__name__)


def _inject_images_from_db(
    markdown: str,
    images: list[DocumentImage],
    workspace_id: int,
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
    ".pdf":  "application/pdf",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _mime_for_ext(ext: str) -> str:
    return _EXT_TO_MIME.get(ext.lower(), "application/octet-stream")


@router.get("/workspace/{workspace_id}", response_model=list[DocumentResponse])
async def list_documents(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all documents in a knowledge base."""
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    kb = result.scalar_one_or_none()

    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    result = await db.execute(
        select(Document).where(Document.workspace_id == workspace_id).order_by(Document.created_at.desc())
    )
    return result.scalars().all()


async def process_document_background(document_id: int, file_path: str, workspace_id: int):
    """Legacy fallback: process document inline when RabbitMQ is unavailable."""
    from app.core.database import async_session_maker
    from app.services.rag_service import get_rag_service

    async with async_session_maker() as db:
        try:
            rag_service = get_rag_service(db, workspace_id)
            await rag_service.process_document(document_id, file_path)
            logger.info(f"Document {document_id} processed successfully (fallback mode)")
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")


@router.post("/upload/{workspace_id}", response_model=DocumentUploadResponse)
async def upload_document(
    workspace_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document to a knowledge base and store the raw file in MinIO."""
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    kb = result.scalar_one_or_none()

    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}"
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )

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
        status=DocumentStatus.PENDING,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Upload raw file to MinIO nexusrag-uploads bucket
    from app.services.storage_service import get_storage_service
    storage = get_storage_service()
    upload_key = storage._make_upload_key(workspace_id, document.id, ext)
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
        # Still continue — document is in DB, will need manual requeue

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
            logger.info(f"Document {document.id} queued for processing (direct publish)")
        except Exception as e:
            logger.error(
                f"Failed to publish parse task for doc {document.id}: {e}. "
                f"Document is PENDING — manual requeue may be needed."
            )
    else:
        logger.info(
            f"Document {document.id} uploaded to MinIO — "
            f"waiting for webhook event to trigger parse"
        )

    return DocumentUploadResponse(
        id=document.id,
        filename=document.original_filename,
        status=document.status,
        message="Document uploaded and queued for processing."
    )


# ---------------------------------------------------------------------------
# Presigned-upload flow (frontend uploads directly to MinIO)
# ---------------------------------------------------------------------------

class PresignRequest(BaseModel):
    filename: str
    file_size: int
    content_type: str | None = None


class PresignResponse(BaseModel):
    document_id: int
    upload_url: str          # Presigned PUT URL pointing directly to MinIO
    minio_key: str           # Object key — needed for /confirm call


class ConfirmRequest(BaseModel):
    document_id: int


@router.post("/upload/{workspace_id}/presign", response_model=PresignResponse)
async def presign_upload(
    workspace_id: int,
    body: PresignRequest,
    db: AsyncSession = Depends(get_db),
):
    """Step 1 of direct-to-MinIO upload.

    Creates the Document record in PENDING state and returns a presigned PUT
    URL.  The frontend must PUT the file bytes directly to that URL, then call
    ``/confirm`` to trigger the parse pipeline.
    """
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    kb = result.scalar_one_or_none()
    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

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

    document.upload_s3_key = upload_key
    await db.commit()

    return PresignResponse(
        document_id=document.id,
        upload_url=presigned_url,
        minio_key=upload_key,
    )


@router.post("/upload/{workspace_id}/confirm", response_model=DocumentUploadResponse)
async def confirm_upload(
    workspace_id: int,
    body: ConfirmRequest,
    db: AsyncSession = Depends(get_db),
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
    if document.upload_s3_key:
        try:
            exists = await storage.object_exists(document.upload_s3_key)
        except Exception as e:
            logger.error(f"MinIO object_exists check failed for doc {document.id}: {e}")
            exists = True  # optimistic — proceed anyway
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File not found in storage. Please retry the upload.",
            )

    if not settings.MINIO_WEBHOOK_ENABLED:
        try:
            from app.queue.publisher import publish_parse_task
            await publish_parse_task(
                document_id=document.id,
                workspace_id=workspace_id,
                minio_key=document.upload_s3_key or "",
                original_filename=document.original_filename,
            )
            logger.info(f"Document {document.id} queued for processing (presign confirm)")
        except Exception as e:
            logger.error(
                f"Failed to publish parse task for doc {document.id}: {e}. "
                "Document stays PENDING — manual requeue may be needed."
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


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get document by ID"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    return document


@router.get("/{document_id}/markdown")
async def get_document_markdown(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get the full structured markdown content of a document (NexusRAG parsed)."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if not document.markdown_s3_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No markdown content available. Document may not have been processed with NexusRAG."
        )

    from app.services.storage_service import get_storage_service
    try:
        markdown = await get_storage_service().download_markdown(document.markdown_s3_key)
    except Exception as e:
        logger.error(f"Failed to fetch markdown from MinIO for doc {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Markdown storage is temporarily unavailable."
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
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all extracted images for a document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

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


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and its chunks from vector store"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status == DocumentStatus.INDEXED:
        try:
            from app.services.rag_service import get_rag_service
            rag_service = get_rag_service(db, document.workspace_id)
            await rag_service.delete_document(document_id)
        except Exception as e:
            logger.warning(f"Failed to delete chunks from vector store: {e}")

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
            logger.warning(f"Failed to delete upload MinIO object for doc {document_id}: {e}")

    # Delete markdown object from MinIO
    if document.markdown_s3_key:
        try:
            await storage.delete_markdown(document.markdown_s3_key)
        except Exception as e:
            logger.warning(f"Failed to delete markdown MinIO object for doc {document_id}: {e}")

    await db.delete(document)
    await db.commit()

"""
OCR API — convert an uploaded PDF into text on demand.

``POST /ocr/extract`` → markdown/text for an uploaded PDF (runs HunyuanOCR).
``GET  /ocr/status``  → {enabled} so the UI can show/hide the tool.

Auth via ``get_principal`` (JWT or API key), mirroring the STT / TTS endpoints —
any authenticated user (including the plain ``user`` role) may use it. The upload
is OCR'd in-process and discarded; it is NOT stored in MinIO or the documents
pipeline.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import settings
from app.core.deps import get_principal
from app.core.exceptions import BadRequestError
from app.models.user import User
from app.schemas.ocr import OCRExtractResponse, OCRStatusResponse
from app.services.ocr_service import get_ocr_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr"])

# Cap on-demand OCR uploads (the parse pipeline has its own limits separately).
_MAX_UPLOAD_MB = 50


@router.get("/status", response_model=OCRStatusResponse)
async def ocr_status(user: User = Depends(get_principal)):
    """Report whether OCR is enabled so the UI can show/hide the tool."""
    return OCRStatusResponse(enabled=settings.HRAG_ENABLE_OCR)


@router.post("/extract", response_model=OCRExtractResponse)
async def extract(
    file: UploadFile = File(...),
    user: User = Depends(get_principal),
):
    """Run OCR on an uploaded PDF and return the extracted text."""
    if not settings.HRAG_ENABLE_OCR:
        raise BadRequestError("OCR is disabled")

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    filename = file.filename or "document.pdf"
    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    if not is_pdf:
        raise BadRequestError(f"Only PDF files are supported (got {content_type!r})")

    data = await file.read()
    if not data:
        raise BadRequestError("Empty PDF upload")

    max_bytes = _MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise BadRequestError(
            f"PDF too large ({len(data) // (1024 * 1024)}MB > {_MAX_UPLOAD_MB}MB)"
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        text = await get_ocr_service().ocr_pdf(tmp_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("OCR extract failed: %s", exc)
        raise BadRequestError("OCR failed") from exc
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    # ocr_pdf() emits a `<!-- page N -->` marker per non-empty page.
    pages = text.count("<!-- page ")
    return OCRExtractResponse(text=text, pages=pages)

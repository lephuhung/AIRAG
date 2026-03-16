"""
HunyuanOCR Service
==================

Handles OCR for scanned / image-based PDFs using Tencent HunyuanOCR
deployed via vLLM (OpenAI-compatible API).

Flow:
  1. Detect whether a PDF is scanned (insufficient selectable text).
  2. Render each page to a PNG image via PyMuPDF (fitz).
  3. Send each page image to HunyuanOCR as a base64-encoded image URL.
  4. Collect per-page OCR text and assemble into a full document text.

API compatibility:
  HunyuanOCR served by vLLM exposes a standard OpenAI chat-completions
  endpoint.  Images are passed as ``image_url`` content parts with a
  ``data:image/png;base64,...`` data URI.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# HunyuanOCR recommended document-parsing prompt (from official documentation).
# - Extracts all body text as Markdown
# - Ignores page headers and footers
# - Renders tables as HTML (handles merged cells / complex layouts better than Markdown)
# - Renders mathematical formulas as LaTeX (compatible with NEXUSRAG_ENABLE_FORMULA_ENRICHMENT)
# - Follows natural reading order
_OCR_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，"
    "其中页眉、页脚部分忽略，"
    "表格用html格式表达，"
    "文档中公式用latex格式表示，"
    "按照阅读顺序组织进行解析。"
)

# Minimum number of characters per page to consider it as having selectable text
_MIN_CHARS_PER_PAGE = 30


class HunyuanOCRService:
    """
    OCR service backed by HunyuanOCR via vLLM OpenAI-compatible API.

    All page rendering and HTTP calls are run in threads / async so they
    do not block the FastAPI event loop.
    """

    def __init__(self) -> None:
        self._api_url = settings.HUNYUAN_OCR_API_URL.rstrip("/")
        self._model = settings.HUNYUAN_OCR_MODEL
        self._threshold = settings.NEXUSRAG_OCR_SCANNED_THRESHOLD
        # Reuse a single async client across calls (created lazily)
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def is_scanned_pdf(file_path: str | Path) -> bool:
        """
        Return True if the PDF is predominantly scanned (image-based).

        Strategy: count pages whose selectable-text character count falls
        below ``_MIN_CHARS_PER_PAGE``.  If the fraction of such pages
        exceeds ``NEXUSRAG_OCR_SCANNED_THRESHOLD`` the file is classified
        as scanned.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            if doc.page_count == 0:
                return False

            text_poor_pages = 0
            for page in doc:
                text = page.get_text("text")
                if len(text.strip()) < _MIN_CHARS_PER_PAGE:
                    text_poor_pages += 1

            ratio = text_poor_pages / doc.page_count
            doc.close()

            logger.info(
                f"PDF scan-check: {file_path} — {text_poor_pages}/{doc.page_count} "
                f"pages below text threshold (ratio={ratio:.2f}, "
                f"threshold={settings.NEXUSRAG_OCR_SCANNED_THRESHOLD})"
            )
            return ratio >= settings.NEXUSRAG_OCR_SCANNED_THRESHOLD

        except ImportError:
            logger.warning(
                "PyMuPDF (fitz) not installed — cannot detect scanned PDFs. "
                "Install with: pip install pymupdf"
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to check if PDF is scanned ({file_path}): {e}")
            return False

    async def ocr_pdf(self, file_path: str | Path) -> str:
        """
        Run OCR on a scanned PDF and return the full extracted text.

        Pages are processed concurrently (up to 4 at a time) to balance
        throughput vs API load.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError(
                "PyMuPDF is required for OCR. Install with: pip install pymupdf"
            )

        doc = fitz.open(str(file_path))
        total_pages = doc.page_count
        logger.info(f"Starting OCR for {file_path} ({total_pages} pages)")

        # Render all pages to PNG bytes (in thread pool — CPU-bound)
        page_images: list[bytes] = await asyncio.to_thread(
            self._render_pages, doc
        )
        doc.close()

        # OCR pages concurrently with a semaphore to limit parallelism
        semaphore = asyncio.Semaphore(4)

        async def ocr_page(page_idx: int, img_bytes: bytes) -> tuple[int, str]:
            async with semaphore:
                text = await self._ocr_image(img_bytes, page_idx + 1)
                return page_idx, text

        tasks = [
            ocr_page(i, img_bytes) for i, img_bytes in enumerate(page_images)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Assemble in page order
        page_texts: list[str] = [""] * total_pages
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"OCR page task failed: {result}")
                continue
            page_idx, text = result
            page_texts[page_idx] = text

        full_text = "\n\n---\n\n".join(t for t in page_texts if t.strip())
        logger.info(
            f"OCR complete for {file_path}: "
            f"{len(full_text)} chars extracted from {total_pages} pages"
        )
        return full_text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_pages(doc) -> list[bytes]:
        """Render all PDF pages to PNG bytes (runs in a thread pool)."""
        import fitz

        page_images: list[bytes] = []
        # 150 DPI is sufficient for OCR; higher = slower network + API calls
        matrix = fitz.Matrix(150 / 72, 150 / 72)

        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pixmap.tobytes("png")
            page_images.append(png_bytes)

        return page_images

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and cache an async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def _ocr_image(self, img_bytes: bytes, page_num: int) -> str:
        """
        Send a single page image to HunyuanOCR and return the extracted text.

        Uses the OpenAI-compatible chat completions endpoint that vLLM exposes.
        Images are encoded as base64 data URIs inside an ``image_url`` content part.
        """
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        },
                        {
                            "type": "text",
                            "text": _OCR_PROMPT,
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 8192,
        }

        client = await self._get_client()
        try:
            response = await client.post(
                f"{self._api_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            logger.debug(f"OCR page {page_num}: {len(text)} chars extracted")
            return text.strip()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HunyuanOCR API error on page {page_num}: "
                f"HTTP {e.response.status_code} — {e.response.text[:200]}"
            )
            return ""
        except Exception as e:
            logger.error(f"HunyuanOCR call failed on page {page_num}: {e}")
            return ""

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_ocr_service: HunyuanOCRService | None = None


def get_ocr_service() -> HunyuanOCRService:
    """Return the module-level HunyuanOCRService singleton."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = HunyuanOCRService()
    return _ocr_service

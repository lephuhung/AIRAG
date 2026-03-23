"""
HunyuanOCR Service
==================

Handles OCR for scanned / image-based PDFs using Tencent HunyuanOCR.

Two backends (selected via HRAG_OCR_LOCAL in .env):

  HRAG_OCR_LOCAL=false  (default)
    → Remote vLLM server at HUNYUAN_OCR_API_URL (OpenAI-compatible API).
    → One async httpx call per page; up to 4 pages in parallel.
    → No local GPU required.

  HRAG_OCR_LOCAL=true
    → vllm.LLM loaded in-process on HRAG_OCR_LOCAL_DEVICE.
    → Requires: pip install vllm  (~10 GB VRAM for HunyuanOCR).
    → Model is loaded lazily on first OCR call and kept warm.
    → Pages are processed sequentially (vLLM handles its own batching).

Both backends:
  - Render PDF pages to PNG via PyMuPDF at 150 DPI.
  - Use the same markdown-extraction OCR prompt.
  - Apply clean_repeated_substrings() to remove vLLM hallucination loops.
  - Assemble per-page texts with "---" page separators.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR prompt — identical for both local and API backends
# ---------------------------------------------------------------------------
# Extracts all body text as Markdown, ignores headers/footers,
# renders tables as HTML and formulas as LaTeX, follows reading order.
_OCR_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，"
    "其中页眉、页脚部分忽略，"
    "表格用html格式表达，"
    "文档中公式用latex格式表示，"
    "按照阅读顺序组织进行解析。"
)

# Minimum selectable characters per page to consider it non-scanned
_MIN_CHARS_PER_PAGE = 30


# ---------------------------------------------------------------------------
# Utility: remove hallucination loops from vLLM output
# ---------------------------------------------------------------------------

def _clean_repeated_substrings(text: str) -> str:
    """
    Remove trailing repeated substrings caused by vLLM generation loops.

    When a model enters a repetition loop it appends the same N-char fragment
    dozens of times.  This function detects such patterns (≥10 consecutive
    repetitions of any fragment 2-len(text)//10 chars long) and strips the
    excess copies, keeping exactly one.

    Only applied to texts longer than 8 000 chars to avoid false positives
    on short but legitimately repetitive content (e.g. table rows).
    """
    n = len(text)
    if n < 8000:
        return text

    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            # Keep the first occurrence, remove the rest
            return text[: n - length * (count - 1)]

    return text


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class HunyuanOCRService:
    """
    OCR service for scanned PDFs.

    Backends:
      - API  (HRAG_OCR_LOCAL=false): async HTTP calls to a remote vLLM server.
      - Local (HRAG_OCR_LOCAL=true): in-process vllm.LLM inference.
    """

    def __init__(self) -> None:
        self._api_url   = settings.HUNYUAN_OCR_API_URL.rstrip("/")
        self._model     = settings.HUNYUAN_OCR_MODEL
        self._threshold = settings.HRAG_OCR_SCANNED_THRESHOLD
        self._local     = settings.HRAG_OCR_LOCAL

        # API backend: reuse a single async HTTP client
        self._client: httpx.AsyncClient | None = None

        # Local backend: lazy-loaded vllm.LLM + processor
        self._llm       = None
        self._processor = None
        self._sampling_params = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def is_scanned_pdf(file_path: str | Path) -> bool:
        """
        Return True if the PDF is predominantly scanned (image-based).

        Counts pages with fewer than _MIN_CHARS_PER_PAGE selectable characters.
        If that fraction meets or exceeds HRAG_OCR_SCANNED_THRESHOLD the
        file is classified as scanned.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            if doc.page_count == 0:
                return False

            total_pages = doc.page_count  # save before close
            text_poor_pages = 0
            for page in doc:
                text = page.get_text("text")
                if len(text.strip()) < _MIN_CHARS_PER_PAGE:
                    text_poor_pages += 1

            ratio = text_poor_pages / total_pages
            doc.close()

            logger.info(
                f"PDF scan-check: {file_path} — {text_poor_pages}/{total_pages} "
                f"pages below text threshold (ratio={ratio:.2f}, "
                f"threshold={settings.HRAG_OCR_SCANNED_THRESHOLD})"
            )
            return ratio >= settings.HRAG_OCR_SCANNED_THRESHOLD

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

        Dispatches to the local vLLM backend or the remote API backend
        depending on HRAG_OCR_LOCAL.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError(
                "PyMuPDF is required for OCR. Install with: pip install pymupdf"
            )

        doc = fitz.open(str(file_path))
        total_pages = doc.page_count
        backend = "local" if self._local else "API"
        logger.info(
            f"[OCR/{backend}] Starting OCR for {file_path} ({total_pages} pages)"
        )

        # Render all pages to PNG bytes (CPU-bound → thread pool)
        page_images: list[bytes] = await asyncio.to_thread(
            self._render_pages, doc
        )
        doc.close()

        if self._local:
            page_texts = await self._ocr_pages_local(page_images)
        else:
            page_texts = await self._ocr_pages_api(page_images)

        full_text = "\n\n---\n\n".join(t for t in page_texts if t.strip())
        logger.info(
            f"[OCR/{backend}] Complete: {len(full_text)} chars from {total_pages} pages"
        )
        return full_text

    async def aclose(self) -> None:
        """Release HTTP client (API mode only)."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Page rendering (shared)
    # ------------------------------------------------------------------

    @staticmethod
    def _render_pages(doc) -> list[bytes]:
        """Render all PDF pages to PNG bytes at 150 DPI (thread-pool safe)."""
        import fitz

        page_images: list[bytes] = []
        matrix = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI

        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            page_images.append(pixmap.tobytes("png"))

        return page_images

    # ------------------------------------------------------------------
    # API backend
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def _ocr_pages_api(self, page_images: list[bytes]) -> list[str]:
        """OCR all pages via remote vLLM API, up to 4 pages concurrently."""
        semaphore = asyncio.Semaphore(4)

        async def ocr_one(idx: int, img_bytes: bytes) -> tuple[int, str]:
            async with semaphore:
                text = await self._ocr_image_api(img_bytes, idx + 1)
                return idx, text

        results = await asyncio.gather(
            *[ocr_one(i, b) for i, b in enumerate(page_images)],
            return_exceptions=True,
        )

        page_texts = [""] * len(page_images)
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"[OCR/API] Page task failed: {r}")
                continue
            idx, text = r
            page_texts[idx] = _clean_repeated_substrings(text)
        return page_texts

    async def _ocr_image_api(self, img_bytes: bytes, page_num: int) -> str:
        """Send one page image to the remote vLLM API and return OCR text."""
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": _OCR_PROMPT},
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
            text = response.json()["choices"][0]["message"]["content"]
            logger.debug(f"[OCR/API] Page {page_num}: {len(text)} chars")
            return text.strip()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"[OCR/API] HTTP error on page {page_num}: "
                f"{e.response.status_code} — {e.response.text[:200]}"
            )
            return ""
        except Exception as e:
            logger.error(f"[OCR/API] Failed on page {page_num}: {e}")
            return ""

    # ------------------------------------------------------------------
    # Local vLLM backend
    # ------------------------------------------------------------------

    def _get_local_llm(self):
        """
        Lazy-load vllm.LLM + AutoProcessor.

        Called inside a thread (via asyncio.to_thread) since vllm.LLM.__init__
        blocks while loading model weights onto the GPU.

        vLLM ≥0.6 removed the `device` constructor argument — GPU placement is
        controlled via CUDA_VISIBLE_DEVICES env var and `gpu_memory_utilization`
        instead.  `gpu_memory_utilization` defaults to 0.9 (≈43 GB on a 47 GB
        card), which is far too large for a 1B OCR model; we cap it at
        HRAG_OCR_GPU_MEMORY_UTILIZATION (default 0.15 ≈ 7 GB).

        CUDA_VISIBLE_DEVICES is set here so that vLLM (which reads it at import
        time in its sub-process) directs the model to the right GPU even when the
        parent process has already initialised CUDA on a different GPU.
        """
        if self._llm is not None:
            return self._llm, self._processor, self._sampling_params

        try:
            import os
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                f"Local OCR requires 'vllm' and 'transformers': pip install vllm. "
                f"Missing: {e}"
            )

        import os
        cuda_device = settings.HRAG_OCR_CUDA_DEVICE
        if cuda_device not in ("", "auto"):
            # Must be set before vLLM spawns its EngineCore subprocess
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
            logger.info( f"[OCR/local] CUDA_VISIBLE_DEVICES={cuda_device}")

        gpu_mem = settings.HRAG_OCR_GPU_MEMORY_UTILIZATION
        max_model_len = settings.HRAG_OCR_MAX_MODEL_LEN

        logger.info(
            f"[OCR/local] Loading {self._model} "
            f"(gpu_memory_utilization={gpu_mem}, max_model_len={max_model_len})"
        )

        llm_kwargs: dict = dict(
            model=self._model,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_mem,
            enforce_eager=True,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self._llm = LLM(**llm_kwargs)
        self._processor = AutoProcessor.from_pretrained(
            self._model, trust_remote_code=True
        )
        self._sampling_params = SamplingParams(temperature=0.0, max_tokens=16384)
        logger.info(f"[OCR/local] Model {self._model} loaded successfully")
        return self._llm, self._processor, self._sampling_params

    def _ocr_pages_local_sync(self, page_images: list[bytes]) -> list[str]:
        """
        Run OCR on all pages using the local vLLM engine (synchronous).

        Called via asyncio.to_thread to avoid blocking the event loop.
        Pages are submitted as a single batch for maximum GPU utilisation.
        """
        from PIL import Image
        import io

        llm, processor, sampling_params = self._get_local_llm()

        # Build vLLM inputs for every page in one batch
        inputs = []
        for img_bytes in page_images:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            messages = [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        # "image" type: vLLM/HunyuanOCR native multi-modal input
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": _OCR_PROMPT},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": [pil_img]},
            })

        logger.info(
            f"[OCR/local] Submitting {len(inputs)} pages to vLLM (batch inference)"
        )
        outputs = llm.generate(inputs, sampling_params)

        page_texts: list[str] = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text.strip()
            text = _clean_repeated_substrings(text)
            logger.debug(f"[OCR/local] Page {i + 1}: {len(text)} chars")
            page_texts.append(text)

        return page_texts

    async def _ocr_pages_local(self, page_images: list[bytes]) -> list[str]:
        """Async wrapper: run local vLLM OCR in a thread pool."""
        return await asyncio.to_thread(self._ocr_pages_local_sync, page_images)


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

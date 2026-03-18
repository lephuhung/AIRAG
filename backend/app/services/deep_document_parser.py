"""
Deep Document Parser
====================

Parses documents using Docling for high-fidelity extraction of text, tables,
and images with structural metadata (page numbers, headings).

Supported formats: PDF, DOCX, PPTX, HTML (via Docling)
Fallback: TXT, MD (via legacy loader)
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.services.models.parsed_document import (
    ExtractedImage,
    ExtractedTable,
    EnrichedChunk,
    ParsedDocument,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vietnamese whitespace normalization
# ---------------------------------------------------------------------------
# Docling's PDF backend (docling-parse) reconstructs text from per-glyph
# coordinates in the PDF content stream.  Many Vietnamese PDFs — especially
# government legislation — encode each glyph with its own positioning
# operator, causing Docling to insert a space between every character:
#
#     "L u ậ t"   instead of   "Luật"
#
# This is a well-known issue with CID-keyed / ToUnicode-mapped Vietnamese
# fonts in the wild.
#
# The algorithm below works line-by-line:
#   1. NFC-normalize Unicode so decomposed diacritics become single codepoints.
#   2. Detect "single-char + single-space" runs — consecutive Unicode letters
#      (including Vietnamese diacritics) separated by exactly ONE space.
#      These are almost certainly spurious per-glyph spaces.
#   3. Rejoin those runs into solid words.
#   4. Collapse remaining multiple spaces to one space.
#
# Lines that look like markdown structure (headings, tables, images, code
# fences, horizontal rules) are left untouched.

# Regex: a run of >=2 single Unicode letters (including accented), each
# separated by exactly ONE space.  We anchor on word boundaries so we don't
# accidentally eat real single-letter words like "I" in English prose.
# The negative lookahead/lookbehind for \S ensures we only match isolated
# char-space-char sequences.
_SCATTERED_CHARS_RE = re.compile(
    r'(?<!\S)'                  # not preceded by a non-space char
    r'(\w)'                     # first char
    r'((?:\s\w){2,})'           # 2+ more " <char>" pairs  (min 3 chars total)
    r'(?!\S)',                   # not followed by a non-space char
)

# Lines we should never touch
_STRUCTURE_LINE_RE = re.compile(
    r'^\s*(?:'
    r'[#]{1,6}\s'              # markdown headings
    r'|[|]'                     # table rows
    r'|[-*_]{3,}'              # horizontal rules
    r'|```'                     # code fences
    r'|!\[.*\]\(.*\)'          # images
    r'|\[.*\]\(.*\)'           # links on their own line
    r'|<!--.*-->'              # HTML comments
    r'|>\s'                     # blockquotes
    r')'
)


def _fix_scattered_vietnamese(text: str) -> str:
    """
    Fix per-glyph spacing artefacts in Vietnamese text produced by Docling.

    Processes each line independently so markdown structure is preserved.
    Only activates when the text looks Vietnamese (contains Vietnamese-specific
    diacritics) to avoid mangling non-Vietnamese content.
    """
    # Quick check: does this text contain Vietnamese-specific characters?
    # Covers:  ă â đ ê ô ơ ư  +  all pre-composed Vietnamese in U+1E00–1EFF
    #          + Latin Extended Additional (ò, ã, ũ etc. in U+00C0-024F)
    _VN_MARKER = re.compile(r'[\u00C0-\u024F\u1E00-\u1EFF]')
    if not _VN_MARKER.search(text):
        return text

    # NFC normalize — combine decomposed diacritics
    text = unicodedata.normalize("NFC", text)

    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        # Skip structural markdown lines
        if _STRUCTURE_LINE_RE.match(line):
            result.append(line)
            continue

        # Skip empty / whitespace-only lines
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue

        # Heuristic: does this line look scattered?
        # Count tokens that are single characters.  If >=60% of tokens are
        # single-char, the line is likely affected by per-glyph spacing.
        tokens = stripped.split()
        if len(tokens) < 3:
            result.append(line)
            continue

        single_count = sum(1 for t in tokens if len(t) == 1)
        ratio = single_count / len(tokens)
        if ratio < 0.5:
            # Not scattered — leave as-is
            result.append(line)
            continue

        # Core fix: rejoin scattered single characters.
        # Split on TWO-or-more consecutive spaces (real word/segment boundaries),
        # then within each segment rejoin consecutive single-char tokens.
        segments = re.split(r'  +', stripped)
        fixed_segments: list[str] = []

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            words = seg.split()
            if not words:
                continue

            # Rejoin consecutive single-char tokens, keep multi-char tokens separate
            merged: list[str] = []
            buf: list[str] = []
            for w in words:
                if len(w) == 1:
                    buf.append(w)
                else:
                    if buf:
                        merged.append("".join(buf))
                        buf = []
                    merged.append(w)
            if buf:
                merged.append("".join(buf))

            fixed_segments.append(" ".join(merged))

        # Preserve leading whitespace (indentation)
        leading = len(line) - len(line.lstrip())
        fixed_line = line[:leading] + " ".join(fixed_segments)
        result.append(fixed_line)

    return "\n".join(result)

# File extensions handled by Docling vs legacy
_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html"}
_LEGACY_EXTENSIONS = {".txt", ".md"}
_OCR_EXTENSIONS = {".pdf"}  # Only PDFs need scanned-page detection

# ---------------------------------------------------------------------------
# Module-level Docling converter singleton
# ---------------------------------------------------------------------------
# Docling loads LayoutModel (~1 GB) + TableFormer (~600 MB) + RapidOCR on first
# use and keeps them in CUDA memory.  By caching the converter at module level
# we pay that cost only ONCE per worker process regardless of how many documents
# are processed.  Each parse worker is a separate OS process, so there is no
# cross-process sharing — that is intentional: CUDA contexts are per-process.
_DOCLING_CONVERTER = None


def _get_global_converter():
    """Return the process-wide singleton DocumentConverter, creating it once."""
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is not None:
        return _DOCLING_CONVERTER

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrAutoOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

    device = settings.NEXUSRAG_DOCLING_DEVICE  # "auto" | "cpu" | "cuda"

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = settings.NEXUSRAG_ENABLE_IMAGE_EXTRACTION
    pipeline_options.images_scale = settings.NEXUSRAG_DOCLING_IMAGES_SCALE
    pipeline_options.do_formula_enrichment = settings.NEXUSRAG_ENABLE_FORMULA_ENRICHMENT
    pipeline_options.ocr_options = OcrAutoOptions(lang=["vi", "en"])
    pipeline_options.accelerator_options = AcceleratorOptions(device=device)

    logger.info(f"[Docling] Initializing DocumentConverter (device={device})")
    _DOCLING_CONVERTER = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    logger.info("[Docling] DocumentConverter ready (singleton)")
    return _DOCLING_CONVERTER


class DeepDocumentParser:
    """
    Parses documents using Docling for rich structural extraction.

    - Converts PDF/DOCX/PPTX/HTML via Docling DocumentConverter
    - Chunks using HybridChunker (semantic + structural)
    - Extracts images and optionally captions them via Gemini Vision
    - Falls back to legacy text extraction for TXT/MD
    """

    def __init__(self, workspace_id: int, output_dir: Optional[Path] = None):
        self.workspace_id = workspace_id
        self.output_dir = output_dir or (
            settings.BASE_DIR / "data" / "docling" / f"kb_{workspace_id}"
        )

    def _get_converter(self):
        """Return the process-wide singleton DocumentConverter."""
        return _get_global_converter()

    @staticmethod
    def is_docling_supported(file_path: str | Path) -> bool:
        """Check if the file format is supported by Docling."""
        return Path(file_path).suffix.lower() in _DOCLING_EXTENSIONS

    async def parse_structure(
        self,
        file_path: str | Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """
        Parse document structure — ZERO LLM calls.

        Returns markdown, raw chunks (no caption enrichment), image files,
        and table markdown.  Caption enrichment and KG ingest are handled
        by separate workers after this method returns.

        For scanned PDFs (when OCR is enabled), pages are first run through
        HunyuanOCR (HTTP call, not LLM via provider) before being chunked.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        start_time = time.time()

        if suffix in _DOCLING_EXTENSIONS:
            # For PDFs: detect scanned pages and run OCR first if needed
            if (
                suffix in _OCR_EXTENSIONS
                and settings.NEXUSRAG_ENABLE_OCR
                and self._is_scanned(path)
            ):
                result = await self._parse_with_ocr(path, document_id, original_filename)
            else:
                result = self._parse_with_docling(path, document_id, original_filename)
        elif suffix in _LEGACY_EXTENSIONS:
            result = self._parse_legacy(path, document_id, original_filename)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {_DOCLING_EXTENSIONS | _LEGACY_EXTENSIONS}"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Parsed document {document_id} ({original_filename}) in {elapsed_ms}ms: "
            f"{result.page_count} pages, {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {result.tables_count} tables"
        )
        return result

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_scanned(file_path: Path) -> bool:
        """Delegate scanned-PDF detection to HunyuanOCRService."""
        from app.services.ocr_service import get_ocr_service
        return get_ocr_service().is_scanned_pdf(file_path)

    async def _parse_with_ocr(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """
        Parse a scanned PDF via HunyuanOCR then convert to ParsedDocument.

        Pipeline:
          1. Run OCR on every page → get full plain-text / markdown.
          2. Write the OCR'd text to a temporary .md file.
          3. Parse that .md file via the legacy text pipeline so chunking
             and metadata work the same way as for normal documents.
        """
        import tempfile

        from app.services.ocr_service import get_ocr_service

        logger.info(
            f"Document {document_id} ({original_filename}) identified as scanned PDF "
            f"— running HunyuanOCR"
        )

        ocr_service = get_ocr_service()
        ocr_text = await ocr_service.ocr_pdf(file_path)

        if not ocr_text.strip():
            logger.warning(
                f"HunyuanOCR returned empty text for {original_filename}. "
                f"Returning empty document (no Docling fallback for scanned PDFs)."
            )
            # Return empty ParsedDocument — do not fall back to Docling for scanned PDFs
            return ParsedDocument(
                document_id=document_id,
                original_filename=original_filename,
                markdown="",
                chunks=[],
                images=[],
                tables=[],
                tables_count=0,
                page_count=0,
            )

        # Write OCR text to a temp file and parse it as markdown
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(ocr_text)
            tmp_path = Path(tmp.name)

        try:
            result = self._parse_legacy(tmp_path, document_id, original_filename)
            # Override markdown with OCR output so the viewer shows the right content
            result.markdown = ocr_text
        finally:
            tmp_path.unlink(missing_ok=True)

        logger.info(
            f"OCR parse complete for {document_id}: "
            f"{len(result.chunks)} chunks, {result.page_count} pages"
        )
        return result

    def _parse_with_docling(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Parse with Docling for rich structural extraction."""
        converter = self._get_converter()

        # Convert document (CPU-bound, no LLM)
        logger.info(f"Docling converting: {file_path}")
        conv_result = converter.convert(str(file_path))
        doc = conv_result.document

        # Extract images (file save only — captions added later by caption worker)
        images, pic_url_list = self._extract_images_with_urls(doc, document_id)

        # Extract tables (markdown only — captions added later by caption worker)
        tables = self._extract_tables(doc, document_id)

        # Export to markdown (with page break markers if supported)
        markdown = self._export_markdown(doc)

        # Post-process: replace image placeholders with real markdown images
        markdown = self._inject_image_references(markdown, pic_url_list)
        # Note: table captions NOT injected here — caption worker will update
        # markdown_content in DB once captions are ready

        # Get page count
        page_count = 0
        if hasattr(doc, "pages") and doc.pages:
            page_count = len(doc.pages)

        # Chunk with HybridChunker — images/tables passed for page-ref tracking
        # but enriched_text will NOT include captions yet (images have caption="")
        chunks = self._chunk_document(doc, document_id, original_filename, images, tables)

        tables_count = len(tables)

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=markdown,
            page_count=page_count,
            chunks=chunks,
            images=images,
            tables=tables,
            tables_count=tables_count,
        )

    def _chunk_document(
        self,
        doc,
        document_id: int,
        original_filename: str,
        images: list[ExtractedImage] | None = None,
        tables: list[ExtractedTable] | None = None,
    ) -> list[EnrichedChunk]:
        """Chunk document using Docling's HybridChunker.

        When *images* / *tables* are provided, each chunk is enriched with
        references to images/tables that appear on the same page.  The
        descriptions are appended to the chunk text so they become part of
        the embedding, making image/table content semantically searchable.
        """
        from docling_core.transforms.chunker import HybridChunker

        chunker = HybridChunker(
            max_tokens=settings.NEXUSRAG_CHUNK_MAX_TOKENS,
            merge_peers=True,
        )

        # Build page→images lookup for O(1) matching
        page_images: dict[int, list[ExtractedImage]] = {}
        if images:
            for img in images:
                page_images.setdefault(img.page_no, []).append(img)

        # Build page→tables lookup for O(1) matching
        page_tables: dict[int, list[ExtractedTable]] = {}
        if tables:
            for tbl in tables:
                page_tables.setdefault(tbl.page_no, []).append(tbl)

        chunks = []
        # Track which images/tables have already been assigned to a chunk
        # on the same page so we avoid duplicating descriptions across many
        # chunks.  Each image/table is assigned to the FIRST chunk on its page.
        assigned_images: set[str] = set()
        assigned_tables: set[str] = set()

        for i, chunk in enumerate(chunker.chunk(doc)):
            # Extract page number from chunk metadata
            page_no = 0
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "page"):
                    page_no = chunk.meta.page or 0
                elif hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        if hasattr(item, "prov") and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, "page_no"):
                                    page_no = prov.page_no or 0
                                    break
                            if page_no > 0:
                                break

            # Extract heading path from chunk metadata
            heading_path = []
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    heading_path = list(chunk.meta.headings)

            # Detect content types
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            # Fix per-glyph spacing in Vietnamese text from Docling chunker
            chunk_text = _fix_scattered_vietnamese(chunk_text)
            has_table = False
            has_code = False
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        label = getattr(item, "label", "") or ""
                        if "table" in label.lower():
                            has_table = True
                        if "code" in label.lower():
                            has_code = True

            contextualized = ""
            if heading_path:
                contextualized = " > ".join(heading_path) + ": " + chunk_text[:100]

            # ── Image-aware enrichment ──────────────────────────────
            chunk_image_refs: list[str] = []  # image_ids
            if page_no > 0 and page_no in page_images:
                for img in page_images[page_no]:
                    if img.image_id not in assigned_images:
                        chunk_image_refs.append(img.image_id)
                        assigned_images.add(img.image_id)

            # Append image descriptions to chunk text so they become
            # part of the embedding vector (semantically searchable).
            enriched_text = chunk_text
            if chunk_image_refs and images:
                img_by_id = {im.image_id: im for im in images}
                desc_parts = []
                for img_id in chunk_image_refs:
                    img = img_by_id.get(img_id)
                    if img and img.caption:
                        desc_parts.append(
                            f"[Image on page {img.page_no}]: {img.caption}"
                        )
                if desc_parts:
                    enriched_text = (
                        chunk_text + "\n\n" + "\n".join(desc_parts)
                    )

            # ── Table-aware enrichment ────────────────────────────────
            chunk_table_refs: list[str] = []
            if page_no > 0 and page_no in page_tables:
                for tbl in page_tables[page_no]:
                    if tbl.table_id not in assigned_tables:
                        chunk_table_refs.append(tbl.table_id)
                        assigned_tables.add(tbl.table_id)

            if chunk_table_refs and tables:
                tbl_by_id = {t.table_id: t for t in tables}
                tbl_parts = []
                for tbl_id in chunk_table_refs:
                    tbl = tbl_by_id.get(tbl_id)
                    if tbl and tbl.caption:
                        tbl_parts.append(
                            f"[Table on page {tbl.page_no} ({tbl.num_rows}x{tbl.num_cols})]: {tbl.caption}"
                        )
                if tbl_parts:
                    enriched_text = enriched_text + "\n\n" + "\n".join(tbl_parts)

            chunks.append(EnrichedChunk(
                content=enriched_text,
                chunk_index=i,
                source_file=original_filename,
                document_id=document_id,
                page_no=page_no,
                heading_path=heading_path,
                image_refs=chunk_image_refs,
                table_refs=chunk_table_refs,
                has_table=has_table,
                has_code=has_code,
                contextualized=contextualized,
            ))

        if images:
            assigned_count = len(assigned_images)
            logger.info(
                f"Image-aware chunking: {assigned_count}/{len(images)} images "
                f"assigned to {len(chunks)} chunks"
            )

        if tables:
            assigned_tbl_count = len(assigned_tables)
            logger.info(
                f"Table-aware chunking: {assigned_tbl_count}/{len(tables)} tables "
                f"assigned to {len(chunks)} chunks"
            )

        return chunks

    def _export_markdown(self, doc) -> str:
        """Export document to markdown with page break markers if supported.

        Applies Vietnamese whitespace normalization to fix per-glyph spacing
        artefacts common in Vietnamese PDFs processed by Docling.
        """
        try:
            md = doc.export_to_markdown(
                page_break_placeholder="\n\n---\n\n",
            )
        except TypeError:
            # Fallback for Docling versions without page_break_placeholder
            md = doc.export_to_markdown()

        return _fix_scattered_vietnamese(md)

    def _extract_images_with_urls(
        self,
        doc,
        document_id: int,
    ) -> tuple[list[ExtractedImage], list[tuple[str, str]]]:
        """
        Extract images and build URL mapping for markdown placeholders.

        Returns:
            (images, pic_url_list) where pic_url_list has one (caption, url)
            tuple per doc.pictures element, in order.
        """
        if not settings.NEXUSRAG_ENABLE_IMAGE_EXTRACTION:
            return [], []

        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        images: list[ExtractedImage] = []
        pic_to_image_idx: list[int] = []  # maps picture index → images list index
        picture_count = 0

        if not hasattr(doc, "pictures") or not doc.pictures:
            return [], []

        for pic in doc.pictures:
            if picture_count >= settings.NEXUSRAG_MAX_IMAGES_PER_DOC:
                pic_to_image_idx.append(-1)
                continue

            image_id = str(uuid.uuid4())

            # Get page number
            page_no = 0
            if hasattr(pic, "prov") and pic.prov:
                for prov in pic.prov:
                    if hasattr(prov, "page_no"):
                        page_no = prov.page_no or 0
                        break

            try:
                image_path = images_dir / f"{image_id}.png"

                if hasattr(pic, "image") and pic.image:
                    pil_image = pic.image.pil_image
                    if pil_image:
                        pil_image.save(str(image_path), format="PNG")
                        width, height = pil_image.size
                    else:
                        pic_to_image_idx.append(-1)
                        continue
                else:
                    pic_to_image_idx.append(-1)
                    continue

                # Get caption
                caption = ""
                if hasattr(pic, "caption_text"):
                    caption = pic.caption_text(doc) if callable(pic.caption_text) else str(pic.caption_text or "")
                elif hasattr(pic, "text"):
                    caption = str(pic.text or "")

                images.append(ExtractedImage(
                    image_id=image_id,
                    document_id=document_id,
                    page_no=page_no,
                    file_path=str(image_path),
                    caption=caption,
                    width=width,
                    height=height,
                ))
                pic_to_image_idx.append(len(images) - 1)
                picture_count += 1

            except Exception as e:
                logger.warning(f"Failed to extract image from document {document_id}: {e}")
                pic_to_image_idx.append(-1)
                continue

        logger.info(f"Extracted {len(images)} images from document {document_id}")

        # NOTE: captioning is intentionally NOT done here.
        # caption_worker.py calls caption_images() / caption_tables() separately
        # so the parse phase stays LLM-free.

        # Build pic_url_list (captions empty at this point — that's fine)
        pic_url_list: list[tuple[str, str]] = []
        for idx in pic_to_image_idx:
            if idx >= 0:
                img = images[idx]
                url = f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
                pic_url_list.append((img.caption, url))
            else:
                pic_url_list.append(("", ""))

        return images, pic_url_list

    def _inject_image_references(
        self, markdown: str, pic_url_list: list[tuple[str, str]]
    ) -> str:
        """Replace <!-- image --> placeholders with ![caption](url) markdown."""
        placeholder_count = len(re.findall(r"<!--\s*image\s*-->", markdown))

        if not pic_url_list:
            if placeholder_count > 0:
                logger.warning(
                    f"Markdown has {placeholder_count} image placeholders but "
                    f"pic_url_list is empty — images will NOT be injected"
                )
            return markdown

        logger.info(
            f"Injecting {len(pic_url_list)} image URLs into "
            f"{placeholder_count} placeholders"
        )

        injected = 0
        pic_iter = iter(pic_url_list)

        def replacer(match):
            nonlocal injected
            try:
                caption, url = next(pic_iter)
                if url:
                    safe_caption = caption.replace("[", "").replace("]", "")
                    # Markdown ![alt](url) must be single-line — collapse newlines
                    safe_caption = " ".join(safe_caption.split())
                    injected += 1
                    return f"\n![{safe_caption}]({url})\n"
                return ""
            except StopIteration:
                return ""

        result = re.sub(r'<!--\s*image\s*-->', replacer, markdown)
        logger.info(f"Injected {injected}/{placeholder_count} image references")
        return result

    # ------------------------------------------------------------------
    # Table extraction & captioning
    # ------------------------------------------------------------------

    def _extract_tables(self, doc, document_id: int) -> list[ExtractedTable]:
        """Extract tables from Docling document."""
        if not hasattr(doc, "tables") or not doc.tables:
            return []

        tables: list[ExtractedTable] = []
        for table in doc.tables:
            table_id = str(uuid.uuid4())

            # Get page number
            page_no = 0
            if hasattr(table, "prov") and table.prov:
                for prov in table.prov:
                    if hasattr(prov, "page_no"):
                        page_no = prov.page_no or 0
                        break

            # Export to markdown
            try:
                content_md = table.export_to_markdown(doc)
            except Exception:
                content_md = ""

            if not content_md.strip():
                continue

            # Get dimensions
            num_rows = 0
            num_cols = 0
            if hasattr(table, "data") and table.data:
                num_rows = getattr(table.data, "num_rows", 0) or 0
                num_cols = getattr(table.data, "num_cols", 0) or 0

            tables.append(ExtractedTable(
                table_id=table_id,
                document_id=document_id,
                page_no=page_no,
                content_markdown=content_md,
                num_rows=num_rows,
                num_cols=num_cols,
            ))

        logger.info(f"Extracted {len(tables)} tables from document {document_id}")
        return tables

    _TABLE_CAPTION_PROMPT = (
        "You are a document analysis assistant. Given a markdown table, "
        "write a concise description that covers:\n"
        "- The purpose/topic of the table\n"
        "- Key column names and what they represent\n"
        "- Notable values, trends, or outliers\n\n"
        "RULES:\n"
        "- Write 2-4 sentences, max 500 characters.\n"
        "- Be factual — describe only what is in the table.\n"
        "- Write in the SAME LANGUAGE as the table content. "
        "If the table is in Vietnamese, write in Vietnamese. "
        "If in English, write in English.\n\n"
        "Table:\n"
    )

    def _caption_tables(self, tables: list[ExtractedTable]) -> None:
        """Caption tables using LLM (text-only, no vision needed)."""
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMMessage

        provider = get_llm_provider()

        for tbl in tables:
            if tbl.caption:
                continue
            try:
                table_md = tbl.content_markdown
                # Truncate large tables
                if len(table_md) > settings.NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS:
                    table_md = table_md[:settings.NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS] + "\n... (truncated)"

                message = LLMMessage(
                    role="user",
                    content=self._TABLE_CAPTION_PROMPT + table_md,
                    images=[],
                )
                result = provider.complete([message])
                if result:
                    tbl.caption = " ".join(result.strip().split())[:500]
            except Exception as e:
                logger.debug(f"Failed to caption table {tbl.table_id}: {e}")

    @staticmethod
    def _inject_table_captions(
        markdown: str, tables: list[ExtractedTable]
    ) -> str:
        """Inject table captions as blockquotes after matching table blocks in markdown."""
        if not tables:
            return markdown

        # Only process tables that have captions
        captioned = [t for t in tables if t.caption]
        if not captioned:
            return markdown

        lines = markdown.split("\n")
        result_lines: list[str] = []
        matched_count = 0

        # Build a lookup: first data row content → table caption
        # (skip the header row and separator row, use the first data row)
        table_lookup: dict[str, ExtractedTable] = {}
        for tbl in captioned:
            tbl_lines = tbl.content_markdown.strip().split("\n")
            # Find first data row (skip header + separator)
            for tl in tbl_lines:
                tl_stripped = tl.strip()
                if tl_stripped.startswith("|") and "---" not in tl_stripped:
                    # Use cell content as key (strip pipes and whitespace)
                    cells = [c.strip() for c in tl_stripped.split("|") if c.strip()]
                    if cells:
                        key = "|".join(cells[:3])  # Use first 3 cells as key
                        table_lookup[key] = tbl
                        break

        i = 0
        while i < len(lines):
            line = lines[i]
            result_lines.append(line)

            # Detect start of a table block (line starts with |)
            if line.strip().startswith("|"):
                # Collect all consecutive table lines
                table_block_start = i
                while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                    i += 1
                    result_lines.append(lines[i])

                # Try to match this table block to a captioned table
                block_lines = lines[table_block_start:i + 1]
                for bl in block_lines:
                    bl_stripped = bl.strip()
                    if bl_stripped.startswith("|") and "---" not in bl_stripped:
                        cells = [c.strip() for c in bl_stripped.split("|") if c.strip()]
                        if cells:
                            key = "|".join(cells[:3])
                            if key in table_lookup:
                                tbl = table_lookup.pop(key)
                                result_lines.append(f"\n> **Table:** {tbl.caption}")
                                matched_count += 1
                                break

            i += 1

        logger.info(
            f"Injected {matched_count}/{len(captioned)} table captions into markdown"
        )
        return "\n".join(result_lines)

    def _caption_images(self, images: list[ExtractedImage]) -> None:
        """Caption images using the configured LLM provider (sync, best-effort).

        Generates detailed descriptions so that image content is
        semantically searchable when embedded alongside text chunks.
        """
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMImagePart, LLMMessage

        provider = get_llm_provider()
        if not provider.supports_vision():
            logger.warning("LLM provider does not support vision — skipping image captioning")
            return

        _CAPTION_PROMPT = (
            "Describe ONLY what you can directly see in this image. "
            "Do NOT infer, assume, or add any information not visible.\n\n"
            "Include:\n"
            "- Type of visual (chart, table, diagram, photo, screenshot, etc.)\n"
            "- ALL specific numbers, percentages, and labels that are VISIBLE in the image\n"
            "- Axis labels, legend text, and category names exactly as shown\n"
            "- Trends or comparisons that are visually obvious\n\n"
            "RULES:\n"
            "- Write 2-4 concise sentences, max 400 characters.\n"
            "- Do NOT start with 'This image shows' or 'Here is'.\n"
            "- Do NOT add any data, context, or interpretation beyond what is visible.\n"
            "- If text in the image is not clearly readable, say so."
        )

        for img in images:
            if img.caption:  # already has caption from document
                continue
            try:
                image_path = Path(img.file_path)
                if not image_path.exists():
                    continue

                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                message = LLMMessage(
                    role="user",
                    content=_CAPTION_PROMPT,
                    images=[LLMImagePart(data=image_bytes, mime_type=img.mime_type)],
                )
                result = provider.complete([message])
                if result:
                    # Collapse to single line — prevents breaking ![alt](url) markdown
                    img.caption = " ".join(result.strip().split())[:500]

            except Exception as e:
                logger.debug(f"Failed to caption image {img.image_id}: {e}")

    def _parse_legacy(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Fallback: parse TXT/MD with legacy loader + RecursiveCharacterTextSplitter."""
        from app.services.document_loader import load_document
        from app.services.chunker import DocumentChunker

        loaded = load_document(str(file_path))
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        text_chunks = chunker.split_text(
            text=loaded.content,
            source=original_filename,
            extra_metadata={"document_id": document_id, "file_type": loaded.file_type},
        )

        # Wrap legacy chunks as EnrichedChunks
        chunks = [
            EnrichedChunk(
                content=tc.content,
                chunk_index=tc.chunk_index,
                source_file=original_filename,
                document_id=document_id,
                page_no=0,
            )
            for tc in text_chunks
        ]

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=loaded.content,
            page_count=loaded.page_count,
            chunks=chunks,
            images=[],
            tables_count=0,
        )

"""
Administrative-document layout reconstruction (shared core)
===========================================================

Both the OCR path (Unlimited-OCR bounding boxes) and the Docling path (digital
PDFs, which carry per-element ``prov.bbox`` + semantic ``label``) feed generic
``LayoutBlock``s into :func:`render_layout_blocks`, which rebuilds the original
Vietnamese administrative layout (Nghị định 30/2020/NĐ-CP) as HTML: a 2-column
national heading, centred titles, left-aligned body, top-right form notes and a
signature/seal block.  Every block keeps ``data-bbox`` + ``data-page`` so the
viewer can locate it, and :func:`strip_layout_html` removes the markup again
before embedding so the vectors stay clean.

Producers build the blocks (text content already HTML-escaped); this module owns
the geometry (content-relative alignment, row grouping, 2-column detection) so
the two paths render identically.
"""
from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass


def esc(text: str) -> str:
    """HTML-escape text and turn newlines into <br/> (for multi-line blocks)."""
    return _html.escape(text).replace("\n", "<br/>")


def normalize_footnotes(text: str) -> str:
    """Convert OCR's LaTeX footnote refs `\\( ^{(3)} \\)` → plain `(3)`."""
    text = re.sub(r"\\\(\s*\^?\s*\{?\s*\(?\s*(\d+)\s*\)?\s*\}?\s*\\\)", r"(\1)", text)
    text = re.sub(r"\^\{\s*([^}]*?)\s*\}", r"\1", text)
    text = text.replace("\\(", "").replace("\\)", "")
    return text


@dataclass
class LayoutBlock:
    """One logical block with its page position.

    ``html`` is the already-built inner HTML for inline kinds (note/title/body),
    or a full element (``<figure>``/``<table>``/``<ul>``) for ``block`` kinds.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    html: str
    kind: str = "body"        # 'note' | 'title' | 'body' | 'figure' | 'table' | 'list'
    block: bool = False        # True → emit html as-is inside an alignment <div>
    multiline: bool = False

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)


def render_layout_blocks(blocks: list[LayoutBlock], page_no: int) -> str:
    """Render blocks (one page) into administrative-layout HTML."""
    if not blocks:
        return ""

    # Alignment is relative to the CONTENT column (min x1 → max x2), not the
    # raster/page width — admin docs leave wide, uneven margins.
    content_left = min(b.x1 for b in blocks)
    content_right = max(b.x2 for b in blocks)
    cw = max(1.0, content_right - content_left)
    ccenter = (content_left + content_right) / 2

    single_heights = sorted(b.h for b in blocks if not b.multiline and b.h > 0)
    line_h = single_heights[len(single_heights) // 2] if single_heights else 24.0

    # Group blocks into rows by vertical overlap (→ detects the 2-column header).
    # Block elements (figures/tables) are kept on their OWN row: a tall figure
    # (e.g. a signature/seal spanning several lines) must not absorb the text
    # rows it overlaps, which would scramble reading order.
    blocks.sort(key=lambda b: (b.y1, b.x1))
    rows: list[list[LayoutBlock]] = []
    for b in blocks:
        if b.block:
            rows.append([b])
            continue
        for row in rows:
            if any(x.block for x in row):
                continue
            ry1 = min(x.y1 for x in row)
            ry2 = max(x.y2 for x in row)
            if min(ry2, b.y2) - max(ry1, b.y1) > 0.4 * max(1.0, min(b.h, ry2 - ry1)):
                row.append(b)
                break
        else:
            rows.append([b])
    rows.sort(key=lambda row: min(x.y1 for x in row))

    def bbox(*bs: LayoutBlock) -> str:
        return (
            f"{int(min(b.x1 for b in bs))},{int(min(b.y1 for b in bs))},"
            f"{int(max(b.x2 for b in bs))},{int(max(b.y2 for b in bs))}"
        )

    def align(b: LayoutBlock) -> str:
        if (b.x1 - content_left) <= cw * 0.05:
            return "left"
        if (content_right - b.x2) <= cw * 0.05:
            return "right"
        return "center"

    parts: list[str] = []
    for row in rows:
        row.sort(key=lambda b: b.x1)

        # 2-column row: exactly two inline blocks straddling the content centre.
        if (
            len(row) == 2
            and not row[0].block
            and not row[1].block
            and row[0].cx < ccenter <= row[1].cx
        ):
            left, right = row
            multiline = (
                left.multiline or right.multiline
                or left.h > 1.8 * line_h or right.h > 1.8 * line_h
            )
            cls = "ocr-header-grid" if multiline else "ocr-row-2col"
            parts.append(
                f'<div class="{cls}">'
                f'<div class="ocr-col-left" data-bbox="{bbox(left)}" data-page="{page_no}">{left.html}</div>'
                f'<div class="ocr-col-right" data-bbox="{bbox(right)}" data-page="{page_no}">{right.html}</div>'
                f"</div>"
            )
            continue

        for b in row:
            a = align(b)
            db = f'data-bbox="{bbox(b)}" data-page="{page_no}"'
            if b.block:
                parts.append(f'<div class="ocr-block ocr-{a}" {db}>{b.html}</div>')
            elif b.kind == "note":
                parts.append(f'<p class="ocr-note" {db}>{b.html}</p>')
            elif b.kind == "title":
                acls = "ocr-center" if a == "center" else f"ocr-{a}"
                parts.append(f'<p class="ocr-title {acls}" {db}>{b.html}</p>')
            else:
                bcls = "ocr-body" if a == "left" else f"ocr-{a}"
                parts.append(f'<p class="{bcls}" {db}>{b.html}</p>')

    return "\n".join(parts)


# Tags we emit; comments (<!-- page N -->) are preserved so the chunker /
# DocumentViewer page dividers keep working.
_LAYOUT_TAG_RE = re.compile(r"<(?!!--)[^>]*>")


def strip_layout_html(text: str) -> str:
    """Strip layout HTML (tags + data-bbox), leaving clean reading-order text.

    No-op when no layout markers are present, so it is safe to apply to every
    chunk (Docling markdown is left untouched). Robust to chunk boundaries that
    split a tag.
    """
    if "data-bbox" not in text:
        return text
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Block boundaries → newline so adjacent blocks (2-column header, list items)
    # don't get glued together once the tags are removed.
    text = re.sub(r"</(div|p|figure|li|tr|table|ul|ol)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</t[dh]>", " ", text, flags=re.IGNORECASE)
    text = _LAYOUT_TAG_RE.sub("", text)          # complete tags (keep <!-- -->)
    text = re.sub(r"<(?!!--)[^>]*$", "", text)    # trailing tag fragment (split chunk)
    text = re.sub(r"^[^<>]*>", "", text)          # leading attr fragment (split chunk)
    text = _html.unescape(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

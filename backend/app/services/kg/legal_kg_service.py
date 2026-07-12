"""
Legal Knowledge Graph Service
==============================

Purpose-built KG extraction pipeline for Vietnamese administrative/legal documents.
Replaces LightRAG's generic extraction with domain-specific, structure-aware pipeline.

Pipeline per document:
  1. Structural Splitter  — split markdown by Điều/Khoản/Điểm
  2. Header Parser        — extract document meta (số hiệu, ngày ban hành, loại VB)
  3. Preamble Parser      — extract CAN_CU from the "Căn cứ..." block
  4. LLM Extractor        — per-article extraction (2 prompt variants)
  5. Neo4j Storage        — MERGE-based upsert with typed relations

Entity Resolution strategy:
  - Organization: Canonicalization via {document_meta} — always full name
  - Person:       Composite Key "[Tên] (ngày sinh | CCCD | đơn vị | không xác định)"
  - Date format:  Python-side normalization to DD/MM/YYYY before Cypher MERGE

Query strategy:
  - All lookups use CONTAINS (case-insensitive) instead of exact match
  - Ensures "Nguyễn Văn A" finds "Nguyễn Văn A (15/03/1975)" in Neo4j
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Optional

from app.core.config import settings
from app.services.llm import get_kg_llm_provider
from app.services.llm_logger import MinIOLoggerService
from app.services.llm.types import LLMMessage
from app.prompts.legal_kg import (
    ENTITY_RESOLVE_SYSTEM_PROMPT,
    ENTITY_RESOLVE_USER_PROMPT,
    LEGAL_KG_SYSTEM_PROMPT,
    LEGAL_KG_USER_PROMPT,
    PERSON_EXTRACT_SYSTEM_PROMPT,
    PERSON_EXTRACT_USER_PROMPT,
    PREAMBLE_SYSTEM_PROMPT,
    PREAMBLE_USER_PROMPT,
    PERSON_DOCUMENT_TRIGGERS,
)

logger = logging.getLogger(__name__)

# Max concurrent LLM calls during extraction
_LLM_SEMAPHORE = asyncio.Semaphore(settings.HRAG_KG_LLM_CONCURRENCY)

# Per-ingest LLM logger (set by ingest() when HRAG_KG_LOG_EXTRACTION is on).
# Module-level _call_llm reads it via this ContextVar so every extraction call
# is captured without threading a logger through every method signature.
# asyncio.gather child tasks inherit the context set before the gather, so the
# per-article extraction tasks all see the same logger.
_kg_log_ctx: ContextVar[Optional[MinIOLoggerService]] = ContextVar(
    "_kg_log_ctx", default=None
)

# ---------------------------------------------------------------------------
# Date normalization utilities
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    re.compile(r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})"),  # DD/MM/YYYY or D/M/YYYY
    re.compile(r"(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})"),  # YYYY/MM/DD
]


def normalize_date(raw: str) -> str:
    """Normalize any date-like string to DD/MM/YYYY. Returns 'không xác định' on failure."""
    if not raw or not raw.strip():
        return "không xác định"
    raw = raw.strip()
    # Try DD/MM/YYYY family first
    m = _DATE_PATTERNS[0].search(raw)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{day:02d}/{month:02d}/{year}"
    # Try YYYY/MM/DD
    m = _DATE_PATTERNS[1].search(raw)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{day:02d}/{month:02d}/{year}"
    # Bare year only → ambiguous
    if re.fullmatch(r"\d{4}", raw):
        return "không xác định"
    return raw  # return as-is if unrecognized


def build_person_composite_key(name: str, props: dict) -> str:
    """
    Build a disambiguated Person entity_id from raw LLM output.
    Priority: ngày sinh → CCCD/thẻ đảng → đơn vị → 'không xác định'
    """
    name = name.strip()
    # Already has composite key injected by LLM
    if "(" in name and ")" in name:
        return name

    ngay_sinh = props.get("ngay_sinh", "")
    if ngay_sinh:
        normalized = normalize_date(ngay_sinh)
        if normalized != "không xác định":
            return f"{name} ({normalized})"

    cccd = props.get("cccd") or props.get("so_the_dang")
    if cccd:
        return f"{name} ({cccd})"

    don_vi = props.get("don_vi_moi") or props.get("don_vi_cu")
    if don_vi:
        return f"{name} ({don_vi})"

    return f"{name} (không xác định)"


# Vietnamese lowercase particles that should NOT be title-cased
_VN_PARTICLES = {
    "và", "của", "tại", "trong", "từ", "về", "theo", "với", "để", "có",
    "cho", "khi", "là", "trên", "đến", "qua", "sau", "thành", "ra",
    "vào", "tới", "bởi", "nếu", "mà", "hay", "hoặc",
}


def normalize_org_name(name: str) -> str:
    """
    Canonical normalization for Organization and Document entity IDs.

    Problem: LLM may output same org with different capitalizations:
      "Sở Thông tin và Truyền thông"
      "Sở Thông Tin và Truyền thông"   (capital T in Tin)
      "SỞ THÔNG TIN VÀ TRUYỀN THÔNG"  (all-caps header)

    Solution: lowercase → title-case with Vietnamese particle exceptions
    All three examples above → "Sở Thông Tin Và Truyền Thông"

    The canonical form is stored as entity_id; original kept as display_name.
    """
    # Step 1: remove '#' and normalize whitespace
    name = name.replace("#", "")
    name = " ".join(name.strip().split())
    if not name:
        return name
    # Step 2: lowercase everything
    words = name.lower().split()
    # Step 3: capitalize each word except Vietnamese particles (keep lower),
    #         but ALWAYS capitalize first word
    result = []
    for i, word in enumerate(words):
        if i == 0 or word not in _VN_PARTICLES:
            # capitalize() handles unicode, lowercases all-but-first char
            result.append(word.capitalize())
        else:
            result.append(word)
    return " ".join(result)


# Pattern for Vietnamese legal document numbers (e.g., 29/2018/QH14, 01/2023/NĐ-CP).
# The type suffix may contain the Vietnamese letter "Đ"/"đ" (NĐ-CP, QĐ-…), which is
# NOT in the ASCII A-Z range — it MUST be in the class, else .group(0) truncates
# "85/2016/NĐ-CP" to "85/2016/N" (and distinct types like NĐ-CP vs NQ-CP collide).
_DOC_NUM_PATTERN = re.compile(r"\d+/\d+/[A-Za-z0-9Đđ-]+", re.IGNORECASE)

# Keywords that indicate an entity MUST be a Document type
_LEGAL_DOC_PREFIXES = re.compile(
    r"^(Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Chỉ thị|Nghị quyết|Hiến pháp|Pháp lệnh)\b",
    re.IGNORECASE
)

# Names opening with a structural-part keyword are Article/clause REFERENCES, not
# documents — even when they embed a số hiệu (e.g. "Khoản 4 Điều 3 Nghị định
# 53/2022/NĐ-CP"). They must be typed Article so the số-hiệu Document key below
# never collapses a clause reference onto the document root.
_ARTICLE_REF_PREFIX = re.compile(
    r"^(Điều|Khoản|Điểm|Mục|Chương|Phần)\b",
    re.IGNORECASE
)


def normalize_entity_id(name: str, entity_type: str) -> str:
    """
    Return canonical entity_id for MERGE key:
    - Document → the embedded số hiệu when present (so "Nghị định 53/2022/NĐ-CP
      (Chính phủ, 2022)", "53/2022/NĐ-CP" and "53/2022/nđ-cp" all merge onto ONE
      node); otherwise normalize_org_name of the full name.
    - Organization, Article, Location → normalize_org_name (case-folded canonical)
    - Person → unchanged (Person uses composite key for disambiguation)
    - Task → whitespace-normalized only
    """
    name = " ".join(name.strip().split())  # strip + collapse spaces
    if entity_type == "Document":
        m = _DOC_NUM_PATTERN.search(name)
        if m:
            return normalize_org_name(m.group(0))
        return normalize_org_name(name)
    if entity_type in ("Organization", "Article", "Location"):
        return normalize_org_name(name)
    return name


def _is_article_ref_name(name: str) -> bool:
    """True when a name opens with a structural-part keyword (Điều/Khoản/…)."""
    return bool(name and _ARTICLE_REF_PREFIX.match(name.strip()))


def _is_legal_doc_name(name: str) -> bool:
    """
    True when a raw entity name denotes a legal document — it carries a số hiệu
    (e.g. ``117/2025/QH15``) or opens with a law-type prefix (``Luật``, ``Nghị
    định``, …) — and is NOT a clause reference. Such entities MUST be typed
    ``Document`` regardless of how the LLM / resolution pass classified them.
    """
    if not name or _is_article_ref_name(name):
        return False
    return bool(_DOC_NUM_PATTERN.search(name) or _LEGAL_DOC_PREFIXES.search(name))


def _force_legal_type(name: str, etype: str) -> str:
    """
    Override an LLM-assigned type for legal-structure names:
      - clause references (Điều/Khoản/…) → Article
      - laws/decrees (số hiệu or law prefix) → Document
    Anything else keeps its original type.
    """
    if _is_article_ref_name(name):
        return "Article"
    if etype != "Document" and _is_legal_doc_name(name):
        return "Document"
    return etype


def _doc_alias_base(s: str) -> str:
    """Canonical form for doc-identity comparison: normalize + drop a trailing
    "(issuer, year)" suffix."""
    return re.sub(r"\s*\([^)]+\)\s*$", "", normalize_org_name(s)).strip()


def _is_doc_root_alias(name: str, doc_name: str = "", doc_title: str = "") -> bool:
    """
    True when ``name`` IS the current document itself — it matches the structured
    doc_name or the document title (EXACT match after normalization + dropping a
    trailing "(issuer, year)" suffix). Used to fold a title/full-name reference
    that only ever appears as a relation endpoint onto the số-hiệu Document root.

    Exact match only — never substring — so a sub-part of the title such as
    "Luật An ninh mạng" (a DIFFERENT document the decree details) is not swallowed.
    """
    if not name:
        return False
    n = _doc_alias_base(name)
    if not n:
        return False
    return any(n == _doc_alias_base(ref) for ref in (doc_name, doc_title) if ref)


# ---------------------------------------------------------------------------
# Generic / junk entity guard (deterministic, model-independent)
# ---------------------------------------------------------------------------
# LLMs keep emitting a small, FINITE set of non-specific "organizations"
# ("Bộ", "Cơ quan nhà nước", "các … có liên quan"), form-template placeholders
# ("(tên đơn vị đề nghị)") and form names ("Mẫu số 02", "Tờ trình"), no matter
# how strongly the prompt forbids them. They survive LLM resolution and get
# re-materialised as nodes from relation endpoints. Drop them here so they never
# become nodes. CONSERVATIVE by design — a precise stoplist + tight patterns —
# so real (even long, comma-listed) names like "Doanh nghiệp viễn thông, doanh
# nghiệp cung cấp dịch vụ …" are NOT dropped.
_GENERIC_ORG_EXACT = frozenset({
    "bộ", "các bộ", "bộ trưởng", "các bộ trưởng", "cơ quan", "các cơ quan",
    "cơ quan ngang bộ", "cơ quan thuộc chính phủ", "cơ quan nhà nước",
    "cơ quan, tổ chức nhà nước", "đơn vị", "các đơn vị", "doanh nghiệp",
    "các doanh nghiệp", "tổ chức", "các tổ chức", "tổ chức chính trị",
    "cá nhân", "tổ chức, cá nhân", "chủ quản hệ thống thông tin",
    "thủ trưởng cơ quan ngang bộ", "thủ trưởng cơ quan thuộc chính phủ",
})
_GENERIC_ORG_PAT = re.compile(
    r"(có liên quan$|^các\s+(cơ quan|tổ chức|bộ|ngành|đơn vị|doanh nghiệp|cá nhân)\b)",
    re.IGNORECASE,
)
_JUNK_PLACEHOLDER_PAT = re.compile(r"\(\s*(tên|chủ quản|đơn vị|cơ quan)\b[^)]*\)", re.IGNORECASE)
_JUNK_FORM_PAT = re.compile(r"^\s*(mẫu số|tờ trình|biểu mẫu|đơn đề nghị)\b", re.IGNORECASE)


def _is_generic_or_junk(name: str, entity_type: str = "") -> bool:
    """
    True when ``name`` is a non-specific / template entity that must NOT become a
    graph node. Placeholder & form names are junk for ANY type; the generic-org
    stoplist/pattern only applies to Organization (or unknown) types so a clause
    ref or document with an incidental match is never dropped.
    """
    n = (name or "").strip()
    if not n:
        return True
    if _JUNK_PLACEHOLDER_PAT.search(n) or _JUNK_FORM_PAT.search(n):
        return True
    if entity_type in ("", "Organization"):
        low = n.lower()
        if low in _GENERIC_ORG_EXACT or _GENERIC_ORG_PAT.search(low):
            return True
    return False


# ---------------------------------------------------------------------------
# Extraction windowing — avoid silently dropping the tail of long articles
# ---------------------------------------------------------------------------

# Max characters fed to the LLM per extraction call. Articles longer than this
# are split into overlapping windows (previously the text was hard-truncated to
# this size with `text[:3000]`, dropping everything after it).
_ARTICLE_EXTRACT_MAX_CHARS = 3000


def _split_text_windows(text: str, max_chars: int, overlap: int = 200) -> list[str]:
    """
    Split ``text`` into windows of at most ``max_chars`` characters with a small
    overlap so entities/relations spanning a boundary are still seen by at least
    one window. Returns ``[text]`` unchanged when it already fits.
    """
    text = text or ""
    if len(text) <= max_chars:
        return [text]
    windows: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        # Prefer to cut on a newline/space boundary near the window end.
        if end < n:
            brk = text.rfind("\n", start + max_chars - overlap, end)
            if brk == -1:
                brk = text.rfind(" ", start + max_chars - overlap, end)
            if brk > start:
                end = brk
        windows.append(text[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return windows


# ---------------------------------------------------------------------------
# Multi-document node/relation ownership (Cypher fragments)
# ---------------------------------------------------------------------------
# Shared entities (e.g. an Organization cited by many documents) must NOT be
# deleted when one citing document is removed. Each node/relationship therefore
# tracks a `document_ids` list; deletion only removes a node/rel once its last
# owning document is gone. These helpers build the Cypher to seed/append that
# list, migrating legacy rows that only have the singular `document_id`.


def _doc_ids_seed(var: str) -> str:
    """Seed expr: existing list, else [singular document_id], else []."""
    return (
        f"coalesce({var}.document_ids, "
        f"CASE WHEN {var}.document_id IS NULL THEN [] ELSE [{var}.document_id] END)"
    )


def _doc_ids_append(var: str, id_param: str) -> str:
    """Append ``$id_param`` to ``var.document_ids`` (idempotent, null-safe)."""
    seed = _doc_ids_seed(var)
    return (
        f"CASE WHEN ${id_param} IS NULL THEN {seed} "
        f"WHEN ${id_param} IN {seed} THEN {seed} "
        f"ELSE {seed} + ${id_param} END"
    )


# ---------------------------------------------------------------------------
# Structural document splitter
# ---------------------------------------------------------------------------

# Patterns for Vietnamese legal document structure boundaries
# Matches "Điều 1. ..." with optional markdown heading prefix "## "
_DIEU_PATTERN = re.compile(
    r"^(?:#{1,6}\s+)?(Điều\s+\d+[a-zA-Z]?\.?\s*.{0,120})",
    re.MULTILINE | re.IGNORECASE,
)
# Docling broken-spacing format: "Đ i ề u 1", "## Đ i ề u 7"
_DIEU_BROKEN_PATTERN = re.compile(
    r"^(?:#{1,4}\s*)?Đ\s*i\s*ề\s*u\s+\d+",
    re.MULTILINE | re.IGNORECASE,
)
_SECTION_HEADERS = re.compile(
    r"^(CHƯƠNG\s+\w+[^\n]*|MỤC\s+\d+[^\n]*|PHẦN\s+\w+[^\n]*)",
    re.MULTILINE | re.IGNORECASE,
)


def _normalize_broken_dieu(text: str) -> str:
    """
    Fix Docling's broken spacing where 'Điều' becomes 'Đ i ề u'.
    
    Transforms:
      '## Đ i ề u 7. Ph ạ m vi ...'  →  '## Điều 7. Ph ạ m vi ...'
    
    This ensures the standard regex can match article boundaries.
    """
    # Fix "Đ i ề u" → "Điều" (with optional markdown heading prefix)
    return re.sub(
        r'((?:^|\n)(?:#{1,4}\s*)?)Đ\s*i\s*ề\s*u(\s+\d+)',
        r'\1Điều\2',
        text,
    )


def split_articles(markdown: str) -> list[dict]:
    """
    Split a Vietnamese legal document markdown into per-Điều chunks.

    Returns a list of dicts:
      {
        "heading": "Điều 5. Tổ chức thực hiện",
        "text": "...",          # full article text including heading
        "index": 5              # article number for reference
      }
    """
    # Pre-process: apply Vietnamese scattered-char fix from the parser
    # This handles Docling's per-glyph spacing issues in both headings and body
    from app.services.parsing.deep_document_parser import _fix_scattered_vietnamese
    markdown = _fix_scattered_vietnamese(markdown)

    # Pre-process: fix broken spacing "Đ i ề u" → "Điều"
    has_broken = bool(_DIEU_BROKEN_PATTERN.search(markdown))
    if has_broken:
        markdown = _normalize_broken_dieu(markdown)
        logger.info("split_articles: fixed broken 'Đ i ề u' spacing from Docling")

    # Find all Điều boundaries
    matches = list(_DIEU_PATTERN.finditer(markdown))
    if not matches:
        # Fallback: return whole document as single chunk
        return [{"heading": "Toàn văn", "text": markdown.strip(), "index": 0}]

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        chunk_text = markdown[start:end].strip()
        heading = match.group(1).strip()

        # Extract article number
        num_match = re.search(r"Điều\s+(\d+)", heading, re.IGNORECASE)
        index = int(num_match.group(1)) if num_match else i + 1

        if chunk_text:
            chunks.append({"heading": heading, "text": chunk_text, "index": index})

    return chunks


# ---------------------------------------------------------------------------
# Document header / preamble parser
# ---------------------------------------------------------------------------

_PREAMBLE_END_PATTERN = re.compile(
    r"(?:QUYẾT ĐỊNH:|QUY ĐỊNH:|THÔNG TƯ:|CHỈ THỊ:|CỬ\s+ÔNG|ĐIỀU 1\b)",
    re.IGNORECASE,
)

_HEADER_PATTERNS = {
    "so_hieu": re.compile(r"Số:\s*([\w\d/\-\.]+)", re.IGNORECASE),
    "ngay_ban_hanh": re.compile(
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE
    ),
    "co_quan_ban_hanh": re.compile(
        r"^([A-ZÀÁẢÃẠĂẮẶẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][^\n]{5,80})\n",
        re.MULTILINE,
    ),
}


def parse_document_meta(markdown: str) -> dict:
    """
    Extract document metadata from the top of a legal document.
    Returns dict with: so_hieu, ngay_ban_hanh, co_quan_ban_hanh, document_name.
    """
    # Work with first 2000 chars (header area)
    header_text = markdown[:2000]

    meta: dict[str, str] = {}

    m = _HEADER_PATTERNS["so_hieu"].search(header_text)
    if m:
        meta["so_hieu"] = m.group(1).strip()

    m = _HEADER_PATTERNS["ngay_ban_hanh"].search(header_text)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        meta["ngay_ban_hanh"] = f"{int(day):02d}/{int(month):02d}/{year}"

    # Co quan ban hanh: first all-caps line near the top
    for line in header_text.splitlines():
        line = line.strip()
        if len(line) > 5 and line.isupper():
            meta["co_quan_ban_hanh"] = line
            break

    # Build human-readable document name
    so_hieu = meta.get("so_hieu", "")
    co_quan = meta.get("co_quan_ban_hanh", "")
    if so_hieu:
        meta["document_name"] = so_hieu
    elif co_quan:
        meta["document_name"] = co_quan[:80]
    else:
        meta["document_name"] = "Văn bản không xác định"

    return meta


def extract_preamble(markdown: str) -> str:
    """Extract the preamble block (before QUYẾT ĐỊNH: / ĐIỀU 1)."""
    end_match = _PREAMBLE_END_PATTERN.search(markdown)
    if end_match:
        return markdown[: end_match.start()].strip()
    return markdown[:1500].strip()  # fallback


def is_personnel_document(markdown: str) -> bool:
    """Detect if Document is a personnel decision (trigger keywords in first 500 chars)."""
    header_lower = markdown[:500].lower()
    return any(trigger in header_lower for trigger in PERSON_DOCUMENT_TRIGGERS)


# ---------------------------------------------------------------------------
# LLM caller with retry
# ---------------------------------------------------------------------------


async def _call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    log_meta: Optional[dict] = None,
) -> str:
    """Call LegalKG LLM with exponential-backoff retry for rate limits.

    When an ingest set up an extraction logger (see _kg_log_ctx), the
    request/response is buffered for the fine-tuning dataset. `log_meta`
    carries per-call tags (stage, article_ref, doc_type, …) used to filter
    the dataset later.
    """
    provider = get_kg_llm_provider()
    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]
    for attempt in range(4):
        try:
            async with _LLM_SEMAPHORE:
                response = await asyncio.wait_for(
                    provider.acomplete(messages, temperature=0.0, max_tokens=max_tokens),
                    timeout=60,
                )
            _logger = _kg_log_ctx.get()
            if _logger is not None:
                model_name = (
                    getattr(provider, "model", None)
                    or getattr(provider, "model_name", None)
                    or getattr(provider, "_model", None)
                    or settings.LEGAL_KG_LLM_MODEL
                )
                _logger.log_llm_call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=response,
                    model=model_name,
                    metadata_extra=log_meta,
                )
            return response
        except Exception as e:
            err = str(e).lower()
            is_rate = "429" in err or "rate" in err or "quota" in err or "resource_exhausted" in err
            if is_rate and attempt < 3:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
    return ""


def _parse_llm_json(raw: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    raw = raw.strip()
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: find the first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"entities": [], "relations": []}


# ---------------------------------------------------------------------------
# Main LegalKGService
# ---------------------------------------------------------------------------


class LegalKGService:
    """
    Per-workspace Knowledge Graph service for Vietnamese administrative/legal documents.

    Drop-in replacement for KnowledgeGraphService.
    Requires Neo4j backend (HRAG_KG_GRAPH_BACKEND=neo4j or the factory selects this).
    """

    def __init__(self, workspace_id: uuid.UUID):
        self.workspace_id = workspace_id
        self._label = f"kb_{workspace_id}"
        self._driver = None

    # ------------------------------------------------------------------
    # Driver management
    # ------------------------------------------------------------------

    async def _get_driver(self):
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                connection_timeout=10,
                connection_acquisition_timeout=15,
            )
            logger.info(f"LegalKGService connected to Neo4j for workspace {self.workspace_id}")
        except ImportError:
            raise RuntimeError("neo4j driver not installed. Run: pip install neo4j")
        return self._driver

    async def cleanup(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Document deletion
    # ------------------------------------------------------------------

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """
        Remove this document's contribution to the workspace graph.

        Nodes and relationships are SHARED across documents (e.g. an Organization
        cited by many documents). Each carries a ``document_ids`` ownership list,
        so we only drop a node/relationship once THIS document was its last owner;
        otherwise we just remove this document_id from the list. This prevents the
        old bug where deleting one document wiped shared entities used by others.

        Called before reprocessing a document to prevent KG duplicates/orphans.
        """
        driver = await self._get_driver()
        label = self._label
        doc_id_str = str(document_id)
        seed_r = _doc_ids_seed("r")
        seed_n = _doc_ids_seed("n")
        try:
            async with driver.session() as session:
                # 1) Relationships: trim ownership, delete when no owner remains.
                rel_res = await (await session.run(
                    f"""
                    MATCH (:`{label}`)-[r]->(:`{label}`)
                    WHERE $doc_id IN {seed_r}
                    WITH r, [x IN {seed_r} WHERE x <> $doc_id] AS remaining
                    SET r.document_ids = remaining,
                        r.document_id = CASE WHEN size(remaining) > 0
                                             THEN remaining[0] ELSE r.document_id END
                    WITH r, remaining WHERE size(remaining) = 0
                    DELETE r
                    """,
                    doc_id=doc_id_str,
                )).consume()

                # 2) Nodes: trim ownership, DETACH DELETE when no owner remains.
                node_res = await (await session.run(
                    f"""
                    MATCH (n:`{label}`)
                    WHERE $doc_id IN {seed_n}
                    WITH n, [x IN {seed_n} WHERE x <> $doc_id] AS remaining
                    SET n.document_ids = remaining,
                        n.document_id = CASE WHEN size(remaining) > 0
                                            THEN remaining[0] ELSE n.document_id END
                    WITH n, remaining WHERE size(remaining) = 0
                    DETACH DELETE n
                    """,
                    doc_id=doc_id_str,
                )).consume()

                logger.info(
                    f"LegalKG delete_document({document_id}): "
                    f"{node_res.counters.nodes_deleted} nodes deleted, "
                    f"{rel_res.counters.relationships_deleted} rels deleted "
                    f"for workspace {self.workspace_id}"
                )
        except Exception as e:
            logger.error(
                f"LegalKG delete_document({document_id}) failed: {e}",
                exc_info=True,
            )
            raise

    async def update_document_metadata(
        self,
        document_id: uuid.UUID,
        doc_number: str | None = None,
        doc_title: str | None = None,
        signer_name: str | None = None,
        issuing_agency: str | None = None,
        published_date: str | None = None,
    ) -> None:
        """
        Sync the root Document node when document metadata is edited by hand.

        Identity rules (see _upsert_document_root for the full model):
          - The node is located by its stable ``document_id`` anchor (NOT by the
            Neo4j internal <id>, which Neo4j may recycle after deletions). If no
            anchored node exists, we log and return — we never MERGE a brand-new
            orphan node here.
          - ``display_name`` (front-end label) is always refreshed from the title.
          - ``entity_id`` (canonical link key) is re-keyed to the số hiệu the
            FIRST time a số hiệu appears (i.e. the node is still on its
            ``doc:{uuid}`` / name fallback). If another Document node already owns
            that số hiệu (e.g. a CAN_CU stub from another document), the two are
            merged via APOC so links converge instead of duplicating.
        """
        logger.info(
            f"LegalKG.update_document_metadata: doc_id={document_id}, "
            f"doc_num='{doc_number}', doc_title='{doc_title}', "
            f"signer='{signer_name}', issuing_agency='{issuing_agency}', "
            f"published_date='{published_date}'"
        )
        driver = await self._get_driver()
        label = self._label
        doc_id_str = str(document_id)

        so_hieu = (doc_number or "").strip()
        new_entity_id = normalize_entity_id(so_hieu, "Document") if so_hieu else None
        display_name = (doc_title or so_hieu or "").strip() or f"Tài liệu {document_id}"

        desc_parts = []
        if signer_name:
            desc_parts.append(f"Người ký: {signer_name}")
        if issuing_agency:
            desc_parts.append(f"Cơ quan ban hành: {issuing_agency}")
        description = "; ".join(desc_parts)

        try:
            async with driver.session() as session:
                # 1) Locate the root node by its stable document_id anchor
                rec = await (await session.run(
                    f"""
                    MATCH (n:`{label}`:`Document` {{document_id: $doc_id}})
                    RETURN n.entity_id AS eid
                    """,
                    doc_id=doc_id_str,
                )).single()
                if rec is None:
                    logger.warning(
                        f"LegalKG update_document_metadata({document_id}): no root "
                        f"Document node anchored by document_id — skipping (KG may "
                        f"not have been built for this document)."
                    )
                    return
                cur_eid = rec["eid"]

                # 2) Re-key entity_id to số hiệu the first time it appears,
                #    merging any pre-existing stub that already owns that số hiệu.
                if new_entity_id and cur_eid != new_entity_id:
                    collision = await (await session.run(
                        f"""
                        MATCH (m:`{label}`:`Document` {{entity_id: $eid}})
                        WHERE m.document_id IS NULL OR m.document_id <> $doc_id
                        RETURN count(m) AS c
                        """,
                        eid=new_entity_id, doc_id=doc_id_str,
                    )).single()
                    if collision and collision["c"] > 0:
                        # Merge stub(s) INTO the anchored root (root listed first so
                        # it survives); mergeRels keeps the stub's incoming links.
                        await session.run(
                            f"""
                            MATCH (root:`{label}`:`Document` {{document_id: $doc_id}})
                            MATCH (stub:`{label}`:`Document` {{entity_id: $eid}})
                            WHERE id(stub) <> id(root)
                            WITH root, collect(stub) AS stubs
                            CALL apoc.refactor.mergeNodes(
                                [root] + stubs,
                                {{properties: 'discard', mergeRels: true}}
                            ) YIELD node
                            RETURN node
                            """,
                            doc_id=doc_id_str, eid=new_entity_id,
                        )
                        logger.info(
                            f"LegalKG update_document_metadata({document_id}): "
                            f"merged stub Document node(s) for số hiệu '{new_entity_id}'"
                        )

                # 3) Refresh the anchored root's display + canonical key + description
                set_clauses = ["n.display_name = $display_name", "n.updated_at = datetime()"]
                params = {"doc_id": doc_id_str, "display_name": display_name}
                if new_entity_id:
                    set_clauses.append("n.entity_id = $entity_id")
                    params["entity_id"] = new_entity_id
                if description:
                    set_clauses.append("n.description = $description")
                    params["description"] = description

                result = await session.run(
                    f"""
                    MATCH (n:`{label}`:`Document` {{document_id: $doc_id}})
                    SET {", ".join(set_clauses)}
                    """,
                    **params,
                )
                summary = await result.consume()
                logger.info(
                    f"LegalKG update_document_metadata({document_id}): "
                    f"properties_set={summary.counters.properties_set}, "
                    f"entity_id={'(re-keyed) ' + new_entity_id if new_entity_id and new_entity_id != cur_eid else '(unchanged)'}"
                )
        except Exception as e:
            logger.error(f"LegalKG update_document_metadata({document_id}) failed: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Ingestion pipeline
    # ------------------------------------------------------------------

    async def ingest(self, markdown_content: str, document_id: Optional[uuid.UUID] = None) -> None:
        """Public ingest entrypoint.

        When HRAG_KG_LOG_EXTRACTION is on, set up the extraction-logging context
        around the real pipeline so every KG LLM call (article extract, preamble,
        entity resolve) is captured for the fine-tuning dataset, then flush the
        buffered calls to MinIO under datasets/legal_kg_extraction/. The flush /
        contextvar reset run in `finally` so a failed ingest never leaks the
        logger into the worker's context or loses partial data.
        """
        if not settings.HRAG_KG_LOG_EXTRACTION:
            await self._ingest_impl(markdown_content, document_id)
            return

        base_meta = {
            "kg_mode": "legal",
            "workspace_id": str(self.workspace_id),
            "document_id": str(document_id) if document_id else None,
        }
        ext_logger = MinIOLoggerService(
            dataset_prefix="legal_kg_extraction", base_meta=base_meta
        )
        token = _kg_log_ctx.set(ext_logger)
        try:
            await self._ingest_impl(
                markdown_content, document_id, log_base_meta=base_meta
            )
        finally:
            _kg_log_ctx.reset(token)
            if document_id is not None:
                await ext_logger.flush_to_minio(
                    workspace_id=self.workspace_id, document_id=document_id
                )

    async def _ingest_impl(
        self,
        markdown_content: str,
        document_id: Optional[uuid.UUID] = None,
        log_base_meta: Optional[dict] = None,
    ) -> None:
        """
        Full LegalKG ingestion pipeline:
          1. Parse document metadata
          2. Extract preamble CAN_CU
          3. Split into articles
          4. LLM-extract per article (with concurrency limit)
          5. Store all results to Neo4j
        """
        if not markdown_content.strip():
            logger.warning(f"LegalKG: empty content for workspace {self.workspace_id}, skipping")
            return

        # Convert UUID to string for Neo4j (Neo4j doesn't support UUID objects)
        doc_id_str = str(document_id) if document_id else None

        # Step 1: Document metadata
        doc_meta = parse_document_meta(markdown_content)
        doc_name = doc_meta.get("document_name", "Văn bản không xác định")
        
        # --- Fetch Rich Header Metadata from Database ---
        doc_type = ""
        doc_num = ""
        doc_title = ""
        loc = ""
        issue_org = ""
        parent_org = ""
        year = "Không rõ năm"

        if document_id:
            try:
                from app.core.database import async_session_maker
                from app.models.document import Document
                from sqlalchemy import select
                from sqlalchemy.orm import selectinload

                async with async_session_maker() as _db:
                    stmt = select(Document).options(selectinload(Document.document_type)).where(Document.id == document_id)
                    db_doc = await _db.scalar(stmt)
                    if db_doc:
                        # Extract what we injected in parse_worker
                        doc_num = db_doc.document_number or ""
                        doc_title = db_doc.document_title or ""
                        doc_type = db_doc.document_type.name if db_doc.document_type else ""
                        loc = db_doc.location or ""
                        issue_org = db_doc.issuing_agency or ""
                        parent_org = db_doc.parent_agency or ""

                        # Fallback for published year
                        pd = db_doc.published_date or ""
                        import re
                        m = re.search(r'\b(20\d{2})\b', pd)
                        if m:
                            year = m.group(1)

                        # Build super-structured doc_name if we found metadata
                        if doc_num and doc_type:
                            context_str = f"{parent_org}, {year}" if parent_org else year
                            doc_name = f"{doc_type} {doc_num} ({context_str})"

            except Exception as _e:
                logger.warning(f"LegalKG: Failed to fetch Document metadata: {_e}")

        # --- Canonical identity for the root Document node ---
        # entity_id (MERGE/link key) = số hiệu when available, so cross-document
        # references by số hiệu merge onto the same node; falls back to the
        # structured doc_name for scanned docs that have no số hiệu yet.
        # display_name (front-end) prefers document_title.
        # The stable update/delete anchor is the document_id property (set below).
        root_so_hieu = (doc_num or doc_meta.get("so_hieu", "")).strip()
        # When the metadata pass failed to surface a số hiệu, recover it from the
        # structured doc_name (which embeds it, e.g. "Nghị định 53/2022/NĐ-CP (…)").
        # Without this, re-ingests where số hiệu extraction differs produce two root
        # nodes for the SAME document (số-hiệu key vs doc_name key) that never merge.
        if not root_so_hieu:
            _m = _DOC_NUM_PATTERN.search(doc_name)
            if _m:
                root_so_hieu = _m.group(0)
        root_key = root_so_hieu or doc_name
        root_display = doc_title or doc_name

        # --- Resolve custom KG system prompt from document_type ---
        custom_kg_prompt: str | None = None
        if document_id:
            try:
                from app.core.database import async_session_maker
                from app.models.document import Document
                from app.models.document_type import DocumentTypeSystemPrompt
                from sqlalchemy import select

                async with async_session_maker() as _db:
                    stmt = select(Document).where(Document.id == document_id)
                    db_doc = await _db.scalar(stmt)
                    if db_doc and db_doc.document_type_id:
                        doc_type_id = db_doc.document_type_id
                        # Lookup custom KG prompt (workspace_id=NULL for global)
                        res = await _db.execute(
                            select(DocumentTypeSystemPrompt.kg_system_prompt).where(
                                DocumentTypeSystemPrompt.document_type_id == doc_type_id,
                                DocumentTypeSystemPrompt.workspace_id.is_(None),
                            )
                        )
                        custom_kg_prompt = res.scalar_one_or_none()
            except Exception as _e:
                logger.warning(f"LegalKG: failed to resolve custom KG prompt: {_e}")

        is_personnel = is_personnel_document(markdown_content)

        # Enrich the dataset's document-level tags now that doc_type is known.
        # base_meta is shared by reference with the extraction logger, so this
        # is reflected in every buffered call for this document.
        if log_base_meta is not None:
            log_base_meta.update({
                "doc_type": doc_type,
                "doc_name": doc_name,
                "is_personnel": is_personnel,
            })

        logger.info(
            f"LegalKG ingest workspace={self.workspace_id} doc='{doc_name}' "
            f"personnel={is_personnel} doc_id={document_id} location={loc}"
        )

        # Step 2: Preamble CAN_CU extraction
        preamble_text = extract_preamble(markdown_content)
        can_cu_list = await self._extract_preamble_can_cu(preamble_text, doc_name)

        # Step 3: Structural split
        articles = split_articles(markdown_content)
        logger.info(f"LegalKG: split into {len(articles)} articles")

        # Step 4: Build rich doc_meta with DB-extracted fields (including document_title)
        rich_meta = dict(doc_meta)
        if doc_title:
            rich_meta["tieu_de"] = doc_title
        doc_meta_str = self._format_doc_meta(rich_meta)
        tasks = [
            self._extract_with_llm(
                article, doc_meta_str, doc_name, is_personnel, document_id, custom_kg_prompt,
                doc_title=doc_title,
                doc_num=doc_num,
                issuing_agency=issue_org,
                published_date=year,
            )
            for article in articles
        ]
        article_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 4.5: Collect all entities + LLM entity resolution
        all_raw_entities: list[dict] = []
        for result in article_results:
            if isinstance(result, Exception) or not result:
                continue
            for ent in result.get("entities", []):
                all_raw_entities.append({
                    "name": ent.get("name", ""),
                    "type": ent.get("type", "Organization"),
                    "description": ent.get("description", ""),
                    "article_ref": ent.get("article_ref", ""),
                })

        # LLM resolution pass — deduplicate + resolve type conflicts + prefer Document Root
        resolved_entities: list[dict] = []
        merged_map: dict[str, str] = {}  # raw_name → canonical_name
        skipped_entities: set[str] = set()  # entity names that were dropped
        if all_raw_entities:
            try:
                resolved_entities, merged_map, skipped_entities = await self._resolve_entities_with_llm(
                    all_raw_entities, doc_meta_str, doc_name, doc_title
                )
                logger.info(
                    f"LegalKG: resolved {len(all_raw_entities)} raw entities → "
                    f"{len(resolved_entities)} unique, {len(merged_map)} merged, {len(skipped_entities)} dropped"
                )
            except Exception as e:
                logger.warning(f"LegalKG: entity resolution failed, using fallback: {e}")
                resolved_entities, merged_map, skipped_entities = self._basic_entity_normalize(all_raw_entities)

        # Step 5: Store all results
        driver = await self._get_driver()
        neo4j_node_id = ""  # Will hold the Neo4j internal <id> for the root Document node
        async with driver.session() as session:
            # Store Document node and get its internal Neo4j <id>
            neo4j_node_id = await self._upsert_document_root(
                session, root_key, root_display, doc_id_str,
                description=doc_meta.get("so_hieu", ""),
            )
            logger.info(f"LegalKG: created root Document node with Neo4j id={neo4j_node_id}")

            # --- Store Root Context (Location & Org Hierarchy) ---
            loc_name = ""
            if loc:
                loc_name = loc if "tỉnh" in loc.lower() or "thành phố" in loc.lower() else f"Tỉnh {loc}"
                await self._upsert_node(session, loc_name, "Location", "", doc_id_str)
                await self._upsert_relation(
                    session, root_key, "BAN_HANH_TAI", loc_name, "Phạm vi địa lý", doc_id_str,
                    source_type="Document", target_type="Location",
                )

            # parse_worker sometimes fills issuing_agency and parent_agency with
            # the SAME value. Detect that (case/whitespace-insensitive, or one
            # containing the other) so we don't build a doubled "X X" org name and
            # a meaningless TRUC_THUOC self-loop.
            norm_issue = normalize_entity_id(issue_org, "Organization") if issue_org else ""
            norm_parent = normalize_entity_id(parent_org, "Organization") if parent_org else ""
            same_org = bool(norm_issue) and bool(norm_parent) and (
                norm_issue == norm_parent
                or norm_parent in norm_issue
                or norm_issue in norm_parent
            )

            if parent_org and issue_org and not same_org:
                # Distinct issuing sub-unit under a parent org → full name + hierarchy.
                combined_issue_org = f"{issue_org} {parent_org}"
                await self._upsert_node(session, combined_issue_org, "Organization", "", doc_id_str)
                await self._upsert_node(session, parent_org, "Organization", "", doc_id_str)

                await self._upsert_relation(
                    session, root_key, "BAN_HANH_BOI", combined_issue_org, "", doc_id_str,
                    source_type="Document", target_type="Organization",
                )
                await self._upsert_relation(
                    session, combined_issue_org, "TRUC_THUOC", parent_org, "", doc_id_str,
                    source_type="Organization", target_type="Organization",
                )

                if loc_name:
                    await self._upsert_relation(
                        session, parent_org, "THUOC_TINH", loc_name, "", doc_id_str,
                        source_type="Organization", target_type="Location",
                    )
            elif issue_org or parent_org:
                # Single issuing org: only one provided, or issue == parent.
                # Prefer the more complete (longer) name.
                org = max((o for o in (issue_org, parent_org) if o), key=len)
                await self._upsert_node(session, org, "Organization", "", doc_id_str)
                await self._upsert_relation(
                    session, root_key, "BAN_HANH_BOI", org, "", doc_id_str,
                    source_type="Document", target_type="Organization",
                )
                if loc_name:
                    await self._upsert_relation(
                        session, org, "THUOC_TINH", loc_name, "", doc_id_str,
                        source_type="Organization", target_type="Location",
                    )


            # Store CAN_CU relations from preamble
            for ref_doc in can_cu_list:
                await self._upsert_node(session, ref_doc, "Document", ref_doc, doc_id_str)
                await self._upsert_relation(
                    session, root_key, "CAN_CU", ref_doc,
                    f"Căn cứ pháp lý: {ref_doc}", doc_id_str,
                    source_type="Document", target_type="Document",
                )

            # Step 5.5: Upsert resolved (deduplicated) entities
            entity_type_map: dict[str, str] = {}   # canonical → entity_type

            # Build entity_type_map from resolved entities
            for ent in resolved_entities:
                canonical = ent.get("canonical_name", "")
                if not canonical:
                    continue
                # Correct legal-structure mistypes: laws/decrees → Document,
                # clause refs (Điều/Khoản/…) → Article. Without the Article split
                # a clause carrying a số hiệu would collapse onto the doc root.
                etype = _force_legal_type(canonical, ent.get("type", "Organization"))
                entity_type_map[canonical] = etype

            # Build canonical_lookup: any name → canonical (from merged_map + merged_from)
            canonical_lookup: dict[str, str] = {}
            # Add from merged_map (LLM-computed raw→canonical)
            for raw_name, canonical in merged_map.items():
                canonical_lookup[raw_name] = canonical
            # Add from resolved entities merged_from (canonical itself + merged raw names)
            for ent in resolved_entities:
                canonical = ent.get("canonical_name", "")
                if not canonical:
                    continue
                ent_type = ent.get("type", "Organization")
                canonical_lookup[canonical] = canonical
                for raw_name in ent.get("merged_from", []):
                    canonical_lookup[raw_name] = canonical
                # Also map normalized form
                normalized = normalize_entity_id(canonical, ent_type)
                if normalized != canonical:
                    canonical_lookup[normalized] = canonical

            # Upsert resolved entities (only canonical ones, not duplicates).
            # Use the (legal-doc-corrected) type from entity_type_map, not the raw
            # LLM type, so a mis-typed law/decree lands on a single Document node.
            for ent in resolved_entities:
                canonical = ent.get("canonical_name", "")
                if not canonical:
                    continue
                await self._upsert_node(
                    session, canonical,
                    entity_type_map.get(canonical, ent.get("type", "Organization")),
                    ent.get("representative_description", ""), doc_id_str,
                )

            # Step 5.6: Store article relations using resolved entity lookup
            for i, result in enumerate(article_results):
                if isinstance(result, Exception) or not result:
                    continue
                await self._store_relations_from_extraction(
                    session, result, root_key, doc_id_str,
                    canonical_lookup, merged_map, skipped_entities, entity_type_map,
                    doc_name=doc_name, doc_title=doc_title,
                )

        entity_count = sum(
            len(r.get("entities", [])) for r in article_results
            if isinstance(r, dict)
        )
        rel_count = sum(
            len(r.get("relations", [])) for r in article_results
            if isinstance(r, dict)
        )
        logger.info(
            f"LegalKG stored: {entity_count} entities, {rel_count} relations "
            f"for workspace {self.workspace_id}"
        )

        # Save kg_root_entity_id (Neo4j internal <id>) back to Document table for future metadata updates
        if doc_id_str and neo4j_node_id:
            try:
                from app.core.database import async_session_maker
                from sqlalchemy import text
                async with async_session_maker() as _db:
                    await _db.execute(
                        text("UPDATE documents SET kg_root_entity_id = :node_id WHERE id = :doc_id"),
                        {"node_id": neo4j_node_id, "doc_id": document_id}
                    )
                    await _db.commit()
                logger.info(f"LegalKG: saved kg_root_entity_id='{neo4j_node_id}' for doc_id={doc_id_str}")
            except Exception as _e:
                logger.warning(f"LegalKG: failed to save kg_root_entity_id for doc_id={doc_id_str}: {_e}")

    def _format_doc_meta(self, meta: dict) -> str:
        parts = []
        if meta.get("so_hieu"):
            parts.append(f"Số hiệu: {meta['so_hieu']}")
        if meta.get("tieu_de"):
            parts.append(f"Tiêu đề: {meta['tieu_de']}")
        if meta.get("co_quan_ban_hanh"):
            parts.append(f"Cơ quan ban hành: {meta['co_quan_ban_hanh']}")
        if meta.get("ngay_ban_hanh"):
            parts.append(f"Ngày ban hành: {meta['ngay_ban_hanh']}")
        return "; ".join(parts) if parts else "Không có thông tin"

    async def _extract_preamble_can_cu(self, preamble_text: str, doc_name: str) -> list[str]:
        """Extract CAN_CU list from preamble using LLM, with regex pre-filter."""
        # Fast regex pre-extraction
        regex_results = re.findall(r"Căn cứ\s+(.+?)(?:;|\n|$)", preamble_text, re.IGNORECASE)
        if regex_results:
            cleaned = [r.strip().rstrip(";,. ") for r in regex_results if len(r.strip()) > 5]
            if cleaned:
                return cleaned

        # Fallback to LLM if regex misses
        if not preamble_text.strip():
            return []
        try:
            user_prompt = PREAMBLE_USER_PROMPT.format(
                preamble_text=preamble_text[:1500],
                document_name=doc_name,
            )
            raw = await _call_llm(
                PREAMBLE_SYSTEM_PROMPT, user_prompt, max_tokens=1024,
                log_meta={"stage": "preamble"},
            )
            data = _parse_llm_json(raw)
            return data.get("can_cu_list", [])
        except Exception as e:
            logger.warning(f"LegalKG preamble extraction failed: {e}")
            return []

    async def _extract_with_llm(
        self,
        article: dict,
        doc_meta_str: str,
        doc_name: str,
        is_personnel: bool,
        document_id: Optional[uuid.UUID],
        custom_system_prompt: str | None = None,
        doc_title: str = "",
        doc_num: str = "",
        issuing_agency: str = "",
        published_date: str = "",
    ) -> dict:
        """Run LLM extraction on a single article, windowing long text."""
        text = article["text"]
        heading = article["heading"]
        article_ref = f"Điều {article['index']}"

        # Choose prompt variant: custom per-document-type > personnel > general.
        # custom & general share the LEGAL user template; personnel has its own.
        if custom_system_prompt:
            system_prompt = custom_system_prompt
            user_template = LEGAL_KG_USER_PROMPT
        elif is_personnel:
            system_prompt = PERSON_EXTRACT_SYSTEM_PROMPT
            user_template = PERSON_EXTRACT_USER_PROMPT
        else:
            system_prompt = LEGAL_KG_SYSTEM_PROMPT
            user_template = LEGAL_KG_USER_PROMPT

        # Long articles are split into overlapping windows so the tail is not
        # dropped; results are merged and de-duplicated across windows.
        windows = _split_text_windows(text, _ARTICLE_EXTRACT_MAX_CHARS)
        if len(windows) > 1:
            logger.info(
                f"LegalKG: {heading} is {len(text)} chars → extracting in "
                f"{len(windows)} windows"
            )

        merged_entities: list[dict] = []
        merged_relations: list[dict] = []
        seen_entities: set[tuple] = set()
        seen_relations: set[tuple] = set()

        for win in windows:
            user_prompt = user_template.format(
                document_title=doc_title or "Không có tiêu đề",
                document_number=doc_num or "Không có số hiệu",
                issuing_agency=issuing_agency or "Không xác định",
                published_date=published_date or "Không xác định",
                article_text=win,
            )
            try:
                raw = await _call_llm(
                    system_prompt, user_prompt,
                    log_meta={
                        "stage": "article_extract",
                        "article_ref": article_ref,
                        "heading": heading,
                        "is_personnel": is_personnel,
                    },
                )
                data = _parse_llm_json(raw)
            except Exception as e:
                logger.warning(f"LegalKG LLM extraction failed for {heading}: {e}")
                continue

            for e in data.get("entities", []):
                key = (str(e.get("name", "")).strip().lower(), str(e.get("type", "")).strip())
                if key in seen_entities:
                    continue
                seen_entities.add(key)
                merged_entities.append(e)
            for r in data.get("relations", []):
                key = (
                    str(r.get("source", "")).strip().lower(),
                    str(r.get("relation", "")).strip().upper(),
                    str(r.get("target", "")).strip().lower(),
                )
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                merged_relations.append(r)

        # Tag each entity/relation with its source article
        for e in merged_entities:
            e.setdefault("article_ref", article_ref)
            e.setdefault("document_id", document_id)
        for r in merged_relations:
            r.setdefault("article_ref", article_ref)
            r.setdefault("document_id", document_id)
        return {"entities": merged_entities, "relations": merged_relations}

    # ------------------------------------------------------------------
    # Entity Resolution (Step 4.5 — deduplicate after LLM extraction)
    # ------------------------------------------------------------------

    # Self-reference patterns for Vietnamese legal documents
    _SELF_REF_PATTERNS = re.compile(
        r"^(văn bản này|quyết định này|nghị định này|thông tư này|"
        r"luật này|bộ luật này|pháp lệnh này|nghị quyết này|chỉ thị này)\s*$",
        re.IGNORECASE,
    )
    # Document number pattern (e.g., 13/2024/NĐ-CP)
    _DOC_NUM_IN_NAME = re.compile(r"\d+/\d+/[A-Z0-9\-]+", re.IGNORECASE)

    def _premark_document_root_references(
        self,
        entity_entries: list[dict],
        doc_name: str,
        doc_title: str = "",
    ) -> tuple[list[dict], dict[str, str], set[str]]:
        """
        Pre-mark entities that are references to the Document Root (Node Cha).

        Returns:
          - remaining_entries: entities that need LLM resolution
          - auto_merged_map: raw_name → doc_name for auto-detected root references
          - auto_dropped: set of entity names auto-dropped
        """
        # Extract all document numbers from doc_name
        doc_numbers = set(self._DOC_NUM_IN_NAME.findall(doc_name))
        # Normalize doc_numbers for comparison
        doc_numbers_normalized = {n.lower() for n in doc_numbers}

        auto_merged: dict[str, str] = {}   # raw_name → doc_name
        auto_dropped: set[str] = set()    # names that are self-refs without meaning
        remaining: list[dict] = []

        for e in entity_entries:
            name = e["raw_name"]
            name_lower = name.lower()

            # Check 0: Generic / junk entity ("Bộ", "Cơ quan nhà nước",
            # "(tên đơn vị…)", "Mẫu số 02"…) — drop so it never becomes a node,
            # and (via skipped_entities downstream) any relation touching it is
            # skipped too.
            if _is_generic_or_junk(name, e["type"]):
                auto_dropped.add(name)
                logger.debug(f"LegalKG: auto-dropped generic/junk '{name}'")
                continue

            # Check 1: Self-reference patterns ("văn bản này", "Nghị định này", etc.)
            if self._SELF_REF_PATTERNS.match(name_lower):
                auto_merged[name] = doc_name
                logger.debug(f"LegalKG: auto-merged self-ref '{name}' → {doc_name}")
                continue

            # Check 2: Entity contains same document number as doc_name
            entity_numbers = set(self._DOC_NUM_IN_NAME.findall(name))
            if entity_numbers and entity_numbers == doc_numbers_normalized:
                # Same document number → merge into Document Root
                auto_merged[name] = doc_name
                logger.debug(f"LegalKG: auto-merged same doc number '{name}' → {doc_name}")
                continue

            # Check 3: Normalize and compare with normalized doc_name
            doc_name_norm = normalize_entity_id(doc_name, "Document")
            name_norm = normalize_entity_id(name, e["type"])
            # Remove common suffixes like "(UBND Tỉnh X, 2024)" for comparison
            doc_name_base = re.sub(r"\s*\([^)]+\)\s*$", "", doc_name_norm).strip()
            name_base = re.sub(r"\s*\([^)]+\)\s*$", "", name_norm).strip()
            if name_base and name_base == doc_name_base:
                auto_merged[name] = doc_name
                logger.debug(f"LegalKG: auto-merged base-match '{name}' → {doc_name}")
                continue

            # Check 4: Match against document_title (Tiêu đề văn bản)
            if doc_title:
                doc_title_norm = normalize_entity_id(doc_title, "Document")
                doc_title_base = re.sub(r"\s*\([^)]+\)\s*$", "", doc_title_norm).strip()
                # If entity matches document_title (after normalization), merge to doc_name
                if name_base and doc_title_base and name_base == doc_title_base:
                    auto_merged[name] = doc_name
                    logger.debug(f"LegalKG: auto-merged title-match '{name}' → {doc_name}")
                    continue
                # Also check if document_title is a substring or vice versa (for partial matches)
                if len(name_base) > 5 and (name_base in doc_title_base or doc_title_base in name_base):
                    auto_merged[name] = doc_name
                    logger.debug(f"LegalKG: auto-merged partial-title-match '{name}' → {doc_name}")
                    continue

            remaining.append(e)

        return remaining, auto_merged, auto_dropped

    async def _resolve_entities_with_llm(
        self,
        raw_entities: list[dict],
        doc_meta_str: str,
        doc_name: str,
        doc_title: str = "",
    ) -> tuple[list[dict], dict[str, str], set[str]]:
        """
        Two-phase entity resolution:
          1. Pre-mark Document Root references (code-based, exact match)
          2. LLM resolution for remaining entities

        Returns 3-tuple:
          - resolved_entities: canonical entities with merged_from
          - merged_map: raw_name → canonical_name (for relation rerouting)
          - skipped_entities: entity names that were dropped
        """
        if not raw_entities:
            return [], {}, set()

        # Pre-normalize: keep raw name + normalized name for each entity
        entity_entries: list[dict] = []
        for ent in raw_entities:
            name = str(ent.get("name", "")).strip()
            name = re.sub(r"^[#\*\- \t]+", "", name).strip()
            if not name:
                continue
            etype = str(ent.get("type", "Organization")).strip()
            norm_name = normalize_entity_id(name, etype)
            entity_entries.append({
                "raw_name": name,
                "norm_name": norm_name,
                "type": etype,
                "description": ent.get("description", ""),
                "article_ref": ent.get("article_ref", ""),
            })

        # === Phase 1: Pre-mark Document Root references (code-based) ===
        remaining_entries, auto_merged, auto_dropped = self._premark_document_root_references(
            entity_entries, doc_name, doc_title
        )
        logger.info(
            f"LegalKG: [Phase1 Pre-mark] doc={doc_name[:50]}... "
            f"total_input={len(entity_entries)} | "
            f"auto_merged={len(auto_merged)} | "
            f"auto_dropped={len(auto_dropped)} | "
            f"remaining_for_llm={len(remaining_entries)}"
        )
        # Detailed log: auto-merged entities
        if auto_merged:
            for raw_name, canonical in auto_merged.items():
                logger.info(f"LegalKG:   [AUTO-MERGED] '{raw_name}' → '{canonical}'")
        # Detailed log: auto-dropped entities
        if auto_dropped:
            for name in auto_dropped:
                logger.info(f"LegalKG:   [AUTO-DROPPED] '{name}'")

        # === Phase 2: LLM resolution for remaining entities ===
        resolved_entities = []
        llm_merged_map: dict[str, str] = {}
        llm_dropped: set[str] = set()

        if remaining_entries:
            # Format entity list for LLM
            entity_lines = []
            for e in remaining_entries:
                entity_lines.append(
                    f'- raw: "{e["raw_name"]}" | norm: "{e["norm_name"]}" | type: {e["type"]} | article: {e["article_ref"]}'
                )
            entity_list_str = "\n".join(entity_lines)

            user_prompt = ENTITY_RESOLVE_USER_PROMPT.format(
                doc_name=doc_name,
                document_title=doc_title or "Không có tiêu đề",
                doc_meta=doc_meta_str,
                entity_list=entity_list_str,
            )

            try:
                raw = await _call_llm(
                    ENTITY_RESOLVE_SYSTEM_PROMPT, user_prompt, max_tokens=2048,
                    log_meta={"stage": "resolve"},
                )
                data = _parse_llm_json(raw)

                conflicts = data.get("type_conflicts", [])
                if conflicts:
                    logger.info(f"LegalKG: LLM resolved {len(conflicts)} type conflicts")

                dropped = data.get("dropped_entities", [])
                if dropped:
                    logger.info(f"LegalKG: LLM dropped {len(dropped)} entities")

                # Build norm_name → canonical from LLM resolved_entities
                name_to_canonical: dict[str, str] = {}
                for ent in data.get("resolved_entities", []):
                    canonical = ent.get("canonical_name", "")
                    if not canonical:
                        continue
                    merged_from = ent.get("merged_from", [])
                    for n in merged_from:
                        name_to_canonical[n] = canonical

                for ent in dropped:
                    dropped_name = ent.get("name", "")
                    if dropped_name:
                        llm_dropped.add(dropped_name)
                    reason = ent.get("reason", "")
                    logger.debug(f"LegalKG: LLM dropped '{dropped_name}' reason: {reason}")

                # Group remaining_entries by canonical to build resolved list
                canonical_map: dict[tuple, dict] = {}
                for e in remaining_entries:
                    canonical = name_to_canonical.get(e["norm_name"], e["norm_name"])
                    key = (canonical, e["type"])
                    if key not in canonical_map:
                        canonical_map[key] = {
                            "canonical_name": canonical,
                            "type": e["type"],
                            "representative_description": e["description"],
                            "source_articles": [e["article_ref"]],
                            "merged_from": [e["raw_name"]],
                        }
                    else:
                        existing = canonical_map[key]
                        if e["description"] and not existing["representative_description"]:
                            existing["representative_description"] = e["description"]
                        if e["article_ref"] not in existing["source_articles"]:
                            existing["source_articles"].append(e["article_ref"])
                        if e["raw_name"] not in existing["merged_from"]:
                            existing["merged_from"].append(e["raw_name"])

                # Build merged_map from LLM resolution
                for (canonical_name, _etype), ent_data in canonical_map.items():
                    for n in ent_data.get("merged_from", []):
                        if n != canonical_name:
                            llm_merged_map[n] = canonical_name

                resolved_entities = list(canonical_map.values())

            except Exception as e:
                logger.warning(f"LegalKG: LLM resolution failed, using fallback: {e}")
                # Fallback: basic normalization for remaining entries
                seen: dict[tuple, dict] = {}
                for e in remaining_entries:
                    key = (e["norm_name"], e["type"])
                    if key not in seen:
                        seen[key] = {
                            "canonical_name": e["norm_name"],
                            "type": e["type"],
                            "representative_description": e["description"],
                            "source_articles": [e["article_ref"]],
                            "merged_from": [e["raw_name"]],
                        }
                resolved_entities = list(seen.values())

        # === Combine Phase 1 (auto) + Phase 2 (LLM) ===
        # merged_map: auto_merged + llm_merged_map
        merged_map = {**auto_merged, **llm_merged_map}
        # skipped_entities: auto_dropped + llm_dropped
        skipped_entities = auto_dropped | llm_dropped

        # === Detailed logging: BEFORE vs AFTER ===
        logger.info(f"LegalKG: ========== ENTITY RESOLUTION SUMMARY ==========")
        logger.info(f"LegalKG: [BEFORE] Total raw entities: {len(raw_entities)}")
        for i, e in enumerate(raw_entities):
            logger.info(f"LegalKG:   [{i+1:3d}] '{e.get('name', '')}' | type={e.get('type', '?')} | article={e.get('article_ref', '?')}")

        logger.info(f"LegalKG: [AFTER] Canonical entities: {len(resolved_entities)}")
        for i, ent in enumerate(resolved_entities):
            canonical = ent.get("canonical_name", "")
            ent_type = ent.get("type", "?")
            merged_from = ent.get("merged_from", [])
            source_articles = ent.get("source_articles", [])
            logger.info(f"LegalKG:   [{i+1:3d}] CANONICAL: '{canonical}' | type={ent_type}")
            logger.info(f"LegalKG:          merged_from: {merged_from}")
            logger.info(f"LegalKG:          articles: {source_articles}")

        if merged_map:
            logger.info(f"LegalKG: [MERGED_MAP] {len(merged_map)} mappings:")
            for raw_name, canonical in merged_map.items():
                logger.info(f"LegalKG:   '{raw_name}' → '{canonical}'")

        if skipped_entities:
            logger.info(f"LegalKG: [SKIPPED] {len(skipped_entities)} entities dropped:")
            for name in skipped_entities:
                logger.info(f"LegalKG:   DROPPED: '{name}'")

        logger.info(f"LegalKG: ==============================================")

        return resolved_entities, merged_map, skipped_entities

    def _basic_entity_normalize(self, raw_entities: list[dict]) -> tuple[list[dict], dict[str, str], set[str]]:
        """
        Fallback resolver when LLM fails: basic normalization only, no deduplication.
        Returns 3-tuple: (resolved_entities, merged_map, skipped_entities)
        - merged_map is empty (no LLM merging in fallback)
        - skipped_entities is empty (no drops in fallback)
        """
        seen: dict[tuple, dict] = {}
        dropped: set[str] = set()
        for ent in raw_entities:
            name = str(ent.get("name", "")).strip()
            name = re.sub(r"^[#\*\- \t]+", "", name).strip()
            if not name:
                continue
            etype = str(ent.get("type", "Organization")).strip()
            # Apply the same generic/junk guard as the LLM path so the fallback
            # doesn't silently let "Bộ"/"Cơ quan nhà nước"/placeholders through.
            if _is_generic_or_junk(name, etype):
                dropped.add(name)
                continue
            key = (normalize_entity_id(name, etype), etype)
            if key not in seen:
                seen[key] = {
                    "canonical_name": normalize_entity_id(name, etype),
                    "type": etype,
                    "representative_description": ent.get("description", ""),
                    "source_articles": [ent.get("article_ref", "")],
                    "merged_from": [name],
                }
        return list(seen.values()), {}, dropped

    def _build_canonical_lookup(self, resolved_entities: list[dict]) -> dict[str, str]:
        """
        Build name→canonical mapping from resolved entity list using merged_from.
        """
        lookup: dict[str, str] = {}
        for ent in resolved_entities:
            canonical = ent.get("canonical_name", "")
            if not canonical:
                continue
            for name in ent.get("merged_from", []):
                lookup[name] = canonical
            lookup[canonical] = canonical
        return lookup

    # ------------------------------------------------------------------
    # Neo4j storage helpers
    # ------------------------------------------------------------------

    async def _upsert_document_root(
        self,
        session,
        entity_name: str,
        display_name: str = "",
        document_id: Optional[str] = None,
        description: str = "",
    ) -> str:
        """
        Create or update the root Document node and return its Neo4j internal <id>.

        Identity model:
          - entity_id   = normalized số hiệu (or structured name fallback) — the
            canonical MERGE/link key. If a stub Document node with this entity_id
            already exists (e.g. created as a CAN_CU target by another document),
            this call claims it as the real root (sets display_name + document_id).
          - document_id = stable Postgres UUID — the anchor used by
            update_document_metadata() / delete_document() to locate this node.
          - display_name = human-readable name (document_title preferred).

        The returned Neo4j internal <id> is still persisted to
        documents.kg_root_entity_id for backward compatibility, but is no longer
        used to locate the node on update (which matches by document_id instead).
        """
        label = self._label
        canonical_id = normalize_entity_id(entity_name, "Document")
        disp = (display_name or entity_name).replace("#", "").strip()

        cypher = f"""
        MERGE (n:`{label}`:`Document` {{entity_id: $entity_id}})
        ON CREATE SET n.entity_type  = 'Document',
                      n.display_name = $display_name,
                      n.description  = $description,
                      n.document_id  = $document_id,
                      n.document_ids = CASE WHEN $document_id IS NULL THEN [] ELSE [$document_id] END,
                      n.created_at   = datetime()
        ON MATCH SET  n.entity_type  = 'Document',
                      n.display_name = $display_name,
                      n.document_id  = CASE WHEN $document_id IS NOT NULL
                                            THEN $document_id ELSE n.document_id END,
                      n.document_ids = {_doc_ids_append("n", "document_id")},
                      n.description  = CASE WHEN $description <> '' THEN $description ELSE n.description END,
                      n.updated_at   = datetime()
        RETURN id(n) as node_id
        """
        result = await session.run(
            cypher,
            entity_id=canonical_id,
            display_name=disp,
            description=description,
            document_id=str(document_id) if document_id else None,
        )
        record = await result.single()
        if record:
            return str(record["node_id"])  # Neo4j internal <id> as string
        return ""

    async def _upsert_node(
        self,
        session,
        entity_id: str,
        entity_type: str,
        description: str = "",
        document_id: Optional[str] = None,
    ) -> None:
        if not entity_id or not entity_id.strip():
            return
        label = self._label

        # Normalize entity_id to canonical form (deduplication key)
        canonical_id = normalize_entity_id(entity_id, entity_type)
        # Keep original as display_name only on CREATE (human-readable)
        # Clean noise characters specifically for display consistency
        display_name = entity_id.replace("#", "").strip()

        cypher = f"""
        MERGE (n:`{label}`:`{entity_type}` {{entity_id: $entity_id}})
        ON CREATE SET n.entity_type  = $entity_type,
                      n.display_name = $display_name,
                      n.description  = $description,
                      n.document_id  = $document_id,
                      n.document_ids = CASE WHEN $document_id IS NULL THEN [] ELSE [$document_id] END,
                      n.created_at   = datetime()
        ON MATCH SET  n.description  = CASE WHEN $description <> '' THEN $description ELSE n.description END,
                      n.document_ids = {_doc_ids_append("n", "document_id")}
        """
        await session.run(
            cypher,
            entity_id=canonical_id,
            entity_type=entity_type,
            display_name=display_name,
            description=description,
            document_id=str(document_id) if document_id else None,
        )
        return canonical_id  # caller may need canonical id for relation lookup

    async def _upsert_relation(
        self,
        session,
        source: str,
        relation_type: str,
        target: str,
        description: str = "",
        document_id: Optional[str] = None,
        article_ref: str = "",
        extra_props: Optional[dict] = None,
        source_type: str = "Organization",
        target_type: str = "Organization",
    ) -> None:
        if not source or not target or not relation_type:
            return
        label = self._label
        # Normalize source/target to canonical form so MATCH finds _upsert_node's MERGE keys
        source_canonical = normalize_entity_id(source, source_type)
        target_canonical = normalize_entity_id(target, target_type)
        # Build extra properties SET clause
        extra_props = extra_props or {}
        prop_sets: list[str] = []
        params: dict[str, Any] = {
            "src": source_canonical, "tgt": target_canonical,
            "desc": description, "doc_id": str(document_id) if document_id else None, "art_ref": article_ref,
        }
        for k, v in extra_props.items():
            safe_key = re.sub(r"\W", "_", k)
            prop_sets.append(f"r.{safe_key} = ${safe_key}")
            params[safe_key] = v

        extra_set = (", " + ", ".join(prop_sets)) if prop_sets else ""

        cypher = f"""
        MATCH (a:`{label}` {{entity_id: $src}})
        MATCH (b:`{label}` {{entity_id: $tgt}})
        MERGE (a)-[r:{relation_type}]->(b)
        ON CREATE SET r.document_ids = CASE WHEN $doc_id IS NULL THEN [] ELSE [$doc_id] END
        ON MATCH SET  r.document_ids = {_doc_ids_append("r", "doc_id")}
        SET r.description  = $desc,
            r.document_id  = $doc_id,
            r.article_ref  = $art_ref,
            r.updated_at   = datetime(){extra_set}
        RETURN count(r) AS c
        """
        result = await session.run(cypher, **params)
        rec = await result.single()
        if rec is None:
            # One/both endpoints had no node with that entity_id — usually a
            # type-mismatch in normalization. Log instead of dropping silently.
            logger.warning(
                f"LegalKG: relation {relation_type} dropped — endpoint node missing "
                f"(src='{source_canonical}' [{source_type}], tgt='{target_canonical}' [{target_type}])"
            )

    async def _store_extraction(
        self,
        session,
        data: dict,
        doc_name: str,
        document_id: Optional[str],
    ) -> None:
        """Upsert all entities and relations from one article's extraction result."""
        entity_map: dict[str, str] = {}  # canonical_id → entity_type
        # Also keep a reverse map: raw_name → canonical_id (for relation source/target lookup)
        canonical_lookup: dict[str, str] = {}

        for ent in data.get("entities", []):
            raw_name = str(ent.get("name", "")).strip()
            # Clean common markdown/list prefixes (e.g. "# ", "- ", "* ")
            raw_name = re.sub(r"^[#\*\- \t]+", "", raw_name).strip()
            
            etype = str(ent.get("type", "Organization")).strip()
            desc = str(ent.get("description", "")).strip()

            if not raw_name:
                continue

            # Coreference resolution: "Luật này", "Quyết định này" -> Current Document Root Node
            if raw_name.lower().endswith(" này") or raw_name.lower() == "này":
                raw_name = doc_name
                etype = "Document"
            # Force Document type for document numbers or legal titles
            elif _DOC_NUM_PATTERN.search(raw_name) or _LEGAL_DOC_PREFIXES.search(raw_name):
                etype = "Document"

            # Normalize Person composite key
            if etype == "Person":
                person_props = ent.get("person_props", {})
                raw_name = build_person_composite_key(raw_name, person_props)
                canonical = raw_name  # Person canonical = composite key as-is
            else:
                canonical = normalize_entity_id(raw_name, etype)

            canonical_lookup[raw_name] = canonical
            entity_map[canonical] = etype
            await self._upsert_node(session, raw_name, etype, desc, document_id)
            
            # Explicitly enforce PART_OF for all Article nodes
            if etype == "Article":
                await self._upsert_relation(
                    session, canonical, "PART_OF", doc_name,
                    "Thuộc văn bản", document_id,
                    source_type="Article", target_type="Document",
                )

        for rel in data.get("relations", []):
            source_raw = str(rel.get("source", "")).strip()
            source_raw = re.sub(r"^[#\*\- \t]+", "", source_raw).strip()
            
            relation_type = str(rel.get("relation", "")).strip().upper()
            
            target_raw = str(rel.get("target", "")).strip()
            target_raw = re.sub(r"^[#\*\- \t]+", "", target_raw).strip()
            
            desc = str(rel.get("description", "")).strip()
            art_ref = rel.get("article_ref", "")
            doc_id = str(rel.get("document_id")) if rel.get("document_id") else document_id
            person_props: dict = rel.get("person_props", {})

            if not source_raw or not target_raw or not relation_type:
                continue

            # Coreference resolution for relations: "Luật này" -> Current Document
            if source_raw.lower().endswith(" này") or source_raw.lower() == "này":
                source_raw = doc_name
                src_type = "Document"
            else:
                is_legal_doc = (
                    source_raw == doc_name or 
                    _DOC_NUM_PATTERN.search(source_raw) or 
                    _LEGAL_DOC_PREFIXES.search(source_raw)
                )
                src_type = entity_map.get(
                    canonical_lookup.get(source_raw, source_raw),
                    "Document" if is_legal_doc else "Organization",
                )

            if target_raw.lower().endswith(" này") or target_raw.lower() == "này":
                target_raw = doc_name
                tgt_type = "Document"
            else:
                is_legal_doc_target = (
                    target_raw == doc_name or
                    _DOC_NUM_PATTERN.search(target_raw) or
                    _LEGAL_DOC_PREFIXES.search(target_raw)
                )
                tgt_type = entity_map.get(
                    canonical_lookup.get(target_raw, target_raw),
                    "Document" if is_legal_doc_target else ("Person" if person_props else "Organization"),
                )

            # Normalize Person target composite key if person_props available
            if tgt_type == "Person" and person_props:
                target_raw = build_person_composite_key(target_raw, person_props)
                source_canonical = canonical_lookup.get(source_raw, normalize_entity_id(source_raw, src_type))
                target_canonical = target_raw  # Person canonical = composite key
            else:
                source_canonical = canonical_lookup.get(source_raw, normalize_entity_id(source_raw, src_type))
                target_canonical = canonical_lookup.get(target_raw, normalize_entity_id(target_raw, tgt_type))

            await self._upsert_node(session, source_raw, src_type, "", doc_id)
            await self._upsert_node(session, target_raw, tgt_type, "", doc_id)

            # Flatten and normalize person_props dates
            flat_props: dict = {}
            for k, v in person_props.items():
                if k in ("ngay_sinh", "ngay_hieu_luc") and v:
                    v = normalize_date(str(v))
                flat_props[k] = v

            await self._upsert_relation(
                session, source_canonical, relation_type, target_canonical,
                desc, doc_id, art_ref, flat_props,
                source_type=src_type, target_type=tgt_type,
            )

    async def _store_relations_from_extraction(
        self,
        session,
        data: dict,
        root_ref: str,
        document_id: Optional[str],
        canonical_lookup: dict[str, str],
        merged_map: dict[str, str],
        skipped_entities: set[str],
        entity_type_map: dict[str, str],
        doc_name: str = "",
        doc_title: str = "",
    ) -> None:
        """
        Upsert relations from one article's extraction result, using pre-resolved
        canonical entity names from the entity resolution pass.

        Edge handling:
          - dropped entities (in skipped_entities): relation is SKIPPED entirely
          - merged entities (in merged_map): source/target is rerouted to canonical
          - doc-root aliases (== doc_name / doc_title): folded onto the số-hiệu root
          - self-loops after resolution: SKIPPED
          - canonical entities: relation upserted normally
        """
        for rel in data.get("relations", []):
            source_raw = str(rel.get("source", "")).strip()
            source_raw = re.sub(r"^[#\*\- \t]+", "", source_raw).strip()

            relation_type = str(rel.get("relation", "")).strip().upper()

            target_raw = str(rel.get("target", "")).strip()
            target_raw = re.sub(r"^[#\*\- \t]+", "", target_raw).strip()

            desc = str(rel.get("description", "")).strip()
            art_ref = rel.get("article_ref", "")
            doc_id = str(rel.get("document_id")) if rel.get("document_id") else document_id
            person_props: dict = rel.get("person_props", {})

            if not source_raw or not target_raw or not relation_type:
                continue

            # --- Handle dropped entities ---
            # If source was explicitly dropped → skip this relation
            if source_raw in skipped_entities:
                logger.debug(f"LegalKG: skipping relation (source dropped): {source_raw}")
                continue
            # If target was explicitly dropped → skip this relation
            if target_raw in skipped_entities:
                logger.debug(f"LegalKG: skipping relation (target dropped): {target_raw}")
                continue

            # --- Resolve source ---
            # Coreference ("Luật này"…) or an explicit reference to THIS document
            # by its title / full name → Document Root.
            if (source_raw.lower().endswith(" này") or source_raw.lower() == "này"
                    or _is_doc_root_alias(source_raw, doc_name, doc_title)):
                source_canonical = root_ref
                src_type = "Document"
            else:
                # Check merged_map first (LLM-detected aliases)
                source_key = merged_map.get(source_raw, source_raw)
                src_canonical_key = canonical_lookup.get(source_key, source_key)
                # Fallback type for endpoints the resolution pass never saw:
                # classify by the raw name — laws/decrees → Document, clause refs
                # (Điều/Khoản/…) → Article, the doc root → Document — so they merge
                # onto the right canonical node instead of spawning a parallel
                # :Organization with the same entity_id.
                src_default = (
                    "Document" if source_raw == root_ref
                    else _force_legal_type(source_raw, "Organization")
                )
                src_type = entity_type_map.get(src_canonical_key, src_default)
                source_canonical = src_canonical_key

            # --- Resolve target ---
            if (target_raw.lower().endswith(" này") or target_raw.lower() == "này"
                    or _is_doc_root_alias(target_raw, doc_name, doc_title)):
                target_canonical = root_ref
                tgt_type = "Document"
            else:
                # Check merged_map first (LLM-detected aliases)
                target_key = merged_map.get(target_raw, target_raw)
                tgt_canonical_key = canonical_lookup.get(target_key, target_key)
                tgt_default = (
                    "Document" if target_raw == root_ref
                    else _force_legal_type(target_raw, "Organization")
                )
                tgt_type = entity_type_map.get(tgt_canonical_key, tgt_default)
                target_canonical = tgt_canonical_key

            # Person composite key for target
            if tgt_type == "Person" and person_props:
                target_canonical = build_person_composite_key(target_raw, person_props)
            # Person composite key for source
            if src_type == "Person" and person_props:
                source_canonical = build_person_composite_key(source_raw, person_props)

            # Flatten and normalize person_props dates
            flat_props: dict = {}
            for k, v in person_props.items():
                if k in ("ngay_sinh", "ngay_hieu_luc") and v:
                    v = normalize_date(str(v))
                flat_props[k] = v

            # Skip self-loops created when both endpoints fold onto the same node
            # (e.g. "53/2022 PART_OF <its own title>" after title→root folding).
            if (normalize_entity_id(source_canonical, src_type)
                    == normalize_entity_id(target_canonical, tgt_type)):
                logger.debug(
                    f"LegalKG: skipping self-loop {relation_type} on {source_canonical}"
                )
                continue

            # Drop relations whose endpoint is a generic/junk entity that only
            # ever appears here (never in the entity list, so premark/skip never
            # saw it). Without this, the endpoint-materialising upsert below
            # would re-create exactly the "Bộ" / "Cơ quan nhà nước" nodes the
            # premark guard removed.
            if (_is_generic_or_junk(source_canonical, src_type)
                    or _is_generic_or_junk(target_canonical, tgt_type)):
                logger.debug(
                    f"LegalKG: skipping relation with generic/junk endpoint: "
                    f"'{source_canonical}' / '{target_canonical}'"
                )
                continue

            # Ensure both endpoints exist so the relation is never silently
            # dropped when an entity was referenced in a relation but never
            # upserted as a standalone node (e.g. its type was resolved
            # differently, giving a different normalized entity_id).
            await self._upsert_node(session, source_canonical, src_type, "", doc_id)
            await self._upsert_node(session, target_canonical, tgt_type, "", doc_id)

            await self._upsert_relation(
                session, source_canonical, relation_type, target_canonical,
                desc, doc_id, art_ref, flat_props,
                source_type=src_type, target_type=tgt_type,
            )

    # ------------------------------------------------------------------
    # Query / RAG context retrieval
    # ------------------------------------------------------------------

    async def query(self, question: str, mode: str = "hybrid", top_k: int = 10) -> str:
        """Alias for get_relevant_context — returns formatted string for RAG."""
        return await self.get_relevant_context(question)

    async def get_relevant_context(
        self,
        question: str,
        max_entities: int = 20,
        max_relationships: int = 30,
    ) -> str:
        """
        Retrieve relevant KG context for a RAG query.
        Uses CONTAINS (case-insensitive) to handle Composite Keys and abbreviated names.
        """
        # Extract keywords from question
        tokens = re.split(r"[\s,\.;:!?]+", question.lower())
        keywords = [t for t in tokens if len(t) >= 2]
        if not keywords:
            return ""

        driver = await self._get_driver()
        label = self._label

        # Build Cypher WHERE with CONTAINS for all keywords (OR-joined)
        where_parts = []
        params: dict = {}
        for i, kw in enumerate(keywords[:10]):  # cap at 10 keywords
            p = f"kw{i}"
            where_parts.append(f"toLower(n.entity_id) CONTAINS ${p}")
            params[p] = kw

        where_clause = " OR ".join(where_parts)

        cypher = f"""
        MATCH (n:`{label}`)
        WHERE {where_clause}
        WITH n LIMIT {max_entities}
        OPTIONAL MATCH (n)-[r]-(m:`{label}`)
        RETURN
            n.entity_id     AS entity_name,
            n.entity_type   AS entity_type,
            n.description   AS entity_desc,
            type(r)          AS rel_type,
            r.description    AS rel_desc,
            startNode(r).entity_id AS rel_src,
            endNode(r).entity_id   AS rel_tgt
        LIMIT {max_entities + max_relationships}
        """

        entity_info: dict[str, dict] = {}
        rels: list[dict] = []

        try:
            async with driver.session() as session:
                result = await session.run(cypher, **params)
                records = await result.data()

            for rec in records:
                ename = rec.get("entity_name", "")
                if ename and ename not in entity_info:
                    entity_info[ename] = {
                        "entity_type": rec.get("entity_type", "Unknown"),
                        "description": rec.get("entity_desc", ""),
                    }
                src, tgt = rec.get("rel_src"), rec.get("rel_tgt")
                if src and tgt and len(rels) < max_relationships:
                    rels.append({
                        "source": src,
                        "target": tgt,
                        "relation": rec.get("rel_type", ""),
                        "description": rec.get("rel_desc", ""),
                    })
        except Exception as e:
            logger.error(f"LegalKG context retrieval failed for workspace {self.workspace_id}: {e}")
            return ""

        if not entity_info and not rels:
            return ""

        return self._format_context(entity_info, rels)

    # Quan hệ VĂN BẢN ↔ VĂN BẢN (khác quan hệ tổ chức/nhân sự)
    DOC_RELATION_TYPES = (
        "CAN_CU", "VIEN_DAN", "SUA_DOI", "THAY_THE", "BAI_BO", "REFERENCES"
    )

    async def get_document_relations(
        self,
        document_ref: str,
        relation_types: list[str] | None = None,
        limit: int = 30,
    ) -> list[dict]:
        """Traversal 1-hop các quan hệ văn bản từ node Document khớp ``document_ref``.

        ``document_ref``: số hiệu ("85/2016") hoặc tên văn bản — match CONTAINS
        (lowercase) trên entity_id, tối đa 3 node anchor.

        Returns: list các dict
            {anchor, source, target, relation, direction ('out'|'in'), description}
        """
        ref = (document_ref or "").strip().lower()
        if not ref:
            return []
        rels = list(relation_types or self.DOC_RELATION_TYPES)

        driver = await self._get_driver()
        label = self._label
        cypher = f"""
        MATCH (n:`{label}`)
        WHERE n.entity_type = 'Document' AND toLower(n.entity_id) CONTAINS $ref
        WITH n LIMIT 3
        MATCH (n)-[r]-(m:`{label}`)
        WHERE type(r) IN $rels
        RETURN DISTINCT
            n.entity_id            AS anchor,
            startNode(r).entity_id AS src,
            endNode(r).entity_id   AS tgt,
            type(r)                AS rel,
            r.description          AS desc
        LIMIT $limit
        """
        out: list[dict] = []
        try:
            async with driver.session() as session:
                result = await session.run(cypher, ref=ref, rels=rels, limit=limit)
                records = await result.data()
            for rec in records:
                anchor = rec.get("anchor") or ""
                src, tgt = rec.get("src") or "", rec.get("tgt") or ""
                if not (src and tgt):
                    continue
                out.append({
                    "anchor": anchor,
                    "source": src,
                    "target": tgt,
                    "relation": rec.get("rel") or "",
                    "direction": "out" if src == anchor else "in",
                    "description": rec.get("desc") or "",
                })
        except Exception as e:
            logger.error(
                f"LegalKG get_document_relations failed for workspace "
                f"{self.workspace_id} ref={document_ref!r}: {e}"
            )
        return out

    def _format_context(self, entity_info: dict, rels: list[dict]) -> str:
        lines = ["=== Kết quả từ Knowledge Graph ===\n"]
        if entity_info:
            lines.append("[ Thực thể liên quan ]")
            for name, info in entity_info.items():
                desc = f" — {info['description']}" if info.get("description") else ""
                lines.append(f"  • {name} [{info['entity_type']}]{desc}")
        if rels:
            lines.append("\n[ Mối quan hệ ]")
            for r in rels:
                desc = f": {r['description']}" if r.get("description") else ""
                lines.append(f"  • {r['source']} —[{r['relation']}]→ {r['target']}{desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Entity / Relationship / Graph Data APIs (drop-in compat)
    # ------------------------------------------------------------------

    async def get_entities(
        self,
        search: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        driver = await self._get_driver()
        label = self._label

        where_parts = [f"1=1"]
        params: dict = {}
        if search:
            where_parts.append("toLower(n.entity_id) CONTAINS toLower($search)")
            params["search"] = search
        if entity_type:
            where_parts.append("n.entity_type = $entity_type")
            params["entity_type"] = entity_type

        cypher = f"""
        MATCH (n:`{label}`)
        WHERE {' AND '.join(where_parts)}
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        RETURN n.entity_id AS name, n.entity_type AS entity_type,
               n.description AS description, degree
        ORDER BY degree DESC
        SKIP {offset} LIMIT {limit}
        """
        try:
            async with driver.session() as session:
                result = await session.run(cypher, **params)
                return await result.data()
        except Exception as e:
            logger.error(f"LegalKG get_entities failed: {e}")
            return []

    async def get_relationships(
        self,
        entity_name: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict]:
        driver = await self._get_driver()
        label = self._label

        if entity_name:
            cypher = f"""
            MATCH (a:`{label}`)-[r]->(b:`{label}`)
            WHERE toLower(a.entity_id) CONTAINS toLower($name)
               OR toLower(b.entity_id) CONTAINS toLower($name)
            RETURN a.entity_id AS source, type(r) AS relation,
                   b.entity_id AS target, r.description AS description,
                   coalesce(r.weight, 1.0) AS weight
            LIMIT {limit}
            """
            params = {"name": entity_name}
        else:
            cypher = f"""
            MATCH (a:`{label}`)-[r]->(b:`{label}`)
            RETURN a.entity_id AS source, type(r) AS relation,
                   b.entity_id AS target, r.description AS description,
                   coalesce(r.weight, 1.0) AS weight
            LIMIT {limit}
            """
            params = {}
        try:
            async with driver.session() as session:
                result = await session.run(cypher, **params)
                return await result.data()
        except Exception as e:
            logger.error(f"LegalKG get_relationships failed: {e}")
            return []

    async def get_graph_data(
        self,
        center_entity: Optional[str] = None,
        max_depth: int = 3,
        max_nodes: int = 150,
    ) -> dict:
        driver = await self._get_driver()
        label = self._label

        if center_entity:
            cypher = f"""
            MATCH (n:`{label}`)
            WHERE toLower(n.entity_id) CONTAINS toLower($center)
            WITH n LIMIT {max_nodes}
            OPTIONAL MATCH (n)-[r]-(m:`{label}`)
            RETURN n.entity_id AS entity_name, n.entity_type AS entity_type,
                   m.entity_id AS neighbor, m.entity_type AS neighbor_type,
                   r.description AS rel_desc,
                   startNode(r).entity_id AS rel_src,
                   endNode(r).entity_id AS rel_tgt
            LIMIT {max_nodes * 3}
            """
            params = {"center": center_entity}
        else:
            cypher = f"""
            MATCH (n:`{label}`)
            WITH n LIMIT {max_nodes}
            OPTIONAL MATCH (n)-[r]-(m:`{label}`)
            RETURN n.entity_id AS entity_name, n.entity_type AS entity_type,
                   m.entity_id AS neighbor, m.entity_type AS neighbor_type,
                   r.description AS rel_desc,
                   startNode(r).entity_id AS rel_src,
                   endNode(r).entity_id AS rel_tgt
            LIMIT {max_nodes * 3}
            """
            params = {}

        try:
            async with driver.session() as session:
                result = await session.run(cypher, **params)
                records = await result.data()
        except Exception as e:
            logger.error(f"LegalKG get_graph_data failed: {e}")
            return {"nodes": [], "edges": [], "is_truncated": False}

        seen_nodes: dict[str, str] = {}
        seen_edges: set = set()
        edges_out: list[dict] = []

        for rec in records:
            if rec.get("entity_name"):
                seen_nodes.setdefault(rec["entity_name"], rec.get("entity_type", "Unknown"))
            if rec.get("neighbor"):
                seen_nodes.setdefault(rec["neighbor"], rec.get("neighbor_type", "Unknown"))
            src, tgt = rec.get("rel_src"), rec.get("rel_tgt")
            if src and tgt and (src, tgt) not in seen_edges:
                seen_edges.add((src, tgt))
                edges_out.append({
                    "source": src,
                    "target": tgt,
                    "label": str(rec.get("rel_desc") or "")[:80],
                    "weight": 1.0,
                })

        degree_map: dict[str, int] = {}
        for e in edges_out:
            degree_map[e["source"]] = degree_map.get(e["source"], 0) + 1
            degree_map[e["target"]] = degree_map.get(e["target"], 0) + 1

        nodes_out = [
            {"id": name, "label": name, "entity_type": etype,
             "degree": degree_map.get(name, 0)}
            for name, etype in seen_nodes.items()
        ]

        return {
            "nodes": nodes_out,
            "edges": edges_out,
            "is_truncated": len(seen_nodes) >= max_nodes,
        }

    async def get_analytics(self) -> dict:
        driver = await self._get_driver()
        label = self._label
        try:
            async with driver.session() as session:
                r1 = await session.run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
                entity_count = (await r1.single() or {}).get("cnt", 0)

                r2 = await session.run(f"MATCH (:`{label}`)-[r]->(:`{label}`) RETURN count(r) AS cnt")
                rel_count = (await r2.single() or {}).get("cnt", 0)

                r3 = await session.run(
                    f"MATCH (n:`{label}`) RETURN n.entity_type AS t, count(*) AS c"
                )
                type_counts = {rec["t"]: rec["c"] for rec in await r3.data()}

                r4 = await session.run(
                    f"""MATCH (n:`{label}`)
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, count(r) AS degree
                    ORDER BY degree DESC LIMIT 10
                    RETURN n.entity_id AS name, n.entity_type AS entity_type, degree"""
                )
                top_entities = await r4.data()
        except Exception as e:
            logger.error(f"LegalKG analytics failed: {e}")
            return {"entity_count": 0, "relationship_count": 0, "entity_types": {}, "top_entities": [], "avg_degree": 0.0}

        return {
            "entity_count": entity_count,
            "relationship_count": rel_count,
            "entity_types": type_counts,
            "top_entities": top_entities,
            "avg_degree": round(rel_count / entity_count, 2) if entity_count else 0.0,
        }

    async def delete_project_data(self) -> None:
        """Delete all KG data for this workspace from Neo4j."""
        driver = await self._get_driver()
        label = self._label
        try:
            async with driver.session() as session:
                result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                summary = await result.consume()
                logger.info(
                    f"LegalKG deleted {summary.counters.nodes_deleted} nodes, "
                    f"{summary.counters.relationships_deleted} rels "
                    f"for workspace {self.workspace_id}"
                )
        except Exception as e:
            logger.error(f"LegalKG delete_project_data failed: {e}")

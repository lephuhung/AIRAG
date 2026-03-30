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
from datetime import datetime
from typing import Any, Optional

from app.core.config import settings
from app.services.llm import get_llm_provider
from app.services.llm.types import LLMMessage
from app.services.legal_kg_prompts import (
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
_LLM_SEMAPHORE = asyncio.Semaphore(4)

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


# Pattern for Vietnamese legal document numbers (e.g., 29/2018/QH14, 01/2023/ND-CP)
_DOC_NUM_PATTERN = re.compile(r"\d+/\d+/[A-Z0-9-]+", re.IGNORECASE)

# Keywords that indicate an entity MUST be a Document type
_LEGAL_DOC_PREFIXES = re.compile(
    r"^(Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Chỉ thị|Nghị quyết|Hiến pháp|Pháp lệnh)\b",
    re.IGNORECASE
)


def normalize_entity_id(name: str, entity_type: str) -> str:
    """
    Return canonical entity_id for MERGE key:
    - Organization, Document, Article, Location → normalize_org_name (case-folded canonical)
    - Person → unchanged (Person uses composite key for disambiguation)
    - Task → whitespace-normalized only
    """
    name = " ".join(name.strip().split())  # strip + collapse spaces
    if entity_type in ("Organization", "Document", "Article", "Location"):
        return normalize_org_name(name)
    return name


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
    from app.services.deep_document_parser import _fix_scattered_vietnamese
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
) -> str:
    """Call LLM with exponential-backoff retry for rate limits."""
    provider = get_llm_provider()
    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]
    for attempt in range(4):
        try:
            async with _LLM_SEMAPHORE:
                return await provider.acomplete(messages, temperature=0.0, max_tokens=max_tokens)
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

    def __init__(self, workspace_id: int):
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
    # Ingestion pipeline
    # ------------------------------------------------------------------

    async def ingest(self, markdown_content: str, document_id: Optional[int] = None) -> None:
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

        # Step 1: Document metadata
        doc_meta = parse_document_meta(markdown_content)
        doc_name = doc_meta.get("document_name", "Văn bản không xác định")
        
        # --- Fetch Rich Header Metadata from Database ---
        doc_type = ""
        doc_num = ""
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

        is_personnel = is_personnel_document(markdown_content)

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

        # Step 4: LLM extraction (concurrent, semaphore-controlled)
        doc_meta_str = self._format_doc_meta(doc_meta)
        tasks = [
            self._extract_with_llm(article, doc_meta_str, doc_name, is_personnel, document_id)
            for article in articles
        ]
        article_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 5: Store all results
        driver = await self._get_driver()
        async with driver.session() as session:
            # Store Document node
            await self._upsert_node(session, doc_name, "Document", doc_meta.get("so_hieu", ""), document_id)
            
            # --- Store Root Context (Location & Org Hierarchy) ---
            loc_name = ""
            if loc:
                loc_name = loc if "tỉnh" in loc.lower() or "thành phố" in loc.lower() else f"Tỉnh {loc}"
                await self._upsert_node(session, loc_name, "Location", "", document_id)
                await self._upsert_relation(
                    session, doc_name, "BAN_HANH_TAI", loc_name, "Phạm vi địa lý", document_id,
                    source_type="Document", target_type="Location",
                )
                
            if parent_org and issue_org:
                combined_issue_org = f"{issue_org} {parent_org}"
                await self._upsert_node(session, combined_issue_org, "Organization", "", document_id)
                await self._upsert_node(session, parent_org, "Organization", "", document_id)
                
                await self._upsert_relation(
                    session, doc_name, "BAN_HANH_BOI", combined_issue_org, "", document_id,
                    source_type="Document", target_type="Organization",
                )
                await self._upsert_relation(
                    session, combined_issue_org, "TRUC_THUOC", parent_org, "", document_id,
                    source_type="Organization", target_type="Organization",
                )
                
                if loc_name:
                    await self._upsert_relation(
                        session, parent_org, "THUOC_TINH", loc_name, "", document_id,
                        source_type="Organization", target_type="Location",
                    )
            elif issue_org:
                await self._upsert_node(session, issue_org, "Organization", "", document_id)
                await self._upsert_relation(
                    session, doc_name, "BAN_HANH_BOI", issue_org, "", document_id,
                    source_type="Document", target_type="Organization",
                )
            elif parent_org:
                # Only parent org, no issuing org (e.g., Quốc Hội)
                await self._upsert_node(session, parent_org, "Organization", "", document_id)
                await self._upsert_relation(
                    session, doc_name, "BAN_HANH_BOI", parent_org, "", document_id,
                    source_type="Document", target_type="Organization",
                )
                if loc_name:
                    await self._upsert_relation(
                        session, parent_org, "THUOC_TINH", loc_name, "", document_id,
                        source_type="Organization", target_type="Location",
                    )


            # Store CAN_CU relations from preamble
            for ref_doc in can_cu_list:
                await self._upsert_node(session, ref_doc, "Document", ref_doc, document_id)
                await self._upsert_relation(
                    session, doc_name, "CAN_CU", ref_doc,
                    f"Căn cứ pháp lý: {ref_doc}", document_id,
                    source_type="Document", target_type="Document",
                )

            # Store article extraction results
            for i, result in enumerate(article_results):
                if isinstance(result, Exception):
                    logger.warning(f"LegalKG: article {i} extraction failed: {result}")
                    continue
                if not result:
                    continue
                await self._store_extraction(session, result, doc_name, document_id)

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

    def _format_doc_meta(self, meta: dict) -> str:
        parts = []
        if meta.get("so_hieu"):
            parts.append(f"Số hiệu: {meta['so_hieu']}")
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
            raw = await _call_llm(PREAMBLE_SYSTEM_PROMPT, user_prompt, max_tokens=1024)
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
        document_id: Optional[int],
    ) -> dict:
        """Run LLM extraction on a single article chunk."""
        text = article["text"]
        heading = article["heading"]
        article_ref = f"Điều {article['index']}"

        # Choose prompt variant
        if is_personnel:
            system_prompt = PERSON_EXTRACT_SYSTEM_PROMPT
            user_prompt = PERSON_EXTRACT_USER_PROMPT.format(
                document_meta=doc_meta_str,
                article_text=text[:3000],
            )
        else:
            system_prompt = LEGAL_KG_SYSTEM_PROMPT
            user_prompt = LEGAL_KG_USER_PROMPT.format(
                document_meta=doc_meta_str,
                article_text=text[:3000],
            )

        try:
            raw = await _call_llm(system_prompt, user_prompt)
            data = _parse_llm_json(raw)
            # Tag each entity/relation with its source article
            for e in data.get("entities", []):
                e.setdefault("article_ref", article_ref)
                e.setdefault("document_id", document_id)
            for r in data.get("relations", []):
                r.setdefault("article_ref", article_ref)
                r.setdefault("document_id", document_id)
            return data
        except Exception as e:
            logger.warning(f"LegalKG LLM extraction failed for {heading}: {e}")
            return {"entities": [], "relations": []}

    # ------------------------------------------------------------------
    # Neo4j storage helpers
    # ------------------------------------------------------------------

    async def _upsert_node(
        self,
        session,
        entity_id: str,
        entity_type: str,
        description: str = "",
        document_id: Optional[int] = None,
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
                      n.created_at   = datetime()
        ON MATCH SET  n.description  = CASE WHEN $description <> '' THEN $description ELSE n.description END
        """
        await session.run(
            cypher,
            entity_id=canonical_id,
            entity_type=entity_type,
            display_name=display_name,
            description=description,
            document_id=document_id,
        )
        return canonical_id  # caller may need canonical id for relation lookup

    async def _upsert_relation(
        self,
        session,
        source: str,
        relation_type: str,
        target: str,
        description: str = "",
        document_id: Optional[int] = None,
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
        prop_sets = []
        params: dict[str, Any] = {
            "src": source_canonical, "tgt": target_canonical,
            "desc": description, "doc_id": document_id, "art_ref": article_ref,
        }
        for k, v in extra_props.items():
            safe_key = re.sub(r"\W", "_", k)
            prop_sets.append(f"r.{safe_key} = ${safe_key}")
            params[safe_key] = v

        extra_set = ("SET " + ", ".join(prop_sets)) if prop_sets else ""

        cypher = f"""
        MATCH (a:`{label}` {{entity_id: $src}})
        MATCH (b:`{label}` {{entity_id: $tgt}})
        MERGE (a)-[r:{relation_type}]->(b)
        SET r.description = $desc,
            r.document_id = $doc_id,
            r.article_ref = $art_ref,
            r.updated_at  = datetime()
        {extra_set}
        """
        await session.run(cypher, **params)

    async def _store_extraction(
        self,
        session,
        data: dict,
        doc_name: str,
        document_id: Optional[int],
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
            doc_id = rel.get("document_id", document_id)
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

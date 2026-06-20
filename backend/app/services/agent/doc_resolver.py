"""
Document Resolver Core
======================

Framework-agnostic core for resolving an ambiguous document reference
(e.g. "Luật An ninh mạng 2025", "Thông tư 15 của Bộ Công an") to candidate
documents — shared by BOTH agent entry points so the logic never diverges:

- ``app/services/agents/resolve_doc_agent.py`` (LangGraph supervisor static path)
  wraps :func:`resolve_candidates` and adds SupervisorState routing / SSE.
- ``app/services/agent/tools.py`` :: ``resolve_document_reference`` (ReAct path)
  wraps :func:`resolve_candidates` and maps to its legacy ``candidates`` dict.

Tiered strategy (performance-optimized, ~2-4s):
  Stage 0  Pure regex extraction (<1ms) → SQL query
  Stage 1  Memory-agent LLM extraction (gemma-4-E4B, ~1-2s) when the DB query is dry
  Stage 2  Vector search fallback when DB+LLM are dry
  Stage 3  Fuzzy similar-title search when everything is dry

This module holds ONLY pure resolution logic — no SupervisorState, no AgentType,
no SSE push_event. Callers decide how to route the returned candidates. Status
updates are surfaced via the optional ``status_cb`` callback.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Optional async callback used to stream human-readable progress to the UI.
StatusCallback = Callable[[str], Awaitable[None]]

# Vector candidates are down-weighted vs DB hits, but NOT so aggressively that a
# genuine semantic match can never clear the MEDIUM scoping threshold (0.30) in
# resolve_doc_agent. (Was 0.6 with a 0.4 default → max ~0.24, i.e. never scoped.)
_VECTOR_SCORE_FACTOR = 0.8
_VECTOR_DEFAULT_SCORE = 0.5


def _format_doc_title(doc_title: str | None, original_filename: str | None) -> str:
    """
    Format document title for display.

    If doc_title is set and non-empty → use it.
    Otherwise → format original_filename into readable title:
      - Strip file extension (.pdf, .docx, etc.)
      - Replace underscores/hyphens with spaces
      - Title-case each word
      - Handle common patterns like "luat117_2025" → "Luật 117/2025"
    """
    if doc_title and doc_title.strip():
        return doc_title.strip()

    if not original_filename:
        return "Văn bản không tên"

    # Strip extension
    import os
    name = os.path.splitext(original_filename)[0]

    # Handle patterns like "luat117_2025", "nd13_2023", "tt15_2024"
    # Extract number and year if present
    num_match = re.search(r'(luat|nd|tt|nq|qd|pl|bl)[_\s]*(\d+)', name, re.IGNORECASE)
    year_match = re.search(r'(20\d{2})', name)

    if num_match and year_match:
        doc_type = num_match.group(1).upper()
        num = num_match.group(2)
        year = year_match.group(1)
        type_map = {"LUAT": "Luật", "ND": "Nghị định", "TT": "Thông tư",
                    "NQ": "Nghị quyết", "QD": "Quyết định", "PL": "Pháp lệnh", "BL": "Bộ luật"}
        type_str = type_map.get(doc_type, doc_type)
        return f"{type_str} {num}/{year}"

    # Fallback: replace separators with spaces, title case
    name = re.sub(r'[_\-]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    if len(name) > 2:
        name = name.title()

    return name if name else "Văn bản không tên"


# ---------------------------------------------------------------------------
# Document type keyword → slug mapping (longest match first)
# ---------------------------------------------------------------------------
_DOC_TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("thông tư liên tịch", "thong_tu_lien_tich"),
    ("bộ luật",            "bo_luat"),
    ("nghị quyết",         "nghi_quyet"),
    ("nghị định",          "nghi_dinh"),
    ("quyết định",         "quyet_dinh"),
    ("pháp lệnh",          "phap_lenh"),
    ("thông tư",           "thong_tu"),
    ("chỉ thị",            "chi_thi"),
    ("luật",               "luat"),
]

# ---------------------------------------------------------------------------
# Issuing agency → (code, doc-number suffixes) — sorted longest name first
# ---------------------------------------------------------------------------
_ISSUING_AGENCY_MAP: list[tuple[str, str, list[str]]] = [
    ("bộ giáo dục và đào tạo",              "BGDĐT",   ["TT-BGDĐT"]),
    ("bộ nông nghiệp và phát triển nông thôn", "BNNPTNT", ["TT-BNNPTNT"]),
    ("bộ thông tin và truyền thông",         "BTTTT",   ["TT-BTTTT"]),
    ("bộ lao động thương binh và xã hội",    "BLĐTBXH", ["TT-BLĐTBXH"]),
    ("bộ tài nguyên và môi trường",          "BTNMT",   ["TT-BTNMT"]),
    ("bộ kế hoạch và đầu tư",               "BKHĐT",   ["TT-BKHĐT"]),
    ("bộ khoa học và công nghệ",             "BKHCN",   ["TT-BKHCN"]),
    ("bộ văn hóa thể thao và du lịch",      "BVHTTDL", ["TT-BVHTTDL"]),
    ("bộ văn hóa thể thao",                 "BVHTTDL", ["TT-BVHTTDL"]),
    ("bộ giao thông vận tải",               "BGTVT",   ["TT-BGTVT"]),
    ("ngân hàng nhà nước",                  "NHNN",    ["TT-NHNN"]),
    ("bộ công thương",                      "BCT",     ["TT-BCT", "VBHN-BCT"]),
    ("bộ công an",                          "BCA",     ["TT-BCA"]),
    ("bộ tài chính",                        "BTC",     ["TT-BTC"]),
    ("bộ tư pháp",                          "BTP",     ["TT-BTP"]),
    ("bộ quốc phòng",                       "BQP",     ["TT-BQP"]),
    ("bộ y tế",                             "BYT",     ["TT-BYT"]),
    ("bộ nội vụ",                           "BNV",     ["TT-BNV"]),
    ("bộ xây dựng",                         "BXD",     ["TT-BXD"]),
    ("bộ ngoại giao",                       "BNG",     ["TT-BNG"]),
    ("ủy ban thường vụ quốc hội",           "UBTVQH15", ["UBTVQH15", "NQ-UBTVQH15"]),
    ("thủ tướng chính phủ",                 "TTg",     ["QĐ-TTg"]),
    ("thủ tướng",                           "TTg",     ["QĐ-TTg"]),
    ("chính phủ",                            "CP",      ["NĐ-CP", "NQ-CP"]),
    ("quốc hội",                            "QH15",    ["QH15", "QH14"]),
    ("chủ tịch nước",                       "CTN",     ["L-CTN"]),
    ("viện kiểm sát nhân dân tối cao",      "VKSNDTC", ["QĐ-VKSNDTC"]),
    ("tòa án nhân dân tối cao",             "TANDTC",  ["QĐ-TANDTC"]),
    ("hội đồng nhân dân",                   "HĐND",    ["NQ-HĐND"]),
    ("ủy ban nhân dân",                     "UBND",    ["QĐ-UBND"]),
]

# Stopwords to exclude from title_keywords
_LEGAL_STOPWORDS = {
    "của", "về", "và", "các", "theo", "trong", "đến", "từ", "tới",
    "là", "có", "được", "này", "đó", "cho", "tôi", "xem", "tìm",
    "tra", "cứu", "hỏi", "văn", "bản", "tài", "liệu", "nội", "dung",
    "số", "năm", "ngày", "tháng", "ban", "hành", "quy", "định",
    "do", "bởi", "với", "một", "hai", "hay", "hoặc", "đây",
}

# Action phrases to strip before parsing (not part of doc reference)
_ACTION_PATTERNS = [
    r'^(?:tóm\s*tắt|tra\s*cứu|tìm|xem|liệt\s*kê|tổng\s*hợp|nội\s*dung)\s+',
    r'^(?:cho\s+tôi\s+xem|hiển\s+thị|lấy)\s+',
]

# Section reference patterns (Điều, Chương, Khoản, Mục, Phụ lục)
_SECTION_PATTERNS = [
    r'(?:điều|chương|khoản|mục|phụ\s*lục)\s+[\dIVXivx]+(?:\.\d+)*',
]

# Vietnam time — document numbers are issued/queried in local (GMT+7) calendar
# years, so derive "current year" from VN time, not the container's UTC clock.
_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


# =============================================================================
# Similar Documents Search (fuzzy title matching)
# =============================================================================

async def _search_similar_documents(
    reference: str,
    workspace_ids: list,
    db,
) -> list[dict]:
    """
    When no exact match is found, search for documents with similar titles.
    Uses keyword extraction + fuzzy ILIKE matching on document_title and original_filename.
    Returns top 5 similar documents sorted by keyword match score.
    """
    try:
        from sqlalchemy import select, or_
        from app.models.document import Document, DocumentStatus

        # Extract meaningful keywords from the reference query
        text = reference.strip()
        for pat in _ACTION_PATTERNS:
            text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

        # Remove section references
        for spat in _SECTION_PATTERNS:
            text = re.sub(spat, "", text, flags=re.IGNORECASE).strip()

        # Extract year to use as a filter (strong signal)
        year_match = re.search(r'\b((?:19|20)\d{2})\b', text)
        year = year_match.group(1) if year_match else None
        if year_match:
            text = (text[:year_match.start()] + text[year_match.end():]).strip()

        # Remove document type keywords to isolate the title part
        for keyword, _ in _DOC_TYPE_KEYWORDS:
            text = re.sub(re.escape(keyword), "", text, count=1, flags=re.IGNORECASE).strip()

        # Extract keywords (keep meaningful 2-char tokens; stopwords filter noise)
        keywords = [
            t for t in text.split()
            if len(t) >= 2 and t.lower() not in _LEGAL_STOPWORDS
        ]

        if not keywords:
            return []

        # Build fuzzy query on document titles
        query = select(Document).where(
            Document.workspace_id.in_(workspace_ids),
            Document.status == DocumentStatus.INDEXED,
        )

        # Build OR conditions for each keyword (ILIKE partial match)
        keyword_conditions = []
        for kw in keywords[:6]:  # Limit to 6 keywords to avoid over-filtering
            kw_lower = kw.lower()
            keyword_conditions.append(
                or_(
                    Document.document_title.ilike(f"%{kw_lower}%"),
                    Document.document_title.ilike(f"%{kw}%"),  # original case too
                    Document.original_filename.ilike(f"%{kw_lower}%"),
                    Document.original_filename.ilike(f"%{kw}%"),
                )
            )

        if keyword_conditions:
            query = query.where(or_(*keyword_conditions))

        # Boost by year if available
        if year:
            query = query.where(Document.published_date.ilike(f"%{year}%"))

        query = query.limit(10)
        result = await db.execute(query)
        docs = result.scalars().all()

        # Score by how many keywords matched
        scored: list[dict] = []
        for doc in docs:
            title_lower = (doc.document_title or "").lower()
            filename_lower = (doc.original_filename or "").lower()
            matched = 0
            matched_keywords = []
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in title_lower or kw_lower in filename_lower:
                    matched += 1
                    matched_keywords.append(kw)

            if matched > 0:
                score = matched / len(keywords)  # Ratio of matched keywords
                scored.append({
                    "document_id": str(doc.id),
                    "title": _format_doc_title(doc.document_title, doc.original_filename),
                    "document_number": doc.document_number or "",
                    "published_date": doc.published_date or "",
                    "score": score,
                    "matched_keywords": matched_keywords,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        logger.info(
            f"[resolve_doc/similar] reference={reference[:80]!r} → "
            f"keywords={keywords!r}, year={year!r}, found {len(scored)} similar documents"
        )
        for s in scored[:5]:
            logger.info(
                f"[resolve_doc/similar]   - {s['title'][:60]!r} "
                f"(matched: {s['matched_keywords']}, score={s['score']:.2f})"
            )
        return scored[:5]

    except Exception as e:
        logger.warning(f"[resolve_doc/similar] search failed: {e}")
        return []


# =============================================================================
# Helpers
# =============================================================================

def _generate_number_candidates(
    number_raw: str,
    suffixes: list[str],
    year: str | None,
) -> list[str]:
    """
    Build candidate document-number strings to OR-search in DB.
    E.g. number_raw="15", suffixes=["TT-BCA"], year=None
    → ["15/2026/TT-BCA", "15/2025/TT-BCA", "15/TT-BCA"]
    """
    current_year = _dt.datetime.now(_VN_TZ).year
    candidates: list[str] = []
    for suffix in suffixes:
        if year:
            candidates.append(f"{number_raw}/{year}/{suffix}")
        else:
            candidates.append(f"{number_raw}/{current_year}/{suffix}")
            candidates.append(f"{number_raw}/{current_year - 1}/{suffix}")
        candidates.append(f"{number_raw}/{suffix}")
    # De-duplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _extract_title_keywords(text: str) -> list[str]:
    """Return meaningful tokens from leftover text for document_title search.

    Keeps 2-char tokens (e.g. "An" in "An ninh mạng", "xã" in "xã hội"); noise
    2-char function words are filtered by _LEGAL_STOPWORDS instead of by length.
    """
    return [
        t for t in text.split()
        if len(t) >= 2 and t.lower() not in _LEGAL_STOPWORDS
    ]


# =============================================================================
# Stage 0: Fast Regex Extraction (no LLM, <1ms)
# =============================================================================

def _extract_by_regex(reference: str) -> dict:
    """
    Extract structured document metadata using regex only.

    Returns dict with keys:
        doc_type_slug, document_number, doc_number_candidates,
        title_keywords, issuing_agency_text, issuing_agency_code,
        year, section_reference, confidence
    """
    text = reference.strip()

    # Strip action phrases
    for pat in _ACTION_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    result: dict = {
        "doc_type_slug": None,
        "document_number": None,
        "doc_number_candidates": [],
        "title_keywords": [],
        "issuing_agency_text": None,
        "issuing_agency_code": None,
        "year": None,
        "section_reference": None,
        "confidence": "low",
    }

    # ── Extract section_reference (Điều X, Chương Y…) ────────────────────────
    for spat in _SECTION_PATTERNS:
        sm = re.search(spat, text, re.IGNORECASE)
        if sm:
            result["section_reference"] = sm.group().strip()
            text = (text[:sm.start()] + " " + text[sm.end():]).strip()
            break

    # ── Extract explicit document number (53/2022/NĐ-CP, 23/TT-BCA, số 361) ─
    doc_num_patterns = [
        r'\b(\d+/\d{4}/[\w\-]+)\b',    # full: 53/2022/NĐ-CP
        r'\b(\d+/[\w\-]{3,})\b',        # short: 23/TT-BCA (suffix ≥3 chars)
        r'\bsố\s+(\d+(?:[./]\d+)*)\b',  # số 361, số 53/2022
    ]
    number_raw: str | None = None
    for dnp in doc_num_patterns:
        dnm = re.search(dnp, text, re.IGNORECASE)
        if dnm:
            result["document_number"] = dnm.group(1) if dnm.lastindex else dnm.group()
            text = (text[:dnm.start()] + text[dnm.end():]).strip()
            # Try to isolate just the numeric part for candidate generation
            m_raw = re.match(r'^(\d+)', result["document_number"])
            if m_raw:
                number_raw = m_raw.group(1)
            break
    # ── Extract year BEFORE the bare-number fallback ─────────────────────────
    # A standalone 4-digit year (e.g. "Luật An ninh mạng 2018") must NOT be
    # mistaken for a document number. Explicit numbers like "53/2022/NĐ-CP" were
    # already consumed above, so this only removes a true publication year.
    ym = re.search(r'\b((?:19|20)\d{2})\b', text)
    if ym:
        result["year"] = ym.group(1)
        text = (text[:ym.start()] + text[ym.end():]).strip()

    # Also capture a bare number after stripping the year (e.g. "TT 15 BCA")
    if not number_raw:
        bare = re.search(r'\b(\d{1,4})\b', text)
        if bare:
            number_raw = bare.group(1)
            text = (text[:bare.start()] + text[bare.end():]).strip()

    # ── Detect document type ──────────────────────────────────────────────────
    for keyword, slug in _DOC_TYPE_KEYWORDS:
        if keyword.lower() in text.lower():
            result["doc_type_slug"] = slug
            text = re.sub(re.escape(keyword), "", text, count=1, flags=re.IGNORECASE).strip()
            break

    # ── Detect issuing agency ─────────────────────────────────────────────────
    agency_suffixes: list[str] = []
    text_lower = text.lower()
    for agency_name, code, suffixes in _ISSUING_AGENCY_MAP:
        if agency_name in text_lower:
            result["issuing_agency_text"] = agency_name.title()
            result["issuing_agency_code"] = code
            agency_suffixes = suffixes
            text = re.sub(re.escape(agency_name), "", text, count=1, flags=re.IGNORECASE).strip()
            # Strip connector words like "của", "do"
            text = re.sub(r'\b(?:của|do|bởi)\b', "", text, flags=re.IGNORECASE).strip()
            break

    # (year already extracted above, before the bare-number fallback)
    text = re.sub(r'\s+', ' ', text).strip()

    # ── Generate doc_number_candidates ───────────────────────────────────────
    if number_raw and agency_suffixes:
        result["doc_number_candidates"] = _generate_number_candidates(
            number_raw, agency_suffixes, result["year"]
        )
    elif number_raw and result["doc_type_slug"]:
        # Infer suffix from doc_type when no agency known
        _type_default_suffix: dict[str, list[str]] = {
            "luat":              ["QH15", "QH14"],
            "nghi_dinh":        ["NĐ-CP"],
            "nghi_quyet":       ["NQ-CP", "NQ-HĐND"],
            "thong_tu":         ["TT"],
            "thong_tu_lien_tich": ["TTLT"],
            "quyet_dinh":       ["QĐ-TTg", "QĐ-UBND"],
            "phap_lenh":        ["UBTVQH15"],
        }
        default_sfx = _type_default_suffix.get(result["doc_type_slug"], [])
        if default_sfx:
            result["doc_number_candidates"] = _generate_number_candidates(
                number_raw, default_sfx, result["year"]
            )
    # If user typed a full explicit number, prepend it as highest-priority candidate
    if result["document_number"] and result["document_number"] not in result["doc_number_candidates"]:
        result["doc_number_candidates"].insert(0, result["document_number"])

    # Preserve the bare leading number (identity signal) for downstream filtering
    result["number_raw"] = number_raw

    # ── Extract title keywords from remaining text (Dạng B) ──────────────────
    title_kws = _extract_title_keywords(text)
    result["title_keywords"] = title_kws
    # Keep backward-compat field
    result["document_title"] = " ".join(title_kws) if title_kws else None

    # ── Confidence assessment ─────────────────────────────────────────────────
    has_candidates = bool(result["doc_number_candidates"])
    has_num = bool(result["document_number"])
    has_type = bool(result["doc_type_slug"])
    has_agency = bool(result["issuing_agency_text"])
    has_title_kw = bool(title_kws)

    if has_num and has_type:
        result["confidence"] = "high"
    elif has_candidates and (has_agency or has_type):
        result["confidence"] = "high"
    elif has_candidates or (has_type and has_title_kw):
        result["confidence"] = "medium"
    elif has_title_kw:
        result["confidence"] = "medium"
    else:
        result["confidence"] = "low"

    logger.info(
        f"[resolve_doc/regex] confidence={result['confidence']}, "
        f"type={result['doc_type_slug']!r}, num={result['document_number']!r}, "
        f"candidates={result['doc_number_candidates']}, "
        f"agency={result['issuing_agency_text']!r}, "
        f"title_kw={result['title_keywords']}, year={result['year']!r}"
    )
    return result


# =============================================================================
# Stage 1: DB Query (fast SQL, no LLM)
# =============================================================================

async def _query_db(
    parsed: dict,
    workspace_ids: list,
    db,
) -> list[dict]:
    """
    Execute SQL query from parsed reference metadata.
    Searches: document_type, document_number (candidates OR), issuing_agency,
              document_title (title_keywords AND), published_date (year).
    """
    try:
        from sqlalchemy import select, or_
        from app.models.document import Document, DocumentStatus
        from app.models.document_type import DocumentType

        doc_type_slug         = parsed.get("doc_type_slug")
        document_number       = parsed.get("document_number")
        doc_number_candidates = parsed.get("doc_number_candidates") or []
        title_keywords        = parsed.get("title_keywords") or []
        issuing_agency_text   = parsed.get("issuing_agency_text")
        year                  = parsed.get("year")

        query = select(Document).where(
            Document.workspace_id.in_(workspace_ids),
            Document.status == DocumentStatus.INDEXED,
        )

        # ── 1. Document type (join) ────────────────────────────────────────
        if doc_type_slug:
            query = query.join(
                DocumentType, Document.document_type_id == DocumentType.id
            ).where(DocumentType.slug == doc_type_slug)

        # ── 2. Document number — OR over all candidates ───────────────────
        if doc_number_candidates:
            num_conds = []
            for c in doc_number_candidates:
                num_conds.extend([
                    Document.document_number == c,
                    Document.document_number.ilike(f"%{c}%"),
                ])
            query = query.where(or_(*num_conds))
        elif document_number:
            norm = document_number.strip()
            query = query.where(or_(
                Document.document_number == norm,
                Document.document_number.ilike(f"%{norm}%"),
            ))

        # ── 3. Issuing agency ─────────────────────────────────────────────
        if issuing_agency_text:
            agency_tokens = [t for t in issuing_agency_text.split() if len(t) > 1]
            if agency_tokens:
                agency_conds = []
                for tok in agency_tokens[:3]:
                    agency_conds.extend([
                        Document.issuing_agency.ilike(f"%{tok}%"),
                        Document.parent_agency.ilike(f"%{tok}%"),
                    ])
                query = query.where(or_(*agency_conds))

        # ── 4. Title keywords — OR (any keyword match, Dạng B: nhớ tên) ──
        # Use OR so documents matching ANY keyword (not all) are returned.
        if title_keywords and not doc_number_candidates:
            keyword_conds = []
            for kw in title_keywords[:6]:
                keyword_conds.append(or_(
                    Document.document_title.ilike(f"%{kw}%"),
                    Document.original_filename.ilike(f"%{kw}%"),
                ))
            query = query.where(or_(*keyword_conds))

        # ── 5. Year ───────────────────────────────────────────────────────
        if year and not doc_number_candidates:
            query = query.where(Document.published_date.ilike(f"%{year}%"))

        query = query.limit(10)
        result = await db.execute(query)
        docs = result.scalars().all()

        # ── Scoring ───────────────────────────────────────────────────────
        scored: list[dict] = []
        for doc in docs:
            score = 0.0
            doc_num_lower = (doc.document_number or "").lower()

            # Exact candidate match → strongest signal
            for c in doc_number_candidates:
                if doc_num_lower == c.lower():
                    score += 0.95
                    break
                elif c.lower() in doc_num_lower:
                    score += 0.70
                    break
            # Fallback: plain document_number match
            if not doc_number_candidates and document_number:
                if document_number.lower() in doc_num_lower:
                    score += 0.85

            # Issuing agency match
            if issuing_agency_text:
                agency_str = f"{doc.issuing_agency or ''} {doc.parent_agency or ''}".lower()
                if any(t.lower() in agency_str for t in issuing_agency_text.split() if len(t) > 1):
                    score += 0.15

            # Type match
            if doc_type_slug and doc.document_type and doc.document_type.slug == doc_type_slug:
                score += 0.25

            # Year match
            if year and doc.published_date and year in str(doc.published_date):
                score += 0.15

            # Title keyword match (Dạng B)
            if title_keywords:
                title_str = f"{doc.document_title or ''} {doc.original_filename or ''}".lower()
                matched = sum(1 for kw in title_keywords if kw.lower() in title_str)
                score += 0.25 * (matched / len(title_keywords))

            score = min(score, 1.0)
            scored.append({
                "document_id": str(doc.id),
                "title": _format_doc_title(doc.document_title, doc.original_filename),
                "document_number": doc.document_number or "",
                "published_date": doc.published_date or "",
                "score": score,
                "strategy": "db_query",
                "section_reference": parsed.get("section_reference"),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"[resolve_doc/db_query] {len(scored)} candidates from DB")
        return scored

    except Exception as e:
        logger.warning(f"[resolve_doc/db_query] failed: {e}", exc_info=True)
        return []


# =============================================================================
# Stage 2 (fallback): Memory Agent LLM Extraction (gemma-4-E4B, ~1-2s)
# =============================================================================

async def _extract_by_llm(reference: str) -> dict:
    """
    LLM-based extraction using the FAST memory agent (gemma-4-E4B).
    Called when regex + DB query return 0 results.
    Enriched prompt with DB schema + VN legal numbering rules.

    The prompt folds "văn bản hướng dẫn về Luật X" intent into title_keywords
    (see resolve_doc_prompt examples), so a separate related_to field is not needed.
    """
    try:
        from app.services.llm import get_memory_agent
        from app.services.llm.types import LLMMessage
        from app.prompts.agents.resolve_doc_prompt import build_extract_prompt
        import json

        llm = get_memory_agent()
        prompt = build_extract_prompt(reference)
        resp = await llm.acomplete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=250,
        )
        text = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        jm = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if jm:
            p = json.loads(jm.group())
            logger.info(f"[resolve_doc/llm_extract] {p}")
            return {
                "doc_type_slug":        p.get("doc_type_slug") or None,
                "document_number":      p.get("document_number") or None,
                "doc_number_candidates": p.get("doc_number_candidates") or [],
                "title_keywords":       p.get("title_keywords") or [],
                "issuing_agency_text":  p.get("issuing_agency_text") or None,
                "year":                 p.get("year") or None,
                "section_reference":    p.get("section_reference") or None,
                "confidence":           "medium",
            }
    except Exception as e:
        logger.warning(f"[resolve_doc/llm_extract] memory agent failed: {e}")
    return {}


def _merge_parsed(parsed: dict, llm_parsed: dict) -> None:
    """Merge LLM-extracted fields into the regex-parsed dict (in place).

    List fields (doc_number_candidates, title_keywords) are unioned;
    scalar fields are filled only when the regex pass left them empty.
    """
    for list_key in ("doc_number_candidates", "title_keywords"):
        llm_list = llm_parsed.get(list_key) or []
        existing = parsed.get(list_key) or []
        for item in llm_list:
            if item not in existing:
                existing.append(item)
        parsed[list_key] = existing
    for k, v in llm_parsed.items():
        if k not in ("doc_number_candidates", "title_keywords") and v and not parsed.get(k):
            parsed[k] = v


# =============================================================================
# Stage 3 (last resort): Vector Search Fallback
# =============================================================================

async def _strategy_vector_fallback(
    reference: str,
    workspace_ids: list,
    db,
) -> list[dict]:
    """
    Vector search fallback — used ONLY when DB gives 0 results.
    Runs semantic search on document titles/content.
    """
    try:
        from app.services.agent.tools import search_documents

        result = await search_documents(
            query=reference,
            top_k=5,
            workspace_ids=workspace_ids,
            existing_citation_ids=set(),
            db=db,
            search_mode="vector",
        )
        sources = result.get("sources", [])
        seen: set[str] = set()
        candidates: list[dict] = []
        for src in sources:
            # Handle both dicts and Pydantic models/dataclasses
            if hasattr(src, "model_dump"):
                src_dict = src.model_dump()
            elif hasattr(src, "__dict__"):
                src_dict = src.__dict__
            else:
                src_dict = src if isinstance(src, dict) else {}

            doc_id = str(src_dict.get("document_id", getattr(src, "document_id", "")))
            source_name = src_dict.get("source", getattr(src, "source", getattr(src, "source_file", "")))
            score = float(src_dict.get("score", getattr(src, "score", _VECTOR_DEFAULT_SCORE)))

            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                candidates.append({
                    "document_id": doc_id,
                    "title": source_name,
                    "document_number": "",
                    "published_date": "",
                    # Down-weight vs DB hits, but keep enough signal that a real
                    # semantic match can still clear the scoping threshold.
                    "score": score * _VECTOR_SCORE_FACTOR,
                    "strategy": "vector",
                    "section_reference": None,
                })
        logger.info(f"[resolve_doc/vector] {len(candidates)} candidates")
        return candidates
    except Exception as e:
        logger.warning(f"[resolve_doc/vector] failed: {e}")
        return []


# =============================================================================
# Candidate Merger & Ranker
# =============================================================================

def _merge_candidates(lists: list[list[dict]]) -> list[dict]:
    """
    Merge candidates from multiple strategies.
    Boosts score when multiple strategies agree on the same document_id.
    """
    merged: dict[str, dict] = {}

    for cand_list in lists:
        for c in cand_list:
            doc_id = str(c.get("document_id", ""))
            if not doc_id:
                continue
            if doc_id not in merged:
                merged[doc_id] = dict(c)
                merged[doc_id]["strategies"] = [c.get("strategy", "?")]
            else:
                # Agreement boost: +30% of second strategy score
                merged[doc_id]["score"] += c.get("score", 0) * 0.3
                merged[doc_id]["strategies"].append(c.get("strategy", "?"))
                if not merged[doc_id].get("section_reference") and c.get("section_reference"):
                    merged[doc_id]["section_reference"] = c["section_reference"]

    result = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return result


# =============================================================================
# Topic-aware disambiguation (use the QUESTION, not just the doc reference)
# =============================================================================

# How much the topic-overlap signal can lift a candidate's score (0..1 scale).
_TOPIC_RERANK_WEIGHT = 0.30


def _topic_tokens(topic: str) -> list[str]:
    """Meaningful content tokens from the user's QUESTION, used to disambiguate.

    Strips the document reference scaffolding (action verbs, section refs, doc-type
    words) so what remains is the subject the user is actually asking about — e.g.
    "Nghị định 85 có bao nhiêu cấp độ hệ thống thông tin" → ["cấp","độ","hệ",
    "thống","thông","tin", ...]. Pure-digit tokens are dropped (a bare "85" is a
    number, not a topic word); surrounding punctuation is stripped.
    """
    t = topic.strip()
    for pat in _ACTION_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()
    for spat in _SECTION_PATTERNS:
        t = re.sub(spat, "", t, flags=re.IGNORECASE).strip()
    for keyword, _slug in _DOC_TYPE_KEYWORDS:
        t = re.sub(re.escape(keyword), "", t, count=1, flags=re.IGNORECASE).strip()
    out: list[str] = []
    for tok in _extract_title_keywords(t):
        tok = tok.strip("?.,;:!()\"'“”…").strip()
        if tok and not tok.isdigit():
            out.append(tok)
    return out


def _number_token_present(text: str, bare: str) -> bool:
    """True if `bare` appears in `text` as a standalone number (not part of a
    longer number). "53" matches "53/2022/NĐ-CP" but NOT "853", "153" or "2053"."""
    if not text or not bare:
        return False
    return re.search(rf'(?<!\d){re.escape(bare)}(?!\d)', text) is not None


async def _rerank_candidates(candidates: list[dict], parsed: dict, topic: str, db) -> list[dict]:
    """Enforce explicit-number identity, THEN topic-overlap reranking.

    1. **Number identity (hard filter).** If the query named an explicit document
       number (e.g. "Nghị định 53"), drop any candidate whose number/title/filename
       does NOT contain that number — a document with a different number is simply
       the wrong document, no matter how well its *topic* matches. If this removes
       everything, return [] so the caller reports "not found" instead of
       confidently answering about an unrelated decree. This is what stops a
       question about "Nghị định 53 ... hệ thống thông tin" from resolving to
       "Nghị định 85" just because ND 85 is *about* hệ thống thông tin.
    2. **Topic rerank (soft boost).** Among the surviving candidates, boost those
       whose title matches the question's subject — used to break ties between
       same-number documents.
    """
    if not candidates:
        return candidates
    try:
        from sqlalchemy import select
        from app.models.document import Document

        ids = [c["document_id"] for c in candidates if c.get("document_id")]
        rows = await db.execute(
            select(
                Document.id, Document.document_number,
                Document.document_title, Document.original_filename,
            ).where(Document.id.in_(ids))
        )
        meta: dict[str, dict] = {}
        for r in rows:
            title_text = f"{r.document_title or ''} {r.original_filename or ''}".lower()
            meta[str(r.id)] = {
                "title_text": title_text,
                "id_text": f"{r.document_number or ''} {title_text}".lower(),
            }

        # ── 1. Number-identity hard filter ───────────────────────────────────
        bare = parsed.get("number_raw")
        if bare:
            def _matches(c: dict) -> bool:
                m = meta.get(str(c.get("document_id")))
                text = m["id_text"] if m else (c.get("title") or "").lower()
                return _number_token_present(text, bare)

            matching = [c for c in candidates if _matches(c)]
            if matching:
                if len(matching) != len(candidates):
                    dropped = [c.get("title") for c in candidates if c not in matching]
                    logger.info(
                        f"[resolve_doc/number] kept {len(matching)}/{len(candidates)} "
                        f"matching số {bare!r}; dropped {dropped}"
                    )
                candidates = matching
            else:
                logger.warning(
                    f"[resolve_doc/number] NO candidate matches số {bare!r} "
                    f"(had {[c.get('title') for c in candidates]}) — returning empty "
                    f"to avoid answering about the wrong document"
                )
                return []

        # ── 2. Topic rerank among surviving candidates ───────────────────────
        tokens = _topic_tokens(topic) if topic else []
        if tokens:
            boosted = False
            for c in candidates:
                m = meta.get(str(c.get("document_id")))
                text = m["title_text"] if m else (c.get("title") or "").lower()
                matched = sum(1 for tok in tokens if tok.lower() in text)
                if matched:
                    bonus = _TOPIC_RERANK_WEIGHT * (matched / len(tokens))
                    c["score"] = min(c.get("score", 0.0) + bonus, 1.0)
                    c.setdefault("strategies", []).append("topic")
                    boosted = True
            if boosted:
                candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                logger.info(
                    f"[resolve_doc/topic] reranked {len(candidates)} by topic "
                    f"tokens={tokens} → top={candidates[0].get('title')!r} "
                    f"score={candidates[0].get('score'):.2f}"
                )
        return candidates
    except Exception as e:
        logger.warning(f"[resolve_doc/rerank] failed: {e}")
        return candidates


# =============================================================================
# Public entry point
# =============================================================================

async def resolve_candidates(
    reference: str,
    workspace_ids: list,
    db,
    *,
    topic: Optional[str] = None,
    use_llm_fallback: bool = True,
    status_cb: Optional[StatusCallback] = None,
) -> dict:
    """
    Resolve a document reference to ranked candidate documents.

    Tiered pipeline (~2-4s):
      1. regex extraction (0ms) → SQL query
      2. LLM extraction (gemma-4-E4B, ~1-2s) when DB returned 0 results
      3. Vector fallback when DB+LLM returned 0 results (searches by `topic`)
      4. Fuzzy similar-title search when everything returned 0 results
      5. Topic rerank: when `topic` (the full question) is given, boost candidates
         whose title matches what the user is actually asking about

    Args:
        reference:        the document reference string (e.g. "Nghị định 85")
        workspace_ids:    workspaces to search within
        db:               AsyncSession
        topic:            the FULL user question — its subject content is used to
                          drive the vector fallback and to disambiguate same-number
                          documents (defaults to `reference` when not given)
        use_llm_fallback: set False to skip the LLM extraction stage
        status_cb:        optional async callback(detail) for UI progress

    Returns dict:
        candidates:        ranked list (score 0..1) — empty when nothing matched
        parsed:            the (possibly LLM-enriched) extraction metadata
        section_reference: Điều/Chương/Khoản reference if present, else None
        similar:           fuzzy similar-title docs (only when candidates empty)
        counts:            {"db", "llm_db", "vector"} candidate counts per stage
    """
    # The question carries disambiguating signal beyond the bare doc reference.
    topic = (topic or reference or "").strip()
    async def _status(detail: str) -> None:
        if status_cb is not None:
            try:
                await status_cb(detail)
            except Exception:
                pass

    # ── Stage 0+1: Regex + DB (no LLM) ───────────────────────────────────────
    parsed = _extract_by_regex(reference)
    db_candidates = await _query_db(parsed, workspace_ids, db)

    # ── Stage 2: LLM fallback (memory agent) ─────────────────────────────────
    # FIX: run whenever DB is dry — NOT gated on regex confidence. A high-
    # confidence regex parse that still misses the DB (wrong number format,
    # doc not uploaded) is exactly when LLM re-extraction helps most.
    llm_candidates: list[dict] = []
    if not db_candidates and use_llm_fallback:
        logger.info("[resolve_doc] DB gave 0 results — invoking memory agent for extraction")
        await _status("Đang phân tích chi tiết văn bản...")
        llm_parsed = await _extract_by_llm(reference)
        if llm_parsed:
            _merge_parsed(parsed, llm_parsed)
            llm_candidates = await _query_db(parsed, workspace_ids, db)

    # ── Stage 3: Vector fallback (last resort) ───────────────────────────────
    vector_candidates: list[dict] = []
    if not (db_candidates + llm_candidates):
        logger.info("[resolve_doc] DB+LLM gave 0 results — running vector fallback")
        await _status("Đang tìm kiếm theo nội dung...")
        # Search by the full question, not just the doc reference — its subject
        # content ("cấp độ hệ thống thông tin") is the strongest disambiguator.
        vector_candidates = await _strategy_vector_fallback(topic, workspace_ids, db)

    # ── Merge & rank ──────────────────────────────────────────────────────────
    all_candidates = _merge_candidates([db_candidates, llm_candidates, vector_candidates])
    logger.info(
        f"[resolve_doc] merged {len(all_candidates)} candidates "
        f"(db={len(db_candidates)}, llm_db={len(llm_candidates)}, vector={len(vector_candidates)})"
    )

    # ── Stage 5: number-identity filter + topic rerank ───────────────────────
    # Always run when there are candidates: (1) enforce the explicit document
    # number so topic content can't override which document was named, then
    # (2) use the question's subject to break ties. Runs for both paths — ReAct
    # passes the full question as `topic`; the static path passes the full query
    # as `reference` (topic == reference there).
    if all_candidates:
        all_candidates = await _rerank_candidates(all_candidates, parsed, topic, db)

    # ── Stage 4: fuzzy similar-title search when nothing matched ─────────────
    similar: list[dict] = []
    if not all_candidates:
        await _status("Đang tìm văn bản tương tự...")
        similar = await _search_similar_documents(reference, workspace_ids, db)

    return {
        "candidates": all_candidates,
        "parsed": parsed,
        "section_reference": parsed.get("section_reference"),
        "similar": similar,
        "counts": {
            "db": len(db_candidates),
            "llm_db": len(llm_candidates),
            "vector": len(vector_candidates),
        },
    }

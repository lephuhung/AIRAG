"""
Resolve Doc Agent
=================

Phase 2: Dedicated agent for resolving ambiguous document references.

Performance-optimized implementation (~2-4s vs old 22s):
- Strategy 0: Pure regex extraction (0ms) → try DB query immediately
- Strategy LLM: Memory agent (Qwen3-4B, ~1-2s) ONLY if regex gives no results
- Early exit: stop searching when high-confidence result found
- Vector search: fallback ONLY if DB gives 0 results

Graph position:
    supervisor (intent=resolve_doc) → resolve_doc_agent → answer_generator / END
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agents.models import SupervisorState

logger = logging.getLogger(__name__)

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
    ("chính phủ",                           "CP",      ["NĐ-CP", "NQ-CP"]),
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


# =============================================================================
# Helpers
# =============================================================================

import datetime as _dt


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
    current_year = _dt.datetime.now().year
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
    """Return meaningful tokens from leftover text for document_title search."""
    return [
        t for t in text.split()
        if len(t) > 2 and t.lower() not in _LEGAL_STOPWORDS
    ]


# =============================================================================
# Step 0: Fast Regex Extraction (no LLM, <1ms)
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
    # Also capture a bare number after stripping type/agency (e.g. "TT 15 BCA")
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

    # ── Extract year ──────────────────────────────────────────────────────────
    ym = re.search(r'\b((?:19|20)\d{2})\b', text)
    if ym:
        result["year"] = ym.group(1)
        text = (text[:ym.start()] + text[ym.end():]).strip()
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
# Step 1: DB Query (fast SQL, no LLM)
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

        # ── 4. Title keywords — AND each kw (Dạng B: nhớ tên) ────────────
        # Only apply when no number candidates (avoid over-filtering)
        if title_keywords and not doc_number_candidates:
            for kw in title_keywords[:4]:
                query = query.where(or_(
                    Document.document_title.ilike(f"%{kw}%"),
                    Document.original_filename.ilike(f"%{kw}%"),
                ))

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
                    score += 0.25

            # Type match
            if doc_type_slug and doc.document_type and doc.document_type.slug == doc_type_slug:
                score += 0.20

            # Year match
            if year and doc.published_date and year in str(doc.published_date):
                score += 0.15

            # Title keyword match (Dạng B)
            if title_keywords:
                title_str = f"{doc.document_title or ''} {doc.original_filename or ''}".lower()
                matched = sum(1 for kw in title_keywords if kw.lower() in title_str)
                score += 0.10 * (matched / len(title_keywords))

            score = min(score, 1.0)
            scored.append({
                "document_id": str(doc.id),
                "title": doc.document_title or doc.original_filename or "",
                "document_number": doc.document_number or "",
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
# Step 2 (fallback): Memory Agent LLM Extraction (Qwen3-4B, ~1-2s)
# =============================================================================

async def _extract_by_llm(reference: str) -> dict:
    """
    LLM-based extraction using the FAST memory agent (Qwen3-4B).
    Called ONLY when regex gives low confidence AND DB returned 0 results.
    Enriched prompt with DB schema + VN legal numbering rules.
    """
    try:
        from app.services.llm import get_memory_agent
        from app.services.llm.types import LLMMessage
        import json

        llm = get_memory_agent()
        prompt = (
            "Bạn là chuyên gia tìm kiếm văn bản pháp luật Việt Nam.\n"
            "Trích xuất metadata để search database. Trả về JSON duy nhất.\n\n"
            "DATABASE SCHEMA (bảng documents):\n"
            "  document_number  : Số ký hiệu, vd: 24/2018/QH14, 15/TT-BCA\n"
            "  document_title   : Tiêu đề, vd: Luật An ninh mạng\n"
            "  issuing_agency   : Cơ quan, vd: Bộ Công an, Chính phủ\n"
            "  published_date   : Năm/ngày, vd: 2018, 2026-03-20\n"
            "  document_type    : luat|bo_luat|nghi_dinh|thong_tu|quyet_dinh|nghi_quyet|phap_lenh|chi_thi|thong_tu_lien_tich\n\n"
            "QUY TẮC SỐ VĂN BẢN VIỆT NAM:\n"
            "  Luật (Quốc hội)      : [số]/[năm]/QH15  vd: 24/2018/QH14, 129/2025/QH15\n"
            "  Nghị định (CP)       : [số]/[năm]/NĐ-CP vd: 83/2026/NĐ-CP\n"
            "  Thông tư Bộ X        : [số]/[năm]/TT-[MÃ] vd: 15/2026/TT-BCA\n"
            "  Quyết định UBND      : [số]/[năm]/QĐ-UBND\n"
            "  Nghị quyết HĐND      : [số]/[năm]/NQ-HĐND\n"
            "  Thông tư liên tịch   : [số]/[năm]/TTLT-[BỘ1]-[BỘ2]\n\n"
            "MÃ CƠ QUAN: BCA=Công an|BTC=Tài chính|BCT=Công Thương|BTP=Tư pháp\n"
            "            BYT=Y tế|BNV=Nội vụ|BGDĐT=Giáo dục|BXD=Xây dựng\n"
            "            BGTVT=Giao thông|NHNN=Ngân hàng NN|CP=Chính phủ|TTg=Thủ tướng\n\n"
            f"Câu hỏi: \"{reference}\"\n\n"
            "Trả về JSON (chỉ JSON):\n"
            '{"doc_type_slug":"","document_number":"","doc_number_candidates":[],'
            '"title_keywords":[],"issuing_agency_text":"","year":"","section_reference":""}\n\n'
            "Ví dụ 1 - nhớ số: \"Thông tư 15 của Bộ Công an\"\n"
            '{"doc_type_slug":"thong_tu","document_number":"15/TT-BCA",'
            '"doc_number_candidates":["15/TT-BCA","15/2025/TT-BCA","15/2026/TT-BCA"],'
            '"title_keywords":[],"issuing_agency_text":"Bộ Công an","year":"","section_reference":""}\n'
            "Ví dụ 2 - nhớ tên: \"Luật An ninh mạng 2018\"\n"
            '{"doc_type_slug":"luat","document_number":"24/2018/QH14",'
            '"doc_number_candidates":["24/2018/QH14"],'
            '"title_keywords":["an ninh","mạng"],"issuing_agency_text":"Quốc hội","year":"2018","section_reference":""}\n'
            "Ví dụ 3 - nhớ nội dung: \"Nghị định về xử phạt vi phạm giao thông\"\n"
            '{"doc_type_slug":"nghi_dinh","document_number":"","doc_number_candidates":[],'
            '"title_keywords":["xử phạt","vi phạm","giao thông"],"issuing_agency_text":"Chính phủ","year":"","section_reference":""}'
        )
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


# =============================================================================
# Step 3 (last resort): Vector Search Fallback
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
            score = float(src_dict.get("score", getattr(src, "score", 0.4)))
            
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                candidates.append({
                    "document_id": doc_id,
                    "title": source_name,
                    "score": score * 0.6,  # Down-weight vs DB
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
# Resolve Doc Agent Node
# =============================================================================

async def resolve_doc_agent_node(state: "SupervisorState") -> dict:
    """
    Phase 2: Multi-strategy document resolution agent.

    Optimized pipeline (~2-4s):
    1. regex extraction (0ms) → SQL query
    2. Early exit if high-confidence DB result found
    3. LLM extraction (Qwen3-4B, ~1-2s) ONLY if step 1 gave 0 results
    4. Vector fallback ONLY if steps 1+3 gave 0 results

    Routing:
    - 0 results       → stream "not found" hint → END
    - Ambiguous       → stream clarification options → END
    - Clear winner    → answer_generator (or rag if section_ref present)
    """
    from app.services.agents.models import AgentType
    from app.services.agent.streaming import push_event, get_current_db

    reference = state.get("rewritten_query", "") or state.get("original_query", "")
    workspace_ids = state.get("workspace_ids", [])
    db = get_current_db()

    logger.info(f"[LANGGRAPH_NODE] Entering resolve_doc_agent_node, reference={reference!r}, workspace_ids={workspace_ids}")
    logger.info(f"[resolve_doc_agent] START reference={reference!r}")
    await push_event(state, "status", {
        "step": "searching",
        "detail": "Đang xác định văn bản...",
    })

    # ── Stage 1: Regex + DB (no LLM) ─────────────────────────────────────────
    parsed = _extract_by_regex(reference)
    db_candidates = await _query_db(parsed, workspace_ids, db)

    # Early exit: if we already have a high-confidence clear winner → done
    if db_candidates and db_candidates[0]["score"] >= 0.85:
        top = db_candidates[0]
        logger.info(
            f"[resolve_doc_agent] EARLY EXIT — high-confidence DB hit: "
            f"doc_id={top['document_id']}, score={top['score']:.2f}"
        )
        return _build_resolved_state(top, parsed, state.get("pending_intent"))

    # ── Stage 2: LLM fallback (memory agent, fast) ───────────────────────────
    # Only invoke if regex gave low confidence AND DB gave 0 results
    llm_candidates: list[dict] = []
    if not db_candidates and parsed.get("confidence") in ("low", "medium"):
        logger.info("[resolve_doc_agent] DB gave 0 results — invoking memory agent for extraction")
        await push_event(state, "status", {
            "step": "searching",
            "detail": "Đang phân tích chi tiết văn bản...",
        })
        llm_parsed = await _extract_by_llm(reference)
        if llm_parsed:
            # Merge list fields — extend without duplicates
            for list_key in ("doc_number_candidates", "title_keywords"):
                llm_list = llm_parsed.get(list_key) or []
                existing = parsed.get(list_key) or []
                for item in llm_list:
                    if item not in existing:
                        existing.append(item)
                parsed[list_key] = existing
            # Merge scalar fields — fill gaps only
            for k, v in llm_parsed.items():
                if k not in ("doc_number_candidates", "title_keywords") and v and not parsed.get(k):
                    parsed[k] = v
            # Re-query DB with enriched parsed
            llm_candidates = await _query_db(parsed, workspace_ids, db)

    # ── Stage 3: Vector fallback (last resort) ────────────────────────────────
    vector_candidates: list[dict] = []
    all_db = db_candidates + llm_candidates
    if not all_db:
        logger.info("[resolve_doc_agent] DB+LLM gave 0 results — running vector fallback")
        await push_event(state, "status", {
            "step": "searching",
            "detail": "Đang tìm kiếm theo nội dung...",
        })
        vector_candidates = await _strategy_vector_fallback(reference, workspace_ids, db)

    # ── Merge & rank ──────────────────────────────────────────────────────────
    all_candidates = _merge_candidates([db_candidates, llm_candidates, vector_candidates])

    logger.info(
        f"[resolve_doc_agent] merged {len(all_candidates)} candidates "
        f"(db={len(db_candidates)}, llm_db={len(llm_candidates)}, vector={len(vector_candidates)})"
    )

    # ── Evaluate & Route ──────────────────────────────────────────────────────

    if not all_candidates:
        msg = (
            f"Không tìm thấy văn bản phù hợp với **\"{reference}\"** trong kho tài liệu.\n\n"
            "Bạn có thể:\n"
            "- Cung cấp số văn bản chính xác (ví dụ: 53/2022/NĐ-CP)\n"
            "- Cung cấp tên đầy đủ của văn bản\n"
            "- Kiểm tra xem văn bản đã được tải lên chưa"
        )
        logger.info("[LANGGRAPH_DECISION] resolve_doc_agent decision: no candidates found")
        await push_event(state, "token", msg)
        return {
            "final_answer": msg,
            "next_agent": AgentType.FINISH,
            "sources": [],
            # NOTE: Do NOT write to kg_summaries here — status messages pollute
            # operator.add accumulation and cause result_evaluator to see phantom results.
        }

    top = all_candidates[0]
    top_score = top.get("score", 0.0)

    # Ambiguity: second candidate score ≥ 75% of top
    is_ambiguous = (
        len(all_candidates) > 1
        and (all_candidates[1].get("score", 0) / max(top_score, 0.01)) >= 0.75
    )

    if is_ambiguous:
        options = []
        for i, c in enumerate(all_candidates[:5], 1):
            title = c.get("title") or f"Văn bản {i}"
            num = c.get("document_number", "")
            label = f"**{i}. {title}**"
            if num:
                label += f" (Số: {num})"
            options.append(label)

        clarify_msg = (
            f"Tìm thấy **{len(options)} văn bản** có thể phù hợp với "
            f"**\"{reference}\"**:\n\n"
            + "\n".join(options)
            + "\n\nBạn muốn tra cứu văn bản nào? "
            "Vui lòng chỉ định số thứ tự hoặc cung cấp thêm thông tin."
        )
        logger.info(f"[LANGGRAPH_DECISION] resolve_doc_agent decision: ambiguous ({len(all_candidates)} candidates)")
        await push_event(state, "clarification", {"message": clarify_msg})
        await push_event(state, "token", clarify_msg)
        return {
            "final_answer": clarify_msg,
            "next_agent": AgentType.FINISH,
            # NOTE: Do NOT write to kg_summaries — clarification messages are metadata,
            # not search results. operator.add would accumulate them as phantom results.
        }

    # Single clear winner
    return _build_resolved_state(top, parsed, state.get("pending_intent"))


def _build_resolved_state(top: dict, parsed: dict, pending_intent: str | None = None) -> dict:
    """Build the state update dict for a successfully resolved document."""
    from app.services.agents.models import AgentType

    doc_id = top.get("document_id")
    # Prefer section_ref from top candidate, fallback to parsed
    section_ref = top.get("section_reference") or parsed.get("section_reference") or ""
    title = top.get("title", "văn bản")
    strategies_used = ", ".join(top.get("strategies", [top.get("strategy", "?")]))
    score = top.get("score", 0.0)

    logger.info(
        f"[LANGGRAPH_DECISION] resolve_doc_agent Resolved → doc_id={doc_id}, score={score:.2f}, "
        f"section_ref={section_ref!r}, strategies=[{strategies_used}], "
        f"pending_intent={pending_intent!r}"
    )

    # Confidence threshold: if score is very low, don't scope search to this document.
    # This prevents false narrowing when resolve_doc picks a wrong document.
    LOW_CONFIDENCE_THRESHOLD = 0.30
    if score < LOW_CONFIDENCE_THRESHOLD:
        logger.warning(
            f"[resolve_doc_agent] Low confidence score={score:.2f} < {LOW_CONFIDENCE_THRESHOLD} "
            f"for doc_id={doc_id}, title={title!r} — will NOT scope search to this document"
        )
        # Clear document_ids so rag_agent searches across all workspaces
        resolved: dict = {
            "document_ids": [],  # Don't scope — low confidence
            "section_reference": section_ref,
            # NOTE: Do NOT write to kg_summaries — status messages are metadata,
            # not search results. operator.add would accumulate them as phantom results.
            "next_agent": AgentType.ANSWER_GENERATOR,
            "should_loop_back": False,
        }
    else:
        resolved: dict = {
            "document_ids": [doc_id] if doc_id else [],
            "section_reference": section_ref,
            # NOTE: Do NOT write to kg_summaries — "Đã xác định văn bản" is metadata,
            # not search content. With operator.add reducer, it would accumulate and
            # cause result_evaluator to see phantom results (has_results=True with 0 actual results).
            "next_agent": AgentType.ANSWER_GENERATOR,
            "should_loop_back": False,
        }

    if section_ref:
        resolved["intent"] = "search_section"
        resolved["next_agent"] = AgentType.RAG
        logger.info(f"[resolve_doc_agent] Has section_ref → rag/search_section")
    elif pending_intent:
        # Phase 4: Respect pending_intent from supervisor plan
        resolved["intent"] = pending_intent
        # Route to RAG if intent is search-related
        if pending_intent in ("search", "kg_query", "search_doc_num", "list_docs"):
            resolved["next_agent"] = AgentType.RAG
        else:
            resolved["next_agent"] = AgentType.ANSWER_GENERATOR
        logger.info(f"[resolve_doc_agent] Using pending_intent={pending_intent!r} → {resolved['next_agent']!r}")
    else:
        # Default fallback
        resolved["intent"] = "summarize"
        logger.info(f"[resolve_doc_agent] No section_ref or pending_intent → answer_generator/summarize")

    return resolved

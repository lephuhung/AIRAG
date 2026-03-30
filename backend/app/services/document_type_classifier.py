"""
Document Type Classifier
========================
Classify Vietnamese administrative / legal documents from their markdown content
using an LLM (Qwen3-4B via vLLM / memory agent).

Return a dictionary with:
  - slug             matches DocumentType.slug in DB, or None.
  - document_number  official document reference number (e.g. "13/2023/NĐ-CP").
  - location         Province or city of issuance.
  - issuing_agency   Direct issuing body.
  - parent_agency    Parent organization.
  - published_date   Date of publication.

Document types are loaded from the database (via LegalKGService or document_types table).
On startup, seed with `seed_document_types(db)` to populate the DB from defaults,
then `get_all_document_types()` will load from DB with caching.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document-type defaults — used only for seeding the DB on first startup
# ---------------------------------------------------------------------------

@dataclass
class _DocType:
    slug: str
    name: str
    description: str


_DEFAULT_DOC_TYPES: list[_DocType] = [
    _DocType("luat",        "Luật",         "Luật do Quốc hội ban hành"),
    _DocType("phap_lenh",   "Pháp lệnh",    "Pháp lệnh của Ủy ban Thường vụ Quốc hội"),
    _DocType("nghi_quyet",  "Nghị quyết",   "Nghị quyết của Quốc hội, HĐND, Đảng, tổ chức"),
    _DocType("nghi_dinh",   "Nghị định",    "Nghị định của Chính phủ"),
    _DocType("quyet_dinh",  "Quyết định",   "Quyết định hành chính của cơ quan nhà nước"),
    _DocType("thong_tu",    "Thông tư",     "Thông tư (bao gồm Thông tư liên tịch) của Bộ, cơ quan ngang Bộ"),
    _DocType("chi_thi",     "Chỉ thị",      "Chỉ thị của Thủ tướng, Bộ trưởng"),
    _DocType("cong_van",    "Công văn",     "Công văn hành chính"),
    _DocType("thong_bao",   "Thông báo",    "Thông báo hành chính"),
    _DocType("bao_cao",     "Báo cáo",      "Báo cáo định kỳ, chuyên đề, tài chính"),
    _DocType("to_trinh",    "Tờ trình",     "Tờ trình lên cấp trên"),
    _DocType("bien_ban",    "Biên bản",     "Biên bản họp, làm việc, kiểm tra"),
    _DocType("hop_dong",    "Hợp đồng",     "Hợp đồng kinh tế, dân sự, lao động"),
    _DocType("ke_hoach",    "Kế hoạch",     "Kế hoạch công tác, triển khai"),
    _DocType("huong_dan",   "Hướng dẫn",    "Hướng dẫn thực hiện, nghiệp vụ"),
    _DocType("don_tu",      "Đơn, Tờ khai", "Đơn xin, đơn đề nghị, tờ khai"),
]


# ---------------------------------------------------------------------------
# In-memory cache — loaded from DB, refreshed every 5 minutes
# ---------------------------------------------------------------------------

_cache: list[dict] | None = None
_cache_loaded_at: float = 0.0
_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes

_lock = asyncio.Lock()


async def _load_doc_types_from_db(db) -> list[dict]:
    """Fetch active document types from the DB."""
    from sqlalchemy import select as _select
    from app.models.document_type import DocumentType

    result = await db.execute(
        _select(DocumentType).where(DocumentType.is_active.is_(True)).order_by(DocumentType.name)
    )
    rows = result.scalars().all()
    return [
        {"slug": r.slug, "name": r.name, "description": r.description}
        for r in rows
    ]


async def get_all_document_types(db=None) -> list[dict]:
    """
    Return all active document types.
    Loads from DB if cache is stale (>5 min) or if explicitly refreshed.
    Pass db=None to use cache only. Pass a db session to force a fresh DB read.
    """
    global _cache, _cache_loaded_at

    now = time.monotonic()

    # Fast path: cache hit
    if _cache is not None and (now - _cache_loaded_at) < _CACHE_TTL_SECONDS:
        return _cache

    async with _lock:
        # Double-check after acquiring lock
        if _cache is not None and (now - _cache_loaded_at) < _CACHE_TTL_SECONDS:
            return _cache

        if db is not None:
            # Caller passed an explicit session — use it
            _cache = await _load_doc_types_from_db(db)
        else:
            # Create our own session
            from app.core.database import async_session_maker
            async with async_session_maker() as _db_session:
                try:
                    _cache = await _load_doc_types_from_db(_db_session)
                except Exception as _err:
                    logger.warning(f"[doc_type] DB load failed, falling back to defaults: {_err}")
                    _cache = [{"slug": d.slug, "name": d.name, "description": d.description} for d in _DEFAULT_DOC_TYPES]

        _cache_loaded_at = time.monotonic()
        logger.info(f"[doc_type] Loaded {len(_cache)} document types from DB (cache refreshed)")
        return _cache


def get_cached_slugs() -> frozenset[str]:
    """Return valid slugs from current cache (sync, for use in validation)."""
    if _cache is None:
        # Return defaults if cache not yet populated
        return frozenset(d.slug for d in _DEFAULT_DOC_TYPES)
    return frozenset(d["slug"] for d in _cache)


async def seed_document_types(db) -> None:
    """
    Seed missing document types into the DB using default list.
    Also seeds default kg_system_prompt (LEGAL_KG_SYSTEM_PROMPT) for each document type
    that doesn't have one yet. Called by main.py on startup. Idempotent.
    """
    from sqlalchemy import select
    from app.models.document_type import DocumentType, DocumentTypeSystemPrompt
    from app.services.legal_kg_prompts import LEGAL_KG_SYSTEM_PROMPT
    from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT

    existing = await db.execute(select(DocumentType).where(DocumentType.slug.in_(
        d.slug for d in _DEFAULT_DOC_TYPES
    )))
    existing_slugs = {r.slug for r in existing.scalars().all()}

    for dt in _DEFAULT_DOC_TYPES:
        if dt.slug not in existing_slugs:
            doc_type = DocumentType(
                slug=dt.slug,
                name=dt.name,
                description=dt.description,
            )
            db.add(doc_type)
            await db.flush()  # get id for the new doc_type

            # Seed default DocumentTypeSystemPrompt with kg_system_prompt
            db.add(DocumentTypeSystemPrompt(
                document_type_id=doc_type.id,
                workspace_id=None,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                kg_system_prompt=LEGAL_KG_SYSTEM_PROMPT,
            ))
            logger.info(f"[doc_type] Seeding: {dt.slug}")

    # Backfill kg_system_prompt for existing document types that don't have one
    all_doc_types = await db.execute(
        select(DocumentType).where(DocumentType.is_active.is_(True))
    )
    for doc_type in all_doc_types.scalars().all():
        # Check if this doc_type already has a prompt row (workspace_id=NULL)
        existing_prompt = await db.execute(
            select(DocumentTypeSystemPrompt).where(
                DocumentTypeSystemPrompt.document_type_id == doc_type.id,
                DocumentTypeSystemPrompt.workspace_id.is_(None),
            )
        )
        prompt_row = existing_prompt.scalar_one_or_none()
        if prompt_row is None:
            db.add(DocumentTypeSystemPrompt(
                document_type_id=doc_type.id,
                workspace_id=None,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                kg_system_prompt=LEGAL_KG_SYSTEM_PROMPT,
            ))
            logger.info(f"[doc_type] Backfilling kg_system_prompt for: {doc_type.slug}")
        elif prompt_row.kg_system_prompt is None:
            prompt_row.kg_system_prompt = LEGAL_KG_SYSTEM_PROMPT
            logger.info(f"[doc_type] Setting default kg_system_prompt for: {doc_type.slug}")

    await db.commit()

def _build_llm_system_prompt(doc_types: list[dict]) -> str:
    """Build the LLM system prompt dynamically using the given doc type list."""
    types_list = "\n".join(
        f"  - {d['slug']}: {d['name']} — {d['description']}" for d in doc_types
    )
    return (
        "Bạn là chuyên gia phân tích siêu dữ liệu văn bản hành chính Việt Nam.\n"
        "Nhiệm vụ: Đọc phần đầu của văn bản (markdown hoặc text OCR trang đầu) và bóc tách các trường thông tin tiêu chuẩn của Header.\n"
        "Trả về JSON thuần với đúng 7 trường sau (không dùng markdown):\n\n"
        "1. \"slug\": loại văn bản (chọn từ danh sách bên dưới, hoặc \"unknown\")\n"
        "2. \"document_number\": số hiệu chính thức, thường bắt đầu bằng Luật số:, số: (ví dụ: \"13/2023/NĐ-CP\", \"29/2018/QH14\",\"23/BC-VPUB\"). Bỏ qua chữ 'Số:', 'Luật số:'. Thường nằm bên dưới parent_agency\n"
        "3. \"document_title\": Tên/Tiêu đề văn bản, thường nằm ngay dưới số ký hiệu (VD: \"Luật Bảo vệ Bí mật nhà nước\", \"Kế hoạch triển khai thực hiện\", \"Giấy mời tham gia\").\n"
        "4. \"parent_agency\": Tên cơ quan chủ quản / cấp trên, thường ở góc TRÊN CÙNG BÊN TRÁI, có trường hợp nằm trên 2 dòng (VD: \"UBND TỈNH HÀ TĨNH\", \"Ủy ban nhân dân \n Tỉnh Hà Tĩnh\").\n"
        "5. \"issuing_agency\": Tên đơn vị ban hành trực tiếp, thường nằm ngay dưới parent_agency (VD: \"VĂN PHÒNG\").\n"
        "6. \"location\": Địa danh ban hành ở góc TRÊN CÙNG BÊN PHẢI (VD: \"Hà Tĩnh\", \"Hà Nội\").\n"
        "7. \"published_date\": Ngày tháng năm ban hành (VD: \"15/01/2026\").\n\n"
        "Các slug hợp lệ:\n"
        + types_list
        + "\n\n"
        "Quy tắc:\n"
        "- Chỉ trả về duy nhất chuỗi JSON, không giải thích.\n"
        "- Nếu giá trị nào không có, đặt là `null`.\n"
        "- Mọi text nên giữ nguyên case gốc nếu được hoặc chuẩn hoá Title Case.\n"
        "Ví dụ đầu ra:\n"
        "{\"slug\": \"luat\", \"document_number\": \"13/2024/QH15\", \"document_title\": \"Luật Bảo vệ Bí mật nhà nước\", \"parent_agency\": \"QUỐC HỘI\", \"issuing_agency\": \"VP QUỐC HỘI\", \"location\": \"Hà Nội\", \"published_date\": \"15/06/2024\"}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def classify_with_llm(markdown_text: str, db=None) -> dict:
    """
    Classify a document and extract rich header metadata via memory agent LLM.

    Returns:
        A dict containing: slug, document_number, location, issuing_agency, parent_agency, published_date.
    """
    from app.services.llm import get_memory_agent
    from app.services.llm.types import LLMMessage

    default_result = {
        "slug": None, "document_number": None, "document_title": None,
        "location": None, "issuing_agency": None, "parent_agency": None, "published_date": None
    }
    if not markdown_text:
        return default_result

    content_str = ""
    try:
        # Load doc types from cache/DB
        doc_types = await get_all_document_types(db)
        valid_slugs = frozenset(d["slug"] for d in doc_types)
        system_prompt = _build_llm_system_prompt(doc_types)

        llm = get_memory_agent()
        preview = markdown_text[:1500].strip()
        messages: list[LLMMessage] = [
            LLMMessage(role="user", content=f"Nội dung văn bản (markdown):\n\n{preview}")
        ]
        result = await llm.acomplete(
            messages,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=200,
        )
        content_str = result if isinstance(result, str) else result.content
        if not content_str:
            return default_result

        # Strip markdown fences if the model wrapped the JSON
        content_str = content_str.strip()
        if content_str.startswith("```"):
            content_str = re.sub(r"^```[a-z]*\n?", "", content_str)
            content_str = content_str.rstrip("`").strip()

        parsed = json.loads(content_str)

        # --- slug ---
        raw_slug = str(parsed.get("slug", "")).strip().lower()
        raw_slug = re.sub(r"[^a-z_]", "", raw_slug)
        slug = raw_slug if raw_slug in valid_slugs else None

        # Clean string helper
        def _clean_str(val):
            if not val or str(val).lower() == "null" or str(val).lower() == "none":
                return None
            return str(val).strip()

        document_number = _clean_str(parsed.get("document_number"))
        if document_number and len(document_number) > 60:
            document_number = None

        # Clean document_title (truncate if too long)
        document_title = _clean_str(parsed.get("document_title"))
        if document_title and len(document_title) > 300:
            document_title = document_title[:300]

        final_res = {
            "slug": slug,
            "document_number": document_number,
            "document_title": document_title,
            "location": _clean_str(parsed.get("location")),
            "issuing_agency": _clean_str(parsed.get("issuing_agency")),
            "parent_agency": _clean_str(parsed.get("parent_agency")),
            "published_date": _clean_str(parsed.get("published_date"))
        }

        logger.info(
            f"[classifier] Extracted rich metadata: {final_res}"
        )
        return final_res

    except (json.JSONDecodeError, KeyError, TypeError) as parse_err:
        logger.debug(f"[classifier] JSON parse failed: {parse_err} — raw: {content_str!r}")
    except Exception as err:
        logger.warning(f"[classifier] LLM classification failed (non-fatal): {err}")

    return default_result



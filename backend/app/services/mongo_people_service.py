"""
MongoDB People Search Service
=============================

Search tools for person records stored in MongoDB.
Hỗ trợ nhiều schema (collections) với field names khác nhau,
được định nghĩa trong `mongo_searchable_map.py`.

Functions:
    search_by_cccd   — exact/regex match trên tất cả schemas có field CCCD
    search_by_name   — regex match trên tất cả schemas có field tên
    search_by_bhxh   — exact/regex match trên tất cả schemas có field BHXH
    search_by_phone  — exact/prefix/contains trên tất cả schemas có field phone
"""

from __future__ import annotations

import asyncio
import logging
import re

from app.services.mongo_client import get_mongo_db
from app.services.mongo_searchable_map import (
    SEARCHABLE_COLLECTION_MAP,
    enrich_display_with_schema,
    get_schema_display_name,
)

logger = logging.getLogger(__name__)

COLLECTION = None  # Không dùng cố định 1 collection nữa — dùng map

# ============================================================================
# Helpers
# ============================================================================


def _normalize_phone(phone: str) -> str:
    """Strip spaces, dashes, dots from phone number."""
    return re.sub(r"[\s\-\.]+", "", phone)


# MongoDB ObjectId is a 24-character hexadecimal string
_OBJECT_ID_RE = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)
# Pattern to strip embedded ObjectId strings from display output
# Matches: [24-hex-chars] OR 24-hex-chars (not preceded/followed by hex chars)
_STRIP_OBJECT_ID_RE = re.compile(r"\[[0-9a-f]{24}\]|(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])", re.IGNORECASE)


def _is_object_id(value) -> bool:
    """Return True if value is or looks like a MongoDB ObjectId (24 hex chars)."""
    if isinstance(value, str):
        return bool(_OBJECT_ID_RE.match(value))
    if isinstance(value, (list, tuple)):
        return any(_is_object_id(v) for v in value)
    if isinstance(value, dict):
        return any(_is_object_id(v) for v in value.values())
    return False


def _sanitize_record(doc: dict) -> dict:
    """
    Return a shallow copy of doc with all ObjectId values removed.
    Recursively sanitizes nested lists and dicts.
    """
    sanitized = {}
    for key, val in doc.items():
        if _is_object_id(val):
            continue
        if isinstance(val, list):
            sanitized[key] = [
                v for v in val
                if not _is_object_id(v)
            ]
        elif isinstance(val, dict):
            sanitized[key] = {
                k: v for k, v in val.items()
                if not _is_object_id(v)
            }
        else:
            sanitized[key] = val
    return sanitized


def _build_display_text(doc: dict, display_fields: list[str], schema: str) -> str:
    """
    Format a MongoDB document into a readable string.

    Strategy:
      - Show ALL configured display_fields (with "—" for null/empty values)
      - If display_fields is empty or < 3 items, fall back to ALL document fields
        (this ensures we always pass maximum context to the LLM)
    """
    if not doc:
        return ""

    label_map = {
        # BHXH fields
        "hoTen": "Họ tên",
        "maSoBhxh": "Mã số BHXH",
        "soTheBhyt": "Số thẻ BHYT",
        "ngaySinhHienThi": "Ngày sinh",
        "trangThaiThe": "Trạng thái thẻ",
        "tyLeBhyt": "Tỷ lệ BHYT",
        "tuNgay": "Từ ngày",
        "denNgay": "Đến ngày",
        "coSoKCB": "Cơ sở KCB",
        "soDienThoai": "Điện thoại",
        "soCmnd": "Số CMND/CCCD",
        "diaChi": "Địa chỉ",
        # LG fields
        "TenHoiVien": "Họ tên",
        "SoDienThoai": "Điện thoại",
        "DiaChi": "Địa chỉ",
        "DiemHoiVien": "Điểm hội viên",
        "TenHangHoiVien": "Hạng hội viên",
        "SoTheHoiVien": "Số thẻ hội viên",
        "SoDinhDanh": "Số CCCD/CMND",
        "NgaySinh": "Ngày sinh",
        # Vacxin fields
        "HO_TEN": "Họ tên",
        "NGAY_SINH": "Ngày sinh",
        "TEN_ME": "Tên mẹ",
        "DIEN_THOAI_ME": "Điện thoại mẹ",
        "MA_DOI_TUONG": "Mã định danh",
        "GIOI_TINH": "Giới tính",
        "PID": "Mã PID",
        # VNVC fields
        "mobile": "Điện thoại",
        "fullName": "Họ tên",
        "fullNam": "Ngày sinh",
        "diaChi": "Địa chỉ",
        # CV19 fields
        "so_dien_thoai": "Điện thoại",
        "ho_ten": "Họ tên",
        "namsinh": "Năm sinh",
        "gioi_tinh": "Giới tính",
        "dia_chi": "Địa chỉ",
        # EVN fields
        "tenKhachHang": "Tên khách hàng",
        "cmnd": "Số CMND",
        "phone": "Điện thoại",
        "diaChiCapDien": "Địa chỉ cấp điện",
        "ngayDangKy": "Ngày đăng ký",
        # Generic
    }

    # Determine which fields to show
    if not display_fields or len(display_fields) < 3:
        # Fall back to ALL document fields (minus _id, id, and system fields)
        # Also skip any field whose value looks like a MongoDB ObjectId (24 hex chars)
        fields_to_show = [
            k for k in doc.keys()
            if k not in ("_id", "id") and not k.startswith("_") and not _is_object_id(doc.get(k))
        ]
    else:
        fields_to_show = display_fields

    parts = [get_schema_display_name(schema)]
    shown_any = False

    for field in fields_to_show:
        val = doc.get(field)
        # Skip ObjectId values (24 hex chars) — they are internal DB references, not meaningful display data
        if _is_object_id(val):
            continue
        label = label_map.get(field, field)
        # Format gender
        if field in ("GIOI_TINH", "gioi_tinh"):
            if val == 1 or val == "1":
                val = "Nam"
            elif val == 0 or val == "0":
                val = "Nữ"
        if val is None or val == "" or val == "None":
            val = "—"
        else:
            val = str(val)
            shown_any = True
        parts.append(f"  • {label}: {val}")

    # If nothing found at all, show raw fields as last resort
    if not shown_any and fields_to_show:
        parts.append("  (không có thêm thông tin chi tiết trong hồ sơ)")

    text = "\n".join(parts)
    # Strip any remaining ObjectId hex strings that may be embedded in compound values
    text = _STRIP_OBJECT_ID_RE.sub("", text)
    return text


def _merge_results(
    results_by_schema: dict[str, list[dict]],
    lookup_type: str,
) -> tuple[list[dict], str, list[str]]:
    """
    Merge results from multiple schemas into a single display string.

    Returns: (persons_list, display_text, schemas_with_results)
    """
    all_persons: list[dict] = []
    lines: list[str] = []
    schemas_with_results: list[str] = []

    total = sum(len(r) for r in results_by_schema.values())
    if total == 0:
        return [], f"Không tìm thấy kết quả nào cho lookup type: {lookup_type}", []

    for schema_name, docs in results_by_schema.items():
        if not docs:
            continue
        schemas_with_results.append(schema_name)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            doc["_source_schema"] = schema_name
            sanitized = _sanitize_record(doc)
            all_persons.append(sanitized)

    if total == 1:
        # Single result
        doc = all_persons[0]
        schema = doc.pop("_source_schema", "")
        cfg = SEARCHABLE_COLLECTION_MAP.get(lookup_type, {}).get("collections", {}).get(schema, {})
        display = _build_display_text(doc, cfg.get("display_fields", []), schema)
        return [doc], display, schemas_with_results

    # Multiple results — use per-schema display_fields (not merged) so field names
    # correctly match each document's actual schema structure
    lines.append(f"Tìm thấy **{total}** kết quả:")
    for i, doc in enumerate(all_persons, 1):
        schema = doc.pop("_source_schema", "")
        cfg = SEARCHABLE_COLLECTION_MAP.get(lookup_type, {}).get("collections", {}).get(schema, {})
        display_fields = cfg.get("display_fields", [])
        # Use per-schema display_fields for this document
        person_display = _build_display_text(doc, display_fields, schema)
        lines.append(f"{'─'*30}\n📋 Kết quả {i}\n{'─'*30}")
        lines.append(person_display)

    return all_persons, "\n".join(lines), schemas_with_results


# ============================================================================
# Multi-schema search functions
# ============================================================================


def _query_single_schema_sync(
    collection_name: str,
    collection,
    query_fields: list[str],
    match_mode: str,
    value: str,
) -> tuple[str, list[dict]]:
    """
    Perform a single-schema MongoDB query synchronously.
    Must be called inside asyncio.to_thread() from _search_multi.

    Returns: (schema_name, docs)
    """
    try:
        if match_mode == "exact":
            or_query = {"$or": [{field: value} for field in query_fields]}
            cursor = collection.find(or_query).limit(10)
            docs = cursor.to_list(length=10)
            # Fallback: case-insensitive regex (for CCCD with different formats in DB)
            if not docs:
                or_query_regex = {
                    "$or": [
                        {field: {"$regex": f"^{re.escape(value)}$", "$options": "i"}}
                        for field in query_fields
                    ]
                }
                cursor = collection.find(or_query_regex).limit(10)
                docs = cursor.to_list(length=10)
            return collection_name, docs

        elif match_mode == "regex":
            or_query = {
                "$or": [
                    {field: {"$regex": value, "$options": "i"}}
                    for field in query_fields
                ]
            }
            cursor = collection.find(or_query).limit(10)
            docs = cursor.to_list(length=10)
            return collection_name, docs

        elif match_mode == "phone":
            norm = _normalize_phone(value)
            # Exact match only — suffix/partial matching is disabled because
            # it generates false positives (e.g. regex .{6}6889$ matches any
            # string ending in 6889, not just phone numbers). If the exact
            # phone number is not in the DB, return empty rather than
            # returning wrong people to the LLM.
            exact_q = {"$or": [{field: norm} for field in query_fields]}
            cursor = collection.find(exact_q).limit(10)
            docs = cursor.to_list(length=10)
            return collection_name, docs

        else:
            return collection_name, []

    except Exception as e:
        logger.warning(f"[_query_single_schema] ❌ {collection_name}.{query_fields}: {e}")
        return collection_name, []


async def _search_multi(
    lookup_type: str,
    value: str,
    match_mode: str = "exact",  # "exact" | "regex" | "phone"
) -> dict:
    """
    Generic multi-schema search using SEARCHABLE_COLLECTION_MAP.

    Optimizations:
      - All schema queries run in PARALLEL via asyncio.gather()
      - Exact match uses equality (can use index) not regex
      - Phone: exact match only — suffix fallback removed (false positive risk)
      - Limit 10 docs per collection
    """
    if lookup_type not in SEARCHABLE_COLLECTION_MAP:
        return {
            "found": False,
            "persons": [],
            "display": f"Không hỗ trợ lookup type: {lookup_type}",
            "lookup_type": lookup_type,
        }

    schema_map = SEARCHABLE_COLLECTION_MAP[lookup_type]["collections"]
    db = get_mongo_db()

    # Build list of asyncio.to_thread tasks — one per schema, all run in parallel
    tasks = []
    for schema_name, cfg in schema_map.items():
        query_fields = cfg.get("fields", [])
        if not query_fields:
            continue
        collection = db[schema_name]
        logger.warning(
            f"[_search_multi] 📡 {schema_name}.{query_fields} — "
            f"lookup_type={lookup_type}, match_mode={match_mode}, value={value!r}"
        )
        task = asyncio.to_thread(
            _query_single_schema_sync,
            schema_name,
            collection,
            query_fields,
            match_mode,
            value,
        )
        tasks.append(task)

    # Wait for all schema queries to complete in parallel
    results_list = await asyncio.gather(*tasks)

    results_by_schema: dict[str, list[dict]] = {}
    for schema_name, docs in results_list:
        results_by_schema[schema_name] = docs
        if docs:
            logger.warning(f"[_search_multi] ✅ {schema_name} → {len(docs)} docs")

    persons, display, schemas = _merge_results(results_by_schema, lookup_type)
    if schemas:
        display = enrich_display_with_schema(display, schemas)
    return {
        "found": bool(persons),
        "persons": persons,
        "display": display,
        "lookup_type": lookup_type,
    }


# ============================================================================
# Public API — 1 function per intent
# ============================================================================


async def search_by_cccd(cccd: str) -> dict:
    """
    Search for a person by CCCD/CMND across all schemas.
    CCCD must be exactly 9 or 12 digits (Vietnamese national ID format).

    Returns: {"found": bool, "persons": list[dict], "display": str}
    """
    cccd_digits = re.sub(r"\D", "", cccd)  # strip non-digits
    if not cccd_digits or len(cccd_digits) not in (9, 12):
        return {"found": False, "persons": [], "display": "Số CCCD không hợp lệ (cần 9 hoặc 12 chữ số)."}

    logger.info(f"[search_by_cccd] Searching CCCD: {cccd_digits}")
    result = await _search_multi("cccd", cccd_digits, match_mode="exact")

    if not result["found"]:
        return {"found": False, "persons": [], "display": f"Không tìm thấy người có CCCD: {cccd_digits}"}

    return result


async def search_by_name(name: str, limit: int = 10) -> dict:
    """
    Search for persons by name (partial, case-insensitive regex).

    Returns: {"found": bool, "persons": list[dict], "display": str}
    """
    if not name or len(name) < 2:
        return {"found": False, "persons": [], "display": "Tên tìm kiếm quá ngắn."}

    logger.info(f"[search_by_name] Searching name: {name}")
    result = await _search_multi("name", name, match_mode="regex")
    result["persons"] = result["persons"][:limit]
    return result


async def search_by_bhxh(so_bhxh: str) -> dict:
    """
    Search for a person by BHXH number across all schemas.
    BHXH number must be digits only, at least 5 digits.

    Returns: {"found": bool, "persons": list[dict], "display": str}
    """
    bhxh_digits = re.sub(r"\D", "", so_bhxh)  # strip non-digits
    if not bhxh_digits or len(bhxh_digits) < 5:
        return {"found": False, "persons": [], "display": "Số BHXH không hợp lệ (cần ít nhất 5 chữ số)."}

    logger.info(f"[search_by_bhxh] Searching BHXH: {bhxh_digits}")
    result = await _search_multi("bhxh", bhxh_digits, match_mode="exact")

    if not result["found"]:
        return {"found": False, "persons": [], "display": f"Không tìm thấy người có số BHXH: {bhxh_digits}"}

    return result


async def search_by_phone(phone: str, limit: int = 10) -> dict:
    """
    Search for persons by phone number across all schemas.
    Phone number must be exactly 10 digits starting with a valid prefix.
    Uses EXACT match only — suffix/partial matching was removed because
    it generated false positives (matching any string ending in the same digits).

    Returns: {"found": bool, "persons": list[dict], "display": str}
    """
    phone_digits = re.sub(r"\D", "", phone)  # strip non-digits
    if len(phone_digits) != 10:
        return {"found": False, "persons": [], "display": "Số điện thoại không hợp lệ (cần đúng 10 chữ số)."}

    logger.info(f"[search_by_phone] Searching phone: {phone_digits}")
    logger.warning(f"[search_by_phone] 🔍 REAL MongoDB QUERY — phone={phone_digits!r}")
    result = await _search_multi("phone", phone_digits, match_mode="phone")
    logger.warning(f"[search_by_phone] ✅ RETURNED — found={result['found']}, count={len(result['persons'])}")
    result["persons"] = result["persons"][:limit]
    return result

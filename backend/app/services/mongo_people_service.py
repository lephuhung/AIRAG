"""
MongoDB People Search Service
=============================

Search tools for person records stored in MongoDB.
Hỗ trợ nhiều schema (collections) với field names khác nhau,
được định nghĩa trong `mongo_searchable_map.py`.

Tính năng:
    - search_by_cccd / bhxh / phone : hỗ trợ TÌM NHIỀU GIÁ TRỊ cùng lúc ($in)
    - search_by_name                : regex (không phân biệt hoa thường), nhiều tên
    - search_by_advanced            : tìm tổ hợp tiêu chí (name + dob + address + phone)
    - GỘP HỒ SƠ: gom kết quả từ mọi schema về cùng 1 người (theo CCCD/BHXH/tên+NS)
      thành một "hồ sơ tổng hợp" thay vì liệt kê rời rạc.
    - FALLBACK: khi MongoDB (10.10.0.120) lỗi/không kết nối được, trả thông báo
      "hệ thống đang bận" thay vì báo "không tìm thấy" (tránh trả lời sai).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter

from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    ExecutionTimeout,
    NetworkTimeout,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from app.services.mongo_client import get_mongo_db
from app.services.mongo_searchable_map import (
    SEARCHABLE_COLLECTION_MAP,
    get_schema_display_name,
)

logger = logging.getLogger(__name__)

COLLECTION = None  # Không dùng cố định 1 collection nữa — dùng map

# Thông báo khi hệ thống tra cứu (MongoDB ngoài) lỗi/không kết nối được.
BUSY_MESSAGE = (
    "⚠️ Hệ thống tra cứu dữ liệu đang bận hoặc tạm thời không kết nối được. "
    "Vui lòng thử lại sau giây lát."
)

# Thời gian tối đa cho MỖI truy vấn 1 schema (ms). Một collection thiếu index
# (vd 80M docs) sẽ bị server hủy sạch sau ngưỡng này (ExecutionTimeout) và được
# BỎ QUA — trả kết quả từ các schema có index thay vì treo cả request.
QUERY_MAX_MS = 5000

# Các lỗi cấp KẾT NỐI/HẠ TẦNG của MongoDB — coi là "hệ thống đang bận".
# (ServerSelectionTimeoutError/NetworkTimeout/AutoReconnect đều là con của
#  ConnectionFailure, nhưng liệt kê tường minh cho rõ ràng.)
_CONN_ERRORS = (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    NetworkTimeout,
    AutoReconnect,
)


class MongoUnavailable(Exception):
    """Raised when MongoDB is unreachable — distinct from 'no results found'."""


def _busy_result(lookup_type: str = "") -> dict:
    return {
        "found": False,
        "error": "unavailable",
        "persons": [],
        "display": BUSY_MESSAGE,
        "lookup_type": lookup_type,
    }


# ============================================================================
# Helpers
# ============================================================================


def _normalize_phone(phone: str) -> str:
    """Strip spaces, dashes, dots from phone number."""
    return re.sub(r"[\s\-\.]+", "", phone)


# MongoDB ObjectId is a 24-character hexadecimal string
_OBJECT_ID_RE = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)
# Pattern to strip embedded ObjectId strings from display output
_STRIP_OBJECT_ID_RE = re.compile(
    r"\[[0-9a-f]{24}\]|(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])", re.IGNORECASE
)


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
    """Return a shallow copy of doc with all ObjectId values removed."""
    sanitized = {}
    for key, val in doc.items():
        if _is_object_id(val):
            continue
        if isinstance(val, list):
            sanitized[key] = [v for v in val if not _is_object_id(v)]
        elif isinstance(val, dict):
            sanitized[key] = {k: v for k, v in val.items() if not _is_object_id(v)}
        else:
            sanitized[key] = val
    return sanitized


def _build_display_text(doc: dict, display_fields: list[str], schema: str) -> str:
    """
    Format a MongoDB document into a readable string.

    Strategy:
      - Show ALL configured display_fields (with "—" for null/empty values)
      - If display_fields is empty or < 3 items, fall back to ALL document fields
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
        # UIDS fields
        "uid": "Facebook UID",
    }

    if not display_fields or len(display_fields) < 3:
        fields_to_show = [
            k
            for k in doc.keys()
            if k not in ("_id", "id")
            and not k.startswith("_")
            and not _is_object_id(doc.get(k))
        ]
    else:
        fields_to_show = display_fields

    parts: list[str] = []
    shown_any = False

    for field in fields_to_show:
        val = doc.get(field)
        if _is_object_id(val):
            continue
        label = label_map.get(field, field)
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
        parts.append(f"      - {label}: {val}")

    if not shown_any and fields_to_show:
        parts.append("      (không có thêm thông tin chi tiết trong hồ sơ)")

    text = "\n".join(parts)
    text = _STRIP_OBJECT_ID_RE.sub("", text)
    return text


# ============================================================================
# Identity extraction & consolidation (gộp thông tin của cùng 1 người)
# ============================================================================

# Canonical attribute → label (thứ tự hiển thị)
_CANON_LABELS = [
    ("name", "Họ tên"),
    ("dob", "Ngày sinh"),
    ("cccd", "CCCD/CMND"),
    ("phone", "Điện thoại"),
    ("bhxh", "Mã số BHXH"),
    ("address", "Địa chỉ"),
]


def _field_for(schema: str, lookup_type: str) -> list[str]:
    cfg = (
        SEARCHABLE_COLLECTION_MAP.get(lookup_type, {})
        .get("collections", {})
        .get(schema)
    )
    if not cfg:
        return []
    fields = cfg.get("fields", [])
    # advanced map uses a dict {canon: field}; others use a list
    if isinstance(fields, dict):
        return list(fields.values())
    return fields


def _extract_canonical(schema: str, doc: dict) -> dict[str, str]:
    """
    Trích các thuộc tính chuẩn hoá (name/cccd/phone/bhxh/dob/address) từ một
    document, dựa trên ánh xạ field trong SEARCHABLE_COLLECTION_MAP.
    """
    out: dict[str, str] = {}

    for canon, lt in (("cccd", "cccd"), ("bhxh", "bhxh"), ("phone", "phone"), ("name", "name")):
        for f in _field_for(schema, lt):
            v = doc.get(f)
            if v not in (None, "", "None"):
                out[canon] = str(v).strip()
                break

    # dob/address (và bù name/phone nếu thiếu) lấy từ map "advanced"
    adv = (
        SEARCHABLE_COLLECTION_MAP.get("advanced", {})
        .get("collections", {})
        .get(schema, {})
        .get("fields", {})
    )
    for canon in ("name", "dob", "address", "phone"):
        if out.get(canon):
            continue
        f = adv.get(canon)
        if f:
            v = doc.get(f)
            if v not in (None, "", "None"):
                out[canon] = str(v).strip()

    return out


def _identity_key(canon: dict[str, str], doc_id: str) -> str:
    """
    Khoá danh tính để gom hồ sơ cùng 1 người.
    Ưu tiên định danh mạnh (CCCD/BHXH); nếu không có thì dùng tên + (NS hoặc SĐT).
    Không gom chỉ dựa trên SĐT vì nhiều người có thể chung 1 số liên hệ.
    """
    if canon.get("cccd"):
        digits = re.sub(r"\D", "", canon["cccd"])
        if digits:
            return f"cccd:{digits}"
    if canon.get("bhxh"):
        digits = re.sub(r"\D", "", canon["bhxh"])
        if digits:
            return f"bhxh:{digits}"
    name = canon.get("name", "").lower().strip()
    extra = canon.get("dob") or canon.get("phone") or ""
    if name:
        return f"np:{name}|{extra}"
    return f"id:{doc_id}"


def _consolidate(results_by_schema: dict[str, list[dict]], lookup_type: str) -> tuple[list[dict], str, list[str]]:
    """
    Gom kết quả từ nhiều schema thành các "hồ sơ tổng hợp" theo từng người.

    Returns: (persons_list, display_text, schemas_with_results)
    """
    # Flatten + tag schema
    flat: list[dict] = []
    for schema_name, docs in results_by_schema.items():
        for doc in docs:
            d = dict(doc)
            d["_id"] = str(d.get("_id", ""))
            d["_source_schema"] = schema_name
            flat.append(d)

    if not flat:
        return [], "", []

    # Group by identity
    groups: dict[str, list[tuple[dict, dict]]] = {}
    order: list[str] = []
    for doc in flat:
        schema = doc["_source_schema"]
        canon = _extract_canonical(schema, doc)
        key = _identity_key(canon, doc["_id"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((doc, canon))

    persons_out: list[dict] = []
    blocks: list[str] = []
    schemas_all: set[str] = set()

    for idx, key in enumerate(order, 1):
        members = groups[key]
        merged: dict[str, list[str]] = {}
        sources: list[str] = []

        for doc, canon in members:
            schema = doc.get("_source_schema", "")
            sources.append(schema)
            schemas_all.add(schema)
            for k, v in canon.items():
                if not v:
                    continue
                merged.setdefault(k, [])
                if v not in merged[k]:
                    merged[k].append(v)
            rec = _sanitize_record({k: v for k, v in doc.items() if k != "_source_schema"})
            # Giữ nguồn + gắn chỉ số nhóm người để frontend gộp card đồng bộ
            # với phần hiển thị text (mỗi nhóm = 1 người, có thể từ nhiều nguồn).
            rec["_source_schema"] = doc.get("_source_schema")
            rec["_person_group"] = idx
            persons_out.append(rec)

        blocks.append(_build_profile_block(idx, merged, members, sources, lookup_type))

    n_persons = len(order)
    n_records = len(flat)
    if n_persons == 1:
        header = f"✅ Tìm thấy **1 người** ({n_records} hồ sơ từ các nguồn dữ liệu):\n"
    else:
        header = f"✅ Tìm thấy **{n_persons} người** (tổng {n_records} hồ sơ):\n"

    display = header + "\n".join(blocks)
    return persons_out, display, sorted(schemas_all)


def _build_profile_block(
    idx: int,
    merged: dict[str, list[str]],
    members: list[tuple[dict, dict]],
    sources: list[str],
    lookup_type: str,
) -> str:
    lines = [f"{'═' * 34}", f"👤 HỒ SƠ #{idx}", f"{'═' * 34}"]

    # Thông tin tổng hợp (gộp giá trị từ mọi nguồn)
    for canon, label in _CANON_LABELS:
        vals = merged.get(canon)
        if vals:
            lines.append(f"  • {label}: {', '.join(vals)}")

    # Nguồn dữ liệu
    cnt = Counter(sources)
    src_parts = []
    for s, n in cnt.items():
        name = get_schema_display_name(s)
        src_parts.append(f"{name} ({n})" if n > 1 else name)
    lines.append(f"  • Nguồn dữ liệu: {', '.join(src_parts)}")

    # Chi tiết theo từng nguồn
    lines.append("  ── Chi tiết theo nguồn ──")
    for doc, _canon in members:
        schema = doc.get("_source_schema", "")
        cfg = (
            SEARCHABLE_COLLECTION_MAP.get(lookup_type, {})
            .get("collections", {})
            .get(schema, {})
        )
        clean = {k: v for k, v in doc.items() if k != "_source_schema"}
        detail = _build_display_text(clean, cfg.get("display_fields", []), schema)
        lines.append(f"  ▸ {get_schema_display_name(schema)}")
        if detail.strip():
            lines.append(detail)

    return "\n".join(lines)


# ============================================================================
# Low-level query (1 schema) — sync, chạy trong asyncio.to_thread()
# ============================================================================


def _query_single_schema_sync(
    collection_name: str,
    collection,
    query_fields: list[str],
    match_mode: str,
    values: list[str],
    limit: int,
) -> tuple[str, list[dict]]:
    """
    Truy vấn 1 schema (đồng bộ). Hỗ trợ nhiều giá trị qua $in.

    Raises MongoUnavailable nếu lỗi kết nối (để báo "hệ thống đang bận").
    """
    try:
        if match_mode == "exact":
            # Tìm chính xác trên field CÓ INDEX (CCCD/CMND/BHXH là chữ số).
            # Đưa thêm biến thể int để khớp khi DB lưu dạng số nguyên thay vì chuỗi.
            # KHÔNG dùng regex 'i' (case-insensitive) vì nó vô hiệu hoá index → collscan.
            in_vals: list = []
            for v in values:
                in_vals.append(v)
                if v.isdigit():
                    try:
                        in_vals.append(int(v))
                    except (ValueError, OverflowError):
                        pass
            q = {"$or": [{field: {"$in": in_vals}} for field in query_fields]}
            docs = collection.find(q).max_time_ms(QUERY_MAX_MS).limit(limit).to_list(length=limit)
            return collection_name, docs

        elif match_mode == "regex":
            # Substring, không phân biệt hoa thường (escape để tránh lỗi regex)
            q = {
                "$or": [
                    {field: {"$regex": re.escape(v), "$options": "i"}}
                    for field in query_fields
                    for v in values
                ]
            }
            docs = collection.find(q).max_time_ms(QUERY_MAX_MS).limit(limit).to_list(length=limit)
            return collection_name, docs

        elif match_mode == "phone":
            norm = [_normalize_phone(v) for v in values]
            q = {"$or": [{field: {"$in": norm}} for field in query_fields]}
            docs = collection.find(q).max_time_ms(QUERY_MAX_MS).limit(limit).to_list(length=limit)
            return collection_name, docs

        else:
            return collection_name, []

    except ExecutionTimeout:
        # Query quá chậm (thường do thiếu index) — bỏ qua schema này, không coi là lỗi hệ thống.
        logger.warning(
            f"[_query_single_schema] ⏱️ TIMEOUT {collection_name}.{query_fields} "
            f"(>{QUERY_MAX_MS}ms — có thể thiếu index) → bỏ qua"
        )
        return collection_name, []
    except _CONN_ERRORS as e:
        logger.error(f"[mongo] connection error on {collection_name}: {e}")
        raise MongoUnavailable(str(e)) from e
    except PyMongoError as e:
        # Lỗi truy vấn (vd regex/operation) — không phải mất kết nối: bỏ qua schema này
        logger.warning(f"[_query_single_schema] {collection_name}.{query_fields}: {e}")
        return collection_name, []
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[_query_single_schema] {collection_name}.{query_fields}: {e}")
        return collection_name, []


async def _gather_schemas(tasks: list) -> dict[str, list[dict]]:
    """
    Chạy song song các task query schema, gom kết quả.

    - Dùng gather(return_exceptions=True) để mọi exception đều được "retrieve"
      (tránh cảnh báo 'Task exception was never retrieved').
    - Mất kết nối toàn bộ (và không lấy được dữ liệu nào) → raise MongoUnavailable.
    - Mất kết nối một phần nhưng vẫn có schema trả dữ liệu → trả phần lấy được.
    """
    results_by_schema: dict[str, list[dict]] = {}
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    unavailable = False
    for oc in outcomes:
        if isinstance(oc, MongoUnavailable):
            unavailable = True
            continue
        if isinstance(oc, Exception):
            logger.warning(f"[_gather_schemas] task error: {oc}")
            continue
        schema_name, docs = oc
        if docs:
            results_by_schema.setdefault(schema_name, []).extend(docs)

    if unavailable and not results_by_schema:
        raise MongoUnavailable("mongodb unavailable")
    return results_by_schema


# ============================================================================
# Multi-schema search (exact/regex/phone) — nhiều giá trị
# ============================================================================


async def _search_multi(
    lookup_type: str,
    values: list[str],
    match_mode: str = "exact",  # "exact" | "regex" | "phone"
    per_schema_limit: int = 10,
):
    """
    Tìm trên tất cả schema có hỗ trợ `lookup_type`, gom về hồ sơ tổng hợp.
    Yields tối đa 1 kết quả tìm thấy, hoặc 1 kết quả "busy" khi mất kết nối.
    """
    if lookup_type not in SEARCHABLE_COLLECTION_MAP:
        yield {
            "found": False,
            "persons": [],
            "display": f"Không hỗ trợ lookup type: {lookup_type}",
            "lookup_type": lookup_type,
        }
        return

    schema_map = SEARCHABLE_COLLECTION_MAP[lookup_type]["collections"]

    try:
        db = get_mongo_db()
    except _CONN_ERRORS as e:
        logger.error(f"[_search_multi] cannot get db: {e}")
        yield _busy_result(lookup_type)
        return

    tasks = []
    for schema_name, cfg in schema_map.items():
        query_fields = cfg.get("fields", [])
        if isinstance(query_fields, dict):  # advanced map shape — skip here
            query_fields = list(query_fields.values())
        if not query_fields:
            continue
        collection = db[schema_name]
        tasks.append(
            asyncio.to_thread(
                _query_single_schema_sync,
                schema_name,
                collection,
                query_fields,
                match_mode,
                values,
                per_schema_limit,
            )
        )

    try:
        results_by_schema = await _gather_schemas(tasks)
    except MongoUnavailable:
        yield _busy_result(lookup_type)
        return

    persons, display, schemas = _consolidate(results_by_schema, lookup_type)
    if persons:
        yield {
            "found": True,
            "persons": persons,
            "display": display,
            "schemas": schemas,
            "lookup_type": lookup_type,
        }


# ============================================================================
# Advanced (Multi-field AND) search
# ============================================================================


def _query_single_schema_advanced_sync(
    collection_name: str,
    collection,
    query_fields_dict: dict[str, str],
    criteria: dict[str, str],
    limit: int,
) -> tuple[str, list[dict]]:
    """Truy vấn AND nhiều trường trên 1 schema. Raises MongoUnavailable nếu mất kết nối."""
    try:
        and_conditions = []
        for key, value in criteria.items():
            if not value:
                continue
            mapped_field = query_fields_dict.get(key)
            if not mapped_field:
                continue

            if key == "phone":
                and_conditions.append({mapped_field: _normalize_phone(value)})
            elif key == "name":
                # Khớp CHÍNH XÁC tên (tránh "Nguyễn Văn A" khớp "Nguyễn Văn Anh")
                and_conditions.append(
                    {mapped_field: {"$regex": f"^{re.escape(value)}$", "$options": "i"}}
                )
            else:
                # dob/address: khớp 'chứa' vì người dùng có thể nhập một phần
                and_conditions.append(
                    {mapped_field: {"$regex": re.escape(value), "$options": "i"}}
                )

        if not and_conditions:
            return collection_name, []

        and_query = {"$and": and_conditions}
        docs = collection.find(and_query).max_time_ms(QUERY_MAX_MS).limit(limit).to_list(length=limit)
        return collection_name, docs

    except ExecutionTimeout:
        logger.warning(
            f"[_query_single_schema_advanced] ⏱️ TIMEOUT {collection_name} "
            f"(>{QUERY_MAX_MS}ms — có thể thiếu index) → bỏ qua"
        )
        return collection_name, []
    except _CONN_ERRORS as e:
        logger.error(f"[mongo] connection error on {collection_name}: {e}")
        raise MongoUnavailable(str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[_query_single_schema_advanced] {collection_name}: {e}")
        return collection_name, []


async def _search_multi_advanced(criteria: dict[str, str], per_schema_limit: int = 10):
    lookup_type = "advanced"
    if lookup_type not in SEARCHABLE_COLLECTION_MAP:
        yield {
            "found": False,
            "persons": [],
            "display": f"Không hỗ trợ lookup type: {lookup_type}",
            "lookup_type": lookup_type,
        }
        return

    schema_map = SEARCHABLE_COLLECTION_MAP[lookup_type]["collections"]

    try:
        db = get_mongo_db()
    except _CONN_ERRORS as e:
        logger.error(f"[_search_multi_advanced] cannot get db: {e}")
        yield _busy_result(lookup_type)
        return

    tasks = []
    for schema_name, cfg in schema_map.items():
        query_fields_dict = cfg.get("fields", {})
        if not query_fields_dict:
            continue
        supported = [k for k in criteria if k in query_fields_dict and criteria[k]]
        if not supported:
            continue
        collection = db[schema_name]
        tasks.append(
            asyncio.to_thread(
                _query_single_schema_advanced_sync,
                schema_name,
                collection,
                query_fields_dict,
                criteria,
                per_schema_limit,
            )
        )

    if not tasks:
        yield {
            "found": False,
            "persons": [],
            "display": "Không có collection nào hỗ trợ các trường tìm kiếm này.",
            "lookup_type": lookup_type,
        }
        return

    try:
        results_by_schema = await _gather_schemas(tasks)
    except MongoUnavailable:
        yield _busy_result(lookup_type)
        return

    persons, display, schemas = _consolidate(results_by_schema, lookup_type)
    if persons:
        yield {
            "found": True,
            "persons": persons,
            "display": display,
            "schemas": schemas,
            "lookup_type": lookup_type,
        }


# ============================================================================
# Value extraction — trích nhiều giá trị từ câu hỏi người dùng
# ============================================================================


def _extract_numbers(text: str, valid_lengths: tuple[int, ...] | None = None, min_len: int = 0) -> list[str]:
    """
    Trích các nhóm chữ số từ `text` (đã bỏ phân tách như khoảng trắng/dấu chấm
    bên trong một nhóm), khử trùng lặp, giữ thứ tự.

    - valid_lengths: nếu có, chỉ giữ nhóm có độ dài nằm trong tập này.
    - min_len: độ dài tối thiểu (khi không dùng valid_lengths).
    """
    # Cho phép số có dấu phân tách bên trong (vd "0912 345 678"): gộp các cụm số
    # nối nhau bởi khoảng trắng/dấu . - thành 1 nhóm trước khi tách.
    joined = re.sub(r"(?<=\d)[\s\.\-]+(?=\d)", "", text)
    raw_groups = re.findall(r"\d+", joined)
    out: list[str] = []
    for g in raw_groups:
        if valid_lengths is not None:
            if len(g) not in valid_lengths:
                continue
        elif len(g) < min_len:
            continue
        if g not in out:
            out.append(g)
    return out


def _limit_for(n_values: int, base: int = 10) -> int:
    return base if n_values <= 1 else min(100, n_values * base)


# ============================================================================
# Public API — 1 function per intent
# ============================================================================


async def search_by_cccd(cccd: str):
    """
    Tìm theo (một hoặc NHIỀU) số CCCD/CMND trên tất cả schema.
    Tự trích các nhóm 9 hoặc 12 chữ số từ chuỗi đầu vào.
    """
    values = _extract_numbers(cccd, valid_lengths=(9, 12))
    if not values:
        yield {
            "found": False,
            "persons": [],
            "display": "Không tìm thấy số CCCD hợp lệ (cần 9 hoặc 12 chữ số).",
        }
        return

    logger.info(f"[search_by_cccd] Searching CCCD: {values}")
    found_any = False
    async for result in _search_multi(
        "cccd", values, match_mode="exact", per_schema_limit=_limit_for(len(values))
    ):
        if result.get("error"):
            yield result
            return
        if result.get("found"):
            found_any = True
        yield result

    if not found_any:
        joined = ", ".join(values)
        yield {"found": False, "persons": [], "display": f"Không tìm thấy người có CCCD: {joined}"}


async def search_by_name(name: str, limit: int = 10):
    """Tìm theo tên (regex, không phân biệt hoa thường)."""
    name = (name or "").strip()
    if len(name) < 2:
        yield {"found": False, "persons": [], "display": "Tên tìm kiếm quá ngắn."}
        return

    logger.info(f"[search_by_name] Searching name: {name!r}")
    found_any = False
    async for result in _search_multi(
        "name", [name], match_mode="regex", per_schema_limit=limit
    ):
        if result.get("error"):
            yield result
            return
        if result.get("found"):
            found_any = True
        yield result

    if not found_any:
        yield {"found": False, "persons": [], "display": f"Không tìm thấy người có tên: {name}"}


async def search_by_bhxh(so_bhxh: str):
    """Tìm theo (một hoặc NHIỀU) số BHXH. Trích các nhóm ≥ 5 chữ số."""
    values = _extract_numbers(so_bhxh, min_len=5)
    if not values:
        yield {
            "found": False,
            "persons": [],
            "display": "Không tìm thấy số BHXH hợp lệ (cần ít nhất 5 chữ số).",
        }
        return

    logger.info(f"[search_by_bhxh] Searching BHXH: {values}")
    found_any = False
    async for result in _search_multi(
        "bhxh", values, match_mode="exact", per_schema_limit=_limit_for(len(values))
    ):
        if result.get("error"):
            yield result
            return
        if result.get("found"):
            found_any = True
        yield result

    if not found_any:
        joined = ", ".join(values)
        yield {"found": False, "persons": [], "display": f"Không tìm thấy người có số BHXH: {joined}"}


async def search_by_phone(phone: str, limit: int = 10):
    """Tìm theo (một hoặc NHIỀU) số điện thoại. Trích các nhóm đúng 10 chữ số."""
    values = _extract_numbers(phone, valid_lengths=(10,))
    if not values:
        yield {
            "found": False,
            "persons": [],
            "display": "Không tìm thấy số điện thoại hợp lệ (cần đúng 10 chữ số).",
        }
        return

    logger.info(f"[search_by_phone] Searching phone: {values}")
    found_any = False
    async for result in _search_multi(
        "phone", values, match_mode="phone", per_schema_limit=_limit_for(len(values))
    ):
        if result.get("error"):
            yield result
            return
        if result.get("found"):
            found_any = True
        yield result

    if not found_any:
        joined = ", ".join(values)
        yield {"found": False, "persons": [], "display": f"Không tìm thấy người có số điện thoại: {joined}"}


async def search_by_advanced(criteria: dict[str, str], limit: int = 10):
    """Tìm theo tổ hợp tiêu chí (name + dob + address + phone)."""
    clean_criteria = {k: str(v).strip() for k, v in criteria.items() if v and str(v).strip()}
    if not clean_criteria:
        yield {"found": False, "persons": [], "display": "Không có tiêu chí tìm kiếm hợp lệ."}
        return

    logger.info(f"[search_by_advanced] Searching: {clean_criteria}")
    found_any = False
    async for result in _search_multi_advanced(clean_criteria, per_schema_limit=limit):
        if result.get("error"):
            yield result
            return
        if result.get("found"):
            found_any = True
        yield result

    if not found_any:
        yield {
            "found": False,
            "persons": [],
            "display": f"Không tìm thấy người khớp với thông tin: {clean_criteria}",
        }

"""
PII redaction for agent traces (people-search data).
====================================================

People search (MongoDB, see app/services/agents/people_agent.py + the mongo_*
tools) returns real citizen data — CCCD/CMND, BHXH numbers, phone numbers, full
names, dates of birth, addresses. Before a trace is persisted as training data
this module masks that PII. Applied centrally in ``AgentTraceService.record``
so no write path can bypass it.

Strategy (defence in depth):
  1. Structural — any dict key that names a PII field has its value masked.
  2. Textual    — long digit runs (CCCD/CMND/BHXH) and VN phone numbers are
                  masked wherever they appear (free-text answers, tool summaries).

Masking keeps a short suffix (e.g. ``*********123``) so records stay
distinguishable/debuggable without exposing the identifier.
"""

from __future__ import annotations

import re
from typing import Any

# Keys whose values are PII regardless of content. Lower-cased, matched as a
# substring so "so_cccd", "cccd_number", "ho_ten_day_du" etc. all hit.
_PII_KEY_PARTS = (
    "cccd",
    "cmnd",
    "bhxh",
    "so_the",
    "the_bhyt",
    "ho_ten",
    "hoten",
    "full_name",
    "fullname",
    "ten_khai_sinh",
    "phone",
    "sdt",
    "dien_thoai",
    "so_dien_thoai",
    "ngay_sinh",
    "ngaysinh",
    "dob",
    "birth",
    "dia_chi",
    "diachi",
    "address",
    "que_quan",
    "noi_o",
    "email",
)

# 9–12 digit identifiers (CMND=9/12, CCCD=12, BHXH=10). Keep last 3.
_NUM_ID_RE = re.compile(r"\b\d{9,12}\b")
# Vietnamese phone numbers: 0xxxxxxxxx or +84xxxxxxxxx. Keep last 3.
_PHONE_RE = re.compile(r"\b(?:\+?84|0)\d{8,10}\b")


def _mask_digits(s: str) -> str:
    keep = 3 if len(s) > 3 else 0
    return "*" * (len(s) - keep) + s[len(s) - keep :]


def redact_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    text = _PHONE_RE.sub(lambda m: _mask_digits(m.group(0)), text)
    text = _NUM_ID_RE.sub(lambda m: _mask_digits(m.group(0)), text)
    return text


def _is_pii_key(key: str) -> bool:
    k = key.lower()
    return any(part in k for part in _PII_KEY_PARTS)


def redact_obj(obj: Any) -> Any:
    """Recursively redact PII from an arbitrary JSON-serialisable structure."""
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if isinstance(k, str) and _is_pii_key(k):
                out[k] = "[REDACTED]" if v not in (None, "", [], {}) else v
            else:
                out[k] = redact_obj(v)
        return out
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


def has_residual_pii(obj: Any) -> bool:
    """Heuristic post-check: True if a numeric identifier survived redaction.

    Used by the export script as a tripwire — it should normally return False.
    """
    if isinstance(obj, dict):
        return any(has_residual_pii(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_residual_pii(v) for v in obj)
    if isinstance(obj, str):
        return bool(_NUM_ID_RE.search(obj) or _PHONE_RE.search(obj))
    return False

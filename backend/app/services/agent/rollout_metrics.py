"""Append-only rollout metrics recorder (Phase 3, Task 7B).

Live metrics carry the arm, hashed request/workspace IDs, timing, terminal
status, citation count, cancellation state, and the four authoritative
security counters. R72: ``grounded_quality`` is NEVER compared or recorded
here — v2 grounded-success completeness is an invariant, not an arm-neutral
quality measure.

Security contract: missing/default security fields are INVALID, never
"safe" — :func:`validate_security_flags` raises on anything but four
explicit booleans, and :func:`record_rollout_metric` refuses to write
without them. There is deliberately NO update/delete helper in this module
(append-only, enforced additionally by a DB trigger).
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: The four authoritative security counters (T7B metric column suffixes).
SECURITY_FIELDS: tuple[str, ...] = (
    "checkpoint_secret",
    "ungrounded_factual_success",
    "acl_leak",
    "duplicate_production_write",
)

VALID_ARMS: frozenset[str] = frozenset({"v1", "v2", "shadow"})


def hash_rollout_id(value: Any) -> str | None:
    """sha256-hash one request/workspace id for metrics (``None`` stays)."""
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_security_flags(security: Mapping[str, Any] | None) -> dict[str, bool]:
    """Validate the four authoritative security counters.

    Every counter must be an explicit ``bool``. Missing keys, ``None``, or
    non-boolean values raise ``ValueError`` — a missing/default field is
    INVALID and must never be interpreted (or stored) as "safe".
    """
    if security is None or not isinstance(security, Mapping):
        raise ValueError(
            "rollout security counters are required "
            f"(missing for {sorted(SECURITY_FIELDS)}); refusing to record"
        )
    validated: dict[str, bool] = {}
    missing = [name for name in SECURITY_FIELDS if name not in security]
    if missing:
        raise ValueError(
            f"rollout security counters missing {missing}; "
            "missing fields are invalid, never safe"
        )
    for name in SECURITY_FIELDS:
        value = security[name]
        if not isinstance(value, bool):
            raise ValueError(
                f"rollout security counter {name!r} must be bool, "
                f"got {value!r}; refusing to record"
            )
        validated[name] = value
    return validated


def _security_producer(name: str, *, value: bool) -> bool:
    """Shared recipe for the four authoritative counter producers.

    Each named producer below is the single module-level entry point that
    maps one terminal observation to its counter. They take the detector's
    explicit boolean verdict — there is no default, no inference, and no
    "assume safe" path; unobserved stays unrecorded (the recorder raises).
    """
    if not isinstance(value, bool):
        raise ValueError(
            f"authoritative security verdict for {name!r} must be bool, "
            f"got {value!r}"
        )
    return value


def security_checkpoint_secret_from_terminal(*, violated: bool) -> bool:
    """Authoritative producer: checkpoint-secret exposure observed or not."""
    return _security_producer("checkpoint_secret", value=violated)


def security_ungrounded_factual_success_from_terminal(*, violated: bool) -> bool:
    """Authoritative producer: ungrounded factual success observed or not."""
    return _security_producer("ungrounded_factual_success", value=violated)


def security_acl_leak_from_terminal(*, violated: bool) -> bool:
    """Authoritative producer: ACL leak observed or not."""
    return _security_producer("acl_leak", value=violated)


def security_duplicate_production_write_from_terminal(*, violated: bool) -> bool:
    """Authoritative producer: duplicate production write observed or not."""
    return _security_producer("duplicate_production_write", value=violated)


def build_metric_payload(
    *,
    arm: str,
    request_id_hash: str,
    workspace_id_hash: str | None,
    started_at: datetime,
    finished_at: datetime | None,
    duration_ms: int | None,
    terminal_status: str,
    citation_count: int,
    cancelled: bool,
    security: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate + build one metric row payload (pure; no DB access)."""
    if arm not in VALID_ARMS:
        raise ValueError(
            f"rollout metric arm must be one of {sorted(VALID_ARMS)}; got {arm!r}"
        )
    if not request_id_hash:
        raise ValueError("rollout metric request_id_hash is required")
    if not terminal_status:
        raise ValueError("rollout metric terminal_status is required")
    flags = validate_security_flags(security)
    return {
        "arm": arm,
        "request_id_hash": request_id_hash,
        "workspace_id_hash": workspace_id_hash,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "terminal_status": terminal_status,
        "citation_count": int(citation_count or 0),
        "cancelled": bool(cancelled),
        "security_checkpoint_secret": flags["checkpoint_secret"],
        "security_ungrounded_factual_success": flags[
            "ungrounded_factual_success"
        ],
        "security_acl_leak": flags["acl_leak"],
        "security_duplicate_production_write": flags[
            "duplicate_production_write"
        ],
    }


async def record_rollout_metric(db: Any, **fields: Any) -> Any:
    """Append one metric row (INSERT only — never update, never delete)."""
    from app.models.agent_rollout_metric import AgentRolloutMetric

    payload = build_metric_payload(
        arm=fields.get("arm"),
        request_id_hash=fields.get("request_id_hash"),
        workspace_id_hash=fields.get("workspace_id_hash"),
        started_at=fields.get("started_at") or datetime.now(timezone.utc),
        finished_at=fields.get("finished_at"),
        duration_ms=fields.get("duration_ms"),
        terminal_status=fields.get("terminal_status"),
        citation_count=fields.get("citation_count", 0),
        cancelled=fields.get("cancelled", False),
        security=fields.get("security"),
    )
    row = AgentRolloutMetric(**payload)
    db.add(row)
    try:
        await db.commit()
    except Exception:
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001 — rollback best effort
            pass
        raise
    return row

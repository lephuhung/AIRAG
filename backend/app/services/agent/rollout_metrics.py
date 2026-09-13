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
import re
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


# ---------------------------------------------------------------------------
# Real terminal detectors (Task 7B fix round 1, R74).
#
# Each detector maps an explicit terminal observation to its counter's
# verdict. They return ``bool | None``: ``None`` means the verdict is
# UNOBSERVABLE from the supplied terminal data — callers MUST treat that
# as invalid (never "safe") and refuse to record the row. These are the
# authoritative producers' inputs: :func:`build_security_verdicts` feeds
# detector outputs through the four named producers above, so the
# producers are used by application code, not just defined.
# ---------------------------------------------------------------------------

_SECRET_MARKERS = (
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(r"sk-(?:live|test)-[A-Za-z0-9]{8,}"),
    re.compile(r"xox[bpas]-[A-Za-z0-9-]{8,}"),
    re.compile(r"checkpoint[_-]?secret", re.IGNORECASE),
)

_SUCCESS_STATUSES = frozenset({"success", "complete", "completed", "ok"})
_CANCELLED_STATUSES = frozenset({"cancelled", "cancelling", "canceled"})


def detect_checkpoint_secret(answer_text: str | None) -> bool | None:
    """Verdict from the terminal answer text (``None`` = unobservable)."""
    if answer_text is None:
        return None
    if not isinstance(answer_text, str):
        return None
    return any(pattern.search(answer_text) for pattern in _SECRET_MARKERS)


def detect_ungrounded_factual_success(
    *,
    terminal_status: str | None,
    citation_count: int | None,
    factual_expected: bool | None,
) -> bool | None:
    """Factual success with zero citations is a violation (v2 invariant).

    ``factual_expected`` states whether the turn was a factual answer turn
    (observed from the serving path: v2 factual routes expect grounding;
    turns that served no sources are non-factual). ``None`` anywhere means
    the verdict is unobservable — invalid, never safe.
    """
    if terminal_status is None or citation_count is None or factual_expected is None:
        return None
    if not factual_expected:
        return False
    if str(terminal_status).lower() not in _SUCCESS_STATUSES:
        return False
    try:
        cites = int(citation_count)
    except (TypeError, ValueError):
        return None
    return cites <= 0


def detect_acl_leak(
    *,
    served_document_ids: list[Any] | tuple[Any, ...] | None,
    allowed_document_ids: list[Any] | tuple[Any, ...] | None,
    scope_bound: bool = False,
) -> bool | None:
    """Any served document outside the allowed set is a leak.

    Both inputs must be explicit observations (the terminal's served
    sources and the request's allowed set). Either missing → unobservable,
    EXCEPT when the caller attests ``scope_bound``: the ingress resolved
    the runtime scope to the authenticated workspaces and retrieval only
    serves in-scope documents, so an existing served observation with no
    narrower document filter is an observed clean verdict (scope-bound
    retrieval), not a default. An empty served set is likewise observed
    clean. A missing served observation is always unobservable.
    """
    if served_document_ids is None:
        return None
    try:
        served = [str(item) for item in served_document_ids]
    except Exception:
        return None
    if allowed_document_ids is None:
        if scope_bound:
            return False
        return None
    try:
        allowed = {str(item) for item in allowed_document_ids}
    except Exception:
        return None
    return any(item not in allowed for item in served)


def detect_duplicate_production_write(
    production_write_count: int | None,
) -> bool | None:
    """More than one production write for one request is a violation."""
    if production_write_count is None:
        return None
    try:
        count = int(production_write_count)
    except (TypeError, ValueError):
        return None
    return count > 1


def build_security_verdicts(
    *,
    answer_text: str | None,
    terminal_status: str | None,
    citation_count: int | None,
    factual_expected: bool | None,
    served_document_ids: list[Any] | tuple[Any, ...] | None,
    allowed_document_ids: list[Any] | tuple[Any, ...] | None,
    production_write_count: int | None,
    scope_bound: bool = False,
) -> dict[str, bool]:
    """Run the four real detectors and validate through the producers.

    Raises ``ValueError`` when ANY verdict is unobservable — the row is
    invalid and must never be recorded as "safe". Every returned verdict
    passes through its named authoritative producer.
    """
    raw: dict[str, bool | None] = {
        "checkpoint_secret": detect_checkpoint_secret(answer_text),
        "ungrounded_factual_success": detect_ungrounded_factual_success(
            terminal_status=terminal_status,
            citation_count=citation_count,
            factual_expected=factual_expected,
        ),
        "acl_leak": detect_acl_leak(
            served_document_ids=served_document_ids,
            allowed_document_ids=allowed_document_ids,
            scope_bound=scope_bound,
        ),
        "duplicate_production_write": detect_duplicate_production_write(
            production_write_count
        ),
    }
    unobserved = sorted(name for name, value in raw.items() if value is None)
    if unobserved:
        raise ValueError(
            "rollout security verdicts unobservable for "
            f"{unobserved}; refusing to record an unobservable row as safe"
        )
    return {
        "checkpoint_secret": security_checkpoint_secret_from_terminal(
            violated=bool(raw["checkpoint_secret"])
        ),
        "ungrounded_factual_success": security_ungrounded_factual_success_from_terminal(
            violated=bool(raw["ungrounded_factual_success"])
        ),
        "acl_leak": security_acl_leak_from_terminal(
            violated=bool(raw["acl_leak"])
        ),
        "duplicate_production_write": security_duplicate_production_write_from_terminal(
            violated=bool(raw["duplicate_production_write"])
        ),
    }


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


async def emit_terminal_rollout_metric(
    db: Any,
    *,
    arm: str,
    request_id: Any,
    workspace_ids: Any,
    started_at: datetime,
    finished_at: datetime | None = None,
    terminal_status: str,
    citation_count: int = 0,
    cancelled: bool = False,
    answer_text: str | None,
    factual_expected: bool | None,
    served_document_ids: list[Any] | tuple[Any, ...] | None,
    allowed_document_ids: list[Any] | tuple[Any, ...] | None,
    production_write_count: int | None,
    scope_bound: bool = False,
) -> Any:
    """Terminal-boundary append-only emission (R74; v1 + v2 arms).

    Runs the four real detectors over the terminal observation, feeds the
    verdicts through the authoritative producers, hashes the request and
    workspace ids, and appends exactly one row. Raises ``ValueError`` when
    any verdict is unobservable — the row is invalid and is never recorded
    as "safe". Callers that must never break serving use
    :func:`try_emit_terminal_rollout_metric`.
    """
    security = build_security_verdicts(
        answer_text=answer_text,
        terminal_status=terminal_status,
        citation_count=citation_count,
        factual_expected=factual_expected,
        served_document_ids=served_document_ids,
        allowed_document_ids=allowed_document_ids,
        production_write_count=production_write_count,
        scope_bound=scope_bound,
    )
    scope = sorted({str(item) for item in (workspace_ids or ()) if str(item)})
    finished = finished_at or datetime.now(timezone.utc)
    try:
        duration_ms = max(0, int((finished - started_at).total_seconds() * 1000))
    except Exception:
        duration_ms = None
    return await record_rollout_metric(
        db,
        arm=arm,
        request_id_hash=hash_rollout_id(request_id),
        workspace_id_hash=hash_rollout_id(scope[0]) if scope else None,
        started_at=started_at,
        finished_at=finished,
        duration_ms=duration_ms,
        terminal_status=terminal_status,
        citation_count=citation_count,
        cancelled=cancelled,
        security=security,
    )


async def try_emit_terminal_rollout_metric(db: Any, **fields: Any) -> Any | None:
    """Best-effort terminal emission: never breaks serving (logs + None)."""
    try:
        return await emit_terminal_rollout_metric(db, **fields)
    except Exception:
        logger.warning("[rollout] terminal metric emission skipped", exc_info=True)
        return None


_CITATION_MARKER_RE = re.compile(
    r"\[(?:[A-Za-z0-9]{4}|MEM-[A-Za-z0-9]+|IMG-[A-Za-z0-9-]+)\]"
)

_SCOPE_ID_KEYS = ("workspace_id", "knowledge_base_id", "kb_id")
_DOC_ID_KEYS = ("document_id", "id")


def count_citation_markers(answer_text: str | None) -> int:
    """Count inline citation markers in a terminal answer (observed)."""
    if not answer_text or not isinstance(answer_text, str):
        return 0
    return len(_CITATION_MARKER_RE.findall(answer_text))


def extract_served_document_ids(sources: Any) -> list[str]:
    """Extract served document/scope ids from terminal sources (observed).

    Prefers scope keys (workspace/KB) when present, else document ids.
    Unparseable entries are skipped; a turn that served nothing yields []
    (observed empty, never a default "safe").
    """
    served: list[str] = []
    try:
        items = tuple(sources or ())
    except TypeError:
        return []
    for item in items:
        try:
            slot = (
                item.get if isinstance(item, dict) else getattr
            )
            found = None
            for key in _SCOPE_ID_KEYS + _DOC_ID_KEYS:
                try:
                    value = slot(key) if isinstance(item, dict) else slot(item, key, None)
                except Exception:
                    value = None
                if value:
                    found = str(value)
                    break
            if found:
                served.append(found)
        except Exception:
            continue
    return served

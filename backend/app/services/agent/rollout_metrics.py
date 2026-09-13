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

#: Sentinel ``terminal_status`` marking an explicitly-INVALID metric row
#: (Task 7B fix round 3, R80). An unobservable security verdict MUST NOT
#: be omitted (omission evades the gate): the emission writes the row with
#: this terminal and explicit ``False`` counters, and the collector/gate
#: counts every such row as an invalid security row (arm invalid). The
#: value fits the existing frozen v3 schema — no new migration.
SECURITY_UNOBSERVABLE_TERMINAL = "security_unobservable"

#: Counters written on a sentinel row: explicit booleans (so the row
#: itself validates) whose ``False`` values MUST be ignored by consumers
#: — the ``terminal_status`` sentinel alone carries the invalid meaning.
_SENTINEL_SECURITY: dict[str, bool] = {
    "checkpoint_secret": False,
    "ungrounded_factual_success": False,
    "acl_leak": False,
    "duplicate_production_write": False,
}


def effective_cancelled(*, cancelled: bool, cancel_requested: bool = False) -> bool:
    """Resolve the row's ``cancelled`` flag (Task 7B fix round 3, R81).

    ``cancelled`` means cancellation was REQUESTED for the run,
    regardless of how the turn terminated; ``terminal_status`` records the
    actual terminal. Failed cancellation = requested + non-cancelled
    terminal (the collector's invariant); a cancelled request that exits
    by cancellation is a success. Pure (testable) — ingresses supply
    ``cancel_requested`` from :func:`was_cancel_requested`.
    """
    return bool(cancelled) or bool(cancel_requested)


async def was_cancel_requested(run_id: Any) -> bool:
    """True when cancellation was requested for ``run_id`` (R81).

    Checks the distributed active-run registry (local set first, then
    Redis when enabled) via the single ``TaskScheduler`` ownership chain.
    Best-effort: any error (or an empty run id) maps to ``False`` — never
    breaks serving, never fabricates a cancellation.
    """
    if not run_id:
        return False
    try:
        from app.services.agents.v2.execution.scheduler import (
            is_run_cancel_requested_async as _cancel_requested,
        )

        return bool(await _cancel_requested(str(run_id)))
    except Exception:
        logger.warning(
            "[rollout] cancel-requested check failed", exc_info=True
        )
        return False


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
    unidentified_served_count: int | None = 0,
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

    Like-vs-like (fix round 2): served ids are document ids and the
    allowed set holds document ids. Served entries whose document id
    could not be determined (see
    :func:`count_served_sources_without_document_id`) cannot be proven
    inside a document allowlist: when such entries exist AND a document
    allowlist applies, the verdict is UNOBSERVABLE (``None`` — invalid,
    never safe, never a fabricated workspace-vs-document violation).
    """
    if served_document_ids is None:
        return None
    try:
        served = [str(item) for item in served_document_ids]
    except Exception:
        return None
    try:
        unidentified = int(unidentified_served_count or 0)
    except (TypeError, ValueError):
        return None
    if allowed_document_ids is None:
        if scope_bound:
            return False
        return None
    if unidentified > 0:
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
    unidentified_served_count: int | None = 0,
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
            unidentified_served_count=unidentified_served_count,
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
    cancel_requested: bool = False,
    answer_text: str | None,
    factual_expected: bool | None,
    served_document_ids: list[Any] | tuple[Any, ...] | None,
    allowed_document_ids: list[Any] | tuple[Any, ...] | None,
    production_write_count: int | None,
    scope_bound: bool = False,
    unidentified_served_count: int | None = 0,
    route: str | None = None,
    response_status: str | None = None,
    capability_call_count: int | None = None,
) -> Any:
    """Terminal-boundary append-only emission (R74; v1 + v2 arms).

    Runs the four real detectors over the terminal observation, feeds the
    verdicts through the authoritative producers, hashes the request and
    workspace ids, and appends exactly one row. ``cancelled`` is resolved
    through :func:`effective_cancelled` with ``cancel_requested`` (R81:
    requested-but-ineffective cancellations are recorded). When any
    verdict is unobservable the row is STILL written — with
    ``terminal_status`` set to :data:`SECURITY_UNOBSERVABLE_TERMINAL`
    (R80: omission is not acceptable; the collector/gate counts the
    sentinel as an invalid security row). The row is never recorded as
    "safe". P0 Task 6: when the optional execution telemetry (``route`` /
    ``response_status`` / ``capability_call_count``) classifies the turn as
    a factual zero-dispatch regression, the stored ``terminal_status`` is
    :data:`FACTUAL_ZERO_DISPATCH_TERMINAL` so the collector counts it as
    an error for rollout gates (spec section 7.2). All three default to
    ``None`` (unobservable → stored terminal unchanged), so existing
    callers see byte-identical behavior. Callers that must never break
    serving use :func:`try_emit_terminal_rollout_metric`.
    """
    row_cancelled = effective_cancelled(
        cancelled=cancelled, cancel_requested=cancel_requested
    )
    try:
        security = build_security_verdicts(
            answer_text=answer_text,
            terminal_status=terminal_status,
            citation_count=citation_count,
            factual_expected=factual_expected,
            served_document_ids=served_document_ids,
            allowed_document_ids=allowed_document_ids,
            production_write_count=production_write_count,
            scope_bound=scope_bound,
            unidentified_served_count=unidentified_served_count,
        )
    except ValueError:
        # R80: unobservable verdict — write the explicitly-invalid row
        # (sentinel terminal, explicit counters) instead of omitting it.
        logger.warning(
            "[rollout] security verdict unobservable; "
            "recording explicitly-invalid sentinel row",
            exc_info=True,
        )
        security = dict(_SENTINEL_SECURITY)
        terminal_status = SECURITY_UNOBSERVABLE_TERMINAL
    if terminal_status != SECURITY_UNOBSERVABLE_TERMINAL:
        # P0 Task 6: execution-telemetry remap runs on the ORIGINAL
        # terminal (detectors above already ran on it). Missing telemetry
        # is unobservable → stored terminal unchanged, never fabricated.
        terminal_status = resolve_execution_terminal(
            terminal_status,
            route=route,
            response_status=response_status,
            capability_call_count=capability_call_count,
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
        cancelled=row_cancelled,
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

#: The v1 intent-classifier status detail for conversational turns, pushed
#: by ``app.services.agent.nodes.intent_classifier``
#: (``intent_labels["greeting"]``). The ingress observes this status event
#: to mark greeting turns non-factual; a drift test pins the label against
#: the classifier source so a rename fails closed loudly instead of
#: fabricating ungrounded violations for greetings.
V1_GREETING_INTENT_DETAIL = "Phân loại: Tin nhắn thông thường"

#: Non-factual v2 routes: the direct route is the codebase's own
#: "non-factual success boundary" and clarify routes ask questions.
_NON_FACTUAL_ROUTES = frozenset({"direct", "clarify"})

#: v2 research routes whose successful answers must be grounded (R72:
#: v2 grounded-success completeness is an invariant).
_FACTUAL_ROUTES = frozenset({"complex_research", "fast_domain"})

_DOC_ID_KEYS = ("document_id",)


def count_citation_markers(answer_text: str | None) -> int:
    """Count inline citation markers in a terminal answer (observed)."""
    if not answer_text or not isinstance(answer_text, str):
        return 0
    return len(_CITATION_MARKER_RE.findall(answer_text))


def _served_document_id_of(item: Any) -> str | None:
    """One entry's document id (like-vs-like: ``document_id`` only)."""
    try:
        if isinstance(item, dict):
            value = item.get("document_id")
        else:
            value = getattr(item, "document_id", None)
    except Exception:
        return None
    if value is None or value is False:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def extract_served_document_ids(sources: Any) -> list[str]:
    """Extract served DOCUMENT ids from terminal sources (observed).

    Like-vs-like (fix round 2): only ``document_id`` is extracted — never
    workspace/KB/scope keys, never ``doc``/``id``/chunk labels — because
    the ingress allowlist holds document ids and comparing a workspace id
    against document ids fabricates ACL violations. Entries without a
    document id contribute nothing here; count them with
    :func:`count_served_sources_without_document_id` (under a document
    filter they make the ACL verdict unobservable, never safe). A turn
    that served nothing yields [] (observed empty, never a default).
    """
    served: list[str] = []
    try:
        items = tuple(sources or ())
    except TypeError:
        return []
    for item in items:
        found = _served_document_id_of(item)
        if found:
            served.append(found)
    return served


def count_served_sources_without_document_id(sources: Any) -> int:
    """Count served entries with no determinable document id.

    ``0`` for no served sources (observed empty). A positive count with an
    applicable document allowlist makes the ACL verdict unobservable
    (invalid, never safe); without a filter, scope-bound retrieval still
    attests in-scope cleanliness.
    """
    try:
        items = tuple(sources or ())
    except TypeError:
        return 0
    unidentified = 0
    for item in items:
        if _served_document_id_of(item) is None:
            try:
                is_empty = not item
            except Exception:
                is_empty = False
            if not is_empty:
                unidentified += 1
    return unidentified


def resolve_factual_expected(
    *,
    arm: str,
    terminal_status: str | None,
    cancelled: bool = False,
    answer_text: str | None = None,
    response_status: str | None = None,
    route: str | None = None,
    greeting_observed: bool = False,
) -> bool | None:
    """Resolve whether a turn was a factual-answer turn (observed, never defaulted).

    Three-valued: ``True``/``False`` are observed verdicts; ``None`` means
    factual-ness is UNOBSERVABLE and the row is invalid (never safe).

    Observed non-factual (``False``): cancellation or a non-success
    terminal (no factual SUCCESS occurred — definitional, not a default),
    an empty answer, a v2 clarify response, a non-factual v2 route
    (``direct``/``clarify`` — the direct route is the codebase's own
    non-factual success boundary), or the v1 greeting-intent status.

    Observed factual (``True``): a successful answer turn on a factual
    path — v2 research routes / observed v2 success (R72: v2
    grounded-success completeness is an invariant, so a v2 success answer
    with zero grounding evidence is a violation, never safe), and v1
    success answers (fail-closed: v1 grounds through served sources and
    inline markers, so a sourceless v1 success answer is a violation).

    Unobservable (``None``): a v2 success-shaped answer whose route and
    response status are both unknown — the invariant cannot be checked,
    so the row is invalid rather than recorded safe.
    """
    if cancelled:
        return False
    status = str(terminal_status or "").lower()
    if status and status in _CANCELLED_STATUSES:
        return False
    if status and status not in _SUCCESS_STATUSES:
        return False
    if not (answer_text or "").strip():
        return False
    if str(response_status or "").lower() == "clarify":
        return False
    normalized_route = str(route or "").lower() or None
    if normalized_route in _NON_FACTUAL_ROUTES:
        return False
    if greeting_observed and str(arm or "").lower() == "v1":
        return False
    if str(arm or "").lower() == "v2":
        if str(response_status or "").lower() in _SUCCESS_STATUSES:
            return True
        if normalized_route in _FACTUAL_ROUTES:
            return True
        return None
    if str(arm or "").lower() == "v1":
        return True
    return None


def count_grounding_evidence(
    answer_text: str | None,
    citations: Any,
    served_document_ids: Any,
) -> int:
    """Count presented grounding evidence for one terminal answer.

    Folds the three observed terminal signals: v1-style inline citation
    markers (``[xxxx]``), v2 presented citations (``citation_id``/``label``
    entries from the terminal payload), and served document ids. Pure
    observation counting — factual-ness is resolved separately by
    :func:`resolve_factual_expected` and must never be derived from these
    signals (that fail-open shape is exactly what fix round 2 removes).
    """
    total = count_citation_markers(answer_text)
    try:
        presented = tuple(citations or ())
    except TypeError:
        presented = ()
    total += len(presented)
    try:
        served = tuple(served_document_ids or ())
    except TypeError:
        served = ()
    total += len(served)
    return total


# ---------------------------------------------------------------------------
# Factual-retrieval execution observability (P0 Task 6, spec section 7.2).
#
# A factual complex turn that reaches terminal state with zero capability
# calls is an internal regression (the pre-P0 failure shape: ~100 ms
# terminal, no plan, no capability call, typed ``insufficient``) unless it
# is an explicitly unsupported/denied route. The regression is recorded
# with the ``FACTUAL_ZERO_DISPATCH_TERMINAL`` sentinel terminal so the
# collector counts it as an error for rollout gates instead of a normal
# insufficient answer. The sentinel fits the existing frozen Text column —
# no migration. Typed unsupported/denied outcomes keep their typed
# outcome (``typed_non_execution`` — never remapped, never an error).
# ---------------------------------------------------------------------------

#: Sentinel ``terminal_status`` for a factual complex terminal answer that
#: dispatched zero capabilities (spec section 7.2 regression). The
#: collector counts every such row as an error; the gate fails sustained
#: zero-dispatch volume via its error-rate regression threshold.
FACTUAL_ZERO_DISPATCH_TERMINAL = "factual_zero_dispatch"

#: Typed outcomes that classify as explicitly non-executed without being a
#: regression: ``denied`` (permission/hard-scope violation — no dispatch by
#: design), ``unsupported``/``unavailable`` (typed dependency-unavailable —
#: dispatches nothing by design), ``clarify`` (user-input-pending suspend —
#: no execution owed yet), ``error`` (already gate-visible as an error).
_TYPED_NON_EXECUTION_OUTCOMES = frozenset(
    {"clarify", "denied", "unsupported", "unavailable", "error"}
)

#: Answer-bearing response statuses eligible for the zero-dispatch
#: regression: an answer was produced on a factual route with no execution
#: behind it. ``success`` with zero grounding is additionally an
#: ungrounded-factual-success security violation (detectors, unchanged);
#: ``insufficient`` with zero dispatch is the pre-P0 bug shape.
_ZERO_DISPATCH_ELIGIBLE_RESPONSES = frozenset({"success", "insufficient"})

#: Columns stored on a metric row (content-hygiene contract: nothing else
#: may persist — no answer text, chunk content, tokens, or secrets).
METRIC_STORED_KEYS: tuple[str, ...] = (
    "arm",
    "request_id_hash",
    "workspace_id_hash",
    "started_at",
    "finished_at",
    "duration_ms",
    "terminal_status",
    "citation_count",
    "cancelled",
    "security_checkpoint_secret",
    "security_ungrounded_factual_success",
    "security_acl_leak",
    "security_duplicate_production_write",
)


def classify_factual_execution(
    *,
    route: str | None,
    response_status: str | None,
    capability_call_count: int | None,
) -> str:
    """Classify one terminal turn's execution posture (pure).

    Returns exactly one of ``"executed"`` / ``"zero_dispatch_regression"``
    / ``"typed_non_execution"`` / ``"non_factual"`` / ``"unobservable"``.
    ``capability_call_count`` is the observed number of capability calls
    for the turn (counts only — never content). ``None`` anywhere
    unclassifiable means ``"unobservable"`` (invalid, never safe, never
    fabricated into a regression). Non-integer or negative counts raise
    ``ValueError`` (fail closed; ``bool`` is rejected like the gate's
    integer inputs).
    """
    if capability_call_count is None:
        return "unobservable"
    if isinstance(capability_call_count, bool) or not isinstance(
        capability_call_count, int
    ):
        raise ValueError(
            "capability_call_count must be an int, "
            f"got {capability_call_count!r}; refusing to classify"
        )
    if capability_call_count < 0:
        raise ValueError(
            "capability_call_count must be >= 0, "
            f"got {capability_call_count!r}; refusing to classify"
        )
    if response_status is None:
        return "unobservable"
    normalized_response = str(response_status).lower()
    if normalized_response in _TYPED_NON_EXECUTION_OUTCOMES:
        return "typed_non_execution"
    normalized_route = str(route or "").lower() or None
    if (
        normalized_route in _FACTUAL_ROUTES
        and normalized_response in _ZERO_DISPATCH_ELIGIBLE_RESPONSES
    ):
        if capability_call_count == 0:
            return "zero_dispatch_regression"
        return "executed"
    if normalized_route in _NON_FACTUAL_ROUTES:
        return "non_factual"
    return "unobservable"


def resolve_execution_terminal(
    terminal_status: str,
    *,
    route: str | None = None,
    response_status: str | None = None,
    capability_call_count: int | None = None,
) -> str:
    """Map the stored terminal through the execution classifier (pure).

    Returns :data:`FACTUAL_ZERO_DISPATCH_TERMINAL` only for an observed
    ``"zero_dispatch_regression"``; every other classification — including
    ``"unobservable"`` and invalid telemetry (``ValueError``) — returns
    ``terminal_status`` unchanged, so missing telemetry never alters a
    stored row and never fabricates a regression.
    """
    try:
        verdict = classify_factual_execution(
            route=route,
            response_status=response_status,
            capability_call_count=capability_call_count,
        )
    except ValueError:
        return terminal_status
    if verdict == "zero_dispatch_regression":
        return FACTUAL_ZERO_DISPATCH_TERMINAL
    return terminal_status


def admitted_retrieval_unit_count(value: int) -> int:
    """Return the admitted-use count carried by ``retrieved_unit_count``.

    ``DocumentRetrieveOutput.retrieved_unit_count`` counts admitted
    ``EvidenceUse`` records, NOT unique/distinct chunks: the same chunk
    admitted twice counts twice, and this helper never dedupes. Callers
    MUST NOT read the result as a unique-chunk count (Task 2 M1, still
    open by design). Non-integer (``bool`` rejected) or negative inputs
    raise ``ValueError``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "retrieved_unit_count must be an int, "
            f"got {value!r}; refusing to record"
        )
    if value < 0:
        raise ValueError(
            f"retrieved_unit_count must be >= 0, got {value!r}; refusing to record"
        )
    return value

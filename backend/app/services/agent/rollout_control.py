"""Deterministic server-owned v2 canary selection (Phase 3, Task 7B).

Controller ruling R8: the runtime selector is NOT a second classifier. It
MUST NOT inspect ``QueryAnalysis``, semantic domains, route reasons, or
query content. Selection reads only:

- the authenticated workspace (server-owned, never request-widened),
- the persisted request ID (server-owned, never a client nonce),
- ``NEXUSRAG_AGENT_V2_BUCKET_SALT`` (runtime secret),
- the DB control row (authoritative within environment ceilings),
- the environment ceilings (``NEXUSRAG_AGENT_V2_ENABLED`` /
  ``..._CANARY_PERCENT`` / ``..._CANARY_WORKSPACES``),
- one server-owned eligibility flag (``is_write_endpoint``) supplied at
  the ingress call site — never derived from ``QueryAnalysis`` inside the
  selector. Only an endpoint/request type deterministically known to be a
  Write routes to v1 BEFORE bucketing.

Every other request is bucketed; a v2 candidate then runs the v2
``QueryAnalysis``/Router, and :func:`requires_v1_fallback` (a pure function
over the ALREADY-resolved analysis/route — consulted by the dispatch path,
never by the selector) decides the post-router fallback to v1 BEFORE any
capability execution. ``CANARY_PERCENT=100`` therefore means 100% of
v2-eligible traffic, never a global replacement of v1.

Ordinary request headers are ignored: there is no header parameter anywhere
in this module. The admin-only override is retained (superadmin + valid
value), but it can never escape an active kill switch.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

CanaryArm = Literal["v1", "v2"]

#: Frozen v2 routes (imported for reference only — never redefined here).
_V2_KNOWN_ROUTES = frozenset({"direct", "clarify", "fast_domain", "complex_research"})

#: Frozen work types (``v2/contracts/routing.py::WorkType`` — referenced,
#: never redefined).
_V2_KNOWN_WORK_TYPES = frozenset(
    {
        "direct",
        "lookup",
        "retrieve",
        "explain",
        "summarize",
        "compare",
        "evaluate",
        "cross_domain",
        "multi_goal",
    }
)

#: Frozen domains (``v2/contracts/routing.py::Domain`` — referenced, never
#: redefined).
_V2_KNOWN_DOMAINS = frozenset(
    {"people", "document", "section", "write", "knowledge_graph", "memory"}
)

#: Frozen route reasons (``v2/contracts/routing.py::RouteReason``).
_V2_KNOWN_REASONS = frozenset(
    {
        "direct_greeting",
        "direct_conversation",
        "essential_ambiguity",
        "unresolved_required_binding",
        "simple_people_lookup",
        "exact_document_metadata",
        "exact_section_retrieval",
        "simple_write_operation",
        "simple_kg_lookup",
        "multi_document_research",
        "cross_domain_dependency",
        "comparison",
        "compliance_evaluation",
        "multi_goal",
        "runtime_dependency",
        "evidence_replanning_required",
    }
)


@dataclass(frozen=True)
class CanaryEnv:
    """Environment ceilings for canary selection (from ``settings``)."""

    enabled: bool = False
    canary_percent: float = 0.0
    canary_workspaces: tuple[str, ...] = ()
    bucket_salt: str = ""


@dataclass(frozen=True)
class RolloutControlSnapshot:
    """The DB control row as read for one request (no caching, ever)."""

    enabled: bool = False
    canary_percent: int = 0
    canary_workspaces: tuple[str, ...] = ()
    kill_switch: bool = False
    version: int = 1


def read_canary_env() -> CanaryEnv:
    """Read the canary environment ceilings from ``settings``."""
    from app.core.config import settings

    return CanaryEnv(
        enabled=bool(getattr(settings, "NEXUSRAG_AGENT_V2_ENABLED", False)),
        canary_percent=float(
            getattr(settings, "NEXUSRAG_AGENT_V2_CANARY_PERCENT", 0) or 0
        ),
        canary_workspaces=tuple(
            parse_canary_workspaces(
                getattr(settings, "NEXUSRAG_AGENT_V2_CANARY_WORKSPACES", "")
            )
        ),
        bucket_salt=str(
            getattr(settings, "NEXUSRAG_AGENT_V2_BUCKET_SALT", "") or ""
        ),
    )


def parse_canary_workspaces(value: Any) -> tuple[str, ...]:
    """Parse the allowlist env/DB value into workspace-id strings."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    else:
        return ()
    return tuple(item for item in items if isinstance(item, str) and item.strip())


async def read_control_row(db: Any) -> Any | None:
    """Read the single control row (``id=1``) for THIS request.

    No caching at any layer: every call issues a fresh SELECT so operator
    retuning (including the kill switch) takes effect on the next request.
    Returns ``None`` when the row is absent (callers fail closed to v1).
    DB errors propagate — the serving wrapper maps them to v1 and logs.
    """
    from sqlalchemy import select

    from app.models.agent_rollout_control import AgentRolloutControl

    result = await db.execute(
        select(AgentRolloutControl).where(AgentRolloutControl.id == 1)
    )
    return result.scalar_one_or_none()


def snapshot_of(row: Any) -> RolloutControlSnapshot | None:
    """Coerce a control ORM row (or mapping) to a snapshot; ``None`` stays."""
    if row is None:
        return None
    workspaces = getattr(row, "canary_workspaces", ())
    return RolloutControlSnapshot(
        enabled=bool(getattr(row, "enabled", False)),
        canary_percent=int(getattr(row, "canary_percent", 0) or 0),
        canary_workspaces=tuple(parse_canary_workspaces(workspaces)),
        kill_switch=bool(getattr(row, "kill_switch", False)),
        version=int(getattr(row, "version", 1) or 1),
    )


def deterministic_bucket(
    *, workspace_id: str, request_id: str, salt: str
) -> float:
    """Deterministic bucket in ``[0, 100)`` for one request.

    ``sha256("<workspace>|<request>|<salt>")`` — the authenticated workspace
    plus the persisted request ID plus the bucket salt. Ordinary request
    headers play no role (there is no such input).
    """
    digest = hashlib.sha256(
        f"{workspace_id}|{request_id}|{salt}".encode("utf-8")
    ).hexdigest()
    return (int(digest[:16], 16) % 10000) / 100.0


def effective_canary_percent(env_percent: float, db_percent: int) -> float:
    """DB control is authoritative WITHIN the environment ceiling."""
    try:
        ceiling = float(env_percent or 0)
    except (TypeError, ValueError):
        ceiling = 0.0
    try:
        wanted = float(db_percent or 0)
    except (TypeError, ValueError):
        wanted = 0.0
    return max(0.0, min(ceiling, wanted))


def _normalize_arm(value: Any) -> str:
    from app.services.agent.runtime_selector import normalize_agent_version

    return normalize_agent_version(value)


def select_canary_arm(
    *,
    workspace_id: str,
    request_id: str,
    control: Any | None,
    env: CanaryEnv,
    is_write_endpoint: bool = False,
    admin_override: str | None = None,
    is_superadmin: bool = False,
) -> str:
    """Select the serving arm for one request (pure, server-owned).

    Order: kill switch → env/DB enable → effective percent → workspace
    allowlist → deterministically-known Write exclusion → admin override →
    deterministic bucket. Anything unmet fails closed to ``"v1"``.
    """
    snapshot = control if isinstance(control, RolloutControlSnapshot) else snapshot_of(control)
    if snapshot is not None and snapshot.kill_switch:
        return "v1"
    if admin_override is not None:
        # The admin-only override is retained: an authenticated superadmin's
        # explicit arm choice bypasses sampling (percent/allowlist/bucket),
        # exactly as it bypassed the configured default before canary. It
        # can never escape the kill switch (checked above); anything else —
        # anonymous, non-admin, or invalid — fails closed to v1.
        if not bool(is_superadmin):
            return "v1"
        return _normalize_arm(admin_override)
    if not env.enabled:
        return "v1"
    if snapshot is None or not snapshot.enabled:
        return "v1"
    effective = effective_canary_percent(env.canary_percent, snapshot.canary_percent)
    if effective <= 0.0:
        return "v1"
    allowlist = snapshot.canary_workspaces or env.canary_workspaces
    if allowlist and str(workspace_id) not in {str(item) for item in allowlist}:
        return "v1"
    if bool(is_write_endpoint):
        # Deterministically-known Write endpoint: v1 BEFORE bucketing.
        # This flag is supplied at the ingress call site; the selector
        # never derives it from QueryAnalysis or query content.
        return "v1"
    bucket = deterministic_bucket(
        workspace_id=str(workspace_id), request_id=str(request_id), salt=env.bucket_salt
    )
    return "v2" if bucket < effective else "v1"


def _slot(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def requires_v1_fallback(analysis: Any, route_decision: Any) -> bool:
    """True when a v2 candidate must fall back to v1 BEFORE any capability
    execution or user-visible output.

    Pure function over the ALREADY-resolved v2 ``QueryAnalysis``/Router
    outcome (consulted by the dispatch path — never by the selector, which
    must not inspect semantic domains). Falls back when the resolved route
    is ``write``, ``evaluate``/legal/compliance, or otherwise unsupported;
    missing/unparseable outcomes fail closed to v1 (never served by v2).
    """
    if analysis is None or route_decision is None:
        return True
    try:
        raw_domains = _slot(analysis, "domains") or ()
        domains = tuple(str(item) for item in raw_domains)
        work_type = _slot(analysis, "work_type")
        route = _slot(route_decision, "route")
        reason = _slot(route_decision, "reason_code")
    except Exception:
        return True
    if any("write" in item.lower() for item in domains):
        return True
    if isinstance(work_type, str):
        lowered = work_type.lower()
        if lowered == "evaluate" or "legal" in lowered or "compliance" in lowered:
            return True
        if work_type not in _V2_KNOWN_WORK_TYPES:
            return True
    else:
        return True
    if any(item not in _V2_KNOWN_DOMAINS for item in domains):
        return True
    if isinstance(reason, str):
        lowered_reason = reason.lower()
        if "compliance" in lowered_reason or "legal" in lowered_reason:
            return True
        if reason not in _V2_KNOWN_REASONS:
            return True
    else:
        return True
    if route not in _V2_KNOWN_ROUTES:
        return True
    return False


async def resolve_serving_arm(
    *,
    db: Any,
    workspace_ids: Any,
    request_id: str,
    is_write_endpoint: bool = False,
) -> str:
    """Serving-arm resolution for chat ingress paths (no per-request override).

    Public chat traffic takes no arm override — the admin evaluation surface
    is the only override and it is admin-only. Reads the env ceilings plus a
    FRESH control row; any DB failure or missing row fails closed to v1.
    The bucket workspace is the first sorted authenticated workspace id so
    multi-workspace callers bucket deterministically.
    """
    env = read_canary_env()
    try:
        row = await read_control_row(db)
    except Exception:
        logger.warning("[canary] control-row read failed; keeping v1", exc_info=True)
        return "v1"
    scope = sorted({str(item) for item in (workspace_ids or ()) if str(item)})
    if not scope or not request_id:
        return "v1"
    try:
        return select_canary_arm(
            workspace_id=scope[0],
            request_id=str(request_id),
            control=row,
            env=env,
            is_write_endpoint=bool(is_write_endpoint),
        )
    except Exception:
        logger.warning("[canary] arm selection failed; keeping v1", exc_info=True)
        return "v1"


async def is_v2_killed(db: Any) -> bool:
    """True when the kill switch is set (best-effort; errors map to False).

    Used by the admin evaluation surface to refuse explicit v2 turns while
    the emergency brake is engaged. Ingress selection treats read errors as
    v1 (fail closed) via :func:`resolve_serving_arm` instead.
    """
    try:
        row = await read_control_row(db)
    except Exception:
        logger.warning("[canary] kill-switch read failed", exc_info=True)
        return False
    return bool(row is not None and bool(getattr(row, "kill_switch", False)))

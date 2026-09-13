#!/usr/bin/env python3
"""Collect the live v2 rollout report (Phase 3, Task 7B; R71 hand-off).

Reads ``agent_rollout_metrics`` rows (APPEND-ONLY live traffic facts) and
emits a ``v2_rollout_live_v1`` JSON report for
``check_v2_rollout_gate.py``. R72: the report NEVER compares
``grounded_quality`` across arms — v2 grounded-success completeness is an
invariant, not an arm-neutral quality measure; user-visible quality is
gated by Task 1's golden preflight.

Live-collection usage (operational hand-off — NOT run in this worktree, per
R71; no live stack here)::

    python backend/scripts/collect_v2_rollout_report.py \
        --dsn "$V2_ROLLOUT_DATABASE_URL" --out /tmp/v2_rollout_report.json

The aggregation core (:func:`summarize_metrics`) is pure and unit-tested
offline with synthetic rows.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from typing import Any

REPORT_SCHEMA = "v2_rollout_live_v1"

ERROR_STATUSES = frozenset({"error", "failed", "timeout"})

#: Terminal statuses that mean a requested cancellation actually stopped
#: the run (a successful cancellation — NOT a cancellation failure).
CANCELLED_TERMINALS = frozenset({"cancelled", "cancelling", "canceled"})

SECURITY_KEYS = (
    "checkpoint_secret",
    "ungrounded_factual_success",
    "acl_leak",
    "duplicate_production_write",
)


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _valid_security(row: dict[str, Any]) -> bool:
    """True iff the row carries four EXPLICIT boolean security counters.

    R72/R74: missing, null, or non-boolean security fields are INVALID —
    never counted as zero violations. Invalid rows are excluded from every
    aggregate and counted in ``invalid_security_rows`` (the arm is marked
    ``valid: false`` and the gate rejects it).
    """
    security = row.get("security")
    if not isinstance(security, dict):
        return False
    return all(
        isinstance(security.get(key), bool) for key in SECURITY_KEYS
    )


def _is_cancel_failure(row: dict[str, Any]) -> bool:
    """Authoritative cancellation-failure invariant (documented).

    Cancellation failure is NOT the cancel rate: a run that was cancelled
    AND ended in a cancelled terminal was successfully stopped (not a
    failure). A run flagged ``cancelled`` that nevertheless completed with
    any other terminal (e.g. ``success`` — output was produced despite the
    cancel) FAILED to stop and counts as a cancellation failure.
    """
    if row.get("cancelled") is not True:
        return False
    return str(row.get("terminal_status", "")).lower() not in CANCELLED_TERMINALS


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metric rows into a ``v2_rollout_live_v1`` report (pure)."""
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("v1", "v2", "shadow"):
        arm_rows = [row for row in rows if row.get("arm") == arm]
        finished = [
            row for row in arm_rows if row.get("finished_at") is not None
        ]
        invalid_security_rows = sum(
            1 for row in finished if not _valid_security(row)
        )
        completed = [row for row in finished if _valid_security(row)]
        durations = [
            float(row["duration_ms"])
            for row in completed
            if row.get("duration_ms") is not None
        ]
        errors = sum(
            1
            for row in completed
            if str(row.get("terminal_status", "")).lower() in ERROR_STATUSES
        )
        security_violations = 0
        for row in completed:
            security = row.get("security") or {}
            security_violations += sum(
                1 for key in SECURITY_KEYS if security.get(key) is True
            )
        cancelled = sum(1 for row in completed if row.get("cancelled") is True)
        cancel_failures = sum(1 for row in completed if _is_cancel_failure(row))
        starts = [
            parsed
            for parsed in (_parse_dt(row.get("started_at")) for row in arm_rows)
            if parsed is not None
        ]
        ends = [
            parsed
            for parsed in (_parse_dt(row.get("finished_at")) for row in completed)
            if parsed is not None
        ]
        hours = (
            max(0.0, (max(ends) - min(starts)).total_seconds() / 3600.0)
            if starts and ends
            else 0.0
        )
        # Continuity (not elapsed span): distinct UTC hour buckets holding
        # >= 1 valid completed sample, plus the largest gap between
        # consecutive valid completions. A pair of samples 24h apart has
        # elapsed hours >= 24 but continuous_hours == 2.
        buckets = {
            int(moment.timestamp() // 3600)
            for moment in ends
        }
        continuous_hours = float(len(buckets))
        ordered = sorted(ends)
        max_gap_hours = 0.0
        for earlier, later in zip(ordered, ordered[1:]):
            gap = (later - earlier).total_seconds() / 3600.0
            if gap > max_gap_hours:
                max_gap_hours = gap
        arms[arm] = {
            "samples": len(arm_rows),
            "completed": len(completed),
            "invalid_security_rows": invalid_security_rows,
            "valid": invalid_security_rows == 0,
            "error_rate": (errors / len(completed)) if completed else 0.0,
            "errors": errors,
            "p50_ms": _percentile(durations, 50),
            "p95_ms": _percentile(durations, 95),
            "security_violations": security_violations,
            "cancelled": cancelled,
            "cancel_rate": (cancelled / len(completed)) if completed else 0.0,
            "cancel_failures": cancel_failures,
            "cancel_failure_rate": (
                (cancel_failures / len(completed)) if completed else 0.0
            ),
            "hours": hours,
            "continuous_hours": continuous_hours,
            "max_gap_hours": max_gap_hours,
        }
    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": generated_at,
        "arms": arms,
    }


def fetch_live_rows(dsn: str) -> list[dict[str, Any]]:
    """Load live metric rows from the rollout database (operational only)."""
    import sqlalchemy

    engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(
                    "SELECT arm, request_id_hash, workspace_id_hash, started_at, "
                    "finished_at, duration_ms, terminal_status, citation_count, "
                    "cancelled, security_checkpoint_secret, "
                    "security_ungrounded_factual_success, security_acl_leak, "
                    "security_duplicate_production_write "
                    "FROM agent_rollout_metrics ORDER BY id"
                )
            )
            rows = []
            for record in result.mappings().all():
                rows.append(
                    {
                        "arm": record["arm"],
                        "request_id_hash": record["request_id_hash"],
                        "workspace_id_hash": record["workspace_id_hash"],
                        "started_at": record["started_at"],
                        "finished_at": record["finished_at"],
                        "duration_ms": record["duration_ms"],
                        "terminal_status": record["terminal_status"],
                        "citation_count": record["citation_count"],
                        "cancelled": record["cancelled"],
                        "security": {
                            "checkpoint_secret": record[
                                "security_checkpoint_secret"
                            ],
                            "ungrounded_factual_success": record[
                                "security_ungrounded_factual_success"
                            ],
                            "acl_leak": record["security_acl_leak"],
                            "duplicate_production_write": record[
                                "security_duplicate_production_write"
                            ],
                        },
                    }
                )
            return rows
    finally:
        engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True, help="Rollout DB DSN")
    parser.add_argument("--out", required=True, help="Report JSON output path")
    args = parser.parse_args(argv)
    rows = fetch_live_rows(args.dsn)
    report = summarize_metrics(rows)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"collected {len(rows)} metric rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

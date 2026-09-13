#!/usr/bin/env python3
"""Check the live v2 rollout gate (Phase 3, Task 7B; R71 hand-off).

Reads a ``v2_rollout_live_v1`` report (see
``collect_v2_rollout_report.py``) and verdicts PASS/FAIL against the
rollback-gate thresholds::

    >= 200 completed samples per arm (v1 + v2)
    >= 24 continuous hours
    zero security violations
    v2 error-rate regression <= 1 percentage point
    v2 p95 regression <= 15%
    cancellation failure <= 0.1%

Golden/preflight report schemas are REJECTED (exit non-zero): the live
checker never verdicts evaluation artifacts. R72: no ``grounded_quality``
comparison exists anywhere on this path.

Live-gate usage (operational hand-off — NOT run in this worktree, per R71)::

    python backend/scripts/collect_v2_rollout_report.py \
        --dsn "$V2_ROLLOUT_DATABASE_URL" --out /tmp/v2_rollout_report.json
    python backend/scripts/check_v2_rollout_gate.py \
        --report /tmp/v2_rollout_report.json

Exit 0 on PASS, 1 on FAIL/rejection. The pure :func:`check_gate` is
unit-tested offline with synthetic reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from collect_v2_rollout_report import REPORT_SCHEMA

MIN_COMPLETED_PER_ARM = 200
MIN_HOURS = 24.0
MAX_ERROR_RATE_REGRESSION_PP = 0.01
MAX_P95_REGRESSION_RATIO = 0.15
MAX_CANCEL_FAILURE_RATE = 0.001


def _is_live_report(report: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(report, dict):
        return False, "report is not a JSON object"
    if report.get("schema") != REPORT_SCHEMA:
        return (
            False,
            f"rejected: not a live rollout report "
            f"(schema={report.get('schema')!r}, expected {REPORT_SCHEMA!r})",
        )
    if report.get("preflight") is True:
        return False, "rejected: preflight reports never pass the live gate"
    if "golden" in report or "cases" in report:
        return False, "rejected: golden/eval artifacts never pass the live gate"
    arms = report.get("arms")
    if not isinstance(arms, dict) or "v1" not in arms or "v2" not in arms:
        return False, "rejected: live report must carry v1 and v2 arms"
    return True, ""


def check_gate(report: dict[str, Any]) -> tuple[bool, list[str]]:
    """Verdict a report against the rollback-gate thresholds (pure)."""
    live, reason = _is_live_report(report)
    if not live:
        return False, [reason]
    failures: list[str] = []
    arms = report["arms"]
    for arm in ("v1", "v2"):
        completed = int(arms[arm].get("completed", 0) or 0)
        if completed < MIN_COMPLETED_PER_ARM:
            failures.append(
                f"{arm}: only {completed} completed samples "
                f"(need >= {MIN_COMPLETED_PER_ARM})"
            )
        hours = float(arms[arm].get("hours", 0.0) or 0.0)
        if hours < MIN_HOURS:
            failures.append(
                f"{arm}: only {hours:.1f} continuous hours "
                f"(need >= {MIN_HOURS:.0f})"
            )
    total_violations = sum(
        int(arms[arm].get("security_violations", 0) or 0) for arm in arms
    )
    if total_violations > 0:
        failures.append(
            f"security: {total_violations} violations (need zero)"
        )
    v1 = arms["v1"]
    v2 = arms["v2"]
    error_regression = float(v2.get("error_rate", 0.0)) - float(
        v1.get("error_rate", 0.0)
    )
    if error_regression > MAX_ERROR_RATE_REGRESSION_PP:
        failures.append(
            f"error-rate: v2 regressed {error_regression * 100:.2f}pp "
            f"(limit {MAX_ERROR_RATE_REGRESSION_PP * 100:.0f}pp)"
        )
    p95_v1 = v1.get("p95_ms")
    p95_v2 = v2.get("p95_ms")
    if p95_v1 is not None and p95_v2 is not None and float(p95_v1) > 0:
        p95_regression = (float(p95_v2) - float(p95_v1)) / float(p95_v1)
        if p95_regression > MAX_P95_REGRESSION_RATIO:
            failures.append(
                f"p95: v2 regressed {p95_regression * 100:.1f}% "
                f"(limit {MAX_P95_REGRESSION_RATIO * 100:.0f}%)"
            )
    for arm in ("v1", "v2"):
        cancel_rate = float(arms[arm].get("cancel_rate", 0.0) or 0.0)
        if cancel_rate > MAX_CANCEL_FAILURE_RATE:
            failures.append(
                f"{arm}: cancellation failure {cancel_rate * 100:.2f}% "
                f"(limit {MAX_CANCEL_FAILURE_RATE * 100:.1f}%)"
            )
    return (len(failures) == 0), failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Report JSON path")
    args = parser.parse_args(argv)
    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    passed, failures = check_gate(report)
    if passed:
        print("ROLLBACK GATE: PASS")
        return 0
    print("ROLLBACK GATE: FAIL")
    for failure in failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

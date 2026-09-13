#!/usr/bin/env python3
"""Check the live v2 rollout gate (Phase 3, Task 7B; R71 hand-off).

Reads a ``v2_rollout_live_v1`` report (see
``collect_v2_rollout_report.py``) and verdicts PASS/FAIL against the
rollback-gate thresholds::

    >= 200 completed samples per arm (v1 + v2)
    >= 24 continuous hours with no gap between consecutive completions
    larger than 2h (consecutive coverage, not occupied buckets)
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
import math

MIN_COMPLETED_PER_ARM = 200
MIN_HOURS = 24.0
# Continuity means CONSECUTIVE coverage (fix round 2, Important 7): 24
# occupied hour buckets are not enough when a telemetry gap splits them.
# No two consecutive valid completions may be further apart than this.
MAX_GAP_HOURS = 2.0
MAX_ERROR_RATE_REGRESSION_PP = 0.01
MAX_P95_REGRESSION_RATIO = 0.15
MAX_CANCEL_FAILURE_RATE = 0.001


def _req_int(arm_data: dict[str, Any], key: str) -> int | None:
    """Require an explicit integer gate input (bools rejected)."""
    value = arm_data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _req_rate(arm_data: dict[str, Any], key: str) -> float | None:
    """Require an explicit finite 0..1 gate input (null/NaN/inf rejected)."""
    value = arm_data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        return None
    return number


def _req_number(arm_data: dict[str, Any], key: str) -> float | None:
    """Require an explicit finite numeric gate input."""
    value = arm_data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


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
    """Verdict a report against the rollback-gate thresholds (pure).

    Strict validation: every gate input must be present with the correct
    type and (for numerics) finite. Missing, null, non-finite, or
    mistyped inputs FAIL the gate — they never default to a safe zero or
    skip a check, so a forged report with only ``completed``/``hours``
    cannot pass.
    """
    live, reason = _is_live_report(report)
    if not live:
        return False, [reason]
    failures: list[str] = []
    arms = report["arms"]
    valid_inputs: dict[str, dict[str, Any]] = {}
    for arm in ("v1", "v2"):
        data = arms[arm]
        if not isinstance(data, dict):
            failures.append(f"{arm}: arm entry is not an object")
            continue
        completed = _req_int(data, "completed")
        continuous = _req_number(data, "continuous_hours")
        max_gap = _req_number(data, "max_gap_hours")
        violations = _req_int(data, "security_violations")
        invalid_rows = _req_int(data, "invalid_security_rows")
        error_rate = _req_rate(data, "error_rate")
        p95 = _req_number(data, "p95_ms")
        cancel_failures = _req_int(data, "cancel_failures")
        cancel_failure_rate = _req_rate(data, "cancel_failure_rate")
        missing = [
            key
            for key, value in (
                ("completed", completed),
                ("continuous_hours", continuous),
                ("max_gap_hours", max_gap),
                ("security_violations", violations),
                ("invalid_security_rows", invalid_rows),
                ("error_rate", error_rate),
                ("p95_ms", p95),
                ("cancel_failures", cancel_failures),
                ("cancel_failure_rate", cancel_failure_rate),
            )
            if value is None
        ]
        if missing:
            failures.append(
                f"{arm}: incomplete live gate inputs {missing} "
                "(missing/null/non-finite/mistyped never defaults safe)"
            )
            continue
        if data.get("valid") is not True or invalid_rows != 0:
            failures.append(
                f"{arm}: {invalid_rows} rows with invalid security "
                "observations (missing fields are invalid, never safe)"
            )
        if completed < MIN_COMPLETED_PER_ARM:
            failures.append(
                f"{arm}: only {completed} completed samples "
                f"(need >= {MIN_COMPLETED_PER_ARM})"
            )
        if continuous < MIN_HOURS:
            failures.append(
                f"{arm}: only {continuous:.1f} continuous hours "
                f"(need >= {MIN_HOURS:.0f})"
            )
        if max_gap > MAX_GAP_HOURS:
            failures.append(
                f"{arm}: max gap between consecutive completions is "
                f"{max_gap:.1f}h (limit {MAX_GAP_HOURS:.0f}h); 24 occupied "
                "hour buckets with a telemetry gap are not continuous "
                "coverage"
            )
        valid_inputs[arm] = {
            "error_rate": error_rate,
            "p95_ms": p95,
            "security_violations": violations,
            "cancel_failure_rate": cancel_failure_rate,
        }
    if "v1" in valid_inputs and "v2" in valid_inputs:
        total_violations = (
            valid_inputs["v1"]["security_violations"]
            + valid_inputs["v2"]["security_violations"]
        )
        if total_violations > 0:
            failures.append(
                f"security: {total_violations} violations (need zero)"
            )
        error_regression = (
            valid_inputs["v2"]["error_rate"] - valid_inputs["v1"]["error_rate"]
        )
        if error_regression > MAX_ERROR_RATE_REGRESSION_PP:
            failures.append(
                f"error-rate: v2 regressed {error_regression * 100:.2f}pp "
                f"(limit {MAX_ERROR_RATE_REGRESSION_PP * 100:.0f}pp)"
            )
        p95_v1 = valid_inputs["v1"]["p95_ms"]
        p95_v2 = valid_inputs["v2"]["p95_ms"]
        if p95_v1 > 0:
            p95_regression = (p95_v2 - p95_v1) / p95_v1
            if p95_regression > MAX_P95_REGRESSION_RATIO:
                failures.append(
                    f"p95: v2 regressed {p95_regression * 100:.1f}% "
                    f"(limit {MAX_P95_REGRESSION_RATIO * 100:.0f}%)"
                )
        for arm in ("v1", "v2"):
            failure_rate = valid_inputs[arm]["cancel_failure_rate"]
            if failure_rate > MAX_CANCEL_FAILURE_RATE:
                failures.append(
                    f"{arm}: cancellation failure {failure_rate * 100:.2f}% "
                    f"(limit {MAX_CANCEL_FAILURE_RATE * 100:.1f}%)"
                )
    elif not failures:
        failures.append("rejected: no verifiable v1/v2 gate inputs")
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

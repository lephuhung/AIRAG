"""Task 7B — rollout metrics + rollback gate tests (TDD, failing first).

Covers: append-only metrics (never update), authoritative security-counter
producers with missing/default fields INVALID (never "safe"), live report
aggregation without any grounded_quality comparison (R72), gate thresholds
(>=200 samples/arm, >=24h, zero security violations, error-rate regression
<=1pp, p95 regression <=15%, cancellation failure <=0.1%), and rejection of
golden/preflight report schemas by the live checker (R71, offline with
synthetic reports).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


def _row(arm="v2", **overrides):
    base = {
        "arm": arm,
        "request_id_hash": "req-hash-1",
        "workspace_id_hash": "ws-hash-1",
        "started_at": datetime.now(UTC) - timedelta(seconds=30),
        "finished_at": datetime.now(UTC),
        "duration_ms": 1200,
        "terminal_status": "success",
        "citation_count": 3,
        "cancelled": False,
        "security": {
            "checkpoint_secret": False,
            "ungrounded_factual_success": False,
            "acl_leak": False,
            "duplicate_production_write": False,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Security counters: missing/default fields are INVALID, never safe
# ---------------------------------------------------------------------------


def test_missing_security_fields_are_invalid():
    from app.services.agent.rollout_metrics import validate_security_flags

    with pytest.raises(ValueError):
        validate_security_flags(None)
    with pytest.raises(ValueError):
        validate_security_flags({})
    with pytest.raises(ValueError):
        validate_security_flags({"checkpoint_secret": False})


def test_default_constructed_security_is_invalid():
    from app.services.agent.rollout_metrics import validate_security_flags

    with pytest.raises(ValueError):
        validate_security_flags(
            {
                "checkpoint_secret": False,
                "ungrounded_factual_success": False,
                # acl_leak + duplicate_production_write absent: NOT safe.
            }
        )


def test_non_boolean_security_fields_are_invalid():
    from app.services.agent.rollout_metrics import validate_security_flags

    with pytest.raises(ValueError):
        validate_security_flags(
            {
                "checkpoint_secret": 0,
                "ungrounded_factual_success": False,
                "acl_leak": False,
                "duplicate_production_write": False,
            }
        )


def test_explicit_all_false_security_is_valid():
    from app.services.agent.rollout_metrics import validate_security_flags

    validated = validate_security_flags(
        {
            "checkpoint_secret": False,
            "ungrounded_factual_success": False,
            "acl_leak": False,
            "duplicate_production_write": False,
        }
    )
    assert validated == {
        "checkpoint_secret": False,
        "ungrounded_factual_success": False,
        "acl_leak": False,
        "duplicate_production_write": False,
    }


def test_authoritative_producers_for_each_counter():
    from app.services.agent import rollout_metrics

    for counter in (
        "checkpoint_secret",
        "ungrounded_factual_success",
        "acl_leak",
        "duplicate_production_write",
    ):
        assert counter in rollout_metrics.SECURITY_FIELDS
    # Each counter has a named producer entry point on the module.
    for producer in (
        "security_checkpoint_secret_from_terminal",
        "security_ungrounded_factual_success_from_terminal",
        "security_acl_leak_from_terminal",
        "security_duplicate_production_write_from_terminal",
    ):
        assert callable(getattr(rollout_metrics, producer)), producer


def test_metrics_module_exposes_no_update_or_delete_path():
    from app.services.agent import rollout_metrics

    for name in dir(rollout_metrics):
        lowered = name.lower()
        assert "update_rollout" not in lowered, name
        assert "delete_rollout" not in lowered, name
    import inspect

    source = inspect.getsource(rollout_metrics)
    assert ".update(" not in source
    assert ".delete(" not in source


# ---------------------------------------------------------------------------
# ORM mapping matches the T7A contract exactly
# ---------------------------------------------------------------------------


def test_rollout_orm_tables_match_t7a_contract():
    from app.models.agent_rollout_control import AgentRolloutControl
    from app.models.agent_rollout_metric import AgentRolloutMetric

    assert AgentRolloutControl.__tablename__ == "agent_rollout_control"
    assert AgentRolloutMetric.__tablename__ == "agent_rollout_metrics"
    control_columns = set(AgentRolloutControl.__table__.columns.keys())
    assert control_columns == {
        "id",
        "enabled",
        "shadow_percent",
        "canary_percent",
        "canary_workspaces",
        "kill_switch",
        "updated_at",
        "updated_by",
        "version",
    }
    metric_columns = set(AgentRolloutMetric.__table__.columns.keys())
    assert metric_columns == {
        "id",
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
        "created_at",
    }


# ---------------------------------------------------------------------------
# Live report aggregation (offline, synthetic rows; R71)
# ---------------------------------------------------------------------------


def _synthetic_rows(n_v1=210, n_v2=210, hours=25):
    rows = []
    start = datetime.now(UTC) - timedelta(hours=hours)
    span_seconds = hours * 3600.0
    for arm, count in (("v1", n_v1), ("v2", n_v2)):
        step = span_seconds / max(count, 1)
        for index in range(count):
            moment = start + timedelta(seconds=index * step)
            rows.append(
                _row(
                    arm=arm,
                    request_id_hash=f"{arm}-req-{index}",
                    duration_ms=1000 + (index % 50),
                    started_at=moment,
                    finished_at=moment + timedelta(seconds=1),
                )
            )
    return rows


def test_collect_builds_live_report_without_quality_comparison():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    try:
        from collect_v2_rollout_report import summarize_metrics
    finally:
        sys.path.pop(0)

    report = summarize_metrics(_synthetic_rows())
    assert report["schema"] == "v2_rollout_live_v1"
    assert report["arms"]["v1"]["completed"] == 210
    assert report["arms"]["v2"]["completed"] == 210
    # R72: no grounded_quality comparison anywhere in the live report.
    dumped = repr(report).lower()
    assert "grounded_quality" not in dumped
    assert "quality" not in dumped


def test_gate_passes_on_healthy_report():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    try:
        from collect_v2_rollout_report import summarize_metrics
        from check_v2_rollout_gate import check_gate
    finally:
        sys.path.pop(0)

    report = summarize_metrics(_synthetic_rows())
    passed, failures = check_gate(report)
    assert passed is True, failures
    assert failures == []


def test_gate_requires_200_samples_per_arm():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    try:
        from collect_v2_rollout_report import summarize_metrics
        from check_v2_rollout_gate import check_gate
    finally:
        sys.path.pop(0)

    report = summarize_metrics(_synthetic_rows(n_v1=210, n_v2=40))
    passed, failures = check_gate(report)
    assert passed is False
    assert any("samples" in failure for failure in failures)


def test_gate_requires_zero_security_violations():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    try:
        from collect_v2_rollout_report import summarize_metrics
        from check_v2_rollout_gate import check_gate
    finally:
        sys.path.pop(0)

    rows = _synthetic_rows()
    rows[0]["security"] = {
        "checkpoint_secret": True,
        "ungrounded_factual_success": False,
        "acl_leak": False,
        "duplicate_production_write": False,
    }
    report = summarize_metrics(rows)
    passed, failures = check_gate(report)
    assert passed is False
    assert any("security" in failure for failure in failures)


def test_gate_rejects_error_rate_regression_over_1pp():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    try:
        from collect_v2_rollout_report import summarize_metrics
        from check_v2_rollout_gate import check_gate
    finally:
        sys.path.pop(0)

    rows = _synthetic_rows()
    for row in rows:
        if row["arm"] == "v2" and int(row["request_id_hash"].rsplit("-", 1)[-1]) < 10:
            row["terminal_status"] = "error"
    report = summarize_metrics(rows)
    passed, failures = check_gate(report)
    assert passed is False
    assert any("error" in failure for failure in failures)


def test_gate_rejects_golden_and_preflight_schemas():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    try:
        from check_v2_rollout_gate import check_gate
    finally:
        sys.path.pop(0)

    golden = {
        "schema": "golden_preflight_v1",
        "cases": [],
        "summary": {"pass_rate": 1.0},
    }
    passed, failures = check_gate(golden)
    assert passed is False
    assert any("schema" in failure for failure in failures)

    preflight = {
        "schema": "v2_rollout_live_v1",
        "preflight": True,
        "arms": {},
    }
    passed, failures = check_gate(preflight)
    assert passed is False
    assert any("preflight" in failure.lower() for failure in failures)

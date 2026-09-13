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


# ---------------------------------------------------------------------------
# Task 7B fix round 1 (R74 + scripts): real detectors, terminal emission,
# strict collector/gate, continuity, cancel-failure signal
# ---------------------------------------------------------------------------


def test_real_detectors_compute_verdicts_from_terminal():
    from app.services.agent import rollout_metrics as metrics

    # checkpoint_secret: scans terminal text for secret markers.
    assert (
        metrics.detect_checkpoint_secret("here is the answer, no secrets")
        is False
    )
    assert (
        metrics.detect_checkpoint_secret("leaked BEGIN PRIVATE KEY block")
        is True
    )
    assert metrics.detect_checkpoint_secret(None) is None
    # acl_leak: served ids outside the allowed set.
    assert (
        metrics.detect_acl_leak(
            served_document_ids=["doc-a"], allowed_document_ids=["doc-a"]
        )
        is False
    )
    assert (
        metrics.detect_acl_leak(
            served_document_ids=["doc-evil"], allowed_document_ids=["doc-a"]
        )
        is True
    )
    assert (
        metrics.detect_acl_leak(
            served_document_ids=None, allowed_document_ids=["doc-a"]
        )
        is None
    )
    # ungrounded_factual_success: factual success with zero citations.
    assert (
        metrics.detect_ungrounded_factual_success(
            terminal_status="success", citation_count=0, factual_expected=True
        )
        is True
    )
    assert (
        metrics.detect_ungrounded_factual_success(
            terminal_status="success", citation_count=2, factual_expected=True
        )
        is False
    )
    assert (
        metrics.detect_ungrounded_factual_success(
            terminal_status="success", citation_count=0, factual_expected=False
        )
        is False
    )
    assert (
        metrics.detect_ungrounded_factual_success(
            terminal_status="success", citation_count=0, factual_expected=None
        )
        is None
    )
    # duplicate_production_write: more than one production write.
    assert metrics.detect_duplicate_production_write(0) is False
    assert metrics.detect_duplicate_production_write(1) is False
    assert metrics.detect_duplicate_production_write(2) is True
    assert metrics.detect_duplicate_production_write(None) is None


def test_unobservable_verdict_makes_row_invalid_never_safe():
    from app.services.agent import rollout_metrics as metrics

    with pytest.raises(ValueError):
        metrics.build_security_verdicts(
            answer_text=None,
            terminal_status="success",
            citation_count=0,
            factual_expected=True,
            served_document_ids=[],
            allowed_document_ids=["doc-a"],
            production_write_count=0,
        )


def test_security_verdicts_use_authoritative_producers():
    from app.services.agent import rollout_metrics as metrics

    verdicts = metrics.build_security_verdicts(
        answer_text="plain answer",
        terminal_status="success",
        citation_count=2,
        factual_expected=True,
        served_document_ids=["doc-a"],
        allowed_document_ids=["doc-a"],
        production_write_count=1,
    )
    assert verdicts == {
        "checkpoint_secret": False,
        "ungrounded_factual_success": False,
        "acl_leak": False,
        "duplicate_production_write": False,
    }


def test_terminal_emission_records_one_append_only_row():
    import asyncio

    from app.services.agent import rollout_metrics as metrics

    added: list = []

    class FakeDB:
        def add(self, row):
            added.append(row)

        async def commit(self):
            return None

        async def rollback(self):
            return None

    async def _run():
        return await metrics.emit_terminal_rollout_metric(
            FakeDB(),
            arm="v2",
            request_id="req-1",
            workspace_ids=["ws-1"],
            started_at=datetime.now(UTC) - timedelta(seconds=2),
            terminal_status="success",
            citation_count=1,
            cancelled=False,
            answer_text="grounded answer",
            factual_expected=True,
            served_document_ids=["doc-a"],
            allowed_document_ids=["doc-a"],
            production_write_count=0,
        )

    row = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())
    assert len(added) == 1
    assert row.arm == "v2"
    assert row.security_acl_leak is False
    assert row.terminal_status == "success"


def test_terminal_emission_refuses_unobservable_row():
    import asyncio

    from app.services.agent import rollout_metrics as metrics

    class FakeDB:
        def add(self, row):  # pragma: no cover
            raise AssertionError("must not record an unobservable row")

        async def commit(self):  # pragma: no cover
            raise AssertionError("must not record an unobservable row")

        async def rollback(self):
            return None

    async def _run():
        with pytest.raises(ValueError):
            await metrics.emit_terminal_rollout_metric(
                FakeDB(),
                arm="v2",
                request_id="req-1",
                workspace_ids=["ws-1"],
                started_at=datetime.now(UTC),
                terminal_status="success",
                citation_count=0,
                cancelled=False,
                answer_text=None,
                factual_expected=True,
                served_document_ids=[],
                allowed_document_ids=["doc-a"],
                production_write_count=0,
            )

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())


def _script_module(name):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def test_collector_rejects_missing_security_as_invalid():
    collector = _script_module("collect_v2_rollout_report")
    gate = _script_module("check_v2_rollout_gate")

    start = datetime.now(UTC) - timedelta(hours=25)
    rows = []
    for index in range(210):
        moment = start + timedelta(minutes=index * 7)
        rows.append(
            _row(
                arm="v2",
                request_id_hash=f"v2-req-{index}",
                started_at=moment,
                finished_at=moment + timedelta(seconds=1),
            )
        )
    # 210 v1 rows with NO security observations at all.
    for index in range(210):
        moment = start + timedelta(minutes=index * 7)
        bad = _row(
            arm="v1",
            request_id_hash=f"v1-req-{index}",
            started_at=moment,
            finished_at=moment + timedelta(seconds=1),
        )
        del bad["security"]
        rows.append(bad)
    report = collector.summarize_metrics(rows)
    assert report["arms"]["v1"]["invalid_security_rows"] == 210
    assert report["arms"]["v1"]["valid"] is False
    passed, _ = gate.check_gate(report)
    assert passed is False


def test_collector_continuity_rejects_gapped_series():
    collector = _script_module("collect_v2_rollout_report")
    gate = _script_module("check_v2_rollout_gate")

    start = datetime.now(UTC) - timedelta(hours=25)
    rows = []
    # Two samples 25h apart: elapsed span >= 24h but no continuity.
    for index, moment in (
        (0, start),
        (1, start + timedelta(hours=25)),
    ):
        for arm in ("v1", "v2"):
            rows.append(
                _row(
                    arm=arm,
                    request_id_hash=f"{arm}-gap-{index}",
                    started_at=moment,
                    finished_at=moment + timedelta(seconds=1),
                )
            )
    report = collector.summarize_metrics(rows)
    assert report["arms"]["v1"]["hours"] >= 24
    assert report["arms"]["v1"]["continuous_hours"] < 24
    passed, failures = gate.check_gate(report)
    assert passed is False
    assert any("continuous" in failure for failure in failures)


def test_collector_cancel_failure_is_not_cancel_rate():
    collector = _script_module("collect_v2_rollout_report")

    start = datetime.now(UTC)
    rows = []
    # A successfully-cancelled run (terminal "cancelled") is NOT a failure;
    # a run flagged cancelled that still completed with success IS one.
    rows.append(
        _row(
            arm="v2",
            request_id_hash="v2-cancel-ok",
            started_at=start,
            finished_at=start + timedelta(seconds=1),
            cancelled=True,
            terminal_status="cancelled",
        )
    )
    rows.append(
        _row(
            arm="v2",
            request_id_hash="v2-cancel-failed",
            started_at=start,
            finished_at=start + timedelta(seconds=1),
            cancelled=True,
            terminal_status="success",
        )
    )
    report = collector.summarize_metrics(rows)
    assert report["arms"]["v2"]["cancelled"] == 2
    assert report["arms"]["v2"]["cancel_failures"] == 1


def test_gate_rejects_incomplete_or_nonnumeric_inputs():
    gate = _script_module("check_v2_rollout_gate")
    collector = _script_module("collect_v2_rollout_report")

    report = collector.summarize_metrics(_synthetic_rows())
    # Missing security_violations must FAIL, never default to zero.
    del report["arms"]["v2"]["security_violations"]
    passed, _ = gate.check_gate(report)
    assert passed is False

    report = collector.summarize_metrics(_synthetic_rows())
    report["arms"]["v2"]["error_rate"] = float("inf")
    passed, _ = gate.check_gate(report)
    assert passed is False

    report = collector.summarize_metrics(_synthetic_rows())
    report["arms"]["v1"]["p95_ms"] = None
    passed, _ = gate.check_gate(report)
    assert passed is False

    # A forged minimal report with sufficient completed/hours still fails:
    # the complete live schema is required.
    forged = {
        "schema": "v2_rollout_live_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "arms": {
            "v1": {"completed": 500, "hours": 30},
            "v2": {"completed": 500, "hours": 30},
        },
    }
    passed, _ = gate.check_gate(forged)
    assert passed is False


# ---------------------------------------------------------------------------
# Task 7B fix round 2: like-ID ACL comparison (new Important), observed
# factual expectation (Critical 2), bounded-gap continuity (Important 7)
# ---------------------------------------------------------------------------


def test_acl_extractor_uses_document_ids_only():
    from app.services.agent import rollout_metrics as metrics

    # A normal source carrying BOTH a workspace id and a document id must
    # compare the document id (like vs like), never the workspace id.
    sources = [
        {
            "workspace_id": "ws-1",
            "knowledge_base_id": "kb-1",
            "document_id": "doc-a",
            "chunk": "c1",
        }
    ]
    served = metrics.extract_served_document_ids(sources)
    assert served == ["doc-a"]
    assert (
        metrics.detect_acl_leak(
            served_document_ids=served, allowed_document_ids=["doc-a"]
        )
        is False
    )
    assert (
        metrics.detect_acl_leak(
            served_document_ids=served, allowed_document_ids=["doc-other"]
        )
        is True
    )


def test_acl_unidentified_served_entries_are_invalid_not_safe():
    from app.services.agent import rollout_metrics as metrics

    # A served entry with no document id cannot be compared like-vs-like:
    # under a document filter the verdict is unobservable (invalid), never
    # "safe" and never a fabricated workspace-vs-document violation.
    sources = [{"workspace_id": "ws-1", "chunk": "c1"}]
    served = metrics.extract_served_document_ids(sources)
    assert served == []
    assert metrics.count_served_sources_without_document_id(sources) == 1
    assert (
        metrics.detect_acl_leak(
            served_document_ids=served,
            allowed_document_ids=["doc-a"],
            unidentified_served_count=1,
        )
        is None
    )
    with pytest.raises(ValueError):
        metrics.build_security_verdicts(
            answer_text="an answer",
            terminal_status="success",
            citation_count=1,
            factual_expected=True,
            served_document_ids=served,
            allowed_document_ids=["doc-a"],
            production_write_count=0,
            unidentified_served_count=1,
        )


def test_resolve_factual_expected_covers_terminal_outcomes():
    from app.services.agent import rollout_metrics as metrics

    # No factual success occurred: error / cancellation / empty answer.
    assert (
        metrics.resolve_factual_expected(
            arm="v2", terminal_status="error", cancelled=False, answer_text="x"
        )
        is False
    )
    assert (
        metrics.resolve_factual_expected(
            arm="v2",
            terminal_status="cancelled",
            cancelled=True,
            answer_text="partial",
        )
        is False
    )
    assert (
        metrics.resolve_factual_expected(
            arm="v2", terminal_status="success", cancelled=False, answer_text="  "
        )
        is False
    )
    # Observed non-factual turns: v2 clarify status / direct+clarify routes.
    assert (
        metrics.resolve_factual_expected(
            arm="v2",
            terminal_status="success",
            cancelled=False,
            answer_text="Which document?",
            response_status="clarify",
        )
        is False
    )
    assert (
        metrics.resolve_factual_expected(
            arm="v2",
            terminal_status="success",
            cancelled=False,
            answer_text="Xin chào!",
            response_status="success",
            route="direct",
        )
        is False
    )
    # v2 research-route success answers must be grounded (R72 invariant).
    assert (
        metrics.resolve_factual_expected(
            arm="v2",
            terminal_status="success",
            cancelled=False,
            answer_text="The decree states ...",
            response_status="success",
            route="complex_research",
        )
        is True
    )
    # v2 success-shaped but route/status unknown: unobservable -> invalid.
    assert (
        metrics.resolve_factual_expected(
            arm="v2",
            terminal_status="success",
            cancelled=False,
            answer_text="The decree states ...",
        )
        is None
    )
    # v1: observed greeting intent is non-factual; other success answers
    # are expected grounded (fail-closed, never defaulted safe).
    assert (
        metrics.resolve_factual_expected(
            arm="v1",
            terminal_status="success",
            cancelled=False,
            answer_text="Xin chào!",
            greeting_observed=True,
        )
        is False
    )
    assert (
        metrics.resolve_factual_expected(
            arm="v1",
            terminal_status="success",
            cancelled=False,
            answer_text="The decree states ...",
        )
        is True
    )


def test_v1_greeting_label_matches_classifier_source():
    from app.services.agent import rollout_metrics as metrics

    # The observed v1 conversational signal must track the classifier's
    # greeting label (drift fails closed loudly instead of fabricating
    # violations for greetings).
    from pathlib import Path

    nodes_source = (
        Path(__file__).resolve().parents[3]
        / "app"
        / "services"
        / "agent"
        / "nodes.py"
    ).read_text(encoding="utf-8")
    assert '"greeting": "Tin nhắn thông thường"' in nodes_source
    assert "Tin nhắn thông thường" in metrics.V1_GREETING_INTENT_DETAIL


def test_count_grounding_evidence_folds_terminal_signals():
    from app.services.agent import rollout_metrics as metrics

    assert metrics.count_grounding_evidence("", [], []) == 0
    # v1 inline markers ([xxxx]) count.
    assert metrics.count_grounding_evidence("Xem [ab12] nhé", [], []) == 1
    # v2 presented citations count.
    assert (
        metrics.count_grounding_evidence(
            "The decree states this.",
            [{"citation_id": "cite-1", "label": "L1"}],
            [],
        )
        == 1
    )
    # Served document ids count.
    assert (
        metrics.count_grounding_evidence("The decree states this.", [], ["doc-a"])
        == 1
    )


def test_intent_cache_hit_still_emits_greeting_signal():
    import asyncio

    from app.services.agent import nodes as nodes_module

    async def _run():
        queue: asyncio.Queue = asyncio.Queue()
        try:
            nodes_module._set_cached_intent(
                "xin chào canary",
                {
                    "intent": "greeting",
                    "rewritten_query": "",
                    "needs_tool": False,
                },
            )
            state = {
                "messages": [{"role": "user", "content": "xin chào canary"}],
                "_event_queue": queue,
            }
            result = await nodes_module.intent_classifier(state)
            assert result["intent"] == "greeting"
            details = []
            while not queue.empty():
                event = queue.get_nowait()
                if isinstance(event, tuple) and event[0] == "status":
                    details.append(event[1].get("detail", ""))
            assert any("Tin nhắn thông thường" in detail for detail in details)
        finally:
            try:
                nodes_module._INTENT_CACHE.pop(
                    nodes_module._get_cache_key("xin chào canary"), None
                )
            except Exception:
                pass

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())


def _gapped_24_bucket_rows():
    from datetime import UTC as _UTC

    start = datetime.now(_UTC) - timedelta(hours=100)
    rows = []
    # 12 consecutive hourly buckets, a 72h telemetry gap, 12 more buckets:
    # 24 distinct hours occupied but NOT consecutive coverage.
    moments = [start + timedelta(hours=index) for index in range(12)]
    moments += [start + timedelta(hours=12 + 72 + index) for index in range(12)]
    for arm in ("v1", "v2"):
        for index, moment in enumerate(moments):
            rows.append(
                _row(
                    arm=arm,
                    request_id_hash=f"{arm}-gap24-{index}",
                    started_at=moment,
                    finished_at=moment + timedelta(seconds=1),
                )
            )
    return rows


def test_gate_rejects_gapped_24_bucket_series():
    collector = _script_module("collect_v2_rollout_report")
    gate = _script_module("check_v2_rollout_gate")

    report = collector.summarize_metrics(_gapped_24_bucket_rows())
    assert report["arms"]["v1"]["continuous_hours"] == 24
    assert report["arms"]["v1"]["max_gap_hours"] >= 72
    passed, failures = gate.check_gate(report)
    assert passed is False
    assert any("gap" in failure for failure in failures)


def test_gate_rejects_missing_max_gap():
    collector = _script_module("collect_v2_rollout_report")
    gate = _script_module("check_v2_rollout_gate")

    report = collector.summarize_metrics(_synthetic_rows())
    del report["arms"]["v2"]["max_gap_hours"]
    passed, _ = gate.check_gate(report)
    assert passed is False

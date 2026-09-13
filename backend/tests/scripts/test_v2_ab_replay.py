"""Task 1 — golden session-SSE A/B preflight harness tests (TDD).

Offline unit suite for the Phase-3 rollout preflight: an authenticated
session-SSE driver with server-side arm selection plus a shared-evaluator
functional/quality comparison. No live backend, no LLM — all HTTP is faked.

Step 1 asserts: the admin-only evaluation endpoint exists; the driver
creates a session; sends server-side arm selection; reads named SSE events
to exactly one terminal; records latency/citations/status; redacts
auth/message PII; never sends client graph-version headers; non-admin
override attempts get 403.

Step 2 asserts: v1 and v2 outputs are evaluated by the SAME preflight
evaluator version; reports persist ``evaluator_version``; comparison is
rejected when versions differ.

Fix round 1 (R10) asserts: real URL composition from a parsed CLI config;
a real SSE frame parser on raw ``text/event-stream`` bytes; the session-SSE
smoke through the real stream endpoint; golden functional/quality scoring;
non-zero exits and no-false-pass comparison.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Fakes (no network)
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, payload) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, str):
            return json.loads(self._payload)
        return self._payload

    @property
    def text(self) -> str:
        if isinstance(self._payload, bytes):
            # Mirror a real transport: the wire yields bytes, `.text` decodes.
            return self._payload.decode("utf-8", errors="replace")
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload)


class FakeHttp:
    """Minimal injectable HTTP client recording every request."""

    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.routes: dict[tuple[str, str], FakeResponse] = {}

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes[(method.upper(), path)] = response

    def _record(self, method: str, path: str, headers, payload):
        self.posts.append(
            {
                "method": method,
                "path": path,
                "headers": dict(headers or {}),
                "json": payload,
            }
        )
        response = self.routes.get((method, path))
        if response is None:
            return FakeResponse(404, {"detail": "no fake route for " + path})
        return response

    def get(self, path: str, *, headers=None) -> FakeResponse:
        return self._record("GET", path, headers, None)

    def post(self, path: str, *, headers=None, json=None) -> FakeResponse:
        return self._record("POST", path, headers, json)

    def delete(self, path: str, *, headers=None) -> FakeResponse:
        return self._record("DELETE", path, headers, None)


def _terminal_complete(answer="Trả lời.", sources=None):
    return [
        {"event": "status", "data": {"step": "start", "detail": "begin"}},
        {"event": "token", "data": {"text": "Trả"}},
        {"event": "sources", "data": {"sources": sources or []}},
        {
            "event": "complete",
            "data": {"answer": answer, "sources": sources or []},
        },
    ]


def _raw_sse_bytes(answer="Trả lời theo Điều 17."):
    frames = [
        ": preflight comment",
        "event: status\ndata: {\"step\": \"start\"}",
        "event: token\ndata: {\"text\": \"Trả\"}",
        "event: sources\ndata: {\"sources\": [{\"document_number\": \"85/2016/NĐ-CP\"}]}",
        f"event: complete\ndata: {json.dumps({'answer': answer, 'sources': [{'document_number': '85/2016/NĐ-CP'}]}, ensure_ascii=False)}",
        "event: heartbeat\ndata: {}",
    ]
    return ("\n\n".join(frames) + "\n\n").encode("utf-8")


def _run_args(tmp_path, arm="v2", token="tok-123"):
    from scripts import ab_eval

    queries = tmp_path / "golden.yaml"
    queries.write_text(
        "cases:\n"
        "  - id: sec-85-d17\n"
        "    query: Tóm tắt điều 17 của Nghị định 85/2016\n"
        "    expect_document: \"85/2016%\"\n"
        "    expect_article: [17]\n"
        "    tags: [section-ref]\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        arm=arm,
        queries=str(queries),
        workspace="ws-1",
        out=str(tmp_path / "report.json"),
        base_url="http://localhost:8080",
        token=token,
    ), ab_eval


def _route_full_stack(http: FakeHttp, ab_eval, *, arm="v2", answer=None):
    answer = answer if answer is not None else "Trả lời theo Điều 17."
    sources = [
        {"document_number": "85/2016/NĐ-CP", "article_label": "Điều 17"}
    ]
    http.route(
        "POST", ab_eval.SESSIONS_PATH, FakeResponse(200, {"session_id": "sess-1"})
    )
    http.route(
        "POST",
        ab_eval.ADMIN_EVALUATE_PATH,
        FakeResponse(200, {"version": arm, "events": _terminal_complete(answer, sources)}),
    )
    http.route(
        "GET",
        ab_eval.ADMIN_STATUS_PATH,
        FakeResponse(200, {"configured_version": arm}),
    )
    http.route(
        "POST",
        ab_eval.session_stream_path("sess-1"),
        FakeResponse(200, _raw_sse_bytes(answer)),
    )
    http.route(
        "DELETE", ab_eval.session_path("sess-1"), FakeResponse(200, {})
    )


# ---------------------------------------------------------------------------
# Step 1 — admin-only evaluation endpoint
# ---------------------------------------------------------------------------


def test_admin_evaluate_endpoint_exists_and_requires_superadmin():
    from app.api.agent_admin import router
    from app.core.deps import require_superadmin

    matches = [
        route
        for route in router.routes
        if "/evaluate" in getattr(route, "path", "")
        and "POST" in (getattr(route, "methods", None) or set())
    ]
    assert matches, "POST /evaluate route missing on agent_admin router"
    route = matches[0]
    deps = list(getattr(route, "dependencies", None) or []) + list(
        getattr(getattr(route, "dependant", None), "dependencies", None) or []
    )
    calls = [getattr(dep, "call", None) for dep in deps]
    # Router-level guard also counts: it applies to every route.
    router_calls = [
        getattr(dep, "call", None)
        for dep in (getattr(router, "dependencies", None) or [])
    ]
    assert require_superadmin in (calls + router_calls)


def test_non_admin_arm_override_returns_403():
    from app.core.exceptions import ForbiddenError
    from app.services.agent.runtime_selector import resolve_request_version

    intruder = SimpleNamespace(is_superadmin=False)
    with pytest.raises(ForbiddenError) as excinfo:
        resolve_request_version(user=intruder, admin_override="v2")
    assert excinfo.value.status_code == 403


def test_driver_creates_session_and_selects_arm_server_side():
    from scripts import ab_eval

    http = FakeHttp()
    http.route(
        "POST",
        ab_eval.SESSIONS_PATH,
        FakeResponse(200, {"session_id": "sess-1"}),
    )
    http.route(
        "POST",
        ab_eval.ADMIN_EVALUATE_PATH,
        FakeResponse(200, {"version": "v2", "events": _terminal_complete()}),
    )

    result = ab_eval.run_eval_turn(
        http,
        base_url="http://test",
        token="tok-123",
        arm="v2",
        query_id="sec-85-d17",
        message="Tóm tắt điều 17 của Nghị định 85/2016",
    )

    paths = [call["path"] for call in http.posts]
    assert ab_eval.SESSIONS_PATH in paths
    assert ab_eval.ADMIN_EVALUATE_PATH in paths
    eval_call = next(
        call for call in http.posts if call["path"] == ab_eval.ADMIN_EVALUATE_PATH
    )
    # Server-side arm selection: the arm travels in the admin endpoint body,
    # never in a client graph-version header.
    assert eval_call["json"]["version"] == "v2"
    assert result["status"] == "complete"


def test_driver_never_sends_client_graph_version_headers():
    from scripts import ab_eval

    http = FakeHttp()
    http.route(
        "POST", ab_eval.SESSIONS_PATH, FakeResponse(200, {"session_id": "s"})
    )
    http.route(
        "POST",
        ab_eval.ADMIN_EVALUATE_PATH,
        FakeResponse(200, {"version": "v1", "events": _terminal_complete()}),
    )
    ab_eval.run_eval_turn(
        http,
        base_url="http://test",
        token="tok-123",
        arm="v1",
        query_id="q1",
        message="hello",
    )
    for call in http.posts:
        lowered = {key.lower() for key in call["headers"]}
        assert not (lowered & ab_eval.CLIENT_GRAPH_VERSION_HEADERS), (
            f"client graph-version header leaked on {call['path']}: "
            f"{sorted(lowered & ab_eval.CLIENT_GRAPH_VERSION_HEADERS)}"
        )


def test_named_sse_events_collapse_to_one_terminal():
    from scripts import ab_eval

    events = _terminal_complete(
        answer="A",
        sources=[{"document_number": "85/2016/NĐ-CP", "article": 17}],
    )
    terminal = ab_eval.collect_terminal(events)
    assert terminal["event"] == "complete"
    assert terminal["data"]["answer"] == "A"

    with pytest.raises(ValueError):
        ab_eval.collect_terminal(
            [dict(event="status", data={}), dict(event="token", data={})]
        )
    with pytest.raises(ValueError):
        ab_eval.collect_terminal(
            _terminal_complete()
            + [{"event": "error", "data": {"message": "x"}}]
        )


def test_turn_records_latency_citations_status():
    from scripts import ab_eval

    http = FakeHttp()
    http.route(
        "POST", ab_eval.SESSIONS_PATH, FakeResponse(200, {"session_id": "s"})
    )
    http.route(
        "POST",
        ab_eval.ADMIN_EVALUATE_PATH,
        FakeResponse(
            200,
            {
                "version": "v1",
                "events": _terminal_complete(
                    sources=[{"document_number": "85/2016/NĐ-CP"}]
                ),
            },
        ),
    )
    result = ab_eval.run_eval_turn(
        http,
        base_url="http://test",
        token="tok",
        arm="v1",
        query_id="sec-85-d17",
        message="q",
    )
    assert result["latency_ms"] >= 0
    assert result["citations"] == ["85/2016/NĐ-CP"]
    assert result["status"] == "complete"
    assert result["arm"] == "v1"


def test_reports_redact_auth_and_message_pii():
    from scripts import ab_eval

    report = ab_eval.build_arm_report(
        arm="v1",
        cases=[
            {
                "query_id": "q1",
                "message": "super secret user text",
                "latency_ms": 5,
                "citations": [],
                "status": "complete",
                "answer": "super secret user text echoed",
            }
        ],
        auth_token="Bearer tok-ultra-secret",
    )
    dumped = json.dumps(report)
    assert "tok-ultra-secret" not in dumped
    assert "super secret user text" not in dumped
    assert report["evaluator_version"] == ab_eval.EVALUATOR_VERSION


# ---------------------------------------------------------------------------
# Step 2 — shared evaluator version
# ---------------------------------------------------------------------------


def test_v1_and_v2_share_one_evaluator_version():
    from scripts import ab_eval
    from scripts import replay_v2

    assert replay_v2.EVALUATOR_VERSION == ab_eval.EVALUATOR_VERSION
    v1 = ab_eval.evaluate_output(
        answer="A", sources=[], status="complete", arm="v1"
    )
    v2 = replay_v2.replay_transcript(_terminal_complete(answer="A"), arm="v2")
    assert v1["evaluator_version"] == v2["evaluator_version"]


def _complete_report(ab_eval, arm="v1", query_id="sec-85-d17"):
    return {
        "arm": arm,
        "evaluator_version": ab_eval.EVALUATOR_VERSION,
        "cases": [
            {
                "query_id": query_id,
                "status": "complete",
                "arm": arm,
                "functional": {
                    "doc_hit": True,
                    "article_hit": True,
                    "negative_pass": None,
                    "functional_pass": True,
                },
            }
        ],
    }


def test_compare_rejects_mismatched_evaluator_versions():
    from scripts import ab_eval

    base = _complete_report(ab_eval)
    stale = dict(base, evaluator_version="preflight-v0")
    ab_eval.compare_reports(base, dict(base))  # same version: fine
    with pytest.raises(ValueError, match="evaluator_version"):
        ab_eval.compare_reports(base, stale)


# ---------------------------------------------------------------------------
# Fix round 1 (R10) — real boundaries, golden quality, no false pass
# ---------------------------------------------------------------------------


def test_url_composition_from_cli_config_has_no_duplicate_api_prefix():
    from scripts import ab_eval

    args = ab_eval.build_parser().parse_args(
        [
            "run",
            "--arm",
            "v2",
            "--queries",
            "tests/retrieval/datasets/golden_retrieval.yaml",
            "--workspace",
            "ws-1",
            "--out",
            "o.json",
        ]
    )
    assert args.base_url == "http://localhost:8080"
    assert (
        ab_eval.compose_url(args.base_url, ab_eval.SESSIONS_PATH)
        == "http://localhost:8080/api/v1/rag/chat/sessions"
    )
    assert (
        ab_eval.compose_url(args.base_url, ab_eval.session_stream_path("s-1"))
        == "http://localhost:8080/api/v1/rag/chat/sessions/s-1/stream"
    )
    assert (
        ab_eval.compose_url(args.base_url, ab_eval.ADMIN_EVALUATE_PATH)
        == "http://localhost:8080/api/v1/admin/agent/evaluate"
    )
    assert (
        ab_eval.compose_url(args.base_url, ab_eval.ADMIN_STATUS_PATH)
        == "http://localhost:8080/api/v1/admin/agent/status"
    )


def test_sse_frame_parser_reads_raw_event_stream_bytes():
    from scripts import ab_eval

    events = ab_eval.parse_sse_stream(_raw_sse_bytes())
    assert [event["event"] for event in events] == [
        "status",
        "token",
        "sources",
        "complete",
        "heartbeat",
    ]
    terminal = ab_eval.collect_first_terminal(events)
    assert terminal["event"] == "complete"
    assert terminal["data"]["answer"] == "Trả lời theo Điều 17."
    assert terminal["data"]["sources"] == [
        {"document_number": "85/2016/NĐ-CP"}
    ]
    with pytest.raises(ValueError):
        ab_eval.collect_first_terminal(
            [{"event": "status", "data": {}}]
        )


def test_run_cmd_drives_session_sse_smoke_and_exits_zero(tmp_path):
    from scripts import ab_eval

    args, _ = _run_args(tmp_path)
    http = FakeHttp()
    _route_full_stack(http, ab_eval)
    code = ab_eval._cmd_run(args, client=http)
    assert code == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["complete_count"] == 1
    case = report["cases"][0]
    assert case["functional"]["doc_hit"] is True
    assert case["functional"]["article_hit"] is True
    assert case["functional"]["functional_pass"] is True
    smoke = report["session_sse"]
    assert smoke["ok"] is True
    assert smoke["session_id"] == "sess-1"
    assert smoke["configured_version"] == "v2"
    stream_call = next(
        call
        for call in http.posts
        if call["path"] == ab_eval.session_stream_path("sess-1")
    )
    assert stream_call["headers"]["Accept"] == "text/event-stream"
    assert "tok-123" not in json.dumps(report)
    assert any(
        call["method"] == "DELETE"
        and call["path"] == ab_eval.session_path("sess-1")
        for call in http.posts
    )


def test_run_cmd_fails_on_arm_mismatch(tmp_path):
    from scripts import ab_eval

    args, _ = _run_args(tmp_path, arm="v2")
    http = FakeHttp()
    _route_full_stack(http, ab_eval, arm="v1")  # server echoes the wrong arm
    code = ab_eval._cmd_run(args, client=http)
    assert code != 0


def test_run_cmd_records_transport_error_and_fails(tmp_path):
    from scripts import ab_eval

    args, _ = _run_args(tmp_path)
    http = FakeHttp()  # no routes: every request 404s / breaks
    code = ab_eval._cmd_run(args, client=http)
    assert code != 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["cases"][0]["status"].startswith("error:")
    assert report["summary"]["complete_count"] == 0


def test_golden_functional_scores_doc_article_and_negative():
    from scripts import ab_eval

    positive = ab_eval.evaluate_functional(
        {"expect_document": "85/2016%", "expect_article": [17]},
        citations=["85/2016/NĐ-CP"],
        answer="Theo Điều 17 của nghị định.",
        status="complete",
        sources=[{"document_number": "85/2016/NĐ-CP", "article_label": "Điều 17"}],
    )
    assert positive["doc_hit"] is True
    assert positive["article_hit"] is True
    assert positive["functional_pass"] is True

    missed_article = ab_eval.evaluate_functional(
        {"expect_document": "85/2016%", "expect_article": [20]},
        citations=["85/2016/NĐ-CP"],
        answer="Theo Điều 17 của nghị định.",
        status="complete",
        sources=[{"document_number": "85/2016/NĐ-CP", "article_label": "Điều 17"}],
    )
    assert missed_article["article_hit"] is False
    assert missed_article["article_missing"] == [20]
    assert missed_article["functional_pass"] is False

    refused = ab_eval.evaluate_functional(
        {"negative": True},
        citations=[],
        answer="Tôi không tìm thấy thông tin trong kho tài liệu.",
        status="complete",
    )
    assert refused["negative_pass"] is True

    hallucinated = ab_eval.evaluate_functional(
        {"negative": True},
        citations=["116/2025/NĐ-CP"],
        answer="Nghị định 13/2023 quy định về bảo vệ dữ liệu cá nhân.",
        status="complete",
    )
    assert hallucinated["negative_pass"] is False
    assert hallucinated["functional_pass"] is False


def test_compare_regresses_on_functional_loss():
    from scripts import ab_eval

    good = _complete_report(ab_eval)
    bad = _complete_report(ab_eval, arm="v2")
    bad["cases"][0]["functional"]["doc_hit"] = False
    bad["cases"][0]["functional"]["functional_pass"] = False
    result = ab_eval.compare_reports(good, bad)
    assert result["regressions"] == [
        {"query_id": "sec-85-d17", "lost": ["functional", "doc_hit"]}
    ]


def test_compare_refuses_incomparable_reports():
    from scripts import ab_eval

    good = _complete_report(ab_eval)
    assert ab_eval.compare_reports(good, _complete_report(ab_eval))["compared"] == 1
    with pytest.raises(ValueError, match="incomparable"):
        ab_eval.compare_reports(good, _complete_report(ab_eval, query_id="other"))
    with pytest.raises(ValueError, match="incomparable"):
        empty = dict(good, cases=[])
        ab_eval.compare_reports(empty, _complete_report(ab_eval))


# ---------------------------------------------------------------------------
# Fix round 2 (R11) — article provenance + answer-presence quality
# ---------------------------------------------------------------------------


def test_article_passes_via_expect_document_plus_article_label():
    from scripts import ab_eval

    result = ab_eval.evaluate_functional(
        {
            "expect_document": "85/2016%",
            "accept_documents": ["361/2025%"],
            "expect_article": [17],
        },
        citations=["85/2016/NĐ-CP"],
        answer="Tóm tắt nội dung.",  # prose alone proves nothing
        status="complete",
        sources=[
            {"document_number": "85/2016/NĐ-CP", "article_label": "Điều 17"}
        ],
    )
    assert result["article_basis"] == "provenance"
    assert result["article_hit"] is True
    assert result["article_missing"] == []
    assert result["functional_pass"] is True


def test_article_fails_when_only_accept_alternate_carries_it():
    from scripts import ab_eval

    result = ab_eval.evaluate_functional(
        {
            "expect_document": "85/2016%",
            "accept_documents": ["361/2025%"],
            "expect_article": [17],
        },
        citations=["361/2025/NĐ-CP"],
        answer="Theo Điều 17 của nghị định.",  # prose must not rescue it
        status="complete",
        sources=[
            {"document_number": "361/2025/NĐ-CP", "article_label": "Điều 17"}
        ],
    )
    assert result["doc_hit"] is True  # alternate still counts at doc level
    assert result["article_hit"] is False
    assert result["article_missing"] == [17]
    assert result["functional_pass"] is False


def test_article_indeterminate_with_no_provenance():
    from scripts import ab_eval

    result = ab_eval.evaluate_functional(
        {"expect_document": "85/2016%", "expect_article": [17]},
        citations=[],
        answer="Theo Điều 17 của nghị định.",  # prose alone is not proof
        status="complete",
        sources=[],
    )
    assert result["article_basis"] == "indeterminate"
    assert result["article_hit"] is None
    assert result["functional_pass"] is not True


def test_compare_regresses_when_answer_is_lost():
    from scripts import ab_eval

    good = _complete_report(ab_eval)
    good["cases"][0]["has_answer"] = True
    bad = _complete_report(ab_eval, arm="v2")
    bad["cases"][0]["has_answer"] = False
    bad["cases"][0]["status"] = "complete"  # status alone hides the loss
    result = ab_eval.compare_reports(good, bad)
    assert result["regressions"] == [
        {"query_id": "sec-85-d17", "lost": ["has_answer"]}
    ]

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
"""
from __future__ import annotations

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
            import json as _json

            return _json.loads(self._payload)
        return self._payload

    @property
    def text(self) -> str:
        if isinstance(self._payload, str):
            return self._payload
        import json as _json

        return _json.dumps(self._payload)


class FakeHttp:
    """Minimal injectable HTTP client recording every request."""

    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.routes: dict[tuple[str, str], FakeResponse] = {}

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes[(method.upper(), path)] = response

    def post(self, path: str, *, headers=None, json=None) -> FakeResponse:
        self.posts.append(
            {"path": path, "headers": dict(headers or {}), "json": json}
        )
        response = self.routes.get(("POST", path))
        if response is None:
            return FakeResponse(404, {"detail": "no fake route for " + path})
        return response


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
    dumped = __import__("json").dumps(report)
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


def test_compare_rejects_mismatched_evaluator_versions():
    from scripts import ab_eval

    base = {"evaluator_version": ab_eval.EVALUATOR_VERSION, "cases": []}
    stale = dict(base, evaluator_version="preflight-v0")
    ab_eval.compare_reports(base, dict(base))  # same version: fine
    with pytest.raises(ValueError, match="evaluator_version"):
        ab_eval.compare_reports(base, stale)

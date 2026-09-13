"""Golden session-SSE A/B preflight driver + shared-evaluator comparison.

Phase-3 Task 1: the shared evaluator-versioned preflight used by later
rollout gates. Two subcommands:

- ``run`` — drive one arm (``v1`` | ``v2``) over a query set through the
  authenticated session-SSE surface with **server-side arm selection** and
  write an evaluator-versioned JSON report.
- ``compare`` — diff two arm reports evaluated by the SAME preflight
  evaluator version; rejects comparison when versions differ.

Global-constraint compliance (binding):

- Arm selection is server-side only: the arm travels in the body of the
  authenticated superadmin evaluation endpoint
  (``POST /api/v1/admin/agent/evaluate`` → ``{"version": arm}``). This
  driver NEVER sends client graph-version headers — any such header raises
  instead of being transmitted (see ``CLIENT_GRAPH_VERSION_HEADERS``).
- The admin-eval endpoint is the ONLY per-request arm override.
- Every report persists ``evaluator_version`` (``EVALUATOR_VERSION`` — the
  single source of truth, shared with ``replay_v2``); ``compare`` rejects
  mismatched versions instead of diffing across evaluators.

Live usage (Compose stack: live backend + providers + AB_TOKEN)::

    make ab ARM=v1 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml \\
        WORKSPACE=$WORKSPACE
    make ab ARM=v2 QUERIES=tests/retrieval/datasets/golden_retrieval.yaml \\
        WORKSPACE=$WORKSPACE

Offline unit suite (no backend, no LLM)::

    python -m pytest tests/scripts/test_v2_ab_replay.py -q
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from typing import Any

# ---------------------------------------------------------------------------
# Shared preflight contract (Step 2: one evaluator version for both arms)
# ---------------------------------------------------------------------------

#: The single preflight evaluator version. Persisted into every report;
#: ``compare`` rejects reports whose versions differ. ``replay_v2`` imports
#: this name — never redefines it — so v1 and v2 outputs are always judged
#: by the SAME evaluator.
EVALUATOR_VERSION = "preflight-v1"

#: Named SSE events the session stream may emit (v1-wire contract, see
#: ``backend/app/services/agent/streaming.py``). Exactly one terminal event
#: ends a turn.
NAMED_SSE_EVENTS = frozenset(
    {
        "status",
        "thinking",
        "sources",
        "images",
        "token",
        "token_rollback",
        "potential_abbreviations",
        "complete",
        "error",
        "heartbeat",
    }
)

#: Terminal events: exactly one must close a turn.
TERMINAL_EVENTS = ("complete", "error")

#: Client graph-version headers are FORBIDDEN on the wire — arm selection is
#: server-side only. Any request carrying one of these raises instead of
#: being sent.
CLIENT_GRAPH_VERSION_HEADERS = frozenset(
    {
        "x-agent-graph-version",
        "x-graph-version",
        "x-nexusrag-agent-version",
        "x-nexusrag-graph-version",
    }
)

SESSIONS_PATH = "/api/v1/rag/chat/sessions"
ADMIN_EVALUATE_PATH = "/api/v1/admin/agent/evaluate"

_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
REDACTED = "[REDACTED]"


class EvaluatorVersionMismatch(ValueError):
    """Reports were produced by different preflight evaluator versions."""


# ---------------------------------------------------------------------------
# Headers: auth only, never a client graph-version header
# ---------------------------------------------------------------------------


def assert_no_client_version_headers(headers: dict[str, str]) -> None:
    """Fail closed if a client graph-version header is present."""
    lowered = {str(key).lower() for key in headers}
    leaked = sorted(lowered & CLIENT_GRAPH_VERSION_HEADERS)
    if leaked:
        raise ValueError(
            "client graph-version headers are forbidden "
            f"(arm selection is server-side only): {leaked}"
        )


def build_auth_headers(token: str) -> dict[str, str]:
    """Auth headers for harness requests: bearer only, nothing else."""
    headers = {"Authorization": f"Bearer {token}"}
    assert_no_client_version_headers(headers)
    return headers


# ---------------------------------------------------------------------------
# Minimal HTTP client (injectable for offline tests)
# ---------------------------------------------------------------------------


class UrllibHttpClient:
    """Live HTTP client over stdlib urllib (used by ``run``)."""

    def __init__(self, base_url: str, timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self, method: str, path: str, *, headers=None, payload=None
    ) -> Any:
        assert_no_client_version_headers(dict(headers or {}))
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json", **dict(headers or {})},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return SimpleResponse(response.status, response.read().decode())

    def post(self, path: str, *, headers=None, json=None) -> Any:
        return self._request("POST", path, headers=headers, payload=json)


class SimpleResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self):
        return json.loads(self.text)


# ---------------------------------------------------------------------------
# Driver (Step 1: session-SSE preflight, server-side arm selection)
# ---------------------------------------------------------------------------


def create_session(
    client: Any,
    *,
    base_url: str,
    token: str,
    workspace_id: str | None = None,
) -> str:
    """Create a chat session; returns the session id.

    Sends auth headers only — never a client graph-version header.
    """
    del base_url  # paths are absolute; the client owns the host.
    headers = build_auth_headers(token)
    payload: dict[str, Any] = {}
    if workspace_id:
        payload["workspace_id"] = workspace_id
    response = client.post(SESSIONS_PATH, headers=headers, json=payload)
    body = response.json()
    session_id = body.get("session_id") or body.get("id")
    if not session_id:
        raise ValueError(f"session creation returned no id: {body!r}")
    return str(session_id)


def evaluate_arm(
    client: Any,
    *,
    base_url: str,
    token: str,
    arm: str,
    message: str,
    workspace_id: str | None = None,
) -> tuple[str, list[dict]]:
    """Run one turn on ``arm`` via the admin-only evaluation endpoint.

    The arm travels in the request BODY (``{"version": arm}``) — server-side
    selection; the ONLY per-request override. Non-admin callers get 403 from
    the server (``require_superadmin``).
    """
    del base_url
    normalized = arm.strip().lower()
    if normalized not in ("v1", "v2"):
        raise ValueError(f"arm must be 'v1' or 'v2'; got {arm!r}")
    headers = build_auth_headers(token)
    payload: dict[str, Any] = {"message": message, "version": normalized}
    if workspace_id:
        payload["workspace_ids"] = [workspace_id]
    response = client.post(ADMIN_EVALUATE_PATH, headers=headers, json=payload)
    if response.status_code == 403:
        raise PermissionError("admin evaluation endpoint returned 403")
    body = response.json()
    return str(body.get("version", normalized)), list(body.get("events", []))


def collect_terminal(events: list[dict]) -> dict:
    """Collapse named SSE events to exactly ONE terminal event.

    Raises ``ValueError`` when there is no terminal event or more than one
    (a turn must end exactly once — ``complete`` or ``error``).
    """
    terminals = [event for event in events if event.get("event") in TERMINAL_EVENTS]
    if len(terminals) != 1:
        raise ValueError(
            f"expected exactly one terminal event {TERMINAL_EVENTS}; "
            f"got {len(terminals)} of {len(events)} events"
        )
    return terminals[0]


def extract_citations(terminal_data: dict) -> list[str]:
    """Citation document numbers recorded from the terminal event sources."""
    citations: list[str] = []
    for source in terminal_data.get("sources") or []:
        if not isinstance(source, dict):
            continue
        doc = source.get("document_number") or source.get("document")
        if doc and doc not in citations:
            citations.append(str(doc))
    return citations


def evaluate_output(
    *, answer: str, sources: list, status: str, arm: str
) -> dict:
    """Judge one arm output with the SHARED preflight evaluator.

    Both v1 and v2 outputs flow through this function (v2 via
    ``replay_v2.replay_transcript``), so the persisted ``evaluator_version``
    is identical by construction.
    """
    citations: list[str] = []
    for source in sources or []:
        if isinstance(source, dict):
            doc = source.get("document_number") or source.get("document")
        else:
            doc = source
        if doc and str(doc) not in citations:
            citations.append(str(doc))
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "arm": arm,
        "status": status,
        "answer_chars": len(answer or ""),
        "citation_count": len(citations),
        "citations": citations,
        "has_answer": bool((answer or "").strip()),
    }


def run_eval_turn(
    client: Any,
    *,
    base_url: str,
    token: str,
    arm: str,
    query_id: str,
    message: str,
    workspace_id: str | None = None,
) -> dict:
    """One preflight turn: session → server-side arm eval → terminal record.

    Records latency/citations/status and redacts auth/message PII before
    returning (the raw message text and bearer token never leave this
    function in the clear).
    """
    started = time.monotonic()
    session_id = create_session(
        client, base_url=base_url, token=token, workspace_id=workspace_id
    )
    version, events = evaluate_arm(
        client,
        base_url=base_url,
        token=token,
        arm=arm,
        message=message,
        workspace_id=workspace_id,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    terminal = collect_terminal(events)
    data = terminal.get("data") or {}
    evaluation = evaluate_output(
        answer=str(data.get("answer", "")),
        sources=list(data.get("sources", [])),
        status=terminal["event"],
        arm=version,
    )
    record = {
        "query_id": query_id,
        "session_id": session_id,
        "arm": version,
        "latency_ms": max(latency_ms, 0),
        "citations": evaluation["citations"],
        "status": evaluation["status"],
        "evaluator_version": EVALUATOR_VERSION,
    }
    return redact_record(record, secrets=(token, message))


# ---------------------------------------------------------------------------
# PII redaction: auth tokens + raw message text never persist in reports
# ---------------------------------------------------------------------------


def redact_record(record: dict, *, secrets: tuple[str, ...] = ()) -> dict:
    """Scrub auth tokens and raw message text from a case record."""
    redacted = dict(record)
    for key in ("message", "answer", "auth_token", "token", "password"):
        if key in redacted:
            redacted[key] = REDACTED
    dumped = json.dumps(redacted, ensure_ascii=False)
    for secret in secrets:
        if secret and len(secret) >= 2 and secret != REDACTED:
            dumped = dumped.replace(secret, REDACTED)
    dumped = _BEARER_RE.sub(f"Bearer {REDACTED}", dumped)
    return json.loads(dumped)


def build_arm_report(
    *, arm: str, cases: list[dict], auth_token: str | None = None
) -> dict:
    """Assemble a redacted, evaluator-versioned arm report."""
    secrets = tuple(
        secret
        for case in cases
        for secret in (case.get("message"), case.get("answer"), auth_token)
        if isinstance(secret, str)
    )
    clean = [redact_record(dict(case), secrets=secrets) for case in cases]
    latencies = [case.get("latency_ms", 0) for case in clean]
    return {
        "arm": arm,
        "evaluator_version": EVALUATOR_VERSION,
        "cases": clean,
        "summary": {
            "case_count": len(clean),
            "complete_count": sum(
                1 for case in clean if case.get("status") == "complete"
            ),
            "mean_latency_ms": (
                sum(latencies) / len(latencies) if latencies else 0
            ),
        },
    }


# ---------------------------------------------------------------------------
# Comparison (Step 2: same evaluator version or refuse)
# ---------------------------------------------------------------------------


def compare_reports(report_a: dict, report_b: dict) -> dict:
    """Diff two arm reports; reject when evaluator versions differ."""
    version_a = report_a.get("evaluator_version")
    version_b = report_b.get("evaluator_version")
    if version_a != version_b:
        raise EvaluatorVersionMismatch(
            "evaluator_version mismatch "
            f"({version_a!r} vs {version_b!r}): both arms must be judged by "
            "the SAME preflight evaluator version"
        )
    by_id_b = {case["query_id"]: case for case in report_b.get("cases", [])}
    regressions: list[str] = []
    compared = 0
    for case in report_a.get("cases", []):
        other = by_id_b.get(case.get("query_id"))
        if other is None:
            continue
        compared += 1
        if case.get("status") == "complete" and other.get("status") != "complete":
            regressions.append(case.get("query_id"))
    return {
        "evaluator_version": version_a,
        "arm_a": report_a.get("arm"),
        "arm_b": report_b.get("arm"),
        "compared": compared,
        "regressions": regressions,
    }


# ---------------------------------------------------------------------------
# CLI: run + compare
# ---------------------------------------------------------------------------


def _resolve_token(args) -> str:
    import os

    if args.token:
        return args.token
    env_token = os.environ.get("AB_TOKEN")
    if env_token:
        return env_token
    user = os.environ.get("AB_USER")
    password = os.environ.get("AB_PASSWORD")
    if user and password:
        return _login(args.base_url, user, password)
    raise SystemExit(
        "no auth: pass --token or export AB_TOKEN (or AB_USER + AB_PASSWORD)"
    )


def _login(base_url: str, user: str, password: str) -> str:
    payload = json.dumps({"email": user, "password": password}).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/auth/login",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode())
    token = body.get("access_token") or body.get("token")
    if not token:
        raise SystemExit("login succeeded but returned no token")
    return token


def _load_queries(path: str) -> list[dict]:
    import yaml  # test-only/live dep; unit tests never touch this path

    with open(path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    cases = doc.get("cases", doc if isinstance(doc, list) else [])
    queries = []
    for case in cases:
        if isinstance(case, dict) and case.get("query"):
            queries.append(case)
    return queries


def _cmd_run(args) -> int:
    client = UrllibHttpClient(args.base_url)
    token = _resolve_token(args)
    queries = _load_queries(args.queries)
    cases: list[dict] = []
    for case in queries:
        query_id = str(case.get("id", case.get("query_id", "query")))
        try:
            record = run_eval_turn(
                client,
                base_url=args.base_url,
                token=token,
                arm=args.arm,
                query_id=query_id,
                message=str(case["query"]),
                workspace_id=args.workspace,
            )
        except Exception as exc:  # noqa: BLE001 — one bad case != dead arm
            record = redact_record(
                {
                    "query_id": query_id,
                    "arm": args.arm,
                    "latency_ms": 0,
                    "citations": [],
                    "status": f"error: {exc}",
                    "evaluator_version": EVALUATOR_VERSION,
                },
                secrets=(token, str(case.get("query", ""))),
            )
        cases.append(record)
    report = build_arm_report(arm=args.arm, cases=cases, auth_token=token)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        f"arm={report['arm']} evaluator={report['evaluator_version']} "
        f"cases={len(cases)} out={args.out}"
    )
    return 0


def _cmd_compare(args) -> int:
    with open(args.a, encoding="utf-8") as handle:
        report_a = json.load(handle)
    with open(args.b, encoding="utf-8") as handle:
        report_b = json.load(handle)
    try:
        result = compare_reports(report_a, report_b)
    except EvaluatorVersionMismatch as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    if result["regressions"]:
        print(f"REGRESSIONS: {result['regressions']}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Golden session-SSE A/B preflight (Phase-3 Task 1)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Drive one arm over a query set.")
    run.add_argument("--arm", required=True, choices=("v1", "v2"))
    run.add_argument("--queries", required=True, help="Golden YAML query set.")
    run.add_argument("--workspace", required=True, help="Workspace UUID.")
    run.add_argument("--out", required=True, help="Report JSON path.")
    run.add_argument("--base-url", default="http://localhost:8080/api/v1")
    run.add_argument("--token", default=None)
    run.set_defaults(func=_cmd_run)

    compare = sub.add_parser("compare", help="Diff two arm reports.")
    compare.add_argument("a", help="First arm report JSON.")
    compare.add_argument("b", help="Second arm report JSON.")
    compare.add_argument("--out", default=None, help="Diff JSON path.")
    compare.set_defaults(func=_cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Golden session-SSE A/B preflight driver + shared-evaluator comparison.

Phase-3 Task 1: the shared evaluator-versioned preflight used by later
rollout gates. Two subcommands:

- ``run`` — drive one arm (``v1`` | ``v2``) over a query set through the
  authenticated session-SSE surface with **server-side arm selection** and
  write an evaluator-versioned JSON report.
- ``compare`` — diff two arm reports evaluated by the SAME preflight
  evaluator version; rejects comparison when versions differ or when the
  reports are incomparable.

Global-constraint compliance (binding):

- Arm selection is server-side only: the measured arm results come from the
  authenticated superadmin evaluation endpoint
  (``POST /api/v1/admin/agent/evaluate`` → ``{"version": arm}``). This
  driver NEVER sends client graph-version headers — any such header raises
  instead of being transmitted (see ``CLIENT_GRAPH_VERSION_HEADERS``).
- The admin-eval endpoint is the ONLY per-request arm override.
- Every report persists ``evaluator_version`` (``EVALUATOR_VERSION`` — the
  single source of truth, shared with ``replay_v2``); ``compare`` rejects
  mismatched versions instead of diffing across evaluators.
- ``--base-url`` is a server ORIGIN with no ``/api/v1`` suffix (default
  ``http://localhost:8080``); every route constant carries the full
  ``/api/v1/...`` prefix and URLs are composed with ``compose_url``.

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
import fnmatch
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
ADMIN_STATUS_PATH = "/api/v1/admin/agent/status"

DEFAULT_BASE_URL = "http://localhost:8080"

_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
REDACTED = "[REDACTED]"

_REFUSAL_RE = re.compile(
    r"không tìm thấy|không có thông tin|ngoài phạm vi|không thể trả lời"
    r"|no information|cannot (find|answer)|not found|out of scope"
    r"|insufficient (sources|information)",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r"[Đđ]iều\s+(\d+)")


class EvaluatorVersionMismatch(ValueError):
    """Reports were produced by different preflight evaluator versions."""


class CompareRefused(ValueError):
    """Reports are incomparable (no overlap or no usable baseline)."""


# ---------------------------------------------------------------------------
# URL composition: origin + full /api/v1/... route (R10.1)
# ---------------------------------------------------------------------------


def compose_url(base_url: str, path: str) -> str:
    """Join a server origin with a full ``/api/v1/...`` route path."""
    if not path.startswith("/"):
        raise ValueError(f"route path must start with '/': {path!r}")
    return base_url.rstrip("/") + path


def session_stream_path(session_id: str) -> str:
    """Session SSE endpoint carrying the created session's id."""
    return f"{SESSIONS_PATH}/{session_id}/stream"


def session_path(session_id: str) -> str:
    """Single-session route (delete-after-smoke)."""
    return f"{SESSIONS_PATH}/{session_id}"


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


def build_sse_headers(token: str) -> dict[str, str]:
    """Headers for the session SSE stream: bearer + event-stream Accept."""
    headers = {**build_auth_headers(token), "Accept": "text/event-stream"}
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
            compose_url(self.base_url, path),
            data=data,
            headers={"Content-Type": "application/json", **dict(headers or {})},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return SimpleResponse(response.status, response.read().decode())

    def get(self, path: str, *, headers=None) -> Any:
        return self._request("GET", path, headers=headers)

    def post(self, path: str, *, headers=None, json=None) -> Any:
        return self._request("POST", path, headers=headers, payload=json)

    def delete(self, path: str, *, headers=None) -> Any:
        return self._request("DELETE", path, headers=headers)


class SimpleResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self):
        return json.loads(self.text)


# ---------------------------------------------------------------------------
# Raw SSE frame parser (session stream is text/event-stream, not JSON)
# ---------------------------------------------------------------------------


def parse_sse_stream(raw: bytes | str) -> list[dict]:
    """Parse raw ``text/event-stream`` bytes into named ``{event, data}``.

    Frames are split on blank lines; ``event:`` names the frame (default
    ``message``), consecutive ``data:`` lines join with ``\\n``, ``:`` comment
    lines are skipped, and each data payload is JSON-decoded when possible
    (kept as a raw string otherwise).
    """
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    events: list[dict] = []
    name: str | None = None
    data_lines: list[str] = []

    def _flush() -> None:
        nonlocal name, data_lines
        if name is None and not data_lines:
            return
        payload = "\n".join(data_lines)
        try:
            data: Any = json.loads(payload) if payload else {}
        except ValueError:
            data = payload
        events.append({"event": name or "message", "data": data})
        name, data_lines = None, []

    for line in text.splitlines():
        if not line.strip():
            _flush()
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.lstrip(" ")
        if field == "event":
            # A new event line without an intervening blank line still
            # terminates the previous frame (tolerant split).
            if name is not None or data_lines:
                _flush()
            name = value or "message"
        elif field == "data":
            data_lines.append(value)
    _flush()
    return events


def collect_first_terminal(events: list[dict]) -> dict:
    """First terminal event of a parsed SSE frame list.

    The live session stream is consumed to its first terminal (``complete``
    or ``error``); later frames (e.g. heartbeats after close) are ignored.
    """
    for event in events:
        if event.get("event") in TERMINAL_EVENTS:
            return event
    raise ValueError(
        f"stream ended with no terminal event {TERMINAL_EVENTS}; "
        f"got {[event.get('event') for event in events]}"
    )


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
    payload: dict[str, Any] = {"title": "AB preflight"}
    if workspace_id:
        payload["workspace_id"] = workspace_id
    response = client.post(SESSIONS_PATH, headers=headers, json=payload)
    body = response.json()
    session_id = body.get("session_id") or body.get("id")
    if not session_id:
        raise ValueError(f"session creation returned no id: {body!r}")
    return str(session_id)


def delete_session(client: Any, *, token: str, session_id: str) -> None:
    """Best-effort delete of the smoke session (route exists: 204/200)."""
    try:
        client.delete(session_path(session_id), headers=build_auth_headers(token))
    except Exception:  # noqa: BLE001 — cleanup must not fail the report
        pass


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


def get_configured_arm(client: Any, *, token: str) -> str:
    """Probe the server-owned configured arm (no client selection exists)."""
    response = client.get(ADMIN_STATUS_PATH, headers=build_auth_headers(token))
    if response.status_code == 403:
        raise PermissionError("admin status endpoint returned 403")
    return str(response.json().get("configured_version", "")).strip().lower()


def run_session_sse_smoke(
    client: Any,
    *,
    token: str,
    arm: str,
    message: str,
    workspace_id: str | None = None,
) -> dict:
    """One turn through the REAL session SSE endpoint (R10.4).

    Creates a session, POSTs ``{session_id}/stream`` with
    ``Accept: text/event-stream``, parses raw ``event:``/``data:`` frames to
    the first terminal, deletes the session, and records the smoke. The
    stream runs on the server-configured arm: the smoke fails when
    ``configured_version != --arm``.
    """
    started = time.monotonic()
    smoke: dict[str, Any] = {
        "session_id": None,
        "configured_version": None,
        "terminal_event": None,
        "latency_ms": 0,
        "ok": False,
    }
    try:
        configured = get_configured_arm(client, token=token)
        smoke["configured_version"] = configured
        if configured != arm:
            smoke["error"] = (
                f"server configured_version={configured!r} != --arm {arm!r}"
            )
            return smoke
        session_id = create_session(
            client, base_url="", token=token, workspace_id=workspace_id
        )
        smoke["session_id"] = session_id
        response = client.post(
            session_stream_path(session_id),
            headers=build_sse_headers(token),
            json={"message": message},
        )
        events = parse_sse_stream(response.text)
        terminal = collect_first_terminal(events)
        smoke["terminal_event"] = terminal.get("event")
        smoke["ok"] = terminal.get("event") == "complete"
        if not smoke["ok"]:
            smoke["error"] = f"smoke terminal was {terminal.get('event')!r}"
    except Exception as exc:  # noqa: BLE001 — smoke failure, not a crash
        smoke["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        smoke["latency_ms"] = max(int((time.monotonic() - started) * 1000), 0)
        if smoke["session_id"]:
            delete_session(client, token=token, session_id=smoke["session_id"])
    return smoke


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


def extract_source_pairs(sources: list) -> list[dict]:
    """Preserve per-source provenance: ``(document_number, article_label)``.

    Sources are NOT flattened to bare document-number strings — the golden
    article check needs the document each article label came from. Bare
    string entries carry a document but no article provenance.
    """
    pairs: list[dict] = []
    for source in sources or []:
        if isinstance(source, dict):
            doc = source.get("document_number") or source.get("document")
            pairs.append(
                {
                    "document_number": str(doc) if doc else None,
                    "article_label": source.get("article_label"),
                }
            )
        elif source:
            pairs.append({"document_number": str(source), "article_label": None})
    return pairs


def article_label_matches(label: Any, number: int) -> bool:
    """True when an ``article_label`` names article ``number``."""
    if label is None:
        return False
    text = str(label).strip()
    if text == str(number):
        return True
    return number in {int(found) for found in _ARTICLE_RE.findall(text)}


def evaluate_output(
    *, answer: str, sources: list, status: str, arm: str
) -> dict:
    """Judge one arm output with the SHARED preflight evaluator.

    Both v1 and v2 outputs flow through this function (v2 via
    ``replay_v2.replay_transcript``), so the persisted ``evaluator_version``
    is identical by construction. ``sources`` keeps full provenance pairs;
    ``citations`` stays the de-duplicated document-number list.
    """
    pairs = extract_source_pairs(sources)
    citations: list[str] = []
    for pair in pairs:
        if pair["document_number"] and pair["document_number"] not in citations:
            citations.append(pair["document_number"])
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "arm": arm,
        "status": status,
        "answer_chars": len(answer or ""),
        "citation_count": len(citations),
        "citations": citations,
        "sources": pairs,
        "has_answer": bool((answer or "").strip()),
    }


# ---------------------------------------------------------------------------
# Golden functional evaluation (R10.5: document/article/negative quality)
# ---------------------------------------------------------------------------


def match_doc_pattern(citation: str, pattern: str) -> bool:
    """SQL-ILIKE match: ``%`` is a wildcard, case-insensitive."""
    cleaned = str(pattern).strip().strip("'\"").lower().replace("%", "*")
    return fnmatch.fnmatchcase(str(citation).lower(), cleaned)


def evaluate_functional(
    case: dict,
    *,
    citations: list[str],
    answer: str,
    status: str,
    sources: list | None = None,
) -> dict:
    """Judge a golden case's expectations against citations + provenance.

    ``expect_document``/``accept_documents`` (ILIKE ``%`` patterns),
    ``expect_article`` (valid ONLY within ``expect_document``: success needs
    a returned source whose document matches ``expect_document`` AND whose
    ``article_label`` names the expected article — an ``accept_documents``
    alternate never satisfies it; answer prose alone never does either),
    ``negative`` (refusal or non-complete terminal expected); ``tags``
    persist for filtering. With no source provenance at all the article
    result is ``indeterminate`` (``article_hit`` None, never a pass). A
    non-complete turn fails functionally by definition.
    """
    complete = status == "complete"
    negative = bool(case.get("negative", False))
    tags = list(case.get("tags", []) or [])
    expected_doc = case.get("expect_document")
    accepted = list(case.get("accept_documents", []) or [])
    expected_articles = list(case.get("expect_article", []) or [])

    doc_hit: bool | None = None
    if expected_doc is not None:
        patterns = [expected_doc, *accepted]
        doc_hit = any(
            match_doc_pattern(citation, pattern)
            for citation in citations
            for pattern in patterns
        )

    article_hit: bool | None = None
    article_missing: list[int] = []
    article_basis: str | None = None
    if expected_articles:
        provenance = [
            pair
            for pair in extract_source_pairs(sources or [])
            if pair["document_number"]
        ]
        if not complete:
            article_hit, article_basis = False, "provenance"
            article_missing = list(expected_articles)
        elif not provenance:
            # No provenance returned: cannot verify — indeterminate, not pass.
            article_hit, article_basis = None, "indeterminate"
            article_missing = list(expected_articles)
        else:
            article_basis = "provenance"
            article_missing = [
                number
                for number in expected_articles
                if not any(
                    match_doc_pattern(pair["document_number"], expected_doc)
                    and article_label_matches(pair["article_label"], number)
                    for pair in provenance
                )
            ]
            article_hit = not article_missing

    negative_pass: bool | None = None
    if negative:
        negative_pass = (not complete) or bool(_REFUSAL_RE.search(answer or ""))

    if negative:
        functional_pass: bool | None = bool(negative_pass)
    elif expected_doc is not None:
        functional_pass = bool(complete and doc_hit) and (
            article_hit is not False
        )
    else:
        functional_pass = bool(complete) if not expected_articles else None
        if expected_articles and functional_pass:
            functional_pass = bool(article_hit)

    return {
        "negative": negative,
        "tags": tags,
        "expect_document": expected_doc,
        "doc_hit": doc_hit,
        "expect_article": expected_articles,
        "article_hit": article_hit,
        "article_missing": article_missing,
        "article_basis": article_basis,
        "negative_pass": negative_pass,
        "functional_pass": functional_pass,
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
    case: dict | None = None,
) -> dict:
    """One preflight turn: session → server-side arm eval → terminal record.

    Records latency/citations/status plus quality stats and golden
    functional results; redacts auth/message PII before returning (raw
    message/answer text and bearer tokens never persist in the clear).
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
    answer = str(data.get("answer", ""))
    evaluation = evaluate_output(
        answer=answer,
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
        "answer_chars": evaluation["answer_chars"],
        "citation_count": evaluation["citation_count"],
        "has_answer": evaluation["has_answer"],
        "functional": evaluate_functional(
            case or {}, citations=evaluation["citations"],
            answer=answer, status=evaluation["status"],
            sources=evaluation["sources"],
        ),
        "evaluator_version": EVALUATOR_VERSION,
    }
    return redact_record(record, secrets=(token, message, answer))


def _failed_case(query_id: str, arm: str, detail: str, *, secrets=()) -> dict:
    return redact_record(
        {
            "query_id": query_id,
            "arm": arm,
            "latency_ms": 0,
            "citations": [],
            "status": f"error: {detail}",
            "answer_chars": 0,
            "citation_count": 0,
            "has_answer": False,
            "functional": evaluate_functional({}, citations=[], answer="",
                                             status="error"),
            "evaluator_version": EVALUATOR_VERSION,
        },
        secrets=tuple(secrets),
    )


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
    *, arm: str, cases: list[dict], auth_token: str | None = None,
    session_sse: dict | None = None,
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
    complete = sum(1 for case in clean if case.get("status") == "complete")
    smoke = dict(session_sse or {})
    if auth_token and smoke.get("error"):
        smoke = redact_record(smoke, secrets=(auth_token,))
    return {
        "arm": arm,
        "evaluator_version": EVALUATOR_VERSION,
        "cases": clean,
        "session_sse": smoke,
        "summary": {
            "case_count": len(clean),
            "complete_count": complete,
            "mean_latency_ms": (
                sum(latencies) / len(latencies) if latencies else 0
            ),
        },
    }


# ---------------------------------------------------------------------------
# Comparison (Step 2: same evaluator version or refuse; never false-pass)
# ---------------------------------------------------------------------------


def compare_reports(report_a: dict, report_b: dict) -> dict:
    """Diff two arm reports; reject version drift and incomparable pairs.

    Refuses (``EvaluatorVersionMismatch`` / ``CompareRefused``) when the
    evaluator versions differ, when there are no common query ids, or when
    either report has zero ``complete`` cases. Otherwise regresses on
    status, golden functional, document-hit, article-hit, and negative
    refusal losses — not only complete→error.
    """
    version_a = report_a.get("evaluator_version")
    version_b = report_b.get("evaluator_version")
    if version_a != version_b:
        raise EvaluatorVersionMismatch(
            "evaluator_version mismatch "
            f"({version_a!r} vs {version_b!r}): both arms must be judged by "
            "the SAME preflight evaluator version"
        )
    complete_a = sum(
        1 for case in report_a.get("cases", []) if case.get("status") == "complete"
    )
    complete_b = sum(
        1 for case in report_b.get("cases", []) if case.get("status") == "complete"
    )
    if complete_a == 0 or complete_b == 0:
        raise CompareRefused(
            f"incomparable reports: complete counts are {complete_a} vs "
            f"{complete_b} (need at least one complete case per arm)"
        )
    by_id_b = {case["query_id"]: case for case in report_b.get("cases", [])}
    regressions: list[dict] = []
    compared = 0
    for case in report_a.get("cases", []):
        other = by_id_b.get(case.get("query_id"))
        if other is None:
            continue
        compared += 1
        lost: list[str] = []
        if case.get("status") == "complete" and other.get("status") != "complete":
            lost.append("status")
        func_a = (case.get("functional") or {}).get("functional_pass")
        func_b = (other.get("functional") or {}).get("functional_pass")
        if func_a is True and func_b is not True:
            lost.append("functional")
        # Answer-presence quality: losing the answer is a regression even
        # for document-only cases (status alone does not catch it).
        if case.get("has_answer") is True and other.get("has_answer") is not True:
            lost.append("has_answer")
        for key in ("doc_hit", "article_hit", "negative_pass"):
            before = (case.get("functional") or {}).get(key)
            after = (other.get("functional") or {}).get(key)
            if before is True and after is not True:
                lost.append(key)
        if lost:
            regressions.append({"query_id": case.get("query_id"), "lost": lost})
    if compared == 0:
        raise CompareRefused(
            "incomparable reports: no common query_id between arms"
        )
    return {
        "evaluator_version": version_a,
        "arm_a": report_a.get("arm"),
        "arm_b": report_b.get("arm"),
        "compared": compared,
        "complete_a": complete_a,
        "complete_b": complete_b,
        "regressions": regressions,
    }


# ---------------------------------------------------------------------------
# Exit codes: failures are recorded AND surfaced (never silent success)
# ---------------------------------------------------------------------------


def run_exit_code(report: dict, *, want_arm: str) -> int:
    """Non-zero when any case missed terminal-complete, the echoed arm
    differs, zero cases ran, or the session-SSE smoke failed."""
    cases = report.get("cases", [])
    if not cases:
        return 1
    if any(case.get("status") != "complete" for case in cases):
        return 1
    if any(case.get("arm") != want_arm for case in cases):
        return 1
    if not (report.get("session_sse") or {}).get("ok"):
        return 1
    return 0


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
        compose_url(base_url, "/api/v1/auth/login"),
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
    import yaml  # live dep; unit tests never touch this path

    with open(path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    cases = doc.get("cases", doc if isinstance(doc, list) else [])
    queries = []
    for case in cases:
        if isinstance(case, dict) and case.get("query"):
            queries.append(case)
    return queries


def _cmd_run(args, client=None) -> int:
    live = client if client is not None else UrllibHttpClient(args.base_url)
    token = _resolve_token(args)
    queries = _load_queries(args.queries)
    cases: list[dict] = []
    for case in queries:
        query_id = str(case.get("id", case.get("query_id", "query")))
        message = str(case["query"])
        try:
            record = run_eval_turn(
                live,
                base_url=args.base_url,
                token=token,
                arm=args.arm,
                query_id=query_id,
                message=message,
                workspace_id=args.workspace,
                case=case,
            )
            if record.get("arm") != args.arm:
                record = _failed_case(
                    query_id, args.arm,
                    f"arm-mismatch (echoed={record.get('arm')!r}, want={args.arm!r})",
                    secrets=(token, message),
                )
        except Exception as exc:  # noqa: BLE001 — recorded, never success
            record = _failed_case(query_id, args.arm, str(exc),
                                  secrets=(token, message))
        cases.append(record)
    smoke: dict = {}
    if queries:
        smoke = run_session_sse_smoke(
            live, token=token, arm=args.arm,
            message=str(queries[0]["query"]), workspace_id=args.workspace,
        )
    else:
        smoke = {"ok": False, "error": "zero cases ran"}
    report = build_arm_report(arm=args.arm, cases=cases, auth_token=token,
                              session_sse=smoke)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    code = run_exit_code(report, want_arm=args.arm)
    print(
        f"arm={report['arm']} evaluator={report['evaluator_version']} "
        f"cases={len(cases)} complete={report['summary']['complete_count']} "
        f"smoke_ok={smoke.get('ok')} out={args.out}"
    )
    return code


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
    except CompareRefused as exc:
        print(f"INCOMPARABLE: {exc}")
        return 1
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
    run.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help="Server origin WITHOUT /api/v1 suffix "
        f"(default: {DEFAULT_BASE_URL}).",
    )
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

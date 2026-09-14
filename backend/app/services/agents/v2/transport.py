"""Versioned public chat/SSE contract (Phase 4D, Task 9).

Single owner of the v1/v2-normalized frontend presentation contract during
canary. The frontend never consumes raw ``SupervisorV2State``, checkpoint
state, ``TaskPlan``, runtime services, hidden model reasoning, or internal
evidence UUIDs — everything crossing this boundary is projected here.

Contract version: ``v2.chat/1``. Payloads are additive and JSON-serializable.

Minimum public event union (spec Phase 4D §11)::

    status
    clarification_required
    clarification_resolved
    citation            (canonical; the v1 ``sources`` wire event normalizes here)
    token
    complete
    error
    cancelled

Rules:

- v1 wire events (``status``/``sources``/``images``/``token``/
  ``token_rollback``/``thinking``/``potential_abbreviations``/
  ``people_data``/``clarification``/``complete``/``error``) normalize into
  the union above via :func:`normalize_wire_event`. Unknown future event
  types normalize to ``[]`` — they must never crash the stream.
- ``thinking`` (hidden model reasoning) never crosses the public boundary;
  UI phase is conveyed via ``status`` instead. No chain-of-thought is
  persisted or projected.
- Clarification is structured: options carry server-issued ``option_id``
  values plus resume metadata. The frontend may submit ONLY a server-issued
  ``option_id`` (:func:`validate_clarification_selection`); raw document
  UUIDs / workspace / binding identities are rejected so the client can
  never fabricate trusted identity.
- Citations are allowlist-projected (:data:`_PUBLIC_CITATION_FIELDS`);
  internal evidence/store identifiers (``evidence_id``, ``task_id``,
  ``run_id``, ``workspace_id``, …) are stripped, never forwarded.
- :func:`feed_sse_chunks` parses real ``event:`` + ``data:`` framing over
  fragmented TCP chunks; data-only frames and malformed JSON are skipped,
  never fatal. :func:`format_public_sse` emits canonical framing.

v1 remains the rollback arm: the v1 wire format itself is frozen — this
module only *projects* it. Only this transport owner and the frontend hook
may evolve the presentation contract.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Final

__all__ = [
    "PUBLIC_CHAT_CONTRACT_VERSION",
    "PUBLIC_EVENT_TYPES",
    "PUBLIC_UI_PHASES",
    "ClarificationSelectionError",
    "build_clarification_required",
    "build_clarification_resolved",
    "validate_clarification_selection",
    "normalize_wire_event",
    "normalize_v2_citations",
    "feed_sse_chunks",
    "format_public_sse",
]

PUBLIC_CHAT_CONTRACT_VERSION: Final[str] = "v2.chat/1"

PUBLIC_EVENT_TYPES: Final[tuple[str, ...]] = (
    "status",
    "clarification_required",
    "clarification_resolved",
    "citation",
    "token",
    "complete",
    "error",
    "cancelled",
)

PUBLIC_UI_PHASES: Final[tuple[str, ...]] = (
    "clarifying",
    "planning",
    "executing",
    "evaluating",
    "generating",
)

# v1 status steps -> public UI phase (no chain-of-thought leaves the backend).
_STATUS_STEP_TO_PHASE: Final[dict[str, str]] = {
    "starting": "planning",
    "analyzing": "planning",
    "understood": "planning",
    "searching": "executing",
    "retrieving": "executing",
    "retrieved": "evaluating",
    "generating": "generating",
    "rollback": "executing",
    "abbreviations": "evaluating",
    "done": "generating",
}

# Session-plumbing wire events that are server-issued message identity, not
# v2 state. Passed through untouched (still version-stamped).
_PASSTHROUGH_EVENTS: Final[frozenset[str]] = frozenset(
    {"ai_message_id", "user_id", "session_title_updated"}
)

# Allowlisted public citation fields. Everything else — internal evidence
# UUIDs, task/run/binding/workspace identity, scores internals — is dropped.
_PUBLIC_CITATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "citation_id",
        "label",
        "document_id",
        "chunk_id",
        "content",
        "page_no",
        "heading_path",
        "source_file",
        "document_number",
        "article_label",
        "validity_status",
        "superseded_by",
    }
)

# Keys that must never cross the public boundary even if present upstream.
_INTERNAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "evidence_id",
        "evidence_ids",
        "task_id",
        "task_ids",
        "run_id",
        "workspace_id",
        "workspace_ids",
        "binding_id",
        "revision_id",
        "use_id",
        "plan_id",
        "checkpoint_id",
    }
)

_PEOPLE_INTERNAL_KEYS: Final[frozenset[str]] = (
    frozenset({"_id", "_embedding", "embedding"}) | _INTERNAL_KEYS
)


class ClarificationSelectionError(ValueError):
    """A clarification reply that is not a server-issued option id."""


def _stable_citation_id(document_id: str, chunk_id: str) -> str:
    digest = hashlib.sha256(f"{document_id}::{chunk_id}".encode("utf-8")).hexdigest()
    return f"cit-{digest[:12]}"


def _citation_label(item: dict) -> str:
    article = item.get("article_label")
    number = item.get("document_number")
    if article and number:
        return f"{article} — {number}"
    if article:
        return str(article)
    if number:
        return str(number)
    source_file = item.get("source_file")
    if source_file:
        return str(source_file)
    return f"Trích dẫn {item.get('chunk_id', '')}".strip()


def normalize_v2_citations(sources: list[dict] | None) -> list[dict]:
    """Project raw source dicts onto the stable public citation type.

    Allowlist-projection only: internal evidence/store identifiers are
    stripped. ``citation_id`` is a stable server-derived handle (never an
    internal evidence UUID); ``document_id``/``chunk_id`` locate the public
    chunk for reload rebuilds without exposing trust/binding identity.
    """
    out: list[dict] = []
    for item in sources or []:
        if not isinstance(item, dict):
            continue
        document_id = str(item.get("document_id") or "")
        chunk_id = str(item.get("chunk_id") or "")
        citation: dict[str, Any] = {
            "citation_id": item.get("citation_id")
            or _stable_citation_id(document_id, chunk_id),
            "label": item.get("label") or _citation_label(item),
        }
        for key in _PUBLIC_CITATION_FIELDS - {"citation_id", "label"}:
            if key in item and key not in _INTERNAL_KEYS:
                citation[key] = item[key]
        if "document_id" not in citation:
            citation["document_id"] = document_id
        if "chunk_id" not in citation:
            citation["chunk_id"] = chunk_id
        out.append(citation)
    return out


def _public_image_refs(image_refs: list) -> list[dict]:
    allowed = {
        "ref_id",
        "image_id",
        "document_id",
        "page_no",
        "caption",
        "url",
        "width",
        "height",
    }
    out = []
    for item in image_refs or []:
        if not isinstance(item, dict):
            continue
        out.append(
            {k: item[k] for k in allowed if k in item and k not in _INTERNAL_KEYS}
        )
    return out


def _public_people(records: list) -> list[dict]:
    out = []
    for item in records or []:
        if not isinstance(item, dict):
            continue
        projected = {
            k: v
            for k, v in item.items()
            if k == "_source_schema" or not k.startswith("_")
        }
        for key in _PEOPLE_INTERNAL_KEYS:
            projected.pop(key, None)
        out.append(projected)
    return out


def _versioned(event: str, data: dict) -> dict:
    payload = {"contract_version": PUBLIC_CHAT_CONTRACT_VERSION}
    payload.update(data)
    return {"event": event, "data": payload}


def build_clarification_required(
    *,
    clarification_id: str,
    question: str,
    options: list[dict],
    resume: dict,
    reason: str = "semantic_ambiguity",
) -> dict:
    """Build a structured public ``clarification_required`` event.

    ``options`` entries carry server-issued ``option_id`` values
    (``{"option_id", "label", "document_id"?}``); ``resume`` carries
    ``{"thread_id", "message_id"}`` so the reply resumes the suspended turn
    instead of minting identity client-side.
    """
    public_options = []
    for index, option in enumerate(options or []):
        if not isinstance(option, dict):
            continue
        option_id = str(option.get("option_id") or f"opt-{index + 1}")
        entry: dict[str, Any] = {
            "option_id": option_id,
            "label": str(option.get("label") or option_id),
        }
        if option.get("document_id") is not None:
            entry["document_id"] = str(option["document_id"])
        public_options.append(entry)
    return _versioned(
        "clarification_required",
        {
            "clarification_id": clarification_id,
            "reason": reason,
            "question": question,
            "options": public_options,
            "resume": {
                "thread_id": str(resume.get("thread_id") or ""),
                "message_id": str(resume.get("message_id") or ""),
            },
        },
    )


def build_clarification_resolved(
    *, clarification_id: str, selected_option_id: str
) -> dict:
    """Build the public ``clarification_resolved`` acknowledgement."""
    return _versioned(
        "clarification_resolved",
        {
            "clarification_id": clarification_id,
            "selected_option_id": selected_option_id,
        },
    )


def validate_clarification_selection(request_data: dict, value: str) -> str:
    """Accept ONLY a server-issued ``option_id`` from this request.

    Raw document UUIDs, workspace ids, or any value not issued in
    ``request_data["options"]`` raise :class:`ClarificationSelectionError`
    — the frontend must never fabricate document/workspace/binding
    identity.
    """
    options = request_data.get("options") or []
    issued = {
        str(o.get("option_id"))
        for o in options
        if isinstance(o, dict) and o.get("option_id")
    }
    if str(value) in issued:
        return str(value)
    raise ClarificationSelectionError(
        f"clarification selection {value!r} was not issued by the server "
        f"(clarification_id={request_data.get('clarification_id')!r})"
    )


def _normalize_v1_clarification(data: dict) -> list[dict]:
    """Legacy v1 ``clarification`` {message, options: [str], context}."""
    raw_options = data.get("options") or []
    options = []
    for index, label in enumerate(raw_options):
        if isinstance(label, dict):
            option_id = str(label.get("option_id") or f"opt-{index + 1}")
            options.append(
                {
                    "option_id": option_id,
                    "label": str(label.get("label") or option_id),
                    **(
                        {"document_id": str(label["document_id"])}
                        if label.get("document_id")
                        else {}
                    ),
                }
            )
        else:
            options.append(
                {"option_id": f"opt-{index + 1}", "label": str(label)}
            )
    context = data.get("context") or {}
    resume: dict[str, str] = {"thread_id": "", "message_id": ""}
    if isinstance(context, dict):
        for key in ("thread_id", "message_id"):
            if context.get(key):
                resume[key] = str(context[key])
    return [
        build_clarification_required(
            clarification_id=str(
                data.get("clarification_id")
                or (context.get("clarification_id") if isinstance(context, dict) else None)
                or "clr-legacy"
            ),
            question=str(data.get("message") or data.get("question") or ""),
            options=options,
            resume=resume,
            reason=str(data.get("reason") or "semantic_ambiguity"),
        )
    ]


def normalize_wire_event(event: str, data: dict) -> list[dict]:
    """Normalize one v1/v2 wire ``(event, data)`` pair onto public events.

    Returns a (possibly empty) list of version-stamped public events.
    Unknown future event types return ``[]`` — forward-compatible, never
    fatal. ``data`` is never mutated.
    """
    payload = dict(data or {})
    if event in _PASSTHROUGH_EVENTS:
        return [_versioned(event, payload)]
    if event == "status":
        step = str(payload.get("step") or "")
        phase = _STATUS_STEP_TO_PHASE.get(step, "executing")
        out: dict[str, Any] = {"step": step, "phase": phase}
        if payload.get("detail") is not None:
            out["detail"] = str(payload.get("detail") or "")
        if payload.get("abbreviations") is not None:
            out["abbreviations"] = list(payload.get("abbreviations") or [])
        return [_versioned("status", out)]
    if event == "token":
        return [_versioned("token", {"text": str(payload.get("text") or "")})]
    if event == "token_rollback":
        # Speculative draft discarded: surfaced as a public status step so
        # the UI clears speculative content without a private vocabulary.
        return [
            _versioned(
                "status",
                {"step": "rollback", "phase": "executing", "detail": ""},
            )
        ]
    if event == "sources":
        return [
            _versioned(
                "citation",
                {
                    "citations": normalize_v2_citations(payload.get("sources")),
                    "image_refs": [],
                    "people": [],
                },
            )
        ]
    if event == "images":
        refs = payload.get("image_refs", payload.get("images", []))
        return [
            _versioned(
                "citation",
                {
                    "citations": [],
                    "image_refs": _public_image_refs(refs or []),
                    "people": [],
                },
            )
        ]
    if event == "people_data":
        return [
            _versioned(
                "citation",
                {
                    "citations": [],
                    "image_refs": [],
                    "people": _public_people(payload.get("people") or []),
                },
            )
        ]
    if event == "potential_abbreviations":
        return [
            _versioned(
                "status",
                {
                    "step": "abbreviations",
                    "phase": "evaluating",
                    "abbreviations": list(payload.get("abbreviations") or []),
                },
            )
        ]
    if event == "thinking":
        # Hidden model reasoning never crosses the public boundary.
        return []
    if event == "clarification":
        return _normalize_v1_clarification(payload)
    if event == "clarification_required":
        return [_versioned("clarification_required", payload)]
    if event == "clarification_resolved":
        return [_versioned("clarification_resolved", payload)]
    if event == "citation":
        return [
            _versioned(
                "citation",
                {
                    "citations": normalize_v2_citations(payload.get("citations")),
                    "image_refs": _public_image_refs(
                        payload.get("image_refs") or []
                    ),
                    "people": _public_people(payload.get("people") or []),
                },
            )
        ]
    if event == "complete":
        sources = payload.get("sources")
        citations = payload.get("citations")
        out_complete: dict[str, Any] = {
            "answer": str(payload.get("answer") or ""),
            "citations": (
                normalize_v2_citations(citations)
                if isinstance(citations, list)
                else normalize_v2_citations(sources if isinstance(sources, list) else [])
            ),
            "image_refs": _public_image_refs(
                payload.get("image_refs", payload.get("images", [])) or []
            ),
            "people": _public_people(payload.get("people_data") or []),
            "potential_abbreviations": list(
                payload.get("potential_abbreviations") or []
            ),
        }
        if isinstance(payload.get("status"), str):
            out_complete["status"] = payload["status"]
        if isinstance(payload.get("related_entities"), list):
            out_complete["related_entities"] = list(payload["related_entities"])
        if payload.get("clarification") is not None:
            out_complete["clarification"] = payload["clarification"]
        return [_versioned("complete", out_complete)]
    if event == "error":
        return [_versioned("error", {"message": str(payload.get("message") or "")})]
    if event == "cancelled":
        return [
            _versioned(
                "cancelled",
                {"reason": str(payload.get("reason") or "cancelled")},
            )
        ]
    # Forward-compatible: unknown future event types never crash the stream.
    return []


def format_public_sse(event: str, data: dict) -> str:
    """Format one public event with canonical ``event:`` + ``data:`` framing."""
    json_data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {json_data}\n\n"


def normalize_sse_frame(frame: str) -> list[str]:
    """Normalize one raw wire SSE frame for the session relay (C1 funnel).

    Parses the frame's ``event:``/``data:`` lines, projects the payload via
    :func:`normalize_wire_event`, and returns canonical public frame(s).
    Frames the normalizer maps to ``[]`` (advisory ``thinking``, unknown
    future types) and unparseable/non-event frames (heartbeats, comments,
    data-only) are returned **untouched** — the relay never drops a byte
    it does not understand, and advisory model output is preserved.
    """
    event_name: str | None = None
    data_lines: list[str] = []
    for line in (frame or "").split("\n"):
        if line.startswith("event:"):
            if event_name is None:
                event_name = line[len("event:"):].strip() or None
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip(" "))
    if event_name is None or not data_lines:
        return [frame]
    try:
        payload = json.loads("\n".join(data_lines))
    except (json.JSONDecodeError, ValueError):
        return [frame]
    if not isinstance(payload, dict):
        return [frame]
    normalized = normalize_wire_event(event_name, payload)
    if not normalized:
        return [frame]
    return [
        format_public_sse(item["event"], item["data"]) for item in normalized
    ]


def feed_sse_chunks(chunks: list[str]) -> list[dict]:
    """Parse fragmented SSE byte-chunks into ``[{event, data}]``.

    Handles real ``event:`` + ``data:`` framing split across arbitrary chunk
    boundaries, multi-line ``data:`` (joined with ``\\n``), ``:`` heartbeat
    comments, data-only frames (ignored), and malformed JSON (skipped).
    """
    buffer = "".join(chunks)
    events: list[dict] = []
    # Split on blank-line frame boundaries; keep any trailing partial frame
    # buffered (dropped here — the caller streams complete turns; a partial
    # tail is never fabricated into an event).
    frames = buffer.split("\n\n")
    for frame in frames:
        current_event: str | None = None
        data_lines: list[str] = []
        for line in frame.split("\n"):
            if not line:
                continue
            if line.startswith(":"):
                continue  # heartbeat/comment
            if line.startswith("event:"):
                current_event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip(" "))
            # Lines without a field prefix (e.g. a fragment split mid-line
            # that a blank boundary cut) are ignored, never fatal.
        if current_event is None or not data_lines:
            continue
        raw = "\n".join(data_lines)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        events.append({"event": current_event, "data": parsed})
    return events

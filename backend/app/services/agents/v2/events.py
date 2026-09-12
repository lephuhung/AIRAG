"""SSE/event helpers for the v2 outer adapter (Phase 2, Task 6).

Pure presentation helpers consumed by the T8 streaming/runner layer; this
module owns no graph, no checkpointer, and no domain business. The SSE wire
format is identical to v1 (``event: {event}\\ndata: {json}\\n\\n``) and every
v1 ``complete``/``error`` key is preserved (``answer``, ``sources``,
``images``, ``potential_abbreviations``, ``people_data``, ``message``), so
the existing frontend hook keeps working without changes. Payloads are
ADDITIVE, not identical: ``complete`` additionally carries ``status`` and a
``citations`` projection (``citation_id`` + ``label`` only — no evidence
internals). T8's contract test must assert additive compatibility (v1 keys
present), not byte identity.

- ``complete`` carries ``answer``/``sources``/``images``/
  ``potential_abbreviations``/``people_data`` (all default to empty);
- ``error`` carries ``message``;
- ``token_rollback`` carries ``{}``.

Terminal mapping: ``success`` and ``clarify`` are ``complete`` events (the
clarification question is the terminal answer of a clarify turn);
``denied``/``insufficient``/``error`` are ``error`` events whose message is
the user-safe response content (internal diagnostics never leave the node
layer, so ``content`` is already safe to surface).
"""
from __future__ import annotations

import json
from typing import Any

from .contracts.response import FinalResponse

__all__ = [
    "SSE_STATUS",
    "SSE_THINKING",
    "SSE_SOURCES",
    "SSE_IMAGES",
    "SSE_TOKEN",
    "SSE_TOKEN_ROLLBACK",
    "SSE_POTENTIAL_ABBREVIATIONS",
    "SSE_PEOPLE_DATA",
    "SSE_ERROR",
    "SSE_COMPLETE",
    "format_sse_event",
    "final_response_payload",
    "terminal_event_for_response",
]

SSE_STATUS = "status"
SSE_THINKING = "thinking"
SSE_SOURCES = "sources"
SSE_IMAGES = "images"
SSE_TOKEN = "token"
SSE_TOKEN_ROLLBACK = "token_rollback"
SSE_POTENTIAL_ABBREVIATIONS = "potential_abbreviations"
SSE_PEOPLE_DATA = "people_data"
SSE_ERROR = "error"
SSE_COMPLETE = "complete"


def _json_default(value: Any) -> Any:
    """Best-effort JSON default matching the v1 ``json_serial`` contract."""
    try:
        from app.services.agent.streaming import json_serial

        return json_serial(value)
    except Exception:  # noqa: BLE001 - v1 streaming must never break v2 events
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return str(value)


def format_sse_event(event: str, data: dict) -> str:
    """Format one SSE event with the exact v1 wire format."""
    return f"event: {event}\ndata: {json.dumps(data, default=_json_default, ensure_ascii=False)}\n\n"


def final_response_payload(
    response: FinalResponse,
    *,
    sources: list[dict] | None = None,
    images: list[dict] | None = None,
    potential_abbreviations: list[dict] | None = None,
    people_data: list[dict] | None = None,
) -> dict:
    """Render a ``FinalResponse`` into the v1 ``complete`` payload shape.

    Citations are presentation output (``citation_id`` + ``label``); evidence
    identity internals stay out of the payload.
    """
    return {
        "status": response.status,
        "answer": response.content,
        "sources": sources if sources is not None else [],
        "images": images if images is not None else [],
        "potential_abbreviations": (
            potential_abbreviations if potential_abbreviations is not None else []
        ),
        "people_data": people_data if people_data is not None else [],
        "citations": [
            {"citation_id": citation.citation_id, "label": citation.label}
            for citation in response.citations
        ],
    }


def terminal_event_for_response(response: FinalResponse) -> tuple[str, dict]:
    """Map a terminal ``FinalResponse`` onto one ``(event, data)`` pair."""
    if response.status in ("success", "clarify"):
        return (SSE_COMPLETE, final_response_payload(response))
    return (SSE_ERROR, {"message": response.content})

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

# Phase 4D (Task 9): the versioned public chat/SSE contract owner. This
# adapter only *projects* onto it — the v1 wire format stays frozen as the
# rollback arm. Import is module-local-safe: ``transport`` depends on no
# app modules (no cycle risk with the v2 package).
from .transport import PUBLIC_CHAT_CONTRACT_VERSION as PUBLIC_CHAT_CONTRACT_VERSION
from .transport import _INTERNAL_KEYS, _PUBLIC_CITATION_FIELDS
from .transport import normalize_wire_event as _normalize_wire_event

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
    "PUBLIC_CHAT_CONTRACT_VERSION",
    "clarification_public_metadata",
    "to_public_event",
    "format_sse_event",
    "final_response_payload",
    "terminal_event_for_response",
    "SYNTHESIS_SPECULATIVE_CLAIM",
    "SYNTHESIS_SPECULATIVE_RESET",
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

#: Advisory custom-stream payloads written by ``generate_node`` (never
#: checkpointed, never traced): the outer streaming adapter owns their
#: presentation as speculative ``token`` events, always retracted by
#: ``token_rollback`` before a repair attempt or any terminal.
SYNTHESIS_SPECULATIVE_CLAIM = "synthesis.speculative_claim"
SYNTHESIS_SPECULATIVE_RESET = "synthesis.speculative_reset"


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


def _citation_public_dict(citation: Any) -> dict:
    """Project one checkpointed citation onto the §11.3 wire allowlist.

    ``PublicCitation`` (grounded-LLM synthesis) carries the full projector
    output; legacy ``RenderedCitation`` contributes ``citation_id``/``label``
    only. Internal identities (``evidence_id``, ``use_id``, …) are stripped
    here AND again by ``normalize_v2_citations`` — they never cross.
    """
    if isinstance(citation, dict):
        raw = citation
    elif hasattr(citation, "model_dump"):
        raw = citation.model_dump(mode="json", exclude_none=True)
    else:
        raw = {
            "citation_id": getattr(citation, "citation_id", None),
            "label": getattr(citation, "label", None),
        }
    return {
        key: value
        for key, value in raw.items()
        if key in _PUBLIC_CITATION_FIELDS
        and key not in _INTERNAL_KEYS
        and value is not None
    }


def final_response_payload(
    response: FinalResponse,
    *,
    sources: list[dict] | None = None,
    images: list[dict] | None = None,
    potential_abbreviations: list[dict] | None = None,
    people_data: list[dict] | None = None,
    clarification: dict | None = None,
) -> dict:
    """Render a ``FinalResponse`` into the v1 ``complete`` payload shape.

    Citations carry the full §11.3 allowlisted projection (``index``,
    ``document_id``, ``chunk_id``, locator/validity metadata) so the
    ``citation`` frame and ``complete`` repeat the identical public set;
    evidence identity internals stay out of the payload. ``clarification``
    (Task 9) carries the public clarify-turn resume block built by
    :func:`clarification_public_metadata` — persisted so a reload can
    rebuild the resume instead of minting identity client-side.
    """
    payload = {
        "status": response.status,
        "answer": response.content,
        "sources": sources if sources is not None else [],
        "images": images if images is not None else [],
        "potential_abbreviations": (
            potential_abbreviations if potential_abbreviations is not None else []
        ),
        "people_data": people_data if people_data is not None else [],
        "citations": [
            _citation_public_dict(citation) for citation in response.citations
        ],
    }
    if clarification is not None:
        payload["clarification"] = clarification
    return payload


def clarification_public_metadata(request, *, thread_id: str) -> dict:
    """Project a checkpointed ``ClarificationRequest`` onto public metadata.

    Options expose server-issued ``option_id`` (the stable candidate id)
    plus display ``label`` only — never the internal document UUID, so the
    frontend can submit the selection without fabricating trusted identity.
    ``resume`` carries the stable thread plus the request expiry; the
    entrypoint re-resolves the persisted request from the checkpoint.
    """
    try:
        candidates = request.candidates if not isinstance(request, dict) else request.get("candidates", ())
    except Exception:
        candidates = ()
    options = []
    for candidate in candidates or ():
        if isinstance(candidate, dict):
            option_id = str(candidate.get("candidate_id") or "")
            label = str(candidate.get("label") or option_id)
        else:
            option_id = str(getattr(candidate, "candidate_id", "") or "")
            label = str(getattr(candidate, "label", None) or option_id)
        if not option_id:
            continue
        options.append({"option_id": option_id, "label": label})
    try:
        clarification_id = request.get("clarification_id") if isinstance(request, dict) else getattr(request, "clarification_id", "")
    except Exception:
        clarification_id = ""
    try:
        expires_at = request.get("expires_at") if isinstance(request, dict) else getattr(request, "expires_at", None)
        expires = expires_at.isoformat() if hasattr(expires_at, "isoformat") else (str(expires_at) if expires_at else None)
    except Exception:
        expires = None
    metadata: dict = {
        "clarification_id": str(clarification_id or ""),
        "options": options,
        "resume": {"thread_id": str(thread_id or "")},
    }
    if expires is not None:
        metadata["resume"]["expires_at"] = expires
    return metadata


def to_public_event(event: str, data: dict) -> list[dict]:
    """Normalize one v1/v2 wire event onto the versioned public contract.

    Thin adapter over :mod:`app.services.agents.v2.transport` — the single
    funnel keeping the hook's presentation model identical across arms.
    """
    return _normalize_wire_event(event, data)


def terminal_event_for_response(response: FinalResponse) -> tuple[str, dict]:
    """Map a terminal ``FinalResponse`` onto one ``(event, data)`` pair."""
    if response.status in ("success", "clarify"):
        return (SSE_COMPLETE, final_response_payload(response))
    return (SSE_ERROR, {"message": response.content})

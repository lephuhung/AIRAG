"""Task 9 RED: versioned public chat/SSE contract.

The public transport DTO union (``backend/app/services/agents/v2/transport.py``)
is the single owner of the v1/v2-normalized frontend presentation contract.
Frontend never consumes raw SupervisorV2State, checkpoint state, TaskPlan,
runtime services, hidden reasoning, or internal evidence UUIDs.
"""
from __future__ import annotations


def test_public_contract_version_is_pinned():
    from app.services.agents.v2 import transport

    assert transport.PUBLIC_CHAT_CONTRACT_VERSION == "v2.chat/1"


def test_public_event_union_covers_minimum_set():
    from app.services.agents.v2 import transport

    assert set(transport.PUBLIC_EVENT_TYPES) >= {
        "status",
        "clarification_required",
        "clarification_resolved",
        "citation",
        "token",
        "complete",
        "error",
        "cancelled",
    }


def test_normalize_v1_sources_become_public_citations_without_internal_ids():
    from app.services.agents.v2 import transport

    v1_data = {
        "sources": [
            {
                "index": "a1b2",
                "chunk_id": "chunk_3",
                "content": "excerpt",
                "document_id": "11111111-1111-1111-1111-111111111111",
                "page_no": 2,
                "heading_path": ["chuong 1"],
                "score": 0.9,
                # Internal-only junk the wire must never forward.
                "evidence_id": "99999999-9999-9999-9999-999999999999",
                "task_id": "task--internal-1",
            }
        ]
    }
    events = transport.normalize_wire_event("sources", v1_data)
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "citation"
    payload = ev["data"]
    assert payload["contract_version"] == "v2.chat/1"
    assert payload["citations"][0]["document_id"] == "11111111-1111-1111-1111-111111111111"
    dumped = str(payload)
    assert "99999999-9999-9999-9999-999999999999" not in dumped
    assert "task-internal-1" not in dumped


def test_clarification_carries_server_issued_option_ids_and_resume_metadata():
    from app.services.agents.v2 import transport

    req = transport.build_clarification_required(
        clarification_id="clr-1",
        question="Which document do you mean?",
        options=[
            {"option_id": "opt-1", "label": "Doc A", "document_id": "11111111-1111-1111-1111-111111111111"},
            {"option_id": "opt-2", "label": "Doc B", "document_id": "22222222-2222-2222-2222-222222222222"},
        ],
        resume={"thread_id": "thread-1", "message_id": "msg-1"},
    )
    assert req["event"] == "clarification_required"
    assert [o["option_id"] for o in req["data"]["options"]] == ["opt-1", "opt-2"]
    assert req["data"]["resume"]["thread_id"] == "thread-1"
    # Frontend may submit only server-issued option ids.
    assert transport.validate_clarification_selection(req["data"], "opt-2") == "opt-2"


def test_clarification_selection_rejects_fabricated_identity():
    from app.services.agents.v2 import transport

    req = transport.build_clarification_required(
        clarification_id="clr-1",
        question="Which document do you mean?",
        options=[
            {"option_id": "opt-1", "label": "Doc A", "document_id": "11111111-1111-1111-1111-111111111111"},
        ],
        resume={"thread_id": "thread-1", "message_id": "msg-1"},
    )
    import pytest

    with pytest.raises(transport.ClarificationSelectionError):
        transport.validate_clarification_selection(
            req["data"], "33333333-3333-3333-3333-333333333333"
        )


def test_unknown_forward_compatible_event_does_not_crash():
    from app.services.agents.v2 import transport

    events = transport.normalize_wire_event("future_shiny", {"anything": 1})
    assert events == []


def test_fragmented_sse_frames_with_event_and_data_framing():
    from app.services.agents.v2 import transport

    chunks = [
        'event: cita',
        'tion\ndata: {"cita',
        'tions": []}\n\n',
        'event: token\ndata: {"text": "hel',
        'lo"}\n\n',
        # Data-only frame without an event line is ignored, never fatal.
        'data: {"text": "orphan"}\n\n',
    ]
    events = transport.feed_sse_chunks(chunks)
    assert [e["event"] for e in events] == ["citation", "token"]
    assert events[1]["data"] == {"text": "hello"}


def test_public_sse_format_uses_event_plus_data_framing():
    from app.services.agents.v2 import transport

    frame = transport.format_public_sse("token", {"text": "hi"})
    assert frame.startswith("event: token\ndata: ")
    assert frame.endswith("\n\n")


def test_events_adapter_projects_onto_public_contract():
    from app.services.agents.v2 import events

    assert events.PUBLIC_CHAT_CONTRACT_VERSION == "v2.chat/1"
    normalized = events.to_public_event(
        "sources", {"sources": [{"document_id": "d1", "chunk_id": "c1"}]}
    )
    assert normalized[0]["event"] == "citation"
    assert normalized[0]["data"]["contract_version"] == "v2.chat/1"
    # Unknown wire events stay forward-compatible.
    assert events.to_public_event("future_shiny", {}) == []


def test_events_adapter_projects_clarification_resume_metadata():
    from app.services.agents.v2 import events
    from app.services.agents.v2.contracts.clarification import (
        ClarificationRequest,
        DocumentCandidate,
    )

    from datetime import datetime
    from uuid import UUID

    request = ClarificationRequest.model_validate(
        {
            "contract_version": "2.0",
            "clarification_id": "clr-9",
            "reason": "required_document_ambiguous",
            "question": "Which one?",
            "unresolved_ref_ids": ("ref-a",),
            "candidates": (
                {
                    "candidate_id": "cand-1",
                    "ordinal": 1,
                    "ref_id": "ref-a",
                    "document_id": UUID("11111111-1111-1111-1111-111111111111"),
                    "label": "Doc A",
                },
            ),
            "expires_at": datetime(2030, 1, 1),
        }
    )
    metadata = events.clarification_public_metadata(request, thread_id="thread-9")
    assert metadata["options"] == [{"option_id": "cand-1", "label": "Doc A"}]
    assert metadata["resume"]["thread_id"] == "thread-9"
    # Internal document UUID never crosses the public boundary.
    assert "11111111-1111-1111-1111-111111111111" not in str(metadata)


def test_final_response_payload_carries_clarification_block():
    from app.services.agents.v2 import events
    from app.services.agents.v2.contracts.response import FinalResponse

    response = FinalResponse.model_validate(
        {"contract_version": "2.0", "status": "clarify", "content": "Which one?"}
    )
    payload = events.final_response_payload(
        response,
        clarification={"clarification_id": "clr-9", "options": [], "resume": {}},
    )
    assert payload["clarification"]["clarification_id"] == "clr-9"
    # Existing callers without the kwarg are unaffected.
    assert "clarification" not in events.final_response_payload(response)

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


def _relay_frames(raw_frame: str) -> list[str]:
    """Drive one raw v1-wire SSE frame through the session relay funnel."""
    from app.services.agents.v2 import transport

    return transport.normalize_sse_frame(raw_frame)


def test_relay_normalizes_v1_sources_into_stamped_citation():
    import json

    raw = (
        'event: sources\ndata: {"sources": [{"document_id": "d1", '
        '"chunk_id": "c1", "content": "excerpt", '
        '"evidence_id": "99999999-9999-9999-9999-999999999999"}]}\n\n'
    )
    frames = _relay_frames(raw)
    assert len(frames) == 1
    assert frames[0].startswith("event: citation\ndata: ")
    payload = json.loads(frames[0].split("\ndata: ", 1)[1])
    assert payload["contract_version"] == "v2.chat/1"
    assert payload["citations"][0]["document_id"] == "d1"
    assert "99999999-9999-9999-9999-999999999999" not in frames[0]


def test_relay_stamps_status_with_phase_and_keeps_token_wire():
    import json

    status_frames = _relay_frames(
        'event: status\ndata: {"step": "analyzing", "detail": "x"}\n\n'
    )
    assert len(status_frames) == 1
    payload = json.loads(status_frames[0].split("\ndata: ", 1)[1])
    assert payload["phase"] == "planning"
    assert payload["contract_version"] == "v2.chat/1"

    token_frames = _relay_frames('event: token\ndata: {"text": "hi"}\n\n')
    assert token_frames == [
        'event: token\ndata: {"contract_version":"v2.chat/1","text":"hi"}\n\n'
    ]


def test_relay_passes_thinking_and_unknown_through_untouched():
    raw_thinking = 'event: thinking\ndata: {"text": "advisory"}\n\n'
    assert _relay_frames(raw_thinking) == [raw_thinking]
    raw_future = 'event: future_shiny\ndata: {"a": 1}\n\n'
    assert _relay_frames(raw_future) == [raw_future]


def test_relay_maps_rollback_and_complete_with_resume_block():
    import json

    rollback = _relay_frames('event: token_rollback\ndata: {}\n\n')
    assert rollback[0].startswith("event: status\n")
    assert json.loads(rollback[0].split("\ndata: ", 1)[1])["step"] == "rollback"

    complete = _relay_frames(
        'event: complete\ndata: {"answer": "Which?", "status": "clarify", '
        '"citations": [], "clarification": {"clarification_id": "clr-1", '
        '"options": [], "resume": {"thread_id": "t"}}}\n\n'
    )
    payload = json.loads(complete[0].split("\ndata: ", 1)[1])
    assert payload["clarification"]["clarification_id"] == "clr-1"
    assert payload["contract_version"] == "v2.chat/1"


def _selection_request():
    from datetime import datetime, timezone
    from uuid import UUID

    from app.services.agents.v2.contracts.clarification import (
        ClarificationRequest,
        DocumentCandidate,
    )

    return ClarificationRequest(
        contract_version="2.0",
        clarification_id="clr-9",
        reason="required_document_ambiguous",
        question="Which one?",
        unresolved_ref_ids=("ref-a",),
        candidates=(
            DocumentCandidate(
                candidate_id="cand-1",
                ordinal=1,
                ref_id="ref-a",
                document_id=UUID("11111111-1111-1111-1111-111111111111"),
                label="Doc A",
            ),
        ),
        expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )


def test_prepare_resume_rejects_fabricated_selection():
    """Task 9 fix round 1 (I3): a selection the server never issued raises
    ``ClarificationError`` before any resume dispatch (caller falls back to
    a fresh turn — never a resume). No DB/runtime touched on this path.
    """
    import asyncio

    import pytest

    from app.services.agent.streaming import prepare_v2_resume_command
    from app.services.agents.v2.nodes.clarification import ClarificationError

    with pytest.raises(ClarificationError):
        asyncio.run(
            prepare_v2_resume_command(
                message_id="msg-1",
                request=_selection_request(),
                runtime_context=object(),
                selected_option_id="33333333-3333-3333-3333-333333333333",
                clarification_id="clr-9",
            )
        )


def test_prepare_resume_rejects_wrong_clarification_id():
    import asyncio

    import pytest

    from app.services.agent.streaming import prepare_v2_resume_command
    from app.services.agents.v2.nodes.clarification import ClarificationError

    with pytest.raises(ClarificationError):
        asyncio.run(
            prepare_v2_resume_command(
                message_id="msg-1",
                request=_selection_request(),
                runtime_context=object(),
                selected_option_id="cand-1",
                clarification_id="clr-other",
            )
        )


def test_relay_preserves_citation_handle_and_source_type():
    """Fix round 2 (N-C1): the answer cites sources by the 4-char ``index``
    code and the UI resolves markers by exact ``index`` match — the public
    boundary must preserve ``index``/``source_type``/``score`` instead of
    coercing every source to an unlocatable vector citation.
    """
    import json

    raw = (
        'event: sources\ndata: {"sources": ['
        '{"index": "a3x9", "chunk_id": "c1", "content": "excerpt", '
        '"document_id": "d1", "page_no": 2, "score": 0.91, '
        '"source_type": "vector"}, '
        '{"index": "k7q2", "chunk_id": "k1", "content": "kg excerpt", '
        '"document_id": "d2", "page_no": 0, "score": 0.77, '
        '"source_type": "kg"}'
        ']}\n\n'
    )
    frames = transport_normalize(raw)
    assert len(frames) == 1
    payload = json.loads(frames[0].split("\ndata: ", 1)[1])
    by_index = {c.get("index"): c for c in payload["citations"]}
    assert set(by_index) == {"a3x9", "k7q2"}
    assert by_index["a3x9"]["source_type"] == "vector"
    assert by_index["a3x9"]["score"] == 0.91
    assert by_index["k7q2"]["source_type"] == "kg"
    assert by_index["k7q2"]["score"] == 0.77


def transport_normalize(raw: str) -> list[str]:
    from app.services.agents.v2 import transport

    return transport.normalize_sse_frame(raw)


def test_relay_stamps_outside_funnel_frames():
    """Fix round 2 (bullet-2 residual): session-plumbing frames emitted
    outside the drain funnel must still reach the client stamped.
    """
    import json

    for raw in (
        'event: status\ndata: {"step": "starting", "detail": "hi"}\n\n',
        'event: user_id\ndata: {"id": "msg_1"}\n\n',
        'event: ai_message_id\ndata: {"message_id": "msg_2"}\n\n',
        'event: session_title_updated\ndata: {"Title": "T"}\n\n',
        'event: error\ndata: {"message": "boom"}\n\n',
    ):
        frames = transport_normalize(raw)
        assert len(frames) == 1, raw
        payload = json.loads(frames[0].split("\ndata: ", 1)[1])
        assert payload.get("contract_version") == "v2.chat/1", raw

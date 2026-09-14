"""Task 8 fix round 1 — Critical/Important findings.

C-1: history must never replace the current semantic query.
C-2: coreference scope must be resolved document IDs, not workspace IDs.
I-1: coreference must be reachable in production (build_draft seam).
I-2: bounded history read (covered via loader LIMIT/load_only assertions).
I-3: ingress seam tests with a real async-CM fake session + loader failure.
Minors M-1..M-5 explicitly deferred.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from uuid import UUID

DOC_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_DOC_ID = UUID("22222222-2222-2222-2222-222222222222")
WORKSPACE_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


def _legacy_result(query: str, refs=()):
    from app.services.agents.semantic_preprocessor import PreprocessingResult

    return PreprocessingResult(
        original_query=query,
        normalized_query=query.strip().lower(),
        abbreviations=[],
        document_refs=list(refs),
        blocking_ambiguities=[],
        preprocessing_status="ok",
        preprocessor_trace=[],
    )


def _resolved_legacy_ref(query: str, span: str, doc_id: UUID, ref_id: str = "r1"):
    from app.services.agents.semantic_preprocessor import DocumentRefEntry

    start = query.index(span)
    return DocumentRefEntry(
        ref_id=ref_id,
        original_span=span,
        span_offset=(start, start + len(span)),
        reference=span,
        section_reference=None,
        document_handle=doc_id,
        candidates=[],
        resolution_status="resolved",
    )


def _conversation(summary: str):
    from app.services.agents.v2.contracts.conversation import ConversationContext

    return ConversationContext(
        summary=summary, active_entities=(), last_focus=None, recent_turns=()
    )


def _request(query: str):
    from app.services.agents.v2.contracts.request import RequestContext

    return RequestContext(
        contract_version="2.0",
        request_id="r1",
        thread_id="t1",
        original_query=query,
        known_documents=(),
    )


# ---------------------------------------------------------------------------
# C-1: non-empty history must not replace the current query
# ---------------------------------------------------------------------------


def test_history_summary_does_not_replace_current_query():
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    query = "chế độ thai sản được quy định thế nào?"

    async def _preprocess(raw: str):
        return _legacy_result(raw)

    adapter = DeterministicSemanticAdapter(preprocess=_preprocess)
    draft = asyncio.run(
        adapter.build_draft(
            _request(query),
            _conversation("Hỏi về Nghị định 15/2020/NĐ-CP. LH 0901234567."),
        )
    )
    assert draft.provisional_contextualized_query == query.strip().lower()
    # person identity comes from the CURRENT query, never from history.
    assert draft.person_refs == ()


def test_current_query_person_identity_survives_history():
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    query = "0901234567 là ai?"

    async def _preprocess(raw: str):
        return _legacy_result(raw)

    adapter = DeterministicSemanticAdapter(preprocess=_preprocess)
    draft = asyncio.run(
        adapter.build_draft(_request(query), _conversation("Hỏi về thai sản."))
    )
    assert len(draft.person_refs) == 1
    assert draft.person_refs[0].label == "0901234567"


# ---------------------------------------------------------------------------
# C-2: resolve_draft_identities scopes coreference to resolved doc IDs
# ---------------------------------------------------------------------------


def test_resolve_draft_identities_resolves_unambiguous_mention():
    from app.services.agents.semantic_preprocessor import DocumentRefEntry
    from app.services.agents.v2.adapters.semantic import (
        draft_from_preprocessing,
        resolve_draft_identities,
    )

    query = "Luật An ninh mạng văn bản này quy định gì?"
    span = "Luật An ninh mạng"
    draft = draft_from_preprocessing(
        _legacy_result(
            query,
            refs=(
                DocumentRefEntry(
                    ref_id="r1",
                    original_span=span,
                    span_offset=(0, len(span)),
                    reference=span,
                    section_reference=None,
                    document_handle=None,
                    candidates=[],
                    resolution_status="deferred",
                ),
            ),
        )
    )

    class _StubResolver:
        async def resolve_reference(
            self, reference, *, question, workspace_ids, db,
            use_llm_fallback=True,
        ):
            return reference.model_copy(
                update={
                    "resolution_status": "resolved",
                    "resolved_document_id": DOC_ID,
                    "candidate_document_ids": (),
                }
            )

        def cached_section_label(self, *args, **kwargs):
            return None

    out = asyncio.run(
        resolve_draft_identities(
            draft,
            question=query,
            identity_resolver=_StubResolver(),
            # NOTE: a workspace/tenant UUID, NOT a document UUID.
            workspace_ids=(WORKSPACE_ID,),
            db=SimpleNamespace(),
        )
    )
    assert [c.resolved_ref_id for c in out.coreferences] == [
        r.ref_id for r in out.document_refs
        if r.resolution_status == "resolved"
    ][:1]
    assert len(out.coreferences) == 1
    assert out.preliminary_ambiguities == ()


# ---------------------------------------------------------------------------
# I-1: build_draft resolves mentions against current-turn authorized refs
# ---------------------------------------------------------------------------


def test_build_draft_resolves_van_ban_nay_to_single_authorized_ref():
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    query = "văn bản này quy định gì?"
    span = "Luật An ninh mạng"
    full = f"{span} {query}"

    async def _preprocess(raw: str):
        return _legacy_result(
            raw, refs=(_resolved_legacy_ref(raw, span, DOC_ID),)
        )

    adapter = DeterministicSemanticAdapter(preprocess=_preprocess)
    draft = asyncio.run(adapter.build_draft(_request(full), _conversation("")))
    assert [(c.mention, c.resolved_ref_id) for c in draft.coreferences] == [
        ("văn bản này", "r1")
    ]
    assert draft.preliminary_ambiguities == ()


def test_build_draft_true_ambiguity_becomes_clarification():
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    query = "văn bản này quy định gì?"
    span_a, span_b = "Luật An ninh mạng", "Luật Đất đai"
    full = f"{span_a} {span_b} {query}"

    async def _preprocess(raw: str):
        return _legacy_result(
            raw,
            refs=(
                _resolved_legacy_ref(raw, span_a, DOC_ID, ref_id="r1"),
                _resolved_legacy_ref(raw, span_b, OTHER_DOC_ID, ref_id="r2"),
            ),
        )

    adapter = DeterministicSemanticAdapter(preprocess=_preprocess)
    draft = asyncio.run(adapter.build_draft(_request(full), _conversation("")))
    assert draft.coreferences == ()
    assert len(draft.preliminary_ambiguities) == 1
    assert "văn bản này" in draft.preliminary_ambiguities[0].description


# ---------------------------------------------------------------------------
# I-3: ingress seam — real async-CM fake session
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class _FakeDB:
    def __init__(self, messages, summaries):
        self._messages = messages
        self._summaries = summaries

    async def execute(self, stmt):
        # Emulate the loader's DB-side DESC ordering (the loader reverses
        # messages back to chronological; summaries are re-sorted by the
        # adapter).
        text = str(stmt)
        if "chat_exchange_summaries" in text:
            rows = sorted(
                self._summaries, key=lambda s: s.exchange_index, reverse=True
            )
            return _FakeResult(rows)
        rows = sorted(
            self._messages,
            key=lambda m: (m.created_at, m.message_id),
            reverse=True,
        )
        return _FakeResult(rows)


class _FakeSession:
    """Async-CM session (unlike the legacy ingress-test fakes)."""

    def __init__(self, db):
        self._db = db
        self.closed = False

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, *args):
        return False

    async def close(self):
        self.closed = True

    async def commit(self):
        return None

    async def rollback(self):
        return None


def _history_fakes():
    from app.models.chat_message import ChatMessage
    from app.models.exchange_summary import ExchangeSummary

    from datetime import datetime, timedelta

    thread = uuid.uuid4()
    base = datetime(2026, 9, 14, 10, 0, 0)
    messages = [
        ChatMessage(
            session_id=thread, message_id="m1", role="user",
            content="Điều 5 Luật An ninh mạng nói gì?",
            created_at=base,
        ),
        ChatMessage(
            session_id=thread, message_id="m2", role="assistant",
            content="Điều 5 quy định...",
            created_at=base + timedelta(seconds=30),
        ),
    ]
    summaries = [
        ExchangeSummary(
            session_id=thread,
            exchange_index=1,
            user_message_id="m1",
            assistant_message_id="m2",
            topic_label="安 ninh mạng",
            summary="Hỏi về Điều 5.",
            key_entities=["Luật An ninh mạng", "Điều 5"],
        )
    ]
    db = _FakeDB(messages, summaries)
    return thread, db


def test_ingress_history_reaches_initial_conversation():
    import app.services.agent.runtime_selector as selector

    thread, db = _history_fakes()

    async def _run():
        async with selector.build_v2_ingress(
            user_id=uuid.uuid4(),
            authenticated_workspace_ids=[WORKSPACE_ID],
            requested_workspace_ids=None,
            raw_query="điều này áp dụng khi nào?",
            thread_id=str(thread),
            can_read_people=False,
            session_factory=lambda: _FakeSession(db),
            lease_session_factory=lambda: _FakeSession(db),
        ) as ingress:
            return ingress.initial_state

    state = asyncio.run(_run())
    conversation = state["conversation"]
    assert [t.content for t in conversation.recent_turns] == [
        "Điều 5 Luật An ninh mạng nói gì?",
        "Điều 5 quy định...",
    ]
    assert conversation.summary == "Hỏi về Điều 5."
    assert [e.kind for e in conversation.active_entities] == [
        "document", "section",
    ]
    # The current raw query is untouched by history.
    assert state["request"].original_query == "điều này áp dụng khi nào?"


def test_ingress_loader_failure_still_yields_valid_envelope():
    import app.services.agent.runtime_selector as selector

    class _FailingEnter(_FakeSession):
        async def __aenter__(self):
            raise RuntimeError("db down")

    async def _run():
        async with selector.build_v2_ingress(
            user_id=uuid.uuid4(),
            authenticated_workspace_ids=[WORKSPACE_ID],
            requested_workspace_ids=None,
            raw_query="chào anh, hỏi về chế độ thai sản?",
            thread_id=str(uuid.uuid4()),
            can_read_people=False,
            session_factory=lambda: _FailingEnter(_FakeDB([], [])),
            lease_session_factory=lambda: _FakeSession(_FakeDB([], [])),
        ) as ingress:
            return ingress.initial_state

    state = asyncio.run(_run())
    assert state["request"].original_query == "chào anh, hỏi về chế độ thai sản?"
    assert state["conversation"].recent_turns == ()
    assert state["conversation"].summary == ""

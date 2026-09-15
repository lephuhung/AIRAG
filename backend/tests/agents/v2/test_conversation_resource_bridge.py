"""Task 1B — server-issued conversation-resource bridge (A.3).

History surfaces only server-issued document identities as
``KnownDocumentResource(source="conversation")``; never UUIDs parsed from
message text. Conversation-backed semantic refs materialize only on a
supported anaphora and still flow through the existing identity/binding
boundary (current ACL recheck, binder-only revision pins).
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from uuid import UUID

import pytest

DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")
DOC_C = UUID("33333333-3333-3333-3333-333333333333")
OUT_OF_SCOPE = UUID("99999999-9999-9999-9999-999999999999")
WORKSPACE_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


def _msg(role="assistant", content="...", **columns):
    row = SimpleNamespace(role=role, content=content)
    for key, value in columns.items():
        setattr(row, key, value)
    return row


# ---------------------------------------------------------------------------
# Extractor: server-issued identities only, chronological, deduped, bounded
# ---------------------------------------------------------------------------


def test_extractor_merges_document_ids_citations_sources_chronologically():
    from app.services.agents.v2.adapters.conversation import (
        conversation_resources_from_legacy,
    )

    messages = [
        _msg("user", "q1", document_ids=[str(DOC_A)]),
        _msg(
            "assistant",
            "a1",
            citations=[{"citation_id": "c1", "label": "A", "document_id": str(DOC_B)}],
            sources=[{"document_id": str(DOC_C), "chunk_id": "ch1"}],
        ),
    ]
    resources = conversation_resources_from_legacy(messages)
    assert [r.document_id for r in resources] == [DOC_A, DOC_B, DOC_C]
    assert all(r.source == "conversation" for r in resources)
    assert [r.resource_id for r in resources] == ["conv-1", "conv-2", "conv-3"]


def test_extractor_dedupes_keep_first_and_ignores_malformed():
    from app.services.agents.v2.adapters.conversation import (
        conversation_resources_from_legacy,
    )

    messages = [
        _msg(
            "assistant",
            "a1",
            document_ids=[str(DOC_A), "not-a-uuid", "", None, 12345],
            citations=[{"document_id": str(DOC_A)}, {"document_id": "bogus"}],
            sources="not-a-list",
        ),
        _msg("assistant", "a2", citations=[{"document_id": str(DOC_A)}]),
    ]
    resources = conversation_resources_from_legacy(messages)
    assert [r.document_id for r in resources] == [DOC_A]


def test_extractor_never_parses_message_text():
    from app.services.agents.v2.adapters.conversation import (
        conversation_resources_from_legacy,
    )

    messages = [_msg("assistant", f"see document {DOC_B} for details")]
    assert conversation_resources_from_legacy(messages) == ()


def test_extractor_is_bounded():
    from app.services.agents.v2.adapters import conversation as conv_mod
    from app.services.agents.v2.adapters.conversation import (
        conversation_resources_from_legacy,
    )

    messages = [_msg("assistant", "a", document_ids=[str(uuid.uuid4())]) for _ in range(40)]
    resources = conversation_resources_from_legacy(messages)
    assert len(resources) == conv_mod.MAX_CONVERSATION_RESOURCES


# ---------------------------------------------------------------------------
# Anaphora gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("văn bản này quy định gì?", ("mention", 0)),
        ("tóm tắt tài liệu này", ("mention", 0)),
        ("tóm tắt file thứ hai", ("ordinal", 2)),
        ("văn bản thứ nhất nói gì?", ("ordinal", 1)),
        ("chế độ thai sản được quy định thế nào?", None),
        ("", None),
    ],
)
def test_detect_conversation_anaphora(query, expected):
    from app.services.agents.v2.semantic.discourse import detect_conversation_anaphora

    found = detect_conversation_anaphora(query)
    if expected is None:
        assert found is None
    else:
        assert (found[0], found[1]) == expected
        assert found[2]  # surface span preserved for the ref shell


# ---------------------------------------------------------------------------
# Adapter projection: anaphora-gated, deterministic, idempotent
# ---------------------------------------------------------------------------


def _request(query, known=()):
    from app.services.agents.v2.contracts.request import RequestContext

    return RequestContext(
        contract_version="2.0",
        request_id="r1",
        thread_id="t1",
        original_query=query,
        known_documents=tuple(known),
    )


def _conv_known(doc_id, resource_id):
    from app.services.agents.v2.contracts.request import KnownDocumentResource

    return KnownDocumentResource(
        resource_id=resource_id, document_id=doc_id, source="conversation"
    )


def _empty_draft():
    from app.services.agents.v2.contracts.semantic import SemanticDraft

    return SemanticDraft(
        provisional_contextualized_query="q",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=(),
    )


def test_projection_single_candidate_mention_yields_one_resolved_ref():
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    draft = DeterministicSemanticAdapter.project_conversation_targets(
        _empty_draft(),
        _request("văn bản này quy định gì?", [_conv_known(DOC_A, "conv-1")]),
    )
    assert len(draft.document_refs) == 1
    ref = draft.document_refs[0]
    assert ref.resolved_document_id == DOC_A
    assert ref.resolution_status == "resolved"
    assert ref.ref_id == "conversation:conv-1"
    assert ref.revision_requirement is None  # binder pins, never history


def test_projection_ordinal_selects_second_candidate():
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    draft = DeterministicSemanticAdapter.project_conversation_targets(
        _empty_draft(),
        _request(
            "tóm tắt file thứ hai",
            [_conv_known(DOC_A, "conv-1"), _conv_known(DOC_B, "conv-2")],
        ),
    )
    assert [r.resolved_document_id for r in draft.document_refs] == [DOC_B]


def test_projection_mention_two_candidates_projects_both_for_clarify():
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    draft = DeterministicSemanticAdapter.project_conversation_targets(
        _empty_draft(),
        _request(
            "văn bản này quy định gì?",
            [_conv_known(DOC_A, "conv-1"), _conv_known(DOC_B, "conv-2")],
        ),
    )
    assert [r.resolved_document_id for r in draft.document_refs] == [DOC_A, DOC_B]


def test_projection_without_anaphora_or_out_of_range_is_silent():
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    known = [_conv_known(DOC_A, "conv-1")]
    assert (
        DeterministicSemanticAdapter.project_conversation_targets(
            _empty_draft(), _request("chế độ thai sản là gì?", known)
        ).document_refs
        == ()
    )
    assert (
        DeterministicSemanticAdapter.project_conversation_targets(
            _empty_draft(), _request("tóm tắt file thứ hai", known)
        ).document_refs
        == ()
    )
    assert (
        DeterministicSemanticAdapter.project_conversation_targets(
            _empty_draft(), _request("văn bản này là gì?")
        ).document_refs
        == ()
    )


def test_projection_is_idempotent_and_ignores_non_conversation_known():
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter
    from app.services.agents.v2.contracts.request import KnownDocumentResource

    known = (
        KnownDocumentResource(
            resource_id="att-1", document_id=DOC_C, source="attachment"
        ),
        _conv_known(DOC_A, "conv-1"),
    )
    once = DeterministicSemanticAdapter.project_conversation_targets(
        _empty_draft(), _request("văn bản này là gì?", known)
    )
    twice = DeterministicSemanticAdapter.project_conversation_targets(
        once, _request("văn bản này là gì?", known)
    )
    assert twice == once
    assert [r.resolved_document_id for r in once.document_refs] == [DOC_A]


def test_build_draft_end_to_end_conversation_mention_resolves_coref():
    """Preprocessor ref + history candidate + mention → deterministic coref."""
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    async def _preprocess(raw: str):
        from app.services.agents.semantic_preprocessor import PreprocessingResult

        return PreprocessingResult(
            original_query=raw,
            normalized_query=raw.strip().lower(),
            abbreviations=[],
            document_refs=[],
            blocking_ambiguities=[],
            preprocessing_status="ok",
            preprocessor_trace=[],
        )

    from app.services.agents.v2.contracts.conversation import ConversationContext

    adapter = DeterministicSemanticAdapter(preprocess=_preprocess)
    draft = asyncio.run(
        adapter.build_draft(
            _request("văn bản này quy định gì?", [_conv_known(DOC_A, "conv-1")]),
            ConversationContext(
                summary="", active_entities=(), last_focus=None, recent_turns=()
            ),
        )
    )
    assert [r.ref_id for r in draft.document_refs] == ["conversation:conv-1"]
    assert [(c.mention, c.resolved_ref_id) for c in draft.coreferences] == [
        ("văn bản này", "conversation:conv-1")
    ]
    assert draft.preliminary_ambiguities == ()


# ---------------------------------------------------------------------------
# Binder boundary: history identity is ACL-rechecked, never directly bound
# ---------------------------------------------------------------------------


def test_binder_rechecks_history_identity_against_current_scope():
    """Out-of-scope historical doc → fail-closed, no binding, no title leak."""
    from app.services.agents.supervisor_v2 import V1BindingResolver

    from app.services.agents.v2.contracts.semantic import DocumentReference

    ref = DocumentReference(
        ref_id="conversation:conv-1",
        original_span="văn bản này",
        normalized_reference="văn bản này",
        requested_role="target",
        revision_requirement=None,
        resolution_status="resolved",
        resolved_document_id=OUT_OF_SCOPE,
    )

    class _EmptyDB:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, *args, **kwargs):
            raise AssertionError("no rows")

    from app.services.agents.v2.contracts.capability import (
        CapabilityRuntimeContext,
    )
    from datetime import UTC, datetime

    runtime = CapabilityRuntimeContext(
        request_id="req-1",
        run_id="run-1",
        user_id=uuid.uuid4(),
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=frozenset(),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    resolver = V1BindingResolver(
        session_factory=lambda: _EmptyDB(), default_role="target"
    )
    with pytest.raises(Exception):
        asyncio.run(resolver.resolve((ref,), runtime))


# ---------------------------------------------------------------------------
# Ingress: loader bundle + merge, fail-open preserved
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
        text = str(stmt)
        if "chat_exchange_summaries" in text:
            return _FakeResult(
                sorted(self._summaries, key=lambda s: s.exchange_index, reverse=True)
            )
        return _FakeResult(
            sorted(
                self._messages,
                key=lambda m: (m.created_at, m.message_id),
                reverse=True,
            )
        )


class _FakeSession:
    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, *args):
        return False

    async def close(self):
        return None

    async def commit(self):
        return None

    async def rollback(self):
        return None


def _history_db():
    from datetime import datetime, timedelta

    from app.models.chat_message import ChatMessage
    from app.models.exchange_summary import ExchangeSummary

    thread = uuid.uuid4()
    base = datetime(2026, 9, 14, 10, 0, 0)
    messages = [
        ChatMessage(
            session_id=thread,
            message_id="m1",
            role="user",
            content="Tóm tắt nghị định A",
            created_at=base,
        ),
        ChatMessage(
            session_id=thread,
            message_id="m2",
            role="assistant",
            content="Tóm tắt...",
            created_at=base + timedelta(seconds=30),
            document_ids=[str(DOC_A)],
            citations=[
                {"citation_id": "c1", "label": "Doc B", "document_id": str(DOC_B)}
            ],
        ),
    ]
    summaries = [
        ExchangeSummary(
            session_id=thread,
            exchange_index=1,
            user_message_id="m1",
            assistant_message_id="m2",
            topic_label="nghị định",
            summary="Hỏi về nghị định A.",
            key_entities=["Nghị định A"],
        )
    ]
    return thread, _FakeDB(messages, summaries)


def test_ingress_merges_conversation_resources_with_current_turn():
    import app.services.agent.runtime_selector as selector
    from app.services.agents.v2.contracts.request import KnownDocumentResource

    thread, db = _history_db()
    current = (
        KnownDocumentResource(
            resource_id=str(DOC_C), document_id=DOC_C, source="api_explicit"
        ),
    )

    async def _run():
        async with selector.build_v2_ingress(
            user_id=uuid.uuid4(),
            authenticated_workspace_ids=[WORKSPACE_ID],
            requested_workspace_ids=None,
            raw_query="văn bản này có hiệu lực khi nào?",
            thread_id=str(thread),
            can_read_people=False,
            known_documents=current,
            session_factory=lambda: _FakeSession(db),
            lease_session_factory=lambda: _FakeSession(db),
        ) as ingress:
            return ingress.initial_state

    state = asyncio.run(_run())
    known = state["request"].known_documents
    assert [(k.source, k.document_id) for k in known] == [
        ("api_explicit", DOC_C),
        ("conversation", DOC_A),
        ("conversation", DOC_B),
    ]
    # Label/text context behavior unchanged.
    assert state["conversation"].summary == "Hỏi về nghị định A."
    assert [t.content for t in state["conversation"].recent_turns] == [
        "Tóm tắt nghị định A",
        "Tóm tắt...",
    ]


def test_ingress_conversation_duplicate_of_current_turn_keeps_ordinal():
    import app.services.agent.runtime_selector as selector
    from app.services.agents.v2.contracts.request import KnownDocumentResource

    thread, db = _history_db()
    current = (
        KnownDocumentResource(
            resource_id=str(DOC_A), document_id=DOC_A, source="api_explicit"
        ),
    )

    async def _run():
        async with selector.build_v2_ingress(
            user_id=uuid.uuid4(),
            authenticated_workspace_ids=[WORKSPACE_ID],
            requested_workspace_ids=None,
            raw_query="tóm tắt file thứ hai",
            thread_id=str(thread),
            can_read_people=False,
            known_documents=current,
            session_factory=lambda: _FakeSession(db),
            lease_session_factory=lambda: _FakeSession(db),
        ) as ingress:
            return ingress.initial_state

    state = asyncio.run(_run())
    conv = [k for k in state["request"].known_documents if k.source == "conversation"]
    assert [k.document_id for k in conv] == [DOC_B]
    assert [k.resource_id for k in conv] == ["conv-1"]


def test_ingress_standalone_and_failure_stay_fail_open():
    import app.services.agent.runtime_selector as selector

    async def _run(thread_id, factory):
        async with selector.build_v2_ingress(
            user_id=uuid.uuid4(),
            authenticated_workspace_ids=[WORKSPACE_ID],
            requested_workspace_ids=None,
            raw_query="văn bản này là gì?",
            thread_id=thread_id,
            can_read_people=False,
            session_factory=factory,
            lease_session_factory=lambda: _FakeSession(_FakeDB([], [])),
        ) as ingress:
            return ingress.initial_state

    state = asyncio.run(
        _run("standalone-abc123", lambda: _FakeSession(_FakeDB([], [])))
    )
    assert state["request"].known_documents == ()
    assert state["conversation"].recent_turns == ()

    class _Failing(_FakeSession):
        async def __aenter__(self):
            raise RuntimeError("db down")

    failed = asyncio.run(_run(str(uuid.uuid4()), lambda: _Failing(_FakeDB([], []))))
    assert failed["request"].known_documents == ()
    assert failed["conversation"].recent_turns == ()

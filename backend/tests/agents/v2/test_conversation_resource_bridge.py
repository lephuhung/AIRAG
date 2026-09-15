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


# ---------------------------------------------------------------------------
# Blank-history fix: blank rows keep resources, leave recent_turns;
# strict non-string / unsupported-role failures are preserved.
# ---------------------------------------------------------------------------


def _legacy_msg(role="assistant", content="...", **columns):
    row = SimpleNamespace(role=role, content=content)
    for key, value in columns.items():
        setattr(row, key, value)
    return row


def test_mixed_text_and_blank_preserves_chronological_nonblank_turns():
    from app.services.agents.v2.adapters.conversation import context_from_legacy

    context = context_from_legacy(
        messages=[
            _legacy_msg("user", "first question"),
            _legacy_msg("assistant", ""),
            _legacy_msg("assistant", "   \n\t  "),
            _legacy_msg("assistant", "second answer"),
        ]
    )
    assert [t.content for t in context.recent_turns] == [
        "first question",
        "second answer",
    ]


def test_whitespace_only_content_skipped():
    from app.services.agents.v2.adapters.conversation import context_from_legacy

    context = context_from_legacy(
        messages=[_legacy_msg("user", "q"), _legacy_msg("assistant", "   ")]
    )
    assert [t.content for t in context.recent_turns] == ["q"]


def test_all_blank_rows_yield_valid_empty_recent_turns():
    from app.services.agents.v2.adapters.conversation import context_from_legacy

    context = context_from_legacy(
        messages=[_legacy_msg("user", ""), _legacy_msg("assistant", "  ")]
    )
    assert context.recent_turns == ()
    assert context.summary == ""


def test_non_string_content_remains_error():
    from app.services.agents.v2.adapters.conversation import (
        ConversationAdapterError,
        context_from_legacy,
    )

    with pytest.raises(ConversationAdapterError):
        context_from_legacy(messages=[_legacy_msg("user", None)])


def test_unsupported_role_remains_error():
    from app.services.agents.v2.adapters.conversation import (
        ConversationAdapterError,
        context_from_legacy,
    )

    with pytest.raises(ConversationAdapterError):
        context_from_legacy(messages=[_legacy_msg("tool", "some text")])
    # Role strictness applies even when the content is blank.
    with pytest.raises(ConversationAdapterError):
        context_from_legacy(messages=[_legacy_msg("tool", "")])


def test_max_recent_turns_bounds_loaded_rows():
    from app.services.agents.v2.adapters.conversation import context_from_legacy

    messages = [_legacy_msg("user", f"q{i}") for i in range(5)]
    context = context_from_legacy(messages=messages, max_recent_turns=2)
    assert [t.content for t in context.recent_turns] == ["q3", "q4"]


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


# I1 (fix round) — real deny semantics: exact RevisionNotReady, no binding,
# no surfaced identity. The workspace guard is stubbed (not the DB driver),
# so both deny shapes exercise production paths. Note: the internal typed
# error carries the document UUID by frozen contract (RevisionNotReady
# formats it in); it stays server-side (logged detail) while the
# user-facing boundary conversion is generic — pinned below.
_DENY_REASON = (
    "document does not exist, is not owned by this workspace, or "
    "is tombstoned"
)
_SECRET_TITLE = "Tiêu Đề Tuyệt Mật Không Bao Giờ Lộ"


def _conversation_ref(doc_id):
    from app.services.agents.v2.contracts.semantic import DocumentReference

    return DocumentReference(
        ref_id="conversation:conv-1",
        original_span="văn bản này",
        normalized_reference="văn bản này",
        requested_role="target",
        revision_requirement=None,
        resolution_status="resolved",
        resolved_document_id=doc_id,
    )


def _deny_runtime():
    from datetime import UTC, datetime

    from app.services.agents.v2.contracts.capability import (
        CapabilityRuntimeContext,
    )

    return CapabilityRuntimeContext(
        request_id="req-1",
        run_id="run-1",
        user_id=uuid.uuid4(),
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=False,
        allowed_capabilities=frozenset(),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


class _UnusedDB:
    """Session the deny stub never queries (deny happens in the loader)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def test_history_identity_out_of_scope_denies_with_real_deny_type(monkeypatch):
    """I1: foreign/tombstoned doc → exact RevisionNotReady, nothing pinned."""
    from app.services.agents.supervisor_v2 import V1BindingResolver
    from app.services.agents.v2.persistence import document_views
    from app.services.agents.v2.persistence.document_views import RevisionNotReady

    async def _deny(db, document_id, workspace_id, **kwargs):
        raise RevisionNotReady(document_id, _DENY_REASON)

    monkeypatch.setattr(
        document_views, "load_current_revision_identity_for_workspace", _deny
    )
    resolver = V1BindingResolver(
        session_factory=lambda: _UnusedDB(), default_role="target"
    )
    with pytest.raises(RevisionNotReady) as excinfo:
        asyncio.run(
            resolver.resolve((_conversation_ref(OUT_OF_SCOPE),), _deny_runtime())
        )
    # Fail-closed: the typed deny (not a driver error), no binding set
    # escapes, and no title ever enters this path.
    assert excinfo.value.code == "REVISION_NOT_READY"
    assert excinfo.value.document_id == str(OUT_OF_SCOPE)
    assert _SECRET_TITLE not in str(excinfo.value)


def test_history_identity_without_current_revision_denies(monkeypatch):
    """I1: legacy doc (guard returns None) → adapter denies, nothing pinned."""
    from app.services.agents.supervisor_v2 import V1BindingResolver
    from app.services.agents.v2.persistence import document_views
    from app.services.agents.v2.persistence.document_views import RevisionNotReady

    async def _no_current(db, document_id, workspace_id, **kwargs):
        return None

    monkeypatch.setattr(
        document_views,
        "load_current_revision_identity_for_workspace",
        _no_current,
    )
    resolver = V1BindingResolver(
        session_factory=lambda: _UnusedDB(), default_role="target"
    )
    with pytest.raises(RevisionNotReady) as excinfo:
        asyncio.run(resolver.resolve((_conversation_ref(DOC_A),), _deny_runtime()))
    assert excinfo.value.code == "REVISION_NOT_READY"


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


# ---------------------------------------------------------------------------
# I2 (fix round) — >=2-candidate downstream: counts-only ambiguity, no
# identity in clarification text, ordinal recovery. Pins the approved A.3
# behavior (no production change): a bare mention over several candidates
# becomes a candidate-free blocking question; restating with an ordinal
# resolves deterministically.
# ---------------------------------------------------------------------------


def _preprocess_no_refs(raw: str):
    from app.services.agents.semantic_preprocessor import PreprocessingResult

    async def _run(inner: str):
        return PreprocessingResult(
            original_query=inner,
            normalized_query=inner.strip().lower(),
            abbreviations=[],
            document_refs=[],
            blocking_ambiguities=[],
            preprocessing_status="ok",
            preprocessor_trace=[],
        )

    return _run(raw)


def _empty_conversation():
    from app.services.agents.v2.contracts.conversation import ConversationContext

    return ConversationContext(
        summary="", active_entities=(), last_focus=None, recent_turns=()
    )


def _draft_for(query, known):
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    adapter = DeterministicSemanticAdapter(preprocess=_preprocess_no_refs)
    return asyncio.run(
        adapter.build_draft(_request(query, known), _empty_conversation())
    )


def test_two_candidates_mention_yields_count_only_ambiguity():
    """I2: bare mention over 2 candidates → 1 generic ambiguity, no refs."""
    draft = _draft_for(
        "văn bản này quy định gì?",
        [_conv_known(DOC_A, "conv-1"), _conv_known(DOC_B, "conv-2")],
    )
    assert [r.ref_id for r in draft.document_refs] == [
        "conversation:conv-1",
        "conversation:conv-2",
    ]
    assert draft.coreferences == ()
    assert len(draft.preliminary_ambiguities) == 1
    description = draft.preliminary_ambiguities[0].description
    assert str(DOC_A) not in description
    assert str(DOC_B) not in description
    assert _SECRET_TITLE not in description


def test_two_candidates_clarification_carries_no_identity():
    """I2: downstream clarification offers no candidates and names none."""
    from app.services.agents.v2.adapters.semantic import finalize_semantic_context
    from app.services.agents.v2.nodes.clarification import build_clarification

    draft = _draft_for(
        "văn bản này quy định gì?",
        [_conv_known(DOC_A, "conv-1"), _conv_known(DOC_B, "conv-2")],
    )
    semantic = finalize_semantic_context(draft)
    request = build_clarification(semantic)
    assert request.reason == "semantic_ambiguity"
    assert request.candidates == ()
    assert request.unresolved_ref_ids == ()
    assert str(DOC_A) not in request.question
    assert str(DOC_B) not in request.question
    assert _SECRET_TITLE not in request.question


def test_ordinal_recovers_second_candidate_after_ambiguity():
    """I2: restated ordinal resolves deterministically, ambiguity clears."""
    draft = _draft_for(
        "tóm tắt file thứ hai",
        [_conv_known(DOC_A, "conv-1"), _conv_known(DOC_B, "conv-2")],
    )
    assert [r.resolved_document_id for r in draft.document_refs] == [DOC_B]
    assert draft.preliminary_ambiguities == ()
    # The ordinal is consumed at projection time (single correct ref), so
    # the coref pass — which counts positions within resolved refs — stays
    # silent; binding still uses the projected document_refs below.
    assert draft.coreferences == ()


def test_denied_binding_surfaces_generic_user_facing_error():
    """I1: UUID/title-laden internal detail → generic boundary text only."""
    from app.services.agents import supervisor_v2
    from app.services.agents.v2.persistence.document_views import RevisionNotReady

    try:
        raise RevisionNotReady(OUT_OF_SCOPE, _DENY_REASON)
    except RevisionNotReady as exc:
        detail = f"binding: {exc} ({_SECRET_TITLE})"
    update = supervisor_v2._typed_boundary_error(detail)
    content = update["final_response"].content
    assert str(OUT_OF_SCOPE) not in content
    assert _SECRET_TITLE not in content
    assert content == supervisor_v2._BOUNDARY_ERROR_CONTENT
    # Conversion is sticky: no route survives to resurrect the turn.
    assert update["route_decision"] is None
    assert update["query_analysis"] is None


# ---------------------------------------------------------------------------
# Blank-history fix at the loader/ingress boundary: blank rows still
# contribute server-issued resources while recent_turns stays valid.
# ---------------------------------------------------------------------------


def _blank_history_db():
    from datetime import datetime, timedelta

    from app.models.chat_message import ChatMessage

    thread = uuid.uuid4()
    base = datetime(2026, 9, 14, 10, 0, 0)
    messages = [
        # Blank attachment-only user row (live-DB shape: empty content).
        ChatMessage(
            session_id=thread,
            message_id="m1",
            role="user",
            content="",
            created_at=base,
            document_ids=[str(DOC_A)],
        ),
        # Blank citation-only assistant row.
        ChatMessage(
            session_id=thread,
            message_id="m2",
            role="assistant",
            content="   ",
            created_at=base + timedelta(seconds=30),
            citations=[
                {"citation_id": "c1", "label": "Doc B", "document_id": str(DOC_B)}
            ],
        ),
    ]
    return thread, _FakeDB(messages, [])


def test_blank_attachment_only_row_contributes_resource_through_ingress():
    import app.services.agent.runtime_selector as selector

    thread, db = _blank_history_db()

    async def _run():
        async with selector.build_v2_ingress(
            user_id=uuid.uuid4(),
            authenticated_workspace_ids=[WORKSPACE_ID],
            requested_workspace_ids=None,
            raw_query="văn bản này có hiệu lực khi nào?",
            thread_id=str(thread),
            can_read_people=False,
            session_factory=lambda: _FakeSession(db),
            lease_session_factory=lambda: _FakeSession(db),
        ) as ingress:
            return ingress.initial_state

    state = asyncio.run(_run())
    known = state["request"].known_documents
    assert [(k.source, k.document_id) for k in known] == [
        ("conversation", DOC_A),
        ("conversation", DOC_B),
    ]
    # Blank rows leave recent_turns (valid empty), not a failed turn.
    assert state["conversation"].recent_turns == ()


def test_blank_citation_only_row_resolves_through_adapter():
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter
    from app.services.agents.v2.contracts.conversation import ConversationContext

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

    adapter = DeterministicSemanticAdapter(preprocess=_preprocess)
    draft = asyncio.run(
        adapter.build_draft(
            _request(
                "văn bản này quy định gì?",
                [_conv_known(DOC_B, "conv-1")],
            ),
            ConversationContext(
                summary="", active_entities=(), last_focus=None, recent_turns=()
            ),
        )
    )
    assert [r.resolved_document_id for r in draft.document_refs] == [DOC_B]

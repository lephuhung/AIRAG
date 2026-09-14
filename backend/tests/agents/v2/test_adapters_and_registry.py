"""Phase 1D Task 11 — typed legacy adapters and the capability registry.

Covers the Task-11 Step-1 requirements:

- ``RequestContext.original_query`` remains the only raw-query owner; the
  semantic adapter never copies it into ``SemanticDraft``/``SemanticContext``.
- Draft → Binding → Finalizer flow.
- Immutable revision lookup through ``persistence.document_views`` (never the
  mutable document-view artifacts).
- Typed adapter outputs — no ``dict[str, Any]`` escape hatch.
- Runtime capability intersection (permissions / feature flags / availability).
- Validation rather than the v1 ``model_construct()`` bypass: an incompatible
  legacy payload is rejected or translated explicitly.

These tests are deliberately database-free: the document adapter calls
``persistence.document_views`` through module attributes, so the immutable
revision lookup is asserted by monkeypatching that module's async functions.
"""
from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from uuid import UUID

import pytest
from pydantic import BaseModel

from app.services.agents.deep_research.contracts import TaskResult as LegacyTaskResult
from app.services.agents.semantic_preprocessor import (
    AbbreviationCandidate,
    AbbreviationEntry,
    BlockingAmbiguity as LegacyBlockingAmbiguity,
    DocumentCandidate,
    DocumentRefEntry,
    PreprocessingResult,
    TraceEvent,
    to_persisted_dict,
)
from app.services.agents.v2.adapters import (
    ConversationAdapterError,
    DeepResearchAdapterError,
    DocumentAdapterError,
    ResolvedDocumentBindings,
    SemanticAdapterError,
    agent_result_from_legacy,
    agent_status_from_legacy,
    context_from_legacy,
    draft_from_persisted_semantic,
    draft_from_preprocessing,
    finalize_semantic_context,
    finalize_semantics,
    resolve_document_binding,
    resolve_document_bindings,
    snapshot_from_legacy,
)
from app.services.agents.v2.capabilities import (
    Capability,
    CapabilityDenied,
    CapabilityNotRegistered,
    CapabilityRegistration,
    CapabilityRegistry,
    CapabilityUnavailable,
    build_capability_registry,
)
from app.services.agents.v2.contracts import capability as capability_contracts
from app.services.agents.v2.contracts.binding import DocumentBindingSet, ScopedDocument
from app.services.agents.v2.contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
)
from app.services.agents.v2.contracts.conversation import (
    ConversationContext,
    ConversationSnapshot,
)
from app.services.agents.v2.contracts.execution import AgentResult, AgentRequest
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.semantic import (
    CurrentRevisionRequirement,
    DocumentReference,
    PinnedRevisionRequirement,
    SemanticContext,
    SemanticDraft,
    SemanticSnapshot,
)
from app.services.agents.v2.contracts.validation import (
    validate_binding_set,
    validate_conversation_snapshot,
    validate_semantic_context,
)
from app.services.agents.v2.persistence import document_views

RAW_QUERY = "NĐ 12/2020 có hiệu lực không?"
NORMALIZED_QUERY = "nghị định 12/2020 có hiệu lực không?"
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")
CURRENT_REVISION_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
PINNED_REVISION_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")

REF_SPAN = "NĐ 12/2020"


# ---------------------------------------------------------------------------
# Legacy fixtures and fakes
# ---------------------------------------------------------------------------


def request_context() -> RequestContext:
    return RequestContext(
        contract_version="2.0",
        request_id="req-1",
        thread_id="thread-1",
        original_query=RAW_QUERY,
        known_documents=(),
    )


def legacy_preprocessing(
    *,
    normalized_query: str | None = NORMALIZED_QUERY,
    resolution_status: str = "resolved",
    document_handle: UUID | None = DOCUMENT_ID,
    candidates: list[DocumentCandidate] | None = None,
    blocked: bool = False,
) -> PreprocessingResult:
    return PreprocessingResult(
        original_query=RAW_QUERY,
        normalized_query=normalized_query,
        abbreviations=[
            AbbreviationEntry(
                span="NĐ",
                span_offset=(0, 2),
                short_form="nđ",
                chosen="Nghị định",
                candidates=[AbbreviationCandidate(full_form="Nghị định", description=None)],
                status="resolved",
                confidence="high",
                source="db_single",
            )
        ],
        document_refs=[
            DocumentRefEntry(
                ref_id="r1",
                original_span=REF_SPAN,
                span_offset=(0, len(REF_SPAN)),
                reference="Nghị định 12/2020",
                document_handle=document_handle,
                candidates=candidates or [],
                resolution_status=resolution_status,
            )
        ],
        blocking_ambiguities=(
            [
                LegacyBlockingAmbiguity(
                    description="Cần làm rõ tài liệu nào được đề cập.",
                    essential=True,
                    source_ref="r1",
                    category="document_identity",
                )
            ]
            if blocked
            else []
        ),
        preprocessing_status="ok",
        preprocessor_trace=[TraceEvent(step="input", started_at=0.0, ended_at=0.1)],
    )


def revision_identity(
    revision_id: UUID = CURRENT_REVISION_ID,
    document_id: UUID = DOCUMENT_ID,
) -> document_views.RevisionArtifactIdentity:
    return document_views.RevisionArtifactIdentity(
        revision_id=revision_id,
        document_id=document_id,
        generation=1,
        build_profile="FULL",
        markdown_artifact_key="markdown.md",
        structure_artifact_key="structure.json",
        embedding_namespace=None,
        embedding_model_hash=None,
        embedding_dimension=None,
        vector_artifact_version=None,
    )


def resolved_reference(
    *,
    ref_id: str = "r1",
    revision_requirement: object = None,
    document_id: UUID = DOCUMENT_ID,
) -> DocumentReference:
    return DocumentReference(
        ref_id=ref_id,
        original_span=REF_SPAN,
        normalized_reference="Nghị định 12/2020",
        requested_role="target",
        revision_requirement=revision_requirement,
        resolution_status="resolved",
        resolved_document_id=document_id,
    )


def runtime_context(
    *,
    allowed: frozenset[str] = frozenset({"document.read", "document.search", "people.lookup"}),
    can_read_people: bool = True,
) -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id="req-1",
        run_id="run-1",
        user_id=USER_ID,
        workspace_ids=(WORKSPACE_ID,),
        can_read_people=can_read_people,
        allowed_capabilities=allowed,
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


class FakeCapability:
    def __init__(self, name: str, *, domain: str = "document") -> None:
        self.descriptor = CapabilityDescriptor(
            name=name,
            domain=domain,
            operation_type="read",
            supports_parallel=False,
        )

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        return AgentResult(
            contract_version="2.0",
            task_id=request.task_id,
            status="success",
            data=None,
            evidence_uses=(),
            coverage_observations=(),
            error=None,
        )


def registration(
    name: str,
    *,
    domain: str = "document",
    feature_flag: str | None = None,
    service: str | None = None,
) -> CapabilityRegistration:
    return CapabilityRegistration(
        capability=FakeCapability(name, domain=domain),
        feature_flag=feature_flag,
        service=service,
    )


class LegacyMessageProtocol(Protocol):
    role: str
    content: str


# ---------------------------------------------------------------------------
# Raw-query ownership (spec §8.3 / §26)
# ---------------------------------------------------------------------------


def test_draft_predicate_does_not_exist_on_semantic_models() -> None:
    assert "original_query" in RequestContext.model_fields
    assert "original_query" not in SemanticDraft.model_fields
    assert "original_query" not in SemanticContext.model_fields


def test_draft_from_preprocessing_does_not_copy_the_raw_query() -> None:
    draft = draft_from_preprocessing(legacy_preprocessing())

    assert isinstance(draft, SemanticDraft)
    assert not hasattr(draft, "original_query")
    assert draft.provisional_contextualized_query == NORMALIZED_QUERY
    assert draft.provisional_contextualized_query != RAW_QUERY


def test_draft_refuses_to_fall_back_to_the_immutable_raw_query() -> None:
    with pytest.raises(SemanticAdapterError):
        draft_from_preprocessing(legacy_preprocessing(normalized_query=None))


def test_finalize_semantic_context_uses_contextualized_normalized_forms() -> None:
    draft = draft_from_preprocessing(legacy_preprocessing())
    context = finalize_semantic_context(draft)

    assert isinstance(context, SemanticContext)
    assert not hasattr(context, "original_query")
    assert context.contextualized_query == NORMALIZED_QUERY
    assert context.normalized_query == NORMALIZED_QUERY
    assert context.normalized_query != RAW_QUERY
    validate_semantic_context(context)


# ---------------------------------------------------------------------------
# Persisted legacy payload: validation instead of model_construct()
# ---------------------------------------------------------------------------


def test_persisted_semantic_payload_translates_to_a_typed_draft() -> None:
    draft = draft_from_persisted_semantic(to_persisted_dict(legacy_preprocessing()))

    assert isinstance(draft, SemanticDraft)
    assert draft.abbreviations[0].abbreviation == "NĐ"
    assert draft.abbreviations[0].expansion == "Nghị định"
    assert draft.document_refs[0].ref_id == "r1"
    assert draft.document_refs[0].resolution_status == "resolved"
    assert draft.document_refs[0].resolved_document_id == DOCUMENT_ID


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("version"),
        lambda payload: payload.update(version="0.9"),
        lambda payload: payload.update(normalized_query=None),
        lambda payload: payload.update(document_refs=[{"ref_id": "r1"}]),
        lambda payload: payload.update(document_refs=[dict(payload["document_refs"][0], resolution_status="bogus")]),
        lambda payload: payload.update(document_refs=[dict(payload["document_refs"][0], document_handle="not-a-uuid")]),
        lambda payload: payload.update(abbreviations=[{"short_form": "nđ", "status": "resolved"}]),
        lambda payload: payload.update(abbreviations=[{"span": "NĐ", "short_form": "nđ", "status": "bogus"}]),
        lambda payload: payload.update(blocking_ambiguities=[{"description": "x", "essential": "yes"}]),
        lambda payload: payload.update(preprocessing_status="bogus"),
        lambda payload: payload.update(unexpected_key="x"),
    ],
)
def test_persisted_semantic_payload_rejects_incompatible_shapes(mutate) -> None:
    payload = to_persisted_dict(legacy_preprocessing())
    mutate(payload)

    with pytest.raises(SemanticAdapterError):
        draft_from_persisted_semantic(payload)


def test_semantic_adapter_never_construct_bypasses_validation() -> None:
    import app.services.agents.v2.adapters.semantic as semantic_adapter

    tree = ast.parse(Path(semantic_adapter.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else ""
            )
            assert name != "model_construct"
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                # The legacy construct-bypassing loader must not be reused.
                assert alias.name != "from_persisted_dict"


def test_legacy_deferred_reference_maps_to_unresolved_not_a_new_status() -> None:
    draft = draft_from_preprocessing(
        legacy_preprocessing(
            resolution_status="deferred", document_handle=None, candidates=[]
        )
    )
    assert draft.document_refs[0].resolution_status == "unresolved"


def test_legacy_inconsistent_resolved_reference_without_handle_is_rejected() -> None:
    with pytest.raises(SemanticAdapterError):
        draft_from_preprocessing(legacy_preprocessing(document_handle=None))


# ---------------------------------------------------------------------------
# Draft -> Binding -> Finalizer flow and immutable revision lookup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_revision_lookup_goes_through_document_views(monkeypatch) -> None:
    calls: list[tuple[UUID, UUID]] = []

    async def fake_current(db, document_id, workspace_id, *, require_vectors=False):
        calls.append((document_id, workspace_id))
        return revision_identity()

    monkeypatch.setattr(
        document_views, "load_current_revision_identity_for_workspace", fake_current
    )

    resolution = await resolve_document_binding(
        object(), resolved_reference(), workspace_id=WORKSPACE_ID
    )

    assert calls == [(DOCUMENT_ID, WORKSPACE_ID)]
    assert resolution.binding == ScopedDocument(
        binding_id="b_r1",
        document_id=DOCUMENT_ID,
        document_revision=str(CURRENT_REVISION_ID),
        role="target",
    )


@pytest.mark.asyncio
async def test_explicit_pinned_revision_uses_the_workspace_guard(monkeypatch) -> None:
    calls: list[tuple[UUID, UUID]] = []

    async def fake_workspace(db, revision_id, workspace_id, *, require_vectors=False):
        calls.append((revision_id, workspace_id))
        return revision_identity(revision_id=revision_id)

    monkeypatch.setattr(
        document_views, "load_revision_identity_for_workspace", fake_workspace
    )

    reference = resolved_reference(
        revision_requirement=PinnedRevisionRequirement(
            kind="pinned", document_revision=str(PINNED_REVISION_ID)
        )
    )
    resolution = await resolve_document_binding(
        object(), reference, workspace_id=WORKSPACE_ID
    )

    assert calls == [(PINNED_REVISION_ID, WORKSPACE_ID)]
    assert resolution.binding.document_revision == str(PINNED_REVISION_ID)


@pytest.mark.asyncio
async def test_pinned_revision_of_the_resolved_document_is_bound(monkeypatch) -> None:
    async def fake_workspace(db, revision_id, workspace_id, *, require_vectors=False):
        return revision_identity(revision_id=revision_id, document_id=DOCUMENT_ID)

    monkeypatch.setattr(
        document_views, "load_revision_identity_for_workspace", fake_workspace
    )

    reference = resolved_reference(
        revision_requirement=PinnedRevisionRequirement(
            kind="pinned", document_revision=str(PINNED_REVISION_ID)
        )
    )
    resolution = await resolve_document_binding(
        object(), reference, workspace_id=WORKSPACE_ID
    )

    assert resolution.binding == ScopedDocument(
        binding_id="b_r1",
        document_id=DOCUMENT_ID,
        document_revision=str(PINNED_REVISION_ID),
        role="target",
    )


@pytest.mark.asyncio
async def test_pinned_revision_belonging_to_another_document_is_rejected(monkeypatch) -> None:
    async def fake_workspace(db, revision_id, workspace_id, *, require_vectors=False):
        return revision_identity(
            revision_id=revision_id, document_id=OTHER_DOCUMENT_ID
        )

    monkeypatch.setattr(
        document_views, "load_revision_identity_for_workspace", fake_workspace
    )

    reference = resolved_reference(
        revision_requirement=PinnedRevisionRequirement(
            kind="pinned", document_revision=str(PINNED_REVISION_ID)
        )
    )
    with pytest.raises(DocumentAdapterError):
        await resolve_document_binding(
            object(), reference, workspace_id=WORKSPACE_ID
        )


@pytest.mark.asyncio
async def test_unresolved_reference_binds_nothing() -> None:
    reference = DocumentReference(
        ref_id="r1",
        original_span=REF_SPAN,
        normalized_reference="Nghị định 12/2020",
        requested_role="target",
        resolution_status="not_found",
        resolved_document_id=None,
    )
    resolution = await resolve_document_binding(
        object(), reference, workspace_id=WORKSPACE_ID
    )
    assert resolution.reference is reference
    assert resolution.binding is None


@pytest.mark.asyncio
async def test_resolved_reference_without_a_role_is_rejected() -> None:
    reference = DocumentReference(
        ref_id="r1",
        original_span=REF_SPAN,
        normalized_reference="Nghị định 12/2020",
        requested_role=None,
        resolution_status="resolved",
        resolved_document_id=DOCUMENT_ID,
    )
    with pytest.raises(DocumentAdapterError):
        await resolve_document_binding(object(), reference, workspace_id=WORKSPACE_ID)


@pytest.mark.asyncio
async def test_legacy_document_without_a_current_revision_is_not_bound(monkeypatch) -> None:
    async def no_current(db, document_id, workspace_id, *, require_vectors=False):
        return None

    monkeypatch.setattr(
        document_views, "load_current_revision_identity_for_workspace", no_current
    )

    with pytest.raises(document_views.RevisionNotReady):
        await resolve_document_binding(
            object(), resolved_reference(), workspace_id=WORKSPACE_ID
        )


@pytest.mark.asyncio
async def test_current_requirement_binding_records_exactly_one_relation(monkeypatch) -> None:
    async def fake_current(db, document_id, workspace_id, *, require_vectors=False):
        return revision_identity()

    monkeypatch.setattr(
        document_views, "load_current_revision_identity_for_workspace", fake_current
    )

    reference = resolved_reference(
        revision_requirement=CurrentRevisionRequirement(kind="current")
    )
    resolved = await resolve_document_bindings(
        object(), (reference,), workspace_id=WORKSPACE_ID
    )
    assert isinstance(resolved, ResolvedDocumentBindings)
    assert [r.binding_id for r in resolved.binding_set.revision_requirement_refs] == ["b_r1"]
    assert resolved.binding_set.revision_requirement_refs[0].ref_id == "r1"


@pytest.mark.asyncio
async def test_ordinary_reference_binding_has_no_revision_relation(monkeypatch) -> None:
    async def fake_current(db, document_id, workspace_id, *, require_vectors=False):
        return revision_identity()

    monkeypatch.setattr(
        document_views, "load_current_revision_identity_for_workspace", fake_current
    )

    resolved = await resolve_document_bindings(
        object(), (resolved_reference(),), workspace_id=WORKSPACE_ID
    )
    assert resolved.binding_set.revision_requirement_refs == ()


@pytest.mark.asyncio
async def test_draft_binding_finalizer_flow(monkeypatch) -> None:
    async def fake_current(db, document_id, workspace_id, *, require_vectors=False):
        return revision_identity()

    monkeypatch.setattr(
        document_views, "load_current_revision_identity_for_workspace", fake_current
    )

    draft = draft_from_preprocessing(legacy_preprocessing())
    resolved = await resolve_document_bindings(
        object(),
        draft.document_refs,
        workspace_id=WORKSPACE_ID,
        default_role="target",
    )
    finalized = finalize_semantics(draft, resolved)

    assert isinstance(finalized.snapshot, SemanticSnapshot)
    assert finalized.snapshot.contract_version == "2.0"
    assert finalized.snapshot.semantic.document_refs[0].resolution_status == "resolved"
    assert finalized.snapshot.semantic.document_refs[0].resolved_document_id == DOCUMENT_ID
    assert isinstance(finalized.bindings, DocumentBindingSet)
    assert len(finalized.bindings.bindings) == 1
    assert finalized.bindings.bindings[0].binding_id == "b_r1"
    validate_binding_set(finalized.bindings, finalized.snapshot.semantic)


def test_document_adapter_never_reads_mutable_document_artifacts() -> None:
    import app.services.agents.v2.adapters.document as document_adapter

    forbidden = {"markdown_s3_key", "chunk_count", "raw_chunks_json"}
    tree = ast.parse(Path(document_adapter.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden
        if isinstance(node, ast.Name):
            assert node.id != "Document"


# ---------------------------------------------------------------------------
# Conversation adapter
# ---------------------------------------------------------------------------


def legacy_exchange(
    *,
    index: int,
    summary: str,
    assistant_message_id: str | None,
    key_entities: list[str] | None,
    user_message_id: str = "m-user",
) -> SimpleNamespace:
    return SimpleNamespace(
        exchange_index=index,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        summary=summary,
        key_entities=key_entities,
    )


def test_conversation_context_translates_legacy_messages_and_summaries() -> None:
    messages = [
        SimpleNamespace(role="user", content="Câu hỏi một"),
        SimpleNamespace(role="assistant", content="Trả lời một"),
    ]
    summaries = [
        legacy_exchange(
            index=1,
            summary="Tóm tắt một",
            assistant_message_id="msg-1",
            key_entities=["Nghị định 12/2020", "Điều 5"],
        )
    ]
    context = context_from_legacy(messages=messages, exchange_summaries=summaries)

    assert isinstance(context, ConversationContext)
    assert [turn.content for turn in context.recent_turns] == ["Câu hỏi một", "Trả lời một"]
    assert context.summary == "Tóm tắt một"
    assert [entity.label for entity in context.active_entities] == [
        "Nghị định 12/2020",
        "Điều 5",
    ]
    # Phase 4C (Task 8): entities carry typed kinds and focus derives
    # from the validated outcomes (most recent entity).
    assert [entity.kind for entity in context.active_entities] == [
        "document",
        "section",
    ]
    assert context.last_focus is not None
    assert context.last_focus.label == "Điều 5"
    assert context.last_focus.kind == "section"


def test_conversation_context_truncates_to_the_recent_window() -> None:
    messages = [SimpleNamespace(role="user", content=f"câu {i}") for i in range(5)]
    context = context_from_legacy(messages=messages, max_recent_turns=2)
    assert [turn.content for turn in context.recent_turns] == ["câu 3", "câu 4"]


def test_conversation_adapter_rejects_an_unknown_legacy_role() -> None:
    messages = [SimpleNamespace(role="tool", content="payload")]
    with pytest.raises(ConversationAdapterError):
        context_from_legacy(messages=messages)


def test_conversation_snapshot_carries_monotonic_summary_and_built_through() -> None:
    summaries = [
        legacy_exchange(
            index=2,
            summary="Tóm tắt hai",
            assistant_message_id="msg-2",
            key_entities=None,
        ),
        legacy_exchange(
            index=1,
            summary="Tóm tắt một",
            assistant_message_id="msg-1",
            key_entities=None,
        ),
    ]
    snapshot = snapshot_from_legacy(
        thread_id="thread-1",
        messages=[SimpleNamespace(role="user", content="câu hỏi")],
        exchange_summaries=summaries,
    )

    assert isinstance(snapshot, ConversationSnapshot)
    assert snapshot.thread_id == "thread-1"
    assert snapshot.summary_version == 2
    assert snapshot.built_through_message_id == "msg-2"
    validate_conversation_snapshot(snapshot)


def test_conversation_snapshot_without_exchanges_starts_at_zero() -> None:
    snapshot = snapshot_from_legacy(
        thread_id="thread-1",
        messages=[SimpleNamespace(role="user", content="câu hỏi")],
    )
    assert snapshot.summary_version == 0
    assert snapshot.built_through_message_id is None


# ---------------------------------------------------------------------------
# Deep-research adapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("legacy_status", "agent_status"),
    [
        ("ok", "success"),
        ("partial", "partial"),
        ("missing", "not_found"),
        ("ambiguous", "needs_input"),
        ("error", "error"),
    ],
)
def test_legacy_task_statuses_map_to_agent_statuses(
    legacy_status: str, agent_status: str
) -> None:
    assert agent_status_from_legacy(legacy_status) == agent_status


def test_unknown_legacy_task_status_is_rejected() -> None:
    with pytest.raises(DeepResearchAdapterError):
        agent_status_from_legacy("finished")


def test_legacy_task_result_translates_to_a_typed_agent_result() -> None:
    legacy = LegacyTaskResult(task_id="t1", status="ok", evidence_ids=["e1"])

    result = agent_result_from_legacy(legacy)

    assert isinstance(result, AgentResult)
    assert result.contract_version == "2.0"
    assert result.task_id == "t1"
    assert result.status == "success"
    assert result.data is None
    assert result.error is None


def test_legacy_error_result_carries_a_typed_agent_error() -> None:
    legacy = LegacyTaskResult(task_id="t1", status="error", error_detail="nguồn lỗi")

    result = agent_result_from_legacy(legacy)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "INTERNAL_ERROR"
    assert result.error.message == "nguồn lỗi"


# ---------------------------------------------------------------------------
# Capability protocol and ACL-filtered registry (spec §11, §16)
# ---------------------------------------------------------------------------


def test_capability_protocol_reuses_the_frozen_contracts() -> None:
    import app.services.agents.v2.capabilities as capabilities_package

    assert capabilities_package.CapabilityDescriptor is capability_contracts.CapabilityDescriptor
    assert capabilities_package.CapabilityRuntimeContext is capability_contracts.CapabilityRuntimeContext
    assert set(inspect.signature(Capability.execute).parameters) == {"self", "request", "runtime"}
    assert inspect.iscoroutinefunction(Capability.execute)
    assert isinstance(FakeCapability("document.read"), Capability)


def test_capability_registry_intersects_runtime_permissions() -> None:
    runtime = runtime_context(allowed=frozenset({"document.read", "people.lookup"}))
    registry = build_capability_registry(
        [
            registration("document.read"),
            registration("document.search"),
            registration("people.lookup", domain="people"),
        ],
        runtime,
    )

    assert [d.name for d in registry.catalog()] == ["document.read", "people.lookup"]
    assert registry.get("document.read").descriptor.name == "document.read"


def test_unauthorized_capability_is_denied_at_execution() -> None:
    runtime = runtime_context(allowed=frozenset({"document.read"}))
    registry = build_capability_registry(
        [registration("document.read"), registration("document.search")], runtime
    )

    with pytest.raises(CapabilityDenied):
        registry.get("document.search")


def test_unknown_capability_is_not_registered() -> None:
    registry = build_capability_registry([registration("document.read")], runtime_context())
    with pytest.raises(CapabilityNotRegistered):
        registry.get("knowledge_graph.query")


def test_feature_flag_gates_capability_catalog_and_execution() -> None:
    registry = build_capability_registry(
        [registration("document.read"), registration("document.search", feature_flag="beta")],
        runtime_context(),
        active_feature_flags=frozenset(),
    )
    assert [d.name for d in registry.catalog()] == ["document.read"]
    with pytest.raises(CapabilityUnavailable):
        registry.get("document.search")


def test_service_availability_gates_capability_catalog_and_execution() -> None:
    registry = build_capability_registry(
        [registration("document.read", service="embed-rerank")],
        runtime_context(),
        available_services=frozenset(),
    )
    assert registry.catalog() == ()
    with pytest.raises(CapabilityUnavailable):
        registry.get("document.read")


def test_people_capability_requires_the_runtime_people_permission() -> None:
    runtime = runtime_context(
        allowed=frozenset({"people.lookup"}), can_read_people=False
    )
    registry = build_capability_registry(
        [registration("people.lookup", domain="people")], runtime
    )

    assert registry.catalog() == ()
    with pytest.raises(CapabilityDenied):
        registry.get("people.lookup")


def test_agent_request_never_carries_workspace_or_permission_flags() -> None:
    assert set(AgentRequest.model_fields) == {
        "contract_version",
        "task_id",
        "objective",
        "input",
    }
    for forbidden in ("workspace_ids", "user_id", "can_read_people", "allowed_capabilities"):
        assert forbidden not in AgentRequest.model_fields
        assert forbidden in CapabilityRuntimeContext.model_fields


# ---------------------------------------------------------------------------
# Typed boundaries: no dict[str, Any] escape hatch
# ---------------------------------------------------------------------------


def _v2_port_modules() -> list[Path]:
    import app.services.agents.v2.adapters as adapters_package
    import app.services.agents.v2.capabilities as capabilities_package

    adapters_dir = Path(adapters_package.__file__).parent
    return [Path(capabilities_package.__file__), *sorted(adapters_dir.glob("*.py"))]


def test_v2_port_modules_do_not_use_dict_or_any_annotations() -> None:
    for path in _v2_port_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "Any":
                raise AssertionError(f"{path} references typing.Any")
        # The capability package builds an internal ``dict[str, Capability]``
        # accumulator; the adapters own the typed boundary and use none.
        if path.parent.name == "capabilities":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                base = node.value
                name = (
                    base.id
                    if isinstance(base, ast.Name)
                    else base.attr
                    if isinstance(base, ast.Attribute)
                    else ""
                )
                if name in {"dict", "Dict"}:
                    raise AssertionError(f"{path} annotates a dict")


def test_adapter_outputs_are_typed_models_not_dictionaries() -> None:
    draft = draft_from_preprocessing(legacy_preprocessing())
    context = finalize_semantic_context(draft)
    snapshot = snapshot_from_legacy(
        thread_id="thread-1", messages=[SimpleNamespace(role="user", content="x")]
    )
    result = agent_result_from_legacy(LegacyTaskResult(task_id="t1", status="ok"))

    for value in (draft, context, snapshot, result):
        assert isinstance(value, BaseModel)
        assert not isinstance(value, dict)


# ---------------------------------------------------------------------------
# Phase 2 Task 2 (appended) — shared atomic capabilities, no registry rewrite
# ---------------------------------------------------------------------------


def _task2_evidence_builder():
    from uuid import uuid4

    from app.services.agents.v2.contracts.evidence import EvidenceUseRef

    class _Evidence:
        async def persist_use(self, **kwargs) -> EvidenceUseRef:
            return EvidenceUseRef(use_id=uuid4())

    return _Evidence()


def _task2_resolver(bindings=None):
    class _Resolver:
        def __init__(self) -> None:
            self._bindings = dict(bindings or {})

        def resolve(self, target_id: str):
            return self._bindings.get(target_id)

    return _Resolver()


def _task2_capabilities() -> list:
    from uuid import UUID as _UUID

    from app.services.agents.v2.capabilities import (
        AbbreviationCapability,
        DocumentReadCapability,
        DocumentSearchCapability,
        KnowledgeGraphCapability,
        MemoryCapability,
        PeopleCapability,
        SectionReadCapability,
    )

    class _People:
        async def lookup(self, query: str):
            return {"record_id": "p-1", "name": "Nguyen Van A"}

    class _Search:
        async def search(self, query: str, person_identifier, workspace_ids):
            return ()

    class _Reader:
        async def read(self, binding, locator):
            from app.services.agents.v2.capabilities import LocatedContent

            return LocatedContent(
                outcome="read", observed_locator=locator, content="text"
            )

    class _SectionReader:
        async def read_section(self, binding, locator):
            from app.services.agents.v2.capabilities import LocatedContent

            return LocatedContent(
                outcome="read", observed_locator=locator, content="text"
            )

    class _Kg:
        async def query(self, query: str):
            return ()

    class _Memory:
        async def lookup(self, query: str):
            return ()

    class _Abbreviations:
        def resolve(self, token: str):
            return None

    evidence = _task2_evidence_builder()
    resolver = _task2_resolver()
    return [
        PeopleCapability(
            service=_People(), evidence=evidence, required_fields=("name",)
        ),
        DocumentSearchCapability(service=_Search()),
        DocumentReadCapability(
            reader=_Reader(), evidence=evidence, resolver=resolver
        ),
        SectionReadCapability(
            reader=_SectionReader(), evidence=evidence, resolver=resolver
        ),
        KnowledgeGraphCapability(client=_Kg(), evidence=evidence),
        MemoryCapability(store=_Memory(), evidence=evidence),
        AbbreviationCapability(service=_Abbreviations()),
    ]


def test_task2_real_capabilities_satisfy_the_capability_protocol() -> None:
    from app.services.agents.v2.capabilities import (
        EvidenceBuilder,
        PinnedTargetResolver,
    )

    for capability in _task2_capabilities():
        assert isinstance(capability, Capability)
        assert isinstance(_task2_evidence_builder(), EvidenceBuilder)
        assert isinstance(_task2_resolver(), PinnedTargetResolver)
        assert set(inspect.signature(type(capability).execute).parameters) == {
            "self",
            "request",
            "runtime",
        }
        assert inspect.iscoroutinefunction(capability.execute)


def test_task2_capability_modules_reuse_the_frozen_contracts() -> None:
    import app.services.agents.v2.capabilities as capabilities_package

    assert capabilities_package.CapabilityDescriptor is capability_contracts.CapabilityDescriptor
    assert capabilities_package.CapabilityInput is capability_contracts.CapabilityInput
    assert capabilities_package.CapabilityOutput is capability_contracts.CapabilityOutput
    assert capabilities_package.CapabilityRuntimeContext is capability_contracts.CapabilityRuntimeContext
    expected_names = {
        "people.lookup",
        "document.search",
        "document.read",
        "section.read",
        "knowledge_graph.query",
        "memory.lookup",
        "abbreviation.resolve",
    }
    assert {c.descriptor.name for c in _task2_capabilities()} == expected_names


def test_task2_capability_modules_have_no_dict_or_any_annotations() -> None:
    import app.services.agents.v2.capabilities as capabilities_package

    package_dir = Path(capabilities_package.__file__).parent
    for path in sorted(package_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "Any":
                raise AssertionError(f"{path.name} references typing.Any")
            if isinstance(node, ast.Subscript):
                base = node.value
                name = (
                    base.id
                    if isinstance(base, ast.Name)
                    else base.attr
                    if isinstance(base, ast.Attribute)
                    else ""
                )
                if name in {"dict", "Dict"}:
                    raise AssertionError(f"{path.name} annotates a dict")


def test_task2_registry_serves_the_real_capabilities() -> None:
    capabilities = _task2_capabilities()
    by_name = {c.descriptor.name: c for c in capabilities}
    registry = build_capability_registry(
        [CapabilityRegistration(capability=c) for c in capabilities],
        runtime_context(
            allowed=frozenset(c.descriptor.name for c in capabilities)
        ),
        active_feature_flags=frozenset(),
        available_services=frozenset(),
    )
    assert registry.get("people.lookup") is by_name["people.lookup"]
    assert registry.get("abbreviation.resolve") is by_name["abbreviation.resolve"]


# ---------------------------------------------------------------------------
# P0 Task 3 fix round 1 (I2): workspace/tombstone-scoped current binding
# ---------------------------------------------------------------------------


class _StubResult:
    def __init__(self, row) -> None:
        self._row = row

    def first(self):
        return self._row


class _StubDB:
    """Database-free stand-in for the AsyncSession surface document_views uses."""

    def __init__(self, *, current_row=None, revision=None, build=None) -> None:
        self._current_row = current_row
        self._revision = revision
        self._build = build

    async def execute(self, stmt):
        return _StubResult(self._current_row)

    async def get(self, model, pk):
        return self._revision

    async def scalar(self, stmt):
        return self._build


def _stub_revision(
    *,
    status: str = "published",
    revision_id: UUID = CURRENT_REVISION_ID,
    document_id: UUID = DOCUMENT_ID,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        revision_id=revision_id,
        document_id=document_id,
        generation=1,
    )


def _stub_build() -> SimpleNamespace:
    return SimpleNamespace(
        build_profile="FULL",
        markdown_artifact_key="markdown.md",
        structure_artifact_key="structure.json",
        embedding_namespace=None,
        embedding_model_hash=None,
        embedding_dimension=None,
        vector_artifact_version=None,
    )


@pytest.mark.asyncio
async def test_ordinary_reference_binding_is_workspace_scoped(monkeypatch) -> None:
    calls: list[tuple[UUID, UUID]] = []

    async def fake_workspace_current(db, document_id, workspace_id, *, require_vectors=False):
        calls.append((document_id, workspace_id))
        return revision_identity()

    monkeypatch.setattr(
        document_views, "load_current_revision_identity_for_workspace", fake_workspace_current
    )

    resolution = await resolve_document_binding(
        object(), resolved_reference(), workspace_id=WORKSPACE_ID
    )

    assert calls == [(DOCUMENT_ID, WORKSPACE_ID)]
    assert resolution.binding == ScopedDocument(
        binding_id="b_r1",
        document_id=DOCUMENT_ID,
        document_revision=str(CURRENT_REVISION_ID),
        role="target",
    )


@pytest.mark.asyncio
async def test_ordinary_reference_in_foreign_workspace_fails_closed(monkeypatch) -> None:
    async def fake_workspace_current(db, document_id, workspace_id, *, require_vectors=False):
        raise document_views.RevisionNotReady(
            document_id,
            "document does not exist, is not owned by this workspace, or is tombstoned",
        )

    monkeypatch.setattr(
        document_views, "load_current_revision_identity_for_workspace", fake_workspace_current
    )

    with pytest.raises(document_views.RevisionNotReady):
        await resolve_document_binding(
            object(), resolved_reference(), workspace_id=WORKSPACE_ID
        )


@pytest.mark.asyncio
async def test_workspace_scoped_current_lookup_rejects_unknown_workspace_or_tombstone() -> None:
    db = _StubDB(current_row=None)
    with pytest.raises(document_views.RevisionNotReady):
        await document_views.load_current_revision_identity_for_workspace(
            db, DOCUMENT_ID, WORKSPACE_ID
        )


@pytest.mark.asyncio
async def test_workspace_scoped_current_lookup_returns_none_for_legacy() -> None:
    db = _StubDB(current_row=(None,))
    assert (
        await document_views.load_current_revision_identity_for_workspace(
            db, DOCUMENT_ID, WORKSPACE_ID
        )
        is None
    )


@pytest.mark.asyncio
async def test_workspace_scoped_current_lookup_resolves_published_identity() -> None:
    db = _StubDB(
        current_row=(CURRENT_REVISION_ID,),
        revision=_stub_revision(),
        build=_stub_build(),
    )
    identity = await document_views.load_current_revision_identity_for_workspace(
        db, DOCUMENT_ID, WORKSPACE_ID
    )
    assert identity.revision_id == CURRENT_REVISION_ID
    assert identity.document_id == DOCUMENT_ID

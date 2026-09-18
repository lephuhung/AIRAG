"""Task 5 — clarification interrupt/resume node.

Interrupt/resume coverage for the v2 clarify path: stable candidate UUID/order,
question-specific ref IDs, expiry, invalid selection, selected candidate no
longer authorized, current ACL replacement, raw reply loaded by ChatMessage.id,
checkpoint without runtime secrets, and the resolved flow restarting at the
Binding Resolver.

Service fakes are wired onto a real ``RuntimeServices`` — never ad-hoc builder
kwargs. ``chat_messages``/``authorization`` are the Phase-2 request-scoped
services on that container (``contracts/state.py``, added in T3).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from langgraph.runtime import Runtime

from app.services.agents.v2.contracts.base import CONTRACT_VERSION
from app.services.agents.v2.contracts.binding import DocumentBindingSet
from app.services.agents.v2.contracts.capability import CapabilityRuntimeContext
from app.services.agents.v2.contracts.clarification import (
    ClarificationRequest,
    ClarificationResolution,
)
from app.services.agents.v2.contracts.conversation import ConversationContext
from app.services.agents.v2.contracts.request import RequestContext
from app.services.agents.v2.contracts.routing import QueryAnalysis, RouteDecision
from app.services.agents.v2.contracts.semantic import (
    BlockingAmbiguity,
    DocumentReference,
    SemanticContext,
)
from app.services.agents.v2.contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    RuntimeServices,
    SupervisorV2State,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_supervisor_state,
)
from app.services.agents.v2.nodes.clarification import (
    ClarificationError,
    ClarificationExpired,
    ClarificationInvalidSelection,
    ClarificationUnauthorized,
    ClarificationUnsatisfiable,
    build_clarification,
    clarify_node,
    interrupt_for_clarification,
    parse_clarification_resolution,
    resume_clarification,
)

USER_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
WORKSPACE_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
NEW_WORKSPACE_ID = UUID("99999999-9999-9999-9999-999999999999")
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")
RESOLVED_DOCUMENT_ID = UUID("33333333-3333-3333-3333-333333333333")


# ---------------------------------------------------------------------------
# Fakes (wired onto a real RuntimeServices)
# ---------------------------------------------------------------------------


class FakeChatMessages:
    """Stand-in for the chat persistence service (raw reply stays there)."""

    def __init__(self, contents: dict[UUID, str] | None = None) -> None:
        self.contents = dict(contents or {})
        self.calls: list[UUID] = []

    async def get_user_message(self, message_id: UUID) -> SimpleNamespace:
        self.calls.append(message_id)
        return SimpleNamespace(id=message_id, content=self.contents[message_id])


class FakeAuthorization:
    """Stand-in for the document authorization service (current ACL).

    ``denied`` is a static blocklist. ``grants`` maps a workspace id to the
    document ids authorized there, so a test can prove the resume-time
    runtime (not the build-time one) decides the outcome.
    """

    def __init__(
        self,
        denied: set[UUID] | None = None,
        grants: dict[UUID, set[UUID]] | None = None,
        service_error: Exception | None = None,
    ) -> None:
        self.denied = set(denied or set())
        self.grants = grants
        self.service_error = service_error
        self.calls: list[tuple[UUID, CapabilityRuntimeContext]] = []

    async def require_document(
        self, document_id: UUID, capability_runtime: CapabilityRuntimeContext
    ) -> None:
        self.calls.append((document_id, capability_runtime))
        if self.service_error is not None:
            raise self.service_error
        if self.grants is not None:
            allowed = any(
                document_id in self.grants.get(workspace, set())
                for workspace in capability_runtime.workspace_ids
            )
            if not allowed:
                raise PermissionError(
                    f"document {document_id} is not authorized in this workspace"
                )
            return
        if document_id in self.denied:
            raise PermissionError(f"document {document_id} is not authorized")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def ambiguous_ref(
    *,
    ref_id: str = "r1",
    candidates: tuple[UUID, ...] = (DOCUMENT_ID, OTHER_DOCUMENT_ID),
) -> DocumentReference:
    return DocumentReference(
        ref_id=ref_id,
        original_span="nghị định 12",
        normalized_reference="nghị định 12",
        requested_role="target",
        revision_requirement=None,
        resolution_status="ambiguous",
        resolved_document_id=None,
        candidate_document_ids=candidates,
    )


def resolved_ref(
    *,
    ref_id: str = "r0",
    document_id: UUID = RESOLVED_DOCUMENT_ID,
) -> DocumentReference:
    return DocumentReference(
        ref_id=ref_id,
        original_span="Nghị định 15/2021",
        normalized_reference="nghị định 15/2021",
        requested_role="target",
        revision_requirement=None,
        resolution_status="resolved",
        resolved_document_id=document_id,
    )


def blocking_semantic() -> SemanticContext:
    return SemanticContext(
        contextualized_query="Xem nghị định 12",
        normalized_query="xem nghị định 12",
        abbreviations=(),
        coreferences=(),
        document_refs=(resolved_ref(), ambiguous_ref()),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(
            BlockingAmbiguity(
                ambiguity_id="r1", description="Hai văn bản cùng số 12."
            ),
        ),
    )


def make_capability_runtime(
    *, workspace_id: UUID = WORKSPACE_ID
) -> CapabilityRuntimeContext:
    return CapabilityRuntimeContext(
        request_id="req-1",
        run_id="run-1",
        user_id=USER_ID,
        workspace_ids=(workspace_id,),
        can_read_people=True,
        allowed_capabilities=frozenset({"document.read"}),
        deadline_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


def make_graph_context(
    *,
    chat: Any = None,
    auth: Any = None,
    workspace_id: UUID = WORKSPACE_ID,
) -> GraphRuntimeContext:
    return GraphRuntimeContext(
        capability_runtime=make_capability_runtime(workspace_id=workspace_id),
        services=RuntimeServices(chat_messages=chat, authorization=auth),
    )


def live_request_for(semantic: SemanticContext) -> ClarificationRequest:
    # Wall-clock-relative so the suite does not expire on a fixed date.
    return build_clarification(semantic, now=datetime.now(UTC))


def unresolved_ref(*, ref_id: str = "r9") -> DocumentReference:
    return DocumentReference(
        ref_id=ref_id,
        original_span="nghị định 99",
        normalized_reference="nghị định 99",
        requested_role="target",
        revision_requirement=None,
        resolution_status="unresolved",
        resolved_document_id=None,
    )


def unresolved_semantic() -> SemanticContext:
    return SemanticContext(
        contextualized_query="Xem nghị định 99",
        normalized_query="xem nghị định 99",
        abbreviations=(),
        coreferences=(),
        document_refs=(unresolved_ref(),),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(
            BlockingAmbiguity(
                ambiguity_id="r9", description="Không tìm thấy nghị định 99."
            ),
        ),
    )


def stale_ambiguity_semantic() -> SemanticContext:
    """A blocking ambiguity naming an already-resolved ref (M1/M2 case)."""
    return SemanticContext(
        contextualized_query="Xem Nghị định 15/2021",
        normalized_query="xem nghị định 15/2021",
        abbreviations=(),
        coreferences=(),
        document_refs=(resolved_ref(),),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(
            BlockingAmbiguity(
                ambiguity_id="r0", description="Ý nào của nghị định được hỏi?"
            ),
        ),
    )


def resume_setup(
    *,
    reply: str,
    denied: set[UUID] | None = None,
    grants: dict[UUID, set[UUID]] | None = None,
    workspace_id: UUID = WORKSPACE_ID,
) -> SimpleNamespace:
    message_id = uuid4()
    chat = FakeChatMessages({message_id: reply})
    auth = FakeAuthorization(denied=denied, grants=grants)
    ctx = make_graph_context(chat=chat, auth=auth, workspace_id=workspace_id)
    return SimpleNamespace(
        message_id=message_id, chat=chat, auth=auth, ctx=ctx
    )


def make_state(
    *,
    semantic: SemanticContext,
    clarification: ClarificationRequest | None = None,
    route: str = "clarify",
) -> SupervisorV2State:
    return SupervisorV2State(
        contract_version="2.0",
        request=RequestContext(
            contract_version="2.0",
            request_id="req-1",
            thread_id="thread-1",
            original_query="Xem nghị định 12",
            known_documents=(),
        ),
        conversation=ConversationContext(
            summary="",
            active_entities=(),
            last_focus=None,
            recent_turns=(),
        ),
        semantic=semantic,
        bindings=DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        query_analysis=QueryAnalysis(
            work_type="retrieve",
            domains=("document",),
            dependency_hints=(),
        ),
        route_decision=RouteDecision(route=route, reason_code="essential_ambiguity"),  # type: ignore[arg-type]
        execution=ExecutionState(plan=None, task_results=(), evidence_evaluation=None),
        clarification=clarification,
        final_response=None,
        synthesis=None,
        checkpoint_schema_revision=2,
        discovery_need=None,
        discovery=None,
        document_selection_clarification=None,
        research_target_selection=None,
    )


# ---------------------------------------------------------------------------
# build_clarification: stable identity, question-specific refs
# ---------------------------------------------------------------------------


def test_candidate_identity_and_order_are_stable() -> None:
    semantic = blocking_semantic()
    first = build_clarification(
        semantic, now=datetime(2030, 5, 1, 12, 0, tzinfo=UTC)
    )
    second = build_clarification(
        semantic, now=datetime(2030, 5, 1, 12, 7, tzinfo=UTC)
    )
    # Same question, rebuilt later: same request id, same candidate UUIDs/order.
    assert first.clarification_id == second.clarification_id
    assert [c.candidate_id for c in first.candidates] == [
        c.candidate_id for c in second.candidates
    ]
    assert [c.ordinal for c in first.candidates] == [0, 1]
    assert [c.document_id for c in first.candidates] == sorted(
        [c.document_id for c in first.candidates], key=str
    )
    # ... but the per-request deadline is frozen at build time, not shared.
    assert first.expires_at != second.expires_at
    assert second.expires_at - first.expires_at == timedelta(minutes=7)


def test_unresolved_ref_ids_are_question_specific() -> None:
    request = build_clarification(blocking_semantic())
    # The resolved ref is not part of this question; the ambiguous one is.
    assert request.unresolved_ref_ids == ("r1",)
    assert request.reason == "required_document_ambiguous"
    assert all(c.ref_id == "r1" for c in request.candidates)
    assert len(request.candidates) == 2


def test_build_is_checkpoint_safe_and_valid() -> None:
    semantic = blocking_semantic()
    request = build_clarification(semantic)
    dumped = request.model_dump(mode="json")
    # No runtime secrets or trusted identity may enter the checkpoint.
    assert set(dumped) == {
        "kind",
        "contract_version",
        "clarification_id",
        "reason",
        "question",
        "unresolved_ref_ids",
        "candidates",
        "expires_at",
    }
    assert "user_id" not in str(dumped)
    assert str(WORKSPACE_ID) not in str(dumped)
    # The clarify resting state passes the frozen aggregate validator.
    validate_supervisor_state(make_state(semantic=semantic, clarification=request))


@pytest.mark.asyncio
async def test_clarify_node_persists_request_for_checkpoint() -> None:
    semantic = blocking_semantic()
    state = make_state(semantic=semantic, clarification=None)
    ctx = make_graph_context()
    update = await clarify_node(state, Runtime(context=ctx))
    request = update["clarification"]
    assert isinstance(request, ClarificationRequest)
    assert request.unresolved_ref_ids == ("r1",)
    # D6: route_node returns clarify with clarification=None; clarify_node owns
    # the persisted request that lets the clarify state checkpoint.
    validate_supervisor_state(make_state(semantic=semantic, clarification=request))


# ---------------------------------------------------------------------------
# interrupt_for_clarification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_carries_checkpoint_payload() -> None:
    from typing import TypedDict

    from langgraph.graph import StateGraph

    request = live_request_for(blocking_semantic())

    class InterruptState(TypedDict, total=False):
        done: bool

    async def clarify(state: InterruptState) -> dict:
        await interrupt_for_clarification(request)
        return {"done": True}

    builder = StateGraph(InterruptState)
    builder.add_node("clarify", clarify)
    builder.set_entry_point("clarify")
    builder.set_finish_point("clarify")
    graph = builder.compile()
    # Outside a graph run interrupt() cannot deliver a payload (it raises
    # RuntimeError there); inside a run it suspends with __interrupt__.
    result = await graph.ainvoke({})
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    assert interrupts[0].value == {
        "clarification": request.model_dump(mode="json")
    }


# ---------------------------------------------------------------------------
# parse + resume: selection, expiry, membership, ACL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_restarts_resolved_flow_at_binding() -> None:
    semantic = blocking_semantic()
    request = live_request_for(semantic)
    first = request.candidates[0]
    handle = resume_setup(reply="1")
    command = await resume_clarification(handle.message_id, request, handle.ctx)
    assert not command.goto  # graph owns navigation (T6 round 2)
    resolution = ClarificationResolution.model_validate(command.resume)
    assert resolution.clarification_id == request.clarification_id
    assert resolution.selected_candidate_id == first.candidate_id
    # The raw reply is loaded by ChatMessage.id and stays authoritative there:
    # the resolution stores only the deterministic selection, no user text.
    assert handle.chat.calls == [handle.message_id]
    assert set(command.resume) == {"kind", "contract_version", "clarification_id", "selected_candidate_id"}
    # The resolution envelope carries no ChatMessage field and only an
    # offered candidate id — the raw reply never enters the checkpoint.
    assert "content" not in command.resume
    assert command.resume["selected_candidate_id"] in {
        c.candidate_id for c in request.candidates
    }


@pytest.mark.asyncio
async def test_resume_accepts_candidate_id_reply() -> None:
    request = live_request_for(blocking_semantic())
    target = request.candidates[1]
    handle = resume_setup(reply=target.candidate_id)
    command = await resume_clarification(handle.message_id, request, handle.ctx)
    assert ClarificationResolution.model_validate(command.resume).selected_candidate_id == (
        target.candidate_id
    )
    assert handle.auth.calls == [(target.document_id, handle.ctx.capability_runtime)]


@pytest.mark.asyncio
async def test_resume_rejects_expired_request() -> None:
    semantic = blocking_semantic()
    request = build_clarification(
        semantic, now=datetime.now(UTC) - timedelta(hours=2)
    )
    assert request.expires_at < datetime.now(UTC)
    handle = resume_setup(reply="1")
    with pytest.raises(ClarificationExpired):
        await resume_clarification(handle.message_id, request, handle.ctx)
    # Expired: no ACL call, no Command.
    assert handle.auth.calls == []


@pytest.mark.asyncio
async def test_resume_rejects_invalid_selection() -> None:
    request = live_request_for(blocking_semantic())
    handle = resume_setup(reply="bản thứ mười không tồn tại")
    with pytest.raises(ClarificationInvalidSelection):
        await resume_clarification(handle.message_id, request, handle.ctx)
    assert handle.auth.calls == []


@pytest.mark.asyncio
async def test_resume_rejects_out_of_range_ordinal() -> None:
    request = live_request_for(blocking_semantic())
    handle = resume_setup(reply="9")
    with pytest.raises(ClarificationInvalidSelection):
        await resume_clarification(handle.message_id, request, handle.ctx)


def test_resume_rejects_forged_candidate() -> None:
    request = live_request_for(blocking_semantic())
    forged = ClarificationResolution(
        contract_version="2.0",
        clarification_id=request.clarification_id,
        selected_candidate_id="cand-forged",
    )
    with pytest.raises(ContractValidationError):
        # Membership is enforced even when parsing is bypassed.
        from app.services.agents.v2.nodes.clarification import _selected_candidate

        _selected_candidate(forged, request)


@pytest.mark.asyncio
async def test_resume_rejects_no_longer_authorized_candidate() -> None:
    request = live_request_for(blocking_semantic())
    target = request.candidates[0]
    handle = resume_setup(reply="1", denied={target.document_id})
    with pytest.raises(ClarificationUnauthorized) as exc_info:
        await resume_clarification(handle.message_id, request, handle.ctx)
    assert exc_info.value.document_id == target.document_id
    assert exc_info.value.clarification_id == request.clarification_id
    assert isinstance(exc_info.value.__cause__, PermissionError)
    assert isinstance(exc_info.value, ClarificationError)


@pytest.mark.asyncio
async def test_resume_uses_current_acl_not_build_time_acl() -> None:
    """The ACL is replaced at resume: the resume-time runtime authorizes."""
    request = live_request_for(blocking_semantic())
    target = request.candidates[0]
    handle = resume_setup(reply="1", workspace_id=NEW_WORKSPACE_ID)
    command = await resume_clarification(handle.message_id, request, handle.ctx)
    assert not command.goto  # graph owns navigation (T6 round 2)
    (document_id, runtime) = handle.auth.calls[0]
    assert document_id == target.document_id
    assert runtime is handle.ctx.capability_runtime
    assert runtime.workspace_ids == (NEW_WORKSPACE_ID,)


@pytest.mark.asyncio
async def test_resume_rejects_dismissal_without_selection() -> None:
    request = live_request_for(blocking_semantic())
    assert parse_clarification_resolution("không", request).selected_candidate_id is None
    handle = resume_setup(reply="không")
    with pytest.raises(ClarificationInvalidSelection):
        await resume_clarification(handle.message_id, request, handle.ctx)


def test_resolution_carries_no_user_text() -> None:
    request = live_request_for(blocking_semantic())
    resolution = parse_clarification_resolution(
        "  LỰA CHỌN 2  ", request
    )
    assert resolution.selected_candidate_id == request.candidates[1].candidate_id
    assert CONTRACT_VERSION == resolution.contract_version


def test_resolved_ref_named_by_ambiguity_is_not_unresolved() -> None:
    # M1/M2: a resolved ref stays out of unresolved_ref_ids even when a
    # blocking ambiguity names it; the reason must not claim not_found.
    request = build_clarification(stale_ambiguity_semantic())
    assert request.unresolved_ref_ids == ()
    assert request.reason == "semantic_ambiguity"
    assert request.candidates == ()
    assert "Ý nào" in request.question


def test_candidate_order_stable_under_permuted_input() -> None:
    # M7: the sorted-document normalization is exercised, not just asserted.
    forward = build_clarification(blocking_semantic())
    reversed_semantic = SemanticContext(
        contextualized_query="Xem nghị định 12",
        normalized_query="xem nghị định 12",
        abbreviations=(),
        coreferences=(),
        document_refs=(
            resolved_ref(),
            ambiguous_ref(candidates=(OTHER_DOCUMENT_ID, DOCUMENT_ID)),
        ),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(
            BlockingAmbiguity(
                ambiguity_id="r1", description="Hai văn bản cùng số 12."
            ),
        ),
    )
    permuted = build_clarification(reversed_semantic)
    assert [c.candidate_id for c in permuted.candidates] == [
        c.candidate_id for c in forward.candidates
    ]
    assert [c.document_id for c in permuted.candidates] == [
        c.document_id for c in forward.candidates
    ]


def test_candidate_free_request_is_unsatisfiable() -> None:
    # I1: an unresolved-only question offers nothing selectable.
    request = build_clarification(unresolved_semantic())
    assert request.reason == "required_document_not_found"
    assert request.candidates == ()
    assert request.unresolved_ref_ids == ("r9",)
    with pytest.raises(ClarificationUnsatisfiable):
        parse_clarification_resolution("nghị định 99 bản mới nhất", request)
    with pytest.raises(ClarificationUnsatisfiable):
        parse_clarification_resolution("không", request)
    # M-NEW-3: unsatisfiable precedes expiry — documented, now pinned.
    expired_free = build_clarification(
        unresolved_semantic(), now=datetime.now(UTC) - timedelta(hours=2)
    )
    assert expired_free.expires_at < datetime.now(UTC)
    with pytest.raises(ClarificationUnsatisfiable):
        parse_clarification_resolution("1", expired_free)


@pytest.mark.asyncio
async def test_resume_candidate_free_request_never_resumes() -> None:
    # I1: every reply to a candidate-free request raises the distinct typed
    # error (no ACL call, no Command) so the runner can treat it as a new
    # turn instead of a bad reply.
    request = build_clarification(unresolved_semantic())
    handle = resume_setup(reply="1")
    with pytest.raises(ClarificationUnsatisfiable) as exc_info:
        await resume_clarification(handle.message_id, request, handle.ctx)
    assert exc_info.value.clarification_id == request.clarification_id
    assert isinstance(exc_info.value, ClarificationError)
    assert handle.auth.calls == []


@pytest.mark.asyncio
async def test_clarify_node_preserves_live_request() -> None:
    # I2: re-execution (LangGraph re-runs the interrupting node from the top
    # on resume) must not silently extend the per-request deadline.
    semantic = blocking_semantic()
    live = build_clarification(semantic, now=datetime.now(UTC))
    state = make_state(semantic=semantic, clarification=live)
    ctx = make_graph_context()
    first = await clarify_node(state, Runtime(context=ctx))
    assert first["clarification"] is live
    second = await clarify_node(
        make_state(semantic=semantic, clarification=first["clarification"]),
        Runtime(context=ctx),
    )
    assert second["clarification"] is live
    assert second["clarification"].expires_at == live.expires_at


@pytest.mark.asyncio
async def test_clarify_node_rebuilds_expired_or_stale_request() -> None:
    # I2: an expired (or question-mismatched) persisted request is replaced.
    semantic = blocking_semantic()
    expired = build_clarification(
        semantic, now=datetime.now(UTC) - timedelta(hours=2)
    )
    assert expired.expires_at < datetime.now(UTC)
    ctx = make_graph_context()
    rebuilt = await clarify_node(
        make_state(semantic=semantic, clarification=expired),
        Runtime(context=ctx),
    )
    assert rebuilt["clarification"].clarification_id == expired.clarification_id
    assert rebuilt["clarification"].expires_at > expired.expires_at
    other = await clarify_node(
        make_state(semantic=unresolved_semantic(), clarification=expired),
        Runtime(context=ctx),
    )
    assert other["clarification"].clarification_id != expired.clarification_id


@pytest.mark.asyncio
async def test_clarify_node_preserves_json_mapping_request() -> None:
    # I-NEW-1: the guard must accept the exact JSON shape the module itself
    # emits (interrupt payload / checkpoint mapping), not just live objects.
    semantic = blocking_semantic()
    live = build_clarification(semantic, now=datetime.now(UTC) - timedelta(minutes=5))
    payload = live.model_dump(mode="json")
    assert isinstance(payload["candidates"], list)
    assert isinstance(payload["expires_at"], str)
    ctx = make_graph_context()
    # Mypy aside: checkpoint round-trips deliver plain mappings here.
    state = make_state(semantic=semantic, clarification=payload)  # type: ignore[arg-type]
    update = await clarify_node(state, Runtime(context=ctx))
    preserved = update["clarification"]
    assert isinstance(preserved, ClarificationRequest)
    # Guard path taken: a rebuild would mint a fresh deadline minutes later.
    assert preserved.expires_at == live.expires_at
    assert preserved.clarification_id == live.clarification_id
    assert preserved.candidates == live.candidates


@pytest.mark.asyncio
async def test_clarify_node_rebuilds_on_stale_candidate_set() -> None:
    # M-NEW-1: same question identity but a changed candidate projection
    # must not carry the stale options forward.
    semantic = blocking_semantic()
    persisted = build_clarification(semantic, now=datetime.now(UTC))
    narrowed = SemanticContext(
        contextualized_query="Xem nghị định 12",
        normalized_query="xem nghị định 12",
        abbreviations=(),
        coreferences=(),
        document_refs=(resolved_ref(), ambiguous_ref(candidates=(DOCUMENT_ID,))),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(
            BlockingAmbiguity(
                ambiguity_id="r1", description="Hai văn bản cùng số 12."
            ),
        ),
    )
    ctx = make_graph_context()
    update = await clarify_node(
        make_state(semantic=narrowed, clarification=persisted),
        Runtime(context=ctx),
    )
    rebuilt = update["clarification"]
    assert [c.document_id for c in rebuilt.candidates] == [DOCUMENT_ID]


@pytest.mark.asyncio
async def test_resume_propagates_non_denial_service_error() -> None:
    # M-NEW-2: a service bug is not misclassified as an authorization denial.
    request = live_request_for(blocking_semantic())
    message_id = uuid4()
    chat = FakeChatMessages({message_id: "1"})
    auth = FakeAuthorization(service_error=RuntimeError("boom"))
    ctx = make_graph_context(chat=chat, auth=auth)
    with pytest.raises(RuntimeError, match="boom"):
        await resume_clarification(message_id, request, ctx)


@pytest.mark.asyncio
async def test_resume_acl_decision_follows_resume_workspace() -> None:
    # M8: the resume-time workspace behaviorally decides, not just travels.
    request = live_request_for(blocking_semantic())
    target = request.candidates[0]
    grants = {WORKSPACE_ID: {target.document_id}, NEW_WORKSPACE_ID: set()}
    old = resume_setup(reply="1", grants=grants, workspace_id=WORKSPACE_ID)
    command = await resume_clarification(old.message_id, request, old.ctx)
    assert not command.goto  # graph owns navigation (T6 round 2)
    new = resume_setup(reply="1", grants=grants, workspace_id=NEW_WORKSPACE_ID)
    with pytest.raises(ClarificationUnauthorized):
        await resume_clarification(new.message_id, request, new.ctx)


@pytest.mark.asyncio
async def test_resume_fails_closed_without_services() -> None:
    request = live_request_for(blocking_semantic())
    message_id = uuid4()
    with pytest.raises(ClarificationError, match="no chat_messages"):
        await resume_clarification(
            message_id, request, make_graph_context(chat=None, auth=None)
        )
    chat = FakeChatMessages({message_id: "1"})
    with pytest.raises(ClarificationError, match="no authorization"):
        await resume_clarification(
            message_id, request, make_graph_context(chat=chat, auth=None)
        )


@pytest.mark.asyncio
async def test_resume_rejects_empty_or_non_string_reply() -> None:
    request = live_request_for(blocking_semantic())
    message_id = uuid4()
    chat = FakeChatMessages({message_id: "   "})
    ctx = make_graph_context(chat=chat, auth=FakeAuthorization())
    with pytest.raises(ClarificationInvalidSelection):
        await resume_clarification(message_id, request, ctx)
    chat.contents[message_id] = None  # type: ignore[dict-item]
    with pytest.raises(ClarificationInvalidSelection):
        await resume_clarification(message_id, request, ctx)

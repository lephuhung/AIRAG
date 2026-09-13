"""
LangGraph Chat Agent — SSE Streaming Endpoint
================================================

New endpoint that uses the LangGraph StateGraph agent instead of the legacy
manual agent loop. Produces IDENTICAL SSE event format so the frontend
requires zero changes.

Route: POST /rag/chat/agent-lg/stream
       POST /rag/chat/sessions/{session_id}/stream-lg  (session-aware variant)

LangGraph supervisor is the only chat agent backend.

SSE Events emitted (via app/services/agent/streaming.py):
    status       → {"step": str, "detail": str}
    thinking     → {"text": str}
    sources      → {"sources": [...]}
    images       → {"image_refs": [...]}
    token        → {"text": str}
    complete     → {"answer": str, "sources": [...], "images": [...], ...}
    error        → {"message": str}
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_principal
from app.models.user import User
from app.models.knowledge_base import KnowledgeBase
from app.models.tenant import TenantUser
from app.schemas.rag import ChatRequest
from app.prompts.chat import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT
from app.services.abbreviation_service import AbbreviationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag/chat", tags=["chat_langgraph"])


# ---------------------------------------------------------------------------
# Helpers (reused from chat_agent.py)
# ---------------------------------------------------------------------------

async def _get_accessible_workspaces_lg(db: AsyncSession, user: User) -> list[uuid.UUID]:
    """Mirror of _get_accessible_workspaces from chat_agent.py."""
    if user.is_superadmin:
        result = await db.execute(select(KnowledgeBase.id))
        return list(result.scalars().all())

    tenant_result = await db.execute(
        select(TenantUser.tenant_id).where(TenantUser.user_id == user.id)
    )
    user_tenant_ids = list(tenant_result.scalars().all())

    from sqlalchemy import or_
    query = select(KnowledgeBase.id).where(
        or_(
            KnowledgeBase.visibility == "public",
            KnowledgeBase.owner_id == user.id,
            KnowledgeBase.tenant_id.in_(user_tenant_ids) if user_tenant_ids else False,
        )
    )
    result = await db.execute(query)
    return list(result.scalars().all())


async def _resolve_system_prompt(
    workspace_ids: list[uuid.UUID],
    primary_id: uuid.UUID,
    db: AsyncSession,
    kb: KnowledgeBase,
) -> str:
    """Resolve document-type-specific system prompt (same logic as chat_agent.py)."""
    base_prompt = kb.system_prompt or DEFAULT_SYSTEM_PROMPT

    try:
        from sqlalchemy import select as _sel, func as _func
        from app.models.document import Document as _Doc, DocumentStatus as _DS
        from app.models.document_type import DocumentType as _DT, DocumentTypeSystemPrompt as _DTSP

        dominant_type_result = await db.execute(
            _sel(_Doc.document_type_id, _func.count(_Doc.id).label("cnt"))
            .where(
                _Doc.workspace_id.in_(workspace_ids),
                _Doc.status.in_([_DS.INDEXED, _DS.BUILDING_KG]),
                _Doc.document_type_id.isnot(None),
            )
            .group_by(_Doc.document_type_id)
            .order_by(_func.count(_Doc.id).desc())
            .limit(1)
        )
        dominant_row = dominant_type_result.first()
        if dominant_row and dominant_row.document_type_id:
            ws_prompt_res = await db.execute(
                _sel(_DTSP).where(
                    _DTSP.document_type_id == dominant_row.document_type_id,
                    _DTSP.workspace_id == primary_id,
                )
            )
            ws_prompt = ws_prompt_res.scalar_one_or_none()
            if ws_prompt:
                base_prompt = ws_prompt.system_prompt
            else:
                global_prompt_res = await db.execute(
                    _sel(_DTSP).where(
                        _DTSP.document_type_id == dominant_row.document_type_id,
                        _DTSP.workspace_id.is_(None),
                    )
                )
                global_prompt = global_prompt_res.scalar_one_or_none()
                if global_prompt:
                    base_prompt = global_prompt.system_prompt
    except Exception as e:
        logger.debug(f"Document-type system prompt resolution failed (non-fatal): {e}")

    return base_prompt + HARD_SYSTEM_PROMPT


from app.services.agent.streaming import json_serial

def _format_sse(event: str, data: dict) -> str:
    """Format as SSE string."""
    return f"event: {event}\ndata: {json.dumps(data, default=json_serial, ensure_ascii=False)}\n\n"


async def _stream_v2_standalone(
    *,
    graph,
    raw_message: str,
    workspace_ids: list[uuid.UUID],
    document_ids,
    user_id: uuid.UUID,
    user_is_superadmin: bool,
    session_id: Optional[str],
    resume_message_id=None,
    terminal_info: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Run one turn on the v2 arm and yield v1-wire SSE strings.

    Builds the real request-scoped v2 ingress (scope ∩, raw query,
    RuntimeServices/registry, dedicated lease session) and streams the turn
    through the T8 production adapter (``stream_v2_turn_to_sse``): one
    terminal event, truncation → rollback/error, terminal lease release
    after the terminal checkpoint. When ``resume_message_id`` answers a
    suspended clarification on this thread, the turn resumes from the
    checkpoint instead of starting fresh. The raw user text was already
    persisted by the caller before normalization; the evidence unit of work
    commits after the terminal response is produced.
    """
    from app.services.agent.runtime_selector import build_v2_ingress
    from app.services.agent.streaming import (
        resolve_v2_resume_command,
        stream_v2_turn_to_sse,
    )

    known_documents: tuple = ()
    if document_ids:
        from app.services.agents.v2.contracts.request import KnownDocumentResource

        known_documents = tuple(
            KnownDocumentResource(
                resource_id=str(document_id),
                document_id=document_id,
                source="api_explicit",
            )
            for document_id in (document_ids or ())
        )
    thread_id = session_id or f"standalone-{uuid.uuid4().hex[:12]}"
    async with build_v2_ingress(
        user_id=user_id,
        authenticated_workspace_ids=workspace_ids,
        requested_workspace_ids=None,
        raw_query=raw_message,
        thread_id=thread_id,
        can_read_people=bool(user_is_superadmin),
        known_documents=known_documents,
        document_ids=tuple(document_ids or ()),
    ) as ingress:
        try:
            resume_command = await resolve_v2_resume_command(
                graph=graph,
                thread_id=thread_id,
                message_id=resume_message_id,
                runtime_context=ingress.runtime_context,
            )
            async for sse_str in stream_v2_turn_to_sse(
                graph=graph,
                initial_state=(
                    None if resume_command is not None
                    else ingress.initial_state
                ),
                resume_command=resume_command,
                runtime_context=ingress.runtime_context,
                thread_id=thread_id,
                plan_resolver=ingress.plan_resolver,
                terminal_info=terminal_info,
            ):
                yield sse_str
            await ingress.commit_evidence()
        except Exception:
            await ingress.rollback_evidence()
            raise


# ---------------------------------------------------------------------------
# Core LangGraph streaming generator
# ---------------------------------------------------------------------------

async def langgraph_chat_stream(
    workspace_ids: list[uuid.UUID],
    request: ChatRequest,
    db: AsyncSession,
    user_id: uuid.UUID,
    user_email: str,
    user_is_superadmin: bool,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Run the LangGraph agent and yield SSE event strings.

    This is the LangGraph replacement for agent_chat_stream() in chat_agent.py.
    Produces identical SSE output — frontend needs no changes.
    """
    from app.core.config import settings
    from app.services.agent.runtime_selector import (
        persist_raw_user_message,
        resolve_agent_graph,
        resolve_runtime_scope,
        resolve_serving_arm,
    )
    from app.services.agent.streaming import stream_agent_to_sse, build_initial_state
    from app.services.agents.v2.execution.scheduler import V1FallbackRequired

    # F2: the runtime scope is the authenticated scope, full stop.
    # ``ChatRequest`` carries no ``workspace_ids`` field, so there is no
    # request-supplied scope to intersect here (never widened by input).
    workspace_ids = list(
        resolve_runtime_scope(
            authenticated_ids=workspace_ids,
            requested_ids=None,
        )
    )

    primary_id = workspace_ids[0] if workspace_ids else None
    if not primary_id:
        yield _format_sse("error", {"message": "No workspace IDs provided"})
        return

    # Fetch primary KB
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == primary_id))
    kb = result.scalar_one_or_none()
    if not kb:
        yield _format_sse("error", {"message": f"Knowledge base {primary_id} not found"})
        return

    # Resolve system prompt
    system_prompt = await _resolve_system_prompt(workspace_ids, primary_id, db, kb)

    # Build history
    history = []
    for m in request.history:
        role = m.role if hasattr(m, "role") else m.get("role", "user")
        content = m.content if hasattr(m, "content") else m.get("content", "")
        history.append({"role": role, "content": content})

    # Persist the RAW user text BEFORE any normalization/semantic work, so
    # chat history owns exactly what the user sent. The persisted row id is
    # the canary bucket's request ID (server-owned).
    from app.services.agent.streaming import persisted_message_uuid

    raw_user_row = None
    try:
        raw_user_row = await persist_raw_user_message(
            db,
            session_id=session_id,
            user_id=user_id,
            raw_text=request.message,
        )
    except Exception as e:
        logger.warning(f"[lg_endpoint] Failed to persist user message: {e}")
        await db.rollback()

    raw_message_uuid = None
    persisted_request_id = f"standalone-{uuid.uuid4().hex[:12]}"
    if raw_user_row is not None:
        try:
            raw_message_uuid = await persisted_message_uuid(db, raw_user_row)
            if raw_message_uuid is not None:
                persisted_request_id = str(raw_message_uuid)
        except Exception as e:
            logger.warning(f"[lg_endpoint] Failed to read message id: {e}")

    # Public chat takes no per-request arm override (the admin evaluation
    # surface is the only override, and it is admin-only). Task 7B canary:
    # server-owned selection over the authenticated scope + persisted
    # request ID; DB failures fail closed to v1 inside the helper.
    version = await resolve_serving_arm(
        db=db,
        workspace_ids=workspace_ids,
        request_id=persisted_request_id,
        is_write_endpoint=False,
    )

    # Expand abbreviations in the incoming message (graph input only — the
    # persisted row above keeps the raw text).
    message = await AbbreviationService.expand_ab_in_text(db, request.message)

    # Build initial LangGraph state
    initial_state = build_initial_state(
        workspace_ids=workspace_ids,
        message=message,
        history=history,
        system_prompt=system_prompt,
        enable_thinking=getattr(request, "enable_thinking", False),
        db=db,
        user_id=user_id,
        session_id=session_id,
        document_ids=getattr(request, "document_ids", None),
        user_can_use_people=user_is_superadmin,
    )

    # Run graph — collect events for persistence
    final_answer = ""
    from datetime import datetime as _dt, timezone as _tz

    turn_started_at = _dt.now(_tz.utc)
    served_arm = version
    final_sources: list[dict] = []
    final_images: list[dict] = []
    final_people_data: list[dict] = []
    collected_steps: list[dict] = []
    step_counter = 0

    # Task 7B (fix round 2) terminal telemetry: observed signals for the
    # canary metric row (v2 `status`/`citations` from the terminal complete
    # payload, v2 route via `terminal_info`, v1 greeting from status steps).
    v2_response_status: Optional[str] = None
    v2_citations: list = []
    v2_terminal_info: dict = {}
    turn_terminal_status = "success"
    turn_cancelled = False
    metric_emitted = False

    def _collect_terminal(sse_str: str) -> None:
        """Parse emitted events to collect data for DB persistence."""
        nonlocal final_answer, step_counter
        nonlocal final_sources, final_images, final_people_data
        nonlocal v2_response_status, v2_citations
        try:
            if sse_str.startswith("event:"):
                lines = sse_str.strip().split("\n")
                ev_type = lines[0].replace("event: ", "").strip()
                data_line = next((l for l in lines if l.startswith("data:")), None)
                if data_line:
                    ev_data = json.loads(data_line[5:].strip())
                    if ev_type == "complete":
                        final_answer = ev_data.get("answer", "")
                        final_sources = ev_data.get("sources", [])
                        final_images = ev_data.get("images", [])
                        if isinstance(ev_data.get("status"), str):
                            v2_response_status = ev_data["status"]
                        if isinstance(ev_data.get("citations"), list):
                            v2_citations = ev_data["citations"]
                    elif ev_type == "people_data":
                        final_people_data = ev_data.get("people", [])
                    elif ev_type == "status":
                        step_counter += 1
                        collected_steps.append({
                            "id": f"step-{step_counter}",
                            "step": ev_data.get("step", ""),
                            "detail": ev_data.get("detail", ""),
                            "status": "completed",
                            "timestamp": 0,
                        })
        except Exception:
            pass  # parsing errors on SSE string are non-fatal

    # Resolve the serving arm through the lazy selector (no direct graph
    # construction on any entrypoint).
    graph = await resolve_agent_graph(version)

    async def _emit_turn_metric() -> None:
        """Emit the turn's terminal metric row (Task 7B, fix round 3).

        Runs in the generator's ``finally`` so EVERY terminal outcome —
        success, error, cancellation (GeneratorExit/CancelledError), and
        fallback (attributed to the serving arm) — gets its row with the
        TRUE cancelled outcome. Every verdict comes from an observed
        terminal signal; an unobservable verdict is written as an
        explicitly-INVALID sentinel row (R80); a requested-but-
        ineffective cancellation is recorded with cancelled=True on the
        actual terminal (R81). Best-effort: never breaks serving.
        """
        nonlocal metric_emitted
        if metric_emitted:
            return
        metric_emitted = True
        try:
            from app.services.agent import rollout_metrics as _metrics

            _allowed_doc_ids = [
                str(d) for d in (getattr(request, "document_ids", None) or [])
            ]
            _served_doc_ids = _metrics.extract_served_document_ids(
                final_sources
            )
            _greeting = any(
                isinstance(step, dict)
                and step.get("detail") == _metrics.V1_GREETING_INTENT_DETAIL
                for step in collected_steps
            )
            _factual = _metrics.resolve_factual_expected(
                arm=served_arm,
                terminal_status=turn_terminal_status,
                cancelled=turn_cancelled,
                answer_text=final_answer,
                response_status=v2_response_status,
                route=(v2_terminal_info or {}).get("route"),
                greeting_observed=_greeting,
            )
            # R81: cancelled means REQUESTED (registry check best-effort);
            # R85: prefer the verdict the adapter captured BEFORE terminal
            # cleanup erased the cancel markers; fall back to a live check
            # only when the adapter did not thread one. R80: ``None``
            # factual expectation flows through — emission writes the
            # explicitly-invalid sentinel row, never omits it.
            _stashed_cancel_requested = (v2_terminal_info or {}).get(
                "cancel_requested"
            )
            if isinstance(_stashed_cancel_requested, bool):
                _cancel_requested = _stashed_cancel_requested
            else:
                _cancel_requested = await _metrics.was_cancel_requested(
                    (v2_terminal_info or {}).get("run_id")
                )
            await _metrics.try_emit_terminal_rollout_metric(
                db,
                arm=served_arm,
                request_id=persisted_request_id,
                workspace_ids=workspace_ids,
                started_at=turn_started_at,
                terminal_status=turn_terminal_status,
                citation_count=_metrics.count_grounding_evidence(
                    final_answer, v2_citations, _served_doc_ids
                ),
                cancelled=turn_cancelled,
                cancel_requested=_cancel_requested,
                answer_text=final_answer,
                factual_expected=_factual,
                served_document_ids=_served_doc_ids,
                allowed_document_ids=_allowed_doc_ids or None,
                scope_bound=True,
                production_write_count=0,
                unidentified_served_count=(
                    _metrics.count_served_sources_without_document_id(
                        final_sources
                    )
                ),
                # P0 Task 6 fix round 1 (F1): observed execution
                # telemetry only — missing stays None (unobservable, never
                # guessed); v1/fallback turns carry no v2 terminal info.
                route=(v2_terminal_info or {}).get("route"),
                response_status=(
                    v2_response_status
                    or (v2_terminal_info or {}).get("response_status")
                ),
                capability_call_count=(v2_terminal_info or {}).get(
                    "capability_call_count"
                ),
            )
        except Exception as e:
            logger.warning(f"[lg_endpoint] rollout metric emission failed: {e}")

    try:
        if version == "v2":
            try:
                async for sse_str in _stream_v2_standalone(
                    graph=graph,
                    raw_message=request.message,
                    workspace_ids=workspace_ids,
                    document_ids=getattr(request, "document_ids", None),
                    user_id=user_id,
                    user_is_superadmin=user_is_superadmin,
                    session_id=session_id,
                    resume_message_id=raw_message_uuid,
                    terminal_info=v2_terminal_info,
                ):
                    yield sse_str
                    _collect_terminal(sse_str)
            except V1FallbackRequired:
                # Task 7B: v2 candidate resolved to a v1-only route before any
                # capability execution — serve v1 with zero v2 output.
                logger.info("[lg_endpoint] v2 candidate fell back to v1")
                graph = await resolve_agent_graph("v1")
                served_arm = "v1"
                async for sse_str in stream_agent_to_sse(graph, initial_state):
                    yield sse_str
                    _collect_terminal(sse_str)
        else:
            async for sse_str in stream_agent_to_sse(graph, initial_state):
                yield sse_str
                _collect_terminal(sse_str)
    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnect / stop: the turn still gets its row with the
        # TRUE cancelled outcome, then the cancellation propagates.
        turn_cancelled = True
        turn_terminal_status = "cancelled"
        raise
    except Exception:
        turn_terminal_status = "error"
        raise
    finally:
        if not turn_cancelled and not final_answer.strip():
            turn_terminal_status = "error"
        await _emit_turn_metric()

    # Persist assistant message + thinking steps
    try:
        from app.models.chat_message import ChatMessage as ChatMessageModel
        import json as _json
        from app.services.agent.streaming import json_serial
        assistant_row = ChatMessageModel(
            message_id=str(uuid.uuid4()),
            role="assistant",
            content=final_answer,
            user_id=user_id,
            session_id=session_id,
            sources=_json.dumps(final_sources, default=json_serial) if final_sources else None,
            agent_steps=_json.dumps(collected_steps, default=json_serial) if collected_steps else None,
            people_data=_json.dumps(final_people_data, default=json_serial) if final_people_data else None,
        )
        db.add(assistant_row)
        await db.commit()
    except Exception as e:
        logger.warning(f"[lg_endpoint] Failed to persist assistant message: {e}")
        await db.rollback()

    # Enqueue the turn for a durable Graphiti personal-memory save. The memory
    # worker does the LLM fact-extraction + Neo4j write with RabbitMQ retry/DLQ;
    # if the broker is unreachable we fall back to an in-process background save.
    if user_id and request.message and final_answer:
        try:
            from app.queue.publisher import publish_memory_save_task

            await publish_memory_save_task(
                user_id=user_id,
                user_message=request.message,
                assistant_message=final_answer,
                session_id=session_id,
            )
        except Exception as e:
            logger.warning(f"[lg_endpoint] Graphiti memory enqueue failed ({e}) — falling back to in-process save")
            try:
                from app.services.memory.graphiti_client import add_conversation_episode
                import asyncio

                uid = user_id
                sid = session_id
                msg = request.message
                ans = final_answer

                async def _bg_save():
                    try:
                        await add_conversation_episode(
                            user_id=uid,
                            user_message=msg,
                            assistant_message=ans,
                            session_id=sid,
                        )
                    except Exception as e2:
                        logger.warning(f"[lg_endpoint] Graphiti episode save failed: {e2}")

                asyncio.create_task(_bg_save())
            except Exception as e2:
                logger.warning(f"[lg_endpoint] Graphiti save task spawn failed: {e2}")


# ---------------------------------------------------------------------------
# FastAPI endpoint — workspace-level
# ---------------------------------------------------------------------------

@router.post("/agent-lg/stream")
async def chat_stream_langgraph(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_principal),
):
    """
    LangGraph SSE streaming chat endpoint (workspace-agnostic).

    Uses all workspaces the user has access to. Accepts either a JWT
    bearer token or an X-API-Key (third-party clients) via get_principal.
    """
    workspace_ids = await _get_accessible_workspaces_lg(db, user)
    if not workspace_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No accessible workspaces found.",
        )

    # F2: no per-request workspace override exists — ``ChatRequest`` has
    # no ``workspace_ids`` field, so the authenticated scope above stays
    # the source of truth (never widened by request input).

    # Capture user.id early — accessing it inside the streaming generator causes
    # SQLAlchemy greenlet errors (lazy loading fails in async context)
    user_id = user.id
    user_email = user.email
    user_is_superadmin = user.is_superadmin
    session_id = None  # Non-session endpoint, no session tracking

    async def _gen():
        async for chunk in langgraph_chat_stream(workspace_ids, request, db, user_id, user_email, user_is_superadmin, session_id):
            yield chunk

    return StreamingResponse(
        content=_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/agent-lg/{workspace_id}/stream")
async def chat_stream_langgraph_workspace(
    workspace_id: uuid.UUID,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_principal),
):
    """
    LangGraph SSE streaming chat endpoint (single workspace).
    Drop-in replacement for the legacy /rag/chat/{workspace_id}/stream.
    """
    # Verify access
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base {workspace_id} not found")

    accessible = await _get_accessible_workspaces_lg(db, user)
    if workspace_id not in accessible:
        raise HTTPException(status_code=403, detail="Access denied to this workspace")

    user_id = user.id
    user_email = user.email
    user_is_superadmin = user.is_superadmin
    session_id = None

    async def _gen():
        async for chunk in langgraph_chat_stream([workspace_id], request, db, user_id, user_email, user_is_superadmin, session_id):
            yield chunk

    return StreamingResponse(
        content=_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

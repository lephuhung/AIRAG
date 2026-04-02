"""
LangGraph Chat Agent — SSE Streaming Endpoint
================================================

New endpoint that uses the LangGraph StateGraph agent instead of the legacy
manual agent loop. Produces IDENTICAL SSE event format so the frontend
requires zero changes.

Route: POST /rag/chat/agent-lg/stream
       POST /rag/chat/sessions/{session_id}/stream-lg  (session-aware variant)

Enabled when: NEXUSRAG_AGENT_BACKEND=langgraph  (default: legacy)

SSE Events emitted (same as legacy chat_agent.py):
    status       → {"step": str, "detail": str}
    thinking     → {"text": str}
    sources      → {"sources": [...]}
    images       → {"image_refs": [...]}
    token        → {"text": str}
    complete     → {"answer": str, "sources": [...], "images": [...], ...}
    error        → {"message": str}
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_active_user
from app.models.user import User
from app.models.knowledge_base import KnowledgeBase
from app.models.tenant import TenantUser
from app.schemas.rag import ChatRequest
from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT
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


def _format_sse(event: str, data: dict) -> str:
    """Format as SSE string."""
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Core LangGraph streaming generator
# ---------------------------------------------------------------------------

async def langgraph_chat_stream(
    workspace_ids: list[uuid.UUID],
    request: ChatRequest,
    db: AsyncSession,
    user: User,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Run the LangGraph agent and yield SSE event strings.

    This is the LangGraph replacement for agent_chat_stream() in chat_agent.py.
    Produces identical SSE output — frontend needs no changes.
    """
    from app.core.config import settings
    from app.services.agent.graph import get_agent_graph
    from app.services.agent.streaming import stream_agent_to_sse, build_initial_state

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

    # Expand abbreviations in the incoming message
    message = await AbbreviationService.expand_ab_in_text(db, request.message)

    # Persist user message
    try:
        from app.models.chat_message import ChatMessage as ChatMessageModel
        user_row = ChatMessageModel(
            message_id=str(uuid.uuid4()),
            role="user",
            content=message,
            user_id=user.id,
            session_id=session_id,
        )
        db.add(user_row)
        await db.commit()
    except Exception as e:
        logger.warning(f"[lg_endpoint] Failed to persist user message: {e}")
        await db.rollback()

    # Build initial LangGraph state
    initial_state = build_initial_state(
        workspace_ids=workspace_ids,
        message=message,
        history=history,
        system_prompt=system_prompt,
        enable_thinking=getattr(request, "enable_thinking", False),
        db=db,
        user_id=user.id,
        session_id=session_id,
        document_ids=getattr(request, "document_ids", None),
    )

    # Run graph — collect events for persistence
    final_answer = ""
    final_sources: list[dict] = []
    final_images: list[dict] = []
    collected_steps: list[dict] = []
    step_counter = 0

    graph = get_agent_graph()

    async for sse_str in stream_agent_to_sse(graph, initial_state):
        yield sse_str

        # Parse emitted events to collect data for DB persistence
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

    # Persist assistant message + thinking steps
    try:
        from app.models.chat_message import ChatMessage as ChatMessageModel
        import json as _json
        assistant_row = ChatMessageModel(
            message_id=str(uuid.uuid4()),
            role="assistant",
            content=final_answer,
            user_id=user.id,
            session_id=session_id,
            sources=_json.dumps(final_sources, default=str) if final_sources else None,
            agent_steps=_json.dumps(collected_steps) if collected_steps else None,
        )
        db.add(assistant_row)
        await db.commit()
    except Exception as e:
        logger.warning(f"[lg_endpoint] Failed to persist assistant message: {e}")
        await db.rollback()

    # Background: save conversation episode to Graphiti knowledge graph
    # Graphiti will extract entities and temporal facts from the turn automatically.
    if user.id and request.message and final_answer:
        try:
            from app.services.graphiti_client import add_conversation_episode
            import asyncio

            uid = user.id
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
                except Exception as e:
                    logger.warning(f"[lg_endpoint] Graphiti episode save failed: {e}")

            asyncio.create_task(_bg_save())
        except Exception as e:
            logger.warning(f"[lg_endpoint] Graphiti save task spawn failed: {e}")


# ---------------------------------------------------------------------------
# FastAPI endpoint — workspace-level
# ---------------------------------------------------------------------------

@router.post("/agent-lg/stream")
async def chat_stream_langgraph(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """
    LangGraph SSE streaming chat endpoint (workspace-agnostic).

    Uses all workspaces the user has access to. Enabled when
    NEXUSRAG_AGENT_BACKEND=langgraph (or called directly).
    """
    workspace_ids = await _get_accessible_workspaces_lg(db, user)
    if not workspace_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No accessible workspaces found.",
        )

    # Override if request specifies workspace IDs
    if hasattr(request, "workspace_ids") and request.workspace_ids:
        workspace_ids = request.workspace_ids

    async def _gen():
        async for chunk in langgraph_chat_stream(workspace_ids, request, db, user):
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
    user: User = Depends(get_current_active_user),
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

    async def _gen():
        async for chunk in langgraph_chat_stream([workspace_id], request, db, user):
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

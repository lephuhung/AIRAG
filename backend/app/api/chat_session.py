"""
REST API for Chat Sessions.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.deps import get_db, get_current_active_user
from app.models.user import User
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.models.knowledge_base import KnowledgeBase
from app.schemas.rag import ChatHistoryResponse, PersistedChatMessage
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag/chat/sessions", tags=["chat_session"])

@router.get("")
async def list_chat_sessions(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """List all chat sessions for the current user."""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
    )
    sessions = result.scalars().all()
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
        }
        for s in sessions
    ]

from pydantic import BaseModel

class CreateSessionRequest(BaseModel):
    title: str = "New Chat"

@router.post("")
async def create_chat_session(
    request: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Create a new chat session."""
    session = ChatSession(
        title=request.title,
        user_id=user.id
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }

@router.delete("/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Delete a chat session and all its messages."""
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    await db.delete(session)
    await db.commit()
    return {"status": "deleted"}

@router.get("/{session_id}/history", response_model=ChatHistoryResponse)
async def get_session_history(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get the message history for a chat session."""
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    msgs = msg_result.scalars().all()

    return ChatHistoryResponse(
        session_id=session_id,
        total=len(msgs),
        messages=[
            PersistedChatMessage(
                id=m.id,
                message_id=m.message_id,
                role=m.role,
                content=m.content,
                sources=m.sources,
                related_entities=m.related_entities,
                image_refs=m.image_refs,
                thinking=m.thinking,
                ratings=m.ratings,
                agent_steps=m.agent_steps,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in msgs
        ],
    )

@router.delete("/{session_id}/history")
async def clear_session_history(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Clear all messages from a chat session."""
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
    await db.commit()
    return {"status": "cleared"}

from fastapi.responses import StreamingResponse
from app.schemas.rag import ChatRequest
from app.api.chat_agent import agent_chat_stream, _get_accessible_workspaces

@router.post("/{session_id}/stream")
async def chat_stream_session(
    session_id: str,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """SSE endpoint for session chat."""
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    import uuid
    user_msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    
    # Save user message
    user_msg = ChatMessage(
        session_id=session_id,
        message_id=user_msg_id,
        role="user",
        content=request.message,
        user_id=user.id,
    )
    db.add(user_msg)
    await db.commit()

    # Get accessible workspaces for cross-workspace search
    workspace_ids = await _get_accessible_workspaces(db, user)

    # Get system prompt (combine from workspaces if needed or default)
    from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT
    system_prompt_to_use = DEFAULT_SYSTEM_PROMPT

    from app.api.chat_agent import sse_with_heartbeat, format_sse_event
    
    # Send default AI message id immediately
    ai_msg_id = f"msg_{uuid.uuid4().hex[:8]}"

    async def _event_generator():
        yield format_sse_event("status", {"step": "starting", "detail": "Initializing agent..."})
        yield format_sse_event("ai_message_id", {"message_id": ai_msg_id})

        accumulated_text = ""
        accumulated_thinking = ""
        final_sources = []
        final_images = []
        final_entities = []

        try:
            # Re-fetch DB session if needed, but we pass the request scoped db
            async for sse_item in agent_chat_stream(
                workspace_ids=workspace_ids,
                message=request.message,
                history=request.history,
                enable_thinking=request.enable_thinking,
                db=db,
                system_prompt=system_prompt_to_use,
                force_search=request.force_search,
            ):
                if sse_item["event"] == "token":
                    accumulated_text += sse_item["data"]["text"]
                elif sse_item["event"] == "thinking":
                    accumulated_thinking += sse_item["data"]["text"]
                elif sse_item["event"] == "sources":
                    final_sources = sse_item["data"]["sources"]
                elif sse_item["event"] == "images":
                    final_images = sse_item["data"]["image_refs"]
                elif sse_item["event"] == "complete":
                    if "related_entities" in sse_item["data"]:
                        final_entities = sse_item["data"]["related_entities"]
                    # Override text if complete event has full answer
                    if "answer" in sse_item["data"]:
                        accumulated_text = sse_item["data"]["answer"]

                yield format_sse_event(sse_item["event"], sse_item["data"])

            # Save assistant message
            ai_msg = ChatMessage(
                session_id=session_id,
                message_id=ai_msg_id,
                role="assistant",
                content=accumulated_text,
                sources=final_sources,
                related_entities=final_entities,
                image_refs=final_images,
                thinking=accumulated_thinking,
            )
            db.add(ai_msg)
            
            # Update session timestamp
            session.title = request.message[:50] + "..." if session.title == "New Chat" else session.title
            
            await db.commit()
            
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield format_sse_event("error", {"message": str(e)})

    return StreamingResponse(
        sse_with_heartbeat(_event_generator()),
        media_type="text/event-stream"
    )

from app.schemas.rag import RateSourceRequest

@router.post("/{session_id}/rate")
async def rate_source(
    session_id: str,
    body: RateSourceRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Rate a source citation in a chat message."""
    # First verify the session belongs to user
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat session not found")

    result = await db.execute(
        select(ChatMessage).where(
            ChatMessage.session_id == session_id,
            ChatMessage.message_id == body.message_id,
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )

    current_ratings = row.ratings or {}
    current_ratings[body.source_index] = body.rating
    row.ratings = current_ratings
    
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(row, "ratings")
    await db.commit()

    return {"success": True, "message_id": body.message_id, "ratings": current_ratings}

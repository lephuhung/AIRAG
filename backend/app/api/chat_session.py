"""
REST API for Chat Sessions.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.deps import get_db, get_current_active_user
from app.models.user import User
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.schemas.rag import (
    ChatHistoryResponse,
    PersistedChatMessage,
    DocumentBrief,
    SessionDocumentsResponse,
)
import logging
import uuid
from datetime import datetime, timezone

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
    session = ChatSession(title=request.title, user_id=user.id)
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
    """Delete a chat session, all its messages, chat files, and MinIO objects."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user.id
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Delete associated MinIO objects for all chat files in this session
    from app.models.chat_file import ChatFile
    from app.services.storage_service import get_storage_service

    chat_files_result = await db.execute(
        select(ChatFile).where(ChatFile.session_id == session_id)
    )
    chat_files = chat_files_result.scalars().all()

    storage = get_storage_service()
    for chat_file in chat_files:
        # Delete original file from MinIO
        if chat_file.minio_original_key:
            try:
                await storage.delete_file(chat_file.minio_original_key)
            except Exception as e:
                logger.warning(f"[session/{session_id}] Failed to delete MinIO file {chat_file.minio_original_key}: {e}")
        # Delete markdown file from MinIO
        if chat_file.minio_markdown_key:
            try:
                await storage.delete_markdown(chat_file.minio_markdown_key)
            except Exception as e:
                logger.warning(f"[session/{session_id}] Failed to delete MinIO markdown {chat_file.minio_markdown_key}: {e}")

    # Delete chat-uploaded documents (is_chat_upload=True) attached to session messages
    from app.models.chat_message import ChatMessage
    msg_result = await db.execute(
        select(ChatMessage.document_ids).where(ChatMessage.session_id == session_id)
    )
    all_doc_ids: list[uuid.UUID] = []
    for row in msg_result.scalars().all():
        if row:
            for doc_id_str in row:
                try:
                    all_doc_ids.append(uuid.UUID(str(doc_id_str)))
                except (ValueError, TypeError):
                    pass

    if all_doc_ids:
        chat_upload_docs_result = await db.execute(
            select(Document).where(
                Document.id.in_(all_doc_ids),
                Document.is_chat_upload == True
            )
        )
        chat_upload_docs = chat_upload_docs_result.scalars().all()

        for doc in chat_upload_docs:
            # Delete from ChromaDB vector store
            try:
                from app.services.rag_service import get_rag_service
                rag_svc = get_rag_service(db, doc.workspace_id)
                await rag_svc.delete_document(doc.id)
            except Exception as e:
                logger.warning(f"[session/{session_id}] Failed to delete ChromaDB chunks for doc {doc.id}: {e}")
            # Delete raw file from MinIO
            if doc.upload_s3_key:
                try:
                    await storage.delete_file(doc.upload_s3_key)
                except Exception as e:
                    logger.warning(f"[session/{session_id}] Failed to delete MinIO file {doc.upload_s3_key}: {e}")
            # Delete markdown from MinIO
            if doc.markdown_s3_key:
                try:
                    await storage.delete_markdown(doc.markdown_s3_key)
                except Exception as e:
                    logger.warning(f"[session/{session_id}] Failed to delete MinIO markdown {doc.markdown_s3_key}: {e}")
            # Delete DB record
            await db.delete(doc)
        if chat_upload_docs:
            logger.info(f"[session/{session_id}] Deleted {len(chat_upload_docs)} chat-upload documents")

    # Delete the session (cascade deletes ChatMessage, ChatFile, ExchangeSummary rows)
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
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user.id
        )
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

    # Convert stored string document_ids back to UUID list
    def _parse_doc_ids(doc_ids_json):
        if not doc_ids_json:
            return None
        try:
            return [uuid.UUID(str(d)) for d in doc_ids_json]
        except (ValueError, TypeError):
            return None

    # Collect all document IDs from messages
    all_doc_ids: list[uuid.UUID] = []
    for m in msgs:
        parsed = _parse_doc_ids(m.document_ids)
        if parsed:
            all_doc_ids.extend(parsed)

    # Fetch document metadata in bulk
    doc_metadata_map: dict[uuid.UUID, Document] = {}
    if all_doc_ids:
        doc_result = await db.execute(
            select(Document).where(Document.id.in_(all_doc_ids))
        )
        for doc in doc_result.scalars().all():
            doc_metadata_map[doc.id] = doc

    return ChatHistoryResponse(
        session_id=session_id,
        total=len(msgs),
        messages=[
            PersistedChatMessage(
                id=m.id,
                message_id=m.message_id,
                role=m.role,
                content=m.content,
                document_ids=_parse_doc_ids(m.document_ids),
                attached_docs=[
                    DocumentBrief.model_validate(doc_metadata_map[doc_id])
                    for doc_id in (_parse_doc_ids(m.document_ids) or [])
                    if doc_id in doc_metadata_map
                ]
                or None,
                sources=m.sources,
                related_entities=m.related_entities,
                image_refs=m.image_refs,
                thinking=m.thinking,
                ratings=m.ratings,
                agent_steps=m.agent_steps,
                potential_abbreviations=m.potential_abbreviations,
                people_data=m.people_data,
                # Stored as naive UTC (datetime.utcnow). Emit an explicit UTC
                # offset so the frontend doesn't misread it as local time
                # (which made VN display 7h behind).
                created_at=(
                    m.created_at.replace(tzinfo=timezone.utc).isoformat()
                    if m.created_at
                    else ""
                ),
            )
            for m in msgs
        ],
    )


@router.get("/{session_id}/documents", response_model=SessionDocumentsResponse)
async def get_session_documents(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get all documents attached to any message in a chat session (for @mention autocomplete)."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user.id
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    msg_result = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
    )
    msgs = msg_result.scalars().all()

    def _parse_doc_ids(doc_ids_json):
        if not doc_ids_json:
            return None
        try:
            return [uuid.UUID(str(d)) for d in doc_ids_json]
        except (ValueError, TypeError):
            return None

    all_doc_ids: list[uuid.UUID] = []
    for m in msgs:
        parsed = _parse_doc_ids(m.document_ids)
        if parsed:
            all_doc_ids.extend(parsed)

    doc_metadata_map: dict[uuid.UUID, Document] = {}
    if all_doc_ids:
        doc_result = await db.execute(
            select(Document).where(Document.id.in_(all_doc_ids))
        )
        for doc in doc_result.scalars().all():
            doc_metadata_map[doc.id] = doc

    documents = [DocumentBrief.model_validate(doc) for doc in doc_metadata_map.values()]
    return SessionDocumentsResponse(documents=documents)


@router.delete("/{session_id}/history")
async def clear_session_history(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Clear all messages from a chat session."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user.id
        )
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
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """SSE endpoint for session chat.

    Routes to LangGraph agent or legacy agent based on NEXUSRAG_AGENT_BACKEND config.
    Frontend URL stays the same — zero frontend changes required.
    """
    from app.core.config import settings

    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user.id
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    import uuid

    user_msg_id = f"msg_{uuid.uuid4().hex[:8]}"

    # Save user message
    # Convert UUIDs to strings for JSON serialization
    doc_ids_json = (
        [str(d) for d in request.document_ids] if request.document_ids else None
    )
    user_msg = ChatMessage(
        session_id=session_id,
        message_id=user_msg_id,
        role="user",
        content=request.message,
        user_id=user.id,
        document_ids=doc_ids_json,
    )
    db.add(user_msg)
    await db.commit()

    # Get accessible workspaces for cross-workspace search
    workspace_ids = await _get_accessible_workspaces(db, user)

    # Get system prompt
    from app.prompts.chat import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT

    system_prompt_to_use = DEFAULT_SYSTEM_PROMPT + HARD_SYSTEM_PROMPT

    # Build conversation context from exchange summaries (if session has history)
    from app.services.conversation_summary_service import (
        get_conversation_summary_service,
    )

    summary_svc = get_conversation_summary_service()
    conversation_context = await summary_svc.get_context_for_session(
        db, session_id, limit=10
    )
    if conversation_context:
        system_prompt_to_use += (
            "\n\n=== CONVERSATION HISTORY (from previous exchanges) ===\n"
            + conversation_context
            + "\n=== END CONVERSATION HISTORY ===\n"
        )

    from app.api.chat_agent import sse_with_heartbeat, format_sse_event

    # Send AI message id immediately
    ai_msg_id = f"msg_{uuid.uuid4().hex[:8]}"

    DEFAULT_TITLES = ["New Chat", "New chat", "Chat mới", "Kho tri thức"]

    # Full background task: save message, generate summary, update title
    async def _background_persist_and_emit(
        session_id: str,
        user_id: str,
        user_message: str,
        text: str,
        thinking: str,
        sources: list,
        images: list,
        steps: list,
        potentials: list,
        people_data: list,
        user_msg_id: str,
        ai_msg_id: str,
    ):
        """Run entirely in background: save message + exchange summary, update title."""
        from app.core.database import async_session_maker

        try:
            async with async_session_maker() as db:
                res = await db.execute(
                    select(ChatSession).where(ChatSession.id == session_id)
                )
                bg_session = res.scalar_one_or_none()
                if not bg_session:
                    return

                # Process steps
                processed_steps = []
                for step in steps:
                    step_copy = step.copy()
                    if step_copy.get("status") == "active":
                        step_copy["status"] = "completed"
                    processed_steps.append(step_copy)

                if not any(s.get("step") == "done" for s in processed_steps):
                    processed_steps.append({
                        "id": f"step_done_{uuid.uuid4().hex[:6]}",
                        "step": "done",
                        "status": "completed",
                        "detail": "Hoàn thành",
                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                    })

                # Save assistant message
                ai_msg = ChatMessage(
                    session_id=session_id,
                    message_id=ai_msg_id,
                    role="assistant",
                    content=text,
                    sources=sources,
                    image_refs=images,
                    thinking=thinking or None,
                    agent_steps=processed_steps,
                    potential_abbreviations=potentials or None,
                    people_data=people_data or None,
                )
                db.add(ai_msg)

                # Determine if title was auto-set
                frontend_auto_title = user_message[:30] if len(user_message) > 30 else user_message
                title_was_auto_set = (
                    bg_session.title in DEFAULT_TITLES
                    or not bg_session.title
                    or bg_session.title == frontend_auto_title
                )

                # Set interim title from message
                if title_was_auto_set:
                    bg_session.title = user_message[:50] + ("..." if len(user_message) > 50 else "")

                await db.commit()

            # Generate exchange summary outside the session scope (LLM call may be slow)
            topic_label = None
            try:
                from app.services.conversation_summary_service import get_conversation_summary_service
                summary_svc = get_conversation_summary_service()
                async with async_session_maker() as db:
                    summary_result = await summary_svc.save_exchange_summary(
                        db=db,
                        session_id=session_id,
                        user_message_id=user_msg_id,
                        assistant_message_id=ai_msg_id,
                        user_message=user_message,
                        assistant_message=text,
                        cited_sources=sources,
                    )

                    if summary_result is not None and summary_result.exchange_index == 1:
                        topic_label = summary_result.topic_label
                        logger.info(f"[session/{session_id}] topic_label: {topic_label}")

                        # Re-fetch session to update title
                        res = await db.execute(
                            select(ChatSession).where(ChatSession.id == session_id)
                        )
                        bg_session = res.scalar_one_or_none()
                        if bg_session and topic_label and title_was_auto_set:
                            bg_session.title = topic_label
                            await db.commit()
                            logger.info(f"[session/{session_id}] Title updated to topic_label: {topic_label}")
            except Exception as e:
                logger.warning(f"[session/{session_id}] Failed to save exchange summary: {e}", exc_info=True)

            # Graphiti save (separate session)
            try:
                from app.services.graphiti_client import add_conversation_episode
                if user_id and user_message and text:
                    await add_conversation_episode(
                        user_id=user_id,
                        user_message=user_message,
                        assistant_message=text,
                        session_id=session_id,
                    )
            except Exception as e:
                logger.error(f"[session/{session_id}] Graphiti save failed: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"[session/{session_id}] Background persistence failed: {e}", exc_info=True)

    # ── Route: LangGraph agent ──────────────────────────────────────────────
    use_langgraph = settings.NEXUSRAG_AGENT_BACKEND.lower() == "langgraph"

    if use_langgraph:
        logger.info(f"[session/{session_id}] Routing to LangGraph agent backend")

        async def _event_generator_lg():
            yield format_sse_event(
                "status",
                {"step": "starting", "detail": "Initializing LangGraph agent..."},
            )
            yield format_sse_event("user_id", {"id": user_msg_id})
            yield format_sse_event("ai_message_id", {"message_id": ai_msg_id})

            accumulated_text = ""
            accumulated_thinking = ""
            final_sources: list = []
            final_images: list = []
            final_steps: list = []
            final_potential_abbreviations: list = []
            final_people_data: list = []

            try:
                from app.services.agents.supervisor import get_supervisor_graph as get_agent_graph
                from app.services.agent.streaming import (
                    stream_agent_to_sse,
                    build_initial_state,
                )
                import json as _json

                history = []
                for m in request.history:
                    role = m.role if hasattr(m, "role") else m.get("role", "user")
                    content = (
                        m.content if hasattr(m, "content") else m.get("content", "")
                    )
                    history.append({"role": role, "content": content})

                initial_state = build_initial_state(
                    workspace_ids=workspace_ids,
                    message=request.message,
                    history=history,
                    system_prompt=system_prompt_to_use,
                    enable_thinking=getattr(request, "enable_thinking", False),
                    db=db,
                    user_id=user.id,
                    session_id=session_id,
                    document_ids=request.document_ids,
                )

                logger.info(
                    f"[session/{session_id}] LangGraph initial_state: "
                    f"document_ids={[str(d) for d in (request.document_ids or [])]} "
                    f"force_search={getattr(request, 'force_search', False)}"
                )

                graph = get_agent_graph()

                async for sse_str in stream_agent_to_sse(graph, initial_state):
                    # Collect data for DB persistence while yielding
                    try:
                        if sse_str.startswith("event:"):
                            lines = sse_str.strip().split("\n")
                            ev_type = lines[0].replace("event: ", "").strip()
                            data_line = next(
                                (l for l in lines if l.startswith("data:")), None
                            )
                            if data_line:
                                ev_data = _json.loads(data_line[5:].strip())
                                if ev_type == "token":
                                    accumulated_text += ev_data.get("text", "")
                                elif ev_type == "thinking":
                                    accumulated_thinking += ev_data.get("text", "")
                                elif ev_type == "sources":
                                    final_sources = ev_data.get("sources", [])
                                elif ev_type == "images":
                                    final_images = ev_data.get(
                                        "image_refs", ev_data.get("images", [])
                                    )
                                elif ev_type == "status":
                                    final_steps.append(ev_data)
                                elif ev_type == "complete":
                                    if "answer" in ev_data:
                                        accumulated_text = ev_data["answer"]
                                elif ev_type == "potential_abbreviations":
                                    final_potential_abbreviations = ev_data.get(
                                        "abbreviations", []
                                    )
                                elif ev_type == "people_data":
                                    final_people_data = ev_data.get("people", [])
                    except Exception:
                        pass

                    yield sse_str

                # All persistence in background — stream closes immediately
                background_tasks.add_task(
                    _background_persist_and_emit,
                    session_id=session_id,
                    user_id=str(user.id),
                    user_message=request.message,
                    text=accumulated_text,
                    thinking=accumulated_thinking,
                    sources=final_sources,
                    images=final_images,
                    steps=final_steps,
                    potentials=final_potential_abbreviations,
                    people_data=final_people_data,
                    user_msg_id=user_msg_id,
                    ai_msg_id=ai_msg_id,
                )

            except Exception as e:
                logger.error(f"[lg/session] LangGraph stream error: {e}", exc_info=True)
                yield format_sse_event("error", {"message": str(e)})

        return StreamingResponse(
            sse_with_heartbeat(_event_generator_lg()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Route: Legacy agent (default) ──────────────────────────────────────
    logger.info(f"[session/{session_id}] Routing to legacy agent backend")

    async def _event_generator():
        yield format_sse_event(
            "status", {"step": "starting", "detail": "Initializing agent..."}
        )
        yield format_sse_event("user_id", {"id": user_msg_id})
        yield format_sse_event("ai_message_id", {"message_id": ai_msg_id})

        accumulated_text = ""
        accumulated_thinking = ""
        final_sources = []
        final_images = []
        final_entities = []
        final_steps = []
        final_people_data = []

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
                user_id=user.id,
                session_id=session_id,
                document_ids=request.document_ids,
            ):
                if sse_item["event"] == "token":
                    accumulated_text += sse_item["data"]["text"]
                elif sse_item["event"] == "thinking":
                    accumulated_thinking += sse_item["data"]["text"]
                elif sse_item["event"] == "sources":
                    final_sources = sse_item["data"]["sources"]
                elif sse_item["event"] == "images":
                    final_images = sse_item["data"]["image_refs"]
                elif sse_item["event"] == "status":
                    final_steps.append(sse_item["data"])
                elif sse_item["event"] == "people_data":
                    final_people_data = sse_item["data"].get("people", [])
                elif sse_item["event"] == "complete":
                    if "related_entities" in sse_item["data"]:
                        final_entities = sse_item["data"]["related_entities"]
                    # Override text if complete event has full answer
                    if "answer" in sse_item["data"]:
                        accumulated_text = sse_item["data"]["answer"]

                yield format_sse_event(sse_item["event"], sse_item["data"])

            # All persistence in background — stream closes immediately
            background_tasks.add_task(
                _background_persist_and_emit,
                session_id=session_id,
                user_id=str(user.id),
                user_message=request.message,
                text=accumulated_text,
                thinking=accumulated_thinking,
                sources=final_sources,
                images=final_images,
                steps=final_steps,
                potentials=[],
                people_data=final_people_data,
                user_msg_id=user_msg_id,
                ai_msg_id=ai_msg_id,
            )

        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield format_sse_event("error", {"message": str(e)})

    return StreamingResponse(
        sse_with_heartbeat(_event_generator()), media_type="text/event-stream"
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
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user.id
        )
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

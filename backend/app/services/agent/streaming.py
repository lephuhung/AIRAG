"""
SSE Streaming Adapter for LangGraph — Queue + ContextVar Architecture
======================================================================

Root-cause fix: LangGraph strips keys không có trong AgentState TypedDict trước
khi truyền vào nodes. _event_queue và _db bị xóa → nodes không thể push events.

Fix: dùng contextvars.ContextVar để truyền queue/db ngoài LangGraph state.
asyncio.create_task() copy context tại thời điểm tạo task → nodes nhìn thấy queue.

Flow:
    stream_agent_to_sse
        ├── tạo event_queue
        ├── set _event_queue_ctx & _db_ctx  ← bypass LangGraph state filtering
        ├── spawn background task: graph.ainvoke(initial_state)
        │       memory_recall  → push_event("status", ...)
        │       intent_classifier → push_event("status", ...)
        │       tool_executor  → push_event("status") + push_event("sources") + push_event("images")
        │       answer_generator → push_event("status") + push_event("token") * N
        └── drain queue → yield SSE events

SSE events (format tương thích 100% với legacy chat_agent.py):
    event: status       {"step": str, "detail": str}
    event: thinking     {"text": str}
    event: sources      {"sources": [...]}
    event: images       {"image_refs": [...]}
    event: token        {"text": str}
    event: complete     {"answer": str, "sources": [...], "images": [...]}
    event: error        {"message": str}
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextvars import ContextVar
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

SSE_HEARTBEAT_INTERVAL = 15  # seconds

# ---------------------------------------------------------------------------
# Module-level ContextVars — survive LangGraph state key filtering
# ---------------------------------------------------------------------------

# Shared asyncio.Queue for SSE events — nodes push, stream_agent_to_sse drains
_event_queue_ctx: ContextVar[asyncio.Queue | None] = ContextVar(
    "_event_queue", default=None
)

# DB session — nodes read via get_current_db() instead of state.get("_db")
_db_ctx: ContextVar = ContextVar("_db", default=None)


def get_current_db():
    """
    Get the DB session from the current async context.
    Use inside LangGraph nodes instead of state.get("_db").
    """
    return _db_ctx.get()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    """Format a dict as an SSE event string."""
    json_data = json.dumps(data, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


# ---------------------------------------------------------------------------
# Main streaming function
# ---------------------------------------------------------------------------

async def stream_agent_to_sse(
    graph,
    initial_state: dict,
) -> AsyncGenerator[str, None]:
    """
    Run the LangGraph agent and yield SSE-formatted strings in real-time.

    Dùng ContextVar thay vì state dict để truyền queue và db vào nodes,
    bypass LangGraph's TypedDict key filtering.
    """
    event_queue: asyncio.Queue = asyncio.Queue()

    # Set contextvars BEFORE create_task — asyncio copies current context into task
    queue_token = _event_queue_ctx.set(event_queue)
    db_token = _db_ctx.set(initial_state.get("_db"))

    # Tracking cho complete event
    final_answer = ""
    all_sources: list = []
    all_images: list = []

    # ── Background task: chạy toàn bộ LangGraph pipeline ───────────────────
    async def _run_graph():
        from app.core.config import settings
        try:
            await graph.ainvoke(initial_state, debug=settings.NEXUSRAG_LG_DEBUG)
        except Exception as e:
            logger.error(f"[stream] Graph execution error: {e}", exc_info=True)
            await event_queue.put(("error", str(e)))
        finally:
            # Sentinel: báo hiệu pipeline đã xong
            await event_queue.put(("done", None))

    # create_task copies current context → task sees _event_queue_ctx & _db_ctx
    task = asyncio.create_task(_run_graph())

    # ── Main loop: drain queue → yield SSE ──────────────────────────────────
    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    event_queue.get(), timeout=SSE_HEARTBEAT_INTERVAL
                )
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue

            if not isinstance(item, tuple):
                continue

            ev_type = item[0]

            if ev_type == "done":
                # Pipeline xong — emit complete event
                yield _sse("complete", {
                    "answer": final_answer,
                    "sources": all_sources,
                    "images": all_images,
                })
                logger.info(
                    f"[stream] Complete: {len(final_answer)} chars, "
                    f"{len(all_sources)} sources, {len(all_images)} images"
                )
                break

            elif ev_type == "status":
                yield _sse("status", item[1])

            elif ev_type == "sources":
                all_sources = item[1]
                yield _sse("sources", {"sources": all_sources})
                logger.info(f"[stream] Emitted {len(all_sources)} sources")

            elif ev_type == "images":
                all_images = item[1]
                yield _sse("images", {"image_refs": all_images})

            elif ev_type == "token":
                text = item[1]
                final_answer += text
                yield _sse("token", {"text": text})

            elif ev_type == "thinking":
                yield _sse("thinking", {"text": item[1]})

            elif ev_type == "error":
                yield _sse("error", {"message": item[1]})
                break

    finally:
        # Reset contextvars
        _event_queue_ctx.reset(queue_token)
        _db_ctx.reset(db_token)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Helper: push event vào queue (dùng trong nodes)
# ---------------------------------------------------------------------------

async def push_event(state: dict, ev_type: str, ev_data) -> None:
    """
    Push một event vào event_queue thông qua ContextVar.

    Đọc queue từ _event_queue_ctx thay vì state.get("_event_queue") để
    bypass LangGraph TypedDict key filtering.

    Sau khi push, gọi asyncio.sleep(0) để yield control về event loop,
    cho phép stream_agent_to_sse nhận event ngay lập tức.
    """
    # Đọc từ ContextVar (bypass LangGraph state filtering)
    queue: asyncio.Queue | None = _event_queue_ctx.get()

    # Fallback: thử đọc từ state nếu contextvar chưa set (e.g. unit test)
    if queue is None and state:
        queue = state.get("_event_queue")

    if queue is not None:
        await queue.put((ev_type, ev_data))
        await asyncio.sleep(0)  # yield control — QUAN TRỌNG cho real-time streaming


# ---------------------------------------------------------------------------
# Build initial state
# ---------------------------------------------------------------------------

def build_initial_state(
    workspace_ids: list[int],
    message: str,
    history: list[dict],
    system_prompt: str,
    enable_thinking: bool,
    db,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    document_ids: Optional[list[int]] = None,
) -> dict:
    """
    Build the initial AgentState dict from a chat request.

    _db được lưu vào _db_ctx ContextVar bởi stream_agent_to_sse.
    Vẫn truyền _db vào state dict để stream_agent_to_sse đọc và set vào ctx.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from app.services.agent.state import DEFAULT_STATE

    messages = []
    for msg in (history or [])[-10:]:
        role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    # Current user message
    messages.append(HumanMessage(content=message))

    return {
        **DEFAULT_STATE,
        "messages": messages,
        "workspace_ids": workspace_ids,
        "document_ids": document_ids,
        "user_id": user_id,
        "session_id": session_id,
        "system_prompt": system_prompt,
        "enable_thinking": enable_thinking,
        # _db lưu ở đây để stream_agent_to_sse đọc và inject vào _db_ctx
        # LangGraph sẽ strip key này trước khi truyền vào nodes
        # → nodes phải dùng get_current_db() thay vì state.get("_db")
        "_db": db,
    }

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
    event: token_rollback {}   # discard speculative tokens streamed before a tool call
    event: complete     {"answer": str, "sources": [...], "images": [...]}
    event: error        {"message": str}
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextvars import ContextVar
from typing import AsyncGenerator, Optional

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

SSE_HEARTBEAT_INTERVAL = 15  # seconds

# ---------------------------------------------------------------------------
# Langfuse client (lazy initialization)
# ---------------------------------------------------------------------------

_langfuse_handler: Optional[CallbackHandler] = None


def get_langfuse_handler() -> Optional[CallbackHandler]:
    """Get or create the Langfuse callback handler (lazy init)."""
    global _langfuse_handler
    if _langfuse_handler is None:
        try:
            get_client()
            _langfuse_handler = CallbackHandler()
            logger.info("[langfuse] CallbackHandler initialized")
        except Exception as e:
            logger.warning(f"[langfuse] Failed to initialize CallbackHandler: {e}")
            return None
    return _langfuse_handler

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


def _last_user_message(messages) -> str:
    """Extract the most recent textual message content (trace input)."""
    for m in reversed(messages or []):
        content = getattr(m, "content", None)
        if isinstance(content, str) and content:
            return content
    return ""


def _summarize_final_state(state) -> dict:
    """Compact, serializable summary of the final AgentState for the trace output.

    Best-effort: missing keys are simply omitted so this never raises.
    """
    if not isinstance(state, dict):
        return {"completed": True}
    summary: dict = {"completed": True}
    for key in ("intent", "next_agent", "query_complexity", "search_mode", "iteration_count"):
        val = state.get(key)
        if val is not None:
            summary[key] = val
    for m in reversed(state.get("messages") or []):
        content = getattr(m, "content", None)
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if isinstance(content, str) and content and role in ("ai", "assistant", None):
            summary["answer"] = content
            break
    srcs = state.get("sources")
    if isinstance(srcs, list):
        summary["sources_count"] = len(srcs)
    return summary


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)


def _sse(event: str, data: dict) -> str:
    """Format a dict as an SSE event string."""
    json_data = json.dumps(data, default=json_serial, ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


# ---------------------------------------------------------------------------
# Main streaming function
# ---------------------------------------------------------------------------

async def stream_agent_events(
    graph,
    initial_state: dict,
    channel: str = "web",
) -> AsyncGenerator[dict, None]:
    """
    Run the LangGraph agent and yield events as dicts ``{"event", "data"}`` in
    real-time. This is the transport-agnostic core used by both the SSE web
    endpoint (``stream_agent_to_sse``) and the in-process Telegram consumer.

    Dùng ContextVar thay vì state dict để truyền queue và db vào nodes,
    bypass LangGraph's TypedDict key filtering.

    Event shapes (giữ tương thích với consumer cũ):
      status/thinking → data nguyên trạng; sources → {"sources"}; images →
      {"image_refs"}; token → {"text"}; token_rollback → {}; complete →
      {"answer","sources","images","potential_abbreviations","people_data"};
      error → {"message"}. Heartbeat phát {"event":"heartbeat","data":{}}.
    """
    event_queue: asyncio.Queue = asyncio.Queue()

    # ── Dataset trace collector (distillation capture) ─────────────────────────
    # Set on a ContextVar BEFORE create_task so the background graph task and all
    # downstream taps (supervisor routing, TracedLLMProvider, tool dispatch)
    # inherit it. Entirely best-effort: never affects the chat response.
    from app.services.agent.trace_collector import (
        TraceCollector, set_collector, reset_collector, trace_enabled,
    )
    collector = TraceCollector(channel=channel) if trace_enabled() else None
    collector_token = set_collector(collector) if collector is not None else None
    if collector is not None:
        try:
            msgs = initial_state.get("messages") or []
            collector.start_run(
                original_query=_last_user_message(msgs),
                workspace_ids=initial_state.get("workspace_ids"),
                document_ids=initial_state.get("document_ids"),
                user_id=initial_state.get("user_id"),
                session_id=initial_state.get("session_id"),
                history_len=max(0, len(msgs) - 1),
            )
        except Exception:
            pass

    # Set contextvars BEFORE create_task — asyncio copies current context into task
    queue_token = _event_queue_ctx.set(event_queue)
    db_token = _db_ctx.set(initial_state.get("_db"))

    # Trace run outcome (persisted in finally)
    run_error: str | None = None
    run_completed = False

    # Tracking cho complete event
    final_answer = ""
    all_sources: list = []
    all_images: list = []
    all_potentials: list = []
    all_people_data: list = []

    # ── Background task: chạy toàn bộ LangGraph pipeline ───────────────────
    async def _run_graph():
        from app.core.config import settings
        langfuse_handler = get_langfuse_handler()
        callbacks = [langfuse_handler] if langfuse_handler else []

        # Langfuse client for the root trace span (None if unavailable)
        lf = None
        if langfuse_handler is not None:
            try:
                lf = get_client()
            except Exception:
                lf = None

        config = {"callbacks": callbacks}

        async def _invoke():
            return await graph.ainvoke(
                initial_state,
                config=config,
                debug=settings.NEXUSRAG_LG_DEBUG,
            )

        try:
            if lf is not None:
                # Root span wraps the whole run so every manual span (nodes AND
                # conditional-edge routers) nests under ONE trace instead of
                # spawning orphan traces. propagate_attributes stamps
                # session/user/tags onto the trace + all child spans so it is
                # filterable in the Langfuse UI.
                sid = initial_state.get("session_id")
                uid = initial_state.get("user_id")
                wids = initial_state.get("workspace_ids") or []
                dids = initial_state.get("document_ids") or []
                tags = ["langgraph", "agent_backend:langgraph"]
                with lf.start_as_current_observation(
                    name="langgraph_chat",
                    as_type="span",
                    input={
                        "message": _last_user_message(initial_state.get("messages")),
                        "workspace_ids": [str(w) for w in wids],
                        "document_ids": [str(d) for d in dids],
                    },
                ) as root:
                    with propagate_attributes(
                        user_id=str(uid) if uid else None,
                        session_id=str(sid) if sid else None,
                        trace_name="langgraph_chat",
                        tags=tags,
                    ):
                        final_state = await _invoke()
                    try:
                        root.update(output=_summarize_final_state(final_state))
                    except Exception:
                        pass
            else:
                await _invoke()
        except Exception as e:
            logger.error(f"[stream] Graph execution error: {e}", exc_info=True)
            await event_queue.put(("error", str(e)))
        finally:
            # Sentinel: báo hiệu pipeline đã xong
            await event_queue.put(("done", None))
            # Ensure buffered observations are delivered to Langfuse
            if lf is not None:
                try:
                    lf.flush()
                except Exception:
                    pass

    # create_task copies current context → task sees _event_queue_ctx & _db_ctx
    task = asyncio.create_task(_run_graph())

    # ── Main loop: drain queue → yield dict events ──────────────────────────
    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    event_queue.get(), timeout=SSE_HEARTBEAT_INTERVAL
                )
            except asyncio.TimeoutError:
                yield {"event": "heartbeat", "data": {}}
                continue

            if not isinstance(item, tuple):
                continue

            ev_type = item[0]

            if ev_type == "done":
                run_completed = True
                # Pipeline xong — emit complete event
                yield {"event": "complete", "data": {
                    "answer": final_answer,
                    "sources": all_sources,
                    "images": all_images,
                    "potential_abbreviations": all_potentials,
                    "people_data": all_people_data,
                }}
                logger.info(
                    f"[stream] Complete: {len(final_answer)} chars, "
                    f"{len(all_sources)} sources, {len(all_images)} images"
                )
                break

            elif ev_type == "status":
                yield {"event": "status", "data": item[1]}

            elif ev_type == "sources":
                all_sources = item[1]
                yield {"event": "sources", "data": {"sources": all_sources}}
                logger.info(f"[stream] Emitted {len(all_sources)} sources")

            elif ev_type == "images":
                all_images = item[1]
                yield {"event": "images", "data": {"image_refs": all_images}}

            elif ev_type == "token":
                text = item[1]
                final_answer += text
                yield {"event": "token", "data": {"text": text}}

            elif ev_type == "token_rollback":
                # Speculative answer tokens turned out to precede a tool call —
                # discard them so the final `complete` answer stays clean.
                final_answer = ""
                yield {"event": "token_rollback", "data": {}}

            elif ev_type == "thinking":
                yield {"event": "thinking", "data": item[1]}

            elif ev_type == "potential_abbreviations":
                all_potentials = item[1]
                yield {"event": "potential_abbreviations", "data": {"abbreviations": all_potentials}}

            elif ev_type == "error":
                run_error = str(item[1])
                yield {"event": "error", "data": {"message": item[1]}}
                break

            elif ev_type == "people_data":
                all_people_data = item[1]
                yield {"event": "people_data", "data": {"people": all_people_data}}
                logger.info(f"[stream] Emitted {len(all_people_data)} people records")

    finally:
        # Reset contextvars
        _event_queue_ctx.reset(queue_token)
        _db_ctx.reset(db_token)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Persist the dataset trace (best-effort; never affects the response).
        if collector is not None:
            try:
                collector.finish(
                    final_answer=final_answer,
                    success=(run_completed and run_error is None),
                    error=run_error or (None if run_completed else "incomplete/cancelled"),
                )
                from app.services.agent_trace_service import AgentTraceService

                await AgentTraceService.record(collector)
            except Exception as e:  # pragma: no cover - best effort
                logger.debug(f"[trace] persist failed: {e}")
            finally:
                if collector_token is not None:
                    reset_collector(collector_token)


async def stream_agent_to_sse(
    graph,
    initial_state: dict,
) -> AsyncGenerator[str, None]:
    """
    SSE wrapper around :func:`stream_agent_events`. Yields SSE-formatted strings
    for the web chat endpoint. Heartbeats become SSE comment lines.
    """
    async for ev in stream_agent_events(graph, initial_state):
        if ev["event"] == "heartbeat":
            yield ": heartbeat\n\n"
        else:
            yield _sse(ev["event"], ev["data"])


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
    workspace_ids: list[uuid.UUID],
    message: str,
    history: list[dict],
    system_prompt: str,
    enable_thinking: bool,
    db,
    user_id: Optional[uuid.UUID] = None,
    session_id: Optional[str] = None,
    document_ids: Optional[list[uuid.UUID]] = None,
    user_can_use_people: bool = False,
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
        "user_can_use_people": user_can_use_people,
        # _db lưu ở đây để stream_agent_to_sse đọc và inject vào _db_ctx
        # LangGraph sẽ strip key này trước khi truyền vào nodes
        # → nodes phải dùng get_current_db() thay vì state.get("_db")
        "_db": db,
    }

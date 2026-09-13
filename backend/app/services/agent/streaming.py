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

from app.services.agent.sources_accumulator import SourcesSnapshotAccumulator, Source as AccSource

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
    sources_acc = SourcesSnapshotAccumulator()
    # Keep track of original source dicts for output (per B4: preserve original format)
    original_sources: list = []
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
                # Use SourcesSnapshotAccumulator as single source of truth for dedup
                # (per B4 contract). Emit original dict format, filtered by deduped keys.
                deduped_keys = set()
                for s in sources_acc.deduplicated():
                    key = (s.document_id or s.doc, s.chunk, s.content_hash, s.source_id or "")
                    deduped_keys.add(key)
                all_sources = []
                for orig in original_sources:
                    if isinstance(orig, dict):
                        doc_id = orig.get("document_id") or orig.get("doc", "")
                        chunk = orig.get("chunk", orig.get("page_or_chunk", ""))
                        content_hash = orig.get("content_hash", orig.get("chunk_id", ""))
                        source_id = orig.get("source_id")
                        key = (doc_id, chunk, content_hash, source_id or "")
                        if key in deduped_keys:
                            all_sources.append(orig)
                    else:
                        all_sources.append(orig)
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
                # Accumulate sources via SourcesSnapshotAccumulator (single source of truth, per B4)
                incoming_sources = item[1] if item[1] else []
                for s in incoming_sources:
                    if isinstance(s, dict):
                        # Create a Source object for deduplication tracking
                        acc_source = AccSource(
                            doc=s.get("doc", ""),
                            chunk=s.get("chunk", s.get("page_or_chunk", "")),
                            content_hash=s.get("content_hash", s.get("chunk_id", "")),
                            source_id=s.get("source_id"),
                            document_id=s.get("document_id"),
                        )
                        sources_acc.add([acc_source])
                        # Also track the original dict for rollback and output
                        original_sources.append(s)
                    elif hasattr(s, "document_id"):
                        sources_acc.add([s])
                        original_sources.append(s)
                    else:
                        original_sources.append(s)
                # Emit deduplicated sources using accumulator (B4 contract)
                # Use accumulator's keys to filter original_sources for deduplication
                deduped_keys = set()
                for s in sources_acc.deduplicated():
                    key = (s.document_id or s.doc, s.chunk, s.content_hash, s.source_id or "")
                    deduped_keys.add(key)
                # Emit original dict format, filtered by deduped keys
                all_sources = []
                for orig in original_sources:
                    if isinstance(orig, dict):
                        doc_id = orig.get("document_id") or orig.get("doc", "")
                        chunk = orig.get("chunk", orig.get("page_or_chunk", ""))
                        content_hash = orig.get("content_hash", orig.get("chunk_id", ""))
                        source_id = orig.get("source_id")
                        key = (doc_id, chunk, content_hash, source_id or "")
                        if key in deduped_keys:
                            all_sources.append(orig)
                    else:
                        all_sources.append(orig)
                yield {"event": "sources", "data": {"sources": all_sources}}
                logger.info(f"[stream] Emitted {len(all_sources)} sources (accumulated)")

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
                # Also reset every other answer accumulator (sources, images,
                # abbreviations, people_data) so the terminal ``complete`` event
                # carries the cleared snapshot — pre-rollback artifacts must NOT
                # survive into the persisted message / SSE relay.
                final_answer = ""
                sources_acc.clear()
                original_sources = []  # Also clear original sources
                all_images = []
                all_potentials = []
                all_people_data = []
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


# ---------------------------------------------------------------------------
# v2 outer runner / streaming adapter (Phase 2, Task 8)
# ---------------------------------------------------------------------------
#
# Preserves the EXISTING SSE contract used by the frontend
# (``frontend/src/hooks/useRAGChatStream.ts``). Events this adapter EMITS:
# status / token / token_rollback / potential_abbreviations (projected from
# the checkpointed semantic abbreviations unknown to the DB) / error /
# complete. ``sources`` / ``images`` / ``people_data`` have NO v2 projection
# yet (the v2 terminal carries citations only; fabricating sources from
# citations would be dishonest) — Phase-3-owned evidence→source projection.
# Payloads stay additive (v2 ``complete`` additionally carries ``status`` +
# ``citations``); the hook needs no change.
#
# Rules (controller rulings):
#
# - Only this OUTER adapter streams user-facing prose: the terminal
#   ``FinalResponse.content`` is chunked into ``token`` events here. v2
#   nodes never push token events (pinned by test).
# - Exactly ONE terminal event per run: ``complete`` for success/clarify,
#   ``error`` for denied/insufficient/error AND for unexpected graph
#   failures (a raw exception never escapes the SSE stream; leases still
#   release as terminal).
# - ``token_rollback`` clears every accumulator. A deadline-truncated
#   dispatch (checkpointed plan with undispatched tasks — the T6 runtime-only
#   truncation channel, deliberately never checkpointed) surfaces as
#   token_rollback + error, never a normal success.
# - Disconnect/cancel prevents success: ``CancelledError``/``GeneratorExit``
#   propagate, no terminal event is emitted, leases release as ``cancelled``.
# - Suspension is detected from the RETURNED state (pinned langgraph
#   1.0.0 suppresses top-level interrupts and returns values carrying
#   ``__interrupt__``; a ``GraphInterrupt`` catch is kept for nested uses):
#   a suspended clarify turn surfaces the question as the single terminal
#   ``complete`` and NEVER releases leases (suspension keeps them active).
# - Stable thread id resumes from the checkpoint. Resume call form (the
#   graph owns navigation; never pass an outer destination along):
#
#       command = await resume_clarification(message_id, request, runtime)
#       await graph.ainvoke(Command(resume=<resolution>), config,
#                           context=runtime_context)
#
#   On resume the runner refreshes existing leases BEFORE continuing, feeds
#   the plan resolver from the checkpointed plan/bindings, and reinjects the
#   CURRENT runtime ACL (checkpointed scope is never trusted).
# - Terminal lease release (``release_run``) happens HERE, only AFTER the
#   terminal checkpoint succeeds — never inside the finalizer, never on
#   suspension (a release-once guard also covers racing disconnects).
# - ``ClarificationUnsatisfiable`` means the reply is a fresh turn, never a
#   resume retry (``V2FreshTurnRequired``, decided by T7's
#   ``clarification_reply_is_fresh_turn`` — the single owner).
# - Non-success terminals over the ingress placeholder semantic surface as
#   errors via T7's ``terminal_state_is_error`` (the single owner) —
#   ``validate_supervisor_state`` is never run here.
#
# All v2-only imports are function-local so this module (imported by every
# entrypoint) stays light and can never create an import cycle with the v2
# package (``v2/events.py`` itself imports ``json_serial`` from here).

_V2_TOKEN_CHUNK_DEFAULT = 120

_V2_TRUNCATION_MESSAGE = (
    "Quá trình thu thập bằng chứng chưa hoàn tất (hết thời gian). "
    "Vui lòng thử lại với phạm vi hẹp hơn."
)

_V2_MISSING_TERMINAL_MESSAGE = "v2 turn ended without a terminal response"

_V2_MISSING_CLARIFICATION_MESSAGE = (
    "clarification is pending but no request was checkpointed"
)

_V2_UNEXPECTED_ERROR_MESSAGE = "Đã xảy ra lỗi khi xử lý yêu cầu."


class V2FreshTurnRequired(ValueError):
    """A clarification reply that can never satisfy its request.

    Raised by :func:`prepare_v2_resume_command` when the reply targets a
    candidate-free request (``ClarificationUnsatisfiable``): the caller MUST
    treat the reply as a brand-new user turn instead of retrying the resume
    (re-ask loops can never resolve it).
    """


class _V2StreamAccumulators:
    """Answer/artifact accumulators for one v2 stream.

    Mirrors the v1 ``stream_agent_events`` accumulator contract so the SSE
    vocabulary stays identical: ``token`` appends prose, artifact events
    replace their snapshot, and ``token_rollback`` clears EVERYTHING (the
    terminal ``complete`` must never carry pre-rollback artifacts).
    """

    def __init__(self) -> None:
        self.answer_text = ""
        self.sources: list = []
        self.images: list = []
        self.people_data: list = []
        self.potential_abbreviations: list = []

    def on_token(self, text: str) -> None:
        self.answer_text += text or ""

    def on_sources(self, sources: list) -> None:
        self.sources = list(sources or [])

    def on_images(self, images: list) -> None:
        self.images = list(images or [])

    def on_people_data(self, people: list) -> None:
        self.people_data = list(people or [])

    def on_potential_abbreviations(self, abbreviations: list) -> None:
        self.potential_abbreviations = list(abbreviations or [])

    def on_rollback(self) -> dict:
        """Clear every accumulator; returns the ``token_rollback`` event."""
        self.answer_text = ""
        self.sources = []
        self.images = []
        self.people_data = []
        self.potential_abbreviations = []
        return {"event": "token_rollback", "data": {}}


def _v2_config(thread_id: str) -> dict:
    """LangGraph config binding the run to its stable checkpoint thread."""
    return {"configurable": {"thread_id": thread_id}}


def _chunk_prose(text: str, size: int) -> list[str]:
    """Deterministically chunk prose for ``token`` events (no tokenizer)."""
    try:
        chunk_size = max(1, int(size or _V2_TOKEN_CHUNK_DEFAULT))
    except (TypeError, ValueError):
        chunk_size = _V2_TOKEN_CHUNK_DEFAULT
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _v2_terminal_event(response) -> tuple[str, dict]:
    """Map a terminal ``FinalResponse`` onto one ``(event, data)`` pair.

    Single funnel for the one-terminal rule (success/clarify -> ``complete``,
    denied/insufficient/error -> ``error``); see ``v2/events.py``.
    """
    from app.services.agents.v2.events import (
        terminal_event_for_response as _terminal_event,
    )

    return _terminal_event(_coerce_final_response(response))


def _coerce_final_response(value):
    """Re-validate a terminal response via the T6 checkpoint recipe.

    Accepts the live model or its checkpoint-serde mapping form; nested
    values revive as ``list``/``dict`` on real round-trips, so delegation
    to ``_coerce_slot`` (JSON round-trip rebuild) is the single owner.
    Raises ``TypeError`` when no terminal response is carried.
    """
    from app.services.agents.supervisor_v2 import _coerce_slot
    from app.services.agents.v2.contracts.response import FinalResponse

    try:
        coerced = _coerce_slot(value, FinalResponse, slot="final_response")
    except Exception as exc:
        raise TypeError(
            "terminal state carries no FinalResponse "
            f"(got {type(value).__name__}: {exc})"
        ) from exc
    if not isinstance(coerced, FinalResponse):
        raise TypeError(
            "terminal state carries no FinalResponse "
            f"(got {type(value).__name__})"
        )
    return coerced


def _coerce_clarification_request(value):
    """Re-validate a persisted request via the T6 checkpoint recipe.

    Accepts the live request or its checkpoint-serde mapping form (see
    :func:`_coerce_final_response`); unparseable values resolve to ``None``
    so the caller surfaces a typed error instead of walking a corrupt
    request.
    """
    from app.services.agents.supervisor_v2 import _coerce_slot
    from app.services.agents.v2.contracts.clarification import ClarificationRequest

    if value is None:
        return None
    try:
        coerced = _coerce_slot(value, ClarificationRequest, slot="clarification")
    except Exception:
        logger.warning("[v2stream] clarification coerce failed", exc_info=True)
        return None
    return coerced if isinstance(coerced, ClarificationRequest) else None


async def _v2_checkpoint_snapshot(graph, config: dict):
    """Load the latest checkpoint snapshot for ``config`` (or ``None``).

    langgraph 1.0.0 exposes sync ``get_state`` and async ``aget_state``;
    both shapes (plus awaitable ``get_state`` fakes) are accepted. Only
    operational load failures resolve to ``None`` — the caller decides.
    """
    import inspect

    try:
        aget_state = getattr(graph, "aget_state", None)
        if callable(aget_state):
            return await aget_state(config)
        get_state = getattr(graph, "get_state", None)
        if not callable(get_state):
            return None
        snapshot = get_state(config)
        if inspect.isawaitable(snapshot):
            return await snapshot
        return snapshot
    except Exception:
        logger.warning("[v2stream] checkpoint load failed", exc_info=True)
        return None


async def _v2_suspend_request(graph, config: dict, state: dict):
    """Detect a suspend turn and load its persisted request (or ``None``).

    Returns ``(suspended, request)``. Suspension is read from the RETURNED
    state (``__interrupt__`` present — the pinned stack suppresses top-level
    interrupts instead of raising) backed by the checkpoint (``next``
    carrying ``clarify_wait`` with a persisted request). A suspended turn is
    NOT terminal: no success/error terminal and no lease release follow.
    """
    values = state if isinstance(state, dict) else {}
    interrupts = values.get("__interrupt__") or ()
    try:
        has_interrupt = len(interrupts) > 0
    except TypeError:
        has_interrupt = bool(interrupts)
    snapshot = await _v2_checkpoint_snapshot(graph, config)
    snapshot_values: dict = {}
    next_nodes: tuple = ()
    if snapshot is not None:
        raw_values = getattr(snapshot, "values", None)
        if isinstance(raw_values, dict):
            snapshot_values = raw_values
        try:
            next_nodes = tuple(getattr(snapshot, "next", None) or ())
        except TypeError:
            next_nodes = ()
    checkpoint_request = _coerce_clarification_request(
        snapshot_values.get("clarification")
    )
    suspended = has_interrupt or (
        "clarify_wait" in next_nodes and checkpoint_request is not None
    )
    if not suspended:
        return False, None
    request = _coerce_clarification_request(values.get("clarification"))
    return True, request if request is not None else checkpoint_request


def _v2_potential_abbreviations(state: dict) -> list:
    """Project unknown abbreviation tokens from checkpointed semantics.

    Mirrors the v1 ``potential_abbreviations`` meaning (backend-identified
    tokens missing from the DB): finalized semantic abbreviations whose
    ``expansion`` is ``None``. Live models and serde mappings both accepted.
    """
    try:
        semantic = (state or {}).get("semantic")
        if semantic is None:
            return []
        if isinstance(semantic, dict):
            items = semantic.get("abbreviations") or ()
        else:
            items = getattr(semantic, "abbreviations", None) or ()
        out = []
        for item in items:
            if isinstance(item, dict):
                token, expansion = item.get("abbreviation"), item.get("expansion")
            else:
                token, expansion = (
                    getattr(item, "abbreviation", None),
                    getattr(item, "expansion", None),
                )
            if token and expansion is None and token not in out:
                out.append(token)
        return out
    except Exception:
        logger.warning("[v2stream] abbreviation projection failed", exc_info=True)
        return []


def _v2_terminal_is_error(state: dict) -> bool:
    """True when the turn already ended non-success (T7-owned rule).

    Decided by ``terminal_state_is_error`` (single owner); a state it cannot
    read falls back to the terminal-response status, never to validation.
    """
    try:
        from app.services.agent.runtime_selector import (
            terminal_state_is_error as _t7_terminal_is_error,
        )

        return bool(_t7_terminal_is_error(state))
    except Exception:
        logger.warning("[v2stream] terminal-error check failed", exc_info=True)
    try:
        final = (state or {}).get("final_response")
        if final is None:
            return True
        status = (
            final.get("status")
            if isinstance(final, dict)
            else getattr(final, "status", None)
        )
        return status != "success" and status != "clarify"
    except Exception:
        return True


async def _v2_checkpoint_values(graph, config: dict) -> dict:
    """Load the latest checkpoint values for ``config`` (mapping form)."""
    snapshot = await _v2_checkpoint_snapshot(graph, config)
    if snapshot is None:
        return {}
    values = getattr(snapshot, "values", None)
    if isinstance(values, dict):
        return values
    if isinstance(snapshot, dict):
        inner = snapshot.get("values", snapshot)
        if isinstance(inner, dict):
            return inner
    return {}


def _v2_execution_slots(state: dict) -> tuple:
    """Return ``(plan, task_results)`` from live or checkpoint-serde state."""
    execution = (state or {}).get("execution")
    if execution is None:
        return None, ()
    if isinstance(execution, dict):
        return execution.get("plan"), tuple(execution.get("task_results") or ())
    return getattr(execution, "plan", None), tuple(
        getattr(execution, "task_results", None) or ()
    )


def _v2_undispatched(state: dict) -> tuple:
    """Task ids in the checkpointed plan with no result yet (T3-N2 recipe).

    The T6 runtime-only truncation channel (``DispatchReport.truncated``) is
    deliberately never checkpointed, so the outer runner re-derives
    incompleteness here: a non-empty remainder means the plan was not fully
    executed and the turn must not present partial results as complete.
    """
    plan, results = _v2_execution_slots(state)
    if plan is None:
        return ()
    try:
        from app.services.agents.supervisor_v2 import (
            undispatched_tasks as _t6_undispatched_tasks,
        )

        return tuple(_t6_undispatched_tasks(plan, tuple(results or ())))
    except Exception:
        logger.warning("[v2stream] undispatched diff failed", exc_info=True)
    try:
        done = set()
        for item in results or ():
            task_id = (
                item.get("task_id")
                if isinstance(item, dict)
                else getattr(item, "task_id", None)
            )
            if task_id is not None:
                done.add(task_id)
        tasks = (
            plan.get("tasks", ())
            if isinstance(plan, dict)
            else getattr(plan, "tasks", None) or ()
        )
        remainder = []
        for task in tasks:
            task_id = (
                task.get("task_id")
                if isinstance(task, dict)
                else getattr(task, "task_id", None)
            )
            if task_id is not None and task_id not in done:
                remainder.append(task_id)
        return tuple(remainder)
    except Exception:
        logger.warning("[v2stream] fallback undispatched diff failed", exc_info=True)
        return ()


def _feed_plan_resolver_from_checkpoint(plan_resolver, checkpoint_state: dict) -> None:
    """Feed the resolver from the checkpointed plan/bindings (T7 seam).

    The resolver is never fed from request input or graph state — only the
    authoritative checkpointed plan plus its binding set.
    """
    if plan_resolver is None:
        return
    try:
        from app.services.agents.supervisor_v2 import _coerce_slot
        from app.services.agents.v2.contracts.binding import DocumentBindingSet
        from app.services.agents.v2.contracts.planning import TaskPlan

        plan, _ = _v2_execution_slots(checkpoint_state or {})
        bindings = (checkpoint_state or {}).get("bindings")
        if plan is None or bindings is None:
            return
        # Real round-trips revive nested contracts as list/dict: rebuild
        # the live tree before feeding (T6 recipe, single owner).
        plan = _coerce_slot(plan, TaskPlan, slot="plan")
        bindings = _coerce_slot(bindings, DocumentBindingSet, slot="bindings")
        feed = getattr(plan_resolver, "feed", None)
        if callable(feed):
            feed(plan, bindings)
    except Exception:
        logger.warning("[v2stream] plan resolver feed failed", exc_info=True)


async def _refresh_v2_resume_leases(*, runtime_context, checkpoint_state: dict) -> int:
    """Refresh existing run leases BEFORE the resume dispatch continues.

    Re-acquires the SAME ``(revision, use)`` rows the scheduler leased at
    dispatch (shared ``refresh_pairs_for_checkpoint`` recipe — no second
    lease logic here): expiry extended, release cleared. Best-effort
    hygiene: failures are logged, never fatal — the nodes fail closed on
    the leases they need.
    """
    try:
        services = getattr(runtime_context, "services", None)
        repo = getattr(services, "retention_leases", None)
        if repo is None:
            return 0
        run_id = runtime_context.capability_runtime.run_id
    except Exception:
        logger.warning("[v2stream] lease refresh setup failed", exc_info=True)
        return 0
    try:
        from app.services.agents.v2.execution.scheduler import (
            refresh_pairs_for_checkpoint as _pairs_for_checkpoint,
        )

        plan, results = _v2_execution_slots(checkpoint_state or {})
        bindings = (checkpoint_state or {}).get("bindings")
        pairs = _pairs_for_checkpoint(plan, bindings, results)
    except Exception:
        logger.warning("[v2stream] lease refresh pairs failed", exc_info=True)
        return 0
    refreshed = 0
    for revision_id, use_id in pairs:
        try:
            await repo.acquire_or_refresh(run_id, revision_id, use_id)
            refreshed += 1
        except Exception:
            logger.warning("[v2stream] lease refresh failed", exc_info=True)
    if refreshed:
        try:
            session = getattr(repo, "session", None)
            commit = getattr(session, "commit", None)
            if callable(commit):
                await commit()
        except Exception:
            logger.warning("[v2stream] lease refresh commit failed", exc_info=True)
    return refreshed


async def _release_v2_run_leases(*, runtime_context, reason: str) -> int:
    """Release the run's leases (terminal boundary only; never raises)."""
    try:
        run_id = str(runtime_context.capability_runtime.run_id)
    except Exception:
        run_id = ""
    if run_id:
        # Awaited terminal cleanup (R75 lifecycle): the distributed
        # registration is removed before this boundary returns, so a
        # resumed run re-registers cleanly. Falls back to the sync
        # wrapper when awaiting fails.
        try:
            from app.services.agents.v2.execution.scheduler import (
                unregister_active_run as _unregister_sync,
            )
            from app.services.agents.v2.execution.scheduler import (
                unregister_active_run_async as _unregister_active_run,
            )

            try:
                await _unregister_active_run(run_id)
            except Exception:
                _unregister_sync(run_id)
        except Exception:
            logger.warning("[v2stream] active-run unregister failed", exc_info=True)
    try:
        services = getattr(runtime_context, "services", None)
        repo = getattr(services, "retention_leases", None)
        if repo is None:
            return 0
        run_id = runtime_context.capability_runtime.run_id
        rowcount = await repo.release_run(run_id, reason)
        try:
            session = getattr(repo, "session", None)
            commit = getattr(session, "commit", None)
            if callable(commit):
                await commit()
        except Exception:
            logger.warning("[v2stream] lease release commit failed", exc_info=True)
        return int(rowcount or 0)
    except Exception:
        logger.warning("[v2stream] lease release failed", exc_info=True)
        return 0


async def prepare_v2_resume_command(
    *,
    message_id,
    request,
    runtime_context,
):
    """Build the verbatim resume ``Command`` for a clarification reply.

    Loads T5's ``resume_clarification`` output and returns it untouched —
    the graph owns navigation. Whether the reply is answerable is decided
    by T7's ``clarification_reply_is_fresh_turn`` (single owner): a
    candidate-free request raises ``V2FreshTurnRequired`` and the caller
    must treat the reply as a new turn, never retry the resume.
    """
    from app.services.agent.runtime_selector import (
        clarification_reply_is_fresh_turn as _is_fresh_turn,
    )
    from app.services.agents.v2.nodes.clarification import resume_clarification

    request = _coerce_clarification_request(request)
    if request is None:
        from app.services.agents.v2.nodes.clarification import ClarificationError

        raise ClarificationError(
            "persisted clarification request does not re-validate; "
            "refusing to resume from it"
        )
    try:
        return await resume_clarification(message_id, request, runtime_context)
    except Exception as exc:
        if _is_fresh_turn(exc):
            raise V2FreshTurnRequired(
                "clarification offers no selectable candidate; "
                "treat the reply as a new turn, do not retry the resume"
            ) from exc
        raise


async def load_v2_pending_clarification(graph, thread_id: str):
    """Return the suspended ``ClarificationRequest`` for ``thread_id``.

    ``None`` when the thread is not suspended in ``clarify_wait`` (fresh
    turn, terminal thread, or missing checkpoint). Entrypoints call this
    BEFORE persisting-driven resume so the reply path is explicit.
    """
    snapshot = await _v2_checkpoint_snapshot(graph, _v2_config(thread_id))
    if snapshot is None:
        return None
    try:
        next_nodes = tuple(getattr(snapshot, "next", None) or ())
    except TypeError:
        return None
    if "clarify_wait" not in next_nodes:
        return None
    values = getattr(snapshot, "values", None)
    if not isinstance(values, dict):
        return None
    return _coerce_clarification_request(values.get("clarification"))


async def resolve_v2_resume_command(
    *,
    graph,
    thread_id: str,
    message_id,
    runtime_context,
):
    """Resolve this turn's resume ``Command`` (or ``None`` for a fresh turn).

    ``None`` when no clarification is pending, when no reply message id is
    available, or when the reply must start a fresh turn
    (``V2FreshTurnRequired``). Any other clarification failure also falls
    back to a fresh turn (logged): the new turn always emits its own
    terminal, so the stream keeps the one-terminal guarantee instead of
    surfacing a resume-plumbing error for a user message.
    """
    from app.services.agents.v2.nodes.clarification import ClarificationError

    if message_id is None:
        return None
    pending = await load_v2_pending_clarification(graph, thread_id)
    if pending is None:
        return None
    try:
        return await prepare_v2_resume_command(
            message_id=message_id,
            request=pending,
            runtime_context=runtime_context,
        )
    except V2FreshTurnRequired:
        logger.info(
            "[v2stream] clarification reply starts a fresh turn "
            "(thread %s)",
            thread_id,
        )
        return None
    except ClarificationError as exc:
        logger.warning(
            "[v2stream] clarification resume unusable (%s); fresh turn",
            exc,
        )
        return None


async def persisted_message_uuid(db, row):
    """Return a just-persisted chat row's UUID primary key (or ``None``).

    ``persist_raw_user_message`` commits (expiring attributes), so refresh
    explicitly on the same session before reading ``row.id`` — the resume
    path needs the ``ChatMessage.id`` form, never the client message id.
    """
    if row is None:
        return None
    try:
        await db.refresh(row)
        return row.id
    except Exception:
        logger.warning("[v2stream] message id refresh failed", exc_info=True)
        return None


async def stream_v2_turn_events(
    *,
    graph,
    runtime_context,
    thread_id: str,
    initial_state: dict | None = None,
    resume_command=None,
    plan_resolver=None,
    token_chunk_size: int = _V2_TOKEN_CHUNK_DEFAULT,
):
    """Run one v2 turn and yield SSE-compatible dict events (see module note).

    Exactly one of ``initial_state`` (first turn) or ``resume_command`` (a
    verbatim :func:`prepare_v2_resume_command` ``Command``) is required.
    ``config`` binds the stable ``thread_id``; ``context`` always carries
    the caller-supplied ``runtime_context`` so the CURRENT ACL replaces
    historical values on resume. Emits exactly one terminal event:
    ``complete`` for success/clarify suspensions and successes, ``error``
    otherwise (including unexpected graph failures — a raw exception never
    escapes). Suspension (``__interrupt__`` / ``clarify_wait``) is NOT
    terminal: the question surfaces as ``complete`` with leases kept.
    """
    from langgraph.errors import GraphInterrupt

    from app.services.agents.v2.execution.scheduler import (
        V1FallbackRequired as _V1FallbackRequired,
    )

    if (initial_state is None) == (resume_command is None):
        raise ValueError("exactly one of initial_state/resume_command is required")
    config = _v2_config(thread_id)
    acc = _V2StreamAccumulators()
    invoke_task = None
    released = False
    try:
        _run_id = str(runtime_context.capability_runtime.run_id or "")
    except Exception:
        _run_id = ""
    if _run_id:
        # Register at run start (awaited, R75 lifecycle): the distributed
        # registry owns the run for its whole lifetime; per-dispatch
        # guards refresh it and terminal release removes it.
        try:
            from app.services.agents.v2.execution.scheduler import (
                register_active_run_async as _register_at_start,
            )

            await _register_at_start(_run_id)
        except Exception:
            logger.warning("[v2stream] active-run register failed", exc_info=True)

    async def _release_once(reason: str) -> int:
        nonlocal released
        if released:
            return 0
        released = True
        return await _release_v2_run_leases(
            runtime_context=runtime_context, reason=reason
        )

    async def _emit_suspend_turn(pending) -> AsyncGenerator[dict, None]:
        # Shared by the returned-state and nested-raise suspend paths:
        # the question arrives whole in the single terminal ``complete``
        # (tokens stream for success terminals only — see below); leases
        # stay active.
        yield {"event": "status", "data": {"step": "generating", "detail": "clarification pending"}}
        event, data = _v2_terminal_event(
            _coerce_final_response(
                {
                    "contract_version": pending.contract_version,
                    "status": "clarify",
                    "content": pending.question,
                    "citations": (),
                }
            )
        )
        yield {"event": event, "data": data}

    try:
        if resume_command is not None:
            yield {"event": "status", "data": {"step": "analyzing", "detail": "Resuming v2 graph..."}}
            checkpoint_state = await _v2_checkpoint_values(graph, config)
            await _refresh_v2_resume_leases(
                runtime_context=runtime_context,
                checkpoint_state=checkpoint_state,
            )
            _feed_plan_resolver_from_checkpoint(plan_resolver, checkpoint_state)
            payload = resume_command
        else:
            yield {"event": "status", "data": {"step": "analyzing", "detail": "Running v2 graph..."}}
            payload = initial_state
        invoke_task = asyncio.create_task(
            graph.ainvoke(payload, config, context=runtime_context)
        )
        try:
            result = await invoke_task
        except GraphInterrupt:
            # Nested-graph compat: top-level suspends RETURN ``__interrupt__``
            # (handled below); a raise only escapes from nested graphs.
            result = {"__interrupt__": (True,)}
        state = result if isinstance(result, dict) else {}
        suspended, pending = await _v2_suspend_request(graph, config, state)
        if suspended:
            # Clarify suspend: the suspension checkpoint keeps the persisted
            # request AND the active leases (never released here). Surface
            # the question as the turn's single terminal ``complete``.
            if pending is None:
                yield {"event": "error", "data": {"message": _V2_MISSING_CLARIFICATION_MESSAGE}}
                return
            async for ev in _emit_suspend_turn(pending):
                yield ev
            return
        remaining = _v2_undispatched(state)
        if remaining:
            # Deadline-truncated dispatch: never present partial results as
            # complete — clear speculative state, then surface the error.
            logger.warning(
                "[v2stream] truncated dispatch: %d task(s) undispatched",
                len(remaining),
            )
            yield acc.on_rollback()
            yield {"event": "error", "data": {"message": _V2_TRUNCATION_MESSAGE}}
            await _release_once("terminal")
            return
        try:
            final = _coerce_final_response(state.get("final_response"))
        except (TypeError, ValueError):
            logger.error("[v2stream] turn ended without a terminal response")
            yield {"event": "error", "data": {"message": _V2_MISSING_TERMINAL_MESSAGE}}
            await _release_once("terminal")
            return
        event, data = _v2_terminal_event(final)
        if event == "complete" and not _v2_terminal_is_error(state):
            # Success only (T7-owned rule): chunk the terminal content into
            # ``token`` events, then emit the single terminal. Clarify
            # terminals (non-success ``complete``) carry the question in the
            # payload itself — no speculative prose precedes them.
            yield {"event": "status", "data": {"step": "generating", "detail": "Streaming v2 answer..."}}
            for chunk in _chunk_prose(data.get("answer", ""), token_chunk_size):
                acc.on_token(chunk)
                yield {"event": "token", "data": {"text": chunk}}
            for abbreviation in _v2_potential_abbreviations(state):
                acc.on_potential_abbreviations([*acc.potential_abbreviations, abbreviation])
            if acc.potential_abbreviations:
                yield {
                    "event": "potential_abbreviations",
                    "data": {"abbreviations": list(acc.potential_abbreviations)},
                }
            await _release_once("terminal")
            yield {"event": event, "data": data}
            return
        await _release_once("terminal")
        yield {"event": event, "data": data}
    except GeneratorExit:
        # Client disconnect mid-run: stop the graph task, never succeed.
        if invoke_task is not None and not invoke_task.done():
            invoke_task.cancel()
        await _release_once("cancelled")
        raise
    except _V1FallbackRequired:
        # Task 7B canary fallback: the v2 candidate resolved to a v1-only
        # route BEFORE any capability execution. Release the (empty) run
        # and re-raise typed — the entrypoint serves v1 with zero v2
        # user-visible output (no terminal event is emitted here).
        if invoke_task is not None and not invoke_task.done():
            invoke_task.cancel()
        await _release_once("terminal")
        raise
    except asyncio.CancelledError:
        # Cancellation prevents all later dispatch and any factual success:
        # no terminal event, leases released as cancelled, error propagates.
        if invoke_task is not None and not invoke_task.done():
            invoke_task.cancel()
        await _release_once("cancelled")
        raise
    except Exception as exc:
        # Unexpected graph failure (I1): like v1, convert to a terminal
        # ``error`` instead of letting a raw exception escape the SSE
        # stream — the frontend would otherwise render a silent partial
        # answer. Leases still release (the run is over), exactly once.
        logger.error("[v2stream] unexpected v2 turn failure: %s", exc, exc_info=True)
        if invoke_task is not None and not invoke_task.done():
            invoke_task.cancel()
        yield acc.on_rollback()
        yield {"event": "error", "data": {"message": _V2_UNEXPECTED_ERROR_MESSAGE}}
        await _release_once("terminal")
        return


async def stream_v2_turn_to_sse(
    *,
    graph,
    runtime_context,
    thread_id: str,
    initial_state: dict | None = None,
    resume_command=None,
    plan_resolver=None,
    token_chunk_size: int = _V2_TOKEN_CHUNK_DEFAULT,
):
    """SSE wrapper around :func:`stream_v2_turn_events` (v1 wire format)."""
    async for ev in stream_v2_turn_events(
        graph=graph,
        runtime_context=runtime_context,
        thread_id=thread_id,
        initial_state=initial_state,
        resume_command=resume_command,
        plan_resolver=plan_resolver,
        token_chunk_size=token_chunk_size,
    ):
        yield _sse(ev["event"], ev["data"])

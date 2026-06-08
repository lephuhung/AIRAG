"""
User Clarification Helper
=========================

V2.1 — Nguyên tắc "Ask User When Uncertain":
Khi hệ thống không chắc chắn về ý định user, **hỏi lại user** thay vì đoán và trả lời sai.

Cung cấp helper thống nhất `ask_user_clarification()` thay vì mỗi node tự build
clarification message riêng. Frontend nhận event `clarification` với format:
    {
        "message": str,       # Câu hỏi cho user
        "options": list[str], # Danh sách lựa chọn (optional)
        "context": dict,      # Metadata để frontend track
    }

Frontend hiển thị câu hỏi, user reply, sau đó gửi lại query kèm context.

Usage:
    from app.services.agents.clarification import ask_user_clarification

    await ask_user_clarification(
        state,
        question="Bạn muốn tra cứu văn bản nào?",
        options=["Luật An ninh mạng 2018", "Luật An toàn thông tin 2015"],
        context={"type": "ambiguous_doc", "candidates": [...]},
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agents.models import SupervisorState

logger = logging.getLogger(__name__)


async def ask_user_clarification(
    state: SupervisorState,
    *,
    question: str,
    options: list[str] | None = None,
    context: dict | None = None,
) -> None:
    """Emit a structured clarification event for the frontend.

    Args:
        state: Current SupervisorState (for contextvar lookup of event queue).
        question: Human-readable question to display to the user (in Vietnamese).
        options: Optional list of pre-canned answers. If empty, user must type.
        context: Optional metadata for frontend to track this clarification
                 (e.g., {"type": "ambiguous_doc", "candidates": [...]}).

    Behavior:
        - Pushes "clarification" SSE event to frontend
        - Sets state["clarification_needed"] = True
        - Sets state["clarification_message"] = question
        - Logs the clarification request for observability
    """
    from app.services.agent.streaming import push_event

    options = options or []
    context = context or {}

    # Update state so downstream nodes know user clarification was requested
    state["clarification_needed"] = True
    state["clarification_message"] = question

    # Emit SSE event
    await push_event(state, "clarification", {
        "message": question,
        "options": options,
        "context": context,
    })

    logger.info(
        f"[clarification] asked user: {question[:100]!r} "
        f"(options={len(options)}, context_keys={list(context.keys())})"
    )


def should_ask_for_doc_reference(query: str, has_named_doc_match: bool) -> bool:
    """Decide whether to ask user for a document reference.

    Used by supervisor's _REQUIRES_DOC_INTENTS path when intent is summarize
    or search_section but no named doc regex matched AND LLM didn't return
    a task_plan with resolve_doc.

    Returns True if we should ask the user instead of guessing.
    """
    # Heuristic: if query mentions "văn bản", "tài liệu", "nghị định" etc.
    # but regex didn't match a specific name, ask the user.
    indicators = ("văn bản", "tài liệu", "nghị định", "thông tư", "luật", "quyết định")
    query_lower = query.lower()
    has_doc_indicator = any(ind in query_lower for ind in indicators)
    return has_doc_indicator and not has_named_doc_match


def should_ask_for_section_reference(extracted_text: str, min_chars: int = 50) -> bool:
    """Decide whether to ask user for a section reference.

    Used by _extract_section_from_markdown when section extraction returns
    empty or too short — the section may not exist or the reference may
    be malformed. Better to ask than to return wrong content.
    """
    return len(extracted_text.strip()) < min_chars

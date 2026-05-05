"""
Write Agent
===========

Single-file agent handling text processing operations:
- summarize      → summarize provided text
- suggest_edits  → editing suggestions
- grammar_check  → grammar/style checking
- format_check   → Word document format checking

Each operation is a prompt-based LLM call.

Prompts moved to: app/prompts/agents/write_agent_prompt.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.prompts.agents import write_agent_prompt

if TYPE_CHECKING:
    from app.services.agents.models import SupervisorState

logger = logging.getLogger(__name__)


def _load_30_standard_from_file() -> str:
    """Load 30/2020/NĐ-CP standards from prompt module."""
    return write_agent_prompt._load_30_standard_from_file()


async def _get_30_2020_standard(state: SupervisorState) -> str:
    """
    Load 30/2020/NĐ-CP standards from static file.
    No RAG dependency - works even when documents are not indexed.
    """
    return _load_30_standard_from_file()


# =============================================================================
# Write Prompts
# =============================================================================

# Re-export from centralized prompts for backward compatibility
WRITE_PROMPTS = write_agent_prompt.WRITE_PROMPTS
FORMAT_CHECK_PROMPT = write_agent_prompt.FORMAT_CHECK_PROMPT
FALLBACK_STANDARD = write_agent_prompt.FALLBACK_STANDARD


# =============================================================================
# Write Agent Node
# =============================================================================

async def write_agent_node(state: SupervisorState) -> dict:
    """
    Execute write operation based on write_action.

    Flow:
    1. Determine write action (summarize/suggest_edits/grammar_check/format_check)
    2. Fetch format data or document content if missing
    3. Get text input (from state.text_input or kg_summaries)
    4. Call appropriate LLM
    5. Return final answer
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage
    from app.services.agent.nodes import strip_thinking_tags
    from app.services.agents.models import AgentType
    from app.services.agent.streaming import push_event

    intent = state.get("intent", "")
    _intent_to_action = {
        "write_summarize": "summarize",
        "write_suggest_edits": "suggest_edits",
        "write_grammar_check": "grammar_check",
        "write_format_check": "format_check",
    }
    
    write_action = state.get("write_action")
    if not write_action and intent in _intent_to_action:
        write_action = _intent_to_action[intent]
        state["write_action"] = write_action
    if not write_action:
        write_action = "summarize"

    # Pre-processing for format_check
    if write_action == "format_check" and not state.get("format_data"):
        doc_ids = state.get("document_ids") or []
        if doc_ids:
            try:
                from app.services.agent.streaming import get_current_db
                from app.services.agent import tools as _agent_tools
                db = get_current_db()
                tool_result = await _agent_tools.get_document_format(document_ids=doc_ids, db=db)
                docs_with_format = tool_result.get("documents", [])
                if docs_with_format:
                    first_doc = docs_with_format[0]
                    if first_doc.get("format_data"):
                        state["format_data"] = first_doc["format_data"]
                        state["file_name"] = first_doc.get("filename", "tài liệu")
            except Exception as e:
                logger.warning(f"[write_agent_node] Failed to fetch format metadata: {e}")

    # Pre-processing for text intents
    text_input = state.get("text_input", "")
    if write_action in {"summarize", "suggest_edits", "grammar_check"} and not text_input:
        doc_ids = state.get("document_ids") or []
        if doc_ids:
            try:
                from app.services.agent.streaming import get_current_db
                from app.services.agent import tools as _agent_tools
                db = get_current_db()
                content_result = await _agent_tools.get_documents_content(document_ids=doc_ids, db=db)
                docs = content_result.get("documents", [])
                combined = "\n\n---\n\n".join(f"**{d['filename']}**\n\n{d['content']}" for d in docs if d.get("content"))
                if combined:
                    text_input = combined
                    state["text_input"] = combined
            except Exception as e:
                logger.warning(f"[write_agent_node] Failed to fetch doc content: {e}")
        else:
            import re
            user_msg = state.get("original_query") or state.get("rewritten_query", "")
            cleaned = re.sub(r"^(tóm tắt|summarize|kiểm tra ngữ pháp|grammar check|đề xuất chỉnh sửa|suggest edits)[:\s]+", "", user_msg, flags=re.IGNORECASE).strip()
            if cleaned:
                text_input = cleaned
                state["text_input"] = cleaned

    logger.info(f"[write_agent] action={write_action!r}, text_input_len={len(text_input)}")

    # Emit status so frontend shows progress immediately
    await push_event(state, "status", {"step": "generating", "detail": "Đang xử lý văn bản..."})

    # Handle format_check separately
    if write_action == "format_check":
        return await _handle_format_check(state)

    # Guard: need text input
    if not text_input:
        # Try to get from kg_summaries (for summarize intent from RAG)
        kg_summaries = state.get("kg_summaries", [])
        if kg_summaries:
            text_input = kg_summaries[0]
        else:
            return {
                "final_answer": "Vui lòng cung cấp văn bản để xử lý.",
                "next_agent": AgentType.FINISH,
            }

    # Get prompt
    prompt_template = WRITE_PROMPTS.get(write_action, WRITE_PROMPTS["summarize"])
    prompt = prompt_template.format(text=text_input)

    # Call LLM
    provider = get_llm_provider()
    answer_parts = []

    try:
        async for chunk in provider.astream(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=2048,
        ):
            if chunk.type == "text" and chunk.text:
                answer_parts.append(chunk.text)
                # Emit token for realtime SSE streaming
                await push_event(state, "token", chunk.text)
    except Exception as e:
        logger.error(f"[write_agent] LLM call failed: {e}")
        return {
            "final_answer": f"Lỗi xử lý văn bản: {e}",
            "next_agent": AgentType.FINISH,
        }

    final_answer = strip_thinking_tags("".join(answer_parts))

    return {
        "final_answer": final_answer,
        "next_agent": AgentType.FINISH,
        "iterations": state.get("iterations", 0) + 1,
    }


async def _handle_format_check(state: SupervisorState) -> dict:
    """
    Handle Word document format checking with RAG-based evaluation.

    Flow (Option A + B):
    1. Extract format metadata from docx (unchanged)
    2. Call RAG to get standard from 30/2020/NĐ-CP (NEW)
    3. Build comprehensive evaluation prompt with both
    4. LLM evaluates and returns detailed report
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage
    from app.services.agent.nodes import strip_thinking_tags
    from app.services.agents.models import AgentType
    from app.services.agent.streaming import push_event

    format_data = state.get("format_data") or {}
    file_name = state.get("file_name", "tài liệu")

    # Emit status
    await push_event(state, "status", {"step": "generating", "detail": "Đang kiểm tra định dạng..."})

    if not format_data:
        return {
            "final_answer": (
                "Không có dữ liệu định dạng để kiểm tra.\n\n"
                "Vui lòng đính kèm một file Word (.docx) và hỏi lại."
            ),
            "next_agent": AgentType.FINISH,
        }

    # Extract format info
    margins = format_data.get("margins", {})
    page_size = format_data.get("page_size", {})
    font_samples = format_data.get("font_samples", [])
    line_spacing_data = format_data.get("line_spacing", [])

    # Build font sizes info
    from collections import Counter
    raw_font_sizes = [f.get("font_size") for f in font_samples if f.get("font_size")]
    size_counts = Counter(raw_font_sizes)
    most_common_sizes = size_counts.most_common(5)

    # Build font list - deduplicate and limit
    unique_fonts = []
    seen = set()
    for f in font_samples:
        name = f.get("font_name")
        if name and name not in seen:
            seen.add(name)
            unique_fonts.append(name)

    # Analyze line spacing
    spacing_types = {}
    for ls in line_spacing_data:
        ls_type = ls.get("line_spacing_type")
        ls_val = ls.get("line_spacing_value")
        if ls_type and ls_val is not None:
            key = f"{ls_type}_{ls_val}"
            spacing_types[key] = spacing_types.get(key, 0) + 1

    if spacing_types:
        most_common_spacing = max(spacing_types, key=spacing_types.get)
        line_spacing_display = f"{most_common_spacing} (gặp {max(spacing_types.values())} lần)"
    else:
        line_spacing_display = "Không trích xuất được"

    # Step 1: Get standard from file (no RAG dependency)
    logger.info("[_handle_format_check] Loading 30/2020/NĐ-CP standards from file...")
    standard_content = await _get_30_2020_standard(state)
    logger.info(f"[_handle_format_check] Got standard content: {len(standard_content)} chars")

    # Step 2: Build comprehensive evaluation prompt with standard
    prompt = FORMAT_CHECK_PROMPT.format(
        file_name=file_name,
        page_width=page_size.get("width", "N/A"),
        page_height=page_size.get("height", "N/A"),
        top=margins.get("top", "N/A"),
        bottom=margins.get("bottom", "N/A"),
        left=margins.get("left", "N/A"),
        right=margins.get("right", "N/A"),
        font_sizes="\n".join([f"- {s}pt ({c} lần)" for s, c in most_common_sizes]) if most_common_sizes else "- Không trích xuất được",
        fonts=", ".join(unique_fonts[:5]) if unique_fonts else "Không trích xuất được",
        line_spacing=line_spacing_display,
        paragraph_count=format_data.get("paragraph_count", 0),
        table_count=format_data.get("table_count", 0),
        standard_content=standard_content,
    )

    # Call LLM
    provider = get_llm_provider()
    answer_parts = []

    try:
        async for chunk in provider.astream(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=2048,
        ):
            if chunk.type == "text" and chunk.text:
                answer_parts.append(chunk.text)
                # Emit token for realtime SSE streaming
                await push_event(state, "token", chunk.text)
    except Exception as e:
        logger.error(f"[write_agent:format_check] LLM call failed: {e}")
        return {
            "final_answer": f"Lỗi kiểm tra định dạng: {e}",
            "next_agent": AgentType.FINISH,
        }

    final_answer = strip_thinking_tags("".join(answer_parts))

    return {
        "final_answer": final_answer,
        "next_agent": AgentType.FINISH,
        "iterations": state.get("iterations", 0) + 1,
        "format_evaluation_result": {
            "is_valid": "đạt" in final_answer.lower() or "phù hợp" in final_answer.lower(),
            "overall_status": (
                "đạt" if "đạt" in final_answer.lower()
                else "yếu" if "yếu" in final_answer.lower()
                else "cần cải thiện"
            ),
            "standard_reference": "30/2020/NĐ-CP",
        },
    }

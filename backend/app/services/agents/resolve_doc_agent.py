"""
Resolve Doc Agent
=================

Phase 2: Dedicated agent for resolving ambiguous document references.

This module is a THIN LangGraph adapter around the shared resolution core in
``app/services/agent/doc_resolver.py`` (:func:`resolve_candidates`). The same
core powers the ReAct path (``tools.py :: resolve_document_reference``) so the
matching logic never diverges between the two agent backends.

Responsibilities kept here (and NOT in the core):
- SSE status streaming (push_event) + Langfuse spans
- Mapping ranked candidates → SupervisorState routing decisions
  (early-exit / not-found / ambiguous / clear-winner / confidence thresholds)

Graph position:
    supervisor (intent=resolve_doc) → resolve_doc_agent → answer_generator / END
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.services.agent.doc_resolver import resolve_candidates
from app.services.agent.langfuse_tracing import _get_langfuse_client

if TYPE_CHECKING:
    from app.services.agents.models import SupervisorState

logger = logging.getLogger(__name__)

# Score thresholds shared across routing decisions
EARLY_EXIT_THRESHOLD = 0.85       # clear DB winner → resolve immediately
HIGH_CONFIDENCE_THRESHOLD = 0.60  # accept & scope search to this document
MEDIUM_CONFIDENCE_THRESHOLD = 0.30  # ask the user to confirm which document
AMBIGUITY_RATIO = 0.75            # 2nd candidate ≥ 75% of top → ambiguous


def _emit_resolve_span(reference: str, workspace_ids: list, output: dict) -> None:
    """Best-effort Langfuse observation for one resolve_doc_agent run."""
    langfuse = _get_langfuse_client()
    if not langfuse:
        return
    try:
        obs = langfuse.start_observation(
            name="resolve_doc_agent",
            input={"reference": reference, "workspace_ids": [str(w) for w in workspace_ids]},
            level="DEFAULT",
        )
        obs.update(output=output)
        obs.end()
    except Exception as e:
        logger.warning(f"[langfuse] resolve_doc_agent span failed: {e}")


# =============================================================================
# Resolve Doc Agent Node
# =============================================================================

async def resolve_doc_agent_node(state: "SupervisorState") -> dict:
    """
    Phase 2: Multi-strategy document resolution agent.

    Delegates extraction/query/scoring to the shared core, then routes:
    - 0 results       → stream "not found" (or similar-docs) hint → END
    - Ambiguous       → stream clarification options → END
    - Clear winner    → answer_generator (or rag if section_ref present)
    """
    from app.services.agents.models import AgentType
    from app.services.agent.streaming import push_event, get_current_db

    reference = state.get("rewritten_query", "") or state.get("original_query", "")
    workspace_ids = state.get("workspace_ids", [])
    db = get_current_db()

    logger.info(f"[LANGGRAPH_NODE] Entering resolve_doc_agent_node, reference={reference!r}, workspace_ids={workspace_ids}")
    logger.info(f"[resolve_doc_agent] START reference={reference!r}")
    await push_event(state, "status", {
        "step": "searching",
        "detail": "Đang xác định văn bản...",
    })

    async def _status_cb(detail: str) -> None:
        await push_event(state, "status", {"step": "searching", "detail": detail})

    # ── Run the shared resolution core ───────────────────────────────────────
    # reference IS the full query here, so pass it as topic too — the question's
    # subject content disambiguates same-number documents during rerank.
    res = await resolve_candidates(
        reference, workspace_ids, db, topic=reference,
        use_llm_fallback=True, status_cb=_status_cb,
    )
    all_candidates = res["candidates"]
    parsed = res["parsed"]
    pending_intent = state.get("pending_intent")

    # ── Early exit: clear high-confidence winner ─────────────────────────────
    if all_candidates and all_candidates[0]["score"] >= EARLY_EXIT_THRESHOLD:
        top = all_candidates[0]
        logger.info(
            f"[resolve_doc_agent] EARLY EXIT — high-confidence hit: "
            f"doc_id={top['document_id']}, score={top['score']:.2f}"
        )
        _emit_resolve_span(reference, workspace_ids, {
            "outcome": "early_exit_db",
            "document_id": top["document_id"],
            "title": top.get("title"),
            "score": top["score"],
            "strategy": top.get("strategy", "db_query"),
        })
        return _build_resolved_state(top, all_candidates, parsed, pending_intent)

    # ── No candidates → similar-doc suggestion or not-found ──────────────────
    if not all_candidates:
        similar_docs = res.get("similar", [])
        if similar_docs:
            similar_lines = [
                "\n\nKhông tìm thấy văn bản chính xác, nhưng có văn bản có tên tương tự:"
            ]
            for i, doc in enumerate(similar_docs[:5], 1):
                title = doc.get("title", "")
                num = doc.get("document_number", "")
                date = doc.get("published_date", "")
                matched = ", ".join(doc.get("matched_keywords", [])[:4])
                line = f"{i}. **{title}**"
                if num:
                    line += f" (Số: {num})"
                if date:
                    line += f" - {date}"
                if matched:
                    line += f"\n   Từ khóa tương tự: {matched}"
                similar_lines.append(line)
            similar_lines.append("\nBạn có đang tìm một trong các văn bản trên không?")
            msg = "\n".join(similar_lines)
            logger.info(f"[LANGGRAPH_DECISION] resolve_doc_agent: no exact match, found {len(similar_docs)} similar documents")
        else:
            msg = (
                f"Không tìm thấy văn bản phù hợp với **\"{reference}\"** trong kho tài liệu.\n\n"
                "Bạn có thể:\n"
                "- Cung cấp số văn bản chính xác (ví dụ: 53/2022/NĐ-CP)\n"
                "- Cung cấp tên đầy đủ của văn bản\n"
                "- Kiểm tra xem văn bản đã được tải lên chưa"
            )
            logger.info("[LANGGRAPH_DECISION] resolve_doc_agent decision: no candidates found")

        await push_event(state, "token", msg)
        _emit_resolve_span(reference, workspace_ids, {"outcome": "not_found"})
        return {
            "final_answer": msg,
            "next_agent": AgentType.FINISH,
            "sources": [],
        }

    top = all_candidates[0]
    top_score = top.get("score", 0.0)

    # ── Ambiguity: 2nd candidate close to the top ────────────────────────────
    is_ambiguous = (
        len(all_candidates) > 1
        and (all_candidates[1].get("score", 0) / max(top_score, 0.01)) >= AMBIGUITY_RATIO
    )
    if is_ambiguous:
        options = []
        for i, c in enumerate(all_candidates[:5], 1):
            title = c.get("title") or f"Văn bản {i}"
            num = c.get("document_number", "")
            label = f"**{i}. {title}**"
            if num:
                label += f" (Số: {num})"
            options.append(label)

        clarify_msg = (
            f"Tìm thấy **{len(options)} văn bản** có thể phù hợp với "
            f"**\"{reference}\"**:\n\n"
            + "\n".join(options)
            + "\n\nBạn muốn tra cứu văn bản nào? "
            "Vui lòng chỉ định số thứ tự hoặc cung cấp thêm thông tin."
        )
        logger.info(f"[LANGGRAPH_DECISION] resolve_doc_agent decision: ambiguous ({len(all_candidates)} candidates)")
        await push_event(state, "clarification", {"message": clarify_msg})
        await push_event(state, "token", clarify_msg)
        _emit_resolve_span(reference, workspace_ids, {
            "outcome": "ambiguous",
            "candidates_count": len(all_candidates),
            "top_score": top_score,
        })
        return {
            "final_answer": clarify_msg,
            "next_agent": AgentType.FINISH,
        }

    # ── Single clear winner ──────────────────────────────────────────────────
    strategies_used = ", ".join(top.get("strategies", [top.get("strategy", "?")]))
    _emit_resolve_span(reference, workspace_ids, {
        "outcome": "resolved",
        "document_id": top.get("document_id"),
        "title": top.get("title"),
        "score": top_score,
        "strategies": strategies_used,
        "section_reference": top.get("section_reference") or parsed.get("section_reference"),
    })
    return _build_resolved_state(top, all_candidates, parsed, pending_intent)


def _build_medium_confidence_state(top: dict, all_candidates: list, parsed: dict, pending_intent: str | None = None) -> dict:
    """
    Build state for medium confidence matches (score 0.30 - 0.59).
    Shows all similar documents and asks user to confirm which one they want.
    """
    from app.services.agents.models import AgentType

    top_score = top.get("score", 0.0)
    section_ref = top.get("section_reference") or parsed.get("section_reference") or ""

    logger.info(
        f"[LANGGRAPH_DECISION] resolve_doc_agent MEDIUM confidence → top_score={top_score:.2f}, "
        f"total_candidates={len(all_candidates)}"
    )

    # Show all candidates (not just top one) for user to choose
    lines = [
        f"Tìm thấy **{len(all_candidates)} văn bản** có thể phù hợp:"
    ]
    for i, c in enumerate(all_candidates[:5], 1):
        title = c.get("title", "văn bản")
        num = c.get("document_number", "")
        score_val = c.get("score", 0.0)
        label = f"{i}. **{title}**"
        if num:
            label += f" (Số: {num})"
        label += f" - Độ chính xác: {score_val:.0%}"
        lines.append(label)

    lines.append("")
    lines.append("Bạn đang tìm văn bản nào? Vui lòng chỉ định số thứ tự.")

    clarify_msg = "\n".join(lines)

    return {
        "final_answer": clarify_msg,
        "next_agent": AgentType.FINISH,
        "sources": [],
        # Keep all document_ids so user can choose
        "document_ids": [c.get("document_id") for c in all_candidates[:5] if c.get("document_id")],
        "section_reference": section_ref,
        "resolve_doc_ambiguous": True,
        "should_loop_back": False,
    }


def _build_resolved_state(top: dict, all_candidates: list, parsed: dict, pending_intent: str | None = None) -> dict:
    """Build the state update dict for a successfully resolved document."""
    from app.services.agents.models import AgentType

    doc_id = top.get("document_id")
    # Prefer section_ref from top candidate, fallback to parsed
    section_ref = top.get("section_reference") or parsed.get("section_reference") or ""
    title = top.get("title", "văn bản")
    strategies_used = ", ".join(top.get("strategies", [top.get("strategy", "?")]))
    score = top.get("score", 0.0)

    logger.info(
        f"[LANGGRAPH_DECISION] resolve_doc_agent Resolved → doc_id={doc_id}, score={score:.2f}, "
        f"section_ref={section_ref!r}, strategies=[{strategies_used}], "
        f"pending_intent={pending_intent!r}"
    )

    if score < MEDIUM_CONFIDENCE_THRESHOLD:
        # LOW confidence: don't scope search to this document
        logger.warning(
            f"[resolve_doc_agent] Low confidence score={score:.2f} < {MEDIUM_CONFIDENCE_THRESHOLD} "
            f"for doc_id={doc_id}, title={title!r} — will NOT scope search to this document"
        )
        resolved: dict = {
            "document_ids": [],  # Don't scope — low confidence
            "section_reference": section_ref,
            # NOTE: Do NOT write to kg_summaries — status messages are metadata,
            # not search results. operator.add would accumulate them as phantom results.
            "next_agent": AgentType.ANSWER_GENERATOR,
            "should_loop_back": False,
        }
    elif score >= HIGH_CONFIDENCE_THRESHOLD:
        # HIGH confidence: resolve document normally
        resolved: dict = {
            "document_ids": [doc_id] if doc_id else [],
            "section_reference": section_ref,
            "next_agent": AgentType.ANSWER_GENERATOR,
            "should_loop_back": False,
        }
    else:
        # MEDIUM confidence (0.30 - 0.59): ask user for confirmation
        return _build_medium_confidence_state(top, all_candidates, parsed, pending_intent)

    if section_ref:
        resolved["intent"] = "search_section"
        resolved["next_agent"] = AgentType.RAG
        logger.info(f"[resolve_doc_agent] Has section_ref → rag/search_section")
    elif pending_intent == "search_section" and not section_ref:
        # section_ref is empty but pending_intent was search_section
        # Fall back to 'search' with document_ids to find content within the resolved document
        resolved["intent"] = "search"
        resolved["pending_intent"] = "search"  # Clear search_section to allow routing to rag
        resolved["next_agent"] = AgentType.RAG
        logger.info(f"[resolve_doc_agent] search_section but no section_ref → fall back to search within doc")
    elif pending_intent:
        # Phase 4: Respect pending_intent from supervisor plan
        resolved["intent"] = pending_intent
        # Route to RAG if intent is search-related
        if pending_intent in ("search", "kg_query", "search_doc_num", "list_docs"):
            resolved["next_agent"] = AgentType.RAG
        else:
            resolved["next_agent"] = AgentType.ANSWER_GENERATOR
        logger.info(f"[resolve_doc_agent] Using pending_intent={pending_intent!r} → {resolved['next_agent']!r}")
    else:
        # Default fallback
        resolved["intent"] = "summarize"
        logger.info(f"[resolve_doc_agent] No section_ref or pending_intent → answer_generator/summarize")

    return resolved

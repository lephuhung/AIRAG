"""
Supervisor Agent
================

LLM-based supervisor that decides which agent to invoke next.
Single LLM call does BOTH intent classification + routing decision.

Graph Flow:
    START → supervisor → [rag | write | direct] → supervisor (loop)
                                                        ↓
                                                       END

Replaces the dead code version that was never integrated.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END

from app.services.agents.models import (
    SupervisorState,
    AgentType,
)

if TYPE_CHECKING:
    from app.services.llm.types import LLMMessage

logger = logging.getLogger(__name__)

# =============================================================================
# Supervisor Prompt
# =============================================================================

_SUPERVISOR_PROMPT = """\
You are a supervisor for a Vietnamese document Q&A system.

Given the user's message, classify the intent and decide which agent should handle.
Available agents:
- "rag": For document search, KG queries, listing/summarizing documents
- "write": For text summarization, editing suggestions, grammar/format checking
- "people": For searching persons by CCCD, name, BHXH, or phone number
- "direct": For greetings and personal questions (answer without document search)
- "finish": When a final answer has been generated and ready to return

Intent categories (use EXACTLY these names in your JSON output):
- "greeting": Simple greetings, hellos, thanks, farewells
- "personal": Questions about the user themselves (name, role, workplace, preferences)
- "search": General questions about document content, data, facts, analysis
- "list_docs": User wants to know what documents/files are available
- "summarize": User wants a summary of a specific document
- "kg_query": Questions about entity relationships, organizational structure, knowledge graph
- "search_doc_num": User asks about document numbers (văn bản số), reference numbers, official IDs
- "search_abbr": User asks about SHORT abbreviations/acronyms (e.g., "BMNN", "TTGT")
- "resolve_doc": User references a specific document by name/type/year without providing ID (e.g., "Luật An ninh mạng 2025", "Nghị định 60/2019", "Điều 27 luật này"). The system must resolve the reference to a document UUID before searching. Key phrases: "Luật X", "Nghị định X", "Thông tư X", "Điều X của luật", "văn bản số X", "tìm/tra cứu văn bản".
- "search_section": User asks to search, summarize, or compare a SPECIFIC section, chapter, or article (e.g., "Chương 3", "Điều 27") of a document.
- "write_summarize": User provides a TEXT PASSAGE (in the message itself) and wants it summarized. Does NOT mention a law/regulation name or document reference.
- "write_suggest_edits": User provides a TEXT PASSAGE and wants editing/improvement suggestions
- "write_grammar_check": User provides a TEXT PASSAGE and wants grammar/style checking
- "write_format_check": User wants to check Word document formatting (margins, fonts, line spacing)
- "mongo_search_cccd": User asks to look up a person by CCCD (Căn cước công dân) number
- "mongo_search_name": User asks to find/search for a person by name
- "mongo_search_bhxh": User asks to look up a person by BHXH (Bảo hiểm xã hội) number
- "mongo_search_phone": User asks to find a person by phone number
- "mongo_search_advanced": User asks to find a person combining multiple conditions (e.g. Name AND Date of Birth AND Address)

Intent → Agent routing guide (MUST follow exactly):
- greeting, personal → "direct"
- search, list_docs, summarize, kg_query, search_doc_num, search_abbr, resolve_doc, search_section → "rag"
- write_summarize, write_suggest_edits, write_grammar_check, write_format_check → "write"
- mongo_search_cccd, mongo_search_name, mongo_search_bhxh, mongo_search_phone, mongo_search_advanced → "people"

CRITICAL: `next_agent` must be ONLY one of: "rag"|"write"|"people"|"direct"|"finish".
NEVER put an intent name in `next_agent`. Use EXACT intent names from the list above.
Examples:
  - phone search → intent "mongo_search_phone", next_agent "people"
  - CCCD search  → intent "mongo_search_cccd",  next_agent "people"
  - dob and name search → intent "mongo_search_advanced", next_agent "people"
  - generic search → intent "search",             next_agent "rag"
  - "Tóm tắt điều 27 Luật An ninh mạng 2025" → intent "resolve_doc", next_agent "rag" (NOT write_summarize!)
  - "Tóm tắt văn bản số 60/2019" → intent "resolve_doc", next_agent "rag"
  - "tìm Nghị định 60/2019" → intent "resolve_doc", next_agent "rag"
  - "tóm tắt đoạn văn sau: [nội dung]" → intent "write_summarize", next_agent "write"

Rules:
1. First turn: classify intent from user message → route to appropriate agent
2. After agent completes: check if final answer is ready → "finish", else loop back
3. Guard: max {max_iterations} iterations to prevent infinite loops

Output format (JSON only, no explanation):
{{"next_agent": "rag"|"write"|"people"|"direct"|"finish", "intent": "<exact intent name from list above>", "reasoning": "<brief reason>"}}
"""

# Intent abbreviation → canonical name (safety net for LLM shortcutting intent names)
_INTENT_NORMALIZE: dict[str, str] = {
    # Common LLM shortcuts for people intents
    "search_phone": "mongo_search_phone",
    "search_cccd": "mongo_search_cccd",
    "search_name": "mongo_search_name",
    "search_bhxh": "mongo_search_bhxh",
    "phone_search": "mongo_search_phone",
    "cccd_search": "mongo_search_cccd",
    "name_search": "mongo_search_name",
    "bhxh_search": "mongo_search_bhxh",
    "search_advanced": "mongo_search_advanced",
    # some LLMs drop the prefix entirely
    "find_phone": "mongo_search_phone",
    "find_person": "mongo_search_name",
    "advanced_search": "mongo_search_advanced",
}

# Intent → Agent routing table used as fallback inside _parse_supervisor_response
_INTENT_TO_AGENT_FALLBACK: dict[str, str] = {
    "greeting": "direct",
    "personal": "direct",
    "search": "rag",
    "list_docs": "rag",
    "summarize": "rag",
    "kg_query": "rag",
    "search_doc_num": "rag",
    "search_abbr": "rag",
    "resolve_doc": "rag",
    "search_section": "rag",
    "write_summarize": "write",
    "write_suggest_edits": "write",
    "write_grammar_check": "write",
    "write_format_check": "write",
    "mongo_search_cccd": "people",
    "mongo_search_name": "people",
    "mongo_search_bhxh": "people",
    "mongo_search_phone": "people",
    "mongo_search_advanced": "people",
}

# =============================================================================
# Helper Functions
# =============================================================================


async def _expand_abbreviations_in_message(message: str) -> tuple[str, bool]:
    """
    Expand all uppercase abbreviations (2+ chars) found in the message.

    Returns (expanded_message, was_modified).
    Uses abbreviation DB to look up full forms.
    Supports both uppercase (BMNN) and lowercase (ntn, bmnn) abbreviations.
    """
    import re

    # Find all uppercase sequences (2+ chars) that could be abbreviations
    abbr_matches = re.findall(r'\b([A-Z]{2,})\b', message)
    # Also find lowercase sequences (2+ chars) that could be abbreviations
    lowercase_matches = re.findall(r'\b([a-z]{2,})\b', message)
    abbr_matches = abbr_matches + lowercase_matches

    if not abbr_matches:
        return message, False

    # Remove duplicates while preserving order
    unique_abbrs = list(dict.fromkeys(abbr_matches))

    try:
        from app.services.agent.streaming import get_current_db
        from sqlalchemy import select
        from app.models.abbreviation import Abbreviation

        db = get_current_db()
        if db is None:
            return message, False

        # Use a single query for all abbreviations
        from sqlalchemy import func
        result = await db.execute(
            select(Abbreviation)
            .where(
                func.lower(Abbreviation.short_form).in_([a.lower() for a in unique_abbrs]),
                Abbreviation.is_active == True,
            )
        )
        all_abbrs_db = result.scalars().all()
        
        # Group by lowercase short_form
        from collections import defaultdict
        abbr_map = defaultdict(list)
        for abbr_obj in all_abbrs_db:
            abbr_map[abbr_obj.short_form.lower()].append(abbr_obj)

        expanded_message = message
        any_expanded = False
        multi_meaning_abbrs = []

        for abbr in unique_abbrs:
            all_matches = abbr_map.get(abbr.lower(), [])

            if len(all_matches) == 0:
                # No match in DB - keep original, will prompt user to add
                logger.debug(f"[abbr_expand] No DB entry for: {abbr}")
                continue

            if len(all_matches) == 1:
                # Single meaning - safe to expand
                abbr_obj = all_matches[0]
                import re as re_module
                pattern = re_module.compile(r'\b' + re_module.escape(abbr) + r'\b', re.IGNORECASE)
                new_message = pattern.sub(abbr_obj.full_form, expanded_message)
                if new_message != expanded_message:
                    logger.info(f"[abbr_expand] {abbr} → {abbr_obj.full_form}")
                    expanded_message = new_message
                    any_expanded = True
            else:
                # Multiple meanings - keep original, let LLM decide from context
                # Record for potential user prompt
                multi_meaning_abbrs.append(abbr)
                short_forms = [f"{m.short_form}={m.full_form}" for m in all_matches]
                logger.info(f"[abbr_expand] Multiple meanings for {abbr}: {' | '.join(short_forms)}")

        return expanded_message, any_expanded

    except Exception as e:
        logger.warning(f"[abbr_expand] Failed to expand abbreviations: {e}")
        return message, False


def _extract_user_message(state: SupervisorState) -> str:
    """Extract the last user message from state.messages.

    On abbreviation loop-back, prefer expanded_query so supervisor
    re-classifies using the full form (e.g., "an ninh mạng") instead of
    the original abbreviation (e.g., "ANM").
    """
    # expanded_query is set by rag_agent when abbreviation is found
    expanded = state.get("expanded_query", "")
    if expanded:
        return expanded

    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "")
        else:
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            if role == "human":
                role = "user"
        if role == "user":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            return content or ""
    return ""


def _parse_supervisor_response(raw: str) -> dict:
    """Parse LLM JSON response with fallbacks."""
    raw = raw.strip()

    # Strip markdown code fences
    if "```json" in raw:
        raw = raw.split("```json")[-1].split("```")[0].strip()
    elif "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1].strip()

    try:
        data = json.loads(raw)
        next_agent = data.get("next_agent", "finish")
        intent = data.get("intent", "search")

        # Normalize intent name (LLM sometimes uses shorthand like "search_phone")
        intent = _INTENT_NORMALIZE.get(intent, intent)

        # Validate next_agent
        valid_agents = {
            AgentType.RAG, AgentType.WRITE, AgentType.PEOPLE,
            AgentType.DIRECT, AgentType.FINISH,
        }
        if next_agent not in valid_agents:
            # LLM sometimes returns an intent name instead of an agent name.
            # Use the fallback table to correct it before logging a warning.
            corrected = _INTENT_TO_AGENT_FALLBACK.get(next_agent)
            if corrected:
                logger.info(
                    f"[supervisor] Corrected next_agent '{next_agent}' → '{corrected}' "
                    f"(LLM returned intent name instead of agent name)"
                )
                next_agent = corrected
            else:
                logger.warning(
                    f"[supervisor] Invalid next_agent '{next_agent}', defaulting to 'finish'"
                )
                next_agent = AgentType.FINISH

        return {
            "next_agent": next_agent,
            "intent": intent,
            "reasoning": data.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        logger.warning(f"[supervisor] Failed to parse JSON: {raw[:100]!r}")
        # Fallback: treat as search intent, finish
        return {"next_agent": AgentType.FINISH, "intent": "search", "reasoning": "Parse failed, defaulting to finish"}


# =============================================================================
# Graph Nodes
# =============================================================================

async def supervisor_node(state: SupervisorState) -> dict:
    """
    Classify intent + decide next agent in ONE LLM call.

    This is the core supervisor logic:
    1. Extract user message
    2. Expand abbreviations in message
    3. Call LLM with supervisor prompt
    4. Parse response and update state
    """
    from app.services.llm import get_memory_agent

    user_message = _extract_user_message(state)
    if not user_message:
        return {
            "next_agent": AgentType.DIRECT,
            "intent": "greeting",
            "iterations": state.get("iterations", 0) + 1,
        }

    # Guard: max iterations
    from app.core.config import settings
    max_iter = getattr(settings, "NEXUSRAG_LG_MAX_ITERATIONS", 5)
    iterations = state.get("iterations", 0)

    # Special case: search_section loop back from rag — don't count against iterations
    # This happens when route_from_rag routes back to supervisor for search_section
    pending_section = state.get("section_reference")
    is_search_section_pending = (
        state.get("intent") == "search_section" and pending_section
    )
    if is_search_section_pending and iterations >= 1:
        # Still going — this is the supervisor re-entering from rag's route_from_rag
        # Just route directly to rag without incrementing iteration counter
        logger.info(f"[supervisor] search_section loop — re-routing to rag (iter={iterations})")
        return {
            "next_agent": AgentType.RAG,
            "intent": "search_section",
            "original_query": user_message,
            "rewritten_query": user_message,
            "iterations": iterations,  # Don't increment
            "should_loop_back": False,
            "section_reference": pending_section,
            "document_ids": state.get("document_ids") or [],
        }

    if iterations >= max_iter:
        logger.warning(f"[supervisor] Max iterations ({max_iter}) reached, forcing finish")
        return {"next_agent": AgentType.FINISH, "intent": state.get("intent", "search")}

    try:
        from app.services.llm.types import LLMMessage as _LLMMsg

        # On loop-back from abbreviation expansion, use expanded_query for re-classification
        expanded = state.get("expanded_query", "")
        query_for_classifier = expanded if expanded else user_message
        was_modified = False
        expanded_message = expanded

        # Expand abbreviations in message before classification
        # This ensures BMNN -> "Bộ môn nghiệp vụ" BEFORE intent classification
        if not expanded:
            expanded_message, was_modified = await _expand_abbreviations_in_message(user_message)
            if was_modified:
                logger.info(f"[supervisor] Abbreviations expanded: {user_message!r} -> {expanded_message!r}")
                query_for_classifier = expanded_message

        classifier = get_memory_agent()
        response_text = ""

        async for chunk in classifier.astream(
            [_LLMMsg(role="user", content=query_for_classifier)],
            system_prompt=_SUPERVISOR_PROMPT.format(max_iterations=max_iter),
            temperature=0.0,
            max_tokens=256,
        ):
            if hasattr(chunk, "text") and chunk.text:
                response_text += str(chunk.text)

        decision = _parse_supervisor_response(response_text)
        logger.info(
            f"[LANGGRAPH_ROUTE] user_message={user_message!r} -> "
            f"next_agent={decision['next_agent']!r}, intent={decision['intent']!r}, "
            f"reasoning={decision.get('reasoning', '')!r}"
        )

        # Build return dict
        # Check if we're looping back with accumulated results - don't reset should_loop_back
        # This happens when rag_agent resolved documents and we need to go to answer_generator
        existing_should_loop = state.get("should_loop_back", False)
        had_results = bool(state.get("kg_summaries") or state.get("sources"))

        # If we have accumulated results from a previous rag loop, go to answer_generator
        if existing_should_loop and had_results:
            logger.info(f"[supervisor] Looping back with {len(state.get('kg_summaries', []))} kg_summaries, going to answer_generator")
            # Preserve document_ids from rag_agent result - needed for summarize in answer_generator
            preserved_doc_ids = state.get("document_ids") or state.get("resolved_document_ids") or []
            return {
                "next_agent": AgentType.ANSWER_GENERATOR,
                "intent": "summarize",  # Force summarize to trigger content fetch in answer_generator
                "original_query": user_message,
                "rewritten_query": state.get("rewritten_query", user_message),
                "iterations": iterations + 1,
                "should_loop_back": False,
                "document_ids": preserved_doc_ids,
            }

        result = {
            "next_agent": decision["next_agent"],
            "intent": decision["intent"],
            "original_query": user_message,
            "rewritten_query": user_message,  # needed by tool functions
            "iterations": iterations + 1,
            # Reset loop flag on each supervisor entry
            "should_loop_back": False,
        }

        # Smart thinking decision: RAG tasks don't need thinking (just retrieval),
        # but complex synthesis with many sources or personal tasks benefit from it
        thinking_intents = {"greeting", "personal", "write_suggest_edits", "write_grammar_check"}
        rag_intents = {"search", "list_docs", "summarize", "kg_query", "search_doc_num", "resolve_doc", "search_section", "search_abbr"}
        intent = decision["intent"]

        # Check if user has personal memory from Graphiti - needs thinking to incorporate
        user_memory_context = state.get("user_memory_context", "")
        has_memory = bool(user_memory_context and "No relevant memories" not in user_memory_context)

        if intent in thinking_intents:
            result["enable_thinking"] = True
        elif has_memory:
            # Personal memory found - needs thinking to incorporate facts correctly
            result["enable_thinking"] = True
            logger.info(f"[supervisor] Thinking enabled for {intent} with personal memory")
        elif intent in rag_intents:
            # RAG tasks: check source count - many results benefit from thinking to synthesize
            source_count = len(state.get("sources", [])) + len(state.get("kg_summaries", []))
            if source_count >= 5:
                result["enable_thinking"] = True  # Complex synthesis needed
                logger.info(f"[supervisor] Thinking enabled for {intent} with {source_count} sources")
            else:
                result["enable_thinking"] = False  # Simple retrieval
        else:
            result["enable_thinking"] = False

        # If abbreviations were expanded, include expanded_query for downstream nodes
        if was_modified:
            result["expanded_query"] = expanded_message
            result["rewritten_query"] = expanded_message

        return result

    except Exception as e:
        logger.error(f"[supervisor] LLM call failed: {e}")
        return {
            "next_agent": AgentType.DIRECT,
            "intent": state.get("intent", "search"),
            "iterations": iterations + 1,
        }


async def direct_answer_node(state: SupervisorState) -> dict:
    """
    Answer greetings/personal questions directly without document retrieval.

    Emits token events via push_event for SSE streaming compatibility.
    """
    logger.info(f"[LANGGRAPH_NODE] Entering direct_answer_node, intent={state.get('intent')!r}")
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage
    from app.services.agent.streaming import push_event

    provider = get_llm_provider()
    messages = state.get("messages", [])
    system_prompt = state.get("system_prompt", "")
    user_memory = state.get("user_memory_context", "")

    # Emit status
    await push_event(state, "status", {"step": "generating", "detail": "Đang trả lời..."})

    # Build LLM messages
    llm_messages: list[LLMMessage] = []
    for msg in (messages or [])[-6:]:
        if isinstance(msg, dict):
            role, content = msg.get("role", "user"), msg.get("content", "")
        else:
            role, content = getattr(msg, "role", "user"), getattr(msg, "content", "")
            
        # Truncate content if it's an assistant message to save context window
        if role == "assistant" and len(content) > 500:
            content = content[:500] + "...\n[Nội dung đã được rút gọn]"
            
        llm_messages.append(LLMMessage(role=role, content=content))

    # Inject memory if available
    effective_system = system_prompt
    if user_memory and "No relevant memories" not in user_memory:
        effective_system = (
            f"{user_memory}\n\n"
            "IMPORTANT: Do NOT copy these facts directly. When using a memory fact, "
            "paraphrase it in your own words and cite it as [MEM-1], [MEM-2], etc. "
            "For example: 'The user works at Công an tỉnh Hà Tĩnh [MEM-1]' instead of copying the fact verbatim. "
            "Only include relevant memories.\n\n"
        ) + effective_system

    answer_parts = []
    try:
        async for chunk in provider.astream(
            messages=llm_messages,
            temperature=0.5,
            max_tokens=2048,
            system_prompt=effective_system,
            think=False,
        ):
            if chunk.text:
                answer_parts.append(chunk.text)
                # Emit token for SSE streaming
                await push_event(state, "token", chunk.text)
    except Exception as e:
        logger.error(f"[direct_answer] LLM streaming failed: {e}")
        answer_parts = ["Xin chào! Tôi có thể giúp gì cho bạn?"]
        await push_event(state, "token", answer_parts[0])

    # Import strip_thinking_tags from nodes
    from app.services.agent.nodes import strip_thinking_tags
    final_answer = strip_thinking_tags("".join(answer_parts))

    return {"final_answer": final_answer, "next_agent": AgentType.FINISH}


# =============================================================================
# Answer Generator Node
# =============================================================================

async def answer_generator_node(state: SupervisorState) -> dict:
    """
    Generate final answer from accumulated sources/context.
    Wraps the answer_generator from nodes.py for supervisor graph.

    Uses merge pattern over DEFAULT_STATE to avoid manual field-by-field copy.
    This stays in sync automatically when new fields are added to either state.

    For write intents (summarize/suggest_edits/grammar_check/format_check),
    the write agent already produces a final_answer — skip LLM generation
    and use the write agent's output directly.
    """
    from app.services.agent.nodes import answer_generator as _orig_ag
    from app.services.agent.state import DEFAULT_STATE

    intent = state.get("intent", "")
    logger.info(f"[LANGGRAPH_NODE] Entering answer_generator_node, intent={intent!r}")

    # Check if agent already produced final_answer and shouldn't be reformatted by answer_generator
    skip_intents = {
        "write_summarize",
        "write_suggest_edits",
        "write_grammar_check",
        "write_format_check",
        "greeting",
        "personal",
    }
    intent = state.get("intent", "")
    existing_final = state.get("final_answer", "")
    if intent in skip_intents and existing_final:
        logger.info(
            f"[answer_generator_node] Skip intent={intent!r} — using existing final_answer "
            f"({len(existing_final)} chars), skipping LLM generation"
        )
        return {
            "final_answer": existing_final,
            "next_agent": AgentType.FINISH,
        }

    # Merge SupervisorState over DEFAULT_STATE — keys present in state win.
    # We iterate state.items() to handle TypedDict gracefully.
    agent_state = {**DEFAULT_STATE}
    for k, v in state.items():
        # Keep None values only when the field is not in DEFAULT_STATE
        # (i.e. don't overwrite meaningful defaults with None from SupervisorState)
        if v is not None or k not in DEFAULT_STATE:
            agent_state[k] = v

    # Override control fields — always force these regardless of state values
    agent_state["tool_called"] = True
    agent_state["existing_citation_ids"] = {}
    agent_state["citation_map"] = {}

    # Call the original answer_generator with AgentState-compatible dict
    result = await _orig_ag(agent_state)

    return {
        "final_answer": result.get("final_answer", ""),
        "next_agent": AgentType.FINISH,
    }


# =============================================================================
# Routing Functions
# =============================================================================

def route_from_supervisor(state: SupervisorState) -> str:
    """Route to appropriate agent based on supervisor's decision.

    Handles the case where supervisor returns next_agent=None on loop back,
    meaning we're done and should END.
    """
    next_agent = state.get("next_agent")
    intent = state.get("intent", "")

    if next_agent is None or next_agent == AgentType.FINISH:
        return END

    # Bypass memory_recall for intents that don't need personal memory context:
    # - greeting/personal: direct_answer_node handles its own minimal memory use
    # - write_format_check/grammar_check/suggest_edits: user provides inline text,
    #   write agent processes it without needing user's past conversation facts
    # - write_summarize: user provides TEXT PASSAGE in the message itself (per
    #   intent definition: "User provides a TEXT PASSAGE and wants it summarized")
    if intent in (
        "greeting",
        "personal",
        "write_format_check",
        "write_grammar_check",
        "write_suggest_edits",
        "write_summarize",
    ):
        if next_agent == AgentType.WRITE:
            return "bypass_memory_to_write"
        if next_agent == AgentType.DIRECT:
            return "direct"  # bypass memory_recall, go straight to direct node

    return next_agent


def route_from_rag(state: SupervisorState) -> str:
    """Route after rag agent completes.

    If abbreviation was found and expanded, loop back to supervisor
    to re-classify with the full form. Otherwise go to answer_generator.

    For search_section intent, we check section_reference:
    - If section_reference is still in state → search_section tool NOT yet executed
      (supervisor set it but tool hasn't run yet) → route back to supervisor
      (supervisor will re-route to rag with same intent, and rag will execute tool)
    - If section_reference is None/empty → tool already ran, kg_summaries has content
      → go to answer_generator
    """
    pending_section = state.get("section_reference")
    intent = state.get("intent")
    loop = state.get("should_loop_back")
    logger.info(f"[route_from_rag] intent={intent!r}, section_reference={pending_section!r}, should_loop_back={loop}")
    if loop:
        return "supervisor"
    if intent == "search_section":
        if pending_section:
            # section_reference still present → tool not yet executed
            # Loop back to supervisor so it re-routes to rag
            return "supervisor"
        return "answer_generator"
    return "answer_generator"


# shouldContinue_from_supervisor removed — was dead code (never called).


# =============================================================================
# Graph Builder
# =============================================================================

def create_supervisor_graph():
    """
    Build and compile the supervisor-based multi-agent graph.

    Flow:
        START → supervisor → [rag | write | direct] → answer_generator → END
                      ↑                               │
                      └───────────────────────────────┘
    """
    graph = StateGraph(SupervisorState)

    # Nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("memory_recall", _memory_recall_wrapper)
    graph.add_node("rag", _rag_agent_wrapper)
    graph.add_node("write", _write_agent_wrapper)
    graph.add_node("people", _people_agent_wrapper)
    graph.add_node("direct", direct_answer_node)
    graph.add_node("answer_generator", answer_generator_node)

    # Edges
    graph.add_edge(START, "supervisor")

    # Conditional edges from supervisor based on next_agent
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            AgentType.RAG: "memory_recall",
            AgentType.WRITE: "memory_recall",
            AgentType.DIRECT: "memory_recall",
            AgentType.PEOPLE: "memory_recall",
            AgentType.ANSWER_GENERATOR: "answer_generator",
            "bypass_memory_to_write": "write",
            END: END,  # handle finish case
        },
    )

    # memory_recall loads personal context, then routes to the target agent
    def route_from_memory(state: SupervisorState) -> str:
        return state.get("next_agent", AgentType.FINISH)

    graph.add_conditional_edges(
        "memory_recall",
        route_from_memory,
        {
            AgentType.RAG: "rag",
            AgentType.WRITE: "write",
            AgentType.DIRECT: "direct",
            AgentType.PEOPLE: "people",
            AgentType.FINISH: END,
        },
    )

    # After rag/write/people/direct agents complete, go to answer_generator
    # (rag uses conditional routing to support abbreviation loop-back)
    graph.add_conditional_edges("rag", route_from_rag, {"supervisor": "supervisor", "answer_generator": "answer_generator"})
    graph.add_edge("write", "answer_generator")
    graph.add_edge("people", "answer_generator")
    graph.add_edge("direct", "answer_generator")

    # answer_generator leads to END
    graph.add_edge("answer_generator", END)

    return graph.compile()


# =============================================================================
# Singleton - cached compiled graph (same pattern as agent/graph.py)
# =============================================================================

import threading as _threading

_supervisor_graph = None
_supervisor_graph_lock = _threading.Lock()


def get_supervisor_graph():
    """Return cached compiled supervisor graph.

    Thread-safe double-checked locking: avoids redundant graph builds
    when multiple coroutines start simultaneously.
    """
    global _supervisor_graph
    if _supervisor_graph is None:
        with _supervisor_graph_lock:
            if _supervisor_graph is None:
                _supervisor_graph = create_supervisor_graph()
    return _supervisor_graph


def reset_supervisor_graph():
    """Force the singleton to rebuild on next call (e.g. after hot-reload)."""
    global _supervisor_graph
    with _supervisor_graph_lock:
        _supervisor_graph = None


# =============================================================================
# Agent Wrappers (imported from new agent files)
# =============================================================================

async def _memory_recall_wrapper(state: SupervisorState) -> dict:
    """
    Load user memories from Graphiti into SupervisorState.user_memory_context.
    Called before direct/write/rag so every agent has access to personal memory.
    """
    import uuid as _uuid

    user_id = state.get("user_id")
    if not user_id:
        return {}

    user_message = state.get("rewritten_query") or state.get("original_query", "")
    if not user_message:
        return {}

    try:
        from app.services.graphiti_client import search_user_memory

        # user_id in state may be int or UUID depending on how it was passed
        uid = user_id
        if isinstance(uid, int):
            uid = _uuid.UUID(int=uid)
        elif isinstance(uid, str):
            uid = _uuid.UUID(uid)

        memory = await search_user_memory(uid, user_message, top_k=5)
        if memory:
            logger.info(f"[memory_recall_wrapper] Graphiti injected {len(memory)} chars")
            return {"user_memory_context": memory}
        else:
            logger.info(f"[memory_recall_wrapper] Graphiti found no relevant memory for user_id={uid}")
    except Exception as e:
        logger.warning(f"[memory_recall_wrapper] failed: {e}")

    return {}


async def _rag_agent_wrapper(state: SupervisorState) -> dict:
    """Wrapper that imports and calls rag_agent_node."""
    from app.services.agents.rag_agent import rag_agent_node

    return await rag_agent_node(state)


async def _write_agent_wrapper(state: SupervisorState) -> dict:
    """Wrapper that imports and calls write_agent_node."""
    from app.services.agents.write_agent import write_agent_node

    return await write_agent_node(state)


async def _people_agent_wrapper(state: SupervisorState) -> dict:
    """Wrapper that imports and calls people_agent_node."""
    from app.services.agents.people_agent import people_agent_node

    return await people_agent_node(state)


# =============================================================================
# Backward Compatibility - Re-export for existing imports
# =============================================================================

# Keep old SupervisorState name for compatibility during transition
SupervisorStateModel = SupervisorState
create_supervisor = create_supervisor_graph

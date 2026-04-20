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
    INTENT_TO_AGENT,
)

if TYPE_CHECKING:
    from app.services.llm.types import LLMMessage

logger = logging.getLogger(__name__)

# =============================================================================
# Supervisor Prompt
# =============================================================================

_SUPERVISOR_PROMPT = """\
You are a supervisor for a Vietnamese document Q&A system.

Given the user's message, classify the intent and decide which agent should handAvailable agents:
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
- "write_summarize": User provides a TEXT PASSAGE and wants it summarized
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
- search, list_docs, summarize, kg_query, search_doc_num, search_abbr → "rag"
- write_summarize, write_suggest_edits, write_grammar_check, write_format_check → "write"
- mongo_search_cccd, mongo_search_name, mongo_search_bhxh, mongo_search_phone, mongo_search_advanced → "people"

CRITICAL: `next_agent` must be ONLY one of: "rag"|"write"|"people"|"direct"|"finish".
NEVER put an intent name in `next_agent`. Use EXACT intent names from the list above.
Examples:
  - phone search → intent "mongo_search_phone", next_agent "people"
  - CCCD search  → intent "mongo_search_cccd",  next_agent "people"
  - dob and name search → intent "mongo_search_advanced", next_agent "people"
  - doc search   → intent "search",             next_agent "rag"

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
    2. Call LLM with supervisor prompt
    3. Parse response and update state
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
    if iterations >= max_iter:
        logger.warning(f"[supervisor] Max iterations ({max_iter}) reached, forcing finish")
        return {"next_agent": AgentType.FINISH, "intent": state.get("intent", "search")}

    try:
        from app.services.llm.types import LLMMessage as _LLMMsg

        # On loop-back from abbreviation expansion, use expanded_query for re-classification
        expanded = state.get("expanded_query", "")
        query_for_classifier = expanded if expanded else user_message

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
            f"[supervisor] decision: next_agent={decision['next_agent']!r}, "
            f"intent={decision['intent']!r}, reasoning={decision.get('reasoning', '')!r}"
        )

        return {
            "next_agent": decision["next_agent"],
            "intent": decision["intent"],
            "original_query": user_message,
            "rewritten_query": user_message,  # needed by tool functions
            "iterations": iterations + 1,
            # Reset loop flag on each supervisor entry
            "should_loop_back": False,
        }

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
        llm_messages.append(LLMMessage(role=role, content=content))

    # Inject memory if available
    effective_system = system_prompt
    if user_memory and "No relevant memories" not in user_memory:
        effective_system = (
            f"USER MEMORY:\n{user_memory}\n\n"
            "Use this info when relevant. Cite memory facts as [MEM-1], [MEM-2], etc.\n"
            "Do NOT include the header 'USER MEMORY' in your response.\n\n"
        ) + effective_system

    answer_parts = []
    try:
        async for chunk in provider.astream(
            messages=llm_messages,
            temperature=0.5,
            max_tokens=512,
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
    """
    from app.services.agent.nodes import answer_generator as _orig_ag
    from app.services.agent.state import DEFAULT_STATE

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
    if next_agent is None or next_agent == AgentType.FINISH:
        return END
    return next_agent


def route_from_rag(state: SupervisorState) -> str:
    """Route after rag agent completes.

    If abbreviation was found and expanded, loop back to supervisor
    to re-classify with the full form. Otherwise go to answer_generator.
    """
    if state.get("should_loop_back"):
        return "supervisor"
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
            AgentType.RAG: "rag",
            AgentType.WRITE: "write",
            AgentType.DIRECT: "direct",
            AgentType.PEOPLE: "people",
            END: END,  # handle finish case
        },
    )

    # After rag/write/people/direct agents complete, go to answer_generator
    # (rag uses conditional routing to support abbreviation loop-back)
    graph.add_conditional_edges("rag", route_from_rag, {"supervisor": "supervisor", "answer_generator": "answer_generator"})
    graph.add_edge("write", "answer_generator")
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

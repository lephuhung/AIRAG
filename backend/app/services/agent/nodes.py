"""
Agent Nodes
===========

LangGraph node functions for the NexusRAG chat agent.

Each node receives the full AgentState and returns a partial state update dict.
Nodes are pure async functions — LangGraph merges the returned dict into state.

Nodes:
    memory_recall       — load user memories from pgvector
    intent_classifier   — Qwen3-4B: classify intent + rewrite query
    tool_executor       — dispatch to the correct tool based on intent
    answer_generator    — main LLM: generate answer with sources in context
    direct_answer       — main LLM: answer greetings/chitchat directly

SSE streaming:
    All nodes call push_event(state, ev_type, ev_data) to push events into the
    shared asyncio.Queue injected by stream_agent_to_sse. answer_generator and
    direct_answer use provider.astream() to push tokens one-by-one in real-time.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.agent.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent classifier system prompt — for Qwen3-4B
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """\
You are an intent classification assistant for a Vietnamese document Q&A system.
Your job is to classify the user's message and rewrite their query for better retrieval.

Respond ONLY with valid JSON. No explanation, no markdown, no extra text.

Intent categories:
- "greeting"  : greetings, thanks, farewells, simple chitchat → no document search needed
- "search"    : questions about document content, data, facts, analysis → needs search
- "list_docs" : user wants to know what documents/files are available
- "summarize" : user wants a summary of a specific document
- "kg_query"  : user asks about entity relationships, organizational charts, knowledge graph

Output format:
{"intent": "<category>", "rewritten_query": "<improved Vietnamese/English search query>", "needs_tool": true|false}

Rules:
- For "greeting": set rewritten_query to "" and needs_tool to false
- For all other intents: rewrite the query to be specific and detailed for retrieval
- If the message contains a document ID, preserve it in the output
- Default to "search" when uncertain

Examples:
User: "xin chào"  → {"intent": "greeting", "rewritten_query": "", "needs_tool": false}
User: "doanh thu 2024 là bao nhiêu?" → {"intent": "search", "rewritten_query": "doanh thu thuần tổng doanh thu năm 2024 theo quý", "needs_tool": true}
User: "có tài liệu gì trong hệ thống?" → {"intent": "list_docs", "rewritten_query": "danh sách tài liệu", "needs_tool": true}
User: "tóm tắt tài liệu ID 5" → {"intent": "summarize", "rewritten_query": "tóm tắt tài liệu 5", "needs_tool": true}
User: "mối quan hệ giữa VietcomBank và VCB" → {"intent": "kg_query", "rewritten_query": "VietcomBank VCB relationship entities", "needs_tool": true}
"""

_VALID_INTENTS = {"greeting", "search", "list_docs", "summarize", "kg_query"}


def _parse_classifier_output(raw: str) -> dict:
    """Parse Qwen3-4B classifier JSON output with safe fallback."""
    raw = raw.strip()

    # Strip markdown code fences if present
    if "```json" in raw:
        raw = raw.split("```json")[-1].split("```")[0].strip()
    elif "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1].strip()

    try:
        data = json.loads(raw)
        intent = data.get("intent", "search")
        if intent not in _VALID_INTENTS:
            logger.warning(f"[classifier] Unknown intent '{intent}', defaulting to 'search'")
            intent = "search"
        return {
            "intent": intent,
            "rewritten_query": data.get("rewritten_query", ""),
            "needs_tool": data.get("needs_tool", True),
        }
    except json.JSONDecodeError:
        logger.warning(f"[classifier] Failed to parse JSON: {raw[:100]!r}, defaulting to search")
        return {"intent": "search", "rewritten_query": "", "needs_tool": True}


# ---------------------------------------------------------------------------
# Node: memory_recall
# ---------------------------------------------------------------------------

async def memory_recall(state: "AgentState") -> dict:
    """
    Load relevant user memories from pgvector and inject into state.
    Non-blocking — if memory search fails, pipeline continues normally.
    """
    from app.services.agent.streaming import push_event, get_current_db

    await push_event(state, "status", {"step": "analyzing", "detail": "Đang tải bộ nhớ người dùng..."})

    user_id = state.get("user_id")
    if not user_id:
        return {"user_memory_context": ""}

    # Extract the last user message text for memory search query
    messages = state.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "user":
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            user_message = content
            break

    if not user_message:
        return {"user_memory_context": ""}

    try:
        from app.api.chat_agent import _search_memories
        # Đọc db từ ContextVar — bypass LangGraph TypedDict key filtering
        db = get_current_db()
        if db is None:
            return {"user_memory_context": ""}

        memory_context = await _search_memories(user_id, user_message, db, top_k=5)
        if memory_context and "No relevant memories" not in memory_context and "No memories found" not in memory_context:
            logger.info(f"[memory_recall] Injected {len(memory_context)} chars for user {user_id}")
            return {"user_memory_context": memory_context}
    except Exception as e:
        logger.warning(f"[memory_recall] Failed: {e}")

    return {"user_memory_context": ""}


# ---------------------------------------------------------------------------
# Node: intent_classifier
# ---------------------------------------------------------------------------

async def intent_classifier(state: "AgentState") -> dict:
    """
    Use Qwen3-4B (via memory agent endpoint) to classify intent and rewrite query.

    Extracts the last user message, calls the classifier, and returns:
        intent, rewritten_query updates to state.
    """
    from app.services.agent.streaming import push_event

    await push_event(state, "status", {"step": "analyzing", "detail": "Đang phân tích câu hỏi..."})

    messages = state.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "user":
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            user_message = content
            break

    if not user_message:
        return {"intent": "search", "rewritten_query": ""}

    try:
        from app.services.llm import get_memory_agent
        from app.services.llm.types import LLMMessage as _LLMMsg

        classifier = get_memory_agent()
        response_text = ""

        async for chunk in classifier.astream(
            [_LLMMsg(role="user", content=user_message)],
            system_prompt=_CLASSIFIER_SYSTEM,
            temperature=0.0,
            max_tokens=128,
        ):
            if chunk.text:
                response_text += chunk.text

        result = _parse_classifier_output(response_text)
        logger.info(
            f"[intent_classifier] intent={result['intent']!r} "
            f"rewritten={result['rewritten_query']!r}"
        )

        # Emit a meaningful status based on intent
        intent_labels = {
            "greeting": "Tin nhắn thông thường",
            "search": "Tìm kiếm tài liệu",
            "list_docs": "Liệt kê tài liệu",
            "summarize": "Tóm tắt tài liệu",
            "kg_query": "Truy vấn đồ thị tri thức",
        }
        intent_label = intent_labels.get(result["intent"], "Tìm kiếm")
        await push_event(state, "status", {
            "step": "searching",
            "detail": f"Phân loại: {intent_label}",
        })

        return {
            "intent": result["intent"],
            "rewritten_query": result["rewritten_query"] or user_message,
        }

    except Exception as e:
        logger.error(f"[intent_classifier] Classifier failed: {e} — defaulting to 'search'")
        return {"intent": "search", "rewritten_query": user_message}


# ---------------------------------------------------------------------------
# Node: tool_executor
# ---------------------------------------------------------------------------

async def tool_executor(state: "AgentState") -> dict:
    """
    Dispatch to the appropriate tool based on classified intent.

    Updates: sources, images, image_parts, kg_summaries, tool_called, iterations
    Pushes: status, sources, images events into the SSE queue.
    """
    from app.services.agent import tools as _tools
    from app.services.agent.streaming import push_event, get_current_db

    intent = state.get("intent", "search")
    query = state.get("rewritten_query") or ""
    workspace_ids = state.get("workspace_ids", [])
    existing_ids = state.get("existing_citation_ids", set())
    # Đọc db từ ContextVar — bypass LangGraph TypedDict key filtering
    db = get_current_db()
    iterations = state.get("iterations", 0) + 1

    logger.info(f"[tool_executor] intent={intent!r} query={query!r}")

    # Emit status event indicating what we're doing
    tool_status_map = {
        "search": "Đang tìm kiếm tài liệu liên quan...",
        "list_docs": "Đang lấy danh sách tài liệu...",
        "summarize": "Đang tóm tắt tài liệu...",
        "kg_query": "Đang truy vấn đồ thị tri thức...",
    }
    await push_event(state, "status", {
        "step": "searching",
        "detail": tool_status_map.get(intent, "Đang xử lý yêu cầu..."),
    })

    result_update: dict = {
        "tool_called": True,
        "iterations": iterations,
        "sources": [],
        "images": [],
        "image_parts": [],
        "kg_summaries": [],
    }

    try:
        if intent == "search":
            from app.core.config import settings
            top_k = settings.HRAG_RERANKER_TOP_K

            tool_result = await _tools.search_documents(
                query=query,
                top_k=top_k,
                workspace_ids=workspace_ids,
                existing_citation_ids=existing_ids,
                db=db,
            )
            result_update["sources"] = tool_result["sources"]
            result_update["images"] = tool_result["images"]
            result_update["image_parts"] = tool_result["image_parts"]
            result_update["kg_summaries"] = tool_result["kg_summaries"]

        elif intent == "list_docs":
            tool_result = await _tools.list_documents(
                workspace_ids=workspace_ids,
                db=db,
            )
            # Store list result as a synthetic "source" for context
            result_update["kg_summaries"] = [tool_result["text"]]

        elif intent == "summarize":
            # Extract document ID from query if present, else use first doc
            import re
            doc_id_match = re.search(r"\b(?:id\s*[:=]?\s*)?(\d+)\b", query, re.IGNORECASE)
            doc_id = int(doc_id_match.group(1)) if doc_id_match else 0

            if doc_id:
                tool_result = await _tools.summarize_document(
                    document_id=doc_id,
                    db=db,
                )
                result_update["kg_summaries"] = [tool_result["text"]]
            else:
                # Fallback to search if no doc ID found
                logger.warning("[tool_executor] summarize intent but no doc_id found — falling back to search")
                from app.core.config import settings
                tool_result = await _tools.search_documents(
                    query=query,
                    top_k=settings.HRAG_RERANKER_TOP_K,
                    workspace_ids=workspace_ids,
                    existing_citation_ids=existing_ids,
                    db=db,
                )
                result_update["sources"] = tool_result["sources"]
                result_update["images"] = tool_result["images"]
                result_update["image_parts"] = tool_result["image_parts"]
                result_update["kg_summaries"] = tool_result["kg_summaries"]

        elif intent == "kg_query":
            tool_result = await _tools.query_knowledge_graph(
                entity=query,
                workspace_ids=workspace_ids,
                db=db,
            )
            result_update["kg_summaries"] = [tool_result["text"]]

        else:
            logger.warning(f"[tool_executor] Unknown intent {intent!r}, defaulting to search")
            from app.core.config import settings
            tool_result = await _tools.search_documents(
                query=query,
                top_k=settings.HRAG_RERANKER_TOP_K,
                workspace_ids=workspace_ids,
                existing_citation_ids=existing_ids,
                db=db,
            )
            result_update["sources"] = tool_result["sources"]
            result_update["images"] = tool_result["images"]
            result_update["image_parts"] = tool_result["image_parts"]
            result_update["kg_summaries"] = tool_result["kg_summaries"]

    except Exception as e:
        logger.error(f"[tool_executor] Tool execution failed: {e}")

    # ── Push sources and images events into the SSE queue ───────────────────
    sources = result_update.get("sources", [])
    images = result_update.get("images", [])

    if sources:
        logger.info(f"[tool_executor] Pushing {len(sources)} sources to SSE queue")
        await push_event(state, "sources", sources)
        await push_event(state, "status", {
            "step": "retrieved",
            "detail": f"Tìm thấy {len(sources)} nguồn tài liệu liên quan",
        })

    if images:
        logger.info(f"[tool_executor] Pushing {len(images)} images to SSE queue")
        await push_event(state, "images", images)

    return result_update


# ---------------------------------------------------------------------------
# Node: answer_generator
# ---------------------------------------------------------------------------

async def answer_generator(state: "AgentState") -> dict:
    """
    Main LLM node — generates final answer using retrieved context.

    Reads sources/images/kg_summaries from state, builds context string,
    streams tokens via provider.astream() and pushes each token to the SSE queue.
    Returns final_answer for state persistence.
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage as _LLMMsg
    from app.services.agent.streaming import push_event

    provider = get_llm_provider()

    await push_event(state, "status", {"step": "generating", "detail": "Đang tạo câu trả lời..."})

    # Build context from accumulated retrieval results
    sources = state.get("sources", [])
    kg_summaries = state.get("kg_summaries", [])
    system_prompt = state.get("system_prompt", "")
    user_memory = state.get("user_memory_context", "")
    messages = state.get("messages", [])
    enable_thinking = state.get("enable_thinking", False)

    # Inject memory into system prompt if available
    effective_system = system_prompt
    if user_memory and "No relevant memories" not in user_memory:
        effective_system = (
            f"AUTHENTICATED USER PROFILE (PERSONAL HISTORY):\n{user_memory}\n\n"
            "Rules for using this profile:\n"
            "1. If the user asks about themselves, answer DIRECTLY from this profile.\n"
            "2. For all other questions, answer from document sources only.\n"
            "3. NEVER blend personal facts with document answers.\n\n"
        ) + effective_system

    # Build context string
    context_parts = []
    if kg_summaries:
        context_parts.append("## Knowledge Graph / Tool Results\n" + "\n\n".join(kg_summaries))

    if sources:
        chunk_parts = []
        for src in sources:
            cid = src.get("index", "??")
            content = src.get("content", "")
            source_file = src.get("source_file", "")
            page_no = src.get("page_no", 0)
            heading_path = src.get("heading_path", [])
            meta_parts = []
            if source_file:
                meta_parts.append(source_file)
            if page_no:
                meta_parts.append(f"page {page_no}")
            if heading_path:
                meta_parts.append(" > ".join(heading_path))
            meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
            chunk_parts.append(f"Source [{cid}]{meta_line}:\n{content}")

        context_parts.append("## Document Chunks\n" + "\n\n---\n\n".join(chunk_parts))

    # Build llm messages — convert from state messages
    llm_messages: list[_LLMMsg] = []
    for msg in (messages or [])[-10:]:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
        llm_messages.append(_LLMMsg(role=role, content=content))

    # Inject context as the last "user" turn supplement
    if context_parts:
        context_text = "\n\n".join(context_parts)
        inject = (
            "\n\n=== RETRIEVED CONTEXT ===\n"
            + context_text
            + "\n=== END CONTEXT ===\n\n"
            "IMPORTANT:\n"
            "- Answer using ONLY the retrieved sources above.\n"
            "- Cite sources using their IDs: e.g. claim text[cid1][cid2].\n"
            "- If the context does not contain the answer, say: 'Tài liệu không chứa thông tin này.'\n"
            "- TABLE DATA: 'Key, Year = Value' pairs are table cells.\n"
        )
        # Append to last user message
        if llm_messages and llm_messages[-1].role == "user":
            llm_messages[-1] = _LLMMsg(
                role="user",
                content=llm_messages[-1].content + inject,
            )
        else:
            llm_messages.append(_LLMMsg(role="user", content=inject))

    from app.core.config import settings

    # Stream tokens in real-time via astream()
    answer_parts: list[str] = []
    thinking_parts: list[str] = []

    try:
        async for chunk in provider.astream(
            messages=llm_messages,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            system_prompt=effective_system,
            think=enable_thinking,
        ):
            # Handle thinking tokens (extended thinking mode)
            if chunk.type == "thinking" and chunk.text:
                thinking_parts.append(chunk.text)
                await push_event(state, "thinking", {"text": chunk.text})

            # Handle regular answer tokens
            elif chunk.type == "text" and chunk.text:
                answer_parts.append(chunk.text)
                await push_event(state, "token", chunk.text)

    except Exception as e:
        logger.error(f"[answer_generator] LLM streaming failed: {e}", exc_info=True)
        # Fallback: try non-streaming
        try:
            result = await provider.acomplete(
                messages=llm_messages,
                temperature=0.1,
                max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                system_prompt=effective_system,
                think=enable_thinking,
            )
            fallback_answer = result if isinstance(result, str) else getattr(result, "content", str(result))
            answer_parts.append(fallback_answer)
            await push_event(state, "token", fallback_answer)
        except Exception as e2:
            logger.error(f"[answer_generator] Fallback also failed: {e2}")
            error_msg = "Xin lỗi, tôi gặp lỗi khi tạo câu trả lời. Vui lòng thử lại."
            answer_parts.append(error_msg)
            await push_event(state, "token", error_msg)

    final_answer = "".join(answer_parts)
    return {"final_answer": final_answer}


# ---------------------------------------------------------------------------
# Node: direct_answer
# ---------------------------------------------------------------------------

async def direct_answer(state: "AgentState") -> dict:
    """
    Answer greetings / chitchat directly without document retrieval.
    Uses the main LLM provider with memory context if available.
    Streams tokens in real-time via push_event.
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage as _LLMMsg
    from app.core.config import settings
    from app.services.agent.streaming import push_event

    provider = get_llm_provider()
    messages = state.get("messages", [])
    system_prompt = state.get("system_prompt", "")
    user_memory = state.get("user_memory_context", "")
    enable_thinking = state.get("enable_thinking", False)

    await push_event(state, "status", {"step": "generating", "detail": "Đang trả lời..."})

    effective_system = system_prompt
    if user_memory and "No relevant memories" not in user_memory:
        effective_system = (
            f"AUTHENTICATED USER PROFILE:\n{user_memory}\n\n"
            "Answer based on user profile when relevant. Do not search documents.\n\n"
        ) + effective_system

    llm_messages: list[_LLMMsg] = []
    for msg in (messages or [])[-6:]:
        if isinstance(msg, dict):
            role, content = msg.get("role", "user"), msg.get("content", "")
        else:
            role, content = getattr(msg, "role", "user"), getattr(msg, "content", "")
        llm_messages.append(_LLMMsg(role=role, content=content))

    answer_parts: list[str] = []

    try:
        async for chunk in provider.astream(
            messages=llm_messages,
            temperature=0.5,
            max_tokens=512,
            system_prompt=effective_system,
            think=enable_thinking,
        ):
            if chunk.text:
                answer_parts.append(chunk.text)
                await push_event(state, "token", chunk.text)

    except Exception as e:
        logger.error(f"[direct_answer] LLM streaming failed: {e}", exc_info=True)
        # Fallback: non-streaming
        try:
            result = await provider.acomplete(
                messages=llm_messages,
                temperature=0.5,
                max_tokens=512,
                system_prompt=effective_system,
                think=enable_thinking,
            )
            fallback = result if isinstance(result, str) else getattr(result, "content", str(result))
            answer_parts.append(fallback)
            await push_event(state, "token", fallback)
        except Exception as e2:
            logger.error(f"[direct_answer] Fallback also failed: {e2}")
            greeting = "Xin chào! Tôi có thể giúp gì cho bạn?"
            answer_parts.append(greeting)
            await push_event(state, "token", greeting)

    return {"final_answer": "".join(answer_parts)}

"""
Chat Agent — Semi-Agentic SSE Streaming for HRAG
====================================================

Provides an SSE streaming endpoint where the LLM decides whether to call
``search_documents`` or answer directly, streaming thinking + tokens in real-time.

SSE Event Types:
  - status:         {"step": str, "detail": str}
  - thinking:       {"text": str}
  - sources:        {"sources": [...]}
  - images:         {"image_refs": [...]}
  - token:          {"text": str}
  - token_rollback: {}
  - complete:       {"answer": str, "sources": [...], ...}
  - error:          {"message": str}
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import string
import uuid
from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.models.user import User
from app.models.tenant import TenantUser
from app.models.knowledge_base import KnowledgeBase
from app.models.document import DocumentImage
from app.schemas.rag import (
    ChatRequest,
    ChatSourceChunk,
    ChatImageRef,
)
from app.services.llm.types import LLMMessage, LLMImagePart, StreamChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_AGENT_ITERATIONS = 3
MAX_VISION_IMAGES = 3
SSE_HEARTBEAT_INTERVAL = 15  # seconds

_CITATION_ID_CHARS = string.ascii_lowercase + string.digits


# ---------------------------------------------------------------------------
# Memory Extraction Agent
# ---------------------------------------------------------------------------

_MEMORY_EXTRACTION_PROMPT = (
    "You are a memory assistant that extracts personal facts about a HUMAN USER from their message.\n\n"
    "CRITICAL RULES:\n"
    "- You MUST only extract information that the HUMAN USER explicitly states about themselves.\n"
    "- NEVER extract statements made by an AI assistant or chatbot.\n"
    "- REJECT any statement starting with phrases like 'Tôi là trợ lý', 'Tôi là một AI', 'I am an AI', "
    "'I am a chatbot', 'As an assistant', 'Là một trợ lý AI'.\n"
    "- NEVER extract generic greetings, chatbot self-introductions, or conversational filler.\n"
    "- If the message is from an AI/chatbot, return exactly: []\n\n"
    "Extract ONLY if the user says things like:\n"
    "- Their name: 'Tôi tên là X', 'My name is X'\n"
    "- Their job or workplace: 'Tôi làm việc tại X', 'I work at X'\n"
    "- Their location: 'Tôi ở X', 'I live in X'\n"
    "- Their devices or tools: 'Tôi dùng iPhone 16', 'Tôi dùng MacBook'\n"
    "- Their preferences: 'Tôi thích X', 'I prefer X'\n"
    "- Direct instructions: 'Hãy gọi tôi là X', 'Always respond in Vietnamese'\n\n"
    "Target Categories:\n"
    "1. fact: Permanent information (name, job, location, devices). IMPORTANCE 10 for name, job.\n"
    "2. preference: Likes, dislikes, styles. IMPORTANCE 5-8.\n"
    "3. instruction: Direct rules for the assistant. IMPORTANCE 10.\n\n"
    "Output format: JSON array only. No explanation. Example:\n"
    '[{"content": "Tên người dùng là Hùng", "category": "fact", "importance": 10}, '
    '{"content": "Thiết bị: iPhone 16", "category": "fact", "importance": 10}]\n'
    "If nothing is worth extracting: []"
)


def _generate_citation_id(existing: set[str]) -> str:
    """Generate a unique 4-char alphanumeric citation ID."""
    while True:
        cid = "".join(random.choices(_CITATION_ID_CHARS, k=4))
        if any(c.isalpha() for c in cid) and cid not in existing:
            return cid


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

# Gemini native function calling
def _get_gemini_tool():
    """Lazily create Gemini Tool to avoid import at module level."""
    from google.genai import types
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description=(
                "Search the knowledge base for relevant document sections. "
                "Use this tool when the user asks about document content, data, or facts. "
                "IMPORTANT: Rewrite the user's question as a detailed, specific search query "
                "to get better retrieval results. "
                "Do NOT use this tool for greetings, chitchat, or non-document questions."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": (
                            "A rewritten, detailed search query based on the user's question. "
                            "Examples: 'revenue?' → 'total revenue figures and financial performance metrics'. "
                            "'AI là gì?' → 'định nghĩa trí tuệ nhân tạo, lịch sử và ứng dụng'"
                        ),
                    },
                    "top_k": {
                        "type": "INTEGER",
                        "description": "Number of relevant chunks to retrieve (default: 5, max: 10)",
                    },
                },
                "required": ["query"],
            },
        ),
    ])


# OpenAI-compatible native function calling (JSON schema format)
def _get_openai_tools() -> list:
    """Return tools in OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": (
                    "Search the knowledge base for relevant document sections. "
                    "Use this tool when the user asks about document content, data, or facts. "
                    "IMPORTANT: Rewrite the user's question as a detailed, specific search query "
                    "to get better retrieval results. "
                    "Do NOT use this tool for greetings, chitchat, or non-document questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A rewritten, detailed search query based on the user's question. "
                                "Examples: 'revenue?' → 'total revenue figures and financial performance metrics'. "
                                "'AI là gì?' → 'định nghĩa trí tuệ nhân tạo, lịch sử và ứng dụng'"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of relevant chunks to retrieve (default: 5, max: 10)",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Ollama prompt-based tool calling — MANDATORY search before answering
# ---------------------------------------------------------------------------

OLLAMA_TOOL_SYSTEM = """\
## TOOLS

You have ONE tool: search_documents.

### Tool: search_documents
Call it by outputting EXACTLY:
<tool_call>{"name": "search_documents", "arguments": {"query": "<rewritten query>"}}</tool_call>

### ABSOLUTE RULES (violations are FATAL errors)

1. **Except for simple conversational messages, ALWAYS CALL search_documents FIRST.**
   Simple conversational messages that do NOT require a tool call:
   - Greetings: "hello", "xin chào", "hi", "hey", "good morning", etc.
   - Acknowledgements: "cảm ơn", "thank you", "thanks", "ok", "got it", etc.
   - Farewells: "bye", "goodbye", "tạm biệt", etc.
   For ALL other messages — questions, requests, factual queries, analysis — you MUST
   call search_documents before answering. Your knowledge is UNRELIABLE; only document
   sources are trustworthy. If you are unsure whether a message needs a search, SEARCH.

2. **Your ENTIRE first response to a searchable query must be ONLY the <tool_call> block.**
   No text before it. No text after it. No explanation. Just the tool call.

3. **Rewrite the query** to be specific and detailed.
   "doanh thu" → "doanh thu thuần, tổng doanh thu theo năm, tăng trưởng doanh thu"
   "AI model" → "AI model architecture, performance benchmarks, training details"

4. After receiving search results, answer using ONLY those sources with citations.
   Format: claim text[source_id]. Example: Doanh thu đạt 4.850 tỷ VNĐ[id12].
"""

OLLAMA_TOOL_REMINDER = (
    "\n\n[SYSTEM REMINDER] If this is a question or request, you MUST call search_documents FIRST. "
    "Output ONLY: <tool_call>{\"name\": \"search_documents\", \"arguments\": {\"query\": \"...\"}}</tool_call> "
    "Exception: simple greetings, thanks, or farewells do NOT require a tool call — respond directly. "
    "For everything else, searching is MANDATORY. "
    "When answering from search results, use the provided source IDs for citations (e.g., [id12]).\n"
)

# ---------------------------------------------------------------------------
# Gemini system prompt reinforcement — enforce tool calling for questions
# ---------------------------------------------------------------------------

GEMINI_TOOL_SYSTEM = """\

## Tool Usage (MANDATORY)

You have one tool: `search_documents`.

### search_documents
Searches the knowledge base for relevant document sections.

### ABSOLUTE RULES:
1. For ALL user questions, requests, factual queries, or analysis — you MUST call \
`search_documents` FIRST before answering. Even if the conversation history \
contains relevant information, you MUST search again to get fresh, accurate sources.
2. Only skip the tool call for simple conversational messages:
   - Greetings: "hello", "xin chào", "hi", "hey", "good morning", etc.
   - Acknowledgements: "cảm ơn", "thank you", "thanks", "ok", "got it", etc.
   - Farewells: "bye", "goodbye", "tạm biệt", etc.
3. Use the unique 4-character ID provided in the search results context (e.g., [id12]) \
for your citations. DO NOT use example IDs from these instructions unless they \
match the search results.
4. NEVER answer a question using information from previous turns without searching. \
Your previous answers may contain outdated or incomplete information.
5. NEVER reuse citation IDs from previous answers. Each answer must have its own \
fresh sources from a new search.
6. Rewrite the user's query to be specific and detailed for better retrieval.
"""

# ---------------------------------------------------------------------------
# OpenAI-compatible system prompt reinforcement
# ---------------------------------------------------------------------------

OPENAI_COMPATIBLE_TOOL_SYSTEM = """
## Tool Usage (MANDATORY)

You have access to the `search_documents` tool.
- You MUST use this tool to answer any questions about document content, specific data, or analysis.
- Do NOT rely on your internal knowledge or previous answers for document-related facts.
- Skipping the tool call for document-related questions is a failure to follow instructions.
- If you are unsure whether a search is needed, PERFORM THE SEARCH.
"""

NATIVE_TOOL_REMINDER = (
    "\n\n[SYSTEM REMINDER] You MUST call the `search_documents` tool before answering this query. "
)


# ---------------------------------------------------------------------------
# SSE Helpers (ported from PageIndex backend/app/api/v1/chat.py)
# ---------------------------------------------------------------------------

def format_sse_event(event: str, data: dict) -> str:
    """Format data as an SSE event string."""
    json_data = json.dumps(data, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


async def sse_with_heartbeat(
    source: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Wrap an SSE generator with periodic heartbeat comments.

    SSE spec allows lines starting with ':' as comments — browsers/clients
    silently ignore them but they keep the TCP connection alive, preventing
    timeouts when the upstream LLM takes a long time to respond.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _pump():
        try:
            async for event in source:
                await queue.put(event)
        except Exception:
            pass
        finally:
            await queue.put(None)  # sentinel

    task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=SSE_HEARTBEAT_INTERVAL
                )
                if event is None:
                    break
                yield event
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Tool executor — retrieval via HRAG
# ---------------------------------------------------------------------------

async def _get_accessible_workspaces(db: AsyncSession, user: User) -> list[int]:
    """Get all knowledge base IDs the user has access to."""
    if user.is_superadmin:
        result = await db.execute(select(KnowledgeBase.id))
        return list(result.scalars().all())

    # Get user's tenants
    tenant_result = await db.execute(select(TenantUser.tenant_id).where(TenantUser.user_id == user.id))
    user_tenant_ids = list(tenant_result.scalars().all())

    from sqlalchemy import or_
    query = select(KnowledgeBase.id).where(
        or_(
            KnowledgeBase.visibility == "public",
            KnowledgeBase.owner_id == user.id,
            KnowledgeBase.tenant_id.in_(user_tenant_ids) if user_tenant_ids else False
        )
    )
    result = await db.execute(query)
    return list(result.scalars().all())


async def _execute_search_documents(
    workspace_ids: list[int],
    query: str,
    top_k: int,
    db: AsyncSession,
    existing_ids: set[str],
) -> tuple[str, list[ChatSourceChunk], list[ChatImageRef], list[dict], list[str]]:
    """Execute document search across multiple workspaces and return best chunks.

    Returns:
        (context_text, sources, image_refs, image_parts_for_vision, kg_summaries)
    """
    from app.services.rag_service import get_rag_service
    from app.services.hrag_service import HRAGService
    from pathlib import Path as _P
    
    all_chunks = []
    all_kg_summaries = []
    
    # Get workspace titles for better labeling
    from app.models.knowledge_base import KnowledgeBase
    ws_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id.in_(workspace_ids)))
    ws_map = {ws.id: ws.name for ws in ws_result.scalars().all()}
    
    for workspace_id in workspace_ids:
        logger.info(f"[RAG] External Search: query='{query}' on workspace {workspace_id}")
        rag_service = get_rag_service(db, workspace_id)
        ws_name = ws_map.get(workspace_id, f"KB {workspace_id}")
        
        chunks = []
        citations = []
        if isinstance(rag_service, HRAGService):
            try:
                result = await rag_service.query_deep(
                    question=query,
                    top_k=min(top_k, 10),
                    mode="hybrid",
                    include_images=False,
                )
                chunks = result.chunks
                citations = result.citations
                if result.knowledge_graph_summary:
                    all_kg_summaries.append(f"### KG Insights from {ws_name}\n{result.knowledge_graph_summary}")
            except Exception as e:
                logger.warning(f"Search failed for workspace {workspace_id}: {e}")
        else:
            from types import SimpleNamespace
            try:
                legacy = rag_service.query(question=query, top_k=min(top_k, 10))
                for i, c in enumerate(legacy.chunks):
                    chunks.append(SimpleNamespace(
                        content=c.content,
                        document_id=int(c.metadata.get("document_id", 0)),
                        chunk_index=i,
                        page_no=int(c.metadata.get("page_no", 0)),
                        heading_path=str(c.metadata.get("heading_path", "")).split(" > ") if c.metadata.get("heading_path") else [],
                        source_file=str(c.metadata.get("source", "")),
                        image_refs=[],
                        score=c.score
                    ))
            except Exception as e:
                logger.warning(f"Legacy search failed for workspace {workspace_id}: {e}")
                
        # Pack chunks with their citation for sorting
        for i, chunk in enumerate(chunks):
            citation = citations[i] if i < len(citations) else None
            score = getattr(chunk, "score", 0.0)
            all_chunks.append((score, chunk, citation, workspace_id))
            
    # Sort all aggregated chunks by score descending
    all_chunks.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"[RAG] Found total {len(all_chunks)} potential chunks across {len(workspace_ids)} workspaces")
    
    # Take top_k
    best_chunks = all_chunks[:top_k]

    # Build sources
    sources: list[ChatSourceChunk] = []
    context_parts: list[str] = []
    chunk_image_ids: list[str] = []
    seen_image_ids: set[str] = set()
    source_pages = set()
    
    for score, chunk, citation, workspace_id in best_chunks:
        cid = _generate_citation_id(existing_ids)
        existing_ids.add(cid)
        sources.append(ChatSourceChunk(
            index=cid,
            chunk_id=f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}",
            content=chunk.content,
            document_id=chunk.document_id,
            page_no=chunk.page_no,
            heading_path=chunk.heading_path,
            score=score,
            source_type="vector",
        ))
        logger.info(f"[RAG] Selected Chunk [{cid}] (KB {workspace_id}) score={score:.3f}: {chunk.content[:60]}...")
        
        # Collect images to fetch
        for iid in getattr(chunk, "image_refs", []) or []:
            if iid and iid not in seen_image_ids:
                seen_image_ids.add(iid)
                chunk_image_ids.append(iid)
                
        if getattr(chunk, "page_no", 0) > 0:
            source_pages.add((getattr(chunk, "document_id", 0), getattr(chunk, "page_no", 0)))
        
        meta_parts = []
        if citation:
            meta_parts.append(citation.source_file)
            if citation.page_no:
                meta_parts.append(f"page {citation.page_no}")
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""
        if heading:
            meta_parts.append(heading)
        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
        context_parts.append(f"Source [{cid}]{meta_line}:\n{chunk.content}")

    context = ""
    if all_kg_summaries:
        context += "## Knowledge Graph Entities & Relationships\n"
        context += "\n\n".join(all_kg_summaries)
        context += "\n\n---\n\n"
        
    context += "## Document Chunks\n"
    context += "\n\n---\n\n".join(context_parts)

    resolved_images: list[DocumentImage] = []
    if chunk_image_ids:
        img_result = await db.execute(
            select(DocumentImage).where(DocumentImage.image_id.in_(chunk_image_ids))
        )
        resolved_images = list(img_result.scalars().all())

    if not resolved_images and source_pages:
        from sqlalchemy import or_, and_
        page_filters = [
            and_(
                DocumentImage.document_id == doc_id,
                DocumentImage.page_no == page_no,
            )
            for doc_id, page_no in source_pages
        ]
        img_result = await db.execute(
            select(DocumentImage).where(or_(*page_filters))
        )
        resolved_images = list(img_result.scalars().all())
        seen = set()
        deduped = []
        for img in resolved_images:
            if img.image_id not in seen:
                seen.add(img.image_id)
                deduped.append(img)
        resolved_images = deduped

    chat_image_refs: list[ChatImageRef] = []
    image_context_parts: list[str] = []
    image_parts: list[dict] = []

    for img in resolved_images[:MAX_VISION_IMAGES]:
        img_ref_id = _generate_citation_id(existing_ids)
        existing_ids.add(img_ref_id)
        # Figure out which workspace this image belongs to in order to construct the correct URL
        # For simplicity we query the document's workspace_id
        workspace_id = img.document.workspace_id if hasattr(img, "document") and img.document else workspace_ids[0] if workspace_ids else 0
        img_url = f"/static/doc-images/kb_{workspace_id}/images/{img.image_id}.png"
        chat_image_refs.append(ChatImageRef(
            ref_id=img_ref_id,
            image_id=img.image_id,
            document_id=img.document_id,
            page_no=img.page_no,
            caption=img.caption or "",
            url=img_url,
            width=img.width,
            height=img.height,
        ))
        cap = f'"{img.caption}"' if img.caption else "no caption"
        image_context_parts.append(f"- [IMG-{img_ref_id}] Page {img.page_no}: {cap}")

        img_path = _P(img.file_path)
        if img_path.exists():
            try:
                img_bytes = img_path.read_bytes()
                mime = img.mime_type or "image/png"
                image_parts.append({
                    "inline_data": {"mime_type": mime, "data": img_bytes},
                    "page_no": img.page_no,
                    "caption": img.caption or "",
                    "img_ref_id": img_ref_id,
                })
            except Exception as e:
                logger.warning(f"Failed to read image {img.image_id}: {e}")

    if image_context_parts:
        context += "\n\nDocument Images:\n" + "\n".join(image_context_parts)

    return context, sources, chat_image_refs, image_parts, all_kg_summaries


# ---------------------------------------------------------------------------
# Memory executor — save and retrieve persistent user memories
# ---------------------------------------------------------------------------

MAX_MEMORIES_PER_USER = 200

async def _save_memory(
    user_id: int,
    content: str,
    category: str,
    importance: int,
    session_id: Optional[str],
    db: AsyncSession,
) -> str:
    """Save a new memory for the user with its embedding."""
    from app.models.user_memory import UserMemory
    from app.services.llm import get_embedding_provider

    # Check limit
    count_result = await db.execute(
        select(func.count()).select_from(UserMemory).where(UserMemory.user_id == user_id)
    )
    count = count_result.scalar() or 0
    if count >= MAX_MEMORIES_PER_USER:
        return f"Memory limit reached ({MAX_MEMORIES_PER_USER}). Cannot save new memory."

    # Generate embedding
    try:
        emb_provider = get_embedding_provider()
        embedding = await emb_provider.embed([content])
        emb_list = embedding[0].tolist() if hasattr(embedding[0], 'tolist') else list(embedding[0])
    except Exception as e:
        logger.warning(f"Failed to generate memory embedding: {e}")
        emb_list = None

    memory = UserMemory(
        user_id=user_id,
        content=content,
        embedding=emb_list,
        category=category,
        importance=max(1, min(10, importance)),
        source_session_id=session_id,
    )
    db.add(memory)
    await db.flush()
    return f"Memory saved: {content[:80]}"


async def _search_memories(
    user_id: int,
    query: str,
    db: AsyncSession,
    top_k: int = 5,
) -> str:
    """Search user memories by semantic similarity using pgvector."""
    logger.info(f"[MEMORY] Searching personal history for: '{query}'")
    from app.models.user_memory import UserMemory
    from app.services.llm import get_embedding_provider

    try:
        emb_provider = get_embedding_provider()
        query_embedding = await emb_provider.embed([query])
        query_vec = query_embedding[0].tolist() if hasattr(query_embedding[0], 'tolist') else list(query_embedding[0])
    except Exception as e:
        logger.warning(f"Failed to generate query embedding for memory search: {e}")
        # Fallback: text search
        result = await db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == user_id)
            .order_by(UserMemory.importance.desc())
            .limit(top_k)
        )
        memories = result.scalars().all()
        if not memories:
            return "No memories found for this user."
        return "\n".join([f"- [{m.category}] {m.content}" for m in memories])

    # pgvector cosine distance search. We combine distance with importance.
    try:
        from pgvector.sqlalchemy import Vector
        # Higher importance and lower distance is better.
        # We order by (distance / importance) to favor important matches even if slightly less similar.
        result = await db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == user_id)
            .where(UserMemory.embedding.isnot(None))
            .order_by(UserMemory.embedding.cosine_distance(query_vec) * (10.0 / (UserMemory.importance + 1)))
            .limit(top_k + 5) # Retrieve more to be sure
        )
        memories = list(result.scalars().all())
    except Exception as e:
        logger.warning(f"pgvector search failed, falling back to importance: {e}")
        result = await db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == user_id)
            .order_by(UserMemory.importance.desc())
            .limit(top_k)
        )
        memories = result.scalars().all()

    if not memories:
        logger.info(f"[MEMORY] No memories found for user_id={user_id} and query='{query}'.")
        return "No relevant memories found."
    
    # Format and log memories
    context_parts = []
    for i, m in enumerate(memories):
        context_parts.append(f"• [{m.category}] {m.content}")
    context = "\n".join(context_parts)
    logger.info(f"[MEMORY] Found {len(memories)} relevant snippets for user_id={user_id}. Content preview: {context[:200]}...")
    return context


# ---------------------------------------------------------------------------
# Memory Extraction Agent

async def _extract_facts_with_llm(message: str) -> list[dict]:
    """Use the Memory Agent (small LLM) to extract structured facts from user message."""
    from app.services.llm import get_memory_agent
    import json

    agent = get_memory_agent()
    system_msg = LLMMessage(role="system", content=_MEMORY_EXTRACTION_PROMPT)
    user_msg = LLMMessage(role="user", content=f"User Message: {message}")

    try:
        response = ""
        async for chunk in agent.astream([system_msg, user_msg]):
            if chunk.text:
                response += chunk.text

        # Clean up JSON if LLM added markdown wrappers
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[-1].split("```")[0].strip()
        elif "```" in response:
            # Handle generic markdown block
            parts = response.split("```")
            if len(parts) >= 3:
                response = parts[1].strip()
            else:
                response = response.strip("`").strip()

        if not response or response == "[]":
            return []

        facts = json.loads(response)
        if isinstance(facts, list):
            return facts
        elif isinstance(facts, dict):
            return [facts]
        return []
    except Exception as e:
        logger.error(f"Memory extraction failed (Ollama error or invalid JSON): {e}")
        return []


async def _auto_save_memory(
    user_id: int,
    message: str,
    session_id: Optional[str],
    db: AsyncSession,
) -> None:
    """Detect and save user preferences/facts using AI extraction."""
    # Skip very short messages
    if len(message.strip()) < 10:
        return

    facts = await _extract_facts_with_llm(message)
    if not facts:
        return

    for fact in facts:
        content = fact.get("content")
        category = fact.get("category", "fact")
        importance = fact.get("importance", 5)

        if not content:
            continue

        try:
            await _save_memory(user_id, content, category, importance, session_id, db)
            logger.info(f"Auto-saved {category} for user {user_id}: {content[:50]}...")
        except Exception as e:
            logger.error(f"Failed to auto-save memory: {e}")


# ---------------------------------------------------------------------------
# Agent loop — semi-agentic streaming
# ---------------------------------------------------------------------------

async def agent_chat_stream(
    workspace_ids: list[int],
    message: str,
    history: list[dict],
    enable_thinking: bool,
    db: AsyncSession,
    system_prompt: str,
    force_search: bool = False,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Semi-agentic chat loop with streaming.

    - force_search=True: pre-search before calling LLM, inject sources as context.
      Guarantees retrieval for every query regardless of model tool-calling ability.
    - force_search=False (default): agentic tool-calling loop.
      Gemini uses native function calling; Ollama uses prompt-based tool calling.

    Yields dicts with 'event' and 'data' keys for SSE formatting.
    """
    from app.services.llm import get_llm_provider
    from app.core.config import settings

    provider = get_llm_provider()
    provider_name = settings.LLM_PROVIDER.lower()
    is_gemini = provider_name == "gemini"
    is_openai_compatible = provider_name == "openai_compatible"

    existing_ids: set[str] = set()
    all_sources: list[ChatSourceChunk] = []
    all_images: list[ChatImageRef] = []
    all_image_parts: list[dict] = []
    all_kg_summaries_collected: list[str] = []

    # Build conversation messages
    messages: list[LLMMessage] = []
    for msg in history[-10:]:
        role = "user" if msg.role == "user" else "assistant"
        messages.append(LLMMessage(role=role, content=msg.content))

    # Build user message
    messages.append(LLMMessage(role="user", content=message))

    # 1. Detect if it's a simple greeting (bypass search logic)
    greeting_detected = False
    import string
    low_msg = message.lower().strip(string.punctuation + " ")
    if low_msg in ["hi", "hello", "xin chào", "chào", "hey", "greetings", "cảm ơn", "thanks", "tạm biệt", "bye"]:
        greeting_detected = True

    # Tool / prompt setup
    tools = None
    effective_system_prompt = system_prompt

    # ── Auto-recall: inject relevant memories into system prompt ──────────
    if user_id:
        try:
            memory_context = await _search_memories(user_id, message, db, top_k=5)
            # Add memory context if available
            if memory_context and "No relevant memories" not in memory_context and "No memories found" not in memory_context:
                logger.info(f"Auto-recall injected {len(memory_context)} chars of memory context for user {user_id}")
                # Move User Context to a separate system block or make it VERY prominent
                messages.insert(0, LLMMessage(
                    role="system", 
                    content=f"AUTHENTICATED USER PROFILE (PERSONAL HISTORY):\n{memory_context}\n\n"
                            "The above data is from our internal database for this specific user. "
                            "If the user asks 'Who am I?', 'What is my name?', 'Where do I work?', or 'What device am I using?', "
                            "you MUST answer based ONLY on the data above. "
                            "When mentioning these facts, simply add the brain icon 🧠 to mark it as a memory. "
                            "DO NOT use numerical IDs like [1] or tags like [MEM-1]."
                ))
        except Exception as e:
            logger.warning(f"Auto-recall failed for user {user_id}: {e}")

    if force_search:
        # ── Force-search mode: pre-search before LLM call ──────────────────
        # Retrieve sources immediately, inject as context. No tool calling needed.
        yield {"event": "status", "data": {"step": "retrieving", "detail": f"Searching: {message[:80]}..."}}

        context, sources, images, img_parts, kg_summaries = await _execute_search_documents(
            workspace_ids, message, 8, db, existing_ids,
        )
        all_sources.extend(sources)
        all_images.extend(images)
        all_image_parts.extend(img_parts)
        all_kg_summaries_collected.extend(kg_summaries)

        if sources:
            yield {"event": "sources", "data": {"sources": [s.model_dump() for s in sources]}}
        if images:
            yield {"event": "images", "data": {"image_refs": [i.model_dump() for i in images]}}

        if sources:
            tool_result_parts = [
                "I have retrieved the following document sources for you.\n",
                "=== DOCUMENT SOURCES ===",
                context,
                "=== END SOURCES ===\n",
                "IMPORTANT:\n"
                "- Read EVERY source above carefully. Answers often require "
                "combining data from MULTIPLE sources.\n"
                "- TABLE DATA: Sources may contain table data as 'Key, Year = Value' pairs. "
                "Example: 'ROE, 2023 = 12,8%' means ROE was 12.8% in 2023.\n"
                "- If no source contains relevant information, check if your User Context has the answer.\n"
                "- If neither sources nor User Context have the answer, say: "
                "\"Tài liệu không chứa thông tin này.\"\n",
            ]
            tool_result_content = "\n".join(tool_result_parts)

            user_images_fs: list[LLMImagePart] = []
            if img_parts:
                for img_data in img_parts:
                    tool_result_content += f"\n[IMG-{img_data['img_ref_id']}] (page {img_data['page_no']}):"
                    user_images_fs.append(LLMImagePart(
                        data=img_data["inline_data"]["data"],
                        mime_type=img_data["inline_data"]["mime_type"],
                    ))

            tool_result_content += f"\n\nNow answer the question: {message}"
            messages.append(LLMMessage(
                role="user",
                content=tool_result_content,
                images=user_images_fs,
            ))
        # tools remain None — model answers directly with provided context
    elif is_gemini:
        tools = [_get_gemini_tool()]
        # Reinforce tool-calling obligation in system prompt for Gemini
        effective_system_prompt += GEMINI_TOOL_SYSTEM
        # Add a strong reminder directly to the user message
        messages[-1] = LLMMessage(
            role="user",
            content=messages[-1].content + NATIVE_TOOL_REMINDER,
        )
    elif is_openai_compatible:
        # OpenAI-compatible (vLLM): Reverting to XML-based tool calling for higher reliability with 30B/Local models
        if not greeting_detected:
            # Switch back to XML prompt-based tool calling
            effective_system_prompt += "\n\n" + OLLAMA_TOOL_SYSTEM
            messages[-1] = LLMMessage(
                role="user",
                content=messages[-1].content + OLLAMA_TOOL_REMINDER,
            )
            logger.info(f"[AGENT] Using XML-based tool calling for {provider_name}")
    else:
        # Ollama: append mandatory tool prompt to system prompt
        effective_system_prompt += "\n\n" + OLLAMA_TOOL_SYSTEM
        # Also append a reminder directly to the user message so the model
        # sees it right before generating — reinforces the tool requirement
        messages[-1] = LLMMessage(
            role="user",
            content=messages[-1].content + OLLAMA_TOOL_REMINDER,
        )

    yield {"event": "status", "data": {"step": "analyzing", "detail": "Analyzing your question..."}}

    accumulated_text = ""
    thinking_text = ""

    logger.info(f"--- AGENT CHAT START (provider={provider_name}, user_id={user_id}, workspaces={workspace_ids}) ---")
    logger.info(f"User Question: {message}")
    logger.info(f"System Prompt Length: {len(effective_system_prompt)}")

    for iteration in range(MAX_AGENT_ITERATIONS):
        iteration_idx = iteration + 1
        logger.info(f"[AGENT] Iteration {iteration_idx}/{MAX_AGENT_ITERATIONS} (tools_available={tools is not None})")
        iteration_text = ""
        function_calls: list[dict] = []
        tokens_yielded = False

        async for chunk in provider.astream(
            messages,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            system_prompt=effective_system_prompt,
            think=enable_thinking,
            tools=tools if (is_gemini or is_openai_compatible) else None,
        ):
            if chunk.type == "thinking":
                thinking_text += chunk.text
                yield {"event": "thinking", "data": {"text": chunk.text}}
            elif chunk.type == "function_call":
                function_calls.append(chunk.function_call)
            elif chunk.type == "text":
                iteration_text += chunk.text
                # Speculative streaming — send tokens if no tool call seen yet
                if not function_calls:
                    accumulated_text += chunk.text
                    tokens_yielded = True
                    yield {"event": "token", "data": {"text": chunk.text}}

        if function_calls:
            # Rollback speculative tokens
            if tokens_yielded:
                accumulated_text = ""
                yield {"event": "token_rollback", "data": {}}

            fc = function_calls[0]
            fc_name = fc.get("name", "")
            fc_args = fc.get("args", {})
            logger.info(f"[AGENT] LLM requested tool: {fc_name} with args: {fc_args}")

            if fc_name == "search_documents":
                query = fc_args.get("query", message)
                top_k = int(fc_args.get("top_k", 8))

                yield {"event": "status", "data": {
                    "step": "retrieving",
                    "detail": f"Searching: {query[:80]}..."
                }}

                context, sources, images, img_parts, kg_summaries = await _execute_search_documents(
                    workspace_ids, query, top_k, db, existing_ids,
                )
                logger.info(f"Search found {len(sources)} chunks and {len(kg_summaries)} KG summaries")
                all_sources.extend(sources)
                all_images.extend(images)
                all_image_parts.extend(img_parts)
                all_kg_summaries_collected.extend(kg_summaries)

                if sources:
                    yield {"event": "sources", "data": {
                        "sources": [s.model_dump() for s in sources]
                    }}
                if images:
                    yield {"event": "images", "data": {
                        "image_refs": [i.model_dump() for i in images]
                    }}

                # Build tool result as user message with sources
                tool_result_parts = [
                    "I have retrieved the following document sources for you.\n",
                    "=== DOCUMENT SOURCES ===",
                    context,
                    "=== END SOURCES ===\n",
                    "IMPORTANT:\n"
                    "- Read EVERY source above carefully. Answers often require "
                    "combining data from MULTIPLE sources.\n"
                    "- TABLE DATA: Sources may contain table data as 'Key, Year = Value' pairs. "
                    "Example: 'ROE, 2023 = 12,8%' means ROE was 12.8% in 2023.\n"
                    "- If no source contains relevant information, check if your User Context has the answer.\n"
                    "- If neither sources nor User Context have the answer, say: "
                    "\"Tài liệu không chứa thông tin này.\"\n",
                ]
                tool_result_content = "\n".join(tool_result_parts)

                # Add image inline references for vision models
                user_images: list[LLMImagePart] = []
                if img_parts:
                    for img_data in img_parts:
                        tool_result_content += f"\n[IMG-{img_data['img_ref_id']}] (page {img_data['page_no']}):"
                        user_images.append(LLMImagePart(
                            data=img_data["inline_data"]["data"],
                            mime_type=img_data["inline_data"]["mime_type"],
                        ))

                tool_result_content += f"\n\nNow answer the question: {message}"

                if is_gemini:
                    # Gemini: use native Content with thought_signature
                    # (required by Gemini 3 for proper multi-turn reasoning)
                    # and native FunctionResponse for the tool result.
                    from google.genai import types as _gtypes

                    raw_content = getattr(provider, "last_response_content", None)
                    if raw_content:
                        # Preserve the model's raw response (with thought_signature)
                        messages.append(LLMMessage(
                            role="assistant",
                            content="",
                            _raw_provider_content=raw_content,
                        ))
                    else:
                        messages.append(LLMMessage(
                            role="assistant",
                            content=f"[Called search_documents(query=\"{query}\")]",
                        ))

                    # Build native FunctionResponse with sources context
                    func_resp_parts = [_gtypes.Part.from_function_response(
                        name="search_documents",
                        response={"result": tool_result_content},
                    )]
                    func_resp_content = _gtypes.Content(
                        role="user",
                        parts=func_resp_parts,
                    )
                    messages.append(LLMMessage(
                        role="user",
                        content="",
                        _raw_provider_content=func_resp_content,
                    ))

                    # Send images as a separate user message for vision
                    if img_parts:
                        img_llm_parts: list[LLMImagePart] = []
                        img_text = "Referenced document images:\n"
                        for img_data in img_parts:
                            img_text += f"[IMG-{img_data['img_ref_id']}] (page {img_data['page_no']})\n"
                            img_llm_parts.append(LLMImagePart(
                                data=img_data["inline_data"]["data"],
                                mime_type=img_data["inline_data"]["mime_type"],
                            ))
                        messages.append(LLMMessage(
                            role="user",
                            content=img_text,
                            images=img_llm_parts,
                        ))

                    # Remove tool-calling instructions since search is done;
                    # keep tools so thinking + tool awareness still works.
                    effective_system_prompt = system_prompt
                else:
                    # Ollama: add text-based assistant + user messages
                    # to maintain proper user/assistant alternation
                    # (prevents two consecutive user messages which confuses
                    # small models like qwen3.5).
                    messages.append(LLMMessage(
                        role="assistant",
                        content=f"[Called search_documents(query=\"{query}\")]",
                    ))
                    messages.append(LLMMessage(
                        role="user",
                        content=tool_result_content,
                        images=user_images,
                    ))
                    # Remove tool prompt from system prompt so the model
                    # answers with sources instead of calling the tool again.
                    effective_system_prompt = system_prompt

                yield {"event": "status", "data": {
                    "step": "generating",
                    "detail": "Generating answer..."
                }}
            else:
                # Unknown tool — treat accumulated text as answer
                logger.warning(f"Unknown tool call: {fc_name}")
                break
        else:
            # No tool call from model — answer is in accumulated_text, done.
            break

    # ── Fallback: model produced no text and no search was done ──────────
    # Small Ollama models (e.g. qwen3.5:4b) may output thinking about
    # needing to search but never produce a <tool_call> tag or any text.
    # Auto-search and retry once to avoid "Unable to generate a response."
    if not accumulated_text and not all_sources and not is_gemini:
        logger.warning(
            "Ollama produced no text and no tool call — fallback to auto-search"
        )
        yield {"event": "status", "data": {
            "step": "retrieving",
            "detail": f"Searching: {message[:80]}..."
        }}

        context, sources, images, img_parts, kg_summaries = await _execute_search_documents(
            workspace_ids, message, 8, db, existing_ids,
        )
        all_kg_summaries_collected.extend(kg_summaries)
        all_sources.extend(sources)
        all_images.extend(images)
        all_image_parts.extend(img_parts)

        if sources:
            yield {"event": "sources", "data": {
                "sources": [s.model_dump() for s in sources]
            }}
        if images:
            yield {"event": "images", "data": {
                "image_refs": [i.model_dump() for i in images]
            }}

        if sources:
            fallback_parts = [
                "I have retrieved the following document sources for you.\n",
                "=== DOCUMENT SOURCES ===",
                context,
                "=== END SOURCES ===\n",
                "IMPORTANT:\n"
                "- Read EVERY source above carefully.\n"
                "- If no source contains relevant information, check if your User Context has the answer.\n"
                "- If neither sources nor User Context have the answer, say: "
                "\"Tài liệu không chứa thông tin này.\"\n",
            ]
            fallback_content = "\n".join(fallback_parts)
            fallback_content += f"\n\nNow answer the question: {message}"

            # Remove old tool system prompt, add sources as context
            fallback_msgs = messages.copy()
            fallback_msgs.append(LLMMessage(role="user", content=fallback_content))

            yield {"event": "status", "data": {
                "step": "generating", "detail": "Generating answer..."
            }}

            async for chunk in provider.astream(
                fallback_msgs,
                temperature=0.1,
                max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                system_prompt=system_prompt,  # original prompt without tool instructions
                think=enable_thinking,
                tools=None,
            ):
                if chunk.type == "thinking":
                    thinking_text += chunk.text
                    yield {"event": "thinking", "data": {"text": chunk.text}}
                elif chunk.type == "text":
                    accumulated_text += chunk.text
                    yield {"event": "token", "data": {"text": chunk.text}}

    # Extract related entities from KG (best-effort)
    related_entities: list[str] = []
    try:
        from app.services.rag_service import get_kg_service
        # Use first workspace for KG summary extraction (best effort)
        target_kg_id = workspace_ids[0] if workspace_ids else None
        if target_kg_id:
            kg = await get_kg_service(target_kg_id)
            entities = await kg.get_entities(limit=500)
            entity_names = {e["name"].lower(): e["name"] for e in entities}
        else:
            entity_names = {}
        
        # Search in generated answer
        text_lower = accumulated_text.lower()
        for lower_name, original_name in entity_names.items():
            if len(lower_name) >= 3 and lower_name in text_lower:
                related_entities.append(original_name)
        
        # Also search in KG summaries from search results (ensure graph shows relevant context)
        if all_kg_summaries_collected:
            summary_text_lower = "\n".join(all_kg_summaries_collected).lower()
            for lower_name, original_name in entity_names.items():
                if original_name not in related_entities:
                    if len(lower_name) >= 3 and lower_name in summary_text_lower:
                        related_entities.append(original_name)
    except Exception:
        pass

    # Strip artifacts
    if accumulated_text:
        accumulated_text = re.sub(r'<unused\d+>:?\s*', '', accumulated_text).strip()
        logger.info(f"[AGENT] Final Answer: {accumulated_text[:200]}...")

    logger.info(f"--- AGENT CHAT COMPLETE (Length={len(accumulated_text) if accumulated_text else 0}) ---")
    logger.info(f"Related Entities Found: {related_entities}")

    yield {"event": "complete", "data": {
        "answer": accumulated_text or "Unable to generate a response.",
        "sources": [s.model_dump() for s in all_sources],
        "image_refs": [i.model_dump() for i in all_images],
        "thinking": thinking_text or None,
        "related_entities": related_entities[:30],
    }}

    # ── Auto-save: detect and save user preferences/facts ────────────────
    if user_id and message:
        try:
            from app.core.database import async_session_maker
            uid = user_id
            sid = session_id
            async def _bg_save():
                try:
                    async with async_session_maker() as bg_db:
                        if uid is not None:
                            await _auto_save_memory(uid, message, sid, bg_db)
                        await bg_db.commit()
                except Exception as e:
                    logger.warning(f"Background auto-save failed: {e}")
            
            import asyncio
            asyncio.create_task(_bg_save())
        except Exception as e:
            logger.warning(f"Auto-save memory task spawn failed: {e}")


# ---------------------------------------------------------------------------
# SSE Streaming endpoint
# ---------------------------------------------------------------------------

async def chat_stream_endpoint(
    workspace_ids: list[int],
    request: ChatRequest,
    db: AsyncSession,
    user: User,
    session_id: str | None = None,
):
    """SSE streaming chat endpoint.

    Called from rag.py router — not a standalone router to avoid circular imports.
    """
    # Verify first workspace access (or all? for now just first as primary)
    primary_id = workspace_ids[0] if workspace_ids else None
    if not primary_id:
        raise HTTPException(status_code=400, detail="No workspace IDs provided")

    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == primary_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Primary knowledge base {primary_id} not found",
        )

    # Build system prompt — use document-type-specific prompt if applicable
    from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT
    base_prompt = kb.system_prompt or DEFAULT_SYSTEM_PROMPT

    # Check if there are documents in this workspace with a dominant document type,
    # and use the corresponding system prompt (workspace-specific > global > default)
    try:
        from sqlalchemy import select as _sel, func as _func
        from app.models.document import Document as _Doc, DocumentStatus as _DS
        from app.models.document_type import DocumentType as _DT, DocumentTypeSystemPrompt as _DTSP

        # Find most common document_type_id in INDEXED documents of these workspaces
        dominant_type_result = await db.execute(
            _sel(_Doc.document_type_id, _func.count(_Doc.id).label("cnt"))
            .where(
                _Doc.workspace_id.in_(workspace_ids),
                _Doc.status.in_([_DS.INDEXED, _DS.BUILDING_KG]),
                _Doc.document_type_id.isnot(None),
            )
            .group_by(_Doc.document_type_id)
            .order_by(_func.count(_Doc.id).desc())
            .limit(1)
        )
        dominant_row = dominant_type_result.first()
        if dominant_row and dominant_row.document_type_id:
            # Try workspace-specific prompt (from primary or any?)
            ws_prompt_res = await db.execute(
                _sel(_DTSP).where(
                    _DTSP.document_type_id == dominant_row.document_type_id,
                    _DTSP.workspace_id == primary_id,
                )
            )
            ws_prompt = ws_prompt_res.scalar_one_or_none()
            if ws_prompt:
                base_prompt = ws_prompt.system_prompt
            else:
                # Try global prompt for this document type
                global_prompt_res = await db.execute(
                    _sel(_DTSP).where(
                        _DTSP.document_type_id == dominant_row.document_type_id,
                        _DTSP.workspace_id.is_(None),
                    )
                )
                global_prompt = global_prompt_res.scalar_one_or_none()
                if global_prompt:
                    base_prompt = global_prompt.system_prompt
    except Exception as _sp_err:
        logger.debug(f"Document-type system prompt resolution failed (non-fatal): {_sp_err}")

    system_prompt = base_prompt + HARD_SYSTEM_PROMPT

    # Build history
    history = [{"role": m.role, "content": m.content} for m in request.history]

    # Persist user message immediately
    try:
        from app.models.chat_message import ChatMessage as ChatMessageModel
        user_row = ChatMessageModel(
            workspace_id=primary_id,
            message_id=str(uuid.uuid4()),
            role="user",
            content=request.message,
            user_id=user.id,
            session_id=session_id,
        )
        db.add(user_row)
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist user message: {e}")
        await db.rollback()

    async def event_generator() -> AsyncGenerator[str, None]:
        final_answer = ""
        final_sources = []
        final_images = []
        final_thinking = None
        final_entities = []

        # Collect agent steps for persistence (ThinkingTimeline survives reload)
        collected_steps: list[dict] = []
        step_counter = 0
        # Track sources/images as they arrive so sources_found inserts BEFORE generating
        streaming_sources: list[dict] = []
        streaming_images: list[dict] = []

        try:
            async for event in agent_chat_stream(
                workspace_ids=workspace_ids,
                message=request.message,
                history=history,
                enable_thinking=request.enable_thinking,
                db=db,
                system_prompt=system_prompt,
                force_search=request.force_search,
                session_id=session_id,
                user_id=user.id,
            ):
                event_type = event["event"]
                event_data = event["data"]

                # Collect status steps; insert sources_found before "generating"
                if event_type == "status":
                    step_name = event_data.get("step", "analyzing")

                    # When generating starts, insert sources_found first (correct order)
                    if step_name == "generating" and streaming_sources:
                        step_counter += 1
                        badges = list(dict.fromkeys(
                            s.get("index", "") for s in streaming_sources[:6]
                        ))
                        collected_steps.append({
                            "id": f"step-{step_counter}",
                            "step": "sources_found",
                            "detail": f"Found {len(streaming_sources)} source{'s' if len(streaming_sources) != 1 else ''}",
                            "status": "completed",
                            "timestamp": 0,
                            "sourceCount": len(streaming_sources),
                            "imageCount": len(streaming_images),
                            "sourceBadges": badges,
                        })
                        streaming_sources.clear()
                        streaming_images.clear()

                    step_counter += 1
                    collected_steps.append({
                        "id": f"step-{step_counter}",
                        "step": step_name,
                        "detail": event_data.get("detail", ""),
                        "status": "completed",
                        "timestamp": 0,
                    })

                # Track sources/images as they arrive
                elif event_type == "sources":
                    streaming_sources.extend(event_data.get("sources", []))

                elif event_type == "images":
                    streaming_images.extend(event_data.get("image_refs", []))

                # Attach thinking text to the analyzing step
                elif event_type == "thinking":
                    thinking_fragment = event_data.get("text", "")
                    for s in collected_steps:
                        if s["step"] == "analyzing":
                            s["thinkingText"] = (s.get("thinkingText") or "") + thinking_fragment
                            break

                elif event_type == "complete":
                    final_answer = event_data.get("answer", "")
                    final_sources = event_data.get("sources", [])
                    final_images = event_data.get("image_refs", [])
                    final_thinking = event_data.get("thinking")
                    final_entities = event_data.get("related_entities", [])

                    # Fallback: if sources arrived but generating step was never emitted
                    if streaming_sources and not any(s["step"] == "sources_found" for s in collected_steps):
                        step_counter += 1
                        badges = list(dict.fromkeys(
                            s.get("index", "") for s in streaming_sources[:6]
                        ))
                        collected_steps.append({
                            "id": f"step-{step_counter}",
                            "step": "sources_found",
                            "detail": f"Found {len(streaming_sources)} source{'s' if len(streaming_sources) != 1 else ''}",
                            "status": "completed",
                            "timestamp": 0,
                            "sourceCount": len(streaming_sources),
                            "imageCount": len(streaming_images),
                            "sourceBadges": badges,
                        })

                    # Done step
                    step_counter += 1
                    collected_steps.append({
                        "id": f"step-{step_counter}",
                        "step": "done",
                        "detail": "Done",
                        "status": "completed",
                        "timestamp": 0,
                    })

                yield format_sse_event(event_type, event_data)

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield format_sse_event("error", {"message": str(e)})
        finally:
            # Persist assistant message ONLY if it's NOT a session-based chat
            # (session-based chat persistence is handled by the caller in chat_session.py)
            if final_answer and not session_id:
                try:
                    from app.models.chat_message import ChatMessage as ChatMessageModel
                    # Note: for multi-workspace search, we associate with the first workspace
                    # or leave NULL if no workspaces.
                    target_workspace_id = workspace_ids[0] if workspace_ids else None
                    
                    assistant_row = ChatMessageModel(
                        workspace_id=target_workspace_id,
                        message_id=str(uuid.uuid4()),
                        role="assistant",
                        content=final_answer,
                        sources=final_sources if final_sources else None,
                        related_entities=final_entities[:30] if final_entities else None,
                        image_refs=final_images if final_images else None,
                        thinking=final_thinking,
                        agent_steps=collected_steps if collected_steps else None,
                        user_id=user.id,
                    )
                    db.add(assistant_row)
                    await db.commit()
                except Exception as e:
                    logger.warning(f"Failed to persist assistant message: {e}")
                    await db.rollback()

    return StreamingResponse(
        sse_with_heartbeat(event_generator()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

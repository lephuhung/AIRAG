# AIRAG LangGraph Agent - Code Examples & Patterns

## 1. Using the Agent

### Basic Usage
```python
from app.services.agent import build_agent_graph, AgentState
from app.services.agent.streaming import build_initial_state, stream_agent_to_sse

# Build graph once (singleton cached)
graph = build_agent_graph()

# Prepare initial state
initial_state = build_initial_state(
    workspace_ids=[1, 2],
    message="BMNN là gì?",
    history=[],  # or conversation history
    system_prompt="You are a helpful assistant.",
    enable_thinking=False,
    db=db_session,
    user_id=123,
    session_id="sess-456",
)

# Stream events in real-time
async for sse_event in stream_agent_to_sse(graph, initial_state):
    print(sse_event)  # SSE-formatted string
    yield sse_event  # Send to client
```

### With Conversation History
```python
history = [
    {"role": "user", "content": "Hệ thống này làm gì?"},
    {"role": "assistant", "content": "Hệ thống này là một RAG..."},
]

initial_state = build_initial_state(
    workspace_ids=[1],
    message="Nó có thể làm những gì?",
    history=history,
    system_prompt="You are an expert on this system.",
    enable_thinking=True,  # Enable extended thinking
    db=db_session,
    user_id=user.id,
    document_ids=[10, 11],  # Optional: limit to specific docs
)

async for sse_event in stream_agent_to_sse(graph, initial_state):
    yield sse_event
```

---

## 2. State Management

### Creating AgentState Manually
```python
from app.services.agent.state import DEFAULT_STATE, AgentState
from langchain_core.messages import HumanMessage, AIMessage

# Start from defaults
state: AgentState = {
    **DEFAULT_STATE,
    "workspace_ids": [1, 2],
    "user_id": 123,
    "messages": [
        HumanMessage(content="What is BMNN?"),
        AIMessage(content="BMNN stands for..."),
        HumanMessage(content="Tell me more."),
    ],
    "system_prompt": "You are helpful.",
    "enable_thinking": False,
}

# LangGraph automatically applies reducers for accumulator fields
# (messages, sources, images, image_parts, kg_summaries)
```

### Accessing State in Nodes
```python
async def my_node(state: AgentState) -> dict:
    """Example node showing state access patterns."""
    
    # Read fields
    messages = state.get("messages", [])
    intent = state.get("intent", "search")
    sources = state.get("sources", [])
    
    # Accumulators are automatically merged by LangGraph
    # e.g., multiple nodes can append to "sources"
    new_sources = [{"index": 1, "content": "..."}]
    
    # Return partial update (only changed fields)
    return {
        "intent": "search",
        "rewritten_query": "Better query",
        "sources": new_sources,  # Will extend, not replace
    }
```

---

## 3. Implementing a Custom Node

### Simple Node Example
```python
import logging
from app.services.agent.streaming import push_event

logger = logging.getLogger(__name__)

async def my_custom_node(state: "AgentState") -> dict:
    """Custom node that processes intent."""
    
    intent = state.get("intent", "search")
    workspace_ids = state.get("workspace_ids", [])
    
    # Emit status event
    await push_event(state, "status", {
        "step": "processing",
        "detail": f"Processing intent: {intent}"
    })
    
    logger.info(f"[my_custom_node] Processing {intent} for workspaces {workspace_ids}")
    
    # Do processing...
    result = "some result"
    
    # Return partial state update
    return {
        "final_answer": result,
        "iterations": state.get("iterations", 0) + 1,
    }
```

### Node with Tool Calls
```python
async def search_node(state: "AgentState") -> dict:
    from app.services.agent.streaming import get_current_db
    from app.services.agent import tools
    
    query = state.get("rewritten_query", "")
    workspace_ids = state.get("workspace_ids", [])
    
    # Get DB from context (not state!)
    db = get_current_db()
    
    try:
        result = await tools.search_documents(
            query=query,
            top_k=5,
            workspace_ids=workspace_ids,
            existing_citation_ids=set(),
            db=db,
        )
        
        # Push to SSE
        await push_event(state, "sources", result["sources"])
        
        return {
            "sources": result["sources"],
            "images": result["images"],
            "tool_called": True,
        }
    except Exception as e:
        logger.error(f"[search_node] Failed: {e}")
        return {"sources": [], "tool_called": False}
```

---

## 4. Routing Logic

### Custom Routing Function
```python
from langgraph.graph import StateGraph

def my_router(state: AgentState) -> str:
    """Route based on intent."""
    intent = state.get("intent", "search")
    
    if intent == "greeting":
        return "direct_answer"
    elif intent.startswith("write_"):
        return "write_executor"
    elif intent == "search_abbr":
        # Custom abbreviation handling
        return "agent_rag_executor"
    else:
        return "default_path"

# In graph builder:
graph = StateGraph(AgentState)
graph.add_node("node1", node1_func)
graph.add_node("node2", node2_func)
graph.add_conditional_edges(
    "intent_classifier",
    my_router,
    {
        "direct_answer": "node1",
        "agent_rag_executor": "node2",
        "default_path": "node2",
    }
)
```

### Loop-Back Routing (Abbreviation Example)
```python
def should_retry_with_expansion(state: AgentState) -> str:
    """Check if abbreviation expansion should trigger retry."""
    
    # Expansion flag indicates there's more work to do
    expanded_query = state.get("expanded_query")
    
    # Don't loop forever
    iterations = state.get("iterations", 0)
    max_iter = 3
    
    if expanded_query and iterations < max_iter:
        logger.info(f"[router] Retrying with expanded query: {expanded_query!r}")
        return "retry"  # Go back to RAG node
    
    return "continue"  # Go to answer generator

graph.add_conditional_edges(
    "agent_rag_executor",
    should_retry_with_expansion,
    {
        "retry": "agent_rag_executor",      # Loop back
        "continue": "answer_generator",     # Continue
    }
)
```

---

## 5. State Transformation Patterns

### Subgraph Input/Output Transform
```python
def transform_input(state: AgentState) -> dict:
    """Map AgentState → ChildGraphState."""
    return {
        "messages": state.get("messages", []),
        "user_id": state.get("user_id"),
        "workspace_ids": state.get("workspace_ids", []),
        "input_text": state.get("text_input", ""),
        "intent": state.get("intent", "search"),
        # Note: exclude keys not in ChildGraphState TypedDict
    }

def transform_output(child_result: dict, existing_ids: set) -> dict:
    """Map ChildGraphState output → AgentState partial update."""
    sources = child_result.get("sources", []) or []
    
    # Process results
    processed = [s for s in sources if s.get("index") not in existing_ids]
    
    return {
        "sources": processed,  # Accumulator field - will extend
        "tool_called": bool(processed),
        "iterations": child_result.get("iterations", 0),
    }

async def node_with_subgraph(state: AgentState) -> dict:
    # Transform input
    child_input = transform_input(state)
    
    # Invoke subgraph
    try:
        child_result = await child_graph.ainvoke(child_input)
    except Exception as e:
        logger.error(f"Subgraph failed: {e}")
        child_result = {"sources": [], "iterations": 0}
    
    # Transform output
    partial = transform_output(child_result, state.get("existing_citation_ids", set()))
    
    # Push to SSE
    if partial["sources"]:
        await push_event(state, "sources", partial["sources"])
    
    return partial
```

---

## 6. Tool Implementation

### Creating a New Tool
```python
async def search_internal_kb(
    query: str,
    workspace_ids: list[int],
    db: "AsyncSession",
) -> dict:
    """Search company internal knowledge base."""
    
    from sqlalchemy import select, and_, or_
    from app.models.kb_article import KBArticle
    
    try:
        result = await db.execute(
            select(KBArticle)
            .where(
                KBArticle.workspace_id.in_(workspace_ids),
                KBArticle.is_published == True,
                or_(
                    KBArticle.title.ilike(f"%{query}%"),
                    KBArticle.content.ilike(f"%{query}%"),
                )
            )
            .limit(10)
        )
        articles = result.scalars().all()
        
        if not articles:
            return {"results": [], "text": f"No articles found for '{query}'"}
        
        lines = [f"Found {len(articles)} articles:"]
        results = []
        for article in articles:
            lines.append(f"- {article.title}")
            results.append({
                "title": article.title,
                "content": article.content,
                "url": f"/kb/{article.id}",
            })
        
        return {
            "results": results,
            "text": "\n".join(lines),
        }
    
    except Exception as e:
        logger.error(f"[search_internal_kb] Failed: {e}")
        return {"results": [], "text": "Error searching KB"}

# Register tool
TOOL_REGISTRY["search_internal_kb"] = "Search company knowledge base"

# Use in node
async def kb_search_node(state: "AgentState") -> dict:
    db = get_current_db()
    result = await search_internal_kb(
        query=state.get("rewritten_query", ""),
        workspace_ids=state.get("workspace_ids", []),
        db=db,
    )
    return {"kg_summaries": [result["text"]]}
```

---

## 7. Abbreviation Search Deep Dive

### Understanding the Expansion Logic
```python
# From tool_executor in nodes.py
abbreviation_results = tool_result.get("results") or (
    [tool_result] if tool_result.get("found") else []
)

if abbreviation_results:
    first_result = abbreviation_results[0]
    full_form = first_result.get("full_form", "")
    
    if full_form:
        # Heuristic: check if query has context beyond the abbreviation
        abbrev = first_result.get("abbreviation", "")
        abbrev_len = len(abbrev)
        query_len = len(query)  # The ORIGINAL query passed to search_abbreviation
        
        # If query significantly longer than abbreviation,
        # it probably contains context questions too
        has_context = query_len > abbrev_len + 10
        
        if has_context:
            # Replace abbreviation with full form
            # Example: "BMNN cũng quản lý những gì?" becomes
            #          "Bộ Môi Trường và Tài Nguyên cũng quản lý những gì?"
            expanded = query.replace(abbrev, full_form)
            result_update["expanded_query"] = expanded
            
            logger.info(
                f"[tool_executor] SEARCH_ABBR: Expanding query for routing. "
                f"Original: {query!r}, Expanded: {expanded!r}"
            )
        else:
            # Simple lookup, no context questions
            logger.info(
                f"[tool_executor] SEARCH_ABBR: Simple abbreviation lookup, "
                f"no expansion needed"
            )

# Later, in _should_continue_after_rag:
expanded_query = state.get("expanded_query")
if expanded_query and iterations < max_iter:
    # RETRY: Go back to agent_rag_executor with expanded query
    # This time it will call search_documents, not search_abbreviation
    return "agent_rag_executor"
else:
    return "answer_generator"
```

### Test Example
```python
# Scenario 1: Simple lookup
query1 = "BMNN"  # len = 4
abbrev_len = 4
query_len = 4
has_context = 4 > 4 + 10  # False
# Result: NO expansion

# Scenario 2: Lookup with context
query2 = "BMNN cũng quản lý những gì?"  # len = 37
abbrev_len = 4
query_len = 37
has_context = 37 > 4 + 10  # True (37 > 14)
# Result: EXPAND and RETRY

# Scenario 3: Edge case
query3 = "BMNN làm gì"  # len = 11
abbrev_len = 4
query_len = 11
has_context = 11 > 4 + 10  # False (11 < 14)
# Result: NO expansion (just under threshold)

# Scenario 4: Clear context
query4 = "BMNN có trách nhiệm gì với môi trường?"  # len = 40
abbrev_len = 4
query_len = 40
has_context = 40 > 4 + 10  # True (40 > 14)
# Result: EXPAND and RETRY
```

---

## 8. SSE Event Handling

### Consuming SSE Events on Frontend
```python
# Frontend JavaScript
const response = await fetch('/api/chat/stream', {
    method: 'POST',
    body: JSON.stringify({ message: "BMNN là gì?" }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n\n');
    
    for (const line of lines) {
        if (line.startsWith('event: status')) {
            const data = JSON.parse(line.split('data: ')[1]);
            updateStatus(data.step, data.detail);
        } else if (line.startsWith('event: token')) {
            const data = JSON.parse(line.split('data: ')[1]);
            appendToken(data.text);  // Append to answer
        } else if (line.startsWith('event: sources')) {
            const data = JSON.parse(line.split('data: ')[1]);
            displaySources(data.sources);
        } else if (line.startsWith('event: complete')) {
            const data = JSON.parse(line.split('data: ')[1]);
            saveComplete(data.answer, data.sources);
        }
    }
}
```

### Backend Endpoint
```python
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/chat/stream")
async def chat_stream(request: Request, message: str, db: AsyncSession):
    """Stream agent responses as SSE."""
    
    from app.services.agent import build_agent_graph
    from app.services.agent.streaming import build_initial_state, stream_agent_to_sse
    
    # Get user from request
    user = await get_current_user(request)
    
    # Build initial state
    initial_state = build_initial_state(
        workspace_ids=user.workspace_ids,
        message=message,
        history=request.chat_history or [],
        system_prompt="You are helpful.",
        enable_thinking=False,
        db=db,
        user_id=user.id,
        session_id=request.session_id,
    )
    
    # Get graph
    graph = build_agent_graph()
    
    # Stream
    async def generate():
        async for sse_event in stream_agent_to_sse(graph, initial_state):
            yield sse_event
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
```

---

## 9. Context Variables Pattern

### Using get_current_db()
```python
from app.services.agent.streaming import get_current_db

async def my_node(state: AgentState) -> dict:
    """Access DB from context, not state."""
    
    # CORRECT: Read from context
    db = get_current_db()
    
    # WRONG: This won't work
    # db = state.get("_db")  # LangGraph strips this key!
    
    # Use DB
    result = await db.execute(...)
    
    return {"final_answer": "..."}
```

### How It Works
```python
# In streaming.py
from contextvars import ContextVar

_db_ctx: ContextVar = ContextVar("_db", default=None)

async def stream_agent_to_sse(graph, initial_state):
    # Set context variable BEFORE creating task
    db_token = _db_ctx.set(initial_state.get("_db"))
    
    async def _run_graph():
        # All nodes created by create_task inherit this context
        await graph.ainvoke(initial_state)
    
    task = asyncio.create_task(_run_graph())  # Context inherited!

def get_current_db():
    """Nodes call this to access DB."""
    return _db_ctx.get()
```

---

## 10. Error Handling Patterns

### Graceful Degradation
```python
async def memory_recall(state: "AgentState") -> dict:
    """Non-blocking memory loading."""
    
    user_id = state.get("user_id")
    if not user_id:
        return {"user_memory_context": ""}
    
    try:
        from app.services.graphiti_client import search_user_memory
        memory_context = await search_user_memory(user_id, message, top_k=5)
        return {"user_memory_context": memory_context}
    except Exception as e:
        # LOG but don't fail — continue without memory
        logger.warning(f"[memory_recall] Graphiti unavailable: {e}")
        return {"user_memory_context": ""}

async def intent_classifier(state: "AgentState") -> dict:
    """Fallback to search on classifier failure."""
    
    try:
        classifier = get_memory_agent()
        response = await classifier.astream(...)
        return _parse_classifier_output(response)
    except Exception as e:
        # LOG and DEFAULT
        logger.error(f"[intent_classifier] Failed: {e}")
        return {
            "intent": "search",
            "rewritten_query": _extract_last_user_message(state),
        }
```

---

## 11. Logging Best Practices

```python
import logging

logger = logging.getLogger(__name__)

async def example_node(state: "AgentState") -> dict:
    intent = state.get("intent", "")
    query = state.get("rewritten_query", "")
    
    # ✓ Log with context
    logger.info(f"[example_node] Starting: intent={intent!r} query={query!r}")
    
    try:
        result = await do_something()
        logger.info(f"[example_node] Success: result_len={len(result)}")
        return {"final_answer": result}
    
    except ValueError as e:
        # ✓ Recoverable error
        logger.warning(f"[example_node] Invalid input: {e}")
        return {"final_answer": "Invalid query"}
    
    except Exception as e:
        # ✓ Fatal error with traceback
        logger.error(f"[example_node] Fatal: {e}", exc_info=True)
        raise
```

---

## 12. Testing Nodes

```python
import pytest
from app.services.agent.state import DEFAULT_STATE, AgentState

@pytest.mark.asyncio
async def test_intent_classifier():
    """Test classifier node."""
    
    state: AgentState = {
        **DEFAULT_STATE,
        "messages": [HumanMessage(content="BMNN là gì?")],
    }
    
    result = await intent_classifier(state)
    
    assert result["intent"] == "search_abbr"
    assert result["rewritten_query"] == "BMNN"

@pytest.mark.asyncio
async def test_abbreviation_expansion():
    """Test query expansion heuristic."""
    
    query_with_context = "BMNN cũng quản lý những gì?"
    abbrev = "BMNN"
    full_form = "Bộ Môi Trường và Tài Nguyên"
    
    # Test expansion condition
    has_context = len(query_with_context) > len(abbrev) + 10
    assert has_context  # Should be True
    
    # Test replacement
    expanded = query_with_context.replace(abbrev, full_form)
    expected = "Bộ Môi Trường và Tài Nguyên cũng quản lý những gì?"
    assert expanded == expected
```


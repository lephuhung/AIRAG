# AIRAG LangGraph Agent Implementation Analysis

**Project:** AIRAG (AI-Powered RAG System)  
**Location:** `/home/lph/Documents/GitHub/AIRAG/backend/app/services/agent/`  
**Last Updated:** 2026-03-26

---

## Directory Structure

```
backend/app/services/agent/
├── __init__.py           # Module exports (build_agent_graph, AgentState)
├── state.py             # LangGraph state definition (AgentState TypedDict)
├── nodes.py             # Node implementations (10K+ lines)
├── graph.py             # Graph builder & routing logic
├── tools.py             # Tool definitions (search, list, summarize, etc.)
├── streaming.py         # SSE streaming adapter with ContextVar workaround
└── [subgraph references]
    ├── agent_rag        # Subgraph for RAG operations
    └── agent_write      # Subgraph for text processing
```

---

## 1. Core State Definition (`state.py`)

### AgentState TypedDict
**Key-value store flowing through LangGraph nodes.**

#### Message & Context Fields:
```python
messages: Annotated[list, add_messages]           # Conversation history
workspace_ids: list[int]                          # Knowledge bases
document_ids: Optional[list[int]]                 # Specific documents
user_id: Optional[int]                            # User identifier
session_id: Optional[str]                         # Session identifier
system_prompt: str                                # Custom system instructions
enable_thinking: bool                             # Extended thinking mode
```

#### Retrieval Accumulators (with reducers):
```python
sources: Annotated[list, lambda a, b: a + b]     # ChatSourceChunk dicts
images: Annotated[list, lambda a, b: a + b]      # ChatImageRef dicts
image_parts: Annotated[list, lambda a, b: a + b] # Raw bytes for vision LLM
kg_summaries: Annotated[list, lambda a, b: a + b] # Knowledge graph insights
abbreviation_results: list[dict]                  # Abbreviation lookups
```

#### Agent Control Fields:
```python
intent: str                                       # Classification result
rewritten_query: str                              # Query after Qwen3-4B rewrite
iterations: int                                   # Loop guard counter
tool_called: bool                                 # RAG execution flag
expanded_query: str                               # Query after abbreviation expansion
```

#### Memory & Output:
```python
user_memory_context: str                          # Graphiti memory injection
final_answer: str                                 # Final response
citation_map: dict                                # Citation ID registry
write_action: str                                 # "summarize" | "suggest_edits" | "grammar_check"
text_input: str                                   # Raw text for write operations
```

#### Valid Intents:
```python
VALID_INTENTS = {
    "greeting",              # Chitchat → direct_answer
    "personal",             # Personal Q → direct_answer + memory
    "search",               # Document search → agent_rag_executor
    "list_docs",            # List documents → agent_rag_executor
    "summarize",            # Summarize doc → agent_rag_executor
    "kg_query",             # Knowledge graph → agent_rag_executor
    "search_doc_num",       # Search by number → agent_rag_executor
    "search_abbr",          # Abbreviation lookup → agent_rag_executor
    "write_summarize",      # Summarize text → write_executor
    "write_suggest_edits",  # Improve text → write_executor
    "write_grammar_check",  # Grammar check → write_executor
}
```

---

## 2. Node Implementations (`nodes.py`)

### 2.1 Node: `memory_recall(state) → dict`

**Purpose:** Load user memories from Graphiti (temporal knowledge graph)

**Flow:**
1. Extract user_id and last user message
2. Call `search_user_memory(user_id, user_message, top_k=5)`
3. Return formatted memory context string

**Output:** `{"user_memory_context": str}`

**Key Features:**
- Non-blocking — gracefully handles Graphiti unavailability
- Injects hybrid search (semantic + BM25 + graph traversal)
- Merged into system prompt by downstream nodes

---

### 2.2 Node: `intent_classifier(state) → dict`

**Purpose:** Classify intent and rewrite query using Qwen3-4B

**Classifier System Prompt:**
- 11 intent categories with detailed rules
- **Special rule for `search_abbr`:** Extract abbreviation only, remove "là gì?" suffix
- **Examples provided:** greetings, personal questions, document search, abbreviation queries
- **JSON Output:** `{intent, rewritten_query, needs_tool, write_action, text_input}`

**Implementation:**
```python
async def intent_classifier(state: AgentState) -> dict:
    user_message = _extract_last_user_message(state)
    classifier = get_memory_agent()  # Qwen3-4B model
    
    response_text = ""
    async for chunk in classifier.astream(
        [LLMMessage(role="user", content=user_message)],
        system_prompt=_CLASSIFIER_SYSTEM,
        temperature=0.0,  # Deterministic
        max_tokens=128,
    ):
        if chunk.text:
            response_text += chunk.text
    
    result = _parse_classifier_output(response_text)
    return {
        "intent": result["intent"],
        "rewritten_query": result["rewritten_query"] or user_message,
        "write_action": result.get("write_action", ""),
        "text_input": result.get("text_input", ""),
    }
```

**JSON Parsing Safety:**
- Strips markdown code fences (`\`\`\`json ... \`\`\``)
- Falls back to default intent="search" on parse failure

**Status Events:** Pushes human-friendly intent labels to SSE

---

### 2.3 Node: `agent_rag_executor(state) → dict`

**Purpose:** Invoke the agent_rag subgraph for all RAG operations

**State Transformation Pipeline:**
```
AgentState → _transform_rag_input() → AgentRagState
    ↓
agent_rag subgraph (routes by intent internally)
    ↓
AgentRagState → _transform_rag_output() → AgentState partial update
```

**Subgraph Output Integration:**
- Routes 6 intent types: `search`, `list_docs`, `summarize`, `kg_query`, `search_doc_num`, `search_abbr`
- Injects RAG final_answer into kg_summaries for answer_generator context
- Pushes sources/images to SSE queue

**Key Output Fields:**
```python
{
    "sources": list,                    # Retrieved document chunks
    "images": list,                     # Image references
    "image_parts": list,                # Raw image bytes
    "kg_summaries": list,               # Summary strings
    "abbreviation_results": list[dict], # Abbreviation data
    "tool_called": True,
    "iterations": 1,
}
```

**Routing Notes:**
- Abbreviation expansion: If query has context beyond abbreviation (len diff > 10),
  replace abbreviation with full_form and set `expanded_query` for retry logic
- Shared citation ID tracking prevents duplicates

---

### 2.4 Node: `answer_generator(state) → dict`

**Purpose:** Generate final answer using main LLM with retrieved context

**Context Assembly:**
1. **Knowledge Graph Results:** `## Knowledge Graph / Tool Results\n{kg_summaries}`
2. **Abbreviation Results:** `## Abbreviation Results\n- **ABBR** = full form\n  Mô tả: description`
3. **Document Chunks:** 
   ```
   Source [citation_id](source_file, page X, heading path):
   {content}
   ```

**System Prompt Injection:**
- USER MEMORY prepended if available (marked "Do NOT include header in response")
- Citation rules: "Cite using [id1][id2]"
- Fallback: "Tài liệu không chứa thông tin này."

**Streaming Implementation:**
```python
async for chunk in provider.astream(
    messages=llm_messages,
    temperature=0.1,
    max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
    system_prompt=effective_system,
    think=enable_thinking,
):
    if chunk.type == "thinking":
        await push_event(state, "thinking", {"text": chunk.text})
    elif chunk.type == "text":
        await push_event(state, "token", chunk.text)  # Real-time SSE
```

**Fallback:** Non-streaming `acomplete()` if streaming fails

**Output:** `{"final_answer": str}`

---

### 2.5 Node: `direct_answer(state) → dict`

**Purpose:** Answer greetings/personal questions without document retrieval

**Differs from answer_generator:**
- No retrieved context injected
- Higher temperature (0.5 vs 0.1) for more conversational tone
- Shorter max_tokens (512 vs configured max)
- Uses memory context if available (with special handling for "personal" intent)

**Memory Injection for Personal:**
```
"Answer directly about the user. Do NOT include the header 'USER MEMORY' in your response."
```

**Output:** `{"final_answer": str}`

---

### 2.6 Node: `write_executor(state) → dict`

**Purpose:** Invoke agent_write subgraph for text processing (summarize/edit/grammar)

**State Transformation:**
```python
def _transform_input(state: AgentState) -> dict:
    write_action = state.get("write_action") or {
        "write_summarize": "summarize",
        "write_suggest_edits": "suggest_edits", 
        "write_grammar_check": "grammar_check",
    }.get(intent, "summarize")
    
    text_input = state.get("text_input") or _extract_last_user_message(state)
    
    return {
        "messages": [],
        "user_id": state.get("user_id"),
        "workspace_ids": state.get("workspace_ids"),
        "text_input": text_input,
        "write_action": write_action,
        "result": "",
        "error": None,
    }
```

**Subgraph Invocation:**
```python
subgraph = _get_write_subgraph()  # Lazy-loaded singleton
write_output = await subgraph.ainvoke(write_input)
```

**Output:** `{"final_answer": str}` (from result or error fallback)

**Streaming:** Chunks result (~80 chars/chunk) into SSE tokens

---

## 3. Graph Builder (`graph.py`)

### Graph Topology

```
START
  ↓
memory_recall
  ↓
intent_classifier
  ├─ greeting/personal → direct_answer
  ├─ write_* → write_executor
  └─ search/list/summarize/kg/abbr/doc_num → agent_rag_executor
      ├─ [abbreviation + context detected]
      └─ → agent_rag_executor (retry with expanded_query)
           ↓
           answer_generator
END
```

### Routing Functions

#### `_route_by_intent(state) → str`
**Conditional edge after intent_classifier:**
```python
def _route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "search")
    
    if intent in ("greeting", "personal"):
        return "direct_answer"
    if intent in ("write_summarize", "write_suggest_edits", "write_grammar_check"):
        return "write_executor"
    return "agent_rag_executor"
```

#### `_should_continue_after_rag(state) → str`
**Conditional edge after agent_rag_executor:**
```python
def _should_continue_after_rag(state: AgentState) -> str:
    max_iter = getattr(settings, "NEXUSRAG_LG_MAX_ITERATIONS", 3)
    iterations = state.get("iterations", 0)
    
    if iterations >= max_iter:
        return "answer_generator"
    
    # Abbreviation expansion triggers re-routing
    expanded_query = state.get("expanded_query")
    if expanded_query and iterations < max_iter:
        return "agent_rag_executor"  # Loop back
    
    return "answer_generator"
```

### Graph Compilation

```python
def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    graph.add_node("memory_recall", memory_recall)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("agent_rag_executor", agent_rag_executor)
    graph.add_node("answer_generator", answer_generator)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("write_executor", write_executor)
    
    graph.add_edge(START, "memory_recall")
    graph.add_edge("memory_recall", "intent_classifier")
    
    graph.add_conditional_edges("intent_classifier", _route_by_intent, {...})
    graph.add_conditional_edges("agent_rag_executor", _should_continue_after_rag, {...})
    
    graph.add_edge("answer_generator", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("write_executor", END)
    
    return graph.compile()
```

### Singleton Pattern
```python
_agent_graph = None

def get_agent_graph() -> StateGraph:
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph

def reset_agent_graph() -> None:
    global _agent_graph
    _agent_graph = None
```

---

## 4. Tool Implementations (`tools.py`)

### Tool Registry
```python
TOOL_REGISTRY = {
    "search_documents": "hybrid vector+KG+BM25 search",
    "list_documents": "list all documents in workspace",
    "summarize_document": "comprehensive document summary",
    "query_knowledge_graph": "entity relationships from KG",
    "search_documents_number": "search by official document number (văn bản số)",
    "search_abbreviation": "search for abbreviation meanings",
}
```

### 4.1 Tool: `search_documents(query, top_k, workspace_ids, existing_citation_ids, db)`

**Wraps:** `_execute_search_documents()` from `api/chat_agent.py`

**Returns:**
```python
{
    "context_text": str,           # Full context string
    "sources": [ChatSourceChunk],  # Document chunks with citations
    "images": [ChatImageRef],      # Image references
    "image_parts": list,           # Raw image bytes for vision model
    "kg_summaries": list,          # Knowledge graph summaries
}
```

---

### 4.2 Tool: `list_documents(workspace_ids, db)`

**Queries:** Documents with status=INDEXED

**Returns:**
```python
{
    "text": "### Workspace: Name\n1. **filename** (ID: X), Y pages, Z chunks",
    "document_count": int,
}
```

---

### 4.3 Tool: `summarize_document(document_id, db)`

**Process:**
1. Fetch document from DB
2. Load markdown from MinIO (truncate to 16K chars)
3. Call main LLM with summarization prompt
4. Return summary

**Returns:**
```python
{
    "text": str,              # Generated summary
    "document_name": str,
    "document_id": int,
}
```

---

### 4.4 Tool: `query_knowledge_graph(entity, workspace_ids, db)`

**Queries:** LightRAG service (naive mode for speed)

**Returns:**
```python
{
    "text": "**Workspace W:** {kg_result}"
}
```

---

### 4.5 Tool: `search_documents_number(query, workspace_ids, db)`

**Queries:** Document.document_number ILIKE query

**Returns:**
```python
{
    "text": "Tìm thấy X tài liệu...",
    "documents": [{"id": int, "filename": str, "document_number": str}],
}
```

---

### 4.6 Tool: `search_abbreviation(abbreviation, workspace_ids, db)`

**⚠️ KEY FUNCTION FOR ABBREVIATION HANDLING**

**Database Query:**
```python
result = await db.execute(
    select(Abbreviation)
    .where(
        Abbreviation.short_form.ilike(f"%{abbreviation}%"),
        Abbreviation.is_active == True,
    )
    .limit(10)
)
```

**Three Return Scenarios:**

1. **Not Found (0 results):**
   ```python
   {
       "text": "Không tìm thấy nghĩa của '{abbr}'...",
       "abbreviation": abbreviation,
       "found": False,
   }
   ```

2. **Single Match (1 result):**
   ```python
   {
       "text": "**ABBR** = full form\nMô tả: description",
       "abbreviation": ab.short_form,
       "full_form": ab.full_form,
       "description": ab.description,
       "found": True,
   }
   ```

3. **Multiple Matches (>1 results):**
   ```python
   {
       "text": "Tìm thấy X kết quả cho '{abbr}':\n1. **ABBR1** = form1\n2. **ABBR2** = form2",
       "abbreviation": abbreviation,
       "found": True,
       "results": [
           {
               "short_form": str,
               "full_form": str,
               "description": str,
           }
       ]
   }
   ```

**Storage in AgentState:**
```python
abbreviation_results = [
    {"short_form": str, "full_form": str, "description": str}
]
```

**Context Injection (in answer_generator):**
```python
## Abbreviation Results
- **ABBR** = full form
  Mô tả: description
```

---

## 5. SSE Streaming Adapter (`streaming.py`)

### ⚠️ Critical Architecture: ContextVar Workaround

**Root Cause Issue:**
LangGraph TypedDict filtering strips non-declared keys before passing state to nodes.
→ Queue and DB session become inaccessible inside nodes

**Solution:**
Use Python `contextvars.ContextVar` to pass queue/db outside LangGraph state

### ContextVar Storage

```python
_event_queue_ctx: ContextVar[asyncio.Queue | None] = ContextVar("_event_queue", default=None)
_db_ctx: ContextVar = ContextVar("_db", default=None)

def get_current_db():
    """Nodes use this instead of state.get("_db")"""
    return _db_ctx.get()
```

### Streaming Flow

```python
async def stream_agent_to_sse(graph, initial_state) -> AsyncGenerator[str]:
    event_queue = asyncio.Queue()
    
    # Set ContextVars BEFORE create_task
    queue_token = _event_queue_ctx.set(event_queue)
    db_token = _db_ctx.set(initial_state.get("_db"))
    
    async def _run_graph():
        await graph.ainvoke(initial_state, debug=settings.NEXUSRAG_LG_DEBUG)
        await event_queue.put(("done", None))
    
    task = asyncio.create_task(_run_graph())  # Inherits context
    
    # Main loop: drain queue → yield SSE
    while True:
        item = await asyncio.wait_for(event_queue.get(), timeout=15)
        
        if item[0] == "done":
            yield _sse("complete", {...})
            break
        elif item[0] == "token":
            final_answer += item[1]
            yield _sse("token", {"text": item[1]})
        # ... other event types
```

### Event Types

```
event: status        → {"step": str, "detail": str}
event: thinking      → {"text": str}
event: sources       → {"sources": [ChatSourceChunk]}
event: images        → {"image_refs": [ChatImageRef]}
event: token         → {"text": str}
event: complete      → {"answer": str, "sources": [...], "images": [...]}
event: error         → {"message": str}
```

### Push Event from Nodes

```python
async def push_event(state: dict, ev_type: str, ev_data) -> None:
    queue = _event_queue_ctx.get()
    if queue is None and state:
        queue = state.get("_event_queue")  # Fallback for tests
    
    if queue is not None:
        await queue.put((ev_type, ev_data))
        await asyncio.sleep(0)  # Yield to event loop
```

### Initial State Building

```python
def build_initial_state(
    workspace_ids: list[int],
    message: str,
    history: list[dict],
    system_prompt: str,
    enable_thinking: bool,
    db,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    document_ids: Optional[list[int]] = None,
) -> dict:
    messages = [HumanMessage(content=msg["content"]) for msg in history]
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
        "_db": db,  # Set by stream_agent_to_sse into _db_ctx
    }
```

---

## 6. Abbreviation Search Flow (search_abbr) - DETAILED

### Intent Classification
**Classifier Rule:**
```
For "search_abbr" (abbreviation queries): extract ONLY the abbreviation itself 
- remove "là gì?", "có nghĩa là gì?", "là viết tắt của gì?", etc. 
- EXACT abbreviation must be preserved (e.g., "BMNN" stays "BMNN", NOT "BMM")
```

### Node Execution Path

1. **intent_classifier:**
   - Input: "BMNN là gì?"
   - Output: `intent="search_abbr", rewritten_query="BMNN"`

2. **_route_by_intent:**
   - Routes to `agent_rag_executor` (not direct_answer or write_executor)

3. **agent_rag_executor:**
   - Calls agent_rag subgraph with `intent="search_abbr", rewritten_query="BMNN"`

4. **tool_executor (inside agent_rag):**
   - Detects `intent == "search_abbr"`
   - Calls `search_abbreviation(abbreviation="BMNN", workspace_ids, db)`

5. **search_abbreviation Tool:**
   - Queries: `Abbreviation.short_form ILIKE "%BMNN%"`
   - Returns one of three scenarios (see section 4.6)

6. **Query Expansion Logic:**
   ```python
   # Check if abbreviation expansion should trigger re-routing
   abbreviation_results = tool_result.get("results") or ([tool_result] if tool_result.get("found") else [])
   
   if abbreviation_results:
       first_result = abbreviation_results[0]
       full_form = first_result.get("full_form", "")
       
       if full_form:
           # If original query has context beyond abbreviation
           abbrev_len = len(first_result.get("abbreviation", ""))
           query_len = len(query)  # "BMNN"
           has_context = query_len > abbrev_len + 10
           
           if has_context:
               # Example: "BMNN là viết tắt của gì và dùng ở đâu" 
               # → Expand to "Bộ Môi Trường và Tài Nguyên là viết tắt của gì và dùng ở đâu"
               expanded = query.replace(abbrev, full_form)
               result_update["expanded_query"] = expanded
               # Will trigger loop back to agent_rag_executor for semantic search
   ```

7. **Optional Loop-Back:**
   - If `expanded_query` set AND `iterations < max_iter(3)`:
     - Route back to `agent_rag_executor` with expanded query
     - Performs semantic search on full form context
   - Else: proceed to `answer_generator`

8. **answer_generator:**
   - Injects abbreviation results as `## Abbreviation Results\n- **BMNN** = Bộ...`
   - Generates final answer

### Example Scenarios

**Scenario A: Simple Abbreviation (No Context)**
```
User: "BMNN là gì?"
  → intent_classifier: intent="search_abbr", rewritten_query="BMNN"
  → search_abbreviation("BMNN")
    → Found: full_form="Bộ Môi Trường và Tài Nguyên"
    → No context (query_len=4, abbrev_len=4, diff < 10)
    → NO expanded_query
  → answer_generator (direct)
```

**Scenario B: Abbreviation with Context**
```
User: "BMNN cũng quản lý những gì?"
  → intent_classifier: rewritten_query="BMNN cũng quản lý những gì?"
  → search_abbreviation("BMNN cũng quản lý những gì?")
    → Found: full_form="Bộ Môi Trường và Tài Nguyên"
    → has_context = True (37 > 4+10)
    → expanded_query="Bộ Môi Trường và Tài Nguyên cũng quản lý những gì?"
  → _should_continue_after_rag: iterations=0 < 3 AND expanded_query set
    → Loop back to agent_rag_executor
  → search_documents("Bộ Môi Trường và Tài Nguyên cũi quản lý những gì?")
    → Hybrid search returns relevant documents
  → answer_generator (with sources + abbreviation context)
```

**Scenario C: Multiple Matches**
```
User: "HTTP là gì?"
  → search_abbreviation("HTTP")
    → Found: results=[
        {"short_form": "HTTP", "full_form": "HyperText Transfer Protocol", ...},
        {"short_form": "HTTP", "full_form": "...", ...}
      ]
    → abbreviation_results stores both
  → answer_generator includes all results in context
```

---

## 7. Key Design Patterns

### 7.1 Subgraph Delegation
- **Main graph nodes:** memory_recall → intent_classifier → routing → subgraphs → answer
- **Subgraph invocation:** agent_rag_executor and write_executor invoke child graphs
- **State transformation:** IN → _transform_*_input() → child → _transform_*_output() → OUT

### 7.2 Lazy-Loaded Singletons
```python
_write_subgraph = None

def _get_write_subgraph():
    global _write_subgraph
    if _write_subgraph is None:
        from app.services.agents.agent_write import create_agent_write
        _write_subgraph = create_agent_write()
    return _write_subgraph
```

### 7.3 Real-Time SSE with AsyncIO
- `asyncio.Queue` passes events from nodes to stream adapter
- `asyncio.create_task()` copies context → inherited ContextVars
- `push_event()` + `asyncio.sleep(0)` enables real-time token streaming

### 7.4 Intent-Based Routing
Three routing decisions:
1. **After intent_classifier:** greeting/personal → direct_answer; write_* → write_executor; else → agent_rag_executor
2. **After agent_rag_executor:** if expanded_query AND iterations < max → retry; else → answer_generator
3. **Terminal nodes:** All lead to END (no further routing)

### 7.5 Citation & Memory Management
- `existing_citation_ids: set` prevents duplicate citations
- `user_memory_context: str` injected into system prompt
- `abbreviation_results: list[dict]` accumulated for context injection

---

## 8. Configuration & Settings

From `app.core.config`:
- `NEXUSRAG_LG_MAX_ITERATIONS`: Max retries for RAG (default: 3)
- `NEXUSRAG_LG_DEBUG`: Enable debug logging
- `HRAG_RERANKER_TOP_K`: Number of retrieved chunks
- `LLM_MAX_OUTPUT_TOKENS`: Main LLM output limit

---

## 9. Error Handling

### Graceful Degradation
- **Memory recall fails:** Continue with empty context (non-blocking)
- **Classifier fails:** Default to intent="search"
- **LLM streaming fails:** Fallback to non-streaming `acomplete()`
- **Tool fails:** Return empty results dict, continue to answer_generator
- **Subgraph fails:** Return empty output dict

### Logging Strategy
```python
logger.info(f"[node_name] Meaningful context: {variable!r}")
logger.warning(f"[node_name] Recoverable issue: {detail}")
logger.error(f"[node_name] Fatal error: {e}", exc_info=True)
```

---

## 10. Summary

**AIRAG LangGraph Agent is a sophisticated multi-intent router implementing:**

1. **Intent Classification:** Qwen3-4B classifier routes to 11 intent types
2. **Dual Paths:** Document RAG (agent_rag) vs Text Writing (agent_write)
3. **Memory Integration:** Graphiti temporal knowledge graph injection
4. **Abbreviation Handling:** Smart abbreviation lookup with query expansion
5. **Real-Time Streaming:** SSE with ContextVar workaround for queue passing
6. **Fallback Chains:** Graceful degradation at every node
7. **Citation Management:** Deduplication via existing_citation_ids tracking

**Graph Flow:** START → memory_recall → intent_classifier → [direct_answer | write_executor | agent_rag_executor] → [optional retry] → answer_generator → END


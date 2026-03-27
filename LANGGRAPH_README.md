# AIRAG LangGraph Agent - Complete Documentation

This directory contains comprehensive documentation of the AIRAG LangGraph agent implementation.

## 📚 Documentation Files

### 1. **LANGGRAPH_ANALYSIS.md** ⭐ START HERE
**Complete technical analysis** (~10,000 words)
- Full state definition (65 fields)
- 6 node implementations with code details
- Graph topology and routing logic
- 6 tool implementations
- SSE streaming architecture (ContextVar workaround)
- Abbreviation search flow with examples
- Design patterns and error handling

👉 **Best for:** Understanding the full system architecture

---

### 2. **LANGGRAPH_QUICK_REFERENCE.md**
**Quick lookup guide** (~2,000 words)
- State fields reference table
- Graph topology ASCII diagram
- Intent routing table
- Abbreviation search examples (simple vs context)
- Key functions reference
- SSE event types
- ContextVar workaround explanation
- Configuration settings
- Debugging checklist

👉 **Best for:** Quick lookups while coding

---

### 3. **LANGGRAPH_CODE_EXAMPLES.md**
**Practical code examples** (~3,000 words)
- Using the agent (basic & with history)
- State management patterns
- Implementing custom nodes
- Routing logic examples
- State transformation patterns
- Creating new tools
- Abbreviation search deep dive
- SSE event handling (frontend & backend)
- Context variables usage
- Error handling patterns
- Logging best practices
- Testing nodes

👉 **Best for:** Copy-paste ready code

---

## 🗂️ Source Files Overview

```
backend/app/services/agent/
├── __init__.py (29 lines)
│   └─ Exports: build_agent_graph, AgentState
│
├── state.py (106 lines)
│   └─ AgentState TypedDict: 65 fields with reducers
│      • Control flow: intent, rewritten_query, iterations, expanded_query
│      • Accumulators: sources, images, kg_summaries, abbreviation_results
│      • Context: messages, user_memory_context, workspace_ids
│      • Output: final_answer, citation_map
│
├── nodes.py (1,103 lines) ⭐ MAIN FILE
│   └─ 6 async node functions:
│      1. memory_recall        – Load Graphiti temporal knowledge
│      2. intent_classifier    – Qwen3-4B: classify & rewrite query
│      3. agent_rag_executor   – Invoke RAG subgraph (6 intent types)
│      4. answer_generator     – Main LLM: generate answer with sources
│      5. direct_answer        – LLM for greetings/personal (no retrieval)
│      6. write_executor       – Invoke write subgraph (3 write intents)
│
│   Plus helpers:
│      • _CLASSIFIER_SYSTEM    – Qwen3-4B system prompt (11 intent rules)
│      • _parse_classifier_output() – JSON parse with fallback
│      • _get_msg_role()       – Extract role from messages
│      • _extract_last_user_message()
│      • _transform_rag_input/output() – State transformation
│      • _transform_input/output()     – Write executor transformation
│      • _get_rag_subgraph()   – Lazy-loaded singleton
│      • _get_write_subgraph() – Lazy-loaded singleton
│
├── graph.py (177 lines)
│   └─ Graph builder & routing:
│      • build_agent_graph()   – StateGraph compiler
│      • _route_by_intent()    – Routing after intent_classifier
│      • _should_continue_after_rag() – Routing after agent_rag_executor
│                                        (handles abbreviation expansion retry)
│      • get_agent_graph()     – Cached singleton getter
│      • reset_agent_graph()   – Force rebuild
│
├── tools.py (437 lines)
│   └─ 6 tool functions:
│      1. search_documents()        – Hybrid search (vector+KG+BM25)
│      2. list_documents()          – List indexed documents
│      3. summarize_document()      – LLM-based summarization
│      4. query_knowledge_graph()   – LightRAG entity lookup
│      5. search_documents_number() – Search by document number
│      6. search_abbreviation()     – ⭐ KEY: Abbreviation lookup
│
│   Tool registry with descriptions
│
└── streaming.py (254 lines)
    └─ SSE streaming + ContextVar workaround:
       • _event_queue_ctx  – ContextVar for asyncio.Queue
       • _db_ctx          – ContextVar for DB session
       • get_current_db() – Nodes call this instead of state.get("_db")
       • push_event()     – Nodes push events to SSE queue
       • stream_agent_to_sse() – Main streaming adapter
       • build_initial_state() – State builder from request
       • _sse()           – Format SSE event string
       • SSE_HEARTBEAT_INTERVAL = 15s
```

---

## 🔄 Graph Flow

```
START
  ↓
[memory_recall]
  └─ Load Graphiti memories
  ↓
[intent_classifier]
  └─ Qwen3-4B: classify intent + rewrite query
  ↓
[CONDITIONAL ROUTING] _route_by_intent()
  ├─ greeting/personal → [direct_answer]
  │                        └─ LLM without retrieval
  │                        └─ END
  │
  ├─ write_* → [write_executor]
  │              ├─ Invoke agent_write subgraph
  │              └─ END
  │
  └─ search/list/summarize/kg/abbr/doc_num → [agent_rag_executor]
     ├─ Transform state
     ├─ Invoke agent_rag subgraph
     ├─ Transform output
     │  └─ Check: abbreviation expansion triggered?
     │     └─ Set expanded_query if yes
     │
     └─ [CONDITIONAL ROUTING] _should_continue_after_rag()
        ├─ expanded_query set AND iterations < max?
        │  └─ YES → Loop back to [agent_rag_executor]
        │
        └─ NO → [answer_generator]
                ├─ Assemble context from:
                │  ├─ sources (document chunks)
                │  ├─ kg_summaries (tool results)
                │  ├─ abbreviation_results (if search_abbr)
                │  └─ user_memory_context
                │
                ├─ Stream tokens via provider.astream()
                ├─ Push each token to SSE queue
                └─ END
```

---

## 🎯 Key Features

### 1. **11 Intent Types**
```
reasoning:
  ✓ greeting       → direct_answer
  ✓ personal       → direct_answer + memory
  ✓ search         → agent_rag → search_documents
  ✓ list_docs      → agent_rag → list_documents
  ✓ summarize      → agent_rag → summarize_document
  ✓ kg_query       → agent_rag → query_knowledge_graph
  ✓ search_doc_num → agent_rag → search_documents_number
  ✓ search_abbr    → agent_rag → search_abbreviation
  ✓ write_summarize → write_executor
  ✓ write_suggest_edits → write_executor
  ✓ write_grammar_check → write_executor
```

### 2. **Smart Abbreviation Expansion**
- Detects if user asked a question about an abbreviation
- Looks up full form from DB
- If query length > abbrev length + 10 chars:
  - Replaces abbreviation with full form
  - Triggers RAG subgraph retry for semantic search
  - Combines results: abbreviation context + retrieved sources

### 3. **Graphiti Memory Integration**
- Loads user conversation memories (top_k=5)
- Hybrid search: semantic + BM25 + graph traversal
- Injected into system prompt for all LLM nodes
- Non-blocking: gracefully continues if Graphiti unavailable

### 4. **Real-Time SSE Streaming**
- Tokens streamed one-by-one as they're generated
- Status events for UX feedback
- Source & image events with metadata
- Complete event with final answer + sources + images

### 5. **ContextVar Workaround** ⭐
- LangGraph strips non-TypedDict keys before node execution
- Solution: Use Python contextvars for queue & DB
- `asyncio.create_task()` inherits context from parent
- Nodes call `get_current_db()` instead of `state.get("_db")`

---

## 🔍 Focus Areas

### search_abbr (Abbreviation Search)
**This is the most complex feature!**

1. **Classifier Rule:**
   - Extract abbreviation ONLY (remove "là gì?" suffix)
   - Preserve exact form (BMNN ≠ BMM)

2. **Three Lookup Scenarios:**
   - Not found: Ask for clarification
   - Single match: Return definition
   - Multiple matches: List all with descriptions

3. **Query Expansion:**
   - Check if original query significantly longer than abbreviation
   - If yes (len diff > 10): Replace abbreviation with full form
   - Triggers RAG subgraph retry for semantic search

4. **Example Path:**
   ```
   User: "BMNN cũng quản lý những gì?"
   → intent_classifier: "search_abbr"
   → search_abbreviation: "BMNN" → "Bộ Môi Trường và Tài Nguyên"
   → Query expansion: 37 chars > 4+10? YES
   → Set expanded_query
   → Loop back to agent_rag_executor
   → search_documents with expanded form
   → answer_generator with sources + abbreviation context
   ```

---

## 🛠️ Development Patterns

### Adding a New Tool
```python
async def my_tool(query: str, workspace_ids: list[int], db: AsyncSession) -> dict:
    # Implementation
    return {"results": [], "text": ""}

# Register
TOOL_REGISTRY["my_tool"] = "Description"

# Use in node (inside agent_rag subgraph)
```

### Adding a New Intent
```python
# 1. Update _CLASSIFIER_SYSTEM in nodes.py (add rule + example)
# 2. Add to VALID_INTENTS in state.py
# 3. Handle in _route_by_intent() in graph.py
# 4. Implement logic in appropriate node or create new subgraph
```

### Implementing a Custom Node
```python
async def my_node(state: AgentState) -> dict:
    db = get_current_db()  # Context access
    await push_event(state, "status", {...})  # SSE event
    
    # Do work...
    
    return {  # Partial state update
        "intent": "...",
        "sources": [...],  # Will extend (accumulator)
    }

# Add to graph:
graph.add_node("my_node", my_node)
graph.add_edge("previous_node", "my_node")
```

---

## 🐛 Debugging Checklist

- [ ] Memory not injected? Check `get_current_db()` works
- [ ] SSE events missing? Verify `push_event()` called
- [ ] Abbreviation not expanding? Check len diff > 10 logic
- [ ] Infinite loop? Check `iterations < max_iter`
- [ ] State missing field? Check reducer definition in state.py
- [ ] Classifier failing? Check Qwen3-4B model available
- [ ] Streaming stuck? Check `await asyncio.sleep(0)` in push_event

---

## 📊 Configuration

From `app.core.config`:
```python
NEXUSRAG_LG_MAX_ITERATIONS = 3      # Abbreviation retry limit
NEXUSRAG_LG_DEBUG = False            # Debug logging
HRAG_RERANKER_TOP_K = 5              # Retrieved chunks count
LLM_MAX_OUTPUT_TOKENS = 2048         # Main LLM output limit
```

---

## 🔗 Related Components

**Not in this directory but used by agent:**
- `backend/app/services/agents/agent_rag/` – RAG subgraph (6 intent types)
- `backend/app/services/agents/agent_write/` – Write subgraph (3 intent types)
- `backend/app/services/graphiti_client.py` – Memory loading
- `backend/app/services/llm/` – LLM providers (Qwen3-4B, main LLM)
- `backend/app/services/knowledge_graph_service.py` – LightRAG
- `backend/app/models/abbreviation.py` – Abbreviation model

---

## 📝 Notes

- All nodes are **async** (`async def`)
- All tools are **async** (`async def`)
- State uses **reducers** for accumulation (sources, images, kg_summaries)
- Graph is **compiled** once and **cached** (singleton)
- Streaming is **real-time** with `asyncio.Queue`
- Error handling is **graceful** (continue with empty results)
- Logging is **detailed** with node context (`[node_name]`)

---

## 🚀 Quick Start

```python
# 1. Import
from app.services.agent import build_agent_graph
from app.services.agent.streaming import build_initial_state, stream_agent_to_sse

# 2. Build graph
graph = build_agent_graph()

# 3. Create state
state = build_initial_state(
    workspace_ids=[1],
    message="BMNN là gì?",
    history=[],
    system_prompt="You are helpful.",
    enable_thinking=False,
    db=db_session,
)

# 4. Stream
async for sse_event in stream_agent_to_sse(graph, state):
    yield sse_event
```

---

**Last Updated:** 2026-03-26  
**Source Files:** 6 Python modules (~2,000 lines total)  
**Documentation:** 3 comprehensive guides

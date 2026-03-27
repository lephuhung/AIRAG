# 🚀 AIRAG LangGraph Agent Documentation - START HERE

Welcome! This directory contains **comprehensive documentation** of the AIRAG LangGraph agent implementation.

## 📚 What's Included

### Documentation Files (Saved in project root)

1. **LANGGRAPH_README.md** (12 KB) ← **START WITH THIS**
   - Overview of the entire system
   - File structure and component breakdown
   - Graph flow diagram
   - Key features and focus areas
   - Recommended reading order
   - 👉 **Best for:** Getting oriented (5 min read)

2. **LANGGRAPH_ANALYSIS.md** (26 KB) ⭐ **MOST COMPREHENSIVE**
   - Complete state definition (65 fields)
   - 6 node implementations with full code details
   - 6 tool implementations
   - Graph topology and routing logic
   - SSE streaming architecture
   - Abbreviation search flow with examples
   - Design patterns and error handling
   - 👉 **Best for:** Deep understanding (30 min read)

3. **LANGGRAPH_QUICK_REFERENCE.md** (13 KB)
   - State fields reference table
   - Graph topology ASCII diagram
   - Intent routing table
   - Abbreviation search examples
   - Key functions reference
   - SSE event types
   - Configuration settings
   - Debugging checklist
   - 👉 **Best for:** Quick lookup while coding (5 min)

4. **LANGGRAPH_CODE_EXAMPLES.md** (19 KB)
   - Using the agent (basic & with history)
   - State management patterns
   - Custom node implementation
   - Routing logic examples
   - State transformation patterns
   - Creating new tools
   - Abbreviation search deep dive
   - SSE event handling
   - Error handling patterns
   - Testing nodes
   - 👉 **Best for:** Copy-paste ready code patterns (20 min)

5. **LANGGRAPH_FILES_READ.txt** (8 KB)
   - Complete file read inventory
   - Statistics and key findings
   - Search_abbr function location
   - RAG & write nodes interaction details
   - 👉 **Best for:** Reference (2 min)

---

## 📖 Reading Recommendations

### For Complete Understanding (1 hour)
1. This file (2 min)
2. LANGGRAPH_README.md (5 min)
3. LANGGRAPH_ANALYSIS.md (30 min)
4. LANGGRAPH_CODE_EXAMPLES.md (20 min)

### For Quick Reference (15 min)
1. LANGGRAPH_README.md (5 min)
2. LANGGRAPH_QUICK_REFERENCE.md (10 min)

### For Implementation (30 min)
1. LANGGRAPH_CODE_EXAMPLES.md (20 min)
2. Source code: `backend/app/services/agent/nodes.py`
3. Source code: `backend/app/services/agent/graph.py`

---

## 🎯 What You'll Learn

### Architecture
- 11 intent types with intelligent routing
- 6 nodes (memory_recall, intent_classifier, agent_rag_executor, answer_generator, direct_answer, write_executor)
- 6 tools (search, list, summarize, knowledge graph, document number search, **abbreviation search**)
- Real-time SSE streaming

### Key Innovation: search_abbr (Abbreviation Search)
- Extracts abbreviations from natural language queries
- Looks up full forms from database
- Smart expansion: If query has context (len > abbrev + 10 chars):
  - Replaces abbreviation with full form
  - Triggers semantic search with expanded query
  - Combines results for comprehensive answer
- Example: "BMNN cũng quản lý những gì?" → expands & searches

### Critical Architecture: ContextVar Workaround
- Problem: LangGraph strips non-TypedDict keys before node execution
- Solution: Python `contextvars` module passes queue & DB through context
- Result: Nodes can access database and push events to SSE stream

### Design Patterns
- Multi-intent routing with fallbacks
- Subgraph delegation with state transformation
- Lazy-loaded singletons for subgraphs
- Graceful error handling (no single failure breaks pipeline)
- Real-time streaming with token-by-token push

---

## 📊 By The Numbers

| Metric | Count |
|--------|-------|
| Source Files | 6 |
| Source Lines of Code | 2,078 |
| Documentation Lines | 2,213 (equal!) |
| Node Types | 6 |
| Tool Functions | 6 |
| Intent Types | 11 |
| State Fields | 65 |
| SSE Event Types | 7 |

---

## 🗂️ Source Code Location

All source code in: `backend/app/services/agent/`

```
agent/
├── __init__.py           # Module exports
├── state.py             # AgentState TypedDict (65 fields)
├── nodes.py             # 6 nodes + helpers (1,103 lines) ⭐
├── graph.py             # Graph builder + routing
├── tools.py             # 6 tool functions
└── streaming.py         # SSE + ContextVar workaround
```

**Main file to understand:** `nodes.py` (1,103 lines)

---

## 🚀 Quick Start Example

```python
from app.services.agent import build_agent_graph
from app.services.agent.streaming import build_initial_state, stream_agent_to_sse

# Build graph
graph = build_agent_graph()

# Create state
state = build_initial_state(
    workspace_ids=[1],
    message="BMNN là gì?",  # User question
    history=[],
    system_prompt="You are helpful.",
    enable_thinking=False,
    db=db_session,
    user_id=123,
)

# Stream events
async for sse_event in stream_agent_to_sse(graph, state):
    yield sse_event  # Real-time SSE to frontend
```

---

## 🔍 Key Concepts at a Glance

### Graph Flow
```
START
  → memory_recall (Graphiti search)
  → intent_classifier (Qwen3-4B)
  → [route by intent]
     ├ greeting/personal → direct_answer → END
     ├ write_* → write_executor → END
     └ search/* → agent_rag_executor
        → [check abbreviation expansion]
           ├ YES & iterations < 3 → loop back
           └ NO → answer_generator → END
```

### State Fields (65 total)
- **Control flow:** intent, rewritten_query, iterations, expanded_query, tool_called
- **Retrieval:** sources, images, kg_summaries, abbreviation_results
- **Context:** messages, user_memory_context, workspace_ids, user_id
- **Output:** final_answer, citation_map
- **Other:** system_prompt, enable_thinking, write_action, text_input, etc.

### Intent Types (11 total)
- **No retrieval:** greeting, personal
- **RAG retrieval:** search, list_docs, summarize, kg_query, search_doc_num, search_abbr
- **Text processing:** write_summarize, write_suggest_edits, write_grammar_check

### Tools (6 total)
- search_documents - Hybrid vector+KG+BM25 search
- list_documents - List indexed documents
- summarize_document - LLM-based summarization
- query_knowledge_graph - LightRAG entity lookup
- search_documents_number - Search by official document number
- **search_abbreviation** - ⭐ Abbreviation lookup with expansion

---

## 🛠️ Development Patterns

### Adding a New Tool
1. Write async function in `tools.py`
2. Register in `TOOL_REGISTRY`
3. Call from appropriate node or subgraph

### Adding a New Intent
1. Update `_CLASSIFIER_SYSTEM` in `nodes.py` (add rule + example)
2. Add to `VALID_INTENTS` in `state.py`
3. Handle in routing functions (`_route_by_intent()`)

### Adding a Custom Node
1. Write async function that receives state
2. Use `get_current_db()` for database access
3. Use `push_event()` for SSE events
4. Return partial state update dict
5. Register in `build_agent_graph()`

---

## 🐛 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| SSE events not received | Check `push_event()` called, verify ContextVar set |
| Abbreviation not expanding | Check query_len > abbrev_len + 10 logic |
| Infinite loop | Verify iterations counter & max_iter check |
| State field missing | Check reducer defined in `state.py` |
| DB access fails | Use `get_current_db()` instead of `state.get("_db")` |
| Classifier failing | Check Qwen3-4B model available |
| Memory not injected | Check Graphiti connection |

---

## 📚 Additional Context

### Related Components (outside this directory)
- `backend/app/services/agents/agent_rag/` - RAG subgraph
- `backend/app/services/agents/agent_write/` - Write subgraph
- `backend/app/services/graphiti_client.py` - Memory loading
- `backend/app/services/llm/` - LLM providers
- `backend/app/models/abbreviation.py` - Abbreviation model

### Configuration
From `app.core.config`:
- `NEXUSRAG_LG_MAX_ITERATIONS` = 3
- `NEXUSRAG_LG_DEBUG` = False
- `HRAG_RERANKER_TOP_K` = 5
- `LLM_MAX_OUTPUT_TOKENS` = 2048

---

## ✅ Checklist for Understanding

- [ ] Read LANGGRAPH_README.md
- [ ] Understand the 11 intent types
- [ ] Know the 6 nodes in the graph
- [ ] Understand abbreviation expansion logic
- [ ] Know how ContextVar workaround works
- [ ] Understand state transformation for subgraphs
- [ ] Review error handling patterns
- [ ] Check code examples for your use case

---

## 📞 Quick Reference

**Main node to understand:** `intent_classifier` → controls entire flow  
**Most complex feature:** `search_abbr` → abbreviation expansion logic  
**Critical architecture:** ContextVar workaround in `streaming.py`  
**Entry point:** `build_agent_graph()` in `graph.py`  
**Initial state:** `build_initial_state()` in `streaming.py`

---

## 🎓 Learning Path

1. **Beginner:** Read LANGGRAPH_README.md
2. **Intermediate:** Study LANGGRAPH_ANALYSIS.md sections 1-3
3. **Advanced:** Deep dive into nodes.py source code
4. **Expert:** Implement custom tools/intents using CODE_EXAMPLES.md

---

**Last Updated:** 2026-03-26  
**Total Documentation:** 5 files, 78 KB, 2,213 lines  
**Source Code:** 6 files, 2,078 lines

Start with **LANGGRAPH_README.md** →

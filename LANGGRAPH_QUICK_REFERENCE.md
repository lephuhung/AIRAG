# AIRAG LangGraph Agent - Quick Reference Guide

## File Locations
```
backend/app/services/agent/
├── __init__.py           # Exports: build_agent_graph, AgentState
├── state.py             # AgentState TypedDict (65 fields)
├── nodes.py             # 6 nodes: memory_recall, intent_classifier, agent_rag_executor, 
│                        #          answer_generator, direct_answer, write_executor
├── graph.py             # Graph builder + routing logic
├── tools.py             # 6 tools: search_documents, list_documents, summarize_document,
│                        #          query_knowledge_graph, search_documents_number, search_abbreviation
└── streaming.py         # SSE streaming + ContextVar workaround
```

## Agent State Fields (Key Only)

### Control Flow
| Field | Type | Purpose |
|-------|------|---------|
| `intent` | str | Classification: greeting, search, list_docs, summarize, kg_query, search_abbr, write_* |
| `rewritten_query` | str | Query after Qwen3-4B rewrite |
| `iterations` | int | Loop counter (max 3) |
| `expanded_query` | str | Query after abbreviation expansion (triggers retry) |
| `tool_called` | bool | RAG execution flag |

### Retrieval Results (Accumulators)
| Field | Reducer | Content |
|-------|---------|---------|
| `sources` | extend | Retrieved document chunks {index, content, source_file, page_no, heading_path} |
| `images` | extend | Image references |
| `image_parts` | extend | Raw image bytes |
| `kg_summaries` | extend | KG query results strings |
| `abbreviation_results` | plain | [{short_form, full_form, description}] |

### User Context
| Field | Type | Purpose |
|-------|------|---------|
| `messages` | list + add_messages | Conversation history (HumanMessage, AIMessage) |
| `user_memory_context` | str | Graphiti memory (5 top results) |
| `workspace_ids` | list[int] | Knowledge bases |
| `user_id` | int | User identifier |

### Output
| Field | Type | Purpose |
|-------|------|---------|
| `final_answer` | str | LLM-generated response |
| `citation_map` | dict | Citation ID registry |

### Write Operations
| Field | Type | Purpose |
|-------|------|---------|
| `write_action` | str | summarize, suggest_edits, grammar_check |
| `text_input` | str | Raw text to process |

---

## Graph Topology (ASCII)

```
                    ┌─ memory_recall ──────┐
                    │  (Graphiti search)   │
                    └──────────┬───────────┘
                               ↓
                    ┌─ intent_classifier ──┐
                    │  (Qwen3-4B)          │
                    └──────────┬───────────┘
                               ↓
                    ┌─ Route by Intent ────┐
        ┌───────────┤ _route_by_intent()   ├───────────┐
        ↓           └──────────────────────┘           ↓
   direct_answer                              agent_rag_executor
   (greeting/                                 (search/list/
    personal)                                  summarize/kg/
        │                                      abbr/doc_num)
        │                                           │
        └───────────────────┬───────────────────────┘
                            ↓
          ┌─ Continue Check? ──────────────┐
          │ _should_continue_after_rag()   │
          │ • expanded_query set? → retry  │
          │ • max iterations? → continue   │
          └───────────┬────────────────────┘
                      ↓
              answer_generator
              (Main LLM final answer)
                      │
                      ↓
                    END

(Parallel path: write_executor for write_* intents)
```

---

## Intent Routing Table

| Intent | Route | Action | Subgraph |
|--------|-------|--------|----------|
| greeting | direct_answer | No retrieval, conversational | None |
| personal | direct_answer | No retrieval, use memory | None |
| search | agent_rag_executor | Hybrid search (vector+KG+BM25) | agent_rag |
| list_docs | agent_rag_executor | List indexed documents | agent_rag |
| summarize | agent_rag_executor | Summarize specific document | agent_rag |
| kg_query | agent_rag_executor | Query knowledge graph | agent_rag |
| search_doc_num | agent_rag_executor | Search by document number | agent_rag |
| search_abbr | agent_rag_executor | Lookup abbreviation meanings | agent_rag |
| write_summarize | write_executor | Summarize user text | agent_write |
| write_suggest_edits | write_executor | Suggest improvements | agent_write |
| write_grammar_check | write_executor | Grammar/style check | agent_write |

---

## Abbreviation Search Flow (search_abbr)

### Example 1: Simple Query
```
User: "BMNN là gì?"
  ↓
intent_classifier → {intent: "search_abbr", rewritten_query: "BMNN"}
  ↓
agent_rag_executor
  ├─ search_abbreviation("BMNN", workspace_ids)
  │  ├─ Query: Abbreviation.short_form ILIKE "%BMNN%"
  │  └─ Result: {found: True, full_form: "Bộ Môi Trường và Tài Nguyên"}
  │
  ├─ Query expansion check:
  │  ├─ abbrev_len = 4, query_len = 4, diff = 0
  │  ├─ has_context = 0 > 4+10? NO
  │  └─ NO expanded_query
  │
  └─ answer_generator
     ├─ Context: "## Abbreviation Results\n- **BMNN** = Bộ Môi Trường..."
     └─ LLM generates answer
```

### Example 2: Query with Context
```
User: "BMNN cũng quản lý những gì?"
  ↓
intent_classifier → {rewritten_query: "BMNN cũng quản lý những gì?"}
  ↓
agent_rag_executor (iteration 1)
  ├─ search_abbreviation("BMNN cũng quản lý những gì?")
  │  └─ Result: {found: True, full_form: "Bộ Môi Trường và Tài Nguyên"}
  │
  ├─ Query expansion check:
  │  ├─ abbrev_len = 4, query_len = 37
  │  ├─ has_context = 37 > 4+10? YES ✓
  │  ├─ expanded_query = "Bộ Môi Trường và Tài Nguyên cũng quản lý những gì?"
  │  └─ SET expanded_query in state
  │
  └─ _should_continue_after_rag
     ├─ expanded_query set? YES
     ├─ iterations < 3? YES (0 < 3)
     └─ LOOP BACK to agent_rag_executor

agent_rag_executor (iteration 2) [RETRY PATH]
  ├─ Use expanded_query for search
  ├─ search_documents("Bộ Môi Trường và Tài Nguyên cũng quản lý những gì?")
  │  └─ Hybrid search returns relevant document chunks
  │
  └─ _should_continue_after_rag
     ├─ expanded_query already used, continue
     ├─ iterations < 3? YES (1 < 3)
     ├─ But no more expansions
     └─ GO TO answer_generator

answer_generator
  ├─ Context:
  │  ├─ "## Abbreviation Results\n- **BMNN** = Bộ Môi Trường..."
  │  └─ "## Document Chunks\nSource [1]: ..."
  └─ LLM generates answer with citations
```

---

## Key Functions Reference

### Nodes

| Node | Input | Output | Key Logic |
|------|-------|--------|-----------|
| **memory_recall** | user_id, user_message | user_memory_context | Search Graphiti (top_k=5) |
| **intent_classifier** | messages | intent, rewritten_query, write_action | Qwen3-4B + JSON parse |
| **agent_rag_executor** | intent, rewritten_query | sources, images, abbreviation_results | Subgraph invocation + transform |
| **answer_generator** | sources, kg_summaries, messages | final_answer | Main LLM + streaming |
| **direct_answer** | messages, user_memory_context | final_answer | LLM without retrieval |
| **write_executor** | text_input, write_action | final_answer | Write subgraph invocation |

### Tools

| Tool | Input | Output | Query Method |
|------|-------|--------|--------------|
| **search_documents** | query, top_k, workspace_ids | sources, images | Hybrid: vector+KG+BM25 |
| **list_documents** | workspace_ids | Formatted document list | DB query: INDEXED status |
| **summarize_document** | document_id | summary_text | LLM: markdown content |
| **query_knowledge_graph** | entity | kg_result_text | LightRAG naive mode |
| **search_documents_number** | query | documents list | DB: document_number ILIKE |
| **search_abbreviation** | abbreviation | {found, full_form, description} | DB: short_form ILIKE |

---

## SSE Streaming Events

```
→ event: status         {"step": str, "detail": str}
→ event: thinking       {"text": str}  (if think=true)
→ event: sources        {"sources": [ChatSourceChunk]}
→ event: images         {"image_refs": [ChatImageRef]}
→ event: token          {"text": str}  (repeated N times)
→ event: complete       {"answer": str, "sources": [], "images": []}
→ event: error          {"message": str}
```

---

## ContextVar Workaround (Critical!)

**Problem:** LangGraph strips non-declared TypedDict keys before passing to nodes

**Solution:** Use Python contextvars
```python
_event_queue_ctx = ContextVar("_event_queue", default=None)
_db_ctx = ContextVar("_db", default=None)

# In nodes:
db = get_current_db()  # Read from context, not state
await push_event(state, "token", "...")  # Queue from context
```

**Why it works:**
- `asyncio.create_task()` copies current context into task
- Context persists across node boundaries
- Bypasses LangGraph TypedDict validation

---

## Configuration Settings

```python
from app.core.config import settings

NEXUSRAG_LG_MAX_ITERATIONS = 3      # Max retries for RAG
NEXUSRAG_LG_DEBUG = False            # Debug logging
HRAG_RERANKER_TOP_K = 5              # Retrieved chunks
LLM_MAX_OUTPUT_TOKENS = 2048         # Main LLM limit
```

---

## Error Handling Strategy

| Component | Failure | Recovery |
|-----------|---------|----------|
| memory_recall | Graphiti down | Continue with empty context |
| intent_classifier | LLM fails | Default to intent="search" |
| search_* tools | DB error | Return empty results |
| answer_generator | Streaming fails | Fallback to non-streaming |
| Subgraphs | Exception | Return empty output dict |

---

## Logging Patterns

```python
logger.info(f"[node_name] Description: {var!r}")
logger.warning(f"[node_name] Recoverable: {detail}")
logger.error(f"[node_name] Fatal: {e}", exc_info=True)
```

Examples:
```python
logger.info(f"[intent_classifier] RAW_LLM_RESPONSE: {response!r}")
logger.info(f"[tool_executor] SEARCH_ABBR: querying abbreviation for query={query!r}")
logger.warning(f"[router] Max iterations ({max_iter}) reached — forcing answer_generator")
```

---

## Module Entry Points

```python
# Import graph
from app.services.agent import build_agent_graph, AgentState

# Build graph
graph = build_agent_graph()

# Build initial state
from app.services.agent.streaming import build_initial_state
state = build_initial_state(
    workspace_ids=[1, 2],
    message="BMNN là gì?",
    history=[{"role": "user", "content": "..."}, ...],
    system_prompt="...",
    enable_thinking=False,
    db=async_session,
    user_id=123,
)

# Stream
from app.services.agent.streaming import stream_agent_to_sse
async for sse_event in stream_agent_to_sse(graph, state):
    yield sse_event  # SSE string
```

---

## Abbreviation Search Classifier Rules

```python
_CLASSIFIER_SYSTEM = """
...
- For "search_abbr" (abbreviation queries): extract ONLY the abbreviation itself
  - remove "là gì?", "có nghĩa là gì?", "là viết tắt của gì?", etc.
  - EXACT abbreviation must be preserved (e.g., "BMNN" stays "BMNN", NOT "BMM")
...
Output: {"intent": "search_abbr", "rewritten_query": "BMNN", "needs_tool": true}
"""
```

---

## Quick Debugging Checklist

- [ ] **Memory injection not working?** Check `get_current_db()` in nodes
- [ ] **SSE events missing?** Verify `push_event()` calls in nodes
- [ ] **Abbreviation not expanding?** Check `query_len > abbrev_len + 10` logic
- [ ] **Intent misclassified?** Review Qwen3-4B prompt rules
- [ ] **Citation duplicates?** Verify `existing_citation_ids` tracking
- [ ] **Streaming stuck?** Check `asyncio.sleep(0)` in `push_event()`
- [ ] **Graph loops forever?** Verify `max_iter` and `iterations` counter
- [ ] **Write operations failing?** Check `text_input` extraction fallback


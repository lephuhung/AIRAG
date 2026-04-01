# LangGraph Agent Architecture

Document version: 2026-04-01
Branch: `KG-building`

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MAIN AGENT GRAPH                                  │
│                                                                         │
│  START                                                                  │
│    │                                                                    │
│    ▼                                                                    │
│  ┌──────────────┐    ┌─────────────────┐    ┌─────────────────────┐    │
│  │ abbr_expander │───▶│  memory_recall  │───▶│ intent_classifier  │    │
│  └──────────────┘    └─────────────────┘    └─────────┬───────────┘    │
│                                                       │                 │
│                              ┌────────────────────────┼────────────┐    │
│                              ▼                        ▼            ▼    │
│                    ┌───────────────┐    ┌──────────────────┐  ┌────────┐│
│                    │ direct_answer  │    │agent_rag_executor│  │  write ││
│                    │  (greeting /   │    │    (subgraph)    │  │execut- ││
│                    │   personal)    │    └────────┬─────────┘  │  or    ││
│                    └───────┬────────┘              │              └────┬───┘│
│                            │                       │                    │     │
│                            ▼                       ▼                    │     │
│                           END     ◀── ─── ─── ─── ─── ┌─────────────────┘     │
│                                               │       │                       │
│                           ┌───────────────────┴───┐  │                       │
│                           │ _should_continue_after │  │                       │
│                           │ _rag()                 │  │                       │
│                           └───────────┬───────────┘  │                       │
│                                       │              │                       │
│         ┌──────────────────────────────┼──────────────┘                       │
│         ▼                              ▼                                      │
│  ┌─────────────┐           ┌─────────────────┐                              │
│  │write_executor│           │answer_generator │                              │
│  │  (subgraph)  │           │                 │                              │
│  └──────┬──────┘           └────────┬────────┘                              │
│         │                            │                                        │
│         ▼                            ▼                                        │
│        END                          END                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Subgraphs:
  agent_rag_executor  →  agent_rag  (search / list_docs / summarize / kg_query / search_doc_num / search_abbr / mongo_search_*)
  write_executor      →  agent_write (summarize / suggest_edits / grammar_check)
```

---

## 1. Main Graph — `build_agent_graph()`

**File:** `backend/app/services/agent/graph.py`

### Node List

| Node | Function | Purpose |
|---|---|---|
| `abbr_expander` | `nodes.abbr_expander` | Detect & expand abbreviations (UPPERCASE 2-6 chars) before routing |
| `memory_recall` | `nodes.memory_recall` | Load user memories from Graphiti (Neo4j) |
| `intent_classifier` | `nodes.intent_classifier` | Qwen3-4B: classify intent + rewrite query |
| `agent_rag_executor` | `nodes.agent_rag_executor` | Invoke `agent_rag` subgraph |
| `direct_answer` | `nodes.direct_answer` | Answer greetings directly (no retrieval) |
| `write_executor` | `nodes.write_executor` | Invoke `agent_write` subgraph |
| `answer_generator` | `nodes.answer_generator` | Main LLM: generate answer with sources |

### Edge Flow

```
START
  │
  ▼
abbr_expander ──────────────────────────────────────────── (runs first, expands abbreviations)
  │
  ▼
memory_recall ───────────────────────────────────────────── (load user memories from Graphiti)
  │
  ▼
intent_classifier ──────────────────────────────────────── (classify: search / list_docs / summarize / kg_query / search_abbr / greeting / personal / write_* / mongo_search_*)
  │
  ├─────────────────────┬──────────────────────────────────┬──────────────────────────────────┐
  ▼                     ▼                                  ▼                                  ▼
direct_answer    agent_rag_executor                  write_executor                       END
(greeting/               │                                  │
personal)               │                                  │
  │                      │                                  │
  ▼                      ▼                                  ▼
 END                _should_continue_after_rag              END
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   agent_rag_executor  write_executor  answer_generator
      (retry loop)    (summarize intent)    │
          │                 │               │
          │                 └───────────────┤
          │                                 │
          └──────────────▶ END ◀────────────┘
```

### Routing Logic

**After `intent_classifier`** — `_route_by_intent()`:
```python
intent = state["intent"]

if intent in ("greeting", "personal"):
    → direct_answer
elif intent in ("write_summarize", "write_suggest_edits", "write_grammar_check"):
    → write_executor
else:
    → agent_rag_executor   # search, list_docs, summarize, kg_query, search_doc_num, search_abbr, mongo_search_*
```

**After `agent_rag_executor`** — `_should_continue_after_rag()`:
```python
if iterations >= MAX_ITER:      # default 3
    → answer_generator

elif intent == "summarize":
    → write_executor             # RAG fetched raw doc → write_executor summarizes

elif expanded_query is set:       # abbreviation + extra context detected
    → agent_rag_executor         # retry with expanded query

else:
    → answer_generator
```

---

## 2. Subgraph: `agent_rag` — `create_agent_rag()`

**File:** `backend/app/services/agents/agent_rag.py`

Invoked by the `agent_rag_executor` node in the main graph.

```
START
  │
  ▼
route_by_intent(intent)
  │
  ├── "search"            ──▶ search_documents_node ──────────────────────────────┐
  ├── "list_docs"         ──▶ list_documents_node ────────────────────────────────┤
  ├── "summarize"         ──▶ summarize_document_node ────────────────────────────┤
  ├── "kg_query"          ──▶ kg_query_node ──────────────────────────────────────┤
  ├── "search_doc_num"    ──▶ search_doc_num_node ───────────────────────────────┤
  ├── "search_abbr"       ──▶ search_abbr_node ───────────────────────────────────┤
  ├── "mongo_search_cccd" ──▶ mongo_search_people_node ──────────────────────────┤
  ├── "mongo_search_name" ──▶ mongo_search_people_node ──────────────────────────┤
  ├── "mongo_search_bhxh" ──▶ mongo_search_people_node ──────────────────────────┤
  └── "mongo_search_phone" ──▶ mongo_search_people_node ─────────────────────────┘
                                      │
                                      ▼
                                     END
```

### Node Details

| Node | Intent(s) | Tool Called | Output |
|---|---|---|---|
| `search_documents_node` | `search` | `HRAGService.query_deep()` hybrid search | `sources`, `images`, `kg_summaries` |
| `list_documents_node` | `list_docs` | DB query (INDEXED docs only) | `final_answer` (formatted doc list) |
| `summarize_document_node` | `summarize` | Read raw markdown from MinIO | `kg_summaries` (raw doc text, up to 16k chars) |
| `kg_query_node` | `kg_query` | `KnowledgeGraphService.query()` | `final_answer` (KG results) |
| `search_doc_num_node` | `search_doc_num` | DB fuzzy match on `document_number` + vector fallback | `final_answer` (doc content) |
| `search_abbr_node` | `search_abbr` | DB lookup in `abbreviation` table | `abbreviation_results` + `final_answer` |
| `mongo_search_people_node` | `mongo_search_cccd/name/bhxh/phone` | `mongo_people_service` lookup | `mongo_results` + `final_answer` (formatted person record) |

---

## 3. Subgraph: `agent_write` — `create_agent_write()`

**File:** `backend/app/services/agents/agent_write.py`

Invoked by the `write_executor` node in the main graph.

```
START
  │
  ▼
route_write_action(write_action)
  │
  ├── "summarize" / "extract_key_points" ──▶ summarize_text_node
  ├── "suggest_edits"                      ──▶ suggest_edits_node
  └── "grammar_check"                      ──▶ check_grammar_node
                                                  │
                                                  ▼
                                                answer
                                                  │
                                                  ▼
                                                 END
```

### Node Details

| Node | Action | LLM Call | Temperature |
|---|---|---|---|
| `summarize_text_node` | `summarize` / `extract_key_points` | Summarize in Vietnamese | 0.3 |
| `suggest_edits_node` | `suggest_edits` | Structured improvement suggestions | 0.3 |
| `check_grammar_node` | `grammar_check` | Grammar + style check | 0.2 |

---

## 4. Agent State — `AgentState`

**File:** `backend/app/services/agent/state.py`

```python
class AgentState(TypedDict):
    # Accumulator fields (use reducers)
    messages          : Annotated[list, add_messages]   # chat history
    sources           : Annotated[list, extend]      # ChatSourceChunk dicts
    images            : Annotated[list, extend]       # ChatImageRef dicts
    image_parts       : Annotated[list, extend]       # raw image bytes for vision LLM
    kg_summaries      : Annotated[list, extend]        # KG insight / tool result strings

    # Control fields
    workspace_ids      : list[int]
    document_ids       : Optional[list[int]]
    user_id            : Optional[int]
    session_id         : Optional[str]
    system_prompt      : str
    enable_thinking    : bool

    intent             : str          # "search" | "list_docs" | "summarize" | ...
    rewritten_query    : str           # query rewritten by Qwen3-4B
    iterations         : int          # loop counter
    tool_called        : bool         # True after first tool execution
    expanded_query     : str          # abbreviation expanded with full form

    user_memory_context: str          # memories from Graphiti
    final_answer        : str
    citation_map        : dict

    write_action        : str          # "summarize" | "suggest_edits" | ...
    text_input          : str          # raw text for write agent
    abbreviation_results : list[dict]   # [{"short_form": ..., "full_form": ...}]
    potential_abbreviations: list[str]  # candidates not found in DB
    mongo_results       : list[dict]   # person records from MongoDB
```

**`AgentRagState`** (subgraph state, `agents/agent_rag.py`):

```python
class AgentRagState(BaseModel):
    messages            : list
    intent              : str | None
    rewritten_query     : str
    workspace_ids        : list
    document_ids         : list | None
    sources              : list
    images               : list
    image_parts          : list
    kg_summaries         : list
    abbreviation_results : list
    mongo_results        : list         # person records from MongoDB
    final_answer         : str | None
```

---

## 5. Intent Taxonomy

```
VALID INTENTS:
├── greeting              → direct_answer (no retrieval)
├── personal              → direct_answer (answer from memory)
├── search                → agent_rag_executor → search_documents_node
├── list_docs             → agent_rag_executor → list_documents_node
├── summarize             → agent_rag_executor → summarize_document_node → write_executor
├── kg_query              → agent_rag_executor → kg_query_node
├── search_doc_num        → agent_rag_executor → search_doc_num_node
├── search_abbr           → agent_rag_executor → search_abbr_node
├── write_summarize       → write_executor → summarize_text_node
├── write_suggest_edits   → write_executor → suggest_edits_node
├── write_grammar_check   → write_executor → check_grammar_node
├── mongo_search_cccd     → agent_rag_executor → mongo_search_people_node (MongoDB)
├── mongo_search_name     → agent_rag_executor → mongo_search_people_node (MongoDB)
├── mongo_search_bhxh     → agent_rag_executor → mongo_search_people_node (MongoDB)
└── mongo_search_phone    → agent_rag_executor → mongo_search_people_node (MongoDB)
```

### MongoDB People Search Intents

| Intent | Lookup Method | MongoDB Collection |
|---|---|---|
| `mongo_search_cccd` | Exact match on `cccd` field | `people` |
| `mongo_search_name` | Case-insensitive regex partial match on `ho_ten` | `people` |
| `mongo_search_bhxh` | Exact/regex match on `so_bhxh` field | `people` |
| `mongo_search_phone` | Exact / ends-with / contains on `so_dien_thoai` | `people` |

---

## 6. Data Flow: Search Intent

```
User message
     │
     ▼
abbr_expander          ─── identifies UPPERCASE tokens ──→ potential_abbreviations
     │                                                            │
     ▼                                                            │
memory_recall          ─── Graphiti Neo4j ──────────────────────▶ user_memory_context
     │                                                            │
     ▼                                                            │
intent_classifier      ─── Qwen3-4B ────────────────────────────▶ intent = "search"
                       ─── rewrite ─────────────────────────────▶ rewritten_query
     │                                                            │
     ▼                                                            │
agent_rag_executor     ─── invoke agent_rag subgraph             │
     │                                                            │
     ▼                                                            │
search_documents_node  ─── HRAGService.query_deep()              │
     │                 • ChromaDB vector search (top 20)          │
     │                 • BM25 retrieval                           │
     │                 • Cross-encoder rerank (top 8)             │
     │                 • KG entity lookup                         │
     │                                                            │
     ▼                                                            │
sources[], images[], kg_summaries[]                              │
     │                                                            │
     ▼                                                            │
answer_generator       ─── streams tokens via provider.astream() │
     │                 • injects sources as context                │
     │                 • streams thinking (if enabled)            │
     │                 • emits token events to SSE queue         │
     │                                                            │
     ▼                                                            │
SSE: token, sources, status, complete ────▶ Frontend
```

---

## 7. External Integrations

```
Graphiti (Neo4j)
  memory_recall ──search_user_memory()──▶ Temporal memories + entity facts

Abbreviation DB (PostgreSQL)
  abbr_expander ──SELECT──▶ Active abbreviations
  search_abbr_node ──SELECT──▶ Full form lookup

Document Storage (MinIO + PostgreSQL)
  summarize_document_node ──download_markdown()──▶ Raw markdown content
  search_documents_node ──HRAGService──▶ Vector + KG retrieval

Knowledge Graph (LightRAG / Neo4j)
  kg_query_node ──KnowledgeGraphService.query()──▶ Entity relationships

MongoDB People Database (MongoDB)
  mongo_search_people_node ──mongo_people_service──▶ Person records
    • search_by_cccd()    ── exact match on cccd
    • search_by_name()    ── regex partial match on ho_ten
    • search_by_bhxh()    ── exact/regex match on so_bhxh
    • search_by_phone()   ── exact/ends-with/contains on so_dien_thoai

LLM Providers (Gemini / Ollama / OpenAI-Compatible)
  • intent_classifier: Qwen3-4B (memory agent)
  • answer_generator: Main LLM (configurable provider)
  • direct_answer: Main LLM
  • write_executor subgraph: Main LLM
```

---

## 8. Configuration

| Config Key | Default | Purpose |
|---|---|---|
| `NEXUSRAG_AGENT_BACKEND` | `legacy` | Set `langgraph` to enable this graph |
| `NEXUSRAG_LG_MAX_ITERATIONS` | `3` | Max loops in `_should_continue_after_rag` |
| `NEXUSRAG_RERANKER_TOP_K` | `8` | Final results after cross-encoder reranking |
| `NEXUSRAG_VECTOR_PREFETCH` | `20` | Over-fetch before reranking |

### MongoDB Configuration

| Config Key | Default | Purpose |
|---|---|---|
| `MONGO_HOST` | `localhost` | MongoDB server hostname |
| `MONGO_PORT` | `27017` | MongoDB server port |
| `MONGO_USER` | `admin` | MongoDB username |
| `MONGO_PASSWORD` | `changeme` | MongoDB password |
| `MONGO_DATABASE` | `people_db` | Database name for people records |
| `MONGO_AUTH_SOURCE` | `admin` | Authentication source database |

Expected MongoDB collection: `people`
Schema: `{_id, ho_ten, cccd, so_bhxh, so_dien_thoai, email, ngay_sinh, gioi_tinh, dia_chi, que_quan, noi_sinh, dan_toc, quoc_tich, truong_cong_tac, chuc_vu, don_vi, trang_thai, ghi_chu, ...}`

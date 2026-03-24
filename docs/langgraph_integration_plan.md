# Kế hoạch tích hợp LangGraph vào NexusRAG

> **Ngày thảo luận:** 2026-03-24
> **Branch triển khai:** `feature/langgraph-agent`
> **Chiến lược:** Module mới song song — không phá vỡ code cũ, có thể A/B test

---

## 1. Bối cảnh & Vấn đề hiện tại

### Kiến trúc agent cũ (`backend/app/api/chat_agent.py`)

```
User → SSE Endpoint (chat_agent.py)
         ↓
    Agent Loop (max 3 vòng, thủ công ~500 dòng)
         ↓
    1 tool duy nhất: search_documents
    [Gemini: native FunctionDeclaration | Ollama: XML <tool_call> | OpenAI: JSON schema]
         ↓
    HRAG Retriever → ChromaDB + KG + BM25 + Reranker
         ↓
    Stream tokens qua SSE
```

### Vấn đề cần giải quyết

| Vấn đề | Chi tiết |
|--------|----------|
| **1 tool duy nhất** | Chỉ có `search_documents`, không thể thêm tool mới dễ dàng |
| **Agent loop thủ công** | ~500 dòng code khó debug, khó test từng bước |
| **Routing hard-coded** | Greeting detection bằng regex/token matching trong code |
| **Không có graph visibility** | Không thể trace flow, không tích hợp LangSmith |
| **Khó mở rộng** | Thêm tool mới phải sửa nhiều chỗ khác nhau |
| **Không có checkpointing** | Mỗi request độc lập, không thể resume |

---

## 2. Kiến trúc mới với LangGraph

### 2.1 Cấu trúc thư mục

```
backend/app/
├── services/
│   ├── hrag_service.py              ← giữ nguyên (không thay đổi)
│   ├── deep_retriever.py            ← giữ nguyên
│   └── agent/                       ← MỚI - LangGraph module
│       ├── __init__.py
│       ├── state.py                 # AgentState TypedDict
│       ├── tools.py                 # Tất cả tools (search, list, summarize...)
│       ├── nodes.py                 # Các node của graph
│       ├── graph.py                 # StateGraph builder & compile
│       └── streaming.py             # SSE adapter cho LangGraph events
├── api/
│   ├── chat_agent.py                ← giữ nguyên (legacy fallback)
│   └── chat_agent_lg.py             ← MỚI - LangGraph endpoint
```

### 2.2 State Design (`state.py`)

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # --- Conversation ---
    messages: Annotated[list, add_messages]   # add_messages reducer: append, không overwrite

    # --- Context ---
    workspace_ids: list[int]
    document_ids: Optional[list[int]]
    user_id: Optional[int]
    session_id: Optional[str]
    system_prompt: str
    enable_thinking: bool

    # --- Retrieval results ---
    sources: list[dict]           # chunks từ search, format: ChatSourceChunk.model_dump()
    images: list[dict]            # image refs, format: ChatImageRef.model_dump()
    image_parts: list[dict]       # raw image bytes cho vision LLM
    kg_summaries: list[str]       # knowledge graph summaries
    existing_citation_ids: set    # citation IDs đã dùng (tránh trùng)

    # --- Agent control ---
    intent: str                   # "greeting" | "search" | "list_docs" | "summarize" | "kg_query"
    rewritten_query: str          # query đã được Qwen3-4B rewrite
    iterations: int               # chống infinite loop (max = 3)
    tool_called: bool             # đã gọi tool chưa

    # --- Memory ---
    user_memory_context: str      # context từ pgvector memory search

    # --- Output ---
    final_answer: str
    citation_map: dict            # citation_id → source info
```

### 2.3 Graph Flow

```
START
  │
  ▼
[memory_recall]                    ← Tải user memories từ pgvector (async)
  │
  ▼
[intent_classifier]                ← Qwen3-4B: phân loại + rewrite query
  │                                   Output JSON: {"intent": "...", "rewritten_query": "..."}
  │
  ├─ "greeting" ──────────────────► [direct_answer]      ──► END
  │
  ├─ "search" ────────────────────► [tool_executor]
  │                                        │
  │                                        ▼
  │                                  [answer_generator]  ──► END
  │
  ├─ "list_docs" ─────────────────► [tool_executor]
  │                                        │
  │                                        ▼
  │                                  [answer_generator]  ──► END
  │
  ├─ "summarize" ─────────────────► [tool_executor]
  │                                        │
  │                                        ▼
  │                                  [answer_generator]  ──► END
  │
  └─ "kg_query" ──────────────────► [tool_executor]
                                           │
                                           ▼
                                     [answer_generator]  ──► END
```

### 2.4 Intent Classifier — Qwen3-4B

**Lý do chọn Qwen3-4B:**
- Task đơn giản (classification + query rewrite) → model nhỏ là đủ
- Local model → latency thấp (~100-300ms), không tốn API cost
- Đã proven: đang dùng Qwen3-4B cho document type classification
- Hỗ trợ tiếng Việt tốt
- Tái dụng `get_memory_agent()` từ `services/llm/__init__.py`

**Prompt design:**
```python
CLASSIFIER_SYSTEM = """
Classify the user's intent and rewrite their query for better document retrieval.
Respond ONLY with valid JSON, no explanation.

Intent categories:
- "greeting": greetings, thanks, farewells, chitchat (no document search needed)
- "search": questions about document content, data, facts (needs search_documents)
- "list_docs": wants to know what documents are available ("có tài liệu gì?")
- "summarize": wants a summary of a specific document
- "kg_query": asks about entity relationships, knowledge graph

Output format:
{"intent": "<category>", "rewritten_query": "<improved search query>", "needs_tool": true|false}

If intent is "greeting", set rewritten_query to "" and needs_tool to false.
"""
```

**Fallback an toàn:**
```python
def parse_classifier_output(response: str) -> dict:
    try:
        data = json.loads(response.strip())
        intent = data.get("intent", "search")
        if intent not in VALID_INTENTS:
            return {"intent": "search", "rewritten_query": "", "needs_tool": True}
        return data
    except json.JSONDecodeError:
        return {"intent": "search", "rewritten_query": "", "needs_tool": True}
```

### 2.5 Tools (`tools.py`)

```python
# Tool 1 — Wrap existing HRAG retriever (giữ nguyên _execute_search_documents)
@tool
async def search_documents(query: str, top_k: int = 8) -> str:
    """Search the knowledge base for relevant document sections."""
    ...

# Tool 2 — NEW
@tool
async def list_documents(workspace_ids: list[int]) -> str:
    """List all documents available in the current workspace(s)."""
    ...

# Tool 3 — NEW
@tool
async def summarize_document(document_id: int) -> str:
    """Get a comprehensive summary of a specific document."""
    # Fetch markdown từ MinIO → call main LLM for summarization
    ...

# Tool 4 — NEW
@tool
async def query_knowledge_graph(entity: str) -> str:
    """Query the knowledge graph for entity relationships."""
    # Gọi KnowledgeGraphService.query()
    ...
```

### 2.6 SSE Streaming Adapter (`streaming.py`)

Giữ nguyên SSE format để **frontend không cần sửa gì**:

```python
async def stream_graph_to_sse(
    graph_app,
    initial_state: AgentState,
) -> AsyncGenerator[str, None]:
    """
    Convert LangGraph astream_events → SSE format hiện tại.

    SSE events mapping:
    - on_chat_model_stream    → "token" event
    - on_tool_start           → "status" event
    - on_tool_end (sources)   → "sources" event
    - on_tool_end (images)    → "images" event
    - on_chain_end (final)    → "complete" event
    - on_node_start (think)   → "thinking" event
    """
```

---

## 3. Lộ trình triển khai

### Giai đoạn 1: Core LangGraph Module ✅ (đang làm)
- [ ] `backend/app/services/agent/__init__.py`
- [ ] `backend/app/services/agent/state.py` — AgentState TypedDict
- [ ] `backend/app/services/agent/tools.py` — search_documents wrapper + 3 tools mới
- [ ] `backend/app/services/agent/nodes.py` — memory_recall, intent_classifier, tool_executor, answer_generator, direct_answer
- [ ] `backend/app/services/agent/graph.py` — StateGraph builder, conditional edges, compile
- [ ] `backend/app/services/agent/streaming.py` — SSE adapter

### Giai đoạn 2: Endpoint & Infrastructure ✅ (đang làm)
- [ ] `backend/app/api/chat_agent_lg.py` — SSE endpoint `/rag/chat/agent-lg/stream`
- [ ] `backend/requirements.txt` — thêm `langgraph>=0.2.0`, `langchain-core>=0.3.0`
- [ ] `backend/app/core/config.py` — thêm `NEXUSRAG_AGENT_BACKEND`, `NEXUSRAG_LG_CLASSIFIER_MODEL`
- [ ] `backend/app/api/router.py` — register route mới
- [ ] `.env.example` — document config mới

### Giai đoạn 3: Bổ sung Tools Mới
- [ ] Tool `list_documents` — query PostgreSQL
- [ ] Tool `summarize_document` — MinIO markdown + LLM
- [ ] Tool `query_knowledge_graph` — LightRAG KG query
- [ ] Test từng tool độc lập

### Giai đoạn 4: Migration & Cleanup
- [ ] A/B test: chạy song song legacy vs LangGraph endpoint
- [ ] So sánh latency, quality, error rate
- [ ] Migrate `chat_session.py` sang dùng endpoint mới
- [ ] Xóa code legacy nếu ổn định

---

## 4. Configuration

### Biến môi trường mới (`.env`)

```bash
# ---- LangGraph Agent ----
# Chọn backend: "langgraph" hoặc "legacy" (default: legacy để không break)
NEXUSRAG_AGENT_BACKEND=legacy

# Model dùng cho intent classification (Qwen3-4B qua memory agent endpoint)
# Mặc định: tái dùng MEMORY_AGENT_BASE_URL và MEMORY_AGENT_MODEL
NEXUSRAG_LG_USE_MEMORY_AGENT_AS_CLASSIFIER=true

# LangGraph checkpointer (state persistence)
# "memory" = in-memory (stateless, default)
# "none" = không dùng checkpointer
NEXUSRAG_LG_CHECKPOINTER=memory

# Max iterations để chống infinite loop
NEXUSRAG_LG_MAX_ITERATIONS=3
```

### Tái dụng config hiện có

| Config mới cần | Tái dùng từ | Giá trị default |
|----------------|-------------|-----------------|
| Classifier LLM | `MEMORY_AGENT_*` | Qwen3-4B qua vLLM |
| Generator LLM | `LLM_PROVIDER` | Gemini / Ollama |
| Max iterations | hard-coded `MAX_AGENT_ITERATIONS` | 3 |

---

## 5. Nguyên tắc thiết kế

### ✅ Phải tuân thủ
1. **Backward compatible:** Frontend không thay đổi gì — SSE format giữ nguyên
2. **Feature flag:** `NEXUSRAG_AGENT_BACKEND=legacy` để rollback ngay nếu có lỗi
3. **Reuse existing code:** `_execute_search_documents`, `_search_memories`, `_save_memory` từ `chat_agent.py`
4. **Fallback an toàn:** Mọi JSON parse fail → intent = "search" (safe default)
5. **Max iterations guard:** Luôn có exit condition tránh infinite loop

### ❌ Không làm
1. Không thay đổi `hrag_service.py`, `deep_retriever.py`
2. Không sửa `chat_agent.py` (legacy code phải giữ nguyên)
3. Không thay đổi SSE event format
4. Không dùng LangChain LLM wrappers — dùng LLM providers hiện có của dự án

---

## 6. So sánh trước/sau

| Tiêu chí | Trước (Legacy) | Sau (LangGraph) |
|----------|----------------|-----------------|
| Số tools | 1 (`search_documents`) | 4+ tools, dễ thêm |
| Agent logic | ~500 dòng thủ công | Graph khai báo rõ ràng |
| Intent routing | Regex/token matching | Qwen3-4B classifier |
| Query rewriting | Nhờ main LLM | Qwen3-4B (riêng, nhanh) |
| Debug | Khó trace flow | LangSmith / log events |
| Test | Khó test từng bước | Mỗi node test độc lập |
| Checkpointing | Không có | Có (memory/postgres) |
| Thêm tool mới | Phải sửa nhiều chỗ | Chỉ thêm 1 function |

---

## 7. Rủi ro & Giảm thiểu

| Rủi ro | Mức độ | Giảm thiểu |
|--------|--------|------------|
| LangGraph adds latency | Thấp | Classifier dùng model nhỏ (<300ms) |
| Qwen3-4B classify sai | Trung bình | Fallback "search" là default an toàn |
| LangGraph dependency conflict | Thấp | Pin version, test isolated |
| Feature regression | Thấp | Legacy endpoint vẫn hoạt động song song |

---

*Cập nhật lần cuối: 2026-03-24*

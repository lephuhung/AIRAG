# AIRAG — Sơ đồ hệ thống

> Nguồn: đọc trực tiếp source (2026-06). LangGraph supervisor là **agent backend duy nhất**
> (web `chat_session.py` và Telegram `telegram_service.py` đều gọi `get_supervisor_graph()`).
> Lưu ý: `CLAUDE.md` còn ghi "legacy mặc định" — đã lỗi thời.

## 1. Kiến trúc tổng thể

```mermaid
flowchart TB
    subgraph clients[Clients]
        WEB[Web UI<br/>React 19 + Vite :5174]
        TG[Telegram Bot]
        API[Third-party<br/>X-API-Key]
    end

    subgraph backend[FastAPI Backend :8080]
        direction TB
        CS["/chat-sessions/{id}/stream<br/>(chat_session.py)"]
        LG["/agent-lg/stream<br/>(chat_agent_lg.py)"]
        TW["/integrations/telegram/webhook<br/>(integrations.py)"]
        SUP[["LangGraph Supervisor Graph<br/>(supervisor.py)"]]
        CS --> SUP
        LG --> SUP
        TW --> TGSVC[telegram_service.process_update] --> SUP
    end

    subgraph workers[Async Workers - RabbitMQ]
        PARSE[parse_worker] --> EMBED[embed_worker]
        PARSE --> CAP[caption_worker]
        PARSE --> KG[kg_worker]
        MEM[memory_worker<br/>Graphiti save]
    end

    subgraph stores[Storage]
        PG[(PostgreSQL<br/>metadata/chat/users)]
        CHROMA[(ChromaDB<br/>vectors)]
        MINIO[(MinIO<br/>raw/markdown/captions)]
        NEO[(Neo4j<br/>Graphiti + LegalKG)]
        LR[(LightRAG<br/>file-based KG)]
        MONGO[(MongoDB<br/>people search)]
    end

    WEB -->|SSE| CS
    API -->|SSE| LG
    TG -->|HTTPS| TW

    SUP --> CHROMA & NEO & LR & MONGO & PG
    EMBED --> CHROMA
    CAP --> MINIO
    KG --> NEO & LR
    SUP --> MEM --> NEO
```

## 2. Luồng Web → LangGraph (SSE streaming)

```mermaid
sequenceDiagram
    autonumber
    participant U as Web UI
    participant EP as chat_session.py<br/>chat_stream_session
    participant ST as streaming.py<br/>stream_agent_to_sse
    participant G as Supervisor Graph
    participant DB as PostgreSQL

    U->>EP: POST /chat-sessions/{id}/stream (JWT)
    EP->>DB: lưu user message
    EP->>EP: _get_accessible_workspaces()<br/>+ conversation summary context
    EP->>EP: build_initial_state(...)
    EP->>ST: stream_agent_to_sse(graph, state)
    loop mỗi event của graph
        G-->>ST: token / status / sources / people_data
        ST-->>EP: SSE string (event: ...\ndata: ...)
        EP-->>U: SSE (buffer + replay)
    end
    G-->>ST: complete (answer, sources, images)
    EP->>DB: lưu assistant message + agent_steps
    EP-)DB: enqueue Graphiti memory save (RabbitMQ)
```

## 3. Luồng Telegram → LangGraph (in-process, không SSE)

```mermaid
sequenceDiagram
    autonumber
    participant TG as Telegram
    participant WH as integrations.py<br/>telegram_webhook
    participant SVC as telegram_service<br/>process_update
    participant G as Supervisor Graph
    participant DB as PostgreSQL

    TG->>WH: POST /telegram/webhook (X-Telegram secret)
    WH->>WH: verify secret token (DB config / .env fallback)
    WH-)SVC: background_tasks.add_task(process_update)
    WH-->>TG: 200 {ok:true} (ngay lập tức)

    Note over SVC: token bot lấy từ DB (ContextVar per-update)
    SVC->>SVC: text bắt đầu "/" ? _handle_command : _handle_question
    SVC->>DB: _get_link(chat_id) → TelegramLink → user_id
    alt chưa link
        SVC-->>TG: "Gửi /link <mã>"
    else đã link
        SVC->>SVC: resolve workspace + _ensure_session (auto rollover)
        SVC->>SVC: build_initial_state(channel=telegram)
        SVC->>G: stream_agent_events(graph, state)
        loop token throttle (EDIT_MIN_INTERVAL_S)
            G-->>SVC: token
            SVC-->>TG: editMessageText (preview, markdown stripped)
        end
        G-->>SVC: complete
        SVC-->>TG: final message (HTML)
        SVC->>DB: lưu assistant message (sources KHÔNG hiển thị)
    end
```

**Khác biệt cốt lõi 2 kênh:** Web stream SSE ra trình duyệt; Telegram tiêu thụ
`stream_agent_events` **in-process** rồi throttle `editMessageText`. Auth: web = JWT;
Telegram = `secret_token` (chỉ chứng minh "từ Telegram") + `TelegramLink` (danh tính user thật).

## 4. Bên trong Supervisor Graph (nodes & edges)

```mermaid
flowchart TD
    START([START]) --> QA[query_analyzer<br/>decompose đa bước]
    QA --> SUP{supervisor<br/>classify intent + route<br/>1 LLM call}

    SUP -->|route_from_supervisor| MEM[memory_recall<br/>Graphiti]
    SUP --> RAG[rag]
    SUP --> RES[resolve_doc_agent]
    SUP --> WR[write]
    SUP --> PPL[people]
    SUP --> DIR[direct]
    SUP --> RX[react_executor<br/>flag: NEXUSRAG_LG_RAG_REACT]
    SUP -->|FINISH| E([END])

    MEM --> QE[query_enricher<br/>inject memory]
    QE -->|personal| DIR
    QE --> RAG
    QE --> WR
    QE --> PPL
    QE --> RX

    RAG -->|route_from_rag| EVAL[result_evaluator<br/>quality check + multi-step]
    RAG -->|abbrev loop / search_section| SUP

    RES --> AG[answer_generator]
    RES -->|search_section| RAG
    RES -->|ambiguous| E

    EVAL --> AG
    EVAL -->|retry| RAG
    EVAL -->|next step| SUP

    PPL --> MF[mongo_formatter]

    AG --> E
    MF --> E
    WR --> E
    DIR --> E
    RX --> E

    classDef flag fill:#fde,stroke:#c39
    class RX flag
```

**Ghi chú điều hướng:**
- `needs_memory=True` → đi qua `memory_recall → query_enricher` trước khi tới agent đích; ngược lại bypass thẳng.
- `resolve_doc` luôn vào `resolve_doc_agent` (hoặc `react_executor` nếu bật ReAct).
- Khi `NEXUSRAG_LG_RAG_REACT=true`, mọi nhánh `rag` được thay bằng `react_executor` (1 vòng tool-calling, terminal).
- Loop guard: `NEXUSRAG_LG_MAX_ITERATIONS` (mặc định 5–6); `result_evaluator` có thể quay lại `supervisor` cho bước kế tiếp.

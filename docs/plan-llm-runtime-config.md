# Plan — Cấu hình LLM Model qua WebUI (Runtime Config)

> ⚡ **V2 REDESIGN (đã duyệt)**: chuyển từ "mỗi role nhập riêng url/key" sang
> kiến trúc 2 tầng **Connections + Assignments** (xem §12) — 1 endpoint khai
> báo 1 lần, các nhiệm vụ chỉ chọn kết nối + model. Thêm 4 role:
> `stt`, `tts`, `embedding`, `rerank`. Các mục §4–§7 bên dưới giữ nguyên làm
> nền tảng kỹ thuật (snapshot cache, version-check, encryption); nơi mâu thuẫn
> thì §12 thắng.

> Trạng thái: **PROPOSAL** (chưa implement). Tài liệu này là bản thiết kế đầy đủ
> phần Backend + Worker cho tính năng cấu hình LLM từ WebUI, không phụ thuộc `.env`.

---

## 1. Mục tiêu & Phạm vi

### Mục tiêu
- Admin cấu hình **model LLM cho từng vai trò** (backend chat, thinking, memory,
  KG extraction, caption…) trực tiếp trên WebUI.
- Không cần sửa `.env`, không cần restart container (cả backend lẫn worker).
- Backward-compatible: **không có override trong DB → dùng đúng giá trị `.env`**
  như hiện tại. Deploy hiện tại không bị phá.

> Lý do tách `vision`: model chat mạnh/rẻ thường là text-centric (DeepSeek-V3,
> Qwen3.6-35B-A3B…). Nếu gộp chung, admin chọn model text cho chat thì
> `caption_worker` gặp `provider.supports_vision() == False` và **bỏ qua toàn bộ
> caption ảnh**. Slot `vision` cho phép "Dùng Main LLM" (default) hoặc chỉ định
> model vision riêng mà không ảnh hưởng chat.

### Trong phạm vi (Phase 1–3)
| Role | Dùng ở đâu | Factory hiện tại |
|---|---|---|
| `main` | Chat answer-gen, direct answer, caption BẢNG (`caption_worker`), `knowledge_graph_service` | `get_llm_provider()` |
| `vision` | Caption ẢNH (`caption_worker`). **Mặc định kế thừa `main`**; chỉ set override khi cần model vision riêng (qwen2.5-vl, gemma3:12b, gemini-2.5-flash…) | `get_llm_provider()` + guard `supports_vision()` |
| `thinking` | Supervisor routing, ReAct judge (LangGraph) | `get_thinking_provider()` |
| `memory_agent` | Intent classify, condense follow-up, query analyzer/enricher, contextual embeddings, conversation summary | `get_memory_agent()` |
| `kg_extract` | Trích xuất thực thể/quan hệ (`kg_worker` → `legal_kg_service`) | `get_kg_llm_provider()` |
| `graphiti` | Memory worker — Graphiti entity/fact extraction | build trực tiếp `AsyncOpenAI(...)` trong `graphiti_client.py` |

Mỗi role gồm: `provider` (gemini | ollama | openai_compatible), `base_url`,
`model`, `api_key`, và tuỳ chọn `extra` (thinking_level, max_tokens,
`max_concurrency`, `is_vllm`).

### Ngoài phạm vi (cố tình loại)
- **Embedding / Reranker** (`HRAG_EMBEDDING_MODEL`, `HRAG_RERANKER_MODEL`):
  load GPU eager lúc startup (`preload_models()`) + đổi model làm lệch vector cũ
  (dimension mismatch với ChromaDB/Neo4j) → giữ nguyên `.env`.
- Per-workspace / per-user model override (Phase 4 tuỳ chọn sau).

---

## 2. Bản đồ code bị ảnh hưởng (kết quả rà soát)

### Nguồn sự thật hiện tại
- `app/core/config.py` — pydantic-settings đọc `.env`, singleton `settings`.
- `app/services/llm/__init__.py` — **nút cổ duy nhất**, 4 factory đều `@lru_cache`:
  - `get_llm_provider()` ← `LLM_PROVIDER`, `OPENAI_COMPATIBLE_*`, `OLLAMA_*`
  - `get_thinking_provider()` ← `NEXUSRAG_LG_THINKING_*` (+ alias `ANTHROPIC_*`)
  - `get_memory_agent()` ← `MEMORY_AGENT_BASE_URL/MODEL/API_KEY/LOCAL`
  - `get_kg_llm_provider()` ← `LEGAL_KG_LLM_PROVIDER/BASE_URL/MODEL/API_KEY`
  - `get_embedding_provider()` ← giữ nguyên (ngoài phạm vi)
- **Ngoại lệ cần refactor riêng**: `app/services/memory/graphiti_client.py`
  tự dựng `AsyncOpenAI(api_key=settings.GRAPHITI_LLM_API_KEY, model=..., base_url=...)`
  thay vì qua factory (vì Graphiti cần client OpenAI gốc).

### Nơi tiêu thụ (điểm gọi factory)
- **Lưu ý sync/async**: phần lớn điểm gọi là hàm **sync** hoặc chạy trong
  threadpool (`asyncio.to_thread(provider.complete, ...)` trong `caption_worker`,
  các node LangGraph, `supervisor.py` gọi `get_memory_agent()` trực tiếp).
  → Factory bắt buộc giữ chữ ký **sync** (xem §5.2 — snapshot cache).
- **Backend process**: `services/agents/supervisor.py` (21 chỗ),
  `agent/nodes.py`, `agent/tools.py`, `agents/write_agent.py`,
  `agents/react_tools.py`, `agents/people_agent.py`, `api/rag.py`,
  `document_type_classifier.py`, `doc_resolver.py`, `deep_document_parser.py`,
  `models/loader.py` (preload).
- **Worker containers** (tiêu thụ message RabbitMQ, có Postgres sẵn):
  - `caption_worker` → `get_llm_provider()` (mỗi document)
  - `kg_worker` → `legal_kg_service` (`get_kg_llm_provider`) +
    `knowledge_graph_service` (`get_llm_provider`, `get_embedding_provider`)
  - `memory_worker` → `graphiti_client` (dựng client riêng)
  - `embed_worker` → **có dùng LLM** khi `HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS=true`:
    `embedding/contextual_embedder.py:157` gọi `get_memory_agent()` để sinh câu
    ngữ cảnh từng chunk trước khi embed → phải nằm trong phạm vi đồng bộ config.
  - `parse_worker` → không dùng LLM

### Hạ tầng liên quan
- Workers đã nối **PostgreSQL** (`app.core.database.async_session_maker`) —
  kênh đồng bộ tự nhiên với backend.
- Workers **không** nối Redis (Redis chỉ gate multi-process của backend).
- Backend chạy được nhiều process (`WEB_CONCURRENCY>1`) khi `REDIS_ENABLED=true`.
- Đã có sẵn: `require_superadmin` dep (`core/deps.py`),
  `audit_service.record_for_actor()`, endpoint read-only `/api/v1/config/status`,
  pattern admin pages ở frontend.

---

## 3. Thiết kế tổng thể

```
┌─────────┐  PUT /admin/llm-config/{role}   ┌──────────────────────────────┐
│ WebUI   │ ───────────────────────────────▶│ Backend                       │
│ (admin) │ ◀───────────────────────────────│  RuntimeConfigService         │
└─────────┘   GET status / POST test        │   ├─ ghi system_settings      │
                                            │   ├─ tăng _config_version     │
                                            │   ├─ invalidate provider cache│
                                            │   └─ audit_log                │
                                            └──────────────┬───────────────┘
                                                           │ PostgreSQL
                                            ┌──────────────▼───────────────┐
                                            │ system_settings               │
                                            │  key | value_enc | updated_at │
                                            │  _config_version = N          │
                                            └──────────────▲───────────────┘
                                                           │ SELECT version
                                                           │ (đầu mỗi message)
                                            ┌──────────────┴───────────────┐
                                            │ Workers (caption/kg/memory)   │
                                            │  version đổi → rebuild provider│
                                            └──────────────────────────────┘
```

### Nguyên tắc cốt lõi
1. **`.env` = default, DB = override.** `EffectiveConfig = merge(settings default,
   DB row nếu có)`. Xoá row → quay về `.env`.
2. **Version-check thay vì push.** Không dùng RabbitMQ/Redis broadcast:
   - Worker restart lúc broadcast sẽ **mất tín hiệu** → vẫn phải poll để bù.
   - Redis bắt worker thêm dependency mới chỉ để tiết kiệm vài giây.
   - Check 1 row `_config_version` đầu mỗi message: ~0.1ms, không đo được.
3. **Atomicity:** resolve config **đầu mỗi job**, giữ nguyên provider suốt job.
   Đổi model giữa chừng không cắt ngang document đang xử lý — job dở hoàn tất
   bằng model cũ, message tiếp theo dùng model mới.

---

## 4. Data model

```sql
-- migrations/create_system_settings.sql
CREATE TABLE IF NOT EXISTS system_settings (
    key         VARCHAR(128) PRIMARY KEY,
    value_enc   TEXT NOT NULL,          -- JSON, api_key mã hoá Fernet
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by  UUID REFERENCES users(id)  -- nullable
);

-- Row đặc biệt, không mã hoá, chỉ để poll rẻ:
INSERT INTO system_settings(key, value_enc) VALUES ('_config_version', '0')
ON CONFLICT DO NOTHING;
```

Key của các role: `llm.main`, `llm.thinking`, `llm.memory_agent`,
`llm.kg_extract`, `llm.graphiti`.

Value shape (JSON):
```json
{
  "provider": "openai_compatible",
  "base_url": "http://10.10.0.240:8000/v1",
  "model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "api_key_enc": "<fernet>",
  "extra": { "thinking_level": "medium", "max_output_tokens": 4096 }
}
```

**Mã hoá**: Fernet symmetric, key phái sinh từ biến môi trường mới
`SETTING_ENCRYPTION_KEY` (derive bằng HKDF/sha256). Mất key cũ → phải nhập lại
API key (fail an toàn, không lộ key dưới dạng plaintext trong DB).

---

## 5. PHASE 1 — Backend core

### 5.1 File mới
| File | Nội dung |
|---|---|
| `app/models/system_setting.py` | SQLAlchemy model `SystemSetting` |
| `migrations/create_system_settings.sql` | SQL migration |
| `app/services/runtime_config.py` | **RuntimeConfigService** — trái tim của feature |

`RuntimeConfigService` (API chính):
```python
class RuntimeConfigService:
    async def get_effective(role: str) -> EffectiveLLMConfig
        # merge settings-default + DB override; cache in-process theo version
    async def list_all() -> dict[str, RoleStatus]     # cho GET admin API
    async def set_override(role, payload, actor)      # ghi DB + bump version + audit
    async def clear_override(role, actor)             # xoá row → về .env default
    async def get_version() -> int                    # SELECT _config_version
    async def bump_version()                          # trong cùng transaction với set/clear
```

Cache in-process: keyed `(version)`; backend API process invalidate ngay khi ghi.
Với `WEB_CONCURRENCY>1`: mỗi request check version (giống worker) hoặc subscribe
Redis pub/sub khi `REDIS_ENABLED=true` (tối ưu, làm sau).

#### ⚠️ Ràng buộc Sync/Async — Snapshot Cache (BẮT BUỘC)
Phần lớn caller là **sync** (`supervisor.py` gọi `get_memory_agent()` trực tiếp,
`caption_worker` chạy `provider.complete` qua `asyncio.to_thread`, các LangGraph
node sync, `conversation_summary_service`). Nếu đổi factory thành `async def`
thì phải sửa hàng chục file caller — không chấp nhận.

→ Thiết kế: factory **giữ nguyên chữ ký sync**, đọc từ một
**In-Memory Snapshot** được làm mới bởi các điểm refresh *async*:
```python
# app/services/runtime_config.py
_snapshot: dict[str, EffectiveLLMConfig] = {}
_snapshot_version: int = -1

def get_effective_sync(role: str) -> EffectiveLLMConfig:
    """SYNC — đọc snapshot. Không bao giờ chạm DB."""
    return _snapshot.get(role) or _build_from_settings(role)

async def refresh_snapshot() -> bool:
    """ASYNC — gọi tại: (a) startup lifespan, (b) request middleware
    (throttled ~1 lần/giây), (c) ensure_fresh_config() của worker,
    (d) ngay sau khi admin lưu/clear override."""
    global _snapshot_version, _snapshot
    v = await _read_db_version()
    if v != _snapshot_version:
        _snapshot = {r: await _load_effective(r) for r in ROLES}
        _snapshot_version = v
        return True   # caller rebuild provider nếu muốn
    return False
```
Các hàm factory trong `app/services/llm/__init__.py` chuyển từ `@lru_cache`
sang đọc `get_effective_sync(role)` + provider cache keyed by
`_snapshot_version`. Signature `get_llm_provider()` … **không đổi** →
không phải refactor caller nào.

### 5.2 Sửa `app/services/llm/__init__.py` (refactor factory)
- Thay `@lru_cache` bằng provider cache keyed by `_snapshot_version`, đọc qua
  **sync snapshot** của RuntimeConfigService (xem §5.1):
```python
_prov_cache: dict[str, tuple[int, LLMProvider]] = {}   # role -> (version, provider)

def get_llm_provider() -> LLMProvider:
    """Giữ NGUYÊN signature sync. Không sửa bất kỳ caller nào."""
    cfg = runtime_config.get_effective_sync("main")
    ver = runtime_config.snapshot_version()
    hit = _prov_cache.get("main")
    if hit and hit[0] == ver:
        return hit[1]
    p = trace_llm(build_provider(cfg), label="main_llm")
    _prov_cache["main"] = (ver, p)
    return p
```
- Role `vision`: `get_llm_provider(role="vision")` — nếu snapshot không có
  override cho `vision` thì fallback đúng cấu hình của `main` (kế thừa).
- **Provider thứ 3 (DeepSeek/OpenRouter/Groq…) chạy qua `openai_compatible`**:
  sửa bug hiện hữu — `extra_body={"chat_template_kwargs": ...}` đang hardcode
  cho MỌI request (dòng ~196/222) là tham số riêng của vLLM; OpenAI thật và một
  số provider nghiêm ngặt trả 400 Unknown parameter. Tách thành flag
  `is_vllm` trong EffectiveConfig (auto-detect khi test connection qua response
  header, hoặc toggle thủ công trên UI) — chỉ gửi extra_body khi bật.
  Reasoning models (`deepseek-reasoner`) OK sẵn nhờ `_strip_think()`.
- `build_provider(cfg)` tách logic if/elif gemini|ollama|openai_compatible ra
  hàm thuần nhận config — dùng chung cho cả test-connection.
- Refresh snapshot chạy ở: lifespan startup, middleware throttled (~1s/lần),
  ngay sau PUT/DELETE admin, và `ensure_fresh_config()` của worker.

### 5.3 Refactor `graphiti_client.py`
Thay 3 dòng đọc `settings.GRAPHITI_LLM_*` bằng `runtime_config.get_effective("graphiti")`.
Client Graphiti được dựng lại khi config đổi (có guard theo version).

### 5.4 Các điểm đọc thẳng `settings.*` liên quan LLM
Audit và chuyển sang effective config: `supervisor.py` (số lượng lớn nhất),
`nodes.py`, `tools.py`, `loader.py`. Ưu tiên những chỗ quyết định model/URL;
những chỗ chỉ đọc flag bật/tắt (`NEXUSRAG_REACT_JUDGE`...) giữ nguyên.

### 5.5 API admin (router mới `app/api/llm_config.py`, prefix `/admin/llm-config`)
Tất cả guard `Depends(require_superadmin)`:

| Method & path | Chức năng |
|---|---|
| `GET /api/v1/admin/llm-config` | Trạng thái mọi role: effective values (api_key masked `sk-***last4`), có override hay không, updated_at/by |
| `PUT /api/v1/admin/llm-config/{role}` | Lưu override. **Body bắt buộc kèm kết quả test** (`test_token` từ bước 2) hoặc server tự test trước khi commit |
| `DELETE /api/v1/admin/llm-config/{role}` | Xoá override → về `.env` default + bump version |
| `POST /api/v1/admin/llm-config/test` | Body: full config thử nghiệm (chưa lưu). Server dựng provider tạm → ping thật (chi tiết fallback bên dưới). Trả `{ok, latency_ms, models[], error}` |
| `POST /api/v1/admin/llm-config/models` | **Auto-load danh sách model**: body `{provider, base_url, api_key}` (nhẹ, không ping completion, không lưu). Trả `{ok, models[], source}`. Nguồn theo provider: `openai_compatible` → `GET {base_url}/models`; `ollama` → `GET {host}/api/tags`; `gemini` → ListModels API; proxy chặn → `{ok:false, source:"none"}` (UI chuyển sang nhập tay). Cache kết quả theo `(base_url, api_key-hash)` ~5 phút trong process để tránh gọi lại liên tục |

**Logic test-connection cho `openai_compatible` (proxy-tolerant):**
1. Thử `GET {base_url}/models` trước — nếu OK → trả về danh sách model để UI
   đổ vào combobox gợi ý.
2. Nếu `/models` fail (404/405/403 — nhiều proxy nội bộ như 9router, reverse
   proxy vLLM chặn endpoint này): **fallback tự động** sang completion siêu ngắn
   `POST {base_url}/chat/completions` với `messages=[{"role":"user",
   "content":"hi"}], max_tokens=1` → vẫn đo được `latency_ms` thật và xác nhận
   endpoint chat chạy bình thường.
3. Trả thêm cờ `models_list_available: bool` để UI biết có được phép hiển thị
   dropdown hay phải cho nhập tay.
Provider khác: `gemini` = list-models, `ollama` = `GET /api/tags`.
| `GET /api/v1/admin/llm-config/events` *(tuỳ chọn)* | SSE stream báo version đổi (UI auto-refresh) |

Mọi set/clear ghi `audit_log` qua `audit_service.record_for_actor()`.

### 5.6 Cập nhật `/api/v1/config/status`
Đọc từ `RuntimeConfigService.list_all()` thay vì `settings` trực tiếp.

---

## 6. PHASE 2 — Worker đồng bộ

### 6.1 Cơ chế (đã chốt: version-check per-message)
Thêm helper chung `app/workers/config_watch.py`:
```python
async def ensure_fresh_config() -> None:
    """SELECT _config_version; nếu khác snapshot trong process → clear provider cache."""
_local_version: int = -1

async def current_version() -> int:
    async with async_session_maker() as db:      # 1 query duy nhất
        return await db.scalar(
            select(SystemSetting.value_enc).where(key == "_config_version")
        )
```

### 6.2 Điểm gắn vào từng worker
Gọi `ensure_fresh_config()` **đầu mỗi handler**, trước khi resolve bất kỳ provider nào:

| Worker | Handler gắn check | Provider bị ảnh hưởng |
|---|---|---|
| `caption_worker` | `_handle_caption_message()` trước `_caption_images_concurrent` | `main`, `vision` |
| `kg_worker` | `handle_kg()` đầu hàm | `kg_extract`, `main` |
| `memory_worker` | handler message memory | `graphiti`, `memory_agent` |
| `embed_worker` | `_handle_embed_message()` **trước** bước contextual enrichment (`contextual_embedder` gọi `get_memory_agent()` khi `HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS=true`) | `memory_agent` |
| `parse_worker` | không gắn (không dùng LLM trong phạm vi) | — |

Chi phí: 1 query đơn giản/message. Có thể thêm TTL 10s trong process nếu muốn
giảm nữa (đổi config muộn nhất sau 10s + 1 message).

### 6.3 Hiệu lực thay đổi (kỳ vọng)
| Thành phần | Thấy config mới sau... |
|---|---|
| Backend API (chat agent, cùng process) | Ngay lập tức |
| Backend đa process (`WEB_CONCURRENCY>1`) | Request tiếp theo (check version per-request) |
| Worker idle | Message kế tiếp vào queue |
| Job đang chạy dở | Hoàn tất bằng model cũ (an toàn, không cắt ngang) |
| `embed_worker` (contextual enrichment) | Document embed kế tiếp |

---

## 7. PHASE 3 — WebUI

### 7.1 Trang `AdminLLMConfigPage` — route `/admin/llm`
Theo pattern admin pages sẵn có (`AdminTenantsPage`, route table trong `App.tsx`,
menu layout). Nội dung:

- **5 card role** (Main / Vision / Thinking / Memory Agent / KG Extraction /
  Graphiti), mỗi card mô tả ngắn vai trò ("Thinking — dùng cho định tuyến
  supervisor & judge").
- Card **Vision** có thêm toggle: **"Dùng Main LLM"** (default, kế thừa) ↔
  **"Chỉ định model riêng"** — khi chọn model riêng mới hiện form
  base_url/model/api_key. Kèm cảnh báo inline nếu model đang chọn không report
  vision capability (`supports_vision()`).

#### Combobox Model — TỰ ĐỘNG LOAD từ endpoint
Flow nhập liệu của admin:
1. Nhập/dán **Base URL + API key** → debounce 600ms khi cả hai hợp lệ →
   frontend **tự gọi `POST /models`** (không cần bấm nút) → đổ danh sách vào
   combobox Model.
2. Sửa URL hoặc key bất kỳ lúc nào → tự load lại danh sách.
3. Endpoint chặn `/models` (`source:"none"`) → badge "Không lấy được danh sách —
   nhập tay", combobox vẫn cho gõ custom value.
4. Model đang dùng (effective) luôn nằm đầu danh sách và được highlight.

#### Quick Presets (1-click fill)
Thanh preset đầu mỗi card, điền sẵn provider + base_url + gợi ý model:
| Preset | Điền sẵn |
|---|---|
| Local vLLM (Memory/9router) | `openai_compatible`, base_url theo alias hệ thống đang chạy |
| Google Gemini | `gemini`, gợi ý `gemini-2.5-flash` / `gemini-3.1-flash-lite-preview` |
| Ollama Local | `ollama`, `http://localhost:11434` |
| **DeepSeek** | `openai_compatible`, `https://api.deepseek.com/v1`, gợi ý `deepseek-chat` / `deepseek-reasoner` |
| OpenAI-compatible API | `openai_compatible`, trống để nhập OpenRouter/DeepSeek/Moonshot/… |
Preset chỉ **prefill form**, chưa lưu — admin vẫn phải Test → Save.

#### Combobox Model (dropdown + free-text)
Nếu test connection lấy được danh sách model (`models_list_available=true`) →
hiển thị dạng combobox dropdown gợi ý; ngược lại (proxy chặn `/models`) →
vẫn là ô text nhập tay tự do. Luôn cho phép gõ custom value kể cả khi có danh sách.
- Form mỗi card: dropdown Provider → hiện đúng trường (base_url/model/api_key);
  ô Model có nút **"Lấy danh sách model"** (gọi `/test`, đổ vào combobox);
  api_key dạng password masked, placeholder `sk-••••1234 (đang dùng .env)`.
- Badge trạng thái mỗi trường: `override` (xanh) vs `.env default` (xám);
  nút **Reset về mặc định** per-role.
- Nút **Kiểm tra kết nối** → hiện latency + số model tìm thấy; lưu chỉ enabled
  khi test pass (server cũng tự test lần nữa trước khi commit).
- Banner cảnh báo khi đổi model đang có job queue dài (đọc từ `/workers` status).

### 7.2 API client
Thêm methods vào `frontend/src/lib/api.ts`; types vào `src/types`.

---

## 8. Rủi ro & biện pháp

| Rủi ro | Biện pháp |
|---|---|
| Multi-process backend giữ provider cũ | Check version per-request; tối ưu sau bằng Redis pub/sub khi `REDIS_ENABLED=true` |
| API key lộ trong DB | Fernet + `SETTING_ENCRYPTION_KEY`; mask tuyệt đối trong mọi response |
| Config sai làm sập chat | Test bắt buộc pass trước khi lưu; resolve lỗi → fail-open về `.env` default + log warning |
| Đổi model giữa stream đang chạy | Provider resolve ở turn/job boundary, không swap mid-stream |
| Mất `SETTING_ENCRYPTION_KEY` | Decrypt fail → coi như chưa có override, fallback `.env`, log rõ |
| Worker container image cũ không có code mới | Version-check là no-op nếu thiếu bảng (try/except → dùng `.env`) — deploy an toàn theo thứ tự: migrate → backend → workers |
| Cache `@lru_cache` cũ còn sót chỗ khác | Grep toàn repo trong lúc review; `clear_thinking_provider_cache()` chuyển sang cơ chế version |
| Snapshot stale khi middleware throttle lỗi / DB chậm | Snapshot giữ giá trị cuối cùng tốt (last-known-good); refresh fail chỉ log, không làm rơi provider đang chạy |
| Admin chọn model non-vision cho slot Vision | UI cảnh báo qua `supports_vision()`; worker vẫn giữ hành vi skip an toàn như hiện tại |
| Bên thứ 3 (DeepSeek/OpenAI thật…) từ chối `extra_body` vLLM | Flag `is_vllm` mặc định OFF; chỉ gửi khi detect/toggle (§5.2) |
| Cloud API bị rate-limit do worker concurrency 4–8 | Trường `max_concurrency` per-role trong config; UI gợi ý giảm về 1–2 khi dùng API tính phí |

## 9. Docs phải sync theo (quy định CLAUDE.md)
- `README.md` — bảng config + tính năng mới
- `CLAUDE.md` + `AGENTS.md` — kiến trúc RuntimeConfig, biến mới `SETTING_ENCRYPTION_KEY`
- `.env.example` — thêm `SETTING_ENCRYPTION_KEY` + chú thích DB override
- `docs/workers.md` — cơ chế version-check của worker
- Chạy `node .gitnexus/run.cjs analyze` sau khi đổi cấu trúc

## 10. Thứ tự triển khai đề xuất
1. **PR-1**: model + migration + `RuntimeConfigService` + refactor factory
   (`llm/__init__.py`, `graphiti_client.py`) — hành vi mặc định KHÔNG đổi khi
   bảng trống.
2. **PR-2**: API admin + audit + test-connection + cập nhật `/config/status`.
3. **PR-3**: worker `config_watch` + gắn vào caption/kg/memory worker.
4. **PR-4**: frontend `AdminLLMConfigPage`.
5. **PR-5** (tuỳ chọn): Redis pub/sub invalidation cho multi-process, SSE events.

## 11. Checklist kiểm thử thủ công (chưa có test suite)
- [ ] Bảng trống → mọi hành vi giống hệt hiện tại (regression)
- [ ] Factory sync: gọi `get_llm_provider()` từ thread (`asyncio.to_thread`) và LangGraph node sync OK — không deadlock, không cần await
- [ ] Override `main` qua API → chat trả lời bằng model mới, không restart
- [ ] Slot `vision` kế thừa `main` khi chưa override; model text-only → caption ảnh bị skip đúng như hiện tại
- [ ] Override `vision` = model riêng (qwen2.5-vl) → caption ảnh chạy bằng model mới trong khi chat vẫn dùng model main
- [ ] Caption worker nhận document mới → dùng model mới (log xác nhận)
- [ ] `HRAG_ENABLE_CONTEXTUAL_EMBEDDINGS=true` + đổi `memory_agent` → embed worker document mới dùng model mới
- [ ] Test connection với proxy CHẶN `/models` → fallback completion 1-token, trả `ok=true` + latency
- [ ] Nhập base_url + api_key DeepSeek → combobox Model tự đổ `deepseek-chat`/`deepseek-reasoner` sau <1s, không cần bấm test
- [ ] Đổi api_key → danh sách model tự load lại

---

## 12. V2 — Kiến trúc 2 tầng: Connections + Assignments

### 12.1 Động cơ
Thiết kế V1 bắt admin nhập base_url + api_key RIÊNG cho từng role — lặp lại
khi cùng 1 endpoint (VD DeepSeek) phục vụ nhiều nhiệm vụ với model khác nhau.
V2 tách: **Connection** (endpoint + key + danh sách model) khai báo MỘT lần;
**Assignment** per-role chỉ là chọn connection + chọn model từ danh sách đó.

### 12.2 Data model v2 (system_settings)
```
key = "llm_conn.<conn_id>"  → {name, provider, base_url, api_key_enc, extra}
key = "llm_role.<role>"     → {conn_id: "<id>" | "@env", model: str}
```
- `conn_id = "@env"` (default mọi role khi chưa phân công) → hành vi .env cũ.
- Xoá connection bị chặn (409) nếu còn role đang tham chiếu.

### 12.3 ROLES v2 (10 role)
main, vision, thinking, memory_agent, kg_extract, graphiti,
**stt, tts, embedding, rerank**

### 12.4 Đặc thù từng role mới
| Role | Cơ chế áp dụng | Cảnh báo UI |
|---|---|---|
| `stt` | Runtime: connection openai_compatible → `/audio/transcriptions`; faster-whisper local vẫn theo .env | — |
| `tts` | Runtime: connection → TTS endpoint (omnivoice/openai-compat) | — |
| `embedding` | **Áp dụng khi RESTART**: lifespan đọc override TRƯỚC `preload_models()` | 🔴 Đổi model = sai lệch dimension vector cũ → phải reindex |
| `rerank` | **Áp dụng khi RESTART** (như embedding) | 🟡 Nhẹ hơn — không ảnh hưởng vector, chỉ cần restart |

### 12.5 API v2
```
GET    /admin/llm-config                        → {roles{conn_id,model,source,resolved{}}, connections{}, version}
PUT    /admin/llm-config/connections/{conn_id}  → {name?, provider, base_url, api_key?, extra?}
DELETE /admin/llm-config/connections/{conn_id}  → 409 nếu còn role tham chiếu
PUT    /admin/llm-config/{role}                 → {conn_id: "<id>"|"@env", model?}
POST   /admin/llm-config/models                 → {conn_id} HOẶC {provider, base_url, api_key}
POST   /admin/llm-config/test                   → như V1 (raw config)
```

### 12.6 WebUI v2
- **Section "Kết nối Models"**: card mỗi connection (tên, provider, base_url,
  api_key masked, nút Load models → badge số model tìm thấy, Test).
- **Section "Phân công nhiệm vụ"**: mỗi dòng = 1 role → 2 dropdown
  [Kết nối (có lựa chọn "Theo .env")] + [Model] (auto-load từ connection đã
  chọn; đổi connection → reload danh sách model).
- Embedding/Rerank hiển thị banner cảnh báo reindex/restart.

### 12.7 Checklist bổ sung
- [ ] Tạo 1 connection DeepSeek → gán 3 role khác nhau 3 model khác nhau → chat/thinking/kg dùng đúng model tương ứng
- [ ] Xoá connection đang được tham chiếu → 409; gán lại role về @env rồi xoá → OK
- [ ] Override embedding + restart backend → preload_models dùng model mới + log warning reindex
- [ ] `DELETE` override → quay về `.env` model
- [ ] Test-connection với base_url sai → chặn lưu, báo lỗi rõ
- [ ] `WEB_CONCURRENCY=2` + REDIS_ENABLED → cả 2 process nhận config mới
- [ ] Audit log ghi đủ actor/action/before/after
- [ ] API key trong DB là ciphertext; response luôn masked

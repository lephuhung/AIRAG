# Đề xuất nâng cấp LangGraph Supervisor — Giảm cứng nhắc trong Routing & Decomposition

> Ngày: 2026-06-14 · Phạm vi: `backend/app/services/agents/supervisor.py` (2180 dòng), `models.py`, `prompts/agents/supervisor_prompt.py`, `query_analyzer_prompt.py`, `tools.py`
> Mục tiêu: chuyển từ **classify-cứng + route-thủ-công** sang **tool-aware planning** linh hoạt, ít regex hard-code, dễ bảo trì.

---

## 1. Tóm tắt điều hành

Hệ thống hiện routing qua **2 lần gọi LLM phân loại chồng nhau** (`query_analyzer` Qwen3-4B + `supervisor` Qwen 35B), sau đó điều phối bằng **6 hàm `route_from_*` thủ công** với hàng loạt **safety-net bằng regex** ghi đè quyết định của LLM. Đây là nguồn gốc cảm giác "cứng nhắc": taxonomy 19 intent + mapping tĩnh `intent → agent → search_mode → tool`, decomposition tuyến tính được lên kế hoạch trước (`current_step_index++`), retry bằng `_FALLBACK_MAP` cố định.

**Đòn bẩy lớn nhất**: các tool đã có sẵn dạng hàm sạch trong `tools.py`. Toàn bộ bộ máy intent-taxonomy + supervisor + routing thực chất là một bản **thay thế thủ công, dễ vỡ cho native tool-calling**. Chuyển sang **ReAct / tool-calling agent** sẽ xoá phần lớn sự cứng nhắc đó.

Khuyến nghị theo thứ tự ưu tiên (impact/risk):

| # | Nâng cấp | Tác động | Rủi ro |
|---|----------|----------|--------|
| P0 | Structured output cho supervisor + gỡ regex parse | Cao | Thấp |
| P0 | Dùng `Command(goto=, update=)` thay 6 hàm `route_from_*` | Cao | Thấp |
| P1 | Hợp nhất `query_analyzer` vào `supervisor` (1 LLM call → plan có cấu trúc) | Cao | Trung bình |
| P2 | **Tool-calling RAG agent** (`create_react_agent`) — bỏ `intent→search_mode` cứng | Rất cao | Trung bình |
| P3 | Planner động + chạy song song sub-query (`Send` API) | Cao | Trung bình |
| P4 | Re-planning bằng reflection thay `_FALLBACK_MAP` | Trung bình | Thấp |

---

## 2. Hiện trạng (đã verify trong code)

```
START → query_analyzer ──→ supervisor ──┬─(needs_memory)→ memory_recall → query_enricher ─┐
        (Qwen3-4B,        (Qwen 35B,    │                                                  ↓
         regex fast-path)  JSON string) └─(bypass)──────────────────────────────→ [rag|write|people|direct]
                                                                                            │
   rag → route_from_rag → result_evaluator → route_from_evaluator ─┬→ answer_generator → END
                                  ↑__________(supervisor_loop)______┘
   resolve_doc_agent → route_from_resolve_doc → [rag | answer_generator | END]
   people → mongo_formatter → END
```

- **12 node**, **6 hàm routing** (`route_from_supervisor/evaluator/rag/resolve_doc/enricher` + `route_from_evaluator`).
- **19 intent** (`models.py:22-50`), mapping tĩnh `INTENT_TO_AGENT` (`models.py:74-93`) ghi chú "không dùng trong runtime" — runtime hỏi LLM, càng cho thấy sự trùng lặp.
- `tools.py` đã có sẵn: `search_documents`, `list_documents`, `summarize_document`, `get_documents_content`, `query_knowledge_graph`, `search_documents_number`, `search_abbreviation`, `resolve_document_reference`, `search_document_section`, `search_people_by_*`.
- **Không dùng** `langgraph.types.Command`, `Send`, `langgraph.prebuilt.create_react_agent`, `with_structured_output`/`bind_tools`. `langgraph>=0.2.0` (pin cũ — nên nâng lên ≥0.6 để có các primitive này).

---

## 3. Chẩn đoán — 8 điểm cứng nhắc

### C1. Hai lần phân loại LLM chồng nhau, hai nguồn sự thật cho `task_plan`
`query_analyzer_node` (`supervisor.py:535`) gọi Qwen3-4B sinh `sub_queries`/`extracted_params`, **nhồi dưới dạng text** vào prompt supervisor (`supervisor.py:776-802`). `supervisor_node` rồi **tự suy ra lại** `task_plan` (`:843-988`). Hai nguồn này được hoà giải bằng code mệnh lệnh ghi đè (`:938-988`) → khó suy luận, dễ lệch.

### C2. Regex safety-net khắp nơi, đánh nhau với LLM
`_PERSONAL_REF_PATTERN`, `_NAMED_DOC_PATTERN`, `_MULTI_DOC_PATTERN`, `_COMPARISON_PATTERN`, `_is_likely_abbreviation`, override `direct→rag` (`:823-887`). Mỗi safety-net là một quy tắc cứng chèn lên quyết định của LLM. Đây chính là cảm giác "cứng nhắc".

### C3. Mapping tĩnh `intent → search_mode → tool` khoá chiến lược truy hồi
`:1028-1039`: `search/summarize/...→vector`, `kg_query→kg`. Chiến lược retrieval **bị khoá bởi nhãn intent**, không phải khám phá theo dữ liệu. Nhiều câu hỏi cần hybrid nhưng bị ép vector-only.

### C4. Decomposition cạn & lên kế hoạch trước, thực thi tuyến tính
`sub_queries` decompose một lần ở `query_analyzer`, rồi chạy tuần tự theo `current_step_index` (`result_evaluator` chỉ `++`). **Không re-plan**: nếu bước 1 cho thấy cần bước 2 khác thì không đổi được. **Không song song** dù các bước độc lập (vd "so sánh NĐ 13 và Luật ANM" có thể resolve 2 doc song song).

### C5. `query_analyzer` fast-path bằng regex bỏ sót câu phức tạp
`:560-578`: chỉ gọi LLM decompose nếu khớp `_MULTI_DOC_PATTERN`. Câu phức tạp không khớp regex → bị coi là "simple", bỏ qua decomposition hoàn toàn.

### C6. Routing trải trên 6 hàm với special-case & magic string
Cờ `should_loop_back`, `section_reference`, `pending_intent`, magic string `"supervisor_loop"` (`:1255`), hack "search_section loop không tăng iteration" (`:674-693`). State 30+ field chồng lấn → khó bảo trì.

### C7. Retry/fallback hard-code, không reflection
`_FALLBACK_MAP = {search→kg_query, kg_query→search}` (`:1260-1264`) chỉ đổi mode 1 lần, không suy luận **tại sao** rỗng (sai từ khoá? sai doc? cần mở rộng?).

### C8. Parse JSON thủ công dễ vỡ
`_parse_supervisor_response` (`:405`) tự cắt ```json, strip `<think>`, `json.loads` trong try/except. Khi parse fail → fallback RAG mù (`:1092-1104`).

---

## 4. Đề xuất nâng cấp

### P0 — Quick wins (ít rủi ro, làm trước)

**(a) Structured output thay regex-parse.** Dùng `with_structured_output(SupervisorDecision)` với Pydantic schema (next_agent, intent, task_plan, needs_memory, ...). Bỏ `_parse_supervisor_response` + strip `<think>` + try/except parse. LLM bị buộc trả đúng schema → hết lớp C8.

```python
class SupervisorDecision(BaseModel):
    next_agent: Literal["rag","write","people","direct","finish","resolve_doc"]
    intent: str
    task_plan: list[str]
    needs_memory: bool
    is_legal_query: bool
    reasoning: str
classifier.with_structured_output(SupervisorDecision)  # hết parse thủ công
```

**(b) `Command` thay các hàm `route_from_*`.** Node trả thẳng `Command(goto=..., update=...)`, gộp cập-nhật-state + điều-hướng vào một chỗ. Xoá `route_from_supervisor/evaluator/rag/resolve_doc/enricher`, xoá protocol magic-string `next_agent`/`"supervisor_loop"`.

```python
async def supervisor_node(state) -> Command:
    d = await classify(state)
    return Command(goto=_target(d), update={"intent": d.intent, "task_plan": d.task_plan})
```

**(c) Gỡ bớt regex safety-net.** Sau khi (a) ổn định, các safety-net C2 chủ yếu vá lỗi parse/LLM yếu — chuyển phần còn cần thiết vào *prompt* (đã có ví dụ tốt) thay vì code mệnh lệnh. Giữ lại tối đa 1-2 net thật sự cần (vd `needs_memory` từ đại từ nhân xưng) và đo lại bằng Langfuse.

### P1 — Hợp nhất `query_analyzer` ⟶ `supervisor` (1 LLM call)

Cho supervisor (model 35B đã mạnh) trả **luôn** plan có cấu trúc trong cùng schema P0(a): `sub_queries`, `extracted_params`, `complexity`. Xoá node `query_analyzer` + regex fast-path (C1, C5). Một nguồn sự thật, một LLM call, không hoà giải. Nếu lo latency: chỉ thêm field plan khi `complexity != simple` (LLM tự quyết trong cùng lần gọi).

### P2 — Tool-calling RAG agent (đòn bẩy chính, trị C3)

Thay nhánh `rag` (intent→search_mode→tool tĩnh) bằng **ReAct agent** dùng `create_react_agent`, expose các hàm `tools.py` đã có làm `@tool`:

```python
from langgraph.prebuilt import create_react_agent
rag_tools = [search_documents, resolve_document_reference, search_document_section,
             query_knowledge_graph, list_documents, search_abbreviation, search_documents_number]
rag_agent = create_react_agent(model, rag_tools, prompt=RAG_SYSTEM)
```

LLM **tự chọn** tool, tự gọi nhiều tool, tự lặp đến khi đủ ngữ cảnh → bỏ taxonomy intent cho nhóm RAG, bỏ `intent→search_mode`, bỏ `_FALLBACK_MAP`, bỏ chuỗi prerequisite thủ công (`resolve_doc` trước `search_section`) vì model gọi `resolve_document_reference` rồi `search_document_section` một cách tự nhiên. Supervisor thu về vai trò **phân nhánh thô**: `rag | write | people | direct` (4 nhánh thay vì 19 intent).

> Lưu ý: cần model hỗ trợ tool-calling tốt. Qwen 35B/3.6 đủ; kiểm tra provider (`OllamaLLMProvider`/`OpenAICompatibleLLMProvider`) bật `tools=`/`tool_choice`. Có thể giữ intent-routing cho `write`/`people` (vốn rõ ràng) và **chỉ** ReAct-hoá nhóm RAG để giảm rủi ro.

### P3 — Planner động + sub-query song song (`Send` API, trị C4)

Cho supervisor sinh **DAG** sub-task (mỗi task có `depends_on`), dùng `Send` để fan-out các task độc lập chạy song song, rồi map-reduce kết quả:

```python
from langgraph.types import Send
def fan_out(state):
    ready = [t for t in state["plan"] if deps_done(t, state)]
    return [Send("rag_worker", {"subquery": t.query, **t.params}) for t in ready]
```

`accumulated_results` đã là `Annotated[list, operator.add]` (`models.py:191`) — hợp với reducer map-reduce. Sau mỗi vòng, re-evaluate plan (re-plan) thay vì `current_step_index++` cứng. Trị thẳng yêu cầu của user: "phân tách câu hỏi để routing" → giờ là phân tách thành DAG chạy song song + định tuyến động.

### P4 — Re-planning bằng reflection (trị C7)

`result_evaluator` thay `_FALLBACK_MAP` tĩnh bằng 1 LLM call ngắn: "kết quả này đã đủ trả lời sub-query chưa? nếu chưa, đề xuất hành động kế tiếp (đổi từ khoá / đổi tool / mở rộng / dừng)". Trả structured `{sufficient: bool, next_action: ...}`. Linh hoạt hơn việc chỉ swap vector↔kg một lần.

---

## 5. So sánh trước/sau

| Khía cạnh | Hiện tại | Sau nâng cấp |
|-----------|----------|--------------|
| Số LLM call phân loại | 2 (analyzer + supervisor) | 1 (supervisor structured) |
| Chọn tool/retrieval | intent→search_mode tĩnh | LLM tool-calling động |
| Decomposition | tuyến tính, plan trước | DAG, song song, re-plan |
| Routing | 6 hàm + magic string + flags | `Command(goto=)` tại node |
| Parse quyết định | regex cắt JSON + strip think | structured output (schema) |
| Safety-net | nhiều regex ghi đè LLM | tối thiểu, đẩy vào prompt |
| Retry | `_FALLBACK_MAP` cố định | reflection LLM |
| Số nhánh supervisor | 19 intent | 4 (rag/write/people/direct) |

---

## 6. Migration path (an toàn, từng bước)

1. **Nâng `langgraph` ≥0.6** trong `requirements.txt` (mở khoá `Command`, `Send`, `create_react_agent`). Smoke-test graph hiện tại.
2. P0(a)+P0(b) — structured output + `Command`. Không đổi hành vi routing, chỉ đổi cơ chế → so sánh Langfuse trace trước/sau.
3. P1 — gộp analyzer. Bật cờ env `NEXUSRAG_LG_UNIFIED_SUPERVISOR` để A/B.
4. P2 — ReAct **chỉ cho nhóm RAG**, sau cờ `NEXUSRAG_LG_RAG_REACT`. Giữ nhánh write/people/direct nguyên.
5. P3 + P4 — sau khi P2 ổn định và có metric.

Mỗi phase độc lập, có cờ env, rollback dễ. Dùng Langfuse (đã tích hợp) làm thước đo: tỉ lệ route đúng, số iteration, latency, độ rỗng kết quả.

---

## 7. Khuyến nghị

Bắt đầu **P0 + P2** vì cho tỉ lệ "giảm cứng nhắc / công sức" cao nhất: P0 dọn nền (structured + `Command`), P2 (ReAct cho RAG) xoá phần lớn taxonomy intent + routing tĩnh — đúng hai điều user nêu. P1/P3/P4 làm sau khi có baseline metric.

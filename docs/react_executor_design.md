# Thiết kế ReAct Executor cho nhóm RAG — LLM tự gọi tool & tự quyết lưu memory

> Ngày: 2026-06-14 · Quyết định đã chốt: **vLLM OpenAI-compatible (native tool-calling)**, **chỉ ReAct-hoá nhóm RAG ở giai đoạn đầu** (giữ write/people/direct như nhánh cũ).
> Mục tiêu: thay nhánh `rag` (intent→search_mode→tool tĩnh + `_FALLBACK_MAP` + chuỗi prerequisite thủ công) bằng **một vòng lặp tool-calling** để LLM tự quyết gọi gì, gọi mấy lần, song song hay không, và lưu gì vào memory — **không regex**.

---

## 0. Nguyên tắc

1. **Supervisor chỉ phân nhánh thô** `rag | write | people | direct`. Khi `rag` → vào `react_executor`. Không còn 19 intent / `search_mode` tĩnh cho nhóm RAG.
2. **Mọi quyết định "gọi gì / lưu gì" là tool-call của LLM**, không phải regex/heuristic.
3. **Tool đã có sẵn** trong `tools.py` — chỉ cần bọc thành tool LLM-facing.
4. **Không phá** write/people/direct ở phase này. Bật/tắt qua cờ env để A/B + rollback.

---

## 1. Phạm vi & vị trí trong graph hiện tại

Giữ nguyên graph, chỉ thay node `rag`:

```
START → supervisor ──┬─ rag      → react_executor → END    ← THAY ĐỔI (1 node, tự lặp)
                     ├─ write    → write_agent    → END    (giữ nguyên)
                     ├─ people   → mongo_formatter→ END    (giữ nguyên)
                     └─ direct   → direct         → END    (giữ nguyên)
```

Các node/khái niệm **bị loại khỏi nhánh RAG** (vẫn để lại cho phase sau nếu cần): `query_analyzer`, `result_evaluator`, `route_from_rag`, `route_from_resolve_doc`, `route_from_evaluator`, `_FALLBACK_MAP`, `sub_queries`, `current_step_index`, `search_mode`, `pending_intent`, `section_reference`, `should_loop_back`. `react_executor` thay thế toàn bộ chuỗi `rag → result_evaluator → answer_generator` cho nhóm RAG.

> Supervisor vẫn có thể giữ `needs_memory` như **gợi ý** (đưa vào prompt), nhưng quyết định recall cuối cùng do executor (LLM gọi `recall_memory`). Có thể bỏ regex `_PERSONAL_REF_PATTERN` sau khi đo.

---

## 2. Tool registry hai tầng (điểm thiết kế quan trọng)

Các hàm trong `tools.py` nhận **cả tham số ngữ nghĩa lẫn tham số ngữ cảnh runtime**:

```python
search_documents(query, top_k, workspace_ids, existing_citation_ids, db, document_ids, search_mode, scoped_to_documents)
resolve_document_reference(reference, workspace_ids, db)
search_document_section(section_reference, workspace_ids, document_ids)
query_knowledge_graph(entity, workspace_ids, db)
list_documents(...) / search_abbreviation(abbreviation, workspace_ids, db) / search_documents_number(query, workspace_ids, db)
```

**LLM chỉ được thấy tham số ngữ nghĩa**; `db`, `workspace_ids`, `existing_citation_ids`, `document_ids` do executor **tự tiêm** từ state. Vì vậy mỗi tool có 2 phần:

```python
# (a) Schema LLM-facing (OpenAI tools format) — chỉ arg ngữ nghĩa
{
  "type": "function",
  "function": {
    "name": "search_documents",
    "description": "Tìm nội dung trong kho văn bản. Dùng khi câu hỏi là khái niệm/chủ đề "
                   "chung hoặc đã biết văn bản cần tìm. mode='vector' cho trích/tóm tắt, "
                   "'kg' cho quan hệ thực thể, 'hybrid' khi không chắc.",
    "parameters": {"type":"object","properties":{
       "query":{"type":"string"},
       "mode":{"type":"string","enum":["vector","kg","hybrid"],"default":"hybrid"}
    },"required":["query"]}
  }
}

# (b) Adapter: LLM args + context state → gọi hàm thật
async def _call_search_documents(args, ctx):
    return await search_documents(
        query=args["query"], search_mode=args.get("mode","hybrid"),
        top_k=ctx["top_k"], workspace_ids=ctx["workspace_ids"],
        existing_citation_ids=ctx["citation_ids"], db=ctx["db"],
        document_ids=ctx.get("document_ids"),
        scoped_to_documents=bool(ctx.get("document_ids")),
    )

TOOL_REGISTRY = {"search_documents": _call_search_documents, ...}
```

Mỗi `description`/`enum` chính là nơi đặt **guardrail domain thay cho regex** (vd: khi nào dùng `resolve_document_reference` vs `search_documents`, định nghĩa "named doc"). Đây là cách bỏ `_NAMED_DOC_PATTERN`, `_MULTI_DOC_PATTERN`, `_COMPARISON_PATTERN`.

### Tool mới
| Tool | Bọc cái gì | Mục đích |
|------|-----------|----------|
| `recall_memory(query)` | `_format_memory_context` / Graphiti search | LLM tự quyết khi nào cần ngữ cảnh cá nhân (thay `needs_memory` regex) |
| `save_memory(fact, kind)` | `add_episode` (graphiti_client) | **LLM tự quyết lưu gì** trong hội thoại (vd user: "nhớ giúp tôi…") |
| `ask_user(question, options?)` | push SSE `clarification` + dừng loop | Hỏi lại khi mơ hồ (thay disambiguation regex) |

---

## 3. Vòng lặp executor

```python
async def react_executor_node(state) -> dict:
    ctx = build_ctx(state)                  # db, workspace_ids, document_ids, citation set, top_k
    msgs = [LLMMessage("system", REACT_SYSTEM_PROMPT),
            LLMMessage("user", state["original_query"])]
    sources, images = [], []

    for step in range(MAX_TOOL_STEPS):      # loop guard (vd 6)
        calls, answer_text = [], ""
        async for c in llm.astream(msgs, tools=TOOL_SCHEMAS, think=enable_thinking):
            if c.type == "function_call":
                calls.append(c.function_call)            # {"name","args"}
            elif c.type == "thinking":
                await push_event(state, "status", {"step":"thinking","detail":c.text})
            elif c.type == "text":
                answer_text += c.text

        if not calls:                        # KHÔNG gọi tool nữa ⇒ đây là câu trả lời cuối
            await stream_tokens(state, answer_text)      # stream ra UI
            return {"final_answer": answer_text, "sources": sources,
                    "images": images, "next_agent": "finish"}

        # ── Phân tách song song: nhiều tool_call trong 1 turn → chạy đồng thời ──
        await push_event(state, "status", {"step":"searching",
            "detail": _describe(calls)})                  # "Đang tra cứu N nguồn..."
        results = await asyncio.gather(
            *[_safe_call(c, ctx) for c in calls], return_exceptions=True)

        # Ghi kết quả tool vào hội thoại + gom citation
        msgs.append(assistant_tool_call_msg(calls, answer_text))
        for c, r in zip(calls, results):
            if isinstance(r, Exception):
                msgs.append(tool_result_msg(c, f"ERROR: {r}"))  # LLM tự xử lý lỗi/đổi cách
                continue
            sources += r.get("sources", []); images += r.get("images", [])
            msgs.append(tool_result_msg(c, _compact_for_llm(r)))  # tránh phình context

        if c_name_in(calls, "ask_user"):     # ask_user đã push clarification → dừng
            return {"final_answer": "", "next_agent":"finish", "clarification_needed":True}

    # Hết bước → tổng hợp lần cuối từ những gì đã có (fail-safe, không "RAG mù")
    return await _force_synthesize(state, msgs, sources, images)
```

Đặc tính chính:
- **Decomposition là emergent**: LLM phát nhiều tool_call → `asyncio.gather` chạy song song ("so sánh NĐ 13 và Luật ANM" → 2 `resolve_document_reference` đồng thời). Không cần planner/`sub_queries`.
- **Prerequisite tự nhiên**: LLM gọi `resolve_document_reference` rồi mới `search_document_section`/`search_documents` với `document_ids` trả về — không cần task_plan.
- **Retry là reflection**: tool trả rỗng → kết quả "0 nguồn" vào hội thoại → LLM tự đổi từ khoá/đổi tool. Thay `_FALLBACK_MAP`.
- **Lỗi tool** trả về cho LLM dưới dạng text → tự phục hồi, không crash sang "RAG mù".

---

## 4. "Lưu nội dung gì" — 3 tầng

| Tầng | Lưu ở đâu | Ai quyết | Cơ chế |
|------|-----------|----------|--------|
| **Working/scratchpad** | `msgs` trong vòng lặp | executor | Kết quả tool nối vào hội thoại; `_compact_for_llm` cắt gọn (giữ text + id nguồn, bỏ payload thừa). Nếu `msgs` quá dài → 1 bước LLM tóm tắt scratchpad. |
| **Evidence/citation** | `sources`, `images` | executor | Mỗi tool trả `sources`; gom để render trích dẫn (giữ cơ chế citation hiện có). |
| **Long-term (Graphiti)** | episode theo `user_{id}` | **LLM** + lưới an toàn | (a) trong loop: LLM gọi `save_memory(fact)` khi nhận ra fact bền; (b) post-turn: giữ `add_conversation_episode`/`_llm_extract_facts` làm lưới an toàn cho fact ngầm. |

Như vậy "lưu gì" hoàn toàn do LLM, **bỏ heuristic `len<10`/skip-question** — logic ấy đã nằm trong prompt của `_llm_extract_facts` và trong quyết định gọi `save_memory`.

---

## 5. Native tool-calling qua vLLM — việc cần làm ở provider

`openai_compatible.astream` đã `kwargs["tools"]=tools` và parse `delta.tool_calls` (`openai_compatible.py:227`). **Nhưng có 2 vấn đề phải sửa**:

1. **Gộp delta của tool_calls theo `index`.** Hiện code `json.loads(tc.function.arguments)` trên **từng delta** (`:233`). OpenAI/vLLM stream arguments thành **nhiều mảnh**; phải buffer theo `tc.index` rồi `json.loads` **một lần khi stream kết thúc** — nếu không, args dài (vd `query` tiếng Việt) sẽ vỡ.
   ```python
   acc = {}  # index -> {"name":..., "args_str":...}
   ...
   for tc in delta.tool_calls:
       e = acc.setdefault(tc.index, {"name":"", "args_str":""})
       if tc.function.name:      e["name"] = tc.function.name
       if tc.function.arguments: e["args_str"] += tc.function.arguments
   # cuối stream:
   for e in acc.values():
       yield StreamChunk(type="function_call",
                         function_call={"name":e["name"], "args":json.loads(e["args_str"] or "{}")})
   ```
2. **`tool_choice`/`parallel_tool_calls`**: cho phép truyền `tool_choice="auto"` và để vLLM bật parallel tool calls (tuỳ model). Thêm tham số mượt vào `astream`.

Yêu cầu vận hành: model phục vụ qua vLLM phải bật template tool-calling (Qwen2.5/Qwen3 hỗ trợ). Kiểm tra `OLLAMA_HOST` chứa `/v1` → đã đi đường `OpenAICompatibleLLMProvider` (supervisor.py:761).

---

## 6. Streaming / SSE

- `thinking` chunk giữa các bước → `push_event(status)` (UI thấy tiến trình "đang tra cứu…").
- Chỉ **stream token của câu trả lời cuối** (turn không có tool_call). Trong turn có tool_call, text đi kèm là "reasoning" → không stream như answer.
- Tool đang chạy → emit `status` mô tả (giữ trải nghiệm hiện tại của `result_evaluator`/`answer_generator`).

---

## 7. State (gọn lại cho nhánh RAG)

Có thể bỏ khỏi đường RAG: `sub_queries`, `extracted_params`, `query_complexity`, `current_step_index`, `accumulated_results`, `retry_count`, `retry_strategy`, `search_mode`, `pending_intent`, `section_reference`, `should_loop_back`, `task_plan`. Giữ: `messages`, `original_query`, `workspace_ids`, `document_ids`, `user_id`, `sources`, `images`, `final_answer`, `next_agent`, `user_memory_context`. (Chưa xoá vội — để lại tới khi write/people cũng chuyển; tránh va chạm các nhánh khác.)

---

## 8. Bỏ regex — bảng ánh xạ

| Regex/heuristic hiện tại (vị trí) | Thay bằng |
|---|---|
| `_NAMED_DOC_PATTERN`, prerequisite inject (supervisor.py:843-865) | LLM gọi `resolve_document_reference`; tool trả `ambiguous`+candidates |
| `_PERSONAL_REF_PATTERN` → `needs_memory` (:823) | LLM gọi `recall_memory` (giữ flag làm gợi ý trong prompt) |
| `_MULTI_DOC_PATTERN` fast-path (:560-578) | nhiều tool_call song song |
| `_COMPARISON_PATTERN`+`needs_comparison` (:563-568) | LLM resolve nhiều doc rồi so sánh trong câu trả lời |
| disambiguation regex (clarification.py) | tool `ask_user` |
| `_FALLBACK_MAP` (:1260) | reflection trong loop |
| `_parse_supervisor_response` JSON cắt tay (:405) | (phase trước) structured output |

---

## 9. Migration (sau cờ env, A/B được)

- `NEXUSRAG_LG_RAG_REACT=false` (default) → giữ nhánh `rag` cũ.
- `=true` → supervisor route `rag` vào `react_executor`.
- `NEXUSRAG_REACT_MAX_TOOL_STEPS=6`, `NEXUSRAG_REACT_TOP_K=...`.

Thứ tự:
1. **Sửa provider** (mục 5: gộp delta tool_calls + tool_choice). Có test riêng: gọi 1 prompt buộc 2 tool song song, assert nhận đủ 2 `function_call` với args hợp lệ.
2. **Tool registry 2 tầng** (schemas + adapters) cho nhóm RAG + `recall_memory`/`save_memory`/`ask_user`.
3. **`react_executor_node`** + nối vào `route_from_supervisor` sau cờ.
4. **A/B bằng Langfuse**: so tỉ lệ trả lời đúng, số bước, latency, độ rỗng, số lần ask_user, citation coverage giữa nhánh cũ và ReAct trên cùng bộ câu hỏi.
5. Khi ổn → bỏ dần regex safety-net (mục 8) và rút gọn state (mục 7).

---

## 10. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|--------|-----------|
| Model gọi tool sai/lặp | docstring rõ + `MAX_TOOL_STEPS` + dedup (cùng name+args trong 1 loop → bỏ) |
| Vỡ args streaming (mục 5.1) | sửa provider + unit test trước khi build loop |
| Phình context khi nhiều tool | `_compact_for_llm` + bước tóm tắt scratchpad |
| Latency tăng do nhiều vòng | parallel tool calls + giảm bước nhờ prompt tốt; đo bằng Langfuse |
| Tool-calling Qwen chưa ổn định | đã chọn vLLM native; kiểm tra template + benchmark bộ prompt thực tế |
| Mất kiểm soát domain (luật VN) | guardrail trong system prompt + description/enum của tool, validation tầng tool (ambiguous → ask_user) |

---

## 11. Kiểm thử (không có test suite chính thức → dùng /docs + script)

- Bộ ~20 câu đại diện: khái niệm chung, doc có tên, điều/khoản, so sánh 2 doc, viết tắt, kg, câu có "tôi/đơn vị tôi", câu mơ hồ (kỳ vọng `ask_user`), câu chứa fact cá nhân (kỳ vọng `save_memory`).
- So nhánh cũ vs ReAct trên cùng bộ, chấm bằng Langfuse trace + đọc tay.
```

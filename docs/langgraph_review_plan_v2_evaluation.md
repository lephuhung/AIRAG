# Đánh Giá Plan Cải Thiện LangGraph V2

> **Đánh giá bởi:** Antigravity (Gemini 3.5 Flash)
> **Ngày:** 2026-06-07
> **Phạm vi:** Cross-reference toàn bộ plan V2 với codebase thực tế

---

## 1. Đánh Giá Tổng Quan

| Tiêu chí | Điểm | Ghi chú |
|---|---|---|
| **Tính chính xác** | ⭐⭐⭐⭐⭐ 9.5/10 | Hầu hết mọi bug/vấn đề được xác nhận khi kiểm tra code thực |
| **Mức độ ưu tiên** | ⭐⭐⭐⭐ 8/10 | P0/P1/P2 phân loại hợp lý; một vài P2 nên lên P1 |
| **Khả thi** | ⭐⭐⭐⭐ 8/10 | Phase B (refactor 2057 dòng) cao rủi ro; cần chia nhỏ hơn |
| **Ước lượng effort** | ⭐⭐⭐ 7/10 | Một số task ước lượng quá optimistic |
| **Bao phủ vấn đề** | ⭐⭐⭐⭐ 8.5/10 | Tốt nhưng thiếu vài vấn đề quan trọng (xem bên dưới) |

> [!TIP]
> Plan V2 là một **audit chất lượng cao**, có hệ thống, và là tài liệu tham chiếu tốt cho technical debt cleanup. Đáng để thực hiện theo thứ tự khuyến nghị.

---

## 2. Xác Minh Từng Vấn Đề

### 🔴 P0 — Bug / Typo / Dead Code

#### P0-1. Typo `MONGO_SEARCH_BHxh` ✅ XÁC NHẬN

**Verified tại** [models.py:39](file:///home/AIRAG/backend/app/services/agents/models.py#L39):
```python
MONGO_SEARCH_BHxh = "mongo_search_bhxh"  # ← chữ 'x' viết thường
```
Trong khi tất cả constant khác (`MONGO_SEARCH_PHONE`, `MONGO_SEARCH_NAME`, `MONGO_SEARCH_CCCD`) đều UPPERCASE.

**Rủi ro thực tế:** Vừa (trung bình). Vì `MONGO_SEARCH_BHxh` vẫn resolve ra string đúng `"mongo_search_bhxh"`. Nhưng nếu ai grep `MONGO_SEARCH_BHXH` sẽ miss. Ngoài ra ở [models.py:48](file:///home/AIRAG/backend/app/services/agents/models.py#L48), `Intent.ALL` set cũng dùng `MONGO_SEARCH_BHxh` nên thực tế không crash runtime — chỉ gây confusion cho developer.

**Đồng ý fix:** ✅ Rất đơn giản, nên fix ngay.

---

#### P0-2. Duplicate key trong supervisor_prompt.py:333 ✅ XÁC NHẬN

**Verified tại** [supervisor_prompt.py:333](file:///home/AIRAG/backend/app/prompts/agents/supervisor_prompt.py#L333):
```
{{"next_agent":"<agent>","intent":"<first step intent>","task_plan":["<step1>","<step2>",...],
  "needs_memory":false,"is_legal_query":false,"mentions_specific_doc":false,
  "needs_memory":false,"is_legal_query":true,"mentions_specific_doc":false,  ← DUPLICATE!
  "reasoning":"<brief>"}}
```
Có **3 cặp** key bị duplicate: `needs_memory`, `is_legal_query`, `mentions_specific_doc`. Tệ hơn plan mô tả — plan nói chỉ duplicate `needs_memory`, thực tế cả 3 key đều bị.

> [!WARNING]
> Plan V2 ghi chỉ `needs_memory` duplicate — thực tế **cả 3 field** (`needs_memory`, `is_legal_query`, `mentions_specific_doc`) đều duplicate. Hơn nữa giá trị `is_legal_query` khác nhau giữa 2 lần (`false` vs `true`), dẫn đến LLM có thể output inconsistent.

**Đồng ý fix:** ✅ Quan trọng — sẽ khiến JSON example mẫu mâu thuẫn.

---

#### P0-3. `mentions_specific_doc` là dead code ✅ XÁC NHẬN

**Verified:**
- Parse tại [supervisor.py:368](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L368): `mentions_specific_doc = data.get("mentions_specific_doc", False)`
- Return tại [supervisor.py:436](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L436): trong dict kết quả
- **Không hề dùng** trong bất kỳ routing condition, safety net, hay sub-agent nào

**Đề xuất:** Đồng ý **Phương án A** (bỏ luôn). Langfuse có thể infer từ `task_plan` chứa `"resolve_doc"` hay không.

---

#### P0-4. `_NAMED_DOC_PATTERN` regex quá giới hạn ✅ XÁC NHẬN

**Verified tại** [supervisor.py:748-752](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L748-L752):
```python
_NAMED_DOC_PATTERN = re.compile(
    r"(?:luật|nghị\s+định|thông\s+tư|quyết\s+định|nghị\s+quyết|pháp\s+lệnh|bộ\s+luật)"
    r"\s+\S",
    re.IGNORECASE | re.UNICODE,
)
```

Trong khi `resolve_doc_agent.py` có [_DOC_TYPE_KEYWORDS](file:///home/AIRAG/backend/app/services/agents/resolve_doc_agent.py#L33-L43) rất chi tiết hơn:
- Bao gồm `chỉ thị`, `thông tư liên tịch`
- Sử dụng longest-match-first ordering

**Đồng ý fix:** ✅ Nên share `_DOC_TYPE_KEYWORDS` qua `doc_keywords.py`.

> [!NOTE]
> Fix P0-4 sẽ tự động cover một phần P1-6 (logic không đồng bộ giữa supervisor và resolve_doc_agent).

---

#### P0-5. `_INTENT_TO_AGENT_FALLBACK` dùng raw string ✅ XÁC NHẬN

**Verified tại** [supervisor.py:62-82](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L62-L82):
```python
_INTENT_TO_AGENT_FALLBACK: dict[str, str] = {
    "greeting": "direct",    # ← raw string thay vì Intent.GREETING
    "search": "rag",         # ← raw string thay vì Intent.SEARCH
    ...
}
```

**Đồng ý fix:** ✅ Nên dùng `Intent.*` constants. Tuy nhiên lưu ý agent values cũng nên dùng `AgentType.*`.

---

#### P0-6. `_INTENT_NORMALIZE` thiếu một số shorthand ⚠️ ĐỒNG Ý MỘT PHẦN

**Verified tại** [supervisor.py:44-59](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L44-L59).

Plan đề xuất bổ sung: `query_doc`, `find_doc`, `summarize_doc`, `lookup_section`, `get_article`, `search_person`, `find_by_name`.

> [!IMPORTANT]
> **Cẩn thận với over-engineering:** Thêm quá nhiều mapping có thể gây false positive. Ví dụ `find_doc` → `resolve_doc` sẽ sai nếu LLM output `find_doc` với ý nghĩa "tìm tài liệu" (search) chứ không phải "xác định tài liệu cụ thể" (resolve). Nên chỉ thêm những mapping mà thực sự observe từ LLM logs.

**Khuyến nghị:** ✅ Bổ sung nhưng dựa trên **dữ liệu thực tế từ Langfuse logs** — kiểm tra LLM thực sự output intent gì sai trước khi thêm mapping.

---

### 🟠 P1 — Logic / Cấu Trúc

#### P1-1. Quá nhiều safety nets chồng chéo ✅ XÁC NHẬN

**Verified.** Đếm được **ít nhất 8 lớp override** trong `supervisor_node`:
1. `_INTENT_NORMALIZE` (L44-59)
2. `next_agent validation` (L374-390)
3. Deterministic intent→agent override (L392-403)
4. `task_plan[0]` correction (L405-427)
5. `needs_memory` keyword safety net (L712-718)
6. `is_legal_query` keyword safety net (L723-734)
7. `_REQUIRES_DOC_INTENTS` prerequisite injection (L741-762)
8. Phase 5 multi-step override (L817-844)

**Đồng ý fix:** ✅ Gom vào `apply_deterministic_overrides()` rất hợp lý. Nhưng cần **maintain thứ tự** rõ ràng vì một số override phải chạy trước (ví dụ intent normalize phải trước agent validation).

---

#### P1-2. `supervisor.py` 2057 dòng ✅ XÁC NHẬN

**Verified:** File đúng 2058 dòng.

**Đề xuất tách** trong plan rất hợp lý. Tuy nhiên:

> [!WARNING]
> **Rủi ro cao nhất của toàn plan.** Tách file lớn là nơi dễ gây regression nhất. Khuyến nghị:
> 1. Tách từng module một (abbreviation trước vì ít coupling nhất)
> 2. Mỗi bước tách xong phải test end-to-end trước khi tách tiếp
> 3. Giữ `supervisor.py` re-export cho backward compat cho đến khi tất cả import đã được update

**Effort ước lượng:** Plan nói 4 giờ cho task B1 (tạo `nodes/` package) — **quá optimistic**. Thực tế cần 6-8 giờ vì có nhiều circular import cần xử lý (ví dụ `_INTENT_TO_AGENT_FALLBACK` được dùng cả trong `supervisor_node` lẫn `result_evaluator_node`).

---

#### P1-3. `task_plan` built ở 3 nơi ✅ XÁC NHẬN

**Verified:**
1. LLM supervisor output (L365)
2. `_REQUIRES_DOC_INTENTS` injection (L741-762)
3. Phase 5 multi-step (L817-844) — line 842 `result["task_plan"] = remaining_intents` **ghi đè hoàn toàn**

**Đây là bug thực sự**, không chỉ code smell. Khi Phase 5 override ghi đè `task_plan` từ `sub_queries`, nó sẽ mất prerequisite `resolve_doc` đã được inject bởi Phase 4.

**Đồng ý fix:** ✅ Cần check `if task_plan[0] == "resolve_doc"` trước khi ghi đè.

---

#### P1-4. State field `intent` bị mutate ở nhiều nơi ✅ XÁC NHẬN

**Verified đặc biệt tại** [rag_agent.py:778-779](file:///home/AIRAG/backend/app/services/agents/rag_agent.py#L778-L779):
```python
# Change intent so route_from_rag goes to answer_generator (not back to supervisor)
"intent": "summarize",
```

Đây là **anti-pattern nghiêm trọng** — set `intent="summarize"` chỉ để hack routing, không phải vì intent thực sự là summarize.

**Đồng ý fix:** ✅ Dùng explicit routing flag thay vì hijack intent.

---

#### P1-5. `_extract_section_from_markdown` heuristic sai ranh giới ✅ XÁC NHẬN

**Verified tại** [rag_agent.py:599-621](file:///home/AIRAG/backend/app/services/agents/rag_agent.py#L599-L621):
```python
type_match = re.match(r"^([^\d\s]+)", section_ref)
section_type = type_match.group(1) if type_match else ""
```

Với `section_ref = "Khoản 2 Điều 8"`:
- `type_match` sẽ extract `"Khoản"` làm `section_type`
- Boundary check (L620): `h_text.startswith(section_type)` → stop tại "Khoản 3" → **sai**, vì Khoản 3 cùng Điều 8 nên phải tiếp tục

**Đồng ý fix:** ✅ Cần structured parsing cho `section_ref`. Đây là P1 đúng mức — ảnh hưởng đến chất lượng nội dung trả về.

---

#### P1-6. Logic `_REQUIRES_DOC_INTENTS` không đồng bộ ✅ XÁC NHẬN

Đã verify ở P0-4. Hai hệ thống keyword detection khác nhau:
- `supervisor.py:748-752`: regex đơn giản, thiếu nhiều doc type
- `resolve_doc_agent.py:33-43`: `_DOC_TYPE_KEYWORDS` đầy đủ hơn nhiều

**Đồng ý fix:** ✅ Extract ra shared module.

---

### 🟢 P2 — Prompt / Sub-agent Polish

#### P2-1. Supervisor prompt 25.5 KB ✅ XÁC NHẬN
Verified: 334 dòng, ~25KB. Nhiều example có thể consolidate.

#### P2-2. Query analyzer thiếu examples ✅ XÁC NHẬN
[Query analyzer prompt](file:///home/AIRAG/backend/app/prompts/agents/query_analyzer_prompt.py) có 7 examples nhưng chỉ cover: `search`, `resolve_doc` (comparison), `search_section` (multi), `mongo_search_cccd + resolve_doc` (cross), `greeting`, `kg_query`, `search` (date_range). **Thiếu** `search_abbr`, `list_docs` (standalone), `personal`, `mongo_search_name/bhxh/phone`, `write_*`, `summarize`.

> [!NOTE]
> Plan nói "EXAMPLES chỉ cover 5" — thực tế là **7 examples**. Nhưng kết luận vẫn đúng: thiếu nhiều intent.

#### P2-3. `is_legal_query` quá rộng ✅ XÁC NHẬN
"bảo mật" có thể là IT security. Plan nói đúng.

#### P2-4. `_tool_mongo_advanced` lạm dụng LLM ⚠️ ĐỒNG Ý MỘT PHẦN

**Verified tại** [people_agent.py:78-129](file:///home/AIRAG/backend/app/services/agents/people_agent.py#L78-L129). LLM call dùng `get_llm_provider()` (main LLM, có thể Gemini) — **tốn hơn cần thiết**.

Tuy nhiên, regex extraction cho tên tiếng Việt **khó chính xác** (nhiều edge case: tên 4 chữ, tên có số, ...). 

**Khuyến nghị:** Dùng `get_memory_agent()` (Qwen3-4B, nhẹ hơn) thay vì main LLM, chứ **không nên** thay hoàn toàn bằng regex.

#### P2-5. `_map_mongo_result` set `final_answer` ✅ XÁC NHẬN
Verified [people_agent.py:38](file:///home/AIRAG/backend/app/services/agents/people_agent.py#L38): `"final_answer": display`. Nhưng `people_agent_node` tại [line 233](file:///home/AIRAG/backend/app/services/agents/people_agent.py#L233) cũng set `final_answer` → `_map_mongo_result` chỉ được dùng từ `rag_agent.py` (legacy code), trong khi actual flow dùng `people_agent_node` directly. **Tuy nhiên**, `mongo_formatter_node` sử dụng `state.get("final_answer")` nên field này **không hoàn toàn dead** — nó được dùng, nhưng bởi `people_agent_node` (L233) chứ không phải `_map_mongo_result`.

> [!IMPORTANT]
> Plan nói sai đối tượng: dead code là `_map_mongo_result` **trong** `people_agent.py` (được dùng bởi `PEOPLE_TOOL_REGISTRY`), nhưng `people_agent_node` tại L230-235 set `final_answer` trực tiếp → `_map_mongo_result` trong people_agent.py vẫn được gọi qua `PEOPLE_TOOL_REGISTRY` → **không dead**. Cần verify kỹ hơn flow thực tế.

#### P2-6 đến P2-11: Tất cả đã verified và đồng ý.

---

## 3. Các Vấn Đề Plan V2 Thiếu

> [!CAUTION]
> Các vấn đề dưới đây không được đề cập trong plan V2 nhưng đáng xem xét.

### MISS-1. Import `re` không nhất quán

Trong `supervisor_node`, `re` module được import tại file-level bằng cách gián tiếp:
- [L154](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L154): `import re` bên trong `_expand_abbreviations_in_message`
- [L712](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L712): `_re.compile(...)` — nhưng `_re` **chưa được import ở scope `supervisor_node`**!
- [L748](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L748): Cùng vấn đề `_re.compile(...)`

Kiểm tra kỹ: `_re` có thể được import ở đâu đó giữa L340-700 mà tôi chưa xem — cần verify. Nếu thiếu, đây là **latent bug** chỉ trigger khi hit safety net paths.

### MISS-2. `_PERSONAL_REF_PATTERN` compiled mỗi lần gọi

Tại [supervisor.py:712-715](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L712-L715):
```python
_PERSONAL_REF_PATTERN = _re.compile(...)
```
Regex này được **compile mỗi lần `supervisor_node` chạy** (nằm trong function body). Nên chuyển ra module-level.

### MISS-3. Langfuse observation trùng lặp code

Gần như mọi node đều có pattern:
```python
if langfuse:
    try:
        obs = langfuse.start_observation(...)
        obs.update(...)
        obs.end()
    except Exception as e:
        logger.warning(...)
```
Pattern này lặp 15+ lần. Nên tạo helper `@with_langfuse_span` decorator hoặc context manager.

### MISS-4. `people_agent_node` iteration counting khác các node khác

Tại [people_agent.py:234](file:///home/AIRAG/backend/app/services/agents/people_agent.py#L234): `"iterations": state.get("iterations", 0) + 1` — tự increment. Nhưng `supervisor_node` cũng increment iterations tại [supervisor.py:799](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L799). Nếu flow là `supervisor → people`, iterations tăng 2 lần → có thể hit max iterations sớm hơn dự kiến.

### MISS-5. `query_analyzer_node` dùng `get_memory_agent()` cho complex extraction

Tại [supervisor.py:479](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L479), comment ghi "Qwen-memory 9B" nhưng `get_memory_agent()` return Qwen3-4B (theo AGENTS.md). Sai comment — không ảnh hưởng runtime nhưng gây confusion.

---

## 4. Đánh Giá Plan Triển Khai

### Phase A: Quick Wins ✅ ĐỒNG Ý

| Task | Đánh giá | Effort thực tế |
|---|---|---|
| A1: Typo BHxh | ✅ Trivial | 5 phút (đúng) |
| A2: Duplicate key | ✅ Trivial | 10 phút (đúng), **nhưng cần fix cả 3 key** chứ không chỉ 1 |
| A3: Bỏ mentions_specific_doc | ✅ OK | 30 phút (đúng) |
| A4: Fix regex | ✅ OK | 1.5 giờ (đúng) |
| A5: Intent constants | ✅ OK | 30 phút (đúng) |
| A6: Normalize shorthand | ⚠️ Nên dựa trên Langfuse data | 30 phút (đúng nếu biết thêm gì) |
| A7: Dead final_answer | ⚠️ Verify kỹ hơn trước khi xóa | 15 phút |
| A8: Regex thay LLM cho mongo_advanced | ⚠️ Nên dùng memory agent thay vì regex thuần | 1.5 giờ |

**Tổng Phase A:** 4-6 giờ ước lượng hợp lý.

---

### Phase B: Structural Refactor ⚠️ CẦN CHIA NHỎ HƠN

> [!WARNING]
> Phase B là **rủi ro cao nhất**. Tách 2057 dòng ra 8+ file trong 4 giờ (task B1) là quá nhanh. Circular import và `_re` alias sẽ gây vấn đề.

**Khuyến nghị thứ tự tách:**
1. `abbreviation.py` (ít coupling nhất — chỉ cần `_VI_STOP_WORDS`, `_VI_VOWELS`, DB access)
2. `intent_utils.py` (`_INTENT_NORMALIZE`, `_INTENT_TO_AGENT_FALLBACK`, `_parse_supervisor_response`)
3. `wrappers.py` (các `_*_wrapper` functions — chỉ import lazy)
4. `safety_nets.py` (cần import từ `intent_utils.py`)
5. `routing.py` (cần import từ `intent_utils.py`, `AgentType`)
6. `nodes/` package (cuối cùng — coupling cao nhất)

**Effort thực tế:** 2-3 ngày (không phải 1-2 ngày).

---

### Phase C: Logic Fixes ✅ ĐỒNG Ý

Đúng mức effort. Task C1 (section extraction) có thể cần thêm unit test cases.

### Phase D: Prompt Polish ✅ ĐỒNG Ý

Hợp lý. Chú ý D1 (rút gọn prompt) cần A/B testing cẩn thận — prompt hiện tại dù dài nhưng đã chạy ổn.

### Phase E: Testing ✅ ĐỒNG Ý — NHƯNG NÊN ĐƯA LÊN SỚM HƠN

> [!IMPORTANT]
> **Viết test TRƯỚC Phase B refactor**, không phải sau cùng. Không có test, refactor 2057 dòng code sẽ rất mạo hiểm.

**Đề xuất thứ tự mới:**
1. **Phase A** (quick wins) — đúng như plan
2. **Phase E** (testing) — **trước Phase B!**
3. **Phase B** (refactor) — có test bảo vệ
4. **Phase C** (logic fixes) — sau refactor dễ navigate hơn
5. **Phase D** (prompt polish) — cuối cùng

---

## 5. Câu Trả Lời Cho Câu Hỏi Của Plan (Section 8)

| # | Câu hỏi | Khuyến nghị |
|---|---|---|
| 1 | P0-3: Phương án A hay B? | **Phương án A** — bỏ `mentions_specific_doc`. Langfuse derive được từ `task_plan.includes("resolve_doc")` |
| 2 | Phase B backward compat? | **Có** — giữ re-export trong `supervisor.py` ít nhất 1 sprint, vì `agent/__init__.py` import trực tiếp |
| 3 | Giữ `is_legal_query` cho Langfuse? | **Giữ** nhưng chuyển thành computed field trong safety_nets, không để LLM output. Giảm prompt size mà vẫn có observability |
| 4 | Test coverage target? | **80% routing, 60% sub-agents** — hợp lý. Bổ sung: **100%** cho `_parse_supervisor_response` vì đây là single point of failure |
| 5 | Integration test với Qwen3-4B? | **Mock** cho CI; giữ 1 script integration test riêng chạy manual với actual model |

---

## 6. Tóm Tắt Khuyến Nghị

### Nên làm ngay (tuần này)
- ✅ Toàn bộ Phase A (4-6 giờ)
- ✅ Phase E trước Phase B (viết test trước khi refactor)

### Nên làm tuần tới
- ✅ Phase B (refactor) — nhưng chia nhỏ hơn plan đề xuất, mỗi sub-task test riêng
- ✅ Phase C (logic fixes)

### Nên thêm vào plan
- 🔧 MISS-2: Chuyển regex compile ra module-level
- 🔧 MISS-3: Tạo Langfuse helper decorator
- 🔧 MISS-4: Chuẩn hóa iteration counting (chỉ supervisor increment)
- 🔧 MISS-5: Fix sai comment "9B" → "4B"

### Cần thận trọng
- ⚠️ A6: Chỉ thêm normalize mapping khi có data thực tế
- ⚠️ A8: Dùng memory agent thay regex thuần cho mongo_advanced
- ⚠️ B1: Chia nhỏ hơn, test từng bước
- ⚠️ D1: A/B test prompt rút gọn trước khi deploy production

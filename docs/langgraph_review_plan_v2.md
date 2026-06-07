# Plan Cải Thiện LangGraph Agent System — V2 (Audit 2026-06-07)

> **Ngày review:** 2026-06-07
> **Người review:** Claude (M3)
> **Review chéo:** Antigravity (Gemini 3.5 Flash) — xem `langgraph_review_plan_v2_evaluation.md`
> **Phạm vi:** Toàn bộ hệ thống LangGraph supervisor trong `backend/app/services/agents/` và `backend/app/prompts/agents/`
> **Plan cũ** (`langgraph_improvement_plan.md`, 2025-05-25) đã được implement xong ở Phase 1-5; plan này tập trung vào các vấn đề phát sinh **sau** khi implement và clean-up cấu trúc code.
> **Version:** 2.1 (cập nhật sau review chéo + feedback user về "ask user when uncertain")

---

## 0. Nguyên Tắc Thiết Kế Mới — "Ask User When Uncertain"

> [!IMPORTANT]
> **Nguyên tắc do user đề xuất, áp dụng cho toàn bộ plan:**
> Với câu hỏi phức tạp, **regex/heuristic chỉ nên dùng khi có độ chính xác cao**. Với case mơ hồ, hệ thống nên **hỏi lại người dùng để cung cấp thêm thông tin**, tránh trả lời sai do đoán.

### Áp dụng cụ thể

| Tình huống | Cách hiện tại (regex) | Cách mới (ask user) |
|---|---|---|
| Multi-meaning abbreviation | Regex detect + LLM disambiguate từ context | Nếu LLM low-confidence → hỏi user chọn nghĩa (đã có) |
| Named document reference (Luật X) | Regex `_NAMED_DOC_PATTERN` match vài từ khóa | Nếu regex không match nhưng LLM output `mentions_specific_doc=true` → **hỏi user cung cấp tên/số chính xác** |
| Section reference (Điều 5 Khoản 2) | Heuristic `_extract_section_from_markdown` parse | Nếu parse được 0 chars → hỏi user xác nhận section hoặc cung cấp từ khóa liên quan |
| Personal keyword (tôi) | Regex `_PERSONAL_REF_PATTERN` | **Giữ regex** vì rõ ràng (không mơ hồ) |
| Disambiguation `tôi` trong ngữ cảnh pháp luật | Hiện không có | Nếu regex match `tôi` NHƯNG intent là legal → hỏi user: "Bạn muốn hỏi về [đơn vị mặc định] hay [đơn vị cụ thể khác]?" |
| Resolve doc ambiguous (nhiều candidates) | Trả về danh sách cho user chọn (đã có) | **Giữ** |
| Compare 2+ documents | Multi-step execution | **Giữ** (đủ thông tin để chạy) |
| Câu hỏi quá ngắn (<10 chars) | Skip query_analyzer | Skip — đủ an toàn cho greeting |

### Implementation

Cần 1 helper chung thay vì mỗi nơi tự build clarification message:

```python
# app/services/agents/clarification.py
async def ask_user_clarification(
    state: SupervisorState,
    *,
    question: str,
    options: list[str] | None = None,
    context: dict | None = None,
) -> None:
    """Emit a structured clarification event for the frontend.
    
    Frontend displays the question (and optional options), collects the
    user's response, and re-sends the query with the answer appended.
    """
    await push_event(state, "clarification", {
        "message": question,
        "options": options or [],
        "context": context or {},
    })
```

Các nơi gọi (sẽ thêm trong Phase A & C):
1. `supervisor._expand_abbreviations_in_message` (đã có cho multi-meaning) — wrap lại
2. `supervisor._REQUIRES_DOC_INTENTS` (P0-4) — thêm fallback ask khi regex miss
3. `rag_agent._extract_section_from_markdown` (P1-5) — thêm fallback ask khi parse 0 chars
4. Mới: Khi `complexity=cross_agent` + không đủ thông tin → ask user bổ sung

### Lợi ích
- Tránh trả lời sai do regex match sai
- Tăng user trust (họ thấy hệ thống "hiểu" họ cần gì)
- Giảm token spend cho retry loops
- Dễ debug — log khi nào hỏi user vs đoán

### Trade-off
- Tăng latency: user phải reply 1 lần trước khi có answer (so với auto-guess)
- UX kém hơn nếu hỏi quá nhiều → cần threshold hợp lý (e.g. chỉ hỏi khi confidence < 0.5)

---

## 1. Executive Summary

Hệ thống LangGraph hiện tại có **kiến trúc tốt** (16 intents, 7 agent types, plan-aware routing, memory recall, result evaluator với retry fallback, multi-step execution, observability với Langfuse). Tuy nhiên, qua audit phát hiện:

- **8 vấn đề P0** (bug/typo/dead code) cần fix ngay
- **8 vấn đề P1** (logic/cấu trúc) cần xử lý trong sprint tới
- **11 vấn đề P2** (prompt/sub-agent polish) có thể làm dần

**Khuyến nghị:** Ưu tiên fix P0 + P1 trước khi thêm intent/agent mới. Code hiện tại đã "feature-rich" nhưng cần consolidate trước khi scale.

| Metric | Hiện tại | Mục tiêu sau plan |
|---|---|---|
| Số dòng `supervisor.py` | 2057 | < 800 (tách thành nhiều file) |
| Số safety-net logic chồng chéo | 6-7 nơi | 1 deterministic post-processor |
| Số nơi increment `iterations` | 9 (supervisor + 4 sub-agents) | 1 (chỉ supervisor) |
| Typo trong constants | 1 (`MONGO_SEARCH_BHxh`) | 0 |
| Dead code fields | 1 (`mentions_specific_doc`) | 0 |
| Lỗi format trong prompt | 3 (3 keys duplicate) | 0 |
| Latent bugs từ missing import | 2 (`_re` chưa import) | 0 |
| Ví dụ query_analyzer | 5 intents | ≥ 12 intents (đầy đủ) |

---

## 2. Vấn Đề Phát Hiện (Theo Mức Độ)

### 🔴 P0 — Bug / Typo / Dead Code (Fix Ngay)

#### P0-1. Typo trong Intent constant
- **File:** `backend/app/services/agents/models.py:39`
- **Hiện tại:** `MONGO_SEARCH_BHxh = "mongo_search_bhxh"`
- **Vấn đề:** Inconsistent casing. Các hằng số khác là `MONGO_SEARCH_PHONE`, `MONGO_SEARCH_NAME` (UPPERCASE) nhưng `MONGO_SEARCH_BHxh` lại có chữ thường `xh`.
- **Rủi ro:** Dễ gây lỗi typo khi grep, refactor; nếu ai dùng `Intent.MONGO_SEARCH_BHxh` thay vì string `"mongo_search_bhxh"` sẽ không match trong `_INTENT_TO_AGENT_FALLBACK` (key là string lowercase).
- **Fix:** Đổi thành `MONGO_SEARCH_BHXH = "mongo_search_bhxh"`.

#### P0-2. Duplicate key trong supervisor_prompt.py output example
- **File:** `backend/app/prompts/agents/supervisor_prompt.py:333`
- **Hiện tại:**
  ```
  ...,"is_legal_query":false,"mentions_specific_doc":false,
  "needs_memory":false,"is_legal_query":true,"mentions_specific_doc":false,  ← DUPLICATE!
  ...
  ```
  Thực tế có **3 cặp key bị duplicate** (theo review chéo): `needs_memory`, `is_legal_query`, `mentions_specific_doc`. Tệ hơn — giá trị `is_legal_query` còn khác nhau giữa 2 lần (`false` vs `true`), gây inconsistent cho LLM.
- **Vấn đề:** Nếu LLM strict JSON parser, sẽ fail. Prompt mẫu này cũng dễ làm LLM bối rối.
- **Fix:** Bỏ duplicate, giữ 1 lần duy nhất với giá trị đúng theo example.

#### P0-3. `mentions_specific_doc` là dead code
- **File:** `backend/app/services/agents/supervisor.py:368, 436` + `supervisor_prompt.py:111-112`
- **Hiện tại:** Được LLM output, được parse, được lưu vào state, **không hề được dùng ở routing decision nào**. Chỉ truyền qua Langfuse span cho observability.
- **Fix — Phương án A (ưu tiên):** Bỏ khỏi prompt + parse để giảm token. Langfuse có thể derive từ `intent` thay thế.
- **Fix — Phương án B:** Wire vào `_REQUIRES_DOC_INTENTS` prerequisite check (thay vì dùng regex `_NAMED_DOC_PATTERN`).

#### P0-4. `_NAMED_DOC_PATTERN` regex quá giới hạn
- **File:** `backend/app/services/agents/supervisor.py:748-752`
- **Hiện tại:**
  ```python
  _NAMED_DOC_PATTERN = re.compile(
      r"(?:luật|nghị\s+định|thông\s+tư|quyết\s+định|nghị\s+quyết|pháp\s+lệnh|bộ\s+luật)"
      r"\s+\S", ...
  )
  ```
- **Vấn đề:** Không match các dạng:
  - "NĐ 13", "TT 15", "QĐ 53/2026" (viết tắt)
  - "Bộ luật Hình sự 2015" (sau type là cụm từ dài)
  - "Luật 24/2018" (sau "Luật" là số, không phải từ)
- **Trong khi đó:** `resolve_doc_agent._DOC_TYPE_KEYWORDS` lại có logic tốt hơn nhiều.
- **Fix:** Tái sử dụng `_DOC_TYPE_KEYWORDS` từ `resolve_doc_agent.py` (hoặc tách ra `app/services/agents/doc_keywords.py` shared).

#### P0-5. `_INTENT_TO_AGENT_FALLBACK` dùng raw string thay vì Intent constants
- **File:** `backend/app/services/agents/supervisor.py:62-82`
- **Hiện tại:** Key là `"mongo_search_cccd"`, `"search_section"`, ... (raw string)
- **Vấn đề:** Nếu đổi tên Intent constant (ví dụ P0-1), phải nhớ đổi 2 chỗ. Dễ drift.
- **Fix:** Import `Intent` class và dùng `Intent.MONGO_SEARCH_CCCD` etc.

#### P0-6. `_INTENT_NORMALIZE` không cover một số LLM shorthand
- **File:** `backend/app/services/agents/supervisor.py:44-59`
- **Hiện tại:** Có `search_bhxh`, `bhxh_search`, `find_phone`, `find_person`, `advanced_search`
- **Vấn đề:** Thiếu các LLM shorthand phổ biến khác:
  - `query_doc`, `find_doc` → `resolve_doc`
  - `summarize_doc` → `summarize`
  - `lookup_section`, `get_article` → `search_section`
  - `search_person`, `find_by_name` → `mongo_search_name`
- **Fix:** Bổ sung thêm mapping, có comment giải thích mỗi entry. **Lưu ý từ review chéo:** chỉ thêm mapping dựa trên Langfuse log thực tế, tránh over-engineer.

#### P0-7. 🔴 BUG: `_re` reference nhưng chưa import — sẽ gây `NameError`
- **File:** `backend/app/services/agents/supervisor.py:712, 748`
- **Hiện tại:**
  ```python
  # Line 712
  _PERSONAL_REF_PATTERN = _re.compile(...)
  # Line 748
  _NAMED_DOC_PATTERN = _re.compile(...)
  ```
- **Vấn đề:** `_re` được reference nhưng **chưa từng được import** ở scope này. Mỗi function khác trong file tự `import re as _re_xxx` (L154, L350, L494, L619, L1943) — nhưng `supervisor_node` thì không. Khi hit keyword safety net paths → `NameError: name '_re' is not defined` → exception → fallback về intent gốc (silent failure) hoặc crash tùy branch.
- **Verify:** Bash output xác nhận không có `import re` hay `as _re` ở file-level; chỉ có 2 reference `_re` tại L712, L748 không có import tương ứng.
- **Fix:**
  1. Thêm `import re` ở top of file
  2. Đổi `_re.compile` → `re.compile` (hoặc `import re as _re` rồi dùng `_re`)
  3. Bonus: chuyển 2 pattern compile ra module-level (xem P1-7)

#### P0-8. Comment sai: "Qwen-memory 9B" nhưng thực tế là "Qwen3-4B"
- **File:** `backend/app/services/agents/supervisor.py:479` (trong `query_analyzer_node`)
- **Hiện tại:** Comment nói "Qwen-memory 9B"
- **Thực tế:** `get_memory_agent()` trả về Qwen3-4B (theo AGENTS.md và CLAUDE.md)
- **Fix:** Đổi comment thành "Qwen-memory 4B (Qwen3-4B)"

---

### 🟠 P1 — Logic / Cấu Trúc (Sprint Tới)

#### P1-1. Supervisor có quá nhiều safety nets chồng chéo
- **File:** `backend/app/services/agents/supervisor.py` (nhiều nơi trong `supervisor_node`)
- **Hiện tại:** Có ít nhất 6-7 lớp override:
  1. `_INTENT_NORMALIZE` (line 44-59)
  2. `next_agent validation` (line 374-390)
  3. **Deterministic intent→agent override** (line 392-403)
  4. `task_plan[0]` correction (line 405-427)
  5. `needs_memory` keyword safety net (line 712-718)
  6. `is_legal_query` keyword safety net (line 723-734)
  7. `_REQUIRES_DOC_INTENTS` prerequisite injection (line 741-762)
  8. (Phase 5) Multi-step override (line 817-844)
- **Vấn đề:** Khi 1 query trigger nhiều safety net cùng lúc, không rõ cái nào thắng. Khó debug.
- **Fix:** Tạo 1 hàm `apply_deterministic_overrides(decision, state) -> decision` gom tất cả override theo thứ tự ưu tiên rõ ràng. Mỗi override có comment giải thích khi nào fire.

#### P1-2. `supervisor.py` 2057 dòng trong 1 file
- **File:** `backend/app/services/agents/supervisor.py`
- **Hiện tại:** Chứa nodes, routing, abbreviation logic, safety nets, graph builder
- **Đề xuất tách thành:**
  - `nodes/` package:
    - `supervisor.py` (~500 dòng) — chỉ `supervisor_node` + `query_analyzer_node`
    - `result_evaluator.py` (~200 dòng) — `result_evaluator_node`
    - `direct.py` (~150 dòng) — `direct_answer_node`
    - `answer_generator.py` (~100 dòng) — `answer_generator_node`
    - `mongo_formatter.py` (~100 dòng) — `mongo_formatter_node`
  - `routing.py` (~250 dòng) — `route_from_supervisor`, `route_from_rag`, `route_from_resolve_doc`, `route_from_evaluator`, `route_from_enricher`
  - `abbreviation.py` (~350 dòng) — `_is_likely_abbreviation`, `_expand_abbreviations_in_message`, `_disambiguate_multi_meaning_abbrs`
  - `safety_nets.py` (~150 dòng) — `apply_deterministic_overrides`, `personal_keyword_check`, `legal_keyword_check`
  - `graph.py` (~150 dòng) — `create_supervisor_graph`, `get_supervisor_graph`, `reset_supervisor_graph`
  - `wrappers.py` (~200 dòng) — các `_memory_recall_wrapper`, `_query_enricher_wrapper`, `_rag_agent_wrapper`, etc.
  - `intent_utils.py` (~100 dòng) — `_INTENT_NORMALIZE`, `_INTENT_TO_AGENT_FALLBACK`, `_parse_supervisor_response`
- **Lợi ích:** Dễ navigate, dễ test từng phần, dễ onboard contributor mới.

#### P1-3. `task_plan` được build ở 3 nơi, dễ conflict
- **File:** `backend/app/services/agents/supervisor.py`
- **Hiện tại:**
  1. LLM (prompt) — primary path
  2. `_REQUIRES_DOC_INTENTS` injection (line 737-762) — nếu intent là summarize/search_section + regex match named doc
  3. Phase 5 multi-step (line 817-844) — từ `sub_queries` của query_analyzer
- **Vấn đề:** Khi 2 và 3 cùng fire (sub_queries có 1 step là summarize trên named doc), code L825-844 **ghi đè cả `task_plan`** mà không check prerequisite đã inject. Có thể bỏ qua resolve_doc.
- **Fix:** Trong Phase 5 override, check nếu `task_plan` đã có `resolve_doc` ở đầu → prepend thay vì replace.

#### P1-4. State field `intent` bị mutate ở nhiều nơi
- **File:** nhiều file
- **Hiện tại:** Set `intent` ở ≥5 nơi:
  1. `supervisor.py:379-792` (loop-back → "summarize")
  2. `supervisor.py:835` (multi-step override)
  3. `rag_agent.py:526` (`_map_resolve_doc_result` → "search_section" nếu có section_ref)
  4. `rag_agent.py:778` (`_map_search_section` → "summarize" để trick routing)
  5. `resolve_doc_agent.py:857, 862, 871` (dựa vào `section_ref` / `pending_intent`)
- **Vấn đề:** Khó theo dõi intent flow. Đặc biệt `rag_agent.py:778` set `intent="summarize"` chỉ để route sang `answer_generator` — anti-pattern.
- **Fix:**
  - Thêm 1 hàm `set_intent(state, new_intent, *, source: str)` log lại mỗi lần mutate.
  - Dùng explicit `route_to_answer_generator: bool` flag thay vì hijack intent.

#### P1-5. `_extract_section_from_markdown` heuristic sai ranh giới
- **File:** `backend/app/services/agents/rag_agent.py:546-639`
- **Hiện tại:** Stop tại cùng `section_type` kế tiếp.
- **Vấn đề:** Với `section_ref = "Khoản 2 Điều 8"`, sẽ sai ranh giới vì "Khoản" có thể nằm trong cùng Điều. Hiện tại hàm chỉ dùng phần đầu ("Khoản") làm boundary, nên sẽ stop tại "Khoản 3" trong cùng Điều — sai.
- **Fix:**
  - Parse `section_ref` thành structured (type + number + parent_type + parent_number).
  - Stop tại boundary hợp lý (parent_type nếu có, hoặc type cao hơn: Chương, Phần, Mục).
  - Viết unit test cho case "Khoản 2 Điều 8", "Chương II Điều 5", "Phụ lục 1".

#### P1-6. Logic `_REQUIRES_DOC_INTENTS` và resolve_doc agent không đồng bộ
- **File:**
  - `supervisor.py:741-762` (chỉ check `summarize`, `search_section` + regex)
  - `resolve_doc_agent.py:_DOC_TYPE_KEYWORDS` (cover nhiều hơn)
- **Vấn đề:** Supervisor có thể miss detection → bỏ qua resolve_doc step → rag_agent bị lỗi vì không có document_ids.
- **Fix:** Extract `_DOC_TYPE_KEYWORDS` và logic detection ra `app/services/agents/doc_keywords.py` shared. **Áp dụng nguyên tắc "ask user when uncertain":** nếu regex miss nhưng LLM output `intent ∈ _REQUIRES_DOC_INTENTS` → hỏi user cung cấp tên/số văn bản thay vì default `search`.

#### P1-7. Regex pattern compile mỗi lần gọi + Langfuse code lặp 15+ lần
- **File:** `supervisor.py:712, 748` (regex) + nhiều file (Langfuse)
- **Hiện tại:**
  - `_PERSONAL_REF_PATTERN`, `_NAMED_DOC_PATTERN` được compile **bên trong `supervisor_node`** → compile mỗi request
  - Pattern Langfuse `try/except + start_observation/update/end` lặp 15+ lần trong supervisor.py
- **Fix:**
  1. Chuyển 2 regex compile ra module-level (cùng với fix P0-7)
  2. Tạo helper `@with_langfuse_span(name, input_fields)` decorator hoặc context manager trong `app/services/agent/langfuse_tracing.py` (đã có file này)
  3. Refactor 15+ nơi dùng pattern trên → gọi helper

#### P1-8. `iterations` bị increment ở 9 nơi → hit max quá sớm
- **File:** `supervisor.py:556, 788, 799, 982` + `rag_agent.py:868` + `write_agent.py:213, 365` + `people_agent.py:234, 252`
- **Hiện tại:** Cả supervisor lẫn 4 sub-agents đều tự increment `iterations`. Với flow `supervisor(1) → rag(2) → supervisor(3) → result_evaluator(retry) → rag(4) → supervisor(5)` đã hit max 6.
- **Vấn đề:** User thực sự muốn nhiều retry → bị cap sớm.
- **Fix:**
  1. Quy ước: **chỉ supervisor (và query_analyzer) mới increment `iterations`**
  2. Sub-agents chỉ return `iterations` không thay đổi, hoặc return delta
  3. Hoặc: đổi `iterations` thành `supervisor_iterations` để rõ scope

---

### 🟢 P2 — Prompt / Sub-agent Polish (Làm Dần)

#### P2-1. Supervisor prompt quá dài (25.5 KB)
- **File:** `backend/app/prompts/agents/supervisor_prompt.py`
- **Hiện tại:** 1 prompt duy nhất 333 dòng
- **Vấn đề:** Quá nhiều examples (8 cho DIRECT, 3 cho SEARCH_ABBR, 4 cho RESOLVE_DOC+SEARCH, 5 cho RESOLVE_DOC+SEARCH_SECTION, 2 cho WRITE, 4 cho PEOPLE). LLM có thể bị "lost in the middle".
- **Fix:**
  - Rút gọn xuống ~15-20 KB
  - Bỏ examples trùng pattern
  - Hoặc tách thành: routing decision prompt + (optional) plan generation prompt (chỉ chạy cho `complexity != "simple"`)

#### P2-2. Query analyzer prompt thiếu examples
- **File:** `backend/app/prompts/agents/query_analyzer_prompt.py`
- **Hiện tại:** INTENT HINTS liệt kê đủ 16 intents nhưng EXAMPLES chỉ cover 5
- **Vấn đề:** LLM có thể không biết cách output cho `personal`, `search_abbr`, `search_doc_num`, `list_docs`, `mongo_search_name`, `mongo_search_bhxh`, `mongo_search_phone`, `mongo_search_advanced`.
- **Fix:** Bổ sung ≥ 1 example cho mỗi intent_hint còn thiếu.

#### P2-3. `is_legal_query` định nghĩa quá rộng
- **File:** `backend/app/prompts/agents/supervisor_prompt.py:110-111`
- **Hiện tại:** "true if the query is about laws, regulations, policies, concepts (quy định, trách nhiệm, luật, nghị định, thông tư, bảo mật...)"
- **Vấn đề:** "bảo mật" có thể là IT security thuần túy, không phải pháp luật → false positive dẫn đến safety net override sai.
- **Fix:** Dùng regex-based check (`_LEGAL_KEYWORDS` set) thay vì để LLM quyết.

#### P2-4. `people_agent._tool_mongo_advanced` lạm dụng LLM
- **File:** `backend/app/services/agents/people_agent.py:78-129`
- **Hiện tại:** Gọi LLM chính để extract criteria từ query
- **Vấn đề:** Tốn thêm 1-2s; criteria có thể extract bằng regex (CCCD 12 số, phone 10-11 số, name capitalized)
- **Fix:** Dùng regex extract trước, chỉ fallback LLM khi regex không match.

#### P2-5. `_map_mongo_result` set `final_answer` trực tiếp
- **File:** `backend/app/services/agents/people_agent.py:38`
- **Hiện tại:** Set `final_answer: display`
- **Vấn đề:** Có comment "Needed by answer_generator mongo branch" — nhưng graph hiện tại route `people → mongo_formatter → END`, không qua `answer_generator`. Field này dead.
- **Fix:** Bỏ field `final_answer` trong `_map_mongo_result`.

#### P2-6. `rag_agent._map_search_section` hijack intent
- **File:** `backend/app/services/agents/rag_agent.py:778`
- **Hiện tại:** Set `intent="summarize"` chỉ để route sang `answer_generator`
- **Vấn đề:** Magic value, anti-pattern
- **Fix:** Dùng explicit flag `route_to_answer_generator: True` (xem P1-4)

#### P2-7. `resolve_doc_agent._query_db` nhiều OR conditions
- **File:** `backend/app/services/agents/resolve_doc_agent.py:336-358`
- **Hiện tại:** Mỗi candidate thêm 2 ILIKE conditions; với `title_keywords[:4]` × 2 = 8 LIKE
- **Vấn đề:** Có thể chậm trên documents lớn
- **Fix:** Cân nhắc dùng `pg_trgm` extension + GIN index, hoặc full-text search `ts_vector`.

#### P2-8. Ví dụ supervisor_prompt thiếu case quan trọng
- **File:** `backend/app/prompts/agents/supervisor_prompt.py`
- **Thiếu examples cho:**
  - Multi-doc comparison: "So sánh NĐ 13 và Luật ANM về bảo vệ dữ liệu" (complexity=comparison)
  - File upload: "File tôi upload có nội dung gì?" (cần `document_ids` từ upload, intent=search/summarize)
  - Multi-section same doc: "Tóm tắt Điều 5 và Điều 7 Luật ANM" (multi_section)
  - Mixed intent: "Tìm CCCD 0123456789 và tra cứu đơn vị theo NĐ 83" (cross_agent)
  - Resolve_doc + search + personal: "Đơn vị tôi cần tuân thủ điều 5 NĐ 83 như thế nào?" (resolve_doc+search_section+memory)
- **Fix:** Bổ sung ít nhất 1 example mỗi case.

#### P2-9. Abbreviation heuristic chưa cover case có dấu
- **File:** `backend/app/services/agents/supervisor.py:88-110`
- **Hiện tại:** `_VI_VOWELS` có full diacritics. OK.
- **Vấn đề:** `_is_likely_abbreviation` không match viết tắt có dấu (ví dụ `CAND` thì rõ ràng viết tắt, nhưng `CÔNG AN` thì cũng có thể).
- **Fix:** Cân nhắc bổ sung whitelist các viết tắt phổ biến (BCA, BTP, ...)

#### P2-10. Thiếu test cho routing decisions
- **Hiện tại:** `backend/test_llm_classifier.py` chỉ test `_llm_classify_identity` (1 hàm)
- **Vấn đề:** Không có test cho supervisor routing, query_analyzer, resolve_doc, abbreviation expansion
- **Fix:** Tạo `backend/tests/test_langgraph_routing.py` với:
  - Unit test cho `_parse_supervisor_response` (mock LLM response)
  - Unit test cho `_expand_abbreviations_in_message` (với mock DB)
  - Unit test cho `_extract_by_regex` trong resolve_doc_agent
  - Integration test cho graph (mock tất cả LLM calls)

#### P2-11. `mongo_formatter_node` chỉ dùng cho display
- **File:** `backend/app/services/agents/supervisor.py:1275-1337`
- **Hiện tại:** Hard-code format prompt ~1KB
- **Vấn đề:** Format chỉ phù hợp cho CCCD/name/phone lookup. Nếu thêm loại person search mới → phải sửa prompt.
- **Fix:** Tách format prompt ra `app/prompts/agents/mongo_formatter_prompt.py` để dễ iterate.

---

## 3. Plan Triển Khai

> [!IMPORTANT]
> **Thứ tự phases đã đổi** theo khuyến nghị của review chéo: **A → E (test) → B (refactor) → C (logic) → D (prompt)**.
> Lý do: viết test trước để có safety net khi refactor 2057 dòng code.

### Phase A: Quick Wins + Bug Fixes (Ước lượng: 6-8 giờ)
| # | Task | Files | Effort | Note |
|---|---|---|---|---|
| A1 | Sửa typo `MONGO_SEARCH_BHxh` → `MONGO_SEARCH_BHXH` | `models.py` | 5 ph | P0-1 |
| A2 | Sửa 3 duplicate keys (không phải 1) trong supervisor_prompt.py:333 | `supervisor_prompt.py` | 15 ph | P0-2 |
| A3 | 🔴 **Fix bug `_re` chưa import** tại L712, L748 (NameError latent) | `supervisor.py` | 30 ph | **P0-7** |
| A4 | Fix comment sai "9B" → "4B" tại L479 | `supervisor.py` | 5 ph | P0-8 |
| A5 | Bỏ `mentions_specific_doc` khỏi prompt + parse (chọn Phương án A) | `supervisor.py`, `supervisor_prompt.py` | 30 ph | P0-3 |
| A6 | Sửa `_NAMED_DOC_PATTERN` cover viết tắt (NĐ, TT, QĐ) + share keywords với resolve_doc + **thêm fallback ask user khi regex miss** | `supervisor.py`, `resolve_doc_agent.py`, `app/services/agents/clarification.py` (mới) | 2 giờ | P0-4 + nguyên tắc "ask user" |
| A7 | Đổi `_INTENT_TO_AGENT_FALLBACK` keys sang `Intent.*` constants | `supervisor.py` | 30 ph | P0-5 |
| A8 | Bổ sung `_INTENT_NORMALIZE` dựa trên Langfuse log thực tế | `supervisor.py` | 30 ph | P0-6 |
| A9 | Tạo helper `clarification.ask_user_clarification()` | `app/services/agents/clarification.py` (mới) | 1 giờ | Nguyên tắc mới |
| A10 | Chuyển 2 regex compile ra module-level | `supervisor.py` | 15 ph | P1-7 (làm sớm cùng A3) |

**Verify sau Phase A:**
- Chạy `test_llm_classifier.py` — phải pass
- Manual test 10 query sample (greeting, search, named doc, ambiguous) — không crash
- Verify `_PERSONAL_REF_PATTERN` chỉ compile 1 lần (log debug)

### Phase E: Testing (LÀM TRƯỚC PHASE B) (Ước lượng: 1-1.5 ngày)
| # | Task | Files | Effort | Note |
|---|---|---|---|---|
| E1 | Tạo `backend/tests/test_langgraph_routing.py` — unit test cho `_parse_supervisor_response` | new file | 4 giờ | **Target 100% coverage** |
| E2 | Tạo `backend/tests/test_abbreviation_expansion.py` — unit test cho `_is_likely_abbreviation`, `_expand_abbreviations_in_message` | new file | 2 giờ | Mock DB |
| E3 | Tạo `backend/tests/test_resolve_doc_regex.py` — unit test cho `_extract_by_regex` | new file | 1 giờ | Cover các case: Luật/NĐ/TT/QĐ, có/không số, có/không năm |
| E4 | Tạo `backend/tests/test_section_extraction.py` — unit test cho `_extract_section_from_markdown` | new file | 1 giờ | Cover "Điều 5", "Khoản 2 Điều 8", "Chương II Điều 3" |
| E5 | Integration test mock full graph flow (10 scenario end-to-end) | new file | 2 giờ | Mock tất cả LLM calls |
| E6 | Snapshot test 30 query thực tế + capture routing decision | new file | 1 giờ | Manual + script |

**Verify sau Phase E:** `pytest backend/tests/` phải pass 100%; output rõ ràng intent/route per query.

### Phase B: Structural Refactor (Ước lượng: 2-3 ngày — đã điều chỉnh theo review chéo)
> [!WARNING]
> Phase B là rủi ro cao nhất. Thứ tự tách file dưới đây theo mức coupling thấp → cao. Mỗi bước phải test end-to-end trước khi tách tiếp.

| # | Task | Files | Effort |
|---|---|---|---|
| B1 | Tạo `abbreviation.py` (ít coupling nhất) | `supervisor.py` → `abbreviation.py` | 2 giờ |
| B2 | Tạo `intent_utils.py` (`_INTENT_NORMALIZE`, `_INTENT_TO_AGENT_FALLBACK`, `_parse_supervisor_response`) | `supervisor.py` → `intent_utils.py` | 2 giờ |
| B3 | Tạo `wrappers.py` (các `_*_wrapper` functions) | `supervisor.py` → `wrappers.py` | 1.5 giờ |
| B4 | Tạo `safety_nets.py` với `apply_deterministic_overrides()` (gom 6-7 lớp override) | `supervisor.py` → `safety_nets.py` | 3 giờ |
| B5 | Tạo `routing.py` (route_from_supervisor/rag/resolve_doc/evaluator/enricher) | `supervisor.py` → `routing.py` | 2 giờ |
| B6 | Tạo `nodes/` package (cuối cùng, coupling cao nhất) | `supervisor.py` → 5-6 file mới | 6-8 giờ |
| B7 | `supervisor.py` còn lại chỉ re-export để backward compat | `supervisor.py` | 30 ph |

**Verify sau Phase B:**
- `pytest backend/tests/` vẫn pass 100%
- Manual test 30 query thực tế — không regression
- `wc -l supervisor.py` < 800 dòng

### Phase C: Logic Fixes (Ước lượng: 1.5-2 ngày)
| # | Task | Files | Effort | Note |
|---|---|---|---|---|
| C1 | Fix `_extract_section_from_markdown` cho case "Khoản X Điều Y" + thêm fallback ask user | `rag_agent.py` | 3 giờ | P1-5 + nguyên tắc "ask user" |
| C2 | Fix Phase 5 multi-step respect prerequisite `task_plan[0] == "resolve_doc"` | `supervisor.py` (sau refactor) | 2 giờ | P1-3 |
| C3 | Tạo `set_intent()` helper, refactor 5+ nơi direct intent mutation | nhiều file | 3 giờ | P1-4 |
| C4 | Thay `_NAMED_DOC_PATTERN` bằng shared `_DOC_TYPE_KEYWORDS` | `supervisor.py`, `resolve_doc_agent.py` | 1 giờ | P1-6 |
| C5 | Tạo `mongo_formatter_prompt.py` riêng | `supervisor.py` → file mới | 30 ph | P2-11 |
| C6 | Chuẩn hóa iteration counting — chỉ supervisor increment | 5 files | 2 giờ | P1-8 |
| C7 | Tạo Langfuse `@with_langfuse_span` decorator/helper, refactor 15+ nơi | `langfuse_tracing.py` + nhiều file | 3 giờ | P1-7 |

**Verify sau Phase C:** Re-run Phase E tests; manual test các case section/iteration phức tạp.

### Phase D: Prompt Polish (Ước lượng: 1-1.5 ngày)
| # | Task | Files | Effort | Note |
|---|---|---|---|---|
| D1 | Rút gọn supervisor_prompt (~25KB → ~15KB) | `supervisor_prompt.py` | 3 giờ | P2-1. **Cẩn thận:** A/B test trước khi deploy prod |
| D2 | Bổ sung examples cho query_analyzer (≥ 12 intents) | `query_analyzer_prompt.py` | 2 giờ | P2-2 |
| D3 | Thay `is_legal_query` LLM flag bằng computed field trong safety_nets | `supervisor.py` | 1.5 giờ | P2-3 |
| D4 | Bổ sung 5-8 examples quan trọng cho supervisor_prompt | `supervisor_prompt.py` | 2 giờ | P2-8 |
| D5 | Áp dụng "ask user" — tích hợp `ask_user_clarification` vào: (1) `_extract_section_from_markdown`, (2) `_REQUIRES_DOC_INTENTS` regex miss, (3) cross_agent low confidence | `rag_agent.py`, `supervisor.py` | 3 giờ | Nguyên tắc mới |

**Verify sau Phase D:**
- Manual test 50 query thực tế + đo LLM accuracy (target > 95%)
- Test clarification flow end-to-end với frontend
- A/B test prompt rút gọn (D1) với 100 query

---

## 4. Files Sẽ Bị Ảnh Hưởng

### Sửa đổi
- `backend/app/services/agents/supervisor.py` — tách lớn, fix logic
- `backend/app/services/agents/models.py` — sửa typo
- `backend/app/services/agents/rag_agent.py` — fix section extraction, sửa mapper
- `backend/app/services/agents/people_agent.py` — bỏ dead code, regex extract
- `backend/app/services/agents/resolve_doc_agent.py` — share keywords
- `backend/app/prompts/agents/supervisor_prompt.py` — fix format, rút gọn
- `backend/app/prompts/agents/query_analyzer_prompt.py` — bổ sung examples

### Tạo mới
- `backend/app/services/agents/nodes/` package
- `backend/app/services/agents/routing.py`
- `backend/app/services/agents/abbreviation.py`
- `backend/app/services/agents/safety_nets.py`
- `backend/app/services/agents/wrappers.py`
- `backend/app/services/agents/graph.py`
- `backend/app/services/agents/intent_utils.py`
- `backend/app/services/agents/doc_keywords.py` (shared)
- `backend/app/prompts/agents/mongo_formatter_prompt.py`
- `backend/tests/test_langgraph_routing.py`
- `backend/tests/test_abbreviation_expansion.py`
- `backend/tests/test_resolve_doc_regex.py`

### Backward compat
- `supervisor.py` cũ sẽ chỉ re-export (`get_supervisor_graph`, `create_supervisor_graph`, `SupervisorStateModel`, `create_supervisor`) để không break các import hiện tại.
- `agent/__init__.py` (`from app.services.agents.supervisor import get_supervisor_graph as build_agent_graph`) vẫn work.

---

## 5. Rủi Ro & Biện Pháp Giảm Thiểu

| Rủi ro | Xác suất | Tác động | Mitigation |
|---|---|---|---|
| Refactor Phase B gây regression | Trung bình | Cao | Giữ backward compat layer; chạy full integration test sau mỗi bước |
| Fix regex doc detection miss một số case | Thấp | Trung bình | A/B test với 50 query có named doc; fallback nếu regex miss |
| Thay LLM `is_legal_query` bằng regex tăng false positive | Thấp | Thấp | Bắt đầu với whitelist keywords VN phổ biến, mở rộng dần |
| Tách file nhiều quá, contributor mới khó onboard | Trung bình | Thấp | Viết `agents/README.md` với kiến trúc tổng quan |
| Bỏ `mentions_specific_doc` break Langfuse dashboard | Thấp | Thấp | Check trước khi bỏ; nếu cần giữ cho observability, đổi tên thành `_derived_field` |

---

## 6. Tiêu Chí Hoàn Thành (Definition of Done)

Mỗi task trong Plan A-E phải:
- [ ] Có test pass (nếu áp dụng được)
- [ ] Không break existing tests
- [ ] Không tăng latency trung bình > 100ms
- [ ] Có log rõ ràng (Langfuse span nếu là routing)
- [ ] Có comment giải thích logic mới (nếu thay đổi behavior)

Sau toàn bộ Phase A-E:
- [ ] `supervisor.py` < 800 dòng
- [ ] 0 typo trong constants
- [ ] 0 dead code fields trong prompt output
- [ ] 0 latent bug (NameError, missing import)
- [ ] 0 nơi increment `iterations` ngoài supervisor
- [ ] ≥ 80% code coverage cho routing logic, 100% cho `_parse_supervisor_response`
- [ ] Manual test 50+ query không có regression
- [ ] Langfuse dashboard vẫn hoạt động bình thường
- [ ] `ask_user_clarification` flow hoạt động end-to-end với frontend

---

## 7. Khuyến Nghị Thứ Tự Thực Hiện

> [!TIP]
> Thứ tự đã cập nhật theo review chéo: **A → E (test) → B (refactor) → C (logic) → D (prompt)**

1. **Ngay hôm nay:** Phase A (6-8 giờ) — fix bug, tạo `clarification.py` skeleton
2. **Tuần này:** Phase E (1-1.5 ngày) — viết test TRƯỚC khi refactor
3. **Tuần này/tới:** Phase B (2-3 ngày) — refactor có test bảo vệ
4. **Tuần sau:** Phase C (1.5-2 ngày) — fix logic + iteration counting
5. **Sprint tiếp theo:** Phase D (1-1.5 ngày) — polish prompt + tích hợp "ask user"
6. **Sprint tiếp theo:** Thêm intent/agent mới (nếu có yêu cầu)

---

## 8. Câu Hỏi Cần Xác Nhận Từ User

Trước khi bắt đầu, cần user quyết:

1. **P0-3 Phương án A hay B?** Bỏ `mentions_specific_doc` (đơn giản) hay wire vào prerequisite check (giữ observability cao hơn)?
   - *Khuyến nghị review chéo: Phương án A — Langfuse derive từ `task_plan.includes("resolve_doc")`*

2. **Phase B có cần backward compat layer không?** Hay OK để refactor sạch + update imports?
   - *Khuyến nghị review chéo: Có — giữ re-export trong `supervisor.py` ít nhất 1 sprint*

3. **Có muốn giữ `is_legal_query` LLM flag làm metadata cho Langfuse (không dùng cho routing)?**
   - *Khuyến nghị review chéo: Giữ nhưng chuyển thành computed field trong safety_nets*

4. **Test coverage target bao nhiêu %?**
   - *Khuyến nghị: 80% routing, 60% sub-agents, **100% cho `_parse_supervisor_response`** (single point of failure)*

5. **Có cần thêm integration test với Qwen3-4B (memory agent) hay chỉ mock?**
   - *Khuyến nghị: Mock cho CI; giữ 1 script integration test riêng chạy manual với actual model*

6. **(Mới) Nguyên tắc "Ask user when uncertain":** Có sẵn sàng chấp nhận tăng latency (user phải reply 1 lần) để giảm trả lời sai? Threshold hỏi user ở confidence < ? (đề xuất < 0.5)
   - *Mặc định đề xuất: < 0.5 cho resolve_doc, < 0.6 cho section extraction, **không bao giờ** cho greeting/personal (rõ ràng)*

7. **(Mới) Phase B chia nhỏ 7 sub-task (B1→B7) theo thứ tự coupling, OK không?**

---

## Appendix A: Thống Kê Code Hiện Tại

| File | Số dòng | Vai trò |
|---|---|---|
| `supervisor.py` | 2057 | Main graph + nodes + routing + abbreviation |
| `supervisor_prompt.py` | 333 | Supervisor system prompt |
| `query_analyzer_prompt.py` | 100 | Query analyzer prompt |
| `rag_agent.py` | 869 | RAG operations (search, list, kg, section, abbr) |
| `resolve_doc_agent.py` | 875 | Document resolution (3-stage fallback) |
| `write_agent.py` | 376 | Write operations (summarize, edits, grammar, format) |
| `people_agent.py` | 254 | MongoDB people search |
| `models.py` | 237 | State, Intent, AgentType enums |
| `write_agent_prompt.py` | 134 | Write prompts + 30/2020 standard loader |

**Tổng:** ~5,200 dòng code + ~600 dòng prompt

## Appendix B: So Sánh Với Plan V1

| V1 (2025-05-25) | V2 (2026-06-07) |
|---|---|
| Smart RAG Search Routing | ✅ Đã implement (search_mode) |
| Resolve Doc Agent | ✅ Đã implement (resolve_doc_agent) |
| Plan-Aware Supervisor | ✅ Đã implement (task_plan, pending_intent) |
| Query Analyzer | ✅ Đã implement (Phase 5) |
| Result Evaluator | ✅ Đã implement (Phase 5) |
| Memory Recall | ✅ Đã implement (Phase 3) |
| **Code structure** | 🔴 Cần refactor (2057 dòng) |
| **Bug fixes** | 🔴 Cần fix P0 (8 issues bao gồm P0-7 latent bug) |
| **Prompt polish** | 🟢 Làm dần |
| **Test coverage** | 🔴 Thiếu test (sẽ viết trước Phase B) |
| **Ask user when uncertain** | 🆕 Nguyên tắc mới trong V2.1 |

Plan V2 tập trung vào **technical debt cleanup** sau khi V1 đã hoàn thành các feature lớn.

---

## Appendix C: Review Chéo Từ Antigravity

File `langgraph_review_plan_v2_evaluation.md` (cùng folder) là đánh giá độc lập từ AI khác (Gemini 3.5 Flash). Các phát hiện quan trọng đã được incorporate:

| # | Phát hiện | Đã update trong V2.1 |
|---|---|---|
| P0-2 | Duplicate thực tế là 3 keys, không phải 1 | ✅ |
| MISS-1 | `_re` chưa import → latent NameError | ✅ Thêm P0-7 |
| MISS-2 | Regex compile mỗi call | ✅ Gộp vào P1-7 |
| MISS-3 | Langfuse code lặp 15+ lần | ✅ Gộp vào P1-7 |
| MISS-4 | Iteration increment ở 9 nơi | ✅ Thêm P1-8 |
| MISS-5 | Comment sai "9B" | ✅ Thêm P0-8 |
| A6 | Chỉ thêm normalize mapping khi có data thực | ✅ Note trong A8 |
| A8 | Dùng memory_agent thay regex thuần | ✅ Note trong D1 (mongo) |
| Phase B | Chia nhỏ 7 sub-task theo coupling | ✅ B1-B7 có thứ tự |
| Phase E | Viết test TRƯỚC refactor | ✅ Phase A → E → B → C → D |
| P0-3 | Phương án A (bỏ) | ✅ Mặc định |
| Phase B | Backward compat | ✅ Mặc định |
| Coverage | 100% cho `_parse_supervisor_response` | ✅ Trong DoD |

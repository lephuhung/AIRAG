# Review: LangGraph Improvement Plan V2.2 — Non-Prompt Sections

> **Scope:** Đánh giá Phase A (Quick Wins), Phase B (Structural Refactor), Phase C (Logic Fixes), Phase E (Testing), Section D (Architecture), Section F (Files Inventory), cross-reference với code thực tế trong `supervisor.py` (2118 dòng), `rag_agent.py` (869 dòng), `resolve_doc_agent.py` (874 dòng), `models.py` (237 dòng), `clarification.py` (109 dòng).
>
> **Prompt review trước đó:** [prompt_review.md](file:///home/AIRAG/docs/prompt_review.md) — đã cover Section A (Prompt-Specific) + Section C (Test Framework).
>
> **Code files cross-verified:**
> - [supervisor.py](file:///home/AIRAG/backend/app/services/agents/supervisor.py) (2118 lines, 90KB)
> - [rag_agent.py](file:///home/AIRAG/backend/app/services/agents/rag_agent.py) (869 lines, 33KB)
> - [models.py](file:///home/AIRAG/backend/app/services/agents/models.py) (237 lines)
> - [clarification.py](file:///home/AIRAG/backend/app/services/agents/clarification.py) (109 lines)
> - [test_langgraph_routing.py](file:///home/AIRAG/backend/tests/agents/test_langgraph_routing.py) (273 lines)
> - [test_abbreviation_expansion.py](file:///home/AIRAG/backend/tests/agents/test_abbreviation_expansion.py)

---

## Phase A — Quick Wins (6-8 giờ)

### A1: Fix typo `MONGO_SEARCH_BHxh` → `MONGO_SEARCH_BHXH` — ✅ ĐÃ FIX

Grep `MONGO_SEARCH_BHxh` (case-sensitive) trả về **0 kết quả**. Tất cả occurrences đều đã chuẩn — `MONGO_SEARCH_BHXH` uppercase đúng.

**Đánh giá plan:** ✅ Task này đã hoàn thành, nên **bỏ khỏi plan** hoặc mark done.

---

### A2: Fix duplicate keys trong supervisor_prompt.py:333 — ⚠️ Overlap với Prompt Step 1

Plan ghi "duplicate keys" nhưng thực tế vấn đề là **thiếu comma giữa `is_legal_query` và `reasoning`** (P0 Critical). Đây là CÙNG bug với Prompt Step 1.

**Đánh giá plan:** ⚠️ A2 và Prompt Step 1 là **duplicate** — nên gộp lại 1 task, tránh fix 2 lần.

---

### A3: Fix latent NameError `_re` chưa import — ✅ ĐÃ FIX

Grep `_re.` trong supervisor.py trả về **0 kết quả**. Code hiện tại dùng:
- L20: `import re` (module-level)
- L199, 395, 539, 664, 2003: `import re as _re_parse`, `_re_qa`, `_re_abbr`, `_re_enrich` (function-scoped, đúng convention)

Không còn bare `_re.` nào bị thiếu import.

**Đánh giá plan:** ✅ Task này đã hoàn thành, nên **bỏ khỏi plan** hoặc mark done.

---

### A4: Fix comment sai "9B" → "4B" — Không verify được

Grep không thấy "9B" trong supervisor.py context liên quan đến model size. Có thể đã fix hoặc comment đã thay đổi.

**Đánh giá plan:** ⚠️ Cần kiểm tra lại nếu còn relevant.

---

### A5: Bỏ `mentions_specific_doc` khỏi prompt + parse — ✅ ĐÃ FIX

Code xác nhận:
- [supervisor.py L413](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L413): Comment rõ ràng `mentions_specific_doc removed (P0-3 — was dead code)`
- [supervisor_prompt.py L112](file:///home/AIRAG/backend/app/prompts/agents/supervisor_prompt.py#L112): Comment `mentions_specific_doc removed in V2.1`
- [test_langgraph_routing.py L235-244](file:///home/AIRAG/backend/tests/agents/test_langgraph_routing.py#L235-L244): Test `test_mentions_specific_doc_field_ignored()` confirm backward compat

**Đánh giá plan:** ✅ Task này đã hoàn thành, nên **bỏ khỏi plan** hoặc mark done.

---

### A6: Mở rộng `_NAMED_DOC_PATTERN` + share keywords với resolve_doc — ⚠️ CHƯA LÀM

Code xác nhận tại [supervisor.py L121](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L121):
```python
# See also: resolve_doc_agent._DOC_TYPE_KEYWORDS — should be unified in P1-6.
```

`_NAMED_DOC_PATTERN` (L122-128) và `_DOC_TYPE_KEYWORDS` (resolve_doc_agent.py L33) vẫn tách biệt. Plan đề xuất tạo `doc_keywords.py` — file này chưa tồn tại.

> [!WARNING]
> Rủi ro: 2 regex/keyword lists không đồng bộ → supervisor nhận diện doc khác resolve_doc_agent → routing sai. Ví dụ: supervisor regex có `"pháp lệnh"` nhưng resolve_doc_agent có thể không, hoặc ngược lại.

**Đánh giá plan:** ✅ Đúng, cần làm. Effort 2h hợp lý — cần refactor cả 2 files + thêm tests.

---

### A7: Đổi `_INTENT_TO_AGENT_FALLBACK` dùng `Intent.*` constants — ✅ ĐÃ FIX

[supervisor.py L84-104](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L84-L104) đã dùng `Intent.*` và `AgentType.*` constants:
```python
_INTENT_TO_AGENT_FALLBACK: dict[str, str] = {
    Intent.GREETING:           AgentType.DIRECT,
    Intent.PERSONAL:           AgentType.DIRECT,
    Intent.SEARCH:             AgentType.RAG,
    ...
}
```

**Đánh giá plan:** ✅ Task này đã hoàn thành, nên **bỏ khỏi plan** hoặc mark done.

---

### A8: Bổ sung `_INTENT_NORMALIZE` mapping — ✅ ĐÃ FIX

[supervisor.py L47-80](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L47-L80): `_INTENT_NORMALIZE` đã có **34 entries** covering:
- People variants: `search_phone`, `find_person`, `find_by_name`, etc.
- RAG variants: `query_doc`, `find_doc`, `lookup_doc`, `resolve_document`
- Section variants: `lookup_section`, `get_article`, `get_section`
- Summarize variants: `summarize_doc`, `doc_summary`

**Đánh giá plan:** ✅ Task này đã hoàn thành, nên **bỏ khỏi plan** hoặc mark done.

---

### A9: Tạo `clarification.py` — unified `ask_user_clarification()` — ✅ ĐÃ LÀM

[clarification.py](file:///home/AIRAG/backend/app/services/agents/clarification.py) (109 dòng) đã tồn tại với:
- `ask_user_clarification()` (L40-81) — emit SSE event `clarification`
- `should_ask_for_doc_reference()` (L84-98) — heuristic cho doc reference
- `should_ask_for_section_reference()` (L101-108) — heuristic cho section reference
- Đã được integrated vào supervisor_node (L805-822)

**Đánh giá plan:** ✅ Task này đã hoàn thành, nên **bỏ khỏi plan** hoặc mark done.

---

### A10: Chuyển regex compile ra module-level — ✅ ĐÃ FIX

[supervisor.py L107-128](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L107-L128):
```python
# =============================================================================
# Module-level compiled regex patterns (P1-7: avoid recompile per call)
# =============================================================================
_PERSONAL_REF_PATTERN: re.Pattern[str] = re.compile(...)
_NAMED_DOC_PATTERN: re.Pattern[str] = re.compile(...)
```
Comment P1-7 reference confirms đây đã được fix theo plan.

**Đánh giá plan:** ✅ Task này đã hoàn thành, nên **bỏ khỏi plan** hoặc mark done.

---

### Tổng kết Phase A

| # | Task | Status | Notes |
|---|------|--------|-------|
| A1 | Typo BHXH | ✅ **Done** | Bỏ khỏi plan |
| A2 | Duplicate keys / JSON comma | ⚠️ **Duplicate** | Gộp với Prompt Step 1 |
| A3 | `_re` NameError | ✅ **Done** | Bỏ khỏi plan |
| A4 | Comment "9B" | ❓ **Unclear** | Cần verify lại |
| A5 | `mentions_specific_doc` | ✅ **Done** | Test coverage đã có |
| A6 | `_NAMED_DOC_PATTERN` unification | ⚠️ **CHƯA LÀM** | Cần tạo `doc_keywords.py` |
| A7 | `Intent.*` constants | ✅ **Done** | Bỏ khỏi plan |
| A8 | `_INTENT_NORMALIZE` | ✅ **Done** | 34 entries |
| A9 | `clarification.py` | ✅ **Done** | 109 dòng, integrated |
| A10 | Module-level regex | ✅ **Done** | Comment P1-7 confirms |

> [!IMPORTANT]
> **7/10 tasks đã hoàn thành.** Chỉ còn A6 (doc keywords unification) là task thực sự cần làm. A2 duplicate với Prompt Step 1. A4 cần verify. Plan nên được cleanup để phản ánh thực tế.

---

## Phase E — Testing (TRƯỚC refactor, 1-1.5 ngày)

### E1: `test_langgraph_routing.py` — ✅ ĐÃ LÀM (partial)

[test_langgraph_routing.py](file:///home/AIRAG/backend/tests/agents/test_langgraph_routing.py) (273 lines) đã có **10 test classes**:

| Test Class | Tests | Coverage |
|-----------|-------|----------|
| `TestHappyPath` | 5 | search, greeting, people, write, finish |
| `TestIntentNormalization` | 5 | search_phone, find_person, query_doc, get_article, summarize_doc |
| `TestInvalidAgentCorrection` | 3 | intent-as-agent, unknown agent, finish passthrough |
| `TestIntentAgentOverride` | 3 | wrong agent for known intent |
| `TestTaskPlan` | 4 | single/multi-step, plan[0] override, empty plan |
| `TestMalformedJSON` | 3 | garbage, empty, truncated |
| `TestMarkdownStripping` | 4 | json fence, generic fence, think tags |
| `TestBackwardCompat` | 1 | mentions_specific_doc ignored |
| `TestPersonalIntent` | 1 | personal → DIRECT |
| `TestNeedsMemory` | 2 | true passes through, default false |

**Còn thiếu:**
- `is_legal_query` passthrough test
- `resolve_doc` intent routing test (route to `AgentType.RESOLVE_DOC`)
- `MONGO_SEARCH_ADVANCED` intent test
- `write_format_check` intent test
- Edge case: `next_agent="resolve_doc"` (not in `valid_agents` set — see Bug B1 below)

**Đánh giá plan:** ⚠️ Effort 4h ghi trong plan là hợp lý cho coverage đầy đủ, nhưng plan nên acknowledge rằng đã có 30+ tests.

---

### E2: `test_abbreviation_expansion.py` — ✅ ĐÃ LÀM

File [test_abbreviation_expansion.py](file:///home/AIRAG/backend/tests/agents/test_abbreviation_expansion.py) (10,936 bytes) đã tồn tại.

**Đánh giá plan:** ✅ Đã hoàn thành. Nên verify coverage (là task riêng, effort thấp).

---

### E3-E5: Chưa verify

Không thấy `test_resolve_doc_regex.py`, `test_section_extraction.py`, integration test file trong `/backend/tests/agents/`.

**Đánh giá plan:** ✅ Cần làm. Effort estimate hợp lý.

---

## Phase B — Structural Refactor (2-3 ngày)

### B1-B7: Tách supervisor.py (2118 dòng) → <800 dòng

Phân tích code hiện tại cho thấy supervisor.py có cấu trúc rõ ràng nhưng **quá dài**:

| Section | Lines | Est. % | Tách thành |
|---------|-------|--------|-----------|
| Imports + constants + regex | 1-155 | 7% | `intent_utils.py` + `abbreviation.py` (partial) |
| `_is_likely_abbreviation()` + `_expand_abbreviations_in_message()` | 158-379 | 10% | `abbreviation.py` ✅ |
| `_parse_supervisor_response()` | 380-490 | 5% | `intent_utils.py` ✅ |
| `query_analyzer_node()` | 498-574 | 4% | Giữ hoặc tách `nodes/query_analyzer.py` |
| `supervisor_node()` | 577-1043 | 22% | **Lớn nhất** — cần tách safety nets ra |
| `direct_answer_node()` | 1046-1112 | 3% | `nodes/direct.py` |
| `result_evaluator_node()` | 1120-1231 | 5% | `nodes/evaluator.py` |
| `answer_generator_node()` | 1295-1328 | 2% | `nodes/answer_generator.py` |
| `mongo_formatter_node()` | 1335-1397 | 3% | `nodes/mongo_formatter.py` |
| Routing functions (5 functions) | 1400-1628 | 11% | `routing.py` ✅ |
| `create_supervisor_graph()` + singleton | 1634-1848 | 10% | Giữ in supervisor.py |
| Agent wrappers (5 wrappers) | 1850-2118 | 13% | `wrappers.py` ✅ |

> [!WARNING]
> **`supervisor_node()` chiếm 466 dòng (22%)** — function dài nhất trong file. Nội bộ có:
> - Abbreviation expansion (L643-689)
> - LLM classifier call (L692-751)
> - Safety nets: needs_memory (L754-760), keyword override direct→rag (L762-776), prerequisite check (L778-800), doc reference fallback (L801-822)
> - Phase 5 multi-step routing (L870-921)
> - Thinking decision (L940-965)
> - Search mode selection (L967-978)
> - Langfuse span (L991-1028)
>
> Các safety nets (L754-822) nên tách ra `safety_nets.py` — plan đúng.

### Đánh giá chi tiết từng B task

| # | Proposed File | Plan Đúng? | Notes |
|---|------|------|-------|
| B1 | `abbreviation.py` | ✅ | ~222 dòng (`_is_likely_abbreviation` + `_expand_abbreviations` + `_disambiguate_multi_meaning_abbrs` + constants `_VI_STOP_WORDS`, `_VI_VOWELS`) |
| B2 | `intent_utils.py` | ✅ | ~140 dòng (`_INTENT_NORMALIZE`, `_INTENT_TO_AGENT_FALLBACK`, `_parse_supervisor_response`) |
| B3 | `wrappers.py` | ✅ | ~270 dòng (5 wrappers: memory_recall, query_enricher, rag, resolve_doc, write, people) |
| B4 | `safety_nets.py` | ✅ | ~70 dòng (needs_memory override, keyword safety net, prerequisite check) — but only 3 nets, not 6-7 |
| B5 | `routing.py` | ✅ | ~230 dòng (route_from_supervisor, route_from_rag, route_from_resolve_doc, route_from_evaluator, route_from_enricher) |
| B6 | `nodes/` package | ✅ | ~330 dòng (direct_answer_node, result_evaluator_node, answer_generator_node, mongo_formatter_node) |
| B7 | `supervisor.py` re-export | ✅ | Chỉ giữ supervisor_node + create_supervisor_graph + singleton |

**Ước tính sau refactor:** supervisor.py ≈ 550-650 dòng (supervisor_node + graph builder + singleton). Target <800 **khả thi**.

> [!IMPORTANT]
> **Rủi ro lớn nhất:** `supervisor_node()` 466 dòng. Ngay cả sau khi tách safety nets ra, nó vẫn ~350 dòng (abbreviation call, LLM call, Phase 5 multi-step, thinking decision, search mode, Langfuse). Đây là function có nhiều **side-effects qua state mutation** → cần test coverage trước khi tách (= Phase E phải hoàn thành trước).

**Đánh giá plan:**
- ✅ Thứ tự B1→B7 hợp lý (coupling thấp→cao)
- ✅ Effort 2-3 ngày realistik
- ⚠️ Plan nên explicit ghi: **Phase E PHẢI hoàn thành trước Phase B** (đã ghi đúng thứ tự trong plan nhưng cần nhấn mạnh dependency)
- ⚠️ B4 ghi "6-7 lớp safety nets" nhưng code thực tế chỉ có **3 safety nets** (needs_memory L754, keyword L762, prerequisite L778) + clarification fallback (L801) = **4 lớp**. Effort 3h → nên **1.5h**.

---

## Phase C — Logic Fixes (1.5-2 ngày)

### C1: Fix `_extract_section_from_markdown` cho "Khoản X Điều Y" — ✅ Cần thiết

Code tại [rag_agent.py L546-639](file:///home/AIRAG/backend/app/services/agents/rag_agent.py#L546-L639) hiện tại chỉ handle `Điều X`, `Chương Y` patterns. **Thiếu:**

1. **"Khoản 3 Điều 5"** — cần extract Điều 5 TRƯỚC, rồi tìm Khoản 3 bên trong
2. **"Điểm a Khoản 2 Điều 7"** — nesting 3 cấp
3. Fallback clarification khi section không tìm thấy — `should_ask_for_section_reference()` đã có helper nhưng chưa được gọi trong `_tool_search_section()`

```python
# rag_agent.py L757-761: chỉ return message, KHÔNG gọi clarification
else:
    return {
        "text": f"Không tìm thấy nội dung điều/khoản '{section_reference}' trong tài liệu.",
        "sources": []
    }
```

**Đánh giá plan:** ✅ Cần thiết. Effort 3h hợp lý — regex phức tạp cho Vietnamese legal section hierarchy.

---

### C2: Fix Phase 5 multi-step respect prerequisite task_plan[0]=resolve_doc — ⚠️ Partly done

Code tại [supervisor.py L877-921](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L877-L921) đã handle multi-step:
```python
if complexity != "simple" and sub_queries and current_step < len(sub_queries):
    current_sq = sub_queries[current_step]
    sq_intent = current_sq.get("intent_hint", "search")
    ...
    result["intent"] = sq_intent
    result["next_agent"] = sq_agent
```

**Nhưng vấn đề:** Phase 5 multi-step routing **override** Phase 4 task_plan. Nếu query_analyzer trả về `sub_queries` VÀ supervisor trả về `task_plan`, Phase 5 code chiến thắng (L877-904 runs trước L866-867). Nghĩa là `task_plan` từ supervisor bị **bỏ qua**.

Ngoài ra, `result_evaluator_node` (L1162) chỉ advance step nếu `complexity != "simple"`. Nếu query_analyzer classify sai (simple → multi), multi-step không fire.

**Đánh giá plan:** ⚠️ Plan ghi "Fix Phase 5 multi-step respect prerequisite task_plan[0]=resolve_doc" — đúng issue nhưng chưa specify giải pháp rõ ràng. Cần define: Phase 4 hay Phase 5 có priority? Đề xuất: **Phase 4 task_plan (supervisor) luôn override Phase 5 sub_queries (query_analyzer)** khi có conflict.

---

### C3: Tạo `set_intent()` helper — ✅ Cần thiết nhưng low priority

Intent được mutate tại **5 nơi** trong supervisor.py:
1. L463: `intent = task_plan[0]` (parse response)
2. L775: `decision["intent"] = "search"` (keyword safety net)
3. L794: `decision["intent"] = "resolve_doc"` (prerequisite check)
4. L895: `result["intent"] = sq_intent` (Phase 5 multi-step)
5. Implicitly via `_INTENT_NORMALIZE` (L417)

Và **2 nơi** ngoài supervisor:
- [rag_agent.py L779](file:///home/AIRAG/backend/app/services/agents/rag_agent.py#L779): `"intent": "summarize"` (map_search_section)
- resolve_doc_agent (implicit via state return)

`set_intent()` helper nên:
1. Normalize intent qua `_INTENT_NORMALIZE`
2. Re-compute `next_agent` qua `_INTENT_TO_AGENT_FALLBACK`
3. Log the mutation (cho Langfuse tracing)

**Đánh giá plan:** ✅ Correct. Effort 3h reasonable cho 7 call sites + tests.

---

### C4: Chuẩn hóa iteration counting — ⚠️ Cần fix

> [!CAUTION]
> **iterations được increment tại 2 nơi khác nhau**, gây double-counting.

Grep results cho `iterations + 1`:

| Location | File | When |
|----------|------|------|
| L601 | supervisor.py | supervisor_node empty message fallback |
| L848 | supervisor.py | supervisor_node loop-back with results |
| L859 | supervisor.py | supervisor_node normal return |
| L1042 | supervisor.py | supervisor_node exception fallback |
| **L855** | **rag_agent.py** | **rag_agent_node success** |
| **L868** | **rag_agent.py** | **rag_agent_node exception** |

**Vấn đề:** Cả supervisor_node VÀ rag_agent_node đều increment `iterations`. Mỗi supervisor→rag cycle tăng **2** thay vì **1**. Max iterations = 5 (default) nhưng effective max = **2.5 cycles** thay vì 5.

Resolve_doc_agent **không** increment → asymmetric.

**Đánh giá plan:** ✅ Cần fix ngay. Effort 2h hợp lý — cần remove increment trong rag_agent.py + verify toàn graph.

---

### C5: Tạo `@with_langfuse_span` decorator — ✅ Cần thiết, high ROI

Grep `langfuse.start_observation` trong supervisor.py trả về **15 occurrences**. Mỗi cái có cùng boilerplate pattern:

```python
if langfuse:
    try:
        obs = langfuse.start_observation(name=..., input={...}, level="DEFAULT")
        obs.update(output={...})
        obs.end()
    except Exception as e:
        logger.warning(f"[langfuse] ... span failed: {e}")
```

Trung bình **10-15 dòng mỗi occurrence** × 15 = **~150-225 dòng boilerplate** — chiếm ~10% file.

`rag_agent.py` đã có `_with_langfuse_span` helper (L38) cho async tool calls, nhưng routing functions trong supervisor.py chưa dùng nó.

**Đánh giá plan:** ✅ High ROI. Effort 3h cho decorator + refactor 15 call sites.

---

### Tổng kết Phase C

| # | Task | Status | Corrected Effort | Priority |
|---|------|--------|-----------------|----------|
| C1 | Fix section extraction | ⚠️ Chưa làm | 3h | P1 — user-facing |
| C2 | Phase 5 vs Phase 4 conflict | ⚠️ Partly done, cần design decision | 2h | P2 — correctness |
| C3 | `set_intent()` helper | ⚠️ Chưa làm | 3h | P2 — maintainability |
| C4 | Iteration counting | 🔴 **Bug** — double-counting | 2h → **1h** (simple fix) | **P0 — correctness** |
| C5 | Langfuse decorator | ⚠️ Chưa làm | 3h | P2 — code quality |

> [!CAUTION]
> **C4 nên được nâng lên P0.** Double iteration counting có nghĩa max_iterations=5 thực tế chỉ cho phép 2-3 cycles → multi-step queries có thể bị terminate sớm.

---

## Phase D — Prompt Polish

Phần lớn đã cover trong [prompt_review.md](file:///home/AIRAG/docs/prompt_review.md). Thêm phát hiện:

### D5: Tích hợp `ask_user_clarification` vào section/doc/clarification — ⚠️ Partly done

`clarification.py` đã có helpers nhưng integration chưa hoàn chỉnh:

- ✅ `supervisor_node` L805-822: Gọi `should_ask_for_doc_reference()` + `ask_user_clarification()` cho missing doc reference
- ❌ `rag_agent.py` L757-761: Section not found → return error text, **KHÔNG gọi** `should_ask_for_section_reference()` → user nhận "Không tìm thấy" thay vì được hỏi clarification
- ❌ `resolve_doc_agent.py`: Ambiguous doc (multiple candidates) → stream options **nhưng không dùng** `ask_user_clarification()` helper → inconsistent event format

**Đánh giá plan:** ⚠️ Effort 3h đúng nhưng plan nên list cụ thể 2 integration points còn thiếu.

---

## Section D — Key Architectural Decisions

### D.1 "Ask User When Uncertain" — ✅ Implemented

`clarification.py` + integration trong supervisor_node confirm nguyên tắc này đã có code backing.

### D.2 Intent Classification Architecture — ✅ Accurate

```
User Query → Query Analyzer (decompose) → Supervisor (classify+route)
                                         → Agent (rag/write/people/direct/resolve_doc)
                                         → Answer Generator (final answer)
                                           ↑
                                   Result Evaluator (retry fallback)
```

Diagram đúng. Nhưng có 1 flow bị thiếu:
- **rag → supervisor (abbreviation loop-back)** — `route_from_rag` trả về `"supervisor"` khi `should_loop_back=True`
- **rag → supervisor (search_section pending)** — `route_from_rag` trả về `"supervisor"` khi search_section + section_reference

Cả 2 flow này bypass result_evaluator → nên document rõ trong diagram.

### D.3 Resolution Strategies — ✅ Accurate nhưng thiếu Stage 0

```
Stage 0: Abbreviation expansion + disambiguation (supervisor_node, BEFORE resolve_doc)
Stage 1: Regex extraction + SQL query (0ms, no LLM)
Stage 2: LLM extraction fallback (Qwen3-4B, ~1-2s)
Stage 3: Vector search (last resort)
```

Plan ghi 3 stages, thực tế có 4 (Stage 0 implicit).

---

## Section F — Files Inventory

### Files sửa đổi — Accuracy check

| File | Plan | Status |
|------|------|--------|
| `supervisor.py` | tách lớn + fix logic | ⚠️ Chưa tách, đã fix nhiều logic |
| `models.py` | sửa typo | ✅ Done (BHXH already correct) |
| `rag_agent.py` | fix section extraction | ⚠️ Chưa fix |
| `people_agent.py` | bỏ dead code | ❓ Cần verify |
| `resolve_doc_agent.py` | share keywords | ⚠️ Chưa làm |
| `supervisor_prompt.py` | fix format + rút gọn | ⚠️ Chưa fix JSON comma |
| `query_analyzer_prompt.py` | bổ sung examples | ⚠️ Chưa làm |
| `answer_instructions.py` | thêm `_PERSONAL_INSTRUCTIONS` | ⚠️ Chưa làm |

### Files tạo mới — Accuracy check

| File | Plan | Status |
|------|------|--------|
| `nodes/` package | Chưa tạo | ❌ |
| `routing.py` | Chưa tạo | ❌ |
| `abbreviation.py` | Chưa tạo | ❌ |
| `safety_nets.py` | Chưa tạo | ❌ |
| `wrappers.py` | Chưa tạo | ❌ |
| `graph.py` | **KHÔNG CẦN** | 🔴 Plan sai — graph builder nên ở trong supervisor.py (backward compat) |
| `intent_utils.py` | Chưa tạo | ❌ |
| `doc_keywords.py` | Chưa tạo | ❌ |
| `clarification.py` | **ĐÃ TỒN TẠI** | ✅ 109 dòng |
| `mongo_formatter_prompt.py` | ⚠️ Inline prompt đã có | Prompt ở supervisor.py L1359-1372, tách ra file mới nếu muốn |
| `disambiguation_prompt.py` | Chưa tạo | ❌ |
| `resolve_doc_prompt.py` | Chưa tạo | ❌ |
| `memory_injection_prompt.py` | Chưa tạo | ❌ |
| `test_langgraph_routing.py` | **ĐÃ TỒN TẠI** | ✅ 273 dòng |
| `test_abbreviation_expansion.py` | **ĐÃ TỒN TẠI** | ✅ |
| `test_resolve_doc_regex.py` | Chưa tạo | ❌ |
| `test_section_extraction.py` | Chưa tạo | ❌ |
| `test_prompt_quality.py` | Chưa tạo | ❌ |

> [!WARNING]
> Plan liệt kê `graph.py` như file mới nhưng thực tế `create_supervisor_graph()` nên ở trong `supervisor.py` (hoặc file refactored mới). User Rules nói rõ: *"There is NO separate `graph.py` — the only graph builder is `create_supervisor_graph()` in `supervisor.py`."* Tạo `graph.py` mới sẽ mâu thuẫn với convention hiện tại.

---

## Section G — Codebase Stats

| File | Plan Lines | Actual Lines | Δ |
|------|-----------|-------------|---|
| `supervisor.py` | 2057 | **2118** | +61 (plan outdated) |
| `supervisor_prompt.py` | 334 | 334 | ✅ |
| `query_analyzer_prompt.py` | 100 | 101 | ~= |
| `rag_agent.py` | 869 | 869 | ✅ |
| `resolve_doc_agent.py` | 874 | 874 | ✅ |
| `write_agent.py` | 375 | — | Not verified |
| `people_agent.py` | 253 | 253 | ✅ |
| `models.py` | 236 | **237** | +1 |
| `write_agent_prompt.py` | 130 | 130 | ✅ |
| `answer_instructions.py` | 187 | 187 | ✅ |

`supervisor.py` đã tăng 61 dòng so với plan → plan nên cập nhật.

---

## Phát Hiện Bổ Sung — Plan Bỏ Sót

### Bug B1: `valid_agents` set thiếu `RESOLVE_DOC`

[supervisor.py L420-423](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L420-L423):
```python
valid_agents = {
    AgentType.RAG, AgentType.WRITE, AgentType.PEOPLE,
    AgentType.DIRECT, AgentType.FINISH,
}
```

`AgentType.RESOLVE_DOC` **không có trong set**. Nếu LLM trả `next_agent="resolve_doc"`, code fall vào L424-431 correction path → tìm `_INTENT_TO_AGENT_FALLBACK.get("resolve_doc")` → không tìm thấy (key là intent name, không phải agent name) → default `AgentType.FINISH` (L436).

**Thực tế không gây lỗi** vì:
1. LLM không biết `resolve_doc` agent (thiếu trong prompt — đã cover trong prompt_review)
2. Intent→agent override (L438-449) sẽ correct nếu intent="resolve_doc"

Nhưng nếu thêm `resolve_doc` vào prompt (Step 2 trong prompt plan), LLM CÓ THỂ trả `next_agent="resolve_doc"` → **sẽ bị đánh rơi**.

> [!CAUTION]
> **Khi fix Prompt Step 2 (thêm resolve_doc vào AVAILABLE AGENTS), PHẢI đồng thời thêm `AgentType.RESOLVE_DOC` vào `valid_agents` set.** Nếu không, LLM route đúng nhưng code reject nó.

---

### Bug B2: `_DIRECT_MAP` duplicate definition

[supervisor.py L1428-1435](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L1428-L1435) và [L1472-1479](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L1472-L1479): `_DIRECT_MAP` được define **2 lần** trong cùng function `route_from_supervisor()`. Lần đầu ở nhánh if, lần hai ở nhánh else. Cả 2 giống nhau.

**Impact:** Không gây bug nhưng confusing. Refactor B5 (routing.py) sẽ fix tự nhiên bằng cách extract ra module-level constant.

---

### Bug B3: `route_from_enricher()` defined INSIDE `create_supervisor_graph()`

[supervisor.py L1705-1756](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L1705-L1756): Function `route_from_enricher()` nằm **bên trong** `create_supervisor_graph()` body. Tất cả routing functions khác (`route_from_supervisor`, `route_from_rag`, `route_from_resolve_doc`, `route_from_evaluator`) đều ở **module level**.

**Impact:** Inconsistent → harder to test. `route_from_enricher` không thể import riêng để unit test.

---

### Observation O1: `mongo_formatter_node` inline prompt nên tách

[supervisor.py L1359-1372](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L1359-L1372): `format_system` prompt (14 dòng) nằm inline trong `mongo_formatter_node`. Plan liệt kê `mongo_formatter_prompt.py` trong Files tạo mới nhưng không có task nào assign cho việc tạo file này.

---

### Observation O2: `INTENT_TO_AGENT` trong models.py thiếu `RESOLVE_DOC`

[models.py L74-93](file:///home/AIRAG/backend/app/services/agents/models.py#L74-L93): `INTENT_TO_AGENT` mapping (reference/documentation) thiếu `Intent.RESOLVE_DOC`. Mapping này comment ghi "không dùng trong runtime" nhưng sai lệch với `_INTENT_TO_AGENT_FALLBACK` (supervisor.py) có resolve_doc → gây confusion.

---

## Tổng Kết Đánh Giá Toàn Bộ Plan V2.2

### Completed Tasks (nên remove từ plan)

| # | Task | Evidence |
|---|------|----------|
| A1 | Typo BHXH | Grep 0 results for wrong case |
| A3 | `_re` NameError | All uses properly scoped |
| A5 | `mentions_specific_doc` | Code + test confirms removed |
| A7 | `Intent.*` constants | All entries use class constants |
| A8 | `_INTENT_NORMALIZE` | 34 entries |
| A9 | `clarification.py` | 109 lines, integrated |
| A10 | Module-level regex | Comment P1-7 confirms |
| E1 | test_langgraph_routing | 273 lines, 31 tests (partial) |
| E2 | test_abbreviation_expansion | 10.9KB file exists |

### New Bugs Found (not in plan)

| # | Bug | Severity | Fix |
|---|-----|----------|-----|
| B1 | `valid_agents` missing `RESOLVE_DOC` | **P1** — will break when Prompt Step 2 lands | Add to set (1 line) |
| B2 | `_DIRECT_MAP` duplicate | P3 — cosmetic | Dedup (Phase B fix) |
| B3 | `route_from_enricher` nested in graph builder | P3 — testability | Extract to module level (Phase B fix) |
| O2 | `INTENT_TO_AGENT` in models.py missing `RESOLVE_DOC` | P3 — documentation | Add entry |

### Effort Corrections

| Phase | Plan Effort | Corrected Effort | Reason |
|-------|-------------|-----------------|--------|
| A | 6-8h | **2-3h** | 7/10 tasks done, only A6 remains |
| B | 2-3 days | 2-3 days | ✅ Accurate |
| B4 | 3h | **1.5h** | Only 4 safety nets, not 6-7 |
| C | 1.5-2 days | 1.5-2 days | ✅ Accurate |
| C4 | 2h | **1h** | Simple fix: remove 2 lines in rag_agent |
| D | 1.5-2 days | **1-1.5 days** | Some D tasks already done |
| E | 1-1.5 days | **0.5-1 day** | E1, E2 already done |

### Khuyến Nghị Thứ Tự Thực Hiện (Updated)

```
1. [5m]   Prompt P0: Fix JSON comma (41 examples + template L333)
2. [5m]   Bug B1: Thêm AgentType.RESOLVE_DOC vào valid_agents (L420-423)
3. [5m]   Prompt 7.4: Thêm resolve_doc vào AVAILABLE AGENTS (đồng thời với B1)
4. [1h]   C4: Fix double iteration counting (remove increment in rag_agent.py)
5. [2h]   A6: Unify _NAMED_DOC_PATTERN + _DOC_TYPE_KEYWORDS → doc_keywords.py
6. [30m]  Prompt P3: _PERSONAL_INSTRUCTIONS + fix direct_answer_node
7. [45m]  Prompt P1: Tách inline prompts (3 file mới)
8. [3h]   E3-E5: Test resolve_doc regex, section extraction, integration
9. [2h]   D2+D4+D5: Examples + clarification integration
10. [2h]  Prompt P2: Giảm supervisor prompt 25-30%
11. [1h]  C4+C3: Iteration fix + set_intent() helper
12. [3h]  C5: Langfuse decorator
13. [3h]  C1: Fix section extraction "Khoản X Điều Y"
14. [2-3d] B1-B7: Structural refactor (AFTER all tests pass)
15. [2h]  Prompt P6: Test framework
```

> [!IMPORTANT]
> **Step 2 + 3 PHẢI đồng thời:** Nếu thêm `resolve_doc` vào prompt (Step 3) mà không thêm vào `valid_agents` (Step 2), routing sẽ break.

> [!IMPORTANT]
> **Step 4 (C4 iteration fix) nên ưu tiên cao** — hiện tại multi-step queries bị terminate sớm do double-counting.

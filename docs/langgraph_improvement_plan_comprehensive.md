# Plan Cải Thiện LangGraph Agent System — Tổng Hợp (V2.2)

> **Tổng hợp:** 2026-06-07
> **V2.2 cập nhật:** Sau prompt_review.md cross-reference + code verification
> **Nguồn gốc:**
> - V1: `langgraph_improvement_plan.md` (feature phases — đã implement xong Phase 1-5)
> - V2+.1: `langgraph_review_plan_v2.md` (technical debt cleanup, 8 P0 + 8 P1 + 11 P2)
> - Cross-review: `langgraph_review_plan_v2_evaluation.md` (5 MISSING items, adjusted phases E→trước B)
> - **Prompt review:** `docs/prompt_review.md` (cross-reference 6 prompts với code thực tế, corrections + findings bổ sung)
>
> **Trạng thái:** Phase 1-5 (feature) đã xong → tập trung V2/V2.1/Prompt-review cleanup

---

## A. Prompt-Specific Improvements (V2.2 — Sau Prompt Review)

> **[!IMPORTANT] Corrections từ prompt_review.md (code cross-verified):**
> - P2 target 40% → **25-30%** cho Qwen3 model (phụ thuộc examples nhiều hơn GPT-4)
> - D3: **KHÔNG** thể dùng computed field cho `is_legal_query` — cần giữ LLM flag vì safety net override (direct→rag) cần detect "Xin chào, Luật ANM nói gì?"
> - P3: `_PERSONAL_INSTRUCTIONS` **không inject qua `get_instructions_for_intent()`** — `direct_answer_node` (L1046-1112) dùng riêng system prompt, cần template mới
> - P1 effort: **30-45 phút** (không phải 15 phút) — cần tách 3 inline prompts + fix 3 callers
> - **Bỏ sót trong plan gốc:** `resolve_doc` agent thiếu trong AVAILABLE AGENTS section của supervisor_prompt
> - **Bỏ sót:** `needs_memory` cũng là dead output (có regex safety net thay thế)
> - **Bỏ sót:** Memory injection prompt trong `direct_answer_node` (L1080-1088) nên tách ra file riêng

### Execution Order Prompt Tasks (9 steps, coordinated)

| Step | Task | Files | Effort | Notes |
|------|------|-------|--------|-------|
| **1** | **P0: Fix JSON comma** — tất cả 41 examples + template L333 | `supervisor_prompt.py` | 5 phút | **Critical** — invalid JSON trong output format |
| **2** | **7.4: Thêm `resolve_doc` agent** vào AVAILABLE AGENTS section | `supervisor_prompt.py` | 5 phút | LLM không biết agent này → luôn route qua rag |
| **3** | **P3: _PERSONAL_INSTRUCTIONS + direct_answer_node** — template + inject | `answer_instructions.py`, `supervisor.py` | 30 phút | Không qua `get_instructions_for_intent()` — inject trực tiếp vào `direct_answer_node` |
| **4** | **P1: Tách inline prompts** — disambiguation, resolve_doc, memory injection (3 file mới) | **3 new files**, `supervisor.py`, `resolve_doc_agent.py` | 45 phút | Effort tăng từ 15→45 phút theo code review thực tế |
| **5** | **D2+D4: Thêm examples** — query_analyzer + supervisor (phối hợp) | 2 prompt files | 2 giờ | Thêm có giá trị TRƯỚC khi cắt redundant |
| **6** | **P2: Giảm supervisor prompt** — verbatim → patterns (25-30% reduction) | `supervisor_prompt.py` | 2 giờ | Target 25-30%, không phải 40% |
| **7** | **D3: is_legal_query** — GIỮ làm LLM flag (KHÔNG computed field) cho safety net override | — | 0 phút | Corrected: không làm computed field, giữ nguyên |
| **8** | **P4: Improve FALLBACK_STANDARD** — bổ sung page size, header/footer rules | `write_agent_prompt.py` | 5 phút | Low risk, quick win |
| **9** | **P6: Test framework** — JSON validity, intent coverage, routing consistency, token counting | **new test file** | 2 giờ | See detailed spec below |

### Prompt Issues Found in Code Review (Cross-Verified)

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| JSON comma error | 41 examples + template L333 thiếu comma → invalid JSON | **P0 Critical** | Step 1 |
| resolve_doc missing | AVAILABLE AGENTS liệt kê 5 agent, thiếu `resolve_doc` | **P1** | Step 2 |
| is_legal_query computed field | **KHÔNG ĐƯỢC** dùng computed — cần cho safety net (direct→rag override tại L765-776) | **Corrected** | Skip |
| Personal instructions | `_PERSONAL_INSTRUCTIONS` không dùng được qua `get_instructions_for_intent()` — `direct_answer_node` không gọi function này | **P1** | Step 3 |
| needs_memory dead output | LLM output field bị override bởi `_PERSONAL_REF_PATTERN` regex — redundant | **P2** | Optional |
| Memory injection inline | L1080-1088 hardcode memory injection block trong `direct_answer_node` | **P1** | Step 4 (part of inline prompts) |
| Query analyzer examples | 7 examples, 0 cho `write_*`, `personal`, section non-named, multi-field | **P2** | Step 5 |
| Supervisor examples density | 52% token cho examples, target giảm 25-30% cho Qwen3 | **P2** | Step 6 |

---

## B. Plan V2.1 — Technical Debt Cleanup

### Phase A — Quick Wins (6-8 giờ)

| # | Task | File | Effort | Notes |
|---|------|------|--------|-------|
| A1 | Fix typo `MONGO_SEARCH_BHxh` → `MONGO_SEARCH_BHXH` | `models.py` | 5 phút | P0-1 |
| A2 | Fix 3 duplicate keys trong supervisor_prompt.py:333 | `supervisor_prompt.py` | 15 phút | P0-2 (covers prompt P0) |
| A3 | **🔴 Fix latent NameError**: `_re` chưa import tại L712, L748 | `supervisor.py` | 30 phút | **P0-7** |
| A4 | Fix comment sai "9B" → "4B" tại L479 | `supervisor.py` | 5 phút | P0-8 |
| A5 | Bỏ `mentions_specific_doc` khỏi prompt + parse | `supervisor.py`, `supervisor_prompt.py` | 30 phút | P0-3, Phương án A |
| A6 | Mở rộng `_NAMED_DOC_PATTERN` + share keywords với resolve_doc | `supervisor.py`, `resolve_doc_agent.py` | 2 giờ | P0-4 + clarify principle |
| A7 | Đổi `_INTENT_TO_AGENT_FALLBACK` dùng `Intent.*` constants | `supervisor.py` | 30 phút | P0-5 |
| A8 | Bổ sung `_INTENT_NORMALIZE` mapping (chỉ theo Langfuse data) | `supervisor.py` | 30 phút | P0-6 |
| A9 | Tạo `clarification.py` — unified `ask_user_clarification()` | **new file** | 1 giờ | Clarification principle |
| A10 | Chuyển regex compile ra module-level | `supervisor.py` | 15 phút | P1-7 |

### Phase E — Testing (TRƯỚC refactor, 1-1.5 ngày)

| # | Task | File | Effort |
|---|------|------|--------|
| E1 | `test_langgraph_routing.py` — test `_parse_supervisor_response` | **new file** | 4 giờ |
| E2 | `test_abbreviation_expansion.py` — test `_is_likely_abbreviation` | **new file** | 2 giờ |
| E3 | `test_resolve_doc_regex.py` — test `_extract_by_regex` | **new file** | 1 giờ |
| E4 | `test_section_extraction.py` — test `_extract_section_from_markdown` | **new file** | 1 giờ |
| E5 | Integration test mock full graph flow (10 scenario) | **new file** | 2 giờ |

### Phase B — Structural Refactor (2-3 ngày)

Tách `supervisor.py` (2057 dòng) → < 800 dòng, coupling thấp→cao:

| # | File mới | Effort |
|---|----------|--------|
| B1 | `abbreviation.py` — heuristic + expand + disambiguate | 2 giờ |
| B2 | `intent_utils.py` — normalize, fallback, parse | 2 giờ |
| B3 | `wrappers.py` — memory_recall, query_enricher wrappers | 1.5 giờ |
| B4 | `safety_nets.py` — `apply_deterministic_overrides()` gom 6-7 lớp | 3 giờ |
| B5 | `routing.py` — route_from_supervisor/rag/resolve_doc/evaluator/enricher | 2 giờ |
| B6 | `nodes/` package — 5-6 file nodes (coupling cao nhất, tách cuối) | 6-8 giờ |
| B7 | `supervisor.py` chỉ còn re-export (backward compat) | 30 phút |

### Phase C — Logic Fixes (1.5-2 ngày)

| # | Task | Effort |
|---|------|--------|
| C1 | Fix `_extract_section_from_markdown` cho "Khoản X Điều Y" + fallback clarify | 3 giờ |
| C2 | Fix Phase 5 multi-step respect prerequisite task_plan[0]=resolve_doc | 2 giờ |
| C3 | Tạo `set_intent()` helper, refactor 5+ nơi intent mutation | 3 giờ |
| C4 | Chuẩn hóa iteration counting — chỉ supervisor increment | 2 giờ |
| C5 | Tạo Langfuse `@with_langfuse_span` decorator, refactor 15+ nơi | 3 giờ |

### Phase D — Prompt Polish (1.5-2 ngày)

| # | Task | Effort | Status |
|---|------|--------|--------|
| D1 | Thêm `resolve_doc` agent vào AVAILABLE AGENTS | 5 phút | ✅ Prompt Step 2 |
| D2 | Bổ sung ≥ 12 examples cho query_analyzer | 2 giờ | ✅ Prompt Step 5 |
| D3 | **KEEP `is_legal_query` as LLM flag** (NOT computed field) | 0 phút | ✅ Corrected — kept |
| D4 | Bổ sung 5-8 examples quan trọng cho supervisor | 2 giờ | ✅ Prompt Step 5 |
| D5 | Tích hợp `ask_user_clarification` vào section/doc/clarification | 3 giờ | V2.1 feature |
| D6 | Rút gọn supervisor_prompt 25-30% (sau thêm examples) | 2 giờ | ✅ Prompt Step 6 |
| D7 | Add `_PERSONAL_INSTRUCTIONS` + fix `direct_answer_node` | 30 phút | ✅ Prompt Step 3 |
| D8 | Improve FALLBACK_STANDARD cho format check | 5 phút | ✅ Prompt Step 8 |
| D9 | Test framework: JSON validity, intent coverage, routing, tokens | 2 giờ | ✅ Prompt Step 9 |

> [!NOTE]
> D3 was corrected: `is_legal_query` **must stay** as LLM-derived flag for safety net override (L765-776). Computing it from intent would miss "Xin chào, Luật ANM nói gì?" → routing would stay `direct/greeting` instead of `rag/search`.

---

## C. Prompt-Specific Test Framework (P6/D9 — Detailed Scope)

From prompt_review.md findings, test framework should cover:

| Test Type | What It Checks | Threshold |
|-----------|---------------|-----------|
| **JSON validity** | Parse ALL examples in supervisor_prompt + query_analyzer → `json.loads()` | 100% must parse |
| **Intent coverage** | Each `Intent` constant has ≥1 example in supervisor_prompt + query_analyzer | ≥12 intents covered |
| **Routing consistency** | Example `next_agent` + `intent` match `_INTENT_TO_AGENT_FALLBACK` | 100% consistent |
| **Token counting** | Count tokens of each prompt, warn if > threshold | supervisor < 20KB, query_analyzer < 8KB |
| **Format template** | L333 output format template is valid JSON | Must parse |

---

## D. Key Architectural Decisions

### 1. Nguyên tắc "Ask User When Uncertain"
- Regex/heuristic chỉ dùng khi accuracy cao
- Khi confidence < 0.5 → hỏi user thay vì đoán
- Áp dụng: abbreviation disambiguation, doc reference, section reference

### 2. Intent Classification Architecture
```
User Query → Query Analyzer (decompose) → Supervisor (classify+route)
                                         → Agent (rag/write/people/direct/resolve_doc)
                                         → Answer Generator (final answer)
                                           ↑
                                   Result Evaluator (retry fallback)
```

### 3. Resolution Strategies (resolve_doc)
```
Stage 1: Regex extraction + SQL query (0ms, no LLM)
  → Early exit if confidence ≥ 0.85
Stage 2: LLM extraction fallback (Qwen3-4B, ~1-2s)
  → Only if DB returns 0 results
Stage 3: Vector search (last resort)
  → Only if Stages 1+2 return 0 results
```

---

## E. Definition of Done

Mỗi task:
- [ ] Có test pass
- [ ] Không break existing tests
- [ ] Không tăng latency > 100ms
- [ ] Log rõ ràng (Langfuse span cho routing)

Sau toàn bộ Phase A-E:
- [ ] `supervisor.py` < 800 dòng
- [ ] 0 typo constants, 0 dead code, 0 latent bugs
- [ ] 0 nơi increment `iterations` ngoài supervisor
- [ ] ≥ 80% routing coverage, 100% `_parse_supervisor_response`
- [ ] JSON examples 100% parseable
- [ ] ≥ 12 intents có examples trong supervisor + query_analyzer
- [ ] Manual test 50+ query không regression
- [ ] `ask_user_clarification` flow hoạt động end-to-end

---

## F. Files Inventory

### Files sửa đổi
- `backend/app/services/agents/supervisor.py` — tách lớn + fix logic
- `backend/app/services/agents/models.py` — sửa typo
- `backend/app/services/agents/rag_agent.py` — fix section extraction
- `backend/app/services/agents/people_agent.py` — bỏ dead code
- `backend/app/services/agents/resolve_doc_agent.py` — share keywords
- `backend/app/prompts/agents/supervisor_prompt.py` — fix format + rút gọn
- `backend/app/prompts/agents/query_analyzer_prompt.py` — bổ sung examples
- `backend/app/prompts/agents/answer_instructions.py` — thêm `_PERSONAL_INSTRUCTIONS`

### Files tạo mới
- `backend/app/services/agents/nodes/` — package
- `backend/app/services/agents/routing.py`
- `backend/app/services/agents/abbreviation.py`
- `backend/app/services/agents/safety_nets.py`
- `backend/app/services/agents/wrappers.py`
- `backend/app/services/agents/graph.py`
- `backend/app/services/agents/intent_utils.py`
- `backend/app/services/agents/doc_keywords.py`
- `backend/app/services/agents/clarification.py`
- `backend/app/prompts/agents/mongo_formatter_prompt.py`
- `backend/app/prompts/agents/disambiguation_prompt.py` — extracted from supervisor.py
- `backend/app/prompts/agents/resolve_doc_prompt.py` — extracted from resolve_doc_agent.py
- `backend/app/prompts/agents/memory_injection_prompt.py` — extracted from direct_answer_node
- `backend/tests/test_langgraph_routing.py`
- `backend/tests/test_abbreviation_expansion.py`
- `backend/tests/test_resolve_doc_regex.py`
- `backend/tests/test_section_extraction.py`
- `backend/tests/test_prompt_quality.py` — NEW

---

## G. Codebase Stats

| File | Lines | Role |
|------|-------|------|
| `supervisor.py` | 2057 | Main graph + nodes + routing + abbreviation |
| `supervisor_prompt.py` | 334 | System prompt |
| `query_analyzer_prompt.py` | 100 | Query decomposer prompt |
| `rag_agent.py` | 869 | RAG operations |
| `resolve_doc_agent.py` | 874 | Document resolution |
| `write_agent.py` | 375 | Write operations |
| `people_agent.py` | 253 | MongoDB people search |
| `models.py` | 236 | State, Intent, AgentType |
| `write_agent_prompt.py` | 130 | Write prompts + standards |
| `answer_instructions.py` | 187 | Modular answer instructions |

**Tổng:** ~5,200 lines code + ~600 lines prompts + ~130 lines instructions

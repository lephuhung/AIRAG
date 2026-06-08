# Review: Prompt Improvements trong LangGraph Agent System

> **Scope:** Đánh giá phần prompt-related items trong [langgraph_improvement_plan_comprehensive.md](file:///home/AIRAG/docs/langgraph_improvement_plan_comprehensive.md), cross-reference với code thực tế.
>
> **Files reviewed:**
> - [supervisor_prompt.py](file:///home/AIRAG/backend/app/prompts/agents/supervisor_prompt.py) (334 lines, ~24KB)
> - [query_analyzer_prompt.py](file:///home/AIRAG/backend/app/prompts/agents/query_analyzer_prompt.py) (101 lines, ~6KB)
> - [answer_instructions.py](file:///home/AIRAG/backend/app/prompts/agents/answer_instructions.py) (187 lines, ~10KB)
> - [write_agent_prompt.py](file:///home/AIRAG/backend/app/prompts/agents/write_agent_prompt.py) (130 lines, ~4KB)
> - [supervisor.py](file:///home/AIRAG/backend/app/services/agents/supervisor.py) (inline prompts: L307-312, L319)
> - [resolve_doc_agent.py](file:///home/AIRAG/backend/app/services/agents/resolve_doc_agent.py) (inline prompt: L451-484)

---

## 1. P0 — JSON Syntax Error (CRITICAL) ✅ Plan chính xác

> [!CAUTION]
> **Tất cả 41 JSON examples trong supervisor_prompt.py đều thiếu dấu phẩy** giữa `is_legal_query` và `reasoning`.

**Bằng chứng:**
```
// Dòng 183, 186, 189, 192, 197, 200, ... 322, 333 — tất cả đều có pattern:
"is_legal_query":false"reasoning":"..."
//                    ^ THIẾU DẤU PHẨY TẠI ĐÂY
```

**Impact:** Đây là **invalid JSON** trong mọi example. LLM học từ examples → output cũng có thể thiếu comma → `_parse_supervisor_response()` sẽ `json.JSONDecodeError` → fallback sang `AgentType.FINISH` (L484-490), tức là **mọi query đều bị DROP**.

**Thực tế:** Hệ thống vẫn chạy được vì:
1. LLM không copy verbatim — nó "hiểu" ý tứ và tự format lại
2. `_parse_supervisor_response()` đã có fallback (L484-490)
3. Nhưng vẫn tăng tỷ lệ parse fail → giảm routing accuracy

**Fix (5 phút):** `sed -i` regex thay `false"reasoning"` → `false,"reasoning"` và `true"reasoning"` → `true,"reasoning"`. Cả file output format template ở L333 cũng bị.

**Đánh giá plan:** ✅ Severity đúng (Critical), effort đúng (5 phút).

---

## 2. P1 — Tách Inline Prompts ra File Riêng ✅ Plan hợp lý, cần bổ sung

Plan đề xuất tách inline prompts cho `resolve_doc` và `disambiguation`. Kiểm tra thực tế:

### 2.1 Disambiguation Prompt (supervisor.py L307-312, L319)

```python
# L307-312: Inline prompt string
prompt = (
    f'Từ viết tắt "{abbr}" có các nghĩa sau:\n{meanings_text}\n\n'
    f'Câu hỏi của user: "{user_message}"\n\n'
    ...
)
# L319: Inline system_prompt
system_prompt="You are a Vietnamese abbreviation disambiguation assistant. Output valid JSON only.",
```

**Vấn đề:**
- System prompt chỉ 1 dòng tiếng Anh, trong khi user prompt tiếng Việt → không nhất quán
- Không có examples → LLM phải "đoán" output format
- Không có fallback pattern nếu LLM output sai format

### 2.2 Resolve Doc LLM Prompt (resolve_doc_agent.py L451-484)

```python
# 34-line inline prompt bên trong hàm _extract_by_llm()
prompt = (
    "Bạn là chuyên gia tìm kiếm văn bản pháp luật Việt Nam.\n"
    ...
)
```

**Vấn đề:**
- Prompt này dài 34 dòng, nằm inline trong function body → khó maintain
- Chứa database schema, examples, và rules — nên ở file riêng
- Nhưng nó chỉ được gọi khi regex + DB fail → **low frequency**, tách riêng là nice-to-have

### 2.3 Thiếu trong plan: Answer generator system prompt

Plan bỏ sót inline prompt tại [supervisor.py L1080-1088](file:///home/AIRAG/backend/app/services/agents/supervisor.py#L1080-L1088):
```python
effective_system = (
    f"{user_memory}\n\n"
    "IMPORTANT: Do NOT copy these facts directly. When using a memory fact, "
    "paraphrase it in your own words and cite it as [MEM-1], [MEM-2], etc. ..."
) + effective_system
```
Đoạn memory injection instruction này nên nằm trong `answer_instructions.py` thay vì hardcode.

**Đánh giá plan:** ⚠️ Đúng hướng nhưng effort ước lượng thấp (15 phút cho tất cả → nên 30-45 phút), và bỏ sót memory injection prompt.

---

## 3. P2 — Giảm Supervisor Prompt ~40% ⚠️ Plan khả thi nhưng cần cẩn thận

### Phân tích kích thước hiện tại

| Section | Lines | Est. Tokens | % |
|---------|-------|-------------|---|
| Header + Agent list | 1-28 | ~200 | 3% |
| Intent Taxonomy | 29-113 | ~1,200 | 18% |
| Task Planning Rules | 114-141 | ~400 | 6% |
| Disambiguation Rules ①-⑧ | 142-174 | ~600 | 9% |
| **Routing Examples** | **175-323** | **~3,500** | **52%** |
| Loop Rules + Output format | 324-334 | ~200 | 3% |
| `{analyzer_context}` (runtime) | — | ~200-500 | variable |
| **Tổng** | **334** | **~6,100** | **100%** |

**Routing Examples chiếm 52% token budget.** Plan đề xuất giảm 40% bằng cách chuyển verbatim examples → patterns. 

### Đánh giá

> [!WARNING]
> Giảm examples cho supervisor prompt cần thận trọng. Đây là model Qwen3-4B/35B, không phải GPT-4 — performance phụ thuộc nhiều vào few-shot examples hơn.

**Nên giữ:**
- Examples cho edge cases khó (disambiguation rules ①-⑧ → **essential**, không nên cắt)
- Ít nhất 2 examples cho mỗi intent group (hiện tại có 2-4 per group → giữ 2)
- Examples cho multi-step task_plan (resolve_doc + search/search_section/summarize)

**Có thể cắt:**
- Redundant examples cùng pattern. Ví dụ trong SEARCH group có 4 examples đều cùng output structure → giảm xuống 2
- Redundant resolve_doc+search examples (hiện 7 examples, pattern giống nhau) → giảm xuống 3
- People group (4 examples, mỗi cái rất predictable) → giảm xuống 2

**Ước tính giảm:** ~30-35% (1,000-1,200 tokens), không đạt 40% nếu muốn giữ quality.

**Đánh giá plan:** ⚠️ Mục tiêu 40% quá aggressive cho Qwen3 model. Đề xuất target 25-30%, keep edge case examples.

---

## 4. P3 — `_PERSONAL_INSTRUCTIONS` cho answer_instructions.py ✅ Cần thiết

Hiện tại [answer_instructions.py](file:///home/AIRAG/backend/app/prompts/agents/answer_instructions.py) có:
- `_RAG_INSTRUCTIONS` ✅
- `_MONGO_INSTRUCTIONS` ✅  
- `_ABBR_INSTRUCTIONS` ✅
- `_LIST_DOCS_INSTRUCTIONS` ✅
- `_THINKING_DIRECTIVE` ✅

**Thiếu:** `_PERSONAL_INSTRUCTIONS` cho intent `personal` (thuộc DIRECT group).

Kiểm tra code: khi intent = `personal` hoặc `greeting`, flow đi qua `direct_answer_node` (L1042-1112). Node này dùng `system_prompt` từ state — **không gọi `get_instructions_for_intent()`**.

```python
# L1059: direct_answer_node lấy system_prompt từ state, KHÔNG dùng answer_instructions
system_prompt = state.get("system_prompt", "")
```

Vậy `_PERSONAL_INSTRUCTIONS` sẽ **không có tác dụng** nếu chỉ thêm vào `answer_instructions.py` mà không sửa `direct_answer_node` để gọi `get_instructions_for_intent()`.

**Đánh giá plan:** ⚠️ Task đúng nhưng thiếu phần sửa `direct_answer_node` → effort nên là 30 phút thay vì 20 phút.

---

## 5. P4 — Improve `FALLBACK_STANDARD` ✅ Đơn giản

Hiện tại [FALLBACK_STANDARD](file:///home/AIRAG/backend/app/prompts/agents/write_agent_prompt.py#L36-L43) chỉ 4 dòng:
```python
FALLBACK_STANDARD = """\
30/2020/NĐ-CP QUY ĐỊNH VỀ THỂ THỨC VĂN BẢN:

1. CĂN LỀ: Trên 2cm, Dưới 2cm, Trái 3cm, Phải 2cm
2. CỠ CHỮ: 13pt cho nội dung, 14pt cho tiêu đề
3. FONT: Times New Roman, Arial
4. KHOẢNG CÁCH DÒNG: 1.5 dòng
"""
```

Fallback chỉ dùng khi `_load_30_standard_from_file()` fail → load từ `docs/30-ND.md` không được.

**Đánh giá plan:** ✅ Đúng severity (Low), effort đúng (5 phút). Nên thêm 3-4 quy định quan trọng nữa (header spacing, footer, page numbering format).

---

## 6. P6 — Test Framework cho Prompt Quality ✅ Cần thiết

Plan không chi tiết lắm. Đề xuất cụ thể:

**Framework nên bao gồm:**
1. **JSON validity test** — parse tất cả examples trong supervisor_prompt → verify JSON syntax
2. **Intent coverage test** — mỗi intent trong `Intent` enum phải có ≥1 example
3. **Routing consistency test** — example outputs phải match `_INTENT_TO_AGENT_FALLBACK` mapping
4. **Token counting** — đếm token count của mỗi prompt, cảnh báo nếu > threshold

**Đánh giá plan:** ✅ Nhưng cần chi tiết hơn về scope và test types.

---

## 7. Findings Bổ Sung — Plan Bỏ Sót

### 7.1 Query Analyzer Prompt thiếu edge case examples

[query_analyzer_prompt.py](file:///home/AIRAG/backend/app/prompts/agents/query_analyzer_prompt.py) chỉ có 7 examples (L80-99). Plan (D2) đề xuất thêm ≥12 examples — **đúng nhưng cần chỉ rõ loại nào thiếu:**

Thiếu examples cho:
- `write_*` intents (write_summarize, write_grammar_check) → 0 examples
- `personal` intent → 0 examples
- `search_section` without named doc → 0 examples
- Multi-field person search → 0 examples
- Ambiguous cases (concept vs named doc: "Luật BMNN" vs "BMNN là gì?")

### 7.2 `is_legal_query` nên là computed field

Plan (D3) đề xuất thay `is_legal_query` LLM flag bằng computed field → **rất hợp lý**.

Hiện tại LLM quyết định `is_legal_query` nhưng nó chỉ được dùng trong keyword safety net (L766-776). Có thể tính từ intent:
```python
_LEGAL_INTENTS = {"search", "search_section", "summarize", "resolve_doc", 
                  "search_abbr", "kg_query", "list_docs", "search_doc_num"}
is_legal_query = intent in _LEGAL_INTENTS
```

Lợi ích: bỏ 1 field khỏi LLM output → giảm complexity, giảm token, giảm lỗi.

### 7.3 Output format template cũng thiếu comma (L333)

```python
# L333 — template output format CŨNG bị lỗi
{{\"next_agent\":\"<agent>\",\"intent\":\"<first step intent>\",\"task_plan\":[\"<step1>\",\"<step2>\",...],\"needs_memory\":false,\"is_legal_query\":false\"reasoning\":\"<brief>\"}}
#                                                                                                                              ^ THIẾU COMMA
```

Đây là P0 bug (đã covered trong plan), nhưng plan chỉ nói "41+ examples" — cần nhấn mạnh **template output format cũng bị**.

### 7.4 Supervisor prompt thiếu `resolve_doc` agent trong AVAILABLE AGENTS

```python
# L24-28: AVAILABLE AGENTS
- "rag"    : Document search...
- "write"  : Text editing...
- "people" : Person record...
- "direct" : Pure greetings...
- "finish" : Final answer...
```

Nhưng `_INTENT_TO_AGENT_FALLBACK` (L94) map `resolve_doc` → `AgentType.RESOLVE_DOC`. Và graph có `resolve_doc_agent_node` riêng. **Prompt không liệt kê `resolve_doc` agent** → LLM không biết có agent này → luôn route resolve_doc qua `rag`.

Thực tế code xử lý bằng override ở L438-449 (`_INTENT_TO_AGENT_FALLBACK`) → hoạt động. Nhưng nếu thêm `resolve_doc` vào available agents list, LLM sẽ route đúng ngay lần đầu, giảm log warning.

---

## Tổng Kết Đánh Giá

| # | Plan Item | Accuracy | Notes |
|---|-----------|----------|-------|
| P0 | JSON syntax error | ✅ **Chính xác** | Critical, cần fix ngay. Template L333 cũng bị |
| P1 | Tách inline prompts | ⚠️ **Thiếu 1 item** | Bỏ sót memory injection prompt (L1080-1088) |
| P2 | Giảm 40% supervisor prompt | ⚠️ **Quá aggressive** | Nên target 25-30% cho Qwen3 model |
| P3 | `_PERSONAL_INSTRUCTIONS` | ⚠️ **Thiếu half** | Cần sửa cả `direct_answer_node` |
| P4 | Improve FALLBACK_STANDARD | ✅ **Đúng** | Low effort, low risk |
| P6 | Test framework | ✅ **Cần thiết** | Cần chi tiết hơn |
| D2 | Thêm examples cho query_analyzer | ✅ **Đúng** | Cần chỉ rõ intents nào thiếu |
| D3 | `is_legal_query` computed field | ✅ **Rất tốt** | Giảm LLM complexity |
| D4 | Thêm examples cho supervisor | ⚠️ **Contradicts P2** | P2 giảm examples, D4 thêm examples → cần balance |
| — | `resolve_doc` agent thiếu trong prompt | 🔴 **Bỏ sót** | Nên thêm vào AVAILABLE AGENTS |

### Khuyến nghị thứ tự thực hiện

```
1. [5m]  P0: Fix JSON comma errors (tất cả 41 examples + template L333)
2. [15m] 7.4: Thêm resolve_doc vào AVAILABLE AGENTS section  
3. [30m] P3: Thêm _PERSONAL_INSTRUCTIONS + sửa direct_answer_node
4. [45m] P1: Tách inline prompts (disambiguation + resolve_doc + memory injection)
5. [2h]  D2+D4: Thêm examples (query_analyzer + supervisor — coordinated)
6. [2h]  P2: Giảm supervisor prompt 25-30% (sau khi thêm examples ở step 5)
7. [1h]  D3: Chuyển is_legal_query thành computed field
8. [5m]  P4: Improve FALLBACK_STANDARD
9. [2h]  P6: Test framework
```

> [!IMPORTANT]
> **Step 5 và 6 phải phối hợp**: thêm examples có giá trị trước (D2+D4), rồi mới cắt redundant examples (P2). Làm P2 trước sẽ giảm mất examples cần thiết.

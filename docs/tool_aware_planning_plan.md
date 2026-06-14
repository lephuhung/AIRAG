# Tool-Aware Planning cho LangGraph Supervisor — SIMPLIFIED

> **Cập nhật 2026-06-09**: Hướng simplified — KHÔNG extract structured profile, không tạo node mới, không thêm state fields. Chỉ fix prompt để LLM so sánh memory vs. document requirements.

## Context

**Vấn đề**: Với câu hỏi *"Thiết bị máy tính của tôi có sử dụng được để soạn thảo tài liệu BMNN không?"*, hệ thống trả về 3 văn bản nói chung về BMNN, không so sánh với thiết bị thực tế của user.

**Root cause** (từ debug session):
- `memory_recall` trả memory text đầy đủ: *"Người dùng đang sử dụng MacBook Pro M3..."*
- `answer_generator` nhận memory text nhưng **chỉ paraphrase**, không so sánh
- LLM không được instruction rõ ràng để SO SÁNH user context vs. doc requirements

**Giải pháp**: Sửa prompt trong `answer_generator` để LLM tự động SO SÁNH khi query có pattern "X của tôi có thể Y không".

---

## Approach

### Nguyên tắc
- **Không extract structured profile**: Memory text đã có device info dạng tự nhiên, LLM đọc được
- **Không tạo node/route mới**: Dùng flow hiện tại, chỉ sửa prompt
- **So sánh trong answer_generator**: Đây là nơi LLM có đầy đủ context (memory + RAG sources) để so sánh
- **Query enrichment đảm bảo search đúng**: `query_enricher` đã rewrite "của tôi" → org name, giữ device reference

### Flow (giữ nguyên, không thay đổi node/routing)
```
START → query_analyzer → supervisor → memory_recall → query_enricher → rag → result_evaluator → answer_generator → END
```

**Thay đổi duy nhất**: Prompt trong `answer_generator` để LLM SO SÁNH thay vì paraphrase.

---

## Changes

### 1. `answer_instructions.py` — Thêm COMPARISON instruction

**File**: `backend/app/prompts/agents/answer_instructions.py`

**Thêm section COMPARISON**:

```python
# Thêm vào cuối file, sau INSTRUCTIONS hiện có:

COMPARISON_INSTRUCTION = """
When user asks a COMPARISON question — "X của tôi có [thể] Y không?" or 
"Đơn vị tôi có đủ điều kiện Z không?" — you MUST:
1. Extract the USER CONTEXT from the memory section (device, OS, org, role, etc.)
2. Extract the REQUIREMENTS from the retrieved documents (what specs, standards, conditions are needed for Y/Z)
3. Compare explicitly: does the user's context meet the requirements? Answer YES/NO/PARTIALLY with reasoning.
4. If the system has no memory about the user's device/org, say "Tôi không có thông tin về thiết bị/tổ chức của bạn" instead of guessing.

DO NOT just summarize documents. You MUST perform the comparison.
"""
```

**Tích hợp vào `answer_generator` trong `nodes.py`**:

Trong `effective_system` build, thêm:
```python
# Sau user_memory_context injection, thêm comparison instruction
if state.get("needs_comparison", False):
    effective_system += "\n\n" + COMPARISON_INSTRUCTION
```

**Trigger**: Set `needs_comparison=True` trong state khi query_analyzer detect comparison pattern (đã có trong code hiện tại ở supervisor.py, truyền qua result_evaluator → answer_generator).

---

### 2. `query_enricher` — Đảm bảo search BMNN requirements, không generic

**File**: `backend/app/services/agents/supervisor.py` — `_query_enricher_wrapper`

**Kiểm tra và fix**: Khi query có "BMNN", enricher phải đảm bảo rewritten_query tìm **yêu cầu thiết bị cho BMNN**, không phải định nghĩa BMNN.

```python
# Thêm vào _query_enricher_wrapper:
if "bmnn" in query_lower:
    # Detect comparison intent → add explicit search qualifier
    if any(kw in query_lower for kw in ["có thể", "dùng để", "sử dụng", "đủ điều"]):
        # Rewrite: "BMNN cần thiết bị gì để soạn thảo" thay vì "BMNN là gì"
        enriched = enriched.replace("BMNN là gì", "yêu cầu thiết bị soạn thảo BMNN")
        enriched = enriched.replace("BMNN quy định", "yêu cầu thiết bị soạn thảo BMNN quy định")
        # ...
```

---

## Files Changed

| File | Change |
|------|--------|
| `backend/app/prompts/agents/answer_instructions.py` | + COMPARISON_INSTRUCTION section |
| `backend/app/services/agent/nodes.py` | + inject COMPARISON_INSTRUCTION when `needs_comparison=True` |
| `backend/app/services/agents/supervisor.py` `_query_enricher_wrapper` | + BMNN-specific query rewrite for comparison intent |

---

## Reused

- `needs_comparison` flag (đã có, truyền từ query_analyzer → result_evaluator → answer_generator)
- `user_memory_context` (đã có trong answer_generator)
- `_query_enricher_wrapper` (đã có)
- `memory_recall` (đã có, trả text đầy đủ)

---

## Verification

### Test query
```
"Thiết bị máy tính của tôi có sử dụng được để soạn thảo tài liệu BMNN không?"
```

### Expected
1. `query_analyzer` → `needs_comparison=True`
2. `memory_recall` → memory text: "Người dùng đang sử dụng MacBook Pro..."
3. `rag` → tìm BMNN requirements (not generic BMNN definition)
4. `answer_generator` → SO SÁNH: "MacBook Pro M3 vs. BMNN device requirements" → YES/NO/PARTIALLY

### Regression
| Query | Expected |
|-------|----------|
| "An ninh mạng là gì?" | summarize only (no comparison) |
| "Tôi là ai?" | direct answer (no comparison) |
| "Đơn vị tôi có cần tuân thủ Luật ANM không?" | memory context used but still comparison → YES |
# Audit logic v2 — Câu hỏi đơn giản & tool tra cứu không hoạt động

- Ngày: 2026-09-14 (UTC)
- Phạm vi: worktree `langgraph-v2` (nhánh `feat/langgraph-v2`, HEAD `f766bda`) so với worktree gốc (`LLM-Optimize`, HEAD `6f83bfd` = v1).
- Phương pháp: đọc code + reproduce `analyze_query`/`decide_route` thật (9 query mẫu), chưa sửa code. Tài liệu này chỉ ghi nhận lỗi để lên plan khắc phục sau.
- File trung tâm: `backend/app/services/agents/v2/nodes/routing.py` (`analyze_query`, `decide_route`, `route_node`), `backend/app/prompts/agents/supervisor_scope.py` (`classify_supervisor_scope`, các regex), `backend/app/services/agents/v2/adapters/semantic.py` (`draft_from_preprocessing`), `backend/app/services/agents/supervisor_v2.py` (`DeterministicSemanticAdapter`, `complex_unavailable_node`, `_wrap_node`), `backend/app/services/agents/v2/nodes/fast_plan.py`, `backend/app/services/agents/v2/nodes/finalizer.py`, `backend/app/services/agents/v2/execution/scheduler.py`.

## Ma trận reproduce (đã chạy trên code hiện tại, `allowed_capabilities` đầy đủ)

| # | Query | `domains` / `work_type` | Route | Nhận xét |
|---|---|---|---|---|
| 1 | "Chính sách nghỉ phép năm là gì?" | `(document,)` / retrieve | `complex_research/multi_document_research` | Lỗi A1 |
| 2 | "Nghỉ ốm cần giấy tờ gì?" | `(document,)` / retrieve | complex | Lỗi A1 |
| 3 | "0901234567 là ai?" | `(knowledge_graph,)` / lookup | `fast_domain/simple_kg_lookup` | Sai tool, lỗi A2 |
| 4 | "079012345678" (CCCD trần) | `(document,)` / retrieve | complex | Lỗi B2 |
| 5 | "079012345678 CCCD này của ai?" | `(people,)` / lookup | `fast_domain/simple_people_lookup` | ✅ đúng duy nhất |
| 6 | "Nguyễn Văn A là ai?" | `(knowledge_graph,)` / lookup | `fast_domain/simple_kg_lookup` | Sai tool, lỗi A2 |
| 7 | "Tìm ông Nguyễn Văn A" | `(people,)` / lookup | `fast_domain/simple_people_lookup` | ✅ đúng |
| 8 | "Điều 5 Luật An ninh mạng nói gì?" | `(document,)` / retrieve | complex | Lỗi A3 |
| 9 | "xin chào" | `(memory,)` / explain | `direct/direct_greeting` | ✅ đúng |

---

## Nhóm A — Routing đẩy nhầm vào `complex_research` (nguyên nhân chính)

Tất cả các nhánh dưới đều nằm trong `decide_route()` (`routing.py:274-357`). Fast chỉ mở khi khớp **chính xác** 1 trong 4 cửa (`simple_people_lookup`, `exact_document_metadata`, `exact_section_retrieval`, `simple_kg_lookup`); mọi trường hợp còn lại — kể cả câu hỏi thường — đều rơi vào complex, mà `complex_boundary` ở Phase-2 là node unavailable (`supervisor_v2.py:675`, `return {}` → finalizer trả `denied/error/insufficient`).

### A1. [Critical] Câu factual không ref nào → `document` → complex catch-all
- Vị trí: `routing.py:139-152` (gán `domains={"document"}` khi không có ref, không phải people, không conversational) + `decide_route()` cửa `exact_document_metadata` đòi `bound_count == 1` + `return` cuối hàm (`multi_document_research`).
- Kích hoạt: mọi câu hỏi nội dung không nêu tên văn bản ("nghỉ phép cần gì?", "chế độ thai sản thế nào?"). `bound_count == 0` nên trượt fast, không phải compare/evaluate → catch-all.
- Đối chiếu v1: `supervisor_node` (LLM) → `rag` → `react_executor_node` luôn có `search_documents`; không cần ref trước.
- Hướng xử lý (để plan sau): cần 1 đường factual mở (retrieval/discovery qua complex thật hoặc fast `general_search`) cho `bound_count == 0`; hiện tại đường này chưa tồn tại.

### A2. [Critical] "là ai" cướp miền `knowledge_graph`, đè `people`
- Vị trí: `_KG_RES` (`routing.py:82-93`) chứa `"là ai"`, `"who is"`, `"thuộc về"`…; `analyze_query` cộng miền KG **trước** nhánh fallback people (`routing.py:128-152` chỉ xét people khi `domains` đang rỗng).
- Kích hoạt: "0901234567 là ai?", "Nguyễn Văn A là ai?" → `domains={knowledge_graph}` → `simple_kg_lookup` thay vì `people.lookup`. Sai tool gốc.
- Hướng xử lý: xét people-identifier (số điện thoại/CCCD/BHXH trong text) **trước** KG; hoặc loại `"là ai"` khỏi `_KG_RES` khi text chứa identifier.

### A3. [Critical] `section.read` không thể tới được
- Vị trí: `draft_from_preprocessing` (`adapters/semantic.py:172-196`) luôn `section_refs=()` → điều kiện `exact_section_retrieval` (`routing.py:~318`, đòi `semantic.section_refs` non-empty + `bound_count == 1`) chết hoàn toàn.
- Kích hoạt: mọi "Điều/Khoản/Chương X Luật Y". Nếu preprocessor bắt được doc-ref thì thành `document.read` 1-pin (sai granularity), không thì complex.
- Hướng xử lý: trích `section_refs` từ preprocessor (v1 có `_SECTION_REF_RE`, `_extract…` nhưng chưa bao giờ port sang draft).

### A4. [Major] Câu `personal` ("tôi là ai?", "hồ sơ của tôi") → complex
- Vị trí: `supervisor_scope.py` có `_PERSONAL_CUE_RE` + scope `"personal"`, nhưng `analyze_query` **không đọc scope personal** — chỉ kiểm tra `== "people"`, còn lại ép `{"document"}`.
- Kích hoạt: "tôi là ai?", "thông tin cá nhân của tôi" → `document/retrieve` → complex → unavailable. v1 route `personal → direct` + memory.
- Hướng xử lý: thêm nhánh `personal → direct` (memory) trong `analyze_query`/`decide_route`.

### A5. [Major] Câu chào kèm nội dung → `direct`, mất retrieval
- Vị trí: `_GREETING_RE`/`_CONVERSATION_RE` của `routing.py:55-60` chỉ neo `^` (match tiền tố), trong khi `_GREETING_RE` của `supervisor_scope.py:396` đòi **cả câu** là chào. `decide_route` (`routing.py:~296`) cho `direct` khi `_is_conversational(text)` và không có intrinsic ref.
- Kích hoạt: "chào anh, cho tôi hỏi về chế độ thai sản?", "cảm ơn anh, còn chế độ ốm đau thì sao?" → `direct_conversation`, không gọi tool.
- Hướng xử lý: dùng chung 1 regex whole-message như scope, hoặc đòi hỏi không còn nội dung factual sau lời chào mới cho direct.

### A6. [Major] Từ khóa `compare/summary/evaluate` match chuỗi con, ép complex
- Vị trí: `_COMPARE_RES` (`"khác nhau"`, `"khác biệt"`…), `_SUMMARY_RES` (`"tổng hợp"`, `"tổng quan"`…), `_EVALUATE_RES` (`"đánh giá"`, `"tuân thủ"`…) + `len(document_refs) >= 2/3` (`routing.py:154-156,168`).
- Kích hoạt: câu khái niệm "sự khác biệt giữa nghỉ phép và nghỉ ốm là gì?" → `comparison`; "tổng hợp giúp tôi…" (động từ thường) → `summarize`; "đánh giá giúp…" → `compliance_evaluation`. Tất cả về complex.
- Hướng xử lý: thu hẹp (word-boundary/regex thay vì substring), phân biệt "so sánh 2 văn bản cụ thể" vs "hỏi khái niệm".

### A7. [Major] Lẫn 2 miền → `cross_domain_dependency`
- Vị trí: `dependency_families >= 2` (`routing.py:170-171`).
- Kích hoạt: vừa có từ KG ("thuộc đơn vị", "là ai") vừa có ref tài liệu; hoặc (khi C1 được sửa) vừa `person_refs` vừa `document_refs`. Mọi câu lai đều complex.
- Hướng xử lý: cho fast lai tối thiểu (people + 1-pin doc) hoặc thứ tự ưu tiên miền thay vì ép complex.

### A8. [Major] `api_explicit` 1-doc cũng bị ép complex
- Vị trí: `decide_route` nhánh `_has_api_explicit_target` (`routing.py:~305`) — chủ ý P0, nhưng nghĩa là file đính kèm/API scope dù đơn giản cũng không bao giờ đi fast.
- Ghi nhận để plan P0/factual-retrieval quyết giữ hay mở fast cho 1-pin explicit.

### A9. [Minor] `write` nuốt cả "bài viết"
- Vị trí: `_WRITE_RES` có `\bviết\b` (`routing.py:62`); `_write_intent` chỉ bị phủ định khi có động từ đọc (`_READ_VERB_RE`).
- Kích hoạt: "bài viết về chế độ thai sản?" (danh từ, không phải yêu cầu viết) → miền `write` → `complex/simple_write_operation` → denied.
- Hướng xử lý: loại trừ "bài viết/bài đăng", hoặc đòi động từ yêu cầu hành động ("hãy viết", "soạn giúp").

---

## Nhóm B — Regex phân loại người thiếu/sai (kể cả sau 2 fix `9e35f03`, `f766bda`)

Hai fix gần nhất chỉ vá nhánh phone-hẹp + dispatch; các khe hở dưới vẫn mở (`supervisor_scope.py:425-449`, `490-571`).

### B1. [Major] CCCD/BHXH trần không keyword → rớt document
- `_CCCD_RE = \b\d{9,12}\b` đòi kèm "cccd/căn cước/id card"; `_BHXX_RE = \b\d{10}\b` đòi kèm "bhxh/bảo hiểm". Số trần "079012345678" → scope `full` → v2 ép `document` → complex (reproduce #4).
- Hướng xử lý: số 12-digit / 9-digit trần → ít nhất cho `clarify` (hỏi loại ID) thay vì document; cân nhắc heuristic độ dài.

### B2. [Major] Overlap miền số: phone 10 số `0xxx` vs CCCD 9–12 vs BHXH 10
- `0901234567` khớp cả `_VN_PHONE_RE` (`\b0\d{9}\b`) lẫn `_CCCD_RE` lẫn `_BHXX_RE`. Hiện tại thứ tự phone-first cứu (`classify` và `people_intent_from_query` đều phone-first), nhưng giòn: đổi thứ tự là sai intent; số bàn 11 số (`0` + 10 số) thì `_VN_PHONE_RE` **không khớp** (đúng 10 số) → rớt.
- Hướng xử lý: chuẩn hoá độ dài theo spec viễn thông/CCCD/BHXH + test ma trận chéo.

### B3. [Major] Tên người không prefix "tìm/tra cứu" → không nhận
- `_PERSON_NAME_LOOKUP_CUE_RE` chỉ nhận `tìm ông/bà… [A-ZĐ]`, `tra cứu người|thông tin`, `số điện thoại của|phone of`.
- Miss: "ông A là ai", "cho tôi thông tin chị B", "email/liên hệ của anh C", tên trần "Nguyễn Văn A".
- Hướng xử lý: mở cue (thông tin/liên hệ/email/chức vụ + tên viết hoa), hoặc NER/lookup-then-verify.

### B4. [Minor] `_VN_PHONE_RE` cứng 10 số, thiếu đầu số mới/dạng cách
- Không nhận số cách nhau ("0901 234 567", "0901-234-567"), `+84`, mã vùng mới. Kiểm tra normalize trước match.
- Hướng xử lý: normalize (bỏ space/dash, `+84` → `0`) rồi mới match.

### B5. [Minor] `_MASKED_ID_RE` (`***411`) là pattern log nội bộ, user thật không gõ
- Nhánh masked chỉ có nghĩa cho replay/log đã redact PII; không phải đường user. Ghi nhận để khỏi nhầm là đã cover "mọi dạng phone".

### B6. [Minor] Greeting 2 nơi, 2 regex khác nhau (đã nói ở A5 nhưng ghi riêng cho dữ liệu)
- `routing.py:55-60` (tiền tố) vs `supervisor_scope.py:396` (whole-message). Hai nguồn sự thật → hành vi khác nhau giữa v1-scope và v2-route.
- Hướng xử lý: 1 nguồn regex dùng chung.

---

## Nhóm C — Mất/khớp lệch dữ liệu giữa các tầng

### C1. [Critical] `draft_from_preprocessing` vứt `person_refs`, `section_refs`, `coreferences`
- Vị trí: `adapters/semantic.py:172-196` (`person_refs=()`, `section_refs=()`, `coreferences=()`).
- Hệ quả: A2/A3/A4; câu follow-up ("văn bản này", "ông ấy") mất ngữ cảnh vì `contextualized_query` chỉ lấy `conversation.summary` (có thể rỗng) mà `coreferences` luôn rỗng.
- Hướng xử lý: port trích section/person từ v1 preprocessor hoặc small-model extractor có validate; tối thiểu log tỉ lệ rỗng.

### C2. [Major] `_apply_resolution` fail-closed cứng khi resolved-nhưng-chưa-pin
- Vị trí: `v2/nodes/context.py:_apply_resolution` — `resolved reference … has no pinned binding` → `ContextNodeError` → marker → finalizer lỗi.
- Kích hoạt: lệch nhịp binding/finalizer (draft rebuild mỗi node, ref-id positional `r1/r2` đổi theo query — chính `DeterministicSemanticAdapter` đã cảnh báo lifetime `ui_selection`/`known_documents` phải rebuild mỗi turn).
- Hướng xử lý: reconcile mềm (re-resolve) thay vì raise, hoặc pin ref-id ổn định theo nội dung.

### C3. [Major] `stable_people_record_id` đổi digest → vỡ ổn định qua bản vá
- Vị trí: `supervisor_v2.py:stable_people_record_id` — `f766bda` thêm salt `group`/`dob` vào material. Cùng 1 người, digest trước/sau khác nhau → `record_id` (`p_…`) khác → evidence/GC/ref cũ không nối được.
- Hướng xử lý: version hoá digest hoặc migration map; test ổn định id.

### C4. [Major] Placeholder `Không rõ tên` đi vào evidence như tên thật
- Vị trí: `_build_people_matches` (`supervisor_v2.py`) — match không tên (schema phone `uids`) giữ placeholder + `required_fields=("name","phone","source")`.
- Rủi ro: synthesis/grounding trình bày "Không rõ tên" như danh tính; `required_fields` chứa giá trị giả.
- Hướng xử lý: đánh dấu `name_unknown=true`, cấm synthesis引用 placeholder làm tên, hoặc tách required_fields.

### C5. [Minor] `_people_group_key` fallback `id:_id` không dedupe liên schema
- Cùng người ở 2 schema khác nhau không gộp (chấp nhận được) nhưng cần ghi nhận để khỏi kỳ vọng dedupe toàn cục.

### C6. [Minor] `blocking_ambiguities` lọc chỉ `essential`
- Vị trí: `adapters/semantic.py:_blocking_ambiguities` — ambiguity non-essential bị rơi; `finalize_blocking_ambiguities` còn trừ tiếp theo binding.
- Rủi ro: ca cần hỏi lại bị ép đi tiếp → insufficient thay vì clarify.

### C7. [Minor] `bound_count` loại pin turn trước ("stale pins excluded")
- Follow-up "tóm tắt văn bản này" sau turn đã pin → `bound_count == 0` → complex thay vì fast. Liên quan C1 (không coreference).

---

## Nhóm D — Không có lưới an toàn khi deterministic sai (trả lời câu "tại sao không dùng LLM")

Chủ ý spec (contract-first §12): *"Greetings, People lookup, exact Section retrieval, bounded Write classify without a model. Uncertain cases **may** use a small model."* — nhưng implementation **chưa làm vế thứ hai**:

### D1. [Critical] Không có nhánh "uncertain → small model"
- Mọi ca regex không cover đều thành `document/retrieve` → complex → unavailable. v1 có lưới: JSON-parse fail → fallback `rag/search` (`supervisor.py:876-879, 1004-1010`); condenser fail → giữ nguyên message (`supervisor.py:811-815`); judge fail-open (`supervisor.py:673-674`).
- Hướng xử lý (plan): thêm 1 trong: (a) small-model classifier cho ca uncertain (đúng như spec cho phép), (b) fast `general_search` cho `bound_count == 0`, (c) fallback discovery trước khi unavailable. Đo precision/recall trên log thật.

### D2. [Major] `allowed_capabilities` thiếu → `runtime_dependency` → complex, im lặng
- `_fast_or_runtime_dependency` (`routing.py:220-224`): đủ điều kiện fast nhưng thiếu capability → complex mà không log/phân biệt "thiếu quyền" vs "sai route". Default param `frozenset()` càng dễ mask trong test.
- Hướng xử lý: log + metric riêng; test với runtime production thật.

### D3. [Major] `FastPlanError` (vd document fast thiếu pin) thành marker lỗi, không clarify/discovery
- `fast_plan_node` raise → `_wrap_node` gắn marker → finalizer. User không được hỏi bổ sung.
- Hướng xử lý: chuyển thành `clarify`/`needs_input` khi thiếu pin.

### D4. [Minor] Deadline truncate → `undispatched_tasks` → insufficient, không retry
- `scheduler.py:383-384`, `supervisor_v2.py:2909`. Ghi nhận để plan SLA/retry, không phải root cause hiện tại.

---

## Nhóm E — Logic phụ trợ cần rà khi sửa (không phải root cause nhưng dễ vỡ theo)

- **E1. Clarify loop**: `_has_open_required_reference` (bất kỳ `document_refs` nào `!= resolved`) → `clarify/unresolved_required_binding`. Nếu resolver không resolve được + user không trả lời, turn kẹt clarify. Cần timeout/discovery-fallback.
- **E2. Finalizer precedence**: `denied > error > insufficient` (`finalizer.py:_typed_synthesis_failure`); `_typed_missing_verdict` phân `clarify` vs `insufficient` theo `_semantic_needs_input`. Khi sửa routing, kiểm tra lại mapping status để khỏi lộ `denied` nhầm.
- **E3. Binding resolver fail-closed khi workspace rỗng** (`supervisor_v2.py:V1BindingResolver.resolve`): refs rỗng → empty set OK; workspace rỗng → fail-closed. Câu multi-workspace cần test union (đã có commit `0218bd8` giữ multi-workspace — regression-test lại sau mọi sửa routing).
- **E4. `people_intent_from_query` trùng logic `classify`** nhưng là 2 bản copy (`supervisor_scope.py:573`): sửa 1 nơi dễ quên nơi còn lại. Gộp hoặc test cặp.
- **E5. `has_intrinsic_refs` bỏ qua `api_explicit`**: greeting kèm `document_ids` transport-only vẫn direct (chủ ý), nhưng nếu explicit là target thật + lời chào mở đầu ("chào anh, tóm tắt file đính kèm") → direct sai. Cần phân biệt transport vs target (liên quan A5/A8).

---

## Phụ lục — File tham chiếu nhanh

- Router: `backend/app/services/agents/v2/nodes/routing.py` (hằng số 55-106, `analyze_query` 119-193, `decide_route` 274-357)
- Scope/regex: `backend/app/prompts/agents/supervisor_scope.py` (regex 396-484, `classify_supervisor_scope` 490-571, `people_intent_from_query` 573-593)
- Semantic port: `backend/app/services/agents/v2/adapters/semantic.py:172-196`
- Context/finalize: `backend/app/services/agents/v2/nodes/context.py` (`_apply_resolution`, `finalize_semantic`)
- Graph/supervisor: `backend/app/services/agents/supervisor_v2.py` (`complex_unavailable_node:675`, `_route_branch:970`, `_complex_branch:999`, people dispatch ~1432-1880, `DeterministicSemanticAdapter:1175`, `undispatched_tasks:2909`)
- Fast plan: `backend/app/services/agents/v2/nodes/fast_plan.py`
- Finalizer: `backend/app/services/agents/v2/nodes/finalizer.py`
- Scheduler: `backend/app/services/agents/v2/execution/scheduler.py`
- Đối chiếu v1: `backend/app/services/agents/supervisor.py` (`supervisor_node:1199`, `react_executor_node:2308`, people chain `3711-3713`, fallback JSON `876-879/1004-1010`), `backend/app/services/agents/react_tools.py` (`RAG_TOOL_SCHEMAS:607`), `backend/app/services/agents/people_agent.py:149,339`
- Spec/plan: `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md` (§4.3, §12), `docs/superpowers/plans/2026-09-11-langgraph-v2-phase2-fast-paths.md`

# Đề xuất nâng cấp Hybrid Supervisor + Deep Agent, ngân sách dưới 30 giây

**Trạng thái:** DRAFT để duyệt; không phải chỉ thị triển khai. Chưa thay đổi runtime, chưa benchmark, chưa chọn phiên bản Deep Agents để deploy.

**Mục tiêu:** Giữ tốc độ của câu hỏi đơn giản; cải thiện so sánh chương/văn bản, tóm tắt tài liệu dài và tổng hợp nhiều nguồn/agent. Không đánh đổi grounding và phân quyền để đạt latency.

**Phạm vi pilot:** So sánh hai chương thuộc hai văn bản đã được xử lý/index. Tóm tắt dài và cross-agent là các bước mở rộng sau khi pilot đạt tiêu chí.

**Prompt đi kèm:** `docs/prompts/complexity-router.vi.txt` — bản độc lập dùng để đánh giá; khi tích hợp phải hợp nhất với schema supervisor trong MỘT lượt classifier, không gọi thêm một router cho mọi request.

## 1. Quyết định kiến trúc dự kiến

```text
Request + principal + context đã xác minh
  -> semantic preprocessing dùng chung
       -> giữ raw query + làm rõ follow-up/viết tắt theo context
       -> trích xuất từng tham chiếu và cặp văn bản–chương
       -> lookup số/tên văn bản trong workspace được phép
       -> resolution status + metadata + candidate handles
  -> routing thống nhất trên semantic context (fast-path chắc chắn hoặc một LLM classifier)
       -> supervisor: pipeline đơn giản hiện tại
       -> clarify: hỏi một câu làm rõ và kết thúc
       -> deepagent: bounded planner/coordinator
            -> tác vụ theo quan hệ phụ thuộc, tối đa hai nhánh song song
            -> công cụ/domain worker hiện tại qua adapter
            -> evidence theo từng tác vụ
            -> tổng hợp một lần + grounding/citation guard
  -> SSE adapter + persistence hiện tại, sửa contract cần thiết
```

- Không đặt Deep Agent phía trước mọi câu hỏi rồi mới gọi supervisor.
- Không chạy `query_analyzer` LLM -> supervisor classifier -> Deep Agent planner nối tiếp. Đường mới hợp nhất phân loại và complexity ở lượt classifier hiện có; Deep Agent chỉ lập kế hoạch một lần sau khi được chọn.
- Các fast-path deterministic chỉ áp dụng cho yêu cầu thuần đơn. Đặc biệt, xuất hiện CCCD/điện thoại không được nuốt mất yêu cầu tra quy định hoặc đối chiếu ở nửa sau câu hỏi.
- Giữ fallback bằng feature flag để tắt toàn bộ nhánh mới; không reset deadline khi đổi nhánh.
- Đây là dùng Deep Agents phía trên LangGraph, không thay LangGraph bằng runtime khác.
- Không bọc nguyên các node đang tự stream câu trả lời thành subagent. Worker trả kết quả có cấu trúc; chỉ tầng tổng hợp ngoài cùng phát answer token. Worker chỉ phát progress/tool status có `task_id`.

### 1.1. Semantic preprocessing trước routing — bổ sung sau review

**Mục đích:** Làm rõ người dùng đang nói về từ viết tắt, văn bản và phần văn bản nào trước khi chọn executor. Không trả lời nghiệp vụ, không lập task plan, không truy xuất hết nội dung ở giai đoạn này.

**Nền tảng hiện có đã kiểm tra:**
- `supervisor.py:_expand_abbreviations_in_message` tra các viết tắt bằng một batch SQL, tách đơn nghĩa/đa nghĩa/chưa biết.
- `supervisor.py:_disambiguate_multi_meaning_abbrs` dùng một lượt memory LLM cho cả nhóm đa nghĩa. Khi tái sử dụng phải kiểm tra chosen nằm trong tập candidate và kết quả bao phủ đủ input; không chỉ tin confidence do model khai.
- `rag_agent.py:_tool_search_doc_num` gọi `agent/tools.py:search_documents_number`. Tool đang fuzzy-match số/filename/storage key, trả tối đa 20 kết quả; không phải bộ xác nhận identity chính xác và không nên truyền cả câu so sánh dài làm query số hiệu.
- `agent/doc_resolver.py:resolve_candidates` là core dùng chung nhưng bao gồm cả LLM, vector, rerank và fuzzy fallback. `use_llm_fallback=False` CHỈ tắt LLM extraction, không tắt vector/rerank. Không gọi toàn bộ core vô điều kiện trước mọi request.

**Workflow dự kiến:**
1. Giữ `original_query` nguyên vẹn; đọc context gần nhất và attachment đã được phân quyền. Làm rõ follow-up nếu cần, giữ span gốc và mapping để không đảo thứ tự “văn bản thứ nhất/thứ hai”.
2. Batch lookup viết tắt. Đơn nghĩa phù hợp ngữ cảnh -> bổ sung alias/normalized query. Đa nghĩa -> giữ candidates, chỉ gọi tối đa một disambiguation batch khi cần và còn ngân sách. Không thay thế bên trong số hiệu, định danh người hoặc chuỗi được trích dẫn một cách mù quáng.
3. Trích xuất TỪNG tham chiếu cùng section binding: `[X, Chương II]`, `[Y, Chương III]`. Dùng cả raw query và alias mapping; không đưa cả câu vào parser vốn resolve một văn bản. Nếu reference số đã rõ, lookup của nó có thể độc lập với bước disambiguation khác; không parallelize các bước thật sự phụ thuộc.
4. Lookup metadata theo số hiệu chuẩn hóa + loại + năm + cơ quan, hoặc tên/attachment đã xác minh; batch hoặc tối đa hai nhánh với DB session riêng. Ưu tiên exact normalized number; fuzzy chỉ tạo candidate. Không tự thêm năm hiện tại hay coi bản mới tạo là bản người dùng muốn. Không có quyền thì không lộ candidate.
5. Trả `resolved/ambiguous/not_found/deferred/error` theo từng reference, cùng metadata/token estimate nếu có. `not_found` chỉ khi lookup đã hoàn tất trong phạm vi; timeout/outage không được đổi thành not_found. Metadata identity không phải evidence nội dung chương.
6. Gate ambiguity: chỉ hỏi nếu mơ hồ thiết yếu ảnh hưởng đối tượng/phạm vi; “BMNN nghĩa là gì?” có thể trả danh sách nghĩa qua supervisor, không bắt chọn trước. Số đầy đủ nhưng không có tài liệu -> thông báo không có trong nguồn, không hỏi người dùng lặp lại số đã rõ. Lookup danh sách theo số có thể có nhiều kết quả hợp lệ mà không cần ép chọn một.
7. Router đọc semantic context, quyết định supervisor/deepagent/clarify. Hai executor tái sử dụng identity đã resolved, không chạy lại toàn bộ viết tắt/number lookup. Khi executor phát hiện tham chiếu MỚI từ yêu cầu phụ thuộc, vẫn được gọi cùng resolver service với deadline còn lại.

**Nguyên tắc latency:** Mọi request đi qua contract preprocessing nhưng các bước không liên quan là no-op; câu chào hoặc tra người thuần đơn không chạy document lookup. Không bắt mọi request trả phí LLM disambiguation/vector search. Tiền xử lý có deadline riêng nằm TRONG 28 giây tổng; `deferred/error` được executor xử lý có kiểm soát hoặc trả giới hạn trung thực, không khởi tạo lại ngân sách.

**Contract `SemanticContext` dự kiến:**

```text
original_query, normalized_query,
abbreviations[{span, short_form, chosen, candidates, status}],
document_refs[{ref_id, original_span, reference, section_reference,
               document_handle, candidates, resolution_status,
               match_basis, version, metadata}],
blocking_ambiguities[], preprocessing_status
```

`document_handle`/candidate metadata do server tạo sau ACL; không do LLM phát minh. Giữ ref_id ổn định để phân biệt hai mục tiêu, kể cả khi chúng trỏ cùng một document. Không rút gọn tất cả thành một `document_ids` mutable dùng chung. `blocking_ambiguities` là bất định về nghĩa/đối tượng cần user, không phải danh sách lỗi infrastructure.

**Ví dụ:** “So sánh Chương II NĐ X và Chương III NĐ Y” -> làm rõ NĐ là Nghị định (nếu context/candidate hỗ trợ), resolve X/Y riêng, giữ section binding -> router chọn deepagent vì phải đọc/so sánh hai phạm vi, dù cả hai identity đã resolved. “Tìm NĐ X” có thể dùng luôn metadata lookup qua supervisor mà không search lại.

## 2. Định nghĩa phức tạp theo công việc, không theo độ dài câu hỏi

### Supervisor

Một mục tiêu mà pipeline hiện tại hoàn thành được bằng một workflow đã biết: hỏi đáp thông thường, tra một thông tin, đọc một điều/chương, tìm văn bản, chỉnh sửa đoạn văn, tóm tắt một nội dung vừa ngân sách một lượt.

`resolve_document -> read_section -> answer` vẫn có thể là workflow đơn giản. Không tính số tool calls đơn thuần để phân loại.

Nhiều nguồn retrieval trả về cho một câu hỏi thông thường không tự động là đa bước. So sánh hai đoạn đã nằm đầy đủ trong input và vừa context một lượt cũng không bắt buộc dùng Deep Agent.

### Deep Agent

Ít nhất một điều kiện có bằng chứng từ yêu cầu/context:

1. Phải truy xuất riêng nhiều đối tượng/phần tài liệu rồi so sánh, hợp nhất hoặc kiểm tra mâu thuẫn.
2. Phải thực hiện nhiều mục tiêu khác loại hoặc kết hợp nhiều domain agent, ví dụ tra người -> lấy định danh -> tìm hồ sơ -> đối chiếu quy định.
3. Bước sau cần dùng kết quả chưa biết của bước trước và workflow đơn giản không biểu diễn được phụ thuộc đó.
4. Metadata runtime xác nhận tóm tắt toàn văn cần chia nhỏ/map-reduce hoặc nhiều tài liệu phải được hợp nhất.

### Clarify

Thiếu định danh/phạm vi thiết yếu, không thể khôi phục chắc chắn từ context. Ví dụ “So sánh hai văn bản” nhưng không có tên, attachment hoặc antecedent rõ ràng. Không dùng `clarify` chỉ vì chưa biết độ dài tài liệu.

### Tóm tắt chưa biết độ dài

Router trả `needs_document_probe=true`; runtime kiểm tra metadata/token count trước khi chọn executor:

- Vừa ngân sách một lượt đã đo -> supervisor.
- Phải chia nhỏ -> Deep Agent.
- Không thể xác định/đọc trong deadline -> trả trạng thái giới hạn trung thực, không coi văn bản là ngắn hoặc tự cắt rồi gọi là tóm tắt đầy đủ.

`single_pass` được tính bằng model/context/output reserve và throughput đã đo, không dùng một mốc số trang cố định. Nếu có bản tóm tắt hợp lệ đã cache, runtime có thể dùng fast-path dù tài liệu gốc dài.

## 3. Ngân sách latency

### Cách đo

- `T_request`: từ lúc backend nhận request, bao gồm queue/admission, auth và routing; kết thúc khi phát terminal event.
- `T_answer_complete`: khi toàn bộ câu trả lời cuối đã được phát, không chỉ first token.
- `TTFT_answer` và `TTFT_progress` đo riêng; không dùng progress sớm để tuyên bố đạt 30 giây.
- Mục tiêu pilot: p95 `T_request < 30s` trên workload và concurrency được ghi rõ; cùng lúc đo tỷ lệ câu trả lời đầy đủ/đúng.
- Deadline server dự kiến 28 giây, giữ khoảng dự phòng cho đóng stream. Không bảo đảm thời gian người dùng nhận được qua mạng luôn dưới 30 giây.
- Tài liệu chưa ingest, quá dài, model quá tải hoặc nguồn unavailable không thể được hứa trả lời đầy đủ trong ngân sách này. Trả partial/clarification/deadline result, không bịa hoặc gắn nhãn complete cho nội dung chưa đủ.

### Phân bổ ban đầu — cần benchmark để hiệu chỉnh

| Khoảng tính từ request | Hoạt động | Quy tắc |
|---|---|---|
| 0–3s | Admission, auth, semantic preprocessing và metadata có sẵn | Batch viết tắt/number lookup; bỏ qua phần không liên quan; bounded disambiguation khi cần |
| 3–5s | Routing thống nhất + metadata probe còn thiếu | Router nhận raw/normalized query cùng resolution statuses |
| 5–7s | Một lượt lập kế hoạch khi cần | Không thêm planner/judge LLM thứ hai để duyệt plan |
| 7–16s | Read/retrieve/domain workers, resolve mới/deferred nếu cần | Tái sử dụng identity đã xác minh; tối đa hai nhánh đồng thời |
| 16–18s | Kiểm tra coverage và một repair nhỏ nếu đủ thời gian | Không bắt đầu thêm retrieval sau mốc này |
| 18–26s | Một lượt tổng hợp, có output cap | Chỉ dùng evidence hoàn tất và được phép truy cập |
| 26–28s | Guard còn lại, finalize/persist thiết yếu, đóng stream | Hết giờ phải có kết quả terminal trung thực |

- Dùng absolute deadline và remaining budget; không cộng timeout 30 giây riêng cho mỗi tool/agent.
- Bắt đầu synthesis sớm hơn nếu dự đoán output/model latency cần nhiều hơn 8 giây. Các mốc trên là initial budget, không phải benchmark đã đạt.
- Nếu đến synthesis cutoff mà chưa đủ nguồn: tổng hợp phần đủ và liệt kê phần thiếu; nếu không có căn cứ hữu ích, trả fallback định sẵn, không gọi thêm LLM vô ích.
- Deadline handler nằm ngoài graph/tool loop. Hủy và chờ cleanup tác vụ con; không để task ghi dữ liệu hoặc tiếp tục stream sau terminal event. Remote provider có thể không dừng compute ngay khi client hủy; phải đo và ghi nhận.
- Không chạy full retry, full judge/reflection, hoặc chuyển sang supervisor với một ngân sách mới sau timeout.
- Bản nháp bị grounding guard thu hồi phải được rollback ở cả frontend, stream accumulator và partial persistence.
- Queue phải bounded: khi không thể admit trong budget, trả busy/retry thay vì xếp hàng không giới hạn. Không restart vLLM.

### Giới hạn pilot ban đầu

- Tối đa 2 document branches đồng thời, không recursive delegation.
- Tối đa 4 domain work items, 6 semantic domain tool calls tổng cộng và 1 repair trong cùng tổng budget. Các lần tìm kiếm/LLM ẩn trong tool cũng phải nhận deadline và được đo; các con số này không tự bảo đảm latency.
- Khởi điểm tối đa 4 coordinator model rounds (bao gồm plan, delegation follow-up và synthesis); tối đa 2 model rounds cho mỗi domain worker cần LLM. Tính cả `task`, planning và tool built-in vào accounting; không coi `max tool steps` của ReAct cũ là cap tự áp dụng cho Deep Agents. Deadline toàn run luôn ưu tiên hơn các cap số lượng này.
- Tối đa 1 lượt plan và 1 lượt synthesis ở coordinator; worker được phép có call cần thiết nhưng vẫn chịu cap toàn run.
- Output synthesis khởi điểm 600–900 token, tùy throughput đo được; cần giới hạn cả output token và thời gian.
- Không yêu cầu mọi lần chạy phải dùng `write_todos`; plan nhỏ có thể nằm trong structured task manifest. Nếu dùng planning middleware, đo thêm chi phí tool/model round và bật rõ ràng theo phiên bản đã pin.

## 4. Contract tối thiểu

### RoutingDecision

```text
execution_mode: supervisor | deepagent | clarify
work_type: lookup | compare | summarize | multi_goal | cross_agent | other
needs_document_probe: boolean
reason_code: single_workflow | inline_content | multi_target_compare |
             multi_goal | cross_agent_dependency | dependent_research |
             long_document | summary_size_unknown | missing_reference
clarification_question: string | null
```

- Structured validation phía server; không dùng một số confidence do LLM tự khai làm xác suất đã hiệu chuẩn.
- Khi tích hợp, thêm `complexity_route: RoutingDecision` vào output supervisor hiện có; giữ `next_agent`, `intent`, `needs_memory` và các field tương thích. Không gán `next_agent=deepagent` trước khi schema/edge hỗ trợ.
- `route_from_supervisor` xem complexity branch trước các override prerequisite/multi-step cũ. Đường deep nhận nguyên yêu cầu đã contextualize, không bị `rewritten_query` của sub-query hoặc abbreviation override ghi đè.
- Với `needs_document_probe`, quyết định executor phải chờ metadata probe; đây không phải lệnh chạy supervisor ngay.
- Lỗi parse/timeout: không gọi classifier lại. Nếu chỉ một workflow đơn rõ ràng, dùng fallback hiện tại; nếu rõ đa bước thì chọn bounded deep hoặc báo giới hạn khi thiếu budget; nếu thiếu đối tượng thì hỏi làm rõ. Không hạ câu hỏi đa bước thành một lookup rồi báo hoàn tất.

- Input routing dùng `SemanticContext` ở mục 1.1. Đã resolve xong identity không đồng nghĩa đọc đủ nội dung hay nhiệm vụ trở thành đơn giản; còn bất định thiết yếu phải được giữ thay vì viết normalized query như đã biết chắc.

### RuntimeContext — không do model quyết định

```text
principal, allowed_workspace_ids, authorized_document_handles,
people_permission, session/run identity, absolute_deadline,
model/config snapshot, tool-budget accounting
```

Không serialize DB session/queue vào checkpoint. Mỗi nhánh concurrent có DB session/context riêng; quyền phải được kiểm tra tại tool/data boundary, không chỉ ở parent prompt.

### TaskResult / Evidence

```text
task_id, status: ok | partial | missing | ambiguous | error,
evidence_ids, coverage: requested/resolved/read/truncated,
missing_requirements, artifact_refs
```

- Evidence store theo run, giữ nguyên nội dung trích dẫn và provenance: source_id, document_id/version, section path, page/chunk, workspace.
- Worker summary chỉ hỗ trợ context; không thay bằng chứng nguyên bản. Gán citation ID tập trung hoặc namespace theo task để tránh va chạm.
- Synthesis đọc evidence qua handle; không cộng mọi kết quả vào một `sources` list rồi dùng `bool(sources)` để đánh giá độ đủ.
- SSE `sources` dùng snapshot tích lũy đã deduplicate; mọi publisher và persistence consumer tuân thủ cùng contract. Nếu muốn delta, phải version protocol và cập nhật mọi consumer.
- Contract terminal có `completion_status` (complete/partial/clarification/deadline/error) và phần thiếu; không lẫn kết thúc transport với hoàn thành nghiệp vụ. Cần test frontend có thể bỏ qua field mới an toàn và hiển thị trạng thái một phần.

## 5. Các giai đoạn nâng cấp dự kiến

### Giai đoạn 0 — Baseline và safety/contract blockers

**Đọc/sửa có kiểm soát:**
- `backend/app/services/agents/models.py`
- `backend/app/services/agents/supervisor.py`
- `backend/app/services/agent/streaming.py`
- `backend/app/api/chat_session.py`

**Deliverable:** Bộ regression tests trước thay đổi; baseline latency/quality; sửa attachment access/delete ownership, resolver FINISH precedence, field comparison bị lọc, source snapshot và rollback persistence. Chỉ sửa phạm vi đã được duyệt; không refactor toàn bộ supervisor.

**Tests dự kiến:** `backend/tests/agents/test_routing_contracts.py`, `test_stream_evidence.py`, `test_session_safety.py`.

**Điều kiện qua:** Không còn blocker phân quyền/citation trên đường sẽ tái sử dụng. Biết model/tool nào chiếm latency và tỷ lệ full answer dưới 30 giây hiện tại.

### Giai đoạn 1A — Semantic preprocessing dùng chung

**Tạo dự kiến:** `backend/app/services/agents/semantic_preprocessor.py` và `backend/tests/agents/test_semantic_preprocessing.py`.

**Tách/tái sử dụng có impact review:** các hàm abbreviation trong `supervisor.py`, metadata lookup theo số từ `agent/tools.py`, resolution primitives trong `agent/doc_resolver.py`; gọi qua service không kèm routing/SSE final-answer. Thêm chính sách fast metadata-only rõ ràng thay vì tưởng `use_llm_fallback=False` tắt mọi fallback.

**Tích hợp:** entry path dùng chung cho web/session/Telegram qua graph hoặc runner phù hợp; bỏ duplicate expansion/lookup ở `chat_agent_lg.py`/supervisor khi marker preprocessing hợp lệ đã tồn tại. Không cho client set marker để bỏ qua xử lý/ACL. Giữ original message cho persistence và intent, normalized query chỉ là derivative có provenance.

**Tests bắt buộc:** đơn nghĩa/đa nghĩa/chưa biết; chosen ngoài candidate; thiếu item trong disambiguation; số trùng năm/cơ quan; fuzzy không được auto-resolve như exact; full-query vs isolated reference; đúng hai cặp chương–văn bản; không sửa CCCD/suffix số hiệu; follow-up; ACL/timeout; 0 LLM calls khi không cần; không lookup lại reference đã resolved; cùng một query cả web/session đi qua preprocessing nhất quán.

**Điều kiện qua:** SemanticContext giữ đúng ý định và target binding, không tự tăng latency câu đơn giản; có trace riêng `semantic_preprocess`, `abbr_lookup`, `doc_identity_lookup`.

### Giai đoạn 1B — Complexity routing ở shadow mode

**Tạo dự kiến:**
- `backend/app/services/agents/complexity.py`: schema, validation, metadata decision, deterministic fallback; nhận SemanticContext từ giai đoạn 1A.
- `backend/app/prompts/agents/complexity_router_prompt.py`: module routing từ prompt draft.
- `backend/tests/agents/test_complexity_routing.py`.
- `backend/tests/prompts/test_complexity_router.py` và dataset có version.

**Tích hợp dự kiến:** `supervisor_scope.py`, parser trong `supervisor.py`, `models.py`, graph entry wiring.

**Deliverable:** Ghi predicted mode nhưng chưa đổi executor. Offline replay trước; shadow production chỉ trên mẫu đã cho phép, không gửi dữ liệu tới provider mới ngoài cấu hình được duyệt. Chưa bỏ analyzer cũ ở shadow; đo riêng overhead, không coi latency shadow là đường production mục tiêu.

**Sau shadow:** Hợp nhất classifier/complexity và bỏ lượt decomposition LLM dư thừa trên đường hybrid. Fast-path thuần đơn được giữ; có regression cho câu bắt đầu bằng greeting/CCCD nhưng có thêm nhiệm vụ.

**Lưu ý:** Prompt `supervisor_prompt.py` là shim; nơi cần sửa thực sự là các section và `_SS_OUTPUT_FORMAT` trong `supervisor_scope.py`. User wrapper hiện liệt kê key cố định, parser và cap `max_tokens=160` đều cần cập nhật. Output tích hợp khởi điểm 320 token, đo lại thay vì làm truncation âm thầm.

### Giai đoạn 2 — Deep Agent pilot cho compare_sections

**Tạo dự kiến:**
- `backend/app/services/agents/deep_research/graph.py`: bounded coordinator factory.
- `backend/app/services/agents/deep_research/contracts.py`: TaskResult/Evidence/runtime context.
- `backend/app/services/agents/deep_research/tools.py`: domain adapters có scope riêng.
- `backend/app/services/agents/deep_research/budget.py`: deadline, call budget, cancellation.
- `backend/app/services/agents/deep_research/evidence.py`: provenance và citation registry.
- `backend/app/services/llm/langchain_adapter.py`: adapter hoặc factory LangChain model theo config snapshot, chỉ nếu cần.
- `backend/tests/agents/test_deep_compare_sections.py`, `test_deep_budget.py`, `test_deep_scope_isolation.py`.

**Tái sử dụng:** document resolver, storage, retrieval, KG và citation/grounding helpers. Tool đọc chương cần hỗ trợ explicit document handle và full structural range; không giả định `search_document_section` hiện tại trả đủ chương.

**Model compatibility gate:** Pin một release thật của Deep Agents và dependency set phù hợp image Docker; test tool binding, tool-call ID, streaming, cancellation, token usage, Langfuse và runtime model hot config. Custom `LLMProvider` không phải `BaseChatModel`; không truyền trực tiếp. Có thể dùng provider integration có sẵn nếu endpoint tương thích được kiểm chứng. Không tự chuyển sang provider bên ngoài.

**Pilot flow:** Tái sử dụng X/Y đã resolved từ preprocessing (chỉ resolve reference mới/deferred nếu còn budget) -> đọc chapter range đầy đủ -> đánh dấu coverage/truncation -> một synthesis có citation -> guard -> terminal status. Một nhánh ambiguous phải được xử lý thành clarification hoặc partial rõ ràng, không tự chọn.

**Điều kiện qua:** So sánh đúng cả hai phạm vi; không trộn UUID; không báo full khi thiếu một phía; đơn giản không bị thêm vòng gọi LLM; cap time/calls thực sự cưỡng chế ở runtime.

### Giai đoạn 3 — Tóm tắt dài và cross-agent

**Tóm tắt:** thêm metadata probe và token-aware chunking. Tóm tắt theo cấu trúc và bounded map-reduce; output ghi rõ coverage. Cache summary theo document version + phạm vi + prompt/model version; kiểm tra ACL trước khi trả cache, không dùng cache cross-tenant không phân quyền. Với tài liệu quá dài chưa cache, 30 giây chỉ có thể trả partial trung thực hoặc yêu cầu thu hẹp.

**Cross-agent:** adapter people/document/KG trả structured result; không gọi các node đang tự stream final answer. People access được kiểm tra lại ở tool; tối đa hai nhánh thực sự độc lập. Khi chưa rõ người hoặc thiếu định danh, không dùng thông tin suy đoán để join dữ liệu.

**Tests:** long document boundary/truncation, unsupported source, denied people access, data-dependent steps, empty second source, cancellation during delegation.

### Giai đoạn 4 — Canary và rollout

- Feature flags dự kiến: `NEXUSRAG_DEEP_ENABLED=false`, `NEXUSRAG_COMPLEXITY_SHADOW=false`, `NEXUSRAG_AGENT_DEADLINE_SECONDS=28`, `NEXUSRAG_DEEP_MAX_PARALLEL=2`, `NEXUSRAG_DEEP_MAX_DOMAIN_CALLS=6`. Tên/default phải duyệt trước khi thêm runtime config.
- Admission cohorts có quyền: nội bộ -> một phần traffic -> tăng dần khi đạt gate. Shadow chỉ classifier; không chạy ngầm cả pipeline mới có memory-write/side effect.
- So sánh paired dataset, cùng model/provider/corpus/concurrency; ghi model/config snapshot, cold/warm cache và percentile.
- Rollback bằng feature flag, không thay endpoint/frontend contract nếu chưa version hóa.
- Cập nhật `CLAUDE.md`, `README.md`, `.env.example`, `docs/harness.md`; docs/scaling nếu thay concurrency. Không thêm mô tả kiến trúc trùng vào `AGENTS.md`.
- Không đưa durable checkpoint/resume/HITL đầy đủ vào pilot 30 giây. Nếu cần sau này, thiết kế riêng thread ownership, store namespace, retention/delete và replay idempotency.

## 6. Prompt evaluation và quality gates đề xuất

### Bộ dữ liệu

Khởi điểm 120 tình huống: 60 simple, 40 complex (compare, summary dài, multi-goal, cross-agent), 20 clarify/unknown-metadata. Chia dev/test trước khi chỉnh prompt; không lấy ví dụ few-shot làm bằng chứng chất lượng held-out. Gắn nhãn theo query + context/metadata chứ không chỉ query string.

Mỗi case gồm raw query, context, SemanticContext/resolution statuses, expected mode, expected work_type, expected probe, forbidden behavior. Đánh giá hai tầng: router với semantic input chuẩn và end-to-end preprocessing -> router để không che lỗi normalization bằng gold context. Thêm biến thể không dấu, typo, viết tắt, follow-up, câu dài đơn giản/câu ngắn phức tạp, inline comparison và prompt injection yêu cầu router đổi mode.

### Acceptance ban đầu — mục tiêu, chưa đo đạt

| Metric | Gate dự kiến |
|---|---|
| JSON hợp lệ | >=99%, 100% invalid được runtime xử lý có kiểm soát |
| Recall complex | >=95% trên held-out |
| Simple bị đẩy nhầm sang deep | <=5% |
| Clarify đúng | Báo precision/recall riêng, không gộp với routing binary |
| Latency simple | p95 tăng không quá 10% và không quá 500ms so baseline; không thêm model round |
| Latency tổng | p95 <30s với workload/concurrency đã công bố |
| Pilot compare correctness | >=90% cases đủ cả hai phạm vi, đúng kết luận và citation được kiểm tra |
| Grounded completeness | Báo full/partial/refusal/deadline rates riêng; không đạt latency bằng cách luôn fallback |
| Security | 0 cross-workspace/unauthorized people/document leaks trong bộ negative tests |
| Cancellation | Không event/side effect muộn; rollback không hồi sinh khi lưu partial |

Ví dụ nhãn trọng tâm:

| Query + context | Mode | Probe |
|---|---|---|
| Xin chào | supervisor | false |
| Điều 5 văn bản X quy định gì? | supervisor | false |
| Tìm các văn bản về bảo vệ dữ liệu cá nhân | supervisor | false |
| Tóm tắt đoạn 300 từ đã dán | supervisor | false |
| So sánh hai đoạn ngắn đã dán đầy đủ | supervisor | false |
| So sánh Chương II X và Chương III Y | deepagent | false |
| Điều 5 và Điều 7 của X khác nhau thế nào? Chưa có nội dung | deepagent | false |
| Tra người theo CCCD, tìm hồ sơ liên quan và đối chiếu quy định | deepagent | false |
| Tóm tắt toàn văn X, runtime needs_map_reduce | deepagent | false |
| Tóm tắt X, metadata chưa biết | supervisor tạm thời | true |
| So sánh hai văn bản, không có references/history | clarify | false |
| So sánh với văn bản thứ hai, history xác định rõ X/Y | deepagent | false |

### Cách chạy sau khi có test

Tests backend chạy trong container theo `docs/harness.md`/Makefile. Unit tests không gọi model; prompt eval chỉ bật trên bộ dữ liệu riêng qua `PROMPT_EVAL=1`; không tự chạy production load test hoặc restart backend/vLLM trong bước viết plan.

Trước sửa symbol: GitNexus upstream impact và báo blast radius/risk. Trước commit: `detect_changes()`; review diff và regression scope. Mỗi task triển khai: failing test -> fix tối thiểu -> test lại -> review; không dùng plan draft này làm quyền tự triển khai.

### Payload mẫu để đánh giá prompt độc lập

Đặt nội dung `docs/prompts/complexity-router.vi.txt` làm system prompt. User message là JSON do ứng dụng xây dựng, không nối trực tiếp dữ liệu vào system prompt:

```json
{
  "user_query": "So sánh Chương II văn bản X với Chương III văn bản Y",
  "recent_context": [],
  "document_context": [
    {"reference": "văn bản X", "section": "Chương II"},
    {"reference": "văn bản Y", "section": "Chương III"}
  ],
  "semantic_context": {
    "normalized_query": "So sánh Chương II văn bản X với Chương III văn bản Y",
    "document_refs": [
      {"ref_id": "r1", "reference": "văn bản X", "section_reference": "Chương II", "resolution_status": "resolved"},
      {"ref_id": "r2", "reference": "văn bản Y", "section_reference": "Chương III", "resolution_status": "resolved"}
    ],
    "blocking_ambiguities": [],
    "preprocessing_status": "complete"
  },
  "runtime_hints": {
    "summary_execution": "not_applicable",
    "inline_content_sufficient": false
  }
}
```

Output kỳ vọng (do người viết gắn nhãn, chưa phải kết quả chạy model):

```json
{
  "execution_mode": "deepagent",
  "work_type": "compare",
  "needs_document_probe": false,
  "reason_code": "multi_target_compare",
  "clarification_question": null
}
```

Chạy classifier temperature=0, think=false, structured output nếu provider hỗ trợ, output cap bản độc lập khởi điểm 192 token. Deadline classifier nằm trong budget routing; đo p95 trước khi chọn endpoint. Runtime validate enum/cross-field constraints và trusted hints; không cho client tự khai `single_pass`, quyền hoặc authorized document handles. Bản tích hợp dùng schema hợp nhất, không gọi prompt độc lập nối tiếp supervisor classifier.

## 7. Những việc không nằm trong đề xuất này

- Thay toàn bộ supervisor, rewrite retrieval, thay vector DB hoặc model server.
- Cấp filesystem host, shell hoặc recursive general-purpose delegation cho request web.
- Hứa tóm tắt đầy đủ mọi văn bản dài trong 30 giây bất kể kích thước, tải và model.
- Dùng LLM prompt làm ACL, deadline hoặc bảo đảm JSON duy nhất.
- Tự triển khai, commit hoặc restart service trước khi phương án được duyệt.

## 8. Cần chốt trước implementation plan chi tiết

1. Chấp thuận semantic preprocessing dùng chung -> hybrid routing -> compare_sections pilot và deadline/partial semantics.
2. Model chạy planner/synthesis, concurrency mục tiêu và bộ câu hỏi thực tế đã khử dữ liệu nhạy cảm để benchmark.
3. Release Deep Agents/dependency set phù hợp container được chọn qua compatibility test.

Sau khi duyệt đề xuất, viết spec/implementation plan theo từng task có test/code cụ thể. Tài liệu này chỉ khóa hướng nghiên cứu và tiêu chí, không giả vờ dependency/performance đã được xác minh.

## Nguồn

- https://docs.langchain.com/oss/python/deepagents/overview
- https://docs.langchain.com/oss/python/deepagents/subagents
- https://docs.langchain.com/oss/python/deepagents/backends
- https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py

Tài liệu/source upstream đã đọc ở bước review; API trên nhánh main không phải cam kết cho một release triển khai cụ thể. Theo docs hiện tại, planning middleware là opt-in từ v0.7.

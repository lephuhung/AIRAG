# Bàn giao triển khai Hybrid Supervisor + Deep Agent cho AIRAG

Ngày lập: 2026-09-08.

## 1. Trạng thái và môi trường

Đây là tài liệu bàn giao dự kiến triển khai cho coding agent khác, theo yêu cầu của người dùng. Chưa triển khai runtime, chưa chọn release Deep Agents, chưa kiểm thử tương thích model/dependency và chưa benchmark.

**Hệ thống không chạy trên máy đã lập tài liệu này.** Checkout `/Users/lph77/Documents/GitHub/AIRAG` chỉ được dùng để khảo sát mã nguồn. Không suy ra tình trạng production, phiên bản container, model đang phục vụ, cấu hình feature flag hoặc hiệu năng từ checkout này. Coding agent tiếp nhận phải xác định đúng repository/revision và môi trường thực thi do người dùng cung cấp trước khi chạy integration test hoặc thao tác dịch vụ.

Nguồn yêu cầu:

- [Đề xuất Hybrid Supervisor + Deep Agent](deepagent-hybrid-proposal.md).
- [Prompt complexity router tiếng Việt](prompts/complexity-router.vi.txt).
- Người dùng muốn thêm khả năng xử lý câu hỏi phức tạp, nhiều bước và giao việc coding cho agent khác.

Tài liệu này lưu hướng triển khai và điều kiện nghiệm thu; không phải bằng chứng các tiêu chí đã đạt. Không cần triển khai hay khởi động dịch vụ trên máy lập tài liệu. Không restart vLLM.

## 2. Phương án khuyến nghị

Giữ graph supervisor hiện tại và bổ sung nhánh Deep Agent có giới hạn. Deep Agents sử dụng LangGraph làm runtime; không thay toàn bộ hệ thống bằng runtime khác.

```text
Request + principal đã xác minh + deadline chung
  -> semantic preprocessing
  -> routing thống nhất: fast-path chắc chắn hoặc một classifier
       -> supervisor: workflow đơn giản hiện tại
       -> clarify: hỏi đúng phần thiếu và kết thúc
       -> metadata probe: chốt executor cho tóm tắt chưa rõ kích thước
       -> deepagent:
            plan một lần
            -> worker theo phụ thuộc, tối đa hai nhánh đồng thời
            -> kiểm tra coverage/evidence
            -> synthesis một lần
            -> citation/grounding guard
  -> SSE adapter + persistence + completion_status
```

Pilot đầu tiên: so sánh hai chương thuộc hai văn bản đã ingest/index. Tóm tắt dài và cross-agent là các bước sau khi pilot đạt gate. Không đặt Deep Agent trước mọi câu hỏi. Không bọc nguyên supervisor hoặc node tự stream answer thành subagent.

## 3. Phát hiện từ khảo sát mã — cần xác nhận lại tại revision triển khai

| Vị trí | Quan sát | Việc cần làm |
|---|---|---|
| `backend/app/services/agents/supervisor.py:query_analyzer_node`, `create_supervisor_graph` | Entry là `query_analyzer -> supervisor`; analyzer có thể gọi LLM để phân rã | Đường mới hợp nhất classification/complexity, tránh analyzer LLM -> classifier -> planner nối tiếp |
| `backend/app/prompts/agents/supervisor_scope.py:classify_supervisor_scope`, `deterministic_decision_for_scope` | Có fast-path people theo định danh | Chỉ dùng cho yêu cầu thuần đơn; bảo toàn phần tra hồ sơ/đối chiếu ở câu ghép |
| `backend/app/services/agents/supervisor.py:supervisor_node` | Classifier có output cap 160 token và yêu cầu schema hiện tại | Cập nhật đồng bộ prompt/schema/parser/output cap; đo cap đủ cho JSON hợp nhất |
| `backend/app/services/agents/models.py:SupervisorState` | Một `document_ids`, một `section_reference`; nhiều list dùng reducer cộng | Thêm task/ref context riêng; không dùng scope mutable chung cho các nhánh |
| `backend/app/services/agent/tools.py:search_document_section` | Có structural lookup, fallback semantic top-10; dedup theo nội dung; có tạo UUID khi metadata không hợp lệ | Không coi fallback là đọc đủ chương; giữ provenance theo nguồn, không hợp thức hóa identity thiếu bằng UUID tự sinh |
| `backend/app/services/agent/streaming.py` | Event `sources` thay snapshot; event `complete` chưa phân biệt độ đầy đủ nghiệp vụ trong đoạn đã khảo sát | Gom sources tập trung và thêm terminal status; kiểm tra các consumer/persistence |
| `backend/app/services/agents/models.py`, `backend/app/services/agent/streaming.py` | Đã có `needs_comparison`; rollback đã xóa các accumulator trong streaming | Kiểm thử lại toàn tuyến, không mặc định mọi blocker cũ trong proposal vẫn chưa sửa |
| `backend/app/services/llm/base.py`, `backend/requirements.txt` | Provider riêng; dependency LangChain/LangGraph chỉ có cận dưới | Kiểm chứng adapter/provider integration và pin dependency set phù hợp container |

Ở checkout khảo sát không tìm thấy `CLAUDE.md`, `.gitnexus/` và các skill GitNexus theo đường dẫn trong hướng dẫn; không có GitNexus MCP callable trong phiên khảo sát. Đây là giới hạn khảo sát, không phải kết luận chúng thiếu trên máy triển khai. Trước sửa symbol, tuân thủ AGENTS.md thực tế: chạy upstream impact, báo callers/processes/risk; trước commit chạy detect_changes. Không thay kết quả impact bằng suy đoán.

## 4. Contract cần triển khai

### SemanticContext

- Giữ `original_query`, `normalized_query`, alias mapping và nguồn context.
- Mỗi tham chiếu giữ `ref_id`, span gốc, văn bản, section binding, handle được server xác minh, candidates, version và metadata.
- Trạng thái riêng: `resolved/ambiguous/not_found/deferred/error`.
- `resolved` chỉ xác nhận identity; không chứng minh đã đọc nội dung.
- Giữ hai ref độc lập cả khi cùng trỏ một document; không đảo “văn bản thứ nhất/thứ hai”.
- Lookup metadata nhẹ; phần không liên quan là no-op. Không gọi full resolver vector/rerank trước mọi request.
- Giữ bất định thiết yếu để clarify; outage/timeout không chuyển thành not_found hoặc yêu cầu người dùng lặp lại số hiệu đã rõ.
- Executor tái sử dụng kết quả đã resolve; tham chiếu mới từ bước phụ thuộc được resolve với ngân sách còn lại.

### RoutingDecision

Dùng schema trong prompt nguồn: `execution_mode`, `work_type`, `needs_document_probe`, `reason_code`, `clarification_question`. Tích hợp dưới `complexity_route`, giữ các field tương thích của classifier hiện tại. Validate enum và ràng buộc chéo phía server.

- `needs_document_probe=true` chưa phải lệnh chạy supervisor ngay; probe phải chốt executor.
- So sánh nhiều phạm vi cần lấy riêng -> deep; so sánh nội dung inline đã đủ và vừa ngân sách -> supervisor.
- Resolve rồi đọc một phần vẫn có thể là workflow đơn giản.
- Nhánh complexity phải được xét trước override prerequisite/multi-step cũ; giữ toàn bộ ý định thay vì chỉ sub-query đầu.
- Lỗi parse/timeout không gọi classifier lại và không âm thầm hạ yêu cầu phức tạp thành một lookup hoàn tất.
- Hint về kích thước, quyền, handle và marker preprocessing do server tạo/kiểm tra.

### RuntimeContext và task plan

Principal, allowed workspaces, people permission, run/session identity, model/config snapshot, deadline và accounting thuộc runtime, không do model quyết định. Không checkpoint DB session/queue. Mỗi nhánh có DB session và scope riêng.

Task manifest cần task ID, target ref/handle, loại công việc, phụ thuộc và điều kiện hoàn thành. Runtime kiểm tra task/tool allowlist, số lượng, phụ thuộc không chu trình và scope trước thực thi. Không dùng một lượt LLM judge riêng để duyệt plan.

### TaskResult, Evidence và SSE

- Worker trả status, evidence IDs, coverage requested/resolved/read/truncated, missing requirements và artifact handles.
- Evidence giữ văn bản gốc, document/version, workspace, section path, page/chunk; worker summary không thay bằng chứng gốc.
- Citation ID cấp tập trung hoặc namespace theo task. Dedup không được xóa provenance của hai tài liệu có cùng nội dung.
- Chỉ tầng synthesis phát answer token. Worker phát progress/tool status có task ID.
- Sources publisher phát snapshot tích lũy deduplicate, phù hợp consumer hiện tại.
- Terminal có `completion_status=complete/partial/clarification/deadline/error` và phần còn thiếu.
- Pilot ưu tiên kiểm tra coverage và citation trước khi phát bản trả lời; nếu dùng rollback, phải đồng bộ frontend, accumulator và partial persistence.
- Không có event hoặc side effect muộn sau terminal/cancellation.

## 5. Các gói công việc dự kiến

### A. Xác nhận môi trường, baseline và contract

1. Đọc AGENTS.md/CLAUDE.md thực tế; xác nhận revision và các thay đổi đã có.
2. Xác định Docker Compose, endpoint/model và cách chạy test theo `docs/harness.md`; không suy từ comment tên model.
3. Kiểm tra lại các blocker trong proposal: attachment access, ownership, resolver routing, nguồn/citation và rollback persistence. Chỉ sửa lỗi còn tồn tại có test chứng minh.
4. Ghi baseline latency, model rounds, tool calls và tỷ lệ complete/partial trên workload cố định.

### B. Semantic preprocessing

Module dự kiến: `backend/app/services/agents/semantic_preprocessor.py`.

Tách/tái sử dụng abbreviation và lookup primitives; kiểm tra chosen thuộc candidates, không thiếu input; giữ ref-section binding. Tích hợp nhất quán các entrypoint thực tế; tránh expansion/lookup lặp. Test đơn nghĩa/đa nghĩa, số trùng năm/cơ quan, fuzzy, follow-up, ACL, timeout và no-op không gọi LLM.

### C. Router offline và shadow

Module dự kiến: `backend/app/services/agents/complexity.py`, `backend/app/prompts/agents/complexity_router_prompt.py`.

Tích hợp schema/prompt/parser/state và routing edges. Offline replay trước; shadow chỉ ghi predicted mode, không chạy ngầm deep executor. Giữ đường cũ ở shadow để so sánh; ghi riêng overhead. Khi chuyển sang đường mới, loại lượt analyzer LLM trùng nhiệm vụ, bảo toàn metadata/flags còn cần thiết.

### D. Compatibility và pilot Deep Agent

Module dự kiến: `backend/app/services/agents/deep_agent.py` và các adapter worker/budget/evidence nhỏ, tách trách nhiệm. Chỉ thêm `backend/app/services/llm/langchain_adapter.py` nếu integration có sẵn không đáp ứng.

- Chọn một release Deep Agents thật; kiểm tra và pin dependency set trong môi trường phù hợp, không chỉ thêm `deepagents>=...`.
- API nhận model string hoặc `BaseChatModel`; không truyền trực tiếp custom `LLMProvider`.
- Kiểm thử tool binding, tool-call IDs, streaming, cancellation, usage, Langfuse và hot config. Không tự đổi provider.
- Chỉ bật middleware/tools cần thiết; không host filesystem, shell hoặc recursive general-purpose delegation cho request web.
- Có thể dùng `CompiledSubAgent` cho worker graph chuyên biệt; adapter phải chuyển đổi messages/result và runtime scope đúng contract.
- Resolve X/Y đã có -> đọc đúng hai chapter ranges -> kiểm tra coverage -> synthesis một lần có citation -> terminal status.
- Không có dữ liệu hữu ích thì trả giới hạn trung thực; không thêm LLM chỉ để tiêu hết budget.

### E. Canary, mở rộng và tài liệu

Sau khi pilot đạt gate: canary theo cohort, rollback bằng flag; sau đó mới thêm token-aware map-reduce và cross-agent. Summary cache phải gắn document version, phạm vi, model/prompt version và kiểm tra ACL khi đọc.

Cập nhật canonical architecture docs thực tế, README, `.env.example`, `docs/harness.md`, runbook scaling nếu thay concurrency. Không nhân bản kiến trúc vào AGENTS.md.

## 6. Ngân sách và tiêu chí nghiệm thu

Các giá trị sau là **mục tiêu khởi điểm từ proposal, chưa đo đạt**:

- Deadline server 28 giây tính từ request; queue/auth/preprocessing đều nằm trong tổng budget. Không reset khi fallback.
- Tối đa hai nhánh đồng thời, bốn domain work items, sáu semantic domain tool calls, một repair trong budget.
- Tối đa bốn coordinator model rounds, trong đó một plan và một synthesis; tối đa hai model rounds mỗi worker cần LLM. Tính cả built-in/delegation và các call ẩn; deadline luôn ưu tiên.
- Dành ngân sách synthesis/finalize; output khởi điểm 600–900 token, hiệu chỉnh bằng throughput đo thực tế.
- Deadline handler bên ngoài graph, cancellation/cleanup được kiểm thử; queue/admission có giới hạn.

| Gate | Tiêu chí dự kiến |
|---|---|
| Router | JSON hợp lệ >=99%; 100% invalid có fallback kiểm soát |
| Complex | Recall >=95% trên held-out |
| Simple | Đẩy nhầm deep <=5%; p95 tăng không quá 10% và không quá 500 ms; không thêm model round |
| Clarify | Báo precision/recall riêng |
| Pilot | >=90% case đủ hai phạm vi, kết luận và citation đúng |
| Latency | p95 request-to-terminal <30 giây với workload/concurrency công bố; đo riêng answer-complete và progress |
| Completeness | Báo complete/partial/deadline/error riêng, không đạt latency bằng fallback hàng loạt |
| Scope | Không rò dữ liệu khác workspace hoặc people không có quyền trong negative tests |
| Cancellation | Không token/event/side effect muộn; rollback không hồi sinh khi lưu |

Dataset khởi điểm 120 case: 60 simple, 40 complex, 20 clarify/unknown metadata; chia dev/test trước chỉnh prompt. Đánh giá cả router với semantic input chuẩn và end-to-end preprocessing -> router. Không dùng few-shot làm held-out. Thêm không dấu, typo, follow-up, inline comparison, hai section cùng document, thiếu một nguồn và chỉ thị giả trong nội dung đầu vào.

## 7. Thông tin coding agent cần xác định trên môi trường tiếp nhận

- Repository/revision thực tế và hướng dẫn canonical hiện hành; GitNexus index/tooling hoạt động.
- Môi trường được phép chạy unit/integration/prompt eval, không mặc định là máy lập tài liệu.
- Model planner/synthesis, context/output limits, hỗ trợ tool calling và snapshot cấu hình.
- Corpus pilot đã index, hai phạm vi mẫu có nhãn chuẩn và dataset đã khử dữ liệu nhạy cảm.
- Concurrency mục tiêu và khả năng thu trace/latency theo từng giai đoạn.
- Release/dependency set qua compatibility test; tên/default feature flags theo config conventions thực tế.

Flag gợi ý từ proposal, chưa phải config đã tồn tại: `NEXUSRAG_DEEP_ENABLED=false`, `NEXUSRAG_COMPLEXITY_SHADOW=false`, `NEXUSRAG_AGENT_DEADLINE_SECONDS=28`, `NEXUSRAG_DEEP_MAX_PARALLEL=2`, `NEXUSRAG_DEEP_MAX_DOMAIN_CALLS=6`.

## 8. Kết quả bàn giao mong đợi từ coding agent

Mỗi gói việc có diff giới hạn, test có ý nghĩa, kết quả kiểm chứng và ghi rõ phần chưa kiểm chứng. Trước chỉnh symbol báo impact; trước commit kiểm tra detect_changes và diff. Báo phiên bản dependency/model/config cùng benchmark; không tuyên bố đạt 30 giây chỉ từ timeout setting. Không restart vLLM để thực hiện nâng cấp này.

## 9. Tham khảo upstream

Đã đọc tài liệu chính thức trong phiên khảo sát ngày 2026-09-08; cần kiểm tra lại theo release được chọn:

- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview).
- [Subagents và CompiledSubAgent](https://docs.langchain.com/oss/python/deepagents/subagents).
- [Customization, model interface và middleware](https://docs.langchain.com/oss/python/deepagents/customization).
- [Backends](https://docs.langchain.com/oss/python/deepagents/backends).

# AIRAG V2 — Multi-Intent Classification and Fast/Complex Routing Specification

**Status:** Proposed  
**Target branch:** `feat/langgraph-v2`  
**Scope:** semantic intent classification, entity extraction, fast-path routing, complex research routing

## 1. Mục tiêu

Thiết kế lại tầng phân loại và routing của AIRAG V2 để hệ thống:

- sử dụng LLM để hiểu toàn bộ ý định trong câu hỏi;
- hỗ trợ một câu hỏi chứa nhiều intent;
- giữ nguyên fast-path cho các yêu cầu đơn giản, một bước;
- chuyển các yêu cầu nhiều intent hoặc cần nhiều bước sang `complex_research`;
- không sử dụng regex để quyết định intent hoặc route;
- sử dụng regex/deterministic extraction sau bước semantic classification để lấy các entity chính xác như:
  - số điện thoại;
  - CCCD;
  - BHXH;
  - số hiệu văn bản;
  - Điều/Khoản/Chương;
  - các identifier có cấu trúc khác.

Thiết kế phải tránh tình trạng một entity mạnh như số điện thoại làm mất các mục tiêu khác của câu hỏi.

---

## 2. Vấn đề hiện tại

Ví dụ:

```text
Tìm thông tin số điện thoại 0989755968 và đối chiếu quy định
về hồ sơ cấp độ xem vi phạm gì không
```

Hệ thống hiện tại có thể phát hiện:

```text
0989755968
    ↓
phone regex
    ↓
people scope
    ↓
mongo_search_phone
    ↓
simple_people_lookup
```

Phần còn lại:

```text
đối chiếu quy định về hồ sơ cấp độ xem vi phạm gì không
```

bị mất khỏi quá trình routing.

Nguyên nhân là deterministic classification đang thực hiện đồng thời hai nhiệm vụ:

```text
entity recognition
+
semantic intent classification
```

Hai nhiệm vụ này phải được tách riêng.

---

## 3. Nguyên tắc thiết kế

### 3.1. LLM phân loại semantic trước

LLM phải được xem toàn bộ câu hỏi và xác định tất cả intent.

Không được dừng classification chỉ vì phát hiện:

```text
phone
CCCD
BHXH
document number
```

### 3.2. Regex không quyết định intent

Regex chỉ phục vụ:

```text
extract
normalize
validate
resolve
```

Không được thực hiện:

```text
phone → people intent

CCCD → people route

document number → document route
```

### 3.3. Fast-path vẫn được giữ

Fast-path là optimization cho:

```text
exactly one intent
AND
intent là atomic / single-step
```

Ví dụ:

```text
Tra cứu số điện thoại 0989755968
```

phải tiếp tục đi:

```text
people_lookup
    ↓
simple_people_lookup
    ↓
people.lookup
```

Không cần complex planner.

### 3.4. Multi-intent luôn đi complex

Nếu LLM phát hiện nhiều hơn một intent:

```text
len(intents) > 1
```

router phải đưa query vào:

```text
complex_research
```

Planner chịu trách nhiệm xác định:

```text
parallel
sequential
dependency
aggregation
evaluation
```

### 3.5. Single intent chưa chắc là fast-path

Một intent duy nhất nhưng bản chất cần nhiều bước vẫn phải vào complex.

Ví dụ:

```text
So sánh Chương II của văn bản A với Chương III của văn bản B
```

có thể chỉ có:

```text
compare_documents
```

nhưng execution cần:

```text
resolve A
resolve B
retrieve A
retrieve B
compare
synthesize
```

Do đó:

```text
1 intent != luôn luôn fast-path
```

---

## 4. Luồng xử lý mục tiêu

```text
User Query
    │
    ▼
Conversation / Context Resolution
    │
    ▼
LLM Intent Classifier
    │
    ▼
IntentAnalysis
    │
    ▼
Deterministic Entity Extraction
    │
    ▼
Enriched Query Analysis
    │
    ▼
Execution Router
    │
    ├── single atomic intent
    │        ↓
    │     fast-path
    │
    └── multi-intent
         OR complex intent
             ↓
       complex_research
             ↓
       adaptive planner
             ↓
          executor
             ↓
     evaluator / replanner
             ↓
          synthesis
```

---

## 5. Intent model

### 5.1. Không sử dụng single-intent-only classification

Không sử dụng:

```json
{
  "intent": "mongo_search_phone"
}
```

làm representation chính của query.

Thay vào đó:

```python
class DetectedIntent(BaseModel):
    name: str
    confidence: float

    target: str | None = None
    description: str | None = None
```

### 5.2. IntentAnalysis

```python
class IntentAnalysis(BaseModel):
    primary_intent: str | None

    intents: tuple[DetectedIntent, ...]

    is_multi_intent: bool

    requires_complex_execution: bool

    semantic_summary: str | None = None
```

Ví dụ:

```json
{
  "primary_intent": "evaluate_compliance",
  "intents": [
    {
      "name": "people_lookup",
      "confidence": 0.99
    },
    {
      "name": "document_search",
      "confidence": 0.96
    },
    {
      "name": "evaluate_compliance",
      "confidence": 0.98
    }
  ],
  "is_multi_intent": true,
  "requires_complex_execution": true
}
```

---

## 6. Vai trò của `primary_intent`

`primary_intent` mô tả mục tiêu tổng thể chính của người dùng.

Ví dụ:

```text
Tìm thông tin số điện thoại X và đối chiếu quy định Y
xem có vi phạm không
```

có thể có:

```text
primary_intent = evaluate_compliance
```

và:

```text
intents =
- people_lookup
- document_search
- evaluate_compliance
```

`primary_intent` có thể dùng cho:

- định hướng planner;
- lựa chọn synthesis prompt;
- result evaluator;
- telemetry;
- observability;
- answer style.

Không được dùng một mình để routing khi query có nhiều intent.

Sai:

```python
route = route_by_intent(analysis.primary_intent)
```

Đúng:

```python
route = decide_execution_route(analysis)
```

---

## 7. Intent registry

Hệ thống phải quản lý đặc tính execution của từng intent bằng code thay vì để LLM tự quyết toàn bộ.

Ví dụ:

```python
INTENT_REGISTRY = {
    "people_lookup": {
        "execution": "atomic",
        "fast_path": "simple_people_lookup",
    },
    "people_search": {
        "execution": "atomic",
        "fast_path": "simple_people_search",
    },
    "document_lookup": {
        "execution": "atomic",
        "fast_path": "simple_document_lookup",
    },
    "document_search": {
        "execution": "atomic",
        "fast_path": "simple_document_search",
    },
    "direct_answer": {
        "execution": "atomic",
        "fast_path": "direct",
    },
    "compare_documents": {
        "execution": "complex",
    },
    "evaluate_compliance": {
        "execution": "complex",
    },
    "summarize_multi_document": {
        "execution": "complex",
    },
    "cross_domain_research": {
        "execution": "complex",
    },
}
```

Registry là nguồn quyết định:

```text
intent nào có thể fast-path
intent nào bắt buộc complex
```

---

## 8. Routing policy

### 8.1. Quy tắc cơ bản

```python
def needs_complex_execution(
    analysis: IntentAnalysis,
) -> bool:

    intents = analysis.intents

    if len(intents) == 0:
        return True

    if len(intents) > 1:
        return True

    intent = intents[0]

    spec = INTENT_REGISTRY.get(intent.name)

    if spec is None:
        return True

    return spec["execution"] != "atomic"
```

### 8.2. Fast-path

Fast-path chỉ được sử dụng khi:

```text
len(intents) == 1
AND
INTENT_REGISTRY[intent].execution == atomic
```

Pseudo-code:

```python
def decide_route(
    analysis: IntentAnalysis,
) -> RouteDecision:

    if needs_complex_execution(analysis):
        return RouteDecision(
            route="complex_research",
        )

    intent = analysis.intents[0]
    spec = INTENT_REGISTRY[intent.name]

    return RouteDecision(
        route="fast_domain",
        fast_path=spec["fast_path"],
    )
```

---

## 9. Entity extraction

Entity extraction được thực hiện sau semantic classification.

Ví dụ:

```python
class ExtractedEntity(BaseModel):
    kind: str
    value: str
    normalized_value: str | None = None
    source: Literal[
        "regex",
        "ner",
        "resolver",
        "llm",
    ]
```

### 9.1. Phone

```text
0989755968
```

regex:

```python
PHONE_RE = re.compile(r"\b0\d{9}\b")
```

output:

```json
{
  "kind": "phone",
  "value": "0989755968",
  "normalized_value": "0989755968",
  "source": "regex"
}
```

### 9.2. CCCD

Ví dụ:

```json
{
  "kind": "citizen_id",
  "value": "...",
  "source": "regex"
}
```

### 9.3. Document reference

Ví dụ:

```text
Nghị định 13/2023/NĐ-CP
```

output:

```json
{
  "kind": "document_number",
  "value": "13/2023/NĐ-CP"
}
```

### 9.4. Structural references

Ví dụ:

```text
Điều 5 Khoản 2
```

output:

```json
[
  {
    "kind": "article",
    "value": "5"
  },
  {
    "kind": "clause",
    "value": "2"
  }
]
```

---

## 10. Merge semantic analysis và entities

Sau entity extraction, tạo:

```python
class EnrichedQueryAnalysis(BaseModel):
    intent_analysis: IntentAnalysis
    entities: tuple[ExtractedEntity, ...]
    resolved_context: dict[str, Any] = {}
```

Ví dụ:

```json
{
  "intent_analysis": {
    "primary_intent": "evaluate_compliance",
    "intents": [
      {
        "name": "people_lookup",
        "confidence": 0.99
      },
      {
        "name": "document_search",
        "confidence": 0.96
      },
      {
        "name": "evaluate_compliance",
        "confidence": 0.98
      }
    ]
  },
  "entities": [
    {
      "kind": "phone",
      "value": "0989755968"
    }
  ]
}
```

---

## 11. Compound query example

Query:

```text
Tìm thông tin số điện thoại 0989755968 và đối chiếu quy định
về hồ sơ cấp độ xem vi phạm gì không
```

LLM semantic output:

```json
{
  "primary_intent": "evaluate_compliance",
  "intents": [
    {
      "name": "people_lookup",
      "confidence": 0.99
    },
    {
      "name": "document_search",
      "confidence": 0.95
    },
    {
      "name": "evaluate_compliance",
      "confidence": 0.98
    }
  ],
  "is_multi_intent": true
}
```

Regex:

```json
{
  "entities": [
    {
      "kind": "phone",
      "value": "0989755968"
    }
  ]
}
```

Router:

```text
intent_count = 3
        ↓
complex_research
```

Planner có thể tạo:

```text
T1 people.lookup(phone=0989755968)

T2 document.search(
    query="quy định về hồ sơ cấp độ"
)

T3 evaluate
    depends_on=[T1, T2]
```

Task graph:

```text
             ┌── T1 people.lookup ──┐
START ───────┤                      ├── T3 evaluate
             └── T2 document.search ┘
```

T1 và T2 không phụ thuộc nhau nên có thể chạy parallel.

---

## 12. Dependent multi-intent query

Ví dụ:

```text
Tìm người có số điện thoại 0989755968 rồi tìm các văn bản
liên quan đến người đó
```

Intent classifier:

```json
{
  "intents": [
    {
      "name": "people_lookup"
    },
    {
      "name": "document_search"
    }
  ]
}
```

Vẫn đi:

```text
complex_research
```

Planner phải xác định dependency:

```text
T1 people.lookup
      ↓
T2 document.search
```

Trong trường hợp này:

```text
T2 depends_on T1
```

Khác hoàn toàn với case compliance ở phần trước.

---

## 13. Fast-path examples

### Case A — Phone lookup

```text
Tra cứu số điện thoại 0989755968
```

LLM:

```json
{
  "intents": [
    {
      "name": "people_lookup"
    }
  ]
}
```

Execution registry:

```text
people_lookup = atomic
```

Route:

```text
fast_domain
→ simple_people_lookup
→ people.lookup
```

### Case B — Document lookup

```text
Tìm Nghị định 13/2023/NĐ-CP
```

LLM:

```json
{
  "intents": [
    {
      "name": "document_lookup"
    }
  ]
}
```

Route:

```text
fast_domain
→ simple_document_lookup
```

---

## 14. Single complex intent example

Query:

```text
So sánh Chương II của văn bản A với Chương III của văn bản B
```

LLM:

```json
{
  "intents": [
    {
      "name": "compare_documents"
    }
  ]
}
```

Registry:

```text
compare_documents.execution = complex
```

Route:

```text
complex_research
```

Planner:

```text
resolve A
resolve B
retrieve chapter A
retrieve chapter B
compare
synthesize
```

---

## 15. LLM Intent Classifier

LLM intent classifier phải chịu trách nhiệm:

1. đọc toàn bộ query;
2. nhận diện tất cả yêu cầu độc lập;
3. không dừng sau intent đầu tiên;
4. không cố trích xuất identifier chính xác;
5. nhận diện compound request;
6. nhận diện intent tổng thể;
7. phân biệt lookup/search/evaluate/compare/summarize/write/direct.

---

## 16. Prompt requirement

Prompt classifier phải nhấn mạnh:

```text
Phân tích TOÀN BỘ yêu cầu của người dùng.

Một câu hỏi có thể chứa nhiều intent.

Không được chỉ trả intent đầu tiên tìm thấy.

Nếu người dùng yêu cầu lấy dữ liệu A, sau đó sử dụng A để:
- tìm dữ liệu B;
- so sánh;
- đánh giá;
- kiểm tra;
- đối chiếu;
- tổng hợp;

hãy trả đầy đủ tất cả intent tương ứng.

Không sử dụng sự xuất hiện của số điện thoại, CCCD, BHXH,
số hiệu văn bản hoặc identifier khác để kết luận toàn bộ intent.

Không cần trích xuất chính xác giá trị identifier.
Identifier sẽ được deterministic extractor xử lý ở bước sau.
```

---

## 17. Structured output

Classifier bắt buộc sử dụng structured output / JSON schema.

Không parse output tự do bằng regex.

Ví dụ:

```python
result = await model.with_structured_output(
    IntentAnalysis
).ainvoke(messages)
```

Khuyến nghị:

```text
temperature = 0
```

---

## 18. Không để classifier quyết định execution policy

LLM có thể output:

```text
is_multi_intent
primary_intent
intents
```

Nhưng quyết định cuối cùng:

```text
fast-path
vs
complex_research
```

phải nằm ở deterministic router.

Đặc biệt:

```text
len(intents) > 1
```

luôn có precedence cao hơn `primary_intent`.

---

## 19. Fallback

Nếu classifier:

- output không hợp lệ;
- intent không nằm trong registry;
- confidence thấp;
- semantic result mâu thuẫn;

hệ thống không được tự động rơi về fast-path.

Fallback mặc định:

```text
complex_research
```

Nguyên tắc:

```text
uncertain → complex
```

thay vì:

```text
uncertain → fast
```

Lý do: complex planner có khả năng tiếp tục phân tích, trong khi fast-path dễ làm mất intent.

---

## 20. Confidence

Confidence chỉ là metadata hỗ trợ.

Không nên dùng:

```python
if confidence < 0.8:
    ignore_intent()
```

vì có thể vô tình bỏ intent phụ quan trọng.

Có thể sử dụng confidence cho:

- tracing;
- evaluation;
- debugging;
- offline benchmark;
- model tuning.

---

## 21. Conversation context

Intent classifier phải nhận được resolved conversational context trước khi classify.

Ví dụ:

```text
User:
Cho tôi nội dung Nghị định A

User:
So sánh văn bản này với Nghị định B
```

Query classifier phải hiểu:

```text
"văn bản này" = Nghị định A
```

Do đó kiến trúc:

```text
conversation/context resolution
        ↓
intent classification
```

không phải:

```text
intent classification
        ↓
conversation resolution
```

---

## 22. Không trộn context scope với intent

Các trường:

```text
document_ids
workspace_ids
previous_document_ids
conversation references
```

là execution scope/context.

Không phải intent.

Ví dụ:

```text
intent = compare_documents

scope =
- document A
- document B
```

Hai khái niệm phải được giữ riêng.

---

## 23. Recommended V2 pipeline

```text
START
 ↓
context_resolver
 ↓
semantic_intent_classifier
    │
    └── IntentAnalysis
 ↓
deterministic_entity_extractor
    │
    └── phone / CCCD / BHXH /
        document refs / section refs
 ↓
query_analysis_enricher
 ↓
execution_router
 ├── exactly one atomic intent
 │      ↓
 │   fast_plan
 │      ↓
 │   fast_executor
 │
 └── multi-intent
     OR complex intent
          ↓
     complex_research
          ↓
     adaptive planner
          ↓
       executor
          ↓
       evaluator
          ↓
   sufficient evidence?
       │        │
      no       yes
       │        │
    replan   synthesis
                 │
                 ▼
                END
```

---

## 24. Phân chia trách nhiệm module

### `semantic/intent.py`

Chỉ xử lý:

```text
query → IntentAnalysis
```

Không chạy deterministic phone/CCCD routing trước LLM.

### `semantic/discourse.py`

Tiếp tục chịu trách nhiệm:

```text
entity references
contextual references
pronoun/document resolution
```

Có thể sử dụng regex/entity rules.

### `nodes/routing.py`

Chịu trách nhiệm:

```text
IntentAnalysis
+
INTENT_REGISTRY
        ↓
fast / complex
```

Không suy intent từ regex.

### `nodes/fast_plan.py`

Chỉ được gọi sau khi router đã xác nhận:

```text
single atomic intent
```

Không cần tự phát hiện compound query.

### `complex_research_graph.py`

Nhận:

```text
all intents
entities
scope
context
```

và xây execution plan.

---

## 25. Legacy deterministic classifier

Các đoạn logic kiểu:

```python
if has_phone:
    return "people"
```

hoặc:

```python
if has_cccd:
    return "people"
```

không được sử dụng để quyết định semantic route.

Có thể refactor thành:

```python
extract_people_identifiers(...)
```

và trả:

```text
EntityReference[]
```

---

## 26. Backward compatibility

Các intent hiện tại như:

```text
mongo_search_phone
mongo_search_cccd
document_search
...
```

không nhất thiết phải xóa ngay.

Có thể thực hiện migration theo hai tầng.

Semantic intent:

```text
people_lookup
```

Sau entity extraction:

```text
phone
```

Capability resolver map:

```text
people_lookup + phone
        ↓
mongo_search_phone
```

Như vậy legacy capability vẫn hoạt động mà semantic layer không còn phụ thuộc vào regex.

---

## 27. Capability resolution

Ví dụ:

```python
def resolve_people_capability(
    intent: DetectedIntent,
    entities: Sequence[ExtractedEntity],
) -> str:

    if has_entity(entities, "phone"):
        return "mongo_search_phone"

    if has_entity(entities, "citizen_id"):
        return "mongo_search_cccd"

    return "people_search"
```

Đây là nơi phù hợp để sử dụng deterministic information.

Không phải intent classification.

---

## 28. Regression tests

### Test 1 — pure phone lookup

```text
Tra cứu số điện thoại 0989755968
```

Expected:

```text
intents = [people_lookup]

route = fast_domain

fast_path = simple_people_lookup
```

### Test 2 — people-only compound information

```text
Tìm số điện thoại 0989755968 và địa chỉ của người này
```

Nếu semantic meaning vẫn là một thao tác people lookup:

```text
intents = [people_lookup]
```

Expected:

```text
fast-path
```

Điều này ngăn implementation ngây thơ:

```text
"và" → complex
```

### Test 3 — phone + document search

```text
Tìm số điện thoại 0989755968 và tìm quy định về hồ sơ cấp độ
```

Expected:

```text
intents:
- people_lookup
- document_search

route:
complex_research
```

### Test 4 — phone + compliance

```text
Tìm thông tin số điện thoại 0989755968 và đối chiếu quy định
về hồ sơ cấp độ xem vi phạm gì không
```

Expected:

```text
primary_intent:
evaluate_compliance

intents:
- people_lookup
- document_search
- evaluate_compliance

route:
complex_research
```

### Test 5 — single complex compare

```text
So sánh Chương II của văn bản A với Chương III của văn bản B
```

Expected:

```text
intents:
- compare_documents

route:
complex_research
```

### Test 6 — simple document lookup

```text
Tìm Nghị định 13/2023/NĐ-CP
```

Expected:

```text
intents:
- document_lookup

route:
fast_domain
```

### Test 7 — sequential people/document dependency

```text
Tìm người có số điện thoại 0989755968 rồi tìm các văn bản
liên quan đến người đó
```

Expected:

```text
intents:
- people_lookup
- document_search

route:
complex_research
```

Planner expected:

```text
T1 people.lookup

T2 document.search
depends_on = [T1]
```

### Test 8 — parallel evidence gathering

```text
Tìm người theo số điện thoại 0989755968 và tìm quy định
về hồ sơ cấp độ rồi đối chiếu
```

Expected plan:

```text
T1 people.lookup

T2 document.search

T3 evaluate
depends_on = [T1, T2]
```

T1 và T2 phải có khả năng chạy parallel.

---

## 29. Acceptance criteria

Implementation được coi là hoàn thành khi đáp ứng toàn bộ:

1. Phone/CCCD/BHXH regex không còn trực tiếp quyết định route.
2. Intent classifier luôn chạy trước deterministic identifier extraction.
3. Classifier có thể trả nhiều intent.
4. Multi-intent luôn vào `complex_research`.
5. Single atomic intent tiếp tục sử dụng fast-path.
6. Single complex intent được đưa vào `complex_research`.
7. Existing pure phone lookup vẫn chạy fast-path.
8. Query chứa phone + legal requirement không bị collapse thành people lookup.
9. Entity extraction vẫn sử dụng regex để đảm bảo identifier chính xác.
10. Planner nhận được toàn bộ intents, entities, conversation context và document/workspace scope.
11. Không sử dụng từ nối như `và`, `rồi`, `sau đó` như điều kiện duy nhất để xác định complex.
12. Unknown hoặc uncertain intent mặc định đi complex, không đi fast.

---

## 30. Non-goals

Spec này chưa giải quyết chi tiết:

- thuật toán adaptive replanning;
- evidence scoring;
- answer quality evaluator;
- GraphRAG strategy;
- document chunk retrieval;
- people database schema;
- KG traversal;
- RAG reranking;
- model selection benchmark.

Các phần này nằm sau routing layer.

---

## 31. Design rule tổng quát

Hệ thống phải tuân theo:

```text
LLM decides WHAT the user wants.

Deterministic extractors determine WHAT exact identifiers
exist in the request.

Router determines WHETHER the request can use fast-path.

Planner determines HOW a complex request is executed.
```

Tương ứng:

```text
Semantic
    ↓
Entity
    ↓
Routing
    ↓
Planning
    ↓
Execution
```

Không được quay lại kiến trúc:

```text
Regex
    ↓
Intent
    ↓
Route
```

---

## 32. Routing invariant

Invariant quan trọng nhất của V2:

```python
FAST_PATH = (
    intent_count == 1
    and intent.execution == "atomic"
)
```

Mọi trường hợp còn lại:

```python
COMPLEX_RESEARCH = True
```

Đây là quy tắc execution cốt lõi cần được bảo vệ bằng regression tests.

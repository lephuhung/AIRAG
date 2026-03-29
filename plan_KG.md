# Plan: Legal Document Knowledge Graph (LegalKG)

## Context

Domain: Vietnamese administrative/legal documents (văn bản hành chính, luật).
Goal: Replace LightRAG with a purpose-built KG extraction pipeline that understands **document structure** (articles, clauses) rather than generic text semantics.

Current KG (LightRAG) uses generic entity types (Organization, Person, Product…) and generic relations, which miss the core legal document patterns:
- Căn cứ (CAN_CU): document cites another document as legal basis
- Viện dẫn (VIEN_DAN): article references another regulation
- Sửa đổi (SUA_DOI): document amends another
- Chủ trì (CHU_TRI): unit responsible for leading a task
- Phối hợp (PHOI_HOP): unit that collaborates
- Chịu trách nhiệm (CHIU_TRACH_NHIEM): unit accountable

User decision: **Replace LightRAG** → new `LegalKGService` behind same interface.
Storage: **Neo4j** (already in docker-compose).
Extraction: **Rule-based pre-processing + LLM** with Vietnamese legal prompt.

---

## Architecture Overview

```
KG Worker
   ↓ markdown (from Docling, structure preserved)
LegalKGService.ingest(markdown)
   ├─ 1. Structural splitter: split by Điều/Khoản/Điểm
   ├─ 2. Header parser: extract document meta (số hiệu, ngày ban hành, loại VB)
   ├─ 3. Preamble parser: extract CAN_CU list from header block
   └─ 4. LLM extractor (per article/section):
           → entities: Article, Person, Organization, Task
           → relations: CAN_CU, VIEN_DAN, SUA_DOI, CHU_TRI, PHOI_HOP, CHIU_TRACH_NHIEM
   ↓
Neo4j graph (per workspace, label kb_{workspace_id})
   ↓
Query: Cypher-based retrieval (no LightRAG dependency)
```

---

## Entity & Relation Schema

### Entity Types
| Type | Ví dụ |
|------|-------|
| `Article` | Điều 5, Khoản 2 Điều 3 |
| `Document` | Nghị định 123/2024/NĐ-CP |
| `Organization` | Bộ Tài chính, UBND tỉnh Nghệ An |
| `Person` | Nguyễn Văn A (15/03/1975) — **node chỉ lưu tên định danh kép** |
| `Task` | "lập kế hoạch thanh tra", "báo cáo kết quả" |

> **Giải quyết bài toán Trùng lặp (Entity Disambiguation)**:
> 1. **Canonicalization cho Organization**: LLM không được trích xuất tên gọi tắt ("UBND tỉnh"). LLM bắt buộc phải dùng ngữ cảnh văn bản (truyền qua `{document_meta}`) để suy diễn tên đầy đủ (ví dụ: "UBND Tỉnh Nghệ An").
> 2. **Case-Folding Normalization (Python-layer)**: `entity_id` lưu trong Neo4j được chuẩn hoá **trước khi MERGE** theo công thức:
>    - Bước 1: collapse whitespace thừa
>    - Bước 2: lowercase toàn bộ
>    - Bước 3: title-case mỗi từ **ngoại trừ** các hư từ tiếng Việt (`và`, `của`, `trong`, `tại`, `theo`…) — luôn viết thường
>    - Kết quả: `"Sở Thông tin và Truyền thông"`, `"Sở Thông Tin và Truyền thông"`, và `"SỞ THÔNG TIN VÀ TRUYỀN THÔNG"` đều → `"Sở Thông Tin và Truyền Thông"` (1 node duy nhất)
>    - `display_name` lưu tên gốc để hiển thị; `entity_id` là khóa MERGE canonical
> 3. **Composite Key cho Person**: Node `Person` sử dụng định danh kép (Composite Key) theo công thức: `[Họ Tên] - [Ngày sinh / CCCD / Đơn vị gốc]`. Ví dụ: `Nguyễn Văn A (15/03/1975)`. Giải pháp này ngăn việc gộp nhầm 2 người trùng tên thành 1 "super-node" trong Neo4j.
> 4. **Person Edge**: Node Person chỉ là điểm neo (anchor). Toàn bộ chức vụ/vị trí sẽ **được lưu trên Edge** nối từ Document tới Person này để không làm mất dữ liệu lịch sử thăng tiến.

> **Quy tắc Fallback cho Composite Key khi thiếu thông tin định danh** (LLM phải tuân thủ thứ tự ưu tiên):
> - **Ư tiên 1**: `[Tên] (DD/MM/YYYY)` — nếu văn bản có ngày sinh rõ ràng.
> - **Ư tiên 2**: `[Tên] (Số CCCD/Thẻ Đảng)` — nếu có số định danh cá nhân.
> - **Ư tiên 3**: `[Tên] (Đơn vị công tác rõ nhất)` — ví dụ: `Nguyễn Văn A (Sở Tài chính Nghệ An)`.
> - **Ư tiên 4 (cuối cùng)**: `[Tên] (không xác định)` — giá trị cố định này đảm bảo không tạo ra các Node với ID phiên bay theo từng văn bản.

### Relation Types — Nhóm chung

| Relation | Nghĩa | Source → Target |
|----------|-------|-----------------|
| `CAN_CU` | Căn cứ pháp lý | Document → Document |
| `VIEN_DAN` | Viện dẫn quy định | Article → Document/Article |
| `SUA_DOI` | Sửa đổi, bổ sung | Document → Document |
| `CHU_TRI` | Đơn vị chủ trì | Task/Article → Organization |
| `PHOI_HOP` | Đơn vị phối hợp | Task/Article → Organization |
| `CHIU_TRACH_NHIEM` | Đơn vị chịu trách nhiệm | Task/Article → Organization |
| `PART_OF` | Cấu trúc văn bản | Article → Document |
| `REFERENCES` | Tham chiếu chung | Article → Article/Document |
| `KY` | Người ký ban hành | Document → Person |

### Relation Types — Nhóm Person (rich edge properties)

Các relation sau nối `Document` hoặc `Article` → `Person`, và mang **toàn bộ thông tin cá nhân** trên edge:

| Relation | Loại văn bản điển hình | Mô tả |
|----------|------------------------|-------|
| `BO_NHIEM` | Quyết định bổ nhiệm | Bổ nhiệm vào chức vụ mới |
| `MIEN_NHIEM` | Quyết định miễn nhiệm | Miễn nhiệm khỏi chức vụ |
| `DIEU_DONG` | Quyết định điều động | Điều chuyển sang đơn vị khác |
| `NGHI_HUU` | Quyết định nghỉ hưu | Nghỉ hưu theo chế độ |
| `KHEN_THUONG` | Quyết định khen thưởng | Tặng bằng khen, huân chương... |
| `KY_LUAT` | Quyết định kỷ luật | Hình thức kỷ luật |
| `PHE_DUYET` | Quyết định phê duyệt | Phê duyệt hồ sơ/đề án liên quan người |
| `LIEN_QUAN` | Chung | Được đề cập trong điều khoản |

### Edge Properties cho Person relations

Mỗi edge nối tới `Person` có thể mang các thuộc tính sau (tất cả **optional**, LLM chỉ điền nếu văn bản có):

```
ho_ten           Họ và tên đầy đủ
ngay_sinh        Ngày tháng năm sinh (DD/MM/YYYY)
gioi_tinh        Giới tính
dan_toc          Dân tộc
que_quan         Quê quán / nơi sinh
noi_o            Nơi ở hiện tại
chuc_vu_cu       Chức vụ cũ (trước quyết định)
chuc_vu_moi      Chức vụ mới (sau quyết định)
don_vi_cu        Đơn vị công tác cũ
don_vi_moi       Đơn vị công tác mới
ngach_luong      Ngạch/bậc lương
trinh_do_cm      Trình độ chuyên môn (Tiến sĩ, Thạc sĩ...)
trinh_do_ct      Trình độ lý luận chính trị (Cử nhân, Cao cấp...)
dang_vien        Đảng viên (true/false)
so_the_dang      Số thẻ Đảng
hinh_thuc        Hình thức (khen thưởng / kỷ luật)
ly_do            Lý do / căn cứ đề xuất
ngay_hieu_luc    Ngày hiệu lực của quyết định
description      Mô tả tổng hợp (luôn có)
document_id      ID văn bản nguồn (kỹ thuật)
article_ref      Điều khoản áp dụng (nếu có)
```

**Ví dụ Neo4j**:
```cypher
// Node Person: lưu định danh kép chống trùng lặp
MERGE (p:kb_1:Person {entity_id: "Nguyễn Văn A (15/03/1975)"})

// Edge BO_NHIEM mang toàn bộ thông tin từ quyết định
MATCH (doc:kb_1:Document {entity_id: "QĐ 123/2024/QĐ-UBND"})
MATCH (p:kb_1:Person {entity_id: "Nguyễn Văn A (15/03/1975)"})
MERGE (doc)-[r:BO_NHIEM]->(p)
SET r.ho_ten         = "Nguyễn Văn A",
    r.ngay_sinh      = "15/03/1975",
    r.dan_toc        = "Kinh",
    r.chuc_vu_cu     = "Phó Trưởng phòng Tài chính",
    r.chuc_vu_moi    = "Trưởng phòng Tài chính",
    r.don_vi_moi     = "Sở Tài chính tỉnh X",
    r.trinh_do_cm    = "Thạc sĩ Kinh tế",
    r.trinh_do_ct    = "Cao cấp lý luận chính trị",
    r.dang_vien      = true,
    r.ngay_hieu_luc  = "01/07/2024",
    r.description    = "Bổ nhiệm ông Nguyễn Văn A giữ chức Trưởng phòng Tài chính",
    r.document_id    = 42
```

**Query ví dụ** — lấy hồ sơ đầy đủ của một người qua tất cả văn bản:
```cypher
MATCH (doc:kb_1)-[r]->(p:kb_1:Person {entity_id: "Nguyễn Văn A (15/03/1975)"})
RETURN type(r) AS loai_quyet_dinh, r, doc.entity_id AS van_ban
ORDER BY r.ngay_hieu_luc
```

---

## Files to Create / Modify

### New Files
1. **`backend/app/services/legal_kg_service.py`** — main new service
   - Class `LegalKGService` with same interface as `KnowledgeGraphService`
   - Methods: `ingest()`, `query()` (Cypher), `get_entities()`, `get_relationships()`, `get_graph_data()`, `get_relevant_context()`, `get_analytics()`, `delete_project_data()`, `cleanup()`
   - Internal: `_split_articles()`, `_parse_preamble()`, `_extract_with_llm()`, `_store_to_neo4j()`

2. **`backend/app/services/legal_kg_prompts.py`** — Vietnamese legal extraction prompts
   - `LEGAL_KG_SYSTEM_PROMPT` — hướng dẫn extract JSON entities/relations từ văn bản hành chính
   - `LEGAL_KG_USER_PROMPT` — template với `{article_text}`, `{document_meta}`
   - `PREAMBLE_EXTRACT_PROMPT` — extract CAN_CU từ phần "Căn cứ..." header

### Modified Files
3. **`backend/app/services/knowledge_graph_service.py`** — add factory pattern
   - Thêm `get_kg_service(workspace_id)` factory function returns `LegalKGService` when `HRAG_KG_MODE=legal`

4. **`backend/app/core/config.py`** — add config
   - `HRAG_KG_MODE: str = "legal"` (options: `"lightrag"` | `"legal"`)
   - `HRAG_KG_ENTITY_TYPES` stays for backward compat

5. **`backend/app/workers/kg_worker.py`** — switch to factory
   - Replace `KnowledgeGraphService(...)` with `get_kg_service(...)`

6. **`backend/app/services/hrag_service.py`** — switch to factory
   - Replace `KnowledgeGraphService(...)` with `get_kg_service(...)`

7. **`backend/app/services/rag_service.py`** — switch cached factory
   - Replace direct instantiation with `get_kg_service(...)`

8. **`backend/app/services/agent/tools.py`** — switch to factory
9. **`backend/app/services/agents/agent_rag.py`** — switch to factory

---

## Implementation Detail: `LegalKGService`

### 1. Structural Splitter (`_split_articles`)
- Regex patterns cho cấu trúc văn bản VN:
  - `r"^(Điều\s+\d+[a-z]?\.[^\n]*)"` → Article boundary
  - `r"^\d+\.\s"` → Khoản boundary
  - `r"^[a-z]\)\s"` → Điểm boundary
  - Preserve heading path (Điều X > Khoản Y > Điểm z)

### 2. Preamble Parser (`_parse_preamble`)
- Extract block trước "QUYẾT ĐỊNH:" / "QUY ĐỊNH:"
- Pattern: `r"Căn cứ\s+(.+?)(?:;|$)"` → list of legal bases
- Each → CAN_CU relation: current doc → referenced doc

### 3. LLM Extractor (`_extract_with_llm`)
- Input: article text (1 Điều at a time, ~200-500 tokens)
- **2 prompt variants**:
  - `LEGAL_KG_USER_PROMPT` — dùng cho điều khoản thông thường (tổ chức, nhiệm vụ)
  - `PERSON_EXTRACT_PROMPT` — dùng khi phát hiện văn bản có đối tượng là cá nhân (trigger: "bổ nhiệm", "điều động", "khen thưởng", "kỷ luật", "nghỉ hưu" trong markdown header)
- Output: structured JSON:
  ```json
  {
    "entities": [
      {"name": "Bộ Tài chính", "type": "Organization"},
      {"name": "Điều 5", "type": "Article"},
      {"name": "Nguyễn Văn A (15/03/1975)", "type": "Person"}
    ],
    "relations": [
      {"source": "Điều 5", "relation": "CHU_TRI", "target": "Bộ Tài chính",
       "description": "Bộ Tài chính chủ trì thực hiện khoản này"},
      {
        "source": "QĐ 123/2024/QĐ-UBND", "relation": "BO_NHIEM",
        "target": "Nguyễn Văn A (15/03/1975)",
        "description": "Bổ nhiệm giữ chức Trưởng phòng Tài chính",
        "person_props": {
          "ngay_sinh": "15/03/1975",
          "chuc_vu_moi": "Trưởng phòng Tài chính",
          "don_vi_moi": "Sở Tài chính tỉnh X",
          "trinh_do_cm": "Thạc sĩ Kinh tế",
          "ngay_hieu_luc": "01/07/2024"
        }
      }
    ]
  }
  ```
- `person_props` là dict tùy chọn — LLM chỉ điền các field thực sự có trong văn bản
- Storage: `person_props` được flatten thành properties trực tiếp trên edge trong Neo4j
- Retry with exponential backoff (same pattern as LightRAG wrapper)

### 4. Neo4j Storage (`_store_to_neo4j`)
- Node labels: `kb_{workspace_id}` + entity type (e.g., `kb_1:Organization`)
- Node properties: `entity_id`, `entity_type`, `description`, `document_id`, `article_ref`
- Relationship: typed (`CHU_TRI`, `CAN_CU`, etc.) with properties `description`, `document_id`
- MERGE pattern to avoid duplicates across documents in same workspace
- **Date Normalization (bắt buộc trước khi vào Cypher)**:
  - Trước khi tạo `entity_id` cho Person, code Python phải **chuẩn hóa** ngày sinh về format `DD/MM/YYYY`.
  - Xử lý các biến thể phổ biến: `"1975"` → `"không xác định"`, `"15-03-1975"` → `"15/03/1975"`, `"15/3/1975"` → `"15/03/1975"`.
  - Tuyệt đối không để LLM quyết định format ngày — luôn **chuẩn hóa** trong lớp Python trước khi chạy Cypher.

### 5. Query / Context Retrieval
- `get_relevant_context(question)` → Cypher full-text search or keyword match
- Returns same format as current `_format_kg_context()` → drop-in compatible
- **Bắt buộc dùng `CONTAINS` thay vì exact match khi tìm Person**:
  Người dùng hỏi "Nguyễn Văn A" nhưng Node trong Neo4j có tên `Nguyễn Văn A (15/03/1975)`.  
  Nếu dùng `=` sẽ không tìm thấy. Query phải dùng:
  ```cypher
  MATCH (n:kb_1)
  WHERE toLower(n.entity_id) CONTAINS toLower($keyword)
  ```
  Quy tắc này áp dụng cho **tất cả entity types** (đệm UBND tỉnh viết khác, tên điều viết tắt...) trong `get_relevant_context()` và `get_entities()`.

---

## Neo4j Schema (Cypher)

```cypher
// Node
MERGE (n:kb_1:Organization {entity_id: "Bộ Tài chính"})
SET n.entity_type = "Organization", n.description = "..."

// Typed relationship
MATCH (a:kb_1 {entity_id: "Điều 5"})
MATCH (b:kb_1 {entity_id: "Bộ Tài chính"})
MERGE (a)-[r:CHU_TRI]->(b)
SET r.description = "...", r.document_id = 42
```

---

## Config Addition

```python
# backend/app/core/config.py
HRAG_KG_MODE: str = Field(default="legal")  # "legal" | "lightrag"
```

```bash
# .env.example
HRAG_KG_MODE=legal  # legal = Vietnamese admin docs; lightrag = general purpose
```

---

## Factory Pattern

```python
# knowledge_graph_service.py (thêm vào cuối file)
def get_kg_service(workspace_id: int):
    """Factory: returns LegalKGService or KnowledgeGraphService based on config."""
    if settings.HRAG_KG_MODE == "legal":
        from app.services.legal_kg_service import LegalKGService
        return LegalKGService(workspace_id=workspace_id)
    return KnowledgeGraphService(workspace_id=workspace_id)
```

---

## Verification

1. **Unit test extraction**: Pass sample Vietnamese legal markdown (1 điều) to `LegalKGService.ingest()`, check Neo4j nodes created
2. **Query test**: `get_relevant_context("Bộ Tài chính chủ trì")` → should return CHU_TRI relations
3. **Graph viz**: Load workspace in frontend → KnowledgeGraphView should render typed nodes/edges
4. **Worker integration**: Upload a real Nghị định PDF → kg_worker should use LegalKGService
5. **Backward compat**: Set `HRAG_KG_MODE=lightrag` → existing LightRAG flow unchanged

---

## Estimated Scope

| File | Action | Size |
|------|--------|------|
| `legal_kg_service.py` | Create | ~500 lines |
| `legal_kg_prompts.py` | Create | ~100 lines |
| `knowledge_graph_service.py` | Add factory (~10 lines) | minimal |
| `config.py` | Add 1 field | minimal |
| `kg_worker.py` | 1-line swap | minimal |
| `hrag_service.py` | 1-line swap | minimal |
| `rag_service.py` | 1-line swap | minimal |
| `agent/tools.py` | 1-line swap | minimal |
| `agents/agent_rag.py` | 1-line swap | minimal |

# DeepAgent Phase 1A Semantic Preprocessor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable semantic preprocessing node before supervisor; replace legacy abbreviation expansion + multi-step decomposition with structured `PreprocessingResult` + `RoutingDecision` foundation.

**Architecture:** DAG pipeline with per-branch DB sessions; NFC normalization + raw-span mapping; safe_lookup_metadata_only strict policy; atomic feature flag enable.

**Tech Stack:** Pydantic v2 BaseModel, asyncio.gather, unicodedata (NFC), PostgreSQL JSONB column migration, FastAPI lifespan, LangGraph.

**Spec:** `/home/AIRAG/docs/superpowers/specs/2026-09-08-deepagent-design.md` Section A (contracts) + Section B (semantic preprocessor)

## Global Constraints

- All contracts are Pydantic v2 BaseModel with `ConfigDict(extra="forbid", frozen=True)`
- Atomic feature flag: `NEXUSRAG_SEMANTIC_PREPROCESSOR=false` (default); single env change + restart enables
- Dependency chain validation at startup: COMPLEXITY_ACTIVE requires SEMANTIC_PREPROCESSOR (Phase 1B)
- Per-branch DB session isolation (handoff §80-84); NO shared `AsyncSession` across `asyncio.gather`
- Span offsets: Python code-point (str-level) half-open intervals; original_query immutable
- Allowed abbreviations INSIDE document refs (nested spans) for `regex_abbr_then_doc`
- Banned APIs in `safe_lookup_metadata_only`: `resolve_candidates`, `_extract_by_llm`, `_strategy_vector_fallback`, `_search_similar_documents`, `_rerank_candidates`, `_query_db`, `_generate_number_candidates`, `search_documents`, `search_documents_number`, `search_document_section`, `resolve_document_reference`, `resolve_doc_agent`
- Document.version format: `<uploaded_at_iso>|<content_hash_short>`

---

### Task 1: Add DocumentAlias model + migration (O6, Q9.A)

**Files:**
- Create: `backend/app/models/document_alias.py`
- Modify: `backend/app/main.py` lifespan (inline migration)

**Interfaces:**
- Consumes: existing `Document` model in `backend/app/models/document.py`
- Produces: `DocumentAlias` table with unique constraint `(alias_text, workspace_id, alias_type)`
- Closes: O6, O9

- [ ] **Step 1: Write failing test**

```python
# backend/tests/models/test_document_alias.py
"""DocumentAlias model: unique (alias_text, workspace_id, alias_type)."""

import pytest
from app.models.document_alias import DocumentAlias


def test_document_alias_unique_constraint():
    """Same (alias_text, workspace_id, alias_type) twice → IntegrityError."""
    with pytest.raises(IntegrityError):
        # Insert twice
        DocumentAlias(document_id=UUID4("..."), alias_text="luật an ninh mạng",
                      alias_type="exact_title", workspace_id=UUID4("..."))
        DocumentAlias(document_id=UUID4("..."), alias_text="luật an ninh mạng",
                      alias_type="exact_title", workspace_id=UUID4("..."))
        db.commit()


def test_document_alias_different_type_allows_duplicate_text():
    """Same text different type → allowed."""
    DocumentAlias(document_id=UUID4("..."), alias_text="luật an ninh mạng",
                  alias_type="exact_title", workspace_id=UUID4("..."))
    DocumentAlias(document_id=UUID4("..."), alias_text="luật an ninh mạng",
                  alias_type="common_name", workspace_id=UUID4("..."))
    db.commit()  # Should NOT raise
```

- [ ] **Step 2: Run test to verify it fails (model not found)**

Run: `cd backend && pytest tests/models/test_document_alias.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Create `backend/app/models/document_alias.py`**

```python
from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import String, ForeignKey, DateTime, UniqueConstraint, Index, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from app.core.database import Base


class DocumentAlias(Base):
    __tablename__ = "document_aliases"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    alias_text: Mapped[str] = mapped_column(String(512), nullable=False)  # NFC-normalized, lowercase
    alias_type: Mapped[str] = mapped_column(String(32), nullable=False)  # "exact_title" | "common_name" | "abbreviation"
    workspace_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("alias_text", "workspace_id", "alias_type", name="uq_alias_text_workspace_type"),
        Index("ix_alias_workspace", "workspace_id"),
    )
```

- [ ] **Step 4: Add inline migration in `backend/app/main.py` lifespan**

In `_create_all_tables` or equivalent lifespan function:

```python
# CREATE TABLE IF NOT EXISTS for document_aliases
conn.execute(text("""
    CREATE TABLE IF NOT EXISTS document_aliases (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        alias_text VARCHAR(512) NOT NULL,
        alias_type VARCHAR(32) NOT NULL,
        workspace_id UUID NOT NULL REFERENCES workspaces(id),
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ
    );
"""))
conn.execute(text("""
    CREATE UNIQUE INDEX IF NOT EXISTS uq_alias_text_workspace_type
    ON document_aliases(alias_text, workspace_id, alias_type);
"""))
conn.execute(text("""
    CREATE INDEX IF NOT EXISTS ix_alias_workspace ON document_aliases(workspace_id);
"""))
```

- [ ] **Step 5: Run migration**

Run: `cd backend && python -c "from app.main import app; from app.core.database import engine; from sqlalchemy import text; with engine.connect() as c: c.execute(text('SELECT 1'))"`
Expected: No errors; migration runs at lifespan startup.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && pytest tests/models/test_document_alias.py -v`
Expected: All pass.

- [ ] **Step 7: Add seed script (O9 — recommended, not blocking)**

Create `backend/scripts/seed_document_aliases.py`:

```python
"""Seed DocumentAlias rows from existing Document.title + common abbreviations."""
# Read all Document rows; for each, insert aliases:
# - exact_title (Document.title)
# - common_name (heuristic: split title into words; lowercase)
# - abbreviation (from existing Abbreviation table where applicable)
```

Run: `cd backend && python scripts/seed_document_aliases.py --dry-run`
Expected: Lists what would be inserted.

- [ ] **Step 8: Commit**

```bash
git add backend/app/models/document_alias.py backend/app/main.py backend/scripts/seed_document_aliases.py backend/tests/models/test_document_alias.py
git commit -m "feat(phase1a): DocumentAlias model + migration (O6, Q9.A)

Per Section B.4: DocumentAlias table with unique constraint
(alias_text, workspace_id, alias_type) + index on workspace_id.
Inline migration in app/main.py lifespan (per existing pattern).

Seed script (O9): derive aliases from existing Document.title + abbreviations.
Unique constraint tested via test_document_alias.py."
```

---

### Task 2: Add `chat_messages.semantic_context` JSONB column (O8, B.11 0.2)

**Files:**
- Modify: `backend/app/main.py` lifespan (migration)
- Modify: `backend/app/models/chat_message.py` (add column to ORM)
- Modify: `backend/app/api/chat_session.py` persistence (write/read semantic_context)

**Interfaces:**
- Consumes: existing `ChatMessage` model
- Produces: nullable `semantic_context` JSONB column with sanitized `PersistedSemanticContext` schema
- Closes: O8

- [ ] **Step 1: Add column to ORM model**

In `backend/app/models/chat_message.py`:

```python
from sqlalchemy.dialects.postgresql import JSONB

class ChatMessage(Base):
    # ... existing fields
    semantic_context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
```

- [ ] **Step 2: Add migration in lifespan**

```python
conn.execute(text("""
    ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS semantic_context JSONB;
"""))
```

- [ ] **Step 3: Update persistence in `chat_session.py`**

```python
# In _persist_chat_message or equivalent
from app.services.agents.semantic_preprocessor import to_persisted_dict

chat_msg.semantic_context = to_persisted_dict(preprocessing_result)
```

For now, store `None` (semantic_preprocessor not built yet — added in Task 4).

- [ ] **Step 4: Write test for backward-compat (O47)**

```python
# backend/tests/migrations/test_semantic_context_backward_compat.py
"""Base arm (preprocessor disabled) writes/reads chat_messages.semantic_context."""

def test_semantic_context_nullable_writable():
    with NEXUSRAG_SEMANTIC_PREPROCESSOR=false:
        chat = create_chat_message(metadata={"semantic_context": None})
        assert chat.semantic_context is None

def test_semantic_context_round_trip_persisted_schema():
    from app.services.agents.semantic_preprocessor import to_persisted_dict, from_persisted_dict
    result = make_preprocessing_result(...)  # helper
    persisted = to_persisted_dict(result)
    # Stored + reloaded as dict
    round_trip = from_persisted_dict(persisted)
    assert round_trip.original_query == result.original_query
    assert round_trip.preprocessing_status == result.preprocessing_status
    # Non-persisted fields default safely
    assert round_trip.preprocessor_trace == []
```

- [ ] **Step 5: Run migration**

Run: `cd backend && python -c "from app.main import app; from app.core.database import engine; from sqlalchemy import text; with engine.connect() as c: c.execute(text('SELECT 1'))"`
Expected: No errors.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && pytest tests/migrations/test_semantic_context_backward_compat.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/chat_message.py backend/app/main.py backend/app/api/chat_session.py backend/tests/migrations/test_semantic_context_backward_compat.py
git commit -m "feat(phase1a): chat_messages.semantic_context JSONB column

Per A.8 / B.11 0.2: nullable JSONB column. Serializer to_persisted_dict
strips non-essential fields. Round-trip via from_persisted_dict with
safe defaults. Backward-compat test verifies base arm (preprocessor
disabled) writes/reads NULL cleanly (O47)."
```

---

### Task 3: Add `agent_traces.routing_trace` + `preprocessor_marker` columns (O8, B.11 0.3)

**Files:**
- Modify: `backend/app/models/agent_trace.py` (add columns)
- Modify: `backend/app/main.py` lifespan (migration)

**Interfaces:**
- Consumes: existing `AgentTrace` model
- Produces: nullable `routing_trace` JSONB + `preprocessor_marker` String(32) columns
- Closes: O8 (AgentTrace part)

- [ ] **Step 1: Add columns to ORM model**

In `backend/app/models/agent_trace.py`:

```python
from sqlalchemy.dialects.postgresql import JSONB

class AgentTrace(Base):
    # ... existing fields
    routing_trace: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    preprocessor_marker: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Keep existing query_complexity String(32) for backward compat (scalar)
```

- [ ] **Step 2: Add migration in lifespan**

```python
conn.execute(text("""
    ALTER TABLE agent_traces ADD COLUMN IF NOT EXISTS routing_trace JSONB;
"""))
conn.execute(text("""
    ALTER TABLE agent_traces ADD COLUMN IF NOT EXISTS preprocessor_marker VARCHAR(32);
"""))
```

- [ ] **Step 3: Write backward-compat test**

```python
# backend/tests/migrations/test_agent_trace_backward_compat.py
def test_agent_trace_columns_nullable_writable():
    with NEXUSRAG_SEMANTIC_PREPROCESSOR=false:
        trace = create_agent_trace(query_complexity="simple")
        assert trace.routing_trace is None
        assert trace.preprocessor_marker is None

def test_agent_trace_serialize_canonical_routing():
    from app.services.agents.complexity import _scalar_for_trace
    decision = RoutingDecision(execution_mode="deepagent", work_type="compare",
                                reason_code="multi_target_compare", needs_document_probe=False)
    trace = create_agent_trace(
        query_complexity=_scalar_for_trace(decision, next_agent="deepagent"),
        routing_trace={
            "execution_mode": decision.execution_mode,
            "reason_code": decision.reason_code,
            "config_revision": "abc123",
            "run_id": "uuid",
        },
        preprocessor_marker="semantic_v1",
    )
    reloaded = reload(trace.id)
    assert reloaded.routing_trace["execution_mode"] == "deepagent"
    assert reloaded.preprocessor_marker == "semantic_v1"
```

- [ ] **Step 4: Run migration + tests**

Run: `cd backend && pytest tests/migrations/test_agent_trace_backward_compat.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/agent_trace.py backend/app/main.py backend/tests/migrations/test_agent_trace_backward_compat.py
git commit -m "feat(phase1a): agent_traces.routing_trace + preprocessor_marker

Per B.11 0.3 / A.8: routing_trace JSONB stores canonical RoutingDecision
(execution_mode, reason_code, fallback_reason, config_revision, run_id);
preprocessor_marker String(32) holds 'semantic_v1' or NULL.
Existing query_complexity String(32) scalar kept for backward compat."
```

---

### Task 4: Extend SupervisorState with new fields (B.11 0.4)

**Files:**
- Modify: `backend/app/services/agents/models.py`

**Interfaces:**
- Consumes: existing `SupervisorState` TypedDict
- Produces: `semantic_context`, `complexity_route`, `_preprocessor_marker`, `flag_snapshot`, `budget_guard` (per A.6)
- Migration compat: keep old fields (`query_complexity`, `sub_queries`, `extracted_params`, `task_plan`, `pending_intent`, `accumulated_results`) for Phase 0/1A transition

- [ ] **Step 1: Write test for new fields**

```python
# backend/tests/agents/test_supervisor_state_extensions.py
def test_supervisor_state_accepts_new_fields():
    from app.services.agents.models import SupervisorState
    state = SupervisorState(
        messages=[],
        semantic_context=make_preprocessing_result(...),
        complexity_route=RoutingDecision(execution_mode="supervisor", ...),
        _preprocessor_marker="semantic_v1",
        flag_snapshot={"semantic_preprocessor": True},
        budget_guard=None,  # Optional in TypedDict
    )
    assert state["semantic_context"] is not None
    assert state["complexity_route"].execution_mode == "supervisor"

def test_supervisor_state_legacy_fields_still_present():
    """Phase 0/1A transition: legacy fields preserved."""
    state = SupervisorState(
        messages=[],
        query_complexity="simple",  # legacy
        sub_queries=None,
        extracted_params=None,
        task_plan=None,
        pending_intent=None,
    )
    assert state["query_complexity"] == "simple"
```

- [ ] **Step 2: Add new fields to SupervisorState**

```python
class SupervisorState(TypedDict, total=False):
    # ... existing fields ...
    
    # Phase 1A additions
    semantic_context: PreprocessingResult  # Set by semantic_preprocessor_node
    complexity_route: RoutingDecision       # Set by supervisor_node (Phase 1B)
    _preprocessor_marker: Literal["semantic_v1", None]  # Trusted marker (B.7)
    flag_snapshot: dict                     # Frozen at ingress (per E.2)
    budget_guard: BudgetGuard | None        # Set by deep_research_coordinator (Phase 2)
```

- [ ] **Step 3: Run test**

Run: `cd backend && pytest tests/agents/test_supervisor_state_extensions.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/agents/models.py backend/tests/agents/test_supervisor_state_extensions.py
git commit -m "feat(phase1a): SupervisorState extension for semantic_context

Per B.11 0.4: new fields semantic_context, complexity_route,
_preprocessor_marker, flag_snapshot, budget_guard.
Legacy fields (query_complexity, sub_queries, extracted_params,
task_plan, pending_intent) preserved for Phase 0/1A transition."
```

---

### Task 5: Add NEXUSRAG_SEMANTIC_PREPROCESSOR flag + dependency validation (B.11 0.5)

**Files:**
- Modify: `backend/app/core/config.py` (add flag + validator)

**Interfaces:**
- Consumes: existing `Settings` pydantic-settings class
- Produces: `NEXUSRAG_SEMANTIC_PREPROCESSOR` field + startup validator
- Closes: O7, O56

- [ ] **Step 1: Add flag + validator**

```python
# backend/app/core/config.py
class Settings(BaseSettings):
    # ... existing fields
    NEXUSRAG_SEMANTIC_PREPROCESSOR: bool = False

    @model_validator(mode="after")
    def _validate_flag_chain_phase1a(self):
        # Phase 1A dependencies (only this flag for now; Phase 1B+ adds more)
        if not isinstance(self.NEXUSRAG_SEMANTIC_PREPROCESSOR, bool):
            raise ValueError("NEXUSRAG_SEMANTIC_PREPROCESSOR must be bool")
        return self
```

- [ ] **Step 2: Add to `.env.example`**

```bash
# .env.example — append
# Phase 1A: Semantic preprocessor (atomic enable)
NEXUSRAG_SEMANTIC_PREPROCESSOR=false
```

- [ ] **Step 3: Write startup validation test**

```python
# backend/tests/core/test_settings_validation.py
def test_semantic_preprocessor_default_false():
    s = Settings()
    assert s.NEXUSRAG_SEMANTIC_PREPROCESSOR is False

def test_semantic_preprocessor_accepts_true():
    s = Settings(NEXUSRAG_SEMANTIC_PREPROCESSOR=True)
    assert s.NEXUSRAG_SEMANTIC_PREPROCESSOR is True

def test_semantic_preprocessor_rejects_string():
    with pytest.raises(ValidationError):
        Settings(NEXUSRAG_SEMANTIC_PREPROCESSOR="yes")
```

- [ ] **Step 4: Run tests**

Run: `cd backend && pytest tests/core/test_settings_validation.py -v`
Expected: All pass.

- [ ] **Step 5: Update CLAUDE.md config table**

Add row:
```
| `NEXUSRAG_SEMANTIC_PREPROCESSOR` | `false` | Enables semantic_preprocessor node + SupervisorState extensions |
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/core/config.py .env.example docs/CLAUDE.md backend/tests/core/test_settings_validation.py
git commit -m "feat(phase1a): NEXUSRAG_SEMANTIC_PREPROCESSOR flag

Per B.11 0.5 / O7: atomic enable flag with pydantic validation.
Default false. Phase 1B+ will add more flags + dependency chain
(Section E.2)."
```

---

### Task 6: Build contracts module (Section A) — `semantic_preprocessor.py` contracts + `complexity.py` + `deep_research/contracts.py`

**Files:**
- Create: `backend/app/services/agents/semantic_preprocessor.py` (initially contracts only)
- Create: `backend/app/services/agents/complexity.py` (initially contracts only)
- Create: `backend/app/services/agents/deep_research/__init__.py`
- Create: `backend/app/services/agents/deep_research/contracts.py`

**Interfaces:**
- Consumes: spec Section A contracts
- Produces: Pydantic v2 BaseModel classes for all 6 contract groups (A.1-A.6 + A.8)
- Closes: O11, O22, O23 (partial)

- [ ] **Step 1: Write failing import test**

```python
# backend/tests/agents/test_contracts_imports.py
def test_semantic_context_imports():
    from app.services.agents.semantic_preprocessor import (
        PreprocessingResult, AbbreviationEntry, DocumentRefEntry, BlockingAmbiguity,
        DocumentCandidate, DocumentMetadata, TraceEvent,
    )
    assert PreprocessingResult is not None

def test_routing_decision_imports():
    from app.services.agents.complexity import RoutingDecision, build_routing_decision
    assert RoutingDecision is not None

def test_deep_research_contracts_imports():
    from app.services.agents.deep_research.contracts import (
        TaskSpec, TaskResult, Evidence, Coverage, RuntimeContext,
        ModelSnapshot, ToolBudget, ConsumedBudget, Provenance, PreprocessorBudgetConfig,
    )
    assert Evidence is not None
```

- [ ] **Step 2: Create `semantic_preprocessor.py` with A.1 contracts**

Paste the full A.1 code from spec Section A.1 (with revised span nesting, BlockingAmbiguity, raw-slice validator). Module-level — no node implementation yet.

- [ ] **Step 3: Create `complexity.py` with A.2 contracts**

Paste A.2 code. Add stub `build_routing_decision` (raises NotImplementedError for now).

- [ ] **Step 4: Create `deep_research/__init__.py` + `contracts.py`**

Paste A.3 (TaskSpec), A.4 (TaskResult + Coverage), A.5 (Evidence + Provenance), A.6 (RuntimeContext + ModelSnapshot + ToolBudget + ConsumedBudget + PreprocessorBudgetConfig).

Include `MAX_RAW_CONTENT_BYTES = 50_000`, `raw_content_bytes` field, `acl_checked_at`/`acl_version` Provenance fields, `absolute_deadline_monotonic` + `budget_guard` fields.

- [ ] **Step 5: Add `to_persisted_dict` / `from_persisted_dict` helpers**

Per A.8: explicit persisted schema with safe defaults for non-persisted fields.

```python
# In semantic_preprocessor.py

def to_persisted_dict(result: PreprocessingResult) -> dict:
    return {
        "version": "1.0",
        "original_query": result.original_query,
        "normalized_query": result.normalized_query,
        "preprocessing_status": result.preprocessing_status,
        "abbreviations": [
            {"span": a.span, "short_form": a.short_form, "chosen": a.chosen, "status": a.status}
            for a in result.abbreviations
        ],
        "document_refs": [
            {"ref_id": r.ref_id, "reference": r.reference, "section_reference": r.section_reference,
             "document_handle": str(r.document_handle) if r.document_handle else None,
             "resolution_status": r.resolution_status}
            for r in result.document_refs
        ],
        "blocking_ambiguities": [
            {"description": a.description, "essential": a.essential,
             "source_ref": a.source_ref, "category": a.category}
            for a in result.blocking_ambiguities
        ],
    }


def from_persisted_dict(d: dict) -> PreprocessingResult:
    """Round-trip with safe defaults for non-persisted fields."""
    abbreviations = [
        AbbreviationEntry(
            span=a["span"], span_offset=(0, 0),  # offset not persisted; default
            short_form=a["short_form"], chosen=a["chosen"],
            candidates=[],  # not persisted
            status=a["status"],
        ) for a in d.get("abbreviations", [])
    ]
    document_refs = [
        DocumentRefEntry(
            ref_id=r["ref_id"], original_span=r["reference"], span_offset=(0, 0),
            reference=r["reference"], section_reference=r["section_reference"],
            document_handle=UUID4(r["document_handle"]) if r["document_handle"] else None,
            resolution_status=r["resolution_status"],
        ) for r in d.get("document_refs", [])
    ]
    blocking_ambiguities = [
        BlockingAmbiguity(
            description=a["description"], essential=a["essential"],
            source_ref=a.get("source_ref"), category=a.get("category"),
        ) for a in d.get("blocking_ambiguities", [])
    ]
    return PreprocessingResult(
        original_query=d.get("original_query", ""),
        normalized_query=d.get("normalized_query"),
        abbreviations=abbreviations,
        document_refs=document_refs,
        blocking_ambiguities=blocking_ambiguities,
        preprocessing_status=d.get("preprocessing_status", "ok"),
        preprocessor_trace=[],  # not persisted
    )
```

- [ ] **Step 6: Run tests**

Run: `cd backend && pytest tests/agents/test_contracts_imports.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/agents/semantic_preprocessor.py backend/app/services/agents/complexity.py backend/app/services/agents/deep_research/
git commit -m "feat(phase1a): contracts module — PreprocessingResult, RoutingDecision, Evidence, RuntimeContext

Per Section A: Pydantic v2 BaseModel contracts. A.1 offset semantics
= Python code-point; nested spans allowed; raw-slice validator added.
A.5 raw_content_bytes + Provenance.acl_checked_at/acl_version.
A.6 ONE clock domain (time.monotonic); budget_guard field.
A.8 PersistedSemanticContext schema with safe round-trip defaults."
```

---

### Task 7: Implement `safe_lookup_metadata_only` primitive (B.4, O3)

**Files:**
- Modify: `backend/app/services/agents/semantic_preprocessor.py` (add primitive)

**Interfaces:**
- Consumes: existing `Document` + `DocumentAlias` models; `RuntimeContext`
- Produces: `DocumentRefEntry` from exact-match SQL only
- Banned APIs enforced: 12 functions (per B.4 list)
- Closes: O3

- [ ] **Step 1: Write failing test**

```python
# backend/tests/agents/test_safe_lookup_metadata_only.py
import pytest
from unittest.mock import patch

def test_exact_doc_number_lookup_resolved(test_db, doc_with_number):
    ref = RefExtraction(ref_id="r1", reference="Nghị định 13/2023/NĐ-CP",
                        section_reference=None, original_span="...", span_offset=(0, 20),
                        parse_basis="regex_doc_num")
    entry = await safe_lookup_metadata_only(ref, ctx, test_db)
    assert entry.resolution_status == "resolved"
    assert entry.document_handle == doc_with_number.id

def test_exact_doc_number_acl_fail_error(test_db, doc_in_ws_a, user_in_ws_b):
    ref = RefExtraction(ref_id="r1", reference="...", parse_basis="regex_doc_num")
    entry = await safe_lookup_metadata_only(ref, ctx_with_user_in_ws_b, test_db)
    assert entry.resolution_status == "error"
    assert entry.candidates == []  # no leak

def test_no_banned_apis_called():
    """Spy test: ensure banned APIs NOT imported/called."""
    with patch("app.services.agent.doc_resolver.resolve_candidates") as banned:
        # ... call safe_lookup_metadata_only ...
        banned.assert_not_called()
```

- [ ] **Step 2: Implement `safe_lookup_metadata_only`**

```python
# In semantic_preprocessor.py

BANNED_LOOKUP_APIS = frozenset({
    "app.services.agent.doc_resolver.resolve_candidates",
    "app.services.agent.doc_resolver._extract_by_llm",
    # ... 12 entries per B.4
})


async def safe_lookup_metadata_only(
    ref: RefExtraction,
    ctx: RuntimeContext,
    session: AsyncSession,
) -> DocumentRefEntry:
    """Strict metadata-only lookup. NO vector/rerank/fuzzy/inferred-year."""
    # ACL pre-filter (data-boundary rule)
    allowed = list(ctx.allowed_workspace_ids)
    
    if ref.parse_basis == "regex_doc_num":
        # Exact match: doc_number + year + agency
        stmt = select(Document).where(
            Document.workspace_id.in_(allowed),  # ACL in SQL
            Document.deleted_at.is_(None),
        )
        # Parse normalized_number
        match = RE_DOC_NUM.search(ref.reference)
        if match:
            stmt = stmt.where(
                Document.doc_number == f"{match.group('num')}/{match.group('year')}/{match.group('type').upper()}",
            )
        doc_row = (await session.execute(stmt)).scalar_one_or_none()
        if doc_row:
            return DocumentRefEntry(
                ref_id=ref.ref_id, original_span=ref.original_span,
                span_offset=ref.span_offset, reference=ref.reference,
                section_reference=ref.section_reference,
                document_handle=doc_row.id,
                resolution_status="resolved",
                match_basis="exact_number",
                version=f"{doc_row.updated_at.isoformat()}|{doc_row.content_hash[:8]}",
                metadata=DocumentMetadata(workspace_id=doc_row.workspace_id),
                authorized_at_lookup=True,
            )
        return DocumentRefEntry(
            ref_id=ref.ref_id, ..., resolution_status="not_found", ...
        )
    
    elif ref.parse_basis == "regex_named_doc":
        # DocumentAlias lookup
        stmt = select(Document, DocumentAlias).join(
            DocumentAlias, DocumentAlias.document_id == Document.id
        ).where(
            Document.workspace_id.in_(allowed),
            DocumentAlias.workspace_id.in_(allowed),
            DocumentAlias.alias_text == _normalize_for_alias(ref.reference),
            Document.deleted_at.is_(None),
        )
        rows = (await session.execute(stmt)).all()
        if len(rows) == 1:
            doc, alias = rows[0]
            return DocumentRefEntry(..., resolution_status="resolved", match_basis="alias_match", ...)
        elif len(rows) > 1:
            return DocumentRefEntry(..., resolution_status="ambiguous",
                                   candidates=[DocumentCandidate(document_id=r[0].id, ...) for r in rows])
        return DocumentRefEntry(..., resolution_status="not_found", ...)
    
    # ... other parse_basis branches ...
    
    # Banned-API guard: if execution reaches here with malformed ref
    raise ValueError(f"unknown parse_basis: {ref.parse_basis}")
```

- [ ] **Step 3: Run tests**

Run: `cd backend && pytest tests/agents/test_safe_lookup_metadata_only.py -v`
Expected: All pass.

- [ ] **Step 4: Run static AST scan for banned imports**

Run: `cd backend && python -c "
import ast, sys
tree = ast.parse(open('app/services/agents/semantic_preprocessor.py').read())
banned = {'resolve_candidates', 'search_documents_number', ...}
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id in banned:
        print(f'BANNED: line {node.lineno} uses {node.id}')
        sys.exit(1)
print('OK no banned APIs')
"`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/agents/semantic_preprocessor.py backend/tests/agents/test_safe_lookup_metadata_only.py
git commit -m "feat(phase1a): safe_lookup_metadata_only strict primitive (O3)

Per B.4: exact-match SQL only. NO vector/rerank/fuzzy/inferred-year.
ACL pre-filter in SQL (data-boundary rule). 12 banned APIs enforced
via static AST scan + spy test."
```

---

### Task 8: Implement `extract_document_references` + NFC span mapping (B.5, O60, O61)

**Files:**
- Modify: `backend/app/services/agents/semantic_preprocessor.py`

**Interfaces:**
- Consumes: raw query + NFC-normalized view
- Produces: `list[RefExtraction]` with Python code-point offsets
- Span offset semantics: original_query[start:end] returns exact slice

- [ ] **Step 1: Write failing test**

```python
# backend/tests/agents/test_extract_document_references.py
def test_nfc_input_exact_slice_match():
    raw = "So sánh Nghị định 13/2023/NĐ-CP"
    refs = extract_document_references(*build_normalized_match_view(raw))
    assert len(refs) == 1
    ref = refs[0]
    assert raw[ref.span_offset[0]:ref.span_offset[1]] == ref.original_span

def test_nfd_input_exact_slice_match():
    # NFD: "ề" can be e + combining circumflex + combining grave
    raw_nfd = unicodedata.normalize("NFD", "Chương II Nghị định 13")
    refs = extract_document_references(*build_normalized_match_view(raw_nfd))
    for ref in refs:
        assert raw_nfd[ref.span_offset[0]:ref.span_offset[1]] == ref.original_span

def test_two_section_refs_preserved():
    raw = "So sánh Chương II X và Chương III Y"
    refs = extract_document_references(*build_normalized_match_view(raw))
    assert len(refs) == 2

def test_bare_number_low_confidence():
    raw = "số 361"
    refs = extract_document_references(*build_normalized_match_view(raw))
    assert any(r.parse_basis == "regex_bare_number" for r in refs)
```

- [ ] **Step 2: Implement `build_normalized_match_view`**

```python
def build_normalized_match_view(raw_query: str) -> tuple[str, list[int]]:
    """NFC-normalized view + raw_offset_map[normalized_offset] = raw_offset."""
    raw_nfc = unicodedata.normalize("NFC", raw_query)
    raw_offset_map: list[int] = []
    raw_pos = 0
    for raw_offset, char in enumerate(raw_query):
        if unicodedata.category(char).startswith("M"):  # combining mark
            raw_offset_map.append(raw_pos)
        else:
            raw_offset_map.append(raw_offset)
            raw_pos = raw_offset + 1
    return raw_nfc.lower(), raw_offset_map


def extract_document_references(normalized: str, raw: str) -> list[RefExtraction]:
    """Run regex on normalized; convert offsets back to raw."""
    _, raw_offset_map = build_normalized_match_view(raw)
    # Note: caller already built map; for direct call, rebuild
    raw_offset_map = [...]
    refs = []
    for basis, pattern, group_map in REGEX_TABLE:
        for match in pattern.finditer(normalized):
            norm_start, norm_end = match.span()
            raw_start = raw_offset_map[norm_start]
            raw_end = raw_offset_map[norm_end - 1] + 1
            original_span = raw[raw_start:raw_end]
            ref = RefExtraction(
                ref_id=f"r{len(refs)+1}",
                original_span=original_span,
                span_offset=(raw_start, raw_end),
                reference=_normalize_reference(match, group_map),
                section_reference=_extract_section_reference(match, group_map),
                parse_basis=basis,
            )
            refs.append(ref)
    return refs
```

- [ ] **Step 3: Run tests**

Run: `cd backend && pytest tests/agents/test_extract_document_references.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/agents/semantic_preprocessor.py backend/tests/agents/test_extract_document_references.py
git commit -m "feat(phase1a): extract_document_references + NFC span mapping (O60, O61)

Per B.5: Python code-point (str-level) half-open offsets.
build_normalized_match_view handles NFD/NFC by mapping normalized_offset
→ raw_offset. Validator _check_raw_slice_equality ensures
original_span == raw[span_offset[0]:span_offset[1]] exactly."
```

---

### Task 9: Implement `expand_abbreviations` + `llm_disambiguate_ambiguous` (B.4)

**Files:**
- Modify: `backend/app/services/agents/semantic_preprocessor.py`

**Interfaces:**
- Consumes: candidates + DB
- Produces: `AbbreviationEntry[]` with `chosen` field populated
- Uses existing `disambiguate_multi_meaning_abbrs` logic (refactored)

- [ ] **Step 1: Write failing test**

```python
# backend/tests/agents/test_expand_abbreviations.py
def test_single_meaning_resolves_without_llm(test_db, abbr_single_meaning):
    candidates = [CandidateAbbr(short_form=abbr_single_meaning.short_form,
                                span_offset=(0, 4), original_span="BMNN")]
    results = await expand_abbreviations(candidates, ctx, test_db)
    assert results[0].status == "resolved"
    assert results[0].chosen is not None
    assert results[0].chosen in {c.full_form for c in results[0].candidates}

def test_multi_meaning_ambiguous_no_llm_call(test_db, abbr_multi_meaning):
    candidates = [CandidateAbbr(short_form="CP", span_offset=(0, 2), original_span="CP")]
    # Without LLM call (budget exhausted)
    with patch("app.services.llm.get_memory_agent") as mock_agent:
        results = await expand_abbreviations(candidates, ctx_with_no_budget, test_db)
        mock_agent.assert_not_called()
    assert results[0].status == "ambiguous"

def test_multi_meaning_llm_disambig_within_budget(test_db, abbr_multi_meaning):
    candidates = [CandidateAbbr(short_form="CP", span_offset=(0, 2), original_span="CP")]
    with patch("app.services.llm.get_memory_agent") as mock_agent:
        mock_agent.return_value.astream.return_value = [{"text": '{"results": [{"abbr": "CP", "chosen": "Chính phủ", "confidence": "high"}]}'}]
        results = await expand_abbreviations(candidates, ctx_with_budget, test_db)
    assert results[0].chosen == "Chính phủ"
```

- [ ] **Step 2: Implement**

```python
async def expand_abbreviations(
    short_forms: list[CandidateAbbr],
    ctx: RuntimeContext,
    session: AsyncSession,
) -> list[AbbreviationEntry]:
    """Batch lookup; multi-meaning → LLM disambig (budget-gated)."""
    results: list[AbbreviationEntry] = []
    
    # Batch DB lookup
    short_form_strs = [c.short_form for c in short_forms]
    db_rows = (await session.execute(
        select(Abbreviation).where(
            func.lower(Abbreviation.short_form).in_([s.lower() for s in short_form_strs]),
            Abbreviation.is_active == True,
        )
    )).scalars().all()
    
    abbr_map: dict[str, list[Abbreviation]] = defaultdict(list)
    for row in db_rows:
        abbr_map[row.short_form.lower()].append(row)
    
    for candidate in short_forms:
        key = candidate.short_form.lower()
        matches = abbr_map.get(key, [])
        if len(matches) == 0:
            results.append(AbbreviationEntry(
                span=candidate.original_span, span_offset=candidate.span_offset,
                short_form=candidate.short_form, candidates=[],
                status="not_in_db",
            ))
        elif len(matches) == 1:
            results.append(AbbreviationEntry(
                span=candidate.original_span, span_offset=candidate.span_offset,
                short_form=candidate.short_form,
                chosen=matches[0].full_form,
                candidates=[AbbreviationCandidate(full_form=m.full_form, description=m.description) for m in matches],
                status="resolved", source="db_single",
            ))
        else:
            results.append(AbbreviationEntry(
                span=candidate.original_span, span_offset=candidate.span_offset,
                short_form=candidate.short_form,
                candidates=[AbbreviationCandidate(...) for m in matches],
                status="ambiguous", source="db_multi",
            ))
    
    # LLM disambig if any ambiguous AND budget allows
    ambigs = [r for r in results if r.status == "ambiguous"]
    if ambigs and _budget_allows_disambig(ctx):
        disambiguated = await llm_disambiguate_ambiguous(ambigs, ctx.original_query, ctx, session)
        for disambig in disambiguated:
            for r in results:
                if r.span_offset == disambig.span_offset and r.short_form == disambig.short_form:
                    r.chosen = disambig.chosen
                    r.status = disambig.status
                    r.confidence = disambig.confidence
                    r.reasoning = disambig.reasoning
                    r.source = "llm_disambig"
    
    return results


async def llm_disambiguate_ambiguous(
    ambigs: list[AbbreviationEntry],
    query: str,
    ctx: RuntimeContext,
    session: AsyncSession,
) -> list[AbbreviationEntry]:
    """One memory-agent call; budget-gated; Pydantic validator rejects bad chosen."""
    try:
        sys_prompt, user_prompt = build_batch_disambiguation_prompt(
            {a.short_form: [{"full_form": c.full_form, "description": c.description} for c in a.candidates]
             for a in ambigs},
            query,
        )
        agent = get_memory_agent()
        
        async def _call():
            return await agent.astream(
                [LLMMessage(role="user", content=user_prompt)],
                system_prompt=sys_prompt,
                temperature=0.0,
                max_tokens=256,
                think=False,
            )
        chunks = await asyncio.wait_for(_call(), timeout=ctx.preprocessing.per_call_timeout_sec)
        # Parse + validate; Pydantic validator rejects chosen ∉ candidates
        ...
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"[preproc] disambig failed: {e}")
        return ambigs  # leave as-is
```

- [ ] **Step 3: Run tests**

Run: `cd backend && pytest tests/agents/test_expand_abbreviations.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/agents/semantic_preprocessor.py backend/tests/agents/test_expand_abbreviations.py
git commit -m "feat(phase1a): expand_abbreviations + llm_disambiguate_ambiguous

Per B.4: batch DB lookup; multi-meaning → LLM disambig (one call,
budget-gated). Pydantic validator auto-rejects chosen not in candidates."
```

---

### Task 10: Implement `preprocess_query` pipeline (B.2 DAG)

**Files:**
- Modify: `backend/app/services/agents/semantic_preprocessor.py`

**Interfaces:**
- Consumes: `query`, `RuntimeContext`
- Produces: `PreprocessingResult` with all fields populated
- DAG: parallel safe branches + sequential `regex_abbr_then_doc`

- [ ] **Step 1: Write failing test**

```python
# backend/tests/agents/test_preprocess_query.py
def test_greeting_fast_path_returns_empty(test_db):
    result = await preprocess_query("Xin chào", ctx)
    assert result.preprocessing_status == "ok"
    assert result.abbreviations == []
    assert result.document_refs == []

def test_compare_two_docs_returns_resolved_refs(test_db, doc_x_chap2, doc_y_chap3):
    query = "So sánh Chương II X và Chương III Y"
    result = await preprocess_query(query, ctx_with_access, test_db)
    assert len(result.document_refs) == 2
    for ref in result.document_refs:
        assert ref.resolution_status == "resolved"
        assert ref.document_handle is not None

def test_nested_abbrev_inside_ref_preserved(test_db, abbr_ND, doc_ND_13):
    query = "So sánh NĐ 13 với NĐ 24"  # "NĐ" inside ref
    result = await preprocess_query(query, ctx, test_db)
    # Abbreviations include "NĐ"; refs include "NĐ 13" + "NĐ 24"
    abbr_spans = {a.span_offset for a in result.abbreviations if a.short_form == "NĐ"}
    ref_spans = [r.span_offset for r in result.document_refs]
    # Abbrev span must be inside one of the ref spans (nested allowed)
    for abbr_span in abbr_spans:
        assert any(rs[0] <= abbr_span[0] and abbr_span[1] <= rs[1] for rs in ref_spans)
```

- [ ] **Step 2: Implement DAG pipeline**

```python
async def preprocess_query(query: str, ctx: RuntimeContext) -> PreprocessingResult:
    # Step 0: Fast-path gate
    fast_result = _should_fast_path(query, ctx)
    if fast_result is not None:
        return fast_result
    
    # Step 1: Sync extraction
    normalized, raw_offset_map = build_normalized_match_view(query)
    refs = extract_document_references(normalized, query)
    abbrs = extract_abbreviation_candidates(query)
    trace = [TraceEvent(step="input", started_at=time.monotonic(), attempt=0)]
    
    # Step 2: Classify by dependency
    abbr_first_refs = [r for r in refs if r.parse_basis == "regex_abbr_then_doc"]
    independent_refs = [r for r in refs if r.parse_basis != "regex_abbr_then_doc"]
    
    # Step 3: DAG execution (parallel safe + sequential abbr_then_doc)
    async with branch_session_factory() as session:
        async def _lookup_independent(rs):
            return await asyncio.gather(*[safe_lookup_metadata_only(r, ctx, session) for r in rs])
        async def _abbr_then_doc(rs):
            if not rs: return []
            async with branch_session_factory() as session_a:
                resolved_abbrs = await expand_abbreviations(abbrs, ctx, session_a)
            updated = _apply_abbr_resolution(rs, resolved_abbrs)
            async with branch_session_factory() as session_b:
                return await asyncio.gather(*[safe_lookup_metadata_only(r, ctx, session_b) for r in updated])
        
        independent_results, abbr_doc_results = await asyncio.gather(
            _lookup_independent(independent_refs),
            _abbr_then_doc(abbr_first_refs),
        )
    
    # ... (continue per B.2 spec)
```

- [ ] **Step 3: Implement `_should_fast_path`**

```python
def _should_fast_path(query: str, ctx: RuntimeContext) -> PreprocessingResult | None:
    """Return PreprocessingResult if fast-path applies; None if full pipeline."""
    q = query.strip()
    if len(q) < 5:
        return PreprocessingResult(original_query=query, preprocessing_status="ok", preprocessor_trace=[])
    if _GREETING_RE.match(q):
        return PreprocessingResult(original_query=query, preprocessing_status="ok", preprocessor_trace=[])
    # ... other fast-path checks per B.3 ...
    return None
```

- [ ] **Step 4: Run tests**

Run: `cd backend && pytest tests/agents/test_preprocess_query.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/agents/semantic_preprocessor.py backend/tests/agents/test_preprocess_query.py
git commit -m "feat(phase1a): preprocess_query DAG pipeline (B.2)

Per B.2: fast-path gate; sync extraction (NFC + raw-span map);
DAG execution with per-branch DB sessions. Parallel safe branches
+ sequential abbr_then_doc. Nested spans preserved for regex_abbr_then_doc."
```

---

### Task 11: Implement `semantic_preprocessor_node` + graph wiring (B.7)

**Files:**
- Modify: `backend/app/services/agents/semantic_preprocessor.py` (add node)
- Modify: `backend/app/services/agents/supervisor.py` (graph wiring + atomic switch)

**Interfaces:**
- Consumes: LangGraph state at ingress
- Produces: state["semantic_context"] + state["_preprocessor_marker"]
- Atomic switch: `NEXUSRAG_SEMANTIC_PREPROCESSOR=false` → old graph; true → new graph

- [ ] **Step 1: Implement node**

```python
async def semantic_preprocessor_node(state: SupervisorState) -> dict:
    """LangGraph node entry. Sets semantic_context + trusted marker."""
    if not settings.NEXUSRAG_SEMANTIC_PREPROCESSOR:
        return {}  # legacy path (shouldn't reach here in old graph)
    
    ctx = _build_runtime_context(state)
    query = _extract_user_message(state)
    result = await preprocess_query(query, ctx)
    
    # Migration compat: derive legacy fields (B.7)
    legacy = _derive_legacy_fields(result)
    
    return {
        "semantic_context": result,
        "query_complexity": legacy.complexity,
        "extracted_params": legacy.params,
        "_preprocessor_marker": "semantic_v1",
    }


def _derive_legacy_fields(result: PreprocessingResult) -> LegacyCompatFields:
    """Phase 0/1A transition: derive legacy query_complexity + extracted_params."""
    complexity = "simple"
    if len(result.document_refs) >= 2:
        complexity = "multi_doc"
    
    params = {
        "document_refs": [{"reference": r.reference, "section": r.section_reference} for r in result.document_refs],
        "sections": [r.section_reference for r in result.document_refs if r.section_reference],
    }
    return LegacyCompatFields(complexity=complexity, params=params)
```

- [ ] **Step 2: Modify supervisor graph wiring**

In `backend/app/services/agents/supervisor.py` `create_supervisor_graph()`:

```python
def create_supervisor_graph() -> StateGraph:
    if settings.NEXUSRAG_SEMANTIC_PREPROCESSOR:
        return _build_new_graph()
    return _build_legacy_graph()  # current


def _build_new_graph() -> StateGraph:
    workflow = StateGraph(SupervisorState)
    workflow.add_node("semantic_preprocessor", semantic_preprocessor_node)
    workflow.add_node("supervisor", supervisor_node)
    # ... existing nodes
    workflow.add_edge(START, "semantic_preprocessor")
    workflow.add_edge("semantic_preprocessor", "supervisor")
    # ... conditional edges from supervisor
    return workflow.compile()
```

- [ ] **Step 3: Suppress duplicate abbrev expansion in supervisor_node**

In `supervisor_node`, check `_preprocessor_marker`:

```python
async def supervisor_node(state):
    marker = state.get("_preprocessor_marker")
    if marker == "semantic_v1":
        # Skip legacy abbr expansion + LLM disambig (already done)
        # Proceed directly to LLM classifier call with semantic_context in prompt
        pass
    else:
        # Legacy path (pre-Phase 1A): run abbr lookup + LLM disambig
        # ... existing logic
        pass
```

- [ ] **Step 4: Test atomic flag switch**

```python
# backend/tests/agents/test_graph_atomic_flag.py
def test_flag_false_uses_legacy_graph():
    with patch("app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR", False):
        graph = create_supervisor_graph()
        # Verify no semantic_preprocessor node
        assert "semantic_preprocessor" not in graph.nodes

def test_flag_true_uses_new_graph():
    with patch("app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR", True):
        graph = create_supervisor_graph()
        assert "semantic_preprocessor" in graph.nodes
```

- [ ] **Step 5: Test `chat_agent_lg.py:179-210` marker at ingress (O17)**

In `chat_agent_lg.py`:

```python
# BEFORE supervisor graph entry
state["_preprocessor_marker"] = "abbrev_done"  # suppress chat_agent_lg's own abbr expansion
```

Verify existing chat_agent_lg.py reads marker and skips.

- [ ] **Step 6: Run tests**

Run: `cd backend && pytest tests/agents/test_graph_atomic_flag.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/agents/semantic_preprocessor.py backend/app/services/agents/supervisor.py backend/app/api/chat_agent_lg.py backend/tests/agents/test_graph_atomic_flag.py
git commit -m "feat(phase1a): semantic_preprocessor_node + atomic graph wiring (B.7, O17)

Per B.7: atomic switch on NEXUSRAG_SEMANTIC_PREPROCESSOR. New graph adds
semantic_preprocessor node; supervisor_node checks _preprocessor_marker
to skip duplicate abbr expansion. chat_agent_lg.py sets marker at
ingress to suppress its own abbr expansion (O17)."
```

---

### Task 12: Phase 1A gate review (O7)

**Files:**
- Verify: all gates pass

**Interfaces:**
- Consumes: all prior Phase 1A tasks
- Produces: Phase 1A gate report

- [ ] **Step 1: Run all Phase 1A tests**

Run: `cd backend && pytest tests/agents/ tests/models/ tests/migrations/ tests/core/ -v`
Expected: All pass.

- [ ] **Step 2: Verify flag-off path still works**

Run: `cd backend && NEXUSRAG_SEMANTIC_PREPROCESSOR=false pytest tests/agents/ -v`
Expected: All pass (legacy path regression-safe).

- [ ] **Step 3: Run end-to-end smoke test**

Run: `make eval-prompts` (per harness.md)
Expected: Eval prompts pass.

- [ ] **Step 4: Write Phase 1A gate report**

Create `backend/tests/reports/phase1a_gate_report.md`:

```markdown
# Phase 1A Gate Report

**Date**: [today]
**Spec**: docs/superpowers/specs/2026-09-08-deepagent-design.md Section B

## Gates
| Gate | Status | Evidence |
|------|--------|----------|
| All contracts importable | PASS | test_contracts_imports.py |
| DocumentAlias migration | PASS | test_document_alias.py |
| chat_messages.semantic_context column | PASS | test_semantic_context_backward_compat.py |
| agent_traces schema | PASS | test_agent_trace_backward_compat.py |
| SupervisorState extensions | PASS | test_supervisor_state_extensions.py |
| Flag atomic switch | PASS | test_graph_atomic_flag.py |
| safe_lookup_metadata_only | PASS | test_safe_lookup_metadata_only.py |
| NFC span mapping | PASS | test_extract_document_references.py |
| Abbreviation expansion | PASS | test_expand_abbreviations.py |
| preprocess_query DAG | PASS | test_preprocess_query.py |
| Pre-existing tests pass | PASS | test_attachment_delete_acl.py + all Phase 0 tests |

## Decision
[ ] Phase 1A PASS — proceed to atomic enable
[ ] Phase 1A FAIL — list blockers
```

- [ ] **Step 5: Commit**

```bash
git add backend/tests/reports/phase1a_gate_report.md
git commit -m "docs(phase1a): gate review — all 11 gates pass

Per O7: contracts + migration + preprocessor + graph wiring + tests.
Phase 1A ready for atomic enable via NEXUSRAG_SEMANTIC_PREPROCESSOR=true."
```

---

## Summary

| Task | Subject | Files | Open items closed |
|------|---------|-------|-------------------|
| 1 | DocumentAlias | new model + migration | O6, O9 |
| 2 | semantic_context column | chat_message + lifespan + persistence | O8 |
| 3 | routing_trace columns | agent_trace + lifespan | O8 |
| 4 | SupervisorState extension | models.py | O7 |
| 5 | Flag + validator | config.py + .env.example + CLAUDE.md | O7, O56 |
| 6 | Contracts module | 3 new files | O11, O22, O23 |
| 7 | safe_lookup primitive | semantic_preprocessor.py | O3 |
| 8 | NFC span mapping | semantic_preprocessor.py | O60, O61 |
| 9 | Abbreviation expansion | semantic_preprocessor.py | O62 (partial) |
| 10 | DAG pipeline | semantic_preprocessor.py | O63 (cross-domain threshold) |
| 11 | Graph wiring | supervisor.py + chat_agent_lg.py | O17, O20 |
| 12 | Gate review | gate report | O7 (verify) |

**Total: 12 atomic commits, Phase 1A ready for atomic enable.**

Phase 1A gate satisfied → proceed to Phase 1B plan (`2026-09-08-deepagent-phase1b-router.md`).

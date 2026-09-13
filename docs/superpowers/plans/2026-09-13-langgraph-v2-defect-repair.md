# LangGraph v2 Defect Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the four confirmed langgraph-v2 defects plus the v1 A/B harness defect from the 2026-09-13 live test, and reindex the legacy corpus so v2 factual routes can read it.

**Architecture:** Keep the six regex generators and add a deterministic arbitration stage before `PreprocessingResult` validation; add a DB longest-match title resolver; replace the finalizer's missing-verdict raise with a typed `clarify`/`insufficient` response driven by the semantic signal; normalize `ChatSourceChunk` at the accumulator boundary; and execute the existing revision-aware reindex against the test stack.

**Tech Stack:** Python 3.11, Pydantic v2, SQLAlchemy async, LangGraph, pytest/pytest-asyncio, FastAPI, Postgres, MinIO.

**Spec:** `docs/superpowers/specs/2026-09-13-langgraph-v2-defect-repair-design.md`

## Global Constraints

- Deployment is **v2-only** for factual traffic; no v1 fallback. A document without a published revision MUST keep terminating as typed unavailable/insufficient.
- **Never restart the vLLM engines** (`hrag-vllm-ocr`, `hrag-vllm-memory`).
- Tests run **inside the `hrag-backend` container** (bind-mounted to this worktree, WORKDIR `/app/backend`): `docker exec hrag-backend pytest ...`.
- Do not hand-set `Document.current_revision_id`; only the revision publish CAS may set it.
- No `ResponseStatus` contract change; `clarify` and `insufficient` already exist.
- Do not expand `_RE_SECTION`; do not implement People/Section extraction.
- Before every commit run `node .gitnexus/run.cjs detect-changes -r /home/AIRAG/.worktrees/langgraph-v2`.
- Run `node .gitnexus/run.cjs impact -r /home/AIRAG/.worktrees/langgraph-v2 -t <symbol>` (upstream) before editing any symbol; stop and warn on HIGH/CRITICAL risk.

---

### Task 1: Extraction arbitration (#1)

**Files:**
- Modify: `backend/app/services/agents/semantic_preprocessor.py`
- Test: `backend/tests/agents/test_semantic_preprocessor_extraction.py` (create)

**Interfaces:**
- Consumes: existing `RefExtraction`, `extract_document_references(normalized, raw)`.
- Produces: `_REF_BASIS_RANK: dict[str, int]`; `_arbitrate_ref_candidates(candidates: list[RefExtraction]) -> list[RefExtraction]`. `extract_document_references` returns arbitrated (pairwise non-overlapping, `r1..rN`) refs.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/agents/test_semantic_preprocessor_extraction.py`:

```python
"""Regression coverage for document-reference extraction arbitration (defect #1)."""
from __future__ import annotations

import pytest

from app.services.agents.semantic_preprocessor import (
    DocumentRefEntry,
    PreprocessingResult,
    build_normalized_match_view,
    extract_document_references,
)


def _extract(query: str):
    normalized, _ = build_normalized_match_view(query)
    return extract_document_references(normalized, query)


def test_official_document_number_emits_one_non_overlapping_ref() -> None:
    refs = _extract("53/2022/NĐ-CP quy định những nội dung gì?")
    assert [(r.span_offset, r.parse_basis) for r in refs] == [
        ((0, 13), "regex_doc_num")
    ]


def test_bare_abbreviation_is_kept_as_a_single_ref() -> None:
    refs = _extract("NĐ là gì?")
    assert [(r.span_offset, r.reference) for r in refs] == [((0, 2), "nđ")]


@pytest.mark.parametrize(
    "query",
    [
        "53/2022/NĐ-CP quy định những nội dung gì?",
        "NĐ là gì?",
        "So sánh 85/2016/NĐ-CP và Luật An ninh mạng",
        "So sánh Luật An ninh mạng và Luật Bảo vệ dữ liệu cá nhân",
    ],
)
def test_arbitrated_refs_are_pairwise_non_overlapping(query: str) -> None:
    spans = sorted(r.span_offset for r in _extract(query))
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert end <= next_start, f"{query!r}: {spans}"


def test_number_query_constructs_a_valid_preprocessing_result() -> None:
    query = "53/2022/NĐ-CP quy định những nội dung gì?"
    normalized, _ = build_normalized_match_view(query)
    entries = [
        DocumentRefEntry(
            ref_id=r.ref_id,
            original_span=r.original_span,
            span_offset=r.span_offset,
            reference=r.reference,
            section_reference=r.section_reference,
            resolution_status="deferred",
        )
        for r in extract_document_references(normalized, query)
    ]
    result = PreprocessingResult(
        original_query=query,
        normalized_query=normalized,
        document_refs=entries,
        preprocessing_status="ok",
        preprocessor_trace=[],
    )
    assert len(result.document_refs) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec hrag-backend pytest tests/agents/test_semantic_preprocessor_extraction.py -v`
Expected: FAIL — `test_official_document_number_emits_one_non_overlapping_ref` sees 4 refs.

- [ ] **Step 3: Add the arbitration stage**

In `backend/app/services/agents/semantic_preprocessor.py`, immediately after `_RE_ABBREVIATION` (before `_normalize_reference_for_doc_num`), insert:

```python
#: Specificity precedence for overlapping document-reference candidates.
#: The six extraction patterns are independent generators and may emit
#: overlapping spans for one real reference (e.g. a full document number and
#: its short-number sub-span). A candidate that overlaps an already-kept
#: higher-ranked candidate is dropped, so emitted ``document_refs`` are
#: pairwise non-overlapping by construction and the frozen
#: ``PreprocessingResult`` validator can never reject a real query.
_REF_BASIS_RANK: dict[str, int] = {
    "regex_doc_num": 60,
    "regex_short_official": 50,
    "regex_section": 40,
    "regex_named_doc": 30,
    "regex_bare_number": 20,
    "regex_abbr_then_doc": 10,
}


def _arbitrate_ref_candidates(
    candidates: list[RefExtraction],
) -> list[RefExtraction]:
    """Keep the most specific non-overlapping document references.

    Rank by specificity, then span length, then start; greedily keep the first
    candidate for each region and drop any candidate overlapping a kept one.
    Surviving ``ref_id``s are renumbered ``r1..rN`` in span order.
    """
    ordered = sorted(
        candidates,
        key=lambda c: (
            -_REF_BASIS_RANK.get(c.parse_basis, 0),
            -(c.span_offset[1] - c.span_offset[0]),
            c.span_offset[0],
        ),
    )
    kept: list[RefExtraction] = []
    for candidate in ordered:
        start, end = candidate.span_offset
        if any(
            start < kept_ref.span_offset[1] and kept_ref.span_offset[0] < end
            for kept_ref in kept
        ):
            continue
        kept.append(candidate)
    kept.sort(key=lambda c: c.span_offset[0])
    return [
        RefExtraction(
            ref_id=f"r{i}",
            original_span=c.original_span,
            span_offset=c.span_offset,
            reference=c.reference,
            section_reference=c.section_reference,
            parse_basis=c.parse_basis,
        )
        for i, c in enumerate(kept, start=1)
    ]
```

Then change the last line of `extract_document_references` from `return results` to:

```python
    return _arbitrate_ref_candidates(results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec hrag-backend pytest tests/agents/test_semantic_preprocessor_extraction.py -v`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
node .gitnexus/run.cjs detect-changes -r /home/AIRAG/.worktrees/langgraph-v2
git add backend/app/services/agents/semantic_preprocessor.py \
        backend/tests/agents/test_semantic_preprocessor_extraction.py
git commit -m "fix(v2): arbitrate overlapping document-reference spans"
```

---

### Task 2: Multi-word title grammar + DB longest-match (#2)

**Files:**
- Modify: `backend/app/services/agents/semantic_preprocessor.py`
- Test: `backend/tests/agents/test_semantic_preprocessor_extraction.py` (append)

**Interfaces:**
- Consumes: Task 1 `extract_document_references`; existing `_lookup_by_alias`, `RefExtraction`, `DocumentRefEntry`, `DocumentCandidate`, `DocumentMetadata`.
- Produces: greedy `_RE_NAMED_DOC`; `_lookup_by_title(ref, ctx, session) -> DocumentRefEntry`; `safe_lookup_metadata_only` routes `regex_named_doc` to it.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/agents/test_semantic_preprocessor_extraction.py`:

```python
def test_multi_word_titles_are_captured_whole() -> None:
    refs = _extract(
        "So sánh Luật An ninh mạng và Luật Bảo vệ dữ liệu cá nhân"
    )
    assert [r.reference for r in refs] == [
        "luật an ninh mạng",
        "luật bảo vệ dữ liệu cá nhân",
    ]


def test_title_stops_before_a_question_phrase() -> None:
    refs = _extract("Luật An ninh mạng quy định gì?")
    assert [r.reference for r in refs] == ["luật an ninh mạng"]


@pytest.mark.asyncio
async def test_title_lookup_selects_the_longest_matching_title() -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from uuid import UUID

    from app.services.agents.semantic_preprocessor import _lookup_by_title

    workspace_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    document_id = UUID("11111111-1111-1111-1111-111111111111")
    context = SimpleNamespace(allowed_workspace_ids=[workspace_id])

    short_doc = SimpleNamespace(
        id=document_id, workspace_id=workspace_id,
        document_title="Luật An ninh", document_number=None,
        updated_at=None, content_hash=None,
    )
    long_doc = SimpleNamespace(
        id=document_id, workspace_id=workspace_id,
        document_title="Luật An ninh mạng", document_number=None,
        updated_at=None, content_hash=None,
    )
    alias_result = MagicMock()
    alias_result.all.return_value = []  # no exact alias
    title_result = MagicMock()
    title_result.scalars.return_value.all.return_value = [short_doc, long_doc]
    session = MagicMock(execute=AsyncMock(side_effect=[alias_result, title_result]))

    reference = RefExtraction(
        ref_id="r1", original_span="Luật An", span_offset=(0, 7),
        reference="luật an", section_reference=None,
        parse_basis="regex_named_doc",
    )
    resolved = await _lookup_by_title(reference, context, session)

    assert resolved.resolution_status == "resolved"
    assert resolved.document_handle == document_id
    assert resolved.match_basis == "fuzzy_title"
    assert resolved.reference == "Luật An ninh mạng"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec hrag-backend pytest tests/agents/test_semantic_preprocessor_extraction.py -v`
Expected: FAIL — titles truncated to one token; `_lookup_by_title` missing.

- [ ] **Step 3: Implement greedy grammar and the title resolver**

In `backend/app/services/agents/semantic_preprocessor.py`, replace the existing `_RE_NAMED_DOC` definition with:

```python
#: Stop words/phrases that end a document title in a free-form legal query.
#: The normalized query is lowercased, so capitalization cannot mark the
#: title boundary; punctuation and these function/verb/question phrases do.
_NAMED_DOC_STOP = (
    r"(?:và|với|của|cho|về|trong|tại|theo|gồm|hay|hoặc|mà|để|khi|nếu|do|bởi|"
    r"từ|đến|tới|trên|dưới|giữa|ngoài|sau|trước|cùng|các|những|một|này|đó|kia|"
    r"ấy|mỗi|mọi|tất|cả|quy\s*định|so\s*sánh|tóm\s*tắt|liệt\s*kê|trình\s*bày|"
    r"áp\s*dụng|sửa\s*đổi|bổ\s*sung|ban\s*hành|gì|nào|như\s*thế|sao|có|không|"
    r"được|là|bao\s*nhiêu|ai|đâu)"
)

_RE_NAMED_DOC = re.compile(
    r"\b(?P<doc>"
    r"(?:luật|nghị\s*định|nghị\s*quyết|quyết\s*định|"
    r"thông\s*tư(?:\s*liên\s*tịch)?|pháp\s*lệnh|chỉ\s*thị)"
    r"(?:\s+(?!" + _NAMED_DOC_STOP + r"\b)[^\s,.;:?!\"()\[\]]+){0,8}"
    r")\s*(?P<num>\d+/\d{4}[/-][A-Za-z]+)?",
    re.IGNORECASE | re.UNICODE,
)
```

Immediately after `_lookup_by_alias`, add:

```python
async def _lookup_by_title(
    ref: RefExtraction, ctx: "RuntimeContext", session: "AsyncSession"
) -> DocumentRefEntry:
    """Longest-match title lookup for a multi-word named document.

    An exact alias match wins first (reusing ``_lookup_by_alias``). Otherwise
    candidate documents whose title starts with the reference are compared and
    the longest title wins; a shared longest title is ambiguous. A miss
    degrades to the alias path's verdict (usually ``not_found``) and never
    raises, so the arbitrated reference is preserved.
    """
    from sqlalchemy import func, select
    from app.models.document import Document

    alias_entry = await _lookup_by_alias(ref, ctx, session)
    if alias_entry.resolution_status != "not_found":
        return alias_entry

    allowed = list(ctx.allowed_workspace_ids)
    needle = ref.reference.lower().strip()
    stmt = (
        select(Document)
        .where(
            Document.workspace_id.in_(allowed),
            func.lower(Document.document_title).like(f"{needle}%"),
        )
    )
    result = await session.execute(stmt)
    docs = list(result.scalars().all())
    matching = [
        d for d in docs
        if (d.document_title or "").lower().strip().startswith(needle)
    ]
    if not matching:
        return alias_entry

    best_title = max((d.document_title or "").strip() for d in matching)
    best_docs = [
        d for d in matching if (d.document_title or "").strip() == best_title
    ]
    if len(best_docs) > 1:
        return DocumentRefEntry(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            resolution_status="ambiguous",
            candidates=[
                DocumentCandidate(
                    document_id=d.id,
                    match_basis="fuzzy_title",
                    confidence=0.6,
                    title=d.document_title,
                    doc_number=d.document_number,
                )
                for d in best_docs
            ],
        )

    doc = best_docs[0]
    version = (
        f"{doc.updated_at.isoformat() if doc.updated_at else ''}"
        f"|{doc.content_hash[:8] if doc.content_hash else ''}"
    )
    return DocumentRefEntry(
        ref_id=ref.ref_id,
        original_span=ref.original_span,
        span_offset=ref.span_offset,
        reference=doc.document_title or ref.reference,
        section_reference=ref.section_reference,
        document_handle=doc.id,
        resolution_status="resolved",
        match_basis="fuzzy_title",
        version=version,
        metadata=DocumentMetadata(workspace_id=doc.workspace_id),
        authorized_at_lookup=True,
    )
```

In `safe_lookup_metadata_only`, change the `regex_named_doc` branch from `return await _lookup_by_alias(ref, ctx, session)` to `return await _lookup_by_title(ref, ctx, session)` (leave the `regex_abbr_then_doc` branch on `_lookup_by_alias`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec hrag-backend pytest tests/agents/test_semantic_preprocessor_extraction.py tests/agents/test_semantic_preprocessor_db_resolution.py -v`
Expected: PASS (including the pre-existing alias regression).

- [ ] **Step 5: Commit**

```bash
node .gitnexus/run.cjs detect-changes -r /home/AIRAG/.worktrees/langgraph-v2
git add backend/app/services/agents/semantic_preprocessor.py \
        backend/tests/agents/test_semantic_preprocessor_extraction.py
git commit -m "fix(v2): capture multi-word law titles and longest-match resolve them"
```

---

### Task 3: Typed terminal outcome for a missing verdict (#3)

**Files:**
- Modify: `backend/app/services/agents/v2/nodes/finalizer.py`
- Test: `backend/tests/agents/v2/test_missing_verdict_terminal.py` (create)

**Interfaces:**
- Consumes: `_finalize_factual(state, context)`, `_emit`, `_NEEDS_INPUT_CONTENT`, `_INSUFFICIENT_CONTENT`, `SemanticContext`.
- Produces: `_semantic_needs_input(state) -> bool`; `_typed_missing_verdict(state) -> dict`. `_finalize_factual` returns a typed response instead of raising when `evidence_evaluation is None`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/agents/v2/test_missing_verdict_terminal.py`:

```python
"""Defect #3: a missing verdict must become a typed clarify/insufficient reply."""
from __future__ import annotations

from uuid import UUID

import pytest

from app.services.agents.v2.contracts.semantic import (
    BlockingAmbiguity,
    DocumentReference,
    SemanticContext,
)
from app.services.agents.v2.contracts.state import ExecutionState
from app.services.agents.v2.nodes.finalizer import _finalize_factual

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _semantic(
    *,
    document_refs: tuple[DocumentReference, ...] = (),
    blocking_ambiguities: tuple[BlockingAmbiguity, ...] = (),
) -> SemanticContext:
    return SemanticContext(
        contextualized_query="câu hỏi",
        normalized_query="câu hỏi",
        abbreviations=(),
        coreferences=(),
        document_refs=document_refs,
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=blocking_ambiguities,
    )


def _ref(status: str) -> DocumentReference:
    return DocumentReference(
        ref_id="r1",
        original_span="Luật ABC",
        normalized_reference="luật abc",
        requested_role="target",
        revision_requirement=None,
        resolution_status=status,  # type: ignore[arg-type]
        resolved_document_id=None,
        candidate_document_ids=(),
    )


def _state(semantic: SemanticContext) -> dict:
    return {
        "execution": ExecutionState(
            plan=None, task_results=(), evidence_evaluation=None
        ),
        "semantic": semantic,
    }


@pytest.mark.asyncio
async def test_unresolved_reference_clarifies() -> None:
    update = await _finalize_factual(_state(_semantic(document_refs=(_ref("unresolved"),))), None)
    assert update["final_response"].status == "clarify"
    assert "Luật ABC" in update["final_response"].content


@pytest.mark.asyncio
async def test_blocking_ambiguity_clarifies() -> None:
    ambiguity = BlockingAmbiguity(
        ambiguity_id="amb1",
        description="Bạn muốn so sánh hai văn bản nào?",
    )
    update = await _finalize_factual(_state(_semantic(blocking_ambiguities=(ambiguity,))), None)
    assert update["final_response"].status == "clarify"


@pytest.mark.asyncio
async def test_no_semantic_signal_is_typed_insufficient() -> None:
    update = await _finalize_factual(_state(_semantic()), None)
    assert update["final_response"].status == "insufficient"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec hrag-backend pytest tests/agents/v2/test_missing_verdict_terminal.py -v`
Expected: FAIL — `_finalize_factual` raises `FinalizerError` (and `_typed_missing_verdict` does not exist).

- [ ] **Step 3: Implement the typed missing-verdict branch**

In `backend/app/services/agents/v2/nodes/finalizer.py`, add the helper immediately before `_finalize_factual`:

```python
#: Statuses that mean the turn cannot proceed without the user naming a
#: document (criterion A: the semantic signal, not the missing verdict, decides
#: clarify vs insufficient).
_NEEDS_INPUT_REF_STATUSES = frozenset({"unresolved", "ambiguous"})


def _semantic_needs_input(state: SupervisorV2State) -> bool:
    """True when an unresolved/ambiguous reference or a blocking ambiguity remains."""
    semantic = state["semantic"]
    if semantic.blocking_ambiguities:
        return True
    return any(
        ref.resolution_status in _NEEDS_INPUT_REF_STATUSES
        for ref in semantic.document_refs
    )


def _typed_missing_verdict(state: SupervisorV2State) -> dict:
    """Turn a factual/complex turn with no checkpointed verdict into a typed reply.

    Never raises, never succeeds: a user-fixable semantic gap clarifies; anything
    else is typed ``insufficient``. Internal identifiers are never surfaced.
    """
    semantic = state["semantic"]
    if _semantic_needs_input(state):
        details = [ambiguity.description for ambiguity in semantic.blocking_ambiguities]
        spans = [
            ref.original_span
            for ref in semantic.document_refs
            if ref.resolution_status in _NEEDS_INPUT_REF_STATUSES
        ]
        if spans:
            details.append("chưa xác định: " + ", ".join(spans))
        content = _NEEDS_INPUT_CONTENT + (" " + " ".join(details) if details else "")
        return _emit(
            FinalResponse(
                contract_version=CONTRACT_VERSION,
                status="clarify",
                content=content,
                citations=(),
            )
        )
    return _emit(
        FinalResponse(
            contract_version=CONTRACT_VERSION,
            status="insufficient",
            content=_INSUFFICIENT_CONTENT,
            citations=(),
        )
    )
```

Then, in `_finalize_factual`, replace:

```python
    evaluation = state["execution"].evidence_evaluation
    if evaluation is None:
        raise FinalizerError(
            "factual finalization requires a checkpointed EvidenceEvaluation; "
            "refusing to emit a response without a verdict"
        )
```

with:

```python
    evaluation = state["execution"].evidence_evaluation
    if evaluation is None:
        return _typed_missing_verdict(state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec hrag-backend pytest tests/agents/v2/test_missing_verdict_terminal.py tests/agents/v2/fast_paths/test_domain_paths.py -v`
Expected: PASS — new cases pass; the existing finalizer suite is unchanged.

- [ ] **Step 5: Commit**

```bash
node .gitnexus/run.cjs detect-changes -r /home/AIRAG/.worktrees/langgraph-v2
git add backend/app/services/agents/v2/nodes/finalizer.py \
        backend/tests/agents/v2/test_missing_verdict_terminal.py
git commit -m "fix(v2): type the missing-verdict terminal as clarify/insufficient"
```

---

### Task 4: v1 A/B harness `ChatSourceChunk` fix

**Files:**
- Modify: `backend/app/services/agent/sources_accumulator.py`
- Modify: `backend/app/services/agent/streaming.py`
- Test: `backend/tests/agents/test_source_snapshot_dedup.py` (append)

**Interfaces:**
- Consumes: `Source`, `SourcesSnapshotAccumulator`, `ChatSourceChunk`.
- Produces: `Source.from_chat_source_chunk(chunk: Any) -> Source`; `streaming.py` uses it for the object branch.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/agents/test_source_snapshot_dedup.py`:

```python
def test_accumulator_accepts_a_chat_source_chunk_object():
    """The v1 SSE path emits ChatSourceChunk objects; the accumulator must not read .chunk."""
    from app.schemas.rag import ChatSourceChunk

    chunk = ChatSourceChunk(
        index="id12",
        chunk_id="chunk_1",
        content="nội dung điều 17",
        document_id="11111111-1111-1111-1111-111111111111",
        source_file="luat.pdf",
        document_number="12/2020/NĐ-CP",
    )
    acc = SourcesSnapshotAccumulator()
    acc.add([Source.from_chat_source_chunk(chunk)])
    sources = acc.deduplicated()
    assert len(sources) == 1
    assert sources[0].chunk == "nội dung điều 17"
    assert sources[0].document_id == "11111111-1111-1111-1111-111111111111"
    assert sources[0].source_id == "id12"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec hrag-backend pytest tests/agents/test_source_snapshot_dedup.py -v`
Expected: FAIL — `AttributeError: 'Source' has no attribute 'from_chat_source_chunk'`.

- [ ] **Step 3: Implement the adapter and use it**

In `backend/app/services/agent/sources_accumulator.py`, change the typing import line `from typing import Optional` to `from typing import Any, Optional`, and add the classmethod to `Source`:

```python
    @classmethod
    def from_chat_source_chunk(cls, chunk: Any) -> "Source":
        """Build a dedup identity from a ``ChatSourceChunk``-shaped object.

        ``ChatSourceChunk`` exposes ``document_id``/``chunk_id``/``content``/
        ``source_file``/``document_number``/``index`` but NOT ``chunk``/
        ``content_hash``/``source_id``; reading it directly was the v1 A/B
        ``AttributeError``. ``content`` owns the dedup text, ``chunk_id`` is the
        stable identity, and ``index`` discriminates provenance.
        """
        import hashlib

        content = getattr(chunk, "content", "") or ""
        content_hash = getattr(chunk, "chunk_id", "") or hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()[:16]
        document_id = str(getattr(chunk, "document_id", "") or "")
        doc = (
            getattr(chunk, "source_file", None)
            or getattr(chunk, "document_number", None)
            or document_id
        )
        return cls(
            doc=doc,
            chunk=content,
            content_hash=content_hash,
            source_id=getattr(chunk, "index", None),
            document_id=document_id,
        )
```

In `backend/app/services/agent/streaming.py`, replace:

```python
                    elif hasattr(s, "document_id"):
                        sources_acc.add([s])
                        original_sources.append(s)
```

with:

```python
                    elif hasattr(s, "document_id"):
                        sources_acc.add([AccSource.from_chat_source_chunk(s)])
                        original_sources.append(s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec hrag-backend pytest tests/agents/test_source_snapshot_dedup.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
node .gitnexus/run.cjs detect-changes -r /home/AIRAG/.worktrees/langgraph-v2
git add backend/app/services/agent/sources_accumulator.py \
        backend/app/services/agent/streaming.py \
        backend/tests/agents/test_source_snapshot_dedup.py
git commit -m "fix(harness): adapt ChatSourceChunk for the sources accumulator"
```

---

### Task 5: Corpus reindex runbook + end-to-end verification (#4)

**Files:**
- Create: `docs/runbooks/langgraph-v2-corpus-reindex.md`

**Interfaces:**
- Consumes: `POST /api/v1/rag/reindex-workspace/{workspace_id}`, `GET /api/v1/workspaces`, the `hrag-*` test stack, `document_views.load_current_revision_identity`.
- Produces: a written runbook and recorded pre/post verification for the `Luật` and `Nghị định` workspaces.

- [ ] **Step 1: Reload the backend so it serves the repaired code**

Run: `make restart-backend` (code is bind-mounted; a restart re-imports it).
Expected: `docker ps` shows `hrag-backend` healthy. (Do NOT touch vLLM.)

- [ ] **Step 2: Write the runbook**

Create `docs/runbooks/langgraph-v2-corpus-reindex.md` with exactly this content:

```markdown
# Runbook: v2 revision-aware corpus reindex

Purpose: publish v2 revision artifacts for legacy documents so
`document_views.load_current_revision_identity()` returns an identity and v2
factual routes can read the corpus. A document without a published revision
MUST keep failing closed; this runbook never weakens that guard.

## Constraints

- NEVER set `Document.current_revision_id` by hand. Only the revision publish
  CAS sets it.
- NEVER restart the vLLM engines (`hrag-vllm-ocr`, `hrag-vllm-memory`).
- Reindex is copy-on-write: the published revision stays current and readable
  until the replacement publishes. Reclamation is GC's job.

## 1. Pre-check (read-only)

```bash
# v2 schema applied?
docker exec hrag-postgres psql -U postgres -d hrag -c "select version from v2_schema_version;"

# legacy documents in the target workspaces (namespace per workspace id)
docker exec hrag-postgres psql -U postgres -d hrag -c \
  "select id, workspace_id, document_number, document_title, status, current_revision_id
     from documents
    where source_deleted_at is null and current_revision_id is null
    order by workspace_id;"
```

List workspace UUIDs through the API (`GET /api/v1/workspaces`) with a
superadmin JWT, or resolve the workspace by name from the pre-check output.

## 2. Execute

```bash
API=http://localhost:8080/api/v1
curl -s -X POST "$API/rag/reindex-workspace/$WS" -H "Authorization: Bearer $TOKEN" | jq
```

Run the smallest workspace first. The endpoint queues each document
(`allocate_reindex_revision` → new monotonic generation) and returns
`document_count`; the parse/embed/kg/caption workers build and publish.

## 3. Verify

```bash
# pointer + published revision + build artifacts
docker exec hrag-postgres psql -U postgres -d hrag -c \
  "select d.id, d.current_revision_id, r.status, b.build_profile,
          b.markdown_artifact_key, b.structure_artifact_key,
          b.embedding_namespace, b.embedding_model_hash
     from documents d
     join document_revisions r on r.revision_id = d.current_revision_id
     left join document_revision_builds b on b.revision_id = r.revision_id
    where d.workspace_id = '$WS';"
```

- `current_revision_id` is non-null and `status = 'published'` for every doc.
- `document_revision_builds` has the profile's required artifacts + embedding manifest.

Boundary check (in-container, read-only):

```bash
docker exec hrag-backend python -c "
import asyncio, uuid
from app.core.database import async_session_maker
from app.services.agents.v2.persistence import document_views
async def main():
    async with async_session_maker() as db:
        ident = await document_views.load_current_revision_identity(db, uuid.UUID('$DOC_ID'))
        print('identity:', ident)
asyncio.run(main())
"
```

Then re-run the authenticated factual SSE probe in the reindexed workspace; it
must return an answer or a typed `insufficient`, never a generic error.

## 4. Abort / rollback

If verification fails, abandon the unpublished revisions so the previous
pointer stays current; never edit the pointer:

```bash
docker exec hrag-backend python -c "
import asyncio, uuid
from app.core.database import async_session_maker
from app.services.agents.v2.persistence.document_revisions import DocumentRevisionsRepository
async def main():
    async with async_session_maker() as db:
        repo = DocumentRevisionsRepository(db)
        for rev in await repo.list_non_published_for_update(uuid.UUID('$DOC_ID')):
            await repo.abandon_revision(rev, reason='reindex verification failed')
        await db.commit()
asyncio.run(main())
"
```

Confirm `current_revision_id` is unchanged and the document still serves its old
revision. Investigate before retrying.
```

- [ ] **Step 3: Execute the runbook against the test stack**

Perform steps 1–3 of the runbook for the smaller workspace first (`Luật`), then
`Nghị định`. Capture the pre-check output, the reindex response, and the verify
output.

Expected:
- Pre-check lists the legacy documents with `current_revision_id` null.
- Reindex response reports `document_count > 0`.
- Verify shows `status = 'published'` and a non-null `current_revision_id`.
- The boundary check prints a `RevisionArtifactIdentity` (not `None`).

- [ ] **Step 4: Confirm v2 document search admits the corpus**

In-container, read-only boundary probe for one reindexed document (adapts the
report's trace):

```bash
docker exec hrag-backend python -c "
import asyncio, uuid
from app.core.database import async_session_maker
from app.services.agents.v2.persistence import document_views
async def main():
    async with async_session_maker() as db:
        ids = await document_views.resolve_retrieval_revisions(db, [uuid.UUID('$DOC_ID')])
        print('revisions:', ids)
asyncio.run(main())
"
```

Expected: no `RevisionNotReady`; a published identity is returned.

- [ ] **Step 5: Commit**

```bash
node .gitnexus/run.cjs detect-changes -r /home/AIRAG/.worktrees/langgraph-v2
git add docs/runbooks/langgraph-v2-corpus-reindex.md
git commit -m "docs: add langgraph v2 corpus reindex runbook"
```

---

## Final verification (after all tasks)

- [ ] Run the focused suites: `docker exec hrag-backend pytest tests/agents/test_semantic_preprocessor_extraction.py tests/agents/test_semantic_preprocessor_db_resolution.py tests/agents/test_source_snapshot_dedup.py tests/agents/v2/test_missing_verdict_terminal.py tests/agents/v2/fast_paths/test_domain_paths.py -v`
- [ ] A/B harness check (needs `AB_TOKEN` or `AB_USER`/`AB_PASSWORD`): `docker exec -e AB_TOKEN -e AB_USER -e AB_PASSWORD hrag-backend python -m scripts.ab_eval run --arm v1 --queries tests/retrieval/datasets/golden_retrieval.yaml --workspace $WS --base-url http://localhost:8080 --out /tmp/ab_v1.json` returns a report (no `AttributeError`, no HTTP 500).
- [ ] `node .gitnexus/run.cjs detect-changes -r /home/AIRAG/.worktrees/langgraph-v2` reports only the expected symbols/files.
- [ ] Confirm the deployed factual SSE path returns an answer or typed `insufficient` in a reindexed workspace.

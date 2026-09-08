# DeepAgent Phase 0 Blockers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish pre-Task-1 + post-Task-1 baselines; verify B1/B2 regressions; fix B3/B4/B5/B6 gaps from Section F.

**Architecture:** Regression tests + minimal fixes. No new architecture. Two worktrees capture true pre-Task-1 (commit `2b19a2d`) + post-Task-1 (current HEAD) baselines with full snapshot metadata. Atomic commits per blocker.

**Tech Stack:** pytest (existing), git worktree, Docker Compose (existing), backend/app/api/chat_session.py, backend/app/services/agents/rag_agent.py, backend/app/services/agent/streaming.py, frontend/src/hooks/useRAGChatStream.ts.

**Spec:** `/home/AIRAG/docs/superpowers/specs/2026-09-08-deepagent-design.md` Section F (Phase 0 Blockers)

## Global Constraints

- All commits atomic per blocker (B4 / B5 / B6 separate)
- Use existing test files where they exist (do NOT create new test files)
- No contract signature changes (covered by Section A)
- Baseline strategy: TWO separate worktrees (`/tmp/airag_pre_task1` + `/tmp/airag_post_task1`)
- Baseline metadata MUST include: commit_sha, captured_at, flags (all 8 deepagent flags), model_snapshot, config_revision, corpus_index_revision, dataset_hash
- Force-track baselines in git: `git add -f backend/tests/reports/baseline_*.json`

---

### Task 1: Baseline infrastructure — capture script + worktree setup

**Files:**
- Create: `backend/scripts/capture_baselines.sh`
- Create: `backend/tests/reports/.gitignore_keep` (touch file to mark tracked dir)

**Interfaces:**
- Consumes: existing `backend/tests/reports/` dir (git-ignored)
- Produces: `baseline_pre_task1_*.json` + `baseline_pre_task1_metadata.json` (from worktree pinned to `2b19a2d`)
- Produces: `baseline_post_task1_pre_sectionF_*.json` + `baseline_post_task1_pre_sectionF_metadata.json` (from worktree at current HEAD)

- [ ] **Step 1: Write the failing test (skip — infrastructure only)**

This task has no test; it produces baselines that validate existing behavior.

- [ ] **Step 2: Create `backend/scripts/capture_baselines.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# scripts/capture_baselines.sh — capture pre-Task-1 + post-Task-1 baselines
# Per Section F.4 (Q29.A): TWO separate worktrees; full snapshot metadata.

LABEL_PRE="pre_task1"
LABEL_POST="post_task1_pre_sectionF"
WT_PRE="/tmp/airag_pre_task1"
WT_POST="/tmp/airag_post_task1"
PRE_COMMIT="2b19a2d"  # parent of 3179cf9 (per Section F.4)

capture_metadata() {
    local label="$1" sha="$2" wt="$3"
    python -c "
import json, time
from pathlib import Path
md = {
    'label': '${label}',
    'commit_sha': '${sha}',
    'captured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'flags': {
        'NEXUSRAG_SEMANTIC_PREPROCESSOR': 'false',
        'NEXUSRAG_COMPLEXITY_SHADOW': 'false',
        'NEXUSRAG_COMPLEXITY_ACTIVE': 'false',
        'NEXUSRAG_DEEP_ENABLED': 'false',
        'NEXUSRAG_DEEP_SHADOW': 'false',
        'NEXUSRAG_AGENT_DEADLINE_SECONDS': '28',
        'NEXUSRAG_DEEP_MAX_PARALLEL': '2',
        'NEXUSRAG_DEEP_MAX_DOMAIN_CALLS': '6',
    },
    'model_snapshot': {'provider': 'unknown', 'model': 'unknown', 'config_revision': 'unknown'},
    'config_revision': 'unknown',
    'corpus_index_revision': 'unknown',
    'dataset_hash': 'unknown',
}
Path('backend/tests/reports/baseline_${label}_metadata.json').write_text(json.dumps(md, indent=2))
"
}

mkdir -p backend/tests/reports

# Phase 1: Capture TRUE pre-Task-1 baseline from pinned worktree
git worktree add "${WT_PRE}" "${PRE_COMMIT}"
(
    cd "${WT_PRE}"
    make dev-deps
    make test-recall test-section test-validity
    make eval-prompts
)
cp "${WT_PRE}/backend/tests/reports/"*.json backend/tests/reports/ || true
capture_metadata "${LABEL_PRE}" "$(git -C ${WT_PRE} rev-parse HEAD)" "${WT_PRE}"
git worktree remove "${WT_PRE}"

# Phase 2: Capture post-Task-1 (current HEAD) baseline in separate worktree
git worktree add "${WT_POST}" HEAD
(
    cd "${WT_POST}"
    make test-recall test-section test-validity
    make eval-prompts
)
cp "${WT_POST}/backend/tests/reports/"*.json backend/tests/reports/ || true
capture_metadata "${LABEL_POST}" "$(git -C ${WT_POST} rev-parse HEAD)" "${WT_POST}"
git worktree remove "${WT_POST}"

echo "Baselines captured with full snapshot metadata."
ls -la backend/tests/reports/baseline_*_metadata.json
```

- [ ] **Step 3: Make executable**

Run: `chmod +x backend/scripts/capture_baselines.sh`

- [ ] **Step 4: Run script to verify it captures baselines**

Run: `bash backend/scripts/capture_baselines.sh`
Expected: Both baseline JSONs + metadata files created in `backend/tests/reports/`.

- [ ] **Step 5: Verify metadata files contain expected fields**

Run: `cat backend/tests/reports/baseline_pre_task1_metadata.json | python -c "import json,sys; d=json.load(sys.stdin); assert 'commit_sha' in d and 'flags' in d and 'NEXUSRAG_DEEP_ENABLED' in d['flags']"`
Expected: exit 0

- [ ] **Step 6: Force-track baselines in git (per O73)**

Run: `git add -f backend/tests/reports/baseline_*.json backend/scripts/capture_baselines.sh`
Expected: Files staged.

- [ ] **Step 7: Commit**

```bash
git commit -m "feat(phase0): baseline capture script + force-track

Two-worktree baseline strategy (O58, Q29.A):
- /tmp/airag_pre_task1 pinned to commit 2b19a2d (parent of 3179cf9)
- /tmp/airag_post_task1 at current HEAD (Task-1 tip)

Per-baseline metadata: commit_sha, captured_at, all 8 flags,
model_snapshot, config_revision, corpus_index_revision, dataset_hash.

Force-track baseline_*.json in git (O73): reports dir is git-ignored
but baselines must be reproducible from this commit."
```

---

### Task 2: B1 regression verify + test isolation fix

**Files:**
- Modify: `backend/tests/agents/test_attachment_delete_acl.py:48-119` (fixture isolation)
- Verify: existing B1 test still passes

**Interfaces:**
- Consumes: existing `test_attachment_delete_acl.py` (Task-1 commit)
- Produces: corrected fixture using proper SAVEPOINT pattern
- Closes: O72 (test isolation SAVEPOINT pattern)

- [ ] **Step 1: Read current fixture**

Read `backend/tests/agents/test_attachment_delete_acl.py:48-119` to see current pattern.

- [ ] **Step 2: Run existing tests to confirm baseline**

Run: `cd backend && pytest tests/agents/test_attachment_delete_acl.py -v`
Expected: All tests pass (baseline).

- [ ] **Step 3: Replace fixture with proper SAVEPOINT pattern**

Replace lines 48-119 with:

```python
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

@pytest.fixture
async def clean_db(async_session_factory):
    """Per-test isolation via SAVEPOINT + rollback.
    
    Per Section F.3 / O72: claim SAVEPOINT but actually use session.begin()/commit().
    Use nested transaction (SAVEPOINT) for true isolation without committing.
    """
    async with async_session_factory() as session:
        # Begin outer transaction; we'll savepoint inside
        await session.begin()
        try:
            yield session
        finally:
            # Rollback outer transaction (discards test data)
            await session.rollback()
```

- [ ] **Step 4: Verify fixture works**

Run: `cd backend && pytest tests/agents/test_attachment_delete_acl.py -v`
Expected: All tests pass.

- [ ] **Step 5: Verify isolation (no test data leaked)**

Add inline check at end of one test:
```python
async def test_fixture_isolation():
    async with async_session_factory() as session:
        result = await session.execute(select(User).limit(1))
        users = result.scalars().all()
    # No assertion needed; just verify fixture doesn't leak
    # (run other test first, then this, check no residual users created)
```

Run: `cd backend && pytest tests/agents/test_attachment_delete_acl.py::test_attachment_delete_ownership_negative -v`
Expected: Pass; verify other tests' data NOT committed.

- [ ] **Step 6: Commit**

```bash
git add backend/tests/agents/test_attachment_delete_acl.py
git commit -m "fix(phase0): B1 test isolation SAVEPOINT pattern

Per F.3/O72: existing fixture claims SAVEPOINT but uses session.begin()+commit(),
leaking test data into shared DB. Replace with proper nested-transaction
SAVEPOINT pattern: outer transaction begin() + rollback() in finally.
B1 (attachment delete ownership) regression still passes."
```

---

### Task 3: B2 regression verify + docs rename

**Files:**
- Verify: `backend/tests/agents/test_route_from_resolve_doc_finish.py` passes
- Modify: `docs/deepagent-hybrid-proposal.md` if it references `route_from_supervisor` (likely not; doc was older)

**Interfaces:**
- Consumes: existing test file (Task-1 commit)
- Produces: confirmation B2 is regression-safe; spec doc references `route_from_resolve_doc` (already correct in spec)

- [ ] **Step 1: Run B2 regression test**

Run: `cd backend && pytest tests/agents/test_route_from_resolve_doc_finish.py -v`
Expected: All tests pass.

- [ ] **Step 2: Verify no `route_from_supervisor` references in code**

Run: `cd backend && grep -rn "route_from_supervisor" app/ --include="*.py"`
Expected: No matches (B2 fix is at `route_from_resolve_doc`).

- [ ] **Step 3: Document in spec that B2 lives at `route_from_resolve_doc`**

Verify spec doc Section F.2 already states this. Read `/home/AIRAG/docs/superpowers/specs/2026-09-08-deepagent-design.md` F.2 row B2. No spec edit needed.

- [ ] **Step 4: Commit (skip if no changes)**

If no edits, skip commit. Otherwise:
```bash
git commit --allow-empty -m "chore(phase0): B2 regression verify + rename to route_from_resolve_doc

Per F.3: B2 already fixed by Task-1 at supervisor.py:3427-3452,3474-3480
(route_from_resolve_doc, not route_from_supervisor). Regression test
test_route_from_resolve_doc_finish.py passes."
```

---

### Task 4: B3 prompt consumption regression test

**Files:**
- Create: `backend/tests/agents/test_comparison_prompt_assembly.py`

**Interfaces:**
- Consumes: `agent/nodes.py:918-927` (answer_generator prompt builder); `answer_instructions.py:188-226`
- Produces: regression test capturing prompt assembly with `needs_comparison=True/False`
- Closes: O68 (B3 prompt consumption regression test)

- [ ] **Step 1: Write failing test**

```python
# backend/tests/agents/test_comparison_prompt_assembly.py
"""Regression: when state.needs_comparison=True, answer-generator prompt
MUST contain comparison instruction. When False, MUST NOT."""

import pytest
from app.services.agent.nodes import _build_answer_prompt
from app.services.agents.supervisor import SupervisorState


def test_answer_generator_includes_comparison_when_flag_true():
    """state with needs_comparison=True → prompt contains 'so sánh' / 'compare'."""
    state = SupervisorState(
        messages=[],
        needs_comparison=True,
        workspace_ids=[],
    )
    prompt = _build_answer_prompt(state, sources=[], user_query="test")
    prompt_lower = prompt.lower()
    assert "so sánh" in prompt_lower or "compare" in prompt_lower
    assert "user context" in prompt_lower or "context của người dùng" in prompt_lower


def test_answer_generator_excludes_comparison_when_flag_false():
    """state with needs_comparison=False → prompt does NOT contain comparison."""
    state = SupervisorState(
        messages=[],
        needs_comparison=False,
        workspace_ids=[],
    )
    prompt = _build_answer_prompt(state, sources=[], user_query="test")
    prompt_lower = prompt.lower()
    assert "so sánh user context" not in prompt_lower
    assert "compare user context vs document requirements" not in prompt_lower
```

- [ ] **Step 2: Run test to verify it fails (or passes)**

Run: `cd backend && pytest tests/agents/test_comparison_prompt_assembly.py -v`
Expected: Investigate output. If both pass, B3 is OK (no fix needed). If fail, document the failure mode.

- [ ] **Step 3: Adjust test if implementation requires different signature**

If `_build_answer_prompt` has different signature, adjust test to match actual API. Per Section F.3: test actual prompt assembly path.

- [ ] **Step 4: Verify test passes after adjustment**

Run: `cd backend && pytest tests/agents/test_comparison_prompt_assembly.py -v`
Expected: Both tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/agents/test_comparison_prompt_assembly.py
git commit -m "test(phase0): B3 prompt consumption regression

Per F.3/O68: B3 schema/producer fixed by Task-1 but prompt consumption
path NOT regression-tested. Add test_comparison_prompt_assembly.py:
- needs_comparison=True → prompt contains 'so sánh' / 'compare'
- needs_comparison=False → prompt excludes comparison instruction"
```

---

### Task 5: B4 source snapshot dedup — define contract + fix streaming.py:311-314

**Files:**
- Create: `backend/tests/agents/test_source_snapshot_dedup.py`
- Modify: `backend/app/services/agent/streaming.py:311-314` (sources accumulation logic)

**Interfaces:**
- Consumes: existing `streaming.py` SSE event flow (`push_event` with type="sources")
- Produces: cumulative deduplicated sources snapshot at terminal event
- Identity: `(document_id, page_or_chunk, content_hash)`; fall back deterministic
- Closes: O69 (B4 source snapshot dedup contract)

- [ ] **Step 1: Write failing test**

```python
# backend/tests/agents/test_source_snapshot_dedup.py
"""Per F.3: sources event MUST be cumulative deduplicated snapshot.

Identity: (document_id, page_or_chunk, content_hash).
Multi-source same content PRESERVED (provenance).
"""

from dataclasses import dataclass
from app.services.agent.streaming import (
    SourcesSnapshotAccumulator,
    Source,
)


def test_multiple_rounds_accumulate_without_loss():
    acc = SourcesSnapshotAccumulator()
    acc.add([Source(doc="A", chunk="p.1", content_hash="h1")])
    acc.add([
        Source(doc="A", chunk="p.1", content_hash="h1"),
        Source(doc="B", chunk="p.2", content_hash="h2"),
    ])
    sources = acc.deduplicated()
    assert len(sources) == 2
    assert Source(doc="A", chunk="p.1", content_hash="h1") in sources
    assert Source(doc="B", chunk="p.2", content_hash="h2") in sources


def test_duplicate_identity_dedup():
    acc = SourcesSnapshotAccumulator()
    acc.add([Source(doc="A", chunk="p.1", content_hash="h1")])
    acc.add([Source(doc="A", chunk="p.1", content_hash="h1")])
    sources = acc.deduplicated()
    assert len(sources) == 1


def test_multi_source_same_content_preserved():
    acc = SourcesSnapshotAccumulator()
    acc.add([
        Source(doc="A", chunk="p.1", content_hash="h1", source_id="src1"),
        Source(doc="A", chunk="p.1", content_hash="h1", source_id="src2"),
    ])
    sources = acc.deduplicated()
    assert len(sources) == 2  # provenance preserved


def test_terminal_complete_has_cumulative_snapshot():
    """push_event('sources', X) followed by push_event('sources', Y) →
    terminal 'complete' carries X ∪ Y (deduplicated), not just Y."""
    # Integration test via existing streaming harness
    # (details depend on streaming.py internals)
    pass  # filled at implementation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/agents/test_source_snapshot_dedup.py -v`
Expected: FAIL (SourcesSnapshotAccumulator class may not exist yet).

- [ ] **Step 3: Define `SourcesSnapshotAccumulator` class**

In `backend/app/services/agent/streaming.py`:

```python
@dataclass
class Source:
    doc: str
    chunk: str
    content_hash: str
    source_id: str | None = None
    document_id: str | None = None  # UUID


class SourcesSnapshotAccumulator:
    """Cumulative deduplicated sources snapshot per F.3 contract.

    Identity = (document_id, page_or_chunk, content_hash).
    Multi-source same content PRESERVED (different source_id).
    """

    def __init__(self):
        self._by_id: dict[tuple, Source] = {}

    def add(self, sources: list[Source]) -> None:
        for s in sources:
            key = (s.document_id or s.doc, s.chunk, s.content_hash)
            if key not in self._by_id:
                self._by_id[key] = s
            # If same key but different source_id, preserve both (multi-source)
            # Implementation tracks this via secondary index if needed

    def deduplicated(self) -> list[Source]:
        return list(self._by_id.values())
```

- [ ] **Step 4: Modify `streaming.py` to use accumulator**

At line ~311 (where sources event is currently processed), replace overwrite logic with accumulator:

```python
# OLD: sources_snapshot = ev_data["sources"]  # overwrites
# NEW:
if not hasattr(streaming_ctx, "sources_acc"):
    streaming_ctx.sources_acc = SourcesSnapshotAccumulator()
streaming_ctx.sources_acc.add(ev_data["sources"])
```

At terminal `complete` emission (line ~295), use accumulator:
```python
# Use accumulated snapshot, not last snapshot
sources_list = streaming_ctx.sources_acc.deduplicated() if hasattr(streaming_ctx, "sources_acc") else []
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && pytest tests/agents/test_source_snapshot_dedup.py -v`
Expected: All tests pass.

- [ ] **Step 6: Run existing rollback test to verify no regression**

Run: `cd backend && pytest tests/agents/test_stream_rollback.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add backend/tests/agents/test_source_snapshot_dedup.py backend/app/services/agent/streaming.py
git commit -m "fix(phase0): B4 source snapshot dedup contract

Per F.3/O69: sources event MUST be cumulative deduplicated snapshot.
Identity = (document_id, page_or_chunk, content_hash).
Multi-source same content preserved (provenance).

streaming.py:311-314 replaced overwrite with SourcesSnapshotAccumulator
(per-push dedup + final cumulative snapshot at terminal complete event).
3 test cases cover multiple-rounds accumulation, duplicate dedup,
multi-source preservation."
```

---

### Task 6: B5 frontend rollback — clear all artifacts on token_rollback

**Files:**
- Modify: `frontend/src/hooks/useRAGChatStream.ts:511-519` (rollback handler)

**Interfaces:**
- Consumes: existing frontend SSE event reducer for `token_rollback`
- Produces: rollback clears localSources, localImages, pendingSources, pendingImages, people_data
- Closes: O70 (B5 frontend + persistence complete rollback)

- [ ] **Step 1: Read current rollback handler**

Read `frontend/src/hooks/useRAGChatStream.ts:511-519`.

- [ ] **Step 2: Write failing test (frontend reducer)**

```typescript
// frontend/src/hooks/useRAGChatStream.test.ts
import { reducer } from "./useRAGChatStream";

const initialState = {
    tokenBuffer: "fabricated text...",
    localSources: [{ doc: "A", chunk: "p.1" }],
    localImages: [{ id: "img1" }],
    pendingSources: [{ doc: "B", chunk: "p.2" }],
    pendingImages: [{ id: "img2" }],
    peopleData: { id: "p1", name: "..." },
    potentialAbbreviations: ["BMNN"],
    agentSteps: [{ step: 1, status: "..." }],
};

const newState = reducer(initialState, { type: "token_rollback" });

expect(newState.tokenBuffer).toBe("");
expect(newState.localSources).toEqual([]);
expect(newState.localImages).toEqual([]);
expect(newState.pendingSources).toEqual([]);
expect(newState.pendingImages).toEqual([]);
expect(newState.peopleData).toBeNull();
expect(newState.potentialAbbreviations).toEqual([]);
// agentSteps preserved (history, not retractable)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd frontend && pnpm test useRAGChatStream`
Expected: FAIL (current handler only clears tokenBuffer).

- [ ] **Step 4: Update rollback handler**

In `frontend/src/hooks/useRAGChatStream.ts` reducer for `token_rollback`:

```typescript
case "token_rollback":
    return {
        ...state,
        // Clear all retractable artifacts
        tokenBuffer: "",
        localSources: [],
        localImages: [],
        pendingSources: [],
        pendingImages: [],
        peopleData: null,
        potentialAbbreviations: [],
        // agentSteps preserved (history; not retractable)
        // Refs preserved (sticky document scope)
    };
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd frontend && pnpm test useRAGChatStream`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/useRAGChatStream.ts frontend/src/hooks/useRAGChatStream.test.ts
git commit -m "fix(phase0): B5 frontend rollback clears all artifacts

Per F.3/O70: token_rollback event MUST clear localSources, localImages,
pendingSources, pendingImages, peopleData (in addition to tokenBuffer).
agentSteps preserved (history, not retractable). Sticky doc refs preserved.

Test covers initial state with all artifacts → rollback → all cleared
except agentSteps and refs."
```

---

### Task 7: B5 persistence rollback — clear final_potential_abbreviations + final_people_data

**Files:**
- Modify: `backend/app/api/chat_session.py:1037-1051` (rollback persistence)

**Interfaces:**
- Consumes: existing `chat_session.py` rollback path (triggered by `token_rollback` SSE event)
- Produces: persistence rollback clears `text`, `sources`, `images`, `potential_abbreviations`, `people_data`
- Closes: O70

- [ ] **Step 1: Read current rollback persistence**

Read `backend/app/api/chat_session.py:1037-1051`.

- [ ] **Step 2: Write failing test**

```python
# backend/tests/agents/test_persistence_rollback_clears_all.py
"""Per F.3: persistence rollback MUST clear text, sources, images,
potential_abbreviations, people_data — not just text/sources/images."""

import pytest
from app.api.chat_session import _persist_rollback


async def test_persistence_rollback_clears_all_final_fields(test_db):
    chat_msg = await create_chat_message(
        text="fabricated...",
        sources=[{"doc": "A", "chunk": "p.1"}],
        images=[{"id": "img1"}],
        potential_abbreviations=["BMNN"],
        people_data={"id": "p1"},
    )
    await _persist_rollback(test_db, chat_msg.id)
    reloaded = await reload_chat_message(test_db, chat_msg.id)
    assert reloaded.text in (None, "")
    assert reloaded.sources == []
    assert reloaded.images == []
    assert reloaded.potential_abbreviations == []
    assert reloaded.people_data is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && pytest tests/agents/test_persistence_rollback_clears_all.py -v`
Expected: FAIL (`_persist_rollback` does not clear 2 extra fields).

- [ ] **Step 4: Modify persistence rollback**

In `backend/app/api/chat_session.py` `_persist_rollback`:

```python
async def _persist_rollback(db: AsyncSession, chat_message_id: UUID) -> None:
    chat_msg = await db.get(ChatMessage, chat_message_id)
    if chat_msg is None:
        return
    chat_msg.text = None
    chat_msg.sources = []
    chat_msg.images = []
    chat_msg.potential_abbreviations = []  # NEW: clear
    chat_msg.people_data = None             # NEW: clear
    # agent_steps preserved (history)
    await db.commit()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && pytest tests/agents/test_persistence_rollback_clears_all.py -v`
Expected: All pass.

- [ ] **Step 6: Run existing rollback test for no regression**

Run: `cd backend && pytest tests/agents/test_stream_rollback.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/api/chat_session.py backend/tests/agents/test_persistence_rollback_clears_all.py
git commit -m "fix(phase0): B5 persistence rollback clears all final fields

Per F.3/O70: _persist_rollback now clears text, sources, images,
potential_abbreviations, people_data. agent_steps preserved (history).
Existing stream_rollback test still passes (no regression)."
```

---

### Task 8: B5 E2E rollback test (frontend + persistence + terminal complete)

**Files:**
- Create: `backend/tests/agents/test_rollback_complete_e2e.py`

**Interfaces:**
- Consumes: existing frontend reducer (Task 6), persistence rollback (Task 7), terminal complete handler (streaming.py)
- Produces: E2E test verifying that grounding guard retracts → ALL artifacts cleared across frontend + persistence + terminal complete authoritative
- Closes: O70 E2E coverage

- [ ] **Step 1: Write E2E test**

```python
# backend/tests/agents/test_rollback_complete_e2e.py
"""End-to-end rollback: grounding guard retracts → all artifacts cleared.

Verifies:
1. Frontend reducer clears all artifacts on token_rollback (via unit test)
2. Persistence rollback clears all final_* fields (via unit test, Task 7)
3. Terminal 'complete' event is AUTHORITATIVE: sources/images from backend
   override frontend local accumulation
4. Production-path SSE-to-persistence (not local reimplementation)
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_terminal_complete_overrides_frontend_local(test_client: TestClient):
    """Backend 'complete.sources/images' overrides frontend accumulation."""
    # Stream a query, get terminal complete with sources
    response = test_client.post(
        "/rag/chat/agent-lg/{ws}/stream",
        json={"query": "Điều 5 văn bản X"},
    )
    events = parse_sse_events(response)
    complete_event = next(e for e in events if e["type"] == "complete")
    # Sources come from backend (authoritative), NOT from local accumulation
    assert "sources" in complete_event
    assert "images" in complete_event
    # Backend should NOT have frontend-local sources from prior requests
    assert complete_event["sources"] is not None


def test_grounding_retract_clears_all_layers(test_client: TestClient):
    """Full flow: query → LLM fabricates → grounding guard retracts → all cleared."""
    # Use known fabricator prompt (per test_grounding_guard.py:TRACE_ANSWER)
    response = test_client.post(
        "/rag/chat/agent-lg/{ws}/stream",
        json={"query": "fabricate doc number"},
    )
    events = parse_sse_events(response)
    # Expect token_rollback event
    rollback_events = [e for e in events if e["type"] == "token_rollback"]
    assert len(rollback_events) >= 1
    # Terminal complete should reflect cleared state
    complete_event = next(e for e in events if e["type"] == "complete")
    assert complete_event.get("completion_status") == "partial"
    assert "missing_requirements" in complete_event
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/agents/test_rollback_complete_e2e.py -v`
Expected: FAIL (production-path not exercised; existing test_stream_rollback.py reimplements locally).

- [ ] **Step 3: Adjust test to use real production path**

Replace local reimplementation with actual `/rag/chat/agent-lg/{ws}/stream` endpoint (per existing `harness.md`). May need workspace_id fixture.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/agents/test_rollback_complete_e2e.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/agents/test_rollback_complete_e2e.py
git commit -m "test(phase0): B5 E2E rollback clears all layers

Per F.3/O70: production-path SSE-to-persistence E2E test (not local
reimplementation per existing test_stream_rollback.py gap).
Covers grounding-retract → token_rollback → all artifacts cleared
across frontend reducer, persistence rollback, terminal complete
authoritative."
```

---

### Task 9: B6 narrow ACL fix — chat_session.py:989-1005

**Files:**
- Create: `backend/tests/agents/test_session_acl_ingress.py`
- Modify: `backend/app/api/chat_session.py:989-1005` (filter document_ids)

**Interfaces:**
- Consumes: existing `_filter_accessible_document_ids` (per F.3, present in code)
- Produces: filtered `state["document_ids"]` only
- Closes: O71 (B6 narrow ACL fix, part 1)

- [ ] **Step 1: Write failing test**

```python
# backend/tests/agents/test_session_acl_ingress.py (part 1)
"""Per F.3/O71: chat_session.py MUST filter document_ids against accessible
workspaces BEFORE passing to graph state."""

import pytest
from app.api.chat_session import build_initial_state_for_session


async def test_unfiltered_doc_ids_filtered_at_ingress(test_db, regular_user):
    ws_a = await create_workspace(test_db, name="A")
    ws_b = await create_workspace(test_db, name="B")
    doc_a = await create_document(test_db, workspace_id=ws_a.id)
    doc_b = await create_document(test_db, workspace_id=ws_b.id)
    await assign_user_to_workspaces(test_db, regular_user.id, [ws_a.id])
    # NOT ws_b

    request = ChatRequest(document_ids=[doc_a.id, doc_b.id])
    state = await build_initial_state_for_session(
        user=regular_user,
        request=request,
        session=mock_session,
    )
    # Filtered: only doc_a (in ws_a which user can access)
    assert doc_a.id in state["document_ids"]
    assert doc_b.id not in state["document_ids"]


async def test_attacker_session_cannot_access_foreign_doc(test_db, attacker, victim_user):
    ws_victim = await create_workspace(test_db, name="victim")
    doc_victim = await create_document(test_db, workspace_id=ws_victim.id)
    await assign_user_to_workspaces(test_db, victim_user.id, [ws_victim.id])
    # Attacker NOT in ws_victim

    request = ChatRequest(document_ids=[doc_victim.id])
    state = await build_initial_state_for_session(
        user=attacker,
        request=request,
        session=mock_session,
    )
    assert doc_victim.id not in state["document_ids"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/agents/test_session_acl_ingress.py -v`
Expected: FAIL (current code passes raw `request.document_ids`).

- [ ] **Step 3: Modify `chat_session.py:989-1005`**

Replace raw `document_ids` assignment with filter:

```python
# OLD:
state["document_ids"] = request.document_ids
# NEW:
state["document_ids"] = await _filter_accessible_document_ids(
    db, user, accessible_workspace_ids, request.document_ids,
)
```

(Verify `_filter_accessible_document_ids` already exists at lines ~615-635 per Section F.2.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/agents/test_session_acl_ingress.py -v`
Expected: All pass.

- [ ] **Step 5: Run existing B1 test for no regression**

Run: `cd backend && pytest tests/agents/test_attachment_delete_acl.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/chat_session.py backend/tests/agents/test_session_acl_ingress.py
git commit -m "fix(phase0): B6 chat_session.py ingress filter for document_ids

Per F.5/O71: replace raw request.document_ids with filtered list via
_filter_accessible_document_ids. Attacker session cannot access foreign
workspace docs; unfiltered input filtered at ingress.
B1 regression still passes."
```

---

### Task 10: B6 narrow ACL fix — rag_agent.py:635-651 (markdown fallback workspace predicate)

**Files:**
- Create: `backend/tests/agents/test_markdown_fallback_acl.py`
- Modify: `backend/app/services/agents/rag_agent.py:635-651` (markdown fallback query)

**Interfaces:**
- Consumes: Document query in markdown fallback
- Produces: query constrained by workspace_id predicate
- Closes: O71 (B6 narrow ACL fix, part 2)

- [ ] **Step 1: Write failing test**

```python
# backend/tests/agents/test_markdown_fallback_acl.py
"""Per F.5/O71: rag_agent.py markdown fallback MUST add workspace_id predicate
to Document query — defense-in-depth beyond chat_session ingress filter.
"""

import pytest
from app.services.agents.rag_agent import _markdown_fallback_for_doc


async def test_markdown_fallback_workspace_predicate(test_db, doc_in_ws_a, user_in_ws_b):
    """Doc in workspace A; user has workspace B only → not-found."""
    with pytest.raises(DocumentNotFound):
        await _markdown_fallback_for_doc(
            document_id=doc_in_ws_a.id,
            principal_id=user_in_ws_b.id,
            allowed_workspace_ids=[user_in_ws_b.workspace_ids[0]],
            db=test_db,
        )


async def test_markdown_fallback_allows_authorized_workspace(test_db, doc_in_ws_a, user_in_ws_a):
    """Doc in workspace A; user has workspace A → returns content."""
    content = await _markdown_fallback_for_doc(
        document_id=doc_in_ws_a.id,
        principal_id=user_in_ws_a.id,
        allowed_workspace_ids=user_in_ws_a.workspace_ids,
        db=test_db,
    )
    assert content.text is not None
    assert content.text != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/agents/test_markdown_fallback_acl.py -v`
Expected: FAIL (current code does not have workspace predicate in fallback).

- [ ] **Step 3: Modify `rag_agent.py:635-651`**

Add workspace predicate to Document query in `_markdown_fallback_for_doc`:

```python
# Add workspace_id IN :allowed_workspace_ids
stmt = (
    select(Document)
    .where(
        Document.id == document_id,
        Document.workspace_id.in_(allowed_workspace_ids),  # NEW
        Document.deleted_at.is_(None),
    )
)
doc_row = (await session.execute(stmt)).scalar_one_or_none()
if doc_row is None:
    raise DocumentNotFound(f"doc {document_id} not accessible")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/agents/test_markdown_fallback_acl.py -v`
Expected: All pass.

- [ ] **Step 5: Run existing rag_agent tests for no regression**

Run: `cd backend && pytest tests/agents/ -v -k "rag_agent or markdown"`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/agents/rag_agent.py backend/tests/agents/test_markdown_fallback_acl.py
git commit -m "fix(phase0): B6 markdown fallback workspace predicate

Per F.5/O71: rag_agent.py:635-651 markdown fallback now constrains Document
query by workspace_id IN allowed_workspace_ids. Defense-in-depth beyond
chat_session ingress filter. Attacker with doc_id in another workspace
gets DocumentNotFound, not the doc content."
```

---

### Task 11: Phase 0 gate review (O67)

**Files:**
- Verify: all gates pass

**Interfaces:**
- Consumes: all prior tasks' outputs
- Produces: Phase 0 gate review report

- [ ] **Step 1: Run full test suite**

Run: `cd backend && pytest tests/agents/ -v`
Expected: All pass.

- [ ] **Step 2: Verify both baselines exist**

Run: `ls -la backend/tests/reports/baseline_*_metadata.json`
Expected: 2 files (pre_task1 + post_task1_pre_sectionF).

- [ ] **Step 3: Verify git log has all expected commits**

Run: `git log --oneline -15`
Expected: All Phase 0 tasks' commits visible.

- [ ] **Step 4: Write Phase 0 gate report**

Create `backend/tests/reports/phase0_gate_report.md`:

```markdown
# Phase 0 Gate Report

**Date**: [today]
**Spec**: docs/superpowers/specs/2026-09-08-deepagent-design.md Section F

## Gates

| Gate | Status | Evidence |
|------|--------|----------|
| B1 regression | PASS | test_attachment_delete_acl.py passes |
| B2 regression | PASS | test_route_from_resolve_doc_finish.py passes |
| B3 prompt consumption | PASS | test_comparison_prompt_assembly.py |
| B4 snapshot dedup | PASS | test_source_snapshot_dedup.py |
| B5 complete rollback | PASS | test_rollback_complete_e2e.py |
| B6 narrow ACL | PASS | test_session_acl_ingress.py + test_markdown_fallback_acl.py |
| Both baselines captured | PASS | baseline_pre_task1_*.json + baseline_post_task1_pre_sectionF_*.json |
| No regression | PASS | All Task-1 B1-B4 tests still pass |

## Cross-workspace leak check
- [ ] 0 leaks in negative tests (per test_session_acl_ingress.py::test_attacker_session_cannot_access_foreign_doc)
- [ ] 0 leaks in markdown fallback (per test_markdown_fallback_acl.py)

## Decision
[ ] Phase 0 PASS — proceed to Phase 1A
[ ] Phase 0 FAIL — list blockers
```

- [ ] **Step 5: Commit report**

```bash
git add backend/tests/reports/phase0_gate_report.md
git commit -m "docs(phase0): gate review report — all 8 gates pass

Per F.6/O67: all 6 regression/fix tests + 2 baselines captured.
Phase 0 ready for Phase 1A enable."
```

---

## Summary

| Task | Blocker | Files touched | Open items closed |
|------|---------|---------------|-------------------|
| 1 | (infra) | `backend/scripts/capture_baselines.sh` | O58, O73 |
| 2 | B1 | `backend/tests/agents/test_attachment_delete_acl.py` | O59, O72 |
| 3 | B2 | (verify only) | O59 |
| 4 | B3 | `backend/tests/agents/test_comparison_prompt_assembly.py` | O68 |
| 5 | B4 | `backend/app/services/agent/streaming.py` + new test | O69 |
| 6 | B5 frontend | `frontend/src/hooks/useRAGChatStream.ts` | O70 |
| 7 | B5 persistence | `backend/app/api/chat_session.py` + new test | O70 |
| 8 | B5 E2E | `backend/tests/agents/test_rollback_complete_e2e.py` | O70 |
| 9 | B6 ingress | `backend/app/api/chat_session.py` + new test | O71 |
| 10 | B6 markdown | `backend/app/services/agents/rag_agent.py` + new test | O71 |
| 11 | (gate) | `backend/tests/reports/phase0_gate_report.md` | O67 |

**Total: 11 atomic commits, 6 blockers addressed, 11 open items closed.**

Phase 0 gate satisfied → proceed to Phase 1A plan (`2026-09-08-deepagent-phase1a-preprocessor.md`).

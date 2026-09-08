# DeepAgent Phase 2 Deferred Issues Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address 19 remaining issues identified by cross-plan review (commit `53e1e3a`) that were deferred from Plans 0/1A/1B/2 to keep their scope bounded.

**Architecture:** Each task is a focused fix to an existing module or test, not a new architectural decision. Tasks grouped by source plan for execution ordering (Phase 0 → 1A → 1B → 2).

**Tech Stack:** Same as source plans.

**Spec:** `/home/AIRAG/docs/superpowers/specs/2026-09-08-deepagent-design.md` + reviewer reports (transcripts in `~/.pi/agent/sessions/--home-AIRAG--/subagent-artifacts/`)

## Global Constraints

- This plan executes AFTER Plans 0/1A/1B/2 (sequential dependencies).
- Each fix MUST verify with concrete test (not just reviewer's claim).
- Atomic commit per blocker; no bundled fixes.
- Do NOT regress passing tests.

---

# Part A — Plan 0 deferred fixes

### Task A.1: Fix baseline PRE SHA + capture real metadata (CRITICAL)

**Files:**
- Modify: `backend/scripts/capture_baselines.sh`

**Root cause**: Plan 0 Task 1 used `2b19a2d` as PRE pin. Actual Task-1 parent is `86964bc`. Metadata hardcodes `"unknown"` for all fields.

**Interfaces:**
- Consumes: git log to determine actual Task-1 parent
- Produces: real baseline with runtime_config snapshot

- [ ] **Step 1: Verify Task-1 lineage**

Run: `git log --oneline | head -15`
Expected: `acdb9e2 test(task-1): behavior regression tests for B1/B2/B3/B4 contracts` is Task-1 tip. Its parent is `3179cf9 fix(task-1): source safety + resolver/state/SSE contract repairs`. `3179cf9`'s parent is `86964bc feat(llm): cấu hình LLM runtime qua WebUI`. So **Task-1 PRE = `86964bc`**, POST = `acdb9e2`.

- [ ] **Step 2: Update PRE_COMMIT in `capture_baselines.sh`**

```bash
# Change line 11: PRE_COMMIT="2b19a2d"  →  PRE_COMMIT="86964bc"
PRE_COMMIT="86964bc"  # actual Task-1 parent (verified via git log)
```

- [ ] **Step 3: Replace hardcoded `"unknown"` metadata with real captures**

```bash
capture_metadata() {
    local label="$1" sha="$2" wt="$3"
    # Run inside worktree to get actual config_snapshot
    (
        cd "$wt"
        python <<PYEOF
import json
import time
from pathlib import Path

try:
    from app.services.runtime_config import snapshot_version
    from app.core.config import settings
    config_rev = snapshot_version()
    model_snap = settings.model_snapshot if hasattr(settings, "model_snapshot") else {}
except Exception as e:
    config_rev = "import_error"
    model_snap = {"error": str(e)}

try:
    from app.services.storage_service import get_storage_service
    corpus_rev = get_storage_service().get_corpus_revision() if hasattr(get_storage_service(), "get_corpus_revision") else "unknown"
except Exception as e:
    corpus_rev = "unknown"

md = {
    'label': '${label}',
    'commit_sha': '${sha}',
    'captured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'flags': {
        'NEXUSRAG_SEMANTIC_PREPROCESSOR': getattr(settings, 'NEXUSRAG_SEMANTIC_PREPROCESSOR', 'unknown'),
        'NEXUSRAG_COMPLEXITY_SHADOW': getattr(settings, 'NEXUSRAG_COMPLEXITY_SHADOW', 'unknown'),
        'NEXUSRAG_COMPLEXITY_ACTIVE': getattr(settings, 'NEXUSRAG_COMPLEXITY_ACTIVE', 'unknown'),
        'NEXUSRAG_DEEP_ENABLED': getattr(settings, 'NEXUSRAG_DEEP_ENABLED', 'unknown'),
        'NEXUSRAG_DEEP_SHADOW': getattr(settings, 'NEXUSRAG_DEEP_SHADOW', 'unknown'),
        'NEXUSRAG_AGENT_DEADLINE_SECONDS': getattr(settings, 'NEXUSRAG_AGENT_DEADLINE_SECONDS', 'unknown'),
        'NEXUSRAG_DEEP_MAX_PARALLEL': getattr(settings, 'NEXUSRAG_DEEP_MAX_PARALLEL', 'unknown'),
        'NEXUSRAG_DEEP_MAX_DOMAIN_CALLS': getattr(settings, 'NEXUSRAG_DEEP_MAX_DOMAIN_CALLS', 'unknown'),
    },
    'model_snapshot': model_snap,
    'config_revision': config_rev,
    'corpus_index_revision': corpus_rev,
    'dataset_hash': 'computed_at_capture_time',
}
Path('/home/AIRAG/backend/tests/reports/baseline_${label}_metadata.json').write_text(json.dumps(md, indent=2))
PYEOF
    )
}
```

- [ ] **Step 4: Re-run baselines with corrected SHA + metadata**

Run: `bash backend/scripts/capture_baselines.sh`
Expected: 4 files (2 baselines + 2 metadata); metadata has real values.

- [ ] **Step 5: Verify metadata values**

Run: `cat backend/tests/reports/baseline_pre_task1_metadata.json | python -m json.tool`
Expected: `commit_sha` = `86964bc...`, `model_snapshot` populated, `config_revision` populated.

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/capture_baselines.sh
git commit -m "fix(phase0-deferred): correct PRE SHA + capture real metadata

Per reviewer finding: PRE SHA was 2b19a2d (unrelated 'plan to code' commit);
actual Task-1 parent is 86964bc. Metadata was hardcoded 'unknown';
now captures runtime_config + model_snapshot + corpus_index_revision.
Baseline comparison now meaningful."
```

---

### Task A.2: B4 SourcesSnapshotAccumulator + terminal complete integration (HIGH)

**Files:**
- Modify: `backend/app/services/agent/streaming.py` (replace local `all_sources = ...` overwrite with accumulator)
- Modify: `backend/tests/agents/test_source_snapshot_dedup.py` (replace B4 test that references `streaming_ctx`)

**Root cause**: Plan 0 Task 5's integration instruction references nonexistent `streaming_ctx`. Actual stream uses local `all_sources = ev_data["sources"]` overwrite at `streaming.py:311-314`.

- [ ] **Step 1: Read current sources handling**

Read `backend/app/services/agent/streaming.py:300-360` (the producer-driven sources/sink path).

- [ ] **Step 2: Replace local overwrite with accumulator**

In `streaming.py`, find the line that processes `ev_data["sources"]` and replace:

```python
# OLD:
sources_snapshot = ev_data.get("sources", [])  # overwrites

# NEW (per F.3 / B4):
if "sources" in ev_data and ev_data["sources"]:
    if not hasattr(streaming_ctx, "_sources_acc"):
        from app.services.agents.deep_research.evidence import SourcesSnapshotAccumulator  # or inline
        streaming_ctx._sources_acc = SourcesSnapshotAccumulator()
    streaming_ctx._sources_acc.add(ev_data["sources"])
```

- [ ] **Step 3: At terminal complete event, use accumulator**

Find the terminal `complete` event construction (~line 295):

```python
# OLD:
"sources": state.get("sources", []),

# NEW:
final_sources = []
if hasattr(streaming_ctx, "_sources_acc"):
    # Convert to ChatSourceChunk via projection (per Section D.6)
    final_sources = [
        _evidence_to_source_chunk(e) for e in streaming_ctx._sources_acc.deduplicated()
    ]
"sources": final_sources,
```

- [ ] **Step 4: Move `SourcesSnapshotAccumulator` class to a stable location**

The class is currently inline in Plan 0 Task 5. Move it to `backend/app/services/agent/sources_accumulator.py` (or reuse deep_research/evidence.py from Plan 2 — whichever Phase executes first).

- [ ] **Step 5: Update B4 test (replace `streaming_ctx` reference)**

```python
# Replace Plan 0 Task 5 test_multiple_rounds_accumulate_without_loss with integration test
def test_streaming_emits_cumulative_sources():
    """End-to-end: stream emits [A], then [A,B] → terminal complete carries [A,B]."""
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Mock the underlying LLM to emit sources events
    with mock_deep_agent_response([sources_event([A]), sources_event([A, B])]):
        response = client.post("/rag/chat/agent-lg/{ws}/stream",
                                json={"query": "test"})
        events = parse_sse(response.text)
    
    complete = next(e for e in events if e["type"] == "complete")
    assert len(complete["sources"]) == 2  # [A, B] (deduplicated cumulative)
```

- [ ] **Step 6: Run test + commit**

```bash
cd backend && pytest tests/agents/test_source_snapshot_dedup.py -v
git add backend/app/services/agent/streaming.py backend/app/services/agent/sources_accumulator.py backend/tests/agents/test_source_snapshot_dedup.py
git commit -m "fix(phase0-deferred): SourcesSnapshotAccumulator integration

Per reviewer finding: integration instructions referenced nonexistent
streaming_ctx. Replace with actual local state + integration test via
public endpoint. Cumulative dedup works end-to-end."
```

---

### Task A.3: Frontend test runner + executable B5 test (HIGH)

**Files:**
- Modify: `frontend/package.json` (add test runner deps)
- Create: `frontend/src/test-utils/mockSSE.ts`
- Create: `frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx`

**Root cause**: Plan 0 Task 6's frontend test references nonexistent `mockSSEResponse` helper; frontend has no test script in package.json; ChatPanel requires providers.

- [ ] **Step 1: Add test deps to `package.json`**

```json
{
  "devDependencies": {
    "@testing-library/react": "^14.0.0",
    "@testing-library/jest-dom": "^6.0.0",
    "vitest": "^1.0.0",
    "happy-dom": "^12.0.0"
  },
  "scripts": {
    "test": "vitest run"
  }
}
```

- [ ] **Step 2: Create `frontend/src/test-utils/mockSSE.ts`**

```typescript
export function mockSSEResponse(events: Array<Record<string, any>>) {
    // Returns a mock ReadableStream that emits the events as SSE chunks
    const encoder = new TextEncoder();
    return new ReadableStream({
        start(controller) {
            for (const event of events) {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
            }
            controller.close();
        },
    });
}
```

- [ ] **Step 3: Create ChatPanel integration test with all providers**

```typescript
import { render, waitFor, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { ChakraProvider } from "@chakra-ui/react";
import { ChatPanel } from "../ChatPanel";
import { mockSSEResponse } from "../../../test-utils/mockSSE";

describe("ChatPanel rollback (B5)", () => {
    it("token_rollback clears localSources, localImages, pendingSources, pendingImages, people_data", async () => {
        global.fetch = vi.fn(() =>
            Promise.resolve(new Response(mockSSEResponse([
                { type: "token", content: "Hello" },
                { type: "sources", sources: [{ doc_id: "A", chunk_id: "p.1" }] },
                { type: "images", images: [{ id: "img1" }] },
                { type: "people_data", people: [{ id: "p1" }] },
                { type: "potential_abbreviations", abbrs: ["BMNN"] },
                { type: "token_rollback" },
                { type: "complete", completion_status: "partial", answer: "" },
            ]), { status: 200, headers: { "Content-Type": "text/event-stream" } })
        );

        const queryClient = new QueryClient();
        render(
            <QueryClientProvider client={queryClient}>
                <MemoryRouter>
                    <ChakraProvider>
                        <ChatPanel />
                    </ChakraProvider>
                </MemoryRouter>
            </QueryClientProvider>
        );

        await waitFor(() => screen.getByText(/partial/i));
        // After rollback, no sources/images/people should be visible
        expect(screen.queryByText(/A, p\.1/)).toBeNull();
    });
});
```

- [ ] **Step 4: Add `useRAGChatStream.ts` rollback handler (if not already done in Phase 0)**

```typescript
case "token_rollback":
    setLocalSources([]);
    setLocalImages([]);
    setPendingSources([]);
    setPendingImages([]);
    setPeopleData(null);
    setPotentialAbbreviations([]);
    setTokenBuffer("");
```

- [ ] **Step 5: Run test**

Run: `cd frontend && pnpm test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/src/test-utils/ frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx frontend/src/hooks/useRAGChatStream.ts
git commit -m "fix(phase0-deferred): executable frontend B5 rollback test

Per reviewer: vitest + RTL added; mockSSE helper created; ChatPanel
integration test with all providers (QueryClient, MemoryRouter,
ChakraProvider). token_rollback clears all artifacts as specified."
```

---

### Task A.4: B5 persistence rollback via public endpoint integration test (HIGH)

**Files:**
- Modify: `backend/tests/agents/test_persistence_rollback_clears_all.py` (rewrite to use public endpoint)

**Root cause**: Plan 0 Task 7's `_persist_rollback` doesn't exist; rollback is inline closure in `_run_and_persist`.

- [ ] **Step 1: Rewrite test using public endpoint**

```python
# backend/tests/agents/test_persistence_rollback_clears_all.py
"""Integration test: rollback via /rag/chat/agent-lg/{ws}/stream SSE endpoint
clears all final fields in chat_messages row."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_rollback_clears_all_final_fields_in_persisted_row(test_db, test_user):
    """Drive token_rollback via SSE; verify persisted row cleared all fields."""
    client = TestClient(app)
    
    # Stream a query that triggers token_rollback (via fabricated answer)
    response = client.post(
        f"/rag/chat/agent-lg/{test_user.workspace_id}/stream",
        json={"message": "fabricate doc number"},  # known fabricator
    )
    
    # Verify response stream contains token_rollback event
    events = parse_sse(response.text)
    rollback_events = [e for e in events if e["type"] == "token_rollback"]
    assert len(rollback_events) >= 1
    
    # Query the persisted row
    last_message = test_db.query(ChatMessage).filter_by(
        session_id=test_user.session_id
    ).order_by(ChatMessage.created_at.desc()).first()
    
    assert last_message.text in (None, "")
    assert last_message.sources == []
    assert last_message.images == []
    assert last_message.potential_abbreviations == []
    assert last_message.people_data is None
```

- [ ] **Step 2: Run test**

Run: `cd backend && pytest tests/agents/test_persistence_rollback_clears_all.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/agents/test_persistence_rollback_clears_all.py
git commit -m "fix(phase0-deferred): B5 persistence rollback integration test

Per reviewer: test drives token_rollback via public SSE endpoint;
verifies persisted ChatMessage row has text=None, sources=[],
images=[], potential_abbreviations=[], people_data=None."
```

---

### Task A.5: B6 markdown fallback test with proper mocking (HIGH)

**Files:**
- Modify: `backend/tests/agents/test_markdown_fallback_acl.py` (proper mocking)

**Root cause**: Test doesn't setup `get_current_db()` context or mock heading search/storage.

- [ ] **Step 1: Rewrite test with full mocking**

```python
# backend/tests/agents/test_markdown_fallback_acl.py
from unittest.mock import patch, MagicMock
import pytest

from app.services.agents.rag_agent import _execute_search_section


async def test_markdown_fallback_workspace_predicate():
    """Doc in workspace A; user has workspace B only → markdown fallback returns 'not found'."""
    fake_db = MagicMock()
    
    # Mock heading search returning 0 results (forces fallback path)
    with patch("app.services.agent.tools.search_document_section",
               return_value={"sources": []}):
        # Mock Document query: simulate workspace mismatch
        with patch("app.services.agent.streaming.get_current_db", return_value=fake_db):
            fake_result = MagicMock()
            fake_result.scalar_one_or_none.return_value = None  # doc NOT in user's workspace
            fake_db.execute.return_value = fake_result
            
            # Mock storage (never called due to ACL fail)
            with patch("app.services.storage_service.get_storage_service") as mock_storage:
                result = await _execute_search_section(
                    section_reference="Chương II",
                    workspace_ids=["ws_b"],
                    document_ids=["doc_in_ws_a"],
                )
                # ACL fail → returns "Không tìm thấy" message
                assert "Không tìm thấy" in result["text"]
                mock_storage.return_value.download_markdown.assert_not_called()


async def test_markdown_fallback_allows_authorized_workspace():
    """Doc in workspace A; user has workspace A → returns content."""
    fake_db = MagicMock()
    fake_doc = MagicMock(markdown_s3_key="docs/test.md", updated_at=MagicMock(isoformat=lambda: "2026-09-08"))
    fake_result = MagicMock()
    fake_result.scalar_one_or_none.return_value = fake_doc
    fake_db.execute.return_value = fake_result
    
    with patch("app.services.agent.tools.search_document_section",
               return_value={"sources": []}):
        with patch("app.services.agent.streaming.get_current_db", return_value=fake_db):
            with patch("app.services.storage_service.get_storage_service") as mock_storage:
                mock_storage.return_value.download_markdown.return_value = "# Chương II\n\nContent here"
                with patch("app.services.agents.rag_agent._extract_section_from_markdown",
                           return_value="Content here"):
                    result = await _execute_search_section(
                        section_reference="Chương II",
                        workspace_ids=["ws_a"],
                        document_ids=["doc_in_ws_a"],
                    )
                    assert "Không tìm thấy" not in result["text"]
                    assert result["text"] == "Content here"
```

- [ ] **Step 2: Run test + commit**

```bash
cd backend && pytest tests/agents/test_markdown_fallback_acl.py -v
git add backend/tests/agents/test_markdown_fallback_acl.py
git commit -m "fix(phase0-deferred): B6 markdown fallback test with full mocking

Per reviewer: mock search_document_section (returns 0 results to force
fallback), mock get_current_db (returns MagicMock), mock storage
service. Verify workspace predicate prevents markdown download."
```

---

### Task A.6: Baseline metrics comparison with correct schema (HIGH)

**Files:**
- Modify: `backend/tests/reports/phase0_gate_report.md` (or the comparison script)

**Root cause**: Plan 0 Task 11 comparison script expects `metrics` schema that capture doesn't produce. Current eval artifacts use `meta`/`aggregate`.

- [ ] **Step 1: Verify actual schema**

Run: `cat backend/tests/reports/baseline_pre_task1.json | python -m json.tool | head -40`
Expected: Schema shows `meta`, `aggregate`, `cases` keys.

- [ ] **Step 2: Update comparison script to read actual schema**

```python
# In Phase 0 gate task Step 4
def extract_metric(artifact, metric_path):
    """Extract metric from actual schema: {'meta': {...}, 'aggregate': {...}, 'cases': [...]}."""
    parts = metric_path.split('.')
    cur = artifact
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def compare_baselines(pre_path, post_path):
    pre = json.loads(Path(pre_path).read_text())
    post = json.loads(Path(post_path).read_text())
    
    metrics = ['latency_p50', 'latency_p95', 'latency_p99', 'completion_rate',
               'refusal_rate_positive', 'refusal_rate_negative']
    
    diff = {}
    for m in metrics:
        pre_val = extract_metric(pre, f'aggregate.{m}') or extract_metric(pre, f'meta.{m}')
        post_val = extract_metric(post, f'aggregate.{m}') or extract_metric(post, f'meta.{m}')
        if pre_val is None or post_val is None:
            diff[m] = {'pre': pre_val, 'post': post_val, 'delta_pct': 'unavailable'}
        else:
            pct = (post_val - pre_val) / max(pre_val, 1e-9) * 100
            diff[m] = {'pre': pre_val, 'post': post_val, 'delta_pct': f'{pct:+.1f}%'}
    return diff
```

- [ ] **Step 3: Update gate report**

Replace Step 4's python snippet with above comparison function.

- [ ] **Step 4: Run + commit**

```bash
cd backend && python -c "from tests.reports.compare_baselines import compare_baselines; print(compare_baselines('tests/reports/baseline_pre_task1.json', 'tests/reports/baseline_post_task1_pre_sectionF.json'))"
git add backend/tests/reports/phase0_gate_report.md backend/tests/reports/compare_baselines.py
git commit -m "fix(phase0-deferred): baseline comparison reads actual schema

Per reviewer: capture produces meta/aggregate/cases schema, not metrics.
Update comparison to extract from aggregate.{metric} or meta.{metric}.
Output now contains actual measurement diffs."
```

---

# Part B — Plan 1A deferred fixes

### Task B.1: Fix migration syntax + async engine test (HIGH)

**Files:**
- Modify: `backend/tests/migrations/test_semantic_context_column_exists.py`

**Root cause**: Migration command `with engine.connect() as c:` after semicolon is invalid; engine is async.

- [ ] **Step 1: Rewrite as async migration test**

```python
# backend/tests/migrations/test_semantic_context_column_exists.py
import pytest
from sqlalchemy import text

@pytest.mark.asyncio
async def test_semantic_context_column_exists():
    """Column exists after lifespan migration."""
    from app.main import lifespan_app
    from app.core.database import async_engine
    
    async with lifespan_app():
        async with async_engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='chat_messages' AND column_name='semantic_context'"
            ))
            assert result.fetchone() is not None
```

- [ ] **Step 2: Run test + commit**

```bash
cd backend && pytest tests/migrations/test_semantic_context_column_exists.py -v
git add backend/tests/migrations/test_semantic_context_column_exists.py
git commit -m "fix(phase1a-deferred): async migration test

Per reviewer: migration command was invalid Python (with after
semicolon); engine is async. Use async test + lifespan context."
```

---

### Task B.2: Move Task 2.5 AFTER Task 6 in checklist order (HIGH)

**Files:**
- Modify: `backend/app/services/agents/semantic_preprocessor.py` (Task 6 step 5 imports `to_persisted_dict`)
- Modify: Plan 1A (renumber Task 2.5 to Task 7.5)

**Root cause**: Task 2.5 explicitly depends on Task 6 but is placed before Tasks 3-6 in checklist order.

- [ ] **Step 1: Renumber Task 2.5 → Task 7.5 in Plan 1A**

Edit Plan 1A file: rename `### Task 2.5` to `### Task 7.5` and move the section after Task 7.

- [ ] **Step 2: Update Task 7 step ordering**

Task 7 step 1 reference updated from `Task 2.5` to `Task 7.5`.

- [ ] **Step 3: Update Step 3 of Task 6 to clarify**

Add note: "Step 5 (Task 7.5) depends on this Task 6 — persistence write happens AFTER to_persisted_dict exists."

- [ ] **Step 4: Commit (commit plan file)**

```bash
git add backend/docs/superpowers/plans/2026-09-08-deepagent-phase1a-preprocessor.md
git commit -m "fix(phase1a-deferred): reorder Task 2.5 to Task 7.5

Per reviewer: Task 2.5 depended on Task 6 serializers but was placed
before Tasks 3-6. Renumber to Task 7.5; place after Task 7."
```

---

### Task B.3: Complete `contracts_validation.py` — dependency IDs + tool allowlist (HIGH)

**Files:**
- Modify: `backend/app/services/agents/contracts_validation.py`
- Modify: `backend/tests/agents/test_contracts_validation.py`

**Root cause**: Current validator omits dependency ID existence check + tool allowlist subset check.

- [ ] **Step 1: Add dependency ID validation**

```python
def validate_task_plan(tasks: list, semantic_context, runtime_context) -> None:
    # ... existing checks ...
    
    # 4. NEW: Every dependency must reference an existing task_id
    task_ids = {t.task_id for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            if dep not in task_ids:
                raise ValueError(f"task {t.task_id} depends on unknown task_id: {dep}")
    
    # 5. NEW: Every allowed_tool must be in RuntimeContext.tool_allowlist
    for t in tasks:
        for tool in t.allowed_tools:
            if tool not in runtime_context.tool_allowlist:
                raise ValueError(f"task {t.task_id} uses tool {tool} not in tool_allowlist")
```

- [ ] **Step 2: Add tests for new validations**

```python
# Add to test_contracts_validation.py
def test_validate_task_plan_rejects_unknown_dep():
    tasks = [
        TaskSpec(task_id="t1", work_type="retrieve_section", depends_on=["t999"],
                 completion_criteria={}),
    ]
    with pytest.raises(ValueError, match="unknown task_id"):
        validate_task_plan(tasks, semantic_context, runtime_context_with_allowlist=["retrieve_section"])


def test_validate_task_plan_rejects_tool_outside_allowlist():
    tasks = [
        TaskSpec(task_id="t1", work_type="retrieve_section",
                 allowed_tools=["delete_files"], completion_criteria={}),
    ]
    with pytest.raises(ValueError, match="not in tool_allowlist"):
        validate_task_plan(tasks, semantic_context, runtime_context_with_allowlist=["retrieve_section"])
```

- [ ] **Step 3: Run + commit**

```bash
cd backend && pytest tests/agents/test_contracts_validation.py -v
git add backend/app/services/agents/contracts_validation.py backend/tests/agents/test_contracts_validation.py
git commit -m "fix(phase1a-deferred): contracts_validation completeness

Per reviewer: validator omitted dependency ID existence check + tool
allowlist subset check. Add both; rejected broken plans."
```

---

### Task B.4: Graph refactor preserves real nodes + flag-on test verifies query_analyzer absent (HIGH)

**Files:**
- Modify: `backend/app/services/agents/supervisor.py` (actual graph builder)
- Modify: `backend/tests/agents/test_graph_atomic_flag.py`

**Root cause**: Plan 1A Task 11 uses ellipses; doesn't preserve real nodes/edges. Flag-on test only checks semantic_preprocessor exists, not query_analyzer absent.

- [ ] **Step 1: Read current `create_supervisor_graph`**

Read `backend/app/services/agents/supervisor.py:3512-3650`.

- [ ] **Step 2: Implement `_build_new_graph` preserving all real nodes**

```python
def _build_new_graph() -> StateGraph:
    """New graph per Q2: removes query_analyzer; keeps all other nodes/edges."""
    workflow = StateGraph(SupervisorState)
    
    # REMOVED: workflow.add_node("query_analyzer", query_analyzer_node)
    # ADDED: workflow.add_node("semantic_preprocessor", semantic_preprocessor_node)
    
    # All other nodes preserved from legacy graph:
    workflow.add_node("semantic_preprocessor", semantic_preprocessor_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("resolve_doc", resolve_doc_node)
    workflow.add_node("write", write_node)
    workflow.add_node("people", people_node)
    workflow.add_node("direct", direct_node)
    workflow.add_node("finish", finish_node)
    if settings.NEXUSRAG_DEEP_ENABLED:
        workflow.add_node("deep_research_coordinator", deep_research_coordinator_node)
    if settings.NEXUSRAG_COMPLEXITY_ACTIVE:  # probe node added when complexity active
        workflow.add_node("metadata_probe", metadata_probe_node)
    
    # Edges: START → semantic_preprocessor → supervisor (no query_analyzer)
    workflow.add_edge(START, "semantic_preprocessor")
    workflow.add_edge("semantic_preprocessor", "supervisor")
    
    # Conditional edges from supervisor (SAME as legacy):
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "rag": "rag",
            "resolve_doc": "resolve_doc",
            "write": "write",
            "people": "people",
            "direct": "direct",
            "finish": END,
            # Phase 1B+ additions:
            "deep_research_coordinator": "deep_research_coordinator",
            "metadata_probe": "metadata_probe",
            "clarification": "clarification_node",
        },
    )
    
    return workflow.compile()
```

- [ ] **Step 3: Update flag-on test to verify query_analyzer absent**

```python
def test_flag_true_uses_new_graph():
    with patch("app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR", True):
        graph = create_supervisor_graph()
        nodes = graph.nodes  # StateGraph exposes compiled nodes
        assert "semantic_preprocessor" in nodes
        assert "query_analyzer" NOT in nodes, "query_analyzer must be removed in new graph per Q2"
        # Verify all other preserved nodes exist
        for n in ["supervisor", "rag", "resolve_doc", "write", "people", "direct", "finish"]:
            assert n in nodes, f"{n} missing from new graph"


def test_flag_false_uses_legacy_graph():
    with patch("app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR", False):
        graph = create_supervisor_graph()
        nodes = graph.nodes
        assert "query_analyzer" in nodes, "legacy graph must keep query_analyzer"
        assert "semantic_preprocessor" not in nodes
```

- [ ] **Step 4: Run + commit**

```bash
cd backend && pytest tests/agents/test_graph_atomic_flag.py -v
git add backend/app/services/agents/supervisor.py backend/tests/agents/test_graph_atomic_flag.py
git commit -m "fix(phase1a-deferred): graph refactor preserves real nodes

Per reviewer: new graph must keep all other nodes (rag, resolve_doc,
write, people, direct, finish) and conditional edges. Flag-on test now
asserts query_analyzer NOT in nodes (per Q2); flag-off keeps it."
```

---

# Part C — Plan 1B deferred fixes

### Task C.1: 7-day rotation logic (not daily) (HIGH)

**Files:**
- Modify: `backend/app/services/observability/shadow_log.py`

**Root cause**: Current rotation uses `file_mtime_date < date.today()` (daily); spec requires 7-day.

- [ ] **Step 1: Replace with true 7-day rotation**

```python
# In _log_routing_shadow, replace rotation logic:
from datetime import date, timedelta

if log_path.exists():
    file_mtime_date = date.fromtimestamp(log_path.stat().st_mtime)
    age_days = (date.today() - file_mtime_date).days
    if age_days >= 7:
        # Rotate: archive with date suffix; collision-safe (counter if exists)
        base_archive = log_path.with_suffix(f".{file_mtime_date.isoformat()}.jsonl")
        archive = base_archive
        counter = 1
        while archive.exists():
            archive = log_path.with_suffix(f".{file_mtime_date.isoformat()}.{counter}.jsonl")
            counter += 1
        os.rename(log_path, archive)
```

- [ ] **Step 2: Test rotation triggers after 7 days**

```python
def test_rotation_triggers_at_7_days(tmp_path):
    log = tmp_path / "shadow.jsonl"
    log.write_text('{"event": "old"}\n')
    # Set mtime to 8 days ago
    eight_days_ago = time.time() - 8 * 86400
    os.utime(log, (eight_days_ago, eight_days_ago))
    
    # Run rotation
    asyncio.run(_log_routing_shadow({...}))
    
    # Old file should be archived; new file fresh
    archives = list(tmp_path.glob("shadow.*.jsonl"))
    assert len(archives) == 1
    assert log.exists() and log.read_text() != '{"event": "old"}\n'


def test_rotation_does_not_trigger_under_7_days(tmp_path):
    log = tmp_path / "shadow.jsonl"
    log.write_text('{"event": "recent"}\n')
    # Set mtime to 3 days ago
    three_days_ago = time.time() - 3 * 86400
    os.utime(log, (three_days_ago, three_days_ago))
    
    asyncio.run(_log_routing_shadow({...}))
    
    # No archive; same file (appended)
    archives = list(tmp_path.glob("shadow.*.jsonl"))
    assert len(archives) == 0
```

- [ ] **Step 3: Process-safe locking note**

Add comment: `asyncio.Lock` is process-local. For multi-worker backends, use Redis lock OR file-based lock (fcntl). Deferred to O45 (DB-backed settings + multi-worker pub/sub).

- [ ] **Step 4: Commit**

```bash
cd backend && pytest tests/observability/test_shadow_log_redaction.py -v
git add backend/app/services/observability/shadow_log.py backend/tests/observability/test_shadow_log_redaction.py
git commit -m "fix(phase1b-deferred): 7-day log rotation (not daily)

Per reviewer: threshold was file_mtime_date < date.today() (daily);
spec requires 7-day. Now age_days >= 7 triggers rotation.
Process-safe locking deferred to O45 (multi-worker pub/sub)."
```

---

### Task C.2: C.7 gates with executable evidence (HIGH)

**Files:**
- Modify: Plan 1B gate report task Step 5

**Root cause**: Current C.7 gates are prose only; no executable commands produce the metrics.

- [ ] **Step 1: Replace gate report Step 5 with executable checks**

Replace the gate report in Plan 1B with:

```markdown
## Gates (executable evidence required)
| Gate | Threshold | Command | Expected output |
|------|-----------|---------|-----------------|
| JSON validity rate | ≥99% | `python scripts/eval_json_validity.py --dataset tests/prompts/datasets/routing_golden.yaml` | `{validity_rate: 0.99}` |
| Invalid JSON fallback | 100% | (same script + invalid-json subset) | `{fallback_rate: 1.0}` |
| Recall complex | ≥95% | `python scripts/eval_recall.py --arm new --dataset tests/prompts/datasets/routing_golden.yaml[:40]` | `{recall: 0.95}` |
| Simple → deep | ≤5% | (same) | `{misroute_rate: 0.05}` |
| Clarify precision/recall | reported | `python scripts/eval_clarify.py` | report file |
| Cross-workspace leak | 0 | `pytest tests/agents/test_cross_workspace_negative.py` | 0 leaks |
| Late events | 0 | `pytest tests/agents/test_late_events_negative.py` | 0 late |
| Shadow log rotation | 7-day | `pytest tests/observability/test_shadow_log_rotation.py` | rotation triggered at age>=7 |
| Pre-existing tests pass | all | `pytest tests/` | all pass |
```

Each threshold command produces a JSON report. The gate review cites the report file path + verdict (PASS/FAIL).
```

- [ ] **Step 2: Commit plan update**

```bash
git add backend/docs/superpowers/plans/2026-09-08-deepagent-phase1b-router.md
git commit -m "fix(phase1b-deferred): C.7 gates with executable evidence

Per reviewer: gates were prose only. Replace with concrete commands
that produce JSON reports; gate review cites report file + verdict."
```

---

# Part D — Plan 2 deferred fixes

### Task D.1: Deep flag admission — AgentType.DEEPAGENT + FlagSnapshot + graph edge (CRITICAL)

**Files:**
- Modify: `backend/app/services/agents/models.py` (add `DEEPAGENT` to `AgentType`)
- Modify: `backend/app/core/config.py` (add deep bundle flags)
- Modify: `backend/app/services/agents/supervisor.py` (add deep_research_coordinator node + edge)
- Modify: `backend/app/services/agents/complexity.py` (FlagSnapshot helper)
- Modify: `backend/tests/agents/test_deep_flags_admission.py` (NEW)

**Root cause**: Phase 2 has NO task adding deep flags, FlagSnapshot, AgentType.DEEPAGENT, or graph edge. Required atomic Phase-2 admission per spec D.3 + E.2.

- [ ] **Step 1: Add `DEEPAGENT` to `AgentType`**

```python
# backend/app/services/agents/models.py
class AgentType:
    RAG = "rag"
    WRITE = "write"
    DIRECT = "direct"
    PEOPLE = "people"
    FINISH = "finish"
    ANSWER_GENERATOR = "answer_generator"
    RESOLVE_DOC = "resolve_doc"
    DEEPAGENT = "deepagent"  # NEW (per spec D.3 + review finding)
```

- [ ] **Step 2: Add deep bundle flags to config**

```python
# backend/app/core/config.py (add to Settings)
NEXUSRAG_DEEP_ENABLED: bool = False
NEXUSRAG_DEEP_SHADOW: bool = False
NEXUSRAG_AGENT_DEADLINE_SECONDS: int = 28
NEXUSRAG_DEEP_MAX_PARALLEL: int = 2
NEXUSRAG_DEEP_MAX_DOMAIN_CALLS: int = 6

@model_validator(mode="after")
def _validate_deep_flag_chain(self):
    # DEEP_SHADOW/ENABLED require COMPLEXITY_ACTIVE
    if self.NEXUSRAG_DEEP_ENABLED and not self.NEXUSRAG_COMPLEXITY_ACTIVE:
        raise ValueError("DEEP_ENABLED requires COMPLEXITY_ACTIVE")
    if self.NEXUSRAG_DEEP_SHADOW and self.NEXUSRAG_DEEP_ENABLED:
        raise ValueError("DEEP_SHADOW and DEEP_ENABLED are mutually exclusive")
    # DEEP_SHADOW requires COMPLEXITY_ACTIVE
    if self.NEXUSRAG_DEEP_SHADOW and not self.NEXUSRAG_COMPLEXITY_ACTIVE:
        raise ValueError("DEEP_SHADOW requires COMPLEXITY_ACTIVE")
    return self
```

- [ ] **Step 3: Add `FlagSnapshot` helper in `complexity.py`**

```python
class FlagSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    semantic_preprocessor: bool
    complexity_shadow: bool
    complexity_active: bool
    deep_enabled: bool
    deep_shadow: bool
    deadline_seconds: int
    max_parallel: int
    max_domain_calls: int
    captured_at: float
    process_pid: int

def create_flag_snapshot(settings: Settings) -> FlagSnapshot:
    return FlagSnapshot(
        semantic_preprocessor=settings.NEXUSRAG_SEMANTIC_PREPROCESSOR,
        complexity_shadow=settings.NEXUSRAG_COMPLEXITY_SHADOW,
        complexity_active=settings.NEXUSRAG_COMPLEXITY_ACTIVE,
        deep_enabled=settings.NEXUSRAG_DEEP_ENABLED,
        deep_shadow=settings.NEXUSRAG_DEEP_SHADOW,
        deadline_seconds=settings.NEXUSRAG_AGENT_DEADLINE_SECONDS,
        max_parallel=settings.NEXUSRAG_DEEP_MAX_PARALLEL,
        max_domain_calls=settings.NEXUSRAG_DEEP_MAX_DOMAIN_CALLS,
        captured_at=time.time(),
        process_pid=os.getpid(),
    )
```

- [ ] **Step 4: Update `SupervisorState` with deep fields**

```python
class SupervisorState(TypedDict, total=False):
    # ... existing fields
    flag_snapshot: FlagSnapshot   # NEW
```

- [ ] **Step 5: Add deep_research_coordinator node + edge**

In `supervisor.py`:

```python
def _build_deep_enabled_graph(workflow):
    """Add deep_research_coordinator node when NEXUSRAG_DEEP_ENABLED."""
    workflow.add_node("deep_research_coordinator", deep_research_coordinator_node)
    workflow.add_conditional_edges("supervisor", _route_after_supervisor, {
        # ... existing keys
        "deep_research_coordinator": "deep_research_coordinator",
    })
    workflow.add_edge("deep_research_coordinator", END)
```

Update `create_supervisor_graph`:

```python
def create_supervisor_graph():
    if settings.NEXUSRAG_SEMANTIC_PREPROCESSOR:
        graph = _build_new_graph()
        if settings.NEXUSRAG_DEEP_ENABLED:
            graph = _add_deep_coordinator(graph)
        return graph
    return _build_legacy_graph()
```

- [ ] **Step 6: Update `_route_after_supervisor` for deep route**

```python
def _route_after_supervisor(state):
    decision = state.get("complexity_route")
    if decision:
        if decision.execution_mode == "deepagent":
            return "deep_research_coordinator"
        if decision.execution_mode == "clarify":
            return "clarification"
        if decision.needs_document_probe:
            return "metadata_probe"
    # legacy priority (existing logic)
    ...
```

- [ ] **Step 7: Write test for atomic Phase 2 admission**

```python
# backend/tests/agents/test_deep_flags_admission.py
def test_deep_enabled_flag_requires_complexity_active():
    with pytest.raises(ValidationError, match="DEEP_ENABLED requires COMPLEXITY_ACTIVE"):
        Settings(NEXUSRAG_SEMANTIC_PREPROCESSOR=True, NEXUSRAG_COMPLEXITY_ACTIVE=False,
                 NEXUSRAG_DEEP_ENABLED=True)

def test_deep_shadow_and_enabled_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        Settings(NEXUSRAG_SEMANTIC_PREPROCESSOR=True, NEXUSRAG_COMPLEXITY_ACTIVE=True,
                 NEXUSRAG_DEEP_ENABLED=True, NEXUSRAG_DEEP_SHADOW=True)

def test_agent_type_deepagent_exists():
    from app.services.agents.models import AgentType
    assert AgentType.DEEPAGENT == "deepagent"

def test_deep_enabled_graph_has_coordinator_node():
    with patch("app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR", True), \
         patch("app.core.config.settings.NEXUSRAG_COMPLEXITY_ACTIVE", True), \
         patch("app.core.config.settings.NEXUSRAG_DEEP_ENABLED", True):
        graph = create_supervisor_graph()
        assert "deep_research_coordinator" in graph.nodes

def test_route_to_deep_when_complexity_says_deepagent():
    state = make_state(complexity_route=RoutingDecision(execution_mode="deepagent", ...))
    assert _route_after_supervisor(state) == "deep_research_coordinator"
```

- [ ] **Step 8: Run + commit**

```bash
cd backend && pytest tests/agents/test_deep_flags_admission.py -v
git add backend/app/services/agents/models.py backend/app/core/config.py backend/app/services/agents/complexity.py backend/app/services/agents/supervisor.py backend/tests/agents/test_deep_flags_admission.py
git commit -m "fix(phase2-deferred): Deep flag admission atomic Phase-2 (CRITICAL)

Per reviewer: Phase 2 had NO task adding DEEPAGENT to AgentType, deep
bundle flags, FlagSnapshot, deep_research_coordinator node, graph edge.
All added atomically. Test asserts dependency chain + atomic admission."
```

---

### Task D.2: External citation projection — Evidence → ChatSourceChunk (CRITICAL)

**Files:**
- Modify: `backend/app/services/agents/deep_research/evidence.py` (add projection)
- Create: `backend/tests/agents/deep_research/test_evidence_projection.py` (NEW)

**Root cause**: Spec `project_external_citation` accepts internal_id and returns int; Plan 2 uses Evidence and places integer results in `sources` where ChatSourceChunk is required.

- [ ] **Step 1: Implement `evidence_to_source_chunk` projection**

```python
# backend/app/services/agents/deep_research/evidence.py (add)
from app.schemas.rag import ChatSourceChunk


def evidence_to_source_chunk(evidence: Evidence) -> ChatSourceChunk:
    """Convert Evidence → ChatSourceChunk for SSE terminal sources."""
    return ChatSourceChunk(
        index=0,  # assigned externally via project_external_citation
        document_id=str(evidence.document_id) if evidence.document_id else None,
        content=evidence.raw_content[:500] if evidence.raw_content else "",  # truncate for terminal event
        page_or_chunk=evidence.page_or_chunk or "",
        section_path=evidence.section_path or "",
        citation_number=evidence.citation_number,
        citation_article=evidence.citation_article,
        similarity_score=None,
    )


def project_external_citations(
    evidence_list: list[Evidence],
) -> list[ChatSourceChunk]:
    """Project + assign stable indices (sorted by doc_id + section_path + page)."""
    sorted_evidence = sorted(
        evidence_list,
        key=lambda e: (str(e.document_id or ""), e.section_path or "", e.page_or_chunk or ""),
    )
    return [
        evidence_to_source_chunk(e).model_copy(update={"index": i})
        for i, e in enumerate(sorted_evidence)
    ]
```

- [ ] **Step 2: Update Plan 2 Task 7 to use this projection**

Replace `project_external_citation(e, state["evidence_registry"])` (line ~624) with `project_external_citations(state["evidence_registry"].all())`.

- [ ] **Step 3: Write test**

```python
# backend/tests/agents/deep_research/test_evidence_projection.py
def test_evidence_projects_to_chat_source_chunk():
    from app.services.agents.deep_research.evidence import project_external_citations
    from app.services.agents.models import Evidence, Provenance
    
    evidence_list = [
        Evidence(evidence_id="e1", task_id="t1", source_id="s1",
                 raw_content="content A", content_hash="h1", content_size_bytes=10,
                 raw_content_bytes=10, document_id=UUID4("11111111-1111-1111-1111-111111111111"),
                 section_path="Chương I", page_or_chunk="p.1",
                 provenance=Provenance(fetcher="deep_worker", fetched_at=0, fetched_by=UUID4("22222222-2222-2222-2222-222222222222"), workspace_scope=[], acl_checked=True, run_id="r1")),
        Evidence(evidence_id="e2", task_id="t2", source_id="s2",
                 raw_content="content B", content_hash="h2", content_size_bytes=10,
                 raw_content_bytes=10, document_id=UUID4("33333333-3333-3333-3333-333333333333"),
                 section_path="Chương II", page_or_chunk="p.2",
                 provenance=Provenance(fetcher="deep_worker", fetched_at=0, fetched_by=UUID4("22222222-2222-2222-2222-222222222222"), workspace_scope=[], acl_checked=True, run_id="r1")),
    ]
    
    chunks = project_external_citations(evidence_list)
    assert len(chunks) == 2
    assert chunks[0].index == 0
    assert chunks[1].index == 1
    # Sorted by document_id (deterministic)
    assert chunks[0].document_id < chunks[1].document_id


def test_projection_dedups_same_content_from_two_sources():
    """Same content_hash from different sources → 2 entries (provenance preserved)."""
    from app.services.agents.deep_research.evidence import project_external_citations
    from app.services.agents.models import Evidence, Provenance
    
    evidence_list = [
        Evidence(evidence_id="e1", task_id="t1", source_id="s1",
                 raw_content="same", content_hash="same_hash", content_size_bytes=4,
                 raw_content_bytes=4, document_id=UUID4("11111111-1111-1111-1111-1111-111111111111"),
                 section_path="Same", page_or_chunk="p.1",
                 provenance=Provenance(fetcher="deep_worker", fetched_at=0, fetched_by=UUID4("22222222-2222-2222-2222-222222222222"), workspace_scope=[], acl_checked=True, run_id="r1")),
        Evidence(evidence_id="e2", task_id="t2", source_id="s2",
                 raw_content="same", content_hash="same_hash", content_size_bytes=4,
                 raw_content_bytes=4, document_id=UUID4("11111111-1111-1111-1111-1111-111111111111"),
                 section_path="Same", page_or_chunk="p.1",
                 provenance=Provenance(fetcher="deep_worker", fetched_at=0, fetched_by=UUID4("22222222-2222-2222-2222-222222222222"), workspace_scope=[], acl_checked=True, run_id="r1")),
    ]
    
    chunks = project_external_citations(evidence_list)
    assert len(chunks) == 2  # provenance preserved
```

- [ ] **Step 4: Run + commit**

```bash
cd backend && pytest tests/agents/deep_research/test_evidence_projection.py -v
git add backend/app/services/agents/deep_research/evidence.py backend/tests/agents/deep_research/test_evidence_projection.py
git commit -m "fix(phase2-deferred): Evidence → ChatSourceChunk projection

Per reviewer: spec signature accepts internal_id returns int; plan
used Evidence + int in ChatSourceChunk[]. Add evidence_to_source_chunk
+ project_external_citations(evidence_list) → list[ChatSourceChunk]
with deterministic sorted indices. Multi-source provenance preserved."
```

---

### Task D.3: Fix `ab_deep_eval.py` — main guard, parser, message field (HIGH)

**Files:**
- Modify: `backend/scripts/ab_deep_eval.py`

**Root cause**: Multiple issues — no main guard, env unused, parser discards SSE event: type, query vs message, no actual A/B selection.

- [ ] **Step 1: Add main guard + restore proper parser**

```python
# In ab_deep_eval.py (after run_arm etc.)

# Restore main guard
if __name__ == "__main__":
    main()


def parse_sse_events(text: str) -> list[dict]:
    """Parse SSE text with event: prefix into events list."""
    events = []
    current_event_type = None
    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event_type = line[7:].strip()
        elif line.startswith("data: "):
            try:
                import json
                payload = json.loads(line[6:])
                if isinstance(payload, dict):
                    payload["type"] = current_event_type or payload.get("type", "unknown")
                    events.append(payload)
            except json.JSONDecodeError:
                pass
            current_event_type = None
    return events


def build_chat_request(case: dict) -> dict:
    """Build ChatRequest dict per backend schema."""
    return {
        "message": case["query"],
        "workspace_id": case.get("workspace_id", ""),
        "session_id": case.get("session_id", ""),
    }


async def run_case(case: dict, arm: str, workspace_id: str, base_url: str) -> dict:
    """Drive endpoint; uses 'message' field per ChatRequest schema."""
    # Server-side arm selection: caller passes arm via different NEXUSRAG_DEEP_ENABLED
    # (already running with the right env); no X-Arm header needed.
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/rag/chat/agent-lg/{workspace_id}/stream",
            json=build_chat_request(case),
        )
    ...
```

- [ ] **Step 2: Pass workspace_id via env or flag (not unused `env`)**

```python
# Replace env unused variable
async def run_case(case: dict, arm: str, workspace_id: str, base_url: str) -> dict:
    """Arm determined by caller (env NEXUSRAG_DEEP_ENABLED=true|false).
    Workspace_id passed explicitly in URL."""
    ...
```

- [ ] **Step 3: Add tests for parser + metrics**

```python
# backend/tests/scripts/test_ab_deep_eval_helpers.py
def test_parse_sse_events_with_event_prefix():
    text = """event: status
data: {"task_id": "t1"}

event: token
data: {"text": "hello"}

event: complete
data: {"answer": "world"}
"""
    events = parse_sse_events(text)
    assert len(events) == 3
    assert events[0]["type"] == "status"
    assert events[1]["type"] == "token"
    assert events[2]["type"] == "complete"


def test_build_chat_request_uses_message_field():
    case = {"query": "test", "workspace_id": "ws_a"}
    req = build_chat_request(case)
    assert "message" in req
    assert req["message"] == "test"
```

- [ ] **Step 4: Commit**

```bash
cd backend && pytest tests/scripts/test_ab_deep_eval_helpers.py -v
git add backend/scripts/ab_deep_eval.py backend/tests/scripts/test_ab_deep_eval_helpers.py
git commit -m "fix(phase2-deferred): ab_deep_eval.py main guard + parser + message field

Per reviewer: no main guard restored; parser discarded 'event:' prefix;
query vs message schema mismatch; env unused. All fixed."
```

---

### Task D.4: Truncation test fixture + UTF-8 byte-safe (HIGH)

**Files:**
- Modify: `backend/tests/agent/test_document_accessor.py` (Task 4)

**Root cause**: Test's `if content.document_version` is always truthy for found doc; doesn't prove source exceeded cap. UTF-8 replacement expansion from truncation not protected.

- [ ] **Step 1: Rewrite truncation test with explicit fixture**

```python
# In test_document_accessor.py
async def test_read_full_section_truncates_at_cap_with_oversized_source(test_db, oversized_doc):
    """Source content > MAX_RAW_CONTENT_BYTES; DocumentAccessor MUST truncate."""
    content = await DocumentAccessor.read_full_section(
        document_id=oversized_doc.id, section_reference="Chương II",
        principal_id=oversized_doc.owner_id, allowed_workspace_ids=oversized_doc.workspace_ids,
        session=test_db,
    )
    # Compliant truncation
    assert len(content.text.encode("utf-8")) <= Evidence.MAX_RAW_CONTENT_BYTES
    # Source exceeded cap → is_truncated MUST be True
    assert content.is_truncated is True


@pytest.fixture
async def oversized_doc(test_db):
    """Doc whose section content exceeds MAX_RAW_CONTENT_BYTES."""
    # Construct markdown larger than cap
    huge_section = "Điều " + ("x" * Evidence.MAX_RAW_CONTENT_BYTES)  # Vietnamese + ASCII
    markdown = f"# Chương II\n\n{huge_section}\n"
    
    doc = await create_document(test_db, markdown=markdown)
    return doc


async def test_truncation_preserves_utf8_boundary():
    """Truncation at MAX_RAW_CONTENT_BYTES MUST NOT break UTF-8 chars."""
    # Construct content with multi-byte chars near truncation boundary
    # Each Vietnamese char is 2-3 bytes in UTF-8
    multibyte_section = "Điều " * (Evidence.MAX_RAW_CONTENT_BYTES // 6)  # 'Điều ' is ~6 bytes
    
    # ... create doc with this content ...
    
    content = await DocumentAccessor.read_full_section(...)
    
    # No Unicode replacement chars
    assert "\ufffd" not in content.text
    # Decoded cleanly
    content.text.encode("utf-8")  # should not raise
```

- [ ] **Step 2: Verify `DocumentAccessor` uses UTF-8-safe truncation**

Check `backend/app/services/agent/document_accessor.py` (Plan 2 Task 4). Truncation should use:

```python
text_bytes = text.encode('utf-8')
if len(text_bytes) > Evidence.MAX_RAW_CONTENT_BYTES:
    # Truncate at byte boundary, then decode safely
    truncated_bytes = text_bytes[:Evidence.MAX_RAW_CONTENT_BYTES]
    # Find last complete UTF-8 char boundary (backtrack if mid-char)
    while truncated_bytes and (truncated_bytes[-1] & 0xC0) == 0x80:
        truncated_bytes = truncated_bytes[:-1]
    text = truncated_bytes.decode('utf-8', errors='replace')
    is_truncated = True
```

- [ ] **Step 3: Commit**

```bash
cd backend && pytest tests/agent/test_document_accessor.py -v
git add backend/tests/agent/test_document_accessor.py backend/app/services/agent/document_accessor.py
git commit -m "fix(phase2-deferred): truncation test fixture + UTF-8 byte-safe

Per reviewer: 'if content.document_version' always truthy for found doc;
UTF-8 replacement expansion not protected. Add oversized fixture;
verify is_truncated + byte size; backtrack at UTF-8 char boundary."
```

---

### Task D.5: Fix ChatPanel path (HIGH)

**Files:**
- Modify: `backend/docs/superpowers/plans/2026-09-08-deepagent-phase2-pilot.md` (Task 14 path)
- Modify: Phase 0 Task 6 cross-reference (consistency)

**Root cause**: Plan 2 Task 14 uses `"frontend/src/components/ChatPanel.tsx"` (nonexistent); actual path is `frontend/src/components/rag/ChatPanel.tsx`.

- [ ] **Step 1: Update Plan 2 Task 14 path**

Edit Plan 2 file Task 14 commit:

```bash
git add frontend/src/hooks/useRAGChatStream.ts frontend/src/components/rag/ChatPanel.tsx frontend/src/hooks/useRAGChatStream.test.ts
```

Change to:

```bash
git add frontend/src/hooks/useRAGChatStream.ts frontend/src/components/rag/ChatPanel.tsx frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx
```

Note: Plan 0 Task 6 already created the integration test at this path; Plan 2 Task 14 only modifies the hook + ChatPanel component.

- [ ] **Step 2: Add cross-reference in Plan 2 Task 14**

Add at top of Task 14:

```markdown
**Files** (continuation from Plan 0 Task 6):
- Modify: `frontend/src/components/rag/ChatPanel.tsx` (UI rendering of completion_status banners)
- (Plan 0 Task 6 already created the integration test at `frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx`)
```

- [ ] **Step 3: Commit plan update**

```bash
git add backend/docs/superpowers/plans/2026-09-08-deepagent-phase2-pilot.md
git commit -m "fix(phase2-deferred): ChatPanel path correction + Phase 0 cross-ref

Per reviewer: actual path is frontend/src/components/rag/ChatPanel.tsx
(not frontend/src/components/ChatPanel.tsx). Plan 0 Task 6 owns the
integration test; Plan 2 Task 14 only modifies hook + component."
```

---

### Task D.6: Cohort DI completion (HIGH)

**Files:**
- Modify: `backend/app/services/cohorts.py`
- Modify: `backend/app/api/admin_cohorts.py`
- Modify: `backend/app/main.py` (register router)
- Modify: `backend/tests/services/test_cohorts.py`

**Root cause**: Plan 2 Task 10 calls undefined `get_user`, `CohortAudit`, `db`.

- [ ] **Step 1: Define `get_user` + import `CohortAudit` properly**

```python
# backend/app/services/cohorts.py
from app.models.user import User
from app.models.cohort_audit import CohortAudit
from app.core.database import async_session_factory

EXPERIMENT_SALT = "deep_canary_v1_salt_2026_09_08"


async def get_user(user_id: UUID) -> User | None:
    """Fetch user by ID from current session."""
    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
```

- [ ] **Step 2: Fix `is_user_in_experiment` signature**

```python
async def is_user_in_experiment(user_id: UUID, experiment_name: str, percent: int) -> bool:
    """Stable hash-based allocation."""
    user = await get_user(user_id)
    if user is None:
        return False
    if user.cohort_id == "force_in": return True
    if user.cohort_id == "force_out": return False
    if user.is_superadmin or (user.cohort_id and user.cohort_id.startswith("internal_")):
        return True
    h = hashlib.sha256(f"{user_id}:{EXPERIMENT_SALT}".encode()).digest()
    bucket = int.from_bytes(h[:4], "big") % 100
    return bucket < percent
```

- [ ] **Step 3: Register admin router in main.py**

```python
# backend/app/main.py
from app.api.admin_cohorts import router as admin_cohorts_router

app.include_router(admin_cohorts_router, prefix="/api/v1/admin")
```

- [ ] **Step 4: Update test to use proper fixtures**

```python
# backend/tests/services/test_cohorts.py
async def test_internal_user_always_in(test_db, internal_user):
    assert await is_user_in_experiment(internal_user.id, "deep_canary", percent=50)

async def test_deterministic_allocation(test_db, regular_user):
    result1 = await is_user_in_experiment(regular_user.id, "deep_canary", 50)
    result2 = await is_user_in_experiment(regular_user.id, "deep_canary", 50)
    assert result1 == result2

async def test_nonexistent_user_returns_false(test_db):
    fake_user_id = UUID4("00000000-0000-0000-0000-000000000000")
    assert await is_user_in_experiment(fake_user_id, "deep_canary", 50) is False
```

- [ ] **Step 5: Commit**

```bash
cd backend && pytest tests/services/test_cohorts.py -v
git add backend/app/services/cohorts.py backend/app/api/admin_cohorts.py backend/app/main.py backend/tests/services/test_cohorts.py
git commit -m "fix(phase2-deferred): cohort DI completion

Per reviewer: get_user + CohortAudit undefined; no router registration.
Add async get_user via async_session_factory; register admin router in
main.py; add None-user test for robustness."
```

---

### Task D.7: Pilot dataset 30 manual + 20 adversarial with all categories (HIGH)

**Files:**
- Modify: `backend/tests/retrieval/datasets/deep_compare_sections_golden.yaml` (5 missing cases)
- Create: `backend/tests/retrieval/datasets/deep_compare_sections_adversarial.yaml` (20 cases)

**Root cause**: Task 8 only enumerates 25 manual cases (5 + 5 + 5 + 5 = 20 + 5 negative = 25, not 30).

- [ ] **Step 1: Verify current counts**

```python
import yaml
manual = yaml.safe_load(open("backend/tests/retrieval/datasets/deep_compare_sections_golden.yaml"))
print(f"Manual cases: {len(manual)}")
from collections import Counter
print(Counter(c["category"] for c in manual))
```

- [ ] **Step 2: Add 5 missing manual cases (per spec: 10 cross-doc + 5 cross-section + 5 inline + 5 negative = 25 manual, but spec said 30; add 5 more cross-doc)**

Add 5 more `cross_document_compare` cases (different doc types — e.g., Thông tư, Quyết định, etc.).

- [ ] **Step 3: Ensure adversarial YAML has exactly 20 cases**

```python
adversarial = yaml.safe_load(open("backend/tests/retrieval/datasets/deep_compare_sections_adversarial.yaml"))
assert len(adversarial) == 20
```

Categories: 5 wrong-doc same Điều N, 3 fabricated, 3 ACL fail, 3 deadline stress, 2 truncation, 4 prompt injection.

- [ ] **Step 4: Update Task 8 in Plan 2 to reference both datasets + verify counts**

Edit Task 8 description:

```markdown
**Files:**
- Create: `backend/tests/retrieval/datasets/deep_compare_sections_golden.yaml` (30 manual cases)
- Create: `backend/tests/retrieval/datasets/deep_compare_sections_adversarial.yaml` (20 adversarial cases)
```

- [ ] **Step 5: Update Task 9 (`ab_deep_eval.py`) to evaluate BOTH datasets**

Edit Task 9 `run_arm` to iterate both files:

```python
async def run_arm(arm: str, workspace_id: str, base_url: str) -> dict:
    """Run A/B arm against 50 cases (30 manual + 20 adversarial)."""
    from yaml import safe_load
    from pathlib import Path
    
    datasets = [
        ("manual", "backend/tests/retrieval/datasets/deep_compare_sections_golden.yaml"),
        ("adversarial", "backend/tests/retrieval/datasets/deep_compare_sections_adversarial.yaml"),
    ]
    all_results = []
    for dataset_name, dataset_path in datasets:
        cases = safe_load(Path(dataset_path).read_text())
        for case in cases:
            result = await run_case(case, arm, workspace_id, base_url)
            result["dataset"] = dataset_name
            all_results.append(result)
    return {...}
```

- [ ] **Step 6: Commit**

```bash
git add backend/tests/retrieval/datasets/deep_compare_sections_*.yaml backend/docs/superpowers/plans/2026-09-08-deepagent-phase2-pilot.md backend/scripts/ab_deep_eval.py
git commit -m "fix(phase2-deferred): pilot dataset 30+20 + ab_deep_eval both datasets

Per reviewer: Task 8 had only 25 cases (not 30); Task 9 evaluated only
one dataset. Add 5 more manual cases; create adversarial YAML with 20;
ab_deep_eval.py iterates both datasets."
```

---

### Task D.8: Phase 2 gate test manifest + D.11/E.8 hard gates (HIGH)

**Files:**
- Modify: Plan 2 Task 15 (gate report)

**Root cause**: Plan 2 Task 15 has no test manifest and omits D.11/E.8 hard gates.

- [ ] **Step 1: Replace gate report with full manifest + D.11/E.8 gates**

Replace Plan 2 Task 15 Step 6 (Write Phase 2 gate report) with:

```markdown
## Test manifest (per reviewer finding #21)

| Test ID | File | Status |
|---------|------|--------|
| T-provider-id | backend/tests/llm/test_provider_tool_call_id.py | REQUIRED PASS |
| T-adapter | backend/tests/llm/test_langchain_adapter.py | REQUIRED PASS |
| T-accessor | backend/tests/agent/test_document_accessor.py | REQUIRED PASS |
| T-deep-tools | backend/tests/agents/deep_research/test_tools.py | REQUIRED PASS |
| T-deep-evidence | backend/tests/agents/deep_research/test_evidence.py | REQUIRED PASS |
| T-deep-budget | backend/tests/agents/deep_research/test_budget.py | REQUIRED PASS |
| T-deep-graph | backend/tests/agents/deep_research/test_graph.py | REQUIRED PASS |
| T-deep-flags | backend/tests/agents/test_deep_flags_admission.py | REQUIRED PASS (from Plan 2.5 D.1) |
| T-deep-citation | backend/tests/agents/deep_research/test_evidence_projection.py | REQUIRED PASS (from Plan 2.5 D.2) |
| T-deep-ab | backend/scripts/ab_deep_eval.py | smoke run |
| T-sse | backend/tests/agent/test_streaming_extended_envelope.py | REQUIRED PASS |
| T-cohort | backend/tests/services/test_cohorts.py | REQUIRED PASS |
| T-metrics | backend/tests/observability/test_metrics_emitters.py | REQUIRED PASS |
| T-rollback-smoke | backend/scripts/rollback_smoke.py | smoke pass |
| T-frontend | frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx | REQUIRED PASS |

## D.11/E.8 hard gates (executable evidence required)

| Gate | Threshold | Command | Expected output |
|------|-----------|---------|-----------------|
| Pilot compare correctness | ≥90% of 30 manual cases | `python scripts/ab_deep_eval.py --arm-b deep` + filter manual | `report.correctness >= 0.90` |
| Synthesis JSON validity | ≥99% | (same report) | `report.json_validity_rate >= 0.99` |
| Cross-workspace leak | 0 | `pytest tests/agents/test_cross_workspace_negative.py` | 0 leaks |
| Grounding guard fail accepted | 0 | (synthesis report) | `report.grounding_fail_accepted == 0` |
| Total latency p95 | <30s | (ab_deep_eval latency report) | `report.latency_p95_ms < 30000` |
| Late events after terminal | 0 | (ab_deep_eval late_event_rate) | `report.late_event_rate == 0` |
| Rollback smoke pass | 5/5 | `python backend/scripts/rollback_smoke.py` | all 5 pass |
| Pre-existing tests pass (Phase 0/1A/1B) | all | `pytest tests/` | all pass |
| Cross-section conflicts fixed (per Plan 2.5) | all | (Plan 2.5 tasks done) | all |
```

Each command produces a JSON/CLI output. Gate review cites the artifact path + verdict.
```

- [ ] **Step 2: Commit plan update**

```bash
git add backend/docs/superpowers/plans/2026-09-08-deepagent-phase2-pilot.md
git commit -m "fix(phase2-deferred): Phase 2 gate test manifest + D.11/E.8 hard gates

Per reviewer: Task 15 had no test manifest; D.11/E.8 hard gates
(>=90% correctness, >=99% JSON, 0 leak, 0 grounding fail, <30s p95, 0 late)
missing. Add manifest with 15 test IDs + 9 hard gates tied to
executable commands."
```

---

# Part E — Cross-plan consistency

### Task E.1: Resolve cross-plan paths and test conventions (HIGH)

**Files:**
- Modify: Plan 0, 1B, 2 (consistency)

**Root cause**: 
- `frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx` (Plan 0) vs `frontend/src/components/ChatPanel.tsx` (Plan 2 wrong path) — both reference same component differently.
- `backend/tests/scripts/` (Plan 1B new convention) not documented.

- [ ] **Step 1: Add convention note to Plan 1B**

In Plan 1B Task 6, add note:

```markdown
**Test directory convention**: `backend/tests/scripts/` holds unit tests for scripts in `backend/scripts/`. Convention introduced for analyzer; consider extending to other scripts.
```

- [ ] **Step 2: Verify Plan 0 / Plan 2 ChatPanel paths consistent**

After Task D.5 (Plan 2 path fix) + Plan 0 Task 6 verified, both should use `frontend/src/components/rag/__tests__/ChatPanel.rollback.integration.test.tsx`.

- [ ] **Step 3: Commit plan updates**

```bash
git add backend/docs/superpowers/plans/2026-09-08-deepagent-phase1b-router.md
git commit -m "fix(phase2-deferred): cross-plan convention notes

Per reviewer: tests/scripts/ convention introduced but not documented;
ChatPanel path consistency between Plan 0 and Plan 2 (both should
use rag/ subdirectory)."
```

---

# Part F — Phase 2.5 Gate Review

### Task F.1: Phase 2.5 gate review

- [ ] **Step 1: Run all Plan 0 tests**

Run: `cd backend && pytest tests/agents/test_attachment_delete_acl.py tests/agents/test_route_from_resolve_doc_finish.py tests/agents/test_comparison_prompt_assembly.py tests/agents/test_source_snapshot_dedup.py tests/agents/test_persistence_rollback_clears_all.py tests/agents/test_rollback_complete_e2e.py tests/agents/test_session_acl_ingress.py tests/agents/test_markdown_fallback_acl.py -v`
Expected: All pass.

- [ ] **Step 2: Run Plan 1A tests**

Run: `cd backend && pytest tests/agents/test_contracts_validation.py tests/agents/test_graph_atomic_flag.py tests/migrations/ -v`
Expected: All pass.

- [ ] **Step 3: Run Plan 1B tests**

Run: `cd backend && pytest tests/observability/test_shadow_log_redaction.py tests/scripts/test_analyze_shadow_log.py -v`
Expected: All pass.

- [ ] **Step 4: Run Plan 2 tests (including deferred D.1, D.2 etc.)**

Run: `cd backend && pytest tests/llm/ tests/agents/deep_research/ tests/agents/test_deep_flags_admission.py tests/services/test_cohorts.py tests/observability/test_metrics_emitters.py -v`
Expected: All pass.

- [ ] **Step 5: Verify both baselines + metadata + cross-script diff**

```bash
ls backend/tests/reports/baseline_*_metadata.json
python -c "from backend.tests.reports.compare_baselines import compare_baselines; print(compare_baselines(...))"
```

- [ ] **Step 6: Write Phase 2.5 gate report**

Create `backend/tests/reports/phase2_5_gate_report.md`:

```markdown
# Phase 2.5 Gate Report (Deferred Issues)

**Date**: [today]
**Spec**: docs/superpowers/specs/2026-09-08-deepagent-design.md
**Deferred Plan**: docs/superpowers/plans/2026-09-08-deepagent-phase2-deferred.md

## Issues addressed

| # | Issue | Plan task | Status |
|---|-------|-----------|--------|
| 1 | PRE SHA wrong | A.1 | PASS |
| 2 | Baseline metadata hardcoded | A.1 | PASS |
| 3 | B4 SourcesSnapshotAccumulator integration | A.2 | PASS |
| 4 | Frontend test not executable | A.3 | PASS |
| 5 | B5 persistence rollback via endpoint | A.4 | PASS |
| 6 | B6 markdown fallback test mocking | A.5 | PASS |
| 7 | Baseline metrics comparison schema | A.6 | PASS |
| 8 | Plan 1A migration syntax | B.1 | PASS |
| 9 | Plan 1A Task 2.5 ordering | B.2 | PASS |
| 10 | contracts_validation completeness | B.3 | PASS |
| 11 | Plan 1A graph refactor + flag-on test | B.4 | PASS |
| 12 | Plan 1B 7-day rotation | C.1 | PASS |
| 13 | C.7 gates executable evidence | C.2 | PASS |
| 14 | Deep flag admission (AgentType.DEEPAGENT) | D.1 | PASS |
| 15 | External citation projection | D.2 | PASS |
| 16 | ab_deep_eval.py main guard + parser + message | D.3 | PASS |
| 17 | Truncation test fixture + UTF-8 byte-safe | D.4 | PASS |
| 18 | ChatPanel path correction | D.5 | PASS |
| 19 | Cohort DI completion | D.6 | PASS |
| 20 | Pilot dataset 30+20 + both datasets | D.7 | PASS |
| 21 | Phase 2 gate test manifest + D.11/E.8 | D.8 | PASS |
| 22 | Cross-plan convention notes | E.1 | PASS |

## Decision
[ ] All 22 deferred issues PASS — plans executable
[ ] FAIL — list remaining blockers
```

- [ ] **Step 7: Commit**

```bash
git add backend/tests/reports/phase2_5_gate_report.md
git commit -m "docs(phase2-deferred): gate review — all 22 deferred issues pass

Per writing-plans skill: Phase 2.5 covers 22 deferred fixes from reviewer
findings. All tests pass + baselines verified + conventions documented.
Plans now executable without known gaps."
```

---

# Summary

| Part | Tasks | Severity | Source |
|------|-------|----------|--------|
| A | A.1-A.6 (6) | CRITICAL + HIGH | Plan 0 deferred |
| B | B.1-B.4 (4) | HIGH | Plan 1A deferred |
| C | C.1-C.2 (2) | HIGH | Plan 1B deferred |
| D | D.1-D.8 (8) | CRITICAL + HIGH | Plan 2 deferred |
| E | E.1 (1) | HIGH | Cross-plan |
| F | F.1 (1) | (gate) | Verification |

**Total: 22 atomic commits. Each addresses a specific reviewer finding.**

**Execution order** (sequential, after Plans 0/1A/1B/2):
1. Part A (Plan 0 fixes) — run while Phase 0 execution proceeds
2. Part B (Plan 1A fixes) — run while Phase 1A execution proceeds
3. Part C (Plan 1B fixes) — run while Phase 1B execution proceeds
4. Part D (Plan 2 fixes) — run while Phase 2 execution proceeds
5. Part E (cross-plan) — anytime after
6. Part F (gate) — last

**Cross-reference**:
- Plan 0: `docs/superpowers/plans/2026-09-08-deepagent-phase0-blockers.md`
- Plan 1A: `docs/superpowers/plans/2026-09-08-deepagent-phase1a-preprocessor.md`
- Plan 1B: `docs/superpowers/plans/2026-09-08-deepagent-phase1b-router.md`
- Plan 2: `docs/superpowers/plans/2026-09-08-deepagent-phase2-pilot.md`
- **Plan 2.5 (this)**: `docs/superpowers/plans/2026-09-08-deepagent-phase2-deferred.md`

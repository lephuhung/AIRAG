# DeepAgent Phase 2 Deep Agent Pilot + Rollout Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Deep Agent pilot (compare_sections) with LangChain adapter; implement rollout infrastructure (cohort, A/B, observability); activate gated by Phase 1B ACTIVE.

**Architecture:** Deep Agents library (pinned v0.2.5) via LangChain adapter; ONE pilot tool (RetrieveSectionTool); Evidence registry per run_id; programmatic coverage check + buffered synthesis + CitationSanitizer; canary rollout with deterministic cohort allocation.

**Tech Stack:** Pydantic v2, LangChain BaseChatModel, asyncio.wait_for + outer cancellation, deepagents library, structlog, Grafana Loki, GitNexus MCP (per AGENTS.md).

**Spec:** `/home/AIRAG/docs/superpowers/specs/2026-09-08-deepagent-design.md` Section A.5/A.6 + Section D + Section E

## Global Constraints

- Deep Agents release pin: `deepagents==0.2.5` + `langchain-core==0.3.X` (pinned AFTER `compat_test.sh` passes per Q24.A)
- ONE pilot tool: `retrieve_section` (per D.10 TOOL_ALLOWLIST)
- Provider patches: `tool_call_id` preserved (OpenAI), `thought_signature` preserved (Gemini), UUID (Ollama); provider-level tests for parallel/fragmented/roundtrip
- DocumentAccessor reads over authoritative markdown (no semantic fallback per B.4)
- ACL re-validation at tool boundary (independent of upstream)
- Outer `asyncio.wait_for(deadline)` + outer task cancellation + child cleanup
- Programmatic coverage check; buffered synthesis + validate-before-emit
- CitationSanitizer validates synthesis ONLY cites registry Evidence
- Pilot dataset: 30 manual + 20 adversarial = 50 unique (no overlap)
- TOOL_ALLOWLIST = {retrieve_section}; MAX_IMMUTABLE_TASKS = 2

---

### Task 1: Provider patches — preserve tool_call_id (O37)

**Files:**
- Modify: `backend/app/services/llm/openai_compatible.py` (lines ~324-335, 505-524)
- Modify: `backend/app/services/llm/gemini.py` (lines ~182-188, 224-244)
- Modify: `backend/app/services/llm/ollama.py` (lines ~171-285)

**Interfaces:**
- Consumes: existing `LLMProvider` / `StreamChunk` / `ToolCall` types
- Produces: `tool_call_id` populated from provider response (NEVER synthetic unless provider genuinely omits)
- Closes: O37

- [ ] **Step 1: Write failing test for OpenAI tool_call_id preservation**

```python
# backend/tests/llm/test_provider_tool_call_id.py
def test_openai_tool_call_id_preserved():
    """OpenAI provider emits provider-assigned tool_call_id (not synthetic)."""
    provider = OpenAICompatibleProvider(...)
    mock_stream = iter([
        StreamChunk(text="", tool_calls=[ToolCall(
            id="call_abc123",  # provider-assigned
            name="search_documents",
            args={"query": "test"},
        )]),
    ])
    chunks = list(provider._stream_chunks(mock_stream))
    assert chunks[0].tool_calls[0].id == "call_abc123"

def test_gemini_thought_signature_preserved():
    """Gemini preserves opaque thought_signature for continuation."""
    provider = GeminiProvider(...)
    # Mock response with thought_signature
    chunks = list(provider._stream_chunks(mock_with_thought_sig))
    assert chunks[0].thought_signature == "opaque-sig-xyz"

def test_ollama_synthetic_uuid():
    """Ollama has no provider ID; assign deterministic UUID."""
    provider = OllamaProvider(...)
    chunks = list(provider._stream_chunks(mock_no_id))
    assert chunks[0].tool_calls[0].id  # UUID present
```

- [ ] **Step 2: Modify `openai_compatible.py` to preserve provider tool_call_id**

```python
# backend/app/services/llm/openai_compatible.py
# Lines ~324-335 (streaming)
async def _stream_chunks(self, raw_stream):
    async for chunk in raw_stream:
        for choice in chunk.choices:
            if choice.delta.tool_calls:
                for tc in choice.delta.tool_calls:
                    yield StreamChunk(
                        text=choice.delta.content or "",
                        tool_calls=[ToolCall(
                            id=tc.id or self._generate_synthetic_id(...),  # fallback only
                            name=tc.function.name,
                            args=tc.function.arguments,
                        )],
                    )
```

- [ ] **Step 3: Modify `gemini.py` to preserve `thought_signature`**

```python
# backend/app/services/llm/gemini.py
yield StreamChunk(
    text=...,
    tool_calls=[...],
    thought_signature=response.candidates[0].content.parts[0].thought_signature,  # NEW
)
```

- [ ] **Step 4: Modify `ollama.py` to assign UUID**

```python
import uuid
yield StreamChunk(
    tool_calls=[ToolCall(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tool_name}:{args_hash}")),
        # deterministic per call content
        ...
    )]
)
```

- [ ] **Step 5: Run tests**

Run: `cd backend && pytest tests/llm/test_provider_tool_call_id.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/llm/openai_compatible.py backend/app/services/llm/gemini.py backend/app/services/llm/ollama.py backend/tests/llm/test_provider_tool_call_id.py
git commit -m "fix(phase2): provider patches preserve tool_call_id + thought_signature (O37)

Per D.3 / O37: OpenAI preserves provider tool_call_id; Gemini preserves
opaque thought_signature for continuation; Ollama assigns deterministic
UUID (uuid5 with content hash). Provider-level tests cover parallel
calls, fragmented arguments, roundtrip."
```

---

### Task 2: Compatibility gate `scripts/compat_test.sh` (O25, Q24.A)

**Files:**
- Create: `scripts/compat_test.sh`

**Interfaces:**
- Consumes: `requirements.txt` after pinning Deep Agents
- Produces: exit 0 if all gates pass; exit 1 otherwise

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# scripts/compat_test.sh — Phase 2 compatibility gate
# Per Q24.A: MUST pass BEFORE Section D implementation lands.

set -euo pipefail

cd backend

echo "=== Phase 2 Compatibility Gate for Deep Agents + AIRAG ==="

# 1. Import test
python -c "import deepagents, langchain_core, langgraph; print('OK imports')"

# 2. BaseChatModel protocol surface
python -c "
from langchain_core.language_models.chat_models import BaseChatModel
import inspect
required = {'_generate', '_agenerate', '_stream', '_astream', '_llm_type'}
have = set(dir(BaseChatModel))
missing = required - have
assert not missing, f'BaseChatModel missing: {missing}'
print('OK BaseChatModel surface')
"

# 3. Adapter roundtrip
pytest -q tests/llm/test_langchain_adapter.py -k "roundtrip or tool_call_id or async"

# 4. Langfuse callback propagation
pytest -q tests/llm/test_langchain_adapter.py -k "langfuse or run_id or config_revision"

# 5. Cancellation
pytest -q tests/llm/test_langchain_adapter.py -k "cancellation or timeout"

# 6. Hot config reload isolation
pytest -q tests/llm/test_langchain_adapter.py -k "config_swap"

echo "=== ALL GATES PASS ==="
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/compat_test.sh`

- [ ] **Step 3: Run (will fail until adapter built in next task)**

Run: `bash scripts/compat_test.sh`
Expected: FAIL (deepagents not installed yet). This is expected; gate runs successfully after Task 3.

- [ ] **Step 4: Commit (script first; gate wired later)**

```bash
git add scripts/compat_test.sh
git commit -m "chore(phase2): compat_test.sh skeleton (O25, Q24.A)

Per Q24.A: pre-implementation gate. Must pass before Deep Agent code lands.
Six checks: imports, BaseChatModel surface, adapter roundtrip, Langfuse
propagation, cancellation, hot config reload."
```

---

### Task 3: Build `langchain_adapter.py` (Q3.A, D.3)

**Files:**
- Create: `backend/app/services/llm/langchain_adapter.py`

**Interfaces:**
- Consumes: existing `LLMProvider` + `ModelSnapshot` + `RuntimeContext`
- Produces: `BaseChatModel` wrapper for Deep Agents
- Propagates: `run_id`, `config_revision`, `langfuse_session_id`, `parent_agent_type="deepagent"`, `task_id`, `cancellation_event`

- [ ] **Step 1: Write failing test**

```python
# backend/tests/llm/test_langchain_adapter.py
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from app.services.llm.langchain_adapter import LangChainLLMAdapter
from app.services.llm.types import LLMMessage, StreamChunk

@pytest.fixture
def mock_provider():
    class MockProvider:
        async def astream(self, messages, **kwargs):
            yield StreamChunk(text="Hello ", tool_calls=[])
            yield StreamChunk(text="world", tool_calls=[])
    return MockProvider()

async def test_agenerate_message_conversion(mock_provider):
    adapter = LangChainLLMAdapter(provider=mock_provider, config_snapshot=mock_snapshot,
                                  run_id="r1", cancellation_event=None)
    result = await adapter._agenerate([HumanMessage(content="test")])
    assert isinstance(result.generations[0].message, AIMessage)
    assert "Hello world" in result.generations[0].message.content

async def test_tool_call_id_propagation(mock_provider_with_tool_call):
    adapter = LangChainLLMAdapter(provider=mock_provider_with_tool_call, ...)
    result = await adapter._agenerate([HumanMessage(content="call tool")])
    msg = result.generations[0].message
    assert msg.tool_calls[0]["id"] == "call_abc123"  # from provider

async def test_cancellation_propagates(mock_provider_slow):
    import asyncio
    adapter = LangChainLLMAdapter(provider=mock_provider_slow,
                                  cancellation_event=asyncio.Event())
    adapter.cancellation_event.set()  # pre-cancel
    with pytest.raises(asyncio.CancelledError):
        await adapter._agenerate([HumanMessage(content="test")])
```

- [ ] **Step 2: Implement `LangChainLLMAdapter`**

(See spec D.3 for full implementation.)

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class LangChainLLMAdapter(BaseChatModel):
    provider: Any
    config_snapshot: ModelSnapshot
    langfuse_handler: Any | None = None
    run_id: str
    cancellation_event: asyncio.Event | None = None
    principal_id: UUID4 | None = None
    task_id: str | None = None
    parent_agent_type: str = "deepagent"
    
    @property
    def _llm_type(self) -> str:
        return f"airag-{self.config_snapshot.provider}"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("Deep Agent uses async path")
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        lc_messages = [_lc_to_llm(m) for m in messages]
        aggregated_text = ""
        aggregated_tool_calls = []
        
        async for chunk in self.provider.astream(lc_messages, ...):
            if self.cancellation_event and self.cancellation_event.is_set():
                raise asyncio.CancelledError("cancellation_event set")
            if run_manager:
                run_manager.on_llm_new_token(chunk.text or "")
            aggregated_text += chunk.text or ""
            if chunk.tool_calls:
                aggregated_tool_calls.extend(chunk.tool_calls)
        
        message = AIMessage(
            content=aggregated_text,
            tool_calls=[{"id": tc.id, "name": tc.name, "args": tc.args}
                        for tc in aggregated_tool_calls],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    @property
    def _identifying_params(self):
        return {
            "provider": self.config_snapshot.provider,
            "model": self.config_snapshot.model,
            "run_id": self.run_id,
            "config_revision": self.config_snapshot.config_revision,
        }
```

- [ ] **Step 3: Run compat_test.sh — must pass now**

Run: `bash scripts/compat_test.sh`
Expected: ALL GATES PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/llm/langchain_adapter.py backend/tests/llm/test_langchain_adapter.py
git commit -m "feat(phase2): LangChainLLMAdapter (BaseChatModel) (Q3.A, D.3)

Per Q3.A / D.3: BaseChatModel wrapper around AIRAG LLMProvider.
Async bridge (_agenerate) converts BaseMessage ↔ LLMMessage.
Preserves provider tool_call_id (Task 1 fix); checks cancellation_event
between chunks; propagates run_id + config_revision via _identifying_params.
scripts/compat_test.sh gates pass."
```

---

### Task 4: Build `DocumentAccessor.read_full_section` (D.5, O24)

**Files:**
- Create: `backend/app/services/agent/document_accessor.py`

**Interfaces:**
- Consumes: `Document` row + MinIO markdown key + authorized workspace_id
- Produces: `SectionContent` with text + page_range + section_path + document_version
- Constraints: NO semantic fallback; ACL pre-filter via Document row query

- [ ] **Step 1: Write failing test**

```python
# backend/tests/agent/test_document_accessor.py
async def test_read_full_section_acl_predicate(test_db, doc_in_ws_a, user_in_ws_b):
    """Doc in ws A; user has ws B only → DocumentAccessError."""
    with pytest.raises(DocumentAccessError):
        await DocumentAccessor.read_full_section(
            document_id=doc_in_ws_a.id, section_reference="Chương II",
            principal_id=user_in_ws_b.id, allowed_workspace_ids=[ws_b.id],
            session=test_db,
        )

async def test_read_full_section_returns_content(test_db, doc_with_markdown, user_with_access):
    content = await DocumentAccessor.read_full_section(
        document_id=doc_with_markdown.id, section_reference="Chương II",
        principal_id=user_with_access.id, allowed_workspace_ids=user_with_access.workspace_ids,
        session=test_db,
    )
    assert content.text is not None
    assert content.section_path == "Chương II"
    assert content.document_version is not None

async def test_read_full_section_truncates_at_cap(test_db, doc_with_large_section, user_with_access):
    content = await DocumentAccessor.read_full_section(
        document_id=doc_with_large_section.id, section_reference="Chương II",
        principal_id=user_with_access.id, allowed_workspace_ids=user_with_access.workspace_ids,
        session=test_db,
    )
    if len(content.text.encode('utf-8')) > Evidence.MAX_RAW_CONTENT_BYTES:
        assert content.is_truncated is True
```

- [ ] **Step 2: Implement `DocumentAccessor`**

```python
# backend/app/services/agent/document_accessor.py
class DocumentAccessError(Exception):
    """Doc not accessible for principal/workspace."""


class SectionContent(BaseModel):
    text: str
    page_range: str | None
    section_path: str
    document_version: str
    is_truncated: bool
    not_found: bool


class DocumentAccessor:
    """Strict structural section reader. No semantic fallback.
    
    Per D.5 / O24: queries Document row with workspace predicate;
    downloads markdown from MinIO; structural parser identifies chapter.
    """
    
    @staticmethod
    async def read_full_section(
        document_id: UUID, section_reference: str,
        principal_id: UUID, allowed_workspace_ids: list[UUID],
        session: AsyncSession,
    ) -> SectionContent:
        # 1. ACL via Document query (data-boundary rule per B.4)
        doc = await session.execute(
            select(Document).where(
                Document.id == document_id,
                Document.workspace_id.in_(allowed_workspace_ids),
                Document.deleted_at.is_(None),
            )
        )
        doc_row = doc.scalar_one_or_none()
        if doc_row is None:
            raise DocumentAccessError(f"doc {document_id} not accessible")
        
        # 2. Download markdown from MinIO
        markdown = await _download_markdown(doc_row.markdown_s3_key)
        
        # 3. Structural parse
        sections = _parse_structural_sections(markdown, doc_row)
        section = _find_section(sections, section_reference, doc_row)
        
        if section is None:
            return SectionContent(text="", page_range=None, section_path=section_reference,
                                  document_version=str(doc_row.updated_at),
                                  is_truncated=False, not_found=True)
        
        text, page_range = section
        
        # 4. Apply retention cap (byte-safe per A.5)
        text_bytes = text.encode('utf-8')
        is_truncated = False
        if len(text_bytes) > Evidence.MAX_RAW_CONTENT_BYTES:
            text = text_bytes[:Evidence.MAX_RAW_CONTENT_BYTES].decode('utf-8', errors='replace')
            is_truncated = True
        
        return SectionContent(
            text=text, page_range=page_range, section_path=section_reference,
            document_version=str(doc_row.updated_at), is_truncated=is_truncated, not_found=False,
        )
```

- [ ] **Step 3: Run tests**

Run: `cd backend && pytest tests/agent/test_document_accessor.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/agent/document_accessor.py backend/tests/agent/test_document_accessor.py
git commit -m "feat(phase2): DocumentAccessor.read_full_section (D.5, O24)

Per D.5: strict structural reader. NO semantic fallback. ACL predicate
in Document query (workspace_id IN allowed). Truncates by bytes at
Evidence.MAX_RAW_CONTENT_BYTES (A.5). Returns SectionContent with
document_version + page_range."
```

---

### Task 5: Build `deep_research/contracts.py` (already in Phase 1A — extend) + `tools.py` + `evidence.py` + `budget.py` (D.5, D.6, D.7)

**Files:**
- Modify: `backend/app/services/agents/deep_research/contracts.py` (extend with budget types)
- Create: `backend/app/services/agents/deep_research/tools.py`
- Create: `backend/app/services/agents/deep_research/evidence.py`
- Create: `backend/app/services/agents/deep_research/budget.py`

(Phase 1A created contracts.py with A.3-A.6; extend here with D-specific types)

- [ ] **Step 1: Add D-specific types to `contracts.py`**

```python
# Add to deep_research/contracts.py
class DeepAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    
    completion_status: Literal["complete", "partial", "clarification", "deadline", "error"]
    answer_text: str | None = None
    evidence_ids: list[str] = []
    sources: list = []   # ChatSourceChunk[] projected via Section D.6
    missing_requirements: list[str] = []
    routing_trace: dict = {}
```

- [ ] **Step 2: Write failing test for `RetrieveSectionTool`**

```python
# backend/tests/agents/deep_research/test_tools.py
async def test_retrieve_section_tool_acl_pass(ctx, semantic_context, doc_with_section):
    tool = RetrieveSectionTool(ctx=ctx, semantic_context=semantic_context)
    result = await tool._arun("r1")
    assert isinstance(result, str)
    assert semantic_context.document_refs[0].document_handle == doc_with_section.id

async def test_retrieve_section_tool_acl_fail(ctx, doc_in_other_workspace):
    ref = DocumentRefEntry(ref_id="r1", reference="...", document_handle=doc_in_other_workspace.id,
                            resolution_status="resolved")
    sc = PreprocessingResult(original_query="...", document_refs=[ref], preprocessing_status="complete",
                              preprocessor_trace=[])
    tool = RetrieveSectionTool(ctx=ctx, semantic_context=sc)
    with pytest.raises(ToolError, match="not authorized"):
        await tool._arun("r1")

async def test_retrieve_section_tool_budget_exhausted(ctx):
    ctx.consumed_budget.domain_tool_calls = ctx.tool_budget.max_domain_tool_calls
    tool = RetrieveSectionTool(ctx=ctx, semantic_context=make_semantic_context())
    with pytest.raises(ToolError, match="budget exhausted"):
        await tool._arun("r1")
```

- [ ] **Step 3: Implement `RetrieveSectionTool`**

(See spec D.5 for full implementation.)

- [ ] **Step 4: Implement `EvidenceRegistry` + `CitationSanitizer`**

(See spec D.6 for full implementation.)

- [ ] **Step 5: Implement `BudgetGuard`**

(See spec D.7 for full implementation; uses `asyncio.wait_for(deadline)` outer + atomic counters.)

- [ ] **Step 6: Run tests**

Run: `cd backend && pytest tests/agents/deep_research/ -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/agents/deep_research/ backend/tests/agents/deep_research/
git commit -m "feat(phase2): deep_research subpackage — tools, evidence, budget (D.5-D.7)

Per D.5-D.7: RetrieveSectionTool (ACL re-validation at boundary,
budget via BudgetGuard); EvidenceRegistry (in-memory dedup by
content_hash, multi-source preserved); CitationSanitizer (validates
synthesis only cites registry Evidence); BudgetGuard (outer
asyncio.wait_for(deadline) + atomic counters via asyncio.Lock)."
```

---

### Task 6: Build `deep_research/graph.py` (D.4)

**Files:**
- Create: `backend/app/services/agents/deep_research/graph.py`

- [ ] **Step 1: Implement `create_deep_research_graph`**

(See spec D.4 for full implementation.)

- [ ] **Step 2: Implement coordinator system prompt**

(See spec D.4 for prompt.)

- [ ] **Step 3: Wire TOOL_ALLOWLIST + MAX_IMMUTABLE_TASKS enforcement (O31, O36)**

```python
TOOL_ALLOWLIST = frozenset({"retrieve_section"})
MAX_IMMUTABLE_TASKS = 2

def create_deep_research_graph(ctx, semantic_context) -> CompiledGraph:
    # Build-time: only RetrieveSectionTool allowed
    tools = [RetrieveSectionTool(ctx=ctx, semantic_context=semantic_context)]
    for tool in tools:
        assert tool.name in TOOL_ALLOWLIST, f"tool {tool.name} not in allowlist"
    
    # Coordinator has no sub-agents, no built-in write_todos, etc.
    agent = create_deep_agent(
        model=LangChainLLMAdapter(...),
        tools=tools,
        system_prompt=build_coordinator_system_prompt(semantic_context, ctx),
        # No middleware (built-in planning disabled)
        # No sub_agents
    )
    return wrap_with_budget_guard(agent, BudgetGuard(ctx=ctx, ...), ...)
```

- [ ] **Step 4: Test TOOL_ALLOWLIST + MAX_IMMUTABLE_TASKS**

```python
# backend/tests/agents/deep_research/test_graph.py
def test_tool_allowlist_enforced(ctx, semantic_context):
    # Build a fake tool with wrong name
    class FakeTool(BaseTool):
        name = "delete_files"
        description = "..."
    with pytest.raises(AssertionError):
        # Direct assertion in create_deep_research_graph
        create_deep_research_graph(ctx, semantic_context)
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/agents/deep_research/graph.py backend/tests/agents/deep_research/test_graph.py
git commit -m "feat(phase2): create_deep_research_graph + tool allowlist (D.4, O31, O36)

Per D.4 / O31 / O36: create_deep_agent with adapter + tools + budget +
evidence. TOOL_ALLOWLIST={retrieve_section}; MAX_IMMUTABLE_TASKS=2.
No sub-agents; built-in tools disabled; pilot scope only."
```

---

### Task 7: SSE integration — extended terminal envelope (D.9, O28)

**Files:**
- Modify: `backend/app/services/agent/streaming.py`

(Per spec D.9: map Deep events to existing 'status' + 'token'; extend 'complete' with completion_status, evidence_ids, missing_requirements.)

- [ ] **Step 1: Add completion_status field to `complete` event**

```python
# backend/app/services/agent/streaming.py
async def emit_terminal_event(state, deep_agent_result: DeepAgentResult):
    external_sources = [
        project_external_citation(e, state["evidence_registry"])
        for e in state["evidence_registry"].all()
    ]
    await push_event(state, "complete", {
        "answer": deep_agent_result.answer_text,
        "sources": external_sources,
        "images": [],
        "potential_abbreviations": [],
        "people_data": None,
        # NEW fields (extended envelope; backward compat)
        "completion_status": deep_agent_result.completion_status,
        "evidence_ids": deep_agent_result.evidence_ids,
        "missing_requirements": deep_agent_result.missing_requirements,
        "routing_trace": deep_agent_result.routing_trace,
    })
```

- [ ] **Step 2: Test extended envelope**

```python
# backend/tests/agent/test_streaming_extended_envelope.py
async def test_complete_event_has_completion_status(test_client):
    response = await test_client.post("/rag/chat/agent-lg/{ws}/stream", json={...})
    events = parse_sse(response)
    complete = next(e for e in events if e["type"] == "complete")
    assert complete["completion_status"] in ("complete", "partial", "clarification", "deadline", "error")
    assert "evidence_ids" in complete
    assert "missing_requirements" in complete

async def test_sources_sent_before_tokens(test_client):
    """Frontend needs source citations before token text."""
    response = await test_client.post("/rag/chat/agent-lg/{ws}/stream", json={...})
    events = parse_sse(response)
    sources_idx = next(i for i, e in enumerate(events) if e["type"] == "sources")
    tokens_idx = next(i for i, e in enumerate(events) if e["type"] == "token")
    assert sources_idx < tokens_idx
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/agent/streaming.py backend/tests/agent/test_streaming_extended_envelope.py
git commit -m "feat(phase2): SSE extended terminal envelope (D.9, O28)

Per D.9: 'complete' event extended with completion_status,
evidence_ids, missing_requirements, routing_trace. Frontend can ignore
new fields (backward compat). Sources sent BEFORE tokens."
```

---

### Task 8: Pilot dataset — 30 manual + 20 adversarial cases (D.11, O26, Q25.A)

**Files:**
- Create: `backend/tests/retrieval/datasets/deep_compare_sections_golden.yaml`

**Interfaces:**
- Consumes: existing 15 supervisor cases (seed)
- Produces: 30 manual + 20 adversarial = 50 unique cases (no overlap, SME-adjudicated)

- [ ] **Step 1: Write 30 manual cases (per SME annotation)**

Categories:
- 10 cross-document compare (2 different docs)
- 5 cross-section same doc
- 5 inline-content variations
- 5 negative (should NOT route to deepagent)

Each case structure:
```yaml
- id: dc_001
  category: cross_document_compare
  query: "So sánh Chương II NĐ 13/2023/NĐ-CP với Chương III NĐ 24/2018/QH14 về bảo vệ dữ liệu cá nhân"
  expected:
    completion_status: complete
    citations_min: 2
    structural_comparison: true
  ground_truth:
    comparison_points: [...]  # SME-labeled
    correct_citations: [...]   # SME-verified doc numbers
```

- [ ] **Step 2: Write 20 adversarial cases**

Categories:
- 5 wrong-doc same Điều N (proves citation accuracy)
- 3 fabricated doc numbers (proves grounding guard)
- 3 ACL fail scenarios (one doc inaccessible)
- 3 deadline stress (large section)
- 2 truncation (oversized chapter)
- 4 prompt injection attempts

- [ ] **Step 3: Verify no overlap between manual + adversarial**

Run: `python -c "
import yaml
manual = yaml.safe_load(open('tests/retrieval/datasets/deep_compare_sections_golden.yaml'))
adversarial = yaml.safe_load(open('tests/retrieval/datasets/deep_compare_sections_adversarial.yaml'))
manual_ids = {c['id'] for c in manual}
adversarial_ids = {c['id'] for c in adversarial}
assert manual_ids.isdisjoint(adversarial_ids), 'Overlap!'
print(f'Manual: {len(manual)} | Adversarial: {len(adversarial)}')
"`
Expected: 30 manual + 20 adversarial = 50 unique, no overlap.

- [ ] **Step 4: Force-track dataset in git**

Run: `git add -f backend/tests/retrieval/datasets/deep_compare_sections_*.yaml`
(Per harness.md: force-track despite gitignore.)

- [ ] **Step 5: Commit**

```bash
git add backend/tests/retrieval/datasets/deep_compare_sections_*.yaml
git commit -m "test(phase2): 30 manual + 20 adversarial pilot cases (D.11, O26, Q25.A)

Per D.11 / Q25.A: 50 unique cases (no overlap). 10 cross-document,
5 cross-section, 5 inline, 5 negative + 20 adversarial (wrong-doc,
fabricated, ACL fail, deadline, truncation, prompt injection).
SME-annotated ground truth. Frozen at Phase 2 prep; NEVER modified."
```

---

### Task 9: `ab_deep_eval.py` — separate A/B script (E.4, O38)

**Files:**
- Create: `backend/scripts/ab_deep_eval.py`

**Interfaces:**
- Consumes: `deep_compare_sections_golden.yaml` + workspace
- Produces: A/B report JSON with completion_status, latency, grounding_fail_rate, late_event_rate

- [ ] **Step 1: Implement script**

```python
# backend/scripts/ab_deep_eval.py
"""A/B eval for Deep Agent compare_sections pilot.

Drives SSE/LangGraph path (NOT /rag/debug-chat). Different from ab_eval.py.
"""
import argparse, asyncio, json, time
from pathlib import Path
import httpx

DATASET = "tests/retrieval/datasets/deep_compare_sections_golden.yaml"


async def run_arm(arm: str, workspace_id: str, base_url: str) -> dict:
    """Run A/B arm against all 50 cases."""
    cases = json.loads(Path(DATASET).read_text())
    results = []
    for case in cases:
        result = await run_case(case, arm, workspace_id, base_url)
        results.append(result)
    
    return {
        "arm": arm,
        "n_cases": len(cases),
        "completion_status_rates": compute_status_rates(results),
        "grounding_fail_rate": compute_grounding_fail_rate(results),
        "late_event_rate": compute_late_event_rate(results),
        "latency_p95_ms": compute_p95([r["latency_ms"] for r in results]),
        "config_snapshot": capture_config_snapshot(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-a", default="base")
    parser.add_argument("--arm-b", default="deep")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--base-url", default="http://localhost:8080/api/v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    report_a = asyncio.run(run_arm(args.arm_a, args.workspace, args.base_url))
    report_b = asyncio.run(run_arm(args.arm_b, args.workspace, args.base_url))
    
    output = {"arm_a": report_a, "arm_b": report_b, "diff": compute_diff(report_a, report_b)}
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add Makefile target**

```makefile
ab-deep:
	cd backend && python scripts/ab_deep_eval.py \
		--arm-a base --arm-b deep \
		--workspace $(WORKSPACE) \
		--output reports/ab_deep_$$(date +%s).json

ab-deep-compare:
	cd backend && python scripts/ab_deep_compare.py $(A) $(B)
```

- [ ] **Step 3: Commit**

```bash
git add backend/scripts/ab_deep_eval.py Makefile
git commit -m "feat(phase2): ab_deep_eval.py — A/B script for pilot (E.4, O38)

Per E.4: separate file (NOT extend ab_eval.py). Drives SSE/LangGraph
path. Reports completion_status rates, grounding_fail_rate,
late_event_rate, latency_p95. Makefile targets ab-deep + ab-deep-compare."
```

---

### Task 10: Cohort infrastructure — `users.cohort_id` + admin endpoint (E.3, O39)

**Files:**
- Modify: `backend/app/models/user.py` (add cohort_id)
- Modify: `backend/app/main.py` lifespan (migration)
- Create: `backend/app/services/cohorts.py`
- Create: `backend/app/api/admin_cohorts.py`

**Interfaces:**
- Consumes: existing `User` model + superadmin auth
- Produces: `users.cohort_id` column + `cohort_audit` table + admin endpoint
- Deterministic sticky allocation via `hash(user_id + salt) % 100`

- [ ] **Step 1: Add column + migration**

```python
# backend/app/models/user.py
class User(Base):
    # ... existing
    cohort_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    cohort_assigned_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
```

- [ ] **Step 2: Add migration + audit table**

```sql
ALTER TABLE users ADD COLUMN IF NOT EXISTS cohort_id VARCHAR(64);
ALTER TABLE users ADD COLUMN IF NOT EXISTS cohort_assigned_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS ix_users_cohort ON users(cohort_id);

CREATE TABLE IF NOT EXISTS cohort_audit (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    cohort_id VARCHAR(64) NOT NULL,
    changed_by UUID NOT NULL REFERENCES users(id),
    reason VARCHAR(256) NOT NULL,
    changed_at TIMESTAMPTZ DEFAULT now()
);
```

- [ ] **Step 3: Implement `cohorts.py`**

```python
# backend/app/services/cohorts.py
import hashlib
EXPERIMENT_SALT = "deep_canary_v1_salt_2026_09_08"


def is_user_in_experiment(user_id: UUID, experiment_name: str, percent: int) -> bool:
    """Stable hash-based allocation."""
    user = get_user(user_id)
    if user.cohort_id == "force_in": return True
    if user.cohort_id == "force_out": return False
    if user.is_superadmin or (user.cohort_id and user.cohort_id.startswith("internal_")):
        return True
    h = hashlib.sha256(f"{user_id}:{EXPERIMENT_SALT}".encode()).digest()
    bucket = int.from_bytes(h[:4], "big") % 100
    return bucket < percent
```

- [ ] **Step 4: Implement admin endpoint**

```python
# backend/app/api/admin_cohorts.py
@router.post("/admin/cohorts/{user_id}")
async def set_user_cohort(
    user_id: UUID, cohort_id: str, reason: str,
    actor: User = Depends(require_superadmin),
):
    user = await db.get(User, user_id)
    user.cohort_id = cohort_id
    user.cohort_assigned_at = datetime.now()
    audit = CohortAudit(user_id=user_id, cohort_id=cohort_id, changed_by=actor.id, reason=reason)
    db.add(audit)
    await db.commit()
```

- [ ] **Step 5: Tests**

```python
# backend/tests/services/test_cohorts.py
def test_internal_user_always_in():
    user = MockUser(is_superadmin=False, cohort_id="internal_qa")
    assert is_user_in_experiment(user.id, "deep_canary", percent=50)

def test_deterministic_allocation():
    user1 = MockUser(cohort_id=None)
    user2 = MockUser(cohort_id=None)
    # Same salt → same allocation for same user_id
    assert is_user_in_experiment(user1.id, "deep_canary", 50) == \
           is_user_in_experiment(user1.id, "deep_canary", 50)

def test_force_in_override():
    user = MockUser(cohort_id="force_in")
    assert is_user_in_experiment(user.id, "deep_canary", 0)  # 0% but force-in
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/models/user.py backend/app/main.py backend/app/services/cohorts.py backend/app/api/admin_cohorts.py backend/tests/services/test_cohorts.py
git commit -m "feat(phase2): users.cohort_id + cohort_audit + admin endpoint (E.3, O39)

Per Q27.A / E.3: nullable cohort_id + cohort_assigned_at columns +
cohort_audit table. Deterministic sticky allocation via hash(user_id+salt).
Internal users always-in; manual force_in/force_out overrides.
Superadmin-only admin endpoint with audit logging."
```

---

### Task 11: Loki log-derived metrics (E.7, O40)

**Files:**
- Create: `backend/app/services/observability/metrics.py`
- Modify: `backend/app/services/observability/dashboard.json` (Grafana)

**Interfaces:**
- Consumes: existing Loki/Promtail/Grafana stack
- Produces: 6 metric events via structured logging (no Prometheus)

- [ ] **Step 1: Implement metric emitters**

```python
# backend/app/services/observability/metrics.py
import structlog

def emit_completion_status(status, run_id, config_revision):
    structlog.get_logger().info("deep_agent_completion_status",
                                 status=status, run_id=run_id, config_revision=config_revision)

def emit_latency(stage, latency_ms, run_id):
    structlog.get_logger().info("deep_agent_latency",
                                 stage=stage, latency_ms=latency_ms, run_id=run_id)

def emit_grounding_guard_fail(run_id, retracted_text_hash):
    structlog.get_logger().warning("deep_agent_grounding_guard_fail",
                                    run_id=run_id, retracted_text_hash=retracted_text_hash)

def emit_citation_sanitizer_fail(run_id, invalid_ids):
    structlog.get_logger().warning("deep_agent_citation_sanitizer_fail",
                                    run_id=run_id, invalid_ids=str(invalid_ids))

def emit_late_event(run_id, event_type):
    """CRITICAL: must be 0. Alert in Grafana."""
    structlog.get_logger().error("deep_agent_late_event",
                                  run_id=run_id, event_type=event_type)

def emit_tool_budget_exhausted(run_id, task_id, budget_type):
    structlog.get_logger().warning("deep_agent_tool_budget_exhausted",
                                    run_id=run_id, task_id=task_id, budget_type=budget_type)
```

- [ ] **Step 2: Add Grafana dashboard JSON**

Create `backend/app/services/observability/dashboard_deep_agent.json`:

```json
{
  "title": "Deep Agent Metrics",
  "panels": [
    {"title": "Completion status rate", "logql": "sum(rate({app=\"backend\"} | json | event=\"deep_agent_completion_status\" | status=\"complete\" [5m])) / sum(rate({app=\"backend\"} | json | event=\"deep_agent_completion_status\" [5m]))"},
    {"title": "Latency p95 by stage", "logql": "quantile_over_time(0.95, {app=\"backend\"} | json | event=\"deep_agent_latency\" | latency_ms [5m] by (stage))"},
    {"title": "CRITICAL: Late events (must be 0)", "logql": "sum(rate({app=\"backend\"} | json | event=\"deep_agent_late_event\" [5m]))", "alert": "> 0"},
    {"title": "Grounding guard fail rate", "logql": "sum(rate({app=\"backend\"} | json | event=\"deep_agent_grounding_guard_fail\" [5m])) / sum(rate({app=\"backend\"} | json | event=\"deep_agent_synthesis_complete\" [5m]))"},
    {"title": "Citation sanitizer fail rate", "logql": "sum(rate({app=\"backend\"} | json | event=\"deep_agent_citation_sanitizer_fail\" [5m])) / sum(rate({app=\"backend\"} | json | event=\"deep_agent_synthesis_complete\" [5m]))"}
  ]
}
```

- [ ] **Step 3: Wire emitters in deep_research/graph.py + tools.py**

Call `emit_completion_status(...)` etc. at relevant points.

- [ ] **Step 4: Test**

```python
# backend/tests/observability/test_metrics_emitters.py
def test_emit_completion_status_logs_structured(caplog):
    emit_completion_status("complete", "r1", "abc")
    assert "deep_agent_completion_status" in caplog.text
    assert '"status": "complete"' in caplog.text
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/observability/metrics.py backend/app/services/observability/dashboard_deep_agent.json backend/app/services/agents/deep_research/ backend/tests/observability/test_metrics_emitters.py
git commit -m "feat(phase2): Loki log-derived metrics + Grafana dashboard (E.7, O40)

Per Q26.B / E.7: 6 emit points via structlog (no Prometheus client).
Grafana dashboard JSON for: completion status, latency p95 by stage,
late events (alert if > 0), grounding guard fail rate, citation
sanitizer fail rate. Wire emitters in deep_research/graph.py + tools.py."
```

---

### Task 12: Rollback smoke dataset + drill (E.5, O41)

**Files:**
- Create: `backend/scripts/rollback_smoke.py`

**Interfaces:**
- Consumes: 5-10 query smoke set
- Produces: pass/fail based on max latency + completion status

- [ ] **Step 1: Define smoke dataset**

```python
# backend/scripts/rollback_smoke.py
SMOKE_DATASET = [
    {"id": "smoke_01", "query": "Điều 5 văn bản X quy định gì?", "expected_status": "complete", "max_latency_ms": 5000},
    {"id": "smoke_02", "query": "Tìm NĐ 13/2023/NĐ-CP", "expected_status": "complete", "max_latency_ms": 5000},
    {"id": "smoke_03", "query": "Tóm tắt văn bản X", "expected_status": "complete", "max_latency_ms": 10000},
    {"id": "smoke_04", "query": "Xin chào", "expected_status": "complete", "max_latency_ms": 2000},
    {"id": "smoke_05", "query": "Điều 5 và Điều 7 của X khác nhau thế nào?", "expected_status": "complete", "max_latency_ms": 10000},
]


async def run_smoke(test_client, dataset):
    results = []
    for case in dataset:
        start = time.monotonic()
        response = await test_client.post("/rag/chat/agent-lg/{ws}/stream", json={"query": case["query"]})
        events = parse_sse(response)
        complete = next(e for e in events if e["type"] == "complete")
        latency_ms = (time.monotonic() - start) * 1000
        
        results.append({
            "id": case["id"],
            "latency_ms": latency_ms,
            "completion_status": complete.get("completion_status"),
            "expected_status": case["expected_status"],
            "max_latency_ms": case["max_latency_ms"],
            "pass": complete.get("completion_status") == case["expected_status"] and latency_ms <= case["max_latency_ms"],
        })
    return results
```

- [ ] **Step 2: Test**

```python
def test_smoke_pass(test_client):
    results = asyncio.run(run_smoke(test_client, SMOKE_DATASET))
    n_pass = sum(1 for r in results if r["pass"])
    assert n_pass == len(results), f"Smoke FAIL: {results}"
```

- [ ] **Step 3: Commit**

```bash
git add backend/scripts/rollback_smoke.py
git commit -m "feat(phase2): rollback smoke dataset (5 queries) + drill (E.5, O41)

Per E.5: smoke_01..smoke_05 cover greeting, single lookup, doc search,
section read, multi-section compare. Each: max_latency_ms threshold +
expected completion_status. Pass criteria: status match AND latency
within bound. Run as part of CI + manually after rollback."
```

---

### Task 13: Documentation updates (E.6, O42)

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `docs/harness.md`
- Modify: `docs/scaling.md`
- Modify: `.env.example`

- [ ] **Step 1: Update `CLAUDE.md` — Architecture section + Config table**

Add Deep Agent to architecture diagram; add Deep Agent rows to config table (5 new flags).

- [ ] **Step 2: Update `README.md` — Highlights + Quick start**

Add Deep Agent to highlights; add Deep Agent flags to quick start.

- [ ] **Step 3: Update `docs/harness.md` — A/B targets**

Document `make ab-deep` + `make ab-deep-compare` + `make compat-test`.

- [ ] **Step 4: Update `docs/scaling.md` — Capacity account**

Add Deep Agent capacity section:
```
`WEB_CONCURRENCY` × admitted deep-eligible requests × `NEXUSRAG_DEEP_MAX_PARALLEL`
= max concurrent upstream LLM calls per backend instance.
Example: 4 × 100 × 2 = 800 concurrent LLM calls. Cluster-wide guard required.
```

- [ ] **Step 5: Update `.env.example` — All Phase 2 flags**

```bash
# Phase 2: Deep Agent pilot
NEXUSRAG_DEEP_ENABLED=false
NEXUSRAG_DEEP_SHADOW=false
NEXUSRAG_AGENT_DEADLINE_SECONDS=28
NEXUSRAG_DEEP_MAX_PARALLEL=2
NEXUSRAG_DEEP_MAX_DOMAIN_CALLS=6
```

- [ ] **Step 6: Atomic per-phase commit**

```bash
git add CLAUDE.md README.md docs/harness.md docs/scaling.md .env.example
git commit -m "docs(phase2): CLAUDE.md + README.md + harness + scaling + .env.example

Per E.6 / O42: architecture diagram includes Deep Agent subpackage;
config table extended with 5 Phase 2 flags; harness documents ab-deep
+ compat-test targets; scaling doc accounts for DEEP_MAX_PARALLEL
fan-out; .env.example lists all flags with defaults."
```

---

### Task 14: Frontend — handle extended terminal envelope + completion_status

**Files:**
- Modify: `frontend/src/hooks/useRAGChatStream.ts`

- [ ] **Step 1: Update reducer to handle `completion_status`**

```typescript
case "complete":
    return {
        ...state,
        // Authoritative backend payload overrides local accumulation
        sources: ev.sources,           // from backend, NOT localSources
        images: ev.images,
        // NEW fields
        completionStatus: ev.completion_status,
        evidenceIds: ev.evidence_ids,
        missingRequirements: ev.missing_requirements,
        routingTrace: ev.routing_trace,
        // Status-based UI rendering
        isPartial: ev.completion_status === "partial",
        isDeadline: ev.completion_status === "deadline",
        isError: ev.completion_status === "error",
    };
```

- [ ] **Step 2: Render partial/deadline state in UI**

In component:
```tsx
{isPartial && (
    <div className="warning-banner">
        Partial answer — {missingRequirements.join(", ")}
    </div>
)}
{isDeadline && (
    <div className="deadline-banner">
        Request exceeded deadline — partial answer shown
    </div>
)}
```

- [ ] **Step 3: Test**

```typescript
test("reducer handles completion_status partial", () => {
    const newState = reducer(initial, {
        type: "complete",
        completion_status: "partial",
        evidence_ids: ["e1"],
        missing_requirements: ["ref_unauthorized"],
        sources: [{...}],
    });
    expect(newState.completionStatus).toBe("partial");
    expect(newState.isPartial).toBe(true);
});
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/hooks/useRAGChatStream.ts frontend/src/components/ChatPanel.tsx frontend/src/hooks/useRAGChatStream.test.ts
git commit -m "feat(phase2): frontend handles completion_status (partial/deadline/error)

Per D.9 / E.7: frontend reducer processes extended terminal envelope.
UI renders warning banners for partial/deadline/error states.
Authoritative backend sources/images override local accumulation."
```

---

### Task 15: Phase 2 gate review

**Files:**
- Verify: all gates pass

- [ ] **Step 1: Run `compat_test.sh`**

Run: `bash scripts/compat_test.sh`
Expected: ALL GATES PASS (proves Deep Agents compatibility gate).

- [ ] **Step 2: Run full test suite**

Run: `cd backend && pytest tests/ -v`
Expected: All pass.

- [ ] **Step 3: Run ab_deep_eval.py smoke**

Run: `make ab-deep WORKSPACE=...`
Expected: Report JSON generated.

- [ ] **Step 4: Run rollback_smoke.py**

Run: `python backend/scripts/rollback_smoke.py`
Expected: All 5 smoke cases pass.

- [ ] **Step 5: Verify all documentation updated**

Check `git diff main..HEAD -- CLAUDE.md README.md docs/harness.md docs/scaling.md .env.example` shows updates.

- [ ] **Step 6: Write Phase 2 gate report**

```markdown
# Phase 2 Gate Report

**Date**: [today]
**Spec**: docs/superpowers/specs/2026-09-08-deepagent-design.md Section D + E

## Gates
| Gate | Status | Evidence |
|------|--------|----------|
| Provider patches preserve tool_call_id (O37) | PASS | test_provider_tool_call_id.py |
| Compatibility gate (Q24.A) | PASS | compat_test.sh |
| LangChain adapter | PASS | test_langchain_adapter.py |
| DocumentAccessor | PASS | test_document_accessor.py |
| deep_research subpackage | PASS | tests/agents/deep_research/ |
| SSE extended envelope (O28) | PASS | test_streaming_extended_envelope.py |
| Pilot dataset 50 cases (O26) | PASS | deep_compare_sections_golden.yaml |
| ab_deep_eval.py (O38) | PASS | smoke run |
| Cohort infrastructure (O39) | PASS | test_cohorts.py |
| Loki metrics (O40) | PASS | test_metrics_emitters.py |
| Rollback smoke (O41) | PASS | smoke pass |
| Documentation updates (O42) | PASS | git diff verified |
| Frontend completion_status | PASS | useRAGChatStream.test.ts |
| Pre-existing tests (Phase 0/1A/1B) | PASS | full pytest suite |

## Decision
[ ] Phase 2 PASS — proceed to canary E.0 internal
[ ] Phase 2 FAIL — list blockers
```

- [ ] **Step 7: Commit**

```bash
git add backend/tests/reports/phase2_gate_report.md
git commit -m "docs(phase2): gate review — all 14 gates pass

Per E.6 / O44: compat_test + full pytest + ab_deep_eval smoke +
rollback_smoke + doc verification. Phase 2 ready for canary rollout
(E.0 internal → E.1 5% → E.2 25% → E.3 50% → E.4 100%)."
```

---

## Summary

| Task | Subject | Files | Open items closed |
|------|---------|-------|-------------------|
| 1 | Provider patches | openai_compatible/gemini/ollama | O37 |
| 2 | Compat gate script | compat_test.sh | O25 |
| 3 | LangChain adapter | langchain_adapter.py | O2 |
| 4 | DocumentAccessor | document_accessor.py | O24 |
| 5 | deep_research subpackage | 5 files | (D.5-D.7) |
| 6 | Coordinator graph | graph.py | O31, O36 |
| 7 | SSE extended envelope | streaming.py | O28 |
| 8 | Pilot dataset | yaml files | O26, O54 |
| 9 | ab_deep_eval.py | script | O38 |
| 10 | Cohort infra | user/cohorts/admin | O39, O46 |
| 11 | Loki metrics | metrics.py + dashboard | O40, O44 |
| 12 | Rollback smoke | rollback_smoke.py | O41 |
| 13 | Doc updates | 5 files | O42 |
| 14 | Frontend completion_status | useRAGChatStream.ts | (D.9) |
| 15 | Gate review | gate_report.md | O67 |

**Total: 15 atomic commits, Phase 2 ready for canary rollout.**

After Phase 2 gate: deploy E.0 internal → E.1 5% (≥50 deep-eligible req) → E.2 25% (≥250) → E.3 50% (≥500) → E.4 100%. Promote flags to default; remove flag code after 2 release cycles (O43).

---

## Cross-Reference: All 4 Plans

| Plan | Phase | Status |
|------|-------|--------|
| `2026-09-08-deepagent-phase0-blockers.md` | Phase 0 (Section F) | TBD execution |
| `2026-09-08-deepagent-phase1a-preprocessor.md` | Phase 1A (Section B) | TBD execution (depends on Phase 0) |
| `2026-09-08-deepagent-phase1b-router.md` | Phase 1B (Section C) | TBD execution (depends on Phase 1A) |
| `2026-09-08-deepagent-phase2-pilot.md` | Phase 2 (Section D + E rollout) | TBD execution (depends on Phase 1A + 1B) |

**Sequential dependencies**: Phase 0 → 1A → 1B → 2. Each phase has its own gate. Bypass gates require explicit user approval + measurement justification.

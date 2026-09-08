# DeepAgent Phase 1B Complexity Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unified LLM call returning both legacy intent/agent fields + new `complexity_route: RoutingDecision`. Shadow mode validates against legacy classifier before activating new path.

**Architecture:** Single LLM call (temperature=0, max_tokens=320); centralized `build_routing_decision` parses + validates + applies trusted-fact fallback; shadow mode logs diff; ACTIVE flag is sole authority after validation.

**Tech Stack:** Pydantic v2, pydantic-settings, structlog, Grafana Loki log queries, FastAPI lifespan.

**Spec:** `/home/AIRAG/docs/superpowers/specs/2026-09-08-deepagent-design.md` Section A.2 + Section C

## Global Constraints

- Unified LLM call: single request returns `{next_agent, intent, task_plan, needs_memory, is_legal_query, pending_intent, complexity_route: RoutingDecision, reasoning}`
- Output cap: 320 tokens (per C.2)
- Anti-downgrade invariant: `cross_domain AND >=2 refs` consistent across all triggers (O63)
- Deep Agent executor scope check: `cross_agent`/`multi_goal` → supervisor fallback when out of pilot scope (O64)
- Shadow mode: `NEXUSRAG_COMPLEXITY_SHADOW=false` (default); `NEXUSRAG_COMPLEXITY_ACTIVE=false` (default); mutually exclusive
- Mounted volume `/app/backend/logs` for shadow logs (Q12.A)
- PII redaction for shadow logs + Langfuse + A/B reports (O52)

---

### Task 1: Add complexity bundle flags + dependency validation (E.2, C.5)

**Files:**
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`
- Modify: `docs/CLAUDE.md`

**Interfaces:**
- Consumes: existing `Settings` from Phase 1A (has `NEXUSRAG_SEMANTIC_PREPROCESSOR`)
- Produces: `NEXUSRAG_COMPLEXITY_SHADOW`, `NEXUSRAG_COMPLEXITY_ACTIVE`, `NEXUSRAG_SHADOW_SAMPLE_RATE`, `NEXUSRAG_SHADOW_LOG_PATH`
- Dependency chain: SHADOW/ACTIVE require SEMANTIC_PREPROCESSOR; mutual exclusivity SHADOW vs ACTIVE

- [ ] **Step 1: Add flags + validator**

```python
# backend/app/core/config.py
class Settings(BaseSettings):
    # ... existing Phase 1A fields
    NEXUSRAG_SEMANTIC_PREPROCESSOR: bool = False
    
    # Phase 1B additions
    NEXUSRAG_COMPLEXITY_SHADOW: bool = False
    NEXUSRAG_COMPLEXITY_ACTIVE: bool = False
    NEXUSRAG_SHADOW_SAMPLE_RATE: float = 0.1
    NEXUSRAG_SHADOW_CANARY_RATE: float = 1.0
    NEXUSRAG_SHADOW_LOG_PATH: str = "/app/backend/logs/routing_shadow.jsonl"
    
    @model_validator(mode="after")
    def _validate_flag_chain(self):
        # COMPLEXITY_SHADOW/ACTIVE require SEMANTIC_PREPROCESSOR
        if self.NEXUSRAG_COMPLEXITY_SHADOW and not self.NEXUSRAG_SEMANTIC_PREPROCESSOR:
            raise ValueError("COMPLEXITY_SHADOW requires SEMANTIC_PREPROCESSOR")
        if self.NEXUSRAG_COMPLEXITY_ACTIVE and not self.NEXUSRAG_SEMANTIC_PREPROCESSOR:
            raise ValueError("COMPLEXITY_ACTIVE requires SEMANTIC_PREPROCESSOR")
        # Mutual exclusivity
        if self.NEXUSRAG_COMPLEXITY_SHADOW and self.NEXUSRAG_COMPLEXITY_ACTIVE:
            raise ValueError("COMPLEXITY_SHADOW and COMPLEXITY_ACTIVE are mutually exclusive")
        # Sample rate bounds
        if not (0.0 <= self.NEXUSRAG_SHADOW_SAMPLE_RATE <= 1.0):
            raise ValueError("NEXUSRAG_SHADOW_SAMPLE_RATE must be in [0.0, 1.0]")
        return self
```

- [ ] **Step 2: Add to `.env.example`**

```bash
# Phase 1B: Complexity router (shadow → active)
NEXUSRAG_COMPLEXITY_SHADOW=false
NEXUSRAG_COMPLEXITY_ACTIVE=false
NEXUSRAG_SHADOW_SAMPLE_RATE=0.1
NEXUSRAG_SHADOW_CANARY_RATE=1.0
NEXUSRAG_SHADOW_LOG_PATH=/app/backend/logs/routing_shadow.jsonl
```

- [ ] **Step 3: Update CLAUDE.md config table**

Add rows for the 4 new flags.

- [ ] **Step 4: Write tests**

```python
# backend/tests/core/test_complexity_flags.py
def test_complexity_requires_semantic_preprocessor():
    with pytest.raises(ValidationError):
        Settings(NEXUSRAG_SEMANTIC_PREPROCESSOR=False, NEXUSRAG_COMPLEXITY_ACTIVE=True)

def test_shadow_active_mutually_exclusive():
    with pytest.raises(ValidationError):
        Settings(NEXUSRAG_SEMANTIC_PREPROCESSOR=True, NEXUSRAG_COMPLEXITY_SHADOW=True,
                 NEXUSRAG_COMPLEXITY_ACTIVE=True)

def test_sample_rate_bounds():
    with pytest.raises(ValidationError):
        Settings(NEXUSRAG_SHADOW_SAMPLE_RATE=1.5)
```

- [ ] **Step 5: Run tests**

Run: `cd backend && pytest tests/core/test_complexity_flags.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/core/config.py .env.example docs/CLAUDE.md backend/tests/core/test_complexity_flags.py
git commit -m "feat(phase1b): complexity bundle flags + dependency chain

Per C.5 / E.2: NEXUSRAG_COMPLEXITY_SHADOW + COMPLEXITY_ACTIVE flags.
Dependency: both require SEMANTIC_PREPROCESSOR=true (Phase 1A).
Mutual exclusivity: SHADOW and ACTIVE cannot coexist. Sample rate
bounded [0.0, 1.0]."
```

---

### Task 2: Mount shadow log volume + implement PII-redacted logger (C.5, O13)

**Files:**
- Modify: `docker-compose.services.yml` (add volume)
- Create: `backend/app/services/observability/shadow_log.py`

**Interfaces:**
- Consumes: existing Loki/Promtail/Grafana stack
- Produces: `routing_shadow.jsonl` JSONL file with PII-redacted events
- PII redaction: hash CCCD/phone to `[CCCD:hash8]`/`[PHONE:hash8]`; hash document_id to 12-char prefix

- [ ] **Step 1: Update docker-compose.services.yml**

```yaml
services:
  hrag-backend:
    volumes:
      - ./backend/logs:/app/backend/logs   # NEW
```

- [ ] **Step 2: Create shadow logger with PII redaction**

```python
# backend/app/services/observability/shadow_log.py
import asyncio
import hashlib
import json
import re
from datetime import datetime

_CCCD_RE = re.compile(r"\b\d{9,12}\b")
_PHONE_RE = re.compile(r"\b0\d{9,10}\b")
_SHADOW_LOCK = asyncio.Lock()


def _redact_user_query(query: str) -> str:
    query = _CCCD_RE.sub(
        lambda m: f"[CCCD:{hashlib.sha256(m.group().encode()).hexdigest()[:8]}]", query
    )
    query = _PHONE_RE.sub(
        lambda m: f"[PHONE:{hashlib.sha256(m.group().encode()).hexdigest()[:8]}]", query
    )
    return query


def _redact_doc_id(doc_id: str | None) -> str:
    if not doc_id:
        return ""
    return hashlib.sha256(doc_id.encode()).hexdigest()[:12]


def _summarize_semantic_context(sc) -> dict:
    """Compact summary for log (no candidates/notes/offsets)."""
    return {
        "n_refs": len(sc.document_refs),
        "n_abbrevs": len(sc.abbreviations),
        "n_blocking": len(sc.blocking_ambiguities),
        "ref_ids": [r.ref_id for r in sc.document_refs],
        "resolution_statuses": [r.resolution_status for r in sc.document_refs],
    }


async def _log_routing_shadow(record: dict) -> None:
    sanitized = {
        "run_id": record["run_id"],
        "ts": datetime.utcnow().isoformat(),
        "config_revision": record.get("config_revision"),
        "prompt_version": record.get("prompt_version"),
        "user_query_redacted": _redact_user_query(record.get("user_query", "")),
        "semantic_context_summary": _summarize_semantic_context(record["semantic_context"]),
        "new_decision": record["new_decision"],
        "legacy_decision": record["legacy_decision"],
        "agreement": record["new_decision"] == record["legacy_decision"],
    }
    path = settings.NEXUSRAG_SHADOW_LOG_PATH
    async with _SHADOW_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
```

- [ ] **Step 3: Test PII redaction**

```python
# backend/tests/observability/test_shadow_log_redaction.py
def test_cccd_redacted():
    text = "Số CCCD 123456789012 cần tra cứu"
    redacted = _redact_user_query(text)
    assert "123456789012" not in redacted
    assert "[CCCD:" in redacted

def test_phone_redacted():
    text = "Gọi 0987654321 để biết thêm"
    redacted = _redact_user_query(text)
    assert "0987654321" not in redacted
    assert "[PHONE:" in redacted

def test_doc_id_hashed():
    redacted = _redact_doc_id("12345678-1234-1234-1234-123456789012")
    assert len(redacted) == 12
    assert redacted != "12345678-1234-1234-1234-123456789012"
```

- [ ] **Step 4: Test concurrent logging**

```python
def test_concurrent_logging_safe():
    import asyncio
    async def log_many():
        await asyncio.gather(*[_log_routing_shadow({"run_id": f"r{i}", ...}) for i in range(100)])
    asyncio.run(log_many())
    # Verify all 100 lines written; no corruption
    with open("/tmp/test_shadow.jsonl") as f:
        lines = f.readlines()
    assert len(lines) == 100
```

- [ ] **Step 5: Run tests**

Run: `cd backend && pytest tests/observability/test_shadow_log_redaction.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.services.yml backend/app/services/observability/shadow_log.py backend/tests/observability/test_shadow_log_redaction.py
git commit -m "feat(phase1b): shadow log volume + PII-redacted logger (C.5, O13)

Per C.5 / Q12.A: docker-compose mounts /app/backend/logs for durable
shadow logs. PII redaction: CCCD/phone hashed; document_id → 12-char prefix.
asyncio.Lock for concurrent-writer safety. Sanitization verified via tests."
```

---

### Task 3: Build complexity router module — `build_routing_decision` (C.4)

**Files:**
- Modify: `backend/app/services/agents/complexity.py`

**Interfaces:**
- Consumes: LLM output dict + `PreprocessingResult` + `RuntimeHints`
- Produces: `(RoutingDecision, FallbackReason | None)`
- Fallback state machine: 10 rules per C.4 (with corrected cross-domain threshold per O63)

- [ ] **Step 1: Define `RuntimeHints` and `FallbackReason`**

```python
# backend/app/services/agents/complexity.py (additions)

from typing import Literal

class RuntimeHints(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    summary_execution: Literal["single_pass", "needs_map_reduce", "unknown", "not_applicable"]
    inline_content_sufficient: bool | None = None  # True/False/None (unknown)
    cross_domain: bool = False                     # Q11.A derived


class FallbackReason(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reason: Literal["parse_failed", "validation_failed", "timeout", "out_of_pilot_scope"]
    detail: str | None = None
```

- [ ] **Step 2: Implement `_detect_user_request_semantics`**

```python
_USER_REQUEST_COMPARE_RE = re.compile(
    r"\b(?:so\s*sánh|đối\s*chiếu|hợp\s*nhất|tìm\s*(?:mâu\s*thuẫn|khác\s*biệt)|merge|compare|diff|tương\s*quan)\b",
    re.IGNORECASE | re.UNICODE,
)
_USER_REQUEST_MULTI_GOAL_RE = re.compile(
    r"\b(?:rồi\s*sau\s*đó|sau\s*đó|tiếp\s*theo|đồng\s*thời|và\s+cũng|để\s*có\s*thể|nhằm\s+để)\b",
    re.IGNORECASE | re.UNICODE,
)

def _detect_user_request_semantics(query: str) -> set[str]:
    sem: set[str] = set()
    if _USER_REQUEST_COMPARE_RE.search(query): sem.add("compare")
    if _USER_REQUEST_MULTI_GOAL_RE.search(query): sem.add("multi_goal")
    # ... summarize, lookup ...
    return sem
```

- [ ] **Step 3: Implement `build_routing_decision`**

Full per C.4 spec (10 fallback rules + cross-domain threshold consistent).

- [ ] **Step 4: Add Deep Agent executor scope check**

```python
def _check_deep_agent_executor_scope(decision: RoutingDecision) -> RoutingDecision:
    """If deepagent selected but pilot only supports compare_sections,
    fall back to supervisor (per O64)."""
    if decision.execution_mode != "deepagent":
        return decision
    if decision.work_type in ("compare", "multi_target_compare"):
        return decision  # within pilot scope
    # cross_agent, multi_goal, summarize/long_document → out of pilot scope (Phase 3+)
    return RoutingDecision(
        execution_mode="supervisor",
        work_type=decision.work_type,
        needs_document_probe=decision.needs_document_probe,
        reason_code="single_workflow",
        clarification_question=None,
    )
```

- [ ] **Step 5: Test all 10 fallback rules**

```python
# backend/tests/agents/test_complexity_router.py (subset)
def test_rule1_compare_multi_target_deepagent():
    sc = make_semantic_context(n_resolved_refs=2, reference_compare=True)
    hints = RuntimeHints(summary_execution="not_applicable", inline_content_sufficient=False)
    decision, reason = build_routing_decision(None, sc, hints)
    assert decision.execution_mode == "deepagent"
    assert decision.work_type == "compare"
    assert decision.reason_code == "multi_target_compare"

def test_rule3_cross_domain_aligned_threshold():
    """cross_domain=true + >=2 refs → deepagent (not just cross_domain alone)."""
    sc = make_semantic_context(n_resolved_refs=1)  # only 1 ref
    hints = RuntimeHints(summary_execution="not_applicable", inline_content_sufficient=False, cross_domain=True)
    decision, reason = build_routing_decision(None, sc, hints)
    assert decision.execution_mode == "supervisor"  # NOT deepagent

def test_rule8_all_not_found_no_clarify():
    sc = make_semantic_context(all_refs_status="not_found")
    decision, reason = build_routing_decision(None, sc, default_hints())
    assert decision.execution_mode == "supervisor"
    assert decision.reason_code == "single_workflow"

def test_anti_downgrade_compare_with_2_refs():
    sc = make_semantic_context(n_resolved_refs=2, query="So sánh X và Y")
    # LLM fails
    decision, reason = build_routing_decision(None, sc, default_hints())
    assert decision.execution_mode == "deepagent"  # anti-downgrade
    assert reason.reason == "parse_failed"

def test_deep_agent_out_of_pilot_scope_falls_back():
    decision = RoutingDecision(execution_mode="deepagent", work_type="cross_agent",
                                reason_code="cross_agent_dependency", needs_document_probe=False)
    result = _check_deep_agent_executor_scope(decision)
    assert result.execution_mode == "supervisor"
    assert result.reason_code == "single_workflow"
```

- [ ] **Step 6: Run tests**

Run: `cd backend && pytest tests/agents/test_complexity_router.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/agents/complexity.py backend/tests/agents/test_complexity_router.py
git commit -m "feat(phase1b): build_routing_decision + 10-rule fallback (C.4)

Per C.4: 10-rule trusted-fact fallback table. Cross-domain threshold
CONSISTENT across all triggers = cross_domain AND >=2 refs (O63).
Anti-downgrade invariant: compare/merge + >=2 refs → deepagent
even if LLM fails. Deep Agent executor scope fallback: cross_agent
and multi_goal fall back to supervisor when out of pilot scope (O64)."
```

---

### Task 4: Build prompt module + extend `supervisor_scope.py` (C.3, C.6)

**Files:**
- Create: `backend/app/prompts/agents/complexity_router_prompt.py`
- Modify: `backend/app/prompts/agents/supervisor_scope.py` (extend `_SS_OUTPUT_FORMAT`)

**Interfaces:**
- Consumes: existing supervisor prompt
- Produces: unified system prompt (legacy intent + complexity rules); JSON output schema

- [ ] **Step 1: Create complexity_router_prompt.py**

```python
# backend/app/prompts/agents/complexity_router_prompt.py
"""Unified prompt for supervisor + complexity router.

Extends existing _SUPERVISOR_PROMPT with complexity rules.
"""


def build_supervisor_system_prompt_with_complexity() -> str:
    return """Bạn là bộ phân loại ĐƯỜNG THỰC THỆ cho hệ thống hỏi đáp tài liệu đa agent.
Phân loại CẢ hai: (1) ý định + agent truyền thống, (2) độ phức tạp + executor mới.

QUY TẮC (giữ nguyên từ complexity-router.vi.txt):
1. Hiểu toàn bộ yêu cầu + context; không route sớm vì greeting/CCCD/"và"/"so sánh"
2. clarify CHỈ KHI semantic_context.blocking_ambiguities chứa ambiguity THIẾT YẾU
   ảnh hưởng đối tượng/kết quả. not_found/deferred/error KHÔNG tự justify clarify.
3. inline_content_sufficient=true → supervisor TRỪ KHI cần external data HOẶC
   cross-domain work (người + tài liệu); cross-domain KHÔNG được downgrad
4. deepagent CHỈ KHI user-request semantics yêu cầu compare/merge/multi-goal
   AND ≥2 phạm vi truy xuất riêng. Ref count alone KHÔNG đủ.
5. summarize: single_pass→supervisor; needs_map_reduce→deepagent; unknown→supervisor+probe

OUTPUT FORMAT (JSON only, no markdown):
{
  "next_agent": "rag|write|people|direct|finish|resolve_doc",
  "intent": "...",
  "task_plan": ["..."],
  "needs_memory": bool,
  "is_legal_query": bool,
  "pending_intent": "...",
  "complexity_route": {
    "execution_mode": "supervisor|deepagent|clarify",
    "work_type": "lookup|compare|summarize|multi_goal|cross_agent|other",
    "needs_document_probe": bool,
    "reason_code": "single_workflow|inline_content|multi_target_compare|multi_goal|cross_agent_dependency|dependent_research|long_document|summary_size_unknown|missing_reference",
    "clarification_question": "..."
  },
  "reasoning": "<one sentence Vietnamese>"
}
"""


def build_supervisor_user_payload(
    user_query: str,
    recent_context: list[str],
    document_context: list[dict],
    semantic_context: PreprocessingResult,
    runtime_hints: RuntimeHints,
) -> dict:
    """Build JSON user message for LLM (data, not instructions)."""
    return {
        "user_query": user_query,
        "recent_context": recent_context,
        "document_context": document_context,
        "semantic_context": {
            "normalized_query": semantic_context.normalized_query,
            "document_refs": [
                {"ref_id": r.ref_id, "reference": r.reference, "section_reference": r.section_reference,
                 "resolution_status": r.resolution_status, "match_basis": r.match_basis}
                for r in semantic_context.document_refs
            ],
            "blocking_ambiguities": [
                {"description": a.description, "essential": a.essential}
                for a in semantic_context.blocking_ambiguities
            ],
            "preprocessing_status": semantic_context.preprocessing_status,
        },
        "runtime_hints": {
            "summary_execution": runtime_hints.summary_execution,
            "inline_content_sufficient": runtime_hints.inline_content_sufficient,
            "cross_domain": runtime_hints.cross_domain,
        },
    }
```

- [ ] **Step 2: Extend `_SS_OUTPUT_FORMAT` in `supervisor_scope.py`**

```python
# backend/app/prompts/agents/supervisor_scope.py
_SS_OUTPUT_FORMAT = """{
  "next_agent": "rag|write|people|direct|finish|resolve_doc",
  "intent": "<intent_string>",
  "task_plan": ["<step1>", "<step2>"],
  "needs_memory": bool,
  "is_legal_query": bool,
  "pending_intent": "<intent_or_null>",
  "complexity_route": {
    "execution_mode": "supervisor|deepagent|clarify",
    "work_type": "lookup|compare|summarize|multi_goal|cross_agent|other",
    "needs_document_probe": bool,
    "reason_code": "single_workflow|inline_content|multi_target_compare|multi_goal|cross_agent_dependency|dependent_research|long_document|summary_size_unknown|missing_reference",
    "clarification_question": "<or_null>"
  },
  "reasoning": "<one_sentence_vietnamese>"
}"""
```

- [ ] **Step 3: Test prompt builder**

```python
# backend/tests/prompts/test_complexity_router_prompt.py
def test_system_prompt_contains_complexity_rules():
    p = build_supervisor_system_prompt_with_complexity()
    assert "complexity_route" in p
    assert "deepagent" in p
    assert "multi_target_compare" in p

def test_user_payload_excludes_candidates():
    sc = make_semantic_context(n_resolved_refs=2)
    sc_with_candidates = sc.model_copy(update={"document_refs": [
        r.model_copy(update={"candidates": [DocumentCandidate(document_id=UUID4("..."), ...)]})
        for r in sc.document_refs
    ]})
    payload = build_supervisor_user_payload("test", [], [], sc_with_candidates, default_hints())
    assert "candidates" not in str(payload)  # not leaked to LLM
```

- [ ] **Step 4: Run tests**

Run: `cd backend && pytest tests/prompts/test_complexity_router_prompt.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/prompts/agents/complexity_router_prompt.py backend/app/prompts/agents/supervisor_scope.py backend/tests/prompts/test_complexity_router_prompt.py
git commit -m "feat(phase1b): unified prompt with complexity rules (C.3, C.6)

Per C.3: system prompt includes 5 complexity rules (from
complexity-router.vi.txt). User payload excludes candidates (data
not instructions). _SS_OUTPUT_FORMAT extended with complexity_route
block. Output cap 320 tokens verified."
```

---

### Task 5: Modify `supervisor_node` to call unified classifier (C.6)

**Files:**
- Modify: `backend/app/services/agents/supervisor.py` (supervisor_node)
- Modify: `backend/app/services/agents/supervisor.py` (replace interpolated raw message)

**Interfaces:**
- Consumes: `state["semantic_context"]` (set by Phase 1A)
- Produces: `state["complexity_route"]` + legacy fields; AgentTrace persistence
- Cross-cutting: gate by SHADOW/ACTIVE; pure-single-goal fast-path kept (O20)

- [ ] **Step 1: Add fast-path preservation (O20)**

Keep `deterministic_decision_for_scope` for greeting/people (no LLM call).

```python
async def supervisor_node(state: SupervisorState) -> dict:
    """Either legacy or unified classifier based on flag."""
    
    # Fast-path: pure-single-goal (greeting/people) → no LLM
    if _is_pure_single_goal(state):
        decision = _fast_path_decision(state)
        # Persist as AgentTrace (scalar only; no new LLM call)
        return _build_supervisor_result(decision, from_llm=False, ...)
    
    # Legacy mode (NEXUSRAG_COMPLEXITY_ACTIVE=false AND SHADOW=false)
    if not settings.NEXUSRAG_COMPLEXITY_SHADOW and not settings.NEXUSRAG_COMPLEXITY_ACTIVE:
        return _run_legacy_classifier(state)
    
    # Active mode
    if settings.NEXUSRAG_COMPLEXITY_ACTIVE:
        return await _run_unified_classifier(state)
    
    # Shadow mode
    new_decision, new_reason = await _run_unified_classifier(state)
    legacy_decision = await _run_legacy_classifier(state)
    await _log_routing_shadow({
        "run_id": state.get("run_id", ""),
        "config_revision": state.get("flag_snapshot", {}).get("config_revision", ""),
        "user_query": state.get("semantic_context").original_query,
        "semantic_context": state.get("semantic_context"),
        "new_decision": new_decision.model_dump(),
        "legacy_decision": legacy_decision.model_dump(),
    })
    # Route by LEGACY (production unchanged)
    return _build_supervisor_result(legacy_decision, from_llm=True, ...)
```

- [ ] **Step 2: Replace interpolated raw message with JSON payload (O23)**

```python
# OLD:
messages = [
    LLMMessage(role="user", content=f"{system_prompt}\n\n{user_query}\n\n{context}"),
]
# NEW (per C.3):
user_payload = build_supervisor_user_payload(
    user_query=state.get("semantic_context").original_query,
    recent_context=[m.content for m in state["messages"]],
    document_context=...,
    semantic_context=state["semantic_context"],
    runtime_hints=_compute_runtime_hints(state),
)
messages = [
    LLMMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False)),
]
# max_tokens=320 (per C.2)
```

- [ ] **Step 3: Implement `_run_unified_classifier`**

```python
async def _run_unified_classifier(state: SupervisorState) -> RoutingDecision:
    """Single LLM call returns unified output; parse + validate + fallback."""
    user_payload = build_supervisor_user_payload(...)
    messages = [LLMMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False))]
    
    llm = get_thinking_provider()
    resp_text = ""
    async for chunk in llm.astream(
        messages,
        system_prompt=build_supervisor_system_prompt_with_complexity(),
        temperature=0.0,
        max_tokens=320,
        think=False,
    ):
        resp_text += chunk.text or ""
    
    parsed = _parse_supervisor_response_v2(resp_text)
    decision, reason = build_routing_decision(
        llm_output=parsed,
        semantic_context=state["semantic_context"],
        runtime_hints=_compute_runtime_hints(state),
    )
    decision = _check_deep_agent_executor_scope(decision)
    return decision, reason
```

- [ ] **Step 4: Implement `_parse_supervisor_response_v2`**

Parses the unified JSON output (with complexity_route block).

- [ ] **Step 5: Implement AgentTrace persistence with scalar + routing_trace (B.8)**

```python
def _persist_agent_trace(state, decision: RoutingDecision, from_llm: bool) -> None:
    trace = AgentTrace(
        session_id=state["session_id"],
        query_complexity=_scalar_for_trace(decision, state.get("next_agent")),
        routing_trace={
            "execution_mode": decision.execution_mode,
            "reason_code": decision.reason_code,
            "config_revision": state.get("flag_snapshot", {}).get("config_revision", ""),
            "run_id": state.get("run_id", ""),
            "model_snapshot": state.get("flag_snapshot", {}).get("model_snapshot", {}),
        },
        preprocessor_marker=state.get("_preprocessor_marker"),
    )
    db.add(trace)
    db.commit()
```

- [ ] **Step 6: Test mode switching**

```python
# backend/tests/agents/test_supervisor_node_modes.py
def test_legacy_mode_no_unified_call(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.NEXUSRAG_COMPLEXITY_ACTIVE", False)
    monkeypatch.setattr("app.core.config.settings.NEXUSRAG_COMPLEXITY_SHADOW", False)
    with patch("app.services.agents.supervisor._run_unified_classifier") as unified:
        await supervisor_node(state)
        unified.assert_not_called()

def test_shadow_mode_both_classifiers_run(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.NEXUSRAG_COMPLEXITY_SHADOW", True)
    with patch("app.services.agents.supervisor._run_unified_classifier") as unified, \
         patch("app.services.agents.supervisor._run_legacy_classifier") as legacy:
        await supervisor_node(state)
        unified.assert_called_once()
        legacy.assert_called_once()

def test_active_mode_only_unified(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.NEXUSRAG_COMPLEXITY_ACTIVE", True)
    with patch("app.services.agents.supervisor._run_unified_classifier") as unified, \
         patch("app.services.agents.supervisor._run_legacy_classifier") as legacy:
        await supervisor_node(state)
        unified.assert_called_once()
        legacy.assert_not_called()
```

- [ ] **Step 7: Run tests**

Run: `cd backend && pytest tests/agents/test_supervisor_node_modes.py -v`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add backend/app/services/agents/supervisor.py backend/tests/agents/test_supervisor_node_modes.py
git commit -m "feat(phase1b): supervisor_node mode switch + unified classifier (C.6)

Per C.6: supervisor_node branches on SHADOW/ACTIVE/legacy flags.
Pure-single-goal fast-path kept (greeting/people, no LLM call, O20).
Replaces interpolated raw message with JSON user payload (O23).
Persists AgentTrace with scalar + routing_trace JSONB (B.8)."
```

---

### Task 6: Build shadow analysis tooling (O16)

**Files:**
- Create: `backend/scripts/analyze_shadow_log.py`

**Interfaces:**
- Consumes: `routing_shadow.jsonl` file
- Produces: agreement rate, disagreement buckets, parse failure rate, latency overhead, per-class confidence intervals

- [ ] **Step 1: Implement analyzer**

```python
# backend/scripts/analyze_shadow_log.py
"""Compute shadow agreement metrics. Run BEFORE activating NEXUSRAG_COMPLEXITY_ACTIVE."""

import argparse
import json
import math
from collections import defaultdict, Counter
from pathlib import Path


def analyze(log_path: str) -> dict:
    decisions = []
    with open(log_path) as f:
        for line in f:
            decisions.append(json.loads(line))
    
    n = len(decisions)
    if n == 0:
        return {"error": "no decisions"}
    
    agreements = sum(1 for d in decisions if d["agreement"])
    agreement_rate = agreements / n
    
    # Disagreement by reason_code (new_decision)
    disagree_by_reason = defaultdict(Counter)
    for d in decisions:
        if not d["agreement"]:
            reason = d["new_decision"].get("reason_code", "unknown")
            work_type = d["new_decision"].get("work_type", "unknown")
            disagree_by_reason[reason][work_type] += 1
    
    # Disagreement by query_type
    disagree_by_query_type = Counter()
    for d in decisions:
        if not d["agreement"]:
            # heuristic: if semantic_context has >=2 refs → complex query
            n_refs = d["semantic_context_summary"]["n_refs"]
            qtype = "complex" if n_refs >= 2 else "simple"
            disagree_by_query_type[qtype] += 1
    
    # Per-class agreement rate with 95% CI
    def wilson_ci(p, n):
        if n == 0: return (0, 0)
        z = 1.96
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        return (max(0, center - margin), min(1, center + margin))
    
    return {
        "n_decisions": n,
        "agreement_rate": agreement_rate,
        "agreement_ci_95": wilson_ci(agreement_rate, n),
        "disagreement_by_reason": dict(disagree_by_reason),
        "disagreement_by_query_type": dict(disagree_by_query_type),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="/app/backend/logs/routing_shadow.jsonl")
    parser.add_argument("--min-queries", type=int, default=1000)
    args = parser.parse_args()
    
    result = analyze(args.log)
    print(json.dumps(result, indent=2))
    
    # Gate: agreement >= 90% AND n >= min_queries
    if result["n_decisions"] < args.min_queries:
        print(f"FAIL: only {result['n_decisions']} queries; need >= {args.min_queries}")
        exit(1)
    if result["agreement_rate"] < 0.90:
        print(f"FAIL: agreement rate {result['agreement_rate']:.2%} < 90%")
        exit(1)
    print("PASS: ready to activate NEXUSRAG_COMPLEXITY_ACTIVE=true")
```

- [ ] **Step 2: Test with sample data**

```python
# backend/scripts/test_analyze_shadow_log.py
def test_analyze_agreement_rate():
    # Create sample log with 100 decisions, 92 agreements
    log_path = "/tmp/test_shadow.jsonl"
    with open(log_path, "w") as f:
        for i in range(100):
            d = {
                "agreement": i < 92,
                "new_decision": {"reason_code": "single_workflow", "work_type": "lookup"},
                "semantic_context_summary": {"n_refs": 1 if i < 50 else 2},
            }
            f.write(json.dumps(d) + "\n")
    
    result = analyze(log_path)
    assert result["n_decisions"] == 100
    assert 0.85 < result["agreement_rate"] < 0.95
```

- [ ] **Step 3: Run tests**

Run: `cd backend && pytest scripts/test_analyze_shadow_log.py -v`
Expected: All pass.

- [ ] **Step 4: Add Makefile target**

```makefile
analyze-shadow:
	cd backend && python scripts/analyze_shadow_log.py --log $(LOG)
```

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/analyze_shadow_log.py backend/scripts/test_analyze_shadow_log.py Makefile
git commit -m "feat(phase1b): shadow log analysis tooling (O16)

Per E.7 / O16: compute agreement rate, disagreement buckets by reason
+ query_type, parse failure rate, 95% CI. Gate: agreement >= 90%
AND n >= 1000 queries before activating NEXUSRAG_COMPLEXITY_ACTIVE."
```

---

### Task 7: Seed 120-case routing dataset + extend eval-prompts report (C.7, O12)

**Files:**
- Create: `backend/tests/prompts/datasets/routing_golden.yaml`
- Modify: `backend/tests/prompts/conftest.py` (extend RoutingMetricsReport)
- Modify: `backend/tests/prompts/test_routing.py` (add tests)

**Interfaces:**
- Consumes: existing 15 supervisor + 10 analyzer cases (seed)
- Produces: 120-case golden dataset (60 simple + 40 complex + 20 clarify); dev/test split; test adapter

- [ ] **Step 1: Expand existing 15 supervisor cases to 120**

Create `backend/tests/prompts/datasets/routing_golden.yaml` with all 120 cases (60 simple + 40 complex + 20 clarify). Each case has:

```yaml
- id: route_001
  category: simple          # simple | complex | clarify | unknown
  query: "Điều 5 văn bản X quy định gì?"
  semantic_context:
    document_refs: [{ref_id: "r1", reference: "văn bản X", section_reference: "Điều 5",
                     resolution_status: "resolved", match_basis: "exact_number"}]
    blocking_ambiguities: []
    preprocessing_status: "complete"
  runtime_hints:
    summary_execution: "not_applicable"
    inline_content_sufficient: false
    cross_domain: false
  expected:
    execution_mode: supervisor
    work_type: lookup
    needs_document_probe: false
    reason_code: single_workflow
    clarification_question: null
```

- [ ] **Step 2: Add test adapter**

```python
# backend/tests/prompts/routing_test_adapter.py
def build_supervisor_payload_from_case(case: dict) -> dict:
    """Construct exact JSON user message the supervisor_node would build."""
    return {
        "user_query": case["user_query"],
        "recent_context": case.get("recent_context", []),
        "document_context": [
            {"reference": r["reference"], "section": r.get("section_reference"), "handle": None}
            for r in case["semantic_context"]["document_refs"]
        ],
        "semantic_context": case["semantic_context"],
        "runtime_hints": case["runtime_hints"],
    }
```

- [ ] **Step 3: Extend `RoutingMetricsReport`**

```python
# backend/tests/prompts/conftest.py
class RoutingMetricsReport(BaseModel):
    n_total: int
    n_json_valid: int
    json_validity_rate: float            # gate: >= 99%
    recall_complex: float                # gate: >= 95%
    simple_to_deep_rate: float           # gate: <= 5%
    clarify_precision: float             # reported separately
    clarify_recall: float                # reported separately
    latency_p95_ms: float                # gate: <= baseline + 500ms
    fallback_reason_breakdown: dict[str, int]
```

- [ ] **Step 4: Dev/test split**

```python
# backend/tests/prompts/test_routing_golden.py
DATASET_PATH = "tests/prompts/datasets/routing_golden.yaml"
DEV_CASES = load_yaml(DATASET_PATH)[:80]   # 80 dev cases
TEST_CASES = load_yaml(DATASET_PATH)[80:]  # 40 held-out test cases

@pytest.mark.parametrize("case", DEV_CASES)
def test_dev_case_routing(case):
    decision, _ = build_routing_decision_from_case(case)
    assert_decision_matches_expected(decision, case["expected"])

@pytest.mark.parametrize("case", TEST_CASES)
def test_test_case_routing(case):
    """Held-out test cases (never in prompt examples)."""
    decision, _ = build_routing_decision_from_case(case)
    assert_decision_matches_expected(decision, case["expected"])
```

- [ ] **Step 5: Run tests**

Run: `cd backend && pytest tests/prompts/test_routing_golden.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add backend/tests/prompts/datasets/routing_golden.yaml backend/tests/prompts/routing_test_adapter.py backend/tests/prompts/conftest.py backend/tests/prompts/test_routing_golden.py
git commit -m "test(phase1b): 120-case routing dataset + test adapter (O12, C.7)

Per C.7: 120 cases (60 simple + 40 complex + 20 clarify). Dev 80 / held-out
40 split. Held-out NEVER appears in prompt examples.
RoutingMetricsReport fields: json_validity_rate (>=99% gate),
recall_complex (>=95% gate), simple_to_deep_rate (<=5% gate).
build_supervisor_payload_from_case constructs exact JSON payload."
```

---

### Task 8: Langfuse session_id = run_id + propagate attributes (E.7, O51, O52)

**Files:**
- Modify: `backend/app/services/agent/langfuse_tracing.py`
- Modify: `backend/app/services/agent/streaming.py` (set session_id at ingress)

**Interfaces:**
- Consumes: existing Langfuse tracing
- Produces: session_id = run_id; metadata includes config_revision, agent_type, execution_mode, cohort_id, task_id

- [ ] **Step 1: Modify `TracedLLMProvider` to accept + propagate attributes**

```python
# backend/app/services/agent/langfuse_tracing.py
class TracedLLMProvider:
    def _emit_observation(self, *, prompt, completion, usage, **metadata):
        obs = langfuse.generation(
            name=metadata.get("agent_type", "llm_call"),
            model=self.config_snapshot.model,
            input=prompt, output=completion, usage=usage,
            metadata={
                "run_id": metadata.get("run_id"),
                "config_revision": metadata.get("config_revision"),
                "agent_type": metadata.get("agent_type"),
                "execution_mode": metadata.get("execution_mode"),
                "cohort_id": metadata.get("cohort_id"),
                "task_id": metadata.get("task_id"),
            },
            session_id=metadata.get("run_id"),  # ← run_id as session_id
            tags=[
                f"config_revision:{metadata.get('config_revision')}",
                f"agent_type:{metadata.get('agent_type')}",
                f"execution_mode:{metadata.get('execution_mode')}",
            ],
        )
```

- [ ] **Step 2: Apply PII redaction to Langfuse payload**

```python
def _redact_langfuse_payload(payload: dict) -> dict:
    if "input" in payload:
        payload["input"] = _redact_user_query(payload["input"])
    if "output" in payload:
        payload["output"] = _redact_user_query(payload["output"])
    if "metadata" in payload and "document_id" in payload["metadata"]:
        payload["metadata"]["document_id"] = _redact_doc_id(payload["metadata"]["document_id"])
    return payload

# Apply before emit
redacted = _redact_langfuse_payload({"input": prompt, "output": completion, ...})
```

- [ ] **Step 3: Set session_id at ingress in `streaming.py`**

```python
# backend/app/services/agent/streaming.py
async def stream_agent_to_sse(...):
    run_id = str(uuid4())
    langfuse_context.update_current_observation(
        session_id=run_id,
        metadata={"run_id": run_id, "config_revision": flag_snapshot.config_revision},
    )
    state["run_id"] = run_id
```

- [ ] **Step 4: Test**

```python
# backend/tests/observability/test_langfuse_attributes.py
def test_session_id_equals_run_id():
    run_id = str(uuid4())
    provider = TracedLLMProvider(...)
    with patch("app.services.observability.langfuse_context") as mock_ctx:
        provider._emit_observation(prompt="...", completion="...", usage={},
                                    run_id=run_id, config_revision="abc", agent_type="supervisor",
                                    execution_mode="supervisor")
        mock_ctx.update_current_observation.assert_called_with(
            session_id=run_id,
            metadata={"run_id": run_id, "config_revision": "abc"},
        )

def test_pii_redacted_in_langfuse():
    payload = {"input": "Số CCCD 123456789012", "output": "..."}
    redacted = _redact_langfuse_payload(payload)
    assert "123456789012" not in redacted["input"]
```

- [ ] **Step 5: Run tests**

Run: `cd backend && pytest tests/observability/test_langfuse_attributes.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/agent/langfuse_tracing.py backend/app/services/agent/streaming.py backend/tests/observability/test_langfuse_attributes.py
git commit -m "feat(phase1b): Langfuse session_id = run_id + PII redaction (E.7, O51, O52)

Per E.7: langfuse_tracing.py propagates run_id, config_revision,
agent_type, execution_mode, cohort_id, task_id. session_id = run_id.
streaming.py sets session_id at ingress. PII redaction applied to
Langfuse payload (input, output, document_id metadata) — same
redaction as shadow logs."
```

---

### Task 9: Phase 1B gate review (O14)

**Files:**
- Verify: all gates pass

**Interfaces:**
- Consumes: all Phase 1A + Phase 1B tasks
- Produces: Phase 1B gate report

- [ ] **Step 1: Run all tests**

Run: `cd backend && pytest tests/ -v`
Expected: All pass.

- [ ] **Step 2: Verify flag combinations**

```bash
# Flag-off legacy path
NEXUSRAG_SEMANTIC_PREPROCESSOR=false make test
# Phase 1A enabled, complexity off
NEXUSRAG_SEMANTIC_PREPROCESSOR=true NEXUSRAG_COMPLEXITY_ACTIVE=false make test
# Phase 1A + 1B shadow
NEXUSRAG_SEMANTIC_PREPROCESSOR=true NEXUSRAG_COMPLEXITY_SHADOW=true NEXUSRAG_SHADOW_SAMPLE_RATE=1.0 make test-shadow
```

- [ ] **Step 3: Run shadow analysis (no real data yet)**

Run: `python backend/scripts/analyze_shadow_log.py`
Expected: `{"error": "no decisions"}` (acceptable for gate report — actual shadow data accumulates post-deploy).

- [ ] **Step 4: Write Phase 1B gate report**

```markdown
# Phase 1B Gate Report

**Date**: [today]
**Spec**: docs/superpowers/specs/2026-09-08-deepagent-design.md Section C

## Gates
| Gate | Status | Evidence |
|------|--------|----------|
| Complexity bundle flags | PASS | test_complexity_flags.py |
| Shadow log infrastructure | PASS | test_shadow_log_redaction.py |
| build_routing_decision 10-rule fallback | PASS | test_complexity_router.py (all 10 rules) |
| Cross-domain threshold consistent (O63) | PASS | test_cross_domain_aligned_threshold |
| Deep Agent executor scope fallback (O64) | PASS | test_deep_agent_out_of_pilot_scope |
| Prompt module + extended output | PASS | test_complexity_router_prompt.py |
| supervisor_node mode switch | PASS | test_supervisor_node_modes.py |
| Shadow analysis tooling (O16) | PASS | test_analyze_shadow_log.py |
| 120-case routing dataset (O12) | PASS | test_routing_golden.py |
| Langfuse session_id = run_id (O51) | PASS | test_langfuse_attributes.py |
| PII redaction for Langfuse (O52) | PASS | test_langfuse_attributes.py |
| Pre-existing tests pass (Phase 0 + 1A) | PASS | full pytest suite |

## Decision
[ ] Phase 1B PASS — proceed to shadow deployment
[ ] Phase 1B FAIL — list blockers
```

- [ ] **Step 5: Commit**

```bash
git add backend/tests/reports/phase1b_gate_report.md
git commit -m "docs(phase1b): gate review — all 12 gates pass

Per O14: complexity router + shadow mode + dataset + Langfuse + PII.
Phase 1B ready for shadow deployment (≥1000 queries before ACTIVE)."
```

---

## Summary

| Task | Subject | Files | Open items closed |
|------|---------|-------|-------------------|
| 1 | Complexity bundle flags | config + .env.example | O7, O14 |
| 2 | Shadow log volume + PII redactor | docker-compose + new module | O13 |
| 3 | build_routing_decision + fallback | complexity.py | O11 (extended), O63, O64 |
| 4 | Prompt module + scope extension | new module + supervisor_scope | O11 (extended) |
| 5 | supervisor_node mode switch | supervisor.py | O22, O23, O20 |
| 6 | Shadow analysis tooling | new script | O16 |
| 7 | 120-case dataset | yaml + adapter + report | O12, O14 (partial) |
| 8 | Langfuse updates | tracing + streaming | O51, O52 |
| 9 | Gate review | gate report | O14 (verify) |

**Total: 9 atomic commits, Phase 1B ready for shadow deployment.**

Phase 1B gate satisfied → after shadow validation (≥1000 queries, ≥90% agreement per analyze_shadow_log.py), proceed to Phase 2 plan (`2026-09-08-deepagent-phase2-pilot.md`).

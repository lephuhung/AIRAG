# LangGraph v2 Phase 0 Compatibility and Benchmark Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover and pin an exact orchestration/checkpoint stack that supports the frozen runtime-context API and passes frozen-contract parity before any production v2 implementation.

**Architecture:** Compatibility discovery precedes candidate benchmarking. An isolated environment probes exact candidate versions for `context_schema`, AsyncPostgresSaver import/setup, and psycopg DSN operation; only passing combinations run the same typed frozen-contract scenarios for native LangGraph and Deep Agents.

**Tech Stack:** Python 3.11, Pydantic v2, candidate LangGraph/Deep Agents, langgraph-checkpoint-postgres, psycopg3, PostgreSQL, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- Do not edit the frozen spec or production routing.
- Do not pre-approve or hardcode a production package version before discovery passes.
- Candidate packages live only in `backend/.venv-v2-benchmark/`; production requirements change only after the winner is recorded.
- Do not fall back from frozen `context_schema` to `config_schema`.
- Phase output records exact LangGraph, checkpoint-postgres, psycopg, Pydantic, and optional Deep Agents versions.
- Before existing-symbol edits run exact impact; before commits run compare-scope detect-changes and stage only named paths.

---

### Task 0: Verify Repository and Candidate API Preconditions

**Files:**
- Create: `backend/scripts/probe_v2_compatibility.py`
- Create: `backend/tests/agents/v2/orchestrator_compat/test_compatibility_probe.py`
- Read: every Modify/Create path listed below

**Interfaces:**
- Produces: `CompatibilityResult` and `probe_stack(python: Path, checkpoint_dsn: str) -> CompatibilityResult`.

- [ ] **Step 1: Verify paths, symbols, and create-path conflicts**

```bash
set -e
for path in backend/requirements.txt Makefile docs/harness.md backend/app/services/agents/supervisor.py; do test -e "$path"; done
for path in backend/scripts/probe_v2_compatibility.py backend/scripts/benchmark_v2_orchestrators.py backend/scripts/check_v2_orchestrator_gate.py; do test ! -e "$path"; done
rg -n 'create_supervisor_graph|StateGraph\(' backend/app/services/agents/supervisor.py
python - <<'PY'
import re, pathlib
plan = pathlib.Path('docs/superpowers/plans/2026-09-11-langgraph-v2-phase0-benchmark.md').read_text()
modify = [p.split(':')[0] for p in re.findall(r'^- Modify: `([^`]+)`', plan, re.M)]
create = [p.split(':')[0] for p in re.findall(r'^- Create: `([^`]+)`', plan, re.M)]
missing = [p for p in modify if not pathlib.Path(p).exists()]
conflict = [p for p in create if pathlib.Path(p).exists()]
assert not missing and not conflict, {'missing': missing, 'conflict': conflict}
assert pathlib.Path('docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md').read_text().startswith('# LangGraph v2')
print(f'phase0 paths ok: {len(set(modify))} modify, {len(set(create))} create')
PY
```

Expected: all existing paths/symbols resolve and create paths do not conflict. Stop the phase on repository drift.

- [ ] **Step 2: Write failing compatibility tests**

```python
from scripts.probe_v2_compatibility import probe_current_interpreter

def test_state_graph_supports_frozen_context_schema() -> None:
    result = probe_current_interpreter("postgresql://postgres:postgres@hrag-postgres:5432/hrag_test")
    assert result.context_schema_supported
    assert result.async_postgres_saver_imported
    assert result.psycopg_dsn_opened
```

Run:

```bash
cd backend && pytest tests/agents/v2/orchestrator_compat/test_compatibility_probe.py -q
```

Expected: FAIL because probe module is absent.

- [ ] **Step 3: Implement exact API and DSN probe**

```python
from inspect import signature
from importlib.metadata import version
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

assert "context_schema" in signature(StateGraph).parameters

# Every langgraph import used by later-phase code examples must resolve.
from langgraph.types import interrupt, Command
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
import langgraph.graph.state as _lg_state
assert hasattr(_lg_state, "CompiledStateGraph")

async with AsyncPostgresSaver.from_conn_string(checkpoint_dsn) as saver:
    await saver.setup()
```

`probe_v2_compatibility.py` reports installed package versions, verifies a minimal `StateGraph(dict, context_schema=RuntimeContext)` compiles, opens a real `postgresql://` psycopg DSN, runs `AsyncPostgresSaver.setup()`, writes/reads a disposable checkpoint thread, and cleans the test thread. It must reject `postgresql+asyncpg://` rather than passing the SQLAlchemy URL through.

- [ ] **Step 4: Discover passing exact candidates in isolation**

```bash
python3 -m venv backend/.venv-v2-benchmark
backend/.venv-v2-benchmark/bin/pip install --upgrade pip
# For each candidate tuple from current package metadata, install exact versions into the isolated venv,
# run the probe, and retain only passing tuples in compatibility-candidates.json.
backend/.venv-v2-benchmark/bin/python backend/scripts/probe_v2_compatibility.py \
  --discover \
  --checkpoint-dsn "$CHECKPOINT_DATABASE_URL" \
  --output backend/tests/reports/v2-compatibility-candidates.json
backend/.venv-v2-benchmark/bin/pip check
```

Expected: report contains at least one passing exact tuple and proof that `context_schema`, AsyncPostgresSaver, and psycopg DSN operations passed. If none passes, Phase 0 stops without changing production requirements.

- [ ] **Step 5: Test and commit probe only**

```bash
cd backend && pytest tests/agents/v2/orchestrator_compat/test_compatibility_probe.py -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/scripts/probe_v2_compatibility.py backend/tests/agents/v2/orchestrator_compat/test_compatibility_probe.py backend/tests/reports/v2-compatibility-candidates.json
git diff --cached --check
git commit -m "test: discover compatible v2 orchestration stack"
```

---

### Task 1: Build Typed Frozen-Contract Parity Scenarios

**Files:**
- Create: `backend/tests/agents/v2/orchestrator_compat/frozen_contracts.py`
- Create: `backend/tests/agents/v2/orchestrator_compat/scenarios.py`
- Create: `backend/tests/agents/v2/orchestrator_compat/test_contract_parity.py`

**Interfaces:**
- Produces: strict/frozen benchmark `SupervisorV2State`, `TaskPlan`, `AgentResult`, `TaskExecutionSummary`, `ClarificationRequest`, `GraphRuntimeContext`, `Scenario`, and `SCENARIOS` matching the approved spec.

- [ ] **Step 1: Write failing typed fixture tests**

```python
from typing import get_type_hints
from .frozen_contracts import SupervisorV2State, TaskPlan
from .scenarios import SCENARIOS

def test_scenarios_use_typed_frozen_contracts() -> None:
    assert SCENARIOS
    assert set(get_type_hints(SupervisorV2State)) >= {"contract_version", "execution"}
    assert any(isinstance(s.initial_state["execution"].plan, TaskPlan) for s in SCENARIOS)

def test_runtime_context_is_not_in_checkpoint_json() -> None:
    scenario = next(s for s in SCENARIOS if s.scenario_id == "acl-resume")
    assert "workspace_ids" not in scenario.checkpoint_json()
```

Run and expect import failure:

```bash
cd backend && pytest tests/agents/v2/orchestrator_compat/test_contract_parity.py -q
```

- [ ] **Step 2: Implement exact typed benchmark contracts and scenarios**

Use `ConfigDict(extra="forbid", frozen=True, strict=True)`. Create literal, deterministic fixtures for: one-task fast plan; three-task DAG; async fan-in; no-evidence `not_found`; no-evidence `TIMEOUT`; append-only replan using `TaskExecutionSummary`; clarification interrupt/resume; changed ACL runtime replacement; cancellation; outer-only streaming. `Scenario.checkpoint_json()` serializes only `SupervisorV2State`; `GraphRuntimeContext` is supplied separately and must never appear in normalized checkpoint data.

- [ ] **Step 3: Prove mandatory parity facts**

Tests assert unique task/target IDs, append-only plan prefix, fan-in associated by task ID, `not_found != TIMEOUT`, current ACL replaces old runtime, cancellation dispatches no later task, one terminal outer event, and checkpoint bytes deserialize back into the same typed state.

```bash
cd backend && pytest tests/agents/v2/orchestrator_compat/test_contract_parity.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit typed fixtures**

```bash
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/tests/agents/v2/orchestrator_compat/frozen_contracts.py backend/tests/agents/v2/orchestrator_compat/scenarios.py backend/tests/agents/v2/orchestrator_compat/test_contract_parity.py
git commit -m "test: add typed v2 orchestrator parity scenarios"
```

---

### Task 2: Benchmark Passing Stacks and Promote Only the Winner

**Files:**
- Create: `backend/requirements-v2-benchmark.txt`
- Create: `backend/scripts/benchmark_v2_orchestrators.py`
- Create: `backend/scripts/check_v2_orchestrator_gate.py`
- Create: `backend/tests/agents/v2/orchestrator_compat/test_gate.py`
- Create: `backend/tests/reports/v2_orchestrator_benchmark.json`
- Create: `docs/benchmarks/langgraph-v2-orchestrator.md`
- Modify: `backend/requirements.txt`
- Modify: `Makefile`
- Modify: `docs/harness.md`

**Interfaces:**
- Consumes: only exact passing tuples from `v2-compatibility-candidates.json` and typed scenarios from Task 1.
- Produces: benchmark report, `evaluate_gate()`, exact winner pins, and documented API proof.

- [ ] **Step 1: Verify current symbols and run impacts**

```bash
set -e
test -e backend/scripts/ab_eval.py
test -e Makefile
test -e docs/harness.md
rg -n 'langgraph|^test:|^ab:' backend/requirements.txt Makefile
```

Then invoke GitNexus upstream impact for the exact discovered symbols `scripts.ab_eval.cmd_run` and `scripts.ab_eval._call`. Expected: paths/symbols resolve and impact is below HIGH; otherwise stop for review rather than inventing or editing a different seam.

- [ ] **Step 2: Write failing gate tests**

Tests reject candidates lacking any compatibility proof or typed scenario; require exact package versions; select native on tie; make Deep Agents eligible only when parity passes, p95 ≤ native×1.15, and checkpoint bytes ≤ native×1.10.

```bash
cd backend && pytest tests/agents/v2/orchestrator_compat/test_gate.py -q
```

Expected: FAIL before checker exists.

- [ ] **Step 3: Materialize discovered requirements, not guessed pins**

`probe_v2_compatibility.py --write-requirements backend/requirements-v2-benchmark.txt` writes the exact passing tuple selected for benchmarking. The file must include exact LangGraph, checkpoint-postgres, psycopg, Pydantic, pytest, and candidate Deep Agents versions from the discovery report; no example version is treated as approved.

```bash
backend/.venv-v2-benchmark/bin/python backend/scripts/probe_v2_compatibility.py \
  --input backend/tests/reports/v2-compatibility-candidates.json \
  --write-requirements backend/requirements-v2-benchmark.txt
backend/.venv-v2-benchmark/bin/pip install --requirement backend/requirements-v2-benchmark.txt
backend/.venv-v2-benchmark/bin/pip check
```

- [ ] **Step 4: Run seeded typed parity benchmark**

Both adapters consume the actual typed Scenario objects. Run warmup 10, iterations 100, seed 20260910; measure p50/p95, peak memory, canonical typed checkpoint bytes, dependency count, interrupt/resume correctness, and outer-event count.

```bash
backend/.venv-v2-benchmark/bin/python backend/scripts/benchmark_v2_orchestrators.py \
  --compatibility-report backend/tests/reports/v2-compatibility-candidates.json \
  --warmup 10 --iterations 100 --seed 20260910 \
  --checkpoint-dsn "$CHECKPOINT_DATABASE_URL" \
  --output backend/tests/reports/v2_orchestrator_benchmark.json
backend/.venv-v2-benchmark/bin/python backend/scripts/check_v2_orchestrator_gate.py \
  backend/tests/reports/v2_orchestrator_benchmark.json
```

Expected: exact winner; every mandatory parity scenario passes; report includes exact versions and API/DSN proof.

- [ ] **Step 5: Promote winner pins and validate container imports**

Replace the current unbounded LangGraph dependency with exact tested winner pins. Add Deep Agents only if it wins. Add exact checkpoint-postgres and psycopg pins proven by the report. Add `CHECKPOINT_DATABASE_URL=postgresql://...` documentation; never reuse `DATABASE_URL=postgresql+asyncpg://...` directly.

```bash
docker exec hrag-backend python -m pip check
docker exec hrag-backend python - <<'PY'
from inspect import signature
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
assert "context_schema" in signature(StateGraph).parameters
print(AsyncPostgresSaver)
PY
```

- [ ] **Step 6: Document, test, and commit**

Document exact versions, probe output, DSN, parity, p50/p95/memory/checkpoint bytes, winner, and rejection reasons. Add Make targets `v2-orchestrator-probe`, `v2-orchestrator-benchmark`, and `v2-orchestrator-gate`.

```bash
cd backend && pytest tests/agents/v2/orchestrator_compat -q
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/requirements-v2-benchmark.txt backend/requirements.txt backend/scripts/probe_v2_compatibility.py backend/scripts/benchmark_v2_orchestrators.py backend/scripts/check_v2_orchestrator_gate.py backend/tests/agents/v2/orchestrator_compat backend/tests/reports/v2-compatibility-candidates.json backend/tests/reports/v2_orchestrator_benchmark.json docs/benchmarks/langgraph-v2-orchestrator.md Makefile docs/harness.md
git diff --cached --check
git commit -m "build: select and pin compatible v2 orchestrator"
node .gitnexus/run.cjs analyze
```

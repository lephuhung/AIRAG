# LangGraph v2 Phase 0 Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the one frozen-contract Write blocker and select/pin a contract-compatible orchestration implementation without changing production routing.

**Architecture:** Frozen contract fixtures drive native LangGraph and an isolated Deep Agents adapter through identical scenarios. Hard parity precedes latency/size comparison; the winner alone is promoted into production dependencies.

**Tech Stack:** Python 3.11, Pydantic v2, LangGraph, optional Deep Agents in isolated virtualenv, pytest, JSON benchmark reports.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- Production remains v1 throughout this plan.
- Benchmark packages install only into `backend/.venv-v2-benchmark/`; do not alter the running backend image.
- Pasted-text Write is a non-factual transform: typed WriteOutput validation is its success gate; retrieved factual content still requires evidence/evaluation/grounding.
- Before editing `Makefile`, run GitNexus impact for the make targets being changed; before editing spec prose, no runtime-symbol impact exists.
- Before each commit run `node .gitnexus/run.cjs detect-changes --scope compare --base-ref main`.

---

### Task 1: Freeze the Pasted-Text Write Rule

**Files:**
- Modify: `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`
- Test: shell assertions in this task

**Interfaces:**
- Consumes: existing Write route and factual synthesis invariant.
- Produces: explicit non-factual transform exception consumed by Phase 2 `write_graph`.

- [ ] **Step 1: Verify the rule is absent**

```bash
python - <<'PY'
from pathlib import Path
text = Path("docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md").read_text()
required = [
    "pasted-text Write is a non-factual transform",
    "WriteOutput schema validation is its success gate",
    "does not create EvidenceRecord, Coverage, EvidenceEvaluation, or AnswerClaim",
]
missing = [value for value in required if value not in text]
assert missing, "amendment already present"
print(missing)
PY
```

Expected: exits 0 and prints at least one missing sentence.

- [ ] **Step 2: Add exactly one behavior paragraph and one acceptance bullet**

```markdown
A pasted-text Write is a non-factual transform. WriteOutput schema validation is its success gate and it does not create EvidenceRecord, Coverage, EvidenceEvaluation, or AnswerClaim. A Write operation that reads external factual sources is not this exception and follows the factual evidence/evaluation/grounding path.
```

Add an acceptance test sentence requiring a grammar-only request to return typed output with no evidence and a source-backed rewrite to require sufficient grounded evidence.

- [ ] **Step 3: Verify exact contract text and frozen status**

```bash
python - <<'PY'
from pathlib import Path
text = Path("docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md").read_text()
for value in (
    "**Status:** Approved design",
    "pasted-text Write is a non-factual transform",
    "WriteOutput schema validation is its success gate",
    "does not create EvidenceRecord, Coverage, EvidenceEvaluation, or AnswerClaim",
):
    assert value in text, value
PY
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 4: Detect and commit only the amendment**

```bash
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md
git diff --cached --check
git commit -m "docs: classify pasted-text write execution"
```

Expected: one documentation-only commit.

---

### Task 2: Build Frozen Contract-Parity Fixtures

**Files:**
- Create: `backend/tests/agents/v2/orchestrator_compat/models.py`
- Create: `backend/tests/agents/v2/orchestrator_compat/scenarios.py`
- Create: `backend/tests/agents/v2/orchestrator_compat/test_contract_parity.py`

**Interfaces:**
- Produces: `Scenario`, `CandidateResult`, `OrchestratorCandidate.run()` and `SCENARIOS`.

- [ ] **Step 1: Check ignore status and create a failing import test**

```bash
git check-ignore -v backend/tests/agents/v2/orchestrator_compat/test_contract_parity.py || true
mkdir -p backend/tests/agents/v2/orchestrator_compat
cat > backend/tests/agents/v2/orchestrator_compat/test_contract_parity.py <<'PY'
from app.services.agents.v2.orchestrator_compat import normalize_result
from .scenarios import SCENARIOS

def test_fixture_ids_are_unique() -> None:
    assert len({scenario.scenario_id for scenario in SCENARIOS}) == len(SCENARIOS)

def test_normalizer_removes_runtime_context() -> None:
    state = {"contract_version": "2.0", "task_ids": ["T1"], "runtime": {"user_id": "secret"}}
    assert normalize_result(state) == {"contract_version": "2.0", "task_ids": ["T1"]}
PY
cd backend && pytest tests/agents/v2/orchestrator_compat/test_contract_parity.py -q
```

Expected: FAIL because the module and scenarios do not exist.

- [ ] **Step 2: Define complete benchmark types**

```python
# backend/tests/agents/v2/orchestrator_compat/models.py
from dataclasses import dataclass
from typing import Awaitable, Callable

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    initial_state: dict[str, object]
    expected_state: dict[str, object]

@dataclass(frozen=True)
class CandidateResult:
    scenario_id: str
    normalized_state: dict[str, object]
    elapsed_ms: float
    checkpoint_bytes: int
    peak_bytes: int

CandidateRunner = Callable[[Scenario], Awaitable[CandidateResult]]

@dataclass(frozen=True)
class OrchestratorCandidate:
    name: str
    run: CandidateRunner
```

Create ten concrete `SCENARIOS`: one-task fast, three-task DAG, async fan-in, not_found, TIMEOUT, append-only replan, clarification interrupt/resume, ACL replacement, cancellation, and outer-only streaming. Each fixture contains literal IDs and expected normalized state.

- [ ] **Step 3: Implement deterministic normalization**

```python
# backend/app/services/agents/v2/orchestrator_compat.py
_RUNTIME_KEYS = frozenset({"runtime", "user_id", "workspace_ids", "deadline_at", "services"})

def normalize_result(value: object) -> object:
    if isinstance(value, dict):
        return {key: normalize_result(item) for key, item in sorted(value.items()) if key not in _RUNTIME_KEYS}
    if isinstance(value, list):
        return [normalize_result(item) for item in value]
    return value
```

- [ ] **Step 4: Run parity fixture tests**

```bash
cd backend && pytest tests/agents/v2/orchestrator_compat/test_contract_parity.py -q
```

Expected: PASS with 2 tests.

- [ ] **Step 5: Detect and commit narrow paths**

```bash
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/app/services/agents/v2/orchestrator_compat.py backend/tests/agents/v2/orchestrator_compat
git commit -m "test: add v2 orchestrator parity fixtures"
```

---

### Task 3: Isolate Candidate Dependencies and Benchmark Both Candidates

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
- Produces: `evaluate_gate(report: BenchmarkReport) -> GateDecision` and a winner with exact package pins.

- [ ] **Step 1: Impact-check the existing make targets**

```bash
# Make targets are not indexed code symbols; inspect exact recipes instead.
grep -nE '^(test|ab):' Makefile
```

Expected: record callers/processes/risk before editing `Makefile`; stop if risk is HIGH/CRITICAL.

- [ ] **Step 2: Create the isolated environment**

```text
# backend/requirements-v2-benchmark.txt
langgraph==0.2.76
langgraph-checkpoint-postgres==2.0.21
psycopg[binary,pool]==3.2.6
deepagents==0.2.5
pydantic==2.10.6
pytest==8.3.5
pytest-asyncio==0.25.3
```

```bash
python3 -m venv backend/.venv-v2-benchmark
backend/.venv-v2-benchmark/bin/pip install --requirement backend/requirements-v2-benchmark.txt
backend/.venv-v2-benchmark/bin/pip check
```

Expected: installation and `pip check` succeed. If a pin is unavailable, update the file to the installed compatible exact version and record it in the benchmark report; do not install into production requirements.

- [ ] **Step 3: Write gate tests first**

```python
from scripts.check_v2_orchestrator_gate import BenchmarkReport, CandidateMetrics, evaluate_gate

def test_tie_selects_native() -> None:
    report = BenchmarkReport(
        native=CandidateMetrics(True, 10.0, 1000),
        deep_agents=CandidateMetrics(True, 10.0, 1000),
    )
    assert evaluate_gate(report).winner == "native"

def test_ineligible_candidate_cannot_win() -> None:
    report = BenchmarkReport(
        native=CandidateMetrics(True, 20.0, 1000),
        deep_agents=CandidateMetrics(False, 1.0, 1),
    )
    assert evaluate_gate(report).winner == "native"
```

Run:

```bash
cd backend && PYTHONPATH=. pytest tests/agents/v2/orchestrator_compat/test_gate.py -q
```

Expected: FAIL because gate types/functions are absent.

- [ ] **Step 4: Implement complete gate types and thresholds**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CandidateMetrics:
    parity_passed: bool
    p95_ms: float
    checkpoint_bytes: int

@dataclass(frozen=True)
class BenchmarkReport:
    native: CandidateMetrics
    deep_agents: CandidateMetrics

@dataclass(frozen=True)
class GateDecision:
    winner: str
    reason: str

def evaluate_gate(report: BenchmarkReport) -> GateDecision:
    if not report.native.parity_passed:
        raise ValueError("native candidate failed mandatory parity")
    deep = report.deep_agents
    if not deep.parity_passed:
        return GateDecision("native", "deep_agents failed parity")
    p95_ok = deep.p95_ms <= report.native.p95_ms * 1.15
    size_ok = deep.checkpoint_bytes <= report.native.checkpoint_bytes * 1.10
    if p95_ok and size_ok and deep.p95_ms < report.native.p95_ms:
        return GateDecision("deep_agents", "eligible and faster")
    return GateDecision("native", "tie or threshold failure selects native")
```

- [ ] **Step 5: Implement and run the seeded benchmark**

`benchmark_v2_orchestrators.py` must import both candidates only inside their runner functions, run all ten scenarios with warmup 10/iterations 100/seed 20260910, measure `perf_counter_ns`, `tracemalloc`, canonical JSON checkpoint bytes, and redact runtime keys through `normalize_result`.

```bash
backend/.venv-v2-benchmark/bin/python backend/scripts/benchmark_v2_orchestrators.py --warmup 10 --iterations 100 --seed 20260910 --output backend/tests/reports/v2_orchestrator_benchmark.json
backend/.venv-v2-benchmark/bin/python backend/scripts/check_v2_orchestrator_gate.py backend/tests/reports/v2_orchestrator_benchmark.json
```

Expected: both commands exit 0 and report one winner.

- [ ] **Step 6: Promote only the winning dependency pins**

If native wins, replace the unbounded LangGraph requirement in `backend/requirements.txt` with the exact tested LangGraph pin and do not add Deep Agents. If Deep Agents wins, add both exact tested pins. In either case add the checkpoint stack tested with the winner as exact pins:

```text
langgraph-checkpoint-postgres==2.0.21
psycopg[binary,pool]==3.2.6
```

If either exact checkpoint pin fails compatibility in the isolated environment, Phase 0 fails: select and record a different exact compatible pair before modifying production requirements. Verify the async API exists with `python -c 'from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver'`. Then rebuild only backend dependencies and run:

```bash
docker exec hrag-backend python -m pip check
docker exec hrag-backend python -c 'from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver; print(AsyncPostgresSaver)'
docker exec hrag-backend pytest tests/agents/v2/orchestrator_compat -q
```

Expected: PASS.

- [ ] **Step 7: Document and commit**

Update `docs/benchmarks/langgraph-v2-orchestrator.md` with versions, host/container details, parity results, p50/p95, memory, checkpoint bytes, winner, and rejection reasons. Add `v2-orchestrator-benchmark` and `v2-orchestrator-gate` Make targets and document them in `docs/harness.md`.

```bash
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
git add backend/requirements-v2-benchmark.txt backend/requirements.txt backend/scripts/benchmark_v2_orchestrators.py backend/scripts/check_v2_orchestrator_gate.py backend/tests/agents/v2/orchestrator_compat backend/tests/reports/v2_orchestrator_benchmark.json docs/benchmarks/langgraph-v2-orchestrator.md Makefile docs/harness.md
git diff --cached --check
git commit -m "build: select and pin v2 orchestrator"
node .gitnexus/run.cjs analyze
```

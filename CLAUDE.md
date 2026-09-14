# AIRAG — Agent Guidance (canonical)

This file is the **single source of truth** for the agent architecture
(`AGENTS.md` points here; do NOT maintain a second copy elsewhere — other docs
link here instead of duplicating it).

## One agent backend, two arms

- The **v1 supervisor graph** (`app/services/agents/supervisor.py`, built by
  `create_supervisor_graph()`) is the default serving arm.
  `app/services/agent/` (singular) is a thin compat shim — not a separate
  architecture.
- The **v2 LangGraph stack** lives under `backend/app/services/agents/v2/`.
  It serves only traffic the rollout control explicitly selects; everything
  else stays on v1 (see "Rollout" below).

## Ownership model (Agent vs Node vs Capability vs Skill)

| Layer | What it is | Authority |
|---|---|---|
| **Agent** | The supervisor entry + tool-calling surface (`agents/v2/tools/`, `supervisor_v2.py`). Proposes work, never executes it. | No execution, persistence, or sufficiency authority. Advisory input only. |
| **Node** | Graph nodes (`agents/v2/nodes/`, `complex_research_graph.py`). Transform state, route, synthesize. | May NOT call capabilities directly; may NOT persist plans. |
| **Capability** | Shared factual implementations (`agents/v2/capabilities/` — document search, section, KG, people lookup). Fast and complex paths share the same implementations. | Executes only via the scheduler, with `AgentRequest` + `CapabilityRuntimeContext`. Never receives supervisor/graph state. |
| **Skill** | Policy helpers (`agents/v2/skills/` — compare, summarize map/reduce). Deterministic workflows, not agents. | No binding, execution, or persistence authority. |

There are **no domain agents and no domain subgraphs**: no `people_agent.py`,
`summary_agent.py`, `comparison_agent.py`, `document_agent.py`,
`section_agent.py`, `kg_agent.py`, and no `v2/domain/*_graph.py`. The single
adaptive planning boundary for complex research is the complex-research
subgraph (`plan → validate_checkpoint → execute → evaluate → decide`, extended
with bounded append-only replan).

## Execution invariant (load-bearing)

```
agent tool call → validated + checkpointed TaskPlan → TaskScheduler → capability
```

- Every factual execution owns a **validated, checkpointed `TaskPlan`** before
  capability dispatch. Validation is the frozen `validate_replan` contract;
  the single constructor for appended plans is
  `replanning.append_replan_tasks` (all owners, including the materialize node
  and the tool gateway, delegate to it).
- The **only** dispatcher is the shared `TaskScheduler`
  (`agents/v2/execution/scheduler.py`). No second scheduler exists.
- **Agent-facing adapters (`agents/v2/tools/`) cannot call capabilities
  directly**: no `capability.execute(`, no `TaskScheduler`, no
  `scheduler.execute(`, no `checkpointer`, no `safe_metadata`/`Mapping[str`
  in `tools/`. They emit proposals (`AgentToolGateway.propose`) and
  observations; the graph validates, checkpoints, then dispatches.
- Capabilities receive **`AgentRequest` (the typed invocation payload) + `CapabilityRuntimeContext` (the only runtime context)** and
  never supervisor/graph state.

## Sensitive observation projection

`AgentToolObservation` never carries raw personal scalars. The projector
(`tools/observations.py`) strips raw People fields (fail-closed), and
`redact_scalar_for_model()` removes `person_identifier` from any model-facing
planning/replanning projection while the checkpointed plan retains it per the
frozen contract. Both a known People scalar and a known runtime secret are
asserted absent from every advisory projection, including the synthesis
`SynthesisInput`/`SynthesisEvidence` boundary (typed source identity never
reaches the synthesis prompt).

## People → Document materialization

Deterministic, governed, append-only: a People task result is hydrated from
evidence the materializer verifies is People evidence owned by the supplying
task (`task_id` + typed `PeopleSourceIdentity`, never a label check); the
materializer appends exactly one deterministic `document.search` task via the
governed append path (validated → checkpointed → dispatched on a second
execute pass). `not_found`/`TIMEOUT`/denied/unavailable append no task and
produce no fabricated input. The planner query is goal-derived (scalar-backed
search that cannot be extracted stays `DEPENDENCY_UNAVAILABLE`).

## Shadow isolation (zero production writes)

Side-effect-free shadow v2 runs (`app/services/agent/shadow_runtime.py`,
`agents/v2/persistence/shadow_checkpoint.py`) compile their own graph against
an **isolated saver**, use **read-only source adapters**, never write
production state (proven by spies on the real store/repository boundaries plus
DB-backed row-count equality), and never emit outbound events (streaming/SSE,
Telegram/notifications, webhook/publish/queue, chat relay). Shadow mirrors the
primary's authorization (`can_read_people = bool(user.is_superadmin)` —
never hardcoded) and is sampled at `NEXUSRAG_AGENT_V2_SHADOW_PERCENT`
(default `0`, disabled). Cancellation is terminal: cleanup revokes the
shadow's output channel so a late-completing shadow is a no-op.

## Rollout: DB-backed canary, kill switch, v1 default

- **v1 is the default and the rollback path.**
  `NEXUSRAG_AGENT_GRAPH_VERSION` defaults to `"v1"`. For ordinary serving
  selection every failure mode fails closed to v1: kill switch set, control
  row absent/disabled, env/DB disabled, percent 0, empty bucket salt while
  enabled, workspace not allowlisted, Write endpoint, selector exception.
  (This fail-closed claim scopes to ordinary serving selection — it does not
  cover the authenticated superadmin admin override below, which bypasses
  the remaining gates but never escapes the kill switch.)
- **Master switch + DB control.** `NEXUSRAG_AGENT_V2_ENABLED` (default
  `false`) is the environment ceiling; the `agent_rollout_control` row
  (`id=1`, schema v3) is authoritative within it. Deploying the code changes
  no traffic (both default off/0).
- **Deterministic server-owned selection** (`select_canary_arm`, pure):
  kill switch → admin override (authenticated superadmin only, bypasses the
  remaining gates including the Write-endpoint exclusion, never escapes the
  kill switch) → env enabled → DB row enabled → effective percent (min of
  env + DB) → workspace allowlist (env ∩ DB when both set) →
  deterministically-known Write exclusion (supplied at the ingress call
  site, never derived from query content) → deterministic
  `workspace+request+salt` bucket.
  `CANARY_PERCENT=100` promotes **100% of v2-eligible traffic only** — never
  a global replacement of v1.
- **Ineligible traffic stays on v1.** Write endpoints route to v1 before
  bucketing; v2 candidates the Router resolves to `write`/`evaluate` fall
  back to v1 before any capability execution (`V1FallbackRequired`); legal /
  unsupported routes never reach v2.
- **Kill switch.** Setting `kill_switch` on the control row routes all new
  requests to v1, increments the control revision, and cancels active v2
  runs (Redis-distributed cancel; in-process otherwise) without success.
  Admin surfaces: `GET/PUT /api/v1/admin/agent/rollout` (superadmin only),
  `POST /api/v1/admin/agent/runs/{run_id}/cancel`.
- **Metrics + gates.** `agent_rollout_metrics` is append-only (DB trigger
  rejects UPDATE/DELETE); terminal v1/v2 turns emit rows, including
  `security_unobservable` sentinel rows for unobservable verdicts so gates
  count them invalid instead of dropping them. Offline report/gate:
  `backend/scripts/collect_v2_rollout_report.py` /
  `check_v2_rollout_gate.py` (synthetic reports offline; live collect is an
  operational step — see `docs/harness.md`).
- **Staged rollout order:** `shadow 5% → internal workspace canary → 5% →
  25% → 50% → 100%` (of eligible traffic). Each stage needs real
  `agent_rollout_metrics` traffic over the live gate window.

## v1 removal criteria (explicitly NOT met)

Global replacement of v1 is out of scope until Write and evaluate have
approved v2 implementations. v1 stays the default until persisted rollout
control promotes v2. Do not document or imply otherwise.

## Staged-rollout runbook (operational hand-off)

Live preflight/canary need the Compose stack + real traffic and are NOT run
from this worktree session. Exact commands live in `docs/harness.md`
("Phase-3 golden preflight" + "Operational hand-off"). Offline validation
that runs here: `tests/agents/v2`, `tests/api`, `tests/migrations/v2`,
`tests/workers` via the Phase-3 harness script, plus the four static guards
(see `docs/harness.md`).

## Keeping architecture docs in sync

Architecture changes MUST update the doc that describes them **in the same
change**:

| Touch | Also touch |
|---|---|
| Agent graph / v2 ownership / rollout / canary / shadow | `CLAUDE.md` (canonical) + `README.md` section pointer |
| Test / eval / A-B entrypoints, `make` targets | `docs/harness.md` |
| Multi-worker / Redis coordination | `docs/scaling.md` |
| Worker roles / queues / control plane | `docs/workers.md` |
| Embedder / reranker / retrieval read path | `docs/embedding.md` |
| Login / JWT / API keys / admin-only surfaces | `docs/auth.md` |
| v1 graph node/edge reference | `backend/docs/langgraph_architecture.md` |
| Supervisor diagram | `backend/app/services/agent/langgraph_diagram.md` |
| Route permission matrix | `backend/docs/route_permissions.md` |
| New / changed config flag or default | `.env.example` |

Rules: `CLAUDE.md` is canonical — other docs stay purpose-scoped, cross-link
here, and never copy the frozen architecture. A stale doc is a bug.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **AIRAG** (11645 symbols, 23340 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/AIRAG/context` | Codebase overview, check index freshness |
| `gitnexus://repo/AIRAG/clusters` | All functional areas |
| `gitnexus://repo/AIRAG/processes` | All execution flows |
| `gitnexus://repo/AIRAG/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

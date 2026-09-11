# LangGraph v2 Implementation Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the frozen LangGraph v2 architecture through gated benchmark, persistence, fast-path, complex-pilot, and rollout plans while v1 remains the production default.

**Architecture:** This index sequences four self-contained implementation plans. Immutable document revisions gate binding, evidence, coverage, and resume; a selected orchestrator then composes an independent `supervisor_v2.py`; only an external selector can expose v2 after compatibility gates pass.

**Tech Stack:** Python 3.11, Pydantic v2, LangGraph, FastAPI, async SQLAlchemy/PostgreSQL 15, RabbitMQ, MinIO, Chroma, Redis, pytest, React/Vitest, Docker Compose, GitNexus.

**Spec:** `docs/superpowers/specs/2026-09-10-langgraph-v2-contract-first-design.md`

## Global Constraints

- Keep `backend/app/services/agents/supervisor.py` as v1 and create independent `backend/app/services/agents/supervisor_v2.py`.
- Keep chat DB as conversation truth; checkpoints hold execution/interrupt/resume only.
- Never checkpoint current user identity, ACL/workspace scope, deadline, DB/client/service objects, or capability registry.
- Persist raw user text before semantic normalization.
- All factual execution owns a checkpointed `TaskPlan`; direct non-factual responses own no plan.
- Do not implement binding/evidence/coverage/reuse until immutable revision publication passes its gate.
- Keep API, SSE event names/payloads, cancellation, citations, and frontend behavior compatible.
- Keep `.orca/` untracked.
- Before editing an existing symbol, run GitNexus upstream impact for the exact symbol listed by its task. Stop for HIGH/CRITICAL risk and obtain review approval.
- Before every commit run `node .gitnexus/run.cjs detect-changes --scope compare --base-ref main`; stage only paths named by that task.
- Re-run `node .gitnexus/run.cjs analyze` after each structural phase.

## Ordered Plan Suite

1. [`2026-09-11-langgraph-v2-phase0-benchmark.md`](2026-09-11-langgraph-v2-phase0-benchmark.md) — close the pasted-text Write contract blocker, benchmark native LangGraph versus isolated Deep Agents, and pin the winner.
2. [`2026-09-11-langgraph-v2-phase1-foundation.md`](2026-09-11-langgraph-v2-phase1-foundation.md) — isolated models then versioned migration; immutable allocate→build→verify→publish pipeline across every producer/worker/retriever; canonical contracts; evidence governance/GC/checkpointing; and adapters.
3. [`2026-09-11-langgraph-v2-phase2-fast-paths.md`](2026-09-11-langgraph-v2-phase2-fast-paths.md) — independent supervisor, context/binding/router, deterministic fast plans, clarification interrupt/resume, external selector, and SSE compatibility.
4. [`2026-09-11-langgraph-v2-phase3-rollout.md`](2026-09-11-langgraph-v2-phase3-rollout.md) — comparison, People→Document, bounded replan/discovery, session-SSE A/B, replay, shadow, canary, and gradual cutover.

## Hard Gates

| Gate | Required evidence | Blocks |
|---|---|---|
| Phase 0 | committed benchmark report, exact orchestrator plus `langgraph-checkpoint-postgres` and async-driver pins, contract parity | production orchestrator package |
| Schema | isolated advisory-locked DDL/backfill completes on populated legacy DB before v2 ORM registration; exact compatible schema version; v2 excluded from unlocked startup `create_all` | any v2 model-backed deployment or selector/shadow/canary choice |
| Revision publication | fresh upload/reindex/minio event draft→artifact build→verification→atomic publish tests; revision-owned image/table/chunk/vector rows; old revision survives reindex; retrieval filters selected revision | binding, evidence, coverage, reuse |
| Phase 1 | contracts, migrations, snapshots, evidence ACL/retention/audit/GC pass | `supervisor_v2.py` |
| Phase 2 | phase-specific §26 fast tests including named synthesis-overflow/reuse tests, checkpoint/resume, raw-ChatMessage clarification, typed complex-unavailable boundary, admin-only eval endpoint, SSE/API compatibility, v1 default | complex pilots |
| Session-SSE A/B | admin-authenticated arm selection and comparable reports; ordinary client headers ignored | replay/shadow |
| Shadow | writable isolated ephemeral stores, zero production writes/events, schema/ACL parity, quality/latency gates | canary |
| Canary | DB-backed per-request kill switch, deterministic bucket, active-run cancellation, ≥200 samples/arm and 24h zero security violations | percentage ramp |

## Prohibited Sequencing

1. Destructive reindex after immutable references exist.
2. EvidenceUse persistence before its TaskPlan checkpoint is durable.
3. People activation before minimization, encryption, retention, deletion, and audited reads.
4. Selector wiring before `get_supervisor_v2_graph()` exposes direct/finalizer/fast/clarify and a typed complex-unavailable boundary, or before schema/checkpointer readiness checks pass.
5. Deep Agents in production requirements before it wins Phase 0.
6. Shadow execution that writes production state, or that rejects normal graph persistence instead of using writable isolated ephemeral stores.
7. Inner graph/domain streaming of user-facing prose.
8. Multi-worker migration without advisory locking.

## Whole-Program Validation

```bash
git diff --check
node .gitnexus/run.cjs detect-changes --scope compare --base-ref main
docker exec hrag-backend pytest tests/agents tests/migrations tests/services tests/api -q
make test-recall
make test-section
make test-validity
make fe-lint
make fe-build
cd frontend && pnpm test -- ChatPanel.rollback
```

Expected: every command exits 0; v1 remains the default until the Phase 3 rollout task explicitly changes deployment configuration.

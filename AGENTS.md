<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **AIRAG** (6972 symbols, 15610 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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

---

# AIRAG — Agent Guidance

**Canonical guidance lives in [`CLAUDE.md`](CLAUDE.md).** This file used to keep a
second full copy of the architecture; that copy drifted out of date, so it now
just points at the single source of truth. Do NOT re-add a duplicate architecture
description here — update `CLAUDE.md` instead (see its "Keeping architecture docs
in sync" rule).

Start here:

| I need… | Read |
|---------|------|
| Architecture, agent graph, conventions, key config variables | [`CLAUDE.md`](CLAUDE.md) |
| Test / eval / A-B entrypoints (the `make` targets) | [`docs/harness.md`](docs/harness.md) |
| Multi-worker / Redis scale-out runbook | [`docs/scaling.md`](docs/scaling.md) |
| vLLM engines (never restart them), workers, embedding, auth | [`docs/`](docs/) |
| OpenViking memory backend (context-DB, LangGraph memory) | [`docs/openviking.md`](docs/openviking.md) |
| Product overview, quick start | [`README.md`](README.md) |

Two facts worth knowing before you touch anything (both spelled out in `CLAUDE.md`):

- **One agent backend.** The supervisor LangGraph in
  `app/services/agents/supervisor.py` (built by `create_supervisor_graph()`) is the
  only graph. `app/services/agent/` (singular) is a thin compat shim that re-exports
  it as `build_agent_graph` — not a separate architecture. There is no live "legacy"
  loop.
- **The whole stack runs via Docker Compose** (`docker-compose.services.yml` +
  `.vllm.yml` + `.langfuse.yml`). The old top-level run scripts (`run_dev.sh`,
  `run_bk.sh`, …) were removed.
- **The backend request path holds no GPU model.** Retrieval (embed + rerank) runs
  in the `embed-rerank` service (:8090, GPU) and Whisper STT in the `stt` sidecar
  (:8091, CPU); the backend calls both over HTTP (`HRAG_EMBED_RERANK_URL`,
  `STT_PROVIDER=openai`). That is what makes `WEB_CONCURRENCY>1` no longer GPU-bound —
  details in `CLAUDE.md` (Model Services) and `docs/scaling.md`.

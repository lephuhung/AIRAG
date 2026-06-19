"""
Export agent traces (``agent_traces`` table) to JSONL for training a smaller model.

The traces are written at runtime by the dataset-capture layer
(``app/services/agent/trace_collector.py`` + ``AgentTraceService``) whenever
``NEXUSRAG_TRACE_DATASET`` is on. Each row holds one agent run: the supervisor
routing decision, every LLM call (dynamic messages + completion; system prompts
are referenced by hash, not stored), and tool calls + results. PII is already
redacted at write time.

This script reshapes those rows into SFT-style JSONL. Three formats:

  * ``router``    — query (+ light context) → {intent, next_agent, task_plan}.
                    For distilling the supervisor's routing decision.
  * ``tool_use``  — query → the sequence of tool calls (name + args + summary).
                    For distilling tool selection / argument filling.
  * ``answer``    — query (+ retrieved context) → final answer. General QA SFT.
  * ``raw``       — the full trace row as-is (one JSON object per line).

Usage (needs Postgres):
    python -m scripts.export_agent_traces --format router   --out router.jsonl
    python -m scripts.export_agent_traces --format tool_use --out tools.jsonl --only-success
    python -m scripts.export_agent_traces --format answer   --since 2026-06-01 --out qa.jsonl
    python -m scripts.export_agent_traces --format raw      --out all.jsonl

Safe to run repeatedly (read-only).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime

from sqlalchemy import select

from app.core.database import async_session_maker
from app.models.agent_trace import AgentTrace
from app.services.agent.trace_redact import has_residual_pii

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_agent_traces")


def _steps_of(row: AgentTrace, step_type: str) -> list[dict]:
    return [s for s in (row.steps or []) if isinstance(s, dict) and s.get("type") == step_type]


def _to_router(row: AgentTrace) -> dict | None:
    routing = _steps_of(row, "routing")
    sup = next((s for s in routing if s.get("data", {}).get("node") == "supervisor"), None)
    if not sup:
        return None
    d = sup["data"]
    return {
        "messages": [
            {"role": "user", "content": row.original_query or ""},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "intent": d.get("intent"),
                        "next_agent": d.get("next_agent"),
                        "needs_memory": d.get("needs_memory"),
                        "task_plan": d.get("task_plan") or [],
                        "search_mode": d.get("search_mode"),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "meta": {"trace_id": str(row.id), "complexity": row.query_complexity},
    }


def _to_tool_use(row: AgentTrace) -> dict | None:
    tools = _steps_of(row, "tool_call")
    if not tools:
        return None
    calls = [
        {"name": t["data"].get("name"), "args": t["data"].get("args"),
         "result_summary": t["data"].get("result_summary")}
        for t in tools
    ]
    return {
        "messages": [
            {"role": "user", "content": row.original_query or ""},
            {"role": "assistant", "content": json.dumps(calls, ensure_ascii=False)},
        ],
        "meta": {"trace_id": str(row.id), "intent": row.intent, "n_tools": len(calls)},
    }


def _to_answer(row: AgentTrace) -> dict | None:
    if not (row.final_answer or "").strip():
        return None
    # Stitch tool result summaries as the retrieved context the answer relied on.
    context = "\n\n".join(
        t["data"].get("result_summary", "")
        for t in _steps_of(row, "tool_call")
        if t["data"].get("result_summary")
    )
    user = row.original_query or ""
    if context:
        user = f"{user}\n\n[Ngữ cảnh truy hồi]\n{context}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row.final_answer},
        ],
        "meta": {"trace_id": str(row.id), "intent": row.intent},
    }


_FORMATTERS = {
    "router": _to_router,
    "tool_use": _to_tool_use,
    "answer": _to_answer,
}


async def export(args: argparse.Namespace) -> None:
    conds = []
    if args.only_success:
        conds.append(AgentTrace.success.is_(True))
    if args.backend:
        conds.append(AgentTrace.backend == args.backend)
    if args.channel:
        conds.append(AgentTrace.channel == args.channel)
    if args.intent:
        conds.append(AgentTrace.intent == args.intent)
    if args.since:
        conds.append(AgentTrace.created_at >= datetime.fromisoformat(args.since))

    stmt = select(AgentTrace).order_by(AgentTrace.created_at.asc())
    for c in conds:
        stmt = stmt.where(c)

    written = 0
    skipped = 0
    pii_hits = 0
    fmt = _FORMATTERS.get(args.format)

    async with async_session_maker() as db:
        rows = (await db.execute(stmt)).scalars().all()

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            if args.format == "raw":
                rec = {
                    "id": str(row.id),
                    "backend": row.backend,
                    "channel": row.channel,
                    "intent": row.intent,
                    "next_agent": row.next_agent,
                    "query_complexity": row.query_complexity,
                    "success": row.success,
                    "original_query": row.original_query,
                    "final_answer": row.final_answer,
                    "steps": row.steps,
                    "meta": row.meta,
                    "token_usage": row.token_usage,
                    "latency_ms": row.latency_ms,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
            else:
                rec = fmt(row)
                if rec is None:
                    skipped += 1
                    continue

            # Tripwire: a numeric identifier surviving redaction means a PII
            # pattern slipped through — drop the record and count it.
            if has_residual_pii(rec):
                pii_hits += 1
                continue

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        f"Exported {written} record(s) to {args.out} "
        f"(format={args.format}, skipped={skipped}, pii_dropped={pii_hits}, total_rows={len(rows)})"
    )
    if pii_hits:
        logger.warning(f"{pii_hits} record(s) DROPPED for residual PII — investigate redaction.")


def main() -> None:
    p = argparse.ArgumentParser(description="Export agent traces to JSONL for training.")
    p.add_argument("--format", choices=["router", "tool_use", "answer", "raw"], default="raw")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--only-success", action="store_true", help="Only successful runs")
    p.add_argument("--backend", help="Filter by backend (e.g. langgraph)")
    p.add_argument("--channel", help="Filter by channel (web | telegram)")
    p.add_argument("--intent", help="Filter by intent")
    p.add_argument("--since", help="ISO date/datetime lower bound, e.g. 2026-06-01")
    asyncio.run(export(p.parse_args()))


if __name__ == "__main__":
    main()

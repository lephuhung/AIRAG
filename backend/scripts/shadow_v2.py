"""Offline shadow v2 runner (Phase 3, Task 6).

Runs ONE side-effect-free shadow turn against the compiled v2 topology
with an isolated saver + read-only adapters, then prints ONLY redacted
metrics (status/route/task count/timing) — never response content.

Usage (from ``backend/``)::

    python scripts/shadow_v2.py --query "Xin chào" --thread shadow-manual-1
    python scripts/shadow_v2.py --query "Xin chào" --percent-gate 10 --sample 4.2

``--percent-gate``/``--sample`` mirror the production sampling rule
without touching production traffic: the run is skipped (exit 0, reason
printed) when the sample falls outside the gate. No database, no
network, no outbound events on any path.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Make `cd backend && python scripts/shadow_v2.py ...` work without an
# external PYTHONPATH: direct script execution puts `backend/scripts` at
# sys.path[0], so bootstrap the `backend/` parent for the `app` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/shadow_v2.py",
        description="Run one side-effect-free v2 shadow turn (metrics only).",
    )
    parser.add_argument("--query", required=True, help="Raw user query to shadow.")
    parser.add_argument(
        "--thread",
        default="shadow-manual-1",
        help="Thread id for the shadow run (isolated namespace).",
    )
    parser.add_argument(
        "--percent-gate",
        type=float,
        default=100.0,
        help="Sampling gate 0..100 (default 100 = always run).",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=0.0,
        help="Deterministic sample in [0, 100) checked against the gate.",
    )
    return parser


async def _run(query: str, thread: str) -> dict:
    from app.services.agent.shadow_runtime import build_shadow_bundle

    bundle = build_shadow_bundle(raw_query=query, thread_id=thread)
    metrics = await bundle.run()
    return metrics.redacted()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from app.services.agent.shadow_runtime import should_run_shadow

    if not should_run_shadow(percent=args.percent_gate, sample=args.sample):
        print(
            json.dumps(
                {
                    "skipped": True,
                    "reason": "sample outside percent gate",
                    "percent_gate": args.percent_gate,
                    "sample": args.sample,
                }
            )
        )
        return 0
    redacted = asyncio.run(_run(args.query, args.thread))
    print(json.dumps(redacted))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

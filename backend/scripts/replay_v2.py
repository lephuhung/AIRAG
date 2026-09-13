"""Replay a recorded session-SSE transcript through the SHARED evaluator.

Phase-3 Task 1 companion to ``scripts/ab_eval.py``: re-judges a recorded v2
transcript (e.g. a golden session capture) with the SAME preflight
evaluator version both arms use. The evaluator version is IMPORTED from
``ab_eval`` — never redefined — so v1/v2 comparisons cannot drift across
evaluators.

Offline usage (no backend, no LLM)::

    python -m scripts.replay_v2 --transcript transcript.json --arm v2
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from scripts.ab_eval import (
    EVALUATOR_VERSION,
    build_arm_report,
    build_case_record,
    collect_terminal,
    evaluate_functional,
    evaluate_output,
)

__all__ = [
    "EVALUATOR_VERSION",
    "replay_transcript",
    "replay_to_report",
    "load_report",
    "main",
]


def replay_transcript(events: list[dict], *, arm: str = "v2") -> dict:
    """Judge a recorded transcript with the shared preflight evaluator.

    Collapses the named SSE events to the single terminal event, then runs
    ``ab_eval.evaluate_output`` — the exact function the ``run`` path uses —
    so the persisted ``evaluator_version`` matches by construction.
    """
    terminal = collect_terminal(events)
    data = terminal.get("data") or {}
    return evaluate_output(
        answer=str(data.get("answer", "")),
        sources=list(data.get("sources", [])),
        status=terminal["event"],
        arm=arm,
    )


def replay_to_report(
    transcripts: dict[str, list[dict]],
    *,
    arm: str = "v2",
    cases: dict[str, dict] | None = None,
) -> dict:
    """Judge many recorded transcripts into one evaluator-versioned report.

    Each case record goes through the SHARED ``build_case_record``
    serializer, so replay reports carry the same keys as run reports
    (``sources`` provenance, ``has_answer``, golden functional fields).
    ``cases`` optionally maps query ids to golden case expectations for
    document/article/negative scoring; without it only status-level
    functional fields apply. Raw answer text is used transiently for
    scoring and never persisted.
    """
    golden = cases or {}
    records: list[dict] = []
    for query_id, events in transcripts.items():
        try:
            terminal = collect_terminal(events)
            data = terminal.get("data") or {}
            answer = str(data.get("answer", ""))
            judged = evaluate_output(
                answer=answer,
                sources=list(data.get("sources", [])),
                status=terminal["event"],
                arm=arm,
            )
            functional = evaluate_functional(
                golden.get(query_id, {}),
                citations=judged["citations"],
                answer=answer,
                status=judged["status"],
                sources=judged["sources"],
            )
            records.append(
                build_case_record(
                    query_id=query_id, arm=arm, evaluation=judged,
                    functional=functional,
                )
            )
        except ValueError as exc:
            evaluation = evaluate_output(answer="", sources=[],
                                         status=f"error: {exc}", arm=arm)
            records.append(
                build_case_record(
                    query_id=query_id, arm=arm, evaluation=evaluation,
                    functional=evaluate_functional(
                        golden.get(query_id, {}), citations=[], answer="",
                        status="error",
                    ),
                )
            )
    return build_arm_report(arm=arm, cases=records)


def load_report(path: str) -> dict:
    """Load a report JSON file (``run`` or ``replay`` output)."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay recorded session-SSE transcripts (shared evaluator)."
    )
    parser.add_argument("--transcript", required=True, help="Transcript JSON.")
    parser.add_argument("--arm", default="v2", choices=("v1", "v2"))
    parser.add_argument("--out", default=None, help="Report JSON path.")
    args = parser.parse_args(argv)

    payload: Any = load_report(args.transcript)
    if isinstance(payload, dict) and "events" in payload:
        transcripts = {payload.get("query_id", "query"): payload["events"]}
    elif isinstance(payload, dict):
        transcripts = {
            key: value
            for key, value in payload.items()
            if isinstance(value, list)
        }
    elif isinstance(payload, list):
        transcripts = {"query": payload}
    else:
        raise SystemExit("unrecognised transcript shape")
    report = replay_to_report(transcripts, arm=args.arm)
    dumped = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(dumped)
    else:
        print(dumped)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Evidence + revision artifact GC worker — Phase 1D Task 9.

One-shot / scheduled driver for the two independent GC batchers:

    Predicate A  app.services.agents.v2.evidence_store.gc
                 ``run_evidence_payload_gc_batch``   (evidence payload only)
    Predicate B  app.services.agents.v2.persistence.revision_gc
                 ``run_revision_artifact_gc_batch``  (revision artifacts only)

Each batcher runs in its **own transaction** so a failure in one does not abort
the other (and so each batch's transaction-scoped advisory lock is released
promptly). A failing batch is logged and retried on the next scheduled run; the
other batch still commits.

Usage::

    python -m app.workers.evidence_gc_worker --once --batch-size 100
    python -m app.workers.evidence_gc_worker --batch-size 100 --interval 3600

The Docker Compose ``evidence-gc`` service runs the loop form on an interval.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Optional

from app.core.database import AsyncSessionLocal
from app.services.agents.v2.evidence_store.gc import (
    EvidencePayloadGcResult,
    run_evidence_payload_gc_batch,
)
from app.services.agents.v2.persistence.revision_gc import (
    RevisionArtifactGcResult,
    run_revision_artifact_gc_batch,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE: int = 100
DEFAULT_INTERVAL_SECONDS: float = 3600.0


@dataclass(frozen=True)
class GcRunSummary:
    """Outcome of one :func:`run_both_batches` pass."""

    evidence: Optional[EvidencePayloadGcResult] = None
    revision: Optional[RevisionArtifactGcResult] = None
    #: Expired retention leases marked released by this pass.
    leases_released: int = 0
    #: ``"<batch>: <exception repr>"`` for every batch that failed.
    failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures


async def sweep_expired_leases(session) -> int:
    """Release expired retention leases; the caller owns the transaction.

    Expiry already unblocks GC (the predicate requires ``expires_at > now()``),
    so this is the hygiene sweep that also gives an orphan lease (lease commit
    succeeded, checkpoint failed) a terminal ``expired`` release reason. Without
    a production caller such leases would stay ``released_at IS NULL`` forever.
    """
    from app.services.agents.v2.persistence.retention_leases import (
        RevisionRetentionLeaseRepository,
    )

    return await RevisionRetentionLeaseRepository(session).sweep_expired()


async def run_both_batches(
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    session_factory: Optional[Callable] = None,
    evidence_batcher: Optional[Callable] = None,
    revision_batcher: Optional[Callable] = None,
    lease_sweeper: Optional[Callable] = None,
) -> GcRunSummary:
    """Run Predicate A then Predicate B then the lease sweep, each in its own
    transaction.

    ``session_factory``/``evidence_batcher``/``revision_batcher``/
    ``lease_sweeper`` are injectable so the worker's transaction separation and
    failure isolation are testable without external services.
    """
    factory = session_factory or AsyncSessionLocal
    evidence_batcher = evidence_batcher or run_evidence_payload_gc_batch
    revision_batcher = revision_batcher or run_revision_artifact_gc_batch
    lease_sweeper = lease_sweeper or sweep_expired_leases

    evidence_result: Optional[EvidencePayloadGcResult] = None
    revision_result: Optional[RevisionArtifactGcResult] = None
    leases_released = 0
    failures: list[str] = []

    # A then B as SEPARATE transactions: a failure in A must not abort B (and
    # must not discard B's committed work, nor vice versa).
    for name, batcher in (
        ("evidence", evidence_batcher),
        ("revision", revision_batcher),
    ):
        try:
            async with factory() as session:
                async with session.begin():
                    result = await batcher(session, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001 — one batch must not kill the other
            logger.exception("[evidence_gc] %s batch failed (continuing)", name)
            failures.append(f"{name}: {exc!r}")
            continue
        if name == "evidence":
            evidence_result = result
        else:
            revision_result = result

    # Retention-lease hygiene sweep — its own transaction so a lease-table
    # problem cannot abort either artifact batch (and vice versa).
    try:
        async with factory() as session:
            async with session.begin():
                leases_released = await lease_sweeper(session)
    except Exception as exc:  # noqa: BLE001 — must not kill the artifact batches
        logger.exception("[evidence_gc] lease sweep failed (continuing)")
        failures.append(f"leases: {exc!r}")

    return GcRunSummary(
        evidence=evidence_result,
        revision=revision_result,
        leases_released=leases_released,
        failures=tuple(failures),
    )


def _log_summary(summary: GcRunSummary) -> None:
    evidence = summary.evidence
    revision = summary.revision
    logger.info(
        "[evidence_gc] pass complete: evidence_purged=%s revision_reclaimed=%s "
        "leases_released=%s failures=%s",
        evidence.purged if evidence else "-",
        revision.reclaimed if revision else "-",
        summary.leases_released,
        list(summary.failures),
    )


async def _run(args: argparse.Namespace) -> int:
    if args.once:
        summary = await run_both_batches(batch_size=args.batch_size)
        _log_summary(summary)
        return 0 if summary.ok else 1

    while True:
        summary = await run_both_batches(batch_size=args.batch_size)
        _log_summary(summary)
        await asyncio.sleep(args.interval)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.workers.evidence_gc_worker",
        description="Run the evidence payload + revision artifact GC batches.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run one pass and exit (exit 1 if any batch failed)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"max rows per batch (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=(
            "seconds between passes when not --once "
            f"(default {DEFAULT_INTERVAL_SECONDS:g})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

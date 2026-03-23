"""
Worker Metrics
==============
Lightweight, in-process metrics for worker observability.

Tracks per-queue counters (processed / failed) and processing-time
histogram.  Logs a summary every ``LOG_INTERVAL`` seconds so operators
can tail the worker log file and get a quick health overview without
an external metrics stack.

Usage (already wired into ``connection.consume``):
    from app.workers.metrics import worker_metrics
    worker_metrics.record_success("hrag.parse", elapsed=1.23)
    worker_metrics.record_failure("hrag.embed", elapsed=0.45)
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

LOG_INTERVAL = 60  # seconds between periodic metric summaries


@dataclass
class _QueueStats:
    processed: int = 0
    failed: int = 0
    total_time: float = 0.0
    last_log_time: float = field(default_factory=time.monotonic)


class WorkerMetrics:
    """Thread-safe (single-threaded asyncio) per-queue counter tracker."""

    def __init__(self) -> None:
        self._stats: dict[str, _QueueStats] = defaultdict(_QueueStats)
        self._log_task_started = False

    def record_success(self, queue_name: str, elapsed: float) -> None:
        s = self._stats[queue_name]
        s.processed += 1
        s.total_time += elapsed
        self._maybe_log(queue_name)

    def record_failure(self, queue_name: str, elapsed: float) -> None:
        s = self._stats[queue_name]
        s.failed += 1
        s.total_time += elapsed
        self._maybe_log(queue_name)

    def _maybe_log(self, queue_name: str) -> None:
        s = self._stats[queue_name]
        now = time.monotonic()
        if now - s.last_log_time >= LOG_INTERVAL:
            total = s.processed + s.failed
            avg_ms = (s.total_time / total * 1000) if total else 0
            logger.info(
                f"[metrics] {queue_name}: "
                f"processed={s.processed} failed={s.failed} "
                f"avg_time={avg_ms:.0f}ms"
            )
            s.last_log_time = now

    def get_stats(self, queue_name: str) -> dict:
        """Return stats dict for a queue (useful for health-check endpoints)."""
        s = self._stats.get(queue_name)
        if s is None:
            return {"processed": 0, "failed": 0, "avg_time_ms": 0}
        total = s.processed + s.failed
        return {
            "processed": s.processed,
            "failed": s.failed,
            "avg_time_ms": round(s.total_time / total * 1000, 1) if total else 0,
        }

    def get_all_stats(self) -> dict[str, dict]:
        """Return stats for all queues."""
        return {q: self.get_stats(q) for q in self._stats}


# Module-level singleton
worker_metrics = WorkerMetrics()

"""
Worker Runner
=============
Entrypoint for worker containers.  Select worker type via WORKER_TYPE env var:

  WORKER_TYPE=parse    → parse worker
  WORKER_TYPE=embed    → embed worker
  WORKER_TYPE=caption  → caption worker
  WORKER_TYPE=kg       → KG worker (scale by workspace count)
  WORKER_TYPE=memory   → Graphiti personal-memory worker

Usage:
  python -m app.workers.runner

Docker CMD:
  CMD ["python", "-m", "app.workers.runner"]

Health server (port 8081):
  GET /health → liveness (worker alive?)
  GET /ready  → readiness (can accept messages?)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from app.core.config import settings
from app.queue import connection as mq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _get_worker_type() -> str:
    """Get worker type from environment."""
    return os.getenv("WORKER_TYPE", "").lower()


async def _run_health_server(port: int) -> None:
    """Run the worker health server (liveness + readiness probes)."""
    from app.workers.health_server import run_health_server as _run
    await _run(port)


async def _run_parse_worker() -> None:
    from app.workers.parse_worker import handle_parse
    await mq.consume(
        mq.EXCHANGE_PARSE, mq.QUEUE_PARSE, "parse",
        handle_parse, prefetch_count=settings.WORKER_PREFETCH_PARSE,
    )


async def _run_embed_worker() -> None:
    from app.workers.embed_worker import handle_embed
    await mq.consume(
        mq.EXCHANGE_EMBED, mq.QUEUE_EMBED, "embed",
        handle_embed, prefetch_count=settings.WORKER_PREFETCH_EMBED,
    )


async def _run_caption_worker() -> None:
    from app.workers.caption_worker import handle_caption
    await mq.consume(
        mq.EXCHANGE_CAPTION, mq.QUEUE_CAPTION, "caption",
        handle_caption, prefetch_count=settings.WORKER_PREFETCH_CAPTION,
    )


async def _run_memory_worker() -> None:
    from app.workers.memory_worker import handle_memory
    await mq.consume(
        mq.EXCHANGE_MEMORY, mq.QUEUE_MEMORY, "memory",
        handle_memory, prefetch_count=settings.WORKER_PREFETCH_MEMORY,
    )


async def _run_kg_worker() -> None:
    """
    KG worker: dynamically subscribes to all workspace queues.

    On startup, scans the DB for all existing workspaces and starts
    a consumer per workspace.  A background polling loop runs every
    WORKER_KG_POLL_INTERVAL seconds to discover new workspaces and
    restart dead consumers for existing ones — no restart required.
    """
    from app.workers.kg_worker import handle_kg

    # Ensure all SQLAlchemy models are registered before querying
    import app.models  # noqa: F401 — registers DocumentType and all relationships

    # Mark DB as connected (models import verifies DB connection works)
    try:
        from app.workers.health_server import update_worker_state
        update_worker_state(db=True)
    except Exception:
        pass

    # Fetch all existing workspace IDs
    from app.core.database import async_session_maker
    from app.models.knowledge_base import KnowledgeBase
    from sqlalchemy import select

    active_workspaces: set[int] = set()
    consumer_tasks: dict[int, asyncio.Task] = {}  # workspace_id -> consumer task

    # Circuit breaker for polling failures
    _circuit_open: bool = False
    _circuit_failures: int = 0
    _CIRCUIT_THRESHOLD: int = 3
    _CIRCUIT_EXTENDED_INTERVAL: int = 120

    async def _start_consumer_for(wid: int) -> None:
        """Start (or restart) a consumer for a single workspace, cancelling any existing task."""
        if wid in consumer_tasks:
            consumer_tasks[wid].cancel()
            try:
                await consumer_tasks[wid]
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[kg_runner] Previous consumer for workspace {wid} exited with: {e}")

        active_workspaces.add(wid)
        task = asyncio.create_task(
            mq.consume_kg(wid, handle_kg),
            name=f"kg-consumer-ws-{wid}",
        )
        consumer_tasks[wid] = task
        logger.info(f"[kg_runner] Started (or restarted) consumer for workspace {wid}")

    async def _start_consumers_for(workspace_ids: list[int]) -> None:
        """Start consumers for workspaces not already tracked."""
        for wid in workspace_ids:
            if wid not in active_workspaces:
                await _start_consumer_for(wid)

    async def _poll_workspaces() -> None:
        """
        Periodically:
          1. Discover new workspaces and start consumers.
          2. Check consumer task health (task.done()); restart dead consumers.
          3. Stop consumers for workspaces deleted from the DB.

        One DB query per cycle — no RabbitMQ Management API polling (the old
        per-queue HTTP check could never trigger a restart anyway; task.done()
        already covers consumer death, and DB diffing covers deletion).
        """
        nonlocal _circuit_open, _circuit_failures

        fast_poll = int(os.getenv("WORKER_KG_POLL_INTERVAL", "30"))
        slow_poll = 300  # 5 min when no workspaces

        while True:
            if _circuit_open:
                poll_interval = _CIRCUIT_EXTENDED_INTERVAL
            else:
                poll_interval = fast_poll if active_workspaces else slow_poll
            await asyncio.sleep(poll_interval)
            try:
                async with async_session_maker() as db:
                    result = await db.execute(select(KnowledgeBase.id))
                    current_ids = {row[0] for row in result.all()}

                # Start consumers for new workspaces
                new_ids = [wid for wid in current_ids if wid not in active_workspaces]
                if new_ids:
                    logger.info(f"[kg_runner] New workspaces detected: {new_ids}")
                    await _start_consumers_for(new_ids)

                # Stop consumers for workspaces deleted from the DB
                for wid in list(active_workspaces - current_ids):
                    logger.info(f"[kg_runner] Workspace {wid} deleted — stopping consumer")
                    active_workspaces.discard(wid)
                    task = consumer_tasks.pop(wid, None)
                    if task:
                        task.cancel()

                # Check consumer task health; restart dead consumers
                for wid in list(active_workspaces):
                    task = consumer_tasks.get(wid)
                    if task and task.done():
                        exc = None if task.cancelled() else task.exception()
                        logger.warning(
                            f"[kg_runner] Workspace {wid} consumer died"
                            f"{f': {exc}' if exc else ''} — restarting"
                        )
                        await _start_consumer_for(wid)

                # Reset circuit breaker on successful poll
                if _circuit_open and _circuit_failures > 0:
                    logger.info(f"[kg_runner] Circuit breaker CLOSED — restored normal polling")
                    _circuit_open = False
                    _circuit_failures = 0

            except Exception as e:
                _circuit_failures += 1
                if _circuit_failures >= _CIRCUIT_THRESHOLD:
                    _circuit_open = True
                    logger.warning(
                        f"[kg_runner] Circuit breaker OPEN — polling every "
                        f"{_CIRCUIT_EXTENDED_INTERVAL}s (consecutive failures: {_circuit_failures})"
                    )
                logger.warning(f"[kg_runner] Workspace poll failed: {e}")

    # ── Initial startup: consume all existing workspaces ───────────────────
    async with async_session_maker() as db:
        result = await db.execute(select(KnowledgeBase.id))
        workspace_ids: list[int] = [row[0] for row in result.all()]

    logger.info(f"[kg_runner] Starting consumers for workspaces: {workspace_ids}")
    await _start_consumers_for(workspace_ids)

    # ── Start background poller for new workspaces and dead-consumer detection
    poller = asyncio.create_task(
        _poll_workspaces(), name="kg-workspace-poller"
    )

    if not workspace_ids:
        logger.info(
            "[kg_runner] No workspaces yet — polling every "
            f"{settings.WORKER_KG_POLL_INTERVAL}s for new ones"
        )

    # Block on the poller only: it supervises the consumer tasks (restarts
    # dead ones, retrieves their exceptions). Gathering the consumers here
    # would tear the whole worker down when a single consumer raised.
    # On cancellation (control-plane pause / graceful shutdown) the child
    # consumer tasks must die with us — they are independent tasks that would
    # otherwise keep consuming while the worker is "paused".
    try:
        await poller
    finally:
        poller.cancel()
        for t in consumer_tasks.values():
            t.cancel()
        await asyncio.gather(poller, *consumer_tasks.values(), return_exceptions=True)


_WORKER_MAP = {
    "parse":   _run_parse_worker,
    "embed":   _run_embed_worker,
    "caption": _run_caption_worker,
    "kg":      _run_kg_worker,
    "memory":  _run_memory_worker,
}


# ══════════════════════════════════════════════════════════════════════════════
# Control-plane (pause / resume / restart / set_prefetch from the admin UI)
# ══════════════════════════════════════════════════════════════════════════════
class _WorkerController:
    """
    Owns the consume-loop task and reacts to hrag.control commands.

    - pause:        cancel the consume task (message in flight drains; unacked
                    messages stay queued in RabbitMQ). Container stays alive.
    - resume:       recreate the consume task.
    - restart:      resolve the shutdown future — main() drains and the process
                    exits; the container's restart policy (always) revives it.
    - set_prefetch: mutate the in-memory settings value and bounce the consume
                    task. Reverts to the env value on next container restart.
    """

    def __init__(self, worker_type: str, stop: asyncio.Future) -> None:
        self.worker_type = worker_type
        self._stop = stop
        self.paused = False
        self.runner: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        # Tasks we cancelled on purpose — main() must not treat their
        # completion as a crash.
        self._intentional: set[asyncio.Task] = set()

    def start(self) -> None:
        self.runner = asyncio.create_task(
            _WORKER_MAP[self.worker_type](), name=f"consume-{self.worker_type}"
        )

    def was_intentional(self, task: asyncio.Task) -> bool:
        return task in self._intentional

    async def _stop_runner(self, drain_timeout: float) -> None:
        task = self.runner
        if task is None:
            return
        self._intentional.add(task)
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=drain_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.warning(f"[control] consume task exited with: {e}")
        self.runner = None

    async def shutdown(self, drain_timeout: float) -> None:
        """Graceful drain used by the SIGTERM / restart path."""
        async with self._lock:
            await self._stop_runner(drain_timeout)

    async def handle_command(self, payload: dict) -> None:
        from app.workers.health_server import update_worker_state

        command = payload.get("command", "")
        drain_timeout = float(os.getenv("WORKER_DRAIN_TIMEOUT", "30"))

        async with self._lock:
            if command == "pause":
                if self.paused:
                    logger.info("[control] pause — already paused")
                    return
                logger.info("[control] PAUSE — draining in-flight message, then idle")
                await self._stop_runner(drain_timeout)
                self.paused = True
                update_worker_state(paused=True)

            elif command == "resume":
                if not self.paused and self.runner is not None and not self.runner.done():
                    logger.info("[control] resume — already consuming")
                    return
                logger.info("[control] RESUME — restarting consume loop")
                self.paused = False
                update_worker_state(paused=False)
                self.start()

            elif command == "restart":
                logger.info(
                    "[control] RESTART — graceful shutdown; container restart "
                    "policy will bring this worker back up"
                )
                if not self._stop.done():
                    self._stop.set_result(None)

            elif command == "set_prefetch":
                try:
                    n = int(payload.get("prefetch", 0))
                except (TypeError, ValueError):
                    n = 0
                if not 1 <= n <= 64:
                    logger.warning(f"[control] set_prefetch ignored — invalid value {payload.get('prefetch')!r}")
                    return
                attr = f"WORKER_PREFETCH_{self.worker_type.upper()}"
                if not hasattr(settings, attr):
                    logger.warning(f"[control] set_prefetch ignored — no setting {attr}")
                    return
                setattr(settings, attr, n)
                logger.info(f"[control] SET_PREFETCH {attr}={n} — bouncing consume loop")
                await self._stop_runner(drain_timeout)
                if not self.paused:
                    self.start()

            else:
                logger.warning(f"[control] unknown command: {payload}")


async def _run_control_listener(worker_type: str, controller: _WorkerController) -> None:
    """Consume hrag.control commands forever; reconnect on failure."""
    while True:
        try:
            await mq.consume_control(worker_type, controller.handle_command)
            logger.warning("[control] listener stream ended — reconnecting in 5s")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[control] listener error: {e} — reconnecting in 5s")
        await asyncio.sleep(5)


async def main() -> None:
    worker_type = _get_worker_type()
    if worker_type not in _WORKER_MAP:
        logger.error(
            f"WORKER_TYPE='{worker_type}' is not valid. "
            f"Choose from: {list(_WORKER_MAP)}"
        )
        sys.exit(1)

    logger.info(f"Starting worker: WORKER_TYPE={worker_type}")

    # ── Health server for liveness/readiness probes ───────────────────────
    # Runs on port 8081 alongside the worker consume loop
    health_task: asyncio.Task | None = None
    try:
        from app.workers.health_server import update_worker_state
        update_worker_state(worker_type=worker_type)

        health_port = int(os.getenv("WORKER_HEALTH_PORT", "8081"))
        health_task = asyncio.create_task(_run_health_server(health_port))
        # NOTE: ready=True and rabbitmq=True are set by mq.consume() after connections are established
        logger.info(f"Health server started on port {health_port}")
    except Exception as e:
        logger.warning(f"Health server failed to start (non-fatal): {e}")

    # ── Verify DB connection and mark as ready ─────────────────────────────
    try:
        import app.models  # noqa: F401 — verifies DB connection
        from app.workers.health_server import update_worker_state
        update_worker_state(db=True)
        logger.info("Database connection verified")
    except Exception as db_err:
        logger.warning(f"Database connection check failed (non-fatal): {db_err}")

    # ── Eager model loading for workers ──────────────────────────────────
    if settings.HRAG_EAGER_MODEL_LOADING:
        try:
            from app.services.models.loader import preload_worker_models
            preload_worker_models(worker_type)
        except Exception as _preload_err:
            logger.warning(f"Worker model pre-load failed (non-fatal): {_preload_err}")

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    drain_timeout = float(os.getenv("WORKER_DRAIN_TIMEOUT", "30"))  # seconds

    def _shutdown(sig: signal.Signals) -> None:
        logger.info(f"Received {sig.name} — initiating graceful shutdown")
        if not stop.done():
            stop.set_result(None)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown, sig)

    controller = _WorkerController(worker_type, stop)
    controller.start()
    control_task = asyncio.create_task(
        _run_control_listener(worker_type, controller), name="control-listener"
    )
    stop_task = asyncio.ensure_future(stop)

    # ── Supervise: shutdown signal / control-plane restarts / crashes ──────
    while True:
        runner = controller.runner
        aws = [stop_task] + ([runner] if runner is not None else [])
        done, _ = await asyncio.wait(
            aws,
            return_when=asyncio.FIRST_COMPLETED,
            # While paused there is no consume task — poll so a resume-created
            # task gets picked up for supervision.
            timeout=None if runner is not None else 2.0,
        )
        if stop_task in done:
            break
        if runner is not None and runner.done():
            if controller.was_intentional(runner):
                continue  # pause / set_prefetch bounce — controller owns it
            # The consume loop ended WITHOUT a shutdown signal — crash or
            # abandon. Log loudly and exit non-zero so the container restarts
            # the worker, instead of the previous behavior: silent exit.
            exc = None if runner.cancelled() else runner.exception()
            if exc is not None:
                logger.critical(f"Worker loop crashed: {exc!r}", exc_info=exc)
            else:
                logger.critical("Worker loop exited unexpectedly (no exception)")
            stop_task.cancel()
            control_task.cancel()
            if health_task is not None:
                health_task.cancel()
            try:
                from app.workers.health_server import update_worker_state
                update_worker_state(ready=False)
            except Exception:
                pass
            await mq.close_connection()
            sys.exit(1)

    # SIGTERM or control-plane restart — graceful drain: let the current
    # message finish or time out, then exit (the container restart policy
    # revives us on `restart`; on `docker stop` it stays down).
    logger.info(f"Graceful drain: waiting up to {drain_timeout}s for in-flight message...")
    control_task.cancel()
    await controller.shutdown(drain_timeout)

    # Shutdown health server
    if health_task is not None:
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass

    # Mark not ready before closing connections
    try:
        from app.workers.health_server import update_worker_state
        update_worker_state(ready=False)
    except Exception:
        pass

    await mq.close_connection()
    logger.info("Worker stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())

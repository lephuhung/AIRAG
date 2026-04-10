"""
Worker Runner
=============
Entrypoint for worker containers.  Select worker type via WORKER_TYPE env var:

  WORKER_TYPE=parse    → parse worker   (4 replicas recommended)
  WORKER_TYPE=caption  → caption worker (2 replicas recommended)
  WORKER_TYPE=kg       → KG worker      (scale by workspace count)

Usage:
  python -m app.workers.runner

Docker CMD:
  CMD ["python", "-m", "app.workers.runner"]
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

    # Fetch all existing workspace IDs
    from app.core.database import async_session_maker
    from app.models.knowledge_base import KnowledgeBase
    from sqlalchemy import select

    active_workspaces: set[int] = set()
    consumer_tasks: dict[int, asyncio.Task] = {}  # workspace_id -> consumer task

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
          2. Check RabbitMQ queue consumer counts; restart consumers
             that have died (consumer count dropped to 0).
        """
        import aiohttp

        poll_interval = settings.WORKER_KG_POLL_INTERVAL
        while True:
            await asyncio.sleep(poll_interval)
            try:
                async with async_session_maker() as db:
                    result = await db.execute(select(KnowledgeBase.id))
                    current_ids = [row[0] for row in result.all()]

                # Start consumers for new workspaces
                new_ids = [wid for wid in current_ids if wid not in active_workspaces]
                if new_ids:
                    logger.info(f"[kg_runner] New workspaces detected: {new_ids}")
                    await _start_consumers_for(new_ids)

                # Check consumer health via RabbitMQ Management API
                import httpx
                for wid in list(active_workspaces):
                    queue_name = f"hrag.kg.{wid}"
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as session:
                            url = (
                                f"{settings.RABBITMQ_MANAGEMENT_URL}/api/queues/"
                                f"%2f/{queue_name}"
                            )
                            resp = await session.get(
                                url,
                                auth=httpx.Auth(
                                    settings.RABBITMQ_MANAGEMENT_USER,
                                    settings.RABBITMQ_MANAGEMENT_PASSWORD,
                                ),
                            )
                            if resp.status_code == 200:
                                q_info = resp.json()
                                consumer_count = q_info.get("consumers", 0)
                                if consumer_count == 0:
                                    logger.warning(
                                        f"[kg_runner] Workspace {wid} queue has 0 consumers — restarting"
                                    )
                                    await _start_consumer_for(wid)
                            elif resp.status_code == 404:
                                # Queue gone — workspace deleted; remove from active set
                                logger.info(f"[kg_runner] Workspace {wid} queue not found — removing")
                                active_workspaces.discard(wid)
                                consumer_tasks.pop(wid, None)
                    except Exception as poll_err:
                        logger.debug(f"[kg_runner] Queue health check failed for {queue_name}: {poll_err}")

            except Exception as e:
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

    # Wait for all consumer tasks + poller (they run indefinitely)
    all_tasks = list(consumer_tasks.values()) + [poller]
    await asyncio.gather(*all_tasks)


_WORKER_MAP = {
    "parse":   _run_parse_worker,
    "embed":   _run_embed_worker,
    "caption": _run_caption_worker,
    "kg":      _run_kg_worker,
}


async def main() -> None:
    worker_type = os.getenv("WORKER_TYPE", "").lower()
    if worker_type not in _WORKER_MAP:
        logger.error(
            f"WORKER_TYPE='{worker_type}' is not valid. "
            f"Choose from: {list(_WORKER_MAP)}"
        )
        sys.exit(1)

    logger.info(f"Starting worker: WORKER_TYPE={worker_type}")

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

    def _shutdown(sig: signal.Signals) -> None:
        logger.info(f"Received {sig.name} — shutting down worker")
        if not stop.done():
            stop.set_result(None)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown, sig)

    runner = asyncio.create_task(_WORKER_MAP[worker_type]())
    await asyncio.wait(
        [runner, asyncio.ensure_future(stop)],
        return_when=asyncio.FIRST_COMPLETED,
    )
    runner.cancel()
    await mq.close_connection()
    logger.info("Worker stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())

"""
Worker Health Server
====================
Lightweight HTTP server for container health/readiness probes.
Runs alongside the worker in the same container.

Usage:
  python -m app.workers.health_server

Health endpoint:  GET /health  → liveness (is worker alive?)
Ready endpoint:    GET /ready   → readiness (is worker ready to accept messages?)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time

logger = logging.getLogger(__name__)

# Module-level state (updated by runner)
_worker_type: str = "unknown"
_uptime_start: float = time.monotonic()
_is_ready: bool = False

# RabbitMQ connection state
_rabbitmq_connected: bool = False
_db_connected: bool = False


def update_worker_state(
    worker_type: str | None = None,
    ready: bool | None = None,
    rabbitmq: bool | None = None,
    db: bool | None = None,
) -> None:
    """Update worker state flags. Called by runner.py during lifecycle."""
    global _worker_type, _is_ready, _rabbitmq_connected, _db_connected
    if worker_type is not None:
        _worker_type = worker_type
    if ready is not None:
        _is_ready = ready
    if rabbitmq is not None:
        _rabbitmq_connected = rabbitmq
    if db is not None:
        _db_connected = db


async def health_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle HTTP health/ready requests."""
    try:
        request_line = await reader.readline()
        if not request_line:
            writer.close()
            return

        method, path, _ = request_line.decode().split()

        if path == "/health":
            status = "ok"
            code = 200
            body = f'{{"status":"{status}","worker_type":"{_worker_type}","uptime_seconds":{time.monotonic() - _uptime_start:.1f}}}'
        elif path == "/ready":
            if _is_ready and _rabbitmq_connected:
                status = "ready"
                code = 200
            else:
                status = "not_ready"
                code = 503
            body = (
                f'{{"ready":{str(_is_ready).lower()},'
                f'"rabbitmq":{str(_rabbitmq_connected).lower()},'
                f'"db":{str(_db_connected).lower()},'
                f'"worker_type":"{_worker_type}"}}'
            )
        else:
            code = 404
            body = '{"error":"not_found"}'

        writer.write(f"HTTP/1.1 {code} OK\r\n".encode())
        writer.write(f"Content-Length: {len(body)}\r\n".encode())
        writer.write("Content-Type: application/json\r\n".encode())
        writer.write("Connection: close\r\n".encode())
        writer.write("\r\n".encode())
        writer.write(body.encode())
        await writer.drain()
    except Exception as e:
        logger.debug(f"Health handler error: {e}")
    finally:
        writer.close()


async def run_health_server(port: int = 8081) -> None:
    """Run the health server on the specified port."""
    server = await asyncio.start_server(health_handler, "0.0.0.0", port)
    logger.info(f"Worker health server listening on port {port}")

    # Graceful shutdown on SIGTERM
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    def _shutdown(sig: signal.Signals) -> None:
        logger.info(f"Health server received {sig.name}")
        stop.set_result(None)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown, sig)

    async with server:
        await stop


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Worker Health Server")
    parser.add_argument("--port", type=int, default=8081, help="Port for health server")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    await run_health_server(args.port)


if __name__ == "__main__":
    asyncio.run(main())
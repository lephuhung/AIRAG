"""
RabbitMQ Management HTTP API Proxy
===================================
Async HTTP client that wraps the RabbitMQ Management REST API (port 15672).
The backend acts as a proxy so the frontend never talks to RabbitMQ directly.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Virtual host — default "/" is encoded as %2F in the management API
_VHOST = "%2F"


class RabbitMQManagement:
    """Async proxy for the RabbitMQ Management HTTP API."""

    def __init__(self) -> None:
        self._base = settings.RABBITMQ_MANAGEMENT_URL.rstrip("/")
        self._auth = (settings.RABBITMQ_MANAGEMENT_USER, settings.RABBITMQ_MANAGEMENT_PASS)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base,
            auth=self._auth,
            timeout=10.0,
        )

    # ------------------------------------------------------------------
    # Cluster overview
    # ------------------------------------------------------------------
    async def get_overview(self) -> dict[str, Any]:
        """GET /api/overview — cluster overview, message rates."""
        async with self._client() as client:
            resp = await client.get("/api/overview")
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Queue operations
    # ------------------------------------------------------------------
    async def list_queues(self) -> list[dict[str, Any]]:
        """GET /api/queues/%2F — all queues with message counts, rates, consumers."""
        async with self._client() as client:
            resp = await client.get(f"/api/queues/{_VHOST}")
            resp.raise_for_status()
            return resp.json()

    async def get_queue(self, name: str) -> dict[str, Any]:
        """GET /api/queues/%2F/{name} — single queue detail."""
        async with self._client() as client:
            resp = await client.get(f"/api/queues/{_VHOST}/{name}")
            resp.raise_for_status()
            return resp.json()

    async def purge_queue(self, name: str) -> None:
        """DELETE /api/queues/%2F/{name}/contents — clear pending messages."""
        async with self._client() as client:
            resp = await client.delete(f"/api/queues/{_VHOST}/{name}/contents")
            resp.raise_for_status()

    # ------------------------------------------------------------------
    # Consumers
    # ------------------------------------------------------------------
    async def list_consumers(self) -> list[dict[str, Any]]:
        """GET /api/consumers/%2F — active consumers (= active workers)."""
        async with self._client() as client:
            resp = await client.get(f"/api/consumers/{_VHOST}")
            resp.raise_for_status()
            return resp.json()


# Singleton instance
_instance: RabbitMQManagement | None = None


def get_rabbitmq_management() -> RabbitMQManagement:
    global _instance
    if _instance is None:
        _instance = RabbitMQManagement()
    return _instance

"""
MongoDB Client — sync singleton via pymongo.

Provides a lazily-initialized MongoClient tied to the settings
MONGO_HOST / MONGO_PORT / MONGO_USER / MONGO_PASSWORD / MONGO_DATABASE.
Call get_mongo_db() to get the database instance.

Supports two auth modes:
  - With auth    : mongodb://user:pass@host:port/?authSource=...
  - No auth (LAN): mongodb://host:port/  (when MONGO_USER is empty)

All MongoDB operations are sync (blocking), so they MUST be called
inside asyncio.to_thread() or run_in_executor() to avoid blocking
the event loop:
    docs = await asyncio.to_thread(collection.find(...).to_list)
"""

from __future__ import annotations

import logging
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def _build_uri() -> str:
    """Build MongoDB connection URI from settings."""
    host = settings.MONGO_HOST
    port = settings.MONGO_PORT
    user = settings.MONGO_USER
    password = settings.MONGO_PASSWORD
    auth_source = settings.MONGO_AUTH_SOURCE or "admin"

    if user and password:
        return (
            f"mongodb://{user}:{password}@{host}:{port}/"
            f"?authSource={auth_source}"
        )
    else:
        # No auth — direct LAN connection
        return f"mongodb://{host}:{port}/"


def get_mongo_client() -> MongoClient:
    """Return the singleton MongoClient (lazy init)."""
    global _client
    if _client is None:
        uri = _build_uri()
        logger.info(
            f"[mongo] Connecting to {settings.MONGO_HOST}:{settings.MONGO_PORT} "
            f"(auth={bool(settings.MONGO_USER)}, database={settings.MONGO_DATABASE})"
        )
        _client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            maxPoolSize=50,
            waitQueueTimeoutMS=5000,
            retryWrites=True,
            retryReads=True,
        )
    return _client


def get_mongo_db() -> Database:
    """Return the configured database instance."""
    global _db
    if _db is None:
        client = get_mongo_client()
        _db = client[settings.MONGO_DATABASE]
    return _db


def close_mongo_client() -> None:
    """Close the singleton client (call on app shutdown)."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        logger.info("[mongo] Connection closed")

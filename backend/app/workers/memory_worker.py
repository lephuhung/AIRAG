"""
Memory Worker
=============
Consumes ``hrag.memory`` messages and persists each user turn as a Graphiti
personal-memory episode (LLM fact-extraction + Neo4j write).

Why a dedicated worker instead of an in-request ``asyncio.create_task``:
  * Durable — the message is persisted on the broker, so a fact survives an API
    process restart/crash between the response and the save.
  * Retried — :func:`add_conversation_episode` raises on a transient Graphiti /
    Neo4j failure, which the ``consume()`` loop turns into a delayed retry
    (5s/15s/60s) and finally a dead-letter, instead of a silent warning.
  * Off the hot path — the two LLM calls + graph write no longer compete with
    the chat response for the API event loop.

Idempotency: re-delivering the same turn re-adds the same fact text to the same
user entity. Graphiti resolves it to the existing entity/edge, so a duplicate
delivery is effectively a no-op rather than a corruption — no extra flag needed.
"""

from __future__ import annotations

import logging
import uuid

from app.queue.messages import MemorySaveMessage
from app.services.graphiti_client import add_conversation_episode

logger = logging.getLogger(__name__)


async def handle_memory(payload: dict) -> None:
    """Process one ``MemorySaveMessage``.

    Raising propagates to the ``consume()`` loop, which schedules a durable
    retry. ``add_conversation_episode`` only raises on a real write failure;
    a turn with no extractable personal fact returns normally (acked, no retry).
    """
    msg = MemorySaveMessage(**payload)

    if not str(msg.user_id).strip() or not msg.user_message.strip():
        logger.debug("[memory_worker] empty user_id/message — skipping")
        return

    user_id = msg.user_id if isinstance(msg.user_id, uuid.UUID) else uuid.UUID(str(msg.user_id))

    await add_conversation_episode(
        user_id=user_id,
        user_message=msg.user_message,
        assistant_message=msg.assistant_message,
        session_id=msg.session_id,
    )

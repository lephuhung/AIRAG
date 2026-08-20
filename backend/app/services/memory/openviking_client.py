"""
OpenViking Memory Client
========================

Agent long-term memory backed by the OpenViking context database
(https://github.com/volcengine/OpenViking — an open-source context database
for AI agents that stores memories as a tiered virtual filesystem under the
``viking://`` protocol).

Mirrors the Graphiti client interface (``graphiti_client.py``) so the LangGraph
supervisor / memory worker can switch backends via ``NEXUSRAG_MEMORY_BACKEND``
without touching the graph logic:

    await initialize_openviking()                        # call once at app startup
    context = await search_user_memory(uid, query)       # → formatted string for system prompt
    await add_conversation_episode(uid, user_msg, ai_msg, session_id)  # background task
    await save_user_fact(uid, fact)                      # explicit "ghi nhớ" tool

Architecture
------------
- Storage : the OpenViking server (docker image ``ghcr.io/volcengine/openviking``,
            port 1933) — a separate service added to docker-compose.services.yml.
- Recall  : semantic ``search`` over ``viking://user/nexusrag_{user_id}/memories``
            (the per-AIRAG-user memory namespace; ``nexusrag_`` prefix avoids
            collisions with other OpenViking applications sharing the server).
- Capture : every conversation turn is appended to an OpenViking session
            (``nexusrag_{session_id}``) and committed; the server's *async*
            background phase runs LLM-driven summary + memory extraction into the
            user's memories namespace (the same LLM AIRAG already uses for
            Graphiti: ``MEMORY_AGENT_BASE_URL`` / ``GRAPHITI_LLM_*``).
- Identity : no separate LLM identity classifier — OpenViking's ``search`` runs
            intent analysis natively; as a fallback, self-referential queries
            list + read the L0/L1 layers of the user's memory directory directly.

The client runs against the OpenViking server in **trusted auth mode** (see
``server.auth_mode: "trusted"`` in the compose ov.conf): the backend is the
trusted server-side party, asserting each AIRAG user via ``X-OpenViking-Account``
+ ``X-OpenViking-User`` headers, authenticated with the root API key. One
per-user client is cached per AIRAG user (same trust model as Graphiti's
``group_id = nexusrag_user_{user_id}``).
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# Memory formatting budget — max characters injected into the system prompt.
# Slightly larger than Graphiti's budget because OpenViking returns structured
# abstracts/overviews that compress well.
_MEMORY_CONTEXT_MAX_CHARS = 1200

# Identity-query patterns (mirror graphiti_client._IDENTITY_PATTERNS): when no
# semantic hit comes back, these trigger a direct read of the whole memory
# directory instead of trusting vector recall for pronoun-heavy short queries.
_IDENTITY_PATTERNS = re.compile(
    r"(?i)"
    r"(tên\s*(của\s*)?tôi|tôi\s*tên|tên\s*tôi|my\s*name)"
    r"|(tôi\s*(là|l[\xE0a]\s*ai|sinh|tuổi)|tôi\s*\w{1,4}\s*ai)"
    r"|(tôi\s*(đang\s+|đã\s+|sẽ\s+|vừa\s+|mới\s+)?(công\s*tác|làm\s*việc|đang\s*ở|sống|học|dùng|sử\s*dụng|sư\s*dụng|xài))"
    r"|(thông\s*tin.*tôi|tôi.*thông\s*tin)"
    r"|(who\s*am\s*i|what.*my\s*(name|job|role|age))"
    r"|(nhớ.*tôi|biết.*tôi|tôi.*là\s*ai)"
)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

# Per-user OpenViking clients (trusted mode: each client asserts ONE OpenViking
# user via X-OpenViking-Account / X-OpenViking-User, backed by the root API key).
_openviking_clients: dict[str, Any] = {}
_openviking_root_client: Any | None = None


def is_enabled() -> bool:
    """True when OPENVIKING_URL is set — OpenViking is an opt-in backend."""
    return bool((settings.OPENVIKING_URL or "").strip())


# ---------------------------------------------------------------------------
# Namespace mapping — AIRAG users → OpenViking users
# ---------------------------------------------------------------------------


def _ov_user_id(user_id: uuid.UUID | int | str) -> str:
    """Map an AIRAG user id onto the OpenViking user namespace.

    AIRAG user ids are ints (legacy) or UUIDs; OpenViking user ids are opaque
    strings. We normalize to the UUID form and prefix with ``nexusrag_`` so
    multiple AIRAG stacks can share one OpenViking server without colliding.
    """
    if isinstance(user_id, uuid.UUID):
        return f"nexusrag_{user_id}"
    if isinstance(user_id, int):
        return f"nexusrag_{uuid.UUID(int=user_id)}"
    # str forms may already be the full UUID string
    try:
        return f"nexusrag_{uuid.UUID(str(user_id))}"
    except (ValueError, AttributeError):
        return f"nexusrag_{user_id}"


def _ov_account() -> str:
    """The shared OpenViking account all AIRAG users live under (trusted mode)."""
    return (settings.OPENVIKING_ACCOUNT or "nexusrag").strip() or "nexusrag"


def _memory_target_uri(user_id: uuid.UUID | int | str) -> str:
    """The per-user memories namespace searched on recall, scoped to the asserted user."""
    return f"viking://user/{_ov_user_id(user_id)}/memories"


def _session_id(session_id: str | None) -> str | None:
    """Map an AIRAG chat session id onto the OpenViking session namespace."""
    if not session_id:
        return None
    try:
        return f"nexusrag_{uuid.UUID(str(session_id))}"
    except (ValueError, AttributeError):
        return f"nexusrag_{session_id}"


# ---------------------------------------------------------------------------
# Client factories (trusted mode)
# ---------------------------------------------------------------------------


def get_openviking_client(user_id: uuid.UUID | int | str | None = None) -> Any:
    """Lazily create an OpenViking AsyncHTTPClient.

    - With a user_id: a client that asserts that user (trusted mode headers
      X-OpenViking-Account / X-OpenViking-User + root API key) — used for all
      per-user data ops (recall, session write, save fact). Cached per user.
    - Without a user_id: a bare root client (health checks only).
    """
    global _openviking_root_client
    from openviking_sdk import AsyncHTTPClient

    if user_id is None:
        if _openviking_root_client is None:
            _openviking_root_client = AsyncHTTPClient(
                url=settings.OPENVIKING_URL,
                api_key=settings.OPENVIKING_API_KEY or None,
                timeout=settings.OPENVIKING_TIMEOUT,
            )
        return _openviking_root_client

    ov_user = _ov_user_id(user_id)
    client = _openviking_clients.get(ov_user)
    if client is None:
        client = AsyncHTTPClient(
            url=settings.OPENVIKING_URL,
            api_key=settings.OPENVIKING_API_KEY or None,
            account=_ov_account(),
            user_id=ov_user,
            timeout=settings.OPENVIKING_TIMEOUT,
        )
        _openviking_clients[ov_user] = client
    return client


async def _ensure_initialized(client: Any) -> None:
    """The SDK raises unless initialize() has built its http client; re-run it
    lazily in case startup init was skipped/failed."""
    if getattr(client, "_http", None) is None:
        await client.initialize()


async def initialize_openviking() -> None:
    """Health-check the server once at app startup. Non-fatal."""
    if not is_enabled():
        logger.info("[openviking] disabled (OPENVIKING_URL empty) — skipping init")
        return
    try:
        client = get_openviking_client()
        await _ensure_initialized(client)
        ok = await client.health()
        if ok:
            logger.info("[openviking] server healthy at %s", settings.OPENVIKING_URL)
        else:
            logger.warning(
                "[openviking] server reachable but /health != ok at %s",
                settings.OPENVIKING_URL,
            )
    except Exception as exc:
        logger.warning(f"[openviking] init failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_memory_context(
    matches: list[dict],
    query: str = "",
    budget: int = _MEMORY_CONTEXT_MAX_CHARS,
) -> str:
    """Flatten OpenViking search MatchedContexts into a system-prompt string.

    Each matched node carries an ``abstract`` (L0, ~100 tokens) and optionally an
    ``overview`` (L1, ~2k tokens). We prefer the more useful overview when present,
    fall back to the abstract, and cap the whole block at ``budget`` chars —
    mirroring graphiti_client._format_memory_context semantics.
    """
    if not matches:
        return ""

    lines: list[str] = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        uri = m.get("uri") or ""
        score = m.get("score") or 0.0
        text = (m.get("overview") or m.get("abstract") or "").strip()
        if not text:
            continue
        line = text if len(text) <= 400 else text[:400] + "…"
        if uri:
            line = f"{line} ({uri})"
        lines.append(f"- {line}")

    if not lines:
        return ""

    joined = "Thông tin cá nhân đã biết về người dùng (OpenViking):\n" + "\n".join(lines)
    if len(joined) > budget:
        # Keep the header + as many full-ish lines as fit.
        cut = joined[:budget]
        cut = cut.rsplit("\n", 1)[0]
        joined = cut + "\n…"
    return joined


# ---------------------------------------------------------------------------
# Identity fallback — list + read the whole memory directory
# ---------------------------------------------------------------------------


async def _identity_recall(user_id: uuid.UUID | int | str) -> str:
    """Read every memory file's L0/L1 layer under the user's memories namespace."""
    client = get_openviking_client(user_id)
    target = _memory_target_uri(user_id)
    await _ensure_initialized(client)
    try:
        entries = await client.ls(target, recursive=True, output="original", node_limit=256)
    except Exception as exc:
        logger.warning(f"[openviking] identity fallback ls failed: {exc}")
        return ""

    matches: list[dict] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        uri = e.get("uri") or e.get("path") or ""
        # Skip directories and hidden metadata handled by read() itself.
        if uri.endswith("/") or e.get("type") in ("dir", "directory"):
            continue
        try:
            text = await client.read(uri, limit=2000)
        except Exception:
            continue
        if text and text.strip():
            matches.append({"uri": uri, "abstract": text.strip()})

    return _format_memory_context(matches)


# ---------------------------------------------------------------------------
# Public API — same signatures as graphiti_client
# ---------------------------------------------------------------------------


async def search_user_memory(
    user_id: uuid.UUID | int | str,
    query: str,
    top_k: int = 5,
) -> str:
    """Search the user's OpenViking memories namespace for relevant context.

    Strategy — 2 layers (mirrors graphiti_client.search_user_memory):
      Layer 1 — semantic search over ``viking://user/nexusrag_{uid}/memories``.
                OpenViking's ``search`` performs its own intent analysis +
                L0→L2 hierarchical retrieval, so it handles pronoun-heavy
                "tôi…" queries better than raw vector similarity.
      Layer 2 — identity fallback: for self-referential queries with no
                semantic hit, list + read the memory directory directly.

    Returns a formatted string for the system prompt, or "" if empty.
    """
    if not query.strip():
        return ""
    if not is_enabled():
        return ""

    logger.info(f"[openviking] search user_id={user_id}, query={query[:80]!r}")

    client = get_openviking_client(user_id)
    await _ensure_initialized(client)
    try:
        result = await client.search(
            query=query,
            target_uri=_memory_target_uri(user_id),
            limit=max(1, top_k),
            score_threshold=getattr(settings, "OPENVIKING_SCORE_THRESHOLD", None),
        )
    except Exception as exc:
        logger.warning(f"[openviking] search failed for user {user_id}: {exc}")
        result = {}

    if not isinstance(result, dict):
        result = {}

    matches = list(result.get("memories") or []) + list(result.get("resources") or [])
    context = _format_memory_context(matches)

    if context:
        logger.info(f"[openviking] injected {len(context)} chars for user {user_id}")
        return context

    # Layer 2 — identity fallback
    if _IDENTITY_PATTERNS.search(query):
        logger.info(f"[openviking] identity query — reading full memory dir for {user_id}")
        context = await _identity_recall(user_id)
        if context:
            return context

    logger.info(f"[openviking] no relevant memory for user {user_id}")
    return ""


# ---------------------------------------------------------------------------
# Episode / fact saving
# ---------------------------------------------------------------------------


async def add_conversation_episode(
    user_id: uuid.UUID | int | str,
    user_message: str,
    assistant_message: str,
    session_id: str | None = None,
) -> None:
    """Append the turn to the user's OpenViking session and commit.

    commit() archives the conversation and launches the server-side *async*
    pipeline: LLM summary + long-term memory extraction into the user's
    ``memories/`` namespace (preferences, events, experiences…). Unlike Graphiti
    (user-turn only), both sides of the dialogue are stored — the OpenViking
    extraction pipeline is designed for full conversation sessions and dedups
    candidate memories against existing ones (LLM decisions: skip/create/merge).
    """
    if not is_enabled():
        return
    if not user_message.strip():
        return

    ov_session = _session_id(session_id) or f"turn_{uuid.uuid4()}"
    client = get_openviking_client(user_id)
    await _ensure_initialized(client)

    try:
        await client.create_session(ov_session)
    except Exception as exc:
        # create_session is idempotent per the server, but treat failure as
        # fatal for this write (let the worker retry/DLQ) rather than
        # half-writing a session.
        logger.warning(f"[openviking] create_session failed: {exc}")
        raise

    try:
        await client.add_message(ov_session, role="user", content=user_message)
        if assistant_message and assistant_message.strip():
            await client.add_message(ov_session, role="assistant", content=assistant_message)
        await client.commit_session(ov_session)
        logger.info(
            f"[openviking] session {ov_session} committed for user {user_id} "
            f"({len(user_message)} user chars)"
        )
    except Exception as exc:
        logger.warning(f"[openviking] session write failed for user {user_id}: {exc}")
        raise


async def save_user_fact(user_id: uuid.UUID | int | str, fact: str) -> None:
    """Write one explicit fact into the user's memories namespace.

    Used by the agent's "ghi nhớ" tool: the fact lands in
    ``viking://user/nexusrag_{uid}/memories/agent_notes/`` so it is immediately
    recallable (vectorized on write) without waiting for a session commit.
    """
    if not is_enabled():
        return
    fact = (fact or "").strip()
    if not fact:
        return

    client = get_openviking_client(user_id)
    await _ensure_initialized(client)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    uri = f"viking://user/{_ov_user_id(user_id)}/memories/agent_notes/note_{stamp}.md"
    try:
        # mode="create": writes a brand-new file (auto-creates the parent dir),
        # only for text-writable extensions — perfect for timestamped note files.
        await client.write(uri, content=fact, mode="create", wait=True)
        logger.info(f"[openviking] saved user fact -> {uri}")
    except Exception as exc:
        logger.warning(f"[openviking] save_user_fact failed for user {user_id}: {exc}")
        raise


def save_user_fact_background(user_id: uuid.UUID | int | str, fact: str) -> None:
    """Non-blocking variant of save_user_fact for the tool path (mirrors
    graphiti_client.save_user_fact_background)."""
    if not is_enabled():
        return
    fact = (fact or "").strip()
    if not fact or not user_id:
        return

    async def _runner() -> None:
        try:
            await save_user_fact(user_id, fact)
        except Exception as exc:
            logger.warning(f"[openviking] background save_user_fact crashed: {exc}")

    try:
        asyncio.create_task(_runner())
    except RuntimeError:
        logger.warning("[openviking] no running loop for background save; skipped")
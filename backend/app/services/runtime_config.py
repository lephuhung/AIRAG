"""
RuntimeConfigService — DB-backed overrides for LLM role configuration.

Design (docs/plan-llm-runtime-config.md §5.1):
  * `.env` = default, `system_settings` table = override. No override row →
    the exact pre-feature `.env` behaviour.
  * SYNC snapshot cache (`get_effective_sync`) so the LLM factories keep their
    synchronous signatures — most callers are sync / run in threadpools.
  * The snapshot is refreshed by ASYNC entry points only: app startup,
    request middleware, `ensure_fresh_config()` in workers, and right after
    admin writes (set_override/clear_override).
  * Every DB access is fail-safe: a missing table (migration not yet run) or a
    decrypt failure falls back to `.env` defaults — the system never breaks
    because of this feature.

Value shape stored per role key ("llm.<role>"):
    {"provider": "openai_compatible", "base_url": "...", "model": "...",
     "api_key_enc": "<fernet>", "extra": {"is_vllm": true, ...}}
The reserved row `_config_version` stores a plain integer as TEXT and is the
only thing polled per message/request.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field

from app.core.config import settings

logger = logging.getLogger(__name__)

# All configurable roles (V2). "vision" inherits "main" unless it has its own
# DB assignment. stt/tts are audio service roles; embedding/rerank apply on
# RESTART only (GPU models preloaded at startup — see plan §12.4).
ROLES = [
    "main", "vision", "thinking", "memory_agent", "kg_extract", "graphiti",
    "stt", "tts", "embedding", "rerank",
]

_VERSION_KEY = "_config_version"
# V2 two-level architecture (plan §12.2):
#   "llm_conn.<conn_id>" → {name, provider, base_url, api_key_enc, extra}
#   "llm_role.<role>"    → {conn_id: "<id>" | "@env", model}
_ROLE_KEY_PREFIX = "llm_role."
_CONN_KEY_PREFIX = "llm_conn."
_ENV_CONN_ID = "@env"
# Domain-separation string for key derivation (must never change — rotating it
# invalidates every stored API key).
_DERIVE_INFO = "airag-system-settings-v1"


@dataclass
class EffectiveLLMConfig:
    """Fully-resolved configuration for one LLM role."""

    provider: str                      # gemini | ollama | openai_compatible
    base_url: str = ""
    model: str = ""
    api_key: str = ""                  # decrypted plaintext (in-memory only)
    extra: dict = field(default_factory=dict)   # is_vllm, max_concurrency, ...
    # "env" → resolved from Settings; "db" → system_settings override
    source: str = "env"


# ---------------------------------------------------------------------------
# Fernet encryption helpers
# ---------------------------------------------------------------------------

def _fernet():
    """Fernet instance derived from SETTING_ENCRYPTION_KEY (fallback: JWT secret).

    Key derivation is deterministic so all processes/workers derive the same
    key without sharing anything beyond the env var.
    """
    from cryptography.fernet import Fernet

    secret = (settings.SETTING_ENCRYPTION_KEY or "").strip() or settings.JWT_SECRET_KEY
    digest = hashlib.sha256(f"{secret}|{_DERIVE_INFO}".encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def _encrypt(plaintext: str) -> str:
    token = _fernet().encrypt(plaintext.encode()).decode()
    return token


def _decrypt(ciphertext: str) -> str | None:
    """Decrypt an API-key token. None on any failure (fail-open to .env)."""
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except Exception as exc:  # InvalidToken, missing key rotation, malformed…
        logger.warning(
            f"[runtime_config] API-key decrypt failed ({exc}) — "
            f"falling back to .env defaults for this entry"
        )
        return None


# ---------------------------------------------------------------------------
# .env defaults (must mirror the pre-feature behaviour exactly)
# ---------------------------------------------------------------------------

def _build_from_settings(role: str) -> EffectiveLLMConfig:
    """Resolve one role purely from pydantic Settings (= legacy behaviour)."""
    p = settings.LLM_PROVIDER.lower()

    if role in ("main", "vision"):
        if p == "gemini":
            return EffectiveLLMConfig(
                provider="gemini",
                api_key=settings.GOOGLE_AI_API_KEY,
                model=settings.LLM_MODEL_FAST,
                extra={"thinking_level": settings.LLM_THINKING_LEVEL},
            )
        if p == "ollama":
            return EffectiveLLMConfig(
                provider="ollama",
                base_url=settings.OLLAMA_HOST,
                model=settings.OLLAMA_MODEL,
                extra={"enable_thinking": settings.OLLAMA_ENABLE_THINKING},
            )
        return EffectiveLLMConfig(
            provider="openai_compatible",
            base_url=settings.OPENAI_COMPATIBLE_BASE_URL,
            model=settings.OPENAI_COMPATIBLE_MODEL,
            api_key=settings.OPENAI_COMPATIBLE_API_KEY,
        )

    if role == "thinking":
        # Mirrors _resolve_thinking_endpoint() in app/services/llm/__init__.py:
        # NEXUSRAG_LG_THINKING_* → ANTHROPIC_* aliases → built-in defaults.
        base_url = (
            (settings.NEXUSRAG_LG_THINKING_BASE_URL or "").strip()
            or (settings.ANTHROPIC_BASE_URL or "").strip()
            or "http://host.docker.internal:20128/v1"
        )
        model = (
            (settings.NEXUSRAG_LG_THINKING_MODEL or "").strip()
            or (settings.ANTHROPIC_DEFAULT_FABLE_MODEL or "").strip()
            or "nexusrag-thinking"
        )
        api_key = (
            (settings.NEXUSRAG_LG_THINKING_API_KEY or "").strip()
            or (settings.ANTHROPIC_AUTH_TOKEN or "").strip()
            or "sk-nexusrag"
        )
        return EffectiveLLMConfig(
            provider="openai_compatible", base_url=base_url, model=model, api_key=api_key
        )

    if role == "memory_agent":
        # NOTE: MEMORY_AGENT_LOCAL=true stays a factory-level legacy branch
        # (LocalVLLMProvider reads Settings directly) — it never becomes a
        # remote effective config.
        return EffectiveLLMConfig(
            provider="openai_compatible",
            base_url=settings.MEMORY_AGENT_BASE_URL,
            model=settings.MEMORY_AGENT_MODEL,
            api_key=settings.MEMORY_AGENT_API_KEY,
        )

    if role == "kg_extract":
        kg_provider = settings.LEGAL_KG_LLM_PROVIDER.lower()
        api_key = (
            settings.GOOGLE_AI_API_KEY if kg_provider == "gemini" else settings.LEGAL_KG_LLM_API_KEY
        )
        return EffectiveLLMConfig(
            provider=kg_provider,
            base_url=settings.LEGAL_KG_LLM_BASE_URL,
            model=settings.LEGAL_KG_LLM_MODEL,
            api_key=api_key,
        )

    if role == "graphiti":
        return EffectiveLLMConfig(
            provider="openai_compatible",
            base_url=settings.GRAPHITI_LLM_BASE_URL,
            model=settings.GRAPHITI_LLM_MODEL,
            api_key=settings.GRAPHITI_LLM_API_KEY,
        )

    if role == "stt":
        # faster_whisper (local GPU/CPU) stays .env-driven; the openai path
        # resolves to a network config so a WebUI connection can override it
        # with the same EffectiveLLMConfig shape.
        if settings.STT_PROVIDER.lower() in ("openai", "openai_compatible"):
            return EffectiveLLMConfig(
                provider="openai_compatible",
                base_url=settings.STT_OPENAI_BASE_URL,
                model=settings.STT_OPENAI_MODEL,
                api_key=settings.STT_OPENAI_API_KEY,
            )
        return EffectiveLLMConfig(
            provider="faster_whisper",
            model=settings.STT_FW_MODEL,
            extra={
                "device": settings.STT_FW_DEVICE,
                "compute_type": settings.STT_FW_COMPUTE_TYPE,
                "download_root": settings.STT_FW_MODEL_DIR,
                "language": settings.STT_LANGUAGE,
            },
        )

    if role == "tts":
        # Only the OmniVoice (OpenAI-compatible audio-speech) engine exists
        # today; represented as a network endpoint so connections can override
        # base_url/model/api_key without a new provider class.
        return EffectiveLLMConfig(
            provider="openai_compatible",
            base_url=settings.TTS_OMNIVOICE_BASE_URL,
            model=settings.TTS_OMNIVOICE_MODEL,
            api_key=settings.TTS_OMNIVOICE_API_KEY,
            extra={
                "voice": settings.TTS_DEFAULT_VOICE,
                "speed": settings.TTS_DEFAULT_SPEED,
            },
        )

    if role == "embedding":
        # RESTART-ONLY semantics: consumed once at process start by
        # services/models/loader.py BEFORE preload_models(). Never hot-swapped.
        return EffectiveLLMConfig(provider="local", model=settings.HRAG_EMBEDDING_MODEL)

    if role == "rerank":
        # RESTART-ONLY semantics (same as embedding).
        return EffectiveLLMConfig(provider="local", model=settings.HRAG_RERANKER_MODEL)

    raise ValueError(f"Unknown LLM role: {role!r}. Supported: {ROLES}")


# ---------------------------------------------------------------------------
# Snapshot cache (sync-readable)
# ---------------------------------------------------------------------------

_snapshot: dict[str, EffectiveLLMConfig] = {}
_snapshot_version: int = -1


def get_effective_sync(role: str) -> EffectiveLLMConfig:
    """SYNC — read the in-process snapshot. Never touches the DB.

    Falls back to `.env` defaults when the snapshot hasn't been refreshed yet
    (e.g. before startup lifespan runs) or the role has no effective override.
    """
    cfg = _snapshot.get(role)
    if cfg is not None:
        return cfg
    return _build_from_settings(role)


def snapshot_version() -> int:
    """SYNC — version the current snapshot was built from."""
    return _snapshot_version


# ---------------------------------------------------------------------------
# Async DB access (all fail-open when the table is missing)
# ---------------------------------------------------------------------------

async def get_version() -> int:
    """Current `_config_version` from the DB (0 when absent/unreadable)."""
    try:
        from sqlalchemy import select

        from app.core.database import async_session_maker
        from app.models.system_setting import SystemSetting

        async with async_session_maker() as db:
            val = await db.scalar(
                select(SystemSetting.value_enc).where(SystemSetting.key == _VERSION_KEY)
            )
            return int(val) if val not in (None, "") else 0
    except Exception as exc:
        logger.debug(f"[runtime_config] version read failed (table missing?): {exc}")
        return 0


_last_refresh_check: float = 0.0
_THROTTLE_SECONDS = 1.0


def _throttle_ok() -> bool:
    """Cheap monotonic-clock gate so per-request checks run at most ~1/s."""
    global _last_refresh_check
    import time as _time

    now = _time.monotonic()
    if now - _last_refresh_check < _THROTTLE_SECONDS:
        return False
    _last_refresh_check = now
    return True


async def maybe_refresh(throttle_seconds: float | None = None) -> bool:
    """Throttled refresh for request hot paths (backend API middleware).

    At most one DB version check per ``throttle_seconds`` (default 1s) per
    process; everything else is a no-op clock read. Fail-open: any error is
    swallowed — the last-known-good snapshot stays in place.
    """
    global _THROTTLE_SECONDS
    if throttle_seconds is not None:
        _THROTTLE_SECONDS = throttle_seconds
    try:
        if not _throttle_ok():
            return False
        return await refresh_snapshot()
    except Exception as exc:
        logger.warning(f"[runtime_config] maybe_refresh failed (ignored): {exc}")
        return False


async def refresh_snapshot() -> bool:
    """Reload the snapshot if the DB version changed. True when reloaded.

    Call sites: startup lifespan, request middleware (throttled), worker
    `ensure_fresh_config()`, immediately after set_override/clear_override.
    """
    global _snapshot_version, _snapshot
    v = await get_version()
    if v == _snapshot_version and _snapshot:
        return False
    new_snap: dict[str, EffectiveLLMConfig] = {}
    for role in ROLES:
        try:
            new_snap[role] = await _load_effective(role)
        except Exception as exc:
            logger.warning(f"[runtime_config] loading {role} failed: {exc}")
            new_snap[role] = _build_from_settings(role)
    _snapshot = new_snap
    _snapshot_version = v
    logger.info(f"[runtime_config] snapshot refreshed → version={v}")
    return True


async def _fetch_row(db, key: str):
    from sqlalchemy import select

    from app.models.system_setting import SystemSetting

    return await db.scalar(select(SystemSetting).where(SystemSetting.key == key))


async def _load_effective(role: str) -> EffectiveLLMConfig:
    """Effective config for one role (V2 two-level resolution).

    Role assignment {conn_id, model} → resolve the referenced connection for
    provider/base_url/api_key. ``conn_id == "@env"`` (or missing/corrupt row,
    or dangling conn_id) → legacy `.env` behaviour.
    """
    try:
        from app.core.database import async_session_maker

        async with async_session_maker() as db:
            row = await _fetch_row(db, f"{_ROLE_KEY_PREFIX}{role}")
    except Exception as exc:
        logger.debug(f"[runtime_config] override read failed for {role}: {exc}")
        row = None

    if row is None:
        return _build_from_settings(role)

    try:
        doc = json.loads(row.value_enc)
    except (TypeError, ValueError) as exc:
        logger.warning(f"[runtime_config] corrupt JSON for {role}: {exc} — using .env")
        return _build_from_settings(role)

    conn_id = str(doc.get("conn_id") or "")
    model = str(doc.get("model") or "")
    if not conn_id or conn_id == _ENV_CONN_ID:
        cfg = _build_from_settings(role)
        # Allow overriding JUST the model on top of the .env connection
        # (provider/base_url/api_key stay from .env). Marked source="db" so
        # the WebUI shows the override badge + reset button.
        if model and model != cfg.model:
            cfg.model = model
            cfg.source = "db"
        return cfg

    try:
        from app.core.database import async_session_maker as _mk

        async with _mk() as db:
            conn_row = await _fetch_row(db, f"{_CONN_KEY_PREFIX}{conn_id}")
    except Exception as exc:
        logger.debug(f"[runtime_config] connection read failed for {conn_id}: {exc}")
        conn_row = None

    if conn_row is None:
        logger.warning(
            f"[runtime_config] role '{role}' references missing connection "
            f"{conn_id!r} — falling back to .env defaults"
        )
        return _build_from_settings(role)

    try:
        conn = json.loads(conn_row.value_enc)
    except (TypeError, ValueError) as exc:
        logger.warning(f"[runtime_config] corrupt JSON for connection {conn_id}: {exc}")
        return _build_from_settings(role)

    api_key = ""
    enc = conn.get("api_key_enc")
    if enc:
        api_key = _decrypt(enc) or ""

    extra = dict(conn.get("extra") or {})
    # Role-level extras (if any) override connection-level ones.
    role_extra = doc.get("extra")
    if isinstance(role_extra, dict):
        extra.update(role_extra)

    return EffectiveLLMConfig(
        provider=str(conn.get("provider") or ""),
        base_url=str(conn.get("base_url") or ""),
        # Role assignment picks the model; empty falls back to the connection's
        # stored default ("extra.model"), then blank.
        model=model or str((conn.get("extra") or {}).get("model") or ""),
        api_key=api_key,
        extra=extra,
        source="db",
    )


async def list_overrides() -> dict[str, dict]:
    """Assignment metadata per role (admin GET): conn_id, model, who/when."""
    out: dict[str, dict] = {}
    try:
        from sqlalchemy import select

        from app.core.database import async_session_maker
        from app.models.system_setting import SystemSetting

        async with async_session_maker() as db:
            rows = (
                await db.execute(
                    select(SystemSetting).where(
                        SystemSetting.key.like(f"{_ROLE_KEY_PREFIX}%")
                    )
                )
            ).scalars().all()
    except Exception as exc:
        logger.debug(f"[runtime_config] list_overrides failed: {exc}")
        return out

    for row in rows:
        role = row.key.removeprefix(_ROLE_KEY_PREFIX)
        if role not in ROLES:
            continue
        try:
            doc = json.loads(row.value_enc)
        except (TypeError, ValueError):
            continue
        out[role] = {
            "override": True,
            "conn_id": str(doc.get("conn_id") or _ENV_CONN_ID),
            "model": str(doc.get("model") or ""),
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "updated_by": str(row.updated_by) if row.updated_by else None,
        }
    return out


async def set_override(role: str, payload: dict, actor_id=None) -> None:
    """Persist a V2 role assignment ``{conn_id, model}`` + bump `_config_version`
    in one transaction, then refresh the local snapshot.

    ``conn_id`` must be an existing connection id or "@env". The caller (API
    layer) validates that; this function re-checks defensively.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown LLM role: {role!r}. Supported: {ROLES}")

    conn_id = str(payload.get("conn_id") or _ENV_CONN_ID).strip()
    model = str(payload.get("model") or "").strip()
    if conn_id != _ENV_CONN_ID and conn_id not in await list_connections():
        raise ValueError(f"Unknown connection {conn_id!r} — create it first")

    doc: dict = {"conn_id": conn_id, "model": model}
    extra = payload.get("extra")
    if isinstance(extra, dict) and extra:
        doc["extra"] = extra

    from sqlalchemy import cast, Integer, String, update

    from app.core.database import async_session_maker
    from app.models.system_setting import SystemSetting

    async with async_session_maker() as db:
        # merge = INSERT or UPDATE (upsert) — role may already have an assignment
        await db.merge(
            SystemSetting(
                key=f"{_ROLE_KEY_PREFIX}{role}",
                value_enc=json.dumps(doc),
                updated_by=actor_id,
            )
        )
        await db.execute(
            update(SystemSetting)
            .where(SystemSetting.key == _VERSION_KEY)
            .values(value_enc=cast(cast(SystemSetting.value_enc, Integer) + 1, String))
        )
        await db.commit()

    await refresh_snapshot()
    logger.info(f"[runtime_config] assignment saved for role={role!r} (version bumped)")


async def clear_override(role: str, actor_id=None) -> None:
    """Remove a role override → back to `.env` defaults. Bumps the version."""
    if role not in ROLES:
        raise ValueError(f"Unknown LLM role: {role!r}. Supported: {ROLES}")

    from sqlalchemy import cast, delete, Integer, String, update

    from app.core.database import async_session_maker
    from app.models.system_setting import SystemSetting

    async with async_session_maker() as db:
        await db.execute(delete(SystemSetting).where(SystemSetting.key == f"{_ROLE_KEY_PREFIX}{role}"))
        await db.execute(
            update(SystemSetting)
            .where(SystemSetting.key == _VERSION_KEY)
            .values(value_enc=cast(cast(SystemSetting.value_enc, Integer) + 1, String))
        )
        await db.commit()

    await refresh_snapshot()
    logger.info(f"[runtime_config] override cleared for role={role!r} (version bumped)")


# ---------------------------------------------------------------------------
# V2 — Model Connections (endpoints declared once, referenced by many roles)
# ---------------------------------------------------------------------------

def _validate_conn_id(conn_id: str) -> str:
    """Connection ids are URL-path safe slugs: [a-zA-Z0-9_-]{1,64}."""
    import re

    cid = (conn_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid):
        raise ValueError(
            "conn_id must be 1-64 chars of [a-zA-Z0-9_-] "
            f"(got {conn_id!r})"
        )
    return cid


async def list_connections() -> dict[str, dict]:
    """All stored connections: conn_id → metadata (API key masked, never plaintext)."""
    out: dict[str, dict] = {}
    try:
        from sqlalchemy import select

        from app.core.database import async_session_maker
        from app.models.system_setting import SystemSetting

        async with async_session_maker() as db:
            rows = (
                await db.execute(
                    select(SystemSetting).where(
                        SystemSetting.key.like(f"{_CONN_KEY_PREFIX}%")
                    )
                )
            ).scalars().all()
    except Exception as exc:
        logger.debug(f"[runtime_config] list_connections failed: {exc}")
        return out

    for row in rows:
        cid = row.key.removeprefix(_CONN_KEY_PREFIX)
        try:
            doc = json.loads(row.value_enc)
        except (TypeError, ValueError) as exc:
            logger.warning(f"[runtime_config] corrupt JSON for connection {cid}: {exc}")
            continue
        out[cid] = {
            "name": str(doc.get("name") or cid),
            "provider": str(doc.get("provider") or ""),
            "base_url": str(doc.get("base_url") or ""),
            "has_api_key": bool(doc.get("api_key_enc")),
            "masked_api_key": "",  # filled by caller if it decrypts; never here
            "extra": dict(doc.get("extra") or {}),
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
    return out


async def get_connection(conn_id: str) -> dict | None:
    """One connection with DECRYPTED api_key — internal use only, never serialize."""
    try:
        from app.core.database import async_session_maker

        async with async_session_maker() as db:
            row = await _fetch_row(db, f"{_CONN_KEY_PREFIX}{_validate_conn_id(conn_id)}")
    except ValueError:
        raise
    except Exception as exc:
        logger.debug(f"[runtime_config] get_connection failed for {conn_id}: {exc}")
        return None
    if row is None:
        return None
    doc = json.loads(row.value_enc)
    api_key = ""
    enc = doc.get("api_key_enc")
    if enc:
        api_key = _decrypt(enc) or ""
    return {
        "conn_id": _validate_conn_id(conn_id),
        "name": str(doc.get("name") or conn_id),
        "provider": str(doc.get("provider") or ""),
        "base_url": str(doc.get("base_url") or ""),
        "api_key": api_key,
        "extra": dict(doc.get("extra") or {}),
    }


async def set_connection(conn_id: str, payload: dict, actor_id=None) -> None:
    """Create/update a connection. Blank ``api_key`` preserves the stored key."""
    cid = _validate_conn_id(conn_id)

    provider = str(payload.get("provider") or "").lower().strip()
    if provider not in ("gemini", "ollama", "openai_compatible"):
        raise ValueError(f"Unknown provider {payload.get('provider')!r}")

    doc: dict = {
        "name": str(payload.get("name") or cid).strip(),
        "provider": provider,
        "base_url": str(payload.get("base_url") or "").strip(),
        "extra": payload.get("extra") or {},
    }

    api_key = (payload.get("api_key") or "").strip()
    if not api_key:
        # Preserve the previously stored key on a URL/name-only update.
        existing = await get_connection(cid)
        if existing and existing["api_key"]:
            doc["api_key_enc"] = _encrypt(existing["api_key"])
    else:
        doc["api_key_enc"] = _encrypt(api_key)

    from sqlalchemy import cast, Integer, String, update

    from app.core.database import async_session_maker
    from app.models.system_setting import SystemSetting

    async with async_session_maker() as db:
        await db.merge(
            SystemSetting(
                key=f"{_CONN_KEY_PREFIX}{cid}",
                value_enc=json.dumps(doc),
                updated_by=actor_id,
            )
        )
        await db.execute(
            update(SystemSetting)
            .where(SystemSetting.key == _VERSION_KEY)
            .values(value_enc=cast(cast(SystemSetting.value_enc, Integer) + 1, String))
        )
        await db.commit()

    await refresh_snapshot()
    logger.info(f"[runtime_config] connection saved: {cid!r} (version bumped)")


async def delete_connection(conn_id: str, actor_id=None, force: bool = False) -> list[str]:
    """Delete a connection.

    Returns the list of roles still referencing it WITHOUT deleting when
    ``force=False`` and references exist — the caller decides to answer 409.
    With ``force=True`` (or no references) deletes + bumps version and returns [].
    """
    global _snapshot
    cid = _validate_conn_id(conn_id)

    refs: list[str] = []
    overrides = await list_overrides()
    refs = sorted(r for r, ov in overrides.items() if ov.get("conn_id") == cid)
    if refs and not force:
        return refs

    from sqlalchemy import cast, delete, Integer, String, update

    from app.core.database import async_session_maker
    from app.models.system_setting import SystemSetting

    async with async_session_maker() as db:
        await db.execute(
            delete(SystemSetting).where(SystemSetting.key == f"{_CONN_KEY_PREFIX}{cid}")
        )
        await db.execute(
            update(SystemSetting)
            .where(SystemSetting.key == _VERSION_KEY)
            .values(value_enc=cast(cast(SystemSetting.value_enc, Integer) + 1, String))
        )
        await db.commit()

    await refresh_snapshot()
    logger.info(f"[runtime_config] connection deleted: {cid!r} (version bumped)")
    return []


# ---------------------------------------------------------------------------
# Default connections — seed LLM services that already exist in the Docker
# network so admins see them out-of-the-box on the WebUI (idempotent: a
# connection is only created when its key is absent, never overwriting
# user edits).
# ---------------------------------------------------------------------------

def _default_connections() -> list[dict]:
    """Build the default connection catalogue from current settings."""
    from app.core.config import settings

    defaults = [
        {
            "conn_id": "vllm-main",
            "name": "vLLM Main",
            "provider": "openai_compatible",
            "base_url": settings.OPENAI_COMPATIBLE_BASE_URL,
            "api_key": settings.OPENAI_COMPATIBLE_API_KEY,
            "extra": {"is_vllm": True},
        },
        {
            "conn_id": "vllm-memory",
            "name": "vLLM Memory (qwen-memory)",
            "provider": "openai_compatible",
            "base_url": settings.MEMORY_AGENT_BASE_URL,
            "api_key": settings.MEMORY_AGENT_API_KEY,
            "extra": {"is_vllm": True},
        },
        {
            "conn_id": "thinking-router",
            "name": "Thinking Router (9router)",
            "provider": "openai_compatible",
            "base_url": settings.NEXUSRAG_LG_THINKING_BASE_URL,
            "api_key": settings.NEXUSRAG_LG_THINKING_API_KEY or "sk-nexusrag",
            "extra": {},
        },
        {
            "conn_id": "embed-rerank",
            "name": "Embed-Rerank Service",
            "provider": "openai_compatible",
            "base_url": "http://embed-rerank:8090",
            "api_key": "sk-nexusrag",
            # Local service with a fixed model pair — prefill defaults so the
            # WebUI never asks the admin to type model names.
            "extra": {
                "kind": "embed_rerank",
                "default_models": [
                    settings.HRAG_EMBEDDING_MODEL,
                    settings.HRAG_RERANKER_MODEL,
                ],
            },
        },
        {
            "conn_id": "vllm-ocr",
            "name": "vLLM OCR (Unlimited-OCR)",
            "provider": "openai_compatible",
            "base_url": "http://vllm-ocr:8001/v1",
            "api_key": "sk-nexusrag",
            "extra": {"is_vllm": True},
        },
        {
            "conn_id": "stt-whisper",
            "name": "STT Whisper",
            "provider": "openai_compatible",
            "base_url": "http://stt:8091/v1",
            "api_key": "sk-nexusrag",
            "extra": {"kind": "stt", "default_models": [settings.STT_FW_MODEL]},
        },
        {
            "conn_id": "tts-omnivoice",
            "name": "TTS Omnivoice",
            "provider": "openai_compatible",
            "base_url": settings.TTS_OMNIVOICE_BASE_URL,
            "api_key": settings.TTS_OMNIVOICE_API_KEY or "sk-nexusrag",
            "extra": {"kind": "tts", "default_models": ["omnivoice"]},
        },
    ]
    return [d for d in defaults if d["base_url"].strip()]


async def _host_resolvable(base_url: str) -> bool:
    """True when the base_url's host resolves in this network context."""
    import asyncio
    import socket
    from urllib.parse import urlparse

    try:
        host = urlparse(base_url).hostname or ""
    except ValueError:
        return False
    if not host:
        return False
    # IPs and localhost are trivially fine.
    if host in ("localhost",) or host.replace(".", "").isdigit():
        return True
    try:
        await asyncio.wait_for(
            asyncio.to_thread(socket.getaddrinfo, host, None), timeout=3.0
        )
        return True
    except Exception:
        return False


async def ensure_default_connections() -> list[str]:
    """Seed built-in connections for services present in the Docker network.

    Idempotent: an existing ``llm_conn.<conn_id>`` row (user-created or seeded)
    is never touched. Unreachable hosts are skipped with a log line. Returns
    the conn_ids that were actually created.
    """
    created: list[str] = []
    existing = await list_connections()
    for d in _default_connections():
        cid = d["conn_id"]
        if cid in existing:
            continue
        if not await _host_resolvable(d["base_url"]):
            logger.info(
                f"[runtime_config] default connection '{cid}' skipped — "
                f"host unreachable ({d['base_url']})"
            )
            continue
        try:
            await set_connection(
                cid,
                {
                    "name": d["name"],
                    "provider": d["provider"],
                    "base_url": d["base_url"],
                    "api_key": d["api_key"],
                    "extra": d["extra"],
                },
                actor_id=None,
            )
            created.append(cid)
            logger.info(f"[runtime_config] default connection seeded: '{cid}'")
        except Exception as exc:
            logger.warning(f"[runtime_config] seeding '{cid}' failed: {exc}")
    return created

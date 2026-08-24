"""
Admin LLM runtime-config API — WebUI-configurable LLM roles (no .env restart).

Backed by app/services/runtime_config.py (system_settings table + snapshot
cache). Every endpoint requires superadmin. API keys are never returned in
plaintext — only a masked form.
"""
from __future__ import annotations

import hashlib
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.core.deps import require_superadmin
from app.models.user import User
from app.schemas.llm_config import (
    ConnectionDeleteResponse,
    ConnectionUpsert,
    LLMConfigTestRequest,
    ModelsListRequest,
    RoleAssignment,
)
from app.services import runtime_config
from app.services.audit_service import AuditService

router = APIRouter(prefix="/admin/llm-config", tags=["admin"])

_TEST_TIMEOUT = 15.0  # seconds per probe request

_VALID_PROVIDERS = {"gemini", "ollama", "openai_compatible"}


# ---------------------------------------------------------------------------
# Probing helpers (shared by /test and /models)
# ---------------------------------------------------------------------------

def _mask_key(key: str) -> str:
    """Masked form for UI display: last 4 chars when long enough."""
    if not key:
        return ""
    return f"••••{key[-4:]}" if len(key) > 8 else "••••"


def _auth_headers(api_key: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _list_models_gemini(api_key: str) -> list[str]:
    """Gemini ListModels → bare model names ('models/gemini-2.5-flash' → stripped)."""
    if not api_key:
        raise ValueError("GOOGLE AI API key is required for provider=gemini")
    async with httpx.AsyncClient(timeout=_TEST_TIMEOUT) as client:
        resp = await client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        (m.get("name") or "").removeprefix("models/")
        for m in data.get("models", [])
        if m.get("name")
    ]


async def _list_models_ollama(base_url: str) -> list[str]:
    from app.core.config import settings

    host = (base_url or settings.OLLAMA_HOST).rstrip("/")
    async with httpx.AsyncClient(timeout=_TEST_TIMEOUT) as client:
        resp = await client.get(f"{host}/api/tags")
        resp.raise_for_status()
        data = resp.json()
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def _looks_like_vllm(headers: httpx.Headers) -> bool:
    """vLLM advertises itself via x-vllm-* response headers / Server string."""
    joined = " ".join(f"{k}: {v}" for k, v in headers.items()).lower()
    return "x-vllm" in joined or ("server:" in joined and "vllm" in joined)


async def _probe_openai_compatible(
    base_url: str, api_key: str, model: str
) -> tuple[bool, float, list[str], bool, str | None]:
    """Probe an OpenAI-compatible endpoint.

    1) GET {base_url}/models — preferred (yields the model list).
    2) On 404/405/403 fall back to a 1-token chat completion so internal
       proxies that hide /models still verify end-to-end + real latency.

    Returns (ok, latency_ms, models, models_list_available, error).
    """
    url = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=_TEST_TIMEOUT) as client:
        t0 = time.perf_counter()
        try:
            resp = await client.get(f"{url}/models", headers=_auth_headers(api_key))
        except httpx.HTTPError as exc:
            return False, 0.0, [], False, False, f"connection failed: {exc}"
        latency_ms = round((time.perf_counter() - t0) * 1000)

        vllm_hint = _looks_like_vllm(resp.headers)

        if resp.status_code == 200:
            try:
                data = resp.json()
                models = [
                    m.get("id", "") for m in data.get("data", []) if m.get("id")
                ]
                return True, latency_ms, models, True, vllm_hint, None
            except ValueError:
                pass  # 200 with non-JSON body — fall through to completion probe

        # Proxy hides /models (404/405/403) → minimal completion fallback.
        t0 = time.perf_counter()
        try:
            comp = await client.post(
                f"{url}/chat/completions",
                headers=_auth_headers(api_key),
                json={
                    "model": model or "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )
        except httpx.HTTPError as exc:
            return False, latency_ms, [], False, vllm_hint, f"/models {resp.status_code}; completion failed: {exc}"
        latency_ms = round((time.perf_counter() - t0) * 1000)
        if comp.status_code == 200:
            return True, latency_ms, [], False, vllm_hint or _looks_like_vllm(comp.headers), None
        detail = ""
        try:
            detail = str(comp.json())[:300]
        except ValueError:
            detail = comp.text[:300]
        return (
            False,
            latency_ms,
            [],
            False,
            vllm_hint,
            f"/models HTTP {resp.status_code}; completion HTTP {comp.status_code}: {detail}",
        )


async def _test_connection(req: LLMConfigTestRequest) -> dict:
    """Full connectivity test per plan §5.5 (list-models → 1-token fallback).

    Health-style services (``kind`` = embed_rerank / stt) expose custom APIs
    with no OpenAI endpoints — they are probed via ``GET {origin}/health``.
    """
    kind = (getattr(req, "kind", "") or "").lower().strip()
    if kind in ("embed_rerank", "stt"):
        from urllib.parse import urlparse

        origin = "{0.scheme}://{0.netloc}".format(urlparse(req.base_url))
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=_TEST_TIMEOUT) as client:
                resp = await client.get(
                    f"{origin}/health", headers=_auth_headers(req.api_key)
                )
        except httpx.HTTPError as exc:
            return {
                "ok": False, "latency_ms": 0, "models": [],
                "models_list_available": False,
                "error": f"health probe failed: {exc}",
            }
        latency_ms = round((time.perf_counter() - t0) * 1000)
        ok = resp.status_code == 200
        return {
            "ok": ok,
            "latency_ms": latency_ms,
            "models": [],
            "models_list_available": False,
            **({} if ok else {"error": f"health probe HTTP {resp.status_code}"}),
        }

    provider = req.provider.lower().strip()

    if provider not in _VALID_PROVIDERS:
        return {
            "ok": False, "latency_ms": 0, "models": [],
            "models_list_available": False,
            "error": f"Unknown provider {req.provider!r}. Supported: {sorted(_VALID_PROVIDERS)}",
        }

    try:
        if provider == "gemini":
            t0 = time.perf_counter()
            models = await _list_models_gemini(req.api_key)
            return {
                "ok": True,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "models": models[:50],
                "models_list_available": True,
            }
        if provider == "ollama":
            t0 = time.perf_counter()
            models = await _list_models_ollama(req.base_url)
            return {
                "ok": True,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "models": models[:50],
                "models_list_available": True,
            }
        # openai_compatible
        ok, latency_ms, models, listed, vllm_hint, error = await _probe_openai_compatible(
            req.base_url, req.api_key, req.model
        )
        result: dict = {
            "ok": ok,
            "latency_ms": latency_ms,
            "models": models[:50],
            "models_list_available": listed,
            "is_vllm_hint": vllm_hint,
        }
        if not ok:
            result["error"] = error
        return result
    except httpx.HTTPError as exc:
        return {
            "ok": False, "latency_ms": 0, "models": [],
            "models_list_available": False,
            "error": f"connection failed: {exc}",
        }
    except ValueError as exc:
        return {
            "ok": False, "latency_ms": 0, "models": [],
            "models_list_available": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Model-list cache (POST /models — cheap listing, no completion ping)
# ---------------------------------------------------------------------------

_MODELS_CACHE_TTL = 300.0  # seconds
_models_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}


def _cache_get(key: tuple[str, str]) -> list[str] | None:
    hit = _models_cache.get(key)
    if hit and (time.monotonic() - hit[0]) < _MODELS_CACHE_TTL:
        return hit[1]
    return None


def _cache_put(key: tuple[str, str], models: list[str]) -> None:
    if len(_models_cache) > 256:  # bound memory; entries are tiny but be tidy
        _models_cache.clear()
    _models_cache[key] = (time.monotonic(), models)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("")
async def get_llm_config(user: User = Depends(require_superadmin)):
    """V2 state: role assignments + connections (API keys masked — plaintext
    never leaves the server)."""
    overrides = await runtime_config.list_overrides()
    roles: dict[str, dict] = {}
    for role in runtime_config.ROLES:
        cfg = runtime_config.get_effective_sync(role)
        ov = overrides.get(role, {})
        roles[role] = {
            "conn_id": ov.get("conn_id", "@env"),
            "model": cfg.model,
            "source": cfg.source,
            "resolved": {
                "provider": cfg.provider,
                "base_url": cfg.base_url,
                "model": cfg.model,
                "masked_api_key": _mask_key(cfg.api_key),
            },
            "updated_at": ov.get("updated_at"),
            "updated_by": ov.get("updated_by"),
        }

    conns = await runtime_config.list_connections()
    for cid, meta in conns.items():
        if meta["has_api_key"]:
            conn = await runtime_config.get_connection(cid)
            meta["masked_api_key"] = _mask_key(conn["api_key"]) if conn else ""

    return {
        "roles": roles,
        "connections": conns,
        "version": runtime_config.snapshot_version(),
    }


@router.put("/connections/{conn_id}")
async def upsert_connection(
    conn_id: str,
    body: ConnectionUpsert,
    user: User = Depends(require_superadmin),
):
    """Create/update a model connection (endpoint declared once, referenced by
    many roles). Blank api_key keeps the previously stored key."""
    try:
        await runtime_config.set_connection(
            conn_id,
            {
                "name": body.name.strip() or conn_id,
                "provider": body.provider,
                "base_url": body.base_url.strip(),
                "api_key": body.api_key or "",
                "extra": dict(body.extra or {}),
            },
            actor_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    await AuditService.record_for_actor(
        user,
        action="llm_conn.set",
        resource_type="llm_connection",
        resource_id=conn_id,
        resource_label=f"{body.name or conn_id} → {body.provider}/{body.base_url}",
        summary=f"LLM connection '{conn_id}' saved ({body.provider}, {body.base_url!r})",
        extra={"provider": body.provider, "base_url": body.base_url},
    )
    return {"ok": True, "conn_id": conn_id, "version": runtime_config.snapshot_version()}


@router.delete("/connections/{conn_id}")
async def delete_connection_endpoint(
    conn_id: str,
    force: bool = False,
    user: User = Depends(require_superadmin),
):
    """Delete a connection. 409 with the referencing-role list when still in use
    unless ``force=true`` (roles then fall back to .env via dangling-ref guard)."""
    try:
        refs = await runtime_config.delete_connection(conn_id, actor_id=user.id, force=force)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if refs and not force:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"Connection '{conn_id}' is referenced by roles: {', '.join(refs)}",
                "referencing_roles": refs,
            },
        )

    await AuditService.record_for_actor(
        user,
        action="llm_conn.delete",
        resource_type="llm_connection",
        resource_id=conn_id,
        summary=f"LLM connection '{conn_id}' deleted (force={force})",
    )
    return ConnectionDeleteResponse(deleted=True, referencing_roles=[]).model_dump()


@router.put("/{role}")
async def assign_role(
    role: str,
    body: RoleAssignment,
    user: User = Depends(require_superadmin),
):
    """Assign a connection (+model) to a role. V2 shape: {conn_id, model}.
    The connection itself was tested when created/saved — no re-probe here."""
    if role not in runtime_config.ROLES:
        raise HTTPException(status_code=404, detail=f"Unknown LLM role: {role}")

    conn_id = body.conn_id.strip()
    if conn_id != "@env" and conn_id not in await runtime_config.list_connections():
        raise HTTPException(status_code=400, detail=f"Unknown connection {conn_id!r}")

    try:
        await runtime_config.set_override(
            role, {"conn_id": conn_id, "model": body.model.strip(), "extra": body.extra},
            actor_id=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    await AuditService.record_for_actor(
        user,
        action="llm_config.assign",
        resource_type="llm_config",
        resource_id=role,
        resource_label=f"{role} → {conn_id}/{body.model}",
        summary=f"Role '{role}' assigned to connection '{conn_id}' (model={body.model!r})",
        extra={"conn_id": conn_id, "model": body.model},
    )

    cfg = runtime_config.get_effective_sync(role)
    return {
        "ok": True,
        "role": role,
        "conn_id": conn_id,
        "resolved": {
            "provider": cfg.provider,
            "base_url": cfg.base_url,
            "model": cfg.model,
            "masked_api_key": _mask_key(cfg.api_key),
        },
        "version": runtime_config.snapshot_version(),
    }


@router.delete("/{role}")
async def clear_llm_config(role: str, user: User = Depends(require_superadmin)):
    """Remove a role's DB override — falls back to .env defaults."""
    if role not in runtime_config.ROLES:
        raise HTTPException(status_code=404, detail=f"Unknown LLM role: {role}")

    await runtime_config.clear_override(role, actor_id=user.id)

    await AuditService.record_for_actor(
        user,
        action="llm_config.clear",
        resource_type="llm_config",
        resource_id=role,
        summary=f"LLM override cleared for role '{role}' (back to .env defaults)",
    )
    return {"role": role, "cleared": True, "version": runtime_config.snapshot_version()}


@router.post("/test")
async def test_llm_connection(
    body: LLMConfigTestRequest,
    user: User = Depends(require_superadmin),  # noqa: ARG001 — auth gate only
):
    """Trial connectivity check for an unsaved config (UI 'Test' button)."""
    return await _test_connection(body)


@router.post("/models")
async def list_provider_models(
    body: ModelsListRequest,
    user: User = Depends(require_superadmin),  # noqa: ARG001 — auth gate only
):
    """Auto-load the model catalogue from an endpoint (plan §7.1, §12.5).

    Accepts either ``{conn_id}`` (resolve credentials from the stored
    connection) or a raw {provider, base_url, api_key} triple. Light probe only
    — never sends a completion. Cached ~5 min per (base_url, key-hash); proxies
    that hide /models return source="none" and the UI switches to free-text.
    """
    provider = (body.provider or "").lower().strip()
    base_url = body.base_url.strip()
    api_key = body.api_key or ""

    if body.conn_id:
        try:
            conn = await runtime_config.get_connection(body.conn_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if conn is None:
            return {
                "ok": False, "models": [], "source": "none",
                "error": f"Unknown connection {body.conn_id!r}",
            }
        provider = conn["provider"]
        base_url = conn["base_url"]
        api_key = conn["api_key"]
        # Health-style services have no model catalogue — UI falls back to
        # free-text model input.
        if (conn.get("extra") or {}).get("kind") in ("embed_rerank", "stt"):
            return {"ok": True, "models": [], "source": "none"}

    if not provider:
        return {
            "ok": False, "models": [], "source": "none",
            "error": "Either conn_id or provider is required",
        }

    cache_key = (
        base_url.rstrip("/").lower(),
        hashlib.sha256(api_key.encode()).hexdigest()[:16],
    ) if provider == "openai_compatible" else (f"{provider}:{base_url}", "")

    cached = _cache_get(cache_key)
    if cached is not None:
        return {"ok": True, "models": cached[:50], "source": "endpoint"}

    try:
        if provider == "gemini":
            models = await _list_models_gemini(api_key)
        elif provider == "ollama":
            models = await _list_models_ollama(base_url)
        elif provider == "openai_compatible":
            url = base_url.rstrip("/")
            async with httpx.AsyncClient(timeout=_TEST_TIMEOUT) as client:
                resp = await client.get(
                    f"{url}/models", headers=_auth_headers(api_key)
                )
            if resp.status_code != 200:
                return {"ok": False, "models": [], "source": "none"}
            models = [
                m.get("id", "") for m in resp.json().get("data", []) if m.get("id")
            ]
        else:
            return {
                "ok": False, "models": [], "source": "none",
                "error": f"Unknown provider {provider!r}",
            }
    except (httpx.HTTPError, ValueError):
        return {"ok": False, "models": [], "source": "none"}

    if not models:
        return {"ok": False, "models": [], "source": "none"}
    _cache_put(cache_key, models)
    return {"ok": True, "models": models[:50], "source": "endpoint"}

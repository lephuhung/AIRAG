"""
Integrations API — Telegram webhook + linking, and generic API-key management.

Routes (mounted under /api/v1):
  POST   /integrations/telegram/webhook        (Telegram → us; secret-header auth)
  POST   /integrations/telegram/link-code      (web user mints a one-time link code)
  GET    /integrations/api-keys                 (list the caller's API keys)
  POST   /integrations/api-keys                 (create a key; plaintext shown once)
  DELETE /integrations/api-keys/{key_id}        (revoke a key)
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_active_user, get_db, require_superadmin
from app.core.security import generate_api_key, verify_totp
from app.models.integration import ApiKey, TelegramBotConfig, TelegramLink
from app.models.user import User
from app.services.integrations import telegram_service

router = APIRouter(prefix="/integrations", tags=["integrations"])

# Path Telegram POSTs updates to (used to suggest a default webhook URL in the UI).
TELEGRAM_WEBHOOK_PATH = "/api/v1/integrations/telegram/webhook"


def _public_webhook_url(request: Request) -> str:
    """Build the public Telegram webhook URL.

    Prefers the configured public origin (`PUBLIC_BASE_URL`, i.e. the Cloudflare
    tunnel domain) so the UI suggests a reachable HTTPS URL instead of the
    internal `http://backend:8080`. Falls back to the request's own base URL.
    """
    base = (settings.PUBLIC_BASE_URL or str(request.base_url)).rstrip("/")
    return base + TELEGRAM_WEBHOOK_PATH


# ──────────────────────────────── Telegram ─────────────────────────────────────

@router.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    """Receive a Telegram update.

    Validates the secret token (set when registering the webhook), then processes
    the update in the background so Telegram receives an immediate 200 and does
    not retry. Always returns {"ok": true} unless the secret is wrong.

    The expected secret comes from the DB-backed bot config (admin UI), falling
    back to the legacy `.env` value for backward compatibility.
    """
    cfg = await telegram_service.get_bot_config(db)
    expected_secret = (cfg.webhook_secret if cfg else None) or settings.TELEGRAM_WEBHOOK_SECRET
    if expected_secret:
        if x_telegram_bot_api_secret_token != expected_secret:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Bad secret token")

    try:
        update = await request.json()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    background_tasks.add_task(telegram_service.process_update, update)
    return {"ok": True}


class LinkCodeRequest(BaseModel):
    # Linking a Telegram chat grants it the user's full permissions, so we gate
    # minting a code behind a fresh 2FA check (2FA must also be enabled first).
    totp_code: str = Field(..., min_length=6, max_length=6)


class LinkCodeResponse(BaseModel):
    code: str
    expires_at: datetime
    deep_link: str | None = None
    ttl_minutes: int


@router.post("/telegram/link-code", response_model=LinkCodeResponse)
async def create_telegram_link_code(
    body: LinkCodeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Mint a one-time code the logged-in user pastes into the bot to link their chat.

    Requires the account to have two-factor (TOTP) enabled AND a valid current
    code — linking a chat hands it the user's full access, so it is a sensitive op.
    """
    if not current_user.totp_enabled:
        # Sentinel the frontend recognises → tell the user to enable 2FA first.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="TWO_FACTOR_NOT_ENABLED",
        )
    if not verify_totp(current_user.totp_secret or "", body.totp_code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid two-factor code",
        )

    code, expires_at = await telegram_service.create_link_code(db, current_user.id)
    cfg = await telegram_service.get_bot_config(db)
    return LinkCodeResponse(
        code=code,
        expires_at=expires_at,
        deep_link=telegram_service.build_deep_link(code, cfg.bot_username if cfg else None),
        ttl_minutes=settings.TELEGRAM_LINK_CODE_TTL_MINUTES,
    )


class TelegramLinkInfo(BaseModel):
    telegram_chat_id: str
    telegram_user_id: str | None = None
    telegram_username: str | None = None
    active_workspace_id: uuid.UUID | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("/telegram/links", response_model=list[TelegramLinkInfo])
async def list_telegram_links(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List Telegram chats linked to the current user's account."""
    rows = (
        await db.execute(
            select(TelegramLink)
            .where(TelegramLink.user_id == current_user.id)
            .order_by(TelegramLink.created_at.desc())
        )
    ).scalars().all()
    return list(rows)


class TelegramLinkAdminInfo(BaseModel):
    """A linked Telegram account plus the owning user — for the admin logs view."""

    telegram_chat_id: str
    telegram_user_id: str | None = None
    telegram_username: str | None = None
    user_id: uuid.UUID
    user_email: str | None = None
    user_name: str | None = None
    created_at: datetime


@router.get("/telegram/links/all", response_model=list[TelegramLinkAdminInfo])
async def list_all_telegram_links(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_superadmin),
):
    """List ALL Telegram links across every user (superadmin). Powers the
    Telegram directory tab on the system-logs page."""
    rows = (
        await db.execute(
            select(TelegramLink, User.email, User.full_name)
            .join(User, User.id == TelegramLink.user_id)
            .order_by(TelegramLink.created_at.desc())
        )
    ).all()
    return [
        TelegramLinkAdminInfo(
            telegram_chat_id=link.telegram_chat_id,
            telegram_user_id=link.telegram_user_id,
            telegram_username=link.telegram_username,
            user_id=link.user_id,
            user_email=email,
            user_name=name,
            created_at=link.created_at,
        )
        for link, email, name in rows
    ]


@router.delete("/telegram/links/{chat_id}", status_code=status.HTTP_200_OK)
async def unlink_telegram(
    chat_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Unlink a Telegram chat from the current user's account."""
    row = (
        await db.execute(
            select(TelegramLink).where(TelegramLink.telegram_chat_id == chat_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Link not found")
    if row.user_id != current_user.id and not current_user.is_superadmin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your link")
    await db.delete(row)
    await db.commit()
    return {"status": "unlinked", "telegram_chat_id": chat_id}


# ─────────────────────── Telegram bot config (superadmin) ──────────────────────

class TelegramConfigInfo(BaseModel):
    """Safe view of the bot config — never exposes the raw token."""

    enabled: bool = True
    bot_username: str | None = None
    bot_id: str | None = None
    webhook_url: str | None = None
    has_token: bool = False
    token_hint: str | None = None  # last 4 chars, for "is this the right token?"
    has_secret: bool = False
    suggested_webhook_url: str | None = None
    updated_at: datetime | None = None


class TelegramConfigUpdate(BaseModel):
    # All optional — only provided fields are changed. An empty string clears.
    bot_token: str | None = None
    webhook_secret: str | None = None  # "" → auto-generate a strong secret
    bot_username: str | None = None
    webhook_url: str | None = None
    enabled: bool | None = None


def _config_info(cfg: TelegramBotConfig | None, request: Request) -> TelegramConfigInfo:
    suggested = _public_webhook_url(request)
    if cfg is None:
        return TelegramConfigInfo(suggested_webhook_url=suggested)
    token = cfg.bot_token or ""
    return TelegramConfigInfo(
        enabled=cfg.enabled,
        bot_username=cfg.bot_username,
        bot_id=cfg.bot_id,
        webhook_url=cfg.webhook_url,
        has_token=bool(token),
        token_hint=token[-4:] if token else None,
        has_secret=bool(cfg.webhook_secret),
        suggested_webhook_url=cfg.webhook_url or suggested,
        updated_at=cfg.updated_at,
    )


async def _get_or_create_config(db: AsyncSession) -> TelegramBotConfig:
    cfg = await telegram_service.get_bot_config(db)
    if cfg is None:
        cfg = TelegramBotConfig(id=1)
        db.add(cfg)
    return cfg


@router.get("/telegram/config", response_model=TelegramConfigInfo)
async def get_telegram_config(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_superadmin),
):
    """Return the system-wide bot config (token masked). Superadmin only."""
    cfg = await telegram_service.get_bot_config(db)
    return _config_info(cfg, request)


@router.put("/telegram/config", response_model=TelegramConfigInfo)
async def update_telegram_config(
    payload: TelegramConfigUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superadmin),
):
    """Create/update the bot config. If a token is supplied, auto-detect the bot
    username/id via getMe. An empty `webhook_secret` auto-generates a strong one."""
    cfg = await _get_or_create_config(db)

    if payload.bot_token is not None:
        cfg.bot_token = payload.bot_token.strip() or None
        # Verify the token + auto-fill identity.
        if cfg.bot_token:
            identity = await telegram_service.fetch_bot_identity(cfg.bot_token)
            if identity is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Token Telegram không hợp lệ (getMe thất bại).",
                )
            cfg.bot_username = identity.get("username") or cfg.bot_username
            cfg.bot_id = str(identity.get("id")) if identity.get("id") is not None else cfg.bot_id

    if payload.webhook_secret is not None:
        cfg.webhook_secret = payload.webhook_secret.strip() or secrets.token_urlsafe(24)

    if payload.bot_username is not None:
        cfg.bot_username = payload.bot_username.strip().lstrip("@") or None

    if payload.webhook_url is not None:
        cfg.webhook_url = payload.webhook_url.strip() or None

    if payload.enabled is not None:
        cfg.enabled = payload.enabled

    cfg.updated_by = admin.id
    await db.commit()
    await db.refresh(cfg)
    return _config_info(cfg, request)


class WebhookActionResult(BaseModel):
    ok: bool
    detail: str
    config: TelegramConfigInfo | None = None


@router.post("/telegram/config/register-webhook", response_model=WebhookActionResult)
async def register_telegram_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superadmin),
):
    """Register the webhook with Telegram using the stored URL + secret.

    Uses the configured `webhook_url` (or the suggested default), generating a
    secret if one isn't set yet.
    """
    cfg = await _get_or_create_config(db)
    if not cfg.bot_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chưa cấu hình bot token.")

    url = cfg.webhook_url or _public_webhook_url(request)
    if not cfg.webhook_secret:
        cfg.webhook_secret = secrets.token_urlsafe(24)
    cfg.webhook_url = url

    result = await telegram_service.register_webhook(cfg.bot_token, url, cfg.webhook_secret)
    cfg.updated_by = admin.id
    await db.commit()
    await db.refresh(cfg)

    ok = bool(result.get("ok"))
    detail = result.get("description") or ("Đã đăng ký webhook." if ok else "Đăng ký webhook thất bại.")
    return WebhookActionResult(ok=ok, detail=detail, config=_config_info(cfg, request))


@router.post("/telegram/config/test", response_model=WebhookActionResult)
async def test_telegram_connection(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_superadmin),
):
    """Check the bot token (getMe) and current webhook registration (getWebhookInfo)."""
    cfg = await telegram_service.get_bot_config(db)
    if cfg is None or not cfg.bot_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chưa cấu hình bot token.")

    identity = await telegram_service.fetch_bot_identity(cfg.bot_token)
    if identity is None:
        return WebhookActionResult(
            ok=False, detail="Token không hợp lệ (getMe thất bại).", config=_config_info(cfg, request)
        )

    info = await telegram_service.fetch_webhook_info(cfg.bot_token)
    wi = info.get("result") or {}
    hooked_url = wi.get("url") or ""
    pending = wi.get("pending_update_count", 0)
    last_err = wi.get("last_error_message")

    parts = [f"Bot @{identity.get('username')}"]
    if hooked_url:
        parts.append(f"webhook: {hooked_url}")
        if pending:
            parts.append(f"{pending} update đang chờ")
        if last_err:
            parts.append(f"lỗi gần nhất: {last_err}")
    else:
        parts.append("webhook: CHƯA đăng ký")
    ok = bool(hooked_url) and not last_err
    return WebhookActionResult(ok=ok, detail=" · ".join(parts), config=_config_info(cfg, request))


# ──────────────────────────────── API keys ─────────────────────────────────────

class ApiKeyCreate(BaseModel):
    name: str = Field(default="API Key", max_length=255)
    scopes: list[str] | None = None


class ApiKeyInfo(BaseModel):
    id: uuid.UUID
    name: str
    prefix: str
    scopes: list[str] | None = None
    revoked: bool
    last_used_at: datetime | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ApiKeyCreated(ApiKeyInfo):
    # Plaintext key — returned ONLY at creation time, never again.
    key: str


@router.get("/api-keys", response_model=list[ApiKeyInfo])
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List API keys owned by (acting as) the current user."""
    rows = (
        await db.execute(
            select(ApiKey)
            .where(ApiKey.user_id == current_user.id)
            .order_by(ApiKey.created_at.desc())
        )
    ).scalars().all()
    return list(rows)


@router.post("/api-keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: ApiKeyCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create an API key that authenticates AS the current user.

    The plaintext key is returned exactly once — store it now; only its hash is kept.
    """
    plaintext, key_hash, prefix = generate_api_key()
    row = ApiKey(
        key_hash=key_hash,
        prefix=prefix,
        name=request.name,
        user_id=current_user.id,
        created_by=current_user.id,
        scopes=request.scopes,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return ApiKeyCreated(key=plaintext, **ApiKeyInfo.model_validate(row).model_dump())


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_200_OK)
async def revoke_api_key(
    key_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Revoke (soft-delete) an API key. Superadmins may revoke any key."""
    row = (
        await db.execute(select(ApiKey).where(ApiKey.id == key_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    if row.user_id != current_user.id and not current_user.is_superadmin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your API key")
    row.revoked = True
    await db.commit()
    return {"status": "revoked", "id": str(key_id)}

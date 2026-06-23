"""
Auth API — register, login, refresh, profile.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_user, get_current_active_user
from app.core.exceptions import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    UnauthorizedError,
)
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_totp_secret,
    totp_provisioning_uri,
    totp_qr_data_uri,
    verify_totp,
)
from app.models.user import User
from app.models.tenant import Tenant, TenantUser
from app.models.invite_token import InviteToken
from app.models.knowledge_base import KnowledgeBase
from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    RefreshResponse,
    UpdateProfileRequest,
    TwoFASetupResponse,
    TwoFAEnableRequest,
    TwoFADisableRequest,
    TwoFAStatusResponse,
)
from app.schemas.user import UserResponse
from app.services.storage_service import get_storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user account. Account is inactive until admin approves,
    unless registering via a valid invite link (auto-activated)."""
    # Check email uniqueness
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    if result.scalar_one_or_none() is not None:
        raise ConflictError("Email already registered")

    # ── Invite token flow ──────────────────────────────────────────────
    invite: InviteToken | None = None
    if body.invite_token:
        result = await db.execute(
            select(InviteToken).where(InviteToken.token == body.invite_token)
        )
        invite = result.scalar_one_or_none()

        if invite is None or not invite.is_active:
            raise BadRequestError("Invalid or expired invite link")

        if datetime.utcnow() > invite.expires_at:
            raise BadRequestError("Invite link has expired")

        if invite.max_uses is not None and invite.use_count >= invite.max_uses:
            raise BadRequestError("Invite link has reached its maximum number of uses")

        if invite.email and invite.email.lower() != body.email.lower().strip():
            raise BadRequestError(
                "This invite link is restricted to a different email address"
            )

        # Verify the tenant is still active
        result = await db.execute(
            select(Tenant).where(
                Tenant.id == invite.tenant_id, Tenant.is_active.is_(True)
            )
        )
        if result.scalar_one_or_none() is None:
            raise BadRequestError(
                "The organization for this invite is no longer active"
            )

    # ── Create user ────────────────────────────────────────────────────
    user = User(
        email=body.email.lower().strip(),
        password_hash=hash_password(body.password),
        full_name=body.full_name.strip(),
        is_active=True if invite else False,  # Auto-activate for invite registrations
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # ── Auto-create personal workspace for the new user ───────────────
    # Short, friendly name based on the user's given name (last token of a
    # Vietnamese full name, e.g. "Lê Văn Hưng" → "Hưng Space"). Fall back to the
    # email local part if no full name was provided.
    _full = (user.full_name or "").strip()
    _given = _full.split()[-1] if _full else user.email.split("@")[0]
    personal_kb = KnowledgeBase(
        name=f"{_given} Space",
        owner_id=user.id,
        visibility="personal",
        is_default=True,
    )
    db.add(personal_kb)
    await db.commit()
    logger.info(
        f"Auto-created personal workspace for user {user.email} (kb_id={personal_kb.id})"
    )

    # ── Invite: auto-approve tenant membership + increment use_count ──
    if invite:
        tenant_user = TenantUser(
            tenant_id=invite.tenant_id,
            user_id=user.id,
            role=invite.role,
            is_approved=True,
        )
        db.add(tenant_user)
        invite.use_count += 1
        await db.commit()
        logger.info(
            f"User {user.email} registered via invite (auto-activated), "
            f"tenant_id={invite.tenant_id}, role={invite.role}"
        )
    elif body.tenant_slug:
        # Standard flow: create pending membership
        result = await db.execute(
            select(Tenant).where(
                Tenant.slug == body.tenant_slug, Tenant.is_active.is_(True)
            )
        )
        tenant = result.scalar_one_or_none()
        if tenant:
            tenant_user = TenantUser(
                tenant_id=tenant.id,
                user_id=user.id,
                role="member",
                is_approved=False,
            )
            db.add(tenant_user)
            await db.commit()
            logger.info(
                f"User {user.email} registered with pending membership to tenant '{body.tenant_slug}'"
            )
        else:
            logger.warning(
                f"Tenant slug '{body.tenant_slug}' not found during registration"
            )

    logger.info(
        f"User registered: {user.email} (id={user.id}, active={user.is_active})"
    )
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Login with email/password → get JWT tokens."""
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.password_hash):
        raise UnauthorizedError("Invalid email or password")

    if not user.is_active:
        raise ForbiddenError(
            "Account not yet approved. Please wait for admin approval."
        )

    # ── Two-factor gate (TOTP / Google Authenticator) ─────────────────────
    if user.totp_enabled:
        if not body.totp_code:
            # Sentinel the frontend recognises to prompt for the 6-digit code.
            raise UnauthorizedError("TWO_FACTOR_REQUIRED")
        if not verify_totp(user.totp_secret or "", body.totp_code):
            raise UnauthorizedError("Invalid two-factor code")

    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.model_validate(user),
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    """Refresh access token using a valid refresh token."""
    from jose import JWTError

    try:
        payload = decode_token(body.refresh_token)
    except JWTError:
        raise UnauthorizedError("Invalid or expired refresh token")

    if payload.get("type") != "refresh":
        raise UnauthorizedError("Invalid token type")

    user_id = payload.get("sub")
    if user_id is None:
        raise UnauthorizedError("Invalid token payload")

    try:
        user_uuid = uuid.UUID(user_id)
    except (ValueError, AttributeError):
        raise UnauthorizedError("Invalid token payload")

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise UnauthorizedError("User not found or inactive")

    new_access_token = create_access_token(user.id)

    return RefreshResponse(access_token=new_access_token)


@router.get("/me", response_model=UserResponse)
async def get_me(
    user: User = Depends(get_current_active_user),
):
    """Get current user profile."""
    return user


@router.put("/me", response_model=UserResponse)
async def update_me(
    body: UpdateProfileRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update current user profile (name, password)."""
    if body.full_name is not None:
        user.full_name = body.full_name.strip()

    # Password change requires current_password verification
    if body.new_password is not None:
        if body.current_password is None:
            raise BadRequestError("current_password is required to change password")
        if not verify_password(body.current_password, user.password_hash):
            raise BadRequestError("Current password is incorrect")
        user.password_hash = hash_password(body.new_password)

    # Shallow-merge free-form preferences (e.g. {"tts": {...}}). Reassign a new
    # dict so SQLAlchemy detects the JSONB mutation.
    if body.settings is not None:
        user.settings = {**(user.settings or {}), **body.settings}

    await db.commit()
    await db.refresh(user)
    return user


# Allowed MIME types for avatar uploads
_AVATAR_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_AVATAR_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


@router.post("/me/avatar", response_model=UserResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload or replace the current user's avatar image.

    Accepts JPEG, PNG, GIF, or WebP up to 5 MB.
    Returns the updated user object with a presigned avatar_url.
    """
    if file.content_type not in _AVATAR_ALLOWED_TYPES:
        raise BadRequestError(
            f"Unsupported image type '{file.content_type}'. "
            "Allowed: jpeg, png, gif, webp."
        )

    data = await file.read()
    if len(data) > _AVATAR_MAX_BYTES:
        raise BadRequestError("Avatar image must be smaller than 5 MB")

    # Derive file extension from content_type (image/jpeg → .jpg etc.)
    _ext_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    ext = _ext_map.get(
        file.content_type, os.path.splitext(file.filename or "")[1] or ".jpg"
    )

    storage = get_storage_service()
    avatar_url = await storage.upload_avatar(user.id, data, file.content_type, ext)

    user.avatar_url = avatar_url
    await db.commit()
    await db.refresh(user)
    logger.info(f"User {user.id} uploaded avatar ({len(data)} bytes)")
    return user


# ── Two-Factor Auth (TOTP / Google Authenticator) ──────────────────────────


@router.get("/2fa/status", response_model=TwoFAStatusResponse)
async def two_fa_status(
    user: User = Depends(get_current_active_user),
):
    """Whether TOTP two-factor is currently active for the user."""
    return TwoFAStatusResponse(enabled=bool(user.totp_enabled))


@router.post("/2fa/setup", response_model=TwoFASetupResponse)
async def two_fa_setup(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Begin enrollment: mint a fresh secret, persist it (pending), and return the
    QR + manual key. 2FA is NOT active yet — the user must confirm via /2fa/enable.

    Re-running this before enabling rotates the secret, which is fine. It is
    rejected once 2FA is already enabled (disable first to re-enroll).
    """
    if user.totp_enabled:
        raise BadRequestError(
            "Two-factor is already enabled. Disable it first to re-enroll."
        )

    secret = generate_totp_secret()
    uri = totp_provisioning_uri(secret, user.email)

    user.totp_secret = secret  # stored pending verification; totp_enabled stays false
    await db.commit()

    logger.info(f"User {user.id} started 2FA enrollment")
    return TwoFASetupResponse(
        secret=secret,
        otpauth_uri=uri,
        qr_data_uri=totp_qr_data_uri(uri),
    )


@router.post("/2fa/enable", response_model=TwoFAStatusResponse)
async def two_fa_enable(
    body: TwoFAEnableRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Confirm enrollment by submitting a valid code → activate 2FA."""
    if user.totp_enabled:
        raise BadRequestError("Two-factor is already enabled.")
    if not user.totp_secret:
        raise BadRequestError("Start setup first via /auth/2fa/setup.")
    if not verify_totp(user.totp_secret, body.code):
        raise BadRequestError("Invalid code. Check your authenticator app and try again.")

    user.totp_enabled = True
    await db.commit()
    await db.refresh(user)
    logger.info(f"User {user.id} enabled 2FA")
    return TwoFAStatusResponse(enabled=True)


@router.post("/2fa/disable", response_model=TwoFAStatusResponse)
async def two_fa_disable(
    body: TwoFADisableRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Turn 2FA off. Requires proof of ownership: a current TOTP code OR the
    account password. Clears the stored secret."""
    if not user.totp_enabled:
        # Idempotent: already off. Also clear any dangling pending secret.
        if user.totp_secret:
            user.totp_secret = None
            await db.commit()
        return TwoFAStatusResponse(enabled=False)

    verified = False
    if body.code and verify_totp(user.totp_secret or "", body.code):
        verified = True
    elif body.password and verify_password(body.password, user.password_hash):
        verified = True

    if not verified:
        raise BadRequestError(
            "Verification failed. Provide a valid authenticator code or your password."
        )

    user.totp_enabled = False
    user.totp_secret = None
    await db.commit()
    await db.refresh(user)
    logger.info(f"User {user.id} disabled 2FA")
    return TwoFAStatusResponse(enabled=False)

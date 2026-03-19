"""
Auth API — register, login, refresh, profile.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_user, get_current_active_user
from app.core.exceptions import BadRequestError, ConflictError, ForbiddenError, UnauthorizedError
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.models.user import User
from app.models.tenant import Tenant, TenantUser
from app.models.invite_token import InviteToken
from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    RefreshResponse,
    UpdateProfileRequest,
)
from app.schemas.user import UserResponse
from app.services.storage_service import get_storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
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
            raise BadRequestError("This invite link is restricted to a different email address")

        # Verify the tenant is still active
        result = await db.execute(
            select(Tenant).where(Tenant.id == invite.tenant_id, Tenant.is_active.is_(True))
        )
        if result.scalar_one_or_none() is None:
            raise BadRequestError("The organization for this invite is no longer active")

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
            select(Tenant).where(Tenant.slug == body.tenant_slug, Tenant.is_active.is_(True))
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
            logger.info(f"User {user.email} registered with pending membership to tenant '{body.tenant_slug}'")
        else:
            logger.warning(f"Tenant slug '{body.tenant_slug}' not found during registration")

    logger.info(f"User registered: {user.email} (id={user.id}, active={user.is_active})")
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
        raise ForbiddenError("Account not yet approved. Please wait for admin approval.")

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

    result = await db.execute(select(User).where(User.id == int(user_id)))
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
    ext = _ext_map.get(file.content_type, os.path.splitext(file.filename or "")[1] or ".jpg")

    storage = get_storage_service()
    avatar_url = await storage.upload_avatar(user.id, data, file.content_type, ext)

    user.avatar_url = avatar_url
    await db.commit()
    await db.refresh(user)
    logger.info(f"User {user.id} uploaded avatar ({len(data)} bytes)")
    return user

"""
Auth API — register, login, refresh, profile.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
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
from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    RefreshResponse,
    UpdateProfileRequest,
)
from app.schemas.user import UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user account. Account is inactive until admin approves."""
    # Check email uniqueness
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    if result.scalar_one_or_none() is not None:
        raise ConflictError("Email already registered")

    user = User(
        email=body.email.lower().strip(),
        password_hash=hash_password(body.password),
        full_name=body.full_name.strip(),
        is_active=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # If tenant_slug provided, create pending membership
    if body.tenant_slug:
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

    logger.info(f"User registered: {user.email} (id={user.id}, active=False)")
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
    if body.password is not None:
        user.password_hash = hash_password(body.password)

    await db.commit()
    await db.refresh(user)
    return user

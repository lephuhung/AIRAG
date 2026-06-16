"""
FastAPI dependency injection — database session + authentication.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import Depends
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.core.exceptions import UnauthorizedError, ForbiddenError, NotFoundError
from app.core.security import decode_token, hash_api_key


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# OAuth2 scheme — auto_error=False so we can return 401 manually
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# API key scheme — third-party clients send their key in the X-API-Key header.
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    """Decode JWT → fetch User from DB → return or raise 401."""
    from app.models.user import User

    if token is None:
        raise UnauthorizedError("Not authenticated")

    try:
        payload = decode_token(token)
    except JWTError:
        raise UnauthorizedError("Invalid or expired token")

    if payload.get("type") != "access":
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

    if user is None:
        raise UnauthorizedError("User not found")

    return user


async def get_current_active_user(
    user=Depends(get_current_user),
):
    """Check user.is_active → return or raise 403 'Account not approved'."""
    if not user.is_active:
        raise ForbiddenError("Account not yet approved. Please wait for admin approval.")
    return user


async def _user_from_api_key(api_key: str, db: AsyncSession):
    """Resolve and validate an API key → its (active) principal User, or raise 401."""
    from app.models.integration import ApiKey
    from app.models.user import User

    key_hash = hash_api_key(api_key)
    result = await db.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.revoked.is_(False))
    )
    api_key_row = result.scalar_one_or_none()
    if api_key_row is None:
        raise UnauthorizedError("Invalid or revoked API key")

    result = await db.execute(select(User).where(User.id == api_key_row.user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise UnauthorizedError("API key principal no longer exists")
    if not user.is_active:
        raise ForbiddenError("API key principal is not active")

    # Best-effort usage stamp — never let it break the request.
    try:
        api_key_row.last_used_at = datetime.utcnow()
        await db.commit()
    except Exception:
        await db.rollback()

    return user


async def get_principal(
    api_key: str | None = Depends(api_key_scheme),
    token: str | None = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    """Authenticate via X-API-Key (third parties) OR JWT bearer (browser).

    Returns the active principal User. API key takes precedence so service
    clients can pass both headers. The returned user carries the exact same
    workspace/tenant permissions whichever path was used.
    """
    if api_key:
        return await _user_from_api_key(api_key, db)

    user = await get_current_user(token=token, db=db)
    if not user.is_active:
        raise ForbiddenError("Account not yet approved. Please wait for admin approval.")
    return user


async def require_superadmin(
    user=Depends(get_current_active_user),
):
    """Check user.is_superadmin → return or raise 403."""
    if not user.is_superadmin:
        raise ForbiddenError("Superadmin access required")
    return user


async def verify_workspace_access(
    workspace_id: uuid.UUID,
    user=None,
    db: AsyncSession = None,
):
    """
    Check workspace exists AND user has access:
    - public: any authenticated user
    - tenant: user must be approved member of workspace.tenant_id
    - personal: user.id == workspace.owner_id
    - legacy (no owner): accessible to all authenticated users
    """
    from app.models.knowledge_base import KnowledgeBase
    from app.models.tenant import TenantUser

    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    kb = result.scalar_one_or_none()

    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    # If user is None (legacy compat during migration), skip access check
    if user is None:
        return kb

    # Superadmin can access everything
    if user.is_superadmin:
        return kb

    # Legacy workspaces without owner — accessible to all
    if kb.owner_id is None:
        return kb

    visibility = kb.visibility or "personal"

    if visibility == "public":
        return kb

    if visibility == "tenant":
        if kb.tenant_id is None:
            return kb  # No tenant set — treat as public
        # Check user is approved member of this tenant
        result = await db.execute(
            select(TenantUser).where(
                TenantUser.tenant_id == kb.tenant_id,
                TenantUser.user_id == user.id,
                TenantUser.is_approved.is_(True),
            )
        )
        if result.scalar_one_or_none() is not None:
            return kb
        raise ForbiddenError("You don't have access to this workspace")

    if visibility == "personal":
        if kb.owner_id == user.id:
            return kb
        raise ForbiddenError("You don't have access to this workspace")

    # Unknown visibility — deny
    raise ForbiddenError("You don't have access to this workspace")

"""
Tenant Management API
=====================
CRUD for tenants + user membership management.
SuperAdmin: create/update/delete tenants, assign tenant admins.
Tenant Admin: approve/reject/manage members.
Any authenticated: list own tenants.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_active_user, require_superadmin
from app.core.exceptions import NotFoundError, ForbiddenError, ConflictError, BadRequestError
from app.models.tenant import Tenant, TenantUser
from app.models.user import User
from app.schemas.tenant import (
    TenantCreate,
    TenantUpdate,
    TenantResponse,
    TenantUserResponse,
    RoleUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenants", tags=["tenants"])


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _get_tenant(tenant_id: int, db: AsyncSession) -> Tenant:
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if tenant is None:
        raise NotFoundError("Tenant", tenant_id)
    return tenant


async def _require_tenant_admin(tenant_id: int, user: User, db: AsyncSession) -> TenantUser:
    """Check user is admin of the given tenant."""
    if user.is_superadmin:
        # Superadmin has implicit admin access to all tenants
        return None  # type: ignore
    result = await db.execute(
        select(TenantUser).where(
            TenantUser.tenant_id == tenant_id,
            TenantUser.user_id == user.id,
            TenantUser.role == "admin",
            TenantUser.is_approved.is_(True),
        )
    )
    tu = result.scalar_one_or_none()
    if tu is None:
        raise ForbiddenError("Tenant admin access required")
    return tu


async def _build_tenant_user_response(tu: TenantUser, db: AsyncSession) -> TenantUserResponse:
    """Build TenantUserResponse with joined user info."""
    result = await db.execute(select(User).where(User.id == tu.user_id))
    user = result.scalar_one_or_none()
    return TenantUserResponse(
        id=tu.id,
        tenant_id=tu.tenant_id,
        user_id=tu.user_id,
        role=tu.role,
        is_approved=tu.is_approved,
        created_at=tu.created_at,
        email=user.email if user else None,
        full_name=user.full_name if user else None,
    )


# ── SuperAdmin: Tenant CRUD ─────────────────────────────────────────────────

@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    body: TenantCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Create a new tenant (SuperAdmin only)."""
    # Check slug uniqueness
    result = await db.execute(select(Tenant).where(Tenant.slug == body.slug))
    if result.scalar_one_or_none() is not None:
        raise ConflictError(f"Tenant with slug '{body.slug}' already exists")

    tenant = Tenant(
        name=body.name,
        slug=body.slug,
        domain=body.domain,
    )
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)
    logger.info(f"Tenant created: {tenant.name} (slug={tenant.slug}, id={tenant.id})")
    return tenant


@router.get("", response_model=list[TenantResponse])
async def list_tenants(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """List all tenants (SuperAdmin only)."""
    result = await db.execute(
        select(Tenant).order_by(Tenant.name)
    )
    return result.scalars().all()


@router.put("/{tenant_id}", response_model=TenantResponse)
async def update_tenant(
    tenant_id: int,
    body: TenantUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Update a tenant (SuperAdmin only)."""
    tenant = await _get_tenant(tenant_id, db)

    if body.name is not None:
        tenant.name = body.name
    if body.slug is not None:
        # Check uniqueness
        result = await db.execute(
            select(Tenant).where(Tenant.slug == body.slug, Tenant.id != tenant_id)
        )
        if result.scalar_one_or_none() is not None:
            raise ConflictError(f"Tenant with slug '{body.slug}' already exists")
        tenant.slug = body.slug
    if body.domain is not None:
        tenant.domain = body.domain
    if body.is_active is not None:
        tenant.is_active = body.is_active

    await db.commit()
    await db.refresh(tenant)
    return tenant


@router.delete("/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_tenant(
    tenant_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Deactivate a tenant (SuperAdmin only)."""
    tenant = await _get_tenant(tenant_id, db)
    tenant.is_active = False
    await db.commit()


@router.post("/{tenant_id}/set-admin/{user_id}", response_model=TenantUserResponse)
async def set_tenant_admin(
    tenant_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_superadmin),
):
    """Assign a user as tenant admin (SuperAdmin only). Creates membership if needed."""
    tenant = await _get_tenant(tenant_id, db)

    # Check target user exists
    result = await db.execute(select(User).where(User.id == user_id))
    target_user = result.scalar_one_or_none()
    if target_user is None:
        raise NotFoundError("User", user_id)

    # Check if membership exists
    result = await db.execute(
        select(TenantUser).where(
            TenantUser.tenant_id == tenant_id,
            TenantUser.user_id == user_id,
        )
    )
    tu = result.scalar_one_or_none()

    if tu is None:
        tu = TenantUser(
            tenant_id=tenant_id,
            user_id=user_id,
            role="admin",
            is_approved=True,
        )
        db.add(tu)
    else:
        tu.role = "admin"
        tu.is_approved = True

    # Also activate the user if not active
    if not target_user.is_active:
        target_user.is_active = True

    await db.commit()
    await db.refresh(tu)
    return await _build_tenant_user_response(tu, db)


# ── Tenant Admin: User Management ───────────────────────────────────────────

@router.get("/{tenant_id}/users", response_model=list[TenantUserResponse])
async def list_tenant_users(
    tenant_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """List all users in a tenant (Tenant Admin or SuperAdmin)."""
    await _get_tenant(tenant_id, db)
    await _require_tenant_admin(tenant_id, user, db)

    result = await db.execute(
        select(TenantUser)
        .where(TenantUser.tenant_id == tenant_id)
        .order_by(TenantUser.created_at)
    )
    tenant_users = result.scalars().all()
    return [await _build_tenant_user_response(tu, db) for tu in tenant_users]


@router.post("/{tenant_id}/users/{user_id}/approve", response_model=TenantUserResponse)
async def approve_user(
    tenant_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Approve a pending user (Tenant Admin or SuperAdmin)."""
    await _get_tenant(tenant_id, db)
    await _require_tenant_admin(tenant_id, user, db)

    result = await db.execute(
        select(TenantUser).where(
            TenantUser.tenant_id == tenant_id,
            TenantUser.user_id == user_id,
        )
    )
    tu = result.scalar_one_or_none()
    if tu is None:
        raise NotFoundError("TenantUser", f"tenant={tenant_id}, user={user_id}")

    tu.is_approved = True

    # Activate the user account
    result = await db.execute(select(User).where(User.id == user_id))
    target_user = result.scalar_one_or_none()
    if target_user and not target_user.is_active:
        target_user.is_active = True

    await db.commit()
    await db.refresh(tu)
    logger.info(f"User {user_id} approved for tenant {tenant_id}")
    return await _build_tenant_user_response(tu, db)


@router.post("/{tenant_id}/users/{user_id}/reject", status_code=status.HTTP_204_NO_CONTENT)
async def reject_user(
    tenant_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Reject and remove a pending user (Tenant Admin or SuperAdmin)."""
    await _get_tenant(tenant_id, db)
    await _require_tenant_admin(tenant_id, user, db)

    result = await db.execute(
        select(TenantUser).where(
            TenantUser.tenant_id == tenant_id,
            TenantUser.user_id == user_id,
        )
    )
    tu = result.scalar_one_or_none()
    if tu is None:
        raise NotFoundError("TenantUser", f"tenant={tenant_id}, user={user_id}")

    await db.delete(tu)
    await db.commit()


@router.delete("/{tenant_id}/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_tenant_user(
    tenant_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Remove a user from the tenant (Tenant Admin or SuperAdmin)."""
    await _get_tenant(tenant_id, db)
    await _require_tenant_admin(tenant_id, user, db)

    result = await db.execute(
        select(TenantUser).where(
            TenantUser.tenant_id == tenant_id,
            TenantUser.user_id == user_id,
        )
    )
    tu = result.scalar_one_or_none()
    if tu is None:
        raise NotFoundError("TenantUser", f"tenant={tenant_id}, user={user_id}")

    await db.delete(tu)
    await db.commit()


@router.put("/{tenant_id}/users/{user_id}/role", response_model=TenantUserResponse)
async def update_user_role(
    tenant_id: int,
    user_id: int,
    body: RoleUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Change a user's role in the tenant (Tenant Admin or SuperAdmin)."""
    await _get_tenant(tenant_id, db)
    await _require_tenant_admin(tenant_id, user, db)

    result = await db.execute(
        select(TenantUser).where(
            TenantUser.tenant_id == tenant_id,
            TenantUser.user_id == user_id,
        )
    )
    tu = result.scalar_one_or_none()
    if tu is None:
        raise NotFoundError("TenantUser", f"tenant={tenant_id}, user={user_id}")

    tu.role = body.role
    await db.commit()
    await db.refresh(tu)
    return await _build_tenant_user_response(tu, db)


# ── Any Authenticated: My Tenants ───────────────────────────────────────────

@router.get("/my", response_model=list[TenantResponse])
async def get_my_tenants(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """List tenants the current user belongs to."""
    result = await db.execute(
        select(Tenant)
        .join(TenantUser, TenantUser.tenant_id == Tenant.id)
        .where(
            TenantUser.user_id == user.id,
            TenantUser.is_approved.is_(True),
            Tenant.is_active.is_(True),
        )
        .order_by(Tenant.name)
    )
    return result.scalars().all()

"""
Tenant Management API
=====================
CRUD for tenants + user membership management + invite links.
SuperAdmin: create/update/delete tenants, assign tenant admins.
Tenant Admin: approve/reject/manage members, create invite links.
Any authenticated: list own tenants.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_active_user, require_superadmin
from app.core.exceptions import NotFoundError, ForbiddenError, ConflictError, BadRequestError
from app.models.tenant import Tenant, TenantUser
from app.models.user import User
from app.models.invite_token import InviteToken
from app.schemas.tenant import (
    TenantCreate,
    TenantUpdate,
    TenantResponse,
    TenantUserResponse,
    RoleUpdateRequest,
)
from app.schemas.invite import (
    InviteCreateRequest,
    InviteResponse,
    InviteValidationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenants", tags=["tenants"])


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _get_tenant(tenant_id: uuid.UUID, db: AsyncSession) -> Tenant:
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()
    if tenant is None:
        raise NotFoundError("Tenant", tenant_id)
    return tenant


async def _require_tenant_admin(tenant_id: uuid.UUID, user: User, db: AsyncSession) -> TenantUser:
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
    """List all tenants with member/pending counts (SuperAdmin only)."""
    # Subquery for member_count (approved) and pending_count (not approved)
    member_count_sq = (
        select(
            TenantUser.tenant_id,
            func.count(TenantUser.id).filter(TenantUser.is_approved.is_(True)).label("member_count"),
            func.count(TenantUser.id).filter(TenantUser.is_approved.is_(False)).label("pending_count"),
        )
        .group_by(TenantUser.tenant_id)
        .subquery()
    )

    result = await db.execute(
        select(
            Tenant,
            func.coalesce(member_count_sq.c.member_count, 0).label("member_count"),
            func.coalesce(member_count_sq.c.pending_count, 0).label("pending_count"),
        )
        .outerjoin(member_count_sq, Tenant.id == member_count_sq.c.tenant_id)
        .order_by(Tenant.name)
    )
    rows = result.all()

    return [
        TenantResponse(
            id=t.id,
            name=t.name,
            slug=t.slug,
            domain=t.domain,
            is_active=t.is_active,
            member_count=mc,
            pending_count=pc,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t, mc, pc in rows
    ]


@router.get("/invite/{token}", response_model=InviteValidationResponse)
async def validate_invite(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """Validate an invite token (public, no auth required). Returns tenant info if valid."""
    result = await db.execute(
        select(InviteToken).where(InviteToken.token == token)
    )
    invite = result.scalar_one_or_none()

    if invite is None or not invite.is_active:
        return InviteValidationResponse(valid=False)

    # Check expiration
    if datetime.utcnow() > invite.expires_at:
        return InviteValidationResponse(valid=False)

    # Check usage limit
    if invite.max_uses is not None and invite.use_count >= invite.max_uses:
        return InviteValidationResponse(valid=False)

    # Get tenant info
    result = await db.execute(
        select(Tenant).where(Tenant.id == invite.tenant_id, Tenant.is_active.is_(True))
    )
    tenant = result.scalar_one_or_none()
    if tenant is None:
        return InviteValidationResponse(valid=False)

    return InviteValidationResponse(
        valid=True,
        tenant_name=tenant.name,
        tenant_slug=tenant.slug,
        email=invite.email,
        expires_at=invite.expires_at.isoformat(),
    )


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


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get single tenant details (SuperAdmin or Tenant Admin)."""
    tenant = await _get_tenant(tenant_id, db)
    await _require_tenant_admin(tenant_id, user, db)

    # We need mc and pc for TenantResponse
    member_count = await db.scalar(
        select(func.count(TenantUser.id)).where(
            TenantUser.tenant_id == tenant_id, TenantUser.is_approved.is_(True)
        )
    )
    pending_count = await db.scalar(
        select(func.count(TenantUser.id)).where(
            TenantUser.tenant_id == tenant_id, TenantUser.is_approved.is_(False)
        )
    )

    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        domain=tenant.domain,
        is_active=tenant.is_active,
        member_count=member_count or 0,
        pending_count=pending_count or 0,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


@router.put("/{tenant_id}", response_model=TenantResponse)
async def update_tenant(
    tenant_id: uuid.UUID,
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
    tenant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Deactivate a tenant (SuperAdmin only)."""
    tenant = await _get_tenant(tenant_id, db)
    tenant.is_active = False
    await db.commit()


@router.post("/{tenant_id}/set-admin/{user_id}", response_model=TenantUserResponse)
async def set_tenant_admin(
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
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
    tenant_id: uuid.UUID,
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
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
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
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
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
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
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
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
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


# ── Invite Link Management ────────────────────────────────────────────────

def _build_invite_url(token: str, request: Request) -> str:
    """Build the frontend invite URL from the token."""
    # Use the request's origin to build the URL
    origin = request.headers.get("origin", "")
    if not origin:
        # Fallback: use the request base URL but point to frontend port
        origin = str(request.base_url).rstrip("/").replace(":8080", ":5174")
    return f"{origin}/register?invite={token}"


def _build_invite_response(invite: InviteToken, request: Request) -> InviteResponse:
    """Build InviteResponse from the model."""
    return InviteResponse(
        id=invite.id,
        token=invite.token,
        tenant_id=invite.tenant_id,
        email=invite.email,
        role=invite.role,
        max_uses=invite.max_uses,
        use_count=invite.use_count,
        expires_at=invite.expires_at,
        created_at=invite.created_at,
        is_active=invite.is_active,
        invite_url=_build_invite_url(invite.token, request),
    )


@router.post("/{tenant_id}/invites", response_model=InviteResponse, status_code=status.HTTP_201_CREATED)
async def create_invite(
    tenant_id: uuid.UUID,
    body: InviteCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Create an invite link for a tenant (Tenant Admin or SuperAdmin)."""
    await _get_tenant(tenant_id, db)
    await _require_tenant_admin(tenant_id, user, db)

    if body.role not in ("admin", "member"):
        raise BadRequestError("Role must be 'admin' or 'member'")

    token = uuid.uuid4().hex
    expires_at = datetime.utcnow() + timedelta(days=body.expires_in_days)

    invite = InviteToken(
        token=token,
        tenant_id=tenant_id,
        created_by=user.id,
        email=body.email.lower().strip() if body.email else None,
        role=body.role,
        max_uses=body.max_uses,
        expires_at=expires_at,
    )
    db.add(invite)
    await db.commit()
    await db.refresh(invite)

    logger.info(f"Invite created for tenant {tenant_id} by user {user.id}, token={token[:8]}...")
    return _build_invite_response(invite, request)


@router.get("/{tenant_id}/invites", response_model=list[InviteResponse])
async def list_invites(
    tenant_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """List active invite links for a tenant (Tenant Admin or SuperAdmin)."""
    await _get_tenant(tenant_id, db)
    await _require_tenant_admin(tenant_id, user, db)

    result = await db.execute(
        select(InviteToken)
        .where(
            InviteToken.tenant_id == tenant_id,
            InviteToken.is_active.is_(True),
        )
        .order_by(InviteToken.created_at.desc())
    )
    invites = result.scalars().all()
    return [_build_invite_response(inv, request) for inv in invites]


@router.delete("/{tenant_id}/invites/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_invite(
    tenant_id: uuid.UUID,
    invite_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Revoke an invite link (Tenant Admin or SuperAdmin)."""
    await _get_tenant(tenant_id, db)
    await _require_tenant_admin(tenant_id, user, db)

    result = await db.execute(
        select(InviteToken).where(
            InviteToken.id == invite_id,
            InviteToken.tenant_id == tenant_id,
        )
    )
    invite = result.scalar_one_or_none()
    if invite is None:
        raise NotFoundError("Invite", invite_id)

    invite.is_active = False
    await db.commit()
    logger.info(f"Invite {invite_id} revoked for tenant {tenant_id} by user {user.id}")




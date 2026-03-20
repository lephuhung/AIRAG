"""
Admin API
=========
SuperAdmin-only endpoints for global user management and system stats.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, require_superadmin
from app.core.exceptions import NotFoundError, BadRequestError
from app.models.user import User
from app.models.tenant import Tenant, TenantUser
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.document_type import DocumentType

from app.schemas.admin import (
    AdminUserUpdate,
    AdminUserDetail,
    AdminUserListResponse,
    AdminStatsResponse,
    AdminPasswordResetRequest,
)
from app.core.security import hash_password
from app.schemas.tenant import TenantUserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_superadmin)])


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _build_admin_user_detail(user: User, db: AsyncSession) -> AdminUserDetail:
    """Build AdminUserDetail with tenant memberships."""
    result = await db.execute(
        select(TenantUser).where(TenantUser.user_id == user.id)
    )
    tenant_users = result.scalars().all()

    memberships = []
    for tu in tenant_users:
        u_result = await db.execute(select(User).where(User.id == tu.user_id))
        u = u_result.scalar_one_or_none()
        t_result = await db.execute(select(Tenant).where(Tenant.id == tu.tenant_id))
        t = t_result.scalar_one_or_none()
        memberships.append(TenantUserResponse(
            id=tu.id,
            tenant_id=tu.tenant_id,
            user_id=tu.user_id,
            role=tu.role,
            is_approved=tu.is_approved,
            created_at=tu.created_at,
            email=u.email if u else None,
            full_name=u.full_name if u else None,
            tenant_name=t.name if t else None,
        ))

    return AdminUserDetail(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_superadmin=user.is_superadmin,
        avatar_url=user.avatar_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
        tenant_memberships=memberships,
    )


# ── Stats ────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_superadmin),
):
    """Aggregate system stats (SuperAdmin only)."""
    total_users = await db.scalar(select(func.count(User.id)))
    active_users = await db.scalar(
        select(func.count(User.id)).where(User.is_active.is_(True))
    )
    pending_users = await db.scalar(
        select(func.count(User.id)).where(User.is_active.is_(False))
    )
    total_tenants = await db.scalar(select(func.count(Tenant.id)))
    total_documents = await db.scalar(select(func.count(Document.id)))
    total_knowledge_bases = await db.scalar(select(func.count(KnowledgeBase.id)))
    
    # Document type breakdown (only types with >0 documents)
    doctype_stats = await db.execute(
        select(DocumentType.name, func.count(Document.id))
        .join(Document, Document.document_type_id == DocumentType.id)
        .group_by(DocumentType.name)
    )
    document_type_breakdown = [
        {"name": row[0], "count": row[1]} for row in doctype_stats.all()
    ]

    return AdminStatsResponse(
        total_users=total_users or 0,
        active_users=active_users or 0,
        pending_users=pending_users or 0,
        total_tenants=total_tenants or 0,
        total_documents=total_documents or 0,
        total_knowledge_bases=total_knowledge_bases or 0,
        document_type_breakdown=document_type_breakdown,
    )


# ── User CRUD ────────────────────────────────────────────────────────────────

@router.get("/users", response_model=AdminUserListResponse)
async def list_users(
    search: str | None = Query(None, description="Search by name or email"),
    is_active: bool | None = Query(None, description="Filter by active status"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_superadmin),
):
    """List all users with search and pagination (SuperAdmin only)."""
    query = select(User)
    count_query = select(func.count(User.id))

    # Apply filters
    if search:
        search_filter = or_(
            User.full_name.ilike(f"%{search}%"),
            User.email.ilike(f"%{search}%"),
        )
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)

    if is_active is not None:
        query = query.where(User.is_active == is_active)
        count_query = count_query.where(User.is_active == is_active)

    # Count total
    total = await db.scalar(count_query) or 0

    # Paginate
    offset = (page - 1) * per_page
    result = await db.execute(
        query.order_by(User.created_at.desc()).offset(offset).limit(per_page)
    )
    users = result.scalars().all()

    user_details = [await _build_admin_user_detail(u, db) for u in users]

    return AdminUserListResponse(
        users=user_details,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/users/{user_id}", response_model=AdminUserDetail)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_superadmin),
):
    """Get user detail + tenant memberships (SuperAdmin only)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise NotFoundError("User", user_id)

    return await _build_admin_user_detail(user, db)


@router.put("/users/{user_id}", response_model=AdminUserDetail)
async def update_user(
    user_id: int,
    body: AdminUserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_superadmin),
):
    """Update user fields: is_active, is_superadmin, full_name (SuperAdmin only)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise NotFoundError("User", user_id)

    if body.is_active is not None:
        user.is_active = body.is_active
    if body.is_superadmin is not None:
        # Prevent removing own superadmin
        if user.id == current_user.id and not body.is_superadmin:
            raise BadRequestError("Cannot remove your own superadmin status")
        user.is_superadmin = body.is_superadmin
    if body.full_name is not None:
        user.full_name = body.full_name

    await db.commit()
    await db.refresh(user)
    logger.info(f"Admin updated user {user_id}: {body.model_dump(exclude_none=True)}")
    return await _build_admin_user_detail(user, db)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_superadmin),
):
    """Delete a user (SuperAdmin only). Cannot delete self."""
    if user_id == current_user.id:
        raise BadRequestError("Cannot delete your own account")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise NotFoundError("User", user_id)

    await db.delete(user)
    await db.commit()
    logger.info(f"Admin deleted user {user_id} ({user.email})")


@router.post("/users/{user_id}/reset-password", response_model=AdminUserDetail)
async def reset_user_password(
    user_id: int,
    body: AdminPasswordResetRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_superadmin),
):
    """Reset a user's password (SuperAdmin only).

    Sets a new password without requiring the current password.
    Used for account recovery and locked-out users.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise NotFoundError("User", user_id)

    user.password_hash = hash_password(body.new_password)
    await db.commit()
    await db.refresh(user)
    logger.info(f"Admin {current_user.id} reset password for user {user_id} ({user.email})")
    return await _build_admin_user_detail(user, db)

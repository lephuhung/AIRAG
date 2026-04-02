"""
Knowledge Base (Workspace) CRUD API endpoints — with auth.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from app.core.deps import get_db, get_current_active_user, verify_workspace_access
from app.core.exceptions import NotFoundError, ForbiddenError
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentStatus
from app.models.user import User
from app.models.tenant import TenantUser
from app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceUpdate,
    WorkspaceResponse,
    WorkspaceSummary,
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


async def _enrich_response(db: AsyncSession, kb: KnowledgeBase) -> WorkspaceResponse:
    """Build WorkspaceResponse with computed counts."""
    total = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == kb.id)
    )
    indexed = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == kb.id,
            Document.status.in_([DocumentStatus.INDEXED, DocumentStatus.BUILDING_KG]),
        )
    )
    return WorkspaceResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        system_prompt=kb.system_prompt,
        document_count=total.scalar() or 0,
        indexed_count=indexed.scalar() or 0,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
        visibility=kb.visibility or "personal",
        owner_id=kb.owner_id,
        tenant_id=kb.tenant_id,
    )


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """List workspaces visible to the current user."""
    if user.is_superadmin:
        # Superadmin sees everything
        result = await db.execute(
            select(KnowledgeBase).order_by(KnowledgeBase.updated_at.desc())
        )
    else:
        # Get user's tenant IDs
        tenant_result = await db.execute(
            select(TenantUser.tenant_id).where(
                TenantUser.user_id == user.id,
                TenantUser.is_approved.is_(True),
            )
        )
        user_tenant_ids = [row[0] for row in tenant_result.all()]

        # Build visibility filter:
        # 1. Public workspaces
        # 2. Tenant workspaces for user's tenants
        # 3. User's own personal workspaces
        # 4. Legacy workspaces (no owner) — treat as public
        conditions = [
            KnowledgeBase.visibility == "public",
            KnowledgeBase.owner_id.is_(None),  # Legacy workspaces
            KnowledgeBase.owner_id == user.id,  # Own workspaces
        ]
        if user_tenant_ids:
            conditions.append(
                (KnowledgeBase.visibility == "tenant") &
                KnowledgeBase.tenant_id.in_(user_tenant_ids)
            )

        result = await db.execute(
            select(KnowledgeBase)
            .where(or_(*conditions))
            .order_by(KnowledgeBase.updated_at.desc())
        )

    kbs = result.scalars().all()
    return [await _enrich_response(db, kb) for kb in kbs]


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    body: WorkspaceCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Create a new knowledge base."""
    kb = KnowledgeBase(
        name=body.name,
        description=body.description,
        owner_id=user.id,
        visibility=body.visibility or "personal",
        tenant_id=body.tenant_id,
    )

    # Validate: tenant visibility requires a tenant_id
    if (body.visibility or "personal") == "tenant" and body.tenant_id is None:
        raise HTTPException(status_code=400, detail="tenant_id is required when visibility is 'tenant'")

    # Validate tenant_id if provided
    if body.tenant_id is not None:
        from app.models.tenant import Tenant
        t_result = await db.execute(select(Tenant).where(Tenant.id == body.tenant_id))
        if t_result.scalar_one_or_none() is None:
            raise HTTPException(status_code=400, detail="Tenant not found")
        # Check user is member of this tenant (unless superadmin)
        if not user.is_superadmin:
            tu_result = await db.execute(
                select(TenantUser).where(
                    TenantUser.tenant_id == body.tenant_id,
                    TenantUser.user_id == user.id,
                    TenantUser.is_approved.is_(True),
                )
            )
            if tu_result.scalar_one_or_none() is None:
                raise ForbiddenError("You're not a member of this tenant")

    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return await _enrich_response(db, kb)


@router.get("/summary", response_model=list[WorkspaceSummary])
async def list_workspace_summaries(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Compact list for dropdown selectors."""
    # Reuse same filter logic as list_workspaces
    if user.is_superadmin:
        result = await db.execute(
            select(KnowledgeBase).order_by(KnowledgeBase.name)
        )
    else:
        tenant_result = await db.execute(
            select(TenantUser.tenant_id).where(
                TenantUser.user_id == user.id,
                TenantUser.is_approved.is_(True),
            )
        )
        user_tenant_ids = [row[0] for row in tenant_result.all()]

        conditions = [
            KnowledgeBase.visibility == "public",
            KnowledgeBase.owner_id.is_(None),
            KnowledgeBase.owner_id == user.id,
        ]
        if user_tenant_ids:
            conditions.append(
                (KnowledgeBase.visibility == "tenant") &
                KnowledgeBase.tenant_id.in_(user_tenant_ids)
            )

        result = await db.execute(
            select(KnowledgeBase)
            .where(or_(*conditions))
            .order_by(KnowledgeBase.name)
        )

    kbs = result.scalars().all()
    summaries = []
    for kb in kbs:
        cnt = await db.execute(
            select(func.count(Document.id)).where(Document.workspace_id == kb.id)
        )
        summaries.append(WorkspaceSummary(
            id=kb.id, name=kb.name, document_count=cnt.scalar() or 0
        ))
    return summaries


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get a knowledge base by ID."""
    kb = await verify_workspace_access(workspace_id, user, db)
    return await _enrich_response(db, kb)


@router.put("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: uuid.UUID,
    body: WorkspaceUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Update a knowledge base name/description."""
    kb = await verify_workspace_access(workspace_id, user, db)

    # Only owner, tenant admin, or superadmin can edit
    if not user.is_superadmin and kb.owner_id is not None and kb.owner_id != user.id:
        # Check if user is tenant admin
        if kb.tenant_id:
            tu_result = await db.execute(
                select(TenantUser).where(
                    TenantUser.tenant_id == kb.tenant_id,
                    TenantUser.user_id == user.id,
                    TenantUser.role == "admin",
                    TenantUser.is_approved.is_(True),
                )
            )
            if tu_result.scalar_one_or_none() is None:
                raise ForbiddenError("Only the owner or tenant admin can edit this workspace")
        else:
            raise ForbiddenError("Only the owner can edit this workspace")

    if body.name is not None:
        kb.name = body.name
    if body.description is not None:
        kb.description = body.description
    if body.system_prompt is not None:
        # Empty string → reset to default (None)
        kb.system_prompt = body.system_prompt or None

    # Superadmin-only: reassign tenant_id / visibility
    if body.tenant_id is not None or body.visibility is not None:
        if not user.is_superadmin:
            raise ForbiddenError("Only superadmin can change workspace tenant or visibility")
        if body.tenant_id is not None:
            from app.models.tenant import Tenant
            t_result = await db.execute(select(Tenant).where(Tenant.id == body.tenant_id))
            if t_result.scalar_one_or_none() is None:
                raise HTTPException(status_code=400, detail="Tenant not found")
            kb.tenant_id = body.tenant_id
        if body.visibility is not None:
            kb.visibility = body.visibility

    await db.commit()
    await db.refresh(kb)
    return await _enrich_response(db, kb)


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Delete a knowledge base and all its documents."""
    kb = await verify_workspace_access(workspace_id, user, db)

    # Only owner, tenant admin, or superadmin can delete
    if not user.is_superadmin and kb.owner_id is not None and kb.owner_id != user.id:
        if kb.tenant_id:
            tu_result = await db.execute(
                select(TenantUser).where(
                    TenantUser.tenant_id == kb.tenant_id,
                    TenantUser.user_id == user.id,
                    TenantUser.role == "admin",
                    TenantUser.is_approved.is_(True),
                )
            )
            if tu_result.scalar_one_or_none() is None:
                raise ForbiddenError("Only the owner or tenant admin can delete this workspace")
        else:
            raise ForbiddenError("Only the owner can delete this workspace")

    # Clean up vector store and KG data
    try:
        from app.services.vector_store import get_vector_store
        vs = get_vector_store(workspace_id)
        vs.delete_collection()
    except Exception:
        pass

    try:
        from app.services.knowledge_graph_service import KnowledgeGraphService
        kg = KnowledgeGraphService(workspace_id)
        await kg.delete_project_data()
    except Exception:
        pass

    # Clean up image files
    import shutil
    from app.core.config import settings
    images_dir = settings.BASE_DIR / "data" / "docling" / f"kb_{workspace_id}"
    if images_dir.exists():
        shutil.rmtree(images_dir, ignore_errors=True)

    await db.delete(kb)
    await db.commit()

"""Audit-log (activity log) API — read-only, superadmin only."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, require_superadmin
from app.models.user import User
from app.schemas.audit_log import AuditLogListResponse, AuditLogResponse
from app.services.audit_service import AuditService

router = APIRouter(prefix="/audit-logs", tags=["audit-logs"])


@router.get("", response_model=AuditLogListResponse)
async def list_audit_logs(
    resource_type: list[str] | None = Query(None),
    action: str | None = Query(None),
    actor_id: uuid.UUID | None = Query(None),
    search: str | None = Query(None),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_superadmin),
):
    """List recorded user/management actions, newest first.

    `resource_type` may be repeated to cover a whole domain
    (e.g. the tenant tab passes tenant + tenant_member + tenant_invite).
    """
    items, total = await AuditService.query(
        db,
        resource_types=resource_type,
        action=action,
        actor_id=actor_id,
        search=search,
        date_from=date_from,
        date_to=date_to,
        page=page,
        per_page=per_page,
    )
    return AuditLogListResponse(
        items=[AuditLogResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        per_page=per_page,
    )

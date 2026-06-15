"""
AuditService — records and queries the application-level activity log.

Writes are best-effort and isolated in their own DB session, so a failure to
audit never breaks (or rolls back) the user's actual request.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_maker
from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


class AuditService:
    @staticmethod
    async def record(
        *,
        action: str,
        resource_type: str,
        actor_id: uuid.UUID | None = None,
        actor_email: str | None = None,
        actor_name: str | None = None,
        resource_id: str | None = None,
        resource_label: str | None = None,
        summary: str | None = None,
        extra: dict | None = None,
        method: str | None = None,
        path: str | None = None,
        status_code: int | None = None,
        ip_address: str | None = None,
        source: str = "auto",
    ) -> None:
        """Persist one audit entry. Never raises — failures are logged only."""
        try:
            async with async_session_maker() as db:
                db.add(
                    AuditLog(
                        actor_id=actor_id,
                        actor_email=actor_email,
                        actor_name=actor_name,
                        action=action,
                        resource_type=resource_type,
                        resource_id=str(resource_id) if resource_id is not None else None,
                        resource_label=resource_label,
                        summary=summary,
                        extra=extra,
                        method=method,
                        path=path,
                        status_code=status_code,
                        ip_address=ip_address,
                        source=source,
                    )
                )
                await db.commit()
        except Exception as e:  # pragma: no cover - best effort
            logger.warning(f"[audit] failed to record {action} {resource_type}: {e}")

    @staticmethod
    async def record_for_actor(actor, **kwargs) -> None:
        """Convenience wrapper that snapshots actor identity from a User."""
        await AuditService.record(
            actor_id=getattr(actor, "id", None),
            actor_email=getattr(actor, "email", None),
            actor_name=getattr(actor, "full_name", None),
            **kwargs,
        )

    @staticmethod
    async def query(
        db: AsyncSession,
        *,
        resource_types: list[str] | None = None,
        action: str | None = None,
        actor_id: uuid.UUID | None = None,
        search: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> tuple[list[AuditLog], int]:
        conds = []
        if resource_types:
            conds.append(AuditLog.resource_type.in_(resource_types))
        if action:
            conds.append(AuditLog.action == action)
        if actor_id:
            conds.append(AuditLog.actor_id == actor_id)
        if date_from:
            conds.append(AuditLog.created_at >= date_from)
        if date_to:
            conds.append(AuditLog.created_at <= date_to)
        if search:
            like = f"%{search.lower()}%"
            conds.append(
                func.lower(
                    func.coalesce(AuditLog.summary, "")
                    + " "
                    + func.coalesce(AuditLog.resource_label, "")
                    + " "
                    + func.coalesce(AuditLog.actor_email, "")
                ).like(like)
            )

        where = and_(*conds) if conds else None

        count_stmt = select(func.count()).select_from(AuditLog)
        if where is not None:
            count_stmt = count_stmt.where(where)
        total = (await db.execute(count_stmt)).scalar_one()

        stmt = select(AuditLog)
        if where is not None:
            stmt = stmt.where(where)
        stmt = (
            stmt.order_by(AuditLog.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
        items = list((await db.execute(stmt)).scalars().all())
        return items, total

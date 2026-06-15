"""
AuditLog model — records user management actions (create/update/delete of
abbreviations, users, tenants, workspaces, document types, …) for the
Activity Log page. This is an application-level audit trail, separate from the
file/Grafana infrastructure logs.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, DateTime, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Who performed the action (snapshot — survives user deletion)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    actor_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    actor_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # What happened: action verb + the kind of entity it touched
    action: Mapped[str] = mapped_column(String(32), index=True)  # create/update/delete/…
    resource_type: Mapped[str] = mapped_column(String(48), index=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_label: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Human-readable one-line description (localized at write time)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Free-form structured context (changed fields, etc.)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Request context
    method: Mapped[str | None] = mapped_column(String(10), nullable=True)
    path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # "auto" (middleware) | "explicit" (in-endpoint call)
    source: Mapped[str] = mapped_column(String(16), default="auto")

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )

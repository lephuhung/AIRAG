"""Pydantic schemas for the audit-log (activity log) API."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    actor_id: uuid.UUID | None = None
    actor_email: str | None = None
    actor_name: str | None = None
    action: str
    resource_type: str
    resource_id: str | None = None
    resource_label: str | None = None
    summary: str | None = None
    extra: dict | None = None
    method: str | None = None
    path: str | None = None
    status_code: int | None = None
    ip_address: str | None = None
    source: str
    created_at: datetime


class AuditLogListResponse(BaseModel):
    items: list[AuditLogResponse]
    total: int
    page: int
    per_page: int

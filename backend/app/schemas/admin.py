"""
Admin schemas — used by superadmin user/tenant management endpoints.
"""
from datetime import datetime
from pydantic import BaseModel, Field

from app.schemas.tenant import TenantUserResponse


class AdminUserUpdate(BaseModel):
    is_active: bool | None = None
    is_superadmin: bool | None = None
    full_name: str | None = Field(None, min_length=1, max_length=255)


class AdminPasswordResetRequest(BaseModel):
    new_password: str = Field(..., min_length=6, max_length=128)


class AdminUserDetail(BaseModel):
    id: int
    email: str
    full_name: str
    is_active: bool
    is_superadmin: bool
    avatar_url: str | None = None
    created_at: datetime
    updated_at: datetime
    tenant_memberships: list[TenantUserResponse] = []

    model_config = {"from_attributes": True}


class AdminUserListResponse(BaseModel):
    users: list[AdminUserDetail]
    total: int
    page: int
    per_page: int


class AdminStatsResponse(BaseModel):
    total_users: int
    active_users: int
    pending_users: int
    total_tenants: int
    total_documents: int
    total_knowledge_bases: int

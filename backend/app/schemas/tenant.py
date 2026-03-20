"""
Tenant schemas.
"""
from datetime import datetime
from pydantic import BaseModel, Field


class TenantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100)
    domain: str | None = None


class TenantUpdate(BaseModel):
    name: str | None = None
    slug: str | None = None
    domain: str | None = None
    is_active: bool | None = None


class TenantResponse(BaseModel):
    id: int
    name: str
    slug: str
    domain: str | None
    is_active: bool
    member_count: int = 0
    pending_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TenantUserResponse(BaseModel):
    id: int
    tenant_id: int
    user_id: int
    role: str
    is_approved: bool
    created_at: datetime
    # Joined user info
    email: str | None = None
    full_name: str | None = None
    # Joined tenant info
    tenant_name: str | None = None

    model_config = {"from_attributes": True}


class RoleUpdateRequest(BaseModel):
    role: str = Field(..., pattern="^(admin|member)$")

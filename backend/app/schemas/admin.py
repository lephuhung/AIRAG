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


class DocumentTypeBreakdown(BaseModel):
    name: str
    count: int


class DateCount(BaseModel):
    date: str
    count: int

class DocumentStatusBreakdown(BaseModel):
    status: str
    count: int

class TopWorkspace(BaseModel):
    id: int
    name: str
    total_size: int
    doc_count: int

class FailedDocument(BaseModel):
    id: int
    filename: str
    workspace_name: str
    error_message: str | None

class PendingApproval(BaseModel):
    user_id: int
    email: str
    tenant_name: str
    role: str

class AdminStatsResponse(BaseModel):
    total_users: int
    active_users: int
    pending_users: int
    total_tenants: int
    total_documents: int
    total_knowledge_bases: int
    document_type_breakdown: list[DocumentTypeBreakdown] = []

    # New advanced metrics
    users_growth: list[DateCount] = []
    chat_growth: list[DateCount] = []
    document_status_breakdown: list[DocumentStatusBreakdown] = []
    top_workspaces: list[TopWorkspace] = []
    recent_failed_docs: list[FailedDocument] = []
    pending_approvals: list[PendingApproval] = []

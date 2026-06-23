"""
Invite token request/response schemas.
"""
import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class InviteCreateRequest(BaseModel):
    email: str | None = None          # lock to specific email
    role: str = "member"              # "admin" | "member"
    max_uses: int | None = None       # None = unlimited
    expires_in_days: int = Field(default=7, ge=1, le=90)


class InviteResponse(BaseModel):
    id: uuid.UUID
    token: str
    tenant_id: uuid.UUID
    email: str | None
    role: str
    max_uses: int | None
    use_count: int
    expires_at: datetime
    created_at: datetime
    is_active: bool
    invite_url: str

    model_config = {"from_attributes": True}


class InviteValidationResponse(BaseModel):
    valid: bool
    tenant_name: str | None = None
    tenant_slug: str | None = None
    email: str | None = None
    expires_at: str | None = None


class InviteAcceptResponse(BaseModel):
    """Result of an already-authenticated user redeeming an invite link."""
    tenant_id: uuid.UUID
    tenant_name: str
    tenant_slug: str
    role: str
    already_member: bool = False  # True if the user was already an approved member

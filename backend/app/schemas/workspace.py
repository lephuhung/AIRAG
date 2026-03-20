"""
Knowledge Base (Workspace) schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from datetime import datetime


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    visibility: str | None = Field(default="personal", pattern="^(public|tenant|personal)$")
    tenant_id: int | None = None


class WorkspaceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    system_prompt: str | None = None
    # Superadmin-only: reassign workspace to a different tenant / change visibility
    tenant_id: int | None = Field(default=None)
    visibility: str | None = Field(default=None, pattern="^(public|tenant|personal)$")


class WorkspaceResponse(BaseModel):
    id: int
    name: str
    description: str | None
    system_prompt: str | None = None
    document_count: int = 0
    indexed_count: int = 0
    created_at: datetime
    updated_at: datetime
    visibility: str = "personal"
    owner_id: int | None = None
    tenant_id: int | None = None

    model_config = {"from_attributes": True}


class WorkspaceSummary(BaseModel):
    """Compact summary for dropdown selectors."""
    id: int
    name: str
    document_count: int = 0

    model_config = {"from_attributes": True}

"""
User schemas.
"""
import uuid
from datetime import datetime
from pydantic import BaseModel


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    is_active: bool
    is_superadmin: bool
    avatar_url: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserBrief(BaseModel):
    """Compact user info for embedding in other responses."""
    id: uuid.UUID
    email: str
    full_name: str

    model_config = {"from_attributes": True}

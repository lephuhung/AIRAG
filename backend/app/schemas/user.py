"""
User schemas.
"""
from datetime import datetime
from pydantic import BaseModel


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    is_active: bool
    is_superadmin: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserBrief(BaseModel):
    """Compact user info for embedding in other responses."""
    id: int
    email: str
    full_name: str

    model_config = {"from_attributes": True}

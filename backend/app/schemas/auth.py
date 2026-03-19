"""
Auth request/response schemas.
"""
from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(..., min_length=1, max_length=255)
    tenant_slug: str | None = None
    invite_token: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UpdateProfileRequest(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=255)
    # Password change: requires current_password for verification
    current_password: str | None = Field(default=None, min_length=1, max_length=128)
    new_password: str | None = Field(default=None, min_length=6, max_length=128)


# Forward ref
from app.schemas.user import UserResponse  # noqa: E402

TokenResponse.model_rebuild()

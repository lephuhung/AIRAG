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
    # Required only when the account has TOTP two-factor enabled.
    totp_code: str | None = Field(default=None, min_length=6, max_length=6)


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
    # Free-form preferences, shallow-merged into users.settings (e.g. {"tts": {...}})
    settings: dict | None = Field(default=None)


# ── Two-factor auth (TOTP / Google Authenticator) ──────────────────────────
class TwoFASetupResponse(BaseModel):
    """Returned when the user starts enrollment — render the QR, keep the secret."""
    secret: str            # base32, for manual entry
    otpauth_uri: str       # otpauth:// URI encoded in the QR
    qr_data_uri: str       # base64 PNG data URI for an <img>


class TwoFAEnableRequest(BaseModel):
    """Confirm enrollment by proving the user can produce a valid code."""
    code: str = Field(..., min_length=6, max_length=6)


class TwoFADisableRequest(BaseModel):
    """Turn 2FA off — requires a current code OR the account password."""
    code: str | None = Field(default=None, min_length=6, max_length=6)
    password: str | None = Field(default=None, min_length=1, max_length=128)


class TwoFAStatusResponse(BaseModel):
    enabled: bool


# Forward ref
from app.schemas.user import UserResponse  # noqa: E402

TokenResponse.model_rebuild()

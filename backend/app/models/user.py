"""
User model — stores user accounts for authentication.
"""

import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_superadmin: Mapped[bool] = mapped_column(Boolean, default=False)
    avatar_url: Mapped[str | None] = mapped_column(
        String(1024), nullable=True, default=None
    )
    # ── Two-factor auth (TOTP / Google Authenticator) ──────────────────────
    # base32 secret; populated during setup, kept even while pending verification.
    totp_secret: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    # Only true after the user verifies a code → login then requires a TOTP code.
    totp_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    # Free-form per-user preferences (e.g. {"tts": {"voice": ..., "speed": 1.0}})
    settings: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    @property
    def two_factor_enabled(self) -> bool:
        """Public-facing alias for `totp_enabled` (used by UserResponse)."""
        return bool(self.totp_enabled)

    chat_files: Mapped[list["ChatFile"]] = relationship(
        "ChatFile",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="ChatFile.created_at",
    )

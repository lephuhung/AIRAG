"""
InviteToken model — invite links for tenant registration.
"""
from datetime import datetime

from sqlalchemy import String, DateTime, Boolean, Integer, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class InviteToken(Base):
    __tablename__ = "invite_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    tenant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    created_by: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="member")  # 'admin' | 'member'
    max_uses: Mapped[int | None] = mapped_column(Integer, nullable=True)  # None = unlimited
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

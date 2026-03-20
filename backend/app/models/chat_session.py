"""
ChatSession model — groups chat messages into distinct conversation sessions across workspaces.
"""
from __future__ import annotations

from datetime import datetime

import uuid
from sqlalchemy import ForeignKey, String, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True, default=lambda: uuid.uuid4().hex)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Chat")
    
    # User who owns this chat session
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", 
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at"
    )

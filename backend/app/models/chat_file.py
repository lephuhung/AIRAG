"""
ChatFile model — stores docx/audio files attached to chat sessions.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import String, ForeignKey, DateTime, Integer, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ChatFile(Base):
    __tablename__ = "chat_files"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE")
    )
    file_name: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(50))
    file_size: Mapped[int] = mapped_column(Integer)
    minio_original_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    minio_markdown_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    markdown_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    report: Mapped[str | None] = mapped_column(Text, nullable=True)
    issues_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    format_metadata: Mapped["FormatMetadata | None"] = relationship(
        "FormatMetadata",
        back_populates="chat_file",
        cascade="all, delete-orphan",
        uselist=False,
    )

    session: Mapped["ChatSession"] = relationship(back_populates="chat_files")
    user: Mapped["User"] = relationship(back_populates="chat_files")
    workspace: Mapped["KnowledgeBase"] = relationship(back_populates="chat_files")

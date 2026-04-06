"""
FormatMetadata model — stores extracted formatting information from docx files.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import String, ForeignKey, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class FormatMetadata(Base):
    __tablename__ = "format_metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chat_file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_files.id", ondelete="CASCADE")
    )
    format_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chat_file: Mapped["ChatFile"] = relationship(back_populates="format_metadata")

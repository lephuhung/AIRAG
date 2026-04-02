"""
ExchangeSummary model — per Q&A pair summarization for conversation context.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Integer, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ExchangeSummary(Base):
    __tablename__ = "chat_exchange_summaries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
    )

    # Thứ tự exchange trong session (1, 2, 3...)
    exchange_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Message IDs của cặp Q&A này
    user_message_id: Mapped[str] = mapped_column(String(50), nullable=False)
    assistant_message_id: Mapped[str] = mapped_column(String(50), nullable=True)

    # Nội dung tóm tắt
    topic_label: Mapped[str] = mapped_column(String(255), nullable=False)
    key_entities: Mapped[list | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)

    # Cited documents/sources from this exchange
    cited_sources: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationship to ChatSession
    session: Mapped["ChatSession"] = relationship(
        "ChatSession",
        back_populates="exchange_summaries"
    )
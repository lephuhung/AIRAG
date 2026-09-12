"""ConversationSnapshot ORM — Phase 1B (Task 7 amendment).

Maps the ``conversation_snapshots`` table. One row per thread: the
rolling-summary persistence projection derived from the authoritative raw
chat messages. ``contract_version`` + the JSONB ``context`` carry the frozen
:class:`~app.services.agents.v2.contracts.conversation.ConversationSnapshot`
payload; ``summary_version`` is the optimistic-lock column and
``built_through_message_id`` the monotonic pointer into raw chat history.

``UNIQUE(thread_id)`` is both the one-current-snapshot-per-thread rule and
the insert arbiter for the first snapshot.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ConversationSnapshot(Base):
    __tablename__ = "conversation_snapshots"

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    contract_version: Mapped[str] = mapped_column(Text, nullable=False)
    summary_version: Mapped[int] = mapped_column(Integer, nullable=False)
    built_through_message_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    context: Mapped[dict] = mapped_column(JSONB, nullable=False)
    taken_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        # One current snapshot per thread — the CAS updates this row in
        # place, so ``(thread_id)`` is the natural arbiter.
        UniqueConstraint("thread_id"),
    )

"""SemanticSnapshot ORM — Phase 1B.

Maps the ``semantic_snapshots`` table. One row per checkpointed
semantic-context projection (history-derived vector state used to
rehydrate a new chat turn). The ``(thread_id, taken_at)`` UNIQUE
key is the natural-history arbiter for snapshot ordering.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SemanticSnapshot(Base):
    __tablename__ = "semantic_snapshots"

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    taken_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        # Mirrors the SQL UNIQUE (thread_id, taken_at) installed by
        # the migration as an *anonymous* UNIQUE constraint (no name).
        # We deliberately omit a constraint ``name`` so that a future
        # ``create_all`` on a fresh DB issues an auto-named UNIQUE
        # that matches the migration's anonymous UNIQUE.
        UniqueConstraint("thread_id", "taken_at"),
    )

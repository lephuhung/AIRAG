"""SemanticSnapshot ORM — Phase 1B (Task 7 amendment).

Maps the ``semantic_snapshots`` table. One row per thread: the finalized
query-meaning projection used to rehydrate a new chat turn. The frozen
:class:`~app.services.agents.v2.contracts.semantic.SemanticSnapshot` contract
has no ``thread_id`` of its own, so ``thread_id`` is the persistence key the
caller supplies; ``contract_version`` + the JSONB ``semantic`` carry the
contract payload.

``UNIQUE(thread_id)`` keeps exactly one current semantic projection per
thread.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SemanticSnapshot(Base):
    __tablename__ = "semantic_snapshots"

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    contract_version: Mapped[str] = mapped_column(Text, nullable=False)
    semantic: Mapped[dict] = mapped_column(JSONB, nullable=False)
    taken_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (UniqueConstraint("thread_id"),)

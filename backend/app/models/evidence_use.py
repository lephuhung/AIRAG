"""EvidenceUse ORM — Phase 1B.

Maps the ``evidence_uses`` table. One row per (TaskSpec,
EvidenceRecord) pair; the row is created when an evidence record is
bound into a task's response.

The brief calls out the **null-safe uniqueness** on ``task_id``:
the migration installs ``ix_evidence_uses_task`` on ``task_id``
(the per-task lookup index). Phase 1D's binding code reads these
rows via the index to assemble the per-task evidence ledger.

The FK to ``evidence_records.evidence_id`` is the only structural
relationship; the binding-audit trail is reconstructed by joining
``evidence_uses`` → ``evidence_records`` → ``binding_audit``.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class EvidenceUse(Base):
    __tablename__ = "evidence_uses"

    use_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidence_records.evidence_id"),
        nullable=False,
    )
    task_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        # Mirrors the ``ix_evidence_uses_task`` index installed by the
        # migration. This is the per-task lookup path the brief calls
        # out as the "null-safe uniqueness" mapping.
        Index("ix_evidence_uses_task", "task_id"),
    )
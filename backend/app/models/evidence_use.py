"""EvidenceUse ORM — Phase 1B (Task 8 amendment).

Maps the ``evidence_uses`` table. One row per how the current run/task uses
immutable evidence: the run key (``run_id``, envelope-level per spec §15.2),
the bound ``task_id``, the ``purpose`` (``discovery`` / ``coverage`` /
``supporting``), the ``evidence_id``, and the run-local ``target_id``
(``NULL`` for discovery uses).

Idempotency is enforced by the null-safe unique index
``uq_evidence_use_key`` over ``(run_id, task_id, evidence_id, purpose,
target_id)`` with ``NULLS NOT DISTINCT`` — a retried append collides even when
``target_id`` is ``NULL``, so the retry returns the existing row's UUID instead
of inserting an uncontrolled duplicate. The index is the ``ON CONFLICT``
arbiter the repository uses.

The FK to ``evidence_records.evidence_id`` is the only structural
relationship; the binding-audit trail is reconstructed by joining
``evidence_uses`` → ``evidence_records`` → ``binding_audit``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

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
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidence_records.evidence_id"),
        nullable=False,
    )
    task_id: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    target_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        # Mirrors the ``ix_evidence_uses_task`` index installed by the
        # migration — the per-task lookup path.
        Index("ix_evidence_uses_task", "task_id"),
        # The ``ON CONFLICT (run_id, task_id, evidence_id, purpose, target_id)``
        # arbiter. ``NULLS NOT DISTINCT`` so targetless uses collide too.
        Index(
            "uq_evidence_use_key",
            "run_id",
            "task_id",
            "evidence_id",
            "purpose",
            "target_id",
            unique=True,
            postgresql_nulls_not_distinct=True,
        ),
    )

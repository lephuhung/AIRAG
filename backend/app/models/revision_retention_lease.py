"""RevisionRetentionLease ORM — Phase 1B.

Maps the ``revision_retention_leases`` table. One row per acquired
GC retention lease; the table is the GC anchor for resumable
worker runs.

``run_id`` + ``revision_id`` + ``evidence_use_id`` form the
null-safe unique key ``uq_revision_lease_run_revision_use``:

- ``NULLS NOT DISTINCT`` semantics (PG15+) means two rows with NULL
  ``evidence_use_id`` still collide on ``(run_id, revision_id)``.
- ``evidence_use_id`` is NULL when no evidence use is bound to the
  lease (the typical case for a worker holding a revision for
  vector / caption / KG work).
- The partial index ``ix_leases_active`` on
  ``(revision_id, expires_at) WHERE released_at IS NULL`` is the
  fast lookup path for "is there an active lease for this revision?".
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
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class RevisionRetentionLease(Base):
    __tablename__ = "revision_retention_leases"

    lease_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.revision_id", ondelete="RESTRICT"),
        nullable=False,
    )
    # NULL when the lease is bound to a revision but not to a
    # specific evidence use. The null-safe UNIQUE lets multiple such
    # leases still collide on (run_id, revision_id) — i.e. a single
    # run can hold at most one active lease per revision.
    evidence_use_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    acquired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    released_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    release_reason: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    __table_args__ = (
        # Mirrors the named SQL constraint
        # ``uq_revision_lease_run_revision_use UNIQUE NULLS NOT
        # DISTINCT (run_id, revision_id, evidence_use_id)`` installed
        # by the migration. SQLAlchemy's ``UniqueConstraint`` does not
        # have a ``NULLS NOT DISTINCT`` flag at the moment (the
        # constraint options are DB-specific), so the null-distinct
        # semantics are an in-DB concern; this ``UniqueConstraint``
        # reproduces the constraint name and column set so a
        # ``create_all`` on a fresh DB matches the migration.
        UniqueConstraint(
            "run_id",
            "revision_id",
            "evidence_use_id",
            name="uq_revision_lease_run_revision_use",
        ),
        # Mirrors the ``ix_leases_run`` index.
        Index("ix_leases_run", "run_id"),
        # The ``ix_leases_active`` partial index is reproduced here
        # with the ``postgresql_where`` predicate so a ``create_all``
        # on a fresh DB issues the same DDL as the migration. SQLAlchemy
        # 2.0's ``Index(..., postgresql_where=...)`` emits
        # ``CREATE INDEX ... WHERE <predicate>`` for the PostgreSQL
        # dialect; the predicate is otherwise ignored, but
        # ``create_all`` is restricted to v2 tables via the
        # ``LEGACY_STARTUP_TABLES`` allowlist so this index never fires
        # at startup — the migration owns the partial form.
        Index(
            "ix_leases_active",
            "revision_id",
            "expires_at",
            postgresql_where=text("released_at IS NULL"),
        ),
    )

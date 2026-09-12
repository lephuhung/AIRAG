"""DocumentRevision ORM — Phase 1B.

Maps the ``document_revisions`` table created by Release 1A
(``app.services.agents.v2.persistence.migrate``). Phase 1B only
*maps* the schema; it never creates or alters the table.

A document revision is an immutable, monotonic per-document
allocation row that captures the full pipeline state for one
attempt to publish a build. Every retry increments ``generation``
(unique per ``document_id``); every published revision becomes the
document's ``current_revision_id`` (legacy pointer).

Status values (column is ``TEXT``, not an enum, because the brief treats
the enum as application-level state and the migration set the column
to free-form TEXT):

- ``draft``      — allocated, no worker stage has run yet
- ``building``   — a worker stage is in progress
- ``verified``   — verify_draft succeeded, awaiting publish CAS
- ``published``  — terminal, atomic publication succeeded (current or historical)
- ``failed``     — terminal, ``failure_class`` recorded
- ``abandoned``  — terminal, ``abandon_reason`` recorded (GC-blocked)

Supersession is tracked via ``superseded_at`` / ``superseded_by`` and does
NOT change ``status``: a superseded revision stays ``published`` so
Predicate B can still see it. There is deliberately no ``superseded`` or
``purged`` status (the repository never writes one).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DocumentRevision(Base):
    __tablename__ = "document_revisions"

    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="RESTRICT"),
        nullable=False,
    )
    # Per-document monotonic allocation counter; unique per document.
    # Application code allocates via SELECT ... FOR UPDATE on the
    # ``documents`` row.
    generation: Mapped[int] = mapped_column(
        BigInteger, nullable=False
    )
    # Retry provenance: if this revision is a retry of an earlier
    # terminal revision, ``retry_of_revision_id`` points at it.
    retry_of_revision_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.revision_id"),
        nullable=True,
    )

    # Lifecycle state (free-form TEXT — see module docstring).
    status: Mapped[str] = mapped_column(Text, nullable=False)

    # Publication timestamp, set exactly when the revision wins its
    # publish CAS (or is published historical). NULL until published.
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Terminal-failure metadata. Populated only when status='failed'.
    failed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    failure_stage: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    failure_class: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    # Terminal-abandon metadata. Populated only when status='abandoned'.
    abandoned_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    abandon_reason: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    # The single GC retention anchor: when the artifact retention clock
    # starts. Set when the revision becomes *permanently non-current*
    # (superseded, tombstoned, failed, or abandoned) — a currently
    # published revision has NO anchor. NULL while draft/building and
    # while it is the current revision.
    artifact_retention_starts_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Independent GC lifecycle metadata.
    superseded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Provenance: the revision that superseded this one (NULL unless
    # this revision was current and lost the pointer to a newer one).
    superseded_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.revision_id"),
        nullable=True,
    )
    artifacts_purged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        # Mirrors the SQL UNIQUE (document_id, generation) installed by
        # the migration as an *anonymous* UNIQUE constraint (no name).
        # The application never allocates two revisions with the same
        # ``generation`` for the same document. We deliberately omit a
        # constraint ``name`` so that a future ``create_all`` on a
        # fresh DB issues an auto-named UNIQUE that matches the
        # migration's anonymous UNIQUE — naming it would invent a
        # constraint the migration never created.
        UniqueConstraint("document_id", "generation"),
        # The retry_of_revision_id FK is naturally nullable; we do not
        # add a CHECK that constrains it to failed revisions because
        # the application owns that rule and the DB cannot distinguish
        # the writer phase.
    )

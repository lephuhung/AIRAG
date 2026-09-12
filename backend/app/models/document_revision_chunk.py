"""DocumentRevisionChunk ORM — Phase 1B.

Maps the ``document_revision_chunks`` table. One row per derived
chunk owned by a revision. ``ordinal`` is the position within the
revision; ``(revision_id, ordinal)`` is unique.

Phase 1C's retrieval code reads these rows to fetch chunk payloads
from MinIO (the chunk payloads themselves are not stored in this
table — only the structural metadata; full text is loaded from the
artifact store keyed by ``chunk_id``).
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    ForeignKey,
    Integer,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DocumentRevisionChunk(Base):
    __tablename__ = "document_revision_chunks"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.revision_id", ondelete="RESTRICT"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        # Mirrors the SQL UNIQUE (revision_id, ordinal) installed by
        # the migration as an *anonymous* UNIQUE constraint (no name).
        # The chunk_id remains the PK so individual rows can be
        # referenced by retrieval callers; ordinal is the position,
        # chunk_id is the identity. We deliberately omit a constraint
        # ``name`` so that a future ``create_all`` on a fresh DB issues
        # an auto-named UNIQUE that matches the migration's anonymous
        # UNIQUE.
        UniqueConstraint("revision_id", "ordinal"),
    )

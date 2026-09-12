"""DocumentRevisionBuild ORM — Phase 1B.

Maps the ``document_revision_builds`` table. One row per immutable
build-profile attempt for a given revision; the
``(revision_id, build_profile)`` UNIQUE key prevents duplicate build
attempts within the same revision.

Embedding manifest columns:

- ``embedding_namespace``     — the Chroma namespace the artifacts
                                are persisted under.
- ``embedding_model_hash``    — hash of the embedding model id+config
                                that produced the vectors.
- ``embedding_dimension``     — vector dimension (matches the model's
                                output shape).
- ``vector_artifact_version`` — schema version of the persisted vector
                                artifact (so consumers can detect
                                upgrades / reindexes).

These four columns together pin a build's vector identity; if any of
them changes the existing vectors are invalidated for downstream
re-use, even if the chunk payloads are byte-identical.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DocumentRevisionBuild(Base):
    __tablename__ = "document_revision_builds"

    build_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.revision_id", ondelete="RESTRICT"),
        nullable=False,
    )
    # Free-form build-profile identifier (e.g. "FULL", "CHAT_UPLOAD",
    # "PARSE_ONLY"). Phase 1C defines the canonical enum.
    build_profile: Mapped[str] = mapped_column(Text, nullable=False)

    # Embedding manifest columns (see module docstring).
    embedding_namespace: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    embedding_model_hash: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    embedding_dimension: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    vector_artifact_version: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        # Mirrors the SQL UNIQUE (revision_id, build_profile) installed
        # by the migration as an *anonymous* UNIQUE constraint (no
        # name). The migration also installs the ``ix_revisions_*``
        # index family on ``document_revisions`` but the builds table
        # only needs the composite UNIQUE for the ON CONFLICT arbiter
        # in Phase 1C's build-pipeline code. We deliberately omit a
        # constraint ``name`` so that a future ``create_all`` on a
        # fresh DB issues an auto-named UNIQUE that matches the
        # migration's anonymous UNIQUE.
        UniqueConstraint("revision_id", "build_profile"),
    )

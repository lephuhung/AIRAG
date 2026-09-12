"""DocumentIngestionAttempt ORM — Phase 1B.

Maps the ``revision_ingestion_attempts`` table. The class name
follows the brief's filename convention (``document_ingestion_attempt.py``)
even though the underlying table is ``revision_ingestion_attempts``
(because attempts are owned by a revision once one exists).

The unique constraint ``uq_revision_ingestion_attempt_key`` on
``(document_id, source_scheme, source_bucket, source_object_key,
build_profile)`` is the ``ON CONFLICT`` arbiter for the ingestion
pipeline. Phase 1C's ``get_or_create_ingestion_attempt`` uses this
key as the idempotency boundary — re-delivery of the same webhook
returns the existing ``revision_id`` instead of allocating a new
generation.

Canonical source component columns (``source_scheme``,
``source_bucket``, ``source_object_key``, ``source_version_id``,
``source_etag``, ``source_size``, ``source_sha256``) capture every
selector used to identify the storage object:

- ``source_scheme``     — protocol (e.g. ``s3v1``)
- ``source_bucket``     — bucket name
- ``source_object_key`` — object key (already canonicalized)
- ``source_version_id`` — S3 VersionId (NULL when versioning is off)
- ``source_etag``       — ETag header (multipart etag is preserved)
- ``source_size``       — byte size
- ``source_sha256``     — content hash; the only content discriminator

The retry counter ``attempt_generation`` (``INT NOT NULL DEFAULT 1``)
is bounded by ``MAX_REVISION_RETRIES``; ``exhausted_at`` is set when
the bound is hit.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DocumentIngestionAttempt(Base):
    __tablename__ = "revision_ingestion_attempts"

    attempt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )

    # Canonical source component columns (see module docstring).
    source_scheme: Mapped[str] = mapped_column(Text, nullable=False)
    source_bucket: Mapped[str] = mapped_column(Text, nullable=False)
    source_object_key: Mapped[str] = mapped_column(Text, nullable=False)
    source_version_id: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    source_etag: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_size: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True
    )
    source_sha256: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    # Retry counter; bumped per retry attempt. ``INT`` (not BIGINT)
    # because the brief caps retries via ``MAX_REVISION_RETRIES``
    # (a small integer).
    attempt_generation: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )
    exhausted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    build_profile: Mapped[str] = mapped_column(Text, nullable=False)

    # FK to the revision this attempt produced (NULL for in-flight
    # attempts before the draft is allocated).
    revision_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.revision_id"),
        nullable=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        # Mirrors the named SQL constraint
        # ``uq_revision_ingestion_attempt_key UNIQUE
        # (document_id, source_scheme, source_bucket, source_object_key,
        # build_profile)`` installed by the migration. The name is
        # preserved so any DDL emitted by ``create_all`` on a fresh
        # DB matches the migration's expectation.
        UniqueConstraint(
            "document_id",
            "source_scheme",
            "source_bucket",
            "source_object_key",
            "build_profile",
            name="uq_revision_ingestion_attempt_key",
        ),
        # Mirrors the ``ix_attempts_doc`` and ``ix_attempts_revision``
        # indexes installed by the migration.
        Index("ix_attempts_doc", "document_id"),
        Index("ix_attempts_revision", "revision_id"),
    )

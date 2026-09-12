"""SourceArrival ORM — Phase 1B.

Maps the ``source_arrivals`` table. One row per MinIO/S3 webhook
event; the table is the staging queue between the webhook receiver
and the ingestion pipeline.

``arrival_identity`` is the canonical idempotency key for the
webhook (a hash of bucket + object_key + version_id + etag + size);
the UNIQUE constraint makes duplicate webhook deliveries a no-op.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SourceArrival(Base):
    __tablename__ = "source_arrivals"

    arrival_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Storage object coordinates. ``version_id`` and ``etag`` are NULL
    # when the bucket has versioning disabled.
    bucket: Mapped[str] = mapped_column(Text, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    version_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    etag: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True
    )

    # Canonical idempotency key (see module docstring).
    arrival_identity: Mapped[str] = mapped_column(Text, nullable=False)

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        # Mirrors the SQL UNIQUE (arrival_identity) installed by the
        # migration as an *anonymous* UNIQUE constraint (no name).
        # Webhook delivery dedup keys off this. We deliberately omit
        # a constraint ``name`` so that a future ``create_all`` on a
        # fresh DB issues an auto-named UNIQUE that matches the
        # migration's anonymous UNIQUE.
        UniqueConstraint("arrival_identity"),
        # Mirrors the ``ix_arrivals_received`` index installed by the
        # migration.
        Index("ix_arrivals_received", "received_at"),
    )

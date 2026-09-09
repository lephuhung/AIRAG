"""
DocumentAlias model — human-readable / abbreviated names for documents.

Per spec Section B.4 (Q9.A): exact-match alias lookup for safe_lookup_metadata_only.
Unique constraint: (alias_text, workspace_id, alias_type).
NFC-normalized + lowercase at write time; queried as-is (exact equality).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, ForeignKey, DateTime, UniqueConstraint, Index, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DocumentAlias(Base):
    __tablename__ = "document_aliases"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    # NFC-normalized, lowercase. Store as-is for exact-match SQL (no lower()/unaccent).
    alias_text: Mapped[str] = mapped_column(String(512), nullable=False)
    # "exact_title" | "common_name" | "abbreviation"
    alias_type: Mapped[str] = mapped_column(String(32), nullable=False)
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "alias_text",
            "workspace_id",
            "alias_type",
            name="uq_alias_text_workspace_type",
        ),
        Index("ix_alias_workspace", "workspace_id"),
        Index("ix_alias_text", "alias_text"),
    )

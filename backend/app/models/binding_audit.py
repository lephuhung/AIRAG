"""BindingAudit ORM — Phase 1B (Task 7 amendment).

Maps the ``binding_audit`` table. One row per binding decision recorded by
the v2 Binding Resolver; this is the immutable audit trail that forensics /
observability code reads when a binding error is reported.

The frozen
:class:`~app.services.agents.v2.contracts.binding.BindingAuditRow` contract is
persisted as ``contract_version`` + the discriminated ``provenance`` JSONB,
with ``provenance_kind`` mirrored as an indexed discriminator for audit
queries. The table is append-only: it has no unique key, because recording
the same binding decision twice is two audit facts, not a duplicate row.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class BindingAudit(Base):
    __tablename__ = "binding_audit"

    audit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    contract_version: Mapped[str] = mapped_column(Text, nullable=False)
    provenance_kind: Mapped[str] = mapped_column(Text, nullable=False)
    provenance: Mapped[dict] = mapped_column(JSONB, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_binding_audit_thread_recorded", "thread_id", "recorded_at"),
    )

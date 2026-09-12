"""BindingAudit ORM — Phase 1B.

Maps the ``binding_audit`` table. One row per binding decision
recorded by the v2 routing layer. This is the immutable audit
trail that downstream forensics / observability code reads when a
binding error is reported.

Only the structural columns are mapped in Phase 1B; the per-binding
detail columns (decision type, evidence_use_ids, run_id, etc.) are
introduced in Phase 1D when the binding-decision code lands.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class BindingAudit(Base):
    __tablename__ = "binding_audit"

    audit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

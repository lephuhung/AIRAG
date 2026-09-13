"""DocumentRevisionStage ORM — P1 Task 1.

Maps the ``document_revision_stages`` table created by the v2 migration
(version 4). One row per (revision, canonical stage); the composite
``PRIMARY KEY (revision_id, stage)`` is the plan-mandated uniqueness
boundary. The ``revision_id`` FK is ``ON DELETE RESTRICT``: stage rows
are revision-owned evidence and must never disappear via cascade.

Canonical stages are ``parse`` / ``embed`` / ``caption`` / ``kg``;
states are ``pending`` / ``running`` / ``completed`` / ``skipped`` /
``failed``. Both sets are enforced by DB CHECK constraints (the column
stays free-form TEXT so the application owns the transition rules;
see :class:`DocumentRevisionsRepository` in
``app.services.agents.v2.persistence.document_revisions``).

``skipped`` is explicit and profile-authorized only: the repository's
``initialize_stages`` writes skipped rows for stages the revision's
immutable build profile does not require, and
``required_stages_complete`` counts a skip only for its matching
profile. ``Document.*_done`` mirror flags are never consulted.
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
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DocumentRevisionStage(Base):
    __tablename__ = "document_revision_stages"

    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.revision_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    # Canonical stage literal (``parse`` / ``embed`` / ``caption`` / ``kg``).
    # Part of the composite primary key: one row per (revision, stage).
    stage: Mapped[str] = mapped_column(Text, primary_key=True)

    # Lifecycle state (free-form TEXT — the DB CHECK constrains the
    # exact literals; the repository owns the transition rules).
    state: Mapped[str] = mapped_column(
        Text, nullable=False, default="pending", server_default="pending"
    )

    # Worker attempt counter. Bumped exactly on ``pending -> running``;
    # redelivered running/completed marks converge without bumping.
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Terminal-failure classification. Populated only when
    # ``state='failed``.
    failure_class: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

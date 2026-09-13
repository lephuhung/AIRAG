"""AgentRolloutMetric ORM — Phase 3 Task 7B.

Maps the ``agent_rollout_metrics`` table OWNED by the Task 7A migration
(``app.services.agents.v2.persistence.migrate``, ``V2_SCHEMA_VERSION=3``).
This module only maps the exact T7A column set — it never creates the
table (no ``create_all``; the migration owns DDL).

APPEND-ONLY: the migration installs a DB trigger rejecting UPDATE/DELETE.
This module exposes NO update/delete helper by construction — inserts only,
via ``app.services.agent.rollout_metrics.record_rollout_metric``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Integer,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AgentRolloutMetric(Base):
    __tablename__ = "agent_rollout_metrics"
    __table_args__ = (
        CheckConstraint(
            "arm IN ('v1', 'v2', 'shadow')",
            name="agent_rollout_metrics_arm_check",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        # T7A migration owns the table (BIGSERIAL): mirror its
        # ``nextval('agent_rollout_metrics_id_seq')`` default exactly so
        # ORM metadata matches the frozen column contract.
        server_default=text("nextval('agent_rollout_metrics_id_seq'::regclass)"),
    )
    arm: Mapped[str] = mapped_column(Text, nullable=False)
    request_id_hash: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_id_hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    terminal_status: Mapped[str] = mapped_column(Text, nullable=False)
    citation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    cancelled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    security_checkpoint_secret: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    security_ungrounded_factual_success: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    security_acl_leak: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    security_duplicate_production_write: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

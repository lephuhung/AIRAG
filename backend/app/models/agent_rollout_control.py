"""AgentRolloutControl ORM — Phase 3 Task 7B.

Maps the ``agent_rollout_control`` table OWNED by the Task 7A migration
(``app.services.agents.v2.persistence.migrate``, ``V2_SCHEMA_VERSION=3``).
This module only maps the exact T7A column set — it never creates the
table (no ``create_all``; the migration owns DDL + the seeded disabled
row ``id=1``).

The single control row (``id=1``) is authoritative for canary selection
within the environment ceilings (``NEXUSRAG_AGENT_V2_*``); see
``app.services.agent.rollout_control``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, Integer, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AgentRolloutControl(Base):
    __tablename__ = "agent_rollout_control"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    shadow_percent: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    canary_percent: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    canary_workspaces: Mapped[Any] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    kill_switch: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_by: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )

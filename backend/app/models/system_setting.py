"""
SystemSetting model — DB-backed runtime overrides for LLM role configuration.

Backs the WebUI LLM-config feature (docs/plan-llm-runtime-config.md): each row
is an override on top of the `.env` default for one LLM role (llm.main,
llm.thinking, …). The special key `_config_version` stores a plain integer that
workers/backend poll to detect config changes without any push infrastructure.

`value_enc` is a JSON document; nested API keys are Fernet-encrypted (see
app/services/runtime_config.py).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SystemSetting(Base):
    __tablename__ = "system_settings"

    # Setting key — role overrides use the "llm.<role>" namespace;
    # "_config_version" is the reserved version-counter row.
    key: Mapped[str] = mapped_column(String(128), primary_key=True)

    # JSON payload with the api_key stored as Fernet ciphertext.
    value_enc: Mapped[str] = mapped_column(Text, nullable=False)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default="now()"
    )
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

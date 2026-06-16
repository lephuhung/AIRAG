"""
Integration models — third-party channel access (API keys) and Telegram linking.

These power the `/integrations/*` endpoints:
  - ApiKey            : generic service-to-service auth for third parties
                        (Telegram bot backend, Zalo, n8n, Slack, ...). The plaintext
                        key is shown ONCE on creation; only its sha256 hash is stored.
  - TelegramLink      : binds a Telegram chat_id to a real AIRAG user so the agent
                        runs with that user's exact permissions / tenant / workspaces.
  - TelegramLinkCode  : short-lived one-time code minted on the web UI, redeemed in
                        the bot via /start <code> to create a TelegramLink.
  - TelegramBotConfig : system-wide singleton holding the bot token / webhook
                        secret / username, configured from the admin UI (NOT .env).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ApiKey(Base):
    """A revocable API key that authenticates a third party AS a given user."""

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # sha256 hex of the plaintext key — the plaintext itself is never stored.
    key_hash: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    # First chars of the plaintext (e.g. "nrk_a1b2c3") for display in the UI.
    prefix: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="API Key")

    # The user this key acts as — requests authenticate as this principal and
    # therefore inherit exactly this user's workspace/tenant permissions.
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    # Who created the key (for audit). Usually a superadmin or the user themself.
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Optional coarse scopes, e.g. ["chat"]. Empty/None == full access of the user.
    scopes: Mapped[list | None] = mapped_column(JSON, nullable=True)

    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TelegramLink(Base):
    """Binds a Telegram chat to an AIRAG user + the chat's active context."""

    __tablename__ = "telegram_links"

    # Telegram chat id is the natural primary key (one link per chat).
    telegram_chat_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # Telegram numeric user id (message.from.id). In private chats this equals
    # telegram_chat_id, but for groups it differs — store it so we always know
    # *which Telegram user* is bound to this AIRAG account.
    telegram_user_id: Mapped[str | None] = mapped_column(
        String(64), index=True, nullable=True
    )

    # Telegram profile snapshot (best-effort, for display / debugging).
    telegram_username: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Active conversation context for this chat. Both nullable:
    #   active_workspace_id is None  -> search across ALL accessible workspaces.
    #   active_session_id is None    -> a fresh session is created on next message.
    active_workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"),
        nullable=True,
    )
    active_session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TelegramLinkCode(Base):
    """One-time, short-lived code minted on the web to link a Telegram chat."""

    __tablename__ = "telegram_link_codes"

    code: Mapped[str] = mapped_column(String(16), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TelegramBotConfig(Base):
    """System-wide Telegram bot settings, managed from the admin UI (one row).

    Replaces the old `.env` (`TELEGRAM_BOT_TOKEN` / `TELEGRAM_WEBHOOK_SECRET` /
    `TELEGRAM_BOT_USERNAME`) so a superadmin can configure the bot, register the
    webhook and test the connection entirely from the web. Always stored under the
    fixed primary key `id == 1`.
    """

    __tablename__ = "telegram_bot_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)

    # Bot credentials / identity. The token is the only secret here.
    bot_token: Mapped[str | None] = mapped_column(String(255), nullable=True)
    webhook_secret: Mapped[str | None] = mapped_column(String(128), nullable=True)
    bot_username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # The bot's own numeric Telegram id (from getMe), for reference.
    bot_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Full public webhook URL registered with Telegram (e.g.
    # https://service.hatinh.local/api/v1/integrations/telegram/webhook).
    webhook_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

"""
Tests for chat_messages.semantic_context column.

Per spec Section A.8 / B.11 0.2: nullable JSONB column.
Column only — persistence write deferred to Task 2.5 (depends on contracts).
"""

from __future__ import annotations

import uuid
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncIterator


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    """SAVEPOINT pattern — clean isolation."""
    from app.core.database import async_session_maker

    async with async_session_maker() as session:
        await session.begin()
        nested = await session.begin_nested()
        try:
            yield session
        finally:
            try:
                await nested.rollback()
            except Exception:
                pass
            await session.rollback()


async def _ensure_user_and_session(db: AsyncSession):
    """Create a user + chat session so FK constraints are satisfied."""
    from app.models.user import User
    from app.models.chat_session import ChatSession

    user = User(
        email=f"test-{uuid.uuid4().hex[:8]}@example.com",
        full_name="Test User",
        password_hash="x",
        is_active=True,
        is_superadmin=False,
    )
    db.add(user)
    await db.flush()

    session = ChatSession(
        id=uuid.uuid4(),
        title="Test",
        user_id=user.id,
    )
    db.add(session)
    await db.flush()
    return session.id


@pytest.mark.asyncio
async def test_semantic_context_column_exists(db: AsyncSession):
    """Column must exist in the DB schema."""
    from sqlalchemy import text
    result = await db.execute(
        text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'chat_messages' AND column_name = 'semantic_context'"
        )
    )
    row = result.fetchone()
    assert row is not None, "semantic_context column not found"
    assert row[1] == "jsonb", f"Expected jsonb, got {row[1]}"


@pytest.mark.asyncio
async def test_semantic_context_nullable(db: AsyncSession):
    """Column must be nullable (preprocessor disabled = NULL)."""
    from sqlalchemy import text
    result = await db.execute(
        text(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = 'chat_messages' AND column_name = 'semantic_context'"
        )
    )
    row = result.fetchone()
    assert row is not None
    assert row[0] == "YES", "semantic_context should be nullable"


@pytest.mark.asyncio
async def test_semantic_context_writes_null(db: AsyncSession):
    """With preprocessor disabled, semantic_context stays NULL."""
    from app.models.chat_session import ChatSession
    from app.models.chat_message import ChatMessage

    session_id = await _ensure_user_and_session(db)

    msg = ChatMessage(
        session_id=session_id,
        message_id=f"msg-sem-{uuid.uuid4().hex[:8]}",
        role="user",
        content="Test query",
        semantic_context=None,  # preprocessor disabled
    )
    db.add(msg)
    await db.commit()

    from sqlalchemy import select
    result_row = await db.execute(
        select(ChatMessage).where(ChatMessage.message_id == msg.message_id)
    )
    reloaded = result_row.scalar_one()
    assert reloaded.semantic_context is None

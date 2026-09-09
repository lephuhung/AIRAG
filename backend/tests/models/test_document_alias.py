"""
Tests for DocumentAlias model.

Per spec Section B.4 (Q9.A): unique constraint on
(alias_text, workspace_id, alias_type); same text different type allowed.
"""

from __future__ import annotations

import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError


# ---------------------------------------------------------------------------
# DB session fixture: SAVEPOINT pattern, rolls back at end of test.
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    """Real AsyncSession; nested transaction (SAVEPOINT) absorbs commits;
    outer transaction rolled back at teardown for clean isolation."""
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


async def _ensure_workspace_and_doc(session: AsyncSession):
    """Create user + workspace + document via ORM so all FK defaults are handled."""
    from app.models.user import User
    from app.models.knowledge_base import KnowledgeBase
    from app.models.document import Document

    user = User(
        email=f"test-{uuid.uuid4().hex[:8]}@example.com",
        full_name="Test User",
        password_hash="x",
        is_active=True,
        is_superadmin=False,
    )
    session.add(user)
    await session.flush()  # get user.id

    ws = KnowledgeBase(
        name="Test Workspace",
        description="test",
        owner_id=user.id,  # FK after user flushed
    )
    session.add(ws)
    await session.flush()  # get ws.id

    doc = Document(
        workspace_id=ws.id,
        filename="test.pdf",
        original_filename="test.pdf",
        file_type="pdf",
        file_size=1024,
    )
    session.add(doc)
    await session.flush()
    return ws.id, doc.id


@pytest.mark.asyncio
async def test_document_alias_unique_constraint_same_triple(db: AsyncSession):
    """Same (alias_text, workspace_id, alias_type) twice → IntegrityError."""
    from app.models.document_alias import DocumentAlias

    ws_id, doc_id = await _ensure_workspace_and_doc(db)

    alias = DocumentAlias(
        document_id=doc_id,
        alias_text="luật an ninh mạng",
        alias_type="exact_title",
        workspace_id=ws_id,
    )
    db.add(alias)
    await db.commit()

    # Second insert with same triple → must raise
    alias2 = DocumentAlias(
        document_id=doc_id,
        alias_text="luật an ninh mạng",
        alias_type="exact_title",
        workspace_id=ws_id,
    )
    db.add(alias2)
    with pytest.raises(IntegrityError):
        await db.commit()
    await db.rollback()


@pytest.mark.asyncio
async def test_document_alias_different_type_allows_duplicate_text(db: AsyncSession):
    """Same text different type → allowed."""
    from app.models.document_alias import DocumentAlias

    ws_id, doc_id = await _ensure_workspace_and_doc(db)

    a1 = DocumentAlias(
        document_id=doc_id,
        alias_text="luật an ninh mạng",
        alias_type="exact_title",
        workspace_id=ws_id,
    )
    a2 = DocumentAlias(
        document_id=doc_id,
        alias_text="luật an ninh mạng",
        alias_type="common_name",
        workspace_id=ws_id,
    )
    a3 = DocumentAlias(
        document_id=doc_id,
        alias_text="luật an ninh mạng",
        alias_type="abbreviation",
        workspace_id=ws_id,
    )
    db.add_all([a1, a2, a3])
    await db.commit()  # Must NOT raise


@pytest.mark.asyncio
async def test_document_alias_different_workspace_allows_duplicate_text(db: AsyncSession):
    """Same text same type different workspace → allowed."""
    from app.models.document_alias import DocumentAlias
    from app.models.knowledge_base import KnowledgeBase

    ws_a_id, doc_id = await _ensure_workspace_and_doc(db)

    ws_b = KnowledgeBase(
        name="Workspace B",
        description="test",
        owner_id=None,
    )
    db.add(ws_b)
    await db.flush()
    ws_b_id = ws_b.id

    a1 = DocumentAlias(
        document_id=doc_id,
        alias_text="luật an ninh mạng",
        alias_type="exact_title",
        workspace_id=ws_a_id,
    )
    a2 = DocumentAlias(
        document_id=doc_id,
        alias_text="luật an ninh mạng",
        alias_type="exact_title",
        workspace_id=ws_b_id,
    )
    db.add_all([a1, a2])
    await db.commit()  # Must NOT raise

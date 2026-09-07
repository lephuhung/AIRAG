"""
Task 1 / B1 — attachment access vs. delete ownership ACL.

The pre-existing ``delete_chat_session`` endpoint only filtered
``is_chat_upload == True`` (chat_session.py L228-L234) and was in scope
only because ``ChatSession.user_id == user.id`` (L166). The bug fixed
here is that an uploader's chat upload appearing in another user's
session (via shared workspace reference) could be deleted by that
other user. Rule B closes the hole by re-requiring
``uploaded_by == user.id`` AT delete time, with the session id passed
in to ensure the doc was attached to THAT session.

Tests use the existing hrag Postgres DB with a SAVEPOINT/ROLLBACK
pattern: every fixture commits inside a SAVEPOINT, and the session-
level transaction is rolled back after the test, so no test data
survives across runs.

No live storage / RAG / MinIO operations: storage_service and RAG
service are stubbed with AsyncMock so the delete path can be
exercised without touching production data.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat_session import (
    _authorize_delete,
    _filter_accessible_document_ids,
)
from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession
from app.models.document import Document, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User


# ---------------------------------------------------------------------------
# DB session fixture: SAVEPOINT pattern, rolls back at end of test.
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    """Real AsyncSession over the configured DB, but every test runs
    inside a nested transaction (SAVEPOINT) that is rolled back at
    teardown so no test data survives. This matches the production
    SQLAlchemy contract without requiring a separate test database.

    Implementation: open an outer transaction, then begin a nested
    transaction (``session.begin_nested()``) inside it. The inner
    SAVEPOINT absorbs any commits the test code issues; we roll back
    the outer transaction at teardown to wipe everything cleanly.
    """
    from app.core.database import async_session_maker

    async with async_session_maker() as session:
        # Outer transaction — never committed; rolled back at teardown.
        await session.begin()
        try:
            yield session
        finally:
            await session.rollback()


@pytest_asyncio.fixture
async def user_a(db: AsyncSession) -> User:
    u = User(
        id=uuid.uuid4(),
        email=f"task1-user-a-{uuid.uuid4().hex[:8]}@example.com",
        full_name="Task1 User A",
        password_hash="x",
        is_active=True,
    )
    db.add(u)
    await db.commit()
    return u


@pytest_asyncio.fixture
async def user_b(db: AsyncSession) -> User:
    u = User(
        id=uuid.uuid4(),
        email=f"task1-user-b-{uuid.uuid4().hex[:8]}@example.com",
        full_name="Task1 User B",
        password_hash="x",
        is_active=True,
    )
    db.add(u)
    await db.commit()
    return u


@pytest_asyncio.fixture
async def make_workspace(db: AsyncSession):
    """Factory: create a KnowledgeBase owned by ``owner`` with optional
    visibility override; returns the KB id.

    The KB is needed because Document.workspace_id has a FK constraint
    to knowledge_bases.id. SAVEPOINT rollback in the ``db`` fixture
    cleans up the KB along with everything else the test created.
    """
    created: list[KnowledgeBase] = []

    async def _factory(owner: User, visibility: str = "public") -> KnowledgeBase:
        kb = KnowledgeBase(
            id=uuid.uuid4(),
            name=f"task1-ws-{uuid.uuid4().hex[:8]}",
            owner_id=owner.id,
            visibility=visibility,
        )
        db.add(kb)
        await db.commit()
        created.append(kb)
        return kb

    return _factory


@pytest.fixture
def fake_storage() -> AsyncMock:
    """AsyncMock replacing ``storage_service.delete_file`` /
    ``delete_markdown``. The production delete path calls these methods
    on the singleton returned by ``get_storage_service()``; we patch
    the singleton so the tests can assert call counts.
    """
    storage = MagicMock()
    storage.delete_file = AsyncMock()
    storage.delete_markdown = AsyncMock()
    return storage


@pytest.fixture
def fake_rag() -> AsyncMock:
    rag = MagicMock()
    rag.delete_document = AsyncMock()
    return rag


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_other_users_chat_upload_attached_to_caller_session_is_NOT_deleted(
    db, fake_storage, fake_rag, user_a, user_b, make_workspace,
):
    """The vulnerable relationship the brief fixes:

    user_b uploaded doc_d (is_chat_upload=True) in workspace W (shared
    with user_a). A message in user_a's session S_A references doc_d
    (e.g. via shared workspace chat history). When user_a calls
    `delete_chat_session(S_A)`, the prior code DID delete doc_d — even
    though user_a is NOT the uploader. This test asserts the fix
    (rule B re-requires uploaded_by == caller).
    """
    kb = await make_workspace(user_b, visibility="public")
    ws = kb.id
    sess_a = ChatSession(id=uuid.uuid4(), user_id=user_a.id, title="A")
    doc_d = Document(
        id=uuid.uuid4(),
        workspace_id=ws,
        uploaded_by=user_b.id,
        is_chat_upload=True,
        upload_s3_key="x",
        markdown_s3_key="y",
        filename="user_b_doc.pdf",
        original_filename="user_b_doc.pdf",
        file_type="pdf",
        file_size=1024,
        status=DocumentStatus.INDEXED,
    )
    msg = ChatMessage(
        id=uuid.uuid4(),
        session_id=sess_a.id,
        role="user",
        content="",
        message_id=f"msg_{uuid.uuid4().hex[:8]}",
        document_ids=[str(doc_d.id)],
    )
    db.add_all([sess_a, doc_d, msg])
    await db.commit()

    allowed = await _authorize_delete(
        db, user_a, [doc_d.id], caller_session_id=str(sess_a.id)
    )
    assert allowed == [], (
        f"another uploader's chat upload must NOT be deletable by user_a; "
        f"got allowed={allowed!r}"
    )


@pytest.mark.asyncio
async def test_own_chat_upload_attached_to_caller_session_IS_deleted(
    db, fake_storage, fake_rag, user_a, make_workspace,
):
    """Positive control: user_a uploaded doc_d (chat upload) and a
    message in user_a's session S_A references doc_d. ``_authorize_delete``
    returns doc_d.id because rule A (uploaded_by == user.id) AND rule B
    (chat upload attached to caller's session) both pass.
    """
    kb = await make_workspace(user_a)
    sess = ChatSession(id=uuid.uuid4(), user_id=user_a.id, title="A")
    doc_d = Document(
        id=uuid.uuid4(),
        workspace_id=kb.id,
        uploaded_by=user_a.id,
        is_chat_upload=True,
        upload_s3_key="x",
        markdown_s3_key="y",
        filename="user_a_doc.pdf",
        original_filename="user_a_doc.pdf",
        file_type="pdf",
        file_size=1024,
        status=DocumentStatus.INDEXED,
    )
    msg = ChatMessage(
        id=uuid.uuid4(),
        session_id=sess.id,
        role="user",
        content="",
        message_id=f"msg_{uuid.uuid4().hex[:8]}",
        document_ids=[str(doc_d.id)],
    )
    db.add_all([sess, doc_d, msg])
    await db.commit()

    allowed = await _authorize_delete(
        db, user_a, [doc_d.id], caller_session_id=str(sess.id)
    )
    assert allowed == [doc_d.id], (
        f"own chat upload attached to caller's session MUST be deletable; "
        f"got allowed={allowed!r}"
    )


@pytest.mark.asyncio
async def test_non_chat_upload_in_session_without_ownership_NOT_deleted(
    db, fake_storage, fake_rag, user_a, user_b, make_workspace,
):
    """Non-chat-upload docs in a shared workspace: read access does NOT
    imply delete authority even if a message in the caller's session
    references the doc.

    user_b uploaded a non-chat-upload doc into a workspace shared with
    user_a. user_a's session has a message referencing that doc. When
    user_a tries to delete their session, the doc must NOT be deleted.
    """
    kb = await make_workspace(user_b, visibility="public")
    sess_a = ChatSession(id=uuid.uuid4(), user_id=user_a.id, title="A")
    doc_d = Document(
        id=uuid.uuid4(),
        workspace_id=kb.id,
        uploaded_by=user_b.id,
        is_chat_upload=False,  # ← workspace upload, not chat
        upload_s3_key="x",
        markdown_s3_key="y",
        filename="user_b_workspace_doc.pdf",
        original_filename="user_b_workspace_doc.pdf",
        file_type="pdf",
        file_size=1024,
        status=DocumentStatus.INDEXED,
    )
    msg = ChatMessage(
        id=uuid.uuid4(),
        session_id=sess_a.id,
        role="user",
        content="",
        message_id=f"msg_{uuid.uuid4().hex[:8]}",
        document_ids=[str(doc_d.id)],
    )
    db.add_all([sess_a, doc_d, msg])
    await db.commit()

    allowed = await _authorize_delete(
        db, user_a, [doc_d.id], caller_session_id=str(sess_a.id)
    )
    assert allowed == [], (
        f"another uploader's non-chat-upload doc must NOT be deletable; "
        f"got allowed={allowed!r}"
    )


@pytest.mark.asyncio
async def test_filter_accessible_document_ids_filters_unscoped_docs(
    db, user_a, make_workspace,
):
    """``_filter_accessible_document_ids`` returns the intersection of
    `requested` with docs whose workspace is in `workspace_ids`. A
    doc in a workspace the user cannot access is dropped.
    """
    accessible_kb = await make_workspace(user_a)
    inaccessible_kb = await make_workspace(user_a)
    accessible_doc = Document(
        id=uuid.uuid4(),
        workspace_id=accessible_kb.id,
        uploaded_by=user_a.id,
        upload_s3_key="x",
        markdown_s3_key="y",
        filename="a_accessible.pdf",
        original_filename="a_accessible.pdf",
        file_type="pdf",
        file_size=1024,
        status=DocumentStatus.INDEXED,
    )
    inaccessible_doc = Document(
        id=uuid.uuid4(),
        workspace_id=inaccessible_kb.id,
        uploaded_by=user_a.id,
        upload_s3_key="x",
        markdown_s3_key="y",
        filename="a_inaccessible.pdf",
        original_filename="a_inaccessible.pdf",
        file_type="pdf",
        file_size=1024,
        status=DocumentStatus.INDEXED,
    )
    db.add_all([accessible_doc, inaccessible_doc])
    await db.commit()

    requested = [accessible_doc.id, inaccessible_doc.id]
    workspace_ids = [accessible_kb.id]

    filtered = await _filter_accessible_document_ids(
        db, user_a, workspace_ids, requested
    )
    assert filtered == [accessible_doc.id], (
        f"unscoped doc must be filtered out; got {filtered!r}"
    )


@pytest.mark.asyncio
async def test_filter_accessible_document_ids_handles_empty_request(db, user_a):
    """Empty requested list returns empty (no LLM call, no SQL call)."""
    filtered = await _filter_accessible_document_ids(db, user_a, [], [])
    assert filtered == []
    filtered = await _filter_accessible_document_ids(db, user_a, [uuid.uuid4()], None)
    assert filtered == []


# ---------------------------------------------------------------------------
# End-to-end endpoint test: helper-only assertions cannot prove endpoint
# wiring. A regression that removes the helper's call site from
# delete_chat_session would pass helper tests. This test drives the FULL
# delete_chat_session endpoint via httpx.AsyncClient + dependency_overrides
# with fake storage and fake RAG; it asserts no destructive operation ran
# on the other-uploader's document.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_chat_session_endpoint_rejects_other_uploader_chat_upload(
    db, fake_storage, fake_rag, user_a, user_b, make_workspace,
):
    """End-to-end through ``delete_chat_session``: user_b uploaded
    doc_d (is_chat_upload=True) in workspace W (shared with user_a).
    A message in user_a's session S_A references doc_d. When user_a
    DELETES session S_A via the endpoint, doc_d must NOT be
    deleted. ``fake_storage`` / ``fake_rag`` mocks are not called
    for doc_d, and the Document row still exists after the call.
    """
    from httpx import ASGITransport, AsyncClient

    from app.core.deps import get_current_active_user, get_db
    from app.main import app  # FastAPI app

    kb = await make_workspace(user_b, visibility="public")
    sess_a = ChatSession(id=uuid.uuid4(), user_id=user_a.id, title="A")
    doc_d = Document(
        id=uuid.uuid4(),
        workspace_id=kb.id,
        uploaded_by=user_b.id,
        is_chat_upload=True,
        upload_s3_key="x",
        markdown_s3_key="y",
        filename="user_b_doc.pdf",
        original_filename="user_b_doc.pdf",
        file_type="pdf",
        file_size=1024,
        status=DocumentStatus.INDEXED,
    )
    msg = ChatMessage(
        id=uuid.uuid4(),
        session_id=sess_a.id,
        role="user",
        content="",
        message_id=f"msg_{uuid.uuid4().hex[:8]}",
        document_ids=[str(doc_d.id)],
    )
    db.add_all([sess_a, doc_d, msg])
    await db.commit()
    doc_d_id = doc_d.id

    # Patch the storage singleton + the RAG factory. We capture the
    # AsyncMock so we can assert no destructive operation ran for doc_d.
    from app.services import storage_service as storage_service_module
    from app.services.retrieval import rag_service as rag_service_module

    with patch.object(
        storage_service_module, "get_storage_service", return_value=fake_storage
    ), patch.object(
        rag_service_module, "get_rag_service", return_value=fake_rag
    ):
        app.dependency_overrides[get_current_active_user] = lambda: user_a
        # Override get_db so the endpoint sees the same session the test
        # fixture built (the in-flight SAVEPOINT).
        app.dependency_overrides[get_db] = lambda: db

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.delete(f"/api/v1/rag/chat/sessions/{sess_a.id}")
        finally:
            app.dependency_overrides.pop(get_current_active_user, None)
            app.dependency_overrides.pop(get_db, None)

    # Endpoint must succeed (own session, deletable).
    assert r.status_code in (200, 204), (
        f"delete_chat_session should succeed for own session; "
        f"got status={r.status_code} body={r.text!r}"
    )

    # CRITICAL: doc_d's storage / RAG / DB delete MUST NOT have run.
    fake_storage.delete_file.assert_not_called()
    fake_storage.delete_markdown.assert_not_called()
    fake_rag.delete_document.assert_not_called()

    # Document row still exists (rule B rejection — defensive check
    # even though the endpoint never deletes it).
    surviving = (
        await db.execute(
            text("SELECT id FROM documents WHERE id = :id"),
            {"id": str(doc_d_id)},
        )
    ).first()
    assert surviving is not None, (
        "Document row was deleted even though the uploader was another user"
    )
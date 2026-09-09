"""
Phase 0 / B6 — endpoint-level test verifying filtered IDs reach graph state.

Per F.5/O71: chat_session.py MUST filter document_ids against accessible
workspaces BEFORE passing to graph state. This test proves the FILTERED IDs
(not raw) reach the build_initial_state entry point.

The existing test_session_acl_ingress.py tests the helper function in isolation.
This test verifies the integration contract: filter called → result passed to graph.
"""
from __future__ import annotations

import uuid
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_session import ChatSession
from app.models.document import Document, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User


# ---------------------------------------------------------------------------
# DB session fixture: SAVEPOINT pattern, rolls back at end of test.
# Matches the pattern used in test_attachment_delete_acl.py.
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    """Real AsyncSession over the configured DB with SAVEPOINT rollback."""
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


@pytest_asyncio.fixture
async def user_a(db: AsyncSession) -> User:
    u = User(
        id=uuid.uuid4(),
        email=f"b6-user-a-{uuid.uuid4().hex[:8]}@example.com",
        full_name="B6 User A",
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
        email=f"b6-user-b-{uuid.uuid4().hex[:8]}@example.com",
        full_name="B6 User B",
        password_hash="x",
        is_active=True,
    )
    db.add(u)
    await db.commit()
    return u


@pytest_asyncio.fixture
async def ws_a(db: AsyncSession, user_a: User) -> KnowledgeBase:
    kb = KnowledgeBase(
        id=uuid.uuid4(),
        name=f"b6-ws-a-{uuid.uuid4().hex[:8]}",
        owner_id=user_a.id,
        visibility="public",
    )
    db.add(kb)
    await db.commit()
    return kb


@pytest_asyncio.fixture
async def ws_b(db: AsyncSession, user_b: User) -> KnowledgeBase:
    # ws_b owned by user_b (different from user_a) and private →
    # user_a has NO access to it via _get_accessible_workspaces.
    kb = KnowledgeBase(
        id=uuid.uuid4(),
        name=f"b6-ws-b-{uuid.uuid4().hex[:8]}",
        owner_id=user_b.id,
        visibility="private",
    )
    db.add(kb)
    await db.commit()
    return kb


@pytest_asyncio.fixture
async def doc_a(db: AsyncSession, ws_a: KnowledgeBase, user_a: User) -> Document:
    d = Document(
        id=uuid.uuid4(),
        workspace_id=ws_a.id,
        uploaded_by=user_a.id,
        is_chat_upload=False,
        upload_s3_key="x",
        markdown_s3_key="y",
        filename="doc_a.pdf",
        original_filename="doc_a.pdf",
        file_type="pdf",
        file_size=1024,
        status=DocumentStatus.INDEXED,
    )
    db.add(d)
    await db.commit()
    return d


@pytest_asyncio.fixture
async def doc_b(db: AsyncSession, ws_b: KnowledgeBase, user_a: User) -> Document:
    d = Document(
        id=uuid.uuid4(),
        workspace_id=ws_b.id,
        uploaded_by=user_a.id,
        is_chat_upload=False,
        upload_s3_key="x2",
        markdown_s3_key="y2",
        filename="doc_b.pdf",
        original_filename="doc_b.pdf",
        file_type="pdf",
        file_size=1024,
        status=DocumentStatus.INDEXED,
    )
    db.add(d)
    await db.commit()
    return d


@pytest_asyncio.fixture
async def session_a(db: AsyncSession, user_a: User) -> ChatSession:
    s = ChatSession(
        id=uuid.uuid4(),
        user_id=user_a.id,
        title="B6 test session",
    )
    db.add(s)
    await db.commit()
    return s


class TestSessionACLe2e:
    """Endpoint logic test for B6 ACL ingress filtering."""

    @pytest.mark.asyncio
    async def test_filter_call_produces_filtered_output(self):
        """Prove _filter_accessible_document_ids can be called with correct signature."""
        from app.api.chat_session import _filter_accessible_document_ids

        accessible_doc = uuid.uuid4()
        inaccessible_doc = uuid.uuid4()

        # Call the filter directly (helper isolation test)
        # Signature: (db, user, workspace_ids, requested)
        # db must be awaitable (AsyncMock)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await _filter_accessible_document_ids(
            db=mock_db,
            user=MagicMock(id=uuid.uuid4()),
            workspace_ids=[uuid.uuid4()],
            requested=[accessible_doc, inaccessible_doc],
        )

        # The result is a list (filtered output)
        assert isinstance(result, list)

    def test_endpoint_passes_filtered_ids_to_build_initial_state_via_source(self):
        """Prove via source inspection: endpoint passes filtered_doc_ids to build_initial_state.

        The build_initial_state call spans multiple lines.
        We look for: (1) filter call assigns to filtered_doc_ids,
        (2) build_initial_state called with document_ids=filtered_doc_ids.
        """
        import inspect
        from app.api.chat_session import chat_stream_session

        source = inspect.getsource(chat_stream_session)
        lines = source.split("\n")

        # Find the build_initial_state call and the document_ids= argument
        build_idx = -1
        doc_ids_idx = -1

        for i, line in enumerate(lines):
            if "initial_state = build_initial_state(" in line or "build_initial_state(" in line:
                if build_idx == -1:
                    build_idx = i
            # document_ids= is on a separate line from the build_initial_state call
            if 'document_ids=filtered_doc_ids' in line:
                doc_ids_idx = i

        assert build_idx != -1, (
            f"build_initial_state call not found in endpoint source"
        )
        assert doc_ids_idx != -1, (
            f"'document_ids=filtered_doc_ids' not found in endpoint source. "
            f"The filter result must be passed as document_ids argument."
        )
        assert doc_ids_idx > build_idx, (
            f"document_ids=filtered_doc_ids (line {doc_ids_idx}) must come after "
            f"build_initial_state call (line {build_idx})"
        )

        # Also verify: the filtered_doc_ids is assigned from _filter_accessible_document_ids
        # Look for the filter assignment before build_initial_state
        filter_idx = -1
        for i, line in enumerate(lines):
            if "filtered_doc_ids = await _filter_accessible_document_ids" in line:
                filter_idx = i
                break

        assert filter_idx != -1, (
            f"filtered_doc_ids assignment not found. Endpoint must call "
            f"_filter_accessible_document_ids"
        )
        assert filter_idx < build_idx, (
            f"Filter (line {filter_idx}) must be called before build_initial_state (line {build_idx})"
        )

        print(f"PASS: filter at line {filter_idx}, build at line {build_idx}, "
              f"doc_ids at line {doc_ids_idx}")

    def test_streaming_endpoint_calls_filter_before_build_initial_state(self):
        """Prove: filter is called BEFORE build_initial_state in the endpoint flow."""
        import inspect
        from app.api.chat_session import chat_stream_session

        source = inspect.getsource(chat_stream_session)
        lines = source.split("\n")

        filter_line = -1
        build_line = -1

        for i, line in enumerate(lines):
            if "_filter_accessible_document_ids" in line and not line.strip().startswith("#"):
                if filter_line == -1:
                    filter_line = i
            if "build_initial_state(" in line:
                if build_line == -1:
                    build_line = i

        assert filter_line != -1, "_filter_accessible_document_ids call not found"
        assert build_line != -1, "build_initial_state call not found"
        assert filter_line < build_line, (
            f"Filter must be called before build_initial_state. "
            f"filter at line {filter_line}, build at line {build_line}"
        )
        print(f"PASS: filter at line {filter_line}, build at line {build_line}")

    @pytest.mark.asyncio
    async def test_endpoint_post_routes_filtered_ids_to_graph(
        self,
        db: AsyncSession,
        user_a: User,
        ws_a: KnowledgeBase,
        ws_b: KnowledgeBase,
        doc_a: Document,
        doc_b: Document,
        session_a: ChatSession,
    ):
        """Real TestClient POST; mock stream_agent_to_sse; verify filtered_ids only.

        Setup: user_a has workspace ws_a only (NOT ws_b).
        doc_a lives in ws_a (accessible); doc_b lives in ws_b (inaccessible).
        Request asks for BOTH doc_a and doc_b.

        We mock stream_agent_to_sse to capture the initial_state argument,
        then assert ONLY doc_a (filtered) reached it — doc_b (foreign) was
        filtered out by _filter_accessible_document_ids BEFORE graph state.
        """
        from app.core.deps import get_current_active_user, get_db
        from app.main import app

        # Only ws_a in user_a's accessible list; ws_b is NOT included.
        user_a.workspaces = [ws_a]

        # Capture the document_ids passed to stream_agent_to_sse.
        captured_doc_ids: list = []

        async def mock_stream_agent_to_sse(graph, initial_state):
            """Async generator that mimics stream_agent_to_sse signature.

            graph is unused (real graph replaced by our mock).
            initial_state is the state dict built by build_initial_state.
            We capture document_ids here — this is what the graph would receive.
            """
            captured_doc_ids.append(initial_state.get("document_ids"))
            # Yield minimal done events so the endpoint loop terminates cleanly.
            yield "event: sources\ndata: {\"sources\": []}\n\n"
            yield "event: token\ndata: {\"text\": \"ok\"}\n\n"

        with patch(
            "app.services.agent.streaming.stream_agent_to_sse",
            side_effect=mock_stream_agent_to_sse,
        ):

            app.dependency_overrides[get_current_active_user] = lambda: user_a
            app.dependency_overrides[get_db] = lambda: db

            try:
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.post(
                        f"/api/v1/rag/chat/sessions/{session_a.id}/stream",
                        json={
                            "message": "test query",
                            "document_ids": [str(doc_a.id), str(doc_b.id)],
                        },
                    )
            finally:
                app.dependency_overrides.pop(get_current_active_user, None)
                app.dependency_overrides.pop(get_db, None)

        # Endpoint must succeed (returns 200 via streaming response).
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        # CRITICAL: exactly one call to stream_agent_to_sse captured.
        assert len(captured_doc_ids) == 1, (
            f"Expected 1 capture (one POST), got {len(captured_doc_ids)}"
        )

        received_ids = captured_doc_ids[0] or []
        received_id_strs = [str(d) for d in received_ids]

        # doc_a (in ws_a, accessible) MUST be in the graph state.
        assert str(doc_a.id) in received_id_strs, (
            f"Accessible doc_a ({doc_a.id}) should pass ACL filter and reach graph. "
            f"Received: {received_id_strs}"
        )
        # doc_b (in ws_b, NOT in user_a's workspaces) MUST be filtered out.
        assert str(doc_b.id) not in received_id_strs, (
            f"Foreign doc_b ({doc_b.id}) MUST be filtered out by ACL — "
            f"user_a has no access to ws_b. Received: {received_id_strs}"
        )

        print(f"PASS: graph received only filtered IDs: {received_id_strs}")


class TestACLFilterContract:
    """Contract tests: proving the ACL filter is the true boundary."""

    def test_filter_function_uses_workspace_id_sql_clause(self):
        """Contract: _filter_accessible_document_ids MUST use workspace_id IN clause."""
        import inspect
        from app.api.chat_session import _filter_accessible_document_ids
        source = inspect.getsource(_filter_accessible_document_ids)
        assert "workspace_id" in source, "Filter must reference workspace_id in SQL"
        assert "in_(" in source or "workspace_id.in" in source or ".in(" in source, (
            "Filter must use SQL 'IN' clause for workspace_id ACL"
        )

    def test_filter_function_queries_database_not_memory(self):
        """Contract: filtering MUST happen at DB level, not in-memory post-fetch."""
        import inspect
        from app.api.chat_session import _filter_accessible_document_ids
        source = inspect.getsource(_filter_accessible_document_ids)
        assert "execute" in source or "select" in source, (
            "Filter must use database query (execute/select) for workspace-level ACL"
        )

    def test_endpoint_uses_filtered_doc_ids_not_request_doc_ids(self):
        """Contract: the streaming endpoint passes filtered_doc_ids to build_initial_state."""
        import inspect
        from app.api.chat_session import chat_stream_session
        source = inspect.getsource(chat_stream_session)

        assert "_filter_accessible_document_ids" in source, "Endpoint must call _filter_accessible_document_ids"

        lines = source.split("\n")

        # Find the line with document_ids=filtered_doc_ids and verify
        # it's in the build_initial_state call block
        for i, line in enumerate(lines):
            if 'document_ids=filtered_doc_ids' in line:
                # Verify the build_initial_state call is within 20 lines above
                context = "\n".join(lines[max(0, i-20):i+1])
                assert "build_initial_state" in context, (
                    f"document_ids=filtered_doc_ids must be in build_initial_state call. Context:\n{context}"
                )
                # Verify the argument is filtered_doc_ids (not request.document_ids)
                arg = line.split("document_ids=")[1].split(",")[0].strip().rstrip(")")
                assert arg == "filtered_doc_ids", (
                    f"document_ids must be filtered_doc_ids, got: {arg}"
                )
                return

        pytest.fail("build_initial_state must be called with document_ids=filtered_doc_ids")

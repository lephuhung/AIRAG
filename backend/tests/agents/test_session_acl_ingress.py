"""
Phase 0 / B6 — session ACL ingress filtering test.

Per F.5/O71: chat_session.py MUST filter document_ids against accessible
workspaces BEFORE passing to graph state. This test verifies the filtering
is correctly applied using mocking to avoid DB complexity.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from app.api.chat_session import _filter_accessible_document_ids


class MockDocument:
    """Mock document with id attribute."""
    def __init__(self, doc_id: uuid.UUID):
        self.id = doc_id


@pytest.mark.asyncio
async def test_filter_accessible_document_ids_returns_requested_when_in_workspace():
    """When doc is in user's workspace, it should be returned."""
    mock_db = MagicMock()
    mock_user = MagicMock()
    mock_user.id = uuid.uuid4()
    
    workspace_ids = [uuid.uuid4()]
    doc_id = uuid.uuid4()
    requested = [doc_id]
    
    # Mock the SQL query to return the doc
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [MockDocument(doc_id)]
    mock_db.execute = AsyncMock(return_value=mock_result)
    
    filtered = await _filter_accessible_document_ids(mock_db, mock_user, workspace_ids, requested)
    
    assert doc_id in filtered


@pytest.mark.asyncio
async def test_filter_accessible_document_ids_excludes_doc_not_in_workspace():
    """When doc is NOT in user's workspace, it should be filtered out."""
    mock_db = MagicMock()
    mock_user = MagicMock()
    mock_user.id = uuid.uuid4()
    
    workspace_ids = [uuid.uuid4()]  # User only has this workspace
    foreign_doc_id = uuid.uuid4()  # Doc in different workspace
    requested = [foreign_doc_id]
    
    # Mock the SQL query to return empty (doc not found in workspace)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)
    
    filtered = await _filter_accessible_document_ids(mock_db, mock_user, workspace_ids, requested)
    
    assert foreign_doc_id not in filtered


@pytest.mark.asyncio
async def test_filter_accessible_document_ids_handles_empty_request():
    """Empty requested list returns empty list."""
    mock_db = MagicMock()
    mock_user = MagicMock()
    mock_user.id = uuid.uuid4()
    
    workspace_ids = [uuid.uuid4()]
    requested: list[uuid.UUID] = []
    
    filtered = await _filter_accessible_document_ids(mock_db, mock_user, workspace_ids, requested)
    
    assert filtered == []


@pytest.mark.asyncio
async def test_filter_accessible_document_ids_handles_none_request():
    """None requested list returns empty list."""
    mock_db = MagicMock()
    mock_user = MagicMock()
    mock_user.id = uuid.uuid4()
    
    workspace_ids = [uuid.uuid4()]
    
    filtered = await _filter_accessible_document_ids(mock_db, mock_user, workspace_ids, None)
    
    assert filtered == []


@pytest.mark.asyncio
async def test_filter_accessible_document_ids_partial_access():
    """Some docs accessible, some not → only accessible docs returned."""
    mock_db = MagicMock()
    mock_user = MagicMock()
    mock_user.id = uuid.uuid4()
    
    accessible_workspace = uuid.uuid4()
    workspace_ids = [accessible_workspace]
    
    accessible_doc = uuid.uuid4()
    inaccessible_doc = uuid.uuid4()
    requested = [accessible_doc, inaccessible_doc]
    
    # Mock the SQL query to return only the accessible doc
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [MockDocument(accessible_doc)]
    mock_db.execute = AsyncMock(return_value=mock_result)
    
    filtered = await _filter_accessible_document_ids(mock_db, mock_user, workspace_ids, requested)
    
    assert accessible_doc in filtered
    assert inaccessible_doc not in filtered

"""
Phase 0 / B6 — markdown fallback ACL test.

Per F.5/O71: rag_agent.py markdown fallback block MUST add workspace_id predicate
to Document query — defense-in-depth beyond chat_session ingress filter.

This test verifies the workspace predicate is applied in the markdown fallback path.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_markdown_fallback_includes_workspace_predicate():
    """Verify the markdown fallback query includes workspace_id filter."""
    # Mock the function parameters
    section_reference = "Chương II"
    workspace_ids = [uuid.uuid4()]
    document_ids = [uuid.uuid4()]
    
    # Create mock objects
    mock_db = MagicMock()
    mock_storage = MagicMock()
    mock_storage.download_markdown = AsyncMock(return_value="# Chương II\n\nContent here")
    
    # Mock the Document query result
    mock_doc = MagicMock()
    mock_doc.id = document_ids[0]
    mock_doc.markdown_s3_key = "docs/test.md"
    mock_doc.original_filename = "test.md"
    
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_doc
    mock_db.execute = AsyncMock(return_value=mock_result)
    
    # Mock get_current_db at the streaming module
    with patch("app.services.agent.streaming.get_current_db", return_value=mock_db):
        with patch("app.services.storage_service.get_storage_service", return_value=mock_storage):
            with patch("app.services.agents.rag_agent._extract_section_from_markdown", return_value="Content here"):
                from app.services.agents.rag_agent import _execute_search_section
                
                result = await _execute_search_section(section_reference, workspace_ids, document_ids)
    
    # Verify the query was executed with workspace predicate
    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args
    
    # Check that the query includes workspace_id predicate
    query_str = str(call_args[0][0])
    assert "workspace_id" in query_str or ".in_(workspace_ids)" in query_str
    
    # Verify document was found (not None)
    assert mock_result.scalar_one_or_none.return_value is not None


@pytest.mark.asyncio
async def test_markdown_fallback_rejects_doc_in_foreign_workspace():
    """Doc in workspace A; user has workspace B only → doc not found."""
    section_reference = "Chương II"
    workspace_ids = [uuid.uuid4()]  # User only has workspace B
    foreign_doc_id = uuid.uuid4()  # Doc in workspace A
    
    mock_db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None  # Doc not found in user's workspace
    mock_db.execute = AsyncMock(return_value=mock_result)
    
    with patch("app.services.agent.streaming.get_current_db", return_value=mock_db):
        from app.services.agents.rag_agent import _execute_search_section
        
        result = await _execute_search_section(section_reference, workspace_ids, [foreign_doc_id])
    
    # Verify the result indicates document not found
    assert "Không tìm thấy" in result["text"]
    assert result["sources"] == []


@pytest.mark.asyncio
async def test_markdown_fallback_allows_doc_in_authorized_workspace():
    """Doc in workspace A; user has workspace A → doc is found."""
    section_reference = "Chương II"
    workspace_ids = [uuid.uuid4()]  # User has workspace A
    doc_id = uuid.uuid4()
    
    mock_db = MagicMock()
    mock_storage = MagicMock()
    mock_storage.download_markdown = AsyncMock(return_value="# Chương II\n\nContent here")
    
    # Mock the Document query result
    mock_doc = MagicMock()
    mock_doc.id = doc_id
    mock_doc.markdown_s3_key = "docs/test.md"
    mock_doc.original_filename = "test.md"
    
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_doc
    mock_db.execute = AsyncMock(return_value=mock_result)
    
    with patch("app.services.agent.streaming.get_current_db", return_value=mock_db):
        with patch("app.services.storage_service.get_storage_service", return_value=mock_storage):
            with patch("app.services.agents.rag_agent._extract_section_from_markdown", return_value="Content here"):
                from app.services.agents.rag_agent import _execute_search_section
                
                result = await _execute_search_section(section_reference, workspace_ids, [doc_id])
    
    # Verify the query was executed
    mock_db.execute.assert_called_once()
    
    # Verify result contains extracted content
    assert "Content here" in result["text"]
    assert len(result["sources"]) > 0

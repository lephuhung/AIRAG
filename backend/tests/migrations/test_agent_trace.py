"""
Tests for agent_traces routing_trace + preprocessor_marker columns.

Per spec Section B.11 0.3: JSONB routing_trace + String(32) preprocessor_marker.
Both nullable; backward-compatible with existing query_complexity scalar.
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


@pytest.mark.asyncio
async def test_routing_trace_column_exists_jsonb(db: AsyncSession):
    """routing_trace column must exist as JSONB."""
    from sqlalchemy import text
    result = await db.execute(
        text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'agent_traces' AND column_name = 'routing_trace'"
        )
    )
    row = result.fetchone()
    assert row is not None, "routing_trace column not found"
    assert row[1] == "jsonb", f"Expected jsonb, got {row[1]}"


@pytest.mark.asyncio
async def test_preprocessor_marker_column_exists(db: AsyncSession):
    """preprocessor_marker column must exist as VARCHAR(32)."""
    from sqlalchemy import text
    result = await db.execute(
        text(
            "SELECT column_name, data_type, character_maximum_length FROM information_schema.columns "
            "WHERE table_name = 'agent_traces' AND column_name = 'preprocessor_marker'"
        )
    )
    row = result.fetchone()
    assert row is not None, "preprocessor_marker column not found"
    assert row[1] == "character varying", f"Expected varchar, got {row[1]}"
    assert row[2] == 32, f"Expected VARCHAR(32), got {row[2]}"


@pytest.mark.asyncio
async def test_routing_trace_writes_canonical_dict(db: AsyncSession):
    """routing_trace stores canonical RoutingDecision fields."""
    from app.models.agent_trace import AgentTrace

    trace = AgentTrace(
        backend="langgraph",
        channel="web",
        original_query="So sánh Nghị định 13 với Nghị định 24",
        success=True,
        routing_trace={
            "execution_mode": "deepagent",
            "reason_code": "multi_target_compare",
            "config_revision": "abc123",
            "run_id": str(uuid.uuid4()),
        },
        preprocessor_marker="semantic_v1",
    )
    db.add(trace)
    await db.commit()

    from sqlalchemy import select
    result_row = await db.execute(
        select(AgentTrace).where(AgentTrace.backend == "langgraph")
    )
    reloaded = result_row.scalars().all()[-1]
    assert reloaded.routing_trace is not None
    assert reloaded.routing_trace["execution_mode"] == "deepagent"
    assert reloaded.preprocessor_marker == "semantic_v1"


@pytest.mark.asyncio
async def test_both_columns_nullable(db: AsyncSession):
    """Both routing_trace and preprocessor_marker must be nullable."""
    from app.models.agent_trace import AgentTrace

    trace = AgentTrace(
        backend="langgraph",
        channel="web",
        original_query="Test query",
        success=True,
        routing_trace=None,
        preprocessor_marker=None,
    )
    db.add(trace)
    await db.commit()

    from sqlalchemy import select
    result_row = await db.execute(select(AgentTrace))
    reloaded = result_row.scalars().all()[-1]
    assert reloaded.routing_trace is None
    assert reloaded.preprocessor_marker is None

"""
AgentTrace model — one row per agent run, capturing the full routing /
LLM / tool-call trace for building distillation datasets for a smaller model.

Written best-effort by ``AgentTraceService.record`` (own session, never raises)
from the single run chokepoint ``stream_agent_events``. PII is redacted before
persistence. Full system prompts are NOT stored (they live in app/prompts/);
only a hash reference per LLM call is kept so the DB stays lean.

Gated by ``settings.NEXUSRAG_TRACE_DATASET`` (default True).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, DateTime, Integer, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # ── Filterable columns (indexed) ─────────────────────────────────────────
    backend: Mapped[str] = mapped_column(String(16), default="langgraph", index=True)
    channel: Mapped[str] = mapped_column(String(16), default="web", index=True)  # web | telegram
    intent: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)
    next_agent: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    query_complexity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Who/where (snapshot — no FK so traces survive row deletion)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # ── Payload ──────────────────────────────────────────────────────────────
    original_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Ordered steps: routing / llm_call / tool_call (see TraceCollector)
    steps: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Run context: workspace_ids, document_ids, history_len, redacted flag, …
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Aggregated token usage across all LLM calls in the run
    token_usage: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )

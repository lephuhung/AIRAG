"""
AgentTraceService — persists a finished :class:`TraceCollector` as one
``agent_traces`` row for building distillation datasets.

Best-effort and isolated in its own DB session (same contract as AuditService):
a failure to record a trace never breaks — or rolls back — the chat response.
PII is redacted here, centrally, so no caller can bypass it.
"""

from __future__ import annotations

import logging

from app.core.database import async_session_maker
from app.models.agent_trace import AgentTrace
from app.services.agent.trace_redact import redact_obj, redact_text

logger = logging.getLogger(__name__)


class AgentTraceService:
    @staticmethod
    async def record(collector) -> None:
        """Persist one agent trace. Never raises — failures are logged only."""
        if collector is None:
            return
        try:
            from app.core.config import settings

            if not getattr(settings, "NEXUSRAG_TRACE_DATASET", True):
                return
        except Exception:
            pass

        try:
            steps = redact_obj(list(getattr(collector, "steps", []) or []))
            meta = redact_obj(dict(getattr(collector, "meta", {}) or {}))
            meta["redacted"] = True

            user_id = None
            session_id = getattr(collector, "meta", {}).get("session_id") if collector.meta else None
            raw_uid = collector.meta.get("user_id") if collector.meta else None
            if raw_uid:
                import uuid as _uuid

                try:
                    user_id = _uuid.UUID(str(raw_uid))
                except (ValueError, TypeError):
                    user_id = None

            async with async_session_maker() as db:
                db.add(
                    AgentTrace(
                        id=collector.trace_id,
                        backend=collector.backend,
                        channel=collector.channel,
                        intent=collector.intent,
                        next_agent=collector.next_agent,
                        query_complexity=collector.query_complexity,
                        success=bool(collector.success),
                        user_id=user_id,
                        session_id=str(session_id)[:64] if session_id else None,
                        original_query=redact_text(collector.original_query),
                        final_answer=redact_text(collector.final_answer),
                        error=collector.error,
                        steps=steps,
                        meta=meta,
                        token_usage=collector.token_usage or None,
                        latency_ms=collector.latency_ms,
                    )
                )
                await db.commit()
        except Exception as e:  # pragma: no cover - best effort
            logger.warning(f"[agent_trace] failed to record trace: {e}")

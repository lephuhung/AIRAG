"""
ConversationSummaryService — generates per-exchange summaries for conversation context.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.models.exchange_summary import ExchangeSummary
from app.services.llm import get_llm_provider
from app.services.llm.types import LLMMessage, LLMResult

logger = logging.getLogger(__name__)


class ConversationSummaryService:
    """Generates and manages per-exchange summaries to reduce context payload."""

    # Prompt template for exchange summarization
    SUMMARIZER_PROMPT = """Bạn là assistant chuyên tóm tắt đoạn hội thoại Q&A.

TÓM TẮT đoạn exchange sau thành JSON:
- **topic_label** (tiếng Việt, 3-8 từ): Chủ đề chính của câu hỏi
- **key_entities** (list 3-5 items): Các entities quan trọng (con số, tên riêng, khái niệm)
- **summary** (2-4 câu tiếng Việt): Tóm tắt nội dung CHÍNH, KHÔNG bịa đặt

EXCHANGE:
User: {user_message}

Assistant: {assistant_message}

Trả lời JSON format, không giải thích thêm:
{{"topic_label": "...", "key_entities": ["...", "..."], "summary": "..."}}"""

    async def generate_exchange_summary(
        self,
        user_message: str,
        assistant_message: str,
    ) -> dict | None:
        """Generate a summary for a single Q&A exchange using LLM."""
        try:
            provider = get_llm_provider()

            user_part = f"User: {user_message}"
            assistant_part = f"Assistant: {assistant_message}" if assistant_message else "Assistant: (pending)"

            prompt = self.SUMMARIZER_PROMPT.format(
                user_message=user_part,
                assistant_message=assistant_part,
            )

            messages = [LLMMessage(role="user", content=prompt)]

            result = await provider.acomplete(
                messages,
                temperature=0.1,
                max_tokens=512,
                system_prompt=None,
                think=False,
            )

            if isinstance(result, LLMResult):
                content = result.content
            else:
                content = str(result)

            # Parse JSON from response
            # Try to extract JSON from markdown code blocks first
            json_str = content
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()

            parsed = json.loads(json_str)
            return {
                "topic_label": parsed.get("topic_label", "Unknown"),
                "key_entities": parsed.get("key_entities", []),
                "summary": parsed.get("summary", ""),
            }

        except Exception as e:
            logger.error(f"Failed to generate exchange summary: {e}")
            return None

    def _format_sources_for_summary(self, sources: list) -> list[dict]:
        """Format sources into a simplified list of document references."""
        if not sources:
            return []

        # Deduplicate by document_id + page_no
        seen = set()
        formatted = []
        for src in sources:
            doc_id = src.get("document_id") or (src.document_id if hasattr(src, "document_id") else None)
            page = src.get("page_no") or (src.page_no if hasattr(src, "page_no") else 0)
            heading = src.get("heading_path") or (src.heading_path if hasattr(src, "heading_path") else [])

            if doc_id is None:
                continue

            key = (doc_id, page)
            if key not in seen:
                seen.add(key)
                heading_str = " > ".join(heading) if heading else ""
                formatted.append({
                    "doc_id": doc_id,
                    "page": page,
                    "heading": heading_str,
                })
        return formatted

    async def save_exchange_summary(
        self,
        db: AsyncSession,
        session_id: str,
        user_message_id: str,
        assistant_message_id: str | None,
        user_message: str,
        assistant_message: str,
        cited_sources: list | None = None,
    ) -> ExchangeSummary | None:
        """Generate and save a summary for a Q&A exchange."""
        # Calculate next exchange_index
        result = await db.execute(
            select(func.max(ExchangeSummary.exchange_index)).where(
                ExchangeSummary.session_id == session_id
            )
        )
        max_index = result.scalar() or 0
        next_index = max_index + 1

        # Generate summary
        summary_data = await self.generate_exchange_summary(
            user_message=user_message,
            assistant_message=assistant_message or "",
        )

        if not summary_data:
            return None

        # Format cited sources
        formatted_sources = self._format_sources_for_summary(cited_sources or [])

        exchange_summary = ExchangeSummary(
            session_id=session_id,
            exchange_index=next_index,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            topic_label=summary_data["topic_label"],
            key_entities=summary_data.get("key_entities"),
            summary=summary_data["summary"],
            cited_sources=formatted_sources if formatted_sources else None,
        )

        db.add(exchange_summary)
        await db.commit()
        await db.refresh(exchange_summary)

        logger.info(f"Saved exchange summary #{next_index} for session {session_id}: {summary_data['topic_label']}")
        return exchange_summary

    async def get_context_for_session(
        self,
        db: AsyncSession,
        session_id: str,
        limit: int = 10,
    ) -> str:
        """Build conversation context string from exchange summaries.

        Returns a formatted string suitable for insertion into system prompt.
        """
        result = await db.execute(
            select(ExchangeSummary)
            .where(ExchangeSummary.session_id == session_id)
            .order_by(ExchangeSummary.exchange_index.desc())
            .limit(limit)
        )
        summaries = result.scalars().all()

        if not summaries:
            return ""

        # Reverse to get chronological order
        summaries = list(reversed(summaries))

        context_parts = []
        for s in summaries:
            entities_str = ", ".join(s.key_entities) if s.key_entities else "N/A"

            # Format cited sources
            sources_info = ""
            if s.cited_sources:
                src_parts = []
                for src in s.cited_sources:
                    heading = src.get("heading", "")
                    page = src.get("page", 0)
                    doc_id = src.get("doc_id", "?")
                    if heading:
                        src_parts.append(f"Doc#{doc_id} (trang {page}, '{heading}')")
                    else:
                        src_parts.append(f"Doc#{doc_id} (trang {page})")
                if src_parts:
                    sources_info = f"\n  Referenced: {'; '.join(src_parts)}"

            context_parts.append(
                f"[Exchange {s.exchange_index}] Topic: {s.topic_label}"
                f"{sources_info}\n"
                f"  Entities: {entities_str}\n"
                f"  Summary: {s.summary}"
            )

        return "\n\n".join(context_parts)

    async def get_recent_exchanges(
        self,
        db: AsyncSession,
        session_id: str,
        limit: int = 5,
    ) -> list[ExchangeSummary]:
        """Get the N most recent exchange summaries."""
        result = await db.execute(
            select(ExchangeSummary)
            .where(ExchangeSummary.session_id == session_id)
            .order_by(ExchangeSummary.exchange_index.desc())
            .limit(limit)
        )
        summaries = result.scalars().all()
        return list(reversed(summaries))


# Singleton instance
_conversation_summary_service: ConversationSummaryService | None = None


def get_conversation_summary_service() -> ConversationSummaryService:
    global _conversation_summary_service
    if _conversation_summary_service is None:
        _conversation_summary_service = ConversationSummaryService()
    return _conversation_summary_service

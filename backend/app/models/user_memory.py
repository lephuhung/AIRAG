"""
UserMemory model — persistent long-term memory for each user.

Stores facts, preferences, and instructions that the AI learns during
conversations.  Uses pgvector for semantic similarity search so the LLM
can efficiently recall relevant memories even across sessions.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, String, Integer, SmallInteger, Text, DateTime, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

# pgvector integration
try:
    from pgvector.sqlalchemy import Vector
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False


# Embedding dimension — must match KG_EMBEDDING_DIMENSION in config
MEMORY_EMBEDDING_DIM = 1024


class UserMemory(Base):
    __tablename__ = "user_memories"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Privacy: each memory belongs to exactly one user
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # The actual memory text, e.g. "User prefers answers in table format"
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Semantic embedding for vector similarity search (pgvector)
    if _HAS_PGVECTOR:
        embedding = mapped_column(Vector(MEMORY_EMBEDDING_DIM), nullable=True)
    else:
        # Fallback: store as JSON array when pgvector is not installed
        embedding = mapped_column(JSON, nullable=True)

    # Classification: preference | fact | instruction | past_decision
    category: Mapped[str] = mapped_column(String(50), default="fact")

    # Priority score 1-10 (higher = more important to recall)
    importance: Mapped[int] = mapped_column(SmallInteger, default=5)

    # Which chat session the AI learned this from (for auditing)
    source_session_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_user_memories_category", "user_id", "category"),
    )

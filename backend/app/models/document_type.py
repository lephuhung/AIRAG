"""
DocumentType & DocumentTypeSystemPrompt models
===============================================

document_types              — danh mục loại văn bản (seed từ classifier)
document_type_system_prompts — system prompt riêng cho từng loại văn bản
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DocumentType(Base):
    """
    Loại văn bản (thông tư, nghị định, công văn, ...).
    Được seed tự động từ document_type_classifier.get_all_document_types()
    khi startup. Người dùng có thể thêm loại mới qua API.
    """
    __tablename__ = "document_types"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    system_prompts: Mapped[list["DocumentTypeSystemPrompt"]] = relationship(
        back_populates="document_type", cascade="all, delete-orphan"
    )
    documents: Mapped[list["Document"]] = relationship(  # type: ignore[name-defined]
        back_populates="document_type"
    )


class DocumentTypeSystemPrompt(Base):
    """
    System prompt riêng cho từng loại văn bản trong một workspace.

    Nếu không có bản ghi cho (workspace_id, document_type_id),
    hệ thống dùng system prompt mặc định của workspace (hoặc DEFAULT_SYSTEM_PROMPT).

    workspace_id = NULL → system prompt mặc định toàn cục cho loại văn bản đó.
    workspace_id = <id> → ghi đè riêng cho workspace đó.
    """
    __tablename__ = "document_type_system_prompts"
    __table_args__ = (
        UniqueConstraint(
            "document_type_id", "workspace_id",
            name="uq_doctype_workspace_prompt",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_type_id: Mapped[int] = mapped_column(
        ForeignKey("document_types.id", ondelete="CASCADE"), nullable=False
    )
    # NULL = global default; non-NULL = workspace-specific override
    workspace_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=True
    )
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    # Custom KG extraction prompt for this document type (NULL = use default LEGAL_KG_SYSTEM_PROMPT)
    kg_system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    document_type: Mapped["DocumentType"] = relationship(back_populates="system_prompts")

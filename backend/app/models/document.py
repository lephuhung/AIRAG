from sqlalchemy import String, ForeignKey, DateTime, Integer, Text, Enum, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
import enum

from app.core.database import Base


class DocumentStatus(str, enum.Enum):
    PENDING      = "pending"        # uploaded, waiting for worker
    PARSING      = "parsing"        # Docling on native PDF/DOCX/PPTX
    OCRING       = "ocring"         # OCR on scanned PDFs
    CHUNKING     = "chunking"       # parse done, sub-tasks dispatched
    EMBEDDING    = "embedding"      # embed_worker running
    BUILDING_KG  = "building_kg"    # embed+captions done, KG still running
    INDEXED      = "indexed"        # all done
    FAILED       = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(50))
    file_size: Mapped[int] = mapped_column(Integer)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, values_callable=lambda enum_cls: [m.value for m in enum_cls]),
        default=DocumentStatus.PENDING,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # HRAG fields
    markdown_s3_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    upload_s3_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    parser_version: Mapped[str | None] = mapped_column(String(50), nullable=True)  # "docling" | "legacy"
    processing_time_ms: Mapped[int] = mapped_column(Integer, default=0)

    # Sub-task completion flags (set independently by each worker)
    embed_done:    Mapped[bool] = mapped_column(default=False)
    captions_done: Mapped[bool] = mapped_column(default=False)
    kg_done:       Mapped[bool] = mapped_column(default=False)

    # Raw chunks JSON stored by parse_worker, consumed by embed_worker
    # Cleared after embed_worker finishes to save space
    raw_chunks_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Digital signature metadata extracted from native PDFs (list of dicts)
    # Each element: {field_name, page, signer_name, organization, email,
    #                issuer, valid_from, valid_until, signing_time, reason, location}
    digital_signatures: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Document type classification (auto-detected by classifier)
    document_type_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_types.id", ondelete="SET NULL"), nullable=True
    )
    # Official document reference number extracted by classifier (e.g. "13/2023/NĐ-CP")
    document_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Document title/subject extracted from header (e.g. "Luật Bảo vệ Bí mật nhà nước")
    document_title: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Rich Header Metadata extracted by LLM from Page 1 OCR
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    issuing_agency: Mapped[str | None] = mapped_column(String(255), nullable=True)
    parent_agency: Mapped[str | None] = mapped_column(String(255), nullable=True)
    published_date: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    # Manual signer name override
    signer_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Root Document node entity_id in Neo4j (used for KG metadata updates)
    kg_root_entity_id: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # User who uploaded this document
    uploaded_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )

    # Relationships
    workspace: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
    document_type: Mapped["DocumentType | None"] = relationship(  # type: ignore[name-defined]
        back_populates="documents", lazy="selectin"
    )
    images: Mapped[list["DocumentImage"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    tables: Mapped[list["DocumentTable"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentImage(Base):
    __tablename__ = "document_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    image_id: Mapped[str] = mapped_column(String(100), unique=True)  # UUID
    page_no: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str] = mapped_column(String(500))
    caption: Mapped[str] = mapped_column(Text, default="")
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    mime_type: Mapped[str] = mapped_column(String(50), default="image/png")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="images")


class DocumentTable(Base):
    __tablename__ = "document_tables"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    table_id: Mapped[str] = mapped_column(String(100), unique=True)
    page_no: Mapped[int] = mapped_column(Integer, default=0)
    content_markdown: Mapped[str] = mapped_column(Text, default="")
    caption: Mapped[str] = mapped_column(Text, default="")
    num_rows: Mapped[int] = mapped_column(Integer, default=0)
    num_cols: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="tables")

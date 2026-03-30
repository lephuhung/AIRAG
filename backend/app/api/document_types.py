"""
Document Types API
==================
CRUD endpoints cho loại văn bản và system prompt tương ứng.

Endpoints:
  GET    /document-types                          — list all types
  POST   /document-types                          — create new type
  GET    /document-types/{slug}                   — get by slug
  PUT    /document-types/{slug}                   — update name/description
  DELETE /document-types/{slug}                   — soft-delete (is_active=false)

  GET    /document-types/{slug}/prompt            — get global system prompt
  PUT    /document-types/{slug}/prompt            — set global system prompt
  GET    /document-types/{slug}/prompt/{ws_id}    — get workspace-specific prompt
  PUT    /document-types/{slug}/prompt/{ws_id}    — set workspace-specific prompt
  DELETE /document-types/{slug}/prompt/{ws_id}    — remove workspace override
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_active_user, require_superadmin
from app.models.document_type import DocumentType, DocumentTypeSystemPrompt
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document-types", tags=["document-types"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class DocumentTypeResponse(BaseModel):
    id: int
    slug: str
    name: str
    description: str | None
    is_active: bool

    model_config = {"from_attributes": True}


class DocumentTypeCreate(BaseModel):
    slug: str
    name: str
    description: str | None = None


class DocumentTypeUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_active: bool | None = None


class SystemPromptResponse(BaseModel):
    document_type_slug: str
    workspace_id: int | None
    system_prompt: str
    kg_system_prompt: str | None
    is_default: bool  # True nếu chưa có bản ghi riêng → dùng DEFAULT_SYSTEM_PROMPT


class SystemPromptSet(BaseModel):
    system_prompt: str
    kg_system_prompt: str | None = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _get_type_by_slug(slug: str, db: AsyncSession) -> DocumentType:
    result = await db.execute(
        select(DocumentType).where(DocumentType.slug == slug)
    )
    doc_type = result.scalar_one_or_none()
    if not doc_type:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document type '{slug}' not found",
        )
    return doc_type


async def _get_workspace(ws_id: int, db: AsyncSession) -> KnowledgeBase:
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == ws_id))
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {ws_id} not found",
        )
    return kb


# ---------------------------------------------------------------------------
# DocumentType CRUD
# ---------------------------------------------------------------------------

@router.get("", response_model=list[DocumentTypeResponse])
async def list_document_types(
    include_inactive: bool = False,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """List all document types."""
    q = select(DocumentType).order_by(DocumentType.name)
    if not include_inactive:
        q = q.where(DocumentType.is_active.is_(True))
    result = await db.execute(q)
    return result.scalars().all()


@router.post("", response_model=DocumentTypeResponse, status_code=status.HTTP_201_CREATED)
async def create_document_type(
    body: DocumentTypeCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Create a new document type. Auto-seeds default KG system prompt."""
    existing = await db.execute(
        select(DocumentType).where(DocumentType.slug == body.slug)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document type with slug '{body.slug}' already exists",
        )
    doc_type = DocumentType(
        slug=body.slug,
        name=body.name,
        description=body.description,
    )
    db.add(doc_type)
    await db.flush()  # get doc_type.id before creating related row

    # Auto-seed default KG system prompt (LEGAL_KG_SYSTEM_PROMPT)
    from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT
    from app.services.legal_kg_prompts import LEGAL_KG_SYSTEM_PROMPT

    prompt_row = DocumentTypeSystemPrompt(
        document_type_id=doc_type.id,
        workspace_id=None,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        kg_system_prompt=LEGAL_KG_SYSTEM_PROMPT,
    )
    db.add(prompt_row)
    await db.commit()
    await db.refresh(doc_type)
    return doc_type


@router.get("/{slug}", response_model=DocumentTypeResponse)
async def get_document_type(slug: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_active_user)):
    """Get a document type by slug."""
    return await _get_type_by_slug(slug, db)


@router.put("/{slug}", response_model=DocumentTypeResponse)
async def update_document_type(
    slug: str,
    body: DocumentTypeUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Update name, description or active status of a document type."""
    doc_type = await _get_type_by_slug(slug, db)
    if body.name is not None:
        doc_type.name = body.name
    if body.description is not None:
        doc_type.description = body.description
    if body.is_active is not None:
        doc_type.is_active = body.is_active
    await db.commit()
    await db.refresh(doc_type)
    return doc_type


@router.delete("/{slug}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_document_type(slug: str, db: AsyncSession = Depends(get_db), user: User = Depends(require_superadmin)):
    """Soft-delete a document type (sets is_active=False)."""
    doc_type = await _get_type_by_slug(slug, db)
    doc_type.is_active = False
    await db.commit()


# ---------------------------------------------------------------------------
# System Prompt endpoints
# ---------------------------------------------------------------------------

async def _resolve_system_prompt(
    doc_type: DocumentType,
    workspace_id: int | None,
    db: AsyncSession,
) -> tuple[str, bool]:
    """
    Return (system_prompt_text, is_default).
    Priority: workspace-specific → global (workspace_id=NULL) → DEFAULT_SYSTEM_PROMPT
    """
    from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT

    # Try workspace-specific first (if workspace_id given)
    if workspace_id is not None:
        res = await db.execute(
            select(DocumentTypeSystemPrompt).where(
                DocumentTypeSystemPrompt.document_type_id == doc_type.id,
                DocumentTypeSystemPrompt.workspace_id == workspace_id,
            )
        )
        row = res.scalar_one_or_none()
        if row:
            return row.system_prompt, False

    # Try global (workspace_id IS NULL)
    res = await db.execute(
        select(DocumentTypeSystemPrompt).where(
            DocumentTypeSystemPrompt.document_type_id == doc_type.id,
            DocumentTypeSystemPrompt.workspace_id.is_(None),
        )
    )
    row = res.scalar_one_or_none()
    if row:
        return row.system_prompt, False

    return DEFAULT_SYSTEM_PROMPT, True


@router.get("/{slug}/prompt", response_model=SystemPromptResponse)
async def get_global_system_prompt(slug: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_active_user)):
    """Get the global system prompt for a document type (workspace_id=NULL)."""
    doc_type = await _get_type_by_slug(slug, db)
    res = await db.execute(
        select(DocumentTypeSystemPrompt).where(
            DocumentTypeSystemPrompt.document_type_id == doc_type.id,
            DocumentTypeSystemPrompt.workspace_id.is_(None),
        )
    )
    row = res.scalar_one_or_none()
    if row:
        return SystemPromptResponse(
            document_type_slug=slug,
            workspace_id=None,
            system_prompt=row.system_prompt,
            kg_system_prompt=row.kg_system_prompt,
            is_default=False,
        )
    from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT
    return SystemPromptResponse(
        document_type_slug=slug,
        workspace_id=None,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        kg_system_prompt=None,
        is_default=True,
    )


@router.put("/{slug}/prompt", response_model=SystemPromptResponse)
async def set_global_system_prompt(
    slug: str,
    body: SystemPromptSet,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Set (upsert) the global system prompt for a document type."""
    doc_type = await _get_type_by_slug(slug, db)

    res = await db.execute(
        select(DocumentTypeSystemPrompt).where(
            DocumentTypeSystemPrompt.document_type_id == doc_type.id,
            DocumentTypeSystemPrompt.workspace_id.is_(None),
        )
    )
    row = res.scalar_one_or_none()
    if row:
        row.system_prompt = body.system_prompt
        if body.kg_system_prompt is not None:
            row.kg_system_prompt = body.kg_system_prompt
    else:
        row = DocumentTypeSystemPrompt(
            document_type_id=doc_type.id,
            workspace_id=None,
            system_prompt=body.system_prompt,
            kg_system_prompt=body.kg_system_prompt,
        )
        db.add(row)
    await db.commit()
    await db.refresh(row)
    return SystemPromptResponse(
        document_type_slug=slug,
        workspace_id=None,
        system_prompt=row.system_prompt,
        kg_system_prompt=row.kg_system_prompt,
        is_default=False,
    )


@router.get("/{slug}/prompt/{workspace_id}", response_model=SystemPromptResponse)
async def get_workspace_system_prompt(
    slug: str,
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    """Get the resolved system prompt for a document type + workspace combo."""
    doc_type = await _get_type_by_slug(slug, db)
    await _get_workspace(workspace_id, db)

    # Try workspace-specific first
    res = await db.execute(
        select(DocumentTypeSystemPrompt).where(
            DocumentTypeSystemPrompt.document_type_id == doc_type.id,
            DocumentTypeSystemPrompt.workspace_id == workspace_id,
        )
    )
    row = res.scalar_one_or_none()
    if row:
        return SystemPromptResponse(
            document_type_slug=slug,
            workspace_id=workspace_id,
            system_prompt=row.system_prompt,
            kg_system_prompt=row.kg_system_prompt,
            is_default=False,
        )

    # Fallback to global
    res = await db.execute(
        select(DocumentTypeSystemPrompt).where(
            DocumentTypeSystemPrompt.document_type_id == doc_type.id,
            DocumentTypeSystemPrompt.workspace_id.is_(None),
        )
    )
    row = res.scalar_one_or_none()
    if row:
        return SystemPromptResponse(
            document_type_slug=slug,
            workspace_id=workspace_id,
            system_prompt=row.system_prompt,
            kg_system_prompt=row.kg_system_prompt,
            is_default=False,
        )

    from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT
    return SystemPromptResponse(
        document_type_slug=slug,
        workspace_id=workspace_id,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        kg_system_prompt=None,
        is_default=True,
    )


@router.put("/{slug}/prompt/{workspace_id}", response_model=SystemPromptResponse)
async def set_workspace_system_prompt(
    slug: str,
    workspace_id: int,
    body: SystemPromptSet,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Set (upsert) a workspace-specific system prompt for a document type."""
    doc_type = await _get_type_by_slug(slug, db)
    await _get_workspace(workspace_id, db)

    res = await db.execute(
        select(DocumentTypeSystemPrompt).where(
            DocumentTypeSystemPrompt.document_type_id == doc_type.id,
            DocumentTypeSystemPrompt.workspace_id == workspace_id,
        )
    )
    row = res.scalar_one_or_none()
    if row:
        row.system_prompt = body.system_prompt
        if body.kg_system_prompt is not None:
            row.kg_system_prompt = body.kg_system_prompt
    else:
        row = DocumentTypeSystemPrompt(
            document_type_id=doc_type.id,
            workspace_id=workspace_id,
            system_prompt=body.system_prompt,
            kg_system_prompt=body.kg_system_prompt,
        )
        db.add(row)
    await db.commit()
    await db.refresh(row)
    return SystemPromptResponse(
        document_type_slug=slug,
        workspace_id=workspace_id,
        system_prompt=row.system_prompt,
        kg_system_prompt=row.kg_system_prompt,
        is_default=False,
    )


@router.delete("/{slug}/prompt/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace_system_prompt(
    slug: str,
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_superadmin),
):
    """Remove the workspace-specific override (falls back to global / default)."""
    doc_type = await _get_type_by_slug(slug, db)
    res = await db.execute(
        select(DocumentTypeSystemPrompt).where(
            DocumentTypeSystemPrompt.document_type_id == doc_type.id,
            DocumentTypeSystemPrompt.workspace_id == workspace_id,
        )
    )
    row = res.scalar_one_or_none()
    if row:
        await db.delete(row)
        await db.commit()

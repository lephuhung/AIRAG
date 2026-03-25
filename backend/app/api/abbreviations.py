from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_active_user, require_superadmin
from app.models.user import User
from app.schemas.abbreviation import (
    AbbreviationCreate,
    AbbreviationUpdate,
    AbbreviationResponse,
    AbbreviationListResponse
)
from app.services.abbreviation_service import AbbreviationService

router = APIRouter(prefix="/abbreviations", tags=["abbreviations"])


@router.post("/", response_model=AbbreviationResponse, status_code=status.HTTP_201_CREATED)
async def create_abbreviation(
    request: AbbreviationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new abbreviation suggestion. 
    If created by superadmin, it will be active by default.
    """
    is_active = current_user.is_superadmin
    return await AbbreviationService.create(db, request, current_user.id, is_active=is_active)


@router.get("/", response_model=AbbreviationListResponse)
async def list_abbreviations(
    search: str | None = Query(None),
    is_active: bool | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List all abbreviations with search and pagination."""
    skip = (page - 1) * per_page
    items, total = await AbbreviationService.get_multi(
        db, skip=skip, limit=per_page, search=search, is_active=is_active
    )
    return AbbreviationListResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page
    )


@router.get("/{abbreviation_id}", response_model=AbbreviationResponse)
async def get_abbreviation(
    abbreviation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific abbreviation."""
    db_obj = await AbbreviationService.get(db, abbreviation_id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Abbreviation not found")
    return db_obj


@router.patch("/{abbreviation_id}", response_model=AbbreviationResponse)
async def update_abbreviation(
    abbreviation_id: int,
    request: AbbreviationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update an abbreviation. 
    If updating 'is_active', requires superadmin or a specific role check if needed.
    """
    db_obj = await AbbreviationService.get(db, abbreviation_id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Abbreviation not found")
    
    # Simple rule: only superadmin or owner can update, BUT only superadmin can ACTIVATE
    # For now, let's keep it simple: Anyone can update their own, only superadmin can activate.
    
    if request.is_active is not None and not current_user.is_superadmin:
        raise HTTPException(
            status_code=403, 
            detail="Only superadmins can activate/deactivate abbreviations"
        )
    
    if not current_user.is_superadmin and db_obj.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
        
    return await AbbreviationService.update(db, db_obj, request)


@router.delete("/{abbreviation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_abbreviation(
    abbreviation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete an abbreviation."""
    db_obj = await AbbreviationService.get(db, abbreviation_id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Abbreviation not found")
        
    if not current_user.is_superadmin and db_obj.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
        
    await AbbreviationService.delete(db, abbreviation_id)
    return None

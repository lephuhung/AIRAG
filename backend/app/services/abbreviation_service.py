import re
import logging
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from app.models.abbreviation import Abbreviation
from app.schemas.abbreviation import AbbreviationCreate, AbbreviationUpdate

logger = logging.getLogger(__name__)


class AbbreviationService:
    @staticmethod
    async def create(db: AsyncSession, obj_in: AbbreviationCreate, user_id: int, is_active: bool = False) -> Abbreviation:
        db_obj = Abbreviation(
            short_form=obj_in.short_form,
            full_form=obj_in.full_form,
            description=obj_in.description,
            user_id=user_id,
            is_active=is_active
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    @staticmethod
    async def get_multi(
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100,
        search: str | None = None,
        is_active: bool | None = None
    ) -> tuple[List[Abbreviation], int]:
        from sqlalchemy import or_, func
        
        query = select(Abbreviation)
        count_query = select(func.count(Abbreviation.id))
        
        if search:
            search_filter = or_(
                Abbreviation.short_form.ilike(f"%{search}%"),
                Abbreviation.full_form.ilike(f"%{search}%")
            )
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)
            
        if is_active is not None:
            query = query.where(Abbreviation.is_active == is_active)
            count_query = count_query.where(Abbreviation.is_active == is_active)
            
        total = await db.scalar(count_query) or 0
        result = await db.execute(
            query.order_by(Abbreviation.updated_at.desc()).offset(skip).limit(limit)
        )
        return list(result.scalars().all()), total

    @staticmethod
    async def get_active(db: AsyncSession) -> List[Abbreviation]:
        result = await db.execute(select(Abbreviation).where(Abbreviation.is_active == True))
        return list(result.scalars().all())

    @staticmethod
    async def get(db: AsyncSession, id: int) -> Optional[Abbreviation]:
        result = await db.execute(select(Abbreviation).where(Abbreviation.id == id))
        return result.scalar_one_or_none()

    @staticmethod
    async def update(db: AsyncSession, db_obj: Abbreviation, obj_in: AbbreviationUpdate) -> Abbreviation:
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    @staticmethod
    async def delete(db: AsyncSession, id: int) -> None:
        await db.execute(delete(Abbreviation).where(Abbreviation.id == id))
        await db.commit()

    @staticmethod
    async def expand_ab_in_text(db: AsyncSession, text: str) -> str:
        """
        Expand all active abbreviations in the given text.
        Uses word-boundary regex to prevent partial matching.
        """
        if not text:
            return text

        active_abs = await AbbreviationService.get_active(db)
        if not active_abs:
            return text

        expanded_text = text
        # Sort by length of short_form descending to handle overlapping (e.g., "AI", "AIE")
        active_abs.sort(key=lambda x: len(x.short_form), reverse=True)

        for abb in active_abs:
            # Use regex with word boundaries (\b) for robust expansion
            pattern = rf"\b{re.escape(abb.short_form)}\b"
            # Case-insensitive replacement while keeping word boundaries
            expanded_text = re.sub(pattern, abb.full_form, expanded_text, flags=re.IGNORECASE)

        if expanded_text != text:
            logger.info(f"Expanded abbreviations: '{text}' -> '{expanded_text}'")

        return expanded_text

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class AbbreviationBase(BaseModel):
    short_form: str = Field(..., min_length=1, max_length=50)
    full_form: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class AbbreviationCreate(AbbreviationBase):
    pass


class AbbreviationUpdate(BaseModel):
    short_form: str | None = Field(default=None, min_length=1, max_length=50)
    full_form: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    is_active: bool | None = None


class AbbreviationResponse(AbbreviationBase):
    id: uuid.UUID
    user_id: uuid.UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AbbreviationListResponse(BaseModel):
    items: list[AbbreviationResponse]
    total: int
    page: int
    per_page: int

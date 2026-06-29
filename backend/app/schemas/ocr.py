"""
OCR request/response schemas.
"""
from __future__ import annotations

from pydantic import BaseModel


class OCRStatusResponse(BaseModel):
    enabled: bool


class OCRExtractResponse(BaseModel):
    text: str
    pages: int = 0

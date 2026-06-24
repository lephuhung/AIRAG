"""
STT request/response schemas.
"""
from __future__ import annotations

from pydantic import BaseModel


class STTTranscribeResponse(BaseModel):
    text: str
    language: str = ""
    duration: float = 0.0


class STTStatusResponse(BaseModel):
    enabled: bool
    provider: str

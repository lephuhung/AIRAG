"""
TTS request/response schemas.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TTSSynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str | None = None  # None → user setting → config default
    speed: float | None = Field(default=None, ge=0.5, le=2.0)
    pitch: float | None = Field(default=None, ge=0.5, le=2.0)


class TTSVoiceResponse(BaseModel):
    id: str
    label: str
    gender: str = ""
    language: str = ""


class TTSVoicesResponse(BaseModel):
    enabled: bool
    provider: str
    voices: list[TTSVoiceResponse]

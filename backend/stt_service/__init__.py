"""Standalone CPU microservice hosting faster-whisper STT over HTTP.

Runs the existing FasterWhisperSTTProvider in-process and exposes an
OpenAI-compatible POST /v1/audio/transcriptions endpoint, so the backend can set
STT_PROVIDER=openai + STT_OPENAI_BASE_URL and hold no Whisper model itself
(one shared CPU copy, backend scales on CPU). See docs/scaling.md.
"""

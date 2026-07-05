"""
Legal Knowledge Graph Prompts
==============================

Vietnamese legal/administrative document extraction prompts for LegalKGService.

This module is deprecated. All prompts have been moved to:
    app/prompts/legal_kg.py

Please update imports to use:
    from app.prompts.legal_kg import (...)

See: app/prompts/legal_kg.md for documentation
"""

# Re-export from new location for backward compatibility
from app.prompts.legal_kg import (
    LEGAL_KG_SYSTEM_PROMPT,
    LEGAL_KG_USER_PROMPT,
    PERSON_EXTRACT_SYSTEM_PROMPT,
    PERSON_EXTRACT_USER_PROMPT,
    PREAMBLE_SYSTEM_PROMPT,
    PREAMBLE_USER_PROMPT,
    ENTITY_RESOLVE_SYSTEM_PROMPT,
    ENTITY_RESOLVE_USER_PROMPT,
    PERSON_DOCUMENT_TRIGGERS,
)

__all__ = [
    "LEGAL_KG_SYSTEM_PROMPT",
    "LEGAL_KG_USER_PROMPT",
    "PERSON_EXTRACT_SYSTEM_PROMPT",
    "PERSON_EXTRACT_USER_PROMPT",
    "PREAMBLE_SYSTEM_PROMPT",
    "PREAMBLE_USER_PROMPT",
    "ENTITY_RESOLVE_SYSTEM_PROMPT",
    "ENTITY_RESOLVE_USER_PROMPT",
    "PERSON_DOCUMENT_TRIGGERS",
]
"""
HRAG Chat System Prompts
===========================
Two-part prompt architecture adapted from Perplexity AI for document Q&A.

This module is deprecated. All prompts have been moved to:
    app/prompts/chat.py

Please update imports to use:
    from app.prompts.chat import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT

See also: app/prompts/chat.md
"""

# Re-export from new location for backward compatibility
from app.prompts.chat import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT

__all__ = ["DEFAULT_SYSTEM_PROMPT", "HARD_SYSTEM_PROMPT"]
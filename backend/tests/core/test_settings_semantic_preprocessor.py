"""
Tests for NEXUSRAG_SEMANTIC_PREPROCESSOR flag.

Per spec Section B.11 0.5 (O7): atomic enable flag.
Default false; must be bool.
"""

from __future__ import annotations

import pytest


def test_semantic_preprocessor_default_false():
    """Default must be False (disabled by default for safe rollout)."""
    from app.core.config import Settings

    s = Settings()
    assert s.NEXUSRAG_SEMANTIC_PREPROCESSOR is False


def test_semantic_preprocessor_accepts_true():
    """Must accept explicit True."""
    from app.core.config import Settings

    s = Settings(NEXUSRAG_SEMANTIC_PREPROCESSOR=True)
    assert s.NEXUSRAG_SEMANTIC_PREPROCESSOR is True


def test_semantic_preprocessor_accepts_false():
    """Must accept explicit False."""
    from app.core.config import Settings

    s = Settings(NEXUSRAG_SEMANTIC_PREPROCESSOR=False)
    assert s.NEXUSRAG_SEMANTIC_PREPROCESSOR is False


def test_settings_singleton_has_flag():
    """Global settings singleton must expose the flag."""
    from app.core.config import settings

    assert hasattr(settings, "NEXUSRAG_SEMANTIC_PREPROCESSOR")
    # Should be bool
    assert isinstance(settings.NEXUSRAG_SEMANTIC_PREPROCESSOR, bool)

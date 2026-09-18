"""Checkpointed multi-intent semantic analysis (multi-intent routing spec §5, §33.3).

``IntentAnalysis`` is the flag-on semantic authority for routing: the
``MultiIntentClassifier`` produces it once per turn inside ``route_node`` and
it is checkpointed beside the route decision so a resumed run replays the
identical analysis instead of re-classifying. It replaces the runtime-only
``IntentDecision`` (single-intent, advisory) on the flag-on path only;
``IntentDecision`` stays for flag-off.

Semantics carried here are facts, never route authority:

- ``intent_id`` is server-assigned by position (``"i1"``, ``"i2"``, ...) —
  the model cannot mint it.
- ``confidence`` is metadata only (tracing/evaluation); it never feeds a
  routing decision (spec §20).
- ``depends_on`` indexes into ``intents`` (strictly earlier positions) and
  distinguishes the sequential dependent case from parallel independent
  intents. Index-based (not name-based) so two intents sharing a name stay
  unambiguous.
- ``requires_complex_execution`` is model advisory; the deterministic
  router recomputes it through ``semantic.intent_registry.needs_complex_execution``
  and never trusts the declared value.
- ``source`` is server-set by the path that produced the analysis
  (``"model"`` for the classifier, ``"deterministic"`` for the
  greeting/personal short-circuit); the model cannot mint it either.
"""
from __future__ import annotations

from typing import Literal

from .base import ContractModel


class DetectedIntent(ContractModel):
    """One semantic intent detected inside the user's full request."""

    intent_id: str
    """Server-assigned positional identifier (``"i1"``, ``"i2"``, ...)."""

    name: str
    """Semantic intent name; resolved against ``INTENT_REGISTRY`` at routing."""

    confidence: float | None = None
    """Metadata only (spec §20): never routing authority, range [0, 1]."""

    depends_on: tuple[int, ...] = ()
    """Indexes into ``IntentAnalysis.intents``, strictly earlier positions only."""

    description: str | None = None


class IntentAnalysis(ContractModel):
    """Checkpointed whole-request intent analysis (spec §5.2, §33.3).

    Route authority depends on ``len(intents)``, so resume must replay the
    identical analysis: this is a ``ContractModel`` persisted inside
    ``SupervisorV2State.intent_analysis`` (checkpoint schema revision 3).
    """

    primary_intent: str | None
    """Overall user goal (spec §6): advisory — never sole routing authority."""

    intents: tuple[DetectedIntent, ...]

    is_multi_intent: bool
    """Must equal ``len(intents) > 1`` exactly (validated)."""

    requires_complex_execution: bool
    """Model advisory only; the router recomputes through the registry."""

    semantic_summary: str | None = None

    source: Literal["model", "deterministic"]
    """Server-set provenance: model classification or the deterministic
    greeting/personal short-circuit (spec §33.1)."""

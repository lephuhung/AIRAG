"""Runtime-only typed v1 intent adapter/cache (Phase 4A, Task 2).

``IntentDecision`` is the minimal typed projection of v1 route/intent
intelligence into v2: what the query means (v1 taxonomy intent plus the two
memory/safety flags the v1 prompt already teaches). It is a ``RuntimeModel`` —
request-scoped, never checkpointed, advisory-only. V2 route authority stays
with the deterministic ``decide_route()`` policy (Task 3 consumes this).

Ownership rules enforced here:

- Deterministic narrow scopes (``greeting`` / ``personal`` / ``people``) reuse
  ``classify_supervisor_scope`` + ``deterministic_decision_for_scope``
  verbatim; no model call, ``source="deterministic"``.
- Everything else reuses the v1 full-taxonomy prompt behavior through the
  ``semantic_router`` role (one provider factory, thinking-inherited); only
  ``intent`` / ``needs_memory`` / ``is_legal_query`` are projected out.
  Legacy supervisor control fields (``next_agent`` / ``pending_intent`` /
  task plans) are never imported, returned, or stored.
- Results are cached per request/turn on the ``IntentClassifier`` instance
  (wired onto ``RuntimeServices.intent_classifier``), so repeated semantic
  draft builds do not repeat classification.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from ..contracts.base import RuntimeModel

__all__ = [
    "IntentDecision",
    "IntentClassifier",
    "IntentClassifierError",
    "classify_deterministic",
]

#: V1 taxonomy intents this adapter may project. Legacy control values
#: (``next_agent`` names such as ``direct`` / ``rag`` / ``people`` /
#: ``finish``) are deliberately absent: they are routing authority, not
#: semantic intent, and must never surface here.
_VALID_INTENTS = frozenset(
    {
        "greeting",
        "personal",
        "search",
        "list_docs",
        "summarize",
        "kg_query",
        "search_doc_num",
        "search_abbr",
        "search_section",
        "resolve_doc",
        "write_summarize",
        "write_suggest_edits",
        "write_grammar_check",
        "write_format_check",
        "mongo_search_cccd",
        "mongo_search_name",
        "mongo_search_bhxh",
        "mongo_search_phone",
        "mongo_search_advanced",
    }
)


class IntentDecision(RuntimeModel):
    """Typed v1 intent projection (runtime-only, advisory, never checkpointed)."""

    intent: str
    source: Literal["deterministic", "model"]
    confidence: float | None = None
    needs_memory: bool = False
    is_legal_query: bool = False


class IntentClassifierError(ValueError):
    """The query cannot be classified into a typed intent."""


def classify_deterministic(
    query: str, *, has_doc_ids: bool = False
) -> IntentDecision | None:
    """Narrow-scope v1 deterministic classification, or ``None`` when unsure.

    Reuses ``classify_supervisor_scope`` + ``deterministic_decision_for_scope``
    verbatim (the same short-circuit v1 uses). Returns ``None`` for every
    scope without an unambiguous deterministic decision so the caller falls
    through to the ``semantic_router`` model path. Only the semantic fields
    are projected; ``next_agent`` / ``pending_intent`` / task plans are
    dropped at this boundary by construction.
    """
    from app.prompts.agents.supervisor_scope import (
        classify_supervisor_scope,
        deterministic_decision_for_scope,
    )

    text = (query or "").strip()
    if not text:
        raise IntentClassifierError("cannot classify an empty query")
    scope = classify_supervisor_scope(text, has_doc_ids=has_doc_ids)
    decision = deterministic_decision_for_scope(scope, text)
    if decision is None:
        return None
    intent = str(decision.get("intent", "search"))
    if intent not in _VALID_INTENTS:
        return None
    return IntentDecision(
        intent=intent,
        source="deterministic",
        confidence=1.0,
        needs_memory=bool(decision.get("needs_memory", False)),
        is_legal_query=bool(decision.get("is_legal_query", False)),
    )


def _parse_model_output(raw: str) -> IntentDecision:
    """Project the full-taxonomy model JSON onto ``IntentDecision``.

    Only ``intent`` / ``needs_memory`` / ``is_legal_query`` are read; any
    legacy control fields the prompt emits are ignored, never stored.
    Unknown intents fall back to ``search`` (the v1 classifier default).
    """
    text = (raw or "").strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[-2].strip() if len(parts) >= 3 else parts[-1].strip()
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    intent = data.get("intent", "search")
    if intent not in _VALID_INTENTS:
        intent = "search"
    confidence = data.get("confidence", None)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = None
    else:
        confidence = max(0.0, min(1.0, float(confidence)))
    return IntentDecision(
        intent=intent,
        source="model",
        confidence=confidence,
        needs_memory=bool(data.get("needs_memory", False)),
        is_legal_query=bool(data.get("is_legal_query", False)),
    )


class IntentClassifier:
    """Request-scoped v1 intent adapter with per-turn cache.

    One instance lives for one request/turn (wired onto
    ``RuntimeServices.intent_classifier``). ``classify()`` checks the
    deterministic narrow scopes first and only then calls the
    ``semantic_router`` model; every result is cached by normalized query
    so repeated semantic draft builds classify once.
    """

    def __init__(
        self, *, provider_factory: Callable[[], Any] | None = None
    ) -> None:
        self._cache: dict[str, IntentDecision] = {}
        self._provider_factory = provider_factory

    @staticmethod
    def _cache_key(query: str) -> str:
        return (query or "").strip()

    def cached(self, query: str) -> IntentDecision | None:
        """Return the cached decision for ``query``, if present."""
        return self._cache.get(self._cache_key(query))

    @property
    def cache_size(self) -> int:
        """Number of cached query decisions (per request/turn)."""
        return len(self._cache)

    def clear(self) -> None:
        """Drop all cached decisions (turn boundary)."""
        self._cache.clear()

    async def classify(
        self, query: str, *, has_doc_ids: bool = False
    ) -> IntentDecision:
        """Classify ``query`` into a typed ``IntentDecision`` (cached)."""
        key = self._cache_key(query)
        if not key:
            raise IntentClassifierError("cannot classify an empty query")
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        deterministic = classify_deterministic(query, has_doc_ids=has_doc_ids)
        if deterministic is not None:
            self._cache[key] = deterministic
            return deterministic
        decision = await self._classify_via_model(key)
        self._cache[key] = decision
        return decision

    async def _classify_via_model(self, query: str) -> IntentDecision:
        """Full-taxonomy v1 prompt behavior via the ``semantic_router`` role."""
        from app.prompts.agents.supervisor_scope import build_supervisor_system_prompt

        if self._provider_factory is not None:
            provider = self._provider_factory()
        else:
            from app.services.llm import get_semantic_router_provider

            provider = get_semantic_router_provider()
        from app.services.llm.types import LLMMessage as _LLMMsg

        system_prompt = build_supervisor_system_prompt("full", max_iterations=3)
        response_text = ""
        async for chunk in provider.astream(
            [_LLMMsg(role="user", content=query)],
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=256,
        ):
            text = getattr(chunk, "text", None)
            if text:
                response_text += str(text)
        return _parse_model_output(response_text)

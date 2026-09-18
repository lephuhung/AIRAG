"""Runtime-only multi-intent semantic classifier (multi-intent routing spec).

``MultiIntentClassifier`` produces the checkpointed ``IntentAnalysis`` the
flag-on route gate consumes. It exists only while
``V2_MULTI_INTENT_ROUTING_ENABLED`` is on — the ingress wires exactly one
instance per turn onto ``RuntimeServices.multi_intent_classifier`` and the
slot stays ``None`` otherwise.

Ownership rules enforced here (spec §16, §24, §31–§33):

- The model analyzes the WHOLE request and may return multiple intents.
  Identifier presence (phone/CCCD/BHXH/document number) never decides the
  whole intent: identifier scopes have NO deterministic short-circuit —
  the exact compound-query collapse this service exists to fix (§33.1).
- Greeting/personal keep the deterministic short-circuit (non-entity
  scopes; no identifier can hide the rest of the query — §33.1) and emit
  ``source="deterministic"``.
- The model cannot mint ``intent_id`` or ``source``: both are
  server-assigned (by position / by path) before strict contract
  validation. ``is_multi_intent`` must be consistent per the frozen
  contract or the whole output is a failure.
- Every failure — empty query, provider exception, unparseable output,
  schema mismatch — returns ``None``; no exception escapes. The route
  gate maps ``None`` to ``complex_research/semantic_uncertainty``
  (uncertain → complex, §19/§33.2), never a legacy fast path.
- Confidence stays metadata only (§20): validated for range, never
  consulted for routing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from ..contracts.intent import DetectedIntent, IntentAnalysis
from ..contracts.validation import validate_intent_analysis

logger = logging.getLogger(__name__)

__all__ = ["MultiIntentClassifier"]

#: Deterministic short-circuit intents allowed before the model (spec §33.1):
#: greeting/personal are non-entity scopes — no identifier can mask the rest
#: of the request. Identifier scopes (people mongo_search_*, evaluate) have
#: NO short-circuit: they reach the model like everything else.
_SHORT_CIRCUIT_INTENTS = frozenset({"greeting", "personal"})

#: Intent vocabulary taught to the model (INTENT_REGISTRY names). Unknown
#: names still pass contract validation — they route complex via
#: ``semantic_uncertainty`` — but the prompt teaches the canonical names so
#: the model names intents the registry can serve.
_INTENT_VOCABULARY = (
    "people_lookup",
    "people_search",
    "document_lookup",
    "document_search",
    "section_lookup",
    "kg_lookup",
    "memory_lookup",
    "direct_answer",
    "greeting",
    "personal",
    "list_documents",
    "summarize",
    "compare_documents",
    "evaluate_compliance",
    "cross_domain_research",
    "write",
)

_SYSTEM_PROMPT = (
    "You are a semantic intent classifier for a Vietnamese document and "
    "people research assistant. Analyze the ENTIRE user request.\n\n"
    "Rules:\n"
    "- A single request may contain MULTIPLE intents. Do NOT stop at the "
    "first intent you find.\n"
    "- If the user asks to obtain data A and then use A to find, compare, "
    "evaluate, check, reconcile, or summarize data B, return ALL "
    "corresponding intents and mark each downstream intent's "
    '"depends_on" with the indexes of the earlier intents it needs '
    "(0-based positions in the intents array).\n"
    "- Do NOT let the presence of a phone number, citizen ID (CCCD), "
    "social insurance number (BHXH), document number, or any other "
    "identifier decide the whole intent. Identifiers are entity details, "
    "not intent.\n"
    "- Do NOT extract exact identifier values; deterministic extractors "
    "handle them in a later step.\n"
    "- primary_intent names the user's overall goal and MUST be one of "
    "the intent names you return, or null.\n"
    "- is_multi_intent MUST be true exactly when you return more than "
    "one intent.\n"
    "- requires_complex_execution is true when the request needs "
    "multiple steps or a non-atomic capability (comparison, compliance "
    "evaluation, cross-domain research, multi-intent), false only for a "
    "single atomic lookup.\n\n"
    "Allowed intent names: " + ", ".join(_INTENT_VOCABULARY) + ".\n\n"
    "Output ONE JSON object only (no markdown, no prose) with exactly "
    "these keys:\n"
    '{"primary_intent": "<intent name or null>", "intents": [{"name": '
    '"<intent name>", "confidence": <0..1 or null>, "depends_on": '
    '[<earlier intent indexes>], "description": "<short reason or '
    'null>"}], "is_multi_intent": <true|false>, '
    '"requires_complex_execution": <true|false>, "semantic_summary": '
    '"<short summary or null>"}'
)

#: Anti-prose user wrapper (same rationale as the v1 routing classifier):
#: chatty proxy models otherwise answer the request instead of classifying.
_USER_TEMPLATE = "USER REQUEST:\n{query}\n\nReturn the JSON object only."


def _deterministic_short_circuit(
    query: str, *, has_doc_ids: bool = False
) -> IntentAnalysis | None:
    """Greeting/personal narrow scope, or ``None`` to reach the model.

    Reuses the v1-shared ``classify_supervisor_scope`` +
    ``deterministic_decision_for_scope`` verbatim (§33.1: semantics
    unchanged — v2 only stops calling them for identifier scopes). Only
    the two non-entity scopes short-circuit; every other deterministic
    decision (people ``mongo_search_*``, the v2 ``evaluate`` scope) is
    deliberately ignored so an identifier can never collapse a compound
    request before the model sees the whole query.
    """
    from app.prompts.agents.supervisor_scope import (
        classify_supervisor_scope,
        deterministic_decision_for_scope,
    )

    text = (query or "").strip()
    if not text:
        return None
    scope = classify_supervisor_scope(text, has_doc_ids=has_doc_ids)
    decision = deterministic_decision_for_scope(scope, text)
    if decision is None:
        return None
    intent = str(decision.get("intent", ""))
    if intent not in _SHORT_CIRCUIT_INTENTS:
        return None
    return IntentAnalysis(
        primary_intent=intent,
        intents=(
            DetectedIntent(
                intent_id="i1", name=intent, confidence=1.0
            ),
        ),
        is_multi_intent=False,
        requires_complex_execution=False,
        source="deterministic",
    )


def _normalize_model_payload(data: dict) -> dict | None:
    """Rebuild the model JSON with server-assigned identity fields.

    The model emits ``IntentAnalysis`` minus ``intent_id``/``source``
    (spec §17): the server assigns ``intent_id`` by position and
    ``source`` by path, so neither can be minted by model output —
    model-supplied values for those keys are dropped, never trusted.
    Unknown top-level/inner keys are dropped at this boundary too; the
    strict contract is the only shape that survives.
    """
    intents_raw = data.get("intents")
    if not isinstance(intents_raw, list):
        return None
    intents: list[dict] = []
    for index, item in enumerate(intents_raw):
        if not isinstance(item, dict):
            return None
        intents.append(
            {
                "intent_id": f"i{index + 1}",
                "name": item.get("name"),
                "confidence": item.get("confidence"),
                "depends_on": item.get("depends_on", ()),
                "description": item.get("description"),
            }
        )
    return {
        "primary_intent": data.get("primary_intent"),
        "intents": intents,
        "is_multi_intent": data.get("is_multi_intent"),
        "requires_complex_execution": data.get("requires_complex_execution"),
        "semantic_summary": data.get("semantic_summary"),
        "source": "model",
    }


#: Inline reasoning tags emitted by thinking-capable models; stripped
#: before the strict parse (reasoning is not response content).
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse_model_output(raw: str) -> IntentAnalysis | None:
    """Strict-parse the complete response into ``IntentAnalysis``.

    Spec §17 / plan §3.5: structured output — the whole response must be
    the JSON object. No prose salvage, no substring extraction, no
    ``_extract_json_object``: anything that is not exactly one JSON
    object fails closed to ``None`` (uncertain → complex, §33.2), never
    a coerced fallback intent like the v1 adapter's ``search``.
    """
    text = _THINK_TAG_RE.sub("", raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    payload = _normalize_model_payload(data)
    if payload is None:
        return None
    try:
        analysis = IntentAnalysis.model_validate_json(json.dumps(payload))
        validate_intent_analysis(analysis)
    except Exception as exc:  # noqa: BLE001 - failure → semantic_uncertainty
        logger.warning(
            "[intent] multi-intent model output failed strict validation: %s",
            type(exc).__name__,
        )
        return None
    return analysis


class MultiIntentClassifier:
    """Request-scoped multi-intent classifier with per-turn cache.

    One instance lives for one request/turn (constructed once by the
    ingress owner — only while ``V2_MULTI_INTENT_ROUTING_ENABLED`` is
    on — and wired onto ``RuntimeServices.multi_intent_classifier``).
    ``classify()`` checks the deterministic greeting/personal narrow
    scope first and only then calls the ``semantic_router`` model;
    results — including failures — are cached by ``(normalized query,
    has_doc_ids)`` so a resumed or repeated call never re-invokes the
    model. Concurrent callers share one in-flight model call per key
    (single-flight), matching ``IntentClassifier`` semantics.
    """

    def __init__(
        self, *, provider_factory: Callable[[], Any] | None = None
    ) -> None:
        # ``None`` is a cached value too: a failed analysis must replay
        # identically within the turn (single-flight determinism).
        self._cache: dict[tuple[str, bool], IntentAnalysis | None] = {}
        self._locks: dict[tuple[str, bool], asyncio.Lock] = {}
        self._provider_factory = provider_factory

    @staticmethod
    def _cache_key(query: str, has_doc_ids: bool = False) -> tuple[str, bool]:
        return ((query or "").strip(), bool(has_doc_ids))

    def cached(
        self, query: str, *, has_doc_ids: bool = False
    ) -> IntentAnalysis | None:
        """Return the cached analysis for ``query``, if present."""
        return self._cache.get(self._cache_key(query, has_doc_ids))

    @property
    def cache_size(self) -> int:
        """Number of cached query analyses (per request/turn)."""
        return len(self._cache)

    def clear(self) -> None:
        """Drop all cached analyses (turn boundary)."""
        self._cache.clear()

    async def classify(
        self, query: str, *, has_doc_ids: bool = False
    ) -> IntentAnalysis | None:
        """Classify ``query`` into ``IntentAnalysis`` (cached, single-flight).

        ``query`` is the finalized ``semantic.contextualized_query``
        (post coreference resolution — spec §21). ``None`` on every
        failure — the route gate maps it to
        ``complex_research/semantic_uncertainty``. No exception escapes.
        """
        key = self._cache_key(query, has_doc_ids)
        if not key[0]:
            return None
        # ``setdefault`` with no await in between is atomic on the event
        # loop: one lock per key, so concurrent callers serialize and the
        # second reuses the first caller's cached result (double-checked).
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._cache:
                return self._cache[key]
            analysis = _deterministic_short_circuit(
                key[0], has_doc_ids=key[1]
            )
            if analysis is None:
                analysis = await self._classify_via_model(key[0])
            self._cache[key] = analysis
            return analysis

    async def _classify_via_model(self, query: str) -> IntentAnalysis | None:
        """Model classification through the ``semantic_router`` role.

        Plan §3.5: one ``acomplete`` call at ``temperature=0.0`` —
        provider-native ``with_structured_output`` does not exist in
        ``LLMProvider`` (deferred to a later plan), so strict
        ``IntentAnalysis`` schema validation of the complete response is
        the structured-output boundary.
        """
        try:
            if self._provider_factory is not None:
                provider = self._provider_factory()
            else:
                from app.services.llm import get_semantic_router_provider

                provider = get_semantic_router_provider()
            from app.services.llm.types import LLMMessage as _LLMMsg

            result = await provider.acomplete(
                [_LLMMsg(role="user", content=_USER_TEMPLATE.format(query=query))],
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=512,
                think=False,
            )
            raw = getattr(result, "content", result)
        except Exception as exc:  # noqa: BLE001 - failure → semantic_uncertainty
            logger.warning(
                "[intent] multi-intent classification failed: %s",
                type(exc).__name__,
            )
            return None
        if not isinstance(raw, str):
            return None
        return _parse_model_output(raw)

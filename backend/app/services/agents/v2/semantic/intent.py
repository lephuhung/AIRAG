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

import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import Any, Literal

from ..contracts.base import RuntimeModel

logger = logging.getLogger(__name__)

__all__ = [
    "IntentDecision",
    "IntentClassifier",
    "IntentClassifierError",
    "classify_deterministic",
    "classify_evaluate",
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
        # Phase 4A follow-up (final review I1): the v1 taxonomy
        # (RAG/WRITE/PEOPLE/DIRECT) has no evaluate/compliance intent, so
        # ``evaluate`` is produced only by the v2-only deterministic narrow
        # scope below — never by the shared v1 model prompt.
        "evaluate",
    }
)

#: Deterministic v2-only ``evaluate`` narrow scope (final review I1,
#: narrowed by final re-review I5). The v1 taxonomy has no evaluate intent,
#: so an explicit compliance/legal *assessment request* must be recognized
#: deterministically or it would be model-classified to ``search`` and
#: silently degrade to targetless retrieval. An assessment request is one
#: of: an explicit assessment head combined with a compliance/legal domain
#: cue; an explicit degree/level phrase; or the yes/no
#: ``…tuân thủ … không?`` form. A bare compliance-topic noun ("quy định về
#: tuân thủ thuế là gì?") is NOT an assessment request and stays on the
#: general-RAG fast path; the bare generic ``đánh giá`` keyword likewise
#: stays demoted (Task-3). See the plan/spec erratum for the ownership
#: rationale (final re-review M8).
#: Explicit assessment heads (Task 1A/M11): an unambiguous assessment
#: action, so a bare legal-topic phrase is enough to form an assessment
#: request ("đánh giá văn bản pháp lý mới ban hành").
_EVALUATE_ASSESS_HEAD_RE = re.compile(
    r"\b(?:đánh\s*giá|thẩm\s*định|assess|evaluate|evaluation)\b",
    re.IGNORECASE | re.UNICODE,
)
#: Dual-sense action heads (Task 1A/M11): lookup/browse/compare senses
#: ("kiểm tra giúp tôi văn bản pháp lý", "review văn bản pháp lý mới
#: nhất") as well as audit senses, so they form an assessment request
#: only together with an explicit compliance/assessment object — never
#: with a bare legal-topic phrase.
_EVALUATE_ACTION_HEAD_RE = re.compile(
    r"\b(?:kiểm\s*tra|rà\s*soát|đối\s*chiếu|xác\s*định|review)\b",
    re.IGNORECASE | re.UNICODE,
)
#: Explicit compliance/assessment objects ("tuân thủ", "compliance",
#: "tính pháp lý"). Checked before the bare topic cue below: "tính
#: pháp lý" contains "pháp lý" but is an assessment object, not a
#: topic mention.
_EVALUATE_COMPLIANCE_RE = re.compile(
    r"\b(?:tuân\s*thủ|compliance|tính\s+pháp\s*lý)\b",
    re.IGNORECASE | re.UNICODE,
)
#: Bare legal-topic phrase ("văn bản pháp lý", "quy định pháp lý").
#: Only an assessment request together with an explicit assessment head;
#: with a dual-sense action head it stays an informational/action query.
_EVALUATE_TOPIC_RE = re.compile(
    r"\bpháp\s*lý\b",
    re.IGNORECASE | re.UNICODE,
)
#: Explicit degree/level assessment phrase ("mức độ tuân thủ"/"mức độ rủi ro").
_EVALUATE_LEVEL_RE = re.compile(
    r"\bmức\s*độ\s+(?:tuân\s*thủ|rủi\s*ro)\b",
    re.IGNORECASE | re.UNICODE,
)
#: Yes/no compliance question ("Tôi có tuân thủ … không?").
_EVALUATE_YN_RE = re.compile(
    r"\b(?:có\s+)?tuân\s*thủ\b[^.!?\n]{0,80}\bkhông\b\s*\?",
    re.IGNORECASE | re.UNICODE,
)


#: Inline reasoning tags emitted by thinking-capable models; stripped before
#: JSON extraction exactly like the v1 supervisor parser.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

#: Anti-prose user wrapper (v1 parity): chatty proxy models otherwise answer
#: the legal question in prose instead of returning routing JSON.
_CLASSIFY_USER_TEMPLATE = (
    "You are a ROUTING classifier. Do NOT answer the user's question. "
    "Output ONE JSON object only (no markdown, no prose) with keys: "
    "intent, needs_memory, is_legal_query.\n\nUSER MESSAGE:\n{query}"
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


def classify_evaluate(
    query: str, *, has_doc_ids: bool = False
) -> IntentDecision | None:
    """Deterministic compliance/evaluate narrow scope (v2-only), or ``None``.

    ``evaluate`` has no v1 taxonomy intent, so it cannot come from the shared
    model prompt. This conservative scope short-circuits an explicit
    compliance/legal *assessment request* to a typed ``evaluate`` decision
    before any model call — exactly like the greeting/personal/people narrow
    scopes above. An assessment request is an explicit assessment head
    (``đánh giá``/``thẩm định``/``assess``/…) combined with a
    compliance/legal cue, a dual-sense action head (``kiểm tra``/``rà
    soát``/``đối chiếu``/``xác định``/``review``) combined with an
    explicit compliance/assessment object (``tuân thủ``/``compliance``/
    ``tính pháp lý``), an explicit degree/level phrase (``mức độ tuân
    thủ``), or the yes/no ``…tuân thủ … không?`` form. A dual-sense
    action verb with only a bare legal-topic phrase (``văn bản pháp
    lý``), bare compliance-topic nouns, and the bare generic ``đánh
    giá`` keyword are NOT assessment requests and stay on the model path
    (advisory) / general-RAG fast path (deterministic route).
    """
    _ = has_doc_ids  # evaluation cue does not depend on attached docs
    text = (query or "").strip()
    if not text:
        raise IntentClassifierError("cannot classify an empty query")
    has_compliance = _EVALUATE_COMPLIANCE_RE.search(text) is not None
    has_topic = has_compliance or _EVALUATE_TOPIC_RE.search(text) is not None
    is_assessment = (
        _EVALUATE_LEVEL_RE.search(text) is not None
        or _EVALUATE_YN_RE.search(text) is not None
        or (
            _EVALUATE_ASSESS_HEAD_RE.search(text) is not None
            and has_topic
        )
        or (
            _EVALUATE_ACTION_HEAD_RE.search(text) is not None
            and has_compliance
        )
    )
    if not is_assessment:
        return None
    return IntentDecision(
        intent="evaluate",
        source="deterministic",
        confidence=1.0,
        needs_memory=False,
        is_legal_query=True,
    )


def _as_bool(value: object) -> bool:
    """Strict boolean projection: only a real ``True`` is true.

    The model prompt asks for JSON booleans, but a chatty model may emit
    ``"false"`` / ``"yes"`` strings; ``bool(...)`` would coerce every
    non-empty string to ``True``. Anything that is not exactly ``True``
    is ``False``.
    """
    return value is True


def _extract_json_object(raw: str) -> tuple[dict | None, bool]:
    """Extract the routing JSON object using v1's tolerant semantics.

    Order mirrors ``_parse_supervisor_response``: strip ``<think>`` tags,
    unwrap a fenced block, try a direct parse, then salvage the
    ``raw[find("{") : rfind("}")+1]`` slice from prose. Returns the
    payload and whether it was salvaged (informational only).
    """
    text = (raw or "").strip()
    text = _THINK_TAG_RE.sub("", text).strip()
    candidate = text
    if "```json" in candidate:
        candidate = candidate.split("```json", 1)[-1].split("```", 1)[0].strip()
    elif "```" in candidate:
        parts = candidate.split("```")
        if len(parts) >= 3:
            candidate = parts[1].strip()
    try:
        data = json.loads(candidate) if candidate else None
        if isinstance(data, dict):
            return data, False
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(candidate[start : end + 1])
            if isinstance(data, dict):
                logger.info(
                    "[intent] Salvaged JSON object embedded in model output "
                    f"({len(candidate)} chars \u2192 {end - start + 1} chars)"
                )
                return data, True
        except json.JSONDecodeError:
            pass
    return None, False


def _parse_model_output(raw: str) -> IntentDecision:
    """Project the full-taxonomy model JSON onto ``IntentDecision``.

    Only ``intent`` / ``needs_memory`` / ``is_legal_query`` are read; any
    legacy control fields the prompt emits are ignored, never stored.
    Unknown intents fall back to ``search`` (the v1 classifier default).
    A total parse failure falls back to ``search`` with ``is_legal_query``
    ``True`` and a warning (v1 parity: retrieval-backed, never silent).
    """
    data, _salvaged = _extract_json_object(raw)
    if data is None:
        logger.warning(
            "[intent] Failed to parse model JSON (falling back to search): %r",
            (raw or "")[:200],
        )
        return IntentDecision(
            intent="search",
            source="model",
            confidence=None,
            needs_memory=False,
            is_legal_query=True,
        )
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
        needs_memory=_as_bool(data.get("needs_memory", False)),
        is_legal_query=_as_bool(data.get("is_legal_query", False)),
    )


class IntentClassifier:
    """Request-scoped v1 intent adapter with per-turn cache.

    One instance lives for one request/turn (constructed once by the
    ingress owner and wired onto ``RuntimeServices.intent_classifier``).
    ``classify()`` checks the deterministic narrow scopes first and only
    then calls the ``semantic_router`` model; every result is cached by
    ``(normalized query, has_doc_ids)`` so repeated semantic draft builds
    classify once. Concurrent callers share one in-flight model call per
    key (single-flight). Consumers must reuse
    ``services.intent_classifier`` and never construct a classifier per
    call — a per-call instance would defeat cross-build caching.
    """

    def __init__(
        self, *, provider_factory: Callable[[], Any] | None = None
    ) -> None:
        self._cache: dict[tuple[str, bool], IntentDecision] = {}
        self._locks: dict[tuple[str, bool], asyncio.Lock] = {}
        self._provider_factory = provider_factory

    @staticmethod
    def _cache_key(query: str, has_doc_ids: bool = False) -> tuple[str, bool]:
        # ``has_doc_ids`` changes ``classify_supervisor_scope``, so it is
        # part of the key: the same text with attached documents may route
        # through a different scope.
        return ((query or "").strip(), bool(has_doc_ids))

    def cached(self, query: str, *, has_doc_ids: bool = False) -> IntentDecision | None:
        """Return the cached decision for ``query``, if present."""
        return self._cache.get(self._cache_key(query, has_doc_ids))

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
        """Classify ``query`` into a typed ``IntentDecision`` (cached, single-flight)."""
        key = self._cache_key(query, has_doc_ids)
        if not key[0]:
            raise IntentClassifierError("cannot classify an empty query")
        # ``setdefault`` with no await in between is atomic on the event
        # loop: one lock per key, so concurrent callers serialize and the
        # second reuses the first caller's cached result (double-checked).
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            hit = self._cache.get(key)
            if hit is not None:
                return hit
            deterministic = classify_deterministic(query, has_doc_ids=has_doc_ids)
            if deterministic is not None:
                self._cache[key] = deterministic
                return deterministic
            evaluate = classify_evaluate(query, has_doc_ids=has_doc_ids)
            if evaluate is not None:
                self._cache[key] = evaluate
                return evaluate
            decision = await self._classify_via_model(key[0])
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
        user_content = _CLASSIFY_USER_TEMPLATE.format(query=query)
        response_text = ""
        async for chunk in provider.astream(
            [_LLMMsg(role="user", content=user_content)],
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=160,  # routing JSON only; keep short so chatty models can't rant
            think=False,  # disable thinking to reduce latency for classification
        ):
            # Thinking-capable providers yield reasoning separately; the
            # routing JSON lives in text chunks only (v1 parity).
            if getattr(chunk, "type", "text") == "thinking":
                continue
            text = getattr(chunk, "text", None)
            if not text:
                continue
            response_text += str(text)
            # Early-stop once a complete JSON object is buffered — prevents
            # waiting for a long prose answer after a valid routing blob.
            if "{" in response_text and "}" in response_text:
                _s, _e = response_text.find("{"), response_text.rfind("}")
                if _e > _s:
                    try:
                        json.loads(response_text[_s : _e + 1])
                        logger.info(
                            "[intent] Early-stop classifier stream "
                            f"({len(response_text)} chars, valid JSON)"
                        )
                        break
                    except json.JSONDecodeError:
                        pass
        return _parse_model_output(response_text)

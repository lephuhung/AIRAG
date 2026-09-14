"""Task 2 — runtime-only typed IntentDecision adapter/cache (RED first).

Covers the Phase 4A semantic seam: deterministic narrow scopes (greeting /
people) without a model call, model fallback through the semantic_router
role with per-request caching, runtime-only placement (never checkpointed),
and no legacy supervisor control-field leakage.
"""
from __future__ import annotations

import pytest

from app.services.agents.v2.contracts.base import ContractModel, RuntimeModel
from app.services.agents.v2.contracts.state import RuntimeServices, SupervisorV2State
from app.services.agents.v2.semantic.intent import (
    IntentDecision,
    IntentClassifier,
    classify_deterministic,
)


def test_intent_decision_is_runtime_only() -> None:
    assert issubclass(IntentDecision, RuntimeModel)
    assert not issubclass(IntentDecision, ContractModel)
    annotations = set(SupervisorV2State.__annotations__)
    assert "intent" not in annotations
    assert "intent_decision" not in annotations


def test_deterministic_greeting_needs_no_model() -> None:
    decision = classify_deterministic("xin chào")
    assert decision is not None
    assert decision.intent == "greeting"
    assert decision.source == "deterministic"
    assert decision.needs_memory is False
    assert decision.is_legal_query is False


def test_deterministic_people_phone() -> None:
    decision = classify_deterministic("0901234567 là ai?")
    assert decision is not None
    assert decision.intent == "mongo_search_phone"
    assert decision.source == "deterministic"


def test_greeting_prefix_factual_is_not_deterministic() -> None:
    assert classify_deterministic("chào anh, hỏi về chế độ thai sản?") is None


@pytest.mark.asyncio
async def test_model_fallback_caches_per_request() -> None:
    calls: list[str] = []

    class _FakeProvider:
        async def astream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(messages[0].content if messages else "")
            yield _FakeChunk(
                '{"intent": "search", "needs_memory": false, "is_legal_query": true}'
            )

    class _FakeChunk:
        def __init__(self, text: str) -> None:
            self.text = text

    classifier = IntentClassifier(provider_factory=lambda: _FakeProvider())
    first = await classifier.classify("chế độ thai sản được quy định thế nào?")
    second = await classifier.classify("chế độ thai sản được quy định thế nào?")
    assert first.intent == "search"
    assert first.source == "model"
    assert first.is_legal_query is True
    assert first is second  # same cached instance, no second model call
    assert len(calls) == 1
    assert "next_agent" not in first.model_dump()
    assert "pending_intent" not in first.model_dump()


def test_runtime_services_carries_intent_cache_slot() -> None:
    services = RuntimeServices()
    assert hasattr(services, "intent_classifier")
    assert services.intent_classifier is None

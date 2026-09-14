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


# ---------------------------------------------------------------------------
# Fix round 1/5: v1-tolerant model-output extraction (I1, M3, M7)
# ---------------------------------------------------------------------------


def _text_chunks(*texts: str):
    class _Chunk:
        def __init__(self, text: str, kind: str = "text") -> None:
            self.text = text
            self.type = kind

    async def _gen():
        for text in texts:
            yield _Chunk(text)

    return _gen()


def _provider_with(chunks) -> object:
    class _FakeProvider:
        async def astream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            if hasattr(chunks, "__aiter__"):
                async for chunk in chunks:
                    yield chunk
            else:
                for chunk in chunks:
                    yield chunk

    return _FakeProvider()


def _chunk(text: str, kind: str = "text") -> object:
    return type("C", (), {"text": text, "type": kind})()


@pytest.mark.asyncio
async def test_model_parses_fenced_json_block() -> None:
    classifier = IntentClassifier(
        provider_factory=lambda: _provider_with(
            _text_chunks(
                '```json\n{"intent": "search_section", '
                '"needs_memory": false, "is_legal_query": true}\n```'
            )
        )
    )
    decision = await classifier.classify("\u0110i\u1ec1u 5 n\u00f3i g\u00ec?")
    assert decision.intent == "search_section"
    assert decision.source == "model"
    assert decision.is_legal_query is True


@pytest.mark.asyncio
async def test_model_parses_prose_embedded_json() -> None:
    classifier = IntentClassifier(
        provider_factory=lambda: _provider_with(
            _text_chunks(
                'Sure, here is the JSON: {"intent": "kg_query", '
                '"needs_memory": false, "is_legal_query": true} hope this helps'
            )
        )
    )
    decision = await classifier.classify("B\u1ed9 C\u00f4ng an c\u00f3 \u0111\u01a1n v\u1ecb n\u00e0o?")
    assert decision.intent == "kg_query"
    assert decision.is_legal_query is True


@pytest.mark.asyncio
async def test_model_ignores_thinking_chunks_and_tags() -> None:
    classifier = IntentClassifier(
        provider_factory=lambda: _provider_with(
            [
                _chunk('{"intent": "search"', kind="thinking"),
                _chunk(
                    "<think>deciding between search and kg</think>"
                    '{"intent": "kg_query", "needs_memory": false, '
                    '"is_legal_query": true}'
                ),
            ]
        )
    )
    decision = await classifier.classify("B\u1ed9 C\u00f4ng an c\u00f3 \u0111\u01a1n v\u1ecb n\u00e0o?")
    assert decision.intent == "kg_query"


@pytest.mark.asyncio
async def test_model_trailing_prose_after_json() -> None:
    classifier = IntentClassifier(
        provider_factory=lambda: _provider_with(
            _text_chunks(
                '{"intent": "summarize", "needs_memory": false, '
                '"is_legal_query": true} That completes the routing.'
            )
        )
    )
    decision = await classifier.classify("T\u00f3m t\u1eaft Ngh\u1ecb \u0111\u1ecbnh A")
    assert decision.intent == "summarize"


@pytest.mark.asyncio
async def test_model_parse_failure_falls_back_legal_and_warns(caplog) -> None:
    import logging

    classifier = IntentClassifier(
        provider_factory=lambda: _provider_with(_text_chunks("definitely not json at all"))
    )
    with caplog.at_level(logging.WARNING, logger="app.services.agents.v2.semantic.intent"):
        decision = await classifier.classify("ch\u1ebf \u0111\u1ed9 thai s\u1ea3n?")
    assert decision.intent == "search"
    assert decision.source == "model"
    assert decision.is_legal_query is True  # v1 parse-failure parity
    assert caplog.records, "expected a fallback warning log"


@pytest.mark.asyncio
async def test_model_string_booleans_not_coerced() -> None:
    classifier = IntentClassifier(
        provider_factory=lambda: _provider_with(
            _text_chunks(
                '{"intent": "search", "needs_memory": "false", '
                '"is_legal_query": "yes"}'
            )
        )
    )
    decision = await classifier.classify("ch\u1ebf \u0111\u1ed9 thai s\u1ea3n?")
    assert decision.needs_memory is False
    assert decision.is_legal_query is False


# ---------------------------------------------------------------------------
# Fix round 1/5: cache key scope + single-flight + no-model deterministic (M1/M2/M6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_key_includes_doc_scope() -> None:
    calls: list[str] = []

    class _FakeProvider:
        async def astream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            calls.append("hit")
            yield type("C", (), {"text": '{"intent": "search"}', "type": "text"})()

    classifier = IntentClassifier(provider_factory=lambda: _FakeProvider())
    query = "ch\u1ebf \u0111\u1ed9 thai s\u1ea3n?"
    await classifier.classify(query)
    await classifier.classify(query, has_doc_ids=True)
    await classifier.classify(query)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_concurrent_classify_single_flight() -> None:
    import asyncio

    calls: list[str] = []
    release = asyncio.Event()

    class _FakeProvider:
        async def astream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            calls.append("hit")
            await release.wait()
            yield type("C", (), {"text": '{"intent": "search"}', "type": "text"})()

    classifier = IntentClassifier(provider_factory=lambda: _FakeProvider())
    query = "ch\u1ebf \u0111\u1ed9 thai s\u1ea3n?"
    task_a = asyncio.create_task(classifier.classify(query))
    task_b = asyncio.create_task(classifier.classify(query))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(task_a, task_b)
    assert len(calls) == 1
    assert first is second


@pytest.mark.asyncio
async def test_classify_deterministic_makes_no_model_call() -> None:
    def _boom():
        raise AssertionError("model must not be consulted for narrow scopes")

    classifier = IntentClassifier(provider_factory=_boom)
    greeting = await classifier.classify("xin ch\u00e0o")
    assert greeting.intent == "greeting" and greeting.source == "deterministic"
    people = await classifier.classify("T\u00ecm \u00f4ng Nguy\u1ec5n V\u0103n A")
    assert people.intent == "mongo_search_name" and people.source == "deterministic"


# ---------------------------------------------------------------------------
# Fix round 1/5: real per-turn construction/pass-through (I2)
# ---------------------------------------------------------------------------


def test_build_runtime_services_passes_through_intent_classifier() -> None:
    from app.services.agents.supervisor_v2 import build_runtime_services

    sentinel = object()
    assert build_runtime_services(intent_classifier=sentinel).intent_classifier is sentinel
    assert build_runtime_services().intent_classifier is None


@pytest.mark.asyncio
async def test_ingress_wires_fresh_turn_intent_classifier() -> None:
    from types import SimpleNamespace
    from uuid import uuid4

    import app.services.agent.runtime_selector as selector
    from app.services.agents.v2.semantic.intent import IntentClassifier

    class _FakeSession:
        async def commit(self):
            return None

        async def rollback(self):
            return None

        async def close(self):
            return None

    async def _fake_preprocess(raw_query: str):
        return SimpleNamespace(normalized_query=raw_query)

    def _kwargs():
        return {
            "user_id": uuid4(),
            "authenticated_workspace_ids": [uuid4()],
            "requested_workspace_ids": None,
            "raw_query": "ch\u1ebf \u0111\u1ed9 thai s\u1ea3n?",
            "thread_id": f"thread-{uuid4().hex[:8]}",
            "session_factory": lambda: _FakeSession(),
            "lease_session_factory": lambda: _FakeSession(),
            "preprocess": _fake_preprocess,
            "available_services": frozenset(),
        }

    async with selector.build_v2_ingress(**_kwargs()) as first:
        classifier = first.runtime_context.services.intent_classifier
        assert isinstance(classifier, IntentClassifier)
    async with selector.build_v2_ingress(**_kwargs()) as second:
        assert isinstance(second.runtime_context.services.intent_classifier, IntentClassifier)
        assert second.runtime_context.services.intent_classifier is not classifier

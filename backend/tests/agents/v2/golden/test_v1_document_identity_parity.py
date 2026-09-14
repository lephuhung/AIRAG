"""Task 5 — v1 document-resolution parity corpus test (Phase 4B characterization gate).

Proves the fixtures in ``document_identity_cases.py`` represent CURRENT v1
behavior before any v2 document-identity change (Tasks 6/7):

- deterministic parse cases: ``_extract_by_regex`` (pure, no DB/LLM/vector)
  must reproduce the recorded v1 parse exactly — scalar fields byte-equal,
  ``doc_number_candidates`` rewired through ``_generate_number_candidates``
  with the recorded suffixes (never pinned to the wall-clock year, so the
  corpus cannot rot on Jan 1);
- number-identity dominance: the real ``_rerank_candidates`` hard filter
  (stub DB, no live infra) must drop a wrong-number candidate even when its
  title matches the question topic, and keep same-number candidates;
- merge semantics: the pure ``_merge_candidates`` agreement boost
  (+30% of the second strategy score, strategies recorded);
- status thresholds: the real ``resolve_doc_agent`` constants
  (EARLY_EXIT 0.85 / HIGH 0.60 / MEDIUM 0.30 / AMBIGUITY_RATIO 0.75) pin the
  recorded ``expected_status`` vocabulary; the ratio predicate is evaluated
  on synthetic score pairs (no live DB);
- DB/LLM/vector-backed candidate IDs and scores are recorded as
  ``v1_pipeline`` expectations (stage order, strategy names, status) — shape
  only, never asserted against live infra;
- fixtures carry no legacy routing keys, no document UUIDs, no secrets/PII.

Verified against current code on 2026-09-14 (see task-5 report for probe
outputs): ``_extract_by_regex`` replays every recorded parse; ``_rerank``
drops "85"-numbered docs for a "số 53" query; ``_merge`` boosts agreement.
"""
from __future__ import annotations

import re
import uuid

import pytest

from app.services.agent.doc_resolver import (
    _extract_by_regex,
    _generate_number_candidates,
    _merge_candidates,
    _number_token_present,
    _query_db,
    _rerank_candidates,
    _topic_tokens,
)
from app.services.agents.resolve_doc_agent import (
    AMBIGUITY_RATIO,
    EARLY_EXIT_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)

from tests.agents.v2.golden.document_identity_cases import (
    DOCUMENT_IDENTITY_CASES,
    REQUIRED_IDS,
)

#: Allowed ``parse_source`` vocabulary. ``regex-deterministic`` = exact replay
#: of the pure Stage-0 extractor; ``pipeline-characterization`` = DB/merge/
#: rerank behavior characterized with stub inputs (no live infra).
PARSE_SOURCES = frozenset({"regex-deterministic", "pipeline-characterization"})

#: Strategy names the v1 pipeline can emit (doc_resolver stages + topic boost).
KNOWN_STRATEGIES = frozenset({"db_query", "llm_db", "vector", "topic", "similar"})

#: v1 outcome vocabulary (resolve_doc_agent routing). ``low_confidence_no_bind``
#: is the fixture label for the ``_build_resolved_state`` LOW path (agent span
#: outcome "resolved" but ``document_ids=[]`` — no scoping, no binding).
KNOWN_STATUSES = frozenset(
    {
        "resolved",
        "ambiguous",
        "medium_confirm",
        "low_confidence_no_bind",
        "not_found",
        "similar_suggest",
    }
)

#: Brief-pinned query strings: catches query drift an id-only gate would miss.
EXPECTED_QUERIES = {
    "full-official-number": "Nghị định 53/2022/NĐ-CP",
    "short-bare-number-luat": "luật số 24",
    "short-bare-number-thongtu": "thông tư 15",
    "number-plus-agency": "Thông tư 15 của Bộ Công an",
    "exact-title": "Luật An ninh mạng",
    "partial-title": "an ninh mạng",
    "year-plus-topic": "quy định về an toàn thông tin năm 2018",
    "topic-only": "văn bản quy định về chế độ thai sản",
    "number-dominates-rerank": "Tóm tắt Nghị định 53 về hệ thống thông tin",
    "ambiguous-candidates": "Thông tư 15 của Bộ Công an",
    "low-confidence-no-candidate": "cho tôi xem",
    "canonical-section-query": "Điều 5 Luật An ninh mạng quy định gì?",
}

_DETERMINISTIC = [
    c for c in DOCUMENT_IDENTITY_CASES if c["parse_source"] == "regex-deterministic"
]


def test_corpus_covers_required_cases() -> None:
    ids = {case["id"] for case in DOCUMENT_IDENTITY_CASES}
    assert REQUIRED_IDS <= ids, f"missing required cases: {REQUIRED_IDS - ids}"
    by_id = {case["id"]: case["query"] for case in DOCUMENT_IDENTITY_CASES}
    for case_id, query in EXPECTED_QUERIES.items():
        assert by_id.get(case_id) == query, f"{case_id}: query drifted"


def test_provenance_vocabulary() -> None:
    for case in DOCUMENT_IDENTITY_CASES:
        assert case["parse_source"] in PARSE_SOURCES, (
            f"{case['id']}: unknown provenance {case['parse_source']!r}"
        )
    assert _DETERMINISTIC, "corpus must contain regex-deterministic v1 cases"


def test_regex_parse_replays_current_v1_behavior() -> None:
    assert _DETERMINISTIC, "corpus must contain deterministic v1 parse cases"
    scalar_keys = (
        "doc_type_slug",
        "document_number",
        "number_raw",
        "title_keywords",
        "issuing_agency_text",
        "issuing_agency_code",
        "year",
        "section_reference",
        "confidence",
    )
    for case in _DETERMINISTIC:
        parsed = _extract_by_regex(case["query"])
        expected = case["v1_parse"]
        for key in scalar_keys:
            assert parsed[key] == expected[key], (
                f"{case['id']}: {key} {parsed[key]!r} != {expected[key]!r}"
            )
        # Candidate wiring: explicit user-typed number stays first; the rest
        # must equal what the generator yields for the recorded suffixes/year
        # (recomputed live so the fixture never pins the wall-clock year).
        suffixes = expected["candidate_suffixes"]
        if suffixes:
            assert parsed["doc_number_candidates"], f"{case['id']}: no candidates"
            if expected["explicit_number_first"]:
                assert parsed["doc_number_candidates"][0] == expected[
                    "document_number"
                ], f"{case['id']}: explicit number must lead"
                generated = [expected["document_number"]] + [
                    c
                    for c in _generate_number_candidates(
                        expected["number_raw"], suffixes, expected["year"]
                    )
                    if c != expected["document_number"]
                ]
            else:
                generated = _generate_number_candidates(
                    expected["number_raw"], suffixes, expected["year"]
                )
            assert parsed["doc_number_candidates"] == generated, (
                f"{case['id']}: candidate wiring drifted"
            )
        else:
            assert parsed["doc_number_candidates"] == [], (
                f"{case['id']}: expected no number candidates"
            )


def test_number_identity_dominates_topic_rerank() -> None:
    """A wrong-number candidate must lose to number identity even when its
    title matches the question topic (the ND53-vs-ND85 guard)."""
    assert _number_token_present("53/2022/nđ-cp", "53")
    assert not _number_token_present("nghị định 85", "53")
    assert not _number_token_present("nghị định 853", "53")
    tokens = _topic_tokens("Tóm tắt Nghị định 53 về hệ thống thông tin")
    assert "53" not in tokens, "bare number is identity, never a topic token"
    assert {"hệ", "thống", "thông", "tin"} <= set(tokens)


class _StubRow:
    def __init__(self, row_id: str, number: str, title: str) -> None:
        self.id = row_id
        self.document_number = number
        self.document_title = title
        self.original_filename = ""


class _StubDB:
    def __init__(self, rows: list[_StubRow]) -> None:
        self._rows = rows

    async def execute(self, _query: object) -> list[_StubRow]:
        return list(self._rows)


@pytest.mark.asyncio
async def test_rerank_hard_filter_drops_wrong_number() -> None:
    """Real ``_rerank_candidates`` with a stub DB: the topic-matching ND85
    lookalike is dropped for a "số 53" query; same-number docs survive and
    the topic-matching one ranks first."""
    candidates = [
        {
            "document_id": "doc-85",
            "title": "Nghị định 85 về hệ thống thông tin",
            "score": 0.90,
            "strategy": "vector",
        },
        {
            "document_id": "doc-53a",
            "title": "Nghị định 53 về hệ thống thông tin",
            "score": 0.60,
            "strategy": "db_query",
        },
        {
            "document_id": "doc-53b",
            "title": "Nghị định 53 về an toàn khác",
            "score": 0.62,
            "strategy": "db_query",
        },
    ]
    db = _StubDB(
        [
            _StubRow("doc-85", "85/2020/NĐ-CP", "Nghị định 85 về hệ thống thông tin"),
            _StubRow("doc-53a", "53/2022/NĐ-CP", "Nghị định 53 về hệ thống thông tin"),
            _StubRow("doc-53b", "53/2021/NĐ-CP", "Nghị định 53 về an toàn khác"),
        ]
    )
    parsed = _extract_by_regex("Tóm tắt Nghị định 53 về hệ thống thông tin")
    out = await _rerank_candidates(
        [dict(c) for c in candidates],
        parsed,
        "Tóm tắt Nghị định 53 về hệ thống thông tin",
        db,  # type: ignore[arg-type]
    )
    kept = [c["document_id"] for c in out]
    assert "doc-85" not in kept, "wrong-number doc must not survive identity filter"
    assert set(kept) == {"doc-53a", "doc-53b"}
    assert kept[0] == "doc-53a", "topic-matching same-number doc ranks first"


@pytest.mark.asyncio
async def test_rerank_empty_when_no_candidate_matches_number() -> None:
    """When NOTHING matches the named number, v1 returns [] (not-found) rather
    than answering about an unrelated document."""
    candidates = [
        {"document_id": "doc-85", "title": "Nghị định 85", "score": 0.9},
    ]
    db = _StubDB([_StubRow("doc-85", "85/2020/NĐ-CP", "Nghị định 85")])
    parsed = _extract_by_regex("Nghị định 53/2022/NĐ-CP")
    out = await _rerank_candidates(
        [dict(c) for c in candidates], parsed, "Nghị định 53/2022/NĐ-CP", db  # type: ignore[arg-type]
    )
    assert out == []


class _StubDoc:
    def __init__(self, doc_id: str, title: str) -> None:
        self.id = doc_id
        self.document_title = title
        self.original_filename = ""
        self.document_number = ""
        self.published_date = ""
        self.issuing_agency = ""
        self.parent_agency = ""
        self.document_type = None


class _StubScalars:
    def __init__(self, docs: list[_StubDoc]) -> None:
        self._docs = docs

    def all(self) -> list[_StubDoc]:
        return list(self._docs)


class _StubResult:
    def __init__(self, docs: list[_StubDoc]) -> None:
        self._docs = docs

    def scalars(self) -> _StubScalars:
        return _StubScalars(self._docs)


class _StubQueryDB:
    """Mimics a populated workspace: arbitrary indexed docs, no filters."""

    def __init__(self, docs: list[_StubDoc]) -> None:
        self._docs = docs

    async def execute(self, _query: object) -> _StubResult:
        return _StubResult(self._docs)


@pytest.mark.asyncio
async def test_empty_parse_db_stage_returns_zero_scored_docs() -> None:
    """I1 code pin: with an all-empty parse ``_query_db`` applies no
    content filter, so a populated workspace yields arbitrary candidates
    scored exactly 0.0 (strategy ``db_query``) — and that non-empty list
    gates off the llm_db/vector/similar stages in ``resolve_candidates``."""
    parsed = _extract_by_regex("cho tôi xem")
    assert parsed["confidence"] == "low"
    db = _StubQueryDB([_StubDoc("doc-x", "X"), _StubDoc("doc-y", "Y")])
    out = await _query_db(parsed, ["ws-1"], db)  # type: ignore[arg-type]
    assert len(out) == 2
    assert all(c["score"] == 0.0 for c in out)
    assert all(c["strategy"] == "db_query" for c in out)


def test_merge_agreement_boost_matches_v1_semantics() -> None:
    """Two strategies agreeing on one doc boost it by +30% of the second
    score and record both strategies."""
    merged = _merge_candidates(
        [
            [{"document_id": "doc-a", "title": "A", "score": 0.70, "strategy": "db_query"}],
            [{"document_id": "doc-a", "title": "A", "score": 0.50, "strategy": "vector"}],
            [{"document_id": "doc-b", "title": "B", "score": 0.72, "strategy": "vector"}],
        ]
    )
    by_id = {c["document_id"]: c for c in merged}
    assert by_id["doc-a"]["score"] == pytest.approx(0.70 + 0.50 * 0.3)
    assert by_id["doc-a"]["strategies"] == ["db_query", "vector"]
    assert merged[0]["document_id"] == "doc-a", "agreement must outrank lone hit"


def test_status_thresholds_match_live_v1_constants() -> None:
    """The recorded v1 status contract must equal the live agent constants —
    any threshold drift fails loudly here, not silently in Task 6."""
    assert EARLY_EXIT_THRESHOLD == 0.85
    assert HIGH_CONFIDENCE_THRESHOLD == 0.60
    assert MEDIUM_CONFIDENCE_THRESHOLD == 0.30
    assert AMBIGUITY_RATIO == 0.75
    # Ratio predicate on synthetic pairs (mirrors resolve_doc_agent logic):
    assert (0.80 / 1.00) >= AMBIGUITY_RATIO, "close 2nd must read ambiguous"
    assert not (0.50 / 1.00) >= AMBIGUITY_RATIO, "distant 2nd must read clear"


def _case(case_id: str) -> dict:
    return next(c for c in DOCUMENT_IDENTITY_CASES if c["id"] == case_id)


def test_low_confidence_case_runs_db_stage_only() -> None:
    """I1: an all-empty parse must not claim the llm/vector/similar stages.
    ``_query_db`` builds only workspace+status filters for it, so a populated
    workspace short-circuits every later stage."""
    pipe = _case("low-confidence-no-candidate")["v1_pipeline"]
    assert pipe["stages"] == ["db_query"], (
        "empty parse never reaches llm_db/vector/similar in a populated workspace"
    )
    assert pipe["expected_status"] == "low_confidence_no_bind"


def test_section_query_status_never_exceeds_exact_title() -> None:
    """I2: for the same matched document the section query scores no higher
    than the exact-title query (same +0.25 type bonus, strictly lower keyword
    ratio from the noise token), so "section resolved" must imply
    "exact-title resolved"."""
    exact = _case("exact-title")
    section = _case("canonical-section-query")
    assert not (
        section["v1_pipeline"]["expected_status"] == "resolved"
        and exact["v1_pipeline"]["expected_status"] != "resolved"
    ), "mutually exclusive statuses: section resolved implies exact resolved"
    exact_kw = exact["v1_parse"]["title_keywords"]
    section_kw = section["v1_parse"]["title_keywords"]
    assert set(exact_kw) <= set(section_kw), "same matchable title signal"
    assert len(section_kw) > len(exact_kw), "noise token dilutes the ratio"


def test_pipeline_expectations_use_known_vocabulary() -> None:
    for case in DOCUMENT_IDENTITY_CASES:
        pipe = case["v1_pipeline"]
        assert pipe["expected_status"] in KNOWN_STATUSES, (
            f"{case['id']}: unknown status {pipe['expected_status']!r}"
        )
        assert set(pipe["stages"]) <= KNOWN_STRATEGIES, (
            f"{case['id']}: unknown stage in {pipe['stages']!r}"
        )
        assert pipe["stages"], f"{case['id']}: must record at least one stage"


def test_fixtures_carry_no_legacy_keys_secrets_or_uuids() -> None:
    banned_keys = {
        "next_agent",
        "pending_intent",
        "task_plan",
        "document_id",
        "document_ids",
        "resolved_document_id",
        "evidence_id",
        "checkpoint",
        "api_key",
    }
    uuid_re = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )

    def walk(value: object, case_id: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in banned_keys, f"{case_id}: banned key {key!r}"
                walk(item, case_id)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, case_id)
        elif isinstance(value, str):
            assert not uuid_re.search(value), f"{case_id}: UUID leaked"
            try:
                uuid.UUID(value)
                raise AssertionError(f"{case_id}: UUID-like value {value!r}")
            except (ValueError, AttributeError):
                pass

    for case in DOCUMENT_IDENTITY_CASES:
        walk(case, case["id"])

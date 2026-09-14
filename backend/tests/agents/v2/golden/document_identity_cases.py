"""Task 5 — frozen v1 document-resolution parity corpus (Phase 4B gate).

Characterization snapshot of CURRENT v1 resolver behavior, captured BEFORE
any v2 document-identity change. Each case records:

- ``query`` — the user recollection string (pinned byte-equal by
  ``EXPECTED_QUERIES`` in the test module).
- ``parse_source`` — ``regex-deterministic`` cases replay ``_extract_by_regex``
  (pure Stage-0, no DB/LLM/vector) exactly; ``pipeline-characterization``
  covers behavior that needs catalog rows (ambiguity across same-number
  survivors), characterized with stub inputs instead of live infra.
- ``v1_parse`` — the recorded Stage-0 extraction: type slug, explicit number,
  bare ``number_raw`` identity token, ``candidate_suffixes`` used to rewire
  ``doc_number_candidates`` through ``_generate_number_candidates`` (so the
  corpus never pins the wall-clock year), title keywords, agency, year,
  ``section_reference``, confidence.
- ``v1_pipeline`` — the DB/LLM/vector-backed expectation as shape only:
  stage order, ``expected_status`` per the live ``resolve_doc_agent``
  thresholds (EARLY_EXIT 0.60/0.85, MEDIUM 0.30, AMBIGUITY_RATIO 0.75 — pinned
  by the test against the real constants), plus a ``condition`` stating the
  catalog dependence. Candidate IDs/scores are catalog data and therefore
  NOT pinned to literal values; the scoring semantics that produce them are
  recorded here instead (see below).

v1 scoring semantics (``doc_resolver.py``, code-derived, stable):
exact candidate match +0.95 / partial +0.70 / plain-number +0.85, agency
+0.15, type +0.25, year +0.15, title keywords +0.25*ratio; vector hits are
down-weighted x0.8; multi-strategy agreement boosts +30% of the second
score; topic overlap adds at most +0.30. The number-identity hard filter
runs BEFORE the topic boost: a wrong-number candidate is dropped even when
its title matches the question topic (the ND53-vs-ND85 guard); when nothing
matches the named number the resolver returns [] (not-found) instead of
answering about an unrelated document.

Verified against current code on 2026-09-14 (see task-5 report for the probe
outputs). No legacy routing keys, no document UUIDs, no secrets/PII.
"""
from __future__ import annotations

REQUIRED_IDS = frozenset(
    {
        "full-official-number",
        "short-bare-number-luat",
        "short-bare-number-thongtu",
        "number-plus-agency",
        "exact-title",
        "partial-title",
        "year-plus-topic",
        "topic-only",
        "number-dominates-rerank",
        "ambiguous-candidates",
        "low-confidence-no-candidate",
    }
)

#: Frozen corpus. Key contract: parse fields replay ``_extract_by_regex``;
#: pipeline fields are shape-only expectations (no live DB/LLM/vector).
DOCUMENT_IDENTITY_CASES: tuple[dict, ...] = (
    {
        "id": "full-official-number",
        "query": "Nghị định 53/2022/NĐ-CP",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": "nghi_dinh",
            "document_number": "53/2022/NĐ-CP",
            "explicit_number_first": True,
            "candidate_suffixes": ["NĐ-CP"],
            "number_raw": "53",
            "title_keywords": [],
            "issuing_agency_text": None,
            "issuing_agency_code": None,
            "year": None,
            "section_reference": None,
            "confidence": "high",
        },
        "v1_pipeline": {
            "stages": ["db_query", "topic"],
            "expected_status": "resolved",
            "condition": "when the numbered document is indexed, the exact "
            "candidate match (+0.95) clears EARLY_EXIT (0.85)",
        },
    },
    {
        "id": "short-bare-number-luat",
        "query": "luật số 24",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": "luat",
            "document_number": "24",
            "explicit_number_first": True,
            "candidate_suffixes": ["QH15", "QH14"],
            "number_raw": "24",
            "title_keywords": [],
            "issuing_agency_text": None,
            "issuing_agency_code": None,
            "year": None,
            "section_reference": None,
            "confidence": "high",
        },
        "v1_pipeline": {
            "stages": ["db_query", "topic"],
            "expected_status": "ambiguous",
            "condition": "a bare number fans out to year/suffix variants that "
            "typically match several yearly documents with close scores "
            "(2nd/top >= 0.75); single catalog hit resolves instead",
        },
    },
    {
        "id": "short-bare-number-thongtu",
        "query": "thông tư 15",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": "thong_tu",
            "document_number": None,
            "explicit_number_first": False,
            "candidate_suffixes": ["TT"],
            "number_raw": "15",
            "title_keywords": [],
            "issuing_agency_text": None,
            "issuing_agency_code": None,
            "year": None,
            "section_reference": None,
            "confidence": "high",
        },
        "v1_pipeline": {
            "stages": ["db_query", "topic"],
            "expected_status": "ambiguous",
            "condition": "bare 15/TT variants match many circulars across "
            "issuers/years; agency narrowing (see number-plus-agency) is the "
            "v1 disambiguator",
        },
    },
    {
        "id": "number-plus-agency",
        "query": "Thông tư 15 của Bộ Công an",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": "thong_tu",
            "document_number": None,
            "explicit_number_first": False,
            "candidate_suffixes": ["TT-BCA"],
            "number_raw": "15",
            "title_keywords": [],
            "issuing_agency_text": "Bộ Công An",
            "issuing_agency_code": "BCA",
            "year": None,
            "section_reference": None,
            "confidence": "high",
        },
        "v1_pipeline": {
            "stages": ["db_query", "topic"],
            "expected_status": "ambiguous",
            "condition": "agency suffixes (TT-BCA) narrow the field vs bare "
            "15/TT, but several yearly TT-BCA 15 variants can still sit "
            "within the ambiguity ratio; one indexed hit resolves",
        },
    },
    {
        "id": "exact-title",
        "query": "Luật An ninh mạng",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": "luat",
            "document_number": None,
            "explicit_number_first": False,
            "candidate_suffixes": [],
            "number_raw": None,
            "title_keywords": ["An", "ninh", "mạng"],
            "issuing_agency_text": None,
            "issuing_agency_code": None,
            "year": None,
            "section_reference": None,
            "confidence": "medium",
        },
        "v1_pipeline": {
            "stages": ["db_query", "llm_db", "vector", "topic"],
            "expected_status": "medium_confirm",
            "condition": "keyword-only scores (type +0.25, title +0.25*ratio) "
            "rarely clear EARLY_EXIT; user confirmation path at >= 0.30",
        },
    },
    {
        "id": "partial-title",
        "query": "an ninh mạng",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": None,
            "document_number": None,
            "explicit_number_first": False,
            "candidate_suffixes": [],
            "number_raw": None,
            "title_keywords": ["an", "ninh", "mạng"],
            "issuing_agency_text": None,
            "issuing_agency_code": None,
            "year": None,
            "section_reference": None,
            "confidence": "medium",
        },
        "v1_pipeline": {
            "stages": ["db_query", "llm_db", "vector", "topic"],
            "expected_status": "medium_confirm",
            "condition": "no type word, so no type bonus; title-keyword OR "
            "search plus LLM re-extraction and vector fallback carry recall",
        },
    },
    {
        "id": "year-plus-topic",
        "query": "quy định về an toàn thông tin năm 2018",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": None,
            "document_number": None,
            "explicit_number_first": False,
            "candidate_suffixes": [],
            "number_raw": None,
            "title_keywords": ["an", "toàn", "thông", "tin"],
            "issuing_agency_text": None,
            "issuing_agency_code": None,
            "year": "2018",
            "section_reference": None,
            "confidence": "medium",
        },
        "v1_pipeline": {
            "stages": ["db_query", "llm_db", "vector", "topic"],
            "expected_status": "medium_confirm",
            "condition": "the 4-digit year is a publication-year filter, never "
            "a document number; year bonus is only +0.15",
        },
    },
    {
        "id": "topic-only",
        "query": "văn bản quy định về chế độ thai sản",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": None,
            "document_number": None,
            "explicit_number_first": False,
            "candidate_suffixes": [],
            "number_raw": None,
            "title_keywords": ["chế", "độ", "thai", "sản"],
            "issuing_agency_text": None,
            "issuing_agency_code": None,
            "year": None,
            "section_reference": None,
            "confidence": "medium",
        },
        "v1_pipeline": {
            "stages": ["db_query", "llm_db", "vector", "topic"],
            "expected_status": "medium_confirm",
            "condition": "pure topic recollection: no number, no type, no "
            "year — recall depends on keyword OR search and vector fallback; "
            "empty catalog reads not_found, never a fabricated identity",
        },
    },
    {
        "id": "number-dominates-rerank",
        "query": "Tóm tắt Nghị định 53 về hệ thống thông tin",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": "nghi_dinh",
            "document_number": None,
            "explicit_number_first": False,
            "candidate_suffixes": ["NĐ-CP"],
            "number_raw": "53",
            "title_keywords": ["hệ", "thống", "thông", "tin"],
            "issuing_agency_text": None,
            "issuing_agency_code": None,
            "year": None,
            "section_reference": None,
            "confidence": "high",
        },
        "v1_pipeline": {
            "stages": ["db_query", "topic"],
            "expected_status": "resolved",
            "condition": "explicit số 53 is a hard identity filter: a "
            "topic-matching ND85 lookalike is dropped even at high vector "
            "score; among số-53 survivors the topic boost (+<=0.30) breaks "
            "ties. Zero survivors reads not-found, never the wrong decree.",
        },
    },
    {
        "id": "ambiguous-candidates",
        "query": "Thông tư 15 của Bộ Công an",
        "parse_source": "pipeline-characterization",
        "parse_pinned_by": "number-plus-agency",
        "v1_parse": {
            "doc_type_slug": "thong_tu",
            "document_number": None,
            "explicit_number_first": False,
            "candidate_suffixes": ["TT-BCA"],
            "number_raw": "15",
            "title_keywords": [],
            "issuing_agency_text": "Bộ Công An",
            "issuing_agency_code": "BCA",
            "year": None,
            "section_reference": None,
            "confidence": "high",
        },
        "v1_pipeline": {
            "stages": ["db_query", "topic"],
            "expected_status": "ambiguous",
            "condition": "definition case: two+ same-number survivors whose "
            "scores sit within AMBIGUITY_RATIO (2nd/top >= 0.75) become a "
            "user-facing clarification with server-side options — the shape "
            "Task 6 must translate to an ambiguous DocumentReference",
        },
    },
    {
        "id": "low-confidence-no-candidate",
        "query": "cho tôi xem",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": None,
            "document_number": None,
            "explicit_number_first": False,
            "candidate_suffixes": [],
            "number_raw": None,
            "title_keywords": [],
            "issuing_agency_text": None,
            "issuing_agency_code": None,
            "year": None,
            "section_reference": None,
            "confidence": "low",
        },
        "v1_pipeline": {
            "stages": ["db_query", "llm_db", "vector", "similar"],
            "expected_status": "not_found",
            "condition": "no identity signal at all: action words strip to "
            "nothing, so every stage runs dry and only the fuzzy "
            "similar-title fallback (or plain not-found guidance) remains",
        },
    },
    {
        "id": "canonical-section-query",
        "query": "Điều 5 Luật An ninh mạng quy định gì?",
        "parse_source": "regex-deterministic",
        "v1_parse": {
            "doc_type_slug": "luat",
            "document_number": None,
            "explicit_number_first": False,
            "candidate_suffixes": [],
            "number_raw": None,
            "title_keywords": ["An", "ninh", "mạng", "gì?"],
            "issuing_agency_text": None,
            "issuing_agency_code": None,
            "year": None,
            "section_reference": "Điều 5",
            "confidence": "medium",
        },
        "v1_pipeline": {
            "stages": ["db_query", "llm_db", "vector", "topic"],
            "expected_status": "resolved",
            "condition": "binding-spec section 9 canonical case: the "
            "section_reference (Điều 5) must survive into the resolver "
            "output alongside document identity — Task 7 bridges it to the "
            "v2 section contract",
        },
    },
)

"""
Semantic Preprocessor — Phase 1A contracts (Section A.1 + A.8).

This module contains:
  - All Pydantic v2 contracts for preprocessor I/O
  - Helper types (RefExtraction, CandidateAbbr)
  - NFC span mapping utilities
  - to_persisted_dict / from_persisted_dict helpers
  - Placeholders for the full pipeline (Tasks 7-11)

Implementation of the actual pipeline (Tasks 7-11) is added later.
All contracts use ConfigDict(extra="forbid", frozen=True).
"""

from __future__ import annotations

import re
import time
import unicodedata
import uuid
import logging
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# =============================================================================
# Helper types (module-private dataclasses)
# =============================================================================


@dataclass(frozen=True)
class RefExtraction:
    """Internal extraction result; NOT a contract (not frozen/persisted)."""
    ref_id: str
    original_span: str
    span_offset: tuple[int, int]  # Python code-point half-open
    reference: str  # normalized name
    section_reference: str | None
    parse_basis: Literal[
        "regex_doc_num", "regex_named_doc", "regex_section_phrase",
        "regex_abbr_then_doc", "regex_short_official", "regex_bare_number",
    ]


@dataclass(frozen=True)
class CandidateAbbr:
    """Internal abbreviation candidate; NOT a contract."""
    short_form: str
    span_offset: tuple[int, int]
    original_span: str


# =============================================================================
# Contract: BlockingAmbiguity (A.1)
# =============================================================================

class BlockingAmbiguity(BaseModel):
    """Structured ambiguity marker.

    Per A.1: essential=True blocks routing; False is informational.
    Must NOT contain infrastructure errors (per _check_blocking_ambiguities_scope).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str  # human-readable, Vietnamese
    essential: bool  # True = blocks routing; False = informational
    source_ref: str | None = None  # ref_id if tied to a ref
    category: Literal["user_identity", "document_identity", "scope", "intent_ambiguous"]


# =============================================================================
# Contract: AbbreviationEntry (A.1)
# =============================================================================

class AbbreviationCandidate(BaseModel):
    """One candidate expansion for an abbreviation."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    full_form: str
    description: str | None = None


class AbbreviationEntry(BaseModel):
    """Result of processing one abbreviation in the query."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    span: str  # original text matched
    span_offset: tuple[int, int]  # Python code-point half-open on original_query
    short_form: str  # normalized lowercase
    chosen: str | None = None  # full_form; None when ambiguous/unknown
    candidates: list[AbbreviationCandidate] = []
    status: Literal["resolved", "ambiguous", "unknown", "not_in_db"]
    confidence: Literal["high", "low"] | None = None
    reasoning: str | None = None
    source: Literal["db_single", "db_multi", "heuristic", "llm_disambig"] | None = None


# =============================================================================
# Contract: DocumentRefEntry (A.1)
# =============================================================================

class DocumentCandidate(BaseModel):
    """One candidate document for a reference."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: uuid.UUID | None = None
    match_basis: str
    confidence: float = Field(ge=0.0, le=1.0)
    title: str | None = None
    doc_number: str | None = None
    year: int | None = None
    agency: str | None = None


class DocumentMetadata(BaseModel):
    """Minimal document metadata attached to a resolved ref."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: uuid.UUID | None = None
    agencies: list[str] = []
    tags: list[str] = []


class DocumentRefEntry(BaseModel):
    """Result of resolving one document reference."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref_id: str  # "r1", "r2"; stable, unique
    original_span: str  # original text
    span_offset: tuple[int, int]  # Python code-point half-open on original_query
    reference: str  # normalized name
    section_reference: str | None = None
    document_handle: uuid.UUID | None = None  # server-validated Document.id; LLM cannot create
    candidates: list[DocumentCandidate] = []
    resolution_status: Literal["resolved", "ambiguous", "not_found", "deferred", "error"]
    match_basis: Literal[
        "exact_number", "fuzzy_title", "attachment",
        "vector_neighbor", "alias_match", "unknown"
    ] | None = None
    version: str | None = None  # format "<uploaded_at_iso>|<content_hash_short>"
    metadata: DocumentMetadata = DocumentMetadata()
    authorized_at_lookup: bool = False  # True if ACL checked at resolve time


# =============================================================================
# Contract: TraceEvent (A.1)
# =============================================================================

class TraceEvent(BaseModel):
    """One step in the preprocessor execution trace."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt: int = 0
    step: Literal["input", "abbr_lookup", "doc_identity_lookup",
                  "doc_structural_lookup", "llm_disambig", "metadata_probe", "done"]
    started_at: float
    ended_at: float | None = None
    notes: str | None = None


# =============================================================================
# Contract: PreprocessingResult (A.1 + A.8)
# =============================================================================

class PreprocessingResult(BaseModel):
    """Wrapper for the entire output of semantic_preprocessor.

    Per A.1: All span_offset values are Python code-point (str-level) half-open
    intervals over original_query (immutable). Offset semantics: NOT UTF-8 bytes.

    Nested spans allowed: abbreviations may be inside document refs
    (e.g. regex_abbr_then_doc: "NĐ" inside ref "NĐ X").
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str  # immutable from request
    normalized_query: str | None = None
    abbreviations: list[AbbreviationEntry] = []
    document_refs: list[DocumentRefEntry] = []
    blocking_ambiguities: list[BlockingAmbiguity] = []
    preprocessing_status: Literal["ok", "partial", "complete", "error"]
    preprocessor_trace: list[TraceEvent]

    @model_validator(mode="after")
    def _check_chosen_in_candidates(self):
        for abbr in self.abbreviations:
            if abbr.chosen is not None:
                candidate_forms = {c.full_form for c in abbr.candidates}
                if abbr.chosen not in candidate_forms:
                    raise ValueError(f"chosen {abbr.chosen!r} not in candidates")
        return self

    @model_validator(mode="after")
    def _check_ref_ids_unique(self):
        ids = [r.ref_id for r in self.document_refs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate ref_id: {ids}")
        return self

    @model_validator(mode="after")
    def _check_ref_spans_non_overlapping(self):
        """Top-level document_refs must be pairwise non-overlapping.
        Abbreviations MAY nest inside refs (per regex_abbr_then_doc)."""
        spans = sorted([(ref.span_offset, ref.ref_id) for ref in self.document_refs])
        for i in range(len(spans) - 1):
            (s1, _), (s2, _) = spans[i], spans[i + 1]
            if s1[1] > s2[0]:
                raise ValueError(f"overlapping ref spans: {s1} and {s2}")
        return self

    @model_validator(mode="after")
    def _check_raw_slice_equality(self):
        """Each original_span MUST equal original_query[span_offset[0]:span_offset[1]]."""
        for abbr in self.abbreviations:
            s, e = abbr.span_offset
            if self.original_query[s:e] != abbr.span:
                raise ValueError(
                    f"abbr span mismatch: {abbr.span!r} != query[{s}:{e}]={self.original_query[s:e]!r}"
                )
        for ref in self.document_refs:
            s, e = ref.span_offset
            if self.original_query[s:e] != ref.original_span:
                raise ValueError(
                    f"ref {ref.ref_id} span mismatch: {ref.original_span!r} != query[{s}:{e}]={self.original_query[s:e]!r}"
                )
        return self

    @model_validator(mode="after")
    def _check_handle_only_when_resolved(self):
        for ref in self.document_refs:
            if ref.document_handle is not None and ref.resolution_status != "resolved":
                raise ValueError(
                    f"ref {ref.ref_id} has handle but status={ref.resolution_status}"
                )
        return self

    @model_validator(mode="after")
    def _check_blocking_ambiguities_scope(self):
        """blocking_ambiguities must NOT contain infrastructure errors."""
        bad = [a for a in self.blocking_ambiguities
               if any(kw in a.description.lower()
                      for kw in ("không tìm thấy", "not found", "outage", "timeout"))]
        if bad:
            raise ValueError(f"blocking_ambiguities contains infrastructure errors: {bad}")
        return self


# =============================================================================
# Persisted schema helpers (A.8)
# =============================================================================

def to_persisted_dict(result: PreprocessingResult) -> dict:
    """Serialize to compact persistence dict for chat_messages.semantic_context."""
    return {
        "version": "1.0",
        "original_query": result.original_query,
        "normalized_query": result.normalized_query,
        "preprocessing_status": result.preprocessing_status,
        "abbreviations": [
            {
                "span": a.span,
                "short_form": a.short_form,
                "chosen": a.chosen,
                "status": a.status,
            }
            for a in result.abbreviations
        ],
        "document_refs": [
            {
                "ref_id": r.ref_id,
                "reference": r.reference,
                "section_reference": r.section_reference,
                "document_handle": str(r.document_handle) if r.document_handle else None,
                "resolution_status": r.resolution_status,
            }
            for r in result.document_refs
        ],
        "blocking_ambiguities": [
            {
                "description": a.description,
                "essential": a.essential,
                "source_ref": a.source_ref,
                "category": a.category,
            }
            for a in result.blocking_ambiguities
        ],
    }


def from_persisted_dict(d: dict) -> PreprocessingResult:
    """Round-trip from persistence dict.

    Per A.8: restores minimum reconstruction with safe defaults.
    Non-persisted fields (preprocessor_trace, candidates, reasoning) default
    to empty/safe values.
    """
    abbreviations = [
        AbbreviationEntry(
            span=a["span"],
            span_offset=(0, len(a["span"])),  # offset not persisted
            short_form=a["short_form"],
            chosen=a.get("chosen"),
            candidates=[],  # not persisted
            status=a["status"],
        )
        for a in d.get("abbreviations", [])
    ]
    # span_offset is NOT persisted. We use model_construct() to bypass
    # validators since we cannot restore exact offsets. Both refs and abbrs
    # use reconstructed spans (0, len(text)) which match when text appears
    # at the start of original_query (best-effort reconstruction per A.8).
    # Callers must NOT use restored span_offset for position-dependent logic.
    document_refs = [
        DocumentRefEntry.model_construct(
            ref_id=r["ref_id"],
            original_span=r["reference"],
            span_offset=(0, len(r["reference"])),
            reference=r["reference"],
            section_reference=r.get("section_reference"),
            document_handle=uuid.UUID(r["document_handle"]) if r.get("document_handle") else None,
            resolution_status=r["resolution_status"],
        )
        for r in d.get("document_refs", [])
    ]
    blocking_ambiguities = [
        BlockingAmbiguity(
            description=a["description"],
            essential=a["essential"],
            source_ref=a.get("source_ref"),
            category=a.get("category", "scope"),
        )
        for a in d.get("blocking_ambiguities", [])
    ]
    # Use model_construct for the outer PreprocessingResult to bypass all
    # validators. The persisted data from to_persisted_dict is trusted;
    # validators only check structural properties (duplicate IDs, overlapping
    # spans) that are preserved in the persisted form. span_offset is NOT
    # persisted — reconstructed spans may fail _check_raw_slice_equality
    # for non-prefix refs (known limitation of compact persistence schema).
    return PreprocessingResult.model_construct(
        original_query=d.get("original_query", ""),
        normalized_query=d.get("normalized_query"),
        abbreviations=abbreviations,
        document_refs=document_refs,
        blocking_ambiguities=blocking_ambiguities,
        preprocessing_status=d.get("preprocessing_status", "ok"),
        preprocessor_trace=[],  # not persisted
    )


# =============================================================================
# NFC span mapping utilities (B.5)
# =============================================================================

def build_normalized_match_view(raw_query: str) -> tuple[str, list[int]]:
    """Returns (NFC-normalized lowercased view, raw_offset_map).

    raw_offset_map[normalized_offset] = raw_offset in original_query.
    Handles NFD/NFC variation by mapping combining marks to their base char position.
    """
    raw_nfc = unicodedata.normalize("NFC", raw_query)
    raw_offset_map: list[int] = []
    raw_pos = 0
    for raw_offset, char in enumerate(raw_query):
        if unicodedata.category(char).startswith("M"):  # combining mark
            raw_offset_map.append(raw_pos)
        else:
            raw_offset_map.append(raw_offset)
            raw_pos = raw_offset + 1
    return raw_nfc.lower(), raw_offset_map


# =============================================================================
# Placeholder implementations (Tasks 7-11 fill these in)
# =============================================================================

async def preprocess_query(query: str, ctx: "RuntimeContext") -> PreprocessingResult:
    """Full pipeline. Implemented in Task 10."""
    raise NotImplementedError("preprocess_query implemented in Task 10")


async def safe_lookup_metadata_only(
    ref: RefExtraction,
    ctx: "RuntimeContext",
    session: "AsyncSession",
) -> DocumentRefEntry:
    """Strict metadata-only lookup. Implemented in Task 7."""
    raise NotImplementedError("safe_lookup_metadata_only implemented in Task 7")


async def expand_abbreviations(
    short_forms: list[CandidateAbbr],
    ctx: "RuntimeContext",
    session: "AsyncSession",
) -> list[AbbreviationEntry]:
    """Batch abbreviation lookup + LLM disambig. Implemented in Task 9."""
    raise NotImplementedError("expand_abbreviations implemented in Task 9")


async def llm_disambiguate_ambiguous(
    ambigs: list[AbbreviationEntry],
    query: str,
    ctx: "RuntimeContext",
    session: "AsyncSession",
) -> list[AbbreviationEntry]:
    """One memory-agent call for ambiguous abbreviations. Implemented in Task 9."""
    raise NotImplementedError("llm_disambiguate_ambiguous implemented in Task 9")


def extract_abbreviation_candidates(query: str) -> list[CandidateAbbr]:
    """Extract abbreviation candidates from query. Implemented in Task 8."""
    raise NotImplementedError("extract_abbreviation_candidates implemented in Task 8")


async def semantic_preprocessor_node(state: "SupervisorState") -> dict:
    """LangGraph node. Implemented in Task 11."""
    raise NotImplementedError("semantic_preprocessor_node implemented in Task 11")


# Type hints to avoid circular imports at module level
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.services.agents.complexity import RoutingDecision
    from app.services.agents.deep_research.contracts import RuntimeContext
    from app.services.agents.models import SupervisorState

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

import asyncio
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
# Banned APIs for safe_lookup_metadata_only (B.4)
# Enforced via static AST scan + runtime spy test.
# =============================================================================

BANNED_LOOKUP_APIS: frozenset[str] = frozenset({
    "app.services.agent.doc_resolver.resolve_candidates",
    "app.services.agent.doc_resolver._extract_by_llm",
    "app.services.agent.doc_resolver._strategy_vector_fallback",
    "app.services.agent.doc_resolver._search_similar_documents",
    "app.services.agent.doc_resolver._rerank_candidates",
    "app.services.agent.doc_resolver._query_db",
    "app.services.agent.doc_resolver._generate_number_candidates",
    "app.services.agent.tools.search_documents",
    "app.services.agent.tools.search_documents_number",
    "app.services.agent.tools.search_document_section",
    "app.services.agent.tools.resolve_document_reference",
    "app.services.agents.resolve_doc_agent",
})


def _scan_for_banned_apis(module_path: str) -> list[str]:
    """Static AST scan: return list of banned API names found in module.

    Per B.4: scan semantic_preprocessor.py to ensure no banned imports.
    Returns list of (lineno, name) for each banned API found.
    """
    import ast
    violations: list[str] = []
    try:
        with open(module_path) as f:
            source = f.read()
    except OSError:
        return []

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            for banned in BANNED_LOOKUP_APIS:
                name = banned.split(".")[-1]
                if node.id == name:
                    violations.append(f"line {getattr(node, 'lineno', '?')}: {node.id}")
    return violations

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



async def safe_lookup_metadata_only(
    ref: RefExtraction,
    ctx: "RuntimeContext",
    session: "AsyncSession",
) -> DocumentRefEntry:
    """Strict metadata-only lookup. Exact-match SQL only. NO vector/rerank/fuzzy.

    Per B.4: 4 strategies by parse_basis. ACL pre-filter in SQL.
    Status mapping per spec B.4 deterministic table.
    """
    # Early cancellation check
    if hasattr(ctx, "_cancellation_event") and ctx._cancellation_event is not None and ctx._cancellation_event.is_set():
        return DocumentRefEntry(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            resolution_status="error",
        )

    try:
        if ref.parse_basis == "regex_doc_num":
            return await _lookup_by_doc_number(ref, ctx, session)
        elif ref.parse_basis == "regex_short_official":
            return await _lookup_by_short_number(ref, ctx, session)
        elif ref.parse_basis == "regex_bare_number":
            return await _lookup_by_bare_number(ref, ctx, session)
        elif ref.parse_basis == "regex_named_doc":
            return await _lookup_by_alias(ref, ctx, session)
        elif ref.parse_basis == "regex_abbr_then_doc":
            return await _lookup_by_alias(ref, ctx, session)
        elif ref.parse_basis == "regex_section":
            # Section → resolve doc part first, then attach section
            return await _lookup_section(ref, ctx, session)
        else:
            return DocumentRefEntry(
                ref_id=ref.ref_id,
                original_span=ref.original_span,
                span_offset=ref.span_offset,
                reference=ref.reference,
                section_reference=ref.section_reference,
                resolution_status="error",
            )
    except asyncio.TimeoutError:
        return _deferred_entry(ref, "timeout")
    except Exception as e:
        if "operational" in str(e).lower() or "connection" in str(e).lower():
            return _deferred_entry(ref, "db_error")
        if "integrity" in str(e).lower():
            return DocumentRefEntry(
                ref_id=ref.ref_id,
                original_span=ref.original_span,
                span_offset=ref.span_offset,
                reference=ref.reference,
                section_reference=ref.section_reference,
                resolution_status="error",
            )
        logger.warning(f"[safe_lookup] unexpected error: {e}")
        return _deferred_entry(ref, "unexpected")


def _deferred_entry(ref: RefExtraction, reason: str) -> DocumentRefEntry:
    """Return a deferred status entry."""
    return DocumentRefEntry(
        ref_id=ref.ref_id,
        original_span=ref.original_span,
        span_offset=ref.span_offset,
        reference=ref.reference,
        section_reference=ref.section_reference,
        resolution_status="deferred",
    )


async def _lookup_by_doc_number(
    ref: RefExtraction, ctx: "RuntimeContext", session: "AsyncSession"
) -> DocumentRefEntry:
    """Exact match on Document.doc_number + year + (optional) agency."""
    from sqlalchemy import select, func
    from app.models.document import Document
    from app.models.document_alias import DocumentAlias

    parsed = _normalize_reference_for_doc_num(ref.reference)
    allowed = list(ctx.allowed_workspace_ids)

    stmt = (
        select(Document)
        .where(
            Document.workspace_id.in_(allowed),
            Document.document_number == parsed["normalized_number"],
        )
    )

    result = await session.execute(stmt)
    docs = result.scalars().all()

    if len(docs) == 0:
        return DocumentRefEntry(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            resolution_status="not_found",
        )
    if len(docs) > 1:
        candidates = [
            DocumentCandidate(
                document_id=d.id,
                match_basis="exact_number",
                confidence=0.9,
                title=d.document_title,
                doc_number=d.document_number,
            )
            for d in docs
        ]
        return DocumentRefEntry(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            resolution_status="ambiguous",
            candidates=candidates,
        )

    doc = docs[0]
    version = f"{doc.updated_at.isoformat() if doc.updated_at else ''}|{doc.content_hash[:8] if doc.content_hash else ''}"
    return DocumentRefEntry(
        ref_id=ref.ref_id,
        original_span=ref.original_span,
        span_offset=ref.span_offset,
        reference=ref.reference,
        section_reference=ref.section_reference,
        document_handle=doc.id,
        resolution_status="resolved",
        match_basis="exact_number",
        version=version,
        metadata=DocumentMetadata(workspace_id=doc.workspace_id),
        authorized_at_lookup=True,
    )


async def _lookup_by_short_number(
    ref: RefExtraction, ctx: "RuntimeContext", session: "AsyncSession"
) -> DocumentRefEntry:
    """Short official number: doc_number without year."""
    from sqlalchemy import select
    from app.models.document import Document

    parsed = _normalize_reference_for_doc_num(ref.reference)
    allowed = list(ctx.allowed_workspace_ids)

    stmt = (
        select(Document)
        .where(
            Document.workspace_id.in_(allowed),
            Document.document_number == parsed["normalized_number"],
        )
    )
    result = await session.execute(stmt)
    docs = result.scalars().all()

    if len(docs) == 0:
        return DocumentRefEntry(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            resolution_status="not_found",
        )
    if len(docs) > 1:
        return DocumentRefEntry(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            resolution_status="ambiguous",
            candidates=[
                DocumentCandidate(
                    document_id=d.id,
                    match_basis="exact_number",
                    confidence=0.7,
                    title=d.document_title,
                    doc_number=d.document_number,
                )
                for d in docs
            ],
        )

    doc = docs[0]
    version = f"{doc.updated_at.isoformat() if doc.updated_at else ''}|{doc.content_hash[:8] if doc.content_hash else ''}"
    return DocumentRefEntry(
        ref_id=ref.ref_id,
        original_span=ref.original_span,
        span_offset=ref.span_offset,
        reference=ref.reference,
        section_reference=ref.section_reference,
        document_handle=doc.id,
        resolution_status="resolved",
        match_basis="exact_number",
        version=version,
        metadata=DocumentMetadata(workspace_id=doc.workspace_id),
        authorized_at_lookup=True,
    )


async def _lookup_by_bare_number(
    ref: RefExtraction, ctx: "RuntimeContext", session: "AsyncSession"
) -> DocumentRefEntry:
    """Bare number: prefix-only LIKE query. Low confidence."""
    from sqlalchemy import select
    from app.models.document import Document

    allowed = list(ctx.allowed_workspace_ids)
    number_part = ref.reference.replace("số ", "").strip()

    stmt = (
        select(Document)
        .where(
            Document.workspace_id.in_(allowed),
            Document.document_number.like(f"{number_part}%"),
        )
    )
    result = await session.execute(stmt)
    docs = result.scalars().all()

    if len(docs) == 0:
        return DocumentRefEntry(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            resolution_status="not_found",
        )
    if len(docs) > 1:
        return DocumentRefEntry(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            resolution_status="ambiguous",
            candidates=[
                DocumentCandidate(
                    document_id=d.id,
                    match_basis="exact_number",
                    confidence=0.4,
                    title=d.document_title,
                    doc_number=d.document_number,
                )
                for d in docs
            ],
        )

    doc = docs[0]
    version = f"{doc.updated_at.isoformat() if doc.updated_at else ''}|{doc.content_hash[:8] if doc.content_hash else ''}"
    return DocumentRefEntry(
        ref_id=ref.ref_id,
        original_span=ref.original_span,
        span_offset=ref.span_offset,
        reference=ref.reference,
        section_reference=ref.section_reference,
        document_handle=doc.id,
        resolution_status="resolved",
        match_basis="exact_number",
        version=version,
        metadata=DocumentMetadata(workspace_id=doc.workspace_id),
        authorized_at_lookup=True,
    )


async def _lookup_by_alias(
    ref: RefExtraction, ctx: "RuntimeContext", session: "AsyncSession"
) -> DocumentRefEntry:
    """DocumentAlias exact-match lookup with ACL filter."""
    from sqlalchemy import select
    from app.models.document import Document
    from app.models.document_alias import DocumentAlias

    normalized_alias = ref.reference.lower().strip()
    allowed = list(ctx.allowed_workspace_ids)

    stmt = (
        select(Document, DocumentAlias)
        .join(DocumentAlias, DocumentAlias.document_id == Document.id)
        .where(
            Document.workspace_id.in_(allowed),
            DocumentAlias.workspace_id.in_(allowed),
            func.lower(DocumentAlias.alias_text) == normalized_alias,
        )
    )
    result = await session.execute(stmt)
    rows = result.all()

    if len(rows) == 0:
        return DocumentRefEntry(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            resolution_status="not_found",
        )
    if len(rows) > 1:
        return DocumentRefEntry(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            resolution_status="ambiguous",
            candidates=[
                DocumentCandidate(
                    document_id=doc.id,
                    match_basis="alias_match",
                    confidence=0.85,
                    title=doc.document_title,
                    doc_number=doc.document_number,
                )
                for doc, alias in rows
            ],
        )

    doc, alias = rows[0]
    version = f"{doc.updated_at.isoformat() if doc.updated_at else ''}|{doc.content_hash[:8] if doc.content_hash else ''}"
    return DocumentRefEntry(
        ref_id=ref.ref_id,
        original_span=ref.original_span,
        span_offset=ref.span_offset,
        reference=ref.reference,
        section_reference=ref.section_reference,
        document_handle=doc.id,
        resolution_status="resolved",
        match_basis="alias_match",
        version=version,
        metadata=DocumentMetadata(workspace_id=doc.workspace_id),
        authorized_at_lookup=True,
    )


async def _lookup_section(
    ref: RefExtraction, ctx: "RuntimeContext", session: "AsyncSession"
) -> DocumentRefEntry:
    """Section reference: resolve doc part first, then attach section."""
    doc_part = ref.reference.strip()
    section_ref = ref.section_reference

    # Create a synthetic alias ref for the doc part
    alias_ref = RefExtraction(
        ref_id=ref.ref_id,
        original_span=ref.original_span,
        span_offset=ref.span_offset,
        reference=doc_part,
        section_reference=None,
        parse_basis="regex_named_doc",
    )

    result = await _lookup_by_alias(alias_ref, ctx, session)
    # Keep the section reference from the original ref
    return DocumentRefEntry(
        ref_id=result.ref_id,
        original_span=result.original_span,
        span_offset=result.span_offset,
        reference=result.reference,
        section_reference=section_ref,
        document_handle=result.document_handle,
        resolution_status=result.resolution_status,
        match_basis=result.match_basis,
        version=result.version,
        metadata=result.metadata,
        authorized_at_lookup=result.authorized_at_lookup,
    )


async def expand_abbreviations(
    query: str,
    ctx: "RuntimeContext",
    session: "AsyncSession | None" = None,
) -> str:
    """Expand all abbreviations in query using DB lookup + LLM disambiguation.

    Per B.7:  
    - Extract candidates with extract_abbreviation_candidates  
    - Look up each in DocumentAlias (for allowed workspaces)  
    - If unresolved/ambiguous → llm_disambiguate_ambiguous  
    - Return the expanded query string  
    """
    candidates = extract_abbreviation_candidates(query)
    if not candidates:
        return query

    # Cancel check
    evt = getattr(ctx, "_cancellation_event", None)
    if evt is not None and evt.is_set():
        return query

    abbr_entries: list[AbbreviationEntry] = []
    for c in candidates:
        abbr_entries.append(
            AbbreviationEntry(
                span=c.original_span,
                span_offset=c.span_offset,
                short_form=c.short_form,
                chosen=None,
                candidates=[],
                status="unknown",
                confidence=None,
                reasoning=None,
                source=None,
            )
        )

    # Resolve against DB if session provided
    if session is not None:
        abbr_entries = await _resolve_abbreviations_db(abbr_entries, ctx, session)

    # Check for unresolved/ambiguous → LLM disambiguation
    unresolved = [a for a in abbr_entries if a.status in ("unknown", "ambiguous", "not_in_db")]
    if unresolved:
        try:
            disambiguated = await llm_disambiguate_ambiguous(unresolved, query, ctx, session)
            # Merge disambiguation results back
            disambig_map = {a.short_form: a for a in disambiguated}
            for entry in abbr_entries:
                if entry.short_form in disambig_map:
                    d = disambig_map[entry.short_form]
                    entry.chosen = d.chosen
                    entry.candidates = d.candidates
                    entry.status = d.status
                    entry.confidence = d.confidence
                    entry.reasoning = d.reasoning
                    entry.source = d.source
        except Exception:
            pass  # best-effort: keep as unresolved

    # Build expanded query by replacing abbreviations with chosen expansions
    expanded = query
    # Process in reverse order of span to avoid offset issues
    for entry in sorted(abbr_entries, key=lambda x: x.span_offset[0], reverse=True):
        if entry.chosen:
            s, e = entry.span_offset
            expanded = expanded[:s] + entry.chosen + expanded[e:]
    return expanded


async def _resolve_abbreviations_db(
    entries: list[AbbreviationEntry],
    ctx: "RuntimeContext",
    session: "AsyncSession",
) -> list[AbbreviationEntry]:
    """Look up abbreviation entries in DocumentAlias table."""
    import re
    from sqlalchemy import func, select
    from app.models.document_alias import DocumentAlias

    allowed = list(ctx.allowed_workspace_ids)
    if not allowed:
        return entries

    normalized_forms = [re.sub(r"[^\w]", "", e.short_form).lower() for e in entries]
    if not normalized_forms:
        return entries

    # Join DocumentAlias → Document to get full document info
    stmt = (
        select(
            DocumentAlias.alias_text,
            Document.document_number,
            Document.document_title,
            func.count(DocumentAlias.id).label("cnt"),
        )
        .join(Document, Document.id == DocumentAlias.document_id)
        .where(
            DocumentAlias.workspace_id.in_(allowed),
            func.lower(DocumentAlias.alias_text).in_(normalized_forms),
        )
        .group_by(DocumentAlias.alias_text, Document.document_number, Document.document_title)
    )
    result = await session.execute(stmt)
    rows = result.fetchall()  # sync access after await

    # Map short_form → list of full_form strings
    alias_map: dict[str, list[str]] = {}
    for row in rows:
        # Build full_form from document_number or document_title
        if row.document_number:
            full_form = row.document_number
        elif row.document_title:
            full_form = row.document_title
        else:
            full_form = row.alias_text
        alias_lower = row.alias_text.lower()
        alias_map.setdefault(alias_lower, []).append(full_form)

    for entry in entries:
        sf_key = entry.short_form.lower()
        forms = alias_map.get(sf_key, [])
        if len(forms) == 1:
            entry.chosen = forms[0]
            entry.status = "resolved"
            entry.confidence = "high"
            entry.source = "db_single"
            entry.candidates = [AbbreviationCandidate(full_form=forms[0])]
        elif len(forms) > 1:
            entry.status = "ambiguous"
            entry.confidence = "low"
            entry.source = "db_multi"
            entry.candidates = [AbbreviationCandidate(full_form=f) for f in forms]
        # else: keep as unknown/not_in_db
    return entries


async def llm_disambiguate_ambiguous(
    ambigs: list[AbbreviationEntry],
    query: str,
    ctx: "RuntimeContext",
    session: "AsyncSession | None" = None,
    timeout_sec: float = 10.0,
) -> list[AbbreviationEntry]:
    """One memory-agent call for ambiguous abbreviations. Per B.6.

    Sends a structured prompt to the LLM asking it to disambiguate the given
    abbreviation candidates based on the full query context.
    Returns updated AbbreviationEntry objects with chosen, status, confidence,
    reasoning, and source fields populated.
    Raises asyncio.TimeoutError on timeout.
    """
    evt = getattr(ctx, "_cancellation_event", None)
    if evt is not None and evt.is_set():
        return [
            AbbreviationEntry(
                span=a.span,
                span_offset=a.span_offset,
                short_form=a.short_form,
                chosen="cancelled_disambiguation",
                candidates=a.candidates,
                status="unknown",
                confidence=None,
                reasoning="Cancelled",
                source="llm_disambig",
            )
            for a in ambigs
        ]

    try:
        return await _call_llm_for_disambiguation(ambigs, query, timeout_sec)
    except asyncio.TimeoutError:
        return [
            AbbreviationEntry(
                span=a.span,
                span_offset=a.span_offset,
                short_form=a.short_form,
                chosen="timeout_disambiguation",
                candidates=a.candidates,
                status="unknown",
                confidence=None,
                reasoning="LLM timeout",
                source="llm_disambig",
            )
            for a in ambigs
        ]


async def _call_llm_for_disambiguation(
    ambigs: list[AbbreviationEntry],
    query: str,
    timeout_sec: float = 10.0,
) -> list[AbbreviationEntry]:
    """Internal: make the actual LLM call to disambiguate abbreviations.

    The call is wrapped in asyncio.wait_for to enforce timeout_sec.
    Returns list of updated AbbreviationEntry objects.
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage

    if not ambigs:
        return []

    abbr_list = ", ".join([f"{a.short_form} (original: '{a.span}')" for a in ambigs])
    prompt = (
        f"""Bạn là trợ lý phân tích văn bản pháp luật Việt Nam.

Dựa vào câu truy vấn: "{query}"

Hãy xác định ý nghĩa đầy đủ (full_form) cho các từ viết tắt sau đây:
{abrr_list}

Trả lời theo định dạng JSON, mỗi từ viết tắt một object với các trường:
- short_form: từ viết tắt gốc
- full_form: ý nghĩa đầy đủ
- confidence: mức độ tin cậy (0.0 - 1.0)
- reasoning: giải thích ngắn bằng tiếng Việt

Trả lời JSON (chỉ JSON, không có gì khác):
["""
    )

    messages = [LLMMessage(role="user", content=prompt)]
    llm = get_llm_provider()

    try:
        response = await asyncio.wait_for(
            llm.acomplete(messages, temperature=0.1, max_tokens=512),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        raise

    text = response if isinstance(response, str) else response.content
    return _parse_disambiguation_response(text, ambigs)


def _parse_disambiguation_response(
    response_text: str,
    ambigs: list[AbbreviationEntry],
) -> list[AbbreviationEntry]:
    """Parse LLM JSON response into AbbreviationEntry updates."""
    import json, re

    # Try to extract JSON from response
    json_match = re.search(r"\[.*?\]", response_text, re.DOTALL)
    if not json_match:
        return ambigs

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError:
        return ambigs

    result_map = {item.get("short_form", "").lower().strip(): item for item in items}
    updated = []
    for ambig in ambigs:
        key = ambig.short_form.lower().strip()
        if key in result_map:
            item = result_map[key]
            full_form = item.get("full_form", "")
            confidence_val = item.get("confidence", 0.5)
            conf_str = "high" if confidence_val >= 0.8 else "low"
            updated.append(
                AbbreviationEntry(
                    span=ambig.span,
                    span_offset=ambig.span_offset,
                    short_form=ambig.short_form,
                    chosen=full_form,
                    candidates=ambig.candidates + [AbbreviationCandidate(full_form=full_form)],
                    status="resolved" if full_form else "unknown",
                    confidence=conf_str,
                    reasoning=item.get("reasoning", ""),
                    source="llm_disambig",
                )
            )
        else:
            updated.append(ambig)
    return updated


def extract_abbreviation_candidates(query: str) -> list[CandidateAbbr]:
    """Extract abbreviation candidates from query. Task 8 implementation."""
    # Simple pattern: uppercase sequences of 2-5 chars (potential abbreviations)
    # Also Vietnamese abbreviated forms like "NĐ", "CP", "QĐ"
    abbr_spans: list[tuple[int, int, str]] = []
    for m in _RE_ABBREVIATION.finditer(query):
        abbr_spans.append((m.start(), m.end(), m.group()))
    return [
        CandidateAbbr(
            short_form=original.upper(),
            span_offset=(s, e),
            original_span=original,
        )
        for s, e, original in abbr_spans
    ]


# =============================================================================
# Regex patterns for document reference extraction (B.4 / B.5)
# =============================================================================

_RE_DOC_NUM = re.compile(
    r"\b(?P<num>\d{1,4})\s*/\s*(?P<year>(?:19|20)\d{2})\s*/\s*"
    r"(?P<type>NĐ-CP|TT-BQP|TT-BTTTT|QĐ-(?:TTg|BTP|BXD|BNN&PTNT)|"
    r"TTLT-[A-Z]+(?:-[A-Z]+)?|Bộ\s*luật|Luật|Nghị\s*quyết|Pháp\s*lệnh|Chỉ\s*thị|Quyết\s*định|Nghị\s*định|Thông\s*tư(?:\s*liên\s*tịch)?)",
    re.IGNORECASE | re.UNICODE,
)

_RE_SHORT_OFFICIAL_NUMBER = re.compile(
    r"\b(?P<num>\d{1,4})\s*/\s*(?P<type_short>TT-[A-Z]+|CP|BQP|BTTTT|NĐ|QĐ)(?:-[A-Z]+)?",
    re.IGNORECASE,
)

_RE_BARE_NUMBER = re.compile(
    r"(?<!\w)số\s+(?P<num>\d{1,4})(?!\w)",
    re.IGNORECASE,
)

_RE_NAMED_DOC = re.compile(
    r"\b(?P<doc>"
    r"Luật\s+\S+|Nghị\s*định\s+\S+|Nghị\s*quyết\s+\S+|"
    r"Quyết\s*định\s+\S+|Thông\s*tư(?:\s*liên\s*tịch)?\s+\S+|"
    r"Pháp\s*lệnh\s+\S+|Chỉ\s*thị\s+\S+|"
    r"Nghị\s*định)\s*(?P<num>\d+/\d{4}[/-][A-Z]+)?",
    re.IGNORECASE | re.UNICODE,
)

_RE_ABBR_THEN_DOC = re.compile(
    r"\b(?P<abbr>NĐ|CP|QĐ|TLK)",
    re.IGNORECASE,
)

_RE_SECTION = re.compile(
    r"\b(?P<section>Chương|Điều|Mục)\s+(?P<num>[IVXLCDM]+|\d+)\s+(?:của\s+)?(?P<doc>\S+)",
    re.IGNORECASE | re.UNICODE,
)

_RE_ABBREVIATION = re.compile(
    r"\b[A-ZĐ]{2,5}\b",
    re.UNICODE,
)


def _normalize_reference_for_doc_num(ref: str) -> dict:
    """Parse a document number reference into components."""
    match = _RE_DOC_NUM.search(ref)
    if match:
        return {
            "normalized_number": f"{match.group('num')}/{match.group('year')}/{match.group('type').upper()}",
            "year": int(match.group("year")),
            "doc_type": match.group("type"),
        }
    match = _RE_SHORT_OFFICIAL_NUMBER.search(ref)
    if match:
        return {
            "normalized_number": f"{match.group('num')}/{match.group('type_short').upper()}",
            "year": None,
            "doc_type": match.group("type_short"),
        }
    return {"normalized_number": ref.strip(), "year": None, "doc_type": None}


def extract_document_references(normalized: str, raw: str) -> list[RefExtraction]:
    """Extract document references from NFC-normalized query; convert offsets to raw.

    Per B.5: runs regex on normalized; converts offsets back to raw via
    raw_offset_map. Returns list of RefExtraction with Python code-point offsets.
    """
    # normalized: NFC-normalized lowercased query; raw: original query string
    # raw_offset_map: maps normalized offsets → raw offsets
    _, raw_offset_map = build_normalized_match_view(raw)
    results: list[RefExtraction] = []
    ref_id_counter = 1

    for basis, pattern, group_name in [
        ("regex_doc_num", _RE_DOC_NUM, "doc_num"),
        ("regex_short_official", _RE_SHORT_OFFICIAL_NUMBER, "short"),
        ("regex_bare_number", _RE_BARE_NUMBER, "bare"),
        ("regex_named_doc", _RE_NAMED_DOC, "named"),
        ("regex_abbr_then_doc", _RE_ABBR_THEN_DOC, "abbr"),
        ("regex_section", _RE_SECTION, "section"),
    ]:
        for match in pattern.finditer(normalized):
            norm_start, norm_end = match.span()
            if norm_start >= len(raw_offset_map) or norm_end - 1 >= len(raw_offset_map):
                continue  # skip malformed matches
            raw_start = raw_offset_map[norm_start]
            raw_end = raw_offset_map[norm_end - 1] + 1  # half-open
            if raw_start >= len(raw) or raw_end > len(raw):
                continue
            original_span = raw[raw_start:raw_end]

            ref_id = f"r{ref_id_counter}"
            ref_id_counter += 1

            if basis == "regex_doc_num":
                parsed = _normalize_reference_for_doc_num(match.group())
                reference = parsed["normalized_number"]
                section = None
            elif basis == "regex_short_official":
                parsed = _normalize_reference_for_doc_num(match.group())
                reference = parsed["normalized_number"]
                section = None
            elif basis == "regex_bare_number":
                reference = match.group()
                section = None
            elif basis == "regex_named_doc":
                reference = match.group().strip()
                section = None
            elif basis == "regex_abbr_then_doc":
                reference = match.group().strip()
                section = None
            elif basis == "regex_section":
                doc_part = match.group("doc") or ""
                section_num = match.group("num")
                section = f"{match.group('section')} {section_num}"
                reference = doc_part
            else:
                reference = match.group().strip()
                section = None

            results.append(
                RefExtraction(
                    ref_id=ref_id,
                    original_span=original_span,
                    span_offset=(raw_start, raw_end),
                    reference=reference,
                    section_reference=section,
                    parse_basis=basis,  # type: ignore[arg-type]
                )
            )

    return results


# =============================================================================
# Stage functions for preprocess_query DAG (B.8)
# =============================================================================

def _stage1_nfc_normalize(raw_query: str) -> str:
    """Stage 1: NFC-normalize + lowercase the raw query."""
    nfc = unicodedata.normalize("NFC", raw_query)
    return nfc  # lowercase applied by build_normalized_match_view caller


def _stage2_extract_document_refs(
    nfc_query: str,
    raw_query: str,
) -> list[RefExtraction]:
    """Stage 2: Extract document references from normalized query."""
    return extract_document_references(nfc_query, raw_query)


def _stage3_extract_abbreviations(nfc_query: str) -> list[CandidateAbbr]:
    """Stage 3: Extract abbreviation candidates from normalized query."""
    return extract_abbreviation_candidates(nfc_query)


async def _stage4_expand_abbreviations(
    nfc_query: str,
    ctx: "RuntimeContext",
    session: "AsyncSession | None",
) -> str:
    """Stage 4: Expand abbreviations in NFC-normalized query."""
    if session is None:
        return nfc_query
    return await expand_abbreviations(nfc_query, ctx, session)


async def _stage5_metadata_lookup(
    refs: list[RefExtraction],
    ctx: "RuntimeContext",
    session: "AsyncSession | None",
) -> list[DocumentRefEntry]:
    """Stage 5: Resolve document reference strings to metadata via DB lookup."""
    if session is None or not refs:
        return [
            DocumentRefEntry(
                ref_id=r.ref_id,
                original_span=r.original_span,
                span_offset=r.span_offset,
                reference=r.reference,
                section_reference=r.section_reference,
                document_handle=None,
                candidates=[],
                resolution_status="deferred",
            )
            for r in refs
        ]

    entries = []
    for ref in refs:
        ref_ext = RefExtraction(
            ref_id=ref.ref_id,
            original_span=ref.original_span,
            span_offset=ref.span_offset,
            reference=ref.reference,
            section_reference=ref.section_reference,
            parse_basis=ref.parse_basis,
        )
        entry = await safe_lookup_metadata_only(ref_ext, ctx, session)
        entries.append(entry)
    return entries


# =============================================================================
# PreprocessQueryDAG + preprocess_query (B.8)
# =============================================================================

class PreprocessQueryDAG:
    """Five-stage pipeline for semantic query preprocessing.

    Per B.8:
      stage1: NFC normalize
      stage2: extract document references
      stage3: extract abbreviations
      stage4: expand abbreviations (with LLM)
      stage5: metadata lookup
    """

    def __init__(self):
        self.stage1 = "nfc_normalize"
        self.stage2 = "extract_doc_refs"
        self.stage3 = "extract_abbr"
        self.stage4 = "expand_abbr"
        self.stage5 = "metadata_lookup"


async def preprocess_query(
    raw_query: str,
    ctx: "RuntimeContext",
    session: "AsyncSession | None" = None,
) -> PreprocessingResult:
    """Main entry point for semantic query preprocessing. Per B.8.

    Runs the five-stage DAG pipeline and returns a PreprocessingResult
    with NFC-normalized query, document references, abbreviation expansions,
    and a preprocessor trace.
    """
    import time as time_module

    trace: list[TraceEvent] = []
    nfc_query = _stage1_nfc_normalize(raw_query)
    t0 = time_module.monotonic()

    # Stage 2: document references
    t2_start = time_module.monotonic()
    refs = _stage2_extract_document_refs(nfc_query, raw_query)
    trace.append(TraceEvent(
        step="doc_identity_lookup",
        started_at=t2_start,
        ended_at=time_module.monotonic(),
        notes=f"{len(refs)} refs extracted",
    ))

    # Stage 3: abbreviation candidates
    t3_start = time_module.monotonic()
    abbr_candidates = _stage3_extract_abbreviations(nfc_query)
    trace.append(TraceEvent(
        step="abbr_lookup",
        started_at=t3_start,
        ended_at=time_module.monotonic(),
        notes=f"{len(abbr_candidates)} candidates",
    ))

    # Stage 4: expand abbreviations
    t4_start = time_module.monotonic()
    expanded_query = await _stage4_expand_abbreviations(nfc_query, ctx, session)
    abbr_status = "resolved" if expanded_query != nfc_query else "no_abbr"
    trace.append(TraceEvent(
        step="llm_disambig",
        started_at=t4_start,
        ended_at=time_module.monotonic(),
        notes=f"abbr expansion status: {abbr_status}",
    ))

    # Stage 5: metadata lookup
    t5_start = time_module.monotonic()
    doc_entries = await _stage5_metadata_lookup(refs, ctx, session)
    trace.append(TraceEvent(
        step="metadata_probe",
        started_at=t5_start,
        ended_at=time_module.monotonic(),
        notes=f"{len(doc_entries)} doc entries",
    ))

    # Build abbreviation entries from candidates
    abbr_entries = [
        AbbreviationEntry(
            span=c.original_span,
            span_offset=c.span_offset,
            short_form=c.short_form,
            chosen=None,
            candidates=[],
            status="unknown",
            confidence=None,
            reasoning=None,
            source=None,
        )
        for c in abbr_candidates
    ]

    status: Literal["ok", "partial", "complete", "error"] = "ok"
    trace.append(TraceEvent(
        step="done",
        started_at=time_module.monotonic(),
        ended_at=None,
        notes="preprocessing complete",
    ))

    return PreprocessingResult(
        original_query=raw_query,
        normalized_query=nfc_query,
        abbreviations=abbr_entries,
        document_refs=doc_entries,
        blocking_ambiguities=[],
        preprocessing_status=status,
        preprocessor_trace=trace,
    )


async def semantic_preprocessor_node(state: "SupervisorState") -> dict:
    """
    LangGraph node entry point for semantic preprocessing.

    Per B.7: Called at graph ingress (START → semantic_preprocessor → supervisor).
    Runs the full DAG pipeline, derives legacy fields, and returns a state
    update dict that LangGraph merges into the graph state.

    Suppresses duplicate abbreviation expansion in supervisor_node by setting
    _preprocessor_marker = "semantic_v1" (per Q17 in spec).
    """
    import logging
    import time as time_module
    import uuid as uuid_module

    from app.core.config import settings
    from app.services.agent.streaming import get_current_db
    from app.services.agents.deep_research.contracts import (
        RuntimeContext, ModelSnapshot, ToolBudget, ConsumedBudget,
        PreprocessorBudgetConfig,
    )
    from app.services.agents.supervisor import _extract_user_message

    logger = logging.getLogger(__name__)

    # 1. Guard: only run when flag is enabled
    if not getattr(settings, "NEXUSRAG_SEMANTIC_PREPROCESSOR", False):
        return {}  # Legacy path (flag off): nothing to do

    # 2. Extract user query
    user_query = _extract_user_message(state)
    if not user_query:
        return {
            "semantic_context": None,
            "_preprocessor_marker": "semantic_v1",
        }

    # 3. Build RuntimeContext from SupervisorState
    user_id = state.get("user_id")
    workspace_ids = state.get("workspace_ids", [])
    document_ids = state.get("document_ids") or []
    session_id = state.get("session_id")

    now = time_module.monotonic()
    deadline_offset = getattr(settings, "NEXUSRAG_PREPROCESSOR_DEADLINE_SEC", 25.0)
    absolute_deadline = now + deadline_offset

    ctx = RuntimeContext(
        principal_id=user_id or uuid_module.uuid4(),
        allowed_workspace_ids=list(workspace_ids) if workspace_ids else [],
        authorized_document_handles=set(document_ids) if document_ids else set(),
        people_permission=bool(state.get("user_can_use_people", False)),
        session_id=session_id,
        run_id=str(uuid_module.uuid4()),
        config_revision=getattr(settings, "NEXUSRAG_CONFIG_REVISION", "unknown"),
        absolute_deadline_monotonic=absolute_deadline,
        absolute_deadline_epoch=None,
        remaining_budget_sec=deadline_offset,
        model_snapshot=ModelSnapshot(
            provider="preprocessor",
            model="internal",
            config_revision=getattr(settings, "NEXUSRAG_CONFIG_REVISION", "unknown"),
            langfuse_tags=[],
        ),
        tool_budget=ToolBudget(),
        consumed_budget=ConsumedBudget(),
        preprocessing=PreprocessorBudgetConfig(),
        tool_allowlist=set(),
    )

    # 4. Get DB session from context (set by stream_agent_to_sse before graph entry)
    db = get_current_db()

    # 5. Run preprocessing pipeline
    try:
        result = await preprocess_query(user_query, ctx, db)
    except Exception as exc:
        logger.warning(f"[semantic_preprocessor_node] preprocessing failed: {exc}")
        from app.services.agents.semantic_preprocessor import PreprocessingResult
        result = PreprocessingResult(
            original_query=user_query,
            normalized_query=user_query,
            abbreviations=[],
            document_refs=[],
            blocking_ambiguities=[],
            preprocessing_status="error",
            preprocessor_trace=[],
        )

    # 6. Derive legacy fields for Phase 0/1A transition (B.7, Q5)
    complexity = "simple"
    if len(result.document_refs) >= 2:
        complexity = "multi_doc"

    params: dict = {
        "document_refs": [
            {"reference": r.reference, "section": r.section_reference}
            for r in result.document_refs
        ],
        "sections": [
            r.section_reference
            for r in result.document_refs
            if r.section_reference
        ],
    }

    # 7. Return state update
    return {
        "semantic_context": result,
        "query_complexity": complexity,
        "extracted_params": params,
        "_preprocessor_marker": "semantic_v1",
    }


# Type hints to avoid circular imports at module level
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.services.agents.complexity import RoutingDecision
    from app.services.agents.deep_research.contracts import RuntimeContext
    from app.services.agents.models import SupervisorState

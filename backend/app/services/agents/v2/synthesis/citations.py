"""The single CitationProjector owner (spec §11.1–11.4).

``GroundedClaim -> EvidenceUseRef -> EvidenceRecord.source -> PublicCitation``
is the only path that fabricates citation metadata; rendering, SSE,
persistence, and history reload all consume this projection.

- ``DocumentSourceIdentity`` projects through the authoritative
  document/revision/locator stores under the current scope.
- ``DerivedSourceIdentity`` recursively expands to authorized locatable
  lineage, deduplicated in deterministic lineage order; a bare derived
  citation is invalid.
- ``PeopleSourceIdentity``/``KnowledgeGraphSourceIdentity`` (and memory) are
  never projected into document synthesis — they fail closed with
  ``citation_unresolvable`` rather than being disguised as documents.

Public metadata is the §11.3 allowlist only; internal evidence/use/task/run/
workspace/binding/revision/checkpoint identities never cross. The public
``index`` is a deterministic four-character alphanumeric with at least one
letter, derived from the source-identity hash with deterministic ordinal
collision resolution — never random like the V1 generator.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID

from ..contracts.evidence import (
    DerivedSourceIdentity,
    DocumentSourceIdentity,
    EvidenceSourceIdentity,
)
from ..contracts.locators import (
    ArticleLocator,
    ChunkRangeLocator,
    DocumentLocator,
    PageRangeLocator,
    SectionLocator,
)
from ..contracts.synthesis import GroundedClaim, PublicCitation
from ..nodes.evaluate import HydratedEvidence

__all__ = [
    "CitationProjectionError",
    "CitationProjection",
    "CitationResolver",
    "ResolvedDocumentCitation",
    "CitationProjector",
    "StoreCitationResolver",
]

#: Spec §11.4: at most three public indexes render for one claim.
MAX_INDEXES_PER_CLAIM = 3

#: V1-compatible charset: lowercase alphanumeric, at least one letter.
_INDEX_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_INDEX_LEN = 4

#: Bounded public excerpt length for the allowlisted ``content`` field.
_EXCERPT_MAX_CHARS = 800


class CitationProjectionError(ValueError):
    """A closed-code citation failure; the candidate enters repair/failure.

    ``code`` is ``citation_unresolvable`` — the only projection failure this
    phase emits. The message is content-free.
    """

    def __init__(self, code: str = "citation_unresolvable") -> None:
        self.code = code
        super().__init__(f"citation projection failed: {code}")


@dataclass(frozen=True)
class ResolvedDocumentCitation:
    """The allowlisted public metadata for one locatable document source.

    Everything here comes from authoritative server stores (document row,
    revision manifest, structure artifact) — never from model output.
    """

    document_id: str
    document_revision: str
    chunk_id: str | None = None
    content: str | None = None
    source_file: str | None = None
    page_no: int | None = None
    heading_path: tuple[str, ...] | None = None
    document_number: str | None = None
    article_label: str | None = None
    validity_status: str | None = None
    superseded_by: str | None = None
    document_title: str | None = None


@runtime_checkable
class CitationResolver(Protocol):
    """Server-side dependency boundary for citation metadata.

    Implementations resolve document sources through the authoritative
    document/revision/locator stores under the current scope and expand
    derived lineage through the evidence store. ``None`` results are
    unresolvable — the projector fails closed.
    """

    async def resolve_document(
        self, source: DocumentSourceIdentity, *, content: str
    ) -> ResolvedDocumentCitation | None:
        """Resolve one document source to its public metadata."""
        ...

    async def resolve_lineage(
        self, evidence_id: UUID
    ) -> EvidenceSourceIdentity | None:
        """Resolve one lineage evidence id to its typed source identity."""
        ...


@dataclass(frozen=True)
class CitationProjection:
    """The projector output: allowlisted citations + per-claim index order."""

    citations: tuple[PublicCitation, ...]
    claim_indexes: dict[str, tuple[str, ...]]


def _index_for(key: str, taken: set[str]) -> str:
    """Deterministic 4-char index: source-identity hash + ordinal suffix.

    The first candidate is ``sha256(key)`` mapped onto the V1 alphabet; when
    it collides (or carries no letter) the digest is re-hashed with a
    deterministic ordinal suffix until a free, letter-bearing index stands.
    """
    ordinal = 0
    while True:
        seed = key if ordinal == 0 else f"{key}:{ordinal}"
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        index = "".join(
            _INDEX_ALPHABET[b % len(_INDEX_ALPHABET)]
            for b in digest[:_INDEX_LEN]
        )
        if any(c.isalpha() for c in index) and index not in taken:
            return index
        ordinal += 1


def _source_key(source: DocumentSourceIdentity) -> str:
    """The dedup/identity key for one public source (never emitted)."""
    return (
        f"{source.document_id}:{source.document_revision}:"
        f"{source.locator.model_dump_json()}"
    )


def _clip_excerpt(content: str | None) -> str | None:
    if content is None:
        return None
    clipped = content.strip()
    if len(clipped) <= _EXCERPT_MAX_CHARS:
        return clipped
    return clipped[: _EXCERPT_MAX_CHARS - 1].rstrip() + "…"


def _label_for(resolved: ResolvedDocumentCitation) -> str:
    """Deterministic public label, mirroring the V1 label order."""
    if resolved.article_label and resolved.document_number:
        return f"{resolved.article_label} — {resolved.document_number}"
    if resolved.document_number:
        return resolved.document_number
    if resolved.article_label:
        return resolved.article_label
    if resolved.source_file:
        return resolved.source_file
    if resolved.document_title:
        return resolved.document_title
    return f"Tài liệu {resolved.document_id[:8]}"


class CitationProjector:
    """The single owner of the public citation projection (spec §11.1)."""

    def __init__(self, resolver: CitationResolver) -> None:
        self._resolver = resolver

    async def _expand(
        self,
        source: EvidenceSourceIdentity,
        *,
        content: str,
        seen: set[str],
        out: list[tuple[str, ResolvedDocumentCitation]],
        resolved_cache: dict[str, ResolvedDocumentCitation],
        depth: int = 0,
    ) -> None:
        """Expand one source identity to locatable document citations.

        Derived identities recurse through their lineage in stored order;
        dedup is by public source key in first-encountered (deterministic
        lineage) order. People/KG/memory identities and unresolvable lineage
        fail closed — they are never disguised as document citations.
        """
        if depth > 16:
            raise CitationProjectionError()
        if isinstance(source, DocumentSourceIdentity):
            key = _source_key(source)
            if key in seen:
                return
            resolved = resolved_cache.get(key)
            if resolved is None:
                resolved = await self._resolver.resolve_document(
                    source, content=content
                )
                if resolved is None:
                    raise CitationProjectionError()
                resolved_cache[key] = resolved
            seen.add(key)
            out.append((key, resolved))
            return
        if isinstance(source, DerivedSourceIdentity):
            if not source.source_evidence_ids:
                raise CitationProjectionError()
            for evidence_id in source.source_evidence_ids:
                ancestor = await self._resolver.resolve_lineage(evidence_id)
                if ancestor is None:
                    raise CitationProjectionError()
                await self._expand(
                    ancestor,
                    content=content,
                    seen=seen,
                    out=out,
                    resolved_cache=resolved_cache,
                    depth=depth + 1,
                )
            return
        # People/KG/memory sources are never projected into document
        # synthesis (spec §11.2); typed unavailable upstream, fail closed here.
        raise CitationProjectionError()

    async def project(
        self,
        claims: tuple[GroundedClaim, ...],
        evidence: tuple[HydratedEvidence, ...],
    ) -> CitationProjection:
        """Project grounded claims to allowlisted public citations.

        Ordering follows first claim use, then cited evidence order; a
        repeated public source reuses its index; at most three indexes render
        per claim.
        """
        by_use_id = {item.use_id: item for item in evidence}
        citations: list[PublicCitation] = []
        by_key: dict[str, PublicCitation] = {}
        taken: set[str] = set()
        claim_indexes: dict[str, tuple[str, ...]] = {}
        resolved_cache: dict[str, ResolvedDocumentCitation] = {}

        for claim in claims:
            expanded: list[tuple[str, ResolvedDocumentCitation]] = []
            seen_keys: set[str] = set()
            for ref in claim.uses:
                item = by_use_id.get(ref.use_id)
                if item is None:
                    raise CitationProjectionError()
                await self._expand(
                    item.source_identity,
                    content=item.content,
                    seen=seen_keys,
                    out=expanded,
                    resolved_cache=resolved_cache,
                )
            indexes: list[str] = []
            for key, resolved in expanded:
                citation = by_key.get(key)
                if citation is None:
                    index = _index_for(key, taken)
                    taken.add(index)
                    citation = PublicCitation(
                        citation_id=index,
                        index=index,
                        label=_label_for(resolved),
                        source_type="vector",
                        document_id=resolved.document_id,
                        chunk_id=resolved.chunk_id,
                        content=_clip_excerpt(resolved.content),
                        source_file=resolved.source_file,
                        page_no=resolved.page_no,
                        heading_path=resolved.heading_path,
                        document_number=resolved.document_number,
                        article_label=resolved.article_label,
                        validity_status=resolved.validity_status,
                        superseded_by=resolved.superseded_by,
                    )
                    by_key[key] = citation
                    citations.append(citation)
                indexes.append(citation.index)
            claim_indexes[claim.claim_id] = tuple(
                indexes[:MAX_INDEXES_PER_CLAIM]
            )

        return CitationProjection(
            citations=tuple(citations), claim_indexes=claim_indexes
        )


# ---------------------------------------------------------------------------
# Production resolver over the authoritative stores
# ---------------------------------------------------------------------------


class StoreCitationResolver:
    """Concrete resolver over the document/evidence stores.

    ``session_factory`` opens short-lived sessions on the primary database
    (documents, revisions, evidence records share it); ``storage`` is the
    artifact store used to read revision structure payloads (defaults to the
    MinIO singleton, injectable for tests).
    """

    def __init__(self, session_factory, *, storage=None) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._chunk_cache: dict[str, dict[str, object]] = {}

    async def resolve_document(
        self, source: DocumentSourceIdentity, *, content: str
    ) -> ResolvedDocumentCitation | None:
        from app.models.document import Document
        from ..persistence.document_views import (
            RevisionNotReady,
            load_revision_chunks,
            load_revision_identity,
        )

        try:
            revision_id = UUID(str(source.document_revision))
        except (ValueError, AttributeError):
            return None
        async with self._session_factory() as db:
            document = await db.get(Document, source.document_id)
            if document is None or document.source_deleted_at is not None:
                return None
            try:
                identity = await load_revision_identity(db, revision_id)
            except RevisionNotReady:
                return None
            if identity.document_id != source.document_id:
                return None
            chunks = await self._chunks(db, identity)
        return self._project(source, document, chunks, content)

    async def _chunks(self, db, identity) -> dict[str, object]:
        """chunk_id -> ChunkRecord for the pinned revision (cached)."""
        from ..persistence.document_views import load_revision_chunks

        key = str(identity.revision_id)
        cached = self._chunk_cache.get(key)
        if cached is None:
            records = await load_revision_chunks(
                db, identity, storage=self._storage
            )
            cached = {record.chunk_id: record for record in records}
            self._chunk_cache[key] = cached
        return cached

    def _project(
        self,
        source: DocumentSourceIdentity,
        document,
        chunks: dict[str, object],
        content: str,
    ) -> ResolvedDocumentCitation | None:
        locator = source.locator
        chunk = None
        chunk_id: str | None = None
        page_no: int | None = None
        if isinstance(locator, ChunkRangeLocator):
            chunk_id = locator.start
            chunk = chunks.get(locator.start)
            if chunk is None:
                return None
        elif isinstance(locator, SectionLocator):
            chunk_id = locator.structure_node_id
            chunk = chunks.get(locator.structure_node_id)
            if chunk is None:
                return None
        elif isinstance(locator, ArticleLocator):
            chunk_id = locator.structure_node_id
            chunk = chunks.get(locator.structure_node_id)
            if chunk is None:
                return None
        elif isinstance(locator, PageRangeLocator):
            page_no = locator.start
        elif not isinstance(locator, DocumentLocator):
            return None

        heading_path: tuple[str, ...] | None = None
        source_file: str | None = None
        article_label: str | None = None
        if chunk is not None:
            page_no = chunk.page_no or page_no
            heading_path = tuple(chunk.heading_path) or None
            source_file = chunk.source_file or None
            if heading_path:
                from app.services.parsing.heading_path import (
                    extract_article_nos,
                )

                article_nos = extract_article_nos(list(heading_path))
                if article_nos:
                    article_label = ", ".join(
                        f"Điều {n}" for n in article_nos[:3]
                    )
        if article_label is None and isinstance(locator, ArticleLocator):
            article_label = (
                f"Điều {locator.article_id}"
                if locator.article_id.isdigit()
                else locator.article_id
            )
        if source_file is None:
            source_file = (
                getattr(document, "original_filename", None)
                or getattr(document, "filename", None)
                or None
            )
        return ResolvedDocumentCitation(
            document_id=str(source.document_id),
            document_revision=source.document_revision,
            chunk_id=chunk_id,
            content=content,
            source_file=source_file,
            page_no=page_no,
            heading_path=heading_path,
            document_number=document.document_number,
            article_label=article_label,
            validity_status=document.validity_status,
            superseded_by=document.superseded_by_number,
            document_title=document.document_title,
        )

    async def resolve_lineage(
        self, evidence_id: UUID
    ) -> EvidenceSourceIdentity | None:
        from ..persistence.evidence import EvidenceRepository

        async with self._session_factory() as db:
            repository = EvidenceRepository(db)
            record = await repository.load_record(evidence_id)
            if record is None:
                return None
            return record.source

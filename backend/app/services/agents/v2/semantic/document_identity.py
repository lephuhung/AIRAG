"""Runtime-only v1 document-identity adapter (Phase 4B, Task 6).

Wraps the proven v1 ``resolve_candidates()`` pipeline
(``app.services.agent.doc_resolver``) and translates its ranked candidates
into v2 ``DocumentReference`` identity facts. The v1 regex/SQL/LLM/vector/
fuzzy/rerank code is reused by call, never cloned here.

Ownership rules enforced here:

- The full contextualized user question is always passed as ``topic``; the
  bare reference span alone never drives disambiguation.
- v1 decision precedence is honoured (``EARLY_EXIT`` binds before the
  close-second rule is considered). A clear winner becomes ``resolved``;
  close candidates become ``ambiguous`` only when at least two distinct
  usable ids survive (the frozen validator rejects 1-candidate
  ambiguity); a lone medium-confidence hit reads ``unresolved`` — still
  actionable, never force-bound. Not-found stays not-found and
  low-confidence candidates are never force-bound.
- Only candidate document UUIDs are projected into the v2 contract.
  Candidate titles/scores/strategies and resolver internals stay out of
  model/frontend projections (no secrets/PII/internal evidence UUIDs).
- The resolver searches only the authorized workspace scope it is given.
- This adapter never pins revisions: ``revision_requirement`` and
  ``requested_role`` pass through untouched, and the existing v2 binding
  resolver remains the only revision-pin authority.
- Expensive resolver work is cached per request on the
  ``DocumentIdentityResolver`` instance, so repeated semantic builds do
  not repeat resolution. The resolver never dispatches capabilities.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Sequence
from uuid import UUID

from app.services.agent.doc_resolver import resolve_candidates
from app.services.agents.resolve_doc_agent import (
    AMBIGUITY_RATIO,
    EARLY_EXIT_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)

from ..contracts.semantic import DocumentReference, DocumentResolutionStatus

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentIdentityError",
    "DocumentIdentityResolver",
    "normalize_section_label",
    "reference_from_candidates",
]

#: How many top candidates an ambiguous reference carries for clarification.
_MAX_AMBIGUOUS_CANDIDATES = 5

#: Authoritative one-turn section locator (Phase 4B, Task 7 only).
#:
#: A single ``Điều/Chương/Khoản/Mục/Phụ lục`` coordinate with no
#: multi-turn coreference. Fullmatch by design: a compound or free-text
#: span (multi-locator, discourse anaphora) is NOT authoritative here and
#: stays ``None`` for Phase 4C. Mirrors the v1 ``_SECTION_PATTERNS``
#: vocabulary, never v1 logic.
_SECTION_LOCATOR_RE = re.compile(
    r"(?:điều|chương|khoản|mục|phụ\s*lục)\s+[\dIVXivx]+(?:\.\d+)*",
    re.IGNORECASE,
)


def normalize_section_label(raw: object) -> str | None:
    """Project a raw section span onto an authoritative one-turn label.

    Returns the whitespace-collapsed label (``Điều 5``) when ``raw`` is a
    single authoritative locator, else ``None``. Advisory-only: the label
    carries no revision coordinate (``structure_node_id`` stays ``None``),
    so routing must use the bounded document fallback, never
    ``section.read``, for a label-only locator.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    label = " ".join(raw.split())
    if _SECTION_LOCATOR_RE.fullmatch(label) is None:
        return None
    return label


class DocumentIdentityError(ValueError):
    """A document reference cannot be resolved to v2 identity facts."""


def _candidate_uuids(candidates: Sequence[dict]) -> list[UUID]:
    """Project ranked candidates onto valid document UUIDs, in rank order.

    Non-UUID/blank ids are dropped (never fabricated into identity); order
    is preserved and duplicates collapsed.
    """
    seen: set[UUID] = set()
    ordered: list[UUID] = []
    for candidate in candidates:
        raw = candidate.get("document_id") if isinstance(candidate, dict) else None
        if not raw:
            continue
        try:
            value = UUID(str(raw))
        except (ValueError, AttributeError):
            continue
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _with_identity(
    reference: DocumentReference,
    status: DocumentResolutionStatus,
    resolved_id: UUID | None,
    candidate_ids: tuple[UUID, ...],
) -> DocumentReference:
    """Project identity facts onto the caller's own reference shell."""
    return reference.model_copy(
        update={
            "resolution_status": status,
            "resolved_document_id": resolved_id,
            "candidate_document_ids": candidate_ids,
        }
    )


def reference_from_candidates(
    reference: DocumentReference,
    candidates: Sequence[dict],
) -> DocumentReference:
    """Translate ranked v1 candidates into a v2 identity fact.

    Thresholds and precedence are the live ``resolve_doc_agent`` constants
    (imported, not copied): the ``EARLY_EXIT`` check runs first exactly as
    v1 evaluates it, then the ``AMBIGUITY_RATIO`` close-second rule, then
    ``HIGH``/``MEDIUM``. ``ambiguous`` is emitted only when at least two
    distinct usable candidate UUIDs survive (the frozen validator rejects
    1-candidate ambiguity): a lone ``MEDIUM`` hit, or a close second with
    only one usable id, reads ``unresolved`` — still actionable, never a
    force-bind and never contract-invalid. Anything below ``MEDIUM``
    (including an empty list or zero usable UUIDs) stays ``not_found`` so
    a low-confidence guess can never become a binding.
    """
    ranked = list(candidates or [])
    if not ranked:
        return _with_identity(reference, "not_found", None, ())
    top_score = float(ranked[0].get("score", 0.0) or 0.0)
    second_score = float(ranked[1].get("score", 0.0) or 0.0) if len(ranked) > 1 else 0.0
    # v1 precedence: the early exit binds before ambiguity is considered.
    if top_score >= EARLY_EXIT_THRESHOLD:
        top_uuids = _candidate_uuids(ranked[:1])
        if top_uuids:
            return _with_identity(reference, "resolved", top_uuids[0], ())
        # Top id unusable: fall through to the ambiguity/unresolved rules.
    uuids = tuple(_candidate_uuids(ranked[:_MAX_AMBIGUOUS_CANDIDATES]))
    is_ambiguous = (
        len(ranked) > 1
        and (second_score / max(top_score, 0.01)) >= AMBIGUITY_RATIO
    )
    if is_ambiguous or (MEDIUM_CONFIDENCE_THRESHOLD <= top_score < HIGH_CONFIDENCE_THRESHOLD):
        if len(set(uuids)) >= 2:
            return _with_identity(reference, "ambiguous", None, uuids)
        if uuids and top_score >= MEDIUM_CONFIDENCE_THRESHOLD:
            return _with_identity(reference, "unresolved", None, ())
        return _with_identity(reference, "not_found", None, ())
    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        top_uuids = _candidate_uuids(ranked[:1])
        if top_uuids:
            return _with_identity(reference, "resolved", top_uuids[0], ())
        return _with_identity(reference, "not_found", None, ())
    return _with_identity(reference, "not_found", None, ())


class DocumentIdentityResolver:
    """Request-scoped v1 identity wrapper with per-turn result cache.

    One instance lives for one request/turn (constructed once by the
    ingress owner alongside ``IntentClassifier``). ``resolve_reference()``
    wraps ``resolve_candidates()`` with the full question as ``topic`` and
    translates the outcome via ``reference_from_candidates()``; the cache
    stores only the translated identity facts (status, resolved id,
    candidate ids) by ``(reference span, topic, workspace scope)`` and
    re-applies them to each caller's own reference, so same-span refs keep
    their ``ref_id``/role/revision requirement. Repeated semantic builds
    resolve once. Concurrent callers share one in-flight resolution per
    key (single-flight). The resolver performs no capability dispatch and
    never imports supervisor/graph state.
    """

    def __init__(self) -> None:
        self._cache: dict[
            tuple[str, str, tuple[str, ...], bool],
            tuple[str, UUID | None, tuple[UUID, ...]],
        ] = {}
        # Advisory one-turn section labels per cache key (Task 7 bridge:
        # the v1 top-level ``section_reference`` is preserved alongside
        # identity facts, never projected into identity itself).
        self._sections: dict[tuple[str, str, tuple[str, ...], bool], str | None] = {}
        self._locks: dict[tuple[str, str, tuple[str, ...], bool], asyncio.Lock] = {}

    @staticmethod
    def _cache_key(
        reference_text: str,
        topic: str,
        workspace_ids: Sequence[Any],
        use_llm_fallback: bool,
    ) -> tuple[str, str, tuple[str, ...], bool]:
        return (
            (reference_text or "").strip(),
            (topic or "").strip(),
            tuple(str(w) for w in workspace_ids),
            bool(use_llm_fallback),
        )

    @property
    def cache_size(self) -> int:
        """Number of cached reference resolutions (per request/turn)."""
        return len(self._cache)

    def clear(self) -> None:
        """Drop all cached resolutions (turn boundary)."""
        self._cache.clear()
        self._sections.clear()

    def cached_section_label(
        self,
        reference: DocumentReference,
        *,
        question: str,
        workspace_ids: Sequence[Any],
        use_llm_fallback: bool = True,
    ) -> str | None:
        """Return the preserved one-turn section label for a prior resolution.

        ``None`` when the reference was never resolved through this
        instance or the v1 result carried no authoritative locator.
        Never resolves: call only after ``resolve_reference``.
        """
        topic = (question or "").strip()
        scope = tuple(workspace_ids or ())
        reference_text = (
            reference.normalized_reference or reference.original_span or ""
        ).strip()
        key = self._cache_key(reference_text, topic, scope, use_llm_fallback)
        return self._sections.get(key)

    async def resolve_reference(
        self,
        reference: DocumentReference,
        *,
        question: str,
        workspace_ids: Sequence[Any],
        db: Any,
        use_llm_fallback: bool = True,
    ) -> DocumentReference:
        """Resolve one draft reference to a v2 identity fact (cached)."""
        topic = (question or "").strip()
        if not topic:
            raise DocumentIdentityError(
                "identity resolution requires the contextualized question "
                "as topic; refusing to resolve from a bare span"
            )
        scope = tuple(workspace_ids or ())
        if not scope:
            raise DocumentIdentityError(
                "identity resolution requires at least one authorized "
                "workspace; refusing to search outside a tenant scope"
            )
        reference_text = (
            reference.normalized_reference or reference.original_span or ""
        ).strip()
        key = self._cache_key(
            reference_text, topic, scope, use_llm_fallback
        )
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            hit = self._cache.get(key)
            if hit is not None:
                status, resolved_id, candidate_ids = hit
                return _with_identity(reference, status, resolved_id, candidate_ids)  # type: ignore[arg-type]
            result = await resolve_candidates(
                reference_text or topic,
                list(scope),
                db,
                topic=topic,
                use_llm_fallback=use_llm_fallback,
            )
            self._sections[key] = normalize_section_label(
                result.get("section_reference")
            )
            resolved = reference_from_candidates(
                reference, result.get("candidates", [])
            )
            self._cache[key] = (
                resolved.resolution_status,
                resolved.resolved_document_id,
                resolved.candidate_document_ids,
            )
            return resolved

    async def resolve_references(
        self,
        references: Sequence[DocumentReference],
        *,
        question: str,
        workspace_ids: Sequence[Any],
        db: Any,
        use_llm_fallback: bool = True,
    ) -> tuple[DocumentReference, ...]:
        """Resolve every draft reference, reusing the per-request cache."""
        return tuple(
            await self.resolve_reference(
                reference,
                question=question,
                workspace_ids=workspace_ids,
                db=db,
                use_llm_fallback=use_llm_fallback,
            )
            for reference in references
        )

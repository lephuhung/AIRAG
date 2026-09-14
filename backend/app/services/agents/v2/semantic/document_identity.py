"""Runtime-only v1 document-identity adapter (Phase 4B, Task 6).

Wraps the proven v1 ``resolve_candidates()`` pipeline
(``app.services.agent.doc_resolver``) and translates its ranked candidates
into v2 ``DocumentReference`` identity facts. The v1 regex/SQL/LLM/vector/
fuzzy/rerank code is reused by call, never cloned here.

Ownership rules enforced here:

- The full contextualized user question is always passed as ``topic``; the
  bare reference span alone never drives disambiguation.
- A clear winner becomes ``resolved``; close candidates become
  ``ambiguous`` with candidate IDs; not-found stays not-found and
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
from typing import Any, Sequence
from uuid import UUID

from app.services.agent.doc_resolver import resolve_candidates
from app.services.agents.resolve_doc_agent import (
    AMBIGUITY_RATIO,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)

from ..contracts.semantic import DocumentReference

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentIdentityError",
    "DocumentIdentityResolver",
    "reference_from_candidates",
]

#: How many top candidates an ambiguous reference carries for clarification.
_MAX_AMBIGUOUS_CANDIDATES = 5


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


def reference_from_candidates(
    reference: DocumentReference,
    candidates: Sequence[dict],
) -> DocumentReference:
    """Translate ranked v1 candidates into a v2 identity fact.

    Thresholds are the live ``resolve_doc_agent`` constants (imported, not
    copied): ``HIGH`` clears to ``resolved``; the ``AMBIGUITY_RATIO``
    close-second rule and the ``MEDIUM`` confirmation band become
    ``ambiguous`` with candidate IDs; anything below ``MEDIUM`` (including
    an empty list or zero usable UUIDs) stays ``not_found`` so a
    low-confidence guess can never become a binding.
    """
    ranked = list(candidates or [])
    if not ranked:
        return reference.model_copy(
            update={
                "resolution_status": "not_found",
                "resolved_document_id": None,
                "candidate_document_ids": (),
            }
        )
    top_score = float(ranked[0].get("score", 0.0) or 0.0)
    second_score = float(ranked[1].get("score", 0.0) or 0.0) if len(ranked) > 1 else 0.0
    is_ambiguous = (
        len(ranked) > 1
        and (second_score / max(top_score, 0.01)) >= AMBIGUITY_RATIO
    )
    if is_ambiguous or (MEDIUM_CONFIDENCE_THRESHOLD <= top_score < HIGH_CONFIDENCE_THRESHOLD):
        uuids = tuple(_candidate_uuids(ranked[:_MAX_AMBIGUOUS_CANDIDATES]))
        if not uuids:
            return reference.model_copy(
                update={
                    "resolution_status": "not_found",
                    "resolved_document_id": None,
                    "candidate_document_ids": (),
                }
            )
        return reference.model_copy(
            update={
                "resolution_status": "ambiguous",
                "resolved_document_id": None,
                "candidate_document_ids": uuids,
            }
        )
    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        uuids = _candidate_uuids(ranked[:1])
        if not uuids:
            return reference.model_copy(
                update={
                    "resolution_status": "not_found",
                    "resolved_document_id": None,
                    "candidate_document_ids": (),
                }
            )
        return reference.model_copy(
            update={
                "resolution_status": "resolved",
                "resolved_document_id": uuids[0],
                "candidate_document_ids": (),
            }
        )
    return reference.model_copy(
        update={
            "resolution_status": "not_found",
            "resolved_document_id": None,
            "candidate_document_ids": (),
        }
    )


class DocumentIdentityResolver:
    """Request-scoped v1 identity wrapper with per-turn result cache.

    One instance lives for one request/turn (constructed once by the
    ingress owner alongside ``IntentClassifier``). ``resolve_reference()``
    wraps ``resolve_candidates()`` with the full question as ``topic`` and
    translates the outcome via ``reference_from_candidates()``; every
    result is cached by ``(reference span, topic, workspace scope)`` so
    repeated semantic builds resolve once. Concurrent callers share one
    in-flight resolution per key (single-flight). The resolver performs
    no capability dispatch and never imports supervisor/graph state.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, tuple[str, ...], bool], DocumentReference] = {}
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
                return hit
            result = await resolve_candidates(
                reference_text or topic,
                list(scope),
                db,
                topic=topic,
                use_llm_fallback=use_llm_fallback,
            )
            resolved = reference_from_candidates(
                reference, result.get("candidates", [])
            )
            self._cache[key] = resolved
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

"""Ephemeral discovery-candidate index over persisted AgentResults (Phase 3, Task 2).

The ``document.search`` capability creates a frozen
``DocumentDiscoveryCandidate(candidate_id, document_id, document_revision)``
owned by its originating ``AgentResult.data``; the graph persists that result.
``DiscoveryCandidateRegistry`` is an EPHEMERAL request/resume index back over
those persisted results (``candidate_id -> DocumentDiscoveryCandidate``,
revision included). It is NOT a persistent store: no table, no model, no
schema artifact is added here.

Lifecycle: constructed via ``from_results`` on start, resume, restart, or
interrupt from the persisted ``AgentResult``s. The model observation exposes
``candidate_id`` only and never ``document_revision``. ``bind`` is the governed
handoff the Binding Resolver owns calling: it revalidates current ACL/scope
through the caller-supplied ``authorize`` hook and pins the candidate's
immutable discovered revision into the new ``ScopedDocument``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from uuid import UUID

from ..contracts.binding import DocumentDiscoveryCandidate, ScopedDocument
from ..contracts.capability import DocumentSearchOutput
from ..contracts.execution import AgentResult

_ALLOWED_BINDING_ROLES = ("discovered", "supporting")


class CandidateNotFound(LookupError):
    """No indexed discovery candidate owns that candidate identity."""


class CandidateBindingDenied(PermissionError):
    """Current ACL/scope revalidation refused the candidate binding."""


class DiscoveryCandidateRegistry:
    """Request/resume-scoped ``candidate_id -> DocumentDiscoveryCandidate`` index."""

    def __init__(
        self,
        candidates: dict[UUID, DocumentDiscoveryCandidate] | None = None,
    ) -> None:
        self._index = dict(candidates) if candidates else {}

    @classmethod
    def from_results(
        cls, results: Iterable[AgentResult]
    ) -> DiscoveryCandidateRegistry:
        """Rebuild the index from persisted originating ``AgentResult``s.

        Only ``document.search`` outputs own candidates. The first persisted
        candidate wins per identity, so a candidate's discovered revision stays
        immutable even if a newer revision publishes before resume.
        """
        index: dict[UUID, DocumentDiscoveryCandidate] = {}
        for result in results:
            data = result.data
            if isinstance(data, DocumentSearchOutput):
                for candidate in data.candidates:
                    index.setdefault(candidate.candidate_id, candidate)
        return cls(index)

    def get(self, candidate_id: UUID) -> DocumentDiscoveryCandidate:
        """Return the server-side candidate; fail closed on unknown identity."""
        try:
            return self._index[candidate_id]
        except KeyError:
            raise CandidateNotFound(
                f"unknown discovery candidate {candidate_id}"
            ) from None

    def __contains__(self, candidate_id: object) -> bool:
        return candidate_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def candidate_ids(self) -> tuple[UUID, ...]:
        """Opaque candidate identities in persisted-result order."""
        return tuple(self._index)

    def bind(
        self,
        candidate_id: UUID,
        *,
        binding_id: str,
        role: str = "discovered",
        authorize: Callable[[DocumentDiscoveryCandidate], bool],
    ) -> ScopedDocument:
        """Create the pinned binding after revalidating current ACL/scope.

        ``authorize`` is the server-side current-ACL/scope check owned by the
        Binding Resolver; a refusal fails closed. The binding pins the
        candidate's immutable discovered revision verbatim.
        """
        if role not in _ALLOWED_BINDING_ROLES:
            raise ValueError(
                f"candidate binding role must be one of {_ALLOWED_BINDING_ROLES}"
            )
        candidate = self.get(candidate_id)
        if not authorize(candidate):
            raise CandidateBindingDenied(
                f"candidate {candidate_id} failed current ACL/scope revalidation"
            )
        return ScopedDocument(
            binding_id=binding_id,
            document_id=candidate.document_id,
            document_revision=candidate.document_revision,
            role=role,  # type: ignore[arg-type]
        )

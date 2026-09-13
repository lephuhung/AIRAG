"""Ephemeral discovery-candidate index over persisted AgentResults (Phase 3, Task 2).

The ``document.search`` capability creates a frozen
``DocumentDiscoveryCandidate(candidate_id, document_id, document_revision)``
owned by its originating ``AgentResult.data``; the graph persists that result.
``DiscoveryCandidateRegistry`` is an EPHEMERAL request/resume index back over
those persisted results (``candidate_id -> DocumentDiscoveryCandidate``,
revision included). It is NOT a persistent store: no table, no model, no
schema artifact is added here, and it never creates a binding.

Lifecycle: constructed via ``from_results`` on start, resume, restart, or
interrupt from the persisted ``AgentResult``s. Checkpoint storage may degrade
nested ``AgentResult.data`` to a plain mapping, so ``from_results`` accepts
both the typed output and its checkpointed mapping shape and rebuilds the
frozen candidates from primitives. The model observation exposes
``candidate_id`` only and never ``document_revision``.

Binding ownership: ``candidate_addition_request`` is a PURE handoff that turns
an indexed candidate into a frozen ``BindingAdditionRequest`` for the Binding
Resolver. The Binding Resolver remains the SOLE owner that revalidates current
ACL/scope and creates/pins the binding; nothing here authorizes anything.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from uuid import UUID

from ..contracts.binding import (
    BindingAdditionRequest,
    DocumentDiscoveryCandidate,
)
from ..contracts.capability import DocumentSearchOutput
from ..contracts.execution import AgentResult

_ALLOWED_REQUEST_ROLES = ("discovered", "supporting")


class CandidateNotFound(LookupError):
    """No indexed discovery candidate owns that candidate identity."""


class InvalidCandidateRole(ValueError):
    """The requested binding role is not a discovery-admissible role."""


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

        Only ``document.search`` outputs own candidates. Checkpoint storage may
        degrade ``AgentResult.data`` to a plain mapping, which is coerced back
        into frozen candidates; entries that cannot be rebuilt are skipped so
        one degraded entry cannot poison a resume. The first persisted
        candidate wins per identity, so a candidate's discovered revision stays
        immutable even if a newer revision publishes before resume.
        """
        index: dict[UUID, DocumentDiscoveryCandidate] = {}
        for result in results:
            for candidate in _search_candidates(result.data):
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

    def candidates(self) -> tuple[DocumentDiscoveryCandidate, ...]:
        """Indexed server-side candidates in persisted-result order."""
        return tuple(self._index.values())


def _search_candidates(data: object) -> tuple[DocumentDiscoveryCandidate, ...]:
    """Coerce document.search outputs, typed or checkpoint-degraded, to candidates."""
    if isinstance(data, DocumentSearchOutput):
        return tuple(data.candidates)
    if isinstance(data, Mapping) and data.get("kind") == "document.search":
        rebuilt: list[DocumentDiscoveryCandidate] = []
        entries = data.get("candidates") or ()
        if isinstance(entries, Mapping):
            entries = (entries,)
        for entry in entries:
            candidate = _rebuild_candidate(entry)
            if candidate is not None:
                rebuilt.append(candidate)
        return tuple(rebuilt)
    return ()


def _rebuild_candidate(entry: object) -> DocumentDiscoveryCandidate | None:
    """Rebuild one frozen candidate from its checkpointed mapping shape."""
    if isinstance(entry, DocumentDiscoveryCandidate):
        return entry
    if not isinstance(entry, Mapping):
        return None
    try:
        return DocumentDiscoveryCandidate(
            candidate_id=UUID(str(entry["candidate_id"])),
            document_id=UUID(str(entry["document_id"])),
            document_revision=str(entry["document_revision"]),
        )
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def candidate_addition_request(
    registry: DiscoveryCandidateRegistry,
    candidate_id: UUID,
    requested_role: str,
) -> BindingAdditionRequest:
    """Pure handoff: indexed candidate -> frozen request for the Binding Resolver.

    Validates the role and requires the candidate to exist in the registry.
    Creates no binding, performs no authorization, and exposes no revision:
    the Binding Resolver revalidates current ACL/scope against the server-side
    candidate (via ``registry.get``) and pins the immutable discovered
    revision itself. The candidate is located through the registry index;
    only the shared scheduler resolves capabilities.
    """
    if requested_role not in _ALLOWED_REQUEST_ROLES:
        raise InvalidCandidateRole(
            f"candidate requested_role must be one of {_ALLOWED_REQUEST_ROLES}"
        )
    for candidate in registry.candidates():
        if candidate.candidate_id == candidate_id:
            break
    else:
        raise CandidateNotFound(f"unknown discovery candidate {candidate_id}")
    return BindingAdditionRequest(
        candidate_id=candidate.candidate_id,
        requested_role=requested_role,  # type: ignore[arg-type]
    )

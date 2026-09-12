"""Document search/read capabilities (Phase 2, Task 2; spec §11, §13.4).

- ``DocumentSearchCapability`` (``document.search``) adapts the v1
  document-search service over the current authorized workspace scope taken
  only from ``CapabilityRuntimeContext``. It returns opaque discovery
  candidates (``DocumentSearchOutput``) and never reports read coverage.
- ``DocumentReadCapability`` (``document.read``) reads planned targets. Each
  ``target_id`` resolves to its pinned, currently-authorized ``ScopedDocument``
  through the constructor-injected ``PinnedTargetResolver`` (fed the
  authoritative checkpointed plan/bindings by T6/T7); an unknown, unpinned, or
  no-longer-authorized target fails closed. Read content is persisted as
  governed document evidence and represented by ``EvidenceUseRef``.

Neither capability receives ``GraphRuntimeContext`` or reads supervisor state.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from . import EvidenceBuilder, PinnedTargetResolver, denied_result, error_result
from ..contracts.base import CONTRACT_VERSION
from ..contracts.binding import DocumentDiscoveryCandidate, ScopedDocument
from ..contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    DocumentReadInput,
    DocumentReadOutput,
    DocumentSearchInput,
    DocumentSearchOutput,
)
from ..contracts.evaluation import CoverageObservation
from ..contracts.evidence import (
    DocumentSourceIdentity,
    EvidenceUseRef,
    Provenance,
)
from ..contracts.execution import AgentRequest, AgentResult, AgentStatus
from ..contracts.locators import DocumentLocator


@runtime_checkable
class DocumentSearchService(Protocol):
    """The adapted v1 document-search service (server-side dependency)."""

    async def search(
        self, query: str, workspace_ids: tuple[UUID, ...]
    ) -> Sequence[DocumentDiscoveryCandidate]:
        """Search the current authorized workspace scope for ``query``."""
        ...


@runtime_checkable
class DocumentContentReader(Protocol):
    """Reads one pinned revision's content (server-side dependency)."""

    async def read(self, binding: ScopedDocument) -> str | None:
        """Return the document text for ``binding`` or ``None`` when missing."""
        ...


class DocumentSearchCapability:
    """Atomic ``document.search`` capability: opaque discovery candidates only."""

    descriptor = CapabilityDescriptor(
        name="document.search",
        domain="document",
        operation_type="search",
        supports_parallel=True,
    )

    def __init__(self, *, service: DocumentSearchService) -> None:
        self._service = service

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        if "document.search" not in runtime.allowed_capabilities:
            return denied_result(
                request.task_id,
                code="PERMISSION_DENIED",
                message="document.search is not permitted for this request",
            )
        if not isinstance(request.input, DocumentSearchInput):
            return error_result(
                request.task_id,
                code="INVALID_INPUT",
                message="document.search requires a document.search input",
            )
        # The governed People→Document dependency scalar (``person_identifier``)
        # is enforced at planning/validation time, not here: the search runs
        # over the current authorized workspace scope from the trusted runtime.
        candidates = await self._service.search(
            request.input.query, runtime.workspace_ids
        )
        if not candidates:
            return AgentResult(
                contract_version=CONTRACT_VERSION,
                task_id=request.task_id,
                status="not_found",
                data=DocumentSearchOutput(
                    kind="document.search", candidates=()
                ),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )
        return AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=request.task_id,
            status="success",
            data=DocumentSearchOutput(
                kind="document.search", candidates=tuple(candidates)
            ),
            evidence_uses=(),
            coverage_observations=(),
            error=None,
        )


class DocumentReadCapability:
    """Atomic ``document.read`` capability: pinned authorized revisions only."""

    descriptor = CapabilityDescriptor(
        name="document.read",
        domain="document",
        operation_type="read",
        supports_parallel=False,
    )

    def __init__(
        self,
        *,
        reader: DocumentContentReader,
        evidence: EvidenceBuilder,
        resolver: PinnedTargetResolver,
    ) -> None:
        self._reader = reader
        self._evidence = evidence
        self._resolver = resolver

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        if "document.read" not in runtime.allowed_capabilities:
            return denied_result(
                request.task_id,
                code="PERMISSION_DENIED",
                message="document.read is not permitted for this request",
            )
        if not isinstance(request.input, DocumentReadInput):
            return error_result(
                request.task_id,
                code="INVALID_INPUT",
                message="document.read requires a document.read input",
            )
        bindings: list[tuple[str, ScopedDocument]] = []
        for target_id in request.input.target_ids:
            binding = self._resolver.resolve(target_id)
            if binding is None:
                return denied_result(
                    request.task_id,
                    code="SCOPE_VIOLATION",
                    message=(
                        f"target {target_id!r} has no pinned authorized revision"
                    ),
                )
            bindings.append((target_id, binding))
        uses: list[EvidenceUseRef] = []
        observations: list[CoverageObservation] = []
        read_count = 0
        for target_id, binding in bindings:
            content = await self._reader.read(binding)
            if content is None:
                observations.append(
                    CoverageObservation(
                        target_id=target_id,
                        observed_locators=(),
                        outcome="missing",
                    )
                )
                continue
            use = await self._evidence.persist_use(
                source=DocumentSourceIdentity(
                    kind="document",
                    document_id=binding.document_id,
                    document_revision=binding.document_revision,
                    locator=DocumentLocator(kind="document"),
                ),
                content=content,
                provenance=Provenance(
                    acquisition_id=uuid4(),
                    fetcher="document.read",
                    fetched_at=datetime.now(timezone.utc),
                ),
                task_id=request.task_id,
                purpose="coverage",
                target_id=target_id,
            )
            uses.append(use)
            observations.append(
                CoverageObservation(
                    target_id=target_id,
                    observed_locators=(DocumentLocator(kind="document"),),
                    outcome="read",
                )
            )
            read_count += 1
        status: AgentStatus = (
            "success"
            if read_count == len(bindings)
            else "partial"
            if read_count
            else "not_found"
        )
        return AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=request.task_id,
            status=status,
            data=DocumentReadOutput(
                kind="document.read", read_unit_count=read_count
            ),
            evidence_uses=tuple(uses),
            coverage_observations=tuple(observations),
            error=None,
        )

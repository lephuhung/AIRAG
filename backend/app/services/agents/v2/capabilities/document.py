"""Document search/read capabilities (Phase 2, Task 2; spec §11, §13.4, §14).

- ``DocumentSearchCapability`` (``document.search``) adapts the v1
  document-search service over the current authorized workspace scope taken
  only from ``CapabilityRuntimeContext``. It returns opaque discovery
  candidates (``DocumentSearchOutput``) and never reports read coverage. The
  governed People→Document dependency scalar (``person_identifier``) is
  passed through to the search port as an authorized query refinement (R92):
  a blank scalar fails closed instead of being silently dropped.
- ``DocumentReadCapability`` (``document.read``) reads planned targets. Each
  distinct ``target_id`` resolves to its ``ResolvedTarget`` (planned
  ``TargetUnit`` + pinned, currently-authorized ``ScopedDocument``) through the
  constructor-injected ``PinnedTargetResolver`` (fed the authoritative
  checkpointed plan/bindings by T6/T7); an unknown, unpinned, or
  no-longer-authorized target fails closed. The reader receives the planned
  ``requested_locator`` and returns a typed ``LocatedContent``; ``read``
  coverage is reported only when the observed locator matches the requested
  one — a ``document.read`` whose target asks for the whole document reports
  ``read`` only when the reader truly returned the whole document. Read
  content is persisted as governed document evidence and represented by
  ``EvidenceUseRef``.

Neither capability receives ``GraphRuntimeContext`` or reads supervisor state.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from . import (
    EvidenceBuilder,
    LocatedContent,
    PinnedTargetResolver,
    ResolvedTarget,
    denied_result,
    dependency_error,
    error_result,
)
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
from ..contracts.locators import ContentLocator


@runtime_checkable
class DocumentSearchService(Protocol):
    """The adapted v1 document-search service (server-side dependency)."""

    async def search(
        self,
        query: str,
        person_identifier: str | None,
        workspace_ids: tuple[UUID, ...],
    ) -> Sequence[DocumentDiscoveryCandidate]:
        """Search the current authorized workspace scope for ``query``.

        ``person_identifier`` is the governed People→Document dependency scalar
        materialized server-side after a successful ``people.lookup``; the
        planner never supplies it directly.
        """
        ...


@runtime_checkable
class DocumentContentReader(Protocol):
    """Reads one pinned revision's content at a planned locator."""

    async def read(
        self, binding: ScopedDocument, locator: ContentLocator
    ) -> LocatedContent:
        """Read ``locator`` from ``binding`` with a typed read outcome."""
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
        scalar = request.input.person_identifier
        if scalar is not None and (
            not isinstance(scalar, str) or not scalar.strip()
        ):
            # R92: a blank scalar is never a governed refinement: fail
            # closed rather than silently running a plain query search that
            # drops the people dependency.
            return error_result(
                request.task_id,
                code="INVALID_INPUT",
                message=(
                    "document.search carries a blank people dependency "
                    "scalar; refusing to run an unrefined search"
                ),
            )
        try:
            candidates = await self._service.search(
                request.input.query, scalar, runtime.workspace_ids
            )
        except Exception as exc:
            return dependency_error(
                request.task_id, capability="document.search", exc=exc
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
        # Dedupe while preserving order: a duplicate target must not mint a
        # second EvidenceUse (the frozen validator rejects duplicate use_ids)
        # and must not inflate read_unit_count.
        seen: set[str] = set()
        distinct_ids: list[str] = []
        for target_id in request.input.target_ids:
            if target_id not in seen:
                seen.add(target_id)
                distinct_ids.append(target_id)
        resolved: list[tuple[str, ResolvedTarget]] = []
        for target_id in distinct_ids:
            target = self._resolver.resolve(target_id)
            if target is None:
                return denied_result(
                    request.task_id,
                    code="SCOPE_VIOLATION",
                    message=(
                        f"target {target_id!r} has no pinned authorized revision"
                    ),
                )
            resolved.append((target_id, target))
        # One acquisition id for every record produced by this execute call.
        acquisition_id = uuid4()
        fetched_at = datetime.now(timezone.utc)
        uses: list[EvidenceUseRef] = []
        observations: list[CoverageObservation] = []
        read_count = 0
        for target_id, target in resolved:
            requested = target.target_unit.requested_locator
            try:
                outcome = await self._reader.read(target.document, requested)
            except Exception as exc:
                return dependency_error(
                    request.task_id, capability="document.read", exc=exc
                )
            if outcome.outcome != "read" or outcome.content is None:
                observations.append(
                    CoverageObservation(
                        target_id=target_id,
                        observed_locators=(
                            ()
                            if outcome.observed_locator is None
                            else (outcome.observed_locator,)
                        ),
                        outcome=(
                            "missing"
                            if outcome.outcome == "read"
                            else outcome.outcome
                        ),
                    )
                )
                continue
            if outcome.observed_locator != requested:
                # Fail closed: the dependency did not verifiably read what the
                # plan requested (e.g. a partial/chunked read against a
                # whole-document target), so no read coverage is reported and
                # no evidence is attributed to this target.
                observations.append(
                    CoverageObservation(
                        target_id=target_id,
                        observed_locators=(),
                        outcome="missing",
                    )
                )
                continue
            try:
                use = await self._evidence.persist_use(
                    source=DocumentSourceIdentity(
                        kind="document",
                        document_id=target.document.document_id,
                        document_revision=target.document.document_revision,
                        locator=requested,
                    ),
                    content=outcome.content,
                    provenance=Provenance(
                        acquisition_id=acquisition_id,
                        fetcher="document.read",
                        fetched_at=fetched_at,
                    ),
                    task_id=request.task_id,
                    purpose="coverage",
                    target_id=target_id,
                )
            except Exception as exc:
                return dependency_error(
                    request.task_id, capability="document.read", exc=exc
                )
            uses.append(use)
            observations.append(
                CoverageObservation(
                    target_id=target_id,
                    observed_locators=(requested,),
                    outcome="read",
                )
            )
            read_count += 1
        status: AgentStatus = (
            "success"
            if read_count == len(resolved)
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

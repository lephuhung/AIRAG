"""Section read capability (Phase 2, Task 2; spec §11, §14).

``SectionReadCapability`` (``section.read``) reads planned section targets.
Like ``document.read``, every distinct ``target_id`` resolves to its
``ResolvedTarget`` (planned ``TargetUnit`` + pinned, currently-authorized
``ScopedDocument``) through the constructor-injected ``PinnedTargetResolver``;
unknown/unpinned/unauthorized targets fail closed.

Coverage discipline: a section read emits the *planned* ``SectionLocator``
from ``TargetUnit.requested_locator`` as its READ ``CoverageObservation`` —
and only when the dependency's observed locator matches it. A reader that
returns a different section fails closed (no read coverage, no evidence for
that target). The capability never emits search coverage — discovery
candidates belong to ``document.search`` results. Read content is persisted as
governed document evidence and represented by ``EvidenceUseRef``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import uuid4

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
from ..contracts.binding import ScopedDocument
from ..contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    SectionReadInput,
    SectionReadOutput,
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
class SectionContentReader(Protocol):
    """Reads one pinned section's content at a planned locator."""

    async def read_section(
        self, binding: ScopedDocument, locator: ContentLocator
    ) -> LocatedContent:
        """Read ``locator`` from ``binding`` with a typed read outcome."""
        ...


class SectionReadCapability:
    """Atomic ``section.read`` capability: READ coverage, never search coverage."""

    descriptor = CapabilityDescriptor(
        name="section.read",
        domain="section",
        operation_type="read",
        supports_parallel=False,
    )

    def __init__(
        self,
        *,
        reader: SectionContentReader,
        evidence: EvidenceBuilder,
        resolver: PinnedTargetResolver,
    ) -> None:
        self._reader = reader
        self._evidence = evidence
        self._resolver = resolver

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        if "section.read" not in runtime.allowed_capabilities:
            return denied_result(
                request.task_id,
                code="PERMISSION_DENIED",
                message="section.read is not permitted for this request",
            )
        if not isinstance(request.input, SectionReadInput):
            return error_result(
                request.task_id,
                code="INVALID_INPUT",
                message="section.read requires a section.read input",
            )
        # Dedupe while preserving order (see DocumentReadCapability).
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
                outcome = await self._reader.read_section(
                    target.document, requested
                )
            except Exception as exc:
                return dependency_error(
                    request.task_id, capability="section.read", exc=exc
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
                # Fail closed: the dependency read a different section than
                # the plan requested, so no read coverage is reported and no
                # evidence is attributed to this target.
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
                        fetcher="section.read",
                        fetched_at=fetched_at,
                    ),
                    task_id=request.task_id,
                    purpose="coverage",
                    target_id=target_id,
                )
            except Exception as exc:
                return dependency_error(
                    request.task_id, capability="section.read", exc=exc
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
            data=SectionReadOutput(
                kind="section.read", read_unit_count=read_count
            ),
            evidence_uses=tuple(uses),
            coverage_observations=tuple(observations),
            error=None,
        )

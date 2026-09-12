"""Section read capability (Phase 2, Task 2; spec §11, §14).

``SectionReadCapability`` (``section.read``) reads planned section targets.
Like ``document.read``, every ``target_id`` resolves to its pinned,
currently-authorized ``ScopedDocument`` through the constructor-injected
``PinnedTargetResolver``; unknown/unpinned/unauthorized targets fail closed.

Coverage discipline: a section read emits READ ``CoverageObservation`` facts
(one per planned target, located by the stable ``SectionLocator``) and never
search coverage — discovery candidates belong to ``document.search`` results.
Read content is persisted as governed document evidence and represented by
``EvidenceUseRef``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import uuid4

from . import EvidenceBuilder, PinnedTargetResolver, denied_result, error_result
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
from ..contracts.locators import SectionLocator


@runtime_checkable
class SectionContentReader(Protocol):
    """Reads one pinned section's content (server-side dependency)."""

    async def read_section(
        self, binding: ScopedDocument, target_id: str
    ) -> tuple[str, str] | None:
        """Return ``(content, structure_node_id)`` or ``None`` when missing."""
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
            section = await self._reader.read_section(binding, target_id)
            if section is None:
                observations.append(
                    CoverageObservation(
                        target_id=target_id,
                        observed_locators=(),
                        outcome="missing",
                    )
                )
                continue
            content, structure_node_id = section
            locator = SectionLocator(
                kind="section", structure_node_id=structure_node_id
            )
            use = await self._evidence.persist_use(
                source=DocumentSourceIdentity(
                    kind="document",
                    document_id=binding.document_id,
                    document_revision=binding.document_revision,
                    locator=locator,
                ),
                content=content,
                provenance=Provenance(
                    acquisition_id=uuid4(),
                    fetcher="section.read",
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
                    observed_locators=(locator,),
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
            data=SectionReadOutput(
                kind="section.read", read_unit_count=read_count
            ),
            evidence_uses=tuple(uses),
            coverage_observations=tuple(observations),
            error=None,
        )

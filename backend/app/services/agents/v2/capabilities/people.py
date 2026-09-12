"""People lookup capability (Phase 2, Task 2; spec §11, §15.3).

Adapts the v1 People lookup service and returns only minimized v2 facts: the
governed record stays in the Evidence Store and the ``AgentResult`` carries a
``PeopleLookupOutput(matched=...)`` plus ``EvidenceUseRef`` handles. Raw People
data never enters the output or the checkpoint.

The capability enforces the *current* trusted authorization on every call:
``runtime.allowed_capabilities`` must contain ``people.lookup`` and
``runtime.can_read_people`` must hold. Workspace/people scope comes only from
``CapabilityRuntimeContext``; ``AgentRequest`` cannot supply it.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import uuid4

from . import EvidenceBuilder, denied_result, error_result
from ..contracts.base import CONTRACT_VERSION
from ..contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    PeopleLookupInput,
    PeopleLookupOutput,
)
from ..contracts.evidence import PeopleSourceIdentity, Provenance
from ..contracts.execution import AgentRequest, AgentResult
from ..evidence_store.governance import (
    EvidenceMinimizationError,
    minimize_people_record,
)


@runtime_checkable
class PeopleLookupService(Protocol):
    """The adapted v1 People lookup (server-side dependency, injected)."""

    async def lookup(self, query: str) -> Mapping[str, object] | None:
        """Return the raw People record for ``query`` or ``None`` when unknown."""
        ...


class PeopleCapability:
    """Atomic ``people.lookup`` capability: governed minimization, typed output."""

    descriptor = CapabilityDescriptor(
        name="people.lookup",
        domain="people",
        operation_type="lookup",
        supports_parallel=True,
    )

    def __init__(
        self,
        *,
        service: PeopleLookupService,
        evidence: EvidenceBuilder,
        required_fields: Collection[str] = ("name",),
    ) -> None:
        self._service = service
        self._evidence = evidence
        self._required_fields = tuple(required_fields)

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        if (
            "people.lookup" not in runtime.allowed_capabilities
            or not runtime.can_read_people
        ):
            return denied_result(
                request.task_id,
                code="PERMISSION_DENIED",
                message="people.lookup is not permitted for this request",
            )
        if not isinstance(request.input, PeopleLookupInput):
            return error_result(
                request.task_id,
                code="INVALID_INPUT",
                message="people.lookup requires a people.lookup input",
            )
        raw = await self._service.lookup(request.input.query)
        if raw is None:
            return AgentResult(
                contract_version=CONTRACT_VERSION,
                task_id=request.task_id,
                status="not_found",
                data=PeopleLookupOutput(kind="people.lookup", matched=False),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )
        try:
            minimized = minimize_people_record(
                raw, required_fields=self._required_fields
            )
        except EvidenceMinimizationError as exc:
            return error_result(
                request.task_id, code="CONTRACT_MISMATCH", message=str(exc)
            )
        record_id = raw.get("record_id", raw.get("id", "unknown"))
        use = await self._evidence.persist_use(
            source=PeopleSourceIdentity(
                kind="people", record_id=str(record_id)
            ),
            content=minimized.content,
            provenance=Provenance(
                acquisition_id=uuid4(),
                fetcher="people.lookup",
                fetched_at=datetime.now(timezone.utc),
            ),
            task_id=request.task_id,
            purpose="supporting",
            target_id=None,
        )
        return AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=request.task_id,
            status="success",
            data=PeopleLookupOutput(kind="people.lookup", matched=True),
            evidence_uses=(use,),
            coverage_observations=(),
            error=None,
        )

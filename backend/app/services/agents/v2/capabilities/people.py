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

from . import EvidenceBuilder, denied_result, dependency_error, error_result
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
        # Multi-match seam: the v1-backed service owns grouped-record
        # dedupe under a private reader (the shadow R64 surface pins the
        # public reader to ``lookup`` only); legacy single-``lookup``
        # services keep the first-only path below untouched.
        lookup_many = getattr(self._service, "lookup_many", None)
        if lookup_many is None:
            lookup_many = getattr(self._service, "_lookup_many", None)
        if callable(lookup_many):
            try:
                matches = await lookup_many(request.input.query)
            except Exception as exc:
                return dependency_error(
                    request.task_id, capability="people.lookup", exc=exc
                )
            return await self._execute_matches(request, matches)
        try:
            raw = await self._service.lookup(request.input.query)
        except Exception as exc:
            return dependency_error(
                request.task_id, capability="people.lookup", exc=exc
            )
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
        record_id = raw.get("record_id", raw.get("id"))
        if record_id is None or str(record_id).strip() == "":
            # No resolvable record handle: fail this record closed rather
            # than minting an unattributable 'unknown' identity.
            return error_result(
                request.task_id,
                code="CONTRACT_MISMATCH",
                message="people.lookup record carries no resolvable record_id",
            )
        try:
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
        except Exception as exc:
            return dependency_error(
                request.task_id, capability="people.lookup", exc=exc
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

    async def _execute_matches(
        self, request: AgentRequest, matches: object
    ) -> AgentResult:
        """One governed ``EvidenceUse`` per distinct supplied match.

        ``matches`` is the service-owned distinct-person list (see
        ``V1PeopleLookupService.lookup_many``): every entry carries its
        stable ``record_id`` plus its minimized task-required mapping, so
        the capability persists one minimized record per person instead of
        the legacy first-only use. An empty list is the typed ``not_found``
        (never a document fallback); malformed entries fail closed.
        """
        if matches is None:
            matches = []
        if not isinstance(matches, (list, tuple)):
            return error_result(
                request.task_id,
                code="CONTRACT_MISMATCH",
                message="people.lookup returned a non-sequence match list",
            )
        if len(matches) == 0:
            return AgentResult(
                contract_version=CONTRACT_VERSION,
                task_id=request.task_id,
                status="not_found",
                data=PeopleLookupOutput(kind="people.lookup", matched=False),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )
        # Defensive dedupe: the service owns grouped-record dedupe, but a
        # duplicated handle must never mint two uses for one person.
        deduped: list = []
        seen_ids: set[str] = set()
        for match in matches:
            record_id = getattr(match, "record_id", None)
            if record_id is None or str(record_id).strip() == "":
                return error_result(
                    request.task_id,
                    code="CONTRACT_MISMATCH",
                    message="people.lookup match carries no resolvable record_id",
                )
            key = str(record_id)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            deduped.append(match)
        acquisition_id = uuid4()
        fetched_at = datetime.now(timezone.utc)
        uses: list = []
        for match in deduped:
            fields = getattr(match, "fields", None)
            if not isinstance(fields, Mapping):
                return error_result(
                    request.task_id,
                    code="CONTRACT_MISMATCH",
                    message="people.lookup match carries no minimized mapping",
                )
            required = getattr(match, "required_fields", None)
            if required is None:
                required = self._required_fields
            try:
                minimized = minimize_people_record(
                    dict(fields), required_fields=tuple(required)
                )
            except EvidenceMinimizationError as exc:
                return error_result(
                    request.task_id, code="CONTRACT_MISMATCH", message=str(exc)
                )
            try:
                use = await self._evidence.persist_use(
                    source=PeopleSourceIdentity(
                        kind="people", record_id=str(match.record_id)
                    ),
                    content=minimized.content,
                    provenance=Provenance(
                        acquisition_id=acquisition_id,
                        fetcher="people.lookup",
                        fetched_at=fetched_at,
                    ),
                    task_id=request.task_id,
                    purpose="supporting",
                    target_id=None,
                )
            except Exception as exc:
                return dependency_error(
                    request.task_id, capability="people.lookup", exc=exc
                )
            uses.append(use)
        return AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=request.task_id,
            status="success",
            data=PeopleLookupOutput(kind="people.lookup", matched=True),
            evidence_uses=tuple(uses),
            coverage_observations=(),
            error=None,
        )

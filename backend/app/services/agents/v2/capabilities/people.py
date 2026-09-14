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

from collections.abc import Collection, Mapping, Sequence
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
    MinimizedEvidence,
    minimize_people_record,
)


@runtime_checkable
class PeopleLookupService(Protocol):
    """The adapted v1 People lookup (server-side dependency, injected)."""

    async def lookup(self, query: str) -> Mapping[str, object] | None:
        """Return the raw People record for ``query`` or ``None`` when unknown."""
        ...


@runtime_checkable
class MultiMatchPeopleLookupService(Protocol):
    """Explicit multi-match seam for ``people.lookup`` (M6).

    A capability-injected adapter returning one minimized match per distinct
    person behind a query. Each entry carries ``record_id`` (a stable
    non-PII handle), ``fields`` (the minimized task-required mapping) and
    ``required_fields``. The capability never probes private readers: the
    adapter arrives by name at construction, and a service without one
    keeps the legacy first-only ``lookup`` path below.
    """

    async def lookup_many(self, query: str) -> Sequence[object]:
        """Return the distinct-person matches for ``query`` (maybe empty)."""
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
        multi_match: MultiMatchPeopleLookupService | None = None,
    ) -> None:
        self._service = service
        self._evidence = evidence
        self._required_fields = tuple(required_fields)
        self._multi_match = multi_match

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
        # Multi-match seam (M6): an explicitly injected named adapter. The
        # shadow R64 surface pins the v1 service's public reader to
        # ``lookup`` only, so the capability never probes private readers
        # (``_lookup_many`` and friends are unreachable from here); legacy
        # single-``lookup`` services keep the first-only path below
        # untouched.
        if self._multi_match is not None:
            try:
                matches = await self._multi_match.lookup_many(
                    request.input.query
                )
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
        # M4: validate + minimize EVERY match BEFORE the first persistence,
        # so a malformed kth match fails closed with zero rows instead of
        # leaving earlier uses orphaned behind an error result. Defensive
        # dedupe rides along: the adapter owns grouped-record dedupe, but a
        # duplicated handle must never mint two uses for one person.
        prepared: list[tuple[str, MinimizedEvidence]] = []
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
            prepared.append((key, minimized))
        acquisition_id = uuid4()
        fetched_at = datetime.now(timezone.utc)
        uses: list = []
        for record_id, minimized in prepared:
            try:
                use = await self._evidence.persist_use(
                    source=PeopleSourceIdentity(
                        kind="people", record_id=record_id
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
                # Runtime persistence failure AFTER prevalidation: uses
                # already committed under this acquisition_id stay in the
                # encrypted run-scoped Evidence Store (no delete API exists,
                # so no ad-hoc deletes/transactions are invented). The
                # residue is encrypted, run-owned, non-PII handles with
                # ACL-enforced reads, and the typed error names the failed
                # capability for the evaluator to mark insufficient.
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

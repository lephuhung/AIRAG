"""Memory lookup capability (Phase 2, Task 2; spec §11).

``MemoryCapability`` (``memory.lookup``) adapts the v1 memory store and returns
a minimized count fact (``MemoryLookupOutput(matched_count=...)``); matched
memory content is persisted as governed memory evidence and represented by
``EvidenceUseRef``.

Like knowledge-graph sources, memory sources are ownerless (no
workspace/revision identity), so authorization is the request-scoped registry
intersection plus the transitive run scope enforced at hydration (spec §24).
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import uuid4

from . import EvidenceBuilder, denied_result, dependency_error, error_result
from ..contracts.base import CONTRACT_VERSION
from ..contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    MemoryLookupInput,
    MemoryLookupOutput,
)
from ..contracts.evidence import (
    EvidenceUseRef,
    MemorySourceIdentity,
    Provenance,
)
from ..contracts.execution import AgentRequest, AgentResult


@runtime_checkable
class MemoryStore(Protocol):
    """The adapted v1 memory store (server-side dependency)."""

    async def lookup(self, query: str) -> Sequence[tuple[str, str]]:
        """Return ``(memory_id, content)`` matches for ``query``."""
        ...


class MemoryCapability:
    """Atomic ``memory.lookup`` capability: minimized count + evidence."""

    descriptor = CapabilityDescriptor(
        name="memory.lookup",
        domain="memory",
        operation_type="lookup",
        supports_parallel=True,
    )

    def __init__(self, *, store: MemoryStore, evidence: EvidenceBuilder) -> None:
        self._store = store
        self._evidence = evidence

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        if "memory.lookup" not in runtime.allowed_capabilities:
            return denied_result(
                request.task_id,
                code="PERMISSION_DENIED",
                message="memory.lookup is not permitted for this request",
            )
        if not isinstance(request.input, MemoryLookupInput):
            return error_result(
                request.task_id,
                code="INVALID_INPUT",
                message="memory.lookup requires a memory.lookup input",
            )
        try:
            matches = await self._store.lookup(request.input.query)
        except Exception as exc:
            return dependency_error(
                request.task_id, capability="memory.lookup", exc=exc
            )
        # Dedupe while preserving order (see KnowledgeGraphCapability).
        seen: set[tuple[str, str]] = set()
        distinct: list[tuple[str, str]] = []
        for memory_id, content in matches:
            key = (memory_id, content)
            if key not in seen:
                seen.add(key)
                distinct.append(key)
        if not distinct:
            return AgentResult(
                contract_version=CONTRACT_VERSION,
                task_id=request.task_id,
                status="not_found",
                data=MemoryLookupOutput(
                    kind="memory.lookup", matched_count=0
                ),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )
        # One acquisition id for every record produced by this execute call.
        acquisition_id = uuid4()
        fetched_at = datetime.now(timezone.utc)
        uses: list[EvidenceUseRef] = []
        try:
            for memory_id, content in distinct:
                uses.append(
                    await self._evidence.persist_use(
                        source=MemorySourceIdentity(
                            kind="memory", memory_id=memory_id
                        ),
                        content=content,
                        provenance=Provenance(
                            acquisition_id=acquisition_id,
                            fetcher="memory.lookup",
                            fetched_at=fetched_at,
                        ),
                        task_id=request.task_id,
                        purpose="supporting",
                        target_id=None,
                    )
                )
        except Exception as exc:
            return dependency_error(
                request.task_id, capability="memory.lookup", exc=exc
            )
        matched = len({use.use_id for use in uses})
        return AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=request.task_id,
            status="success",
            data=MemoryLookupOutput(
                kind="memory.lookup", matched_count=matched
            ),
            evidence_uses=tuple(uses),
            coverage_observations=(),
            error=None,
        )

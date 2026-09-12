"""Knowledge-graph query capability (Phase 2, Task 2; spec §11).

``KnowledgeGraphCapability`` (``knowledge_graph.query``) adapts the v1
knowledge-graph client and returns a minimized count fact
(``KnowledgeGraphOutput(matched_entity_count=...)``); matched entity content is
persisted as governed knowledge-graph evidence and represented by
``EvidenceUseRef``.

Ownerless sources carry no workspace/revision identity, so per-record ACL
resolution does not apply here; authorization is the request-scoped registry
intersection (the capability is absent unless allowed) plus the transitive run
scope enforced by the Evidence Store at hydration (spec §24).
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import uuid4

from . import EvidenceBuilder, denied_result, error_result
from ..contracts.base import CONTRACT_VERSION
from ..contracts.capability import (
    CapabilityDescriptor,
    CapabilityRuntimeContext,
    KnowledgeGraphInput,
    KnowledgeGraphOutput,
)
from ..contracts.evidence import (
    EvidenceUseRef,
    KnowledgeGraphSourceIdentity,
    Provenance,
)
from ..contracts.execution import AgentRequest, AgentResult


@runtime_checkable
class KnowledgeGraphClient(Protocol):
    """The adapted v1 knowledge-graph client (server-side dependency)."""

    async def query(self, query: str) -> Sequence[tuple[str, str]]:
        """Return ``(entity_or_relation_id, content)`` matches for ``query``."""
        ...


class KnowledgeGraphCapability:
    """Atomic ``knowledge_graph.query`` capability: minimized count + evidence."""

    descriptor = CapabilityDescriptor(
        name="knowledge_graph.query",
        domain="knowledge_graph",
        operation_type="lookup",
        supports_parallel=True,
    )

    def __init__(
        self, *, client: KnowledgeGraphClient, evidence: EvidenceBuilder
    ) -> None:
        self._client = client
        self._evidence = evidence

    async def execute(
        self, request: AgentRequest, runtime: CapabilityRuntimeContext
    ) -> AgentResult:
        if "knowledge_graph.query" not in runtime.allowed_capabilities:
            return denied_result(
                request.task_id,
                code="PERMISSION_DENIED",
                message="knowledge_graph.query is not permitted for this request",
            )
        if not isinstance(request.input, KnowledgeGraphInput):
            return error_result(
                request.task_id,
                code="INVALID_INPUT",
                message="knowledge_graph.query requires a knowledge_graph.query input",
            )
        matches = await self._client.query(request.input.query)
        if not matches:
            return AgentResult(
                contract_version=CONTRACT_VERSION,
                task_id=request.task_id,
                status="not_found",
                data=KnowledgeGraphOutput(
                    kind="knowledge_graph.query", matched_entity_count=0
                ),
                evidence_uses=(),
                coverage_observations=(),
                error=None,
            )
        uses: list[EvidenceUseRef] = []
        for entity_id, content in matches:
            uses.append(
                await self._evidence.persist_use(
                    source=KnowledgeGraphSourceIdentity(
                        kind="knowledge_graph",
                        entity_or_relation_id=entity_id,
                    ),
                    content=content,
                    provenance=Provenance(
                        acquisition_id=uuid4(),
                        fetcher="knowledge_graph.query",
                        fetched_at=datetime.now(timezone.utc),
                    ),
                    task_id=request.task_id,
                    purpose="supporting",
                    target_id=None,
                )
            )
        return AgentResult(
            contract_version=CONTRACT_VERSION,
            task_id=request.task_id,
            status="success",
            data=KnowledgeGraphOutput(
                kind="knowledge_graph.query",
                matched_entity_count=len(uses),
            ),
            evidence_uses=tuple(uses),
            coverage_observations=(),
            error=None,
        )

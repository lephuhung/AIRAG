"""Typed model observation projections (Phase 3, Task 2).

``CapabilityOutput`` is NOT model-visible by default. ``ObservationProjector``
runs only after the shared scheduler has executed and the graph has persisted
the task, and it exposes an ``AgentToolObservation``: task status, evidence-use
identities, coverage observations, and one typed per-capability projection.

There is no free-form mapping escape hatch and no free-form string metadata:
adding an observable field requires a typed model change here, and an unknown
``result_kind`` fails closed with ``ObservationProjectionUnavailable``.

Sensitivity rules enforced below:

- People exposes status/use IDs/availability only; the governed scalar is
  materialized server-side and never observed.
- ``DocumentSearchObservation`` exposes opaque ``candidate_ids`` only; it never
  exposes ``document_revision``. The Binding Resolver remains the sole owner
  that creates/pins the binding from the server-side candidate.
- Retrieved document text is untrusted data and is never returned to the
  planner; read projections expose only read-unit counts.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union
from uuid import UUID

from pydantic import Field

from ..contracts.base import ContractModel
from ..contracts.capability import (
    DocumentReadOutput,
    DocumentSearchOutput,
    KnowledgeGraphOutput,
    PeopleLookupOutput,
    SectionReadOutput,
)
from ..contracts.evaluation import CoverageObservation
from ..contracts.execution import AgentResult, AgentStatus


class PeopleLookupObservation(ContractModel):
    """Planner-visible People fact: availability only, never the scalar."""

    kind: Literal["people.lookup"] = "people.lookup"
    matched: bool
    dependency_scalar_available: bool


class DocumentSearchObservation(ContractModel):
    """Planner-visible discovery fact: opaque candidate identity only."""

    kind: Literal["document.search"] = "document.search"
    candidate_count: int
    candidate_ids: tuple[UUID, ...]


class DocumentReadObservation(ContractModel):
    kind: Literal["document.read"] = "document.read"
    read_unit_count: int


class SectionReadObservation(ContractModel):
    kind: Literal["section.read"] = "section.read"
    read_unit_count: int


class KnowledgeGraphObservation(ContractModel):
    kind: Literal["knowledge_graph.query"] = "knowledge_graph.query"
    matched_entity_count: int


class NoObservation(ContractModel):
    kind: Literal["none"] = "none"


ToolObservationProjection = Annotated[
    Union[
        PeopleLookupObservation,
        DocumentSearchObservation,
        DocumentReadObservation,
        SectionReadObservation,
        KnowledgeGraphObservation,
        NoObservation,
    ],
    Field(discriminator="kind"),
]


class AgentToolObservation(ContractModel):
    """Implementation-only model observation: typed and minimized."""

    task_id: str
    status: AgentStatus
    evidence_use_ids: tuple[UUID, ...]
    coverage: tuple[CoverageObservation, ...]
    result_kind: str
    projection: ToolObservationProjection


class ObservationProjectionUnavailable(ValueError):
    """No typed projector exists for a result kind; failing closed, never generic."""


class ObservationProjector:
    """Projects persisted capability results into minimized typed observations.

    Minimization is structural, not textual: ``PeopleLookupObservation``
    declares exactly ``{kind, matched, dependency_scalar_available}`` (frozen
    ``extra="forbid"``), so no raw People field or scalar value has any
    typed carrier into the planner observation -- the schema itself is the
    guarantee, and the tests pin its exact field set. There is deliberately
    no substring scanner: scanning serialized JSON for field names cannot see
    actual values and would false-positive on opaque hex (e.g. a UUID
    containing ``cccd``).
    """

    @staticmethod
    def project(
        result: AgentResult,
        *,
        dependency_scalar_available: bool | None = None,
    ) -> AgentToolObservation:
        """Project one persisted ``AgentResult``; fail closed on unknown kinds.

        ``dependency_scalar_available`` is the checkpointed materialization
        decision for this People task (R28): pass the value the deterministic
        materializer recorded (extractable under current governance right
        now). When ``None`` (no materialization decision known) the flag is
        ``False`` -- fail closed -- because ``success`` + ``matched`` alone
        cannot prove the scalar is extractable (R24: missing scalar,
        expired/unauthorized hydration). An explicit value always wins.
        """
        evidence_use_ids = tuple(ref.use_id for ref in result.evidence_uses)
        data = result.data
        if data is None:
            return AgentToolObservation(
                task_id=result.task_id,
                status=result.status,
                evidence_use_ids=evidence_use_ids,
                coverage=result.coverage_observations,
                result_kind="none",
                projection=NoObservation(kind="none"),
            )
        if isinstance(data, PeopleLookupOutput):
            if dependency_scalar_available is None:
                available = False
            else:
                available = dependency_scalar_available
            projection = PeopleLookupObservation(
                kind="people.lookup",
                matched=data.matched,
                dependency_scalar_available=available,
            )
        elif isinstance(data, DocumentSearchOutput):
            projection = DocumentSearchObservation(
                kind="document.search",
                candidate_count=len(data.candidates),
                candidate_ids=tuple(
                    candidate.candidate_id for candidate in data.candidates
                ),
            )
        elif isinstance(data, DocumentReadOutput):
            projection = DocumentReadObservation(
                kind="document.read", read_unit_count=data.read_unit_count
            )
        elif isinstance(data, SectionReadOutput):
            projection = SectionReadObservation(
                kind="section.read", read_unit_count=data.read_unit_count
            )
        elif isinstance(data, KnowledgeGraphOutput):
            projection = KnowledgeGraphObservation(
                kind="knowledge_graph.query",
                matched_entity_count=data.matched_entity_count,
            )
        else:
            raise ObservationProjectionUnavailable(
                f"no model observation projector for result kind {data.kind!r}"
            )
        observation = AgentToolObservation(
            task_id=result.task_id,
            status=result.status,
            evidence_use_ids=evidence_use_ids,
            coverage=result.coverage_observations,
            result_kind=data.kind,
            projection=projection,
        )
        return observation


# The embedded discriminated-union alias resolves explicitly (contracts §3 rule).
AgentToolObservation.model_rebuild()

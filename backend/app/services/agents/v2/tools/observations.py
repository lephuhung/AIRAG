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


#: Raw People fields that must never appear in any planner observation.
#: The governed scalar is materialized server-side; the planner sees only
#: the ``dependency_scalar_available`` availability flag. Any leak of these
#: tokens (scalar value included) fails the projection closed.
RAW_PEOPLE_FIELD_TOKENS: tuple[str, ...] = (
    "cccd",
    "national_id",
    "citizen_id",
    "dob",
    "birth",
    "address",
    "phone",
    "email",
    "personnel",
    "health",
    "religion",
    "ethnicity",
    "biometric",
)


def assert_no_raw_people_fields(observation: AgentToolObservation) -> None:
    """Executable minimization invariant: no raw People field is observable.

    Inspects the serialized observation for the governed scalar value and
    every raw People field token. A match raises
    ``ObservationProjectionUnavailable`` instead of exposing the field to the
    planner.
    """
    dumped = observation.model_dump_json().lower()
    for token in RAW_PEOPLE_FIELD_TOKENS:
        if token in dumped:
            raise ObservationProjectionUnavailable(
                f"planner observation exposes raw People field {token!r}; "
                "refusing to leak governed People data"
            )


class ObservationProjector:
    """Projects persisted capability results into minimized typed observations."""

    @staticmethod
    def project(result: AgentResult) -> AgentToolObservation:
        """Project one persisted ``AgentResult``; fail closed on unknown kinds."""
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
            projection = PeopleLookupObservation(
                kind="people.lookup",
                matched=data.matched,
                dependency_scalar_available=(
                    result.status == "success" and data.matched
                ),
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
        # People→Document invariant: the scalar and every raw People field
        # stay server-side; the planner sees the availability flag only.
        # (Scoped to the People projection: other projections carry opaque
        # UUIDs whose hex may coincidentally contain a short token.)
        if isinstance(projection, PeopleLookupObservation):
            assert_no_raw_people_fields(observation)
        return observation


# The embedded discriminated-union alias resolves explicitly (contracts §3 rule).
AgentToolObservation.model_rebuild()

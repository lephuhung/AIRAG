"""Frozen capability contracts (spec §11, §13.4, §16).

This module is the single owner of the frozen capability boundary:
``CapabilityDescriptor``, ``CapabilityInput``, ``CapabilityOutput``, and the
runtime-only ``CapabilityRuntimeContext``. The capability package (Phase 2)
imports these types and never redefines or field-extends them.

Each input/output variant carries only domain-required fields, is a strict
frozen model, and is reached only through the discriminated union on ``kind`` —
there is no ``dict[str, Any]`` escape hatch. A variant's ``kind`` is the
capability name registered in the request-scoped capability registry.

Variant ownership: the spec assigns each variant to its capability module. Phase 1
does not ship capability modules, so the variants live here with the union they
compose; capability modules import them for their typed signatures.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from .base import ContractModel
from .binding import DocumentDiscoveryCandidate
from .routing import Domain
from .semantic import AbbreviationResolution


class CapabilityRuntimeContext(ContractModel):
    """Spec §11: current trusted permission/scope for one capability call.

    Runtime-only: never checkpointed, never model-supplied. ``config_revision``
    is deliberately absent because no capability changes behavior from it.
    """

    request_id: str
    run_id: str
    user_id: UUID
    workspace_ids: tuple[UUID, ...]
    can_read_people: bool
    allowed_capabilities: frozenset[str]
    deadline_at: datetime


class CapabilityDescriptor(ContractModel):
    """Spec §16: minimal planner catalog entry built from the runtime registry."""

    name: str
    domain: Domain
    operation_type: Literal["lookup", "search", "read", "transform", "resolve"]
    supports_parallel: bool


class PeopleLookupInput(ContractModel):
    kind: Literal["people.lookup"]
    query: str


class DocumentSearchInput(ContractModel):
    """Discovery search over the current authorized workspace scope.

    ``person_identifier`` is the deterministic People→Document dependency scalar
    materialized server-side by the dependency node before the task is appended;
    it is never model-supplied and never appears in planner observations.
    """

    kind: Literal["document.search"]
    query: str
    person_identifier: str | None = None


class DocumentReadInput(ContractModel):
    """Reads one or more planned targets; document identity resolves via the plan."""

    kind: Literal["document.read"]
    target_ids: tuple[str, ...] = Field(min_length=1)


class SectionReadInput(ContractModel):
    kind: Literal["section.read"]
    target_ids: tuple[str, ...] = Field(min_length=1)


class WriteInput(ContractModel):
    kind: Literal["write"]
    instruction: str
    target_ids: tuple[str, ...] = ()


class KnowledgeGraphInput(ContractModel):
    kind: Literal["knowledge_graph.query"]
    query: str


class MemoryLookupInput(ContractModel):
    kind: Literal["memory.lookup"]
    query: str


class AbbreviationResolveInput(ContractModel):
    kind: Literal["abbreviation.resolve"]
    tokens: tuple[str, ...] = Field(min_length=1)


CapabilityInput = Annotated[
    PeopleLookupInput
    | DocumentSearchInput
    | DocumentReadInput
    | SectionReadInput
    | WriteInput
    | KnowledgeGraphInput
    | MemoryLookupInput
    | AbbreviationResolveInput,
    Field(discriminator="kind"),
]


class PeopleLookupOutput(ContractModel):
    """Minimized People fact: the governed record stays in the Evidence Store."""

    kind: Literal["people.lookup"]
    matched: bool


class DocumentSearchOutput(ContractModel):
    """Spec §13.4: discovery returns candidates; it never reports read coverage."""

    kind: Literal["document.search"]
    candidates: tuple[DocumentDiscoveryCandidate, ...]


class DocumentReadOutput(ContractModel):
    """Checkpoint-safe read fact; read content is persisted as evidence only."""

    kind: Literal["document.read"]
    read_unit_count: int


class SectionReadOutput(ContractModel):
    kind: Literal["section.read"]
    read_unit_count: int


class WriteOutput(ContractModel):
    kind: Literal["write"]
    applied: bool


class KnowledgeGraphOutput(ContractModel):
    kind: Literal["knowledge_graph.query"]
    matched_entity_count: int


class MemoryLookupOutput(ContractModel):
    kind: Literal["memory.lookup"]
    matched_count: int


class AbbreviationResolveOutput(ContractModel):
    kind: Literal["abbreviation.resolve"]
    resolutions: tuple[AbbreviationResolution, ...]


CapabilityOutput = Annotated[
    PeopleLookupOutput
    | DocumentSearchOutput
    | DocumentReadOutput
    | SectionReadOutput
    | WriteOutput
    | KnowledgeGraphOutput
    | MemoryLookupOutput
    | AbbreviationResolveOutput,
    Field(discriminator="kind"),
]


# Spec §3: unions are defined before use; models that embed them rebuild
# explicitly so the discriminated aliases are fully resolved.
CapabilityDescriptor.model_rebuild()

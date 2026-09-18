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

    ``person_identifier`` is the governed People→Document dependency scalar that
    ``PeopleDocumentDependencyAdapter`` materializes server-side after a
    successful ``people.lookup`` task; the planner never supplies it, and
    validation requires any search task carrying it to depend on that
    ``people.lookup`` task. It *is* planner-visible through
    ``ResearchPlanningInput.current_plan`` (the whole ``TaskPlan``, including
    ``TaskSpec.input``, is handed to the replanner), so it is checkpointed
    personal data and needs the §13.3 retention/ACL/encryption/audit controls in
    later phases.
    """

    kind: Literal["document.search"]
    query: str
    person_identifier: str | None = None


class DocumentRetrieveInput(ContractModel):
    """Revision-aware factual retrieval over authorized scope or pinned targets.

    Empty ``target_ids`` means retrieval over the current authenticated
    workspace scope. Non-empty ``target_ids`` reference planned target units
    and resolve only through the authoritative checkpointed plan and bindings;
    they never carry raw document or revision IDs. ``query`` is model-supplied
    only through a governed proposal; output carries counts only, with content
    persisted as governed Evidence records.
    """

    kind: Literal["document.retrieve"]
    query: str = Field(min_length=1)
    target_ids: tuple[str, ...] = ()
    top_k: int = Field(default=8, ge=1, le=20)


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
    | DocumentRetrieveInput
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


MatchKind = Literal[
    "exact_document_number",
    "exact_normalized_title",
    "lexical_title",
    "semantic",
]


class DocumentIdentityMatch(ContractModel):
    """Discovery spec §8.2: checkpoint-safe identity/rank/calibration metadata.

    One record per returned candidate; no candidate text enters this contract.
    """

    candidate_id: UUID
    document_id: UUID
    document_revision: str
    rank: int
    confidence: float | None
    match_kind: MatchKind
    calibration_version: str | None


class DocumentSearchOutput(ContractModel):
    """Spec §13.4: discovery returns candidates; it never reports read coverage."""

    kind: Literal["document.search"]
    candidates: tuple[DocumentDiscoveryCandidate, ...]
    identity_matches: tuple[DocumentIdentityMatch, ...] = ()


class DocumentRetrieveOutput(ContractModel):
    """Checkpoint-safe retrieval fact; retrieved content lives in evidence only."""

    kind: Literal["document.retrieve"]
    retrieved_unit_count: int = Field(ge=0)


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
    | DocumentRetrieveOutput
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

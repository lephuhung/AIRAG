"""Finalized and draft query meaning (spec §8.3, §8.4).

``RequestContext.original_query`` stays the sole authoritative raw query;
``SemanticContext`` never copies it and never copies resulting binding IDs.
``SemanticDraft`` and small-model output are internal and unpersisted; only the
finalized ``SemanticSnapshot`` is persisted.
"""
from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from .base import ContractModel, ContractVersion
from .binding import DocumentRole
from .conversation import EntityReference

DocumentResolutionStatus = Literal["unresolved", "resolved", "ambiguous", "not_found", "error"]


class CurrentRevisionRequirement(ContractModel):
    """Explicit current/latest semantics; revalidated for freshness on reuse."""

    kind: Literal["current"]


class PinnedRevisionRequirement(ContractModel):
    """Explicit request for one known specific/historical revision."""

    kind: Literal["pinned"]
    document_revision: str


RevisionRequirement = Annotated[
    CurrentRevisionRequirement | PinnedRevisionRequirement,
    Field(discriminator="kind"),
]


class AbbreviationResolution(ContractModel):
    """Spec §8.4: one protected/normalized abbreviation occurrence.

    ``expansion`` is ``None`` when the token is preserved (unknown, or ambiguous
    without conditional disambiguation); blocking ambiguity is reported through
    ``BlockingAmbiguity`` rather than a second status field here.
    """

    abbreviation: str
    expansion: str | None = None


class CoreferenceResolution(ContractModel):
    """Spec §8.3: a discourse mention resolved to an already-known reference ID."""

    mention: str
    resolved_ref_id: str | None = None


class SectionReference(ContractModel):
    """Spec §8.3/§10: a section named in the query.

    ``label`` is the surface form used when reporting ambiguity; the canonical
    coordinate is ``structure_node_id`` once resolution succeeds.
    """

    ref_id: str
    label: str
    structure_node_id: str | None = None


class BlockingAmbiguity(ContractModel):
    """Spec §8.3: an ambiguity that blocks routing until it is resolved."""

    ambiguity_id: str
    description: str


class DocumentReference(ContractModel):
    """Spec §8.3: a document reference resolved by the Binding Resolver.

    ``revision_requirement`` is optional and carries only an explicit revision
    instruction: ``None`` is the ordinary pin-once reference, ``current`` requires
    freshness revalidation on resume/reuse, and ``pinned`` selects one known
    immutable revision during initial resolution.
    """

    ref_id: str
    original_span: str
    normalized_reference: str
    requested_role: DocumentRole | None
    revision_requirement: RevisionRequirement | None = None
    resolution_status: DocumentResolutionStatus
    resolved_document_id: UUID | None
    candidate_document_ids: tuple[UUID, ...] = ()


class SemanticDraft(ContractModel):
    """Spec §8.3: ephemeral draft meaning; never persisted, never versioned."""

    provisional_contextualized_query: str
    abbreviations: tuple[AbbreviationResolution, ...]
    coreferences: tuple[CoreferenceResolution, ...]
    document_refs: tuple[DocumentReference, ...]
    person_refs: tuple[EntityReference, ...]
    section_refs: tuple[SectionReference, ...]
    preliminary_ambiguities: tuple[BlockingAmbiguity, ...]


class SemanticContext(ContractModel):
    """Spec §8.3: the finalized query meaning persisted in ``SemanticSnapshot``."""

    contextualized_query: str
    normalized_query: str
    abbreviations: tuple[AbbreviationResolution, ...]
    coreferences: tuple[CoreferenceResolution, ...]
    document_refs: tuple[DocumentReference, ...]
    person_refs: tuple[EntityReference, ...]
    section_refs: tuple[SectionReference, ...]
    blocking_ambiguities: tuple[BlockingAmbiguity, ...]


class SemanticSnapshot(ContractModel):
    """Spec §3/§8.3: the persisted semantic boundary."""

    contract_version: ContractVersion
    semantic: SemanticContext


# Spec §3: models that embed discriminated-union aliases are rebuilt explicitly.
DocumentReference.model_rebuild()
SemanticDraft.model_rebuild()
SemanticContext.model_rebuild()
SemanticSnapshot.model_rebuild()

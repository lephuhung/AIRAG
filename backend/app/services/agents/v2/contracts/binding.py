"""Document identity, binding, target units, and discovery proposals (spec §9).

``ScopedDocument`` owns document ID, the immutable revision pinned for this run,
and the semantic role — nothing else. Locators and completion criteria belong to
``TargetUnit``; lineage belongs to the ``BindingAuditRow`` audit boundary; the
unresolved-reference projection stays in ``SemanticContext.document_refs`` and is
never copied here.
"""
from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from .base import ContractModel, ContractVersion

DocumentRole = Literal["target", "reference", "supporting", "discovered"]


class ScopedDocument(ContractModel):
    """Spec §9: the canonical bound-document fact owned by the Binding Resolver."""

    binding_id: str
    document_id: UUID
    document_revision: str
    role: DocumentRole


class UserBindingProvenance(ContractModel):
    """Binding created from an explicit user document reference."""

    kind: Literal["user_reference"]
    binding_id: str
    source_ref_id: str


class DiscoveredBindingProvenance(ContractModel):
    """Binding created from a discovery candidate produced by a task."""

    kind: Literal["discovered"]
    binding_id: str
    source_task_id: str


class PromotedBindingProvenance(ContractModel):
    """Binding created by promoting an existing binding."""

    kind: Literal["promotion"]
    binding_id: str
    source_binding_id: str
    reason: str


BindingProvenance = Annotated[
    UserBindingProvenance | DiscoveredBindingProvenance | PromotedBindingProvenance,
    Field(discriminator="kind"),
]


class BindingAuditRow(ContractModel):
    """Spec §3/§9: the persisted binding-audit boundary (not event sourcing)."""

    contract_version: ContractVersion
    provenance: BindingProvenance


class BindingRevisionRequirement(ContractModel):
    """Spec §9: maps a current-required binding back to its authoritative reference.

    Exactly one relation exists for a binding whose source reference carries
    ``CurrentRevisionRequirement``; ordinary, explicitly pinned, and discovered
    bindings carry none.
    """

    binding_id: str
    ref_id: str


class DocumentBindingSet(ContractModel):
    """Spec §9: nested checkpoint state; inherits the root checkpoint version."""

    bindings: tuple[ScopedDocument, ...]
    revision_requirement_refs: tuple[BindingRevisionRequirement, ...]


class DocumentDiscoveryCandidate(ContractModel):
    """Spec §9: discovery candidate owned by its originating ``AgentResult``.

    ``candidate_id`` is a globally unique UUID so parallel discovery tasks cannot
    collide; the containing result's ``task_id`` supplies task lineage.
    """

    candidate_id: UUID
    document_id: UUID
    document_revision: str


class BindingAdditionRequest(ContractModel):
    """Spec §9: proposal to add a discovered/supporting binding."""

    candidate_id: UUID
    requested_role: Literal["discovered", "supporting"]


class BindingPromotionRequest(ContractModel):
    """Spec §9: proposal to promote an existing binding."""

    source_binding_id: str
    requested_role: Literal["reference", "supporting"]


# BindingAuditRow references the discriminated provenance union (spec §3: unions
# are defined before use; rebuild explicitly so the alias is fully resolved).
BindingAuditRow.model_rebuild()

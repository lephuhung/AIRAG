"""Clarification request and resolution (spec §20).

Candidate identity/order is stable within one clarification, ``selected_candidate_id``
is the authoritative deterministic selection, and ``expires_at`` freezes a stable
per-request resume deadline. The raw clarification reply remains authoritative in
chat persistence and is not copied here.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from .base import ContractModel, ContractVersion

ClarificationReason = Literal[
    "required_document_not_found",
    "required_document_ambiguous",
    "required_unit_unreadable",
    "semantic_ambiguity",
]


class DocumentCandidate(ContractModel):
    """Spec §20: one stable candidate offered to the user."""

    candidate_id: str
    ordinal: int
    ref_id: str
    document_id: UUID
    label: str


class ClarificationRequest(ContractModel):
    """Spec §3/§20: the persisted clarification request envelope."""

    kind: Literal["semantic"] = "semantic"
    contract_version: ContractVersion
    clarification_id: str
    reason: ClarificationReason
    question: str
    unresolved_ref_ids: tuple[str, ...]
    candidates: tuple[DocumentCandidate, ...]
    expires_at: datetime


class ClarificationResolution(ContractModel):
    """Spec §3/§20: the deterministic selection made by the user."""

    kind: Literal["semantic"] = "semantic"
    contract_version: ContractVersion
    clarification_id: str
    selected_candidate_id: str | None


MAX_PUBLIC_CHOICES = 3


class DocumentSelectionChoice(ContractModel):
    """Discovery spec §15: one safe public choice inside a selection slot."""

    choice_token: str
    title: str
    document_number: str | None


class DocumentSelectionSlot(ContractModel):
    """Discovery spec §15: public per-slot choice frame."""

    slot_id: str
    slot_label: str
    min_selections: int
    max_selections: int
    choices: tuple[DocumentSelectionChoice, ...]


class DocumentSelectionClarification(ContractModel):
    """Discovery spec §15: the persisted document-selection request.

    Raw choice tokens live only here so the exact public request can be
    replayed after restart; the internal manifest checkpoints digests only.
    """

    kind: Literal["document_selection"]
    contract_version: ContractVersion
    clarification_id: str
    question: str
    slots: tuple[DocumentSelectionSlot, ...]
    expires_at: datetime


class DocumentSelectionManifestEntry(ContractModel):
    """Discovery spec §15: one internal choice→aggregate binding."""

    choice_token_digest: str
    slot_id: str
    aggregate_id: UUID
    document_id: UUID
    document_revision: str


class DocumentSelectionManifest(ContractModel):
    """Discovery spec §15: internal checkpointed digest manifest."""

    clarification_id: str
    entries: tuple[DocumentSelectionManifestEntry, ...]
    expires_at: datetime
    status: Literal["pending", "consumed"]


class DocumentSlotResolution(ContractModel):
    """Discovery spec §15: the user's token selections for one slot."""

    slot_id: str
    choice_tokens: tuple[str, ...]


class DocumentSelectionResolution(ContractModel):
    """Discovery spec §15: runner-supplied document-selection resolution."""

    kind: Literal["document_selection"]
    contract_version: ContractVersion
    clarification_id: str
    selections: tuple[DocumentSlotResolution, ...]
    declined: bool

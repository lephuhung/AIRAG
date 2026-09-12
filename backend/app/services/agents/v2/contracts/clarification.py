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

    contract_version: ContractVersion
    clarification_id: str
    reason: ClarificationReason
    question: str
    unresolved_ref_ids: tuple[str, ...]
    candidates: tuple[DocumentCandidate, ...]
    expires_at: datetime


class ClarificationResolution(ContractModel):
    """Spec §3/§20: the deterministic selection made by the user."""

    contract_version: ContractVersion
    clarification_id: str
    selected_candidate_id: str | None

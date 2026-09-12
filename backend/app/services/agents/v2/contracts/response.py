"""Grounding output and the final response boundary (spec §19.2).

``RenderedCitation`` is deterministic presentation output created only after
grounding. It deliberately omits a claim ID because the current API/frontend has
no claim-highlighting interaction, and ``FinalResponse`` stores no derivable
evidence-ID projection.
"""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from .base import ContractModel, ContractVersion

ResponseStatus = Literal["success", "clarify", "denied", "insufficient", "error"]


class RenderedCitation(ContractModel):
    """Spec §19.2: presented citation; identity is not model authority."""

    citation_id: str
    evidence_id: UUID
    label: str


class FinalResponse(ContractModel):
    """Spec §3/§19.2: the persisted/transported response boundary."""

    contract_version: ContractVersion
    status: ResponseStatus
    content: str
    citations: tuple[RenderedCitation, ...] = ()

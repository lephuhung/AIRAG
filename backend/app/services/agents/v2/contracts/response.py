"""Grounding output and the final response boundary (spec §19.2).

``RenderedCitation`` is deterministic presentation output created only after
grounding. It deliberately omits a claim ID because the current API/frontend has
no claim-highlighting interaction, and ``FinalResponse`` stores no derivable
evidence-ID projection.

``FinalResponse.citations`` also accepts ``PublicCitation`` (spec §11.3): the
grounded-LLM synthesis path stores the CitationProjector's allowlisted public
projection verbatim — the same set the SSE ``citation`` frame and ``complete``
payload repeat — while legacy extractive citations keep ``RenderedCitation``.
"""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from .base import ContractModel, ContractVersion
from .synthesis import PublicCitation

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
    citations: tuple[RenderedCitation | PublicCitation, ...] = ()

"""Synthesis hydration and one claim-to-EvidenceUse mapping (spec §19).

``SynthesisInput`` is model-facing and contains only semantic/evaluation facts and
use references; plans, bindings, and runtime policy stay with the Evidence
Hydrator. ``SynthesisEvidence`` is an ephemeral hydrator projection, and
``AnswerClaim`` is the single authoritative claim-to-EvidenceUse relationship.
"""
from __future__ import annotations

from uuid import UUID

from .base import ContractModel
from .binding import DocumentRole
from .evaluation import EvidenceEvaluation
from .evidence import EvidenceUseRef
from .semantic import SemanticContext


class SynthesisInput(ContractModel):
    """Spec §19.1: model-facing factual synthesis request.

    Constructed only after ``EvidenceEvaluation`` exists and is ``sufficient``.
    """

    semantic: SemanticContext
    evaluation: EvidenceEvaluation
    evidence_uses: tuple[EvidenceUseRef, ...]


class SynthesisRuntimeContext(ContractModel):
    """Spec §19.1: runtime synthesis budget; never a business input."""

    max_evidence_items: int
    max_total_chars: int
    max_total_tokens: int


class SynthesisEvidence(ContractModel):
    """Spec §19.1: ephemeral hydrator projection of one admitted use.

    ``role`` and ``target_id`` are computed from the admitted use plus canonical
    plan/binding stores and are never persisted back into evidence.
    """

    use_id: UUID
    content: str
    role: DocumentRole | None
    target_id: str | None
    source_label: str | None


class AnswerClaim(ContractModel):
    """Spec §19.2: the standalone grounding unit.

    Every listed use ID must be a member of the hydrated admitted current-run use
    set; there is no duplicate citation relationship.
    """

    claim_id: str
    text: str
    evidence_use_ids: tuple[UUID, ...]


class AnswerDraft(ContractModel):
    """Spec §19.2: one synthesized draft with its claim set."""

    content: str
    claims: tuple[AnswerClaim, ...]

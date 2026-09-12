"""Coverage and evaluation (spec §14, §18).

Coverage references target IDs and never duplicates binding identity: the
evaluator resolves ``target_id → TargetUnit → binding_id → ScopedDocument``.
Contradictions reference admitted current-run EvidenceUses so target and purpose
context survives when ``prior_evaluation`` drives replanning.
"""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from .base import ContractModel
from .locators import ContentLocator

CoverageOutcome = Literal["read", "missing", "unreadable", "truncated"]
CoverageStatus = Literal["read_complete", "read_partial", "missing", "unreadable", "truncated"]
EvaluationStatus = Literal["sufficient", "insufficient", "contradictory", "needs_input"]


class CoverageObservation(ContractModel):
    """Spec §14: what a capability observed for one planned target."""

    target_id: str
    observed_locators: tuple[ContentLocator, ...]
    outcome: CoverageOutcome


class CoverageItem(ContractModel):
    """Spec §14: evaluator-owned coverage verdict for one target."""

    target_id: str
    observed_locators: tuple[ContentLocator, ...]
    status: CoverageStatus


class Coverage(ContractModel):
    """Spec §14: evaluator-owned coverage facts."""

    items: tuple[CoverageItem, ...]


class Contradiction(ContractModel):
    """Spec §18: an unresolved or reported conflict between admitted uses."""

    contradiction_id: str
    claim_a: str
    claim_b: str
    evidence_use_ids: tuple[UUID, ...]


class MissingRequirement(ContractModel):
    """Spec §18: evaluator-owned gap for one target and criterion kind.

    ``semantic_criterion_id`` is required for ``criterion_kind="semantic"`` and
    forbidden for coverage, so one criterion has exactly one identity.
    """

    target_id: str
    criterion_kind: Literal["coverage", "semantic"]
    semantic_criterion_id: str | None = None
    description: str


class EvidenceEvaluation(ContractModel):
    """Spec §18: the evaluator's sufficiency verdict."""

    status: EvaluationStatus
    coverage: Coverage
    missing: tuple[MissingRequirement, ...]
    contradictions: tuple[Contradiction, ...]

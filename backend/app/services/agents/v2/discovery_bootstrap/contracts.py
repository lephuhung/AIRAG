"""Data-only discovery bootstrap contracts (discovery spec §7, §8).

This module contains frozen data models only. It must never import
``contracts/validation``, ``contracts/state``, ``contracts/planning``, or
``discovery_bootstrap/validation`` so the dependency direction stays one-way and
``contracts/state`` can type its root slots against these concrete models.
"""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import model_validator

from ..contracts.base import ContractModel
from ..contracts.capability import DocumentIdentityMatch, MatchKind
from ..contracts.clarification import DocumentSelectionManifest
from ..contracts.locators import ContentLocator

__all__ = [
    "DiscoveryCheckpoint",
    "DiscoveryNeed",
    "DiscoverySelection",
    "DocumentIdentityMatch",
    "MatchKind",
    "ProbeCandidateMatch",
    "ResearchTargetSelection",
    "SearchProbe",
    "SelectedBindingRef",
    "SlotBindingSelection",
    "SlotCandidateAggregate",
    "TargetSlot",
]


class DiscoveryNeed(ContractModel):
    """Discovery spec §4.1: observability-only route summary; never a routing input."""

    required: bool
    reason: Literal["unresolved_document_slot"] | None

    @model_validator(mode="after")
    def _reason_iff_required(self) -> DiscoveryNeed:
        if self.required != (self.reason is not None):
            raise ValueError(
                "DiscoveryNeed.required must equal (reason is not None)"
            )
        return self


class TargetSlot(ContractModel):
    """Discovery spec §7: one logical research slot owned by the selection."""

    slot_id: str
    intended_role: Literal["target", "reference"]
    subject_hint: str
    requested_locator: ContentLocator
    required: bool
    min_selections: int
    max_selections: int
    explicit_binding_ids: tuple[str, ...]
    source: Literal["semantic_reference", "explicit_resource", "research_need"]


class SelectedBindingRef(ContractModel):
    """Discovery spec §7: one selected binding inside a slot.

    ``target_id`` is allocated server-side during final selection and reused
    verbatim as ``TargetUnit.target_id``.
    """

    target_id: str
    binding_id: str
    selected_aggregate_id: UUID | None
    authority: Literal[
        "explicit_binding",
        "exact_match_policy",
        "confidence_policy",
        "user_choice",
    ]


class SlotBindingSelection(ContractModel):
    """Discovery spec §7: the selected bindings of one logical slot."""

    slot_id: str
    selections: tuple[SelectedBindingRef, ...]


class ResearchTargetSelection(ContractModel):
    """Discovery spec §3.2: durable owner of frozen slots and selected bindings."""

    work_type: Literal["summarize", "compare"]
    target_slots: tuple[TargetSlot, ...]
    slot_bindings: tuple[SlotBindingSelection, ...]
    context_binding_ids: tuple[str, ...]


class SearchProbe(ContractModel):
    """Discovery spec §8.1: one accepted bounded identity-search probe."""

    probe_id: str
    slot_id: str
    query: str
    round: Literal[1, 2]
    origin: Literal["semantic", "planner_fallback"]


class ProbeCandidateMatch(ContractModel):
    """Discovery spec §8.2: join of a probe-origin task and one identity match."""

    probe_id: str
    slot_id: str
    task_id: str
    candidate_id: UUID
    document_id: UUID
    document_revision: str
    rank: int
    confidence: float | None
    match_kind: MatchKind
    calibration_version: str | None


class SlotCandidateAggregate(ContractModel):
    """Discovery spec §8.2: one authoritative document identity per slot."""

    aggregate_id: UUID
    slot_id: str
    document_id: UUID
    document_revision: str
    source_candidate_ids: tuple[UUID, ...]
    source_probe_ids: tuple[str, ...]
    source_task_ids: tuple[str, ...]
    best_match_kind: MatchKind
    aggregate_confidence: float | None
    best_rank: int


class DiscoverySelection(ContractModel):
    """Discovery spec §8.3: the recorded selection of one slot."""

    slot_id: str
    selected_aggregate_ids: tuple[UUID, ...]
    authority: Literal["exact_match_policy", "confidence_policy", "user_choice"]


class DiscoveryCheckpoint(ContractModel):
    """Discovery spec §8.4: authoritative derived discovery state."""

    discovery_id: UUID
    target_slots: tuple[TargetSlot, ...]
    accepted_probes: tuple[SearchProbe, ...]
    candidate_matches: tuple[ProbeCandidateMatch, ...]
    slot_aggregates: tuple[SlotCandidateAggregate, ...]
    selections: tuple[DiscoverySelection, ...]
    rounds_consumed: int
    probes_consumed: int
    status: Literal["searching", "ranking", "clarification", "selected", "unavailable"]
    clarification_manifest: DocumentSelectionManifest | None


# Spec §3: models embedding discriminated-union aliases rebuild explicitly.
TargetSlot.model_rebuild()

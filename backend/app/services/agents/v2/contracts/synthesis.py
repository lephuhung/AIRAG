"""Synthesis hydration, claim-to-EvidenceUse mapping, and the bounded
synthesis checkpoint (spec §13.3–13.5, §19).

``SynthesisInput`` is model-facing and contains only semantic/evaluation facts and
use references; plans, bindings, and runtime policy stay with the Evidence
Hydrator. ``SynthesisEvidence`` is an ephemeral hydrator projection, and
``AnswerClaim`` is the single authoritative claim-to-EvidenceUse relationship.

``SynthesisCheckpoint`` is the durable owner of the grounded-LLM synthesis
state machine: the stable ``E``-handle manifest, the parsed candidate, the
grounded artifact, and the closed failure code. It never carries raw evidence
plaintext, prompt text, unparsed model output, reasoning, secrets, ACL state,
or stack traces.
"""
from __future__ import annotations

from typing import Literal

from uuid import UUID

from .base import ContractModel, ContractVersion
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


# ---------------------------------------------------------------------------
# Grounded LLM synthesis checkpoint (spec §13.3–13.5)
# ---------------------------------------------------------------------------

SynthesisPhase = Literal[
    "prepared", "attempt_reserved", "candidate", "grounded", "failed"
]
"""Spec §13.3: the durable phases of the bounded synthesis state machine."""

ClaimPresentation = Literal["summary", "detail", "caveat"]
"""Spec §11.5: the closed per-claim presentation kinds the renderer owns."""


class HandleManifestEntry(ContractModel):
    """Spec §8.4/§13.4: one opaque ``E``-handle bound to an exact use.

    Handles are assigned by the server in selected-evidence order before the
    first provider call and are never rebound on repair or resume.
    """

    handle: str
    use: EvidenceUseRef


class ParsedClaim(ContractModel):
    """Spec §9.1/§13.4: one validated model claim citing opaque handles only."""

    claim_id: str
    text: str
    handles: tuple[str, ...]
    presentation: ClaimPresentation


class ParsedCandidate(ContractModel):
    """Spec §13.4: the parsed, schema-valid claim proposal (phase=candidate).

    Claim ids are server-assigned (``claim-N``); the model's raw output is
    never persisted.
    """

    claims: tuple[ParsedClaim, ...]


class GroundedClaim(ContractModel):
    """Spec §10.2/§13.4: one claim grounded to exact resolved uses."""

    claim_id: str
    text: str
    uses: tuple[EvidenceUseRef, ...]
    presentation: ClaimPresentation


class PublicCitation(ContractModel):
    """Spec §11.3: the allowlisted public citation projection.

    Only presentation-safe fields cross this boundary; internal
    evidence/use/task/run/workspace/binding/revision/checkpoint identities are
    forbidden. ``index`` is the deterministic four-character public handle.
    """

    citation_id: str
    index: str
    label: str
    source_type: str
    score: float | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    content: str | None = None
    source_file: str | None = None
    page_no: int | None = None
    heading_path: tuple[str, ...] | None = None
    document_number: str | None = None
    article_label: str | None = None
    validity_status: str | None = None
    superseded_by: str | None = None


class GroundedArtifact(ContractModel):
    """Spec §13.4: the grounded, rendered, citation-projected answer.

    ``content`` is the server-rendered Markdown with public markers already
    inserted; ``citations`` is the allowlisted projector output replayed by
    SSE, persistence, and reload.
    """

    claims: tuple[GroundedClaim, ...]
    content: str
    citations: tuple[PublicCitation, ...]


class SynthesisCheckpoint(ContractModel):
    """Spec §13.4: the durable bounded-synthesis state machine checkpoint.

    ``None`` on the supervisor aggregate is the canonical idle value; a
    non-null checkpoint always carries the stable handle manifest plus the
    phase-appropriate payload. ``attempts_started`` increments and checkpoints
    before every provider call, so a crash consumes the attempt.
    """

    contract_version: ContractVersion
    phase: SynthesisPhase
    attempts_started: Literal[0, 1, 2]
    handle_manifest: tuple[HandleManifestEntry, ...]
    candidate: ParsedCandidate | None = None
    grounded: GroundedArtifact | None = None
    failure_code: str | None = None

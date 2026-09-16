"""Claim-first grounding for the bounded synthesis state machine.

Spec §10.1–10.2: ``ParsedClaim[]`` resolves its opaque ``E``-handles through
the checkpointed handle manifest into exact ``EvidenceUseRef``s, re-validates
claim structure, rehydrates the cited uses under current authority when a
checkpoint boundary was crossed, runs the deterministic anchor guards, and
emits ``GroundedClaim[]``. Production never parses rendered Markdown back
into claim identity — grounding happens on the parsed claims themselves.

Every failure is a closed code on :class:`SynthesisGroundingError`; the codes
are content-free (no claim text, evidence text, or internal identifiers
beyond the server-assigned ``claim-N`` id).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..contracts.evidence import EvidenceUseRef
from ..contracts.synthesis import (
    GroundedClaim,
    HandleManifestEntry,
    ParsedCandidate,
    ParsedClaim,
)
from ..nodes.evaluate import HydratedEvidence
from ..nodes.grounding import split_assertions
from .anchors import extract_anchors, unsupported_anchors

__all__ = [
    "SynthesisGroundingError",
    "ClaimGroundingResult",
    "ground_claims",
]

#: Spec §9.1: a claim cites one to three manifest handles.
_MAX_HANDLES_PER_CLAIM = 3


class SynthesisGroundingError(ValueError):
    """A closed-code grounding failure; the candidate enters repair/failure.

    ``code`` is one of the spec §16 closed failure codes
    (``unknown_evidence_handle``, ``handle_manifest_mismatch``,
    ``schema_invalid``, ``claim_multi_sentence``,
    ``claim_anchor_unsupported``, ``resume_revalidation_failed``). The
    message carries only the code and the server-assigned claim id — never
    claim text, evidence text, or model output.
    """

    def __init__(self, code: str, *, claim_id: str | None = None) -> None:
        self.code = code
        self.claim_id = claim_id
        detail = f" claim={claim_id}" if claim_id else ""
        super().__init__(f"synthesis grounding failed: {code}{detail}")


@dataclass(frozen=True)
class ClaimGroundingResult:
    """Grounded claims plus the cited hydrated evidence they resolve to.

    ``evidence`` is the union of cited uses in first-cited order — the exact
    input the citation projector needs to map ``EvidenceUseRef`` → source
    identity without a second store lookup.
    """

    claims: tuple[GroundedClaim, ...]
    evidence: tuple[HydratedEvidence, ...]


def _resolve_claim_uses(
    claim: ParsedClaim, manifest: tuple[HandleManifestEntry, ...]
) -> tuple[EvidenceUseRef, ...]:
    """Resolve the claim's handles through the checkpointed manifest only."""
    # Lazy import: synthesis/handles.py is owned by the manifest slice and
    # may land after this module; the import is resolved at call time.
    from .handles import HandleManifestError, resolve_handles

    try:
        return resolve_handles(claim.handles, manifest)
    except HandleManifestError as exc:
        raise SynthesisGroundingError(exc.code, claim_id=claim.claim_id) from exc


def _validate_claim_structure(claim: ParsedClaim) -> None:
    """Deterministic claim validation (spec §9.2/§10.5).

    The adapter already enforces these bounds; re-checking here keeps the
    guarantee honest when a checkpointed candidate is validated on resume.
    """
    if not claim.text.strip():
        raise SynthesisGroundingError("schema_invalid", claim_id=claim.claim_id)
    if not claim.handles:
        raise SynthesisGroundingError("schema_invalid", claim_id=claim.claim_id)
    if len(set(claim.handles)) > _MAX_HANDLES_PER_CLAIM:
        raise SynthesisGroundingError("schema_invalid", claim_id=claim.claim_id)
    if len(split_assertions(claim.text)) > 1:
        raise SynthesisGroundingError(
            "claim_multi_sentence", claim_id=claim.claim_id
        )


async def _rehydrate(
    uses: tuple[EvidenceUseRef, ...],
    *,
    hydrator,
    runtime,
    plan,
    bindings,
) -> tuple[HydratedEvidence, ...]:
    """Rehydrate cited uses under current authority after a checkpoint
    boundary (spec §10.2/§13.5). Denied uses fail closed — a handle is never
    rebound to a surviving substitute."""
    if runtime is None or plan is None or bindings is None:
        raise ValueError(
            "rehydration requires runtime, plan, and bindings"
        )
    return await hydrator.hydrate_for_evaluation(
        uses, runtime=runtime, plan=plan, bindings=bindings
    )


async def ground_claims(
    candidate: ParsedCandidate | Iterable[ParsedClaim],
    manifest: tuple[HandleManifestEntry, ...],
    *,
    evidence: Iterable[HydratedEvidence] | None = None,
    hydrator=None,
    runtime=None,
    plan=None,
    bindings=None,
    literals: tuple[str, ...] = (),
) -> ClaimGroundingResult:
    """Ground parsed claims to exact resolved uses.

    :param candidate: the parsed candidate (or bare claim tuple) to ground.
    :param manifest: the checkpointed ``E1..En`` handle manifest.
    :param evidence: hydrated evidence for the manifest uses (fresh path).
    :param hydrator: when supplied, the cited uses are rehydrated under
        current authority (resume path) and the fresh result replaces
        ``evidence`` for those uses.
    :param literals: configured closed literals for anchor extraction.

    :raises SynthesisGroundingError: closed failure code; never leaks content.
    """
    claims = (
        tuple(candidate.claims)
        if isinstance(candidate, ParsedCandidate)
        else tuple(candidate)
    )
    if not claims:
        raise SynthesisGroundingError("schema_invalid")
    seen_ids: set[str] = set()
    for claim in claims:
        if claim.claim_id in seen_ids:
            raise SynthesisGroundingError(
                "schema_invalid", claim_id=claim.claim_id
            )
        seen_ids.add(claim.claim_id)
        _validate_claim_structure(claim)

    resolved: list[tuple[ParsedClaim, tuple[EvidenceUseRef, ...]]] = []
    cited_order: list[EvidenceUseRef] = []
    for claim in claims:
        uses = _resolve_claim_uses(claim, manifest)
        resolved.append((claim, uses))
        for ref in uses:
            if ref not in cited_order:
                cited_order.append(ref)

    if hydrator is not None:
        hydrated = await _rehydrate(
            tuple(cited_order),
            hydrator=hydrator,
            runtime=runtime,
            plan=plan,
            bindings=bindings,
        )
    else:
        hydrated = tuple(evidence or ())
    by_use_id = {item.use_id: item for item in hydrated}

    grounded: list[GroundedClaim] = []
    cited_evidence: list[HydratedEvidence] = []
    for claim, uses in resolved:
        items: list[HydratedEvidence] = []
        for ref in uses:
            item = by_use_id.get(ref.use_id)
            # A cited use that fails current ACL/revision/expiry/tombstone/
            # lineage rehydration fails closed; the handle is never rebound.
            if item is None or item.purpose == "discovery":
                raise SynthesisGroundingError(
                    "resume_revalidation_failed", claim_id=claim.claim_id
                )
            items.append(item)
        item_anchors = tuple(
            extract_anchors(item.content, literals=literals) for item in items
        )
        failing = unsupported_anchors(
            extract_anchors(claim.text, literals=literals), item_anchors
        )
        if failing:
            raise SynthesisGroundingError(
                "claim_anchor_unsupported", claim_id=claim.claim_id
            )
        grounded.append(
            GroundedClaim(
                claim_id=claim.claim_id,
                text=claim.text,
                uses=uses,
                presentation=claim.presentation,
            )
        )
        seen_use_ids = {item.use_id for item in cited_evidence}
        for item in items:
            if item.use_id not in seen_use_ids:
                seen_use_ids.add(item.use_id)
                cited_evidence.append(item)
    return ClaimGroundingResult(
        claims=tuple(grounded), evidence=tuple(cited_evidence)
    )

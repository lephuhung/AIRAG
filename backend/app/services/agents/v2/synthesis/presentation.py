"""Server-owned presentation policy (spec §6.1).

A deterministic pure function selects the presentation strategy from the
checkpointed route/plan plus the kinds of evidence admitted by governed
hydration. The model never chooses this strategy, and the decision path
itself never does I/O: every input is already checkpointed state or the
admitted ``HydratedEvidence`` projection. ``resolve_derived_lineage`` is the
optional pre-resolution seam: callers with store access (the synthesis
graph's prepare node) resolve non-admitted derived lineage through the same
``resolve_lineage`` the CitationProjector uses and hand the resulting map
in, so the decision stays pure.

Rules (spec §6.1):

- document retrieval, section reads, summarize-final output, and complex
  final answers whose supporting claims all resolve to document-backed
  sources -> ``document_grounded_llm``;
- People lookup -> ``people_card`` (the existing card path; the synthesis
  model is never invoked);
- direct/clarify/denied/insufficient -> ``direct`` (existing non-LLM
  presentation);
- KG-only evidence, or any required target whose admitted evidence cannot
  resolve to document-backed locatable lineage -> ``typed_unavailable`` —
  never a fabricated document citation;
- derived evidence is eligible only when every cited lineage source
  recursively projects to document identity — admitted items first, then
  the store-resolved ``lineage`` map for members outside the admitted set
  (spec §8.3).
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from ..contracts.evidence import DerivedSourceIdentity, DocumentSourceIdentity

if TYPE_CHECKING:
    from ..contracts.evaluation import EvidenceEvaluation
    from ..contracts.evidence import EvidenceSourceIdentity
    from ..contracts.planning import TaskPlan
    from ..contracts.routing import RouteDecision
    from ..nodes.evaluate import HydratedEvidence

__all__ = [
    "PresentationMode",
    "decide_presentation",
    "document_backed_use_ids",
    "resolve_derived_lineage",
]


PresentationMode = Literal[
    "document_grounded_llm",
    "people_card",
    "direct",
    "typed_unavailable",
]
"""Spec §6.1: the closed set of server-owned presentation strategies."""

#: Mirrors the CitationProjector's expansion bound (spec §11.2): lineage
#: deeper than this can never be projected into a document citation, so it
#: is not document-backed. Resolution and the backed check both fail closed
#: at the same bound.
_MAX_LINEAGE_DEPTH = 16


async def resolve_derived_lineage(
    evidence: tuple[HydratedEvidence, ...],
    resolve_lineage: Callable[
        [UUID], Awaitable[EvidenceSourceIdentity | None]
    ],
) -> dict[UUID, EvidenceSourceIdentity]:
    """Resolve non-admitted derived lineage to typed source identities.

    Breadth-first over the ``source_evidence_ids`` of admitted derived items,
    through the same ``resolve_lineage`` callable the CitationProjector's
    resolver exposes (``CitationResolver.resolve_lineage``), so the backed
    check verifies exactly the lineage the projector will later expand
    (spec §8.3, §11.2). Members already in ``evidence`` are never resolved —
    their admitted ``source_identity`` is authoritative. Unresolvable ids
    are simply absent from the result, and traversal stops at the same
    depth bound the projector enforces, so callers fail closed either way.
    """
    admitted_ids = {item.evidence_id for item in evidence}
    resolved: dict[UUID, EvidenceSourceIdentity] = {}
    seen: set[UUID] = set()
    frontier: list[tuple[UUID, int]] = []
    for item in evidence:
        source = item.source_identity
        if isinstance(source, DerivedSourceIdentity):
            for evidence_id in dict.fromkeys(source.source_evidence_ids):
                if evidence_id not in admitted_ids and evidence_id not in seen:
                    seen.add(evidence_id)
                    frontier.append((evidence_id, 1))
    while frontier:
        evidence_id, depth = frontier.pop(0)
        source = await resolve_lineage(evidence_id)
        if source is None:
            continue
        resolved[evidence_id] = source
        if isinstance(source, DerivedSourceIdentity) and (
            depth < _MAX_LINEAGE_DEPTH
        ):
            for child_id in dict.fromkeys(source.source_evidence_ids):
                if child_id not in admitted_ids and child_id not in seen:
                    seen.add(child_id)
                    frontier.append((child_id, depth + 1))
    return resolved


def document_backed_use_ids(
    evidence: tuple[HydratedEvidence, ...],
    *,
    lineage: Mapping[UUID, EvidenceSourceIdentity] | None = None,
) -> frozenset[UUID]:
    """Use ids whose source recursively projects to document lineage.

    A ``document`` identity is backed by definition. A ``derived`` identity
    is backed only when every ``source_evidence_ids`` entry resolves to an
    identity that is itself backed — admitted items first, then the
    store-resolved ``lineage`` map for members outside the admitted set
    (the overflow tail persisted by ``hydrate_for_synthesis``; spec §8.3).
    ``people``, ``knowledge_graph``, and ``memory`` identities are never
    document-backed: they cannot produce a locatable document citation in
    this phase, so they must not cross the synthesis boundary as claim
    support. Unresolvable, cyclic, or over-deep lineage is never backed —
    fail closed, never a fabricated citation.
    """
    lineage = lineage or {}
    by_evidence_id = {item.evidence_id: item for item in evidence}
    heights: dict[UUID, int | None] = {}

    def _height(evidence_id: UUID, stack: frozenset[UUID]) -> int | None:
        """Minimum derivation depth to document sources; None = unbacked.

        ``None`` covers every unverifiable case: unresolvable lineage, a
        non-document leaf, cyclic lineage, or a chain deeper than the
        projector's expansion bound. Heights are entry-independent, so
        memoization is sound.
        """
        if evidence_id in heights:
            return heights[evidence_id]
        item = by_evidence_id.get(evidence_id)
        source = (
            item.source_identity if item is not None else lineage.get(evidence_id)
        )
        if source is None:
            # Lineage references an evidence record that is neither admitted
            # nor resolvable under current authority: not verifiably
            # document-backed.
            heights[evidence_id] = None
            return None
        if isinstance(source, DocumentSourceIdentity):
            heights[evidence_id] = 0
            return 0
        if not isinstance(source, DerivedSourceIdentity):
            heights[evidence_id] = None
            return None
        if not source.source_evidence_ids or evidence_id in stack:
            # Empty or cyclic lineage can never fully project to documents.
            heights[evidence_id] = None
            return None
        child_heights = [
            height
            for source_id in source.source_evidence_ids
            if (height := _height(source_id, stack | {evidence_id}))
            is not None
        ]
        if len(child_heights) != len(source.source_evidence_ids):
            heights[evidence_id] = None
            return None
        total = 1 + max(child_heights)
        heights[evidence_id] = (
            total if total <= _MAX_LINEAGE_DEPTH else None
        )
        return heights[evidence_id]

    return frozenset(
        item.use_id
        for item in evidence
        if _height(item.evidence_id, frozenset()) is not None
    )


def decide_presentation(
    *,
    route: RouteDecision | None,
    plan: TaskPlan | None,
    evidence: tuple[HydratedEvidence, ...],
    evaluation: EvidenceEvaluation | None = None,
    lineage: Mapping[UUID, EvidenceSourceIdentity] | None = None,
) -> PresentationMode:
    """Deterministic presentation decision over route/plan/evidence kinds.

    Pure: no I/O, no model, no clock. ``evidence`` is the admitted hydrated
    set (post-governor); discovery-purpose uses are ignored because they can
    never support claims. ``evaluation`` is optional context — a
    non-``sufficient`` verdict keeps the existing non-LLM presentation.
    """
    if route is None:
        return "typed_unavailable"
    if route.route in ("direct", "clarify"):
        return "direct"
    if route.reason_code == "simple_write_operation":
        # Denied write operations keep the existing typed terminal.
        return "direct"
    if route.reason_code == "simple_people_lookup":
        return "people_card"
    if evaluation is not None and evaluation.status != "sufficient":
        return "direct"

    claim_eligible = tuple(
        item for item in evidence if item.purpose != "discovery"
    )
    backed = document_backed_use_ids(claim_eligible, lineage=lineage)
    backed_items = tuple(
        item for item in claim_eligible if item.use_id in backed
    )
    if not backed_items:
        if claim_eligible and all(
            item.source_identity.kind == "people" for item in claim_eligible
        ):
            return "people_card"
        return "typed_unavailable"

    required_target_ids = (
        tuple(unit.target_id for unit in plan.target_units)
        if plan is not None
        else ()
    )
    backed_targets = {
        item.target_id for item in backed_items if item.target_id is not None
    }
    if any(tid not in backed_targets for tid in required_target_ids):
        # A required target has no document-backed admitted evidence: never
        # answer one side or fabricate a citation (spec §6.1, §8.2).
        return "typed_unavailable"
    return "document_grounded_llm"

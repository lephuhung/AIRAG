"""Target-aware synthesis evidence selection (spec §8.2–8.3).

``SynthesisEvidenceSelector`` replaces the FIFO ``apply_budget_split`` head
selection on the LLM path. Admitted hydrated evidence is grouped by logical
target (``target_id``) plus a global/supporting group (``target_id is None``
or ``role == "supporting"``), preserving deterministic in-group order, then
filled in two passes:

1. coverage pass — at least one eligible item for every required target that
   has admitted evidence;
2. round-robin pass — further items across groups in stable order until the
   budget (minus explicit reserves) is exhausted.

The reserves account for system/schema instructions, the contextualized
query, evidence delimiters/handles, and bounded output tokens, so prompt +
evidence + reserved output never exceed the configured synthesis budget.

Fail-closed rule (spec §8.2): a required target that ends the selection with
zero selected items fails with the closed code ``selection_missing_target``
and an empty selection — the synthesis boundary must never silently answer
one side of a compare.

Eligibility mirrors the presentation policy: discovery-purpose uses are never
selected, and only document-backed items (``document`` identity, or
``derived`` whose lineage recursively resolves to document identity — via
the admitted set plus the store-resolved ``lineage`` map for members outside
it) may cross the synthesis boundary (spec §8.1, §8.3).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..contracts.synthesis import SynthesisRuntimeContext
from ..nodes.synthesize import estimate_tokens_for_chars
from .presentation import document_backed_use_ids

if TYPE_CHECKING:
    from ..nodes.evaluate import HydratedEvidence
    from ..contracts.evidence import EvidenceSourceIdentity

__all__ = [
    "SELECTION_MISSING_TARGET",
    "SelectionReserves",
    "SelectionResult",
    "SynthesisEvidenceSelector",
]


#: Closed failure code (spec §8.2): a required target had admitted evidence
#: but none survives in the selected prompt set. Typed data on
#: ``SelectionResult`` — never an exception carrying content.
SELECTION_MISSING_TARGET = "selection_missing_target"

#: Default reserves (spec §8.2): budget held back from evidence so the
#: system/schema instructions, contextualized query, evidence
#: delimiters/handles, and bounded output always fit alongside the selection.
DEFAULT_SYSTEM_RESERVE_CHARS = 4_000
DEFAULT_QUERY_RESERVE_CHARS = 2_000
DEFAULT_PER_ITEM_OVERHEAD_CHARS = 64
DEFAULT_OUTPUT_RESERVE_TOKENS = 4_000


@dataclass(frozen=True)
class SelectionReserves:
    """Explicit budget reserves subtracted before evidence selection.

    ``system_chars``/``query_chars`` reserve character budget for the
    system/schema instructions and the contextualized query;
    ``per_item_overhead_chars`` is charged per selected item for evidence
    delimiters and its opaque ``E``-handle; ``output_tokens`` reserves token
    budget for the bounded structured output.
    """

    system_chars: int = DEFAULT_SYSTEM_RESERVE_CHARS
    query_chars: int = DEFAULT_QUERY_RESERVE_CHARS
    per_item_overhead_chars: int = DEFAULT_PER_ITEM_OVERHEAD_CHARS
    output_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS


@dataclass(frozen=True)
class SelectionResult:
    """Deterministic selection outcome.

    ``selected`` preserves selection order (coverage picks first, then
    round-robin fill); on failure it is empty so a partial selection can
    never be used accidentally. ``failure_code`` is ``None`` on success or a
    closed code (``selection_missing_target``) on failure;
    ``missing_target_ids`` names the uncovered required targets.
    """

    selected: tuple[HydratedEvidence, ...] = ()
    failure_code: str | None = None
    missing_target_ids: tuple[str, ...] = ()


@dataclass
class _Group:
    """One selection group with a scan cursor (round-robin state)."""

    items: list[HydratedEvidence] = field(default_factory=list)
    cursor: int = 0


class SynthesisEvidenceSelector:
    """Deterministic target-aware selector over admitted hydrated evidence.

    Pure and side-effect free: the same inputs always produce the same
    ``SelectionResult``, so retries and resume re-derivation converge.
    """

    def __init__(self, *, reserves: SelectionReserves | None = None) -> None:
        self._reserves = reserves or SelectionReserves()

    def select(
        self,
        evidence: tuple[HydratedEvidence, ...],
        *,
        required_target_ids: tuple[str, ...],
        budget: SynthesisRuntimeContext,
        lineage: Mapping[UUID, EvidenceSourceIdentity] | None = None,
    ) -> SelectionResult:
        """Select claim-eligible evidence within budget.

        ``evidence`` is the admitted hydrated set in deterministic order;
        ``required_target_ids`` are the plan's target units that must be
        represented; ``budget`` is the runtime synthesis budget the reserves
        are subtracted from. ``lineage`` optionally carries store-resolved
        source identities for derived lineage outside the admitted set
        (``resolve_derived_lineage``); without it, derived items whose
        lineage is not co-admitted are not backed and never selected.
        """
        reserves = self._reserves
        char_budget = max(
            0,
            budget.max_total_chars
            - reserves.system_chars
            - reserves.query_chars,
        )
        token_budget = max(
            0,
            budget.max_total_tokens
            - reserves.output_tokens
            - estimate_tokens_for_chars(
                reserves.system_chars + reserves.query_chars
            ),
        )
        item_budget = max(0, budget.max_evidence_items)

        backed = document_backed_use_ids(evidence, lineage=lineage)
        eligible = [
            item
            for item in evidence
            if item.purpose != "discovery" and item.use_id in backed
        ]

        groups: dict[str | None, _Group] = {}
        for item in eligible:
            key = self._group_key(item)
            groups.setdefault(key, _Group()).items.append(item)

        # Stable group order: required targets first (in required order),
        # then other target groups in first-seen order, then the
        # global/supporting group last.
        group_order: list[str | None] = [
            tid for tid in required_target_ids if tid in groups
        ]
        group_order.extend(
            key
            for key in groups
            if key is not None and key not in group_order
        )
        if None in groups:
            group_order.append(None)

        selected: list[HydratedEvidence] = []
        selected_ids: set[UUID] = set()
        used_chars = 0
        used_tokens = 0

        def fits(item: HydratedEvidence) -> bool:
            cost_chars = len(item.content) + reserves.per_item_overhead_chars
            cost_tokens = estimate_tokens_for_chars(cost_chars)
            return (
                len(selected) + 1 <= item_budget
                and used_chars + cost_chars <= char_budget
                and used_tokens + cost_tokens <= token_budget
            )

        def take(item: HydratedEvidence) -> None:
            nonlocal used_chars, used_tokens
            selected.append(item)
            selected_ids.add(item.use_id)
            cost_chars = len(item.content) + reserves.per_item_overhead_chars
            used_chars += cost_chars
            used_tokens += estimate_tokens_for_chars(cost_chars)

        # Pass 1 — coverage: one eligible item per required target. The scan
        # covers every eligible item attributed to the target (a
        # ``supporting``-role item still counts for its target's coverage
        # even though it fills from the global group).
        for tid in required_target_ids:
            for item in eligible:
                if (
                    item.target_id == tid
                    and item.use_id not in selected_ids
                    and fits(item)
                ):
                    take(item)
                    break

        # Pass 2 — round-robin fill across groups in stable order. Each
        # group contributes at most one item per round; an item that does
        # not fit is skipped (the cursor advances) so a large item never
        # blocks smaller items behind it.
        progressed = True
        while progressed:
            progressed = False
            for key in group_order:
                group = groups[key]
                while group.cursor < len(group.items):
                    item = group.items[group.cursor]
                    group.cursor += 1
                    if item.use_id in selected_ids:
                        continue
                    if not fits(item):
                        continue
                    take(item)
                    progressed = True
                    break

        selected_targets = {
            item.target_id
            for item in selected
            if item.target_id is not None
        }
        missing = tuple(
            tid
            for tid in required_target_ids
            if tid not in selected_targets
        )
        if missing:
            return SelectionResult(
                selected=(),
                failure_code=SELECTION_MISSING_TARGET,
                missing_target_ids=missing,
            )
        return SelectionResult(selected=tuple(selected))

    @staticmethod
    def _group_key(item: HydratedEvidence) -> str | None:
        if item.target_id is None or item.role == "supporting":
            return None
        return item.target_id

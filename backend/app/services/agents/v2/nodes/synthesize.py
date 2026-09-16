"""Factual synthesis over admitted evidence (Phase 2, Task 4).

Factual/domain synthesis requires ``EvidenceEvaluation.status ==
'sufficient'`` — the boundary rejects every other status via
``validate_synthesis_input``. ``hydrate_for_synthesis`` resolves admitted
``EvidenceUseRef`` to ephemeral ``SynthesisEvidence`` under current
ACL/expiry/revision checks (the hydrator enforces them against the explicit
plan/bindings passed on every call); discovery-only uses are excluded, and
every ``AnswerClaim`` is validated to cite only the admitted current-run use
set.

Budget and overflow: the synthesis budget (``SynthesisRuntimeContext``) is
runtime policy supplied per call — Phase-2 nodes pass
``DEFAULT_SYNTHESIS_BUDGET``. Hydration returns the within-budget head plus a
persisted overflow-derived item. Overflow is never a silent truncation: the
tail is persisted as a validated DERIVED ``EvidenceRecord`` with source
lineage plus a new supporting ``EvidenceUse`` (supporting, so it never
creates read coverage), and the single shared ``synthesize_and_lease``
helper leases every freshly minted use (evidence-only) and commits the lease
before returning — every node path that synthesizes (synthesize, ground
resume, finalizer resume) goes through it, so no path mints uses unleased.
Both the record insert (idempotent by content-hash + source identity) and
the use append (idempotent by run/task/evidence/purpose/target) converge on
retry, so deterministic re-derivation never duplicates overflow artifacts.

Draft handoff (legacy presentation path): the non-LLM People presentation
stores the validated draft in the runtime-only ``AnswerDraftChannel``
(``RuntimeServices.answer_draft_channel``, keyed by run id, never
checkpointed); the finalizer consumes the grounded result. The production
document path is the bounded synthesis subgraph
(``synthesis/graph.py``) mounted as the supervisor's ``synthesize`` node —
it checkpoints a ``SynthesisCheckpoint`` per durable phase and never uses
the channel.

The helpers below are the deterministic extractive composition used by the
People presentation and compat tests: user-visible content preserves
first-seen whole evidence items, while grounding claims are built per
distinct normalized assertion and merge every supporting use. A bounded
model may instead supply a draft through the ``DraftBuilder`` seam, but it
receives only ``SynthesisInput`` plus ``SynthesisEvidence`` — never
authorization, plans, or bindings — and its output is validated against the
admitted set before use.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID

from langgraph.runtime import Runtime

from ..contracts.binding import DocumentBindingSet
from ..contracts.evidence import EvidenceUseRef
from ..contracts.planning import TaskPlan
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.synthesis import (
    AnswerClaim,
    AnswerDraft,
    SynthesisEvidence,
    SynthesisInput,
    SynthesisRuntimeContext,
)
from ..contracts.validation import (
    validate_answer_draft,
    validate_synthesis_input,
)
from .evaluate import EvidenceHydrator, HydratedEvidence

__all__ = [
    "SynthesisError",
    "DEFAULT_SYNTHESIS_BUDGET",
    "DraftBuilder",
    "SynthesisResult",
    "estimate_tokens_for_chars",
    "apply_budget_split",
    "hydrate_for_synthesis",
    "build_extractive_draft",
    "synthesize_answer",
    "synthesize_and_lease",
    "synthesize_node",
]


class SynthesisError(ValueError):
    """Synthesis inputs are not synthesizable; nothing may be drafted."""


#: Phase-2 node synthesis budget (runtime policy; T6/T7 may thread an
#: explicit budget through ``synthesize_answer`` instead).
DEFAULT_SYNTHESIS_BUDGET = SynthesisRuntimeContext(
    max_evidence_items=20,
    max_total_chars=200_000,
    max_total_tokens=50_000,
)


@runtime_checkable
class DraftBuilder(Protocol):
    """Bounded-model seam for draft composition over governed evidence.

    Receives only the validated ``SynthesisInput`` and the projected
    ``SynthesisEvidence`` — never authorization, plans, bindings, or tools.
    The returned draft is validated against the admitted use set, so the
    builder cannot smuggle uncited claims into a grounded answer.
    """

    async def build(
        self,
        synthesis_input: SynthesisInput,
        evidence: tuple[SynthesisEvidence, ...],
    ) -> AnswerDraft: ...


@dataclass(frozen=True)
class SynthesisResult:
    """The validated draft plus the admitted evidence that supports it."""

    draft: AnswerDraft
    evidence: tuple[HydratedEvidence, ...]
    admitted_use_ids: frozenset[UUID]


def estimate_tokens_for_chars(char_count: int) -> int:
    """Deterministic token over-approximation for budget accounting.

    Ceiling of ``chars / 4``: coarse but deterministic and monotone, so budget
    decisions converge across node re-derivations. Implementations (including
    the production hydrator) must use this helper rather than a tokenizer so
    the head/tail split is identical wherever it is computed.
    """
    if char_count <= 0:
        return 0
    return (char_count + 3) // 4


def apply_budget_split(
    evidence: tuple[HydratedEvidence, ...],
    budget: SynthesisRuntimeContext,
) -> tuple[tuple[HydratedEvidence, ...], tuple[HydratedEvidence, ...]]:
    """Split admitted evidence into a within-budget head and an overflow tail.

    Whole items only (an item is never split): items are included in order
    while the cumulative item count, character total, and deterministic token
    estimate all stay within budget. Deterministic in the inputs.
    """
    head: list[HydratedEvidence] = []
    chars = 0
    tokens = 0
    for item in evidence:
        item_tokens = estimate_tokens_for_chars(len(item.content))
        if (
            len(head) + 1 > budget.max_evidence_items
            or chars + len(item.content) > budget.max_total_chars
            or tokens + item_tokens > budget.max_total_tokens
        ):
            return tuple(head), tuple(evidence[len(head):])
        head.append(item)
        chars += len(item.content)
        tokens += item_tokens
    return tuple(head), ()


def _require_hydrator(runtime: GraphRuntimeContext) -> EvidenceHydrator:
    hydrator = runtime.services.evidence_hydrator
    if hydrator is None:
        raise SynthesisError(
            "no evidence_hydrator wired on runtime.services; refusing to "
            "synthesize without governed hydration"
        )
    return hydrator


def _project(item: HydratedEvidence) -> SynthesisEvidence:
    if not item.content or not item.content.strip():
        raise SynthesisError(
            f"hydrated use {item.use_id} carries no content; refusing to "
            "synthesize from an empty projection"
        )
    return SynthesisEvidence(
        use_id=item.use_id,
        content=item.content,
        role=item.role,
        target_id=item.target_id,
        source_label=item.source_label,
    )


async def hydrate_for_synthesis(
    use_refs: tuple[EvidenceUseRef, ...],
    runtime: GraphRuntimeContext,
    *,
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    budget: SynthesisRuntimeContext,
) -> tuple[SynthesisEvidence, ...]:
    """Resolve admitted uses to ephemeral synthesis projections.

    Discovery-only uses are excluded. The hydrator enforces current
    ACL/expiry/revision checks against the explicit plan/bindings plus the
    synthesis budget (persisting an overflow tail as derived evidence); this
    function projects the admitted rich items to the model-facing
    ``SynthesisEvidence`` shape. Freshly minted overflow uses are leased
    (evidence-only) and committed before returning, so this exported entry
    point never leaves uses unleased.
    """
    hydrator = _require_hydrator(runtime)
    hydrated = await hydrator.hydrate_for_synthesis(
        tuple(use_refs),
        runtime=runtime,
        plan=plan,
        bindings=bindings,
        budget=budget,
    )
    admitted: list[HydratedEvidence] = []
    for item in hydrated:
        if not isinstance(item, HydratedEvidence):
            raise SynthesisError(
                "evidence_hydrator returned a non-HydratedEvidence "
                "projection; refusing to synthesize"
            )
        if item.purpose == "discovery":
            continue
        admitted.append(item)
    await _lease_fresh_uses(
        {ref.use_id for ref in use_refs}, tuple(admitted), runtime
    )
    return tuple(_project(item) for item in admitted)


def build_extractive_draft(
    evidence: tuple[SynthesisEvidence, ...],
) -> AnswerDraft:
    """Deterministic extractive draft: one claim per distinct assertion.

    User-visible content preserves the prior whole-item composition: duplicate
    normalized contents merge and first-seen item text/order wins. Claims use
    the finer ``split_assertions`` granularity that grounding maps over;
    identical normalized assertions merge every supporting use in first-seen
    source order. This keeps answer rendering stable while preventing repeated
    markers and sentences in overlapping chunks from becoming spuriously
    ambiguous. Deterministic first-seen assertion order owns claim numbering.
    """
    if not evidence:
        raise SynthesisError(
            "no admitted synthesis evidence; a sufficient evaluation whose "
            "uses are all denied at synthesis time cannot be drafted"
        )
    # Lazy import: grounding imports this module at load time, so a
    # top-level import would be circular; at call time both are loaded and
    # the draft builder is guaranteed the exact mapping semantics.
    from .grounding import normalize_assertion, split_assertions

    texts: dict[str, str] = {}
    uses: dict[str, list[UUID]] = {}
    seen: dict[str, set[UUID]] = {}
    order: list[str] = []
    content_parts: list[str] = []
    content_seen: set[str] = set()
    for item in evidence:
        content_key = " ".join(item.content.split())
        if content_key not in content_seen:
            content_seen.add(content_key)
            content_parts.append(item.content)
        for assertion in split_assertions(item.content):
            key = normalize_assertion(assertion)
            if not key:
                continue
            if key not in texts:
                texts[key] = assertion
                uses[key] = []
                seen[key] = set()
                order.append(key)
            if item.use_id not in seen[key]:
                seen[key].add(item.use_id)
                uses[key].append(item.use_id)
    if not order:
        raise SynthesisError(
            "no admitted synthesis evidence; a sufficient evaluation whose "
            "uses are all denied at synthesis time cannot be drafted"
        )
    claims = tuple(
        AnswerClaim(
            claim_id=f"claim-{index + 1}",
            text=texts[key],
            evidence_use_ids=tuple(uses[key]),
        )
        for index, key in enumerate(order)
    )
    return AnswerDraft(content="\n\n".join(content_parts), claims=claims)


async def synthesize_answer(
    *,
    synthesis_input: SynthesisInput,
    runtime: GraphRuntimeContext,
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    budget: SynthesisRuntimeContext,
    draft_builder: DraftBuilder | None = None,
) -> SynthesisResult:
    """Validate the boundary, hydrate admitted evidence, and draft the answer.

    ``validate_synthesis_input`` rejects every non-``sufficient`` evaluation;
    the draft (extractive default or bounded ``draft_builder``) is validated
    against the admitted current-run use set before it is returned. The draft
    covers the within-budget head only (hydrator-appended overflow items are
    admitted for governance but would re-violate the budget if drafted),
    except the empty-head edge, which drafts from the derived items so a
    sufficient evaluation still yields an answer.
    """
    validate_synthesis_input(synthesis_input)
    hydrator = _require_hydrator(runtime)
    hydrated = await hydrator.hydrate_for_synthesis(
        tuple(synthesis_input.evidence_uses),
        runtime=runtime,
        plan=plan,
        bindings=bindings,
        budget=budget,
    )
    admitted = tuple(
        item for item in hydrated if item.purpose != "discovery"
    )
    for item in admitted:
        if not isinstance(item, HydratedEvidence):
            raise SynthesisError(
                "evidence_hydrator returned a non-HydratedEvidence "
                "projection; refusing to synthesize"
            )
    if not admitted:
        raise SynthesisError(
            "no admitted synthesis evidence; a sufficient evaluation whose "
            "uses are all denied at synthesis time cannot be drafted"
        )
    input_ids = {ref.use_id for ref in synthesis_input.evidence_uses}
    head_items = tuple(item for item in admitted if item.use_id in input_ids)
    draft_source = head_items or admitted
    projected = tuple(_project(item) for item in draft_source)
    if draft_builder is None:
        draft = build_extractive_draft(projected)
    else:
        draft = draft_builder.build(synthesis_input, projected)
        if inspect.isawaitable(draft):
            draft = await draft
        if not isinstance(draft, AnswerDraft):
            raise SynthesisError(
                "draft builder returned a non-AnswerDraft; refusing to use it"
            )
    admitted_ids = frozenset(item.use_id for item in admitted)
    validate_answer_draft(draft, admitted_ids)
    await _lease_fresh_uses(input_ids, admitted, runtime)
    return SynthesisResult(
        draft=draft, evidence=admitted, admitted_use_ids=admitted_ids
    )


def _synthesis_input_of(state: SupervisorV2State) -> SynthesisInput:
    evaluation = state["execution"].evidence_evaluation
    if evaluation is None or evaluation.status != "sufficient":
        raise SynthesisError(
            "factual synthesis requires a sufficient EvidenceEvaluation; "
            f"got {evaluation.status if evaluation is not None else None!r}"
        )
    # Task 7A: a summarize-reduce pass owns the admitted order — its
    # deterministic ReduceSpec lineage is the synthesis input when present.
    ordered = evaluation.synthesis_use_order
    if ordered is not None:
        return SynthesisInput(
            semantic=state["semantic"],
            evaluation=evaluation,
            evidence_uses=tuple(ordered),
        )
    seen: set[UUID] = set()
    refs: list[EvidenceUseRef] = []
    for result in state["execution"].task_results:
        for ref in result.evidence_uses:
            if ref.use_id not in seen:
                seen.add(ref.use_id)
                refs.append(ref)
    return SynthesisInput(
        semantic=state["semantic"],
        evaluation=evaluation,
        evidence_uses=tuple(refs),
    )


async def _commit_lease_session(runtime: GraphRuntimeContext) -> None:
    repo = runtime.services.retention_leases
    if repo is None:  # pragma: no cover - guarded by the caller
        raise SynthesisError(
            "synthesis retained a new evidence use but no retention-lease "
            "service is wired"
        )
    session = getattr(repo, "session", None)
    commit = getattr(session, "commit", None)
    if commit is None:
        raise SynthesisError(
            "retention-lease repository exposes no commitable session; the "
            "overflow-use lease cannot be committed"
        )
    result = commit()
    if inspect.isawaitable(result):
        await result


async def _lease_fresh_uses(
    input_ids: set[UUID],
    admitted: tuple[HydratedEvidence, ...],
    runtime: GraphRuntimeContext,
) -> None:
    """Lease overflow-derived uses (evidence-only) and commit before returning.

    Fresh use ids are exactly the admitted ids minus the input refs: the
    scheduler already leased every dispatch-created use, so only the
    hydrator-persisted overflow uses need a lease here. Re-derivation
    converges (record insert and use append are both idempotent), so a repeat
    call refreshes the same lease row instead of duplicating it. Called by
    every exported synthesize path, so no minting path leaves uses unleased.
    """
    fresh = [item for item in admitted if item.use_id not in input_ids]
    if not fresh:
        return
    repo = runtime.services.retention_leases
    if repo is None:
        raise SynthesisError(
            "synthesis retained a new evidence use but no retention-lease "
            "service is wired; refusing to leave it unleashed"
        )
    run_id = runtime.capability_runtime.run_id
    for item in fresh:
        leased = repo.acquire_or_refresh(run_id, None, item.use_id)
        if inspect.isawaitable(leased):
            await leased
    await _commit_lease_session(runtime)


async def synthesize_and_lease(
    *,
    synthesis_input: SynthesisInput,
    runtime: GraphRuntimeContext,
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    budget: SynthesisRuntimeContext,
    draft_builder: DraftBuilder | None = None,
) -> SynthesisResult:
    """The single shared node-facing synthesize path.

    Every node path that synthesizes (synthesize, ground resume, finalizer
    resume) goes through this helper; leasing itself lives in
    ``synthesize_answer``, so direct callers of the exported entry point
    are covered too.
    """
    return await synthesize_answer(
        synthesis_input=synthesis_input,
        runtime=runtime,
        plan=plan,
        bindings=bindings,
        budget=budget,
        draft_builder=draft_builder,
    )


async def synthesize_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Placeholder for the supervisor's ``synthesize`` node name.

    The production path is the bounded synthesis subgraph mounted by
    ``make_synthesis_boundary_node`` (``synthesis/graph.py``); this symbol
    only keeps the supervisor's node-name registry honest — the wrapped
    boundary replaces it at graph construction. The extractive helpers above
    (``build_extractive_draft``/``synthesize_answer``/``synthesize_and_lease``)
    remain for compat tests and the non-LLM People presentation.
    """
    _ = (state, runtime)
    return {}

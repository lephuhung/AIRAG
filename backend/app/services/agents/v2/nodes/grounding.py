"""Grounding and deterministic citations (Phase 2, Task 4).

Grounding requires every material factual assertion in ``AnswerDraft.content``
to map to exactly one declared ``AnswerClaim``: a normalized assertion maps
to a claim when it equals or is contained in exactly one claim text. Zero
matches (unmapped/unsupported) or multiple matches (ambiguous) fail the
assertion. On failure the bounded ``ReviseHook`` gets exactly one chance to
repair the draft (revise once); if the draft is still unmapped — or no
reviser is wired — grounding raises ``GroundingInsufficient`` instead of
emitting an ungrounded factual success.

``CitationRenderer`` (``render_citations``) runs only after grounding: it
assigns deterministic ``cite-{n}`` IDs in first-seen claim order over the
admitted evidence, so repeated renders converge. ``RenderedCitation`` carries
no claim ID by contract.

``ground_node`` consumes the draft from the runtime-only ``AnswerDraftChannel``
(normal path: no re-synthesis) and grounds it with no reviser wired in
Phase 2: success stores the grounded result back in the channel for the
finalizer and returns no state update, while a grounding failure checkpoints
the typed ``insufficient`` response this node owns (user-safe content: no
internal target/criterion identifiers). On a channel miss (restart between
nodes) it re-derives deterministically ONCE through the shared
``synthesize_and_lease`` helper — which also leases any freshly minted
overflow uses — instead of silently re-running on every node.
"""
from __future__ import annotations

import inspect
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID

from langgraph.runtime import Runtime

from ..contracts.base import CONTRACT_VERSION
from ..contracts.response import FinalResponse, RenderedCitation
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.synthesis import AnswerDraft
from ..contracts.validation import (
    validate_answer_draft,
    validate_final_response,
)
from .context import _context_of
from .evaluate import HydratedEvidence, _channel_of
from .execute import require_checkpointed_plan
from .synthesize import (
    DEFAULT_SYNTHESIS_BUDGET,
    SynthesisError,
    _synthesis_input_of,
    synthesize_and_lease,
)

__all__ = [
    "GroundingError",
    "GroundingInsufficient",
    "GroundingReport",
    "ReviseHook",
    "GroundedAnswer",
    "split_assertions",
    "normalize_assertion",
    "ground_answer",
    "render_citations",
    "ground_node",
]

#: Split content into assertions on blank lines and sentence boundaries.
_SENTENCE_SPLIT_RE = re.compile(r"\n+|(?<=[.!?…。？！])\s+")

#: User-safe insufficient content: internal target/criterion identifiers
#: never reach the user-facing response (diagnostics stay in logs/reports).
_INSUFFICIENT_CONTENT = "Không đủ căn cứ đã xác minh."


class GroundingError(ValueError):
    """The grounding input itself is invalid; nothing may be mapped."""


class GroundingInsufficient(ValueError):
    """Material assertions remain unmapped after the single revision chance."""

    def __init__(self, draft: AnswerDraft, report: "GroundingReport") -> None:
        super().__init__(
            "answer draft has unmapped factual assertions after one revision; "
            "refusing an ungrounded factual success"
        )
        self.draft = draft
        self.report = report


@dataclass(frozen=True)
class GroundingReport:
    """The deterministic assertion-to-claim mapping outcome."""

    mapped: tuple[tuple[str, str], ...]
    unmapped_assertions: tuple[str, ...]
    ambiguous_assertions: tuple[tuple[str, tuple[str, ...]], ...]


@runtime_checkable
class ReviseHook(Protocol):
    """Bounded-model seam: one repair chance for an unmapped draft.

    Receives the failing draft plus the deterministic mapping report and
    returns a replacement draft, which is re-validated and re-mapped exactly
    once. Never called more than once per ``ground_answer`` invocation.
    """

    async def __call__(
        self, draft: AnswerDraft, report: GroundingReport
    ) -> AnswerDraft: ...


@dataclass(frozen=True)
class GroundedAnswer:
    """A fully mapped draft with its deterministic citations."""

    draft: AnswerDraft
    citations: tuple[RenderedCitation, ...]
    admitted_use_ids: frozenset[UUID]


def normalize_assertion(text: str) -> str:
    """Normalized comparison form: NFC, collapsed whitespace, casefolded.

    Trailing sentence punctuation is stripped so an assertion split off its
    terminator still matches the claim text it came from.
    """
    collapsed = " ".join(unicodedata.normalize("NFC", text).split()).casefold()
    return re.sub(r"[.!?…。？！]+$", "", collapsed)


def split_assertions(content: str) -> tuple[str, ...]:
    """Split draft content into material factual assertions.

    Blank lines and sentence terminators delimit assertions; empty segments
    are dropped. Content without any delimiter is a single assertion.
    """
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(content)]
    assertions = tuple(part for part in parts if part)
    return assertions if assertions else (content.strip(),)


def _mapping_report(
    draft: AnswerDraft,
) -> GroundingReport:
    mapped: list[tuple[str, str]] = []
    unmapped: list[str] = []
    ambiguous: list[tuple[str, tuple[str, ...]]] = []
    for assertion in split_assertions(draft.content):
        normalized = normalize_assertion(assertion)
        candidates = tuple(
            claim.claim_id
            for claim in draft.claims
            if normalized
            and (
                normalized == normalize_assertion(claim.text)
                or normalized in normalize_assertion(claim.text)
            )
        )
        if len(candidates) == 1:
            mapped.append((assertion, candidates[0]))
        elif len(candidates) > 1:
            ambiguous.append((assertion, candidates))
        else:
            unmapped.append(assertion)
    return GroundingReport(
        mapped=tuple(mapped),
        unmapped_assertions=tuple(unmapped),
        ambiguous_assertions=tuple(ambiguous),
    )


def _is_clean(report: GroundingReport) -> bool:
    return not report.unmapped_assertions and not report.ambiguous_assertions


async def ground_answer(
    *,
    draft: AnswerDraft,
    evidence: tuple[HydratedEvidence, ...],
    revise: ReviseHook | None = None,
) -> GroundedAnswer:
    """Ground the draft against admitted evidence, revising at most once.

    The draft is fail-closed validated first: every claim use id must already
    be a member of the admitted current-run use set. On mapping failure the
    reviser (when wired) repairs once and the result is re-mapped without a
    further revision chance; otherwise ``GroundingInsufficient`` is raised.
    """
    admitted_use_ids = frozenset(item.use_id for item in evidence)
    validate_answer_draft(draft, admitted_use_ids)
    report = _mapping_report(draft)
    if _is_clean(report):
        return GroundedAnswer(
            draft=draft,
            citations=render_citations(draft, evidence),
            admitted_use_ids=admitted_use_ids,
        )
    if revise is not None:
        revised = revise(draft, report)
        if inspect.isawaitable(revised):
            revised = await revised
        if not isinstance(revised, AnswerDraft):
            raise GroundingError(
                "revision hook returned a non-AnswerDraft; refusing to use it"
            )
        try:
            validate_answer_draft(revised, admitted_use_ids)
        except Exception as exc:
            raise GroundingInsufficient(revised, _mapping_report(revised)) from exc
        second = _mapping_report(revised)
        if _is_clean(second):
            return GroundedAnswer(
                draft=revised,
                citations=render_citations(revised, evidence),
                admitted_use_ids=admitted_use_ids,
            )
        raise GroundingInsufficient(revised, second)
    raise GroundingInsufficient(draft, report)


def render_citations(
    draft: AnswerDraft, evidence: tuple[HydratedEvidence, ...]
) -> tuple[RenderedCitation, ...]:
    """Deterministic citation IDs over admitted evidence, after grounding.

    First-seen claim order owns the numbering (``cite-1``, ``cite-2``, …);
    re-renders converge. Labels come from hydration's source label with a
    stable evidence-id fallback. Claims may share evidence (merged duplicate
    contents repeat the same citation id).
    """
    by_use_id = {item.use_id: item for item in evidence}
    evidence_order: list[UUID] = []
    for claim in draft.claims:
        for use_id in claim.evidence_use_ids:
            item = by_use_id.get(use_id)
            if item is None:
                raise GroundingError(
                    f"claim {claim.claim_id} cites use {use_id}, which is not "
                    "in the admitted evidence set"
                )
            if item.evidence_id not in evidence_order:
                evidence_order.append(item.evidence_id)
    index = {evidence_id: position + 1 for position, evidence_id in enumerate(evidence_order)}
    label_by_evidence: dict[UUID, str] = {}
    for item in evidence:
        label_by_evidence.setdefault(
            item.evidence_id,
            item.source_label or f"evidence-{str(item.evidence_id)[:8]}",
        )
    citations: list[RenderedCitation] = []
    seen: set[UUID] = set()
    for claim in draft.claims:
        for use_id in claim.evidence_use_ids:
            evidence_id = by_use_id[use_id].evidence_id
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            citations.append(
                RenderedCitation(
                    citation_id=f"cite-{index[evidence_id]}",
                    evidence_id=evidence_id,
                    label=label_by_evidence[evidence_id],
                )
            )
    return tuple(citations)


def _insufficient_response() -> FinalResponse:
    response = FinalResponse(
        contract_version=CONTRACT_VERSION,
        status="insufficient",
        content=_INSUFFICIENT_CONTENT,
        citations=(),
    )
    validate_final_response(response)
    return response


async def ground_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Ground the channeled draft; checkpoint typed insufficiency on failure.

    Normal path consumes the draft ``synthesize_node`` stored in the
    runtime-only channel (no re-synthesis) and stores the grounded result
    back for the finalizer. No reviser is wired in Phase 2 — an unmapped
    draft checkpoints ``insufficient`` here rather than emitting factual
    success. On a channel miss the draft is re-derived deterministically
    ONCE through the shared ``synthesize_and_lease`` helper (fresh overflow
    uses leased before returning); a synthesis failure there also becomes a
    typed ``insufficient`` rather than an escaping exception.
    """
    context = _context_of(runtime)
    run_id = context.capability_runtime.run_id
    channel = _channel_of(context)
    entry = channel.get(run_id) if channel is not None else None
    if entry is not None and entry.synthesis_error is not None:
        return {"final_response": _insufficient_response()}
    if entry is not None and entry.draft is not None:
        draft, evidence = entry.draft, entry.evidence
    else:
        plan = require_checkpointed_plan(state)
        try:
            derived = await synthesize_and_lease(
                synthesis_input=_synthesis_input_of(state),
                runtime=context,
                plan=plan,
                bindings=state["bindings"],
                budget=DEFAULT_SYNTHESIS_BUDGET,
            )
        except SynthesisError:
            return {"final_response": _insufficient_response()}
        draft, evidence = derived.draft, derived.evidence
    try:
        grounded = await ground_answer(draft=draft, evidence=evidence)
    except GroundingInsufficient:
        return {"final_response": _insufficient_response()}
    if channel is not None:
        channel.store_grounded(
            run_id, draft=grounded.draft, citations=grounded.citations
        )
    return {}

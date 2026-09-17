"""The bounded grounded-LLM synthesis subgraph (spec §12–§13.5, plan Task 5).

Mounted as the supervisor's ``synthesize`` node through
:func:`make_synthesis_boundary_node`, mirroring the ``complex_boundary``
pattern: the subgraph is compiled WITHOUT a checkpointer so it inherits the
supervisor's saver (and therefore the shadow run's isolated saver).

Unlike the complex boundary, synthesis must checkpoint *inside* the boundary
before every provider call — a crash during generation consumes the reserved
attempt. The boundary therefore pins a stable child checkpoint namespace
(``"synthesis"``): the child's per-node checkpoints are durable under the
supervisor saver, and a boundary re-entry after a crash resumes the child at
its pending node (``aget_state(...).next``) instead of restarting it. A
``Command(update={"resumed": True})`` marks the resumed pass so ``generate``
can tell a consumed reservation from a fresh one — the reserved attempt is
never re-called.

Topology::

    START ──(entry route on synthesis.phase)──► prepare ──► reserve
        │                                        │            │
        │                                        │            ▼
        │                                        │         generate ──► validate_ground ──► decide
        │                                        │            ▲                              │
        │                                        │            └──── repair ◄─────────────────┤
        │                                        │                                           ▼
        │                                        └──────────────► failed / handled      finalize_artifact ──► END

Every durable phase transition is a node boundary writing
``{"synthesis": SynthesisCheckpoint(...)}``; the boundary merges the child's
``synthesis`` slot back onto the supervisor aggregate.

Non-``document_grounded_llm`` presentations never enter the state machine:
``people_card`` runs the existing non-LLM presentation (extractive draft +
grounding + channel handoff, zero model calls) and ``typed_unavailable`` /
``direct`` fail closed with ``presentation_unsupported_source``.
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, StreamWriter
from pydantic import ValidationError

from ..contracts.base import CONTRACT_VERSION
from ..contracts.binding import DocumentBindingSet
from ..contracts.evaluation import EvidenceEvaluation
from ..contracts.evidence import EvidenceUseRef
from ..contracts.planning import TaskPlan
from ..contracts.routing import RouteDecision
from ..contracts.semantic import SemanticContext
from ..contracts.state import (
    ExecutionState,
    GraphRuntimeContext,
    SupervisorV2State,
)
from ..contracts.synthesis import (
    GroundedArtifact,
    ParsedCandidate,
    SynthesisCheckpoint,
)
from ..contracts.validation import (
    validate_answer_draft,
    validate_synthesis_checkpoint,
)
from ..nodes.context import node_context
from ..nodes.evaluate import HydratedEvidence, _channel_of
from ..nodes.grounding import GroundingInsufficient, ground_answer
from ..nodes.synthesize import (
    DEFAULT_SYNTHESIS_BUDGET,
    SynthesisError,
    _lease_fresh_uses,
    _project,
    _synthesis_input_of,
    build_extractive_draft,
    synthesize_and_lease,
)
from .adapter import DraftBuildFailure, HandledEvidence
from .citations import CitationProjectionError, CitationProjector
from .grounding import SynthesisGroundingError, ground_claims
from .handles import build_handle_manifest
from .presentation import decide_presentation, resolve_derived_lineage
from .render import RenderError, render_grounded_claims
from app.services.agent.synthesis_tracing import record_synthesis_outcome
from .selection import SynthesisEvidenceSelector

__all__ = [
    "SynthesisSubgraphState",
    "build_synthesis_subgraph",
    "build_synthesis_state",
    "merge_synthesis_result_into_supervisor",
    "make_synthesis_boundary_node",
    "normalize_synthesis_state",
]

#: Stable child checkpoint namespace under the supervisor saver. Pinning the
#: namespace (instead of letting LangGraph mint a fresh ``boundary:<uuid>``
#: per invocation) is what makes the child's per-node checkpoints resumable:
#: a boundary re-entry after a crash finds the pending child checkpoint and
#: continues it instead of restarting — the attempt ledger survives.
_CHILD_CHECKPOINT_NS = "synthesis"

#: Closed failure codes (spec §16) owned by this module. The adapter,
#: selector, grounding, and projector emit their own closed codes.
PRESENTATION_UNSUPPORTED = "presentation_unsupported_source"
SELECTION_EMPTY = "selection_empty"
ATTEMPT_INTERRUPTED = "attempt_interrupted"
ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
DEADLINE_EXHAUSTED = "deadline_exhausted"
RESUME_REVALIDATION_FAILED = "resume_revalidation_failed"
CITATION_UNRESOLVABLE = "citation_unresolvable"
PROVIDER_ERROR = "provider_error"

#: Failures that may consume the single repair attempt (spec §12.3).
#: Authority/revalidation failures are never repairable — a denied use fails
#: closed rather than being substituted.
_REPAIRABLE_CODES = frozenset(
    {
        "malformed_json",
        "schema_invalid",
        "claim_limit_exceeded",
        "claim_multi_sentence",
        "unknown_evidence_handle",
        "handle_manifest_mismatch",
        "claim_anchor_unsupported",
        "citation_unresolvable",
        "provider_error",
        "provider_timeout",
        ATTEMPT_INTERRUPTED,
    }
)


class SynthesisSubgraphState(TypedDict, total=False):
    """Checkpointed child state for the bounded synthesis subgraph.

    ``synthesis`` is the durable ``SynthesisCheckpoint`` slot shared with the
    supervisor aggregate; the remaining fields are the parent→child input
    projection plus ephemeral intra-run handoffs (``pending_failure`` feeds
    the repair context, ``prior_candidate`` carries the safely parsed failed
    proposal, ``resumed`` marks a crash-resumed pass, ``handled`` marks a
    non-LLM presentation that produced no checkpoint).
    """

    contract_version: str
    run_id: str
    semantic: SemanticContext
    bindings: DocumentBindingSet
    route_decision: RouteDecision | None
    plan: TaskPlan | None
    evaluation: EvidenceEvaluation | None
    evidence_uses: tuple[EvidenceUseRef, ...]
    synthesis: SynthesisCheckpoint | None
    pending_failure: dict[str, Any] | None
    prior_candidate: ParsedCandidate | None
    resumed: bool
    handled: bool
    #: Internal routing hint written by nodes for the conditional edges;
    #: never consumed by the parent merge.
    next: str


class SynthesisGraphError(ValueError):
    """The synthesis boundary cannot proceed; fail closed, never default."""


# ---------------------------------------------------------------------------
# State coercion + small helpers
# ---------------------------------------------------------------------------


def _coerce_slot(value: Any, model: Any, *, slot: str) -> Any:
    """Re-validate one child slot (mirrors the supervisor/complex coercers)."""
    if value is None:
        return None
    if isinstance(value, model):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return model.model_validate_json(value.model_dump_json())
        except (ValidationError, TypeError, ValueError) as exc:
            raise SynthesisGraphError(
                f"synthesis slot {slot!r} does not re-validate as "
                f"{model.__name__}: {exc}"
            ) from exc
    if isinstance(value, Mapping):
        payload = dict(value)
        try:
            candidate = model.model_validate(payload)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return model.model_validate_json(candidate.model_dump_json())
        except ValidationError:
            pass
        try:
            return model.model_validate_json(json.dumps(payload, default=str))
        except (ValidationError, TypeError, ValueError) as exc:
            raise SynthesisGraphError(
                f"synthesis slot {slot!r} is not a valid "
                f"{model.__name__}: {exc}"
            ) from exc
    raise SynthesisGraphError(
        f"synthesis slot {slot!r} is not a {model.__name__} "
        f"(got {type(value).__name__})"
    )

def _coerce_pending_failure(value: Any) -> dict[str, Any] | None:
    """``pending_failure`` is a plain ``{"code", "claim_indexes"}`` mapping."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    raise SynthesisGraphError(
        "synthesis slot 'pending_failure' is not a mapping "
        f"(got {type(value).__name__})"
    )


def normalize_synthesis_state(
    state: SynthesisSubgraphState,
) -> SynthesisSubgraphState:
    """Coerce a resumed checkpoint mapping back into usable contracts.

    The child inherits the supervisor saver, so a resumed pass re-enters
    mid-graph with serde-degraded nested contracts (mappings/lists where
    live models stood). Every slot a node walks is re-validated here.
    """
    return SynthesisSubgraphState(
        contract_version=str(state.get("contract_version", CONTRACT_VERSION)),
        run_id=str(state.get("run_id", "")),
        semantic=_coerce_slot(
            state.get("semantic"), SemanticContext, slot="semantic"
        ),
        bindings=_coerce_slot(
            state.get("bindings"), DocumentBindingSet, slot="bindings"
        ),
        route_decision=_coerce_slot(
            state.get("route_decision"), RouteDecision, slot="route_decision"
        ),
        plan=_coerce_slot(state.get("plan"), TaskPlan, slot="plan"),
        evaluation=_coerce_slot(
            state.get("evaluation"), EvidenceEvaluation, slot="evaluation"
        ),
        evidence_uses=tuple(
            _coerce_slot(ref, EvidenceUseRef, slot="evidence_uses[]")
            for ref in (state.get("evidence_uses") or ())
        ),
        synthesis=_coerce_slot(
            state.get("synthesis"), SynthesisCheckpoint, slot="synthesis"
        ),
        pending_failure=_coerce_pending_failure(state.get("pending_failure")),
        prior_candidate=_coerce_slot(
            state.get("prior_candidate"), ParsedCandidate, slot="prior_candidate"
        ),
        resumed=bool(state.get("resumed", False)),
        handled=bool(state.get("handled", False)),
    )


def _checkpoint_of(state: SynthesisSubgraphState) -> SynthesisCheckpoint | None:
    """Return the live checkpoint, tolerating a serde-degraded mapping."""
    return _coerce_slot(
        state.get("synthesis"), SynthesisCheckpoint, slot="synthesis"
    )


def _deadline_remaining(context: GraphRuntimeContext) -> bool:
    """True when the shared turn deadline has not yet passed (§14.2)."""
    deadline = getattr(context.capability_runtime, "deadline_at", None)
    if deadline is None:
        return True
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return datetime.now(UTC) < deadline


def _failed_checkpoint(
    checkpoint: SynthesisCheckpoint | None, code: str
) -> SynthesisCheckpoint:
    """Build the terminal failed checkpoint, preserving manifest + attempts."""
    return SynthesisCheckpoint(
        contract_version=CONTRACT_VERSION,
        phase="failed",
        attempts_started=(
            checkpoint.attempts_started if checkpoint is not None else 0
        ),
        handle_manifest=(
            checkpoint.handle_manifest if checkpoint is not None else ()
        ),
        failure_code=code,
    )


def _failure(code: str, *claim_indexes: int) -> dict[str, Any]:
    return {"code": code, "claim_indexes": list(claim_indexes)}


def _claim_index(claim_id: str | None) -> int | None:
    """``claim-N`` -> ``N - 1``: the 0-based claim position.

    ``claim_indexes`` are 0-based everywhere they surface (the adapter emits
    the parsed claim's positional index, and the repair prompt's
    ``prior_candidate`` claims serialize as a JSON array), so the repair
    prompt's claim indexes carry a single convention.
    """
    if not claim_id:
        return None
    try:
        index = int(str(claim_id).rsplit("-", 1)[1]) - 1
    except (IndexError, ValueError):
        return None
    return index if index >= 0 else None


def _require_builder(context: GraphRuntimeContext) -> Any:
    builder = context.services.answer_draft_builder
    if builder is None:
        raise SynthesisGraphError(
            "no answer_draft_builder wired on runtime.services; refusing to "
            "synthesize without the bounded-model seam"
        )
    return builder


def _require_projector(context: GraphRuntimeContext) -> CitationProjector:
    resolver = getattr(context.services, "citation_resolver", None)
    if resolver is None:
        raise SynthesisGraphError(
            "no citation_resolver wired on runtime.services; refusing to "
            "project citations without the authoritative resolver"
        )
    return CitationProjector(resolver)


# ---------------------------------------------------------------------------
# People presentation (existing non-LLM path — spec §6.1)
# ---------------------------------------------------------------------------


async def _run_people_presentation(
    *,
    synthesis_input,
    runtime: GraphRuntimeContext,
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    admitted: tuple[HydratedEvidence, ...] | None = None,
) -> None:
    """Run the existing non-LLM People presentation end to end.

    People evidence never crosses the synthesis-model boundary (spec §15);
    the existing presentation is the deterministic extractive draft +
    assertion grounding handed to the finalizer through the runtime-only
    ``AnswerDraftChannel``. No ``SynthesisCheckpoint`` is produced — the
    synthesis state machine did not run.

    ``admitted`` reuses the caller's already-governed hydration result
    (post-discovery-filter); when omitted the shared synthesize path
    hydrates under the same budget itself.
    """
    channel = _channel_of(runtime)
    if channel is None:
        raise SynthesisError(
            "people presentation requires the answer_draft_channel handoff; "
            "refusing to silently drop the answer"
        )
    run_id = runtime.capability_runtime.run_id
    try:
        if admitted is not None:
            # Reuse the caller's hydration: no second governed hydration
            # call. Mirror synthesize_answer's draft source — the
            # within-budget head, or the derived items when the head is
            # empty — and lease nothing twice (the caller already leased).
            input_ids = {ref.use_id for ref in synthesis_input.evidence_uses}
            head_items = tuple(
                item for item in admitted if item.use_id in input_ids
            )
            draft = build_extractive_draft(
                tuple(_project(item) for item in (head_items or admitted))
            )
            # Same draft contract as synthesize_answer: claims may cite
            # admitted use ids only.
            validate_answer_draft(
                draft, frozenset(item.use_id for item in admitted)
            )
            evidence = admitted
        else:
            result = await synthesize_and_lease(
                synthesis_input=synthesis_input,
                runtime=runtime,
                plan=plan,
                bindings=bindings,
                budget=DEFAULT_SYNTHESIS_BUDGET,
            )
            draft, evidence = result.draft, result.evidence
        grounded = await ground_answer(draft=draft, evidence=evidence)
        # v1 parity: the people card answer is the consolidated profile
        # text the v1 search already produced (captured on the shared
        # request-scoped people service during dispatch), not the
        # minimized-JSON extractive concatenation. Applied to the GROUNDED
        # draft so a grounding revision cannot rebuild content from the
        # claim texts and drop it; claims/citations are untouched — they
        # still project from the governed evidence uses. A missing
        # snapshot keeps the extractive content.
        people_lookup = getattr(runtime.services, "people_lookup", None)
        snapshot = (
            people_lookup.people_display_snapshot()
            if people_lookup is not None
            and hasattr(people_lookup, "people_display_snapshot")
            else None
        )
        grounded_draft = grounded.draft
        if snapshot is not None:
            _persons, display = snapshot
            if display:
                grounded_draft = grounded_draft.model_copy(
                    update={"content": display}
                )
    except (SynthesisError, GroundingInsufficient) as exc:
        channel.store_failure(run_id, reason=str(exc))
        return
    channel.store_draft(run_id, draft=draft, evidence=evidence)
    channel.store_grounded(
        run_id, draft=grounded_draft, citations=grounded.citations
    )


# ---------------------------------------------------------------------------
# Nodes — each durable phase transition is a node boundary
# ---------------------------------------------------------------------------


async def prepare_node(
    state: SynthesisSubgraphState, runtime: Any
) -> dict:
    """Presentation policy → governed hydration → selection → manifest.

    Writes the ``prepared`` checkpoint (phase boundary) on success. A
    non-``document_grounded_llm`` mode never enters the state machine:
    ``people_card`` runs the existing non-LLM presentation inline (``handled``,
    no checkpoint); ``typed_unavailable``/``direct`` become a pending closed
    failure the ``failed`` node checkpoints.
    """
    context = node_context(runtime)
    state = normalize_synthesis_state(state)
    plan = state.get("plan")
    evaluation = state.get("evaluation")
    if plan is None or evaluation is None:
        return {
            "pending_failure": _failure(PRESENTATION_UNSUPPORTED),
            "next": "failed",
        }

    # Route-level people presentation: no hydration needed to decide, and
    # the existing path hydrates internally — zero wasted governed calls.
    route = state.get("route_decision")
    if route is not None and route.reason_code == "simple_people_lookup":
        await _run_people_presentation(
            synthesis_input=_child_synthesis_input(state),
            runtime=context,
            plan=plan,
            bindings=state["bindings"],
        )
        return {"handled": True, "next": "end"}

    hydrator = context.services.evidence_hydrator
    if hydrator is None:
        raise SynthesisError(
            "no evidence_hydrator wired on runtime.services; refusing to "
            "synthesize without governed hydration"
        )
    hydrated = await hydrator.hydrate_for_synthesis(
        tuple(state.get("evidence_uses") or ()),
        runtime=context,
        plan=plan,
        bindings=state["bindings"],
        budget=DEFAULT_SYNTHESIS_BUDGET,
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
    input_ids = {ref.use_id for ref in state.get("evidence_uses") or ()}
    await _lease_fresh_uses(input_ids, tuple(admitted), context)

    # §8.3: resolve derived lineage beyond the admitted set through the
    # same resolver the CitationProjector uses, so a derived item whose
    # ancestors were trimmed by the budget can still prove document
    # backing. No resolver -> no lineage map -> derived items stay
    # unbacked and fail closed (the projector would fail closed anyway).
    citation_resolver = getattr(context.services, "citation_resolver", None)
    lineage = (
        await resolve_derived_lineage(
            tuple(admitted), citation_resolver.resolve_lineage
        )
        if citation_resolver is not None
        else None
    )

    mode = decide_presentation(
        route=route,
        plan=plan,
        evidence=tuple(admitted),
        evaluation=evaluation,
        lineage=lineage,
    )
    if mode == "people_card":
        # All-People evidence on a non-people route keeps the existing
        # presentation; the synthesis model is never invoked. The hydration
        # above is reused — a second governed hydration call would be pure
        # waste on this path.
        await _run_people_presentation(
            synthesis_input=_child_synthesis_input(state),
            runtime=context,
            plan=plan,
            bindings=state["bindings"],
            admitted=tuple(admitted),
        )
        return {"handled": True, "next": "end"}
    if mode != "document_grounded_llm":
        return {
            "pending_failure": _failure(PRESENTATION_UNSUPPORTED),
            "next": "failed",
        }

    selection = SynthesisEvidenceSelector().select(
        tuple(admitted),
        required_target_ids=tuple(
            unit.target_id for unit in plan.target_units
        ),
        budget=DEFAULT_SYNTHESIS_BUDGET,
        lineage=lineage,
    )
    if selection.failure_code is not None:
        return {
            "pending_failure": _failure(selection.failure_code),
            "next": "failed",
        }
    if not selection.selected:
        return {
            "pending_failure": _failure(SELECTION_EMPTY),
            "next": "failed",
        }
    if context.services.answer_draft_builder is None:
        # Missing bounded-model seam: fail before any attempt is consumed.
        return {
            "pending_failure": _failure(PROVIDER_ERROR),
            "next": "failed",
        }

    manifest = build_handle_manifest(selection.selected)
    checkpoint = SynthesisCheckpoint(
        contract_version=CONTRACT_VERSION,
        phase="prepared",
        attempts_started=0,
        handle_manifest=manifest,
    )
    validate_synthesis_checkpoint(checkpoint)
    return {"synthesis": checkpoint, "next": "reserve"}


def _child_synthesis_input(state: SynthesisSubgraphState):
    """Build the SynthesisInput for the legacy presentation path."""
    from ..contracts.synthesis import SynthesisInput

    return SynthesisInput(
        semantic=state["semantic"],
        evaluation=state["evaluation"],
        evidence_uses=tuple(state.get("evidence_uses") or ()),
    )


async def reserve_node(
    state: SynthesisSubgraphState, runtime: Any
) -> dict:
    """Consume one attempt: ``attempts_started += 1`` checkpointed BEFORE the
    provider call (spec §12.1). The reservation is durable, so a crash during
    generation consumes the attempt."""
    context = node_context(runtime)
    state = normalize_synthesis_state(state)
    checkpoint = _checkpoint_of(state)
    if checkpoint is None:
        raise SynthesisGraphError(
            "reserve requires a prepared synthesis checkpoint"
        )
    if checkpoint.attempts_started >= 2:
        return {
            "pending_failure": _failure(ATTEMPT_BUDGET_EXHAUSTED),
            "next": "failed",
        }
    if not _deadline_remaining(context):
        # §13.5: a prepared-phase resume reserves only while the deadline
        # remains — an expired deadline fails closed without consuming an
        # attempt or calling the provider.
        return {
            "pending_failure": _failure(DEADLINE_EXHAUSTED),
            "next": "failed",
        }
    reserved = checkpoint.model_copy(
        update={
            "phase": "attempt_reserved",
            "attempts_started": checkpoint.attempts_started + 1,
            "candidate": None,
            "grounded": None,
            "failure_code": None,
        }
    )
    validate_synthesis_checkpoint(reserved)
    # ``resumed`` is consumed by the first resumed node; a fresh reservation
    # always clears it so a later generate never misreads a stale flag.
    return {"synthesis": reserved, "resumed": False, "next": "generate"}


async def generate_node(
    state: SynthesisSubgraphState, runtime: Any, writer: StreamWriter = None
) -> dict:
    """One provider call via the privacy-safe builder.

    The outcome is checkpointed as a parsed bounded ``ParsedCandidate``
    (phase ``candidate``) or surfaced as a pending closed failure — never
    raw output. On a crash-resumed pass (``resumed``) the reservation is
    already consumed: no provider call, the interruption becomes the pending
    failure the repair decision evaluates.

    Speculative claim streaming: when LangGraph injects a ``writer`` (the
    parent runs with ``astream`` + ``subgraphs=True``), claim text pieces
    (``synthesis.speculative_delta``) and completed claims
    (``synthesis.speculative_claim``) are written to the custom stream as
    advisory payloads; a repair pass emits ``synthesis.speculative_reset``
    first.
    Under plain ``ainvoke`` the writer is a no-op. Nothing partial is ever
    checkpointed — only the parsed candidate lands on ``synthesis``.
    """
    context = node_context(runtime)
    state = normalize_synthesis_state(state)
    checkpoint = _checkpoint_of(state)
    if checkpoint is None or checkpoint.phase != "attempt_reserved":
        raise SynthesisGraphError(
            "generate requires an attempt_reserved synthesis checkpoint"
        )
    if state.get("resumed"):
        # The reserved provider call was consumed by the crashed pass.
        return {
            "pending_failure": _failure(ATTEMPT_INTERRUPTED),
            "resumed": False,
            "next": "validate_ground",
        }

    builder = _require_builder(context)
    hydrator = context.services.evidence_hydrator
    if hydrator is None:
        raise SynthesisError(
            "no evidence_hydrator wired on runtime.services; refusing to "
            "synthesize without governed hydration"
        )
    # Rehydrate the exact manifest uses under current authority; a denied
    # use fails closed — a handle is never rebound to a substitute (§13.5).
    hydrated = await hydrator.hydrate_for_evaluation(
        tuple(entry.use for entry in checkpoint.handle_manifest),
        runtime=context,
        plan=state["plan"],
        bindings=state["bindings"],
    )
    by_use_id = {item.use_id: item for item in hydrated}
    evidence_items: list[HandledEvidence] = []
    for entry in checkpoint.handle_manifest:
        item = by_use_id.get(entry.use.use_id)
        if item is None or item.purpose == "discovery":
            return {
                "pending_failure": _failure(RESUME_REVALIDATION_FAILED),
                "next": "validate_ground",
            }
        evidence_items.append(
            HandledEvidence(
                handle=entry.handle, use=entry.use, content=item.content
            )
        )

    pending = state.get("pending_failure")
    repair_context = None
    if pending is not None:
        repair_context = {
            "failure_codes": [pending["code"]],
            "claim_indexes": list(pending.get("claim_indexes") or ()),
        }
        prior = state.get("prior_candidate")
        if isinstance(prior, ParsedCandidate):
            repair_context["prior_candidate"] = prior

    from ..events import (
        SYNTHESIS_SPECULATIVE_CLAIM,
        SYNTHESIS_SPECULATIVE_DELTA,
        SYNTHESIS_SPECULATIVE_RESET,
    )

    emit = writer if callable(writer) else None
    if emit is not None and repair_context is not None:
        emit({"kind": SYNTHESIS_SPECULATIVE_RESET})
    on_claim = None
    on_delta = None
    if emit is not None:
        def on_claim(index: int, presentation: str, text: str) -> None:
            emit(
                {
                    "kind": SYNTHESIS_SPECULATIVE_CLAIM,
                    "index": index,
                    "presentation": presentation,
                    "text": text,
                }
            )

        def on_delta(index: int, presentation: str, text: str) -> None:
            emit(
                {
                    "kind": SYNTHESIS_SPECULATIVE_DELTA,
                    "index": index,
                    "presentation": presentation,
                    "text": text,
                }
            )

    result = await builder.build(
        state["semantic"].contextualized_query,
        tuple(evidence_items),
        repair_context=repair_context,
        on_claim=on_claim,
        on_delta=on_delta,
    )
    if isinstance(result, DraftBuildFailure):
        return {
            "pending_failure": _failure(result.code, *result.claim_indexes),
            "next": "validate_ground",
        }
    if not isinstance(result, ParsedCandidate):
        raise SynthesisGraphError(
            "answer_draft_builder returned neither ParsedCandidate nor "
            "DraftBuildFailure; refusing to checkpoint unknown output"
        )
    candidate = checkpoint.model_copy(
        update={"phase": "candidate", "candidate": result}
    )
    validate_synthesis_checkpoint(candidate)
    return {
        "synthesis": candidate,
        "pending_failure": None,
        "prior_candidate": None,
        "next": "validate_ground",
    }


async def validate_ground_node(
    state: SynthesisSubgraphState, runtime: Any
) -> dict:
    """Manifest resolution → claim/anchor validation → rehydration under
    current authority → citation projection → render → ``grounded``
    checkpoint. A pending failure skips grounding entirely and flows to the
    repair/failed decision."""
    context = node_context(runtime)
    state = normalize_synthesis_state(state)
    if state.get("pending_failure") is not None:
        return {"next": "decide"}
    checkpoint = _checkpoint_of(state)
    if checkpoint is None:
        raise SynthesisGraphError(
            "validate_ground requires a synthesis checkpoint"
        )

    hydrator = context.services.evidence_hydrator
    if hydrator is None:
        raise SynthesisError(
            "no evidence_hydrator wired on runtime.services; refusing to "
            "ground without governed hydration"
        )

    if checkpoint.phase == "grounded":
        # Resume rule (§13.5): zero model calls — rehydrate the exact cited
        # uses and re-run current authorization + citation projection before
        # the artifact may be (re-)emitted. Any failure is terminal, never
        # repairable: the model is not re-asked around changed authority.
        grounded = checkpoint.grounded
        if grounded is None:
            raise SynthesisGraphError(
                "grounded checkpoint carries no grounded artifact"
            )
        try:
            cited: list[EvidenceUseRef] = []
            for claim in grounded.claims:
                for ref in claim.uses:
                    if ref not in cited:
                        cited.append(ref)
            hydrated = await hydrator.hydrate_for_evaluation(
                tuple(cited),
                runtime=context,
                plan=state["plan"],
                bindings=state["bindings"],
            )
            by_use_id = {item.use_id: item for item in hydrated}
            if any(ref.use_id not in by_use_id for ref in cited):
                raise SynthesisGroundingError(RESUME_REVALIDATION_FAILED)
            projector = _require_projector(context)
            # Re-projection is a revalidation gate only: the emitted
            # artifact keeps the checkpointed citations verbatim so
            # citation identity stays stable across a resume (a re-emitted
            # answer must not silently renumber or relabel its citations).
            # Resolver drift therefore never rewrites the artifact — it can
            # only fail closed via CitationProjectionError.
            await projector.project(
                grounded.claims,
                tuple(by_use_id[ref.use_id] for ref in cited),
            )
        except (SynthesisGroundingError, CitationProjectionError):
            return {
                "pending_failure": _failure(RESUME_REVALIDATION_FAILED),
                "next": "decide",
            }
        return {"next": "finalize_artifact"}

    if checkpoint.phase != "candidate" or checkpoint.candidate is None:
        raise SynthesisGraphError(
            f"validate_ground cannot run on phase {checkpoint.phase!r}"
        )

    try:
        result = await ground_claims(
            checkpoint.candidate,
            checkpoint.handle_manifest,
            hydrator=hydrator,
            runtime=context,
            plan=state["plan"],
            bindings=state["bindings"],
        )
        projector = _require_projector(context)
        projection = await projector.project(result.claims, result.evidence)
        content = render_grounded_claims(result.claims, projection)
    except SynthesisGroundingError as exc:
        index = _claim_index(exc.claim_id)
        return {
            "pending_failure": _failure(
                exc.code, *(() if index is None else (index,))
            ),
            "prior_candidate": checkpoint.candidate,
            "next": "decide",
        }
    except (CitationProjectionError, RenderError):
        return {
            "pending_failure": _failure(CITATION_UNRESOLVABLE),
            "prior_candidate": checkpoint.candidate,
            "next": "decide",
        }

    artifact = GroundedArtifact(
        claims=result.claims,
        content=content,
        citations=projection.citations,
    )
    grounded_checkpoint = checkpoint.model_copy(
        update={
            "phase": "grounded",
            "candidate": None,
            "grounded": artifact,
        }
    )
    validate_synthesis_checkpoint(grounded_checkpoint)
    return {
        "synthesis": grounded_checkpoint,
        "pending_failure": None,
        "prior_candidate": None,
        "next": "finalize_artifact",
    }


async def interrupted_node(
    state: SynthesisSubgraphState, runtime: Any
) -> dict:
    """Resume rule for ``attempt_reserved`` (§13.5): the reserved call is
    consumed — record the closed interruption code for the repair decision."""
    node_context(runtime)
    _ = normalize_synthesis_state(state)
    return {
        "pending_failure": _failure(ATTEMPT_INTERRUPTED),
        "next": "decide",
    }


async def decide_node(
    state: SynthesisSubgraphState, runtime: Any
) -> dict:
    """Repair/failed decision (spec §12.3–12.4): one repair when the failure
    is repairable, the attempt budget remains, and the turn deadline leaves
    room; otherwise the ``failed`` node checkpoints the closed code."""
    context = node_context(runtime)
    state = normalize_synthesis_state(state)
    pending = state.get("pending_failure")
    if pending is None:
        raise SynthesisGraphError(
            "decide requires a pending failure; refusing to guess a route"
        )
    code = pending.get("code")
    if not code:
        raise SynthesisGraphError(
            "pending failure carries no closed code; refusing to guess"
        )
    checkpoint = _checkpoint_of(state)
    attempts = checkpoint.attempts_started if checkpoint is not None else 0
    repairable = code in _REPAIRABLE_CODES
    deadline_ok = _deadline_remaining(context)
    if repairable and attempts < 2 and deadline_ok:
        return {"next": "reserve"}
    if repairable and attempts < 2 and not deadline_ok:
        # A repairable failure with no deadline left is terminal as
        # deadline_exhausted — the honest closed code for the terminal.
        return {
            "pending_failure": _failure(DEADLINE_EXHAUSTED),
            "next": "failed",
        }
    return {"next": "failed"}


async def failed_node(
    state: SynthesisSubgraphState, runtime: Any
) -> dict:
    """Checkpoint the terminal closed failure (phase ``failed``)."""
    node_context(runtime)
    state = normalize_synthesis_state(state)
    checkpoint = _checkpoint_of(state)
    pending = state.get("pending_failure")
    code = (
        pending.get("code")
        if pending is not None
        else (checkpoint.failure_code if checkpoint is not None else None)
    )
    if not code:
        raise SynthesisGraphError(
            "failed node requires a pending failure or a failed checkpoint"
        )
    failed = _failed_checkpoint(checkpoint, code)
    validate_synthesis_checkpoint(failed)
    # §16: every terminal synthesis failure emits the content-free outcome
    # record — including codes the adapter never observes
    # (selection_empty, presentation_unsupported_source,
    # resume_revalidation_failed, deadline_exhausted, citation_unresolvable,
    # attempt_interrupted). Adapter-observed codes get a second, terminal
    # record; the per-attempt record stays the attempt-level signal.
    record_synthesis_outcome(
        outcome="failed",
        failure_code=code,
        repair=failed.attempts_started > 1,
    )
    return {
        "synthesis": failed,
        "pending_failure": None,
        "prior_candidate": None,
        "next": "end",
    }


async def finalize_artifact_node(
    state: SynthesisSubgraphState, runtime: Any
) -> dict:
    """Terminal success boundary: the ``grounded`` checkpoint is already the
    final_response-bound artifact the supervisor merge carries back."""
    node_context(runtime)
    state = normalize_synthesis_state(state)
    checkpoint = _checkpoint_of(state)
    if checkpoint is None or checkpoint.phase != "grounded":
        raise SynthesisGraphError(
            "finalize_artifact requires a grounded synthesis checkpoint"
        )
    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _entry_route(state: SynthesisSubgraphState) -> str:
    """Fresh-run entry routing on the checkpointed phase (spec §13.5).

    Mid-run resumes bypass this entirely — LangGraph continues the child at
    its pending node, which is exactly the phase the checkpoint recorded.
    """
    checkpoint = state.get("synthesis")
    phase = getattr(checkpoint, "phase", None)
    if phase is None and isinstance(checkpoint, Mapping):
        phase = checkpoint.get("phase")
    if phase == "prepared":
        return "reserve"
    if phase == "attempt_reserved":
        return "interrupted"
    if phase == "candidate":
        return "validate_ground"
    if phase == "grounded":
        return "validate_ground"
    if phase == "failed":
        return "failed"
    return "prepare"


def _prepare_next(state: SynthesisSubgraphState) -> str:
    return state.get("next") or "reserve"


def _generate_next(state: SynthesisSubgraphState) -> str:
    return state.get("next") or "validate_ground"


def _validate_next(state: SynthesisSubgraphState) -> str:
    return state.get("next") or "decide"


def _decide_next(state: SynthesisSubgraphState) -> str:
    return state.get("next") or "failed"


def _failed_next(state: SynthesisSubgraphState) -> str:
    return "end"


def _build_synthesis_graph() -> StateGraph:
    graph = StateGraph(
        SynthesisSubgraphState, context_schema=GraphRuntimeContext
    )
    graph.add_node("prepare", prepare_node)
    graph.add_node("reserve", reserve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate_ground", validate_ground_node)
    graph.add_node("interrupted", interrupted_node)
    graph.add_node("decide", decide_node)
    graph.add_node("failed", failed_node)
    graph.add_node("finalize_artifact", finalize_artifact_node)
    graph.add_conditional_edges(
        START,
        _entry_route,
        {
            "prepare": "prepare",
            "reserve": "reserve",
            "interrupted": "interrupted",
            "validate_ground": "validate_ground",
            "failed": "failed",
        },
    )
    graph.add_conditional_edges(
        "prepare",
        _prepare_next,
        {"reserve": "reserve", "failed": "failed", "end": END},
    )
    graph.add_conditional_edges(
        "reserve",
        _generate_next,
        {"generate": "generate", "failed": "failed"},
    )
    graph.add_conditional_edges(
        "generate",
        _generate_next,
        {"validate_ground": "validate_ground", "failed": "failed"},
    )
    graph.add_conditional_edges(
        "validate_ground",
        _validate_next,
        {
            "decide": "decide",
            "finalize_artifact": "finalize_artifact",
            "failed": "failed",
        },
    )
    graph.add_edge("interrupted", "decide")
    graph.add_conditional_edges(
        "decide", _decide_next, {"reserve": "reserve", "failed": "failed"}
    )
    graph.add_conditional_edges("failed", _failed_next, {"end": END})
    graph.add_edge("finalize_artifact", END)
    return graph


def build_synthesis_subgraph() -> Any:
    """Compile the bounded synthesis subgraph WITHOUT a checkpointer.

    Attached as the supervisor's ``synthesize`` node it inherits the
    supervisor's saver (and therefore the shadow run's isolated saver)
    rather than opening its own.
    """
    return _build_synthesis_graph().compile()


# ---------------------------------------------------------------------------
# Parent <-> child mapping + boundary node
# ---------------------------------------------------------------------------


def build_synthesis_state(
    state: SupervisorV2State, *, run_id: str
) -> SynthesisSubgraphState:
    """Explicit parent -> child mapping (checkpointed fields only)."""
    execution = state["execution"]
    return SynthesisSubgraphState(
        contract_version=state["contract_version"],
        run_id=run_id,
        semantic=state["semantic"],
        bindings=state["bindings"],
        route_decision=state["route_decision"],
        plan=execution.plan,
        evaluation=execution.evidence_evaluation,
        evidence_uses=_synthesis_input_of(state).evidence_uses,
        synthesis=state["synthesis"],
        pending_failure=None,
        prior_candidate=None,
        resumed=False,
        handled=False,
    )


def merge_synthesis_result_into_supervisor(
    state: SupervisorV2State, child: SynthesisSubgraphState
) -> dict:
    """Merge the child's ``synthesis`` slot back onto the parent aggregate."""
    _ = state
    return {"synthesis": child.get("synthesis")}


def make_synthesis_boundary_node(synthesis_subgraph: Any) -> Any:
    """Wrap the compiled subgraph as the supervisor's ``synthesize`` node.

    The boundary pins the child's checkpoint namespace so a crash mid-run
    leaves a resumable pending checkpoint; a re-entry resumes it (marking
    ``resumed`` so a consumed attempt is never re-called), merges a
    completed-but-unmerged child output, or starts a fresh run for the
    ``grounded``/``failed`` resume rules.
    """

    async def synthesize_boundary_node(
        state: SupervisorV2State,
        runtime: Any,
        config: Any = None,
    ) -> dict:
        from langgraph.config import get_config

        context = node_context(runtime)
        execution = _coerce_slot(
            state.get("execution"), ExecutionState, slot="execution"
        )
        evaluation = execution.evidence_evaluation if execution else None
        if evaluation is None or evaluation.status != "sufficient":
            # Non-sufficient verdicts own existing typed terminals at the
            # finalizer; the synthesis state machine never starts.
            return {}

        base = config if config is not None else get_config()
        base_configurable = dict(base.get("configurable", {}))
        # The boundary node's own resume marker: True only when the parent
        # crashed mid-``synthesize`` and is re-running this node. Strip it
        # (and any replay state) from the child config — the child's resume
        # is driven explicitly by the pending-checkpoint branch below.
        parent_resuming = bool(
            base_configurable.pop("__pregel_resuming", False)
        )
        base_configurable.pop("__pregel_replay_state", None)
        child_config = {
            **base,
            "configurable": {
                **base_configurable,
                "checkpoint_ns": _CHILD_CHECKPOINT_NS,
            },
        }
        snapshot = await synthesis_subgraph.aget_state(child_config)
        pending = bool(
            snapshot and snapshot.next and parent_resuming
        )
        if pending:
            # Resume the crashed child pass at its pending node. The
            # ``resumed`` mark lets ``generate`` treat a standing
            # attempt_reserved checkpoint as consumed.
            resumed_generate = "generate" in tuple(snapshot.next)
            child_output = await synthesis_subgraph.ainvoke(
                Command(update={"resumed": resumed_generate}),
                config=child_config,
                context=context,
            )
        elif (
            parent_resuming
            and snapshot
            and snapshot.values
            and snapshot.values.get("synthesis") is not None
        ):
            # The child completed but the merge never checkpointed (crash
            # between child END and the boundary write): merge the finished
            # output instead of re-running. Only reachable on a parent
            # resume — a fresh turn never trusts a stale completed child.
            child_output = dict(snapshot.values)
        else:
            child_input = build_synthesis_state(
                state, run_id=context.capability_runtime.run_id
            )
            child_output = await synthesis_subgraph.ainvoke(
                child_input, config=child_config, context=context
            )
        child_output = normalize_synthesis_state(child_output)
        return merge_synthesis_result_into_supervisor(state, child_output)

    return synthesize_boundary_node

"""Evaluator authority: coverage, missing requirements, status (Phase 2, Task 4).

``evaluate_evidence`` is the shared FUNCTION that owns the sufficiency verdict
(amendment §7: there is deliberately NO ``EvidenceEvaluator`` service, and
``RuntimeServices`` excludes it). Deterministic rules own coverage,
revision/locator compatibility, ACL/expiry/source validity (via hydration
admission plus the evaluator's own revision re-check), criterion presence,
and status precedence::

    hydrate (governed admission only)
    -> build_coverage (plan × bindings × observations × admitted uses)
    -> find_missing_requirements (coverage + semantic-criterion verdicts)
    -> task-evidence gate (every evidence-supplying task needs an admitted use)
    -> analyze_contradictions_bounded (judge-assisted, deterministic precedence)
    -> needs_input > contradictory > insufficient > sufficient

Sufficiency is coupled to evidence availability: a targetless plan whose task
was denied/errored/``not_found`` (zero admitted uses) is ``insufficient``,
never ``sufficient``; a plan with zero tasks is ``insufficient``. Coverage
confirmation additionally re-checks the hydrated record revision against the
pinned binding revision, so a revision mismatch cannot complete coverage even
if a hydrator admitted the use.

A bounded model may assist ONLY semantic-criterion/contradiction
interpretation through the optional ``SemanticJudge`` seam, and only over
already-governed evidence: the judge receives hydrated content plus the
criterion — never authorization, bindings, revisions, plans, or tools — and it
cannot self-certify factual success (deterministic coverage, the
task-evidence gate, and precedence still gate ``sufficient``).
``evaluate_node`` wires no judge in Phase 2; the seam exists for T7
production wiring.

The typed ``EvidenceHydrator`` Protocol is defined here: hydration for
evaluation plus hydration for synthesis (plus the overflow-derived persist the
synthesis budget requires). Following the ``execute_node`` precedent, the
current ``TaskPlan``, ``DocumentBindingSet``, ``SynthesisRuntimeContext``
budget, and runtime authorization travel as explicit call arguments (the plan
is created mid-run, so wiring-time feeding cannot supply it). Denials are
SKIPPED (never raised): the admitted set shrinks and the resulting missing
requirements name the gap, so nothing is silently dropped. A missing
``evidence_hydrator`` service fails closed.

The runtime-only ephemeral handoff (``AnswerDraftChannel``) is also defined
here so ``synthesize``/``grounding``/``finalizer`` share one leaf owner
without an import cycle: the synthesize node stores the validated draft keyed
by run id, the ground node consumes it, and the finalizer consumes the
grounded result. The channel is never checkpointed and is not a frozen
contract; on a miss (e.g. a restart between nodes) the consumer re-derives
deterministically ONCE from checkpointed state.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable
from uuid import UUID

from langgraph.runtime import Runtime

from ..contracts.binding import DocumentBindingSet, ScopedDocument
from ..contracts.evaluation import (
    Contradiction,
    Coverage,
    CoverageItem,
    CoverageStatus,
    EvidenceEvaluation,
    MissingRequirement,
)
from ..contracts.evidence import EvidenceClassification, EvidencePurpose, EvidenceSourceIdentity, EvidenceUseRef
from ..contracts.execution import AgentResult
from ..contracts.locators import (
    ArticleLocator,
    ChunkRangeLocator,
    ContentLocator,
    DocumentLocator,
    PageRangeLocator,
    SectionLocator,
)
from ..contracts.planning import (
    CoverageCriterion,
    SemanticCriterion,
    TargetUnit,
    TaskPlan,
)
from ..contracts.response import RenderedCitation
from ..contracts.semantic import SemanticContext
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.synthesis import AnswerDraft, SynthesisRuntimeContext
from ..contracts.validation import validate_evidence_evaluation
from .context import _context_of
from .execute import execution_update, require_checkpointed_plan

__all__ = [
    "EvaluationError",
    "HydratedEvidence",
    "EvidenceHydrator",
    "SemanticJudge",
    "AnswerDraftChannel",
    "ChannelEntry",
    "locator_covers",
    "build_coverage",
    "find_missing_requirements",
    "every_expecting_task_has_evidence",
    "has_needs_input_result",
    "has_blocking_conflict",
    "analyze_contradictions_bounded",
    "evaluate_evidence",
    "evaluate_node",
]


class EvaluationError(ValueError):
    """Evaluation cannot produce a verdict; nothing may be certified."""


@dataclass(frozen=True)
class HydratedEvidence:
    """Ephemeral governed projection of one admitted current-run use.

    Runtime-internal (never checkpointed, never model-facing): ``role`` is
    computed from the admitted use plus the canonical plan/binding stores,
    ``locator``/``classification``/``document_revision`` travel with the
    content so deterministic coverage and the synthesis budget need no second
    store lookup, and ``document_revision`` lets the evaluator re-check the
    pinned revision independently of hydration admission. ``target_id`` is the
    admitted use's own target; discovery uses never hydrate. ``source_identity``
    is the typed ``EvidenceSourceIdentity`` resolved from the admitted
    ``EvidenceRecord`` (R29): it lets server-side consumers (e.g. the
    People→Document materializer) verify the source kind without trusting the
    human-readable ``source_label``, which remains the ONLY source descriptor
    that reaches model-facing projections. The typed identity itself is never
    serialized into a model prompt. It defaults to ``None`` for
    backward-compatible doubles; production hydration always populates it,
    and consumers MUST treat a missing identity as unverified (fail closed).
    """

    use_id: UUID
    evidence_id: UUID
    task_id: str
    purpose: EvidencePurpose
    target_id: str | None
    content: str
    role: Any
    source_label: str | None
    classification: EvidenceClassification
    locator: ContentLocator | None
    document_revision: str | None = None
    source_identity: EvidenceSourceIdentity | None = None


@dataclass
class ChannelEntry:
    """One run's ephemeral synthesis/grounding handoff (never checkpointed)."""

    draft: AnswerDraft | None = None
    evidence: tuple[HydratedEvidence, ...] = ()
    grounded_draft: AnswerDraft | None = None
    citations: tuple[RenderedCitation, ...] = ()
    synthesis_error: str | None = None


class AnswerDraftChannel:
    """Runtime-only synthesize → ground → finalizer handoff, keyed by run id.

    Lives on ``RuntimeServices.answer_draft_channel`` (wired by T6/T7); it is
    never checkpointed and is not a frozen contract. Normal path: exactly one
    synthesize stores the draft, one ground consumes it and stores the
    grounded result, the finalizer consumes the grounded result and synthesizes
    nothing. Resume safety: the store overwrites per run id, and a consumer
    that misses (restart between nodes, unwired channel) re-derives
    deterministically ONCE from checkpointed state instead of silently
    re-running on every node.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ChannelEntry] = {}

    def store_draft(
        self,
        run_id: str,
        *,
        draft: AnswerDraft,
        evidence: tuple[HydratedEvidence, ...],
    ) -> None:
        """Store this run's validated draft (overwrites any prior entry)."""
        self._entries[run_id] = ChannelEntry(draft=draft, evidence=evidence)

    def store_grounded(
        self,
        run_id: str,
        *,
        draft: AnswerDraft,
        citations: tuple[RenderedCitation, ...],
    ) -> None:
        """Store this run's grounded result alongside its draft/evidence."""
        current = self._entries.get(run_id) or ChannelEntry()
        self._entries[run_id] = ChannelEntry(
            draft=current.draft,
            evidence=current.evidence,
            grounded_draft=draft,
            citations=citations,
        )

    def store_failure(self, run_id: str, *, reason: str) -> None:
        """Record a synthesis failure so downstream nodes type it, not redo it.

        The drafted result stays absent; the ground node checkpoints its
        owned ``insufficient`` without re-deriving, and the finalizer emits
        the denied/insufficient/error response from the checkpointed task
        outcomes. Nothing re-synthesizes on this entry.
        """
        current = self._entries.get(run_id) or ChannelEntry()
        self._entries[run_id] = ChannelEntry(
            draft=current.draft,
            evidence=current.evidence,
            grounded_draft=current.grounded_draft,
            citations=current.citations,
            synthesis_error=reason,
        )

    def get(self, run_id: str) -> ChannelEntry | None:
        """Return this run's entry, or ``None`` on a channel miss."""
        return self._entries.get(run_id)


@runtime_checkable
class EvidenceHydrator(Protocol):
    """Typed hydration boundary owned by the Evidence Store.

    The concrete adapter (``evidence_store/hydration.py``,
    ``GovernorEvidenceHydrator``) enforces current ACL/expiry/tombstone/
    revision/derived validation through the Phase-1 ``EvidenceGovernor``;
    ``hydrate_for_synthesis`` additionally enforces the synthesis budget and
    persists an overflow tail as validated derived evidence (never a silent
    truncation). Denied uses are skipped (audited by the implementation) and
    absent from the returned admitted set. Concrete governor/session wiring is
    injected by T6/T7.
    """

    async def hydrate_for_evaluation(
        self,
        use_refs: tuple[EvidenceUseRef, ...],
        *,
        runtime: GraphRuntimeContext,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
    ) -> tuple[HydratedEvidence, ...]:
        """Hydrate admitted uses for evaluation (content + locator + class)."""
        ...

    async def hydrate_for_synthesis(
        self,
        use_refs: tuple[EvidenceUseRef, ...],
        *,
        runtime: GraphRuntimeContext,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
        budget: SynthesisRuntimeContext,
    ) -> tuple[HydratedEvidence, ...]:
        """Hydrate admitted uses for synthesis within budget (+overflow item)."""
        ...

    async def persist_derived_summary(
        self,
        *,
        content: str,
        source_evidence_ids: tuple[UUID, ...],
        task_id: str,
        target_id: str | None,
        provenance: Any,
        classification: EvidenceClassification,
        run_id: str,
        plan: TaskPlan,
        bindings: DocumentBindingSet,
    ) -> HydratedEvidence:
        """Persist a validated DERIVED record + supporting use; return it hydrated."""
        ...


@runtime_checkable
class SemanticJudge(Protocol):
    """Bounded-model seam for semantic interpretation over governed evidence.

    The judge receives hydrated content and the criterion under judgment —
    never authorization, bindings, revisions, plans, or tools — and its output
    is advisory: deterministic coverage, the task-evidence gate, and status
    precedence still gate the verdict, so it cannot self-certify factual
    success.
    """

    async def assess_criterion(
        self,
        *,
        criterion: SemanticCriterion,
        evidence: tuple[HydratedEvidence, ...],
    ) -> bool:
        """True when the semantic criterion is satisfied by the evidence."""
        ...

    async def detect_contradictions(
        self,
        *,
        evidence: tuple[HydratedEvidence, ...],
    ) -> tuple[Contradiction, ...]:
        """Conflicts between admitted current-run uses (empty when none)."""
        ...


#: Capabilities whose tasks are expected to supply synthesis-eligible
#: evidence. Discovery-only capabilities (``document.search``) and
#: non-evidence capabilities never satisfy the task-evidence gate.
_EVIDENCE_SUPPLYING_CAPABILITIES = frozenset(
    {
        "document.read",
        "section.read",
        "people.lookup",
        "knowledge_graph.query",
        "memory.lookup",
    }
)


def _require_hydrator(runtime: GraphRuntimeContext) -> EvidenceHydrator:
    hydrator = runtime.services.evidence_hydrator
    if hydrator is None:
        raise EvaluationError(
            "no evidence_hydrator wired on runtime.services; refusing to "
            "certify evidence without governed hydration"
        )
    return hydrator


def _channel_of(runtime: GraphRuntimeContext) -> AnswerDraftChannel | None:
    channel = runtime.services.answer_draft_channel
    if channel is None:
        return None
    if not isinstance(channel, AnswerDraftChannel):
        raise EvaluationError(
            "runtime.services.answer_draft_channel is not an "
            "AnswerDraftChannel; refusing to use it"
        )
    return channel


def locator_covers(
    requested: ContentLocator, observed: ContentLocator
) -> str:
    """Deterministic locator compatibility: ``full`` / ``partial`` / ``none``.

    Equal coordinates are full coverage. A part (section/article/page/chunk)
    observed against a whole-document request is partial coverage. Anything
    else — including a different section than requested — covers nothing.
    """
    if observed == requested:
        return "full"
    if isinstance(requested, DocumentLocator) and isinstance(
        observed, (SectionLocator, ArticleLocator, PageRangeLocator, ChunkRangeLocator)
    ):
        return "partial"
    return "none"


def _coverage_criterion(unit: TargetUnit) -> CoverageCriterion | None:
    for criterion in unit.completion_criteria:
        if isinstance(criterion, CoverageCriterion):
            return criterion
    return None


def _semantic_criteria(unit: TargetUnit) -> tuple[SemanticCriterion, ...]:
    return tuple(
        criterion
        for criterion in unit.completion_criteria
        if isinstance(criterion, SemanticCriterion)
    )


def _binding_for_target(
    plan: TaskPlan, bindings: DocumentBindingSet, target_id: str
) -> ScopedDocument | None:
    unit = next(
        (item for item in plan.target_units if item.target_id == target_id), None
    )
    if unit is None:
        return None
    return next(
        (binding for binding in bindings.bindings if binding.binding_id == unit.binding_id),
        None,
    )


def build_coverage(
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    results: tuple[AgentResult, ...],
    hydrated: tuple[HydratedEvidence, ...],
) -> Coverage:
    """Evaluator-owned coverage facts for every planned target.

    A locator counts only when a read observation AND an admitted coverage use
    confirm it AND the use's record revision still matches the pinned binding
    revision: an observed read whose use was denied (stale revision, expiry,
    tombstone, failed derived validation) — or whose revision drifted past
    hydration admission — completes nothing. Discovery and supporting purposes
    never complete coverage.
    """
    observations_by_target: dict[str, list[Any]] = {}
    for result in results:
        for observation in result.coverage_observations:
            observations_by_target.setdefault(observation.target_id, []).append(
                observation
            )
    admitted_locators: dict[str, set[ContentLocator]] = {}
    for item in hydrated:
        if (
            item.purpose != "coverage"
            or item.target_id is None
            or item.locator is None
        ):
            continue
        binding = _binding_for_target(plan, bindings, item.target_id)
        if binding is None:
            continue
        if item.document_revision != binding.document_revision:
            continue
        admitted_locators.setdefault(item.target_id, set()).add(item.locator)
    items: list[CoverageItem] = []
    for unit in plan.target_units:
        observations = observations_by_target.get(unit.target_id, ())
        observed: list[ContentLocator] = [
            locator
            for observation in observations
            for locator in observation.observed_locators
        ]
        agreed = {
            locator
            for observation in observations
            if observation.outcome == "read"
            for locator in observation.observed_locators
            if locator in admitted_locators.get(unit.target_id, set())
        }
        if any(locator_covers(unit.requested_locator, locator) == "full" for locator in agreed):
            status: CoverageStatus = "read_complete"
        elif any(
            locator_covers(unit.requested_locator, locator) == "partial" for locator in agreed
        ):
            status = "read_partial"
        elif any(observation.outcome == "unreadable" for observation in observations):
            status = "unreadable"
        elif any(observation.outcome == "truncated" for observation in observations):
            status = "truncated"
        else:
            status = "missing"
        items.append(
            CoverageItem(
                target_id=unit.target_id,
                observed_locators=tuple(observed),
                status=status,
            )
        )
    return Coverage(items=tuple(items))


def find_missing_requirements(
    plan: TaskPlan,
    coverage: Coverage,
    hydrated: tuple[HydratedEvidence, ...],
    semantic_verdicts: Mapping[str, bool],
) -> tuple[MissingRequirement, ...]:
    """Missing coverage plus unsatisfied semantic criteria (criterion presence).

    Coverage below the unit's minimum (default ``read_complete``) is a
    coverage gap. A semantic criterion is satisfied only by an explicit judge
    verdict for its ``criterion_id``; without one it stays missing — presence
    is deterministic even when judgment is model-assisted. ``hydrated``
    additionally lets task-level gaps name their targets: an
    evidence-supplying task with no admitted use flags each of its targets
    that coverage did not already flag, so the gap is never silent.
    """
    status_by_target = {item.target_id: item.status for item in coverage.items}
    missing: list[MissingRequirement] = []
    flagged: set[str] = set()
    for unit in plan.target_units:
        status = status_by_target.get(unit.target_id, "missing")
        criterion = _coverage_criterion(unit)
        minimum = criterion.minimum_status if criterion is not None else "read_complete"
        if status == "read_complete":
            pass
        elif status == "read_partial" and minimum == "read_partial":
            pass
        else:
            missing.append(
                MissingRequirement(
                    target_id=unit.target_id,
                    criterion_kind="coverage",
                    semantic_criterion_id=None,
                    description=(
                        f"target {unit.target_id} requires {minimum} coverage "
                        f"(observed: {status})"
                    ),
                )
            )
            flagged.add(unit.target_id)
        for semantic_criterion in _semantic_criteria(unit):
            if not semantic_verdicts.get(semantic_criterion.criterion_id, False):
                missing.append(
                    MissingRequirement(
                        target_id=unit.target_id,
                        criterion_kind="semantic",
                        semantic_criterion_id=semantic_criterion.criterion_id,
                        description=(
                            f"target {unit.target_id} semantic criterion "
                            f"{semantic_criterion.criterion_id} is unsatisfied: "
                            f"{semantic_criterion.description}"
                        ),
                    )
                )
    admitted_tasks = {
        item.task_id for item in hydrated if item.purpose != "discovery"
    }
    for task in plan.tasks:
        if task.capability not in _EVIDENCE_SUPPLYING_CAPABILITIES:
            continue
        if task.task_id in admitted_tasks:
            continue
        for target_id in getattr(task.input, "target_ids", None) or ():
            if target_id not in flagged:
                missing.append(
                    MissingRequirement(
                        target_id=target_id,
                        criterion_kind="coverage",
                        semantic_criterion_id=None,
                        description=(
                            f"task {task.task_id} produced no admitted "
                            f"evidence for target {target_id}"
                        ),
                    )
                )
                flagged.add(target_id)
    return tuple(missing)


def every_expecting_task_has_evidence(
    plan: TaskPlan,
    hydrated: tuple[HydratedEvidence, ...],
) -> bool:
    """Task-evidence gate: sufficiency needs evidence behind every supplier.

    Every task whose capability is expected to supply synthesis-eligible
    evidence must own at least one admitted non-discovery use. Targetless
    tasks (People/KG/memory lookups) cannot name a target in a
    ``MissingRequirement``, so this gate — not the target-level list — is
    what keeps a denied/errored/``not_found`` lookup from certifying
    ``sufficient``. Plans with no evidence-supplying tasks at all (zero
    tasks, discovery-only) never satisfy the gate.
    """
    expecting = [
        task
        for task in plan.tasks
        if task.capability in _EVIDENCE_SUPPLYING_CAPABILITIES
    ]
    if not expecting:
        return False
    admitted_tasks = {
        item.task_id for item in hydrated if item.purpose != "discovery"
    }
    return all(task.task_id in admitted_tasks for task in expecting)


def has_needs_input_result(results: tuple[AgentResult, ...]) -> bool:
    """True when any task result asks for user input."""
    return any(result.status == "needs_input" for result in results)


def has_blocking_conflict(contradictions: tuple[Contradiction, ...]) -> bool:
    """A genuine conflict blocks: at least two distinct sides clash.

    A contradiction naming a single use is reported on the evaluation but is
    not a blocking cross-source conflict, so a supported (single-sided)
    contradiction may still stand beside a ``sufficient`` verdict.
    """
    return any(len(set(item.evidence_use_ids)) >= 2 for item in contradictions)


async def analyze_contradictions_bounded(
    hydrated: tuple[HydratedEvidence, ...],
    semantic_judge: SemanticJudge | None = None,
) -> tuple[Contradiction, ...]:
    """Bounded contradiction interpretation over governed evidence only.

    Without a judge there are no interpreted conflicts (deterministic rules
    cannot read semantics); with one, its reported contradictions are returned
    verbatim for the deterministic precedence gate — the judge never decides
    the status itself. Every referenced use id must belong to the admitted
    hydration set; a contradiction citing outside evidence is rejected rather
    than certified.
    """
    if semantic_judge is None:
        return ()
    contradictions = semantic_judge.detect_contradictions(evidence=hydrated)
    if inspect.isawaitable(contradictions):
        contradictions = await contradictions
    if not isinstance(contradictions, tuple):
        contradictions = tuple(contradictions)
    admitted = {item.use_id for item in hydrated}
    for contradiction in contradictions:
        if not isinstance(contradiction, Contradiction):
            raise EvaluationError(
                "semantic judge returned a non-Contradiction conflict; "
                "refusing to certify"
            )
        unknown = [
            use_id
            for use_id in contradiction.evidence_use_ids
            if use_id not in admitted
        ]
        if unknown:
            raise EvaluationError(
                f"contradiction {contradiction.contradiction_id} cites "
                f"evidence uses outside the admitted set: {unknown}; "
                "refusing to certify"
            )
    return contradictions


async def _assess_semantic_criteria(
    plan: TaskPlan,
    hydrated: tuple[HydratedEvidence, ...],
    semantic_judge: SemanticJudge | None,
) -> dict[str, bool]:
    criteria = [
        criterion
        for unit in plan.target_units
        for criterion in _semantic_criteria(unit)
    ]
    if not criteria or semantic_judge is None:
        return {}
    verdicts: dict[str, bool] = {}
    for criterion in criteria:
        if criterion.criterion_id in verdicts:
            continue
        verdict = semantic_judge.assess_criterion(
            criterion=criterion, evidence=hydrated
        )
        if inspect.isawaitable(verdict):
            verdict = await verdict
        verdicts[criterion.criterion_id] = bool(verdict)
    return verdicts


async def evaluate_evidence(
    *,
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    results: tuple[AgentResult, ...],
    semantic: SemanticContext,
    runtime: GraphRuntimeContext,
    semantic_judge: SemanticJudge | None = None,
) -> EvidenceEvaluation:
    """Produce the validated sufficiency verdict for the checkpointed plan."""
    hydrator = _require_hydrator(runtime)
    hydrated = await hydrator.hydrate_for_evaluation(
        tuple(ref for result in results for ref in result.evidence_uses),
        runtime=runtime,
        plan=plan,
        bindings=bindings,
    )
    for item in hydrated:
        if not isinstance(item, HydratedEvidence):
            raise EvaluationError(
                "evidence_hydrator returned a non-HydratedEvidence projection; "
                "refusing to certify"
            )
    coverage = build_coverage(plan, bindings, results, hydrated)
    verdicts = await _assess_semantic_criteria(plan, hydrated, semantic_judge)
    missing = find_missing_requirements(plan, coverage, hydrated, verdicts)
    contradictions = await analyze_contradictions_bounded(hydrated, semantic_judge)
    if semantic.blocking_ambiguities or has_needs_input_result(results):
        status = "needs_input"
    elif has_blocking_conflict(contradictions):
        status = "contradictory"
    elif missing or not every_expecting_task_has_evidence(plan, hydrated):
        status = "insufficient"
    else:
        status = "sufficient"
    evaluation = EvidenceEvaluation(
        status=status,  # type: ignore[arg-type]
        coverage=coverage,
        missing=missing,
        contradictions=contradictions,
    )
    validate_evidence_evaluation(evaluation, plan)
    return evaluation


async def evaluate_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Checkpoint the evaluator's verdict for the authoritative plan."""
    context = _context_of(runtime)
    plan = require_checkpointed_plan(state)
    evaluation = await evaluate_evidence(
        plan=plan,
        bindings=state["bindings"],
        results=state["execution"].task_results,
        semantic=state["semantic"],
        runtime=context,
    )
    return execution_update(state, evidence_evaluation=evaluation)

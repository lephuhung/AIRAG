"""Evaluator authority: coverage, missing requirements, status (Phase 2, Task 4).

``evaluate_evidence`` is the shared FUNCTION that owns the sufficiency verdict
(amendment §7: there is deliberately NO ``EvidenceEvaluator`` service, and
``RuntimeServices`` excludes it). Deterministic rules own coverage,
revision/locator compatibility, ACL/expiry/source validity (via hydration
admission), criterion presence, and status precedence::

    hydrate (governed admission only)
    -> build_coverage (plan × bindings × observations × admitted uses)
    -> find_missing_requirements (coverage + semantic-criterion verdicts)
    -> analyze_contradictions_bounded (judge-assisted, deterministic precedence)
    -> needs_input > contradictory > insufficient > sufficient

A bounded model may assist ONLY semantic-criterion/contradiction
interpretation through the optional ``SemanticJudge`` seam, and only over
already-governed evidence: the judge receives hydrated content plus the
criterion — never authorization, bindings, revisions, plans, or tools — and it
cannot self-certify factual success (deterministic coverage and precedence
still gate ``sufficient``). ``evaluate_node`` wires no judge in Phase 2; the
seam exists for T7 production wiring.

The typed ``EvidenceHydrator`` Protocol is defined here: hydration for
evaluation plus hydration for synthesis (plus the overflow-derived persist the
synthesis budget requires). Implementations are fed the canonical
plan/bindings/budget stores at wiring time (T6/T7, mirroring the
``PinnedTargetResolver`` precedent), so the call shape stays
``(use_refs, runtime)`` and capabilities never see supervisor/graph state.
Denials are SKIPPED (never raised): the admitted set shrinks and the resulting
missing requirements name the gap, so nothing is silently dropped. A missing
``evidence_hydrator`` service fails closed.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable
from uuid import UUID

from langgraph.runtime import Runtime

from ..contracts.binding import DocumentBindingSet
from ..contracts.evaluation import (
    Contradiction,
    Coverage,
    CoverageItem,
    CoverageStatus,
    EvidenceEvaluation,
    MissingRequirement,
)
from ..contracts.evidence import EvidenceClassification, EvidencePurpose, EvidenceUseRef
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
from ..contracts.semantic import SemanticContext
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.validation import validate_evidence_evaluation
from .context import _context_of
from .execute import execution_update, require_checkpointed_plan

__all__ = [
    "EvaluationError",
    "HydratedEvidence",
    "EvidenceHydrator",
    "SemanticJudge",
    "locator_covers",
    "build_coverage",
    "find_missing_requirements",
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
    computed from the admitted use plus the canonical plan/binding stores, and
    ``locator``/``classification`` travel with the content so deterministic
    coverage and the synthesis budget need no second store lookup. ``target_id``
    is the admitted use's own target; discovery uses never hydrate.
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


@runtime_checkable
class EvidenceHydrator(Protocol):
    """Typed hydration boundary owned by the Evidence Store (wired by T6/T7).

    The implementation is fed the canonical plan/binding/budget stores at
    wiring time and enforces current ACL/expiry/tombstone/revision/derived
    validation on every call; decryption happens last inside the governor.
    ``hydrate_for_synthesis`` additionally enforces the synthesis budget and
    persists an overflow tail as validated derived evidence (never a silent
    truncation). Denied uses are skipped (audited by the implementation) and
    absent from the returned admitted set.
    """

    async def hydrate_for_evaluation(
        self,
        use_refs: tuple[EvidenceUseRef, ...],
        runtime: GraphRuntimeContext,
    ) -> tuple[HydratedEvidence, ...]:
        """Hydrate admitted uses for evaluation (content + locator + class)."""
        ...

    async def hydrate_for_synthesis(
        self,
        use_refs: tuple[EvidenceUseRef, ...],
        runtime: GraphRuntimeContext,
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
    ) -> HydratedEvidence:
        """Persist a validated DERIVED record + supporting use; return it hydrated."""
        ...


@runtime_checkable
class SemanticJudge(Protocol):
    """Bounded-model seam for semantic interpretation over governed evidence.

    The judge receives hydrated content and the criterion under judgment —
    never authorization, bindings, revisions, plans, or tools — and its output
    is advisory: deterministic coverage and status precedence still gate the
    verdict, so it cannot self-certify factual success.
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


def _require_hydrator(runtime: GraphRuntimeContext) -> EvidenceHydrator:
    hydrator = runtime.services.evidence_hydrator
    if hydrator is None:
        raise EvaluationError(
            "no evidence_hydrator wired on runtime.services; refusing to "
            "certify evidence without governed hydration"
        )
    return hydrator


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


def build_coverage(
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    results: tuple[AgentResult, ...],
    hydrated: tuple[HydratedEvidence, ...],
) -> Coverage:
    """Evaluator-owned coverage facts for every planned target.

    A locator counts only when BOTH a read observation and an admitted
    coverage use confirm it: an observed read whose use was denied (stale
    revision, expiry, tombstone, failed derived validation) completes nothing.
    Discovery and supporting purposes never complete coverage.
    """
    del bindings  # revision identity is enforced at hydration admission.
    observations_by_target: dict[str, list[Any]] = {}
    for result in results:
        for observation in result.coverage_observations:
            observations_by_target.setdefault(observation.target_id, []).append(
                observation
            )
    admitted_locators: dict[str, set[ContentLocator]] = {}
    for item in hydrated:
        if item.purpose == "coverage" and item.target_id is not None and item.locator is not None:
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
    is deterministic even when judgment is model-assisted.
    """
    del hydrated  # semantic presence needs verdicts, not content, here.
    status_by_target = {item.target_id: item.status for item in coverage.items}
    missing: list[MissingRequirement] = []
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
    return tuple(missing)


def has_needs_input_result(results: tuple[AgentResult, ...]) -> bool:
    """True when any task result asks for user input."""
    return any(result.status == "needs_input" for result in results)


def has_blocking_conflict(contradictions: tuple[Contradiction, ...]) -> bool:
    """Every reported contradiction blocks: conflicts never certify success."""
    return bool(contradictions)


async def analyze_contradictions_bounded(
    hydrated: tuple[HydratedEvidence, ...],
    semantic_judge: SemanticJudge | None = None,
) -> tuple[Contradiction, ...]:
    """Bounded contradiction interpretation over governed evidence only.

    Without a judge there are no interpreted conflicts (deterministic rules
    cannot read semantics); with one, its reported contradictions are returned
    verbatim for the deterministic precedence gate — the judge never decides
    the status itself.
    """
    if semantic_judge is None:
        return ()
    contradictions = semantic_judge.detect_contradictions(evidence=hydrated)
    if inspect.isawaitable(contradictions):
        contradictions = await contradictions
    if not isinstance(contradictions, tuple):
        contradictions = tuple(contradictions)
    for contradiction in contradictions:
        if not isinstance(contradiction, Contradiction):
            raise EvaluationError(
                "semantic judge returned a non-Contradiction conflict; "
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
        runtime,
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
    elif missing:
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

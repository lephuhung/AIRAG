"""Minimized/redacted model projection for the governed replanner (Task 11).

The authoritative replan envelope is the redacted ``ResearchPlanningInput``
built by ``build_model_replan_input`` (current plan with the governed scalar
redacted, one ``TaskExecutionSummary`` per attempted task, current validated
use refs, latest ``EvidenceEvaluation``). The model sees only
:class:`ReplannerModelInput` — evaluator gap identity (target IDs, criterion
kinds, short descriptions), contradiction IDs with truncated claims, outcome
statuses, the prior-use COUNT (never use IDs), catalog names, discovery
flags, and budget numbers.

Trusted identity (document/revision UUIDs, binding IDs), internal evidence
use UUIDs, governed people scalars, runtime services, and raw evidence never
cross this boundary. The server re-attaches trigger lineage (attempted task
IDs + prior evidence-use IDs) when it constructs ``ReplanTaskOrigin``.
"""
from __future__ import annotations

from ..contracts.base import RuntimeModel
from ..contracts.evaluation import (
    Contradiction,
    EvidenceEvaluation,
    MissingRequirement,
)
from ..contracts.execution import TaskExecutionSummary
from ..contracts.planning import ResearchPlanningInput

__all__ = [
    "ReplannerBudgetView",
    "ReplannerCapabilityRef",
    "ReplannerContradictionRef",
    "ReplannerDiscoveryView",
    "ReplannerGapRef",
    "ReplannerModelInput",
    "ReplannerOutcomeRef",
    "ReplannerTargetRef",
    "build_replanner_model_input",
]

#: Server-side truncation bounds for evaluator prose reaching the model.
_GAP_DESCRIPTION_LIMIT = 500
_CLAIM_LIMIT = 300


class ReplannerTargetRef(RuntimeModel):
    """One replannable target: plan target ID + locator kind only."""

    target_id: str
    locator_kind: str


class ReplannerGapRef(RuntimeModel):
    """One minimized evaluator gap (coverage or semantic)."""

    target_id: str
    criterion_kind: str
    description: str


class ReplannerContradictionRef(RuntimeModel):
    """One contradiction, ID + truncated claims (context only, never a trigger)."""

    contradiction_id: str
    claim_a: str
    claim_b: str


class ReplannerOutcomeRef(RuntimeModel):
    """One attempted-task outcome (status + typed error code, never a rerun)."""

    task_id: str
    status: str
    error_code: str | None = None


class ReplannerCapabilityRef(RuntimeModel):
    """One catalog entry the replanner may propose (already minimal)."""

    name: str
    domain: str
    operation_type: str
    supports_parallel: bool


class ReplannerDiscoveryView(RuntimeModel):
    """Discovery flags the replanner must respect (restrict, never grant)."""

    allow_reference_discovery: bool
    allow_supporting_discovery: bool
    max_discovered_documents: int


class ReplannerBudgetView(RuntimeModel):
    """Budget numbers the replanner must respect (scheduler still enforces)."""

    max_tasks_remaining: int
    max_replans_remaining: int
    max_parallel_branches: int


class ReplannerModelInput(RuntimeModel):
    """Runtime-only minimized evaluator-gap input (never checkpointed)."""

    query: str
    work_type: str
    domains: tuple[str, ...]
    targets: tuple[ReplannerTargetRef, ...]
    gaps: tuple[ReplannerGapRef, ...]
    contradictions: tuple[ReplannerContradictionRef, ...]
    outcomes: tuple[ReplannerOutcomeRef, ...]
    prior_use_count: int
    capability_catalog: tuple[ReplannerCapabilityRef, ...]
    discovery_policy: ReplannerDiscoveryView
    budget: ReplannerBudgetView


def _truncate(text: str, limit: int) -> str:
    cleaned = (text or "").strip()
    return cleaned[:limit]


def _project_gaps(
    evaluation: EvidenceEvaluation | None,
) -> tuple[tuple[ReplannerGapRef, ...], tuple[ReplannerContradictionRef, ...]]:
    """Project the evaluator verdict to minimized gap refs (frozen types in)."""
    if evaluation is None:
        return (), ()
    gaps: list[ReplannerGapRef] = []
    missing: tuple[MissingRequirement, ...] = evaluation.missing or ()
    for requirement in missing:
        gaps.append(
            ReplannerGapRef(
                target_id=requirement.target_id,
                criterion_kind=str(requirement.criterion_kind),
                description=_truncate(
                    requirement.description, _GAP_DESCRIPTION_LIMIT
                ),
            )
        )
    contradictions: list[ReplannerContradictionRef] = []
    conflicts: tuple[Contradiction, ...] = evaluation.contradictions or ()
    for conflict in conflicts:
        contradictions.append(
            ReplannerContradictionRef(
                contradiction_id=conflict.contradiction_id,
                claim_a=_truncate(conflict.claim_a, _CLAIM_LIMIT),
                claim_b=_truncate(conflict.claim_b, _CLAIM_LIMIT),
            )
        )
    return tuple(gaps), tuple(contradictions)


def build_replanner_model_input_from_envelope(
    planning_input: ResearchPlanningInput,
) -> ReplannerModelInput:
    """Project the redacted replan envelope to the minimized model input.

    Allowlist projection over the ALREADY-redacted envelope: target IDs +
    locator kinds (no binding IDs, no document/revision UUIDs), gap identity
    + truncated descriptions, contradiction IDs + truncated claims, outcome
    statuses + error codes, the prior-use count (never use IDs), catalog
    names/domains, policy flags, budget numbers.
    """
    gaps, contradictions = _project_gaps(planning_input.prior_evaluation)
    current = planning_input.current_plan
    targets: list[ReplannerTargetRef] = []
    if current is not None:
        for unit in current.target_units:
            targets.append(
                ReplannerTargetRef(
                    target_id=unit.target_id,
                    locator_kind=str(unit.requested_locator.kind),
                )
            )
    outcomes: list[ReplannerOutcomeRef] = []
    summaries: tuple[TaskExecutionSummary, ...] = planning_input.task_outcomes or ()
    for outcome in summaries:
        outcomes.append(
            ReplannerOutcomeRef(
                task_id=outcome.task_id,
                status=str(outcome.status),
                error_code=str(outcome.error_code)
                if outcome.error_code is not None
                else None,
            )
        )
    return ReplannerModelInput(
        query=planning_input.semantic.contextualized_query,
        work_type=str(planning_input.query_analysis.work_type),
        domains=tuple(
            str(domain) for domain in planning_input.query_analysis.domains
        ),
        targets=tuple(targets),
        gaps=gaps,
        contradictions=contradictions,
        outcomes=tuple(outcomes),
        prior_use_count=len(planning_input.prior_evidence_uses or ()),
        capability_catalog=tuple(
            ReplannerCapabilityRef(
                name=entry.name,
                domain=str(entry.domain),
                operation_type=str(entry.operation_type),
                supports_parallel=bool(entry.supports_parallel),
            )
            for entry in planning_input.capability_catalog
        ),
        discovery_policy=ReplannerDiscoveryView(
            allow_reference_discovery=bool(
                planning_input.discovery_policy.allow_reference_discovery
            ),
            allow_supporting_discovery=bool(
                planning_input.discovery_policy.allow_supporting_discovery
            ),
            max_discovered_documents=int(
                planning_input.discovery_policy.max_discovered_documents
            ),
        ),
        budget=ReplannerBudgetView(
            max_tasks_remaining=int(planning_input.budget.max_tasks_remaining),
            max_replans_remaining=int(planning_input.budget.max_replans_remaining),
            max_parallel_branches=int(planning_input.budget.max_parallel_branches),
        ),
    )


def build_replanner_model_input(state: object, runtime: object) -> ReplannerModelInput:
    """Model-facing replan projection from checkpointed state (never stored).

    Routes through the governed redacted envelope
    (``build_model_replan_input``) so the scalar redaction and the validated
    observation boundary stay the single model-facing path.
    """
    from ..complex_research_graph import build_model_replan_input

    return build_replanner_model_input_from_envelope(
        build_model_replan_input(state, runtime)  # type: ignore[arg-type]
    )

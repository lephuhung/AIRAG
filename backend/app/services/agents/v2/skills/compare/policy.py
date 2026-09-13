"""Compare skill policy: bounded two-target comparison (Phase 3, Task 3).

Framework-neutral source of truth: given an ephemeral ``ResearchPlanningInput``
the policy proposes the initial ``TaskPlan`` for a ``compare`` request — one
whole-document ``document.read`` per side, no dependencies (parallel safe
reads), initial origin. It never executes a capability, never checkpoints, and
never reaches past validation: the returned plan still passes through
``validate_task_plan`` in the subgraph ``validate_checkpoint`` node, the
supervisor saver, and the shared scheduler.

Pilot scope (initial-plan-only):

- exactly two bound documents, one with role ``target`` and one with role
  ``reference`` — any other arity or role pair fails closed;
- ``document.read`` must be present in the request-scoped capability catalog;
- discovery/replan are rejected here (T5 adds them): only ``compare`` is
  supported, every other work type fails closed so the caller can return the
  typed ``COMPLEX_RESEARCH_UNAVAILABLE`` boundary instead of a fabricated plan.
"""
from __future__ import annotations

from ...contracts.binding import ScopedDocument
from ...contracts.capability import DocumentReadInput
from ...contracts.locators import DocumentLocator
from ...contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    ResearchPlanningInput,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from ...contracts.validation import ContractValidationError, validate_task_plan

__all__ = [
    "COMPARE_WORK_TYPE",
    "supports_work_type",
    "build_compare_plan",
]

#: The only work type this pilot plans. Everything else fails closed.
COMPARE_WORK_TYPE = "compare"

#: Exact two-sided comparison: one target side plus one reference side.
MAX_COMPARISON_SIDES = 2

_TARGET_SIDE_ID = "t1"
_REFERENCE_SIDE_ID = "t2"


def supports_work_type(work_type: str) -> bool:
    """True only for the ``compare`` pilot; ``evaluate`` and the rest are out."""
    return work_type == COMPARE_WORK_TYPE


def _compare_sides(bindings: object) -> tuple[ScopedDocument, ScopedDocument]:
    """Return the (target, reference) bound documents in deterministic order.

    Fails closed unless the run binds exactly two documents with one
    ``target`` role and one ``reference`` role: the planner cannot
    autonomously add targets or reinterpret roles.
    """
    ordered = sorted(bindings.bindings, key=lambda binding: binding.binding_id)  # type: ignore[union-attr]
    if len(ordered) != MAX_COMPARISON_SIDES:
        raise ContractValidationError(
            f"compare requires exactly {MAX_COMPARISON_SIDES} bound documents, "
            f"got {len(ordered)}; refusing to fabricate comparison sides"
        )
    by_role = {binding.role: binding for binding in ordered}
    target = by_role.get("target")
    reference = by_role.get("reference")
    if target is None or reference is None or len(by_role) != MAX_COMPARISON_SIDES:
        raise ContractValidationError(
            "compare requires one 'target' side and one 'reference' side "
            f"(got roles {sorted(by_role)}); refusing to reinterpret roles"
        )
    return target, reference


def _require_read_capability(planning_input: ResearchPlanningInput) -> None:
    """Fail closed when the current runtime catalog cannot serve the reads."""
    names = {entry.name for entry in planning_input.capability_catalog}
    if "document.read" not in names:
        raise ContractValidationError(
            "compare requires 'document.read' in the request-scoped capability "
            "catalog; refusing to plan an undispatchable read"
        )


def build_compare_plan(planning_input: ResearchPlanningInput) -> TaskPlan:
    """Propose the initial two-target bounded comparison plan.

    Raises :class:`ContractValidationError` for any out-of-pilot input
    (unsupported work type, wrong arity/roles, missing read capability, or a
    plan that fails frozen validation) — the caller turns that into the typed
    unavailable boundary, never a partial plan.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        raise ContractValidationError(
            f"compare skill cannot plan work type "
            f"{planning_input.query_analysis.work_type!r}; out of pilot scope"
        )
    _require_read_capability(planning_input)
    target, reference = _compare_sides(planning_input.bindings)
    units = (
        TargetUnit(
            target_id=_TARGET_SIDE_ID,
            binding_id=target.binding_id,
            requested_locator=DocumentLocator(kind="document"),
            completion_criteria=(CoverageCriterion(kind="coverage"),),
        ),
        TargetUnit(
            target_id=_REFERENCE_SIDE_ID,
            binding_id=reference.binding_id,
            requested_locator=DocumentLocator(kind="document"),
            completion_criteria=(CoverageCriterion(kind="coverage"),),
        ),
    )
    tasks = (
        TaskSpec(
            task_id="T1",
            capability="document.read",
            task_objective=(
                "Read the target side for comparison "
                f"(document {target.document_id})"
            ),
            input=DocumentReadInput(kind="document.read", target_ids=("t1",)),
            depends_on=(),
            origin=InitialTaskOrigin(kind="initial"),
        ),
        TaskSpec(
            task_id="T2",
            capability="document.read",
            task_objective=(
                "Read the reference side for comparison "
                f"(document {reference.document_id})"
            ),
            input=DocumentReadInput(kind="document.read", target_ids=("t2",)),
            depends_on=(),
            origin=InitialTaskOrigin(kind="initial"),
        ),
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id=f"compare-{target.binding_id}-{reference.binding_id}",
        goal=planning_input.semantic.contextualized_query,
        target_units=units,
        tasks=tasks,
    )
    validate_task_plan(plan, planning_input.bindings)
    return plan

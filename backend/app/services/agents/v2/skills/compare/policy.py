"""Compare skill policy: bounded two-target comparison (Phase 3, Task 3).

Framework-neutral source of truth: given an ephemeral ``ResearchPlanningInput``
the policy proposes the initial ``TaskPlan`` for a ``compare`` request — one
bounded read per side, no dependencies (parallel safe reads), initial origin.
It never executes a capability, never checkpoints, and never reaches past
validation: the returned plan still passes through ``validate_task_plan`` in
the subgraph ``validate_checkpoint`` node, the supervisor saver, and the
shared scheduler.

Pilot scope (initial-plan-only):

- exactly two bound documents, one with role ``target`` and one with role
  ``reference`` — any other arity or role pair fails closed;
- each side's ``requested_locator`` derives from the query semantics: a
  ``SemanticContext.section_refs`` entry naming a chapter/section for that
  side (joined via the canonical ``binding_id_for_ref`` convention, whose
  single owner is ``adapters/document.py``) becomes the exact
  ``SectionLocator`` coordinate (and selects ``section.read``); only a side
  with no finer coordinate falls back to a whole-document ``DocumentLocator``
  (``document.read``);
- every used read capability must be present in the request-scoped capability
  catalog;
- discovery/replan are rejected here (T5 adds them): only ``compare`` is
  supported, every other work type fails closed so the caller can return the
  typed ``COMPLEX_RESEARCH_UNAVAILABLE`` boundary instead of a fabricated plan.
"""
from __future__ import annotations

from ...adapters.document import binding_id_for_ref
from ...contracts.binding import ScopedDocument
from ...contracts.capability import DocumentReadInput, SectionReadInput
from ...contracts.locators import ContentLocator, DocumentLocator, SectionLocator
from ...contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    ResearchPlanningInput,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from ...contracts.semantic import SemanticContext
from ...contracts.validation import ContractValidationError, validate_task_plan

__all__ = [
    "COMPARE_WORK_TYPE",
    "MAX_COMPARISON_SIDES",
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


def _locator_for(binding: ScopedDocument, semantic: SemanticContext) -> ContentLocator:
    """Exact requested coordinate for one side: section when named, else whole.

    A section reference names its side through the canonical binding-ID
    convention (``binding_id_for_ref``); the first match in ``ref_id`` order
    wins deterministically. A side with no named section keeps the
    whole-document coordinate — the only permitted fallback.
    """
    matches = sorted(
        (
            reference
            for reference in semantic.section_refs
            if reference.structure_node_id is not None
            and binding_id_for_ref(reference.ref_id) == binding.binding_id
        ),
        key=lambda reference: reference.ref_id,
    )
    if matches:
        structure_node_id = matches[0].structure_node_id
        assert structure_node_id is not None
        return SectionLocator(kind="section", structure_node_id=structure_node_id)
    return DocumentLocator(kind="document")


def _read_task(
    task_id: str,
    target_id: str,
    binding: ScopedDocument,
    locator: ContentLocator,
) -> TaskSpec:
    """One bounded side read: section reads for section coordinates."""
    if isinstance(locator, SectionLocator):
        return TaskSpec(
            task_id=task_id,
            capability="section.read",
            task_objective=(
                f"Read section {locator.structure_node_id} for comparison "
                f"(document {binding.document_id})"
            ),
            input=SectionReadInput(kind="section.read", target_ids=(target_id,)),
            depends_on=(),
            origin=InitialTaskOrigin(kind="initial"),
        )
    return TaskSpec(
        task_id=task_id,
        capability="document.read",
        task_objective=(
            "Read the comparison side "
            f"(document {binding.document_id})"
        ),
        input=DocumentReadInput(kind="document.read", target_ids=(target_id,)),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def _require_read_capabilities(
    planning_input: ResearchPlanningInput, tasks: tuple[TaskSpec, ...]
) -> None:
    """Fail closed when the runtime catalog cannot serve the planned reads."""
    names = {entry.name for entry in planning_input.capability_catalog}
    for task in tasks:
        if task.capability not in names:
            raise ContractValidationError(
                f"compare requires {task.capability!r} in the request-scoped "
                "capability catalog; refusing to plan an undispatchable read"
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
            "compare skill cannot plan work type "
            f"{planning_input.query_analysis.work_type!r}; out of pilot scope"
        )
    target, reference = _compare_sides(planning_input.bindings)
    semantic = planning_input.semantic
    target_locator = _locator_for(target, semantic)
    reference_locator = _locator_for(reference, semantic)
    units = (
        TargetUnit(
            target_id=_TARGET_SIDE_ID,
            binding_id=target.binding_id,
            requested_locator=target_locator,
            completion_criteria=(CoverageCriterion(kind="coverage"),),
        ),
        TargetUnit(
            target_id=_REFERENCE_SIDE_ID,
            binding_id=reference.binding_id,
            requested_locator=reference_locator,
            completion_criteria=(CoverageCriterion(kind="coverage"),),
        ),
    )
    tasks = (
        _read_task("T1", _TARGET_SIDE_ID, target, target_locator),
        _read_task("T2", _REFERENCE_SIDE_ID, reference, reference_locator),
    )
    _require_read_capabilities(planning_input, tasks)
    plan = TaskPlan(
        contract_version="2.0",
        plan_id=f"compare-{target.binding_id}-{reference.binding_id}",
        goal=planning_input.semantic.contextualized_query,
        target_units=units,
        tasks=tasks,
    )
    validate_task_plan(plan, planning_input.bindings)
    return plan

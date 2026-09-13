"""Summarize skill policy: bounded single-target summary (Phase 3, Task 5).

Framework-neutral source of truth, mirroring ``skills/compare/policy.py``: given
an ephemeral ``ResearchPlanningInput`` the policy proposes the initial
``TaskPlan`` for a ``summarize`` request — one bounded read of one bound
document, no dependencies, initial origin. It never executes a capability,
never checkpoints, and never reaches past validation: the returned plan still
passes through ``validate_task_plan`` in the subgraph ``validate_checkpoint``
node, the supervisor saver, and the shared scheduler.

Routing (amendment §6): the bounded case stays on this single read (fast when
the router chooses it); the large/iterative case uses the complex planner with
a deterministic map/reduce workflow proposed as governed work. The evaluator
owns coverage; synthesis/grounding stay outside agent authority. There is NO
``summary_agent``.

Pilot scope (initial-plan-only):

- exactly one bound document with role ``target`` — any other arity or role
  fails closed;
- the ``requested_locator`` derives from the query semantics: a
  ``SemanticContext.section_refs`` entry for this side (joined via the
  canonical ``binding_id_for_ref`` convention, whose single owner is
  ``adapters/document.py``) becomes the exact ``SectionLocator`` coordinate
  (and selects ``section.read``); with no finer coordinate the plan falls back
  to a whole-document ``DocumentLocator`` (``document.read``);
- the used read capability must be present in the request-scoped capability
  catalog;
- discovery/replan are rejected here (the replan loop owns appends): only
  ``summarize`` is supported, every other work type fails closed so the caller
  can return the typed ``COMPLEX_RESEARCH_UNAVAILABLE`` boundary instead of a
  fabricated plan.
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
    "SUMMARIZE_WORK_TYPE",
    "MAX_SUMMARY_TARGETS",
    "build_summarize_plan",
    "supports_work_type",
]

#: The only work type this pilot plans. Everything else fails closed.
SUMMARIZE_WORK_TYPE = "summarize"

#: Exact single-target bounded summary: one target side, one bounded read.
MAX_SUMMARY_TARGETS = 1

_SUMMARY_TARGET_ID = "t1"


def supports_work_type(work_type: str) -> bool:
    """True only for the ``summarize`` pilot; ``compare``/``evaluate`` are out."""
    return work_type == SUMMARIZE_WORK_TYPE


def _summary_target(bindings: object) -> ScopedDocument:
    """Return the single bound ``target`` document in deterministic order.

    Fails closed unless the run binds exactly one document with role
    ``target``: the planner cannot autonomously add targets or reinterpret
    roles.
    """
    ordered = sorted(bindings.bindings, key=lambda binding: binding.binding_id)  # type: ignore[union-attr]
    if len(ordered) != MAX_SUMMARY_TARGETS:
        raise ContractValidationError(
            f"summarize requires exactly {MAX_SUMMARY_TARGETS} bound document, "
            f"got {len(ordered)}; refusing to fabricate summary targets"
        )
    target = ordered[0]
    if target.role != "target":
        raise ContractValidationError(
            f"summarize requires a 'target' document (got role {target.role!r}); "
            "refusing to reinterpret roles"
        )
    return target


def _locator_for(binding: ScopedDocument, semantic: SemanticContext) -> ContentLocator:
    """Exact requested coordinate: section when named, else whole document.

    A section reference names its side through the canonical binding-ID
    convention (``binding_id_for_ref``); the first match in ``ref_id`` order
    wins deterministically.
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


def _read_task(target_id: str, binding: ScopedDocument, locator: ContentLocator) -> TaskSpec:
    """One bounded summary read: section reads for section coordinates."""
    if isinstance(locator, SectionLocator):
        return TaskSpec(
            task_id="T1",
            capability="section.read",
            task_objective=(
                f"Read section {locator.structure_node_id} for summary "
                f"(document {binding.document_id})"
            ),
            input=SectionReadInput(kind="section.read", target_ids=(target_id,)),
            depends_on=(),
            origin=InitialTaskOrigin(kind="initial"),
        )
    return TaskSpec(
        task_id="T1",
        capability="document.read",
        task_objective=(
            "Read the summary target "
            f"(document {binding.document_id})"
        ),
        input=DocumentReadInput(kind="document.read", target_ids=(target_id,)),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def _require_read_capability(
    planning_input: ResearchPlanningInput, task: TaskSpec
) -> None:
    """Fail closed when the runtime catalog cannot serve the planned read."""
    names = {entry.name for entry in planning_input.capability_catalog}
    if task.capability not in names:
        raise ContractValidationError(
            f"summarize requires {task.capability!r} in the request-scoped "
            "capability catalog; refusing to plan an undispatchable read"
        )


def build_summarize_plan(planning_input: ResearchPlanningInput) -> TaskPlan:
    """Propose the initial one-target bounded summary plan.

    Raises :class:`ContractValidationError` for any out-of-pilot input
    (unsupported work type, wrong arity/role, missing read capability, or a
    plan that fails frozen validation) — the caller turns that into the typed
    unavailable boundary, never a partial plan.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        raise ContractValidationError(
            "summarize skill cannot plan work type "
            f"{planning_input.query_analysis.work_type!r}; out of pilot scope"
        )
    target = _summary_target(planning_input.bindings)
    locator = _locator_for(target, planning_input.semantic)
    unit = TargetUnit(
        target_id=_SUMMARY_TARGET_ID,
        binding_id=target.binding_id,
        requested_locator=locator,
        completion_criteria=(CoverageCriterion(kind="coverage"),),
    )
    task = _read_task(_SUMMARY_TARGET_ID, target, locator)
    _require_read_capability(planning_input, task)
    plan = TaskPlan(
        contract_version="2.0",
        plan_id=f"summarize-{target.binding_id}",
        goal=planning_input.semantic.contextualized_query,
        target_units=(unit,),
        tasks=(task,),
    )
    validate_task_plan(plan, planning_input.bindings)
    return plan

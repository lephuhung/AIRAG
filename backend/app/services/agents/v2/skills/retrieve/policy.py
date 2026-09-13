"""Retrieve skill policy: deterministic factual retrieval (P0 Task 4).

Framework-neutral source of truth, mirroring ``skills/compare/policy.py``:
given an ephemeral ``ResearchPlanningInput`` the policy proposes the initial
``TaskPlan`` for a ``retrieve`` request — exactly one ``document.retrieve``
task, no dependencies, initial origin. It never executes a capability, never
checkpoints, and never reaches past validation: the returned plan still passes
through ``validate_task_plan`` in the subgraph ``validate_checkpoint`` node,
the supervisor saver, and the shared scheduler.

Pilot scope (initial-plan-only):

- scoped request (one or more ``target``-role bindings): one ``TargetUnit``
  per distinct resolved document (whole-document ``DocumentLocator``) with a
  ``CoverageCriterion(minimum_status="read_partial")``, and a single task
  referencing every unit;
- unscoped request (no ``target``-role bindings): no target units and a single
  targetless task over the authenticated workspace scope;
- the ``document.retrieve`` capability must be present in the request-scoped
  capability catalog, otherwise the policy fails closed so the caller returns
  the typed ``COMPLEX_RESEARCH_UNAVAILABLE`` boundary and dispatches nothing.

Deduplication (Task 3 M3): a query-internal reference and an ``api_explicit``
resource may resolve to the same document, yielding two ``target`` bindings
for one resolved document identity. Units are deduplicated by resolved
``document_id`` (first in ``binding_id`` order wins deterministically);
distinct documents are never merged, so the hard scope is preserved.
"""
from __future__ import annotations

from uuid import UUID

from ...contracts.binding import ScopedDocument
from ...contracts.capability import DocumentRetrieveInput
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
    "RETRIEVE_TOP_K",
    "RETRIEVE_WORK_TYPE",
    "build_retrieve_plan",
    "supports_work_type",
]

#: The only work type this pilot plans. Everything else fails closed.
RETRIEVE_WORK_TYPE = "retrieve"

#: Fixed page size for the single deterministic retrieval task.
RETRIEVE_TOP_K = 8

#: The capability this policy may plan. Absence fails closed (unavailable).
RETRIEVE_CAPABILITY = "document.retrieve"


def supports_work_type(work_type: str) -> bool:
    """True only for the ``retrieve`` pilot; ``evaluate`` and the rest are out."""
    return work_type == RETRIEVE_WORK_TYPE


def _explicit_target_bindings(bindings: object) -> tuple[ScopedDocument, ...]:
    """Distinct ``target``-role bindings in deterministic order.

    Only explicit document targets are planned (references, supporting, and
    discovered roles are never retrieval targets). Bindings that resolve to
    the same document identity collapse to the first in ``binding_id`` order;
    distinct documents are never merged, so the hard scope is preserved.
    """
    ordered = sorted(
        bindings.bindings,  # type: ignore[union-attr]
        key=lambda binding: binding.binding_id,
    )
    seen: set[UUID] = set()
    distinct: list[ScopedDocument] = []
    for binding in ordered:
        if binding.role != "target":
            continue
        if binding.document_id in seen:
            continue
        seen.add(binding.document_id)
        distinct.append(binding)
    return tuple(distinct)


def _require_retrieve_capability(planning_input: ResearchPlanningInput) -> None:
    """Fail closed when the runtime catalog cannot serve the retrieval."""
    names = {entry.name for entry in planning_input.capability_catalog}
    if RETRIEVE_CAPABILITY not in names:
        raise ContractValidationError(
            f"retrieve requires {RETRIEVE_CAPABILITY!r} in the request-scoped "
            "capability catalog; refusing to plan an undispatchable retrieval"
        )


def build_retrieve_plan(planning_input: ResearchPlanningInput) -> TaskPlan:
    """Propose the initial deterministic retrieval plan.

    Raises :class:`ContractValidationError` for any out-of-pilot input
    (unsupported work type, missing retrieve capability, or a plan that fails
    frozen validation) — the caller turns that into the typed unavailable
    boundary, never a partial plan and never a dispatch.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        raise ContractValidationError(
            "retrieve skill cannot plan work type "
            f"{planning_input.query_analysis.work_type!r}; out of pilot scope"
        )
    _require_retrieve_capability(planning_input)
    query = planning_input.semantic.contextualized_query
    targets = _explicit_target_bindings(planning_input.bindings)
    units = tuple(
        TargetUnit(
            target_id=f"t{index + 1}",
            binding_id=binding.binding_id,
            requested_locator=DocumentLocator(kind="document"),
            completion_criteria=(
                CoverageCriterion(
                    kind="coverage",
                    minimum_status="read_partial",
                    allow_partial_reason=(
                        "retrieval chunks establish partial document coverage; "
                        "a full-document read is not required for grounded citations"
                    ),
                ),
            ),
        )
        for index, binding in enumerate(targets)
    )
    task = TaskSpec(
        task_id="T1",
        capability=RETRIEVE_CAPABILITY,
        task_objective=f"Retrieve revision-pinned evidence for: {query}",
        input=DocumentRetrieveInput(
            kind="document.retrieve",
            query=query,
            target_ids=tuple(unit.target_id for unit in units),
            top_k=RETRIEVE_TOP_K,
        ),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )
    if targets:
        plan_id = "retrieve-" + "-".join(binding.binding_id for binding in targets)
    else:
        plan_id = "retrieve-unscoped"
    plan = TaskPlan(
        contract_version="2.0",
        plan_id=plan_id,
        goal=query,
        target_units=units,
        tasks=(task,),
    )
    validate_task_plan(plan, planning_input.bindings)
    return plan

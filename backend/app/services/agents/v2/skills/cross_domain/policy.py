"""Cross-domain skill policy: governed cross-family plans (Task 12).

Framework-neutral source of truth: given an ephemeral ``ResearchPlanningInput``
the policy proposes the initial ``TaskPlan`` for a ``cross_domain`` request
through exactly one of two governed branches — never both, never invented:

- named person: the People→Document pilot's governed first step, a single
  targetless ``people.lookup`` (T1); the deterministic materializer appends
  the governed dependent on success and the recovery replan owns
  ``not_found``/``TIMEOUT``. Fails closed without the capability in the
  current catalog — the planner never fabricates a person to look up;
- no person but bound documents (the generic Task-12 branch): one bounded
  read per distinct settled-role binding, no dependencies, initial origin —
  the same validated shape as the multi-goal skill, built on the shared
  ``bounded_reads`` mechanics.

It never executes a capability, never checkpoints, and never reaches past
validation: the returned plan still passes through ``validate_task_plan`` in
the subgraph ``validate_checkpoint`` node, the supervisor saver, and the
shared scheduler. A ``cross_domain`` input with neither a named person nor a
bound document fails closed so the governed model path (or the typed
unavailable boundary when unwired) still owns it.
"""
from __future__ import annotations

from ...contracts.capability import PeopleLookupInput
from ...contracts.planning import (
    InitialTaskOrigin,
    ResearchPlanningInput,
    TaskPlan,
    TaskSpec,
)
from ...contracts.validation import ContractValidationError, validate_task_plan
from ..bounded_reads import (
    coverage_unit,
    locator_for,
    plannable_bindings,
    read_task,
    require_read_capabilities,
)

__all__ = [
    "CROSS_DOMAIN_WORK_TYPE",
    "MAX_CROSS_DOMAIN_TARGETS",
    "build_cross_domain_plan",
    "covers_input",
    "supports_work_type",
]

#: The only work type this pilot plans. Everything else fails closed.
CROSS_DOMAIN_WORK_TYPE = "cross_domain"

#: Hard ceiling on the generic bound-read branch: a single initial proposal
#: stays a bounded fan-out of reads, never an open-ended crawl.
MAX_CROSS_DOMAIN_TARGETS = 8

#: Families the generic bound-read branch can serve deterministically.
#: Anything else (people without a named person ref, knowledge_graph,
#: memory, write) needs family-specific work the read fan-out cannot plan,
#: so those inputs stay uncovered for the governed model path.
_DOCUMENT_FAMILIES = frozenset({"document", "section"})


def _generic_branch_eligible(planning_input: ResearchPlanningInput) -> bool:
    """True when the bound-read branch can serve every declared family (I1).

    The frozen router assigns ``cross_domain`` only when ≥2 dependency
    families exist, and the read fan-out serves document/section families
    alone. Claiming a wider input (e.g. document + knowledge_graph) would
    silently drop the second family while the run still reports
    ``sufficient`` — so those inputs stay uncovered for the governed model
    path, which can plan the missing family. The named-person branch is
    unaffected: the pilot's first lookup is the governed answer there."""
    domains = set(planning_input.query_analysis.domains)
    if not domains <= _DOCUMENT_FAMILIES:
        return False
    count = len(plannable_bindings(planning_input.bindings))
    return 1 <= count <= MAX_CROSS_DOMAIN_TARGETS


def supports_work_type(work_type: str) -> bool:
    """True only for the ``cross_domain`` pilot; ``compare`` and the rest are out."""
    return work_type == CROSS_DOMAIN_WORK_TYPE


def covers_input(planning_input: ResearchPlanningInput) -> bool:
    """True when the deterministic skill owns this input (Task 12 planner seam).

    Ownership is work-type plus intake: a named person (the governed first
    lookup) or, for the generic bound-read branch, settled-role bindings
    whose declared families are all servable (document/section only — I1).
    A covered input whose build still refuses (missing catalog capability,
    over task budget) is a deliberate final refusal — the model path must
    not second-guess it. Fully open or multi-family inputs (no person, no
    bound documents, or a second family the reads cannot serve) stay
    uncovered so the governed model path still owns them.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        return False
    if bool(planning_input.semantic.person_refs):
        return True
    return _generic_branch_eligible(planning_input)


def _first_person(planning_input: ResearchPlanningInput):  # type: ignore[no-untyped-def]
    """First named person in deterministic ``ref_id`` order."""
    return min(
        planning_input.semantic.person_refs, key=lambda reference: reference.ref_id
    )


def _people_first_task(planning_input: ResearchPlanningInput) -> TaskSpec:
    """The pilot's governed first lookup for the first named person."""
    first = _first_person(planning_input)
    return TaskSpec(
        task_id="T1",
        capability="people.lookup",
        task_objective=f"Resolve person {first.label} ({first.ref_id})",
        input=PeopleLookupInput(
            kind="people.lookup",
            query=planning_input.semantic.contextualized_query,
        ),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def build_cross_domain_plan(planning_input: ResearchPlanningInput) -> TaskPlan:
    """Propose the initial governed cross-domain plan.

    Raises :class:`ContractValidationError` for any out-of-pilot input
    (unsupported work type, neither a named person nor a bound document,
    more bound documents than the hard ceiling or the remaining task budget
    allows, missing capability, or a plan that fails frozen validation) —
    the caller turns that into the governed model path (when a planner is
    wired and the input is open) or the typed unavailable boundary, never a
    partial plan and never a dispatch.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        raise ContractValidationError(
            "cross_domain skill cannot plan work type "
            f"{planning_input.query_analysis.work_type!r}; out of pilot scope"
        )
    if bool(planning_input.semantic.person_refs):
        catalog = {entry.name for entry in planning_input.capability_catalog}
        if "people.lookup" not in catalog:
            raise ContractValidationError(
                "cross_domain pilot needs 'people.lookup' in the request-scoped "
                "capability catalog; refusing to plan an undispatchable lookup"
            )
        first = _first_person(planning_input)
        plan = TaskPlan(
            contract_version="2.0",
            plan_id=f"people-first-{first.ref_id}",
            goal=planning_input.semantic.contextualized_query,
            target_units=(),
            tasks=(_people_first_task(planning_input),),
        )
        validate_task_plan(plan, planning_input.bindings)
        return plan
    sides = plannable_bindings(planning_input.bindings)
    if not sides:
        raise ContractValidationError(
            "cross_domain needs a named person or at least one bound document, "
            "got neither; refusing to fabricate cross-domain targets"
        )
    if not _generic_branch_eligible(planning_input):
        raise ContractValidationError(
            "cross_domain generic branch serves document and section families "
            f"only, got domains {sorted(planning_input.query_analysis.domains)}; "
            "refusing a one-sided plan that would drop a dependency family"
        )
    if len(sides) > MAX_CROSS_DOMAIN_TARGETS:
        raise ContractValidationError(
            f"cross_domain plans at most {MAX_CROSS_DOMAIN_TARGETS} bounded "
            f"reads, got {len(sides)} bound documents; refusing an open-ended "
            "fan-out"
        )
    if len(sides) > planning_input.budget.max_tasks_remaining:
        raise ContractValidationError(
            f"cross_domain needs {len(sides)} read task(s) but the task budget "
            f"allows {planning_input.budget.max_tasks_remaining}; refusing a "
            "plan that cannot fit the remaining budget"
        )
    semantic = planning_input.semantic
    locators = [locator_for(side, semantic) for side in sides]
    units = tuple(
        coverage_unit(f"t{index + 1}", side, locator)
        for index, (side, locator) in enumerate(zip(sides, locators))
    )
    tasks = tuple(
        read_task(
            f"T{index + 1}",
            f"t{index + 1}",
            side,
            locator,
            objective_noun="cross-domain",
        )
        for index, (side, locator) in enumerate(zip(sides, locators))
    )
    require_read_capabilities(
        {entry.name for entry in planning_input.capability_catalog},
        tasks,
        owner="cross_domain",
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="cross-domain-" + "-".join(side.binding_id for side in sides),
        goal=planning_input.semantic.contextualized_query,
        target_units=units,
        tasks=tasks,
    )
    validate_task_plan(plan, planning_input.bindings)
    return plan

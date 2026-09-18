"""Governed discovery plan expansion (discovery spec §13).

``expand_discovery_plan`` is the single authoritative append constructor for
the bootstrap transition: it appends selected target units and factual tasks
into the same plan lineage that already carries the checkpointed probe tasks,
runs the frozen :func:`validate_plan_expansion` invariant set (spec §13.1), and
returns only a validated plan. It is pure — no lease, persistence, database,
registry, scheduler, or runtime calls; the Phase-4 ``research_expand_node``
owns leasing at the graph boundary through the existing ``_lease_pinned_state``
session before the returned plan may be checkpointed.

This is deliberately NOT ``validate_replan``: the existing rule that replans
cannot change target units stays unchanged, and only this constructor may
append ``TargetUnit``/``TaskSpec`` tuples for a discovery expansion (the AST
ownership guard enforces it structurally).
"""
from __future__ import annotations

from ..contracts.binding import DocumentBindingSet
from ..contracts.execution import AgentResult
from ..contracts.planning import (
    DiscoveryExpansionTaskOrigin,
    DiscoveryProbeTaskOrigin,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from ..contracts.validation import (
    _input_target_ids,
    _selected_refs,
    validate_agent_result,
    validate_task_plan,
)
from ..contracts.validation_support import _fail, _require_unique
from .contracts import DiscoveryCheckpoint, ResearchTargetSelection
from .validation import (
    validate_discovery_checkpoint,
    validate_research_target_selection,
)

__all__ = [
    "expand_discovery_plan",
    "factual_tasks",
    "validate_plan_expansion",
]


def factual_tasks(
    plan: TaskPlan,
    *,
    discovery_checkpoint: DiscoveryCheckpoint | None,
) -> tuple[TaskSpec, ...]:
    """Return the tasks that consume research-task budget in ``plan``.

    Only a ``DiscoveryProbeTaskOrigin`` task is exempt from the research task
    count, and only while a ``DiscoveryCheckpoint`` context exists — without
    one the origin label alone never grants free capacity. Exemption validity
    is still proven separately by ``validate_task_plan`` (exact accepted-probe
    match, ``document.search``/``DocumentSearchInput`` shape): a malformed or
    spoofed probe origin is excluded here but fails there, so it can never
    become free capacity inside an accepted plan.
    """

    factual: list[TaskSpec] = []
    for task in plan.tasks:
        if isinstance(task.origin, DiscoveryProbeTaskOrigin):
            if discovery_checkpoint is None:
                _fail(
                    f"task {task.task_id} carries a discovery probe origin but "
                    "no DiscoveryCheckpoint context was supplied"
                )
            continue
        factual.append(task)
    return tuple(factual)


def validate_plan_expansion(
    current: TaskPlan,
    proposed: TaskPlan,
    checkpoint: DiscoveryCheckpoint,
    selection: ResearchTargetSelection,
    bindings: DocumentBindingSet,
    outcomes: tuple[AgentResult, ...],
    *,
    capability_names: frozenset[str],
    max_research_tasks: int,
    total_task_limit: int,
) -> None:
    """Spec §13.1: the frozen expansion invariant set.

    Raises :class:`ContractValidationError` on any violation; returns ``None``
    only when the whole invariant set holds. ``current`` is the checkpointed
    bootstrap plan (probe tasks only); ``proposed`` is the candidate
    append-only successor; ``outcomes`` are the typed checkpointed
    ``AgentResult``s of the attempted bootstrap tasks.
    """

    # Invariant 1: plan identity, goal, and contract version are unchanged.
    if proposed.contract_version != current.contract_version:
        _fail("plan expansion must keep the contract_version of the current plan")
    if proposed.plan_id != current.plan_id:
        _fail("plan expansion must keep the plan_id of the current plan")
    if proposed.goal != current.goal:
        _fail("plan expansion must keep the goal of the current plan")

    # Invariant 2: the checkpointed bootstrap plan is probe-only and carries
    # no target units; existing tasks and target units are exact prefixes, so
    # every existing task/unit is unchanged.
    if current.target_units:
        _fail("the current bootstrap plan must not carry target units")
    non_probe_tasks = [
        task.task_id
        for task in current.tasks
        if not isinstance(task.origin, DiscoveryProbeTaskOrigin)
    ]
    if non_probe_tasks:
        _fail(
            f"the current bootstrap plan carries non-probe task(s) "
            f"{sorted(non_probe_tasks)}"
        )
    if (
        len(proposed.tasks) < len(current.tasks)
        or proposed.tasks[: len(current.tasks)] != current.tasks
    ):
        _fail("plan expansion must keep existing tasks as an exact prefix")
    if (
        len(proposed.target_units) < len(current.target_units)
        or proposed.target_units[: len(current.target_units)] != current.target_units
    ):
        _fail("plan expansion must keep existing target units as an exact prefix")
    new_units = proposed.target_units[len(current.target_units) :]
    new_tasks = proposed.tasks[len(current.tasks) :]

    # Invariant 3: at least one target unit and one factual task appended.
    if not new_units:
        _fail("plan expansion must append at least one target unit")
    if not new_tasks:
        _fail("plan expansion must append at least one factual task")

    # Expansion is authorized only after discovery settled into selections.
    if checkpoint.status != "selected":
        _fail(
            f"plan expansion requires a 'selected' DiscoveryCheckpoint, got "
            f"{checkpoint.status!r}"
        )

    # Invariants 5, 6, and the selection half of 4/11: slot cardinality,
    # required minimums, distinct-document rules, the same-document/two-
    # distinct-locator compare exception, exact slot equality with the
    # checkpoint, and aggregate↔binding↔slot identity — all against the
    # CURRENT bindings and checkpoint.
    validate_research_target_selection(selection, bindings, checkpoint)

    selected_refs = _selected_refs(selection)
    binding_by_id = {binding.binding_id: binding for binding in bindings.bindings}
    aggregates_by_id = {
        aggregate.aggregate_id: aggregate
        for aggregate in checkpoint.slot_aggregates
    }
    selected_aggregates_by_slot = {
        record.slot_id: frozenset(record.selected_aggregate_ids)
        for record in checkpoint.selections
    }

    # Invariant 7 (units): new target IDs are unique and cannot shadow
    # existing (bootstrap) unit IDs.
    _require_unique(
        (unit.target_id for unit in new_units), "appended TargetUnit.target_id"
    )
    existing_target_ids = {unit.target_id for unit in current.target_units}

    for unit in new_units:
        if unit.target_id in existing_target_ids:
            _fail(
                f"appended target unit {unit.target_id} shadows an existing "
                "target id"
            )
        # Invariant 4: every new unit resolves to exactly one checkpoint-
        # selected SelectedBindingRef, uses that exact binding, and uses the
        # owning slot's exact checkpointed locator.
        selected = selected_refs.get(unit.target_id)
        if selected is None:
            _fail(
                f"appended target unit {unit.target_id} does not resolve to a "
                "selected binding"
            )
        slot, ref = selected
        if unit.binding_id != ref.binding_id:
            _fail(
                f"appended target unit {unit.target_id} binding "
                f"{unit.binding_id} does not match selected binding "
                f"{ref.binding_id}"
            )
        if unit.requested_locator != slot.requested_locator:
            _fail(
                f"appended target unit {unit.target_id} locator does not match "
                f"the owning slot {slot.slot_id} locator"
            )
        binding = binding_by_id.get(unit.binding_id)
        if binding is None:
            _fail(
                f"appended target unit {unit.target_id} references unknown "
                f"binding {unit.binding_id}"
            )
        if ref.authority == "explicit_binding":
            if ref.selected_aggregate_id is not None:
                _fail(
                    f"explicit-binding selection {ref.target_id} must not "
                    "carry a selected_aggregate_id"
                )
            continue
        # Invariants 5 + 11: a discovery-backed selected binding resolves
        # through selected_aggregate_id to the same slot and the exact
        # (document_id, document_revision) of the binding, and only an
        # aggregate recorded in DiscoveryCheckpoint.selections may authorize
        # the expansion.
        aggregate = aggregates_by_id.get(ref.selected_aggregate_id)
        if (
            aggregate is None
            or aggregate.slot_id != slot.slot_id
            or aggregate.document_id != binding.document_id
            or aggregate.document_revision != binding.document_revision
        ):
            _fail(
                f"selected aggregate {ref.selected_aggregate_id} does not "
                f"match slot {slot.slot_id} and binding {ref.binding_id} "
                "identity"
            )
        if ref.selected_aggregate_id not in selected_aggregates_by_slot.get(
            slot.slot_id, frozenset()
        ):
            _fail(
                f"aggregate {ref.selected_aggregate_id} is not recorded in "
                f"the checkpointed selections for slot {slot.slot_id} and "
                "cannot authorize expansion"
            )

    # Invariant 7 (tasks): new task IDs are unique and cannot shadow
    # bootstrap task IDs.
    _require_unique(
        (task.task_id for task in new_tasks), "appended TaskSpec.task_id"
    )
    existing_task_ids = {task.task_id for task in current.tasks}
    new_unit_ids = {unit.target_id for unit in new_units}
    new_task_target_ids: set[str] = set()

    for task in new_tasks:
        if task.task_id in existing_task_ids:
            _fail(
                f"appended task {task.task_id} shadows an existing task id"
            )
        # Invariant 8: current catalog capabilities and only selected/
        # appended target units; a targetless task has no expansion lineage.
        if task.capability not in capability_names:
            _fail(
                f"appended task {task.task_id} capability {task.capability!r} "
                "is not in the current capability catalog"
            )
        target_ids = _input_target_ids(task.input)
        if not target_ids:
            _fail(
                f"appended task {task.task_id} must read at least one "
                "appended selected unit"
            )
        for target_id in target_ids:
            if target_id not in new_unit_ids:
                _fail(
                    f"appended task {task.task_id} references target "
                    f"{target_id} that is not an appended selected unit"
                )
            new_task_target_ids.add(target_id)
        # Invariant 9 (shallow): every new task carries the expansion origin;
        # the exact slot/aggregate/source-task correspondence is proven by the
        # final selection- and discovery-aware validate_task_plan below.
        if not isinstance(task.origin, DiscoveryExpansionTaskOrigin):
            _fail(
                f"appended task {task.task_id} must carry a "
                "DiscoveryExpansionTaskOrigin"
            )
    unread_units = new_unit_ids - new_task_target_ids
    if unread_units:
        _fail(
            f"appended target unit(s) {sorted(unread_units)} have no "
            "factual task in the expansion"
        )

    # Invariant 10: typed checkpointed outcomes cover every attempted
    # bootstrap (probe) task exactly once, reference only current-plan tasks,
    # and preserve their non-success statuses.
    _require_unique(
        (result.task_id for result in outcomes), "AgentResult.task_id"
    )
    current_task_ids = {task.task_id for task in current.tasks}
    probe_task_ids = {
        task.task_id
        for task in current.tasks
        if isinstance(task.origin, DiscoveryProbeTaskOrigin)
    }
    outcome_by_task: dict[str, AgentResult] = {}
    for result in outcomes:
        if result.task_id not in current_task_ids:
            _fail(
                f"discovery outcome {result.task_id} does not belong to the "
                "bootstrap plan"
            )
        validate_agent_result(result, current)
        outcome_by_task[result.task_id] = result
    missing_outcomes = probe_task_ids - set(outcome_by_task)
    if missing_outcomes:
        _fail(
            f"attempted bootstrap task(s) {sorted(missing_outcomes)} have no "
            "checkpointed outcome"
        )
    extra_outcomes = set(outcome_by_task) - probe_task_ids
    if extra_outcomes:
        _fail(
            f"checkpointed outcome(s) {sorted(extra_outcomes)} do not belong "
            "to attempted bootstrap probe tasks"
        )
    # Every member of a checkpoint-selected aggregate requires a successful
    # source outcome; non-success statuses are preserved, never reinterpreted.
    for record in checkpoint.selections:
        for aggregate_id in record.selected_aggregate_ids:
            aggregate = aggregates_by_id.get(aggregate_id)
            if aggregate is None:
                _fail(
                    f"checkpoint selection names aggregate {aggregate_id} "
                    "absent from slot_aggregates"
                )
            for source_task_id in aggregate.source_task_ids:
                result = outcome_by_task.get(source_task_id)
                if result is None:
                    _fail(
                        f"selected aggregate {aggregate_id} source task "
                        f"{source_task_id} has no checkpointed outcome"
                    )
                if result.status != "success":
                    _fail(
                        f"selected aggregate {aggregate_id} source task "
                        f"{source_task_id} outcome status {result.status!r}; "
                        "a successful search outcome is required"
                    )

    # Invariants 10 (aggregate members) and 12 (discovery budget): rebuild
    # every candidate match from the accepted probes and checkpointed
    # outcomes, compare it to the checkpointed matches and aggregates on
    # (task_id, document_id, document_revision), and re-enforce the
    # probe/round/top-k budgets.
    validate_discovery_checkpoint(
        checkpoint, plan=current, task_results=outcomes
    )

    # Invariant 12 (research and total budgets): max_research_tasks caps the
    # factual task count of the final plan; total_task_limit caps ALL tasks.
    factual_count = len(
        factual_tasks(proposed, discovery_checkpoint=checkpoint)
    )
    if factual_count > max_research_tasks:
        _fail(
            f"expanded plan carries {factual_count} factual task(s) but the "
            f"research budget allows {max_research_tasks}"
        )
    if len(proposed.tasks) > total_task_limit:
        _fail(
            f"expanded plan carries {len(proposed.tasks)} task(s) but the "
            f"total task limit is {total_task_limit}"
        )

    # Invariant 13 (and the deep half of 9): the complete plan passes
    # selection- and discovery-aware validate_task_plan against the current
    # bindings and checkpoint — expansion-origin slots, aggregates, and
    # source-search lineage correspond exactly to the units each task reads.
    validate_task_plan(
        proposed,
        bindings,
        target_selection=selection,
        discovery_checkpoint=checkpoint,
    )


def expand_discovery_plan(
    current: TaskPlan,
    checkpoint: DiscoveryCheckpoint,
    selection: ResearchTargetSelection,
    bindings: DocumentBindingSet,
    outcomes: tuple[AgentResult, ...],
    appended_units: tuple[TargetUnit, ...],
    appended_tasks: tuple[TaskSpec, ...],
    *,
    capability_names: frozenset[str],
    max_research_tasks: int,
    total_task_limit: int,
) -> TaskPlan:
    """The single authoritative discovery-expansion append constructor.

    Builds the append-only successor of the checkpointed bootstrap plan,
    validates it through :func:`validate_plan_expansion`, and returns only the
    validated plan. Pure: no lease, persistence, database, registry,
    scheduler, or runtime call happens here — the governed caller
    (``research_expand_node``, Phase 4) owns leasing before checkpointing.
    """

    proposed = current.model_copy(
        update={
            "target_units": current.target_units + tuple(appended_units),
            "tasks": current.tasks + tuple(appended_tasks),
        }
    )
    validate_plan_expansion(
        current,
        proposed,
        checkpoint,
        selection,
        bindings,
        outcomes,
        capability_names=capability_names,
        max_research_tasks=max_research_tasks,
        total_task_limit=total_task_limit,
    )
    return proposed

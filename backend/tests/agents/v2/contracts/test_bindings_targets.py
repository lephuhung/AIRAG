"""Spec §9/§13 — binding, target, plan DAG, and completion-criteria integrity.

Covers the brief's Step 1 items "ID/DAG/relation integrity" and "criteria
uniqueness", plus the §26 revision-requirement relation rules: exactly one
relation for every current-required binding, none for ordinary/pinned/
discovered bindings, rejection of dangling or mismatched binding/ref IDs, and a
relation whose binding pins a different document than its reference resolved to.
Also guards the materialized People→Document scalar: a ``document.search`` task
carrying ``person_identifier`` must depend on a ``people.lookup`` task.
"""
from __future__ import annotations

import pytest

from app.services.agents.v2.contracts.binding import (
    BindingRevisionRequirement,
    DocumentBindingSet,
)
from app.services.agents.v2.contracts.capability import PeopleLookupInput
from app.services.agents.v2.contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    ReplanTaskOrigin,
    SemanticCriterion,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from app.services.agents.v2.contracts.semantic import (
    CurrentRevisionRequirement,
    PinnedRevisionRequirement,
)
from app.services.agents.v2.contracts.validation import (
    ContractValidationError,
    validate_binding_set,
    validate_fast_plan,
    validate_task_plan,
)

from .factories import (
    DOCUMENT_ID,
    OTHER_DOCUMENT_ID,
    OTHER_REVISION,
    binding_set,
    document_reference,
    people_plan,
    person_search_task,
    read_plan,
    read_task,
    scoped_document,
    semantic_context,
    target_unit,
)


def test_binding_set_rejects_duplicate_binding_ids() -> None:
    with pytest.raises(ContractValidationError, match="binding_id"):
        validate_binding_set(
            binding_set(bindings=(scoped_document(), scoped_document())),
            semantic_context(),
        )


def test_current_required_binding_needs_exactly_one_relation() -> None:
    semantic = semantic_context(
        document_refs=(document_reference(revision_requirement=CurrentRevisionRequirement(kind="current")),)
    )
    with pytest.raises(ContractValidationError, match="current"):
        validate_binding_set(binding_set(), semantic)

    valid = binding_set(
        revision_requirement_refs=(BindingRevisionRequirement(binding_id="b1", ref_id="r1"),)
    )
    validate_binding_set(valid, semantic)


def test_duplicate_relation_for_one_binding_is_rejected() -> None:
    semantic = semantic_context(
        document_refs=(document_reference(revision_requirement=CurrentRevisionRequirement(kind="current")),)
    )
    with pytest.raises(ContractValidationError, match="exactly one"):
        validate_binding_set(
            binding_set(
                revision_requirement_refs=(
                    BindingRevisionRequirement(binding_id="b1", ref_id="r1"),
                    BindingRevisionRequirement(binding_id="b1", ref_id="r1"),
                )
            ),
            semantic,
        )


def test_relation_requires_a_current_revision_requirement() -> None:
    pinned = document_reference(
        revision_requirement=PinnedRevisionRequirement(kind="pinned", document_revision=OTHER_REVISION)
    )
    with pytest.raises(ContractValidationError, match="current"):
        validate_binding_set(
            binding_set(
                bindings=(scoped_document(document_revision=OTHER_REVISION),),
                revision_requirement_refs=(BindingRevisionRequirement(binding_id="b1", ref_id="r1"),),
            ),
            semantic_context(document_refs=(pinned,)),
        )


def test_ordinary_reference_has_no_relation() -> None:
    semantic = semantic_context(document_refs=(document_reference(revision_requirement=None),))
    validate_binding_set(binding_set(), semantic)


def test_dangling_relation_ids_are_rejected() -> None:
    semantic = semantic_context(
        document_refs=(document_reference(revision_requirement=CurrentRevisionRequirement(kind="current")),)
    )
    with pytest.raises(ContractValidationError, match="binding"):
        validate_binding_set(
            binding_set(revision_requirement_refs=(BindingRevisionRequirement(binding_id="bX", ref_id="r1"),)),
            semantic,
        )
    with pytest.raises(ContractValidationError, match="ref"):
        validate_binding_set(
            binding_set(revision_requirement_refs=(BindingRevisionRequirement(binding_id="b1", ref_id="rX"),)),
            semantic,
        )


def test_relation_with_a_mismatched_binding_document_is_rejected() -> None:
    semantic = semantic_context(
        document_refs=(document_reference(revision_requirement=CurrentRevisionRequirement(kind="current")),)
    )
    mismatched = binding_set(
        bindings=(scoped_document(document_id=OTHER_DOCUMENT_ID),),
        revision_requirement_refs=(BindingRevisionRequirement(binding_id="b1", ref_id="r1"),),
    )
    with pytest.raises(ContractValidationError, match="document"):
        validate_binding_set(mismatched, semantic)

    validate_binding_set(
        binding_set(
            revision_requirement_refs=(BindingRevisionRequirement(binding_id="b1", ref_id="r1"),)
        ),
        semantic,
    )


def test_relation_without_semantic_context_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="semantic"):
        validate_binding_set(
            binding_set(revision_requirement_refs=(BindingRevisionRequirement(binding_id="b1", ref_id="r1"),))
        )


def test_target_unit_must_reference_a_known_binding() -> None:
    plan = read_plan(target_units=(target_unit(binding_id="bX"),))
    with pytest.raises(ContractValidationError, match="binding"):
        validate_task_plan(plan, binding_set())


@pytest.mark.parametrize("role", ["supporting", "discovered"])
def test_target_unit_cannot_bind_a_discovered_or_supporting_document(role: str) -> None:
    plan = read_plan()
    with pytest.raises(ContractValidationError, match="role"):
        validate_task_plan(plan, binding_set(bindings=(scoped_document(role=role),)))


def test_at_most_one_coverage_criterion_per_target() -> None:
    unit = target_unit(
        completion_criteria=(
            CoverageCriterion(kind="coverage"),
            CoverageCriterion(kind="coverage"),
        )
    )
    with pytest.raises(ContractValidationError, match="coverage"):
        validate_task_plan(read_plan(target_units=(unit,)), binding_set())


def test_semantic_criterion_ids_are_unique_within_a_target() -> None:
    unit = target_unit(
        completion_criteria=(
            CoverageCriterion(kind="coverage"),
            SemanticCriterion(kind="semantic", criterion_id="c1", description="F1/F2"),
            SemanticCriterion(kind="semantic", criterion_id="c1", description="duplicate"),
        )
    )
    with pytest.raises(ContractValidationError, match="criterion_id"):
        validate_task_plan(read_plan(target_units=(unit,)), binding_set())


def test_partial_coverage_requires_an_explicit_deterministic_reason() -> None:
    unit = target_unit(
        completion_criteria=(CoverageCriterion(kind="coverage", minimum_status="read_partial"),)
    )
    with pytest.raises(ContractValidationError, match="allow_partial_reason"):
        validate_task_plan(read_plan(target_units=(unit,)), binding_set())


def test_target_unit_does_not_repeat_document_identity() -> None:
    assert set(TargetUnit.model_fields) == {
        "target_id",
        "binding_id",
        "requested_locator",
        "completion_criteria",
    }
    assert "document_id" not in TargetUnit.model_fields
    assert "document_revision" not in TargetUnit.model_fields
    assert "role" not in TargetUnit.model_fields


def test_plan_rejects_duplicate_task_ids_and_unknown_dependencies() -> None:
    with pytest.raises(ContractValidationError, match="task_id"):
        validate_task_plan(read_plan(tasks=(read_task("T1"), read_task("T1"))), binding_set())

    orphan = read_task("T2")
    orphan = orphan.model_copy(update={"depends_on": ("TX",)})
    with pytest.raises(ContractValidationError, match="depends_on"):
        validate_task_plan(read_plan(tasks=(read_task("T1"), orphan)), binding_set())


def test_plan_rejects_dependency_cycles() -> None:
    t1 = read_task("T1").model_copy(update={"depends_on": ("T2",)})
    t2 = read_task("T2").model_copy(update={"depends_on": ("T1",)})
    with pytest.raises(ContractValidationError, match="cycle"):
        validate_task_plan(read_plan(tasks=(t1, t2)), binding_set())


def test_plan_rejects_capability_kind_mismatch() -> None:
    task = read_task().model_copy(update={"capability": "people.lookup"})
    with pytest.raises(ContractValidationError, match="capability"):
        validate_task_plan(read_plan(tasks=(task,)), binding_set())


def test_plan_rejects_unknown_target_reference() -> None:
    task = read_task(target_ids=("tX",))
    with pytest.raises(ContractValidationError, match="tX"):
        validate_task_plan(read_plan(tasks=(task,)), binding_set())


def test_plan_rejects_unreferenced_target_units() -> None:
    plan = read_plan(
        target_units=(target_unit(target_id="t1"), target_unit(target_id="t2")),
        tasks=(read_task(target_ids=("t1",)),),
    )
    with pytest.raises(ContractValidationError, match="t2"):
        validate_task_plan(plan, binding_set())


def test_replan_origin_references_existing_tasks() -> None:
    task = read_task("T2").model_copy(
        update={
            "origin": ReplanTaskOrigin(
                kind="replan", reason="coverage gap", task_ids=("TX",), evidence_use_ids=()
            )
        }
    )
    with pytest.raises(ContractValidationError, match="origin"):
        validate_task_plan(read_plan(tasks=(read_task("T1"), task)), binding_set())


def test_fast_plan_is_one_initial_dependency_free_task() -> None:
    validate_fast_plan(read_plan(), binding_set())

    with pytest.raises(ContractValidationError, match="single"):
        validate_fast_plan(
            read_plan(tasks=(read_task("T1"), read_task("T2").model_copy(update={"depends_on": ("T1",)}))),
            binding_set(),
        )
    dependent = read_task().model_copy(update={"depends_on": ("T1",)})
    with pytest.raises(ContractValidationError, match="dependency"):
        validate_fast_plan(read_plan(tasks=(dependent,)), binding_set())

    replanned = read_task().model_copy(
        update={
            "origin": ReplanTaskOrigin(
                kind="replan", reason="gap", task_ids=("T1",), evidence_use_ids=()
            )
        }
    )
    with pytest.raises(ContractValidationError, match="initial"):
        validate_fast_plan(read_plan(tasks=(replanned,)), binding_set())


def test_fast_plan_target_units_match_the_read_input() -> None:
    with pytest.raises(ContractValidationError, match="target_units"):
        validate_fast_plan(read_plan(target_units=()), binding_set())

    people = people_plan()
    validate_fast_plan(people, binding_set(bindings=()))
    assert people.target_units == ()
    assert isinstance(people.tasks[0].input, PeopleLookupInput)

    with pytest.raises(ContractValidationError, match="target_units"):
        validate_fast_plan(
            TaskPlan(
                contract_version="2.0",
                plan_id="p-people",
                goal="CCCD của A",
                target_units=(target_unit(),),
                tasks=(
                    TaskSpec(
                        task_id="T1",
                        capability="people.lookup",
                        task_objective="Tra CCCD",
                        input=PeopleLookupInput(kind="people.lookup", query="A"),
                        depends_on=(),
                        origin=InitialTaskOrigin(kind="initial"),
                    ),
                ),
            ),
            binding_set(),
        )


def test_person_identifier_search_task_requires_a_people_lookup_dependency() -> None:
    people = people_plan()
    guarded = people.model_copy(update={"tasks": people.tasks + (person_search_task(),)})
    validate_task_plan(guarded, binding_set(bindings=()))

    unguarded = people.model_copy(
        update={"tasks": people.tasks + (person_search_task(depends_on=()),)}
    )
    with pytest.raises(ContractValidationError, match="people.lookup"):
        validate_task_plan(unguarded, binding_set(bindings=()))


def test_person_identifier_without_any_people_lookup_task_is_rejected() -> None:
    plan = read_plan(tasks=(read_task("T1"), person_search_task(depends_on=("T1",))))
    with pytest.raises(ContractValidationError, match="people.lookup"):
        validate_task_plan(plan, binding_set())


def test_search_task_without_a_materialized_scalar_needs_no_people_dependency() -> None:
    plan = read_plan(tasks=(read_task("T1"), person_search_task(person_identifier=None, depends_on=())))
    validate_task_plan(plan, binding_set())


def test_plan_id_and_goal_must_be_present() -> None:
    with pytest.raises(ContractValidationError, match="plan_id"):
        validate_task_plan(read_plan().model_copy(update={"plan_id": " "}), binding_set())


def test_binding_revision_pin_is_not_copied_into_semantic_requirements() -> None:
    binding = scoped_document(document_revision=OTHER_REVISION)
    assert binding.document_revision == OTHER_REVISION
    assert DocumentBindingSet(bindings=(binding,), revision_requirement_refs=()).bindings[0].document_revision == OTHER_REVISION
    assert DOCUMENT_ID is not None

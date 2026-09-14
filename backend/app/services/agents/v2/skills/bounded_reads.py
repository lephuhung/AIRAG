"""Shared bounded-read builders for the Task-12 expansion skills (internal).

``multi_goal``, ``cross_domain`` (generic branch), and ``evaluate`` all
propose the same governed shape — one bounded read per distinct already-bound
document, no dependencies — so the mechanics live here once instead of being
cloned per skill. This module is NOT a skill: it owns no work type, proposes
no plan, executes nothing, and checkpoints nothing. Each skill keeps its own
intake thresholds, plan-ID scheme, and budget caps, and still passes its plan
through frozen ``validate_task_plan`` at the governed entry.
"""
from __future__ import annotations

from uuid import UUID

from ..adapters.document import binding_id_for_ref
from ..contracts.binding import ScopedDocument
from ..contracts.capability import DocumentReadInput, SectionReadInput
from ..contracts.locators import ContentLocator, DocumentLocator, SectionLocator
from ..contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    TargetUnit,
    TaskSpec,
)
from ..contracts.semantic import SemanticContext
from ..contracts.validation import ContractValidationError

__all__ = [
    "PLANNABLE_ROLES",
    "coverage_unit",
    "locator_for",
    "plannable_bindings",
    "read_task",
    "require_read_capabilities",
]

#: Only settled roles anchor target units; discovered/supporting roles can
#: never become execution targets, so the planner cannot promote unsettled
#: discovery into targets.
PLANNABLE_ROLES = frozenset({"target", "reference"})


def plannable_bindings(bindings: object) -> tuple[ScopedDocument, ...]:
    """Distinct settled-role bindings in deterministic order.

    Bindings that resolve to the same document identity collapse to the first
    in ``binding_id`` order (mirroring the retrieve skill), so one document
    is never read twice; distinct documents are never merged, so the hard
    scope is preserved.
    """
    ordered = sorted(
        bindings.bindings,  # type: ignore[union-attr]
        key=lambda binding: binding.binding_id,
    )
    seen: set[UUID] = set()
    distinct: list[ScopedDocument] = []
    for binding in ordered:
        if binding.role not in PLANNABLE_ROLES:
            continue
        if binding.document_id in seen:
            continue
        seen.add(binding.document_id)
        distinct.append(binding)
    return tuple(distinct)


def locator_for(binding: ScopedDocument, semantic: SemanticContext) -> ContentLocator:
    """Exact requested coordinate for one side: section when named, else whole.

    Same convention as the compare skill: a section reference names its side
    through the canonical binding-ID convention (``binding_id_for_ref``); the
    first match in ``ref_id`` order wins deterministically. A side with no
    named section keeps the whole-document coordinate — the only permitted
    fallback.
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


def read_task(
    task_id: str,
    target_id: str,
    binding: ScopedDocument,
    locator: ContentLocator,
    *,
    objective_noun: str,
) -> TaskSpec:
    """One bounded side read: section reads for section coordinates."""
    if isinstance(locator, SectionLocator):
        return TaskSpec(
            task_id=task_id,
            capability="section.read",
            task_objective=(
                f"Read section {locator.structure_node_id} for {objective_noun} "
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
            f"Read the {objective_noun} side "
            f"(document {binding.document_id})"
        ),
        input=DocumentReadInput(kind="document.read", target_ids=(target_id,)),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def coverage_unit(
    target_id: str,
    binding: ScopedDocument,
    locator: ContentLocator,
) -> TargetUnit:
    """One target unit with the default coverage criterion (read_complete).

    Read observations (``document.read``/``section.read``) establish
    ``read_complete`` at evaluation, so the default criterion is satisfiable
    by construction — unlike retrieval chunks, which establish only
    ``read_partial``.
    """
    return TargetUnit(
        target_id=target_id,
        binding_id=binding.binding_id,
        requested_locator=locator,
        completion_criteria=(CoverageCriterion(kind="coverage"),),
    )


def require_read_capabilities(
    catalog_names: set[str], tasks: tuple[TaskSpec, ...], *, owner: str
) -> None:
    """Fail closed when the runtime catalog cannot serve the planned reads."""
    for task in tasks:
        if task.capability not in catalog_names:
            raise ContractValidationError(
                f"{owner} requires {task.capability!r} in the request-scoped "
                "capability catalog; refusing to plan an undispatchable read"
            )

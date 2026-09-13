"""People -> Document deterministic dependency materialization (Phase 3, Task 4).

The governed People scalar is materialized server-side into a concrete
``DocumentSearchInput`` BEFORE the dependent task is appended and checkpointed.
The exact sequence (R23) is::

    T1 people.lookup
    -> AgentResult / governed EvidenceUse
    -> deterministic materialization (this module)
    -> current ACL + expiry + minimization (governed hydrator)
    -> extract exact approved scalar server-side
    -> build concrete DocumentSearchInput
    -> append T2 TaskSpec(input=<concrete input>)
    -> validate_replan
    -> retention-lease commit (caller node, before the checkpoint)
    -> checkpoint updated TaskPlan
    -> TaskScheduler dispatches T2 unchanged

Rules enforced here:

- ``depends_on`` expresses ordering ONLY. Materialization happens before T2 is
  appended/checkpointed, never inside the scheduler.
- Scalar extraction reads ONLY the governed People ``EvidenceRecord`` content
  admitted through the request-scoped evidence hydrator under current runtime
  authorization/expiry. It never calls the People connector and never reads a
  raw record.
- Exactly one approved identifier field (``PERSON_IDENTIFIER_FIELD``). Any
  other field -- including raw aliases such as ``cccd`` -- never satisfies the
  extractor: absent/blank/ambiguous/unauthorized/expired means no T2, never a
  guessed or blank ``person_identifier``.
- Distinct typed outcomes: ``not_found`` -> no T2; ``denied``
  (``PERMISSION_DENIED``) -> no T2; ``failed`` (``TIMEOUT``/error) -> no T2;
  ``needs_input``/``unavailable`` -> no T2. Denial, timeout, and error remain
  DISTINCT from ``not_found``.
- No ``TaskOutputRef`` is added to the frozen contracts; every frozen type is
  imported, never redefined.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from ..contracts.capability import DocumentSearchInput, PeopleLookupOutput
from ..contracts.evidence import PeopleSourceIdentity
from ..contracts.execution import AgentResult, TaskExecutionSummary
from ..contracts.planning import (
    DiscoveryPolicy,
    ReplanTaskOrigin,
    ResearchBudgetView,
    TaskPlan,
    TaskSpec,
)
from ..contracts.validation import validate_replan

__all__ = [
    "PERSON_IDENTIFIER_FIELD",
    "MaterializationKind",
    "MaterializationError",
    "PeopleDocumentMaterialization",
    "append_materialized_dependent",
    "build_dependent_search_task",
    "extract_person_identifier",
    "materialize_person_dependency",
    "redact_scalar_for_model",
]

#: The single approved People identifier field the materializer may extract.
#: It is the only scalar the governed People evidence is allowed to supply to
#: a downstream ``document.search``; every other raw field (``cccd`` alias,
#: DOB, address, phone, email, personnel data) is ignored by the extractor.
PERSON_IDENTIFIER_FIELD: str = "national_id"

MaterializationKind = Literal[
    "materialized", "not_found", "denied", "failed", "needs_input", "unavailable"
]


class MaterializationError(ValueError):
    """The dependency cannot be materialized; no dependent task may be built."""


@dataclass(frozen=True)
class PeopleDocumentMaterialization:
    """The typed outcome of one deterministic materialization attempt.

    ``kind == "materialized"`` is the ONLY outcome that carries a concrete
    ``DocumentSearchInput`` (with the exact governed scalar) and may proceed
    to the T2 append. Every other kind carries ``input=None``/``scalar=None``
    and means no T2. ``error_code`` preserves the typed People failure
    (``PERMISSION_DENIED``/``TIMEOUT``/...) so denial, timeout, and error stay
    distinct from ``not_found``; ``evidence_use_ids`` preserves the admitted
    triggering-use context for the ``ReplanTaskOrigin``.
    """

    kind: MaterializationKind
    scalar: str | None
    input: DocumentSearchInput | None
    error_code: str | None
    reason: str
    evidence_use_ids: tuple[UUID, ...] = ()


def extract_person_identifier(content: str) -> str | None:
    """Extract the exact approved scalar from governed minimized content.

    Returns the stripped scalar iff the canonical minimized JSON object carries
    a non-blank string at exactly ``PERSON_IDENTIFIER_FIELD``. Returns ``None``
    (fail closed) for any other shape: non-JSON, non-object, absent field,
    blank value, or non-string value. There is no alias fallback and no
    guessing -- a raw ``cccd``/``phone``/``email`` field never satisfies this.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get(PERSON_IDENTIFIER_FIELD)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _outcome(
    kind: MaterializationKind,
    *,
    reason: str,
    error_code: str | None = None,
) -> PeopleDocumentMaterialization:
    return PeopleDocumentMaterialization(
        kind=kind,
        scalar=None,
        input=None,
        error_code=error_code,
        reason=reason,
    )


async def materialize_person_dependency(
    *,
    people_task_id: str,
    people_result: AgentResult,
    runtime: Any,
    plan: TaskPlan,
    bindings: Any,
    query: str,
) -> PeopleDocumentMaterialization:
    """Materialize the governed People scalar into a concrete search input.

    ``query`` is the caller-supplied (checkpointed plan goal or planner
    proposal) search text; only the ``person_identifier`` scalar is governed
    materialization. ``runtime`` supplies the request-scoped evidence hydrator
    (current ACL/expiry) and ``plan``/``bindings`` scope the hydration exactly
    like evaluation. The People connector is never called and no raw record is
    ever read: only hydrator-admitted governed content is examined.
    """
    if people_result.task_id != people_task_id:
        raise MaterializationError(
            f"people result for task {people_result.task_id!r} cannot supply "
            f"the dependency of task {people_task_id!r}"
        )
    owner = next(
        (task for task in plan.tasks if task.task_id == people_task_id), None
    )
    if owner is None:
        raise MaterializationError(
            f"task {people_task_id!r} is not in the checkpointed plan; "
            "refusing to materialize for an unplanned task"
        )
    if owner.capability != "people.lookup":
        raise MaterializationError(
            f"task {people_task_id!r} is a {owner.capability!r} task, not "
            "people.lookup; refusing to materialize from the wrong source"
        )
    status = people_result.status
    if status == "not_found":
        return _outcome("not_found", reason="people.lookup found no record; no T2")
    if status == "denied":
        code = (
            people_result.error.code
            if people_result.error is not None
            else "PERMISSION_DENIED"
        )
        return _outcome(
            "denied",
            reason="people.lookup was denied; no T2",
            error_code=str(code),
        )
    if status == "error":
        code = (
            people_result.error.code
            if people_result.error is not None
            else "INTERNAL_ERROR"
        )
        return _outcome(
            "failed",
            reason=f"people.lookup failed with {code}; no T2",
            error_code=str(code),
        )
    if status == "needs_input":
        return _outcome(
            "needs_input", reason="people.lookup needs input; no T2"
        )
    if status != "success":
        return _outcome(
            "unavailable",
            reason=f"people.lookup status {status!r} cannot supply a scalar; no T2",
        )
    data = people_result.data
    if not isinstance(data, PeopleLookupOutput) or not data.matched:
        return _outcome(
            "unavailable",
            reason="people.lookup success carries no matched record; no T2",
        )
    if not people_result.evidence_uses:
        return _outcome(
            "unavailable",
            reason="people.lookup success carries no governed EvidenceUse; no T2",
        )
    services = getattr(runtime, "services", None)
    hydrator = getattr(services, "evidence_hydrator", None) if services else None
    if hydrator is None:
        raise MaterializationError(
            "no evidence_hydrator wired on runtime.services; refusing to "
            "materialize without governed hydration"
        )
    if not query or not query.strip():
        raise MaterializationError(
            "the dependent document.search needs a non-blank query; refusing "
            "to fabricate one"
        )
    admitted = await hydrator.hydrate_for_evaluation(
        tuple(people_result.evidence_uses),
        runtime=runtime,
        plan=plan,
        bindings=bindings,
    )
    scalars: list[str] = []
    admitted_use_ids: list[UUID] = []
    owned_items = 0
    for item in admitted or ():
        # R29: each hydrated item must be People evidence owned by the
        # supplying people.lookup task, verified on the TYPED source
        # identity (non-spoofable) -- never on the human-readable label.
        # A use attached to another task, a target-bound read use, or a
        # non-People record is ignored -- fail closed -- never a scalar
        # source.
        if getattr(item, "task_id", None) != people_task_id:
            continue
        if not isinstance(getattr(item, "source_identity", None), PeopleSourceIdentity):
            continue
        if getattr(item, "purpose", None) != "supporting":
            continue
        if getattr(item, "target_id", None) is not None:
            continue
        if getattr(item, "locator", None) is not None:
            continue
        if getattr(item, "document_revision", None) is not None:
            continue
        owned_items += 1
        content = getattr(item, "content", None)
        scalar = extract_person_identifier(content) if content is not None else None
        if scalar is None:
            continue
        use_id = getattr(item, "use_id", None)
        if use_id is not None:
            admitted_use_ids.append(use_id)
        if scalar not in scalars:
            scalars.append(scalar)
    if not admitted:
        return _outcome(
            "unavailable",
            reason=(
                "no governed People evidence admitted under current "
                "ACL/expiry; no T2"
            ),
        )
    if not owned_items:
        return _outcome(
            "unavailable",
            reason=(
                "admitted evidence is not People evidence owned by task "
                f"{people_task_id!r}; no T2"
            ),
        )
    if not scalars:
        return _outcome(
            "unavailable",
            reason=(
                "admitted People evidence carries no extractable "
                f"{PERSON_IDENTIFIER_FIELD!r} scalar; no T2"
            ),
        )
    if len(scalars) > 1:
        return _outcome(
            "unavailable",
            reason="admitted People evidence carries conflicting scalars; no T2",
        )
    scalar = scalars[0]
    return PeopleDocumentMaterialization(
        kind="materialized",
        scalar=scalar,
        input=DocumentSearchInput(
            kind="document.search",
            query=query.strip(),
            person_identifier=scalar,
        ),
        error_code=None,
        reason="governed scalar materialized server-side",
        evidence_use_ids=tuple(admitted_use_ids),
    )


def build_dependent_search_task(
    outcome: PeopleDocumentMaterialization,
    *,
    people_task_id: str,
    query: str,
    next_task_id: str,
) -> TaskSpec:
    """Build the concrete T2 ``TaskSpec`` from a materialized outcome.

    Only ``kind == "materialized"`` may build: any other outcome raises
    instead of fabricating a guessed/blank ``person_identifier``. The caller
    supplies ONLY the search query; the ``person_identifier`` always comes
    from the materialized scalar -- the append path can never overwrite it.
    The concrete input is built fresh here (single construction site), never
    carried over from a caller-supplied object.
    """
    if outcome.kind != "materialized" or outcome.scalar is None or outcome.input is None:
        raise MaterializationError(
            f"dependency outcome {outcome.kind!r} ({outcome.reason}) cannot "
            "build a dependent task; no T2"
        )
    if not next_task_id or not next_task_id.strip():
        raise MaterializationError("the dependent task needs a non-blank task id")
    if not query or not query.strip():
        raise MaterializationError(
            "the dependent document.search needs a non-blank query; refusing "
            "to fabricate one"
        )
    concrete = DocumentSearchInput(
        kind="document.search",
        query=query.strip(),
        person_identifier=outcome.scalar,
    )
    return TaskSpec(
        task_id=next_task_id,
        capability="document.search",
        task_objective=f"Search documents for the person resolved by {people_task_id}",
        input=concrete,
        depends_on=(people_task_id,),
        origin=ReplanTaskOrigin(
            kind="replan",
            reason=f"people.lookup dependency {people_task_id} materialized",
            task_ids=(people_task_id,),
            evidence_use_ids=outcome.evidence_use_ids,
        ),
    )


def append_materialized_dependent(
    *,
    current: TaskPlan,
    outcomes: tuple[TaskExecutionSummary, ...],
    outcome: PeopleDocumentMaterialization,
    query: str,
    next_task_id: str,
    policy: DiscoveryPolicy,
    budget: ResearchBudgetView,
) -> TaskPlan:
    """Append T2 and validate the append-only replan (no checkpoint here).

    Builds the concrete T2 spec and returns ``validate_replan``'s accepted
    plan. The caller node commits retention leases and checkpoints the
    returned plan before the scheduler ever sees T2. Raises
    ``MaterializationError`` for a non-materialized outcome and
    ``ContractValidationError`` for an invalid append.
    """
    if outcome.kind != "materialized":
        raise MaterializationError(
            f"dependency outcome {outcome.kind!r} ({outcome.reason}) cannot "
            "append a dependent task; no T2"
        )
    taken = {task.task_id for task in current.tasks}
    if next_task_id in taken:
        raise MaterializationError(
            f"dependent task id {next_task_id!r} is already in the plan"
        )
    task = build_dependent_search_task(
        outcome,
        people_task_id=_people_task_id_for(current, outcome),
        query=query,
        next_task_id=next_task_id,
    )
    proposed = current.model_copy(update={"tasks": current.tasks + (task,)})
    return validate_replan(current, proposed, outcomes, policy, budget)


def redact_scalar_for_model(plan: TaskPlan) -> TaskPlan:
    """Return a model-facing projection of ``plan`` with scalars redacted (R25).

    The frozen contract keeps ``person_identifier`` on the checkpointed
    ``DocumentSearchInput`` (planner-visible through
    ``ResearchPlanningInput.current_plan`` by contract). This helper is the
    defence-in-depth boundary for model-facing planning/replanning
    projections: every ``document.search`` input in the returned plan carries
    ``person_identifier=None`` while task identity, ordering, objectives, and
    ``depends_on`` are preserved, so the planner still sees that T2 depends
    on T1 without ever seeing the scalar. T5 (model planner owner) MUST
    consume this helper for any model-facing plan projection; the
    checkpointed plan itself is never redacted.
    """
    redacted: list[TaskSpec] = []
    for task in plan.tasks:
        if isinstance(task.input, DocumentSearchInput) and task.input.person_identifier is not None:
            redacted.append(
                task.model_copy(
                    update={
                        "input": DocumentSearchInput(
                            kind="document.search",
                            query=task.input.query,
                            person_identifier=None,
                        )
                    }
                )
            )
        else:
            redacted.append(task)
    return plan.model_copy(update={"tasks": tuple(redacted)})


def _people_task_id_for(
    current: TaskPlan, outcome: PeopleDocumentMaterialization
) -> str:
    """Resolve the single owning people.lookup task for the append."""
    _ = outcome
    owners = [
        task.task_id for task in current.tasks if task.capability == "people.lookup"
    ]
    if len(owners) != 1:
        raise MaterializationError(
            f"the dependent task needs exactly one owning people.lookup task, "
            f"found {owners}; refusing to guess the dependency"
        )
    return owners[0]

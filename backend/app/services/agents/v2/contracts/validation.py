"""Pure validators for the canonical v2 contracts (spec §13, §15, §16, §25, §26).

Every function is deterministic, side-effect free, and takes already-typed
contract values: the same validators serve the planner, the replan gateway, the
evaluator, grounding, and checkpoint loading. They raise
:class:`ContractValidationError` (or :class:`IncompatibleCheckpointError`) rather
than returning partial results, so a caller can never act on an unresolved,
cross-run, target-incompatible, or purpose-incompatible fact.

Capability execution and persistence are out of scope here; the Evidence Store and
repositories apply the runtime checks (ACL, retention, key availability) that
cannot be expressed on pure values.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeVar
from uuid import UUID

from .base import CONTRACT_VERSION, ContractModel
from .binding import (
    BindingRevisionRequirement,
    DocumentBindingSet,
    ScopedDocument,
)
from .capability import (
    DocumentReadInput,
    SectionReadInput,
    WriteInput,
)
from .clarification import ClarificationRequest, ClarificationResolution
from .conversation import ConversationContext, ConversationSnapshot
from .evaluation import EvidenceEvaluation
from .evidence import (
    DerivedSourceIdentity,
    DocumentSourceIdentity,
    EvidenceRecord,
    EvidenceStoreRow,
    EvidenceUse,
    KnowledgeGraphSourceIdentity,
    MemorySourceIdentity,
    PeopleSourceIdentity,
)
from .execution import AgentResult, TaskExecutionSummary
from .planning import (
    CoverageCriterion,
    DiscoveryPolicy,
    InitialTaskOrigin,
    ReplanTaskOrigin,
    ResearchBudgetView,
    ResearchPlanningInput,
    SemanticCriterion,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from .request import RequestContext
from .response import FinalResponse
from .routing import QueryAnalysis, RouteDecision
from .semantic import CurrentRevisionRequirement, DocumentReference, SemanticContext
from .state import ExecutionState, SupervisorV2State
from .synthesis import AnswerDraft, SynthesisInput


class ContractValidationError(ValueError):
    """A canonical contract value violates a frozen v2 invariant."""


class IncompatibleCheckpointError(ContractValidationError):
    """A checkpoint/fixture is not a compatible v2 payload and must not be migrated."""


_READ_CAPABILITIES = frozenset({"document.read", "section.read"})

# Spec §13.4: only these input variants reference logical targets.
_TARGET_BEARING_INPUTS = (DocumentReadInput, SectionReadInput, WriteInput)

# Spec §7: keys every v2 checkpoint aggregate must carry.
_CHECKPOINT_REQUIRED_KEYS = (
    "request",
    "conversation",
    "semantic",
    "bindings",
    "execution",
    "query_analysis",
    "route_decision",
    "clarification",
    "final_response",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ModelT = TypeVar("ModelT", bound=ContractModel)


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _require_non_blank(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be a non-blank string")


def _require_unique(values: Iterable[object], field: str) -> None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            _fail(f"duplicate {field}: {value!r}")
        seen.add(value)


def _input_target_ids(capability_input: object) -> tuple[str, ...]:
    if isinstance(capability_input, _TARGET_BEARING_INPUTS):
        return capability_input.target_ids
    return ()


def _task_by_id(plan: TaskPlan, task_id: str) -> TaskSpec:
    for task in plan.tasks:
        if task.task_id == task_id:
            return task
    _fail(f"reference to unknown task {task_id}")


def _target_by_id(plan: TaskPlan, target_id: str) -> TargetUnit:
    for unit in plan.target_units:
        if unit.target_id == target_id:
            return unit
    _fail(f"reference to unknown target {target_id}")


def _binding_by_id(bindings: DocumentBindingSet, binding_id: str) -> ScopedDocument:
    for binding in bindings.bindings:
        if binding.binding_id == binding_id:
            return binding
    _fail(f"reference to unknown binding {binding_id}")


def _document_reference_by_id(semantic: SemanticContext, ref_id: str) -> DocumentReference:
    for reference in semantic.document_refs:
        if reference.ref_id == ref_id:
            return reference
    _fail(f"revision-requirement relation references unknown ref {ref_id}")


def _as_model(value: object, key: str, model: type[ModelT]) -> ModelT:
    if isinstance(value, model):
        return value
    if isinstance(value, Mapping):
        try:
            return model.model_validate(value)
        except Exception as error:  # noqa: BLE001 - normalised into a checkpoint error
            raise IncompatibleCheckpointError(
                f"checkpoint {key} is not a valid {model.__name__}: {error}"
            ) from error
    raise IncompatibleCheckpointError(f"checkpoint {key} is not a {model.__name__}")


def _optional_model(
    state: Mapping[str, object], key: str, model: type[ModelT]
) -> ModelT | None:
    value = state.get(key)
    if value is None:
        return None
    return _as_model(value, key, model)


# ---------------------------------------------------------------------------
# Spec §8 — request, conversation, semantic
# ---------------------------------------------------------------------------


def validate_request_context(request: RequestContext) -> None:
    """Spec §8.1: persisted request identity, one raw query, unique resources."""

    _require_non_blank(request.request_id, "RequestContext.request_id")
    _require_non_blank(request.thread_id, "RequestContext.thread_id")
    _require_non_blank(request.original_query, "RequestContext.original_query")
    _require_unique(
        (resource.resource_id for resource in request.known_documents),
        "KnownDocumentResource.resource_id",
    )
    for resource in request.known_documents:
        _require_non_blank(resource.resource_id, "KnownDocumentResource.resource_id")


def validate_conversation_context(context: ConversationContext) -> None:
    """Spec §8.2: minimal discourse state."""

    _require_unique(
        (entity.ref_id for entity in context.active_entities), "ActiveEntity.ref_id"
    )
    for entity in context.active_entities:
        _require_non_blank(entity.ref_id, "ActiveEntity.ref_id")
    if context.last_focus is not None:
        _require_non_blank(context.last_focus.ref_id, "EntityReference.ref_id")
    for turn in context.recent_turns:
        _require_non_blank(turn.content, "ConversationTurn.content")


def validate_conversation_snapshot(snapshot: ConversationSnapshot) -> None:
    """Spec §3/§8.2: versioned conversation persistence boundary."""

    _require_contract_version(snapshot.contract_version, "ConversationSnapshot")
    _require_non_blank(snapshot.thread_id, "ConversationSnapshot.thread_id")
    if snapshot.summary_version < 0:
        _fail("ConversationSnapshot.summary_version must be non-negative")
    validate_conversation_context(snapshot.context)


def validate_semantic_context(semantic: SemanticContext) -> None:
    """Spec §8.3: finalized query meaning and document-reference invariants."""

    _require_non_blank(semantic.contextualized_query, "SemanticContext.contextualized_query")
    _require_non_blank(semantic.normalized_query, "SemanticContext.normalized_query")
    _require_unique(
        (reference.ref_id for reference in semantic.document_refs),
        "DocumentReference.ref_id",
    )
    for reference in semantic.document_refs:
        _validate_document_reference(reference)
    _require_unique(
        (ambiguity.ambiguity_id for ambiguity in semantic.blocking_ambiguities),
        "BlockingAmbiguity.ambiguity_id",
    )
    _require_unique(
        (reference.ref_id for reference in semantic.section_refs), "SectionReference.ref_id"
    )
    _require_unique(
        (reference.ref_id for reference in semantic.person_refs), "EntityReference.ref_id"
    )


def _validate_document_reference(reference: DocumentReference) -> None:
    _require_non_blank(reference.ref_id, "DocumentReference.ref_id")
    _require_non_blank(reference.original_span, "DocumentReference.original_span")
    _require_non_blank(reference.normalized_reference, "DocumentReference.normalized_reference")
    status = reference.resolution_status
    if status == "resolved":
        if reference.resolved_document_id is None:
            _fail(
                f"resolved document reference {reference.ref_id} requires a canonical document id"
            )
    elif reference.resolved_document_id is not None:
        _fail(
            f"document reference {reference.ref_id} with status {status!r} must not carry a canonical document id"
        )
    if status == "ambiguous":
        if len(set(reference.candidate_document_ids)) < 2:
            _fail(
                f"ambiguous document reference {reference.ref_id} requires at least two distinct candidates and no canonical id"
            )
    elif status in ("unresolved", "not_found", "error") and reference.candidate_document_ids:
        _fail(
            f"document reference {reference.ref_id} with status {status!r} must not carry candidates"
        )


# ---------------------------------------------------------------------------
# Spec §9 — bindings and revision-requirement relations
# ---------------------------------------------------------------------------


def validate_binding_set(
    bindings: DocumentBindingSet, semantic: SemanticContext | None = None
) -> None:
    """Spec §9/§26: binding ID integrity and revision-requirement semantics.

    ``semantic`` is required whenever a relation exists: a relation is only valid
    for a reference carrying ``CurrentRevisionRequirement``, must resolve both its
    IDs, and must be unique per binding and per reference.
    """

    for binding in bindings.bindings:
        _require_non_blank(binding.binding_id, "ScopedDocument.binding_id")
        _require_non_blank(binding.document_revision, "ScopedDocument.document_revision")
    _require_unique(
        (binding.binding_id for binding in bindings.bindings), "ScopedDocument.binding_id"
    )
    binding_ids = {binding.binding_id for binding in bindings.bindings}

    relations_by_binding: dict[str, BindingRevisionRequirement] = {}
    relations_by_ref: dict[str, BindingRevisionRequirement] = {}
    for relation in bindings.revision_requirement_refs:
        _require_non_blank(relation.binding_id, "BindingRevisionRequirement.binding_id")
        _require_non_blank(relation.ref_id, "BindingRevisionRequirement.ref_id")
        if relation.binding_id not in binding_ids:
            _fail(
                f"revision-requirement relation references unknown binding {relation.binding_id}"
            )
        if relation.binding_id in relations_by_binding:
            _fail(
                f"binding {relation.binding_id} must have exactly one revision-requirement relation"
            )
        relations_by_binding[relation.binding_id] = relation
        if relation.ref_id in relations_by_ref:
            _fail(
                f"reference {relation.ref_id} must have exactly one revision-requirement relation"
            )
        relations_by_ref[relation.ref_id] = relation
        if semantic is None:
            _fail(
                "revision-requirement relations require the semantic context to resolve their ref_id"
            )
        reference = _document_reference_by_id(semantic, relation.ref_id)
        if not isinstance(reference.revision_requirement, CurrentRevisionRequirement):
            _fail(
                f"revision-requirement relation for reference {relation.ref_id} requires a current revision requirement"
            )

    if semantic is None:
        return
    for reference in semantic.document_refs:
        if (
            isinstance(reference.revision_requirement, CurrentRevisionRequirement)
            and reference.resolution_status == "resolved"
            and reference.ref_id not in relations_by_ref
        ):
            _fail(
                f"current-required document reference {reference.ref_id} has no revision-requirement relation"
            )


# ---------------------------------------------------------------------------
# Spec §12 — query analysis and routing
# ---------------------------------------------------------------------------


def validate_query_analysis(analysis: QueryAnalysis) -> None:
    """Spec §12: deterministic-first analysis carries no policy fields."""

    if not analysis.domains:
        _fail("QueryAnalysis.domains must contain at least one domain")
    _require_unique(analysis.domains, "QueryAnalysis.domains")
    for hint in analysis.dependency_hints:
        _require_non_blank(hint.hint_id, "SemanticDependencyHint.hint_id")
        _require_non_blank(hint.description, "SemanticDependencyHint.description")
    _require_unique(
        (hint.hint_id for hint in analysis.dependency_hints), "SemanticDependencyHint.hint_id"
    )

# ---------------------------------------------------------------------------
# Spec §13 — plans, DAGs, criteria, tasks
# ---------------------------------------------------------------------------


def validate_task_plan(plan: TaskPlan, bindings: DocumentBindingSet) -> None:
    """Spec §13/§26: ID, DAG, binding, target, and criteria integrity for a plan."""

    _validate_plan_structure(plan)
    binding_by_id = {binding.binding_id: binding for binding in bindings.bindings}
    for unit in plan.target_units:
        binding = binding_by_id.get(unit.binding_id)
        if binding is None:
            _fail(
                f"target unit {unit.target_id} references unknown binding {unit.binding_id}"
            )
        if binding.role not in ("target", "reference"):
            _fail(
                f"target unit {unit.target_id} cannot bind a {binding.role!r} document: target/reference roles are required"
            )
    referenced = {
        target_id for task in plan.tasks for target_id in _input_target_ids(task.input)
    }
    for unit in plan.target_units:
        if unit.target_id not in referenced:
            _fail(f"target unit {unit.target_id} is not referenced by any task input")


def validate_fast_plan(plan: TaskPlan, bindings: DocumentBindingSet) -> None:
    """Spec §12/§26: a fast route checkpoints one deterministic initial task."""

    if len(plan.tasks) != 1:
        _fail("fast plan must checkpoint exactly a single task")
    task = plan.tasks[0]
    if task.depends_on:
        _fail("fast plan task must be dependency-free")
    if not isinstance(task.origin, InitialTaskOrigin):
        _fail("fast plan task must have an initial origin (no replan)")
    if bool(plan.target_units) != bool(_input_target_ids(task.input)):
        _fail(
            "fast plan target_units must match the targets referenced by its single task input"
        )
    validate_task_plan(plan, bindings)


def validate_replan(
    current: TaskPlan,
    proposed: TaskPlan,
    outcomes: tuple[TaskExecutionSummary, ...],
    policy: DiscoveryPolicy,
    budget: ResearchBudgetView,
) -> TaskPlan:
    """Spec §16/§17/§26: bounded, append-only replanning.

    Capability-catalog membership is a runtime check (the gateway validates the
    proposed capability against the current request-scoped registry); everything
    deterministic about the proposal is enforced here.
    """

    _validate_plan_structure(proposed)
    if proposed.plan_id != current.plan_id:
        _fail("replan must keep the plan_id of the current plan")
    if proposed.goal != current.goal:
        _fail("replan must keep the goal of the current plan")
    if (
        len(proposed.tasks) < len(current.tasks)
        or proposed.tasks[: len(current.tasks)] != current.tasks
    ):
        _fail("replan must be append-only and keep completed tasks unchanged")
    if proposed.target_units != current.target_units:
        _fail("replan must not add or replace target units")
    new_tasks = proposed.tasks[len(current.tasks) :]
    if not new_tasks:
        _fail("replan must append at least one task")
    for task in new_tasks:
        if not isinstance(task.origin, ReplanTaskOrigin):
            _fail(f"task {task.task_id} appended by a replan must have a ReplanTaskOrigin")
    validate_task_outcomes(outcomes, current)
    if budget.max_replans_remaining < 1:
        _fail("no replan budget remaining")
    if len(new_tasks) > budget.max_tasks_remaining:
        _fail(
            f"replan proposes {len(new_tasks)} task(s) but the task budget allows {budget.max_tasks_remaining}"
        )
    if budget.max_parallel_branches < 1:
        _fail("ResearchBudgetView.max_parallel_branches must be at least 1")
    _validate_replan_discovery(new_tasks, policy)
    return proposed


def validate_task_outcomes(
    outcomes: tuple[TaskExecutionSummary, ...], plan: TaskPlan
) -> None:
    """Spec §16/§26: one summary per attempted task, typed failure preserved."""

    task_ids = {task.task_id for task in plan.tasks}
    seen: set[str] = set()
    for outcome in outcomes:
        _require_non_blank(outcome.task_id, "TaskExecutionSummary.task_id")
        if outcome.task_id not in task_ids:
            _fail(f"task outcome references unknown task {outcome.task_id}")
        if outcome.task_id in seen:
            _fail(f"duplicate TaskExecutionSummary for task {outcome.task_id}")
        seen.add(outcome.task_id)
        if outcome.status == "error":
            if outcome.error_code is None:
                _fail(f"task outcome {outcome.task_id} has status error but no error_code")
        elif outcome.status != "denied" and outcome.error_code is not None:
            _fail(
                f"task outcome {outcome.task_id} carries error_code {outcome.error_code!r} for status {outcome.status!r}"
            )
        if (
            outcome.status == "denied"
            and outcome.error_code is not None
            and outcome.error_code != "PERMISSION_DENIED"
        ):
            _fail(
                f"task outcome {outcome.task_id} is denied but reports {outcome.error_code!r}"
            )


def _validate_plan_structure(plan: TaskPlan) -> None:
    _require_non_blank(plan.plan_id, "TaskPlan.plan_id")
    _require_non_blank(plan.goal, "TaskPlan.goal")
    for unit in plan.target_units:
        _require_non_blank(unit.target_id, "TargetUnit.target_id")
        _require_non_blank(unit.binding_id, "TargetUnit.binding_id")
    _require_unique((unit.target_id for unit in plan.target_units), "TargetUnit.target_id")
    for unit in plan.target_units:
        _validate_completion_criteria(unit)

    for task in plan.tasks:
        _require_non_blank(task.task_id, "TaskSpec.task_id")
        _require_non_blank(task.task_objective, "TaskSpec.task_objective")
    _require_unique((task.task_id for task in plan.tasks), "TaskSpec.task_id")

    target_ids = {unit.target_id for unit in plan.target_units}
    task_ids = {task.task_id for task in plan.tasks}
    for task in plan.tasks:
        if task.capability != task.input.kind:
            _fail(
                f"task {task.task_id} capability {task.capability!r} does not match its input kind {task.input.kind!r}"
            )
        _require_unique(task.depends_on, f"TaskSpec {task.task_id} depends_on")
        if task.task_id in task.depends_on:
            _fail(f"task {task.task_id} cannot depend on itself")
        for dependency in task.depends_on:
            if dependency not in task_ids:
                _fail(f"task {task.task_id} depends_on unknown task {dependency}")
        for target_id in _input_target_ids(task.input):
            if target_id not in target_ids:
                _fail(f"task {task.task_id} references unknown target {target_id}")
        _validate_task_origin(task, task_ids)
    _require_acyclic(plan.tasks)


def _validate_completion_criteria(unit: TargetUnit) -> None:
    coverage = [
        criterion
        for criterion in unit.completion_criteria
        if isinstance(criterion, CoverageCriterion)
    ]
    if len(coverage) > 1:
        _fail(f"target unit {unit.target_id} declares more than one coverage criterion")
    for criterion in coverage:
        if criterion.minimum_status == "read_partial":
            _require_non_blank(
                criterion.allow_partial_reason,
                f"TargetUnit {unit.target_id} CoverageCriterion.allow_partial_reason",
            )
        elif criterion.allow_partial_reason is not None:
            _fail(
                f"target unit {unit.target_id} declares allow_partial_reason for read_complete coverage"
            )
    semantic = [
        criterion
        for criterion in unit.completion_criteria
        if isinstance(criterion, SemanticCriterion)
    ]
    for criterion in semantic:
        _require_non_blank(
            criterion.criterion_id, f"TargetUnit {unit.target_id} SemanticCriterion.criterion_id"
        )
        _require_non_blank(
            criterion.description, f"TargetUnit {unit.target_id} SemanticCriterion.description"
        )
    _require_unique(
        (criterion.criterion_id for criterion in semantic),
        f"TargetUnit {unit.target_id} SemanticCriterion.criterion_id",
    )


def _validate_task_origin(task: TaskSpec, task_ids: set[str]) -> None:
    origin = task.origin
    if isinstance(origin, InitialTaskOrigin):
        return
    _require_non_blank(origin.reason, f"ReplanTaskOrigin.reason for task {task.task_id}")
    _require_unique(origin.task_ids, "ReplanTaskOrigin.task_ids")
    _require_unique(origin.evidence_use_ids, "ReplanTaskOrigin.evidence_use_ids")
    if task.task_id in origin.task_ids:
        _fail(f"task {task.task_id} origin cannot reference its own task id")
    for reference in origin.task_ids:
        if reference not in task_ids:
            _fail(f"task {task.task_id} origin references unknown task {reference}")


def _require_acyclic(tasks: tuple[TaskSpec, ...]) -> None:
    remaining = {task.task_id: set(task.depends_on) for task in tasks}
    while remaining:
        ready = {task_id for task_id, deps in remaining.items() if not deps}
        if not ready:
            _fail(f"task dependency cycle detected among {sorted(remaining)}")
        for task_id in ready:
            del remaining[task_id]
        for deps in remaining.values():
            deps -= ready


def _validate_replan_discovery(
    new_tasks: tuple[TaskSpec, ...], policy: DiscoveryPolicy
) -> None:
    discovery_tasks = [task for task in new_tasks if task.capability == "document.search"]
    if not discovery_tasks:
        return
    if not (policy.allow_reference_discovery or policy.allow_supporting_discovery):
        _fail("replan proposes document.search but discovery is disabled by policy")
    if policy.max_discovered_documents < 1:
        _fail("replan proposes document.search but max_discovered_documents is 0")


def validate_research_planning_input(planning_input: ResearchPlanningInput) -> None:
    """Spec §16: initial planning and replanning receive different execution facts."""

    validate_semantic_context(planning_input.semantic)
    validate_binding_set(planning_input.bindings, planning_input.semantic)
    validate_query_analysis(planning_input.query_analysis)
    if not planning_input.capability_catalog:
        _fail("ResearchPlanningInput.capability_catalog must not be empty")
    _require_unique(
        (descriptor.name for descriptor in planning_input.capability_catalog),
        "CapabilityDescriptor.name",
    )
    if planning_input.budget.max_tasks_remaining < 0:
        _fail("ResearchBudgetView.max_tasks_remaining must be non-negative")
    if planning_input.budget.max_replans_remaining < 0:
        _fail("ResearchBudgetView.max_replans_remaining must be non-negative")
    if planning_input.budget.max_parallel_branches < 1:
        _fail("ResearchBudgetView.max_parallel_branches must be at least 1")
    if planning_input.current_plan is None:
        if (
            planning_input.task_outcomes
            or planning_input.prior_evidence_uses
            or planning_input.prior_evaluation is not None
        ):
            _fail(
                "initial planning receives no task outcomes, prior evidence uses, or prior evaluation"
            )
        return
    if planning_input.prior_evaluation is None:
        _fail("replanning requires the latest prior evaluation")
    validate_task_plan(planning_input.current_plan, planning_input.bindings)
    if planning_input.task_outcomes:
        validate_task_outcomes(planning_input.task_outcomes, planning_input.current_plan)


# ---------------------------------------------------------------------------
# Spec §13.3 / §14 — capability results and coverage observations
# ---------------------------------------------------------------------------


def validate_agent_result(result: AgentResult, plan: TaskPlan) -> None:
    """Spec §13.3/§14/§26: checkpointed execution facts only.

    Read capabilities produce coverage observations and search never does, so a
    capability result cannot smuggle read coverage past the evaluator.
    """

    task = _task_by_id(plan, result.task_id)
    if result.data is not None and result.data.kind != task.capability:
        _fail(
            f"task result {result.task_id} carries data of kind {result.data.kind!r} for capability {task.capability!r}"
        )
    if result.status in ("denied", "error") and result.error is None:
        _fail(f"task result {result.task_id} has status {result.status!r} but no AgentError")
    if result.status not in ("denied", "error", "needs_input") and result.error is not None:
        _fail(
            f"task result {result.task_id} carries an AgentError for status {result.status!r}"
        )
    for observation in result.coverage_observations:
        if task.capability not in _READ_CAPABILITIES:
            _fail(
                f"capability {task.capability!r} cannot report read coverage observations"
            )
        _target_by_id(plan, observation.target_id)
    _require_unique(
        (ref.use_id for ref in result.evidence_uses), "AgentResult.evidence_uses.use_id"
    )
    if (
        task.capability in _READ_CAPABILITIES
        and result.status in ("success", "partial")
        and not result.coverage_observations
    ):
        _fail(
            f"read capability {task.capability!r} returned {result.status!r} without coverage observations"
        )


# ---------------------------------------------------------------------------
# Spec §15 — evidence identity and use
# ---------------------------------------------------------------------------


def validate_evidence_record(record: EvidenceRecord) -> None:
    """Spec §15.1/§15.3: evidence identity is complete and source-typed."""

    _require_non_blank(record.content, "EvidenceRecord.content")
    _require_non_blank(record.content_hash, "EvidenceRecord.content_hash")
    _require_non_blank(record.provenance.fetcher, "Provenance.fetcher")
    source = record.source
    if isinstance(source, DocumentSourceIdentity):
        _require_non_blank(
            source.document_revision, "DocumentSourceIdentity.document_revision"
        )
    elif isinstance(source, PeopleSourceIdentity):
        _require_non_blank(source.record_id, "PeopleSourceIdentity.record_id")
    elif isinstance(source, KnowledgeGraphSourceIdentity):
        _require_non_blank(
            source.entity_or_relation_id, "KnowledgeGraphSourceIdentity.entity_or_relation_id"
        )
    elif isinstance(source, MemorySourceIdentity):
        _require_non_blank(source.memory_id, "MemorySourceIdentity.memory_id")
    elif isinstance(source, DerivedSourceIdentity):
        if not source.source_evidence_ids:
            _fail("DerivedSourceIdentity.source_evidence_ids must not be empty")
        _require_unique(
            source.source_evidence_ids, "DerivedSourceIdentity.source_evidence_ids"
        )


def validate_evidence_store_row(row: EvidenceStoreRow) -> None:
    """Spec §15.3: storage policy lives only on the persisted store row."""

    _require_contract_version(row.contract_version, "EvidenceStoreRow")
    validate_evidence_record(row.record)


def validate_evidence_use(use: EvidenceUse) -> None:
    """Spec §15.2: purpose/target compatibility for one run-local use."""

    _require_non_blank(use.task_id, "EvidenceUse.task_id")
    if use.purpose == "coverage" and use.target_id is None:
        _fail("a coverage evidence use requires a target_id")
    if use.purpose == "discovery" and use.target_id is not None:
        _fail("a discovery evidence use must be targetless")
    if use.target_id is not None:
        _require_non_blank(use.target_id, "EvidenceUse.target_id")


def validate_evidence_use_resolution(use: EvidenceUse, plan: TaskPlan) -> None:
    """Spec §15.2: every use resolves to one current checkpointed task/target."""

    validate_evidence_use(use)
    task = _task_by_id(plan, use.task_id)
    if use.target_id is not None:
        _target_by_id(plan, use.target_id)
        return
    if use.purpose == "supporting" and _input_target_ids(task.input):
        _fail(
            f"a targetless supporting evidence use requires a validated targetless task; {task.task_id} reads targets"
        )


def validate_target_use_revision(
    use: EvidenceUse,
    plan: TaskPlan,
    bindings: DocumentBindingSet,
    record: EvidenceRecord,
) -> None:
    """Spec §14/§15.2/§26: a target-bound use must match the pinned revision.

    A revision mismatch cannot complete coverage, derived evidence never creates
    authoritative read coverage, and a target resolves its document identity only
    through its binding.
    """

    validate_evidence_use_resolution(use, plan)
    if use.target_id is None:
        return
    if isinstance(record.source, DerivedSourceIdentity):
        _fail("derived evidence never creates authoritative read coverage or a target-bound use")
    if not isinstance(record.source, DocumentSourceIdentity):
        _fail(
            f"a target-bound use requires a document evidence record, got {record.source.kind!r}"
        )
    unit = _target_by_id(plan, use.target_id)
    binding = _binding_by_id(bindings, unit.binding_id)
    if (
        record.source.document_id != binding.document_id
        or record.source.document_revision != binding.document_revision
    ):
        _fail(
            f"evidence revision {record.source.document_revision!r} does not match the revision "
            f"{binding.document_revision!r} pinned by binding {binding.binding_id}"
        )


def validate_derived_evidence_faithfulness(
    record: EvidenceRecord,
    source_records: tuple[EvidenceRecord, ...],
    *,
    faithfulness_validated: bool,
) -> None:
    """Spec §15.3/§26: derived evidence needs validated recursive sources.

    Validation state is Evidence Store metadata, so it is supplied by the caller
    rather than stored on ``EvidenceRecord``.
    """

    if not isinstance(record.source, DerivedSourceIdentity):
        _fail("record is not derived evidence")
    if not faithfulness_validated:
        _fail(
            "derived evidence must pass recursive source-faithfulness validation before it is synthesis-eligible"
        )
    resolved = {source.evidence_id for source in source_records}
    unresolved = [
        evidence_id
        for evidence_id in record.source.source_evidence_ids
        if evidence_id not in resolved
    ]
    if unresolved:
        _fail(f"derived evidence has unresolved recursive source(s): {unresolved}")


# ---------------------------------------------------------------------------
# Spec §18 — evaluation
# ---------------------------------------------------------------------------


def validate_evidence_evaluation(evaluation: EvidenceEvaluation, plan: TaskPlan) -> None:
    """Spec §18/§26: evaluation references authoritative planned targets/criteria."""

    _require_unique(
        (item.target_id for item in evaluation.coverage.items), "CoverageItem.target_id"
    )
    for item in evaluation.coverage.items:
        _require_non_blank(item.target_id, "CoverageItem.target_id")
        _target_by_id(plan, item.target_id)
    for requirement in evaluation.missing:
        unit = _target_by_id(plan, requirement.target_id)
        _require_non_blank(requirement.description, "MissingRequirement.description")
        semantic_ids = {
            criterion.criterion_id
            for criterion in unit.completion_criteria
            if isinstance(criterion, SemanticCriterion)
        }
        has_coverage = any(
            isinstance(criterion, CoverageCriterion)
            for criterion in unit.completion_criteria
        )
        if requirement.criterion_kind == "semantic":
            _require_non_blank(
                requirement.semantic_criterion_id, "MissingRequirement.semantic_criterion_id"
            )
            if requirement.semantic_criterion_id not in semantic_ids:
                _fail(
                    f"missing requirement references unknown semantic criterion "
                    f"{requirement.semantic_criterion_id} in target {requirement.target_id}"
                )
        else:
            if requirement.semantic_criterion_id is not None:
                _fail("a coverage missing requirement must not carry semantic_criterion_id")
            if not has_coverage:
                _fail(
                    f"missing coverage requirement references target {requirement.target_id} without a coverage criterion"
                )
    if evaluation.status == "sufficient" and evaluation.missing:
        _fail("a sufficient evaluation cannot report missing requirements")
    for contradiction in evaluation.contradictions:
        _require_non_blank(contradiction.contradiction_id, "Contradiction.contradiction_id")
        if not contradiction.evidence_use_ids:
            _fail(
                f"contradiction {contradiction.contradiction_id} must reference admitted evidence uses"
            )
        _require_unique(contradiction.evidence_use_ids, "Contradiction.evidence_use_ids")


# ---------------------------------------------------------------------------
# Spec §19 — synthesis and grounding
# ---------------------------------------------------------------------------


def validate_synthesis_input(synthesis_input: SynthesisInput) -> None:
    """Spec §19.1/§26: factual synthesis requires a sufficient evaluation."""

    validate_semantic_context(synthesis_input.semantic)
    if synthesis_input.evaluation.status != "sufficient":
        _fail(
            f"factual synthesis requires a sufficient EvidenceEvaluation, got {synthesis_input.evaluation.status!r}"
        )
    _require_unique(
        (ref.use_id for ref in synthesis_input.evidence_uses), "SynthesisInput.evidence_uses.use_id"
    )


def validate_synthesis_use(use: EvidenceUse, plan: TaskPlan) -> None:
    """Spec §15.2/§19.1: discovery uses cannot synthesize or support a claim."""

    if use.purpose == "discovery":
        _fail("a discovery evidence use is not synthesis-eligible")
    validate_evidence_use_resolution(use, plan)


def validate_admitted_claim_use_ids(
    claim_use_ids: tuple[UUID, ...], admitted_use_ids: frozenset[UUID]
) -> None:
    """Spec §19.1: claim use IDs are a unique subset of the admitted use set."""

    if not claim_use_ids:
        _fail("claim has empty evidence_use_ids; every claim needs at least one admitted use")
    _require_unique(claim_use_ids, "AnswerClaim.evidence_use_ids")
    for use_id in claim_use_ids:
        if use_id not in admitted_use_ids:
            _fail(
                f"unsupported claim evidence_use_id {use_id}; it is not in the admitted synthesis set"
            )


def validate_answer_draft(draft: AnswerDraft, admitted_use_ids: frozenset[UUID]) -> None:
    """Spec §19.2/§26: claims are unique, non-empty, and use admitted IDs only."""

    _require_non_blank(draft.content, "AnswerDraft.content")
    if not draft.claims:
        _fail("AnswerDraft must declare at least one claim for its factual content")
    _require_unique((claim.claim_id for claim in draft.claims), "AnswerClaim.claim_id")
    for claim in draft.claims:
        _require_non_blank(claim.claim_id, "AnswerClaim.claim_id")
        _require_non_blank(claim.text, "AnswerClaim.text")
        validate_admitted_claim_use_ids(claim.evidence_use_ids, admitted_use_ids)


def validate_final_response(response: FinalResponse) -> None:
    """Spec §19.2: citation identity is deterministic presentation output."""

    _require_contract_version(response.contract_version, "FinalResponse")
    _require_unique(
        (citation.citation_id for citation in response.citations),
        "RenderedCitation.citation_id",
    )
    for citation in response.citations:
        _require_non_blank(citation.citation_id, "RenderedCitation.citation_id")
        _require_non_blank(citation.label, "RenderedCitation.label")


# ---------------------------------------------------------------------------
# Spec §20 — clarification
# ---------------------------------------------------------------------------


def validate_clarification_request(
    request: ClarificationRequest, semantic: SemanticContext
) -> None:
    """Spec §20: the persisted question references only known, asked-about refs."""

    _require_contract_version(request.contract_version, "ClarificationRequest")
    _require_non_blank(request.question, "ClarificationRequest.question")
    known_ref_ids = {reference.ref_id for reference in semantic.document_refs}
    for ref_id in request.unresolved_ref_ids:
        if ref_id not in known_ref_ids:
            _fail(f"clarification references unknown document ref {ref_id}")
    _require_unique(
        (candidate.candidate_id for candidate in request.candidates),
        "DocumentCandidate.candidate_id",
    )
    _require_unique(
        (candidate.ordinal for candidate in request.candidates), "DocumentCandidate.ordinal"
    )
    for candidate in request.candidates:
        _require_non_blank(candidate.candidate_id, "DocumentCandidate.candidate_id")
        _require_non_blank(candidate.ref_id, "DocumentCandidate.ref_id")
        _require_non_blank(candidate.label, "DocumentCandidate.label")
        if candidate.ordinal < 0:
            _fail("DocumentCandidate.ordinal must be non-negative")
        if candidate.ref_id not in known_ref_ids:
            _fail(f"clarification candidate references unknown document ref {candidate.ref_id}")
        if candidate.ref_id not in request.unresolved_ref_ids:
            _fail(
                f"clarification candidate for ref {candidate.ref_id} is not in unresolved_ref_ids"
            )


def validate_clarification_resolution(
    request: ClarificationRequest, resolution: ClarificationResolution
) -> None:
    """Spec §20: resume validates expiry-adjacent identity and candidate membership."""

    _require_contract_version(resolution.contract_version, "ClarificationResolution")
    if resolution.clarification_id != request.clarification_id:
        _fail(
            f"clarification_id mismatch: resolution closes {resolution.clarification_id!r}, "
            f"expected {request.clarification_id!r}"
        )
    if resolution.selected_candidate_id is None:
        return
    candidate_ids = {candidate.candidate_id for candidate in request.candidates}
    if resolution.selected_candidate_id not in candidate_ids:
        _fail(
            f"selected candidate {resolution.selected_candidate_id!r} is not part of clarification "
            f"{request.clarification_id!r}"
        )


# ---------------------------------------------------------------------------
# Spec §7 / §21 / §25 — checkpoint aggregates
# ---------------------------------------------------------------------------


def validate_checkpoint_payload(payload: Mapping[str, object]) -> None:
    """Spec §3/§25: reject incompatible checkpoints instead of migrating them.

    This is the version/shape gate applied before any loaded checkpoint is used.
    Nested persisted envelopes inside the aggregate must declare the same version;
    a missing or foreign version is rejected outright.
    """

    declared = payload.get("contract_version")
    if declared != CONTRACT_VERSION:
        raise IncompatibleCheckpointError(
            f"checkpoint payload declares contract_version {declared!r}, expected {CONTRACT_VERSION!r}"
        )
    missing = [key for key in _CHECKPOINT_REQUIRED_KEYS if key not in payload]
    if missing:
        raise IncompatibleCheckpointError(
            f"checkpoint payload is missing required key(s): {', '.join(missing)}"
        )
    for slot in ("request", "clarification", "final_response"):
        value = payload.get(slot)
        if value is not None:
            _require_declared_checkpoint_version(value, boundary=slot)
    execution = payload.get("execution")
    if execution is None:
        return
    if isinstance(execution, Mapping):
        plan = execution.get("plan")
        if plan is not None:
            _require_declared_checkpoint_version(plan, boundary="execution.plan")
        results = execution.get("task_results")
        if results is None:
            return
        if not isinstance(results, (list, tuple)):
            raise IncompatibleCheckpointError("execution.task_results must be a sequence")
        for index, result in enumerate(results):
            _require_declared_checkpoint_version(
                result, boundary=f"execution.task_results[{index}]"
            )
        return
    plan = getattr(execution, "plan", None)
    if plan is not None:
        _require_declared_checkpoint_version(plan, boundary="execution.plan")
    for index, result in enumerate(getattr(execution, "task_results", ())):
        _require_declared_checkpoint_version(result, boundary=f"execution.task_results[{index}]")


def validate_supervisor_state(state: SupervisorV2State) -> None:
    """Spec §7/§21/§26: validate the aggregate and all its cross-references."""

    declared = state.get("contract_version")
    if declared != CONTRACT_VERSION:
        raise IncompatibleCheckpointError(
            f"SupervisorV2State declares contract_version {declared!r}, expected {CONTRACT_VERSION!r}"
        )
    request = _as_model(state["request"], "request", RequestContext)
    conversation = _as_model(state["conversation"], "conversation", ConversationContext)
    semantic = _as_model(state["semantic"], "semantic", SemanticContext)
    bindings = _as_model(state["bindings"], "bindings", DocumentBindingSet)
    execution = _as_model(state["execution"], "execution", ExecutionState)
    query_analysis = _optional_model(state, "query_analysis", QueryAnalysis)
    route_decision = _optional_model(state, "route_decision", RouteDecision)
    clarification = _optional_model(state, "clarification", ClarificationRequest)
    final_response = _optional_model(state, "final_response", FinalResponse)

    validate_request_context(request)
    validate_conversation_context(conversation)
    validate_semantic_context(semantic)
    validate_binding_set(bindings, semantic)
    if query_analysis is not None:
        validate_query_analysis(query_analysis)
    if route_decision is not None:
        if route_decision.route == "clarify" and clarification is None:
            _fail("clarify route requires a persisted ClarificationRequest")
        if route_decision.route == "direct" and execution.plan is not None:
            _fail(
                "direct route executes no capability and must keep ExecutionState.plan=None"
            )
    if clarification is not None:
        validate_clarification_request(clarification, semantic)
    if final_response is not None:
        validate_final_response(final_response)
    _validate_execution_state(execution, bindings)


def _validate_execution_state(execution: ExecutionState, bindings: DocumentBindingSet) -> None:
    if execution.plan is None:
        if execution.task_results:
            _fail("task results require a checkpointed TaskPlan")
        if execution.evidence_evaluation is not None:
            _fail("evidence evaluation requires a checkpointed TaskPlan")
        return
    validate_task_plan(execution.plan, bindings)
    for result in execution.task_results:
        validate_agent_result(result, execution.plan)
    if execution.evidence_evaluation is not None:
        validate_evidence_evaluation(execution.evidence_evaluation, execution.plan)


def _require_contract_version(declared: object, boundary: str) -> None:
    if declared != CONTRACT_VERSION:
        raise IncompatibleCheckpointError(
            f"{boundary} declares contract_version {declared!r}, expected {CONTRACT_VERSION!r}"
        )


def _require_declared_checkpoint_version(value: object, *, boundary: str) -> None:
    if isinstance(value, ContractModel):
        declared = getattr(value, "contract_version", None)
    elif isinstance(value, Mapping):
        declared = value.get("contract_version")
    else:
        raise IncompatibleCheckpointError(
            f"checkpoint {boundary} is not a persisted v2 envelope"
        )
    if declared != CONTRACT_VERSION:
        raise IncompatibleCheckpointError(
            f"checkpoint {boundary} declares contract_version {declared!r}, expected {CONTRACT_VERSION!r}"
        )

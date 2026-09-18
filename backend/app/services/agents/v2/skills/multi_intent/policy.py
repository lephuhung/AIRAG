"""Multi-intent skill policy: parallel evidence tasks from IntentAnalysis.

Framework-neutral source of truth (multi-intent routing spec §33.4): given an
ephemeral ``ResearchPlanningInput`` carrying a checkpointed
:class:`IntentAnalysis`, the policy proposes the initial ``TaskPlan`` for a
``multi_intent`` request — one **targetless** evidence task per independent
evidence-bearing intent, ``depends_on=()`` (parallel topology; the shared
scheduler still dispatches sequentially).

Intent → capability resolution is registry-derived (``INTENT_REGISTRY``),
never model-trusted:

- ``people_lookup``/``people_search`` → ``people.lookup``
- ``document_lookup``/``document_search``/``list_documents`` →
  ``document.retrieve`` (empty ``target_ids`` — evidence, not discovery
  candidates; ``document.search`` stays materializer/discovery owned)
- ``kg_lookup`` → ``knowledge_graph.query``
- ``memory_lookup``/``personal`` → ``memory.lookup``

Intents whose registry ``capability`` is ``None`` (``evaluate_compliance``,
``compare_documents``, ``summarize``, ``cross_domain_research``, ``write``,
``direct_answer``, ``greeting``) emit **no task** — the pipeline
``evaluate``/``synthesize`` nodes own them, and ``evaluate`` stays a graph
node rather than a plan task.

Coverage/refusal semantics (a covering skill's refusal is final):

- **Covered:** every intent is independent (``depends_on == ()``), every
  intent name resolves in the registry, every capability-bearing intent maps
  to a targetless-emittable capability, and at least one such intent exists.
- **Refuse (final when covered):** missing ``IntentAnalysis``, empty intents,
  any ``depends_on`` (sequential semantics belong to the governed model
  planner and the people→document materializer), an unknown intent name, a
  capability that cannot run targetless, a capability missing from the
  request-scoped catalog, zero emitted tasks, or more tasks than
  ``budget.max_tasks_remaining``.
- **Uncovered:** dependent intents, unknown names, or intents needing
  target-bearing capabilities stay with the governed model path (or the
  typed unavailable boundary when unwired) — never the unscoped-retrieve
  fallback, which would silently drop intents and re-create the
  compound-request defect.

It never executes a capability, never checkpoints, and never fabricates a
person scalar: the returned plan still passes through ``validate_task_plan``
in the subgraph ``validate_checkpoint`` node, the supervisor saver, and the
shared scheduler.
"""
from __future__ import annotations

from ...contracts.capability import (
    CapabilityInput,
    DocumentRetrieveInput,
    KnowledgeGraphInput,
    MemoryLookupInput,
    PeopleLookupInput,
)
from ...contracts.planning import (
    InitialTaskOrigin,
    ResearchPlanningInput,
    TaskPlan,
    TaskSpec,
)
from ...contracts.validation import ContractValidationError, validate_task_plan
from ...semantic.intent_registry import INTENT_REGISTRY

__all__ = [
    "MULTI_INTENT_WORK_TYPE",
    "build_multi_intent_plan",
    "covers_input",
    "supports_work_type",
]

#: The only work type this skill plans. Everything else fails closed.
MULTI_INTENT_WORK_TYPE = "multi_intent"

#: Governed capabilities the skill can emit as targetless evidence tasks.
#: Target-bearing capabilities (``section.read``, ``document.read``) and
#: discovery-owned ``document.search`` are deliberately absent — intents
#: mapping to them stay uncovered for the governed model path, which can
#: bind targets the deterministic skill cannot.
_TARGETLESS_CAPABILITIES = frozenset(
    {
        "people.lookup",
        "document.retrieve",
        "knowledge_graph.query",
        "memory.lookup",
    }
)


def supports_work_type(work_type: str) -> bool:
    """True only for ``multi_intent``; every other work type is out."""
    return work_type == MULTI_INTENT_WORK_TYPE


def _intent_capability(name: str) -> str | None:
    """Registry capability for an intent name; raises on an unknown name."""
    spec = INTENT_REGISTRY.get(name)
    if spec is None:
        raise ContractValidationError(
            f"multi_intent skill cannot plan unknown intent {name!r}; "
            "refusing to fabricate an execution mapping"
        )
    return spec.get("capability")


def covers_input(planning_input: ResearchPlanningInput) -> bool:
    """True when the deterministic skill owns this input (planner seam).

    Ownership is work type plus intake: a populated ``IntentAnalysis`` whose
    intents are all independent, all registry-resolvable, and include at
    least one intent the skill can serve targetlessly. A covered input whose
    build still refuses (missing catalog capability, over task budget) is a
    deliberate final refusal — the model path must not second-guess it.
    Dependent intents, unknown names, and intents needing target-bearing
    capabilities stay uncovered so the governed model path still owns them.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        return False
    analysis = planning_input.intent_analysis
    if analysis is None or not analysis.intents:
        return False
    saw_evidence = False
    for intent in analysis.intents:
        if intent.depends_on:
            return False
        spec = INTENT_REGISTRY.get(intent.name)
        if spec is None:
            return False
        capability = spec.get("capability")
        if capability is None:
            continue
        if capability not in _TARGETLESS_CAPABILITIES:
            return False
        saw_evidence = True
    return saw_evidence


def _task_input(capability: str, query: str) -> CapabilityInput:
    """Construct the typed targetless input server-side (no model scalars)."""
    if capability == "people.lookup":
        return PeopleLookupInput(kind="people.lookup", query=query)
    if capability == "document.retrieve":
        return DocumentRetrieveInput(kind="document.retrieve", query=query)
    if capability == "knowledge_graph.query":
        return KnowledgeGraphInput(kind="knowledge_graph.query", query=query)
    if capability == "memory.lookup":
        return MemoryLookupInput(kind="memory.lookup", query=query)
    raise ContractValidationError(
        f"multi_intent skill cannot emit capability {capability!r} "
        "targetlessly; refusing an ungovernable task"
    )


def build_multi_intent_plan(planning_input: ResearchPlanningInput) -> TaskPlan:
    """Propose the initial governed multi-intent evidence plan.

    Raises :class:`ContractValidationError` for any out-of-scope input —
    unsupported work type, missing/empty ``IntentAnalysis``, any dependent
    intent, an unknown intent name, a capability the skill cannot emit
    targetlessly or that is missing from the request-scoped catalog, zero
    evidence-bearing intents, a task count over the remaining budget, or a
    plan that fails frozen validation. The caller turns that into the
    governed model path (when the input is uncovered and a planner is
    wired) or the typed unavailable boundary — never a partial plan, never
    the unscoped-retrieve fallback, and never a dispatch.
    """
    if not supports_work_type(planning_input.query_analysis.work_type):
        raise ContractValidationError(
            "multi_intent skill cannot plan work type "
            f"{planning_input.query_analysis.work_type!r}; out of skill scope"
        )
    analysis = planning_input.intent_analysis
    if analysis is None or not analysis.intents:
        raise ContractValidationError(
            "multi_intent planning requires a populated IntentAnalysis; "
            "refusing to plan without checkpointed intents"
        )
    catalog = {entry.name for entry in planning_input.capability_catalog}
    query = planning_input.semantic.contextualized_query
    tasks: list[TaskSpec] = []
    for intent in analysis.intents:
        if intent.depends_on:
            raise ContractValidationError(
                f"multi_intent intent {intent.intent_id!r} declares "
                "depends_on; sequential semantics belong to the governed "
                "model planner and the people-document materializer"
            )
        capability = _intent_capability(intent.name)
        if capability is None:
            # Pipeline-owned intent (evaluate/compare/summarize/conversation):
            # the evaluate/synthesize graph nodes own it — no plan task.
            continue
        if capability not in _TARGETLESS_CAPABILITIES:
            raise ContractValidationError(
                f"multi_intent intent {intent.name!r} maps to capability "
                f"{capability!r}, which needs bound targets; refusing to "
                "fabricate target units"
            )
        if capability not in catalog:
            raise ContractValidationError(
                f"multi_intent intent {intent.name!r} needs capability "
                f"{capability!r}, absent from the request-scoped catalog; "
                "refusing to plan an undispatchable task"
            )
        tasks.append(
            TaskSpec(
                task_id=f"T{len(tasks) + 1}",
                capability=capability,
                task_objective=(
                    f"Gather evidence for intent {intent.name} "
                    f"({intent.intent_id})"
                ),
                input=_task_input(capability, query),
                depends_on=(),
                origin=InitialTaskOrigin(kind="initial"),
            )
        )
    if not tasks:
        raise ContractValidationError(
            "multi_intent analysis carries no evidence-bearing intent; "
            "pipeline-owned intents emit no plan task"
        )
    if len(tasks) > planning_input.budget.max_tasks_remaining:
        raise ContractValidationError(
            f"multi_intent needs {len(tasks)} evidence task(s) but the task "
            f"budget allows {planning_input.budget.max_tasks_remaining}; "
            "refusing a plan that cannot fit the remaining budget"
        )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="multi-intent-" + "-".join(
            intent.intent_id for intent in analysis.intents
        ),
        goal=planning_input.semantic.contextualized_query,
        target_units=(),
        tasks=tuple(tasks),
    )
    validate_task_plan(
        plan,
        planning_input.bindings,
        target_selection=planning_input.target_selection,
        discovery_checkpoint=planning_input.discovery_checkpoint,
    )
    return plan

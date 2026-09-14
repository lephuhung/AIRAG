"""Governed Adaptive Planner service (Phase 5, Task 10).

``AdaptivePlanner`` is a runtime-only, request-scoped service (wired onto
``RuntimeServices.adaptive_planner``): it owns NO checkpointed state, dispatches
NO capabilities, and resolves NO document identity. It proposes an initial
:class:`TaskPlan` that the governed entry (``validate_checkpoint_node``)
still validates, leases, and checkpoints before the shared ``TaskScheduler``
sees it.

Ordering: deterministic skill policies win whenever they cover the work type
(compare / retrieve / summarize / multi-goal / cross-domain / evaluate) — a
covering skill's refusal is final and never falls through to the model; the
model path runs only for below-intake work no skill covers. The model
returns an ordered step
list over server-issued binding IDs; every ``TaskSpec`` input is constructed
server-side from typed constructors, so a model cannot smuggle scalars,
identity, or authorization. Any failure raises :class:`PlannerError` and the
caller keeps the typed unavailable boundary with zero dispatch.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import PrivateAttr

from ..contracts.base import RuntimeModel
from ..contracts.capability import (
    AbbreviationResolveInput,
    DocumentReadInput,
    DocumentRetrieveInput,
    DocumentSearchInput,
    KnowledgeGraphInput,
    MemoryLookupInput,
    PeopleLookupInput,
    SectionReadInput,
)
from ..contracts.locators import DocumentLocator
from ..contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    ResearchPlanningInput,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from ..contracts.state import GraphRuntimeContext
from ..contracts.validation import (
    ContractValidationError,
    validate_research_planning_input,
    validate_task_plan,
)
from .projection import build_planner_model_input

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptivePlanner",
    "PlannerError",
]


class PlannerError(ValueError):
    """The planner cannot produce a governable initial proposal."""


#: Capabilities the model may propose. ``write`` is deliberately absent: the
#: pilot has no governed write planning, so a write proposal fails closed.
_PLANNER_TARGETLESS = frozenset(
    {
        "people.lookup",
        "document.search",
        "knowledge_graph.query",
        "memory.lookup",
    }
)
_PLANNER_TARGET_BEARING = frozenset(
    {
        "document.read",
        "section.read",
    }
)
_PLANNER_OPTIONAL_TARGETS = frozenset({"document.retrieve"})
_PLANNER_TOKEN_BEARING = frozenset({"abbreviation.resolve"})
_PLANNER_CAPABILITIES = (
    _PLANNER_TARGETLESS | _PLANNER_TARGET_BEARING | _PLANNER_OPTIONAL_TARGETS | _PLANNER_TOKEN_BEARING
)


def _skill_covers_work_type(planning_input: ResearchPlanningInput) -> bool:
    """True when a deterministic skill owns this work type (Task 12 expansion).

    Coverage is work-type ownership, not success: ``compare``/``retrieve``/
    ``summarize`` are always skill-owned; ``multi_goal``, ``cross_domain``,
    and ``evaluate`` are skill-owned exactly when their intake predicate
    holds (``covers_input``: routing arity met, a named person or bound
    documents present). A covering skill's ``ContractValidationError``
    refusal is deliberate and final — the model path must not second-guess
    it. Below-intake inputs stay uncovered so the governed model path still
    owns genuinely open work.
    """
    from ..skills.compare import policy as compare_policy
    from ..skills.cross_domain import policy as cross_domain_policy
    from ..skills.evaluate import policy as evaluate_policy
    from ..skills.multi_goal import policy as multi_goal_policy
    from ..skills.retrieve import policy as retrieve_policy
    from ..skills.summarize import policy as summarize_policy

    work_type = planning_input.query_analysis.work_type
    if (
        compare_policy.supports_work_type(work_type)
        or retrieve_policy.supports_work_type(work_type)
        or summarize_policy.supports_work_type(work_type)
    ):
        return True
    if multi_goal_policy.supports_work_type(work_type):
        return multi_goal_policy.covers_input(planning_input)
    if cross_domain_policy.supports_work_type(work_type):
        return cross_domain_policy.covers_input(planning_input)
    if evaluate_policy.supports_work_type(work_type):
        return evaluate_policy.covers_input(planning_input)
    return False


#: Roles a planner step may target. The planner references server-issued
#: binding IDs only; user/reference roles that cannot anchor a target unit
#: (discovered/supporting) are rejected so the planner cannot promote
#: unsettled discovery into targets.
_PLANNABLE_BINDING_ROLES = frozenset({"target", "reference"})

_PLANNER_SYSTEM_PROMPT = (
    "You are a GOVERNED planning proposer for a factual research system. "
    "You NEVER execute tools and your output is advisory-only: a validator "
    "rebuilds every task server-side and rejects anything ungovernable.\n"
    "Output ONE JSON object only (no markdown, no prose) with key 'tasks': "
    "an ordered list of steps. Each step has: 'capability' (one of the "
    "catalog names you were given), 'task_objective' (short label), "
    "'targets' (binding IDs from the bindings list, verbatim — inventing an "
    "ID rejects the whole proposal), 'depends_on' (0-based indexes of "
    "earlier steps only). 'abbreviation.resolve' steps also carry 'tokens' "
    "(non-blank strings). Rules: document.read/section.read need at least "
    "one target; people.lookup/document.search/knowledge_graph.query/"
    "memory.lookup take no targets; document.retrieve may be targetless; "
    "document.search only when the discovery policy allows it."
)

_PLANNER_USER_TEMPLATE = (
    "Propose an initial execution plan for this request.\n\n"
    "PLANNING INPUT (minimized, redacted):\n{planning_json}\n\n"
    "Respond with ONE JSON object only: "
    '{{"tasks": [{{"capability": ..., "task_objective": ..., '
    '"targets": [...], "depends_on": [...]}}]}}'
)


def _fail(message: str) -> PlannerError:
    return PlannerError(message)


class AdaptivePlanner(RuntimeModel):
    """Request-scoped governed planner (runtime-only, never checkpointed).

    Constructed once per request by the ingress owner and wired onto
    ``RuntimeServices.adaptive_planner`` (or injected with a
    ``provider_factory`` in tests). ``propose_initial`` is the single entry:
    deterministic skill first, governed model proposal otherwise.
    """

    model_config = {"extra": "forbid", "frozen": False, "arbitrary_types_allowed": True}

    _provider_factory: Callable[[], Any] | None = PrivateAttr(default=None)

    def __init__(self, *, provider_factory: Callable[[], Any] | None = None) -> None:
        super().__init__()
        self._provider_factory = provider_factory

    def _provider(self) -> Any:
        if self._provider_factory is not None:
            return self._provider_factory()
        from app.services.llm import get_planner_provider

        return get_planner_provider()

    async def propose_initial(
        self,
        planning_input: ResearchPlanningInput,
        runtime: GraphRuntimeContext,
    ) -> Any:
        """Propose the initial plan: deterministic skill first, else model.

        Returns the shared ``InitialProposal`` (plan + optional reduce spec)
        so the governed entry leases/checkpoints exactly like the skill path.
        The model path runs only when no deterministic skill covers the work
        type (a covering skill's refusal is re-raised untouched). Raises
        :class:`PlannerError` when no governable proposal exists; the caller
        then keeps the typed unavailable boundary (zero dispatch).
        """
        from ..complex_research_graph import InitialProposal, build_initial_proposal

        try:
            validate_research_planning_input(planning_input)
        except ContractValidationError as exc:
            raise _fail(f"refusing to plan from an invalid envelope: {exc}") from exc
        if planning_input.current_plan is not None:
            raise _fail("the adaptive planner proposes initial plans only")
        try:
            return build_initial_proposal(planning_input)
        except ContractValidationError:
            if _skill_covers_work_type(planning_input):
                # The covering skill refused deliberately (arity, roles, or
                # catalog): re-raise untouched so the caller keeps the typed
                # unavailable boundary instead of a model second opinion.
                raise
        model_input = build_planner_model_input(planning_input)
        steps = await self._propose_steps(model_input)
        plan = self._build_plan(planning_input, runtime, steps)
        try:
            validate_task_plan(plan, planning_input.bindings)
        except ContractValidationError as exc:
            raise _fail(f"model proposal failed validation: {exc}") from exc
        self._check_runtime_governance(plan, runtime, planning_input)
        return InitialProposal(plan=plan, reduce_spec=None)

    async def _propose_steps(self, model_input: Any) -> tuple[Mapping[str, Any], ...]:
        """Collect the model step list (proposal only, never executed)."""
        from app.services.llm.types import LLMMessage as _LLMMsg

        provider = self._provider()
        user_content = _PLANNER_USER_TEMPLATE.format(
            planning_json=model_input.model_dump_json()
        )
        response_text = ""
        try:
            stream = provider.astream(
                [_LLMMsg(role="user", content=user_content)],
                system_prompt=_PLANNER_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=800,
                think=False,
            )
            async for chunk in stream:
                if getattr(chunk, "type", "text") == "thinking":
                    continue
                text = getattr(chunk, "text", None)
                if not text:
                    continue
                response_text += str(text)
                if "{" in response_text and "}" in response_text:
                    start, end = response_text.find("{"), response_text.rfind("}")
                    if end > start:
                        try:
                            json.loads(response_text[start : end + 1])
                            break
                        except json.JSONDecodeError:
                            pass
        except PlannerError:
            raise
        except Exception as exc:
            raise _fail(f"planner model call failed: {exc}") from exc
        payload = _extract_json_object(response_text)
        if not isinstance(payload, dict):
            raise _fail("planner model did not return a JSON object")
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not all(
            isinstance(step, Mapping) for step in tasks
        ):
            raise _fail("planner proposal must carry a 'tasks' list of step objects")
        return tuple(tasks)

    def _build_plan(
        self,
        planning_input: ResearchPlanningInput,
        runtime: GraphRuntimeContext,
        steps: tuple[Mapping[str, Any], ...],
    ) -> TaskPlan:
        """Construct the plan server-side from validated model steps.

        Task IDs, target-unit IDs, origins, goal, and plan ID are server-set;
        every input is built from a typed constructor (the model supplies
        only capability names, objectives, binding-ID targets, dependency
        indexes, and abbreviation tokens). Anything ungovernable raises
        :class:`PlannerError` before anything is leased or checkpointed.
        """
        catalog = {entry.name for entry in planning_input.capability_catalog}
        allowed = runtime.capability_runtime.allowed_capabilities
        budget = planning_input.budget
        if not steps:
            raise _fail("planner proposal must contain at least one task")
        if len(steps) > budget.max_tasks_remaining:
            raise _fail(
                f"planner proposes {len(steps)} task(s) but the task budget "
                f"allows {budget.max_tasks_remaining}"
            )
        binding_by_id = {
            binding.binding_id: binding for binding in planning_input.bindings.bindings
        }
        normalized: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            capability = step.get("capability")
            if capability not in _PLANNER_CAPABILITIES:
                raise _fail(
                    f"planner step {index} proposes unsupported capability "
                    f"{capability!r}; refusing an ungovernable proposal"
                )
            if capability not in catalog:
                raise _fail(
                    f"planner step {index} capability {capability!r} is not in "
                    "the current capability catalog"
                )
            if capability not in allowed:
                raise _fail(
                    f"planner step {index} capability {capability!r} is not in "
                    "the current allowed capabilities; refusing scope widening"
                )
            if capability == "people.lookup" and not runtime.capability_runtime.can_read_people:
                raise _fail(
                    f"planner step {index} proposes 'people.lookup' without "
                    "people permission for this request"
                )
            objective = step.get("task_objective")
            if not isinstance(objective, str) or not objective.strip():
                raise _fail(f"planner step {index} needs a non-blank task_objective")
            targets = step.get("targets", [])
            if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
                raise _fail(f"planner step {index} targets must be a list")
            target_ids = tuple(targets)
            if any(not isinstance(target, str) or not target for target in target_ids):
                raise _fail(f"planner step {index} targets must be binding-ID strings")
            if len(set(target_ids)) != len(target_ids):
                raise _fail(f"planner step {index} repeats a target")
            for binding_id in target_ids:
                binding = binding_by_id.get(binding_id)
                if binding is None:
                    raise _fail(
                        f"planner step {index} references unknown binding "
                        f"{binding_id!r}; the planner cannot mint identity"
                    )
                if binding.role not in _PLANNABLE_BINDING_ROLES:
                    raise _fail(
                        f"planner step {index} targets binding {binding_id!r} "
                        f"with role {binding.role!r}; only target/reference "
                        "bindings anchor target units"
                    )
            if capability in _PLANNER_TARGET_BEARING and not target_ids:
                raise _fail(
                    f"planner step {index} capability {capability!r} needs a target"
                )
            if capability in _PLANNER_TARGETLESS and target_ids:
                raise _fail(
                    f"planner step {index} capability {capability!r} takes no targets"
                )
            if capability == "document.search" and not (
                planning_input.discovery_policy.allow_reference_discovery
                or planning_input.discovery_policy.allow_supporting_discovery
            ):
                raise _fail(
                    f"planner step {index} proposes document.search but "
                    "discovery is disabled by policy"
                )
            if (
                capability == "document.search"
                and planning_input.discovery_policy.max_discovered_documents < 1
            ):
                raise _fail(
                    f"planner step {index} proposes document.search but "
                    "max_discovered_documents is 0"
                )
            depends = step.get("depends_on", [])
            if not isinstance(depends, Sequence) or isinstance(depends, (str, bytes)):
                raise _fail(f"planner step {index} depends_on must be a list")
            for dependency in depends:
                if not isinstance(dependency, int) or isinstance(dependency, bool):
                    raise _fail(
                        f"planner step {index} depends_on must be step indexes"
                    )
                if dependency < 0 or dependency >= len(steps):
                    raise _fail(
                        f"planner step {index} depends on unknown step {dependency}"
                    )
                if dependency == index:
                    raise _fail(f"planner step {index} cannot depend on itself")
                if dependency > index:
                    raise _fail(
                        f"planner step {index} depends on a later step "
                        f"{dependency}; steps must order topologically"
                    )
            tokens: tuple[str, ...] = ()
            if capability == "abbreviation.resolve":
                raw_tokens = step.get("tokens", [])
                if (
                    not isinstance(raw_tokens, Sequence)
                    or isinstance(raw_tokens, (str, bytes))
                    or not raw_tokens
                ):
                    raise _fail(
                        f"planner step {index} abbreviation.resolve needs "
                        "a non-empty tokens list"
                    )
                tokens = tuple(raw_tokens)
                if any(
                    not isinstance(token, str) or not token.strip() for token in tokens
                ):
                    raise _fail(
                        f"planner step {index} abbreviation tokens must be "
                        "non-blank strings"
                    )
            elif step.get("tokens") is not None:
                raise _fail(
                    f"planner step {index} carries tokens for capability "
                    f"{capability!r}; refusing smuggled input"
                )
            normalized.append(
                {
                    "capability": capability,
                    "objective": objective.strip()[:500],
                    "targets": target_ids,
                    "depends_on": tuple(depends),
                    "tokens": tokens,
                }
            )
        entry_width = sum(1 for step in normalized if not step["depends_on"])
        if entry_width > budget.max_parallel_branches:
            raise _fail(
                f"planner entry fan-out width {entry_width} exceeds "
                f"max_parallel_branches {budget.max_parallel_branches}"
            )
        # One target unit per referenced binding, first-seen order; the
        # server assigns every ID so the model cannot collide or alias.
        unit_by_binding: dict[str, str] = {}
        units: list[TargetUnit] = []
        for step in normalized:
            for binding_id in step["targets"]:
                if binding_id not in unit_by_binding:
                    target_id = f"t{len(units) + 1}"
                    unit_by_binding[binding_id] = target_id
                    units.append(
                        TargetUnit(
                            target_id=target_id,
                            binding_id=binding_id,
                            requested_locator=DocumentLocator(kind="document"),
                            completion_criteria=(
                                CoverageCriterion(kind="coverage"),
                            ),
                        )
                    )
        query = planning_input.semantic.contextualized_query
        tasks: list[TaskSpec] = []
        for position, step in enumerate(normalized):
            task_id = f"T{position + 1}"
            depends_on = tuple(f"T{dep + 1}" for dep in step["depends_on"])
            target_ids = tuple(unit_by_binding[binding_id] for binding_id in step["targets"])
            capability = step["capability"]
            task_input: Any
            if capability == "people.lookup":
                task_input = PeopleLookupInput(kind="people.lookup", query=query)
            elif capability == "document.search":
                task_input = DocumentSearchInput(
                    kind="document.search", query=query, person_identifier=None
                )
            elif capability == "document.retrieve":
                task_input = DocumentRetrieveInput(
                    kind="document.retrieve",
                    query=query,
                    target_ids=target_ids,
                    top_k=8,
                )
            elif capability == "document.read":
                task_input = DocumentReadInput(
                    kind="document.read", target_ids=target_ids
                )
            elif capability == "section.read":
                task_input = SectionReadInput(
                    kind="section.read", target_ids=target_ids
                )
            elif capability == "knowledge_graph.query":
                task_input = KnowledgeGraphInput(
                    kind="knowledge_graph.query", query=query
                )
            elif capability == "memory.lookup":
                task_input = MemoryLookupInput(kind="memory.lookup", query=query)
            elif capability == "abbreviation.resolve":
                task_input = AbbreviationResolveInput(
                    kind="abbreviation.resolve", tokens=step["tokens"]
                )
            else:  # pragma: no cover — allowlist check above is exhaustive.
                raise _fail(f"unsupported capability {capability!r}")
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    capability=capability,
                    task_objective=step["objective"],
                    input=task_input,
                    depends_on=depends_on,
                    origin=InitialTaskOrigin(kind="initial"),
                )
            )
        work_type = str(planning_input.query_analysis.work_type)
        return TaskPlan(
            contract_version="2.0",
            plan_id=f"adaptive-{work_type}",
            goal=query,
            target_units=tuple(units),
            tasks=tuple(tasks),
        )

    def _check_runtime_governance(
        self,
        plan: TaskPlan,
        runtime: GraphRuntimeContext,
        planning_input: ResearchPlanningInput,
    ) -> None:
        """Runtime-only checks the frozen validator cannot express.

        The frozen ``validate_task_plan`` owns shape/DAG/binding/target
        integrity; this layer re-asserts current authorization (catalog ∩
        allowed ∩ people flag), the discovery policy, and the entry fan-out
        width against the live request — the same runtime intersection the
        replan gateway enforces.
        """
        catalog = {entry.name for entry in planning_input.capability_catalog}
        allowed = runtime.capability_runtime.allowed_capabilities
        for task in plan.tasks:
            if task.capability not in catalog:
                raise _fail(
                    f"task {task.task_id} capability {task.capability!r} left "
                    "the current capability catalog"
                )
            if task.capability not in allowed:
                raise _fail(
                    f"task {task.task_id} capability {task.capability!r} is not "
                    "in the current allowed capabilities"
                )
            if (
                task.capability == "people.lookup"
                and not runtime.capability_runtime.can_read_people
            ):
                raise _fail(
                    f"task {task.task_id} capability 'people.lookup' is not "
                    "permitted for this request"
                )
            if task.capability == "document.search" and not (
                planning_input.discovery_policy.allow_reference_discovery
                or planning_input.discovery_policy.allow_supporting_discovery
            ):
                raise _fail(
                    f"task {task.task_id} proposes document.search but "
                    "discovery is disabled by policy"
                )


def _extract_json_object(raw: str) -> Any:
    """Extract the proposal JSON object (direct parse, then prose salvage)."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None

"""Governed Adaptive Replanner service (Phase 5, Task 11).

``AdaptiveReplanner`` is a runtime-only, request-scoped service (wired onto
``RuntimeServices.adaptive_replanner``): it owns NO checkpointed state,
dispatches NO capabilities, and resolves NO document identity. It proposes a
:data:`ReplanProposal` (new tasks only — never an authoritative appended
plan) that the governed entry (``validate_checkpoint_node``) still appends
through the single ``append_replan_tasks`` site, leases, and checkpoints
before the shared ``TaskScheduler`` sees it.

Ordering: the deterministic gap policy wins whenever it can construct (a
returned proposal is used untouched with zero model calls); the model path
runs only for advisable evaluator gaps the deterministic policy cannot
build (e.g. the read capabilities left the runtime catalog but an
alternative read or authorized discovery remains). Anything ungovernable —
unknown targets, scope widening, over-budget fan-out, model failure —
yields ``None`` so the caller spends the budget and finalizes with zero
dispatch. Completed tasks are immutable: the replanner never reruns a
completed task and never creates target units.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import PrivateAttr

from ..contracts.base import RuntimeModel
from ..contracts.capability import (
    DocumentReadInput,
    DocumentRetrieveInput,
    DocumentSearchInput,
    SectionReadInput,
)
from ..contracts.evaluation import (
    Contradiction,
    EvidenceEvaluation,
    MissingRequirement,
)
from ..contracts.execution import TaskExecutionSummary
from ..contracts.locators import SectionLocator
from ..contracts.planning import (
    ReplanTaskOrigin,
    ResearchPlanningInput,
    TaskSpec,
)
from ..contracts.state import GraphRuntimeContext
from .replan_projection import (
    ReplannerModelInput,
    build_replanner_model_input_from_envelope,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveReplanner",
    "Contradiction",
    "EvidenceEvaluation",
    "MissingRequirement",
    "ReplannerError",
    "TaskExecutionSummary",
]


class ReplannerError(ValueError):
    """The replanner cannot produce a governable append-only proposal."""


#: Capabilities the model may propose on replan. Read-family re-reads carry
#: existing target IDs; ``document.search`` is targetless and policy-gated.
#: ``people.lookup`` never re-runs on replan (the recovery path owns
#: not_found), and write/KG/memory/abbreviation stay out of the replan
#: pilot — a proposal outside this set fails closed.
_REPLAN_READ_FAMILY = frozenset(
    {"document.read", "section.read", "document.retrieve"}
)
_REPLAN_TARGETLESS = frozenset({"document.search"})
_REPLAN_CAPABILITIES = _REPLAN_READ_FAMILY | _REPLAN_TARGETLESS

#: Model step keys. Anything else is smuggled input and rejects the proposal.
_REPLAN_STEP_KEYS = frozenset(
    {
        "capability",
        "task_objective",
        "targets",
        "depends_on_steps",
        "depends_on_completed",
    }
)

_REPLAN_SYSTEM_PROMPT = (
    "You are a GOVERNED replan proposer for a factual research system. "
    "You NEVER execute tools and your output is advisory-only: a validator "
    "rebuilds every task server-side and rejects anything ungovernable.\n"
    "Output ONE JSON object only (no markdown, no prose) with key 'tasks': "
    "an ordered list of re-read steps for the listed coverage gaps. Each "
    "step has: 'capability' (document.read, section.read, or "
    "document.retrieve for gap targets; document.search only when the "
    "discovery policy allows it), 'task_objective' (short label), "
    "'targets' (target IDs from the targets list, verbatim — inventing an "
    "ID rejects the whole proposal), 'depends_on_steps' (0-based indexes "
    "of earlier steps only), 'depends_on_completed' (already-completed "
    "task IDs from the outcomes list, verbatim). Rules: read-family steps "
    "need at least one target; document.search takes no targets; "
    "contradictions and semantic gaps are context only — re-read the "
    "coverage-gap target instead of inventing new work."
)

_REPLAN_USER_TEMPLATE = (
    "Propose a bounded append-only replan for these evaluator gaps.\n\n"
    "REPLAN INPUT (minimized, redacted):\n{planning_json}\n\n"
    "Respond with ONE JSON object only: "
    '{{"tasks": [{{"capability": ..., "task_objective": ..., '
    '"targets": [...], "depends_on_steps": [...], '
    '"depends_on_completed": [...]}}]}}'
)


def _fail(message: str) -> ReplannerError:
    return ReplannerError(message)


class AdaptiveReplanner(RuntimeModel):
    """Request-scoped governed replanner (runtime-only, never checkpointed).

    Constructed once per request by the ingress owner and wired onto
    ``RuntimeServices.adaptive_replanner`` (or injected with a
    ``provider_factory`` in tests). ``propose_replan`` is the single entry:
    deterministic gap policy first, governed model proposal otherwise.
    Total function: returns a :class:`ReplanProposal` or ``None`` (the
    caller then spends the budget so ``decide`` finalizes); it never raises
    for model content and never dispatches.
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

    async def propose_replan(
        self,
        state: Any,
        runtime: GraphRuntimeContext,
    ) -> Any:
        """Propose the bounded append-only replan: deterministic first, else model.

        Returns the shared ``ReplanProposal`` (new tasks + validation
        inputs) so the governed entry appends + validates + leases +
        checkpoints exactly like the deterministic path. ``None`` means no
        governable replan exists: the caller spends the budget and
        finalizes with zero dispatch.
        """
        from ..complex_research_graph import (
            build_model_replan_input,
            build_replan_proposal,
            replan_advisable,
        )

        deterministic = build_replan_proposal(state, runtime)
        if deterministic is not None:
            return deterministic
        try:
            if not replan_advisable(state):
                return None
            envelope = build_model_replan_input(state, runtime)
            return await self._propose_from_envelope(envelope, state, runtime)
        except ReplannerError as exc:
            logger.debug("governed replan refused: %s", exc)
            return None

    async def _propose_from_envelope(
        self,
        envelope: ResearchPlanningInput,
        state: Any,
        runtime: GraphRuntimeContext,
    ) -> Any:
        """Model path for advisable gaps the deterministic policy cannot build."""
        current = envelope.current_plan
        if current is None:
            raise _fail("replan requires a checkpointed plan")
        budget = envelope.budget
        if budget.max_replans_remaining < 1:
            raise _fail("no replan budget remaining")
        if budget.max_tasks_remaining < 1:
            raise _fail("no task budget remaining for a replan")
        model_input = build_replanner_model_input_from_envelope(envelope)
        steps = await self._propose_steps(model_input)
        return self._build_proposal(envelope, state, runtime, steps)

    async def _propose_steps(self, model_input: ReplannerModelInput) -> tuple[Mapping[str, Any], ...]:
        """Collect the model step list (proposal only, never executed)."""
        from app.services.llm.types import LLMMessage as _LLMMsg

        provider = self._provider()
        user_content = _REPLAN_USER_TEMPLATE.format(
            planning_json=model_input.model_dump_json()
        )
        response_text = ""
        try:
            stream = provider.astream(
                [_LLMMsg(role="user", content=user_content)],
                system_prompt=_REPLAN_SYSTEM_PROMPT,
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
        except Exception as exc:
            raise _fail(f"replanner model call failed: {exc}") from exc
        payload = _extract_json_object(response_text)
        if not isinstance(payload, dict):
            raise _fail("replanner model did not return a JSON object")
        found = payload.get("tasks")
        if not isinstance(found, list) or not all(
            isinstance(step, Mapping) for step in found
        ):
            raise _fail("replanner proposal must carry a 'tasks' list of step objects")
        return tuple(found)

    def _build_proposal(
        self,
        envelope: ResearchPlanningInput,
        state: Any,
        runtime: GraphRuntimeContext,
        steps: tuple[Mapping[str, Any], ...],
    ) -> Any:
        """Construct the replan proposal server-side from validated model steps.

        New task IDs, ``ReplanTaskOrigin`` lineage (attempted task IDs +
        prior evidence-use IDs, both server-resolved), and every task input
        are server-set; the model supplies only capability names,
        objectives, existing target IDs, and dependency references. The
        returned ``ReplanProposal`` carries new tasks only — the single
        authoritative append + frozen validation still happen in the
        governed entry point.
        """
        from ..complex_research_graph import ReplanProposal
        from ..replanning import entry_fanout_width

        current = envelope.current_plan
        assert current is not None
        catalog = {entry.name for entry in envelope.capability_catalog}
        allowed = runtime.capability_runtime.allowed_capabilities
        budget = envelope.budget
        if not steps:
            raise _fail("replanner proposal must contain at least one task")
        if len(steps) > budget.max_tasks_remaining:
            raise _fail(
                f"replanner proposes {len(steps)} task(s) but the task budget "
                f"allows {budget.max_tasks_remaining}"
            )
        existing_ids: set[str] = set()
        for task in current.tasks:
            existing_ids.add(task.task_id)
        target_ids: set[str] = set()
        section_targets: set[str] = set()
        for unit in current.target_units:
            target_ids.add(unit.target_id)
            if isinstance(unit.requested_locator, SectionLocator):
                section_targets.add(unit.target_id)
        gap_description: dict[str, str] = {}
        missing = envelope.prior_evaluation.missing if envelope.prior_evaluation is not None else ()
        for requirement in missing or ():
            gap_description.setdefault(
                requirement.target_id, (requirement.description or "").strip()[:200]
            )
        attempted: dict[str, list[str]] = {}
        for task in current.tasks:
            for target_id in getattr(task.input, "target_ids", None) or ():
                attempted.setdefault(target_id, []).append(task.task_id)
        outcome_ids: set[str] = set()
        for outcome in envelope.task_outcomes or ():
            outcome_ids.add(outcome.task_id)
        prior_use_ids = tuple(ref.use_id for ref in envelope.prior_evidence_uses or ())
        taken: set[str] = set(existing_ids)
        base_index = len(current.tasks) + 1
        normalized: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            unknown = set(step.keys()) - _REPLAN_STEP_KEYS
            if unknown:
                raise _fail(
                    f"replanner step {index} carries unexpected keys "
                    f"{sorted(unknown)}; refusing smuggled input"
                )
            capability = step.get("capability")
            if capability not in _REPLAN_CAPABILITIES:
                raise _fail(
                    f"replanner step {index} proposes unsupported capability "
                    f"{capability!r}; refusing an ungovernable proposal"
                )
            if capability not in catalog:
                raise _fail(
                    f"replanner step {index} capability {capability!r} is not in "
                    "the current capability catalog"
                )
            if capability not in allowed:
                raise _fail(
                    f"replanner step {index} capability {capability!r} is not in "
                    "the current allowed capabilities; refusing scope widening"
                )
            objective = step.get("task_objective")
            if not isinstance(objective, str) or not objective.strip():
                raise _fail(f"replanner step {index} needs a non-blank task_objective")
            raw_targets = step.get("targets", [])
            if not isinstance(raw_targets, Sequence) or isinstance(
                raw_targets, (str, bytes)
            ):
                raise _fail(f"replanner step {index} targets must be a list")
            chosen = tuple(raw_targets)
            if any(not isinstance(item, str) or not item for item in chosen):
                raise _fail(
                    f"replanner step {index} targets must be target-ID strings"
                )
            if len(set(chosen)) != len(chosen):
                raise _fail(f"replanner step {index} repeats a target")
            for target_id in chosen:
                if target_id not in target_ids:
                    raise _fail(
                        f"replanner step {index} references unknown target "
                        f"{target_id!r}; the replanner cannot mint targets"
                    )
            if capability in _REPLAN_READ_FAMILY and not chosen:
                raise _fail(
                    f"replanner step {index} capability {capability!r} needs a target"
                )
            if capability in _REPLAN_TARGETLESS and chosen:
                raise _fail(
                    f"replanner step {index} capability {capability!r} takes no targets"
                )
            if capability == "section.read" and any(
                target_id not in section_targets for target_id in chosen
            ):
                raise _fail(
                    f"replanner step {index} proposes section.read for a "
                    "non-section target; refusing a mistargeted re-read"
                )
            if capability == "document.search" and not (
                envelope.discovery_policy.allow_reference_discovery
                or envelope.discovery_policy.allow_supporting_discovery
            ):
                raise _fail(
                    f"replanner step {index} proposes document.search but "
                    "discovery is disabled by policy"
                )
            if (
                capability == "document.search"
                and envelope.discovery_policy.max_discovered_documents < 1
            ):
                raise _fail(
                    f"replanner step {index} proposes document.search but "
                    "max_discovered_documents is 0"
                )
            raw_step_deps = step.get("depends_on_steps", [])
            if not isinstance(raw_step_deps, Sequence) or isinstance(
                raw_step_deps, (str, bytes)
            ):
                raise _fail(f"replanner step {index} depends_on_steps must be a list")
            for dependency in raw_step_deps:
                if not isinstance(dependency, int) or isinstance(dependency, bool):
                    raise _fail(
                        f"replanner step {index} depends_on_steps must be step indexes"
                    )
                if dependency < 0 or dependency >= len(steps):
                    raise _fail(
                        f"replanner step {index} depends on unknown step {dependency}"
                    )
                if dependency == index:
                    raise _fail(f"replanner step {index} cannot depend on itself")
                if dependency > index:
                    raise _fail(
                        f"replanner step {index} depends on a later step "
                        f"{dependency}; steps must order topologically"
                    )
            raw_completed = step.get("depends_on_completed", [])
            if not isinstance(raw_completed, Sequence) or isinstance(
                raw_completed, (str, bytes)
            ):
                raise _fail(
                    f"replanner step {index} depends_on_completed must be a list"
                )
            completed_deps: list[str] = []
            for reference in raw_completed:
                if not isinstance(reference, str) or not reference:
                    raise _fail(
                        f"replanner step {index} depends_on_completed must be "
                        "task-ID strings"
                    )
                if reference not in existing_ids:
                    raise _fail(
                        f"replanner step {index} depends on unknown completed "
                        f"task {reference!r}"
                    )
                if reference in completed_deps:
                    raise _fail(
                        f"replanner step {index} repeats completed dependency "
                        f"{reference!r}"
                    )
                completed_deps.append(reference)
            normalized.append(
                {
                    "capability": capability,
                    "objective": objective.strip()[:500],
                    "targets": chosen,
                    "step_deps": tuple(raw_step_deps),
                    "completed_deps": tuple(completed_deps),
                }
            )
        fresh: list[TaskSpec] = []
        position = base_index
        while f"T{position}" in taken:
            position += 1
        for step in normalized:
            task_id = f"T{position}"
            position += 1
            while f"T{position}" in taken or any(
                candidate.task_id == f"T{position}" for candidate in fresh
            ):
                position += 1
            taken.add(task_id)
            # Step-index dependencies resolve to real new task IDs in the
            # remap pass below (taken collisions may skip dense T-ids).
            depends_on = tuple(step["completed_deps"])
            capability = step["capability"]
            target_ids_tuple = tuple(step["targets"])
            if capability == "document.read":
                task_input: Any = DocumentReadInput(
                    kind="document.read", target_ids=target_ids_tuple
                )
            elif capability == "section.read":
                task_input = SectionReadInput(
                    kind="section.read", target_ids=target_ids_tuple
                )
            elif capability == "document.retrieve":
                task_input = DocumentRetrieveInput(
                    kind="document.retrieve",
                    query=current.goal,
                    target_ids=target_ids_tuple,
                    top_k=8,
                )
            elif capability == "document.search":
                task_input = DocumentSearchInput(
                    kind="document.search",
                    query=current.goal,
                    person_identifier=None,
                )
            else:  # pragma: no cover — allowlist check above is exhaustive.
                raise _fail(f"unsupported capability {capability!r}")
            trigger: list[str] = []
            for reference in step["completed_deps"]:
                if reference in outcome_ids and reference not in trigger:
                    trigger.append(reference)
            for target_id in target_ids_tuple:
                for reference in attempted.get(target_id, ()):
                    if reference in outcome_ids and reference not in trigger:
                        trigger.append(reference)
            if step["capability"] in _REPLAN_READ_FAMILY and target_ids_tuple:
                first_gap = gap_description.get(target_ids_tuple[0], "")
                reason = (
                    f"evaluator-driven replan for target {target_ids_tuple[0]}"
                    + (f": {first_gap}" if first_gap else "")
                )
            else:
                reason = (
                    "evaluator-driven bounded discovery fallback without "
                    "a materialized scalar"
                )
            fresh.append(
                TaskSpec(
                    task_id=task_id,
                    capability=capability,
                    task_objective=step["objective"],
                    input=task_input,
                    depends_on=depends_on,
                    origin=ReplanTaskOrigin(
                        kind="replan",
                        reason=reason,
                        task_ids=tuple(trigger),
                        evidence_use_ids=prior_use_ids,
                    ),
                )
            )
        # Remap step-index dependencies to the actual new task IDs: the
        # depends_on built above assumed dense T-ids; rebuild from order so
        # skipped IDs (taken collisions) cannot dangle.
        new_ids = [candidate.task_id for candidate in fresh]
        remapped: list[TaskSpec] = []
        for order, candidate in enumerate(fresh):
            step = normalized[order]
            depends_on = tuple(step["completed_deps"]) + tuple(
                new_ids[dependency] for dependency in step["step_deps"]
            )
            remapped.append(candidate.model_copy(update={"depends_on": depends_on}))
        width = entry_fanout_width(tuple(remapped))
        if width > budget.max_parallel_branches:
            raise _fail(
                f"replan entry fan-out width {width} exceeds "
                f"max_parallel_branches {budget.max_parallel_branches}"
            )
        return ReplanProposal(
            new_tasks=tuple(remapped),
            outcomes=tuple(envelope.task_outcomes or ()),
            policy=envelope.discovery_policy,
            budget=envelope.budget,
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

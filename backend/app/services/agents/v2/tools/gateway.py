"""Proposal + execution-guard side of the governed tool gateway (Phase 3, Task 2).

``AgentToolGateway.propose`` performs exactly::

    proposal
    -> convert to a TaskSpec proposal
    -> validate_replan against the current runtime catalog (append-only)
    -> return accepted append-only plan or typed rejection

It is a proposal adapter only: it never persists a plan, never dispatches work,
and never calls a capability. The subgraph ``validate_checkpoint`` node writes
the accepted plan into state, the supervisor saver persists it, the ``execute``
node runs the shared scheduler, and ``ObservationProjector`` projects the
result only after that persisted execution.

``require_planned_dispatch`` is the matching pure pre-dispatch guard: a task
must exist in the validated persisted plan AND stay visible in the current
runtime catalog before the scheduler may dispatch it. Unknown, unauthorized,
unavailable, or unplanned references fail closed here; the shared scheduler
remains the only path that resolves a capability and owns the precise typed
denial at dispatch time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from ..capabilities import CapabilityDenied
from ..contracts.capability import CapabilityInput, DocumentSearchInput
from ..contracts.planning import (
    DiscoveryPolicy,
    ReplanTaskOrigin,
    ResearchBudgetView,
    TaskPlan,
    TaskSpec,
)
from ..contracts.state import GraphRuntimeContext
from ..contracts.validation import ContractValidationError, validate_replan

RejectionCode = Literal[
    "unknown_capability",
    "unauthorized_capability",
    "unavailable_capability",
    "invalid_proposal",
    "invalid_plan",
    "budget_exhausted",
]


@dataclass(frozen=True)
class CapabilityInvocationProposal:
    """One framework tool call interpreted as a task proposal (never a dispatch)."""

    capability: str
    objective: str
    input: CapabilityInput
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProposalRejection:
    """Typed fail-closed reason for a refused proposal."""

    code: RejectionCode
    message: str


@dataclass(frozen=True)
class ToolProposalOutcome:
    """Accepted append-only plan, or the current plan unchanged plus a rejection."""

    accepted: bool
    plan: TaskPlan
    rejection: ProposalRejection | None = None


class UnplannedCapabilityDispatch(ValueError):
    """A dispatch was attempted for a task with no validated persisted TaskSpec."""


_DEFAULT_POLICY = DiscoveryPolicy(
    allow_reference_discovery=False,
    allow_supporting_discovery=False,
    max_discovered_documents=0,
)
_DEFAULT_BUDGET = ResearchBudgetView(
    max_tasks_remaining=8,
    max_replans_remaining=3,
    max_parallel_branches=2,
)


class AgentToolGateway:
    """Validates tool-call proposals into append-only plans; never dispatches."""

    def __init__(
        self,
        *,
        discovery_policy: DiscoveryPolicy | None = None,
        budget: ResearchBudgetView | None = None,
    ) -> None:
        self._policy = discovery_policy if discovery_policy is not None else _DEFAULT_POLICY
        self._budget = budget if budget is not None else _DEFAULT_BUDGET

    async def propose(
        self,
        proposal: CapabilityInvocationProposal,
        current_plan: TaskPlan,
        runtime: GraphRuntimeContext,
    ) -> ToolProposalOutcome:

        """Convert one proposal to an appended TaskSpec and validate it closed."""
        rejection = self._check_proposal_shape(proposal)
        if rejection is None:
            rejection = self._check_runtime_catalog(proposal, runtime)
        if rejection is not None:
            return ToolProposalOutcome(
                accepted=False, plan=current_plan, rejection=rejection
            )
        task_id = self._next_task_id(current_plan)
        try:
            candidate = TaskSpec(
                task_id=task_id,
                capability=proposal.capability,
                task_objective=proposal.objective,
                input=proposal.input,
                depends_on=tuple(proposal.depends_on),
                origin=ReplanTaskOrigin(
                    kind="replan",
                    reason=proposal.objective,
                    task_ids=tuple(proposal.depends_on),
                    evidence_use_ids=(),
                ),
            )
            proposed = current_plan.model_copy(
                update={"tasks": current_plan.tasks + (candidate,)}
            )
            accepted_plan = validate_replan(
                current_plan, proposed, (), self._policy, self._budget
            )
        except ValidationError as error:
            # Malformed proposal payloads never escape: the frozen
            # CapabilityInput boundary is enforced as a typed rejection.
            return ToolProposalOutcome(
                accepted=False,
                plan=current_plan,
                rejection=ProposalRejection(
                    code="invalid_proposal", message=str(error)
                ),
            )
        except ContractValidationError as error:
            message = str(error)
            code: RejectionCode = (
                "budget_exhausted" if "budget" in message.lower() else "invalid_plan"
            )
            return ToolProposalOutcome(
                accepted=False,
                plan=current_plan,
                rejection=ProposalRejection(code=code, message=message),
            )
        return ToolProposalOutcome(accepted=True, plan=accepted_plan)

    @staticmethod
    def _check_proposal_shape(proposal: CapabilityInvocationProposal) -> ProposalRejection | None:
        if not proposal.capability or not proposal.capability.strip():
            return ProposalRejection(
                code="invalid_proposal", message="proposal capability must be non-blank"
            )
        if not proposal.objective or not proposal.objective.strip():
            return ProposalRejection(
                code="invalid_proposal", message="proposal objective must be non-blank"
            )
        input_kind = getattr(proposal.input, "kind", None)
        if input_kind != proposal.capability:
            return ProposalRejection(
                code="invalid_proposal",
                message=(
                    f"proposal capability {proposal.capability!r} does not match "
                    f"its input kind {input_kind!r}"
                ),
            )
        if (
            isinstance(proposal.input, DocumentSearchInput)
            and proposal.input.person_identifier is not None
        ):
            # The People->Document scalar is materialized server-side by the
            # deterministic dependency adapter after a governed people.lookup;
            # the planner may never supply it.
            return ProposalRejection(
                code="invalid_proposal",
                message="proposal must not supply a materialized person_identifier",
            )
        return None

    @staticmethod
    def _check_runtime_catalog(
        proposal: CapabilityInvocationProposal, runtime: GraphRuntimeContext
    ) -> ProposalRejection | None:
        # Current trusted runtime authority ALWAYS narrows: registry membership
        # is an additional check, never an early return that skips these.
        # Capability resolution for dispatch belongs to the shared scheduler,
        # which owns the precise typed denial.
        trusted = runtime.capability_runtime
        if proposal.capability not in trusted.allowed_capabilities:
            return ProposalRejection(
                code="unauthorized_capability",
                message=f"capability {proposal.capability!r} is not permitted",
            )
        if proposal.capability == "people.lookup" and not trusted.can_read_people:
            return ProposalRejection(
                code="unauthorized_capability",
                message="capability 'people.lookup' is not permitted",
            )
        services = runtime.services
        registry = getattr(services, "capability_registry", None) if services else None
        if registry is not None:
            if proposal.capability not in registry.capability_names():
                return ProposalRejection(
                    code="unauthorized_capability",
                    message=(
                        f"capability {proposal.capability!r} is not permitted "
                        "or available for this request"
                    ),
                )
        return None

    @staticmethod
    def _next_task_id(current_plan: TaskPlan) -> str:
        taken = {task.task_id for task in current_plan.tasks}
        index = len(current_plan.tasks) + 1
        while f"T{index}" in taken:
            index += 1
        return f"T{index}"


def require_planned_dispatch(
    plan: TaskPlan, task_id: str, runtime: GraphRuntimeContext
) -> TaskSpec:
    """Pure pre-dispatch guard: planned in the persisted plan AND visible now.

    Raises :class:`UnplannedCapabilityDispatch` when no TaskSpec owns
    ``task_id``, and :class:`CapabilityDenied` when the capability left the
    current runtime catalog (unknown, revoked, or gated). Performs no
dispatch and resolves no capability itself.
    """
    for task in plan.tasks:
        if task.task_id == task_id:
            break
    else:
        raise UnplannedCapabilityDispatch(
            f"task {task_id!r} has no validated persisted TaskSpec"
        )
    trusted = runtime.capability_runtime
    if task.capability not in trusted.allowed_capabilities:
        raise CapabilityDenied(
            f"capability {task.capability!r} is not permitted for this request"
        )
    if task.capability == "people.lookup" and not trusted.can_read_people:
        raise CapabilityDenied("capability 'people.lookup' is not permitted")
    services = runtime.services
    registry = getattr(services, "capability_registry", None) if services else None
    if registry is not None and task.capability not in registry.capability_names():
        raise CapabilityDenied(
            f"capability {task.capability!r} is not permitted or available "
            "for this request"
        )
    return task

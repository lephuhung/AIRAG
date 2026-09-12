"""Execute node: dispatch the checkpointed plan (Phase 2, Task 3).

``execute_node`` reads the authoritative plan via ``require_checkpointed_plan``
— dispatch without checkpoint ownership is a ``MissingCheckpointedPlan`` and
no capability is ever called — builds the shared ``TaskScheduler`` from
``runtime.services.capability_registry``, executes with the checkpointed prior
results and bindings, and returns a full frozen ``ExecutionState`` partial via
``execution_update`` (exact field name ``evidence_evaluation``, never an
alias). Follows the Task-1 node-injection convention: the node accepts the
framework ``Runtime[GraphRuntimeContext]`` or, for unit tests, the context
itself.
"""
from __future__ import annotations

from langgraph.runtime import Runtime

from ..contracts.evaluation import EvidenceEvaluation
from ..contracts.execution import AgentResult
from ..contracts.planning import TaskPlan
from ..contracts.state import ExecutionState, GraphRuntimeContext, SupervisorV2State
from ..execution.scheduler import TaskScheduler
from .context import _context_of

__all__ = [
    "MissingCheckpointedPlan",
    "require_checkpointed_plan",
    "execution_update",
    "execute_node",
]


class MissingCheckpointedPlan(ValueError):
    """Execute requires a checkpointed TaskPlan; nothing may be dispatched."""


def require_checkpointed_plan(state: SupervisorV2State) -> TaskPlan:
    """Return the authoritative checkpointed plan or fail closed."""
    plan = state["execution"].plan
    if plan is None:
        raise MissingCheckpointedPlan(
            "execute requires a checkpointed TaskPlan: the fast_domain route "
            "must checkpoint its plan before any capability is dispatched"
        )
    return plan


def execution_update(
    state: SupervisorV2State,
    *,
    plan: TaskPlan | None = None,
    task_results: tuple[AgentResult, ...] | None = None,
    evidence_evaluation: EvidenceEvaluation | None = None,
) -> dict:
    """Return a full frozen ExecutionState partial.

    Uses the frozen field name ``evidence_evaluation`` (never an alias
    ``evaluation``); unset slots keep the current checkpointed values.
    """
    current = state["execution"]
    return {
        "execution": ExecutionState(
            plan=plan if plan is not None else current.plan,
            task_results=task_results if task_results is not None else current.task_results,
            evidence_evaluation=(
                evidence_evaluation
                if evidence_evaluation is not None
                else current.evidence_evaluation
            ),
        )
    }


async def execute_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Dispatch the checkpointed plan's ready tasks through the scheduler."""
    context = _context_of(runtime)
    plan = require_checkpointed_plan(state)
    scheduler = TaskScheduler(context.services.capability_registry)
    results = await scheduler.execute(
        plan=plan,
        runtime=context,
        prior_results=state["execution"].task_results,
        bindings=state["bindings"],
    )
    return execution_update(state, task_results=results)

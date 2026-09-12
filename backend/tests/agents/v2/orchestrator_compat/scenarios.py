"""Deterministic Phase 0 parity scenarios.

Every `Scenario` carries the canonical `initial_state` (as a typed
`SupervisorV2State`), the `runtime_context` that pairs with it (request-scoped,
**not** checkpointed), and a `checkpoint_json()` helper that serializes only the
state — never the runtime. The 10 scenarios cover the brief's full inventory:

1. one-task fast plan               — `fast_one_task`
2. three-task DAG                    — `dag_three_tasks`
3. async fan-in                      — `async_fan_in`
4. no-evidence `not_found`           — `no_evidence_not_found`
5. no-evidence `TIMEOUT`             — `no_evidence_timeout`
6. append-only replan                — `append_only_replan`
7. clarification interrupt/resume    — `clarification_resume`
8. changed ACL runtime replacement   — `acl_runtime_replaced`
9. cancellation                      — `cancellation_no_later_task`
10. outer-only streaming             — `outer_only_streaming`

All fixtures are literal and deterministic: stable UUIDs from
`uuid.UUID(int=...)` and stable timestamps from `datetime(2026, 9, 11, …, tzinfo=UTC)`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast
from uuid import UUID, uuid4

from .frozen_contracts import (
    AgentError,
    AgentResult,
    AgentStatus,
    CapabilityRuntimeContext,
    ClarificationRequest,
    ContractModel,
    ConversationContext,
    DocumentBindingSet,
    DocumentCandidate,
    EvidenceEvaluation,
    EvidenceUseRef,
    ExecutionState,
    FinalResponse,
    GraphRuntimeContext,
    QueryAnalysis,
    ReplanTaskOrigin,
    RequestContext,
    RouteDecision,
    RuntimeServices,
    ScopedDocument,
    SemanticContext,
    SupervisorV2State,
    TaskExecutionSummary,
    TaskPlan,
    TaskSpec,
    InitialTaskOrigin,
)


# ---------------------------------------------------------------------------
# Helpers — stable IDs and timestamps so scenarios are byte-for-byte deterministic.
# ---------------------------------------------------------------------------

_NS = UUID("00000000-0000-0000-0000-000000000000")


def _uid(slot: int) -> UUID:
    """Stable UUID from a slot number so diffs don't churn."""
    return UUID(int=slot)


_FIXED_TS = datetime(2026, 9, 11, 0, 0, 0, tzinfo=timezone.utc)


def _task(slot: int) -> str:
    return f"task-{slot:03d}"


def _target(slot: int) -> str:
    return f"target-{slot:03d}"


def _binding(slot: int) -> str:
    return f"binding-{slot:03d}"


def _doc(slot: int) -> UUID:
    return _uid(slot)


def _make_request(query: str, slot: int = 1) -> RequestContext:
    return RequestContext(
        contract_version="2.0",
        request_id=f"req-{slot:03d}",
        thread_id=f"thread-{slot:03d}",
        original_query=query,
        known_documents=(),
    )


def _make_conversation() -> ConversationContext:
    return ConversationContext(summary="")


def _make_semantic() -> SemanticContext:
    return SemanticContext(summary="")


def _make_bindings(bindings: tuple[ScopedDocument, ...]) -> DocumentBindingSet:
    return DocumentBindingSet(bindings=bindings)


def _make_query_analysis(work_type: str, domains: tuple[str, ...]) -> QueryAnalysis:
    return QueryAnalysis(
        contract_version="2.0",
        work_type=cast(Any, work_type),  # cast keeps the literal list narrow here
        domains=cast(Any, domains),
    )


def _make_route_decision(route: str, reason: str) -> RouteDecision:
    return RouteDecision(
        contract_version="2.0",
        route=cast(Any, route),
        reason=cast(Any, reason),
    )


def _make_runtime(
    user_id: str = "user-001",
    workspaces: tuple[str, ...] = ("ws-default",),
) -> GraphRuntimeContext:
    return GraphRuntimeContext(
        capability_runtime=CapabilityRuntimeContext(
            user_id=user_id,
            workspace_ids=workspaces,
        ),
        services=RuntimeServices(
            capabilities_registry_id="registry-001",
        ),
    )


def _initial_task_spec(task_id: str, capability: str, objective: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability=capability,
        task_objective=objective,
        input=_placeholder_input(),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def _placeholder_input() -> Any:
    """Build a valid CapabilityInput for fixture scenarios."""
    from .frozen_contracts import _PlaceholderCapabilityInput
    return _PlaceholderCapabilityInput(kind="placeholder")


def _placeholder_output() -> Any:
    from .frozen_contracts import _PlaceholderCapabilityOutput
    return _PlaceholderCapabilityOutput(kind="placeholder")


def _append_only_prefix(
    plan: TaskPlan,
    replan_tasks: tuple[TaskSpec, ...],
) -> TaskPlan:
    """Return a new plan that preserves the original tasks as a prefix and appends.

    Used by the append-only replan scenario to model the contract that new tasks
    extend — never replace — the existing plan.
    """
    return TaskPlan(
        contract_version="2.0",
        plan_id=plan.plan_id,
        goal=plan.goal,
        target_units=plan.target_units,
        tasks=plan.tasks + replan_tasks,
    )


# ---------------------------------------------------------------------------
# Scenario container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    description: str
    initial_state: SupervisorV2State
    runtime_context: GraphRuntimeContext
    expected_outcome: str

    def checkpoint_json(self) -> str:
        """Serialize only the checkpointable state — runtime is excluded.

        Uses `model_dump(mode="json")` to coerce Pydantic v2 models to JSON-safe
        dicts, then `json.dumps(..., sort_keys=True)` for deterministic byte
        ordering. Runtime context keys never appear because we only serialize
        `initial_state`.
        """
        payload: dict[str, Any] = {}
        for key, value in self.initial_state.items():
            if isinstance(value, ContractModel):
                payload[key] = value.model_dump(mode="json")
            elif value is None:
                payload[key] = None
            else:
                payload[key] = value
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 1. One-task fast plan
# ---------------------------------------------------------------------------

def _scenario_fast_one_task() -> Scenario:
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-fast-001",
        goal="Answer 'CCCD của A là gì'",
        target_units=(),
        tasks=(
            _initial_task_spec(
                task_id=_task(1),
                capability="people.lookup",
                objective="Resolve person A and return CCCD",
            ),
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("CCCD của A là gì", slot=1),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("people_lookup", ("people",)),
        "route_decision": _make_route_decision("fast_domain", "simple_people_lookup"),
        "execution": ExecutionState(plan=plan),
        "clarification": None,
        "final_response": None,
    }
    return Scenario(
        scenario_id="fast_one_task",
        description="Deterministic fast plan: one TaskSpec, no dependencies, no replan.",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="success:one_task",
    )


# ---------------------------------------------------------------------------
# 2. Three-task DAG (linear chain; fan-in is #3)
# ---------------------------------------------------------------------------

def _scenario_dag_three_tasks() -> Scenario:
    binding_a = ScopedDocument(
        document_id=_doc(11), document_revision="rev-A-1", role="primary",
    )
    binding_b = ScopedDocument(
        document_id=_doc(12), document_revision="rev-B-1", role="primary",
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-dag-001",
        goal="Compare Điều 5 of A and B",
        target_units=(),
        tasks=(
            _initial_task_spec(
                task_id=_task(11),
                capability="document.search",
                objective="Find both documents",
            ),
            _initial_task_spec(
                task_id=_task(12),
                capability="section.read",
                objective="Read Điều 5 of A",
            ).__class__(  # rebuild TaskSpec with dependency
                task_id=_task(12),
                capability="section.read",
                task_objective="Read Điều 5 of A",
                input=_placeholder_input(),
                depends_on=(_task(11),),
                origin=InitialTaskOrigin(kind="initial"),
            ),
            _initial_task_spec(
                task_id=_task(13),
                capability="section.read",
                objective="Read Điều 5 of B",
            ).__class__(
                task_id=_task(13),
                capability="section.read",
                task_objective="Read Điều 5 of B",
                input=_placeholder_input(),
                depends_on=(_task(11),),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("So sánh Điều 5 của A và B", slot=2),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings((binding_a, binding_b)),
        "query_analysis": _make_query_analysis("compare", ("section", "section")),
        "route_decision": _make_route_decision("complex_research", "comparison"),
        "execution": ExecutionState(plan=plan),
        "clarification": None,
        "final_response": None,
    }
    return Scenario(
        scenario_id="dag_three_tasks",
        description="Three-task DAG: search → 2 parallel reads.",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="success:fan_in",
    )


# ---------------------------------------------------------------------------
# 3. Async fan-in — the same TaskPlan shape as #2 but with one AgentResult
#    that proves fan-in is by `task_id`, not by arrival order.
# ---------------------------------------------------------------------------

def _scenario_async_fan_in() -> Scenario:
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-fanin-001",
        goal="Gather two independent reads then combine",
        target_units=(),
        tasks=(
            _initial_task_spec(
                task_id=_task(21),
                capability="document.search",
                objective="Locate source documents",
            ),
            _initial_task_spec(
                task_id=_task(22),
                capability="section.read",
                objective="Read A.5",
            ).__class__(
                task_id=_task(22),
                capability="section.read",
                task_objective="Read A.5",
                input=_placeholder_input(),
                depends_on=(_task(21),),
                origin=InitialTaskOrigin(kind="initial"),
            ),
            _initial_task_spec(
                task_id=_task(23),
                capability="section.read",
                objective="Read B.5",
            ).__class__(
                task_id=_task(23),
                capability="section.read",
                task_objective="Read B.5",
                input=_placeholder_input(),
                depends_on=(_task(21),),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )
    # task 22 completes AFTER task 23 — fan-in order must not matter.
    results = (
        AgentResult(
            contract_version="2.0",
            task_id=_task(23),
            status="success",
            data=_placeholder_output(),
            evidence_uses=(EvidenceUseRef(evidence_use_id=_uid(231)),),
        ),
        AgentResult(
            contract_version="2.0",
            task_id=_task(22),
            status="success",
            data=_placeholder_output(),
            evidence_uses=(EvidenceUseRef(evidence_use_id=_uid(221)),),
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("So sánh A.5 và B.5", slot=3),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("compare", ("section",)),
        "route_decision": _make_route_decision("complex_research", "comparison"),
        "execution": ExecutionState(plan=plan, task_results=results),
        "clarification": None,
        "final_response": None,
    }
    return Scenario(
        scenario_id="async_fan_in",
        description="Fan-in by task_id; arrival order must not matter.",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="success:fan_in_by_id",
    )


# ---------------------------------------------------------------------------
# 4. No-evidence `not_found`
# ---------------------------------------------------------------------------

def _scenario_no_evidence_not_found() -> Scenario:
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-nf-001",
        goal="Locate document by exact title",
        target_units=(),
        tasks=(
            _initial_task_spec(
                task_id=_task(31),
                capability="document.search",
                objective="Find document by title",
            ),
        ),
    )
    results = (
        AgentResult(
            contract_version="2.0",
            task_id=_task(31),
            status="not_found",
            data=None,
            evidence_uses=(),
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("Find document 'missing'", slot=4),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("document_search", ("document",)),
        "route_decision": _make_route_decision("fast_domain", "exact_document_metadata"),
        "execution": ExecutionState(plan=plan, task_results=results),
        "clarification": None,
        "final_response": None,
    }
    return Scenario(
        scenario_id="no_evidence_not_found",
        description="Capability returned `not_found` — empty evidence set, no error code.",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="not_found",
    )


# ---------------------------------------------------------------------------
# 5. No-evidence `TIMEOUT`
# ---------------------------------------------------------------------------

def _scenario_no_evidence_timeout() -> Scenario:
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-to-001",
        goal="Read a section that times out",
        target_units=(),
        tasks=(
            _initial_task_spec(
                task_id=_task(41),
                capability="section.read",
                objective="Read Điều 5",
            ),
        ),
    )
    results = (
        AgentResult(
            contract_version="2.0",
            task_id=_task(41),
            status="error",
            data=None,
            evidence_uses=(),
            error=AgentError(
                code="TIMEOUT",
                message="read exceeded 30s budget",
                retryable=True,
            ),
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("Đọc Điều 5 của A", slot=5),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("section_read", ("section",)),
        "route_decision": _make_route_decision("fast_domain", "exact_section_retrieval"),
        "execution": ExecutionState(plan=plan, task_results=results),
        "clarification": None,
        "final_response": None,
    }
    return Scenario(
        scenario_id="no_evidence_timeout",
        description="Capability returned `error` with code=TIMEOUT — distinct from `not_found`.",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="timeout",
    )


# ---------------------------------------------------------------------------
# 6. Append-only replan using TaskExecutionSummary
# ---------------------------------------------------------------------------

def _scenario_append_only_replan() -> Scenario:
    plan_initial = TaskPlan(
        contract_version="2.0",
        plan_id="plan-replan-001",
        goal="Investigate ambiguous person",
        target_units=(),
        tasks=(
            _initial_task_spec(
                task_id=_task(51),
                capability="people.lookup",
                objective="Resolve 'A'",
            ),
        ),
    )
    # First attempt: not_found. Replan adds one task. The replan is recorded
    # as a TaskSpec with ReplanTaskOrigin; the *summaries* drive the planner's
    # next attempt — not the underlying AgentResult.
    summary = TaskExecutionSummary(task_id=_task(51), status="not_found")
    replan_spec = TaskSpec(
        task_id=_task(52),
        capability="people.lookup",
        task_objective="Resolve 'A' with disambiguation hint",
        input=_placeholder_input(),
        depends_on=(),
        origin=ReplanTaskOrigin(
            kind="replan",
            reason="initial lookup returned not_found",
            task_ids=(_task(51),),
            evidence_use_ids=(),
        ),
    )
    plan_after = _append_only_prefix(plan_initial, (replan_spec,))
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("CCCD của A là gì", slot=6),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("people_lookup", ("people",)),
        "route_decision": _make_route_decision("complex_research", "evidence_replanning_required"),
        "execution": ExecutionState(plan=plan_after),
        "clarification": None,
        "final_response": None,
    }
    return Scenario(
        scenario_id="append_only_replan",
        description="Plan grows; original task stays as the prefix. Summary (51, not_found) drives planner.",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="replan:appended",
    ), summary


# ---------------------------------------------------------------------------
# 7. Clarification interrupt/resume
# ---------------------------------------------------------------------------

def _scenario_clarification_resume() -> Scenario:
    clarification = ClarificationRequest(
        contract_version="2.0",
        clarification_id="clarify-001",
        reason="required_document_ambiguous",
        question="Bạn muốn xem bản nào?",
        unresolved_ref_ids=("ref-A",),
        candidates=(
            DocumentCandidate(candidate_id="cand-1", document_id=_doc(71), label="Bản 2024"),
            DocumentCandidate(candidate_id="cand-2", document_id=_doc(72), label="Bản 2025"),
        ),
        expires_at=_FIXED_TS,
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("Xem Điều 5", slot=7),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("section_read", ("section",)),
        "route_decision": _make_route_decision("clarify", "essential_ambiguity"),
        "execution": ExecutionState(),
        "clarification": clarification,
        "final_response": None,
    }
    return Scenario(
        scenario_id="clarification_resume",
        description="Interrupt → resume with selected candidate.",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="clarify:resume",
    )


# ---------------------------------------------------------------------------
# 8. ACL runtime replacement (NEW workspace must REPLACE old)
# ---------------------------------------------------------------------------

def _scenario_acl_runtime_replaced() -> Scenario:
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-acl-001",
        goal="Resolve person in new workspace",
        target_units=(),
        tasks=(
            _initial_task_spec(
                task_id=_task(81),
                capability="people.lookup",
                objective="Lookup in current workspace",
            ),
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("A thuộc đơn vị nào", slot=8),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("kg_query", ("kg",)),
        "route_decision": _make_route_decision("fast_domain", "simple_kg_lookup"),
        "execution": ExecutionState(plan=plan),
        "clarification": None,
        "final_response": None,
    }
    return Scenario(
        scenario_id="acl_runtime_replaced",
        description="Resume swaps the runtime context — old workspaces must not appear.",
        initial_state=state,
        runtime_context=_make_runtime(user_id="user-002", workspaces=("ws-new",)),
        expected_outcome="success:acl_replaced",
    )


# ---------------------------------------------------------------------------
# 9. Cancellation — no later task dispatched
# ---------------------------------------------------------------------------

def _scenario_cancellation() -> Scenario:
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-cancel-001",
        goal="Two-step plan, cancelled after task 1",
        target_units=(),
        tasks=(
            _initial_task_spec(
                task_id=_task(91),
                capability="document.search",
                objective="Find document",
            ),
            _initial_task_spec(
                task_id=_task(92),
                capability="section.read",
                objective="Read section",
            ).__class__(
                task_id=_task(92),
                capability="section.read",
                task_objective="Read section",
                input=_placeholder_input(),
                depends_on=(_task(91),),
                origin=InitialTaskOrigin(kind="initial"),
            ),
        ),
    )
    # Only task 91 ran (and was cancelled). task 92 must never execute.
    results = (
        AgentResult(
            contract_version="2.0",
            task_id=_task(91),
            status="error",
            data=None,
            evidence_uses=(),
            error=AgentError(code="CANCELLED", message="user cancel", retryable=False),
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("Hủy", slot=9),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("document_search", ("document",)),
        "route_decision": _make_route_decision("fast_domain", "exact_document_metadata"),
        "execution": ExecutionState(plan=plan, task_results=results),
        "clarification": None,
        "final_response": None,
    }
    return Scenario(
        scenario_id="cancellation_no_later_task",
        description="Task 91 cancelled → task 92 must NOT have an AgentResult.",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="cancelled",
    )


# ---------------------------------------------------------------------------
# 10. Outer-only streaming — one terminal outer event per scenario
# ---------------------------------------------------------------------------

def _scenario_outer_only_streaming() -> Scenario:
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-stream-001",
        goal="Simple greet, no capability needed",
        target_units=(),
        tasks=(),
    )
    final = FinalResponse(
        contract_version="2.0",
        status="success",
        content="Xin chào!",
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("Xin chào", slot=10),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": None,
        "route_decision": _make_route_decision("direct", "direct_greeting"),
        "execution": ExecutionState(plan=plan),
        "clarification": None,
        "final_response": final,
    }
    return Scenario(
        scenario_id="outer_only_streaming",
        description="Direct route → exactly one terminal outer event (FinalResponse).",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="direct:one_terminal",
    )


# ---------------------------------------------------------------------------
# Collect all scenarios. The append-only-replan scenario carries an extra
# TaskExecutionSummary that downstream tests inspect via `_summaries`.
# ---------------------------------------------------------------------------

_SCENARIOS: list[Scenario] = []
_SUMMARIES: dict[str, TaskExecutionSummary] = {}


def _register(scenario: Scenario, *extras: Any) -> None:
    _SCENARIOS.append(scenario)
    for extra in extras:
        if isinstance(extra, TaskExecutionSummary):
            _SUMMARIES[scenario.scenario_id] = extra


_register(_scenario_fast_one_task())
_register(_scenario_dag_three_tasks())
_register(_scenario_async_fan_in())
_register(_scenario_no_evidence_not_found())
_register(_scenario_no_evidence_timeout())
_register(*_scenario_append_only_replan())
_register(_scenario_clarification_resume())
_register(_scenario_acl_runtime_replaced())
_register(_scenario_cancellation())
_register(_scenario_outer_only_streaming())


SCENARIOS: tuple[Scenario, ...] = tuple(_SCENARIOS)


def summary_for(scenario_id: str) -> TaskExecutionSummary | None:
    """Return the replan TaskExecutionSummary attached to a given scenario, if any."""
    return _SUMMARIES.get(scenario_id)

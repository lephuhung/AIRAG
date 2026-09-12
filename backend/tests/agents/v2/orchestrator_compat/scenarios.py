"""Deterministic Phase 0 parity scenarios.

Every `Scenario` carries the canonical `initial_state` (typed
`SupervisorV2State`), the `runtime_context` that pairs with it (request-scoped,
**not** checkpointed), and a `checkpoint_json()` helper that serializes only the
state — never the runtime. The 10 scenarios cover the brief's full inventory:

1. one-task fast plan               — `fast_one_task`
2. three-task DAG (real TargetUnits) — `dag_three_tasks`
3. async fan-in                      — `async_fan_in`
4. no-evidence `not_found`           — `no_evidence_not_found`
5. no-evidence `TIMEOUT`             — `no_evidence_timeout`
6. append-only replan (rejected on prefix violation) — `append_only_replan`
7. clarification interrupt/resume    — `clarification_resume`
8. changed ACL runtime replacement   — `acl-resume`  (literal id from brief Step 1)
9. cancellation                      — `cancellation_no_later_task`
10. outer-only streaming             — `outer_only_streaming` (direct ⇒ plan=None)

All fixtures are literal and deterministic: stable UUIDs from
`uuid.UUID(int=...)` and stable timestamps from `datetime(2026, 9, 11, …, tzinfo=UTC)`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .frozen_contracts import (
    AgentError,
    AgentResult,
    BindingRevisionRequirement,
    ClarificationRequest,
    ContractModel,
    ConversationContext,
    CoverageCriterion,
    CoverageObservation,
    DocumentBindingSet,
    DocumentCandidate,
    DocumentLocator,
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
    SectionLocator,
    SemanticContext,
    SupervisorV2State,
    TargetUnit,
    TaskExecutionSummary,
    TaskPlan,
    TaskSpec,
    InitialTaskOrigin,
    _Phase0CapabilityStandIn,
)


# ---------------------------------------------------------------------------
# Helpers — stable IDs and timestamps so scenarios are byte-for-byte deterministic.
# ---------------------------------------------------------------------------

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


def _placeholder_io() -> _Phase0CapabilityStandIn:
    return _Phase0CapabilityStandIn(kind="phase0_placeholder")


def _make_request(query: str, slot: int = 1) -> RequestContext:
    return RequestContext(
        contract_version="2.0",
        request_id=f"req-{slot:03d}",
        thread_id=f"thread-{slot:03d}",
        original_query=query,
        known_documents=(),
    )


def _make_conversation() -> ConversationContext:
    return ConversationContext(
        summary="",
        active_entities=(),
        last_focus=None,
        recent_turns=(),
    )


def _make_semantic() -> SemanticContext:
    return SemanticContext(
        contextualized_query="",
        normalized_query="",
        abbreviations=(),
        coreferences=(),
        document_refs=(),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def _make_bindings(
    bindings: tuple[ScopedDocument, ...],
    revision_refs: tuple[BindingRevisionRequirement, ...] = (),
) -> DocumentBindingSet:
    return DocumentBindingSet(
        bindings=bindings, revision_requirement_refs=revision_refs,
    )


def _make_query_analysis(
    work_type: str,
    domains: tuple[str, ...],
) -> QueryAnalysis:
    return QueryAnalysis(work_type=work_type, domains=domains)


def _make_route_decision(route: str, reason_code: str) -> RouteDecision:
    return RouteDecision(route=route, reason_code=reason_code)


def _make_runtime(
    user_id: str = "user-001",
    workspaces: tuple[str, ...] = ("ws-default",),
) -> GraphRuntimeContext:
    """Build the placeholder runtime for fixture scenarios.

    The real `CapabilityRuntimeContext` will be supplied by Phase 1D — its field
    shape is forbidden to redefine here, so we pass the placeholder.
    """
    return GraphRuntimeContext(
        capability_runtime=_placeholder_io(),
        services=RuntimeServices(capabilities_registry_id="registry-001"),
    )


def _initial_task_spec(
    task_id: str, capability: str, objective: str,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        capability=capability,
        task_objective=objective,
        input=_placeholder_io(),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )


def _task_spec_with_deps(
    task_id: str, capability: str, objective: str,
    depends_on: tuple[str, ...],
) -> TaskSpec:
    """Build a TaskSpec that depends on prior tasks."""
    return TaskSpec(
        task_id=task_id,
        capability=capability,
        task_objective=objective,
        input=_placeholder_io(),
        depends_on=depends_on,
        origin=InitialTaskOrigin(kind="initial"),
    )


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


def assert_append_only(old: TaskPlan, new: TaskPlan) -> None:
    """Spec §17 invariant: replan must preserve the original tasks as a prefix.

    Raises `ValueError` if the new plan's first `len(old.tasks)` slots are not
    `==` (by task_id) the old plan's. This is the parity check the brief's
    Step 3 mandates (final execution gate #20).
    """
    new_prefix_ids = [t.task_id for t in new.tasks[: len(old.tasks)]]
    old_ids = [t.task_id for t in old.tasks]
    if new_prefix_ids != old_ids:
        raise ValueError(
            f"append-only violation: expected prefix {old_ids}, "
            f"got {new_prefix_ids} in plan {new.plan_id}",
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
        "query_analysis": _make_query_analysis("lookup", ("people",)),
        "route_decision": _make_route_decision("fast_domain", "simple_people_lookup"),
        "execution": ExecutionState(
            plan=plan, task_results=(), evidence_evaluation=None,
        ),
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
# 2. Three-task DAG (linear chain) with REAL TargetUnits.
# ---------------------------------------------------------------------------

def _scenario_dag_three_tasks() -> Scenario:
    binding_a = ScopedDocument(
        binding_id=_binding(11),
        document_id=_doc(11),
        document_revision="rev-A-1",
        role="target",
    )
    binding_b = ScopedDocument(
        binding_id=_binding(12),
        document_id=_doc(12),
        document_revision="rev-B-1",
        role="target",
    )
    target_a = TargetUnit(
        target_id=_target(11),
        binding_id=_binding(11),
        requested_locator=SectionLocator(
            kind="section", structure_node_id="node-5-A",
        ),
        completion_criteria=(
            CoverageCriterion(
                kind="coverage",
                minimum_status="read_complete",
                allow_partial_reason=None,
            ),
        ),
    )
    target_b = TargetUnit(
        target_id=_target(12),
        binding_id=_binding(12),
        requested_locator=SectionLocator(
            kind="section", structure_node_id="node-5-B",
        ),
        completion_criteria=(
            CoverageCriterion(
                kind="coverage",
                minimum_status="read_complete",
                allow_partial_reason=None,
            ),
        ),
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-dag-001",
        goal="Compare Điều 5 of A and B",
        target_units=(target_a, target_b),
        tasks=(
            _initial_task_spec(
                task_id=_task(11),
                capability="document.search",
                objective="Find both documents",
            ),
            _task_spec_with_deps(
                task_id=_task(12),
                capability="section.read",
                objective="Read Điều 5 of A",
                depends_on=(_task(11),),
            ),
            _task_spec_with_deps(
                task_id=_task(13),
                capability="section.read",
                objective="Read Điều 5 of B",
                depends_on=(_task(11),),
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
        "execution": ExecutionState(
            plan=plan, task_results=(), evidence_evaluation=None,
        ),
        "clarification": None,
        "final_response": None,
    }
    return Scenario(
        scenario_id="dag_three_tasks",
        description="Three-task DAG: search → 2 parallel reads (real TargetUnits).",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="success:fan_in",
    )


# ---------------------------------------------------------------------------
# 3. Async fan-in — same plan shape as #2 but with AgentResult order jumbled.
# ---------------------------------------------------------------------------

def _scenario_async_fan_in() -> Scenario:
    binding_a = ScopedDocument(
        binding_id=_binding(21),
        document_id=_doc(21),
        document_revision="rev-A-1",
        role="target",
    )
    target_a = TargetUnit(
        target_id=_target(21),
        binding_id=_binding(21),
        requested_locator=SectionLocator(
            kind="section", structure_node_id="node-5-A",
        ),
        completion_criteria=(CoverageCriterion(kind="coverage"),),
    )
    binding_b = ScopedDocument(
        binding_id=_binding(22),
        document_id=_doc(22),
        document_revision="rev-B-1",
        role="target",
    )
    target_b = TargetUnit(
        target_id=_target(22),
        binding_id=_binding(22),
        requested_locator=SectionLocator(
            kind="section", structure_node_id="node-5-B",
        ),
        completion_criteria=(CoverageCriterion(kind="coverage"),),
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-fanin-001",
        goal="Gather two independent reads then combine",
        target_units=(target_a, target_b),
        tasks=(
            _initial_task_spec(
                task_id=_task(21),
                capability="document.search",
                objective="Locate source documents",
            ),
            _task_spec_with_deps(
                task_id=_task(22),
                capability="section.read",
                objective="Read A.5",
                depends_on=(_task(21),),
            ),
            _task_spec_with_deps(
                task_id=_task(23),
                capability="section.read",
                objective="Read B.5",
                depends_on=(_task(21),),
            ),
        ),
    )
    # task 22 completes AFTER task 23 — fan-in order must not matter.
    results = (
        AgentResult(
            contract_version="2.0",
            task_id=_task(23),
            status="success",
            data=_placeholder_io(),
            evidence_uses=(EvidenceUseRef(use_id=_uid(231)),),
            coverage_observations=(
                CoverageObservation(
                    target_id=_target(22),
                    observed_locators=(
                        SectionLocator(kind="section", structure_node_id="node-5-B"),
                    ),
                    outcome="read",
                ),
            ),
            error=None,
        ),
        AgentResult(
            contract_version="2.0",
            task_id=_task(22),
            status="success",
            data=_placeholder_io(),
            evidence_uses=(EvidenceUseRef(use_id=_uid(221)),),
            coverage_observations=(
                CoverageObservation(
                    target_id=_target(21),
                    observed_locators=(
                        SectionLocator(kind="section", structure_node_id="node-5-A"),
                    ),
                    outcome="read",
                ),
            ),
            error=None,
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("So sánh A.5 và B.5", slot=3),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings((binding_a, binding_b)),
        "query_analysis": _make_query_analysis("compare", ("section",)),
        "route_decision": _make_route_decision("complex_research", "comparison"),
        "execution": ExecutionState(
            plan=plan, task_results=results, evidence_evaluation=None,
        ),
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
    binding = ScopedDocument(
        binding_id=_binding(31),
        document_id=_doc(31),
        document_revision="rev-1",
        role="target",
    )
    target = TargetUnit(
        target_id=_target(31),
        binding_id=_binding(31),
        requested_locator=DocumentLocator(kind="document"),
        completion_criteria=(CoverageCriterion(kind="coverage"),),
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-nf-001",
        goal="Locate document by exact title",
        target_units=(target,),
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
            coverage_observations=(),
            error=None,
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("Find document 'missing'", slot=4),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings((binding,)),
        "query_analysis": _make_query_analysis("retrieve", ("document",)),
        "route_decision": _make_route_decision("fast_domain", "exact_document_metadata"),
        "execution": ExecutionState(
            plan=plan, task_results=results, evidence_evaluation=None,
        ),
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
    binding = ScopedDocument(
        binding_id=_binding(41),
        document_id=_doc(41),
        document_revision="rev-1",
        role="target",
    )
    target = TargetUnit(
        target_id=_target(41),
        binding_id=_binding(41),
        requested_locator=SectionLocator(
            kind="section", structure_node_id="node-5",
        ),
        completion_criteria=(CoverageCriterion(kind="coverage"),),
    )
    plan = TaskPlan(
        contract_version="2.0",
        plan_id="plan-to-001",
        goal="Read a section that times out",
        target_units=(target,),
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
            coverage_observations=(),
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
        "bindings": _make_bindings((binding,)),
        "query_analysis": _make_query_analysis("retrieve", ("section",)),
        "route_decision": _make_route_decision("fast_domain", "exact_section_retrieval"),
        "execution": ExecutionState(
            plan=plan, task_results=results, evidence_evaluation=None,
        ),
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
# 6. Append-only replan using TaskExecutionSummary.
#    Includes a deliberately-prefix-violating plan for parity test I2.
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
    summary = TaskExecutionSummary(task_id=_task(51), status="not_found")
    replan_spec = TaskSpec(
        task_id=_task(52),
        capability="people.lookup",
        task_objective="Resolve 'A' with disambiguation hint",
        input=_placeholder_io(),
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
        "query_analysis": _make_query_analysis("lookup", ("people",)),
        "route_decision": _make_route_decision(
            "complex_research", "evidence_replanning_required",
        ),
        "execution": ExecutionState(
            plan=plan_after, task_results=(), evidence_evaluation=None,
        ),
        "clarification": None,
        "final_response": None,
    }
    # Prefix-violating plan: replaces task-051 with task-052. Used by I2.
    violating_replan = TaskPlan(
        contract_version="2.0",
        plan_id=plan_initial.plan_id,
        goal=plan_initial.goal,
        target_units=plan_initial.target_units,
        tasks=(
            TaskSpec(
                task_id=_task(52),
                capability="people.lookup",
                task_objective="Resolve 'A' with disambiguation hint",
                input=_placeholder_io(),
                depends_on=(),
                origin=ReplanTaskOrigin(
                    kind="replan",
                    reason="prefix-violating replan",
                    task_ids=(_task(51),),
                    evidence_use_ids=(),
                ),
            ),
        ),
    )
    return Scenario(
        scenario_id="append_only_replan",
        description="Plan grows; original task stays as the prefix. "
                    "Summary (51, not_found) drives planner.",
        initial_state=state,
        runtime_context=_make_runtime(),
        expected_outcome="replan:appended",
    ), summary, plan_initial, violating_replan


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
            DocumentCandidate(
                candidate_id="cand-1", ordinal=0, ref_id="ref-A",
                document_id=_doc(71), label="Bản 2024",
            ),
            DocumentCandidate(
                candidate_id="cand-2", ordinal=1, ref_id="ref-A",
                document_id=_doc(72), label="Bản 2025",
            ),
        ),
        expires_at=_FIXED_TS,
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("Xem Điều 5", slot=7),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("retrieve", ("section",)),
        "route_decision": _make_route_decision("clarify", "essential_ambiguity"),
        "execution": ExecutionState(
            plan=None, task_results=(), evidence_evaluation=None,
        ),
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
# 8. ACL runtime replacement (NEW workspace must REPLACE old).
#    Brief Step 1 references literal id `acl-resume`.
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
                capability="knowledge_graph.query",
                objective="Lookup person A's unit in current workspace",
            ),
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("A thuộc đơn vị nào", slot=8),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("lookup", ("knowledge_graph",)),
        "route_decision": _make_route_decision("fast_domain", "simple_kg_lookup"),
        "execution": ExecutionState(
            plan=plan, task_results=(), evidence_evaluation=None,
        ),
        "clarification": None,
        "final_response": None,
    }
    return Scenario(
        scenario_id="acl-resume",
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
            _task_spec_with_deps(
                task_id=_task(92),
                capability="section.read",
                objective="Read section",
                depends_on=(_task(91),),
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
            coverage_observations=(),
            error=AgentError(code="CANCELLED", message="user cancel", retryable=False),
        ),
    )
    state: SupervisorV2State = {
        "contract_version": "2.0",
        "request": _make_request("Hủy", slot=9),
        "conversation": _make_conversation(),
        "semantic": _make_semantic(),
        "bindings": _make_bindings(()),
        "query_analysis": _make_query_analysis("retrieve", ("document",)),
        "route_decision": _make_route_decision("fast_domain", "exact_document_metadata"),
        "execution": ExecutionState(
            plan=plan, task_results=results, evidence_evaluation=None,
        ),
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
# 10. Outer-only streaming — direct route, plan=None per spec §12.
# ---------------------------------------------------------------------------

def _scenario_outer_only_streaming() -> Scenario:
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
        "execution": ExecutionState(
            plan=None, task_results=(), evidence_evaluation=None,
        ),
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
# TaskExecutionSummary, the original plan, and a prefix-violating plan that
# downstream parity tests inspect.
# ---------------------------------------------------------------------------

_SCENARIOS: list[Scenario] = []
_SUMMARIES: dict[str, TaskExecutionSummary] = {}
_PLANS: dict[str, TaskPlan] = {}
_VIOLATING_PLANS: dict[str, TaskPlan] = {}


def _register(
    scenario: Scenario,
    summary: TaskExecutionSummary | None = None,
    original_plan: TaskPlan | None = None,
    violating_plan: TaskPlan | None = None,
) -> None:
    _SCENARIOS.append(scenario)
    if summary is not None:
        _SUMMARIES[scenario.scenario_id] = summary
    if original_plan is not None:
        _PLANS[scenario.scenario_id] = original_plan
    if violating_plan is not None:
        _VIOLATING_PLANS[scenario.scenario_id] = violating_plan


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


def original_plan_for(scenario_id: str) -> TaskPlan | None:
    """Return the *original* (pre-replan) plan attached to a given scenario."""
    return _PLANS.get(scenario_id)


def violating_plan_for(scenario_id: str) -> TaskPlan | None:
    """Return a deliberately-prefix-violating plan for a given scenario."""
    return _VIOLATING_PLANS.get(scenario_id)

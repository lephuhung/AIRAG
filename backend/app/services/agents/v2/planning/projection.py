"""Minimized/redacted model projection for the governed planner (Task 10).

``ResearchPlanningInput`` is the authoritative runtime envelope (frozen
contract, full identity). The model sees only :class:`PlannerModelInput` —
a runtime-only projection carrying the query, work type, binding IDs + roles,
person/section labels, catalog names, discovery flags, and budget numbers.
Trusted identity (document/revision UUIDs), internal evidence UUIDs, governed
people scalars, and raw evidence never cross this boundary.
"""
from __future__ import annotations

from ..contracts.base import RuntimeModel
from ..contracts.planning import ResearchPlanningInput

__all__ = [
    "PlannerBindingRef",
    "PlannerBudgetView",
    "PlannerCapabilityRef",
    "PlannerDiscoveryView",
    "PlannerModelInput",
    "PlannerPersonRef",
    "PlannerSectionRef",
    "build_planner_model_input",
]


class PlannerBindingRef(RuntimeModel):
    """One plannable binding: server ID + role only, never document identity."""

    binding_id: str
    role: str


class PlannerPersonRef(RuntimeModel):
    """One person mention from the user's own query text (label only)."""

    ref_id: str
    label: str


class PlannerSectionRef(RuntimeModel):
    """One section mention from the user's own query text (label only)."""

    ref_id: str
    label: str


class PlannerCapabilityRef(RuntimeModel):
    """One catalog entry the planner may propose (already minimal)."""

    name: str
    domain: str
    operation_type: str
    supports_parallel: bool


class PlannerDiscoveryView(RuntimeModel):
    """Discovery flags the planner must respect (restrict, never grant)."""

    allow_reference_discovery: bool
    allow_supporting_discovery: bool
    max_discovered_documents: int


class PlannerBudgetView(RuntimeModel):
    """Budget numbers the planner must respect (scheduler still enforces)."""

    max_tasks_remaining: int
    max_replans_remaining: int
    max_parallel_branches: int


class PlannerModelInput(RuntimeModel):
    """Runtime-only minimized/redacted planner input (never checkpointed)."""

    query: str
    work_type: str
    domains: tuple[str, ...]
    bindings: tuple[PlannerBindingRef, ...]
    person_refs: tuple[PlannerPersonRef, ...]
    section_refs: tuple[PlannerSectionRef, ...]
    capability_catalog: tuple[PlannerCapabilityRef, ...]
    discovery_policy: PlannerDiscoveryView
    budget: PlannerBudgetView


def build_planner_model_input(planning_input: ResearchPlanningInput) -> PlannerModelInput:
    """Project the authoritative envelope to the minimized model input.

    Allowlist projection: binding IDs + roles (no document/revision UUIDs),
    person/section ref IDs + user-supplied labels (no resolved identity),
    catalog names/domains (no registry internals), policy flags + budget
    numbers. Initial planning carries no current plan, outcomes, or evidence,
    so there is nothing further to redact here; any future replan projection
    MUST route through ``redact_scalar_for_model`` first.
    """
    semantic = planning_input.semantic
    return PlannerModelInput(
        query=semantic.contextualized_query,
        work_type=str(planning_input.query_analysis.work_type),
        domains=tuple(str(domain) for domain in planning_input.query_analysis.domains),
        bindings=tuple(
            PlannerBindingRef(binding_id=binding.binding_id, role=str(binding.role))
            for binding in planning_input.bindings.bindings
        ),
        person_refs=tuple(
            PlannerPersonRef(ref_id=reference.ref_id, label=reference.label)
            for reference in semantic.person_refs
        ),
        section_refs=tuple(
            PlannerSectionRef(ref_id=reference.ref_id, label=reference.label)
            for reference in semantic.section_refs
        ),
        capability_catalog=tuple(
            PlannerCapabilityRef(
                name=entry.name,
                domain=str(entry.domain),
                operation_type=str(entry.operation_type),
                supports_parallel=bool(entry.supports_parallel),
            )
            for entry in planning_input.capability_catalog
        ),
        discovery_policy=PlannerDiscoveryView(
            allow_reference_discovery=bool(
                planning_input.discovery_policy.allow_reference_discovery
            ),
            allow_supporting_discovery=bool(
                planning_input.discovery_policy.allow_supporting_discovery
            ),
            max_discovered_documents=int(
                planning_input.discovery_policy.max_discovered_documents
            ),
        ),
        budget=PlannerBudgetView(
            max_tasks_remaining=int(planning_input.budget.max_tasks_remaining),
            max_replans_remaining=int(planning_input.budget.max_replans_remaining),
            max_parallel_branches=int(planning_input.budget.max_parallel_branches),
        ),
    )

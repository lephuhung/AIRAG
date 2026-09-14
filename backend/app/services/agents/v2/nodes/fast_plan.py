"""Deterministic fast-plan builder + node (Phase 2, Task 3).

``build_fast_plan`` maps one ``fast_domain`` route to one checkpointed
``TaskPlan`` with a single initial task — no planner model is called. People
and knowledge-graph lookups carry zero target units; document/section reads
carry one target unit per required pin (document) or per named section on the
single pin (section), and the single task input references exactly those
targets. Reference-free factual retrieval carries zero target units with a
targetless ``document.retrieve`` task over the workspace scope. ``fast_plan_node`` checkpoints that plan via ``reset_execution``
(clearing stale results and evaluation: a fast plan is initial planning,
which receives no outcomes) before any capability is dispatched. Follows the
Task-1 node-injection convention via ``_context_of``.
"""
from __future__ import annotations

import hashlib

from langgraph.runtime import Runtime

from ..adapters.document import binding_id_for_ref
from ..contracts.base import CONTRACT_VERSION
from ..contracts.binding import DocumentBindingSet
from ..contracts.capability import (
    CapabilityInput,
    DocumentReadInput,
    DocumentRetrieveInput,
    KnowledgeGraphInput,
    PeopleLookupInput,
    SectionReadInput,
)
from ..contracts.locators import DocumentLocator, SectionLocator
from ..contracts.planning import (
    CoverageCriterion,
    InitialTaskOrigin,
    TargetUnit,
    TaskPlan,
    TaskSpec,
)
from ..contracts.routing import QueryAnalysis, RouteDecision
from ..contracts.semantic import SemanticContext
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.validation import validate_fast_plan, validate_task_plan
from .context import _context_of
from .execute import reset_execution

__all__ = [
    "FastPlanError",
    "build_fast_plan",
    "fast_plan_node",
]


class FastPlanError(ValueError):
    """The fast plan cannot be built from this routing outcome."""


#: Router reason code -> the one shared capability serving the fast route.
#: Bounded summaries arrive here as document/section reads (the router maps a
#: one-pin summary to exact_document_metadata); there is no summary agent.
#: Reference-free factual retrieval arrives as ``document.retrieve`` (the
#: router maps unpinned retrieve/document to targetless_document_retrieval).
_FAST_CAPABILITY_FOR_REASON = {
    "simple_people_lookup": "people.lookup",
    "exact_document_metadata": "document.read",
    "exact_section_retrieval": "section.read",
    "simple_kg_lookup": "knowledge_graph.query",
    "targetless_document_retrieval": "document.retrieve",
}

#: Fast capabilities whose plans carry zero target units.
_TARGETLESS_CAPABILITIES = frozenset({"people.lookup", "knowledge_graph.query"})


def _stable_task_id(normalized_query: str, reason_code: str, index: int) -> str:
    digest = hashlib.sha256(
        f"{normalized_query}\x00{reason_code}\x00{index}".encode("utf-8")
    ).hexdigest()[:12]
    return f"fast_{reason_code}_{index}_{digest}"


def _stable_plan_id(task_id: str) -> str:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    return f"plan_{digest}"


def _current_target_pins(
    semantic: SemanticContext, bindings: DocumentBindingSet
) -> list:
    """Pins referenced by the current semantic projection (stale pins excluded).

    Ordering is by binding id so re-delivery converges on identical plans.
    """
    by_id = {binding.binding_id: binding for binding in bindings.bindings}
    wanted = {binding_id_for_ref(reference.ref_id) for reference in semantic.document_refs}
    return [
        by_id[binding_id]
        for binding_id in sorted(wanted)
        if binding_id in by_id
        and by_id[binding_id].role in ("target", "reference")
    ]


def _coverage_criteria() -> tuple[CoverageCriterion, ...]:
    return (CoverageCriterion(kind="coverage"),)


def _build_required_targets(
    semantic: SemanticContext,
    bindings: DocumentBindingSet,
    capability: str,
) -> tuple[TargetUnit, ...]:
    if capability in _TARGETLESS_CAPABILITIES:
        return ()
    pins = _current_target_pins(semantic, bindings)
    if capability == "document.retrieve":
        # Reference-free factual RAG: no pins means a targetless task over
        # the workspace scope (empty ``target_ids`` by contract). Pinned
        # targets, if ever present under this reason, get one unit per pin
        # exactly like the pinned read path; the router only emits this
        # reason when nothing is pinned.
        return tuple(
            TargetUnit(
                target_id=f"t_{binding.binding_id}",
                binding_id=binding.binding_id,
                requested_locator=DocumentLocator(kind="document"),
                completion_criteria=_coverage_criteria(),
            )
            for binding in pins
        )
    if capability == "document.read":
        if not pins:
            raise FastPlanError(
                "document fast plan requires a pinned target binding"
            )
        return tuple(
            TargetUnit(
                target_id=f"t_{binding.binding_id}",
                binding_id=binding.binding_id,
                requested_locator=DocumentLocator(kind="document"),
                completion_criteria=_coverage_criteria(),
            )
            for binding in pins
        )
    if capability == "section.read":
        if len(pins) != 1:
            raise FastPlanError(
                "section fast plan requires exactly one pinned binding, "
                f"got {len(pins)}"
            )
        located = [
            reference
            for reference in semantic.section_refs
            if reference.structure_node_id
        ]
        if not located:
            raise FastPlanError(
                "section fast plan requires a section coordinate "
                "(structure_node_id)"
            )
        binding = pins[0]
        return tuple(
            TargetUnit(
                target_id=f"t_{binding.binding_id}_s{index}",
                binding_id=binding.binding_id,
                requested_locator=SectionLocator(
                    kind="section",
                    structure_node_id=reference.structure_node_id or "",
                ),
                completion_criteria=_coverage_criteria(),
            )
            for index, reference in enumerate(located)
        )
    raise FastPlanError(f"fast plan has no targets for capability {capability!r}")


def _capability_input(
    capability: str,
    targets: tuple[TargetUnit, ...],
    semantic: SemanticContext,
) -> CapabilityInput:
    if capability == "people.lookup":
        return PeopleLookupInput(kind="people.lookup", query=semantic.normalized_query)
    if capability == "knowledge_graph.query":
        return KnowledgeGraphInput(
            kind="knowledge_graph.query", query=semantic.normalized_query
        )
    target_ids = tuple(unit.target_id for unit in targets)
    if capability == "document.retrieve":
        return DocumentRetrieveInput(
            kind="document.retrieve",
            query=semantic.normalized_query,
            target_ids=target_ids,
            # Must match skills/retrieve/policy.py::RETRIEVE_TOP_K (kept as
            # a literal so nodes stay decoupled from skill policy modules).
            top_k=8,
        )
    if capability == "document.read":
        return DocumentReadInput(kind="document.read", target_ids=target_ids)
    if capability == "section.read":
        return SectionReadInput(kind="section.read", target_ids=target_ids)
    raise FastPlanError(f"fast plan has no input for capability {capability!r}")


def build_fast_plan(
    semantic: SemanticContext,
    bindings: DocumentBindingSet,
    analysis: QueryAnalysis,
    route: RouteDecision,
) -> TaskPlan:
    """Build the deterministic single-task plan for a ``fast_domain`` route.

    ``analysis`` is accepted (it is the router's evidence for the route) but
    the capability mapping is owned by the route's reason code, so a bounded
    summary arrives as a document/section read and a comparison never arrives
    here at all. Raises ``FastPlanError`` (a ``ValueError``) unless the route
    is ``fast_domain`` with a known fast reason code. The plan is validated
    before it is returned; no planner model is called.
    """
    if route.route != "fast_domain":
        raise FastPlanError(
            f"fast plan requires fast_domain route, got {route.route!r}"
        )
    capability = _FAST_CAPABILITY_FOR_REASON.get(route.reason_code)
    if capability is None:
        raise FastPlanError(
            f"fast plan has no capability for reason {route.reason_code!r}"
        )
    task_id = _stable_task_id(semantic.normalized_query, route.reason_code, 0)
    targets = _build_required_targets(semantic, bindings, capability)
    task = TaskSpec(
        task_id=task_id,
        capability=capability,
        task_objective=semantic.normalized_query,
        input=_capability_input(capability, targets, semantic),
        depends_on=(),
        origin=InitialTaskOrigin(kind="initial"),
    )
    plan = TaskPlan(
        contract_version=CONTRACT_VERSION,
        plan_id=_stable_plan_id(task_id),
        goal=semantic.normalized_query,
        target_units=targets,
        tasks=(task,),
    )
    validate_task_plan(plan, bindings)
    validate_fast_plan(plan, bindings)
    return plan


async def fast_plan_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Checkpoint the deterministic fast plan before any dispatch."""
    _context_of(runtime)
    analysis = state["query_analysis"]
    route = state["route_decision"]
    if analysis is None or route is None:
        raise FastPlanError(
            "fast plan requires the checkpointed query analysis and route decision"
        )
    plan = build_fast_plan(state["semantic"], state["bindings"], analysis, route)
    return reset_execution(plan)

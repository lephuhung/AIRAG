"""Minimal query analysis and route decision (spec §12).

Analysis is deterministic-first and carries only semantic facts. Capability
mapping, answer policy, and semantic-complexity telemetry stay with the
router/planner/answer layers, so none of them appear here.
"""
from __future__ import annotations

from typing import Literal

from .base import ContractModel

WorkType = Literal[
    "direct",
    "lookup",
    "retrieve",
    "explain",
    "summarize",
    "compare",
    "evaluate",
    "cross_domain",
    "multi_goal",
    # Multi-intent routing (spec §33.3): emitted only by the flag-on
    # semantic path; never produced while V2_MULTI_INTENT_ROUTING_ENABLED
    # is off, so flag-off wire shapes stay byte-identical.
    "multi_intent",
]

Domain = Literal["people", "document", "section", "write", "knowledge_graph", "memory"]

Route = Literal["direct", "clarify", "fast_domain", "complex_research"]

RouteReason = Literal[
    "direct_greeting",
    "direct_conversation",
    "essential_ambiguity",
    "unresolved_required_binding",
    "simple_people_lookup",
    "exact_document_metadata",
    "exact_section_retrieval",
    # Phase 4A Task 4 ruling: reference-free factual retrieval without an
    # explicit document binding is a bounded targetless workspace retrieval
    # fast path (``document.retrieve`` with empty ``target_ids``), closing
    # the ``v2_reason_proposal`` deferred by the Task-1 parity corpus.
    "targetless_document_retrieval",
    "simple_write_operation",
    "simple_kg_lookup",
    "multi_document_research",
    "cross_domain_dependency",
    "comparison",
    "compliance_evaluation",
    "multi_goal",
    "runtime_dependency",
    "evidence_replanning_required",
    # Multi-intent routing (spec §33.2/§33.3): emitted only by the flag-on
    # semantic path; never produced while V2_MULTI_INTENT_ROUTING_ENABLED
    # is off, so flag-off wire shapes stay byte-identical.
    "multi_intent",
    "semantic_uncertainty",
]


class SemanticDependencyHint(ContractModel):
    """Spec §12: a semantic hint that one domain depends on another.

    The shape is not fully defined by the spec; the minimal hint the analyzer
    emits is an identity plus the dependency it observed.
    """

    hint_id: str
    description: str


class QueryAnalysis(ContractModel):
    """Spec §12: deterministic-first analysis of the finalized query."""

    work_type: WorkType
    domains: tuple[Domain, ...]
    dependency_hints: tuple[SemanticDependencyHint, ...] = ()


class RouteDecision(ContractModel):
    """Spec §12: execution topology chosen by the deterministic router."""

    route: Route
    reason_code: RouteReason

"""Deterministic-first routing node (Phase 2, Task 1).

``analyze_query`` derives semantic facts (work type, domains, dependency hints)
from the finalized query meaning; ``decide_route`` maps those facts to one of
the four frozen routes (``direct`` / ``clarify`` / ``fast_domain`` /
``complex_research``). Domain business logic stays outside the node, and no
domain-agent name ever appears as a route.

Routing table::

    direct non-factual        -> direct
    required ambiguity        -> clarify
    one bounded operation     -> fast_domain
    bounded one-doc summary   -> fast_domain
    two-target comparison     -> complex_research
    cross-domain dependency   -> complex_research
    multi-goal/iterative RAG  -> complex_research
    Write                     -> typed unavailable (WriteUnavailableError)

Routing may inspect request-scoped capability availability (via the injected
``allowed_capabilities``); availability never reaches business state. Write has
no Phase-2 capability module, so it always raises the typed unavailable error —
a dedicated ``CapabilityUnavailable`` subtype, never a route string.
"""
from __future__ import annotations

import re
from typing import Any

from langgraph.runtime import Runtime

from ..capabilities import CapabilityUnavailable
from ..contracts.binding import DocumentBindingSet
from ..contracts.routing import QueryAnalysis, RouteDecision, SemanticDependencyHint
from ..contracts.semantic import SemanticContext
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.validation import validate_query_analysis
from .context import _context_of

__all__ = [
    "WriteUnavailableError",
    "analyze_query",
    "decide_route",
    "route_node",
]


class WriteUnavailableError(CapabilityUnavailable):
    """Write is requested but no Write capability exists in Phase 2."""


_GREETING_RE = re.compile(
    r"^(xin chào|chào|hello|hi|hey|good morning|good afternoon)\b"
)
_CONVERSATION_RE = re.compile(
    r"^(cảm ơn|cám ơn|thank|thanks|tạm biệt|bye|goodbye|ok|okay|ừ|ừm|vâng|dạ)\b"
)
_WRITE_RES = (
    re.compile(r"\bviết\b"),
    re.compile(r"\bsoạn\b"),
    re.compile(r"soạn thảo"),
    re.compile(r"\bwrite\b"),
    re.compile(r"\bdraft\b"),
    re.compile(r"\bcompose\b"),
    re.compile(r"tạo báo cáo"),
    re.compile(r"lập báo cáo"),
)
_READ_VERB_RE = re.compile(r"\b(đọc|xem|tìm|tra cứu|read|find|search|show)\b")
_COMPARE_RES = (
    "so sánh",
    "compare",
    "comparison",
    "đối chiếu",
    "khác nhau",
    "khác biệt",
)
_SUMMARY_RES = ("tóm tắt", "summar", "tóm lược", "tổng quan", "tổng hợp")
_EVALUATE_RES = ("đánh giá", "tuân thủ", "compliance", "evaluate", "evaluation")
_KG_RES = (
    "thuộc đơn vị",
    "đơn vị nào",
    "thuộc phòng",
    "thuộc ban",
    "thuộc về",
    "là ai",
    "mối quan hệ",
    "who is",
    "who was",
    "reports to",
)

#: Collapsed families used for dependency counting: sections live inside
#: documents (spec §10), so {document, section} is one family.
_FAMILY = {"section": "document", "document": "document"}
_DEPENDENCY_FAMILIES = ("people", "document", "knowledge_graph")

_FAST_CAPABILITY = {
    "people": "people.lookup",
    "document": "document.read",
    "section": "section.read",
    "knowledge_graph": "knowledge_graph.query",
}


def _contains(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _write_intent(text: str) -> bool:
    if _READ_VERB_RE.search(text):
        return False
    return any(pattern.search(text) for pattern in _WRITE_RES)


def analyze_query(
    semantic: SemanticContext, bindings: DocumentBindingSet
) -> QueryAnalysis:
    """Deterministic-first analysis of the finalized query meaning."""
    text = semantic.normalized_query.casefold()
    domains: set[str] = set()
    if semantic.person_refs:
        domains.add("people")
    if semantic.document_refs:
        domains.add("document")
    if semantic.section_refs:
        domains.add("section")
    write = _write_intent(text)
    if write:
        domains.add("write")
    compare = _contains(text, _COMPARE_RES) or len(semantic.document_refs) >= 2
    summary = _contains(text, _SUMMARY_RES)
    evaluate = _contains(text, _EVALUATE_RES)
    if _contains(text, _KG_RES):
        domains.add("knowledge_graph")

    families = {_FAMILY.get(domain, domain) for domain in domains}
    dependency_families = sorted(set(_DEPENDENCY_FAMILIES) & families)
    hints = tuple(
        SemanticDependencyHint(
            hint_id=f"dep-{index + 1}",
            description=f"{first}->{second} dependency",
        )
        for index, (first, second) in enumerate(
            zip(dependency_families, dependency_families[1:])
        )
    )
    multi_goal = (compare and summary) or len(semantic.document_refs) >= 3

    if len(dependency_families) >= 2:
        work_type = "cross_domain"
    elif multi_goal:
        work_type = "multi_goal"
    elif compare:
        work_type = "compare"
    elif evaluate:
        work_type = "evaluate"
    elif summary:
        work_type = "summarize"
    elif domains <= {"people"} or domains == {"knowledge_graph"} or not domains:
        work_type = "lookup"
    else:
        work_type = "retrieve"

    if not domains:
        # No semantic signal at all: conversational greeting or a ref-less
        # factual query that later stages resolve via discovery.
        domains = {"memory"} if _is_conversational(text) else {"document"}

    analysis = QueryAnalysis(
        work_type=work_type,  # type: ignore[arg-type]
        domains=tuple(sorted(domains)),  # type: ignore[arg-type]
        dependency_hints=hints,
    )
    validate_query_analysis(analysis)
    return analysis


def _is_conversational(text: str) -> bool:
    return bool(_GREETING_RE.search(text) or _CONVERSATION_RE.search(text))


def _is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.search(text))


def _has_open_required_reference(semantic: SemanticContext) -> bool:
    return any(
        reference.resolution_status != "resolved"
        for reference in semantic.document_refs
    )


def _fast_or_runtime_dependency(
    domain: str, allowed_capabilities: frozenset[str], reason_code: Any
) -> RouteDecision:
    if _FAST_CAPABILITY[domain] in allowed_capabilities:
        return RouteDecision(route="fast_domain", reason_code=reason_code)
    return RouteDecision(route="complex_research", reason_code="runtime_dependency")


def decide_route(
    analysis: QueryAnalysis,
    semantic: SemanticContext,
    bindings: DocumentBindingSet,
    *,
    allowed_capabilities: frozenset[str] = frozenset(),
) -> RouteDecision:
    """Map deterministic analysis facts to one frozen route."""
    if "write" in analysis.domains:
        raise WriteUnavailableError(
            "write is unavailable: Phase 2 ships no Write capability module"
        )
    if semantic.blocking_ambiguities:
        return RouteDecision(route="clarify", reason_code="essential_ambiguity")
    if _has_open_required_reference(semantic):
        return RouteDecision(route="clarify", reason_code="unresolved_required_binding")

    text = semantic.normalized_query.casefold()
    has_refs = bool(semantic.document_refs or semantic.person_refs or semantic.section_refs)
    if _is_conversational(text) and not has_refs:
        reason = "direct_greeting" if _is_greeting(text) else "direct_conversation"
        return RouteDecision(route="direct", reason_code=reason)  # type: ignore[arg-type]

    bound_count = len(bindings.bindings)
    if analysis.work_type == "lookup" and analysis.domains == ("people",):
        return _fast_or_runtime_dependency("people", allowed_capabilities, "simple_people_lookup")
    if (
        analysis.work_type == "retrieve"
        and analysis.domains == ("document",)
        and bound_count == 1
    ):
        return _fast_or_runtime_dependency(
            "document", allowed_capabilities, "exact_document_metadata"
        )
    if (
        analysis.work_type == "retrieve"
        and "section" in analysis.domains
        and set(analysis.domains) <= {"document", "section"}
        and semantic.section_refs
        and bound_count == 1
    ):
        return _fast_or_runtime_dependency(
            "section", allowed_capabilities, "exact_section_retrieval"
        )
    if analysis.work_type == "summarize" and analysis.domains == ("document",) and bound_count == 1:
        return _fast_or_runtime_dependency(
            "document", allowed_capabilities, "exact_document_metadata"
        )
    if analysis.work_type == "lookup" and analysis.domains == ("knowledge_graph",):
        return _fast_or_runtime_dependency(
            "knowledge_graph", allowed_capabilities, "simple_kg_lookup"
        )

    if analysis.work_type == "cross_domain":
        return RouteDecision(route="complex_research", reason_code="cross_domain_dependency")
    if analysis.work_type == "compare":
        return RouteDecision(route="complex_research", reason_code="comparison")
    if analysis.work_type == "multi_goal":
        return RouteDecision(route="complex_research", reason_code="multi_goal")
    if analysis.work_type == "evaluate":
        return RouteDecision(route="complex_research", reason_code="compliance_evaluation")
    return RouteDecision(route="complex_research", reason_code="multi_document_research")


async def route_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Analyze the finalized query and decide the execution topology."""
    context = _context_of(runtime)
    analysis = analyze_query(state["semantic"], state["bindings"])
    decision = decide_route(
        analysis,
        state["semantic"],
        state["bindings"],
        allowed_capabilities=context.capability_runtime.allowed_capabilities,
    )
    return {"query_analysis": analysis, "route_decision": decision}

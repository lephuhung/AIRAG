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
    Write                     -> complex_research/simple_write_operation

Write (and any other work with no Phase-2 capability, e.g. compliance
evaluation) is a typed-unavailable OUTCOME, never a raise: unsupported work
routes to ``complex_research`` with an explicit reason code so T6's
``complex_boundary`` returns the typed unavailable response. Unsupported work
is never silently routed to ``fast_domain``.

Routing may inspect request-scoped capability availability (via the injected
``allowed_capabilities``); availability never reaches business state. Fast-path
gates count only pins referenced by the current semantic projection, never
stale merged pins from prior turns.
"""
from __future__ import annotations

import re
from typing import Any

from langgraph.runtime import Runtime

from ..adapters.document import binding_id_for_ref
from ..contracts.binding import DocumentBindingSet
from ..contracts.request import RequestContext
from ..contracts.routing import QueryAnalysis, RouteDecision, SemanticDependencyHint
from ..contracts.semantic import SemanticContext
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.validation import validate_query_analysis
from .context import _context_of

__all__ = [
    "analyze_query",
    "decide_route",
    "route_node",
]

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


def analyze_query(semantic: SemanticContext) -> QueryAnalysis:
    """Deterministic-first analysis of the finalized query meaning."""
    text = semantic.normalized_query.casefold()
    domains: set[str] = set()
    if semantic.person_refs:
        domains.add("people")
    if semantic.document_refs:
        domains.add("document")
    if semantic.section_refs:
        domains.add("section")
    if _write_intent(text):
        domains.add("write")
    if _contains(text, _KG_RES):
        domains.add("knowledge_graph")

    if not domains:
        if _is_conversational(text):
            # Conversational turns consult discourse memory only; they are not
            # lookups and must never read as a memory.lookup fast path.
            analysis = QueryAnalysis(
                work_type="explain",
                domains=("memory",),
                dependency_hints=(),
            )
            validate_query_analysis(analysis)
            return analysis
        # A ref-less factual query can only proceed via discovery.
        domains = {"document"}

    compare = _contains(text, _COMPARE_RES) or len(semantic.document_refs) >= 2
    summary = _contains(text, _SUMMARY_RES)
    evaluate = _contains(text, _EVALUATE_RES)

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
    elif domains <= {"people"} or domains == {"knowledge_graph"}:
        work_type = "lookup"
    else:
        work_type = "retrieve"

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


def _current_bound_count(
    semantic: SemanticContext, bindings: DocumentBindingSet
) -> int:
    """Pins referenced by the current semantic projection (stale pins excluded)."""
    current_ids = {binding_id_for_ref(reference.ref_id) for reference in semantic.document_refs}
    return sum(1 for binding in bindings.bindings if binding.binding_id in current_ids)


def _fast_or_runtime_dependency(
    domain: str, allowed_capabilities: frozenset[str], reason_code: Any
) -> RouteDecision:
    if _FAST_CAPABILITY[domain] in allowed_capabilities:
        return RouteDecision(route="fast_domain", reason_code=reason_code)
    return RouteDecision(route="complex_research", reason_code="runtime_dependency")


def _api_explicit_ref_ids(request: RequestContext | None) -> frozenset[str]:
    """Namespaced ref IDs for the current turn's ``api_explicit`` resources.

    Only ``KnownDocumentResource`` entries with ``source == "api_explicit"``
    are projected; raw query text and model output can never mint the
    namespace, so membership proves the target came from the API-supplied
    scope. Empty when the request is absent or carries no explicit scope.
    """
    if request is None:
        return frozenset()
    return frozenset(
        f"api_explicit:{known.resource_id}"
        for known in request.known_documents
        if known.source == "api_explicit"
    )


def _has_api_explicit_target(
    semantic: SemanticContext, request: RequestContext | None
) -> bool:
    """True when the current turn pins an API-explicit hard-scope target.

    Matches only *resolved* semantic references whose namespaced
    ``api_explicit:<resource_id>`` ID names a current-turn
    ``api_explicit`` resource on the request: raw query text and model
    output can never mint the namespace, so a match proves the target
    came from the API-supplied scope.
    """
    explicit = _api_explicit_ref_ids(request)
    if not explicit:
        return False
    return any(
        reference.ref_id in explicit
        and reference.resolution_status == "resolved"
        for reference in semantic.document_refs
    )


def decide_route(
    analysis: QueryAnalysis,
    semantic: SemanticContext,
    bindings: DocumentBindingSet,
    *,
    allowed_capabilities: frozenset[str] = frozenset(),
    request: RequestContext | None = None,
) -> RouteDecision:
    """Map deterministic analysis facts to one frozen route (never raises)."""
    if "write" in analysis.domains:
        # Typed-unavailable outcome: T6's complex_boundary owns the response.
        return RouteDecision(
            route="complex_research", reason_code="simple_write_operation"
        )
    if semantic.blocking_ambiguities:
        return RouteDecision(route="clarify", reason_code="essential_ambiguity")
    if _has_open_required_reference(semantic):
        return RouteDecision(route="clarify", reason_code="unresolved_required_binding")

    text = semantic.normalized_query.casefold()
    explicit_ids = _api_explicit_ref_ids(request)
    # Transport-only explicit targets must not defeat the conversational
    # guard: a greeting that merely carries `document_ids` stays direct.
    # Only intrinsic query references (person/section refs, or a document
    # ref outside the current-turn explicit namespace) count here.
    has_intrinsic_refs = bool(semantic.person_refs or semantic.section_refs) or any(
        reference.ref_id not in explicit_ids
        for reference in semantic.document_refs
    )
    if _is_conversational(text) and not has_intrinsic_refs:
        reason = "direct_greeting" if _is_greeting(text) else "direct_conversation"
        return RouteDecision(route="direct", reason_code=reason)  # type: ignore[arg-type]

    bound_count = _current_bound_count(semantic, bindings)
    if analysis.work_type == "retrieve" and _has_api_explicit_target(
        semantic, request
    ):
        # Hard-scoped factual retrieval (P0): an API-explicit document is a
        # retrieval target, never the exact-document metadata/read fast path
        # — including the one-document case.
        return RouteDecision(
            route="complex_research", reason_code="multi_document_research"
        )
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
    analysis = analyze_query(state["semantic"])
    decision = decide_route(
        analysis,
        state["semantic"],
        state["bindings"],
        allowed_capabilities=context.capability_runtime.allowed_capabilities,
        request=state["request"],
    )
    return {"query_analysis": analysis, "route_decision": decision}

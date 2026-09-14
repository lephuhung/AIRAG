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

import inspect
import logging
import re
from typing import Any

from langgraph.runtime import Runtime

from app.prompts.agents.supervisor_scope import classify_supervisor_scope

from ..adapters.document import binding_id_for_ref
from ..contracts.binding import DocumentBindingSet
from ..contracts.request import RequestContext
from ..contracts.routing import (
    Domain,
    QueryAnalysis,
    RouteDecision,
    SemanticDependencyHint,
    WorkType,
)
from ..contracts.semantic import SemanticContext
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.validation import validate_query_analysis
from ..semantic.intent import IntentDecision
from .context import _context_of

logger = logging.getLogger(__name__)

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


#: Phase 4A Task 3: v1 intent taxonomy -> v2 (work_type, domains).
#: Typed intent is advisory semantic input only; the deterministic
#: ``decide_route()`` policy keeps route authority. ``resolve_doc`` is
#: deliberately ABSENT: it is a semantic-prerequisite marker without
#: terminal semantics (the v1 task plan is dropped at the Task-2
#: boundary), so it falls back to the reference/text path instead of a
#: fixed mapping. Intents outside the v1 taxonomy likewise return
#: ``None`` so the caller falls back to the legacy deterministic path.
_INTENT_ANALYSIS: dict[str, tuple[str, ...]] = {
    "greeting": ("direct", "memory"),
    "personal": ("direct", "memory"),
    "mongo_search_cccd": ("lookup", "people"),
    "mongo_search_phone": ("lookup", "people"),
    "mongo_search_bhxh": ("lookup", "people"),
    "mongo_search_name": ("lookup", "people"),
    "mongo_search_advanced": ("lookup", "people"),
    "search": ("retrieve", "document"),
    "search_doc_num": ("retrieve", "document"),
    "search_section": ("retrieve", "document", "section"),
    "summarize": ("summarize", "document"),
    "kg_query": ("lookup", "knowledge_graph"),
    "list_docs": ("retrieve", "document"),
    "search_abbr": ("retrieve", "document"),
    "write_summarize": ("retrieve", "write"),
    "write_suggest_edits": ("retrieve", "write"),
    "write_grammar_check": ("retrieve", "write"),
    "write_format_check": ("retrieve", "write"),
}


def _ref_domains(semantic: SemanticContext) -> set[str]:
    """Typed reference-derived domains (no lexical signals)."""
    domains: set[str] = set()
    if semantic.person_refs:
        domains.add("people")
    if semantic.document_refs:
        domains.add("document")
    if semantic.section_refs:
        domains.add("section")
    return domains


def _analysis_for_intent(
    intent: IntentDecision | str,
) -> tuple[WorkType, tuple[Domain, ...]] | None:
    """Map a typed v1 intent onto v2 analysis facts (``None`` when unknown).

    ``resolve_doc`` returns ``None`` by design (see the table comment):
    without the dropped v1 task plan only the reference/text path knows
    the terminal semantics (summarize vs section vs search).
    """
    name = getattr(intent, "intent", intent)
    if not isinstance(name, str):
        return None
    if name == "resolve_doc":
        return None
    mapped = _INTENT_ANALYSIS.get(name)
    if mapped is None:
        return None
    return mapped[0], tuple(sorted(mapped[1:]))  # type: ignore[return-value]


def _dependency_hints_for(
    domains: set[str],
) -> tuple[SemanticDependencyHint, ...]:
    """Domain-family dependency hints shared by both analysis paths."""
    families = {_FAMILY.get(domain, domain) for domain in domains}
    dependency_families = sorted(set(_DEPENDENCY_FAMILIES) & families)
    return tuple(
        SemanticDependencyHint(
            hint_id=f"dep-{index + 1}",
            description=f"{first}->{second} dependency",
        )
        for index, (first, second) in enumerate(
            zip(dependency_families, dependency_families[1:])
        )
    )


def _typed_work_type(
    base_work_type: WorkType, domains: set[str], document_ref_count: int
) -> WorkType:
    """Work type for the typed path: intent base plus ref-count arity.

    Mirrors the legacy precedence (cross-domain > multi-goal > compare)
    without any text signal: generic lexical patterns stay demoted under
    typed intent, while typed reference counts keep their topology.
    """
    families = {_FAMILY.get(domain, domain) for domain in domains}
    dependency_families = set(_DEPENDENCY_FAMILIES) & families
    if len(dependency_families) >= 2:
        return "cross_domain"
    if document_ref_count >= 3:
        return "multi_goal"
    if document_ref_count >= 2:
        return "compare"
    return base_work_type


def analyze_query(
    semantic: SemanticContext, *, intent: IntentDecision | str | None = None
) -> QueryAnalysis:
    """Deterministic-first analysis of the finalized query meaning.

    When a typed v1 ``IntentDecision`` is supplied, its taxonomy mapping
    above is authoritative and the generic regex/keyword classifiers
    (“là ai”, “khác biệt”, “tổng hợp”, “đánh giá”) do not govern.
    Typed reference-derived semantics survive: the intent-mapped domains
    are unioned with the reference-derived domains and the
    reference-count arity rules (two targets → compare, three or more →
    multi_goal) still apply, mirroring the legacy precedence without any
    text signal. ``IntentDecision``
    confidence is never projected: frozen ``QueryAnalysis`` carries no
    confidence field. Without typed intent (or for ``resolve_doc`` /
    unknown intents) the legacy deterministic behavior is preserved
    exactly.
    """
    if intent is not None:
        mapped = _analysis_for_intent(intent)
        if mapped is not None:
            base_work_type, base_domains = mapped
            domains = set(base_domains) | _ref_domains(semantic)
            analysis = QueryAnalysis(
                work_type=_typed_work_type(
                    base_work_type, domains, len(semantic.document_refs)
                ),
                domains=tuple(sorted(domains)),  # type: ignore[arg-type]
                dependency_hints=_dependency_hints_for(domains),
            )
            validate_query_analysis(analysis)
            return analysis
        # resolve_doc / unknown intent name: fall through to the legacy
        # path rather than acting on an unmapped taxonomy value.
    text = semantic.normalized_query.casefold()
    domains = _ref_domains(semantic)
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
        # An exact unscoped people-identifier query (phone/CCCD/BHXH/name
        # cue recognized by the deterministic supervisor-scope classifier)
        # is a people lookup, never document discovery. This branch is
        # reachable only with no refs at all, so document/section refs keep
        # document semantics authoritative by construction.
        if classify_supervisor_scope(semantic.normalized_query) == "people":
            domains = {"people"}
        else:
            # A ref-less factual query can only proceed via discovery.
            domains = {"document"}

    compare = _contains(text, _COMPARE_RES) or len(semantic.document_refs) >= 2
    summary = _contains(text, _SUMMARY_RES)
    evaluate = _contains(text, _EVALUATE_RES)

    hints = _dependency_hints_for(domains)
    dependency_families = sorted(
        set(_DEPENDENCY_FAMILIES)
        & {_FAMILY.get(domain, domain) for domain in domains}
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


async def _intent_for_route(
    classifier: Any, request: RequestContext
) -> IntentDecision | None:
    """Fetch the turn-scoped typed v1 intent, if the service is wired.

    Uses the cached decision when the semantic seam already classified
    this turn, otherwise classifies once (the classifier single-flights
    per turn). A missing service or a classifier failure returns ``None``
    so routing falls back to the legacy deterministic path; typed intent
    is advisory-only and must never break routing.
    """
    if classifier is None:
        return None
    query = request.original_query
    has_doc_ids = bool(request.known_documents)
    try:
        cached_fn = getattr(classifier, "cached", None)
        if cached_fn is not None:
            cached = cached_fn(query, has_doc_ids=has_doc_ids)
            if cached is not None:
                return cached
        classify = getattr(classifier, "classify", None)
        if classify is None:
            return None
        result = classify(query, has_doc_ids=has_doc_ids)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as exc:  # noqa: BLE001 - advisory input must not break routing
        logger.warning("[route] intent classification failed, using legacy path: %s", exc)
        return None


async def route_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Analyze the finalized query and decide the execution topology.

    Typed v1 intent (when the turn-scoped classifier is wired) is the
    semantic input to ``analyze_query``; the deterministic
    ``decide_route()`` policy keeps route authority.
    """
    context = _context_of(runtime)
    intent = await _intent_for_route(
        context.services.intent_classifier, state["request"]
    )
    analysis = analyze_query(state["semantic"], intent=intent)
    decision = decide_route(
        analysis,
        state["semantic"],
        state["bindings"],
        allowed_capabilities=context.capability_runtime.allowed_capabilities,
        request=state["request"],
    )
    return {"query_analysis": analysis, "route_decision": decision}

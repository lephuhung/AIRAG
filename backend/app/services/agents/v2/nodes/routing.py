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

Supersession note (Task 12, binding spec §15): ``evaluate``/compliance is no
longer an example of unsupported work — it is served by the governed v2
complex DAG (``skills/evaluate`` bounded evidence reads + evidence
evaluation + grounded synthesis, ``requires_v1_fallback`` lifted for the
``evaluate``/``compliance_evaluation`` outcome). The paragraph above now
applies to genuinely uncovered work (e.g. write, or work no skill covers).

Routing may inspect request-scoped capability availability (via the injected
``allowed_capabilities`` permission grant plus the actual request-scoped
capability catalog when the caller supplies it); availability never reaches
business state. A capability that is allowed but absent from the actual
catalog gets a deterministic ``runtime_dependency`` fallback — routing never
enters an execution path that cannot exist. Reference-free factual retrieval
(``retrieve``/``document`` with no pinned target) is a bounded targetless
``document.retrieve`` fast path where the catalog serves it. Fast-path
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
from ..contracts.intent import IntentAnalysis
from ..contracts.semantic import SemanticContext
from ..contracts.state import GraphRuntimeContext, SupervisorV2State
from ..contracts.validation import validate_intent_analysis, validate_query_analysis
from ..semantic.intent import IntentDecision
from ..semantic.intent_registry import INTENT_REGISTRY
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

#: Capability serving reference-free factual retrieval. Unlike the pinned
#: ``document.read`` fast path, ``document.retrieve`` supports empty
#: ``target_ids`` (workspace-scope retrieval); its presence in the
#: request-scoped catalog is the workspace-search policy signal. The literal
#: must match ``skills/retrieve/policy.py::RETRIEVE_CAPABILITY`` (kept as a
#: literal so nodes stay decoupled from skill policy modules).
_TARGETLESS_RETRIEVE_CAPABILITY = "document.retrieve"


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
#: fixed mapping. ``personal`` is likewise deliberately ABSENT (final
#: review I4): its typed ``direct``/``memory`` mapping had no
#: ``work_type == "direct"`` branch in ``decide_route``, so typed
#: ``personal`` fell through to ``complex_research`` while the legacy
#: path served it fast (``simple_kg_lookup`` / targetless retrieval).
#: Until a frozen-reason ruling records how typed ``direct`` should be
#: served, ``personal`` falls back to the legacy deterministic path so
#: simple fast-path behavior is preserved. ``evaluate`` is a v2-only
#: addition (final review I1): the v1 taxonomy has no evaluate intent, so
#: this mapping is reachable only via the deterministic
#: ``classify_evaluate`` narrow scope, never the shared v1 model prompt.
#: Intents outside the v1 taxonomy likewise return ``None`` so the caller
#: falls back to the legacy deterministic path.
_INTENT_ANALYSIS: dict[str, tuple[str, ...]] = {
    "greeting": ("direct", "memory"),
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
    "evaluate": ("evaluate", "document"),
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


def _analyze_from_intent_analysis(
    semantic: SemanticContext, intent_analysis: IntentAnalysis
) -> QueryAnalysis:
    """Flag-on analysis: registry-projected facts from ``IntentAnalysis``.

    The model decides WHAT the user wants (spec §31); this projection maps
    each detected intent name through ``INTENT_REGISTRY`` onto the same
    ``work_type``/``domains`` facts the deterministic machinery consumes —
    replacing ``_INTENT_ANALYSIS`` on the flag-on path only. Multi-intent
    (or empty/unknown) analyses carry ``work_type="multi_intent"`` so the
    ``decide_route`` intent gate owns the outcome; single known intents
    keep the registry work type plus the usual ref-arity promotion.
    Identifier extraction stays entity extraction: only typed reference
    domains (``_ref_domains``) union in, never identifier-regex verdicts.
    """
    intents = intent_analysis.intents
    specs = [INTENT_REGISTRY.get(intent.name) for intent in intents]
    domains = _ref_domains(semantic)
    for spec in specs:
        if spec is not None:
            domains.update(spec.get("domains", ()))
    if not domains:
        # ``domains`` is a non-empty contract; an all-unknown/empty analysis
        # still needs a domain tuple. The ref-less factual default
        # (``document``) matches the legacy branch — routing outcome is
        # already complex via the intent gate, so this is shape-only.
        domains = {"document"}
    text = semantic.normalized_query.casefold()
    # Same defence-in-depth as the typed path: the v1-owned write
    # boundary stays enforced even under flag-on semantics.
    if _write_intent(text):
        domains.add("write")
    if len(intents) == 1 and specs[0] is not None:
        work_type: WorkType = _typed_work_type(
            specs[0].get("work_type", "retrieve"),  # type: ignore[arg-type]
            domains,
            len(semantic.document_refs),
        )
    else:
        work_type = "multi_intent"
    analysis = QueryAnalysis(
        work_type=work_type,
        domains=tuple(sorted(domains)),  # type: ignore[arg-type]
        dependency_hints=_dependency_hints_for(domains),
    )
    validate_query_analysis(analysis)
    return analysis


def analyze_query(
    semantic: SemanticContext,
    *,
    intent: IntentDecision | str | None = None,
    intent_analysis: IntentAnalysis | None = None,
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

    ``intent_analysis`` (flag-on ``IntentAnalysis``) is mutually
    exclusive with ``intent``: it projects through ``INTENT_REGISTRY``
    instead of the v1 taxonomy map, and ``work_type="multi_intent"``
    marks every non-single-known-intent analysis for the ``decide_route``
    intent gate. Flag-off callers pass neither beyond ``intent``.
    """
    if intent is not None and intent_analysis is not None:
        raise ValueError("intent and intent_analysis are mutually exclusive")
    if intent_analysis is not None:
        return _analyze_from_intent_analysis(semantic, intent_analysis)
    if intent is not None:
        mapped = _analysis_for_intent(intent)
        if mapped is not None:
            base_work_type, base_domains = mapped
            domains = set(base_domains) | _ref_domains(semantic)
            # Final review Mod5: the v1-owned write boundary stays
            # defence-in-depth even when a write request is misclassified
            # to a read taxonomy intent (e.g. ``search``). Typed intent must
            # not silently drop the write domain that the legacy path would
            # have surfaced to ``simple_write_operation``.
            text = semantic.normalized_query.casefold()
            if _write_intent(text):
                domains.add("write")
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
    capability: str,
    allowed_capabilities: frozenset[str],
    reason_code: Any,
    available_capabilities: frozenset[str] | None = None,
) -> RouteDecision:
    """Fast when the capability is permitted AND deployable, else fallback.

    ``allowed_capabilities`` is the request permission grant;
    ``available_capabilities`` is the actual request-scoped catalog (the
    registry view) when the caller can supply it. A capability that is
    allowed but absent from the catalog gets the deterministic
    ``runtime_dependency`` outcome — never a route into a path that cannot
    exist. ``None`` keeps the allowed-only gate: the pre-existing pinned
    fast branches use it deliberately so the shared scheduler + registry
    stays the unavailable authority (typed denied / DEPENDENCY_UNAVAILABLE
    outcomes); only the new targetless retrieval branch passes the
    catalog through.
    """
    if capability not in allowed_capabilities:
        return RouteDecision(route="complex_research", reason_code="runtime_dependency")
    if (
        available_capabilities is not None
        and capability not in available_capabilities
    ):
        return RouteDecision(route="complex_research", reason_code="runtime_dependency")
    return RouteDecision(route="fast_domain", reason_code=reason_code)


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
    available_capabilities: frozenset[str] | None = None,
    request: RequestContext | None = None,
    intent_analysis: IntentAnalysis | None = None,
) -> RouteDecision:
    """Map deterministic analysis facts to one frozen route (never raises).

    ``available_capabilities`` is the actual request-scoped capability
    catalog when the caller can supply it (``route_node`` passes the
    registry view); the targetless retrieval fast gate requires the
    capability in both the permission grant and the catalog. ``None``
    keeps the allowed-only gate for pure-function callers. Pre-existing
    pinned fast branches intentionally keep the allowed-only gate so the
    shared scheduler + registry remains their unavailable authority.
    """
    if "write" in analysis.domains:
        # Typed-unavailable outcome: T6's complex_boundary owns the response.
        return RouteDecision(
            route="complex_research", reason_code="simple_write_operation"
        )
    if semantic.blocking_ambiguities:
        return RouteDecision(route="clarify", reason_code="essential_ambiguity")
    if _has_open_required_reference(semantic):
        return RouteDecision(route="clarify", reason_code="unresolved_required_binding")

    # Multi-intent gate (spec §33.2/§33.3, flag-on only): after clarify
    # still wins, before every fast branch. ``len > 1`` always outranks
    # ``primary_intent`` (§18); empty or registry-unknown single intents
    # are semantic uncertainty — uncertain → complex, never a fast path.
    # Single known intents fall through: complex intents keep their
    # specific reasons via the existing work-type branches, atomic
    # intents keep the existing fast gates (bound count, section
    # coordinate, conversational guard still apply).
    if intent_analysis is not None:
        intents = intent_analysis.intents
        if len(intents) > 1:
            return RouteDecision(
                route="complex_research", reason_code="multi_intent"
            )
        if len(intents) == 0 or intents[0].name not in INTENT_REGISTRY:
            return RouteDecision(
                route="complex_research", reason_code="semantic_uncertainty"
            )

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

    bound_count = _current_bound_count(semantic, bindings)
    # Phase 4A Task 4: reference-free factual retrieval is a bounded
    # targetless fast path. This branch sits before the conversational
    # guard so a greeting-prefixed factual query (``chào anh, hỏi về chế
    # độ thai sản?``) stays factual — and it is safe there because it
    # requires zero pinned targets, so a greeting that merely carries
    # transport-only explicit targets (bound_count >= 1) still falls
    # through to the guard and stays direct. All other factual fast paths
    # keep their original position below the guard.
    if (
        analysis.work_type == "retrieve"
        and analysis.domains == ("document",)
        and bound_count == 0
    ):
        # Bounded targetless document retrieval: no explicit binding is
        # required for general factual RAG. The ``document.retrieve``
        # presence in the request-scoped catalog is the workspace-search
        # policy signal; absence fails closed to ``runtime_dependency``.
        return _fast_or_runtime_dependency(
            _TARGETLESS_RETRIEVE_CAPABILITY,
            allowed_capabilities,
            "targetless_document_retrieval",
            available_capabilities,
        )
    if _is_conversational(text) and not has_intrinsic_refs:
        reason = "direct_greeting" if _is_greeting(text) else "direct_conversation"
        return RouteDecision(route="direct", reason_code=reason)  # type: ignore[arg-type]

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
        # Allowed-only gate by design: the shared scheduler + registry is
        # the unavailable authority here, so a denied/gated capability
        # still yields its typed execution outcome (denied /
        # DEPENDENCY_UNAVAILABLE) instead of a vague router fallback.
        return _fast_or_runtime_dependency(
            _FAST_CAPABILITY["people"],
            allowed_capabilities,
            "simple_people_lookup",
        )
    if (
        analysis.work_type == "retrieve"
        and analysis.domains == ("document",)
        and bound_count == 1
    ):
        return _fast_or_runtime_dependency(
            _FAST_CAPABILITY["document"],
            allowed_capabilities,
            "exact_document_metadata",
        )
    if (
        analysis.work_type == "retrieve"
        and "section" in analysis.domains
        and set(analysis.domains) <= {"document", "section"}
        and semantic.section_refs
        and bound_count == 1
        # Task 7 bridge: section.read needs a revision-specific
        # structure_node_id. A label-only locator ("Điều 5" with no
        # coordinate) must never claim the capability — it would fail
        # closed at dispatch. The bounded document fallback below owns it.
        and any(reference.structure_node_id for reference in semantic.section_refs)
    ):
        return _fast_or_runtime_dependency(
            _FAST_CAPABILITY["section"],
            allowed_capabilities,
            "exact_section_retrieval",
        )
    if (
        analysis.work_type == "retrieve"
        and "section" in analysis.domains
        and set(analysis.domains) <= {"document", "section"}
        and semantic.section_refs
        and bound_count == 1
    ):
        # Bounded document fallback for a label-only section locator:
        # one pinned document, no revision coordinate — a single-pin
        # document.read fast path, never the Planner.
        return _fast_or_runtime_dependency(
            _FAST_CAPABILITY["document"],
            allowed_capabilities,
            "exact_document_metadata",
        )
    # A label-only section locator adds the ``section`` domain without pinning
    # a coordinate (Task 7 bridge): a bounded one-document summarize naming a
    # section is still a single-pin document.read fast path, never the Planner.
    if analysis.work_type == "summarize" and set(analysis.domains) <= {"document", "section"} and bound_count == 1:
        return _fast_or_runtime_dependency(
            _FAST_CAPABILITY["document"],
            allowed_capabilities,
            "exact_document_metadata",
        )
    if analysis.work_type == "lookup" and analysis.domains == ("knowledge_graph",):
        return _fast_or_runtime_dependency(
            _FAST_CAPABILITY["knowledge_graph"],
            allowed_capabilities,
            "simple_kg_lookup",
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


def _multi_intent_flag_on() -> bool:
    """``V2_MULTI_INTENT_ROUTING_ENABLED`` live read (monkeypatchable)."""
    try:
        from app.core.config import get_settings

        return bool(
            getattr(get_settings(), "V2_MULTI_INTENT_ROUTING_ENABLED", False)
        )
    except Exception:  # noqa: BLE001 - config unreadable → flag off
        return False


async def _intent_analysis_for_route(
    state: SupervisorV2State, services: Any
) -> IntentAnalysis | None:
    """Fetch the flag-on ``IntentAnalysis`` — checkpoint first, never re-classify.

    Resume determinism: a checkpointed analysis in ``state["intent_analysis"]``
    replays identically without touching the classifier. Otherwise the
    request-scoped ``services.multi_intent_classifier`` classifies the
    finalized ``semantic.contextualized_query`` (post coreference
    resolution — spec §21). A missing service, a classifier failure, or a
    ``None`` result returns ``None`` so the caller produces the forced
    ``semantic_uncertainty`` complex outcome — never the legacy
    ``_intent_for_route`` path (§33.2).
    """
    existing = state.get("intent_analysis")
    if existing is not None:
        return existing
    classifier = getattr(services, "multi_intent_classifier", None)
    if classifier is None:
        return None
    request = state["request"]
    semantic = state["semantic"]
    query = semantic.contextualized_query or request.original_query
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
        if result is not None:
            validate_intent_analysis(result)
        return result
    except Exception as exc:  # noqa: BLE001 - failure → semantic_uncertainty
        logger.warning(
            "[route] multi-intent classification failed: %s", type(exc).__name__
        )
        return None


async def _available_catalog(
    services: Any,
) -> frozenset[str] | None:
    """Actual request-scoped capability catalog, or ``None`` when unknown.

    Reads the registry view (``capability_names()``) without dispatching
    anything: capabilities receive ``AgentRequest`` +
    ``CapabilityRuntimeContext`` only from the shared scheduler, never from
    nodes. A missing/unreadable registry returns ``None`` so routing keeps
    the legacy allowed-only gate; the catalog gate is availability-only and
    must never break routing.
    """
    registry = getattr(services, "capability_registry", None)
    names_fn = getattr(registry, "capability_names", None)
    if not callable(names_fn):
        return None
    try:
        return frozenset(names_fn())
    except Exception as exc:  # noqa: BLE001 - availability gate must not break routing
        logger.warning("[route] capability catalog unreadable, using allowed-only gate: %s", exc)
        return None


async def route_node(
    state: SupervisorV2State,
    runtime: "Runtime[GraphRuntimeContext]",
) -> dict:
    """Analyze the finalized query and decide the execution topology.

    Typed v1 intent (when the turn-scoped classifier is wired) is the
    semantic input to ``analyze_query``; the deterministic
    ``decide_route()`` policy keeps route authority. Fast gates check the
    actual registry catalog in addition to the permission grant, so a
    missing runtime capability falls back deterministically instead of
    routing into a path that cannot exist.
    """
    context = _context_of(runtime)
    services = context.services
    # Flag-on path (spec §33): LLM-first ``IntentAnalysis`` is the semantic
    # input; ``INTENT_REGISTRY`` + ``decide_route`` keep execution
    # authority. Also entered when a checkpointed analysis exists so a
    # resumed turn replays the identical decision. Flag-off is the
    # byte-identical deterministic path below — ``decide_route`` never
    # sees ``intent_analysis``.
    if _multi_intent_flag_on() or state.get("intent_analysis") is not None:
        intent_analysis = await _intent_analysis_for_route(state, services)
        if intent_analysis is None:
            # Every flag-on failure mode lands on complex, never a fast path.
            analysis = QueryAnalysis(
                work_type="multi_intent",
                domains=("document",),
                dependency_hints=(),
            )
            validate_query_analysis(analysis)
            decision = RouteDecision(
                route="complex_research", reason_code="semantic_uncertainty"
            )
        else:
            analysis = analyze_query(
                state["semantic"], intent_analysis=intent_analysis
            )
            decision = decide_route(
                analysis,
                state["semantic"],
                state["bindings"],
                allowed_capabilities=context.capability_runtime.allowed_capabilities,
                available_capabilities=await _available_catalog(services),
                request=state["request"],
                intent_analysis=intent_analysis,
            )
        return {
            "query_analysis": analysis,
            "route_decision": decision,
            "intent_analysis": intent_analysis,
        }
    intent = await _intent_for_route(
        services.intent_classifier, state["request"]
    )
    analysis = analyze_query(state["semantic"], intent=intent)
    decision = decide_route(
        analysis,
        state["semantic"],
        state["bindings"],
        allowed_capabilities=context.capability_runtime.allowed_capabilities,
        available_capabilities=await _available_catalog(services),
        request=state["request"],
    )
    return {"query_analysis": analysis, "route_decision": decision}

"""Intent execution registry (multi-intent routing spec §7, §8.1).

Pure data: the registry is the authoritative mapping from a detected intent
name to its execution properties. It is deliberately free of imports from
``contracts.routing`` so it can be consumed by both the router
(``nodes/routing.py``) and the planner/skills without a dependency cycle —
the only import is the ``IntentAnalysis`` contract type for the
``needs_complex_execution`` signature.

Registry semantics:

- ``execution``: ``"atomic"`` intents may keep a deterministic fast path
  when they are the single detected intent; ``"complex"`` intents always
  route to the complex-research boundary.
- ``fast_path``: the candidate ``RouteReason`` for the single-atomic case.
  It is a *candidate* only — the existing downstream gates (bound count,
  section coordinate, conversational guard) still apply after the intent
  gate, so ``document_lookup`` with no bound target never claims
  ``document.read``.
- ``work_type``/``domains``: the ``QueryAnalysis`` projection a
  single-atomic intent feeds into the existing ``analyze_query`` →
  ``decide_route`` machinery on the flag-on path.
- ``capability``: the governed capability that supplies evidence for this
  intent inside a multi-intent plan (``None`` for intents the pipeline
  nodes — evaluate/synthesize — own instead of a plan task).

``document_search`` maps to ``document.retrieve``, not ``document.search``
(spec §33.4): ``document.search`` emits discovery candidates requiring
``settle`` + ``V2_ALLOW_*_DISCOVERY`` (default-off); ``document.retrieve``
emits evidence directly.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from ..contracts.intent import IntentAnalysis

__all__ = [
    "INTENT_REGISTRY",
    "IntentSpec",
    "needs_complex_execution",
]


class IntentSpec(TypedDict, total=False):
    """Execution properties of one registered intent name (pure data)."""

    execution: Literal["atomic", "complex"]
    fast_path: str | None
    work_type: str
    domains: tuple[str, ...]
    capability: str | None


INTENT_REGISTRY: dict[str, IntentSpec] = {
    "people_lookup": {
        "execution": "atomic",
        "fast_path": "simple_people_lookup",
        "work_type": "lookup",
        "domains": ("people",),
        "capability": "people.lookup",
    },
    "people_search": {
        "execution": "atomic",
        "fast_path": "simple_people_lookup",
        "work_type": "lookup",
        "domains": ("people",),
        "capability": "people.lookup",
    },
    "document_lookup": {
        "execution": "atomic",
        "fast_path": "targetless_document_retrieval",
        "work_type": "retrieve",
        "domains": ("document",),
        "capability": "document.retrieve",
    },
    "document_search": {
        "execution": "atomic",
        "fast_path": "targetless_document_retrieval",
        "work_type": "retrieve",
        "domains": ("document",),
        "capability": "document.retrieve",
    },
    "section_lookup": {
        "execution": "atomic",
        "fast_path": "exact_section_retrieval",
        "work_type": "retrieve",
        "domains": ("document", "section"),
        "capability": "section.read",
    },
    "kg_lookup": {
        "execution": "atomic",
        "fast_path": "simple_kg_lookup",
        "work_type": "lookup",
        "domains": ("knowledge_graph",),
        "capability": "knowledge_graph.query",
    },
    "memory_lookup": {
        "execution": "atomic",
        "fast_path": None,
        "work_type": "explain",
        "domains": ("memory",),
        "capability": "memory.lookup",
    },
    "direct_answer": {
        "execution": "atomic",
        "fast_path": "direct_conversation",
        "work_type": "direct",
        "domains": ("memory",),
        "capability": None,
    },
    "greeting": {
        "execution": "atomic",
        "fast_path": "direct_greeting",
        "work_type": "direct",
        "domains": ("memory",),
        "capability": None,
    },
    "personal": {
        "execution": "atomic",
        "fast_path": "direct_conversation",
        "work_type": "explain",
        "domains": ("memory",),
        "capability": "memory.lookup",
    },
    "list_documents": {
        "execution": "atomic",
        "fast_path": "targetless_document_retrieval",
        "work_type": "retrieve",
        "domains": ("document",),
        "capability": "document.retrieve",
    },
    "summarize": {
        "execution": "complex",
        "work_type": "summarize",
        "domains": ("document",),
        "capability": None,
    },
    "compare_documents": {
        "execution": "complex",
        "work_type": "compare",
        "domains": ("document",),
        "capability": None,
    },
    "evaluate_compliance": {
        "execution": "complex",
        "work_type": "evaluate",
        "domains": ("document",),
        "capability": None,
    },
    "cross_domain_research": {
        "execution": "complex",
        "work_type": "cross_domain",
        "domains": (),
        "capability": None,
    },
    "write": {
        "execution": "complex",
        "work_type": "retrieve",
        "domains": ("write",),
        "capability": None,
    },
}


def needs_complex_execution(analysis: IntentAnalysis) -> bool:
    """Spec §8.1 truth table: whether the analysis must run complex research.

    Empty intents → ``True``; two or more intents → ``True``; an intent name
    missing from the registry → ``True``; a single registered intent whose
    ``execution`` is not ``"atomic"`` → ``True``. Only a single registered
    atomic intent returns ``False`` (it may keep a deterministic fast path,
    subject to the downstream gates).
    """
    intents = analysis.intents
    if len(intents) != 1:
        return True
    spec = INTENT_REGISTRY.get(intents[0].name)
    if spec is None:
        return True
    return spec.get("execution") != "atomic"

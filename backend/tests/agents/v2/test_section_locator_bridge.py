"""Task 7 (Phase 4B): bridge the minimum section signal for the simple fast path.

Canonical case: "Dieu 5 Luat An ninh mang quy dinh gi?" must preserve the
authoritative one-turn section locator (Dieu/Chuong/Khoan) as a v2
``SectionReference`` and keep candidate document IDs across the
preprocessing persistence/translation boundary. A label-only locator
(no revision-specific ``structure_node_id``) must NEVER route to
``section.read`` (it would fail closed); it falls back to the bounded
single-pin ``document.read`` fast path. Full multi-turn coreference/focus
is explicitly out of scope (Phase 4C).
"""
from __future__ import annotations

import uuid
from typing import Any, Sequence
from uuid import UUID

from app.services.agents.semantic_preprocessor import (
    DocumentCandidate,
    DocumentRefEntry,
    PreprocessingResult,
    TraceEvent,
    to_persisted_dict,
)
from app.services.agents.v2.adapters.semantic import (
    draft_from_persisted_semantic,
    draft_from_preprocessing,
)
from app.services.agents.v2.contracts.binding import DocumentBindingSet
from app.services.agents.v2.contracts.routing import QueryAnalysis
from app.services.agents.v2.contracts.semantic import (
    DocumentReference,
    SectionReference,
    SemanticContext,
)
from app.services.agents.v2.nodes.routing import analyze_query, decide_route

DOC_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_DOC_ID = UUID("22222222-2222-2222-2222-222222222222")

_FULL_CAPABILITIES = frozenset(
    {
        "people.lookup",
        "document.search",
        "document.retrieve",
        "document.read",
        "section.read",
        "knowledge_graph.query",
        "memory.lookup",
        "abbreviation.resolve",
    }
)


def _legacy_doc_ref(
    *,
    ref_id: str = "r1",
    section_reference: str | None = "Điều 5",
    status: str = "resolved",
    document_handle: UUID | None = DOC_ID,
    candidates: list | None = None,
) -> DocumentRefEntry:
    query = "Điều 5 Luật An ninh mạng quy định gì?"
    span = "Luật An ninh mạng"
    start = query.index(span)
    return DocumentRefEntry(
        ref_id=ref_id,
        original_span=span,
        span_offset=(start, start + len(span)),
        reference="Luật An ninh mạng",
        section_reference=section_reference,
        document_handle=document_handle,
        candidates=candidates or [],
        resolution_status=status,  # type: ignore[arg-type]
    )


def _legacy_result(*, refs: list | None = None) -> PreprocessingResult:
    query = "Điều 5 Luật An ninh mạng quy định gì?"
    return PreprocessingResult(
        original_query=query,
        normalized_query=query,
        abbreviations=[],
        document_refs=refs if refs is not None else [_legacy_doc_ref()],
        blocking_ambiguities=[],
        preprocessing_status="ok",
        preprocessor_trace=[TraceEvent(step="input", started_at=0.0, ended_at=0.1)],
    )


def test_preprocessing_preserves_simple_section_locator() -> None:
    draft = draft_from_preprocessing(_legacy_result())
    assert len(draft.section_refs) == 1
    section = draft.section_refs[0]
    assert section.label == "Điều 5"
    assert section.structure_node_id is None


def test_preprocessing_without_section_has_no_section_refs() -> None:
    draft = draft_from_preprocessing(
        _legacy_result(refs=[_legacy_doc_ref(section_reference=None)])
    )
    assert draft.section_refs == ()


def test_persisted_round_trip_preserves_section_and_candidates() -> None:
    ref = _legacy_doc_ref(
        status="ambiguous",
        document_handle=None,
        candidates=[
            DocumentCandidate(
                document_id=DOC_ID, match_basis="exact_number", confidence=0.9
            ),
            DocumentCandidate(
                document_id=OTHER_DOC_ID, match_basis="exact_number", confidence=0.8
            ),
        ],
    )
    payload = to_persisted_dict(_legacy_result(refs=[ref]))
    draft = draft_from_persisted_semantic(payload)
    assert len(draft.section_refs) == 1
    assert draft.section_refs[0].label == "Điều 5"
    assert len(draft.document_refs) == 1
    assert set(draft.document_refs[0].candidate_document_ids) == {
        DOC_ID,
        OTHER_DOC_ID,
    }


def _resolved_doc_ref(ref_id: str = "r1") -> DocumentReference:
    return DocumentReference(
        ref_id=ref_id,
        original_span="Luật An ninh mạng",
        normalized_reference="Luật An ninh mạng",
        requested_role=None,
        revision_requirement=None,
        resolution_status="resolved",
        resolved_document_id=DOC_ID,
        candidate_document_ids=(),
    )


def _semantic_with_label_only_section() -> SemanticContext:
    return SemanticContext(
        contextualized_query="Điều 5 Luật An ninh mạng quy định gì?",
        normalized_query="điều 5 luật an ninh mạng quy định gì?",
        abbreviations=(),
        coreferences=(),
        document_refs=(_resolved_doc_ref(),),
        person_refs=(),
        section_refs=(SectionReference(ref_id="s1", label="Điều 5"),),
        blocking_ambiguities=(),
    )


def _binding_set() -> DocumentBindingSet:
    from app.services.agents.v2.adapters.document import binding_id_for_ref
    from app.services.agents.v2.contracts.binding import ScopedDocument

    return DocumentBindingSet(
        bindings=(
            ScopedDocument(
                binding_id=binding_id_for_ref("r1"),
                document_id=DOC_ID,
                document_revision="rev-1",
                role="target",
            ),
        ),
        revision_requirement_refs=(),
    )


def test_label_only_section_never_routes_to_section_read() -> None:
    semantic = _semantic_with_label_only_section()
    analysis = analyze_query(semantic, intent="search_section")
    assert analysis.work_type == "retrieve"
    assert set(analysis.domains) == {"document", "section"}
    decision = decide_route(
        analysis, semantic, _binding_set(), allowed_capabilities=_FULL_CAPABILITIES
    )
    assert decision.route == "fast_domain"
    # Bounded document fallback: no Planner, no section.read without a coordinate.
    assert decision.reason_code == "exact_document_metadata"


def test_located_section_still_routes_to_section_read() -> None:
    semantic = _semantic_with_label_only_section().model_copy(
        update={
            "section_refs": (
                SectionReference(
                    ref_id="s1", label="Điều 5", structure_node_id="node-5"
                ),
            )
        }
    )
    analysis = analyze_query(semantic, intent="search_section")
    decision = decide_route(
        analysis, semantic, _binding_set(), allowed_capabilities=_FULL_CAPABILITIES
    )
    assert decision.route == "fast_domain"
    assert decision.reason_code == "exact_section_retrieval"


class _StubIdentityResolver:
    """Minimal stand-in: resolves identity and reports a section label."""

    def __init__(self, section_label: str | None = "Điều 5") -> None:
        self._section_label = section_label
        self.calls = 0

    async def resolve_reference(
        self,
        reference: DocumentReference,
        *,
        question: str,
        workspace_ids: Sequence[Any],
        db: Any,
        use_llm_fallback: bool = True,
    ) -> DocumentReference:
        self.calls += 1
        return reference.model_copy(
            update={
                "resolution_status": "resolved",
                "resolved_document_id": DOC_ID,
                "candidate_document_ids": (),
            }
        )

    def cached_section_label(
        self,
        reference: DocumentReference,
        *,
        question: str,
        workspace_ids: Sequence[Any],
        use_llm_fallback: bool = True,
    ) -> str | None:
        return self._section_label


def test_summarize_with_label_only_section_stays_fast() -> None:
    """I1 pin: a bounded one-document summarize naming a section locator

    must keep the document.read fast path (never the Planner). The plan
    itself must build: a route label pointing at an unmappable capability
    would raise FastPlanError at checkpoint time.
    """
    from app.services.agents.v2.nodes.fast_plan import build_fast_plan

    semantic = _semantic_with_label_only_section()
    analysis = analyze_query(semantic, intent="summarize")
    assert analysis.work_type == "summarize"
    assert set(analysis.domains) == {"document", "section"}
    bindings = _binding_set()
    decision = decide_route(
        analysis, semantic, bindings, allowed_capabilities=_FULL_CAPABILITIES
    )
    assert decision.route == "fast_domain"
    assert decision.reason_code == "exact_document_metadata"
    plan = build_fast_plan(semantic, bindings, analysis, decision)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].capability == "document.read"


def test_resolve_draft_identities_preserves_resolver_section() -> None:
    import asyncio

    from app.services.agents.v2.adapters.semantic import resolve_draft_identities
    from app.services.agents.v2.contracts.semantic import SemanticDraft

    draft = SemanticDraft(
        provisional_contextualized_query="Điều 5 Luật An ninh mạng quy định gì?",
        abbreviations=(),
        coreferences=(),
        document_refs=(
            DocumentReference(
                ref_id="r1",
                original_span="Luật An ninh mạng",
                normalized_reference="Luật An ninh mạng",
                requested_role=None,
                revision_requirement=None,
                resolution_status="unresolved",
                resolved_document_id=None,
                candidate_document_ids=(),
            ),
        ),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=(),
    )
    enriched = asyncio.run(
        resolve_draft_identities(
            draft,
            question="Điều 5 Luật An ninh mạng quy định gì?",
            identity_resolver=_StubIdentityResolver(),  # type: ignore[arg-type]
            workspace_ids=(uuid.uuid4(),),
            db=object(),
        )
    )
    assert enriched.document_refs[0].resolution_status == "resolved"
    assert len(enriched.section_refs) == 1
    assert enriched.section_refs[0].label == "Điều 5"
    assert enriched.section_refs[0].structure_node_id is None

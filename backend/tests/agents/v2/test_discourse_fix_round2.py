"""Task 8 fix round 2 — NEW-CRITICAL-1 / NEW-IMPORTANT-1.

Intended behavior (stated explicitly): a mention with ZERO visible
candidates is "no local referent", not a choice — it resolves nothing
and blocks nothing. Only true ambiguity (>=2 visible candidates)
becomes a BlockingAmbiguity. Ambiguity IDs are mention-derived so the
two producer seams (build_draft + resolve_draft_identities) are
idempotent when chained.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID

DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")

SCOPE = (DOC_A, DOC_B)


def _doc_ref(ref_id: str, doc_id: UUID):
    from app.services.agents.v2.contracts.semantic import DocumentReference

    return DocumentReference(
        ref_id=ref_id,
        original_span="doc",
        normalized_reference="doc",
        requested_role=None,
        revision_requirement=None,
        resolution_status="resolved",
        resolved_document_id=doc_id,
        candidate_document_ids=(),
    )


def test_zero_ref_anaphora_is_silent_not_blocking():
    from app.services.agents.v2.semantic.discourse import resolve_coreferences

    for query in (
        "văn bản này quy định gì?",
        "điều này áp dụng khi nào?",
        "tóm tắt file thứ hai",
    ):
        corefs, ambiguities = resolve_coreferences(
            query,
            document_refs=(),
            person_refs=(),
            section_refs=(),
            allowed_document_ids=SCOPE,
            can_read_people=False,
        )
        assert corefs == (), query
        assert ambiguities == (), query


def test_zero_ref_anaphora_end_to_end_through_build_draft():
    from app.services.agents.semantic_preprocessor import PreprocessingResult
    from app.services.agents.supervisor_v2 import DeterministicSemanticAdapter

    query = "văn bản này quy định gì?"

    async def _preprocess(raw: str):
        return PreprocessingResult(
            original_query=raw,
            normalized_query=raw.strip().lower(),
            abbreviations=[],
            document_refs=[],
            blocking_ambiguities=[],
            preprocessing_status="ok",
            preprocessor_trace=[],
        )

    async def _run():
        adapter = DeterministicSemanticAdapter(preprocess=_preprocess)
        from app.services.agents.v2.contracts.conversation import (
            ConversationContext,
        )
        from app.services.agents.v2.contracts.request import RequestContext

        return await adapter.build_draft(
            RequestContext(
                contract_version="2.0",
                request_id="r1",
                thread_id="t1",
                original_query=query,
                known_documents=(),
            ),
            ConversationContext(
                summary="Tóm tắt phiên trước.",
                active_entities=(),
                last_focus=None,
                recent_turns=(),
            ),
        )

    draft = asyncio.run(_run())
    assert draft.coreferences == ()
    assert draft.preliminary_ambiguities == ()


def test_person_mention_without_permission_is_silent():
    from app.services.agents.v2.contracts.conversation import EntityReference
    from app.services.agents.v2.semantic.discourse import resolve_coreferences

    corefs, ambiguities = resolve_coreferences(
        "ông ấy làm việc ở đâu?",
        document_refs=(),
        person_refs=(
            EntityReference(ref_id="p1", kind="person", label="0901234567"),
        ),
        section_refs=(),
        allowed_document_ids=SCOPE,
        can_read_people=False,
    )
    assert corefs == ()
    assert ambiguities == ()


def test_ambiguity_ids_are_stable_across_calls():
    from app.services.agents.v2.semantic.discourse import resolve_coreferences

    kwargs = dict(
        document_refs=(_doc_ref("r1", DOC_A), _doc_ref("r2", DOC_B)),
        person_refs=(),
        section_refs=(),
        allowed_document_ids=SCOPE,
        can_read_people=False,
    )
    _, first = resolve_coreferences("văn bản này quy định gì?", **kwargs)
    _, second = resolve_coreferences("văn bản này quy định gì?", **kwargs)
    assert len(first) == 1
    assert [a.ambiguity_id for a in first] == [a.ambiguity_id for a in second]


def test_chained_seams_stay_merge_clean():
    """build_draft output + resolve_draft_identities on the same turn.

    Simulates the documented Task 7 chaining: the same mention must not
    produce duplicate ambiguity IDs (fail-closed ContractValidationError)
    nor duplicate coreference links.
    """
    from app.services.agents.v2.adapters.semantic import resolve_draft_identities
    from app.services.agents.v2.contracts.validation import (
        validate_semantic_context,
    )
    from app.services.agents.v2.contracts.semantic import SemanticContext
    from app.services.agents.v2.semantic.discourse import resolve_coreferences

    refs = (_doc_ref("r1", DOC_A), _doc_ref("r2", DOC_B))
    first_corefs, first_amb = resolve_coreferences(
        "văn bản này quy định gì?",
        document_refs=refs,
        person_refs=(),
        section_refs=(),
        allowed_document_ids=SCOPE,
        can_read_people=False,
    )
    assert len(first_amb) == 1

    from app.services.agents.v2.contracts.semantic import SemanticDraft

    draft = SemanticDraft(
        provisional_contextualized_query="văn bản này quy định gì?",
        abbreviations=(),
        coreferences=first_corefs,
        document_refs=refs,
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=first_amb,
    )

    class _PassThroughResolver:
        async def resolve_reference(
            self, reference, *, question, workspace_ids, db,
            use_llm_fallback=True,
        ):
            return reference

        def cached_section_label(self, *args, **kwargs):
            return None

    out = asyncio.run(
        resolve_draft_identities(
            draft,
            question="văn bản này quy định gì?",
            identity_resolver=_PassThroughResolver(),
            workspace_ids=(UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),),
            db=SimpleNamespace(),
        )
    )
    ids = [a.ambiguity_id for a in out.preliminary_ambiguities]
    assert len(ids) == len(set(ids)) == 1
    links = [(c.mention, c.resolved_ref_id) for c in out.coreferences]
    assert len(links) == len(set(links))
    # The merged draft must survive frozen validation (uniqueness rule).
    validate_semantic_context(
        SemanticContext(
            contextualized_query=out.provisional_contextualized_query,
            normalized_query=out.provisional_contextualized_query,
            abbreviations=(),
            coreferences=out.coreferences,
            document_refs=out.document_refs,
            person_refs=(),
            section_refs=(),
            blocking_ambiguities=out.preliminary_ambiguities,
        )
    )

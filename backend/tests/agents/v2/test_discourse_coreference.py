"""Task 8 (Phase 4C): typed discourse/coreference with ACL confinement.

RED-test: ``app.services.agents.v2.semantic.discourse`` does not exist yet.
Every case asserts real behavior required by the brief:

- stable typed identity populates ``person_refs``;
- ``active_entities`` carry typed kinds (not flattened ``concept``);
- ``last_focus`` derives from validated outcomes;
- unambiguous ``van ban nay`` / ``dieu nay`` / ``ong ay`` /
  ``file thu hai`` resolve; true ambiguity becomes clarification;
- history NEVER reauthorizes out-of-scope resources.
"""
from __future__ import annotations

from uuid import UUID

DOC_A = UUID("11111111-1111-1111-1111-111111111111")
DOC_B = UUID("22222222-2222-2222-2222-222222222222")
OUT_OF_SCOPE = UUID("99999999-9999-9999-9999-999999999999")

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


def test_person_refs_populated_for_stable_phone_identity():
    from app.services.agents.v2.semantic.discourse import extract_person_refs

    refs = extract_person_refs("0901234567 là ai?")
    assert len(refs) == 1
    assert refs[0].kind == "person"
    assert "0901234567" in refs[0].label


def test_person_refs_empty_without_stable_identity():
    from app.services.agents.v2.semantic.discourse import extract_person_refs

    assert extract_person_refs("chế độ thai sản được quy định thế nào?") == ()


def test_active_entities_are_typed_not_flattened():
    from app.services.agents.v2.semantic.discourse import typed_active_entities

    entities = typed_active_entities(
        ["Nghị định 15/2020/NĐ-CP", "Nguyễn Văn A", "Điều 5", "chế độ thai sản"]
    )
    by_label = {entity.label: entity.kind for entity in entities}
    assert by_label["Nghị định 15/2020/NĐ-CP"] == "document"
    assert by_label["Nguyễn Văn A"] == "person"
    assert by_label["Điều 5"] == "section"
    assert by_label["chế độ thai sản"] == "concept"


def test_last_focus_derives_from_validated_outcomes():
    from app.services.agents.v2.semantic.discourse import (
        derive_last_focus,
        typed_active_entities,
    )

    entities = typed_active_entities(["Nghị định 15/2020/NĐ-CP", "Điều 5"])
    focus = derive_last_focus(entities)
    assert focus is not None
    assert focus.label == "Điều 5"
    assert focus.kind == "section"


def test_unambiguous_van_ban_nay_resolves_in_scope():
    from app.services.agents.v2.semantic.discourse import resolve_coreferences

    corefs, ambiguities = resolve_coreferences(
        "văn bản này quy định gì?",
        document_refs=(_doc_ref("r1", DOC_A),),
        person_refs=(),
        section_refs=(),
        active_entities=(),
        allowed_document_ids=SCOPE,
        can_read_people=False,
    )
    assert ambiguities == ()
    assert len(corefs) == 1
    assert corefs[0].resolved_ref_id == "r1"


def test_history_never_reauthorizes_out_of_scope():
    from app.services.agents.v2.semantic.discourse import resolve_coreferences

    corefs, ambiguities = resolve_coreferences(
        "văn bản này quy định gì?",
        document_refs=(_doc_ref("r1", OUT_OF_SCOPE),),
        person_refs=(),
        section_refs=(),
        active_entities=(),
        allowed_document_ids=SCOPE,
        can_read_people=False,
    )
    # The out-of-scope pin is invisible: no resolution. Zero visible
    # candidates is "no local referent" — silent, never a binding and
    # never a candidate-free blocking question (fix round 2).
    assert corefs == ()
    assert ambiguities == ()


def test_true_ambiguity_becomes_clarification():
    from app.services.agents.v2.semantic.discourse import resolve_coreferences

    corefs, ambiguities = resolve_coreferences(
        "văn bản này quy định gì?",
        document_refs=(_doc_ref("r1", DOC_A), _doc_ref("r2", DOC_B)),
        person_refs=(),
        section_refs=(),
        active_entities=(),
        allowed_document_ids=SCOPE,
        can_read_people=False,
    )
    assert corefs == ()
    assert len(ambiguities) == 1


def test_ong_ay_resolves_person_only_with_permission():
    from app.services.agents.v2.contracts.conversation import EntityReference
    from app.services.agents.v2.semantic.discourse import resolve_coreferences

    person = EntityReference(ref_id="p1", kind="person", label="0901234567")
    corefs, _ = resolve_coreferences(
        "ông ấy làm việc ở đâu?",
        document_refs=(),
        person_refs=(person,),
        section_refs=(),
        active_entities=(),
        allowed_document_ids=SCOPE,
        can_read_people=True,
    )
    assert len(corefs) == 1
    assert corefs[0].resolved_ref_id == "p1"

    corefs_denied, ambiguities = resolve_coreferences(
        "ông ấy làm việc ở đâu?",
        document_refs=(),
        person_refs=(person,),
        section_refs=(),
        active_entities=(),
        allowed_document_ids=SCOPE,
        can_read_people=False,
    )
    # Denied permission leaves zero visible candidates: silent, never a
    # blocking question the user could not answer by selection (round 2).
    assert corefs_denied == ()
    assert ambiguities == ()


def test_file_thu_hai_resolves_ordinal():
    from app.services.agents.v2.semantic.discourse import resolve_coreferences

    corefs, ambiguities = resolve_coreferences(
        "tóm tắt file thứ hai",
        document_refs=(_doc_ref("r1", DOC_A), _doc_ref("r2", DOC_B)),
        person_refs=(),
        section_refs=(),
        active_entities=(),
        allowed_document_ids=SCOPE,
        can_read_people=False,
    )
    assert ambiguities == ()
    assert len(corefs) == 1
    assert corefs[0].resolved_ref_id == "r2"


def test_dieu_nay_resolves_section():
    from app.services.agents.v2.contracts.semantic import SectionReference
    from app.services.agents.v2.semantic.discourse import resolve_coreferences

    section = SectionReference(ref_id="s1", label="Điều 5", structure_node_id=None)
    corefs, ambiguities = resolve_coreferences(
        "điều này áp dụng khi nào?",
        document_refs=(),
        person_refs=(),
        section_refs=(section,),
        active_entities=(),
        allowed_document_ids=SCOPE,
        can_read_people=False,
    )
    assert ambiguities == ()
    assert len(corefs) == 1
    assert corefs[0].resolved_ref_id == "s1"


def test_conversation_adapter_types_entities_and_focus():
    from app.services.agents.v2.adapters.conversation import context_from_legacy

    class _Msg:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content

    class _Summary:
        exchange_index = 1
        user_message_id = "u1"
        assistant_message_id = "a1"
        summary = "Hỏi về Nghị định 15."
        key_entities = ["Nghị định 15/2020/NĐ-CP", "Điều 5"]

    context = context_from_legacy(
        messages=[_Msg("user", "Điều 5 nói gì?")],
        exchange_summaries=[_Summary()],
    )
    kinds = {entity.label: entity.kind for entity in context.active_entities}
    assert kinds["Nghị định 15/2020/NĐ-CP"] == "document"
    assert kinds["Điều 5"] == "section"
    assert context.last_focus is not None
    assert context.last_focus.label == "Điều 5"


def test_semantic_draft_carries_person_refs():
    from app.services.agents.semantic_preprocessor import PreprocessingResult
    from app.services.agents.v2.adapters.semantic import draft_from_preprocessing

    query = "0901234567 là ai?"
    result = PreprocessingResult(
        original_query=query,
        normalized_query=query,
        abbreviations=[],
        document_refs=[],
        blocking_ambiguities=[],
        preprocessing_status="ok",
        preprocessor_trace=[],
    )
    draft = draft_from_preprocessing(result)
    assert len(draft.person_refs) == 1
    assert draft.person_refs[0].kind == "person"

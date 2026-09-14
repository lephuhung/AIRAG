"""Task 6 — adapt v1 document resolver to the v2 contract (RED first).

Covers the Phase 4B identity seam: the wrapper calls the proven v1
``resolve_candidates()`` (never a cloned regex/SQL/LLM/vector pipeline),
passes the full contextualized question as ``topic``, translates a clear
winner to ``resolved``, close candidates to ``ambiguous`` + candidate IDs,
keeps not-found as not-found, never force-binds low-confidence candidates,
stays inside the authorized workspace scope, leaves revision pinning to the
existing v2 binding resolver, and caches expensive resolver work per
request. Stub DB / monkeypatched resolver only — no live infra, no
capability dispatch.
"""
from __future__ import annotations

import uuid

import pytest

from app.services.agents.v2.adapters.semantic import (
    SemanticAdapterError,
    resolve_draft_identities,
)
from app.services.agents.v2.contracts.binding import DocumentBindingSet
from app.services.agents.v2.contracts.semantic import (
    CurrentRevisionRequirement,
    DocumentReference,
    SemanticContext,
    SemanticDraft,
)
from app.services.agents.v2.contracts.validation import (
    validate_binding_set,
    validate_semantic_context,
)
from app.services.agents.v2.semantic.document_identity import (
    DocumentIdentityResolver,
    reference_from_candidates,
)

DOC_A = str(uuid.uuid4())
DOC_B = str(uuid.uuid4())
DOC_C = str(uuid.uuid4())


def _cand(doc_id: str, score: float, **extra: object) -> dict:
    base: dict = {
        "document_id": doc_id,
        "title": f"doc {doc_id[:8]}",
        "document_number": "",
        "published_date": "",
        "score": score,
        "strategy": "db_query",
    }
    base.update(extra)
    return base


def _ref(**overrides: object) -> DocumentReference:
    base: dict = {
        "ref_id": "r1",
        "original_span": "Nghị định 53",
        "normalized_reference": "Nghị định 53",
        "requested_role": None,
        "revision_requirement": None,
        "resolution_status": "not_found",
        "resolved_document_id": None,
        "candidate_document_ids": (),
    }
    base.update(overrides)
    return DocumentReference(**base)  # type: ignore[arg-type]


def test_clear_winner_resolves() -> None:
    ref = reference_from_candidates(
        _ref(), [_cand(DOC_A, 0.95), _cand(DOC_B, 0.40)]
    )
    assert ref.resolution_status == "resolved"
    assert str(ref.resolved_document_id) == DOC_A
    assert ref.candidate_document_ids == ()


def test_close_second_is_ambiguous_with_candidate_ids() -> None:
    # Below EARLY_EXIT (0.85): the close-second rule owns this band.
    ref = reference_from_candidates(
        _ref(), [_cand(DOC_A, 0.80), _cand(DOC_B, 0.72)]
    )
    assert ref.resolution_status == "ambiguous"
    assert ref.resolved_document_id is None
    assert {str(v) for v in ref.candidate_document_ids} == {DOC_A, DOC_B}


def test_empty_candidates_stays_not_found() -> None:
    ref = reference_from_candidates(_ref(), [])
    assert ref.resolution_status == "not_found"
    assert ref.resolved_document_id is None
    assert ref.candidate_document_ids == ()


def test_low_confidence_is_never_force_bound() -> None:
    ref = reference_from_candidates(_ref(), [_cand(DOC_A, 0.10)])
    assert ref.resolution_status == "not_found"
    assert ref.resolved_document_id is None


def test_medium_single_candidate_stays_unbound_and_validator_legal() -> None:
    """C1: one MEDIUM hit cannot be ``ambiguous`` (frozen validator needs
    >=2 distinct candidates), so it reads ``unresolved`` — no id, no
    candidates — instead of a force-bound or contract-invalid fact."""
    ref = reference_from_candidates(_ref(), [_cand(DOC_A, 0.45)])
    assert ref.resolution_status == "unresolved"
    assert ref.resolved_document_id is None
    assert ref.candidate_document_ids == ()


def test_early_exit_winner_resolves_despite_close_second() -> None:
    """I2: v1 precedence — top >= EARLY_EXIT (0.85) binds before the
    close-second rule is considered (resolve_doc_agent.py:100-105)."""
    ref = reference_from_candidates(
        _ref(), [_cand(DOC_A, 0.95), _cand(DOC_B, 0.90)]
    )
    assert ref.resolution_status == "resolved"
    assert str(ref.resolved_document_id) == DOC_A
    assert ref.candidate_document_ids == ()


def test_close_second_below_early_exit_stays_ambiguous() -> None:
    ref = reference_from_candidates(
        _ref(), [_cand(DOC_A, 0.80), _cand(DOC_B, 0.75)]
    )
    assert ref.resolution_status == "ambiguous"
    assert ref.resolved_document_id is None
    assert {str(v) for v in ref.candidate_document_ids} == {DOC_A, DOC_B}


def test_ambiguous_with_one_usable_uuid_is_not_contract_ambiguous() -> None:
    """C1: a close second whose id is unusable leaves a single usable
    UUID — that must read ``unresolved``, never 1-candidate ambiguous."""
    ref = reference_from_candidates(
        _ref(), [_cand(DOC_A, 0.80), _cand("not-a-uuid", 0.75)]
    )
    assert ref.resolution_status == "unresolved"
    assert ref.resolved_document_id is None
    assert ref.candidate_document_ids == ()


def test_non_uuid_candidate_ids_are_dropped_never_fabricated() -> None:
    ref = reference_from_candidates(
        _ref(), [_cand("not-a-uuid", 0.95), {"document_id": "", "score": 0.9}]
    )
    assert ref.resolution_status == "not_found"
    assert ref.resolved_document_id is None


def test_identity_never_pins_a_revision() -> None:
    ref = reference_from_candidates(_ref(), [_cand(DOC_A, 0.95)])
    assert ref.revision_requirement is None
    assert ref.requested_role is None


@pytest.mark.asyncio
async def test_resolver_passes_full_question_as_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    async def _fake_resolve(
        reference: str,
        workspace_ids: list,
        db: object,
        *,
        topic: str | None = None,
        **kwargs: object,
    ) -> dict:
        seen["reference"] = reference
        seen["topic"] = topic
        seen["workspace_ids"] = workspace_ids
        return {
            "candidates": [_cand(DOC_A, 0.95)],
            "parsed": {},
            "section_reference": None,
            "similar": [],
            "counts": {},
        }

    monkeypatch.setattr(
        "app.services.agents.v2.semantic.document_identity.resolve_candidates",
        _fake_resolve,
    )
    resolver = DocumentIdentityResolver()
    ref = await resolver.resolve_reference(
        _ref(),
        question="Điều 5 Luật An ninh mạng quy định gì?",
        workspace_ids=[uuid.uuid4()],
        db=object(),
    )
    assert seen["topic"] == "Điều 5 Luật An ninh mạng quy định gì?"
    assert ref.resolution_status == "resolved"


@pytest.mark.asyncio
async def test_resolver_stays_inside_authorized_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    async def _fake_resolve(
        reference: str,
        workspace_ids: list,
        db: object,
        **kwargs: object,
    ) -> dict:
        seen["workspace_ids"] = list(workspace_ids)
        return {
            "candidates": [],
            "parsed": {},
            "section_reference": None,
            "similar": [],
            "counts": {},
        }

    monkeypatch.setattr(
        "app.services.agents.v2.semantic.document_identity.resolve_candidates",
        _fake_resolve,
    )
    ws = uuid.uuid4()
    resolver = DocumentIdentityResolver()
    ref = await resolver.resolve_reference(
        _ref(), question="Luật An ninh mạng", workspace_ids=[ws], db=object()
    )
    assert seen["workspace_ids"] == [ws]
    assert ref.resolution_status == "not_found"


@pytest.mark.asyncio
async def test_resolver_caches_expensive_work_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def _fake_resolve(
        reference: str,
        workspace_ids: list,
        db: object,
        **kwargs: object,
    ) -> dict:
        calls.append(reference)
        return {
            "candidates": [_cand(DOC_C, 0.95)],
            "parsed": {},
            "section_reference": None,
            "similar": [],
            "counts": {},
        }

    monkeypatch.setattr(
        "app.services.agents.v2.semantic.document_identity.resolve_candidates",
        _fake_resolve,
    )
    resolver = DocumentIdentityResolver()
    ws = [uuid.uuid4()]
    db = object()
    first = await resolver.resolve_reference(
        _ref(), question="Nghị định 53/2022/NĐ-CP", workspace_ids=ws, db=db
    )
    second = await resolver.resolve_reference(
        _ref(), question="Nghị định 53/2022/NĐ-CP", workspace_ids=ws, db=db
    )
    # The bare span drives the reference arg; the full question drives topic.
    assert calls == ["Nghị định 53"]
    assert first == second
    assert resolver.cache_size == 1


@pytest.mark.asyncio
async def test_cache_preserves_each_callers_ref_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I1: the cache stores identity facts, not the first caller's whole
    reference — a same-span second ref keeps its ref_id, role, and
    revision requirement."""

    async def _fake_resolve(
        reference: str,
        workspace_ids: list,
        db: object,
        **kwargs: object,
    ) -> dict:
        return {
            "candidates": [_cand(DOC_A, 0.95)],
            "parsed": {},
            "section_reference": None,
            "similar": [],
            "counts": {},
        }

    monkeypatch.setattr(
        "app.services.agents.v2.semantic.document_identity.resolve_candidates",
        _fake_resolve,
    )
    resolver = DocumentIdentityResolver()
    ws = [uuid.uuid4()]
    db = object()
    first = await resolver.resolve_reference(
        _ref(ref_id="r1", requested_role="target"),
        question="Nghị định 53 quy định gì?",
        workspace_ids=ws,
        db=db,
    )
    second = await resolver.resolve_reference(
        _ref(
            ref_id="r2",
            requested_role="reference",
            revision_requirement=CurrentRevisionRequirement(kind="current"),
        ),
        question="Nghị định 53 quy định gì?",
        workspace_ids=ws,
        db=db,
    )
    assert first.ref_id == "r1"
    assert second.ref_id == "r2"
    assert second.requested_role == "reference"
    assert isinstance(
        second.revision_requirement, CurrentRevisionRequirement
    )
    # Same identity facts, own ref shell.
    assert second.resolution_status == "resolved"
    assert second.resolved_document_id == first.resolved_document_id


def _context(*refs: DocumentReference) -> SemanticContext:
    return SemanticContext(
        contextualized_query="Nghị định 53 quy định gì?",
        normalized_query="Nghị định 53 quy định gì?",
        abbreviations=(),
        coreferences=(),
        document_refs=tuple(refs),
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=(),
    )


def test_produced_refs_pass_frozen_validators_and_empty_binding_set() -> None:
    """Item 4: every translator outcome must survive the real
    ``validate_semantic_context``/``validate_binding_set`` path that
    ``finalize_semantic`` runs — including the single-MEDIUM case."""
    refs = (
        reference_from_candidates(
            _ref(ref_id="r1"), [_cand(DOC_A, 0.95), _cand(DOC_B, 0.40)]
        ),
        reference_from_candidates(
            _ref(ref_id="r2"), [_cand(DOC_A, 0.80), _cand(DOC_B, 0.75)]
        ),
        reference_from_candidates(_ref(ref_id="r3"), [_cand(DOC_C, 0.45)]),
        reference_from_candidates(_ref(ref_id="r4"), []),
    )
    assert [r.resolution_status for r in refs] == [
        "resolved",
        "ambiguous",
        "unresolved",
        "not_found",
    ]
    semantic = _context(*refs)
    validate_semantic_context(semantic)
    validate_binding_set(
        DocumentBindingSet(bindings=(), revision_requirement_refs=()),
        semantic,
    )


def _draft(*refs: DocumentReference) -> SemanticDraft:
    return SemanticDraft(
        provisional_contextualized_query="Nghị định 53 quy định gì?",
        abbreviations=(),
        coreferences=(),
        document_refs=tuple(refs),
        person_refs=(),
        section_refs=(),
        preliminary_ambiguities=(),
    )


@pytest.mark.asyncio
async def test_draft_enrichment_resolves_only_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_resolve(
        reference: str,
        workspace_ids: list,
        db: object,
        **kwargs: object,
    ) -> dict:
        return {
            "candidates": [_cand(DOC_A, 0.95)],
            "parsed": {},
            "section_reference": None,
            "similar": [],
            "counts": {},
        }

    monkeypatch.setattr(
        "app.services.agents.v2.semantic.document_identity.resolve_candidates",
        _fake_resolve,
    )
    already = _ref(
        ref_id="r0",
        resolution_status="resolved",
        resolved_document_id=uuid.UUID(DOC_B),
    )
    draft = await resolve_draft_identities(
        _draft(already, _ref()),
        question="Nghị định 53 quy định gì?",
        identity_resolver=DocumentIdentityResolver(),
        workspace_ids=[uuid.uuid4()],
        db=object(),
    )
    # The pre-resolved ref passes through untouched; the open ref resolves.
    assert draft.document_refs[0] == already
    assert draft.document_refs[1].resolution_status == "resolved"
    assert str(draft.document_refs[1].resolved_document_id) == DOC_A
    # Enrichment never mints revision requirements: binding stays with v2.
    assert draft.document_refs[1].revision_requirement is None


@pytest.mark.asyncio
async def test_draft_enrichment_without_resolver_fails_closed() -> None:
    with pytest.raises(SemanticAdapterError):
        await resolve_draft_identities(
            _draft(_ref()),
            question="Nghị định 53 quy định gì?",
            identity_resolver=None,
            workspace_ids=[uuid.uuid4()],
            db=object(),
        )

"""Defect #3: a missing verdict must become a typed clarify/insufficient reply."""
from __future__ import annotations

from uuid import UUID

import pytest

from app.services.agents.v2.contracts.semantic import (
    BlockingAmbiguity,
    DocumentReference,
    SemanticContext,
)
from app.services.agents.v2.contracts.state import ExecutionState
from app.services.agents.v2.nodes.finalizer import _finalize_factual

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _semantic(
    *,
    document_refs: tuple[DocumentReference, ...] = (),
    blocking_ambiguities: tuple[BlockingAmbiguity, ...] = (),
) -> SemanticContext:
    return SemanticContext(
        contextualized_query="câu hỏi",
        normalized_query="câu hỏi",
        abbreviations=(),
        coreferences=(),
        document_refs=document_refs,
        person_refs=(),
        section_refs=(),
        blocking_ambiguities=blocking_ambiguities,
    )


def _ref(status: str) -> DocumentReference:
    return DocumentReference(
        ref_id="r1",
        original_span="Luật ABC",
        normalized_reference="luật abc",
        requested_role="target",
        revision_requirement=None,
        resolution_status=status,  # type: ignore[arg-type]
        resolved_document_id=None,
        candidate_document_ids=(),
    )


def _state(semantic: SemanticContext) -> dict:
    return {
        "execution": ExecutionState(
            plan=None, task_results=(), evidence_evaluation=None
        ),
        "semantic": semantic,
    }


@pytest.mark.asyncio
async def test_unresolved_reference_clarifies() -> None:
    update = await _finalize_factual(_state(_semantic(document_refs=(_ref("unresolved"),))), None)
    assert update["final_response"].status == "clarify"
    assert "Luật ABC" in update["final_response"].content


@pytest.mark.asyncio
async def test_blocking_ambiguity_clarifies() -> None:
    ambiguity = BlockingAmbiguity(
        ambiguity_id="amb1",
        description="Bạn muốn so sánh hai văn bản nào?",
    )
    update = await _finalize_factual(_state(_semantic(blocking_ambiguities=(ambiguity,))), None)
    assert update["final_response"].status == "clarify"


@pytest.mark.asyncio
async def test_no_semantic_signal_is_typed_insufficient() -> None:
    update = await _finalize_factual(_state(_semantic()), None)
    assert update["final_response"].status == "insufficient"
